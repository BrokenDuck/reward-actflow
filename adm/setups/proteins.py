from typing import Any, Sequence
from argparse import ArgumentParser
from importlib_resources import files
from pathlib import Path
from hydra.utils import instantiate
from torch import Tensor, device
import torch.nn.functional as F
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
import esm
import numpy as np
from tqdm import tqdm
from vendi_score import vendi
import gpytorch

from matplotlib.figure import Figure
from matplotlib import pyplot as plt

from sgpo.models.continuous import ContinuousModel
from sgpo.models.pretraining.model.continuous_diffusion import GaussianDiffusionTransformer
from flowgym.types import FlowTensor
from flowgym.base_models import BaseModel
from flowgym.schedulers import Scheduler, NoiseSchedule
from flowgym.environments import Environment
from flowgym.utils import append_dims

from active_pretraining.problem_setup import ProblemSetup, SampleFile, Batch
import sgpo
from sgpo.oracle.train_oracle import OracleModel

CREILOV_WILD_TYPE = "MAGLRHTFVVADATLPDCPLVYASEGFYAMTGYGPDEVLGHNARFLQGEGTDPKEVQKIRDAIKKGEACSVRLLNYRKDGTPFWNLLTVTPIKTPDGRVSKFVGVQVDVTSKTEGKALA"
ORACLE_HIDDEN_DIM = 400
ORACLE_DROPOUT = 0.1
ORACLE_BATCH_SIZE = 128
HAMMING_PENALTY_CUTOFF = 70
HAMMING_PENALTY_RATE = 0.99


def shim():
    import importlib
    try:
        import deepspeed
    except Exception:
        # deepspeed not installed yet; nothing to do
        deepspeed = None

    if deepspeed is not None:
        # if utils.is_initialized already exists, do nothing
        try:
            if not (hasattr(deepspeed, "utils") and hasattr(deepspeed.utils, "is_initialized")):
                # prefer comm.is_initialized if available
                comm = None
                try:
                    # new DeepSpeed exposes the helper under deepspeed.comm (or deepspeed.comm.comm)
                    from deepspeed import comm as _ds_comm  # most common
                    comm = _ds_comm
                except Exception:
                    # fallback: try the nested comm module path sometimes used
                    try:
                        from deepspeed.comm import comm as _ds_comm2
                        comm = _ds_comm2
                    except Exception:
                        comm = None

                import types
                if not hasattr(deepspeed, "utils"):
                    deepspeed.utils = types.SimpleNamespace()

                if comm is not None and hasattr(comm, "is_initialized"):
                    deepspeed.utils.is_initialized = comm.is_initialized
                else:
                    deepspeed.utils.is_initialized = lambda: False
        except Exception:
            pass


shim()


class DDPMNoiseSchedule(NoiseSchedule[FlowTensor]):
    def __init__(self, scheduler: Scheduler[FlowTensor]) -> None:
        super().__init__()
        self.scheduler = scheduler
    

    def __call__(self, x: FlowTensor, t: Tensor) -> FlowTensor:
        return self.scheduler.beta(x, t) / self.scheduler.alpha(x, t)


class CosineSchedule(Scheduler[FlowTensor]):
    def __init__(self) -> None:
        super().__init__()
        # self._noise_schedule = DDPMNoiseSchedule(self)


    def alpha(self, x: FlowTensor, t: Tensor) -> FlowTensor:
        sqrt_alpha_bar = torch.cos((1. - t) * torch.pi / 2.)
        return FlowTensor(append_dims(sqrt_alpha_bar, x.data.ndim))


    def alpha_dot(self, x: FlowTensor, t: Tensor) -> FlowTensor:
        sqrt_alpha_bar_dot = torch.pi / 2. * torch.sin((1. - t) * torch.pi / 2.)
        return FlowTensor(append_dims(sqrt_alpha_bar_dot.data, x.data.ndim))
    

    def beta(self, x: FlowTensor, t: Tensor) -> FlowTensor:
        alpha_bar = torch.cos((1. - t) * torch.pi / 2.).square()
        beta = torch.sqrt(1. - alpha_bar)
        return FlowTensor(append_dims(beta, x.data.ndim))


    def beta_dot(self, x: FlowTensor, t: Tensor) -> FlowTensor:
        alpha_bar_dot = 2. * torch.cos((1. - t) * torch.pi / 2.) * torch.pi / 2. * torch.sin((1. - t) * torch.pi / 2.)
        return -FlowTensor(append_dims(alpha_bar_dot, x.data.ndim)) / (2. * self.beta(x, t))


class ProteinModel(BaseModel[FlowTensor]):

    output_type = "endpoint"

    def __init__(self, config: DictConfig | ListConfig, device: device | None):
        super().__init__(device)
        model = config.model.model
        model_name = config.pretrained_ckpt
        seq_len = config.data.seq_len

        self.sgpo_model: ContinuousModel = instantiate(model, model_name=model_name, seq_len=seq_len, device=device, _recursive_=True)
        self.network = self.sgpo_model.model.network
        self._scheduler = CosineSchedule()
    
    
    @property
    def scheduler(self) -> Scheduler[FlowTensor]:
        return self._scheduler


    def preprocess(self, x: FlowTensor, **kwargs: Any) -> tuple[FlowTensor, dict[str, Any]]:
        # TODO maybe map to and from ESM embeddings???
        return x, kwargs
    

    def postprocess(self, x: FlowTensor) -> FlowTensor:
        # TODO maybe map to and from ESM embeddings???
        return x
    

    def sample_p0(self, n: int, **kwargs: Any) -> tuple[FlowTensor, dict[str, Any]]:
        return FlowTensor(self.sgpo_model.get_start(n)), kwargs
    

    def forward(self, x: FlowTensor, t: Tensor, **kwargs: Any) -> FlowTensor: 
        if 'debug' in kwargs and kwargs['debug']:
            return FlowTensor(torch.randn_like(x.data).to(x.device))
        return FlowTensor(self.network_forward(x.data, t)['xstart'])


    def embed_to_sequence(self, embeds: FlowTensor) -> Sequence[str]:
        epsilon = 1e-4 # TODO change magic number
        t = (1. - epsilon) * torch.ones((embeds.data.shape[0],)).to(self.device)
        out = self.network_forward(embeds.data, t)

        tokens = out['probs'].argmax(dim=-1)
        strings = [self.sgpo_model.tokenizer.untokenize(s) for s in tokens]
        return strings


    def network_forward(self, x: torch.Tensor, t: torch.Tensor) -> dict[str, torch.Tensor]:
        t = 1. - t.to(self.device)
        x = x.to(self.device)
        net = self.sgpo_model
        infill_mask = (torch.ones(net.seq_len) != net.tokenizer.pad_id-100).to(net.device) # TODO change magic number
        attn_mask = torch.ones((x.shape[0], x.shape[1]), dtype=torch.bool, device=net.device)
                               
        idx = (t * len(net.noise_schedule.sigmas)).round().clamp(0, len(net.noise_schedule.sigmas) - 1).long()
        sigma = net.noise_schedule.sigmas.to(net.device)[idx].reshape((x.shape[0], 1, 1)).float()
        f_out = net.model.network(x.float()/(sigma**2 + 1).sqrt(), idx, attn_mask=attn_mask)

        out = net.model.network.pred_xstart(
            x.float(),
            idx,
            attn_mask=attn_mask,
            sequence_output=f_out['sequence_output'],
            infill_mask=infill_mask
        )

        out['sequence_output'] = f_out['sequence_output']
        return out


class ProteinProblemSetup(ProblemSetup[FlowTensor]):
    def __init__(self, args: dict[str, Any], device: device | None):
        super().__init__(args)
        cfg_path: str = args['cfg_path']
        self.threshold = args['threshold']
        self.lengthscale = args['lengthscale_vendi'] if 'lengthscale_vendi' in args else None
        config = OmegaConf.load(cfg_path)

        self._base_model = ProteinModel(config, device=device)
        self.esmfold = esm.pretrained.esmfold_v1().eval().to(device)

        oracle_path = Path(args['oracle_path']) if 'oracle_path' in args else files(sgpo) / Path('oracle/checkpoints/CreiLOV')
        self.oracle_ensemble = self._load_oracle_ensemble(oracle_path)


    @classmethod
    def add_args(cls, parser: ArgumentParser): # TODO interface with hydra somehow to get hierarchical configs... for now use static one from existing run
        default_path = files(sgpo) / Path('configs/sample_config.yaml')
        parser.add_argument('--cfg_path', type=str, default=default_path, help='Path for diffusion model config file')
        parser.add_argument('--threshold', type=float, default=65., help='Validity threshold for pLDDT')
        parser.add_argument('--lengthscale_vendi', type=float, default=2.)
        default_oracle_path = files(sgpo) / Path('oracle/checkpoints/CreiLOV')
        parser.add_argument('--oracle_path', type=str, default=default_oracle_path, help='Path to oracle ensemble checkpoint directory')


    def _load_oracle_ensemble(self, oracle_path: Path) -> Sequence[OracleModel]:
        oracle_path = Path(oracle_path)
        if not oracle_path.exists():
            return []
        model_files = sorted(oracle_path.glob('*.pth'))
        if not model_files:
            return []
        seq_len = len(CREILOV_WILD_TYPE)
        alphabet_size = 20
        input_dim = seq_len * alphabet_size
        ensemble = []
        for f in model_files:
            model = OracleModel(input_dim=input_dim, hidden_dim=ORACLE_HIDDEN_DIM, dropout_rate=ORACLE_DROPOUT)
            model.load_state_dict(torch.load(f, map_location='cpu'))
            model.eval()
            ensemble.append(model)
        return ensemble

    @property
    def base_model(self) -> ProteinModel:
        return self._base_model
    
    
    @property
    def device(self) -> torch.device:
        return self._base_model.device


    def validity(self, samples: FlowTensor, kwargs: dict[str, Any]) -> torch.Tensor:
        strings = self.base_model.embed_to_sequence(samples)
        
        with torch.no_grad():
            results = self.esmfold.infer(list(strings))        
        
        plddt = results['mean_plddt']

        threshold = kwargs['threshold'] if 'threshold' in kwargs else self.threshold
        return torch.where(plddt > threshold, 1., 0.)


    @property
    def feature_layer(self) -> str: # TODO maybe this is good? it's already supposed to be an ESM embedding...
        """The name of the layer from which to extract features for the GP."""
        return 'input'


    def postprocess_latents(self, batch: Batch[FlowTensor]) -> FlowTensor:
        return batch.latents


    def postprocess_features(self, latents: FlowTensor, feats: Any) -> torch.Tensor:
        data: torch.Tensor = feats.data
        return data.mean(dim=-1)


    def visualize_sample(self, env: Environment[FlowTensor], batch: Batch[FlowTensor]) -> Figure:
        """Produce a matplotlib figure for visualizing the sample in the problem setup.

        Parameters
        ----------
        env : Environment[D]
            The environment in which the samples were generated.
        batch : Batch[D]
            The batch.
        """

        fig, _ = plt.subplots()

        return fig
    

    def save_sample(self, sample: FlowTensor, kwargs: dict[str, Any], filename: Path):
        """Save a *single* sample to the disk.
        
        Parameters
        ----------
        sample : D
            The sample to save, batch-size 1.
        kwargs : dict
            The keyword arguments used to generate the sample.
        filename : Path
            The file path where to save the sample, without extension.
        """
        torch.save(sample.data, f'{filename}.pt')
    

    def eval_sampling_kwargs(self, n: int) -> dict[str, Any]:
        """Provide keyword arguments for sampling during evaluation.

        Parameters
        ----------
        n : int
            The number of samples to be drawn.

        Returns
        -------
        dict
            A dictionary of keyword arguments for sampling.
        """
        return {}
    

    def compute_metrics(self, batch: Batch[FlowTensor]) -> dict[str, float]:
        """Compute relevant (global) metrics for the problem setup.

        Parameters
        ----------
        batch : Batch[D]
            The batch.

        Returns
        -------
        dict[str, float]
            A dictionary of computed metrics.
        """
        embeddings = batch.samples.data
        X = embeddings.mean(dim=1)
        kernel = gpytorch.kernels.RBFKernel()
        if self.lengthscale is not None:
            kernel.lengthscale = self.lengthscale
        K = kernel(X, X).cpu().numpy()
        vendi_score_val = vendi.score_K(K)

        return {"vendi": float(vendi_score_val)}
    

    @staticmethod
    def _onehot_encode(sequence: str) -> np.ndarray:
        alphabet = "ACDEFGHIKLMNPQRSTVWY"
        encoding = np.zeros((len(sequence), len(alphabet)))
        for i, aa in enumerate(sequence):
            if aa in alphabet:
                encoding[i, alphabet.index(aa)] = 1
        return encoding.flatten()

    @staticmethod
    def _hamming_distance(s1: str, s2: str) -> int:
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))

    # TODO this punished OOD generations... maybe it will be a problem
    @staticmethod
    def _hamming_penalty(distance: int, cutoff: int = HAMMING_PENALTY_CUTOFF, rate: float = HAMMING_PENALTY_RATE) -> float:
        if distance <= cutoff:
            return 1.0
        return rate ** (distance - cutoff)

    def _oracle_predict(self, sequences: Sequence[str]) -> np.ndarray:
        """Run the oracle ensemble on a list of sequences, returning mean fitness predictions."""
        if not self.oracle_ensemble:
            return np.full(len(sequences), float('nan'))

        encodings = np.array([self._onehot_encode(seq) for seq in sequences])
        X = torch.tensor(encodings, dtype=torch.float32)

        all_predictions = np.zeros((len(self.oracle_ensemble), len(sequences)))
        with torch.no_grad():
            for i, model in enumerate(self.oracle_ensemble):
                preds = model(X).cpu().numpy().reshape(-1)
                penalties = np.array([
                    self._hamming_penalty(self._hamming_distance(CREILOV_WILD_TYPE, seq))
                    for seq in sequences
                ])
                preds = preds * penalties
                preds = np.maximum(preds, 0.0)
                all_predictions[i] = preds

        return np.mean(all_predictions, axis=0)

    def compute_sample_metrics(self, sample_files: list[SampleFile]) -> dict[str, dict[str, float]]:
        """Compute relevant metrics on individual samples.

        Parameters
        ----------
        sample_files : list[SampleFile]
            List of files containing the samples to compute metrics on.

        Returns
        -------
        dict[str, dict[str, float]]
            A dictionary mapping sample names to their computed metrics.
        """
        if not self.oracle_ensemble:
            return dict()

        # Load embeddings and convert to sequences
        names = []
        embeddings = []
        for sf in sample_files:
            data = torch.load(sf.file, map_location='cpu')
            embeddings.append(data)
            names.append(sf.file.stem)

        stacked = FlowTensor(torch.stack(embeddings))
        sequences = self.base_model.embed_to_sequence(stacked)

        # Run oracle ensemble
        fitness_scores = self._oracle_predict(sequences)

        results: dict[str, dict[str, float]] = {}
        for name, fitness, seq, sf in zip(names, fitness_scores, sequences, sample_files):
            metrics: dict[str, float] = {'fitness': float(fitness)}
            metrics['hamming_distance'] = float(self._hamming_distance(CREILOV_WILD_TYPE, seq))
            metrics['is_valid'] = float(sf.is_valid)
            results[name] = metrics

        return results