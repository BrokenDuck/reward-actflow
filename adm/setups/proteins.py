from typing import Any, Sequence, List
from argparse import ArgumentParser
from importlib_resources import files
from pathlib import Path
from hydra.utils import instantiate
from torch import Tensor, device
import torch
from omegaconf import OmegaConf
import esm
import numpy as np
from vendi_score import vendi
import gpytorch

from matplotlib.figure import Figure
from matplotlib import pyplot as plt

from sgpo.models.continuous import ContinuousModel
from sgpo.models.pretraining.collaters import ESMTokenizer
from diffusiongym import DDTensor, Reward, base_model_registry, reward_registry
from diffusiongym.base_models import BaseModel
from diffusiongym.schedulers import Scheduler, NoiseSchedule
from diffusiongym.environments import Environment
from diffusiongym.utils import append_dims

from adm.setups.problem_setup import ProblemSetup, SampleFile, Batch
import sgpo
from sgpo.oracle.train_oracle import OracleModel

CREILOV_WILD_TYPE = "MAGLRHTFVVADATLPDCPLVYASEGFYAMTGYGPDEVLGHNARFLQGEGTDPKEVQKIRDAIKKGEACSVRLLNYRKDGTPFWNLLTVTPIKTPDGRVSKFVGVQVDVTSKTEGKALA"
CREILOV_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

ORACLE_HIDDEN_DIM = 400
ORACLE_DROPOUT = 0.1
ORACLE_BATCH_SIZE = 128
HAMMING_PENALTY_CUTOFF = 70
HAMMING_PENALTY_RATE = 0.99

DEFAULT_PLDDT_THRESHOLD = 65.


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


class DDPMNoiseSchedule(NoiseSchedule[DDTensor]):
    def __init__(self, scheduler: Scheduler[DDTensor]) -> None:
        super().__init__()
        self.scheduler = scheduler
    

    def __call__(self, x: DDTensor, t: Tensor) -> DDTensor:
        return self.scheduler.beta(x, t) / self.scheduler.alpha(x, t)


class CosineSchedule(Scheduler[DDTensor]):
    def __init__(self) -> None:
        super().__init__()
        # self._noise_schedule = DDPMNoiseSchedule(self)


    def alpha(self, x: DDTensor, t: Tensor) -> DDTensor:
        sqrt_alpha_bar = torch.cos((1. - t) * torch.pi / 2.)
        return DDTensor(append_dims(sqrt_alpha_bar, x.data.ndim))


    def alpha_dot(self, x: DDTensor, t: Tensor) -> DDTensor:
        sqrt_alpha_bar_dot = torch.pi / 2. * torch.sin((1. - t) * torch.pi / 2.)
        return DDTensor(append_dims(sqrt_alpha_bar_dot.data, x.data.ndim))
    

    def beta(self, x: DDTensor, t: Tensor) -> DDTensor:
        alpha_bar = torch.cos((1. - t) * torch.pi / 2.).square()
        beta = torch.sqrt(1. - alpha_bar)
        return DDTensor(append_dims(beta, x.data.ndim))


    def beta_dot(self, x: DDTensor, t: Tensor) -> DDTensor:
        alpha_bar_dot = 2. * torch.cos((1. - t) * torch.pi / 2.) * torch.pi / 2. * torch.sin((1. - t) * torch.pi / 2.)
        return -DDTensor(append_dims(alpha_bar_dot, x.data.ndim)) / (2. * self.beta(x, t))


@base_model_registry.register("proteins/continuous_ESM")
class ProteinModel(BaseModel[DDTensor]):

    output_type = "endpoint"

    def __init__(self, cfg_path: str, device: device | None):
        super().__init__(device)
        config = OmegaConf.load(cfg_path)

        model = config.model.model
        model_name = config.pretrained_ckpt
        seq_len = config.data.seq_len

        self.sgpo_model: ContinuousModel = instantiate(model, model_name=model_name, seq_len=seq_len, device=device, _recursive_=True)
        self.network = self.sgpo_model.model.network
        self._scheduler = CosineSchedule()
    
    
    @property
    def scheduler(self) -> Scheduler[DDTensor]:
        return self._scheduler


    def preprocess(self, x: DDTensor, **kwargs: Any) -> tuple[DDTensor, dict[str, Any]]:
        # TODO maybe map to and from ESM embeddings???
        return x, kwargs
    

    def postprocess(self, x: DDTensor) -> DDTensor:
        epsilon = 1e-4
        t = (1. - epsilon) * torch.ones((x.data.shape[0],)).to(self.device)
        out = self.network_forward(x.data, t)
        return DDTensor(out['probs'])
    

    def sample_p0(self, n: int, **kwargs: Any) -> tuple[DDTensor, dict[str, Any]]:
        return DDTensor(self.sgpo_model.get_start(n)), kwargs
    

    def forward(self, x: DDTensor, t: Tensor, **kwargs: Any) -> DDTensor: 
        if 'debug' in kwargs and kwargs['debug']:
            return DDTensor(torch.randn_like(x.data).to(x.device))
        return DDTensor(self.network_forward(x.data, t)['xstart'])


    def probs_to_sequence(self, probs: DDTensor) -> Sequence[str]:
        tokens = probs.data.argmax(dim=-1)
        return [self.sgpo_model.tokenizer.untokenize(s) for s in tokens]


    def probs_to_embedding(self, probs: DDTensor) -> DDTensor:
        network = self.sgpo_model.model.network
        with torch.no_grad():
            emb_table = network.esm_model.embed_tokens.weight.detach().cpu().float()
            emb_table = emb_table / (emb_table.norm(dim=-1, keepdim=True) + 1e-8)
            emb_table = emb_table * (network.in_channels ** 0.5)

        esm_embeds = probs.data @ emb_table
        return DDTensor(esm_embeds)


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


# TODO make general e.g. TrpB compatible as well
@reward_registry.register("proteins/fitness")
class ProteinFitnessReward(Reward[DDTensor]):
    """Fitness reward for protein sequences using an oracle ensemble.

    Expects ``sample`` to be token-probability tensors [batch, seq_len, vocab_size]
    as produced by ``ProteinModel.postprocess``.
    """

    def __init__(self, cfg_path: str | None = None, oracle_path: str | None = None) -> None:
        
        if cfg_path is not None:
            config = OmegaConf.load(cfg_path)
            tokenizer_cfg = OmegaConf.select(config, 'model.model.tokenizer')

            if tokenizer_cfg is not None:
                self.tokenizer = instantiate(tokenizer_cfg, sequences=True)
            else:
                self.tokenizer = ESMTokenizer(esm_model_name='esm2_t12_35M_UR50D')
        else:
            self.tokenizer = ESMTokenizer(esm_model_name='esm2_t12_35M_UR50D')
        resolved_path = Path(oracle_path) if oracle_path is not None else Path(str(files(sgpo) / Path('oracle/checkpoints/CreiLOV')))

        self.oracle_ensemble = self._load_oracle_ensemble(CREILOV_WILD_TYPE, CREILOV_ALPHABET, resolved_path)

    def __call__(self, sample: DDTensor, latent: DDTensor, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = sample.data.argmax(dim=-1)
        sequences = [self.tokenizer.untokenize(s) for s in tokens]
        fitness_scores = self._oracle_predict(sequences)
        reward = torch.tensor(fitness_scores, dtype=torch.float32)
        return reward, torch.ones_like(reward)

    @staticmethod
    def _load_oracle_ensemble(wild_type: str, alphabet: str, oracle_path: Path) -> Sequence[OracleModel]:
        oracle_path = Path(oracle_path)
        if not oracle_path.exists():
            return []
        model_files = sorted(oracle_path.glob('*.pth'))
        if not model_files:
            return []
        seq_len = len(wild_type)
        alphabet_size = len(alphabet)
        input_dim = seq_len * alphabet_size
        ensemble = []
        for f in model_files:
            model = OracleModel(input_dim=input_dim, hidden_dim=ORACLE_HIDDEN_DIM, dropout_rate=ORACLE_DROPOUT)
            model.load_state_dict(torch.load(f, map_location='cpu'))
            model.eval()
            ensemble.append(model)
        return ensemble

    @staticmethod
    def _onehot_encode(sequence: str, alphabet: str = CREILOV_ALPHABET) -> np.ndarray:
        encoding = np.zeros((len(sequence), len(alphabet)))
        for i, aa in enumerate(sequence):
            if aa in alphabet:
                encoding[i, alphabet.index(aa)] = 1
        return encoding.flatten()

    @staticmethod
    def _hamming_distance(s1: str, s2: str) -> int:
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))

    # TODO this punishes OOD generations... maybe it will be a problem
    @staticmethod
    def _hamming_penalty(distance: int, cutoff: int = HAMMING_PENALTY_CUTOFF, rate: float = HAMMING_PENALTY_RATE) -> float:
        if distance <= cutoff:
            return 1.0
        return rate ** (distance - cutoff)

    def _oracle_predict(self, sequences: Sequence[str]) -> np.ndarray:
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


@reward_registry.register("proteins/creilov")
class CreiLOVFitnessReward(ProteinFitnessReward):
    def __init__(self, cfg_path: str | None = None):
        creilov_path = str(files(sgpo) / Path('oracle/checkpoints/CreiLOV'))
        super().__init__(cfg_path, creilov_path)


class ProteinProblemSetup(ProblemSetup[DDTensor]):
    def __init__(self, args: dict[str, Any], device: device | None):
        super().__init__(args)
        cfg_path: str = args['cfg_path']
        self.threshold = args['threshold'] if 'threshold' in args else DEFAULT_PLDDT_THRESHOLD
        self.lengthscale = args['lengthscale_vendi'] if 'lengthscale_vendi' in args else None

        self._base_model = ProteinModel(cfg_path, device=device)
        self.esmfold = esm.pretrained.esmfold_v1().eval().to(device) if 'no_verifier' not in args or not args['no_verifier'] else None


    @classmethod
    def add_args(cls, parser: ArgumentParser): # TODO interface with hydra somehow to get hierarchical configs... for now use static one from existing run
        default_path = files(sgpo) / Path('configs/sample_config.yaml')
        parser.add_argument('--cfg_path', type=str, default=default_path, help='Path for diffusion model config file')
        parser.add_argument('--threshold', type=float, default=DEFAULT_PLDDT_THRESHOLD, help='Validity threshold for pLDDT')
        parser.add_argument('--lengthscale_vendi', type=float, default=2.)

    @property
    def base_model(self) -> ProteinModel:
        return self._base_model
    
    
    @property
    def device(self) -> torch.device:
        return self._base_model.device


    def validity(self, samples: DDTensor, kwargs: dict[str, Any]) -> torch.Tensor:
        if self.esmfold is None:
            return torch.ones((len(samples),)).bool()

        strings = self.base_model.probs_to_sequence(samples)

        bs = kwargs['batch_size'] if 'batch_size' in kwargs else 32
        
        str_list = list(strings)
        plddts = []
        with torch.no_grad():
            for i in range(0, len(str_list), bs):
                sublist = str_list[i: min(len(str_list), i + bs)]
                results = self.esmfold.infer(sublist)        
                plddt = results['mean_plddt']
                plddts.append(plddt.squeeze())

        plddt = torch.cat(plddts)

        threshold = kwargs['threshold'] if 'threshold' in kwargs else self.threshold
        return torch.where(plddt > threshold, 1, 0).bool()


    @property
    def feature_layer(self) -> str: # TODO maybe this is good? it's already supposed to be an ESM embedding...
        """The name of the layer from which to extract features for the GP."""
        return 'input'


    def postprocess_latents(self, batch: Batch[DDTensor]) -> DDTensor:
        return batch.latents


    def postprocess_features(self, latents: DDTensor, feats: Any) -> torch.Tensor:
        data: torch.Tensor = feats.data
        return data.mean(dim=-1)


    def visualize_sample(self, env: Environment[DDTensor], batch: Batch[DDTensor]) -> Figure:
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
    

    def save_sample(self, sample: DDTensor, kwargs: dict[str, Any], filename: Path) -> Path | None:
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
        torch.save(sample.data.clone(), f'{filename}.pt')
        return Path(f'{filename}.pt')


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
    

    def compute_metrics(self, sample_files: List[SampleFile]) -> dict[str, float]:
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

        names = []
        samples = []
        for sf in sample_files:
            data = torch.load(sf.file, map_location='cpu')
            samples.append(data)
            names.append(sf.file.stem)
        
        probs = torch.vstack(samples).float()
        esm_embeds = self.base_model.probs_to_embedding(DDTensor(probs)).data

        X = esm_embeds.mean(dim=1)
        kernel = gpytorch.kernels.RBFKernel()
        if self.lengthscale is not None:
            kernel.lengthscale = self.lengthscale
        K = kernel(X, X).cpu().numpy()
        vendi_score_val = vendi.score_K(K)

        tokens = probs.argmax(dim=-1)
        sequences = [self.base_model.sgpo_model.tokenizer.untokenize(s) for s in tokens]
        mutated_positions = [
            i for i in range(len(CREILOV_WILD_TYPE))
            if any(seq[i] != CREILOV_WILD_TYPE[i] for seq in sequences)
        ]
        if mutated_positions:
            n = len(sequences)
            position_entropies = []
            for pos in mutated_positions:
                counts = {}
                for seq in sequences:
                    aa = seq[pos]
                    counts[aa] = counts.get(aa, 0) + 1
                entropy = -sum((c / n) * np.log(c / n) for c in counts.values())
                position_entropies.append(entropy)
            shannon = float(np.mean(position_entropies))
        else:
            shannon = 0.0

        return {"vendi": float(vendi_score_val), "shannon_entropy": shannon}
    

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
        return dict()