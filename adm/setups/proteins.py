from typing import Any, Sequence
from argparse import ArgumentParser
from importlib_resources import files
from pathlib import Path
from hydra.utils import instantiate
from torch import Tensor, device
import torch
from omegaconf import OmegaConf
import esm
import esm.esmfold.v1.esmfold as _esmfold_module  # needed for compute_tm stub
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

from adm.setups.problem_setup import ProblemSetup
from adm.uncertainty import UncertaintyEstimator
from adm.utils import Batch
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
SPHERE_EXCLUSION_THRESHOLD = 0.35
REFERENCE_EVAL_DIR = Path("/cluster/scratch/kprotopapas/base_model/eval/base")

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

    def __call__(self, sample: DDTensor | None, latent: DDTensor, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        source = sample if sample is not None else latent
        tokens = source.data.argmax(dim=-1)
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
        self.validity_batch_size = args['validity_batch_size'] if 'validity_batch_size' in args else 32

        self._base_model = ProteinModel(cfg_path, device=device)
        esmfold_chunk_size = args.get('esmfold_chunk_size', None)

        if esmfold_chunk_size is not None:
            self.esmfold = esm.pretrained.esmfold_v1().eval().half().to(device)
            # We only use mean_plddt from ESMFold, never pTM. Stub out compute_tm so the
            # forward pass doesn't crash in fp16 (ptm_head logits contain NaN in fp16).
            _esmfold_module.compute_tm = lambda logits, *args, **kwargs: logits.new_zeros(())
            self.esmfold.set_chunk_size(esmfold_chunk_size)
        self._reference_embeddings: torch.Tensor | None = None


    @classmethod
    def add_args(cls, parser: ArgumentParser): # TODO interface with hydra somehow to get hierarchical configs... for now use static one from existing run
        default_path = files(sgpo) / Path('configs/sample_config.yaml')
        parser.add_argument('--cfg_path', type=str, default=default_path, help='Path for diffusion model config file')
        parser.add_argument('--threshold', type=float, default=DEFAULT_PLDDT_THRESHOLD, help='Validity threshold for pLDDT')
        parser.add_argument('--lengthscale_vendi', type=float, default=2.)
        parser.add_argument('--validity_batch_size', type=int, default=32, help='Batch size for ESMFold validity checks')
        parser.add_argument('--esmfold_chunk_size', type=int, default=None, help='Chunk size for ESMFold attention (smaller = less VRAM, slower)')

    @property
    def base_model(self) -> ProteinModel:
        return self._base_model
    
    
    @property
    def device(self) -> torch.device:
        return self._base_model.device


    def validity(self, samples: DDTensor, kwargs: dict[str, Any]) -> torch.Tensor:
        strings = self.base_model.probs_to_sequence(samples)

        bs = kwargs.get('batch_size', self.validity_batch_size)
        
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
    def feature_layer(self) -> str:
        return 'sgpo_model.model.network.encoder'


    def postprocess_latents(self, batch: Batch[DDTensor]) -> DDTensor:
        return batch.latents


    def postprocess_features(self, latents: DDTensor, feats: Any) -> torch.Tensor:
        hidden: torch.Tensor = feats[0]  # [batch, seq_len, hidden_size]
        return hidden.mean(dim=1)  # [batch, hidden_size]


    def visualize_sample(self, env: Environment[DDTensor], uncertainty: UncertaintyEstimator[DDTensor], batch: Batch[DDTensor]) -> Figure:
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
    

    def save_samples(self, samples: DDTensor, kwargs: dict, dir: Path) -> bool:
        torch.save(samples.data.clone(), dir / "samples.pt")
        return True

    def load_samples(self, dir: Path) -> tuple[DDTensor, dict]:
        data = torch.load(dir / "samples.pt", map_location="cpu")
        return DDTensor(data), {}


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
    

    def get_sequence_array(self, samples: DDTensor, kwargs: dict) -> np.ndarray:
        """Extract sequences as integer arrays of shape (n, seq_len)."""
        probs = samples.data.float()
        tokens = probs.argmax(dim=-1)
        sequences = [self.base_model.sgpo_model.tokenizer.untokenize(s) for s in tokens]
        return np.array([[ord(c) for c in seq] for seq in sequences], dtype=np.int32)

    @staticmethod
    def _identity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        L = a.shape[1]
        return (a[:, None, :] == b[None, :, :]).sum(axis=-1) / L

    @staticmethod
    def compute_novelty(current_seqs: np.ndarray, reference_seqs: np.ndarray) -> float:
        """Novelty = 1 - mean max sequence identity to reference set."""
        if len(current_seqs) == 0 or len(reference_seqs) == 0:
            return 0.0
        identity = ProteinProblemSetup._identity_matrix(current_seqs, reference_seqs)
        return float(1.0 - identity.max(axis=1).mean())

    @staticmethod
    def compute_cumulative_cluster_metrics(
        current_seqs: np.ndarray,
        cumulative_centers: np.ndarray | None,
        threshold: float = SPHERE_EXCLUSION_THRESHOLD,
    ) -> tuple[dict[str, float], np.ndarray]:
        """Track cluster expansion across iterations using sequence identity sphere exclusion."""
        self_identity = ProteinProblemSetup._identity_matrix(current_seqs, current_seqs)
        n = len(current_seqs)
        picked = [0]
        for i in range(1, n):
            if all(self_identity[i, j] < threshold for j in picked):
                picked.append(i)
        current_centers = current_seqs[picked]

        if cumulative_centers is None:
            return {
                "cumulative_clusters": float(len(current_centers)),
                "coverage_of_cumulative": 1.0,
                "new_clusters": float(len(current_centers)),
                "clusters_lost": 0.0,
            }, current_centers.copy()

        def _max_identity(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
            return ProteinProblemSetup._identity_matrix(query, reference).max(axis=1)

        coverage_sim = _max_identity(cumulative_centers, current_seqs)
        covered = (coverage_sim >= threshold).sum()
        n_prev = len(cumulative_centers)
        clusters_lost = n_prev - int(covered)

        novelty_sim = _max_identity(current_centers, cumulative_centers)
        new_centers = current_centers[novelty_sim < threshold]

        updated = np.vstack([cumulative_centers, new_centers]) if len(new_centers) > 0 else cumulative_centers.copy()

        return {
            "cumulative_clusters": float(len(updated)),
            "coverage_of_cumulative": float(covered) / n_prev if n_prev > 0 else 1.0,
            "new_clusters": float(len(new_centers)),
            "clusters_lost": float(clusters_lost),
        }, updated

    def _load_reference_embeddings(self) -> torch.Tensor:
        samples = torch.load(REFERENCE_EVAL_DIR / "samples.pt", map_location="cpu")
        valids = torch.load(REFERENCE_EVAL_DIR / "valids.pt", map_location="cpu")
        valid_probs = samples[valids].float()
        embeddings = self.base_model.probs_to_embedding(DDTensor(valid_probs)).data
        return embeddings.mean(dim=1)

    @staticmethod
    def compute_fid(current_embs: np.ndarray, reference_embs: np.ndarray) -> float:
        from scipy.linalg import sqrtm

        if len(current_embs) < 2 or len(reference_embs) < 2:
            return float("nan")

        mu1 = current_embs.mean(axis=0)
        mu2 = reference_embs.mean(axis=0)
        sigma1 = np.cov(current_embs, rowvar=False)
        sigma2 = np.cov(reference_embs, rowvar=False)

        diff = mu1 - mu2
        covmean = sqrtm(sigma1 @ sigma2)
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))

    def compute_metrics(self, samples: DDTensor, kwargs: dict) -> dict[str, float]:
        probs = samples.data.float()
        esm_embeds = self.base_model.probs_to_embedding(DDTensor(probs)).data

        X = esm_embeds.mean(dim=1)

        if self._reference_embeddings is None:
            self._reference_embeddings = self._load_reference_embeddings()
        n = len(X)
        idx = torch.randperm(len(self._reference_embeddings))[:n]
        fid = self.compute_fid(X.cpu().numpy(), self._reference_embeddings[idx].numpy())

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

        p = probs.clamp(min=1e-10)
        token_entropy = -(p * p.log()).sum(dim=-1).mean()
        result: dict[str, float] = {"vendi": float(vendi_score_val), "shannon_entropy": shannon, "fid": fid, "avg_token_entropy": float(token_entropy)}

        if len(sequences) >= 2:
            seq_array = np.array([[ord(c) for c in seq] for seq in sequences], dtype=np.int32)
            self_identity = self._identity_matrix(seq_array, seq_array)
            n = len(seq_array)
            picked = [0]
            for i in range(1, n):
                if all(self_identity[i, j] < SPHERE_EXCLUSION_THRESHOLD for j in picked):
                    picked.append(i)
            result["sphere_exclusion_diversity"] = len(picked) / n
            result["n_clusters"] = float(len(picked))

        return result


    def compute_sample_metrics(self, samples: DDTensor, kwargs: dict) -> list[dict[str, float]]:
        return [{} for _ in range(len(samples.data))]
