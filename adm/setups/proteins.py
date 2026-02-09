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

from matplotlib.figure import Figure

from sgpo.models.continuous import ContinuousModel
from sgpo.models.pretraining.model.continuous_diffusion import GaussianDiffusionTransformer
from flowgym.types import FlowTensor
from flowgym.base_models import BaseModel
from flowgym.schedulers import Scheduler, NoiseSchedule
from flowgym.environments import Environment
from flowgym.utils import append_dims

from active_pretraining.problem_setup import ProblemSetup, SampleFile, Batch
import sgpo


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
        return -FlowTensor(alpha_bar_dot) / (2. * self.beta(x, t))


class ProteinModel(BaseModel[FlowTensor]):

    output_type = "endpoint"

    def __init__(self, config: DictConfig | ListConfig, device: device | None):
        super().__init__(device)
        model = config.model.model
        model_name = config.pretrained_ckpt
        seq_len = config.data.seq_len

        self.sgpo_model: ContinuousModel = instantiate(model, model_name=model_name, seq_len=seq_len, device=device, _recursive_=True)
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
        return FlowTensor(self.network_forward(x.data, t)['xstart'])


    def embed_to_sequence(self, embeds: FlowTensor) -> Sequence[str]:
        epsilon = 1e-4 # TODO change magic number
        t = (1. - epsilon) * torch.ones((embeds.data.shape[0],)).to(self.device)
        out = self.network_forward(embeds.data, t)

        tokens = out['probs'].argmax(dim=-1)
        strings = [self.sgpo_model.tokenizer.untokenize(s) for s in tokens]
        return strings


    def network_forward(self, x: torch.Tensor, t: torch.Tensor) -> dict[str, torch.Tensor]:
        t = 1. - t
        net = self.sgpo_model
        infill_mask = (torch.ones(net.seq_len) != net.tokenizer.pad_id-100).to(net.device) # TODO change magic number
        attn_mask = torch.ones((x.shape[0], x.shape[1]), dtype=torch.bool, device=net.device)
                               
        idx = (t * len(net.noise_schedule.sigmas)).round().clamp(0, len(net.noise_schedule.sigmas) - 1).long()
        sigma = net.noise_schedule.sigmas.to(net.device)[idx].reshape((x.shape[0], 1, 1))
        f_out = net.model.network(x/(sigma**2 + 1).sqrt(), idx, attn_mask=attn_mask)

        out = net.model.network.pred_xstart(
            x,
            t,
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
        config = OmegaConf.load(cfg_path)

        self._base_model = ProteinModel(config, device=device)
        self.esmfold = esm.pretrained.esmfold_v1().eval().to(device)


    @classmethod
    def add_args(cls, parser: ArgumentParser): # TODO interface with hydra somehow to get hierarchical configs... for now use static one from existing run
        default_path = files(sgpo) / Path('configs/sample_config.yaml')
        parser.add_argument('--cfg_path', type=str, default=default_path, help='Path for diffusion model config file')


    @property
    def base_model(self) -> ProteinModel:
        return self._base_model


    def validity(self, samples: FlowTensor, kwargs: dict[str, Any]) -> torch.Tensor:
        # TODO pass samples.data or self.network_forward(samples.data, t=0.0005)? 
        logits = self.base_model.sgpo_model.model.network.cls(samples.data) # TODO hardcoded for GaussianDiffusionTransformer
        probas = F.softmax(logits)
        tokens = probas.argmax(dim=-1)
        strings = [self.base_model.tokenizer.untokenize(s) for s in tokens]
        
        results = []
        with torch.no_grad():
            for s in strings:
                results.append(self.esmfold.infer(s))        
            
        threshold = kwargs['threshold']
        plddt = torch.vstack([r['mean_plddt'] for r in results])
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
        raise NotImplementedError
    

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
        raise NotImplementedError
    

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
        raise NotImplementedError
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
        return dict()
    

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