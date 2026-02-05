from typing import Any
from argparse import ArgumentParser
from importlib_resources import files
from pathlib import Path
from hydra.utils import instantiate
from torch import Tensor, device
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf


from sgpo.models.continuous import ContinuousModel
from flowgym.types import FlowTensor
from flowgym.base_models import BaseModel
from flowgym.schedulers import Scheduler, NoiseSchedule
from flowgym.utils import append_dims

from active_pretraining.problem_setup import ProblemSetup, SampleFile, Batch
import sgpo


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
        raise NotImplementedError
    

    def postprocess(self, x: FlowTensor) -> FlowTensor:
        # TODO maybe map to and from ESM embeddings???
        raise NotImplementedError
    

    def sample_p0(self, n: int, **kwargs: Any) -> tuple[FlowTensor, dict[str, Any]]:
        return super().sample_p0(n, **kwargs)
    

    def forward(self, x: FlowTensor, t: Tensor, **kwargs: Any) -> FlowTensor:
        seq_len: int= self.sgpo_model.seq_len
        infill_mask = (torch.ones(seq_len) != self.sgpo_model.tokenizer.pad_id-100).to(self.device) 
        attn_mask = torch.ones((x.data.shape[0], seq_len),dtype=torch.bool, device=self.device)

        with torch.no_grad():
            sigma = self.scheduler.sigma(x, t).data # TODO or just use idxs and bypass everything? test both
            n = len(self.sgpo_model.noise_schedule.sigmas)
            idxs = (t * n).round().clamp_max(n - 1).int()
            f_out = self.sgpo_model.model.network.forward(x/(sigma**2 + 1).sqrt(), idxs, attn_mask=attn_mask)
            out = self.sgpo_model.model.network.pred_xstart(
                x,
                t,
                attn_mask=attn_mask,
                sequence_output=f_out['sequence_output'],
                infill_mask=infill_mask
            )
            x1 = out['xstart']
        
    
        return FlowTensor(x1)



class ProteinProblemSetup(ProblemSetup[FlowTensor]):
    def __init__(self, args: dict[str, Any], device: device | None):
        super().__init__(args)
        cfg_path: str = args['cfg_path']
        config = OmegaConf.load(cfg_path)

        self._base_model = ProteinModel(config, device=device)


    @classmethod
    def add_args(cls, parser: ArgumentParser): # TODO interface with hydra somehow to get hierarchical configs... for now use static one from existing run
        default_path = files(sgpo) / Path('configs/sample_config.yaml')
        parser.add_argument('--cfg_path', type='str', default=default_path)


    @property
    def base_model(self) -> ProteinModel:
        return self._base_model


    def validity(self, samples: FlowTensor, kwargs: dict[str, Any]) -> torch.Tensor:
        tokens = self.base_model.tokenizer


    @property
    @abstractmethod
    def feature_layer(self) -> str:
        """The name of the layer from which to extract features for the GP."""
        raise NotImplementedError


    def postprocess_latents(self, batch: Batch[FlowTensor]) -> FlowTensor:
        """Post-process latents generated from the base model.

        Parameters
        ----------
        batch : Batch[D]
            The sample.

        Returns
        -------
        latent : D
            The post-processed latents.
        """
        return batch.latents


    @abstractmethod
    def postprocess_features(self, latents: FlowTensor, feats: Any) -> torch.Tensor:
        """Post-process features extracted from the base model.

        Parameters
        ----------
        latents : D
            The batch.
        feats : Any
            The raw features extracted from the base model.

        Returns
        -------
        features : torch.Tensor
            The post-processed features.
        """
        raise NotImplementedError
    

    @abstractmethod
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
    

    @abstractmethod
    def save_sample(self, sample: D, kwargs: dict[str, Any], filename: Path):
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
    

    def compute_metrics(self, batch: Batch[D]) -> dict[str, float]:
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