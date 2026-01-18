from typing import Generic, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from flowgym import D, BaseModel, Environment
from matplotlib.figure import Figure
import os


@dataclass
class SampleFile:
    is_valid: bool
    file: str


class ProblemSetup(ABC, Generic[D]):
    def __init__(self, args: dict[str, Any]):
        """Initialize the problem setup with given arguments.

        Parameters
        ----------
        args : dict[str, Any]
            A dictionary of arguments to configure the problem setup.
        """
        pass

    @property
    @abstractmethod
    def base_model(self) -> BaseModel[D]:
        """The (data pre-trained) generative flow model used in the problem setup."""
        raise NotImplementedError

    @abstractmethod
    def validity(self, x: D, kwargs: dict) -> torch.Tensor:
        """Validity/verifier function that checks whether a sample is valid or not.

        Parameters
        ----------
        x : D
            The sample to check.

        kwargs : Any
            Additional keyword arguments that may be needed for validity checking.

        Returns
        -------
        torch.Tensor
            A boolean tensor indicating whether each sample is valid.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def feature_layer(self) -> str:
        """The name of the layer from which to extract features for the GP."""
        raise NotImplementedError

    def latent_postprocess(self, latents: D, valids: torch.Tensor, kwargs: dict) -> D:
        """Post-process latents generated from the base model.

        Parameters
        ----------
        latents : D
            The latents to post-process.

        valids : torch.Tensor
            A boolean tensor indicating whether each latent is valid.

        kwargs : dict
            Keyword arguments.

        Returns
        -------
        D
            The post-processed sample.
        """
        return latents

    @abstractmethod
    def feature_postprocess(self, x: D, feats: Any) -> torch.Tensor:
        """Post-process features extracted from the base model.

        Parameters
        ----------
        x : D
            The input sample corresponding to the features.

        feats : Any
            The raw features extracted from the base model.

        Returns
        -------
        torch.Tensor
            The post-processed features.
        """
        raise NotImplementedError

    @abstractmethod
    def visualize_sample(
        self, env: Environment[D], samples: list[D], valids: list[torch.Tensor]
    ) -> Figure:
        """Should output a matplotlib figure for visualizing the sample in the problem setup.

        Parameters
        ----------
        env : Environment[D]
            The environment in which the samples were generated.

        samples : list[D]
            The samples to visualize, in order of obtaining them.

        valids : list[torch.Tensor]
            The validity tensors corresponding to the samples.
        """
        raise NotImplementedError

    @abstractmethod
    def save_sample(self, sample: D, kwargs: dict, filename: os.PathLike | str):
        """Save a sample to the disk.
        
        Parameters
        ----------
        sample : D
            The sample to save.

        kwargs : dict
            Additional keyword arguments that may be needed to save the sample.

        filename : os.PathLike | str
            The file path where to save the sample, without extension. The method will add the
            appropriate extension.
        """
        raise NotImplementedError

    def eval_sampling_kwargs(self, n: int) -> dict:
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

    def compute_metrics(
        self,
        samples: list[D],
        valids: list[torch.Tensor],
        kwargs: list[dict],
    ) -> dict[str, float]:
        """Compute relevant (global) metrics for the problem setup.
        
        Parameters
        ----------
        samples : list[D]
            The samples for which to compute metrics.

        valids : list[torch.Tensor]
            The validity tensor corresponding to the samples.

        kwargs : list[dict]
            Keyword arguments corresponding to the samples.

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
