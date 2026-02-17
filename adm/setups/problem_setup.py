from typing import Generic, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from diffusiongym import D, BaseModel, Environment
from matplotlib.figure import Figure
from pathlib import Path
from argparse import ArgumentParser

from adm.utils import Batch


@dataclass
class SampleFile:
    is_valid: bool
    file: Path


class ProblemSetup(ABC, Generic[D]):
    def __init__(self, args: dict[str, Any]):
        """Initialize the problem setup with given arguments.

        Parameters
        ----------
        args : dict[str, Any]
            A dictionary of arguments to configure the problem setup.
        """
        pass

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        """Add problem setup specific arguments to the parser.

        Parameters
        ----------
        parser : ArgumentParser
            The argument parser to which to add arguments.
        """
        pass

    @property
    @abstractmethod
    def base_model(self) -> BaseModel[D]:
        """The (data pre-trained) generative flow model used in the problem setup."""
        raise NotImplementedError

    @abstractmethod
    def validity(self, samples: D, kwargs: dict[str, Any]) -> torch.Tensor:
        """Validity/verifier function that checks whether a sample is valid or not.

        Parameters
        ----------
        samples : D
            The samples to check.
        kwargs : dict
            The keyword arguments used to generate the samples.

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

    def postprocess_latents(self, batch: Batch[D]) -> D:
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
    def postprocess_features(self, latents: D, feats: Any) -> torch.Tensor:
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
    def visualize_sample(self, env: Environment[D], batch: Batch[D]) -> Figure:
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
    def save_sample(self, sample: D, kwargs: dict[str, Any], filename: Path) -> Path | None:
        """Save a *single* sample to the disk.
        
        Parameters
        ----------
        sample : D
            The sample to save, batch-size 1.
        kwargs : dict
            The keyword arguments used to generate the sample.
        filename : Path
            The file path where to save the sample, without extension.

        Returns
        -------            
        Path | None
            The path to the saved sample file, None if failed.
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
        dict[str, Any]
            A dictionary of keyword arguments for sampling.
        """
        return {}

    def compute_metrics(self, sample_files: list[SampleFile]) -> dict[str, float]:
        """Compute relevant (global) metrics for the problem setup.
        
        Parameters
        ----------
        sample_files : list[SampleFile]
            List of files containing the samples to compute metrics on.

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
