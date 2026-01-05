from typing import Generic, Any
from abc import ABC, abstractmethod

import torch
from flowgym import D, BaseModel, Environment
from matplotlib.figure import Figure


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
    def validity(self, x: D) -> torch.Tensor:
        """Validity/verifier function that checks whether a sample is valid or not.

        Parameters
        ----------
        x : D
            The sample to check.

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

    def sample_postprocess(self, x: D) -> D:
        """Post-process samples generated from the base model.

        Parameters
        ----------
        x : D
            The sample to post-process.

        Returns
        -------
        D
            The post-processed sample.
        """
        return x

    @abstractmethod
    def feature_postprocess(self, feats: Any) -> torch.Tensor:
        """Post-process features extracted from the base model.

        Parameters
        ----------
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
