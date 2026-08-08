from abc import ABC, abstractmethod
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import torch
from diffusiongym import FineTuningSetup
from diffusiongym.types import DDBatch
from matplotlib.figure import Figure

from reward_actflow.uncertainty import UncertaintyEstimator
from reward_actflow.utils import Batch


class ProblemSetup[D: DDBatch](ABC):
    """Everything about an ActFlow problem that is not the fine-tuning algorithm.

    The generative side (geometry, base distribution, codec, network) is supplied
    to diffusiongym as a registered `ModalityProvider`; a setup only names it via
    `modality_id` / `modality_kwargs`. What stays here is the part diffusiongym
    has no opinion about: the black-box verifier, which representation the
    surrogate is fitted on, and how to look at a run.
    """

    def __init__(self, args: dict[str, Any], device: torch.device | None = None):
        """Initialize the problem setup with given arguments.

        Parameters
        ----------
        args : dict[str, Any]
            A dictionary of arguments to configure the problem setup.
        device : torch.device, optional
            Device to place the problem on.
        """

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        """Add problem setup specific arguments to the parser."""

    @property
    @abstractmethod
    def modality_id(self) -> str:
        """Registry id of the `ModalityProvider` backing this problem."""
        raise NotImplementedError

    @property
    def modality_kwargs(self) -> dict[str, Any]:
        """Constructor overrides for the modality provider."""
        return {}

    @abstractmethod
    def validity(self, samples: D, kwargs: dict[str, Any]) -> torch.Tensor:
        """Black-box verifier: whether each sample is valid.

        This is `v` of Algorithm 1 line 5.

        Parameters
        ----------
        samples : D
            The samples to check.
        kwargs : dict
            The conditioning used to generate the samples.

        Returns
        -------
        torch.Tensor
            A boolean tensor indicating whether each sample is valid.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def feature_layer(self) -> str:
        """Layer the surrogate's features are read from, or `"input"`.

        Dot notation, resolved relative to the `FlowModel` wrapper — so a path
        into a wrapped network includes the wrapper's attribute name.
        """
        raise NotImplementedError

    def postprocess_latents(self, batch: Batch[D]) -> D:
        """Post-process latents generated from the flow model."""
        return batch.latents

    @abstractmethod
    def postprocess_features(self, latents: D, feats: Any) -> torch.Tensor:
        """Map the raw hooked activation to a 2-D `(n, feat_dim)` tensor.

        Parameters
        ----------
        latents : D
            The batch the features were extracted for.
        feats : Any
            The raw features extracted from the model.
        """
        raise NotImplementedError

    @abstractmethod
    def visualize_sample(
        self,
        setup: FineTuningSetup,
        uncertainty: UncertaintyEstimator[D],
        batch: Batch[D],
    ) -> tuple[Figure, dict[str, float]]:
        """Produce a matplotlib figure summarizing one iteration.

        Returns the figure *and* any metrics computed along the way, so that
        expensive shared work — for the toy, a 50k-sample draw — is not repeated
        by a separate metrics call.

        Parameters
        ----------
        setup : FineTuningSetup
            The assembled environment, policies, dynamics and time grid.
        uncertainty : UncertaintyEstimator[D]
            Current surrogate.
        batch : Batch[D]
            The batch collected this iteration.
        """
        raise NotImplementedError

    @abstractmethod
    def save_samples(self, samples: D, kwargs: dict, dir: Path) -> bool:
        """Save a batch of samples to disk.

        Notes
        -----
        You do not need to save all conditioning, only what evaluation needs.
        """
        raise NotImplementedError

    @abstractmethod
    def load_samples(self, dir: Path) -> tuple[D, dict]:
        """Load a batch of samples saved by `save_samples`."""
        raise NotImplementedError

    def eval_sampling_kwargs(self, n: int) -> dict[str, Any]:
        """Conditioning to use when drawing `n` evaluation samples."""
        return {}

    def compute_metrics(self, samples: D, kwargs: dict) -> dict[str, float]:
        """Compute global metrics for the problem setup."""
        return {}

    def compute_sample_metrics(
        self, samples: D, kwargs: dict
    ) -> list[dict[str, float]]:
        """Compute relevant metrics on individual samples."""
        return [{} for _ in range(len(samples))]

    # ------------------------------------------------------------------
    # ActFlow-R: optional, not needed by base ActFlow (task_agnostic.py never
    # calls any of these three).
    # ------------------------------------------------------------------

    def task_reward(self, samples: D, kwargs: dict[str, Any]) -> torch.Tensor:
        """Black-box task reward `r~(x)`, ActFlow-R Algorithm 1 line 6.

        Queried only on the verifier-valid subset. Raises by default; a setup
        that ActFlow-R runs on must override this, but base ActFlow's loop
        never calls it, so leaving it unimplemented is not a latent bug for
        every other setup.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement a task reward. "
            "ActFlow-R needs one; base ActFlow never calls this."
        )

    def anchor_latents(self, n: int, device: torch.device) -> D | None:
        """The frozen anchor pool `P`, Algorithm 1's `p_data`, in *latent*
        space — the space `UpdateFlow`'s flow-matching loss regresses on,
        which need not equal sample space (it does for the toy's
        `IdentityCodec`, not in general).

        `None` (the default) tells `ActFlowRLoop` to draw `P` from
        `p_1^{theta_0}` instead — always a valid fallback, so a setup only
        needs to override this when `p_data` exists and is preferred over the
        pretrained model's own samples.
        """
        return None

    def diagnostic_coordinates(self, latents: D) -> torch.Tensor:
        """A fixed, low-dimensional descriptor for `diagnostics.py` — cluster
        counts and per-cell density ratios need *some* metric space, and it
        must deliberately not be `phi_s` (the surrogate's representation
        under the *current* iterate, which drifts as theta moves and would
        silently change what "the same cell" means between iterations).

        Raises by default; a setup that ActFlow-R runs diagnostics on must
        override this. Returns a `(n, k)` tensor, `k` small (the diagnostics
        that consume this assume `k <= 2`).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement diagnostic_coordinates. "
            "ActFlow-R's cluster-count and density-ratio diagnostics need one."
        )

    def visualize_reward_sample(
        self,
        setup: FineTuningSetup,
        uncertainty: UncertaintyEstimator[D],
        reward_uncertainty: UncertaintyEstimator[D],
        anchors: D,
        batch: Batch[D],
    ) -> tuple[Figure, dict[str, float]] | None:
        """ActFlow-R's richer figure: the reward field, the frozen anchor
        pool, and this iteration's samples, alongside `visualize_sample`'s
        usual uncertainty/support panels — so whether the model's support
        (the "generable set") is moving toward higher reward is visible
        frame to frame, not just whether it is growing.

        `None` (the default) tells `ActFlowRLoop` to fall back to
        `visualize_sample`; a setup only needs to override this to get the
        extra panels, ActFlow-R runs correctly either way.
        """
        return None
