"""Surrogate uncertainty estimation over flow-model representations.

Two objects live here:

  FlowFeatureExtractor  maps a terminal sample x_1 to the representation
      phi_s(x_1) that the surrogate is fitted on. This is the `phi_s^t` of
      ActFlow Algorithm 1: the sample is noised to the representation time-step
      s and pushed through the *current* flow model, and a named intermediate
      layer is read off with a forward hook.
  UncertaintyEstimator  the surrogate itself, supplying a posterior mean and an
      uncertainty per sample (line 3 of the algorithm, `sigma_t`).

The estimator is deliberately *not* a reward. The reward is a separate object
(`reward_actflow.rewards.uncertainty.ActFlowUncertaintyReward`) that implements
diffusiongym's `RewardEvaluator` protocol and decides how the mean, the
uncertainty and the black-box verifier combine into a scalar.
"""

from abc import abstractmethod
from argparse import ArgumentParser
from collections.abc import Callable, Mapping
from typing import Any

import torch
from diffusiongym.core.schedule import AffineSchedule
from diffusiongym.core.space import LatentGeometry
from diffusiongym.types import DDBatch
from diffusiongym.utils import dict_to_device
from torch import nn


class FlowFeatureExtractor[D: DDBatch](nn.Module):
    """Extract features for a sample from a specific layer of a flow model.

    Parameters
    ----------
    model : FlowModel[D]
        The flow model to read representations from. Bind this to the *train*
        policy: `phi_s^t` in Algorithm 1 is the representation under the current
        iterate, not under the frozen pretrained model.
    geometry : LatentGeometry[D]
        Used to draw the base noise when constructing x_t.
    schedule : AffineSchedule
        Interpolation schedule, so that x_t = a(t) * x_base + b(t) * x_1 matches
        the path the model was trained on.
    layer : str
        Name of the layer to read, in dot notation relative to `model`
        (e.g. `"_mlp.blocks.1"`). Numeric segments index into a container.
        The special value `"input"` skips the model entirely and uses the sample
        itself as the representation.
    timestep : float
        Representation time-step s in (0, 1). Earlier timesteps give more
        semantic features, later ones more low-level features.
    postprocess : Callable[[D, Any], torch.Tensor]
        Maps the raw hooked activation to a 2-D `(n, feat_dim)` tensor.
    """

    def __init__(
        self,
        model: Any,
        geometry: LatentGeometry[D],
        schedule: AffineSchedule,
        layer: str,
        timestep: float,
        postprocess: Callable[[D, Any], torch.Tensor],
    ):
        super().__init__()
        self.model = model
        self.geometry = geometry
        self.schedule = schedule
        self.layer = layer
        self.timestep = timestep
        # A list rather than a scalar slot so that "nothing was captured" is a
        # length check rather than a None check the type checker narrows away.
        self._features: list[Any] = []
        self.postprocess = postprocess

        if layer != "input":
            self._register_hook()

    @property
    def is_static(self) -> bool:
        """Whether the representation is independent of the model's weights.

        True only for `"input"`, where the sample is its own representation. Any
        hooked layer moves with theta, so its features cannot be cached across
        iterations.
        """
        return self.layer == "input"

    def _get_module_by_name(self, module: Any, name: str):
        for part in name.split("."):
            module = module[int(part)] if part.isdigit() else getattr(module, part)

        return module

    def _register_hook(self):
        target = self._get_module_by_name(self.model, self.layer)

        def hook_fn(_, __, output):
            self._features.append(output)

        self._hook = target.register_forward_hook(hook_fn)

    def forward(self, x1: D, **conditioning: Any) -> torch.Tensor:
        if self.layer == "input":
            return self.postprocess(x1, x1)

        self._features.clear()

        x0 = self.geometry.standard_normal_like(x1)
        t = self.timestep * torch.ones(len(x1), device=x1.device)
        xt = x0 * self.schedule.a(t) + x1 * self.schedule.b(t)

        # Forward pass through the model to populate the hook
        _ = self.model(xt, t, conditioning=conditioning)

        if not self._features:
            raise RuntimeError(f"No features captured from layer '{self.layer}'")

        feats = self.postprocess(x1, self._features[-1])
        return feats / feats.norm(dim=-1, keepdim=True)

    def remove_hook(self):
        self._hook.remove()


class UncertaintyEstimator[D: DDBatch]:
    """Abstract surrogate supplying a posterior mean and an uncertainty.

    Parameters
    ----------
    feat_extractor : FlowFeatureExtractor[D]
        Feature extractor that maps latents to features.
    feat_dim : int
        Dimensionality of the extracted features.
    device : torch.device, optional
        Device to fit the surrogate on. Defaults to CPU.
    args : dict[str, Any]
        A dictionary of arguments to configure the estimator.
    """

    def __init__(
        self,
        feat_extractor: FlowFeatureExtractor[D],
        feat_dim: int,
        device: torch.device | str | None = None,
        args: Mapping[str, Any] | None = None,
    ):
        self.feat_extractor = feat_extractor
        self.feat_dim = feat_dim
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        self.args = dict(args or {})
        self.num_observations = 0
        self.label_std = 0.0
        self.label_mean = 0.0
        self._feature_cache: list[torch.Tensor] = []

        self._init_estimator()

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        """Add estimator-specific arguments to the parser."""

    @abstractmethod
    def _init_estimator(self):
        """Initialize the estimator."""

    @abstractmethod
    def _update_estimator(self, feats: torch.Tensor, labels: torch.Tensor):
        """Update the estimator with new features and labels.

        Parameters
        ----------
        feats : torch.Tensor
            The extracted features, shape `(n, feat_dim)`.
        labels : torch.Tensor
            The corresponding labels, shape `(n,)`.
        """

    @abstractmethod
    def _mean_and_uncertainty(
        self, feats: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the posterior mean and uncertainty, each of shape `(n,)`."""

    def _get_feats(self, latent: D, **conditioning: Any) -> torch.Tensor:
        latent = latent.to(self.device)
        conditioning = dict_to_device(conditioning, self.device)
        return self.feat_extractor(latent, **conditioning)

    def mean_and_uncertainty(
        self, latent: D, **conditioning: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Posterior mean and uncertainty for a batch of latents.

        Not wrapped in `no_grad`: the differentiable terminal cost that Adjoint
        Matching needs differentiates straight through this call.

        Before the first observation the surrogate is not conditioned on
        anything, so rather than let the backing implementation fail on an empty
        training set this returns the flat prior — zero mean, unit uncertainty.
        Every sample then scores identically, which is the correct behaviour for
        `D_0 = None`: there is nothing to prefer yet.

        The prior is built *from the features* rather than out of thin air so
        that it keeps an autograd graph with an identically-zero gradient.
        Adjoint Matching differentiates the terminal cost on its very first
        iteration, and a detached constant would abort the run rather than
        contribute the zero adjoint a flat prior actually implies.
        """
        feats = self._get_feats(latent, **conditioning)

        if self.num_observations == 0:
            flat = feats.sum(dim=-1) * 0.0
            return flat, flat + 1.0

        return self._mean_and_uncertainty(feats)

    def _extract_all(
        self,
        latents: list[D],
        conditioning: list[dict[str, Any]],
    ) -> list[torch.Tensor]:
        """Features for every batch in `D`, reusing earlier work when it is valid.

        `set_data` is called on the whole replay buffer every iteration, so a
        naive implementation extracts O(|D|) features per round and O(T^2) over a
        run. That is invisible on the toy — `feature_layer = "input"` returns the
        sample itself without touching the model — but it is a full network pass
        over the entire buffer per iteration for anything else.

        Caching is only sound when the representation does not depend on the
        model: `phi_s^t` is defined against the *current* iterate, so for a
        hooked layer the features genuinely change as theta moves and must be
        recomputed. `FlowFeatureExtractor.is_static` draws that line.
        """
        if not self.feat_extractor.is_static:
            with torch.no_grad():
                return [
                    self._get_feats(x, **kw)
                    for x, kw in zip(latents, conditioning, strict=True)
                ]

        # The buffer is append-only, so a cached prefix stays valid. Guard on the
        # per-batch sizes anyway: a caller that replaced rather than appended
        # would otherwise silently fit the surrogate to stale features.
        cached = self._feature_cache
        if len(cached) > len(latents) or any(
            cached[i].shape[0] != len(latents[i]) for i in range(len(cached))
        ):
            cached = []

        with torch.no_grad():
            for i in range(len(cached), len(latents)):
                cached.append(self._get_feats(latents[i], **conditioning[i]))

        self._feature_cache = cached
        return cached

    def set_data(
        self,
        latents: list[D],
        labels: list[torch.Tensor],
        conditioning: list[dict[str, Any]],
    ):
        """Refit the surrogate on the full observation set `D_t`.

        Parameters
        ----------
        latents : list[D]
            Latents per collected batch, in iteration order.
        labels : list[torch.Tensor]
            Labels per batch — verifier outcomes in the task-agnostic case.
        conditioning : list[dict[str, Any]]
            Conditioning used to obtain each batch of latents.
        """
        feats = self._extract_all(latents, conditioning)

        feats_tensor = torch.cat(feats, dim=0)
        labels_tensor = torch.cat(labels, dim=0)

        # Recorded before scaling because it is the thing that goes degenerate:
        # while the verifier keeps returning the same answer for every sample,
        # z-scoring maps every label to exactly 0, the posterior mean is flat,
        # and any reward that multiplies by the mean vanishes. That is correct
        # (constant labels carry no information) but it is invisible unless the
        # spread is reported. `label_mean` is recorded alongside it so a caller
        # can de-normalise the surrogate's z-scored mean/LCB back into the
        # label's own units (e.g. reporting a reward-GP posterior in raw reward
        # rather than in standard deviations of `E_t`).
        #
        # `correction=0` (population std, not Bessel's): a reward buffer of
        # size 1 is a real, reachable state (ActFlow-R's E_t is the
        # verified-valid subset only, and can be a single point for several
        # iterations), and the default `correction=1` divides by `n - 1` = 0
        # there, turning one NaN into a NaN GP fit and NaN sampling weights
        # everywhere downstream.
        self.label_mean = float(labels_tensor.mean().item())
        self.label_std = float(labels_tensor.std(correction=0).item())

        labels_tensor = (labels_tensor - self.label_mean) / (self.label_std + 1e-8)
        labels_tensor = labels_tensor.to(self.device)

        self.num_observations = int(feats_tensor.shape[0])
        self._update_estimator(feats_tensor, labels_tensor)
