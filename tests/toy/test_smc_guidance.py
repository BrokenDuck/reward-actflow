"""Tests for ActFlow-R's SMC-guided collection (reward_actflow.smc_guidance)."""

import diffusiongym
import pytest
import torch

import reward_actflow.toy
import reward_actflow.trainers  # noqa: F401  (registers mixture_replay)
from reward_actflow.smc_guidance import (
    GUIDANCE_MODES,
    collect_with_guidance,
    smc_collect,
)
from reward_actflow.uncertainty import UncertaintyEstimator

DEVICE = torch.device("cpu")


class _StubEstimator(UncertaintyEstimator):
    """Surrogate whose outputs are a function of the latent, so log_potential
    can be checked without a real GP."""

    def __init__(self, mean_fn, uncertainty_fn):
        self.mean_fn = mean_fn
        self.uncertainty_fn = uncertainty_fn
        self.num_observations = 1

    def _init_estimator(self): ...

    def _update_estimator(self, feats, labels): ...

    def _mean_and_uncertainty(self, feats): ...

    def mean_and_uncertainty(self, latent, **conditioning):
        x = latent.data
        return self.mean_fn(x), self.uncertainty_fn(x)


def _setup(**algorithm_kwargs):
    return diffusiongym.make(
        modality="actflow/toy",
        reward="actflow/acquisition",
        algorithm="mixture_replay",
        discretization_steps=8,
        device=DEVICE,
        modality_kwargs={"pretrain_steps": 0},
        algorithm_kwargs=algorithm_kwargs,
    )


def _bind_favoring_positive_x(setup):
    """zeta=0, reward surrogate mu_r(x) = 8*x[:,0] (favours +x), sigma_r=0."""
    reward = setup.environment.reward
    reward.bind(
        estimator=_StubEstimator(
            mean_fn=lambda x: torch.zeros(x.shape[0]),
            uncertainty_fn=lambda x: torch.zeros(x.shape[0]),
        ),
        verifier=None,
    )
    reward.bind_reward_surrogate(
        estimator=_StubEstimator(
            mean_fn=lambda x: x[:, 0] * 8.0,
            uncertainty_fn=lambda x: torch.zeros(x.shape[0]),
        )
    )
    reward.set_zeta(0.0)
    return reward


def test_guidance_modes_constant():
    assert set(GUIDANCE_MODES) == {"none", "smc"}


def test_unknown_guidance_raises():
    setup = _setup()
    _bind_favoring_positive_x(setup)
    with pytest.raises(ValueError, match="Unknown guidance"):
        collect_with_guidance(
            guidance="bogus",
            context=setup.context,
            algorithm=setup.algorithm,
            dynamics=setup.dynamics,
            n=4,
            time_grid=setup.time_grid,
            conditioning={},
        )


def test_guidance_none_delegates_to_the_algorithms_own_collect():
    setup = _setup()
    _bind_favoring_positive_x(setup)
    gen1 = torch.Generator().manual_seed(0)
    gen2 = torch.Generator().manual_seed(0)

    via_dispatch = collect_with_guidance(
        guidance="none",
        context=setup.context,
        algorithm=setup.algorithm,
        dynamics=setup.dynamics,
        n=8,
        time_grid=setup.time_grid,
        conditioning={},
        generator=gen1,
    )
    direct = setup.algorithm.collect(
        context=setup.context,
        dynamics=setup.dynamics,
        n=8,
        time_grid=setup.time_grid,
        conditioning={},
        generator=gen2,
    )
    assert torch.allclose(via_dispatch.latent.data, direct.latent.data)


def test_smc_requires_an_acquisition_reward():
    setup = diffusiongym.make(
        modality="actflow/toy",
        reward="actflow/uncertainty",
        algorithm="mixture_replay",
        discretization_steps=8,
        device=DEVICE,
        modality_kwargs={"pretrain_steps": 0},
    )
    with pytest.raises(TypeError, match="ActFlowRAcquisitionReward"):
        smc_collect(
            context=setup.context,
            dynamics=setup.dynamics,
            acq_beta=1.0,
            n=4,
            time_grid=setup.time_grid,
            conditioning={},
        )


def test_smc_requires_a_positive_beta():
    setup = _setup()
    _bind_favoring_positive_x(setup)
    with pytest.raises(ValueError, match="acq_beta"):
        smc_collect(
            context=setup.context,
            dynamics=setup.dynamics,
            acq_beta=0.0,
            n=4,
            time_grid=setup.time_grid,
            conditioning={},
        )


def test_guidance_smc_shifts_the_collected_batch_toward_high_acquisition():
    """The whole point: with an acquisition favouring +x, SMC-guided
    collection should measurably pull the batch positive relative to a plain
    (unguided) rollout of the same policy."""
    n = 128
    long_grid = torch.linspace(0.0, 1.0, 21)

    setup_smc = _setup()
    _bind_favoring_positive_x(setup_smc)
    smc_experience = collect_with_guidance(
        guidance="smc",
        context=setup_smc.context,
        algorithm=setup_smc.algorithm,
        dynamics=setup_smc.dynamics,
        n=n,
        time_grid=long_grid,
        conditioning={},
        acq_beta=1.0,
        generator=torch.Generator().manual_seed(0),
    )

    setup_plain = _setup()
    _bind_favoring_positive_x(setup_plain)
    plain_experience = collect_with_guidance(
        guidance="none",
        context=setup_plain.context,
        algorithm=setup_plain.algorithm,
        dynamics=setup_plain.dynamics,
        n=n,
        time_grid=long_grid,
        conditioning={},
        generator=torch.Generator().manual_seed(0),
    )

    smc_mean_x = smc_experience.latent.data[:, 0].mean().item()
    plain_mean_x = plain_experience.latent.data[:, 0].mean().item()
    assert smc_mean_x > plain_mean_x + 0.3, (smc_mean_x, plain_mean_x)


def test_smc_experience_has_no_replay_source_yet():
    """`replay` is filled in later, by ActFlowRLoop.prepare_experience()."""
    setup = _setup()
    _bind_favoring_positive_x(setup)
    experience = smc_collect(
        context=setup.context,
        dynamics=setup.dynamics,
        acq_beta=1.0,
        n=4,
        time_grid=setup.time_grid,
        conditioning={},
    )
    assert experience.replay is None
