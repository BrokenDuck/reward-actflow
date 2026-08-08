"""Tests for ActFlow-R's UpdateFlow: MixtureReplay (Algorithm 2), driven
directly by hand-built `ReplaySource`s rather than through `ActFlowRLoop`.
"""

import diffusiongym
import pytest
import torch
from diffusiongym.types import DDTensor

import reward_actflow.toy
import reward_actflow.trainers  # noqa: F401  (registers mixture_replay)
from reward_actflow.trainers.mixture_replay import (
    MixtureExperience,
    ReplaySource,
)

DEVICE = torch.device("cpu")


def _setup(**algorithm_kwargs):
    return diffusiongym.make(
        modality="actflow/toy",
        reward="actflow/uncertainty",
        algorithm="mixture_replay",
        discretization_steps=8,
        device=DEVICE,
        modality_kwargs={"pretrain_steps": 0},
        algorithm_kwargs=algorithm_kwargs,
    )


def _replay(*, n_anchors=32, n_data=32, weights=None, seed=0):
    gen = torch.Generator().manual_seed(seed)
    anchors = DDTensor(torch.randn(n_anchors, 2, generator=gen))
    data = DDTensor(torch.randn(n_data, 2, generator=gen))
    w = weights if weights is not None else torch.ones(n_data)
    return ReplaySource(
        anchors=anchors,
        anchor_conditioning={},
        data=data,
        data_conditioning={},
        weights=w,
    )


def _experience(replay=None):
    return MixtureExperience(
        latent=DDTensor(torch.zeros(1, 2)),
        rewards=torch.zeros(1),
        valid=None,
        conditioning={},
        replay=replay,
    )


def test_registered_as_mixture_replay():
    assert "mixture_replay" in diffusiongym.algorithm_registry


def test_update_refuses_a_missing_replay_source():
    setup = _setup()
    with pytest.raises(ValueError, match="ReplaySource"):
        setup.algorithm.update(
            context=setup.context, experience=_experience(replay=None)
        )


def test_anchor_fraction_is_respected():
    """lambda is the support guarantee (p >= lambda*p_P); if the observed
    split diverges from anchor_frac, that guarantee is not what's actually
    happening at runtime."""
    setup = _setup(anchor_frac=0.3, steps_per_update=20, batch_size=50)
    metrics = setup.algorithm.update(
        context=setup.context, experience=_experience(_replay())
    )
    assert metrics["anchor_frac_actual"] == pytest.approx(0.3, abs=0.02)


def test_lambda_one_never_touches_the_data_buffer():
    setup = _setup(anchor_frac=1.0, steps_per_update=5, batch_size=16)
    metrics = setup.algorithm.update(
        context=setup.context, experience=_experience(_replay())
    )
    assert metrics["anchor_frac_actual"] == pytest.approx(1.0)


def test_lambda_zero_and_unit_weights_reduce_to_a_plain_refit():
    """The spec's own reduction: anchor_frac=0 never touches P, and uniform
    w means the observed sampling distribution over D+ is uniform too —
    indistinguishable from a plain flow-matching refit."""
    setup = _setup(anchor_frac=0.0, steps_per_update=5, batch_size=16)
    replay = _replay(weights=torch.ones(32))
    metrics = setup.algorithm.update(
        context=setup.context, experience=_experience(replay)
    )
    assert metrics["anchor_frac_actual"] == pytest.approx(0.0)
    assert metrics["ess_frac_postclip"] == pytest.approx(1.0, abs=1e-4)


def test_empty_data_buffer_falls_back_to_anchors_only():
    """D+ empty at t=0 must not crash — training proceeds on the anchor
    fraction that is available, not on nothing."""
    setup = _setup(anchor_frac=0.5, steps_per_update=5, batch_size=16)
    replay = _replay(n_data=0, weights=torch.zeros(0))
    metrics = setup.algorithm.update(
        context=setup.context, experience=_experience(replay)
    )
    assert metrics["steps_skipped"] == 0.0
    assert metrics["anchor_frac_actual"] == pytest.approx(1.0)


def test_empty_everything_skips_every_step_without_crashing():
    setup = _setup(anchor_frac=0.0, steps_per_update=5, batch_size=16)
    replay = _replay(n_data=0, weights=torch.zeros(0))
    metrics = setup.algorithm.update(
        context=setup.context, experience=_experience(replay)
    )
    assert metrics["steps_skipped"] == 5.0
    assert metrics["loss"] == 0.0


def test_weight_clip_bounds_the_proposal():
    weights = torch.tensor([1.0] * 31 + [1000.0])
    setup = _setup(
        anchor_frac=0.0, weight_clip=2.0, steps_per_update=1, batch_size=1000
    )
    replay = _replay(weights=weights)
    metrics = setup.algorithm.update(
        context=setup.context, experience=_experience(replay)
    )
    assert metrics["w_clip_fraction"] == pytest.approx(1.0 / 32)


def test_synchronize_rollout_policy_copies_train_into_rollout():
    setup = _setup()
    with torch.no_grad():
        for p in setup.context.policies.train.parameters():
            p.add_(1.0)
    setup.algorithm.synchronize_rollout_policy(context=setup.context)
    train = setup.context.policies.train.state_dict()
    rollout = setup.context.policies.rollout.state_dict()
    for key in train:
        assert torch.equal(train[key], rollout[key])


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"anchor_frac": -0.1}, "anchor_frac"),
        ({"anchor_frac": 1.1}, "anchor_frac"),
        ({"weight_clip": 0.5}, "weight_clip"),
        ({"steps_per_update": 0}, "steps_per_update"),
        ({"batch_size": 0}, "batch_size"),
    ],
)
def test_constructor_validates_its_arguments(kwargs, message):
    from reward_actflow.trainers.mixture_replay import MixtureReplay

    with pytest.raises(ValueError, match=message):
        MixtureReplay(**kwargs)


def test_requires_stochastic_rollout():
    from reward_actflow.trainers.mixture_replay import MixtureReplay

    assert MixtureReplay().requirements.needs_stochastic_rollout is True
