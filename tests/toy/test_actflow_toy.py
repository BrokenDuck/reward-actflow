"""End-to-end tests for the ActFlow toy problem on the diffusiongym core stack."""

import logging
from dataclasses import dataclass
from typing import Any, ClassVar

import diffusiongym
import pytest
import torch

import reward_actflow.toy  # noqa: F401  (registers actflow/toy and actflow/uncertainty)
from reward_actflow.explore import (
    ALGORITHM_DEFAULTS,
    ActFlowLoop,
    ExploreConfig,
    build_setup,
    refresh_reference,
)
from reward_actflow.rewards.uncertainty import (
    ActFlowUncertaintyReward,
    SoftGateTerminalCost,
)
from reward_actflow.setups.toy import ToyProblemSetup
from reward_actflow.toy.validity import staircase_validity
from reward_actflow.uncertainty import GPUncertaintyEstimator, UncertaintyEstimator

DEVICE = torch.device("cpu")

BLACK_BOX_GATES = ("hard", "mult", "validity")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _args(**overrides) -> dict:
    base = {
        "uncertainty_estimator": "gp",
        "gp_kernel": "rbf",
        "gp_lengthscale": 0.5,
        "toy_pretrain_steps": 0,
        "toy_checkpoint": None,
        "toy_width": 16,
        "toy_depth": 1,
    }
    return base | overrides


def _config(tmp_path, **overrides) -> ExploreConfig:
    defaults = {
        "dir": tmp_path,
        "num_iters": 3,
        "samples_per_iter": 16,
        "num_steps": 8,
        "gate": "hard",
        "algorithm": "diffusion_nft",
    }
    return ExploreConfig.construct_from_args(_args(**(defaults | overrides)))


def _build(tmp_path, **overrides):
    args = _args(**overrides)
    config = _config(tmp_path, **overrides)
    problem = ToyProblemSetup(args, device=DEVICE)
    setup, uncertainty = build_setup(problem, config, args, DEVICE)
    return problem, setup, uncertainty, config


class _StubEstimator(UncertaintyEstimator):
    """Surrogate with hand-chosen outputs, so gate behaviour is checkable."""

    def __init__(self, mean, uncertainty):
        self.mean = mean
        self.uncertainty = uncertainty
        self.num_observations = 1

    def _init_estimator(self): ...

    def _update_estimator(self, feats, labels): ...

    def _mean_and_uncertainty(self, feats):
        return self.mean, self.uncertainty

    def mean_and_uncertainty(self, latent, **conditioning):
        return self.mean, self.uncertainty


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


def test_staircase_validity_covers_the_three_rectangles():
    # x is offset by +0.5 inside the verifier, so these are raw coordinates.
    points = torch.tensor(
        [
            [-1.0, 1.5],  # base blob, inside `top`
            [-2.5, 0.0],  # inside the `middle` corridor
            [2.0, -1.5],  # far right of `bottom`
            [0.0, 0.0],  # between top and bottom, outside the corridor
            [3.5, 1.5],  # right of `top`
        ]
    )
    expected = torch.tensor([True, True, True, False, False])
    assert torch.equal(staircase_validity(points), expected)


# ---------------------------------------------------------------------------
# Reward gates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gate", ["hard", "mult", "soft", "sigmoid"])
def test_gate_ranks_valid_uncertain_above_invalid_uncertain(gate):
    """The whole point of gating: equal uncertainty must not mean equal reward."""
    uncertainty = torch.tensor([1.0, 1.0, 0.1, 0.1])
    mean = torch.tensor([1.0, -1.0, 1.0, -1.0])  # confident valid / invalid
    labels = torch.tensor([True, False, True, False])

    reward = ActFlowUncertaintyReward(gate=gate, invalid_floor=1.0)
    reward.bind(
        estimator=_StubEstimator(mean, uncertainty),
        verifier=lambda sample, cond: labels,
    )

    r = reward.score(sample=None, latent=torch.zeros(4, 2), conditioning={})

    # valid-and-uncertain beats invalid-and-uncertain ...
    assert r[0] > r[1]
    # ... and also beats valid-but-already-known.
    assert r[0] > r[2]


def test_raw_gate_cannot_tell_valid_from_invalid():
    """The `--no_verifier` ablation is supposed to be blind to validity."""
    uncertainty = torch.tensor([1.0, 1.0])
    reward = ActFlowUncertaintyReward(gate="raw")
    reward.bind(
        estimator=_StubEstimator(torch.tensor([1.0, -1.0]), uncertainty),
        verifier=lambda sample, cond: torch.tensor([True, False]),
    )

    r = reward.score(sample=None, latent=torch.zeros(2, 2), conditioning={})
    assert torch.equal(r[0], r[1])


def test_validity_gate_ignores_uncertainty():
    """The `--no_uncertainty` ablation is supposed to be blind to sigma."""
    reward = ActFlowUncertaintyReward(gate="validity", invalid_floor=1.0)
    reward.bind(
        estimator=_StubEstimator(torch.zeros(2), torch.tensor([5.0, 0.01])),
        verifier=lambda sample, cond: torch.tensor([True, True]),
    )

    r = reward.score(sample=None, latent=torch.zeros(2, 2), conditioning={})
    assert torch.equal(r[0], r[1])


def test_reward_raises_before_bind():
    reward = ActFlowUncertaintyReward(gate="hard")
    assert not reward.is_bound
    with pytest.raises(RuntimeError, match="before bind"):
        reward.score(sample=None, latent=torch.zeros(2, 2), conditioning={})


def test_unknown_gate_is_rejected():
    with pytest.raises(ValueError, match="Unknown gate"):
        ActFlowUncertaintyReward(gate="nonsense")  # ty: ignore[invalid-argument-type]


def test_terminal_cost_negates_the_reward_and_is_differentiable():
    """Adjoint Matching minimizes this, so the sign is load-bearing."""
    latent = torch.tensor([[0.3, 0.4], [1.0, -0.5]], requires_grad=True)

    reward = ActFlowUncertaintyReward(gate="soft")
    reward.bind(
        estimator=_StubEstimator(latent[:, 0], latent[:, 1].abs()),
        verifier=lambda sample, cond: torch.ones(2, dtype=torch.bool),
    )

    cost = SoftGateTerminalCost(reward)(latent, conditioning={})
    r = reward.score(sample=None, latent=latent, conditioning={})

    assert torch.allclose(cost, -r)
    grad = torch.autograd.grad(cost.sum(), latent)[0]
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


def test_soft_gate_collapses_on_uninformative_labels_but_sigmoid_does_not():
    """While every verifier answer agrees, z-scored labels make `mu` flat.

    `soft` then scores every sample exactly 0 — a dead objective that Adjoint
    Matching rejects — whereas `sigmoid` falls back to `sigma / 2` and keeps the
    uncertainty signal.
    """
    uncertainty = torch.tensor([1.0, 0.25])
    flat_mean = torch.zeros(2)

    soft = ActFlowUncertaintyReward(gate="soft")
    soft.bind(
        estimator=_StubEstimator(flat_mean, uncertainty),
        verifier=lambda sample, cond: torch.ones(2, dtype=torch.bool),
    )
    r_soft = soft.score(sample=None, latent=torch.zeros(2, 2), conditioning={})
    assert torch.equal(r_soft, torch.zeros(2))

    smooth = ActFlowUncertaintyReward(gate="sigmoid")
    smooth.bind(
        estimator=_StubEstimator(flat_mean, uncertainty),
        verifier=lambda sample, cond: torch.ones(2, dtype=torch.bool),
    )
    r_sigmoid = smooth.score(sample=None, latent=torch.zeros(2, 2), conditioning={})
    assert torch.allclose(r_sigmoid, uncertainty / 2)
    assert r_sigmoid[0] > r_sigmoid[1]


def test_label_std_reports_an_uninformative_verifier(tmp_path):
    """The collapse above is invisible unless the label spread is logged."""
    from diffusiongym.types import DDTensor

    args = _args()
    config = _config(tmp_path)
    problem = ToyProblemSetup(args, device=DEVICE)
    _, uncertainty = build_setup(problem, config, args, DEVICE)

    observed = DDTensor(torch.randn(8, 2))
    uncertainty.set_data([observed], [torch.ones(8)], [{}])
    assert uncertainty.label_std == 0.0

    uncertainty.set_data([observed], [torch.tensor([1.0] * 4 + [0.0] * 4)], [{}])
    assert uncertainty.label_std > 0.0


def test_label_mean_tracks_the_raw_label_scale(tmp_path):
    """`label_mean` de-normalises a z-scored posterior back to raw label units.

    Recorded before z-scoring in `set_data`, same as `label_std` — without it,
    reporting a reward-GP's mean/LCB in the reward's own units would need the
    caller to re-derive the normaliser it was fit against.
    """
    from diffusiongym.types import DDTensor

    args = _args()
    config = _config(tmp_path)
    problem = ToyProblemSetup(args, device=DEVICE)
    _, uncertainty = build_setup(problem, config, args, DEVICE)

    observed = DDTensor(torch.randn(8, 2))
    uncertainty.set_data([observed], [torch.full((8,), 5.0)], [{}])
    assert uncertainty.label_mean == pytest.approx(5.0)

    uncertainty.set_data([observed], [torch.tensor([1.0, 3.0] * 4)], [{}])
    assert uncertainty.label_mean == pytest.approx(2.0)


def test_gp_uncertainty_estimator_reports_standard_deviation(tmp_path):
    """The GP used to return posterior *variance*; every estimator now reports
    standard deviation so beta/eta acquisition coefficients are on the same
    scale as the ensemble backend and transfer between the two."""
    from diffusiongym.types import DDTensor

    args = _args(gp_lengthscale=0.2)
    config = _config(tmp_path)
    problem = ToyProblemSetup(args, device=DEVICE)
    _, uncertainty = build_setup(problem, config, args, DEVICE)
    assert isinstance(uncertainty, GPUncertaintyEstimator)

    observed = DDTensor(torch.zeros(8, 2))
    uncertainty.set_data([observed], [torch.arange(8, dtype=torch.float)], [{}])

    _, sigma = uncertainty.mean_and_uncertainty(DDTensor(torch.full((4, 2), 10.0)))
    # A posterior variance well above 1 is expected here (unit-scale RBF prior
    # plus likelihood noise); its square root should not be.
    assert (sigma < 3.0).all(), sigma


def test_terminal_cost_refuses_a_black_box_gate():
    reward = ActFlowUncertaintyReward(gate="hard")
    with pytest.raises(ValueError, match="no gradient"):
        SoftGateTerminalCost(reward)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def test_providers_are_registered():
    assert "actflow/toy" in diffusiongym.modality_registry
    assert "actflow/uncertainty" in diffusiongym.reward_provider_registry


@pytest.mark.parametrize("algorithm", ALGORITHM_DEFAULTS)
def test_every_algorithm_assembles_with_a_differentiable_gate(tmp_path, algorithm):
    extra = {"warmup_iters": 1} if algorithm == "adjoint_matching" else {}
    _, setup, _, _ = _build(tmp_path, algorithm=algorithm, gate="sigmoid", **extra)
    assert setup.algorithm is not None
    # `make()` only builds a reference policy where the requirements ask for one.
    needs_reference = setup.algorithm.requirements.needs_reference_policy
    assert (setup.context.policies.reference is not None) == needs_reference


@pytest.mark.parametrize("gate", BLACK_BOX_GATES)
def test_adjoint_matching_is_refused_for_black_box_gates(tmp_path, gate):
    """A verifier call has no gradient; AM must fail at assembly, not silently."""
    with pytest.raises(ValueError, match="differentiable terminal cost"):
        _build(tmp_path, algorithm="adjoint_matching", gate=gate, warmup_iters=1)


def test_adjoint_matching_requires_a_warmup(tmp_path):
    with pytest.raises(ValueError, match="warmup_iters"):
        _config(tmp_path, algorithm="adjoint_matching", gate="sigmoid")


def test_adjoint_matching_refuses_the_collapsing_soft_gate(tmp_path):
    """`soft` is zero everywhere until the verifier disagrees with itself."""
    with pytest.raises(ValueError, match="gate sigmoid"):
        _config(tmp_path, algorithm="adjoint_matching", gate="soft", warmup_iters=1)


def test_policies_start_identical(tmp_path):
    """Pretraining per `model()` call would desync train/rollout/reference."""
    _, setup, _, _ = _build(tmp_path, algorithm="flow_grpo", gate="hard")
    train = setup.context.policies.train.state_dict()
    rollout = setup.context.policies.rollout.state_dict()
    reference = setup.context.policies.reference.state_dict()

    for key in train:
        assert torch.equal(train[key], rollout[key])
        assert torch.equal(train[key], reference[key])


def test_ablation_flags_resolve_to_gates(tmp_path):
    assert _config(tmp_path, no_verifier=True).gate == "raw"
    assert _config(tmp_path, no_uncertainty=True).gate == "validity"
    # Explicit gates survive when neither flag is set.
    assert _config(tmp_path, gate="soft").gate == "soft"


def test_algorithm_kwargs_override_the_fused_loop_defaults(tmp_path):
    config = _config(tmp_path, algorithm_kwargs={"ema_decay": 0.5})
    assert config.algorithm_kwargs["ema_decay"] == 0.5
    assert (
        config.algorithm_kwargs["inner_epochs"]
        == (ALGORITHM_DEFAULTS["diffusion_nft"]["inner_epochs"])
    )


def test_construct_from_args_returns_the_subclass_type(tmp_path):
    """`construct_from_args` is a `@classmethod`, not a `@staticmethod`, so a
    config subclass (e.g. `ActFlowRConfig`) builds an instance of itself —
    with its own extra fields — rather than a plain `ExploreConfig`."""

    @dataclass(frozen=True)
    class _ExtendedConfig(ExploreConfig):
        extra: int = 7

    result = _ExtendedConfig.construct_from_args(_args(dir=tmp_path, extra=9))
    assert isinstance(result, _ExtendedConfig)
    assert result.extra == 9


def test_allowed_algorithms_is_extendable_by_a_subclass(tmp_path):
    """`ALLOWED_ALGORITHMS` is a `ClassVar`, so a subclass can widen the set
    `__post_init__` accepts without `ExploreConfig` itself accepting it."""

    @dataclass(frozen=True)
    class _ExtendedConfig(ExploreConfig):
        ALLOWED_ALGORITHMS: ClassVar[tuple[str, ...]] = (
            *ALGORITHM_DEFAULTS,
            "made_up_algorithm",
        )

    with pytest.raises(ValueError, match="algorithm must be one of"):
        _config(tmp_path, algorithm="made_up_algorithm")

    _ExtendedConfig.construct_from_args(
        _args(dir=tmp_path, algorithm="made_up_algorithm")
    )  # does not raise


# ---------------------------------------------------------------------------
# The reference anchor
# ---------------------------------------------------------------------------


def test_refresh_reference_re_anchors_to_the_current_iterate(tmp_path):
    _, setup, _, _ = _build(tmp_path, algorithm="flow_grpo", gate="hard")
    context = setup.context

    with torch.no_grad():
        for p in context.policies.train.parameters():
            p.add_(1.0)

    train = context.policies.train.state_dict()
    reference = context.policies.reference.state_dict()
    assert any(not torch.equal(train[k], reference[k]) for k in train)

    assert refresh_reference(context) is True

    reference = context.policies.reference.state_dict()
    for key in train:
        assert torch.equal(train[key], reference[key])


def test_refresh_reference_is_a_no_op_without_a_reference(tmp_path):
    _, setup, _, _ = _build(tmp_path, algorithm="diffusion_nft", gate="hard")
    assert setup.context.policies.reference is None
    assert refresh_reference(setup.context) is False


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def _run_loop(tmp_path, **overrides):
    args = _args(**overrides)
    config = _config(tmp_path, **overrides)
    problem = ToyProblemSetup(args, device=DEVICE)
    setup, uncertainty = build_setup(problem, config, args, DEVICE)
    loop = ActFlowLoop(
        problem=problem,
        setup=setup,
        uncertainty=uncertainty,
        config=config,
        logger=logging.getLogger("test"),
    )
    loop.run()
    return loop


@pytest.mark.parametrize("gate", ["hard", "mult", "soft", "sigmoid"])
def test_loop_runs_and_accumulates_observations(tmp_path, gate):
    loop = _run_loop(tmp_path, gate=gate)

    assert len(loop.observations) == 3
    assert all(len(b) == 16 for b in loop.observations)
    # The surrogate is deliberately one batch behind: Algorithm 1 fits sigma_t on
    # D_t and only then draws x_{t+1}, so the batch collected in the final
    # iteration has not been folded in yet.
    assert loop.uncertainty.num_observations == 32

    for batch in loop.observations:
        assert batch.valids.dtype == torch.bool
        assert batch.rewards.shape == (16,)
        assert torch.isfinite(batch.rewards).all()

    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "timings.csv").exists()
    assert len(list((tmp_path / "frames").glob("*.png"))) == 3
    assert (tmp_path / "checkpoints" / "last.pt").exists()


def test_visualize_every_throttles_figures(tmp_path):
    """The figure is the expensive part of a long run, so it must be skippable."""
    loop = _run_loop(tmp_path, num_iters=4, visualize_every=2)
    assert len(loop.observations) == 4
    assert sorted(p.stem for p in (tmp_path / "frames").glob("*.png")) == [
        "0000",
        "0002",
    ]


def test_cold_start_leaves_the_policy_untouched(tmp_path):
    """`D_0` is empty, so sigma is flat and every sample scores identically.

    DiffusionNFT's positive and negative branches then cancel exactly, which is
    the right behaviour: with nothing observed there is nothing to prefer, and
    the first iteration should not push the policy anywhere.

    Uses the `raw` gate because it is the one whose cold-start reward really is
    constant. A verifier-consulting gate still separates valid from invalid on
    iteration 0 and moves the policy — correctly so, since the verifier is
    informative from the very first query even when the surrogate is not.
    """
    args = _args(gate="raw")
    config = _config(tmp_path, gate="raw", num_iters=1)
    problem = ToyProblemSetup(args, device=DEVICE)
    setup, uncertainty = build_setup(problem, config, args, DEVICE)

    train_policy: Any = setup.context.policies.train
    before = {k: v.clone() for k, v in train_policy.state_dict().items()}

    loop = ActFlowLoop(
        problem=problem,
        setup=setup,
        uncertainty=uncertainty,
        config=config,
        logger=logging.getLogger("test"),
    )
    loop.run()

    after = train_policy.state_dict()
    for key, value in before.items():
        assert torch.allclose(value, after[key], atol=1e-7), key


def test_surrogate_sees_every_earlier_batch_before_scoring(tmp_path):
    """Line 3 refits on all of `D_t` before line 4 draws `x_{t+1}`.

    Two iterations means one refit that saw nothing (`D_0` is empty) and one
    that saw the first batch, so the final surrogate is conditioned on exactly
    `samples_per_iter` points.
    """
    loop = _run_loop(tmp_path, gate="hard", num_iters=2)
    assert loop.uncertainty.num_observations == 16


def test_hooks_fire_in_algorithm_1_order(tmp_path):
    """The four extension hooks exist so `ActFlowRLoop` can add ActFlow-R's
    extra steps without overriding `run()`; pin their order here so a future
    edit to `run()` cannot silently reshuffle them. No-op for base `ActFlowLoop`
    (see `test_loop_runs_and_accumulates_observations`), so this only tests
    that the hooks are called, and in the right place."""
    events: list[str] = []

    class _RecordingLoop(ActFlowLoop):
        def pre_collect(self, iteration):
            events.append(f"pre_collect:{iteration}")
            return {}

        def post_collect(self, iteration, batch):
            events.append(f"post_collect:{iteration}")
            return {}

        def prepare_experience(self, experience):
            events.append("prepare_experience")
            return experience

        def extra_metrics(self, iteration, batch):
            events.append(f"extra_metrics:{iteration}")
            return {}

    args = _args()
    config = _config(tmp_path, num_iters=2, gate="hard")
    problem = ToyProblemSetup(args, device=DEVICE)
    setup, uncertainty = build_setup(problem, config, args, DEVICE)
    loop = _RecordingLoop(
        problem=problem,
        setup=setup,
        uncertainty=uncertainty,
        config=config,
        logger=logging.getLogger("test"),
    )
    loop.run()

    assert events == [
        "pre_collect:0",
        "post_collect:0",
        "prepare_experience",
        "extra_metrics:0",
        "pre_collect:1",
        "post_collect:1",
        "prepare_experience",
        "extra_metrics:1",
    ]


@pytest.mark.parametrize("backend", ["exact", "inducing", "grid"])
def test_every_backend_keeps_uncertainty_high_away_from_data(tmp_path, backend):
    """The exploration signal is "high sigma = unexplored"; scaling must keep it.

    Both approximations could plausibly break this — inducing-point methods by
    underestimating variance away from their inducing points, the grid by
    interpolating across it — so it is checked for all three.
    """
    from diffusiongym.types import DDTensor

    args = _args(
        gp_backend=backend, gp_inducing=64, gp_grid_size=64, gp_lengthscale=0.2
    )
    config = _config(tmp_path)
    problem = ToyProblemSetup(args, device=DEVICE)
    _, uncertainty = build_setup(problem, config, args, DEVICE)
    assert isinstance(uncertainty, GPUncertaintyEstimator)

    # Everything observed sits in one corner, as it does early in a real run.
    observed = DDTensor(torch.rand(400, 2) * 0.5 - 1.0)
    uncertainty.set_data([observed], [torch.randint(0, 2, (400,)).float()], [{}])

    _, near = uncertainty.mean_and_uncertainty(DDTensor(torch.full((1, 2), -0.75)))
    _, far = uncertainty.mean_and_uncertainty(DDTensor(torch.full((1, 2), 3.0)))
    assert far.item() > near.item()


def test_grid_backend_tolerates_samples_outside_its_grid(tmp_path):
    """SKI raises on out-of-bounds input; the policy can generate anywhere."""
    from diffusiongym.types import DDTensor

    args = _args(gp_backend="grid", gp_grid_size=32, gp_grid_limit=4.0)
    config = _config(tmp_path)
    problem = ToyProblemSetup(args, device=DEVICE)
    _, uncertainty = build_setup(problem, config, args, DEVICE)

    uncertainty.set_data([DDTensor(torch.randn(32, 2))], [torch.zeros(32)], [{}])
    _, far = uncertainty.mean_and_uncertainty(DDTensor(torch.full((2, 2), 500.0)))
    assert torch.isfinite(far).all()


def test_grid_backend_refuses_high_dimensional_features(tmp_path):
    """The grid holds grid_size ** feat_dim points; molecules would not fit."""
    args = _args(gp_backend="grid")
    config = _config(tmp_path)
    problem = ToyProblemSetup(args, device=DEVICE)
    _, uncertainty = build_setup(problem, config, args, DEVICE)

    with pytest.raises(ValueError, match="gp_backend inducing"):
        type(uncertainty)(uncertainty.feat_extractor, feat_dim=64, args=args)


def test_feature_cache_only_extends_and_only_when_the_model_cannot_move_it(tmp_path):
    """Caching is sound for `"input"` features and unsound for a hooked layer."""
    from diffusiongym.types import DDTensor

    args = _args()
    config = _config(tmp_path)
    problem = ToyProblemSetup(args, device=DEVICE)
    _, uncertainty = build_setup(problem, config, args, DEVICE)
    assert uncertainty.feat_extractor.is_static  # toy uses feature_layer="input"

    first = DDTensor(torch.randn(8, 2))
    second = DDTensor(torch.randn(8, 2))
    uncertainty.set_data([first], [torch.zeros(8)], [{}])
    cached_first = uncertainty._feature_cache[0]

    uncertainty.set_data([first, second], [torch.zeros(8)] * 2, [{}] * 2)
    assert len(uncertainty._feature_cache) == 2
    # The prefix is reused, not recomputed.
    assert uncertainty._feature_cache[0] is cached_first

    # A caller that replaces rather than appends invalidates the cache.
    replacement = DDTensor(torch.randn(4, 2))
    uncertainty.set_data([replacement], [torch.zeros(4)], [{}])
    assert len(uncertainty._feature_cache) == 1
    assert uncertainty._feature_cache[0].shape[0] == 4


def test_gp_estimator_reports_variance_at_observed_points(tmp_path):
    """A fitted GP should be less uncertain where it has data than where it has none."""
    args = _args(gp_lengthscale=0.2)
    config = _config(tmp_path)
    problem = ToyProblemSetup(args, device=DEVICE)
    _, uncertainty = build_setup(problem, config, args, DEVICE)
    assert isinstance(uncertainty, GPUncertaintyEstimator)

    from diffusiongym.types import DDTensor

    observed = DDTensor(torch.zeros(8, 2))
    uncertainty.set_data([observed], [torch.arange(8, dtype=torch.float)], [{}])

    _, near = uncertainty.mean_and_uncertainty(DDTensor(torch.zeros(1, 2)))
    _, far = uncertainty.mean_and_uncertainty(DDTensor(torch.full((1, 2), 10.0)))
    assert near.item() < far.item()
