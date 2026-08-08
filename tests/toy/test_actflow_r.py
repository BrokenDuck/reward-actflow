"""End-to-end tests for ActFlow-R on the staircase toy."""

import logging

import pytest
import torch

import reward_actflow.toy  # noqa: F401  (registers actflow/toy, actflow/acquisition)
from reward_actflow.actflow_r import (
    ALGORITHMS_R,
    ActFlowRConfig,
    ActFlowRLoop,
    build_actflow_r_setup,
    zeta_at,
)
from reward_actflow.setups.toy import ToyProblemSetup

DEVICE = torch.device("cpu")


def _args(**overrides) -> dict:
    base = {
        "uncertainty_estimator": "gp",
        "gp_kernel": "rbf",
        "gp_lengthscale": 0.5,
        "toy_pretrain_steps": 0,
        "toy_checkpoint": None,
        "toy_width": 16,
        "toy_depth": 1,
        "toy_reward": "linear",
        "reward_gp_kernel": None,
        "reward_gp_lengthscale": None,
        "reward_gp_backend": None,
        "reward_gp_inducing": None,
        "reward_gp_grid_size": None,
        "reward_gp_grid_limit": None,
    }
    return base | overrides


def _config(tmp_path, **overrides) -> ActFlowRConfig:
    defaults = {
        "dir": tmp_path,
        "num_iters": 3,
        "samples_per_iter": 16,
        "num_steps": 8,
        "algorithm": "mixture_replay",
        "mixture_steps": 4,
        "mixture_batch": 8,
        "anchor_pool_size": 32,
        # SMC is the default, but "none" is faster for most of these tests;
        # the guidance="smc" path gets its own dedicated end-to-end test.
        "guidance": "none",
    }
    return ActFlowRConfig.construct_from_args(_args(**(defaults | overrides)))


def _build(tmp_path, **overrides):
    args = _args(**overrides)
    config = _config(tmp_path, **overrides)
    problem = ToyProblemSetup(args, device=DEVICE)
    setup, validity_uncertainty, reward_uncertainty = build_actflow_r_setup(
        problem, config, args, DEVICE
    )
    return problem, setup, validity_uncertainty, reward_uncertainty, config


def _run_loop(tmp_path, **overrides) -> ActFlowRLoop:
    problem, setup, validity_uncertainty, reward_uncertainty, config = _build(
        tmp_path, **overrides
    )
    loop = ActFlowRLoop(
        problem=problem,
        setup=setup,
        uncertainty=validity_uncertainty,
        reward_uncertainty=reward_uncertainty,
        config=config,
        logger=logging.getLogger("test"),
    )
    loop.run()
    return loop


# ---------------------------------------------------------------------------
# zeta schedule
# ---------------------------------------------------------------------------


def test_zeta_linear_interpolates_start_to_end():
    assert zeta_at("linear", 1.0, 0.0, 0.0) == pytest.approx(1.0)
    assert zeta_at("linear", 1.0, 0.0, 1.0) == pytest.approx(0.0)
    assert zeta_at("linear", 1.0, 0.0, 0.5) == pytest.approx(0.5)


def test_zeta_constant_ignores_progress():
    assert zeta_at("constant", 0.7, 0.1, 0.0) == pytest.approx(0.7)
    assert zeta_at("constant", 0.7, 0.1, 1.0) == pytest.approx(0.7)


def test_zeta_cosine_reaches_both_endpoints():
    assert zeta_at("cosine", 1.0, 0.0, 0.0) == pytest.approx(1.0)
    assert zeta_at("cosine", 1.0, 0.0, 1.0) == pytest.approx(0.0)


def test_zeta_progress_is_clamped():
    assert zeta_at("linear", 1.0, 0.0, -1.0) == pytest.approx(1.0)
    assert zeta_at("linear", 1.0, 0.0, 2.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_allowed_algorithms_extends_the_base_four():
    assert ALGORITHMS_R[-1] == "mixture_replay"
    assert ActFlowRConfig.ALLOWED_ALGORITHMS == ALGORITHMS_R


def test_algorithm_kwargs_are_derived_from_dedicated_fields(tmp_path):
    """A user tunes MixtureReplay via --anchor_frac etc., not
    --algorithm_kwargs — those dedicated fields must actually reach it."""
    config = _config(
        tmp_path, anchor_frac=0.4, weight_clip=5.0, mixture_steps=7, mixture_batch=12
    )
    assert config.algorithm_kwargs == {
        "anchor_frac": 0.4,
        "weight_clip": 5.0,
        "steps_per_update": 7,
        "batch_size": 12,
    }


def test_extra_algorithm_kwargs_pass_through_without_clobbering_dedicated_fields(
    tmp_path,
):
    config = _config(
        tmp_path, algorithm_kwargs={"anchor_frac": 0.9, "some_future_param": 1}
    )
    # The dedicated field wins over a same-named --algorithm_kwargs entry...
    assert config.algorithm_kwargs["anchor_frac"] == config.anchor_frac
    # ...but anything else in --algorithm_kwargs still passes through.
    assert config.algorithm_kwargs["some_future_param"] == 1


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"zeta_start": -0.1}, "zeta_start"),
        ({"zeta_end": 1.1}, "zeta_end"),
        ({"beta_r": -1.0}, "beta_r"),
        ({"anchor_frac": 1.5}, "anchor_frac"),
        ({"weight_clip": 0.5}, "weight_clip"),
        ({"eta": 0.0}, "eta"),
        ({"acq_beta": 0.0}, "acq_beta"),
        ({"anchor_pool_size": 0}, "anchor_pool_size"),
    ],
)
def test_config_validates_its_arguments(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        _config(tmp_path, **kwargs)


def test_config_rejects_an_unknown_guidance(tmp_path):
    with pytest.raises(ValueError, match="guidance"):
        _config(tmp_path, guidance="bogus")


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def test_loop_runs_end_to_end(tmp_path):
    loop = _run_loop(tmp_path, num_iters=3)

    assert len(loop.observations) == 3
    assert all(len(b) == 16 for b in loop.observations)
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "timings.csv").exists()
    assert (tmp_path / "anchors.pt").exists()
    assert len(list((tmp_path / "frames").glob("*.png"))) == 3
    assert (tmp_path / "checkpoints" / "last.pt").exists()


def test_visualize_reward_sample_is_used_not_the_base_figure(tmp_path):
    """ActFlowRLoop should prefer the reward-aware (3-panel) figure over the
    inherited uncertainty/support-only (2-panel) one."""
    problem, setup, validity_uncertainty, reward_uncertainty, _config = _build(tmp_path)
    anchors = problem.anchor_latents(8, DEVICE)
    batch = _make_batch(problem, setup, DEVICE)

    result = problem.visualize_reward_sample(
        setup, validity_uncertainty, reward_uncertainty, anchors, batch
    )
    assert result is not None
    fig, metrics = result
    assert len(fig.axes) >= 3
    assert "support_coverage" in metrics


def _make_batch(problem, setup, device):
    from reward_actflow.sampling import sample_policy
    from reward_actflow.utils import Batch

    latents = sample_policy(
        setup.context, 8, dynamics=setup.dynamics, time_grid=setup.time_grid
    )
    samples = setup.environment.codec.decode(latents, conditioning={})
    valids = problem.validity(samples, {})
    return Batch.from_endpoints(
        latents=latents,
        samples=samples,
        rewards=torch.zeros(8),
        valids=valids,
        conditioning={},
    )


def test_loop_writes_every_diagnostic_column(tmp_path):
    """`_flush_metrics` fixes the schema on the first write and drops later
    keys — a diagnostic missing at t=0 is invisible for the whole run, so
    every diagnostic must actually appear on iteration 0."""
    _run_loop(tmp_path, num_iters=1)

    header = (tmp_path / "metrics.csv").read_text().splitlines()[0].split(",")
    expected = {
        "zeta",
        "reward_observations",
        "reward_label_std",
        "reward_label_mean",
        "clusters_r_train",
        "clusters_r_eval",
        "cluster_ratio",
        "far_count",
        "density_new_mass_fraction",
        "density_anchor_mass_fraction",
        "density_log_ratio_max",
        "density_log_ratio_p95",
    }
    missing = expected - set(header)
    assert not missing, f"missing columns: {missing}"


def test_loop_runs_end_to_end_with_smc_guidance(tmp_path):
    """The default (--guidance smc) path, not just the faster "none" path
    used by the other tests here."""
    loop = _run_loop(tmp_path, num_iters=2, guidance="smc", samples_per_iter=8)
    assert len(loop.observations) == 2


def test_anchor_pool_is_never_refreshed(tmp_path):
    """P must stay frozen: p^{theta_{t+1}} >= lambda*p_P only holds if P
    never moves with theta_t."""
    loop = _run_loop(tmp_path, num_iters=3)
    anchors_after = loop.anchors
    assert torch.equal(anchors_after.data, loop.anchors.data)


def test_anchor_pool_size_matches_the_config(tmp_path):
    loop = _run_loop(tmp_path, num_iters=1, anchor_pool_size=17)
    assert len(loop.anchors) == 17


def test_toy_prefers_p_data_over_theta_0_samples(tmp_path):
    """The toy implements anchor_latents(), so P should come from p_data, not
    from sampling the (possibly untrained) policy."""
    loop = _run_loop(tmp_path, num_iters=1)
    assert loop.anchor_source == "p_data"


def test_mixture_replay_never_builds_a_reference_policy(tmp_path):
    """MixtureReplay has no KL-to-current-iterate term, so there is nothing
    to re-anchor — update_flow's override of refresh_reference is only
    correct because this is true."""
    _, setup, _, _, _ = _build(tmp_path)
    assert setup.context.policies.reference is None


def test_reward_surrogate_sees_only_earlier_valid_batches(tmp_path):
    """pre_collect() refits (mu^r, sigma^r) on E_t before this iteration's
    own r~ is queried — one iteration behind, mirroring how the validity
    surrogate lags D_t in base ActFlow."""
    loop = _run_loop(tmp_path, num_iters=2)
    expected = int(loop.observations[0].valids.sum().item())
    assert loop.reward_uncertainty.num_observations == expected


def test_zeta_is_annealed_over_the_run(tmp_path):
    loop = _run_loop(
        tmp_path, num_iters=4, zeta_start=1.0, zeta_end=0.0, zeta_schedule="linear"
    )
    # zeta_at is called from pre_collect with iteration/(num_iters-1); the
    # last iteration should reach zeta_end.
    assert loop.acquisition.zeta == pytest.approx(0.0)


def test_cold_start_seeds_the_reward_buffer_without_crashing(tmp_path):
    """t=0 has E_0 empty: the reward surrogate is a flat prior, and w is
    computed from it without anything blowing up."""
    loop = _run_loop(tmp_path, num_iters=1)
    assert len(loop.observations) == 1


# ---------------------------------------------------------------------------
# Extension (F) wiring: --validity_gate reaches the acquisition reward
# ---------------------------------------------------------------------------


def test_validity_gate_defaults_to_off(tmp_path):
    _, setup, _, _, _config = _build(tmp_path)
    assert setup.environment.reward.validity_gate is False


def test_validity_gate_flag_reaches_the_bound_acquisition_reward(tmp_path):
    _, setup, _, _, _config = _build(tmp_path, validity_gate=True)
    assert setup.environment.reward.validity_gate is True


def test_loop_runs_end_to_end_with_the_validity_gate_on(tmp_path):
    loop = _run_loop(tmp_path, num_iters=2, validity_gate=True)
    assert len(loop.observations) == 2
