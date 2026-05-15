import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from diffusiongym import DDTensor
from adm.explore import ExploreConfig, ExploreLoop


N_PER_BATCH = 4
N_VALID = 2
FEAT_DIM = 4
FT_MIN = 8


def fake_sample(n=N_PER_BATCH):
    valids = torch.zeros(n, dtype=torch.bool)
    valids[:N_VALID] = True
    return SimpleNamespace(
        sample=DDTensor(torch.randn(n, FEAT_DIM)),
        latent=DDTensor(torch.randn(n, FEAT_DIM)),
        rewards=torch.zeros(n),
        valids=valids,
        kwargs={},
    )


def make_loop(run_dir, num_iters=6):
    base_model = nn.Linear(FEAT_DIM, FEAT_DIM)

    problem = MagicMock()
    problem.base_model = base_model
    problem.postprocess_latents = lambda batch: batch.latents

    def fake_validity(samples, kwargs):
        n = len(samples)
        v = torch.zeros(n, dtype=torch.bool)
        v[:N_VALID] = True
        return v

    problem.validity = fake_validity

    uncertainty = MagicMock()
    uncertainty.mean_and_uncertainty.return_value = (
        torch.zeros(N_PER_BATCH),
        torch.zeros(N_PER_BATCH),
    )

    config = ExploreConfig(
        folder=run_dir,
        ft_min_dataset_size=FT_MIN,
        samples_per_iter=N_PER_BATCH,
        sample_batch_size=N_PER_BATCH,
        ft_steps=1,
        ft_batch_size=2,
        eval_samples=0,
        eval_valid_samples=0,
        eval_every=99999,
        num_iters=num_iters,
    )

    fake_env = MagicMock()
    fake_env.batch_sample.side_effect = lambda n, bs, **kw: fake_sample(n)

    with patch("adm.explore.construct_env", return_value=fake_env):
        loop = ExploreLoop(
            problem_setup=problem,
            uncertainty=uncertainty,
            reward=MagicMock(),
            config=config,
            logger=logging.getLogger("test"),
        )

    loop.env = fake_env
    loop.visualize_iter = MagicMock()
    return loop


# With FT_MIN=8 and N_VALID=2 per batch:
#   use_guidance kicks in when total_valid > 4, which happens at iter 3 start (after 3 offline iters)
#   so iters 0-2 are offline, iters 3-5 are guided


class TestResumability:
    def test_loop_state_file_created_after_run(self, tmp_path):
        loop = make_loop(tmp_path / "run", num_iters=3)
        with patch("adm.explore.train_base_model"), patch("adm.explore.write_video"):
            loop.explore_loop(3, loop.config.samples_per_iter)
        assert (tmp_path / "run" / "loop_state.pt").exists()

    def test_loop_state_records_last_completed_iteration(self, tmp_path):
        loop = make_loop(tmp_path / "run", num_iters=3)
        with patch("adm.explore.train_base_model"), patch("adm.explore.write_video"):
            loop.explore_loop(3, loop.config.samples_per_iter)
        state = torch.load(tmp_path / "run" / "loop_state.pt", map_location="cpu")
        assert state["iteration"] == 2

    def test_loop_state_tracks_offline_valid_count(self, tmp_path):
        loop = make_loop(tmp_path / "run", num_iters=3)
        with patch("adm.explore.train_base_model"), patch("adm.explore.write_video"):
            loop.explore_loop(3, loop.config.samples_per_iter)
        state = torch.load(tmp_path / "run" / "loop_state.pt", map_location="cpu")
        assert state["offline_valid_count"] == 3 * N_VALID
        assert state["guided_valid_count"] == 0

    def test_loop_state_tracks_guided_batches(self, tmp_path):
        loop = make_loop(tmp_path / "run", num_iters=6)
        with patch("adm.explore.train_base_model"), patch("adm.explore.write_video"):
            loop.explore_loop(6, loop.config.samples_per_iter)
        state = torch.load(tmp_path / "run" / "loop_state.pt", map_location="cpu")
        assert len(state["offline_batches"]) == 3
        assert len(state["guided_batches"]) == 3
        assert state["offline_valid_count"] == 3 * N_VALID
        assert state["guided_valid_count"] == 3 * N_VALID

    def test_resumed_loop_skips_completed_iters(self, tmp_path):
        run_dir = tmp_path / "run"

        loop1 = make_loop(run_dir, num_iters=4)
        with patch("adm.explore.train_base_model"), patch("adm.explore.write_video"):
            loop1.explore_loop(4, loop1.config.samples_per_iter)

        loop2 = make_loop(run_dir)
        with patch("adm.explore.train_base_model"), patch("adm.explore.write_video"):
            loop2.explore_loop(6, loop2.config.samples_per_iter)

        assert loop2.env.batch_sample.call_count == 2

    def test_resumed_loop_produces_same_counts_as_full_run(self, tmp_path):
        loop_full = make_loop(tmp_path / "full", num_iters=6)
        with patch("adm.explore.train_base_model"), patch("adm.explore.write_video"):
            loop_full.explore_loop(6, loop_full.config.samples_per_iter)
        full_state = torch.load(tmp_path / "full" / "loop_state.pt", map_location="cpu")

        loop1 = make_loop(tmp_path / "interrupted", num_iters=4)
        with patch("adm.explore.train_base_model"), patch("adm.explore.write_video"):
            loop1.explore_loop(4, loop1.config.samples_per_iter)

        loop2 = make_loop(tmp_path / "interrupted")
        with patch("adm.explore.train_base_model"), patch("adm.explore.write_video"):
            loop2.explore_loop(6, loop2.config.samples_per_iter)
        resumed_state = torch.load(tmp_path / "interrupted" / "loop_state.pt", map_location="cpu")

        assert full_state["offline_valid_count"] == resumed_state["offline_valid_count"]
        assert full_state["guided_valid_count"] == resumed_state["guided_valid_count"]
        assert len(full_state["offline_batches"]) == len(resumed_state["offline_batches"])
        assert len(full_state["guided_batches"]) == len(resumed_state["guided_batches"])

    def test_resume_rebuilds_uncertainty_from_saved_batches(self, tmp_path):
        run_dir = tmp_path / "run"

        loop1 = make_loop(run_dir, num_iters=3)
        with patch("adm.explore.train_base_model"), patch("adm.explore.write_video"):
            loop1.explore_loop(3, loop1.config.samples_per_iter)

        loop2 = make_loop(run_dir)
        with patch("adm.explore.train_base_model"), patch("adm.explore.write_video"):
            loop2.explore_loop(6, loop2.config.samples_per_iter)

        rebuild_call = loop2.uncertainty.set_data.call_args_list[0]
        assert len(rebuild_call.args[0]) == 3

    def test_resume_restores_model_checkpoint(self, tmp_path):
        run_dir = tmp_path / "run"

        loop1 = make_loop(run_dir, num_iters=3)
        with patch("adm.explore.train_base_model"), patch("adm.explore.write_video"):
            loop1.explore_loop(3, loop1.config.samples_per_iter)

        saved_state = torch.load(run_dir / "checkpoints" / "last.pt", map_location="cpu")

        loop2 = make_loop(run_dir)
        with torch.no_grad():
            for p in loop2.base_model.parameters():
                p.fill_(999.0)

        with patch("adm.explore.train_base_model"), patch("adm.explore.write_video"):
            loop2.explore_loop(6, loop2.config.samples_per_iter)

        for name, param in loop2.base_model.named_parameters():
            assert torch.allclose(param.cpu(), saved_state[name]), f"param {name} not restored"

    def test_no_resume_without_state_file(self, tmp_path):
        run_dir = tmp_path / "run"
        loop = make_loop(run_dir, num_iters=3)
        with patch("adm.explore.train_base_model"), patch("adm.explore.write_video"):
            loop.explore_loop(3, loop.config.samples_per_iter)
        assert loop.env.batch_sample.call_count == 3
