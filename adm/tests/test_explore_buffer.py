import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

import pytest
import torch
import torch.nn as nn

from diffusiongym import DDTensor
from adm.explore import ExploreConfig, ExploreLoop


N_PER_BATCH = 4
N_VALID = 2        # half of each batch is valid
FEAT_DIM = 4
FT_MIN = 8         # so ft_min // 2 = 4 valid samples per phase


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


def make_loop(run_dir, buffer_dir=None, num_iters=6):
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
        torch.zeros(N_PER_BATCH), torch.zeros(N_PER_BATCH),
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
        buffer_dir=buffer_dir,
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


class TestSharedBuffer:
    def test_offline_batches_written_to_buffer(self, tmp_path):
        buffer_dir = tmp_path / "buffer"
        buffer_dir.mkdir()
        loop = make_loop(tmp_path / "run_a", buffer_dir=buffer_dir)

        with patch("adm.explore.train_base_model"), patch("adm.explore.write_video"):
            loop.explore_loop(loop.config.num_iters, loop.config.samples_per_iter)

        # 2 offline iters (each gives 2 valid) fill the ft_min//2=4 threshold
        batch_files = list(buffer_dir.glob("*/batch_*.pt"))
        assert len(batch_files) == 2

    def test_count_file_tracks_valid_samples(self, tmp_path):
        buffer_dir = tmp_path / "buffer"
        buffer_dir.mkdir()
        loop = make_loop(tmp_path / "run_a", buffer_dir=buffer_dir)

        with patch("adm.explore.train_base_model"), patch("adm.explore.write_video"):
            loop.explore_loop(loop.config.num_iters, loop.config.samples_per_iter)

        count_files = list(buffer_dir.glob("*/count.txt"))
        assert len(count_files) == 1
        assert int(count_files[0].read_text()) == 4  # 2 batches × 2 valid

    def test_no_fine_tuning_before_enough_guided_samples(self, tmp_path):
        buffer_dir = tmp_path / "buffer"
        buffer_dir.mkdir()
        # 4 iters: 2 offline + 2 guided — guided_valid=4 reached end of iter 3,
        # but has_enough_data is checked at start of iter, so ft only fires at iter 4
        loop = make_loop(tmp_path / "run_a", buffer_dir=buffer_dir, num_iters=4)

        with patch("adm.explore.train_base_model") as mock_train, \
             patch("adm.explore.write_video"):
            loop.explore_loop(loop.config.num_iters, loop.config.samples_per_iter)

        assert mock_train.call_count == 0

    def test_fine_tuning_starts_after_both_phases_complete(self, tmp_path):
        buffer_dir = tmp_path / "buffer"
        buffer_dir.mkdir()
        # 6 iters: ft fires at iters 4 and 5
        loop = make_loop(tmp_path / "run_a", buffer_dir=buffer_dir, num_iters=6)

        with patch("adm.explore.train_base_model") as mock_train, \
             patch("adm.explore.write_video"):
            loop.explore_loop(loop.config.num_iters, loop.config.samples_per_iter)

        assert mock_train.call_count == 2

    def test_only_valid_samples_passed_to_fine_tuning(self, tmp_path):
        buffer_dir = tmp_path / "buffer"
        buffer_dir.mkdir()
        loop = make_loop(tmp_path / "run_a", buffer_dir=buffer_dir, num_iters=6)

        with patch("adm.explore.train_base_model") as mock_train, \
             patch("adm.explore.write_video"):
            loop.explore_loop(loop.config.num_iters, loop.config.samples_per_iter)

        # First ft call fires at iter 4; batch from iter 4 is appended to guided_batches
        # before fine-tuning, so guided has 3 batches (iters 2, 3, 4) × 2 valid = 6
        #   shared offline buffer: 2 batches × 2 valid = 4
        #   guided: 3 batches × 2 valid = 6
        #   total valid = 10
        first_data = mock_train.call_args_list[0].args[2]  # data: list[DDTensor]
        assert len(first_data[0]) == 10

    def test_uncertainty_uses_local_offline_batches_before_buffer_loaded(self, tmp_path):
        buffer_dir = tmp_path / "buffer"
        buffer_dir.mkdir()
        loop = make_loop(tmp_path / "run_a", buffer_dir=buffer_dir, num_iters=6)

        with patch("adm.explore.train_base_model"), patch("adm.explore.write_video"):
            loop.explore_loop(loop.config.num_iters, loop.config.samples_per_iter)

        set_data_calls = loop.uncertainty.set_data.call_args_list
        # Iter 0: offline_batches=[batch_0], shared not loaded → set_data([latents_0], ...)
        # Iter 1: offline_batches=[batch_0, batch_1] → set_data([latents_0, latents_1], ...)
        assert len(set_data_calls[0].args[0]) == 1   # 1 local offline batch
        assert len(set_data_calls[1].args[0]) == 2   # 2 local offline batches

    def test_uncertainty_uses_shared_buffer_after_offline_done(self, tmp_path):
        buffer_dir = tmp_path / "buffer"
        buffer_dir.mkdir()
        loop = make_loop(tmp_path / "run_a", buffer_dir=buffer_dir, num_iters=6)

        with patch("adm.explore.train_base_model"), patch("adm.explore.write_video"):
            loop.explore_loop(loop.config.num_iters, loop.config.samples_per_iter)

        set_data_calls = loop.uncertainty.set_data.call_args_list
        # Iter 2: offline_done, buffer loaded (2 shared) + 1 guided = 3 batches
        # Iter 3: 2 shared + 2 guided = 4 batches
        assert len(set_data_calls[2].args[0]) == 3
        assert len(set_data_calls[3].args[0]) == 4

    def test_two_runs_share_offline_buffer(self, tmp_path):
        buffer_dir = tmp_path / "buffer"
        buffer_dir.mkdir()

        # Run A: 2 offline iters → buffer has 4 valid, fills threshold
        loop_a = make_loop(tmp_path / "run_a", buffer_dir=buffer_dir, num_iters=2)
        with patch("adm.explore.train_base_model"), patch("adm.explore.write_video"):
            loop_a.explore_loop(loop_a.config.num_iters, loop_a.config.samples_per_iter)

        assert loop_a._total_buffer_valid_count() == 4

        # Run B sees buffer already full, skips offline phase entirely
        loop_b = make_loop(tmp_path / "run_b", buffer_dir=buffer_dir, num_iters=4)
        with patch("adm.explore.train_base_model") as mock_b, \
             patch("adm.explore.write_video"):
            loop_b.explore_loop(loop_b.config.num_iters, loop_b.config.samples_per_iter)

        # Run B: all 4 iters are guided; guided_valid reaches 4 end of iter 1,
        # so ft fires at iters 2 and 3
        assert mock_b.call_count == 2

        # Run B wrote no offline batches to the buffer
        run_b_buffer = buffer_dir / "run_b"
        assert list(run_b_buffer.glob("batch_*.pt")) == []

    def test_combined_buffer_from_two_runs_used_for_fine_tuning(self, tmp_path):
        buffer_dir = tmp_path / "buffer"
        buffer_dir.mkdir()

        # Run A contributes 1 offline batch (2 valid)
        loop_a = make_loop(tmp_path / "run_a", buffer_dir=buffer_dir, num_iters=1)
        with patch("adm.explore.train_base_model"), patch("adm.explore.write_video"):
            loop_a.explore_loop(loop_a.config.num_iters, loop_a.config.samples_per_iter)
        assert loop_a._total_buffer_valid_count() == 2

        # Run B: iter 0 is offline (contributes 2 valid → buffer now full at 4),
        #        iters 1-2 are guided, fine-tuning fires at iter 3
        loop_b = make_loop(tmp_path / "run_b", buffer_dir=buffer_dir, num_iters=6)
        with patch("adm.explore.train_base_model") as mock_b, \
             patch("adm.explore.write_video"):
            loop_b.explore_loop(loop_b.config.num_iters, loop_b.config.samples_per_iter)

        assert mock_b.call_count > 0

        # First ft fires at iter 3 for run B; batch from iter 3 already appended:
        #   shared offline buffer: A's batch + B's offline batch = 4 valid
        #   guided: 3 batches (iters 1, 2, 3) × 2 valid = 6
        #   total valid = 10
        first_data = mock_b.call_args_list[0].args[2]
        assert len(first_data[0]) == 10
