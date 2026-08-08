"""Tests for `Batch` and `filter_out_invalids` (reward_actflow.utils).

No fixtures/registration needed — plain `DDTensor` batches, mirroring
`tests/toy/test_actflow_toy.py`'s style (module-level helper, not a
`pytest.fixture`).
"""

import pytest
import torch
from diffusiongym.types import DDTensor

from reward_actflow.utils import Batch, filter_out_invalids


def _batch(
    n: int,
    *,
    valid: torch.Tensor | None = None,
    task_rewards: torch.Tensor | None = None,
) -> Batch:
    data = torch.arange(n * 2, dtype=torch.float).reshape(n, 2)
    return Batch(
        samples=DDTensor(data),
        latents=DDTensor(data),
        rewards=torch.arange(n, dtype=torch.float),
        valids=valid if valid is not None else torch.ones(n, dtype=torch.bool),
        kwargs={},
        task_rewards=task_rewards,
    )


def test_filter_out_invalids_handles_an_all_invalid_batch():
    """`Batch.concat` cannot build a batch from an empty list — it reads the
    state type off the first element — so `t=0` with no verified-valid rows
    used to raise `IndexError` here rather than yield a length-0 batch."""
    batch = _batch(4, valid=torch.zeros(4, dtype=torch.bool))
    result = filter_out_invalids([batch])
    assert len(result) == 0


def test_filter_out_invalids_keeps_only_valid_rows():
    batch = _batch(4, valid=torch.tensor([True, False, True, False]))
    result = filter_out_invalids([batch])
    assert len(result) == 2
    assert torch.equal(result.rewards, torch.tensor([0.0, 2.0]))


def test_filter_out_invalids_concatenates_across_batches():
    a = _batch(2, valid=torch.tensor([True, False]))
    b = _batch(2, valid=torch.tensor([False, True]))
    result = filter_out_invalids([a, b])
    assert len(result) == 2
    assert torch.equal(result.rewards, torch.tensor([0.0, 1.0]))


def test_select_supports_repeated_indices():
    """Importance resampling (drawing from D+ with probability ~ w) needs
    repeats; `__getitem__` on a plain int cannot do this in one call."""
    batch = _batch(4)
    result = batch.select(torch.tensor([0, 0, 2]))
    assert len(result) == 3
    assert torch.equal(result.rewards, torch.tensor([0.0, 0.0, 2.0]))
    assert torch.equal(result.samples.data, batch.samples.data[[0, 0, 2]])


def test_batch_concat_roundtrips_task_rewards():
    a = _batch(2, task_rewards=torch.tensor([1.0, 2.0]))
    b = _batch(2, task_rewards=torch.tensor([3.0, 4.0]))
    result = Batch.concat([a, b])
    assert torch.equal(result.task_rewards, torch.tensor([1.0, 2.0, 3.0, 4.0]))


def test_batch_concat_raises_on_partial_task_rewards():
    """A silently dropped task_rewards column is worse than a crash."""
    a = _batch(2, task_rewards=torch.tensor([1.0, 2.0]))
    b = _batch(2, task_rewards=None)
    with pytest.raises(ValueError, match="task_rewards"):
        Batch.concat([a, b])


def test_to_and_cpu_preserve_task_rewards():
    batch = _batch(2, task_rewards=torch.tensor([1.0, 2.0]))
    moved = batch.cpu()
    assert moved.task_rewards is not None
    assert torch.equal(moved.task_rewards, batch.task_rewards)


def test_select_preserves_none_task_rewards():
    batch = _batch(4, task_rewards=None)
    result = batch.select(torch.tensor([1, 2]))
    assert result.task_rewards is None
