import argparse
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import torch
from diffusiongym.types import DDBatch
from diffusiongym.utils import index_dict
from torch.utils.data._utils.collate import default_collate


@dataclass
class Batch[D: DDBatch]:
    """One iteration's worth of observations, i.e. a slice of `D_t`.

    `valids` holds the black-box verifier's answer `y = v(x)` (Algorithm 1
    line 5). `rewards` holds the scalar the fine-tuning algorithm actually
    optimized, which already folds `valids` in via the reward gate.
    """

    samples: D
    latents: D
    rewards: torch.Tensor
    valids: torch.Tensor
    kwargs: dict[str, Any]
    #: Black-box task reward r~(x), ActFlow-R Algorithm 1 line 6 — queried on
    #: the valid subset only, so this stays `None` until something asks for it.
    #: Base ActFlow has no task reward and never sets this field.
    task_rewards: torch.Tensor | None = None

    @classmethod
    def from_endpoints(
        cls,
        *,
        latents: D,
        samples: D,
        rewards: torch.Tensor,
        valids: torch.Tensor,
        conditioning: Mapping[str, Any],
        task_rewards: torch.Tensor | None = None,
    ) -> "Batch[D]":
        """Build a batch from the terminal endpoints of one collection round."""
        return cls(
            samples=samples,
            latents=latents,
            rewards=rewards,
            valids=valids,
            kwargs=dict(conditioning),
            task_rewards=task_rewards,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def to(self, device: torch.device | str) -> "Batch[D]":
        return Batch(
            samples=self.samples.to(device),
            latents=self.latents.to(device),
            rewards=self.rewards.to(device),
            valids=self.valids.to(device),
            kwargs={
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in self.kwargs.items()
            },
            task_rewards=(
                self.task_rewards.to(device) if self.task_rewards is not None else None
            ),
        )

    def cpu(self) -> "Batch[D]":
        return self.to("cpu")

    def __getitem__(self, idx: int) -> "Batch[D]":
        return Batch(
            samples=self.samples[idx],
            latents=self.latents[idx],
            rewards=self.rewards[idx : idx + 1],
            valids=self.valids[idx : idx + 1],
            kwargs=index_dict(self.kwargs, idx, idx + 1),
            task_rewards=(
                self.task_rewards[idx : idx + 1]
                if self.task_rewards is not None
                else None
            ),
        )

    def select(self, idx: torch.Tensor) -> "Batch[D]":
        """Vectorised sub-selection by an index tensor (repeats allowed).

        Unlike `__getitem__`, which returns one Python `Batch` per row and pays
        for it at O(|D|) objects per call, this indexes every field once — and
        is the only path that supports *repeated* indices, which importance
        resampling (drawing from `D+` with probability proportional to `w`)
        needs.
        """
        idx = idx.to(dtype=torch.long, device=self.rewards.device)
        return Batch(
            samples=self.samples.index_select(idx),
            latents=self.latents.index_select(idx),
            rewards=self.rewards[idx],
            valids=self.valids[idx],
            kwargs=_select_kwargs(self.kwargs, idx),
            task_rewards=self.task_rewards[idx]
            if self.task_rewards is not None
            else None,
        )

    @staticmethod
    def concat[T: DDBatch](batches: Sequence["Batch[T]"]) -> "Batch[T]":
        data_type = type(batches[0].samples)

        all_kwargs: dict[str, Any] = {}
        for key in batches[0].kwargs:
            values = [b.kwargs[key] for b in batches]
            if isinstance(values[0], torch.Tensor):
                all_kwargs[key] = torch.cat(values, dim=0)
            elif isinstance(values[0], list):
                all_kwargs[key] = [item for v in values for item in v]
            else:
                all_kwargs[key] = default_collate(values)

        has_task_rewards = [b.task_rewards is not None for b in batches]
        if any(has_task_rewards) and not all(has_task_rewards):
            raise ValueError(
                "Batch.concat: task_rewards is set on some batches and not "
                "others. A silently dropped column is worse than a crash here "
                "— query the task reward for every batch, or none."
            )
        task_rewards = (
            torch.cat([b.task_rewards for b in batches], dim=0)  # ty: ignore[no-matching-overload]
            if all(has_task_rewards)
            else None
        )

        return Batch(
            samples=data_type.concat([b.samples for b in batches]),
            latents=data_type.concat([b.latents for b in batches]),
            rewards=torch.cat([b.rewards for b in batches], dim=0),
            valids=torch.cat([b.valids for b in batches], dim=0),
            kwargs=all_kwargs,
            task_rewards=task_rewards,
        )


def _select_kwargs(kwargs: Mapping[str, Any], idx: torch.Tensor) -> dict[str, Any]:
    """Sub-select a conditioning dict by an index tensor (repeats allowed).

    `diffusiongym.utils.index_dict` only supports int/slice starts; this is
    the tensor-index counterpart, modelled on `orw_cfm._index_conditioning`.
    """
    result: dict[str, Any] = {}
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            result[k] = v[idx]
        elif isinstance(v, list):
            result[k] = [v[i] for i in idx.tolist()]
        else:
            result[k] = v
    return result


def filter_out_invalids[D: DDBatch](batches: Sequence[Batch[D]]) -> Batch[D]:
    """Concatenate the verifier-passing rows of every batch.

    Vectorised (one `select` call per input batch, not one `Batch` per row),
    and safe when every row in every batch is invalid: `Batch.concat` cannot
    build a batch from an empty list (it reads the state type off the first
    element), so a run whose first `D+` is empty saw an `IndexError` here
    before this rewrite. This returns a genuine zero-length `Batch` of the
    right type instead — `len(result) == 0`, not a crash.
    """
    kept = [b.select(b.valids.nonzero(as_tuple=True)[0]) for b in batches]
    non_empty = [b for b in kept if len(b) > 0]
    return Batch.concat(non_empty) if non_empty else kept[0]


def write_video(frame_paths: Sequence[Path], video_path: Path, fps: int):
    if len(frame_paths) == 0:
        return

    with imageio.get_writer(video_path, fps=fps, codec="libx264") as writer:
        for frame_path in frame_paths:
            writer.append_data(imageio.imread(frame_path))  # ty: ignore[unresolved-attribute]


def serialize_args(args: argparse.Namespace) -> dict:
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}


def setup_logger(folder: Path | None = None, verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger("reward_actflow")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        "[%(asctime)s] (%(levelname)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(ch)

    if folder is not None:
        fh = logging.FileHandler(folder / "log.txt")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
