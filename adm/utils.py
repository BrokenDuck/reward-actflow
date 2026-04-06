from typing import Generic, Any, TypeVar
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate
from torchvision.transforms.functional import to_pil_image
from diffusiongym import D, Sample, DDMixin, DDTensor
from diffusiongym.utils import index_dict
from PIL import Image
from pathlib import Path
import argparse
import imageio.v2 as imageio
import open_clip
import logging


T = TypeVar("T", bound=DDMixin)


@dataclass
class Batch(Generic[D]):
    samples: D
    latents: D
    rewards: torch.Tensor
    valids: torch.Tensor
    kwargs: dict[str, Any]

    @classmethod
    def from_sample(cls, sample: "Sample[T]") -> "Batch[T]":
        return cls(  # type: ignore
            samples=sample.sample,  # type: ignore
            latents=sample.latent,  # type: ignore
            rewards=sample.rewards,
            valids=sample.valids,
            kwargs=sample.kwargs,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> "Batch[D]":
        return Batch(
            samples=self.samples[idx],
            latents=self.latents[idx],
            rewards=self.rewards[idx:idx+1],
            valids=self.valids[idx:idx+1],
            kwargs=index_dict(self.kwargs, idx, idx + 1),
        )

    @staticmethod
    def concat(batches: list["Batch[T]"]) -> "Batch[T]":
        data_type = type(batches[0].samples)

        all_kwargs = []
        for batch in batches:
            for i in range(len(batch)):
                all_kwargs.append(index_dict(batch.kwargs, i))

        return Batch(
            samples=data_type.collate([b.samples for b in batches]),
            latents=data_type.collate([b.latents for b in batches]),
            rewards=torch.cat([b.rewards for b in batches], dim=0),
            valids=torch.cat([b.valids for b in batches], dim=0),
            kwargs=default_collate(all_kwargs),  # type: ignore
        )


def filter_out_invalids(batches: list[Batch[D]]) -> Batch[D]:
    valid_batches: list[Batch[D]] = []

    for batch in batches:
        for i in range(len(batch)):
            if batch.valids[i]:
                valid_batches.append(batch[i])

    return Batch.concat(valid_batches)


def filter_out_valids(batches: list[Batch[D]]) -> Batch[D] | None:
    """Return only the invalid samples from the batches, or None if there are none."""
    invalid_batches: list[Batch[D]] = []

    for batch in batches:
        for i in range(len(batch)):
            if not batch.valids[i]:
                invalid_batches.append(batch[i])

    if len(invalid_batches) == 0:
        return None

    return Batch.concat(invalid_batches)


def write_video(frame_paths: list[Path], video_path: Path, fps: int):
    if len(frame_paths) == 0:
        return

    with imageio.get_writer(video_path, fps=fps, codec="libx264") as writer:
        for frame_path in frame_paths:
            writer.append_data(imageio.imread(frame_path))  # type: ignore


def serialize_args(args: argparse.Namespace) -> dict:
    out = {}
    for k, v in vars(args).items():
        if isinstance(v, Path):
            out[k] = str(v)
        else:
            out[k] = v
    return out


def setup_logger(folder: Path | None = None, verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger("active_pretraining")
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


def add_valid_border(images: torch.Tensor, valids: torch.Tensor, thickness: int = 2) -> torch.Tensor:
    images = images.clone()
    if images.shape[1] == 1:
        images = torch.cat([images, images, images], dim=1)

    for i, valid in enumerate(valids):
        if valid:
            images[i, 1, :thickness, :] = 1.0
            images[i, 1, -thickness:, :] = 1.0
            images[i, 1, :, :thickness] = 1.0
            images[i, 1, :, -thickness:] = 1.0

    return images


def to_pil_images(x: DDTensor) -> list[Image.Image]:
    img_list = []
    for i in range(len(x)):
        img_list.append(to_pil_image(x.data[i].to(dtype=torch.float)))
    return img_list


class CLIP:
    def __init__(self, device: torch.device):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="laion2b_s34b_b79k",
            device=device,
        )
        self.model.eval()  # type: ignore
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")

    @torch.no_grad()
    def embed_images(self, images: list[Image.Image]) -> torch.Tensor:
        images = [self.preprocess(img).unsqueeze(0) for img in images]  # type: ignore
        feats = torch.cat([self.model.encode_image(img.to(self.device)) for img in images])  # type: ignore
        return feats / feats.norm(dim=-1, keepdim=True)


    @torch.no_grad()
    def embed_texts(self, texts: list[str]) -> torch.Tensor:
        tokenized_texts = self.tokenizer(texts).to(self.device)
        feats = self.model.encode_text(tokenized_texts)  # type: ignore
        return feats / feats.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def score(self, images: list[Image.Image], texts: list[str]) -> torch.Tensor:
        assert len(images) == len(texts)

        img_feats = self.embed_images(images)
        txt_feats = self.embed_texts(texts)

        return 100 * F.cosine_similarity(img_feats, txt_feats, dim=-1).clamp(min=0)
