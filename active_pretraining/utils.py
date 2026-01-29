from typing import Any, TypeVar, Optional, Generic
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate
from flowgym import D
from PIL import Image
import open_clip


T = TypeVar("T")


@dataclass
class Batch(Generic[D]):
    samples: D
    latents: D
    valids: torch.Tensor
    kwargs: dict[str, Any]

    def __post_init__(self):
        if len(self.samples) != len(self.latents) or len(self.samples) != len(self.valids):
            raise ValueError(f"Length of samples, latents, and valids must be the same, got ({len(self.samples)}, {len(self.latents)}, {len(self.valids)})")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> "Batch[D]":
        return Batch(
            samples=self.samples[idx],
            latents=self.latents[idx],
            valids=self.valids[idx:idx+1],
            kwargs=index_dict(self.kwargs, idx),
        )

    @staticmethod
    def concat(batches: list["Batch[D]"]) -> "Batch[D]":
        batch_type = type(batches[0].samples)

        batch_samples = batch_type.collate([b.samples for b in batches])
        batch_latents = batch_type.collate([b.latents for b in batches])
        batch_valids = torch.cat([b.valids for b in batches], dim=0)
        all_kwargs = []
        for batch in batches:
            for i in range(len(batch)):
                all_kwargs.append(index_dict(batch.kwargs, i))

        batch_kwargs = default_collate(all_kwargs)  # type: ignore

        return Batch(
            samples=batch_samples,
            latents=batch_latents,
            valids=batch_valids,
            kwargs=batch_kwargs,  # type: ignore
        )


def index_dict(d: T, start: int, end: Optional[int] = None) -> T:
    """Recursively index into the leaves of a nested dictionary.

    Parameters
    ----------
    d : T
        Any value, if a dictionary, will be processed recursively.

    start : int
        The index to select from list/tensor leaves.
    
    end : Optional[int], optional
        The end index to select from list/tensor leaves, by default None.

    Returns
    -------
    T
        If d is a dictionary, returns a dictionary with the same keys and indexed leaves.
    """
    if end is None:
        idx = start
    else:
        idx = slice(start, end)

    if isinstance(d, dict):
        return {k: index_dict(v, start, end) for k, v in d.items()}  # type: ignore

    elif isinstance(d, (list, tuple, torch.Tensor)):
        return d[idx]  # type: ignore

    elif isinstance(d, (float, int, str)):
        return d

    else:
        raise TypeError(f"Unsupported leaf type: {type(d)}")


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
