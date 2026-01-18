from typing import Any, TypeVar, Optional

import torch
import torch.nn.functional as F
from PIL import Image
import open_clip


T = TypeVar("T")


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
        return {k: index_dict(v, start, end) for k, v in d.items()}

    elif isinstance(d, (list, tuple, torch.Tensor)):
        return d[idx]

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
