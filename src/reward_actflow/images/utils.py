"""Image-domain helpers.

Moved here out of `reward_actflow.utils`, which the ActFlow loop imports on
every run: these pull in `open_clip` and `torchvision`, and `open_clip` is
neither declared in `pyproject.toml` nor installed, so their presence made the
shared utils module unimportable.
"""

import torch
import torch.nn.functional as F
from diffusiongym.types import DDTensor
from PIL import Image


def add_valid_border(
    images: torch.Tensor, valids: torch.Tensor, thickness: int = 2
) -> torch.Tensor:
    """Draw a green border around the images the verifier accepted."""
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
    from torchvision.transforms.functional import to_pil_image

    return [to_pil_image(x.data[i].to(dtype=torch.float)) for i in range(len(x))]


class CLIP:
    """CLIP ViT-B/32 image and text embeddings, plus their cosine score."""

    def __init__(self, device: torch.device):
        import open_clip  # ty: ignore[unresolved-import]

        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="laion2b_s34b_b79k",
            device=device,
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")

    @torch.no_grad()
    def embed_images(self, images: list[Image.Image]) -> torch.Tensor:
        processed = [self.preprocess(img).unsqueeze(0) for img in images]
        feats = torch.cat(
            [self.model.encode_image(img.to(self.device)) for img in processed]
        )
        return feats / feats.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def embed_texts(self, texts: list[str]) -> torch.Tensor:
        tokenized_texts = self.tokenizer(texts).to(self.device)
        feats = self.model.encode_text(tokenized_texts)
        return feats / feats.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def score(self, images: list[Image.Image], texts: list[str]) -> torch.Tensor:
        assert len(images) == len(texts)

        img_feats = self.embed_images(images)
        txt_feats = self.embed_texts(texts)

        return 100 * F.cosine_similarity(img_feats, txt_feats, dim=-1).clamp(min=0)
