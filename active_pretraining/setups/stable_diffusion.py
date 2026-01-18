from typing import Any, Optional

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import save_image, make_grid
from flowgym import BaseModel, Environment, FlowTensor, ConstantNoiseSchedule
from flowgym.images import SD15BaseModel
from vendi_score import vendi
import open_clip
from PIL import Image
from matplotlib.figure import Figure
import os

from active_pretraining.problem_setup import ProblemSetup, SampleFile
from active_pretraining.utils import add_valid_border


class StableDiffusionProblemSetup(ProblemSetup[FlowTensor]):
    def __init__(self, args: dict[str, Any], device: Optional[torch.device]=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.score_threshold = args["sd_score_threshold"]
        prompts = args.get("sd_prompts", None)
        if isinstance(prompts, list) and len(prompts) == 0:
            prompts = None

        self._base_model = SD15BaseModel(
            cfg_scale=args["sd_cfg_scale"],
            prompts=prompts,
            device=device,
        )
        self._base_model.scheduler.noise_schedule = ConstantNoiseSchedule(0)
        self._base_model.eval()

        self.clip = CLIP(device)

    @property
    def base_model(self) -> BaseModel[FlowTensor]:
        return self._base_model
    
    def _to_pil_images(self, x: FlowTensor) -> list[Image.Image]:
        img_list = []
        for i in range(len(x)):
            img_list.append(to_pil_image(x.data[i].to(dtype=torch.float)))
        return img_list

    def validity(self, x: FlowTensor, kwargs: dict[str, Any]) -> torch.Tensor:
        text_list = kwargs["prompt"]
        img_list = self._to_pil_images(x)
        scores = self.clip.score(img_list, text_list)
        return scores > self.score_threshold

    @property
    def feature_layer(self) -> str:
        return "unet.mid_block"

    def feature_postprocess(self, x: FlowTensor, feats: torch.Tensor) -> torch.Tensor:
        # If CFG, only use conditional features
        if feats.shape[0] == 2 * len(x):
            feats, _ = feats.chunk(2)

        return feats.mean(dim=[-2, -1])

    def visualize_sample(
        self,
        env: Environment[FlowTensor],
        samples: list[FlowTensor],
        valids: list[torch.Tensor],
    ) -> Figure:
        x = samples[-1].data.cpu()
        v = valids[-1].cpu()
        grid = make_grid(add_valid_border(x, v, thickness=16), nrow=8)

        fig = Figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(grid.permute(1, 2, 0).numpy())
        ax.axis("off")

        return fig

    def save_sample(self, sample: FlowTensor, kwargs: dict, filename: os.PathLike | str):
        x = sample.data.cpu()
        save_image(x, f"{filename}.png")

        # Save prompt
        with open(f"{filename}_prompt.txt", "w") as f:
            f.write(kwargs.get("prompt", "prompt not found"))

    def compute_metrics(
        self,
        samples: list[FlowTensor],
        valids: list[torch.Tensor],
        kwargs: list[dict],
    ) -> dict[str, float]:
        img_list = self._to_pil_images(FlowTensor.collate(samples))
        feats = self.clip.embed_images(img_list)
        return { "vendi": vendi.score_X(feats) }

    def compute_sample_metrics(self, sample_files: list[SampleFile]) -> dict[str, dict[str, float]]:
        # todo: aesthetic score?
        return dict()


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
