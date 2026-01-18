from typing import Any, Optional

import torch
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import save_image, make_grid
from flowgym import BaseModel, Environment, FlowTensor, ConstantNoiseSchedule
from flowgym.images import SD15BaseModel
from vendi_score import vendi
from PIL import Image
from matplotlib.figure import Figure
import os

from active_pretraining.problem_setup import ProblemSetup, SampleFile
from active_pretraining.utils import add_valid_border, CLIP, Batch


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

    @torch.no_grad()
    def validity(self, samples: FlowTensor, kwargs: dict[str, Any]) -> torch.Tensor:
        text_list = kwargs["prompt"]
        img_list = self._to_pil_images(samples)
        scores = self.clip.score(img_list, text_list)
        return scores > self.score_threshold

    @property
    def feature_layer(self) -> str:
        return "unet.mid_block"

    def postprocess_features(self, latents: FlowTensor, feats: torch.Tensor) -> torch.Tensor:
        # If CFG, only use conditional features
        if feats.shape[0] == 2 * len(latents):
            feats, _ = feats.chunk(2)

        return feats.mean(dim=[-2, -1])

    def visualize_batch(self, env: Environment[FlowTensor], batch: Batch[FlowTensor]) -> Figure:
        x = batch.samples.data.cpu()
        v = batch.valids.cpu()
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

    def compute_metrics(self, batches: list[Batch[FlowTensor]]) -> dict[str, float]:
        img_list = self._to_pil_images(FlowTensor.collate([b.samples for b in batches]))
        feats = self.clip.embed_images(img_list)
        return { "vendi": vendi.score_X(feats) }

    def compute_sample_metrics(self, sample_files: list[SampleFile]) -> dict[str, dict[str, float]]:
        # todo: aesthetic score?
        return dict()

