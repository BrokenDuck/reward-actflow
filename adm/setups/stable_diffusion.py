from typing import Any, Optional

import torch
from torchvision.utils import save_image, make_grid
from diffusiongym import BaseModel, Environment, DDTensor, ConstantNoiseSchedule
from diffusiongym.images import SD15BaseModel, AestheticReward
from vendi_score import vendi
from matplotlib.figure import Figure
from argparse import ArgumentParser
from pathlib import Path
from peft import LoraConfig
import pyiqa

from adm.setups.problem_setup import ProblemSetup, SampleFile
from adm.utils import add_valid_border, CLIP, Batch, to_pil_images


class StableDiffusionProblemSetup(ProblemSetup[DDTensor]):
    def __init__(self, args: dict[str, Any], device: Optional[torch.device]=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.clip_threshold = args["sd_clip_thresh"]
        self.topiq_threshold = args["sd_topiq_thresh"]
        self.niqe_threshold = args["sd_niqe_thresh"]
        self.aes_threshold = args["sd_aesthetic_thresh"]

        prompts = args.get("sd_prompts", None)
        if isinstance(prompts, list) and len(prompts) == 0:
            prompts = None

        self._base_model = SD15BaseModel(
            min_cfg_scale=args["sd_min_cfg_scale"],
            max_cfg_scale=args["sd_max_cfg_scale"],
            prompts=prompts,
            device=device,
        )

        text_encoder = self._base_model.pipe.text_encoder
        text_encoder.requires_grad_(False)
        text_encoder.eval()

        # Add LoRA
        peft_config = LoraConfig(
            r=args["sd_lora_rank"],
            lora_alpha=args["sd_lora_alpha"],
            lora_dropout=args["sd_lora_dropout"],
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        )
        self._base_model.unet.add_adapter(peft_config)

        self._base_model.scheduler.noise_schedule = ConstantNoiseSchedule(0)
        self._base_model.eval()

        self.clip = CLIP(device)
        self.topiq =  pyiqa.create_metric("topiq_nr", device=device)
        self.niqe = pyiqa.create_metric("niqe", device=device)
        self.aes_reward = AestheticReward()

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        parser.add_argument("--sd_prompts", type=str, nargs="+", default=None)
        parser.add_argument("--sd_min_cfg_scale", type=float, default=0.0)
        parser.add_argument("--sd_max_cfg_scale", type=float, default=6.0)
        parser.add_argument("--sd_lora_rank", type=int, default=8)
        parser.add_argument("--sd_lora_alpha", type=float, default=1.0)
        parser.add_argument("--sd_lora_dropout", type=float, default=0.2)
        parser.add_argument("--sd_clip_thresh", type=float, default=28.0)
        parser.add_argument("--sd_topiq_thresh", type=float, default=0.45)
        parser.add_argument("--sd_niqe_thresh", type=float, default=4.2)
        parser.add_argument("--sd_aesthetic_thresh", type=float, default=5.0)

    @property
    def base_model(self) -> BaseModel[DDTensor]:
        return self._base_model

    @torch.no_grad()
    def validity(self, samples: DDTensor, kwargs: dict[str, Any]) -> torch.Tensor:
        text_list = kwargs["prompt"]
        img_list = to_pil_images(samples)
        clip_score = self.clip.score(img_list, text_list).squeeze()

        imgs = samples.data
        topiq_score = self.topiq(imgs).squeeze()
        niqe_score = self.niqe(imgs).squeeze()
        aes_score = self.aes_reward(samples, None)[0].squeeze()

        return (
            (clip_score >= self.clip_threshold)
            & (topiq_score >= self.topiq_threshold)
            & (niqe_score <= self.niqe_threshold)
            & (aes_score >= self.aes_threshold)
        )

    @property
    def feature_layer(self) -> str:
        return "unet.mid_block"

    def postprocess_features(self, latents: DDTensor, feats: torch.Tensor) -> torch.Tensor:
        # If CFG, only use conditional features
        if feats.shape[0] == 2 * len(latents):
            feats, _ = feats.chunk(2)

        return feats.mean(dim=[-2, -1])

    def visualize_sample(self, env: Environment[DDTensor], batch: Batch[DDTensor]) -> Figure:
        x = batch.samples.data.cpu()
        v = batch.valids.cpu()
        grid = make_grid(add_valid_border(x, v, thickness=16), nrow=8)

        fig = Figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(grid.permute(1, 2, 0).numpy())
        ax.axis("off")

        return fig

    def save_sample(self, sample: DDTensor, kwargs: dict, filename: Path) -> Path:
        x = sample.data.cpu()
        save_image(x, filename.with_suffix(".png"))

        # Save prompt
        with open(filename.with_suffix("_prompt.txt"), "w") as f:
            f.write(kwargs.get("prompt", "prompt not found"))

        # TODO: find some way to save image and prompt together, e.g. in a single file or directory
        return filename.with_suffix(".png")

    # def compute_metrics(self, batch: Batch[DDTensor]) -> dict[str, float]:
    #     img_list = to_pil_images(batch.samples)
    #     feats = self.clip.embed_images(img_list)
    #     return { "vendi": vendi.score_X(feats) }

    def compute_sample_metrics(self, sample_files: list[SampleFile]) -> dict[str, dict[str, float]]:
        # todo: aesthetic score?
        return dict()
