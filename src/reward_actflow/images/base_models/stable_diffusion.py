"""Pre-trained base model for Stable Diffusion 2."""

import json
import random
from importlib.resources import open_text
from typing import Any, cast

import torch
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)
from diffusiongym import BaseModel, DDTensor, DiffusionScheduler
from diffusiongym.registry import base_model_registry
from diffusiongym.utils import append_dims


class StableDiffusionBaseModel(BaseModel[DDTensor]):
    """Stable Diffusion base model.

    Parameters
    ----------
    model_name : str
        Name of the Stable Diffusion diffusers model to use, e.g., "sd-legacy/stable-diffusion-v1-5".
    cfg_scale : float
        Classifier-free guidance scale to use during sampling. If 0.0, no CFG is used.
    min_cfg_scale : float, optional
        If provided, the minimum CFG scale to use during sampling.
    max_cfg_scale : float, optional
        If provided, the maximum CFG scale to use during sampling.
    prompts : list[str], optional
        List of prompts to use for sampling if no prompt is provided through the input. If None,
        a default set of prompts is used.
    device : torch.device, optional
        Device to load the model on. If None, uses the default.
    """

    output_type = "epsilon"

    def __init__(
        self,
        model_name: str,
        cfg_scale: float = 0.0,
        min_cfg_scale: float | None = None,
        max_cfg_scale: float | None = None,
        prompts: list[str] | None = None,
        device: torch.device | None = None,
    ):
        super().__init__(device)

        pipe = StableDiffusionPipeline.from_pretrained(model_name).to(self.device)
        self.pipe = pipe
        self.unet: UNet2DModel = pipe.unet

        self.channels: int = self.unet.config["in_channels"]
        self.dim: int = self.unet.config["sample_size"]

        pipe.scheduler.alphas_cumprod = pipe.scheduler.alphas_cumprod.to(device)
        self._scheduler = DiffusionScheduler(pipe.scheduler.alphas_cumprod)

        self.cfg_scale = cfg_scale
        self.min_cfg_scale = min_cfg_scale
        self.max_cfg_scale = max_cfg_scale

        if (self.min_cfg_scale is not None) and (self.max_cfg_scale is not None):
            assert self.max_cfg_scale >= self.min_cfg_scale, (
                "max_cfg_scale must be greater than or equal to min_cfg_scale."
            )

            if self.min_cfg_scale <= 0.0 and self.max_cfg_scale <= 0.0:
                self.min_cfg_scale = None
                self.max_cfg_scale = None
                self.cfg_scale = 0.0

        self.p_dropout = 0.1

        if prompts is None:
            with open_text(
                "reward_actflow.images.base_models", "refl_data.json", encoding="utf-8"
            ) as f:
                refl_data = json.load(f)

            prompts = [item["text"] for item in refl_data]

        self.prompts = prompts

    @property
    def scheduler(self) -> DiffusionScheduler:
        return self._scheduler

    @property
    def do_cfg(self) -> bool:
        return (self.cfg_scale > 0.0) or (
            self.min_cfg_scale is not None and self.max_cfg_scale is not None
        )

    def sample_p0(self, n: int, **kwargs) -> tuple[DDTensor, dict[str, Any]]:
        """Sample n latent datapoints from the base distribution :math:`p_0`."""
        prompt = kwargs.get("prompt", None)
        cfg_scale = kwargs.get("cfg_scale", None)

        if prompt is None:
            prompt = random.choices(self.prompts, k=n)

        if cfg_scale is None:
            cfg_scale = torch.full((n,), self.cfg_scale)
            if (self.min_cfg_scale is not None) and (self.max_cfg_scale is not None):
                cfg_scale = (
                    torch.rand(n) * (self.max_cfg_scale - self.min_cfg_scale)
                    + self.min_cfg_scale
                )

        if isinstance(cfg_scale, float) or len(cfg_scale) == 1:
            cfg_scale = cfg_scale * torch.ones((n,))

        if not isinstance(prompt, list):
            prompt = [prompt] * n

        if len(prompt) != n:
            raise ValueError(
                f"The prompt must be a list of strings with length equal to the batch size, got length {len(prompt)}."
            )

        if len(cfg_scale) != n:
            raise ValueError(
                f"The cfg scale tensor must be length equal to the batch size, got length {len(cfg_scale)}."
            )

        return (
            DDTensor(
                torch.randn(n, self.channels, self.dim, self.dim, device=self.device)
            ),
            {"prompt": prompt, "cfg_scale": cfg_scale},
        )

    def preprocess(self, x: DDTensor, **kwargs) -> tuple[DDTensor, dict[str, Any]]:
        """Encode the prompt (if provided instead of encoder_hidden_states)."""
        n = len(x)
        encoder_hidden_states = kwargs.get("encoder_hidden_states")
        prompt = kwargs.get("prompt")
        neg_prompt = kwargs.get("neg_prompt", "")

        if encoder_hidden_states is None and prompt is None:
            raise ValueError(
                "Either encoder_hidden_states or a prompt needs to be provided."
            )

        if encoder_hidden_states is None:
            if not isinstance(prompt, list):
                prompt = [prompt] * n

            assert len(prompt) == n, (
                "The prompt must be a list of strings with length equal to the batch size."
            )

            if neg_prompt is not None:
                if not isinstance(neg_prompt, list):
                    neg_prompt = [neg_prompt] * n

                if len(neg_prompt) != n:
                    raise ValueError(
                        "The negative prompt must be a list of strings with length equal to the ",
                        "batch size.",
                    )

            prompt_embeds, neg_prompt_embeds = self.pipe.encode_prompt(
                prompt, self.device, 1, self.do_cfg, neg_prompt
            )
            encoder_hidden_states = prompt_embeds
            if neg_prompt_embeds is not None:
                encoder_hidden_states = torch.cat(
                    [prompt_embeds, neg_prompt_embeds], dim=0
                )
            else:
                encoder_hidden_states = prompt_embeds

        return x, {
            "encoder_hidden_states": encoder_hidden_states,
            "prompt": prompt,
            "neg_prompt": neg_prompt,
            "cfg_scale": kwargs["cfg_scale"].to(x.device),
        }

    def postprocess(self, x: DDTensor) -> DDTensor:
        """Decode the images from the latent space."""
        x = x / self.pipe.vae.config.scaling_factor
        decoded = torch.cat(
            [self.pipe.vae.decode(xi.unsqueeze(0)).sample for xi in x.data], dim=0
        )

        decoded = (decoded + 1) / 2
        decoded = decoded.clamp(0, 1)

        return DDTensor(decoded)

    def forward(self, x: DDTensor, t: torch.Tensor, **kwargs) -> DDTensor:
        r"""Forward pass, outputting :math:`\epsilon(x_t, t)`."""
        x_tensor = x.data
        k = self.scheduler.model_input(t)

        encoder_hidden_states = kwargs.get("encoder_hidden_states")
        if encoder_hidden_states is None:
            raise ValueError("encoder_hidden_states must be provided in kwargs.")

        if (
            not self.do_cfg
            or self.training
            or encoder_hidden_states.shape[0] == x_tensor.shape[0]
        ):
            out = cast(
                "torch.Tensor", self.unet(x_tensor, k, encoder_hidden_states).sample
            )
            return DDTensor(out)

        x_tensor = torch.cat([x_tensor, x_tensor], dim=0)
        k = torch.cat([k, k], dim=0)
        out = cast("torch.Tensor", self.unet(x_tensor, k, encoder_hidden_states).sample)

        # Classifier-free guidance
        cond, uncond = out.chunk(2)
        cfg_scale = append_dims(kwargs["cfg_scale"], cond.ndim)
        out = (cfg_scale + 1) * cond - cfg_scale * uncond

        return DDTensor(out)

    def train_loss(
        self,
        x1: DDTensor,
        xt: DDTensor | None = None,
        t: torch.Tensor | None = None,
        pred: DDTensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Add prompt dropout for training when CFG is enabled."""
        if self.do_cfg and self.p_dropout > 0:
            cond_embeds = kwargs["encoder_hidden_states"]
            uncond_embeds, _ = self.pipe.encode_prompt(
                prompt="",
                device=x1.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
            uncond_embeds = uncond_embeds.expand(cond_embeds.shape[0], -1, -1)

            mask = (
                torch.rand(cond_embeds.shape[0], device=cond_embeds.device)
                < self.p_dropout
            )
            mask = mask[:, None, None]

            kwargs["encoder_hidden_states"] = torch.where(
                mask, uncond_embeds, cond_embeds
            )

        return super().train_loss(x1, xt, t, pred, **kwargs)


@base_model_registry.register("images/sd2")
class SD2BaseModel(StableDiffusionBaseModel):
    """Pre-trained 512x512 Stable Diffusion 2 base."""

    def __init__(
        self,
        cfg_scale: float = 0.0,
        min_cfg_scale: float | None = None,
        max_cfg_scale: float | None = None,
        prompts: list[str] | None = None,
        device: torch.device | None = None,
    ):
        super().__init__(
            "PeggyWang/stable-diffusion-2-base",
            cfg_scale=cfg_scale,
            min_cfg_scale=min_cfg_scale,
            max_cfg_scale=max_cfg_scale,
            prompts=prompts,
            device=device,
        )


@base_model_registry.register("images/sd1.5")
class SD15BaseModel(StableDiffusionBaseModel):
    """Pre-trained 512x512 Stable Diffusion 1.5."""

    def __init__(
        self,
        cfg_scale: float = 0.0,
        min_cfg_scale: float | None = None,
        max_cfg_scale: float | None = None,
        prompts: list[str] | None = None,
        device: torch.device | None = None,
    ):
        super().__init__(
            "sd-legacy/stable-diffusion-v1-5",
            cfg_scale=cfg_scale,
            min_cfg_scale=min_cfg_scale,
            max_cfg_scale=max_cfg_scale,
            prompts=prompts,
            device=device,
        )
