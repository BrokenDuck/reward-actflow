"""Pre-trained base model for Diffusion Transformer."""

from typing import Any, cast

import torch
from diffusers.pipelines.dit.pipeline_dit import DiTPipeline

from diffusiongym import BaseModel, DDTensor, DiffusionScheduler
from diffusiongym.registry import base_model_registry


@base_model_registry.register("images/dit")
class DiTBaseModel(BaseModel[DDTensor]):
    """Pre-trained 256x256 ImageNet transformer diffusion model.

    Uses the `facebookresearch/DiT-XL-2-256` model from the `diffusers` library.
    """

    output_type = "epsilon"

    def __init__(self, cfg_scale: float = 0.0, device: torch.device | None = None):
        super().__init__(device)

        pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256").to(device)
        self.pipe = pipe
        self.transformer = pipe.transformer

        self.channels: int = self.transformer.config["in_channels"]
        self.dim: int = self.transformer.config["sample_size"]

        pipe.scheduler.alphas_cumprod = pipe.scheduler.alphas_cumprod.to(device)
        self._scheduler = DiffusionScheduler(pipe.scheduler.alphas_cumprod)

        self.cfg_scale = cfg_scale

    @property
    def scheduler(self) -> DiffusionScheduler:
        """Scheduler used for sampling."""
        return self._scheduler

    @property
    def do_cfg(self) -> bool:
        return self.cfg_scale > 0.0

    def sample_p0(self, n: int, **kwargs) -> tuple[DDTensor, dict[str, Any]]:
        """Sample n latent datapoints from the base distribution :math:`p_0`."""
        class_labels = kwargs.get("class_labels", None)

        if class_labels is None:
            class_labels = torch.randint(0, 1000, (n,), device=self.device)

        if isinstance(class_labels, int):
            class_labels = torch.tensor([class_labels] * n, device=self.device)

        if isinstance(class_labels, list):
            class_labels = torch.tensor(class_labels, device=self.device)

        if len(class_labels) != n:
            raise ValueError(f"The class_label must be a list of integers with length equal to the batch size, got length {len(class_labels)}.")

        return (
            DDTensor(torch.randn(n, self.channels, self.dim, self.dim, device=self.device)),
            {"class_labels": class_labels},
        )

    def preprocess(self, x: DDTensor, **kwargs) -> tuple[DDTensor, dict[str, Any]]:
        """Encode the prompt (if provided instead of encoder_hidden_states)."""
        class_labels = kwargs.get("class_labels", None)
        if class_labels is None:
            raise ValueError("class_labels must be provided in kwargs.")

        return x, {"class_labels": class_labels}

    def postprocess(self, x: DDTensor) -> DDTensor:
        """Decode the images from the latent space."""
        x = x / self.pipe.vae.config.scaling_factor
        decoded = torch.cat([self.pipe.vae.decode(xi.unsqueeze(0)).sample for xi in x.data], dim=0)

        decoded = (decoded + 1) / 2
        decoded = decoded.clamp(0, 1)

        return DDTensor(decoded)

    def forward(self, x: DDTensor, t: torch.Tensor, **kwargs) -> DDTensor:
        r"""Forward pass, outputting :math:`\epsilon(x_t, t)`."""
        x_tensor = x.data
        k = self.scheduler.model_input(t)

        class_labels = kwargs.get("class_labels")
        if class_labels is None:
            raise ValueError("class_labels must be provided in kwargs.")

        if not self.do_cfg or self.training:
            out = cast("torch.Tensor", self.transformer(x_tensor, k, class_labels).sample[:, :4])
            return DDTensor(out)

        x_tensor = torch.cat([x_tensor, x_tensor], dim=0)
        k = torch.cat([k, k], dim=0)
        class_null = torch.zeros_like(class_labels)
        class_labels = torch.cat([class_labels, class_null], dim=0)
        out = cast("torch.Tensor", self.transformer(x_tensor, k, class_labels).sample)

        # Classifier-free guidance
        cond, uncond = out[:, :4].chunk(2)
        out = (self.cfg_scale + 1) * cond - self.cfg_scale * uncond

        return DDTensor(out)
