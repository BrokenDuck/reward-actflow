from dataclasses import dataclass

from pathlib import Path
import argparse


@dataclass(frozen=True)
class ActivePretrainingConfig:
    # Experiment directory
    folder: Path

    # Feature extraction and Gaussian Process
    feat_timestep: float = 0.9
    gp_kernel: str = "rbf"
    gp_lengthscale: float = 0.1

    # Uncertainty reward and uncertainty sampling algorithm
    uncertainty_weight : float = 100.0
    reward_opt_algo: str = "dps"
    
    # Fine-tuning
    ft_min_dataset_size: int = 64
    ft_batch_size: int = 64
    ft_accumulate_steps: int = 1
    ft_steps: int = 500
    ft_lr: float = 1e-4
    ft_weight_decay: float = 0.0

    # Sampling and evaluation
    num_steps: int = 100
    eval_samples: int = 0
    eval_batch_size: int = 64
    eval_every: int = 10
    video_fps: int = 4
    
    # Flags
    no_uncertainty: bool = False
    no_verifier: bool = False

    def __post_init__(self):
        # Create experiment directory if it doesn't exist
        self.folder.mkdir(parents=True, exist_ok=True)

        # Validation
        if not (0 <= self.feat_timestep <= 1):
            raise ValueError(f"feat_timestep must be in [0, 1], got {self.feat_timestep}")

        if self.uncertainty_weight < 0:
            raise ValueError(f"uncertainty_weight cannot be negative, got {self.uncertainty_weight}")

        if self.gp_lengthscale < 0:
            raise ValueError(f"gp_lengthscale cannot be negative, got {self.gp_lengthscale}")

        allowed_algos = { "dps", "svdd" }
        if self.reward_opt_algo not in allowed_algos:
            raise ValueError(f"reward_opt_algo must be one of {allowed_algos}")

        allowed_kernels = { "rbf", "linear" }
        if self.gp_kernel not in allowed_kernels:
            raise ValueError(f"gp_kernel must be one of {allowed_kernels}")

    @staticmethod
    def construct_from_args(args: argparse.Namespace | dict) -> "ActivePretrainingConfig":
        if isinstance(args, argparse.Namespace):
            args = vars(args)

        # Map argparse flags to config names
        name_mapping = { "dir": "folder" }
        
        # We only take keys that exist in the dataclass fields
        config_fields = {f.name for f in ActivePretrainingConfig.__dataclass_fields__.values()}
        clean_kwargs = {}

        for key, value in args.items():
            # Check if the key needs a rename
            target_key = name_mapping.get(key, key)
            
            # Only add if it's a valid field in our config
            if target_key in config_fields:
                clean_kwargs[target_key] = value

        return ActivePretrainingConfig(**clean_kwargs)
