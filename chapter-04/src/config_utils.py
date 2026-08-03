"""Configuration management for video generation training.

Accompanies Chapter 4 of "Video Generation with AI" (O'Reilly) by Joseph Enochs.

Implements the Latte-style configuration system from the "Configuration-Driven
Training" section: hierarchical YAML files loaded with OmegaConf, with support
for command-line overrides, variable interpolation, and immutable configs.
"""

import argparse
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

# Default configuration mirroring config/train_video.yaml from the chapter.
# This mirrors the configuration structure used in Latte-style production systems.
DEFAULT_CONFIG_YAML = """
model:
  name: "Latte-XL/2"
  image_size: 256
  num_frames: 16
  frame_interval: 3
  latent_size: 32
  num_layers: 24
  num_heads: 16
  hidden_size: 1152
  patch_size: 2
  # VAE configuration
  vae_model: "stabilityai/sd-vae-ft-ema"
  # Text encoder
  text_encoder: "t5-base"

training:
  batch_size: 4
  gradient_accumulation_steps: 8
  learning_rate: 1e-4
  warmup_steps: 10000
  max_steps: 1000000
  # Optimization
  optimizer: "adamw"
  weight_decay: 0.01
  beta1: 0.9
  beta2: 0.999
  epsilon: 1e-8
  # Gradient management
  gradient_clip: 1.0
  mixed_precision: true
  tf32: true  # Latte's TF32 optimization
  # EMA
  use_ema: true
  ema_decay: 0.9999

dataset:
  name: "ucf101"
  data_path: "/path/to/ucf101"
  cache_dir: "/path/to/cache"
  num_workers: 8
  # Video sampling
  sample_rate: 4
  clip_duration: 2.0
  # Augmentation
  random_flip: true
  color_jitter: 0.1

diffusion:
  timesteps: 1000
  beta_schedule: "linear"
  beta_start: 0.0001
  beta_end: 0.02
  # Loss configuration
  loss_type: "mse"
  use_min_snr: false  # Start simple like Latte

logging:
  output_dir: "./outputs"
  logging_steps: 100
  save_steps: 5000
  validation_steps: 1000
  tensorboard: true
  wandb:
    project: "video-diffusion"
    entity: "my-team"
"""


def load_config(config_path: str, overrides: list = None) -> DictConfig:
    """
    Load configuration with OmegaConf, supporting overrides.
    This mirrors the pattern used for Latte-style training.
    """
    # Load base configuration
    base_config = OmegaConf.load(config_path)

    # Apply command-line overrides if provided
    if overrides:
        override_config = OmegaConf.from_dotlist(overrides)
        config = OmegaConf.merge(base_config, override_config)
    else:
        config = base_config

    # Resolve interpolations (${model.hidden_size} references)
    OmegaConf.resolve(config)

    # Make configuration immutable to prevent accidental changes
    OmegaConf.set_struct(config, True)

    return config


def setup_training_environment(config: DictConfig) -> Path:
    """
    Prepare the environment described by a training configuration.

    Enables TF32 when requested (Latte's optimization), creates the output
    directory, and saves a copy of the resolved config for reproducibility.
    Model, dataset, and optimizer construction live in the sibling modules
    (video_dataset.py and trainer.py).
    """
    # Enable TF32 for better performance (Latte optimization)
    if config.training.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Create output directory
    output_dir = Path(config.logging.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration for reproducibility
    OmegaConf.save(config, output_dir / "config.yaml")

    return output_dir


def parse_args():
    """Command-line interface matching Latte's style."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/train_video.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--override",
        nargs="+",
        help="Override config values (e.g., training.batch_size=8)",
    )
    return parser.parse_args()
