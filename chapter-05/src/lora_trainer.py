"""LoRA training loop for CogVideoX.

Companion module for Chapter 5, "Fine-Tuning for Specific Video Tasks",
from the book "Video Generation with AI". Implements the production LoRA
trainer from the CogVideo repository: component separation (VAE, text
encoder, transformer), selective freezing (only the transformer gets
LoRA adapters), and the core diffusion training step where videos are
encoded to latent space, noised according to the diffusion schedule,
and the model learns to predict that noise. Only the LoRA parameters
update during backpropagation; the base model stays frozen.
"""

from typing import Dict

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel, DDPMScheduler
from transformers import T5EncoderModel

from lora_config import CogVideoLoRAConfig, apply_lora_to_model


class CogVideoLoRATrainer:
    """Production LoRA trainer from CogVideo repository."""

    def __init__(self,
                 model_path: str,
                 output_dir: str,
                 rank: int = 128,
                 learning_rate: float = 2e-5):

        # Load base model components
        self.transformer = CogVideoXTransformer3DModel.from_pretrained(
            model_path, subfolder="transformer"
        )
        self.vae = AutoencoderKLCogVideoX.from_pretrained(
            model_path, subfolder="vae"
        )
        self.text_encoder = T5EncoderModel.from_pretrained(
            model_path, subfolder="text_encoder"
        )

        # Apply LoRA to transformer
        lora_config = CogVideoLoRAConfig.get_lora_config(
            model_size="2b", rank=rank
        )
        self.transformer = apply_lora_to_model(
            self.transformer, lora_config
        )

        # Freeze VAE and text encoder
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        # Setup optimizer (only LoRA parameters)
        lora_params = filter(
            lambda p: p.requires_grad,
            self.transformer.parameters()
        )
        self.optimizer = torch.optim.AdamW(
            lora_params, lr=learning_rate
        )

        # Initialize scheduler for noise addition
        self.scheduler = DDPMScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """Encode video frames to latent space using VAE."""
        # video shape: [B, C, F, H, W]
        with torch.no_grad():
            latent = self.vae.encode(video).latent_dist.sample()
            latent = latent * self.vae.config.scaling_factor
        return latent

    def training_step(self, batch: Dict) -> torch.Tensor:
        """Single training step with LoRA fine-tuning."""

        # Encode video to latent space
        video_latents = self.encode_video(batch["video"])

        # Encode text prompt
        prompt_embeds = self.text_encoder(
            batch["input_ids"]
        )[0]

        # Sample timestep for diffusion
        timesteps = torch.randint(
            0, 1000, (video_latents.shape[0],)
        ).to(video_latents.device)

        # Add noise to latents
        noise = torch.randn_like(video_latents)
        noisy_latents = self.scheduler.add_noise(
            video_latents, noise, timesteps
        )

        # Predict noise with LoRA-adapted transformer
        model_pred = self.transformer(
            hidden_states=noisy_latents,
            encoder_hidden_states=prompt_embeds,
            timestep=timesteps,
        ).sample

        # Compute loss
        loss = F.mse_loss(model_pred, noise)

        # Backward pass (only LoRA parameters update)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(
            self.transformer.parameters(), max_norm=1.0
        )

        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss
