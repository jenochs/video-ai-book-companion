"""Diffusion noise schedule, training loop, and classifier-free guidance.

Accompanies Chapter 3 of "Video Generation with AI" (O'Reilly) by Joseph Enochs.

NoiseScheduler implements the minimal forward process (add noise according to
a linear beta schedule) and a DDPM-style reverse step. DiffusionTrainer is the
chapter's minimal training loop: sample random timesteps, add noise, predict
that noise, and optimize a mean squared error objective.
"""

import torch
import torch.nn.functional as F


class NoiseScheduler:
    """Minimal linear-beta noise schedule for diffusion training and sampling.

    The chapter's training loop calls `add_noise` and reads `num_timesteps`;
    this class provides both, plus a basic DDPM reverse `step` used in the
    notebook's generation demo. Production systems use faster samplers and
    improved schedules, but the core mechanics are the same.
    """

    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        return self

    def add_noise(self, clean, noise, timesteps):
        """Forward process: corrupt clean latents at the given timesteps."""
        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus = (1.0 - self.alphas_cumprod[timesteps]) ** 0.5

        # Broadcast over [batch, channels, frames, height, width]
        while sqrt_alpha_prod.dim() < clean.dim():
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)

        return sqrt_alpha_prod * clean + sqrt_one_minus * noise

    @torch.no_grad()
    def step(self, noise_pred, t, x):
        """One DDPM reverse step: remove predicted noise at integer timestep t."""
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alphas_cumprod[t]

        # Predicted mean of the previous (less noisy) latent
        coef = beta_t / (1.0 - alpha_bar_t) ** 0.5
        mean = (x - coef * noise_pred) / alpha_t ** 0.5

        if t > 0:
            mean = mean + (beta_t ** 0.5) * torch.randn_like(x)
        return mean


class DiffusionTrainer:
    """Minimal diffusion training loop from the chapter."""

    def __init__(self, model, noise_scheduler, optimizer, device='cuda'):
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.optimizer = optimizer
        self.device = device

    def train_step(self, batch):
        # batch contains: latent_video, text_embed
        latent_video = batch['latent_video'].to(self.device)
        text_embed = batch['text_embed'].to(self.device)
        batch_size = latent_video.shape[0]

        # Sample random timesteps for each video in batch
        timesteps = torch.randint(
            0, self.noise_scheduler.num_timesteps,
            (batch_size,), device=self.device,
        ).long()

        # Sample noise to add to the latents
        noise = torch.randn_like(latent_video)

        # Add noise to the latent videos according to the noise schedule
        noisy_latents = self.noise_scheduler.add_noise(
            latent_video, noise, timesteps
        )

        # Predict the noise with the model
        noise_pred = self.model(
            noisy_latents,
            timesteps,
            encoder_hidden_states=text_embed,
        )

        # Calculate MSE loss between predicted and actual noise
        loss = F.mse_loss(noise_pred, noise)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        return loss.item()


def classifier_free_guidance(noise_uncond, noise_text, guidance_scale=7.5):
    """Combine conditional and unconditional predictions.

    Implements the chapter's formula: the unconditional prediction plus a
    scaled difference toward the text-conditioned prediction. Larger guidance
    scales sharpen prompt adherence at the cost of diversity.
    """
    return noise_uncond + guidance_scale * (noise_text - noise_uncond)
