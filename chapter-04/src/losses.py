"""Loss functions for video diffusion training.

Accompanies Chapter 4 of "Video Generation with AI" (O'Reilly) by Joseph Enochs.

Implements the "Loss Functions and Metrics" section: Latte's simple MSE
objective plus the optional Min-SNR weighting that CogVideo pioneered.
The chapter's analysis of production models (Table 4-1) shows that MSE
variants power every leading system, so start simple and only add
complexity when a specific problem demands it.
"""

import torch
import torch.nn.functional as F


def training_step(model, diffusion, x_latent, device, model_kwargs=None):
    """
    Latte's actual training step from train.py.
    The entire loss is just MSE between predicted and target noise.
    """
    model_kwargs = model_kwargs or {}

    # Sample random timesteps for each video in the batch
    t = torch.randint(0, diffusion.num_timesteps, (x_latent.shape[0],), device=device)

    # Get loss from diffusion.training_losses (which internally uses MSE)
    loss_dict = diffusion.training_losses(model, x_latent, t, model_kwargs)

    # Extract the loss - it is just MSE, nothing more
    loss = loss_dict["loss"].mean()

    return loss


def linear_alphas_cumprod(num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
    """Cumulative alpha products for a linear beta schedule.

    Provides the alphas_cumprod tensor that Min-SNR weighting needs.
    """
    betas = torch.linspace(beta_start, beta_end, num_timesteps)
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0)


def compute_loss_with_min_snr(pred_noise, target_noise, timesteps, alphas_cumprod, gamma=5.0):
    """
    MSE loss with Min-SNR weighting to balance learning.
    Only use if simple MSE shows poor detail generation.
    """
    # Start with base MSE
    mse_loss = F.mse_loss(pred_noise, target_noise, reduction='none')

    # Calculate signal-to-noise ratio
    snr = alphas_cumprod[timesteps] / (1 - alphas_cumprod[timesteps] + 1e-8)

    # Apply Min-SNR weighting
    min_snr = torch.minimum(snr, torch.ones_like(snr) * gamma)
    weight = min_snr / snr

    # Weight the loss
    weighted_loss = mse_loss * weight.view(-1, 1, 1, 1, 1)

    return weighted_loss.mean()
