"""Latte-style transformer block and a small assembled video generation model.

Accompanies Chapter 3 of "Video Generation with AI" (O'Reilly) by Joseph Enochs.

LatteTransformerBlock combines spatial attention, cross-attention with text,
temporal attention, a feed-forward network, and AdaLN timestep modulation in
the ordering discussed in the chapter. LatteVideoModel wires patch embedding,
timestep embedding, a stack of blocks, and an unpatchify head into a compact
noise-prediction model suitable for the companion notebook's toy experiments.
"""

import math

import torch
import torch.nn as nn

from attention_blocks import (
    AdaLN,
    CrossAttention,
    SpatialAttentionBlock,
    TemporalAttentionBlock,
)
from patch_embedding import VideoPatchEmbed


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding followed by a small MLP.

    Reuses the sinusoidal strategy from positional encoding, applied to
    diffusion timesteps instead of spatial positions. Nearby timesteps get
    similar embeddings, which helps the model generalize across the schedule.
    """

    def __init__(self, dim=256):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, timesteps):
        # timesteps: [batch] integer tensor
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / half
        )
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


class LatteTransformerBlock(nn.Module):
    """Complete transformer block with spatial, temporal, and cross attention."""

    def __init__(self, dim, num_heads=8, num_frames=16,
                 num_patches_per_frame=256, context_dim=768,
                 timestep_dim=256, use_adaln=True):
        super().__init__()
        self.use_adaln = use_adaln

        # Attention layers
        self.spatial_attn = SpatialAttentionBlock(dim, num_heads, num_frames)
        self.temporal_attn = TemporalAttentionBlock(
            dim, num_heads, num_frames, num_patches_per_frame
        )
        self.cross_attn = CrossAttention(dim, context_dim, num_heads)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

        # AdaLN modulation (if enabled)
        if use_adaln:
            self.adaln = AdaLN(dim, timestep_dim)

    def forward(self, x, timestep_embed, text_embed):
        # Apply AdaLN modulation if enabled
        if self.use_adaln:
            x = self.adaln(x, timestep_embed)

        # Spatial attention within frames
        x = self.spatial_attn(x)

        # Cross-attention with text
        x = self.cross_attn(x, text_embed)

        # Temporal attention across frames
        x = self.temporal_attn(x)

        # Feed-forward network
        x = x + self.ffn(x)

        return x


class LatteVideoModel(nn.Module):
    """Small Latte-style noise-prediction model for latent video.

    Wires together the chapter's components: patch embedding with dual
    positional encoding, timestep embedding, a stack of alternating-attention
    transformer blocks, and a linear head that projects tokens back to latent
    patches (unpatchify). Sized for notebook-scale experiments, not for
    production training.
    """

    def __init__(self, latent_channels=4, image_size=32, patch_size=2,
                 num_frames=16, dim=768, depth=4, num_heads=8,
                 context_dim=768, timestep_dim=256):
        super().__init__()
        self.latent_channels = latent_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_frames = num_frames

        self.patch_embed = VideoPatchEmbed(
            patch_size=patch_size,
            in_channels=latent_channels,
            embed_dim=dim,
            num_frames=num_frames,
            image_size=image_size,
        )
        self.time_embed = TimestepEmbedding(timestep_dim)

        num_patches_per_frame = self.patch_embed.num_patches_per_frame
        self.blocks = nn.ModuleList([
            LatteTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                num_frames=num_frames,
                num_patches_per_frame=num_patches_per_frame,
                context_dim=context_dim,
                timestep_dim=timestep_dim,
            )
            for _ in range(depth)
        ])

        self.final_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(
            dim, patch_size * patch_size * latent_channels
        )

    def unpatchify(self, x):
        """Invert the patch reshaping: token sequence back to latent video."""
        B = x.shape[0]
        p = self.patch_size
        T = self.num_frames
        hp = wp = self.image_size // p
        C = self.latent_channels

        # Tokens are ordered [T, hp, wp] with patch dim ordered [p1, p2, C]
        x = x.reshape(B, T, hp, wp, p, p, C)
        x = x.permute(0, 6, 1, 2, 4, 3, 5)  # [B, C, T, hp, p1, wp, p2]
        x = x.reshape(B, C, T, hp * p, wp * p)
        return x

    def forward(self, x, timesteps, encoder_hidden_states):
        # x: noisy latents [batch, latent_channels, frames, height, width]
        # timesteps: [batch] integer tensor
        # encoder_hidden_states: text embeddings [batch, seq_len, context_dim]
        t_emb = self.time_embed(timesteps)
        tokens = self.patch_embed(x)

        for block in self.blocks:
            tokens = block(tokens, t_emb, encoder_hidden_states)

        tokens = self.head(self.final_norm(tokens))
        return self.unpatchify(tokens)
