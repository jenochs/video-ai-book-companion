"""Spatial, temporal, and cross attention blocks, plus adaptive layer norm.

Accompanies Chapter 3 of "Video Generation with AI" (O'Reilly) by Joseph Enochs.

These blocks implement Latte's factorized attention strategy. Spatial
attention relates patches within each frame, temporal attention tracks each
spatial position across frames, cross-attention injects text conditioning,
and AdaLN modulates features based on the diffusion timestep.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttentionBlock(nn.Module):
    """Temporal dimension: how frames relate to each other."""

    def __init__(self, dim, num_heads=8, num_frames=16,
                 num_patches_per_frame=256, causal=False):
        super().__init__()
        self.num_heads = num_heads
        self.num_frames = num_frames
        self.num_patches_per_frame = num_patches_per_frame
        self.causal = causal  # for autoregressive generation
        self.scale = (dim // num_heads) ** -0.5

        # Multi-head attention components
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

        # Create causal mask if needed
        if causal:
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(num_frames, num_frames)),
            )

    def forward(self, x, text_embed=None):
        # x shape: [batch, num_patches_total, dim]
        B, N, C = x.shape

        # Reshape to separate temporal and spatial dimensions:
        # group patches by spatial position across frames
        x = x.reshape(B, self.num_frames, self.num_patches_per_frame, C)
        x = x.permute(0, 2, 1, 3)  # [B, patches_per_frame, frames, C]
        x = x.reshape(B * self.num_patches_per_frame, self.num_frames, C)

        # Apply layer norm and save for residual
        shortcut = x
        x = self.norm(x)

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(
            B * self.num_patches_per_frame, self.num_frames,
            3, self.num_heads, C // self.num_heads,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Apply causal mask if needed (for autoregressive generation)
        if self.causal:
            mask_value = -torch.finfo(attn.dtype).max
            mask = self.causal_mask[:self.num_frames, :self.num_frames]
            mask = mask.reshape(1, 1, self.num_frames, self.num_frames)
            attn = attn.masked_fill(mask == 0, mask_value)

        # Apply softmax and attention
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(
            B * self.num_patches_per_frame, self.num_frames, C
        )

        # Project and add residual
        x = self.proj(x)
        x = shortcut + x

        # Reshape back to original format
        x = x.reshape(B, self.num_patches_per_frame, self.num_frames, C)
        x = x.permute(0, 2, 1, 3)  # [B, frames, patches_per_frame, C]
        x = x.reshape(B, N, C)

        return x


class SpatialAttentionBlock(nn.Module):
    """Spatial dimension: how patches relate within a frame."""

    def __init__(self, dim, num_heads=8, num_frames=16):
        super().__init__()
        self.num_heads = num_heads
        self.num_frames = num_frames
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x shape: [batch, num_patches_total, dim]
        B, N, C = x.shape
        patches_per_frame = N // self.num_frames

        # Reshape to process each frame independently
        x = x.reshape(B * self.num_frames, patches_per_frame, C)

        # Standard multi-head self-attention
        shortcut = x
        x = self.norm(x)

        qkv = self.qkv(x).reshape(
            B * self.num_frames, patches_per_frame,
            3, self.num_heads, C // self.num_heads,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention computation
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(
            B * self.num_frames, patches_per_frame, C
        )
        x = self.proj(x)
        x = shortcut + x

        # Reshape back to full sequence
        x = x.reshape(B, N, C)
        return x


class CrossAttention(nn.Module):
    """Cross-attention: video patches query text embeddings."""

    def __init__(self, dim, context_dim=768, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        # Separate projections for Q (from video) and K, V (from text)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(context_dim, dim, bias=False)
        self.to_v = nn.Linear(context_dim, dim, bias=False)

        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(context_dim)

    def forward(self, x, context):
        # x: video features [batch, num_patches, dim]
        # context: text embeddings [batch, seq_len, context_dim]
        B, N, C = x.shape

        # Normalize inputs
        shortcut = x
        x = self.norm(x)
        context = self.context_norm(context)

        # Create queries from video features
        q = self.to_q(x).reshape(B, N, self.num_heads, C // self.num_heads)
        q = q.permute(0, 2, 1, 3)  # [B, heads, N, head_dim]

        # Create keys and values from text embeddings
        k = self.to_k(context).reshape(B, -1, self.num_heads, C // self.num_heads)
        k = k.permute(0, 2, 1, 3)  # [B, heads, seq_len, head_dim]

        v = self.to_v(context).reshape(B, -1, self.num_heads, C // self.num_heads)
        v = v.permute(0, 2, 1, 3)  # [B, heads, seq_len, head_dim]

        # Compute attention scores: Q @ K^T
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values: attn @ V
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # Project and add residual
        out = self.proj(out)
        out = shortcut + out

        return out


class AdaLN(nn.Module):
    """Adaptive layer normalization for timestep conditioning."""

    def __init__(self, dim, timestep_dim=256):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)

        # MLP to generate scale and shift from timestep
        self.timestep_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(timestep_dim, dim * 2),
        )

    def forward(self, x, timestep_embed):
        # Generate scale (gamma) and shift (beta) from timestep
        adaln_params = self.timestep_mlp(timestep_embed)
        adaln_params = adaln_params.unsqueeze(1)  # Add sequence dimension
        gamma, beta = adaln_params.chunk(2, dim=-1)

        # Apply adaptive normalization
        x = self.norm(x)
        x = (1 + gamma) * x + beta

        return x
