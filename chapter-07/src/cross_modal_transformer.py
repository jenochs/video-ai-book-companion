"""Unified audio-video transformer blocks for Chapter 7 of Video Generation with AI.

This module accompanies Chapter 7, "Audio-Visual Synchronization with
Cross-Modal Fusion". It implements a runnable, scaled-down version of the
LTX-2 unified DiT: the frozen Modality dataclass, AdaLN timestep
modulation, and the BasicAVTransformerBlock with bidirectional
cross-modal attention between a video stream and an audio stream.

The full LTX-2.3 model stacks 48 of these blocks at 4096 video dim and
2048 audio dim (19B transformer parameters). The versions here keep the
same structure and information flow at toy sizes so every step runs on a
CPU in seconds.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass(frozen=True)
class Modality:
    """Everything one modality carries through the transformer.

    Mirrors ltx_core/model/transformer/modality.py. The dataclass is
    frozen: fields never mutate in place. When a field must change, a new
    instance is created with dataclasses.replace(), which keeps the
    forward pass free of hidden state changes.
    """

    enabled: bool  # Whether this modality is active
    latent: Tensor  # [B, seq_len, channels]
    timesteps: Tensor  # [B, seq_len] per-token timesteps
    positions: Tensor  # [B, dims, seq_len, 2] position bounds
    context: Tensor  # Text embeddings for this modality
    context_mask: Tensor | None = None


class AdaLNModulation(nn.Module):
    """Adaptive Layer Normalization conditioning on the diffusion timestep.

    Produces six vectors per block: shift, scale, and gate for the
    attention branch and for the feedforward branch. The gates let the
    model change how strongly each layer contributes as denoising moves
    from coarse structure to fine detail.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)
        self.timestep_proj = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 6 * dim)
        )

    def forward(self, timestep_embedding: Tensor) -> tuple[Tensor, ...]:
        """Return (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp).

        Args:
            timestep_embedding: [B, dim] embedding of the current timestep.
        """
        b = timestep_embedding.shape[0]
        table = self.scale_shift_table[None]  # [1, 6, dim]
        emb = self.timestep_proj(timestep_embedding).view(b, 6, -1)
        return tuple((table + emb).unsqueeze(2).chunk(6, dim=1))


def timestep_embedding(timesteps: Tensor, dim: int, max_period: int = 10_000) -> Tensor:
    """Standard sinusoidal timestep embedding, shape [B, dim]."""
    half = dim // 2
    freqs = torch.exp(
        -torch.log(torch.tensor(float(max_period)))
        * torch.arange(half, dtype=torch.float32)
        / half
    ).to(timesteps.device)
    args = timesteps.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class BasicAVTransformerBlock(nn.Module):
    """One unified audio-video transformer block.

    Follows ltx_core/model/transformer/transformer.py. Each block holds
    two parallel paths, video and audio, each with self-attention, text
    cross-attention, cross-modal attention, and a feedforward network,
    all modulated by AdaLN. The cross-modal step is bidirectional: audio
    attends to video and video attends to audio, using the updated,
    text-conditioned states, so timing decisions are coordinated at
    every layer.
    """

    def __init__(
        self,
        video_dim: int = 128,
        audio_dim: int = 64,
        num_heads: int = 4,
        text_dim: int = 96,
        ff_mult: int = 4,
    ) -> None:
        super().__init__()

        def attn(q_dim: int, kv_dim: int) -> nn.MultiheadAttention:
            return nn.MultiheadAttention(
                embed_dim=q_dim,
                num_heads=num_heads,
                kdim=kv_dim,
                vdim=kv_dim,
                batch_first=True,
            )

        def ff(dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(dim, dim * ff_mult),
                nn.GELU(),
                nn.Linear(dim * ff_mult, dim),
            )

        # Video path components
        self.attn1 = attn(video_dim, video_dim)  # Video self-attention
        self.attn2 = attn(video_dim, text_dim)  # Video cross-attention (text)
        self.audio_to_video_attn = attn(video_dim, audio_dim)  # Audio -> Video
        self.ff = ff(video_dim)
        self.video_adaln = AdaLNModulation(video_dim)
        self.norm_v1 = nn.LayerNorm(video_dim, elementwise_affine=False)
        self.norm_v2 = nn.LayerNorm(video_dim)
        self.norm_v3 = nn.LayerNorm(video_dim)
        self.norm_v_ff = nn.LayerNorm(video_dim, elementwise_affine=False)

        # Audio path components
        self.audio_attn1 = attn(audio_dim, audio_dim)  # Audio self-attention
        self.audio_attn2 = attn(audio_dim, text_dim)  # Audio cross-attention (text)
        self.video_to_audio_attn = attn(audio_dim, video_dim)  # Video -> Audio
        self.audio_ff = ff(audio_dim)
        self.audio_adaln = AdaLNModulation(audio_dim)
        self.norm_a1 = nn.LayerNorm(audio_dim, elementwise_affine=False)
        self.norm_a2 = nn.LayerNorm(audio_dim)
        self.norm_a3 = nn.LayerNorm(audio_dim)
        self.norm_a_ff = nn.LayerNorm(audio_dim, elementwise_affine=False)

    @staticmethod
    def _modulated(norm: nn.LayerNorm, x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        return norm(x) * (1 + scale.squeeze(1)) + shift.squeeze(1)

    def forward(
        self,
        video: Tensor,
        audio: Tensor,
        video_context: Tensor,
        audio_context: Tensor,
        video_t_emb: Tensor,
        audio_t_emb: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Run one block. All tensors are [B, seq, dim] shaped.

        Returns the updated (video, audio) hidden states.
        """
        vs_msa, vsc_msa, vg_msa, vs_mlp, vsc_mlp, vg_mlp = self.video_adaln(video_t_emb)
        as_msa, asc_msa, ag_msa, as_mlp, asc_mlp, ag_mlp = self.audio_adaln(audio_t_emb)

        # 1. Self-attention with AdaLN modulation (paths are independent here)
        v_in = self._modulated(self.norm_v1, video, vs_msa, vsc_msa)
        video = video + vg_msa.squeeze(1) * self.attn1(v_in, v_in, v_in)[0]

        a_in = self._modulated(self.norm_a1, audio, as_msa, asc_msa)
        audio = audio + ag_msa.squeeze(1) * self.audio_attn1(a_in, a_in, a_in)[0]

        # 2. Text cross-attention for both modalities
        video = video + self.attn2(self.norm_v2(video), video_context, video_context)[0]
        audio = (
            audio
            + self.audio_attn2(self.norm_a2(audio), audio_context, audio_context)[0]
        )

        # 3. Bidirectional cross-modal attention on text-conditioned states.
        # Audio queries attend to video keys/values, and video queries
        # attend to audio keys/values. This is where synchronization lives.
        video_kv = self.norm_v3(video)
        audio_kv = self.norm_a3(audio)
        audio = audio + self.video_to_audio_attn(audio_kv, video_kv, video_kv)[0]
        video = video + self.audio_to_video_attn(video_kv, audio_kv, audio_kv)[0]

        # 4. Feedforward with AdaLN modulation
        v_ff_in = self._modulated(self.norm_v_ff, video, vs_mlp, vsc_mlp)
        video = video + vg_mlp.squeeze(1) * self.ff(v_ff_in)

        a_ff_in = self._modulated(self.norm_a_ff, audio, as_mlp, asc_mlp)
        audio = audio + ag_mlp.squeeze(1) * self.audio_ff(a_ff_in)

        return video, audio


class MiniAVTransformer(nn.Module):
    """A small LTXModel-style stack of unified audio-video blocks.

    Mirrors the LTXModel design from ltx_core/model/transformer/model.py:
    input projections per modality, a stack of BasicAVTransformerBlocks,
    and output projections back to latent channels. Both modalities are
    denoised together at every layer of every step.
    """

    def __init__(
        self,
        num_layers: int = 2,
        video_dim: int = 128,
        audio_dim: int = 64,
        video_latent_channels: int = 128,
        audio_latent_channels: int = 128,
        text_dim: int = 96,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.video_dim = video_dim
        self.audio_dim = audio_dim

        # Input processing (patchify projections)
        self.video_proj_in = nn.Linear(video_latent_channels, video_dim)
        self.audio_proj_in = nn.Linear(audio_latent_channels, audio_dim)

        # Unified transformer blocks for both modalities
        self.transformer_blocks = nn.ModuleList(
            [
                BasicAVTransformerBlock(
                    video_dim=video_dim,
                    audio_dim=audio_dim,
                    num_heads=num_heads,
                    text_dim=text_dim,
                )
                for _ in range(num_layers)
            ]
        )

        # Output processing
        self.video_norm_out = nn.LayerNorm(video_dim)
        self.audio_norm_out = nn.LayerNorm(audio_dim)
        self.video_proj_out = nn.Linear(video_dim, video_latent_channels)
        self.audio_proj_out = nn.Linear(audio_dim, audio_latent_channels)

    def forward(self, video: Modality, audio: Modality) -> tuple[Tensor, Tensor]:
        """Denoise both modalities jointly. Returns (video_pred, audio_pred)."""
        v = self.video_proj_in(video.latent)
        a = self.audio_proj_in(audio.latent)

        # One embedding per sample from the mean token timestep.
        v_t_emb = timestep_embedding(video.timesteps.float().mean(dim=1), self.video_dim)
        a_t_emb = timestep_embedding(audio.timesteps.float().mean(dim=1), self.audio_dim)

        for block in self.transformer_blocks:
            v, a = block(
                video=v,
                audio=a,
                video_context=video.context,
                audio_context=audio.context,
                video_t_emb=v_t_emb,
                audio_t_emb=a_t_emb,
            )

        video_pred = self.video_proj_out(self.video_norm_out(v))
        audio_pred = self.audio_proj_out(self.audio_norm_out(a))
        return video_pred, audio_pred
