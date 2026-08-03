"""Audio representation utilities for Chapter 7 of Video Generation with AI.

This module accompanies Chapter 7, "Audio-Visual Synchronization with
Cross-Modal Fusion". It implements the audio front end of the LTX-2 style
pipeline in a compact, runnable form: waveform to log-mel conversion,
per-channel latent normalization, and the patchifier that turns audio
latents into transformer-ready sequences with real-time timing metadata.

The classes mirror the structures described in the book
(ltx_core/model/audio_vae/ops.py and ltx_core/components/patchifiers.py)
but are scaled so they run on a laptop CPU without any checkpoints.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

try:
    import torchaudio
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "torchaudio is required for chapter 7 audio utilities. "
        "Install it with: pip install torchaudio"
    ) from exc


# LTX-2 audio processing defaults (Table 7-3 in the book).
SAMPLE_RATE = 16_000
MEL_BINS = 64
MEL_HOP_LENGTH = 160  # 10 ms frame shift at 16 kHz
N_FFT = 1024  # 64 ms analysis window
LATENT_DOWNSAMPLE_FACTOR = 4  # Audio VAE temporal compression


class AudioProcessor(nn.Module):
    """Convert raw waveforms to log-mel spectrograms.

    Follows the AudioProcessor from ltx_core/model/audio_vae/ops.py.
    With the default parameters the output has 64 mel bins and about
    100 frames per second, which is the resolution LTX-2 uses to keep
    speech timing aligned with video frames.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        mel_bins: int = MEL_BINS,
        mel_hop_length: int = MEL_HOP_LENGTH,
        n_fft: int = N_FFT,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=mel_hop_length,
            f_min=0.0,
            f_max=sample_rate / 2.0,  # Nyquist
            n_mels=mel_bins,
            mel_scale="slaney",
        )

    def resample_waveform(
        self, waveform: Tensor, orig_sample_rate: int, target_sample_rate: int
    ) -> Tensor:
        """Resample a waveform if its rate differs from the target rate."""
        if orig_sample_rate == target_sample_rate:
            return waveform
        return torchaudio.functional.resample(
            waveform, orig_sample_rate, target_sample_rate
        )

    def waveform_to_mel(self, waveform: Tensor, waveform_sample_rate: int) -> Tensor:
        """Compute a log-mel spectrogram shaped [B, C, time, freq].

        Args:
            waveform: Audio tensor shaped [B, C, samples].
            waveform_sample_rate: Sample rate of the input waveform.
        """
        waveform = self.resample_waveform(
            waveform, waveform_sample_rate, self.sample_rate
        )
        mel = self.mel_transform(waveform)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        return mel.permute(0, 1, 3, 2)  # [B, C, time, freq]


class PerChannelStatistics(nn.Module):
    """Dataset-wide per-channel normalization for audio latents.

    In LTX-2 these buffers are loaded from the checkpoint. Each latent
    channel is standardized with its own mean and standard deviation,
    which keeps noise scales consistent across channels during diffusion
    and makes cross-modal attention with video latents behave well.
    Buffer names match the checkpoint keys ("std-of-means",
    "mean-of-means"), which is why they use hyphens.
    """

    def __init__(self, latent_channels: int = 128) -> None:
        super().__init__()
        # Identity defaults so the module is usable without a checkpoint.
        self.register_buffer("std-of-means", torch.ones(latent_channels))
        self.register_buffer("mean-of-means", torch.zeros(latent_channels))

    def _reshape(self, stats: Tensor, x: Tensor) -> Tensor:
        # Broadcast [C] statistics over [B, C, ...] tensors.
        shape = [1, -1] + [1] * (x.dim() - 2)
        return stats.view(*shape)

    def normalize(self, x: Tensor) -> Tensor:
        """Standardize: (x - mean) / std, per channel."""
        mean = self._reshape(self.get_buffer("mean-of-means"), x)
        std = self._reshape(self.get_buffer("std-of-means"), x)
        return (x - mean) / std

    def un_normalize(self, x: Tensor) -> Tensor:
        """Inverse transform: x * std + mean, per channel."""
        mean = self._reshape(self.get_buffer("mean-of-means"), x)
        std = self._reshape(self.get_buffer("std-of-means"), x)
        return x * std + mean


class AudioPatchifier:
    """Convert audio latents to token sequences with timing metadata.

    Follows the AudioPatchifier from ltx_core/components/patchifiers.py.
    Audio uses patch_size=1: frequency bins become channels rather than
    spatial patches, so full frequency resolution is preserved in every
    latent frame. Each latent frame covers about 40 ms of audio
    (4 mel frames x 10 ms hop), roughly one video latent frame at 25 fps.
    """

    def __init__(
        self,
        patch_size: int = 1,
        sample_rate: int = SAMPLE_RATE,
        hop_length: int = MEL_HOP_LENGTH,
        audio_latent_downsample_factor: int = LATENT_DOWNSAMPLE_FACTOR,
        is_causal: bool = True,
    ) -> None:
        self.patch_size = patch_size
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.audio_latent_downsample_factor = audio_latent_downsample_factor
        self.is_causal = is_causal

    def patchify(self, audio_latents: Tensor) -> Tensor:
        """[B, C, T, F] -> [B, T, C*F]: flatten frequency into features."""
        b, c, t, f = audio_latents.shape
        return audio_latents.permute(0, 2, 1, 3).reshape(b, t, c * f)

    def unpatchify(self, audio_tokens: Tensor, channels: int, mel_bins: int) -> Tensor:
        """[B, T, C*F] -> [B, C, T, F]: restore latent structure."""
        b, t, _ = audio_tokens.shape
        return audio_tokens.reshape(b, t, channels, mel_bins).permute(0, 2, 1, 3)

    def latent_frame_times_sec(self, start_latent: int, end_latent: int) -> Tensor:
        """Map latent frame indices to real-time seconds.

        This is the timing metadata that lets the transformer align audio
        events with video frames at sub-frame precision.
        """
        audio_latent_frame = torch.arange(start_latent, end_latent)
        audio_mel_frame = audio_latent_frame * self.audio_latent_downsample_factor
        if self.is_causal:
            audio_mel_frame = (
                audio_mel_frame + 1 - self.audio_latent_downsample_factor
            ).clip(min=0)
        return audio_mel_frame * self.hop_length / self.sample_rate


def make_synthetic_scene_audio(
    duration_sec: float = 2.0, sample_rate: int = SAMPLE_RATE
) -> Tensor:
    """Build a small synthetic test signal shaped [1, 1, samples].

    The signal layers three components a mel-spectrogram makes easy to
    see: a low hum, a rising chirp, and two sharp clicks. The clicks act
    as stand-ins for visual events (a door slam, a footstep) when we
    reason about synchronization.
    """
    n = int(duration_sec * sample_rate)
    t = torch.arange(n) / sample_rate

    hum = 0.20 * torch.sin(2 * math.pi * 110.0 * t)
    f0, f1 = 300.0, 3000.0
    chirp_phase = 2 * math.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration_sec))
    chirp = 0.15 * torch.sin(chirp_phase)

    clicks = torch.zeros(n)
    for click_time in (0.5, 1.5):
        start = int(click_time * sample_rate)
        length = int(0.01 * sample_rate)  # 10 ms burst
        envelope = torch.exp(-torch.arange(length) / (0.002 * sample_rate))
        clicks[start : start + length] += 0.8 * envelope

    waveform = (hum + chirp + clicks).clamp(-1.0, 1.0)
    return waveform.reshape(1, 1, -1)
