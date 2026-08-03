"""HiFi-GAN style vocoder for Chapter 7 of Video Generation with AI.

This module accompanies Chapter 7, "Audio-Visual Synchronization with
Cross-Modal Fusion". It implements the mel-to-waveform vocoder used at
the end of the LTX-2 audio pipeline: transposed-convolution upsampling
stages interleaved with Multi-Receptive Field (MRF) ResBlock fusion,
following ltx_core/model/audio_vae/vocoder.py.

The default configuration matches the book's production numbers
(240x total upsampling, stereo output at 24 kHz). A `tiny` constructor
gives a version small enough to run forward passes on CPU instantly.
The weights here are random, so outputs are noise until trained; the
point is the architecture and the shape arithmetic.
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

LRELU_SLOPE = 0.1


class ResBlock(nn.Module):
    """Dilated residual block operating at a single kernel size.

    Three dilated conv pairs (dilations 1, 3, 5) widen the receptive
    field so the block sees audio patterns at several temporal spans.
    """

    def __init__(self, channels: int, kernel_size: int, dilations=(1, 3, 5)) -> None:
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    dilation=d,
                    padding=(kernel_size - 1) * d // 2,
                )
                for d in dilations
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels, channels, kernel_size, padding=(kernel_size - 1) // 2
                )
                for _ in dilations
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        for conv1, conv2 in zip(self.convs1, self.convs2):
            residual = x
            x = conv1(F.leaky_relu(x, LRELU_SLOPE))
            x = conv2(F.leaky_relu(x, LRELU_SLOPE))
            x = x + residual
        return x


class Vocoder(nn.Module):
    """HiFi-GAN style mel-to-waveform synthesizer.

    Total upsampling is the product of `upsample_rates`. LTX-2 uses
    6 x 5 x 2 x 2 x 2 = 240 because mel-spectrograms arrive at about
    100 frames per second and the output waveform needs 24,000 samples
    per second: 24,000 / 100 = 240.
    """

    def __init__(
        self,
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        upsample_rates: List[int] = [6, 5, 2, 2, 2],
        upsample_kernel_sizes: List[int] = [16, 15, 8, 4, 4],
        upsample_initial_channel: int = 1024,
        stereo: bool = True,
        mel_bins: int = 64,
        output_sample_rate: int = 24_000,
    ) -> None:
        super().__init__()
        self.stereo = stereo
        self.mel_bins = mel_bins
        self.output_sample_rate = output_sample_rate
        self.num_upsamples = len(upsample_rates)
        self.num_kernels = len(resblock_kernel_sizes)

        # Input: stereo mel arrives as [B, 2, T, mel_bins] and is folded
        # to [B, 2*mel_bins, T] (128 channels in the production config).
        in_channels = 2 * mel_bins if stereo else mel_bins
        self.conv_pre = nn.Conv1d(in_channels, upsample_initial_channel, 7, padding=3)

        # Upsampling stages with transposed convolutions
        self.ups = nn.ModuleList()
        for i, (stride, kernel_size) in enumerate(
            zip(upsample_rates, upsample_kernel_sizes)
        ):
            self.ups.append(
                nn.ConvTranspose1d(
                    upsample_initial_channel // (2**i),
                    upsample_initial_channel // (2 ** (i + 1)),
                    kernel_size,
                    stride,
                    padding=(kernel_size - stride) // 2,
                )
            )

        # Multi-receptive field fusion ResBlocks: one per kernel per stage
        self.resblocks = nn.ModuleList()
        for i in range(self.num_upsamples):
            channels = upsample_initial_channel // (2 ** (i + 1))
            for kernel_size in resblock_kernel_sizes:
                self.resblocks.append(ResBlock(channels, kernel_size))

        final_channels = upsample_initial_channel // (2**self.num_upsamples)
        self.conv_post = nn.Conv1d(
            final_channels, 2 if stereo else 1, 7, padding=3
        )

        self.upsample_factor = math.prod(upsample_rates)

    @classmethod
    def tiny(cls) -> "Vocoder":
        """A small configuration for CPU demos. Still 240x upsampling."""
        return cls(
            resblock_kernel_sizes=[3, 7],
            upsample_rates=[6, 5, 2, 2, 2],
            upsample_kernel_sizes=[12, 15, 4, 4, 4],
            upsample_initial_channel=64,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Synthesize a waveform from mel features.

        Args:
            x: Stereo mel tensor [B, 2, T, mel_bins] (or [B, 1, T, mel_bins]
               for mono configurations).

        Returns:
            Waveform tensor [B, 2, T * upsample_factor] (stereo) in [-1, 1].
        """
        # Rearrange stereo mel: [B, S, T, F] -> [B, S*F, T]
        b, s, t, f = x.shape
        x = x.permute(0, 1, 3, 2).reshape(b, s * f, t)

        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)

            # Apply parallel ResBlocks with different receptive fields
            # and average their outputs for multi-scale fusion.
            start = i * self.num_kernels
            end = start + self.num_kernels
            block_outputs = torch.stack(
                [self.resblocks[idx](x) for idx in range(start, end)], dim=0
            )
            x = block_outputs.mean(dim=0)

        x = self.conv_post(F.leaky_relu(x))
        return torch.tanh(x)
