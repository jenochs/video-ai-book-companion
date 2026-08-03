"""Video VAE encoder and decoder for latent space compression.

Accompanies Chapter 3 of "Video Generation with AI" (O'Reilly) by Joseph Enochs.

The encoder follows the chapter's listing: progressive spatial downsampling
through 3D convolutions whose kernels use size 1 on the time axis, so spatial
resolution shrinks 8x while temporal resolution is preserved. The decoder is a
minimal mirror of the encoder, added so the companion notebook can complete the
encode -> generate -> decode round trip described in the chapter.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VideoVAEEncoder(nn.Module):
    """Compress video frames into a latent representation.

    Input shape: [batch, channels, frames, height, width].
    Kernel sizes (1, 3, 3) and (1, 4, 4) preserve the temporal dimension
    while reducing each spatial dimension by a total factor of 8.
    """

    def __init__(self, in_channels=3, latent_channels=4, base_channels=128):
        super().__init__()
        # Progressive downsampling through 3D convolutions
        # kernel_size=(1, h, w): the leading 1 preserves time
        self.conv1 = nn.Conv3d(
            in_channels, base_channels,
            kernel_size=(1, 3, 3),
            stride=(1, 1, 1),
            padding=(0, 1, 1),
        )
        self.conv2 = nn.Conv3d(
            base_channels, base_channels * 2,
            kernel_size=(1, 4, 4),
            stride=(1, 2, 2),  # 2x downsample
            padding=(0, 1, 1),
        )
        self.conv3 = nn.Conv3d(
            base_channels * 2, base_channels * 4,
            kernel_size=(1, 4, 4),
            stride=(1, 2, 2),  # 4x total
            padding=(0, 1, 1),
        )
        self.conv4 = nn.Conv3d(
            base_channels * 4, base_channels * 4,
            kernel_size=(1, 4, 4),
            stride=(1, 2, 2),  # 8x total
            padding=(0, 1, 1),
        )

        # Project to latent distribution parameters
        self.conv_mu = nn.Conv3d(
            base_channels * 4, latent_channels,
            kernel_size=(1, 3, 3), padding=(0, 1, 1),
        )
        self.conv_logvar = nn.Conv3d(
            base_channels * 4, latent_channels,
            kernel_size=(1, 3, 3), padding=(0, 1, 1),
        )

        # Normalization for training stability
        self.norm = nn.GroupNorm(
            num_groups=32, num_channels=base_channels * 4,
        )

    def forward(self, x):
        # x shape: [batch, channels, frames, height, width]

        # Encode through convolutional layers
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))  # 2x downsample
        h = F.relu(self.conv3(h))  # 4x downsample
        h = F.relu(self.conv4(h))  # 8x downsample
        h = self.norm(h)

        # Get distribution parameters
        mu = self.conv_mu(h)
        logvar = self.conv_logvar(h)

        # Reparameterization trick for sampling
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return z, mu, logvar

    def encode_to_latent(self, x):
        """Deterministic inference path: return only the latent mean."""
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = self.norm(h)
        mu = self.conv_mu(h)
        return mu


class VideoVAEDecoder(nn.Module):
    """Minimal decoder that mirrors VideoVAEEncoder.

    Not part of the chapter's printed listings. It is included so the
    companion notebook can decode generated latents back to pixel space,
    completing step 4 of the generation pipeline described in the chapter.
    """

    def __init__(self, out_channels=3, latent_channels=4, base_channels=128):
        super().__init__()
        self.conv_in = nn.Conv3d(
            latent_channels, base_channels * 4,
            kernel_size=(1, 3, 3), padding=(0, 1, 1),
        )
        self.up1 = nn.ConvTranspose3d(
            base_channels * 4, base_channels * 4,
            kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1),
        )
        self.up2 = nn.ConvTranspose3d(
            base_channels * 4, base_channels * 2,
            kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1),
        )
        self.up3 = nn.ConvTranspose3d(
            base_channels * 2, base_channels,
            kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1),
        )
        self.norm = nn.GroupNorm(num_groups=32, num_channels=base_channels)
        self.conv_out = nn.Conv3d(
            base_channels, out_channels,
            kernel_size=(1, 3, 3), padding=(0, 1, 1),
        )

    def forward(self, z):
        # z shape: [batch, latent_channels, frames, height, width]
        h = F.relu(self.conv_in(z))
        h = F.relu(self.up1(h))  # 2x upsample
        h = F.relu(self.up2(h))  # 4x total
        h = F.relu(self.up3(h))  # 8x total
        h = self.norm(h)
        return torch.sigmoid(self.conv_out(h))
