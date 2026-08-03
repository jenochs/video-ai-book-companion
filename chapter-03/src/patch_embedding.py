"""Video patch embedding with dual (spatial and temporal) positional encoding.

Accompanies Chapter 3 of "Video Generation with AI" (O'Reilly) by Joseph Enochs.

VideoPatchEmbed divides each latent frame into non-overlapping patches,
projects them into an embedding space, and adds separate spatial and temporal
sinusoidal positional encodings so the model knows both where each patch sits
within a frame and which frame it belongs to.
"""

import numpy as np
import torch
import torch.nn as nn


def get_sincos_encoding(n_position, d_hid):
    """Build a sinusoidal positional encoding table.

    Returns a float tensor of shape [1, n_position, d_hid]. Sine is applied
    to even indices and cosine to odd indices, giving each position a unique
    smooth signature that generalizes to unseen sequence lengths.
    """

    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sincos_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    # Apply sin to even indices
    sincos_table[:, 0::2] = np.sin(sincos_table[:, 0::2])
    # Apply cos to odd indices
    sincos_table[:, 1::2] = np.cos(sincos_table[:, 1::2])

    return torch.FloatTensor(sincos_table).unsqueeze(0)


class VideoPatchEmbed(nn.Module):
    """Patch embedding with dual positional encoding for latent video.

    Input shape: [batch, channels, frames, height, width].
    Output shape: [batch, frames * patches_per_frame, embed_dim].
    """

    def __init__(self, patch_size=2, in_channels=4, embed_dim=768,
                 num_frames=16, image_size=32):
        super().__init__()
        self.patch_size = patch_size

        # Calculate total patches
        self.num_patches_per_frame = (image_size // patch_size) ** 2
        self.num_patches = self.num_patches_per_frame * num_frames

        # Linear projection: patches -> embeddings
        patch_dim = in_channels * patch_size * patch_size
        self.proj = nn.Linear(patch_dim, embed_dim)

        # Dual positional encodings
        self.spatial_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches_per_frame, embed_dim)
        )
        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(1, num_frames, embed_dim)
        )

        # Initialize with sinusoidal patterns
        self._init_pos_encoding()

    def _init_pos_encoding(self):
        # Initialize spatial positions
        self.spatial_pos_embed.data.copy_(
            get_sincos_encoding(
                self.num_patches_per_frame,
                self.spatial_pos_embed.shape[-1],
            )
        )

        # Initialize temporal positions
        self.temporal_pos_embed.data.copy_(
            get_sincos_encoding(
                self.temporal_pos_embed.shape[1],
                self.temporal_pos_embed.shape[-1],
            )
        )

    def forward(self, x):
        # x shape: [batch, channels, frames, height, width]
        B, C, T, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0

        # Reshape into patches
        x = x.reshape(B, C, T, H // self.patch_size, self.patch_size,
                      W // self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 5, 4, 6, 1).reshape(
            B, T * (H // self.patch_size) * (W // self.patch_size),
            C * self.patch_size * self.patch_size,
        )

        # Apply linear projection
        x = self.proj(x)  # [B, T * num_patches_per_frame, embed_dim]

        # Add dual positional encoding
        x = x.reshape(B, T, -1, x.shape[-1])

        # Add spatial position to each frame
        x = x + self.spatial_pos_embed.unsqueeze(1)

        # Add temporal position across frames
        x = x + self.temporal_pos_embed.unsqueeze(2)

        # Flatten back to sequence
        x = x.reshape(B, -1, x.shape[-1])

        return x
