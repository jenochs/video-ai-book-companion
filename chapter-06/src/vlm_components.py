"""Core VLM building blocks for Chapter 6 of Video Generation with AI (O'Reilly).

This module accompanies Chapter 6, "Vision Language Models for Video
Understanding". It implements the projector and patch merger components from
the three-component VLM architecture (vision encoder, projector, language
model), plus helpers for vision token counting and the 3D position index
construction used by Qwen3-VL's Interleaved-MRoPE.
"""

import torch
import torch.nn as nn


def count_vision_tokens(frame_height: int, frame_width: int, patch_size: int = 14,
                        num_frames: int = 8, merge_size: int = 1) -> int:
    """Count the vision tokens a video produces before or after spatial merging.

    Args:
        frame_height: Frame height in pixels.
        frame_width: Frame width in pixels.
        patch_size: Side length of each square patch (Qwen3-VL uses 14).
        num_frames: Number of sampled frames.
        merge_size: Spatial merge factor (1 = no merging, 2 = 2x2 pooling).

    Returns:
        Total vision token count for the video.
    """
    patches_h = frame_height // (patch_size * merge_size)
    patches_w = frame_width // (patch_size * merge_size)
    return patches_h * patches_w * num_frames


class Projector(nn.Module):
    """Basic MLP projector that maps vision features into the LLM embedding space.

    The vision encoder speaks a low-dimensional "visual dialect" (for example
    1024 dims) while the language model expects its own embedding size (for
    example 4096 dims). This two-layer MLP bridges the two.
    """

    def __init__(self, vision_dim: int, llm_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(vision_dim, llm_dim)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(llm_dim, llm_dim)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        # vision_features: [batch, num_patches, vision_dim]
        x = self.linear1(vision_features)
        x = self.gelu(x)
        x = self.linear2(x)
        # output: [batch, num_patches, llm_dim]
        return x


class Merger(nn.Module):
    """Patch merger that pools adjacent patches before projecting to the LLM space.

    A spatial_merge_size of 2 combines each 2x2 group of adjacent patches into
    a single token, cutting the token count by a factor of 4 while retaining
    essential spatial information.
    """

    def __init__(self, vision_dim: int, llm_dim: int, spatial_merge_size: int = 2):
        super().__init__()
        self.merge_size = spatial_merge_size
        # Input dim multiplied by merge_size squared because pooled patches are concatenated
        merged_dim = vision_dim * (spatial_merge_size ** 2)
        self.mlp = nn.Sequential(
            nn.Linear(merged_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )

    def forward(self, vision_features: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        """Reshape to a spatial grid, pool adjacent patches, then project.

        Args:
            vision_features: [batch, grid_h * grid_w, vision_dim]
            grid_h: Patch grid height before merging.
            grid_w: Patch grid width before merging.

        Returns:
            [batch, (grid_h // m) * (grid_w // m), llm_dim] where m is merge_size.
        """
        batch, _, dim = vision_features.shape
        m = self.merge_size
        x = vision_features.view(batch, grid_h, grid_w, dim)
        # Group each m x m neighborhood and concatenate along the channel axis
        x = x.view(batch, grid_h // m, m, grid_w // m, m, dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(batch, (grid_h // m) * (grid_w // m), dim * m * m)
        return self.mlp(x)


def build_3d_position_ids(num_frames: int, grid_h: int, grid_w: int,
                          spatial_merge_size: int = 1) -> torch.Tensor:
    """Construct 3D (temporal, height, width) position indices for vision tokens.

    Simplified, runnable version of Qwen3-VL's get_rope_index_3 logic from
    rope2d.py. Every vision token receives three position values so the model
    knows when (frame) and where (row, column) it appears.

    Args:
        num_frames: Temporal grid size (number of sampled frames).
        grid_h: Patch grid height before merging.
        grid_w: Patch grid width before merging.
        spatial_merge_size: Merge factor applied to the spatial grid.

    Returns:
        LongTensor of shape [3, num_tokens] where row 0 holds temporal indices,
        row 1 height indices, and row 2 width indices.
    """
    llm_grid_t = num_frames
    llm_grid_h = grid_h // spatial_merge_size
    llm_grid_w = grid_w // spatial_merge_size

    # t_index: [0,0,...,0, 1,1,...,1, 2,2,...,2, ...]
    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(
        -1, llm_grid_h * llm_grid_w
    ).flatten()

    # h_index: [0,0,0,1,1,1,2,2,2, 0,0,0,1,1,1,2,2,2, ...]
    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(
        llm_grid_t, -1, llm_grid_w
    ).flatten()

    # w_index: [0,1,2,0,1,2,0,1,2, 0,1,2,0,1,2,0,1,2, ...]
    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(
        llm_grid_t, llm_grid_h, -1
    ).flatten()

    return torch.stack([t_index, h_index, w_index])
