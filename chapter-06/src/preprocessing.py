"""Video preprocessing utilities for Chapter 6 of Video Generation with AI (O'Reilly).

This module accompanies Chapter 6, "Vision Language Models for Video
Understanding". It covers the input side of the video understanding pipeline:
FPS-based frame extraction, smart resizing that preserves aspect ratio while
respecting a pixel budget, ImageNet normalization, adaptive FPS for long
videos, and token budget estimation.
"""

import math

import numpy as np

# Standard normalization statistics for ViT-based vision encoders
IMAGENET_MEAN = [0.48145466, 0.4578275, 0.40821073]
IMAGENET_STD = [0.26862954, 0.26130258, 0.27577711]


def fetch_video(video_path: str, fps: float = 2.0, min_frames: int = 4,
                max_frames: int = None) -> np.ndarray:
    """Extract frames from a video at the specified FPS using decord.

    Qwen3-VL uses FPS-based sampling rather than a fixed frame count, which
    better preserves temporal structure across videos of varying lengths.

    Args:
        video_path: Path to the video file.
        fps: Target frames per second for extraction.
        min_frames: Minimum frames to extract (short clips still get context).
        max_frames: Maximum frames to extract (protects memory on long videos).

    Returns:
        Array of shape [num_frames, height, width, channels], dtype uint8.
    """
    from decord import VideoReader  # Imported lazily so the module loads without decord

    video = VideoReader(video_path)
    native_fps = video.get_avg_fps()
    duration = len(video) / native_fps

    # Calculate frame count based on target FPS
    num_frames = int(duration * fps)
    num_frames = max(num_frames, min_frames)
    if max_frames:
        num_frames = min(num_frames, max_frames)

    # Sample frames at regular intervals
    frame_indices = np.linspace(0, len(video) - 1, num_frames, dtype=int)
    frames = video.get_batch(frame_indices).asnumpy()
    return frames


def smart_resize(height: int, width: int, factor: int = 28,
                 min_pixels: int = None, max_pixels: int = None):
    """Resize dimensions to be divisible by factor while staying within pixel limits.

    Maintains aspect ratio and ensures total pixels fall within
    [min_pixels, max_pixels], the same strategy Qwen3-VL uses to keep patch
    counts manageable without distorting the input.

    Returns:
        (height, width) tuple of adjusted dimensions.
    """
    if min_pixels is None:
        min_pixels = 28 * 28 * 16   # ~12K pixels minimum
    if max_pixels is None:
        max_pixels = 28 * 28 * 576  # ~450K pixels maximum

    current_pixels = height * width

    # Scale down if exceeding maximum
    if current_pixels > max_pixels:
        scale = math.sqrt(max_pixels / current_pixels)
        height = int(height * scale)
        width = int(width * scale)

    # Scale up if below minimum
    if current_pixels < min_pixels:
        scale = math.sqrt(min_pixels / current_pixels)
        height = int(height * scale)
        width = int(width * scale)

    # Round to nearest multiple of factor
    height = round(height / factor) * factor
    width = round(width / factor) * factor

    # Ensure at least one patch in each dimension
    height = max(height, factor)
    width = max(width, factor)

    return height, width


def normalize_frames(frames: np.ndarray) -> np.ndarray:
    """Normalize uint8 frames with ImageNet statistics: (pixel - mean) / std."""
    frames = frames.astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    return (frames - mean) / std


def adaptive_fps(duration: float, max_frames: int = 128) -> float:
    """Calculate an FPS that keeps a video within max_frames.

    Standard rate is 2.0 FPS; longer videos are sampled more sparsely so the
    total frame count never exceeds max_frames.
    """
    if duration * 2.0 <= max_frames:
        return 2.0  # Standard rate
    return max_frames / duration  # Reduced rate


def estimate_vision_tokens(num_frames: int, frame_height: int, frame_width: int,
                           patch_size: int = 14, merge_size: int = 2) -> int:
    """Estimate the vision token count for a video before sending it to the model.

    tokens = num_frames x (height / patch_size / merge_size)
                        x (width / patch_size / merge_size)

    For 8 frames at 448x448 with patch_size=14 and merge_size=2 this gives
    8 x 16 x 16 = 2,048 vision tokens. Add text tokens on top and make sure
    the total fits within the model's context window.
    """
    tokens_h = frame_height // (patch_size * merge_size)
    tokens_w = frame_width // (patch_size * merge_size)
    return num_frames * tokens_h * tokens_w
