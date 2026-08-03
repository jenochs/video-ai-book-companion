"""Temporal and spatial augmentation for video data.

This module accompanies Chapter 2 of "Video Generation with AI"
(O'Reilly). Augmentation enriches a video dataset without new
collection: temporal augmentation resamples playback speed, and
spatial augmentation applies per-frame transforms such as flips,
small rotations, and brightness changes.
"""

import cv2
import numpy as np


def temporal_augmentation(video_frames, speed_factor=1.5):
    """Apply temporal augmentation by changing playback speed.

    A speed_factor above 1.0 drops frames (faster playback); below 1.0
    repeats frames (slower playback). Returns the resampled frame list.
    """
    if speed_factor == 1.0:
        return video_frames

    # Sample frames at a different rate
    indices = np.arange(0, len(video_frames), speed_factor)
    indices = indices.astype(int)
    indices = indices[indices < len(video_frames)]

    return [video_frames[i] for i in indices]


def spatial_augmentation(frame, flip=True, rotate=True, brightness=1.0):
    """Apply spatial augmentations to a single frame.

    Optionally applies a random horizontal flip, a random rotation
    between -10 and +10 degrees, and a brightness adjustment. Returns
    the augmented frame, leaving the input untouched.
    """
    augmented = frame.copy()

    # Horizontal flip
    if flip and np.random.random() > 0.5:
        augmented = cv2.flip(augmented, 1)

    # Random rotation
    if rotate:
        angle = np.random.uniform(-10, 10)
        h, w = augmented.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        augmented = cv2.warpAffine(augmented, matrix, (w, h))

    # Brightness adjustment
    if brightness != 1.0:
        augmented = cv2.convertScaleAbs(augmented, alpha=brightness, beta=0)

    return augmented
