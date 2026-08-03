"""Video dataset pipeline for training video generation models.

Accompanies Chapter 4 of "Video Generation with AI" (O'Reilly) by Joseph Enochs.

Implements the "Dataset Preparation and Loading" section: a video dataset with
intelligent clip indexing, HDF5 caching, temporally consistent augmentation,
and an efficient collate function for PyTorch DataLoaders.
"""

import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    """
    Efficient video dataset with intelligent caching and sampling.
    Handles videos of varying lengths and resolutions gracefully.
    """

    def __init__(self,
                 video_dir: str,
                 metadata_path: str,
                 cache_dir: str = None,
                 clip_length: int = 16,
                 frame_size: tuple = (256, 256),
                 frame_skip: int = 4,
                 augment: bool = True):

        self.video_dir = Path(video_dir)
        self.clip_length = clip_length
        self.frame_size = frame_size
        self.frame_skip = frame_skip  # Sample every Nth frame for temporal coverage
        self.augment = augment

        # Load metadata (video paths, captions, durations)
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        print(f"Loaded {len(self.metadata)} videos from metadata")

        # Initialize cache for preprocessed frames (100x speedup)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(exist_ok=True, parents=True)

        # Build clip index for efficient sampling
        self.clip_index = self._build_clip_index()
        print(f"Built index with {len(self.clip_index)} valid clips")

    def _build_clip_index(self):
        """
        Build index of all valid clips across all videos.
        Handles varying video lengths by extracting multiple clips per video.
        """
        clip_index = []

        for video_info in self.metadata:
            video_path = self.video_dir / video_info['path']
            duration_frames = video_info['num_frames']

            # Calculate how many frames we need for one clip
            clip_frames = self.clip_length * self.frame_skip

            if duration_frames >= clip_frames:
                # Video is long enough for at least one clip
                max_start = duration_frames - clip_frames

                # Extract up to 10 clips per video for balanced sampling
                num_clips = min(10, max_start // clip_frames + 1)

                for i in range(num_clips):
                    # Distribute clips evenly across video duration
                    start_frame = int(i * max_start / max(num_clips - 1, 1))
                    clip_index.append({
                        'video_path': video_path,
                        'start_frame': start_frame,
                        'caption': video_info['caption'],
                        'video_id': video_info['id']
                    })

        return clip_index

    def _load_video_frames(self, video_path, start_frame):
        """
        Load frames with intelligent caching strategy.
        Falls back to video decoding if cache miss occurs.
        """
        # Check cache first (100x faster than decoding)
        cache_path = None
        if self.cache_dir:
            cache_key = f"{video_path.stem}_{start_frame}"
            cache_path = self.cache_dir / f"{cache_key}.h5"

            if cache_path.exists():
                with h5py.File(cache_path, 'r') as f:
                    frames = f['frames'][:]
                return torch.from_numpy(frames)

        # Fall back to video decoding
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        for i in range(self.clip_length):
            # Skip frames according to frame_skip parameter
            for _ in range(self.frame_skip - 1):
                ret, _ = cap.read()
                if not ret:
                    cap.release()
                    return None  # Handle corrupted videos gracefully

            # Read the frame we actually keep
            ret, frame = cap.read()
            if not ret:
                cap.release()
                return None

            # Resize and convert color space
            frame = cv2.resize(frame, self.frame_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        # Normalize to [0, 1] range
        frames = np.stack(frames).astype(np.float32) / 255.0

        # Cache for future use
        if self.cache_dir:
            with h5py.File(cache_path, 'w') as f:
                f.create_dataset('frames', data=frames, compression='gzip')

        return torch.from_numpy(frames)

    def _augment_frames(self, frames):
        """
        Apply augmentations consistently across all frames.
        Maintains temporal coherence while increasing diversity.
        """
        if not self.augment:
            return frames

        B, H, W, C = frames.shape

        # Random horizontal flip (50% chance, same for all frames)
        if torch.rand(1) > 0.5:
            frames = frames.flip(dims=[2])

        # Random crop (90% of original size, same position for all frames)
        crop_size = int(H * 0.9)
        y = torch.randint(0, H - crop_size + 1, (1,)).item()
        x = torch.randint(0, W - crop_size + 1, (1,)).item()
        frames = frames[:, y:y+crop_size, x:x+crop_size]

        # Resize back to original dimensions
        frames = torch.nn.functional.interpolate(
            frames.permute(0, 3, 1, 2),  # BHWC -> BCHW for interpolation
            size=(H, W),
            mode='bilinear',
            align_corners=False
        ).permute(0, 2, 3, 1)  # Back to BHWC

        # Color jitter (plus or minus 10% brightness, consistent across frames)
        brightness = 1 + (torch.rand(1) - 0.5) * 0.2
        frames = frames * brightness
        frames = frames.clamp(0, 1)

        return frames

    def __len__(self):
        return len(self.clip_index)

    def __getitem__(self, idx):
        """Get a single training sample."""
        clip_info = self.clip_index[idx]

        # Load and decode frames
        frames = self._load_video_frames(
            clip_info['video_path'],
            clip_info['start_frame']
        )

        if frames is None:
            # Corrupted video: return random valid sample instead
            return self.__getitem__(torch.randint(len(self), (1,)).item())

        # Apply augmentations
        frames = self._augment_frames(frames)

        # Convert from THWC to TCHW format (channels first for GPU)
        frames = frames.permute(0, 3, 1, 2).unsqueeze(0)  # [1, T, C, H, W]

        return {
            'frames': frames,
            'caption': clip_info['caption'],
            'video_id': clip_info['video_id']
        }


def video_collate_fn(batch):
    """Combine samples into a batch."""
    frames = torch.cat([item['frames'] for item in batch], dim=0)
    captions = [item['caption'] for item in batch]
    video_ids = [item['video_id'] for item in batch]

    return {
        'frames': frames,  # [B, T, C, H, W]
        'captions': captions,  # List of strings
        'video_ids': video_ids  # List of IDs for tracking
    }
