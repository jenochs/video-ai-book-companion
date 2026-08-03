"""Scene boundary detection and adaptive frame extraction.

This module accompanies Chapter 2 of "Video Generation with AI"
(O'Reilly). Segmenting long footage into coherent scenes and pulling
representative frames from each one is a standard preprocessing step
for scalable video pipelines. The detector here uses simple frame
differencing so it runs anywhere OpenCV runs; production pipelines
often layer PySceneDetect or semantic (embedding-based) detection on
top of the same interface.
"""

from pathlib import Path

import cv2
import numpy as np


class SceneDetector:
    """Detect scene boundaries in videos using frame differences."""

    def __init__(self, threshold=30.0):
        self.threshold = threshold

    def detect_scenes(self, video_path):
        """Detect scene changes and return a list of boundary records.

        Each record contains the frame number, timestamp in seconds,
        and the mean pixel difference that triggered the boundary.
        """
        cap = cv2.VideoCapture(str(video_path))
        scenes = []
        frame_num = 0
        prev_gray = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                # Calculate frame difference
                diff = cv2.absdiff(prev_gray, gray)
                mean_diff = np.mean(diff)

                # Detect scene change
                if mean_diff > self.threshold:
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    scenes.append({
                        "frame": frame_num,
                        "time": frame_num / fps if fps > 0 else 0,
                        "intensity": mean_diff,
                    })

            prev_gray = gray
            frame_num += 1

        cap.release()
        return scenes


class AdaptiveFrameExtractor:
    """Extract frames adaptively based on content changes.

    Frames are captured whenever the mean difference from the previous
    frame exceeds a threshold, so static stretches contribute few frames
    and dynamic stretches contribute more. A minimum count is enforced
    by falling back to evenly spaced sampling.
    """

    def extract_frames(self, video_path, min_frames=5, max_frames=30,
                       threshold=20):
        """Extract frames when significant changes occur.

        Returns (frames, frame_indices).
        """
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        frame_indices = []
        prev_gray = None
        frame_num = 0

        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Always include first frame
            if prev_gray is None:
                frames.append(frame)
                frame_indices.append(frame_num)
            else:
                # Calculate difference
                diff = cv2.absdiff(prev_gray, gray)
                mean_diff = np.mean(diff)

                # Extract frame if significant change
                if mean_diff > threshold:
                    frames.append(frame)
                    frame_indices.append(frame_num)

            prev_gray = gray
            frame_num += 1

        cap.release()

        # Ensure minimum frames with evenly spaced sampling
        if len(frames) < min_frames:
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            for i in np.linspace(0, total_frames - 1,
                                 min_frames - len(frames), dtype=int):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                    frame_indices.append(int(i))

            cap.release()

        return frames, frame_indices

    def save_frames(self, frames, output_dir, prefix="frame"):
        """Save extracted frames to disk as JPEG files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, frame in enumerate(frames):
            filename = output_dir / f"{prefix}_{i:04d}.jpg"
            cv2.imwrite(str(filename), frame)

        print(f"Saved {len(frames)} frames to {output_dir}")
