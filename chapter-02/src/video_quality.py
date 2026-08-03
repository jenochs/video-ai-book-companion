"""Raw video analysis and OpenCV quality assessment.

This module accompanies Chapter 2 of "Video Generation with AI"
(O'Reilly). It covers the first stage of a video preprocessing
pipeline: inspecting basic properties of raw footage and scoring
clips for sharpness and motion before heavier processing begins.
"""

import cv2
import numpy as np


def analyze_raw_video(video_path):
    """Analyze basic properties of raw video data.

    Returns a dict with resolution, frame rate, frame count, duration
    in seconds, and an estimated uncompressed size in gigabytes
    (assuming 3 bytes per pixel for RGB frames).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open: {video_path}")

    # Core properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Derived metrics
    duration_seconds = frame_count / fps if fps > 0 else 0
    bytes_per_frame = width * height * 3  # RGB
    estimated_size_gb = (bytes_per_frame * frame_count) / (1024 ** 3)

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": duration_seconds,
        "estimated_size_gb": estimated_size_gb,
    }


def stream_video(source, max_frames=3):
    """Simulate streaming video by reading frames one by one.

    This does not create a real network stream. It demonstrates
    sequential frame access, the principle that underlies most
    streaming systems. Returns the buffered frames.
    """
    cap = cv2.VideoCapture(str(source))
    frame_count = 0
    buffer = []
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        buffer.append(frame)
        print(
            f"Frame {frame_count}: "
            f"{{'frame_shape': {frame.shape}, 'buffer_len': {len(buffer)}}}"
        )
        frame_count += 1
    cap.release()
    return buffer


class OpenCVQualityAssessor:
    """Assess video quality using OpenCV metrics.

    Samples a handful of frames from a clip and reports resolution,
    frame rate, duration, average sharpness (Laplacian variance),
    and average motion (Farneback optical flow magnitude).
    """

    def __init__(self, sharpness_threshold=100):
        self.sharpness_threshold = sharpness_threshold

    def assess(self, video_path, sample_frames=5):
        """Assess video quality by sampling frames."""
        cap = cv2.VideoCapture(str(video_path))

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        # Sample frames for quality assessment
        frame_indices = np.linspace(0, total_frames - 1, sample_frames, dtype=int)
        sharpness_scores = []
        motion_scores = []
        prev_gray = None

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Measure sharpness using Laplacian variance
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness_scores.append(laplacian.var())

            # Measure motion using optical flow
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                motion = np.mean(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2))
                motion_scores.append(motion)

            prev_gray = gray

        cap.release()

        avg_sharpness = np.mean(sharpness_scores) if sharpness_scores else 0
        avg_motion = np.mean(motion_scores) if motion_scores else 0

        return {
            "resolution": f"{width}x{height}",
            "fps": fps,
            "duration": duration,
            "avg_sharpness": avg_sharpness,
            "avg_motion": avg_motion,
            "is_sharp": avg_sharpness > self.sharpness_threshold,
        }
