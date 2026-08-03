"""Video evaluation helpers accompanying Chapter 1 of "Video Generation with AI".

Implements the chapter's basic evaluation function, which balances three
measurable signals: temporal consistency (SSIM between consecutive frames),
visual sharpness (variance of the Laplacian), and motion magnitude (one minus
the SSIM between the first and last frames). Prompt adherence, the fourth
metric discussed in the chapter, stays a human judgment call.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim


def load_frames(video_path):
    """Read a video file and return its frames as a list of RGB arrays."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def evaluate_video_basics(video_path, show_frames=True):
    """Evaluate a generated video and return a dict of basic quality metrics.

    Returns a dict with 'consistency', 'sharpness', 'motion', and 'frames',
    or None if the video has too few frames to evaluate. Interpretation
    thresholds follow the chapter: no single metric tells the whole story,
    so read them together and keep the intended use case in mind.
    """
    print(f"\nBasic Video Evaluation: {video_path}")
    print("=" * 40)

    frames = load_frames(video_path)
    if len(frames) < 2:
        print("Not enough frames to evaluate")
        return None

    print("Video Properties:")
    print(f"   - Total frames: {len(frames)}")
    print(f"   - Duration: {len(frames) / 24:.1f} seconds (at 24 FPS)")
    print(f"   - Resolution: {frames[0].shape[1]}x{frames[0].shape[0]}")

    # Temporal consistency: SSIM between consecutive frames. High averages
    # mean smooth transitions, but a score near 1.0 can flag a static clip.
    print("\nTemporal Consistency:")
    consistency_scores = []
    for i in range(1, min(len(frames), 10)):
        gray1 = cv2.cvtColor(frames[i - 1], cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        consistency_scores.append(ssim(gray1, gray2))
    avg_consistency = np.mean(consistency_scores)
    print(f"   - Average consistency: {avg_consistency:.3f}")

    if avg_consistency > 0.95:
        print("   - Very smooth transitions (might be too static?)")
    elif avg_consistency > 0.85:
        print("   - Good temporal consistency")
    elif avg_consistency > 0.75:
        print("   - Moderate consistency (dynamic content)")
    else:
        print("   - Low consistency (very dynamic or flickering)")

    # Visual sharpness: variance of the Laplacian of the middle frame.
    # Higher variance means more edge detail, though intentional blur
    # (such as depth of field) also lowers the score.
    print("\nVisual Quality:")
    middle_frame = frames[len(frames) // 2]
    gray_middle = cv2.cvtColor(middle_frame, cv2.COLOR_RGB2GRAY)
    sharpness = cv2.Laplacian(gray_middle, cv2.CV_64F).var()
    print(f"   - Sharpness score: {sharpness:.0f}")

    if sharpness > 500:
        print("   - Very sharp and detailed")
    elif sharpness > 200:
        print("   - Good visual clarity")
    elif sharpness > 100:
        print("   - Moderate sharpness")
    else:
        print("   - Soft or blurry appearance")

    # Motion magnitude: one minus the SSIM between the first and last frames.
    print("\nMotion Characteristics:")
    motion_score = 1 - ssim(
        cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY),
        cv2.cvtColor(frames[-1], cv2.COLOR_RGB2GRAY),
    )
    print(f"   - Overall motion: {motion_score:.3f}")

    if motion_score < 0.1:
        print("   - Very little motion (static scene?)")
    elif motion_score < 0.3:
        print("   - Subtle, controlled motion")
    elif motion_score < 0.5:
        print("   - Moderate motion")
    else:
        print("   - Dynamic scene with significant motion")

    if show_frames:
        plot_frame_samples(frames)

    return {
        "consistency": avg_consistency,
        "sharpness": sharpness,
        "motion": motion_score,
        "frames": len(frames),
    }


def plot_frame_samples(frames):
    """Plot the first, middle, and last frames to inspect progression visually."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(frames[0])
    axes[0].set_title("First Frame")
    axes[0].axis("off")
    axes[1].imshow(frames[len(frames) // 2])
    axes[1].set_title("Middle Frame")
    axes[1].axis("off")
    axes[2].imshow(frames[-1])
    axes[2].set_title("Last Frame")
    axes[2].axis("off")
    plt.suptitle("Video Frame Samples", fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "my_first_ai_video.mp4"
    evaluate_video_basics(path)
