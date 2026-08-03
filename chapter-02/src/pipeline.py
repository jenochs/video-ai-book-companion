"""Scalable filtering, annotation, and dataset balancing pipelines.

This module accompanies Chapter 2 of "Video Generation with AI"
(O'Reilly). It links the individual tools from the chapter into an
end-to-end preprocessing flow modeled on large-scale systems such as
Open-Sora and HunyuanVideo: quality filtering, scene detection and
frame extraction, annotation generation, and concept balancing.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from .video_quality import OpenCVQualityAssessor
from .scene_detector import SceneDetector, AdaptiveFrameExtractor


def extract_annotated_clips(annotations, action_type):
    """Extract video segments containing a specific action.

    Takes a list of annotation dicts (each with at least "action" and
    "frame_id" keys) and returns clip records for the requested action.
    """
    clips = []
    for ann in annotations:
        if ann.get("action") == action_type:
            clips.append({
                "start_frame": ann.get("start_frame", ann["frame_id"]),
                "end_frame": ann.get("end_frame", ann["frame_id"]),
                "confidence": ann.get("confidence", 1.0),
            })
    return clips


class ScalableFilteringPipeline:
    """Scalable pipeline for filtering video datasets.

    Applies the same categories of checks used by large-scale projects:
    minimum resolution, duration bounds, and sharpness. Each video gets
    a pass/fail result with human-readable rejection reasons.
    """

    def __init__(self, min_resolution=128, min_duration=2.0, max_duration=60.0):
        self.min_resolution = min_resolution
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.assessor = OpenCVQualityAssessor()

    def process_video(self, video_path):
        """Process a single video through the filtering pipeline."""
        video_path = Path(video_path)

        # Assess quality
        quality = self.assessor.assess(video_path)

        # Apply filters
        passed = True
        reasons = []

        # Resolution filter
        width, height = quality["resolution"].split("x")
        if min(int(width), int(height)) < self.min_resolution:
            passed = False
            reasons.append(f"Resolution too low: {quality['resolution']}")

        # Duration filter
        if quality["duration"] < self.min_duration:
            passed = False
            reasons.append(f"Too short: {quality['duration']:.1f}s")
        elif quality["duration"] > self.max_duration:
            passed = False
            reasons.append(f"Too long: {quality['duration']:.1f}s")

        # Quality filter
        if not quality["is_sharp"]:
            passed = False
            reasons.append("Video is too blurry")

        return {
            "video_path": video_path,
            "passed": passed,
            "quality": quality,
            "rejection_reasons": reasons,
        }

    def batch_process(self, video_paths):
        """Process multiple videos and print a summary."""
        results = []
        for path in video_paths:
            result = self.process_video(path)
            results.append(result)

            status = "Passed" if result["passed"] else "Failed"
            print(f"{Path(path).name}: {status}")
            for reason in result["rejection_reasons"]:
                print(f"    - {reason}")

        passed = sum(1 for r in results if r["passed"])
        print(f"\nSummary: {passed}/{len(results)} videos passed filtering")
        return results


class ScalableAnnotationPipeline:
    """Pipeline for generating video annotations.

    Produces per-video metadata and scene boundaries. Object and action
    annotations are mocked here for demonstration; in practice you would
    plug in an object detector (for example YOLO) and an action
    recognition or vision-language model.
    """

    def __init__(self):
        self.detector = SceneDetector()

    def generate_annotations(self, video_path):
        """Generate comprehensive annotations for a video."""
        video_path = Path(video_path)

        # Basic metadata
        cap = cv2.VideoCapture(str(video_path))
        metadata = {
            "filename": video_path.name,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
        cap.release()

        # Scene detection
        scenes = self.detector.detect_scenes(video_path)

        # Mock object detection (in practice, use YOLO or similar)
        mock_objects = [
            {"frame": 10, "object": "cat", "confidence": 0.95},
            {"frame": 50, "object": "toy", "confidence": 0.87},
        ]

        # Mock actions (in practice, use an action recognition model)
        mock_actions = [
            {"start": 0, "end": 30, "action": "playing"},
            {"start": 31, "end": 60, "action": "resting"},
        ]

        return {
            "metadata": metadata,
            "scenes": scenes,
            "objects": mock_objects,
            "actions": mock_actions,
        }

    def save_annotations(self, annotations, output_path):
        """Save annotations to a JSON file."""
        with open(output_path, "w") as f:
            json.dump(annotations, f, indent=2, default=str)
        print(f"Annotations saved to {output_path}")


class ConceptBalancer:
    """Balance a dataset based on concept distribution.

    Mirrors the concept balancing stage in pipelines like Open-Sora,
    where clustering and uniform sampling keep any single concept from
    dominating the training set.
    """

    def __init__(self, target_samples_per_concept=100):
        self.target_samples = target_samples_per_concept

    def analyze_distribution(self, annotations_list):
        """Count how often each object and action concept appears."""
        concept_counts = {}

        for anno in annotations_list:
            # Count objects
            for obj in anno.get("objects", []):
                concept = obj["object"]
                concept_counts[concept] = concept_counts.get(concept, 0) + 1

            # Count actions
            for action in anno.get("actions", []):
                concept = action["action"]
                concept_counts[concept] = concept_counts.get(concept, 0) + 1

        return concept_counts

    def balance_dataset(self, videos_with_annotations):
        """Balance a dataset by sampling videos per concept.

        Takes (video, annotations) pairs and returns a deduplicated list
        of videos with at most target_samples_per_concept per concept.
        """
        # Group videos by concept
        concept_videos = {}

        for video, anno in videos_with_annotations:
            for obj in anno.get("objects", []):
                concept = obj["object"]
                concept_videos.setdefault(concept, []).append(video)

        # Sample videos for balance
        balanced_dataset = set()

        for concept, videos in concept_videos.items():
            sample_size = min(len(videos), self.target_samples)
            sampled = np.random.choice(videos, sample_size, replace=False)
            balanced_dataset.update(sampled)

        return list(balanced_dataset)


def run_complete_pipeline(video_paths, workspace):
    """Run the complete video preprocessing pipeline.

    Stages: quality filtering, scene detection and frame extraction,
    annotation generation, dataset balancing, and summary statistics.
    Expects a workspace directory containing frames/ and annotations/
    subfolders (they are created if missing).
    """
    workspace = Path(workspace)
    video_paths = [Path(p) for p in video_paths]
    (workspace / "frames").mkdir(parents=True, exist_ok=True)
    (workspace / "annotations").mkdir(parents=True, exist_ok=True)

    print("Starting Complete Video Processing Pipeline")
    print("=" * 60)

    # Stage 1: Quality Assessment and Filtering
    print("\nStage 1: Quality Assessment and Filtering")
    filter_pipeline = ScalableFilteringPipeline()
    filter_results = filter_pipeline.batch_process(video_paths)

    # Get videos that passed filtering
    passed_videos = [r["video_path"] for r in filter_results if r["passed"]]

    if not passed_videos:
        print("No videos passed filtering. Using all videos for demonstration.")
        passed_videos = video_paths

    # Stage 2: Scene Detection and Frame Extraction
    print("\nStage 2: Scene Detection and Frame Extraction")
    extractor = AdaptiveFrameExtractor()

    for video_path in passed_videos:
        frames, _ = extractor.extract_frames(
            video_path, min_frames=3, max_frames=10
        )
        frames_dir = workspace / "frames" / video_path.stem
        extractor.save_frames(frames[:5], frames_dir)  # Save first 5 frames

    # Stage 3: Annotation Generation
    print("\nStage 3: Annotation Generation")
    annotator = ScalableAnnotationPipeline()
    all_annotations = []

    for video_path in passed_videos:
        annotations = annotator.generate_annotations(video_path)
        all_annotations.append(annotations)

        anno_path = workspace / "annotations" / f"{video_path.stem}.json"
        annotator.save_annotations(annotations, anno_path)

    # Stage 4: Dataset Balancing
    print("\nStage 4: Dataset Balancing")
    balancer = ConceptBalancer()
    distribution = balancer.analyze_distribution(all_annotations)

    print("Concept distribution in dataset:")
    for concept, count in distribution.items():
        print(f"  {concept}: {count}")

    # Stage 5: Summary Statistics
    print("\nPipeline Summary")
    print("=" * 60)
    total_frames = sum(
        len(list((workspace / "frames" / v.stem).glob("*.jpg")))
        for v in passed_videos
        if (workspace / "frames" / v.stem).exists()
    )
    print(f"Total videos processed: {len(video_paths)}")
    print(f"Videos passed filtering: {len(passed_videos)}")
    print(f"Total frames extracted: {total_frames}")
    print(f"Annotations generated: {len(all_annotations)}")

    return {
        "filter_results": filter_results,
        "annotations": all_annotations,
        "passed_videos": passed_videos,
    }
