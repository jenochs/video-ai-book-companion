"""Reusable video preprocessing modules accompanying Chapter 2 of
"Video Generation with AI" (O'Reilly).

These modules extract the substantial classes and functions from the
Chapter 2 companion notebook so they can be imported and reused in
scripts and larger pipelines.
"""

from .video_quality import analyze_raw_video, stream_video, OpenCVQualityAssessor
from .ffmpeg_exporter import compress_video, FFmpegExporter
from .scene_detector import SceneDetector, AdaptiveFrameExtractor
from .pipeline import (
    extract_annotated_clips,
    ScalableFilteringPipeline,
    ScalableAnnotationPipeline,
    ConceptBalancer,
    run_complete_pipeline,
)
from .augmentation import temporal_augmentation, spatial_augmentation

__all__ = [
    "analyze_raw_video",
    "stream_video",
    "OpenCVQualityAssessor",
    "compress_video",
    "FFmpegExporter",
    "SceneDetector",
    "AdaptiveFrameExtractor",
    "extract_annotated_clips",
    "ScalableFilteringPipeline",
    "ScalableAnnotationPipeline",
    "ConceptBalancer",
    "run_complete_pipeline",
    "temporal_augmentation",
    "spatial_augmentation",
]
