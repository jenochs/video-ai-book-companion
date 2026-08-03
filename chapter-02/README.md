# Chapter 2 - Understanding and Preparing Video Data for Model Training

**Learning objective:** identify different types of video data, understand their unique challenges, and use practical tools leveraged in scalable data pipelines for AI.

Video is high-dimensional, storage-intensive, and often inconsistent across sources. Without systematic preprocessing, datasets become redundant or biased, which reduces model quality and makes training harder to scale. This chapter introduces the methods and tools required to prepare video data for generative pipelines: ingesting clips, detecting scenes, extracting representative frames, applying augmentations, and generating annotations. Projects like Open-Sora and HunyuanVideo filtered tens of millions of raw clips down to a few million high-quality samples with exactly these kinds of pipelines, and by the end of the chapter you will have a scaled-down toolkit built on the same principles.

## What you will build

Working through the companion notebook, you will:

- Set up an organized pipeline workspace with folders for raw files, frames, scenes, annotations, and metadata
- Load a curated gallery of pet videos (the ones you generated in Chapter 1) as a toy dataset
- Analyze raw video properties: resolution, frame rate, duration, and estimated uncompressed size
- Compress video with FFmpeg and the H.264 codec, and measure the compression ratio
- Work with annotated video, extracting clips that contain a specific labeled action
- Simulate streaming by reading frames sequentially into a buffer
- Score clips for sharpness (Laplacian variance) and motion (optical flow) with an OpenCV quality assessor
- Trim and resize videos at the stream level with an FFmpeg exporter
- Detect scene boundaries and extract representative keyframes
- Run scalable filtering and annotation pipelines that apply resolution, duration, and quality checks, then generate per-video metadata, scenes, objects, and actions
- Extract frames adaptively based on content change, and balance a dataset by concept distribution
- Apply temporal (playback speed) and spatial (flip, rotation, brightness) augmentation
- Combine every stage into an end to end preprocessing pipeline

## Chapter outline

- Hands-On: Working with Video Datasets
  - Setting Up the Workspace
  - Loading the Dataset
- Common Challenges in Handling Video Data
  - Scalability
  - Privacy and ethics
  - Compliance and trust
- Working with Video Datasets
- Types of Video Data
  - Raw Video Data (Hands-On: Analyzing Raw Video Properties)
  - Compressed Video Data (Hands-On: Comparing Raw and Compressed Video)
  - Annotated Video Data
  - Hands-On: Simulating Video Streaming
- Choosing Data Based on Use Case
- Sourcing and Collecting Video Data
  - Publicly available datasets
  - Custom video capture
  - Collaborative data sharing
- Automated Filtering Pipelines
- Tools for Video Handling
  - Hands-On: Screening Quality with OpenCV
  - Hands-On: Normalizing and Exporting with FFmpeg
  - Hands-On: Detecting Scenes with PySceneDetect
- Video Processing Pipelines for AI
- Summary

## Notebooks

### Pet video gallery and pipeline

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jenochs/video-ai-book-companion/blob/main/chapter-02/notebooks/Chapter2_Enhanced_Pet_Pipeline_v2.ipynb)

`notebooks/Chapter2_Enhanced_Pet_Pipeline_v2.ipynb` is the working dataset notebook for the chapter. It downloads the nine pet videos used throughout the hands-on sections, builds the thumbnail gallery and interactive preview player, and runs the processing pipeline against real files. Start here if you want to see the dataset the chapter works with. Also available as a [live Colab copy](https://colab.research.google.com/drive/1wSmWhxXmlP85C_roKJXVlHL-S0UHGGlQ).

### Full chapter walkthrough

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jenochs/video-ai-book-companion/blob/main/chapter-02/notebooks/chapter_2_companion_notebook.ipynb)

`notebooks/chapter_2_companion_notebook.ipynb` contains every hands-on example from the chapter, from workspace setup through the complete end to end pipeline. The easiest way to run either notebook is Colab, which already includes ffmpeg, OpenCV, and everything else they need.

## Reusable modules

The substantial classes from the chapter are also available as importable modules in `src/`:

| Module | Contents |
| --- | --- |
| `src/video_quality.py` | `analyze_raw_video`, `stream_video`, `OpenCVQualityAssessor` |
| `src/ffmpeg_exporter.py` | `compress_video`, `FFmpegExporter` (trim, transcode and resize) |
| `src/scene_detector.py` | `SceneDetector`, `AdaptiveFrameExtractor` |
| `src/pipeline.py` | `ScalableFilteringPipeline`, `ScalableAnnotationPipeline`, `ConceptBalancer`, `run_complete_pipeline` |
| `src/augmentation.py` | `temporal_augmentation`, `spatial_augmentation` |

## Running locally

```bash
git clone https://github.com/jenochs/video-ai-book-companion.git
cd video-ai-book-companion/chapter-02
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/chapter_2_companion_notebook.ipynb
```

System dependencies:

- **ffmpeg** must be installed as a system binary; it is not a pip package. On Ubuntu or Debian run `sudo apt-get install ffmpeg`, on macOS run `brew install ffmpeg`, on Windows download a build from [ffmpeg.org](https://ffmpeg.org/download.html) and add it to your PATH. Verify with `ffmpeg -version`.
- **OpenCV** installs via pip as `opencv-python`, but on minimal Linux systems it may need shared libraries such as `libGL`. If `import cv2` fails with a `libGL.so.1` error, run `sudo apt-get install libgl1` (or use `opencv-python-headless`).
- The notebook writes its workspace to `/content/video_pipeline_demo`, which exists on Colab. When running locally, change the `workspace` path in the setup cell to a local directory such as `./video_pipeline_demo`.

## Check your understanding

1. Why is raw video often impractical at scale, even for short clips?
2. What does a codec do, and how does H.265 (HEVC) compare with H.264 (AVC)?
3. What kinds of checks does an automated filtering pipeline apply before video reaches the training stage, and why does this matter?
4. In the chapter's toolkit, what distinct roles do OpenCV, FFmpeg, and PySceneDetect play?

<details><summary>Answers</summary>

1. Uncompressed frames store every pixel: at 3 bytes per RGB pixel, even the short 720x480 wizard-cat clip works out to roughly 0.27 GB. Raw video preserves maximum fidelity, which matters in filmmaking, medical imaging, and autonomous driving, but its storage and processing demands make it unmanageable for large training datasets.
2. A codec (coder-decoder) encodes video into a smaller form and decodes it back for playback. H.265 offers roughly the same visual fidelity as H.264 at about half the bit rate, at the cost of heavier computation for both encoding and decoding. Traditional codecs work by dividing frames into macroblocks and applying transforms such as the Discrete Cosine Transform.
3. Typical checks include resolution and duration filters, sharpness measured by Laplacian variance, motion levels estimated with optical flow, scene detection to split long recordings, plus deduplication and concept balancing at scale. Open-Sora used this approach to filter more than 20 million videos down to 7.7 million high-quality clips; automated filtering improves downstream model performance by keeping only clean, consistent data.
4. OpenCV handles frame-level processing: quality screening, sharpness and motion metrics, and transformations. FFmpeg operates at the stream level: encoding, compression, trimming, and resizing for batch operations. PySceneDetect adds higher-level structure by finding scene boundaries such as hard cuts and fades, so long videos can be split into coherent clips for training.

</details>

---

This companion accompanies Chapter 2 of *Video Generation with AI* by Joseph Enochs (O'Reilly Media, Print ISBN 9798341653344).
