# Chapter 1 - Foundations of Video Generation

This folder contains the companion code for Chapter 1 of *Video Generation with AI* by Joseph Enochs (O'Reilly). The chapter's learning objective is to understand the foundational principles and emerging methods used to generate video with AI. You start by generating a real video in minutes with CogVideoX-2B, then step back to see why video is so much harder than still images: models must keep motion, identity, lighting, and perspective coherent across every frame, not just compose a single one. Along the way you trace the architectural evolution from GANs and VAEs to the diffusion transformers that power today's leading models, and you build simple, repeatable tools for judging the videos you make.

## What you will build

Working through the notebook, you will:

- Set up a video generation environment (PyTorch, Diffusers, and supporting libraries) and verify your GPU.
- Load the CogVideoX-2B pipeline with memory optimizations (CPU offload, VAE tiling and slicing) so it runs on modest hardware.
- Generate your first AI video from a text prompt plus a negative prompt, then save and preview it as an MP4.
- Visualize the diffusion process, the CogVideoX-2B pipeline architecture (T5 text encoder, diffusion transformer, 3D VAE), and the complexity gap between images and video.
- Implement a basic evaluation function that measures temporal consistency (SSIM), visual sharpness (Laplacian variance), and motion magnitude.
- Run two structured experiments: how inference steps (25 vs 50) affect quality, speed, and motion, and how prompt style (minimal, detailed, artistic) shapes the output.
- Produce summary charts of the quality vs speed tradeoff, metric subjectivity, and recommended settings by use case.
- Create and evaluate a video from your own prompt in a creative workspace cell.

## Chapter outline

1. Hands-On: Your First AI Video
   - Environment Setup
   - Loading the Model
   - Generate Your First Video
   - Save and Display Your Achievement
2. Introduction to Generative Video Models
   - What Makes Video Unique in AI (temporal consistency, computational demands, sequence of events)
   - Key Use Cases for Generative Video (entertainment, advertising, healthcare, cultural preservation)
3. The Architectures Behind the Models
   - GANs, Diffusion Models, and Variational Autoencoders
   - Modern Generative Techniques for Video (the DiT architecture and its components)
4. Open Models and Baseline Tools (Wan, CogVideoX, Open-Sora 2.0, HunyuanVideo, Stable Video Diffusion)
5. Evaluation and Early Experiments (temporal consistency, motion magnitude, sharpness, prompt adherence)
6. Getting Started with AI Video Generation
   - Exercise 1: Implementing Basic Evaluation Functions
   - Exercise 2: How Inference Steps Impact Quality
   - Experiment 2: How Prompt Styles Influence Output
7. Learning from Your Experiments (plus the Evaluation Guide)
8. Hands-On Exercise: Create and Evaluate Your Own Video
9. Summary

## Run the notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jenochs/video-ai-book-companion/blob/main/chapter-01/notebooks/chapter_1_companion_notebook.ipynb)

The notebook lives at [`notebooks/chapter_1_companion_notebook.ipynb`](notebooks/chapter_1_companion_notebook.ipynb) with saved outputs, so you can read the results without running anything.

The author also maintains a shared Colab copy: <https://colab.research.google.com/drive/1_mkeaydErmYW1ugrh8SS4qV_jF_3EGoI>. Open it and choose File > Save a copy in Drive to experiment with your own version.

## Running locally

```bash
pip install -r requirements.txt
jupyter notebook notebooks/chapter_1_companion_notebook.ipynb
```

The reusable code from the chapter also lives in [`src/`](src/) as plain Python modules:

- `src/generation.py`: environment check, CogVideoX pipeline loading with optimizations, and video generation helpers.
- `src/evaluation.py`: the `evaluate_video_basics` function for consistency, sharpness, and motion metrics.
- `src/experiments.py`: the inference steps and prompt style experiments.
- `src/visualization.py`: the learning summary charts.

### Hardware notes

- Video generation needs a CUDA GPU. CogVideoX-2B runs in float16 on consumer GPUs with 8 GB or more of VRAM once you enable model CPU offload plus VAE tiling and slicing, as the notebook does. The free Colab T4 tier works for the chapter's exercises.
- The larger CogVideoX-5B, mentioned as an option, wants a professional workstation class GPU (roughly 24 GB or more of VRAM).
- Expect a 49 frame, 720x480 clip at 50 inference steps to take several minutes on a T4. Fewer steps or fewer frames speed things up at some cost in quality, which is exactly what Exercise 2 measures.
- Python 3.12 or greater is recommended. The book's known-good reference environment pins `diffusers==0.33.1` and `transformers==4.52.4`; `requirements.txt` uses looser bounds for flexibility.

## Check your understanding

1. Why is video generation exponentially harder than image generation, in terms of raw data?
2. In a GAN, what roles do the Generator and Discriminator play, and what failure mode makes GANs tricky for video?
3. In the CogVideoX-2B pipeline, what do the T5 text encoder, the diffusion transformer, and the 3D VAE each contribute?
4. A generated video scores a near-perfect SSIM temporal consistency of 0.999. Why is that not automatically good news?

<details><summary>Answers</summary>

1. Each frame multiplies the data to model. A single 720x480 frame with three color channels is about 1,036,800 values; a 2 second clip of the same content balloons to roughly 49,766,400 values, and every frame must stay coherent with all the others in motion, lighting, and identity.
2. The Generator creates candidate videos from noise and the Discriminator judges whether each one is real or generated, driving both to improve. Training can destabilize, leading to mode collapse (repetitive outputs) and temporal flickering across frames, which worsens for longer sequences.
3. The T5 text encoder turns the prompt into dense context embeddings; the diffusion transformer iteratively denoises latent representations using 3D full attention across space and time; the 3D VAE decodes the final latents back into full resolution frames, preserving motion and visual detail.
4. High SSIM only means consecutive frames are very similar. A static image repeated 50 times would score near 1.0. You need to pair it with a motion measure and your own judgment; the 25 step rose experiment scored 0.999 consistency precisely because it barely moved.

</details>

---

From *Video Generation with AI* by Joseph Enochs, published by O'Reilly Media (Print ISBN 9798341653344).
