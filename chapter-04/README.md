# Chapter 4 - Training Video Generation Models

Companion code for Chapter 4 of *Video Generation with AI* (O'Reilly) by Joseph Enochs.

Chapter 3 built the complete Latte architecture: VAE compression, patch embedding, spatial and temporal attention, diffusion, and text conditioning. This chapter turns that architecture into a working model through a systematic training pipeline. The temporal dimension makes this harder than image training: datasets are massive, memory constraints are severe, and optimization is delicate. The approach throughout follows Latte's pragmatic playbook: start simple, validate each component, and add complexity only when it earns its keep.

## What you will build

- A configuration system using hierarchical YAML files and OmegaConf, with command-line overrides and immutable configs, so every experiment is reproducible.
- A video dataset pipeline with intelligent clip indexing, HDF5 frame caching (roughly 100x faster than re-decoding), and augmentations that stay consistent across frames to preserve motion.
- A full training loop: AdamW with selective weight decay, linear warmup plus cosine annealing, gradient accumulation and clipping, optional mixed precision, TF32, EMA weights, and full-state checkpointing.
- Diffusion loss functions: Latte's plain MSE objective, plus optional Min-SNR weighting for when detail generation lags.
- Distributed training with PyTorch DDP, and a clear-eyed view of when model, pipeline, or 3D parallelism becomes necessary.

The notebook runs every component end to end on tiny synthetic videos and a small stand-in denoiser, so you can execute the whole pipeline in minutes and then scale the same code up to real data.

## Section outline

Matching the chapter's structure:

1. **Configuration-Driven Training**
   - YAML Configuration Structure
   - Loading and Managing Configurations
2. **Dataset Preparation and Loading**
   - Video Dataset Architecture (initialization, clip indexing, cached frame loading, temporal-consistent augmentation, batching)
   - Multi-Resolution Training Strategy
   - Temporal Sampling and Frame Selection
3. **Training Infrastructure and Optimization**
   - Anatomy of a Training Loop
   - Memory-Efficient Training Strategies
   - The trainer in seven parts: initialization, distributed setup, optimizer and scheduler, gradient accumulation, training step, loss computation, epoch loop
   - Learning Rate Scheduling and Warmup
   - Gradient Management and Stability
4. **Loss Functions and Metrics**
   - Loss Design: From Simple to Complex (what production models actually use)
   - Simple MSE (Latte) and optional Min-SNR weighting
5. **Scaling and Distributed Training**
   - Data Parallelism (the Latte approach)
   - Advanced Parallelism and Practical Considerations

## Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jenochs/video-ai-book-companion/blob/main/chapter-04/notebooks/chapter_4_companion_notebook.ipynb)

`notebooks/chapter_4_companion_notebook.ipynb` runs top to bottom in Colab. All training demos default to tiny settings (a few steps, 64 x 64 synthetic clips) controlled by a clearly marked configuration cell near the top. Scale the values in that cell up when you point the pipeline at real data.

## Reusable modules

The substantial code lives in `src/` so you can import it into your own training scripts:

- `src/config_utils.py`: OmegaConf loading, overrides, and training environment setup.
- `src/video_dataset.py`: `VideoDataset` with clip indexing, HDF5 caching, and consistent augmentation, plus `video_collate_fn`.
- `src/trainer.py`: `VideoModelTrainer` with the full training loop, EMA, checkpointing, and DDP support.
- `src/losses.py`: Latte's MSE training step and Min-SNR weighted loss.

## Running locally

```bash
git clone https://github.com/jenochs/video-ai-book-companion.git
cd video-ai-book-companion/chapter-04
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/chapter_4_companion_notebook.ipynb
```

To use the modules directly:

```python
import sys; sys.path.append("src")
from video_dataset import VideoDataset, video_collate_fn
from trainer import VideoModelTrainer
```

Multi-GPU training uses standard PyTorch DDP, launched with `torchrun --nproc_per_node=<gpus> your_train_script.py`.

## Hardware notes

Be realistic about what training takes:

- **The notebook demos**: run anywhere, including CPU-only Colab. A T4 makes them faster but is not required.
- **Small-scale real training** (Latte-style model, 256 x 256, 16 frames): plan on a 24 GB GPU (RTX 3090/4090, A10G) as a practical floor, using gradient accumulation, gradient checkpointing, and TF32 or mixed precision. A single batch of 16-frame clips at 256 x 256 needs about 4 GB for activations alone, before gradients, optimizer states, and parameters.
- **Serious training runs**: Latte-scale pretraining uses multiple A100/H100-class GPUs (8 to 16 with DDP) for days to weeks. Models beyond roughly 10 B parameters (HunyuanVideo at 13 B, Wan2.2 at 27 B) need model, pipeline, or 3D parallelism, not just DDP.
- Video decoding is the other bottleneck: budget fast local storage for the HDF5 cache, and expect the first epoch to be slow while the cache builds.

## Check your understanding

1. Why extract multiple fixed-length clips per video during indexing instead of sampling videos directly?
2. Video augmentation samples its random parameters once per clip and applies them to every frame. What breaks if you sample them per frame instead?
3. Latte trains with plain MSE loss while many papers propose multi-component losses. What does the chapter's survey of production models (2022 to 2025) conclude, and when would you add Min-SNR weighting?
4. Your model no longer fits on a single 24 GB GPU even with gradient checkpointing. Which parallelism strategy do you reach for, and why is DDP no longer enough?

<details>
<summary>Answers</summary>

1. Videos vary from seconds to hours, so iterating over files would bias training toward long videos. Extracting up to a fixed number (10) of evenly spaced clips per video equalizes sampling probability regardless of duration and captures diverse temporal moments rather than clustering at video beginnings.

2. Temporal coherence breaks. A per-frame random crop or flip moves objects between frames in ways that look like impossible motion, so the model learns jittery, inconsistent dynamics. Sampling once per clip preserves object trajectories while still adding diversity.

3. Every leading production model relies on MSE variants: pure MSE (Latte), SNR-weighted MSE (CogVideo), or flow matching, which is MSE on a velocity target (Open-Sora, HunyuanVideo, Wan2.2). The differentiators are scale, data quality, and training stability, not exotic losses. Add Min-SNR weighting only if you observe poor fine-detail generation, a sign of learning imbalance across noise levels.

4. Model parallelism (splitting layers across GPUs), possibly combined with pipeline parallelism, and 3D parallelism at the largest scales. DDP requires every GPU to hold a complete copy of the model, so once a single copy exceeds one GPU's memory, replicating it is no longer possible and the model itself must be sharded.

</details>

## The book

This is the companion repository for *Video Generation with AI* by Joseph Enochs, published by O'Reilly Media (Print ISBN 9798341653344). The book covers the concepts in depth; this repository gives you the running code. Chapter 5 continues with fine-tuning pretrained video models for specific tasks.
