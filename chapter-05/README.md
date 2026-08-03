# Chapter 5 - Fine-Tuning for Specific Video Tasks

Companion materials for Chapter 5 of *Video Generation with AI* by Joseph Enochs.

## Overview

This chapter moves from theory to production fine-tuning. Building on the CogVideo model introduced in Chapter 1 and the training intuition from Chapter 4's Latte experiments, you use CogVideo's fine-tuning framework from Tsinghua University to specialize a video model for a target style. Two approaches carry the chapter: LoRA (low-rank adaptation) for parameter-efficient fine-tuning on 16-24 GB of VRAM, and supervised fine-tuning (SFT) with DeepSpeed ZeRO for maximum quality when you have the hardware. Along the way the chapter covers domain adaptation, few-shot and zero-shot techniques, evaluation of fine-tuned models, and the practical commands to run your first fine-tuning job.

The core idea behind LoRA: instead of updating a full weight matrix W, learn two small low-rank matrices A and B so that W' = W + BA. Applied to the query, key, value, and output projections of the CogVideoX transformer, this cuts trainable parameters by over 90% while keeping 90-95% of full fine-tuning quality.

## What you will build

The companion notebook fine-tunes CogVideoX-2B with LoRA on the Disney Steamboat Willie dataset, end to end:

- Environment setup with pinned versions of diffusers, transformers, accelerate, peft, and TensorBoard
- TensorBoard configured inside the notebook for live training visualization
- A `DisneyDatasetDownloader` that pulls videos and prompts from the Wild-Heart/Disney-VideoGeneration-Dataset on HuggingFace
- A `CogVideoLoRAConfig` targeting the transformer's attention projections, with memory estimates per rank and model size
- A `DisneyVideoDataset` and a `CogVideoTrainer` that logs loss, learning rate, gradient norms, GPU memory, and LoRA weight histograms to TensorBoard
- A training run with Accelerate, mixed precision, gradient accumulation, and periodic checkpoints
- Training metric plots pulled back out of the TensorBoard event logs
- Side-by-side generation comparing the base model against the fine-tuned model on Disney-style prompts, rendered as inline HTML video players
- Model export with a HuggingFace model card and config, ready to upload to the Hub

## Open in Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jenochs/video-ai-book-companion/blob/main/chapter-05/notebooks/chapter_5_companion_notebook.ipynb)

## Chapter outline

The manuscript covers:

1. **Fine-Tuning Methods** - LoRA and SFT, with VRAM and resolution requirements per model
2. **LoRA: Low-Rank Adaptation for Video Models** - how LoRA works in CogVideo, rank selection, and the production `CogVideoLoRAConfig` and `CogVideoLoRATrainer`
3. **Supervised Fine-Tuning (SFT) with DeepSpeed** - ZeRO stages 1-3, gradient checkpointing, and mixed precision (fp16 vs bf16)
4. **Choosing Between LoRA and SFT** - a decision guide by dataset size, hardware, and quality ceiling
5. **Common Fine-Tuning Issues** - tensor size mismatches (the 8N+1 frame rule), CUDA OOM, non-converging training, mixed precision failures, and LoRA weights that will not load
6. **Domain Adaptation and Transfer Learning** - gradient reversal, domain discriminators, domain-specific batch norm, and cross-domain video generation
7. **Few-Shot and Zero-Shot Video Generation** - meta-learning and compositional generation
8. **Prompt Engineering and In-Context Learning** - gradient-free adaptation through prompting
9. **Evaluation and Benchmarking Fine-Tuned Models** - task-specific metrics, ablation studies, transfer effectiveness
10. **Practical Execution: Running Your First Fine-Tuning** - environment setup, dataset preparation, LoRA and SFT training scripts, multi-GPU scaling, and inference with fine-tuned weights

## Repository contents

```
chapter-05/
├── README.md
├── requirements.txt
├── notebooks/
│   └── chapter_5_companion_notebook.ipynb   # Canonical Colab notebook (25 cells)
├── src/
│   ├── lora_config.py        # CogVideoLoRAConfig, apply_lora_to_model, training configs
│   ├── lora_trainer.py       # CogVideoLoRATrainer with the diffusion training step
│   ├── deepspeed_config.py   # ZeRO-2 configuration for SFT
│   ├── domain_adaptation.py  # Gradient reversal, discriminator, adaptation trainer
│   └── inference.py          # Load LoRA weights and generate videos
└── assets/
```

The `src/` modules are extracted from the chapter manuscript so you can import and adapt them outside the notebook.

## Running locally

```bash
git clone https://github.com/jenochs/video-ai-book-companion.git
cd video-ai-book-companion/chapter-05
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/chapter_5_companion_notebook.ipynb
```

To use the standalone modules:

```python
import sys
sys.path.append("src")

from lora_config import CogVideoLoRAConfig
config = CogVideoLoRAConfig.get_lora_config(model_size="2b", rank=64)
```

## Hardware notes

Be realistic about GPU requirements before you start. Fine-tuning CogVideoX is not a CPU or laptop task:

- **LoRA on CogVideoX-2B**: 16 GB VRAM minimum (RTX 4080 class, or a Colab T4). Training runs 30 minutes to 2 hours.
- **LoRA on CogVideoX-5B**: 24 GB VRAM (RTX 4090 class).
- **SFT**: 36-80 GB VRAM (A100/H100 class), 2-12 hours of training. DeepSpeed ZeRO-2 with gradient checkpointing brings CogVideoX-2B/5B down to about 24 GB at roughly 0.7x speed; ZeRO-3 with CPU offload fits in 16 GB plus system RAM but runs 3-5x slower.
- **Precision**: fp16 works on RTX 20-series or newer; bf16 requires Ampere or newer (RTX 30-series, A100, H100).
- **Frame counts must be 8N+1** (9, 17, 25, 33, 41, 49). Use 17x384x384 for quick tests and 49x480x720 for production runs.

On Colab, select a GPU runtime (T4 or better). The notebook trains the 2B model with rank 64 LoRA, which fits in 16 GB.

## Check your understanding

1. LoRA replaces the update to a weight matrix W with the product BA of two low-rank matrices. For W of size 1024x1024 and rank r=16, roughly how many trainable parameters does LoRA use, and what reduction is that?
2. You launch training and hit "The size of tensor a (45) must match the size of tensor b (44)". What constraint did you violate, and what frame counts are valid?
3. When should you choose SFT over LoRA, and what does DeepSpeed ZeRO-2 partition to make SFT fit on smaller GPUs?
4. During training, which CogVideoX components stay frozen, and what does the transformer actually learn to predict at each training step?

<details>
<summary>Answers</summary>

1. B is 1024x16 and A is 16x1024, so 16,384 + 16,384 = 32,768 trainable parameters versus about 1 million for the full matrix, a roughly 97% reduction. Most fine-tuning changes lie in a low-dimensional subspace, which is why quality holds up.
2. CogVideoX requires frame counts that follow the 8N+1 pattern: 9, 17, 25, 33, 41, 49, and so on. A 45-frame clip breaks the latent alignment. Use 17x384x384 for testing or 49x480x720 for production.
3. Choose SFT when you have 40 GB+ of GPU memory (or multiple GPUs), more than about 1,000 training videos, a large domain shift such as medical imaging, and you need the last 5-10% of quality. ZeRO-2 partitions optimizer states and gradients across devices and can offload the optimizer to CPU RAM, which is how a 5B model trains on a single A100.
4. The VAE and the T5 text encoder stay frozen; only the LoRA adapter parameters in the transformer receive gradients. Each step encodes the video to latents, adds noise at a random diffusion timestep, and trains the transformer to predict that noise with an MSE loss.

</details>

## About the book

*Video Generation with AI* by Joseph Enochs, published by O'Reilly Media. Print ISBN 9798341653344.
