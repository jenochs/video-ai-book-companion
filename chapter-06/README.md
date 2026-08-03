# Chapter 6 - Vision Language Models for Video Understanding

Companion code for Chapter 6 of *Video Generation with AI* (O'Reilly) by Joseph Enochs.

Chapters 3-5 generated video from text with diffusion models. This chapter goes the other direction: vision language models (VLMs) take video plus a question and produce text, turning video understanding into a language modeling problem. Working with Qwen3-VL, you will study the three-component architecture (vision encoder, projector, language model), see how 3D positional encoding gives the model spatial-temporal awareness, and build a complete video question-answering system that you can fine-tune for your own domain with LoRA.

## What you will build

- A `VideoQA` class that answers natural language questions about any video file
- A `VideoQAWithGrounding` extension that returns timestamped event timelines as structured segments
- Runnable implementations of the projector, patch merger, and 3D position index construction from the Qwen3-VL architecture
- A preprocessing toolkit: FPS-based frame extraction, smart resizing, normalization, adaptive FPS, and token budget estimation
- A LoRA fine-tuning setup for adapting Qwen3-VL to domain-specific video, plus a simple evaluation framework

## Section outline

Following the manuscript's structure:

1. From Generation to Understanding - the paradigm shift from diffusion to autoregressive token prediction
2. Introducing Qwen3-VL - model family, Interleaved-MRoPE, DeepStack, and long context support
3. VLM Architecture - the three-component model: vision encoder, projector, language model
4. Putting It Together: A VideoQA Class - the complete pipeline as a reusable class
5. 3D Positional Encoding for Video - temporal, height, and width indices; timestamps vs frame indices; temporal grounding
6. Video Understanding Pipeline - frame extraction, smart resizing, normalization, task types, complete inference, handling long videos
7. Adapting to Your Domain: Dataset Preparation - Qwen conversation format for training data
8. Training with LoRA - parameter-efficient fine-tuning and using fine-tuned models
9. Evaluation - measuring video QA accuracy

## Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jenochs/video-ai-book-companion/blob/main/chapter-06/notebooks/chapter_6_companion_notebook.ipynb)

The notebook runs top to bottom on a free Colab runtime. Architecture and preprocessing sections work on CPU; cells that download a full Qwen3-VL checkpoint are gated behind an opt-in `RUN_MODEL_DEMOS` flag, and the LoRA training loop behind `RUN_FINETUNING`. It defaults to the 2B checkpoint (~4 GB download) rather than the 8B model used in the book so it fits on a T4 GPU.

## Running locally

```bash
git clone https://github.com/jenochs/video-ai-book-companion.git
cd video-ai-book-companion/chapter-06
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/chapter_6_companion_notebook.ipynb
```

The reusable modules live in `src/`:

- `vlm_components.py` - projector, patch merger, token counting, 3D position index construction
- `preprocessing.py` - frame extraction, smart resize, normalization, adaptive FPS, token budgets
- `video_qa.py` - `VideoQA`, `VideoQAWithGrounding`, `TemporalSegment`
- `finetune.py` - LoRA configuration, fine-tuned model loading, evaluation framework

## Hardware notes

- No GPU needed: architecture demos, preprocessing, 3D position encoding, and the temporal segment parser all run on CPU.
- 16 GB VRAM (Colab T4): Qwen3-VL-2B-Instruct inference in bfloat16, the notebook default.
- 18 GB+ VRAM: Qwen3-VL-8B-Instruct, the checkpoint used in the book. On smaller cards, use 4-bit quantization.
- Fine-tuning: LoRA on the 8B model wants 24 GB+ VRAM with batch size 1 and gradient accumulation; the 2B model trains comfortably on 16 GB.
- `flash-attn` is optional and only helps on supported NVIDIA GPUs.

## Check your understanding

1. What are the three components of a VLM, and what does each one do?
2. With `spatial_merge_size=2`, an 8-frame video produces 2,048 raw vision tokens. How many reach the language model, and why?
3. Why does Qwen3-VL give each vision token three position indices instead of one sequential index?
4. Which layers does the chapter's LoRA configuration target, and roughly what fraction of parameters does it train?

<details>
<summary>Answers</summary>

1. The vision encoder converts frames into sequences of visual feature vectors, the projector (merger) maps those features into the language model's embedding space while reducing token count, and the language model processes the combined visual and text tokens to generate the response.
2. 512 tokens. The patch merger pools each 2x2 group of adjacent patches into a single token, a 4x reduction (75% fewer tokens), while projecting into the LLM embedding space.
3. A single sequential index loses spatial-temporal structure: the model could not tell whether two tokens come from the same place in different frames or different places in the same frame. Separate temporal (t), height (h), and width (w) indices let attention capture motion across frames, spatial relationships within a frame, and temporal grounding to timestamps.
4. The attention projections of the language model: `q_proj`, `k_proj`, `v_proj`, and `o_proj`. With r=64 and alpha=128, roughly 0.5% of the model's parameters are trainable.

</details>

## The book

This repository accompanies *Video Generation with AI* by Joseph Enochs, published by O'Reilly Media (Print ISBN 9798341653344). The book covers the full arc from diffusion-based video generation through video understanding and multimodal systems; this chapter's companion code covers the video understanding side.
