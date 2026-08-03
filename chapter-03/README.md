# Chapter 3 - Implementing Video Generation Pipelines

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jenochs/video-ai-book-companion/blob/main/chapter-03/notebooks/chapter_3_companion_notebook.ipynb)

Companion code for Chapter 3 of *Video Generation with AI* by Joseph Enochs.

## Overview

Chapter 2 covered preparing video data. This chapter builds the architecture that turns prepared data into a functioning generative model, using Latte (Latent Diffusion Transformer) as the concrete reference. The components you implement here, latent compression, tokenization, factorized attention, diffusion, and text conditioning, generalize beyond any single generation model.

The companion notebook implements every code listing from the chapter, fixed up to run top to bottom on a free Colab instance. It creates its own tiny synthetic video clip, so there is nothing to download, and the one heavy dependency (the T5 text encoder) sits behind an opt-in flag.

## What you will build

- A 3D convolutional VAE encoder that compresses video 8x in space while preserving temporal resolution, plus the reparameterization trick and a deterministic inference path
- A video tokenizer and patch embedding module with dual (spatial and temporal) sinusoidal positional encodings
- Spatial and temporal attention blocks, including causal masking for autoregressive generation, arranged in Latte's alternating pattern
- A minimal diffusion setup: linear-beta noise scheduler, forward noising visualization, and the MSE noise-prediction training loop
- Text conditioning with T5 embeddings, cross-attention where video patches query the text, AdaLN timestep modulation, and classifier-free guidance
- A complete toy pipeline: assemble the model, train it for a few steps on the synthetic clip, sample from noise with guidance, and decode back to pixels

## Section outline

Following the chapter's structure:

1. Building the Foundation: Data Pipeline and Encoding
   - Latent Space Compression with VAE
   - Understanding 3D Convolutions for Video VAE
2. Video Tokenization: From Pixels to Patches
3. Patch Embedding and Positional Encoding
4. Attention Mechanisms: Understanding Relationships in Space and Time
   - Spatial Attention: Coherence Within Frames
   - Temporal Attention: Continuity Across Frames
   - The Alternating Block Architecture
5. The Diffusion Process: From Noise to Video
   - Forward and Reverse Processes
   - Noise Prediction and Scheduling
6. Adaptive Conditioning with AdaLN
7. Conditioning and Control: From Text to Video
   - Text Encoding with T5
   - Cross-Attention Integration
   - Classifier-Free Guidance
8. Latte Model Component Integration
9. Putting It All Together: The Generation Pipeline

## Repository layout

```
chapter-03/
  notebooks/chapter_3_companion_notebook.ipynb   the runnable companion notebook
  src/video_vae.py          VAE encoder and minimal decoder
  src/patch_embedding.py    patch embedding with dual positional encoding
  src/attention_blocks.py   spatial, temporal, cross attention, AdaLN
  src/latte_model.py        transformer block, timestep embedding, assembled model
  src/diffusion.py          noise scheduler, training loop, classifier-free guidance
  requirements.txt
```

## Running locally

```bash
git clone https://github.com/jenochs/video-ai-book-companion.git
cd video-ai-book-companion/chapter-03
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/chapter_3_companion_notebook.ipynb
```

To reuse the modules in your own scripts, add `src/` to your path:

```python
import sys
sys.path.append('path/to/chapter-03/src')

from video_vae import VideoVAEEncoder
from latte_model import LatteVideoModel
```

## Hardware notes

- The notebook runs end to end on CPU in a few minutes; every demo uses a tiny synthetic clip and a roughly 1M parameter model.
- A GPU (any Colab T4 or better) speeds up the training and sampling cells but is not required.
- Setting `LOAD_T5 = True` downloads the t5-small encoder (about 250 MB) and benefits from a GPU; the default uses stand-in embeddings and stays offline.
- Training a real Latte-B (151M parameters) is a multi-GPU job and is the subject of Chapter 4. Table 3-2 in the chapter explains why: full 3D attention at 256 x 256 would need roughly 256 GB of attention memory, which the alternating design reduces to about 5 GB.

## Check your understanding

1. Why do video VAE encoders use kernel sizes like (1, 3, 3) and (1, 4, 4), with the leading 1 on the temporal axis?
2. The patch embedding module adds two separate positional encodings instead of one. What does each encode, and why does the model need both?
3. What problem does the alternating spatial/temporal block design solve compared to full 3D attention, and what is the cost of the factorization?
4. During diffusion training, what does the model actually predict at each step, and how does classifier-free guidance reuse that prediction at inference time?

<details><summary>Answers</summary>

1. The leading 1 keeps frames temporally independent inside the encoder: the VAE's job is to compress space, not to model motion. The (1, 3, 3) kernels extract spatial features without temporal blur, and the (1, 4, 4) kernels with stride (1, 2, 2) halve spatial resolution while leaving the frame count untouched, for an 8x total spatial reduction.

2. The spatial encoding tells each patch where it sits within its frame, and the temporal encoding tells it which frame it belongs to. Separating them lets the model reason about "where in the frame" independently of "when in the sequence," which is exactly the split the spatial and temporal attention blocks rely on.

3. Full 3D attention over all tokens costs O((HWT)^2), roughly 256 GB of attention memory at 256 x 256, which is impractical. Alternating spatial and temporal blocks reduces this to O(T x H^2 x W^2 + T^2 x H x W), around 5 GB. The cost is a blind spot: spatial layers never see time and temporal layers never see the whole frame, so coherence across both must emerge over many alternations.

4. The model predicts the noise that was added at the sampled timestep, trained with a simple MSE objective; predicting noise is more stable than predicting the clean video directly. At inference, classifier-free guidance runs the model twice, with and without the text prompt, and forms the final prediction as the unconditional estimate plus a guidance weight times the difference, sharpening prompt adherence.

</details>

## The book

This companion accompanies *Video Generation with AI* by Joseph Enochs (O'Reilly, Print ISBN 9798341653344). The book explains the concepts, architecture decisions, and production context behind every component implemented here.
