# Chapter 7 - Audiovisual Synchronization with Cross-Modal Fusion

Companion code for Chapter 7 of *Video Generation with AI* (O'Reilly) by Joseph Enochs.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jenochs/video-ai-book-companion/blob/main/chapter-07/notebooks/chapter_7_companion_notebook.ipynb)

## Overview

The video generation systems from earlier chapters produce silent films: beautiful to watch, missing half the story. Humans detect lip-sync errors as small as 45 ms, about two frames at 24 FPS, which is why adding audio after the fact fails. This chapter takes the other road: generating audio and video simultaneously, with synchronization enforced by the architecture itself.

The chapter is built around the LTX-2 framework and its unified Diffusion Transformer: 48 layers, each carrying a larger video stream and a smaller audio stream joined by bidirectional cross-modal attention. Audio tokens attend to video tokens and vice versa at every layer, so alignment emerges from training on paired data rather than from explicit supervision. The companion notebook rebuilds each piece of that pipeline at a scale that runs on a free Colab CPU, then shows how to drive the real 19B+ model when you have the hardware.

## What you will build

- A mel-spectrogram front end (the `AudioProcessor`) that converts waveforms into the 64-bin, 100-frames-per-second representation LTX-2 uses, tested on a synthetic scene with click events you can see land on exact frames
- A runnable miniature of the unified DiT: the frozen `Modality` dataclass, AdaLN timestep modulation, and the `BasicAVTransformerBlock` with all four attention types (self, text cross, audio-to-video, video-to-audio)
- A demonstration that perturbing only the audio input changes the video prediction, the signature of bidirectional cross-modal fusion
- The audio latent machinery: per-channel statistics normalization, the `AudioPatchifier` with real-time timecodes (40 ms per latent frame), and a HiFi-GAN style vocoder with 240x upsampling and multi-receptive-field fusion
- A complete toy two-stage inference run: 40 CFG-guided Euler steps for structure, 2x upsampling, then 4 distilled sigma steps without CFG for detail, with both modalities denoised jointly at every step
- Opt-in recipes for the real LTX-2: two-stage text-to-video, image-to-video, speech synthesis with `<S>...<E>` tags, and LoRA fine-tuning with `with_audio: true`

## Section outline

Following the chapter's structure:

1. **The Missing Modality: Audio** - why post-hoc audio fails and why joint generation is the answer
2. **Audio Representation: From Waveforms to Mel-Spectrograms** - sheet music for machines
3. **Introducing LTX-2** - the unified 48-layer DiT, generation modes, and model specifications
4. **Unified DiT Architecture** - the LTXModel design, the Modality dataclass, bidirectional cross-modal attention, and AdaLN
5. **Audio Pipeline Deep Dive** - mel processing, the audio VAE encoder and decoder, per-channel statistics, the HiFi-GAN style vocoder, and the AudioPatchifier
6. **Two-Stage Inference Pipeline** - the ModelLedger, CFG-guided Stage 1, spatial upsampling, distilled Stage 2, and final decoding
7. **Hands-On Implementation** - environment setup, hardware requirements, text-to-video, image-to-video, speech synthesis, and LoRA fine-tuning
8. **Architectural Evolution: From Research to Production** - what changed from Ovi's twin backbones to LTX-2's unified design

## Repository layout

```
chapter-07/
  notebooks/
    chapter_7_companion_notebook.ipynb   # Main companion notebook
  src/
    audio_representation.py    # Mel front end, latent stats, patchifier
    cross_modal_transformer.py # Modality, AdaLN, BasicAVTransformerBlock
    vocoder.py                 # HiFi-GAN style mel-to-waveform synthesis
    two_stage_inference.py     # CFG, Euler stepping, joint denoising loop
  assets/                      # Figures and supporting files
  requirements.txt
```

The `src/` modules are cleaned, importable versions of the notebook code, useful when you want the chapter's components inside your own projects.

## Running locally

```bash
git clone https://github.com/jenochs/video-ai-book-companion.git
cd video-ai-book-companion/chapter-07
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/chapter_7_companion_notebook.ipynb
```

Python 3.10 or newer is recommended. Everything except the clearly marked opt-in cells runs on CPU in a few minutes.

## Hardware notes

- **The notebook's core content**: any machine, CPU only, no downloads beyond pip packages. All audio examples use small synthetic signals.
- **Real LTX-2 inference**: 24 GB VRAM with FP8 quantization (512x768, 49 frames) or 48 GB+ for BF16 at 1024x1536, 121 frames. Expect roughly 40 GB of model downloads (19B checkpoint, spatial upsampler, distilled LoRA, Gemma 3 text encoder).
- **LoRA fine-tuning**: 32 GB VRAM with the low-VRAM config, 80 GB for the standard config.
- **Full fine-tuning**: 4-8x 80 GB GPUs with FSDP.

These cells are guarded by a `RUN_LTX2_PIPELINE` flag that defaults to `False`, so the notebook always runs top to bottom.

## Check your understanding

1. Why does generating video first and adding audio afterward fail, even with a perfect analysis of the generated video?
2. In the `BasicAVTransformerBlock`, cross-modal attention runs after text cross-attention rather than before it. Why does that ordering matter for synchronization?
3. The LTX-2 vocoder upsamples by exactly 240x. Where does that number come from?
4. Stage 2 of the inference pipeline uses only 4 steps and no classifier-free guidance. Why can it get away with so much less work than Stage 1?

<details>
<summary>Answers</summary>

1. Human perception demands alignment that visual cues cannot fully specify: lip-sync errors of about 45 ms (roughly two frames at 24 FPS) are detectable, and phoneme-level sync requires 40-80 ms precision. Inferring exact event timing from ambiguous rendered frames is a hard inverse problem, so post-hoc audio is fundamentally limited. Joint generation sidesteps it by letting cross-modal attention coordinate both modalities during denoising.

2. By step 3 of the block, both streams have already incorporated the text prompt. Audio tokens therefore attend to video tokens that carry the prompt's description of the visual scene, and vice versa. Cross-modal attention on these text-conditioned states lets each modality align to what the other is actually going to depict, which is what makes the timing coordination meaningful.

3. Mel-spectrograms arrive at about 100 frames per second (16 kHz sample rate divided by the 160-sample hop), and the output waveform needs 24,000 samples per second. 24,000 / 100 = 240, factored as 6 x 5 x 2 x 2 x 2 across five transposed-convolution stages, with larger factors applied while channel counts are high.

4. Stage 1 already established structure: motion, layout, and coarse audio alignment, all guided by CFG at scale 4.0 over 40 steps. Stage 2 starts from the upsampled Stage 1 latent rather than from pure noise, so it only needs to sharpen detail. A distilled LoRA trained for few-step refinement covers the remaining denoising in 4 sigma values, and guidance is unnecessary because the content is already prompt-aligned.

</details>

## The book

*Video Generation with AI* by Joseph Enochs is published by O'Reilly Media (Print ISBN 9798341653344). The book covers the full journey from diffusion foundations to production video generation systems; this chapter's deep dive into audio-visual synchronization builds directly on the video generation and understanding systems of Chapters 3-6, and leads into ComfyUI production workflows in Chapter 8.
