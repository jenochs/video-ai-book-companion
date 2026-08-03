# Chapter 8 - From Models to Systems

Companion materials for Chapter 8 of *Video Generation with AI* (O'Reilly) by Joseph Enochs.

## Overview

A model is not a product. This chapter moves from driving LTX-2 out of Python (Chapter 7) to the systems work that turns a checkpoint into something an actual person can install and use: capability detection, license state, model downloads with recovery, progress streaming, security, and concurrency. It does so through eight architectural patterns drawn from LTX Desktop, read side by side with ComfyUI's community plugin mesh and InvokeAI's service-oriented monolith, and ends with a capstone that implements the same custom feature (exposing LTX-2.3's audio-to-video pipeline) on all three platforms.

This is a systems chapter, so the companion notebook needs no GPU. Service and API code runs as in-notebook demonstrations: handler functions are invoked directly instead of starting long-lived servers, GPU pipelines arrive through the same service seam production uses, with hand-written fakes plugged in. Platform-specific code (Electron TypeScript, the ComfyUI node, the InvokeAI invocation) appears as clearly labeled reference listings or is written to installable files.

## What you will build

- The `decide_force_api_generations` capability policy, exercised across macOS, Windows, Linux, and unknown-platform scenarios
- A Python analog of the typed IPC bridge: one schema, validation at the boundary in both directions
- Per-session bearer tokens with constant-time comparison, plus a cross-platform whitelist path validator
- The `DownloadingSession` per-file progress state and the five-variant `GenerationState` discriminated union with exhaustive pattern matching
- Service Protocols with hand-written fakes and a `ServiceBundle` composition seam (no mock libraries)
- A lock-aware `AudioToVideoHandler` running end to end: submit, poll for state transitions, observe the concurrency guard reject a second job
- The capstone extension for all three platforms: the LTX Desktop handler (run against fakes), a ComfyUI custom node file, and an InvokeAI typed invocation file

## Section outline

Following the chapter's headings:

1. The Gap Between a Model and a Product
2. What "Production AI System" Means in 2026
3. Three Reference Architectures
4. LTX Desktop: Eight Patterns
   - Pattern 1: The Two-Process Architecture
   - Pattern 2: Backend as Source of Truth
   - Pattern 3: The Typed IPC Bridge
   - Pattern 4: Localhost as a Security Boundary
   - Pattern 5: Model Lifecycle: Download, License, Storage
   - Pattern 6: State as Discriminated Unions
   - Pattern 7: Service Protocols and Fakes (No Mocks)
   - Pattern 8: Lock-Aware Handlers
5. The Landscape: Three Architectures, One Problem (ComfyUI, InvokeAI, comparison matrix)
6. Capstone: Building a Custom Extension Across Three Platforms
7. Architectural Principles Recap
8. Summary and Conclusion: The Video AI Landscape

## Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jenochs/video-ai-book-companion/blob/main/chapter-08/notebooks/chapter_8_companion_notebook.ipynb)

`notebooks/chapter_8_companion_notebook.ipynb` runs top to bottom on any Colab runtime, CPU included.

## Source modules

Reusable implementations of the patterns live in `src/`:

- `runtime_policy.py`: capability detection and the published runtime policy (Pattern 2)
- `security.py`: session tokens, constant-time comparison, path validation (Pattern 4)
- `state_machine.py`: `DownloadingSession` and the `GenerationState` discriminated union (Patterns 5 and 6)
- `services.py`: service Protocols, fakes, and the `ServiceBundle` (Pattern 7)
- `handlers.py`: `HandlerBase`, `with_state_lock`, `TaskRunner`, and the capstone `AudioToVideoHandler` (Pattern 8 and capstone)

## Running locally

```bash
git clone https://github.com/jenochs/video-ai-book-companion.git
cd video-ai-book-companion/chapter-08
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/chapter_8_companion_notebook.ipynb
```

To use the modules directly:

```bash
cd src
python -c "from runtime_policy import build_runtime_policy; print(build_runtime_policy().as_payload())"
```

## Hardware notes

- The notebook and all `src/` modules run on CPU; Python 3.10 or newer is required for structural pattern matching.
- No model weights are downloaded. GPU pipelines are represented by fakes behind the same Protocols production uses.
- Actually running the capstone against LTX-2.3 requires the platform in question (LTX Desktop fork, ComfyUI, or InvokeAI), the LTX-2.3 model files (roughly 44 GB checkpoint, 7 GB upsampler, 25 GB Gemma 3 encoder), and an NVIDIA GPU with at least 15 GB of VRAM, the same threshold the chapter's capability policy enforces.

## Check your understanding

1. Why does `decide_force_api_generations` return True for an unknown platform, and what is the general principle behind that choice?
2. In the typed IPC bridge, why does the handler validate both the input and the output against the shared schema?
3. What does the discriminated-union `GenerationState` make impossible that the single nullable-field struct allows?
4. The `AudioToVideoHandler.start()` method holds the state lock, but the pipeline call in `_run()` does not. Why is that split essential?

<details><summary>Answers</summary>

1. It fails closed: on a platform the policy was never tested against, routing to the hosted API is the safe default. The principle is that capability decisions should degrade predictably rather than optimistically, and they are made once, in one layer, then published.
2. Input validation catches malformed or malicious renderer payloads at the boundary; output validation catches handler bugs that produce unexpected response shapes. Both sides check against the same source, so the contract cannot drift silently.
3. Invalid combinations of fields, such as a "complete" state with no output path or a "running" state carrying an error message. Each variant carries only the fields meaningful to its phase, so illegal states are unrepresentable and a type checker can verify that every match handles every variant.
4. Holding the lock during a 90-second generation would serialize every other state read and write, freezing the UI. The discipline is: mutate state briefly under the lock, submit heavy work to the TaskRunner, and have the worker reacquire the lock only for the short writes that record progress and completion.

</details>

## The book

Chapter 8 of *Video Generation with AI* by Joseph Enochs, published by O'Reilly Media. Print ISBN 9798341653344.
