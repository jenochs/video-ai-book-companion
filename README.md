# Video Generation with AI — Companion Repository

Companion code for the book **Video Generation with AI** by Joseph Enochs (O'Reilly).

Print ISBN 9798341653344 · Ebook ISBN 9798341653337

This repository holds the runnable code, notebooks, and assets referenced
throughout the book. Code is organized by chapter, with each chapter also
available on its own branch (e.g. `chapter-8`) pinned to the commits the book cites.

## Layout

```
chapter-08/
  src/         # Python modules and scripts
  notebooks/   # Jupyter notebooks
  assets/      # configs, sample inputs, and supporting files
```

## Chapter 8 — Building a Custom Extension Across Three Platforms

The Chapter 8 capstone exposes LTX-2.3's audio-to-video pipeline as a
user-accessible feature across three reference architectures:

- **LTX Desktop** — vendor-controlled extension
- **Invoke AI** — typed invocations, mid-sized ecosystem
- **ComfyUI** — duck-typed nodes

Check out the pinned branch for this chapter:

```bash
git checkout chapter-8
```

## Getting started

```bash
git clone https://github.com/jenochs/video-ai-book-companion.git
cd video-ai-book-companion
```

## License

See [LICENSE](LICENSE).
