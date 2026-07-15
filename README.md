# Video Generation with AI — Companion Repository

Companion code for the book **Video Generation with AI** by Joseph Enochs (O'Reilly).

Print ISBN 9798341653344 · Ebook ISBN 9798341653337

This repository holds the runnable code, notebooks, and assets referenced throughout the book,
organized by chapter on the `main` branch.

## Layout

```
chapter-02/
  notebooks/   # Jupyter/Colab notebooks
  src/         # Python modules and scripts
  assets/      # configs, sample inputs, and supporting files
chapter-03/
chapter-08/
```

## Chapters

| Chapter | Contents |
| ------- | -------- |
| [chapter-02](chapter-02/) | Understanding and preparing video data — the pet-video pipeline |
| [chapter-03](chapter-03/) | A scaled-down production data pipeline — filtering, deduplication, annotation |
| [chapter-08](chapter-08/) | Building a custom extension across three platforms (LTX Desktop, InvokeAI, ComfyUI) |

## Getting started

```bash
git clone https://github.com/jenochs/video-ai-book-companion.git
cd video-ai-book-companion
```

Notebooks are designed to run in Google Colab. Open any `.ipynb` above and use
**Open in Colab**, or run them locally with Jupyter.

## License

See [LICENSE](LICENSE).
