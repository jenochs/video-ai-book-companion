# Video Generation with AI - Companion Repository

Code, notebooks, and reusable modules for the book **Video Generation with AI**.

Joseph Enochs, O'Reilly Media. Print ISBN 9798341653344, Ebook ISBN 9798341653337. [Book page on O'Reilly](https://www.oreilly.com/library/view/video-generation-with/9798341653337/).

## About this repository

This repository contains the code, notebooks, and reusable modules for every chapter of the book. Each chapter folder is self contained: it has its own README with an overview and self-check questions, a runnable companion notebook, a `src/` directory with importable modules, and a `requirements.txt` scoped to that chapter. You can work through the chapters in order or jump straight to the topic you care about.

## Chapters

| Chapter | Title | Notebook |
| ------- | ----- | -------- |
| 1 | [Foundations of Video Generation](chapter-01/) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jenochs/video-ai-book-companion/blob/main/chapter-01/notebooks/chapter_1_companion_notebook.ipynb) |
| 2 | [Understanding and Preparing Video Data for Model Training](chapter-02/) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jenochs/video-ai-book-companion/blob/main/chapter-02/notebooks/chapter_2_companion_notebook.ipynb) |
| 3 | [Implementing Video Generation Pipelines](chapter-03/) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jenochs/video-ai-book-companion/blob/main/chapter-03/notebooks/chapter_3_companion_notebook.ipynb) |
| 4 | [Training Video Generation Models](chapter-04/) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jenochs/video-ai-book-companion/blob/main/chapter-04/notebooks/chapter_4_companion_notebook.ipynb) |
| 5 | [Fine-Tuning for Specific Video Tasks](chapter-05/) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jenochs/video-ai-book-companion/blob/main/chapter-05/notebooks/chapter_5_companion_notebook.ipynb) |
| 6 | [Vision Language Models for Video Understanding](chapter-06/) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jenochs/video-ai-book-companion/blob/main/chapter-06/notebooks/chapter_6_companion_notebook.ipynb) |
| 7 | [Audiovisual Synchronization with Cross-Modal Fusion](chapter-07/) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jenochs/video-ai-book-companion/blob/main/chapter-07/notebooks/chapter_7_companion_notebook.ipynb) |
| 8 | [From Models to Systems](chapter-08/) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jenochs/video-ai-book-companion/blob/main/chapter-08/notebooks/chapter_8_companion_notebook.ipynb) |

## Quick start

Clone the repository:

```bash
git clone https://github.com/jenochs/video-ai-book-companion.git
cd video-ai-book-companion
```

Then pick one of two paths:

**Run in Colab.** Click an "Open in Colab" badge in the table above (or in any chapter README). This is the easiest way to try the notebooks, and Colab gives you a free GPU for the chapters that need one.

**Run locally.** Change into a chapter folder, install its requirements, and start Jupyter:

```bash
cd chapter-01
pip install -r requirements.txt
jupyter lab
```

A root `requirements.txt` with the shared core dependencies is also provided if you want one environment that covers the basics across chapters.

## Hardware guidance

- **Chapters 1, 5, 6, and 7 want a GPU.** The free T4 in Colab is enough for chapter 1 (CogVideoX-2B generation) and chapter 6 (small Qwen VL models). The LoRA fine tuning in chapter 5 is happier with 16 GB of VRAM or more. Chapter 7 (LTX-2 style audio video generation) also benefits from a larger GPU.
- **Chapters 2, 3, 4, and 8 run on CPU.** Their notebooks use tiny demo settings by default, so you can follow along on a laptop. Heavier configurations are noted in each chapter README.

## Repository layout

Each chapter folder, `chapter-01` through `chapter-08`, follows the same structure:

```
chapter-NN/
  notebooks/         # the runnable companion notebook
  src/               # importable Python modules used by the notebook
  assets/            # configs, sample inputs, and supporting files
  requirements.txt   # dependencies for this chapter
  README.md          # overview, section outline, and self-check questions
```

## Errata and questions

Found a bug in an example, or a cell that will not run? Please open an issue on the [GitHub issue tracker](https://github.com/jenochs/video-ai-book-companion/issues). For errors in the book text itself, use the [O'Reilly errata page](https://www.oreilly.com/library/view/video-generation-with/9798341653337/) for the book. See [CONTRIBUTING.md](CONTRIBUTING.md) if you would like to submit a fix.

## License

The code in this repository is released under the MIT License. See [LICENSE](LICENSE).
