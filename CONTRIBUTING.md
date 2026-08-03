# Contributing

Thanks for your interest in improving the companion code for *Video Generation with AI*. Fixes, clarifications, and better examples are all welcome.

## Reporting a bug in an example

Open an issue on the [issue tracker](https://github.com/jenochs/video-ai-book-companion/issues) and include:

- the chapter and notebook (for example, chapter-05, `chapter_5_companion_notebook.ipynb`)
- the cell that fails (its number or the first line of its code)
- the full error message or traceback
- where you ran it (Colab or local, plus GPU type if relevant)

That is usually enough to reproduce the problem quickly.

## Proposing a fix

1. Fork the repository.
2. Create a branch for your change (`fix/chapter-05-lora-config` or similar).
3. Make the change and run the affected notebook top to bottom to confirm it works.
4. Open a pull request against `main` with a short description of what was broken and how you fixed it.

Small, focused pull requests are easiest to review and merge.

## Style notes

- Notebooks must run top to bottom in Colab with no manual intervention.
- Keep heavy downloads (model weights, large datasets) behind opt-in flags so a quick read-through stays cheap.
- Match the existing structure: notebook code that is worth reusing belongs in the chapter's `src/` modules.
- Keep chapter `requirements.txt` files minimal and pinned only where a version actually matters.

Questions that are not bugs are welcome on the issue tracker too.
