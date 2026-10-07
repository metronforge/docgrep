# ADR-0017: OCR through the ocrmypdf command, with a registry and sibling delegation

**Status:** Accepted
**Date:** 2026-10-07
**Builds on:** ADR-0005 (SQLite for manifest and OCR registry), ADR-0015 (DjVu through djvulibre subprocesses).

## Context

Scan discovery finds PDF and DjVu files without a usable text layer: scans without text, and PDFs whose text layer is garbage. They are skipped by the index until they are made searchable. OCR is slow (seconds per page) and runs as a deliberate, supervised batch, often overnight, so cancelling, progress and continuing an interrupted batch matter. The original files must never change, and the index must not hold the same book twice, once from the original and once from its OCR'd copy.

The established OCR tool is ocrmypdf, which drives Tesseract and writes a PDF with an invisible text layer. It can be installed as a Python package or as a system command; either way it needs Tesseract, Ghostscript and their language data from the operating system.

## Decision

1. **Engine.** OCR runs the system `ocrmypdf` command as a subprocess, not a Python extra: nothing heavy enters the lock file, the licence position of ADR-0014 is unaffected because the tool is invoked and not linked or bundled, and the dependency is optional at runtime: only `docgrep ocr` needs it, and without it that command fails with `ocr_tools_missing` and the Debian install command before any work. DjVu files are first converted with `ddjvu -format=pdf`. Both tools run with an argument list and absolute paths, never through a shell, in a process group of their own, with their error output capped, logged at DEBUG and scrubbed of paths before it reaches a failure message.
2. **Languages.** The default is `rus+eng+ukr` (`DOCGREP_OCR_LANGUAGES`), overridable per batch; every requested language must be listed by `tesseract --list-langs` before any work starts.
3. **Output.** The searchable PDF is written next to the original as `<stem>.ocr.pdf` (for a DjVu `book.djvu` it is `book.ocr.pdf`). It is produced in a hidden temporary file in the same directory, checked to open with the input's page count, and renamed into place without ever overwriting a file that is not this original's registered output. The original is never modified.
4. **Modes.** Files without text (or with too little) use `--skip-text`, which keeps any page that has text. A garbled text layer cannot be kept: `--skip-text` would leave those pages untouched, so such files use `--force-ocr`, which discards the existing text and rebuilds it from the page images.
5. **Registry and delegation.** Two tables in the manifest database (ADR-0005), added by an additive migration: the OCR registry (input path, format and content hash, output path and hash, engine and version, languages, mode, time) and a delegation table (input path and hash, output path, time) that marks an original as delegated to its OCR'd sibling. Both are written in one transaction per file. A registry entry is live while its output exists; a delegation is live while its output exists and the original's content hash still equals the recorded one. Build and update skip an original with a live delegation as `skipped:delegated_to_ocr` (removing passages it had), and index the output as an ordinary PDF. Deleting the output, or changing the original, ends the delegation and the original is indexed again. The registry, not the file name, decides what is an OCR output, and an output is never processed again.
6. **Lock and run state.** A batch holds the exclusive index lock for its whole duration and is recorded as a run of kind `ocr`, so `docgrep status` shows it and `docgrep cancel` reaches it. Build resume and update only consider build runs, so an OCR run never hides or fakes a resumable build.
7. **Resume.** There is no resume state: running the same command again skips every file whose registry entry is live (`skipped:already_processed`), which continues an interrupted or cancelled batch.
8. **Cancel.** Ctrl-C, SIGTERM and `docgrep cancel` stop a batch. In the default `terminate` mode (`DOCGREP_OCR_CANCEL_MODE`) the running tool's process group gets SIGTERM, then SIGKILL after 10 seconds; its temporary files are removed and the file is reported `cancelled`. In `finish` mode the current file completes first. Completed files stay registered.
9. **No default time limit.** By the owner's decision the per-file time limit `DOCGREP_OCR_TIMEOUT_SECONDS` defaults to 0, no limit: the owner watches long runs. When set, a file over the limit is stopped and reported `failed:timeout`.
10. **Opt-in override.** UC-6 refuses to OCR a file classified as native, to avoid damaging a good text layer. As a deliberate, opt-in deviation, `--override-native` (only with explicit paths) processes such a file anyway, always with `--force-ocr`: some PDFs carry a wrong text layer that looks fine to the classifier. The original stays untouched, so deleting the output reverts it.

## Consequences

Installing ocrmypdf and the Tesseract language packages from the distribution (Debian: `apt install ocrmypdf tesseract-ocr tesseract-ocr-eng tesseract-ocr-rus tesseract-ocr-ukr`) is enough; nothing changes in the Python environment. The corpus gains one `.ocr.pdf` file per processed original, visible to the user and to any other tool. A crash can leave a hidden `.part` file next to an output; it is not a document and can be deleted. Running `docgrep update` after a batch is a separate, deliberate step. A garbled DjVu text layer also uses `--force-ocr`, although DjVu conversions otherwise use `--skip-text`.
