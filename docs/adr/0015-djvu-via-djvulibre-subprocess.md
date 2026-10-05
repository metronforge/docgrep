# ADR-0015: Read DjVu text through djvulibre subprocesses

**Status:** Accepted
**Date:** 2026-10-05

## Context

The library contains DjVu books, many of them with an embedded text layer from earlier OCR. docgrep needs that text for indexing and needs to know which DjVu files have no usable text for scan discovery. The reference implementation for DjVu is djvulibre, whose command-line tools (`djvused`, `djvutxt`) are packaged by Linux distributions. Python bindings exist but are not widely packaged and would add a compiled dependency.

## Decision

DjVu text is read by running the djvulibre command-line tools as subprocesses: `djvused -e n` for the page count and `djvutxt --detail=page` for the text of each page. They are invoked with an argument list and an absolute path, never through a shell, with a timeout and a cap on the captured output, behind a small runner interface that tests replace with a fake. No Python binding is used.

The tools are an optional system dependency. When they are missing, only DjVu sources fail, with the reason `djvu_tools_missing`; everything else is indexed, and the DjVu sources are processed again by a later `docgrep update` once the tools are installed.

The tools are invoked as separate programs, not linked into docgrep or bundled with it, so the licence position described in ADR-0014 is unaffected.

OCR of DjVu files and an OCR registry are out of scope here.

## Consequences

Installing djvulibre (on Debian-family systems the `djvulibre-bin` package) is enough to index DjVu text layers; nothing changes in the Python environment. Each DjVu file costs two process starts per build or update, which is small next to embedding its text. The parser depends on the output format of `djvutxt --detail=page`; a change in that format shows up as `djvu_invalid` failures, not as wrong text.
