# ADR-0018: Read DOCX with python-docx in body order

**Status:** Accepted
**Date:** 2026-10-15

## Context

The library contains Word documents in the Office Open XML format (`.docx`). Discovery has recognised them since the first slice, but no parser was registered, so they were recorded as `skipped:unsupported`. Indexing them is part of the parsing step of UC-03.

A DOCX file is a ZIP package of XML parts. The text that matters for search is in the main document part, where paragraphs and tables alternate in reading order. Unlike PDF and DjVu, a DOCX has no fixed pagination: page breaks depend on the renderer, fonts and printer settings, and the page numbers stored in a file (rendered page breaks) are only a hint from the last application that saved it.

## Decision

DOCX files are parsed with the `python-docx` library, a core dependency like PyMuPDF. It is imported when the first DOCX is parsed; when it is missing, only DOCX sources fail, with the reason `docx_library_missing`, and the next `docgrep update` retries them like any failed source, as with `djvu_tools_missing` (ADR-0015).

- **Body order.** The children of the document body are read in order, paragraphs and tables interleaved. Paragraph text includes runs, hyperlinks and tracked insertions.
- **Tables.** Each non-empty row becomes one line, with cells joined by ` | `. A cell spanning several columns is one cell element and is read once; the continuation cells of a vertical merge are skipped, so merged text is never duplicated. The rows of a nested table follow the row that contains it.
- **Headings.** A paragraph is a heading when it has an outline level, set directly or through its style and the styles it is based on. Built-in heading styles carry an outline level in every UI language, so localized styles such as a Russian `Заголовок 1` are found without matching display names. Only when no outline level is found do the style ids and names `Heading N` and `Title` count. Outline level 9 (body text) is never a heading. `Title` sits above `Heading 1`. Headings build the heading path the same way as for EPUB.
- **Locator.** The location of a passage is its heading path and its position in the source (the chunk offset), as for EPUB. No page numbers are produced or estimated.
- **Metadata.** Title and author come from the core properties, only when the package has a core properties part. The creation date is not a publication year and is not used.
- **No extraction to disk.** The package is read in place from the file. Before python-docx reads it, the parser checks the file header and the ZIP directory. An OLE compound file (what Office writes for a password-protected DOCX) fails with `encrypted_content`. A file that is not a ZIP archive fails with `not_a_zip`. A ZIP without `[Content_Types].xml` fails with `not_ooxml`. Encrypted members fail with `encrypted_content`. Members larger than the EPUB member bound fail with `member_too_large`. Any other damage fails with `unreadable`. All of these are per-source parse errors, and failure messages carry no absolute paths.
- **No text.** A document whose body yields no text, for example one holding only images, is recorded as `skipped:requires_ocr`, the convention of EPUB. Scan discovery and the `ocr` command still cover only PDF and DjVu.

Only `.docx` is routed to the parser. `.doc`, `.docm`, `.dotx`, `.dotm`, `.rtf` and `.odt` stay unsupported.

## Out of scope

- **Headers and footers** repeat on every page and would add the same passage many times. They rarely carry content worth finding.
- **Footnotes and endnotes** are separate parts, and python-docx has no reading API for them. Placing them in the text would need anchors that the passage model does not have. They are left for a later decision.
- **Comments** are review annotations, not document content.
- **Tracked-change deletions** are text the author removed. Reading the document as it would be after accepting all changes (insertions in, deletions out) matches what a reader sees.
- **Text boxes and shapes** sit outside the body flow, in drawing markup. Their reading position is not defined, and some are decoration.
- **Content controls at body level** (for example a generated table of contents or a cover page) are not read. python-docx does not expose them, and a table of contents would only repeat the headings.
- **Images** are ignored. OCR of embedded images is out of scope here.

Parsing never fails because one of these is present. They are skipped.

## Consequences

`python-docx` and its dependency `lxml` join the core dependencies. Both are under permissive licences, so the licence position of ADR-0014 is unaffected. An existing environment needs the new dependency installed before DOCX files can be read. Until it is installed, DOCX files are recorded as failed with `docx_library_missing` and are retried by the next update.

Sources recorded as `skipped:unsupported` before this change are picked up by the next `docgrep update` without any new mechanism. Update already treats an unsupported source whose file type now has a registered parser as newly supported, and discovery already classified `.docx` files as DOCX. Tests show this for DOCX.

Passages from DOCX files have no page number, so citations point to a section. The parser declares version `1`. A later change to what it extracts, such as reading footnotes, raises the version, and the next update extracts DOCX sources again.

## Alternatives considered

- **docx2txt** returns the text of a document as one string. It has no styles, so no heading path, and no table structure. It also extracts images to disk on request, which is not wanted.
- **mammoth** converts DOCX to HTML through a style map, and the HTML could reuse the EPUB extraction. Its style map matches styles by name, so headings defined only by outline level (custom or localized styles) would be missed. The HTML step would also put a second representation between the file and the verbatim passages.
- **Converting through LibreOffice** (`soffice --headless --convert-to`) would handle `.doc`, `.rtf` and `.odt` too. It is rejected because it adds a large system dependency, and the conversion is slow and stateful: it needs a user profile, it is unsafe to run concurrently, and each file needs a process start and temporary files.
