# ADR-0021: Sanitize source metadata on read

**Status:** Accepted
**Date:** 2026-10-17

## Context

Parsers store a source's title, author and year exactly as the file declares them. On a real library a share of those values is junk: years such as `101` (a date like `0101-...` read as a year) and `1601` (the Windows FILETIME epoch), authors such as `Unknown` or `python-docx`, titles such as `book title`, a bare number or the file name, and OCR outputs whose year is the date they were processed. The junk is shown in browse and search results, and it satisfies filters: `year_to: 1990` returned modern books whose stored year is `101`. Later features (citations) must not use these values either. Issue #22.

## Decision

**Sanitize on read; the stored values stay untouched.** No parser, manifest row or vector payload changes, and no reindex or migration is needed. The rules can change later without touching the index, and the raw values stay available for diagnosis.

**One pure function in the domain.** `sanitize_source(source, current_year=..., origin=...)` in `docgrep.domain.metadata` returns the cleaned `Source` and a `MetadataAudit` of what was dropped or inherited, with the raw value and a reason code per field. It does no I/O; the current year comes from the caller's injected clock.

**Applied wherever a source is shown or matched**, so every front agrees:

- browse: before the filter, `has_metadata`, `metadata_dropped`, sorting and paging;
- search: when the filter is resolved against the manifest, so `matches()` and `sources_excluded_for_missing_metadata` see the cleaned values, and on the returned passages' sources after ranking (not on the first-stage candidates).

**Dropping means null.** No value is invented, guessed or derived: no year or title from the file name, no author from the path.

**The rules.** Values are compared stripped, with inner whitespace collapsed and case-folded; a placeholder must match the whole value.

- *Year* is dropped below 1450 or above the current year (`implausible_year`), when it is exactly 1601 (`filetime_epoch`), and, for an OCR output, when it equals the calendar year of the registry entry's `created_at` (`processing_year`).
- *Author* is dropped when empty (`empty`), made only of digits, spaces, dots and dashes (`numeric_only`), or one of a small placeholder table (`placeholder`): `unknown`, `anonymous`, `n/a`, `none`, `-`, `author`, `admin`, `administrator`, `user`, `microsoft office user`, `windows user`, `python-docx`.
- *Title* is dropped when empty, numeric-only, a placeholder (`untitled`, `book title`, `title`, `unknown`, `document`, `document1`, `new document`, `scan`), or a file name (`file_name`): it starts with `Microsoft Word - ` or `Microsoft Excel - `, ends with an authoring-file extension (`.doc`, `.docx`, `.indd`, `.qxd`, `.pmd`, `.tex`, `.rtf`, `.cdr`, `.pdf`), or equals the source's file name or stem.

The rules are conservative: anything not covered is kept, including long or all-capital titles, several authors in one field and every language.

**OCR outputs inherit from their original.** For a PDF that is the live output of an OCR registry entry (ADR-0017), a missing or dropped title, author or year is taken from the original's manifest row once that row's own value has passed the same rules (an original is never treated as an OCR output). An original whose value is junk too leaves the field null. The registry is read once per request: `OcrOutputOwners.live_outputs()` returns every live output path with its input path and processing stamp in one query. Browse derives `ocr_status` from the same map.

**The audit is visible.** Each browse entry carries `metadata_dropped`, present only when something was dropped or inherited: per field the `raw` value, the `reason` (`implausible_year`, `filetime_epoch`, `processing_year`, `placeholder`, `numeric_only`, `file_name`, `empty`, or `inherited` for a missing value taken from the original) and `inherited_from` (the original's path). The browse filter `metadata_dropped` (true: only affected sources, false: only the others) lists them; `docgrep sources --dropped-metadata` sets it. Search results show cleaned values and carry no audit; the search filter has no `metadata_dropped` field.

## Known limits

- Year 1970 (the Unix epoch) and 1900 are kept: both are legitimate publication years, and only the parser can tell an epoch default from a real date.
- Epoch detection inside the EPUB and PDF parsers and publisher-specific date fields are left to parser-level fixes.
- An original that is delegated to its OCR'd sibling is skipped before parsing, so its manifest row often has no metadata to inherit; inheritance helps where the row has some.
- A title is compared with the source's own file name only: an OCR output `book.ocr.pdf` titled `book` keeps that title.

## Consequences

Junk years no longer satisfy year bounds and count as missing metadata, and junk authors count as missing for `has_metadata`. Every request that sanitizes reads the OCR registry once and, for a search that returns an OCR output without a filter, the manifest once. Changing a rule changes what all fronts show at once, without a rebuild.

## Alternatives considered

- **Clean at parse time and store the cleaned values:** needs a reindex for every rule change and loses the raw values.
- **A migration that rewrites the manifest and the payloads:** the same loss, and two stores to keep consistent.
- **Fall back to a year or title taken from the file name:** a guess presented as metadata; possible later as an explicit opt-in.
