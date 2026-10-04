"""Generate small EPUB files with zipfile, so no binary fixture is committed."""

import io
import zipfile
from collections.abc import Callable, Sequence
from pathlib import Path

CONTAINER = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<container version="1.0" '
    'xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
    '<rootfiles><rootfile full-path="{path}" '
    'media-type="application/oebps-package+xml"/></rootfiles></container>'
)
NAV_TEXT = "Navigation contents list"


def xhtml(body: str, *, encoding: str = "utf-8", declare: bool = True) -> bytes:
    """A content document with a head that must never reach the text."""
    declaration = f'<?xml version="1.0" encoding="{encoding}"?>\n' if declare else ""
    return (
        f"{declaration}<!DOCTYPE html>\n"
        '<html xmlns="http://www.w3.org/1999/xhtml">'
        "<head><title>Head title</title>"
        "<style>p { color: red; }</style>"
        "<script>var hidden = 1;</script></head>"
        f"<body>{body}</body></html>"
    ).encode(encoding)


def epub_bytes(
    chapters: Sequence[tuple[str, bytes]],
    *,
    version: int = 3,
    title: str | None = "A Book",
    creators: Sequence[str] = ("Ada Lovelace",),
    date: str | None = "1843-07-01",
    date_event: str | None = None,
    spine: Sequence[str] | None = None,
    nav: bool = True,
    encryption: str | None = None,
    package_dir: str = "OEBPS",
    extra: Sequence[tuple[str, bytes]] = (),
    edit: Callable[[str, bytes], bytes | None] | None = None,
) -> bytes:
    """An EPUB 2 or 3 archive.

    ``chapters`` are (file name, content) pairs in manifest order; ``spine``
    gives the reading order by file name and defaults to manifest order. With
    ``nav`` a navigation document (EPUB 3) or an NCX listed in the spine
    (EPUB 2) is added, which a parser must skip. ``edit`` may change any
    member before it is written, or return None to leave it out.
    """
    opf_ns = "http://www.idpf.org/2007/opf"
    meta: list[str] = []
    if title is not None:
        meta.append(f"<dc:title>{title}</dc:title>")
    meta.extend(f"<dc:creator>{creator}</dc:creator>" for creator in creators)
    if date is not None:
        event = f' opf:event="{date_event}"' if date_event else ""
        meta.append(f"<dc:date{event}>{date}</dc:date>")
    items = [
        f'<item id="c{index}" href="{name}" media-type="application/xhtml+xml"/>'
        for index, (name, _) in enumerate(chapters)
    ]
    ids = {name: f"c{index}" for index, (name, _) in enumerate(chapters)}
    order = [ids[name] for name in (spine or [name for name, _ in chapters])]
    members: list[tuple[str, bytes]] = []
    toc = ""
    if nav and version == 3:
        items.append(
            '<item id="nav" href="nav.xhtml" properties="nav" '
            'media-type="application/xhtml+xml"/>'
        )
        order.insert(0, "nav")
        members.append(("nav.xhtml", xhtml(f"<nav><ol><li>{NAV_TEXT}</li></ol></nav>")))
    if nav and version == 2:
        items.append(
            '<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>'
        )
        toc = ' toc="ncx"'
        order.insert(0, "ncx")
        members.append(
            (
                "toc.ncx",
                (
                    '<?xml version="1.0"?><ncx><navMap><navPoint><navLabel><text>'
                    f"{NAV_TEXT}</text></navLabel></navPoint></navMap></ncx>"
                ).encode(),
            )
        )
    package = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<package xmlns="{opf_ns}" version="{version}.0" unique-identifier="id">'
        '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/" '
        f'xmlns:opf="{opf_ns}">{"".join(meta)}</metadata>'
        f"<manifest>{''.join(items)}</manifest>"
        f"<spine{toc}>"
        + "".join(f'<itemref idref="{item}"/>' for item in order)
        + "</spine></package>"
    )
    prefix = f"{package_dir}/" if package_dir else ""
    entries: list[tuple[str, bytes]] = [
        ("mimetype", b"application/epub+zip"),
        (
            "META-INF/container.xml",
            CONTAINER.format(path=f"{prefix}content.opf").encode(),
        ),
    ]
    if encryption is not None:
        entries.append(("META-INF/encryption.xml", encryption.encode()))
    entries.append((f"{prefix}content.opf", package.encode()))
    entries.extend((f"{prefix}{name}", content) for name, content in members)
    entries.extend((f"{prefix}{name}", content) for name, content in chapters)
    entries.extend(extra)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in entries:
            edited = content if edit is None else edit(name, content)
            if edited is not None:
                archive.writestr(name, edited)
    return buffer.getvalue()


def write_epub(path: Path, data: bytes) -> Path:
    """Write EPUB bytes to ``path``, creating its directory, and return it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def encryption_xml(*algorithms: str) -> str:
    """META-INF/encryption.xml with one encrypted entry per algorithm."""
    entries = "".join(
        '<enc:EncryptedData xmlns:enc="http://www.w3.org/2001/04/xmlenc#">'
        f'<enc:EncryptionMethod Algorithm="{algorithm}"/>'
        f'<enc:CipherData><enc:CipherReference URI="OEBPS/file{index}"/>'
        "</enc:CipherData></enc:EncryptedData>"
        for index, algorithm in enumerate(algorithms)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<encryption xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
        f"{entries}</encryption>"
    )


FONT_OBFUSCATION = "http://www.idpf.org/2008/embedding"
AES = "http://www.w3.org/2001/04/xmlenc#aes128-cbc"
