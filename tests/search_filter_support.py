"""A generated Markdown and text library for filtered search (UC-02).

Many decoy sources match the query strongly; the two sources by Arnold match
it weakly, so with a small ``top_k`` an unfiltered first stage never reaches
them. Two sources have no author and two have no year.
"""

from pathlib import Path

from docgrep.settings import Settings

QUERY = "comet orbit"
DECOYS = 12
TOP_K = 4
ARNOLD = ("books/arnold.md", "books/late.md")


def _markdown(path: Path, body: str, **front: str | int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "---\n" + "".join(f"{key}: {value}\n" for key, value in front.items()) + "---\n"
        if front
        else ""
    )
    path.write_text(header + body + "\n", encoding="utf-8")


def write_filter_library(corpus: Path) -> None:
    for number in range(DECOYS):
        _markdown(
            corpus / "decoys" / f"d{number:02}.md",
            f"Comet orbit, comet orbit, comet orbit {number}.",
            author="Decoy Writer",
            year=2001 + number,
        )
    _markdown(
        corpus / ARNOLD[0],
        "Mathematical methods of classical mechanics mention a comet once, "
        "between rigid bodies, oscillations and the geometry of tides.",
        title="Mathematical Methods",
        author="V. I. Arnold",
        year=1974,
    )
    _markdown(
        corpus / ARNOLD[1],
        "Late lectures on ordinary differential equations and one comet.",
        author="Vladimir ARNOLD",
        year=1990,
    )
    _markdown(corpus / "notes" / "plain.md", "A plain comet orbit note.")
    (corpus / "papers").mkdir(parents=True, exist_ok=True)
    (corpus / "papers" / "kepler.txt").write_text(
        "Kepler on the orbit of Mars and of a comet.\n", encoding="utf-8"
    )


def filter_settings(root: Path, **extra: object) -> Settings:
    return Settings(
        corpus_root=root / "corpus",
        data_dir=root / "data",
        backend="fake",
        chunk_size=64,
        chunk_overlap=8,
        top_k=TOP_K,
        top_n=TOP_K,
        **extra,  # type: ignore[arg-type]  # qdrant_url and qdrant_alias by name
    )
