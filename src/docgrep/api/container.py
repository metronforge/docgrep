"""Composition root for local indexing and retrieval."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from docgrep import __version__
from docgrep.adapters.chunking import TokenChunker, WhitespaceTokenizer
from docgrep.adapters.corpus import enumerate_corpus
from docgrep.adapters.manifest import SQLiteManifest
from docgrep.adapters.models.deterministic import (
    DeterministicEmbedder,
    LexicalReranker,
)
from docgrep.adapters.models.real import BgeEmbedder, BgeReranker
from docgrep.adapters.models.sparse import BM25SparseEncoder
from docgrep.adapters.parsers.markdown import MarkdownParser
from docgrep.adapters.parsers.pdf import PdfParser
from docgrep.adapters.parsers.text import TextParser
from docgrep.adapters.qdrant import QdrantPassageStore
from docgrep.domain.models import FileType, SearchResult, Source
from docgrep.domain.protocols import (
    DocumentParser,
    EmbeddingProvider,
    Reranker,
    TokenSpanTokenizer,
)
from docgrep.settings import Settings
from docgrep.usecases.build_index import BuildIndex, BuildSummary
from docgrep.usecases.find_passages import FindPassages


class SearchOperation(Protocol):
    """Run a search with an optional result limit."""

    def run(self, query: str, *, top_n: int | None = None) -> SearchResult: ...


class BuildOperation(Protocol):
    """Run one full index build."""

    def run(self) -> BuildSummary: ...


@dataclass(frozen=True, slots=True)
class Services:
    """Operations exposed to HTTP and CLI transports."""

    search: SearchOperation
    build: BuildOperation


class _CorpusEnumerator:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def enumerate(self) -> tuple[Source, ...]:
        return enumerate_corpus(
            self._settings.corpus_root,
            self._settings.include_globs,
            self._settings.exclude_globs,
            self._settings.data_dir,
        )


class _LazyModelTokenizer:
    def __init__(self, embedder: BgeEmbedder) -> None:
        self._embedder = embedder

    def token_spans(self, text: str) -> tuple[tuple[int, int], ...]:
        return self._embedder.tokenizer.token_spans(text)


def create_container(settings: Settings) -> Services:
    """Wire selected models to local Qdrant, SQLite, parsers, and use cases."""
    embedder: EmbeddingProvider
    reranker: Reranker
    tokenizer: TokenSpanTokenizer
    if settings.backend == "fake":
        embedder = DeterministicEmbedder()
        reranker = LexicalReranker()
        tokenizer = WhitespaceTokenizer()
    else:
        real_embedder = BgeEmbedder(settings.embedding_model, settings.device)
        embedder = real_embedder
        reranker = BgeReranker(settings.reranker_model, settings.device)
        tokenizer = _LazyModelTokenizer(real_embedder)

    sparse = BM25SparseEncoder()
    store = QdrantPassageStore(settings.data_dir / "qdrant")
    manifest = SQLiteManifest(settings.data_dir)
    parsers: Mapping[FileType, DocumentParser] = {
        FileType.TXT: TextParser(settings.corpus_root),
        FileType.MARKDOWN: MarkdownParser(settings.corpus_root),
        FileType.PDF: PdfParser(settings.corpus_root),
    }
    config: Mapping[str, str | int | float] = {
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    }
    return Services(
        search=FindPassages(
            embedder=embedder,
            sparse_encoder=sparse,
            retriever=store,
            reranker=reranker,
            top_k=settings.top_k,
            default_top_n=settings.top_n,
            top_n_max=settings.top_n_max,
            max_query_len=settings.max_query_len,
        ),
        build=BuildIndex(
            data_dir=settings.data_dir,
            enumerator=_CorpusEnumerator(settings),
            parsers=parsers,
            chunker=TokenChunker(
                tokenizer, settings.chunk_size, settings.chunk_overlap
            ),
            embedder=embedder,
            sparse_encoder=sparse,
            passage_writer=store,
            manifest=manifest,
            config=config,
            docgrep_version=__version__,
        ),
    )
