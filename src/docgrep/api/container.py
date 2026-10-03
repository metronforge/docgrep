"""Composition root for local indexing and retrieval."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from docgrep import __version__
from docgrep.adapters.chunking import TokenChunker, WhitespaceTokenizer
from docgrep.adapters.corpus import enumerate_corpus, scan_corpus
from docgrep.adapters.filelock import FileIndexLock
from docgrep.adapters.manifest import SQLiteManifest
from docgrep.adapters.models.deterministic import (
    DeterministicEmbedder,
    LexicalReranker,
)
from docgrep.adapters.models.real import (
    BgeEmbedder,
    BgeReranker,
    load_tokenizer,
)
from docgrep.adapters.models.sparse import BM25SparseEncoder
from docgrep.adapters.parsers.markdown import MarkdownParser
from docgrep.adapters.parsers.pdf import PdfParser
from docgrep.adapters.parsers.text import TextParser
from docgrep.adapters.qdrant import QdrantPassageStore
from docgrep.api.log_config import configure_logging
from docgrep.domain.models import FileType, ScannedFile, SearchResult, Source
from docgrep.domain.protocols import (
    DocumentParser,
    EmbeddingProvider,
    Reranker,
    TokenSpanTokenizer,
)
from docgrep.settings import Settings
from docgrep.usecases.build_index import BuildIndex, BuildSummary
from docgrep.usecases.cancel_index import CancelIndex
from docgrep.usecases.dry_run import DryRun
from docgrep.usecases.find_passages import FindPassages


class SearchOperation(Protocol):
    """Run a search with an optional result limit."""

    def run(self, query: str, *, top_n: int | None = None) -> SearchResult: ...


class BuildOperation(Protocol):
    """Run one full index build."""

    def run(self) -> BuildSummary: ...

    def request_cancel(self) -> None: ...


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

    def scan(self) -> tuple[ScannedFile, ...]:
        return scan_corpus(
            self._settings.corpus_root,
            self._settings.include_globs,
            self._settings.exclude_globs,
            self._settings.data_dir,
        )


class _LazyTokenizerOnly:
    """Tokenize with the embedding model's tokenizer without loading its weights."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._tokenizer: TokenSpanTokenizer | None = None

    def token_spans(self, text: str) -> tuple[tuple[int, int], ...]:
        if self._tokenizer is None:
            self._tokenizer = load_tokenizer(self._model_name)
        return self._tokenizer.token_spans(text)


class _LazyModelTokenizer:
    def __init__(self, embedder: BgeEmbedder) -> None:
        self._embedder = embedder

    def token_spans(self, text: str) -> tuple[tuple[int, int], ...]:
        return self._embedder.tokenizer.token_spans(text)


def _parsers(settings: Settings) -> Mapping[FileType, DocumentParser]:
    return {
        FileType.TXT: TextParser(settings.corpus_root),
        FileType.MARKDOWN: MarkdownParser(settings.corpus_root),
        FileType.PDF: PdfParser(settings.corpus_root),
    }


def create_dry_run(settings: Settings) -> DryRun:
    """Wire the read-only corpus preview.

    It needs no index store, manifest, lock or embedding model, so creating it
    touches nothing on disk. Sampling chunks with the embedding model's
    tokenizer, loaded without its weights, or with the whitespace tokenizer of
    the fake backend.
    """
    configure_logging(settings.log_level)
    tokenizer: TokenSpanTokenizer = (
        WhitespaceTokenizer()
        if settings.backend == "fake"
        else _LazyTokenizerOnly(settings.embedding_model)
    )
    return DryRun(
        scanner=_CorpusEnumerator(settings),
        parsers=_parsers(settings),
        chunker=TokenChunker(tokenizer, settings.chunk_size, settings.chunk_overlap),
    )


def create_cancel(settings: Settings) -> CancelIndex:
    """Wire the cancel request.

    It needs only the lock and the manifest, so it works while another process
    owns the local Qdrant store. A data directory without a manifest has never
    seen a build, and nothing is created for it.
    """
    configure_logging(settings.log_level)
    manifest = (
        SQLiteManifest(settings.data_dir)
        if (settings.data_dir / "manifest.sqlite3").exists()
        else None
    )
    return CancelIndex(
        lock=FileIndexLock(settings.data_dir / "index.lock"), manifest=manifest
    )


def create_container(settings: Settings) -> Services:
    """Wire selected models to local Qdrant, SQLite, parsers, and use cases."""
    configure_logging(settings.log_level)
    embedder: EmbeddingProvider
    reranker: Reranker
    tokenizer: TokenSpanTokenizer
    if settings.backend == "fake":
        embedder = DeterministicEmbedder()
        reranker = LexicalReranker()
        tokenizer = WhitespaceTokenizer()
    else:
        real_embedder = BgeEmbedder(
            settings.embedding_model, settings.device, dtype=settings.embedding_dtype
        )
        embedder = real_embedder
        reranker = BgeReranker(
            settings.reranker_model, settings.device, dtype=settings.reranker_dtype
        )
        tokenizer = _LazyModelTokenizer(real_embedder)

    sparse = BM25SparseEncoder()
    store = (
        QdrantPassageStore(url=settings.qdrant_url, alias=settings.qdrant_alias)
        if settings.qdrant_url
        else QdrantPassageStore(
            settings.data_dir / "qdrant", alias=settings.qdrant_alias
        )
    )
    manifest = SQLiteManifest(settings.data_dir)
    parsers = _parsers(settings)
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
            lock=FileIndexLock(settings.data_dir / "index.lock"),
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
            upsert_batch_size=settings.upsert_batch_size,
            log_context={
                "corpus_root": settings.corpus_root.resolve().name,
                "backend": settings.backend,
            },
        ),
    )
