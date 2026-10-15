"""Composition root for local indexing and retrieval."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from docgrep import __version__
from docgrep.adapters.chunking import TokenChunker, WhitespaceTokenizer
from docgrep.adapters.corpus import describe_source, enumerate_corpus, scan_corpus
from docgrep.adapters.filelock import FileIndexLock
from docgrep.adapters.manifest import SQLiteManifest, SQLiteOcrRegistry
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
from docgrep.adapters.ocrmypdf import DdjvuConverter, LocalOcrFiles, OcrmypdfEngine
from docgrep.adapters.parsers.djvu import DjvuParser
from docgrep.adapters.parsers.docx import DocxParser
from docgrep.adapters.parsers.epub import EpubParser
from docgrep.adapters.parsers.markdown import MarkdownParser
from docgrep.adapters.parsers.pdf import PdfParser
from docgrep.adapters.parsers.text import TextParser
from docgrep.adapters.qdrant import QdrantPassageStore
from docgrep.adapters.scans import FileScanProbe
from docgrep.api.log_config import configure_logging
from docgrep.domain.models import FileType, ScannedFile, SearchResult, Source
from docgrep.domain.protocols import (
    DocumentParser,
    EmbeddingProvider,
    ManifestReader,
    Reranker,
    StatusStore,
    TokenSpanTokenizer,
    parser_version,
)
from docgrep.settings import Settings
from docgrep.usecases.build_index import BuildIndex, BuildSummary
from docgrep.usecases.cancel_index import CancelIndex
from docgrep.usecases.discover_scans import DiscoverScans
from docgrep.usecases.dry_run import DryRun
from docgrep.usecases.find_passages import FindPassages
from docgrep.usecases.inspect_status import InspectStatus
from docgrep.usecases.process_scans import ProcessScans
from docgrep.usecases.update_index import UpdateIndex, UpdatePlan, UpdateSummary


class SearchOperation(Protocol):
    """Run a search with an optional result limit and duplicate handling."""

    def run(
        self,
        query: str,
        *,
        top_n: int | None = None,
        collapse_duplicates: bool | None = None,
    ) -> SearchResult: ...


class BuildOperation(Protocol):
    """Run one full index build."""

    def run(self, *, refuse_resumable: bool | None = None) -> BuildSummary: ...

    def resume(self, *, retry_failed: bool = True) -> BuildSummary: ...

    def request_cancel(self) -> None: ...


class UpdateOperation(Protocol):
    """Run one incremental update of the published index."""

    def run(
        self,
        *,
        retry_failed: bool = True,
        on_plan: Callable[[UpdatePlan], None] | None = None,
    ) -> UpdateSummary: ...

    def request_cancel(self) -> None: ...


@dataclass(frozen=True, slots=True)
class Services:
    """Operations exposed to HTTP and CLI transports."""

    search: SearchOperation
    build: BuildOperation
    update: UpdateOperation | None = None


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

    def describe(self, path: str) -> Source:
        return describe_source(self._settings.corpus_root, path)


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
        FileType.EPUB: EpubParser(settings.corpus_root),
        FileType.DOCX: DocxParser(settings.corpus_root),
        FileType.DJVU: DjvuParser(settings.corpus_root),
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


def create_discover_scans(settings: Settings) -> DiscoverScans:
    """Wire the read-only scan discovery (UC-5).

    It needs only the corpus scan and a probe for PDF and DjVu pages: no
    index store, manifest, lock or model, and it creates nothing on disk.
    The OCR status of the files comes from ``open_ocr_registry_view``,
    passed to each run.
    """
    configure_logging(settings.log_level)
    return DiscoverScans(
        scanner=_CorpusEnumerator(settings),
        probe=FileScanProbe(settings.corpus_root),
        density_lower=settings.scan_density_lower,
        density_upper=settings.scan_density_upper,
    )


def open_ocr_registry_view(settings: Settings) -> SQLiteOcrRegistry:
    """A read-only view of the OCR registry for scan discovery.

    It creates and migrates nothing; a missing manifest reads as nothing
    processed.
    """
    return SQLiteOcrRegistry(settings.data_dir, settings.corpus_root, read_only=True)


def create_process_scans(settings: Settings) -> ProcessScans:
    """Wire OCR processing (UC-6).

    It needs the corpus, the scan probe, the manifest database (runs and the
    OCR registry), the index lock and the system OCR tools; no index store
    and no model. Opening it creates or migrates the manifest database.
    """
    configure_logging(settings.log_level)
    corpus = _CorpusEnumerator(settings)
    return ProcessScans(
        lock=FileIndexLock(settings.data_dir / "index.lock"),
        manifest=SQLiteManifest(settings.data_dir),
        scanner=corpus,
        discovery=create_discover_scans(settings),
        registry=SQLiteOcrRegistry(settings.data_dir, settings.corpus_root),
        engine=OcrmypdfEngine(settings.corpus_root),
        converter=DdjvuConverter(settings.corpus_root),
        files=LocalOcrFiles(settings.corpus_root),
        languages=settings.ocr_languages,
        jobs=settings.ocr_jobs,
        timeout_seconds=settings.ocr_timeout_seconds,
        cancel_mode=settings.ocr_cancel_mode,
    )


# Seconds to wait for a Qdrant server when only reporting its status.
STATUS_STORE_TIMEOUT = 2


def _vector_config(settings: Settings) -> dict[str, str | int | float]:
    """The settings, besides the model, that decide what vectors contain."""
    return {
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        # Together with the embedding model and its dimension, which the build
        # adds, these are the keys a resume compares (VECTOR_CONFIG_KEYS).
        "tokenizer": "whitespace"
        if settings.backend == "fake"
        else settings.embedding_model,
        "normalize_embeddings": 1,
    }


def create_status(settings: Settings) -> InspectStatus:
    """Wire the read-only status report.

    It loads no model, takes no lock and creates nothing: the manifest is
    opened without migrating it, a local store that does not exist is not
    created, and a server is pinged with a short timeout.
    """
    configure_logging(settings.log_level)
    manifest_file = settings.data_dir / "manifest.sqlite3"
    local_store = settings.data_dir / "qdrant"

    def open_manifest() -> ManifestReader | None:
        if not manifest_file.is_file():
            return None
        return SQLiteManifest(settings.data_dir, read_only=True)

    def open_store() -> StatusStore | None:
        if settings.qdrant_url:
            return QdrantPassageStore(
                url=settings.qdrant_url,
                alias=settings.qdrant_alias,
                timeout=STATUS_STORE_TIMEOUT,
            )
        if not local_store.is_dir():
            return None
        return QdrantPassageStore(local_store, alias=settings.qdrant_alias)

    runtime: dict[str, str | int | float] = _vector_config(settings)
    if settings.backend == "fake":
        runtime["embedding_model"] = DeterministicEmbedder.model_name
        runtime["embedding_dimension"] = DeterministicEmbedder.dimension
    else:
        runtime["embedding_model"] = settings.embedding_model
    return InspectStatus(
        lock=FileIndexLock(settings.data_dir / "index.lock"),
        open_manifest=open_manifest,
        open_store=open_store,
        store_mode="server" if settings.qdrant_url else "local",
        runtime_config=runtime,
        # Constructing the parsers reads nothing; only their versions are used.
        parser_versions={
            file_type: parser_version(parser)
            for file_type, parser in _parsers(settings).items()
        },
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


def create_container(
    settings: Settings, *, refuse_resumable_by_default: bool = False
) -> Services:
    """Wire selected models to local Qdrant, SQLite, parsers, and use cases.

    With ``refuse_resumable_by_default`` a build that is not told otherwise
    refuses to discard a resumable run.
    """
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
    # The manifest above has created or migrated the OCR tables.
    delegations = SQLiteOcrRegistry(
        settings.data_dir, settings.corpus_root, read_only=True
    )
    parsers = _parsers(settings)
    config = _vector_config(settings)
    lock = FileIndexLock(settings.data_dir / "index.lock")
    chunker = TokenChunker(tokenizer, settings.chunk_size, settings.chunk_overlap)
    log_context: dict[str, str | int | float] = {
        "corpus_root": settings.corpus_root.resolve().name,
        "backend": settings.backend,
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
            collapse_duplicates=settings.collapse_duplicates,
        ),
        build=BuildIndex(
            lock=lock,
            enumerator=_CorpusEnumerator(settings),
            parsers=parsers,
            chunker=chunker,
            embedder=embedder,
            sparse_encoder=sparse,
            passage_writer=store,
            manifest=manifest,
            config=config,
            docgrep_version=__version__,
            upsert_batch_size=settings.upsert_batch_size,
            refuse_resumable_by_default=refuse_resumable_by_default,
            log_context=log_context,
            delegations=delegations,
        ),
        update=UpdateIndex(
            lock=lock,
            corpus=_CorpusEnumerator(settings),
            parsers=parsers,
            chunker=chunker,
            embedder=embedder,
            sparse_encoder=sparse,
            index=store,
            manifest=manifest,
            config=config,
            upsert_batch_size=settings.upsert_batch_size,
            log_context=log_context,
            delegations=delegations,
        ),
    )
