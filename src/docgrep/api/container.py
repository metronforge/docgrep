"""Composition root for local indexing and retrieval."""

import logging
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol

from docgrep import __version__
from docgrep.adapters.chunking import TokenChunker, WhitespaceTokenizer
from docgrep.adapters.corpus import (
    describe_source,
    enumerate_corpus,
    require_available,
    root_readable,
    scan_corpus,
)
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
from docgrep.adapters.models.sparse import SparseEncoderSet, TermFrequencyEncoder
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
from docgrep.api.roots import corpus_roots
from docgrep.domain.context import ContextResult, OutlineEntry
from docgrep.domain.errors import IndexIncomplete, ValidationError
from docgrep.domain.models import (
    FileType,
    ManifestSourceRecord,
    Passage,
    ScannedFile,
    SearchResult,
    Source,
)
from docgrep.domain.protocols import (
    CommittedSourceReader,
    DocumentParser,
    EmbeddingProvider,
    ManifestReader,
    OcrOutputOwners,
    PassageContextReader,
    Reranker,
    SourceRecordReader,
    StatusStore,
    TokenSpanTokenizer,
    parser_version,
)
from docgrep.domain.roots import CorpusRoots
from docgrep.domain.run_state import latest_root_prefixes
from docgrep.domain.search_effort import DenseSearchEffort, default_rerank_pool
from docgrep.domain.sources import BrowseRequest, BrowseResult, SourceFilter
from docgrep.settings import Settings
from docgrep.usecases.browse_sources import BrowseSources
from docgrep.usecases.build_index import BuildIndex, BuildSummary
from docgrep.usecases.cancel_index import CancelIndex
from docgrep.usecases.discover_scans import DiscoverScans
from docgrep.usecases.dry_run import DryRun
from docgrep.usecases.evaluate_retrieval import EvaluateRetrieval
from docgrep.usecases.expand_context import ExpandContext
from docgrep.usecases.find_passages import FindPassages
from docgrep.usecases.inspect_status import InspectStatus
from docgrep.usecases.process_scans import ProcessScans
from docgrep.usecases.reencode_sparse import ReencodeSparse
from docgrep.usecases.update_index import UpdateIndex, UpdatePlan, UpdateSummary

_LOG = logging.getLogger(__name__)


class SearchOperation(Protocol):
    """Run a search with an optional result limit and duplicate handling."""

    def run(
        self,
        query: str,
        *,
        top_n: int | None = None,
        collapse_duplicates: bool | None = None,
        source_filter: SourceFilter | None = None,
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
        override_guard: bool = False,
    ) -> UpdateSummary: ...

    def request_cancel(self) -> None: ...

    def forget_plan(self, name: str) -> tuple[ManifestSourceRecord, ...]: ...

    def passages_of(self, records: Sequence[ManifestSourceRecord]) -> int: ...

    def forget_root(
        self, name: str, *, on_plan: Callable[[UpdatePlan], None] | None = None
    ) -> UpdateSummary: ...


class BrowseOperation(Protocol):
    """List one page of indexed sources."""

    def run(self, request: BrowseRequest) -> BrowseResult: ...


class ContextOperation(Protocol):
    """Return passages with the text around them."""

    def run(
        self, passage_ids: Sequence[str], *, window_tokens: int | None = None
    ) -> ContextResult: ...


@dataclass(frozen=True, slots=True)
class Services:
    """Operations exposed to HTTP and CLI transports."""

    search: SearchOperation
    build: BuildOperation
    update: UpdateOperation | None = None
    browse: BrowseOperation | None = None
    context: ContextOperation | None = None


class _CorpusEnumerator:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._roots = corpus_roots(settings)

    def require_available(self) -> None:
        require_available(self._roots)

    def enumerate(self) -> tuple[Source, ...]:
        return enumerate_corpus(
            self._roots,
            self._settings.include_globs,
            self._settings.exclude_globs,
            self._settings.data_dir,
        )

    def scan(self) -> tuple[ScannedFile, ...]:
        return scan_corpus(
            self._roots,
            self._settings.include_globs,
            self._settings.exclude_globs,
            self._settings.data_dir,
        )

    def describe(self, path: str) -> Source:
        return describe_source(self._roots, path)


class _RootCatalog:
    """The roots of this process with the prefixes the index records, read
    from the manifest on every call; nothing is created or migrated."""

    def __init__(self, settings: Settings, roots: CorpusRoots | None = None) -> None:
        self._data_dir = settings.data_dir
        self._roots = corpus_roots(settings) if roots is None else roots

    def roots(self) -> CorpusRoots:
        if not (self._data_dir / "manifest.sqlite3").is_file():
            return self._roots
        try:
            manifest = SQLiteManifest(self._data_dir, read_only=True)
            return self._roots.with_recorded(latest_root_prefixes(manifest))
        except Exception:
            _LOG.debug("root_prefixes_unreadable", exc_info=True)
            return self._roots


def _log_root_name(settings: Settings, roots: CorpusRoots) -> str:
    """The folder name logged as ``corpus_root``: the legacy root, or else
    the first root."""
    root = roots.legacy or next(iter(roots), None)
    return (root.path if root else settings.corpus_root.resolve()).name


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


def _parsers(
    settings: Settings, roots: CorpusRoots | None = None
) -> Mapping[FileType, DocumentParser]:
    found = corpus_roots(settings) if roots is None else roots
    return {
        FileType.TXT: TextParser(found),
        FileType.MARKDOWN: MarkdownParser(found),
        FileType.PDF: PdfParser(found),
        FileType.EPUB: EpubParser(found),
        FileType.DOCX: DocxParser(found),
        FileType.DJVU: DjvuParser(found),
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
    roots = corpus_roots(settings)
    return DryRun(
        scanner=_CorpusEnumerator(settings),
        parsers=_parsers(settings, roots),
        chunker=TokenChunker(tokenizer, settings.chunk_size, settings.chunk_overlap),
        roots=_RootCatalog(settings, roots),
    )


def create_discover_scans(settings: Settings) -> DiscoverScans:
    """Wire the read-only scan discovery (UC-5).

    It needs only the corpus scan and a probe for PDF and DjVu pages: no
    index store, manifest, lock or model, and it creates nothing on disk.
    The OCR status of the files comes from ``open_ocr_registry_view``,
    passed to each run.
    """
    configure_logging(settings.log_level)
    roots = corpus_roots(settings)
    return DiscoverScans(
        scanner=_CorpusEnumerator(settings),
        probe=FileScanProbe(roots),
        density_lower=settings.scan_density_lower,
        density_upper=settings.scan_density_upper,
        roots=_RootCatalog(settings, roots),
    )


def open_ocr_registry_view(settings: Settings) -> SQLiteOcrRegistry:
    """A read-only view of the OCR registry for scan discovery.

    It creates and migrates nothing; a missing manifest reads as nothing
    processed.
    """
    return SQLiteOcrRegistry(
        settings.data_dir, _RootCatalog(settings).roots(), read_only=True
    )


def create_process_scans(settings: Settings) -> ProcessScans:
    """Wire OCR processing (UC-6).

    It needs the corpus, the scan probe, the manifest database (runs and the
    OCR registry), the index lock and the system OCR tools; no index store
    and no model. Opening it creates or migrates the manifest database.
    """
    configure_logging(settings.log_level)
    corpus = _CorpusEnumerator(settings)
    roots = corpus_roots(settings)
    manifest = SQLiteManifest(settings.data_dir)
    known = roots.with_recorded(latest_root_prefixes(manifest))
    return ProcessScans(
        lock=FileIndexLock(settings.data_dir / "index.lock"),
        manifest=manifest,
        scanner=corpus,
        discovery=create_discover_scans(settings),
        registry=SQLiteOcrRegistry(settings.data_dir, known),
        engine=OcrmypdfEngine(known),
        converter=DdjvuConverter(known),
        files=LocalOcrFiles(known),
        availability=corpus,
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
        roots=_RootCatalog(settings),
        root_readable=lambda root: root_readable(root.path),
    )


def index_missing(settings: Settings) -> bool:
    """Whether no index was ever built for this data directory.

    Checked before the composition root is created, because opening it would
    create the data directory and an empty store.
    """
    if not (settings.data_dir / "manifest.sqlite3").is_file():
        return True
    return not settings.qdrant_url and not (settings.data_dir / "qdrant").is_dir()


class _NoSources:
    """The committed sources of a data directory without a manifest."""

    def committed_sources(self) -> tuple[ManifestSourceRecord, ...]:
        return ()

    def get_source(self, path: str) -> ManifestSourceRecord | None:
        return None


class _NoPassages:
    """The passage reads of a data directory without an index."""

    def fetch_passages(
        self, passage_ids: Sequence[str]
    ) -> tuple[tuple[Passage, Source] | None, ...]:
        return tuple(None for _ in passage_ids)

    def source_outline(self, source_id: str) -> tuple[OutlineEntry, ...]:
        return ()

    def passage_texts(self, passage_ids: Collection[str]) -> Mapping[str, Passage]:
        return {}


class _ReadStore:
    """Open the store on the first read, and only then.

    A local store that does not exist is not created: it has no published
    index, so a count raises ``IndexIncomplete`` and nothing is found. A
    store that cannot be opened fails every read the same way.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._store: QdrantPassageStore | None = None
        self._failure: Exception | None = None

    def _opened(self) -> QdrantPassageStore | None:
        if self._failure is not None:
            raise self._failure
        if self._store is None:
            settings = self._settings
            local = settings.data_dir / "qdrant"
            if not settings.qdrant_url and not local.is_dir():
                return None
            try:
                self._store = (
                    QdrantPassageStore(
                        url=settings.qdrant_url, alias=settings.qdrant_alias
                    )
                    if settings.qdrant_url
                    else QdrantPassageStore(local, alias=settings.qdrant_alias)
                )
            except Exception as error:
                self._failure = error
                raise
        return self._store

    def count_source_passages(self, source_id: str) -> int:
        store = self._opened()
        if store is None:
            raise IndexIncomplete("No index is published")
        return store.count_source_passages(source_id)

    def fetch_passages(
        self, passage_ids: Sequence[str]
    ) -> tuple[tuple[Passage, Source] | None, ...]:
        store = self._opened()
        if store is None:
            return tuple(None for _ in passage_ids)
        return store.fetch_passages(passage_ids)

    def source_outline(self, source_id: str) -> tuple[OutlineEntry, ...]:
        store = self._opened()
        return () if store is None else store.source_outline(source_id)

    def passage_texts(self, passage_ids: Collection[str]) -> Mapping[str, Passage]:
        store = self._opened()
        return {} if store is None else store.passage_texts(passage_ids)

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None


@contextmanager
def open_browse(settings: Settings) -> Iterator[BrowseSources]:
    """Wire source browsing (UC-08) for a process of its own, such as the CLI.

    It loads no model, takes no lock and creates nothing: the manifest and
    the OCR registry are opened read-only, and the store only when a count
    is needed. The store is closed on exit.
    """
    configure_logging(settings.log_level)
    manifest: CommittedSourceReader = (
        SQLiteManifest(settings.data_dir, read_only=True)
        if (settings.data_dir / "manifest.sqlite3").is_file()
        else _NoSources()
    )
    store = _ReadStore(settings)
    try:
        yield BrowseSources(
            manifest=manifest,
            store=store,
            ocr=open_ocr_registry_view(settings),
            roots=_RootCatalog(settings),
        )
    finally:
        store.close()


def _expand_context(
    settings: Settings,
    *,
    store: PassageContextReader,
    manifest: SourceRecordReader,
    ocr: OcrOutputOwners | None,
) -> ExpandContext:
    return ExpandContext(
        store=store,
        manifest=manifest,
        ocr=ocr,
        max_passages=settings.context_max_passages,
        default_window_tokens=settings.context_window_tokens,
        max_window_tokens=settings.context_max_window_tokens,
    )


def unindexed_context(settings: Settings) -> ExpandContext:
    """The context operation of a data directory where no index was built:
    it validates requests and finds nothing, without creating anything."""
    configure_logging(settings.log_level)
    return _expand_context(
        settings, store=_NoPassages(), manifest=_NoSources(), ocr=None
    )


@contextmanager
def open_context(settings: Settings) -> Iterator[ExpandContext]:
    """Wire passage context (UC-12) for a process of its own, such as the CLI.

    It loads no model, takes no lock and creates nothing: the manifest and
    the OCR registry are opened read-only, and the store only on the first
    read, after the request is validated. The store is closed on exit.
    """
    configure_logging(settings.log_level)
    if index_missing(settings):
        yield unindexed_context(settings)
        return
    store = _ReadStore(settings)
    try:
        yield _expand_context(
            settings,
            store=store,
            manifest=SQLiteManifest(settings.data_dir, read_only=True),
            ocr=open_ocr_registry_view(settings),
        )
    finally:
        store.close()


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


def _models(
    settings: Settings,
) -> tuple[EmbeddingProvider, Reranker, TokenSpanTokenizer]:
    """The embedder, the reranker and the chunking tokenizer of the backend."""
    if settings.backend == "fake":
        return DeterministicEmbedder(), LexicalReranker(), WhitespaceTokenizer()
    embedder = BgeEmbedder(
        settings.embedding_model, settings.device, dtype=settings.embedding_dtype
    )
    reranker = BgeReranker(
        settings.reranker_model, settings.device, dtype=settings.reranker_dtype
    )
    return embedder, reranker, _LazyModelTokenizer(embedder)


@contextmanager
def open_evaluation(
    settings: Settings,
    *,
    top_k: int | None = None,
    rerank_pool: int | None = None,
    dense_hnsw_ef: DenseSearchEffort | None = None,
) -> Iterator[EvaluateRetrieval]:
    """Wire the retrieval evaluation (ADR-0024) for a process of its own.

    Read-only: no lock is taken, the manifest and the OCR registry are
    opened without migrating them, and the store is only queried. The
    models are those of search, loaded once; ``top_k``, ``rerank_pool`` and
    ``dense_hnsw_ef`` replace the configured first-stage size, rerank pool
    and dense search effort (an unset pool follows ``top_k`` as the
    configured default does, ADR-0026). Raises
    ``IndexIncomplete`` when no index was ever built, before anything is
    opened, and ``ValidationError`` for a pool above twice the first-stage
    size. The store is closed on exit.
    """
    configure_logging(settings.log_level)
    if index_missing(settings):
        raise IndexIncomplete("No index has been built yet")
    first_stage = settings.top_k if top_k is None else top_k
    pool = settings.rerank_pool if rerank_pool is None else rerank_pool
    if pool is None:
        pool = default_rerank_pool(first_stage)
    effort = settings.dense_hnsw_ef if dense_hnsw_ef is None else dense_hnsw_ef
    if pool > 2 * first_stage:
        raise ValidationError(
            f"rerank_pool must not exceed 2 * top_k ({2 * first_stage})",
            details={"rerank_pool": pool, "top_k": first_stage},
        )
    manifest = SQLiteManifest(settings.data_dir, read_only=True)
    store = (
        QdrantPassageStore(
            url=settings.qdrant_url,
            alias=settings.qdrant_alias,
            dense_hnsw_ef=effort,
        )
        if settings.qdrant_url
        else QdrantPassageStore(
            settings.data_dir / "qdrant",
            alias=settings.qdrant_alias,
            dense_hnsw_ef=effort,
        )
    )
    try:
        embedder, reranker, _ = _models(settings)
        yield EvaluateRetrieval(
            search=FindPassages(
                embedder=embedder,
                sparse_encoder=TermFrequencyEncoder(),
                sparse_encoders=SparseEncoderSet(),
                retriever=store,
                reranker=reranker,
                top_k=first_stage,
                rerank_pool=pool,
                dense_hnsw_ef=effort,
                default_top_n=settings.top_n,
                top_n_max=settings.top_n_max,
                max_query_len=settings.max_query_len,
                collapse_duplicates=settings.collapse_duplicates,
                manifest=manifest,
                ocr=open_ocr_registry_view(settings),
                roots=_RootCatalog(settings),
            ),
            probe=store,
        )
    finally:
        store.close()


@contextmanager
def open_reencode(settings: Settings) -> Iterator[ReencodeSparse]:
    """Wire the sparse re-encode (ADR-0025) for a process of its own.

    It loads no model: only the store, the manifest, the index lock and the
    sparse tokenizer. Raises ``IndexIncomplete`` when no index was ever
    built, before anything is opened or created. The store is closed on
    exit.
    """
    configure_logging(settings.log_level)
    if index_missing(settings):
        raise IndexIncomplete("No index has been built yet")
    store = (
        QdrantPassageStore(url=settings.qdrant_url, alias=settings.qdrant_alias)
        if settings.qdrant_url
        else QdrantPassageStore(
            settings.data_dir / "qdrant", alias=settings.qdrant_alias
        )
    )
    try:
        yield ReencodeSparse(
            lock=FileIndexLock(settings.data_dir / "index.lock"),
            index=store,
            manifest=SQLiteManifest(settings.data_dir),
            encoders=SparseEncoderSet(),
        )
    finally:
        store.close()


def create_container(
    settings: Settings, *, refuse_resumable_by_default: bool = False
) -> Services:
    """Wire selected models to local Qdrant, SQLite, parsers, and use cases.

    With ``refuse_resumable_by_default`` a build that is not told otherwise
    refuses to discard a resumable run.
    """
    configure_logging(settings.log_level)
    embedder, reranker, tokenizer = _models(settings)
    sparse = TermFrequencyEncoder()
    encoders = SparseEncoderSet()
    store = (
        QdrantPassageStore(
            url=settings.qdrant_url,
            alias=settings.qdrant_alias,
            dense_hnsw_ef=settings.dense_hnsw_ef,
        )
        if settings.qdrant_url
        else QdrantPassageStore(
            settings.data_dir / "qdrant",
            alias=settings.qdrant_alias,
            dense_hnsw_ef=settings.dense_hnsw_ef,
        )
    )
    manifest = SQLiteManifest(settings.data_dir)
    roots = corpus_roots(settings)
    catalog = _RootCatalog(settings, roots)
    # The manifest above has created or migrated the OCR tables.
    delegations = SQLiteOcrRegistry(
        settings.data_dir,
        roots.with_recorded(latest_root_prefixes(manifest)),
        read_only=True,
    )
    parsers = _parsers(settings, roots)
    config = _vector_config(settings)
    lock = FileIndexLock(settings.data_dir / "index.lock")
    chunker = TokenChunker(tokenizer, settings.chunk_size, settings.chunk_overlap)
    corpus = _CorpusEnumerator(settings)
    log_context: dict[str, str | int | float] = {
        "corpus_root": _log_root_name(settings, roots),
        "backend": settings.backend,
    }
    return Services(
        search=FindPassages(
            embedder=embedder,
            sparse_encoder=sparse,
            sparse_encoders=encoders,
            retriever=store,
            reranker=reranker,
            top_k=settings.top_k,
            rerank_pool=settings.effective_rerank_pool,
            dense_hnsw_ef=settings.dense_hnsw_ef,
            default_top_n=settings.top_n,
            top_n_max=settings.top_n_max,
            max_query_len=settings.max_query_len,
            collapse_duplicates=settings.collapse_duplicates,
            manifest=manifest,
            ocr=delegations,
            roots=catalog,
        ),
        build=BuildIndex(
            lock=lock,
            enumerator=corpus,
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
            roots=catalog,
            availability=corpus,
            sparse_encoders=encoders,
            sparse_store=store,
        ),
        update=UpdateIndex(
            lock=lock,
            corpus=corpus,
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
            roots=catalog,
            availability=corpus,
            sparse_encoders=encoders,
            guard=settings.removal_guard,
        ),
        # The search store is shared, so a local store is never opened twice.
        browse=BrowseSources(
            manifest=manifest, store=store, ocr=delegations, roots=catalog
        ),
        context=_expand_context(
            settings, store=store, manifest=manifest, ocr=delegations
        ),
    )
