"""Native worker process helpers for multi-core Saguaro indexing."""

from __future__ import annotations

import json
import logging
import mmap
import os
import re
import time
from array import array
from typing import Any

from saguaro.indexing.native_runtime import get_native_runtime
from saguaro.parsing.parser import SAGUAROParser
from saguaro.query.corpus_rules import canonicalize_rel_path, classify_file_role
from saguaro.utils.entity_ids import entity_identity

logger = logging.getLogger(__name__)

_IDENT_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_MAX_EMBED_TEXT_CHARS = int(os.getenv("SAGUARO_MAX_EMBED_TEXT_CHARS", "24000"))
_EMBED_HEAD_CHARS = int(os.getenv("SAGUARO_EMBED_HEAD_CHARS", "16000"))
_EMBED_TAIL_CHARS = int(os.getenv("SAGUARO_EMBED_TAIL_CHARS", "8000"))

_worker_projection_map: mmap.mmap | None = None
_worker_projection_file: Any | None = None
_worker_projection_path: str | None = None
_worker_projection: object | None = None
_worker_shm: Any | None = None
_native_indexer = None
_parser = None
_trie_handle = None


def _initialize_worker() -> None:
    """Initialize process-local native/runtime state once."""
    global _native_indexer, _parser

    if _native_indexer is None:
        _native_indexer = get_native_runtime()
    if _parser is None:
        _parser = SAGUAROParser()


def _load_codebook_manual(path: str):
    """Load the optional native trie codebook without importing TensorFlow."""
    global _trie_handle, _native_indexer

    if _trie_handle is not None:
        return _trie_handle
    if not path or not os.path.exists(path):
        return None

    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle) or {}
        superwords = data.get("superwords", [])
        if not superwords:
            return None

        offsets = [0]
        all_tokens: list[int] = []
        ids: list[int] = []
        current_offset = 0
        for entry in superwords:
            tokens = [int(item) for item in entry.get("token_ids", [])]
            all_tokens.extend(tokens)
            current_offset += len(tokens)
            offsets.append(current_offset)
            ids.append(int(entry.get("superword_id", 0)))

        _trie_handle = _native_indexer.create_trie()
        _native_indexer.build_trie_from_table(_trie_handle, offsets, all_tokens, ids)
        return _trie_handle
    except Exception as exc:
        logger.warning("Failed to load native codebook manually: %s", exc)
        return None


SHM_NAME = "saguaro_projection_v3"


def _close_worker_shm() -> None:
    """Release any attached projection mapping for this worker."""
    global _worker_projection_file, _worker_projection_map, _worker_projection_path, _worker_shm
    mapping = _worker_projection_map
    handle = _worker_projection_file
    _worker_projection_map = None
    _worker_projection_file = None
    _worker_projection_path = None
    _worker_shm = None
    if mapping is not None:
        try:
            mapping.close()
        except Exception:
            pass
    if handle is None:
        return
    close = getattr(handle, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def _projection_buffer(
    *,
    shm_name: str,
    vocab_size: int,
    active_dim: int,
) -> object:
    global _worker_projection, _worker_projection_file, _worker_projection_map, _worker_projection_path

    if _worker_projection is not None:
        return _worker_projection
    if _worker_shm is not None and hasattr(_worker_shm, "buf"):
        return _worker_shm.buf

    if (
        _worker_projection_map is not None
        and _worker_projection_path == shm_name
    ):
        return _worker_projection_map

    _close_worker_shm()
    handle = open(shm_name, "r+b")
    mapping = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_WRITE)
    _worker_projection_file = handle
    _worker_projection_map = mapping
    _worker_projection_path = shm_name
    return mapping


def _reset_worker_projection() -> None:
    """Drop any cached local projection buffer for this worker."""
    global _worker_projection
    _worker_projection = None


def _extract_terms(text: str, limit: int = 64) -> list[str]:
    seen: list[str] = []
    for token in _IDENT_RE.findall(text or ""):
        normalized = token.lower()
        if normalized in seen or len(normalized) < 3:
            continue
        seen.append(normalized)
        if len(seen) >= limit:
            break
    return seen


def _entity_payload(entity: Any, repo_path: str) -> tuple[str, dict[str, Any]]:
    metadata = dict(getattr(entity, "metadata", {}) or {})
    rel_path = canonicalize_rel_path(getattr(entity, "file_path", ""), repo_path=repo_path)
    symbol_terms = [str(term).lower() for term in metadata.get("symbol_terms", []) or []]
    path_terms = [str(term).lower() for term in metadata.get("path_terms", []) or []]
    doc_terms = [str(term).lower() for term in metadata.get("doc_terms", []) or []]
    if not symbol_terms:
        symbol_terms = _extract_terms(str(getattr(entity, "name", "")), limit=48)
    if not path_terms:
        path_terms = _extract_terms(rel_path.replace("/", " "), limit=32)
    if not doc_terms:
        doc_terms = _extract_terms(str(getattr(entity, "content", ""))[:4096], limit=96)
    body = str(getattr(entity, "content", "") or "")
    if len(body) > _MAX_EMBED_TEXT_CHARS:
        body = body[:_EMBED_HEAD_CHARS] + "\n...\n" + body[-_EMBED_TAIL_CHARS:]
    text = "\n".join(
        [
            f"name {getattr(entity, 'name', '')}",
            f"type {getattr(entity, 'type', '')}",
            f"file {rel_path}",
            (
                f"lines {int(getattr(entity, 'start_line', 0) or 0)} "
                f"{int(getattr(entity, 'end_line', 0) or 0)}"
            ),
            "symbols " + " ".join(symbol_terms[:24]),
            "paths " + " ".join(path_terms[:24]),
            "docs " + " ".join(doc_terms[:32]),
            "",
            body,
        ]
    )
    payload = {
        "terms": sorted(set(symbol_terms + path_terms + doc_terms)),
        "symbol_terms": symbol_terms,
        "path_terms": path_terms,
        "doc_terms": doc_terms,
        "entity_kind": metadata.get("entity_kind", getattr(entity, "type", "symbol")),
        "parent_symbol": metadata.get("parent_symbol"),
        "file_role": metadata.get("file_role", classify_file_role(rel_path)),
        "chunk_role": metadata.get("chunk_role"),
        "stale_at_index_time": False,
    }
    return text, payload


def _run_native_pipeline(
    *,
    texts: list[str],
    projection_buffer: object,
    vocab_size: int,
    active_dim: int,
    trie: Any,
    num_threads: int,
    batch_capacity: int,
    max_total_texts: int,
    target_dim: int,
) -> list[array]:
    runtime = _native_indexer
    if runtime is None:
        dim = int(target_dim or active_dim)
        return [array("f", [0.0]) * dim for _ in texts]
    return runtime.full_pipeline(
        texts=texts,
        projection_buffer=projection_buffer,
        vocab_size=vocab_size,
        dim=int(target_dim or active_dim),
        max_length=512,
        trie=trie,
        num_threads=num_threads,
    )


def process_batch_worker_native(
    file_paths: list[str],
    active_dim: int,
    total_dim: int,
    vocab_size: int,
    shm_name: str = SHM_NAME,
    codebook_path: str | None = None,
    repo_path: str | None = None,
    num_threads: int = 1,
    batch_capacity: int | None = None,
    max_total_text_chars: int | None = None,
    enforce_quotas: bool | None = None,
    emit_callback: Any | None = None,
    max_total_texts: int | None = None,
) -> tuple[list[dict[str, Any]], list[array] | None, list[str], dict[str, Any]]:
    """Process a batch of files and emit vectors plus metadata."""
    del enforce_quotas

    started = time.perf_counter()
    _initialize_worker()
    projection = _projection_buffer(
        shm_name=shm_name,
        vocab_size=int(vocab_size),
        active_dim=int(active_dim),
    )
    trie = _load_codebook_manual(codebook_path) if codebook_path else None
    repo_root = os.path.abspath(repo_path or os.getcwd())
    capacity = max(1, int(batch_capacity or os.getenv("SAGUARO_NATIVE_BATCH_CAPACITY", "128") or 128))
    total_text_limit = int(
        max_total_texts
        or max_total_text_chars
        or os.getenv("SAGUARO_NATIVE_MAX_TOTAL_TEXTS", "0")
        or 0
    )

    meta_rows: list[dict[str, Any]] = []
    vector_rows: list[array] = []
    touched: list[str] = []
    parse_seconds = 0.0
    pipeline_seconds = 0.0
    files_with_entities = 0
    quota_hits = 0
    text_chars_processed = 0

    for file_path in [os.path.abspath(path) for path in file_paths if path]:
        if os.path.exists(file_path) and file_path not in touched:
            touched.append(file_path)

        parse_started = time.perf_counter()
        try:
            entities = list(_parser.parse_file(file_path) or [])
        except Exception as exc:
            logger.debug("Worker failed on %s: %s", file_path, exc)
            continue
        parse_seconds += max(0.0, time.perf_counter() - parse_started)
        if not entities:
            continue

        file_texts: list[str] = []
        file_meta: list[dict[str, Any]] = []
        for idx, entity in enumerate(entities):
            text, payload = _entity_payload(entity, repo_root)
            text_chars_processed += len(text)
            file_texts.append(text)

            resolved_file = os.path.abspath(getattr(entity, "file_path", file_path) or file_path)
            identity = entity_identity(
                repo_root,
                resolved_file,
                str(getattr(entity, "name", "")),
                str(getattr(entity, "type", "symbol")),
                int(getattr(entity, "start_line", 0) or 0),
            )
            file_meta.append(
                {
                    "entity_id": identity["entity_id"],
                    "segment_id": f"{identity['entity_id']}::segment::{idx}",
                    "name": identity["display_name"],
                    "qualified_name": identity["qualified_name"],
                    "type": str(getattr(entity, "type", "symbol")),
                    "file": identity["rel_file"].replace("\\", "/"),
                    "line": int(getattr(entity, "start_line", 0) or 0),
                    "end_line": int(getattr(entity, "end_line", 0) or 0),
                    "terms": payload["terms"],
                    "entity_kind": payload["entity_kind"],
                    "symbol_terms": payload["symbol_terms"],
                    "path_terms": payload["path_terms"],
                    "doc_terms": payload["doc_terms"],
                    "parent_symbol": payload["parent_symbol"],
                    "file_role": payload["file_role"],
                    "chunk_role": payload["chunk_role"],
                    "stale_at_index_time": payload["stale_at_index_time"],
                }
            )

        if len(file_texts) > capacity:
            quota_hits += (len(file_texts) - 1) // capacity

        pipeline_started = time.perf_counter()
        doc_vectors = _run_native_pipeline(
            texts=file_texts,
            projection_buffer=projection,
            vocab_size=int(vocab_size),
            active_dim=int(active_dim),
            trie=trie,
            num_threads=max(1, int(num_threads or 1)),
            batch_capacity=capacity,
            max_total_texts=total_text_limit,
            target_dim=int(active_dim),
        )
        pipeline_seconds += max(0.0, time.perf_counter() - pipeline_started)
        if len(doc_vectors) == 0:
            continue

        files_with_entities += 1
        if emit_callback is not None:
            emit_callback(file_meta, doc_vectors)
        else:
            meta_rows.extend(file_meta)
            vector_rows.extend(doc_vectors)

    vector_payload = vector_rows if vector_rows else None
    metrics = {
        "parse_seconds": round(parse_seconds, 6),
        "pipeline_seconds": round(pipeline_seconds, 6),
        "files_with_entities": int(files_with_entities),
        "emitted_vector_bytes": int(
            sum(len(vec) * 4 for vec in vector_rows) if vector_rows else 0
        ),
        "queue_wait_seconds": 0.0,
        "queue_depth_max": 0,
        "quota_hits": int(quota_hits),
        "entities_dropped_by_quota": 0,
        "text_chars_processed": int(text_chars_processed),
        "duration_seconds": round(max(0.0, time.perf_counter() - started), 6),
    }
    return meta_rows, vector_payload, touched, metrics


process_batch_worker_memory_optimized = process_batch_worker_native
