"""Deterministic duplicate and redundancy detection."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any

_SUPPORTED_SUFFIXES = {
    ".py",
    ".pyi",
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hpp",
    ".hh",
    ".hxx",
}
_IGNORED_PREFIXES = (
    ".git/",
    ".saguaro/",
    ".anvil/",
    "repo_analysis/",
    "Saguaro/",
    "build/",
    "dist/",
    "venv/",
    "core/native/build/",
    "saguaro/native/build/",
    "saguaro/native/build_release/",
    "saguaro/native/build_test/",
)
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT_RE = re.compile(r"//.*?$|#.*?$", re.MULTILINE)
_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_WS_RE = re.compile(r"\s+")


class DuplicateAnalyzer:
    """Find exact and normalized duplicate code clusters."""

    def __init__(self, repo_path: str) -> None:
        self.repo_path = os.path.abspath(repo_path)

    def analyze(self, path: str = ".") -> dict[str, Any]:
        target = (
            path
            if os.path.isabs(path)
            else os.path.abspath(os.path.join(self.repo_path, path))
        )
        exact: dict[str, list[dict[str, Any]]] = {}
        structural: dict[str, list[dict[str, Any]]] = {}
        file_rows: list[dict[str, Any]] = []
        for rel_path in self._iter_files(target):
            row = self._fingerprint(rel_path)
            if row is None:
                continue
            file_rows.append(row)
            exact.setdefault(str(row["exact_hash"]), []).append(row)
            structural.setdefault(str(row["structural_hash"]), []).append(row)

        clusters: list[dict[str, Any]] = []
        for fingerprint, items in exact.items():
            if len(items) < 2:
                continue
            clusters.append(
                self._cluster_payload(
                    items=items,
                    fingerprint=fingerprint,
                    kind="exact",
                    confidence=0.99,
                )
            )
        for fingerprint, items in structural.items():
            if len(items) < 2:
                continue
            paths = {item["path"] for item in items}
            if any(
                paths == {cluster_path for cluster_path in cluster["paths"]}
                for cluster in clusters
            ):
                continue
            clusters.append(
                self._cluster_payload(
                    items=items,
                    fingerprint=fingerprint,
                    kind="structural",
                    confidence=0.78,
                )
            )

        clusters.sort(
            key=lambda item: (
                float(item.get("confidence", 0.0)),
                int(item.get("total_lines", 0)),
                int(item.get("file_count", 0)),
            ),
            reverse=True,
        )
        return {
            "status": "ok",
            "count": len(clusters),
            "clusters": clusters,
            "scanned_files": len(file_rows),
        }

    def explain(self, path: str) -> dict[str, Any]:
        rel_path = self._safe_relpath(path)
        report = self.analyze(path=".")
        for cluster in report.get("clusters", []):
            if rel_path in set(cluster.get("paths", [])):
                return {"status": "ok", "path": rel_path, "cluster": cluster}
        return {"status": "missing", "path": rel_path}

    def file_cluster_map(self, path: str = ".") -> dict[str, list[dict[str, Any]]]:
        report = self.analyze(path=path)
        file_map: dict[str, list[dict[str, Any]]] = {}
        for cluster in report.get("clusters", []):
            for rel_path in cluster.get("paths", []):
                file_map.setdefault(str(rel_path), []).append(cluster)
        return file_map

    def _iter_files(self, target: str) -> list[str]:
        if os.path.isfile(target):
            rel = self._safe_relpath(target)
            return [rel] if self._in_scope(rel) else []
        files: list[str] = []
        for root, dirs, names in os.walk(target):
            rel_root = self._safe_relpath(root)
            if not self._in_scope(rel_root, is_dir=True):
                dirs[:] = []
                continue
            dirs[:] = [
                item
                for item in sorted(dirs)
                if self._in_scope(os.path.join(rel_root, item), is_dir=True)
            ]
            for name in sorted(names):
                rel = self._safe_relpath(os.path.join(root, name))
                if self._in_scope(rel):
                    files.append(rel)
        return files

    def _in_scope(self, rel_path: str, *, is_dir: bool = False) -> bool:
        normalized = rel_path.replace("\\", "/")
        if normalized.startswith("./"):
            normalized = normalized[2:]
        if not normalized:
            return True
        if any(normalized.startswith(prefix) for prefix in _IGNORED_PREFIXES):
            return False
        if is_dir:
            return True
        return Path(normalized).suffix.lower() in _SUPPORTED_SUFFIXES

    def _fingerprint(self, rel_path: str) -> dict[str, Any] | None:
        abs_path = os.path.join(self.repo_path, rel_path)
        try:
            source = Path(abs_path).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return None
        normalized = self._normalize_source(source)
        structural = self._structural_normalize(normalized)
        line_count = len(source.splitlines())
        if not normalized.strip() or line_count < 3 or len(normalized) < 32:
            return None
        return {
            "path": rel_path,
            "kind": Path(rel_path).suffix.lower(),
            "line_count": line_count,
            "exact_hash": hashlib.sha1(normalized.encode("utf-8")).hexdigest(),
            "structural_hash": hashlib.sha1(structural.encode("utf-8")).hexdigest(),
        }

    @staticmethod
    def _normalize_source(source: str) -> str:
        token = _BLOCK_COMMENT_RE.sub("", source)
        token = _LINE_COMMENT_RE.sub("", token)
        token = _WS_RE.sub(" ", token)
        return token.strip()

    @staticmethod
    def _structural_normalize(source: str) -> str:
        token = _NUMBER_RE.sub("NUM", source)
        token = _IDENT_RE.sub("ID", token)
        token = _WS_RE.sub(" ", token)
        return token.strip()

    @staticmethod
    def _cluster_payload(
        *,
        items: list[dict[str, Any]],
        fingerprint: str,
        kind: str,
        confidence: float,
    ) -> dict[str, Any]:
        ordered = sorted(items, key=lambda item: str(item["path"]))
        return {
            "id": f"dup::{kind}::{fingerprint[:12]}",
            "kind": kind,
            "fingerprint": fingerprint,
            "confidence": confidence,
            "file_count": len(ordered),
            "total_lines": sum(int(item.get("line_count", 0) or 0) for item in ordered),
            "paths": [item["path"] for item in ordered],
        }

    def _safe_relpath(self, path: str) -> str:
        target = os.path.abspath(path)
        try:
            rel = os.path.relpath(target, self.repo_path)
        except ValueError:
            rel = os.path.basename(target)
        rel = rel.replace("\\", "/")
        if rel.startswith("./"):
            rel = rel[2:]
        return rel
