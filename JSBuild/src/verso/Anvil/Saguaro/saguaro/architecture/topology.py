"""Repository topology and architecture conformance analysis."""

from __future__ import annotations

import ast
import logging
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from saguaro.utils.file_utils import build_corpus_manifest

logger = logging.getLogger(__name__)

_CODE_SUFFIXES = {
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
_TEXT_SUFFIXES = _CODE_SUFFIXES | {
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".sh",
}
_C_FAMILY_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hpp",
    ".hh",
    ".hxx",
}
_IMPORT_RE = re.compile(r'^\s*#include\s*[<"]([^">]+)[">]', re.MULTILINE)


def _normalize_rel_path(path: str) -> str:
    token = str(path or "").replace("\\", "/").strip()
    if token.startswith("./"):
        token = token[2:]
    while token.startswith("/"):
        token = token[1:]
    while "//" in token:
        token = token.replace("//", "/")
    return token


def _top_level_token(path: str) -> str:
    rel = _normalize_rel_path(path)
    if not rel:
        return ""
    return rel.split("/", 1)[0]


def _path_matches_prefix(path: str, prefix: str) -> bool:
    normalized_path = _normalize_rel_path(path)
    normalized_prefix = _normalize_rel_path(prefix).rstrip("/")
    if not normalized_prefix:
        return False
    return normalized_path == normalized_prefix or normalized_path.startswith(
        normalized_prefix + "/"
    )


def _suffix_kind(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in {".h", ".hpp", ".hh", ".hxx"}:
        return "header"
    if suffix in {".c", ".cc", ".cpp", ".cxx"}:
        return "native_source"
    if suffix in {".py", ".pyi"}:
        return "python"
    if suffix in {".md", ".txt"}:
        return "documentation"
    if suffix in {".json", ".yaml", ".yml", ".toml"}:
        return "config"
    if suffix == ".sh":
        return "script"
    return "other"


@dataclass(frozen=True)
class ZoneRule:
    """Layout zone definition loaded from versioned policy data."""

    name: str
    roots: tuple[str, ...]
    allowed_dependencies: tuple[str, ...]
    allow_file_kinds: tuple[str, ...]
    description: str

    def matches(self, rel_path: str) -> bool:
        return any(_path_matches_prefix(rel_path, root) for root in self.roots)


class LayoutPolicy:
    """Versioned repository layout and dependency policy."""

    def __init__(self, repo_path: str, policy_path: str | None = None) -> None:
        self.repo_path = os.path.abspath(repo_path)
        self.policy_path = os.path.abspath(
            policy_path
            or os.path.join(self.repo_path, "standards", "REPO_LAYOUT.yaml")
        )
        self.payload = self._load()
        self.version = int(self.payload.get("version", 1) or 1)
        self.authoritative_roots = tuple(
            sorted(
                {
                    _normalize_rel_path(item)
                    for item in self.payload.get("authoritative_roots", [])
                    if _normalize_rel_path(item)
                }
            )
        )
        self.ignored_roots = tuple(
            sorted(
                {
                    _normalize_rel_path(item)
                    for item in self.payload.get("ignored_roots", [])
                    if _normalize_rel_path(item)
                }
            )
        )
        self.zones = self._load_zones()

    def _load(self) -> dict[str, Any]:
        if not os.path.exists(self.policy_path):
            return {"version": 1, "authoritative_roots": [], "ignored_roots": [], "zones": []}
        with open(self.policy_path, encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        return payload if isinstance(payload, dict) else {}

    def _load_zones(self) -> tuple[ZoneRule, ...]:
        zones: list[ZoneRule] = []
        for item in self.payload.get("zones", []) or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            zones.append(
                ZoneRule(
                    name=name,
                    roots=tuple(
                        _normalize_rel_path(root)
                        for root in item.get("roots", []) or []
                        if _normalize_rel_path(root)
                    ),
                    allowed_dependencies=tuple(
                        str(dep).strip()
                        for dep in item.get("allowed_dependencies", []) or []
                        if str(dep).strip()
                    ),
                    allow_file_kinds=tuple(
                        str(kind).strip()
                        for kind in item.get("allow_file_kinds", []) or []
                        if str(kind).strip()
                    ),
                    description=str(item.get("description") or "").strip(),
                )
            )
        zones.sort(key=lambda zone: max((len(root) for root in zone.roots), default=0), reverse=True)
        return tuple(zones)

    def classify(self, rel_path: str) -> str:
        for zone in self.zones:
            if zone.matches(rel_path):
                return zone.name
        return "unknown"

    def zone_rule(self, zone_name: str) -> ZoneRule | None:
        for zone in self.zones:
            if zone.name == zone_name:
                return zone
        return None

    def is_authoritative(self, rel_path: str) -> bool:
        if not self.authoritative_roots:
            return True
        top = _top_level_token(rel_path)
        return top in self.authoritative_roots or _normalize_rel_path(rel_path) in {
            root for root in self.authoritative_roots if "/" in root
        }

    def is_ignored(self, rel_path: str) -> bool:
        normalized = _normalize_rel_path(rel_path)
        return any(_path_matches_prefix(normalized, root) for root in self.ignored_roots)


class ArchitectureAnalyzer:
    """Build a topology map and deterministic architecture findings."""

    def __init__(self, repo_path: str, policy_path: str | None = None) -> None:
        self.repo_path = os.path.abspath(repo_path)
        self.policy = LayoutPolicy(self.repo_path, policy_path=policy_path)
        self._manifest = build_corpus_manifest(self.repo_path)
        self._module_to_file = self._build_module_index()
        self._known_files: set[str] = set()
        self._known_c_family_files: set[str] = set()
        self._basename_to_c_family_files: dict[str, tuple[str, ...]] = {}
        self._include_resolution_cache: dict[tuple[str, str], str | None] = {}

    def map(self, path: str = ".") -> dict[str, Any]:
        map_start = time.perf_counter()
        files = self._collect_files(path)
        self._prime_resolution_indexes(files)
        self._log_stage("collect_files", map_start, file_count=len(files))

        dep_start = time.perf_counter()
        dependencies, reverse_dependencies = self._dependency_maps(files)
        self._log_stage("dependency_maps", dep_start, file_count=len(dependencies))

        assign_start = time.perf_counter()
        assignments = self._zone_assignments(files)
        self._log_stage("zone_assignments", assign_start, file_count=len(assignments))

        violation_start = time.perf_counter()
        violations = self._violations(assignments, dependencies)
        self._log_stage("violations", violation_start, finding_count=len(violations))
        crossings = self._zone_crossings(assignments, dependencies)
        files_by_zone = Counter(item["zone"] for item in assignments.values())
        misplaced = [item for item in violations if str(item.get("rule_id")).startswith("AES-LAYOUT")]
        self._log_stage(
            "map_total",
            map_start,
            file_count=len(assignments),
            finding_count=len(violations),
        )
        return {
            "status": "ok",
            "policy": {
                "path": self.policy.policy_path,
                "version": self.policy.version,
                "authoritative_roots": list(self.policy.authoritative_roots),
                "ignored_roots": list(self.policy.ignored_roots),
                "zones": [
                    {
                        "name": zone.name,
                        "roots": list(zone.roots),
                        "allowed_dependencies": list(zone.allowed_dependencies),
                        "allow_file_kinds": list(zone.allow_file_kinds),
                        "description": zone.description,
                    }
                    for zone in self.policy.zones
                ],
            },
            "summary": {
                "file_count": len(assignments),
                "authoritative_file_count": sum(
                    1 for item in assignments.values() if bool(item.get("authoritative"))
                ),
                "zone_mapped_count": sum(
                    1 for item in assignments.values() if str(item.get("zone")) != "unknown"
                ),
                "misplaced_count": len(misplaced),
                "illegal_dependency_count": sum(
                    1
                    for item in violations
                    if str(item.get("rule_id")) == "AES-ARCH-101"
                ),
            },
            "files_by_zone": dict(sorted(files_by_zone.items())),
            "roots": self._root_summary(assignments),
            "zones": assignments,
            "dependencies": dependencies,
            "reverse_dependencies": reverse_dependencies,
            "zone_crossings": crossings,
            "violations": violations,
        }

    def zones(self, path: str | None = None) -> dict[str, Any]:
        report = self.map(path or ".")
        if path and path not in {".", "./"}:
            rel = self._safe_relpath(path)
            detail = report["zones"].get(rel)
            return {
                "status": "ok" if detail else "missing",
                "path": rel,
                "zone": detail,
            }
        return {
            "status": "ok",
            "count": len(report["zones"]),
            "zones": report["zones"],
        }

    def verify(self, path: str = ".") -> dict[str, Any]:
        report = self.map(path)
        findings = list(report.get("violations", []))
        status = "pass" if not findings else "fail"
        return {
            "status": status,
            "count": len(findings),
            "findings": findings,
            "summary": report.get("summary", {}),
            "policy": report.get("policy", {}),
        }

    def explain(self, path: str) -> dict[str, Any]:
        rel = self._safe_relpath(path)
        report = self.map(path)
        zone_row = report.get("zones", {}).get(rel)
        if not zone_row:
            return {"status": "missing", "path": rel}
        violations = [
            row for row in report.get("violations", []) if str(row.get("file") or "") == rel
        ]
        dependencies = report.get("dependencies", {}).get(rel, [])
        reverse_dependencies = report.get("reverse_dependencies", {}).get(rel, [])
        return {
            "status": "ok",
            "path": rel,
            "zone": zone_row,
            "dependencies": dependencies,
            "reverse_dependencies": reverse_dependencies,
            "violations": violations,
        }

    def health(self, path: str = ".") -> dict[str, Any]:
        report = self.map(path)
        summary = dict(report.get("summary", {}))
        file_count = max(int(summary.get("file_count", 0) or 0), 1)
        zone_mapped = int(summary.get("zone_mapped_count", 0) or 0)
        authoritative = int(summary.get("authoritative_file_count", 0) or 0)
        misplaced = int(summary.get("misplaced_count", 0) or 0)
        illegal = int(summary.get("illegal_dependency_count", 0) or 0)
        return {
            "status": "ok",
            "authoritative_roots": list(self.policy.authoritative_roots),
            "zone_count": len(self.policy.zones),
            "files_by_zone": report.get("files_by_zone", {}),
            "mapped_files": zone_mapped,
            "mapped_files_pct": round((zone_mapped / file_count) * 100.0, 1),
            "authoritative_files_pct": round((authoritative / file_count) * 100.0, 1),
            "misplaced_files": misplaced,
            "illegal_zone_crossings": illegal,
        }

    def _collect_files(self, path: str) -> list[str]:
        target_abs = (
            path
            if os.path.isabs(path)
            else os.path.abspath(os.path.join(self.repo_path, path))
        )
        if os.path.isfile(target_abs):
            rel = self._safe_relpath(target_abs)
            if self.policy.is_ignored(rel):
                return []
            return [rel]
        files: list[str] = []
        target_prefix = os.path.relpath(target_abs, self.repo_path).replace("\\", "/")
        if target_prefix == ".":
            target_prefix = ""
        for abs_path in self._manifest.files:
            if not abs_path.startswith(target_abs):
                continue
            rel = self._safe_relpath(abs_path)
            suffix = Path(abs_path).suffix.lower()
            if self.policy.is_ignored(rel):
                continue
            if suffix and suffix not in _TEXT_SUFFIXES:
                continue
            if target_prefix and not rel.startswith(target_prefix):
                continue
            files.append(rel)
        return sorted(set(files))

    def _zone_assignments(self, files: list[str]) -> dict[str, dict[str, Any]]:
        rows: dict[str, dict[str, Any]] = {}
        for rel in files:
            zone = self.policy.classify(rel)
            kind = _suffix_kind(rel)
            rule = self.policy.zone_rule(zone)
            allowed_kinds = list(rule.allow_file_kinds) if rule else []
            rows[rel] = {
                "path": rel,
                "root": _top_level_token(rel),
                "zone": zone,
                "kind": kind,
                "authoritative": self.policy.is_authoritative(rel),
                "policy_root": next(
                    (root for root in (rule.roots if rule else ()) if _path_matches_prefix(rel, root)),
                    None,
                ),
                "allow_file_kinds": allowed_kinds,
            }
        return rows

    def _build_module_index(self) -> dict[str, str]:
        module_index: dict[str, str] = {}
        for abs_path in self._manifest.files:
            name = os.path.basename(abs_path)
            if not name.endswith((".py", ".pyi")):
                continue
            rel = self._safe_relpath(abs_path)
            if self.policy.is_ignored(rel):
                continue
            module = rel[: -len(Path(name).suffix)].replace("/", ".")
            if module.endswith(".__init__"):
                module = module[: -len(".__init__")]
            module_index[module] = rel
        return module_index

    def _prime_resolution_indexes(self, files: list[str]) -> None:
        self._known_files = set(files)
        self._known_c_family_files = {
            rel for rel in files if Path(rel).suffix.lower() in _C_FAMILY_SUFFIXES
        }
        basename_map: dict[str, list[str]] = defaultdict(list)
        for rel in self._known_c_family_files:
            basename = Path(rel).name
            if basename:
                basename_map[basename].append(rel)
        self._basename_to_c_family_files = {
            key: tuple(
                sorted(
                    values,
                    key=lambda item: (len(item.split("/")), len(item), item),
                )
            )
            for key, values in basename_map.items()
        }
        self._include_resolution_cache = {}

    def _dependency_maps(
        self, files: list[str]
    ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
        forward: dict[str, list[dict[str, Any]]] = {}
        reverse: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for rel in files:
            edges = self._scan_dependencies(rel)
            forward[rel] = edges
            for edge in edges:
                target = str(edge.get("target") or "")
                if target:
                    reverse[target].append(
                        {
                            "source": rel,
                            "relation": edge.get("relation", "related"),
                            "line": int(edge.get("line", 0) or 0),
                        }
                    )
        return (
            {key: sorted(value, key=lambda item: (str(item.get("target") or ""), int(item.get("line", 0) or 0))) for key, value in forward.items()},
            {key: sorted(value, key=lambda item: (str(item.get("source") or ""), int(item.get("line", 0) or 0))) for key, value in reverse.items()},
        )

    def _scan_dependencies(self, rel_path: str) -> list[dict[str, Any]]:
        abs_path = os.path.join(self.repo_path, rel_path)
        try:
            source = Path(abs_path).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return []
        suffix = Path(rel_path).suffix.lower()
        if suffix in {".py", ".pyi"}:
            return self._python_dependencies(rel_path, source)
        if suffix in _C_FAMILY_SUFFIXES:
            return self._c_family_dependencies(rel_path, source)
        return []

    def _python_dependencies(self, rel_path: str, source: str) -> list[dict[str, Any]]:
        edges: list[dict[str, Any]] = []
        try:
            tree = ast.parse(source)
        except Exception:
            return edges
        current_module = rel_path[: -len(Path(rel_path).suffix)].replace("/", ".")
        if current_module.endswith(".__init__"):
            current_module = current_module[: -len(".__init__")]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    resolved = self._resolve_module_to_file(alias.name, current_module)
                    if not resolved:
                        continue
                    edges.append(
                        {
                            "target": resolved,
                            "relation": "imports",
                            "line": int(getattr(node, "lineno", 1) or 1),
                        }
                    )
            elif isinstance(node, ast.ImportFrom):
                module_name = str(node.module or "")
                resolved = self._resolve_relative_module(
                    current_module=current_module,
                    module_name=module_name,
                    level=int(getattr(node, "level", 0) or 0),
                )
                target = self._resolve_module_to_file(resolved, current_module)
                if target:
                    edges.append(
                        {
                            "target": target,
                            "relation": "imports",
                            "line": int(getattr(node, "lineno", 1) or 1),
                        }
                    )
        return edges

    def _resolve_relative_module(
        self,
        *,
        current_module: str,
        module_name: str,
        level: int,
    ) -> str:
        if level <= 0:
            return module_name
        parts = [part for part in current_module.split(".") if part]
        if parts:
            parts = parts[: max(0, len(parts) - level)]
        if module_name:
            parts.extend([part for part in module_name.split(".") if part])
        return ".".join(parts)

    def _resolve_module_to_file(
        self, module_name: str, current_module: str
    ) -> str | None:
        candidates = []
        if module_name:
            candidates.append(module_name)
        if "." in current_module and module_name:
            base = current_module.rsplit(".", 1)[0]
            candidates.append(f"{base}.{module_name}")
        for candidate in candidates:
            if candidate in self._module_to_file:
                return self._module_to_file[candidate]
            init_candidate = f"{candidate}.__init__"
            if init_candidate in self._module_to_file:
                return self._module_to_file[init_candidate]
        return None

    def _c_family_dependencies(self, rel_path: str, source: str) -> list[dict[str, Any]]:
        edges: list[dict[str, Any]] = []
        current_dir = os.path.dirname(rel_path)
        for match in _IMPORT_RE.finditer(source):
            include_target = str(match.group(1) or "").strip()
            if not include_target:
                continue
            resolved = self._resolve_include(current_dir=current_dir, include_target=include_target)
            if not resolved:
                continue
            line = source.count("\n", 0, match.start()) + 1
            edges.append(
                {
                    "target": resolved,
                    "relation": "includes",
                    "line": int(line),
                }
            )
        return edges

    def _resolve_include(self, *, current_dir: str, include_target: str) -> str | None:
        normalized = _normalize_rel_path(include_target)
        cache_key = (current_dir, normalized)
        if cache_key in self._include_resolution_cache:
            return self._include_resolution_cache[cache_key]
        direct = os.path.join(current_dir, normalized) if current_dir else normalized
        direct = _normalize_rel_path(direct)
        resolved = self._resolve_include_from_indexes(
            current_dir=current_dir,
            normalized=normalized,
            direct=direct,
        )
        self._include_resolution_cache[cache_key] = resolved
        return resolved

    def _resolve_include_from_indexes(
        self,
        *,
        current_dir: str,
        normalized: str,
        direct: str,
    ) -> str | None:
        for candidate in (direct, normalized):
            if candidate in self._known_c_family_files:
                return candidate
            if candidate and os.path.exists(os.path.join(self.repo_path, candidate)):
                return candidate

        basename = Path(normalized).name
        if not basename:
            return None
        candidates = self._basename_to_c_family_files.get(basename, ())
        if not candidates:
            return None

        scored = sorted(
            candidates,
            key=lambda rel: (
                self._include_match_score(
                    rel=rel,
                    current_dir=current_dir,
                    normalized=normalized,
                    direct=direct,
                ),
                -len(rel),
                rel,
            ),
            reverse=True,
        )
        return scored[0] if scored else None

    @staticmethod
    def _include_match_score(
        *,
        rel: str,
        current_dir: str,
        normalized: str,
        direct: str,
    ) -> int:
        score = 0
        if rel == direct:
            score += 100
        if rel == normalized:
            score += 90
        if normalized and rel.endswith(f"/{normalized}"):
            score += 80
        if current_dir and rel.startswith(f"{current_dir}/"):
            score += 20
        if rel.endswith(f"/{Path(normalized).name}") or rel == Path(normalized).name:
            score += 10
        return score

    def _zone_crossings(
        self,
        assignments: dict[str, dict[str, Any]],
        dependencies: dict[str, list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        counts: dict[tuple[str, str], int] = Counter()
        for rel, edges in dependencies.items():
            source_zone = str(assignments.get(rel, {}).get("zone") or "unknown")
            for edge in edges:
                target = str(edge.get("target") or "")
                target_zone = str(assignments.get(target, {}).get("zone") or "unknown")
                counts[(source_zone, target_zone)] += 1
        rows = [
            {"from_zone": src, "to_zone": dst, "count": count}
            for (src, dst), count in sorted(
                counts.items(),
                key=lambda item: (-item[1], item[0][0], item[0][1]),
            )
        ]
        return rows

    def _violations(
        self,
        assignments: dict[str, dict[str, Any]],
        dependencies: dict[str, list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []
        for rel, row in sorted(assignments.items()):
            kind = str(row.get("kind") or "other")
            zone = str(row.get("zone") or "unknown")
            if not bool(row.get("authoritative")):
                findings.append(
                    self._finding(
                        rule_id="AES-LAYOUT-001",
                        file=rel,
                        message="File is outside declared authoritative roots.",
                        severity="P1",
                    )
                )
            if zone == "unknown":
                findings.append(
                    self._finding(
                        rule_id="AES-LAYOUT-002",
                        file=rel,
                        message="File does not map to any declared architecture zone.",
                        severity="P1",
                    )
                )
                continue
            rule = self.policy.zone_rule(zone)
            if rule and rule.allow_file_kinds and kind not in set(rule.allow_file_kinds):
                findings.append(
                    self._finding(
                        rule_id="AES-LAYOUT-002",
                        file=rel,
                        message=(
                            f"File kind '{kind}' is not allowed in zone '{zone}'."
                        ),
                        severity="P1",
                    )
                )

        for rel, edges in sorted(dependencies.items()):
            source_zone = str(assignments.get(rel, {}).get("zone") or "unknown")
            source_rule = self.policy.zone_rule(source_zone)
            allowed = set(source_rule.allowed_dependencies if source_rule else ())
            for edge in edges:
                target = str(edge.get("target") or "")
                target_zone = str(assignments.get(target, {}).get("zone") or "unknown")
                if source_zone == "unknown" or target_zone == "unknown":
                    continue
                if allowed and target_zone not in allowed:
                    findings.append(
                        self._finding(
                            rule_id="AES-ARCH-101",
                            file=rel,
                            line=int(edge.get("line", 1) or 1),
                            message=(
                                f"Illegal zone dependency: {source_zone} -> {target_zone} via {target}"
                            ),
                            severity="P1",
                            context=f"{edge.get('relation', 'related')}:{target}",
                        )
                    )
                if self._public_header_violation(rel, target):
                    findings.append(
                        self._finding(
                            rule_id="AES-ARCH-102",
                            file=rel,
                            line=int(edge.get("line", 1) or 1),
                            message=(
                                f"Public header must not include internal header: {target}"
                            ),
                            severity="P1",
                            context=target,
                        )
                    )
        return sorted(
            findings,
            key=lambda item: (
                str(item.get("file") or ""),
                int(item.get("line", 0) or 0),
                str(item.get("rule_id") or ""),
                str(item.get("context") or ""),
            ),
        )

    @staticmethod
    def _public_header_violation(source: str, target: str) -> bool:
        source_norm = _normalize_rel_path(source)
        target_norm = _normalize_rel_path(target)
        if Path(source_norm).suffix.lower() not in {".h", ".hpp", ".hh", ".hxx"}:
            return False
        return "/public/" in f"/{source_norm}/" and "/internal/" in f"/{target_norm}/"

    @staticmethod
    def _finding(
        *,
        rule_id: str,
        file: str,
        message: str,
        severity: str,
        line: int = 1,
        context: str | None = None,
    ) -> dict[str, Any]:
        return {
            "rule_id": rule_id,
            "file": file,
            "line": int(line),
            "message": message,
            "severity": severity,
            "aal": "AAL-1" if severity in {"P0", "P1"} else "AAL-2",
            "closure_level": "blocking" if severity in {"P0", "P1"} else "guarded",
            "domain": ["universal"],
            "evidence_refs": [],
            "context": context or "",
        }

    def _root_summary(self, assignments: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        root_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for item in assignments.values():
            root = str(item.get("root") or "")
            zone = str(item.get("zone") or "unknown")
            if root:
                root_counts[root][zone] += 1
        return [
            {
                "root": root,
                "zone_counts": dict(sorted(counter.items())),
                "total_files": int(sum(counter.values())),
            }
            for root, counter in sorted(root_counts.items())
        ]

    def _safe_relpath(self, path: str) -> str:
        target = os.path.abspath(path)
        try:
            rel = os.path.relpath(target, self.repo_path)
        except ValueError:
            rel = os.path.basename(target)
        normalized = _normalize_rel_path(rel)
        return normalized if normalized != "." else ""

    def _log_stage(self, stage: str, started_at: float, **fields: Any) -> None:
        elapsed = time.perf_counter() - started_at
        threshold = float(os.getenv("SAGUARO_ARCH_SLOW_STAGE_SECONDS", "2.0") or 2.0)
        if logger.isEnabledFor(logging.INFO) or elapsed >= threshold:
            payload = ", ".join(
                f"{key}={value}" for key, value in sorted(fields.items())
            )
            message = (
                f"ArchitectureAnalyzer stage={stage} elapsed={elapsed:.3f}s"
                + (f" {payload}" if payload else "")
            )
            if elapsed >= threshold:
                logger.warning(message)
            else:
                logger.info(message)
