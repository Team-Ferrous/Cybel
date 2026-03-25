"""Repository analysis pack generation."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List

IGNORE_DIRS = {
    ".git",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    ".anvil",
    ".saguaro",
    "venv",
    "node_modules",
}
LANGUAGE_MAP = {
    ".py": "python",
    ".c": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".h": "c_header",
    ".hpp": "cpp_header",
    ".json": "json",
    ".md": "markdown",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".sh": "shell",
}
BUILD_FILES = {
    "pyproject.toml",
    "setup.py",
    "requirements.txt",
    "package.json",
    "CMakeLists.txt",
    "Makefile",
    "build.gradle",
    "Cargo.toml",
}
MATH_HINTS = ("matrix", "gradient", "optimizer", "tensor", "eigen", "loss", "fft")
PERF_HINTS = ("simd", "avx", "openmp", "thread", "latency", "throughput", "vectorize")


class AnalysisPackBuilder:
    """Builds normalized repo analysis packs for attached or cached repositories."""

    def __init__(
        self, *, max_symbols_per_file: int = 8, max_usages_per_symbol: int = 12
    ):
        self.max_symbols_per_file = max_symbols_per_file
        self.max_usages_per_symbol = max_usages_per_symbol

    def build(
        self, repo_id: str, repo_path: str, *, role: str = "analysis"
    ) -> Dict[str, Any]:
        repo_root = os.path.abspath(repo_path)
        file_records = self._catalog_files(repo_root)
        token_index = self._build_token_index(file_records)

        language_counts: Counter[str] = Counter()
        build_files: List[str] = []
        test_files: List[str] = []
        entry_points: List[str] = []
        tech_stack: set[str] = set()

        saguaro_enabled = self._saguaro_enabled(repo_root)
        for record in file_records:
            language = str(record["language"])
            if language != "unknown":
                language_counts[language] += 1
            if record["classification"] == "build":
                build_files.append(str(record["path"]))
            if record["classification"] == "test":
                test_files.append(str(record["path"]))
            if os.path.basename(str(record["path"])) in {"main.py", "setup.py"}:
                entry_points.append(str(record["path"]))
            tech_stack.update(record["tags"])
            record["usage_traces"] = self._build_usage_traces(
                record,
                token_index,
                repo_root,
                saguaro_enabled=saguaro_enabled,
            )
            record.pop("_content_lines", None)

        total_nonempty = sum(int(record["nonempty_lines"]) for record in file_records)
        cpp_files = sum(
            1 for record in file_records if str(record["language"]).startswith("c")
        )
        python_files = sum(
            1 for record in file_records if record["language"] == "python"
        )
        directories = len(
            {os.path.dirname(str(record["path"])) for record in file_records}
        )
        repo_dossier = self._build_repo_dossier(
            repo_id=repo_id,
            repo_root=repo_root,
            role=role,
            file_records=file_records,
            language_counts=language_counts,
            entry_points=entry_points,
            build_files=build_files,
            test_files=test_files,
            tech_stack=sorted(tech_stack),
            file_count=len(file_records),
            loc=total_nonempty,
            python_files=python_files,
            cpp_files=cpp_files,
            directories=directories,
        )
        build_fingerprint = self._build_fingerprint(
            repo_root=repo_root,
            build_files=build_files,
            tech_stack=tech_stack,
            file_records=file_records,
        )
        capability_matrix = self._capability_matrix(
            language_counts=language_counts,
            build_fingerprint=build_fingerprint,
        )
        semantic_inventory = self._semantic_inventory(file_records)
        repo_dossier["build_fingerprint"] = build_fingerprint
        repo_dossier["capability_matrix"] = capability_matrix
        repo_dossier["semantic_inventory"] = semantic_inventory

        return {
            "repo_id": repo_id,
            "repo_path": repo_root,
            "role": role,
            "analysis_mode": "sequential_file_subagents",
            "trace_provider": (
                "static_scan+saguaro" if saguaro_enabled else "static_scan"
            ),
            "file_count": len(file_records),
            "python_files": python_files,
            "cpp_files": cpp_files,
            "directories": directories,
            "loc": total_nonempty,
            "languages": dict(language_counts),
            "build_files": build_files,
            "test_files": test_files,
            "entry_points": sorted(set(entry_points)),
            "tech_stack": sorted(tech_stack),
            "build_fingerprint": build_fingerprint,
            "capability_matrix": capability_matrix,
            "semantic_inventory": semantic_inventory,
            "repo_dossier": repo_dossier,
            "files": file_records,
        }

    def _catalog_files(self, repo_root: str) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for index, rel_path in enumerate(self._iter_repo_files(repo_root), start=1):
            absolute_path = os.path.join(repo_root, rel_path)
            content = self._read_text(absolute_path)
            symbols, imports = self._extract_symbols(rel_path, content)
            records.append(
                {
                    "order": index,
                    "path": rel_path,
                    "language": self._language_for_path(rel_path),
                    "classification": self._classify_path(rel_path),
                    "size_bytes": os.path.getsize(absolute_path),
                    "line_count": len(content.splitlines()),
                    "nonempty_lines": sum(
                        1 for line in content.splitlines() if line.strip()
                    ),
                    "digest": hashlib.sha1(content.encode("utf-8")).hexdigest(),
                    "symbols": symbols[: self.max_symbols_per_file],
                    "imports": imports,
                    "tags": self._tags_for_content(rel_path, content),
                    "analysis_notes": self._analysis_notes(rel_path, content),
                    "_content_lines": content.splitlines(),
                }
            )
        return records

    def _build_token_index(
        self, file_records: Iterable[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        index: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for record in file_records:
            for line_number, line in enumerate(
                record.get("_content_lines", []), start=1
            ):
                for token in set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", line)):
                    index[token].append(
                        {
                            "path": record["path"],
                            "line": line_number,
                            "snippet": line.strip()[:160],
                        }
                    )
        return index

    def _build_usage_traces(
        self,
        record: Dict[str, Any],
        token_index: Dict[str, List[Dict[str, Any]]],
        repo_root: str,
        *,
        saguaro_enabled: bool,
    ) -> List[Dict[str, Any]]:
        traces: List[Dict[str, Any]] = []
        for symbol in record.get("symbols", []):
            symbol_name = str(symbol.get("name") or "")
            if not symbol_name:
                continue
            matches = []
            for occurrence in token_index.get(symbol_name, []):
                if occurrence["path"] == record["path"] and occurrence["line"] == int(
                    symbol.get("line") or 0
                ):
                    continue
                matches.append(occurrence)
            trace: Dict[str, Any] = {
                "symbol": symbol_name,
                "kind": symbol.get("kind", "symbol"),
                "definition_line": int(symbol.get("line") or 0),
                "reference_count": len(matches),
                "references": matches[: self.max_usages_per_symbol],
                "provider": "static_scan",
            }
            if saguaro_enabled:
                impact = self._run_saguaro_impact(repo_root, str(record["path"]))
                if impact:
                    trace["provider"] = "static_scan+saguaro"
                    trace["saguaro_impact"] = impact
            traces.append(trace)
        return traces

    def _extract_symbols(
        self, rel_path: str, content: str
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        if rel_path.endswith(".py"):
            return self._extract_python_symbols(rel_path, content)
        return self._extract_generic_symbols(content)

    @staticmethod
    def _extract_python_symbols(
        rel_path: str, content: str
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        try:
            tree = ast.parse(content, filename=rel_path)
        except SyntaxError:
            return [], []
        symbols: List[Dict[str, Any]] = []
        imports: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                symbols.append(
                    {"name": node.name, "kind": "class", "line": node.lineno}
                )
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbols.append(
                    {"name": node.name, "kind": "function", "line": node.lineno}
                )
            elif isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        symbols.sort(key=lambda item: (int(item["line"]), str(item["name"])))
        return symbols, sorted(set(imports))

    @staticmethod
    def _extract_generic_symbols(
        content: str,
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        symbols: List[Dict[str, Any]] = []
        for line_number, line in enumerate(content.splitlines(), start=1):
            class_match = re.search(
                r"\b(class|struct)\s+([A-Za-z_][A-Za-z0-9_]*)", line
            )
            if class_match:
                symbols.append(
                    {
                        "name": class_match.group(2),
                        "kind": class_match.group(1),
                        "line": line_number,
                    }
                )
            func_match = re.search(
                r"\b([A-Za-z_][A-Za-z0-9_:<>]*)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
                line,
            )
            if func_match and func_match.group(2) not in {
                "if",
                "for",
                "while",
                "switch",
            }:
                symbols.append(
                    {
                        "name": func_match.group(2),
                        "kind": "function",
                        "line": line_number,
                    }
                )
        return symbols[:12], []

    @staticmethod
    def _iter_repo_files(repo_root: str) -> List[str]:
        rel_paths: List[str] = []
        for root, dirs, files in os.walk(repo_root):
            dirs[:] = sorted(entry for entry in dirs if entry not in IGNORE_DIRS)
            for filename in sorted(files):
                rel_paths.append(
                    os.path.relpath(os.path.join(root, filename), repo_root)
                )
        rel_paths.sort()
        return rel_paths

    @staticmethod
    def _read_text(path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                return handle.read()
        except Exception:
            return ""

    @staticmethod
    def _language_for_path(rel_path: str) -> str:
        return LANGUAGE_MAP.get(os.path.splitext(rel_path)[1].lower(), "unknown")

    @staticmethod
    def _classify_path(rel_path: str) -> str:
        filename = os.path.basename(rel_path)
        normalized = rel_path.replace("\\", "/")
        if filename in BUILD_FILES:
            return "build"
        if filename.startswith("test_") or normalized.startswith("tests/"):
            return "test"
        if filename.endswith((".md", ".rst")):
            return "docs"
        if filename.endswith((".json", ".toml", ".yaml", ".yml")):
            return "config"
        return "source"

    @staticmethod
    def _tags_for_content(rel_path: str, content: str) -> List[str]:
        lowered = content.lower()
        tags: set[str] = set()
        if any(term in lowered for term in MATH_HINTS):
            tags.add("math")
        if any(term in lowered for term in PERF_HINTS):
            tags.add("performance")
        if "import pytest" in lowered or "assert " in lowered:
            tags.add("testing")
        if rel_path.endswith((".cc", ".cpp", ".h", ".hpp")):
            tags.add("native")
        if rel_path.endswith(".py"):
            tags.add("python")
        if "cmake" in lowered or os.path.basename(rel_path) == "CMakeLists.txt":
            tags.add("build")
        return sorted(tags)

    @staticmethod
    def _analysis_notes(rel_path: str, content: str) -> List[str]:
        notes: List[str] = []
        lowered = content.lower()
        if "todo" in lowered:
            notes.append("Contains TODO markers.")
        if "fixme" in lowered:
            notes.append("Contains FIXME markers.")
        if rel_path.endswith((".cc", ".cpp", ".h", ".hpp")) and "openmp" in lowered:
            notes.append("Native parallelism markers detected.")
        if rel_path.endswith(".py") and "except:" in content:
            notes.append("Bare except detected.")
        return notes

    def _build_repo_dossier(
        self,
        *,
        repo_id: str,
        repo_root: str,
        role: str,
        file_records: List[Dict[str, Any]],
        language_counts: Counter[str],
        entry_points: List[str],
        build_files: List[str],
        test_files: List[str],
        tech_stack: List[str],
        file_count: int,
        loc: int,
        python_files: int,
        cpp_files: int,
        directories: int,
    ) -> Dict[str, Any]:
        summary = {
            "file_count": file_count,
            "loc": loc,
            "python_files": python_files,
            "cpp_files": cpp_files,
            "directories": directories,
            "languages": dict(language_counts),
        }
        payload_for_digest = {
            "repo_id": repo_id,
            "role": role,
            "repo_path": repo_root,
            "summary": summary,
            "entry_points": sorted(set(entry_points)),
            "build_files": sorted(set(build_files)),
            "test_files": sorted(set(test_files)),
            "tech_stack": tech_stack,
            "reuse_candidates": self._reuse_candidates(file_records),
            "risk_signals": self._risk_signals(file_records),
        }
        digest = hashlib.sha1(
            json.dumps(payload_for_digest, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        envelope = {
            "schema_version": "evidence_envelope.v1",
            "producer": "AnalysisPackBuilder",
            "evidence_type": "repo_dossier",
            "summary": (
                f"Repo dossier for {repo_id}: {file_count} files, "
                f"{python_files} Python, {cpp_files} native."
            ),
            "query": f"repo_dossier::{repo_id}",
            "sources": [{"path": repo_root, "role": role}],
            "artifacts": [],
            "metrics": {
                "file_count": file_count,
                "loc": loc,
                "python_files": python_files,
                "cpp_files": cpp_files,
            },
            "digest": digest,
        }
        return {
            "schema_version": "repo_dossier.v1",
            "dossier_id": f"{repo_id}:repo_dossier",
            "repo_id": repo_id,
            "repo_path": repo_root,
            "repo_role": role,
            "digest": digest,
            "summary": summary,
            "entry_points": sorted(set(entry_points)),
            "build_files": sorted(set(build_files)),
            "test_files": sorted(set(test_files)),
            "tech_stack": tech_stack,
            "reuse_candidates": payload_for_digest["reuse_candidates"],
            "risk_signals": payload_for_digest["risk_signals"],
            "evidence_envelope": envelope,
        }

    @staticmethod
    def _build_fingerprint(
        *,
        repo_root: str,
        build_files: List[str],
        tech_stack: List[str],
        file_records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        primary = "none"
        build_names = {os.path.basename(path) for path in build_files}
        if "CMakeLists.txt" in build_names:
            primary = "cmake"
        elif "Cargo.toml" in build_names:
            primary = "cargo"
        elif "package.json" in build_names:
            primary = "npm"
        elif build_names & {"pyproject.toml", "requirements.txt", "setup.py"}:
            primary = "python"
        include_roots = sorted(
            {
                os.path.dirname(str(record.get("path") or ""))
                for record in file_records
                if str(record.get("language") or "").startswith(("c", "cpp"))
            }
        )
        return {
            "repo_root": repo_root,
            "primary_build_system": primary,
            "build_files": sorted(set(build_files)),
            "include_roots": include_roots[:16],
            "tech_stack": sorted(set(tech_stack)),
            "build_backed": primary in {"cmake", "cargo", "npm", "python"},
        }

    @staticmethod
    def _capability_matrix(
        *,
        language_counts: Counter[str],
        build_fingerprint: Dict[str, Any],
    ) -> Dict[str, Any]:
        deep_languages = {
            language
            for language in language_counts
            if language in {"python", "c", "cpp", "c_header", "cpp_header"}
        }
        per_language = {}
        total = max(1, sum(int(count) for count in language_counts.values()))
        for language, count in language_counts.items():
            deep = language in deep_languages
            per_language[language] = {
                "count": int(count),
                "lexical": True,
                "ast": deep,
                "cfg": deep,
                "dfg": deep,
                "symbol": deep,
                "build_backed": bool(build_fingerprint.get("build_backed")) and language in {"c", "cpp", "c_header", "cpp_header"},
                "confidence": "high" if deep else "medium",
            }
        deep_count = sum(int(language_counts.get(language, 0)) for language in deep_languages)
        return {
            "per_language": per_language,
            "deep_languages": sorted(deep_languages),
            "deep_parse_coverage_percent": round((deep_count / total) * 100.0, 1),
        }

    @staticmethod
    def _semantic_inventory(file_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        symbols = []
        for record in file_records:
            path_value = str(record.get("path") or "")
            for symbol in record.get("symbols", [])[:4]:
                symbols.append(
                    {
                        "name": symbol.get("name", ""),
                        "kind": symbol.get("kind", "symbol"),
                        "path": path_value,
                        "line": int(symbol.get("line", 0) or 0),
                    }
                )
        return {
            "top_symbols": symbols[:32],
            "function_interior_candidates": [
                item for item in symbols if item.get("kind") in {"function", "method"}
            ][:24],
        }

    @staticmethod
    def _reuse_candidates(file_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        for record in file_records:
            if str(record.get("classification")) != "source":
                continue
            symbols = list(record.get("symbols") or [])
            traces = list(record.get("usage_traces") or [])
            score = float(
                len(symbols)
                + sum(int(item.get("reference_count", 0)) for item in traces) * 0.5
            )
            if score <= 0:
                continue
            candidates.append(
                {
                    "path": str(record.get("path") or ""),
                    "language": str(record.get("language") or "unknown"),
                    "symbol_count": len(symbols),
                    "reference_count": sum(
                        int(item.get("reference_count", 0)) for item in traces
                    ),
                    "score": round(score, 2),
                }
            )
        candidates.sort(key=lambda item: (-float(item["score"]), item["path"]))
        return candidates[:10]

    @staticmethod
    def _risk_signals(file_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        signals: List[Dict[str, Any]] = []
        for record in file_records:
            notes = list(record.get("analysis_notes") or [])
            if not notes:
                continue
            signals.append(
                {
                    "path": str(record.get("path") or ""),
                    "classification": str(record.get("classification") or "source"),
                    "notes": notes[:4],
                }
            )
        return signals[:20]

    @staticmethod
    def _saguaro_enabled(repo_root: str) -> bool:
        return shutil.which("saguaro") is not None and os.path.exists(
            os.path.join(repo_root, ".saguaro")
        )

    @staticmethod
    def _run_saguaro_impact(repo_root: str, rel_path: str) -> List[str]:
        try:
            completed = subprocess.run(
                ["saguaro", "impact", "--path", rel_path],
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
                timeout=20,
            )
        except Exception:
            return []
        if completed.returncode != 0:
            return []
        lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        return lines[:12]
