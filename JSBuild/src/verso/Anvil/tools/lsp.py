from __future__ import annotations

import ast
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class CachedDiagnostics:
    fingerprint: Tuple[int, int]
    diagnostics: List[Dict[str, Any]]


class LSPDiagnosticsCache:
    """Deterministic diagnostics cache keyed by file fingerprint."""

    def __init__(self, root_dir: str = ".") -> None:
        self.root_dir = os.path.abspath(root_dir)
        self._cache: Dict[str, CachedDiagnostics] = {}

    def _resolve(self, path: str) -> str:
        candidate = os.path.abspath(
            path if os.path.isabs(path) else os.path.join(self.root_dir, path)
        )
        if os.path.commonpath([candidate, self.root_dir]) != self.root_dir:
            raise ValueError(f"Path outside project root: {path}")
        return candidate

    def _fingerprint(self, file_path: str) -> Tuple[int, int]:
        st = os.stat(file_path)
        return int(st.st_mtime_ns), int(st.st_size)

    def _syntax_diagnostics(self, file_path: str) -> List[Dict[str, Any]]:
        if not file_path.endswith(".py"):
            return []
        try:
            source = Path(file_path).read_text(encoding="utf-8", errors="ignore")
            compile(source, file_path, "exec")
            return []
        except SyntaxError as exc:
            return [
                {
                    "source": "python",
                    "severity": "error",
                    "code": "syntax-error",
                    "message": exc.msg,
                    "line": int(exc.lineno or 0),
                    "col": int(exc.offset or 0),
                    "path": os.path.relpath(file_path, self.root_dir),
                }
            ]

    def _ruff_diagnostics(self, file_path: str) -> List[Dict[str, Any]]:
        if not file_path.endswith(".py"):
            return []
        try:
            result = subprocess.run(
                ["ruff", "check", "--output-format", "json", file_path],
                cwd=self.root_dir,
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return []

        if not result.stdout.strip():
            return []
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            return []

        diagnostics: List[Dict[str, Any]] = []
        for item in data:
            diagnostics.append(
                {
                    "source": "ruff",
                    "severity": "error",
                    "code": item.get("code"),
                    "message": item.get("message"),
                    "line": item.get("location", {}).get("row", 0),
                    "col": item.get("location", {}).get("column", 0),
                    "path": os.path.relpath(
                        item.get("filename", file_path), self.root_dir
                    ),
                }
            )
        return diagnostics

    def diagnostics_for_file(self, path: str) -> List[Dict[str, Any]]:
        full_path = self._resolve(path)
        if not os.path.exists(full_path):
            return [
                {
                    "source": "lsp",
                    "severity": "error",
                    "code": "file-not-found",
                    "message": "File not found",
                    "line": 0,
                    "col": 0,
                    "path": path,
                }
            ]

        fingerprint = self._fingerprint(full_path)
        cached = self._cache.get(full_path)
        if cached and cached.fingerprint == fingerprint:
            return cached.diagnostics

        diagnostics = self._syntax_diagnostics(full_path) + self._ruff_diagnostics(full_path)
        diagnostics.sort(
            key=lambda item: (
                item.get("path", ""),
                int(item.get("line", 0)),
                int(item.get("col", 0)),
                str(item.get("code", "")),
            )
        )
        self._cache[full_path] = CachedDiagnostics(
            fingerprint=fingerprint, diagnostics=diagnostics
        )
        return diagnostics


class LSPTools:
    """Deterministic pseudo-LSP tooling for definitions, references, diagnostics."""

    def __init__(self, root_dir: str = "."):
        self.root_path = os.path.abspath(root_dir)
        self._diagnostics = LSPDiagnosticsCache(self.root_path)

    def _iter_python_files(self, path: Optional[str] = None) -> List[str]:
        root = self.root_path
        if path:
            candidate = os.path.abspath(
                path if os.path.isabs(path) else os.path.join(self.root_path, path)
            )
            if os.path.isfile(candidate):
                return [candidate] if candidate.endswith(".py") else []
            if os.path.isdir(candidate):
                root = candidate

        files: List[str] = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                d
                for d in dirnames
                if d not in {".git", "venv", "__pycache__", ".saguaro", ".anvil"}
            ]
            for filename in filenames:
                if filename.endswith(".py"):
                    files.append(os.path.join(dirpath, filename))
        return files

    def get_definition(
        self,
        symbol: Optional[str] = None,
        file_path: Optional[str] = None,
        line: Optional[int] = None,
        col: Optional[int] = None,
    ) -> str:
        if symbol:
            matches: List[str] = []
            for candidate in self._iter_python_files():
                try:
                    source = Path(candidate).read_text(encoding="utf-8", errors="ignore")
                    tree = ast.parse(source)
                except Exception:
                    continue
                for node in ast.walk(tree):
                    if isinstance(
                        node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                    ) and node.name == symbol:
                        rel = os.path.relpath(candidate, self.root_path)
                        matches.append(f"{rel}:{node.lineno}")
                if matches:
                    break
            if not matches:
                return f"No definition found for symbol '{symbol}'."
            return "\n".join(matches)

        if file_path and line is not None and col is not None:
            return f"Definition lookup by position is not implemented ({file_path}:{line}:{col})."
        return "Provide `symbol` for lsp_definition."

    def get_references(
        self,
        symbol: Optional[str] = None,
        file_path: Optional[str] = None,
        line: Optional[int] = None,
        col: Optional[int] = None,
    ) -> List[str]:
        if symbol:
            try:
                result = subprocess.run(
                    ["rg", "-n", "-F", symbol, self.root_path, "-g", "*.py"],
                    cwd=self.root_path,
                    capture_output=True,
                    text=True,
                    timeout=20,
                    check=False,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired):
                result = None

            if result and result.stdout.strip():
                return result.stdout.strip().splitlines()[:200]
            return []

        if file_path and line is not None and col is not None:
            return [f"Reference lookup by position is not implemented: {file_path}:{line}:{col}"]
        return []

    def get_diagnostics(
        self, file_path: Optional[str] = None, path: Optional[str] = None
    ) -> Any:
        target = path or file_path
        if target:
            if os.path.isdir(target) or os.path.isdir(
                os.path.join(self.root_path, target)
            ):
                all_diags: List[Dict[str, Any]] = []
                for candidate in self._iter_python_files(target):
                    rel = os.path.relpath(candidate, self.root_path)
                    all_diags.extend(self._diagnostics.diagnostics_for_file(rel))
                return json.dumps(all_diags, indent=2, sort_keys=True)
            return json.dumps(
                self._diagnostics.diagnostics_for_file(target),
                indent=2,
                sort_keys=True,
            )

        all_diags: List[Dict[str, Any]] = []
        for candidate in self._iter_python_files():
            rel = os.path.relpath(candidate, self.root_path)
            all_diags.extend(self._diagnostics.diagnostics_for_file(rel))
            if len(all_diags) > 2000:
                break
        if not all_diags:
            return "No issues found"
        return f"Found {len(all_diags)} diagnostics"


lsp_client = LSPTools(".")


def lsp_get_definition(
    symbol: Optional[str] = None,
    file_path: Optional[str] = None,
    line: Optional[int] = None,
    col: Optional[int] = None,
    **_: Any,
):
    return lsp_client.get_definition(
        symbol=symbol, file_path=file_path, line=line, col=col
    )


def lsp_find_references(
    symbol: Optional[str] = None,
    file_path: Optional[str] = None,
    line: Optional[int] = None,
    col: Optional[int] = None,
    **_: Any,
):
    return lsp_client.get_references(
        symbol=symbol, file_path=file_path, line=line, col=col
    )


def lsp_get_diagnostics(
    path: Optional[str] = None,
    file_path: Optional[str] = None,
    **_: Any,
):
    return lsp_client.get_diagnostics(path=path, file_path=file_path)
