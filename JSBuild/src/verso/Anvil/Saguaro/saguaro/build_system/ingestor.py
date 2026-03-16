"""Structured build topology ingestion for Saguaro."""

from __future__ import annotations

import json
import os
import re
import shlex
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from saguaro.utils.file_utils import build_corpus_manifest

_IGNORE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
}
_CMAKE_TARGET_RE = re.compile(r"CMakeFiles/(?P<name>[^/]+)\.dir/")
_TARGET_CMD_RE = re.compile(
    r"(?P<kind>add_library|add_executable)\s*\((?P<body>.*?)\)",
    re.DOTALL | re.IGNORECASE,
)
_LINK_CMD_RE = re.compile(
    r"target_link_libraries\s*\((?P<body>.*?)\)",
    re.DOTALL | re.IGNORECASE,
)
_HEADER_EXTS = {".h", ".hh", ".hpp", ".hxx"}
_SOURCE_EXTS = {".c", ".cc", ".cpp", ".cxx", ".m", ".mm", ".cu"}
_TARGET_KEYWORDS = {"STATIC", "SHARED", "MODULE", "OBJECT", "INTERFACE", "ALIAS", "EXCLUDE_FROM_ALL"}
_VISIBILITY_KEYWORDS = {"PRIVATE", "PUBLIC", "INTERFACE"}


@dataclass(slots=True)
class BuildTarget:
    """Structured build target description."""

    name: str
    type: str
    file: str
    dependencies: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    includes: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    language: str = "unknown"
    generators: list[str] = field(default_factory=list)


class BuildGraphIngestor:
    """Ingest build topology from structured build-system metadata."""

    def __init__(self, root_dir: str) -> None:
        self.root_dir = os.path.abspath(root_dir)
        self.targets: dict[str, BuildTarget] = {}
        self._stats = {
            "compile_databases": 0,
            "cmake_file_api_replies": 0,
            "target_directories": 0,
        }
        self._compiled_sources: set[str] = set()
        self._manifest = build_corpus_manifest(self.root_dir)
        self._files_by_name = self._index_manifest_files()

    def ingest(self) -> dict[str, Any]:
        """Scan repository build metadata and return a normalized graph."""
        self._scan_python()
        self._scan_node()
        self._scan_compile_databases()
        self._scan_cmake_file_api()
        self._scan_target_directories()
        self._scan_cmake()
        self._scan_makefile()

        target_sources = {
            source
            for target in self.targets.values()
            for source in target.sources
            if source and Path(source).suffix.lower() in (_SOURCE_EXTS | _HEADER_EXTS)
        }
        compiled_source_count = len(self._compiled_sources)
        structured_source_count = len(target_sources)
        coverage_basis = compiled_source_count or structured_source_count
        coverage_percent = (
            round((structured_source_count / coverage_basis) * 100.0, 1)
            if coverage_basis
            else 0.0
        )

        return {
            "root": self.root_dir,
            "target_count": len(self.targets),
            "targets": {k: self._target_to_dict(v) for k, v in sorted(self.targets.items())},
            "structured_inputs": dict(self._stats),
            "source_coverage": {
                "compiled_sources": compiled_source_count,
                "owned_sources": structured_source_count,
                "coverage_percent": coverage_percent,
            },
        }

    def _target_to_dict(self, target: BuildTarget) -> dict[str, Any]:
        return {
            "type": target.type,
            "file": self._relpath(target.file),
            "deps": sorted(dict.fromkeys(target.dependencies)),
            "sources": [self._relpath(path) for path in target.sources],
            "includes": [self._relpath(path) for path in target.includes],
            "artifacts": [self._relpath(path) for path in target.artifacts],
            "language": target.language,
            "generators": sorted(dict.fromkeys(target.generators)),
        }

    def _upsert_target(
        self,
        key: str,
        *,
        name: str,
        target_type: str,
        file_path: str,
        dependencies: list[str] | None = None,
        sources: list[str] | None = None,
        includes: list[str] | None = None,
        artifacts: list[str] | None = None,
        language: str = "unknown",
        generator: str | None = None,
    ) -> None:
        target = self.targets.get(key)
        if target is None:
            target = BuildTarget(
                name=name,
                type=target_type,
                file=os.path.abspath(file_path),
                language=language,
            )
            self.targets[key] = target
        else:
            if target.type == "unknown" and target_type != "unknown":
                target.type = target_type
            if target.language == "unknown" and language != "unknown":
                target.language = language
            if target.file.endswith("TargetDirectories.txt") and not file_path.endswith(
                "TargetDirectories.txt"
            ):
                target.file = os.path.abspath(file_path)

        for attr_name, values in (
            ("dependencies", dependencies or []),
            ("sources", sources or []),
            ("includes", includes or []),
            ("artifacts", artifacts or []),
        ):
            current = getattr(target, attr_name)
            for item in values:
                normalized = (
                    str(item).strip()
                    if attr_name == "dependencies"
                    else self._normalize_path(item)
                )
                if normalized and normalized not in current:
                    current.append(normalized)

        if generator and generator not in target.generators:
            target.generators.append(generator)

    def _scan_python(self) -> None:
        for path in self._files_by_name.get("pyproject.toml", []):
            self._parse_pyproject(Path(path))
        for path in self._files_by_name.get("setup.py", []):
            self._parse_setup_py(path)

    def _parse_pyproject(self, path: Path) -> None:
        try:
            payload = tomllib.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return

        project = payload.get("project") if isinstance(payload, dict) else {}
        poetry = (((payload.get("tool") or {}).get("poetry")) if isinstance(payload, dict) else {}) or {}
        name = str(
            (project or {}).get("name")
            or (poetry or {}).get("name")
            or path.parent.name
            or "python_project"
        ).strip()
        deps = []
        if isinstance(project, dict):
            deps.extend([str(item) for item in project.get("dependencies", [])])
        if isinstance(poetry, dict):
            deps.extend(
                key
                for key in (poetry.get("dependencies") or {}).keys()
                if key != "python"
            )
        self._upsert_target(
            f"py:{name}",
            name=name,
            target_type="lib",
            file_path=str(path),
            dependencies=deps,
            sources=[str(path)],
            language="python",
            generator="pyproject",
        )

    def _parse_setup_py(self, path: str) -> None:
        try:
            content = Path(path).read_text(encoding="utf-8")
        except Exception:
            return

        name_match = re.search(r'name=["\']([^"\']+)["\']', content)
        name = name_match.group(1) if name_match else Path(path).parent.name or "unknown_python"
        deps = re.findall(r'[\'"]([a-zA-Z0-9_\-]+)[<>=]', content)
        self._upsert_target(
            f"py:{name}",
            name=name,
            target_type="lib",
            file_path=path,
            dependencies=deps,
            sources=[path],
            language="python",
            generator="setup.py",
        )

    def _scan_node(self) -> None:
        for item in self._files_by_name.get("package.json", []):
            path = Path(item)
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            dependencies = payload.get("dependencies")
            dev_dependencies = payload.get("devDependencies")
            deps = list(dependencies.keys()) if isinstance(dependencies, dict) else []
            if isinstance(dev_dependencies, dict):
                deps.extend(dev_dependencies.keys())
            name = str(payload.get("name") or path.parent.name or "unknown_node")
            self._upsert_target(
                f"npm:{name}",
                name=name,
                target_type="lib",
                file_path=str(path),
                dependencies=[str(item) for item in deps],
                sources=[str(path)],
                language="javascript",
                generator="package.json",
            )

    def _scan_compile_databases(self) -> None:
        for path in self._find_files("compile_commands.json"):
            self._stats["compile_databases"] += 1
            self._parse_compile_database(path)

    def _parse_compile_database(self, path: str) -> None:
        try:
            entries = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(entries, list):
            return

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            abs_file = self._normalize_path(
                os.path.join(str(entry.get("directory") or ""), str(entry.get("file") or ""))
            )
            if not abs_file:
                continue
            self._compiled_sources.add(abs_file)
            target_name = self._target_name_from_compile_entry(entry) or Path(abs_file).stem
            key = f"cmake:{target_name}"
            command_tokens = self._compile_tokens(entry)
            includes = self._extract_include_paths(command_tokens, directory=str(entry.get("directory") or ""))
            output_path = self._normalize_path(
                os.path.join(str(entry.get("directory") or ""), str(entry.get("output") or ""))
            )
            self._upsert_target(
                key,
                name=target_name,
                target_type=self._infer_target_type(target_name),
                file_path=path,
                sources=[abs_file],
                includes=includes,
                artifacts=[output_path] if output_path else [],
                language=self._language_for_path(abs_file),
                generator="compile_commands",
            )

    def _scan_cmake_file_api(self) -> None:
        for index_path in self._find_file_api_indexes():
            self._stats["cmake_file_api_replies"] += 1
            self._parse_cmake_file_api(index_path)

    def _find_file_api_indexes(self) -> list[str]:
        return sorted(
            path
            for path in self._manifest.files
            if "/.cmake/api/v1/reply/" in path.replace("\\", "/")
            and os.path.basename(path).startswith("index-")
            and path.endswith(".json")
        )

    def _parse_cmake_file_api(self, index_path: str) -> None:
        reply_dir = os.path.dirname(index_path)
        try:
            index = json.loads(Path(index_path).read_text(encoding="utf-8"))
        except Exception:
            return

        objects = index.get("objects") if isinstance(index, dict) else None
        if not isinstance(objects, list):
            return

        for item in objects:
            if not isinstance(item, dict):
                continue
            if str(item.get("kind") or "") != "codemodel":
                continue
            json_file = str(item.get("jsonFile") or "")
            if not json_file:
                continue
            self._parse_codemodel(os.path.join(reply_dir, json_file))

    def _parse_codemodel(self, codemodel_path: str) -> None:
        try:
            payload = json.loads(Path(codemodel_path).read_text(encoding="utf-8"))
        except Exception:
            return

        configurations = payload.get("configurations") if isinstance(payload, dict) else None
        if not isinstance(configurations, list):
            return

        reply_dir = os.path.dirname(codemodel_path)
        for config in configurations:
            if not isinstance(config, dict):
                continue
            target_refs = config.get("targets") or []
            targets_by_id = {
                str(ref.get("id") or ""): str(ref.get("name") or "")
                for ref in target_refs
                if isinstance(ref, dict)
            }
            for ref in target_refs:
                if not isinstance(ref, dict):
                    continue
                ref_name = str(ref.get("name") or "")
                json_file = str(ref.get("jsonFile") or "")
                if not ref_name or not json_file:
                    continue
                try:
                    target = json.loads(
                        Path(os.path.join(reply_dir, json_file)).read_text(encoding="utf-8")
                    )
                except Exception:
                    continue
                if not isinstance(target, dict):
                    continue
                sources = [
                    self._normalize_path(str(item.get("path") or ""))
                    for item in (target.get("sources") or [])
                    if isinstance(item, dict) and str(item.get("path") or "")
                ]
                includes: list[str] = []
                for group in target.get("compileGroups") or []:
                    if not isinstance(group, dict):
                        continue
                    for include in group.get("includes") or []:
                        if not isinstance(include, dict):
                            continue
                        path = self._normalize_path(str(include.get("path") or ""))
                        if path:
                            includes.append(path)
                dependencies = []
                for dep in target.get("dependencies") or []:
                    if not isinstance(dep, dict):
                        continue
                    dep_name = str(dep.get("id") or "")
                    dependencies.append(targets_by_id.get(dep_name, dep_name))
                artifacts = [
                    self._normalize_path(str(item.get("path") or ""))
                    for item in (target.get("artifacts") or [])
                    if isinstance(item, dict) and str(item.get("path") or "")
                ]
                self._upsert_target(
                    f"cmake:{ref_name}",
                    name=ref_name,
                    target_type=self._map_cmake_type(str(target.get("type") or "")),
                    file_path=codemodel_path,
                    dependencies=[dep for dep in dependencies if dep],
                    sources=[item for item in sources if item],
                    includes=[item for item in includes if item],
                    artifacts=[item for item in artifacts if item],
                    language=self._language_for_sources(sources),
                    generator="cmake_file_api",
                )

    def _scan_target_directories(self) -> None:
        for path in self._find_files("TargetDirectories.txt"):
            self._stats["target_directories"] += 1
            try:
                lines = Path(path).read_text(encoding="utf-8").splitlines()
            except Exception:
                continue
            for line in lines:
                target_name = self._target_name_from_target_dir(line)
                if not target_name:
                    continue
                self._upsert_target(
                    f"cmake:{target_name}",
                    name=target_name,
                    target_type=self._infer_target_type(target_name),
                    file_path=path,
                    language="native",
                    generator="target_directories",
                )

    def _scan_cmake(self) -> None:
        for path in self._find_files("CMakeLists.txt"):
            self._parse_cmake(path)

    def _parse_cmake(self, path: str) -> None:
        try:
            content = Path(path).read_text(encoding="utf-8")
        except Exception:
            return

        link_map: dict[str, list[str]] = {}
        for match in _LINK_CMD_RE.finditer(content):
            tokens = self._cmake_tokens(match.group("body"))
            if not tokens:
                continue
            target_name = tokens[0]
            deps = [
                token
                for token in tokens[1:]
                if token not in _VISIBILITY_KEYWORDS and not token.startswith("$<")
            ]
            link_map.setdefault(target_name, []).extend(deps)

        for match in _TARGET_CMD_RE.finditer(content):
            kind = str(match.group("kind") or "").lower()
            tokens = self._cmake_tokens(match.group("body"))
            if not tokens:
                continue
            target_name = tokens[0]
            args = [
                token
                for token in tokens[1:]
                if token not in _TARGET_KEYWORDS and token not in _VISIBILITY_KEYWORDS
            ]
            sources = [
                self._normalize_path(os.path.join(os.path.dirname(path), token))
                for token in args
                if self._looks_like_source_path(token)
            ]
            self._upsert_target(
                f"cmake:{target_name}",
                name=target_name,
                target_type="lib" if kind == "add_library" else "bin",
                file_path=path,
                dependencies=link_map.get(target_name, []),
                sources=[item for item in sources if item],
                language=self._language_for_sources(sources),
                generator="cmake_lists",
            )

    def _scan_makefile(self) -> None:
        # Placeholder for future structured make ingestion.
        return

    def _find_files(self, filename: str) -> list[str]:
        return sorted(self._files_by_name.get(filename, []))

    def _index_manifest_files(self) -> dict[str, list[str]]:
        indexed: dict[str, list[str]] = {}
        for path in self._manifest.files:
            indexed.setdefault(os.path.basename(path), []).append(path)
        return indexed

    def _normalize_path(self, path: str) -> str:
        raw = str(path or "").strip()
        if not raw:
            return ""
        expanded = os.path.abspath(raw if os.path.isabs(raw) else os.path.join(self.root_dir, raw))
        return expanded.replace("\\", "/")

    def _relpath(self, path: str) -> str:
        normalized = self._normalize_path(path)
        if not normalized:
            return ""
        try:
            rel = os.path.relpath(normalized, self.root_dir)
        except ValueError:
            return normalized
        return rel.replace("\\", "/") if not rel.startswith("..") else normalized

    @staticmethod
    def _compile_tokens(entry: dict[str, Any]) -> list[str]:
        arguments = entry.get("arguments")
        if isinstance(arguments, list):
            return [str(item) for item in arguments]
        command = str(entry.get("command") or "").strip()
        if not command:
            return []
        try:
            return shlex.split(command)
        except ValueError:
            return command.split()

    def _extract_include_paths(self, tokens: list[str], *, directory: str) -> list[str]:
        includes: list[str] = []
        idx = 0
        while idx < len(tokens):
            token = tokens[idx]
            candidate = ""
            if token in {"-I", "-isystem"} and idx + 1 < len(tokens):
                candidate = tokens[idx + 1]
                idx += 2
            elif token.startswith("-I") and len(token) > 2:
                candidate = token[2:]
                idx += 1
            elif token.startswith("-isystem") and len(token) > len("-isystem"):
                candidate = token[len("-isystem") :]
                idx += 1
            else:
                idx += 1
            if not candidate:
                continue
            normalized = self._normalize_path(
                candidate if os.path.isabs(candidate) else os.path.join(directory, candidate)
            )
            if normalized:
                includes.append(normalized)
        return includes

    @staticmethod
    def _target_name_from_compile_entry(entry: dict[str, Any]) -> str:
        for raw in (entry.get("output"), entry.get("command")):
            text = str(raw or "")
            match = _CMAKE_TARGET_RE.search(text)
            if match:
                return str(match.group("name") or "")
        return ""

    @staticmethod
    def _target_name_from_target_dir(path: str) -> str:
        match = _CMAKE_TARGET_RE.search(str(path or "").replace("\\", "/"))
        return str(match.group("name") or "") if match else ""

    @staticmethod
    def _map_cmake_type(raw_type: str) -> str:
        item = str(raw_type or "").upper()
        if "EXECUTABLE" in item:
            return "bin"
        if "TEST" in item:
            return "test"
        if "LIBRARY" in item:
            return "lib"
        if "UTILITY" in item:
            return "utility"
        return "unknown"

    @staticmethod
    def _infer_target_type(name: str) -> str:
        lowered = str(name or "").lower()
        if "test" in lowered:
            return "test"
        if "tool" in lowered or "main" in lowered:
            return "bin"
        return "lib"

    @staticmethod
    def _language_for_path(path: str) -> str:
        suffix = Path(path).suffix.lower()
        if suffix == ".py":
            return "python"
        if suffix in {".js", ".ts", ".tsx", ".jsx"}:
            return "javascript"
        if suffix in (_SOURCE_EXTS | _HEADER_EXTS):
            return "native"
        return "unknown"

    def _language_for_sources(self, sources: list[str]) -> str:
        languages = {self._language_for_path(path) for path in sources if path}
        if "native" in languages:
            return "native"
        if "python" in languages:
            return "python"
        if "javascript" in languages:
            return "javascript"
        return "unknown"

    @staticmethod
    def _cmake_tokens(body: str) -> list[str]:
        collapsed = re.sub(r"#.*", "", str(body or ""))
        return [token for token in re.split(r"[\s\r\n]+", collapsed) if token]

    @staticmethod
    def _looks_like_source_path(token: str) -> bool:
        stripped = str(token or "").strip().strip("\"'")
        if not stripped or stripped.startswith("$<") or stripped.startswith("${"):
            return False
        suffix = Path(stripped).suffix.lower()
        return suffix in (_SOURCE_EXTS | _HEADER_EXTS)
