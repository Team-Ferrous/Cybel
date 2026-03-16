import ast
import os
import re
from typing import Any, Dict, List, Optional, Set

from saguaro.parsing.parser import SAGUAROParser
from saguaro.refactor.planner import RefactorPlanner


class EvidenceBuilder:
    """Build grounded evidence with direct reads, parser metadata, and dependency graphs."""

    def __init__(self, saguaro_tools, registry, console, repo_root: str = "."):
        self.saguaro = saguaro_tools
        self.registry = registry
        self.console = console
        self.repo_root = os.path.abspath(repo_root)
        self.parser = SAGUAROParser()
        self.refactor_planner = RefactorPlanner(self.repo_root)
        self._session_file_cache: Dict[str, Dict[str, Any]] = {}

    def build(self, query: str, target_file: Optional[str] = None) -> Dict[str, Any]:
        primary_file = self._normalize_file_path(target_file) if target_file else None
        evidence: Dict[str, Any] = {
            "primary_file": primary_file,
            "codebase_files": [],
            "file_contents": {},
            "skeletons": {},
            "workspace_map": {},
            "imports": {},
            "entities": {},
            "integration_map": {},
            "dependency_graph": {"edges": {}, "reverse_edges": {}},
            "tree_views": {},
            "validation": {},
        }

        self._pass_discovery(query, evidence)
        self._pass_workspace_map(evidence)
        self._pass_structure(evidence)
        self._pass_full_content(evidence)
        self._pass_integration_mapping(evidence)
        self._pass_validation(evidence)
        evidence["codebase_files"] = sorted(
            set(evidence["codebase_files"])
            | set(evidence["file_contents"].keys())
            | set(evidence["skeletons"].keys())
        )
        return evidence

    def _pass_discovery(self, query: str, evidence: Dict[str, Any]) -> None:
        files: Set[str] = set()
        if evidence["primary_file"]:
            files.add(evidence["primary_file"])

        api = getattr(self.saguaro, "substrate", None)
        api = getattr(api, "_api", None)
        if api is None:
            raise RuntimeError(
                "SAGUARO_STRICT_UNAVAILABLE: substrate API missing in EvidenceBuilder."
            )

        query_errors: List[str] = []

        for semantic_query in self._decompose_query(query, evidence["primary_file"]):
            try:
                result = api.query(semantic_query, k=10)
            except Exception as exc:
                query_errors.append(f"{semantic_query}: {exc}")
                continue

            for item in result.get("results", []):
                file_path = self._normalize_file_path(item.get("file"))
                if not file_path:
                    continue
                files.add(file_path)
                entity_name = item.get("name")
                if entity_name:
                    evidence["entities"][entity_name] = {
                        "file": file_path,
                        "line": item.get("line"),
                        "type": item.get("type"),
                        "score": item.get("score"),
                    }

        min_required = 1 if evidence["primary_file"] else 3
        if len(files) < min_required:
            detail = f"; query_errors={query_errors[:3]}" if query_errors else ""
            raise RuntimeError(
                "SAGUARO_STRICT_FALLBACK_DISABLED: semantic discovery returned "
                f"insufficient files ({len(files)} < {min_required}).{detail}"
            )

        evidence["codebase_files"] = sorted(files)

    def _pass_structure(self, evidence: Dict[str, Any]) -> None:
        api = getattr(self.saguaro.substrate, "_api", None)
        if api is None:
            raise RuntimeError(
                "SAGUARO_STRICT_UNAVAILABLE: substrate API missing in EvidenceBuilder."
            )

        for file_path in evidence["codebase_files"]:
            try:
                skeleton = api.skeleton(file_path)
            except Exception:
                continue

            formatted = self._format_skeleton(skeleton)
            if formatted:
                evidence["skeletons"][file_path] = formatted

            abs_path = os.path.join(self.repo_root, file_path)
            tree_entities = self.parser.parse_file(abs_path)
            if tree_entities:
                evidence["tree_views"][file_path] = [
                    {
                        "name": entity.name,
                        "type": entity.type,
                        "start_line": entity.start_line,
                        "end_line": entity.end_line,
                    }
                    for entity in tree_entities
                ]

                for entity in tree_entities:
                    if entity.type in {"class", "function", "method", "file"}:
                        evidence["entities"].setdefault(
                            entity.name,
                            {
                                "file": file_path,
                                "line": entity.start_line,
                                "type": entity.type,
                                "score": None,
                            },
                        )

    def _pass_workspace_map(self, evidence: Dict[str, Any]) -> None:
        api = getattr(self.saguaro.substrate, "_api", None)
        if api is None:
            raise RuntimeError(
                "SAGUARO_STRICT_UNAVAILABLE: substrate API missing in EvidenceBuilder."
            )

        try:
            top_listing = api.list_directory(".", recursive=False)
            recursive_listing = api.list_directory(
                ".",
                recursive=True,
                extensions=[".py", ".cc", ".cpp", ".h", ".hpp", ".md", ".json"],
            )
        except Exception as exc:
            raise RuntimeError(
                f"SAGUARO_STRICT_WORKSPACE_MAP_FAILED: {exc}"
            ) from exc

        top_entries = top_listing.get("entries", []) if isinstance(top_listing, dict) else []
        recursive_entries = (
            recursive_listing.get("entries", []) if isinstance(recursive_listing, dict) else []
        )

        top_dirs = sorted(
            entry.get("path", "")
            for entry in top_entries
            if entry.get("type") == "directory"
        )
        code_files = [
            entry.get("path", "")
            for entry in recursive_entries
            if entry.get("type") == "file"
        ]
        key_paths = []
        for path in code_files:
            if any(
                marker in path
                for marker in (
                    "main.py",
                    "cli/repl.py",
                    "core/agent.py",
                    "core/unified_chat_loop.py",
                    "core/ollama_client.py",
                    "core/agents/subagent.py",
                    "saguaro/api.py",
                    "config/settings.py",
                )
            ):
                key_paths.append(path)
        key_paths = sorted(dict.fromkeys(key_paths))[:20]

        snippet_lines: List[str] = []
        for entry in top_entries[:40]:
            entry_type = entry.get("type", "?")
            path = entry.get("path", "")
            if not path:
                continue
            prefix = "dir " if entry_type == "directory" else "file"
            snippet_lines.append(f"- {prefix}: {path}")

        evidence["workspace_map"] = {
            "top_dirs": top_dirs[:20],
            "key_paths": key_paths,
            "tree_snippet": "\n".join(snippet_lines[:40]),
        }

    def _pass_full_content(self, evidence: Dict[str, Any]) -> None:
        api = getattr(self.saguaro.substrate, "_api", None)
        if api is None:
            raise RuntimeError(
                "SAGUARO_STRICT_UNAVAILABLE: substrate API missing in EvidenceBuilder."
            )

        prioritized_files: List[str] = []
        if evidence["primary_file"]:
            prioritized_files.append(evidence["primary_file"])

        for file_path in evidence["codebase_files"]:
            if file_path not in prioritized_files:
                prioritized_files.append(file_path)

        for file_path in prioritized_files[:15]:
            line_count = self._line_count(file_path)
            if line_count > 500:
                slices = []
                for entity in (evidence.get("tree_views", {}).get(file_path, []) or [])[:4]:
                    entity_name = entity.get("name")
                    if not entity_name:
                        continue
                    try:
                        snippet = self.saguaro.slice(f"{file_path}.{entity_name}")
                    except Exception:
                        snippet = ""
                    if isinstance(snippet, str) and snippet and not snippet.startswith("Error"):
                        slices.append(f"[SLICE {entity_name}]\n{snippet}")
                if slices:
                    evidence["file_contents"][file_path] = "\n\n".join(slices)
                    continue
                raise RuntimeError(
                    "SAGUARO_STRICT_SLICE_REQUIRED: refusing full-file fallback read for "
                    f"large file {file_path}."
                )

            content = self._read_file_cached(file_path)
            if content:
                evidence["file_contents"][file_path] = content

    def _pass_integration_mapping(self, evidence: Dict[str, Any]) -> None:
        resolved_files: List[str] = []
        for file_path, content in evidence["file_contents"].items():
            imports = self._extract_imports(content)
            evidence["imports"][file_path] = imports
            for imported in imports:
                evidence["integration_map"].setdefault(imported, []).append(file_path)
            resolved_files.append(os.path.join(self.repo_root, file_path))

        if not resolved_files:
            return

        graph = self.refactor_planner._build_dependency_graph(resolved_files)
        evidence["dependency_graph"] = {
            "edges": {
                self._relpath(src): sorted(self._relpath(dst) for dst in targets)
                for src, targets in graph.edges.items()
            },
            "reverse_edges": {
                self._relpath(dst): sorted(self._relpath(src) for src in sources)
                for dst, sources in graph.reverse_edges.items()
            },
        }

    def _pass_validation(self, evidence: Dict[str, Any]) -> None:
        for entity_name, entity_info in evidence["entities"].items():
            file_path = entity_info.get("file")
            content = evidence["file_contents"].get(file_path, "")
            evidence["validation"][entity_name] = bool(content) and bool(
                re.search(rf"\b{re.escape(entity_name)}\b", content)
            )

    def _decompose_query(
        self, query: str, primary_file: Optional[str] = None
    ) -> List[str]:
        queries = [query.strip()]
        if primary_file:
            base = os.path.splitext(os.path.basename(primary_file))[0]
            queries.append(base)
            queries.append(base.replace("_", " "))

        file_refs = re.findall(r"[\w./-]+\.(?:py|cc|cpp|h|js|ts|md)", query)
        for ref in file_refs:
            base = os.path.splitext(os.path.basename(ref))[0]
            queries.append(base)
            queries.append(base.replace("_", " "))

        tech_terms = re.findall(r"\b[A-Z][A-Za-z0-9_]+\b|\b[a-z]+_[a-z0-9_]+\b", query)
        queries.extend(tech_terms[:5])

        seen: Set[str] = set()
        ordered: List[str] = []
        for item in queries:
            cleaned = " ".join(item.split()).strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                ordered.append(cleaned)
        return ordered

    def _extract_imports(self, content: str) -> List[str]:
        imports: Set[str] = set()
        try:
            tree = ast.parse(content)
        except SyntaxError:
            tree = None

        if tree is not None:
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.add(node.module)
        else:
            for match in re.finditer(r"from\s+([\w.]+)\s+import", content):
                imports.add(match.group(1))
            for match in re.finditer(r"import\s+([\w.]+)", content):
                imports.add(match.group(1))

        return sorted(imports)

    def _format_skeleton(self, skeleton: Dict[str, Any]) -> str:
        if not isinstance(skeleton, dict):
            return ""

        lines = [
            f"File: {skeleton.get('file_path', '')}",
            f"Language: {skeleton.get('language', '')}",
        ]
        for symbol in skeleton.get("symbols", []):
            name = symbol.get("name", "<unknown>")
            symbol_type = symbol.get("type", "symbol")
            line = symbol.get("line_start")
            if line:
                lines.append(f"- {symbol_type} {name} (line {line})")
            else:
                lines.append(f"- {symbol_type} {name}")
        return "\n".join(lines)

    def _normalize_file_path(self, path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        if os.path.isabs(path):
            try:
                return os.path.relpath(path, self.repo_root)
            except ValueError:
                return None
        candidate = os.path.abspath(os.path.join(self.repo_root, path))
        if os.path.exists(candidate):
            return os.path.relpath(candidate, self.repo_root)
        return path

    def _relpath(self, path: str) -> str:
        if os.path.isabs(path):
            return os.path.relpath(path, self.repo_root)
        return path

    def _extract_files_from_grep(self, grep_output: Any) -> List[str]:
        if not isinstance(grep_output, str):
            return []
        files: List[str] = []
        for line in grep_output.splitlines():
            match = re.match(r"([^:\n]+):\d+:", line.strip())
            if not match:
                continue
            normalized = self._normalize_file_path(match.group(1))
            if normalized:
                files.append(normalized)
        return files

    def _read_file_cached(self, file_path: str) -> Optional[str]:
        abs_path = os.path.join(self.repo_root, file_path)
        try:
            mtime = os.path.getmtime(abs_path)
        except Exception:
            mtime = None

        cached = self._session_file_cache.get(file_path)
        if cached and cached.get("mtime") == mtime:
            return cached.get("content")

        content = self.registry.dispatch("read_file", {"path": file_path})
        if isinstance(content, str) and not content.startswith("Error"):
            self._session_file_cache[file_path] = {"mtime": mtime, "content": content}
            return content
        return None

    def _line_count(self, file_path: str) -> int:
        abs_path = os.path.join(self.repo_root, file_path)
        try:
            with open(abs_path, "r", encoding="utf-8", errors="ignore") as handle:
                return sum(1 for _ in handle)
        except Exception:
            return 0
