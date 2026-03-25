"""SAGUARO Parsing Module
Wraps tree-sitter to extract semantically enhanced entities from source code.
"""

import ast
import hashlib
import json
import logging
import os
import re
import warnings
from typing import Any

logger = logging.getLogger(__name__)

try:
    import importlib.util

    TREE_SITTER_AVAILABLE = importlib.util.find_spec("tree_sitter") is not None
    if TREE_SITTER_AVAILABLE:
        from tree_sitter import Language, Parser  # noqa: F401
except ImportError:
    TREE_SITTER_AVAILABLE = False
    logger.warning(
        "tree-sitter not installed. AST parsing is disabled; lightweight backends remain available."
    )


class CodeEntity:
    """Represent a parsed code entity."""

    def __init__(
        self,
        name: str,
        type: str,
        content: str,
        start_line: int,
        end_line: int,
        file_path: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the entity metadata."""
        self.name = name
        self.type = type
        self.content = content
        self.start_line = start_line
        self.end_line = end_line
        self.file_path = file_path
        self.metadata = dict(metadata or {})

    def __repr__(self) -> str:
        return f"<CodeEntity {self.name} ({self.type})>"


class SAGUAROParser:
    """Parse source files into code entities."""

    _LANGUAGE_BY_FILENAME = {
        "cmakelists.txt": "cmake",
        "makefile": "make",
        "dockerfile": "docker",
        "build": "bazel",
        "build.bazel": "bazel",
        "workspace": "bazel",
        "workspace.bazel": "bazel",
        "jenkinsfile": "java",
        "meson.build": "cmake",
        "meson_options.txt": "cmake",
        "rakefile": "ruby",
        "gemfile": "ruby",
        "vagrantfile": "ruby",
        "procfile": "shell",
        "justfile": "make",
        ".env": "config",
        ".env.local": "config",
        ".env.development": "config",
        ".env.production": "config",
        ".editorconfig": "config",
    }
    _LIGHTWEIGHT_LANGUAGES = {
        "actionscript",
        "ada",
        "aql",
        "asciidoc",
        "assembly",
        "awk",
        "bazel",
        "c",
        "clojure",
        "cobol",
        "coldfusion",
        "config",
        "cpp",
        "crystal",
        "csv",
        "css",
        "cypher",
        "dlang",
        "django_template",
        "docker",
        "ejs",
        "elixir",
        "emacs_lisp",
        "erlang",
        "flatbuffers",
        "fortran",
        "fsharp",
        "gdscript",
        "graphql",
        "hack",
        "handlebars",
        "haskell",
        "hcl",
        "hdl",
        "html",
        "jinja",
        "json",
        "julia",
        "latex",
        "lisp",
        "lua",
        "make",
        "matlab",
        "mdx",
        "mql",
        "nim",
        "nix",
        "objective_c",
        "ocaml",
        "octave",
        "org",
        "pascal",
        "perl",
        "php",
        "pli",
        "powershell",
        "prolog",
        "proto",
        "pug",
        "puppet",
        "qml",
        "r",
        "rpg",
        "rst",
        "ruby",
        "rust",
        "scala",
        "sed",
        "shell",
        "smalltalk",
        "solidity",
        "sql",
        "swift",
        "tcl",
        "thrift",
        "toml",
        "typst",
        "unrealscript",
        "vim",
        "vlang",
        "wat",
        "wolfram",
        "xml",
        "yaml",
        "zig",
        "cmake",
        "java",
        "csharp",
        "kotlin",
    }
    _TREE_SITTER_BACKENDS = {
        "javascript": {
            "ts_language": "javascript",
            "query": (
            "(function_declaration name: (identifier) @fn.name) @fn.def "
            "(class_declaration name: (identifier) @class.name) @class.def "
            "(variable_declarator name: (identifier) @fn.name value: (arrow_function) @fn.value) @fn.def "
            "(variable_declarator name: (identifier) @fn.name value: (function_expression) @fn.value) @fn.def"
            ),
        },
        "typescript": {
            "ts_language": "typescript",
            "query": (
            "(function_declaration name: (identifier) @fn.name) @fn.def "
            "(class_declaration name: (type_identifier) @class.name) @class.def "
            "(variable_declarator name: (identifier) @fn.name value: (arrow_function) @fn.value) @fn.def "
            "(variable_declarator name: (identifier) @fn.name value: (function_expression) @fn.value) @fn.def"
            ),
            "tsx_query": (
                "(export_statement declaration: (function_declaration name: (identifier) @function.name) @function.def) "
                "(function_declaration name: (identifier) @function.name) @function.def "
                "(class_declaration name: (type_identifier) @class.name) @class.def "
                "(variable_declarator name: (identifier) @function.name value: (arrow_function) @function.value) @function.def"
            ),
        },
        "c": {
            "ts_language": "c",
            "query": (
            "(function_definition declarator: (function_declarator declarator: (identifier) @fn.name) @fn.def) "
            "(declaration declarator: (function_declarator declarator: (identifier) @fn.name) @fn.def) "
            "(struct_specifier name: (type_identifier) @class.name) @class.def "
            "(enum_specifier name: (type_identifier) @class.name) @class.def "
            "(union_specifier name: (type_identifier) @class.name) @class.def"
            ),
        },
        "cpp": {
            "ts_language": "cpp",
            "query": (
            "(function_definition declarator: (function_declarator declarator: (identifier) @fn.name) @fn.def) "
            "(declaration declarator: (function_declarator declarator: (identifier) @fn.name) @fn.def) "
            "(class_specifier name: (type_identifier) @class.name) @class.def "
            "(struct_specifier name: (type_identifier) @class.name) @class.def "
            "(enum_specifier name: (type_identifier) @class.name) @class.def "
            "(namespace_definition name: (namespace_identifier) @class.name) @class.def"
            ),
        },
        "go": {
            "ts_language": "go",
            "query": (
            "(function_declaration name: (identifier) @fn.name) @fn.def "
            "(method_declaration name: (field_identifier) @fn.name) @fn.def "
            "(type_spec name: (type_identifier) @class.name) @class.def"
            ),
        },
        "rust": {
            "ts_language": "rust",
            "query": (
            "(function_item name: (identifier) @fn.name) @fn.def "
            "(struct_item name: (type_identifier) @class.name) @class.def "
            "(enum_item name: (type_identifier) @class.name) @class.def "
            "(trait_item name: (type_identifier) @class.name) @class.def"
            ),
        },
        "java": {
            "ts_language": "java",
            "query": (
            "(class_declaration name: (identifier) @class.name) @class.def "
            "(interface_declaration name: (identifier) @class.name) @class.def "
            "(enum_declaration name: (identifier) @class.name) @class.def "
            "(method_declaration name: (identifier) @fn.name) @fn.def "
            "(constructor_declaration name: (identifier) @fn.name) @fn.def"
            ),
        },
        "csharp": {
            "ts_language": "c_sharp",
            "query": (
            "(class_declaration name: (identifier) @class.name) @class.def "
            "(interface_declaration name: (identifier) @class.name) @class.def "
            "(struct_declaration name: (identifier) @class.name) @class.def "
            "(enum_declaration name: (identifier) @class.name) @class.def "
            "(method_declaration name: (identifier) @fn.name) @fn.def"
            ),
        },
        "kotlin": {
            "ts_language": "kotlin",
            "query": (
            "(class_declaration name: (type_identifier) @class.name) @class.def "
            "(object_declaration name: (type_identifier) @class.name) @class.def "
            "(function_declaration name: (simple_identifier) @fn.name) @fn.def"
            ),
        },
        "php": {
            "ts_language": "php",
            "query": (
            "(class_declaration name: (name) @class.name) @class.def "
            "(interface_declaration name: (name) @class.name) @class.def "
            "(trait_declaration name: (name) @class.name) @class.def "
            "(function_definition name: (name) @fn.name) @fn.def "
            "(method_declaration name: (name) @fn.name) @fn.def"
            ),
        },
        "swift": {
            "ts_language": "swift",
            "query": (
            "(class_declaration name: (type_identifier) @class.name) @class.def "
            "(struct_declaration name: (type_identifier) @class.name) @class.def "
            "(protocol_declaration name: (type_identifier) @class.name) @class.def "
            "(function_declaration name: (simple_identifier) @fn.name) @fn.def"
            ),
        },
        "shell": {
            "ts_language": "bash",
            "query": "(function_definition name: (word) @function.name) @function.def",
        },
        "scala": {
            "ts_language": "scala",
            "query": (
                "(object_definition name: (identifier) @class.name) @class.def "
                "(class_definition name: (identifier) @class.name) @class.def "
                "(trait_definition name: (identifier) @class.name) @class.def "
                "(function_definition name: (identifier) @function.name) @function.def"
            ),
        },
        "ruby": {
            "ts_language": "ruby",
            "query": (
                "(class name: (constant) @class.name) @class.def "
                "(module name: (constant) @module.name) @module.def "
                "(method name: (identifier) @function.name) @function.def"
            ),
        },
        "css": {
            "ts_language": "css",
            "query": (
                "(rule_set (selectors (class_selector) @selector.name)) @selector.def "
                "(keyframes_statement (keyframes_name) @animation.name) @animation.def"
            ),
        },
        "html": {
            "ts_language": "html",
            "query": "(element (start_tag (tag_name) @element.name)) @element.def",
        },
        "json": {
            "ts_language": "json",
            "query": "(pair key: (string (string_content) @key.name)) @key.def",
        },
        "yaml": {
            "ts_language": "yaml",
            "query": (
                "(block_mapping_pair key: (flow_node (plain_scalar (string_scalar) @key.name))) @key.def"
            ),
        },
        "toml": {
            "ts_language": "toml",
            "query": (
                "(table (bare_key) @section.name) @section.def "
                "(pair (bare_key) @key.name) @key.def"
            ),
        },
        "elixir": {
            "ts_language": "elixir",
            "query": (
                "(call target: (identifier) @_mod (#eq? @_mod \"defmodule\") (arguments (alias) @module.name)) @module.def "
                "(call target: (identifier) @_fn (#match? @_fn \"defp?\") (arguments (call target: (identifier) @function.name))) @function.def"
            ),
        },
        "erlang": {
            "ts_language": "erlang",
            "query": (
                "(module_attribute (atom) @module.name) @module.def "
                "(fun_decl (function_clause (atom) @function.name)) @function.def"
            ),
        },
        "haskell": {
            "ts_language": "haskell",
            "query": "(function name: (variable) @function.name) @function.def",
        },
        "objective_c": {
            "ts_language": "objc",
            "query": (
                "(class_interface name: (identifier) @class.name) @class.def "
                "(class_implementation name: (identifier) @class.name) @class.def "
                "(method_declaration (identifier) @method.name) @method.def "
                "(method_definition (identifier) @method.name) @method.def"
            ),
        },
    }
    _TREE_SITTER_NAME_NODE_TYPES = {
        "alias",
        "atom",
        "bare_key",
        "class_selector",
        "constant",
        "keyframes_name",
        "identifier",
        "type_identifier",
        "field_identifier",
        "simple_identifier",
        "name",
        "namespace_identifier",
        "plain_scalar",
        "string_content",
        "string_scalar",
        "tag_name",
        "variable",
        "word",
    }

    def __init__(self) -> None:
        """Initialize parser backends."""
        self.languages = {}
        self.get_language = None
        self.get_parser = None
        self._native_runtime = None
        if TREE_SITTER_AVAILABLE:
            try:
                from tree_sitter_languages import get_language, get_parser

                # Validate the backend once to avoid per-file failures on
                # incompatible tree_sitter/tree_sitter_languages combinations.
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"Language\(path, name\) is deprecated.*",
                        category=FutureWarning,
                    )
                    get_parser("python")
                    get_language("python")
                self.get_language = get_language
                self.get_parser = get_parser
            except ImportError:
                logger.warning(
                    "tree_sitter_languages not found. AST backends are disabled."
                )
            except Exception as e:
                logger.warning(
                    "tree_sitter_languages backend is incompatible with installed "
                    "tree_sitter runtime (%s). AST backends are disabled.",
                    e,
                )
        try:
            from saguaro.indexing.native_runtime import NativeIndexRuntime

            self._native_runtime = NativeIndexRuntime()
        except Exception:
            self._native_runtime = None

    @staticmethod
    def _match_tree_sitter_capture_names_python(
        *,
        def_starts: list[int],
        def_ends: list[int],
        def_type_ids: list[int],
        name_starts: list[int],
        name_ends: list[int],
        name_type_ids: list[int],
    ) -> list[int]:
        """Match name captures to the innermost compatible definition."""

        matches = [-1] * len(def_starts)
        candidates_by_name: list[list[tuple[int, int]]] = []
        for name_idx, (name_start, name_end, name_type) in enumerate(
            zip(name_starts, name_ends, name_type_ids, strict=False)
        ):
            candidates: list[tuple[int, int]] = []
            for def_idx, (def_start, def_end, def_type) in enumerate(
                zip(def_starts, def_ends, def_type_ids, strict=False)
            ):
                if int(def_type) != int(name_type):
                    continue
                if int(def_start) <= int(name_start) and int(name_end) <= int(def_end):
                    span = int(def_end) - int(def_start)
                    candidates.append((span, def_idx))
            candidates.sort(key=lambda item: (item[0], item[1]))
            candidates_by_name.append(candidates)

        for name_idx, candidates in enumerate(candidates_by_name):
            if not candidates:
                continue
            _span, def_idx = candidates[0]
            matches[def_idx] = name_idx
        return matches

    def _match_tree_sitter_capture_names(
        self,
        *,
        def_starts: list[int],
        def_ends: list[int],
        def_type_ids: list[int],
        name_starts: list[int],
        name_ends: list[int],
        name_type_ids: list[int],
    ) -> list[int]:
        runtime = getattr(self, "_native_runtime", None)
        if runtime is not None:
            try:
                return runtime.match_capture_names(
                    def_starts=def_starts,
                    def_ends=def_ends,
                    def_type_ids=def_type_ids,
                    name_starts=name_starts,
                    name_ends=name_ends,
                    name_type_ids=name_type_ids,
                )
            except Exception:
                pass
        return self._match_tree_sitter_capture_names_python(
            def_starts=def_starts,
            def_ends=def_ends,
            def_type_ids=def_type_ids,
            name_starts=name_starts,
            name_ends=name_ends,
            name_type_ids=name_type_ids,
        )

    def parse_file(self, file_path: str) -> list[CodeEntity]:
        """Parses a file and returns a list of CodeEntities."""
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return []

        with open(file_path, encoding="utf-8", errors="ignore") as f:
            content = f.read()

        entities = []
        tree = None
        lang_name = self._detect_language(file_path, content)
        if lang_name == "python":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return self._parse_python_ast(file_path, content)

        if self.declares_ast_language(lang_name):
            if not self.supports_ast_language(lang_name):
                logger.debug(
                    "Declared AST backend unavailable for %s (%s)", file_path, lang_name
                )
                return []
            entities, tree = self._parse_tree_sitter_entities(file_path, content, lang_name)

        if (
            not entities
            and not self.declares_ast_language(lang_name)
            and lang_name in self._LIGHTWEIGHT_LANGUAGES
        ):
            return self._parse_shell_cmake(file_path, content, lang_name)

        # If entities found via TS, return them + file-summary and file entities.
        if entities:
            imports = self._extract_imports(tree.root_node, content, lang_name)
            exports = [
                entity.name
                for entity in entities
                if entity.type not in {"dependency_graph", "file", "file_summary", "section"}
            ]
            dependency_payload = self._build_dependency_payload(
                entities=entities,
                imports=imports,
                exports=exports,
            )
            entities.extend(
                self._section_entities(
                    file_path=file_path,
                    content=content,
                    base_entities=entities,
                    imports=imports,
                )
            )
            entities.append(
                self._file_summary_entity(
                    file_path=file_path,
                    content=content,
                    imports=imports,
                    exports=exports,
                )
            )
            entities.append(
                CodeEntity(
                    name=os.path.basename(file_path),
                    type="file",
                    content=content,
                    start_line=1,
                    end_line=content.count("\n") + 1,
                    file_path=file_path,
                    metadata=self._entity_metadata(
                        file_path=file_path,
                        entity_name=os.path.basename(file_path),
                        entity_type="file",
                        content=content,
                        chunk_role="full",
                    ),
                )
            )
            entities.append(
                CodeEntity(
                    name="dependency_graph",
                    type="dependency_graph",
                    content=json.dumps(dependency_payload),
                    start_line=1,
                    end_line=1,
                    file_path=file_path,
                    metadata=self._entity_metadata(
                        file_path=file_path,
                        entity_name="dependency_graph",
                        entity_type="dependency_graph",
                        content=" ".join(imports + exports),
                        chunk_role="dependency",
                    ),
                )
            )
            return entities

        logger.debug("No explicit parser backend for %s (%s)", file_path, lang_name)
        return []

    def declares_ast_language(self, lang_name: str) -> bool:
        normalized = str(lang_name or "").strip().lower()
        if normalized == "python":
            return True
        return normalized in self._TREE_SITTER_BACKENDS

    def supports_ast_language(self, lang_name: str) -> bool:
        normalized = str(lang_name or "").strip().lower()
        if normalized == "python":
            return True
        backend = self._tree_sitter_backend(normalized)
        return bool(backend and self.get_language and self.get_parser)

    def supports_structural_language(self, lang_name: str) -> bool:
        normalized = str(lang_name or "").strip().lower()
        if normalized == "python":
            return True
        if self.declares_ast_language(normalized):
            return self.supports_ast_language(normalized)
        return bool(
            normalized in self._LIGHTWEIGHT_LANGUAGES
        )

    def supports_dependency_graph_language(self, lang_name: str) -> bool:
        return self.supports_structural_language(lang_name)

    def supports_dependency_quality_language(self, lang_name: str) -> bool:
        return self.supports_structural_language(lang_name)

    def language_capabilities(self, lang_name: str) -> dict[str, bool]:
        descriptor = self.language_support_descriptor(lang_name)
        return {
            "ast": bool(descriptor.get("ast")),
            "structural": bool(descriptor.get("structural")),
            "dependency_graph": bool(descriptor.get("dependency_graph")),
            "dependency_quality": bool(descriptor.get("dependency_quality")),
        }

    def parser_plugin_registry(self) -> dict[str, dict[str, Any]]:
        registry: dict[str, dict[str, Any]] = {}
        declared = set(self._LIGHTWEIGHT_LANGUAGES) | set(self._TREE_SITTER_BACKENDS) | {
            "python"
        }
        for language in sorted(declared):
            backend = self._tree_sitter_backend(language)
            registry[language] = {
                "language": language,
                "backend_kind": (
                    "tree_sitter"
                    if backend
                    else ("lightweight" if language in self._LIGHTWEIGHT_LANGUAGES or language == "python" else "none")
                ),
                "tree_sitter_language": str((backend or {}).get("ts_language") or ""),
                "declared_ast": self.declares_ast_language(language),
                "runtime_ready": self.supports_ast_language(language),
                "structural_ready": self.supports_structural_language(language),
            }
        return registry

    def environment_probe(self) -> dict[str, Any]:
        tree_sitter_present = importlib.util.find_spec("tree_sitter") is not None
        tree_sitter_languages_present = (
            importlib.util.find_spec("tree_sitter_languages") is not None
        )
        tensorflow_present = importlib.util.find_spec("tensorflow") is not None
        registry = self.parser_plugin_registry()
        grammar_probe_failures = sorted(
            language
            for language, entry in registry.items()
            if entry.get("declared_ast") and not entry.get("runtime_ready")
        )
        return {
            "parser_environment_ready": bool(
                tree_sitter_present and tree_sitter_languages_present
            ),
            "modules": {
                "tree_sitter": tree_sitter_present,
                "tree_sitter_languages": tree_sitter_languages_present,
                "tensorflow": tensorflow_present,
            },
            "parser_plugin_count": len(registry),
            "grammar_probe_failures": grammar_probe_failures,
            "language_golden_pass_rate": round(
                (
                    (
                        len(registry) - len(grammar_probe_failures)
                    )
                    / max(1, len(registry))
                )
                * 100.0,
                1,
            ),
        }

    def language_support_descriptor(self, lang_name: str) -> dict[str, Any]:
        normalized = str(lang_name or "").strip().lower()
        backend = self._tree_sitter_backend(normalized)
        lexical = bool(
            normalized == "python"
            or normalized in self._LIGHTWEIGHT_LANGUAGES
            or normalized in self._TREE_SITTER_BACKENDS
        )
        ast = self.supports_ast_language(normalized)
        structural = self.supports_structural_language(normalized)
        dependency_graph = self.supports_dependency_graph_language(normalized)
        dependency_quality = self.supports_dependency_quality_language(normalized)
        if dependency_quality:
            support_tier = "deep"
        elif structural:
            support_tier = "structural"
        elif lexical:
            support_tier = "lexical"
        else:
            support_tier = "unknown"
        return {
            "language": normalized,
            "lexical": lexical,
            "declared_ast": self.declares_ast_language(normalized),
            "ast": ast,
            "structural": structural,
            "dependency_graph": dependency_graph,
            "dependency_quality": dependency_quality,
            "support_tier": support_tier,
            "backend_kind": (
                "tree_sitter"
                if backend
                else ("lightweight" if normalized in self._LIGHTWEIGHT_LANGUAGES or normalized == "python" else "none")
            ),
            "tree_sitter_language": str((backend or {}).get("ts_language") or ""),
        }

    def _parse_tree_sitter_entities(
        self, file_path: str, content: str, lang_name: str
    ) -> tuple[list[CodeEntity], Any]:
        backend = self._tree_sitter_backend(lang_name)
        if not backend or not self.supports_ast_language(lang_name):
            return [], None

        try:
            ts_language = str(backend.get("ts_language") or lang_name)
            if lang_name == "typescript" and str(file_path).lower().endswith(".tsx"):
                ts_language = "tsx"
            parser = self._safe_get_parser(ts_language)
            tree = parser.parse(bytes(content, "utf8"))
            query_key = "tsx_query" if ts_language == "tsx" else "query"
            query_src = str(backend.get(query_key) or backend.get("query") or "").strip()
            if not query_src:
                return [], tree
            query = self._safe_get_language(ts_language).query(query_src)
            captures = query.captures(tree.root_node)
            entities: list[CodeEntity] = []
            processed_nodes = set()
            for node, tag in captures:
                if node.id in processed_nodes or not tag.endswith(".def"):
                    continue
                entity_type = self._normalize_tree_sitter_entity_type(
                    tag.split(".", 1)[0]
                )
                name = self._tree_sitter_capture_name(
                    node=node,
                    entity_type=entity_type,
                    captures=captures,
                    content=content,
                ) or self._tree_sitter_entity_name(node=node, content=content)
                if not name:
                    continue
                snippet = content[node.start_byte : node.end_byte]
                entities.append(
                    CodeEntity(
                        name=name,
                        type=entity_type,
                        content=snippet,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        file_path=file_path,
                        metadata=self._entity_metadata(
                            file_path=file_path,
                            entity_name=name,
                            entity_type=entity_type,
                            content=snippet,
                        ),
                    )
                )
                processed_nodes.add(node.id)
            return entities, tree
        except Exception as e:
            logger.debug("Tree-sitter parse failed for %s: %s", file_path, e)
            return [], None

    def _extract_imports(
        self, root_node: Any, content: str, lang_name: str
    ) -> list[str]:
        imports = []
        if lang_name == "python":
            query = self._safe_get_language("python").query("""
            (import_from_statement module_name: (dotted_name) @mod)
            (import_statement name: (dotted_name) @mod)
            """)
            for node, _ in query.captures(root_node):
                imports.append(content[node.start_byte : node.end_byte])
        elif lang_name in {"javascript", "typescript"}:
            if root_node is not None and self.get_language:
                try:
                    query = self._safe_get_language(lang_name).query(
                        "(import_statement) @imp"
                    )
                    for node, _ in query.captures(root_node):
                        imports.append(content[node.start_byte : node.end_byte].strip())
                except Exception:
                    pass
            if not imports:
                for line in content.splitlines():
                    stripped = line.strip()
                    if stripped.startswith("import ") or stripped.startswith("export "):
                        imports.append(stripped)
        elif lang_name in {"c", "cpp"}:
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("#include"):
                    imports.append(line)
        elif lang_name == "go" or lang_name in {"java", "kotlin", "swift"}:
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("import "):
                    imports.append(line)
        elif lang_name == "rust":
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("use "):
                    imports.append(line)
        elif lang_name == "csharp":
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("using "):
                    imports.append(line)
        elif lang_name in {"scala", "fsharp"}:
            for line in content.splitlines():
                line = line.strip()
                if line.startswith(("import ", "open ")):
                    imports.append(line)
        elif lang_name == "ruby":
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("require ") or line.startswith("require_relative "):
                    imports.append(line)
        elif lang_name in {"r", "julia", "haskell"}:
            for line in content.splitlines():
                line = line.strip()
                if line.startswith(("import ", "using ", "library(", "require ")):
                    imports.append(line)
        elif lang_name == "ocaml":
            for line in content.splitlines():
                line = line.strip()
                if line.startswith(("open ", "include ")):
                    imports.append(line)
        elif lang_name == "clojure":
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("(ns ") or line.startswith("(require "):
                    imports.append(line)
        elif lang_name == "erlang":
            for line in content.splitlines():
                line = line.strip()
                if line.startswith(("-include", "-import", "-include_lib")):
                    imports.append(line)
        elif lang_name == "elixir":
            for line in content.splitlines():
                line = line.strip()
                if line.startswith(("import ", "alias ", "use ", "require ")):
                    imports.append(line)
        elif lang_name in {"php", "hack"}:
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("use ") or line.startswith(("require", "include")):
                    imports.append(line)
        elif lang_name in {"shell", "powershell"}:
            for line in content.splitlines():
                line = line.strip()
                if (
                    line.startswith("source ")
                    or line.startswith(". ")
                    or line.startswith("export ")
                    or line.lower().startswith("import-module ")
                ):
                    imports.append(line)
        elif lang_name == "cmake":
            for line in content.splitlines():
                line = line.strip()
                lowered = line.lower()
                if lowered.startswith(
                    ("include(", "find_package(", "add_subdirectory(")
                ):
                    imports.append(line)
        elif lang_name in {"proto", "thrift", "flatbuffers"}:
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("import "):
                    imports.append(line)
        elif lang_name == "graphql":
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("#import "):
                    imports.append(line)
        elif lang_name == "make":
            for line in content.splitlines():
                line = line.strip()
                if line.startswith(("include ", "-include ")):
                    imports.append(line)
        elif lang_name == "docker":
            for line in content.splitlines():
                line = line.strip()
                if line.upper().startswith("FROM "):
                    imports.append(line)
        elif lang_name == "bazel":
            for line in content.splitlines():
                line = line.strip()
                if line.startswith(("load(", "package(", "workspace(")):
                    imports.append(line)
        elif lang_name == "zig":
            for line in content.splitlines():
                line = line.strip()
                if "@import(" in line or line.startswith("usingnamespace "):
                    imports.append(line)
        elif lang_name == "nim":
            for line in content.splitlines():
                line = line.strip()
                if line.startswith(("import ", "from ", "include ")):
                    imports.append(line)
        elif lang_name == "hdl":
            for line in content.splitlines():
                line = line.strip()
                lowered = line.lower()
                if (
                    line.startswith("`include")
                    or lowered.startswith("use ")
                    or lowered.startswith("library ")
                ):
                    imports.append(line)
        elif lang_name == "fortran":
            for line in content.splitlines():
                line = line.strip()
                lowered = line.lower()
                if lowered.startswith("use ") or lowered.startswith("include "):
                    imports.append(line)
        elif lang_name == "pascal":
            for line in content.splitlines():
                line = line.strip()
                lowered = line.lower()
                if lowered.startswith("uses ") or line.startswith("{$I"):
                    imports.append(line)
        elif lang_name == "ada":
            for line in content.splitlines():
                line = line.strip()
                lowered = line.lower()
                if lowered.startswith(("with ", "use ")):
                    imports.append(line)
        elif lang_name == "cobol":
            for line in content.splitlines():
                line = line.strip()
                if line.lower().startswith("copy "):
                    imports.append(line)
        elif lang_name == "solidity":
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("import "):
                    imports.append(line)
        elif lang_name == "objective_c":
            for line in content.splitlines():
                line = line.strip()
                if line.startswith(("#import", "@import")):
                    imports.append(line)
        elif lang_name == "assembly":
            for line in content.splitlines():
                line = line.strip()
                if line.lower().startswith((".include", "%include", "include ")):
                    imports.append(line)
        elif lang_name in {"vlang", "crystal", "dlang", "actionscript", "matlab", "octave"}:
            for line in content.splitlines():
                line = line.strip()
                if line.startswith(("import ", "using ", "include ")):
                    imports.append(line)
        elif lang_name == "wolfram":
            for line in content.splitlines():
                line = line.strip()
                if line.startswith(("Get[", "<<", "Needs[")):
                    imports.append(line)
        elif lang_name == "prolog":
            for line in content.splitlines():
                line = line.strip()
                if line.startswith(":- use_module"):
                    imports.append(line)
        elif lang_name == "tcl":
            for line in content.splitlines():
                line = line.strip()
                if line.startswith(("package require ", "source ")):
                    imports.append(line)
        elif lang_name == "coldfusion":
            for match in re.findall(
                r"""<(?:cfinclude|cfimport)\b[^>]*(?:template|path)\s*=\s*["']([^"']+)["']""",
                content,
                flags=re.IGNORECASE,
            ):
                imports.append(match.strip())
        elif lang_name == "gdscript":
            for line in content.splitlines():
                line = line.strip()
                if "preload(" in line or line.startswith(("extends ", "class_name ")):
                    imports.append(line)
        elif lang_name == "haxe":
            for line in content.splitlines():
                line = line.strip()
                if line.startswith(("import ", "using ")):
                    imports.append(line)
        elif lang_name in {"jinja", "django_template", "ejs", "handlebars", "pug", "erb", "blade"}:
            template_import_patterns = [
                r"""{%\s*(?:include|import|extends|from)\s+["']([^"']+)["']""",
                r"""<%-?\s*include\s*\(\s*["']([^"']+)["']\s*\)""",
                r"""\{\{>\s*([A-Za-z0-9_./-]+)\s*\}\}""",
                r"""^\s*(?:include|extends)\s+([A-Za-z0-9_./-]+)\s*$""",
            ]
            for pattern in template_import_patterns:
                for match in re.findall(pattern, content, flags=re.IGNORECASE | re.MULTILINE):
                    imports.append(str(match).strip())
        elif lang_name == "latex":
            for line in content.splitlines():
                line = line.strip()
                if line.startswith((r"\usepackage", r"\input", r"\include")):
                    imports.append(line)
        elif lang_name == "typst":
            for line in content.splitlines():
                line = line.strip()
                if line.startswith(("#import", "#include")):
                    imports.append(line)
        elif lang_name in {"mdx", "rst", "asciidoc", "org"}:
            for line in content.splitlines():
                line = line.strip()
                if line.startswith(("!INCLUDE ", ".. include::", "#+INCLUDE:")):
                    imports.append(line)
        elif lang_name == "nix":
            for line in content.splitlines():
                line = line.strip()
                if "import " in line or line.startswith("inherit "):
                    imports.append(line)
        elif lang_name == "hcl":
            for line in content.splitlines():
                line = line.strip()
                if "source" in line or line.startswith(("module ", "provider ", "terraform ")):
                    imports.append(line)
        elif lang_name == "puppet":
            for line in content.splitlines():
                line = line.strip()
                if line.startswith(("include ", "require ", "contain ")):
                    imports.append(line)
        elif lang_name == "qml":
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("import "):
                    imports.append(line)
        elif lang_name == "vim":
            for line in content.splitlines():
                line = line.strip()
                if line.startswith(("source ", "runtime ")):
                    imports.append(line)
        elif lang_name == "emacs_lisp":
            for line in content.splitlines():
                line = line.strip()
                if line.startswith(("(require ", "(load ")):
                    imports.append(line)
        elif lang_name == "wat":
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("(import "):
                    imports.append(line)
        elif lang_name == "html":
            for match in re.findall(
                r"""<(?:script|link|img)\b[^>]*(?:src|href)\s*=\s*["']([^"']+)["']""",
                content,
                flags=re.IGNORECASE,
            ):
                imports.append(match.strip())
        elif lang_name == "css":
            for line in content.splitlines():
                line = line.strip()
                if line.lower().startswith("@import "):
                    imports.append(line)
        elif lang_name in {"json", "yaml", "toml", "xml", "config"}:
            for line in content.splitlines():
                line = line.strip()
                lowered = line.lower()
                if lowered.startswith(("include", "imports", "$include", "source")) and (
                    ":" in line or "=" in line
                ):
                    imports.append(line)
            if lang_name == "xml":
                for match in re.findall(
                    r"""<xi:include\b[^>]*href\s*=\s*["']([^"']+)["']""",
                    content,
                    flags=re.IGNORECASE,
                ):
                    imports.append(match.strip())
        return sorted(list(set(imports)))

    def _parse_python_ast(self, file_path: str, content: str) -> list[CodeEntity]:
        entities: list[CodeEntity] = []
        lines = content.splitlines()

        try:
            tree = ast.parse(content)
        except SyntaxError:
            # Keep non-empty indexability even for syntactically invalid files.
            return [
                CodeEntity(
                    name=os.path.basename(file_path),
                    type="file",
                    content=content,
                    start_line=1,
                    end_line=max(1, len(lines)),
                    file_path=file_path,
                )
            ]

        exports: list[str] = []
        constant_names: set[str] = set()
        imports: list[str] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                exports.append(node.name)
                entities.append(
                    CodeEntity(
                        name=node.name,
                        type="class",
                        content=self._python_node_content(lines, node),
                        start_line=node.lineno,
                        end_line=getattr(node, "end_lineno", node.lineno),
                        file_path=file_path,
                        metadata=self._entity_metadata(
                            file_path=file_path,
                            entity_name=node.name,
                            entity_type="class",
                            content=self._python_node_content(lines, node),
                        ),
                    )
                )
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        qualified_name = f"{node.name}.{child.name}"
                        entities.append(
                            CodeEntity(
                                name=qualified_name,
                                type="method",
                                content=self._python_node_content(lines, child),
                                start_line=child.lineno,
                                end_line=getattr(child, "end_lineno", child.lineno),
                                file_path=file_path,
                                metadata=self._entity_metadata(
                                    file_path=file_path,
                                    entity_name=qualified_name,
                                    entity_type="method",
                                    content=self._python_node_content(lines, child),
                                    parent_symbol=node.name,
                                ),
                            )
                        )
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                exports.append(node.name)
                entities.append(
                    CodeEntity(
                        name=node.name,
                        type="function",
                        content=self._python_node_content(lines, node),
                        start_line=node.lineno,
                        end_line=getattr(node, "end_lineno", node.lineno),
                        file_path=file_path,
                        metadata=self._entity_metadata(
                            file_path=file_path,
                            entity_name=node.name,
                            entity_type="function",
                            content=self._python_node_content(lines, node),
                        ),
                    )
                )
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                for name in self._python_assignment_targets(node):
                    if not (name.isupper() or name.endswith("_CONFIG")):
                        continue
                    constant_names.add(name)
                    exports.append(name)
                    entities.append(
                        CodeEntity(
                            name=name,
                            type="constant",
                            content=self._python_node_content(lines, node),
                            start_line=getattr(node, "lineno", 1),
                            end_line=getattr(
                                node, "end_lineno", getattr(node, "lineno", 1)
                            ),
                            file_path=file_path,
                            metadata=self._entity_metadata(
                                file_path=file_path,
                                entity_name=name,
                                entity_type="constant",
                                content=self._python_node_content(lines, node),
                            ),
                        )
                    )

        entities.extend(
            self._section_entities(
                file_path=file_path,
                content=content,
                base_entities=entities,
                imports=sorted(set(imports)),
            )
        )
        entities.append(
            self._file_summary_entity(
                file_path=file_path,
                content=content,
                imports=sorted(set(imports)),
                exports=exports,
            )
        )
        dependency_payload = self._build_python_dependency_payload(
            tree=tree,
            imports=sorted(set(imports)),
            exports=exports,
            constant_names=constant_names,
        )
        entities.append(
            CodeEntity(
                name="dependency_graph",
                type="dependency_graph",
                content=json.dumps(dependency_payload),
                start_line=1,
                end_line=1,
                file_path=file_path,
                metadata=self._entity_metadata(
                    file_path=file_path,
                    entity_name="dependency_graph",
                    entity_type="dependency_graph",
                    content=" ".join(imports + exports),
                    chunk_role="dependency",
                ),
            )
        )
        entities.append(
            CodeEntity(
                name=os.path.basename(file_path),
                type="file",
                content=content,
                start_line=1,
                end_line=max(1, len(lines)),
                file_path=file_path,
                metadata=self._entity_metadata(
                    file_path=file_path,
                    entity_name=os.path.basename(file_path),
                    entity_type="file",
                    content=content,
                    chunk_role="full",
                ),
            )
        )
        return entities

    def _entity_metadata(
        self,
        *,
        file_path: str,
        entity_name: str,
        entity_type: str,
        content: str,
        parent_symbol: str | None = None,
        chunk_role: str | None = None,
    ) -> dict[str, Any]:
        file_role = self._file_role(file_path)
        path_terms = self._path_terms(file_path)
        symbol_terms = self._symbol_terms(entity_name)
        doc_terms = self._extract_terms(content, limit=64)
        terms = sorted(set(path_terms + symbol_terms + doc_terms))
        role_tags = self._role_tags(
            file_path=file_path,
            entity_name=entity_name,
            entity_type=entity_type,
            content=content,
        )
        feature_families = self._feature_families(
            file_path=file_path,
            entity_name=entity_name,
            entity_type=entity_type,
            terms=terms,
            role_tags=role_tags,
        )
        boundary_markers = self._boundary_markers(
            file_path=file_path,
            entity_name=entity_name,
            content=content,
        )
        return {
            "entity_kind": entity_type,
            "parent_symbol": parent_symbol,
            "file_role": file_role,
            "chunk_role": chunk_role,
            "path_terms": path_terms,
            "symbol_terms": symbol_terms,
            "doc_terms": doc_terms,
            "terms": terms,
            "role_tags": role_tags,
            "feature_families": feature_families,
            "boundary_markers": boundary_markers,
            "signature_fingerprint": self._signature_fingerprint(
                file_path=file_path,
                entity_name=entity_name,
                entity_type=entity_type,
                role_tags=role_tags,
                feature_families=feature_families,
            ),
            "structural_fingerprint": self._structural_fingerprint(
                file_path=file_path,
                entity_name=entity_name,
                entity_type=entity_type,
                content=content,
                feature_families=feature_families,
            ),
            "parser_uncertainty": self._parser_uncertainty(
                entity_type=entity_type,
                role_tags=role_tags,
                boundary_markers=boundary_markers,
            ),
        }

    def _file_summary_entity(
        self,
        *,
        file_path: str,
        content: str,
        imports: list[str],
        exports: list[str],
    ) -> CodeEntity:
        lines = content.splitlines()
        module_doc = self._summarize_docstring(content)
        parts = [
            f"file {os.path.basename(file_path)}",
            f"path {file_path}",
            f"role {self._file_role(file_path)}",
        ]
        if exports:
            parts.append("exports " + " ".join(sorted(dict.fromkeys(exports))[:24]))
        if imports:
            parts.append("imports " + " ".join(sorted(dict.fromkeys(imports))[:24]))
        if module_doc:
            parts.append("summary " + module_doc)
        summary_text = "\n".join(parts)
        return CodeEntity(
            name=os.path.basename(file_path),
            type="file_summary",
            content=summary_text,
            start_line=1,
            end_line=max(1, len(lines)),
            file_path=file_path,
            metadata=self._entity_metadata(
                file_path=file_path,
                entity_name=os.path.basename(file_path),
                entity_type="file_summary",
                content=summary_text,
                chunk_role="summary",
            ),
        )

    def _section_entities(
        self,
        *,
        file_path: str,
        content: str,
        base_entities: list[CodeEntity],
        imports: list[str],
    ) -> list[CodeEntity]:
        lines = content.splitlines()
        if len(lines) < 180:
            return []

        sections: list[tuple[int, int, str]] = []
        structural = [
            entity
            for entity in base_entities
            if entity.type in {"class", "function", "method", "constant"}
        ]
        if structural:
            current_start = 1
            current_end = 1
            current_names: list[str] = []
            for entity in sorted(structural, key=lambda item: (item.start_line, item.end_line)):
                if not current_names:
                    current_start = entity.start_line
                    current_end = entity.end_line
                    current_names = [entity.name]
                    continue
                if entity.end_line - current_start > 170 or len(current_names) >= 4:
                    sections.append((current_start, current_end, ", ".join(current_names[:4])))
                    current_start = entity.start_line
                    current_end = entity.end_line
                    current_names = [entity.name]
                else:
                    current_end = max(current_end, entity.end_line)
                    current_names.append(entity.name)
            if current_names:
                sections.append((current_start, current_end, ", ".join(current_names[:4])))

        if not sections:
            chunk_size = 140
            overlap = 24
            start = 1
            chunk_index = 1
            while start <= len(lines):
                end = min(len(lines), start + chunk_size - 1)
                sections.append((start, end, f"chunk_{chunk_index}"))
                if end == len(lines):
                    break
                start = end - overlap + 1
                chunk_index += 1

        entities: list[CodeEntity] = []
        for index, (start, end, label) in enumerate(sections, start=1):
            section_lines = "\n".join(lines[start - 1 : end])
            header = f"section {index} {label}\nimports {' '.join(imports[:16])}\n"
            section_name = f"{os.path.basename(file_path)}::section_{index}"
            entities.append(
                CodeEntity(
                    name=section_name,
                    type="section",
                    content=header + section_lines,
                    start_line=start,
                    end_line=end,
                    file_path=file_path,
                    metadata=self._entity_metadata(
                        file_path=file_path,
                        entity_name=section_name,
                        entity_type="section",
                        content=header + section_lines,
                        chunk_role="section",
                    ),
                )
            )
        return entities

    @staticmethod
    def _python_node_content(lines: list[str], node: ast.AST) -> str:
        start = max(1, int(getattr(node, "lineno", 1)))
        end = max(start, int(getattr(node, "end_lineno", start)))
        return "\n".join(lines[start - 1 : end])

    @staticmethod
    def _python_assignment_targets(node: ast.AST) -> list[str]:
        names: list[str] = []
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.append(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.append(node.target.id)
        return names

    @staticmethod
    def _build_python_dependency_payload(
        tree: ast.Module,
        imports: list[str],
        exports: list[str],
        constant_names: set[str],
    ) -> dict[str, object]:
        exported = set(exports)
        edges: dict[tuple[str, str, str], dict[str, object]] = {}

        def add_edge(source: str, target: str, relation: str, line: int) -> None:
            if not source or not target or source == target:
                return
            key = (source, target, relation)
            if key not in edges:
                edges[key] = {
                    "from": source,
                    "to": target,
                    "relation": relation,
                    "line": int(line or 1),
                }

        def resolve_callee(expr: ast.AST, *, class_name: str | None = None) -> str:
            if isinstance(expr, ast.Name):
                qualified = f"{class_name}.{expr.id}" if class_name else ""
                if qualified and qualified in exported:
                    return qualified
                return expr.id
            if isinstance(expr, ast.Attribute):
                if (
                    class_name
                    and isinstance(expr.value, ast.Name)
                    and expr.value.id in {"self", "cls"}
                ):
                    return f"{class_name}.{expr.attr}"
                return expr.attr
            return ""

        def scan_source(
            source: str,
            scan_nodes: list[ast.stmt],
            *,
            class_name: str | None = None,
        ) -> None:
            for inner in ast.walk(ast.Module(body=scan_nodes, type_ignores=[])):
                if isinstance(inner, ast.Call):
                    callee = resolve_callee(inner.func, class_name=class_name)
                    if callee in exported:
                        add_edge(source, callee, "calls", getattr(inner, "lineno", 1))
                elif isinstance(inner, ast.Name) and isinstance(inner.ctx, ast.Load):
                    if inner.id in constant_names:
                        add_edge(source, inner.id, "reads", getattr(inner, "lineno", 1))

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                scan_source(node.name, list(node.body))
            elif isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        scan_source(
                            f"{node.name}.{child.name}",
                            list(child.body),
                            class_name=node.name,
                        )

        return {
            "imports": imports,
            "exports": exports,
            "internal_edges": list(edges.values()),
        }

    @staticmethod
    def _summarize_docstring(content: str) -> str:
        match = re.search(r'"""(.*?)"""', content, re.DOTALL) or re.search(
            r"'''(.*?)'''", content, re.DOTALL
        )
        if not match:
            return ""
        summary = " ".join(match.group(1).strip().split())
        return summary[:240]

    def _extract_terms(self, text: str, limit: int = 64) -> list[str]:
        expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(text or ""))
        seen: list[str] = []
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_]{1,}", expanded):
            normalized = token.lower()
            if normalized.isdigit() or normalized in seen:
                continue
            seen.append(normalized)
            if len(seen) >= limit:
                break
        return seen

    def _role_tags(
        self,
        *,
        file_path: str,
        entity_name: str,
        entity_type: str,
        content: str,
    ) -> list[str]:
        lowered_path = file_path.lower().replace("\\", "/")
        lowered_name = str(entity_name or "").lower()
        parts = [part for part in lowered_path.split("/") if part]
        tags = {entity_type, self._file_role(file_path)}
        if entity_type in {"function", "method", "class"}:
            tags.add("symbol")
        if any(part in {"core", "runtime", "engine", "service"} for part in parts):
            tags.update({"core_runtime", "service"})
        if any(part in {"cli", "command", "commands", "shell", "terminal"} for part in parts):
            tags.update({"cli", "entrypoint", "orchestration"})
        if any(part in {"registry", "registries", "plugin", "plugins"} for part in parts):
            tags.update({"registry", "plugin"})
        if any(part in {"state", "session", "context"} for part in parts):
            tags.update({"state", "session"})
        if any(part in {"artifact", "artifacts", "report", "reporting", "output", "outputs"} for part in parts):
            tags.update({"artifact", "report_surface"})
        if any(part in {"framework", "frameworks", "adapter", "adapters"} for part in parts):
            tags.update({"adapter", "framework"})
        if any(part in {"target", "targets"} for part in parts):
            tags.add("target")
        if any(part in {"tests", "test"} for part in parts) or lowered_path.startswith("tests/"):
            tags.add("test_harness")
        if lowered_path.endswith(".md"):
            tags.add("doc")
        if entity_type == "dependency_graph":
            tags.add("graph_surface")
        if any(item in lowered_name for item in {"pipeline", "model"}):
            tags.add("pipeline")
        if any(item in lowered_name for item in {"ffi", "bridge", "ctypes", "cffi"}):
            tags.add("native_boundary")
        if "ffi" in lowered_path:
            tags.add("native_boundary")
        if "qsg" in lowered_path:
            tags.add("qsg_surface")
        if "saguaro" in lowered_path:
            tags.add("saguaro_surface")
        if "native" in lowered_path:
            tags.add("native")
        if re.search(r"\b(return|yield|await|async)\b", content):
            tags.add("callable_body")
        return sorted(item for item in tags if item and item != "unknown")

    def _feature_families(
        self,
        *,
        file_path: str,
        entity_name: str,
        entity_type: str,
        terms: list[str],
        role_tags: list[str],
    ) -> list[str]:
        lowered_path = file_path.lower()
        lowered_name = str(entity_name or "").lower()
        candidates = {lowered_path, lowered_name, *[str(item).lower() for item in terms + role_tags]}
        families: set[str] = set()
        mapping = {
            "adapter": "framework_adapter",
            "artifact": "artifact_output",
            "attack": "attack_orchestration",
            "dataflow": "dataflow",
            "diagnostic": "diagnostics",
            "evaluat": "evaluation_pipeline",
            "extractor": "extractor",
            "ffi": "native_integration",
            "graph": "dataflow",
            "index": "query_engine",
            "model": "evaluation_pipeline",
            "optim": "optimization",
            "pipeline": "evaluation_pipeline",
            "query": "query_engine",
            "registry": "target_registry",
            "report": "reporting",
            "security": "security_analysis",
            "session": "runtime_state",
            "state": "runtime_state",
            "target": "target_registry",
        }
        for candidate in candidates:
            for needle, family in mapping.items():
                if needle in candidate:
                    families.add(family)
        if entity_type == "dependency_graph":
            families.add("extractor")
        return sorted(families)

    def _boundary_markers(
        self,
        *,
        file_path: str,
        entity_name: str,
        content: str,
    ) -> list[str]:
        lowered = "\n".join([file_path.lower(), str(entity_name or "").lower(), str(content or "").lower()])
        markers = []
        for marker in (
            "ctypes",
            "cffi",
            "pybind11",
            "extern c",
            "pyo3",
            "tensorflow",
            "ffi",
        ):
            if marker in lowered:
                markers.append(marker.replace(" ", "_"))
        return sorted(set(markers))

    def _signature_fingerprint(
        self,
        *,
        file_path: str,
        entity_name: str,
        entity_type: str,
        role_tags: list[str],
        feature_families: list[str],
    ) -> str:
        payload = "|".join(
            [
                os.path.basename(file_path).lower(),
                str(entity_name or "").lower(),
                str(entity_type or "").lower(),
                ",".join(sorted(role_tags)),
                ",".join(sorted(feature_families)),
            ]
        )
        return f"sig:{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:12]}"

    def _structural_fingerprint(
        self,
        *,
        file_path: str,
        entity_name: str,
        entity_type: str,
        content: str,
        feature_families: list[str],
    ) -> str:
        metrics = [
            f"type={entity_type}",
            f"lines={len(str(content or '').splitlines())}",
            f"imports={len(re.findall(r'\bimport\b', str(content or '')))}",
            f"calls={len(re.findall(r'\w+\s*\(', str(content or '')))}",
            f"families={','.join(sorted(feature_families))}",
            f"name={str(entity_name or '').lower()}",
            f"path={os.path.basename(file_path).lower()}",
        ]
        payload = "|".join(metrics)
        return f"struct:{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:12]}"

    @staticmethod
    def _parser_uncertainty(
        *,
        entity_type: str,
        role_tags: list[str],
        boundary_markers: list[str],
    ) -> str:
        if entity_type in {"file", "file_summary", "dependency_graph"}:
            return "medium"
        if "test_harness" in set(role_tags) and not boundary_markers:
            return "medium"
        return "low"

    def _path_terms(self, file_path: str) -> list[str]:
        rel = file_path.replace("\\", "/")
        tokens = []
        for piece in rel.split("/"):
            tokens.extend(self._extract_terms(piece, limit=8))
        return sorted(dict.fromkeys(tokens))

    def _symbol_terms(self, symbol: str) -> list[str]:
        return self._extract_terms(symbol.replace(".", " "), limit=32)

    def _file_role(self, file_path: str) -> str:
        lowered = file_path.replace("\\", "/").lower()
        if lowered.startswith(("tests/", "test/")) or "/tests/" in lowered:
            return "test"
        if lowered.startswith(("docs/", "doc/")) or lowered.endswith(
            (".md", ".mdx", ".rst", ".txt", ".adoc", ".asciidoc", ".org", ".tex", ".typ")
        ):
            return "doc"
        if lowered.startswith(("audit/", "benchmarks/")):
            return "bench"
        return "source"

    def detect_language(self, file_path: str, content: str = "") -> str:
        """Return the canonical parser language id for a file path."""
        return self._detect_language(file_path, content)

    def _detect_language(self, file_path: str, content: str = "") -> str:
        lower = file_path.lower()
        base = os.path.basename(lower)
        if base in self._LANGUAGE_BY_FILENAME:
            return self._LANGUAGE_BY_FILENAME[base]
        if base == "cmakelists.txt" or lower.endswith(".cmake"):
            return "cmake"
        if lower.endswith((".sh", ".bash", ".zsh", ".fish", ".bat", ".cmd")):
            return "shell"
        if lower.endswith((".ps1", ".psm1", ".psd1")):
            return "powershell"
        if lower.endswith((".py", ".pyi", ".pyx", ".pxd", ".ipynb")):
            return "python"
        if lower.endswith((".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".mm")):
            return "cpp"
        if lower.endswith(".m"):
            if re.search(r"^\s*(?:@interface|@implementation|#import|@protocol)\b", content, re.M):
                return "objective_c"
            if re.search(r"^\s*(?:function|classdef)\b", content, re.M):
                return "matlab"
            return "cpp"
        if lower.endswith(".c"):
            return "c"
        if lower.endswith((".js", ".jsx", ".mjs", ".cjs", ".vue", ".svelte", ".astro")):
            return "javascript"
        if lower.endswith((".ts", ".tsx", ".mts", ".cts")):
            return "typescript"
        if lower.endswith(".go"):
            return "go"
        if lower.endswith(".rs"):
            return "rust"
        if lower.endswith((".java", ".groovy")):
            return "java"
        if lower.endswith((".scala", ".sc")):
            return "scala"
        if lower.endswith((".cs", ".vb")):
            return "csharp"
        if lower.endswith((".fs", ".fsx")):
            return "fsharp"
        if lower.endswith((".kt", ".kts")):
            return "kotlin"
        if lower.endswith((".swift", ".dart")):
            return "swift"
        if lower.endswith(".zig"):
            return "zig"
        if lower.endswith((".nim", ".nims")):
            return "nim"
        if lower.endswith(".v"):
            if re.search(
                r"^\s*(?:module|interface|package|`include|always|assign)\b",
                content,
                re.I | re.M,
            ):
                return "hdl"
            return "vlang"
        if lower.endswith((".vh", ".sv", ".svh", ".vhd", ".vhdl", ".hdl")):
            return "hdl"
        if lower.endswith((".f", ".for", ".f77", ".f90", ".f95", ".f03", ".f08")):
            return "fortran"
        if lower.endswith((".pas", ".lpr", ".dpr")):
            return "pascal"
        if lower.endswith(".pp"):
            if re.search(r"^\s*(?:class|define|node)\s+\{?", content, re.I | re.M):
                return "puppet"
            return "pascal"
        if lower.endswith((".adb", ".ads", ".ada")):
            return "ada"
        if lower.endswith((".cob", ".cbl", ".cpy")):
            return "cobol"
        if lower.endswith(".sol"):
            return "solidity"
        if lower.endswith(".vy"):
            return "solidity"
        if lower.endswith((".d",)):
            return "dlang"
        if lower.endswith((".cr",)):
            return "crystal"
        if lower.endswith((".asm", ".s", ".sx", ".S")):
            return "assembly"
        if lower.endswith((".rpg", ".rpgle")):
            return "rpg"
        if lower.endswith((".pli", ".pl1")):
            return "pli"
        if lower.endswith((".html", ".htm", ".xhtml")):
            return "html"
        if lower.endswith((".css", ".scss", ".sass", ".less")):
            return "css"
        if lower.endswith((".ini", ".cfg", ".conf", ".properties")):
            return "config"
        if lower.endswith(".gradle") or lower.endswith(".sbt"):
            return "config"
        if lower.endswith((".tf", ".tfvars", ".hcl")):
            return "hcl"
        if lower.endswith(".nix"):
            return "nix"
        if lower.endswith(".blade.php"):
            return "blade"
        if lower.endswith(".php") or lower.endswith(".phtml"):
            if content.lstrip().startswith("<?hh"):
                return "hack"
            return "php"
        if lower.endswith((".rb", ".rake", ".gemspec", ".ru")):
            return "ruby"
        if lower.endswith(".r"):
            return "r"
        if lower.endswith(".jl"):
            return "julia"
        if lower.endswith((".mat",)):
            return "matlab"
        if lower.endswith((".nb", ".wl", ".wls")):
            return "wolfram"
        if lower.endswith((".hs", ".lhs")):
            return "haskell"
        if lower.endswith((".ml", ".mli")):
            return "ocaml"
        if lower.endswith((".clj", ".cljs", ".cljc", ".edn")):
            return "clojure"
        if lower.endswith((".lisp", ".lsp", ".scm", ".rkt")):
            return "lisp"
        if lower.endswith((".erl", ".hrl")):
            return "erlang"
        if lower.endswith((".ex", ".exs")):
            return "elixir"
        if lower.endswith((".pl", ".pm")):
            if re.search(r"^\s*:-\s*(?:module|use_module)\b", content, re.M):
                return "prolog"
            return "perl"
        if lower.endswith(".lua"):
            return "lua"
        if lower.endswith(".sql"):
            return "sql"
        if lower.endswith((".proto", ".thrift")):
            return "proto" if lower.endswith(".proto") else "thrift"
        if lower.endswith(".fbs"):
            return "flatbuffers"
        if lower.endswith((".graphql", ".gql")):
            return "graphql"
        if lower.endswith((".cypher", ".cql")):
            return "cypher"
        if lower.endswith(".aql"):
            return "aql"
        if lower.endswith(".mql"):
            return "mql"
        if lower.endswith((".bzl", ".bazel")):
            return "bazel"
        if lower.endswith((".json", ".jsonc", ".json5")):
            return "json"
        if lower.endswith(".csv"):
            return "csv"
        if lower.endswith((".yaml", ".yml")):
            return "yaml"
        if lower.endswith(".toml"):
            return "toml"
        if lower.endswith(".xml"):
            return "xml"
        if lower.endswith(".mdx"):
            return "mdx"
        if lower.endswith((".j2", ".jinja", ".jinja2")):
            return "jinja"
        if lower.endswith((".dtl",)):
            return "django_template"
        if lower.endswith(".ejs"):
            return "ejs"
        if lower.endswith((".hbs", ".mustache")):
            return "handlebars"
        if lower.endswith((".pug", ".jade")):
            return "pug"
        if lower.endswith(".erb"):
            return "erb"
        if lower.endswith((".tex", ".sty", ".cls")):
            return "latex"
        if lower.endswith(".typ"):
            return "typst"
        if lower.endswith(".rst"):
            return "rst"
        if lower.endswith((".adoc", ".asciidoc")):
            return "asciidoc"
        if lower.endswith(".org"):
            return "org"
        if lower.endswith(".qml"):
            return "qml"
        if lower.endswith(".el"):
            return "emacs_lisp"
        if lower.endswith(".vim"):
            return "vim"
        if lower.endswith(".awk"):
            return "awk"
        if lower.endswith(".sed"):
            return "sed"
        if lower.endswith(".wat"):
            return "wat"
        if lower.endswith(".tcl"):
            return "tcl"
        if lower.endswith(".st"):
            return "smalltalk"
        if lower.endswith(".hx"):
            return "haxe"
        if lower.endswith(".gd"):
            return "gdscript"
        if lower.endswith(".uc"):
            return "unrealscript"
        if lower.endswith((".cfm", ".cfc")):
            return "coldfusion"
        if lower.endswith(".as"):
            return "actionscript"
        if lower.endswith(".sc"):
            return "scala"
        if lower.endswith((".fs", ".fsx")):
            return "fsharp"

        if not os.path.splitext(base)[1]:
            first_line = (content.splitlines()[:1] or [""])[0].strip().lower()
            if first_line.startswith("#!") and any(
                marker in first_line for marker in ("sh", "bash", "zsh")
            ):
                return "shell"
            if first_line.startswith("#!") and any(
                marker in first_line for marker in ("pwsh", "powershell")
            ):
                return "powershell"
            if base.startswith(".env"):
                return "config"
            if base in {".vimrc", "vimrc"}:
                return "vim"
            if base in {".emacs", "init.el"}:
                return "emacs_lisp"
        return "unknown"

    def _parse_shell_cmake(
        self, file_path: str, content: str, lang_name: str
    ) -> list[CodeEntity]:
        lines = content.splitlines()
        entities: list[CodeEntity] = []
        exports: list[str] = []

        if lang_name == "shell":
            patterns = [
                (
                    re.compile(
                        r"^\s*(?:function\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*(?:\(\s*\))?\s*\{"
                    ),
                    "function",
                ),
                (
                    re.compile(r"^\s*([A-Z][A-Z0-9_]*)\s*=\s*.+$"),
                    "constant",
                ),
            ]
        elif lang_name == "powershell":
            patterns = [
                (
                    re.compile(
                        r"^\s*function\s+([A-Za-z_][A-Za-z0-9_-]*)\s*\{",
                        re.IGNORECASE,
                    ),
                    "function",
                ),
                (
                    re.compile(r"^\s*\$([A-Za-z_][A-Za-z0-9_]*)\s*=", re.IGNORECASE),
                    "variable",
                ),
            ]
        elif lang_name == "cmake":
            patterns = [
                (
                    re.compile(
                        r"^\s*(?:add_library|add_executable)\s*\(\s*([A-Za-z_][A-Za-z0-9_.-]*)",
                        re.IGNORECASE,
                    ),
                    "target",
                ),
                (
                    re.compile(
                        r"^\s*(?:function|macro)\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)",
                        re.IGNORECASE,
                    ),
                    "function",
                ),
                (
                    re.compile(r"^\s*set\s*\(\s*([A-Z][A-Z0-9_]+)", re.IGNORECASE),
                    "variable",
                ),
            ]
        elif lang_name in {"javascript", "typescript"}:
            patterns = [
                (
                    re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)"),
                    "class",
                ),
                (
                    re.compile(
                        r"^\s*export\s+(?:default\s+)?function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
                        re.IGNORECASE,
                    ),
                    "function",
                ),
                (
                    re.compile(r"^\s*function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("),
                    "function",
                ),
                (
                    re.compile(
                        r"^\s*(?:const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:async\s*)?\(?[^=]*=>"
                    ),
                    "function",
                ),
                (
                    re.compile(
                        r"^\s*export\s+default\s+function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
                        re.IGNORECASE,
                    ),
                    "function",
                ),
            ]
        elif lang_name in {"c", "cpp"}:
            patterns = [
                (
                    re.compile(
                        r"^\s*(?:class|struct|enum|namespace)\s+([A-Za-z_][A-Za-z0-9_:]*)\b"
                    ),
                    "class",
                ),
                (
                    re.compile(
                        r"^\s*(?:template\s*<[^>]+>\s*)?(?:inline\s+)?[A-Za-z_~][A-Za-z0-9_:<>\s\*&~,]*\s+([A-Za-z_~][A-Za-z0-9_:~]*)\s*\([^;]*\)\s*(?:const)?\s*[;{]"
                    ),
                    "function",
                ),
                (
                    re.compile(r"^\s*#define\s+([A-Z][A-Z0-9_]+)\b"),
                    "constant",
                ),
            ]
        elif lang_name in {"rust", "java", "csharp", "kotlin", "swift", "php"}:
            patterns = [
                (
                    re.compile(
                        r"^\s*(?:class|struct|interface|enum)\s+([A-Za-z_][A-Za-z0-9_]*)"
                    ),
                    "class",
                ),
                (
                    re.compile(
                        r"^\s*[A-Za-z_][A-Za-z0-9_:<>\s\*&]*\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^;]*\)\s*\{"
                    ),
                    "function",
                ),
                (
                    re.compile(r"^\s*func\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("),
                    "function",
                ),
            ]
        elif lang_name == "zig":
            patterns = [
                (
                    re.compile(r"^\s*(?:pub\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("),
                    "function",
                ),
                (
                    re.compile(
                        r"^\s*const\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:struct|enum|union)\b"
                    ),
                    "class",
                ),
                (
                    re.compile(r'^\s*test\s+"([^"]+)"'),
                    "test",
                ),
            ]
        elif lang_name == "nim":
            patterns = [
                (
                    re.compile(
                        r"^\s*(?:proc|func|method|template|macro)\s+([A-Za-z_][A-Za-z0-9_]*)"
                    ),
                    "function",
                ),
                (
                    re.compile(r"^\s*type\s+([A-Za-z_][A-Za-z0-9_]*)\s*\*?\s*="),
                    "class",
                ),
            ]
        elif lang_name == "hdl":
            patterns = [
                (
                    re.compile(r"^\s*module\s+([A-Za-z_][A-Za-z0-9_$]*)\b", re.IGNORECASE),
                    "module",
                ),
                (
                    re.compile(
                        r"^\s*interface\s+([A-Za-z_][A-Za-z0-9_$]*)\b", re.IGNORECASE
                    ),
                    "module",
                ),
                (
                    re.compile(r"^\s*entity\s+([A-Za-z_][A-Za-z0-9_]*)\s+is\b", re.IGNORECASE),
                    "module",
                ),
                (
                    re.compile(
                        r"^\s*package\s+([A-Za-z_][A-Za-z0-9_]*)\s+is\b", re.IGNORECASE
                    ),
                    "package",
                ),
            ]
        elif lang_name == "fortran":
            patterns = [
                (
                    re.compile(r"^\s*module\s+([A-Za-z_][A-Za-z0-9_]*)\b", re.IGNORECASE),
                    "module",
                ),
                (
                    re.compile(r"^\s*program\s+([A-Za-z_][A-Za-z0-9_]*)\b", re.IGNORECASE),
                    "module",
                ),
                (
                    re.compile(
                        r"^\s*(?:subroutine|function)\s+([A-Za-z_][A-Za-z0-9_]*)\b",
                        re.IGNORECASE,
                    ),
                    "function",
                ),
            ]
        elif lang_name == "pascal":
            patterns = [
                (
                    re.compile(
                        r"^\s*(?:unit|program)\s+([A-Za-z_][A-Za-z0-9_]*)\s*;?",
                        re.IGNORECASE,
                    ),
                    "module",
                ),
                (
                    re.compile(
                        r"^\s*(?:procedure|function)\s+([A-Za-z_][A-Za-z0-9_]*)",
                        re.IGNORECASE,
                    ),
                    "function",
                ),
                (
                    re.compile(r"^\s*type\s+([A-Za-z_][A-Za-z0-9_]*)\s*=", re.IGNORECASE),
                    "class",
                ),
            ]
        elif lang_name == "ada":
            patterns = [
                (
                    re.compile(
                        r"^\s*package\s+([A-Za-z_][A-Za-z0-9_.]*)\s+is\b", re.IGNORECASE
                    ),
                    "module",
                ),
                (
                    re.compile(
                        r"^\s*(?:procedure|function)\s+([A-Za-z_][A-Za-z0-9_]*)\b",
                        re.IGNORECASE,
                    ),
                    "function",
                ),
                (
                    re.compile(
                        r"^\s*task\s+type\s+([A-Za-z_][A-Za-z0-9_]*)\b", re.IGNORECASE
                    ),
                    "class",
                ),
            ]
        elif lang_name == "cobol":
            patterns = [
                (
                    re.compile(r"^\s*PROGRAM-ID\.\s+([A-Za-z0-9_-]+)\.?\s*$", re.IGNORECASE),
                    "module",
                ),
                (
                    re.compile(r"^\s*([A-Za-z0-9-]+)\s+SECTION\.\s*$", re.IGNORECASE),
                    "section",
                ),
            ]
        elif lang_name == "solidity":
            patterns = [
                (
                    re.compile(
                        r"^\s*(?:contract|interface|library)\s+([A-Za-z_][A-Za-z0-9_]*)\b"
                    ),
                    "class",
                ),
                (
                    re.compile(r"^\s*function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("),
                    "function",
                ),
                (
                    re.compile(r"^\s*event\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("),
                    "event",
                ),
                (
                    re.compile(r"^\s*struct\s+([A-Za-z_][A-Za-z0-9_]*)\b"),
                    "class",
                ),
            ]
        elif lang_name == "objective_c":
            patterns = [
                (
                    re.compile(r"^\s*@(?:interface|implementation|protocol)\s+([A-Za-z_][A-Za-z0-9_]*)"),
                    "class",
                ),
                (
                    re.compile(r"^\s*[+-]\s*\([^)]*\)\s*([A-Za-z_][A-Za-z0-9_:]*)"),
                    "method",
                ),
            ]
        elif lang_name == "assembly":
            patterns = [
                (
                    re.compile(r"^\s*([A-Za-z_.$][A-Za-z0-9_.$@]*)\s*:\s*(?:[#;].*)?$"),
                    "label",
                ),
                (
                    re.compile(r"^\s*\.macro\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE),
                    "macro",
                ),
            ]
        elif lang_name in {"vlang", "crystal", "dlang", "actionscript", "hack", "haxe"}:
            patterns = [
                (
                    re.compile(
                        r"^\s*(?:class|struct|interface|enum|trait|module|object|protocol|typedef)\s+([A-Za-z_][A-Za-z0-9_.:]*)"
                    ),
                    "class",
                ),
                (
                    re.compile(
                        r"^\s*(?:pub\s+)?(?:fn|def|function)\s+([A-Za-z_][A-Za-z0-9_!?]*)\s*\("
                    ),
                    "function",
                ),
            ]
        elif lang_name in {"ruby", "perl", "lua"}:
            patterns = [
                (
                    re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_!?=]*)"),
                    "function",
                ),
                (
                    re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_:]*)"),
                    "class",
                ),
            ]
        elif lang_name in {"r", "octave"}:
            patterns = [
                (
                    re.compile(
                        r"^\s*([A-Za-z_][A-Za-z0-9_.]*)\s*<-\s*function\s*\(",
                        re.IGNORECASE,
                    ),
                    "function",
                ),
            ]
        elif lang_name == "matlab":
            patterns = [
                (
                    re.compile(
                        r"^\s*function\s+(?:\[[^\]]+\]\s*=\s*|[A-Za-z_][A-Za-z0-9_]*\s*=\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*\(",
                        re.IGNORECASE,
                    ),
                    "function",
                ),
                (
                    re.compile(r"^\s*classdef\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE),
                    "class",
                ),
            ]
        elif lang_name == "julia":
            patterns = [
                (
                    re.compile(r"^\s*function\s+([A-Za-z_][A-Za-z0-9_!]*)\s*\("),
                    "function",
                ),
                (
                    re.compile(r"^\s*module\s+([A-Za-z_][A-Za-z0-9_]*)\b"),
                    "module",
                ),
            ]
        elif lang_name == "wolfram":
            patterns = [
                (
                    re.compile(r"^\s*([A-Za-z$][A-Za-z0-9$]*)\s*\[.*\]\s*:=", re.IGNORECASE),
                    "function",
                ),
                (
                    re.compile(r'^\s*BeginPackage\["([^"]+)"\]'),
                    "module",
                ),
            ]
        elif lang_name == "haskell":
            patterns = [
                (
                    re.compile(r"^\s*([a-z][A-Za-z0-9_']*)\s*::"),
                    "function",
                ),
                (
                    re.compile(r"^\s*data\s+([A-Z][A-Za-z0-9_']*)\b"),
                    "class",
                ),
            ]
        elif lang_name in {"scala", "fsharp"}:
            patterns = [
                (
                    re.compile(
                        r"^\s*(?:class|object|trait|type|module)\s+([A-Za-z_][A-Za-z0-9_.']*)",
                        re.IGNORECASE,
                    ),
                    "class",
                ),
                (
                    re.compile(r"^\s*(?:def|let)\s+([A-Za-z_][A-Za-z0-9_']*)", re.IGNORECASE),
                    "function",
                ),
            ]
        elif lang_name == "ocaml":
            patterns = [
                (
                    re.compile(r"^\s*let\s+(?:rec\s+)?([a-z_][A-Za-z0-9_']*)\b"),
                    "function",
                ),
                (
                    re.compile(r"^\s*module\s+([A-Z][A-Za-z0-9_]*)\b"),
                    "module",
                ),
            ]
        elif lang_name == "clojure":
            patterns = [
                (
                    re.compile(r"^\s*\(defn-?\s+([A-Za-z_][A-Za-z0-9_*!?<>\-]*)"),
                    "function",
                ),
                (
                    re.compile(r"^\s*\(def\s+([A-Za-z_][A-Za-z0-9_*!?<>\-]*)"),
                    "variable",
                ),
            ]
        elif lang_name == "prolog":
            patterns = [
                (
                    re.compile(r"^\s*:-\s*module\(([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE),
                    "module",
                ),
                (
                    re.compile(r"^\s*([a-z][A-Za-z0-9_]*)\s*\([^)]*\)\s*:-"),
                    "function",
                ),
            ]
        elif lang_name == "tcl":
            patterns = [
                (
                    re.compile(r"^\s*proc\s+([A-Za-z_][A-Za-z0-9_:]*)\s+\{", re.IGNORECASE),
                    "function",
                ),
                (
                    re.compile(r"^\s*namespace\s+eval\s+([A-Za-z_][A-Za-z0-9_:]*)", re.IGNORECASE),
                    "module",
                ),
            ]
        elif lang_name == "smalltalk":
            patterns = [
                (
                    re.compile(r"^\s*[A-Za-z_][A-Za-z0-9_]*\s+subclass:\s+#([A-Za-z_][A-Za-z0-9_]*)"),
                    "class",
                ),
            ]
        elif lang_name == "erlang":
            patterns = [
                (
                    re.compile(r"^\s*-module\(([A-Za-z_][A-Za-z0-9_]*)\)\."),
                    "module",
                ),
                (
                    re.compile(r"^\s*([a-z][A-Za-z0-9_]*)\s*\(.*\)\s*->"),
                    "function",
                ),
            ]
        elif lang_name == "elixir":
            patterns = [
                (
                    re.compile(r"^\s*defmodule\s+([A-Za-z_][A-Za-z0-9_.]*)\s+do"),
                    "module",
                ),
                (
                    re.compile(r"^\s*defp?\s+([a-z_][A-Za-z0-9_!?]*)\s*(?:\(|do)"),
                    "function",
                ),
            ]
        elif lang_name == "coldfusion":
            patterns = [
                (
                    re.compile(r"""<cfcomponent\b[^>]*(?:displayname|name)\s*=\s*["']([^"']+)["']""", re.IGNORECASE),
                    "module",
                ),
                (
                    re.compile(r"""<cffunction\b[^>]*name\s*=\s*["']([^"']+)["']""", re.IGNORECASE),
                    "function",
                ),
            ]
        elif lang_name == "gdscript":
            patterns = [
                (
                    re.compile(r"^\s*class_name\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE),
                    "class",
                ),
                (
                    re.compile(r"^\s*func\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.IGNORECASE),
                    "function",
                ),
            ]
        elif lang_name == "unrealscript":
            patterns = [
                (
                    re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\s+extends\b", re.IGNORECASE),
                    "class",
                ),
                (
                    re.compile(r"^\s*function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.IGNORECASE),
                    "function",
                ),
            ]
        elif lang_name == "sql":
            patterns = [
                (
                    re.compile(
                        r"^\s*create\s+(?:or\s+replace\s+)?(?:function|procedure|view|table)\s+([A-Za-z_][A-Za-z0-9_.]*)",
                        re.IGNORECASE,
                    ),
                    "symbol",
                )
            ]
        elif lang_name == "proto":
            patterns = [
                (
                    re.compile(
                        r"^\s*(?:message|service|enum)\s+([A-Za-z_][A-Za-z0-9_]*)"
                    ),
                    "type",
                )
            ]
        elif lang_name == "thrift":
            patterns = [
                (
                    re.compile(
                        r"^\s*(?:struct|service|enum|union|exception)\s+([A-Za-z_][A-Za-z0-9_]*)",
                        re.IGNORECASE,
                    ),
                    "type",
                )
            ]
        elif lang_name == "flatbuffers":
            patterns = [
                (
                    re.compile(
                        r"^\s*(?:table|struct|enum|rpc_service)\s+([A-Za-z_][A-Za-z0-9_]*)",
                        re.IGNORECASE,
                    ),
                    "type",
                )
            ]
        elif lang_name == "graphql":
            patterns = [
                (
                    re.compile(
                        r"^\s*(?:type|interface|enum|input|scalar|union)\s+([A-Za-z_][A-Za-z0-9_]*)"
                    ),
                    "type",
                )
            ]
        elif lang_name == "make":
            patterns = [
                (
                    re.compile(r"^\s*([A-Za-z0-9_.-]+)\s*:(?![=])"),
                    "target",
                ),
                (
                    re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*[:+?]?=\s*"),
                    "variable",
                ),
            ]
        elif lang_name == "docker":
            patterns = [
                (
                    re.compile(
                        r"^\s*(FROM|RUN|CMD|ENTRYPOINT|COPY|ADD|ENV|ARG|WORKDIR|EXPOSE)\b",
                        re.IGNORECASE,
                    ),
                    "instruction",
                )
            ]
        elif lang_name == "bazel":
            patterns = [
                (
                    re.compile(
                        r"^\s*(?:cc_|py_|java_|go_|rust_|sh_|genrule|test_suite)[A-Za-z0-9_]*\s*\(",
                        re.IGNORECASE,
                    ),
                    "rule",
                )
            ]
        elif lang_name in {"cypher", "aql", "mql"}:
            patterns = [
                (
                    re.compile(r"^\s*(MATCH|MERGE|CALL|FOR|LET|SELECT)\b", re.IGNORECASE),
                    "query",
                )
            ]
        elif lang_name in {"jinja", "django_template"}:
            patterns = [
                (
                    re.compile(r"""{%\s*block\s+([A-Za-z_][A-Za-z0-9_]*)\s*%}""", re.IGNORECASE),
                    "block",
                ),
                (
                    re.compile(r"""{%\s*macro\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(""", re.IGNORECASE),
                    "macro",
                ),
            ]
        elif lang_name == "ejs":
            patterns = [
                (
                    re.compile(r"""<%[-=]?\s*function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(""", re.IGNORECASE),
                    "function",
                )
            ]
        elif lang_name == "handlebars":
            patterns = [
                (
                    re.compile(r'\{\{#\*inline\s+"([^"]+)"', re.IGNORECASE),
                    "partial",
                )
            ]
        elif lang_name == "pug":
            patterns = [
                (
                    re.compile(r"^\s*mixin\s+([A-Za-z_][A-Za-z0-9_-]*)", re.IGNORECASE),
                    "macro",
                ),
                (
                    re.compile(r"^\s*block\s+([A-Za-z_][A-Za-z0-9_-]*)", re.IGNORECASE),
                    "block",
                ),
            ]
        elif lang_name in {"erb", "blade"}:
            patterns = [
                (
                    re.compile(r"""@section\(\s*['"]([^'"]+)['"]\s*\)""", re.IGNORECASE),
                    "block",
                ),
                (
                    re.compile(r"""<%[-=]?\s*def\s+([A-Za-z_][A-Za-z0-9_]*)""", re.IGNORECASE),
                    "function",
                ),
            ]
        elif lang_name == "html":
            patterns = [
                (
                    re.compile(r"""<[^>]*\sid=["']([A-Za-z_][A-Za-z0-9_-]*)["']"""),
                    "section",
                ),
                (
                    re.compile(r"<([a-z][a-z0-9_]*-[a-z0-9_-]+)\b", re.IGNORECASE),
                    "component",
                ),
            ]
        elif lang_name == "css":
            patterns = [
                (
                    re.compile(r"^\s*([.#][A-Za-z_][A-Za-z0-9_-]*)\s*\{"),
                    "selector",
                ),
                (
                    re.compile(r"^\s*@keyframes\s+([A-Za-z_][A-Za-z0-9_-]*)\b", re.IGNORECASE),
                    "animation",
                ),
                (
                    re.compile(r"^\s*@(?:media|supports|container)\s+([^{]+)", re.IGNORECASE),
                    "rule",
                ),
            ]
        elif lang_name == "mdx":
            patterns = [
                (
                    re.compile(r"^\s*export\s+(?:const|function)\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE),
                    "function",
                ),
                (
                    re.compile(r"^\s*#\s+(.+)$"),
                    "section",
                ),
            ]
        elif lang_name == "json":
            patterns = [
                (
                    re.compile(r'^\s*"([A-Za-z0-9_.-]+)"\s*:\s*'),
                    "key",
                )
            ]
        elif lang_name == "csv":
            patterns = [
                (
                    re.compile(r"^\s*([^,\n]+(?:,[^,\n]+)+)\s*$"),
                    "section",
                )
            ]
        elif lang_name == "yaml":
            patterns = [
                (
                    re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_.-]*)\s*:\s*(?:.+)?$"),
                    "key",
                )
            ]
        elif lang_name == "toml":
            patterns = [
                (
                    re.compile(r"^\s*\[([A-Za-z0-9_.-]+)\]\s*$"),
                    "section",
                ),
                (
                    re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_.-]*)\s*=\s*.+$"),
                    "key",
                ),
            ]
        elif lang_name == "xml":
            patterns = [
                (
                    re.compile(r"^\s*<(?![!?/])([A-Za-z_][A-Za-z0-9_.:-]*)\b"),
                    "element",
                )
            ]
        elif lang_name == "latex":
            patterns = [
                (
                    re.compile(r"""^\\(?:part|chapter|section|subsection|subsubsection)\{([^}]+)\}"""),
                    "section",
                ),
                (
                    re.compile(r"""^\\newcommand\{\\([^}]+)\}"""),
                    "macro",
                ),
            ]
        elif lang_name == "typst":
            patterns = [
                (
                    re.compile(r"^\s*=+\s+(.+)$"),
                    "section",
                ),
                (
                    re.compile(r"^\s*#let\s+([A-Za-z_][A-Za-z0-9_-]*)\s*=", re.IGNORECASE),
                    "variable",
                ),
            ]
        elif lang_name == "rst":
            patterns = [
                (
                    re.compile(r"^\s*\.\.\s+([A-Za-z0-9_-]+)::"),
                    "directive",
                )
            ]
        elif lang_name == "asciidoc":
            patterns = [
                (
                    re.compile(r"^\s*=+\s+(.+)$"),
                    "section",
                )
            ]
        elif lang_name == "org":
            patterns = [
                (
                    re.compile(r"^\s*\*+\s+(.+)$"),
                    "section",
                )
            ]
        elif lang_name == "config":
            patterns = [
                (
                    re.compile(r"^\s*\[([A-Za-z0-9_.-]+)\]\s*$"),
                    "section",
                ),
                (
                    re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_.-]*)\s*[:=]\s*.+$"),
                    "key",
                ),
            ]
        elif lang_name == "nix":
            patterns = [
                (
                    re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_-]*)\s*=\s*"),
                    "key",
                )
            ]
        elif lang_name == "hcl":
            patterns = [
                (
                    re.compile(
                        r'^\s*(?:resource|data|module|variable|output|provider)\s+"([^"]+)"',
                        re.IGNORECASE,
                    ),
                    "block",
                ),
                (
                    re.compile(r"^\s*(locals)\s*\{", re.IGNORECASE),
                    "block",
                ),
            ]
        elif lang_name == "puppet":
            patterns = [
                (
                    re.compile(r"^\s*(?:class|define|node)\s+([A-Za-z_][A-Za-z0-9_:]*)", re.IGNORECASE),
                    "class",
                )
            ]
        elif lang_name == "qml":
            patterns = [
                (
                    re.compile(r"^\s*([A-Z][A-Za-z0-9_]*)\s*\{"),
                    "component",
                ),
                (
                    re.compile(r"^\s*function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.IGNORECASE),
                    "function",
                ),
            ]
        elif lang_name == "vim":
            patterns = [
                (
                    re.compile(r"^\s*function!?\s+([A-Za-z0-9_#:]+)\s*\(", re.IGNORECASE),
                    "function",
                ),
                (
                    re.compile(r"^\s*command!?\s+([A-Za-z0-9_]+)\b", re.IGNORECASE),
                    "command",
                ),
            ]
        elif lang_name == "emacs_lisp":
            patterns = [
                (
                    re.compile(r"^\s*\(defun\s+([A-Za-z_][A-Za-z0-9_-]*)", re.IGNORECASE),
                    "function",
                ),
                (
                    re.compile(r"^\s*\(def(?:var|custom|const)\s+([A-Za-z_][A-Za-z0-9_-]*)", re.IGNORECASE),
                    "variable",
                ),
            ]
        elif lang_name == "awk":
            patterns = [
                (
                    re.compile(r"^\s*function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.IGNORECASE),
                    "function",
                )
            ]
        elif lang_name == "sed":
            patterns = [
                (
                    re.compile(r"^\s*:([A-Za-z_][A-Za-z0-9_]*)"),
                    "label",
                )
            ]
        elif lang_name == "wat":
            patterns = [
                (
                    re.compile(r"^\s*\((module)\b"),
                    "module",
                ),
                (
                    re.compile(r"^\s*\(func\s+\$([A-Za-z_][A-Za-z0-9_]*)"),
                    "function",
                ),
            ]
        else:
            patterns = []

        for index, line in enumerate(lines, start=1):
            for pattern, entity_type in patterns:
                match = pattern.search(line)
                if not match:
                    continue
                groups = [group for group in match.groups() if group]
                name = groups[0] if groups else match.group(0).strip()
                exports.append(name)
                block_end = (
                    min(len(lines), index + 20) if entity_type == "function" else index
                )
                snippet = "\n".join(lines[index - 1 : block_end])
                entities.append(
                    CodeEntity(
                        name=name,
                        type=entity_type,
                        content=snippet,
                        start_line=index,
                        end_line=max(index, block_end),
                        file_path=file_path,
                        metadata=self._entity_metadata(
                            file_path=file_path,
                            entity_name=name,
                            entity_type=entity_type,
                            content=snippet,
                        ),
                    )
                )
                break

        imports = self._extract_imports(None, content, lang_name)
        entities.extend(
            self._section_entities(
                file_path=file_path,
                content=content,
                base_entities=entities,
                imports=sorted(set(imports)),
            )
        )
        entities.append(
            self._file_summary_entity(
                file_path=file_path,
                content=content,
                imports=sorted(set(imports)),
                exports=exports,
            )
        )
        entities.append(
            CodeEntity(
                name="dependency_graph",
                type="dependency_graph",
                content=json.dumps(
                    self._build_dependency_payload(
                        entities=entities,
                        imports=sorted(set(imports)),
                        exports=exports,
                    )
                ),
                start_line=1,
                end_line=1,
                file_path=file_path,
                metadata={
                    "chunk_role": "dependency",
                    "file_role": self._file_role(file_path),
                    "path_terms": self._path_terms(file_path),
                    "doc_terms": self._extract_terms(" ".join(imports + exports), limit=64),
                },
            )
        )

        entities.append(
            CodeEntity(
                name=os.path.basename(file_path),
                type="file",
                content=content,
                start_line=1,
                end_line=max(1, len(lines)),
                file_path=file_path,
                metadata={
                    "chunk_role": "full",
                    "file_role": self._file_role(file_path),
                    "path_terms": self._path_terms(file_path),
                    "doc_terms": self._extract_terms(content, limit=64),
                },
            )
        )
        return entities

    def _tree_sitter_entity_name(self, *, node: Any, content: str) -> str:
        queue = list(getattr(node, "children", []) or [])
        while queue:
            current = queue.pop(0)
            if getattr(current, "type", "") in self._TREE_SITTER_NAME_NODE_TYPES:
                return content[current.start_byte : current.end_byte].strip()
            queue.extend(list(getattr(current, "children", []) or []))
        return ""

    def _tree_sitter_capture_name(
        self,
        *,
        node: Any,
        entity_type: str,
        captures: list[tuple[Any, str]],
        content: str,
    ) -> str:
        name_tag = f"{entity_type}.name"
        best_match = ""
        best_span = None
        for candidate, tag in captures:
            if tag != name_tag:
                continue
            if candidate.start_byte < node.start_byte or candidate.end_byte > node.end_byte:
                continue
            span = candidate.end_byte - candidate.start_byte
            if best_span is None or span < best_span:
                best_span = span
                best_match = content[candidate.start_byte : candidate.end_byte].strip()
        return best_match

    @staticmethod
    def _normalize_tree_sitter_entity_type(entity_type: str) -> str:
        normalized = str(entity_type or "").strip().lower()
        return {
            "fn": "function",
        }.get(normalized, normalized)

    def _build_dependency_payload(
        self,
        *,
        entities: list[CodeEntity],
        imports: list[str],
        exports: list[str],
    ) -> dict[str, object]:
        structural = [
            entity
            for entity in entities
            if entity.type not in {"dependency_graph", "file", "file_summary", "section"}
        ]
        export_names = list(dict.fromkeys([*exports, *[entity.name for entity in structural if entity.name]]))
        type_lookup = {entity.name: entity.type for entity in structural if entity.name}
        internal_edges: dict[tuple[str, str, str], dict[str, object]] = {}

        for entity in structural:
            if not entity.name:
                continue
            for target in export_names:
                if not target or target == entity.name:
                    continue
                relative_line = self._reference_line_offset(entity.content, target)
                if relative_line is None:
                    continue
                relation = self._dependency_relation(
                    source_type=entity.type,
                    target_type=type_lookup.get(target, ""),
                )
                internal_edges[(entity.name, target, relation)] = {
                    "from": entity.name,
                    "to": target,
                    "relation": relation,
                    "line": int(entity.start_line + relative_line),
                }

        return {
            "imports": sorted(set(imports)),
            "exports": export_names,
            "internal_edges": list(internal_edges.values()),
        }

    @staticmethod
    def _dependency_relation(*, source_type: str, target_type: str) -> str:
        callable_types = {"function", "method", "test"}
        if source_type in callable_types and target_type in callable_types:
            return "calls"
        return "references"

    def _reference_line_offset(self, snippet: str, symbol: str) -> int | None:
        variants = [str(symbol or "").strip()]
        if "." in symbol:
            variants.append(symbol.rsplit(".", 1)[-1])
        for candidate in dict.fromkeys([item for item in variants if item]):
            pattern = self._symbol_reference_pattern(candidate)
            for offset, line in enumerate(snippet.splitlines()):
                if pattern.search(line):
                    return offset
        return None

    @staticmethod
    def _symbol_reference_pattern(symbol: str) -> re.Pattern[str]:
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", symbol):
            return re.compile(rf"\b{re.escape(symbol)}\b")
        return re.compile(re.escape(symbol))

    def _safe_get_language(self, lang_name: str) -> Any:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Language\(path, name\) is deprecated.*",
                category=FutureWarning,
            )
            return self.get_language(lang_name)

    def _safe_get_parser(self, lang_name: str) -> Any:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Language\(path, name\) is deprecated.*",
                category=FutureWarning,
            )
            return self.get_parser(lang_name)

    def _tree_sitter_backend(self, lang_name: str) -> dict[str, str] | None:
        normalized = str(lang_name or "").strip().lower()
        backend = self._TREE_SITTER_BACKENDS.get(normalized)
        if not backend:
            return None
        return dict(backend)
