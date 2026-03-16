"""Utilities for coverage."""

import logging
import os

from saguaro.parsing.parser import SAGUAROParser
from saguaro.utils.file_utils import build_corpus_manifest

logger = logging.getLogger(__name__)


class CoverageReporter:
    """Provide CoverageReporter support."""

    def __init__(self, root_path: str) -> None:
        """Initialize the instance."""
        self.root_path = os.path.abspath(root_path)
        self.parser = SAGUAROParser()
        self.ext_map = {
            # Core & Systems
            ".py": "Python",
            ".pyi": "Python",
            ".pyx": "Python",
            ".pxd": "Python",
            ".ipynb": "Python",
            ".c": "C",
            ".h": "C/C++ Header",
            ".cpp": "C++",
            ".cc": "C++",
            ".cxx": "C++",
            ".hpp": "C++ Header",
            ".hh": "C++ Header",
            ".hxx": "C++ Header",
            ".m": "C++",
            ".mm": "C++",
            ".rs": "Rust",
            ".go": "Go",
            ".java": "Java",
            ".cs": "C#",
            ".swift": "Swift",
            ".kt": "Kotlin",
            ".kts": "Kotlin",
            ".scala": "Scala",
            ".sc": "Scala",
            # Web
            ".js": "JavaScript",
            ".jsx": "JavaScript",
            ".mjs": "JavaScript",
            ".ts": "TypeScript",
            ".tsx": "TypeScript",
            ".html": "HTML",
            ".htm": "HTML",
            ".css": "CSS",
            ".scss": "Sass",
            ".sass": "Sass",
            ".less": "Less",
            ".php": "PHP",
            ".phtml": "PHP",
            ".as": "ActionScript",
            ".vue": "Vue",
            ".svelte": "Svelte",
            ".dart": "Dart",
            ".hx": "Haxe",
            ".gd": "GDScript",
            ".uc": "UnrealScript",
            # Scripting
            ".rb": "Ruby",
            ".rake": "Ruby",
            ".pl": "Perl",
            ".pm": "Perl",
            ".lua": "Lua",
            ".sh": "Shell",
            ".bash": "Shell",
            ".zsh": "Shell",
            ".fish": "Shell",
            ".ps1": "PowerShell",
            ".psm1": "PowerShell",
            ".psd1": "PowerShell",
            ".bat": "Shell",
            ".cmd": "Shell",
            ".tcl": "Tcl",
            ".awk": "AWK",
            ".sed": "Sed",
            # Data/ML
            ".r": "R",
            ".jl": "Julia",
            ".sql": "SQL",
            ".m": "Objective-C",
            ".mat": "MATLAB",
            ".wl": "Mathematica",
            ".wls": "Mathematica",
            ".nb": "Mathematica",
            # Functional
            ".hs": "Haskell",
            ".erl": "Erlang",
            ".hrl": "Erlang",
            ".ex": "Elixir",
            ".exs": "Elixir",
            ".clj": "Clojure",
            ".cljs": "Clojure",
            ".cljc": "Clojure",
            ".edn": "Clojure",
            ".lisp": "Lisp",
            ".lsp": "Lisp",
            ".scm": "Lisp",
            ".rkt": "Lisp",
            ".ml": "OCaml",
            ".mli": "OCaml",
            ".fs": "F#",
            ".fsx": "F#",
            ".el": "Emacs Lisp",
            ".st": "Smalltalk",
            ".pl1": "PL/I",
            ".pli": "PL/I",
            # Config/Data Formats
            ".md": "Markdown",
            ".mdx": "MDX",
            ".json": "JSON",
            ".jsonc": "JSON",
            ".json5": "JSON",
            ".yaml": "YAML",
            ".yml": "YAML",
            ".toml": "TOML",
            ".xml": "XML",
            ".ini": "INI",
            ".conf": "Config",
            ".env": "Config",
            ".dockerfile": "Docker",
            ".cmake": "CMake",
            ".bazel": "Bazel",
            ".bzl": "Bazel",
            ".tf": "HCL",
            ".tfvars": "HCL",
            ".hcl": "HCL",
            ".nix": "Nix",
            ".qml": "QML",
            ".j2": "Jinja",
            ".jinja": "Jinja",
            ".jinja2": "Jinja",
            ".dtl": "Django Template",
            ".ejs": "EJS",
            ".hbs": "Handlebars",
            ".mustache": "Handlebars",
            ".pug": "Pug",
            ".jade": "Pug",
            ".erb": "ERB",
            ".tex": "LaTeX",
            ".sty": "LaTeX",
            ".cls": "LaTeX",
            ".typ": "Typst",
            ".csv": "CSV",
            ".adoc": "AsciiDoc",
            ".asciidoc": "AsciiDoc",
            ".org": "Org",
            ".gradle": "Config",
            ".sbt": "Config",
            # Legacy/Other
            ".f": "Fortran",
            ".f90": "Fortran",
            ".f95": "Fortran",
            ".asm": "Assembly",
            ".s": "Assembly",
            ".wat": "WebAssembly",
            ".cob": "COBOL",
            ".cbl": "COBOL",
            ".vb": "Visual Basic",
            ".pas": "Pascal",
            ".pp": "Pascal",
            ".ada": "Ada",
            ".d": "D",
            ".cr": "Crystal",
            ".rpg": "RPG",
            ".rpgle": "RPG",
            ".sol": "Solidity",
            ".vy": "Solidity",
            ".graphql": "GraphQL",
            ".proto": "Protocol Buffer",
            ".thrift": "Protocol Buffer",
            ".fbs": "FlatBuffers",
            ".cypher": "Cypher",
            ".cql": "Cypher",
            ".aql": "AQL",
            ".mql": "MQL",
            ".vim": "Vim Script",
            ".txt": "Text",
            ".in": "Text",
        }
        self.filename_map = {
            "CMakeLists.txt": "CMake",
            "Dockerfile": "Docker",
            "Makefile": "Make",
            "Gnumakefile": "Make",
            "BUILD": "Bazel",
            "BUILD.bazel": "Bazel",
            "WORKSPACE": "Bazel",
            "WORKSPACE.bazel": "Bazel",
            "meson.build": "CMake",
            "meson_options.txt": "CMake",
            "justfile": "Make",
            "Jenkinsfile": "Groovy",
            "Vagrantfile": "Ruby",
            "Rakefile": "Ruby",
            "Gemfile": "Ruby Gemfile",
            "Procfile": "Procfile",
            "requirements.txt": "Pip Requirements",
            "Pipfile": "Pipfile",
            "pyproject.toml": "Python Project",
            "Cargo.toml": "Rust Crate",
            "package.json": "NPM Package",
            "go.mod": "Go Module",
            "go.sum": "Go Sum",
            "LICENSE": "Text",
            "NOTICE": "Text",
            "AUTHORS": "Text",
            "OWNERS": "Text",
            ".gitignore": "Git Ignore",
            ".dockerignore": "Docker Ignore",
            ".vimrc": "Vim Script",
            ".emacs": "Emacs Lisp",
        }
        self.excludes = {
            ".git",
            ".venv",
            "venv",
            "node_modules",
            "repo_analysis",
            "__pycache__",
            "build",
            "dist",
            ".idea",
            ".vscode",
            ".saguaro",
            ".ruff_cache",
            ".pytest_cache",
            ".mypy_cache",
            ".git_cache",
        }
        self.structural_supported_languages = {
            "Ada",
            "C",
            "C#",
            "C++",
            "C/C++ Header",
            "C++ Header",
            "COBOL",
            "CMake",
            "CSS",
            "Dart",
            "Docker",
            "Elixir",
            "Erlang",
            "Fortran",
            "F#",
            "Go",
            "GraphQL",
            "Groovy",
            "Haskell",
            "HTML",
            "INI",
            "Java",
            "JavaScript",
            "Julia",
            "Kotlin",
            "Less",
            "Lisp",
            "Lua",
            "Make",
            "OCaml",
            "Pascal",
            "Perl",
            "PHP",
            "PowerShell",
            "Protocol Buffer",
            "Python",
            "R",
            "Ruby",
            "Rust",
            "Sass",
            "Scala",
            "Shell",
            "Solidity",
            "SQL",
            "Svelte",
            "Swift",
            "TypeScript",
            "Visual Basic",
            "Vue",
            "XML",
            "YAML",
        }
        self.language_parser_map = {
            "Ada": "ada",
            "Bazel": "bazel",
            "C": "c",
            "C#": "csharp",
            "COBOL": "cobol",
            "C++": "cpp",
            "C/C++ Header": "cpp",
            "C++ Header": "cpp",
            "CMake": "cmake",
            "CSV": "csv",
            "CSS": "css",
            "Config": "config",
            "Dart": "swift",
            "Docker": "docker",
            "ActionScript": "actionscript",
            "AQL": "aql",
            "AsciiDoc": "asciidoc",
            "Assembly": "assembly",
            "AWK": "awk",
            "ColdFusion": "coldfusion",
            "Crystal": "crystal",
            "Cypher": "cypher",
            "D": "dlang",
            "Django Template": "django_template",
            "EJS": "ejs",
            "Emacs Lisp": "emacs_lisp",
            "Elixir": "elixir",
            "Erlang": "erlang",
            "ERB": "erb",
            "FlatBuffers": "flatbuffers",
            "Fortran": "fortran",
            "F#": "fsharp",
            "GDScript": "gdscript",
            "Go": "go",
            "GraphQL": "graphql",
            "Groovy": "java",
            "Hack": "hack",
            "Handlebars": "handlebars",
            "Haskell": "haskell",
            "HCL": "hcl",
            "Haxe": "haxe",
            "HTML": "html",
            "INI": "config",
            "Java": "java",
            "JavaScript": "javascript",
            "Jinja": "jinja",
            "Julia": "julia",
            "Kotlin": "kotlin",
            "LaTeX": "latex",
            "Less": "css",
            "Lisp": "lisp",
            "Lua": "lua",
            "Make": "make",
            "MATLAB": "matlab",
            "Mathematica": "wolfram",
            "MDX": "mdx",
            "MQL": "mql",
            "Nix": "nix",
            "OCaml": "ocaml",
            "Objective-C": "objective_c",
            "Org": "org",
            "Pascal": "pascal",
            "Perl": "perl",
            "PHP": "php",
            "PL/I": "pli",
            "PowerShell": "powershell",
            "Prolog": "prolog",
            "Protocol Buffer": "proto",
            "Python": "python",
            "Pug": "pug",
            "Puppet": "puppet",
            "QML": "qml",
            "R": "r",
            "RPG": "rpg",
            "Ruby": "ruby",
            "Rust": "rust",
            "RST": "rst",
            "Sass": "css",
            "Scala": "scala",
            "Sed": "sed",
            "Shell": "shell",
            "Smalltalk": "smalltalk",
            "Solidity": "solidity",
            "SQL": "sql",
            "Svelte": "javascript",
            "Swift": "swift",
            "Tcl": "tcl",
            "Thrift": "thrift",
            "Typst": "typst",
            "TypeScript": "typescript",
            "UnrealScript": "unrealscript",
            "Visual Basic": "csharp",
            "Vim Script": "vim",
            "Vue": "javascript",
            "WebAssembly": "wat",
            "XML": "xml",
            "YAML": "yaml",
            "JSON": "json",
            "TOML": "toml",
        }

    def generate_report(
        self,
        *,
        structural: bool = False,
        by_language: bool = False,
    ) -> dict:
        """Handle generate report."""
        stats = {
            "total_files": 0,
            "languages": {},
            "ast_supported_files": 0,
            "dependency_supported_files": 0,
            "dependency_quality_supported_files": 0,
            "structural_supported_files": 0,
            "blind_files": 0,
            "blind_list": [],
        }

        # Check parser capabilities
        ts_available = False
        try:
            import importlib.util

            ts_available = importlib.util.find_spec("tree_sitter") is not None
        except ImportError:
            pass

        if by_language:
            stats["language_breakdown"] = {}

        manifest = build_corpus_manifest(self.root_path)
        for full_path in manifest.files:
            file = os.path.basename(full_path)
            ext = os.path.splitext(file)[1]

            lang = None
            if file in self.filename_map:
                lang = self.filename_map[file]
            elif ext in self.ext_map:
                lang = self.ext_map[ext]

            if lang:
                stats["languages"][lang] = stats["languages"].get(lang, 0) + 1
                stats["total_files"] += 1
                if self._is_ast_supported(lang, ts_available):
                    stats["ast_supported_files"] += 1
                if self._is_dependency_supported(lang):
                    stats["dependency_supported_files"] += 1
                if self._is_dependency_quality_supported(lang):
                    stats["dependency_quality_supported_files"] += 1
                if self._is_structural_supported(lang):
                    stats["structural_supported_files"] += 1
                if by_language:
                    lang_row = stats["language_breakdown"].setdefault(
                        lang,
                        {
                            "files": 0,
                            "ast_supported_files": 0,
                            "dependency_supported_files": 0,
                            "dependency_quality_supported_files": 0,
                            "structural_supported_files": 0,
                        },
                    )
                    lang_row["files"] += 1
                    if self._is_ast_supported(lang, ts_available):
                        lang_row["ast_supported_files"] += 1
                    if self._is_dependency_supported(lang):
                        lang_row["dependency_supported_files"] += 1
                    if self._is_dependency_quality_supported(lang):
                        lang_row["dependency_quality_supported_files"] += 1
                    if self._is_structural_supported(lang):
                        lang_row["structural_supported_files"] += 1
            else:
                stats["blind_files"] += 1
                if len(stats["blind_list"]) < 20:
                    stats["blind_list"].append(os.path.relpath(full_path, self.root_path))

        total = int(stats.get("total_files", 0) or 0)
        ast = int(stats.get("ast_supported_files", 0) or 0)
        dependency = int(stats.get("dependency_supported_files", 0) or 0)
        dependency_quality = int(stats.get("dependency_quality_supported_files", 0) or 0)
        structural_count = int(stats.get("structural_supported_files", 0) or 0)
        stats["ast_coverage_percent"] = round((ast / total) * 100, 1) if total else 0.0
        stats["dependency_coverage_percent"] = (
            round((dependency / total) * 100, 1) if total else 0.0
        )
        stats["dependency_quality_coverage_percent"] = (
            round((dependency_quality / total) * 100, 1) if total else 0.0
        )
        stats["structural_coverage_percent"] = (
            round((structural_count / total) * 100, 1) if total else 0.0
        )
        stats["coverage_percent"] = stats["dependency_quality_coverage_percent"]
        stats["requested_coverage_metric"] = (
            "structural" if structural else "dependency_quality"
        )
        stats["requested_coverage_percent"] = (
            stats["structural_coverage_percent"]
            if structural
            else stats["dependency_quality_coverage_percent"]
        )
        return stats

    def print_report(
        self, *, structural: bool = False, by_language: bool = False
    ) -> None:
        """Handle print report."""
        stats = self.generate_report(structural=structural, by_language=by_language)
        total = stats["total_files"]
        ast = stats["ast_supported_files"]
        dependency_quality = stats.get("dependency_quality_supported_files", 0)
        structural_count = stats.get("structural_supported_files", 0)

        print("\n=== SAGUARO Coverage Report ===")
        print(f"Total Tracked Files: {total}")

        print("\n[Languages Detected]")
        for lang, count in sorted(
            stats["languages"].items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {lang}: {count}")

        print("\n[AST Coverage]")
        if total > 0:
            coverage = (ast / total) * 100
            print(f"  Coverage: {coverage:.1f}% ({ast}/{total} files)")
        else:
            print("  Coverage: N/A")

        print("\n[Structural Coverage]")
        if total > 0:
            coverage = (structural_count / total) * 100
            print(f"  Coverage: {coverage:.1f}% ({structural_count}/{total} files)")
        else:
            print("  Coverage: N/A")

        print("\n[Dependency Quality Coverage]")
        if total > 0:
            coverage = (dependency_quality / total) * 100
            print(f"  Coverage: {coverage:.1f}% ({dependency_quality}/{total} files)")
        else:
            print("  Coverage: N/A")

        if by_language:
            print("\n[By Language]")
            for lang, row in sorted(
                stats.get("language_breakdown", {}).items(),
                key=lambda item: item[0].lower(),
            ):
                print(
                    f"  {lang}: files={row['files']}, "
                    f"structural={row['structural_supported_files']}, "
                    f"dependency_quality={row['dependency_quality_supported_files']}, "
                    f"ast={row['ast_supported_files']}"
                )

        print("\n[Blind Spots (Unsupported Files)]")
        print(f"  Total Unknown: {stats['blind_files']}")
        if stats["blind_list"]:
            print("  Sample:")
            for f in stats["blind_list"]:
                print(f"    - {f}")

    def _is_structural_supported(self, language: str) -> bool:
        parser_language = self._parser_language(language)
        if parser_language:
            return self.parser.supports_structural_language(parser_language)
        return language in self.structural_supported_languages

    def _is_ast_supported(self, language: str, tree_sitter_available: bool) -> bool:
        _ = tree_sitter_available
        parser_language = self._parser_language(language)
        return self.parser.supports_ast_language(parser_language)

    def _is_dependency_supported(self, language: str) -> bool:
        parser_language = self._parser_language(language)
        return self.parser.supports_dependency_graph_language(parser_language)

    def _is_dependency_quality_supported(self, language: str) -> bool:
        parser_language = self._parser_language(language)
        return self.parser.supports_dependency_quality_language(parser_language)

    def _parser_language(self, language: str) -> str:
        return self.language_parser_map.get(language, str(language or "").strip().lower())
