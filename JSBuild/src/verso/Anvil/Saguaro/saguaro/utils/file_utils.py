"""Utilities for repository-aware file discovery."""

import os
import shutil
import subprocess
import sys
from collections.abc import Generator
from dataclasses import dataclass, field

CODE_EXTENSIONS = {
    # Python ecosystem
    ".py",
    ".pyi",
    ".pyx",
    ".pxd",
    ".ipynb",
    # C / C++ / Objective-C
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".m",
    ".mm",
    # JavaScript / TypeScript / React
    ".js",
    ".mjs",
    ".cjs",
    ".jsx",
    ".ts",
    ".tsx",
    ".vue",
    ".svelte",
    ".astro",
    # JVM / .NET / mobile
    ".java",
    ".kt",
    ".kts",
    ".scala",
    ".sc",
    ".groovy",
    ".cs",
    ".fs",
    ".fsx",
    ".vb",
    ".swift",
    ".dart",
    ".hx",
    ".gd",
    ".uc",
    ".as",
    # Systems / native / hardware
    ".go",
    ".rs",
    ".zig",
    ".nim",
    ".cr",
    ".d",
    ".asm",
    ".wat",
    ".v",
    ".sv",
    ".svh",
    ".vhd",
    ".vhdl",
    # Scripting / shell
    ".rb",
    ".rake",
    ".php",
    ".phtml",
    ".pl",
    ".pm",
    ".lua",
    ".sh",
    ".bash",
    ".zsh",
    ".fish",
    ".ps1",
    ".psm1",
    ".psd1",
    ".bat",
    ".cmd",
    ".tcl",
    ".awk",
    ".sed",
    # Functional and logic languages
    ".hs",
    ".lhs",
    ".ml",
    ".mli",
    ".clj",
    ".cljs",
    ".cljc",
    ".edn",
    ".lisp",
    ".lsp",
    ".scm",
    ".rkt",
    ".el",
    ".st",
    ".erl",
    ".hrl",
    ".ex",
    ".exs",
    # Scientific / data / schemas
    ".r",
    ".R",
    ".jl",
    ".sql",
    ".proto",
    ".thrift",
    ".fbs",
    ".graphql",
    ".gql",
    ".cypher",
    ".cql",
    ".aql",
    ".ql",
    ".qll",
    ".dbscheme",
    ".bzl",
    ".mql",
    ".csv",
    # Legacy / other languages
    ".f",
    ".f90",
    ".f95",
    ".pas",
    ".pp",
    ".adb",
    ".ads",
    ".ada",
    ".cob",
    ".cbl",
    ".rpg",
    ".rpgle",
    ".pli",
    ".pl1",
    ".sol",
    ".vy",
    # Markup / style / config
    ".html",
    ".css",
    ".scss",
    ".sass",
    ".less",
    ".json",
    ".jsonc",
    ".json5",
    ".yaml",
    ".yml",
    ".toml",
    ".xml",
    ".qml",
    ".cmake",
    ".ini",
    ".cfg",
    ".conf",
    ".env",
    ".properties",
    ".gradle",
    ".sbt",
    ".bazel",
    ".bzl",
    ".tf",
    ".tfvars",
    ".hcl",
    ".nix",
    ".vim",
    ".j2",
    ".jinja",
    ".jinja2",
    ".dtl",
    ".ejs",
    ".hbs",
    ".mustache",
    ".pug",
    ".jade",
    ".erb",
    ".tex",
    ".sty",
    ".cls",
    ".typ",
    ".adoc",
    ".asciidoc",
    ".org",
    # Documentation
    ".md",
    ".mdx",
    ".rst",
    ".txt",
}

CODE_FILENAMES = {
    "Dockerfile",
    "Makefile",
    "CMakeLists.txt",
    "BUILD",
    "BUILD.bazel",
    "WORKSPACE",
    "WORKSPACE.bazel",
    "meson.build",
    "meson_options.txt",
    "Rakefile",
    "Gemfile",
    "Gnumakefile",
    "Jenkinsfile",
    "Procfile",
    "Vagrantfile",
    "justfile",
    ".vimrc",
    ".emacs",
}


def resolve_saguaro_binary(repo_path: str = ".") -> str | list[str]:
    """Resolve the saguaro executable using a priority fallback chain.

    Returns either a string (direct binary path) or a list of strings
    (e.g. [sys.executable, "-m", "saguaro.cli"]) for subprocess invocation.

    Resolution order:
      1. PATH lookup (global install via pip install / pipx)
      2. Repo-local venv (legacy per-repo venv)
      3. sys.executable -m saguaro.cli (always works if package is importable)
    """
    # 1. Global PATH lookup
    which = shutil.which("saguaro")
    if which:
        return which

    # 2. Repo-local venv (legacy support)
    repo_abs = os.path.abspath(repo_path)
    for venv_dir in ("venv", ".venv"):
        candidate = os.path.join(repo_abs, venv_dir, "bin", "saguaro")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    # 3. Module invocation fallback (always available)
    return [sys.executable, "-m", "saguaro.cli"]


def resolve_saguaro_cmd(repo_path: str = ".") -> list[str]:
    """Like resolve_saguaro_binary but always returns a list for subprocess use."""
    result = resolve_saguaro_binary(repo_path)
    if isinstance(result, list):
        return result
    return [result]


@dataclass(slots=True)
class CorpusManifest:
    """Canonical repository file manifest shared across Saguaro surfaces."""

    root_path: str
    files: list[str]
    source: str
    candidate_count: int
    excluded_count: int
    excluded_roots: dict[str, int] = field(default_factory=dict)


def _is_code_candidate(path: str) -> bool:
    name = os.path.basename(path)
    _, ext = os.path.splitext(name)
    return name in CODE_FILENAMES or ext in CODE_EXTENSIONS


def _git_toplevel(root_path: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", root_path, "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    top = result.stdout.strip()
    return os.path.abspath(top) if top else None


def _git_ls_files(root_path: str) -> list[str] | None:
    top = _git_toplevel(root_path)
    if not top:
        return None
    try:
        result = subprocess.run(
            ["git", "-C", top, "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            check=True,
            capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    target_root = os.path.abspath(root_path)
    prefix = "" if target_root == top else os.path.relpath(target_root, top).replace("\\", "/").rstrip("/") + "/"
    discovered: list[str] = []
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        rel = raw.decode("utf-8", errors="ignore").replace("\\", "/")
        if prefix and not rel.startswith(prefix):
            continue
        abs_path = os.path.join(top, rel)
        if os.path.isfile(abs_path) and _is_code_candidate(abs_path):
            discovered.append(os.path.abspath(abs_path))
    resolved_root = os.path.abspath(root_path)
    if discovered or resolved_root == os.path.abspath(top):
        return sorted(set(discovered))
    # Nested external repos under an ignored parent tree need a raw walk fallback.
    return None


def _walk_files(root_path: str, exclusions: list[str] | None = None) -> list[str]:
    from saguaro.query.corpus_rules import canonicalize_rel_path, is_excluded_path

    resolved_root = os.path.abspath(root_path)
    discovered: list[str] = []
    for root, dirs, files in os.walk(resolved_root):
        pruned_dirs: list[str] = []
        for name in list(dirs):
            candidate_dir = os.path.join(root, name)
            rel = canonicalize_rel_path(candidate_dir, repo_path=resolved_root)
            if is_excluded_path(rel, patterns=exclusions, repo_path=resolved_root):
                continue
            pruned_dirs.append(name)
        dirs[:] = pruned_dirs
        for name in files:
            candidate = os.path.join(root, name)
            rel = canonicalize_rel_path(candidate, repo_path=resolved_root)
            if is_excluded_path(rel, patterns=exclusions, repo_path=resolved_root):
                continue
            if _is_code_candidate(candidate):
                discovered.append(os.path.abspath(candidate))
    return sorted(set(discovered))


def build_corpus_manifest(
    root_path: str,
    exclusions: list[str] | None = None,
) -> CorpusManifest:
    """Build a .gitignore-aware manifest with Saguaro policy exclusions applied."""
    from saguaro.query.corpus_rules import canonicalize_rel_path, filter_indexable_files

    resolved_root = os.path.abspath(root_path)
    extra_patterns = list(exclusions or [])
    git_candidates = _git_ls_files(resolved_root)
    raw_candidates = (
        git_candidates
        if git_candidates is not None
        else _walk_files(resolved_root, exclusions=extra_patterns)
    )
    filtered = filter_indexable_files(
        raw_candidates,
        repo_path=resolved_root,
        patterns=extra_patterns,
    )

    excluded_roots: dict[str, int] = {}
    filtered_set = set(filtered)
    for candidate in raw_candidates:
        if candidate in filtered_set:
            continue
        rel = canonicalize_rel_path(candidate, repo_path=resolved_root)
        root = rel.split("/", 1)[0] if rel else ""
        if root:
            excluded_roots[root] = excluded_roots.get(root, 0) + 1

    return CorpusManifest(
        root_path=resolved_root,
        files=filtered,
        source="git" if git_candidates is not None else "walk",
        candidate_count=len(raw_candidates),
        excluded_count=max(0, len(raw_candidates) - len(filtered)),
        excluded_roots=dict(sorted(excluded_roots.items())),
    )


def get_code_files(root_path: str, exclusions: list[str] = None) -> list[str]:
    """Return repository code/config files using the shared corpus manifest."""
    return build_corpus_manifest(root_path, exclusions=exclusions).files


def iter_code_files(
    root_path: str, exclusions: list = None, batch_size: int = 64
) -> Generator[list, None, None]:
    """Generator-based file discovery that yields batches of files.

    MEMORY OPTIMIZED: Does not hold all file paths in memory at once.
    Files are discovered and yielded in batches for immediate processing.

    Args:
        root_path: Directory to scan
        exclusions: Additional patterns to exclude
        batch_size: Number of files per batch

    Yields:
        Lists of file paths, each containing up to batch_size files
    """
    manifest = build_corpus_manifest(root_path, exclusions=exclusions)
    batch = []
    for file_path in manifest.files:
        batch.append(file_path)
        if len(batch) >= batch_size:
            yield batch
            batch = []

    # Yield remaining files
    if batch:
        yield batch
