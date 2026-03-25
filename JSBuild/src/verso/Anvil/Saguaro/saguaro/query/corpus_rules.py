"""Canonical corpus filtering and file-role helpers for SAGUARO."""

from __future__ import annotations

import fnmatch
import os
from functools import lru_cache

import yaml

DEFAULT_EXCLUDE_PATTERNS = [
    ".anvil/**",
    ".anvil_backups/**",
    ".granite/**",
    ".granite_backups/**",
    ".saguaro/**",
    "core/native/build/**",
    "core/native/split/**",
    "core/native/*_wrapper.py",
    "repo_analysis/**",
    "Saguaro/saguaro/artifacts/codebooks/**",
    "saguaro_restored+temp/**",
    "_legacy_saguaro_to_remove/**",
    "autoresearch-master/**",
    "audit/**/checkpoint.json",
    "audit/**/manifest.json",
    "audit/**/summary.json",
    "audit/**/*.ndjson",
    "audit/**/*.db",
    "benchmarks/**/*.json",
    "*.egg-info/**",
]


def _match_aliases(rel: str) -> list[str]:
    aliases = [rel]
    if rel.startswith("Saguaro/saguaro/"):
        aliases.append("saguaro/" + rel[len("Saguaro/saguaro/") :])
    elif rel.startswith("Saguaro/"):
        aliases.append(rel[len("Saguaro/") :])
    return aliases


def canonicalize_rel_path(value: str, repo_path: str | None = None) -> str:
    """Normalize a repository-relative path without collapsing authoritative trees."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    if repo_path and os.path.isabs(raw):
        rel = os.path.relpath(raw, repo_path)
    else:
        rel = raw
    if len(rel) >= 2 and rel[0] == rel[-1] == '"':
        rel = rel[1:-1]
    rel = rel.replace("\\", "/")
    parts = [part for part in rel.split("/") if part not in {"", "."}]
    return "/".join(parts)


@lru_cache(maxsize=32)
def _repo_policy_patterns(repo_path: str) -> tuple[str, ...]:
    policy_path = os.path.join(repo_path, "standards", "scan_exclusion_policy.yaml")
    if not os.path.exists(policy_path):
        return ()
    try:
        with open(policy_path, encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    except Exception:
        return ()
    globs = payload.get("exclude_globs")
    if not isinstance(globs, list):
        return ()
    return tuple(str(item).strip() for item in globs if str(item).strip())


@lru_cache(maxsize=32)
def _config_exclusion_patterns(repo_path: str) -> tuple[str, ...]:
    config_path = os.path.join(repo_path, ".saguaro", "config.yaml")
    if not os.path.exists(config_path):
        return ()
    try:
        with open(config_path, encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    except Exception:
        return ()
    configured = (((payload.get("indexing") or {}).get("exclude")) if isinstance(payload, dict) else None) or []
    if not isinstance(configured, list):
        return ()
    return tuple(str(item).strip() for item in configured if str(item).strip())


def load_corpus_patterns(
    repo_path: str | None = None,
    patterns: list[str] | None = None,
) -> list[str]:
    """Return the authoritative exclusion list for the current repository."""
    merged = list(DEFAULT_EXCLUDE_PATTERNS)
    if repo_path:
        resolved = os.path.abspath(repo_path)
        merged.extend(_repo_policy_patterns(resolved))
        merged.extend(_config_exclusion_patterns(resolved))
    if patterns:
        merged.extend(str(item).strip() for item in patterns if str(item).strip())
    deduped: list[str] = []
    seen: set[str] = set()
    for pattern in merged:
        if pattern in seen:
            continue
        seen.add(pattern)
        deduped.append(pattern)
    return deduped


def is_excluded_path(
    path: str,
    patterns: list[str] | None = None,
    repo_path: str | None = None,
) -> bool:
    """Return True when a repository-relative path should be excluded."""
    rel = canonicalize_rel_path(path, repo_path=repo_path)
    if not rel:
        return True
    candidates = load_corpus_patterns(repo_path=repo_path, patterns=patterns)
    aliases = _match_aliases(rel)
    return any(
        fnmatch.fnmatch(alias, pattern)
        for alias in aliases
        for pattern in candidates
    )


def filter_indexable_files(
    paths: list[str],
    repo_path: str,
    patterns: list[str] | None = None,
) -> list[str]:
    """Filter file paths down to canonical index candidates."""
    filtered: list[str] = []
    resolved_repo = os.path.abspath(repo_path)
    for path in paths:
        rel = canonicalize_rel_path(path, repo_path=resolved_repo)
        if not rel or is_excluded_path(rel, patterns=patterns, repo_path=resolved_repo):
            continue
        filtered.append(os.path.join(resolved_repo, rel))
    return sorted(set(filtered))


def corpus_manifest(
    repo_path: str,
    patterns: list[str] | None = None,
):
    """Build the shared corpus manifest for indexing and analysis surfaces."""
    from saguaro.utils.file_utils import build_corpus_manifest

    resolved_repo = os.path.abspath(repo_path)
    return build_corpus_manifest(
        resolved_repo,
        exclusions=load_corpus_patterns(resolved_repo, patterns=patterns),
    )


def classify_file_role(path: str) -> str:
    """Classify a repository-relative path for retrieval priors."""
    rel = canonicalize_rel_path(path)
    lowered = rel.lower()
    if not rel:
        return "unknown"
    if lowered.startswith(("tests/", "test/")) or "/tests/" in lowered:
        return "test"
    if lowered.startswith(("docs/", "doc/")) or rel.endswith(
        (".md", ".mdx", ".rst", ".txt", ".adoc", ".asciidoc", ".org", ".tex", ".typ")
    ):
        return "doc"
    if lowered.startswith(("audit/", "benchmarks/")):
        return "bench"
    if lowered.startswith(("core/", "agents/", "cli/", "config/", "tools/")):
        return "source"
    if lowered.endswith(
        (".json", ".jsonc", ".json5", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".tf", ".tfvars", ".hcl", ".nix")
    ):
        return "config"
    return "source"
