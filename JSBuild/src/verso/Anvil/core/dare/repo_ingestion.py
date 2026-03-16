"""Repository ingestion and profiling for DARE."""

from __future__ import annotations

import os
from collections import Counter
from typing import Any, Dict, List, Optional

from core.dare.knowledge_base import DareKnowledgeBase
from core.dare.models import RepoProfile
from saguaro.services.comparative import ComparativeAnalysisService

LANGUAGE_MAP = {
    ".py": "python",
    ".c": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".h": "c_header",
    ".hpp": "cpp_header",
    ".md": "markdown",
    ".json": "json",
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


class RepoIngestionEngine:
    """Ingest and analyze one or more local repositories."""

    def __init__(
        self,
        repos: List[str],
        knowledge_base: DareKnowledgeBase,
        repo_roles: Optional[Dict[str, str]] = None,
    ):
        self.repos = [os.path.abspath(path) for path in repos]
        self.kb = knowledge_base
        self.repo_roles = {
            os.path.abspath(path): role for path, role in (repo_roles or {}).items()
        }

    def ingest_all(self) -> Dict[str, RepoProfile]:
        profiles: Dict[str, RepoProfile] = {}
        for repo in self.repos:
            profiles[repo] = self._analyze_single_repo(repo)
        if len(profiles) > 1:
            self._cross_repo_analysis(profiles)
        return profiles

    def _analyze_single_repo(self, repo: str) -> RepoProfile:
        role = self.repo_roles.get(repo, "analysis")
        comparative = ComparativeAnalysisService(repo)
        created = comparative.create_session(
            path=repo,
            alias=os.path.basename(repo),
            quarantine=role != "target",
            trust_level="high" if role == "target" else "medium",
            build_profile="auto",
            rebuild=False,
        )
        pack = dict(
            (
                comparative.corpus(
                    action="show",
                    corpus_id=str(created["session"]["corpus_id"]),
                ).get("analysis_pack")
                or {}
            )
        )
        language_counts: Counter[str] = Counter(pack.get("languages", {}))
        entry_points = list(pack.get("entry_points", []))
        modules: List[str] = []
        test_files = list(pack.get("test_files", []))
        build_files = list(pack.get("build_files", []))
        dependency_graph: Dict[str, List[str]] = {}
        patterns = {
            "todo_comments": 0,
            "fixme_comments": 0,
            "bare_except": 0,
            "print_statements": 0,
        }
        loc = int(pack.get("loc", 0))
        file_count = int(pack.get("file_count", 0))
        tech_stack = set(pack.get("tech_stack", []))

        for file_record in pack.get("files", []):
            rel_path = str(file_record.get("path") or "")
            if file_record.get("language") == "python":
                for symbol in file_record.get("symbols", []):
                    if symbol.get("kind") == "class":
                        module_name = os.path.splitext(rel_path.replace(os.sep, "."))[0]
                        modules.append(module_name)
                        break
            dependency_graph[rel_path] = sorted(set(file_record.get("imports", [])))
            for note in file_record.get("analysis_notes", []):
                lowered = str(note).lower()
                if "todo" in lowered:
                    patterns["todo_comments"] += 1
                if "fixme" in lowered:
                    patterns["fixme_comments"] += 1
                if "bare except" in lowered:
                    patterns["bare_except"] += 1

        if os.path.exists(os.path.join(repo, "pyproject.toml")) or os.path.exists(
            os.path.join(repo, "requirements.txt")
        ):
            tech_stack.add("python")
        if os.path.exists(os.path.join(repo, "CMakeLists.txt")):
            tech_stack.add("cmake")

        profile = RepoProfile(
            repo_path=repo,
            repo_name=os.path.basename(repo),
            role=role,
            read_only=True,
            file_count=file_count,
            loc=loc,
            language_breakdown=dict(language_counts),
            entry_points=sorted(set(entry_points)),
            modules=sorted(set(modules)),
            test_files=sorted(set(test_files)),
            build_files=sorted(set(build_files)),
            dependency_graph=dependency_graph,
            detected_patterns=patterns,
            tech_stack=sorted(tech_stack),
            notes=self._build_notes(
                repo, build_files, test_files, tech_stack, patterns
            ),
            metadata={
                "analysis_pack": pack,
                "corpus_session": dict(created.get("session") or {}),
                "trace_provider": pack.get("producer", "ComparativeAnalysisService.native_index"),
                "analysis_mode": "native_comparative_corpus",
            },
        )
        report = self._profile_to_markdown(profile)
        self.kb.store(
            category="analysis",
            topic=f"{profile.repo_name}_ingestion",
            content=report,
            source="repo-ingestion",
            confidence="high",
            tags=["repo", "ingestion", profile.role],
            metadata=profile.to_dict(),
        )
        return profile

    def _cross_repo_analysis(self, profiles: Dict[str, RepoProfile]) -> None:
        comparative_summary: dict[str, Any] | None = None
        try:
            target_repo = next(
                (profile.repo_path for profile in profiles.values() if profile.role == "target"),
                next(iter(profiles.keys())),
            )
            comparative = ComparativeAnalysisService(target_repo)
            comparative_summary = comparative.compare(
                target=target_repo,
                candidates=[path for path in profiles if os.path.abspath(path) != os.path.abspath(target_repo)],
                top_k=6,
            )
        except Exception:
            comparative_summary = None
        common_stack = None
        language_union: Counter[str] = Counter()
        for profile in profiles.values():
            tech_stack = set(profile.tech_stack)
            common_stack = (
                tech_stack if common_stack is None else common_stack & tech_stack
            )
            language_union.update(profile.language_breakdown)
        lines = [
            "# Cross-Repo Analysis",
            "",
            f"Repos analyzed: {len(profiles)}",
            f"Common tech stack: {', '.join(sorted(common_stack or set())) or 'none'}",
            "",
            "## Language Footprint",
        ]
        for language, count in language_union.most_common():
            lines.append(f"- {language}: {count}")
        if comparative_summary:
            lines.extend(["", "## Comparative Frontier"])
            for comparison in list(comparative_summary.get("comparisons") or [])[:5]:
                candidate = dict(comparison.get("candidate") or {})
                lines.append(f"- {candidate.get('corpus_id', 'candidate')}: {comparison.get('report_text', '')}")
        self.kb.store(
            category="analysis",
            topic="cross_repo_analysis",
            content="\n".join(lines) + "\n",
            source="repo-ingestion",
            confidence="medium",
            tags=["cross-repo", "analysis"],
            dependencies=[
                os.path.join(
                    self.kb.kb_dir,
                    "analysis",
                    f"{self.kb._slugify(profile.repo_name + '_ingestion')}.md",
                )
                for profile in profiles.values()
            ],
            metadata={"comparative_summary": comparative_summary},
        )

    @staticmethod
    def _build_notes(
        repo: str,
        build_files: List[str],
        test_files: List[str],
        tech_stack: set[str],
        patterns: Dict[str, int],
    ) -> List[str]:
        notes: List[str] = []
        if not build_files:
            notes.append("No standard build files detected.")
        if not test_files:
            notes.append("No tests detected under conventional locations.")
        if "simd" in tech_stack and "openmp" in tech_stack:
            notes.append("Performance-oriented native stack detected (SIMD + OpenMP).")
        if patterns.get("bare_except", 0):
            notes.append(
                "Bare except handlers detected; error taxonomy may need tightening."
            )
        if repo.endswith("Anvil"):
            notes.append("Primary workspace repo detected.")
        return notes

    @staticmethod
    def _profile_to_markdown(profile: RepoProfile) -> str:
        lines = [
            f"# Repository Ingestion: {profile.repo_name}",
            "",
            f"- Path: `{profile.repo_path}`",
            f"- Role: {profile.role}",
            f"- Read-only: {profile.read_only}",
            f"- Files: {profile.file_count}",
            f"- LOC: {profile.loc}",
            "",
            "## Languages",
        ]
        for language, count in sorted(profile.language_breakdown.items()):
            lines.append(f"- {language}: {count}")
        lines.extend(["", "## Entry Points"])
        lines.extend([f"- `{item}`" for item in profile.entry_points] or ["- none"])
        lines.extend(["", "## Build Files"])
        lines.extend([f"- `{item}`" for item in profile.build_files] or ["- none"])
        lines.extend(["", "## Notes"])
        lines.extend([f"- {item}" for item in profile.notes] or ["- none"])
        return "\n".join(lines) + "\n"
