"""Requirement extraction from markdown sources."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Iterable

from saguaro.parsing.markdown import (
    MarkdownDocument,
    MarkdownNode,
    MarkdownStructureParser,
)
from saguaro.requirements.model import (
    RequirementClassification,
    RequirementModality,
    RequirementPolarity,
    RequirementRecord,
    RequirementStrength,
    build_requirement_id,
    build_section_anchor,
    normalize_requirement_text,
)

_PATH_TOKEN_RE = re.compile(
    r"`([^`\n]+)`|((?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.[A-Za-z0-9_.-]+)|\b([A-Za-z0-9_.-]+\.[A-Za-z0-9_.-]+)\b"
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9`(])")
_NORMATIVE_SECTION_HINTS = (
    "acceptance",
    "constraint",
    "obligation",
    "compliance",
    "verification",
    "requirement",
    "criteria",
    "must",
    "should",
    "shall",
)
_CONTRACT_SECTION_HINTS = (
    "implementation contract",
    "execution contract",
    "acceptance criteria",
)
_ROADMAP_SECTION_HINTS = (
    "recommended cli surface",
    "recommended data contracts",
    "minimal viable product definition",
    "what should be avoided",
    "suggested file layout additions",
    "immediate next steps",
    "what the system should become",
)
_ROADMAP_ACTIONABLE_SUBSECTIONS = (
    "health and verification results",
    "deliverables",
    "required capabilities",
    "data model",
    "new artifact",
    "relation states",
    "evidence types",
    "cli additions",
    "exit criteria",
    "mapping pipeline",
    "witness classes",
    "validation states",
    "new intermediate representation",
    "new node families",
    "new edge families",
    "required orchestration primitives",
    "packet types",
    "tiered parser strategy",
    "initial packs",
    "pack responsibilities",
    "new benchmark families",
    "new agent-facing capabilities",
    "ide surfaces",
    "governance surfaces",
    "docs and requirements",
    "traceability and validation",
    "math and disparate relations",
    "weak-model packetization",
    "scientific packs",
    "requirementnode",
    "omnirelation",
    "witnessrecord",
    "counterexamplerecord",
    "fields",
)
_ROADMAP_PROGRAM_SECTION_HINTS = (
    "implementation program",
    "program rules",
    "workstream",
)
_ROADMAP_IDEA_SECTION_HINTS = (
    "candidate implementation phases",
    "proposed capabilities",
    "high-impact new ideas",
    "inventive research",
    "inventive ideas",
    "capability proposals",
    "prototype candidates",
    "moonshot ideas",
    "ideas worth prototyping",
)
_ROADMAP_IDEA_FIELD_HINTS = (
    "name:",
    "core insight:",
    "external inspiration",
    "external analogy",
    "why it fits",
    "exact places in this codebase",
    "exact wiring points:",
    "existing primitives",
    "new primitive",
    "new subsystem",
    "why this creates value",
    "why this creates moat",
    "main risk",
    "failure mode",
    "smallest credible first experiment",
    "first experiment",
    "confidence level",
    "confidence:",
)
_ROADMAP_PROGRAM_METADATA_PREFIXES = (
    "goal:",
    "capabilities covered:",
    "primary files:",
    "deliverables:",
    "exit criteria:",
    "prototype order:",
)
_ROADMAP_EXCLUDED_SECTION_HINTS = (
    "program rules",
    "phase plan overview",
    "purpose",
    "methodology",
    "executive summary",
    "direct answer to the user request",
    "current runtime reality",
    "current saguaro capabilities",
    "current gaps",
    "external research synthesis",
    "research-to-roadmap conclusions",
    "architectural north star",
    "current capability matrix",
    "roadmap principles",
    "how this roadmap uses what saguaro already has",
    "high-impact new ideas",
    "scientific and engineering scope expansion",
    "how saguaro should interact with anvil after this roadmap",
    "concrete gap closure",
    "recommended success metrics",
    "risks",
    "decision summary",
    "final position",
    "research references",
    "appendix",
    "sign-off",
    "objective",
    "why this is first",
    "implementation tasks",
    "new idea",
    "metrics",
    "chronicle integration",
    "internal benchmark datasets",
    "why config matters for end-to-end validation",
    "operating assumption",
    "design rules",
    "suggested phase order",
    "test and verification matrix",
    "what not to do",
)
_ROADMAP_GENERIC_IDEA_TITLES = {
    "roadmap",
    "inventive research roadmap",
    "proposed capabilities",
    "practical capabilities",
    "moonshot capabilities",
    "high-impact new ideas",
    "inventive research",
    "inventive ideas",
    "capability proposals",
    "critical pressure test",
    "synthesis",
    "first-principles framing",
    "external research scan",
    "repo grounding summary",
    "hidden assumptions",
    "candidate implementation phases",
}
_ROADMAP_ALLOW_SINGLE_TOKEN_SECTIONS = (
    "fields",
    "packet types",
    "initial packs",
    "relation states",
    "witness classes",
    "validation states",
    "required orchestration primitives",
    "pack responsibilities",
    "new node families",
    "new edge families",
)
_ROADMAP_ALLOW_COLON_SECTIONS = (
    "minimal viable product definition",
    "what should be avoided",
)
_MODALITY_RULES: list[tuple[re.Pattern[str], RequirementClassification]] = [
    (
        re.compile(r"\b(must not|required not|shall not)\b", re.IGNORECASE),
        RequirementClassification(
            modality=RequirementModality.MUST,
            strength=RequirementStrength.MANDATORY,
            polarity=RequirementPolarity.NEGATIVE,
            keyword="must not",
        ),
    ),
    (
        re.compile(r"\bmust\b", re.IGNORECASE),
        RequirementClassification(
            modality=RequirementModality.MUST,
            strength=RequirementStrength.MANDATORY,
            keyword="must",
        ),
    ),
    (
        re.compile(r"\bshall\b", re.IGNORECASE),
        RequirementClassification(
            modality=RequirementModality.SHALL,
            strength=RequirementStrength.MANDATORY,
            keyword="shall",
        ),
    ),
    (
        re.compile(r"\brequired\b", re.IGNORECASE),
        RequirementClassification(
            modality=RequirementModality.REQUIRED,
            strength=RequirementStrength.MANDATORY,
            keyword="required",
        ),
    ),
    (
        re.compile(r"\bshould\b", re.IGNORECASE),
        RequirementClassification(
            modality=RequirementModality.SHOULD,
            strength=RequirementStrength.RECOMMENDED,
            keyword="should",
        ),
    ),
    (
        re.compile(r"\brecommended\b", re.IGNORECASE),
        RequirementClassification(
            modality=RequirementModality.RECOMMENDED,
            strength=RequirementStrength.RECOMMENDED,
            keyword="recommended",
        ),
    ),
    (
        re.compile(r"\bmay\b", re.IGNORECASE),
        RequirementClassification(
            modality=RequirementModality.MAY,
            strength=RequirementStrength.OPTIONAL,
            keyword="may",
        ),
    ),
    (
        re.compile(r"\boptional\b", re.IGNORECASE),
        RequirementClassification(
            modality=RequirementModality.OPTIONAL,
            strength=RequirementStrength.OPTIONAL,
            keyword="optional",
        ),
    ),
    (
        re.compile(r"\bwill\b", re.IGNORECASE),
        RequirementClassification(
            modality=RequirementModality.WILL,
            strength=RequirementStrength.UNSPECIFIED,
            keyword="will",
        ),
    ),
]
_VERIFICATION_HINT_RE = re.compile(
    r"(pytest\s+[^\n`]+|saguaro\s+verify[^\n`]+)",
    re.IGNORECASE,
)


@dataclass(slots=True)
class RequirementExtractionResult:
    """Bundle extraction results for one or more markdown sources."""

    source_paths: tuple[str, ...]
    requirements: list[RequirementRecord]
    graph_loaded: bool

    def requirement_ids(self) -> list[str]:
        """Return extracted requirement identifiers in order."""
        return [item.requirement_id for item in self.requirements]


class GraphSnapshot:
    """Best-effort access to existing graph outputs."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        self.graph_path = repo_root / ".saguaro" / "graph" / "code_graph.json"
        self.loaded = False
        self.known_files: set[str] = set()
        self.basename_index: dict[str, set[str]] = {}
        self._load()

    def _load(self) -> None:
        payload: dict[str, object] | None = None
        for candidate in (
            self.repo_root / ".saguaro" / "graph" / "code_graph.json",
            self.repo_root / ".saguaro" / "graph" / "graph.json",
            self.repo_root / ".saguaro" / "code_graph.json",
        ):
            if not candidate.exists():
                continue
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            self.graph_path = candidate
            break
        if payload is None:
            return
        graph = payload.get("graph") if isinstance(payload, dict) else None
        if isinstance(graph, dict):
            payload = graph
        files = payload.get("files", {}) if isinstance(payload, dict) else {}
        if not isinstance(files, dict):
            return
        self.known_files = {str(path) for path in files}
        basename_index: dict[str, set[str]] = {}
        for path in self.known_files:
            basename_index.setdefault(Path(path).name, set()).add(path)
        self.basename_index = basename_index
        self.loaded = True

    def resolve_token(self, token: str) -> list[str]:
        """Resolve a path-like token against graph outputs and the repo."""
        candidate = token.strip().strip("`").strip(".,:;()[]")
        candidate = re.sub(r":\d+(?:-\d+)?$", "", candidate)
        if not candidate:
            return []
        matches: set[str] = set()
        if "/" in candidate:
            if candidate in self.known_files:
                matches.add(candidate)
            elif (self.repo_root / candidate).exists():
                matches.add(candidate)
        else:
            basename_matches = self.basename_index.get(Path(candidate).name, set())
            if len(basename_matches) == 1:
                matches.update(basename_matches)
        return sorted(matches)


class RequirementExtractor:
    """Extract normative requirements from markdown."""

    def __init__(
        self,
        repo_root: str | Path = ".",
        parser: MarkdownStructureParser | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.parser = parser or MarkdownStructureParser()
        self.graph = GraphSnapshot(self.repo_root)

    def discover_docs(self, path: str | Path = ".") -> list[Path]:
        """Discover markdown files under a path."""
        root = Path(path)
        root = root if root.is_absolute() else self.repo_root / root
        if not root.exists():
            if root.suffix.lower() in {".md", ".mdx"}:
                raise FileNotFoundError(f"markdown file not found: {root}")
            return []
        if root.is_file():
            return [root] if root.suffix.lower() in {".md", ".mdx"} else []
        if root.resolve() == self.repo_root:
            return sorted(
                [
                    item
                    for item in self.repo_root.iterdir()
                    if item.is_file() and item.suffix.lower() in {".md", ".mdx"}
                ]
            )
        docs: list[Path] = []
        for file_path in root.rglob("*"):
            if any(
                part in {".git", ".saguaro", "venv", ".venv", "__pycache__"}
                for part in file_path.parts
            ):
                continue
            if file_path.is_file() and file_path.suffix.lower() in {".md", ".mdx"}:
                docs.append(file_path)
        return sorted(docs)

    def extract(self, path: str | Path = ".") -> RequirementExtractionResult:
        """Extract requirements from all markdown files under a path."""
        return self.extract_paths(self.discover_docs(path))

    def extract_file(self, path: str | Path) -> RequirementExtractionResult:
        """Extract requirements from a markdown file on disk."""
        source_path = Path(path)
        abs_path = (
            source_path if source_path.is_absolute() else self.repo_root / source_path
        )
        rel_path = self._rel_path(abs_path)
        text = abs_path.read_text(encoding="utf-8")
        return self.extract_text(text, source_path=rel_path)

    def extract_text(
        self,
        text: str,
        *,
        source_path: str,
    ) -> RequirementExtractionResult:
        """Extract requirements from markdown text."""
        document = self.parser.parse(text, source_path=source_path)
        requirements = self._extract_document(
            document,
            profile=self._document_profile(document),
        )
        return RequirementExtractionResult(
            source_paths=(source_path,),
            requirements=requirements,
            graph_loaded=self.graph.loaded,
        )

    def extract_paths(self, paths: Iterable[str | Path]) -> RequirementExtractionResult:
        """Extract requirements from multiple markdown files."""
        source_paths: list[str] = []
        requirements: list[RequirementRecord] = []
        for path in paths:
            result = self.extract_file(path)
            source_paths.extend(result.source_paths)
            requirements.extend(result.requirements)
        return RequirementExtractionResult(
            source_paths=tuple(source_paths),
            requirements=requirements,
            graph_loaded=self.graph.loaded,
        )

    def _extract_document(
        self,
        document: MarkdownDocument,
        *,
        profile: str,
    ) -> list[RequirementRecord]:
        occurrence_map: dict[tuple[str, str, str], int] = {}
        requirements: list[RequirementRecord] = []
        contract_prefixes = self._contract_prefixes(document)
        section_refs: dict[tuple[str, ...], dict[str, list[str]]] = {}
        if profile == "roadmap":
            for section in document.nodes_of_kind("section"):
                if not section.section_path:
                    continue
                section_refs[tuple(section.section_path)] = self._collect_refs(
                    self._roadmap_section_text(section)
                )
        if profile == "roadmap":
            requirements.extend(
                self._extract_roadmap_idea_requirements(
                    document,
                    occurrence_map=occurrence_map,
                )
            )
            requirements.extend(
                self._extract_roadmap_program_phase_requirements(
                    document,
                    occurrence_map=occurrence_map,
                )
            )
        for node in document.walk():
            if node.kind not in {"paragraph", "list_item", "blockquote", "table"}:
                continue
            in_contract = bool(contract_prefixes) and self._within_contract(
                node.section_path, contract_prefixes
            )
            if (
                profile == "roadmap"
                and len(node.section_path) < 3
                and not in_contract
                and not any(
                    token in self._roadmap_section_haystack(node.section_path)
                    for token in (
                        _ROADMAP_SECTION_HINTS
                        + _ROADMAP_ACTIONABLE_SUBSECTIONS
                        + _ROADMAP_PROGRAM_SECTION_HINTS
                        + _ROADMAP_IDEA_SECTION_HINTS
                    )
                )
            ):
                continue
            if contract_prefixes and not in_contract:
                continue
            candidates = self._candidate_texts(node)
            for candidate in candidates:
                statement = candidate["text"]
                if not statement:
                    continue
                if (
                    profile == "roadmap"
                    and not in_contract
                    and self._roadmap_section_path_is_idea(
                        node.section_path
                    )
                ):
                    continue
                if (
                    profile == "roadmap"
                    and not in_contract
                    and self._roadmap_statement_is_program_metadata(
                        statement,
                        section_path=node.section_path,
                    )
                ):
                    continue
                if (
                    profile == "roadmap"
                    and not in_contract
                    and not self._roadmap_statement_is_actionable(
                        statement,
                        section_path=node.section_path,
                    )
                ):
                    continue
                if (
                    profile == "roadmap"
                    and not in_contract
                    and self._roadmap_statement_is_idea_metadata(
                        statement,
                        section_path=node.section_path,
                    )
                ):
                    continue
                classification = self._classify_statement(
                    statement,
                    section_path=node.section_path,
                    profile=profile,
                )
                if classification is None:
                    continue
                normalized = normalize_requirement_text(statement)
                occurrence_key = (
                    str(document.source_path or ""),
                    build_section_anchor(node.section_path),
                    normalized,
                )
                occurrence_map[occurrence_key] = (
                    occurrence_map.get(occurrence_key, 0) + 1
                )
                ordinal = occurrence_map[occurrence_key]
                requirement_id = build_requirement_id(
                    source_path=str(document.source_path or ""),
                    section_path=node.section_path,
                    normalized_statement=normalized,
                    ordinal=ordinal,
                )
                refs = self._collect_refs(statement)
                if profile == "roadmap":
                    inherited_refs = section_refs.get(tuple(node.section_path), {})
                    if inherited_refs:
                        explicit_statement_refs = any(
                            refs.get(key)
                            for key in (
                                "code_refs",
                                "test_refs",
                                "verification_refs",
                            )
                        )
                        for key in (
                            "code_refs",
                            "test_refs",
                            "verification_refs",
                            "graph_refs",
                        ):
                            if refs.get(key):
                                continue
                            if key == "graph_refs" and explicit_statement_refs:
                                continue
                            refs[key] = list(inherited_refs.get(key) or [])
                if (
                    profile == "default"
                    and "roadmap" in str(document.source_path or "").lower()
                    and not any(refs.values())
                ):
                    continue
                requirements.append(
                    RequirementRecord(
                        requirement_id=requirement_id,
                        source_path=str(document.source_path or ""),
                        section_path=node.section_path,
                        section_anchor=build_section_anchor(node.section_path),
                        line_start=int(candidate["line_start"]),
                        line_end=int(candidate["line_end"]),
                        statement=statement.strip(),
                        normalized_statement=normalized,
                        classification=classification,
                        block_kind=str(candidate["block_kind"]),
                        ordinal=ordinal,
                        code_refs=tuple(refs["code_refs"]),
                        test_refs=tuple(refs["test_refs"]),
                        verification_refs=tuple(refs["verification_refs"]),
                        graph_refs=tuple(refs["graph_refs"]),
                        metadata={
                            "source_node_kind": node.kind,
                            "source_section_path": list(node.section_path),
                            "profile": profile,
                            **(
                                {"concept_kind": "roadmap_contract"}
                                if profile == "roadmap" and in_contract
                                else {}
                            ),
                        },
                    )
                )
        return requirements

    def _extract_roadmap_idea_requirements(
        self,
        document: MarkdownDocument,
        *,
        occurrence_map: dict[tuple[str, str, str], int],
    ) -> list[RequirementRecord]:
        requirements: list[RequirementRecord] = []
        for node in document.nodes_of_kind("section"):
            if not node.section_path or not self._roadmap_section_is_idea_candidate(
                node
            ):
                continue
            statement = self._roadmap_idea_statement(node)
            if not statement:
                continue
            normalized = normalize_requirement_text(statement)
            occurrence_key = (
                str(document.source_path or ""),
                build_section_anchor(node.section_path),
                normalized,
            )
            occurrence_map[occurrence_key] = occurrence_map.get(occurrence_key, 0) + 1
            ordinal = occurrence_map[occurrence_key]
            requirement_id = build_requirement_id(
                source_path=str(document.source_path or ""),
                section_path=node.section_path,
                normalized_statement=normalized,
                ordinal=ordinal,
            )
            refs = self._collect_refs(self._roadmap_section_text(node))
            requirements.append(
                RequirementRecord(
                    requirement_id=requirement_id,
                    source_path=str(document.source_path or ""),
                    section_path=node.section_path,
                    section_anchor=build_section_anchor(node.section_path),
                    line_start=node.line_start,
                    line_end=node.line_end,
                    statement=statement,
                    normalized_statement=normalized,
                    classification=RequirementClassification(
                        modality=RequirementModality.RECOMMENDED,
                        strength=RequirementStrength.RECOMMENDED,
                        keyword="idea",
                    ),
                    block_kind="section_idea",
                    ordinal=ordinal,
                    code_refs=tuple(refs["code_refs"]),
                    test_refs=tuple(refs["test_refs"]),
                    verification_refs=tuple(refs["verification_refs"]),
                    graph_refs=tuple(refs["graph_refs"]),
                    metadata={
                        "source_node_kind": "section",
                        "source_section_path": list(node.section_path),
                        "profile": "roadmap",
                        "concept_kind": "roadmap_idea",
                        "phase_id": self._normalize_roadmap_phase_id(
                            self._roadmap_labeled_value(
                                self._roadmap_section_text(node),
                                labels=(
                                    "suggested `phase_id`",
                                    "suggested phase_id",
                                    "phase_id",
                                ),
                            )
                        ),
                    },
                )
            )
        return requirements

    def _extract_roadmap_program_phase_requirements(
        self,
        document: MarkdownDocument,
        *,
        occurrence_map: dict[tuple[str, str, str], int],
    ) -> list[RequirementRecord]:
        requirements: list[RequirementRecord] = []
        for node in document.nodes_of_kind("section"):
            if not node.section_path or not self._roadmap_section_is_program_phase(node):
                continue
            statement = self._roadmap_phase_statement(node)
            if not statement:
                continue
            normalized = normalize_requirement_text(statement)
            occurrence_key = (
                str(document.source_path or ""),
                build_section_anchor(node.section_path),
                normalized,
            )
            occurrence_map[occurrence_key] = occurrence_map.get(occurrence_key, 0) + 1
            ordinal = occurrence_map[occurrence_key]
            requirement_id = build_requirement_id(
                source_path=str(document.source_path or ""),
                section_path=node.section_path,
                normalized_statement=normalized,
                ordinal=ordinal,
            )
            text = self._roadmap_section_text(node)
            refs = self._collect_refs(text)
            requirements.append(
                RequirementRecord(
                    requirement_id=requirement_id,
                    source_path=str(document.source_path or ""),
                    section_path=node.section_path,
                    section_anchor=build_section_anchor(node.section_path),
                    line_start=node.line_start,
                    line_end=node.line_end,
                    statement=statement,
                    normalized_statement=normalized,
                    classification=RequirementClassification(
                        modality=RequirementModality.REQUIRED,
                        strength=RequirementStrength.MANDATORY,
                        keyword="phase",
                    ),
                    block_kind="section_phase",
                    ordinal=ordinal,
                    code_refs=tuple(refs["code_refs"]),
                    test_refs=tuple(refs["test_refs"]),
                    verification_refs=tuple(refs["verification_refs"]),
                    graph_refs=tuple(refs["graph_refs"]),
                    metadata={
                        "source_node_kind": "section",
                        "source_section_path": list(node.section_path),
                        "profile": "roadmap",
                        "concept_kind": "roadmap_phase",
                        "phase_id": self._normalize_roadmap_phase_id(
                            self._roadmap_labeled_value(
                                text,
                                labels=("phase_id",),
                            )
                        ),
                    },
                )
            )
        return requirements

    def _candidate_texts(self, node: MarkdownNode) -> list[dict[str, object]]:
        if node.kind == "table":
            candidates: list[dict[str, object]] = []
            for row in node.rows:
                row_text = " | ".join(cell for cell in row if cell)
                if row_text:
                    candidates.append(
                        {
                            "text": row_text,
                            "line_start": node.line_start,
                            "line_end": node.line_end,
                            "block_kind": "table_row",
                        }
                    )
            return candidates
        sentences = [
            part.strip()
            for part in _SENTENCE_SPLIT_RE.split(node.text.replace("\n", " "))
            if part.strip()
        ]
        if not sentences:
            sentences = [node.text.strip()]
        return [
            {
                "text": sentence,
                "line_start": node.line_start,
                "line_end": node.line_end,
                "block_kind": node.kind,
            }
            for sentence in sentences
        ]

    def _classify_statement(
        self,
        statement: str,
        *,
        section_path: tuple[str, ...],
        profile: str,
    ) -> RequirementClassification | None:
        for pattern, classification in _MODALITY_RULES:
            if pattern.search(statement):
                return classification
        if self._section_is_normative(section_path, profile=profile):
            return RequirementClassification(
                modality=RequirementModality.IMPLICIT,
                strength=RequirementStrength.UNSPECIFIED,
                keyword=None,
            )
        return None

    @staticmethod
    def _section_is_normative(
        section_path: tuple[str, ...],
        *,
        profile: str,
    ) -> bool:
        if profile == "roadmap":
            haystack = RequirementExtractor._roadmap_section_haystack(section_path)
            if any(token in haystack for token in _ROADMAP_SECTION_HINTS):
                return True
            if any(token in haystack for token in _ROADMAP_ACTIONABLE_SUBSECTIONS):
                return True
            if any(token in haystack for token in _ROADMAP_PROGRAM_SECTION_HINTS):
                return True
            if any(token in haystack for token in _ROADMAP_IDEA_SECTION_HINTS):
                return True
        haystack = " ".join(section_path).lower()
        if any(token in haystack for token in _CONTRACT_SECTION_HINTS):
            return True
        return any(token in haystack for token in _NORMATIVE_SECTION_HINTS)

    @staticmethod
    def _document_profile(document: MarkdownDocument) -> str:
        source = str(document.source_path or "").lower()
        root_sections = [
            node.title or "" for node in document.nodes_of_kind("section")[:2]
        ]
        title_hint = " ".join(root_sections).lower()
        filename_has_roadmap = "roadmap" in Path(source).stem
        has_roadmap_sections = any(
            any(
                token in RequirementExtractor._roadmap_section_haystack(node.section_path)
                for token in _ROADMAP_SECTION_HINTS
            )
            for node in document.nodes_of_kind("section")
        )
        if filename_has_roadmap and (
            "roadmap" in title_hint or has_roadmap_sections
        ):
            return "roadmap"
        return "default"

    def _contract_prefixes(self, document: MarkdownDocument) -> list[tuple[str, ...]]:
        prefixes: list[tuple[str, ...]] = []
        for node in document.walk():
            if node.kind != "section" or not node.section_path:
                continue
            label = " / ".join(node.section_path).lower()
            if any(token in label for token in _CONTRACT_SECTION_HINTS):
                prefixes.append(tuple(node.section_path))
        prefixes.sort(key=len)
        return prefixes

    @staticmethod
    def _within_contract(
        section_path: tuple[str, ...],
        prefixes: list[tuple[str, ...]],
    ) -> bool:
        return any(
            section_path[: len(prefix)] == prefix
            or section_path[-len(prefix) :] == prefix
            for prefix in prefixes
        )

    def _roadmap_statement_is_actionable(
        self,
        statement: str,
        *,
        section_path: tuple[str, ...],
    ) -> bool:
        lowered = statement.strip().lower()
        if not lowered or lowered == "---":
            return False

        section_hint = self._roadmap_section_haystack(section_path)
        leaf_hint = self._strip_outline_prefix(section_path[-1]).lower() if section_path else ""
        actionable_section = any(
            token in section_hint
            for token in (
                _ROADMAP_SECTION_HINTS
                + _ROADMAP_ACTIONABLE_SUBSECTIONS
                + _ROADMAP_PROGRAM_SECTION_HINTS
                + _ROADMAP_IDEA_SECTION_HINTS
            )
        )
        excluded_section = any(
            token in section_hint for token in _ROADMAP_EXCLUDED_SECTION_HINTS
        )
        actionable_leaf = any(
            token in leaf_hint
            for token in (
                _ROADMAP_SECTION_HINTS
                + _ROADMAP_ACTIONABLE_SUBSECTIONS
                + _ROADMAP_PROGRAM_SECTION_HINTS
            )
        )
        if excluded_section and not actionable_leaf:
            return False
        if self._roadmap_statement_is_idea_metadata(
            statement,
            section_path=section_path,
        ):
            return False

        has_artifact_hint = bool(
            "`" in statement
            or "/" in statement
            or statement.startswith("saguaro ")
            or statement.startswith("./venv/bin/saguaro ")
        )
        if lowered.endswith(":") and not has_artifact_hint:
            return any(token in section_hint for token in _ROADMAP_ALLOW_COLON_SECTIONS)

        normalized = lowered.strip("`")
        if " " not in normalized and not any(
            token in section_hint for token in _ROADMAP_ALLOW_SINGLE_TOKEN_SECTIONS
        ):
            return False

        return actionable_section

    def _roadmap_statement_is_idea_metadata(
        self,
        statement: str,
        *,
        section_path: tuple[str, ...],
    ) -> bool:
        section_hint = self._roadmap_section_haystack(section_path)
        if not any(token in section_hint for token in _ROADMAP_IDEA_SECTION_HINTS):
            return False
        lowered = statement.strip().lower()
        return any(lowered.startswith(prefix) for prefix in _ROADMAP_IDEA_FIELD_HINTS)

    def _roadmap_statement_is_program_metadata(
        self,
        statement: str,
        *,
        section_path: tuple[str, ...],
    ) -> bool:
        section_hint = self._roadmap_section_haystack(section_path)
        if not any(token in section_hint for token in _ROADMAP_PROGRAM_SECTION_HINTS):
            return False
        lowered = statement.strip().lower()
        return any(lowered.startswith(prefix) for prefix in _ROADMAP_PROGRAM_METADATA_PREFIXES)

    def _roadmap_section_is_idea_candidate(self, node: MarkdownNode) -> bool:
        if len(node.section_path) < 2:
            return False
        section_hint = self._roadmap_section_haystack(node.section_path)
        if "moonshot capabilities" in section_hint:
            return False
        if not any(token in section_hint for token in _ROADMAP_IDEA_SECTION_HINTS):
            return False
        title = self._strip_outline_prefix(node.title or "")
        if not title or title.lower() in _ROADMAP_GENERIC_IDEA_TITLES:
            return False
        text = self._roadmap_section_text(node)
        if any(token in text.lower() for token in _ROADMAP_IDEA_FIELD_HINTS):
            return True
        non_section_children = [
            child
            for child in node.children
            if child.kind != "section" and child.text.strip()
        ]
        return len(non_section_children) >= 1

    def _roadmap_section_is_program_phase(self, node: MarkdownNode) -> bool:
        if len(node.section_path) < 2:
            return False
        section_hint = self._roadmap_section_haystack(node.section_path)
        if not any(token in section_hint for token in _ROADMAP_PROGRAM_SECTION_HINTS):
            return False
        title = self._strip_outline_prefix(node.title or "").lower()
        return bool(re.match(r"^phase\s+\d+\b", title))

    @staticmethod
    def _roadmap_section_path_is_idea(section_path: tuple[str, ...]) -> bool:
        return any(
            RequirementExtractor._strip_outline_prefix(part).lower().startswith("idea ")
            or bool(
                re.match(
                    r"^[pm]\d+\.",
                    RequirementExtractor._strip_outline_prefix(part).lower(),
                )
            )
            for part in section_path
        )

    @staticmethod
    def _roadmap_section_haystack(section_path: tuple[str, ...]) -> str:
        relevant = section_path[1:] if len(section_path) > 1 else section_path
        return " ".join(relevant).lower()

    def _roadmap_idea_statement(self, node: MarkdownNode) -> str:
        title = self._strip_outline_prefix(node.title or "")
        if not title:
            return ""
        text = self._roadmap_section_text(node)
        experiment = self._roadmap_labeled_value(
            text,
            labels=("smallest credible first experiment", "first experiment"),
        )
        if experiment:
            return f"Implement roadmap idea '{title}'. First experiment: {experiment}"
        core_insight = self._roadmap_labeled_value(
            text,
            labels=("core insight",),
        )
        if core_insight:
            return f"Implement roadmap idea '{title}'. Core insight: {core_insight}"
        return f"Implement roadmap idea '{title}'."

    def _roadmap_phase_statement(self, node: MarkdownNode) -> str:
        title = self._strip_outline_prefix(node.title or "")
        if not title:
            return ""
        text = self._roadmap_section_text(node)
        phase_title = self._roadmap_labeled_value(text, labels=("phase title",))
        objective = self._roadmap_labeled_value(text, labels=("objective",))
        subject = phase_title or title
        if objective:
            return f"Implement roadmap phase '{subject}'. Objective: {objective}"
        return f"Implement roadmap phase '{subject}'."

    @staticmethod
    def _strip_outline_prefix(value: str) -> str:
        return re.sub(r"^\s*\d+(?:\.\d+)*[.)]?\s*", "", value).strip()

    def _roadmap_section_text(self, node: MarkdownNode) -> str:
        chunks: list[str] = []
        for child in node.walk():
            if child.kind in {"section", "code_fence", "list"}:
                continue
            if child.kind == "table":
                for row in child.rows:
                    row_text = " | ".join(cell for cell in row if cell)
                    if row_text:
                        chunks.append(row_text)
                continue
            text = child.text.strip()
            if text:
                chunks.append(text)
        return "\n".join(chunks)

    @staticmethod
    def _roadmap_labeled_value(text: str, *, labels: tuple[str, ...]) -> str:
        for line in text.splitlines():
            stripped = line.strip()
            candidate = stripped
            if candidate.startswith(("-", "*")):
                candidate = candidate[1:].strip()
            normalized_candidate = candidate.replace("`", "")
            lowered = normalized_candidate.lower()
            for label in labels:
                prefix = f"{label.replace('`', '').lower()}:"
                if lowered.startswith(prefix):
                    return candidate.split(":", 1)[1].strip()
        return ""

    @staticmethod
    def _normalize_roadmap_phase_id(value: str) -> str:
        normalized = value.strip()
        if len(normalized) >= 2 and normalized[0] == normalized[-1] == "`":
            return normalized[1:-1].strip()
        return normalized

    def _collect_refs(self, statement: str) -> dict[str, list[str]]:
        code_refs: set[str] = set()
        test_refs: set[str] = set()
        verification_refs: set[str] = set()
        graph_refs: set[str] = set()

        for match in _PATH_TOKEN_RE.finditer(statement):
            token = next(group for group in match.groups() if group)
            resolved = self.graph.resolve_token(token)
            if not resolved:
                if "/" in token and (self.repo_root / token).exists():
                    resolved = [token]
            for path in resolved:
                if path.startswith("tests/"):
                    test_refs.add(path)
                else:
                    code_refs.add(path)
                if path in self.graph.known_files:
                    graph_refs.add(path)

        for match in _VERIFICATION_HINT_RE.finditer(statement):
            verification_refs.add(match.group(1).strip())

        if test_refs and not verification_refs:
            verification_refs.add("pytest " + " ".join(sorted(test_refs)))

        return {
            "code_refs": sorted(code_refs),
            "test_refs": sorted(test_refs),
            "verification_refs": sorted(verification_refs),
            "graph_refs": sorted(graph_refs),
        }

    def _rel_path(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.repo_root).as_posix()
        except ValueError:
            return path.resolve().as_posix()
