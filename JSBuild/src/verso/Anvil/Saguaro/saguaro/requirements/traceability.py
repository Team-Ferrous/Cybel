"""Persistent traceability ledger for extracted requirements."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from saguaro.requirements.extractor import (
    RequirementExtractionResult,
    RequirementExtractor,
)
from saguaro.requirements.model import RequirementRecord


@dataclass(frozen=True, slots=True)
class TraceabilityRecord:
    """Represent one v2 traceability entry."""

    version: str
    trace_id: str
    run_id: str
    requirement_id: str
    source_path: str
    design_refs: tuple[str, ...]
    section_path: tuple[str, ...]
    line_start: int
    line_end: int
    statement: str
    normalized_statement: str
    modality: str
    strength: str
    polarity: str
    code_refs: tuple[str, ...]
    test_refs: tuple[str, ...]
    verification_refs: tuple[str, ...]
    graph_refs: tuple[str, ...]
    changed_files: tuple[str, ...]
    owner: str
    timestamp: str
    source_hash: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-friendly mapping."""
        payload = asdict(self)
        payload["design_refs"] = list(self.design_refs)
        payload["section_path"] = list(self.section_path)
        payload["code_refs"] = list(self.code_refs)
        payload["test_refs"] = list(self.test_refs)
        payload["verification_refs"] = list(self.verification_refs)
        payload["graph_refs"] = list(self.graph_refs)
        payload["changed_files"] = list(self.changed_files)
        return payload

    @classmethod
    def from_requirement(
        cls,
        requirement: RequirementRecord,
        *,
        owner: str,
        run_id: str,
        timestamp: str,
    ) -> "TraceabilityRecord":
        """Build a record from an extracted requirement."""
        changed_files = sorted(
            {
                requirement.source_path,
                *requirement.code_refs,
                *requirement.test_refs,
            }
        )
        trace_seed = "||".join(
            [
                requirement.requirement_id,
                requirement.source_hash,
                "|".join(changed_files),
            ]
        )
        trace_digest = hashlib.sha1(trace_seed.encode("utf-8")).hexdigest()[:14]
        return cls(
            version="2",
            trace_id=f"trace::req::{trace_digest}",
            run_id=run_id,
            requirement_id=requirement.requirement_id,
            source_path=requirement.source_path,
            design_refs=(requirement.design_ref(),),
            section_path=requirement.section_path,
            line_start=requirement.line_start,
            line_end=requirement.line_end,
            statement=requirement.statement,
            normalized_statement=requirement.normalized_statement,
            modality=requirement.classification.modality.value,
            strength=requirement.classification.strength.value,
            polarity=requirement.classification.polarity.value,
            code_refs=requirement.code_refs,
            test_refs=requirement.test_refs,
            verification_refs=requirement.verification_refs,
            graph_refs=requirement.graph_refs,
            changed_files=tuple(changed_files),
            owner=owner,
            timestamp=timestamp,
            source_hash=requirement.source_hash,
            metadata=dict(requirement.metadata),
        )


class TraceabilityService:
    """Build and persist requirement traceability records."""

    def __init__(
        self,
        repo_root: str | Path = ".",
        extractor: RequirementExtractor | None = None,
        graph_service: Any | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.extractor = extractor or RequirementExtractor(repo_root=self.repo_root)
        self.graph_service = graph_service
        self.traceability_dir = self.repo_root / "standards" / "traceability"
        self.ledger_path = self.traceability_dir / "TRACEABILITY.v2.jsonl"
        self.compat_ledger_path = self.traceability_dir / "TRACEABILITY.jsonl"
        self.cache_dir = self.repo_root / ".saguaro" / "traceability"
        self.cache_path = self.cache_dir / "cache.json"
        self.compliance_dir = self.repo_root / ".anvil" / "compliance"

    def build(self, path: str | Path = ".") -> dict[str, Any]:
        """Compatibility wrapper that extracts and returns a full payload."""
        result = self.build_from_markdown_paths(self.extractor.discover_docs(path))
        payload = self._load_payload()
        return {
            "status": "ok",
            "generated_at": payload.get("generated_at"),
            "generation_id": payload.get("generation_id"),
            "requirements": payload.get("requirements", []),
            "records": payload.get("records", []),
            "summary": {
                "requirement_count": int(result.get("requirement_count", 0) or 0),
                "record_count": int(result.get("record_count", 0) or 0),
                "appended_count": int(result.get("appended_count", 0) or 0),
                "skipped_count": int(result.get("skipped_count", 0) or 0),
            },
        }

    def status(self, requirement_id: str) -> dict[str, Any]:
        """Return records for a single requirement."""
        payload = self._load_payload()
        requirement = next(
            (
                item
                for item in payload.get("requirements", [])
                if item.get("id") == requirement_id
                or item.get("requirement_id") == requirement_id
            ),
            None,
        )
        records = [
            item
            for item in payload.get("records", [])
            if item.get("requirement_id") == requirement_id
        ]
        if requirement is None:
            return {
                "status": "missing",
                "requirement_id": requirement_id,
                "records": [],
            }
        return {
            "status": "ok",
            "requirement": requirement,
            "records": records,
            "count": len(records),
        }

    def diff(self) -> dict[str, Any]:
        """Diff the latest two cached snapshots."""
        history_path = self.cache_dir / "history.json"
        if not history_path.exists():
            return {"status": "insufficient_history", "entries": 0}
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if len(history) < 2:
            return {"status": "insufficient_history", "entries": len(history)}
        latest = history[-1]
        previous = history[-2]
        latest_ids = {item["trace_id"] for item in latest.get("records", [])}
        previous_ids = {item["trace_id"] for item in previous.get("records", [])}
        return {
            "status": "ok",
            "latest_generation": latest.get("run_id"),
            "previous_generation": previous.get("run_id"),
            "added": sorted(latest_ids - previous_ids),
            "removed": sorted(previous_ids - latest_ids),
        }

    def orphaned(self) -> dict[str, Any]:
        """List requirements with no linked code or test refs."""
        payload = self._load_payload()
        requirements = [
            item
            for item in payload.get("requirements", [])
            if not item.get("code_refs") and not item.get("test_refs")
        ]
        return {
            "status": "ok",
            "count": len(requirements),
            "requirements": requirements,
        }

    def build_from_markdown_paths(
        self,
        paths: Iterable[str | Path],
        *,
        owner: str = "saguaro-requirements",
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Extract requirements from markdown files and persist traceability."""
        extraction = self.extractor.extract_paths(paths)
        records = self.build_records(
            extraction.requirements,
            owner=owner,
            run_id=run_id,
        )
        persistence = self.persist(records)
        return {
            "source_paths": list(extraction.source_paths),
            "requirement_count": len(extraction.requirements),
            "record_count": len(records),
            "graph_loaded": extraction.graph_loaded,
            "requirements": [item.to_dict() for item in extraction.requirements],
            **persistence,
        }

    def build_records(
        self,
        requirements: Iterable[RequirementRecord],
        *,
        owner: str = "saguaro-requirements",
        run_id: str | None = None,
        timestamp: str | None = None,
    ) -> list[TraceabilityRecord]:
        """Build traceability records from extracted requirements."""
        normalized_timestamp = timestamp or dt.datetime.now(dt.timezone.utc).isoformat()
        normalized_run_id = run_id or f"traceability-{normalized_timestamp}"
        graph_nodes = self._graph_nodes()
        return [
            TraceabilityRecord.from_requirement(
                self._augment_requirement(requirement, graph_nodes),
                owner=owner,
                run_id=normalized_run_id,
                timestamp=normalized_timestamp,
            )
            for requirement in requirements
        ]

    def persist(self, records: Iterable[TraceabilityRecord]) -> dict[str, Any]:
        """Persist changed traceability records to JSONL and cache."""
        record_list = list(records)
        self.traceability_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        cache = self._load_cache()
        known = dict(cache.get("requirements", {}))
        appended = 0
        skipped = 0

        with self.ledger_path.open(
            "a", encoding="utf-8"
        ) as handle, self.compat_ledger_path.open(
            "a", encoding="utf-8"
        ) as compat_handle:
            for record in record_list:
                fingerprint = self._fingerprint(record)
                cache_entry = known.get(record.requirement_id)
                if cache_entry and cache_entry.get("fingerprint") == fingerprint:
                    skipped += 1
                    continue
                payload = json.dumps(record.to_dict(), sort_keys=True) + "\n"
                handle.write(payload)
                compat_handle.write(payload)
                known[record.requirement_id] = {
                    "fingerprint": fingerprint,
                    "source_hash": record.source_hash,
                    "trace_id": record.trace_id,
                    "source_path": record.source_path,
                    "updated_at": record.timestamp,
                }
                appended += 1

        latest_run_id = record_list[-1].run_id if record_list else None
        cache_payload = {
            "version": 2,
            "ledger_path": self.ledger_path.relative_to(self.repo_root).as_posix(),
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "run_id": latest_run_id,
            "requirements": known,
            "records": [record.to_dict() for record in record_list],
            "requirements_expanded": self._expand_requirements(record_list),
        }
        self.cache_path.write_text(
            json.dumps(cache_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._append_history(cache_payload)
        self._write_compliance_bundle(cache_payload)

        return {
            "appended_count": appended,
            "skipped_count": skipped,
            "ledger_path": self.ledger_path.as_posix(),
            "cache_path": self.cache_path.as_posix(),
        }

    def load_records(self) -> list[dict[str, Any]]:
        """Load persisted ledger records."""
        if not self.ledger_path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in self.ledger_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            records.append(json.loads(line))
        return records

    def extract_and_build(
        self,
        path: str | Path,
        *,
        owner: str = "saguaro-requirements",
        run_id: str | None = None,
    ) -> tuple[RequirementExtractionResult, list[TraceabilityRecord]]:
        """Convenience helper for tests and callers that need both layers."""
        extraction = self.extractor.extract_file(path)
        records = self.build_records(
            extraction.requirements, owner=owner, run_id=run_id
        )
        return extraction, records

    def _load_payload(self) -> dict[str, Any]:
        cache = self._load_cache()
        return {
            "generation_id": cache.get("run_id"),
            "generated_at": cache.get("generated_at"),
            "requirements": list(cache.get("requirements_expanded", [])),
            "records": list(cache.get("records", [])),
        }

    def _load_cache(self) -> dict[str, Any]:
        if not self.cache_path.exists():
            return {"version": 2, "requirements": {}}
        try:
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"version": 2, "requirements": {}}

    def _append_history(self, payload: dict[str, Any]) -> None:
        history_path = self.cache_dir / "history.json"
        if history_path.exists():
            try:
                history = json.loads(history_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                history = []
        else:
            history = []
        history.append(
            {
                "run_id": payload.get("run_id"),
                "generated_at": payload.get("generated_at"),
                "records": list(payload.get("records", [])),
            }
        )
        history_path.write_text(
            json.dumps(history[-10:], indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def _write_compliance_bundle(self, payload: dict[str, Any]) -> None:
        run_id = str(
            payload.get("run_id")
            or f"traceability-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d%H%M%S')}"
        )
        compliance_root = self.compliance_dir / run_id
        compliance_root.mkdir(parents=True, exist_ok=True)
        compliance_payload = {
            "run_id": run_id,
            "generated_at": payload.get("generated_at"),
            "traceability": list(payload.get("records", [])),
            "requirements": list(payload.get("requirements_expanded", [])),
        }
        (compliance_root / "traceability.json").write_text(
            json.dumps(compliance_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _expand_requirements(records: list[TraceabilityRecord]) -> list[dict[str, Any]]:
        return [
            {
                "id": record.requirement_id,
                "requirement_id": record.requirement_id,
                "file": record.source_path,
                "heading_path": list(record.section_path),
                "text_raw": record.statement,
                "text_norm": record.normalized_statement,
                "strength": record.strength,
                "modality": record.modality,
                "tags": [],
                "equation_ids": [],
                "status": "implemented" if record.code_refs else "unknown",
                "code_refs": list(record.code_refs),
                "test_refs": list(record.test_refs),
                "verification_refs": list(record.verification_refs),
                "graph_refs": list(record.graph_refs),
                "line_start": record.line_start,
                "line_end": record.line_end,
                "metadata": dict(record.metadata),
            }
            for record in records
        ]

    @staticmethod
    def _fingerprint(record: TraceabilityRecord) -> str:
        payload = {
            "requirement_id": record.requirement_id,
            "source_hash": record.source_hash,
            "design_refs": list(record.design_refs),
            "modality": record.modality,
            "strength": record.strength,
            "polarity": record.polarity,
            "code_refs": list(record.code_refs),
            "test_refs": list(record.test_refs),
            "verification_refs": list(record.verification_refs),
            "graph_refs": list(record.graph_refs),
            "changed_files": list(record.changed_files),
            "metadata": dict(record.metadata),
        }
        return hashlib.sha1(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _graph_nodes(self) -> list[dict[str, Any]]:
        if self.graph_service is None:
            return []
        try:
            graph = self.graph_service.load_graph()
        except Exception:
            return []
        return list((graph.get("nodes") or {}).values())

    def _augment_requirement(
        self, requirement: RequirementRecord, graph_nodes: list[dict[str, Any]]
    ) -> RequirementRecord:
        graph_files = {
            str(node.get("file") or "")
            for node in graph_nodes
            if str(node.get("file") or "")
        }
        code_refs = self._clean_ref_list(requirement.code_refs)
        test_refs = self._clean_ref_list(requirement.test_refs)
        verification_refs = self._clean_ref_list(requirement.verification_refs)
        graph_refs = self._clean_ref_list(requirement.graph_refs)
        is_qsg_idea = self._is_qsg_roadmap_idea(requirement)

        if self._is_roadmap_requirement(requirement):
            code_refs.extend(
                self._existing_paths(
                    self._roadmap_code_refs(requirement),
                    graph_files=graph_files,
                )
            )
            test_refs.extend(
                self._existing_paths(
                    self._roadmap_test_refs(requirement),
                    graph_files=graph_files,
                )
            )
            graph_refs.extend(code_refs)
            if test_refs:
                graph_refs.extend(test_refs)
            verification_refs.extend(
                self._roadmap_verification_refs(requirement, test_refs)
            )

        if graph_nodes and (not code_refs or not graph_refs) and not is_qsg_idea:
            requirement_tokens = self._tokenize(
                " ".join(
                    [
                        requirement.statement,
                        *requirement.section_path,
                        requirement.source_path,
                    ]
                )
            )
            ranked: list[tuple[float, dict[str, Any]]] = []
            for node in graph_nodes:
                file_path = str(node.get("file") or "")
                node_text = " ".join(
                    [
                        str(node.get("name") or ""),
                        str(node.get("qualified_name") or ""),
                        file_path,
                    ]
                )
                node_tokens = self._tokenize(node_text)
                overlap = requirement_tokens.intersection(node_tokens)
                if not overlap:
                    continue
                score = len(overlap) / max(len(requirement_tokens), 1)
                ranked.append((score, node))
            ranked.sort(key=lambda item: (-item[0], str(item[1].get("file") or "")))
            for score, node in ranked[:5]:
                file_path = str(node.get("file") or "")
                qualified = str(
                    node.get("qualified_name") or node.get("name") or file_path
                )
                graph_refs.append(qualified)
                if (
                    "/test" in file_path or file_path.startswith("tests/")
                ) and not test_refs:
                    test_refs.append(file_path)
                elif score >= 0.18 and not code_refs:
                    code_refs.append(file_path)

        if code_refs and not test_refs:
            test_refs.extend(self._infer_tests_for_code_refs(code_refs, graph_files))

        if (
            test_refs
            and not verification_refs
            and self._is_roadmap_requirement(requirement)
        ):
            verification_refs.append(
                "pytest " + " ".join(sorted(dict.fromkeys(test_refs)))
            )

        return replace(
            requirement,
            code_refs=tuple(self._existing_artifact_refs(code_refs)),
            test_refs=tuple(self._existing_artifact_refs(test_refs)),
            verification_refs=tuple(self._clean_ref_list(verification_refs)),
            graph_refs=tuple(self._clean_ref_list(graph_refs)),
        )

    @staticmethod
    def _is_roadmap_requirement(requirement: RequirementRecord) -> bool:
        return "roadmap" in requirement.source_path.lower()

    def _existing_paths(
        self,
        candidates: list[str],
        *,
        graph_files: set[str],
    ) -> list[str]:
        return [
            candidate
            for candidate in self._clean_ref_list(candidates)
            if candidate in graph_files or (self.repo_root / candidate).exists()
        ]

    @staticmethod
    def _clean_ref_list(refs: Iterable[str]) -> list[str]:
        return [
            ref.strip()
            for ref in refs
            if isinstance(ref, str) and ref.strip()
        ]

    def _existing_artifact_refs(self, refs: Iterable[str]) -> list[str]:
        return [
            ref
            for ref in self._clean_ref_list(refs)
            if (self.repo_root / ref).exists()
        ]

    def _roadmap_code_refs(self, requirement: RequirementRecord) -> list[str]:
        section_hint = " / ".join(requirement.section_path).lower()
        statement = requirement.statement.strip().strip("`")
        is_qsg_idea = self._is_qsg_roadmap_idea(requirement)
        refs: list[str] = []
        if is_qsg_idea:
            refs.extend(self._qsg_idea_code_refs(requirement))
        if statement.startswith("saguaro ") or statement.startswith(
            "./venv/bin/saguaro "
        ):
            refs.extend(["saguaro/cli.py", "saguaro/api.py"])
            refs.extend(self._command_module_refs(statement))
        if "requirementnode" in section_hint:
            refs.extend(
                [
                    "saguaro/requirements/model.py",
                    "saguaro/requirements/traceability.py",
                ]
            )
        if "omnirelation" in section_hint:
            refs.extend(["saguaro/omnigraph/model.py", "saguaro/omnigraph/store.py"])
        if "witnessrecord" in section_hint or "counterexamplerecord" in section_hint:
            refs.extend(
                ["saguaro/requirements/model.py", "saguaro/validation/witnesses.py"]
            )
        if "markdown structural parsing" in section_hint:
            refs.extend(
                ["saguaro/parsing/markdown.py", "saguaro/requirements/extractor.py"]
            )
        if "runtime stabilization and runtime unification" in section_hint:
            refs.extend(["saguaro/api.py", "saguaro/cli.py", "saguaro/health.py"])
        if "recommended cli surface" in section_hint:
            refs.extend(["saguaro/cli.py", "saguaro/api.py"])
        if "requirement graph and traceability ledger" in section_hint:
            refs.extend(
                [
                    "saguaro/requirements/traceability.py",
                    "saguaro/requirements/model.py",
                ]
            )
        if "doc-to-code candidate mapping" in section_hint:
            refs.extend(
                [
                    "saguaro/requirements/extractor.py",
                    "saguaro/requirements/traceability.py",
                ]
            )
        if "witness and validation engine" in section_hint:
            refs.extend(
                ["saguaro/validation/engine.py", "saguaro/validation/witnesses.py"]
            )
        if (
            "math intermediate representation" in section_hint
            or "math and disparate relations" in section_hint
        ):
            refs.extend(["saguaro/math/__init__.py", "saguaro/math/engine.py"])
        if "disparate relation omni-graph" in section_hint:
            refs.extend(["saguaro/omnigraph/model.py", "saguaro/omnigraph/store.py"])
        if "weak-model orchestration runtime" in section_hint:
            refs.extend(["saguaro/packets/model.py", "saguaro/packets/builders.py"])
        if "multilingual parsing depth upgrade" in section_hint:
            refs.extend(["saguaro/parsing/parser.py", "saguaro/parsing/markdown.py"])
        if "benchmarks, evals, and drift analytics" in section_hint:
            refs.extend(
                [
                    "saguaro/chronicle/diff.py",
                    "saguaro/requirements/traceability.py",
                    "saguaro/validation/engine.py",
                ]
            )
        if "ide, agent, and governance integration" in section_hint:
            refs.extend(
                [
                    "domains/code_intelligence/saguaro_substrate.py",
                    "tools/saguaro_tools.py",
                    "tools/registry.py",
                    "core/unified_chat_loop.py",
                    "saguaro/sentinel/policy.py",
                ]
            )
        if (
            "scientific repository packs" in section_hint
            or "scientific packs" in section_hint
        ):
            refs.extend(["saguaro/packs/base.py"])
        if (
            "minimal viable product definition" in section_hint
            or "immediate next steps" in section_hint
        ):
            refs.extend(
                [
                    "saguaro/requirements/extractor.py",
                    "saguaro/requirements/traceability.py",
                    "saguaro/validation/engine.py",
                    "saguaro/packets/builders.py",
                ]
            )
        if "qsg inference pipeline upgrade roadmap" in section_hint and not is_qsg_idea:
            refs.extend(
                [
                    "core/qsg/runtime_contracts.py",
                    "core/native/native_qsg_engine.py",
                    "core/qsg/continuous_engine.py",
                    "core/qsg/latent_bridge.py",
                    "core/memory/latent_memory.py",
                    "core/memory/fabric/policies.py",
                    "saguaro/state/ledger.py",
                    "benchmarks/native_qsg_benchmark.py",
                    "cli/commands/features.py",
                ]
            )
        if "14.5 governance and verification" in section_hint:
            refs.extend(
                [
                    "domains/verification/auto_verifier.py",
                    "core/native/native_qsg_engine.py",
                    "core/qsg/runtime_contracts.py",
                ]
            )
        if "what should be avoided" in section_hint:
            refs.extend(
                ["saguaro/requirements/traceability.py", "saguaro/packets/model.py"]
            )
        if "benchmark audit upgrade roadmap" in section_hint:
            refs.extend(
                [
                    "audit/runner/benchmark_suite.py",
                    "audit/runner/suite_preflight.py",
                    "audit/runner/assurance_control_plane.py",
                ]
            )
        if "reliable benchmarking and variance control" in section_hint:
            refs.extend(
                [
                    "audit/control_plane/variance.py",
                    "audit/control_plane/topology.py",
                ]
            )
        if "health and verification results" in section_hint:
            refs.extend(
                [
                    "saguaro/health.py",
                    "saguaro/coverage.py",
                    "saguaro/cli.py",
                ]
            )
        if "data model principle" in section_hint:
            refs.extend(
                [
                    "audit/control_plane/nodes.py",
                    "audit/control_plane/capsules.py",
                    "audit/control_plane/traceability.py",
                ]
            )
        if "governance and verification work" in section_hint:
            refs.extend(
                [
                    "audit/control_plane/compiler.py",
                    "audit/control_plane/traceability.py",
                    "audit/runner/benchmark_suite.py",
                ]
            )
        if "exit criteria for the roadmap itself" in section_hint:
            refs.extend(
                [
                    "audit/control_plane/compiler.py",
                    "audit/control_plane/ledger.py",
                    "audit/control_plane/topology.py",
                    "audit/control_plane/comparators.py",
                    "audit/control_plane/traceability.py",
                    "audit/runner/benchmark_suite.py",
                ]
            )
        pack_name = statement.removeprefix("`").removesuffix("`")
        if pack_name.endswith("_pack"):
            refs.append(f"saguaro/packs/{pack_name}.py")
        if statement.endswith("Packet"):
            refs.append("saguaro/packets/model.py")
        return list(dict.fromkeys(refs))

    def _qsg_idea_code_refs(self, requirement: RequirementRecord) -> list[str]:
        section_hint = " / ".join(requirement.section_path).lower()
        refs: list[str] = []
        mapping: list[tuple[str, list[str]]] = [
            (
                "idea 01. avx2-first binary discipline",
                [
                    "core/native/CMakeLists.txt",
                    "core/native/fast_attention.cpp",
                    "core/native/native_ops.py",
                    "core/native/qsg_parallel_kernels.cpp",
                    "core/native/simd_ops.cpp",
                    "core/native/thread_config.cpp",
                ],
            ),
            (
                "idea 02. optional amx leaf, not amx architecture",
                [
                    "core/native/amx_kernels.cpp",
                    "core/native/amx_kernels.h",
                    "core/native/CMakeLists.txt",
                    "core/native/native_ops.py",
                    "core/native/native_qsg_engine.py",
                    "core/native/split/compat/native_ops_compat.cpp",
                    "core/qsg/runtime_contracts.py",
                ],
            ),
            (
                "idea 03. cpu flashattention rebuild around cache tiles",
                [
                    "core/native/fast_attention.cpp",
                    "core/native/fast_attention_wrapper.py",
                    "core/native/qsg_forward.py",
                ],
            ),
            (
                "idea 04. projection-sampler fusion",
                [
                    "core/native/native_qsg_engine.py",
                    "core/native/qsg_parallel_kernels.cpp",
                    "core/native/simd_ops.cpp",
                    "core/native/simd_ops_wrapper.py",
                ],
            ),
            (
                "idea 05. affinity and l3-domain control plane",
                [
                    "core/native/native_qsg_engine.py",
                    "core/native/thread_config.cpp",
                ],
            ),
            (
                "idea 06. performance envelope ledger",
                [
                    "benchmarks/native_kernel_microbench.py",
                    "benchmarks/native_qsg_benchmark.py",
                    "core/native/native_qsg_engine.py",
                ],
            ),
            (
                "idea 07. deltalog as the authoritative time crystal primitive",
                [
                    "saguaro/indexing/coordinator.py",
                    "saguaro/indexing/engine.py",
                    "saguaro/indexing/tracker.py",
                    "saguaro/services/platform.py",
                    "saguaro/state/ledger.py",
                    "tools/file_ops.py",
                ],
            ),
            (
                "idea 08. executioncapsule abi",
                [
                    "core/campaign/control_plane.py",
                    "core/native/parallel_generation.py",
                    "core/qsg/latent_bridge.py",
                    "shared_kernel/event_store.py",
                ],
            ),
            (
                "idea 09. latentpacket abi v2",
                [
                    "core/memory/fabric/models.py",
                    "core/memory/latent_memory.py",
                    "core/native/parallel_generation.py",
                    "core/qsg/latent_bridge.py",
                ],
            ),
            (
                "idea 10. memorytierpolicy",
                [
                    "core/memory/fabric/store.py",
                    "core/memory/latent_memory.py",
                    "core/memory/project_memory.py",
                    "core/native/native_qsg_engine.py",
                    "core/qsg/latent_bridge.py",
                ],
            ),
            (
                "idea 11. intent-conditioned latent steering on native path",
                [
                    "core/native/latent_steering.py",
                    "core/native/native_qsg_engine.py",
                    "core/native/parallel_generation.py",
                ],
            ),
            (
                "idea 12. draft frontier controller",
                [
                    "core/native/native_qsg_engine.py",
                    "core/native/parallel_generation.py",
                    "core/native/qsg_parallel_kernels.cpp",
                ],
            ),
            (
                "idea 13. prompt-lookup and radix prefix reuse for cpu",
                [
                    "core/native/parallel_generation.py",
                    "core/native/qsg_parallel_kernels.cpp",
                    "core/qsg/continuous_engine.py",
                    "core/qsg/ollama_adapter.py",
                ],
            ),
            (
                "idea 14. observation-window kv compression",
                [
                    "core/native/model_graph_wrapper.py",
                    "core/native/native_qsg_engine.py",
                ],
            ),
            (
                "idea 15. time crystal drift controller as closed loop",
                [
                    "core/native/model_graph_wrapper.py",
                    "core/native/native_qsg_engine.py",
                ],
            ),
            (
                "idea 16. github-like time crystal delta execution",
                [
                    "audit/provenance/capture.py",
                    "saguaro/chronicle/core.py",
                    "saguaro/indexing/coordinator.py",
                    "saguaro/state/ledger.py",
                    "saguaro/watcher.py",
                    "tools/file_ops.py",
                ],
            ),
            (
                "idea 17. repo-delta memory",
                [
                    "core/memory/fabric/store.py",
                    "core/qsg/latent_bridge.py",
                    "saguaro/state/ledger.py",
                ],
            ),
            (
                "idea 18. checkpointed reasoning lanes",
                [
                    "agents/unified_master.py",
                    "core/native/parallel_generation.py",
                    "core/unified_chat_loop.py",
                ],
            ),
            (
                "idea 19. native grammar fast lanes",
                [
                    "core/native/native_qsg_engine.py",
                    "core/native/qsg_parallel_kernels.cpp",
                    "core/native/simd_ops.cpp",
                ],
            ),
            (
                "idea 20. native replay tape",
                [
                    "core/native/runtime_telemetry.py",
                    "core/telemetry/black_box.py",
                    "shared_kernel/event_store.py",
                ],
            ),
            (
                "idea 21. governance invariants for qsg",
                [
                    "core/aes/governance.py",
                    "domains/verification/auto_verifier.py",
                    "tools/verify.py",
                ],
            ),
            (
                "idea 22. mission replay from executioncapsules and deltalog",
                [
                    "core/campaign/control_plane.py",
                    "core/qsg/latent_bridge.py",
                    "saguaro/state/ledger.py",
                    "shared_kernel/event_store.py",
                ],
            ),
            (
                "idea 23. performance twin",
                [
                    "core/native/runtime_telemetry.py",
                ],
            ),
            (
                "idea 24. repo-coupled cognitive runtime",
                [
                    "core/native/native_qsg_engine.py",
                    "core/native/parallel_generation.py",
                    "core/qsg/latent_bridge.py",
                    "saguaro/state/ledger.py",
                ],
            ),
        ]
        for token, code_paths in mapping:
            if token in section_hint:
                refs.extend(code_paths)
                break
        return refs

    def _roadmap_test_refs(self, requirement: RequirementRecord) -> list[str]:
        section_hint = " / ".join(requirement.section_path).lower()
        statement = requirement.statement.strip().strip("`")
        is_qsg_idea = self._is_qsg_roadmap_idea(requirement)
        refs: list[str] = []
        if is_qsg_idea:
            refs.extend(self._qsg_idea_test_refs(requirement))
        if statement.startswith("saguaro ") or statement.startswith(
            "./venv/bin/saguaro "
        ):
            refs.extend(["tests/test_saguaro_interface.py"])
            refs.extend(self._command_test_refs(statement))
        if (
            "requirementnode" in section_hint
            or "doc-to-code candidate mapping" in section_hint
        ):
            refs.extend(["tests/test_saguaro_requirements.py"])
        if "traceability" in section_hint:
            refs.extend(["tests/test_saguaro_traceability.py"])
        if "markdown structural parsing" in section_hint:
            refs.extend(["tests/test_saguaro_markdown_parser.py"])
        if "runtime stabilization and runtime unification" in section_hint:
            refs.extend(["tests/test_saguaro_interface.py"])
        if "recommended cli surface" in section_hint:
            refs.extend(["tests/test_saguaro_interface.py"])
        if "witness and validation engine" in section_hint:
            refs.extend(["tests/test_saguaro_validate_docs.py"])
        if "disparate relation omni-graph" in section_hint:
            refs.extend(["tests/test_saguaro_omnigraph.py"])
        if "weak-model orchestration runtime" in section_hint or statement.endswith(
            "Packet"
        ):
            refs.extend(["tests/test_saguaro_packets.py"])
        if "multilingual parsing depth upgrade" in section_hint:
            refs.extend(
                [
                    "tests/test_saguaro_parser_languages.py",
                    "tests/test_saguaro_markdown_parser.py",
                ]
            )
        if "benchmarks, evals, and drift analytics" in section_hint:
            refs.extend(["tests/test_saguaro_traceability.py"])
        if "ide, agent, and governance integration" in section_hint:
            refs.extend(["tests/test_saguaro_interface.py"])
        if "scientific repository packs" in section_hint or statement.endswith("_pack"):
            refs.extend(["tests/test_saguaro_science_packs.py"])
        if (
            "math intermediate representation" in section_hint
            or "math and disparate relations" in section_hint
        ):
            refs.extend(["tests/test_saguaro_math.py"])
        if (
            "minimal viable product definition" in section_hint
            or "immediate next steps" in section_hint
        ):
            refs.extend(
                [
                    "tests/test_saguaro_requirements.py",
                    "tests/test_saguaro_traceability.py",
                    "tests/test_saguaro_validate_docs.py",
                    "tests/test_saguaro_packets.py",
                ]
            )
        if "qsg inference pipeline upgrade roadmap" in section_hint and not is_qsg_idea:
            refs.extend(
                [
                    "tests/test_qsg_continuous_engine.py",
                    "tests/test_latent_package_capture.py",
                    "tests/test_development_replay.py",
                    "tests/test_runtime_telemetry.py",
                    "tests/test_state_ledger.py",
                    "tests/test_qsg_runtime_contracts.py",
                ]
            )
        if "roadmap" in section_hint and not is_qsg_idea:
            refs.extend(["tests/test_saguaro_roadmap_validator.py"])
        if "benchmark audit upgrade roadmap" in section_hint:
            refs.extend(["tests/audit/test_benchmark_suite.py"])
        if "health and verification results" in section_hint:
            refs.extend(
                [
                    "tests/test_saguaro_traceability.py",
                    "tests/test_saguaro_roadmap_validator.py",
                ]
            )
        if (
            "data model principle" in section_hint
            or "exit criteria for the roadmap itself" in section_hint
        ):
            refs.extend(
                [
                    "tests/audit/test_benchmark_suite.py",
                    "tests/test_saguaro_traceability.py",
                ]
            )
        return list(dict.fromkeys(refs))

    def _qsg_idea_test_refs(self, requirement: RequirementRecord) -> list[str]:
        section_hint = " / ".join(requirement.section_path).lower()
        refs: list[str] = []
        mapping: list[tuple[str, list[str]]] = [
            (
                "idea 01. avx2-first binary discipline",
                [
                    "tests/test_native_qsg_benchmark.py",
                    "tests/test_qsg_runtime_contracts.py",
                    "tests/test_native_qsg_engine.py",
                ],
            ),
            (
                "idea 02. optional amx leaf, not amx architecture",
                [
                    "tests/test_qsg_runtime_contracts.py",
                    "tests/test_native_qsg_engine.py",
                ],
            ),
            (
                "idea 03. cpu flashattention rebuild around cache tiles",
                [
                    "tests/test_fast_attention_mqa.py",
                    "tests/test_fused_attention_integration.py",
                ],
            ),
            (
                "idea 04. projection-sampler fusion",
                [
                    "tests/test_native_qsg_engine.py",
                ],
            ),
            (
                "idea 05. affinity and l3-domain control plane",
                [
                    "tests/test_native_qsg_engine.py",
                    "tests/test_native_qsg_benchmark.py",
                ],
            ),
            (
                "idea 06. performance envelope ledger",
                [
                    "tests/test_native_qsg_benchmark.py",
                    "tests/test_qsg_runtime_contracts.py",
                ],
            ),
            (
                "idea 07. deltalog as the authoritative time crystal primitive",
                [
                    "tests/test_state_ledger.py",
                ],
            ),
            (
                "idea 08. executioncapsule abi",
                [
                    "tests/test_development_replay.py",
                    "tests/test_qsg_continuous_engine.py",
                ],
            ),
            (
                "idea 09. latentpacket abi v2",
                [
                    "tests/test_latent_package_capture.py",
                    "tests/test_qsg_continuous_engine.py",
                ],
            ),
            (
                "idea 10. memorytierpolicy",
                [
                    "tests/test_native_qsg_engine.py",
                ],
            ),
            (
                "idea 11. intent-conditioned latent steering on native path",
                [
                    "tests/test_native_qsg_engine.py",
                    "tests/test_runtime_telemetry.py",
                ],
            ),
            (
                "idea 12. draft frontier controller",
                [
                    "tests/test_native_qsg_engine.py",
                    "tests/test_native_parallel_generation_engine.py",
                ],
            ),
            (
                "idea 13. prompt-lookup and radix prefix reuse for cpu",
                [
                    "tests/test_qsg_continuous_engine.py",
                ],
            ),
            (
                "idea 14. observation-window kv compression",
                [
                    "tests/test_native_qsg_engine.py",
                ],
            ),
            (
                "idea 15. time crystal drift controller as closed loop",
                [
                    "tests/test_native_qsg_engine.py",
                    "tests/test_runtime_telemetry.py",
                ],
            ),
            (
                "idea 16. github-like time crystal delta execution",
                [
                    "tests/test_state_ledger.py",
                ],
            ),
            (
                "idea 17. repo-delta memory",
                [
                    "tests/test_latent_package_capture.py",
                ],
            ),
            (
                "idea 18. checkpointed reasoning lanes",
                [
                    "tests/test_native_parallel_generation_engine.py",
                ],
            ),
            (
                "idea 19. native grammar fast lanes",
                [
                    "tests/test_native_qsg_engine.py",
                ],
            ),
            (
                "idea 20. native replay tape",
                [
                    "tests/test_black_box_recorder.py",
                ],
            ),
            (
                "idea 21. governance invariants for qsg",
                [
                    "tests/test_saguaro_verifier.py",
                    "tests/test_runtime_telemetry.py",
                ],
            ),
            (
                "idea 22. mission replay from executioncapsules and deltalog",
                [
                    "tests/test_development_replay.py",
                    "tests/test_latent_package_capture.py",
                ],
            ),
            (
                "idea 23. performance twin",
                [
                    "tests/test_runtime_telemetry.py",
                ],
            ),
            (
                "idea 24. repo-coupled cognitive runtime",
                [
                    "tests/test_development_replay.py",
                    "tests/test_state_ledger.py",
                    "tests/test_latent_package_capture.py",
                ],
            ),
        ]
        for token, test_paths in mapping:
            if token in section_hint:
                refs.extend(test_paths)
                break
        return refs

    @staticmethod
    def _is_qsg_roadmap_idea(requirement: RequirementRecord) -> bool:
        if "qsg_inference_roadmap.md" not in requirement.source_path.lower():
            return False
        concept_kind = (
            str(requirement.metadata.get("concept_kind") or "").strip().lower()
        )
        if concept_kind == "roadmap_idea":
            return True
        return any(
            part.strip().lower().startswith("idea ")
            for part in requirement.section_path
        )

    @staticmethod
    def _roadmap_verification_refs(
        requirement: RequirementRecord,
        test_refs: list[str],
    ) -> list[str]:
        statement = requirement.statement.strip().strip("`")
        refs: list[str] = []
        if statement.startswith("saguaro ") or statement.startswith(
            "./venv/bin/saguaro "
        ):
            refs.append(statement)
        if test_refs:
            refs.append("pytest " + " ".join(sorted(dict.fromkeys(test_refs))))
        return refs

    @staticmethod
    def _command_module_refs(statement: str) -> list[str]:
        command = statement.replace("./venv/bin/", "").replace("saguaro ", "", 1)
        head = command.split()
        if not head:
            return []
        mapping = {
            "docs": [
                "saguaro/parsing/markdown.py",
                "saguaro/requirements/extractor.py",
            ],
            "requirements": [
                "saguaro/requirements/extractor.py",
                "saguaro/requirements/model.py",
            ],
            "traceability": ["saguaro/requirements/traceability.py"],
            "validate": [
                "saguaro/validation/engine.py",
                "saguaro/validation/witnesses.py",
            ],
            "math": ["saguaro/math/__init__.py", "saguaro/math/engine.py"],
            "omnigraph": ["saguaro/omnigraph/model.py", "saguaro/omnigraph/store.py"],
            "packet": ["saguaro/packets/model.py", "saguaro/packets/builders.py"],
            "packs": ["saguaro/packs/base.py"],
            "roadmap": ["saguaro/roadmap/validator.py"],
        }
        return mapping.get(head[0], [])

    @staticmethod
    def _command_test_refs(statement: str) -> list[str]:
        command = statement.replace("./venv/bin/", "").replace("saguaro ", "", 1)
        head = command.split()
        if not head:
            return []
        mapping = {
            "docs": ["tests/test_saguaro_markdown_parser.py"],
            "requirements": ["tests/test_saguaro_requirements.py"],
            "traceability": ["tests/test_saguaro_traceability.py"],
            "validate": ["tests/test_saguaro_validate_docs.py"],
            "math": ["tests/test_saguaro_math.py"],
            "omnigraph": ["tests/test_saguaro_omnigraph.py"],
            "packet": ["tests/test_saguaro_packets.py"],
            "packs": ["tests/test_saguaro_science_packs.py"],
            "roadmap": ["tests/test_saguaro_roadmap_validator.py"],
        }
        return mapping.get(head[0], [])

    def _infer_tests_for_code_refs(
        self,
        code_refs: list[str],
        graph_files: set[str],
    ) -> list[str]:
        inferred: list[str] = []
        for code_ref in code_refs:
            path = Path(code_ref)
            stem = path.stem
            parent = path.parent.as_posix()
            candidates = [
                f"tests/test_{stem}.py",
                f"tests/{stem}_test.py",
                f"tests/{path.with_suffix('').as_posix()}.py",
                f"tests/{path.with_suffix('').as_posix()}_test.py",
            ]
            if parent and parent != ".":
                candidates.append(f"tests/{parent}/test_{stem}.py")
                candidates.append(f"tests/{parent}/{stem}_test.py")
            for candidate in candidates:
                if candidate in graph_files or (self.repo_root / candidate).exists():
                    inferred.append(candidate)
                    break
        return inferred

    @staticmethod
    def _tokenize(value: str) -> set[str]:
        tokens = set()
        current: list[str] = []
        for ch in value.lower():
            if ch.isalnum():
                current.append(ch)
            else:
                if len(current) > 2:
                    tokens.add("".join(current))
                current = []
        if len(current) > 2:
            tokens.add("".join(current))
        return tokens
