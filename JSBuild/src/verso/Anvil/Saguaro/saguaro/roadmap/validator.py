"""Validate roadmap markdown files and emit completion graphs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from typing import Any

from domains.verification.auto_verifier import AutoVerifier
from saguaro.cpu import CPUScanner
from saguaro.omnigraph.model import OmniNode, OmniRelation
from saguaro.parsing import RuntimeSymbolResolver
from saguaro.requirements.traceability import TraceabilityService
from saguaro.validation.hotspot_capsules import load_hotspot_capsule_manifest
from saguaro.validation.witnesses import WitnessAggregator


class RoadmapValidator:
    """Build roadmap-oriented validation reports and completion graphs."""

    def __init__(self, repo_path: str, graph_service: Any | None = None) -> None:
        self.repo_path = os.path.abspath(repo_path)
        self.graph_service = graph_service
        self.base_dir = os.path.join(self.repo_path, ".saguaro", "roadmap")
        self.traceability = TraceabilityService(
            repo_root=self.repo_path,
            graph_service=self.graph_service,
        )
        self.witnesses = WitnessAggregator()
        self.runtime_symbols = RuntimeSymbolResolver(self.repo_path)

    def validate(self, path: str = ".") -> dict[str, Any]:
        """Validate a roadmap file and return completion-oriented output."""
        validation = self._validation_report(path)
        graph = self.build_graph(path=path, validation=validation)
        requirements = self._summarize_requirements(validation)
        summary = {
            "count": len(requirements),
            "completed": sum(
                1 for item in requirements if item["completion_state"] == "completed"
            ),
            "partial": sum(
                1 for item in requirements if item["completion_state"] == "partial"
            ),
            "missing": sum(
                1 for item in requirements if item["completion_state"] == "missing"
            ),
            "blockers": sum(len(item["blockers"]) for item in requirements),
        }
        coverage = self._coverage_summary(requirements)
        worklist = sorted(
            [item for item in requirements if item["completion_state"] != "completed"],
            key=lambda item: (
                self._completion_rank(item["completion_state"]),
                -len(item["blockers"]),
                item["id"],
            ),
        )
        return {
            "status": "ok",
            "path": path,
            "generation_id": validation.get("generation_id"),
            "summary": summary,
            "coverage": coverage,
            "requirements": requirements,
            "worklist": worklist,
            "gate_report": {
                "path": validation.get("gate_report", {}).get("artifact_path", ""),
                "summary": validation.get("gate_report", {}).get("summary", {}),
            },
            "graph": {
                "path": graph["graph_path"],
                "summary": graph["summary"],
            },
        }

    def build_graph(
        self,
        *,
        path: str = ".",
        validation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist a roadmap completion graph for the requested roadmap."""
        report = validation or self._validation_report(path)
        requirements = self._summarize_requirements(report)
        nodes: dict[str, OmniNode] = {}
        relations: dict[str, OmniRelation] = {}
        generation_id = str(
            report.get("generation_id") or f"roadmap-{int(time.time())}"
        )
        roadmap_node_id = f"roadmap::{self._stable_fragment(path)}"
        nodes[roadmap_node_id] = OmniNode(
            id=roadmap_node_id,
            type="roadmap",
            label=path,
            file=str(path),
            metadata={
                "requirement_count": len(requirements),
                "completed": sum(
                    1
                    for item in requirements
                    if item["completion_state"] == "completed"
                ),
                "partial": sum(
                    1 for item in requirements if item["completion_state"] == "partial"
                ),
                "missing": sum(
                    1 for item in requirements if item["completion_state"] == "missing"
                ),
            },
        )

        for item in requirements:
            section_node_id = self._section_node_id(
                item["source_path"], item["section_path"]
            )
            if section_node_id not in nodes:
                section_label = " / ".join(item["section_path"]) or item["source_path"]
                nodes[section_node_id] = OmniNode(
                    id=section_node_id,
                    type="roadmap_section",
                    label=section_label,
                    file=item["source_path"],
                    metadata={
                        "source_path": item["source_path"],
                        "section_path": item["section_path"],
                    },
                )
                relations[f"{roadmap_node_id}->{section_node_id}::contains"] = (
                    OmniRelation(
                        id=f"{roadmap_node_id}->{section_node_id}::contains",
                        src_type="roadmap",
                        src_id=roadmap_node_id,
                        dst_type="roadmap_section",
                        dst_id=section_node_id,
                        relation_type="contains",
                        evidence_types=["markdown"],
                        confidence=1.0,
                        verified=True,
                        drift_state="fresh",
                        generation_id=generation_id,
                    )
                )

            req_id = item["id"]
            nodes[req_id] = OmniNode(
                id=req_id,
                type="roadmap_requirement",
                label=item["statement"],
                file=item["source_path"],
                metadata={
                    "validation_state": item["validation_state"],
                    "completion_state": item["completion_state"],
                    "blockers": item["blockers"],
                    "strength": item["strength"],
                    "section_path": item["section_path"],
                    "code_ref_count": len(item["code_refs"]),
                    "test_ref_count": len(item["test_refs"]),
                    "graph_ref_count": len(item["graph_refs"]),
                },
            )
            relations[f"{section_node_id}->{req_id}::contains"] = OmniRelation(
                id=f"{section_node_id}->{req_id}::contains",
                src_type="roadmap_section",
                src_id=section_node_id,
                dst_type="roadmap_requirement",
                dst_id=req_id,
                relation_type="contains",
                evidence_types=["markdown"],
                confidence=1.0,
                verified=True,
                drift_state="fresh",
                generation_id=generation_id,
            )

            for relation_type, refs in (
                ("implemented_by", item["code_refs"]),
                ("witnessed_by", item["test_refs"]),
                ("supported_by", item["graph_refs"]),
                ("verified_by", item["verification_refs"]),
            ):
                for ref in refs:
                    artifact_id = f"artifact::{ref}"
                    nodes.setdefault(
                        artifact_id,
                        OmniNode(
                            id=artifact_id,
                            type="artifact",
                            label=ref,
                            file=ref,
                            metadata={"artifact_path": ref},
                        ),
                    )
                    relations[f"{req_id}->{artifact_id}::{relation_type}"] = (
                        OmniRelation(
                            id=f"{req_id}->{artifact_id}::{relation_type}",
                            src_type="roadmap_requirement",
                            src_id=req_id,
                            dst_type="artifact",
                            dst_id=artifact_id,
                            relation_type=relation_type,
                            evidence_types=["traceability"],
                            confidence=0.82,
                            verified=relation_type != "supported_by",
                            drift_state="fresh",
                            generation_id=generation_id,
                        )
                    )

            for blocker in item["blockers"]:
                blocker_id = f"gap::{req_id}::{blocker}"
                nodes[blocker_id] = OmniNode(
                    id=blocker_id,
                    type="roadmap_gap",
                    label=blocker,
                    file=item["source_path"],
                    metadata={
                        "requirement_id": req_id,
                        "severity": self._blocker_severity(blocker),
                    },
                )
                relations[f"{req_id}->{blocker_id}::blocked_by"] = OmniRelation(
                    id=f"{req_id}->{blocker_id}::blocked_by",
                    src_type="roadmap_requirement",
                    src_id=req_id,
                    dst_type="roadmap_gap",
                    dst_id=blocker_id,
                    relation_type="blocked_by",
                    evidence_types=["validation"],
                    confidence=0.95,
                    verified=True,
                    drift_state="fresh",
                    generation_id=generation_id,
                )

        payload = {
            "status": "ok",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "generation_id": generation_id,
            "roadmap_path": path,
            "nodes": {key: value.to_dict() for key, value in nodes.items()},
            "relations": {key: value.to_dict() for key, value in relations.items()},
            "summary": {
                "node_count": len(nodes),
                "relation_count": len(relations),
                "requirement_count": len(requirements),
                "completed_count": sum(
                    1
                    for item in requirements
                    if item["completion_state"] == "completed"
                ),
                "partial_count": sum(
                    1 for item in requirements if item["completion_state"] == "partial"
                ),
                "missing_count": sum(
                    1 for item in requirements if item["completion_state"] == "missing"
                ),
                "gap_count": sum(
                    1 for item in nodes.values() if item.type == "roadmap_gap"
                ),
            },
        }
        os.makedirs(self.base_dir, exist_ok=True)
        graph_path = os.path.join(
            self.base_dir,
            f"{self._stable_fragment(path)}.json",
        )
        with open(graph_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        payload["graph_path"] = graph_path
        return payload

    def _validation_report(self, path: str) -> dict[str, Any]:
        payload = self.traceability.build(path)
        runtime_symbols = self.runtime_symbols.build_symbol_manifest(persist=True)
        hotspot_manifest = load_hotspot_capsule_manifest(self.repo_path)
        if self._should_refresh_hotspot_capsules(payload, hotspot_manifest):
            try:
                CPUScanner(self.repo_path).scan(
                    path="core/native",
                    arch="x86_64-avx2",
                    limit=5,
                )
            except Exception:
                pass
            hotspot_manifest = load_hotspot_capsule_manifest(self.repo_path)
        grouped_records: dict[str, list[dict[str, Any]]] = {}
        for record in payload.get("records", []):
            grouped_records.setdefault(
                str(record.get("requirement_id") or ""), []
            ).append(record)

        requirements = []
        for requirement in payload.get("requirements", []):
            requirement_id = str(
                requirement.get("id") or requirement.get("requirement_id") or ""
            )
            normalized = self._normalized_requirement(requirement)
            records = grouped_records.get(requirement_id, [])
            witnesses = self.witnesses.build(
                normalized,
                records,
                generation_id=str(payload.get("generation_id") or "trace"),
            )
            gate_context = self._gate_context(
                normalized,
                records,
                hotspot_manifest=hotspot_manifest,
                runtime_symbols=runtime_symbols,
            )
            requirements.append(
                {
                    "requirement": normalized,
                    "records": records,
                    "witnesses": [item.to_dict() for item in witnesses],
                    "gate_context": gate_context,
                    "state": self.witnesses.classify_state(records, witnesses),
                }
            )
        gate_report = self._persist_gate_report(
            path,
            requirements,
            hotspot_manifest=hotspot_manifest,
            runtime_symbols=runtime_symbols,
        )
        return {
            "status": "ok",
            "generation_id": payload.get("generation_id"),
            "runtime_symbols": runtime_symbols,
            "hotspot_manifest": hotspot_manifest,
            "gate_report": gate_report,
            "requirements": requirements,
        }

    def _summarize_requirements(
        self, validation: dict[str, Any]
    ) -> list[dict[str, Any]]:
        items = []
        for item in validation.get("requirements", []):
            requirement = dict(item.get("requirement") or {})
            req_id = str(
                requirement.get("id") or requirement.get("requirement_id") or ""
            )
            source_path = str(
                requirement.get("file") or requirement.get("source_path") or ""
            )
            blockers = self._blockers(item)
            validation_state = str(item.get("state") or "unknown")
            completion_state = self._completion_state(validation_state, blockers)
            metadata = dict(requirement.get("metadata") or {})
            items.append(
                {
                    "id": req_id,
                    "source_path": source_path,
                    "statement": str(
                        requirement.get("text_raw")
                        or requirement.get("statement")
                        or requirement.get("label")
                        or req_id
                    ),
                    "section_path": list(
                        requirement.get("heading_path")
                        or requirement.get("section_path")
                        or []
                    ),
                    "validation_state": validation_state,
                    "completion_state": completion_state,
                    "strength": str(requirement.get("strength") or ""),
                    "concept_kind": str(
                        metadata.get("concept_kind")
                        or ("roadmap_contract" if source_path.lower().endswith("roadmap.md") else "")
                    ),
                    "phase_id": str(metadata.get("phase_id") or ""),
                    "code_refs": self._ref_list(item.get("records"), "code_refs"),
                    "test_refs": self._ref_list(item.get("records"), "test_refs"),
                    "graph_refs": self._ref_list(item.get("records"), "graph_refs"),
                    "verification_refs": self._ref_list(
                        item.get("records"), "verification_refs"
                    ),
                    "gate_context": dict(item.get("gate_context") or {}),
                    "blockers": blockers,
                }
            )
        return items

    @staticmethod
    def _coverage_summary(requirements: list[dict[str, Any]]) -> dict[str, Any]:
        by_kind: dict[str, dict[str, int]] = {}
        for item in requirements:
            kind = str(item.get("concept_kind") or "unclassified")
            bucket = by_kind.setdefault(
                kind,
                {"count": 0, "completed": 0, "partial": 0, "missing": 0},
            )
            bucket["count"] += 1
            state = str(item.get("completion_state") or "missing")
            if state in bucket:
                bucket[state] += 1
        return {
            "kinds": by_kind,
            "phase_ids": sorted(
                {
                    str(item.get("phase_id") or "")
                    for item in requirements
                    if str(item.get("phase_id") or "")
                }
            ),
        }

    @staticmethod
    def _normalized_requirement(requirement: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(requirement)
        normalized.setdefault(
            "id",
            str(requirement.get("id") or requirement.get("requirement_id") or ""),
        )
        normalized.setdefault(
            "file",
            str(requirement.get("file") or requirement.get("source_path") or ""),
        )
        normalized.setdefault(
            "text_raw",
            str(requirement.get("text_raw") or requirement.get("statement") or ""),
        )
        return normalized

    @staticmethod
    def _ref_list(records: Any, key: str) -> list[str]:
        refs: set[str] = set()
        for record in list(records or []):
            refs.update(str(item) for item in list(record.get(key) or []) if str(item))
        return sorted(refs)

    def _blockers(self, validation_item: dict[str, Any]) -> list[str]:
        records = list(validation_item.get("records") or [])
        witnesses = list(validation_item.get("witnesses") or [])
        requirement = dict(validation_item.get("requirement") or {})
        blockers: list[str] = []
        code_refs = self._ref_list(records, "code_refs")
        test_refs = self._ref_list(records, "test_refs")
        graph_refs = self._ref_list(records, "graph_refs")
        verification_refs = self._ref_list(records, "verification_refs")
        declared = self._declared_artifacts(str(requirement.get("text_raw") or ""))
        gate_context = dict(validation_item.get("gate_context") or {})
        if not code_refs:
            blockers.append("missing_code_refs")
        if not test_refs:
            blockers.append("missing_test_refs")
        if declared["code_refs"] and not set(declared["code_refs"]).issubset(
            set(code_refs)
        ):
            blockers.append("missing_declared_code_artifacts")
        if declared["test_refs"] and not set(declared["test_refs"]).issubset(
            set(test_refs)
        ):
            blockers.append("missing_declared_test_artifacts")
        if not graph_refs:
            blockers.append("missing_graph_refs")
        if not verification_refs:
            blockers.append("missing_verification_refs")
        if not any(item.get("result") == "pass" for item in witnesses):
            blockers.append("missing_passing_witness")
        if gate_context.get("requires_runtime_symbols") and float(
            gate_context.get("runtime_symbol_coverage_pct", 0.0) or 0.0
        ) < 80.0:
            blockers.append("runtime_symbol_coverage_below_gate")
        if not gate_context.get("authoritative_contract", True):
            blockers.append("non_authoritative_roadmap")
        if gate_context.get("requires_hotspot_capsules"):
            if int(gate_context.get("capsule_count", 0) or 0) <= 0:
                blockers.append("missing_hotspot_capsules")
            if int(gate_context.get("stale_evidence_count", 0) or 0) > 0:
                blockers.append("stale_hotspot_evidence")
            if int(gate_context.get("contradictory_evidence_count", 0) or 0) > 0:
                blockers.append("contradictory_hotspot_evidence")
        return blockers

    @staticmethod
    def _completion_state(
        validation_state: str, blockers: list[str] | None = None
    ) -> str:
        if validation_state == "implemented_witnessed":
            base_state = "completed"
        elif validation_state in {"implemented_unwitnessed", "partially_implemented"}:
            base_state = "partial"
        else:
            base_state = "missing"
        blocker_set = set(blockers or [])
        if not blocker_set:
            return base_state
        if {
            "missing_code_refs",
            "missing_passing_witness",
            "non_authoritative_roadmap",
        } & blocker_set:
            return "missing"
        if base_state != "completed" and {
            "missing_test_refs",
            "missing_verification_refs",
        }.issubset(blocker_set):
            return "missing"
        if base_state == "completed":
            return "partial"
        return base_state

    @staticmethod
    def _completion_rank(state: str) -> int:
        return {"missing": 0, "partial": 1, "completed": 2}.get(state, 3)

    @staticmethod
    def _blocker_severity(blocker: str) -> str:
        if blocker in {
            "missing_code_refs",
            "missing_passing_witness",
            "missing_declared_code_artifacts",
            "runtime_symbol_coverage_below_gate",
            "missing_hotspot_capsules",
            "stale_hotspot_evidence",
            "contradictory_hotspot_evidence",
        }:
            return "high"
        if blocker in {
            "missing_test_refs",
            "missing_verification_refs",
            "missing_declared_test_artifacts",
        }:
            return "medium"
        return "low"

    @staticmethod
    def _declared_artifacts(statement: str) -> dict[str, list[str]]:
        code_refs: set[str] = set()
        test_refs: set[str] = set()
        for match in re.finditer(r"`([^`]+)`", statement):
            candidate = str(match.group(1) or "").strip()
            if not candidate or "/" not in candidate:
                continue
            if any(ch.isspace() for ch in candidate):
                continue
            lowered = candidate.lower()
            if lowered.startswith("tests/") and lowered.endswith(".py"):
                test_refs.add(candidate)
            elif re.search(r"\.(py|pyi|pyx|pxd|c|cc|cpp|cxx|h|hh|hpp|hxx)$", lowered):
                code_refs.add(candidate)
        return {
            "code_refs": sorted(code_refs),
            "test_refs": sorted(test_refs),
        }

    @staticmethod
    def _stable_fragment(value: str) -> str:
        digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:12]
        return digest

    def _section_node_id(self, source_path: str, section_path: list[str]) -> str:
        label = " / ".join(section_path) if section_path else source_path
        return f"section::{self._stable_fragment(source_path + '::' + label)}"

    def _should_refresh_hotspot_capsules(
        self,
        payload: dict[str, Any],
        hotspot_manifest: dict[str, Any],
    ) -> bool:
        requirements = list(payload.get("requirements") or [])
        if not any(self._requires_hotspot_capsules(item) for item in requirements):
            return False
        if int(hotspot_manifest.get("capsule_count", 0) or 0) <= 0:
            return True
        generated_at = float(hotspot_manifest.get("generated_at_epoch", 0.0) or 0.0)
        if generated_at <= 0:
            return True
        return (time.time() - generated_at) > 3600.0

    def _gate_context(
        self,
        requirement: dict[str, Any],
        records: list[dict[str, Any]],
        *,
        hotspot_manifest: dict[str, Any],
        runtime_symbols: dict[str, Any],
    ) -> dict[str, Any]:
        statement = str(requirement.get("text_raw") or "")
        declared = self._declared_artifacts(statement)
        record_code_refs = self._ref_list(records, "code_refs")
        relevant_paths = set(record_code_refs) | set(declared["code_refs"])
        relevant_capsules = [
            capsule
            for capsule in list(hotspot_manifest.get("capsules") or [])
            if self._capsule_matches_requirement(capsule, relevant_paths)
        ]
        matched_capsules = bool(relevant_capsules)
        if self._requires_hotspot_capsules(requirement) and not relevant_capsules:
            relevant_capsules = list(hotspot_manifest.get("capsules") or [])
        stale_evidence_count = sum(
            1
            for capsule in relevant_capsules
            if self._capsule_is_stale(
                capsule,
                relevant_paths if matched_capsules else set(),
            )
        )
        contradictory_evidence_count = sum(
            1
            for capsule in relevant_capsules
            if (capsule.get("completeness") or {}).get("contradictions")
            or not (capsule.get("completeness") or {}).get("complete", False)
        )
        return {
            "requires_hotspot_capsules": self._requires_hotspot_capsules(requirement),
            "requires_runtime_symbols": self._requires_runtime_symbols(requirement),
            "authoritative_contract": self._is_authoritative_roadmap(requirement),
            "capsule_count": len(relevant_capsules),
            "capsule_scope": (
                "matched_requirement_paths"
                if matched_capsules
                else ("manifest_fallback" if relevant_capsules else "none")
            ),
            "stale_evidence_count": stale_evidence_count,
            "contradictory_evidence_count": contradictory_evidence_count,
            "runtime_symbol_coverage_pct": float(
                runtime_symbols.get("coverage_percent", 0.0) or 0.0
            ),
            "artifact_path": str(hotspot_manifest.get("artifact_path") or ""),
        }

    def _persist_gate_report(
        self,
        path: str,
        requirements: list[dict[str, Any]],
        *,
        hotspot_manifest: dict[str, Any],
        runtime_symbols: dict[str, Any],
    ) -> dict[str, Any]:
        os.makedirs(os.path.join(self.repo_path, ".anvil", "validation"), exist_ok=True)
        violations: list[dict[str, Any]] = []
        summary = {
            "stale_evidence_count": 0,
            "contradictory_evidence_count": 0,
            "blocked_promotion_count": 0,
            "runtime_symbol_coverage_pct": float(
                runtime_symbols.get("coverage_percent", 0.0) or 0.0
            ),
            "hotspot_capsule_count": int(hotspot_manifest.get("capsule_count", 0) or 0),
        }
        requirement_rows = []
        for item in requirements:
            gate_context = dict(item.get("gate_context") or {})
            blockers = self._blockers(item)
            stale_count = int(gate_context.get("stale_evidence_count", 0) or 0)
            contradictory_count = int(
                gate_context.get("contradictory_evidence_count", 0) or 0
            )
            summary["stale_evidence_count"] += stale_count
            summary["contradictory_evidence_count"] += contradictory_count
            if {
                "runtime_symbol_coverage_below_gate",
                "missing_hotspot_capsules",
                "stale_hotspot_evidence",
                "contradictory_hotspot_evidence",
            } & set(blockers):
                summary["blocked_promotion_count"] += 1
            if blockers:
                violations.extend(
                    {
                        "file": str(item.get("requirement", {}).get("file") or path),
                        "line": 1,
                        "rule_id": blocker.upper(),
                        "message": blocker,
                        "severity": self._blocker_severity(blocker),
                    }
                    for blocker in blockers
                )
            requirement_rows.append(
                {
                    "id": str(item.get("requirement", {}).get("id") or ""),
                    "blockers": blockers,
                    "gate_context": gate_context,
                }
            )
        report = {
            "status": "ok",
            "roadmap_path": path,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "summary": summary,
            "runtime_symbols": runtime_symbols,
            "hotspot_manifest_path": str(hotspot_manifest.get("artifact_path") or ""),
            "requirements": requirement_rows,
            "counterexamples": AutoVerifier.build_counterexamples(violations),
        }
        artifact_path = os.path.join(
            self.repo_path,
            ".anvil",
            "validation",
            f"roadmap_gate_{self._stable_fragment(path)}.json",
        )
        with open(artifact_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        report["artifact_path"] = os.path.relpath(artifact_path, self.repo_path).replace(
            "\\", "/"
        )
        return report

    @staticmethod
    def _requires_hotspot_capsules(requirement: dict[str, Any]) -> bool:
        text = " ".join(
            [
                str(requirement.get("text_raw") or ""),
                " ".join(list(requirement.get("heading_path") or [])),
            ]
        ).lower()
        return any(
            token in text
            for token in (
                "hotspot",
                "evidence capsule",
                "hotspot capsule",
                "schedule",
                "kernel",
            )
        )

    @staticmethod
    def _requires_runtime_symbols(requirement: dict[str, Any]) -> bool:
        text = " ".join(
            [
                str(requirement.get("text_raw") or ""),
                " ".join(list(requirement.get("heading_path") or [])),
            ]
        ).lower()
        return any(
            token in text
            for token in (
                "runtime symbol",
                "ffi",
                "symbol closure",
                "native wrapper",
                "binding",
            )
        )

    def _is_authoritative_roadmap(self, requirement: dict[str, Any]) -> bool:
        source_path = str(requirement.get("file") or requirement.get("source_path") or "")
        filename = os.path.basename(source_path).lower()
        section_path = [str(part) for part in list(requirement.get("section_path") or [])]
        title = section_path[0].lower() if section_path else ""
        section_text = " / ".join(section_path).lower()
        relative_path = (
            os.path.relpath(
                os.path.join(self.repo_path, source_path),
                self.repo_path,
            ).replace("\\", "/")
            if source_path
            else ""
        )
        overrides = self._authoritative_roadmap_overrides()
        if source_path and (
            source_path in overrides
            or relative_path in overrides
            or filename in overrides
        ):
            return True
        text = " ".join(
            [
                filename,
                title,
                section_text,
                str(requirement.get("text_raw") or ""),
            ]
        ).lower()
        if any(
            token in text
            for token in (
                "research roadmap",
                "inventive research",
                "roadmap prompt",
                "research prompt",
            )
        ) or any(token in filename for token in ("research", "prompt")):
            return False
        return "roadmap" in filename

    def _authoritative_roadmap_overrides(self) -> set[str]:
        override_path = os.path.join(
            self.repo_path,
            ".anvil",
            "validation",
            "roadmap_authority_overrides.json",
        )
        if not os.path.exists(override_path):
            return set()
        try:
            with open(override_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return set()
        entries = payload.get("authoritative_paths") if isinstance(payload, dict) else []
        return {
            str(item).strip()
            for item in list(entries or [])
            if str(item).strip()
        }

    @staticmethod
    def _capsule_matches_requirement(
        capsule: dict[str, Any],
        relevant_paths: set[str],
    ) -> bool:
        source_path = str(capsule.get("source", {}).get("file") or "")
        if not relevant_paths:
            return True
        return source_path in relevant_paths

    def _capsule_staleness_paths(
        self,
        capsule: dict[str, Any],
        relevant_paths: set[str],
    ) -> set[str]:
        normalized_relevant = {
            path
            for path in (
                self._normalize_repo_relpath(item) for item in sorted(relevant_paths)
            )
            if path
        }
        source_path = self._normalize_repo_relpath(
            str(capsule.get("source", {}).get("file") or "")
        )
        if source_path and source_path in normalized_relevant:
            return {source_path}
        if source_path:
            return {source_path}
        return normalized_relevant

    def _normalize_repo_relpath(self, path: str) -> str:
        normalized = str(path or "").strip()
        if not normalized:
            return ""
        normalized = normalized.replace("\\", "/")
        if os.path.isabs(normalized):
            try:
                relative = os.path.relpath(normalized, self.repo_path).replace("\\", "/")
            except ValueError:
                return ""
            if relative.startswith("../"):
                return ""
            return relative
        return normalized.lstrip("./")

    def _capsule_is_stale(
        self,
        capsule: dict[str, Any],
        relevant_paths: set[str],
    ) -> bool:
        generated_at = float(capsule.get("generated_at_epoch", 0.0) or 0.0)
        if generated_at <= 0:
            return True
        comparison_paths = self._capsule_staleness_paths(capsule, relevant_paths)
        if not comparison_paths:
            return True
        for rel_path in comparison_paths:
            abs_path = os.path.join(self.repo_path, rel_path)
            if os.path.exists(abs_path) and os.path.getmtime(abs_path) > generated_at:
                return True
        return False
