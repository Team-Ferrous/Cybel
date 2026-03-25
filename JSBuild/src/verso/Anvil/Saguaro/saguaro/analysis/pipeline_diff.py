"""Diff utilities for comparing two pipeline traces.

The diff engine accepts either:
- ``PipelineTrace`` / ``PipelineStage`` dataclasses
- serialized ``dict`` payloads emitted by API/CLI trace commands
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from saguaro.analysis.pipeline_tracer import PipelineStage, PipelineTrace


@dataclass(slots=True)
class PipelineDiffResult:
    """Structured diff between two traces."""

    added_stages: list[str]
    removed_stages: list[str]
    changed_stages: list[dict[str, Any]]
    added_edges: list[str]
    removed_edges: list[str]
    changed_edges: list[dict[str, Any]]


class PipelineDiff:
    """Compares pipeline traces while preserving stable identifiers."""

    def diff(self, before: Any, after: Any) -> dict[str, Any]:
        before_trace = self._normalize_trace(before)
        after_trace = self._normalize_trace(after)
        before_stages = {
            str(stage["id"]): stage for stage in list(before_trace.get("stages") or [])
        }
        after_stages = {
            str(stage["id"]): stage for stage in list(after_trace.get("stages") or [])
        }

        before_ids = set(before_stages)
        after_ids = set(after_stages)
        added_stage_ids = sorted(after_ids - before_ids)
        removed_stage_ids = sorted(before_ids - after_ids)

        changed_stages: list[dict[str, Any]] = []
        for stage_id in sorted(before_ids.intersection(after_ids)):
            b = before_stages[stage_id]
            a = after_stages[stage_id]
            stage_delta: dict[str, Any] = {}
            if str(b.get("name") or "") != str(a.get("name") or ""):
                stage_delta["name"] = {
                    "before": b.get("name"),
                    "after": a.get("name"),
                }
            if str(b.get("file_path") or "") != str(a.get("file_path") or ""):
                stage_delta["file_path"] = {
                    "before": b.get("file_path"),
                    "after": a.get("file_path"),
                }
            if list(b.get("annotations") or []) != list(a.get("annotations") or []):
                stage_delta["annotations"] = {
                    "before": list(b.get("annotations") or []),
                    "after": list(a.get("annotations") or []),
                }
            if dict(b.get("complexity") or {}) != dict(a.get("complexity") or {}):
                stage_delta["complexity"] = {
                    "before": dict(b.get("complexity") or {}),
                    "after": dict(a.get("complexity") or {}),
                }
            if stage_delta:
                changed_stages.append({"stage_id": stage_id, "changes": stage_delta})

        before_edges = self._edge_index(list(before_trace.get("edges") or []))
        after_edges = self._edge_index(list(after_trace.get("edges") or []))
        before_edge_keys = set(before_edges)
        after_edge_keys = set(after_edges)
        added_edge_keys = sorted(after_edge_keys - before_edge_keys)
        removed_edge_keys = sorted(before_edge_keys - after_edge_keys)

        changed_edges: list[dict[str, Any]] = []
        for edge_key in sorted(before_edge_keys.intersection(after_edge_keys)):
            b_edge = before_edges[edge_key]
            a_edge = after_edges[edge_key]
            if b_edge == a_edge:
                continue
            changed_edges.append(
                {
                    "edge": edge_key,
                    "before": b_edge,
                    "after": a_edge,
                }
            )

        result = PipelineDiffResult(
            added_stages=added_stage_ids,
            removed_stages=removed_stage_ids,
            changed_stages=changed_stages,
            added_edges=added_edge_keys,
            removed_edges=removed_edge_keys,
            changed_edges=changed_edges,
        )
        return {
            "added_stages": result.added_stages,
            "removed_stages": result.removed_stages,
            "changed_stages": result.changed_stages,
            "added_edges": result.added_edges,
            "removed_edges": result.removed_edges,
            "changed_edges": result.changed_edges,
            "summary": {
                "added_stage_count": len(result.added_stages),
                "removed_stage_count": len(result.removed_stages),
                "changed_stage_count": len(result.changed_stages),
                "added_edge_count": len(result.added_edges),
                "removed_edge_count": len(result.removed_edges),
                "changed_edge_count": len(result.changed_edges),
            },
        }

    @staticmethod
    def _edge_index(edges: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for edge in edges:
            src = str(edge.get("from") or "")
            dst = str(edge.get("to") or "")
            rel = str(edge.get("relation") or "")
            cond = str(edge.get("condition") or "")
            label = str(edge.get("label") or "")
            key = f"{src}->{dst}:{rel}:{cond}:{label}"
            if key:
                out[key] = dict(edge)
        return out

    @classmethod
    def _normalize_trace(cls, trace: Any) -> dict[str, Any]:
        """Normalize trace payloads to a dict-based shape for stable diffing."""
        if isinstance(trace, PipelineTrace):
            return {
                "stages": [cls._normalize_stage(stage) for stage in list(trace.stages or [])],
                "edges": [dict(edge) for edge in list(trace.edges or [])],
            }
        if isinstance(trace, dict):
            raw_stages = trace.get("stages") or []
            raw_edges = trace.get("edges") or []
            return {
                "stages": [cls._normalize_stage(stage) for stage in list(raw_stages)],
                "edges": [dict(edge) for edge in list(raw_edges) if isinstance(edge, dict)],
            }
        return {"stages": [], "edges": []}

    @staticmethod
    def _normalize_stage(stage: Any) -> dict[str, Any]:
        if isinstance(stage, dict):
            return {
                "id": str(stage.get("id") or ""),
                "name": str(stage.get("name") or ""),
                "file_path": str(stage.get("file_path") or stage.get("file") or ""),
                "annotations": list(stage.get("annotations") or []),
                "complexity": dict(stage.get("complexity") or {}),
            }
        if isinstance(stage, PipelineStage):
            return {
                "id": str(stage.id),
                "name": str(stage.name),
                "file_path": str(stage.file_path),
                "annotations": list(stage.annotations or []),
                "complexity": dict(stage.complexity or {}),
            }
        return {
            "id": str(getattr(stage, "id", "") or ""),
            "name": str(getattr(stage, "name", "") or ""),
            "file_path": str(getattr(stage, "file_path", "") or ""),
            "annotations": list(getattr(stage, "annotations", []) or []),
            "complexity": dict(getattr(stage, "complexity", {}) or {}),
        }


__all__ = ["PipelineDiffResult", "PipelineDiff"]
