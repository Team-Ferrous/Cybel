"""Witness aggregation for requirement validation."""

from __future__ import annotations

import hashlib
import time
from typing import Any

from saguaro.requirements.model import WitnessRecord


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class WitnessAggregator:
    """Aggregate witness classes over traceability records."""

    def build(
        self,
        requirement: dict[str, Any],
        records: list[dict[str, Any]],
        *,
        generation_id: str,
    ) -> list[WitnessRecord]:
        witnesses: list[WitnessRecord] = []
        if not records:
            witnesses.append(
                self._witness(
                    requirement_id=requirement["id"],
                    witness_type="static",
                    artifact_id=requirement["file"],
                    generation_id=generation_id,
                    result="missing",
                    details={"reason": "No traceability records"},
                )
            )
            return witnesses

        strongest = max(self._record_score(item) for item in records)
        witnesses.append(
            self._witness(
                requirement_id=requirement["id"],
                witness_type="static",
                artifact_id=str(
                    records[0].get("artifact_id")
                    or records[0].get("trace_id")
                    or requirement["file"]
                ),
                generation_id=generation_id,
                result="pass" if strongest >= 0.58 else "weak",
                details={"strongest_confidence": strongest},
            )
        )
        test_records = [
            item
            for item in records
            if item.get("artifact_type") == "test_case" or list(item.get("test_refs") or [])
        ]
        witnesses.append(
            self._witness(
                requirement_id=requirement["id"],
                witness_type="test",
                artifact_id=str(
                    test_records[0].get("artifact_id")
                    or (list(test_records[0].get("test_refs") or [requirement["file"]])[0] if test_records else requirement["file"])
                ) if test_records else requirement["file"],
                generation_id=generation_id,
                result="pass" if test_records else "missing",
                details={"count": len(test_records)},
            )
        )
        config_records = [
            item
            for item in records
            if item.get("artifact_file", "").endswith((".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"))
            or any(
                str(ref).endswith((".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"))
                for ref in list(item.get("verification_refs") or [])
            )
        ]
        witnesses.append(
            self._witness(
                requirement_id=requirement["id"],
                witness_type="config",
                artifact_id=str(config_records[0].get("artifact_id") or requirement["file"]) if config_records else requirement["file"],
                generation_id=generation_id,
                result="pass" if config_records else "missing",
                details={"count": len(config_records)},
            )
        )
        bridge_records = [
            item
            for item in records
            if "bridge" in " ".join(item.get("notes") or []).lower()
            or any("bridge" in str(ref).lower() for ref in list(item.get("code_refs") or []))
        ]
        witnesses.append(
            self._witness(
                requirement_id=requirement["id"],
                witness_type="trace",
                artifact_id=str(bridge_records[0].get("artifact_id") or requirement["file"]) if bridge_records else requirement["file"],
                generation_id=generation_id,
                result="pass" if bridge_records else "missing",
                details={"count": len(bridge_records)},
            )
        )
        return witnesses

    @staticmethod
    def classify_state(records: list[dict[str, Any]], witnesses: list[WitnessRecord]) -> str:
        """Map witness posture to a validation state."""
        if not records:
            return "unimplemented"
        best = max(WitnessAggregator._record_score(item) for item in records)
        has_graph_support = any(list(item.get("graph_refs") or []) for item in records)
        passed_tests = any(item.witness_type == "test" and item.result == "pass" for item in witnesses)
        if best >= 0.74 and passed_tests:
            return "implemented_witnessed"
        if best >= 0.74:
            return "implemented_unwitnessed"
        if best >= 0.45:
            return "partially_implemented"
        if has_graph_support and best >= 0.3:
            return "partially_implemented"
        return "unknown"

    @staticmethod
    def _record_score(record: dict[str, Any]) -> float:
        if "confidence" in record:
            return float(record.get("confidence", 0.0) or 0.0)
        score = 0.2
        if list(record.get("code_refs") or []):
            score += 0.4
        if list(record.get("test_refs") or []):
            score += 0.3
        if list(record.get("graph_refs") or []):
            score += 0.1
        return min(score, 0.95)

    @staticmethod
    def _witness(
        *,
        requirement_id: str,
        witness_type: str,
        artifact_id: str,
        generation_id: str,
        result: str,
        details: dict[str, Any],
    ) -> WitnessRecord:
        digest = hashlib.sha1(
            f"{requirement_id}|{witness_type}|{artifact_id}|{generation_id}".encode("utf-8")
        ).hexdigest()[:12]
        return WitnessRecord(
            id=f"WIT-{digest}".upper(),
            requirement_id=requirement_id,
            witness_type=witness_type,
            artifact_id=artifact_id,
            result=result,
            observed_at=_now_iso(),
            generation_id=generation_id,
            details=details,
        )
