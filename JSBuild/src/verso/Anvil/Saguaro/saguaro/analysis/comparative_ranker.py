from __future__ import annotations

from collections import defaultdict
from typing import Any


class ComparativeRankFusion:
    """Fuse multiple ranking channels and calibrate recommendation confidence."""

    def __init__(self, *, rrf_k: int = 60) -> None:
        self.rrf_k = max(1, int(rrf_k))

    def fuse(
        self,
        *,
        channel_rows: dict[str, list[tuple[int, int, float, list[str]]]],
        top_k: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        fused: dict[tuple[int, int], dict[str, Any]] = {}
        channel_scores: dict[str, dict[str, float]] = {}
        for channel_name, rows in sorted(channel_rows.items()):
            sorted_rows = sorted(rows, key=lambda item: (-float(item[2]), item[0], item[1]))
            channel_scores[channel_name] = {}
            for rank, (target_index, candidate_index, raw_score, evidence) in enumerate(
                sorted_rows,
                start=1,
            ):
                key = (int(target_index), int(candidate_index))
                fused_row = fused.setdefault(
                    key,
                    {
                        "target_index": int(target_index),
                        "candidate_index": int(candidate_index),
                        "rrf_score": 0.0,
                        "best_raw_score": 0.0,
                        "evidence": [],
                        "channels": set(),
                        "channel_scores": {},
                    },
                )
                rrf_score = 1.0 / float(self.rrf_k + rank)
                fused_row["rrf_score"] += rrf_score
                fused_row["best_raw_score"] = max(
                    float(fused_row.get("best_raw_score") or 0.0),
                    float(raw_score),
                )
                fused_row["evidence"].extend(list(evidence or []))
                fused_row["channels"].add(channel_name)
                fused_row["channel_scores"][channel_name] = round(float(raw_score), 4)
                channel_scores[channel_name][f"{target_index}:{candidate_index}"] = round(
                    float(raw_score),
                    4,
                )
        fused_rows = []
        for row in fused.values():
            row["fused_score"] = round(
                float(row["rrf_score"]) + (0.35 * float(row["best_raw_score"] or 0.0)),
                4,
            )
            row["channels"] = sorted(row["channels"])
            row["evidence"] = sorted(set(str(item) for item in row["evidence"]))[:16]
            fused_rows.append(row)
        fused_rows.sort(
            key=lambda item: (
                -float(item.get("fused_score") or 0.0),
                -float(item.get("best_raw_score") or 0.0),
                int(item.get("candidate_index") or 0),
                int(item.get("target_index") or 0),
            )
        )
        top_rows = fused_rows[: max(1, int(top_k))]
        top_scores = [float(item.get("fused_score") or 0.0) for item in top_rows[:2]]
        margin = 0.0
        if top_scores:
            margin = top_scores[0] - (top_scores[1] if len(top_scores) > 1 else 0.0)
        telemetry = {
            "channel_scores": channel_scores,
            "fused_rank_count": len(top_rows),
            "top1_top2_margin": round(float(margin), 4),
        }
        return top_rows, telemetry

    def calibrate_relation(
        self,
        *,
        relation_score: float,
        actionability_score: float,
        feature_families: list[str],
        shared_role_tags: list[str],
        evidence_channels: list[str],
        noise_flags: list[str],
        build_alignment: dict[str, Any],
        target_pack: dict[str, Any],
        candidate_pack: dict[str, Any],
        top_margin: float,
    ) -> dict[str, Any]:
        confidence = (
            0.34 * float(relation_score)
            + 0.33 * float(actionability_score)
            + min(0.12, 0.03 * len(feature_families))
            + min(0.1, 0.025 * len(shared_role_tags))
            + min(0.12, 0.03 * len(evidence_channels))
            + min(0.08, 0.5 * float(top_margin))
        )
        negative_evidence: list[dict[str, Any]] = []
        build_depth = str(
            ((build_alignment.get("candidate") or {}).get("build_fingerprint_depth") or "shallow")
        )
        if build_depth == "shallow":
            confidence -= 0.08
            negative_evidence.append(
                {
                    "kind": "build_truth",
                    "reason": "candidate_build_truth_shallow",
                    "severity": "medium",
                }
            )
        test_surface_count = int(
            len(candidate_pack.get("test_files") or []) + len(target_pack.get("test_files") or [])
        )
        if test_surface_count == 0:
            confidence -= 0.08
            negative_evidence.append(
                {
                    "kind": "verification",
                    "reason": "missing_test_surface",
                    "severity": "medium",
                }
            )
        if len(noise_flags) >= 2:
            confidence -= 0.12
            negative_evidence.append(
                {
                    "kind": "noise",
                    "reason": "multiple_noise_flags",
                    "severity": "high",
                }
            )
        unsupported_regions = set(
            ((candidate_pack.get("language_truth_matrix") or {}).get("unsupported_language_regions") or [])
        )
        candidate_languages = set((candidate_pack.get("languages") or {}).keys())
        if unsupported_regions & candidate_languages:
            confidence -= 0.06
            negative_evidence.append(
                {
                    "kind": "parser_support",
                    "reason": "unsupported_language_regions_present",
                    "severity": "medium",
                }
            )
        confidence = max(0.0, min(0.99, confidence))
        abstain_reason = None
        if confidence < 0.28:
            abstain_reason = "low_calibrated_confidence"
        elif not feature_families and len(shared_role_tags) < 2 and confidence < 0.38:
            abstain_reason = "insufficient_mechanism_evidence"
        elif build_depth == "shallow" and float(top_margin) < 0.015 and confidence < 0.42:
            abstain_reason = "weak_build_grounding"
        label = "high" if confidence >= 0.78 else ("medium" if confidence >= 0.52 else "low")
        return {
            "calibrated_confidence": round(confidence, 4),
            "confidence_label": label,
            "abstain_reason": abstain_reason,
            "negative_evidence": negative_evidence,
            "no_port": abstain_reason is not None,
        }


def compress_program_groups(
    relations: list[dict[str, Any]],
    *,
    max_groups: int = 12,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str, str, str, tuple[str, ...]], list[dict[str, Any]]] = defaultdict(list)
    for relation in relations:
        key = (
            str(relation.get("recommended_subsystem") or ""),
            str(relation.get("target_path") or ""),
            str(relation.get("posture") or ""),
            tuple(sorted(str(item) for item in relation.get("feature_families") or [])),
        )
        grouped[key].append(relation)
    program_groups = []
    duplicate_count = 0
    for (_subsystem, target_path, posture, families), rows in grouped.items():
        rows.sort(
            key=lambda item: (
                -float(item.get("calibrated_confidence") or item.get("relation_score") or 0.0),
                str(item.get("source_path") or ""),
            )
        )
        duplicate_count += max(0, len(rows) - 1)
        exemplar = dict(rows[0])
        program_groups.append(
            {
                "program_group_id": exemplar.get("recommendation_id")
                or f"group:{target_path}:{posture}",
                "recommended_subsystem": exemplar.get("recommended_subsystem"),
                "recommended_subsystem_label": exemplar.get("recommended_subsystem_label"),
                "target_path": target_path,
                "posture": posture,
                "feature_families": list(families),
                "source_count": len(rows),
                "source_paths": [str(item.get("source_path") or "") for item in rows[:8]],
                "dominant_relation_type": exemplar.get("relation_type"),
                "top_actionability_score": float(
                    exemplar.get("actionability_score") or 0.0
                ),
                "calibrated_confidence": float(
                    exemplar.get("calibrated_confidence")
                    or exemplar.get("relation_score")
                    or 0.0
                ),
                "expected_value": list(exemplar.get("expected_value") or []),
                "program_summary": exemplar.get("why_port")
                or exemplar.get("rationale")
                or "Compressed from repeated comparative relations.",
                "proof_graph_id": exemplar.get("proof_graph_id"),
            }
        )
    program_groups.sort(
        key=lambda item: (
            -float(item.get("calibrated_confidence") or 0.0),
            -int(item.get("source_count") or 0),
            str(item.get("target_path") or ""),
        )
    )
    top_groups = program_groups[: max_groups]
    telemetry = {
        "duplicate_suppression_count": duplicate_count,
        "compression_ratio": round(
            len(top_groups) / max(1, len(relations)),
            4,
        ),
        "canonical_program_count": len(top_groups),
    }
    return top_groups, telemetry
