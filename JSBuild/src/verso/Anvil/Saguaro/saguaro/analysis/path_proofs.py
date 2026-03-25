from __future__ import annotations

from typing import Any


class PathProofBuilder:
    """Build lightweight landing-zone proof graphs for comparative recommendations."""

    def build(
        self,
        *,
        relation: dict[str, Any],
        target_pack: dict[str, Any],
        candidate_pack: dict[str, Any],
    ) -> dict[str, Any]:
        source_path = str(relation.get("source_path") or "")
        target_path = str(relation.get("target_path") or "")
        feature_families = list(relation.get("feature_families") or [])
        shared_role_tags = list(
            ((relation.get("evidence") or {}).get("shared_role_tags") or [])
        )
        target_evidence = dict(
            ((target_pack.get("file_graph_evidence") or {}).get(target_path) or {})
        )
        source_evidence = dict(
            ((candidate_pack.get("file_graph_evidence") or {}).get(source_path) or {})
        )
        edges = []
        nodes = [
            {"id": f"source:{source_path}", "label": source_path, "kind": "source_path"},
            {"id": f"target:{target_path}", "label": target_path, "kind": "target_path"},
        ]
        if feature_families:
            nodes.append(
                {
                    "id": f"family:{feature_families[0]}",
                    "label": ", ".join(feature_families[:3]),
                    "kind": "feature_family",
                }
            )
            edges.append(
                {
                    "from": f"source:{source_path}",
                    "to": f"family:{feature_families[0]}",
                    "relation": "implements",
                }
            )
            edges.append(
                {
                    "from": f"family:{feature_families[0]}",
                    "to": f"target:{target_path}",
                    "relation": "lands_in",
                }
            )
        if shared_role_tags:
            nodes.append(
                {
                    "id": f"role:{shared_role_tags[0]}",
                    "label": ", ".join(shared_role_tags[:3]),
                    "kind": "shared_role",
                }
            )
            edges.append(
                {
                    "from": f"source:{source_path}",
                    "to": f"role:{shared_role_tags[0]}",
                    "relation": "operates_as",
                }
            )
            edges.append(
                {
                    "from": f"role:{shared_role_tags[0]}",
                    "to": f"target:{target_path}",
                    "relation": "aligns_with",
                }
            )
        build_targets = list(
            ((target_pack.get("build_fingerprint") or {}).get("build_hints") or {}).get("targets")
            or []
        )
        if build_targets:
            nodes.append(
                {
                    "id": f"build:{build_targets[0]}",
                    "label": str(build_targets[0]),
                    "kind": "build_target",
                }
            )
            edges.append(
                {
                    "from": f"build:{build_targets[0]}",
                    "to": f"target:{target_path}",
                    "relation": "compiled_through",
                }
            )
        dependency_edges = int(target_evidence.get("edge_count") or 0) + int(
            source_evidence.get("edge_count") or 0
        )
        confidence = min(
            0.99,
            0.4
            + (0.12 if feature_families else 0.0)
            + (0.1 if shared_role_tags else 0.0)
            + (0.08 if build_targets else 0.0)
            + min(0.15, dependency_edges * 0.01),
        )
        return {
            "proof_graph_id": relation.get("recommendation_id")
            or f"proof:{source_path}:{target_path}",
            "schema_version": "landing_zone_proof_graph.v1",
            "nodes": nodes,
            "edges": edges,
            "proof_path_count": max(1, len(edges) // 2),
            "proof_avg_depth": 2.0 if edges else 0.0,
            "landing_zone_confidence": round(confidence, 4),
            "source_file_graph": source_evidence,
            "target_file_graph": target_evidence,
            "path_summary": (
                f"{source_path} aligns with {target_path} through "
                f"{', '.join(feature_families or shared_role_tags or ['comparative evidence'])}."
            ),
        }
