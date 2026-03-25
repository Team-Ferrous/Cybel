"""Benchmark fixtures, calibration, and scoring utilities for SAGUARO retrieval quality."""

from __future__ import annotations

import json
import os
import time
from typing import Any


DEFAULT_QUERY_CALIBRATION = {
    "high": {"min_score": 1.45, "min_margin": 0.22},
    "moderate": {"min_score": 0.95, "min_margin": 0.12},
}


def load_benchmark_cases(base_path: str) -> list[dict[str, Any]]:
    """Load benchmark cases from a fixture file or directory."""
    if os.path.isdir(base_path):
        cases: list[dict[str, Any]] = []
        for name in sorted(os.listdir(base_path)):
            if not name.endswith(".json"):
                continue
            cases.extend(load_benchmark_cases(os.path.join(base_path, name)))
        return cases
    with open(base_path, encoding="utf-8") as f:
        data = json.load(f) or []
    if isinstance(data, dict):
        data = data.get("cases", [])
    return [item for item in data if isinstance(item, dict)]


def score_benchmark_results(
    cases: list[dict[str, Any]],
    results_by_query: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Score query results against benchmark expectations."""
    total = 0
    top1_hits = 0
    top3_hits = 0
    by_category: dict[str, dict[str, Any]] = {}
    for case in cases:
        query = str(case.get("query") or "").strip()
        expected = {
            str(path).replace("\\", "/")
            for path in case.get("expected_paths", []) or []
            if path
        }
        if not query or not expected:
            continue
        total += 1
        results = results_by_query.get(query, [])
        ranked_paths = [
            str(item.get("file") or "").replace("\\", "/") for item in results[:3]
        ]
        top1 = bool(ranked_paths[:1] and ranked_paths[0] in expected)
        top3 = any(path in expected for path in ranked_paths)
        if top1:
            top1_hits += 1
        if top3:
            top3_hits += 1
        category = str(case.get("category") or "uncategorized")
        bucket = by_category.setdefault(
            category,
            {"total": 0, "top1_hits": 0, "top3_hits": 0},
        )
        bucket["total"] += 1
        bucket["top1_hits"] += int(top1)
        bucket["top3_hits"] += int(top3)

    for bucket in by_category.values():
        total_bucket = max(1, int(bucket["total"]))
        bucket["top1_precision"] = round(bucket["top1_hits"] / total_bucket, 3)
        bucket["top3_recall"] = round(bucket["top3_hits"] / total_bucket, 3)

    total = max(1, total)
    return {
        "total": total,
        "top1_hits": top1_hits,
        "top3_hits": top3_hits,
        "top1_precision": round(top1_hits / total, 3),
        "top3_recall": round(top3_hits / total, 3),
        "categories": by_category,
    }


def derive_query_calibration(
    cases: list[dict[str, Any]],
    results_by_query: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Fit score and margin bands from benchmark outcomes."""
    observations: list[dict[str, Any]] = []
    for case in cases:
        query = str(case.get("query") or "").strip()
        expected = {
            str(path).replace("\\", "/")
            for path in case.get("expected_paths", []) or []
            if path
        }
        if not query or not expected:
            continue
        rows = list(results_by_query.get(query, []) or [])
        if not rows:
            continue
        top = rows[0]
        next_score = float(rows[1].get("score", 0.0) or 0.0) if len(rows) > 1 else 0.0
        score = float(top.get("score", 0.0) or 0.0)
        margin = round(score - next_score, 6)
        top_file = str(top.get("file") or "").replace("\\", "/")
        observations.append(
            {
                "query": query,
                "score": score,
                "margin": margin,
                "hit": top_file in expected,
            }
        )

    if not observations:
        return {
            **DEFAULT_QUERY_CALIBRATION,
            "sample_size": 0,
            "recorded_at": time.time(),
            "observed": {},
        }

    high = _select_threshold(
        observations,
        precision_target=0.95,
        min_support=max(3, len(observations) // 12),
        fallback=DEFAULT_QUERY_CALIBRATION["high"],
    )
    moderate = _select_threshold(
        observations,
        precision_target=0.85,
        min_support=max(5, len(observations) // 8),
        fallback=DEFAULT_QUERY_CALIBRATION["moderate"],
    )
    if moderate["min_score"] > high["min_score"]:
        moderate["min_score"] = high["min_score"]
    if moderate["min_margin"] > high["min_margin"]:
        moderate["min_margin"] = high["min_margin"]

    return {
        "high": {
            "min_score": round(float(high["min_score"]), 6),
            "min_margin": round(float(high["min_margin"]), 6),
        },
        "moderate": {
            "min_score": round(float(moderate["min_score"]), 6),
            "min_margin": round(float(moderate["min_margin"]), 6),
        },
        "sample_size": len(observations),
        "recorded_at": time.time(),
        "observed": {
            "high_precision": round(float(high["precision"]), 3),
            "high_coverage": round(float(high["coverage"]), 3),
            "moderate_precision": round(float(moderate["precision"]), 3),
            "moderate_coverage": round(float(moderate["coverage"]), 3),
        },
    }


def load_query_calibration(saguaro_dir: str) -> dict[str, Any]:
    """Load query calibration thresholds from disk."""
    path = os.path.join(saguaro_dir, "query_calibration.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f) or {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def persist_query_calibration(saguaro_dir: str, payload: dict[str, Any]) -> None:
    """Persist query calibration thresholds for live query-time use."""
    os.makedirs(saguaro_dir, exist_ok=True)
    path = os.path.join(saguaro_dir, "query_calibration.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _select_threshold(
    observations: list[dict[str, Any]],
    *,
    precision_target: float,
    min_support: int,
    fallback: dict[str, float],
) -> dict[str, float]:
    score_candidates = sorted(
        {round(float(item["score"]), 3) for item in observations},
        reverse=True,
    )
    margin_candidates = sorted(
        {round(float(item["margin"]), 3) for item in observations},
        reverse=True,
    )
    best: dict[str, float] | None = None
    total = float(len(observations))
    for min_score in score_candidates:
        for min_margin in margin_candidates:
            subset = [
                item
                for item in observations
                if float(item["score"]) >= min_score and float(item["margin"]) >= min_margin
            ]
            if len(subset) < min_support:
                continue
            hits = sum(1 for item in subset if item["hit"])
            precision = hits / len(subset)
            if precision < precision_target:
                continue
            coverage = len(subset) / total if total else 0.0
            candidate = {
                "min_score": min_score,
                "min_margin": min_margin,
                "precision": precision,
                "coverage": coverage,
            }
            if best is None:
                best = candidate
                continue
            if (
                candidate["coverage"] > best["coverage"]
                or (
                    candidate["coverage"] == best["coverage"]
                    and candidate["min_score"] < best["min_score"]
                )
                or (
                    candidate["coverage"] == best["coverage"]
                    and candidate["min_score"] == best["min_score"]
                    and candidate["min_margin"] < best["min_margin"]
                )
            ):
                best = candidate
    if best is not None:
        return best

    fallback_subset = [
        item
        for item in observations
        if float(item["score"]) >= float(fallback["min_score"])
        and float(item["margin"]) >= float(fallback["min_margin"])
    ]
    hits = sum(1 for item in fallback_subset if item["hit"])
    precision = hits / len(fallback_subset) if fallback_subset else 0.0
    coverage = len(fallback_subset) / total if total else 0.0
    return {
        "min_score": float(fallback["min_score"]),
        "min_margin": float(fallback["min_margin"]),
        "precision": precision,
        "coverage": coverage,
    }
