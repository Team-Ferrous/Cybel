"""Benchmark helpers for ALMF retrieval and replay."""

from __future__ import annotations

from collections import defaultdict
import math
import statistics
import time
from typing import Any, Dict, Iterable, List

from core.memory.fabric.policies import LatentCompatibilityPolicy
from core.memory.fabric.retrieval_planner import MemoryRetrievalPlanner
from core.memory.fabric.store import MemoryFabricStore


def recall_at_k(ranked_ids: Iterable[str], relevant_ids: Iterable[str], k: int) -> float:
    relevant = {str(item) for item in relevant_ids if item}
    if not relevant:
        return 1.0
    hits = len(relevant & set(list(ranked_ids)[: max(1, int(k))]))
    return float(hits) / float(len(relevant))


def ndcg_at_k(ranked_ids: Iterable[str], relevant_ids: Iterable[str], k: int) -> float:
    ranked = list(ranked_ids)[: max(1, int(k))]
    relevant = {str(item) for item in relevant_ids if item}
    if not relevant:
        return 1.0
    dcg = 0.0
    for index, item in enumerate(ranked):
        if item in relevant:
            dcg += 1.0 / math.log2(float(index) + 2.0)
    ideal = sum(1.0 / math.log2(float(index) + 2.0) for index in range(min(len(relevant), len(ranked))))
    if ideal <= 0:
        return 0.0
    return dcg / ideal


def false_memory_rate(ranked_ids: Iterable[str], relevant_ids: Iterable[str]) -> float:
    ranked = [str(item) for item in ranked_ids if item]
    if not ranked:
        return 0.0
    relevant = {str(item) for item in relevant_ids if item}
    false_hits = sum(1 for item in ranked if item not in relevant)
    return float(false_hits) / float(len(ranked))


class MemoryBenchmarkRunner:
    """Run ALMF benchmark families against the current store."""

    def __init__(
        self,
        store: MemoryFabricStore,
        planner: MemoryRetrievalPlanner,
    ) -> None:
        self.store = store
        self.planner = planner

    def run_suite(
        self,
        *,
        campaign_id: str,
        cases: List[Dict[str, Any]],
        gate_thresholds: Dict[str, float] | None = None,
    ) -> Dict[str, Any]:
        started = time.time()
        case_results = [self.run_case(campaign_id=campaign_id, case=case) for case in cases]
        families: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for result in case_results:
            families[str(result.get("family") or "unknown")].append(result)

        family_metrics = {}
        for family, items in families.items():
            family_metrics[family] = {
                "case_count": len(items),
                "top_k_evidence_recall": _mean(item["metrics"]["top_k_evidence_recall"] for item in items),
                "contradiction_recall": _mean(item["metrics"]["contradiction_recall"] for item in items),
                "ndcg": _mean(item["metrics"]["ndcg"] for item in items),
                "answer_grounding_score": _mean(item["metrics"]["answer_grounding_score"] for item in items),
                "false_memory_rate": _mean(item["metrics"]["false_memory_rate"] for item in items),
                "exact_replay_success_rate": _mean(item["metrics"]["exact_replay_success_rate"] for item in items),
                "degraded_replay_success_rate": _mean(item["metrics"]["degraded_replay_success_rate"] for item in items),
                "warm_start_token_savings": _mean(item["metrics"]["warm_start_token_savings"] for item in items),
            }

        thresholds = {
            "top_k_evidence_recall": 0.6,
            "ndcg": 0.55,
            "answer_grounding_score": 0.5,
            "false_memory_rate_max": 0.5,
        }
        thresholds.update(gate_thresholds or {})
        gates = self.evaluate_gates(family_metrics, thresholds)
        return {
            "schema_version": "almf.benchmark.v1",
            "campaign_id": campaign_id,
            "started_at": started,
            "completed_at": time.time(),
            "case_results": case_results,
            "family_metrics": family_metrics,
            "core_metrics": {
                "top_k_evidence_recall": _mean(
                    item["metrics"]["top_k_evidence_recall"] for item in case_results
                ),
                "contradiction_recall": _mean(
                    item["metrics"]["contradiction_recall"] for item in case_results
                ),
                "ndcg": _mean(item["metrics"]["ndcg"] for item in case_results),
                "answer_grounding_score": _mean(
                    item["metrics"]["answer_grounding_score"] for item in case_results
                ),
                "false_memory_rate": _mean(
                    item["metrics"]["false_memory_rate"] for item in case_results
                ),
            },
            "benchmark_gates": gates,
        }

    def run_case(self, *, campaign_id: str, case: Dict[str, Any]) -> Dict[str, Any]:
        query_text = str(case.get("query_text") or "")
        limit = int(case.get("limit") or 5)
        result = self.planner.retrieve(
            campaign_id=str(case.get("campaign_id") or campaign_id),
            query_text=query_text,
            planner_mode=str(case.get("planner_mode") or "benchmark"),
            memory_kinds=case.get("memory_kinds"),
            repo_context=case.get("repo_context"),
            limit=limit,
        )
        ranked_ids = [str(item.get("memory_id") or "") for item in result.get("results", [])]
        expected_ids = list(case.get("expected_memory_ids") or [])
        contradiction_ids = list(case.get("contradiction_memory_ids") or [])

        replay_metrics = self._evaluate_replay(case, expected_ids)
        metrics = {
            "top_k_evidence_recall": recall_at_k(ranked_ids, expected_ids, limit),
            "contradiction_recall": recall_at_k(ranked_ids, contradiction_ids, limit),
            "ndcg": ndcg_at_k(ranked_ids, expected_ids, limit),
            "answer_grounding_score": self._grounding_score(ranked_ids, expected_ids),
            "false_memory_rate": false_memory_rate(ranked_ids, expected_ids),
            **replay_metrics,
        }
        return {
            "family": str(case.get("family") or "research_recall"),
            "query_text": query_text,
            "expected_memory_ids": expected_ids,
            "retrieved_memory_ids": ranked_ids,
            "metrics": metrics,
            "latency_ms": float(result.get("latency_ms") or 0.0),
            "read_id": result.get("read_id"),
        }

    def evaluate_gates(
        self,
        family_metrics: Dict[str, Dict[str, Any]],
        thresholds: Dict[str, float],
    ) -> Dict[str, Any]:
        failures = []
        for family, metrics in sorted(family_metrics.items()):
            if metrics["top_k_evidence_recall"] < float(thresholds["top_k_evidence_recall"]):
                failures.append(f"{family}: top_k_evidence_recall below threshold")
            if metrics["ndcg"] < float(thresholds["ndcg"]):
                failures.append(f"{family}: ndcg below threshold")
            if metrics["answer_grounding_score"] < float(thresholds["answer_grounding_score"]):
                failures.append(f"{family}: answer_grounding_score below threshold")
            if metrics["false_memory_rate"] > float(thresholds["false_memory_rate_max"]):
                failures.append(f"{family}: false_memory_rate above threshold")
        return {
            "passed": not failures,
            "failures": failures,
            "thresholds": thresholds,
        }

    def _evaluate_replay(self, case: Dict[str, Any], expected_ids: List[str]) -> Dict[str, float]:
        replay_memory_id = str(
            case.get("replay_memory_id")
            or (expected_ids[0] if expected_ids else "")
        )
        if not replay_memory_id:
            return {
                "exact_replay_success_rate": 0.0,
                "degraded_replay_success_rate": 0.0,
                "warm_start_token_savings": 0.0,
            }
        package = self.store.latest_latent_package(replay_memory_id)
        if package is None:
            return {
                "exact_replay_success_rate": 0.0,
                "degraded_replay_success_rate": 0.0,
                "warm_start_token_savings": 0.0,
            }
        compatibility = LatentCompatibilityPolicy.evaluate(
            package,
            model_family=str(case.get("model_family") or package.get("model_family") or "qsg-python"),
            hidden_dim=int(case.get("hidden_dim") or package.get("hidden_dim") or 1),
            tokenizer_hash=str(case.get("tokenizer_hash") or package.get("tokenizer_hash") or ""),
            prompt_protocol_hash=str(
                case.get("prompt_protocol_hash") or package.get("prompt_protocol_hash") or ""
            ),
            qsg_runtime_version=str(
                case.get("qsg_runtime_version") or package.get("qsg_runtime_version") or ""
            ),
            quantization_profile=str(
                case.get("quantization_profile") or package.get("quantization_profile") or ""
            ),
        )
        tensor = self.store.load_latent_tensor(package)
        warm_baseline = float(case.get("prompt_reconstruction_tokens") or 0.0)
        warm_restore = float(case.get("warm_start_tokens") or max(1, tensor.shape[-1] if tensor is not None else 1))
        return {
            "exact_replay_success_rate": 1.0 if compatibility["compatible"] and tensor is not None else 0.0,
            "degraded_replay_success_rate": 1.0 if tensor is not None else 0.0,
            "warm_start_token_savings": max(0.0, warm_baseline - warm_restore),
        }

    @staticmethod
    def _grounding_score(ranked_ids: List[str], expected_ids: List[str]) -> float:
        if not ranked_ids:
            return 0.0
        expected = {str(item) for item in expected_ids if item}
        if not expected:
            return 1.0
        grounded = sum(1 for item in ranked_ids if item in expected)
        return float(grounded) / float(len(ranked_ids))


def _mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    if not values:
        return 0.0
    return float(statistics.fmean(values))
