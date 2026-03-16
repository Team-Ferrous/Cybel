"""Higher-rigor retrieval pipeline for SAGUARO query ranking."""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from saguaro.indexing.stats import idf_for_term
from saguaro.query.corpus_rules import canonicalize_rel_path, classify_file_role


@dataclass
class ScoreBreakdown:
    semantic: float = 0.0
    lexical: float = 0.0
    graph: float = 0.0
    exact_symbol: float = 0.0
    exact_qualified: float = 0.0
    exact_path: float = 0.0
    symbol_overlap: float = 0.0
    path_overlap: float = 0.0
    doc_overlap: float = 0.0
    domain_prior: float = 0.0
    file_prior: float = 0.0
    aggregate_bonus: float = 0.0
    stale_penalty: float = 0.0
    final: float = 0.0


@dataclass
class ConfidenceResult:
    level: str
    margin: float
    reasons: list[str] = field(default_factory=list)


@dataclass
class RetrievalCandidate:
    row: dict[str, Any]
    score_breakdown: ScoreBreakdown
    matched_features: list[str]
    confidence: ConfidenceResult | None = None


class QueryPipeline:
    """Merge retrieval signals, aggregate entities, and calibrate confidence."""

    def __init__(
        self,
        *,
        repo_path: str,
        load_stats: Callable[[], dict[str, Any]],
        extract_terms: Callable[[str, int], list[str]],
        result_is_in_repo: Callable[[dict[str, Any]], bool],
    ) -> None:
        self.repo_path = os.path.abspath(repo_path)
        self._load_stats = load_stats
        self._extract_terms = extract_terms
        self._result_is_in_repo = result_is_in_repo

    def run(
        self,
        *,
        text: str,
        strategy: str,
        semantic_rows: list[dict[str, Any]],
        lexical_rows: list[dict[str, Any]],
        graph_rows: list[dict[str, Any]],
        k: int,
        explain: bool,
        stale_files: set[str] | None = None,
        auto_refreshed: bool = False,
        auto_refreshed_files: list[str] | None = None,
    ) -> dict[str, Any]:
        query = (text or "").strip()
        query_lower = query.lower()
        query_terms = self._extract_terms(query, limit=32)
        intent = self._classify_intent(query, query_terms)
        stats = self._load_stats()
        stale_files = {canonicalize_rel_path(path, self.repo_path) for path in stale_files or set()}
        merged = self._merge_rows(semantic_rows, lexical_rows, graph_rows)
        merged.extend(self._aggregate_file_rows(merged, intent=intent, query_terms=query_terms))

        scored: list[RetrievalCandidate] = []
        for row in merged:
            if not self._result_is_in_repo(row):
                continue
            candidate = self._score_candidate(
                row,
                query=query,
                query_lower=query_lower,
                query_terms=query_terms,
                intent=intent,
                strategy=strategy,
                stats=stats,
                stale_files=stale_files,
            )
            scored.append(candidate)

        scored.sort(
            key=lambda item: (
                item.score_breakdown.final,
                item.score_breakdown.exact_qualified,
                item.score_breakdown.exact_path,
                item.score_breakdown.semantic,
            ),
            reverse=True,
        )
        self._apply_confidence(scored)

        top = scored[: max(1, int(k))]
        results: list[dict[str, Any]] = []
        for idx, candidate in enumerate(top, start=1):
            row = dict(candidate.row)
            row["rank"] = idx
            row["score"] = round(candidate.score_breakdown.final, 6)
            row["confidence"] = candidate.confidence.level if candidate.confidence else "low"
            row["matched_features"] = list(candidate.matched_features)
            row["stale"] = bool(candidate.score_breakdown.stale_penalty)
            row["reason"] = self._reason_from_candidate(candidate)
            if explain:
                row["explanation"] = {
                    "score_breakdown": asdict(candidate.score_breakdown),
                    "matched_features": list(candidate.matched_features),
                    "confidence": asdict(candidate.confidence) if candidate.confidence else {},
                    "intent": intent,
                    "index_age_seconds": self._index_age_seconds(stats),
                    "stale_candidates": sorted(stale_files),
                    "auto_refreshed": auto_refreshed,
                    "auto_refreshed_files": list(auto_refreshed_files or []),
                    "strategy": strategy,
                    "execution_strategy": strategy,
                    "matched_terms": row.get("matched_terms", []),
                    "graph_path": row.get("graph_path", []),
                    "provenance": row.get("provenance", []),
                    "candidate_pool": int(row.get("candidate_pool", 0) or 0),
                    "cpu_prefiltered": bool(row.get("cpu_prefiltered", False)),
                    "reason": row["reason"],
                }
            else:
                row.pop("explanation", None)
            results.append(row)

        return {
            "results": results,
            "intent": intent,
            "stale_candidates": sorted(stale_files),
            "auto_refreshed": auto_refreshed,
            "auto_refreshed_files": list(auto_refreshed_files or []),
            "index_age_seconds": self._index_age_seconds(stats),
        }

    def _merge_rows(
        self,
        semantic_rows: list[dict[str, Any]],
        lexical_rows: list[dict[str, Any]],
        graph_rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged: dict[tuple[str, str, int], dict[str, Any]] = {}

        def ensure(row: dict[str, Any]) -> dict[str, Any]:
            file_key = canonicalize_rel_path(str(row.get("file") or ""), self.repo_path)
            key = (
                file_key,
                str(row.get("qualified_name") or row.get("name") or ""),
                int(row.get("line", 0) or 0),
            )
            entry = merged.setdefault(
                key,
                {
                    "name": row.get("name", ""),
                    "qualified_name": row.get("qualified_name"),
                    "entity_id": row.get("entity_id"),
                    "type": row.get("type", "symbol"),
                    "file": file_key or row.get("file", ""),
                    "line": int(row.get("line", 0) or 0),
                    "end_line": int(row.get("end_line", 0) or 0),
                    "semantic_score": 0.0,
                    "lexical_score": 0.0,
                    "graph_score": 0.0,
                    "matched_terms": [],
                    "graph_path": [],
                    "candidate_pool": int(row.get("candidate_pool", 0) or 0),
                    "cpu_prefiltered": bool(row.get("cpu_prefiltered", False)),
                    "provenance": [],
                    "file_role": row.get("file_role") or classify_file_role(file_key),
                    "chunk_role": row.get("chunk_role"),
                    "parent_symbol": row.get("parent_symbol"),
                    "symbol_terms": list(row.get("symbol_terms", []) or []),
                    "path_terms": list(row.get("path_terms", []) or []),
                    "doc_terms": list(row.get("doc_terms", []) or []),
                    "stale_at_index_time": bool(row.get("stale_at_index_time", False)),
                },
            )
            return entry

        for row in semantic_rows:
            entry = ensure(row)
            entry["semantic_score"] = max(entry["semantic_score"], float(row.get("semantic_score", row.get("score", 0.0))))
            entry["lexical_score"] = max(entry["lexical_score"], float(row.get("lexical_score", 0.0)))
            entry["candidate_pool"] = max(entry["candidate_pool"], int(row.get("candidate_pool", 0) or 0))
            entry["cpu_prefiltered"] = bool(row.get("cpu_prefiltered", entry["cpu_prefiltered"]))
            entry["provenance"] = sorted(set(entry["provenance"]) | {"semantic"})
            entry["matched_terms"] = sorted(set(entry["matched_terms"]) | set(row.get("matched_terms", [])))
        for row in lexical_rows:
            entry = ensure(row)
            entry["lexical_score"] = max(entry["lexical_score"], float(row.get("lexical_score", 0.0)))
            entry["matched_terms"] = sorted(set(entry["matched_terms"]) | set(row.get("matched_terms", [])))
            entry["provenance"] = sorted(set(entry["provenance"]) | {"lexical", "graph"})
        for row in graph_rows:
            entry = ensure(row)
            entry["graph_score"] = max(entry["graph_score"], float(row.get("graph_score", 0.0)))
            entry["matched_terms"] = sorted(set(entry["matched_terms"]) | set(row.get("matched_terms", [])))
            if row.get("graph_path"):
                entry["graph_path"] = row.get("graph_path", [])
            entry["provenance"] = sorted(set(entry["provenance"]) | {"graph"})
        return list(merged.values())

    def _aggregate_file_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        intent: str,
        query_terms: list[str],
    ) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            file_key = canonicalize_rel_path(str(row.get("file") or ""), self.repo_path)
            if not file_key:
                continue
            grouped.setdefault(file_key, []).append(row)

        aggregate_rows: list[dict[str, Any]] = []
        broad = intent in {"path", "runtime", "mixed"}
        for file_key, items in grouped.items():
            if not broad and len(items) < 2:
                continue
            ordered = sorted(
                items,
                key=lambda item: (
                    float(item.get("semantic_score", 0.0))
                    + float(item.get("lexical_score", 0.0))
                    + float(item.get("graph_score", 0.0))
                ),
                reverse=True,
            )
            best = ordered[0]
            matched_terms = sorted({term for item in ordered for term in item.get("matched_terms", [])})
            matched_term_count = len(set(matched_terms) & set(query_terms))
            if matched_term_count == 0 and len(ordered) < 3:
                continue
            aggregate_rows.append(
                {
                    "name": os.path.basename(file_key),
                    "qualified_name": file_key,
                    "entity_id": f"file::{file_key}",
                    "type": "file",
                    "file": file_key,
                    "line": 1,
                    "end_line": max(int(best.get("end_line", 1) or 1), 1),
                    "semantic_score": max(float(item.get("semantic_score", 0.0)) for item in ordered[:3]),
                    "lexical_score": max(float(item.get("lexical_score", 0.0)) for item in ordered[:3]),
                    "graph_score": max(float(item.get("graph_score", 0.0)) for item in ordered[:3]),
                    "matched_terms": matched_terms,
                    "graph_path": best.get("graph_path", []),
                    "candidate_pool": max(int(item.get("candidate_pool", 0) or 0) for item in ordered[:3]),
                    "cpu_prefiltered": any(bool(item.get("cpu_prefiltered", False)) for item in ordered[:3]),
                    "provenance": sorted({prov for item in ordered[:3] for prov in item.get("provenance", [])} | {"aggregate"}),
                    "file_role": classify_file_role(file_key),
                    "chunk_role": "aggregate",
                    "aggregate_support": len(ordered[:5]),
                    "symbol_terms": sorted(
                        {
                            term
                            for item in ordered
                            for term in item.get("symbol_terms", [])
                        }
                    ),
                    "path_terms": sorted(
                        {
                            term
                            for item in ordered
                            for term in item.get("path_terms", [])
                        }
                    ),
                    "doc_terms": sorted(
                        {
                            term
                            for item in ordered
                            for term in item.get("doc_terms", [])
                        }
                    ),
                }
            )
        return aggregate_rows

    def _score_candidate(
        self,
        row: dict[str, Any],
        *,
        query: str,
        query_lower: str,
        query_terms: list[str],
        intent: str,
        strategy: str,
        stats: dict[str, Any],
        stale_files: set[str],
    ) -> RetrievalCandidate:
        name = str(row.get("name") or "")
        qualified = str(row.get("qualified_name") or "")
        file_key = canonicalize_rel_path(str(row.get("file") or ""), self.repo_path)
        file_role = str(row.get("file_role") or classify_file_role(file_key))
        row_type = str(row.get("type") or "")
        matched_terms = set(row.get("matched_terms", []) or [])
        query_term_set = set(query_terms)
        matched_query_terms = matched_terms & query_term_set

        exact_symbol = 0.0
        exact_qualified = 0.0
        exact_path = 0.0
        if name and name.lower() == query_lower:
            exact_symbol = 1.0
        elif name and name.lower() in query_lower:
            exact_symbol = 0.5
        if qualified and qualified.lower() == query_lower:
            exact_qualified = 1.2
        elif qualified and qualified.lower() in query_lower:
            exact_qualified = 0.7
        if file_key and file_key.lower() == query_lower:
            exact_path = 1.1
        elif file_key and file_key.lower() in query_lower:
            exact_path = 0.7
        else:
            filename = os.path.basename(file_key).lower()
            if filename and filename in query_lower:
                exact_path = 0.55

        semantic = float(row.get("semantic_score", row.get("score", 0.0)))
        lexical = float(row.get("lexical_score", 0.0))
        graph = float(row.get("graph_score", 0.0))
        symbol_terms = set(row.get("symbol_terms", []) or [])
        path_terms = set(row.get("path_terms", []) or []) or set(self._extract_terms(file_key, limit=24))
        doc_terms = set(row.get("doc_terms", []) or [])
        symbol_overlap = min(0.32, 0.11 * len(query_term_set & symbol_terms))
        path_overlap = min(0.28, 0.09 * len(query_term_set & path_terms))
        doc_overlap = min(0.18, 0.06 * len(query_term_set & doc_terms))
        if matched_query_terms:
            lexical += min(
                0.45,
                sum(min(0.18, 0.04 * idf_for_term(stats, term)) for term in matched_query_terms),
            )

        domain_prior = 0.0
        if file_key.startswith("core/native/") and query_term_set & {
            "native",
            "qsg",
            "runtime",
            "thread",
            "telemetry",
            "decode",
        }:
            domain_prior += 0.28
        if file_key.startswith("core/qsg/") and query_term_set & {"qsg", "engine", "continuous"}:
            domain_prior += 0.16
        if file_key.startswith(("audit/", "benchmarks/")) and query_term_set & {
            "benchmark",
            "audit",
            "runner",
        }:
            domain_prior += 0.22
        if file_key.startswith("saguaro/query/") and query_term_set & {
            "query",
            "ranking",
            "retrieval",
            "confidence",
            "calibration",
            "eval",
        }:
            domain_prior += 0.24
        if file_key.startswith("saguaro/native/") and "saguaro" not in query_term_set and query_term_set & {
            "native",
            "runtime",
            "telemetry",
            "thread",
            "decode",
        }:
            domain_prior -= 0.18
        if file_key.startswith("core/telemetry/") and query_term_set & {"qsg", "native"}:
            domain_prior -= 0.08

        file_prior = 0.0
        if row_type == "file_summary":
            file_prior += 0.42 if intent in {"path", "runtime", "mixed"} else 0.18
        elif row_type == "section":
            file_prior += 0.26 if intent in {"runtime", "mixed"} else 0.08
        elif row_type == "class":
            file_prior += 0.24 if self._looks_symbolic(query) else 0.06
        elif row_type == "file":
            file_prior += 0.22 if intent in {"path", "runtime", "mixed"} else 0.04
        if row.get("chunk_role") == "aggregate":
            file_prior += 0.34 if intent in {"path", "runtime", "mixed"} else 0.08
        if file_role == "source":
            file_prior += 0.05
        elif file_role == "test" and "test" not in query_terms and "tests" not in query_terms:
            file_prior -= 0.22
        elif file_role == "doc" and "doc" not in query_terms and "docs" not in query_terms:
            file_prior -= 0.16

        aggregate_bonus = 0.0
        if row.get("aggregate_support"):
            aggregate_bonus += min(0.24, 0.06 * int(row.get("aggregate_support", 0) or 0))
        if len(matched_query_terms) >= 2:
            aggregate_bonus += min(0.2, 0.08 * len(matched_query_terms))

        stale_penalty = 0.0
        if file_key in stale_files or bool(row.get("stale_at_index_time", False)):
            stale_penalty = 0.38

        weights = {
            "semantic": (0.7, 0.15, 0.05),
            "lexical": (0.0, 0.8, 0.1),
            "graph": (0.0, 0.3, 0.55),
            "hybrid": (0.42, 0.28, 0.14),
        }.get(strategy, (0.42, 0.28, 0.14))
        final = (
            semantic * weights[0]
            + lexical * weights[1]
            + graph * weights[2]
            + exact_symbol * 0.38
            + exact_qualified * 0.34
            + exact_path * 0.31
            + symbol_overlap
            + path_overlap
            + doc_overlap
            + domain_prior
            + file_prior
            + aggregate_bonus
            - stale_penalty
        )
        if row_type in {"method", "function"} and intent in {"path", "runtime", "mixed"}:
            final -= 0.08
        if row_type in {"file_summary", "file"} and intent == "symbol":
            final -= 0.05

        breakdown = ScoreBreakdown(
            semantic=round(semantic, 6),
            lexical=round(lexical, 6),
            graph=round(graph, 6),
            exact_symbol=round(exact_symbol, 6),
            exact_qualified=round(exact_qualified, 6),
            exact_path=round(exact_path, 6),
            symbol_overlap=round(symbol_overlap, 6),
            path_overlap=round(path_overlap, 6),
            doc_overlap=round(doc_overlap, 6),
            domain_prior=round(domain_prior, 6),
            file_prior=round(file_prior, 6),
            aggregate_bonus=round(aggregate_bonus, 6),
            stale_penalty=round(stale_penalty, 6),
            final=round(final, 6),
        )
        matched_features = []
        if exact_symbol:
            matched_features.append("exact_symbol")
        if exact_qualified:
            matched_features.append("exact_qualified_name")
        if exact_path:
            matched_features.append("exact_path")
        if symbol_overlap:
            matched_features.append("symbol_overlap")
        if path_overlap:
            matched_features.append("path_overlap")
        if doc_overlap:
            matched_features.append("doc_overlap")
        if domain_prior:
            matched_features.append("domain_prior")
        if matched_query_terms:
            matched_features.append("term_overlap")
        if graph > 0.0:
            matched_features.append("graph_proximity")
        if row.get("chunk_role") == "aggregate":
            matched_features.append("file_aggregate")
        if row_type == "file_summary":
            matched_features.append("file_summary")
        if stale_penalty:
            matched_features.append("stale_penalty")
        return RetrievalCandidate(row=row, score_breakdown=breakdown, matched_features=matched_features)

    @staticmethod
    def _looks_symbolic(text: str) -> bool:
        return bool(re.search(r"[A-Z][A-Za-z0-9_]+", text or "")) or "." in (text or "")

    def _classify_intent(self, query: str, terms: list[str]) -> str:
        if self._looks_symbolic(query):
            return "symbol"
        if "/" in query or "\\" in query or query.endswith((".py", ".cpp", ".cc", ".md")):
            return "path"
        if "_" in query or len(terms) >= 5:
            return "runtime"
        return "mixed"

    def _apply_confidence(self, scored: list[RetrievalCandidate]) -> None:
        if not scored:
            return
        calibration = self._load_stats().get("query_confidence", {}) or {}
        high = calibration.get("high", {}) or {}
        moderate = calibration.get("moderate", {}) or {}
        high_min_score = float(high.get("min_score", 1.45) or 1.45)
        high_min_margin = float(high.get("min_margin", 0.22) or 0.22)
        moderate_min_score = float(moderate.get("min_score", 0.95) or 0.95)
        moderate_min_margin = float(moderate.get("min_margin", 0.12) or 0.12)
        for idx, candidate in enumerate(scored):
            next_score = scored[idx + 1].score_breakdown.final if idx + 1 < len(scored) else 0.0
            margin = round(candidate.score_breakdown.final - next_score, 6)
            reasons: list[str] = []
            if candidate.score_breakdown.stale_penalty:
                level = "low"
                reasons.append("stale target file")
            elif margin < 0.05 or candidate.score_breakdown.final < 0.45:
                level = "abstain"
                reasons.append("insufficient score separation")
            elif candidate.score_breakdown.final >= high_min_score and margin >= high_min_margin:
                level = "high"
                reasons.append("dominant ranked result")
            elif candidate.score_breakdown.final >= moderate_min_score and margin >= moderate_min_margin:
                level = "moderate"
                reasons.append("clear but not dominant winner")
            else:
                level = "low"
                reasons.append("weak evidence margin")
            candidate.confidence = ConfidenceResult(level=level, margin=margin, reasons=reasons)

    @staticmethod
    def _reason_from_candidate(candidate: RetrievalCandidate) -> str:
        features = candidate.matched_features or ["scored_match"]
        confidence = candidate.confidence.level if candidate.confidence else "low"
        return f"{confidence} confidence via {', '.join(features[:4])}."

    @staticmethod
    def _index_age_seconds(stats: dict[str, Any]) -> float | None:
        last_index = stats.get("last_index", {}) or {}
        timestamp = last_index.get("timestamp")
        if not timestamp:
            return None
        try:
            import time

            return round(max(0.0, time.time() - float(timestamp)), 3)
        except Exception:
            return None
