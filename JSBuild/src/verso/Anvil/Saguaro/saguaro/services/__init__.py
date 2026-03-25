"""Internal service layer for CPU-first Saguaro platform features."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "AppService",
    "ComparativeAnalysisService",
    "EvidenceService",
    "EvalService",
    "GraphService",
    "MetricsService",
    "ParseService",
    "QueryService",
    "ResearchService",
    "VerifyService",
]


def __getattr__(name: str):
    if name == "ComparativeAnalysisService":
        return import_module(".comparative", __name__).ComparativeAnalysisService
    if name in {
        "AppService",
        "EvalService",
        "EvidenceService",
        "GraphService",
        "MetricsService",
        "ParseService",
        "QueryService",
        "ResearchService",
        "VerifyService",
    }:
        return getattr(import_module(".platform", __name__), name)
    raise AttributeError(name)
