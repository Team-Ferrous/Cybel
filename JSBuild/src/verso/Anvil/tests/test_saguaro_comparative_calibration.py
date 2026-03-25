from __future__ import annotations

from pathlib import Path

from saguaro.services.comparative import ComparativeAnalysisService


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_compare_emits_calibrated_confidence_and_negative_evidence(tmp_path: Path) -> None:
    target = tmp_path / "target"
    external = tmp_path / "repo_analysis" / "candidate"
    _write(
        target / "core" / "registry.py",
        "class Registry:\n    def add(self, name: str) -> str:\n        return name.lower()\n",
    )
    _write(
        external / "counterfit" / "core" / "targets.py",
        "class Registry:\n    def add(self, name: str) -> str:\n        return name.strip().lower()\n",
    )

    report = ComparativeAnalysisService(str(target)).compare(
        candidates=[str(external)],
        top_k=8,
    )

    comparison = report["comparisons"][0]
    ranked = comparison["primary_recommendations"] + comparison["secondary_recommendations"]
    assert ranked
    assert comparison["summary"]["rank_fusion"]["top1_top2_margin"] >= 0.0
    assert all("calibrated_confidence" in item for item in ranked)
    assert all("negative_evidence" in item for item in ranked)
    assert report["negative_evidence"] is not None
