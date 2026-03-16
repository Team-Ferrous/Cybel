from __future__ import annotations

from pathlib import Path

from saguaro.services.comparative import ComparativeAnalysisService


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_compare_compiles_program_ir_and_path_proofs(tmp_path: Path) -> None:
    target = tmp_path / "target"
    external = tmp_path / "repo_analysis" / "candidate"
    _write(
        target / "analysis" / "report.py",
        "def render_report(name: str) -> str:\n    return f'report:{name}'\n",
    )
    _write(
        target / "tests" / "test_report.py",
        "def test_render_report():\n    assert True\n",
    )
    _write(
        external / "reporting" / "text.py",
        "def render_report(name: str) -> str:\n    return f'candidate:{name}'\n",
    )

    report = ComparativeAnalysisService(str(target)).compare(
        candidates=[str(external)],
        top_k=8,
    )

    comparison = report["comparisons"][0]
    recipe = comparison["migration_recipes"][0]
    relation = (
        comparison["primary_recommendations"]
        or comparison["secondary_recommendations"]
        or comparison["analogous_mechanisms"]
    )[0]

    assert recipe["lowering_ir"]["ir_version"] == "migration_program_ir.v2"
    assert recipe["lowering_ir"]["rollback_criteria"]
    assert relation["proof_graph"]["path_summary"]
    assert relation["proof_graph"]["landing_zone_confidence"] >= 0.0
