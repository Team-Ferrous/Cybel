from __future__ import annotations

from pathlib import Path

from saguaro.analysis.report import ReportGenerator
from saguaro.services.comparative import ComparativeAnalysisService


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_compare_emits_phasepacks_and_leaderboards(tmp_path: Path) -> None:
    target = tmp_path / "target"
    external = tmp_path / "repo_analysis" / "candidate"
    _write(target / "auth.py", "def authenticate(name: str) -> str:\n    return name\n")
    _write(external / "token_auth.py", "def authenticate(name: str) -> str:\n    return name.strip()\n")

    report = ComparativeAnalysisService(str(target)).compare(
        candidates=[str(external)],
        top_k=6,
        export_datatables=True,
    )

    assert report["phase_packets"]
    assert report["portfolio_leaderboard"]
    assert Path(report["artifacts"]["phasepack_path"]).exists()
    assert Path(report["artifacts"]["leaderboard_path"]).exists()
    rendered = (tmp_path / "comparative.md")
    ReportGenerator(str(target)).save_comparative_markdown(report, str(rendered))
    content = rendered.read_text(encoding="utf-8")
    assert "## Phase Packets" in content
    assert "## Portfolio Leaderboard" in content
