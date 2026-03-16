from pathlib import Path

from core.dare.knowledge_base import DareKnowledgeBase
from core.dare.models import ResearchSource
from core.dare.web_research import WebResearchEngine


def _make_repo(root: Path) -> None:
    (root / "core").mkdir(parents=True, exist_ok=True)
    (root / "tests").mkdir(parents=True, exist_ok=True)
    (root / "core" / "__init__.py").write_text("", encoding="utf-8")
    (root / "core" / "attention.py").write_text(
        "def fused_attention(x):\n    return x\n",
        encoding="utf-8",
    )
    (root / "tests" / "test_attention.py").write_text(
        "from core.attention import fused_attention\n\n\ndef test_attention():\n    assert fused_attention(1) == 1\n",
        encoding="utf-8",
    )
    (root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")


class _StaticSource:
    def __init__(self, items):
        self.items = list(items)

    def search(self, topic: str, max_results: int = 5):
        del topic
        return self.items[:max_results]


def test_research_until_satisfied_frontier_acquires_and_analyzes_repo(
    tmp_path, monkeypatch
):
    repo_root = tmp_path / "competitor_repo"
    _make_repo(repo_root)

    kb = DareKnowledgeBase(root_dir=str(tmp_path))
    engine = WebResearchEngine(kb, workspace_root=str(tmp_path))
    engine.sources = {
        "docs": _StaticSource(
            [
                ResearchSource(
                    source_type="docs",
                    title="Official Docs",
                    url="https://example.test/docs",
                    summary="Reference docs for fused attention kernels.",
                    confidence="high",
                )
            ]
        ),
        "github": _StaticSource(
            [
                ResearchSource(
                    source_type="github",
                    title="org/fused-attention",
                    url="https://github.com/org/fused-attention",
                    summary="Optimized fused attention implementation for CPU inference.",
                    confidence="medium",
                    metadata={"stargazers_count": 4200, "language": "Python"},
                )
            ]
        ),
        "forums": _StaticSource(
            [
                ResearchSource(
                    source_type="forums",
                    title="Benchmark thread",
                    url="https://example.test/forum",
                    summary="Users discuss latency and documentation quality.",
                    confidence="medium",
                )
            ]
        ),
        "arxiv": _StaticSource(
            [
                ResearchSource(
                    source_type="arxiv",
                    title="Kernel Paper",
                    url="https://arxiv.org/abs/1234.5678",
                    summary="Describes fused attention kernels.",
                    confidence="high",
                )
            ]
        ),
        "scholar": _StaticSource([]),
    }
    monkeypatch.setattr(
        engine,
        "download_competitor_repo",
        lambda repo_url, dest_root=None: str(repo_root),
    )

    report = engine.research_until_satisfied(
        "fused attention cpu throughput", min_sources=3
    )

    assert report.metadata["mode"] == "frontier"
    assert len(report.metadata["acquired_repos"]) == 1
    acquired = report.metadata["acquired_repos"][0]
    assert acquired["local_path"] == str(repo_root)
    analysis_entries = kb.list_entries(category="analysis", limit=20)
    assert any("competitor_repo_ingestion" in entry.path for entry in analysis_entries)
    assert any("deep_analysis" in entry.path for entry in analysis_entries)


def test_competitive_analysis_records_local_paths_for_acquired_repos(
    tmp_path, monkeypatch
):
    repo_root = tmp_path / "competitor_repo"
    _make_repo(repo_root)

    kb = DareKnowledgeBase(root_dir=str(tmp_path))
    engine = WebResearchEngine(kb, workspace_root=str(tmp_path))
    engine.sources["github"] = _StaticSource(
        [
            ResearchSource(
                source_type="github",
                title="org/fused-attention",
                url="https://github.com/org/fused-attention",
                summary="Optimized fused attention implementation for CPU inference.",
                confidence="medium",
                metadata={"stargazers_count": 4200, "language": "Python"},
            )
        ]
    )
    engine.sources["forums"] = _StaticSource([])
    monkeypatch.setattr(
        engine,
        "download_competitor_repo",
        lambda repo_url, dest_root=None: str(repo_root),
    )

    report = engine.competitive_analysis("fused attention")

    assert report.metadata["acquired_repos"]
    assert report.competitors[0].local_path == str(repo_root)
