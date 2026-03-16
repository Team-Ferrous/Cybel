from core.campaign.campaign_generator import CampaignGenerator
from core.dare.knowledge_base import DareKnowledgeBase
from core.dare.pipeline import DarePipeline


def _make_repo(root):
    (root / "core").mkdir()
    (root / "tests").mkdir()
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


def test_pipeline_analyze_synthesize_and_sculpt(tmp_path):
    _make_repo(tmp_path)
    pipeline = DarePipeline(root_dir=str(tmp_path))

    profiles = pipeline.analyze([str(tmp_path)])
    assert str(tmp_path) in profiles
    assert pipeline.status()["analysis_reports"] == 1

    roadmap = pipeline.synthesize({"description": "Rewrite attention stack with DARE"})
    assert roadmap.phases

    campaign_path = pipeline.sculpt()
    code = (tmp_path / ".anvil" / "campaigns" / "generated" / "rewrite_attention_stack_with_dare_campaign.py").read_text(
        encoding="utf-8"
    )
    assert campaign_path.endswith("_campaign.py")
    assert "Knowledge Hydration" in code
    assert "phase_final_verification" in code


def test_campaign_generator_accepts_dare_roadmap(tmp_path):
    _make_repo(tmp_path)
    pipeline = DarePipeline(root_dir=str(tmp_path))
    pipeline.analyze([str(tmp_path)])
    roadmap = pipeline.synthesize({"description": "Generate DARE campaign"})
    kb = DareKnowledgeBase(root_dir=str(tmp_path))

    generator = CampaignGenerator(agent=None)
    output_dir = tmp_path / "campaigns"
    path = generator.generate_from_dare_roadmap(roadmap, kb, output_dir=str(output_dir))

    assert path.startswith(str(output_dir))
    assert "campaign.py" in path
