from core.dare.knowledge_base import DareKnowledgeBase


def test_store_and_query_entry(tmp_path):
    kb = DareKnowledgeBase(root_dir=str(tmp_path), campaign_id="campaign-1")
    path = kb.store(
        category="analysis",
        topic="attention patterns",
        content="# Findings\nFused QKV helps CPU inference.",
        source="unit-test",
        confidence="high",
        tags=["attention", "cpu"],
    )

    assert path.endswith("analysis/attention_patterns.md")
    results = kb.query("Fused QKV", category="analysis")
    assert len(results) == 1
    assert results[0].topic == "attention patterns"


def test_merge_findings_persists_synthesis(tmp_path):
    kb = DareKnowledgeBase(root_dir=str(tmp_path), campaign_id="campaign-1")
    first = kb.store(
        category="analysis",
        topic="repo one",
        content="Repo one uses AVX2 in attention paths.",
        source="unit-test",
        confidence="medium",
        tags=["repo"],
    )
    second = kb.store(
        category="research",
        topic="forum evidence",
        content="Forum users ask for clearer benchmarks.",
        source="unit-test",
        confidence="medium",
        tags=["forum"],
    )

    merged = kb.merge_findings([first, second], output_topic="combined evidence")
    assert "Repo one uses AVX2" in merged
    assert "Forum users ask for clearer benchmarks." in merged
    report = kb.get_full_report(category="synthesis")
    assert "combined evidence" in report
