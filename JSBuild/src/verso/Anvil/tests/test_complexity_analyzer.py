from core.reasoning.complexity_analyzer import TaskComplexityAnalyzer


def test_complexity_analyzer_low_complexity():
    analyzer = TaskComplexityAnalyzer()
    profile = analyzer.analyze("What is this module?")
    assert 0.0 <= profile.complexity_score <= 1.0
    assert profile.subagent_count >= 1
    assert profile.max_steps_per_agent >= 4


def test_complexity_analyzer_high_complexity():
    analyzer = TaskComplexityAnalyzer()
    profile = analyzer.analyze(
        "Implement end-to-end multi-agent roadmap: refactor core/agents/subagent.py and "
        "core/unified_chat_loop.py, integrate orchestration pipeline, and migrate architecture.",
        candidate_files=[
            "core/agents/subagent.py",
            "core/unified_chat_loop.py",
            "core/reasoning/coconut.py",
        ],
        previous_entropy=0.9,
    )
    assert profile.complexity_score > 0.6
    assert profile.subagent_coconut is True
    assert profile.coconut_depth >= 4
    assert profile.reasoning_depth >= 4
