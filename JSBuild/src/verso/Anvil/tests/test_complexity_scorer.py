from core.reasoning.complexity_scorer import ComplexityScorer


def test_simple_question_scores_low():
    scorer = ComplexityScorer()
    profile = scorer.score_request("What does ExitCommand do?")
    assert 1 <= profile.score <= 3
    assert profile.coconut_frequency == "none"
    assert profile.subagent_coconut is False


def test_architecture_question_scores_medium_high():
    scorer = ComplexityScorer()
    profile = scorer.score_request(
        "How does the architecture coordinate subagents and message bus flow?"
    )
    assert profile.score >= 6
    assert profile.coconut_frequency in {"per_phase", "per_step"}


def test_multifile_refactor_scores_high():
    scorer = ComplexityScorer()
    profile = scorer.score_request(
        "Refactor the entire context compression system across core/context.py, core/unified_chat_loop.py, and core/agents/subagent.py end-to-end.",
    )
    assert profile.score >= 8
    assert profile.subagent_coconut is True
    assert profile.coconut_paths >= 6
