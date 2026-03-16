from core.agents.subagent_quality_gate import SubagentQualityGate


class StubBrain:
    def __init__(self, mapping):
        self.mapping = mapping

    def embeddings(self, text):
        return self.mapping.get(text, [0.0, 0.0, 1.0])


def test_quality_gate_accepts_valid_citations(tmp_path):
    target = tmp_path / "sample.py"
    target.write_text("def foo():\n    return 1\n", encoding="utf-8")

    gate = SubagentQualityGate(
        repo_root=str(tmp_path),
        brain=StubBrain(
            {
                "query": [1.0, 0.0, 0.0],
                "Analysis in sample.py:L1\n```python\ndef foo():\n    return 1\n```": [0.9, 0.0, 0.0],
            }
        ),
    )
    payload = {
        "subagent_analysis": "Analysis in sample.py:L1\n```python\ndef foo():\n    return 1\n```",
        "codebase_files": ["sample.py"],
        "compliance": {
            "trace_id": "trace-1",
            "evidence_bundle_id": "bundle-1",
        },
    }
    result = gate.evaluate(payload, "query", complexity_score=3)
    assert result["accepted"] is True


def test_quality_gate_rejects_missing_file_path(tmp_path):
    gate = SubagentQualityGate(repo_root=str(tmp_path), brain=StubBrain({}))
    payload = {"subagent_analysis": "Claim from missing.py:L12", "codebase_files": []}
    result = gate.evaluate(payload, "query", complexity_score=5)
    assert result["accepted"] is False
    assert result["hallucination"]["missing_paths"]


def test_quality_gate_low_alignment_requests_retry(tmp_path):
    target = tmp_path / "sample.py"
    target.write_text("def bar():\n    return 2\n", encoding="utf-8")

    gate = SubagentQualityGate(
        repo_root=str(tmp_path),
        brain=StubBrain(
            {
                "query": [1.0, 0.0],
                "sample.py:L1 sample.py:L2": [0.0, 1.0],
            }
        ),
    )
    payload = {
        "subagent_analysis": "sample.py:L1 sample.py:L2",
        "codebase_files": ["sample.py"],
        "compliance": {
            "trace_id": "trace-2",
            "evidence_bundle_id": "bundle-2",
        },
    }
    result = gate.evaluate(payload, "query", complexity_score=6)
    assert result["should_retry"] is True
    assert result["alignment"]["score"] < 0.3
