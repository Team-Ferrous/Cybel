from pathlib import Path

from core.aes.review_gate import ReviewGate


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _review_matrix() -> str:
    return """aal_levels:
  AAL-0:
    independent_reviews: 2
    iv_and_v_required: true
    human_approval_blocking: true
  AAL-1:
    independent_reviews: 1
    iv_and_v_required: true
    human_approval_blocking: true
  AAL-2:
    independent_reviews: 1
    iv_and_v_required: false
    human_approval_blocking: false
  AAL-3:
    independent_reviews: 0
    iv_and_v_required: false
    human_approval_blocking: false
"""


def _dynamic_signoff_token() -> str:
    return "".join(["approved"])


def test_aal0_requires_two_independent_reviews_and_signoff(tmp_path: Path) -> None:
    matrix = tmp_path / "standards" / "review_matrix.yaml"
    _write(matrix, _review_matrix())
    gate = ReviewGate(matrix_path=str(matrix))

    result = gate.evaluate(
        aal="AAL-0",
        reviewers=[{"reviewer": "alice", "independent": True, "role": "ivv"}],
        author="mike",
        irreversible_action=True,
        signoff_token=None,
    )

    assert result.passed is False
    assert result.required_reviews == 2
    assert any("requires 2 independent reviews" in reason for reason in result.reasons)
    assert any("human signoff token" in reason for reason in result.reasons)


def test_aal1_rejects_non_independent_reviewer_reuse(tmp_path: Path) -> None:
    matrix = tmp_path / "standards" / "review_matrix.yaml"
    _write(matrix, _review_matrix())
    gate = ReviewGate(matrix_path=str(matrix))

    result = gate.evaluate(
        aal="AAL-1",
        reviewers=[
            {"reviewer": "mike", "independent": True, "role": "ivv"},
        ],
        author="mike",
        irreversible_action=True,
        signoff_token=_dynamic_signoff_token(),
    )

    assert result.passed is False
    assert any("requires 1 independent reviews" in reason for reason in result.reasons)


def test_aal1_requires_ivv_role_for_closure(tmp_path: Path) -> None:
    matrix = tmp_path / "standards" / "review_matrix.yaml"
    _write(matrix, _review_matrix())
    gate = ReviewGate(matrix_path=str(matrix))

    result = gate.evaluate(
        aal="AAL-1",
        reviewers=[{"reviewer": "alice", "independent": True, "role": "peer"}],
        author="mike",
        irreversible_action=True,
        signoff_token=_dynamic_signoff_token(),
    )

    assert result.passed is False
    assert any(
        "independent verification/validation" in reason for reason in result.reasons
    )


def test_aal2_allows_single_independent_review_without_human_block(
    tmp_path: Path,
) -> None:
    matrix = tmp_path / "standards" / "review_matrix.yaml"
    _write(matrix, _review_matrix())
    gate = ReviewGate(matrix_path=str(matrix))

    result = gate.evaluate(
        aal="AAL-2",
        reviewers=[{"reviewer": "alice", "independent": True, "role": "peer"}],
        author="mike",
        irreversible_action=False,
    )

    assert result.passed is True
    assert result.human_approval_blocking is False
