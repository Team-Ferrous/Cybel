from __future__ import annotations

from pathlib import Path

from saguaro.sentinel.engines.base import BaseEngine
from saguaro.sentinel.engines.aes import AESEngine
from saguaro.sentinel.engines.native import NativeEngine
from saguaro.sentinel.engines.semantic import SemanticEngine
from saguaro.sentinel.verifier import SentinelVerifier


class RaisingEngine(BaseEngine):
    def run(self, path_arg: str = ".") -> list[dict]:
        raise RuntimeError("boom")


class StaticEngine(BaseEngine):
    def __init__(self, repo_path: str, violations: list[dict]) -> None:
        super().__init__(repo_path)
        self._violations = violations

    def run(self, path_arg: str = ".") -> list[dict]:
        return list(self._violations)


class SnapshotStorage:
    def __init__(self, snapshot: dict) -> None:
        self._snapshot = snapshot

    def get_latest_snapshot(self) -> dict:
        return dict(self._snapshot)


def test_verifier_fail_closed_on_engine_failure(tmp_path: Path) -> None:
    verifier = SentinelVerifier(str(tmp_path), engines=[])
    verifier.engines = [RaisingEngine(str(tmp_path))]

    violations = verifier.verify_all(path_arg=str(tmp_path / "pkg"))

    assert len(violations) == 1
    assert violations[0]["rule_id"] == "SENTINEL-ENGINE-FAILURE"
    assert violations[0]["closure_level"] == "blocking"


def test_semantic_engine_parse_error_is_non_fatal_for_verifier(
    tmp_path: Path, monkeypatch
) -> None:
    snapshot = {"id": 7, "description": "baseline", "hd_state_blob": b"baseline"}
    monkeypatch.setattr(
        "saguaro.sentinel.engines.semantic.ChronicleStorage",
        lambda: SnapshotStorage(snapshot),
    )

    engine = SemanticEngine(str(tmp_path))
    monkeypatch.setattr(engine, "_calculate_current_state", lambda: b"current")

    def _raise_parse_error(*_args, **_kwargs):
        raise ValueError("TensorProto ParseFromString failed")

    monkeypatch.setattr(
        "saguaro.sentinel.engines.semantic.SemanticDiff.calculate_drift",
        _raise_parse_error,
    )

    verifier = SentinelVerifier(str(tmp_path), engines=[])
    verifier.engines = [engine]

    violations = verifier.verify_all(path_arg=str(tmp_path))

    assert violations == []


def test_semantic_engine_skips_legacy_parse_error_details(
    tmp_path: Path, monkeypatch
) -> None:
    snapshot = {"id": 11, "description": "baseline", "hd_state_blob": b"baseline"}
    monkeypatch.setattr(
        "saguaro.sentinel.engines.semantic.ChronicleStorage",
        lambda: SnapshotStorage(snapshot),
    )

    engine = SemanticEngine(str(tmp_path))
    monkeypatch.setattr(engine, "_calculate_current_state", lambda: b"current")
    monkeypatch.setattr(
        "saguaro.sentinel.engines.semantic.SemanticDiff.calculate_drift",
        lambda *_args, **_kwargs: (
            1.0,
            {"error": "Could not parse tensor proto"},
        ),
    )

    violations = engine.run()

    assert violations == []


def test_semantic_engine_emits_violation_for_real_drift(
    tmp_path: Path, monkeypatch
) -> None:
    snapshot = {"id": 13, "description": "baseline", "hd_state_blob": b"baseline"}
    monkeypatch.setattr(
        "saguaro.sentinel.engines.semantic.ChronicleStorage",
        lambda: SnapshotStorage(snapshot),
    )

    engine = SemanticEngine(str(tmp_path))
    engine.drift_threshold = 0.2
    monkeypatch.setattr(engine, "_calculate_current_state", lambda: b"current")
    monkeypatch.setattr(
        "saguaro.sentinel.engines.semantic.SemanticDiff.calculate_drift",
        lambda *_args, **_kwargs: (
            0.55,
            {"status": "ok", "similarity": 0.45},
        ),
    )

    violations = engine.run()

    assert len(violations) == 1
    assert violations[0]["rule_id"] == "SEMANTIC-DRIFT"


def test_verifier_scopes_results_to_requested_path_and_excludes_reference_roots(
    tmp_path: Path,
) -> None:
    target = tmp_path / "pkg" / "module.py"
    target.parent.mkdir(parents=True)
    target.write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "pkg" / "nested.py").write_text("print('nested')\n", encoding="utf-8")

    verifier = SentinelVerifier(str(tmp_path), engines=[])
    verifier.engines = [
        StaticEngine(
            str(tmp_path),
            [
                {
                    "rule_id": "AES-CR-2",
                    "file": "pkg/module.py",
                    "line": 1,
                    "severity": "P1",
                    "message": "Target issue",
                    "closure_level": "blocking",
                },
                {
                    "rule_id": "AES-CR-2",
                    "file": "pkg/nested.py",
                    "line": 1,
                    "severity": "P1",
                    "message": "Sibling issue",
                    "closure_level": "blocking",
                },
                {
                    "rule_id": "AES-CR-2",
                    "file": "Saguaro/generated.py",
                    "line": 1,
                    "severity": "P1",
                    "message": "Reference tree issue",
                    "closure_level": "blocking",
                },
                {
                    "rule_id": "AES-CR-2",
                    "file": ".anvil/generated.py",
                    "line": 1,
                    "severity": "P1",
                    "message": "Generated artifact issue",
                    "closure_level": "blocking",
                },
            ],
        )
    ]

    violations = verifier.verify_all(path_arg=str(target))

    assert len(violations) == 1
    assert violations[0]["file"] == "pkg/module.py"
    assert violations[0]["message"] == "Target issue"


def test_verifier_excludes_reference_roots_when_scanning_repo_root(tmp_path: Path) -> None:
    verifier = SentinelVerifier(str(tmp_path), engines=[])
    verifier.engines = [
        StaticEngine(
            str(tmp_path),
            [
                {
                    "rule_id": "AES-CR-2",
                    "file": "pkg/module.py",
                    "line": 1,
                    "severity": "P1",
                    "message": "Primary issue",
                    "closure_level": "blocking",
                },
                {
                    "rule_id": "AES-CR-2",
                    "file": "Saguaro/generated.py",
                    "line": 1,
                    "severity": "P1",
                    "message": "Reference tree issue",
                    "closure_level": "blocking",
                },
            ],
        )
    ]

    violations = verifier.verify_all(path_arg=str(tmp_path))

    assert len(violations) == 1
    assert violations[0]["file"] == "pkg/module.py"


def test_native_and_aes_do_not_duplicate_structured_aes_rules(tmp_path: Path) -> None:
    target = tmp_path / "saguaro" / "module.py"
    target.parent.mkdir(parents=True)
    target.write_text("try:\n    pass\nexcept:\n    pass\n", encoding="utf-8")
    (tmp_path / "standards").mkdir(parents=True)
    (tmp_path / "standards" / "AES_RULES.json").write_text(
        """
[
  {
    "id": "AES-CR-2",
    "section": "3.3",
    "text": "No bare except",
    "severity": "AAL-0",
    "engine": "agent",
    "auto_fixable": false,
    "domain": ["universal"],
    "language": ["python"],
    "check_function": "core.aes.checks.universal_checks.check_no_bare_except"
  }
]
        """.strip(),
        encoding="utf-8",
    )

    verifier = SentinelVerifier(str(tmp_path), engines=[])
    verifier.engines = [NativeEngine(str(tmp_path)), AESEngine(str(tmp_path))]

    violations = verifier.verify_all(path_arg=str(target))

    assert [item["rule_id"] for item in violations] == ["AES-CR-2"]


def test_ruff_engine_runs_ruff_backed_aes_rules(tmp_path: Path) -> None:
    target = tmp_path / "module.py"
    target.write_text(
        "from pathlib import Path\nimport os\n\n_ = (Path, os)\n",
        encoding="utf-8",
    )
    (tmp_path / "standards").mkdir(parents=True)
    (tmp_path / "standards" / "AES_RULES.json").write_text(
        """
[
  {
    "id": "AES-PY-2",
    "section": "7",
    "text": "Imports SHOULD be deterministic and ordered.",
    "severity": "AAL-2",
    "engine": "ruff",
    "auto_fixable": false,
    "domain": ["universal"],
    "language": ["python"],
    "check_function": "core.aes.checks.ruff_checks.check_ruff_import_order"
  }
]
        """.strip(),
        encoding="utf-8",
    )

    verifier = SentinelVerifier(str(tmp_path), engines=["ruff"])

    violations = verifier.verify_all(path_arg=str(target))

    assert [item["rule_id"] for item in violations] == ["AES-PY-2"]


def test_ruff_and_aes_do_not_duplicate_ruff_backed_aes_rules(tmp_path: Path) -> None:
    target = tmp_path / "module.py"
    target.write_text(
        "from pathlib import Path\nimport os\n\n_ = (Path, os)\n",
        encoding="utf-8",
    )
    (tmp_path / "standards").mkdir(parents=True)
    (tmp_path / "standards" / "AES_RULES.json").write_text(
        """
[
  {
    "id": "AES-PY-2",
    "section": "7",
    "text": "Imports SHOULD be deterministic and ordered.",
    "severity": "AAL-2",
    "engine": "ruff",
    "auto_fixable": false,
    "domain": ["universal"],
    "language": ["python"],
    "check_function": "core.aes.checks.ruff_checks.check_ruff_import_order"
  }
]
        """.strip(),
        encoding="utf-8",
    )

    verifier = SentinelVerifier(str(tmp_path), engines=["ruff", "aes"])

    violations = verifier.verify_all(path_arg=str(target))

    assert [item["rule_id"] for item in violations] == ["AES-PY-2"]


def test_ruff_engine_excludes_reference_root_when_scanning_repo_root(
    tmp_path: Path,
) -> None:
    good_target = tmp_path / "saguaro" / "module.py"
    good_target.parent.mkdir(parents=True)
    good_target.write_text("import os\n\nprint(os)\n", encoding="utf-8")

    reference_target = tmp_path / "Saguaro" / "generated.py"
    reference_target.parent.mkdir(parents=True)
    reference_target.write_text(
        "from pathlib import Path\nimport os\n\nprint(Path, os)\n",
        encoding="utf-8",
    )

    verifier = SentinelVerifier(str(tmp_path), engines=["ruff"])

    violations = verifier.verify_all(path_arg=str(tmp_path))

    assert violations == []
