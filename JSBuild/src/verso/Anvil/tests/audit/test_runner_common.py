from __future__ import annotations

from pathlib import Path

import pytest

from audit.runner.common import load_optional_json_dict, read_json_dict


def test_load_optional_json_dict_returns_none_for_missing_or_invalid_inputs(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{not-json", encoding="utf-8")

    assert load_optional_json_dict(None) is None
    assert load_optional_json_dict(missing) is None
    assert load_optional_json_dict(invalid) is None


def test_read_json_dict_requires_object_payload(tmp_path: Path) -> None:
    target = tmp_path / "payload.json"
    target.write_text('{"status": "ok"}', encoding="utf-8")

    assert read_json_dict(target) == {"status": "ok"}

    target.write_text('["wrong-shape"]', encoding="utf-8")
    with pytest.raises(TypeError):
        read_json_dict(target)
