from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from core.aes.rule_registry import AESRuleRegistry

try:
    from hypothesis import given, strategies as st
except ModuleNotFoundError:
    given = None
    st = None


def _rule_strategy() -> st.SearchStrategy[dict]:
    return st.fixed_dictionaries(
        {
            "id": st.text(alphabet=st.characters(min_codepoint=48, max_codepoint=122), min_size=3, max_size=16),
            "section": st.text(min_size=1, max_size=8),
            "text": st.text(min_size=1, max_size=120),
            "severity": st.sampled_from(["AAL-0", "AAL-1", "AAL-2", "AAL-3"]),
            "engine": st.sampled_from(["agent", "native", "semantic", "ruff", "human"]),
            "auto_fixable": st.booleans(),
            "domain": st.lists(st.sampled_from(["universal", "ml", "quantum", "physics", "hpc"]), min_size=1, max_size=2, unique=True),
            "language": st.lists(st.sampled_from(["python", "json", "yaml", "c++", "md", "toml", "txt"]), min_size=1, max_size=3, unique=True),
        }
    )


if given is not None and st is not None:

    @given(st.lists(_rule_strategy(), min_size=1, max_size=8))
    def test_rule_registry_load_round_trip(rules: list[dict]) -> None:
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "AES_RULES.json"
            path.write_text(json.dumps(rules), encoding="utf-8")

            registry = AESRuleRegistry()
            registry.load(str(path))

            assert len(registry.rules) == len(rules)
            first_id = rules[0]["id"]
            assert registry.get_rule(first_id) is not None

else:

    def test_rule_registry_load_round_trip(tmp_path: Path) -> None:
        rules = [
            {
                "id": "AES-TEST-1",
                "section": "t",
                "text": "test",
                "severity": "AAL-2",
                "engine": "agent",
                "auto_fixable": False,
                "domain": ["universal"],
                "language": ["python"],
            }
        ]
        path = tmp_path / "AES_RULES.json"
        path.write_text(json.dumps(rules), encoding="utf-8")
        registry = AESRuleRegistry()
        registry.load(str(path))
        assert registry.get_rule("AES-TEST-1") is not None
