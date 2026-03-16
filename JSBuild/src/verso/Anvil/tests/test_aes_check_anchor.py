from __future__ import annotations

import json
from pathlib import Path

from core.aes.checks.registry_anchor import REGISTERED_AES_CHECKS


def test_registry_anchor_covers_catalog_declared_check_functions() -> None:
    rules_path = Path("standards/AES_RULES.json")
    payload = json.loads(rules_path.read_text(encoding="utf-8"))
    declared = {
        str(rule.get("check_function"))
        for rule in payload
        if rule.get("check_function")
    }
    missing = sorted(
        check_function
        for check_function in declared
        if check_function.startswith("core.aes.")
        and check_function not in REGISTERED_AES_CHECKS
    )
    assert missing == []
