from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


ALLOWED_EXECUTION_MODES = {"static", "artifact", "runtime_gate", "workflow_gate", "manual"}


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a top-level mapping")
    return payload


def _derive_title(rule_id: str, text: str) -> str:
    stem = (text or rule_id).strip().split(".")[0].strip()
    if stem:
        return stem[:96]
    return rule_id


def _prefix_matches(rule_id: str, prefixes: list[str]) -> bool:
    return any(rule_id.startswith(prefix) for prefix in prefixes)


def _status_from_defaults(severity: str, defaults: dict[str, Any]) -> str:
    mapping = defaults.get("status_by_severity", {}) or {}
    return str(mapping.get(str(severity), "advisory"))


def _execution_mode_from_defaults(engine: str, defaults: dict[str, Any]) -> str:
    mapping = defaults.get("engine_execution_modes", {}) or {}
    return str(mapping.get(str(engine), "static"))


def _merge_dicts(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        if key in {"selectors", "parameters"} and isinstance(value, dict):
            prior = merged.get(key) if isinstance(merged.get(key), dict) else {}
            merged[key] = {**prior, **value}
        elif key in {"required_artifacts", "source_refs"} and isinstance(value, list):
            prior = merged.get(key) if isinstance(merged.get(key), list) else []
            merged[key] = list(dict.fromkeys([*prior, *value]))
        else:
            merged[key] = value
    return merged


def _build_rule(
    seed_rule: dict[str, Any],
    catalog: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    defaults = catalog.get("defaults", {}) or {}
    overrides = catalog.get("overrides", {}) or {}
    groups = catalog.get("groups", []) or []
    rule_id = str(seed_rule["id"])

    enriched: dict[str, Any] = {
        "id": rule_id,
        "section": str(seed_rule.get("section", "")),
        "text": str(seed_rule.get("text", "")).strip(),
        "severity": str(seed_rule.get("severity", "AAL-2")),
        "engine": str(seed_rule.get("engine", "agent")),
        "auto_fixable": bool(seed_rule.get("auto_fixable", False)),
        "domain": list(seed_rule.get("domain", []) or []),
        "language": list(seed_rule.get("language", []) or []),
        "check_function": seed_rule.get("check_function"),
        "pattern": seed_rule.get("pattern"),
        "negative_lookahead": seed_rule.get("negative_lookahead"),
        "cwe": list(seed_rule.get("cwe", []) or []),
        "svl_min": seed_rule.get("svl_min"),
        "title": _derive_title(rule_id, str(seed_rule.get("text", ""))),
        "source_version": defaults.get("source_version", "v2"),
        "source_refs": [],
        "precedence": int(defaults.get("precedence", 100)),
        "selectors": dict(defaults.get("selectors", {}) or {}),
        "execution_mode": _execution_mode_from_defaults(seed_rule.get("engine", "agent"), defaults),
        "parameters": dict(defaults.get("parameters", {}) or {}),
        "required_artifacts": list(defaults.get("required_artifacts", []) or []),
        "waiverable": bool(defaults.get("waiverable", False)),
        "rollout_stage": str(defaults.get("rollout_stage", "ratchet")),
        "status": _status_from_defaults(str(seed_rule.get("severity", "AAL-2")), defaults),
    }

    for group in groups:
        if not isinstance(group, dict):
            continue
        match_ids = [str(item) for item in group.get("match_ids", []) or []]
        match_prefixes = [str(item) for item in group.get("match_prefixes", []) or []]
        if rule_id in match_ids or _prefix_matches(rule_id, match_prefixes):
            enriched = _merge_dicts(enriched, {k: v for k, v in group.items() if k not in {"name", "match_ids", "match_prefixes"}})

    if rule_id in overrides:
        enriched = _merge_dicts(enriched, overrides[rule_id])

    enriched["domains"] = list(enriched.get("domains", enriched.get("domain", []) or []))
    enriched["languages"] = list(
        enriched.get("languages", enriched.get("language", []) or [])
    )
    enriched["domain"] = list(enriched["domains"])
    enriched["language"] = list(enriched["languages"])
    enriched["title"] = str(enriched.get("title") or _derive_title(rule_id, enriched["text"]))
    enriched["execution_mode"] = str(enriched.get("execution_mode") or "static")
    enriched["source_refs"] = list(enriched.get("source_refs", []) or [])
    enriched["required_artifacts"] = list(enriched.get("required_artifacts", []) or [])
    enriched["selectors"] = dict(enriched.get("selectors", {}) or {})
    enriched["parameters"] = dict(enriched.get("parameters", {}) or {})
    enriched["parameters"]["thresholds"] = thresholds.get("thresholds", {})
    return enriched


def _validate_rule(rule: dict[str, Any], seen: set[str]) -> None:
    rule_id = str(rule["id"])
    if rule_id in seen:
        raise ValueError(f"duplicate rule id: {rule_id}")
    seen.add(rule_id)

    if not rule.get("source_refs"):
        raise ValueError(f"rule {rule_id} missing source_refs")
    if int(rule.get("precedence", 0)) < 0:
        raise ValueError(f"rule {rule_id} has invalid precedence")
    if str(rule.get("execution_mode")) not in ALLOWED_EXECUTION_MODES:
        raise ValueError(f"rule {rule_id} has invalid execution_mode")


def compile_catalog(repo_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    catalog_path = repo_root / "standards" / "aes" / "rule_catalog.yaml"
    thresholds_path = repo_root / "standards" / "aes" / "thresholds.yaml"
    obligations_path = repo_root / "standards" / "aes" / "obligation_matrix.yaml"

    catalog = _load_yaml(catalog_path)
    thresholds = _load_yaml(thresholds_path)
    obligations = _load_yaml(obligations_path)

    seed_path = repo_root / str(catalog.get("legacy_seed_path"))
    seed_rules = json.loads(seed_path.read_text(encoding="utf-8"))
    if not isinstance(seed_rules, list):
        raise ValueError("legacy seed must be a list of rules")

    compiled: list[dict[str, Any]] = []
    seen: set[str] = set()
    for seed_rule in seed_rules:
        if not isinstance(seed_rule, dict):
            raise ValueError(f"invalid seed rule payload: {seed_rule!r}")
        compiled_rule = _build_rule(seed_rule, catalog, thresholds)
        _validate_rule(compiled_rule, seen)
        compiled.append(compiled_rule)

    compiled.sort(key=lambda item: (item["section"], item["id"]))
    compiled_obligations = {
        "catalog_version": obligations.get("catalog_version", 1),
        "authoritative_package_root": catalog.get("authoritative_package_root", "saguaro"),
        "excluded_reference_roots": list(catalog.get("excluded_reference_roots", []) or []),
        "defaults": obligations.get("defaults", {}) or {},
        "thresholds": thresholds.get("thresholds", {}) or {},
        "obligations": obligations.get("obligations", []) or [],
    }
    return compiled, compiled_obligations


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile AES catalog into runtime JSON artifacts.")
    parser.add_argument("--repo", default=".", help="Repository root")
    args = parser.parse_args()

    repo_root = Path(args.repo).resolve()
    compiled_rules, compiled_obligations = compile_catalog(repo_root)

    rules_path = repo_root / "standards" / "AES_RULES.json"
    obligations_path = repo_root / "standards" / "AES_OBLIGATIONS.json"
    rules_path.write_text(json.dumps(compiled_rules, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    obligations_path.write_text(
        json.dumps(compiled_obligations, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
