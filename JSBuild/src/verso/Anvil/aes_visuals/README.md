# AES Visual Directives

This folder contains Anvil-owned, model-readable guidance derived from public standards referenced by AES documentation.

## Layout
- `v1/`: Baseline policy profile aligned to AES pinned external baselines.
- `v2/`: Expanded policy profile with stricter safety/lifecycle and evidence requirements.

## Shared JSON Schema
Both versions use the same `directives.json` shape:
- `schema_version`
- `artifact`
- `profile`
- `generated_on`
- `owner`
- `upstream_context`
- `directives[]` with:
  - `directive_id`
  - `title`
  - `rationale`
  - `enforcement_targets`
  - `implementation_patterns`
  - `verification_checks`
  - `source_refs`

Date baseline for this artifact set: `2026-03-03`.
