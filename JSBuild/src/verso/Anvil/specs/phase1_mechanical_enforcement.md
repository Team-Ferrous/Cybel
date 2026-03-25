# Phase 1 Mechanical Enforcement Spec

## Scope

Phase 1 implements hard mechanical enforcement for AES policy in Sentinel, CLI/API verification surfaces, and agent write paths.
This phase is limited to:
- Sentinel `aes` engine integration and deterministic execution ordering
- violation schema extension with closure metadata
- dual-source rule loading (`standards/AES_RULES.json` then `.saguaro.rules`)
- AES severity normalization and artifact-blocking policy for AAL-0/AAL-1
- `saguaro verify` AAL/domain/artifact-required flags
- post-write AES verification hook wiring and write-path bypass protection
- tool-output provenance tagging and credential redaction
- strict-mode preflight script and enforcement-aligned lint/rule configuration

This phase does not include broader agent governance rewrites (Phase 2+) or domain template generation (Phase 4).

## Traceability

- `AES-PH1-SEN-001`: Sentinel must register first-class `aes` engine.
- `AES-PH1-SEN-002`: violation records must include `aal`, `domain`, `rule_id`, `closure_level`, `evidence_refs`.
- `AES-PH1-SEN-003`: Sentinel must load rules from `standards/AES_RULES.json` before `.saguaro.rules`, preserving legacy compatibility.
- `AES-PH1-SEN-004`: policy must normalize severities into `P0..P3` closure behavior with AAL-aware blocking.
- `AES-PH1-SEN-005`: AAL-0/AAL-1 paths must block when required artifacts are missing when configured.
- `AES-PH1-CLI-001`: `saguaro verify` must expose `--aal`, `--domain`, `--require-trace`, `--require-evidence`, `--require-valid-waivers`.
- `AES-PH1-CLI-002`: engine ordering must be deterministic: `native -> ruff -> semantic -> aes`.
- `AES-PH1-HOOK-001`: all write-capable tool paths must execute post-write verification hooks.
- `AES-PH1-HOOK-002`: hook receipts must be traceable/auditable for write operations.
- `AES-PH1-SEC-001`: persisted tool outputs must be redacted for credentials and tagged with provenance (`tool_name`, `args_hash`, `timestamp`, `trace_id`).
- `AES-PH1-CFG-001`: `.saguaro.rules` and `ruff.toml` must align to Phase 1 enforcement contract.
- `AES-PH1-OPS-001`: strict-mode preflight must fail fast when Sentinel/AES prerequisites are not available.

## Design Decisions

1. `AESEngine` consumes `AESRuleRegistry`, `AALClassifier`, and `DomainDetector` so check execution is deterministic and centrally controlled.
2. `SentinelVerifier` owns execution ordering and shared verify context (`aal`, `domain`, required artifacts), with each engine receiving the same policy envelope.
3. Policy normalization maps mixed severity vocabularies (`error/warning`, `AAL-*`, `P*`) to a single closure behavior model.
4. Post-write verification is enforced in the single tool hot path (`BaseAgent._execute_tool`) to prevent helper-level bypasses.
5. Provenance metadata is attached before history persistence so downstream audit logs remain machine-actionable.
6. Artifact checks are surfaced as explicit violations (with rule IDs) rather than implicit warnings to support gate automation.

## Verification Plan

- unit tests for Sentinel engine registration and deterministic ordering
- unit tests for rule loader precedence and migration warning behavior
- unit tests for policy severity normalization and AAL-aware blocking behavior
- unit tests for `saguaro verify` CLI/API flags and scoping behavior
- unit tests for post-write hook execution on write tools and bypass rejection behavior
- unit tests for tool-output redaction/provenance metadata
- targeted verification run:
  - `pytest` for new/changed AES Sentinel and hook tests
  - `saguaro verify . --engines native,ruff,semantic,aes --format json` (or constrained equivalent for environment)

## Exit Criteria

Phase 1 is complete when:
- `aes` engine runs as first-class Sentinel engine with deterministic ordering
- AAL-0/AAL-1 blocking behavior is enforced for configured policy/artifact requirements
- write-capable tool paths cannot skip post-write verification hooks
- `saguaro verify` exposes and honors Phase 1 flag contract
- roadmap Phase 1 addendum and repair-map entries include test and evidence references
