# Phase 0 AES Foundation Spec

## Scope

Phase 0 implements the AES runtime control plane for Foundation & Standards Infrastructure.
This phase is limited to:
- canonical AES standards artifacts under `standards/`
- deterministic AAL and domain classification under `core/aes/`
- runtime-loadable AES rule registry
- prompt assembly support for AES condensed and domain-specific guidance
- rollout flags for staged enforcement
- hook lifecycle expansion and receipt capture
- legacy master-agent routing through the unified path

This phase does not implement Sentinel `aes` engine execution or full policy hard-blocking. Those remain Phase 1 work.

## Traceability

- `AES-PH0-ARCH-001`: rollout flags must exist in runtime config
- `AES-PH0-ARCH-002`: legacy master flow must route to unified orchestration path
- `AES-PH0-ARCH-003`: hook registry must support post-write and finalization lifecycle points
- `AES-PH0-DOC-001`: canonical AES documents and domain rules must live under `standards/`
- `AES-PH0-RULE-001`: AES rules must load from `standards/AES_RULES.json`
- `AES-PH0-CLS-001`: AAL classification must be deterministic for identical inputs
- `AES-PH0-CLS-002`: domain detection must be deterministic for identical inputs
- `AES-PH0-PRM-001`: prompt assembly must always inject `AES_CONDENSED.md`
- `AES-PH0-PRM-002`: prompt assembly must inject domain rules when markers are detected
- `AES-PH0-EVD-001`: traceability, evidence bundle, and waiver schemas must be machine-readable

## Design Decisions

1. The rule registry is JSON-backed and imports check callables lazily via dotted paths.
2. AAL severity ordering is strictest-wins: `AAL-0` > `AAL-1` > `AAL-2` > `AAL-3`.
3. Prompt assembly is additive and backward compatible: existing prompt templates remain intact and AES context is appended.
4. Hook receipts are stored in-memory in execution context so later phases can persist them without another interface break.
5. `agents/master.py` becomes a compatibility adapter that delegates mission execution to `UnifiedMasterAgent` when AES enforcement is enabled.
6. Runtime compliance IDs (`trace_id`, `evidence_bundle_id`, `waiver_id`) must be non-null in all primary loop entry points and deterministic fallbacks.
7. Helper editors/refactorers that can write files must route through a delegated tool executor so hook enforcement remains centralized.

## Verification Plan

- unit tests for classifier determinism and strictest-wins changeset behavior
- unit tests for domain detector marker loading
- unit tests for registry load, lookup, and callable resolution
- unit tests for hook receipt generation and lifecycle registration
- unit tests for prompt builder AES/domain injection
- unit tests for runtime compliance propagation and helper write-path routing
- schema validation tests for all required Phase 0 schemas and artifacts

## Exit Criteria

Phase 0 is complete when:
- all new artifacts load without adapters
- all Phase 0 write-capable helper paths route through the unified tool executor contract
- tests covering classifier, detector, registry, prompt assembly, and schemas pass
- roadmap Phase 0 addendum references evidence from the test run
