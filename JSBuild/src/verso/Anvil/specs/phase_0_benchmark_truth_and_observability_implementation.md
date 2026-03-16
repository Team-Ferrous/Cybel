## Phase 0: Benchmark Truth and Observability

- Phase id: `0`
- Roadmap section: `## 6. Phase 0: Benchmark Truth and Observability`
- Acceptance references:
  - `## 6.10 Acceptance criteria for Phase 0`
  - `## 22. Phase 16: Concrete Acceptance Criteria`
  - `## 29. Build and Tooling Recommendations`

### Scope classification

- `IN_SCOPE_IMPLEMENTABLE`
  - Extend benchmark provenance so run artifacts capture required benchmark metadata at the run level.
  - Persist benchmark metadata into preflight and environment artifacts for reproducible audit review.
  - Add tests proving the metadata contract is emitted.
- `IN_SCOPE_BUT_BLOCKED_BY_PREREQUISITE`
  - Native hot-path thread-local telemetry redesign from `6.8` is broader than this bounded audit pass.
  - Stage-accurate scheduler-overhead isolation from graph compute needs deeper benchmark/runtime coupling than the current artifact gap fix.
- `OUT_OF_SCOPE_FUTURE_PHASE`
  - Native runtime ownership, fused sampler, paged KV, speculative execution, and non-AR runtime work.
- `OUT_OF_SCOPE_UNRELATED`
  - Existing hot-path refactors already in progress under `core/native/*`.

### In-scope files

- `audit/provenance/capture.py`
- `audit/runner/suite_preflight.py`
- `audit/runner/benchmark_suite.py`
- `tests/audit/test_benchmark_suite.py`

### Prerequisite assumptions

- The existing benchmark suite remains the source of truth for run artifact layout.
- `perf` probing and Saguaro health checks already exist and should not be redesigned in this phase.
- Model provenance already exposes per-model quantization metadata via `quant_variant`.

### Implementation steps

1. Extend runtime provenance capture with benchmark-relevant host metadata:
   - kernel release
   - microcode when detectable
   - governor mode
   - huge page mode
   - NUMA policy when detectable
   - memory speed when detectable
   - normalized OpenMP environment fields
2. Build a run-level benchmark metadata bundle during preflight from:
   - git SHA
   - model list and quantization profile
   - thread and affinity environment
   - host platform metadata
   - huge page / NUMA / governor state
3. Persist that bundle in both `preflight.json` and `environment.json`.
4. Add targeted tests that assert the new benchmark metadata contract.

### Verification steps

1. `pytest tests/audit/test_benchmark_suite.py`
2. `source venv/bin/activate && saguaro verify . --engines native,ruff,semantic --format json`

### Exit criteria

- Run artifacts expose the missing experiment metadata required by roadmap section `6.6`.
- Metadata is emitted from a single provenance/preflight path rather than duplicated across benchmark lanes.
- Targeted tests pass.
- Repo verification is executed and any unrelated failures are called out explicitly.
