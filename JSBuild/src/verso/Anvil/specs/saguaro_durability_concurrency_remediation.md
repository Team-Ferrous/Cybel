# Saguaro Durability and Concurrency Remediation

## Scope
- Replace non-atomic committed index writes with atomic persistence helpers.
- Introduce repo-scoped many-reader/one-writer locks for index and state operations.
- Add committed index manifest validation and staged promotion for indexing.
- Remove silent tracker/vector-store recovery that previously reset live state.
- Add explicit recovery tooling and integrity-aware health/reporting.

## Primary-Owned Files
- `saguaro/api.py`
- `saguaro/health.py`
- `saguaro/indexing/tracker.py`
- `saguaro/indexing/stats.py`
- `saguaro/state/ledger.py`
- `saguaro/storage/memmap_vector_store.py`
- `saguaro/storage/vector_store.py`
- `saguaro/storage/atomic_fs.py`
- `saguaro/storage/locks.py`
- `saguaro/storage/index_state.py`
- `saguaro/errors.py`
- `saguaro/cli.py`
- `tests/test_saguaro_durability.py`
- `tests/test_saguaro_query_accuracy.py`

## Shared Files
- `saguaro/services/platform.py`
- `tests/test_saguaro_platform_foundation.py`

## Read-Only Dependencies
- `saguaro/query/*`
- `saguaro/analysis/*`
- `saguaro/services/*`
- `tests/fixtures/*`

## Prerequisite Assumptions
- POSIX `fcntl.flock` is available in the primary runtime.
- Existing graph atomic-write behavior remains the reference model.
- Query-time auto-refresh is allowed only when explicitly requested.

## Implementation Steps
1. Add atomic filesystem helpers for JSON, YAML, text, and JSONL writes.
2. Add repo-scoped advisory locks with metadata sidecars.
3. Add committed index manifest helpers and validation.
4. Convert tracker, stats, config, and ledger state writes to atomic persistence.
5. Harden memmap and legacy vector stores against silent reset-on-corruption.
6. Change index rebuild to stage artifacts and promote them atomically.
7. Make health, doctor, query, impact, deadcode, unwired, and graph operations lock-aware.
8. Add `recover` API/CLI flow to quarantine corrupt live artifacts and promote intact staged generations.
9. Add regression tests for corruption handling, durability, and explicit query refresh behavior.

## Verification
- `./venv/bin/pytest tests/test_saguaro_durability.py tests/test_saguaro_query_accuracy.py tests/test_saguaro_platform_foundation.py -q`
- `./venv/bin/python -m py_compile saguaro/errors.py saguaro/storage/atomic_fs.py saguaro/storage/locks.py saguaro/storage/index_state.py saguaro/storage/memmap_vector_store.py saguaro/storage/vector_store.py saguaro/indexing/tracker.py saguaro/indexing/stats.py saguaro/state/ledger.py saguaro/health.py saguaro/services/platform.py saguaro/api.py saguaro/cli.py`
- `source venv/bin/activate && saguaro verify . --engines native,ruff,semantic --format json`

## Artifact Namespace
- `specs/saguaro_durability_concurrency_remediation.md`

## Exit Criteria
- No committed index/tracker/config write path uses naked overwrite semantics.
- Semantic query fails closed when manifest integrity is not `ready`.
- Index rebuild promotes staged artifacts instead of deleting live state up front.
- Recovery can quarantine broken live artifacts and restore an intact staged generation.
- Targeted regression suite passes.
