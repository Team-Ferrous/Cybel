# Saguaro Migration Manifest

Canonical runtime root: `Saguaro/saguaro`

Behavioral donor root: `saguaro_restored+temp`

Policy:

- `restored`: restore from `saguaro_restored+temp` and make it the baseline.
- `current`: keep current `Saguaro/saguaro` version.
- `current-later`: preserve current file, but keep it off the default hot path until gated validation passes.

## Current Checkpoint

- Completed:
  - phase 1 migration manifest freeze
  - phase 2 core restore for API, CLI, parser, indexing, and storage surfaces
  - remaining operational bucket reconciliation
  - remaining analysis/perception/platform bucket reconciliation
  - remaining validation/governance bucket reconciliation
  - remaining storage/query/math/ops compatibility bucket reconciliation
  - authoritative `Saguaro/saguaro` path normalization
  - removal of the dead legacy CLI `index` branch
  - structured `index_journal.jsonl` surface and `debuginfo` journal exposure
  - graph artifact normalization before manifest snapshot to keep `graph/graph.json` manifest-stable
- Pending before the next full-index revalidation:
  - rerun full CLI `index`, `query`, `health`, and `verify` against the migrated tree

## Intentional Canonical Deltas

These files remain intentionally different from `saguaro_restored+temp` after migration closure because they carry required compatibility, logging, or integrity fixes in the canonical runtime:

- `api.py`
- `build_system/ingestor.py`
- `cli.py`
- `indexing/auto_scaler.py`
- `indexing/engine.py`
- `indexing/native_worker.py`
- `indexing/tracker.py`
- `parsing/parser.py`
- `query/corpus_rules.py`
- `services/platform.py`
- `storage/native_vector_store.py`
- `storage/vector_store.py`
- `utils/file_utils.py`

## Current-Only Files

| File | Baseline | Bucket |
| --- | --- | --- |
| `indexing/native_coordinator.py` | `current-later` | indexing |
| `indexing/native_runtime.py` | `current-later` | indexing |
| `native/loader.py` | `current-later` | operational |

## Diverged Shared Files

| File | Baseline | Bucket |
| --- | --- | --- |
| `__init__.py` | `restored` | operational |
| `agents/perception.py` | `restored` | secondary-analysis-validation |
| `analysis/dead_code.py` | `restored` | secondary-analysis-validation |
| `analysis/duplicates.py` | `restored` | secondary-analysis-validation |
| `analysis/entry_points.py` | `restored` | secondary-analysis-validation |
| `analysis/impact.py` | `restored` | secondary-analysis-validation |
| `api.py` | `restored` | core-runtime |
| `architecture/topology.py` | `restored` | secondary-analysis-validation |
| `build_system/ingestor.py` | `restored` | secondary-analysis-validation |
| `cli.py` | `restored` | core-runtime |
| `coverage.py` | `restored` | operational |
| `cpu/native_runtime.py` | `restored` | operational |
| `health.py` | `restored` | core-runtime |
| `indexing/auto_scaler.py` | `restored` | indexing |
| `indexing/backends.py` | `restored` | indexing |
| `indexing/engine.py` | `restored` | indexing |
| `indexing/memory_optimized_engine.py` | `restored` | indexing |
| `indexing/native_indexer_bindings.py` | `restored` | indexing |
| `indexing/native_worker.py` | `restored` | indexing |
| `indexing/stats.py` | `restored` | indexing |
| `indexing/tracker.py` | `restored` | indexing |
| `math/pipeline.py` | `restored` | secondary-analysis-validation |
| `native/__init__.py` | `restored` | operational |
| `native/ops/fused_text_tokenizer.py` | `restored` | secondary-analysis-validation |
| `native/ops/lib_loader.py` | `restored` | secondary-analysis-validation |
| `omnigraph/store.py` | `restored` | secondary-analysis-validation |
| `ops/fused_text_tokenizer.py` | `restored` | secondary-analysis-validation |
| `ops/lib_loader.py` | `restored` | secondary-analysis-validation |
| `ops/quantum_ops.py` | `restored` | secondary-analysis-validation |
| `parsing/markdown.py` | `restored` | parse-query |
| `parsing/parser.py` | `restored` | parse-query |
| `query/corpus_rules.py` | `restored` | parse-query |
| `query/pipeline.py` | `restored` | parse-query |
| `reality/store.py` | `restored` | secondary-analysis-validation |
| `requirements/extractor.py` | `restored` | secondary-analysis-validation |
| `roadmap/validator.py` | `restored` | secondary-analysis-validation |
| `sentinel/engines/aes.py` | `restored` | secondary-analysis-validation |
| `sentinel/verifier.py` | `restored` | secondary-analysis-validation |
| `services/platform.py` | `restored` | core-runtime |
| `state/ledger.py` | `restored` | core-runtime |
| `storage/atomic_fs.py` | `restored` | storage |
| `storage/index_state.py` | `restored` | storage |
| `storage/locks.py` | `restored` | storage |
| `storage/memmap_vector_store.py` | `restored` | storage |
| `storage/native_vector_store.py` | `restored` | storage |
| `storage/vector_store.py` | `restored` | storage |
| `utils/file_utils.py` | `restored` | secondary-analysis-validation |
| `validation/engine.py` | `restored` | secondary-analysis-validation |
| `validation/hotspot_capsules.py` | `restored` | secondary-analysis-validation |
| `validation/witnesses.py` | `restored` | secondary-analysis-validation |

## Phase 2 Restoration Set

These files are restored first to re-establish a correctness-first CLI and indexing path:

- `api.py`
- `cli.py`
- `health.py`
- `services/platform.py`
- `state/ledger.py`
- `parsing/parser.py`
- `query/corpus_rules.py`
- `query/pipeline.py`
- `indexing/auto_scaler.py`
- `indexing/backends.py`
- `indexing/engine.py`
- `indexing/memory_optimized_engine.py`
- `indexing/native_indexer_bindings.py`
- `indexing/native_worker.py`
- `indexing/stats.py`
- `indexing/tracker.py`
- `storage/atomic_fs.py`
- `storage/index_state.py`
- `storage/locks.py`
- `storage/memmap_vector_store.py`
- `storage/native_vector_store.py`
- `storage/vector_store.py`

## Default Hot Path After Stabilization Cut

- restored indexing engine and worker behavior
- shared projection local fallback allowed
- no native-only coordinator on the default path
- no native batch row-writer on the default path
