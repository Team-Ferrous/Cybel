# Native QSG Runtime Tuning

This document is the operator and engineer reference for the native QSG runtime tuning system.

For tier semantics, read [native_qsg_audits.md](/home/mike/Documents/Github/Anvil/docs/benchmarking/native_qsg_audits.md). For release operation, read [native_qsg_gold.md](/home/mike/Documents/Github/Anvil/docs/benchmarking/native_qsg_gold.md).

## What Tuning Does

Runtime tuning is a controlled search over the production all-on QSG path. It does not disable QSG features and it does not run feature-off ablations.

The tuner now searches three control surfaces:

- thread tuple: `decode_threads`, `batch_threads`, `ubatch`
- continuous batching: `max_active_requests`, `scheduler_policy`, `batch_wait_timeout_ms`, `continuous_interleaved_streams`
- pager envelope: `state_page_rows`, `max_prefill_rows_per_iteration`

The tuner writes a v2 tuning contract that includes:

- thread configuration
- continuous batching configuration
- pager configuration
- objective vector
- safety envelope
- admission metadata
- capability ledger

## Admission Model

The system does not blindly calibrate. It first computes an admission decision:

- `skip`: current contracts are acceptable or the calling context is not allowed to auto-tune
- `probe`: bounded first-run bootstrap, used by REPL or `anvil.py`
- `search`: Gold-grade refresh budget
- `revoke`: existing contract is stale enough that it should be replaced before trust is restored

The current auto-run policy is:

- REPL first-run: may run `probe`
- `gold`: may run `search`
- `platinum`: may run `deep_search`
- `bronze`, `silver`, `gold-fast`: never auto-run tuning

## Search Modes

### Probe

Use case: first-time REPL or Anvil startup.

Properties:

- minimal thread search
- reduced continuous batching surface
- bounded quality sample limit
- no or near-zero kernel overhead

Typical invocation:

```bash
python -m audit.runner.benchmark_suite \
  --profile calibrate \
  --calibration-mode probe \
  --calibration-source repl_startup \
  --calibration-target-profile gold
```

### Search

Use case: Gold pre-certification refresh or explicit manual refresh.

Properties:

- two-stage thread search
- bounded continuous batching sweep
- quality gating
- reduced kernel microbench pass for hotspot watchlists

Typical invocation:

```bash
./scripts/run_native_qsg_suite.sh --profile calibrate
```

### Deep Search

Use case: Platinum pre-certification refresh.

Properties:

- expanded finalist set
- wider scheduler surface
- larger kernel hotspot pass
- tighter safety envelope review

Typical invocation:

```bash
python -m audit.runner.benchmark_suite \
  --profile calibrate \
  --calibration-mode deep_search \
  --calibration-source campaign \
  --calibration-target-profile platinum
```

## Objective Model

The tuner does not rank candidates with a single opaque score.

It uses a constraint-first selection model:

1. preserve quality
2. preserve safety envelope
3. maximize scheduler-aware goodput
4. minimize TTFT and queue wait
5. prefer lower runtime variance and cleaner PMU behavior

The main objective fields written into the contract are:

- `decode_tps_median`
- `ttft_ms_median`
- `ttft_ms_p95`
- `queue_wait_ms_p95`
- `fairness`
- `decode_goodput_tps`
- `hotspot_penalty`

## Safety Envelope

Every promoted contract carries a rollback-oriented safety envelope.

The current envelope includes:

- fairness floor
- queue-wait ceiling
- quality regression policy
- decode-TPS regression floor
- hotspot watch penalty

If the runtime later violates this envelope, the contract should be treated as stale or revocable.

## Artifacts

Calibration writes the normal audit run bundle under `audit/runs/<run_id>/`, plus tuning-specific operator artifacts:

- `summary.json["calibration"]`
- `reports/tuning_receipt.json`
- `reports/tuning_remediation.json`
- `continuous/<model>.json`
- `kernel/summary.json`
- `eval/quality_summary.json`

The most important tuning artifacts are:

- `reports/tuning_receipt.json`
  - selected runtime envelope
  - admission metadata
  - written contract paths
  - winning candidate summary
- `reports/tuning_remediation.json`
  - stale or missing model list
  - fairness or queue-wait alerts
  - hotspot watchlist recommendations

## Reading A Tuning Contract

Look at these fields first:

- `thread_config`
- `continuous_config`
- `pager_config`
- `objective_vector`
- `safe_envelope`
- `admission`
- `capability_ledger`
- `calibration_stats.scheduler_metrics`
- `calibration_stats.hotspot_metrics`

## Recommended Operator Flow

1. Start with `bronze` or `silver`.
2. If tuning debt is reported, decide whether to refresh now or defer until `gold`.
3. Run `calibrate` explicitly when you want to inspect the tuning frontier before release gates.
4. Run `gold` or `platinum` and review `reports/tuning_receipt.json`.
5. If `reports/tuning_remediation.json` contains hotspot or fairness actions, use that as the next optimization loop.

## Useful Environment Controls

Default REPL bootstrap behavior:

```bash
echo "${ANVIL_RUNTIME_TUNING_BOOTSTRAP_POLICY:-on_first_run}"
```

Disable REPL bootstrap:

```bash
export ANVIL_RUNTIME_TUNING_BOOTSTRAP_POLICY=explicit
```

Disable bootstrap completely:

```bash
export ANVIL_DISABLE_RUNTIME_TUNING_BOOTSTRAP=1
```
