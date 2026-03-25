# Native QSG Audit Tiers

This document is the operator reference for the native QSG benchmark audit system.

The audit system now treats certification and optimization as separate concerns:

- certification tiers run the real all-on QSG pipeline only
- calibration is the only place that runs runtime tuning search
- audit quality is four-family evidence, not just coherence
- performance evidence is expanded to include native, scheduler, and kernel-level detail
- release tiers do not run feature-off ablations
- auto-calibration is restricted to REPL first-run bootstrap plus Gold and Platinum pre-certification

## Environment

Run from the repo root:

```bash
source venv/bin/activate
saguaro health
./scripts/run_native_qsg_suite.sh --profile bronze
```

The suite is single-host and single-model-at-a-time by design. It intentionally avoids parallel model execution because that corrupts TTFT, scheduler wait, hotspot attribution, and quality evidence.

## Tier Ladder

### Bronze

Purpose: fast all-on admission audit for development hosts.

Command:

```bash
./scripts/run_native_qsg_suite.sh --profile bronze
```

Typical wall clock: 10-20 minutes.

Bronze runs:

- `canonical_all_on`
- `continuous_scheduler`
- `kernel_microbench`
- `quality_eval`

Use Bronze when:

- you need a quick correctness and performance read on the real QSG path
- you want hotspot evidence without committing to a release campaign
- you are validating a local optimization patch

### Silver

Purpose: engineering optimization audit.

Command:

```bash
./scripts/run_native_qsg_suite.sh --profile silver
```

Typical wall clock: 30-60 minutes.

Silver is the default operator loop. It stays all-on, does not run feature-off variants, and is the right tier for daily tuning and regression tracking.

Silver does not auto-run calibration. If tuning contracts are missing or stale it still runs the all-on audit with fallback runtime settings and records the tuning debt in the artifacts.

### Gold-Fast

Purpose: pre-release regression gate with stricter all-on evidence.

Command:

```bash
./scripts/run_native_qsg_suite.sh --profile gold-fast
```

Typical wall clock: 60-90 minutes.

Gold-fast is the fast cert gate. It uses the all-on pipeline, requires stronger contract alignment, and is the stage to run before a full certification sweep.

Gold-fast does not auto-run calibration. It is intentionally a fast gate, not a tuning stage.

### Gold

Purpose: internal certification audit.

Command:

```bash
./scripts/run_native_qsg_suite.sh --profile gold
```

Typical wall clock: 2-4 hours.

When you run `gold` through `./scripts/run_native_qsg_suite.sh`, the script chains:

1. `bronze`
2. `silver`
3. `gold-fast`
4. `gold`

Set `ANVIL_SUITE_DISABLE_GOLD_STAGE_CHAIN=1` only if you deliberately want to run the final stage without the full campaign chain.

Before the `gold` stage runs, the wrapper performs a runtime tuning admission check. If contracts are missing or stale it launches `calibrate` in `search` mode, then re-checks readiness before the final stage.

### Platinum

Purpose: publication-grade and deep-stress certification.

Command:

```bash
./scripts/run_native_qsg_suite.sh --profile platinum
```

Typical wall clock: 4-8 hours.

Platinum chains:

1. `bronze`
2. `silver`
3. `gold-fast`
4. `gold`
5. `platinum`

Use Platinum when you want the deepest end-to-end evidence package before publishing benchmark results or signing off a certifiable release.

Before the `platinum` stage runs, the wrapper performs the same admission check but escalates calibration to `deep_search` mode when refresh is required.

### Calibrate

Purpose: optimizer-only runtime tuning search.

Command:

```bash
./scripts/run_native_qsg_suite.sh --profile calibrate
```

Calibration is not a generic daily audit tier anymore. It is a bounded runtime tuning service used in three places:

- REPL or `anvil.py` first-run bootstrap in `probe` mode
- Gold pre-certification refresh in `search` mode
- Platinum pre-certification refresh in `deep_search` mode

Calibration runs:

- thread and `ubatch` search
- continuous batching search
- scheduler and pager envelope selection
- quality gating on the winning all-on candidate
- capability-ledger and tuning-contract writeback
- tuning receipt and remediation packet generation

Run it when:

- `silver` or `gold-fast` reports stale or missing tuning contracts and you want to refresh them explicitly
- the host changed
- benchmark harness code changed materially
- you want to reseed the REPL bootstrap envelope manually

## What The Suite Measures

### Native attempt metrics

Per measured canonical attempt the suite records:

- wall throughput: `wall_tokens_per_second`
- prefill throughput: `prefill_tps_raw`, `prefill_tps`, `effective_prefill_tps`
- decode throughput: `decode_tps`
- end-to-end throughput: `e2e_tps`
- TTFT: `ttft_ms`
- per-token latency distribution: `p10_ms`, `p25_ms`, `p50_ms`, `p75_ms`, `p95_ms`, `p99_ms`, `stddev_ms`, `min_ms`, `max_ms`
- scheduler overhead: `scheduler_queue_wait_ms`, `scheduler_iteration_ms`
- stage timings: `graph_prefill_avg_ms`, `graph_decode_avg_ms`, `sample_avg_ms`, `logits_processor_avg_ms`, `penalty_avg_ms`, `suppression_avg_ms`
- speculative acceptance accounting: accepted, rejected, and proposed parallel tokens

### Scheduler surface metrics

The continuous scheduler lane records:

- `ttft_ms_p50`
- `ttft_ms_p95`
- `tpot_ms_p50`
- `queue_wait_ms_p95`
- `queue_wait_ms_p99`
- `scheduler_iteration_ms_p95`
- `decode_tps_global`
- `decode_goodput_tps`
- per-agent decode throughput
- fairness
- fragmentation and coconut/drift telemetry where available

### Kernel evidence

The kernel harness records:

- per-kernel latency and variance
- PMU-derived signals when available
- estimated recoverable gain
- decode-share attribution
- hotspot confidence and stability

### Quality evidence

Quality is now four-family evidence:

- held-out perplexity
- token-confidence calibration
- coherence rubric
- deterministic task accuracy

Accuracy is generated through `audit/eval/native_logits.py` and supports deterministic answer grading such as option-letter and contains-style matching.

## Lane Semantics

### `canonical_all_on`

This is the production-path audit lane. It always runs with the QSG feature set enabled. This lane is the source of truth for release-tier throughput, TTFT, token latency, stage timing, and quality results.

### `continuous_scheduler`

This lane stress-tests queueing and multi-request scheduler behavior. It is where TTFT tail, TPOT, fairness, and queue-wait evidence come from.

### `kernel_microbench`

This lane isolates kernel and stage hotspots. It is the fastest way to answer “where is the next recoverable win?”

### `quality_eval`

This lane runs all-on quality scoring for:

- perplexity corpora
- confidence corpora
- coherence rubric prompts
- accuracy corpora

### `thread_matrix`

This is calibration-only. It is not part of Bronze, Silver, Gold-fast, Gold, or Platinum.

### `calibration`

This is a governed tuning lane, not a release lane. It tunes:

- thread tuple: `decode_threads`, `batch_threads`, `ubatch`
- continuous batching: `max_active_requests`, `scheduler_policy`, `batch_wait_timeout_ms`, `continuous_interleaved_streams`
- pager envelope: `state_page_rows`, `max_prefill_rows_per_iteration`

The lane emits an admission decision:

- `skip`
- `probe`
- `search`
- `revoke`

## Artifacts

Every run writes `audit/runs/<run_id>/`.

The most important files are:

- `summary.json`: top-level audit result
- `metrics_rollup.json`: reduced machine-readable view for dashboards and comparisons
- `native/attempts.ndjson`: normalized attempt records with expanded runtime metrics
- `native/phases.ndjson`: per-phase timing records including scheduler phases
- `quality/attempts.ndjson` or `eval/quality_summary.json`: quality-family outputs
- `continuous/<model>.json`: scheduler surface results
- `kernel/summary.json`: kernel microbenchmark outputs
- `reports/tuning_receipt.json`: selected runtime envelope and admission details
- `reports/tuning_remediation.json`: operator-facing next-step packet for tuning debt or weak scheduler behavior
- `agent_handoff.json`: hotspot-first optimization handoff bundle
- `reports/executive_summary.md`: human-readable summary

## Reading The Summary

Each model summary now carries more than the legacy `decode_tps_p50`, `e2e_tps_p50`, and `ttft_ms_p95`.

Key fields to read:

- `wall_tps_p50`, `wall_tps_p95`
- `prefill_tps_p50`, `prefill_tps_p95`
- `decode_tps_p50`, `decode_tps_p95`
- `e2e_tps_p50`, `e2e_tps_p95`
- `ttft_ms_p50`, `ttft_ms_p95`
- `per_token_latency_p10_ms` through `per_token_latency_p99_ms`
- `scheduler_queue_wait_ms_p50`, `scheduler_queue_wait_ms_p95`
- `scheduler_iteration_ms_p50`, `scheduler_iteration_ms_p95`
- `continuous_decode_tps_global_p50`, `continuous_decode_tps_global_p95`
- `continuous_tpot_ms_p50`
- `continuous_queue_wait_ms_p95`
- `continuous_fairness_min`
- `runtime_stage_avg_ms`
- `perplexity`
- `confidence_mean`
- `coherence_pass_rate`
- `accuracy_pass_rate`
- `accuracy_exact_match_rate`

## Operator Workflow

1. Activate the repo environment.

```bash
source venv/bin/activate
```

2. Check Saguaro and host health.

```bash
saguaro health
./venv/bin/saguaro verify . --engines native,ruff,semantic --format json
```

3. Run `bronze` after substantial runtime changes.

```bash
./scripts/run_native_qsg_suite.sh --profile bronze
```

4. Run `silver` for the normal optimization loop.

```bash
./scripts/run_native_qsg_suite.sh --profile silver
```

5. If tuning debt is reported, refresh contracts explicitly or let Gold/Platinum refresh it during the chained campaign.

```bash
./scripts/run_native_qsg_suite.sh --profile calibrate
```

6. Run `gold-fast` before a release candidate.

```bash
./scripts/run_native_qsg_suite.sh --profile gold-fast
```

7. Run `gold` for certification or `platinum` for publication-grade evidence.

```bash
./scripts/run_native_qsg_suite.sh --profile gold
./scripts/run_native_qsg_suite.sh --profile platinum
```

## Compatibility Notes

- `smoke` is now a compatibility alias for `bronze`
- feature-off ablation lanes are no longer part of release audits
- if you need combinatorial feature or thread experiments, use `calibrate` or add a dedicated exploratory profile instead of mutating the certification tiers
