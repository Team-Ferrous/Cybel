# Gold And Platinum Native QSG Audits

This document is the release-operator companion to [native_qsg_audits.md](/home/mike/Documents/Github/Anvil/docs/benchmarking/native_qsg_audits.md).

## Gold

Run:

```bash
./scripts/run_native_qsg_suite.sh --profile gold
```

The script chains:

1. `bronze`
2. `silver`
3. `gold-fast`
4. `gold`

Gold is the internal certification lane. It consumes:

- all-on canonical performance evidence
- scheduler surface evidence
- kernel hotspot evidence
- quality evidence across perplexity, confidence, coherence, and accuracy
- tuning-contract validation

Gold is the right tier when you want a release decision for the real QSG pipeline.

Gold now owns calibration admission. If the wrapper sees stale or missing tuning contracts, it launches `calibrate` in `search` mode before the final `gold` stage and records the tuned runtime envelope in the run artifacts.

## Platinum

Run:

```bash
./scripts/run_native_qsg_suite.sh --profile platinum
```

The script chains:

1. `bronze`
2. `silver`
3. `gold-fast`
4. `gold`
5. `platinum`

Platinum is the deepest tier. Use it when you want:

- a publication-grade benchmark package
- the heaviest end-to-end stress evidence
- maximum confidence before posting results or signing off a certification campaign

Platinum uses the same admission model but upgrades the tuning budget to `deep_search`.

## What To Review

For both `gold` and `platinum`, inspect:

- `summary.json`
- `metrics_rollup.json`
- `continuous/<model>.json`
- `kernel/summary.json`
- `eval/quality_summary.json`
- `agent_handoff.json`
- `reports/tuning_receipt.json`
- `reports/tuning_remediation.json`
- `reports/publication_manifest.json` for `platinum`

The most important model-level fields are:

- `decode_tps_p50`, `decode_tps_p95`
- `e2e_tps_p50`, `e2e_tps_p95`
- `ttft_ms_p50`, `ttft_ms_p95`
- `scheduler_queue_wait_ms_p95`
- `continuous_tpot_ms_p50`
- `continuous_decode_tps_global_p50`
- `continuous_queue_wait_ms_p95`
- `continuous_fairness_min`
- `perplexity`
- `confidence_mean`
- `coherence_pass_rate`
- `accuracy_pass_rate`

For the tuning path, also inspect:

- `summary.json["calibration"]["admission"]`
- `summary.json["calibration"]["contracts"]`
- `summary.json["calibration"]["scheduler_frontiers"]`
- `summary.json["calibration"]["kernel_summary"]`

## Failure Interpretation

- If `bronze` fails, the host or pipeline is not ready for release gating.
- If `silver` fails, the system usually has a performance or quality regression that should be fixed before stricter gates.
- If `gold-fast` fails, the candidate is not ready for the full cert sweep.
- If `gold` fails, treat it as a release blocker.
- If `platinum` fails, treat it as a publication or certification blocker even if lower tiers passed.
