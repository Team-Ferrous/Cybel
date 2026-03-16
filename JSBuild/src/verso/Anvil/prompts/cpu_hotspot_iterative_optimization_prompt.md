# Prompt: CPU Hotspot Iterative Optimization (3-Model Sequential, Non-Lazy)

Use this prompt when running an agent to optimize CPU inference kernels with strict audit evidence.

---

You are optimizing CPU performance in `/home/mike/Documents/Github/Anvil`.

## Objective
Run an iterative loop for exactly these models in order:
1. `granite4:tiny-h`
2. `qwen3.5:4b`
3. `qwen3.5:9b`

In each iteration: audit -> pick hotspot(s) -> patch C++ kernel(s) -> rebuild -> re-audit -> compare. Stop only when no further gains are demonstrated by the defined criteria.

## Non-Negotiable Rules
- Hotspot-first only: every code patch must map to current `summary.json` hotspot evidence (`cpp_file`, `cpp_function`, `pct_of_decode`).
- No lazy tuning: keep benchmark knobs fixed (`decode_threads`, `batch_threads`, `ubatch`, runs/warmups, prompt).
- No gate-dodging: never relax telemetry/coherence/baseline gates or schema expectations.
- Runtime edits allowed only in C++ kernel paths (no Python audit gating edits).
- One hotspot family per iteration for clear attribution.

## Mandatory Commands

### Baseline (iteration 0)
```bash
source venv/bin/activate
RUN_ID="native-cpu-loop-$(date +%Y%m%d)-iter00"
./scripts/run_native_qsg_suite.sh --run-id "$RUN_ID"
```

### Hotspot readout
```bash
SUMMARY="audit/runs/$RUN_ID/summary.json"
jq -r '.models[] | [.model, .decode_tps_p50, .decode_time_accounted_pct, .stage_hotspots[0].stage, .stage_hotspots[0].pct_of_decode, .stage_hotspots[0].cpp_file, .stage_hotspots[0].cpp_function] | @tsv' "$SUMMARY"
```

### Kernel-targeted exploration (Saguaro SOP)
```bash
saguaro query "<chosen hotspot function>" --k 5
saguaro agent skeleton <chosen cpp file>
saguaro agent slice <chosen symbol> --depth 2
```

### Rebuild
```bash
bash core/native/build_native.sh
```

### Re-audit
```bash
PREV_SUMMARY="$SUMMARY"
RUN_ID="native-cpu-loop-$(date +%Y%m%d)-iter01"
./scripts/run_native_qsg_suite.sh \
  --run-id "$RUN_ID" \
  --experiment "cpu-hotspot-iter01"
SUMMARY="audit/runs/$RUN_ID/summary.json"
```

## Required Output Per Iteration
Return a compact report with:
- `run_id`
- changed C++ files/functions
- top 3 hotspots per model before and after
- `decode_tps_p50` delta per model
- `decode_time_accounted_pct` per model
- explicit decision: continue loop or stop (with criterion hit)

## Stop Criteria (Must Enforce)
Stop after 2 consecutive iterations where:
- all models improve less than `+1.5%` decode TPS vs prior iteration, and
- no model reduces top hotspot share by `>=2.0` percentage points.

Abort immediately if any:
- `pass=false`
- `failure_count > 0`
- `decode_time_accounted_pct` outside `[95, 105]`

## Final Verification (Mandatory)
```bash
saguaro verify . --engines native,ruff,semantic
```

If verification fails, do not claim completion.
