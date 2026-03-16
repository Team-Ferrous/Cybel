# Script Runbook: CPU Hotspot Optimization Loop

This is a copy/paste runbook for iterative CPU kernel optimization with the tuned `silver` benchmark as the default loop and `gold` reserved for final certification evidence.

## 1) Initialize Session

```bash
cd /home/mike/Documents/Github/Anvil
source venv/bin/activate
```

## 2) Baseline Audit (Iter00)

```bash
RUN_ID="native-cpu-loop-$(date +%Y%m%d)-iter00"
./scripts/run_native_qsg_suite.sh --run-id "$RUN_ID"

SUMMARY="audit/runs/$RUN_ID/summary.json"
```

## 3) Print Hotspot Table

```bash
jq -r '
  .models[] as $m |
  ($m.stage_hotspots[:3] // [])[] |
  [$m.model, .stage, .pct_of_decode, .cpp_file, .cpp_function] | @tsv
' "$SUMMARY"
```

## 4) Enforce Patch Eligibility (Before Editing)

Choose target from step 3 only if:
- hotspot `pct_of_decode >= 10.0`
- target belongs to C++ kernel path
- patch scope is one hotspot family this iteration

## 5) Patch + Rebuild

```bash
# Edit only selected C++ hotspot files

bash core/native/build_native.sh
```

## 6) Re-Audit (IterNN)

```bash
PREV_SUMMARY="$SUMMARY"
ITER="01"
RUN_ID="native-cpu-loop-$(date +%Y%m%d)-iter${ITER}"

./scripts/run_native_qsg_suite.sh \
  --run-id "$RUN_ID" \
  --experiment "cpu-hotspot-iter${ITER}"

SUMMARY="audit/runs/$RUN_ID/summary.json"
```

## 7) Compare Prior vs Current

```bash
python - <<'PY' "$PREV_SUMMARY" "$SUMMARY"
import json, sys
p = json.load(open(sys.argv[1]))
c = json.load(open(sys.argv[2]))
pp = {m['model']: m for m in p.get('models', [])}
cc = {m['model']: m for m in c.get('models', [])}
models = ['granite4:tiny-h', 'qwen3.5:4b', 'qwen3.5:9b']
for m in models:
    a, b = pp.get(m, {}), cc.get(m, {})
    ad, bd = float(a.get('decode_tps_p50', 0) or 0), float(b.get('decode_tps_p50', 0) or 0)
    d = ((bd - ad) / ad * 100.0) if ad else 0.0
    ah = float(((a.get('stage_hotspots') or [{}])[0].get('pct_of_decode', 0) or 0))
    bh = float(((b.get('stage_hotspots') or [{}])[0].get('pct_of_decode', 0) or 0))
    print(f"{m}\tdecode_delta_pct={d:.2f}\thotspot_delta_pct_points={bh-ah:.2f}\tpass={b.get('pass')}")
print(f"overall_pass={c.get('overall_pass')}\tpass={c.get('pass')}\tfailure_count={c.get('failure_count')}")
PY
```

## 8) Stop / Continue Decision

Continue unless both are true for 2 consecutive iterations:
- each model `decode_delta_pct < +1.5%`
- no model `hotspot_delta_pct_points <= -2.0`

Stop immediately on any failure:
- `pass=false`
- `failure_count > 0`
- any `decode_time_accounted_pct` not in `[95, 105]`

## 9) Final Certification Pass

```bash
./scripts/run_native_qsg_suite.sh --profile gold
```

## 10) End-of-Task Verification

```bash
saguaro verify . --engines native,ruff,semantic
```
