# Native QSG Benchmark Host State

This runbook covers the host-side setup for the native QSG audit tiers.

For tier semantics and artifact interpretation, read [native_qsg_audits.md](/home/mike/Documents/Github/Anvil/docs/benchmarking/native_qsg_audits.md).

## Active Profiles

- `bronze`: reduced all-on admission audit
- `silver`: engineering audit default
- `gold-fast`: fast pre-release gate
- `gold`: certification audit
- `platinum`: publication-grade and deep-stress certification
- `calibrate`: explicit tuning-contract refresh

`smoke` is a compatibility alias for `bronze`.

## Mandatory Basics

Run from the repo root:

```bash
source venv/bin/activate
saguaro health
```

Strict profiles require a healthy host and working tooling. The main checks are:

- repo `venv` active
- `saguaro health` succeeds
- model digests validate
- `perf` is usable on certify-mode profiles
- CPU governor and THP satisfy the checked-in host contract

## Quick Commands

Check current host state:

```bash
cat /proc/sys/kernel/perf_event_paranoid
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
cat /sys/kernel/mm/transparent_hugepage/enabled
perf stat -o /dev/null true
source venv/bin/activate && saguaro health
```

Useful bootstrap packages on Debian or Ubuntu:

```bash
sudo apt-get install -y build-essential cmake pkg-config libnuma-dev libhwloc-dev linux-tools-common numactl
```

If you need kernel-matched perf tooling:

```bash
sudo apt-get install -y linux-tools-$(uname -r)
```

## Fixes For Strict Profiles

Lower `perf_event_paranoid` for the current boot:

```bash
sudo sysctl -w kernel.perf_event_paranoid=-1
```

Set CPU governor to `performance`:

```bash
for cpu in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor; do
  echo performance | sudo tee "$cpu" >/dev/null
done
```

Set THP to `madvise` for the current boot:

```bash
echo madvise | sudo tee /sys/kernel/mm/transparent_hugepage/enabled >/dev/null
```

Persist `perf_event_paranoid`:

```bash
echo 'kernel.perf_event_paranoid = -1' | sudo tee /etc/sysctl.d/99-anvil-benchmarks.conf >/dev/null
sudo sysctl --system
```

## Tuning Contracts

Calibration is now admission-driven instead of blanket-run.

That means:

- REPL or `anvil.py` first-run bootstrap may launch `calibrate` in `probe` mode
- `gold` and `platinum` may auto-refresh contracts when admission reports stale or missing runtime tuning
- `bronze`, `silver`, and `gold-fast` do not auto-run calibration
- you can still refresh contracts explicitly with `calibrate`

Refresh contracts explicitly:

```bash
./scripts/run_native_qsg_suite.sh --profile calibrate
```

Disable REPL first-run bootstrap if needed:

```bash
export ANVIL_RUNTIME_TUNING_BOOTSTRAP_POLICY=explicit
```

## Recommended Operator Flow

1. Check host state.
2. Run `bronze` or `silver`.
3. If tuning debt is reported during `silver` or `gold-fast`, run `calibrate` explicitly if you want a fresh contract before release gates.
4. Run `gold-fast`.
5. Run `gold` or `platinum`. Those stages may auto-refresh tuning if admission requires it.

## Run Locations

Every run writes under `audit/runs/<run_id>/`.

For host and preflight issues inspect:

- `preflight.json`
- `suite_status.json`
- `summary.json`
- `reports/executive_summary.md`
