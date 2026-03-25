#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

VENV_STATUS="missing"
if [[ -d "$REPO_ROOT/venv" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/venv/bin/activate"
  VENV_STATUS="activated:$REPO_ROOT/venv"
fi

# Silver is the engineering default loop. Gold remains the stricter
# certification campaign and will still chain prerequisite stages.
DEFAULT_PROFILE="${ANVIL_SUITE_DEFAULT_PROFILE:-silver}"

detect_cpu_list() {
  local candidate=""
  local path=""
  for path in \
    "/sys/fs/cgroup/cpuset.cpus.effective" \
    "/sys/fs/cgroup/cpuset/cpuset.cpus" \
    "/sys/devices/system/cpu/online"; do
    if [[ -r "$path" ]]; then
      candidate="$(tr -d '[:space:]' < "$path" || true)"
      if [[ -n "$candidate" ]]; then
        echo "$candidate"
        return 0
      fi
    fi
  done
  local online
  online="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"
  if [[ "$online" =~ ^[0-9]+$ ]] && [[ "$online" -gt 0 ]]; then
    echo "0-$((online - 1))"
    return 0
  fi
  echo "0"
}

if [[ -z "${ANVIL_SUITE_TARGET_CPUS:-}" ]]; then
  export ANVIL_SUITE_TARGET_CPUS="$(detect_cpu_list)"
fi

parse_profile_name() {
  local token=""
  local next_is_profile=0
  for token in "$@"; do
    if [[ "$next_is_profile" == "1" ]]; then
      echo "$token"
      return 0
    fi
    case "$token" in
      --profile)
        next_is_profile=1
        ;;
      --profile=*)
        echo "${token#--profile=}"
        return 0
        ;;
    esac
  done
  echo "$DEFAULT_PROFILE"
}

PROFILE_NAME="$(parse_profile_name "$@")"
if [[ "${PROFILE_NAME}" == "smoke" ]]; then
  PROFILE_NAME="bronze"
fi

parse_run_id() {
  local token=""
  local next_is_run_id=0
  for token in "$@"; do
    if [[ "$next_is_run_id" == "1" ]]; then
      echo "$token"
      return 0
    fi
    case "$token" in
      --run-id)
        next_is_run_id=1
        ;;
      --run-id=*)
        echo "${token#--run-id=}"
        return 0
        ;;
    esac
  done
  echo ""
}

RUN_ID="$(parse_run_id "$@")"

parse_models_arg() {
  local token=""
  local next_is_models=0
  for token in "$@"; do
    if [[ "$next_is_models" == "1" ]]; then
      echo "$token"
      return 0
    fi
    case "$token" in
      --models)
        next_is_models=1
        ;;
      --models=*)
        echo "${token#--models=}"
        return 0
        ;;
    esac
  done
  echo ""
}

MODEL_FILTER="$(parse_models_arg "$@")"

build_forwarded_args() {
  FORWARDED_ARGS=()
  local token=""
  local skip_next=0
  for token in "$@"; do
    if [[ "$skip_next" == "1" ]]; then
      skip_next=0
      continue
    fi
    case "$token" in
      --profile|--run-id)
        skip_next=1
        ;;
      --profile=*|--run-id=*)
        ;;
      *)
        FORWARDED_ARGS+=("$token")
        ;;
    esac
  done
}

FORWARDED_ARGS=()
build_forwarded_args "$@"
export ANVIL_SUITE_UI_MODE="${ANVIL_SUITE_UI_MODE:-glassbox}"
export ANVIL_SUITE_LOG_LEVEL="${ANVIL_SUITE_LOG_LEVEL:-trace}"
export ANVIL_SUITE_VERBOSE_TERMINAL="${ANVIL_SUITE_VERBOSE_TERMINAL:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

# If the caller shell is CPU-affined too narrowly, expand to all online CPUs
# for benchmark fidelity (unless explicitly disabled).
TASKSET_PREFIX=()
AFFINITY_NOTE="unmodified"
if [[ "${ANVIL_DISABLE_AUTO_AFFINITY_EXPAND:-0}" != "1" ]]; then
  if command -v taskset >/dev/null 2>&1; then
    TARGET_LIST="${ANVIL_SUITE_TARGET_CPUS}"
    if [[ -n "${TARGET_LIST:-}" ]]; then
      BEFORE_AFFINITY="$(taskset -cp $$ 2>/dev/null | awk -F': ' '{print $2}' | tail -n1)"
      taskset -cp "$TARGET_LIST" $$ >/dev/null 2>&1 || true
      AFTER_AFFINITY="$(taskset -cp $$ 2>/dev/null | awk -F': ' '{print $2}' | tail -n1)"
      if taskset -c "$TARGET_LIST" true >/dev/null 2>&1; then
        TASKSET_PREFIX=(taskset -c "$TARGET_LIST")
        AFFINITY_NOTE="taskset-prefix:${TARGET_LIST}"
      fi
      if [[ -n "${AFTER_AFFINITY:-}" && "${AFTER_AFFINITY}" != "$TARGET_LIST" ]]; then
        echo "warning: CPU affinity remains constrained (${AFTER_AFFINITY}); using max visible threads in this scope." >&2
        if [[ -n "${BEFORE_AFFINITY:-}" && "${BEFORE_AFFINITY}" != "${AFTER_AFFINITY}" ]]; then
          echo "warning: affinity changed from ${BEFORE_AFFINITY} to ${AFTER_AFFINITY}, but not full ${TARGET_LIST}." >&2
        fi
        AFFINITY_NOTE="constrained:${AFTER_AFFINITY}"
      elif [[ -n "${AFTER_AFFINITY:-}" ]]; then
        AFFINITY_NOTE="expanded:${BEFORE_AFFINITY:-unknown}->${AFTER_AFFINITY}"
      fi
    fi
  fi
fi

run_suite_python() {
  if [[ "${#TASKSET_PREFIX[@]}" -gt 0 ]]; then
    "${TASKSET_PREFIX[@]}" python -m audit.runner.benchmark_suite "$@"
    return $?
  fi
  python -m audit.runner.benchmark_suite "$@"
}

helper_python() {
  if [[ "${#TASKSET_PREFIX[@]}" -gt 0 ]]; then
    "${TASKSET_PREFIX[@]}" python - "$@"
    return $?
  fi
  python - "$@"
}

if [[ -z "${RUN_ID}" ]]; then
  RUN_ID="$(
    PROFILE_NAME="${PROFILE_NAME}" REPO_ROOT="${REPO_ROOT}" helper_python <<'PY'
import os
from pathlib import Path

from audit.runner.benchmark_suite import _default_run_id
from audit.runner.suite_profiles import load_suite_profile

repo_root = Path(os.environ["REPO_ROOT"])
profile_name = os.environ["PROFILE_NAME"]
spec = load_suite_profile(repo_root / "audit" / "profiles" / f"native_qsg_{profile_name}.yaml")
print(_default_run_id(repo_root, spec))
PY
)"
fi

RUN_ROOT="$REPO_ROOT/audit/runs/$RUN_ID"
TRANSCRIPT_PATH="$RUN_ROOT/terminal_transcript.log"
mkdir -p "$RUN_ROOT"
touch "$TRANSCRIPT_PATH"
export ANVIL_SUITE_TRANSCRIPT_FROM_STDERR=1
exec > >(tee -a "$TRANSCRIPT_PATH") 2>&1

log_shell() {
  printf '[%s] WRAP  %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

maybe_auto_calibrate() {
  if [[ "${ANVIL_DISABLE_AUTO_CALIBRATE:-0}" == "1" ]]; then
    return 0
  fi

  PROFILE_NAME="${PROFILE_NAME}" REPO_ROOT="${REPO_ROOT}" MODEL_FILTER="${MODEL_FILTER}" helper_python <<'PY'
import os
import sys
from pathlib import Path

from audit.runner.suite_profiles import load_suite_profile
from audit.runner.suite_certification import (
    assess_runtime_tuning,
    bootstrap_runtime_tuning,
    format_runtime_tuning_summary,
    has_benchmark_evidence,
    mark_runtime_tuning_deferred,
    resolve_runtime_tuning_bootstrap_policy,
    should_bootstrap_runtime_tuning,
)

repo_root = Path(os.environ["REPO_ROOT"])
profile_name = os.environ["PROFILE_NAME"]
raw_models = str(os.environ.get("MODEL_FILTER", "")).strip()
models = [item.strip() for item in raw_models.split(",") if item.strip()]
spec = load_suite_profile(repo_root / "audit" / "profiles" / f"native_qsg_{profile_name}.yaml")
if str(spec.tuning_contract_policy or "").strip() != "required":
    raise SystemExit(0)
if str(spec.profile_name or "").strip().lower() not in {"gold", "platinum"}:
    raise SystemExit(0)
report = assess_runtime_tuning(
    repo_root,
    profile_name=profile_name,
    models=models or None,
    invocation_source="campaign",
)
policy = resolve_runtime_tuning_bootstrap_policy()
for line in format_runtime_tuning_summary(report):
    print(line, file=sys.stderr)
if bool(report.get("ready")):
    raise SystemExit(0)
if not should_bootstrap_runtime_tuning(
    report,
    policy=policy,
    has_prior_benchmark_evidence=has_benchmark_evidence(repo_root),
):
    deferred = mark_runtime_tuning_deferred(
        report,
        reason=f"bootstrap_policy={policy}",
        policy=policy,
    )
    for line in format_runtime_tuning_summary(deferred):
        print(line, file=sys.stderr)
    raise SystemExit(0)
print(f"Running `{profile_name}` calibration bootstrap.", file=sys.stderr)
report = bootstrap_runtime_tuning(
    repo_root,
    profile_name=profile_name,
    models=models or None,
    auto_run=True,
    stream_output=True,
    invocation_source="campaign",
)
for line in format_runtime_tuning_summary(report):
    print(line, file=sys.stderr)
if str(report.get("status") or "") == "failed":
    sys.exit(int(report.get("bootstrap_return_code") or 1))
PY
}

campaign_stage_sequence() {
  PROFILE_NAME="${PROFILE_NAME}" REPO_ROOT="${REPO_ROOT}" helper_python <<'PY'
import os

from audit.runner.suite_certification import campaign_stage_sequence

for stage in campaign_stage_sequence(os.environ["PROFILE_NAME"]):
    print(stage)
PY
}

run_stage_profile() {
  local stage="$1"
  local child_cmd=("$0" "${FORWARDED_ARGS[@]}" --profile "$stage")
  log_shell "campaign_stage_start=$stage"
  log_shell "campaign_stage_command=${child_cmd[*]}"
  if ANVIL_SUITE_DISABLE_GOLD_STAGE_CHAIN=1 "${child_cmd[@]}"; then
    log_shell "campaign_stage_complete=$stage rc=0"
  else
    local rc=$?
    log_shell "campaign_stage_failed=$stage rc=$rc"
    return $rc
  fi
}

# Benchmarks should not inherit single-thread OpenMP caps from unrelated tooling.
if [[ -z "${OMP_NUM_THREADS:-}" || "${OMP_NUM_THREADS}" == "1" ]]; then
  if [[ -n "${ANVIL_SUITE_TARGET_CPUS:-}" ]]; then
    OMP_AUTO="$(python - <<'PY'
import os
raw = os.getenv("ANVIL_SUITE_TARGET_CPUS", "")
total = 0
for part in raw.split(","):
    part = part.strip()
    if not part:
        continue
    if "-" in part:
        lo, _, hi = part.partition("-")
        try:
            a = int(lo)
            b = int(hi)
        except Exception:
            continue
        if b < a:
            a, b = b, a
        total += (b - a + 1)
    else:
        try:
            int(part)
        except Exception:
            continue
        total += 1
print(max(1, total))
PY
)"
  else
    OMP_AUTO="$(getconf _NPROCESSORS_ONLN)"
  fi
  export OMP_NUM_THREADS="${ANVIL_OMP_NUM_THREADS:-${OMP_AUTO}}"
fi

RUNNER_ARGS=("$@")
if [[ -z "$(parse_run_id "$@")" ]]; then
  RUNNER_ARGS+=(--run-id "$RUN_ID")
fi

log_shell "repo_root=$REPO_ROOT"
log_shell "venv_status=$VENV_STATUS"
log_shell "default_profile=$DEFAULT_PROFILE"
log_shell "profile=$PROFILE_NAME run_id=$RUN_ID"
log_shell "run_root=$RUN_ROOT"
log_shell "transcript=$TRANSCRIPT_PATH"
log_shell "python=$(command -v python)"
log_shell "python_version=$(python --version 2>&1)"
log_shell "ui_mode=$ANVIL_SUITE_UI_MODE log_level=$ANVIL_SUITE_LOG_LEVEL"
log_shell "target_cpus=${ANVIL_SUITE_TARGET_CPUS:-unset}"
log_shell "affinity=$AFFINITY_NOTE"
log_shell "omp_num_threads=${OMP_NUM_THREADS:-unset} omp_proc_bind=${OMP_PROC_BIND:-unset} omp_places=${OMP_PLACES:-unset}"
log_shell "models=${MODEL_FILTER:-all}"
if [[ "${#TASKSET_PREFIX[@]}" -gt 0 ]]; then
  log_shell "taskset_prefix=${TASKSET_PREFIX[*]}"
else
  log_shell "taskset_prefix=none"
fi

if [[ "${ANVIL_SUITE_DISABLE_GOLD_STAGE_CHAIN:-0}" != "1" && ( "${PROFILE_NAME}" == "gold" || "${PROFILE_NAME}" == "platinum" ) ]]; then
  mapfile -t CAMPAIGN_STAGES < <(campaign_stage_sequence)
  log_shell "campaign_sequence=${CAMPAIGN_STAGES[*]}"
  for stage in "${CAMPAIGN_STAGES[@]}"; do
    if [[ "$stage" == "${PROFILE_NAME}" ]]; then
      continue
    fi
    run_stage_profile "$stage"
  done
fi

log_shell "handoff_command=python -m audit.runner.benchmark_suite ${RUNNER_ARGS[*]}"
maybe_auto_calibrate
run_suite_python "${RUNNER_ARGS[@]}"
