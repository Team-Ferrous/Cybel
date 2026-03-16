# AES Compliance Dashboard Specification

Owner: Platform Governance
Cadence: Weekly review, monthly trend review, quarterly policy calibration

## Objective

Provide continuous visibility into AES conformance so regressions are detected before release and prioritization is driven by objective risk trends.

## Core Metrics

1. Trace closure lag
- Definition: median hours between first high-assurance change detection and valid trace/evidence closure.
- Target (SLO): <= 24h for AAL-0/1, <= 72h for AAL-2.

2. Evidence bundle completeness rate
- Definition: percentage of high-assurance changes with complete evidence artifacts.
- Target (SLO): >= 98% rolling 30 days.

3. Waiver staleness
- Definition: count of waivers expired or expiring within 7 days.
- Target (SLO): 0 expired waivers on protected branches.

4. Independent review coverage
- Definition: percentage of AAL-0/1 changes with independent review proof attached.
- Target (SLO): 100%.

5. Red-team completion rate
- Definition: percentage of required red-team artifacts present for AAL-0/1 changes.
- Target (SLO): 100%, with 0 unresolved critical findings.

6. Compliance gate pass rate
- Definition: percentage of CI runs passing verify/deadcode/impact/audit hard-gate bundle.
- Target (SLO): >= 95% per rolling 14 days.

## Data Sources

- `saguaro verify --engines native,ruff,semantic,aes --format json`
- `saguaro verify --aes-report --format json`
- `saguaro deadcode --format json`
- `saguaro audit --format json`
- `aiChangeLog/*.json` chronicle delta receipts
- `standards/traceability/TRACEABILITY.jsonl`
- waiver stores under `standards/waivers*.jsonl`

## Alerting Policy

- P0 alert:
  - any unresolved critical red-team finding
  - any missing independent review on AAL-0/1 change
  - any expired waiver in active branch
- P1 alert:
  - evidence completeness < 98%
  - trace closure lag breach for 2 consecutive days
- P2 alert:
  - compliance gate pass rate < 95% for 3 consecutive days

## Reporting Surfaces

- CI artifact bundle: `.anvil/artifacts/phase6/*`
- Weekly governance digest with trend deltas and top failing rule families
- Monthly backlog planning input with risk-ranked remediation candidates
