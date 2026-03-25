# AES External Baseline Review Process

Owner: Standards Governance
Cadence: Quarterly (or emergency review on upstream critical standard changes)

## Objective

Keep AES control mappings aligned with external baseline evolution without uncontrolled churn.

## Baselines In Scope

- OWASP ASVS
- CWE Top 25
- SLSA provenance guidance
- OpenTelemetry semantic conventions
- NIST controls referenced by AES policy

## Quarterly Workflow

1. Intake
- Gather upstream baseline release notes and version deltas.
- Record proposed updates in `standards/AES_SOURCE_BASELINES.md` draft section.

2. Impact Analysis
- Map changed controls to existing `standards/AES_RULES.json` entries.
- Identify controls requiring:
  - rule additions,
  - rule severity changes,
  - policy or evidence schema updates.

3. Migration Ticketing
- Open migration tickets for each impacted control family.
- Tag tickets by AAL criticality and expected rollout phase.

4. Validation
- Run:
  - `saguaro verify . --engines native,ruff,semantic,aes --format json`
  - `saguaro audit --format json`
- Compare pre/post compliance deltas and false-positive rates.

5. Governance Signoff
- Require explicit signoff from:
  - standards owner,
  - security owner,
  - runtime governance owner.
- No baseline version bump is allowed without recorded signoff.

6. Publication
- Merge updated `standards/AES_SOURCE_BASELINES.md` and migration plan.
- Publish release note for affected teams.

## Exit Criteria Per Review

- All changed baseline controls are mapped or explicitly deferred with risk acceptance.
- No high-assurance control loses enforceability during transition.
- Drift tickets are scheduled with owners and due dates.
