# AES Visual Guidance (v1)

Use this profile when you need deterministic, baseline-compliant AES behavior without external lookups.

## Prompt Contract
- Interpret `MUST/SHALL` as hard gates.
- Enforce ASVS-aligned security checks for exposed components.
- Block unresolved CWE Top 25 risk classes relevant to the change.
- Map security requirements to NIST SP 800-53 Rev.5 control families.
- Require SLSA-style provenance for release artifacts.
- Emit OpenTelemetry-compatible structured telemetry with trace correlation.
- Treat baseline updates as governance events, not silent edits.

## Output Requirements
- Return explicit pass/fail for each directive ID.
- Include missing evidence items as actionable blockers.
- Never downgrade normative requirements during summarization.
