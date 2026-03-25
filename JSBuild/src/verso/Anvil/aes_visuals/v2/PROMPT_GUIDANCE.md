# AES Visual Guidance (v2)

Use this profile for high-assurance tasks where AES outputs must stay auditable without external context.

## Prompt Contract
- Fail closed on normative ambiguity: preserve RFC 2119/8174 semantics.
- Enforce assurance-class selection before implementation.
- Apply ASVS + CWE controls with explicit drift handling for upstream updates.
- Require NIST family mapping for high-risk requirements.
- Use CERT/MISRA-aligned constraints for native critical code.
- Require SLSA-style provenance and attestations for critical artifacts.
- Enforce OpenTelemetry semantic consistency and cross-signal correlation.
- Require evidence-backed claims for quantum/noise/numerical stability assertions.

## Output Requirements
- Emit directive-by-directive verdicts (`pass`, `fail`, `waived`).
- Include missing evidence and exact remediation action.
- Escalate unresolved high-assurance failures instead of auto-resolving.
