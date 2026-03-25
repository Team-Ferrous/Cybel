# Anvil Engineering Standard (AES) v2.0

## Purpose

AES defines the default engineering, verification, security, and governance rules that Anvil agents must follow. Normative language uses RFC 2119 terms.

## 1. Severity Framework

| AAL | Meaning | Typical Scope | Required Posture |
|---|---|---|---|
| AAL-0 | Catastrophic impact | crypto, SIMD kernels, silent corruption paths | fail closed, independent review, full evidence bundle |
| AAL-1 | Critical impact | ML training, quantum circuits, simulation kernels | strict verification, traceability, review signoff |
| AAL-2 | Major impact | config, CLI, orchestration, tooling | regression tests, structured review |
| AAL-3 | Limited impact | docs, notebooks, non-critical text | review and hygiene checks |

## 2. Core Architecture Mandates

- `AES-ARCH-1`: Blocking enforcement MUST happen in runtime gates, not prompt prose alone.
- `AES-ARCH-2`: All code-modifying workflows MUST pass through a unified hook bus.
- `AES-ARCH-3`: Legacy orchestration paths MUST be compatibility adapters only.
- `AES-ARCH-4`: Rule loading MUST be machine-readable and deterministic.
- `AES-VIS-1`: Visual governance packs MUST include model-readable v1/v2 directive manifests.
- `AES-VIS-2`: Visual governance manifests MUST preserve schema-valid typed directive fields.
- `AES-PRM-1`: Prompt contracts MUST expose required governance keys and freshness-safe references.

## 3. Traceability and Evidence

- `AES-TR-1`: High-assurance work MUST preserve requirement -> design -> code -> test -> verification links.
- `AES-TR-2`: AAL-0/1 completion MUST include a valid evidence bundle.
- `AES-TR-3`: Waivers MUST be bounded, time-limited, and carry compensating controls.

## 4. Cleanroom and Defect Prevention

- `AES-CR-1`: Root-cause fixes MUST be preferred over masking behavior.
- `AES-CR-2`: Bare exception swallowing MUST NOT be used.
- `AES-CR-3`: Fallback paths MUST NOT silently bypass required verification.

## 5. Complexity and Error Contracts

- `AES-CPLX-1`: Risk-bearing functions SHOULD stay within bounded complexity.
- `AES-ERR-1`: Public interfaces SHOULD declare error contracts and failure modes.
- `AES-ERR-2`: Silent `None` returns in exception paths SHOULD be treated as suspect.

## 6. Security and Supply Chain

- `AES-SEC-1`: Hardcoded secrets MUST NOT be committed.
- `AES-SEC-2`: Dynamic evaluation (`eval`, `exec`) MUST NOT be used outside explicit sandboxed contexts.
- `AES-SEC-3`: Exposed service surfaces MUST carry threat-model/misuse-case evidence and provenance markers.
- `AES-SUP-1`: Dependency baselines MUST be explicit and auditable.
- `AES-SUP-2`: Pinned dependencies MUST include integrity hash metadata.
- `AES-SUP-3`: Release/build workflows MUST include provenance or attestation markers.
- `AES-SUP-4`: Release/build workflows MUST include artifact-signing controls.

## 7. Python and C++ Standards

- `AES-PY-1`: Public Python APIs SHOULD carry type hints.
- `AES-PY-2`: Imports SHOULD be deterministic and ordered.
- `AES-PY-3`: Mutable default arguments SHOULD NOT be used.
- `AES-PY-4`: Dynamic execution MUST NOT be used outside explicitly sandboxed environments.
- `AES-PY-5`: Wildcard imports MUST NOT be used in governed Python code.
- `AES-PY-6`: File operations MUST use context managers in governed Python code.
- `AES-CPP-1`: Native high-performance paths MUST document SIMD/OpenMP assumptions.
- `AES-CPP-2`: Scalar or reference validation paths MUST exist for high-risk kernels.
- `AES-CPP-3`: Raw allocation patterns MUST be governed by RAII ownership conventions.
- `AES-CPP-4`: Error-bearing native return paths SHOULD use `[[nodiscard]]` contracts.
- `AES-CPP-5`: C-style casts SHOULD be eliminated in high-assurance native code paths.

## 8. Domain-Specific Rules

### ML
- `AES-ML-1`: Training loops MUST check for non-finite gradients before optimizer updates.
- `AES-ML-2`: Unstable numerics MUST be guarded with stable formulations.
- `AES-ML-4`: Data ingest boundaries SHOULD validate schema, shape, and dtype.
- `AES-ML-5`: Training configuration SHOULD record seeds and versioning.

### Quantum
- `AES-QC-2`: Parameterized circuits SHOULD avoid magic-angle literals in gate construction.
- `AES-QC-3`: Noisy simulations SHOULD specify a noise model explicitly.
- `AES-QC-4`: Backend execution SHOULD follow explicit compilation/transpilation.
- `AES-QC-5`: Shot counts SHOULD meet minimum confidence thresholds.

### Physics
- `AES-PHYS-1`: Simulation cores SHOULD monitor conservation-critical quantities.
- `AES-PHYS-3`: Hamiltonian systems SHOULD avoid non-symplectic integrator markers.

### HPC
- `AES-HPC-1`: Parallel constructs SHOULD make scheduling and reduction assumptions explicit.
- `AES-HPC-2`: Alignment-sensitive kernels SHOULD declare alignment intent in code or design notes.
- `AES-HPC-3`: OpenMP regions SHOULD declare explicit data-sharing or scheduling clauses.
- `AES-HPC-4`: SIMD kernels SHOULD keep a scalar or reference oracle.

## 9. Agent Governance

- `AES-AG-1`: Agents MUST surface their assurance context when performing non-trivial implementation work.
- `AES-AG-2`: Prompt guidance is advisory; runtime policy is authoritative.
- `AES-AG-3`: Hook receipts and verification evidence SHOULD be retained for later audit.

## 10. Observability

- `AES-OBS-1`: Golden-signal relevant metrics SHOULD be captured before and after major changes.
- `AES-OBS-2`: Verification output SHOULD identify blocking and advisory defects separately.

## 11. Review and Closure

- `AES-REV-1`: AAL-0/1 work MUST have independent review.
- `AES-REV-2`: Items lacking code, tests, evidence, or roadmap update are `PARTIAL`, not `DONE`.
- `AES-REV-3`: Baseline version changes MUST be explicit and auditable.

## Runtime Projection

`standards/AES_RULES.json` is the authoritative machine-readable projection of this standard. Every rule ID listed above is expected to exist in the registry with explicit severity, engine ownership, and closure metadata so later phases can enforce the same rule set mechanically.
