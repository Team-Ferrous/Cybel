# AES Condensed

AES is always active.

Severity:
- `AAL-0`: catastrophic paths, fail closed, independent review, full evidence
- `AAL-1`: critical paths, strict verification, traceability, review signoff
- `AAL-2`: major paths, regression-tested with evidence
- `AAL-3`: low-risk paths, hygiene-reviewed

Hard mandates:
- runtime gates override prompt prose
- preserve `requirement -> design -> code -> test -> verification`
- use root-cause fixes, not masking patches
- no bare exception swallowing
- no silent fallback around required verification
- no `eval` or `exec` outside explicit sandboxing
- high-assurance work missing artifacts is blocking, not advisory

Required artifacts for `AAL-0`/`AAL-1`:
- traceability record
- evidence bundle
- review signoff
- valid waiver if deviating from policy

Domain reminders:
- ML: finite gradient checks, stable numerics, validated ingest, reproducibility manifest
- Quantum: parameterized gates, transpilation before backend run, explicit noise/shot assumptions
- Physics: conservation monitoring, symplectic integrator discipline for Hamiltonian systems
- HPC: explicit OpenMP clauses, alignment contracts, scalar/reference oracles

Completion rule:
- code present
- tests pass
- evidence present
- roadmap updated
- otherwise mark `PARTIAL`
