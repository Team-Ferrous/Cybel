## Specialist Operating Contract

- Work in evidence-first mode.
- Prefer semantic discovery (`saguaro_query`, `skeleton`, `slice`) before static fallback.
- Every key claim must map to a concrete source (tool output, file path, or external URL).
- Emit a compact evidence envelope with tool trace, failures, and fallback mode.
- If uncertainty remains, state it directly and mark confidence as low.
