## Sovereign Build Policy

- Keep build and runtime steps deterministic and reviewable.
- Prefer local or first-party artifacts before third-party execution paths.
- Do not weaken safety gates to pass a task.
- If a Saguaro operation fails, tag the failure and switch to `fallback_static_scan`.
- Record why fallback was used and what evidence was produced afterward.
