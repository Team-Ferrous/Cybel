# Severity Decision Tree

1. Does the file change silent-corruption, cryptographic, SIMD, aligned-memory, or irreversible execution logic?
   - Yes: `AAL-0`
2. Does the file change training loops, gradient flow, quantum circuits, or simulation/integrator cores?
   - Yes: `AAL-1`
3. Does the file change orchestration, config, CLI, logging, or batch workflow control?
   - Yes: `AAL-2`
4. Otherwise:
   - `AAL-3`

Strictest impacted file wins for a changeset.
