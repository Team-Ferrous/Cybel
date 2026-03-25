---
description: Execute a markdown roadmap to implement features and run tests
---
// turbo-all

1. **Analyze the Strategy**
   - Read the provided roadmap file (e.g., `HIGHNOON_INTEGRATION_ROADMAP.md` or similar).
   - Identify the user's high-level goal and the specific phases/tasks listed.

2. **Phase-by-Phase Execution**
   - Iterate through the roadmap items from top to bottom.
   - For each incomplete item (checked `[ ]`):
     - **Context**: Read relevant existing files to understand where new code fits.
     - **Implement**: Create or modify files as specified.
     - **Verify**: Run specific tests if listed in the roadmap (or general project tests).
     - **Mark Complete**: Update the roadmap file to mark the item as `[x]`.

3. **Final Verification**
   - After all tasks in a phase or the entire roadmap are done:
   - Run the full verification suite mentioned in the "Verification Plan" section.
   - Ensure `saguaro verify .` passes (or equivalent project lint/test check).

4. **Completion Report**
   - Summarize what was achieved.
   - List any skipped items or issues encountered.
