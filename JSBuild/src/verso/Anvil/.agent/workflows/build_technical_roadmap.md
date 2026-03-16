---
description: Build a detailed technical roadmap using the 6-stage methodology
---
// turbo-all

1. **Stage 1 — Deep Codebase Reconnaissance**
   - Map the top-level directory structure.
   - Find all files related to the topic by name pattern and concept search.
   - Outline every relevant file (classes, methods, line counts) — do NOT read full files yet.
   - Deep-read only the critical files: base classes, orchestrators, state stores, configuration.
   - Identify existing test files for affected subsystems.

2. **Stage 2 — External Research & Evidence Gathering**
   - Run 2-4 targeted web searches covering:
     - Architecture patterns relevant to the task.
     - Academic/arxiv papers on the domain.
     - Tooling, benchmarks, and frameworks.
     - (Optional) Competitor or prior art analysis.
   - Extract concrete patterns, citations, and tool names.
   - Synthesize into a gap analysis: what the codebase lacks vs. state-of-the-art.

3. **Stage 3 — Current State Assessment**
   - Build a markdown table: `| Subsystem | Exists | Maturity | Key Gaps |`.
   - Be honest: use `Solid / Good / Basic / Stub` for maturity.
   - Gaps must be specific and actionable, not vague.

4. **Stage 4 — Architecture Overview Diagram**
   - Produce a Mermaid `graph TD` or `graph LR` diagram.
   - Show 5-15 nodes covering the high-level workflow or component layout.
   - Label edges with conditions, especially for loops and branches.

5. **Stage 5 — Write the Roadmap Phases**
   - For each phase, specify:
     - Phase header with context.
     - Every file change with `[NEW|MODIFY|REWRITE|DELETE]` action + clickable file link.
     - Code sketches for non-obvious designs (signatures only, not full implementations).
     - Database/data model changes enumerated explicitly.
   - End with New Files and Modified Files summary tables.

6. **Stage 6 — Verification Plan & Implementation Order**
   - List exact `pytest` commands and new test files with coverage descriptions.
   - Specify manual verification steps.
   - Provide a dependency-sorted implementation order.

7. **Quality Check**
   - Verify: assessment table complete, diagram present, all changes trace to gaps,
     file paths are absolute, code sketches present for complex designs, verification
     plan has exact commands, implementation order is dependency-sorted.
