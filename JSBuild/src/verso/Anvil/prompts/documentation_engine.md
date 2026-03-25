# HighNoon Documentation Engine — Code Analysis & Doc Maintenance Prompt

**Classification:** Formula-1 / NASA-STD-8739.8B / DO-178C Grade
**How To Use:** Paste this into a NEW chat when you need to analyze the codebase and create, update, or maintain any documentation. Specify the documentation target at the bottom.

---

## Standards Anchoring (Read First)

Documentation in this project is a first-class engineering artifact — not an afterthought. Per NASA NPR 7150.2D and DO-178C, documentation is a formal deliverable with traceability requirements.

| Standard | What It Governs Here |
|---|---|
| **NASA-STD-8739.8B** | Software assurance — documentation is a verifiable artifact, not commentary |
| **NASA NPR 7150.2D** | Bidirectional traceability — every doc traces to code, every code traces to docs |
| **DO-178C** (Boeing/Airbus) | DAL-D minimum classification for all documentation; zero stale references |
| **FIA F1 Telemetry** | Compounding knowledge — documentation compounds across sessions like telemetry data |
| **Google SRE** | Runbook accuracy — if documentation says it works a certain way, it MUST work that way |
| **CMMI Level 5** (Raytheon) | Quantitative documentation — doc coverage metrics, not vibes |
| **IBM Cleanroom** | Specification-first — documentation defines the contract, code implements it |
| **JEDEC** (Samsung/Micron) | Spec-driven closure — no undocumented public interfaces |

---

## Engineering Directives (NON-NEGOTIABLE)

Formula 1 / NASA level thinking, planning, decision making, analysis, and documentation. Every piece of documentation must be accurate, complete, and traceable to the actual codebase. No guessing, no stale content, no aspirational descriptions of features that don't exist.

**You are STRICTLY FORBIDDEN from:**
- Documenting features that do not exist in the current codebase — per DO-178C: zero dead references
- Using placeholder descriptions like `TODO: describe this` or `[insert details]` — write COMPLETE documentation
- Copying docstrings verbatim without verifying they match the actual implementation — per NASA NPR 7150.2D: traceability means verified traceability
- Describing behavior that contradicts the code — per Google SRE: if the doc says X, the code must do X
- Leaving stale parameter lists, return types, or function signatures — per JEDEC: spec must match implementation
- Writing documentation without reading the actual source code first — per IBM Cleanroom: specification requires understanding
- Guessing at internal behavior — per NASA FTA: trace to exact file:line, not "this probably does..."
- Documenting deprecated or removed components without marking them as such

---

## Documentation Classification (DO-178C Aligned)

Every documentation task must be classified. Documentation quality scales with the criticality of what it describes.

| Priority | DO-178C DAL | What It Covers | Required Rigor |
|---|---|---|---|
| **P0** | DAL-A (Catastrophic) | API docs for core model ops, training engine, C++ kernel interfaces | Every parameter, return type, exception, and side effect documented. Code-verified. |
| **P1** | DAL-B (Hazardous) | User guides, training workflows, configuration reference, curriculum learning | Step-by-step accuracy. All code examples tested and working. |
| **P2** | DAL-C (Major) | Architecture docs, design decisions, module overviews | Accurate high-level descriptions with correct cross-references. |
| **P3** | DAL-D (Minor) | READMEs, changelogs, contribution guides, cosmetic doc fixes | Review for accuracy. No broken links. |

---

## Documentation Scope

### What Lives Where

| Doc Type | Location | Format |
|---|---|---|
| **API Reference** | `docs/api/` | One `.md` per module — models, training, inference, cli, config, tokenization |
| **User Guides** | `docs/guides/` | Step-by-step workflow guides — training, curriculum, agent tools, HPO, WebUI, hardware daemon |
| **Architecture / Design** | `docs/architecture/` | Distributed training, cluster setup, architecture overviews |
| **Operations Runbooks** | `docs/operations/` | Install, deployment, and runtime operations procedures |
| **Compliance** | `docs/compliance/` | ITAR, export controls, regulatory |
| **Prompt Library** | `docs/prompts/` | Reusable analysis/implementation/documentation prompt packs |
| **Workflow SOPs** | `docs/workflows/` | Execution workflows and phase-gated SOPs |
| **Reference Materials** | `docs/reference/` | PDFs and static reference artifacts |
| **Getting Started** | `docs/getting-started.md` | Installation, quick start, first model |
| **Index / Navigation** | `docs/index.md` | Landing page with cross-links to all section indexes |
| **Release Notes** | `docs/RELEASE_NOTES.md` | Version history, breaking changes, migration notes |
| **Inline Docstrings** | Source files (`.py`, `.cc`, `.h`) | Function/class/module-level docstrings |
| **Specs** | `specs/` | Feature specifications per NASA IV&V |
| **Change Logs** | `aiChangeLog/` | Session-level engineering logs per F1 telemetry protocol |

---

## Documentation Analysis & Generation Workflow

```
┌──────────────────────────────────────────────────────────────┐
│  PHASE 1: CODE ANALYSIS                                      │
│  ─────────────────────                                       │
│  1. READ the target source files — understand every function,│
│     class, parameter, return type, and side effect            │
│  2. READ existing docs for the target area (if any)          │
│  3. DIFF code vs docs — identify gaps, stale content,        │
│     incorrect descriptions, missing parameters               │
│  4. MAP the public API surface — what MUST be documented     │
│  5. TRACE dependencies — what does this module import/use?   │
│  6. IDENTIFY examples — what usage patterns exist in tests   │
│     or other modules?                                         │
├──────────────────────────────────────────────────────────────┤
│  PHASE 2: GAP ANALYSIS                                       │
│  ─────────────────────                                       │
│  7. LIST all undocumented public functions/classes            │
│  8. LIST all stale docs (describe behavior that no longer    │
│     matches code)                                             │
│  9. LIST all missing cross-references                         │
│  10. LIST broken links or references to removed files         │
│  11. CLASSIFY each gap by priority (P0–P3)                   │
├──────────────────────────────────────────────────────────────┤
│  PHASE 3: DOCUMENTATION GENERATION / UPDATE                  │
│  ──────────────────────────────────────────                   │
│  12. WRITE or UPDATE documentation — COMPLETE output         │
│  13. VERIFY every code example compiles/runs                  │
│  14. VERIFY every parameter list matches the actual signature│
│  15. VERIFY every cross-reference link targets a real file    │
│  16. UPDATE docs/index.md navigation if new docs created     │
├──────────────────────────────────────────────────────────────┤
│  PHASE 4: DOCSTRING MAINTENANCE                              │
│  ──────────────────────────────                               │
│  17. UPDATE inline docstrings in source files if stale       │
│  18. ADD docstrings to undocumented public interfaces        │
│  19. ENSURE docstring format is consistent (Google style     │
│      for Python, Doxygen for C++)                             │
├──────────────────────────────────────────────────────────────┤
│  PHASE 5: VERIFICATION                                       │
│  ─────────────────────                                       │
│  20. RUN link checker across all docs                         │
│  21. RECORD documentation coverage metrics (Chronicle)       │
│  22. REPORT summary of changes made                          │
└──────────────────────────────────────────────────────────────┘
```

---

## Mandatory Protocols (Every Documentation Task)

### 1. Chronicle Protocol (F1 Telemetry)
Open or create `aiChangeLog/YYYY-MM-DD-documentation.md`. Record:
- Which docs were created, updated, or deleted
- Documentation coverage before and after (count of documented vs undocumented public interfaces)
- List of stale content that was corrected
- List of new cross-references added

### 2. Code-First Analysis (IBM Cleanroom)
**NEVER write documentation from memory or assumptions.** Always:
- Open the actual source file and read the current implementation
- Verify function signatures, parameter types, default values, and return types against the live code
- Check for recent changes in `git log` that may have altered behavior
- If a docstring says one thing and the code does another, the CODE is the source of truth — update the doc

### 3. Traceability Matrix (DO-178C + NASA NPR 7150.2D)
Maintain a mental (or explicit) traceability map:

```
Source Code (file:line) ←→ Docstring ←→ API Reference Doc ←→ User Guide ←→ Index Link
```

Every public interface must have a complete chain. If any link is broken, fix it.

### 4. Red Team Self-Check
Before finalizing documentation:
- **Accuracy:** Does every statement match the code? Verify with `grep` or `view_code_item`
- **Completeness:** Are all public parameters, return types, and exceptions documented?
- **Freshness:** Are there references to removed features, old file paths, or deprecated APIs?
- **Links:** Do all cross-reference links resolve to real files?
- **Examples:** Do all code examples reflect current API signatures?
- **Consistency:** Is terminology consistent across all docs? (e.g., don't call it "QULS" in one place and "Quantum Uncertainty" in another without defining the relationship)

### 5. Documentation Format Standards

**Python Docstrings (Google Style):**
```python
def function_name(param1: type, param2: type = default) -> ReturnType:
    """Brief one-line summary.

    Extended description if needed. Explain the WHY,
    not just the WHAT.

    Args:
        param1: Description of param1.
        param2: Description of param2. Defaults to X.

    Returns:
        Description of what is returned.

    Raises:
        ValueError: When param1 is invalid because...
        RuntimeError: When the C++ op fails because...

    Example:
        >>> result = function_name("input", param2=42)
        >>> print(result)
    """
```

**C++ Docstrings (Doxygen):**
```cpp
/**
 * @brief Brief one-line summary.
 *
 * Extended description. Explain SIMD requirements,
 * alignment constraints, and thread safety.
 *
 * @param param1 Description. Must be alignas(64).
 * @param param2 Description.
 * @return Description of return value.
 * @throws std::runtime_error When...
 *
 * @note This function requires AVX2. Falls back to scalar on unsupported hardware.
 * @warning Not thread-safe without external synchronization.
 *
 * @par Complexity
 * O(n) where n is the batch size.
 *
 * @par SIMD
 * AVX2 primary, AVX-512 secondary, ARM NEON tertiary.
 */
```

**Markdown API Reference:**
```markdown
## `function_name(param1, param2=default)`

Brief description.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `param1` | `type` | Required | Description |
| `param2` | `type` | `default` | Description |

### Returns

| Type | Description |
|---|---|
| `ReturnType` | Description |

### Raises

| Exception | Condition |
|---|---|
| `ValueError` | When... |

### Example

\```python
result = function_name("input", param2=42)
\```
```

### 6. Cross-Reference Protocol
When creating or updating docs:
- Every new doc page must be linked from `docs/index.md`
- Every API function referenced in a guide must link to its API reference page
- Every guide referenced in an API doc must link back
- Use relative links within `docs/` — e.g., `[Training Guide](../guides/training.md)`

---

## Subagent Execution Model (DO-178C Independence)

For large documentation tasks, spawn subagents with SPECIFIC scopes:

- **Analysis subagent:** Reads source code, identifies public API surface, lists gaps — does NOT write docs
- **Writing subagent:** Takes the gap analysis and writes/updates documentation — verified against code
- **Verification subagent:** Checks all links, verifies code examples, validates parameter accuracy

Per DO-178C: the analyzer and the writer should be independent. The verifier checks both.

Each subagent deliverable must include:
- **Files analyzed:** List of source files read
- **Gaps found / Gaps resolved:** Count with specifics
- **Links verified:** Count of valid / broken links
- **Coverage delta:** Before/after documentation coverage

---

## Codebase Focus

- What we have enabled in `config.py` and used in `block_factory.py` are the primary documentation targets
- Document the ACTUAL architecture — not aspirational features
- C++ ops like `quantumlinear` need clear API documentation with SIMD/alignment requirements
- Training engine (`enterprise_training_engine_audit.py`) is the most critical user-facing system to document
- The audit system and its artifacts should have clear operational documentation

---

## Documentation Target

**Document the following:**

[Specify the documentation task. Examples:
- "Analyze all public functions in `enterprise_training_engine_audit.py` and create/update `docs/api/training.md` with complete API reference"
- "Audit all docs in `docs/guides/` for stale content — update any references that no longer match the code"
- "Create a new architecture overview doc covering the MoE routing, QHD, and TimeCrystal subsystems"
- "Update `docs/index.md` to reflect all current documentation and fix broken links"
- "Add Google-style docstrings to all public functions in `block_factory.py`"
- "Full documentation coverage pass — analyze every module and create missing API reference docs"
- "Update `docs/getting-started.md` to reflect the current installation process and API"]
