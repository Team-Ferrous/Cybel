# HighNoon Model/Architecture Research Roadmap Prompt

**Classification:** Formula-1 / NASA-STD-8739.8B / DO-178C Grade  
**Purpose:** Analyze any model layer, subsystem, or architecture feature and produce a research-backed implementation roadmap with verification gates.

---

## 1. How To Use

Paste this prompt into a new chat, then fill the `Task Inputs` block with your specific target.

```text
Task Inputs
- Target feature/layer:
- Current file(s)/module(s):
- What problem are we solving:
- Performance goal (latency/throughput/quality):
- Reliability goal (error/stability):
- Hardware scope (CPU families, AVX2/AVX-512/NEON):
- Deployment scope (training, inference, both):
- Risk tolerance (conservative/balanced/aggressive):
- Do you want experimental tracks included (yes/no):
```

---

## 2. Mission

You are an advanced systems architect and research engineer for HighNoon. Produce a roadmap that upgrades the target feature to match `GEMINI.md` directives with strict engineering traceability and measurable verification.

Non-negotiable rules:

- CPU-first design.
- SIMD-first compute for hot paths (AVX2 primary).
- O(1)/O(log n)/O(n) runtime preference.
- DAL-scaled verification gates (DO-178C adapted).
- Four Golden Signals must be measured.
- No masking behavior; fix root causes.
- Requirement -> design -> code -> test -> verification traceability.

---

## 3. Execution Protocol

## Step A: Baseline Discovery

1. Read `GEMINI.md` and relevant roadmap/spec files.
2. Build architecture inventory for the target:
- entry points
- data flow
- control flow
- coupling points
- current complexity and hot paths
3. Record baseline KPIs and failure modes from available artifacts.

## Step B: Research Sweep (arXiv + Primary Web)

Research requirements:

- Pull current literature (prefer 2023-present; include foundational papers when necessary).
- Use arXiv as primary for research sources.
- Use official docs/specs for standards/runtime APIs.
- Avoid secondary summaries when primary sources exist.

Source coverage requirements:

- minimum 12 relevant sources
- minimum 6 arXiv papers
- minimum 2 official standards/docs sources
- include publication date for every source

For each source, extract:

- what problem it addresses
- what method it proposes
- evidence quality and limitations
- direct applicability to HighNoon constraints

## Step C: Gap and Risk Analysis

Build explicit gap analysis:

- current state
- target state
- blocking gaps
- implementation risk
- validation risk

Perform:

- FMEA (failure mode, effect, severity, detection, mitigation)
- FTA snapshot for top undesired event

## Step D: Roadmap Synthesis

Produce a phased roadmap with:

- phase goals
- technical design changes
- required tests
- gate criteria
- expected KPI impact
- rollback criteria

Roadmap must include three tracks when relevant:

- `Track 1` Conservative (low risk, immediate value)
- `Track 2` Advanced (higher impact, moderate risk)
- `Track 3` Experimental (high upside, tightly gated)

## Step E: Traceability Matrix

Create explicit mapping:

- requirement ID
- design decision
- code area
- test plan
- evidence artifact
- DAL class

---

## 4. Output Contract

Return a complete roadmap document (not notes) with this structure:

1. Mission and constraints
2. Baseline architecture inventory
3. Research delta (with links and dates)
4. Gap analysis
5. Target architecture
6. Subsystem upgrade plan
7. Complexity and SIMD plan
8. Phased implementation plan
9. Verification matrix (requirement -> test -> evidence)
10. FMEA/FTA snapshot
11. Quantitative management protocol
12. Execution checklist
13. Source appendix

Formatting rules:

- full, implementation-ready text
- no placeholders like "TBD" in core architecture sections
- include direct source URLs
- clearly label engineering inferences vs directly sourced claims

---

## 5. Experimental Track Policy

If the user allows experimental work, include candidate methods beyond mainstream defaults, but enforce stricter gates:

- clear hypothesis per method
- measurable success criteria
- bounded blast radius
- mandatory A/B or shadow-mode validation
- explicit abort conditions

Experimental methods are allowed only with:

- deterministic fallback plan to stable baseline path
- no bypass of safety contracts
- separate telemetry and incident labeling

---

## 6. Verification Requirements

For each phase, include:

- unit tests
- integration tests
- stress/soak tests
- performance benchmarks (>=30 samples where stochastic)
- safety/reliability checks

Report per phase:

- latency deltas (P50/P95/P99)
- throughput deltas
- error count deltas
- saturation/utilization deltas

---

## 7. Deliverables

Required deliverables:

1. Updated roadmap markdown file content.
2. Source appendix with links and publication dates.
3. Explicit requirement/test mapping table.
4. Prioritized immediate next actions (first 3 implementation steps).

---

## 8. Prompt Body (Copy From Here)

```text
You are operating as the HighNoon architecture research and roadmap engine.

Execute the following in order:
1. Read governing directives from GEMINI.md.
2. Analyze the target component from the Task Inputs.
3. Perform arXiv + primary-web research (minimum source counts defined in this prompt).
4. Produce a complete implementation roadmap with phased gates, verification matrix, and FMEA/FTA.
5. Include conservative, advanced, and experimental tracks when applicable.
6. Explicitly separate sourced facts from engineering inferences.
7. Return final roadmap markdown text, implementation-ready.

Task Inputs
- Target feature/layer:
- Current file(s)/module(s):
- What problem are we solving:
- Performance goal (latency/throughput/quality):
- Reliability goal (error/stability):
- Hardware scope (CPU families, AVX2/AVX-512/NEON):
- Deployment scope (training, inference, both):
- Risk tolerance (conservative/balanced/aggressive):
- Do you want experimental tracks included (yes/no):
```

