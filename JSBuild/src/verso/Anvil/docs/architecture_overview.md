# Anvil & HighNoon Architecture Overview

## 1. Executive Summary

The Anvil repository houses a sophisticated, multi-agent AI coding assistant designed for enterprise-level operations. It is governed by principles drawn from high-reliability organizations (e.g., NASA-STD-8739.8B, DO-178C). The system is built upon three foundational pillars:
1.  **Anvil (Granite Agent)**: A multi-agent orchestration framework capable of planning, executing, and verifying complex software engineering tasks.
2.  **Saguaro (Quantum Codebase OS)**: A semantic code intelligence engine that provides deterministic, high-fidelity context for autonomous agents without relying on error-prone heuristic searches.
3.  **HighNoon**: A strictly CPU-first, SIMD-optimized architectural constraint model designed for maximum determinism and transparency, ensuring readiness for future quantum-computational paradigms.

This document serves as the top-level architectural reference for the system.

## 2. Architecture Overview

At the highest level (C4 Context), Anvil operates as a Developer Assistant interface, integrating deeply with the local codebase while leveraging specialized agents and C++ optimized inference/retrieval subsystems.

### C4 Container Level

-   **Granite REPL (`cli/repl.py`):** The primary user interface. A read-eval-print loop providing seamless chat, semantic execution, and mission initiation.
-   **Agent Core (`core/`):** The brain of the system, housing the state machines and interaction loops:
    -   *Conversational Loop*: For simple QA using minimal context.
    -   *EnhancedAgenticLoop*: For multi-step tasks (Planning -> Execution -> Verification).
    -   *Orchestrator*: Manages distributed multi-agent operations.
-   **Saguaro Engine (`saguaro/`):** Native in-tree indexing and semantic query engine exposed via Python API and CLI entry points.
-   **HighNoon Compute Layer (`core/native/` & `saguaro/native/`):** Deeply optimized, SIMD-first (AVX2+) mathematical primitives enforcing O(1)/O(n) complexity mandates.

## 3. Design Decisions

### CPU-First & SIMD Compute Mandate
**Decision:** All compute-heavy code must be SIMD-optimized C++ running primarily on CPUs, rather than relying exclusively on black-box GPUs.
**Rationale:** Enables strict deterministic execution paths, rigorous auditability (DAL-A/B compliance), and establishes a classical-simulation baseline compatible with future quantum workflows.

### Repository as System of Record
**Decision:** Agents operate purely on in-repo artifacts. Unwritten context does not exist.
**Rationale:** Forces explicit state tracking (e.g., `task.md`, `implementation_plan.md`) ensuring human-in-the-loop oversight and perfectly reproducible workflows.

### Saguaro-First Protocol
**Decision:** Agents must prioritize `saguaro query` and semantic slicing over primitive tools like `grep_search`.
**Rationale:** Maximizes token efficiency (10-100x fewer tokens) and prevents catastrophic context rot by feeding models structured, dependency-aware code slices.

## 4. Core Components

### 4.1 MasterAgent & SubAgents
Coordinates execution. `MasterAgent` maintains high-level objectives and delegates isolated tasks to specialized `SubAgent`s (e.g., Planner, Worker, Verifier) communicating via the Task Graph.

### 4.2 Saguaro Semantic Indexer
Scans `.py`, `.cc`, and `.h` files, using C++ quantum operations for tokenization and embedding. It serializes a persistent semantic graph in the `.saguaro/` directory, maintaining O(1) retrieval times.

### 4.3 Sentinel Verification Engine
Integrated into Saguaro (`saguaro verify .`), this subsystem runs Native regex checks, Ruff linting, and Semantic drift detection to enforce Level 3 Governance before any PR is generated.

## 5. Security Model

Anvil operates under enterprise constraints (OWASP Top 10 + SOC2 context):
-   **Sandboxed Modifications:** Agents test changes in a virtual patch format via `saguaro agent patch` before committing to disk.
-   **Verification Hierarchy:** Core SIMD kernels follow DAL-A rigor; Python control loops follow DAL-B.
-   **Traceability:** All actions map back to explicit user requests or scheduled tasks, preventing rogue execution loops.

## 6. Deployment & Performance Characteristics

-   **Performance Contract:** The system strictly monitors Golden Signals (`step_time_avg_s`, TTFT, `tokens_per_sec_avg`). Memory accesses use `alignas(64)` cache alignments to prevent thread false-sharing.
-   **Deployment:** Packaged via `setup.py` and native CMake targets. Requires a strict `pip install -e .` which triggers the native C++ compilation (`build_native.sh`).

## 7. Appendices

-   **Workflow Policies:** See `docs/workflows/AGENT_AUDIT_ANALYSIS_WORKFLOW.md`
-   **Governance:** Governed by `GEMINI.md` and standard NASA/DO-178C engineering principles.
