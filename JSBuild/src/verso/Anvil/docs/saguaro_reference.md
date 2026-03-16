# Saguaro Reference Guide

## 1. Executive Summary

**Saguaro** is a "Quantum Codebase Operating System" (Q-COS) that provides deterministic, high-fidelity semantic understanding of large code repositories. Instead of relying on heuristic text matching (like `grep` or `find`), Saguaro builds a mathematically rigorous vector index of the codebase using C++ operations, which makes code search, refactoring analysis, and drift detection instantaneous and highly accurate.

## 2. Architecture Overview

Saguaro acts as the semantic layer bridging raw codebase files and the `MasterAgent`/`SubAgent` interaction models.

### Key Components:
- **Index Engine (`libsaguaro`):** A custom C++ native extension (`native saguaro core`) that tokenizes and mathematically embeds source code `.py`, `.cc`, and `.h` files.
- **Semantic Store (`.saguaro/`):** A persistent graph database and vector store that stores holographic embeddings for O(1) codebase traversal.
- **Command-Line Interface:** Exposes verbs like `query`, `skeleton`, `slice`, and `patch`.
- **SSAI (Standard Agent Interface):** A standardized adapter providing safe access to agents for modifying code iteratively within Sandboxes.

## 3. Core Protocols & Capabilities

### The Saguaro-First Protocol (SOP)
AI Agents operating in the Anvil ecosystem **must** use Saguaro native tools over standard Bash tools. This constraint exists to guarantee context efficiency: Saguaro delivers dependency-aware code slices requiring 10-100x fewer tokens than reading a full file.

*   **Discovery:** `saguaro query "concept"` finding code by meaning, replacing `grep`.
*   **Perception:** `saguaro agent skeleton file.py` and `saguaro agent slice MyClass.method --depth 2` rather than parsing full files.
*   **Action:** Code writes must run through the Sandbox (`saguaro agent patch -> verify -> commit`).

### The Sentinel Verification Engine
Run via `saguaro verify .`, Sentinel cross-references proposed changes against Level 3 Governance logic. It aggregates signals from:
1.  **Native Engine:** Custom regex and pattern matchers for headers and secrets.
2.  **Ruff:** Standardized Python linting.
3.  **Semantic Engine:** Quantum state drift checkers ensuring architectural integrity is not violated.

### Unwired Feature Discovery
Use `saguaro unwired` to detect isolated unreachable feature islands (connected internal clusters that are not reachable from runtime entrypoints). The analyzer performs graph traversal from detected roots, then classifies unreachable components as either `unwired_feature` or `unreachable_fragment` with compact summaries for token-efficient triage.

## 4. Advanced Agent Capabilities

Saguaro goes beyond search by providing dynamic lifecycle workflows:
- **The Chronicle (`saguaro chronicle`):** Automates semantic changelog tracking by analyzing state differences snapshot-to-snapshot, documenting *how the meaning* of code changed.
- **The Legislator (`saguaro legislation`):** Discovers ad-hoc coding patterns over time and drafts new rules to `.saguaro.rules.draft`, turning emergent behavior into formal technical roadmaps.
- **Change Intelligence:** Simulates future volatility states (`simulate volatility`) and regression risks for specific files.

## 5. Deployment & Execution
- **Indexing:** Performed manually (`saguaro index --path .`) or programmatically before a mission.
- **Failures:** `Inhomogeneous shape` errors signify vector drift or corruption resulting in a required `rm -rf .saguaro` reset.

## 6. Security Model
Changes evaluated against code are kept isolated via `saguaro agent patch`. A generated `sandbox_id` allows risk prediction (`saguaro agent impact <sandbox_id>`) and logic validation (`saguaro agent verify <sandbox_id>`) before a final commit back to the physical disk.
