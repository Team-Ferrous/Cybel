# Agent Workflows Framework Reference

## 1. Executive Summary

The Anvil repository implements a sophisticated **Agent Workflows Framework** responsible for translating high-level human objectives into actionable execution graphs. Operating primarily out of the `core/` directory, this framework selects contextual "Loops" based on input complexity, ranging from simple conversational chat interactions to fully autonomous, multi-agent enterprise deployments involving Planning, Execution, and Verification phases.

## 2. Architecture Overview

At the heart of Anvil is a dynamic execution router that evaluates the complexity of a user's instruction and assigns it to appropriate logical controllers.

### Core Routing Tiers
1.  **Conversational Loop (`Simple Chat`):** Fast, non-destructive interactions heavily constrained in operational scope. The agent is permitted to read codebase files and formulate responses without mutating state.
2.  **Basic Agentic Loop (`run_loop`):** A standard ReAct (Reason-Act) execution engine providing a full Think-Act cycle with iterative feedback.
3.  **EnhancedAgenticLoop (Enterprise):** Invoked for complex software engineering transformations.
4.  **AgentOrchestrator (Autonomous):** Used for massive objectives requiring parallelized `TaskUnits` delegated across multiple SubAgents.

## 3. The EnhancedAgenticLoop Protocol

The Enterprise Loop operates strictly via a rigid three-phase execution model, generating transparent Markdown artifacts throughout the cycle to adhere to the "Repository as System of Record" mandate.

### Phase 1: Planning
-   The Agent researches the objective strictly through `saguaro query` and semantic skeletal indexing.
-   It generates an `implementation_plan.md` outlining exact files to modify and the rationale behind those changes.
-   *Constraint:* Mutative actions are halted pending explicit User Review or Automated Approval.

### Phase 2: Execution
-   The Agent translates the approved plan into `saguaro agent patch` operations.
-   Changes are tested in Sandbox environments.
-   The Agent tracks its checklist line-by-line via continuous updates to its persistent `.md` artifacts.

### Phase 3: Verification
-   The Agent evaluates the applied patches against the `saguaro verify .` Sentinel engine, ensuring the changes satisfy Level 3 Governance.
-   Testing and static analysis checks are finalized.
-   Self-correction routines are invoked upon failure. Once fully resolved, a `walkthrough.md` is generated as an audit trail.

## 4. Multi-Agent Orchestration

### MasterAgent
The primary orchestrator interacting with the User Prompt. It breaks apart complex feature requests into smaller logic chunks (Task Graph nodes).

### SubAgents
Dedicated worker processes assigned to specific isolated execution paths. A typical configuration may involve:
-   **Planner SubAgent:** Investigates domain boundaries and architects the implementation path.
-   **Worker SubAgent:** Implements line-by-line file updates.
-   **Verifier SubAgent:** Aggressively attempts to break the implementation using strict QA mandates and integration testing logic.

## 5. Artifact & Memory Management

The framework avoids internal, invisible "state". All cognitive context is written out sequentially:
-   **Short-term:** Execution tokens residing within the contextual `<thinking>` boundaries.
-   **Long-term:** Persistent task progression files generated in `.anvil/` (e.g., `task.md`, `implementation_plan.md`, `walkthrough.md`), providing complete transparency into the agent’s logic.
-   **Chronological Memory:** Important architectural pivots are committed to `aiChangeLog/YYYY-MM-DD-<task>.md` via the Spec-Interview Pattern.
