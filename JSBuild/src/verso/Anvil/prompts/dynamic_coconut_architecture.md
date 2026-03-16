# Dynamic COCONUT & Subagent Architecture Integration Prompt

**Role**: Principal AI/ML Systems Architect

**Objective**: 
Conduct a deep-dive investigation into the Anvil codebase using Saguaro to design a comprehensive technical roadmap for integrating **Dynamic COCONUT (Continuous Thought)** and **Content-Aware Dynamic Subagent Orchestration**.

## Project Context
Anvil currently utilizes a MasterAgent/SubAgent architecture. We want to upgrade this architecture to heavily leverage COCONUT (Continuous Thought) space, giving the agents "free" token space for reasoning while preserving multi-agent orchestration. We must move away from static text-based Chain-of-Thought (CoT) and static subagent allocation to a highly dynamic, content-aware system.

## Core Directives

1.  **Native Tool Calling inside COCONUT Space**:
    *   **Goal**: Enable agents to perform tool calls (e.g., Saguaro queries, file edits, sandbox testing) directly from within the continuous thought loop.
    *   **Challenge**: Tool execution results are discrete text tokens or JSON. You must design a mechanism to seamlessly project these discrete tool results back into the continuous latent reasoning space without breaking the agent's structural "train of thought".
    
2.  **Dynamic, Content-Aware Subagent Allocation**:
    *   **Goal**: The MasterAgent should dynamically instantiate sequential or parallel subagents based on the real-time complexity of the task at hand, rather than relying on fixed or hardcoded agent chains.
    *   **Sub-Goal**: Subagents should be spun up, handed a localized continuous thought vector (context proxy), and retired once their specific sub-task is complete, allowing for dynamic scale-out on complex tasks and rapid execution on simple tasks.

3.  **Dynamic COCONUT Depth (Adaptive Compute)**:
    *   **Goal**: The amount of continuous thought (the number of latent steps/length of the COCONUT stream) must adapt dynamically to the actual content being processed.
    *   **Mechanism**: Easy code/tasks = shallow COCONUT depth. Complex, highly-entangled logic = deep COCONUT depth. Investigate how to implement an entropy, complexity, or confidence-based heuristic to govern this allocation dynamically.

## Mandatory Saguaro Tooling Protocol

You are strictly required to use the **Saguaro-First Protocol** (Q-COS) during your analysis:
*   Do NOT use `grep_search`, `find_by_name`, or `view_file` to naively read large swathes of code. 
*   Use `saguaro query` to find existing implementations of:
    *   `SubAgent` and `MasterAgent` orchestration.
    *   Inference loops and LLM generation (e.g., `llmfit`, tokenizers).
    *   Tool calling dispatchers.
*   Use `saguaro agent skeleton` and `saguaro agent slice` to deeply review the required classes.
*   Use `saguaro refactor plan` to assess the impact of adding dynamic COCONUT state propagation between agents.

## Roadmap Deliverable Format

Your final output must be a markdown file named `ROADMAP_DYNAMIC_COCONUT.md` in the project root. The document MUST adhere to the Anvil Engineering Standard (AES) Phase 2 requirements, specifically identifying the Assurance Assurance Level (AAL) for these core modifications.

Structure your roadmap as follows:

### Phase 1: Codebase Discovery & Architectural Baseline
*   [ ] Details of the Saguaro queries run and the core files identified (`agents/`, `llm/`, `tools/`).
*   [ ] Current limitations in the discrete tool calling pipeline when factoring in continuous states.
*   [ ] AES AAL Classification for the upgrade (Expected: AAL-0 or AAL-1 Critical Path).

### Phase 2: Design - Native Tooling in Continuous Latent Space
*   [ ] Engineering design: How continuous vectors will trigger discrete tool boundaries (e.g., a specific latent threshold or special `<TOOL_START>` token equivalent).
*   [ ] Engineering design: Resuming continuous thought from discrete tool outputs (e.g., mapping textual results through a specific embedding adapter back into the continuous stream).

### Phase 3: Design - Adaptive Subagent Chaining & COCONUT Allocation
*   [ ] Logic for the MasterAgent to predict task complexity and dynamically size the SubAgent chain.
*   [ ] Heuristics/Model inputs required to throttle/extend COCONUT latent iterations per agent step without manual intervention.

### Phase 4: Implementation & Verification Steps (AES Compliant)
*   [ ] Concrete Python interfaces/abstract base classes to be added/modified.
*   [ ] AAL validation gates (Latency/Throughput impact analysis of COCONUT depth scaling, Golden Signals).
*   [ ] Control dynamics (Loop health evidence and anomalies for the Subagent spawn/terminate loop).

## Execution Trigger
To begin, query Saguaro for current agent orchestration and LLM generation loops, map the intersections where discrete tokens are generated for tool calls, and begin constructing your roadmap.
