# Anvil Repository Walkthrough

Welcome to the **Anvil** repository. This document provides a step-by-step guide to how the system is structured and how it executes, specifically focusing on the REPL (Read-Eval-Print Loop) entry point.

## 📂 Repository Structure

The project is organized into several key directories:

| Directory | Description |
| :--- | :--- |
| `cli/` | Command-line interface logic, including the REPL (`repl.py`) and slash commands. |
| `core/` | The "brain" of the agent. Contains loop logic, memory management, and orchestrator. |
| `tools/` | External capabilities the agent can "call" (e.g., file system, web search, LSP). |
| `config/` | System-wide settings and configuration management. |
| `saguaro/` | Native in-tree semantic intelligence package used directly by Anvil. |
| `tests/` | Automated test suites for verifying agent capabilities. |

---

## 🚀 The Entry Point: `cli/repl.py`

The REPL is the primary way users interact with Anvil. When you run the agent, you are executing the `AgentREPL` class in `cli/repl.py`.

### 1. Initialization (`__init__`)
- **Console Setup**: Uses `rich` for a premium, stylized terminal output.
- **Managers**: Initializes `SaguaroSubstrate` (for indexing) and `ProjectMemory`.
- **Command Registry**: Loads "Slash Commands" (e.g., `/help`, `/mode`, `/plan`) into a central dispatcher.
- **Lazy Loading**: Systems like the `EnhancedAgenticLoop` and `AgentOrchestrator` are loaded only when needed to save resources.

### 2. Startup & Indexing
Upon launch, `cli/repl.py` triggers environment readiness and semantic indexing checks so the agent can ground itself in the workspace before you even ask a question.

### 3. The Main Loop (`run`)
The `run()` method starts an infinite loop using `prompt_toolkit`.
- **Slash Commands**: If your input starts with `/`, it's dispatched to the `CommandRegistry`.
- **Mission Handover**: Regular text is passed to `run_mission()`.

---

## 🔁 Mission Execution: The Loops

Anvil selects the best "loop" for your request based on its complexity.

### 1. `Simple Chat` (Conversational)
**File**: `core/agent.py`
- Used for questions (e.g., "How does this work?").
- Restricted toolset: Can search the web and read files but won't modify code.
- Uses a "Thinking Budget" to reason about the repository context before answering.

### 2. `run_loop` (Basic Agentic)
**File**: `core/agent.py`
- A standard Think-Act cycle.
- The agent analyzes the request, calls a tool, observes the result, and continues until complete.

### 3. `EnhancedAgenticLoop` (Enterprise)
**File**: `core/enhanced_loop.py`
- Triggered for complex objectives (e.g., "Implement feature X").
- **Phases**:
    - **Planning**: Researches the codebase and creates an `implementation_plan.md`.
    - **Execution**: Implements changes based on the approved plan.
    - **Verification**: Runs tests and self-corrects if something breaks.

### 4. `AgentOrchestrator` (Autonomous)
**File**: `core/orchestrator/scheduler.py`
- Manages a **Task Graph**.
- Breaks down massive requests into smaller `TaskUnits` handled by specialized sub-agents (Planner, Worker, Verifier).

---

## 🧠 Core Technologies

### Semantic Analysis (`SemanticEngine`)
Instead of just matching text, Anvil understands code *concepts*. It builds a semantic view of your project, allowing it to find relevant logic even if the filenames are ambiguous.

### Thinking Protocol
Anvil is designed with a **System 2 Thinking** approach. It uses `<thinking>` blocks to:
1. Deconstruct request first principles.
2. Map out dependency graphs.
3. Reflect on tool results before proceeding.

### Artifact Management
Plans, tasks, and walkthroughs are not just transient chat messages. They are saved as persistent `.md` files in the `.anvil/` directory, providing a verifiable audit trail of what the agent did.
