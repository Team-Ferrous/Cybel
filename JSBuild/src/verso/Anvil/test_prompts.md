# Granite Agent Test Prompts

This document contains a set of test prompts designed to evaluate the Granite Agent's performance across various tasks, including code understanding, generation, refactoring, and problem-solving.

## Instructions

1.  **Run each prompt individually** in the Granite Agent REPL or via `main.py`.
2.  **Observe the agent's behavior**, output, and success in completing the task.
3.  **Note any issues**, unexpected behavior, or areas for improvement.
4.  Once all prompts have been run, the agent will analyze the results and provide a grade, along with suggestions for upgrading its capabilities.

---

## Test Cases

### 1. Code Understanding & Explanation

**Prompt:** "Explain the purpose and functionality of the `core/unified_chat_loop.py` file. Describe its main classes, methods, and how it integrates with other parts of the agent."

### 2. Simple Code Generation

**Prompt:** "Write a Python function `calculate_fibonacci(n)` that returns the nth Fibonacci number using recursion. Include a docstring and type hints."

### 3. Refactoring Existing Code

**Prompt:** "Refactor the `cli/repl.py` file to extract the command parsing logic into a separate function or class. Ensure existing functionality remains intact."

### 4. Bug Fixing (Hypothetical)

**Prompt:** "Assume there's a bug in `core/model_manager.py` where `load_model` sometimes fails to load models larger than 10GB due to a memory allocation error. Propose a solution and implement a placeholder fix that logs a warning if a large model is detected."

### 5. Test Generation

**Prompt:** "Generate a `pytest` test suite for the `core/env_manager.py` file, covering its primary functions and edge cases."

### 6. Configuration Modification

**Prompt:** "Add a new configuration option named `AGENT_LOG_LEVEL` to `config/settings.py` with a default value of `'INFO'`. Make sure it's accessible via `ConfigManager`."

### 7. Creative Task / New Feature

**Prompt:** "Design and outline a new sub-agent called `DocumentationAgent` that can automatically generate `README.md` files for Python modules based on their docstrings and code structure. Provide the basic class structure in `agents/documentation.py`."

### 8. Performance Optimization (Conceptual)

**Prompt:** "Identify potential performance bottlenecks in the `Saguaro` component's semantic search (e.g., in `Saguaro/search.py`) and suggest three concrete optimization strategies, without implementing them."

---

**After running all these prompts, please provide feedback and be ready for the agent to analyze your observations.**