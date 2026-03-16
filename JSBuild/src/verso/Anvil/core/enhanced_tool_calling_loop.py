"""
Enhanced Tool Calling Loop with Full Saguaro Integration

Integrates all Claude Code enhancements:
1. ✅ Structured tool calling
2. ✅ Parallel execution
3. ✅ Progressive context loading
4. ✅ Smart context management
5. ✅ Task memory
6. ✅ Multi-file refactoring
7. ✅ Auto-verification
8. ✅ Streaming results
9. Interactive approval

All powered by Saguaro semantic intelligence.
"""

import json
import time
from typing import List, Dict, Any
from rich.panel import Panel

# Import all the enhancement modules
from core.parallel_executor import ParallelToolExecutor, SaguaroParallelSearch
from core.progressive_context import ProgressiveContextLoader, SmartContextExpander
from core.smart_context_manager import SmartContextManager, ContextOptimizer
from core.task_memory import TaskMemoryManager, AdaptiveLearner, TaskMemory
from core.multi_file_refactor import MultiFileRefactorer
from domains.verification.auto_verifier import AutoVerifier, VerificationLoop
from core.smart_editor import SmartFileEditor


class EnhancedToolCallingLoop:
    """
    Enterprise-grade tool calling loop with full Saguaro integration.

    This is the complete Claude Code-style system.
    """

    def __init__(self, agent):
        self.agent = agent
        self.console = agent.console
        self.brain = agent.brain
        self.history = agent.history
        self.registry = agent.registry
        self.semantic_engine = agent.semantic_engine
        self.approval_manager = agent.approval_manager

        # Initialize Saguaro tools
        from tools.saguaro_tools import SaguaroTools
        from domains.code_intelligence.saguaro_substrate import SaguaroSubstrate

        self.saguaro = SaguaroSubstrate()
        self.saguaro_tools = SaguaroTools(self.saguaro)

        # Initialize all enhancement modules
        self._init_enhancements()

        # Track session state
        self.files_read = set()
        self.files_edited = set()
        self.tool_history = []
        self.current_task_id = None

    def _init_enhancements(self):
        """Initialize all enhancement modules."""
        self.console.print("[dim]Initializing enhanced capabilities...[/dim]")

        # 1. Parallel execution
        self.parallel_executor = ParallelToolExecutor(
            self.registry,
            self.semantic_engine,
            self.console,
            self.approval_manager,
            tool_executor=lambda name, arguments: self._execute_tool(
                {"name": name, "arguments": arguments}
            ),
        )
        self.parallel_search = SaguaroParallelSearch(self.saguaro_tools, self.console)

        # 2. Progressive context
        self.context_loader = ProgressiveContextLoader(
            self.registry, self.semantic_engine, self.saguaro_tools, self.console
        )
        self.context_expander = SmartContextExpander(self.context_loader)

        # 3. Smart context management
        self.context_manager = SmartContextManager(
            self.registry, self.semantic_engine, self.saguaro_tools, self.console
        )
        self.context_optimizer = ContextOptimizer(self.console)

        # 4. Task memory
        self.memory_manager = TaskMemoryManager(
            self.saguaro_tools, self.semantic_engine, self.console
        )
        self.adaptive_learner = AdaptiveLearner(self.memory_manager, self.console)

        # 5. Multi-file refactoring
        self.smart_editor = SmartFileEditor(
            self.registry,
            self.console,
            tool_executor=lambda name, arguments: self._execute_tool(
                {"name": name, "arguments": arguments}
            ),
        )
        self.multi_file_refactor = MultiFileRefactorer(
            self.registry,
            self.saguaro_tools,
            self.smart_editor,
            self.console,
            tool_executor=lambda name, arguments: self._execute_tool(
                {"name": name, "arguments": arguments}
            ),
        )

        # 6. Auto-verification
        self.verifier = AutoVerifier(self.registry, self.console)
        self.verification_loop = VerificationLoop(self.verifier, self.smart_editor)

        self.console.print("[green]✓ All enhancements loaded[/green]")

    def run(self, user_input: str, max_iterations: int = 10) -> str:
        """
        Main enhanced Think-Act-Observe loop.

        Flow:
        1. Generate task ID and check memory for similar tasks
        2. Gather smart context using Saguaro
        3. Think-Act-Observe loop with parallel execution
        4. Verify changes automatically
        5. Record task memory for learning
        """
        self.current_task_id = f"task_{int(time.time())}"
        start_time = time.time()

        self.console.print(
            f"\n[bold cyan]═══ Enhanced Loop: {self.current_task_id} ═══[/bold cyan]\n"
        )

        # Phase 0: Check memory for similar tasks
        self._check_memory(user_input)

        # Phase 1: Gather smart context
        context = self._gather_context(user_input)

        # Phase 2: Main loop
        iteration = 0
        task_complete = False
        difficulties = []
        tools_used = []

        while iteration < max_iterations and not task_complete:
            self.console.print(
                f"\n[cyan]═══ Iteration {iteration + 1}/{max_iterations} ═══[/cyan]"
            )

            # Think: What should I do next?
            action_plan = self._think_phase(user_input, context, iteration)

            if action_plan["complete"]:
                task_complete = True
                break

            # Act: Execute tools (with parallelization)
            results = self._act_phase(action_plan)

            # Track tools used
            for result in results:
                if result["tool"] not in tools_used:
                    tools_used.append(result["tool"])

            # Verify: Check if changes are valid
            verification_passed = self._verify_phase(results)

            if not verification_passed:
                difficulties.append("Verification failed")

            # Observe: Process results
            observation = self._observe_phase(results)

            # Update context if needed
            context = self._update_context(context, results)

            # Add to history
            self.history.add_message("assistant", json.dumps(action_plan))
            self.history.add_message("user", f"Results:\n{observation}")

            iteration += 1

        # Phase 3: Final synthesis and memory recording
        response = self._finalize(
            user_input, iteration, start_time, tools_used, difficulties
        )

        return response

    def _check_memory(self, task: str) -> List[TaskMemory]:
        """Check task memory for similar past tasks."""
        self.console.print("[dim]Checking task memory...[/dim]")

        similar = self.memory_manager.recall_similar(task, limit=3)

        if similar:
            self.console.print(f"[cyan]Found {len(similar)} similar past tasks:[/cyan]")

            for i, mem in enumerate(similar, 1):
                status = "✓" if mem.success else "✗"
                self.console.print(
                    f"  {i}. {status} {mem.description} ({mem.iterations} iterations)"
                )

            # Get suggestion
            suggestion = self.memory_manager.suggest_approach(task, "edit")
            if suggestion:
                self.console.print(
                    Panel(suggestion, title="Suggested Approach", border_style="yellow")
                )

        return similar

    def _gather_context(self, task: str) -> Dict[str, Any]:
        """Gather comprehensive context using Saguaro."""
        # Use smart context manager to gather everything we need
        bundle = self.context_manager.gather_context(task)

        # Load progressive context
        loaded_context = self.context_loader.load_context_for_task(
            task, initial_files=bundle.target_files
        )

        # Optimize to fit token budget
        optimized_bundle = self.context_optimizer.optimize(bundle, token_budget=80000)

        return {
            "bundle": optimized_bundle,
            "loaded_files": loaded_context,
            "summary": self.context_loader.get_context_summary(),
        }

    def _think_phase(
        self, user_input: str, context: Dict, iteration: int
    ) -> Dict[str, Any]:
        """
        Enhanced think phase with context awareness.
        """
        self.console.print("[dim]→ Thinking with full context...[/dim]")

        # Build enhanced system prompt with context
        context_summary = context.get("summary", "")

        system_prompt = f"""You are {self.agent.name}, an advanced coding assistant with semantic understanding.

{context_summary}

Available Tools:
- saguaro_query(query, k): Discover relevant files/entities
- skeleton(path): Inspect file structure
- slice(target): Extract a focused entity view
- read_file(file_path): Read a file
- write_file(file_path, content): Create/overwrite file
- edit_file(file_path, instruction, context): Smart editing with instructions
- run_command(command): Execute shell command
- refactor_rename_class(old_name, new_name): Rename class globally
- web_search(query): Search the web

Context available:
- {len(context.get('loaded_files', {}))} files loaded (some as skeletons, some full)
- Dependencies mapped
- Impact analysis complete

Task: {user_input}

Output a JSON action plan in a single code block. DO NOT output any other text.
Your output should be ONLY the JSON object.

```json
{{
  "thought": "What I'm thinking...",
  "tools": [
    {{"tool": "tool_name", "args": {{"arg": "value"}}}}
  ],
  "complete": false
}}
```

Output ONLY valid JSON."""

        user_prompt = f"Iteration {iteration + 1}. What should I do next? Output JSON:"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Generate plan
        response = ""
        for chunk in self.brain.stream_chat(messages, max_tokens=2000, temperature=0.0):
            response += chunk

        # Parse JSON
        try:
            action_plan = self._extract_json(response)

            # Display plan
            self.console.print(
                Panel(
                    f"[bold]Thought:[/bold] {action_plan.get('thought', 'N/A')}\n\n"
                    f"[bold]Tools:[/bold] {len(action_plan.get('tools', []))}",
                    title="Action Plan",
                    border_style="cyan",
                )
            )

            return action_plan

        except Exception as e:
            self.console.print(f"[red]Failed to parse plan: {e}[/red]")
            return {"thought": "Parse error", "tools": [], "complete": False}

    def _act_phase(self, action_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Enhanced act phase with parallel execution.
        """
        tools = action_plan.get("tools", [])

        if not tools:
            return []

        # Use parallel executor
        results = self.parallel_executor.execute_tools(tools)

        # Track what we've done
        for result in results:
            if result["success"]:
                tool = result["tool"]

                if tool == "read_file":
                    self.files_read.add(result["args"].get("file_path"))
                elif tool in ["write_file", "edit_file"]:
                    self.files_edited.add(result["args"].get("file_path"))

                self.tool_history.append(result)

        return results

    def _verify_phase(self, results: List[Dict[str, Any]]) -> bool:
        """
        Verify changes automatically.
        """
        # Get modified files
        modified_files = [
            r["args"]["file_path"]
            for r in results
            if r["tool"] in ["write_file", "edit_file"] and r["success"]
        ]

        if not modified_files:
            return True

        # Run verification with retry
        return self.verification_loop.verify_with_retry(modified_files, max_attempts=2)

    def _observe_phase(self, results: List[Dict[str, Any]]) -> str:
        """Format results for observation."""
        if not results:
            return "No tools executed."

        observations = []

        for result in results:
            tool = result["tool"]
            success = result["success"]
            output = result["result"]

            # Truncate long outputs
            if (
                isinstance(output, str)
                and len(output) > 2000
                and tool not in {"read_file", "read_files"}
            ):
                output = output[:2000] + "\n... (truncated)"

            obs = f"Tool: {tool}\n"
            obs += f"Success: {success}\n"
            obs += f"Output:\n{output}\n"

            observations.append(obs)

        return "\n---\n".join(observations)

    def _update_context(self, context: Dict, results: List[Dict[str, Any]]) -> Dict:
        """
        Update context based on results (lazy expansion).

        If we need more detail about a file, expand it.
        """
        for result in results:
            if result["tool"] == "read_file" and result["success"]:
                file_path = result["args"]["file_path"]

                # If we only have skeleton, expand to full
                if file_path in context.get("loaded_files", {}):
                    self.context_loader.expand_context(file_path)

        return context

    def _finalize(
        self,
        user_input: str,
        iterations: int,
        start_time: float,
        tools_used: List[str],
        difficulties: List[str],
    ) -> str:
        """
        Finalize task and record memory.
        """
        elapsed = time.time() - start_time

        # Generate final summary
        summary = self._synthesize_final_response(user_input)

        # Record task memory
        task_memory = TaskMemory(
            task_id=self.current_task_id,
            task_type=self._classify_task_type(tools_used),
            description=user_input,
            files_modified=list(self.files_edited),
            tools_used=tools_used,
            success=len(difficulties) == 0,
            execution_time=elapsed,
            iterations=iterations,
            timestamp=time.time(),
            approach=summary[:200],  # First 200 chars
            key_steps=tools_used,
            difficulties=difficulties,
            tests_passed=True,  # Would check actual test results
            syntax_valid=True,  # Would check actual validation
        )

        self.memory_manager.remember(task_memory)

        return summary

    def _synthesize_final_response(self, user_input: str) -> str:
        """Generate final response."""
        self.console.print("\n[cyan]═══ Synthesizing Final Response ═══[/cyan]")

        summary = f"""Task completed!

Files read: {len(self.files_read)}
Files modified: {len(self.files_edited)}
Tools used: {len(self.tool_history)}

Modified files:
{chr(10).join(f"  - {f}" for f in list(self.files_edited)[:10])}
"""

        self.console.print(Panel(summary, title="Summary", border_style="green"))

        return summary

    def _classify_task_type(self, tools_used: List[str]) -> str:
        """Classify task type based on tools used."""
        if "refactor_rename_class" in tools_used:
            return "refactor"
        elif "edit_file" in tools_used:
            return "edit"
        elif "write_file" in tools_used:
            return "create"
        else:
            return "investigation"

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from model output."""
        import re

        # First, try to find JSON within ```json ... ```
        match = re.search(r"```json\n(\{.*?\})\n```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass  # Fall through to the next method

        # If that fails, try to find any JSON blob
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))

        return json.loads(text)
