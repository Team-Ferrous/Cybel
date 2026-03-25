from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict

import numpy as np
from rich.panel import Panel
from rich.table import Table

from config.settings import MASTER_MODEL

logger = logging.getLogger(__name__)


def run_loop(agent: Any, user_input: str) -> Dict[str, Any]:
    """Core thinking/action loop extracted from BaseAgent for lower method complexity."""
    start_time = time.time()
    agent._ensure_runtime_compliance_context(user_input)
    agent.history.add_message("user", user_input)

    step_count = 0
    final_response = ""
    tool_stats = {}

    while step_count < agent.max_autonomous_steps:
        step_count += 1
        logger.info(
            f"Agent {agent.name} starting step {step_count}/{agent.max_autonomous_steps}"
        )

        messages = agent.history.get_messages()
        system_prompt = agent._build_system_prompt(user_input)

        oneshot = [
            {"role": "user", "content": "Analyze the file structure."},
            {
                "role": "assistant",
                "content": '<thinking>\nI will list the files in the current directory to understand the project structure.\n</thinking>\n<tool_call>\n{"name": "run_command", "arguments": {"command": "ls", "_context_updates": []}}\n</tool_call>',
            },
            {"role": "tool", "content": "[src/, tests/, README.md, config/]"},
        ]

        chat_messages = [{"role": "system", "content": system_prompt}] + oneshot + messages

        adaptive_tier = "unknown"
        if agent.adaptive_context_manager and step_count == 1:
            context_items = (
                agent.proactive_context.file_list
                if hasattr(agent.proactive_context, "file_list")
                else []
            )
            adaptive_tier, adaptive_params = (
                agent.adaptive_context_manager.recommend_tier_with_lookahead(
                    user_input=user_input,
                    system_prompt=system_prompt,
                    context_items=context_items,
                    history=messages,
                )
            )
            if agent.output_format != "json":
                agent.console.print(
                    f"[dim][Adaptive Context] Tier: {adaptive_tier}, Context: {adaptive_params['num_ctx']}[/dim]"
                )

        recovery_hint = ""
        if (
            chat_messages
            and chat_messages[-1]["role"] == "tool"
            and "Error" in chat_messages[-1]["content"]
        ):
            recovery_hint = "\n[SYSTEM NOTE: Previous tool failed. Use other tools to diagnose or fix the environment before retrying.]"

        if agent.coconut_enabled:
            try:
                context_str = user_input + "\n" + (messages[-1]["content"] if messages else "")
                context_emb = agent.brain.get_embeddings(context_str)

                if context_emb is not None:
                    from core.thinking import EnhancedThinkingSystem

                    thinking_sys = EnhancedThinkingSystem(coconut_enabled=True)
                    refined_emb = thinking_sys.deep_think(np.array(context_emb))

                    if refined_emb is not None:
                        logger.info(
                            f"COCONUT: Latent reasoning completed for step {step_count}"
                        )

                        if thinking_sys.coconut.amplitudes is not None:
                            amplitudes = thinking_sys.coconut.amplitudes
                            table = Table(
                                title="COCONUT Multipath Amplitudes",
                                show_header=True,
                                header_style="bold magenta",
                            )
                            table.add_column("Path", style="cyan")
                            table.add_column("Amplitude", style="green")

                            for i, amp in enumerate(amplitudes):
                                table.add_row(f"Path {i+1}", f"{amp:.4f}")

                            agent.console.print(
                                Panel(
                                    table,
                                    title="[bold yellow]COCONUT Latent Reasoning[/bold yellow]",
                                    border_style="yellow",
                                )
                            )
            except Exception as e:
                logger.warning(f"COCONUT reasoning failed: {e}")

        full_response = agent._stream_response(
            chat_messages, assistant_prefix="<thinking>" + recovery_hint
        )
        agent.history.add_message("assistant", full_response)
        final_response = full_response
        final_response = full_response

        tool_calls = agent._extract_tool_calls(full_response)

        if not tool_calls:
            break

        for tool_call in tool_calls:
            try:
                name = tool_call["name"]
                tool_stats[name] = tool_stats.get(name, 0) + 1
                result = agent._execute_tool(tool_call)

                if agent.output_format != "json":
                    tool_name = tool_call.get("name")
                    full_visibility_tools = {
                        "read_file",
                        "read_files",
                        "slice",
                        "skeleton",
                        "query",
                        "saguaro_query",
                    }
                    should_truncate_display = (
                        tool_name not in full_visibility_tools
                        and len(result) > 10000
                    )
                    if should_truncate_display:
                        agent.console.print(
                            f"[dim]Result truncated ({len(result)} chars)[/dim]"
                        )
                        agent.console.print(
                            Panel(
                                result[:10000]
                                + "\n\n[bold yellow]... (output truncated for brevity) ...[/bold yellow]",
                                title="Tool Output (Partial)",
                                border_style="yellow",
                                expand=False,
                            )
                        )
                    else:
                        agent.console.print(
                            Panel(
                                result,
                                title="Tool Output",
                                border_style="green",
                                expand=False,
                            )
                        )

                agent._record_tool_result_message(
                    tool_call["name"],
                    tool_call.get("arguments", {}),
                    result,
                )
            except Exception as e:
                if agent.error_recovery:
                    recovery = agent.error_recovery.handle_error(
                        e,
                        {
                            "tool_name": tool_call.get("name"),
                            "step": step_count,
                            "user_input": user_input,
                        },
                    )
                    if recovery.recovered:
                        agent.history.add_message(
                            "tool",
                            f"Tool {tool_call.get('name')} failed but recovered via {recovery.strategy_used.value}",
                        )
                        continue

                agent.console.print(f"[bold red]Tool Execution Error:[/bold red] {e}")
                agent.history.add_message("tool", f"Tool Execution Error: {e}")
                break

    if step_count >= agent.max_autonomous_steps:
        if agent.output_format != "json":
            agent.console.print(
                f"[bold red]{agent.name} reached max autonomous steps.[/bold red]"
            )

    duration = (time.time() - start_time) * 1000

    if agent.perf_monitor:
        tokens_generated = len(final_response) // 4
        elapsed_sec = duration / 1000.0
        agent.perf_monitor.log(
            tokens=tokens_generated,
            elapsed=elapsed_sec,
            ctx_size=32768,
            tier=adaptive_tier if "adaptive_tier" in locals() else "unknown",
            model=agent.config.get("model", MASTER_MODEL),
        )

        if agent.output_format != "json":
            stats = agent.perf_monitor.get_stats()
            if stats["total_requests"] > 0:
                agent.console.print(
                    f"[dim][Perf] {stats['avg_throughput']:.1f} tokens/sec avg[/dim]"
                )

    return {
        "response": final_response,
        "stats": {
            "steps": step_count,
            "tool_calls": tool_stats,
            "duration_ms": int(duration),
        },
        "error": None,
    }


def simple_chat(agent: Any, user_input: str) -> str:
    """Conversational chat loop extracted from BaseAgent for lower method complexity."""
    from domains.task_execution.enhanced_loop import LoopSelector

    runtime_compliance = agent._format_runtime_compliance_context(user_input)

    if LoopSelector._is_repo_query(user_input):
        agent.console.print("[dim]Performing 0-shot Saguaro grounding scan...[/dim]")
        scan_res = agent._execute_tool(
            {"name": "saguaro_query", "arguments": {"query": user_input[:200], "k": 5}}
        )
        agent._record_tool_result_message(
            "saguaro_query",
            {"query": user_input[:200], "k": 5, "_context_updates": []},
            f"Initial Workspace Scan Result:\n{scan_res}",
        )

    agent.history.add_message("user", user_input)

    whitelist = [
        "web_search",
        "web_fetch",
        "browser_visit",
        "saguaro_query",
        "read_file",
        "read_files",
        "skeleton",
        "slice",
        "impact",
        "run_command",
        "lsp_definition",
        "lsp_references",
        "analyze_codebase",
        "verify",
        "delegate",
        "update_memory_bank",
        "search_arxiv",
        "fetch_arxiv_paper",
        "search_reddit",
        "search_hackernews",
        "search_stackoverflow",
    ]
    whitelisted_schemas = [s for s in agent.tool_schemas if s["name"] in whitelist]
    tools_json = json.dumps(whitelisted_schemas, indent=2)

    step_count = 0
    max_steps = 15

    thinking_protocol = """
### LONG-FORM THINKING PROTOCOL (SYSTEM 2)
1. **DEEP UNDERSTANDING**: For every new objective, start with `<thinking type="understanding">`. Deconstruct the request into first principles.
2. **STRATEGIC PLANNING**: Before tool use, use `<thinking type="planning">`. Map out the dependency graph of your actions.
3. **LATENT REASONING (CoCoNut)**: Leverage your latent reasoning paths to explore edge cases before projecting to natural language. Use the `think` tool if you need an explicit pause for reflection.
4. **SELF-CORRECTION**: If a tool result is unexpected, immediately use `<thinking type="correction">` to adjust your mental model.
5. **TOOL REFLECTION**: After every tool result, explicitly classify the outcome as `SUCCESS`, `PARTIAL`, or `FAILURE` in your next thinking block.
6. **VERIFICATION**: Always verify your assumptions by reading files or running tests. Never "hallucinate" code content.
7. **MULTI-TURN EXPLORATION**: If the first tool result isn't enough, continue thinking and acting until the objective is fully realized.
""".strip()

    exploration_mandate = """
### MANDATORY EXPLORATION PROTOCOL
CRITICAL: If the user asks about code, functions, classes, systems, components, implementations, architecture, or "how X works", you MUST:
1. Call saguaro_query to find relevant files
2. Use read_file, skeleton, or slice to inspect the actual code
3. NEVER answer from memory alone - always gather fresh evidence from the codebase

This applies to ANY question about:
- How something works (e.g., "how does X work?", "explain X")
- What code does (e.g., "what is X?", "describe X")
- Where things are implemented (e.g., "where is X?")
- System architecture or design

### STRICT TOOL CALL FORMAT
You MUST use the following XML-based syntax for ALL tool calls. No exceptions.
<tool_call>
{"name": "tool_name", "arguments": {"arg1": "value1"}}
</tool_call>

### CONTEXT COMPRESSION (REQUIRED)
Every tool call MUST include `_context_updates`.
- Use `_context_updates: []` when nothing should be compressed.
- Summarize only stale `[tcN]` results.
- Results without `[tcN]` are already compressed and must not be re-compressed.

Crucially, you MUST provide at least one sentence of natural language content either before or after your tool calls. NEVER output ONLY tool calls or ONLY thinking blocks.

### VERIFIED EVIDENCE
Your final response MUST include a brief section (e.g., "Verification" or "Evidence") summarizing which files or tool results were used to confirm your answer.
""".strip()

    from core.task_state import TaskStateManager
    from core.agent_mode import AgentMode

    task_manager = TaskStateManager()

    task_manager.start_task(
        name=f"Chat: {user_input[:40]}...",
        mode=AgentMode.PLANNING,
        status="Analyzing query and collecting initial context",
    )

    try:
        while step_count < max_steps:
            step_count += 1
            context_pressure = agent._context_pressure_guidance(user_input)
            model_name = getattr(agent.brain, "model_name", "")
            model_guidance = agent.prompt_manager.get_model_family_compression_guidance(
                model_name
            )

            cwd_context = agent._get_cwd_context()
            proactive_context = agent.proactive_context.get_context_prompt()

            project_memory_context = ""
            memory_bank_path = os.path.join(os.getcwd(), "GRANITE.md")
            if os.path.exists(memory_bank_path):
                with open(memory_bank_path, "r") as f:
                    project_memory_context = (
                        f"\n# PROJECT MEMORY BANK (GRANITE.md)\n{f.read()}\n"
                    )

            semantic_context = ""
            if (
                step_count == 1
                and hasattr(agent, "semantic_engine")
                and agent.semantic_engine
            ):
                relevant_files = agent.semantic_engine.get_context_for_objective(user_input)
                if relevant_files:
                    semantic_context = "\n# RELEVANT WORKSPACE CONTEXT\n"
                    for rel_path in relevant_files[:5]:
                        skeleton = ""
                        try:
                            skeleton = agent.semantic_engine.get_skeleton(rel_path) or ""
                        except Exception:
                            skeleton = ""
                        semantic_context += f"File: {rel_path}\n"
                        if skeleton:
                            semantic_context += f"{skeleton}\n"
                    semantic_context += "\n# SEMANTIC SEARCH RESULTS\n"
                    semantic_context += "The following files seem relevant to your objective. Use `read_file` or `skeleton` to explore them:\n"
                    for rel_path in relevant_files[:5]:
                        semantic_context += f"- {rel_path}\n"

            system_prompt = f"""You are **{agent.name}**, a world-class autonomous software engineer. 
You are currently operating in the **granite-agent** repository.

{agent.prompt_manager.get_master_prompt(
    agent_name=agent.name,
    context_type="conversational",
    task_text=user_input,
)}

{runtime_compliance}

### MISSION
You are in CONVERSATION mode. Your goal is to provide high-fidelity, evidence-based answers. Use **Saguaro Q-COS** as your primary source of truth for repository intelligence.

{exploration_mandate}

### STEP-BY-STEP ANALYSIS
1. **Analyze**: Use `<thinking type="understanding">` to determine what files or information you need.
2. **Execute**: Call semantic tools (e.g., `query`, `skeleton`, `slice`) to gather evidence from the Saguaro index.
3. **Verify**: Classify tool outputs in `<thinking type="reflection">`.
4. **Answer**: Provide your final response only after you have verified the facts.

{project_memory_context}
{cwd_context}
{proactive_context}
{semantic_context}

### MANDATORY RESEARCH ADVISORY
If Semantic Search results (based on Saguaro) are provided above, you MUST prioritize investigating those files using `skeleton` or `slice`. Do not provide a final answer until you have "seen" the implementation.

{thinking_protocol}

### SAGUARO-FIRST PROTOCOL
Always use **Saguaro Q-COS** tools (**query**, **skeleton**, **slice**, **impact**) for code exploration. This provides high-fidelity architectural context with minimal token overhead.

### ANTI-LOOP PROTOCOL
To prevent repetitive outputs and "semantic loops" when analyzing complex or low-level components (especially native code, C++, or performance optimizations):
1. **Focus on High-Level Summaries**: Do NOT attempt to explain every line of low-level code. Instead, synthesize its *purpose*, *input/output*, and *integration points* within the broader system.
2. **Avoid Repetition**: If you find your thoughts or generated text becoming repetitive (e.g., using the same phrases or cycling through similar ideas), immediately break the pattern.
3. **Explicitly State Limitations**: If you cannot synthesize a non-repetitive, meaningful explanation of a specific low-level detail, state "I lack sufficient high-level context or specialized knowledge to fully explain this low-level component without additional guidance."
4. **Ask for Clarification**: If stuck, ask the user a specific, guiding question to move forward.
5. **Prioritize Actions**: If internal reasoning becomes circular, prioritize using tools to gather new information or attempt a different approach.

### DELEGATION & ACTION
If the user request requires significant implementation, debugging, or complex refactoring, use the `delegate` tool to spawn a specialized subagent instead of attempting it in chat.

### TOOL PROTOCOL
Use the whitelisted tools below to explore the repository or fetch information.
Call tools using this syntax:
<tool_call>
{{"name": "tool_name", "arguments": {{"arg1": "val1", "_context_updates": []}}}}
</tool_call>

### CONTEXT PRESSURE
{context_pressure}
{model_guidance}

### WHITELISTED TOOLS
{tools_json}

Respond naturally. ALWAYS use `<thinking>` blocks to reason exhaustively before acting or answering.
"""
            messages = agent.history.get_messages()

            if step_count == 1 and semantic_context:
                messages.append(
                    {
                        "role": "system",
                        "content": "[SYSTEM ADVISORY: Research needed. Call a tool before providing natural language.]",
                    }
                )

            chat_messages = [{"role": "system", "content": system_prompt}] + messages

            prefix = (
                '<thinking type="understanding">'
                if step_count == 1 and not messages[-1]["role"] == "tool"
                else ""
            )

            if agent.coconut_enabled:
                try:
                    context_emb = agent.brain.get_embeddings(
                        user_input + "\n" + (messages[-1]["content"] if messages else "")
                    )
                    if context_emb is not None:
                        from core.thinking import EnhancedThinkingSystem

                        thinking_sys = EnhancedThinkingSystem(coconut_enabled=True)
                        refined_emb = thinking_sys.deep_think(np.array(context_emb))
                        if (
                            refined_emb is not None
                            and thinking_sys.coconut
                            and thinking_sys.coconut.amplitudes is not None
                        ):
                            from rich.table import Table

                            amplitudes = thinking_sys.coconut.amplitudes
                            table = Table(
                                title="COCONUT Multipath Amplitudes",
                                show_header=True,
                                header_style="bold magenta",
                            )
                            table.add_column("Path", style="cyan")
                            table.add_column("Amplitude", style="green")
                            for i, amp in enumerate(amplitudes):
                                table.add_row(f"Path {i+1}", f"{amp:.4f}")
                            agent.console.print(
                                Panel(
                                    table,
                                    title="[bold yellow]Chat COCONUT[/bold yellow]",
                                    border_style="yellow",
                                )
                            )
                except Exception as e:
                    logger.warning(f"Chat COCONUT failed: {e}")

            task_manager.update_boundary(
                status=f"Step {step_count}: Generating/Streaming response"
            )

            full_response = agent._stream_response(
                chat_messages,
                assistant_prefix=prefix,
                whitelist=whitelist,
                stop_on_tool=True,
            )
            agent.history.add_message("assistant", full_response)

            tool_calls = agent._extract_tool_calls(full_response)

            if not tool_calls and (
                "tool<" in full_response
                or "tool_name" in full_response
                or "[TOOL_CALLS]" in full_response
            ):
                agent.console.print(
                    "[bold red]Self-Correction: Model used malformed tool format. Re-prompting...[/bold red]"
                )
                agent.history.add_message(
                    "tool",
                    'Error: Your tool call was malformed. You MUST use the strict JSON format: <tool_call>{"name": "...", "arguments": {...}}</tool_call>. Do not use XML attributes or other custom tags.',
                )
                task_manager.update_boundary(status="Correcting malformed tool call")
                continue

            if not tool_calls:
                is_delegating = any(
                    "delegate_to" in msg.get("content", "")
                    for msg in chat_messages
                    if msg.get("role") == "assistant"
                )

                if (
                    step_count == 1
                    and not is_delegating
                    and any(
                        kw in user_input.lower()
                        for kw in [
                            "how",
                            "what",
                            "explain",
                            "describe",
                            "work",
                            "implement",
                            "function",
                            "class",
                            "code",
                            "bridge",
                            "system",
                            "native",
                            "coconut",
                            "saguaro",
                            "agent",
                            "loop",
                            "tool",
                            "method",
                            "module",
                            "component",
                        ]
                    )
                ):
                    agent.console.print(
                        "[yellow]⚠ No tools used. Forcing exploration...[/yellow]"
                    )
                    agent.history.add_message(
                        "system",
                        "ERROR: You generated only thinking blocks without calling any tools. "
                        "This question requires you to explore the codebase. You MUST call at least one tool "
                        "(e.g., saguaro_query, read_file, skeleton, slice) to gather evidence "
                        "before providing a final answer. Call a tool now.",
                    )
                    task_manager.update_boundary(status="Retrying with forced tool use")
                    continue

                task_manager.end_task()
                return full_response

            for tool_call in tool_calls:
                if tool_call["name"] in whitelist:
                    task_manager.update_boundary(
                        status=f"Executing tool: {tool_call['name']}"
                    )
                    result = agent._execute_tool(tool_call)
                    if agent.output_format != "json":
                        trunc_res = (
                            str(result)[:2000] + "..."
                            if len(str(result)) > 2000
                            else str(result)
                        )
                        agent.console.print(
                            Panel(
                                trunc_res,
                                title=f"Tool: {tool_call['name']}",
                                border_style="green",
                                expand=False,
                            )
                        )
                    agent._record_tool_result_message(
                        tool_call["name"],
                        tool_call.get("arguments", {}),
                        str(result),
                    )
                else:
                    agent.history.add_message(
                        "tool",
                        f"Error: Tool '{tool_call['name']}' is not whitelisted for conversation.",
                    )
    finally:
        if task_manager.get_state():
            task_manager.end_task()

    return full_response
