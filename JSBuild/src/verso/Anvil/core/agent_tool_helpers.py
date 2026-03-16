from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List

from rich.panel import Panel

from core.context_compression import (
    apply_context_updates,
    ensure_context_updates_arg,
    extract_context_updates,
)

logger = logging.getLogger(__name__)


def _tool_symbols(name: Any, args: Dict[str, Any]) -> list[str]:
    symbols: set[str] = set()
    for key in ("symbol", "entity", "target", "class_name", "function_name"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            symbols.add(value.strip())
    if str(name) == "slice":
        target = args.get("entity") or args.get("target")
        if isinstance(target, str) and target.strip():
            symbols.add(target.strip())
    return sorted(symbols)


def _tool_files(args: Dict[str, Any]) -> list[str]:
    files: set[str] = set()
    for key in ("file_path", "path", "AbsolutePath"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            files.add(value.strip())
    for key in ("paths", "files"):
        value = args.get(key)
        if isinstance(value, list):
            files.update(str(item).strip() for item in value if str(item).strip())
        elif isinstance(value, dict):
            files.update(str(item).strip() for item in value.keys() if str(item).strip())
    return sorted(files)


def extract_tool_calls(text: str) -> List[Dict[str, Any]]:
    # Robustness: Clean up potential role marker leakage which can break JSON parsing if they appear inside tags
    text = re.sub(r"assistant<\|end_of_role\|>", "", text)
    text = re.sub(r"<\|start_of_role\|>assistant<\|end_of_role\|>", "", text)

    xml_pattern = r"<tool_call>\s*({.*?})\s*</tool_call>"
    native_pattern = r"tool\|>\s*({.*?})\s*(?:<\||</tool_call>|$)"
    xml_attr_pattern = r"tool<tool_name=[\"'](\w+)[\"']\s*(?:arguments=[\"'](.*?)[\"'])?(?:>|/?>)(.*?)(?:</tool>|$)"
    loose_pattern = r"<tool_call\s*({.*?})\s*(?:/?>|</tool_call>)"
    name_and_args_pattern = r"<tool_call>\s*(\w+)\s*(\{.*?\})\s*</tool_call>"

    tool_calls = []
    seen_indices = set()

    for pattern_tuple in [
        (xml_pattern, "json_only"),
        (native_pattern, "json_only"),
        (loose_pattern, "json_only"),
        (name_and_args_pattern, "name_and_json"),
    ]:
        pattern = pattern_tuple[0]
        parse_type = pattern_tuple[1]
        for match in re.finditer(pattern, text, re.DOTALL):
            if any(i in seen_indices for i in range(*match.span())):
                continue

            if parse_type == "json_only":
                try:
                    tool_calls.append(json.loads(match.group(1)))
                    for i in range(*match.span()):
                        seen_indices.add(i)
                except json.JSONDecodeError:
                    try:
                        fixed = re.sub(r"(\w+):", r'"\1":', match.group(1))
                        tool_calls.append(json.loads(fixed))
                        for i in range(*match.span()):
                            seen_indices.add(i)
                    except json.JSONDecodeError:
                        pass
            elif parse_type == "name_and_json":
                tool_name = match.group(1)
                json_args = match.group(2)
                try:
                    args = json.loads(json_args)
                    tool_calls.append({"name": tool_name, "arguments": args})
                    for i in range(*match.span()):
                        seen_indices.add(i)
                except json.JSONDecodeError:
                    pass

    for match in re.finditer(xml_attr_pattern, text, re.DOTALL):
        if any(i in seen_indices for i in range(*match.span())):
            continue
        name = match.group(1)
        raw_args = match.group(2) or match.group(3) or "{}"

        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            args = {}
            arg_matches = re.finditer(r"<(\w+)>(.*?)</\1>", raw_args, re.DOTALL)
            for am in arg_matches:
                args[am.group(1)] = am.group(2).strip()

            if not args and ":" in raw_args:
                m = re.search(r'["\']?(\w+)["\']?:\s*[" ]?([^" ]*)[" ]?', raw_args)
                if m:
                    args[m.group(1)] = m.group(2)

        tool_calls.append({"name": name, "arguments": args})
        for i in range(*match.span()):
            seen_indices.add(i)

    return tool_calls


def _hook_failures(context: Dict[str, Any], hook_type: str) -> List[Dict[str, Any]]:
    receipts = context.get("hook_receipts", [])
    if not isinstance(receipts, list):
        return []
    failures: List[Dict[str, Any]] = []
    for receipt in receipts:
        if not isinstance(receipt, dict):
            continue
        if receipt.get("hook_type") == hook_type and receipt.get("outcome") == "error":
            failures.append(receipt)
    return failures


def _format_hook_failure(failures: List[Dict[str, Any]], hook_type: str) -> str:
    first = failures[0] if failures else {}
    hook_name = first.get("hook_name", "unknown_hook")
    message = first.get("error", "unknown hook execution error")
    return f"AES GATE: {hook_type} hook '{hook_name}' failed: {message}"


def execute_tool(agent: Any, tool_call: Dict[str, Any], retries: int = 3, delay: int = 2) -> str:
    name = tool_call.get("name")
    args = dict(tool_call.get("arguments", {}) or {})
    write_tools = {
        "write_file",
        "edit_file",
        "write_files",
        "apply_patch",
        "delete_file",
        "move_file",
        "rollback_file",
    }
    is_write_tool = name in write_tools
    agent._log_timeline_event(
        "tool:start",
        {"tool_name": name, "arg_keys": sorted(args.keys())},
    )
    files = _tool_files(args)
    symbols = _tool_symbols(name, args)
    if hasattr(agent, "_record_runtime_event"):
        agent._record_runtime_event(
            "tool_start",
            phase="tool",
            status="start",
            files=files,
            symbols=symbols,
            tool_calls=[str(name)],
            metadata={
                "tool_name": name,
                "arg_keys": sorted(args.keys()),
                "attempts_allowed": retries,
            },
            source=getattr(agent, "name", "agent"),
        )
    ensure_context_updates_arg(args)
    context_updates = extract_context_updates(args)
    if context_updates:
        outcome = apply_context_updates(
            agent.history.get_messages(),
            context_updates,
            on_compressed=agent._on_tool_message_compressed,
        )
        if outcome.get("applied"):
            agent.history.save()

    if name in {"read_file", "read_files"} and "max_chars" in args:
        try:
            requested_cap = int(args.get("max_chars"))
        except (TypeError, ValueError):
            requested_cap = None
        has_line_window = (
            args.get("start_line") is not None or args.get("end_line") is not None
        )
        if requested_cap is not None and requested_cap < 4000 and not has_line_window:
            logger.warning(
                "Ignoring tiny max_chars=%s for %s without line window.",
                requested_cap,
                name,
            )
            args.pop("max_chars", None)

    if not agent.approval_manager.request_approval(name, args):
        agent.console.print("[bold red]Tool execution denied by user.[/bold red]")
        agent._log_timeline_event(
            "tool:denied",
            {"tool_name": name, "reason": "approval_denied"},
        )
        if hasattr(agent, "_record_runtime_event"):
            agent._record_runtime_event(
                "tool_denied",
                phase="tool",
                status="denied",
                files=files,
                symbols=symbols,
                tool_calls=[str(name)],
                metadata={"tool_name": name, "reason": "approval_denied"},
                source=getattr(agent, "name", "agent"),
            )
        return f"Tool '{name}' execution was denied by the user."

    if is_write_tool and not agent.hook_registry.has_hooks("post_write_verify"):
        agent._log_timeline_event(
            "tool:end",
            {
                "tool_name": name,
                "status": "error",
                "error": "missing_post_write_verify_hooks",
            },
        )
        if hasattr(agent, "_record_runtime_event"):
            agent._record_runtime_event(
                "tool_blocked",
                phase="tool",
                status="error",
                files=files,
                symbols=symbols,
                tool_calls=[str(name)],
                metadata={
                    "tool_name": name,
                    "reason": "missing_post_write_verify_hooks",
                },
                source=getattr(agent, "name", "agent"),
            )
        return (
            "AES GATE: write-capable tool blocked because no post_write_verify "
            "hooks are registered."
        )

    logger.info(f"Executing tool: {name} with args: {json.dumps(args)}")
    agent.console.print(
        Panel(
            f"[bold cyan]Executing Tool:[/bold cyan] {name}\n[dim]{json.dumps(args)}[/dim]",
            border_style="cyan",
        )
    )

    for attempt in range(retries):
        try:
            hook_context = {
                "agent": agent,
                "tool_name": name,
                "tool_args": args,
                "trace_id": getattr(agent, "current_mission_id", None),
                "hook_receipts": [],
            }
            agent.hook_registry.execute("pre_tool_use", hook_context)
            pre_failures = _hook_failures(hook_context, "pre_tool_use")
            if pre_failures:
                raise RuntimeError(_format_hook_failure(pre_failures, "pre_tool_use"))
            agent._active_tool_execution = True
            try:
                result = agent.registry.dispatch(name, args)
            finally:
                agent._active_tool_execution = False
            hook_context["tool_result"] = result
            agent.hook_registry.execute("post_tool_use", hook_context)
            post_failures = _hook_failures(hook_context, "post_tool_use")
            if post_failures:
                raise RuntimeError(_format_hook_failure(post_failures, "post_tool_use"))

            if is_write_tool:
                hook_context["write_targets"] = agent._extract_write_targets(name, args)
                agent.hook_registry.execute("post_write_verify", hook_context)
                verify_failures = _hook_failures(hook_context, "post_write_verify")
                if verify_failures:
                    raise RuntimeError(
                        _format_hook_failure(verify_failures, "post_write_verify")
                    )

                post_write = hook_context.get("post_write_verification")
                if not isinstance(post_write, dict) or not post_write.get("verified"):
                    raise RuntimeError(
                        "AES GATE: post-write verification did not produce a valid "
                        "verified receipt."
                    )
                if hook_context.get("write_blocked") or post_write.get("blocked"):
                    blocked_message = (
                        hook_context.get("tool_result")
                        or "AES GATE: post-write verification failed."
                    )
                    agent._last_tool_execution_meta = {
                        "hook_receipts": hook_context.get("hook_receipts", []),
                        "post_write_verification": post_write,
                    }
                    agent._log_timeline_event(
                        "tool:end",
                        {
                            "tool_name": name,
                            "status": "blocked",
                            "result_chars": len(str(blocked_message)),
                        },
                    )
                    if hasattr(agent, "_record_runtime_event"):
                        agent._record_runtime_event(
                            "tool_blocked",
                            phase="tool",
                            status="blocked",
                            files=files or hook_context.get("write_targets"),
                            symbols=symbols,
                            tests=list(
                                (
                                    hook_context.get("post_write_verification") or {}
                                ).get("tests", [])
                            ),
                            tool_calls=[str(name)],
                            metadata={
                                "tool_name": name,
                                "result_chars": len(str(blocked_message)),
                            },
                            source=getattr(agent, "name", "agent"),
                        )
                    return str(blocked_message)

            logger.debug(f"Tool {name} raw result length: {len(str(result))}")
            truncated = agent.truncator.truncate(name, args, str(result))
            agent._last_tool_execution_meta = {
                "hook_receipts": hook_context.get("hook_receipts", []),
                "post_write_verification": hook_context.get("post_write_verification"),
            }
            agent._log_timeline_event(
                "tool:end",
                {
                    "tool_name": name,
                    "status": "ok",
                    "result_chars": len(truncated),
                },
            )
            if hasattr(agent, "_record_runtime_event"):
                agent._record_runtime_event(
                    "tool_end",
                    phase="tool",
                    status="ok",
                    files=files or hook_context.get("write_targets"),
                    symbols=symbols,
                    tests=list(
                        ((hook_context.get("post_write_verification") or {}).get("tests"))
                        or []
                    ),
                    tool_calls=[str(name)],
                    metadata={
                        "tool_name": name,
                        "result_chars": len(truncated),
                        "attempt": attempt + 1,
                    },
                    source=getattr(agent, "name", "agent"),
                )
            return truncated
        except Exception as e:
            if str(e).startswith("AES GATE:"):
                agent.console.print(f"[red]{e}[/red]")
                agent._log_timeline_event(
                    "tool:end",
                    {
                        "tool_name": name,
                        "status": "blocked",
                        "error": str(e),
                        "attempts": attempt + 1,
                    },
                )
                if hasattr(agent, "_record_runtime_event"):
                    agent._record_runtime_event(
                        "tool_blocked",
                        phase="tool",
                        status="blocked",
                        files=files,
                        symbols=symbols,
                        tool_calls=[str(name)],
                        metadata={
                            "tool_name": name,
                            "error": str(e),
                            "attempt": attempt + 1,
                        },
                        source=getattr(agent, "name", "agent"),
                    )
                agent._last_tool_execution_meta = {
                    "hook_receipts": hook_context.get("hook_receipts", [])
                    if "hook_context" in locals()
                    else [],
                }
                return str(e)
            if attempt < retries - 1:
                agent.console.print(
                    f"[yellow]Tool execution failed. Retrying in {delay} seconds...[/yellow]"
                )
                time.sleep(delay)
            else:
                agent.console.print(
                    f"[red]Tool execution failed after {retries} attempts.[/red]"
                )
                agent._log_timeline_event(
                    "tool:end",
                    {
                        "tool_name": name,
                        "status": "error",
                        "error": str(e),
                        "attempts": retries,
                    },
                )
                if hasattr(agent, "_record_runtime_event"):
                    agent._record_runtime_event(
                        "tool_error",
                        phase="tool",
                        status="error",
                        files=files,
                        symbols=symbols,
                        tool_calls=[str(name)],
                        metadata={
                            "tool_name": name,
                            "error": str(e),
                            "attempts": retries,
                        },
                        source=getattr(agent, "name", "agent"),
                    )
                agent._last_tool_execution_meta = {}
                return f"Error executing {name}: {str(e)}"

    return f"Error executing {name}: unknown failure"
