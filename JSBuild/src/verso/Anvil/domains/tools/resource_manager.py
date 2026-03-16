import asyncio
import time
from typing import Any, Dict, Callable
from shared_kernel.event_store import get_event_store


class ToolResourceManager:
    """
    Manages resource budgets and timeouts for tool executions.
    Prevents runaway tools from exhausting CPU/Memory or hanging the loop.
    """

    def __init__(self):
        # Default budgets
        self.budgets = {
            "grep": {"timeout": 30, "max_output_chars": 50000},
            "run_command": {"timeout": 300, "max_output_chars": 100000},
            "web_search": {"timeout": 15, "max_output_chars": 10000},
            "analyze_codebase": {"timeout": 600, "max_output_chars": 200000},
            "deadcode": {"timeout": 180, "max_output_chars": 400000},
            # Grounding-heavy file reads need larger ceilings.
            "read_file": {"timeout": 120, "max_output_chars": 2000000},
            "read_files": {"timeout": 180, "max_output_chars": 3000000},
        }
        self.default_timeout = 60
        self.default_max_output = 50000

    async def execute_with_budget(
        self, tool_name: str, func: Callable, kwargs: Dict[str, Any]
    ) -> Any:
        """
        Executes a tool within its allocated budget.
        """
        budget = self.budgets.get(tool_name, {})
        timeout = budget.get("timeout", self.default_timeout)
        max_output = budget.get("max_output_chars", self.default_max_output)

        start_time = time.time()

        try:
            # Handle sync/async transparency
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(func(**kwargs), timeout=timeout)
            else:
                # Run sync functions in a thread to allow timeout interruption
                # Note: This is an approximation for CPU-bound tasks
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, lambda: func(**kwargs)
                    ),
                    timeout=timeout,
                )

            # Post-process: enforce output size limits
            result_str = str(result)
            if len(result_str) > max_output:
                truncated_result = (
                    result_str[:max_output]
                    + f"\n\n[TRUNCATED: Output exceeded {max_output} characters]"
                )
                return truncated_result

            return result

        except asyncio.TimeoutExpired:
            get_event_store().emit(
                event_type="TOOL_TIMEOUT",
                source="ToolResourceManager",
                payload={"tool": tool_name, "timeout": timeout},
            )
            return f"Error: Tool '{tool_name}' timed out after {timeout} seconds."
        except Exception as e:
            return f"Error executing tool '{tool_name}': {str(e)}"
        finally:
            elapsed = time.time() - start_time
            # Log high-resource usage
            if elapsed > (timeout * 0.8):
                get_event_store().emit(
                    event_type="HIGH_RESOURCE_USAGE",
                    source="ToolResourceManager",
                    payload={"tool": tool_name, "elapsed": elapsed, "limit": timeout},
                )
