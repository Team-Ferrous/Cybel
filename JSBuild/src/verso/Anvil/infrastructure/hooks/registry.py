from datetime import datetime, timezone
from typing import Dict, List, Any
from infrastructure.hooks.base import Hook


class HookRegistry:
    """Central registry for hooks."""

    FAIL_CLOSED_HOOKS = {
        "pre_tool_use",
        "post_tool_use",
        "post_write_verify",
        "pre_finalize",
        "pre_irreversible_action",
    }

    def __init__(self):
        self.hooks: Dict[str, List[Hook]] = {
            "pre_tool_use": [],
            "post_tool_use": [],
            "post_write_verify": [],
            "pre_dispatch": [],
            "post_dispatch": [],
            "pre_finalize": [],
            "pre_irreversible_action": [],
            "pre_agent_start": [],
            "post_agent_end": [],
            "on_error": [],
            "on_thinking": [],
        }

    def register(self, hook_type: str, hook: Hook):
        """Register hook for lifecycle event."""
        if hook_type not in self.hooks:
            self.hooks[hook_type] = []
        self.hooks[hook_type].append(hook)

    def has_hooks(self, hook_type: str) -> bool:
        """Return True when at least one hook is registered for the lifecycle event."""
        return bool(self.hooks.get(hook_type))

    def execute(self, hook_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute all hooks of given type."""
        if hook_type not in self.hooks:
            return context

        receipts = context.setdefault("hook_receipts", [])
        for hook in self.hooks[hook_type]:
            receipt = {
                "hook_name": hook.name,
                "hook_type": hook_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "trace_id": context.get("trace_id"),
            }
            try:
                # Hooks modify context in place, but we expect return val too
                res = hook.execute(context)
                if res:
                    context = res
                receipt["outcome"] = "ok"
            except Exception as e:
                # Critical lifecycle hooks are fail-closed by design.
                print(f"Error in hook {hook.name}: {e}")
                receipt["outcome"] = "error"
                receipt["error"] = str(e)
                receipts.append(receipt)
                if hook_type in self.FAIL_CLOSED_HOOKS:
                    raise RuntimeError(
                        f"Fail-closed hook '{hook.name}' ({hook_type}) error: {e}"
                    ) from e
                continue
            receipts.append(receipt)

        return context
