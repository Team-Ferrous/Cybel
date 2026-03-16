import unittest
from core.hooks.base import Hook
from core.hooks.registry import HookRegistry


class MockHook(Hook):
    def __init__(self, name="mock"):
        self._name = name
        self.executed = False

    @property
    def name(self) -> str:
        return self._name

    def execute(self, context):
        self.executed = True
        context["modified"] = True
        return context


class FailingHook(Hook):
    def __init__(self, name="failing"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def execute(self, context):
        raise RuntimeError("boom")


class TestHooks(unittest.TestCase):
    def setUp(self):
        self.registry = HookRegistry()

    def test_hook_registration(self):
        hook = MockHook()
        self.registry.register("test_event", hook)
        self.assertEqual(len(self.registry.hooks["test_event"]), 1)

    def test_hook_execution(self):
        hook = MockHook()
        self.registry.register("test_event", hook)

        ctx = {"initial": True}
        new_ctx = self.registry.execute("test_event", ctx)

        self.assertTrue(hook.executed)
        self.assertTrue(new_ctx.get("modified"))
        self.assertTrue(new_ctx.get("initial"))
        self.assertEqual(new_ctx["hook_receipts"][0]["hook_name"], "mock")
        self.assertEqual(new_ctx["hook_receipts"][0]["outcome"], "ok")

    def test_phase0_lifecycle_points_exist(self):
        for event_name in (
            "post_write_verify",
            "pre_finalize",
            "pre_irreversible_action",
        ):
            self.assertIn(event_name, self.registry.hooks)

    def test_fail_closed_hook_types_raise(self):
        self.registry.register("pre_tool_use", FailingHook())
        with self.assertRaises(RuntimeError):
            self.registry.execute("pre_tool_use", {"trace_id": "t1"})

    def test_non_critical_hook_failure_is_recorded(self):
        self.registry.register("test_event", FailingHook())
        ctx = self.registry.execute("test_event", {"trace_id": "t1"})
        self.assertEqual(ctx["hook_receipts"][0]["outcome"], "error")


if __name__ == "__main__":
    unittest.main()
