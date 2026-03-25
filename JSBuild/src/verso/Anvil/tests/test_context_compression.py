import unittest

from core.context_compression import (
    CONTEXT_UPDATES_KEY,
    apply_context_updates,
    auto_compress_dead_end_reads,
    ensure_context_updates_arg,
    extract_context_updates,
    infer_next_tc_id,
    inject_context_updates_param,
    label_tool_result,
)


class TestContextCompression(unittest.TestCase):
    def test_schema_injection_adds_required_param(self):
        schema = {
            "name": "read_file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        }
        injected = inject_context_updates_param(schema)
        params = injected["parameters"]
        self.assertIn(CONTEXT_UPDATES_KEY, params["properties"])
        self.assertIn(CONTEXT_UPDATES_KEY, params["required"])

    def test_extract_context_updates_pops_from_args(self):
        args = {
            "path": "core/agent.py",
            "_context_updates": [{"tc1": "stale result"}, {"[tc2]": "summary"}],
        }
        updates = extract_context_updates(args)
        self.assertEqual(len(updates), 2)
        self.assertNotIn("_context_updates", args)

    def test_ensure_context_updates_arg_sets_default(self):
        args = {"path": "foo.py"}
        ensure_context_updates_arg(args)
        self.assertIn("_context_updates", args)
        self.assertEqual(args["_context_updates"], [])

    def test_apply_context_updates_rewrites_message(self):
        messages = [
            {"role": "tool", "content": label_tool_result("Tool output", 1), "tc_id": 1}
        ]
        result = apply_context_updates(messages, [{"tc1": "Short summary"}])
        self.assertEqual(result["applied"], 1)
        self.assertEqual(messages[0]["content"], "Short summary")
        self.assertTrue(messages[0]["is_compressed"])

    def test_infer_next_tc_id(self):
        messages = [
            {"role": "tool", "content": "[tc2] something"},
            {"role": "tool", "content": "already compressed", "compressed_from_tc": "tc4"},
        ]
        self.assertEqual(infer_next_tc_id(messages), 5)

    def test_auto_compress_dead_end_reads(self):
        messages = [
            {
                "role": "tool",
                "content": "[tc1] Tool 'read_file' Result: lots of code",
                "tool_name": "read_file",
                "tool_args": {"path": "src/a.py"},
                "tc_id": 1,
            },
            {"role": "assistant", "content": "Proceeding with other path."},
            {"role": "assistant", "content": "No mention of that file."},
            {"role": "assistant", "content": "Still no references."},
            {"role": "assistant", "content": "Working elsewhere."},
            {"role": "assistant", "content": "Continuing."},
            {"role": "assistant", "content": "Done."},
        ]
        compressed = auto_compress_dead_end_reads(messages, min_age_messages=3)
        self.assertIn("tc1", compressed)
        self.assertTrue(messages[0]["is_compressed"])
        self.assertNotIn("[tc1]", messages[0]["content"])


if __name__ == "__main__":
    unittest.main()
