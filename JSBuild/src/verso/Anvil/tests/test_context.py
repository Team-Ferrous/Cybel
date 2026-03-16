import unittest
from core.context import ContextManager


class TestContextManager(unittest.TestCase):
    def setUp(self):
        self.cm = ContextManager(max_tokens=100, system_prompt_tokens=10)
        # available = 90 tokens

    def test_count_tokens(self):
        # Tokenizer backends may vary; count should still be monotonic and positive.
        text = "1234"
        small = self.cm.count_tokens(text)
        text = "12345678"
        large = self.cm.count_tokens(text)
        self.assertGreaterEqual(small, 1)
        self.assertGreaterEqual(large, small)

    def test_window_truncation(self):
        # Create messages
        # Message overhead = 4 tokens
        # "1234" (1 token) -> 5 tokens per message

        msgs = [{"role": "user", "content": "1234"} for _ in range(20)]
        # Total = 20 * 5 = 100 tokens. Available = 90.
        # Should drop first 2 messages? (10 tokens -> 90)

        window = self.cm.get_context_window(msgs)
        self.assertTrue(len(window) < 20)
        self.assertTrue(len(window) > 0)

        # Verify total tokens in window <= 90
        total = sum(self.cm.count_message_tokens(m) for m in window)
        self.assertTrue(total <= 90)

    def test_fill_percentage(self):
        msgs = [{"role": "user", "content": "hello"}]
        stats = self.cm.get_fill_percentage(msgs)
        self.assertIn("used_tokens", stats)
        self.assertIn("max_tokens", stats)
        self.assertIn("fill_percentage", stats)


if __name__ == "__main__":
    unittest.main()
