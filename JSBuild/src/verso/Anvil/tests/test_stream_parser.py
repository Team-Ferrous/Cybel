import unittest
from core.response_utils import ResponseStreamParser


class TestResponseStreamParser(unittest.TestCase):
    def test_basic_content(self):
        parser = ResponseStreamParser()
        # "Hello" is 5 chars, less than 6 safety buffer
        events = list(parser.process_chunk("Hello"))
        self.assertEqual(len(events), 0)

        # Adding " world" makes it 11 chars. 11 - 6 = 5 chars yielded
        events = list(parser.process_chunk(" world"))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].content, "Hello")

        events = list(parser.finalize())
        self.assertEqual(events[0].content, " world")

    def test_thinking_detection(self):
        parser = ResponseStreamParser()
        # "Hello "<thinking type="planning">
        events = list(parser.process_chunk('Hello <thinking type="planning">'))
        # Should yield "Hello " and thinking_start
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].type, "content")
        self.assertEqual(events[0].content, "Hello ")
        self.assertEqual(events[1].type, "thinking_start")

        # "Plan step 1</thinking> Done"
        events = list(parser.process_chunk("Plan step 1</thinking> Done"))
        # Should yield thinking_chunk ("Plan step 1") and thinking_end
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].type, "thinking_chunk")
        self.assertEqual(events[0].content, "Plan step 1")
        self.assertEqual(events[1].type, "thinking_end")

        events += list(parser.finalize())
        self.assertTrue(any(e.content == " Done" for e in events))

    def test_split_tags(self):
        parser = ResponseStreamParser()
        # Split tag: "<thin", "king>Test</thinking>"
        events = list(parser.process_chunk("<thin"))
        self.assertEqual(len(events), 0)

        events = list(parser.process_chunk("king>Test</thinking>"))
        # Should yield thinking_start, thinking_chunk, thinking_end
        self.assertEqual(len(events), 3)
        self.assertEqual(events[0].type, "thinking_start")
        self.assertEqual(events[1].type, "thinking_chunk")
        self.assertEqual(events[1].content, "Test")
        self.assertEqual(events[2].type, "thinking_end")

    def test_role_tag_stripping(self):
        parser = ResponseStreamParser()
        events = list(parser.process_chunk("Content<|end_of_text|>More"))
        events += list(parser.finalize())
        # Should contain "Content" and "More" but NOT "<|end_of_text|>"
        full_content = "".join(e.content for e in events if e.type == "content")
        self.assertIn("Content", full_content)
        self.assertIn("More", full_content)
        self.assertNotIn("<|end_of_text|>", full_content)


if __name__ == "__main__":
    unittest.main()
