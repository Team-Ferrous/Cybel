import re
import json


def debug_extraction(text):
    xml_pattern = r"<tool_call>\s*({.*})\s*</tool_call>"
    native_pattern = r"(?:\|)?tool\|>\s*({.*})\s*(?:<\||</?tool_call>|$)"

    print(f"Text: {repr(text)}")
    for name, pattern in [("XML", xml_pattern), ("Native", native_pattern)]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            content = match.group(1)
            print(f"  {name} Match found!")
            print(f"  Captured Group 1: {repr(content)}")
            try:
                val = json.loads(content)
                print(f"  JSON parsed successfully: {val}")
            except Exception as e:
                print(f"  JSON parse error: {e}")
        else:
            print(f"  {name} No match found.")


if __name__ == "__main__":
    debug_extraction(
        'tool|> {"name": "test_unquoted", "arguments": {"path": "."}} /tool_call>'
    )
