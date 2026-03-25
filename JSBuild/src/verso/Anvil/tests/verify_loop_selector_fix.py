import sys
import os

# Add project root to sys.path
sys.path.append(os.getcwd())

from domains.task_execution.enhanced_loop import LoopSelector


def test_selector():
    test_cases = [
        ("How do I list files?", "simple"),
        ("What is the weather?", "simple"),
        ("Analyze the repository structure.", "enhanced"),
        ("Review the current codebase for security issues.", "enhanced"),
        ("Summarize the main logic in agents/worker.py", "enhanced"),
        ("Explain how the orchestrator works.", "simple"),
        ("Implement a new feature to support deep indexing.", "enhanced"),
    ]

    passed = 0
    for query, expected in test_cases:
        result = LoopSelector.select(query)
        if result == expected:
            print(f"✅ PASS: '{query}' -> {result}")
            passed += 1
        else:
            print(f"❌ FAIL: '{query}' -> Expected {expected}, got {result}")

    print(f"\nSummary: {passed}/{len(test_cases)} passed.")
    return passed == len(test_cases)


if __name__ == "__main__":
    if not test_selector():
        sys.exit(1)
