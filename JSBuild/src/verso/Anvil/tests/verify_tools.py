from tools.registry import ToolRegistry
import os


def test_registry():
    print("Testing ToolRegistry...")
    registry = ToolRegistry(".")

    # Test 1: Shell run_command
    print("Test 1: Shell run_command")
    res = registry.dispatch(
        "run_command",
        {
            "command": "echo 'Hello Tool' | grep 'Tool' > test_output.txt && cat test_output.txt"
        },
    )
    print(f"Result: {res}")
    assert "Hello Tool" in res, "Shell command failed"
    # cleanup
    if os.path.exists("test_output.txt"):
        os.remove("test_output.txt")

    # Test 2: File read (Targeting main.py which exists)
    print("Test 2: File read")
    res = registry.dispatch("read_file", {"path": "main.py", "end_line": 5})
    print(f"Result: {res}")
    assert "SAGUARO" in res or "main" in res or "import" in res, "File read failed"

    # Test 3: Saguaro Verify
    print("Test 3: Saguaro Verify")
    res = registry.dispatch("verify", {"path": "."})
    print(f"Result: len={len(res)}")
    # Saguaro verify outputs analysis
    assert (
        "Verify" in res
        or "drift" in res
        or "No violations" in res
        or "Scanning" in res
        or res.strip()
    ), "Verify returned empty result"

    print("Success: ToolRegistry verified.")


if __name__ == "__main__":
    test_registry()
