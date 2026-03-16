from tools.registry import ToolRegistry


def verify_upgrade_tool():
    print("Testing Upgrade Tool Dispatch...")
    try:
        registry = ToolRegistry(".")
        # Dry run check
        res = registry.dispatch("upgrade", {"action": "check"})
        print(f"Upgrade Check Result: {res}")
        assert "Status" in res, "Upgrade check failed"
        print("Success.")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    verify_upgrade_tool()
