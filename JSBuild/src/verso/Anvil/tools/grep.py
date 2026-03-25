def grep(
    pattern: str, path: str = ".", is_regex: bool = False, file_pattern: str = "*"
) -> str:
    """Disabled in strict grounding mode."""
    return (
        "Error: grep is disabled in strict grounding mode. "
        "Use saguaro_query for discovery, then skeleton/slice/read_file for verification."
    )
