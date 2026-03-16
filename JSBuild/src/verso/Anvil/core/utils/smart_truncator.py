"""
Smart Truncator for tool outputs.
Provides Claude Code-style progressive disclosure for large tool results.
"""

from typing import Dict, Any


class SmartTruncator:
    """
    Handles large tool outputs by providing summaries and windowed views.
    """

    def __init__(self, char_threshold: int = 100000, context_name: str = "general"):
        self.char_threshold = char_threshold
        self.context_name = context_name

    def truncate(self, name: str, args: Dict[str, Any], result: str) -> str:
        """
        Main entry point for smart truncation.
        """
        # Grounding-critical tools should not be truncated by default.
        full_visibility_tools = {
            "read_file",
            "read_files",
            "slice",
            "skeleton",
            "query",
            "saguaro_query",
        }
        if name in full_visibility_tools and "max_chars" not in args:
            return result

        # Honor explicit max_chars request if provided by the tool caller
        effective_threshold = args.get("max_chars", self.char_threshold)
        if not isinstance(effective_threshold, int) or effective_threshold <= 0:
            effective_threshold = self.char_threshold

        if len(result) <= effective_threshold:
            return result

        # Routing based on tool name
        if name == "list_dir":
            return self._truncate_list_dir(result)
        elif name in ["read_file", "read_files"]:
            return self._truncate_file_content(args, result)
        elif name == "grep":
            return self._truncate_search_results(result)
        elif name == "run_command":
            return self._truncate_command_output(result)

        # Generic truncation
        return self._generic_truncate(result)

    def _truncate_list_dir(self, result: str) -> str:
        lines = result.splitlines()
        total_items = len(lines)
        head = lines[:100]
        tail = lines[-10:]

        summary = (
            f"\n\n[TRUNCATED: {total_items} items total. Showing first 100 entries.]\n"
            f"[ADVICE: Use 'grep' to search for specific files, or 'list_dir' on a specific subdirectory to see more.]\n"
        )
        return (
            "\n".join(head) + summary + "\n".join(tail) if total_items > 110 else result
        )

    def _truncate_file_content(self, args: Dict[str, Any], result: str) -> str:
        lines = result.splitlines()
        total_lines = len(lines)
        file_path = args.get("path", args.get("file_path", "unknown file"))

        # Keep more context for files
        head = lines[:200]
        tail = lines[-50:]

        summary = (
            f"\n\n[TRUNCATED: File '{file_path}' has {total_lines} lines total. "
            f"Showing lines 1-200 and last 50 lines.]\n"
            f"[PROGRESSIVE DISCLOSURE: To read a specific range, use 'read_file(path=\"{file_path}\", start_line=X, end_line=Y)'. "
            f"Use 'grep' to find specific patterns or 'skeleton' to see structure.]\n\n"
        )
        return "\n".join(head) + summary + "\n".join(tail)

    def _truncate_search_results(self, result: str) -> str:
        lines = result.splitlines()
        total_matches = len(lines)
        head = lines[:100]

        summary = (
            f"\n\n[TRUNCATED: Found {total_matches} matches. Showing first 100.]\n"
            f"[ADVICE: Narrow your search with a more specific regex or by targeting a subdirectory.]\n"
        )
        return "\n".join(head) + summary

    def _truncate_command_output(self, result: str) -> str:
        # Commands often have important stack traces at the end
        if len(result) < 15000:
            return result

        head = result[:5000]
        tail = result[-5000:]

        summary = (
            f"\n\n... [TRUNCATED: Output is {len(result)} characters. "
            f"Showing first 5KB and last 5KB of output.] ...\n"
            f"[ADVICE: If you need to see the full output, consider redirecting to a file or using 'grep' on the command output.]\n\n"
        )
        return head + summary + tail

    def _generic_truncate(self, result: str) -> str:
        head = result[: self.char_threshold // 2]
        tail = result[-(self.char_threshold // 4) :]

        summary = f"\n\n... [TRUNCATED: Large output ({len(result)} characters). Showing head and tail segments.] ...\n\n"
        return head + summary + tail
