from typing import Dict, Any, List, Optional, Callable


class ToolResultValidator:
    """
    Validates tool outputs against schemas before passing back to agents.
    Prevents "garbage in, garbage out" scenarios.
    """

    def __init__(self):
        # Result schemas for core tools
        # Format: {tool_name: validator_func}
        self.validators: Dict[str, Callable[[Any], List[str]]] = {
            "read_file": self._validate_read_file,
            "list_dir": self._validate_list_dir,
            "grep": self._validate_grep,
        }

    def validate(self, tool_name: str, result: Any) -> Optional[str]:
        """
        Validates result and returns an error message if invalid, else None.
        """
        if tool_name not in self.validators:
            return None  # No validation for this tool

        errors = self.validators[tool_name](result)
        if errors:
            return f"Error: Tool '{tool_name}' returned invalid format: {', '.join(errors)}"

        return None

    def _validate_read_file(self, result: Any) -> List[str]:
        # read_file should return a string
        if not isinstance(result, str):
            return ["Result must be a string"]
        return []

    def _validate_list_dir(self, result: Any) -> List[str]:
        # list_dir should return a list of strings
        if not isinstance(result, list):
            return ["Result must be a list"]
        if result and not all(isinstance(x, str) for x in result):
            return ["All elements in list must be strings"]
        return []

    def _validate_grep(self, result: Any) -> List[str]:
        # grep should return a string (concatenated matches)
        if not isinstance(result, str):
            return ["Result must be a string"]
        return []
