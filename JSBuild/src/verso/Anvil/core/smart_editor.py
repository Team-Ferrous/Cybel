"""
Smart File Editor (Claude Code Style)

Key Features:
1. Minimal diffs - only change what's needed
2. Automatic syntax validation
3. Preserves formatting and style
4. Clear visualization of changes
5. Rollback on error
"""

import difflib
import re
import ast
from typing import Any, Callable, Dict, List, Optional
from rich.syntax import Syntax
from rich.panel import Panel


class SmartFileEditor:
    """
    Intelligent file editing with minimal diffs and validation.
    """

    def __init__(
        self,
        registry,
        console,
        tool_executor: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
    ):
        self.registry = registry
        self.console = console
        self.backup_cache = {}
        self.tool_executor = tool_executor

    def _dispatch(self, name: str, arguments: Dict[str, Any]) -> Any:
        if self.tool_executor is not None:
            return self.tool_executor(name, arguments)
        return self.registry.dispatch(name, arguments)

    def edit_with_instruction(
        self, file_path: str, instruction: str, context: str = ""
    ) -> bool:
        """
        Edit a file based on natural language instruction.

        Args:
            file_path: Path to file to edit
            instruction: Natural language edit instruction
            context: Additional context about the change

        Returns:
            True if successful, False otherwise
        """
        self.console.print(f"[cyan]Editing {file_path}...[/cyan]")

        # 1. Read current file
        current_content = self._dispatch("read_file", {"file_path": file_path})
        if not current_content or current_content.startswith("Error"):
            self.console.print(f"[red]Failed to read file: {current_content}[/red]")
            return False

        # 2. Backup current version
        self._backup_file(file_path, current_content)

        # 3. Generate edit using model
        try:
            new_content = self._generate_edit(
                file_path, current_content, instruction, context
            )

            # 4. Validate new content
            if not self._validate_content(file_path, new_content):
                self.console.print("[red]Validation failed, rolling back[/red]")
                return False

            # 5. Show diff
            self._show_diff(file_path, current_content, new_content)

            # 6. Apply changes
            result = self._dispatch(
                "write_file", {"file_path": file_path, "content": new_content}
            )

            if result and not result.startswith("Error"):
                self.console.print(f"[green]✓ Successfully edited {file_path}[/green]")
                return True
            else:
                self.console.print(f"[red]Failed to write file: {result}[/red]")
                self._restore_backup(file_path)
                return False

        except Exception as e:
            self.console.print(f"[red]Error during edit: {e}[/red]")
            self._restore_backup(file_path)
            return False

    def find_and_replace(
        self,
        file_path: str,
        search_pattern: str,
        replace_with: str,
        use_regex: bool = False,
        max_replacements: int = -1,
    ) -> bool:
        """
        Find and replace with smart matching.

        Args:
            file_path: File to edit
            search_pattern: Pattern to find
            replace_with: Replacement text
            use_regex: Whether to use regex matching
            max_replacements: Max replacements (-1 for all)

        Returns:
            True if successful
        """
        current_content = self._dispatch("read_file", {"file_path": file_path})
        if not current_content or current_content.startswith("Error"):
            return False

        self._backup_file(file_path, current_content)

        # Perform replacement
        if use_regex:
            new_content = re.sub(
                search_pattern,
                replace_with,
                current_content,
                count=max_replacements if max_replacements > 0 else 0,
            )
        else:
            if max_replacements == -1:
                new_content = current_content.replace(search_pattern, replace_with)
            else:
                new_content = current_content.replace(
                    search_pattern, replace_with, max_replacements
                )

        # Check if anything changed
        if new_content == current_content:
            self.console.print(
                f"[yellow]No matches found for '{search_pattern}'[/yellow]"
            )
            return False

        # Validate and apply
        if self._validate_content(file_path, new_content):
            self._show_diff(file_path, current_content, new_content)
            self._dispatch(
                "write_file", {"file_path": file_path, "content": new_content}
            )
            self.console.print(f"[green]✓ Replaced in {file_path}[/green]")
            return True
        else:
            self._restore_backup(file_path)
            return False

    def insert_at_location(
        self,
        file_path: str,
        location: Dict[str, any],
        content: str,
        position: str = "after",
    ) -> bool:
        """
        Insert content at a specific location in the file.

        Args:
            file_path: File to edit
            location: How to find the location (line number, function name, etc.)
            content: Content to insert
            position: "before" or "after" the location

        Returns:
            True if successful
        """
        current_content = self._dispatch("read_file", {"file_path": file_path})
        if not current_content or current_content.startswith("Error"):
            return False

        self._backup_file(file_path, current_content)

        lines = current_content.split("\n")

        # Find insertion point
        if "line_number" in location:
            insert_line = location["line_number"]
            if position == "after":
                insert_line += 1
        elif "function_name" in location:
            # Find function and insert after its definition
            insert_line = self._find_function_end(lines, location["function_name"])
        else:
            self.console.print("[red]Invalid location specifier[/red]")
            return False

        # Insert content
        lines.insert(insert_line, content)
        new_content = "\n".join(lines)

        # Validate and apply
        if self._validate_content(file_path, new_content):
            self._show_diff(file_path, current_content, new_content)
            self._dispatch(
                "write_file", {"file_path": file_path, "content": new_content}
            )
            self.console.print(f"[green]✓ Inserted content in {file_path}[/green]")
            return True
        else:
            self._restore_backup(file_path)
            return False

    def _generate_edit(
        self, file_path: str, current_content: str, instruction: str, context: str
    ) -> str:
        """
        Use model to generate the edited version.
        """
        # Import here to avoid circular dependency
        from core.ollama_client import DeterministicOllama
        from config.settings import MASTER_MODEL

        brain = DeterministicOllama(MASTER_MODEL)

        prompt = f"""You are a precise code editor. You will receive:
1. A file's current content
2. An instruction for what to change
3. Optional context

Your task: Output the COMPLETE edited file with the changes applied.

File: {file_path}

Instruction: {instruction}

Context: {context}

Current Content:
```
{current_content}
```

Output the complete edited file (with the changes applied). Output ONLY the file content, no explanations.

Edited File:
```"""

        messages = [
            {"role": "system", "content": "You are a precise code editor."},
            {"role": "user", "content": prompt},
        ]

        response = ""
        for chunk in brain.stream_chat(messages, max_tokens=20000, temperature=0.0):
            response += chunk

        # Extract code from markdown if present
        code_match = re.search(
            r"```(?:python|javascript|typescript|java|cpp|c|go|rust|)?\n(.*?)```",
            response,
            re.DOTALL,
        )
        if code_match:
            return code_match.group(1).strip()

        return response.strip()

    def _validate_content(self, file_path: str, content: str) -> bool:
        """
        Validate the new content before applying.

        Checks:
        - Syntax (for Python files)
        - Indentation
        - Basic structure
        """
        # Python syntax validation
        if file_path.endswith(".py"):
            try:
                ast.parse(content)
                return True
            except SyntaxError as e:
                self.console.print(f"[red]Syntax error: {e}[/red]")
                return False

        # For other languages, just basic checks
        # TODO: Add validators for other languages

        return True

    def _show_diff(self, file_path: str, old_content: str, new_content: str):
        """
        Display a beautiful diff of changes.
        """
        old_lines = old_content.split("\n")
        new_lines = new_content.split("\n")

        # Generate unified diff
        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"{file_path} (before)",
            tofile=f"{file_path} (after)",
            lineterm="",
        )

        diff_text = "\n".join(diff)

        if diff_text:
            # Count changes
            additions = diff_text.count("\n+")
            deletions = diff_text.count("\n-")

            self.console.print(
                Panel(
                    Syntax(diff_text, "diff", theme="monokai"),
                    title=f"[bold]Changes to {file_path}[/bold]",
                    subtitle=f"[green]+{additions}[/green] [red]-{deletions}[/red]",
                    border_style="cyan",
                )
            )

    def _backup_file(self, file_path: str, content: str):
        """Store backup of file in memory."""
        self.backup_cache[file_path] = content

    def _restore_backup(self, file_path: str):
        """Restore file from backup."""
        if file_path in self.backup_cache:
            self._dispatch(
                "write_file",
                {"file_path": file_path, "content": self.backup_cache[file_path]},
            )
            self.console.print(f"[yellow]Restored {file_path} from backup[/yellow]")

    def _find_function_end(self, lines: List[str], function_name: str) -> int:
        """
        Find the line number where a function ends.

        Simple heuristic:
        - Find 'def function_name'
        - Find next line at same or lower indentation level
        """
        in_function = False
        function_indent = 0

        for i, line in enumerate(lines):
            if f"def {function_name}" in line:
                in_function = True
                # Calculate indentation
                function_indent = len(line) - len(line.lstrip())
                continue

            if in_function:
                current_indent = len(line) - len(line.lstrip())
                # If we hit a line at same/lower indent and it's not blank, function ended
                if line.strip() and current_indent <= function_indent:
                    return i

        return len(lines)  # If not found, insert at end


class MultiFileEditor:
    """
    Coordinate edits across multiple files.
    Ensures consistency (imports, function signatures, etc.)
    """

    def __init__(self, smart_editor: SmartFileEditor):
        self.smart_editor = smart_editor
        self.console = smart_editor.console

    def refactor_function_signature(
        self, function_name: str, new_signature: str
    ) -> bool:
        """
        Refactor a function signature across all files that use it.

        1. Find function definition
        2. Update definition
        3. Find all call sites
        4. Update call sites to match new signature
        """
        self.console.print(f"[cyan]Refactoring function: {function_name}[/cyan]")

        # TODO: Implement cross-file refactoring
        # This requires:
        # - AST-based analysis
        # - Call graph construction
        # - Coordinated edits

        return False

    def rename_class(self, old_name: str, new_name: str) -> bool:
        """
        Rename a class across entire codebase.

        1. Find class definition
        2. Rename definition
        3. Update all imports
        4. Update all usages
        """
        self.console.print(f"[cyan]Renaming class: {old_name} → {new_name}[/cyan]")

        # TODO: Implement class renaming
        # This is complex - need to handle:
        # - Class definition
        # - Imports (from x import OldName)
        # - Direct usage (obj = OldName())
        # - Type hints (def foo(x: OldName))

        return False
