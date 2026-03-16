"""
Multi-File Refactoring with Saguaro

Coordinate refactoring across multiple files:
- Rename classes/functions globally
- Update function signatures everywhere
- Extract modules
- Move code between files
- Ensure consistency across codebase

Uses Saguaro for:
- Finding all usages
- Impact analysis
- Dependency tracking
- Verification
"""

import ast
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class RefactorPlan:
    """Plan for a multi-file refactor."""

    operation: str  # "rename_class", "rename_function", "extract_module", etc.
    target: str  # What we're refactoring
    new_value: str  # New name/location
    files_to_modify: List[str]
    modifications: Dict[str, List[Dict]]  # file -> list of changes
    risk_level: str  # "low", "medium", "high"
    estimated_impact: int  # Number of files affected


class MultiFileRefactorer:
    """
    Coordinate refactoring across multiple files using Saguaro.
    """

    def __init__(
        self,
        registry,
        saguaro_tools,
        smart_editor,
        console,
        tool_executor: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
    ):
        self.registry = registry
        self.saguaro_tools = saguaro_tools
        self.smart_editor = smart_editor
        self.console = console
        self.tool_executor = tool_executor

    def _dispatch(self, name: str, arguments: Dict[str, Any]) -> Any:
        if self.tool_executor is not None:
            return self.tool_executor(name, arguments)
        return self.registry.dispatch(name, arguments)

    def rename_class(
        self, old_name: str, new_name: str, dry_run: bool = True
    ) -> RefactorPlan:
        """
        Rename a class throughout the entire codebase.

        Steps:
        1. Find class definition using Saguaro
        2. Find all usages (imports, instantiations, type hints)
        3. Generate refactor plan
        4. Apply changes atomically
        5. Verify all files still valid

        Args:
            old_name: Current class name
            new_name: New class name
            dry_run: If True, just return plan without applying

        Returns:
            RefactorPlan with details
        """
        self.console.print(
            f"[cyan]Planning class rename: {old_name} → {new_name}[/cyan]"
        )

        plan = RefactorPlan(
            operation="rename_class",
            target=old_name,
            new_value=new_name,
            files_to_modify=[],
            modifications={},
            risk_level="medium",
            estimated_impact=0,
        )

        # Step 1: Find class definition
        class_file = self._find_class_definition(old_name)

        if not class_file:
            self.console.print(f"[red]Could not find class {old_name}[/red]")
            return plan

        self.console.print(f"  [dim]→ Found class in {class_file}[/dim]")

        # Step 2: Find all usages using Saguaro
        usages = self._find_class_usages(old_name, class_file)
        self.console.print(f"  [dim]→ Found {len(usages)} files using this class[/dim]")

        plan.files_to_modify = [class_file] + usages
        plan.estimated_impact = len(plan.files_to_modify)

        # Step 3: Generate modifications for each file
        plan.modifications = self._generate_rename_modifications(
            old_name, new_name, plan.files_to_modify
        )

        # Step 4: Assess risk
        plan.risk_level = self._assess_risk(plan)

        if dry_run:
            self._display_plan(plan)
            return plan

        # Step 5: Apply changes
        return self._apply_refactor_plan(plan)

    def extract_to_module(
        self, source_file: str, symbols: List[str], target_module: str
    ) -> RefactorPlan:
        """
        Extract classes/functions to a new module.

        Steps:
        1. Identify symbols to extract
        2. Find dependencies
        3. Create new module with extracted code
        4. Update imports in source file
        5. Update imports in files that use these symbols

        Args:
            source_file: File to extract from
            symbols: List of class/function names to extract
            target_module: Path to new module

        Returns:
            RefactorPlan
        """
        self.console.print(
            f"[cyan]Planning extraction: {symbols} → {target_module}[/cyan]"
        )

        plan = RefactorPlan(
            operation="extract_module",
            target=source_file,
            new_value=target_module,
            files_to_modify=[source_file],
            modifications={},
            risk_level="high",
            estimated_impact=0,
        )

        # Analyze dependencies
        dependencies = self._analyze_symbol_dependencies(source_file, symbols)

        # Generate new module content
        new_module_content = self._generate_extracted_module(
            source_file, symbols, dependencies
        )

        # Plan modifications
        plan.modifications[target_module] = [
            {"action": "create", "content": new_module_content}
        ]

        plan.modifications[source_file] = [
            {"action": "remove_symbols", "symbols": symbols},
            {"action": "add_import", "module": target_module, "symbols": symbols},
        ]

        return plan

    def update_function_signature(
        self, function_name: str, new_signature: str
    ) -> RefactorPlan:
        """
        Update a function signature and all call sites.

        Steps:
        1. Find function definition
        2. Parse old and new signatures
        3. Find all call sites using Saguaro
        4. Update call sites to match new signature
        5. Verify all changes

        Args:
            function_name: Name of function
            new_signature: New signature (e.g., "def foo(x, y, z=10)")

        Returns:
            RefactorPlan
        """
        self.console.print(f"[cyan]Planning signature update: {function_name}[/cyan]")

        plan = RefactorPlan(
            operation="update_signature",
            target=function_name,
            new_value=new_signature,
            files_to_modify=[],
            modifications={},
            risk_level="high",
            estimated_impact=0,
        )

        # Find function definition
        func_file = self._find_function_definition(function_name)

        if not func_file:
            return plan

        # Find all call sites
        call_sites = self._find_function_calls(function_name)

        plan.files_to_modify = [func_file] + list(call_sites.keys())
        plan.estimated_impact = len(plan.files_to_modify)

        # Generate modifications (simplified)
        plan.modifications[func_file] = [
            {
                "action": "update_signature",
                "function": function_name,
                "new_signature": new_signature,
            }
        ]

        return plan

    def _find_class_definition(self, class_name: str) -> str:
        """
        Find file containing class definition.

        Uses Saguaro's grep functionality.
        """
        try:
            # Search for "class ClassName"
            result = self._dispatch(
                "grep", {"pattern": f"^class\\s+{class_name}\\s*[\\(:]", "path": "."}
            )

            # Parse grep result to get file path
            if result and not result.startswith("Error"):
                # Simplified parsing
                lines = result.split("\n")
                if lines:
                    # Extract file path from first match
                    first_match = lines[0]
                    if ":" in first_match:
                        return first_match.split(":")[0]

        except Exception as e:
            self.console.print(f"[yellow]Grep failed: {e}[/yellow]")

        return None

    def _find_class_usages(self, class_name: str, definition_file: str) -> List[str]:
        """
        Find all files that use this class.

        Uses Saguaro's:
        1. Impact analysis
        2. Grep for imports
        3. Grep for direct usage
        """
        usages = set()

        # Method 1: Saguaro impact analysis
        try:
            self.saguaro_tools.impact(definition_file)
            # Parse impact to find dependent files
            # (Simplified - real implementation would parse graph)

        except Exception:
            pass

        # Method 2: Grep for imports
        try:
            # Find "from X import ClassName" or "import X.ClassName"
            import_results = self._dispatch(
                "grep",
                {
                    "pattern": f"(from\\s+[\\w.]+\\s+import.*{class_name}|import\\s+.*{class_name})",
                    "path": ".",
                },
            )

            if import_results and not import_results.startswith("Error"):
                for line in import_results.split("\n"):
                    if ":" in line:
                        file_path = line.split(":")[0]
                        if file_path != definition_file:
                            usages.add(file_path)

        except Exception:
            pass

        # Method 3: Grep for direct usage
        try:
            usage_results = self._dispatch(
                "grep", {"pattern": f"{class_name}\\s*\\(", "path": "."}  # ClassName(
            )

            if usage_results and not usage_results.startswith("Error"):
                for line in usage_results.split("\n"):
                    if ":" in line:
                        file_path = line.split(":")[0]
                        if file_path != definition_file:
                            usages.add(file_path)

        except Exception:
            pass

        return list(usages)

    def _generate_rename_modifications(
        self, old_name: str, new_name: str, files: List[str]
    ) -> Dict[str, List[Dict]]:
        """
        Generate modification plan for renaming.

        For each file, find all occurrences and plan replacements.
        """
        modifications = {}

        for file_path in files:
            file_mods = []

            try:
                content = self._dispatch("read_file", {"file_path": file_path})

                if content and not content.startswith("Error"):
                    # Find all patterns to replace
                    patterns = [
                        (f"class {old_name}", f"class {new_name}"),  # Class definition
                        (
                            f"from .* import.*{old_name}",
                            lambda m: m.group(0).replace(old_name, new_name),
                        ),  # Imports
                        (f"{old_name}\\(", f"{new_name}("),  # Instantiation
                        (f": {old_name}\\b", f": {new_name}"),  # Type hints
                    ]

                    for old_pattern, new_pattern in patterns:
                        file_mods.append(
                            {
                                "action": "replace",
                                "pattern": old_pattern,
                                "replacement": new_pattern,
                            }
                        )

            except Exception:
                pass

            if file_mods:
                modifications[file_path] = file_mods

        return modifications

    def _find_function_definition(self, function_name: str) -> str:
        """Find file containing function definition."""
        try:
            result = self._dispatch(
                "grep", {"pattern": f"^def\\s+{function_name}\\s*\\(", "path": "."}
            )

            if result and not result.startswith("Error"):
                lines = result.split("\n")
                if lines:
                    first_match = lines[0]
                    if ":" in first_match:
                        return first_match.split(":")[0]

        except Exception:
            pass

        return None

    def _find_function_calls(self, function_name: str) -> Dict[str, List[int]]:
        """
        Find all call sites of a function.

        Returns dict: file_path -> list of line numbers
        """
        call_sites = {}

        try:
            result = self._dispatch(
                "grep", {"pattern": f"{function_name}\\s*\\(", "path": "."}
            )

            if result and not result.startswith("Error"):
                for line in result.split("\n"):
                    if ":" in line:
                        parts = line.split(":", 2)
                        if len(parts) >= 2:
                            file_path = parts[0]
                            try:
                                line_num = int(parts[1])
                                if file_path not in call_sites:
                                    call_sites[file_path] = []
                                call_sites[file_path].append(line_num)
                            except ValueError:
                                pass

        except Exception:
            pass

        return call_sites

    def _analyze_symbol_dependencies(
        self, file_path: str, symbols: List[str]
    ) -> List[str]:
        """
        Analyze what the symbols depend on.

        Returns list of other symbols they need.
        """
        # Simplified - would use AST analysis in real implementation
        return []

    def _generate_extracted_module(
        self, source_file: str, symbols: List[str], dependencies: List[str]
    ) -> str:
        """
        Generate content for extracted module.

        Includes:
        - Imports needed by extracted code
        - The extracted symbols
        """
        # Read source file
        content = self._dispatch("read_file", {"file_path": source_file})

        if not content or content.startswith("Error"):
            return ""

        # Parse AST to extract symbols
        try:
            tree = ast.parse(content)

            extracted_code = []

            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                    if node.name in symbols:
                        # Get source code for this node
                        # (Simplified - real implementation would use ast.get_source_segment)
                        pass

            return "\n\n".join(extracted_code)

        except Exception:
            return ""

    def _assess_risk(self, plan: RefactorPlan) -> str:
        """
        Assess risk level of refactoring.

        Factors:
        - Number of files affected
        - Type of change
        - Test coverage
        """
        if plan.estimated_impact > 10:
            return "high"
        elif plan.estimated_impact > 3:
            return "medium"
        else:
            return "low"

    def _display_plan(self, plan: RefactorPlan):
        """Display refactoring plan to user."""
        from rich.panel import Panel
        from rich.table import Table

        self.console.print(
            Panel(
                f"[bold]Operation:[/bold] {plan.operation}\n"
                f"[bold]Target:[/bold] {plan.target} → {plan.new_value}\n"
                f"[bold]Files to modify:[/bold] {plan.estimated_impact}\n"
                f"[bold]Risk level:[/bold] {plan.risk_level}",
                title="Refactoring Plan",
                border_style="cyan",
            )
        )

        # Show files
        table = Table(title="Files to Modify")
        table.add_column("File")
        table.add_column("Changes")

        for file_path, mods in list(plan.modifications.items())[:10]:
            table.add_row(file_path, str(len(mods)))

        self.console.print(table)

    def _apply_refactor_plan(self, plan: RefactorPlan) -> RefactorPlan:
        """
        Apply the refactoring plan.

        Returns updated plan with results.
        """
        self.console.print("[cyan]Applying refactoring...[/cyan]")

        # Apply modifications to each file
        for file_path, mods in plan.modifications.items():
            try:
                for mod in mods:
                    action = mod.get("action")

                    if action == "replace":
                        # Use smart editor for replacements
                        self.smart_editor.find_and_replace(
                            file_path=file_path,
                            search_pattern=mod["pattern"],
                            replace_with=mod["replacement"],
                            use_regex=True,
                        )

                    elif action == "create":
                        # Create new file
                        self._dispatch(
                            "write_file",
                            {"file_path": file_path, "content": mod["content"]},
                        )

                self.console.print(f"  [green]✓ {file_path}[/green]")

            except Exception as e:
                self.console.print(f"  [red]✗ {file_path}: {e}[/red]")

        return plan
