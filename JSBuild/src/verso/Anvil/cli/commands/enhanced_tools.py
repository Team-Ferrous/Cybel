"""
Enhanced Tool Calling Commands

Commands for the Claude Code-style enhanced tool calling loop.
"""

from cli.commands.base import SlashCommand
from rich.panel import Panel
from rich.table import Table


class EnhancedToolsCommand(SlashCommand):
    """Toggle or run enhanced tool calling mode."""

    def name(self):
        return "enhanced"

    def description(self):
        return "Use Claude Code-style enhanced tool calling loop"

    def help_text(self):
        return """Enhanced Tool Calling Loop

Usage:
  /enhanced on          Enable enhanced mode
  /enhanced off         Disable enhanced mode
  /enhanced <task>      Run task with enhanced loop
  /enhanced status      Show enhanced mode status

Features:
  - Structured JSON tool calling
  - Parallel tool execution
  - Progressive context loading
  - Smart context management with Saguaro
  - Task memory and learning
  - Multi-file refactoring
  - Automatic verification
  - Think-Act-Observe cycle

Examples:
  /enhanced on
  /enhanced Refactor the authentication module
  /enhanced status
"""

    def execute(self, args: str, agent):
        args = args.strip()

        if not args or args == "status":
            self._show_status(agent)
        elif args == "on":
            agent.use_enhanced_tools = True
            agent.console.print(
                Panel(
                    "[green]✓ Enhanced tool calling mode enabled[/green]\n\n"
                    "All coding tasks will now use:\n"
                    "- Structured tool calling\n"
                    "- Parallel execution\n"
                    "- Progressive context\n"
                    "- Task memory\n"
                    "- Auto-verification",
                    title="Enhanced Mode",
                    border_style="green",
                )
            )
        elif args == "off":
            agent.use_enhanced_tools = False
            agent.console.print("[yellow]Enhanced tool calling mode disabled[/yellow]")
        else:
            # Run task with enhanced loop
            self._run_enhanced(args, agent)

        return True

    def _show_status(self, agent):
        """Show enhanced mode status."""
        enabled = getattr(agent, "use_enhanced_tools", False)
        enhanced_loop_enabled = getattr(agent, "enhanced_loop_enabled", True)

        if enabled:
            status_text = "[green]Enabled (Claude Code mode)[/green]"
            border_style = "green"
        elif enhanced_loop_enabled:
            status_text = "[cyan]Unified Loop (enhanced features active)[/cyan]"
            border_style = "cyan"
        else:
            status_text = "[yellow]Basic mode[/yellow]"
            border_style = "yellow"

        stats = ""

        # Check unified loop stats
        unified_loop = getattr(agent, "_unified_loop", None)
        if unified_loop:
            loop_stats = unified_loop.get_stats()
            stats += "\n\nUnified Loop Stats:"
            stats += f"\n  Files read: {loop_stats['files_read']}"
            stats += f"\n  Files edited: {loop_stats['files_edited']}"
            stats += f"\n  Task memories: {loop_stats['task_memories']}"

        # Check enhanced tool loop stats
        enhanced_loop = getattr(agent, "_enhanced_tool_loop", None)
        if enhanced_loop and hasattr(enhanced_loop, "memory_manager"):
            memory_count = len(enhanced_loop.memory_manager.memories)
            stats += f"\n\nEnhanced Tool Loop memories: {memory_count}"

        agent.console.print(
            Panel(
                f"Status: {status_text}{stats}",
                title="Enhanced Tool Calling",
                border_style=border_style,
            )
        )

    def _run_enhanced(self, task: str, agent):
        """Run a task with the enhanced tool calling loop."""
        agent.console.print(
            Panel(
                f"[bold cyan]Running with Enhanced Tool Calling Loop[/bold cyan]\n\n"
                f"Task: {task}",
                border_style="cyan",
            )
        )

        # Lazy-load the enhanced tool calling loop
        if not hasattr(agent, "_enhanced_tool_loop"):
            from core.enhanced_tool_calling_loop import EnhancedToolCallingLoop

            agent._enhanced_tool_loop = EnhancedToolCallingLoop(agent)

        # Run the task
        result = agent._enhanced_tool_loop.run(task)

        agent.console.print(
            Panel(result, title="Enhanced Loop Result", border_style="green")
        )


class ToolMemoryCommand(SlashCommand):
    """View and manage tool calling task memory."""

    def name(self):
        return "toolmemory"

    def description(self):
        return "View task memory from enhanced tool calling"

    def help_text(self):
        return """Tool Memory Management

Usage:
  /toolmemory               List all task memories
  /toolmemory <type>        Filter by task type (edit, create, refactor, etc.)
  /toolmemory similar <q>   Find similar tasks
  /toolmemory patterns <t>  Show success patterns for type

Examples:
  /toolmemory
  /toolmemory edit
  /toolmemory similar add authentication
  /toolmemory patterns refactor
"""

    def execute(self, args: str, agent):
        # Try to get memory manager from unified loop first, then enhanced loop
        memory_manager = None

        unified_loop = getattr(agent, "_unified_loop", None)
        if unified_loop and hasattr(unified_loop, "memory_manager"):
            memory_manager = unified_loop.memory_manager

        if memory_manager is None:
            enhanced_loop = getattr(agent, "_enhanced_tool_loop", None)
            if enhanced_loop and hasattr(enhanced_loop, "memory_manager"):
                memory_manager = enhanced_loop.memory_manager

        if memory_manager is None:
            agent.console.print(
                "[yellow]No task memory available yet. Run some tasks first![/yellow]"
            )
            return True

        args = args.strip()

        if not args:
            self._list_all(memory_manager, agent)
        elif args.startswith("similar "):
            query = args[8:]
            self._find_similar(query, memory_manager, agent)
        elif args.startswith("patterns "):
            task_type = args[9:]
            self._show_patterns(task_type, memory_manager, agent)
        else:
            # Filter by type
            self._list_by_type(args, memory_manager, agent)

        return True

    def _list_all(self, memory_manager, agent):
        """List all task memories."""
        memories = memory_manager.memories

        if not memories:
            agent.console.print("[yellow]No task memories yet[/yellow]")
            return

        table = Table(title=f"Task Memory ({len(memories)} tasks)")
        table.add_column("ID", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Description", style="white")
        table.add_column("Status", style="green")
        table.add_column("Iterations", justify="right")

        for mem in sorted(memories, key=lambda m: m.timestamp, reverse=True)[:20]:
            status = "✓" if mem.success else "✗"
            status_style = "green" if mem.success else "red"

            table.add_row(
                mem.task_id[-8:],
                mem.task_type,
                (
                    mem.description[:50] + "..."
                    if len(mem.description) > 50
                    else mem.description
                ),
                f"[{status_style}]{status}[/{status_style}]",
                str(mem.iterations),
            )

        agent.console.print(table)

    def _list_by_type(self, task_type: str, memory_manager, agent):
        """List memories filtered by type."""
        memories = memory_manager.recall_by_type(task_type, success_only=False)

        if not memories:
            agent.console.print(f"[yellow]No {task_type} tasks found[/yellow]")
            return

        table = Table(title=f"{task_type.title()} Tasks ({len(memories)})")
        table.add_column("Description", style="white")
        table.add_column("Status", style="green")
        table.add_column("Time (s)", justify="right")

        for mem in memories[:20]:
            status = "✓" if mem.success else "✗"
            status_style = "green" if mem.success else "red"

            table.add_row(
                (
                    mem.description[:60] + "..."
                    if len(mem.description) > 60
                    else mem.description
                ),
                f"[{status_style}]{status}[/{status_style}]",
                f"{mem.execution_time:.1f}",
            )

        agent.console.print(table)

    def _find_similar(self, query: str, memory_manager, agent):
        """Find similar past tasks."""
        similar = memory_manager.recall_similar(query, limit=5)

        if not similar:
            agent.console.print("[yellow]No similar tasks found[/yellow]")
            return

        agent.console.print(f"\n[bold cyan]Similar tasks to: {query}[/bold cyan]\n")

        for i, mem in enumerate(similar, 1):
            status = "✓" if mem.success else "✗"
            status_style = "green" if mem.success else "red"

            panel_content = f"""[bold]Description:[/bold] {mem.description}
[bold]Type:[/bold] {mem.task_type}
[bold]Status:[/bold] [{status_style}]{status}[/{status_style}]
[bold]Approach:[/bold] {mem.approach}
[bold]Iterations:[/bold] {mem.iterations}"""

            if mem.difficulties:
                panel_content += (
                    f"\n[bold]Difficulties:[/bold] {', '.join(mem.difficulties)}"
                )

            agent.console.print(
                Panel(
                    panel_content, title=f"Similar Task {i}", border_style=status_style
                )
            )

    def _show_patterns(self, task_type: str, memory_manager, agent):
        """Show success patterns for a task type."""
        patterns = memory_manager.get_success_patterns(task_type)

        if patterns["total_tasks"] == 0:
            agent.console.print(f"[yellow]No {task_type} tasks found[/yellow]")
            return

        content = f"""[bold]Success Rate:[/bold] {patterns['success_rate']*100:.1f}%
[bold]Total Tasks:[/bold] {patterns['total_tasks']}
[bold]Avg Iterations:[/bold] {patterns.get('avg_iterations', 0):.1f}
[bold]Avg Time:[/bold] {patterns.get('avg_execution_time', 0):.1f}s

[bold]Common Tools:[/bold]"""

        for tool, count in patterns.get("common_tools", []):
            content += f"\n  • {tool} ({count}x)"

        if patterns.get("common_difficulties"):
            content += "\n\n[bold]Common Difficulties:[/bold]"
            for diff, count in patterns["common_difficulties"]:
                content += f"\n  • {diff} ({count}x)"

        agent.console.print(
            Panel(content, title=f"Success Patterns: {task_type}", border_style="cyan")
        )
