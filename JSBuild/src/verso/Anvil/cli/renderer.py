"""
CLI Renderer - Claude Code-style minimal terminal UI.

Design principles:
- Clean, minimal aesthetic with subtle colors
- No heavy boxes or panels cluttering the screen
- Real-time task progress with simple spinners and status lines
- Markdown rendering for responses only
- System status in a non-intrusive bottom bar
"""

from typing import List, Optional
import sys
import psutil
from datetime import datetime
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.theme import Theme
from rich.text import Text
from rich.box import MINIMAL
from rich.rule import Rule
from rich.status import Status

# Modern, minimal color palette
ANVIL_THEME = Theme(
    {
        "info": "dim cyan",
        "warning": "yellow",
        "error": "red",
        "success": "green",
        "user": "bold white",
        "assistant": "white",
        "thinking": "dim italic",
        "tool": "cyan",
        "tool.name": "bold cyan",
        "tool.result": "dim white",
        "phase": "bold blue",
        "phase.active": "bold cyan",
        "phase.done": "dim green",
        "status": "dim white",
        "prompt": "bold cyan",
        "accent": "cyan",
        "muted": "dim white",
    }
)


class CLIRenderer:
    """
    Minimal terminal UI renderer inspired by Claude Code.
    
    Key features:
    - Clean welcome with just model info
    - Inline task status with spinners (not panels)
    - Streaming response display
    - Minimal chrome, maximum content
    """

    def __init__(self, console: Console = None):
        self.console = console or Console(theme=ANVIL_THEME)
        self.live = None
        self.layout = None
        self.main_buffer = []
        self._current_status: Optional[Status] = None
        self._task_lines: List[Text] = []

    def start_live_dashboard(self, model_name: str, mode: str):
        """Start a minimal live display (used during execution)."""
        # Don't use complex layouts - just track state
        self._task_lines = []
        self.live = None  # We'll use inline status updates instead

    def stop_live_dashboard(self):
        """Stop any active live display."""
        if self._current_status:
            self._current_status.stop()
            self._current_status = None
        if self.live:
            self.live.stop()
            self.live = None

    def update_dashboard(
        self, content=None, context_usage=None, active_tools=None, model=None, mode=None
    ):
        """Update inline status (no-op for minimal UI - we use explicit methods)."""
        pass

    def print_welcome_screen(self, model_name: str, mode: str):
        """Print a clean, minimal welcome banner."""
        # Clear any previous output for clean start
        self.console.print()
        
        # Simple gradient-style header
        self.console.print(
            Text("┌─ ", style="dim cyan") + 
            Text("ANVIL", style="bold cyan") + 
            Text(" AGENT ", style="bold white") +
            Text("─" * 50, style="dim cyan") + 
            Text("┐", style="dim cyan")
        )
        
        # Model and mode info (clean, inline)
        self.console.print(
            Text("│  ", style="dim cyan") +
            Text(f"Model: {model_name}", style="dim white") +
            Text("  •  ", style="dim cyan") +
            Text(f"Mode: {mode}", style="dim white")
        )
        
        # System info (very subtle)
        mem = psutil.virtual_memory()
        cpu = psutil.cpu_percent()
        self.console.print(
            Text("│  ", style="dim cyan") +
            Text(f"RAM: {mem.percent:.0f}%  CPU: {cpu:.0f}%  ", style="dim") +
            Text(f"{datetime.now().strftime('%H:%M')}", style="dim")
        )
        
        self.console.print(
            Text("└", style="dim cyan") + 
            Text("─" * 68, style="dim cyan") + 
            Text("┘", style="dim cyan")
        )
        self.console.print()

    def print_system(self, text: str):
        """Print a system message (subtle, inline)."""
        self.console.print(Text(f"  {text}", style="dim"))

    def print_error(self, text: str):
        """Print an error message."""
        self.console.print(Text(f"✗ {text}", style="red"))

    def print_warning(self, text: str):
        """Print a warning message."""
        self.console.print(Text(f"⚠ {text}", style="yellow"))

    def print_success(self, text: str):
        """Print a success message."""
        self.console.print(Text(f"✓ {text}", style="green"))

    def stream_token(self, token: str):
        """Print a single token for streaming output."""
        sys.stdout.write(token)
        sys.stdout.flush()

    def new_line(self):
        """Force a newline."""
        self.console.print()

    # ─────────────────────────────────────────────────────────────────────────────
    # TASK PROGRESS (Claude Code style - inline with spinners)
    # ─────────────────────────────────────────────────────────────────────────────

    def start_task(self, description: str) -> Status:
        """Start a task with a spinner."""
        self._current_status = self.console.status(
            f"[cyan]⠋[/cyan] {description}",
            spinner="dots",
            spinner_style="cyan"
        )
        self._current_status.start()
        return self._current_status

    def update_task(self, description: str):
        """Update the current task description."""
        if self._current_status:
            self._current_status.update(f"[cyan]⠋[/cyan] {description}")

    def complete_task(self, description: str):
        """Complete the current task."""
        if self._current_status:
            self._current_status.stop()
            self._current_status = None
        self.console.print(Text(f"  ✓ {description}", style="green"))

    def fail_task(self, description: str):
        """Mark current task as failed."""
        if self._current_status:
            self._current_status.stop()
            self._current_status = None
        self.console.print(Text(f"  ✗ {description}", style="red"))

    def print_phase(self, phase_name: str, status: str = "active"):
        """Print a phase indicator (Understanding, Evidence, Execution, Synthesis)."""
        if status == "active":
            self.console.print(Text(f"\n● {phase_name}", style="bold cyan"))
        elif status == "done":
            self.console.print(Text(f"  ✓ {phase_name}", style="dim green"))
        else:
            self.console.print(Text(f"  ○ {phase_name}", style="dim"))

    def print_subagent(self, name: str, message: str, status: str = "running"):
        """Print subagent status."""
        if status == "running":
            self.console.print(Text(f"    ⠋ {name}: {message}", style="cyan"))
        elif status == "done":
            self.console.print(Text(f"    ✓ {name}: {message}", style="dim green"))
        else:
            self.console.print(Text(f"    • {name}: {message}", style="dim"))

    # ─────────────────────────────────────────────────────────────────────────────
    # RESPONSE OUTPUT
    # ─────────────────────────────────────────────────────────────────────────────

    def print_response(self, response: str):
        """Print the final response with clean formatting."""
        self.console.print()
        
        # Simple divider
        self.console.print(Rule(style="dim cyan"))
        self.console.print()
        
        # Render markdown
        try:
            md = Markdown(response)
            self.console.print(md)
        except Exception:
            # Fallback to plain text
            self.console.print(response)
        
        self.console.print()

    def print_thinking_start(self):
        """Indicate thinking has started (minimal)."""
        self.console.print(Text("\n  🧠 Thinking...", style="dim italic"))

    def print_coconut_paths(
        self, amplitudes, path_summaries: Optional[List[str]] = None, num_paths: int = 4
    ):
        """Display COCONUT paths in a minimal format."""
        import numpy as np

        if not isinstance(amplitudes, np.ndarray):
            amplitudes = np.array(amplitudes)
        if len(amplitudes.shape) > 1:
            amplitudes = amplitudes.flatten()
        amplitudes = amplitudes[:num_paths]

        # Find best path
        best_idx = int(np.argmax(amplitudes))

        self.console.print(Text("\n  🥥 COCONUT Reasoning Paths:", style="dim"))
        
        for i, amp in enumerate(amplitudes):
            amp_val = float(amp)
            bar_len = int(amp_val * 20 / max(amplitudes))
            bar = "█" * bar_len + "░" * (20 - bar_len)
            
            if i == best_idx:
                self.console.print(
                    Text(f"     Path {i+1}: [{bar}] {amp_val:.3f} ", style="cyan") +
                    Text("← selected", style="bold green")
                )
            else:
                self.console.print(
                    Text(f"     Path {i+1}: [{bar}] {amp_val:.3f}", style="dim")
                )
        
        self.console.print()

    def print_markdown(self, text: str):
        """Render markdown text."""
        try:
            md = Markdown(text)
            self.console.print(md)
        except Exception:
            self.console.print(text)

    def print_panel(self, content, title=None, style="white", border_style="dim"):
        """Print a minimal panel (used sparingly)."""
        self.console.print(
            Panel(
                content,
                title=title,
                border_style=border_style,
                box=MINIMAL,
                padding=(0, 1),
            )
        )

    def print_diff_panel(self, diff_text: str):
        """Print a syntax-highlighted diff."""
        from rich.syntax import Syntax

        self.console.print()
        self.console.print(Text("  Proposed Changes:", style="bold"))
        syntax = Syntax(diff_text, "diff", theme="monokai", line_numbers=True)
        self.console.print(syntax)
        self.console.print()

    # Legacy method kept for compatibility
    def _add_to_live(self, renderable):
        """Add content to buffer (legacy)."""
        self.main_buffer.append(renderable)


def get_bottom_toolbar_text(
    mode, model, context_usage="?", editing_mode="", coconut_info="", engine_info=""
):
    """Returns formatted text for prompt_toolkit bottom toolbar (minimal style)."""
    from prompt_toolkit.formatted_text import HTML

    # Clean, minimal toolbar
    return HTML(
        f"<style fg='#888888'>"
        f" {mode} • {model} • {engine_info} • {coconut_info} • {editing_mode} "
        f"</style>"
    )
