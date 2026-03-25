"""
Progress Dashboard - Minimal inline status display.

Claude Code-style: Simple inline progress indicators without heavy panels or trees.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
import time

from rich.console import Console
from rich.text import Text


@dataclass
class AgentStatus:
    """Status of an individual agent."""
    name: str
    status: str  # "idle", "running", "waiting", "completed", "failed"
    progress: float  # 0.0 to 1.0
    message: str = ""
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    @property
    def duration(self) -> Optional[float]:
        """Get duration in seconds."""
        if self.start_time is None:
            return None
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time


@dataclass
class PhaseStatus:
    """Status of an execution phase."""
    name: str
    status: str  # "pending", "in_progress", "completed", "failed"
    progress: float = 0.0
    agents: Dict[str, AgentStatus] = field(default_factory=dict)
    message: str = ""
    start_time: Optional[float] = None
    end_time: Optional[float] = None


class StreamOutputContainer:
    """Container for streaming output renderables."""
    
    def __init__(self):
        self.renderables: List = []
    
    def clear(self):
        """Clear all renderables."""
        self.renderables = []


class LiveProgressDashboard:
    """
    Minimal inline progress display.
    
    Uses simple text output instead of complex Rich Live panels.
    Updates are printed inline for a clean, Claude Code-like experience.
    """

    # Status icons (simple, minimal)
    STATUS_ICONS = {
        "pending": "○",
        "in_progress": "●",
        "running": "⠋",
        "completed": "✓",
        "failed": "✗",
        "skipped": "○",
        "waiting": "◌",
        "idle": "○",
    }

    STATUS_COLORS = {
        "pending": "dim",
        "in_progress": "cyan",
        "running": "cyan",
        "completed": "green",
        "failed": "red",
        "skipped": "dim",
        "waiting": "yellow",
        "idle": "dim",
    }

    def __init__(
        self,
        console: Optional[Console] = None,
        title: str = "",
        renderer=None,
    ):
        self.console = console or Console()
        self.renderer = renderer
        self.title = title
        self.phases: Dict[str, PhaseStatus] = {}
        self.phase_order: List[str] = []
        self.start_time = time.time()
        self._last_phase: Optional[str] = None
        self._printed_phases: set = set()
        # Stream output container for handling streaming content
        self.stream_output = StreamOutputContainer()

    def add_phase(
        self, name: str, status: str = "pending", message: str = ""
    ) -> PhaseStatus:
        """Add a new phase."""
        phase = PhaseStatus(
            name=name,
            status=status,
            message=message,
            start_time=time.time() if status == "in_progress" else None,
        )
        self.phases[name] = phase
        self.phase_order.append(name)
        
        # Print phase start if in_progress
        if status == "in_progress":
            self._print_phase_start(name, message)
        
        return phase

    def update_phase(
        self, name: str, status: str, progress: float = None, message: str = None
    ):
        """Update phase status with inline output."""
        if name not in self.phases:
            return

        phase = self.phases[name]
        old_status = phase.status
        phase.status = status

        if progress is not None:
            phase.progress = progress
        if message is not None:
            phase.message = message

        # Track timing
        if old_status != "in_progress" and status == "in_progress":
            phase.start_time = time.time()
            self._print_phase_start(name, message or "")
        elif status in ["completed", "failed", "skipped"]:
            phase.end_time = time.time()
            if name not in self._printed_phases:
                self._print_phase_complete(name, status)

    def _print_phase_start(self, name: str, message: str = ""):
        """Print phase start indicator."""
        if name in self._printed_phases:
            return
        icon = self.STATUS_ICONS["in_progress"]
        msg = f" {message}" if message else ""
        self.console.print(Text(f"\n{icon} {name}{msg}", style="cyan"))
        self._last_phase = name

    def _print_phase_complete(self, name: str, status: str):
        """Print phase completion."""
        self._printed_phases.add(name)
        icon = self.STATUS_ICONS.get(status, "✓")
        color = self.STATUS_COLORS.get(status, "dim")
        duration = ""
        if name in self.phases and self.phases[name].start_time:
            elapsed = time.time() - self.phases[name].start_time
            duration = f" ({elapsed:.1f}s)"
        self.console.print(Text(f"  {icon} {name}{duration}", style=color))

    def add_agent(
        self, phase_name: str, agent_name: str, message: str = "", status: str = "idle"
    ) -> Optional[AgentStatus]:
        """Add an agent to a phase."""
        if phase_name not in self.phases:
            return None

        agent = AgentStatus(
            name=agent_name,
            status=status,
            progress=0.0,
            message=message,
            start_time=time.time() if status == "running" else None,
        )
        self.phases[phase_name].agents[agent_name] = agent

        # Print agent start if running
        if status == "running":
            self._print_agent(agent_name, message, status)

        return agent

    def update_agent(
        self, agent_name: str, status: str, progress: float = 0.0, message: str = ""
    ):
        """Update agent status."""
        for phase in self.phases.values():
            if agent_name in phase.agents:
                agent = phase.agents[agent_name]
                old_status = agent.status
                agent.status = status
                agent.progress = progress
                if message:
                    agent.message = message

                # Track timing
                if old_status != "running" and status == "running":
                    agent.start_time = time.time()
                elif status in ["completed", "failed"]:
                    agent.end_time = time.time()
                    self._print_agent(agent_name, message, status, agent.duration)

                # Update phase progress
                total_progress = sum(a.progress for a in phase.agents.values())
                phase.progress = total_progress / len(phase.agents) if phase.agents else 0.0
                return

    def _print_agent(self, name: str, message: str, status: str, duration: float = None):
        """Print agent status inline."""
        icon = self.STATUS_ICONS.get(status, "•")
        color = self.STATUS_COLORS.get(status, "dim")
        dur = f" ({duration:.1f}s)" if duration else ""
        msg = f": {message}" if message else ""
        self.console.print(Text(f"    {icon} {name}{msg}{dur}", style=color))

    def live(self, refresh_per_second: int = 4):
        """Context manager - no-op for minimal UI."""
        from contextlib import contextmanager

        @contextmanager
        def noop():
            yield self

        return noop()

    def update_display(self):
        """No-op for inline display."""
        pass

    def get_summary(self) -> str:
        """Get text summary of execution."""
        lines = []
        elapsed = time.time() - self.start_time

        for phase_name in self.phase_order:
            phase = self.phases[phase_name]
            icon = self.STATUS_ICONS.get(phase.status, "○")
            lines.append(f"{icon} {phase.name}")

        lines.append(f"\nTotal: {elapsed:.1f}s")
        return "\n".join(lines)

    def process_stream_event(self, event):
        """Process streaming events (minimal handling)."""
        if hasattr(event, 'type'):
            if event.type == "content" and hasattr(event, 'content'):
                self.console.print(event.content, end="")


class SimpleProgressTracker:
    """Lightweight progress tracker."""

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self.steps: List[str] = []
        self.current_step: Optional[str] = None
        self.start_time = time.time()

    def start_step(self, description: str):
        """Start a new step."""
        self.current_step = description
        self.console.print(Text(f"  ⠋ {description}", style="cyan"))

    def complete_step(self, message: str = "Done"):
        """Complete current step."""
        if self.current_step:
            self.steps.append(self.current_step)
            self.console.print(Text(f"  ✓ {message}", style="green"))
            self.current_step = None

    def fail_step(self, error: str):
        """Mark current step as failed."""
        if self.current_step:
            self.console.print(Text(f"  ✗ {error}", style="red"))
            self.current_step = None

    def summary(self):
        """Print summary."""
        elapsed = time.time() - self.start_time
        self.console.print(Text(f"\n  Completed {len(self.steps)} steps in {elapsed:.1f}s", style="dim"))
