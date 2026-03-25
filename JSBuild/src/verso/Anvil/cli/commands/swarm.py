"""REPL command to execute the sequential swarm coordinator."""

from __future__ import annotations

import asyncio
from typing import Any, List

from rich.panel import Panel

from cli.commands.base import SlashCommand
from domains.coordination.sequential_swarm import (
    SequentialSwarmCoordinator,
    SwarmTopology,
)


class SwarmCommand(SlashCommand):
    """Run a deterministic sequential swarm for a single objective."""

    @property
    def name(self) -> str:
        return "swarm"

    @property
    def aliases(self) -> list[str]:
        return ["sw"]

    @property
    def description(self) -> str:
        return "Run sequential swarm coordination across specialist agent types."

    def execute(self, args: List[str], context: Any) -> None:
        if not args:
            return (
                "Usage: /swarm [--topology hierarchical|sequential|round_robin] "
                "[--agents planner,coder,reviewer] <objective>"
            )

        topology = "hierarchical"
        agents = "planner,coder,reviewer"
        objective_parts: list[str] = []

        i = 0
        while i < len(args):
            token = str(args[i] or "").strip()
            if token == "--topology" and i + 1 < len(args):
                topology = str(args[i + 1] or "").strip().lower()
                i += 2
                continue
            if token == "--agents" and i + 1 < len(args):
                agents = str(args[i + 1] or "").strip()
                i += 2
                continue
            objective_parts.append(token)
            i += 1

        objective = " ".join(objective_parts).strip()
        if not objective:
            return (
                "Usage: /swarm [--topology hierarchical|sequential|round_robin] "
                "[--agents planner,coder,reviewer] <objective>"
            )

        topology_enum = {
            "hierarchical": SwarmTopology.HIERARCHICAL,
            "sequential": SwarmTopology.SEQUENTIAL,
            "round_robin": SwarmTopology.ROUND_ROBIN,
            # Backwards-compatible aliases.
            "ring": SwarmTopology.ROUND_ROBIN,
            "mesh": SwarmTopology.SEQUENTIAL,
        }.get(topology, SwarmTopology.HIERARCHICAL)

        agent_types = [item.strip() for item in agents.split(",") if item.strip()]
        if not agent_types:
            return "Swarm requires at least one agent in --agents."

        coordinator = SequentialSwarmCoordinator(
            parent_agent=context,
            topology=topology_enum,
        )
        sequence = [
            {"type": agent_type, "instruction": objective}
            for agent_type in agent_types
        ]

        try:
            result = asyncio.run(
                coordinator.execute_swarm(
                    task_description=objective,
                    agent_sequence=sequence,
                )
            )
        except RuntimeError:
            # Fallback when an event loop is already running.
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(
                    coordinator.execute_swarm(
                        task_description=objective,
                        agent_sequence=sequence,
                    )
                )
            finally:
                loop.close()

        status = str(result.get("status") or "unknown")
        final_summary = str(result.get("final_summary") or "").strip() or "(no summary)"
        panel_body = (
            f"Status: {status}\n"
            f"Topology: {topology_enum.value}\n"
            f"Agents: {', '.join(agent_types)}\n\n"
            f"{final_summary}"
        )
        context.console.print(
            Panel(panel_body, title="Swarm Coordination", border_style="cyan")
        )
        context.history.add_message("system", f"Swarm command result:\n{panel_body}")
        return None
