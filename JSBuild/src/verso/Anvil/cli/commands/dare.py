"""REPL entry point for the DARE subsystem."""

from __future__ import annotations

import json
from typing import Any

from cli.commands.base import SlashCommand
from core.dare.pipeline import DarePipeline


class DareCommand(SlashCommand):
    """Expose DARE workflows through a single slash command."""

    @property
    def name(self) -> str:
        """Return the primary slash command name."""
        return "dare"

    @property
    def description(self) -> str:
        """Return a short help description."""
        return "Run Deep Analysis & Autonomous Research Engine workflows."

    @property
    def aliases(self) -> list[str]:
        """Return command aliases."""
        return ["research_lab"]

    def execute(self, args: list[str], context: Any) -> bool:
        """Dispatch a DARE subcommand."""
        if not args:
            usage = (
                "[yellow]Usage: /dare "
                "<analyze|research|compete|synthesize|sculpt|run|status|kb> ..."
                "[/yellow]"
            )
            context.console.print(usage)
            return True

        pipeline = self._get_pipeline(context)

        command = args[0].lower()
        tail = args[1:]

        if command == "analyze":
            return self._handle_analyze(context, pipeline, tail)

        if command == "research":
            return self._handle_research(context, pipeline, tail)

        if command == "compete":
            return self._handle_compete(context, pipeline, tail)

        if command == "synthesize":
            return self._handle_synthesize(context, pipeline, tail)

        if command == "sculpt":
            return self._handle_sculpt(context, pipeline, tail)

        if command == "run":
            report = pipeline.run()
            context.console.print(json.dumps(report.to_dict(), indent=2, default=str))
            return True

        if command == "status":
            context.console.print(json.dumps(pipeline.status(), indent=2, default=str))
            return True

        if command == "kb":
            return self._handle_kb(context, pipeline, tail)

        context.console.print(f"[yellow]Unknown DARE command:[/yellow] {command}")
        return True

    @staticmethod
    def _get_pipeline(context: Any) -> DarePipeline:
        pipeline = getattr(context, "_dare_pipeline", None)
        if pipeline is None:
            pipeline = DarePipeline(
                root_dir=getattr(context, "root_dir", "."),
                console=context.console,
                brain=getattr(context, "brain", None),
                ownership_registry=getattr(context, "ownership_registry", None),
            )
            context._dare_pipeline = pipeline
        return pipeline

    @staticmethod
    def _handle_analyze(context: Any, pipeline: DarePipeline, args: list[str]) -> bool:
        profiles = pipeline.analyze(args or [getattr(context, "root_dir", ".")])
        for profile in profiles.values():
            message = (
                f"[bold cyan]{profile.repo_name}[/bold cyan] "
                f"files={profile.file_count} loc={profile.loc} role={profile.role}"
            )
            context.console.print(message)
        return True

    @staticmethod
    def _handle_research(context: Any, pipeline: DarePipeline, args: list[str]) -> bool:
        if not args:
            context.console.print("[yellow]Usage: /dare research <topic>[/yellow]")
            return True
        report = pipeline.research(" ".join(args))
        context.console.print(report.summary)
        return True

    @staticmethod
    def _handle_compete(context: Any, pipeline: DarePipeline, args: list[str]) -> bool:
        if not args:
            context.console.print("[yellow]Usage: /dare compete <domain>[/yellow]")
            return True
        report = pipeline.compete(" ".join(args))
        context.console.print(report.summary)
        return True

    @staticmethod
    def _handle_synthesize(
        context: Any,
        pipeline: DarePipeline,
        args: list[str],
    ) -> bool:
        payload = {"description": " ".join(args)} if args else None
        roadmap = pipeline.synthesize(payload)
        context.console.print(roadmap.to_markdown())
        return True

    @staticmethod
    def _handle_sculpt(context: Any, pipeline: DarePipeline, args: list[str]) -> bool:
        campaign_path = pipeline.sculpt(" ".join(args) if args else None)
        context.console.print(f"[green]Generated campaign:[/green] {campaign_path}")
        return True

    @staticmethod
    def _handle_kb(context: Any, pipeline: DarePipeline, args: list[str]) -> bool:
        if not args:
            context.console.print(
                "[yellow]Usage: /dare kb <search|report> ...[/yellow]"
            )
            return True
        subcommand = args[0].lower()
        if subcommand == "search":
            query = " ".join(args[1:]).strip()
            results = pipeline.kb_search(query or "*")
            context.console.print(json.dumps(results, indent=2, default=str))
            return True
        if subcommand == "report":
            category = args[1] if len(args) > 1 else None
            context.console.print(pipeline.kb_report(category))
            return True
        context.console.print(
            f"[yellow]Unknown DARE kb command:[/yellow] {subcommand}"
        )
        return True
