from core.agents.analysis import AnalysisAgent
from core.ollama_client import DeterministicOllama
from config.settings import MASTER_MODEL


class AnalysisLoop:
    """
    SubAgent Loop for deep codebase analysis and research.
    Combines Saguaro semantic tools with standard search/read tools.
    """

    def __init__(self, agent_repl):
        self.repl = agent_repl
        self.console = agent_repl.console
        self.brain = agent_repl.brain or DeterministicOllama(MASTER_MODEL)
        self.orchestrator = agent_repl.orchestrator
        # Initialize the analyst with the shared semantic engine
        self.analyst = AnalysisAgent(
            semantic_engine=self.orchestrator.semantic_engine, brain=self.brain
        )

    def run(self, topic: str):
        self.console.rule(f"[bold magenta]Analysis Loop: {topic}[/bold magenta]")

        # Ensure workspace is indexed
        if not getattr(self.orchestrator.semantic_engine, "_indexed", False):
            self.console.print(
                "[dim]Syncing workspace index for code analysis...[/dim]"
            )
            self.orchestrator.semantic_engine.analyze_workspace()

        # Execute the analysis cycle
        # The AnalysisAgent.run_loop will handle the multi-step reasoning
        # and tool usage (grep, skeleton, read_file, etc.)
        result = self.analyst.analyze_topic(topic)

        self.console.print("\n[bold green]Analysis Complete.[/bold green]")
        self.console.print(
            Panel(
                result["analysis"], title="Final Research Report", border_style="green"
            )
        )

        return result["analysis"]


from rich.panel import Panel
