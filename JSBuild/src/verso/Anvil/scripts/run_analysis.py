import os
import sys

# Add the project root to the Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from agents.unified_master import UnifiedMasterAgent
from core.loops.codebase_analysis_loop import CodebaseAnalysisLoop
from tools.saguaro_tools import SaguaroTools
from domains.code_intelligence.saguaro_substrate import SaguaroSubstrate  # For SaguaroSubstrate instance
from rich.console import Console


def run_analysis():
    """
    Runs the codebase analysis loop.
    """
    console = Console()
    agent = UnifiedMasterAgent()
    agent.root_dir = ROOT_DIR

    console.print("[bold cyan]Initiating Codebase Analysis...[/bold cyan]")

    # Define the report directory
    report_base_dir = os.path.join(agent.root_dir, ".anvil")
    report_dir = os.path.join(report_base_dir, "analysis_reports")

    # Initialize SaguaroTools
    saguaro = SaguaroSubstrate()
    saguaro_tools = SaguaroTools(saguaro)

    # Instantiate and run the CodebaseAnalysisLoop
    analysis_loop = CodebaseAnalysisLoop(
        agent=agent,
        console=console,
        saguaro_tools=saguaro_tools,
        report_dir=report_dir,
        registry=agent.registry,  # Pass the agent's registry
    )

    try:
        summary_stats = analysis_loop.run()
        console.print(
            "[bold green]Codebase Analysis Completed Successfully![/bold green]"
        )
        console.print(
            f"Reports available in: [link=file://{summary_stats['reports_directory']}]{summary_stats['reports_directory']}[/link]"
        )
        console.print(
            f"Overall summary: [link=file://{summary_stats['overall_summary_file']}]{summary_stats['overall_summary_file']}[/link]"
        )
    except Exception as e:
        console.print(f"[bold red]Codebase Analysis failed: {e}[/bold red]")
        import traceback

        console.print(traceback.format_exc())


if __name__ == "__main__":
    run_analysis()
