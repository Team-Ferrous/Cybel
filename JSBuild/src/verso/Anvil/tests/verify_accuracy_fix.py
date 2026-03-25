import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from core.agent import BaseAgent
from core.agents.repo_analyzer import RepoAnalysisSubagent
from rich.console import Console


def verify_accuracy():
    console = Console()
    # Initialize a BaseAgent to get the correct brain and context
    master_agent = BaseAgent(name="MasterVerifier", console=console)
    from core.approval import ApprovalMode

    master_agent.approval_manager.mode = ApprovalMode.FULL_AUTO

    # We want to see the thinking and tool calls to verify it's following the protocol
    analyzer = RepoAnalysisSubagent(
        task="Analyze the core architecture of this repository, specifically the reasoning system.",
        parent_name="Verifier",
        parent_agent=master_agent,
        console=console,
    )
    analyzer.approval_manager.mode = ApprovalMode.FULL_AUTO

    console.print(
        "[bold]Launching RepoAnalysisSubagent for accuracy verification...[/bold]"
    )
    result = analyzer.run(root_dir=".")

    summary = result.get("summary", "").lower()
    full_response = result.get("full_response", "").lower()

    console.print("\n[bold green]--- VERIFICATION RESULTS ---[/bold green]")

    # 1. Check for Coconut identification
    if "coconut" in summary or "continuous thought" in summary:
        console.print(
            "[green]✓ Correctly identified Coconut/Continuous Thought reasoning system.[/green]"
        )
    else:
        console.print(
            "[red]✗ Failed to identify Coconut/Continuous Thought reasoning system.[/red]"
        )

    # 2. Check for Flask hallucination (Negative Grounding)
    if "flask" in summary or "django" in summary or "fastapi" in summary:
        console.print(
            "[red]✗ Hallucinated a web framework (Flask/Django/FastAPI).[/red]"
        )
    else:
        console.print("[green]✓ No web framework hallucinations detected.[/green]")

    # 3. Check for evidence-based claims
    if "unified_chat_loop.py" in full_response or "coconut_bridge.py" in full_response:
        console.print("[green]✓ Provided evidence from actual codebase files.[/green]")
    else:
        console.print(
            "[yellow]⚠ No specific file evidence mentioned in response (check tool usage).[/yellow]"
        )

    # Save summary for manual inspection if needed
    with open("tests/verification_summary.txt", "w") as f:
        f.write(result.get("summary", ""))


if __name__ == "__main__":
    verify_accuracy()
