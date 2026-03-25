import os
import time
from typing import List, Dict, Any

from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from core.subagents.codebase_analyzer import CodebaseAnalyzerSubagent
from tools.saguaro_tools import SaguaroTools


class CodebaseAnalysisLoop:
    """
    A specialized loop for exhaustive codebase analysis.

    This loop iterates through all files in the repository (respecting .gitignore
    and .geminiignore) and generates structured analysis reports using subagents
    and Saguaro tooling. It does not modify any files.
    """

    def __init__(
        self, agent, console, saguaro_tools: SaguaroTools, report_dir: str, registry
    ):
        self.agent = agent
        self.console = console
        self.saguaro_tools = saguaro_tools
        self.report_dir = report_dir
        self.registry = registry  # Added registry
        self.root_dir = os.getcwd()  # Assume current working directory is repo root

        os.makedirs(self.report_dir, exist_ok=True)
        self.console.print(
            f"[green]Initialized CodebaseAnalysisLoop. Reports will be saved to: {self.report_dir}[/green]"
        )

    def run(self) -> Dict[str, Any]:
        """
        Executes the codebase analysis.

        Returns:
            A dictionary containing summary statistics of the analysis.
        """
        self.console.print(
            Panel(
                "[bold blue]Starting Codebase Analysis Loop[/bold blue]\n"
                "[dim]Iterating through all files to generate detailed reports...[/dim]",
                border_style="blue",
            )
        )

        all_files = self._get_all_repo_files(self.root_dir)
        self.console.print(f"  [cyan]Found {len(all_files)} files to analyze.[/cyan]")

        analysis_results = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console,
        ) as progress:
            file_analysis_task = progress.add_task(
                "[green]Analyzing files...", total=len(all_files)
            )

            for i, file_path in enumerate(all_files):
                progress.update(
                    file_analysis_task,
                    description=f"[green]Analyzing: {file_path}[/green]",
                )

                # Instantiate subagent for each file or a shared one if context allows
                # For now, let's process directly or use a shared subagent instance
                analyzer_subagent = CodebaseAnalyzerSubagent(
                    parent_agent=self.agent,
                    console=self.console,
                    saguaro_tools=self.saguaro_tools,
                    report_dir=self.report_dir,
                    file_path=file_path,
                )

                try:
                    result = analyzer_subagent.process_file(file_path)
                    analysis_results.append(result)
                except Exception as e:
                    self.console.print(f"  [red]Error analyzing {file_path}: {e}[/red]")
                    analysis_results.append({"file_path": file_path, "error": str(e)})

                progress.update(file_analysis_task, advance=1)

        self.console.print(
            Panel(
                "[bold green]Codebase Analysis Complete[/bold green]\n"
                "[dim]Generating overall summary report...[/dim]",
                border_style="green",
            )
        )

        # Generate overall summary
        overall_summary = self._generate_overall_summary(analysis_results)
        self.console.print(
            f"  [green]Overall summary saved to: {overall_summary}[/green]"
        )

        return {
            "total_files_scanned": len(all_files),
            "total_files_analyzed": len(
                [r for r in analysis_results if "error" not in r]
            ),
            "reports_directory": self.report_dir,
            "overall_summary_file": overall_summary,
        }

    def _get_all_repo_files(self, current_dir: str) -> List[str]:
        """
        Recursively lists all files in the repository, respecting .gitignore and .geminiignore,
        using the `list_directory` tool.
        """
        all_files = []
        try:
            entries = self.registry.dispatch(
                "list_dir",
                {
                    "dir_path": current_dir,
                    "file_filtering_options": {
                        "respect_git_ignore": True,
                        "respect_gemini_ignore": True,
                    },
                },
            )
            print(f"DEBUG: Entries for {current_dir}: {entries}")

            for entry in entries:
                if entry in ['.', '..']:
                    continue
                full_path = os.path.join(current_dir, entry)
                if os.path.isfile(full_path):
                    all_files.append(full_path)
                elif os.path.isdir(full_path):
                    all_files.extend(self._get_all_repo_files(full_path))
        except Exception as e:
            self.console.print(
                f"  [yellow]⚠ Error listing directory {current_dir}: {e}[/yellow]"
            )
        return all_files

    def _generate_overall_summary(self, analysis_results: List[Dict[str, Any]]) -> str:
        """
        Generates an overall summary report for the entire codebase analysis.
        """
        summary_file_path = os.path.join(self.report_dir, "summary.md")
        with open(summary_file_path, "w") as f:
            f.write("# Codebase Analysis Summary Report\n\n")
            f.write(f"Generated on: {time.ctime()}\n\n")
            f.write(f"Total files scanned: {len(analysis_results)}\n")
            f.write(
                f"Total files successfully analyzed: {len([r for r in analysis_results if 'error' not in r])}\n\n"
            )

            f.write("## Analysis Findings per File\n\n")
            for result in analysis_results:
                file_path = result.get("file_path", "N/A")
                if "error" in result:
                    f.write(f"- `{file_path}`: [red]Error - {result['error']}[/red]\n")
                else:
                    # Construct relative path for the link
                    report_filename = (
                        os.path.basename(file_path).replace(".", "_") + ".md"
                    )
                    f.write(
                        f"- `{file_path}`: [green]Analyzed[/green] (Report: [link={report_filename}]View[/link])\n"
                    )

            f.write("\n## Key Insights (Placeholder)\n\n")
            f.write(
                "This section will contain aggregated insights, common patterns, architectural overview, and potential improvement areas identified across the codebase.\n"
            )
            f.write(
                "This will be generated by the LLM from the individual file analysis reports.\n"
            )

        return summary_file_path
