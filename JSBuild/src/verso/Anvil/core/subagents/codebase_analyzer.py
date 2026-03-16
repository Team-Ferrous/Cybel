import os
import re
from typing import Dict, Any, List

from core.agent import BaseAgent
from core.agents.subagent import SubAgent
from tools.saguaro_tools import SaguaroTools
import logging

logger = logging.getLogger(__name__)


class CodebaseAnalyzerSubagent(SubAgent):
    """
    Dedicated subagent for analyzing a single file within the codebase.

    This subagent leverages Saguaro tooling to provide deep insights into
    a file's structure, dependencies, and semantic context. It generates
    a structured Markdown report for each file analyzed.
    """

    def __init__(
        self,
        parent_agent: BaseAgent,
        console,
        saguaro_tools: SaguaroTools,
        report_dir: str,
        file_path: str,
    ):
        task = f"Analyze file: {file_path}"
        super().__init__(
            task=task,
            parent_agent=parent_agent,
            quiet=True,  # Operate quietly, reporting through Markdown files
        )
        self.console = console
        self.saguaro_tools = saguaro_tools
        self.report_dir = report_dir
        self.file_path = file_path
        self.context_budget = 100000  # Context budget for analysis prompts
        self.system_prompt = self._build_specialist_prompt()

    def _build_specialist_prompt(self) -> str:
        """Build specialized system prompt for file analysis."""
        return """You are CodebaseAnalyzer, a specialized subagent for deep file analysis.

Your role:
- Analyze the provided file's content, structure, and Saguaro-derived information.
- Generate a comprehensive Markdown report for the file.
- DO NOT modify any code. Your output is strictly reports.

Tools available:
- read_file: Read full file content
- skeleton: Get file structure (functions/classes without bodies)
- slice: Extract specific entity from file
- impact: Analyze the impact of changes to a file/entity
- query: Perform semantic search for related concepts/usages

Output format:
Provide a structured Markdown report with the following sections:
# File Analysis Report: <file_path>

## 1. Overview
- Briefly describe the file's purpose based on its content and name.

## 2. File Content
```<language_extension>
<full_file_content_or_truncated_if_large>
```

## 3. Structure (Skeleton)
```
<file_skeleton_output>
```

## 4. Key Entities (Classes, Functions)
- List important classes/functions found.
- For each, briefly explain its role.

## 5. Dependencies and Impact (Saguaro Impact Analysis)
- Summarize the output of Saguaro's `impact` tool for this file.
- List other files or entities that depend on this file, or that this file depends on.

## 6. Semantic Connections (Saguaro Query)
- Summarize findings from Saguaro's `query` tool based on key terms from the file.
- List related files, concepts, or common usage patterns.

## 7. Potential Improvements / Insights
- Suggest any insights about this file:
    - Code smells
    - Performance considerations
    - Testability concerns
    - Documentation gaps
    - Potential refactoring opportunities
- This should be purely analytical, not prescriptive.

Be thorough but concise. Ensure all code blocks are properly formatted."""

    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analyzes a single file and generates a Markdown report.

        Args:
            file_path: The path to the file to analyze.

        Returns:
            Dict containing analysis metadata and path to the generated report.
        """
        logger.info(f"Analyzing file: {file_path}")
        self.console.print(f"    [dim]→ Processing: {file_path}[/dim]")

        full_content = ""
        skeleton_content = ""
        impact_analysis_content = "N/A"
        semantic_query_content = "N/A"

        try:
            # 1. Read full file content
            full_content = self.registry.dispatch("read_file", {"file_path": file_path})
            if full_content.startswith("Error"):
                full_content = f"Error reading file: {full_content}"
            language = self._detect_language(file_path)

            # 2. Get file skeleton
            skeleton_content = self.saguaro_tools.skeleton(file_path)
            if skeleton_content.startswith("Error"):
                skeleton_content = f"Error getting skeleton: {skeleton_content}"

            # 3. Perform Impact Analysis
            impact_analysis_content = self.saguaro_tools.impact(file_path)
            if impact_analysis_content.startswith("Error"):
                impact_analysis_content = (
                    f"Error performing impact analysis: {impact_analysis_content}"
                )

            # 4. Perform Semantic Query for connections
            key_terms = self._extract_key_terms_from_content(full_content)
            if key_terms:
                semantic_query_content = self.saguaro_tools.query(
                    f"What is related to {', '.join(key_terms[:3])} in {file_path}?",
                    k=5,
                )
                if semantic_query_content.startswith("Error"):
                    semantic_query_content = (
                        f"Error performing semantic query: {semantic_query_content}"
                    )

            # Build context for LLM to generate report
            analysis_context = f"""
File Path: {file_path}
Language: {language}

## Full File Content:
```{language}
{full_content[:5000]} # Truncate for prompt if very large
```

## File Structure (Skeleton):
```
{skeleton_content}
```

## Saguaro Impact Analysis:
```
{impact_analysis_content}
```

## Saguaro Semantic Query Results:
```
{semantic_query_content}
```

Based on the above information, generate a detailed file analysis report following the specified format.
"""

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": analysis_context},
            ]

            report_markdown = ""
            for chunk in self.brain.stream_chat(
                messages, max_tokens=self.context_budget, temperature=0.0
            ):
                report_markdown += chunk

            # Ensure proper Markdown formatting for file content and skeleton
            report_markdown = report_markdown.replace("<language_extension>", language)

            # Sanitize filename for report
            report_filename = (
                re.sub(r"[^a-zA-Z0-9_.-]", "_", file_path).replace(os.sep, "__") + ".md"
            )
            report_file_path = os.path.join(self.report_dir, report_filename)

            with open(report_file_path, "w") as f:
                f.write(report_markdown)

            estimated_tokens = len(analysis_context) // 4 + len(report_markdown) // 4
            envelope = self._build_evidence_envelope(
                file_path=file_path,
                summary=f"Generated file analysis report for {file_path}",
                artifacts=[{"type": "file_report", "path": report_file_path}],
                metrics={"tokens_used": estimated_tokens},
            )
            return {
                "file_path": file_path,
                "status": "success",
                "report_path": report_file_path,
                "tokens_used": estimated_tokens,  # Rough estimate
                "evidence_envelope": envelope,
            }

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return {
                "file_path": file_path,
                "status": "error",
                "error": str(e),
                "evidence_envelope": self._build_evidence_envelope(
                    file_path=file_path,
                    summary=f"File analysis failed: {e}",
                    artifacts=[],
                    metrics={"tokens_used": 0},
                ),
            }

    def _detect_language(self, file_path: str) -> str:
        """Simple language detection based on file extension."""
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        if ext == ".py":
            return "python"
        if ext == ".js":
            return "javascript"
        if ext == ".ts":
            return "typescript"
        if ext == ".java":
            return "java"
        if ext == ".cpp" or ext == ".h":
            return "cpp"
        if ext == ".go":
            return "go"
        if ext == ".md":
            return "markdown"
        if ext == ".json":
            return "json"
        if ext == ".yaml" or ext == ".yml":
            return "yaml"
        if ext == ".xml":
            return "xml"
        return "text"

    def _extract_key_terms_from_content(self, content: str) -> List[str]:
        """Extracts key terms from file content for semantic querying."""
        # Simple regex for now to get capitalized words (class names, constants) and function names
        key_terms = []
        # Match Python class names (e.g., "class MyClass:")
        key_terms.extend(re.findall(r"class\s+([A-Z][a-zA-Z0-9_]*)", content))
        # Match Python function names (e.g., "def my_function():")
        key_terms.extend(re.findall(r"def\s+([a-z_][a-zA-Z0-9_]*)", content))
        # Match capitalized words (could be variables, classes in other langs)
        key_terms.extend(re.findall(r"\b[A-Z][a-zA-Z0-9_]{2,}\b", content))

        # Filter out common short words or generic terms
        stop_words = {
            "def",
            "class",
            "import",
            "from",
            "return",
            "if",
            "else",
            "for",
            "while",
            "try",
            "except",
        }
        key_terms = [
            term
            for term in key_terms
            if term.lower() not in stop_words and len(term) > 2
        ]

        return list(set(key_terms))[:5]  # Return up to 5 unique key terms

    def _build_evidence_envelope(
        self,
        *,
        file_path: str,
        summary: str,
        artifacts: List[Dict[str, Any]],
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "schema_version": "evidence_envelope.v1",
            "producer": "CodebaseAnalyzerSubagent",
            "evidence_type": "file_analysis",
            "summary": summary,
            "query": self.task,
            "sources": [{"path": file_path, "role": "analysis_local"}],
            "artifacts": list(artifacts),
            "metrics": dict(metrics),
        }
