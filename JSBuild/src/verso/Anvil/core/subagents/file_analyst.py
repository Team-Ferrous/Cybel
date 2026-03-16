"""
File Analysis Subagent

Specialized subagent for deep semantic file analysis.
- Own 200k context window (separate from master)
- Shares brain (DeterministicOllama) with master for memory efficiency
- Read-only tools: saguaro_query, skeleton, slice, read_file
- Returns structured summaries for master to synthesize
"""

from typing import Any, Dict, List, Tuple
from core.agent import BaseAgent
from core.agents.subagent import SubAgent
from core.token_budget import TokenBudgetManager
import logging

logger = logging.getLogger(__name__)


class FileAnalysisSubagent(SubAgent):
    """
    Dedicated subagent for analyzing large file sets.

    Use Case:
    When iterative search finds >10 files, delegate to this specialist
    to avoid overwhelming master's context window.

    Process:
    1. Load files progressively (skeletons first)
    2. Identify key entities relevant to query
    3. Load full content for critical sections
    4. Generate structured summary with citations
    5. Return summary + metadata to master
    """

    def __init__(
        self, parent_agent: BaseAgent, files: List[str], query: str, quiet: bool = False
    ):
        """
        Initialize file analysis subagent.

        Args:
            parent_agent: Master agent to share brain with
            files: List of file paths to analyze
            query: The question/task to answer
        """
        task = f"Analyze these files to answer: {query}"
        super().__init__(task=task, parent_agent=parent_agent, quiet=quiet)

        self.files = files
        self.query = query
        self.context_budget = 175000  # Reserve 25k for response in 200k window
        self.system_prompt = self._build_specialist_prompt()

    def _build_specialist_prompt(self) -> str:
        """Build specialized system prompt for file analysis."""
        return """You are FileAnalyst, a specialized subagent for deep semantic file analysis. Your purpose is to provide NATURAL LANGUAGE ANALYSIS, not code reproductions.

CRITICAL RULES - FOLLOW EXACTLY:
1. NEVER output skeleton code, pseudocode, or raw code structures as your main response
2. Your output must be PROSE EXPLANATIONS in natural language
3. Code snippets are ONLY allowed as brief inline examples (3-5 lines max) to illustrate points
4. If you find yourself writing "def ...", "class ...", or "import ..." as main content, STOP and rewrite as prose

Your role:
- Analyze code files to answer specific questions with NATURAL LANGUAGE EXPLANATIONS
- Identify key entities (functions, classes) and explain their PURPOSE and ROLE in prose
- Infer design patterns, architectural choices, and engineering trade-offs
- Provide structured, evidence-based analysis with citations to the source code
- Prioritize quality and depth of insight over completeness

Tools available:
- saguaro_query: Find relevant code and files deterministically
- skeleton: Get file structure (functions/classes without bodies)
- slice: Extract specific entity from file
- read_file: Read full file content after narrowing scope

REQUIRED OUTPUT FORMAT (use EXACTLY this structure):

## 1. High-Level Summary
A concise 2-3 sentence answer to the query. PROSE ONLY, no code.

## 2. Purpose & Architecture
Explain in NATURAL LANGUAGE what this code does and why it exists.
- What problem does it solve?
- How does it fit into the larger system?
- What design patterns are used?

## 3. Key Components
For each important class/function, provide:
- **Name**: The identifier
- **Role**: What it DOES (1-2 sentences of prose)
- **Significance**: WHY it matters

## 4. How It Works
Explain the FLOW in natural language. You may include 1-2 very short code snippets (3-5 lines) as examples.

## 5. Integration Points
How does this code connect to other parts of the system? Reference file paths and line numbers.

## 6. Critical Files
List the most relevant files that support your analysis (with file:line citations).

REMEMBER: The master agent will synthesize your findings for the user. Your analysis must be EXPLANATORY, not code reproduction."""

    def analyze(self) -> Dict[str, Any]:
        """
        Analyze files and return structured findings.

        Returns:
            Dict with:
                - summary: Structured text summary
                - key_files: List of most relevant files
                - entities_found: Dict of entity_name -> file_path
                - critical_content: Dict of file_path -> content (for key files)
                - token_usage: Estimated tokens used
        """
        logger.info(
            f"FileAnalysisSubagent: Analyzing {len(self.files)} files for query: {self.query}"
        )
        self.console.print(
            f"  [cyan]FileAnalyst: Analyzing {len(self.files)} files for query...[/cyan]"
        )

        # Phase 1: Load skeletons for all files
        skeletons = self._load_skeletons()

        # Phase 2: Identify critical files based on skeletons
        critical_files, critical_entities = self._identify_critical_files(skeletons)

        self.console.print(
            f"  [cyan]FileAnalyst: Identified {len(critical_files)} critical files[/cyan]"
        )

        # Phase 3: Load full content for critical files
        full_content = self._load_full_content(critical_files)

        # Phase 4: Generate structured summary
        summary, entities = self._generate_summary(
            skeletons, full_content, critical_entities
        )

        # Phase 5: Calculate token usage
        total_tokens = self._estimate_tokens(summary, full_content)

        return {
            "summary": summary,
            "key_files": critical_files,
            "entities_found": entities,
            "critical_content": full_content,
            "token_usage": total_tokens,
            "evidence_envelope": self._build_evidence_envelope(
                summary=summary,
                key_files=critical_files,
                entities=entities,
                token_usage=total_tokens,
            ),
        }

    def _load_skeletons(self) -> Dict[str, str]:
        """
        Load skeletons for all files.

        Skeletons provide structure without full content (~10-20% tokens).

        Returns:
            Dict of file_path -> skeleton_content
        """
        skeletons = {}

        # Use Saguaro for skeleton extraction
        from tools.saguaro_tools import SaguaroTools
        from domains.code_intelligence.saguaro_substrate import SaguaroSubstrate

        saguaro = SaguaroSubstrate()
        saguaro_tools = SaguaroTools(saguaro)

        for file_path in self.files[:30]:  # Limit to prevent overload
            try:
                skeleton = saguaro_tools.skeleton(file_path)
                if skeleton and not skeleton.startswith("Error"):
                    skeletons[file_path] = skeleton
                    self.console.print(f"    [dim]→ Loaded skeleton: {file_path}[/dim]")
            except Exception as e:
                self.console.print(
                    f"    [yellow]⚠ Failed to load skeleton for {file_path}: {e}[/yellow]"
                )

        return skeletons

    def _identify_critical_files(
        self, skeletons: Dict[str, str]
    ) -> Tuple[List[str], Dict[str, List[str]]]:
        """
        Identify which files are most critical for the query, and relevant entities within them.

        Uses semantic matching and keyword analysis.

        Args:
            skeletons: Dict of file_path -> skeleton_content

        Returns:
            Tuple:
                - List of critical file paths (up to 10)
                - Dict of file_path -> List of critical entity names (functions/classes)
        """
        import re

        # Extract key terms from query
        query_lower = self.query.lower()
        key_terms = re.findall(r"\b[a-z_][a-z0-9_]{2,}\b", query_lower)

        scored_files = []
        critical_entities_in_files: Dict[str, List[str]] = {}

        # Regex to find class and function definitions
        # This regex is simplified and might need refinement for complex cases
        class_pattern = re.compile(r"class\s+([A-Za-z_][A-Za-z0-9_]*)\s*[:(]")
        func_pattern = re.compile(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")

        for file_path, skeleton in skeletons.items():
            score = 0.0
            skeleton_lower = skeleton.lower()
            current_file_critical_entities: List[str] = []

            # Score 1: Query terms in skeleton
            for term in key_terms:
                if term in skeleton_lower:
                    score += 2.0

            # Score 2: Important keywords (authentication, session, etc.)
            important_keywords = [
                "class",
                "def",
                "function",
                "auth",
                "session",
                "user",
                "config",
            ]
            for keyword in important_keywords:
                if keyword in skeleton_lower:
                    score += 0.5

            # Score 3: Core directories
            if any(pattern in file_path for pattern in ["/core/", "/src/", "/main"]):
                score += 1.0

            # Score 4: Python implementation files (not tests)
            if file_path.endswith(".py") and "test" not in file_path.lower():
                score += 1.0

                # Identify critical entities within Python files
                for pattern, entity_type in [
                    (class_pattern, "class"),
                    (func_pattern, "function"),
                ]:
                    for match in pattern.finditer(skeleton):
                        entity_name = match.group(1)
                        entity_lower = entity_name.lower()

                        # Check if entity name or nearby skeleton content is relevant to query
                        if any(term in entity_lower for term in key_terms) or any(
                            term
                            in skeleton_lower[
                                max(0, match.start() - 200) : min(
                                    len(skeleton_lower), match.end() + 200
                                )
                            ]
                            for term in key_terms
                        ):

                            if entity_name not in current_file_critical_entities:
                                current_file_critical_entities.append(entity_name)
                                score += 1.5  # Boost score for relevant entities

            if score > 0:
                scored_files.append((score, file_path))
                if current_file_critical_entities:
                    critical_entities_in_files[file_path] = (
                        current_file_critical_entities
                    )

        # Sort by score and return top files
        scored_files.sort(reverse=True, key=lambda x: x[0])

        critical_files = [fp for _, fp in scored_files[:10]]  # Top 10

        # Filter critical_entities_in_files to only include entities from critical_files
        filtered_critical_entities = {
            fp: critical_entities_in_files[fp]
            for fp in critical_files
            if fp in critical_entities_in_files
        }

        return critical_files, filtered_critical_entities

    def _load_full_content(self, critical_files: List[str]) -> Dict[str, str]:
        """
        Load full content for critical files.

        Args:
            critical_files: List of file paths

        Returns:
            Dict of file_path -> content
        """
        full_content = {}
        # Increased budget for critical content loading
        budget_mgr = TokenBudgetManager(200000)

        for file_path in critical_files:
            try:
                content = self.registry.dispatch("read_file", {"file_path": file_path})

                if content and not content.startswith("Error"):
                    tokens = budget_mgr.count_tokens(content)

                    if budget_mgr.fits_in_budget(tokens):
                        full_content[file_path] = content
                        budget_mgr.allocate(tokens, file_path)
                        self.console.print(
                            f"    [green]✓ Loaded full content: {file_path} ({tokens} tokens)[/green]"
                        )
                    else:
                        # Truncate to fit budget
                        from core.token_budget import smart_truncate

                        truncated = smart_truncate(content, budget_mgr.remaining())
                        full_content[file_path] = truncated
                        budget_mgr.allocate(
                            budget_mgr.count_tokens(truncated), f"{file_path}_truncated"
                        )
                        self.console.print(
                            f"    [yellow]⚠ Loaded truncated: {file_path}[/yellow]"
                        )

            except Exception as e:
                self.console.print(f"    [red]✗ Failed to load {file_path}: {e}[/red]")

        return full_content

    def _generate_summary(
        self,
        skeletons: Dict[str, str],
        full_content: Dict[str, str],
        critical_entities: Dict[str, List[str]],
    ) -> tuple:
        """
        Generate structured summary using the brain.

        Args:
            skeletons: Dict of file_path -> skeleton
            full_content: Dict of file_path -> full_content
            critical_entities: Dict of file_path -> List of critical entity names

        Returns:
            (summary_text, entities_dict)
        """
        # Build context for the model
        context_parts = []

        # Use Saguaro for slicing
        from tools.saguaro_tools import SaguaroTools
        from domains.code_intelligence.saguaro_substrate import SaguaroSubstrate

        saguaro = SaguaroSubstrate()
        saguaro_tools = SaguaroTools(saguaro)

        # Add skeletons (structure overview)
        context_parts.append("## File Structures (Skeletons)")
        for file_path, skeleton in list(skeletons.items())[:15]:
            context_parts.append(f"\n### {file_path}")
            context_parts.append(f"```\n{skeleton[:1000]}\n```")

        # Add sliced critical entities
        if critical_entities:
            context_parts.append("\n## Critical Entities (Sliced)")
            for file_path, entities in critical_entities.items():
                for entity_name in entities:
                    target = f"{file_path}.{entity_name}"
                    try:
                        sliced_content = saguaro_tools.slice(target)
                        if sliced_content and not sliced_content.startswith("Error"):
                            context_parts.append(f"\n### {target}")
                            context_parts.append(f"```python\n{sliced_content}\n```")
                    except Exception as e:
                        self.console.print(
                            f"    [yellow]⚠ Failed to slice {target}: {e}[/yellow]"
                        )

        # Add full content (implementation details)
        context_parts.append("\n## Critical File Contents")
        for file_path, content in full_content.items():
            context_parts.append(f"\n### {file_path}")
            # Limit to 5000 chars per file
            content_preview = content[:5000] if len(content) > 5000 else content
            context_parts.append(f"```\n{content_preview}\n```")

        context_text = "\n".join(context_parts)

        # Build analysis prompt
        prompt = f"""Analyze these files to answer: "{self.query}"

{context_text}

Provide a structured analysis with:

1. **Key Findings**: Direct answer to the query
2. **Critical Files**: Most relevant files (cite file:line)
3. **Entities Found**: Important classes/functions
4. **Implementation Details**: How things work
5. **Missing Information**: What's unclear or needs investigation

Be concise but thorough. Use markdown formatting."""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        # Generate summary using shared brain
        summary = ""
        try:
            for chunk in self.brain.stream_chat(
                messages, max_tokens=4000, temperature=0.0
            ):
                summary += chunk
        except Exception as e:
            summary = f"Error generating summary: {e}"

        # VALIDATION: Check if output is skeleton-like garbage
        if self._is_skeleton_output(summary):
            self.console.print(
                "    [yellow]⚠ Detected skeleton code in response, regenerating as prose...[/yellow]"
            )
            summary = self._regenerate_as_prose(summary, context_text)

        # Extract entities from summary (simple regex for now)
        import re

        entities = {}
        entity_pattern = r"`([A-Z][a-zA-Z0-9_]+)`"  # Matches `ClassName`
        for match in re.findall(entity_pattern, summary):
            # Try to find which file it's in
            for file_path in list(full_content.keys()) + list(skeletons.keys()):
                if match in skeletons.get(file_path, "") or match in full_content.get(
                    file_path, ""
                ):
                    entities[match] = file_path
                    break

        return summary, entities

    def _is_skeleton_output(self, text: str) -> bool:
        """Check if output looks like skeleton code instead of analysis."""
        if not text or len(text) < 100:
            return False
        lines = text.strip().split("\n")
        if len(lines) < 5:
            return False
        # Count lines that look like code
        code_patterns = (
            "def ",
            "class ",
            "import ",
            "from ",
            "@",
            "    def ",
            "    class ",
            "async def ",
        )
        code_lines = sum(
            1 for line_text in lines if line_text.strip().startswith(code_patterns)
        )
        # If more than 50% of lines are code, it's skeleton output
        return code_lines / len(lines) > 0.5

    def _regenerate_as_prose(self, skeleton_output: str, original_context: str) -> str:
        """Regenerate the analysis as prose when skeleton code was returned."""
        regen_prompt = f"""The previous analysis incorrectly returned code structure instead of prose explanation.

Original question: {self.query}

The code structures identified:
{skeleton_output[:3000]}

Please provide a NATURAL LANGUAGE ANALYSIS instead:
1. What does this code DO? (explain in prose)
2. What is the PURPOSE of each component? (not the code signature)  
3. How do the components work TOGETHER? (describe the flow)
4. What are the key design decisions?

OUTPUT ONLY PROSE. No code blocks except brief (3-5 line) examples to illustrate specific points."""

        messages = [
            {
                "role": "system",
                "content": "You are an expert code analyst who explains code in clear, natural language. Never output raw code structures.",
            },
            {"role": "user", "content": regen_prompt},
        ]

        result = ""
        try:
            for chunk in self.brain.stream_chat(
                messages, max_tokens=3000, temperature=0.1
            ):
                result += chunk
        except Exception as e:
            result = f"Regeneration failed: {e}\n\nOriginal output:\n{skeleton_output}"

        return result

    def _estimate_tokens(self, summary: str, full_content: Dict[str, str]) -> int:
        """Estimate total tokens used in analysis."""
        budget_mgr = TokenBudgetManager(0)  # Just for counting

        total = budget_mgr.count_tokens(summary)
        for content in full_content.values():
            total += budget_mgr.count_tokens(content)

        return total

    def _build_evidence_envelope(
        self,
        *,
        summary: str,
        key_files: List[str],
        entities: Dict[str, str],
        token_usage: int,
    ) -> Dict[str, Any]:
        return {
            "schema_version": "evidence_envelope.v1",
            "producer": "FileAnalysisSubagent",
            "evidence_type": "multi_file_analysis",
            "summary": summary[:400],
            "query": self.query,
            "sources": [{"path": path, "role": "analysis_local"} for path in key_files],
            "artifacts": [],
            "metrics": {
                "token_usage": token_usage,
                "key_file_count": len(key_files),
                "entity_count": len(entities),
            },
        }
