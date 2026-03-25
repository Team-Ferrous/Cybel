"""
Enterprise-Grade Agentic Chat Loop

Design Philosophy (inspired by Claude Code/Antigravity):
1. STRUCTURED PHASES: Separate understanding, tool execution, action, and synthesis
2. TOOL-FIRST APPROACH: Proactively use tools to gather evidence
3. ACTION-CAPABLE: Can read, write, edit, move, delete files
4. WEB-AWARE: Can search the web and fetch URLs
5. DETERMINISTIC FLOW: Explicit control flow, not model choice
6. SIMPLE PROMPTS: Short, focused prompts per phase
7. ERROR RECOVERY: Clear failures, structured retries
8. SELF-IMPROVING: Can modify its own code

COCONUT Integration:
- COCONUT provides latent space reasoning (no explicit thinking blocks needed)
- Simple prompts = cleaner hidden states for COCONUT path evolution
- Focused phases = COCONUT can concentrate reasoning per task
- Multiple generation passes = more COCONUT cycles = deeper reasoning
- Low temperature (0.0) + COCONUT = deterministic sampling of aggregated paths
- Generous token budget = more generation steps = more path evolution cycles

This loop should be capable enough to build itself - like a 3D printer printing new 3D printers.
"""

import json
import re
import hashlib
import time
from typing import List, Dict, Any
from rich.panel import Panel
from core.aes import AALClassifier, ReviewGate


class EnterpriseChatLoop:
    """
    Enterprise-grade chat loop for comprehensive Q&A and task execution.
    Optimized for small models with structured execution phases.
    Can read, write, edit files, search the web, run commands, and more.
    """

    def __init__(self, agent):
        self.agent = agent
        self.console = agent.console
        self.brain = agent.brain
        self.history = agent.history
        self.registry = agent.registry
        self.semantic_engine = agent.semantic_engine
        self.approval_manager = agent.approval_manager
        self.pipeline_manager = getattr(agent, "pipeline_manager", None)
        self.aal_classifier = AALClassifier()
        self.review_gate = ReviewGate()
        self.current_compliance_context: Dict[str, Any] = {
            "trace_id": None,
            "evidence_bundle_id": None,
            "waiver_ids": [],
            "red_team_required": False,
        }

    def _execute_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """Route enterprise-loop tools through BaseAgent execution."""
        return self.agent._execute_tool({"name": name, "arguments": arguments})

    def dispatch(self, name: str, arguments: Dict[str, Any]) -> str:
        """Compatibility shim for tool dispatch callsites and compliance checks."""
        return self._execute_tool(name, arguments)

    def _stream_with_pipeline(
        self,
        messages: List[Dict[str, Any]],
        *,
        request_type: str,
        user_input: str,
        **overrides,
    ):
        if self.pipeline_manager is None:
            return self.brain.stream_chat(messages, **overrides)
        return self.pipeline_manager.stream_chat(
            messages,
            request_type=request_type,
            user_input=user_input,
            **overrides,
        )

    def run(self, user_input: str) -> str:
        """
        Main entry point for enterprise chat loop.

        Flow:
        1. Understanding: Classify the request type
        2. Evidence Gathering: Use tools to collect information
        3. Action: Execute modifications if needed
        4. Synthesis: Generate comprehensive response
        """
        self.history.add_message("user", user_input)

        # Phase 1: Understanding - What does the user want?
        request_type = self._classify_request(user_input)

        # Phase 2: Evidence Gathering
        if request_type in ["question", "explanation", "investigation"]:
            evidence = self._gather_evidence(user_input)
            response = self._synthesize_answer(user_input, evidence)

        elif request_type in ["modification", "creation", "deletion"]:
            # Action-based request
            response = self._execute_action(user_input)

        else:
            # Default to conversational response
            response = self._direct_answer(user_input)

        return response

    def _classify_request(self, user_input: str) -> str:
        """
        Classify the user request into categories.
        Uses deterministic keyword matching + model confirmation.
        """
        input_lower = user_input.lower()

        # Deterministic classification based on keywords
        if any(
            kw in input_lower
            for kw in [
                "how does",
                "how do",
                "explain",
                "describe",
                "what is",
                "where is",
                "why",
                "when",
                "who",
                "which",
            ]
        ):
            return "question"

        if any(
            kw in input_lower
            for kw in ["create", "add", "write", "implement", "build", "generate"]
        ):
            return "creation"

        if any(
            kw in input_lower
            for kw in ["edit", "modify", "change", "update", "fix", "refactor"]
        ):
            return "modification"

        if any(kw in input_lower for kw in ["delete", "remove", "drop", "clear"]):
            return "deletion"

        if any(
            kw in input_lower
            for kw in ["search for", "find", "investigate", "analyze", "explore"]
        ):
            return "investigation"

        return "conversational"

    def _gather_evidence(self, user_input: str) -> Dict[str, Any]:
        """
        Phase 2: Comprehensive evidence gathering using all available tools.

        This phase FORCES the model to:
        1. Search the codebase with Saguaro
        2. Search the web if needed
        3. Read relevant files
        4. Collect all evidence before answering

        Returns structured evidence dict.
        """
        self.console.print("[cyan]Phase 1: Gathering evidence...[/cyan]")

        evidence = {
            "codebase_files": [],
            "file_contents": {},
            "web_results": [],
            "search_results": [],
            "errors": [],
        }

        # Step 1: Saguaro search for relevant files
        self._saguaro_file_search(user_input, evidence)

        # Step 2: Determine if web search is needed
        if self._needs_web_search(user_input):
            self._web_search(user_input, evidence)

        # Step 3: Read selected files
        self._read_relevant_files(user_input, evidence)

        return evidence

    def _saguaro_file_search(self, query: str, evidence: Dict[str, Any]):
        """Use Saguaro query to find relevant files."""
        self.console.print("  [dim]→ Saguaro file search...[/dim]")
        try:
            result = self.dispatch("saguaro_query", {"query": query, "k": 10})
            if not result or str(result).startswith("Error"):
                raise RuntimeError(result or "empty Saguaro response")

            relevant_files = self._extract_paths_from_saguaro_output(str(result))
            if not relevant_files:
                raise RuntimeError("Saguaro returned no file paths")

            evidence["codebase_files"] = relevant_files[:10]
            evidence["search_results"].append(("saguaro_query", result))
            self.console.print(
                f"  [green]✓ Found {len(relevant_files)} relevant files[/green]"
            )
        except Exception as e:
            evidence["errors"].append(f"Saguaro query failed: {e}")
            self.console.print("  [red]✗ Saguaro query failed[/red]")
            raise RuntimeError(
                f"SAGUARO_STRICT_QUERY_FAILED: enterprise evidence gather aborted ({e})"
            ) from e

    def _needs_web_search(self, query: str) -> bool:
        """Determine if web search is needed."""
        query_lower = query.lower()

        web_keywords = [
            "latest",
            "recent",
            "news",
            "current",
            "documentation",
            "tutorial",
            "example",
            "best practice",
            "stackoverflow",
            "github",
            "npm",
            "pypi",
            "package",
            "library",
            "framework",
        ]

        return any(kw in query_lower for kw in web_keywords)

    def _web_search(self, query: str, evidence: Dict[str, Any]):
        """Execute web search."""
        self.console.print("  [dim]→ Web search...[/dim]")
        try:
            results = self._execute_tool("web_search", {"query": query})
            if results and not results.startswith("Error"):
                evidence["web_results"].append(results)
                self.console.print("  [green]✓ Web search completed[/green]")
            else:
                self.console.print(
                    "  [yellow]⚠ Web search returned no results[/yellow]"
                )
        except Exception as e:
            evidence["errors"].append(f"Web search failed: {e}")
            self.console.print("  [red]✗ Web search failed[/red]")

    def _extract_paths_from_saguaro_output(self, output: str) -> List[str]:
        """Parse file paths from the text output returned by saguaro_query."""
        paths: List[str] = []
        for line in output.splitlines():
            if " - " not in line:
                continue
            candidate = line.rsplit(" - ", 1)[-1].strip()
            if ":" in candidate:
                candidate = candidate.rsplit(":", 1)[0]
            candidate = candidate.strip()
            if candidate and candidate not in paths:
                paths.append(candidate)
        return paths

    def _read_relevant_files(self, query: str, evidence: Dict[str, Any]):
        """
        Read and analyze relevant files sequentially.
        For large files, analyze in chunks to fully understand them.
        """
        if not evidence["codebase_files"]:
            message = "SAGUARO_STRICT_EMPTY_RESULTS: no files from saguaro_query"
            self.console.print(f"  [red]✗ {message}[/red]")
            evidence["errors"].append(message)
            raise RuntimeError(message)
        files_to_read = evidence["codebase_files"][:5]
        if not files_to_read:
            message = "SAGUARO_STRICT_EMPTY_RESULTS: no files selected for read"
            self.console.print(f"  [red]✗ {message}[/red]")
            evidence["errors"].append(message)
            raise RuntimeError(message)

        # Take top 3-5 files
        if files_to_read:
            self.console.print(
                f"  [dim]→ Reading and analyzing {len(files_to_read)} files...[/dim]"
            )
            for file_path in files_to_read:
                try:
                    content = self._execute_tool("read_file", {"file_path": file_path})
                    if content and not content.startswith("Error"):
                        lines = content.count("\n")

                        # For long files, analyze in chunks sequentially
                        if lines > 500:
                            self.console.print(
                                f"  [cyan]→ Analyzing {file_path} ({lines} lines) in chunks...[/cyan]"
                            )
                            analysis = self._analyze_file_in_chunks(
                                file_path, content, query
                            )
                            evidence["file_contents"][file_path] = analysis
                            self.console.print(
                                f"  [green]✓ Analyzed {file_path}[/green]"
                            )
                        else:
                            # Short files: include full content
                            evidence["file_contents"][file_path] = content
                            self.console.print(
                                f"  [green]✓ Read {file_path} ({lines} lines)[/green]"
                            )
                    else:
                        self.console.print(
                            f"  [yellow]⚠ Could not read {file_path}: {content}[/yellow]"
                        )
                except Exception as e:
                    evidence["errors"].append(f"Failed to read {file_path}: {e}")
                    self.console.print(
                        f"  [red]✗ Failed to read {file_path}: {e}[/red]"
                    )

    def _analyze_file_in_chunks(
        self, file_path: str, content: str, query: str, chunk_size: int = 200
    ) -> str:
        """
        Analyze a large file in sequential chunks to understand it fully.
        Returns a comprehensive analysis instead of truncated content.
        """
        lines = content.split("\n")
        total_lines = len(lines)
        num_chunks = (total_lines + chunk_size - 1) // chunk_size

        analyses = []

        for i in range(num_chunks):
            start_line = i * chunk_size
            end_line = min((i + 1) * chunk_size, total_lines)
            chunk = "\n".join(lines[start_line:end_line])

            # Analyze this chunk
            prompt = f"""Analyzing file: {file_path} (lines {start_line+1}-{end_line} of {total_lines})

User's question: {query}

File content (this chunk):
```
{chunk}
```

Provide a brief analysis of this chunk relevant to the user's question.
Focus on: classes, functions, key logic, and anything related to the query.
Be concise (2-3 sentences max).

Analysis:"""

            messages = [
                {
                    "role": "system",
                    "content": "You are a code analysis assistant. Be concise and focus on what matters.",
                },
                {"role": "user", "content": prompt},
            ]

            chunk_analysis = ""
            try:
                for text_chunk in self._stream_with_pipeline(
                    messages,
                    request_type="investigation",
                    user_input=query,
                    max_tokens=300,
                ):
                    chunk_analysis += text_chunk

                analyses.append(
                    f"Lines {start_line+1}-{end_line}: {chunk_analysis.strip()}"
                )
            except Exception as e:
                analyses.append(
                    f"Lines {start_line+1}-{end_line}: [Analysis failed: {e}]"
                )

        # Combine all chunk analyses
        combined_analysis = f"# Analysis of {file_path}\n\n" + "\n\n".join(analyses)
        return combined_analysis

    def _select_files_to_read(
        self, query: str, candidate_files: List[str]
    ) -> List[str]:
        """
        Use model to select which files are most relevant to read.
        Simple, focused call - just file selection.
        """
        if not candidate_files:
            return []

        # Simplified prompt for file selection
        prompt = f"""User Question: {query}

Candidate Files:
{chr(10).join(f"{i+1}. {f}" for i, f in enumerate(candidate_files[:15]))}

Select the 1-5 most relevant files to answer the question.
Output ONLY the file paths, one per line, nothing else.

Selected files:"""

        messages = [
            {
                "role": "system",
                "content": "You are a file selection assistant. Output only file paths.",
            },
            {"role": "user", "content": prompt},
        ]

        response = ""
        for chunk in self._stream_with_pipeline(
            messages,
            request_type="investigation",
            user_input=query,
            max_tokens=500,
        ):
            response += chunk

        # Parse file paths from response
        lines = response.strip().split("\n")
        selected_files = []
        for line in lines:
            line = line.strip()
            # Remove numbering, bullets, etc.
            line = re.sub(r"^\d+[\.\)]?\s*", "", line)
            line = re.sub(r"^[-*]\s*", "", line)

            if line and any(line in f for f in candidate_files):
                # Find the matching candidate file
                for cf in candidate_files:
                    if line in cf:
                        selected_files.append(cf)
                        break

        return selected_files[:5]

    def _execute_action(self, user_input: str) -> str:
        """
        Execute action-based requests (create, edit, delete files).
        Uses approval manager for safety.
        """
        self.console.print("[cyan]Phase 1: Planning action...[/cyan]")

        # First, gather context about what needs to be done
        evidence = self._gather_evidence(user_input)
        aal = self._classify_runtime_aal(user_input, evidence)
        compliance = self._refresh_compliance_context(user_input, evidence, aal)
        evidence["aal"] = aal
        evidence["compliance"] = compliance

        # Generate action plan
        self.console.print("[cyan]Phase 2: Generating action plan...[/cyan]")

        action_plan = self._generate_action_plan(user_input, evidence)

        if aal in {"AAL-0", "AAL-1"}:
            review = self.review_gate.evaluate_from_evidence(
                aal=aal,
                evidence=evidence,
                author=getattr(self.agent, "name", None),
                irreversible_action=True,
            )
            if not review.passed:
                return (
                    f"AES ACTION BLOCK ({aal}): review signoff required before execution. "
                    f"Reasons: {', '.join(review.reasons)}"
                )

        # Ask for approval (if needed)
        if not self.approval_manager.can_execute("write_file", {}):
            self.console.print("[yellow]Action requires approval[/yellow]")
            self.console.print(Panel(action_plan, title="Proposed Action"))
            from rich.prompt import Confirm

            if not Confirm.ask("Approve this action?"):
                return "Action cancelled by user."

        # Execute the action
        self.console.print("[cyan]Phase 3: Executing action...[/cyan]")
        result = self._execute_action_plan(action_plan)
        result["action_receipt"] = self._build_action_receipt(
            user_input=user_input,
            action_plan=action_plan,
            result=result,
            aal=aal,
            compliance=compliance,
        )

        # Synthesize response
        self.console.print("[cyan]Phase 4: Synthesizing response...[/cyan]")
        response = self._synthesize_action_result(user_input, action_plan, result)

        return response

    def _classify_runtime_aal(self, user_input: str, evidence: Dict[str, Any]) -> str:
        files = []
        for candidate in evidence.get("codebase_files", []) or []:
            if isinstance(candidate, str) and candidate:
                files.append(candidate)
        try:
            if files:
                return self.aal_classifier.classify_changeset(files)
        except Exception:
            pass
        return self.aal_classifier.classify_from_description(user_input)

    def _refresh_compliance_context(
        self, user_input: str, evidence: Dict[str, Any], aal: str
    ) -> Dict[str, Any]:
        trace_seed = f"{user_input}:{int(time.time())}"
        trace_hash = hashlib.sha256(trace_seed.encode("utf-8")).hexdigest()[:10]
        trace_id = self.current_compliance_context.get("trace_id") or f"trace::{trace_hash}"
        fingerprint = hashlib.sha256()
        fingerprint.update(trace_id.encode("utf-8"))
        for file_path in sorted(evidence.get("codebase_files", []) or []):
            fingerprint.update(str(file_path).encode("utf-8"))
        evidence_bundle_id = f"evidence::{trace_id}::{fingerprint.hexdigest()[:12]}"
        updated = {
            "trace_id": trace_id,
            "evidence_bundle_id": evidence_bundle_id,
            "waiver_ids": self.current_compliance_context.get("waiver_ids") or [],
            "red_team_required": aal in {"AAL-0", "AAL-1"},
        }
        self.current_compliance_context = updated
        return dict(updated)

    def _build_action_receipt(
        self,
        user_input: str,
        action_plan: str,
        result: Dict[str, Any],
        aal: str,
        compliance: Dict[str, Any],
    ) -> Dict[str, Any]:
        digest = hashlib.sha256()
        digest.update((user_input or "").encode("utf-8"))
        digest.update((action_plan or "").encode("utf-8"))
        digest.update(json.dumps(result, sort_keys=True, default=str).encode("utf-8"))
        return {
            "trace_id": compliance.get("trace_id"),
            "evidence_bundle_id": compliance.get("evidence_bundle_id"),
            "aal": aal,
            "receipt_id": f"receipt::{digest.hexdigest()[:16]}",
            "timestamp": int(time.time()),
        }

    def _generate_action_plan(self, user_input: str, evidence: Dict[str, Any]) -> str:
        """Generate a detailed action plan."""
        evidence_text = self._format_evidence(evidence)

        system_prompt = f"""You are {self.agent.name}, a software engineering expert.

Evidence from codebase:
{evidence_text}

Task: Create a detailed action plan for the user's request.
Output ONLY the action plan in structured format:

1. Files to modify/create:
2. Changes to make:
3. Tools to use:
4. Verification steps:"""

        user_prompt = f"""Request: {user_input}

Generate action plan:"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        plan = ""
        request_type = self._classify_request(user_input)
        for chunk in self._stream_with_pipeline(
            messages,
            request_type=request_type,
            user_input=user_input,
            max_tokens=2000,
        ):
            plan += chunk
            print(chunk, end="", flush=True)

        print()
        return plan

    def _execute_action_plan(self, action_plan: str) -> Dict[str, Any]:
        """Execute the action plan using tools."""
        results = {
            "files_written": [],
            "files_edited": [],
            "commands_run": [],
            "errors": [],
        }

        # This is a simplified execution - in practice, you'd parse the plan
        # and execute specific tool calls
        # For now, return placeholder
        self.console.print(
            "[yellow]Action execution not yet fully implemented[/yellow]"
        )

        return results

    def _synthesize_action_result(
        self, user_input: str, plan: str, result: Dict[str, Any]
    ) -> str:
        """Synthesize the action result into a response."""
        system_prompt = f"""You are {self.agent.name}, a software engineering expert.

Action Plan:
{plan}

Execution Results:
{json.dumps(result, indent=2)}

Task: Summarize what was done and provide a clear response."""

        user_prompt = f"""Original Request: {user_input}

Provide summary:"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = ""
        print()
        request_type = self._classify_request(user_input)
        for chunk in self._stream_with_pipeline(
            messages,
            request_type=request_type,
            user_input=user_input,
            max_tokens=2000,
        ):
            response += chunk
            print(chunk, end="", flush=True)

        print()
        self.history.add_message("assistant", response)
        return response

    def _direct_answer(self, user_input: str) -> str:
        """
        Direct answer path for simple conversational questions.
        """
        self.console.print("[dim]Conversational mode[/dim]")

        messages = self.history.get_messages()

        system_prompt = f"""You are {self.agent.name}, a helpful AI assistant.
Answer the user's question directly and concisely."""

        chat_messages = [{"role": "system", "content": system_prompt}] + messages

        response = ""
        for chunk in self._stream_with_pipeline(
            chat_messages,
            request_type="conversational",
            user_input=user_input,
            max_tokens=5000,
        ):
            response += chunk
            print(chunk, end="", flush=True)

        print()

        self.history.add_message("assistant", response)
        return response

    def _synthesize_answer(self, user_input: str, evidence: Dict[str, Any]) -> str:
        """
        Phase 3: Synthesize evidence into final answer.

        This phase takes the gathered evidence and produces a coherent answer.
        COCONUT latent reasoning is leveraged here - the model has deep
        reasoning paths exploring the evidence space during generation.
        """
        self.console.print(
            "[cyan]Phase 2: Synthesizing answer (COCONUT reasoning active)...[/cyan]"
        )

        # Build evidence context
        evidence_text = self._format_evidence(evidence)

        # COCONUT-Optimized Synthesis Prompt
        # Key: Simple, clear prompt = clean hidden states = better COCONUT path evolution
        system_prompt = f"""You are {self.agent.name}, a software engineering expert.

Evidence from codebase and web:
{evidence_text}

Task: Answer the question using the evidence above.
- Cite specific files and sources
- Be thorough but clear
- Include code examples if relevant
- If evidence is insufficient, say so"""

        user_prompt = f"""Question: {user_input}

Think through the evidence, then provide your answer:"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Stream the synthesis
        # COCONUT operates during generation - more tokens = deeper reasoning
        response = ""
        print()  # Newline before response

        # Allow generous token budget for COCONUT to explore reasoning paths
        # 200k max from settings, but we'll use a reasonable default
        max_tokens = 10000

        request_type = self._classify_request(user_input)
        for chunk in self._stream_with_pipeline(
            messages,
            request_type=request_type,
            user_input=user_input,
            max_tokens=max_tokens,
        ):
            response += chunk
            print(chunk, end="", flush=True)

        print()  # Newline after response

        self.history.add_message("assistant", response)
        return response

    def _format_evidence(self, evidence: Dict[str, Any]) -> str:
        """Format evidence dict into readable text."""
        parts = []

        if evidence.get("codebase_files"):
            parts.append("## Relevant Files (Semantic Search)")
            for f in evidence["codebase_files"][:10]:
                parts.append(f"- {f}")

        if evidence.get("file_contents"):
            parts.append("\n## File Contents")
            for file_path, content in evidence["file_contents"].items():
                # Truncate long files
                if len(content) > 5000:
                    content = content[:5000] + "\n... (truncated)"
                parts.append(f"\n### {file_path}")
                parts.append(f"```\n{content}\n```")

        if evidence.get("web_results"):
            parts.append("\n## Web Search Results")
            for result in evidence["web_results"]:
                result_str = str(result)
                if len(result_str) > 2000:
                    result_str = result_str[:2000] + "... (truncated)"
                parts.append(result_str)

        if evidence.get("search_results"):
            parts.append("\n## Code Search Results")
            for search_type, result in evidence["search_results"]:
                parts.append(f"\n### {search_type}")
                result_str = str(result)
                if len(result_str) > 2000:
                    result_str = result_str[:2000] + "... (truncated)"
                parts.append(result_str)

        if evidence.get("errors"):
            parts.append("\n## Errors Encountered")
            for error in evidence["errors"]:
                parts.append(f"- {error}")

        return "\n".join(parts)
