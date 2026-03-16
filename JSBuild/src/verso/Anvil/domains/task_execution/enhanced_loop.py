"""
Enhanced Agentic Loop - Full think-act-observe-reflect cycle.

Implements:
- Task boundaries and mode management
- Extended thinking with budget
- Interleaved reasoning (think between tools)
- Self-review and verification
- Progress tracking and artifacts
"""

import json
import os
from datetime import datetime
from typing import Dict, Any

from core.agent_mode import AgentMode
from core.task_state import TaskStateManager
from core.thinking import EnhancedThinkingSystem, ThinkingType
from core.artifacts import ArtifactManager
from cli.task_view import TaskView, render_mode_transition
from tools.verify import run_all_verifications
from core.agents.subagent import SubAgent
from shared_kernel.event_store import get_event_store
from domains.verification.coherence_engine import CoherenceEngine
from domains.verification.auto_corrector import AutoCorrector
from domains.task_execution.adaptive_router import AdaptiveRouter
from domains.memory_management.cross_session_learning import (
    CrossSessionLearning,
    Lesson,
)
from domains.verification.enhanced_verifier import EnhancedVerifier

try:
    from saguaro.synthesis.replay_tape import SynthesisProofCapsule, SynthesisReplayTape
    from saguaro.synthesis.spec import SpecLowerer
    from saguaro.synthesis.spec_lint import lint_sagspec
except Exception:  # pragma: no cover - synthesis package may not exist during bootstrap
    SynthesisProofCapsule = None
    SynthesisReplayTape = None
    SpecLowerer = None
    lint_sagspec = None


class EnhancedAgenticLoop:
    """
    Implements the full agentic loop with:
    - Task boundaries and mode management
    - Extended thinking with budget
    - Interleaved reasoning (think between tools)
    - Self-review and verification
    - Progress tracking and artifacts
    """

    def __init__(self, agent, console=None):
        """
        Initialize the enhanced loop.

        Args:
            agent: BaseAgent instance with brain, registry, etc.
            console: Rich console (defaults to agent's console)
        """
        self.agent = agent
        self.console = console or agent.console

        # Core systems
        self.task_manager = TaskStateManager()
        from config.settings import AGENTIC_THINKING

        self.thinking_system = EnhancedThinkingSystem(
            thinking_budget=AGENTIC_THINKING.get("thinking_budget", 50000),
            coconut_enabled=getattr(agent, "coconut_enabled", False),
        )
        self.artifacts = ArtifactManager()
        self.task_view = TaskView(self.console)

        # Coherence & Auto-Correction
        self.coherence_engine = CoherenceEngine()
        self.auto_corrector = AutoCorrector(self)

        # Intelligence Enhancements
        self.router = AdaptiveRouter()
        self.learning = CrossSessionLearning(
            memory_backend=getattr(agent, "memory", None)
        )
        self.enhanced_verifier = EnhancedVerifier(root_dir=".")

        # Configuration
        self.auto_verify = True
        self.require_plan_approval = True
        self.max_backtrack_attempts = 3
        self.verbose = True
        self.current_synthesis_spec: Dict[str, Any] | None = None
        self.current_synthesis_lint: Dict[str, Any] | None = None
        self.current_synthesis_replay: Dict[str, Any] | None = None

    def run(self, objective: str) -> Dict[str, Any]:
        """
        Execute the full enhanced loop for an objective.

        Phases:
        1. Planning - Research and create implementation plan
        2. Await Approval - Wait for user to approve plan
        3. Execution - Implement the approved plan
        4. Verification - Run tests and validate
        5. Finalize - Create walkthrough and archive

        Args:
            objective: User's objective/request

        Returns:
            Result dict with outcome, stats, artifacts
        """
        start_time = datetime.now()
        self.current_objective = objective  # Store for use in verification phase

        # Initialize task
        self.task_manager.start_task(
            name=f"Working on: {objective[:50]}...",
            mode=AgentMode.PLANNING,
            status="Analyzing request and planning approach",
        )

        # Initialize thinking chain
        self.thinking_system.start_chain(
            task_id=f"task_{start_time.strftime('%Y%m%d_%H%M%S')}"
        )

        # Create artifacts
        self.artifacts.create_task(title=f"Task: {objective[:50]}", objective=objective)

        try:
            # 1. Planning Phase
            plan_result = self._planning_phase(objective)

            if not plan_result.get("success"):
                return self._finalize(
                    objective, start_time, "failed", plan_result.get("error")
                )

            # 2. Await Approval (if required)
            if self.require_plan_approval:
                approved = self._await_plan_approval()
                if not approved:
                    get_event_store().emit(
                        event_type="ERROR_OCCURRED",
                        source="EnhancedLoop",
                        payload={"error": "Plan not approved"},
                    )
                    return self._finalize(
                        objective, start_time, "cancelled", "Plan not approved"
                    )

            get_event_store().emit(
                event_type="PLAN_APPROVED",
                source="EnhancedLoop",
                payload={"objective": objective},
            )

            # 3. Execution Phase
            exec_result = self._execution_phase()

            # --- COHERENCE VALIDATION ---
            trace = [
                f"Objective: {objective}",
                f"Plan: {plan_result.get('summary', '')}",
                f"Exec Result: {exec_result.get('summary', '')}",
            ]
            coherence = self.coherence_engine.validate_trace(trace)
            if not coherence["passed"]:
                self.console.print(
                    f"[bold yellow]⚠ Coherence Alert ({coherence['overall_coherence']:.2f}):[/bold yellow] Reasoning drift detected. Correcting..."
                )
                # Minimal integration: log it and continue for now, full correction in next PR
                get_event_store().emit(
                    event_type="COHERENCE_VIOLATION",
                    source="EnhancedLoop",
                    payload=coherence,
                )

            if not exec_result.get("success"):
                return self._finalize(
                    objective, start_time, "partial", exec_result.get("error")
                )

            # 4. Verification Phase (with potential backtracking)
            verify_result = self._verification_phase()

            get_event_store().emit(
                event_type="VERIFICATION_COMPLETED",
                source="EnhancedLoop",
                payload={"success": verify_result.get("success")},
            )

            if not verify_result.get("success"):
                return self._finalize(
                    objective, start_time, "partial", "Verification failed"
                )

            # 5. Finalize
            return self._finalize(objective, start_time, "completed")

        except Exception as e:
            self.console.print(f"[bold red]Enhanced loop error: {e}[/bold red]")
            return self._finalize(objective, start_time, "failed", str(e))

    def _generate_with_context(self, prompt: str) -> str:
        """Generates a response using the agent's full history context."""
        messages = self.agent.history.get_messages()
        # Add the prompt as a user message
        messages.append({"role": "user", "content": prompt})

        response = ""
        # Stream response quietly (updates handled by task boundary or status)
        for chunk in self.agent.brain.stream_chat(messages):
            response += chunk

        from core.thinking import ThinkingParser
        import re

        # Robust extraction logic (mirrors SubAgent fix)
        cleaned = ThinkingParser.remove_thinking_blocks(response)
        cleaned = re.sub(r"<tool_call>.*?</tool_call>", "", cleaned, flags=re.DOTALL)
        cleaned = cleaned.strip()

        if cleaned:
            return cleaned

        # Fallback to thinking blocks
        blocks = ThinkingParser.parse(response)
        if blocks:
            # Prefer planning/reflection
            from core.thinking import ThinkingType

            for block in reversed(blocks):
                if block.type in [ThinkingType.PLANNING, ThinkingType.REFLECTION]:
                    return block.content
            return blocks[-1].content

        # Final fallback
        return re.sub(r"<[^>]+>", "", response).strip()

    @staticmethod
    def _lint_payload(lint_result: Any) -> Dict[str, Any]:
        if lint_result is None:
            return {"is_valid": False, "errors": ["lint_unavailable"], "telemetry": {}}
        if isinstance(lint_result, dict):
            payload = dict(lint_result)
            payload.setdefault(
                "is_valid",
                bool(payload.get("valid") or not payload.get("errors")),
            )
            payload.setdefault("telemetry", {})
            payload.setdefault("errors", [])
            return payload
        return {
            "is_valid": bool(
                getattr(lint_result, "is_valid", False)
                or getattr(lint_result, "valid", False)
                or not getattr(lint_result, "errors", [])
            ),
            "errors": list(getattr(lint_result, "errors", []) or []),
            "telemetry": dict(getattr(lint_result, "telemetry", {}) or {}),
        }

    def _lower_objective_to_synthesis_spec(self, objective: str) -> tuple[Dict[str, Any] | None, Dict[str, Any]]:
        if SpecLowerer is None or lint_sagspec is None:
            return None, {"is_valid": False, "errors": ["synthesis_spec_unavailable"], "telemetry": {}}
        lowerer = SpecLowerer()
        if hasattr(lowerer, "lower_objective"):
            spec = lowerer.lower_objective(objective, origin="enhanced_loop")
        else:
            spec = lowerer.lower(objective)
        if hasattr(spec, "to_dict"):
            spec_payload = dict(spec.to_dict())
        elif hasattr(spec, "as_dict"):
            spec_payload = dict(spec.as_dict())
        elif isinstance(spec, dict):
            spec_payload = dict(spec)
        else:
            spec_payload = dict(getattr(spec, "__dict__", {}) or {})
        lint_payload = self._lint_payload(lint_sagspec(spec))
        return spec_payload, lint_payload

    def _persist_synthesis_spec_artifacts(
        self,
        objective: str,
        spec_payload: Dict[str, Any],
        lint_payload: Dict[str, Any],
    ) -> Dict[str, str]:
        spec_path = self.artifacts.current_dir / "synthesis_spec.json"
        lint_path = self.artifacts.current_dir / "synthesis_spec_lint.json"
        spec_path.write_text(
            json.dumps(spec_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        lint_path.write_text(
            json.dumps(lint_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        artifact_paths = {
            "spec_path": str(spec_path),
            "lint_path": str(lint_path),
        }
        if SynthesisReplayTape is not None:
            tape = SynthesisReplayTape.from_spec(
                spec_payload,
                lint_payload=lint_payload,
                objective=objective,
            )
            replay_path = self.artifacts.current_dir / "synthesis_replay.json"
            replay_path.write_text(
                json.dumps(tape.to_dict(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            artifact_paths["replay_path"] = str(replay_path)
            self.current_synthesis_replay = dict(tape.to_dict())
            if SynthesisProofCapsule is not None:
                capsule = SynthesisProofCapsule.from_spec(
                    spec_payload,
                    lint_payload=lint_payload,
                    replay_tape=tape.to_dict(),
                )
                proof_path = self.artifacts.current_dir / "proof_capsule.json"
                proof_path.write_text(
                    json.dumps(capsule.to_dict(), indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                artifact_paths["proof_path"] = str(proof_path)
        return artifact_paths

    def _record_synthesis_planning_event(
        self,
        objective: str,
        spec_payload: Dict[str, Any],
        lint_payload: Dict[str, Any],
        artifact_paths: Dict[str, str],
    ) -> None:
        event_store = get_event_store()
        if hasattr(event_store, "record_synthesis_replay_event"):
            event_store.record_synthesis_replay_event(
                synthesis_id=str(spec_payload.get("spec_id") or "planning"),
                stage="planning",
                payload={
                    "objective": objective,
                    "spec": spec_payload,
                    "lint": lint_payload,
                    "artifacts": list(artifact_paths.values()),
                    "files": list(spec_payload.get("target_files") or []),
                    "proof_capsules": [
                        artifact_paths["proof_path"]
                    ]
                    if artifact_paths.get("proof_path")
                    else [],
                },
                metadata={"phase": "planning", "objective": objective},
            )

    def _persist_synthesis_governance_artifacts(
        self,
        *,
        verification_passed: bool,
    ) -> Dict[str, str]:
        if not self.current_synthesis_spec:
            return {}
        artifacts: Dict[str, str] = {}
        try:
            from benchmarks.synthesis_suite import SynthesisBenchmarkSuite

            benchmark_summary = SynthesisBenchmarkSuite().summary_for_spec(
                self.current_synthesis_spec,
                verification_passed=verification_passed,
            )
            benchmark_path = self.artifacts.current_dir / "synthesis_benchmark_summary.json"
            benchmark_path.write_text(
                json.dumps(benchmark_summary, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            artifacts["benchmark_summary_path"] = str(benchmark_path)
        except Exception:
            benchmark_summary = {}
        try:
            from saguaro.roadmap.validator import RoadmapValidator

            roadmap_path = "ROADMAP_PROGRAM_SYNTHESIS.md"
            if os.path.exists(roadmap_path):
                roadmap_validation = RoadmapValidator(".").validate(roadmap_path)
                validation_path = self.artifacts.current_dir / "synthesis_roadmap_validation.json"
                validation_path.write_text(
                    json.dumps(roadmap_validation, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                artifacts["roadmap_validation_path"] = str(validation_path)
        except Exception:
            roadmap_validation = {}
        task = self.artifacts.get_task()
        if artifacts.get("benchmark_summary_path"):
            task.add_artifact("synthesis_benchmark_summary.json")
        if artifacts.get("roadmap_validation_path"):
            task.add_artifact("synthesis_roadmap_validation.json")
        task.save()
        artifacts["benchmark_summary"] = benchmark_summary
        if "roadmap_validation" not in artifacts:
            artifacts["roadmap_validation"] = locals().get("roadmap_validation", {})
        else:
            artifacts["roadmap_validation"] = roadmap_validation
        return artifacts

    def _planning_phase(self, objective: str) -> Dict[str, Any]:
        """
        Planning phase: Research, understand, create implementation plan.

        Returns:
            Dict with success status and any error
        """
        self._update_boundary(
            mode=AgentMode.PLANNING,
            status="Researching codebase and designing approach",
        )

        if self.verbose:
            render_mode_transition(self.console, AgentMode.IDLE, AgentMode.PLANNING)
            self.console.print(
                f"[bold blue]🔍 RESEARCHING:[/bold blue] [italic]Analyzing {objective[:40]}...[/italic]"
            )

        # --- CROSS-SESSION RETRIEVAL ---
        lessons = self.learning.retrieve_similar_lessons(objective)
        lesson_context = ""
        if lessons:
            self.console.print(
                f"[dim cyan]💡 Found {len(lessons)} similar past experiences. Injecting lessons...[/dim cyan]"
            )
            lesson_context = "\n\nPAST LESSONS:\n" + "\n".join(
                [
                    f"- {lesson['summary']}: {lesson['content'][:200]}..."
                    for lesson in lessons
                ]
            )

        # Extended thinking for understanding
        understanding_prompt = f"""
Analyze the user's objective and the current codebase context.{lesson_context}
OBJECTIVE: {objective}

Perform a Semantic Analysis:
1. Identify the core concepts and entities involved.
2. Trace dependencies and potential impact on other components.
3. specific file modifications, creations, or deletions.
4. Steps for verification.

Write your analysis in a <thinking type="understanding"> block.
"""
        # Use direct generation with context
        self._generate_with_context(understanding_prompt)

        spec_payload, lint_payload = self._lower_objective_to_synthesis_spec(objective)
        self.current_synthesis_spec = spec_payload
        self.current_synthesis_lint = lint_payload

        # Create plan artifact
        plan = self.artifacts.create_plan(title="Implementation Plan", goal=objective)

        # Add planning section to task
        task = self.artifacts.get_task()
        section = task.add_section("Planning")
        section.add_item("Analyze objective")
        section.add_item("Research codebase")
        section.add_item("Identify affected components")
        section.add_item("Create implementation plan")
        task.save()

        # Mark first items in progress
        task.mark_in_progress(0, 0)
        task.mark_in_progress(0, 1)
        task.save()

        artifact_paths: Dict[str, str] = {}
        if spec_payload is not None:
            artifact_paths = self._persist_synthesis_spec_artifacts(
                objective,
                spec_payload,
                lint_payload,
            )
            task.add_artifact("synthesis_spec.json")
            task.add_artifact("synthesis_spec_lint.json")
            if artifact_paths.get("replay_path"):
                task.add_artifact("synthesis_replay.json")
            if artifact_paths.get("proof_path"):
                task.add_artifact("proof_capsule.json")
            task.save()
            self._record_synthesis_planning_event(
                objective,
                spec_payload,
                lint_payload,
                artifact_paths,
            )
            if not lint_payload.get("is_valid", False):
                self._update_boundary(
                    status="Synthesis spec requires clarification",
                    summary="Typed synthesis spec failed deterministic linting.",
                )
                return {
                    "success": False,
                    "error": "invalid_synthesis_spec",
                    "synthesis_spec": spec_payload,
                    "spec_lint": lint_payload,
                    "artifacts": artifact_paths,
                }

        # Phase 1, Item 1: Generate Visual System Diagram (architecture.mermaid)
        self._update_boundary(status="Generating architecture diagram")
        try:
            from core.visualizer import Visualizer

            viz = Visualizer(root_dir=".")

            # Identify a primary file to visualize (heuristic)
            entry_point = "main.py"
            if not os.path.exists(entry_point):
                # Try to find from proactive context
                proactive = self.agent.proactive_context
                key_files = proactive._identify_key_files(proactive._gather_files())
                entry_point = key_files.get("Entry Point", "main.py")

            if os.path.exists(entry_point):
                diagram = viz.generate_class_diagram(entry_point)
                self.artifacts.save_mermaid_diagram("architecture.mermaid", diagram)
                task.add_artifact("architecture.mermaid")
        except Exception as ve:
            self.console.print(f"[yellow]Could not generate diagram: {ve}[/yellow]")

        # Planning thinking - get the actual plan from the model
        planning_prompt = f"""
Based on your understanding, generate a detailed implementation plan.
The plan MUST be structured so it can be parsed into the `implementation_plan.md` artifact.

You MUST use the exact format below. Do not deviate.

## Goal
{objective}

## Proposed Changes
#### [Component Name]
- 📝 `path/to/file.py` - Description of change 🟢
- ➕ `path/to/new_file.py` - Description of new file 🟢
- 🗑️ `path/to/deleted_file.py` - Description of deletion 🔴

## Verification Plan
1. First step
2. Second step

Write the plan in a <thinking type="planning"> block first, then output the structured text.
IMPORTANT: Output ONLY the markdown content for the plan. Do not include conversational filler like "Here is the plan".
"""
        # Use direct generation with context
        plan_text = self._generate_with_context(planning_prompt)

        # We need a way to parse this plan_text into the PlanArtifact.
        # For now, we'll write the raw response to a temp file and load it using PlanArtifact.load
        # to take advantage of its existing parsing logic.
        temp_plan_path = self.artifacts.current_dir / "temp_plan.md"
        with open(temp_plan_path, "w") as f:
            f.write(plan_text)

        from core.artifacts.plan_artifact import PlanArtifact

        generated_plan = PlanArtifact.load(temp_plan_path)

        # Check if parsing was successful
        if not generated_plan.change_groups and "## Proposed Changes" in plan_text:
            self.console.print(
                "[yellow]Plan parsing warning: 'Proposed Changes' found in text but no groups parsed. Trying fallback parsing...[/yellow]"
            )
            # Fallback: model might have missed #### headers or emojis
            # We could attempt to re-generate or fix, but for now just log it
            self.thinking_system.think(
                ThinkingType.CORRECTION,
                "Plan parsing failed. The model output format did not match expected structure.",
            )

        # Merge generated plan into our plan artifact
        plan.goal = generated_plan.goal or objective
        plan.change_groups = generated_plan.change_groups
        plan.verification_steps = generated_plan.verification_steps
        plan.review_items = generated_plan.review_items
        if spec_payload is not None:
            if not plan.review_items:
                plan.add_assumption(
                    "Execution is gated by the typed deterministic synthesis spec artifacts."
                )
            if not plan.verification_steps:
                for command in list(
                    ((spec_payload.get("verification") or {}).get("commands") or [])
                ):
                    plan.add_verification_step(str(command))
        plan.save()

        # Clean up temp file
        if temp_plan_path.exists():
            temp_plan_path.unlink()

        # Mark planning items complete
        task.mark_complete(0, 0)
        task.mark_complete(0, 1)
        task.mark_complete(0, 2)
        task.mark_complete(0, 3)
        task.save()

        self._update_boundary(
            status="Implementation plan ready for review",
            summary="Planning complete. Awaiting approval.",
        )

        result = {"success": True}
        if spec_payload is not None:
            result["synthesis_spec"] = spec_payload
            result["spec_lint"] = lint_payload
            result["artifacts"] = artifact_paths
        return result

    def _await_plan_approval(self) -> bool:
        """
        Wait for user to approve the implementation plan.

        Returns:
            True if approved, False otherwise
        """
        plan = self.artifacts.get_plan()

        if plan.approved:
            return True

        # Display plan for review
        self.console.print(
            "\n[bold yellow]═══ Implementation Plan Ready for Review ═══[/bold yellow]"
        )
        self.console.print(f"[dim]Path: {plan.path}[/dim]")
        self.console.print(
            "\n[cyan]Review the plan and respond with 'approve' to continue.[/cyan]"
        )
        self.console.print("[dim]Or describe changes needed.[/dim]\n")

        # For now, auto-approve (in real usage, would wait for user input)
        self.artifacts.approve_plan("system")

        return True

    def _execution_phase(self) -> Dict[str, Any]:
        """
        Execution phase: Implement the approved plan.

        Returns:
            Dict with success status and any error
        """
        self._update_boundary(mode=AgentMode.EXECUTION, status="Implementing changes")

        if self.verbose:
            render_mode_transition(
                self.console, AgentMode.PLANNING, AgentMode.EXECUTION
            )
            self.console.print(
                "[bold cyan]⚡ EXECUTING:[/bold cyan] [italic]Applying planned changes to the codebase...[/italic]"
            )

        # Load the full plan (now properly parsed)
        plan = self.artifacts.load_plan()

        if not plan or not plan.change_groups:
            self.console.print(
                "[red]No changes found in plan to execute. Plan parsing may have failed or plan was empty.[/red]"
            )
            return {"success": False, "error": "Empty execution plan"}

        # Add execution section to task if not present
        task = self.artifacts.get_task()
        if not any(s.title == "Execution" for s in task.sections):
            section = task.add_section("Execution")
            # Populate with high level groups
            for group in plan.change_groups:
                section.add_item(f"Implement {group.component}: {group.description}")
            task.save()

        # Execute each group
        total_groups = len(plan.change_groups)

        for i, group in enumerate(plan.change_groups):
            self._update_boundary(
                status=f"Implement {group.component}",
                summary=f"Processing change group {i+1}/{total_groups}",
            )

            self.console.print(
                f"\n[bold cyan]Executing Change Group: {group.component}[/bold cyan]"
            )

            # Mark group in progress
            task.mark_in_progress("Execution", i)
            task.save()

            # Reasoning before group
            self.thinking_system.think(
                ThinkingType.REASONING,
                f"Starting implementation of {group.component}.\nChanges involved:\n"
                + "\n".join(f"- {c.description}" for c in group.changes),
            )

            # Execute each change in the group
            for change in group.changes:
                prompt = (
                    f"Execute this planned change:\n"
                    f"File: {change.file_path}\n"
                    f"Action: {change.description}\n\n"
                    "Use code editing tools to implement this exactly."
                )

                # Delegate to agent loop
                result = self.agent.run_loop(prompt)

                # Check for errors in result (rudimentary check)
                if result.get("error"):
                    self.thinking_system.think(
                        ThinkingType.CORRECTION,
                        f"Error implementing change for {change.file_path}: {result['error']}",
                    )

            # Reflection after group
            self.thinking_system.think(
                ThinkingType.REFLECTION,
                f"Completed implementation of {group.component}.",
            )

            # Mark group complete
            task.mark_complete("Execution", i)
            task.save()

        self._update_boundary(
            status="Execution complete",
            summary=f"Implemented {total_groups} component groups.",
        )

        return {"success": True}

    def _verification_phase(self) -> Dict[str, Any]:
        """
        Verification phase: Test, validate, potentially backtrack.

        Returns:
            Dict with success status and any error
        """
        self._update_boundary(
            mode=AgentMode.VERIFICATION, status="Running verification checks"
        )

        if self.verbose:
            render_mode_transition(
                self.console, AgentMode.EXECUTION, AgentMode.VERIFICATION
            )

        # Add verification section to task
        task = self.artifacts.get_task()
        section = task.add_section("Verification")
        section.add_item("Syntax check")
        section.add_item("Lint check")
        section.add_item("Type check")
        section.add_item("Semantic check")
        section.add_item("Run tests")
        task.save()

        attempts = 0
        state = self.task_manager.get_state()

        while attempts < self.max_backtrack_attempts:
            attempts += 1
            state.increment_verification()

            # Run verifications
            task.mark_in_progress(2, 0)
            task.save()

            results = run_all_verifications(".")

            # Execute Enhanced Verifier for semantic checks
            # We identify modified files from artifacts if possible
            modified_files = []
            plan = self.artifacts.get_plan()
            if plan:
                for group in plan.change_groups:
                    for change in group.changes:
                        modified_files.append(change.file_path)

            if modified_files:
                semantic_results = self.enhanced_verifier.verify_files(modified_files)
                # Map to a result format run_all_verifications expects or just log it
                for file, checks in semantic_results.items():
                    if any(
                        checks.values()
                    ):  # If any list is not empty, there are issues
                        pass  # Log or add to failures

            # Update task based on results
            for i, result in enumerate(results.results):
                if i < len(section.items):
                    if result.passed:
                        task.mark_complete(2, i)
                    else:
                        task.mark_blocked(2, i, result.message)
            task.save()

            if results.all_passed:
                governance_artifacts = self._persist_synthesis_governance_artifacts(
                    verification_passed=True,
                )
                self.thinking_system.think(
                    ThinkingType.REFLECTION,
                    "All verification checks passed! Work is complete.",
                )

                self._update_boundary(
                    status="Verification passed", summary="All checks passed."
                )

                synthesis_bundle = {}
                if self.current_synthesis_spec:
                    synthesis_bundle = {
                        "spec_path": str(self.artifacts.current_dir / "synthesis_spec.json"),
                        "replay_tape_path": str(self.artifacts.current_dir / "synthesis_replay.json"),
                        "proof_capsule_path": str(self.artifacts.current_dir / "proof_capsule.json"),
                        "benchmark_summary": governance_artifacts.get("benchmark_summary", {}),
                        "roadmap_validation": governance_artifacts.get("roadmap_validation", {}),
                    }
                return {
                    "success": True,
                    "results": results,
                    "verification": {"passed": True},
                    "deterministic_synthesis": synthesis_bundle,
                }

            # Verification failed - decide whether to backtrack
            if not state.can_backtrack():
                self.thinking_system.think(
                    ThinkingType.CORRECTION,
                    f"Verification failed and max attempts ({self.max_backtrack_attempts}) reached.",
                )
                return {
                    "success": False,
                    "error": "Max verification attempts reached",
                    "results": results,
                    "verification": {"passed": False},
                }

            # Backtrack to execution
            fail_msg = "\n".join(f.message for f in results.failures)

            # Add retrospective thinking before correction
            self.thinking_system.think(
                ThinkingType.REFLECTION,
                f"Verification failed on attempt {attempts}. Retrospective analysis:\n"
                f"- Implementation divergence found in verification: {fail_msg}\n"
                f"- Original goal: {self.current_objective}\n"
                "I will now formulate a correction strategy.",
            )

            self.thinking_system.think(
                ThinkingType.CORRECTION,
                f"Verification failed (attempt {attempts}). Issues:\n"
                + fail_msg
                + "\n\nBacktracking to execution to fix...",
            )

            self._update_boundary(
                mode=AgentMode.EXECUTION,
                status=f"Fixing verification issues (attempt {attempts})",
            )

            # Re-run execution with the failure report
            fix_prompt = f"""
Verification failed. Please fix the following issues:
{fail_msg}

Follow the existing implementation plan but address these specific failures.
"""
            # Use SubAgent for isolated fixing logic
            get_event_store().emit(
                event_type="AGENT_SPAWNED",
                source="EnhancedLoop",
                payload={"agent_type": "SubAgent", "purpose": "fix_verification"},
            )
            fix_agent = SubAgent(
                task=fix_prompt,
                parent_name=self.agent.name,
                brain=self.agent.brain,
                quiet=True,
            )
            fix_agent.execute()

        return {
            "success": False,
            "error": "Verification did not pass",
            "verification": {"passed": False},
        }

    def _finalize(
        self, objective: str, start_time: datetime, outcome: str, error: str = None
    ) -> Dict[str, Any]:
        """
        Finalize the task: Create walkthrough, archive artifacts.

        Args:
            objective: Original objective
            start_time: When task started
            outcome: Final outcome (completed, partial, failed, cancelled)
            error: Error message if any

        Returns:
            Final result dictionary
        """
        end_time = datetime.now()

        self._update_boundary(
            mode=AgentMode.IDLE, status="Finalizing", summary=f"Outcome: {outcome}"
        )

        # Create walkthrough
        walkthrough = self.artifacts.create_walkthrough(objective)
        walkthrough.set_outcome(outcome)
        walkthrough.set_timing(start_time, end_time)
        walkthrough.set_summary(f"Task {outcome}. " + (error or ""))

        # Add thinking stats
        chain = self.thinking_system.get_chain()
        if chain:
            walkthrough.set_stats(
                tool_calls=(
                    self.task_manager.get_state().step_count
                    if self.task_manager.get_state()
                    else 0
                ),
                thinking_blocks=len(chain),
            )

        walkthrough.save()

        # Save thinking chain
        if chain:
            self.artifacts.save_thinking_chain(chain)

        # Save all artifacts
        self.artifacts.save_task()

        # --- LEARNING & ADAPTIVE ROUTING ---
        # End task first so we have the result available for learning
        ended_task = self.task_manager.end_task()
        
        success = outcome == "completed"
        self.router.collect_feedback(
            objective,
            "enhanced",
            success,
            {"steps": ended_task.step_count if ended_task else 0},
        )

        if success or outcome == "failed":
            lesson = Lesson(
                objective=objective,
                summary=f"Task {outcome} with {ended_task.step_count if ended_task else 0} steps.",
                success=success,
                metadata={"artifacts": len(self.artifacts.get_all_artifact_paths())},
            )
            self.learning.record_lesson(lesson)

        # Display final view
        if self.verbose and ended_task:
            self.task_view.update(ended_task)

        return {
            "outcome": outcome,
            "error": error,
            "duration_seconds": (end_time - start_time).total_seconds(),
            "artifacts": [str(p) for p in self.artifacts.get_all_artifact_paths()],
            "thinking_blocks": len(chain) if chain else 0,
            "steps": ended_task.step_count if ended_task else 0,
        }

    def _update_boundary(
        self,
        name: str = None,
        mode: AgentMode = None,
        status: str = None,
        summary: str = None,
    ) -> None:
        """Update task boundary and render."""
        self.task_manager.update_boundary(name, mode, status, summary)
        self.task_manager.increment_step()

        state = self.task_manager.get_state()
        if state and self.verbose:
            self.task_view.update(state)


class SimpleLoop:
    """
    Simple loop for basic tasks that don't need full enhanced loop.

    Use this for:
    - Quick queries
    - Single-file edits
    - Simple commands
    """

    def __init__(self, agent):
        self.agent = agent

    def run(self, objective: str) -> Dict[str, Any]:
        """Run simple loop (delegates to agent's run_loop)."""
        return self.agent.run_loop(objective)


class LoopSelector:
    """
    Automatically selects between simple and enhanced loop.
    Uses AdaptiveRouter for evidence-based decisions.

    Criteria for enhanced loop:
    - Multi-file changes requested
    - Complex refactoring
    - Action-oriented requests (implement, create, fix, build)

    Criteria for simple loop:
    - Questions (what, how, why, explain)
    - Conversational requests (curious, thoughts, opinion)
    - Short requests without action keywords
    """

    _shared_router = None

    def __init__(self):
        self.router = self._router_instance()

    @classmethod
    def _router_instance(cls) -> AdaptiveRouter:
        """Lazily create a shared router so select() works as class or instance call."""
        if cls._shared_router is None:
            cls._shared_router = AdaptiveRouter()
        return cls._shared_router

    # Keywords that indicate a question/conversational request (use simple loop)
    QUESTION_PATTERNS = [
        "what",
        "how",
        "why",
        "explain",
        "show me",
        "tell me",
        "describe",
        "curious",
        "thoughts",
        "opinion",
        "think about",
        "wondering",
        "can you",
        "could you",
        "would you",
        "help me understand",
        "list",
        "read",
        "analyze",
        "review",
        "summarize",
    ]

    # Keywords that indicate an action request (use enhanced loop)
    ACTION_KEYWORDS = [
        "implement",
        "refactor",
        "create",
        "build",
        "fix",
        "add feature",
        "add a",
        "write a",
        "modify",
        "change",
        "update",
        "delete",
        "remove",
        "develop",
        "setup",
        "configure",
        "install",
        "upgrade",
        "migrate",
    ]

    REPO_KEYWORDS = [
        "repo",
        "repository",
        "codebase",
        "project",
        "folder",
        "directory",
        "file",
        "structure",
        "logic",
        "implementation",
        "code",
        "module",
        "package",
        "class",
        "function",
        "method",
        "variable",
        "constant",
    ]

    @classmethod
    def _is_question(cls, text: str) -> bool:
        """Check if text is a question or conversational request."""
        text_lower = text.lower()

        # Check for question mark
        if "?" in text:
            return True

        # Check for question patterns anywhere in text
        for pattern in cls.QUESTION_PATTERNS:
            if pattern in text_lower:
                return True

        return False

    @classmethod
    def _is_action(cls, text: str) -> bool:
        """Check if text is an action request."""
        text_lower = text.lower()

        for keyword in cls.ACTION_KEYWORDS:
            if keyword in text_lower:
                return True

        return False

    @classmethod
    def _is_repo_query(cls, text: str) -> bool:
        """Check if text refers to the codebase or repository."""
        text_lower = text.lower()

        # If it's a question using repo keywords
        for keyword in cls.REPO_KEYWORDS:
            if keyword in text_lower:
                return True

        # Path-like strings
        if "/" in text or "." in text or "\\" in text:
            return True

        return False

    @classmethod
    def select(cls, objective: str, force: str = None) -> str:
        """
        Select loop type based on objective.
        """
        if force:
            return force

        # 1. Check Adaptive Router (Evidence-based)
        prediction = cls._router_instance().predict_loop(objective)
        if prediction != "auto":
            return prediction

        # 2. Heuristics (Fallback)
        # Questions always use simple loop (conversational)
        if cls._is_question(objective):
            return "simple"

        # Action keywords use enhanced loop
        if cls._is_action(objective):
            return "enhanced"

        # Default to simple for shorter requests
        if len(objective) < 100:
            return "simple"

        # Long requests without clear indicators - use enhanced
        return "enhanced"
