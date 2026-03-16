from core.orchestrator.graph import TaskGraph, TaskUnit
from domains.code_intelligence.semantic_engine import SemanticEngine
from shared_kernel.event_store import get_event_store
from core.agents.planner import PlannerAgent
from core.agents.worker import WorkerAgent
from core.agents.verifier import VerifierAgent
from core.ollama_client import DeterministicOllama
from config.settings import MASTER_MODEL, AGENTIC_THINKING
from typing import List, Dict, Any, Optional


class TaskQueue:
    """A prioritized queue of TaskUnits."""

    def __init__(self, graph: TaskGraph):
        self.graph = graph

    def next_ready(self) -> Optional[TaskUnit]:
        ready = self.graph.get_ready_tasks()
        if not ready:
            return None
        # Simple FIFO for now within ready tasks
        return ready[0]


class AgentOrchestrator:
    """
    State machine that manages the lifecycle of an agentic objective.
    """

    def __init__(
        self,
        root_dir: str = ".",
        brain: Optional[DeterministicOllama] = None,
        semantic_engine: Optional[SemanticEngine] = None,
        env_info: Optional[Dict[str, str]] = None,
        console: Optional[Any] = None,
    ):
        self.root_dir = root_dir
        self.console = console

        if brain is not None:
            self.brain = brain
        else:
            self.brain = DeterministicOllama(MASTER_MODEL)

        self.env_info = env_info or {}

        if semantic_engine is not None:
            self.semantic_engine = semantic_engine
        else:
            self.semantic_engine = SemanticEngine(root_dir=root_dir, brain=self.brain)

        # Initialize RAG index if not already done
        if not getattr(self.semantic_engine, "_indexed", False):
            self.semantic_engine.analyze_workspace()

        self.graph = TaskGraph()
        self.queue = TaskQueue(self.graph)
        self.global_context: Dict[str, Any] = {}
        self.status = "IDLE"
        self.results: List[str] = []
        self.max_retries = 3
        self.total_budget = AGENTIC_THINKING.get("thinking_budget", 50000)
        self.spent_budget = 0

    def plan(self, objective: str):
        """Phase 1: Inception & Semantic Analysis"""
        self.status = "PLANNING"
        get_event_store().emit(
            event_type="AGENT_SPAWNED",
            source="Orchestrator",
            payload={"agent_type": "planner"},
        )
        planner = PlannerAgent(
            brain=self.brain, env_info=self.env_info, console=self.console
        )
        tasks = planner.plan(objective)

        for task in tasks:
            self.graph.add_task(task)

        get_event_store().emit(
            event_type="PLAN_APPROVED",
            source="Orchestrator",
            payload={"task_count": len(tasks)},
        )
        self.status = "READY"

    def run(self, objective: str):
        """Sequential Execution Loop with Self-Correction"""
        if self.status == "IDLE":
            # Context Hydration via RAG
            print("[Orchestrator] Starting Context Hydration (Semantic Analysis)...")
            relevant_files = self.semantic_engine.get_context_for_objective(objective)
            self.global_context["relevant_files"] = relevant_files
            print(
                f"[Orchestrator] Context hydrated. Found {len(relevant_files)} relevant files."
            )

            print("[Orchestrator] Entering Planning Phase...")
            self.plan(objective)
            print(
                f"[Orchestrator] Planning complete. Task Graph initialized with {len(self.graph.tasks)} tasks."
            )

        print("[Orchestrator] Sequential Execution loop started...")
        self.status = "EXECUTING"
        retry_count = 0

        while retry_count < self.max_retries:
            while True:
                task = self.queue.next_ready()
                if not task:
                    if self.graph.is_complete():
                        break
                    else:
                        self.status = "BLOCKED"
                        break

                try:
                    summary = self._execute_task(task)
                    self.results.append(summary)
                    task.status = "COMPLETED"
                    task.result = {"summary": summary}
                except Exception as e:
                    task.status = "FAILED"
                    task.result = {"error": str(e)}
                    break

            # Phase 3: Synthesis / Verification
            if self.status == "EXECUTING" and self.graph.is_complete():
                self.status = "VERIFYING"
                get_event_store().emit(
                    event_type="AGENT_SPAWNED",
                    source="Orchestrator",
                    payload={"agent_type": "verifier"},
                )
                verifier = VerifierAgent(
                    brain=self.brain, env_info=self.env_info, console=self.console
                )
                verified = verifier.verify(objective, self.results)

                get_event_store().emit(
                    event_type="VERIFICATION_COMPLETED",
                    source="Orchestrator",
                    payload={"success": verified},
                )

                if verified:
                    self.status = "COMPLETED"
                    summary = "\n".join([f"- {res}" for res in self.results])
                    return f"Objective fulfilled successfully.\n\nExecution Summary:\n{summary}"
                else:
                    # Self-Correction Step
                    retry_count += 1
                    if retry_count < self.max_retries:
                        self._trigger_self_correction(objective)
                        continue
                    else:
                        self.status = "FAILED"
                        get_event_store().emit(
                            event_type="ERROR_OCCURRED",
                            source="Orchestrator",
                            payload={"error": "Max retries reached in verification"},
                        )
                        return f"Objective failed after {self.max_retries} attempts. See logs for details."
            else:
                break

        return "Objective failed or was blocked. Status: " + self.status

    def _trigger_self_correction(self, objective: str):
        """Re-plans based on verification failure."""
        self.status = "CORRECTING"
        # Logic to analyze current results and objective to find gaps
        # For now, we simple re-plan with the current results as context
        correction_objective = f"Review the previous attempts and fix the issues for: {objective}. Previous results: {self.results}"
        self.plan(correction_objective)  # Appends new tasks to the graph
        self.status = "EXECUTING"

    def _execute_task(self, task: TaskUnit) -> str:
        task.status = "IN_PROGRESS"
        get_event_store().emit(
            event_type="TASK_ASSIGNED",
            source="Orchestrator",
            payload={"task_id": task.id, "instruction": task.instruction},
        )

        # Calculate sub-task budget
        remaining_tasks = (
            len([t for t in self.graph.tasks.values() if t.status == "PENDING"]) + 1
        )
        (self.total_budget - self.spent_budget) // remaining_tasks

        get_event_store().emit(
            event_type="AGENT_SPAWNED",
            source="Orchestrator",
            payload={"agent_type": "worker", "task_id": task.id},
        )
        worker = WorkerAgent(
            task, brain=self.brain, env_info=self.env_info, console=self.console
        )
        # Apply budget (simulated via generation params for now)
        # In a real system, we'd pass this to the model's 'num_predict' or similar

        summary = worker.execute()

        # Estimate spent tokens (simplified)
        self.spent_budget += len(summary)  # Placeholder for actual token counting

        return summary
