import os
import json
import time
from typing import Dict, Any, List, Type, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich import box
from rich.tree import Tree

# Subagent Imports
from core.agents.subagent import SubAgent
from core.agents.repo_analyzer import RepoAnalysisSubagent
from core.agents.researcher import ResearchSubagent
from core.agents.debugger import DebugSubagent
from core.agents.implementor import ImplementationSubagent
from core.agents.planner_agent import PlanningSubagent
from core.agents.tester import TestingSubagent

# Phase 4.2: Subagent Communication
from core.subagent_communication import (
    MessageBus,
    MessageType,
    Priority,
    CoordinationProtocol,
)


class ExecutionMode(Enum):
    """Workflow execution modes."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"


@dataclass
class SubagentStep:
    agent: Type[SubAgent]
    inputs: Dict[str, str]  # Template strings like "{var}"
    outputs: List[str]
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    condition: Optional[Callable[[Dict], bool]] = None  # For conditional execution
    metadata: Dict[str, Any] = field(default_factory=dict)


class SubagentWorkflow:
    """
    Composable subagent workflow with advanced orchestration.

    Phase 3.3 Enhancements:
    - Parallel execution support
    - Conditional branching
    - Workflow visualization
    - Performance tracking

    Phase 4.2 Enhancements:
    - Message bus for inter-agent communication
    - Coordination protocols (handoff, barrier sync, request-response)
    - Shared context management
    """

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.steps: List[SubagentStep] = []
        self.performance_metrics: Dict[str, Any] = {}
        self.execution_history: List[Dict[str, Any]] = []

        # Phase 4.2: Communication infrastructure
        self.message_bus: Optional[MessageBus] = None
        self.coordination: Optional[CoordinationProtocol] = None

    def add_step(
        self,
        agent: Type[SubAgent],
        inputs: Dict,
        outputs: List[str],
        execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL,
        condition: Optional[Callable[[Dict], bool]] = None,
        metadata: Optional[Dict] = None,
    ):
        """
        Add subagent step to workflow.

        Phase 3.3: Enhanced with execution modes and conditions.

        Args:
            agent: Subagent class to execute
            inputs: Input parameter mapping (can use {var} templates)
            outputs: Output variable names to store in context
            execution_mode: Sequential, parallel, or conditional
            condition: Optional condition function for conditional execution
            metadata: Optional metadata for this step
        """
        self.steps.append(
            SubagentStep(
                agent=agent,
                inputs=inputs,
                outputs=outputs,
                execution_mode=execution_mode,
                condition=condition,
                metadata=metadata or {},
            )
        )

    def add_parallel_steps(self, steps: List[tuple]):
        """
        Add multiple steps to run in parallel.

        Args:
            steps: List of tuples (agent, inputs, outputs)
        """
        for agent, inputs, outputs in steps:
            self.add_step(
                agent=agent,
                inputs=inputs,
                outputs=outputs,
                execution_mode=ExecutionMode.PARALLEL,
            )

    def visualize(self, console: Optional[Console] = None) -> str:
        """
        Generate visual representation of workflow.
        Phase 3.3: Workflow visualization using Rich Tree.
        """
        if console is None:
            console = Console()

        tree = Tree(f"[bold magenta]🔄 Workflow: {self.name}[/bold magenta]")

        if self.description:
            tree.add(f"[dim]{self.description}[/dim]")

        current_parallel_group = []

        for i, step in enumerate(self.steps, 1):
            step_label = f"[cyan]{i}. {step.agent.__name__}[/cyan]"

            if step.execution_mode == ExecutionMode.PARALLEL:
                current_parallel_group.append((i, step))
            else:
                # Flush any pending parallel group
                if current_parallel_group:
                    parallel_branch = tree.add("[yellow]⚡ Parallel Execution[/yellow]")
                    for p_idx, p_step in current_parallel_group:
                        p_label = f"[cyan]{p_idx}. {p_step.agent.__name__}[/cyan]"
                        p_node = parallel_branch.add(p_label)
                        p_node.add(
                            f"[dim]Inputs: {', '.join(p_step.inputs.keys())}[/dim]"
                        )
                        p_node.add(f"[dim]Outputs: {', '.join(p_step.outputs)}[/dim]")
                    current_parallel_group = []

                # Add sequential step
                step_node = tree.add(step_label)
                step_node.add(f"[dim]Inputs: {', '.join(step.inputs.keys())}[/dim]")
                step_node.add(f"[dim]Outputs: {', '.join(step.outputs)}[/dim]")

                if step.condition:
                    step_node.add("[yellow]⚠ Conditional execution[/yellow]")

        # Flush remaining parallel group
        if current_parallel_group:
            parallel_branch = tree.add("[yellow]⚡ Parallel Execution[/yellow]")
            for p_idx, p_step in current_parallel_group:
                p_label = f"[cyan]{p_idx}. {p_step.agent.__name__}[/cyan]"
                p_node = parallel_branch.add(p_label)
                p_node.add(f"[dim]Inputs: {', '.join(p_step.inputs.keys())}[/dim]")
                p_node.add(f"[dim]Outputs: {', '.join(p_step.outputs)}[/dim]")

        console.print(tree)
        return str(tree)

    def execute(self, context: Dict, runtime_context: Any) -> Dict:
        """
        Execute workflow with data flow between subagents.

        Phase 3.3 Enhancements:
        - Parallel execution support
        - Conditional branching
        - Performance tracking
        - Detailed execution history

        Phase 4.2 Enhancements:
        - Message bus initialization and agent registration
        - Coordination protocols for handoffs
        - Shared context across agents

        Args:
            context: Initial context variables
            runtime_context: AgentREPL instance with brain and console

        Returns:
            Final context after all steps
        """
        console = runtime_context.console
        start_time = time.time()

        console.print(
            Panel(
                (
                    f"[bold magenta]🔄 Executing Workflow: {self.name}[/bold magenta]\n"
                    f"[dim]{self.description}[/dim]"
                    if self.description
                    else ""
                ),
                style="bold magenta",
            )
        )

        # Phase 4.2: Initialize message bus for inter-agent communication
        self.message_bus = MessageBus(console)
        self.coordination = CoordinationProtocol(self.message_bus, console)

        # Register all agents with message bus
        for i, step in enumerate(self.steps):
            agent_id = f"{step.agent.__name__}_{i}"
            self.message_bus.register_agent(
                agent_id=agent_id,
                subscriptions=[f"workflow.{self.name}", "global"],
                metadata={
                    "step_index": i,
                    "inputs": step.inputs,
                    "outputs": step.outputs,
                },
            )

        # Store workflow context in shared context
        self.message_bus.set_shared_context("workflow_name", self.name)
        self.message_bus.set_shared_context("initial_context", context)

        # Visualize workflow before execution
        self.visualize(console)

        # Group steps by execution mode
        step_groups = self._group_steps_by_execution()

        for group_idx, (mode, steps) in enumerate(step_groups):
            if mode == ExecutionMode.PARALLEL:
                console.print(
                    f"\n[bold yellow]⚡ Parallel Group {group_idx + 1} ({len(steps)} agents)[/bold yellow]"
                )
                self._execute_parallel_steps(steps, context, runtime_context)
            else:
                # Sequential execution
                for step in steps:
                    if not self._execute_step(step, context, runtime_context):
                        # Step failed and user chose not to continue
                        break

        # Calculate performance metrics
        total_time = time.time() - start_time
        self.performance_metrics = {
            "workflow_name": self.name,
            "total_steps": len(self.steps),
            "execution_time_sec": total_time,
            "steps_completed": len(self.execution_history),
            "success_rate": self._calculate_success_rate(),
        }

        console.print(
            Panel(
                f"[bold green]✅ Workflow Execution Complete[/bold green]\n\n"
                f"[cyan]Steps Completed:[/cyan] {len(self.execution_history)}/{len(self.steps)}\n"
                f"[cyan]Total Time:[/cyan] {total_time:.2f}s\n"
                f"[cyan]Success Rate:[/cyan] {self.performance_metrics['success_rate']:.1f}%",
                style="bold green",
            )
        )

        # Phase 4.2: Print message bus statistics
        if self.message_bus:
            console.print("\n[bold]Communication Statistics:[/bold]")
            self.message_bus.print_stats()

            # Optionally visualize message flow
            if Confirm.ask("Show message flow visualization?", default=False):
                self.message_bus.visualize_message_flow()

            # Export message log
            log_path = f".anvil/workflows/{self.name}_messages.json"
            self.message_bus.export_message_log(log_path)

            # Cleanup: unregister all agents
            for i, step in enumerate(self.steps):
                agent_id = f"{step.agent.__name__}_{i}"
                self.message_bus.unregister_agent(agent_id)

        return context

    def _group_steps_by_execution(self) -> List[tuple]:
        """Group consecutive steps by execution mode."""
        groups = []
        current_group = []
        current_mode = None

        for step in self.steps:
            if step.execution_mode == ExecutionMode.PARALLEL:
                if current_mode == ExecutionMode.PARALLEL:
                    current_group.append(step)
                else:
                    # Flush previous group
                    if current_group:
                        groups.append((current_mode, current_group))
                    current_group = [step]
                    current_mode = ExecutionMode.PARALLEL
            else:
                # Flush parallel group if switching to sequential
                if current_mode == ExecutionMode.PARALLEL:
                    groups.append((current_mode, current_group))
                    current_group = []

                # Sequential steps executed individually
                groups.append((ExecutionMode.SEQUENTIAL, [step]))
                current_mode = ExecutionMode.SEQUENTIAL
                current_group = []

        # Flush remaining group
        if current_group:
            groups.append((current_mode, current_group))

        return groups

    def _execute_step(
        self, step: SubagentStep, context: Dict, runtime_context: Any
    ) -> bool:
        """
        Execute a single subagent step.

        Phase 4.2: Enhanced with message bus integration for inter-agent communication.

        Returns:
            True if execution should continue, False to stop workflow
        """
        console = runtime_context.console
        step_idx = self.steps.index(step) + 1
        agent_id = f"{step.agent.__name__}_{step_idx - 1}"

        console.print(
            f"\n[bold cyan]▶ Step {step_idx}/{len(self.steps)}: {step.agent.__name__}[/bold cyan]"
        )

        # Phase 4.2: Send status update via message bus
        if self.message_bus:
            self.message_bus.update_status(agent_id, "starting")
            self.message_bus.publish(
                topic=f"workflow.{self.name}",
                sender=agent_id,
                payload={"event": "step_started", "step": step_idx},
            )

        # Check condition
        if step.condition and not step.condition(context):
            console.print("[yellow]⊘ Step skipped (condition not met)[/yellow]")
            if self.message_bus:
                self.message_bus.update_status(agent_id, "skipped")
            return True

        # Resolve inputs
        step_inputs = self._resolve_inputs(step.inputs, context, console)

        # Display Inputs
        console.print(f"[dim]Inputs: {json.dumps(step_inputs, indent=2)}[/dim]")

        # Instantiate Agent
        task_desc = step.metadata.get(
            "description", f"Running step {step_idx} of {self.name}"
        )
        agent_instance = step.agent(
            task=task_desc,
            parent_name="Workflow",
            brain=runtime_context.brain,
            console=runtime_context.console,
        )

        # Phase 4.2: Update agent status to busy
        if self.message_bus:
            self.message_bus.update_status(agent_id, "busy")

        step_start = time.time()

        try:
            # Execute Agent
            result = agent_instance.run(**step_inputs)

            # Handle Outputs
            response_text = (
                result.get("response", "") if isinstance(result, dict) else str(result)
            )

            # Map outputs to context
            if len(step.outputs) == 1:
                context[step.outputs[0]] = response_text
            else:
                for out_key in step.outputs:
                    context[out_key] = response_text

            step_time = time.time() - step_start

            # Record execution
            self.execution_history.append(
                {
                    "step_index": step_idx,
                    "agent": step.agent.__name__,
                    "success": True,
                    "execution_time": step_time,
                    "outputs": step.outputs,
                }
            )

            # Phase 4.2: Send success message and handoff to next agent
            if self.message_bus:
                self.message_bus.update_status(agent_id, "completed")
                self.message_bus.send(
                    sender=agent_id,
                    recipient=None,  # Broadcast
                    message_type=MessageType.RESULT,
                    payload={"outputs": step.outputs, "result": response_text},
                    priority=Priority.NORMAL,
                )

                # Handoff to next agent if not last step
                if step_idx < len(self.steps):
                    next_agent_id = f"{self.steps[step_idx].agent.__name__}_{step_idx}"
                    self.coordination.handoff(
                        from_agent=agent_id,
                        to_agent=next_agent_id,
                        payload={"context": context, "previous_result": response_text},
                    )

            console.print(f"[green]✔ Step complete ({step_time:.2f}s)[/green]")
            return True

        except Exception as e:
            step_time = time.time() - step_start

            # Record failure
            self.execution_history.append(
                {
                    "step_index": step_idx,
                    "agent": step.agent.__name__,
                    "success": False,
                    "execution_time": step_time,
                    "error": str(e),
                }
            )

            console.print(f"[red]✗ Step failed: {e}[/red]")

            # Phase 4.2: Send error message via message bus
            if self.message_bus:
                self.message_bus.update_status(agent_id, "failed")
                self.message_bus.send(
                    sender=agent_id,
                    recipient=None,  # Broadcast
                    message_type=MessageType.ERROR,
                    payload={"error": str(e), "step": step_idx},
                    priority=Priority.HIGH,
                )

            # Prompt to continue
            if not Confirm.ask("Continue to next step despite failure?", default=False):
                return False

            return True

    def _execute_parallel_steps(
        self, steps: List[SubagentStep], context: Dict, runtime_context: Any
    ):
        """
        Execute multiple steps in parallel (conceptual).

        Phase 4.2: Enhanced with barrier synchronization and result aggregation via message bus.

        Note: True parallel execution would require threading/multiprocessing.
        For now, this executes sequentially but treats outputs as independent.
        """
        console = runtime_context.console

        parallel_results = []
        barrier_id = f"parallel_group_{int(time.time())}"

        # Phase 4.2: Publish parallel execution start event
        if self.message_bus:
            self.message_bus.publish(
                topic=f"workflow.{self.name}",
                sender="coordinator",
                payload={
                    "event": "parallel_start",
                    "num_agents": len(steps),
                    "barrier_id": barrier_id,
                },
            )

        for step in steps:
            step_idx = self.steps.index(step) + 1
            agent_id = f"{step.agent.__name__}_{step_idx - 1}"
            console.print(
                f"\n[cyan]  ⚡ Parallel Branch {step_idx}: {step.agent.__name__}[/cyan]"
            )

            # Phase 4.2: Update agent status
            if self.message_bus:
                self.message_bus.update_status(agent_id, "busy")

            # Resolve inputs
            step_inputs = self._resolve_inputs(step.inputs, context, console)

            # Instantiate Agent
            task_desc = step.metadata.get("description", f"Parallel step {step_idx}")
            agent_instance = step.agent(
                task=task_desc,
                parent_name="Workflow-Parallel",
                brain=runtime_context.brain,
                console=runtime_context.console,
            )

            step_start = time.time()

            try:
                # Execute
                result = agent_instance.run(**step_inputs)
                response_text = (
                    result.get("response", "")
                    if isinstance(result, dict)
                    else str(result)
                )

                step_time = time.time() - step_start

                parallel_results.append(
                    {
                        "step": step,
                        "success": True,
                        "result": response_text,
                        "execution_time": step_time,
                    }
                )

                # Record execution
                self.execution_history.append(
                    {
                        "step_index": step_idx,
                        "agent": step.agent.__name__,
                        "success": True,
                        "execution_time": step_time,
                        "parallel": True,
                    }
                )

                # Phase 4.2: Publish result and reach barrier
                if self.message_bus:
                    self.message_bus.update_status(agent_id, "completed")
                    self.message_bus.publish(
                        topic=f"{barrier_id}.result",
                        sender=agent_id,
                        payload={"result": response_text, "outputs": step.outputs},
                        priority=Priority.HIGH,
                    )

                    # Barrier synchronization
                    self.coordination.barrier_sync(agent_id, barrier_id, len(steps))

                console.print(f"[green]  ✔ Branch complete ({step_time:.2f}s)[/green]")

            except Exception as e:
                step_time = time.time() - step_start

                parallel_results.append(
                    {
                        "step": step,
                        "success": False,
                        "error": str(e),
                        "execution_time": step_time,
                    }
                )

                self.execution_history.append(
                    {
                        "step_index": step_idx,
                        "agent": step.agent.__name__,
                        "success": False,
                        "execution_time": step_time,
                        "parallel": True,
                        "error": str(e),
                    }
                )

                # Phase 4.2: Send error message
                if self.message_bus:
                    self.message_bus.update_status(agent_id, "failed")
                    self.message_bus.send(
                        sender=agent_id,
                        recipient=None,
                        message_type=MessageType.ERROR,
                        payload={"error": str(e), "step": step_idx},
                        priority=Priority.HIGH,
                    )

                console.print(f"[red]  ✗ Branch failed: {e}[/red]")

        # Merge parallel results into context
        for res in parallel_results:
            if res["success"]:
                for out_key in res["step"].outputs:
                    context[out_key] = res["result"]

        # Phase 4.2: Aggregate results using coordination protocol
        if self.message_bus and self.coordination:
            worker_agents = [
                f"{step.agent.__name__}_{self.steps.index(step)}" for step in steps
            ]
            aggregated = self.coordination.aggregate_results(
                coordinator="workflow_coordinator",
                worker_agents=worker_agents,
                task_topic=barrier_id,
                timeout=5.0,
            )

            console.print(
                f"[dim]Aggregated {len(aggregated)} results from message bus[/dim]"
            )

        console.print("[green]⚡ Parallel group complete[/green]")

    def _resolve_inputs(
        self, inputs: Dict[str, str], context: Dict, console: Console
    ) -> Dict[str, Any]:
        """Resolve input templates against context."""
        step_inputs = {}

        for key, val_template in inputs.items():
            try:
                step_inputs[key] = val_template.format(**context)
            except KeyError:
                if "{" in val_template and "}" in val_template:
                    console.print(
                        f"[yellow]⚠ Context missing for '{val_template}'. Using raw value.[/yellow]"
                    )
                    step_inputs[key] = val_template
                else:
                    step_inputs[key] = val_template
            except Exception:
                step_inputs[key] = val_template

        return step_inputs

    def _calculate_success_rate(self) -> float:
        """Calculate success rate of executed steps."""
        if not self.execution_history:
            return 0.0

        successful = sum(
            1 for entry in self.execution_history if entry.get("success", False)
        )
        return (successful / len(self.execution_history)) * 100

    def export_metrics(self, output_path: str = ".anvil/workflows/metrics.json"):
        """Export workflow performance metrics to JSON."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        metrics_data = {
            "workflow": self.name,
            "description": self.description,
            "performance": self.performance_metrics,
            "execution_history": self.execution_history,
            "timestamp": time.time(),
        }

        with open(output_path, "w") as f:
            json.dump(metrics_data, f, indent=2)

        return output_path


# Predefined Workflows
class FeatureDevelopmentWorkflow(SubagentWorkflow):
    """
    End-to-end feature development workflow.
    Phase 3.3: Enhanced with parallel research and proper task flow.
    """

    def __init__(self):
        super().__init__(
            "feature-development",
            "Research, Plan, Implement, Test - complete feature lifecycle",
        )

        # Parallel research phase
        self.add_parallel_steps(
            [
                (
                    ResearchSubagent,
                    {"research_query": "{feature_description}"},
                    ["research_findings"],
                ),
                (RepoAnalysisSubagent, {"root_dir": "."}, ["architecture_analysis"]),
            ]
        )

        # Sequential implementation
        self.add_step(
            PlanningSubagent,
            {"objective": "{feature_description}"},
            ["implementation_plan"],
            metadata={"description": "Create comprehensive implementation plan"},
        )

        self.add_step(
            ImplementationSubagent,
            {"spec": "{implementation_plan}"},
            ["code_changes"],
            metadata={"description": "Implement features following plan"},
        )

        self.add_step(
            TestingSubagent,
            {"target": "{code_changes}"},
            ["test_suite"],
            metadata={"description": "Generate comprehensive test suite"},
        )


class BugFixWorkflow(SubagentWorkflow):
    """
    Systematic bug investigation and fix workflow.
    Phase 3.3: New predefined workflow.
    """

    def __init__(self):
        super().__init__(
            "bug-fix", "Debug, Analyze, Fix, Verify - systematic bug resolution"
        )

        self.add_step(
            DebugSubagent,
            {"error_log": "{bug_description}"},
            ["root_cause_analysis"],
            metadata={"description": "Identify root cause of bug"},
        )

        self.add_step(
            ImplementationSubagent,
            {"spec": "{root_cause_analysis}"},
            ["bug_fix"],
            metadata={"description": "Implement minimal fix"},
        )

        self.add_step(
            TestingSubagent,
            {"target": "{bug_fix}"},
            ["regression_tests"],
            metadata={"description": "Verify fix and prevent regression"},
        )


class CodeReviewWorkflow(SubagentWorkflow):
    """
    Comprehensive code review and refactoring workflow.
    Phase 3.3: New predefined workflow.
    """

    def __init__(self):
        super().__init__(
            "code-review", "Analyze architecture, identify issues, suggest improvements"
        )

        # Parallel analysis
        self.add_parallel_steps(
            [
                (RepoAnalysisSubagent, {"root_dir": "."}, ["architecture_review"]),
                (TestingSubagent, {"target": "."}, ["coverage_analysis"]),
            ]
        )

        self.add_step(
            ResearchSubagent,
            {"research_query": "best practices for {tech_stack}"},
            ["best_practices"],
            metadata={"description": "Research industry best practices"},
        )

        self.add_step(
            PlanningSubagent,
            {
                "objective": "Refactor based on {architecture_review} and {best_practices}"
            },
            ["refactor_plan"],
            metadata={"description": "Create refactoring roadmap"},
        )


class WorkflowWizard:
    """
    Elite workflow composition and orchestration system.

    Phase 3.3 Enhancements:
    - Multi-agent parallel workflows
    - Interactive visual composer
    - Predefined enterprise workflows
    - Integration with LoopBuilder
    - Workflow registry and persistence
    """

    TEMPLATES = {
        "1": {
            "name": "Feature Development",
            "description": "End-to-end feature lifecycle: Research → Plan → Implement → Test",
            "workflow_class": FeatureDevelopmentWorkflow,
        },
        "2": {
            "name": "Bug Fix Pipeline",
            "description": "Systematic bug resolution: Debug → Analyze → Fix → Verify",
            "workflow_class": BugFixWorkflow,
        },
        "3": {
            "name": "Code Review & Refactor",
            "description": "Architecture review with best practices analysis",
            "workflow_class": CodeReviewWorkflow,
        },
        "4": {
            "name": "Custom Loop Builder",
            "description": "Generate specialized, task-optimized execution loop (AI-powered)",
            "workflow_class": None,
        },
        "5": {
            "name": "Visual Workflow Composer",
            "description": "Build custom multi-agent workflow with interactive UI",
            "workflow_class": None,
        },
    }

    def __init__(self, agent):
        self.agent = agent
        self.console = agent.console
        from core.loops.loop_builder import LoopBuilder

        self.loop_builder = LoopBuilder(agent)
        self.workflow_registry: Dict[str, SubagentWorkflow] = {}

    def run(self):
        """
        Main entry point for Workflow Wizard.
        Phase 3.3: Enhanced with visual composer and better UX.
        """
        self.console.print(
            Panel(
                "[bold magenta]🧙 Anvil Workflow Wizard v4.0[/bold magenta]\n"
                "[dim]Multi-Agent Workflow Orchestration & Loop Composition[/dim]",
                box=box.DOUBLE,
            )
        )

        # Display templates in a table
        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("#", style="cyan", width=3)
        table.add_column("Workflow", style="green")
        table.add_column("Description", style="dim")

        for key, template in self.TEMPLATES.items():
            table.add_row(key, template["name"], template["description"])

        self.console.print("\n")
        self.console.print(table)

        choice = Prompt.ask(
            "\n[bold]Select workflow[/bold]",
            choices=list(self.TEMPLATES.keys()),
            default="1",
        )

        template = self.TEMPLATES[choice]

        if choice == "1":
            # Feature Development
            desc = Prompt.ask("\n[bold]Enter feature description[/bold]")
            wf = FeatureDevelopmentWorkflow()
            return wf.execute({"feature_description": desc}, self.agent)

        elif choice == "2":
            # Bug Fix
            bug_desc = Prompt.ask("\n[bold]Describe the bug or error[/bold]")
            wf = BugFixWorkflow()
            return wf.execute({"bug_description": bug_desc}, self.agent)

        elif choice == "3":
            # Code Review
            tech_stack = Prompt.ask(
                "\n[bold]What is your tech stack?[/bold]", default="Python"
            )
            wf = CodeReviewWorkflow()
            return wf.execute({"tech_stack": tech_stack}, self.agent)

        elif choice == "4":
            # Custom Loop Builder
            return self.run_loop_builder()

        elif choice == "5":
            # Visual Workflow Composer
            return self.run_visual_composer()

    def run_loop_builder(self):
        """
        Run interactive loop builder.
        Phase 3.3: Integration with enhanced LoopBuilder.
        """
        self.console.print(
            "\n[bold magenta]🎨 Interactive Loop Builder[/bold magenta]\n"
        )

        use_interactive = Confirm.ask("Use interactive UI?", default=True)

        if use_interactive:
            path = self.loop_builder.interactive_builder()
        else:
            objective = Prompt.ask("Describe the specific mission for this loop")
            path = self.loop_builder.generate_loop(objective)

        if not path:
            return "Loop builder cancelled"

        # Ask if user wants to run immediately
        if Confirm.ask("Run the generated loop now?", default=True):
            objective = Prompt.ask("Enter initial input for the loop")
            loop = self.loop_builder.instantiate_loop(path, objective)
            return loop.run(objective)

        return f"Loop generated at {path}"

    def run_visual_composer(self):
        """
        Interactive visual workflow composer.
        Phase 3.3: New visual workflow design UI.
        """
        self.console.print(
            "\n[bold magenta]🎨 Visual Workflow Composer[/bold magenta]\n"
        )

        workflow_name = Prompt.ask("Workflow name", default="custom-workflow")
        workflow_desc = Prompt.ask("Workflow description", default="")

        wf = SubagentWorkflow(workflow_name, workflow_desc)

        # Available subagents
        available_agents = {
            "1": ("Research", ResearchSubagent, ["research_query"]),
            "2": ("Repository Analysis", RepoAnalysisSubagent, ["root_dir"]),
            "3": ("Debug", DebugSubagent, ["error_log"]),
            "4": ("Implementation", ImplementationSubagent, ["spec"]),
            "5": ("Planning", PlanningSubagent, ["objective"]),
            "6": ("Testing", TestingSubagent, ["target"]),
        }

        self.console.print("\n[bold]Available Subagents:[/bold]")
        agent_table = Table(show_header=True, header_style="bold cyan", box=box.SIMPLE)
        agent_table.add_column("#", style="cyan")
        agent_table.add_column("Agent", style="green")
        agent_table.add_column("Required Inputs", style="dim")

        for key, (name, _, inputs) in available_agents.items():
            agent_table.add_row(key, name, ", ".join(inputs))

        self.console.print(agent_table)

        # Build workflow step by step
        step_count = 0
        while True:
            step_count += 1
            self.console.print(f"\n[bold]Step {step_count}:[/bold]")

            agent_choice = Prompt.ask(
                "Select agent (or 'done' to finish)",
                choices=list(available_agents.keys()) + ["done"],
                default="done" if step_count > 1 else "1",
            )

            if agent_choice == "done":
                break

            agent_name, agent_class, required_inputs = available_agents[agent_choice]

            # Get inputs
            inputs = {}
            for input_param in required_inputs:
                input_value = Prompt.ask(
                    f"  Input for '{input_param}'", default=f"{{{input_param}}}"
                )
                inputs[input_param] = input_value

            # Get outputs
            output_var = Prompt.ask(
                "  Output variable name", default=f"step_{step_count}_output"
            )

            # Execution mode
            if step_count > 1:
                parallel = Confirm.ask(
                    "  Run in parallel with previous step?", default=False
                )
                mode = ExecutionMode.PARALLEL if parallel else ExecutionMode.SEQUENTIAL
            else:
                mode = ExecutionMode.SEQUENTIAL

            # Add step
            wf.add_step(
                agent=agent_class,
                inputs=inputs,
                outputs=[output_var],
                execution_mode=mode,
                metadata={"description": f"{agent_name} step"},
            )

            self.console.print(f"[green]✔ Step {step_count} added[/green]")

        # Visualize final workflow
        self.console.print("\n[bold]Workflow Preview:[/bold]")
        wf.visualize(self.console)

        # Save workflow
        if Confirm.ask("\nSave this workflow?", default=True):
            self.save_workflow(wf)

        # Execute workflow
        if Confirm.ask("Execute workflow now?", default=True):
            initial_context = {}

            # Ask for initial context variables
            self.console.print("\n[bold]Initial Context:[/bold]")
            self.console.print(
                "[dim]Enter values for workflow inputs (empty to skip)[/dim]"
            )

            for param in [
                "feature_description",
                "research_query",
                "objective",
                "spec",
                "target",
                "error_log",
                "root_dir",
            ]:
                value = Prompt.ask(f"  {param}", default="")
                if value:
                    initial_context[param] = value

            return wf.execute(initial_context, self.agent)

        return f"Workflow '{workflow_name}' created"

    def generate_dynamic_loop(self, objective: str):
        """Build and run a task-optimized loop."""
        try:
            path = self.loop_builder.generate_loop(objective)
            loop = self.loop_builder.instantiate_loop(path, objective)

            self.console.print(
                "[bold green]▶ Running Task-Optimized Loop...[/bold green]"
            )
            return loop.run(objective)
        except Exception as e:
            return f"Failed to build/run dynamic loop: {e}"

    def run_legacy(self, template_name):
        # Fallback to existing logic for legacy templates
        # (Simplified for brevity, assuming existing logic was ok or I can implement basics)
        if template_name == "Code Analysis":
            return self.run_code_analysis()
        return f"Legacy {template_name} running..."

    def run_code_analysis(self):
        """Legacy Saguaro analysis"""
        from domains.code_intelligence.saguaro_substrate import SaguaroSubstrate

        saguaro = SaguaroSubstrate()
        self.console.rule("[bold magenta]Wizard: Code Analysis[/bold magenta]")
        self.console.print(saguaro.agent_skeleton("."))
        return "Code Analysis workflow completed."

    def generate_custom_workflow(self, description: str):
        """
        Generate a dynamic workflow using subagents.
        """
        self.console.print(
            f"Generating custom workflow for: [italic]{description}[/italic]"
        )

        # We want the LLM to write python code that instantiates SubagentWorkflow and adds steps
        prompt = f"""
        You are the Anvil Meta-Agent. Write a Python function `create_workflow()` that returns a `SubagentWorkflow` object.
        
        Available Subagents:
        - RepoAnalysisSubagent (args: root_dir)
        - ResearchSubagent (args: research_query)
        - DebugSubagent (args: error_log)
        - ImplementationSubagent (args: spec)
        - PlanningSubagent (args: objective)
        - TestingSubagent (args: target)
        
        Goal: {description}
        
        Code must import classes from core.agents.*.
        Example:
        ```python
        from core.wizard import SubagentWorkflow
        from core.agents.researcher import ResearchSubagent
        
        def create_workflow():
            wf = SubagentWorkflow("custom")
            wf.add_step(ResearchSubagent, inputs={{"research_query": "{{input}}"}}, outputs=["findings"])
            return wf
        ```
        """

        response = self.agent.run_loop(prompt)
        text = response.get("response", "")

        # Extract code
        import re

        match = re.search(r"```python(.*?)```", text, re.DOTALL)
        if match:
            code = match.group(1).strip()

            # Save to temp file and load
            os.makedirs(".anvil/workflows", exist_ok=True)
            path = ".anvil/workflows/custom_latest.py"
            with open(path, "w") as f:
                f.write(code)

            self.console.print(f"[green]Workflow generated at {path}[/green]")

            try:
                import importlib.util

                spec = importlib.util.spec_from_file_location("custom_wf", path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                if hasattr(module, "create_workflow"):
                    wf = module.create_workflow()
                    # Ask for initial context
                    ctx = {}
                    # Naively ask for 'input' if referenced?
                    # For now just prompt for 'input'
                    val = Prompt.ask("Enter value for '{input}' context")
                    ctx["input"] = val

                    return wf.execute(ctx, self.agent)
            except Exception as e:
                return f"Failed to load/execute workflow: {e}"

        return "Failed to generate workflow code."

    def save_workflow(
        self, workflow: SubagentWorkflow, registry_name: Optional[str] = None
    ):
        """
        Save workflow to registry.
        Phase 3.4: Workflow persistence and registry.

        Args:
            workflow: SubagentWorkflow instance to save
            registry_name: Optional custom name for registry
        """
        registry_name = registry_name or workflow.name

        # Add to in-memory registry
        self.workflow_registry[registry_name] = workflow

        # Serialize to disk
        workflow_dir = ".anvil/workflows"
        os.makedirs(workflow_dir, exist_ok=True)

        # Serialize workflow structure
        workflow_data = {
            "name": workflow.name,
            "description": workflow.description,
            "steps": [],
            "created_at": time.time(),
        }

        for step in workflow.steps:
            step_data = {
                "agent_class": step.agent.__name__,
                "agent_module": step.agent.__module__,
                "inputs": step.inputs,
                "outputs": step.outputs,
                "execution_mode": step.execution_mode.value,
                "metadata": step.metadata,
            }
            workflow_data["steps"].append(step_data)

        # Save to JSON
        workflow_path = os.path.join(workflow_dir, f"{registry_name}.json")

        with open(workflow_path, "w") as f:
            json.dump(workflow_data, f, indent=2)

        self.console.print(
            f"[green]✔ Workflow saved to registry: {workflow_path}[/green]"
        )
        return workflow_path

    def load_workflow(self, registry_name: str) -> Optional[SubagentWorkflow]:
        """
        Load workflow from registry.
        Phase 3.4: Workflow loading and reuse.

        Args:
            registry_name: Name of workflow in registry

        Returns:
            SubagentWorkflow instance or None if not found
        """
        # Check in-memory registry first
        if registry_name in self.workflow_registry:
            return self.workflow_registry[registry_name]

        # Load from disk
        workflow_path = os.path.join(".anvil/workflows", f"{registry_name}.json")

        if not os.path.exists(workflow_path):
            self.console.print(
                f"[red]Workflow '{registry_name}' not found in registry[/red]"
            )
            return None

        try:
            with open(workflow_path, "r") as f:
                workflow_data = json.load(f)

            # Reconstruct workflow
            wf = SubagentWorkflow(
                name=workflow_data["name"],
                description=workflow_data.get("description", ""),
            )

            # Map agent class names back to classes
            agent_map = {
                "ResearchSubagent": ResearchSubagent,
                "RepoAnalysisSubagent": RepoAnalysisSubagent,
                "DebugSubagent": DebugSubagent,
                "ImplementationSubagent": ImplementationSubagent,
                "PlanningSubagent": PlanningSubagent,
                "TestingSubagent": TestingSubagent,
            }

            for step_data in workflow_data["steps"]:
                agent_class_name = step_data["agent_class"]
                agent_class = agent_map.get(agent_class_name)

                if not agent_class:
                    self.console.print(
                        f"[yellow]⚠ Unknown agent class: {agent_class_name}[/yellow]"
                    )
                    continue

                wf.add_step(
                    agent=agent_class,
                    inputs=step_data["inputs"],
                    outputs=step_data["outputs"],
                    execution_mode=ExecutionMode(
                        step_data.get("execution_mode", "sequential")
                    ),
                    metadata=step_data.get("metadata", {}),
                )

            # Cache in registry
            self.workflow_registry[registry_name] = wf

            self.console.print(
                f"[green]✔ Workflow '{registry_name}' loaded from registry[/green]"
            )
            return wf

        except Exception as e:
            self.console.print(f"[red]Failed to load workflow: {e}[/red]")
            return None

    def list_workflows(self) -> List[str]:
        """
        List all available workflows in registry.
        Phase 3.4: Workflow browsing.

        Returns:
            List of workflow names
        """
        workflow_dir = ".anvil/workflows"

        if not os.path.exists(workflow_dir):
            return []

        workflows = []

        for filename in os.listdir(workflow_dir):
            if filename.endswith(".json"):
                workflow_name = filename[:-5]  # Remove .json
                workflows.append(workflow_name)

        return workflows

    def browse_workflows(self):
        """
        Interactive workflow browser.
        Phase 3.4: Workflow discovery and execution.
        """
        workflows = self.list_workflows()

        if not workflows:
            self.console.print("[yellow]No saved workflows found[/yellow]")
            return

        self.console.print("\n[bold magenta]📚 Workflow Registry[/bold magenta]\n")

        workflow_table = Table(
            show_header=True, header_style="bold cyan", box=box.ROUNDED
        )
        workflow_table.add_column("#", style="cyan", width=3)
        workflow_table.add_column("Name", style="green")
        workflow_table.add_column("Steps", style="dim")

        for i, workflow_name in enumerate(workflows, 1):
            # Load metadata
            try:
                workflow_path = os.path.join(
                    ".anvil/workflows", f"{workflow_name}.json"
                )
                with open(workflow_path, "r") as f:
                    data = json.load(f)
                    step_count = len(data.get("steps", []))
            except Exception:
                step_count = "?"

            workflow_table.add_row(str(i), workflow_name, str(step_count))

        self.console.print(workflow_table)

        # Select and run
        if Confirm.ask("\nLoad a workflow?", default=True):
            choice_idx = (
                int(
                    Prompt.ask(
                        "Select workflow #",
                        choices=[str(i) for i in range(1, len(workflows) + 1)],
                    )
                )
                - 1
            )

            selected_name = workflows[choice_idx]
            wf = self.load_workflow(selected_name)

            if wf:
                # Visualize
                wf.visualize(self.console)

                if Confirm.ask("\nExecute this workflow?", default=True):
                    # Get initial context
                    initial_context = {}
                    self.console.print(
                        "\n[dim]Enter initial context (empty to skip)[/dim]"
                    )

                    for param in [
                        "feature_description",
                        "research_query",
                        "objective",
                        "bug_description",
                    ]:
                        value = Prompt.ask(f"  {param}", default="")
                        if value:
                            initial_context[param] = value

                    return wf.execute(initial_context, self.agent)

    def export_registry(self, output_path: str = ".anvil/workflows/registry.json"):
        """
        Export complete workflow registry metadata.
        Phase 3.4: Registry export for sharing/backup.
        """
        workflows = self.list_workflows()

        registry_data = {
            "total_workflows": len(workflows),
            "workflows": [],
            "exported_at": time.time(),
        }

        for workflow_name in workflows:
            workflow_path = os.path.join(".anvil/workflows", f"{workflow_name}.json")

            try:
                with open(workflow_path, "r") as f:
                    workflow_data = json.load(f)

                registry_data["workflows"].append(
                    {
                        "name": workflow_name,
                        "description": workflow_data.get("description", ""),
                        "steps": len(workflow_data.get("steps", [])),
                        "created_at": workflow_data.get("created_at", 0),
                    }
                )
            except Exception:
                continue

        with open(output_path, "w") as f:
            json.dump(registry_data, f, indent=2)

        self.console.print(f"[green]✔ Registry exported to {output_path}[/green]")
        return output_path

    def run_workflow(self, name: str, context: Dict):
        """
        Run a saved workflow by name.
        Phase 3.4: Direct workflow execution.

        Args:
            name: Workflow name in registry
            context: Initial context variables

        Returns:
            Workflow execution result
        """
        wf = self.load_workflow(name)

        if wf:
            return wf.execute(context, self.agent)

        return f"Workflow '{name}' not found"
