import enum
from typing import List, Dict, Any, Optional
from domains.coordination.shared_context import SharedMemoryNamespace
from core.agents.subagent import SubAgent
from shared_kernel.event_store import get_event_store
from domains.verification.coherence_engine import CoherenceEngine


class SwarmTopology(enum.Enum):
    HIERARCHICAL = "hierarchical"  # Queen -> Workers
    SEQUENTIAL = "sequential"  # Agent A -> Agent B -> Agent C
    ROUND_ROBIN = "round_robin"  # Cycles through agents


class SequentialSwarmCoordinator:
    """
    Coordinates a swarm of virtual agents executing sequentially.
    Optimized for CPU/Memory constraints.
    """

    def __init__(
        self,
        parent_agent,
        topology: SwarmTopology = SwarmTopology.HIERARCHICAL,
        shared_context: Optional[SharedMemoryNamespace] = None,
    ):
        self.parent_agent = parent_agent
        self.topology = topology
        self.shared_context = shared_context or SharedMemoryNamespace(
            compressor=getattr(parent_agent, "compressor", None)
        )
        self.event_store = get_event_store()
        self.coherence_engine = CoherenceEngine()

    async def execute_swarm(
        self, task_description: str, agent_sequence: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Executes a sequence of agents according to the chosen topology.
        agent_sequence: List of dicts with {'type': str, 'instruction': str, 'tools': Optional[List[str]]}
        """
        self.event_store.emit(
            event_type="SWARM_STARTED",
            source="SequentialSwarmCoordinator",
            payload={
                "topology": self.topology.value,
                "agent_count": len(agent_sequence),
            },
        )

        results = []

        for i, agent_info in enumerate(agent_sequence):
            agent_type = agent_info.get("type", "generic")
            instruction = agent_info.get("instruction", "")
            tools = agent_info.get("tools")

            # Inject relevant context from previous agents
            prior_context = self.shared_context.get_relevant_context(instruction, k=2)
            context_str = ""
            if prior_context:
                context_str = "\n\nRelevant context from prior steps:\n" + "\n".join(
                    [f"- From {c['agent_id']}: {c['summary']}" for c in prior_context]
                )

            full_task = f"{instruction}{context_str}"

            # Create and run the agent
            # Note: In a real implementation, we would look up the specific class for agent_type
            # For Phase 2, we use SubAgent as the base for all.
            agent_instance = self._create_agent(agent_type, full_task, tools)

            self.event_store.emit(
                event_type="AGENT_SPAWNED",
                source="SequentialSwarmCoordinator",
                payload={
                    "agent_type": agent_type,
                    "step": i,
                    "topology": self.topology.value,
                },
            )

            # Sequential execution: one at a time
            result = agent_instance.run()

            # Update shared memory
            self.shared_context.update(
                agent_id=f"{agent_type}_{i}",
                summary=result.get("summary", ""),
                data=result,
            )

            results.append({"agent_type": agent_type, "output": result})

            # Anti-drift: If Hierarchical, Queen (parent) validates output
            if self.topology == SwarmTopology.HIERARCHICAL:
                validation_passed = self._validate_step(agent_type, result)
                if not validation_passed:
                    # TODO: Implement rollback or correction logic
                    pass

        self.event_store.emit(
            event_type="SWARM_COMPLETED",
            source="SequentialSwarmCoordinator",
            payload={"topology": self.topology.value, "steps": len(results)},
        )

        return {
            "status": "completed",
            "results": results,
            "final_summary": results[-1]["output"].get("summary") if results else "",
        }

    def _create_agent(
        self, agent_type: str, task: str, tools: Optional[List[str]] = None
    ) -> SubAgent:
        """Creates a specialist agent instance based on the registry."""
        from domains.agent_lifecycle.registry import get_agent_config

        config = get_agent_config(agent_type)

        # Merge tools: priority to explicit tools, then registry, then all
        final_tools = tools if tools is not None else config.get("tools")

        agent = SubAgent(
            task=task,
            parent_name=(
                self.parent_agent.name
                if hasattr(self.parent_agent, "name")
                else "Coordinator"
            ),
            parent_agent=self.parent_agent,
            quiet=False,
        )

        # Apply registry-derived system prompt and tools
        agent.system_prompt = config.get("system_prompt", agent.system_prompt)

        if final_tools:
            agent.tools = final_tools
            # Re-filter schemas
            all_schemas = agent.registry.get_schemas().get("tools", [])
            agent.tool_schemas = [s for s in all_schemas if s["name"] in final_tools]

        return agent

    def _validate_step(self, agent_type: str, result: Dict[str, Any]) -> bool:
        """Queen agent validation logic using mathematical coherence."""
        summary = result.get("summary", "")
        if not summary:
            return False

        # Perform mathematical validation on the summary/trace
        coherence_results = self.coherence_engine.validate_trace([summary])

        passed = coherence_results["passed"]

        if not passed:
            self.event_store.emit(
                event_type="COHERENCE_VIOLATION",
                source="SequentialSwarmCoordinator",
                payload={"agent_type": agent_type, "results": coherence_results},
            )

        return passed
