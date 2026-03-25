"""
Task Memory System with Saguaro

Remembers successful patterns and learns from experience:
- Stores successful solutions in Saguaro memory
- Recalls similar past tasks
- Learns which approaches work best
- Adapts strategies based on history
"""

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path

from core.memory.fabric import MemoryFabricStore, MemoryProjector

@dataclass
class TaskMemory:
    """Record of a completed task."""

    task_id: str
    task_type: str  # "edit", "create", "refactor", "debug", etc.
    description: str
    files_modified: List[str]
    tools_used: List[str]
    success: bool
    execution_time: float
    iterations: int
    timestamp: float

    # Solution details
    approach: str  # Description of approach taken
    key_steps: List[str]
    difficulties: List[str]

    # Verification results
    tests_passed: bool
    syntax_valid: bool

    # Saguaro metadata
    semantic_signature: Optional[List[float]] = None  # Embedding vector
    related_patterns: List[str] = None  # IDs of similar tasks
    workflow_graph: Optional[Dict[str, Any]] = None
    risk_budget: Optional[Dict[str, Any]] = None
    provenance_bundle: Optional[Dict[str, Any]] = None
    runtime_symbols: List[Dict[str, Any]] = field(default_factory=list)
    counterexamples: List[Dict[str, Any]] = field(default_factory=list)


class TaskMemoryManager:
    """
    Manage memory of past tasks using Saguaro.

    Stores memories both:
    1. In local JSON (fast access)
    2. In Saguaro memory (semantic search)
    """

    def __init__(self, saguaro_tools, semantic_engine, console):
        self.saguaro_tools = saguaro_tools
        self.semantic_engine = semantic_engine
        self.console = console

        # Local cache
        self.memory_file = Path(".anvil/memory/task_history.json")
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        self.fabric_store = MemoryFabricStore.from_db_path(
            str(self.memory_file.parent / "almf.db")
        )
        self.memory_projector = MemoryProjector()

        self.memories: List[TaskMemory] = []
        self._load_memories()

    def remember(self, task: TaskMemory):
        """
        Store a task memory.

        Stores both locally and in Saguaro memory for semantic search.
        """
        self.console.print(f"[dim]→ Recording task memory: {task.task_id}[/dim]")
        if task.workflow_graph is None:
            task.workflow_graph = self._workflow_graph(task)

        # Generate semantic signature using Saguaro
        if self.semantic_engine:
            try:
                # Get embedding for task description
                task.semantic_signature = self._get_embedding(task.description)

            except Exception as e:
                self.console.print(
                    f"[yellow]Could not generate semantic signature: {e}[/yellow]"
                )

        # Store in Saguaro memory
        try:
            self.saguaro_tools.memory(
                action="store",
                key=task.task_id,
                value=json.dumps(asdict(task)),
                tier="persistent",
                tags=[task.task_type, "success" if task.success else "failure"],
            )

        except Exception as e:
            raise RuntimeError(
                f"SAGUARO_STRICT_MEMORY_STORE_FAILED: {e}"
            ) from e

        # Add to local cache
        self.memories.append(task)
        memory = self.fabric_store.create_memory(
            memory_kind="task_memory",
            payload_json=asdict(task),
            campaign_id="task_memory",
            workspace_id="task_memory",
            source_system="task_memory_manager",
            summary_text=task.description,
            task_packet_id=task.task_id,
            lifecycle_state="completed" if task.success else "failed",
            importance_score=0.85 if task.success else 0.5,
            confidence_score=1.0 if task.tests_passed else 0.6,
        )
        self.fabric_store.register_alias(
            memory.memory_id,
            "task_memory",
            task.task_id,
            campaign_id="task_memory",
        )
        self.memory_projector.project_memory(
            self.fabric_store,
            memory,
            include_multivector=True,
        )

        # Save to disk
        self._save_memories()

        self.console.print("[green]✓ Task memory recorded[/green]")

    def recall_similar(self, task_description: str, limit: int = 5) -> List[TaskMemory]:
        """
        Recall similar past tasks.

        Uses Saguaro's semantic search to find tasks with similar descriptions.
        """
        self.console.print("[dim]→ Recalling similar tasks...[/dim]")

        similar = []

        try:
            payload = self.saguaro_tools.memory(action="recall", query=task_description)
        except Exception as e:
            raise RuntimeError(
                f"SAGUARO_STRICT_MEMORY_RECALL_FAILED: {e}"
            ) from e

        similar = self._parse_recalled_memories(payload, limit=limit)

        self.console.print(f"[dim]→ Found {len(similar)} similar tasks[/dim]")

        return similar

    def _parse_recalled_memories(self, payload: Any, limit: int) -> List[TaskMemory]:
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception as e:
                raise RuntimeError(
                    f"SAGUARO_STRICT_MEMORY_PARSE_FAILED: {e}"
                ) from e

        if not isinstance(payload, dict):
            raise RuntimeError(
                "SAGUARO_STRICT_MEMORY_PARSE_FAILED: recall payload is not a dict."
            )

        matches = payload.get("matches", [])
        if not isinstance(matches, list):
            raise RuntimeError(
                "SAGUARO_STRICT_MEMORY_PARSE_FAILED: recall payload missing 'matches'."
            )

        recalled: List[TaskMemory] = []
        for match in matches:
            if not isinstance(match, dict):
                continue
            raw_value = match.get("value")
            if raw_value is None:
                continue
            try:
                task_data = json.loads(str(raw_value))
                recalled.append(TaskMemory(**task_data))
            except Exception:
                continue

        recalled.sort(key=lambda item: item.timestamp, reverse=True)
        return recalled[:limit]

    def recall_by_type(
        self, task_type: str, success_only: bool = True
    ) -> List[TaskMemory]:
        """
        Recall tasks of a specific type.

        Args:
            task_type: Type of task ("edit", "create", etc.)
            success_only: Only return successful tasks
        """
        results = [
            m
            for m in self.memories
            if m.task_type == task_type and (not success_only or m.success)
        ]

        return sorted(results, key=lambda m: m.timestamp, reverse=True)

    def get_success_patterns(self, task_type: str) -> Dict[str, Any]:
        """
        Analyze successful patterns for a task type.

        Returns insights like:
        - Most common tools used
        - Average iterations needed
        - Common difficulties
        - Success rate
        """
        tasks = self.recall_by_type(task_type, success_only=False)

        if not tasks:
            return {"success_rate": 0, "total_tasks": 0}

        successful = [t for t in tasks if t.success]

        # Aggregate statistics
        patterns = {
            "success_rate": len(successful) / len(tasks),
            "total_tasks": len(tasks),
            "avg_iterations": (
                sum(t.iterations for t in successful) / len(successful)
                if successful
                else 0
            ),
            "avg_execution_time": (
                sum(t.execution_time for t in successful) / len(successful)
                if successful
                else 0
            ),
        }

        # Most common tools
        tool_counts = {}
        for task in successful:
            for tool in task.tools_used:
                tool_counts[tool] = tool_counts.get(tool, 0) + 1

        patterns["common_tools"] = sorted(
            tool_counts.items(), key=lambda x: x[1], reverse=True
        )[:5]

        # Common difficulties
        difficulty_counts = {}
        for task in tasks:
            for diff in task.difficulties:
                difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1

        patterns["common_difficulties"] = sorted(
            difficulty_counts.items(), key=lambda x: x[1], reverse=True
        )[:5]

        patterns["workflow_graph"] = self._aggregate_workflow_graph(successful)
        patterns["workflow_templates"] = self._workflow_templates(successful)

        return patterns

    def suggest_approach(self, task_description: str, task_type: str) -> Optional[str]:
        """
        Suggest an approach based on similar past tasks.

        Returns a string describing recommended approach.
        """
        # Find similar successful tasks
        similar = self.recall_similar(task_description, limit=3)
        successful_similar = [t for t in similar if t.success]

        if not successful_similar:
            # Fall back to general patterns for this type
            patterns = self.get_success_patterns(task_type)

            if patterns["total_tasks"] > 0:
                suggestion = f"Based on {patterns['total_tasks']} similar tasks:\n"
                suggestion += f"- Success rate: {patterns['success_rate']*100:.1f}%\n"

                if patterns["common_tools"]:
                    tools = [t[0] for t in patterns["common_tools"][:3]]
                    suggestion += f"- Common tools: {', '.join(tools)}\n"
                workflow = patterns.get("workflow_graph") or {}
                recommended_steps = workflow.get("recommended_steps") or []
                if recommended_steps:
                    suggestion += f"- Suggested workflow: {', '.join(recommended_steps[:5])}\n"

                return suggestion

            return None

        # Build suggestion from successful similar tasks
        suggestion = f"Based on {len(successful_similar)} similar successful tasks:\n\n"

        for i, task in enumerate(successful_similar[:2], 1):
            suggestion += f"{i}. {task.approach}\n"
            suggestion += f"   Key steps: {', '.join(task.key_steps[:3])}\n"
            if task.difficulties:
                suggestion += f"   Watch out for: {', '.join(task.difficulties[:2])}\n"
            suggestion += "\n"

        return suggestion

    def _load_memories(self):
        """Load memories from disk."""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, "r") as f:
                    data = json.load(f)

                self.memories = [TaskMemory(**m) for m in data]

                self.console.print(
                    f"[dim]Loaded {len(self.memories)} task memories[/dim]"
                )

            except Exception as e:
                self.console.print(f"[yellow]Failed to load memories: {e}[/yellow]")

    def _save_memories(self):
        """Save memories to disk."""
        try:
            with open(self.memory_file, "w") as f:
                json.dump([asdict(m) for m in self.memories], f, indent=2)

        except Exception as e:
            self.console.print(f"[yellow]Failed to save memories: {e}[/yellow]")

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding vector for text using Saguaro."""
        if not self.semantic_engine:
            return None

        try:
            # Use semantic engine to get embedding
            # (Simplified - real implementation would access embeddings directly)
            return None  # Placeholder

        except Exception:
            return None

    def _workflow_graph(self, task: TaskMemory) -> Dict[str, Any]:
        steps = [step for step in task.key_steps if step]
        transitions = []
        for left, right in zip(steps, steps[1:]):
            transitions.append({"from": left, "to": right, "count": 1})
        return {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "nodes": [{"id": step, "type": "step"} for step in dict.fromkeys(steps)],
            "edges": transitions,
            "files_modified": list(task.files_modified),
            "tools_used": list(task.tools_used),
        }

    def _aggregate_workflow_graph(self, tasks: List[TaskMemory]) -> Dict[str, Any]:
        node_counts: Dict[str, int] = {}
        edge_counts: Dict[tuple[str, str], int] = {}
        for task in tasks:
            graph = task.workflow_graph or self._workflow_graph(task)
            for node in graph.get("nodes", []):
                node_id = str(node.get("id") or "")
                if node_id:
                    node_counts[node_id] = node_counts.get(node_id, 0) + 1
            for edge in graph.get("edges", []):
                left = str(edge.get("from") or "")
                right = str(edge.get("to") or "")
                if left and right:
                    edge_counts[(left, right)] = edge_counts.get((left, right), 0) + 1
        return {
            "nodes": [
                {"id": node_id, "count": count}
                for node_id, count in sorted(
                    node_counts.items(), key=lambda item: (-item[1], item[0])
                )
            ],
            "edges": [
                {"from": left, "to": right, "count": count}
                for (left, right), count in sorted(
                    edge_counts.items(),
                    key=lambda item: (-item[1], item[0][0], item[0][1]),
                )
            ],
            "recommended_steps": [
                node_id
                for node_id, _count in sorted(
                    node_counts.items(), key=lambda item: (-item[1], item[0])
                )[:5]
            ],
        }

    def _workflow_templates(self, tasks: List[TaskMemory]) -> List[Dict[str, Any]]:
        templates: List[Dict[str, Any]] = []
        for task in tasks[:5]:
            graph = task.workflow_graph or self._workflow_graph(task)
            templates.append(
                {
                    "task_id": task.task_id,
                    "approach": task.approach,
                    "tools_used": list(task.tools_used),
                    "workflow_graph": graph,
                }
            )
        return templates

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import math

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0

        return dot_product / (magnitude1 * magnitude2)


class AdaptiveLearner:
    """
    Learn and adapt strategies based on task memory.

    Uses Saguaro to identify patterns in successes and failures.
    """

    def __init__(self, memory_manager: TaskMemoryManager, console):
        self.memory = memory_manager
        self.console = console

    def analyze_failures(self, task_type: str) -> Dict[str, Any]:
        """
        Analyze failed tasks to learn what to avoid.

        Returns:
        - Common failure patterns
        - Tools that led to failures
        - Conditions that predict failure
        """
        failures = [
            m
            for m in self.memory.memories
            if m.task_type == task_type and not m.success
        ]

        if not failures:
            return {"total_failures": 0}

        analysis = {
            "total_failures": len(failures),
            "common_difficulties": [],
            "risky_tools": [],
        }

        # Aggregate difficulties
        diff_counts = {}
        for task in failures:
            for diff in task.difficulties:
                diff_counts[diff] = diff_counts.get(diff, 0) + 1

        analysis["common_difficulties"] = sorted(
            diff_counts.items(), key=lambda x: x[1], reverse=True
        )[:5]

        # Find tools used more in failures than successes
        successes = [
            m for m in self.memory.memories if m.task_type == task_type and m.success
        ]

        failure_tools = {}
        for task in failures:
            for tool in task.tools_used:
                failure_tools[tool] = failure_tools.get(tool, 0) + 1

        success_tools = {}
        for task in successes:
            for tool in task.tools_used:
                success_tools[tool] = success_tools.get(tool, 0) + 1

        # Find tools with higher failure rate
        risky = []
        for tool, fail_count in failure_tools.items():
            succ_count = success_tools.get(tool, 0)
            total = fail_count + succ_count

            if total > 0:
                failure_rate = fail_count / total
                if failure_rate > 0.5:  # More failures than successes
                    risky.append((tool, failure_rate))

        analysis["risky_tools"] = sorted(risky, key=lambda x: x[1], reverse=True)[:3]

        return analysis

    def recommend_strategy(
        self, task_description: str, task_type: str
    ) -> Dict[str, Any]:
        """
        Recommend optimal strategy based on learned patterns.

        Returns:
        - recommended_approach: Description
        - tools_to_use: List of effective tools
        - tools_to_avoid: List of risky tools
        - estimated_iterations: Expected number of iterations
        - confidence: How confident we are (based on sample size)
        """
        # Get success patterns
        patterns = self.memory.get_success_patterns(task_type)

        # Get similar tasks
        self.memory.recall_similar(task_description, limit=5)

        # Analyze failures
        failure_analysis = self.analyze_failures(task_type)

        # Build recommendation
        recommendation = {
            "recommended_approach": None,
            "tools_to_use": [],
            "tools_to_avoid": [],
            "estimated_iterations": 3,  # Default
            "confidence": 0.0,
        }

        if patterns["total_tasks"] > 0:
            recommendation["confidence"] = min(patterns["total_tasks"] / 10, 1.0)

            # Recommend common tools
            if patterns.get("common_tools"):
                recommendation["tools_to_use"] = [
                    t[0] for t in patterns["common_tools"][:3]
                ]

            # Suggest avoiding risky tools
            if failure_analysis.get("risky_tools"):
                recommendation["tools_to_avoid"] = [
                    t[0] for t in failure_analysis["risky_tools"]
                ]

            # Estimate iterations
            if patterns.get("avg_iterations"):
                recommendation["estimated_iterations"] = int(
                    patterns["avg_iterations"] * 1.5
                )  # Buffer

        # Get approach from similar tasks
        recommendation["recommended_approach"] = self.memory.suggest_approach(
            task_description, task_type
        )

        return recommendation


class ContextCompressionMemory:
    """
    Persist compressed tool-result summaries across sessions.
    """

    def __init__(self, store_path: str = ".anvil/memory/context_summaries.json"):
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._data: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self.store_path.exists():
            self._data = {}
            return
        try:
            with open(self.store_path, "r") as f:
                payload = json.load(f)
            self._data = payload if isinstance(payload, dict) else {}
        except Exception:
            self._data = {}

    def _save(self) -> None:
        with open(self.store_path, "w") as f:
            json.dump(self._data, f, indent=2)

    def remember_summary(
        self,
        session_id: str,
        tc_id: int,
        summary: str,
        tool_name: Optional[str] = None,
        tool_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not session_id or tc_id <= 0 or not summary:
            return
        session = self._data.setdefault(session_id, {})
        session[f"tc{tc_id}"] = {
            "summary": summary,
            "tool_name": tool_name,
            "tool_args": tool_args or {},
            "updated_at": time.time(),
        }
        self._save()

    def get_summary_map(self, session_id: str) -> Dict[str, str]:
        session = self._data.get(session_id, {})
        result: Dict[str, str] = {}
        for tc_label, payload in session.items():
            if isinstance(payload, dict) and payload.get("summary"):
                result[tc_label] = payload["summary"]
        return result

    def get_updates_payload(self, session_id: str) -> List[Dict[str, str]]:
        summary_map = self.get_summary_map(session_id)
        return [{tc: summary} for tc, summary in sorted(summary_map.items())]
