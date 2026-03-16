"""Typed roadmap task graph utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Set


@dataclass
class TaskNode:
    item_id: str
    phase_id: str
    title: str
    depends_on: List[str] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)


class CampaignTaskGraph:
    """Dependency-valid roadmap graph."""

    def __init__(self, nodes: Iterable[TaskNode] | None = None) -> None:
        self.nodes: Dict[str, TaskNode] = {}
        for node in nodes or []:
            self.add_node(node)

    def add_node(self, node: TaskNode) -> None:
        self.nodes[node.item_id] = node

    def validate(self) -> List[str]:
        errors: List[str] = []
        for node in self.nodes.values():
            for dependency in node.depends_on:
                if dependency not in self.nodes:
                    errors.append(f"{node.item_id} depends on missing item {dependency}")
        errors.extend(self._detect_cycles())
        return errors

    def ready_items(self, completed: Iterable[str]) -> List[TaskNode]:
        completed_set = set(completed)
        ready: List[TaskNode] = []
        for node in self.nodes.values():
            if node.item_id in completed_set:
                continue
            if all(dependency in completed_set for dependency in node.depends_on):
                ready.append(node)
        ready.sort(key=lambda item: (item.phase_id, item.item_id))
        return ready

    def to_dict(self) -> Dict[str, object]:
        return {
            "nodes": [asdict(self.nodes[key]) for key in sorted(self.nodes)],
            "validation_errors": self.validate(),
        }

    def _detect_cycles(self) -> List[str]:
        temporary: Set[str] = set()
        permanent: Set[str] = set()
        cycles: List[str] = []

        def visit(node_id: str, stack: List[str]) -> None:
            if node_id in permanent:
                return
            if node_id in temporary:
                cycle = " -> ".join(stack + [node_id])
                cycles.append(f"cycle:{cycle}")
                return
            temporary.add(node_id)
            node = self.nodes.get(node_id)
            if node is not None:
                for dependency in node.depends_on:
                    visit(dependency, stack + [node_id])
            temporary.remove(node_id)
            permanent.add(node_id)

        for node_id in sorted(self.nodes):
            visit(node_id, [])
        return cycles
