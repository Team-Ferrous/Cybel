from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ComponentDescriptor:
    qualified_name: str
    file_path: str
    component_type: str
    language: str
    terms: list[str] = field(default_factory=list)
    contracts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ComponentCatalog:
    def __init__(self, components: list[ComponentDescriptor] | None = None) -> None:
        self.components = list(components or [])

    @classmethod
    def from_entities(
        cls,
        entities: list[Any],
        *,
        language: str = "python",
    ) -> "ComponentCatalog":
        components = []
        for entity in entities:
            metadata = dict(getattr(entity, "metadata", {}) or {})
            components.append(
                ComponentDescriptor(
                    qualified_name=str(
                        metadata.get("qualified_name")
                        or metadata.get("display_name")
                        or getattr(entity, "name", "")
                    ),
                    file_path=str(getattr(entity, "file_path", "")),
                    component_type=str(getattr(entity, "type", "unknown")),
                    language=language,
                    terms=list(metadata.get("terms") or []),
                    contracts=list(metadata.get("contracts") or []),
                    metadata=metadata,
                )
            )
        return cls(components)

    def by_language(self, language: str) -> list[ComponentDescriptor]:
        return [item for item in self.components if item.language == language]

    def to_payload(self) -> list[dict[str, Any]]:
        return [component.as_dict() for component in self.components]
