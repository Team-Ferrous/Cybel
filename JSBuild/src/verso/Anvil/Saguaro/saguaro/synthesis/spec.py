from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class SpecConstraint:
    kind: str
    expression: str
    description: str = ""
    required: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SpecEvidenceRef:
    kind: str
    ref: str
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SpecVerification:
    commands: list[str] = field(default_factory=list)
    tests: list[str] = field(default_factory=list)
    proofs_required: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SagSpec:
    objective: str
    title: str
    stage: str
    language: str
    target_files: list[str]
    outputs: dict[str, str]
    inputs: dict[str, str] = field(default_factory=dict)
    constraints: list[SpecConstraint] = field(default_factory=list)
    evidence: list[SpecEvidenceRef] = field(default_factory=list)
    verification: SpecVerification = field(default_factory=SpecVerification)
    deterministic: bool = True
    origin: str = "deterministic_lowerer"
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "title": self.title,
            "stage": self.stage,
            "language": self.language,
            "target_files": list(self.target_files),
            "outputs": dict(self.outputs),
            "inputs": dict(self.inputs),
            "constraints": [item.as_dict() for item in self.constraints],
            "evidence": [item.as_dict() for item in self.evidence],
            "verification": self.verification.as_dict(),
            "deterministic": self.deterministic,
            "origin": self.origin,
            "metadata": dict(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        return self.as_dict()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SagSpec":
        return cls(
            objective=str(payload.get("objective") or ""),
            title=str(payload.get("title") or ""),
            stage=str(payload.get("stage") or "bounded_function"),
            language=str(payload.get("language") or "python"),
            target_files=[str(item) for item in list(payload.get("target_files") or [])],
            outputs={
                str(key): str(value)
                for key, value in dict(payload.get("outputs") or {}).items()
            },
            inputs={
                str(key): str(value)
                for key, value in dict(payload.get("inputs") or {}).items()
            },
            constraints=[
                SpecConstraint(
                    kind=str(item.get("kind") or ""),
                    expression=str(item.get("expression") or ""),
                    description=str(item.get("description") or ""),
                    required=bool(item.get("required", True)),
                )
                for item in list(payload.get("constraints") or [])
            ],
            evidence=[
                SpecEvidenceRef(
                    kind=str(item.get("kind") or ""),
                    ref=str(item.get("ref") or ""),
                    note=str(item.get("note") or ""),
                )
                for item in list(payload.get("evidence") or [])
            ],
            verification=SpecVerification(
                commands=[str(item) for item in list(dict(payload.get("verification") or {}).get("commands") or [])],
                tests=[str(item) for item in list(dict(payload.get("verification") or {}).get("tests") or [])],
                proofs_required=bool(
                    dict(payload.get("verification") or {}).get("proofs_required", True)
                ),
            ),
            deterministic=bool(payload.get("deterministic", True)),
            origin=str(payload.get("origin") or "deterministic_lowerer"),
            metadata=dict(payload.get("metadata") or {}),
        )

    def canonical_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True)

    def stable_digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def completeness_score(self) -> float:
        checks = [
            bool(self.objective.strip()),
            bool(self.title.strip()),
            bool(self.stage.strip()),
            bool(self.language.strip()),
            bool(self.target_files),
            bool(self.outputs),
            bool(self.verification.commands),
        ]
        return round(sum(1.0 for item in checks if item) / len(checks), 3)

    def proposed_changes(self) -> list[dict[str, str]]:
        description = self.metadata.get("change_summary") or self.objective
        return [
            {"file_path": path, "description": str(description), "change_type": "modify"}
            for path in self.target_files
        ]


class SpecLowerer:
    """Deterministically lower bounded objectives into `SagSpec`."""

    _PATH_PATTERN = re.compile(r"`([^`]+\.[A-Za-z0-9_]+)`|([A-Za-z0-9_./-]+\.[A-Za-z0-9_]+)")

    def lower_objective(self, objective: str, *, origin: str = "deterministic_lowerer") -> SagSpec:
        objective_text = str(objective or "").strip()
        markdown_text = self._load_markdown_source(objective_text)
        if markdown_text is not None:
            return self.lower_markdown_roadmap(
                markdown_text,
                source=objective_text,
                origin=origin,
            )
        target_files = self._target_files(objective_text)
        language = self._language_for_targets(objective_text, target_files)
        output_name = self._output_name(objective_text, target_files)
        verification = self._verification(language, target_files)
        constraints = self._constraints(objective_text, language)
        evidence = self._evidence_refs(target_files)
        metadata = {
            "change_summary": self._change_summary(objective_text),
            "objective_tokens": self._objective_tokens(objective_text),
        }
        return SagSpec(
            objective=objective_text,
            title=self._title(objective_text),
            stage="bounded_function",
            language=language,
            target_files=target_files,
            outputs={output_name: self._default_output_type(language)},
            inputs=self._default_inputs(objective_text, language),
            constraints=constraints,
            evidence=evidence,
            verification=verification,
            deterministic=True,
            origin=origin,
            metadata=metadata,
        )

    def lower(self, objective: str) -> SagSpec:
        return self.lower_objective(objective)

    def lower_markdown_roadmap(
        self,
        markdown_text: str,
        *,
        source: str = "",
        origin: str = "markdown_roadmap",
    ) -> SagSpec:
        text = str(markdown_text or "")
        objective = self._markdown_goal(text) or self._change_summary(text)
        target_files = self._markdown_target_files(text) or self._target_files(text)
        language = self._language_for_targets(text, target_files)
        verification = SpecVerification(
            commands=self._markdown_verification(text)
            or self._verification(language, target_files).commands,
            tests=[
                item
                for item in self._markdown_target_files(text)
                if item.startswith("tests/")
            ],
            proofs_required=True,
        )
        metadata = {
            "change_summary": self._change_summary(objective),
            "objective_tokens": self._objective_tokens(objective),
            "roadmap_source": source,
        }
        return SagSpec(
            objective=objective,
            title=self._markdown_title(text),
            stage="roadmap_lowering",
            language=language,
            target_files=[
                item
                for item in target_files
                if not item.startswith("tests/")
            ],
            outputs={self._output_name(objective, target_files): self._default_output_type(language)},
            inputs=self._default_inputs(objective, language),
            constraints=self._constraints(objective, language),
            evidence=self._evidence_refs(target_files),
            verification=verification,
            deterministic=True,
            origin=origin,
            metadata=metadata,
        )

    def from_model_payload(
        self,
        payload: str | dict[str, Any],
        *,
        objective: str = "",
        origin: str = "model_payload",
    ) -> SagSpec:
        if isinstance(payload, dict):
            data = dict(payload)
        else:
            data = self._extract_json(payload)
        if data:
            data.setdefault("objective", objective)
            data.setdefault("origin", origin)
            return SagSpec.from_dict(data)
        return self.lower_objective(objective or str(payload), origin=origin)

    def _extract_json(self, payload: str) -> dict[str, Any]:
        text = str(payload or "").strip()
        for pattern in (r"<sagspec>(.*?)</sagspec>", r"(\{.*\})"):
            match = re.search(pattern, text, re.DOTALL)
            if not match:
                continue
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
        return {}

    def _load_markdown_source(self, objective: str) -> str | None:
        if not objective:
            return None
        if objective.endswith(".md") and os.path.exists(objective):
            return open(objective, encoding="utf-8").read()
        if "\n#" in objective or objective.lstrip().startswith("#"):
            return objective
        return None

    def _target_files(self, objective: str) -> list[str]:
        seen: list[str] = []
        for match in self._PATH_PATTERN.finditer(objective):
            path = str(match.group(1) or match.group(2) or "").strip().strip(".,)")
            if path and path not in seen and ("/" in path or "." in path):
                seen.append(path)
        if seen:
            return seen
        lowered = objective.lower()
        if "cpp" in lowered or "c++" in lowered:
            return ["generated/synthesis_candidate.cpp"]
        return ["generated/synthesis_candidate.py"]

    def _language_for_targets(self, objective: str, target_files: list[str]) -> str:
        lowered = objective.lower()
        if any(path.endswith((".cc", ".cpp", ".cxx", ".hpp", ".h")) for path in target_files):
            return "cpp"
        if "cpp" in lowered or "c++" in lowered or "native" in lowered:
            return "cpp"
        return "python"

    def _output_name(self, objective: str, target_files: list[str]) -> str:
        tokens = [token for token in re.findall(r"[A-Za-z][A-Za-z0-9_]+", objective.lower()) if token not in {"the", "and", "for", "with"}]
        if tokens:
            return "_".join(tokens[:3])
        stem = target_files[0].rsplit("/", 1)[-1].split(".", 1)[0]
        return f"build_{stem}"

    def _default_output_type(self, language: str) -> str:
        return "double" if language == "cpp" else "float"

    def _default_inputs(self, objective: str, language: str) -> dict[str, str]:
        lowered = objective.lower()
        if any(token in lowered for token in {"normalize", "clamp", "bound", "ratio"}):
            scalar = "double" if language == "cpp" else "float"
            return {"value": scalar, "lower": scalar, "upper": scalar}
        scalar = "double" if language == "cpp" else "float"
        return {"value": scalar}

    def _verification(self, language: str, target_files: list[str]) -> SpecVerification:
        tests = []
        commands = []
        if language == "cpp":
            commands.append("pytest tests/test_translation_validation.py tests/test_solver.py")
        else:
            commands.append("pytest tests/test_sagspec.py tests/test_sagspec_lowering.py")
        commands.append("./venv/bin/saguaro verify . --engines native,ruff,semantic --format json")
        for path in target_files:
            stem = path.rsplit("/", 1)[-1].split(".", 1)[0]
            tests.append(f"tests/test_{stem}.py")
        return SpecVerification(commands=commands, tests=tests, proofs_required=True)

    def _constraints(self, objective: str, language: str) -> list[SpecConstraint]:
        scalar = "double" if language == "cpp" else "float"
        constraints = [
            SpecConstraint(
                kind="type_contract",
                expression=f"inputs -> {scalar}, outputs -> {scalar}",
                description="Scalar numeric lane only",
            ),
            SpecConstraint(
                kind="verification_path",
                expression="must_use_sanctioned_verification",
                description="All syntheses must route through repository verification",
            ),
        ]
        lowered = objective.lower()
        if any(token in lowered for token in {"normalize", "bound", "clamp"}):
            constraints.append(
                SpecConstraint(
                    kind="range_safety",
                    expression="output must remain within declared bounds",
                    description="Bounded numeric helper",
                )
            )
        return constraints

    def _evidence_refs(self, target_files: list[str]) -> list[SpecEvidenceRef]:
        refs = [SpecEvidenceRef(kind="target_file", ref=path) for path in target_files]
        refs.append(
            SpecEvidenceRef(
                kind="verification_command",
                ref="./venv/bin/saguaro verify . --engines native,ruff,semantic --format json",
            )
        )
        return refs

    def _title(self, objective: str) -> str:
        text = " ".join(word.capitalize() for word in self._objective_tokens(objective)[:6])
        return text or "Deterministic Synthesis Task"

    def _change_summary(self, objective: str) -> str:
        return objective.strip() or "Implement bounded deterministic synthesis task"

    def _objective_tokens(self, objective: str) -> list[str]:
        return [token for token in re.findall(r"[A-Za-z0-9_]+", objective.lower()) if token]

    def _markdown_goal(self, markdown_text: str) -> str:
        match = re.search(
            r"^##\s+Goal\s*\n(.*?)(?:\n##|\Z)",
            markdown_text,
            re.MULTILINE | re.DOTALL,
        )
        if match:
            return " ".join(match.group(1).split()).strip()
        heading = re.search(r"^#\s+(.+)$", markdown_text, re.MULTILINE)
        return str(heading.group(1)).strip() if heading else ""

    def _markdown_title(self, markdown_text: str) -> str:
        match = re.search(r"^#\s+(.+)$", markdown_text, re.MULTILINE)
        if match:
            return str(match.group(1)).strip()
        return "Roadmap Synthesis Task"

    def _markdown_target_files(self, markdown_text: str) -> list[str]:
        files: list[str] = []
        for match in self._PATH_PATTERN.finditer(markdown_text):
            path = str(match.group(1) or match.group(2) or "").strip().strip(".,)")
            if (
                path
                and " " not in path
                and not path.startswith(("pytest", "./venv/bin/saguaro", "saguaro "))
                and path not in files
            ):
                files.append(path)
        return files

    def _markdown_verification(self, markdown_text: str) -> list[str]:
        commands = []
        for line in markdown_text.splitlines():
            stripped = line.strip()
            if stripped.startswith(("1.", "2.", "3.", "4.", "- `pytest", "- `./venv/bin/saguaro")):
                command = re.sub(r"^\d+\.\s*", "", stripped)
                command = command.lstrip("- ").strip("`")
                if "pytest " in command or "saguaro verify" in command:
                    commands.append(command)
        return commands
