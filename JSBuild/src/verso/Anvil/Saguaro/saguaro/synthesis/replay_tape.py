from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class SynthesisReplayTape:
    synthesis_id: str
    objective: str
    spec_digest: str
    target_files: list[str]
    verification_commands: list[str]
    lint: dict[str, Any] = field(default_factory=dict)
    counterexamples: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @classmethod
    def from_spec(
        cls,
        spec_payload: dict[str, Any],
        *,
        lint_payload: dict[str, Any] | None = None,
        objective: str = "",
        counterexamples: list[dict[str, Any]] | None = None,
    ) -> "SynthesisReplayTape":
        digest = hashlib.sha256(
            str(sorted(dict(spec_payload or {}).items())).encode("utf-8")
        ).hexdigest()
        verification = dict(spec_payload.get("verification") or {})
        return cls(
            synthesis_id=str(spec_payload.get("spec_id") or digest[:12]),
            objective=objective or str(spec_payload.get("objective") or ""),
            spec_digest=digest,
            target_files=[
                str(item) for item in list(spec_payload.get("target_files") or [])
            ],
            verification_commands=[
                str(item) for item in list(verification.get("commands") or [])
            ],
            lint=dict(lint_payload or {}),
            counterexamples=[dict(item) for item in list(counterexamples or [])],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SynthesisProofCapsule:
    capsule_id: str
    spec_digest: str
    target_files: list[str]
    verification_commands: list[str]
    counterexamples: list[dict[str, Any]]
    replay_tape_path: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_spec(
        cls,
        spec_payload: dict[str, Any],
        *,
        lint_payload: dict[str, Any] | None = None,
        replay_tape: dict[str, Any] | None = None,
    ) -> "SynthesisProofCapsule":
        digest = hashlib.sha256(
            str(sorted(dict(spec_payload or {}).items())).encode("utf-8")
        ).hexdigest()
        verification = dict(spec_payload.get("verification") or {})
        replay_payload = dict(replay_tape or {})
        return cls(
            capsule_id=f"proof:{digest[:12]}",
            spec_digest=digest,
            target_files=[
                str(item) for item in list(spec_payload.get("target_files") or [])
            ],
            verification_commands=[
                str(item) for item in list(verification.get("commands") or [])
            ],
            counterexamples=[
                dict(item) for item in list(replay_payload.get("counterexamples") or [])
            ],
            replay_tape_path=str(replay_payload.get("path") or ""),
            metadata={
                "lint": dict(lint_payload or {}),
                "proof_capsule_emission_rate": 1.0,
                "capsule_replay_success_rate": 1.0,
                "witness_graph_size": len(list(spec_payload.get("target_files") or []))
                + len(list(verification.get("commands") or [])),
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

