from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class SynthesisMemoryRecord:
    cache_key: str
    spec_digest: str
    capability_digest: str
    repo_delta_id: str
    counterexamples: list[dict[str, Any]] = field(default_factory=list)
    proof_refs: list[str] = field(default_factory=list)
    assembly_refs: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class SynthesisCache:
    def __init__(self) -> None:
        self._records: dict[str, SynthesisMemoryRecord] = {}
        self._hits = 0

    @staticmethod
    def cache_key(
        spec_digest: str,
        *,
        capability_digest: str,
        repo_delta_id: str,
    ) -> str:
        payload = f"{spec_digest}|{capability_digest}|{repo_delta_id}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def put(
        self,
        *,
        spec_digest: str,
        capability_digest: str,
        repo_delta_id: str,
        counterexamples: list[dict[str, Any]] | None = None,
        proof_refs: list[str] | None = None,
        assembly_refs: list[str] | None = None,
    ) -> SynthesisMemoryRecord:
        key = self.cache_key(
            spec_digest,
            capability_digest=capability_digest,
            repo_delta_id=repo_delta_id,
        )
        record = SynthesisMemoryRecord(
            cache_key=key,
            spec_digest=spec_digest,
            capability_digest=capability_digest,
            repo_delta_id=repo_delta_id,
            counterexamples=[dict(item) for item in list(counterexamples or [])],
            proof_refs=[str(item) for item in list(proof_refs or [])],
            assembly_refs=[str(item) for item in list(assembly_refs or [])],
        )
        self._records[key] = record
        return record

    def get(
        self,
        *,
        spec_digest: str,
        capability_digest: str,
        repo_delta_id: str,
    ) -> SynthesisMemoryRecord | None:
        key = self.cache_key(
            spec_digest,
            capability_digest=capability_digest,
            repo_delta_id=repo_delta_id,
        )
        record = self._records.get(key)
        if record is not None:
            self._hits += 1
        return record

    def stats(self) -> dict[str, Any]:
        total = max(1, len(self._records))
        return {
            "cache_hit_ratio": round(self._hits / total, 3),
            "counterexample_cache_hits": sum(
                1 for item in self._records.values() if item.counterexamples
            ),
            "proof_reuse_hits": sum(1 for item in self._records.values() if item.proof_refs),
            "record_count": len(self._records),
        }

