import time
from typing import Any, Dict, List, Optional, Sequence

from core.memory.fabric.models import LatentPackageRecord, RepoDeltaMemoryRecord
from core.native.qsg_state_kernels_wrapper import qsg_state_weighted_merge
from core.qsg.runtime_contracts import DeltaWatermark
from core.qsg.runtime_contracts import TypedLatentSegment
import numpy as np


class LatentThought:
    def __init__(
        self,
        type: str,
        content: str,
        vector: Optional[Sequence[float]] = None,
    ):
        self.type = type
        self.content = content
        self.timestamp = time.time()
        self.vector = self._coerce_vector(vector)

    @staticmethod
    def _coerce_vector(vector: Optional[Sequence[float]]) -> Optional[List[float]]:
        if vector is None:
            return None
        try:
            values = [float(value) for value in vector]
        except Exception:
            return None
        if not values:
            return None
        for value in values:
            if value != value or value in (float("inf"), float("-inf")):
                return None
        return values


class LatentMemory:
    """
    Persists thinking traces for context enrichment and post-mortem analysis.
    """

    def __init__(self, max_size: int = 20):
        self.thoughts: List[LatentThought] = []
        self.max_size = max_size

    def add_thought(
        self,
        type: str,
        content: str,
        vector: Optional[Sequence[float]] = None,
    ):
        self.thoughts.append(LatentThought(type, content, vector=vector))
        if len(self.thoughts) > self.max_size:
            self.thoughts.pop(0)

    def add_state(
        self, type: str, content: str, vector: Optional[Sequence[float]]
    ) -> None:
        self.add_thought(type=type, content=content, vector=vector)

    def get_recent_vectors(self, limit: int = 4) -> List[List[float]]:
        vectors: List[List[float]] = []
        for thought in reversed(self.thoughts):
            if thought.vector is not None:
                vectors.append(thought.vector)
            if len(vectors) >= max(1, int(limit)):
                break
        vectors.reverse()
        return vectors

    def get_merged_vector(self, limit: int = 4) -> Optional[List[float]]:
        vectors = self.get_recent_vectors(limit=limit)
        if not vectors:
            return None
        if len(vectors) == 1:
            merged = list(vectors[0])
        else:
            width = len(vectors[0])
            if width <= 0:
                return None
            merged = []
            for idx in range(width):
                merged.append(
                    sum(float(vector[idx]) for vector in vectors) / float(len(vectors))
                )
        if not merged:
            return None
        for value in merged:
            if value != value or value in (float("inf"), float("-inf")):
                return None
        return [float(v) for v in merged]

    def get_summary(self) -> str:
        """Returns a summarized map of reasoning traces."""
        if not self.thoughts:
            return "No previous reasoning traces."

        summary = "LATENT REASONING RECAP:\n"
        for t in self.thoughts:
            summary += f"- [{t.type}] {t.content[:100]}...\n"
        return summary

    def get_context_prompt(self) -> str:
        if not self.thoughts:
            return ""
        return f"\n### PREVIOUS REASONING TRACES\nThis summarizes why previous decisions were made:\n{self.get_summary()}"

    def export_state(self, limit: int = 4) -> Dict[str, Any]:
        return {
            "summary": self.get_summary(),
            "vectors": self.get_recent_vectors(limit=limit),
            "merged_vector": self.get_merged_vector(limit=limit),
            "thought_count": len(self.thoughts),
        }

    def build_segments(self) -> tuple[List[TypedLatentSegment], list[list[float]]]:
        vectors = self.get_recent_vectors(limit=self.max_size)
        if not vectors:
            vectors = [[0.0]]
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)

        row_count = int(matrix.shape[0])
        hidden_dim = int(matrix.shape[1])
        recent_weights = np.linspace(1.0, 2.0, num=row_count, dtype=np.float32)
        recent_weights /= float(np.sum(recent_weights) or 1.0)
        session_weights = np.linspace(
            float(row_count), 1.0, num=row_count, dtype=np.float32
        )
        session_weights /= float(np.sum(session_weights) or 1.0)
        summary_weights = np.full((row_count,), 1.0, dtype=np.float32)
        summary_weights /= float(np.sum(summary_weights) or 1.0)

        tensor = [
            [
                float(value)
                for value in qsg_state_weighted_merge(matrix, recent_weights)
            ],
            [
                float(value)
                for value in qsg_state_weighted_merge(matrix, summary_weights)
            ],
            [
                float(value)
                for value in qsg_state_weighted_merge(matrix, session_weights)
            ],
        ]
        segments = [
            TypedLatentSegment(
                segment_id="micro_branch",
                segment_kind="branch_state",
                row_start=0,
                row_count=1,
                hidden_dim=hidden_dim,
                codec="float16",
                importance=0.95,
                metadata={"resolution": "micro"},
            ),
            TypedLatentSegment(
                segment_id="meso_summary",
                segment_kind="summary_state",
                row_start=1,
                row_count=1,
                hidden_dim=hidden_dim,
                codec="float16",
                importance=0.8,
                metadata={"resolution": "meso"},
            ),
            TypedLatentSegment(
                segment_id="macro_session",
                segment_kind="session_state",
                row_start=2,
                row_count=1,
                hidden_dim=hidden_dim,
                codec="float16",
                importance=0.6,
                metadata={"resolution": "macro"},
            ),
        ]
        return segments, tensor

    def build_package(
        self,
        *,
        memory_id: str,
        capture_stage: str,
        summary_text: str = "",
        model_family: str = "qsg-python",
        model_revision: str = "v1",
        prompt_protocol_hash: str = "almf.v1",
        supporting_memory_ids: Optional[List[str]] = None,
        creation_reason: str = "",
        expiry_seconds: float = 3600.0,
        capability_digest: str = "",
        delta_watermark: Dict[str, Any] | None = None,
        execution_capsule_id: str = "",
    ) -> tuple[LatentPackageRecord, list[list[float]]]:
        segments, tensor = self.build_segments()
        now = time.time()
        normalized_delta = DeltaWatermark.from_dict(delta_watermark)
        repo_delta_memory = RepoDeltaMemoryRecord.from_delta_watermark(
            normalized_delta.as_dict(),
            capability_digest=capability_digest,
            summary_text=summary_text or self.get_summary(),
            source_stage=capture_stage,
        )
        package = LatentPackageRecord(
            latent_package_id=f"latent_{int(now * 1000)}",
            memory_id=memory_id,
            branch_id=f"{memory_id}:{capture_stage}",
            model_family=model_family,
            model_revision=model_revision,
            tokenizer_hash="tokenizer:unknown",
            adapter_hash="adapter:none",
            prompt_protocol_hash=prompt_protocol_hash,
            hidden_dim=len(tensor[0]),
            qsg_runtime_version="qsg.v1",
            quantization_profile="float16",
            capture_stage=capture_stage,
            summary_text=summary_text or self.get_summary(),
            latent_packet_abi_version=3,
            execution_capsule_version=3,
            execution_capsule_id=str(execution_capsule_id),
            capability_digest=str(capability_digest),
            delta_watermark_json=normalized_delta.as_dict(),
            compatibility_json={
                "model_family": model_family,
                "hidden_dim": len(tensor[0]),
                "tokenizer_hash": "tokenizer:unknown",
                "prompt_protocol_hash": prompt_protocol_hash,
                "qsg_runtime_version": "qsg.v1",
                "latent_packet_abi_version": 3,
                "execution_capsule_version": 3,
                "execution_capsule_id": str(execution_capsule_id),
                "capability_digest": str(capability_digest),
                "delta_watermark": normalized_delta.as_dict(),
                "tensor_codec": "float16",
                "segments": [segment.as_dict() for segment in segments],
                "segment_count": len(segments),
                "repo_delta_memory": repo_delta_memory.as_dict(),
            },
            supporting_memory_ids=list(supporting_memory_ids or []),
            creation_reason=creation_reason,
            created_at=now,
            expires_at=now + max(60.0, float(expiry_seconds)),
        )
        return package, tensor
