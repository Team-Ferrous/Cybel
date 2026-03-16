import numpy as np
import logging
import hashlib
import time
from typing import Dict, Any
import uuid
from .backend_interface import CoconutBackend

logger = logging.getLogger(__name__)


class NativeBackend(CoconutBackend):
    """
    CPU-optimized backend using the C++ native bridge.
    Uses SIMD instructions (AVX2/AVX512) for high performance.
    """

    def __init__(self, **config):
        self.dim = config.get("embedding_dim", 4096)
        self.num_paths = config.get("num_paths", 4)
        self.steps = config.get("steps", 2)

        # Initialize the native bridge - explicitly force CPU mode to avoid recursion
        from core.native.coconut_bridge import CoconutNativeBridge

        self.bridge = CoconutNativeBridge(embedding_dim=self.dim, use_gpu=False)

        if self.bridge.use_gpu:
            logger.warning(
                "NativeBackend initialized with GPU enabled bridge. This is unusual."
            )

        self.last_amplitudes = None
        self.last_packet: Dict[str, Any] | None = None
        self.last_session_record: Dict[str, Any] | None = None

        # We need weights for evolution.
        # In this backend, we'll use randomized weights if they aren't provided,
        # but in practice these should come from the model or a pre-trained state.
        # For the "thinking" block, we use a simple evolution.
        self.norm_gamma = np.ones(self.dim, dtype=np.float32)
        self.norm_beta = np.zeros(self.dim, dtype=np.float32)

        hidden_dim = self.dim * 4
        self.w1 = np.random.normal(0, 0.02, (self.dim, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.w2 = np.random.normal(0, 0.02, (hidden_dim, self.dim)).astype(np.float32)
        self.b2 = np.zeros(self.dim, dtype=np.float32)

    def explore(self, context_embedding: np.ndarray) -> np.ndarray:
        """Execute COCONUT BFS via C++ native bridge."""
        # Ensure [Batch, Dim] shape
        if len(context_embedding.shape) == 1:
            context_embedding = context_embedding[None, :]
        elif len(context_embedding.shape) == 3:
            # [Batch, Seq, Dim] -> [Batch, Dim]
            context_embedding = np.mean(context_embedding, axis=1)

        context_embedding.shape[0]

        # 1. Expand
        paths = self.bridge.expand_paths(context_embedding, self.num_paths)

        # 2. Evolve
        for _ in range(self.steps):
            paths = self.bridge.evolve_paths(
                paths,
                self.norm_gamma,
                self.norm_beta,
                self.w1,
                self.b1,
                self.w2,
                self.b2,
                hidden_dim=self.dim * 4,
            )

        # 3. Score
        amplitudes = self.bridge.score_paths(paths, context_embedding)
        self.last_amplitudes = amplitudes[0] if amplitudes.size > 0 else None

        # 4. Aggregate
        output = self.bridge.aggregate_paths(paths, amplitudes)
        session_id = f"coconut-{uuid.uuid4().hex}"
        norm_summary = {
            "input_l2_mean": float(np.linalg.norm(context_embedding, axis=-1).mean()),
            "output_l2_mean": float(np.linalg.norm(output, axis=-1).mean()),
            "amplitude_mean": (
                float(np.asarray(amplitudes, dtype=np.float32).mean())
                if amplitudes.size > 0
                else 0.0
            ),
        }
        checksum = hashlib.sha256(
            output.astype(np.float32, copy=False).tobytes()
        ).hexdigest()
        self.last_packet = {
            "packet_version": 1,
            "sequence_id": session_id,
            "checkpoint_id": "",
            "hidden_dimension": int(output.shape[-1]) if output.ndim else int(self.dim),
            "norm_summary": norm_summary,
            "checksum": checksum,
            "branch_score": (
                float(np.asarray(amplitudes, dtype=np.float32).max())
                if amplitudes.size > 0
                else 0.0
            ),
            "stop_policy": "coconut_native_steps",
            "created_at_ns": time.time_ns(),
        }
        self.last_session_record = {
            "sequence_id": session_id,
            "num_paths": int(self.num_paths),
            "steps": int(self.steps),
            "input_shape": [int(dim) for dim in context_embedding.shape],
            "path_shape": [int(dim) for dim in paths.shape],
            "output_shape": [int(dim) for dim in output.shape],
            "amplitudes": (
                np.asarray(amplitudes, dtype=np.float32).reshape(-1).tolist()
                if amplitudes.size > 0
                else []
            ),
            "packet": dict(self.last_packet),
        }

        return output

    def get_device_info(self) -> Dict[str, Any]:
        has_lib = hasattr(self.bridge, "lib")
        return {
            "backend": "native",
            "device": "CPU (SIMD)" if has_lib else "CPU (Fallback)",
            "num_paths": self.num_paths,
            "thought_steps": self.steps,
            "has_native_lib": has_lib,
        }

    @property
    def is_available(self) -> bool:
        # It's always available as it can fall back to NumPy if lib is missing,
        # but we specifically want this when the native lib IS present.
        return True

    def get_last_session_record(self) -> Dict[str, Any] | None:
        if self.last_session_record is None:
            return None
        return dict(self.last_session_record)
