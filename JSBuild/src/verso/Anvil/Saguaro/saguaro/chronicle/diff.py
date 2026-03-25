"""Chronicle Diff: Semantic Drift Calculation."""

import logging
import os
from typing import Any

logger = logging.getLogger("saguaro.chronicle.diff")

# Suppress TensorFlow C++ chatter in CLI contexts unless overridden.
os.environ.setdefault(
    "TF_CPP_MIN_LOG_LEVEL",
    os.getenv("SAGUARO_TF_CPP_MIN_LOG_LEVEL", "3"),
)

try:
    import numpy as np
except ImportError:
    np = None
    logger.warning("Numpy not found. SemanticDiff running in degraded mode.")

try:
    from saguaro.ops.holographic import deserialize_hd_state
except Exception:
    deserialize_hd_state = None


class SemanticDiff:
    """Calculates the 'Semantic Drift' or distance between two Time Crystal states.
    Uses cosine similarity on the hyperdimensional bundles.
    """

    @staticmethod
    def _looks_like_tensorproto(blob: bytes) -> bool:
        """Best-effort sniff for TensorProto payloads."""
        if len(blob) < 2:
            return False
        if blob[0] != 0x08:
            return False
        header = blob[:64]
        return (b"\x12" in header) or (b"\x22" in header)

    @staticmethod
    def _vector_is_usable(vec: Any) -> bool:
        if np is None:
            return False
        arr = np.asarray(vec)
        if arr.size == 0:
            return False
        if not np.isfinite(arr).all():
            return False
        # Guard against protobuf bytes interpreted as float vectors.
        return float(np.max(np.abs(arr))) < 1e8

    @staticmethod
    def _decode_projection_blob(
        blob: bytes,
    ) -> tuple[dict[str, tuple[str, bool, int]] | None, dict[str, Any]]:
        """Decode ledger-backed projection blobs emitted by the Chronicle snapshot path."""
        if not blob:
            return None, {"status": "indeterminate", "reason": "empty_blob"}

        try:
            text = blob.decode("utf-8")
        except UnicodeDecodeError:
            return None, {"status": "indeterminate", "reason": "utf8_decode_failed"}

        if "\x00" in text:
            return None, {"status": "indeterminate", "reason": "binary_projection_blob"}

        records: dict[str, tuple[str, bool, int]] = {}
        malformed = 0
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = line.rsplit(":", 3)
            if len(parts) != 4:
                malformed += 1
                continue
            path, hash_value, deleted_flag, logical_clock_text = parts
            if not path:
                malformed += 1
                continue
            try:
                logical_clock = int(logical_clock_text)
            except ValueError:
                malformed += 1
                continue
            deleted = str(deleted_flag).strip().lower() in {"1", "true", "yes"}
            records[path] = (hash_value, deleted, logical_clock)

        if not records:
            return None, {
                "status": "indeterminate",
                "reason": "projection_parse_failed",
                "malformed_lines": malformed,
            }

        return records, {
            "status": "ok",
            "decode_mode": "projection",
            "line_count": len(records),
            "malformed_lines": malformed,
        }

    @staticmethod
    def _calculate_projection_drift(
        state_a: dict[str, tuple[str, bool, int]],
        state_b: dict[str, tuple[str, bool, int]],
        *,
        meta_a: dict[str, Any],
        meta_b: dict[str, Any],
    ) -> tuple[float, dict[str, Any]]:
        paths = sorted(set(state_a) | set(state_b))
        if not paths:
            return 0.0, {
                "status": "ok",
                "comparison_mode": "projection",
                "decode_mode": "projection",
                "path_count": 0,
                "similarity": 1.0,
                "state_a": meta_a,
                "state_b": meta_b,
            }

        added = 0
        removed = 0
        changed = 0
        unchanged = 0
        deleted_flip = 0

        for path in paths:
            left = state_a.get(path)
            right = state_b.get(path)
            if left is None:
                added += 1
                continue
            if right is None:
                removed += 1
                continue
            if left == right:
                unchanged += 1
                continue
            changed += 1
            if left[1] != right[1]:
                deleted_flip += 1

        divergence = added + removed + changed
        drift = divergence / len(paths)
        similarity = 1.0 - drift
        return float(drift), {
            "status": "ok",
            "comparison_mode": "projection",
            "decode_mode": "projection",
            "path_count": len(paths),
            "unchanged_count": unchanged,
            "added_count": added,
            "removed_count": removed,
            "changed_count": changed,
            "deleted_flip_count": deleted_flip,
            "similarity": float(similarity),
            "state_a": meta_a,
            "state_b": meta_b,
        }

    @staticmethod
    def _decode_blob(
        blob: bytes,
        dtype: Any,
        *,
        mode: str = "auto",
    ) -> tuple[Any, dict[str, Any]]:
        if np is None:
            return None, {"status": "indeterminate", "reason": "numpy_unavailable"}

        if not blob:
            return np.zeros((0,), dtype=dtype), {
                "status": "indeterminate",
                "reason": "empty_blob",
            }

        decode_mode = str(mode or "auto").strip().lower()
        if decode_mode not in {"auto", "raw", "tensorproto"}:
            decode_mode = "auto"

        itemsize = np.dtype(dtype).itemsize
        is_aligned = len(blob) % itemsize == 0
        decode_errors: list[str] = []

        # Fast/default path for current native/NumPy-backed state serialization.
        if decode_mode in {"auto", "raw"}:
            should_try_raw = is_aligned and (
                decode_mode == "raw" or not SemanticDiff._looks_like_tensorproto(blob)
            )
            if not should_try_raw:
                if decode_mode == "raw":
                    return np.zeros((0,), dtype=dtype), {
                        "status": "indeterminate",
                        "reason": "invalid_blob_size",
                        "blob_size": len(blob),
                    }
            else:
                try:
                    vec = np.frombuffer(blob, dtype=dtype).reshape(-1)
                    if SemanticDiff._vector_is_usable(vec):
                        return vec, {"status": "ok", "decode_mode": "raw"}
                    decode_errors.append("raw:unusable_vector")
                except Exception as exc:
                    decode_errors.append(f"raw:{exc}")
                if decode_mode == "raw":
                    return np.zeros((0,), dtype=dtype), {
                        "status": "indeterminate",
                        "reason": "raw_decode_invalid",
                        "blob_size": len(blob),
                        "decode_errors": decode_errors,
                    }

        # Compatibility path for legacy TensorProto snapshots.
        if decode_mode in {"auto", "tensorproto"}:
            if deserialize_hd_state is not None:
                try:
                    decoded = deserialize_hd_state(
                        blob,
                        dtype=None,
                        allow_tensorflow_parse=True,
                    )
                    arr = (
                        decoded.numpy() if hasattr(decoded, "numpy") else np.asarray(decoded)
                    )
                    arr = arr.astype(dtype, copy=False).reshape(-1)
                    if SemanticDiff._vector_is_usable(arr):
                        return arr, {"status": "ok", "decode_mode": "tensorproto"}
                    decode_errors.append("tensorproto:unusable_vector")
                except Exception as exc:
                    decode_errors.append(f"tensorproto:{exc}")

            if decode_mode == "tensorproto":
                return np.zeros((0,), dtype=dtype), {
                    "status": "indeterminate",
                    "reason": "tensorproto_decode_failed",
                    "blob_size": len(blob),
                    "decode_errors": decode_errors,
                }

        # Final fallback: if auto mode skipped raw due proto sniff, still allow
        # aligned raw decode as a last resort.
        if decode_mode == "auto" and is_aligned:
            try:
                vec = np.frombuffer(blob, dtype=dtype).reshape(-1)
                if SemanticDiff._vector_is_usable(vec):
                    return vec, {"status": "ok", "decode_mode": "raw_fallback"}
                decode_errors.append("raw_fallback:unusable_vector")
            except Exception as exc:
                decode_errors.append(f"raw_fallback:{exc}")

        reason = "invalid_blob_size" if not is_aligned else "decode_failed"
        return np.zeros((0,), dtype=dtype), {
            "status": "indeterminate",
            "reason": reason,
            "blob_size": len(blob),
            "decode_errors": decode_errors,
        }

    @staticmethod
    def calculate_drift(
        state_a_blob: bytes,
        state_b_blob: bytes,
        dtype: Any = None,
        *,
        decode_mode: str = "auto",
        fail_open_on_decode_error: bool = True,
    ) -> tuple[float, dict[str, Any]]:
        """Calculate semantic drift between two serialized HD states.

        Args:
            state_a_blob: Bytes of first state tensor
            state_b_blob: Bytes of second state tensor
            dtype: Numpy dtype of the blobs

        Returns:
            (drift_score, details)
            drift_score is 0.0 to 1.0 (0.0 = identical, 1.0 = orthogonal/opposite)
        """
        try:
            if np is None:
                logger.warning("Cannot calculate drift without numpy")
                return 0.0, {"status": "indeterminate", "reason": "numpy_unavailable"}

            if dtype is None:
                dtype = np.float32

            if not state_a_blob or not state_b_blob:
                logger.warning("Empty state blob provided for drift calculation")
                return 0.0, {
                    "status": "indeterminate",
                    "reason": "empty_state_blob",
                }

            if decode_mode == "auto":
                projection_a, projection_meta_a = SemanticDiff._decode_projection_blob(
                    state_a_blob
                )
                projection_b, projection_meta_b = SemanticDiff._decode_projection_blob(
                    state_b_blob
                )
                if (
                    projection_meta_a.get("status") == "ok"
                    and projection_meta_b.get("status") == "ok"
                    and projection_a is not None
                    and projection_b is not None
                ):
                    return SemanticDiff._calculate_projection_drift(
                        projection_a,
                        projection_b,
                        meta_a=projection_meta_a,
                        meta_b=projection_meta_b,
                    )

            vec_a, meta_a = SemanticDiff._decode_blob(
                state_a_blob,
                dtype,
                mode=decode_mode,
            )
            vec_b, meta_b = SemanticDiff._decode_blob(
                state_b_blob,
                dtype,
                mode=decode_mode,
            )
            if (
                meta_a.get("status") != "ok"
                or meta_b.get("status") != "ok"
                or vec_a is None
                or vec_b is None
                or vec_a.size == 0
                or vec_b.size == 0
            ):
                details = {
                    "status": "indeterminate",
                    "reason": "decode_state_failure",
                    "decode_mode": decode_mode,
                    "state_a": meta_a,
                    "state_b": meta_b,
                }
                if fail_open_on_decode_error:
                    return 0.0, details
                return 1.0, {"error": "invalid_blob_size", **details}

            if vec_a.shape != vec_b.shape:
                logger.warning(
                    "Shape mismatch for semantic drift comparison: %s vs %s",
                    vec_a.shape,
                    vec_b.shape,
                )
                details = {
                    "status": "indeterminate",
                    "reason": "incompatible_state_shape",
                    "shape_a": tuple(int(x) for x in vec_a.shape),
                    "shape_b": tuple(int(x) for x in vec_b.shape),
                    "decode_mode": decode_mode,
                    "state_a": meta_a,
                    "state_b": meta_b,
                }
                if fail_open_on_decode_error:
                    return 0.0, details
                return 1.0, {"error": "incompatible_state_shape", **details}

            norm_a = np.linalg.norm(vec_a)
            norm_b = np.linalg.norm(vec_b)
            if norm_a == 0 or norm_b == 0:
                details = {
                    "status": "indeterminate",
                    "reason": "zero_norm_state",
                    "decode_mode": decode_mode,
                    "state_a": meta_a,
                    "state_b": meta_b,
                    "norm_a": float(norm_a),
                    "norm_b": float(norm_b),
                }
                if fail_open_on_decode_error:
                    return 0.0, details
                return 1.0, {"error": "zero_norm_state", **details}

            similarity = np.dot(vec_a, vec_b) / (norm_a * norm_b)
            similarity = float(np.clip(similarity, -1.0, 1.0))
            drift = 1.0 - similarity

            return float(drift), {
                "status": "ok",
                "similarity": float(similarity),
                "norm_a": float(norm_a),
                "norm_b": float(norm_b),
                "decode_mode": str(decode_mode or "auto"),
                "state_a": meta_a,
                "state_b": meta_b,
            }

        except Exception as e:
            logger.error(f"Error calculating drift: {e}")
            return 0.0, {
                "status": "indeterminate",
                "reason": "drift_exception",
                "error": str(e),
            }

    @staticmethod
    def human_readable_report(drift_score: float) -> str:
        """Handle human readable report."""
        if drift_score < 0.01:
            return "Stable (No significant semantic range)"
        if drift_score < 0.1:
            return "Minor Drift (Refactoring or small tweaks)"
        if drift_score < 0.4:
            return "Moderate Drift (Feature addition or logic change)"
        return "Major Shift (Architectural change or rewrite)"
