"""Utilities for semantic."""

import json
import logging
import os
from typing import Any

from saguaro.indexing.auto_scaler import get_repo_stats_and_config
from saguaro.indexing.engine import IndexEngine

from ...chronicle.diff import SemanticDiff
from ...chronicle.storage import ChronicleStorage
from ...state.ledger import StateLedger
from .base import BaseEngine

logger = logging.getLogger(__name__)


class SemanticEngine(BaseEngine):
    """Checks for semantic drift and other quantum/holographic violations."""

    def __init__(self, repo_path: str) -> None:
        """Initialize the instance."""
        super().__init__(repo_path)
        self.storage = ChronicleStorage()
        self.drift_threshold = 0.2
        self.drift_enabled = True
        self.decode_mode = "auto"
        self.fail_open_on_decode_error = True

        # Initialize tokenizer coherence when optional deps are present.
        self.coherence = None
        try:
            from saguaro.tokenization.vocab import CoherenceManager

            self.coherence = CoherenceManager()
            self.coherence.initialize()
        except Exception as e:
            logger.warning(
                f"Semantic coherence unavailable; continuing without it: {e}"
            )

    def set_policy(self, config: dict[str, Any]) -> None:
        """Set policy."""
        super().set_policy(config)
        self.drift_threshold = float(config.get("drift_tolerance", 0.2))
        self.drift_enabled = bool(config.get("semantic_drift_enabled", True))
        decode_mode = str(config.get("semantic_decode_mode", "auto")).strip().lower()
        if decode_mode not in {"auto", "raw", "tensorproto"}:
            decode_mode = "auto"
        self.decode_mode = decode_mode
        self.fail_open_on_decode_error = bool(
            config.get("semantic_fail_open_on_decode_error", True)
        )

    @staticmethod
    def _is_tensorflow_parse_error(value: Any) -> bool:
        message = str(value or "").lower()
        markers = (
            "parse tensor",
            "tensor proto",
            "tensorproto",
            "parsefromstring",
        )
        return any(marker in message for marker in markers)

    @classmethod
    def _is_indeterminate_drift(cls, details: Any) -> bool:
        if not isinstance(details, dict):
            return False

        if str(details.get("status", "")).lower() == "indeterminate":
            return True

        warning = str(details.get("warning", "")).lower()
        reason = str(details.get("reason", "")).lower()
        if "indeterminate" in warning or "indeterminate" in reason:
            return True

        state_a = details.get("state_a")
        state_b = details.get("state_b")
        if isinstance(state_a, dict) and str(state_a.get("status", "")).lower() == "indeterminate":
            return True
        if isinstance(state_b, dict) and str(state_b.get("status", "")).lower() == "indeterminate":
            return True

        degraded_reasons = {
            "indeterminate_state",
            "decode_failed",
            "decode_state_failure",
            "empty_state",
            "empty_state_blob",
            "invalid_blob_size",
            "parse_error",
        }
        if warning in degraded_reasons or reason in degraded_reasons:
            return True

        return cls._is_tensorflow_parse_error(details.get("error"))

    def run(self, path_arg: str = ".") -> list[dict[str, Any]]:
        """Handle run."""
        _ = path_arg
        logger.info("Running SemanticEngine...")
        violations = []

        if not self.drift_enabled:
            logger.info("Semantic drift checking disabled by policy.")
            return violations

        # 1. Drift Detection
        latest_snapshot = self.storage.get_latest_snapshot()

        if not latest_snapshot:
            # No baseline, so can't calculate drift.
            logger.info("No semantic baseline found. Skipping drift check.")
            return violations

        # Calculate Real State from current codebase
        current_state_blob = self._calculate_current_state()

        try:
            drift, details = SemanticDiff.calculate_drift(
                latest_snapshot["hd_state_blob"],
                current_state_blob,
                decode_mode=self.decode_mode,
                fail_open_on_decode_error=self.fail_open_on_decode_error,
            )
        except Exception as e:
            if self._is_tensorflow_parse_error(e):
                logger.warning(
                    "Semantic drift parse failure detected; skipping SEMANTIC-DRIFT "
                    "violation. error=%s",
                    e,
                )
                return violations
            raise

        if self._is_indeterminate_drift(details):
            logger.warning(
                "Semantic drift is indeterminate; skipping SEMANTIC-DRIFT violation. "
                "reason=%s",
                details.get("reason", details.get("warning", "unknown")),
            )
            return violations

        logger.info(f"Semantic Drift: {drift:.4f}")

        if drift > self.drift_threshold:
            violations.append(
                {
                    "file": "chronicle.db",  # logical file
                    "line": 0,
                    "rule_id": "SEMANTIC-DRIFT",
                    "message": f"Semantic drift {drift:.2f} exceeds threshold {self.drift_threshold}. {SemanticDiff.human_readable_report(drift)}",
                    "severity": "warning",
                    "aal": "AAL-2",
                    "domain": ["universal"],
                    "closure_level": "guarded",
                    "evidence_refs": [],
                    "context": f"Baseline: Snapshot #{latest_snapshot['id']} ({latest_snapshot['description']})",
                }
            )

        return violations

    def _calculate_current_state(self) -> bytes:
        """Compute the holographic state of the current working directory.
        Uses IndexEngine logic in-memory (no disk write).
        """
        root_dir = self.repo_path
        saguaro_dir = os.path.join(root_dir, ".saguaro")

        if not os.path.exists(saguaro_dir):
            return b""  # Empty state if not init

        try:
            projection_blob = StateLedger(root_dir).state_projection_blob()
            if projection_blob:
                return projection_blob
        except Exception as e:
            logger.debug("Unable to build ledger projection state: %s", e)

        try:
            # Start from autoscaled defaults, then pin to persisted index dimensions.
            stats = get_repo_stats_and_config(root_dir)

            config_path = os.path.join(saguaro_dir, "config.yaml")
            if os.path.exists(config_path):
                try:
                    import yaml

                    with open(config_path, encoding="utf-8") as f:
                        persisted = yaml.safe_load(f) or {}
                    if "active_dim" in persisted:
                        stats["active_dim"] = int(persisted["active_dim"])
                    if "total_dim" in persisted:
                        stats["total_dim"] = int(persisted["total_dim"])
                except Exception as e:
                    logger.debug("Unable to load persisted semantic config: %s", e)

            # Vector-store metadata is the final source of truth for current index dim.
            index_meta_path = os.path.join(saguaro_dir, "vectors", "index_meta.json")
            if os.path.exists(index_meta_path):
                try:
                    with open(index_meta_path, encoding="utf-8") as f:
                        meta = json.load(f) or {}
                    if "dim" in meta:
                        stats["total_dim"] = int(meta["dim"])
                except Exception as e:
                    logger.debug("Unable to read vector-store metadata: %s", e)

            # Active dimensions cannot exceed total dimensions.
            stats["active_dim"] = min(
                int(stats.get("active_dim", 4096)),
                int(stats.get("total_dim", 8192)),
            )

            engine = IndexEngine(root_dir, saguaro_dir, stats)

            # Use the new optimized compute_state method
            # This handles scanning and processing without persistence
            return engine.compute_state()

        except Exception as e:
            logger.error(f"Failed to calculate current state: {e}")
            return b""
