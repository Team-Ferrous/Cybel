"""
Enhanced Thinking System - Structured thinking blocks with CoCoNut integration.

Implements:
- ThinkingBlock: Individual thinking units with types
- ThinkingChain: Accumulated thinking within a task
- EnhancedThinkingSystem: Full thinking pipeline with CoCoNut
"""

import logging
import re
import json
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from core.aes import AALClassifier, AESRuleRegistry, DomainDetector

logger = logging.getLogger(__name__)


class ThinkingType(Enum):
    """Types of structured thinking blocks."""

    UNDERSTANDING = "understanding"  # Comprehend the problem
    PLANNING = "planning"  # Design approach
    REASONING = "reasoning"  # Work through logic
    REFLECTION = "reflection"  # Evaluate progress
    CORRECTION = "correction"  # Fix mistakes
    COMPLIANCE = "compliance"  # AES compliance reasoning

    @property
    def color(self) -> str:
        """Rich-compatible color for rendering."""
        return {
            ThinkingType.UNDERSTANDING: "cyan",
            ThinkingType.PLANNING: "blue",
            ThinkingType.REASONING: "green",
            ThinkingType.REFLECTION: "yellow",
            ThinkingType.CORRECTION: "red",
            ThinkingType.COMPLIANCE: "magenta",
        }[self]

    @property
    def emoji(self) -> str:
        """Emoji representation for display."""
        return {
            ThinkingType.UNDERSTANDING: "🔍",
            ThinkingType.PLANNING: "📋",
            ThinkingType.REASONING: "💭",
            ThinkingType.REFLECTION: "🪞",
            ThinkingType.CORRECTION: "🔧",
            ThinkingType.COMPLIANCE: "🛡",
        }[self]


@dataclass
class ThinkingBlock:
    """Represents a structured thinking step."""

    type: ThinkingType
    content: str
    timestamp: datetime = field(default_factory=datetime.now)

    def render(self, show_timestamp: bool = False) -> str:
        """Render as formatted string."""
        ts = f" [{self.timestamp.strftime('%H:%M:%S')}]" if show_timestamp else ""
        return f"{self.type.emoji} [{self.type.value.upper()}]{ts}\n{self.content}"

    def to_xml(self) -> str:
        """Render as XML-style thinking block."""
        return f'<thinking type="{self.type.value}">\n{self.content}\n</thinking>'

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "type": self.type.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
        }


class ThinkingChain:
    """Accumulates thinking blocks within a task."""

    def __init__(
        self,
        task_id: str = "default",
        compliance_context: Optional[Dict[str, Any]] = None,
    ):
        self.task_id = task_id
        self.blocks: List[ThinkingBlock] = []
        self.started_at = datetime.now()
        self.compliance_context = compliance_context or {
            "trace_id": None,
            "evidence_bundle_id": None,
            "red_team_required": False,
            "waiver_ids": [],
            "waiver_id": None,
        }

    def add(self, type: ThinkingType, content: str) -> ThinkingBlock:
        """Add a new thinking block to the chain."""
        block = ThinkingBlock(type=type, content=content)
        self.blocks.append(block)
        return block

    def add_understanding(self, content: str) -> ThinkingBlock:
        """Shortcut for adding understanding block."""
        return self.add(ThinkingType.UNDERSTANDING, content)

    def add_planning(self, content: str) -> ThinkingBlock:
        """Shortcut for adding planning block."""
        return self.add(ThinkingType.PLANNING, content)

    def add_reasoning(self, content: str) -> ThinkingBlock:
        """Shortcut for adding reasoning block."""
        return self.add(ThinkingType.REASONING, content)

    def add_reflection(self, content: str) -> ThinkingBlock:
        """Shortcut for adding reflection block."""
        return self.add(ThinkingType.REFLECTION, content)

    def add_correction(self, content: str) -> ThinkingBlock:
        """Shortcut for adding correction block."""
        return self.add(ThinkingType.CORRECTION, content)

    def add_compliance(self, content: str) -> ThinkingBlock:
        """Shortcut for adding compliance block."""
        return self.add(ThinkingType.COMPLIANCE, content)

    def get_by_type(self, type: ThinkingType) -> List[ThinkingBlock]:
        """Get all blocks of a specific type."""
        return [b for b in self.blocks if b.type == type]

    def render(self, show_timestamps: bool = False) -> str:
        """Render the full chain."""
        if not self.blocks:
            return "No thinking recorded."
        return "\n\n".join(b.render(show_timestamps) for b in self.blocks)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "task_id": self.task_id,
            "started_at": self.started_at.isoformat(),
            "compliance_context": self.compliance_context,
            "blocks": [b.to_dict() for b in self.blocks],
        }

    def save(self, path: str) -> None:
        """Save to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ThinkingChain":
        """Load from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        chain = cls(
            task_id=data.get("task_id", "loaded"),
            compliance_context=data.get("compliance_context"),
        )
        chain.started_at = datetime.fromisoformat(data["started_at"])

        for block_data in data.get("blocks", []):
            block = ThinkingBlock(
                type=ThinkingType(block_data["type"]),
                content=block_data["content"],
                timestamp=datetime.fromisoformat(block_data["timestamp"]),
            )
            chain.blocks.append(block)

        return chain

    def __len__(self) -> int:
        return len(self.blocks)


class ThinkingParser:
    """Parses thinking blocks from model output."""

    # Pattern for structured thinking: <thinking type="...">...</thinking>
    STRUCTURED_PATTERN = re.compile(
        r'<thinking\s+type=["\'](\w+)["\']\s*>(.*?)</thinking>',
        re.DOTALL | re.IGNORECASE,
    )

    # Pattern for simple thinking: <thinking>...</thinking>
    SIMPLE_PATTERN = re.compile(
        r"<thinking>(.*?)</thinking>", re.DOTALL | re.IGNORECASE
    )

    @classmethod
    def parse(cls, text: str) -> List[ThinkingBlock]:
        """Extract all thinking blocks from text."""
        blocks = []

        # First try structured format
        for match in cls.STRUCTURED_PATTERN.finditer(text):
            type_str = match.group(1).lower()
            content = match.group(2).strip()

            try:
                thinking_type = ThinkingType(type_str)
            except ValueError:
                thinking_type = ThinkingType.REASONING  # Default

            blocks.append(ThinkingBlock(type=thinking_type, content=content))

        # If no structured blocks, try simple format (default to REASONING)
        if not blocks:
            for match in cls.SIMPLE_PATTERN.finditer(text):
                content = match.group(1).strip()
                blocks.append(
                    ThinkingBlock(type=ThinkingType.REASONING, content=content)
                )

        return blocks

    @classmethod
    def remove_thinking_blocks(cls, text: str) -> str:
        """Remove all thinking blocks from text (return only actions/output)."""
        text = cls.STRUCTURED_PATTERN.sub("", text)
        text = cls.SIMPLE_PATTERN.sub("", text)
        from core.response_utils import clean_response

        return clean_response(text)


class EnhancedThinkingSystem:
    """
    Full thinking pipeline with optional CoCoNut integration.

    Features:
    - Thinking budget management
    - Structured thinking blocks
    - CoCoNut latent reasoning (when enabled)
    - Chain persistence
    """

    def __init__(
        self,
        thinking_budget: Optional[int] = None,
        coconut_enabled: Optional[bool] = None,
        **extra_config,
    ):
        from config.settings import AGENTIC_THINKING, COCONUT_CONFIG, MASTER_MODEL

        self.thinking_budget = (
            thinking_budget
            if thinking_budget is not None
            else AGENTIC_THINKING.get("thinking_budget", 300000)
        )
        self.thinking_tokens_used = 0
        self.coconut_enabled = (
            coconut_enabled
            if coconut_enabled is not None
            else AGENTIC_THINKING.get("coconut_enabled", True)
        )
        self.brain = extra_config.pop("brain", None)
        self.model_name = extra_config.pop("model_name", MASTER_MODEL)

        merged_config: Dict[str, Any] = {**COCONUT_CONFIG, **extra_config}
        merged_config["embedding_dim"] = self._resolve_embedding_dim(
            merged_config.get("embedding_dim", "auto")
        )
        self.config = merged_config
        self.repo_root = Path(__file__).resolve().parents[1]
        self.aal_classifier = AALClassifier()
        self.domain_detector = DomainDetector()
        self.rule_registry = AESRuleRegistry()
        rules_path = self.repo_root / "standards" / "AES_RULES.json"
        if rules_path.exists():
            self.rule_registry.load(str(rules_path))

        self.default_compliance_context: Dict[str, Any] = {
            "trace_id": None,
            "evidence_bundle_id": None,
            "red_team_required": False,
            "waiver_ids": [],
            "waiver_id": None,
        }
        self.current_chain: Optional[ThinkingChain] = None
        self.parser = ThinkingParser()

        # CoCoNut integration (eagerly loaded)
        self._coconut = None
        if self.coconut_enabled:
            try:
                from core.reasoning.coconut import ContinuousThoughtBlock

                coconut_extra = {
                    k: v
                    for k, v in self.config.items()
                    if k
                    not in {
                        "embedding_dim",
                        "num_paths",
                        "steps",
                        "backend",
                        "use_fft",
                        "persistent_freq_state",
                        "deterministic",
                    }
                }
                self._coconut = ContinuousThoughtBlock(
                    embedding_dim=self.config.get("embedding_dim", 4096),
                    num_paths=self.config.get("num_paths", 4),
                    steps=self.config.get("steps", 2),
                    use_gpu=False,
                    backend=self.config.get("backend", "native"),
                    gpu_device="/CPU:0",
                    use_fft=self.config.get("use_fft", True),
                    persistent_freq_state=self.config.get("persistent_freq_state", True),
                    deterministic=self.config.get("deterministic", True),
                    **coconut_extra,
                )

                if self._coconut is not None:
                    device_info = self._coconut.get_device_info()
                    logger.info(
                        f"COCONUT reasoning system active. Backend: {device_info.get('backend')} on {device_info.get('device')}"
                    )

            except Exception as e:
                logger.warning(
                    f"Failed to initialize COCONUT reasoning: {e}. Falling back to standard thinking."
                )
                self._coconut = None

    def _resolve_embedding_dim(self, configured_dim: Any) -> int:
        if configured_dim not in (None, "auto"):
            try:
                return int(configured_dim)
            except Exception:
                logger.warning(
                    "Invalid COCONUT embedding_dim=%r; falling back to auto-detect",
                    configured_dim,
                )

        detected = self._detect_embedding_dim_from_brain()
        if detected:
            logger.info("COCONUT embedding_dim auto-detected via embeddings: %d", detected)
            return detected

        detected = self._detect_embedding_dim_from_local_model()
        if detected:
            logger.info(
                "COCONUT embedding_dim auto-detected via local GGUF metadata: %d",
                detected,
            )
            return detected

        logger.warning("COCONUT embedding_dim auto-detect failed; defaulting to 4096")
        return 4096

    def _detect_embedding_dim_from_brain(self) -> Optional[int]:
        if self.brain is None:
            return None
        try:
            if hasattr(self.brain, "embeddings"):
                emb = self.brain.embeddings("embedding-dimension-probe")
            elif hasattr(self.brain, "get_embeddings"):
                emb = self.brain.get_embeddings("embedding-dimension-probe")
            else:
                return None
            arr = np.asarray(emb, dtype=np.float32).reshape(-1)
            if arr.size > 0:
                return int(arr.size)
        except Exception:
            return None
        return None

    def _detect_embedding_dim_from_local_model(self) -> Optional[int]:
        if not self.model_name:
            return None
        try:
            from core.model.gguf_loader import get_loader

            loader = get_loader(str(self.model_name))
            value = int(loader.get_embedding_dim())
            if 64 <= value <= 32768:
                return value
        except Exception:
            return None
        return None

    @property
    def coconut(self):
        """Returns the pre-initialized CoCoNut block."""
        return self._coconut

    def set_compliance_context(
        self,
        trace_id: Optional[str] = None,
        evidence_bundle_id: Optional[str] = None,
        red_team_required: Optional[bool] = None,
        waiver_ids: Optional[List[str]] = None,
        waiver_id: Optional[str] = None,
        run_id: Optional[str] = None,
        aal: Optional[str] = None,
        domains: Optional[List[str]] = None,
        changed_files: Optional[List[str]] = None,
        hot_paths: Optional[List[str]] = None,
        public_api_changes: Optional[List[str]] = None,
        dependency_changes: Optional[List[str]] = None,
        required_rule_ids: Optional[List[str]] = None,
        required_runtime_gates: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        self.default_compliance_context = self._normalize_compliance_context(
            compliance_context={
                "run_id": run_id,
                "aal": aal,
                "domains": domains,
                "changed_files": changed_files,
                "hot_paths": hot_paths,
                "public_api_changes": public_api_changes,
                "dependency_changes": dependency_changes,
                "required_rule_ids": required_rule_ids,
                "required_runtime_gates": required_runtime_gates,
                "trace_id": trace_id,
                "evidence_bundle_id": evidence_bundle_id,
                "red_team_required": red_team_required,
                "waiver_ids": waiver_ids,
                "waiver_id": waiver_id,
            }
        )
        if self.current_chain is not None:
            self.current_chain.compliance_context = dict(self.default_compliance_context)
        return dict(self.default_compliance_context)

    @staticmethod
    def _normalize_waiver_ids(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, (list, tuple, set)):
            normalized = [str(item).strip() for item in value if str(item).strip()]
            return list(dict.fromkeys(normalized))
        return []

    @staticmethod
    def _required_compliance_fields() -> tuple[str, ...]:
        return ("trace_id", "evidence_bundle_id", "red_team_required", "waiver_ids")

    def _missing_compliance_fields(self, context: Dict[str, Any]) -> list[str]:
        missing: list[str] = []
        if not context.get("trace_id"):
            missing.append("trace_id")
        if not context.get("evidence_bundle_id"):
            missing.append("evidence_bundle_id")
        if not isinstance(context.get("red_team_required"), bool):
            missing.append("red_team_required")
        if not isinstance(context.get("waiver_ids"), list):
            missing.append("waiver_ids")
        return missing

    def _normalize_compliance_context(
        self,
        task_id: Optional[str] = None,
        compliance_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        context = {
            "run_id": None,
            "aal": "AAL-3",
            "domains": [],
            "changed_files": [],
            "hot_paths": [],
            "public_api_changes": [],
            "dependency_changes": [],
            "required_rule_ids": [],
            "required_runtime_gates": [],
            "trace_id": None,
            "evidence_bundle_id": None,
            "red_team_required": False,
            "waiver_ids": [],
            "waiver_id": None,
        }
        if compliance_context:
            context.update(compliance_context)

        trace_id = context.get("trace_id") or f"trace::{task_id or datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        trace_id = str(trace_id).strip()
        if not trace_id:
            trace_id = f"trace::{task_id or datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        evidence_bundle_id = context.get("evidence_bundle_id") or f"evidence::{trace_id}"
        evidence_bundle_id = str(evidence_bundle_id).strip()
        if not evidence_bundle_id:
            evidence_bundle_id = f"evidence::{trace_id}"

        waiver_ids = self._normalize_waiver_ids(context.get("waiver_ids"))
        if not waiver_ids and context.get("waiver_id"):
            waiver_ids = self._normalize_waiver_ids(context.get("waiver_id"))
        waiver_id = waiver_ids[0] if waiver_ids else context.get("waiver_id")

        return {
            "run_id": context.get("run_id") or trace_id,
            "aal": str(context.get("aal") or "AAL-3").upper(),
            "domains": list(context.get("domains", []) or []),
            "changed_files": list(context.get("changed_files", []) or []),
            "hot_paths": list(context.get("hot_paths", []) or []),
            "public_api_changes": list(context.get("public_api_changes", []) or []),
            "dependency_changes": list(context.get("dependency_changes", []) or []),
            "required_rule_ids": list(context.get("required_rule_ids", []) or []),
            "required_runtime_gates": list(
                context.get("required_runtime_gates", []) or []
            ),
            "trace_id": trace_id,
            "evidence_bundle_id": evidence_bundle_id,
            "red_team_required": bool(context.get("red_team_required")),
            "waiver_ids": waiver_ids,
            "waiver_id": waiver_id,
        }

    def _build_compliance_block(
        self,
        task_id: str,
        compliance_context: Dict[str, Any],
        files: Optional[List[str]] = None,
    ) -> str:
        tracked_files = [item for item in (files or []) if item]
        if tracked_files:
            aal = self.aal_classifier.classify_changeset(tracked_files)
            domains = sorted(self.domain_detector.detect_domains(tracked_files))
        else:
            aal = "AAL-2"
            domains = []

        rule_ids = sorted(
            set(rule.id for rule in self.rule_registry.get_rules_for_aal(aal))
            | set(
                rule.id
                for domain in domains
                for rule in self.rule_registry.get_rules_for_domain(domain)
            )
        )
        required_verification = {
            "AAL-0": "MC/DC coverage, red-team, independent review, closure artifacts",
            "AAL-1": "branch coverage, red-team, review signoff, closure artifacts",
            "AAL-2": "functional + regression verification with evidence",
            "AAL-3": "basic verification + hygiene review",
        }.get(aal, "functional verification with evidence")
        anti_patterns = [
            "bare exception swallowing",
            "verification bypass fallback",
            "clipping/clamping to mask defects",
            "missing traceability evidence",
        ]
        waiver_ids = compliance_context.get("waiver_ids") or []
        return "\n".join(
            [
                f"Task ID: {task_id}",
                f"AAL Classification: {aal}",
                f"Domains: {', '.join(domains) if domains else 'universal'}",
                "Applicable AES Rules: " + (", ".join(rule_ids[:24]) if rule_ids else "none"),
                "Required Verification: " + required_verification,
                "Anti-Patterns to Avoid: " + ", ".join(anti_patterns),
                "Compliance IDs: "
                + f"trace_id={compliance_context.get('trace_id')}, "
                + f"evidence_bundle_id={compliance_context.get('evidence_bundle_id')}, "
                + f"red_team_required={bool(compliance_context.get('red_team_required'))}, "
                + f"waiver_ids={waiver_ids if waiver_ids else 'none'}",
            ]
        )

    def generate_thinking_prompt(
        self,
        task: str,
        files: Optional[List[str]] = None,
        mode: str = "understanding",
    ) -> str:
        """Generate prompt with mechanically inserted compliance block."""
        compliance_context = self._normalize_compliance_context(
            task_id=task or "task",
            compliance_context=dict(self.default_compliance_context),
        )
        compliance_block = (
            '<thinking type="compliance">\n'
            + self._build_compliance_block(task or "task", compliance_context, files)
            + "\n</thinking>\n"
        )
        return compliance_block + self.get_thinking_prompt(mode)

    def _deterministic_compliance_repair(
        self,
        task_id: str,
        normalized_context: Dict[str, Any],
    ) -> tuple[Dict[str, Any], bool]:
        missing = self._missing_compliance_fields(normalized_context)
        if not missing:
            return normalized_context, False
        repaired = self._normalize_compliance_context(
            task_id=task_id,
            compliance_context=normalized_context,
        )
        remaining = self._missing_compliance_fields(repaired)
        if remaining:
            raise ValueError(
                "Unable to repair compliance context; missing fields remain: "
                + ", ".join(remaining)
            )
        return repaired, True

    def start_chain(
        self,
        task_id: str = None,
        compliance_context: Optional[Dict[str, Any]] = None,
        files: Optional[List[str]] = None,
    ) -> ThinkingChain:
        """Start a new thinking chain with mandatory compliance insertion."""
        task_id = task_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        initial_missing = (
            self._missing_compliance_fields(compliance_context or {})
            if compliance_context is not None
            else []
        )
        normalized_context = self._normalize_compliance_context(
            task_id=task_id,
            compliance_context=compliance_context
            or dict(self.default_compliance_context),
        )
        repaired_context, did_repair = self._deterministic_compliance_repair(
            task_id=task_id,
            normalized_context=normalized_context,
        )
        did_repair = did_repair or bool(initial_missing)
        self.current_chain = ThinkingChain(
            task_id=task_id,
            compliance_context=repaired_context,
        )
        self.thinking_tokens_used = 0
        compliance_block = self._build_compliance_block(task_id, repaired_context, files)
        self.current_chain.add_compliance(compliance_block)
        if did_repair:
            self.current_chain.add_correction(
                "Deterministic compliance-context repair applied before continuing."
            )
        return self.current_chain

    def get_chain(self) -> Optional[ThinkingChain]:
        """Get current thinking chain."""
        return self.current_chain

    def think(self, type: ThinkingType, content: str) -> ThinkingBlock:
        """Record a thinking block."""
        if not self.current_chain:
            self.start_chain()

        block = self.current_chain.add(type, content)

        # Track token usage (rough estimate: 4 chars per token)
        self.thinking_tokens_used += len(content) // 4

        return block

    def deep_think(self, context_embedding: np.ndarray) -> Optional[np.ndarray]:
        """
        Use CoCoNut to explore latent reasoning paths.

        Args:
            context_embedding: [Batch, Dim] embedding of current context

        Returns:
            Refined embedding after CoCoNut exploration, or None if disabled
        """
        if not self.coconut_enabled or self.coconut is None:
            return None

        return self.coconut.explore(context_embedding)

    def rank_evidence(
        self, evidence_chunks: List[str], context_embedding: np.ndarray
    ) -> List[Tuple[int, float]]:
        """Rank evidence chunks by COCONUT-refined relevance."""
        if not evidence_chunks:
            return []

        reference = np.asarray(context_embedding, dtype=np.float32).reshape(-1)
        if reference.size == 0:
            return [(idx, 0.0) for idx in range(len(evidence_chunks))]

        if self.coconut_enabled and self.coconut is not None:
            try:
                refined = self.deep_think(reference.reshape(1, -1))
                if refined is not None:
                    reference = np.asarray(refined, dtype=np.float32).reshape(-1)
            except Exception:
                pass

        scored: List[Tuple[int, float]] = []
        for idx, chunk in enumerate(evidence_chunks):
            chunk_embedding = self._embed_text(chunk, target_dim=reference.size)
            if chunk_embedding is None:
                chunk_embedding = self._lexical_embedding(chunk, target_dim=reference.size)
            score = self._cosine_similarity(reference, chunk_embedding)
            scored.append((idx, float(score)))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    def _embed_text(self, text: str, target_dim: int) -> Optional[np.ndarray]:
        if not self.brain:
            return None
        try:
            if hasattr(self.brain, "embeddings"):
                emb = self.brain.embeddings(text)
            elif hasattr(self.brain, "get_embeddings"):
                emb = self.brain.get_embeddings(text)
            else:
                return None
            arr = np.asarray(emb, dtype=np.float32).reshape(-1)
            if arr.size == 0 or not np.isfinite(arr).all():
                return None
            return self._fit_embedding_dim(arr, target_dim)
        except Exception:
            return None

    def _fit_embedding_dim(self, embedding: np.ndarray, target_dim: int) -> np.ndarray:
        arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if arr.size >= target_dim:
            return arr[:target_dim]
        padded = np.zeros(target_dim, dtype=np.float32)
        padded[: arr.size] = arr
        return padded

    def _lexical_embedding(self, text: str, target_dim: int) -> np.ndarray:
        vec = np.zeros(target_dim, dtype=np.float32)
        if target_dim <= 0:
            return vec
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{1,}", (text or "").lower())[:256]:
            vec[hash(token) % target_dim] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        a_arr = np.asarray(a, dtype=np.float32).reshape(-1)
        b_arr = np.asarray(b, dtype=np.float32).reshape(-1)
        dim = min(a_arr.size, b_arr.size)
        if dim == 0:
            return 0.0
        a_arr = a_arr[:dim]
        b_arr = b_arr[:dim]
        denom = float(np.linalg.norm(a_arr) * np.linalg.norm(b_arr))
        if denom <= 1e-8:
            return 0.0
        return float(np.dot(a_arr, b_arr) / denom)

    def increase_budget(self, factor: float = 2.0) -> int:
        """Increase thinking budget for complex problems."""
        self.thinking_budget = int(self.thinking_budget * factor)
        return self.thinking_budget

    def remaining_budget(self) -> int:
        """Get remaining thinking token budget."""
        return max(0, self.thinking_budget - self.thinking_tokens_used)

    def is_over_budget(self) -> bool:
        """Check if thinking budget is exhausted."""
        return self.thinking_tokens_used >= self.thinking_budget

    def parse_response(self, text: str) -> List[ThinkingBlock]:
        """Parse thinking blocks from model response and add to chain."""
        blocks = self.parser.parse(text)

        if self.current_chain:
            for block in blocks:
                self.current_chain.blocks.append(block)

        return blocks

    def save_chain(self, path: str) -> None:
        """Save current chain to file."""
        if self.current_chain:
            self.current_chain.save(path)

    def get_thinking_prompt(self, mode: str = "understanding") -> str:
        """Get a prompt that encourages structured thinking."""
        compliance = self.default_compliance_context
        waiver_ids = compliance.get("waiver_ids") or []
        compliance_line = (
            "Compliance IDs: "
            f"trace_id={compliance.get('trace_id') or 'REQUIRED'}, "
            f"evidence_bundle_id={compliance.get('evidence_bundle_id') or 'REQUIRED'}, "
            f"red_team_required={bool(compliance.get('red_team_required'))}, "
            f"waiver_ids={waiver_ids if waiver_ids else 'NONE'}\n"
        )
        prompts = {
            "compliance": '<thinking type="compliance">\n'
            + compliance_line
            + "Validate AES constraints before synthesis:\n",
            "understanding": '<thinking type="understanding">\n'
            + compliance_line
            + "Let me understand this problem:\n",
            "planning": '<thinking type="planning">\n'
            + compliance_line
            + "My approach will be:\n",
            "reasoning": '<thinking type="reasoning">\n'
            + compliance_line
            + "Working through this step-by-step:\n",
            "reflection": '<thinking type="reflection">\n'
            + compliance_line
            + "Evaluating my progress:\n",
            "correction": '<thinking type="correction">\n'
            + compliance_line
            + "I need to fix this:\n",
        }
        return prompts.get(mode, "<thinking>\n")
