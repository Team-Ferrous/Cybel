"""
Error Recovery Manager - Intelligent error recovery with exponential backoff and learning.

Provides automatic error recovery strategies for common failure modes:
- Out of memory (OOM): Reduce context, retry with smaller batch
- Timeout: Switch to faster model tier
- API failures: Exponential backoff with jitter
- Logic errors: Invoke debug loop
- Tool errors: Fallback to alternative tools

Features:
- Error classification and strategy selection
- Exponential backoff with configurable limits
- Context trimming for OOM recovery
- Learning from past recoveries
- Circuit breaker pattern for flaky operations
"""

import logging
import time
import random
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Classification of error types."""

    OOM = "out_of_memory"
    TIMEOUT = "timeout"
    API_ERROR = "api_error"
    LOGIC_ERROR = "logic_error"
    TOOL_ERROR = "tool_error"
    CONTEXT_TOO_LARGE = "context_too_large"
    RATE_LIMIT = "rate_limit"
    NETWORK_ERROR = "network_error"
    UNKNOWN = "unknown"


class RecoveryStrategy(Enum):
    """Available recovery strategies."""

    REDUCE_CONTEXT_AND_RETRY = "reduce_context_and_retry"
    SWITCH_TO_FASTER_MODEL = "switch_to_faster_model"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    INVOKE_DEBUG_LOOP = "invoke_debug_loop"
    FALLBACK_TO_ALTERNATIVE_TOOL = "fallback_to_alternative_tool"
    TRIM_HISTORY = "trim_history"
    REDUCE_BATCH_SIZE = "reduce_batch_size"
    ABORT = "abort"


@dataclass
class RecoveryResult:
    """Result of a recovery attempt."""

    recovered: bool
    strategy_used: RecoveryStrategy
    retries_attempted: int
    error_message: Optional[str] = None
    context_modifications: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CircuitBreakerState:
    """State of a circuit breaker for a specific operation."""

    operation_name: str
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    state: str = "closed"  # "closed", "open", "half_open"
    failure_threshold: int = 5
    reset_timeout_seconds: int = 60


class ErrorRecoveryManager:
    """
    Automatic error recovery with intelligent strategy selection.

    Recovery Flow:
    1. Classify error type
    2. Select appropriate recovery strategy
    3. Execute recovery with retries and backoff
    4. Log outcomes for learning
    5. Update circuit breakers

    Example:
        manager = ErrorRecoveryManager()

        try:
            result = risky_operation()
        except Exception as e:
            recovery = manager.handle_error(e, {
                'step': 5,
                'context_size': 10000
            })

            if recovery.recovered:
                # Retry with adjusted context
                result = risky_operation()
            else:
                # Unrecoverable
                raise
    """

    # Error classification patterns
    ERROR_PATTERNS = {
        ErrorType.OOM: [
            "out of memory",
            "oom",
            "memory error",
            "allocation failed",
            "cuda out of memory",
            "cannot allocate",
        ],
        ErrorType.TIMEOUT: [
            "timeout",
            "timed out",
            "deadline exceeded",
            "request timeout",
        ],
        ErrorType.API_ERROR: [
            "api error",
            "connection refused",
            "service unavailable",
            "bad gateway",
            "http 5",
            "http 4",
        ],
        ErrorType.RATE_LIMIT: [
            "rate limit",
            "too many requests",
            "quota exceeded",
            "http 429",
        ],
        ErrorType.CONTEXT_TOO_LARGE: [
            "context length",
            "token limit",
            "context too long",
            "maximum context",
            "exceeds context window",
        ],
        ErrorType.NETWORK_ERROR: [
            "network error",
            "connection error",
            "dns",
            "unreachable",
        ],
        ErrorType.TOOL_ERROR: ["tool error", "command failed", "execution error"],
    }

    # Strategy mapping
    STRATEGY_MAP = {
        ErrorType.OOM: RecoveryStrategy.REDUCE_CONTEXT_AND_RETRY,
        ErrorType.CONTEXT_TOO_LARGE: RecoveryStrategy.TRIM_HISTORY,
        ErrorType.TIMEOUT: RecoveryStrategy.SWITCH_TO_FASTER_MODEL,
        ErrorType.API_ERROR: RecoveryStrategy.EXPONENTIAL_BACKOFF,
        ErrorType.RATE_LIMIT: RecoveryStrategy.EXPONENTIAL_BACKOFF,
        ErrorType.NETWORK_ERROR: RecoveryStrategy.EXPONENTIAL_BACKOFF,
        ErrorType.TOOL_ERROR: RecoveryStrategy.FALLBACK_TO_ALTERNATIVE_TOOL,
        ErrorType.LOGIC_ERROR: RecoveryStrategy.INVOKE_DEBUG_LOOP,
        ErrorType.UNKNOWN: RecoveryStrategy.ABORT,
    }

    def __init__(self, max_retries: int = 3, base_backoff_seconds: float = 1.0):
        self.max_retries = max_retries
        self.base_backoff = base_backoff_seconds

        # Circuit breakers for flaky operations
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {}

        # Recovery history for learning
        self.recovery_history: List[Dict] = []

    def handle_error(self, error: Exception, context: Dict[str, Any]) -> RecoveryResult:
        """
        Handle an error with automatic recovery.

        Args:
            error: The exception that occurred
            context: Execution context (step, objective, context_size, etc.)

        Returns:
            RecoveryResult with recovery outcome
        """
        # 1. Classify error
        error_type = self._classify_error(error)
        logger.info(f"Classified error as: {error_type.value}")

        # 2. Select recovery strategy
        strategy = self._select_strategy(error_type, context)
        logger.info(f"Selected recovery strategy: {strategy.value}")

        # 3. Execute recovery
        result = self._execute_recovery(strategy, error, context)

        # 4. Log for learning
        self._log_recovery(error_type, strategy, result, context)

        return result

    def _classify_error(self, error: Exception) -> ErrorType:
        """Classify error based on message patterns."""
        error_msg = str(error).lower()

        for error_type, patterns in self.ERROR_PATTERNS.items():
            if any(pattern in error_msg for pattern in patterns):
                return error_type

        # Check exception type
        if isinstance(error, MemoryError):
            return ErrorType.OOM
        elif isinstance(error, TimeoutError):
            return ErrorType.TIMEOUT
        elif "ConnectionError" in type(error).__name__:
            return ErrorType.NETWORK_ERROR

        return ErrorType.UNKNOWN

    def _select_strategy(
        self, error_type: ErrorType, context: Dict
    ) -> RecoveryStrategy:
        """Select appropriate recovery strategy."""
        # Check if we have custom strategy overrides in context
        if "recovery_strategy" in context:
            return context["recovery_strategy"]

        # Use default mapping
        return self.STRATEGY_MAP.get(error_type, RecoveryStrategy.ABORT)

    def _execute_recovery(
        self, strategy: RecoveryStrategy, error: Exception, context: Dict
    ) -> RecoveryResult:
        """Execute the recovery strategy."""
        logger.debug(f"Executing recovery strategy: {strategy.value}")
        aal = str(context.get("aal", "AAL-2")).upper()
        waiver_ids = self._normalized_waiver_ids(context)

        # AES-CR-ROOT-1 / Phase 3: prohibit silent downgrade paths in high-assurance flows.
        if self._is_masking_strategy(strategy) and self._is_high_aal(aal):
            if not self._has_approved_waiver(waiver_ids):
                logger.warning(
                    "AES recovery guard blocked %s for %s (missing waiver).",
                    strategy.value,
                    aal,
                )
                return RecoveryResult(
                    recovered=False,
                    strategy_used=strategy,
                    retries_attempted=0,
                    error_message=(
                        f"AES: {strategy.value} blocked for {aal} without approved waiver"
                    ),
                    context_modifications={
                        "aal": aal,
                        "waiver_ids": waiver_ids,
                        "aes_compliant": False,
                        "escalate_supervised": True,
                    },
                )

        if strategy == RecoveryStrategy.REDUCE_CONTEXT_AND_RETRY:
            result = self._reduce_context(context)

        elif strategy == RecoveryStrategy.TRIM_HISTORY:
            result = self._trim_history(context)

        elif strategy == RecoveryStrategy.EXPONENTIAL_BACKOFF:
            result = self._exponential_backoff(context)

        elif strategy == RecoveryStrategy.SWITCH_TO_FASTER_MODEL:
            result = self._switch_model(context)

        elif strategy == RecoveryStrategy.FALLBACK_TO_ALTERNATIVE_TOOL:
            result = self._fallback_tool(context)

        elif strategy == RecoveryStrategy.INVOKE_DEBUG_LOOP:
            result = self._invoke_debug_loop(error, context)

        elif strategy == RecoveryStrategy.REDUCE_BATCH_SIZE:
            result = self._reduce_batch_size(context)

        else:  # ABORT
            result = RecoveryResult(
                recovered=False,
                strategy_used=strategy,
                retries_attempted=0,
                error_message=f"Unrecoverable error: {strategy.value}",
            )

        result.context_modifications.setdefault("aal", aal)
        result.context_modifications.setdefault("waiver_ids", waiver_ids)
        result.context_modifications.setdefault("aes_compliant", True)
        return result

    @staticmethod
    def _is_high_aal(aal: str) -> bool:
        return str(aal or "AAL-3").upper() in {"AAL-0", "AAL-1"}

    @staticmethod
    def _is_masking_strategy(strategy: RecoveryStrategy) -> bool:
        return strategy in {
            RecoveryStrategy.SWITCH_TO_FASTER_MODEL,
            RecoveryStrategy.FALLBACK_TO_ALTERNATIVE_TOOL,
        }

    @staticmethod
    def _normalized_waiver_ids(context: Dict[str, Any]) -> List[str]:
        waiver_ids = context.get("waiver_ids") or []
        if not waiver_ids and context.get("waiver_id"):
            waiver_ids = [context.get("waiver_id")]
        return [str(item).strip() for item in waiver_ids if str(item).strip()]

    @staticmethod
    def _has_approved_waiver(waiver_ids: List[str]) -> bool:
        approved = {"fallback-waiver", "high-aal-override", "aes-recovery-approved"}
        return bool(set(waiver_ids).intersection(approved))

    def _reduce_context(self, context: Dict) -> RecoveryResult:
        """Reduce context size and retry."""
        current_size = context.get("context_size", 0)
        if current_size == 0:
            return RecoveryResult(
                recovered=False,
                strategy_used=RecoveryStrategy.REDUCE_CONTEXT_AND_RETRY,
                retries_attempted=0,
                error_message="No context to reduce",
            )

        # Reduce by 25%
        new_size = int(current_size * 0.75)
        logger.info(f"Reducing context from {current_size} to {new_size}")

        return RecoveryResult(
            recovered=True,
            strategy_used=RecoveryStrategy.REDUCE_CONTEXT_AND_RETRY,
            retries_attempted=1,
            context_modifications={"context_size": new_size},
        )

    def _trim_history(self, context: Dict) -> RecoveryResult:
        """Trim message history to fit context window."""
        # Keep only recent N messages
        keep_messages = context.get("keep_recent_messages", 10)

        logger.info(f"Trimming history to keep only {keep_messages} recent messages")

        return RecoveryResult(
            recovered=True,
            strategy_used=RecoveryStrategy.TRIM_HISTORY,
            retries_attempted=1,
            context_modifications={"trim_to_messages": keep_messages},
        )

    def _exponential_backoff(self, context: Dict) -> RecoveryResult:
        """Wait with exponential backoff then retry."""
        retry_count = context.get("retry_count", 0)

        if retry_count >= self.max_retries:
            return RecoveryResult(
                recovered=False,
                strategy_used=RecoveryStrategy.EXPONENTIAL_BACKOFF,
                retries_attempted=retry_count,
                error_message=f"Max retries ({self.max_retries}) exceeded",
            )

        # Calculate backoff with jitter
        backoff = self.base_backoff * (2**retry_count)
        jitter = random.uniform(0, backoff * 0.1)  # 10% jitter
        wait_time = backoff + jitter

        logger.info(
            f"Exponential backoff: waiting {wait_time:.2f}s (attempt {retry_count + 1}/{self.max_retries})"
        )
        time.sleep(wait_time)

        return RecoveryResult(
            recovered=True,
            strategy_used=RecoveryStrategy.EXPONENTIAL_BACKOFF,
            retries_attempted=retry_count + 1,
            context_modifications={"retry_count": retry_count + 1},
        )

    def _switch_model(self, context: Dict) -> RecoveryResult:
        """Switch to faster model tier."""
        current_model = context.get("model", "unknown")

        # Model tier downgrade mapping
        faster_models = {
            "granite4:tiny-h": "granite4:tiny-h",  # Already fastest
            "qwen2.5-coder:7b": "granite4:tiny-h",
            "granite-3.1-dense:8b": "qwen2.5-coder:7b",
        }

        new_model = faster_models.get(current_model, current_model)

        if new_model == current_model:
            logger.warning(
                f"Already using fast model: {current_model}, cannot downgrade further"
            )
            return RecoveryResult(
                recovered=False,
                strategy_used=RecoveryStrategy.SWITCH_TO_FASTER_MODEL,
                retries_attempted=0,
                error_message="Cannot switch to faster model",
            )

        logger.info(f"Switching model: {current_model} → {new_model}")

        return RecoveryResult(
            recovered=True,
            strategy_used=RecoveryStrategy.SWITCH_TO_FASTER_MODEL,
            retries_attempted=1,
            context_modifications={"model": new_model},
        )

    def _fallback_tool(self, context: Dict) -> RecoveryResult:
        """Fallback to alternative tool."""
        failed_tool = context.get("tool_name", "unknown")

        # Tool fallback mapping
        tool_alternatives = {
            "search_code": "grep",  # If semantic search fails, use basic grep
            "verify": "run_command",  # If verify fails, try basic command
        }

        alternative = tool_alternatives.get(failed_tool)

        if not alternative:
            return RecoveryResult(
                recovered=False,
                strategy_used=RecoveryStrategy.FALLBACK_TO_ALTERNATIVE_TOOL,
                retries_attempted=0,
                error_message=f"No alternative tool for: {failed_tool}",
            )

        logger.info(f"Falling back from {failed_tool} to {alternative}")

        return RecoveryResult(
            recovered=True,
            strategy_used=RecoveryStrategy.FALLBACK_TO_ALTERNATIVE_TOOL,
            retries_attempted=1,
            context_modifications={"tool_name": alternative},
        )

    def _invoke_debug_loop(self, error: Exception, context: Dict) -> RecoveryResult:
        """Invoke debug loop to analyze error."""
        logger.info("Invoking debug loop for error analysis")

        # Signal that debug loop should be invoked
        return RecoveryResult(
            recovered=True,
            strategy_used=RecoveryStrategy.INVOKE_DEBUG_LOOP,
            retries_attempted=1,
            context_modifications={
                "invoke_debug_loop": True,
                "error_traceback": str(error),
            },
        )

    def _reduce_batch_size(self, context: Dict) -> RecoveryResult:
        """Reduce batch size for processing."""
        current_batch = context.get("batch_size", 1)
        new_batch = max(1, current_batch // 2)

        logger.info(f"Reducing batch size: {current_batch} → {new_batch}")

        return RecoveryResult(
            recovered=True,
            strategy_used=RecoveryStrategy.REDUCE_BATCH_SIZE,
            retries_attempted=1,
            context_modifications={"batch_size": new_batch},
        )

    def _log_recovery(
        self,
        error_type: ErrorType,
        strategy: RecoveryStrategy,
        result: RecoveryResult,
        context: Dict,
    ):
        """Log recovery attempt for learning."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type.value,
            "strategy": strategy.value,
            "recovered": result.recovered,
            "retries": result.retries_attempted,
            "context_snippet": str(context)[:200],
        }

        self.recovery_history.append(entry)
        logger.debug(f"Recovery logged: {entry}")

    def get_circuit_breaker(self, operation_name: str) -> CircuitBreakerState:
        """Get or create circuit breaker for an operation."""
        if operation_name not in self.circuit_breakers:
            self.circuit_breakers[operation_name] = CircuitBreakerState(
                operation_name=operation_name
            )
        return self.circuit_breakers[operation_name]

    def record_success(self, operation_name: str):
        """Record successful operation."""
        cb = self.get_circuit_breaker(operation_name)
        cb.success_count += 1
        cb.failure_count = 0  # Reset on success

        # Close circuit if it was half-open
        if cb.state == "half_open":
            cb.state = "closed"
            logger.info(f"Circuit breaker closed for: {operation_name}")

    def record_failure(self, operation_name: str):
        """Record failed operation and update circuit breaker."""
        cb = self.get_circuit_breaker(operation_name)
        cb.failure_count += 1
        cb.last_failure_time = datetime.now()

        # Open circuit if threshold exceeded
        if cb.failure_count >= cb.failure_threshold and cb.state == "closed":
            cb.state = "open"
            logger.warning(
                f"Circuit breaker opened for: {operation_name} ({cb.failure_count} failures)"
            )

    def is_circuit_open(self, operation_name: str) -> bool:
        """Check if circuit breaker is open."""
        cb = self.get_circuit_breaker(operation_name)

        if cb.state == "closed":
            return False

        if cb.state == "open":
            # Check if reset timeout has passed
            if cb.last_failure_time:
                elapsed = (datetime.now() - cb.last_failure_time).total_seconds()
                if elapsed > cb.reset_timeout_seconds:
                    cb.state = "half_open"
                    logger.info(f"Circuit breaker half-open for: {operation_name}")
                    return False

            return True

        # Half-open: allow one attempt
        return False

    def get_stats(self) -> Dict:
        """Get recovery statistics."""
        if not self.recovery_history:
            return {"total_recoveries": 0}

        total = len(self.recovery_history)
        successful = sum(1 for r in self.recovery_history if r["recovered"])

        by_type = {}
        by_strategy = {}

        for entry in self.recovery_history:
            error_type = entry["error_type"]
            strategy = entry["strategy"]

            by_type[error_type] = by_type.get(error_type, 0) + 1
            by_strategy[strategy] = by_strategy.get(strategy, 0) + 1

        return {
            "total_recoveries": total,
            "successful_recoveries": successful,
            "success_rate": successful / total if total > 0 else 0.0,
            "by_error_type": by_type,
            "by_strategy": by_strategy,
            "circuit_breakers": {
                name: {"state": cb.state, "failures": cb.failure_count}
                for name, cb in self.circuit_breakers.items()
            },
        }
