from typing import Any, Dict, List, Optional


class ContextManager:
    """
    Manages conversation context, token counting, and windowing.
    """

    def __init__(self, max_tokens=128000, system_prompt_tokens=1000):
        self.max_tokens = max_tokens
        # Reserve space for system prompt and tool usage
        self.available_tokens = max_tokens - system_prompt_tokens
        self._native_counter_unavailable = False
        self._tiktoken_unavailable = False
        self._native_tokenize = None
        self._tiktoken_encoder = None

    def count_tokens(self, text: str) -> int:
        """
        Estimates token count with native/tiktoken fallbacks.
        """
        if not text:
            return 0

        native = self._count_tokens_native(text)
        if native is not None:
            return native

        tiktoken_count = self._count_tokens_tiktoken(text)
        if tiktoken_count is not None:
            return tiktoken_count

        return max(1, len(text) // 4)

    def _count_tokens_native(self, text: str) -> Optional[int]:
        """Best-effort token counting via Saguaro native tokenizer."""
        if self._native_counter_unavailable:
            return None

        try:
            if self._native_tokenize is None:
                from saguaro.ops.fused_text_tokenizer import fused_text_tokenize_batch

                self._native_tokenize = fused_text_tokenize_batch

            max_len = max(8192, min(262144, len(text) + 64))
            _, lengths = self._native_tokenize(
                [text],
                add_special_tokens=False,
                max_length=max_len,
                inject_thinking=False,
            )
            if hasattr(lengths, "numpy"):
                lengths = lengths.numpy().tolist()
            return int(lengths[0])
        except Exception:
            self._native_counter_unavailable = True
            return None

    def _count_tokens_tiktoken(self, text: str) -> Optional[int]:
        """Fallback token counting via tiktoken."""
        if self._tiktoken_unavailable:
            return None

        try:
            if self._tiktoken_encoder is None:
                import tiktoken

                self._tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
            return len(self._tiktoken_encoder.encode(text))
        except Exception:
            self._tiktoken_unavailable = True
            return None

    def count_message_tokens(self, message: Dict) -> int:
        """Counts tokens in a message object."""
        content = message.get("content", "")
        # Add slight overhead for role serialization
        return self.count_tokens(content) + 4

    def get_context_window(self, messages: List[Dict]) -> List[Dict]:
        """
        Returns the list of messages that fit within the context window.
        Prioritizes recent messages (sliding window).
        """
        total_tokens = 0
        window_messages = []

        # Iterate backwards
        for msg in reversed(messages):
            msg_tokens = self.count_message_tokens(msg)
            if total_tokens + msg_tokens > self.available_tokens:
                break

            window_messages.insert(0, msg)
            total_tokens += msg_tokens

        return window_messages

    def get_stats(self, messages: List[Dict]) -> str:
        """Returns usage stats string."""
        fill = self.get_fill_percentage(messages)
        return (
            "Context Usage: "
            f"{fill['used_tokens']}/{fill['max_tokens']} tokens "
            f"({fill['fill_percentage']:.1f}%)"
        )

    def get_fill_percentage(self, messages: List[Dict]) -> Dict[str, float]:
        """Return context usage metadata."""
        used = sum(self.count_message_tokens(m) for m in messages)
        pct = (used / self.max_tokens * 100.0) if self.max_tokens else 0.0
        return {
            "used_tokens": used,
            "max_tokens": self.max_tokens,
            "fill_percentage": pct,
        }


class ContextBudgetAllocator:
    """Split total context window into stable component budgets."""

    def __init__(self, total_budget: int = 400000):
        self.total_budget = int(total_budget)
        self.budgets = {
            "master": int(self.total_budget * 0.45),
            "coconut": int(self.total_budget * 0.05),
            "subagent_pool": int(self.total_budget * 0.40),
            "system": int(self.total_budget * 0.10),
        }
        self._rebalance_rounding_delta()

    @property
    def coconut_latent_budget(self) -> int:
        """Independent COCONUT latent-space capacity (not token-context budget)."""
        from config.settings import COCONUT_CONFIG

        return int(COCONUT_CONFIG.get("context_budget", 400000))

    def _rebalance_rounding_delta(self) -> None:
        current = sum(self.budgets.values())
        delta = self.total_budget - current
        if delta:
            self.budgets["master"] += delta

    def get_budget(self, component: str, default: int = 0) -> int:
        return int(self.budgets.get(component, default))

    def get_subagent_budget(self, active_subagent_count: int) -> int:
        active = max(1, int(active_subagent_count or 1))
        return max(4000, int(self.budgets["subagent_pool"] / active))

    def recommend_total_budget(self, recommended_total_budget: int) -> Dict[str, int]:
        self.total_budget = int(max(60000, recommended_total_budget))
        self.budgets["master"] = int(self.total_budget * 0.45)
        self.budgets["coconut"] = int(self.total_budget * 0.05)
        self.budgets["subagent_pool"] = int(self.total_budget * 0.40)
        self.budgets["system"] = int(self.total_budget * 0.10)
        self._rebalance_rounding_delta()
        return dict(self.budgets)

    def summary(self) -> Dict[str, Any]:
        payload = {"total_budget": self.total_budget}
        payload.update(self.budgets)
        return payload
