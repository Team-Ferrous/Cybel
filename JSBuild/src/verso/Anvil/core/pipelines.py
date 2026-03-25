from __future__ import annotations

from typing import Any, Dict, Optional


class PipelineManager:
    """Resolve request-type-specific generation presets."""

    CONFIGS: Dict[str, Dict[str, Any]] = {
        "generation": {"temperature": 0.2, "top_p": 0.9, "max_tokens": 4096},
        "review": {"temperature": 0.0, "seed": 720720, "max_tokens": 2048},
        "creative": {
            "temperature": 0.8,
            "top_p": 0.95,
            "repeat_penalty": 1.1,
            "max_tokens": 4096,
        },
        "deterministic": {"temperature": 0.0, "seed": 42, "max_tokens": 2048},
    }

    _CREATIVE_KEYWORDS = (
        "brainstorm",
        "creative",
        "story",
        "poem",
        "fiction",
        "tagline",
        "slogan",
        "marketing copy",
        "lyrics",
        "joke",
    )

    def __init__(self, brain):
        self.brain = brain

    def get_pipeline_params(self, name: str) -> Dict[str, Any]:
        return dict(self.CONFIGS.get(name, self.CONFIGS["generation"]))

    def resolve_pipeline_name(
        self,
        request_type: Optional[str],
        *,
        user_input: Optional[str] = None,
    ) -> str:
        request = str(request_type or "").strip().lower()
        if request in {"question", "explanation", "investigation"}:
            return "review"
        if request in {"modification", "deletion"}:
            return "deterministic"
        if request == "creation":
            return "creative" if self._has_explicit_creative_intent(user_input) else "generation"
        if self._has_explicit_creative_intent(user_input):
            return "creative"
        return "generation"

    def resolve_generation_kwargs(
        self,
        *,
        request_type: Optional[str] = None,
        pipeline_name: Optional[str] = None,
        user_input: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        resolved_name = pipeline_name or self.resolve_pipeline_name(
            request_type,
            user_input=user_input,
        )
        params = self.get_pipeline_params(resolved_name)
        if overrides:
            params.update(overrides)
        return params

    def chat(
        self,
        messages,
        *,
        request_type: Optional[str] = None,
        pipeline_name: Optional[str] = None,
        user_input: Optional[str] = None,
        **overrides,
    ):
        params = self.resolve_generation_kwargs(
            request_type=request_type,
            pipeline_name=pipeline_name,
            user_input=user_input,
            overrides=overrides,
        )
        return self.brain.chat(messages, **params)

    def stream_chat(
        self,
        messages,
        *,
        request_type: Optional[str] = None,
        pipeline_name: Optional[str] = None,
        user_input: Optional[str] = None,
        **overrides,
    ):
        params = self.resolve_generation_kwargs(
            request_type=request_type,
            pipeline_name=pipeline_name,
            user_input=user_input,
            overrides=overrides,
        )
        return self.brain.stream_chat(messages, **params)

    def execute_with_pipeline(
        self,
        pipeline_name: str,
        prompt: str,
        system_prompt: str | None = None,
        **overrides,
    ):
        params = self.resolve_generation_kwargs(
            pipeline_name=pipeline_name,
            overrides=overrides,
        )
        return self.brain.generate(prompt, system_prompt=system_prompt, **params)

    @classmethod
    def _has_explicit_creative_intent(cls, user_input: Optional[str]) -> bool:
        text = str(user_input or "").strip().lower()
        if not text:
            return False
        return any(keyword in text for keyword in cls._CREATIVE_KEYWORDS)
