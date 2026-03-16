from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Callable


@dataclass(slots=True)
class TranslationWitness:
    source_digest: str
    target_digest: str
    equivalent: bool
    checked_cases: int
    counterexamples: list[dict[str, Any]] = field(default_factory=list)
    telemetry: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class TranslationValidator:
    """Validate narrow lowerings by replaying deterministic case sets."""

    def validate_callables(
        self,
        source: Callable[..., Any],
        target: Callable[..., Any],
        *,
        cases: list[dict[str, Any]],
    ) -> TranslationWitness:
        counterexamples: list[dict[str, Any]] = []
        for case in cases:
            source_value = source(**case)
            target_value = target(**case)
            if source_value != target_value:
                counterexamples.append(
                    {
                        "inputs": dict(case),
                        "source": source_value,
                        "target": target_value,
                    }
                )
        return TranslationWitness(
            source_digest=self._digest_callable(source),
            target_digest=self._digest_callable(target),
            equivalent=not counterexamples,
            checked_cases=len(cases),
            counterexamples=counterexamples,
            telemetry={
                "translation_validation_pass_rate": 1.0 if not counterexamples else 0.0,
                "ir_mismatch_count": len(counterexamples),
                "undefined_behavior_block_count": 0,
            },
        )

    def validate_expression_pair(
        self,
        source_expr: str,
        target_expr: str,
        *,
        cases: list[dict[str, Any]],
    ) -> TranslationWitness:
        safe_globals = {"min": min, "max": max, "abs": abs}

        def _source(**kwargs: Any) -> Any:
            return eval(source_expr, safe_globals, kwargs)

        def _target(**kwargs: Any) -> Any:
            return eval(target_expr, safe_globals, kwargs)

        return self.validate_callables(_source, _target, cases=cases)

    @staticmethod
    def _digest_callable(fn: Callable[..., Any]) -> str:
        return hashlib.sha256(repr(fn).encode("utf-8")).hexdigest()

