from __future__ import annotations

import json
import math
import os
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.native.native_qsg_engine import NativeQSGEngine


def _log_softmax(logits: list[float]) -> list[float]:
    if not logits:
        return []
    anchor = max(float(v) for v in logits)
    shifted = [math.exp(float(v) - anchor) for v in logits]
    denom = max(sum(shifted), 1.0e-30)
    log_denom = math.log(denom)
    return [float(v) - anchor - log_denom for v in logits]


def _softmax(logits: list[float]) -> list[float]:
    return [math.exp(v) for v in _log_softmax(logits)]


def _entropy(probs: Iterable[float]) -> float:
    value = 0.0
    for prob in probs:
        p = float(prob)
        if p > 0.0:
            value -= p * math.log(p)
    return value


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    idx = min(
        len(ordered) - 1,
        max(0, int(round((percentile / 100.0) * (len(ordered) - 1)))),
    )
    return ordered[idx]


def _mean(values: Iterable[float]) -> float:
    values_list = [float(value) for value in values]
    return (sum(values_list) / float(len(values_list))) if values_list else 0.0


@contextmanager
def _patched_env(overrides: dict[str, str] | None):
    original: dict[str, str | None] = {}
    try:
        for key, value in (overrides or {}).items():
            original[key] = os.environ.get(key)
            os.environ[str(key)] = str(value)
        yield
    finally:
        for key, previous in original.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


@dataclass
class NativeLogitScorer:
    model: str
    context_length: int = 2048
    env_overrides: dict[str, str] | None = None

    def __post_init__(self) -> None:
        self._env_ctx = _patched_env(self.env_overrides)
        self._env_ctx.__enter__()
        self.engine = NativeQSGEngine(
            self.model, context_length=int(self.context_length)
        )

    def close(self) -> None:
        try:
            close = getattr(self.engine, "close", None)
            if callable(close):
                close()
        finally:
            self._env_ctx.__exit__(None, None, None)

    def reset(self) -> None:
        self.engine.reset_kv_cache()

    def tokenize(self, text: str) -> list[int]:
        return list(self.engine.tokenize(text))

    def score_sequence(
        self, tokens: list[int], *, reset: bool = True
    ) -> dict[str, Any]:
        if reset:
            self.reset()
        if len(tokens) <= 1:
            return {
                "token_count": 0,
                "logprobs": [],
                "confidences": [],
                "entropies": [],
                "nll": 0.0,
                "correctness": [],
            }
        logprobs: list[float] = []
        confidences: list[float] = []
        entropies: list[float] = []
        correctness: list[int] = []
        for idx in range(1, len(tokens)):
            prefix = list(tokens[:idx])
            target = int(tokens[idx])
            logits = [float(v) for v in self.engine._get_logits_for_tokens(prefix)]
            if target >= len(logits):
                continue
            log_probs = _log_softmax(logits)
            probs = [math.exp(v) for v in log_probs]
            conf = float(probs[target])
            prediction = int(max(range(len(log_probs)), key=lambda i: log_probs[i]))
            logprobs.append(float(log_probs[target]))
            confidences.append(conf)
            entropies.append(_entropy(probs))
            correctness.append(1 if prediction == target else 0)
        return {
            "token_count": len(logprobs),
            "logprobs": logprobs,
            "confidences": confidences,
            "entropies": entropies,
            "nll": -sum(logprobs),
            "correctness": correctness,
        }

    def score_continuation(
        self,
        prefix_tokens: list[int],
        continuation_tokens: list[int],
        *,
        reset: bool = True,
    ) -> dict[str, Any]:
        if reset:
            self.reset()
        if not continuation_tokens:
            return {
                "token_count": 0,
                "logprobs": [],
                "confidences": [],
                "entropies": [],
                "nll": 0.0,
                "correctness": [],
            }
        context = list(prefix_tokens)
        logprobs: list[float] = []
        confidences: list[float] = []
        entropies: list[float] = []
        correctness: list[int] = []
        for token in continuation_tokens:
            logits = [float(v) for v in self.engine._get_logits_for_tokens(context)]
            target = int(token)
            if target >= len(logits):
                context.append(target)
                continue
            log_probs = _log_softmax(logits)
            probs = [math.exp(v) for v in log_probs]
            prediction = int(max(range(len(log_probs)), key=lambda idx: log_probs[idx]))
            logprobs.append(float(log_probs[target]))
            confidences.append(float(probs[target]))
            entropies.append(_entropy(probs))
            correctness.append(1 if prediction == target else 0)
            context.append(target)
        return {
            "token_count": len(logprobs),
            "logprobs": logprobs,
            "confidences": confidences,
            "entropies": entropies,
            "nll": -sum(logprobs),
            "correctness": correctness,
        }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        payload = json.loads(raw)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def evaluate_perplexity(
    *,
    model: str,
    corpus_path: Path,
    context_length: int,
    env_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    scorer = NativeLogitScorer(
        model=model, context_length=context_length, env_overrides=env_overrides
    )
    try:
        rows = load_jsonl(corpus_path)
        total_nll = 0.0
        total_tokens = 0
        docs: list[dict[str, Any]] = []
        for item in rows:
            text = str(item.get("text") or "")
            tokens = scorer.tokenize(text)
            scored = scorer.score_sequence(tokens, reset=True)
            token_count = int(scored["token_count"])
            nll = float(scored["nll"])
            total_nll += nll
            total_tokens += token_count
            docs.append(
                {
                    "sample_id": item.get("sample_id"),
                    "domain": item.get("domain"),
                    "tokens_scored": token_count,
                    "nll": nll,
                }
            )
        cross_entropy = (total_nll / float(total_tokens)) if total_tokens else 0.0
        ppl = math.exp(cross_entropy) if total_tokens else 0.0
        return {
            "schema_version": "native_qsg_eval.perplexity.v1",
            "model": model,
            "corpus_path": str(corpus_path),
            "tokens_scored": total_tokens,
            "nll": total_nll,
            "cross_entropy": cross_entropy,
            "perplexity": ppl,
            "documents": docs,
        }
    finally:
        scorer.close()


def _expected_calibration_error(
    confidences: list[float], correctness: list[int], bins: int = 10
) -> float:
    if not confidences or not correctness:
        return 0.0
    total = len(confidences)
    ece = 0.0
    for bucket in range(bins):
        low = bucket / float(bins)
        high = (bucket + 1) / float(bins)
        members = [
            idx
            for idx, conf in enumerate(confidences)
            if low <= conf < high or (bucket == bins - 1 and conf == 1.0)
        ]
        if not members:
            continue
        acc = sum(correctness[idx] for idx in members) / float(len(members))
        mean_conf = sum(confidences[idx] for idx in members) / float(len(members))
        ece += abs(acc - mean_conf) * (len(members) / float(total))
    return ece


def _format_mcq_prompt(prompt: str, options: dict[str, str]) -> str:
    lines = [prompt.rstrip()]
    for label, text in sorted(options.items()):
        lines.append(f"{label}. {text}")
    lines.append("Answer:")
    return "\n".join(line for line in lines if line)


def _free_text_confidence_record(
    scorer: NativeLogitScorer,
    item: dict[str, Any],
) -> tuple[dict[str, Any], list[float], list[float], list[int], int]:
    text = str(item.get("text") or "")
    tokens = scorer.tokenize(text)
    scored = scorer.score_sequence(tokens, reset=True)
    confidences = [float(v) for v in list(scored["confidences"])]
    entropies = [float(v) for v in list(scored["entropies"])]
    correctness = [int(v) for v in list(scored["correctness"])]
    record = {
        "sample_id": item.get("sample_id"),
        "domain": item.get("domain"),
        "format": "free_text",
        "tokens_scored": int(scored["token_count"]),
        "mean_confidence": _mean(confidences),
        "confidence_p50": _percentile(confidences, 50.0),
        "confidence_p95": _percentile(confidences, 95.0),
        "mean_entropy": _mean(entropies),
        "entropy_p95": _percentile(entropies, 95.0),
    }
    return record, confidences, entropies, correctness, int(scored["token_count"])


def _mcq_confidence_record(
    scorer: NativeLogitScorer,
    item: dict[str, Any],
) -> tuple[dict[str, Any], list[float], list[float], list[int], int]:
    prompt = str(item.get("prompt") or "")
    options = {
        str(label): str(text)
        for label, text in dict(item.get("options") or {}).items()
        if str(label).strip() and str(text).strip()
    }
    prompt_tokens = scorer.tokenize(_format_mcq_prompt(prompt, options))
    option_labels = sorted(options)
    option_scores: dict[str, float] = {}
    option_token_counts: dict[str, int] = {}
    token_count = 0
    for label in option_labels:
        continuation_tokens = scorer.tokenize(options[label])
        scored = scorer.score_continuation(
            prompt_tokens,
            continuation_tokens,
            reset=True,
        )
        continuation_count = int(scored["token_count"])
        token_count += continuation_count
        option_token_counts[label] = continuation_count
        if continuation_count <= 0:
            option_scores[label] = float("-inf")
            continue
        option_scores[label] = -float(scored["nll"]) / float(continuation_count)
    valid_scores = [score for score in option_scores.values() if math.isfinite(score)]
    if valid_scores:
        anchor = max(valid_scores)
        weights = {
            label: (math.exp(score - anchor) if math.isfinite(score) else 0.0)
            for label, score in option_scores.items()
        }
        denom = max(sum(weights.values()), 1.0e-30)
        option_probs = {
            label: float(weight / denom) for label, weight in weights.items()
        }
        predicted_option = max(option_probs, key=option_probs.get)
        confidence = float(option_probs[predicted_option])
        entropy = _entropy(option_probs.values())
    else:
        option_probs = {label: 0.0 for label in option_labels}
        predicted_option = ""
        confidence = 0.0
        entropy = 0.0
    correct_option = str(item.get("correct_option") or "").strip()
    correctness = [1 if predicted_option == correct_option and predicted_option else 0]
    record = {
        "sample_id": item.get("sample_id"),
        "domain": item.get("domain"),
        "format": "mcq",
        "tokens_scored": token_count,
        "mean_confidence": confidence,
        "confidence_p50": confidence,
        "confidence_p95": confidence,
        "mean_entropy": entropy,
        "entropy_p95": entropy,
        "predicted_option": predicted_option,
        "correct_option": correct_option,
        "correct": bool(correctness[0]),
        "option_probabilities": option_probs,
        "option_token_counts": option_token_counts,
    }
    return record, [confidence], [entropy], correctness, token_count


def evaluate_confidence(
    *,
    model: str,
    corpus_path: Path,
    context_length: int,
    env_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    scorer = NativeLogitScorer(
        model=model, context_length=context_length, env_overrides=env_overrides
    )
    try:
        rows = load_jsonl(corpus_path)
        all_confidences: list[float] = []
        all_entropies: list[float] = []
        all_correctness: list[int] = []
        records: list[dict[str, Any]] = []
        total_tokens_scored = 0
        for item in rows:
            if item.get("text") is not None:
                record, confidences, entropies, correctness, tokens_scored = (
                    _free_text_confidence_record(scorer, item)
                )
            else:
                record, confidences, entropies, correctness, tokens_scored = (
                    _mcq_confidence_record(scorer, item)
                )
            all_confidences.extend(confidences)
            all_entropies.extend(entropies)
            all_correctness.extend(correctness)
            total_tokens_scored += int(tokens_scored)
            records.append(record)
        mean_conf = _mean(all_confidences)
        mean_entropy = _mean(all_entropies)
        return {
            "schema_version": "native_qsg_eval.confidence.v1",
            "model": model,
            "corpus_path": str(corpus_path),
            "tokens_scored": total_tokens_scored,
            "mean_token_confidence": mean_conf,
            "p50_token_confidence": _percentile(all_confidences, 50.0),
            "p95_token_confidence": _percentile(all_confidences, 95.0),
            "mean_entropy": mean_entropy,
            "entropy_p95": _percentile(all_entropies, 95.0),
            "expected_calibration_error": _expected_calibration_error(
                all_confidences,
                all_correctness,
            ),
            "records": records,
            "documents": records,
        }
    finally:
        scorer.close()


def _finalize_generated_text(
    scorer: NativeLogitScorer,
    generated_tokens: list[int],
) -> tuple[str, str]:
    raw_generated = str(scorer.engine.detokenize(generated_tokens))
    decode_generated_tokens = getattr(scorer.engine, "decode_generated_tokens", None)
    if callable(decode_generated_tokens):
        return str(decode_generated_tokens(generated_tokens)), raw_generated
    finalize_response_text = getattr(scorer.engine, "finalize_response_text", None)
    if callable(finalize_response_text):
        return str(finalize_response_text(raw_generated)), raw_generated
    return raw_generated, raw_generated


def _normalize_answer(text: str) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "").strip().lower())
    normalized = normalized.strip(" .,:;!?\"'`()[]{}")
    return normalized


def _extract_option_letter(text: str) -> str:
    match = re.search(r"\b([A-Z])\b", str(text or "").upper())
    return str(match.group(1) if match else "")


def evaluate_accuracy(
    *,
    model: str,
    corpus_path: Path,
    context_length: int,
    env_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    scorer = NativeLogitScorer(
        model=model, context_length=context_length, env_overrides=env_overrides
    )
    try:
        rows = load_jsonl(corpus_path)
        records: list[dict[str, Any]] = []
        pass_count = 0
        exact_match_count = 0
        contains_match_count = 0
        option_match_count = 0
        generated_tokens_total = 0
        for item in rows:
            prompt = str(item.get("prompt") or "")
            prepare_prompt_tokens = getattr(
                scorer.engine, "prepare_prompt_tokens", None
            )
            if callable(prepare_prompt_tokens):
                prompt_tokens = list(prepare_prompt_tokens(prompt))
            else:
                prompt_tokens = scorer.tokenize(prompt)
            scorer.reset()
            max_new_tokens = int(item.get("max_new_tokens", 8) or 8)
            output_tokens = scorer.engine.generate(
                prompt_tokens=list(prompt_tokens),
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
            )
            completion_tokens = list(output_tokens[len(prompt_tokens) :])
            generated_tokens_total += len(completion_tokens)
            generated_text, raw_generated_text = _finalize_generated_text(
                scorer, completion_tokens
            )
            grading = str(item.get("grading") or "exact").strip().lower()
            expected_answer = str(item.get("expected_answer") or "")
            expected_norm = _normalize_answer(expected_answer)
            predicted_norm = _normalize_answer(generated_text)
            alias_norms = [
                _normalize_answer(alias)
                for alias in list(item.get("aliases") or [])
                if _normalize_answer(alias)
            ]
            exact_match = bool(expected_norm) and predicted_norm == expected_norm
            contains_match = bool(expected_norm) and expected_norm in predicted_norm
            alias_match = any(alias in predicted_norm for alias in alias_norms)
            expected_option = _extract_option_letter(
                expected_answer
            ) or _extract_option_letter(" ".join(alias_norms))
            predicted_option = _extract_option_letter(generated_text)
            option_match = bool(expected_option) and predicted_option == expected_option
            if grading == "option_letter":
                passed = option_match or exact_match
            elif grading == "contains":
                passed = contains_match or alias_match or exact_match
            else:
                passed = exact_match or contains_match or alias_match
            pass_count += int(passed)
            exact_match_count += int(exact_match)
            contains_match_count += int(contains_match or alias_match)
            option_match_count += int(option_match)
            records.append(
                {
                    "sample_id": item.get("sample_id"),
                    "grading": grading,
                    "expected_answer": expected_answer,
                    "generated_text": generated_text,
                    "raw_generated_text": raw_generated_text,
                    "normalized_prediction": predicted_norm,
                    "normalized_expected": expected_norm,
                    "predicted_option": predicted_option,
                    "expected_option": expected_option,
                    "exact_match": exact_match,
                    "contains_match": contains_match or alias_match,
                    "option_match": option_match,
                    "pass": passed,
                    "generated_token_count": len(completion_tokens),
                }
            )
        sample_count = len(records)
        return {
            "schema_version": "native_qsg_eval.accuracy.v1",
            "model": model,
            "corpus_path": str(corpus_path),
            "samples": sample_count,
            "generated_tokens": generated_tokens_total,
            "pass_count": pass_count,
            "pass_rate": (pass_count / float(sample_count)) if sample_count else 0.0,
            "exact_match_rate": (
                exact_match_count / float(sample_count) if sample_count else 0.0
            ),
            "contains_match_rate": (
                contains_match_count / float(sample_count) if sample_count else 0.0
            ),
            "option_match_rate": (
                option_match_count / float(sample_count) if sample_count else 0.0
            ),
            "records": records,
            "documents": records,
        }
    finally:
        scorer.close()
