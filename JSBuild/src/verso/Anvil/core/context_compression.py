"""
Runtime context compression helpers.

Implements piggybacked tool-result compression via `_context_updates`
and `[tcN]` addressable tool-result labels.
"""

from __future__ import annotations

import copy
import math
import re
import time
from typing import Any, Callable, Dict, Iterable, List, MutableMapping, Optional


CONTEXT_UPDATES_KEY = "_context_updates"
TC_LABEL_PATTERN = re.compile(r"^\s*\[tc(?P<id>\d+)\]\s*")
TC_REF_PATTERN = re.compile(r"^\s*\[?tc(?P<id>\d+)\]?\s*$", re.IGNORECASE)
WORD_PATTERN = re.compile(r"[a-zA-Z_][a-zA-Z0-9_./-]*")

CONTEXT_UPDATES_PARAM: Dict[str, Any] = {
    "type": "array",
    "description": (
        "REQUIRED. Pass [] if nothing to compress. Otherwise provide summaries for "
        "older tool results by tc id, e.g. [{\"tc1\": \"summary\"}]. Only summarize "
        "[tcN] results you no longer need in full. Results without [tcN] are already "
        "compressed; do not re-compress them."
    ),
    "default": [],
    "items": {
        "type": "object",
        "additionalProperties": {"type": "string"},
    },
}


def parse_tc_id(value: Any) -> Optional[int]:
    """Parse tc identifiers from `tc3` or `[tc3]` strings."""
    if isinstance(value, int):
        return value if value > 0 else None
    if not isinstance(value, str):
        return None
    match = TC_REF_PATTERN.match(value.strip())
    if not match:
        return None
    tc_id = int(match.group("id"))
    return tc_id if tc_id > 0 else None


def find_tc_id_in_message(message: Dict[str, Any]) -> Optional[int]:
    """Find tc id using metadata first, then `[tcN]` prefix in content."""
    tc_id = message.get("tc_id")
    if isinstance(tc_id, int) and tc_id > 0:
        return tc_id

    content = str(message.get("content", ""))
    match = TC_LABEL_PATTERN.match(content)
    if match:
        return int(match.group("id"))

    compressed_from = message.get("compressed_from_tc")
    return parse_tc_id(compressed_from)


def strip_tc_label(content: str) -> str:
    """Remove a leading `[tcN]` label."""
    if not content:
        return content
    return TC_LABEL_PATTERN.sub("", content, count=1).lstrip()


def label_tool_result(content: str, tc_id: int) -> str:
    """Prefix tool result content with `[tcN]`."""
    raw = str(content or "")
    if TC_LABEL_PATTERN.match(raw):
        return raw
    return f"[tc{tc_id}] {raw}"


def infer_next_tc_id(messages: Iterable[Dict[str, Any]]) -> int:
    """Return the next monotonic tc id for the current conversation."""
    max_id = 0
    for msg in messages:
        tc_id = find_tc_id_in_message(msg)
        if tc_id and tc_id > max_id:
            max_id = tc_id
    return max_id + 1


def build_tc_id_map(messages: List[Dict[str, Any]]) -> Dict[int, int]:
    """Map tc id to message index."""
    mapping: Dict[int, int] = {}
    for idx, msg in enumerate(messages):
        tc_id = find_tc_id_in_message(msg)
        if tc_id is not None:
            mapping[tc_id] = idx
    return mapping


def normalize_context_updates(updates: Any) -> List[Dict[str, str]]:
    """Normalize `_context_updates` payload to [{tcN: summary}, ...]."""
    normalized: List[Dict[str, str]] = []
    if not isinstance(updates, list):
        return normalized

    for item in updates:
        if not isinstance(item, dict):
            continue
        clean_item: Dict[str, str] = {}
        for key, value in item.items():
            tc_id = parse_tc_id(key)
            if tc_id is None:
                continue
            summary = str(value or "").strip()
            if not summary:
                continue
            clean_item[f"tc{tc_id}"] = summary
        if clean_item:
            normalized.append(clean_item)
    return normalized


def ensure_context_updates_arg(tool_args: MutableMapping[str, Any]) -> None:
    """Ensure `_context_updates` key exists."""
    if CONTEXT_UPDATES_KEY not in tool_args:
        tool_args[CONTEXT_UPDATES_KEY] = []


def extract_context_updates(tool_args: MutableMapping[str, Any]) -> List[Dict[str, str]]:
    """Pop and normalize `_context_updates` from tool args."""
    if not isinstance(tool_args, MutableMapping):
        return []
    raw = tool_args.pop(CONTEXT_UPDATES_KEY, [])
    return normalize_context_updates(raw)


def inject_context_updates_param(tool_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Inject required `_context_updates` parameter into a tool schema."""
    schema = copy.deepcopy(tool_schema)
    params = schema.setdefault("parameters", {"type": "object"})
    if params.get("type") != "object":
        params["type"] = "object"
    properties = params.setdefault("properties", {})
    properties[CONTEXT_UPDATES_KEY] = copy.deepcopy(CONTEXT_UPDATES_PARAM)
    required = params.setdefault("required", [])
    if CONTEXT_UPDATES_KEY not in required:
        required.append(CONTEXT_UPDATES_KEY)
    return schema


def inject_context_updates_into_all(tool_schemas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Inject `_context_updates` into every tool schema."""
    return [inject_context_updates_param(schema) for schema in tool_schemas]


def apply_context_updates(
    messages: List[Dict[str, Any]],
    updates: List[Dict[str, str]],
    tc_id_map: Optional[Dict[int, int]] = None,
    on_compressed: Optional[Callable[[int, Dict[str, Any], str], None]] = None,
) -> Dict[str, Any]:
    """
    Apply LLM-provided summaries to `[tcN]` tool result messages.

    Returns dict with `applied`, `missing`, and `skipped`.
    """
    normalized = normalize_context_updates(updates)
    if not normalized:
        return {"applied": 0, "missing": [], "skipped": 0}

    mapping = tc_id_map or build_tc_id_map(messages)
    applied = 0
    skipped = 0
    missing: List[str] = []

    for update in normalized:
        for tc_label, summary in update.items():
            tc_id = parse_tc_id(tc_label)
            if tc_id is None:
                skipped += 1
                continue

            idx = mapping.get(tc_id)
            if idx is None:
                missing.append(f"tc{tc_id}")
                continue

            msg = messages[idx]
            msg["content"] = summary
            msg["is_compressed"] = True
            msg["compression_summary"] = summary
            msg["compressed_at"] = time.time()
            msg["compressed_from_tc"] = f"tc{tc_id}"
            msg["tc_id"] = tc_id

            if on_compressed:
                on_compressed(tc_id, msg, summary)

            applied += 1

    strip_tc_labels_from_compressed(messages)
    return {"applied": applied, "missing": missing, "skipped": skipped}


def strip_tc_labels_from_compressed(messages: List[Dict[str, Any]]) -> None:
    """Ensure compressed messages no longer carry `[tcN]` prefixes."""
    for msg in messages:
        if msg.get("is_compressed"):
            msg["content"] = strip_tc_label(str(msg.get("content", "")))


def lexical_relevance_score(task: str, text: str) -> float:
    """Simple lexical overlap score in [0, 1]."""
    task_tokens = {t.lower() for t in WORD_PATTERN.findall(task or "")}
    text_tokens = {t.lower() for t in WORD_PATTERN.findall(text or "")}
    if not task_tokens or not text_tokens:
        return 0.0
    overlap = task_tokens.intersection(text_tokens)
    return len(overlap) / max(len(task_tokens), 1)


def embedding_relevance_score(task: str, text: str, semantic_engine: Any = None) -> float:
    """
    Best-effort embedding similarity score.
    Falls back to lexical overlap if embeddings are unavailable.
    """
    if not task or not text:
        return 0.0

    brain = getattr(semantic_engine, "brain", None) if semantic_engine else None
    embed_fn = getattr(brain, "get_embeddings", None)
    if not callable(embed_fn):
        return lexical_relevance_score(task, text)

    try:
        vec_a = embed_fn(task)
        vec_b = embed_fn(text[:12000])
        if not vec_a or not vec_b:
            return lexical_relevance_score(task, text)

        dot = sum(float(a) * float(b) for a, b in zip(vec_a, vec_b))
        mag_a = math.sqrt(sum(float(a) * float(a) for a in vec_a))
        mag_b = math.sqrt(sum(float(b) * float(b) for b in vec_b))
        if mag_a <= 0 or mag_b <= 0:
            return lexical_relevance_score(task, text)
        return max(0.0, min(1.0, dot / (mag_a * mag_b)))
    except Exception:
        return lexical_relevance_score(task, text)


def find_low_relevance_tc_ids(
    task: str,
    messages: List[Dict[str, Any]],
    semantic_engine: Any = None,
    threshold: float = 0.12,
) -> List[str]:
    """Identify low-relevance uncompressed `[tcN]` tool results."""
    low_relevance: List[str] = []
    for msg in messages:
        if msg.get("role") != "tool" or msg.get("is_compressed"):
            continue
        tc_id = find_tc_id_in_message(msg)
        if tc_id is None:
            continue
        score = embedding_relevance_score(task, str(msg.get("content", "")), semantic_engine)
        if score < threshold:
            low_relevance.append(f"tc{tc_id}")
    return low_relevance


def _was_path_referenced(path: str, later_messages: List[Dict[str, Any]]) -> bool:
    if not path:
        return False
    basename = path.split("/")[-1]
    stem = basename.rsplit(".", 1)[0]
    probes = []
    for probe in [path.lower(), basename.lower(), stem.lower()]:
        if "/" in probe or len(probe) >= 3:
            probes.append(probe)
    for msg in later_messages:
        content = str(msg.get("content", "")).lower()
        for probe in probes:
            if "/" in probe:
                if probe in content:
                    return True
            else:
                if re.search(rf"\b{re.escape(probe)}\b", content):
                    return True
    return False


def auto_compress_dead_end_reads(
    messages: List[Dict[str, Any]],
    min_age_messages: int = 6,
    on_compressed: Optional[Callable[[int, Dict[str, Any], str], None]] = None,
) -> List[str]:
    """
    Auto-compress stale `read_file`-style tool results that were never referenced.
    """
    compressed: List[str] = []
    total = len(messages)
    for idx, msg in enumerate(messages):
        if msg.get("role") != "tool" or msg.get("is_compressed"):
            continue

        tc_id = find_tc_id_in_message(msg)
        if tc_id is None:
            continue

        tool_name = str(msg.get("tool_name", ""))
        if tool_name not in {"read_file", "read_files", "skeleton", "slice"}:
            continue

        if total - idx - 1 < min_age_messages:
            continue

        args = msg.get("tool_args", {}) or {}
        path = args.get("path") or args.get("file_path") or args.get("target")
        later_messages = messages[idx + 1 :]

        if _was_path_referenced(str(path or ""), later_messages):
            continue

        summary = f"{tool_name} on {path or 'target'} had no later references."
        msg["content"] = summary
        msg["is_compressed"] = True
        msg["compression_summary"] = summary
        msg["compressed_at"] = time.time()
        msg["compressed_from_tc"] = f"tc{tc_id}"
        msg["tc_id"] = tc_id
        strip_tc_labels_from_compressed([msg])
        compressed.append(f"tc{tc_id}")
        if on_compressed:
            on_compressed(tc_id, msg, summary)

    return compressed
