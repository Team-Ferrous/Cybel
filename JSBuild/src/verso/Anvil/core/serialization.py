from abc import ABC, abstractmethod
from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Dict, Any


class SerializableMixin(ABC):
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serializes the object's state into a dictionary."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Deserializes an object from a dictionary."""
        pass


_SECRET_PATTERNS = (
    (
        re.compile(
            r"(?i)\b(api[_-]?key|secret|token|password)\b\s*[:=]\s*([^\s,;]+)"
        ),
        r"\1=***REDACTED***",
    ),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "***REDACTED_AWS_KEY***"),
    (
        re.compile(r"(?i)authorization:\s*bearer\s+[a-z0-9\-._~+/]+=*"),
        "authorization: bearer ***REDACTED***",
    ),
)


def redact_secret_material(text: str) -> str:
    """Redact obvious credential patterns from persisted content."""
    redacted = text
    for pattern, replacement in _SECRET_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def serialize_tool_provenance(
    tool_name: str, tool_args: Dict[str, Any], trace_id: str | None
) -> Dict[str, Any]:
    """Produce deterministic provenance metadata for tool outputs."""
    args_json = json.dumps(tool_args or {}, sort_keys=True, default=str)
    args_hash = hashlib.sha256(args_json.encode("utf-8")).hexdigest()
    return {
        "tool_name": tool_name,
        "args_hash": args_hash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trace_id": trace_id,
    }
