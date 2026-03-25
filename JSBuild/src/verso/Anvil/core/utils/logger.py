import json
import logging
import os
import time
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional


def setup_logging(
    log_file=".anvil/logs/granite.log", level=logging.INFO, console_level=None
):
    """Sets up unified logging for the project."""
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Root logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler (with rotation)
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5
    )
    file_fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    # Optional Console Handler
    if console_level:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_fmt = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
        console_handler.setFormatter(console_fmt)
        logger.addHandler(console_handler)

    # Silence noisy loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("markdown_it").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("rich").setLevel(logging.WARNING)

    logging.info("Logging initialized.")


def get_active_log_file() -> Optional[str]:
    """Return the active rotating log file path if configured."""
    root = logging.getLogger()
    for handler in root.handlers:
        if isinstance(handler, RotatingFileHandler):
            return handler.baseFilename
    return None


def emit_structured_event(
    logger: logging.Logger,
    component: str,
    event: str,
    *,
    mission_id: Optional[str] = None,
    model: Optional[str] = None,
    phase: Optional[str] = None,
    duration_ms: Optional[int] = None,
    metrics: Optional[Dict[str, Any]] = None,
    level: int = logging.INFO,
) -> None:
    """Emit a structured JSON log event with a standard schema."""
    payload: Dict[str, Any] = {
        "ts_ms": int(time.time() * 1000),
        "component": component,
        "event": event,
    }
    if mission_id:
        payload["mission_id"] = mission_id
    if model:
        payload["model"] = model
    if phase:
        payload["phase"] = phase
    if duration_ms is not None:
        payload["duration_ms"] = int(duration_ms)
    payload["metrics"] = metrics or {}

    try:
        message = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
    except Exception:
        payload["metrics"] = {"serialization_error": True}
        message = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)

    logger.log(level, message)


def get_logger(name):
    """Returns a logger instance for a module."""
    return logging.getLogger(name)
