"""Utilities for profiling."""

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager

logger = logging.getLogger("saguaro.profiler")


class Profiler:
    """Provide Profiler support."""
    def __init__(self, threshold_ms: float = 100.0) -> None:
        """Initialize the instance."""
        self.threshold = threshold_ms
        self.stats: dict[str, float] = {}

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        """Handle measure."""
        start = time.time()
        try:
            yield
        finally:
            duration = (time.time() - start) * 1000
            self.stats[name] = duration
            if duration > self.threshold:
                logger.warning(f"SLOW OP [{name}]: {duration:.2f}ms")


profiler = Profiler()
