"""
Intelligent Loop Router - ML-based loop selection with historical performance tracking.

This router selects the optimal execution loop (simple, run_loop, enhanced, orchestrator)
based on task embeddings, complexity classification, and historical performance data.

Features:
- Semantic task embedding for similarity matching
- Historical performance tracking per loop type
- Task complexity classification (simple/medium/complex/extreme)
- Adaptive routing based on success rates
- Fallback to heuristic-based selection
"""

import logging
import json
from typing import Optional, Dict, List
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from config.settings import AES_CONFIG

logger = logging.getLogger(__name__)


def get_aes_rollout_flags() -> Dict[str, bool]:
    """Expose AES rollout state to routing and future policy gates."""
    return dict(AES_CONFIG)


@dataclass
class TaskPerformanceRecord:
    """Record of a task's execution performance."""

    task_hash: str  # Hash of task embedding
    task_snippet: str  # First 100 chars of task
    loop_type: str  # Which loop was used
    duration_sec: float  # Execution time
    success: bool  # Whether task succeeded
    timestamp: str  # ISO timestamp
    complexity: str  # simple/medium/complex/extreme
    num_steps: int = 0  # Number of steps taken
    error_message: Optional[str] = None  # Error if failed


class PerformanceDatabase:
    """
    Lightweight JSON-based database for tracking loop performance.

    Structure:
    {
        "records": [TaskPerformanceRecord, ...],
        "loop_stats": {
            "simple": {"total": N, "success": M, "avg_duration": X},
            ...
        }
    }
    """

    def __init__(self, db_path: str = ".anvil/loop_performance.json"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._load()

    def _load(self) -> Dict:
        """Load database from disk."""
        if self.db_path.exists():
            try:
                with open(self.db_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load performance DB: {e}. Starting fresh.")

        return {"records": [], "loop_stats": {}}

    def _save(self):
        """Save database to disk."""
        try:
            with open(self.db_path, "w") as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save performance DB: {e}")

    def record(
        self,
        task: str,
        loop_type: str,
        duration: float,
        success: bool,
        complexity: str = "medium",
        num_steps: int = 0,
        error: Optional[str] = None,
    ):
        """Record a task execution."""
        # Create simple hash for task (first 100 chars hashed)
        task_snippet = task[:100]
        task_hash = str(hash(task_snippet))

        record = TaskPerformanceRecord(
            task_hash=task_hash,
            task_snippet=task_snippet,
            loop_type=loop_type,
            duration_sec=duration,
            success=success,
            timestamp=datetime.now().isoformat(),
            complexity=complexity,
            num_steps=num_steps,
            error_message=error,
        )

        self.data["records"].append(asdict(record))

        # Update stats
        self._update_stats()
        self._save()

        logger.debug(
            f"Recorded: {loop_type} - {success} - {duration:.2f}s - {task_snippet}"
        )

    def _update_stats(self):
        """Update aggregated statistics."""
        stats = {}

        for record in self.data["records"]:
            loop = record["loop_type"]
            if loop not in stats:
                stats[loop] = {"total": 0, "success": 0, "total_duration": 0.0}

            stats[loop]["total"] += 1
            if record["success"]:
                stats[loop]["success"] += 1
            stats[loop]["total_duration"] += record["duration_sec"]

        # Calculate derived metrics
        for loop, data in stats.items():
            data["success_rate"] = (
                data["success"] / data["total"] if data["total"] > 0 else 0.0
            )
            data["avg_duration"] = (
                data["total_duration"] / data["total"] if data["total"] > 0 else 0.0
            )

        self.data["loop_stats"] = stats

    def get_stats(self, loop_type: Optional[str] = None) -> Dict:
        """Get statistics for a specific loop or all loops."""
        if loop_type:
            return self.data["loop_stats"].get(loop_type, {})
        return self.data["loop_stats"]

    def find_similar_tasks(self, task_hash: str, limit: int = 5) -> List[Dict]:
        """Find similar tasks by hash (simple similarity for now)."""
        # For now, return recent records with same hash
        similar = [r for r in self.data["records"] if r["task_hash"] == task_hash]
        return similar[-limit:]  # Return most recent


class TaskComplexityClassifier:
    """
    Classifies task complexity based on features.

    Complexity Levels:
    - simple: Questions, read-only operations, < 50 words
    - medium: Single file edits, straightforward implementations
    - complex: Multi-file changes, refactoring, architecture decisions
    - extreme: Large-scale migrations, framework upgrades, system redesign
    """

    # Indicators of complexity
    SIMPLE_KEYWORDS = [
        "what",
        "how",
        "why",
        "explain",
        "show",
        "tell",
        "describe",
        "list",
        "read",
        "analyze",
        "review",
        "summarize",
        "check",
    ]

    COMPLEX_KEYWORDS = [
        "refactor",
        "redesign",
        "migrate",
        "upgrade",
        "restructure",
        "architect",
        "implement feature",
        "add system",
        "integrate",
        "multi",
        "across",
        "entire",
        "all files",
        "codebase-wide",
    ]

    EXTREME_KEYWORDS = [
        "migrate framework",
        "upgrade major version",
        "rewrite",
        "replace architecture",
        "distributed system",
        "microservices",
        "complete overhaul",
        "rearchitect",
        "full rewrite",
    ]

    CAMPAIGN_KEYWORDS = [
        "campaign",
        "phase",
        "multi-phase",
        "end to end",
        "file by file",
        "production readiness",
    ]

    @classmethod
    def classify(cls, task: str) -> str:
        """
        Classify task complexity.

        Returns: "simple", "medium", "complex", or "extreme"
        """
        task_lower = task.lower()
        word_count = len(task.split())

        if any(kw in task_lower for kw in cls.CAMPAIGN_KEYWORDS):
            return "extreme"

        # Extreme complexity
        if any(kw in task_lower for kw in cls.EXTREME_KEYWORDS):
            return "extreme"
        if word_count > 200 and ("entire" in task_lower or "all" in task_lower):
            return "extreme"

        # Complex
        if any(kw in task_lower for kw in cls.COMPLEX_KEYWORDS):
            return "complex"
        if word_count > 100:
            return "complex"

        # Simple
        if any(kw in task_lower for kw in cls.SIMPLE_KEYWORDS):
            return "simple"
        if word_count < 20 and "?" in task:
            return "simple"

        # Default to medium
        return "medium"


class IntelligentLoopRouter:
    """
    Intelligent loop routing with ML-based selection and performance tracking.

    Routing Strategy:
    1. Classify task complexity
    2. Query historical performance for similar tasks
    3. Select loop with best success rate for this complexity
    4. Fall back to heuristics if insufficient data

    Loop Types:
    - simple: Questions, read-only tasks (uses SimpleChat)
    - run_loop: Basic agentic tasks (think-act-observe cycle)
    - enhanced: Complex implementations (EnhancedAgenticLoop with planning)
    - orchestrator: Massive tasks requiring sub-agent delegation
    - enhanced: Complex implementations and debugging tasks
    """

    # Minimum samples required before trusting ML routing
    MIN_SAMPLES_FOR_ML = 3

    # Default loop selection per complexity (fallback)
    DEFAULT_ROUTING = {
        "simple": "simple",
        "medium": "enhanced",
        "complex": "enhanced",
        "extreme": "orchestrator",
    }

    def __init__(self, db_path: str = ".anvil/loop_performance.json"):
        self.classifier = TaskComplexityClassifier()
        self.performance_db = PerformanceDatabase(db_path)

        logger.info(
            f"Intelligent Loop Router initialized. Stats: {self.performance_db.get_stats()}"
        )

    def route(self, user_input: str, force: Optional[str] = None) -> str:
        """
        Select optimal loop type for the given input.

        Args:
            user_input: User's task/question
            force: Force a specific loop type (overrides routing)

        Returns:
            Loop type: "simple", "run_loop", "enhanced", or "orchestrator"
        """
        # Handle force override
        if force:
            logger.info(f"Loop routing forced to: {force}")
            return force

        # Detect debug logs (special case)
        if self._is_debug_log(user_input):
            logger.info("Detected debug log, routing to enhanced loop")
            return "enhanced"

        # Campaign requests should run through orchestrator-level loops.
        if self._is_campaign_request(user_input):
            logger.info("Detected campaign-style request, routing to orchestrator")
            return "orchestrator"

        # 1. Classify complexity
        complexity = self.classifier.classify(user_input)
        logger.debug(f"Task complexity: {complexity}")

        # 2. Check historical performance (ML-based routing)
        loop_type = self._route_by_history(user_input, complexity)

        if loop_type:
            logger.info(f"ML routing selected: {loop_type} (complexity: {complexity})")
            return loop_type

        # 3. Fallback to heuristic routing
        loop_type = self._route_by_heuristics(user_input, complexity)
        logger.info(
            f"Heuristic routing selected: {loop_type} (complexity: {complexity})"
        )
        return loop_type

    def _is_debug_log(self, text: str) -> bool:
        """Detect if input is an error log."""
        indicators = [
            "traceback",
            "error:",
            "exception:",
            "fail",
            "stack trace",
            'file "',
        ]
        return any(kw in text.lower() for kw in indicators) or (
            len(text.splitlines()) > 5 and 'File "' in text
        )

    @staticmethod
    def _is_campaign_request(text: str) -> bool:
        lowered = text.lower()
        signals = [
            "campaign",
            "multi-phase",
            "phase 1",
            "phase 0",
            "file by file",
            "end-to-end",
            "end to end",
            "/campaign",
        ]
        return any(signal in lowered for signal in signals)

    def _route_by_history(self, task: str, complexity: str) -> Optional[str]:
        """
        Route based on historical performance of similar tasks.

        Returns None if insufficient data.
        """
        # Simple task hash for similarity
        task_hash = str(hash(task[:100]))
        similar_tasks = self.performance_db.find_similar_tasks(task_hash)

        # Need minimum samples
        if len(similar_tasks) < self.MIN_SAMPLES_FOR_ML:
            logger.debug(
                f"Insufficient history ({len(similar_tasks)} samples), using heuristics"
            )
            return None

        # Find loop with best success rate among similar tasks
        loop_scores = {}
        for record in similar_tasks:
            loop = record["loop_type"]
            if loop not in loop_scores:
                loop_scores[loop] = {"success": 0, "total": 0}

            loop_scores[loop]["total"] += 1
            if record["success"]:
                loop_scores[loop]["success"] += 1

        # Calculate success rates
        best_loop = None
        best_rate = 0.0
        for loop, scores in loop_scores.items():
            rate = scores["success"] / scores["total"] if scores["total"] > 0 else 0.0
            if rate > best_rate:
                best_rate = rate
                best_loop = loop

        if best_loop and best_rate > 0.5:  # At least 50% success rate
            logger.debug(f"Historical best: {best_loop} ({best_rate:.1%} success rate)")
            return best_loop

        return None

    def _route_by_heuristics(self, task: str, complexity: str) -> str:
        """
        Fallback heuristic routing based on keywords and patterns.
        """
        task_lower = task.lower()

        if self._is_campaign_request(task):
            return "orchestrator"

        # Simple questions
        if any(
            task_lower.startswith(q)
            for q in ["what", "how", "why", "explain", "describe"]
        ):
            return "simple"

        # Very short conversational
        if len(task.split()) < 10 and not any(
            kw in task_lower for kw in ["implement", "create", "build", "fix"]
        ):
            return "simple"

        # Use default routing by complexity
        return self.DEFAULT_ROUTING.get(complexity, "enhanced")

    def record_performance(
        self,
        task: str,
        loop_type: str,
        duration: float,
        success: bool,
        num_steps: int = 0,
        error: Optional[str] = None,
    ):
        """Record performance of a completed task."""
        complexity = self.classifier.classify(task)
        self.performance_db.record(
            task=task,
            loop_type=loop_type,
            duration=duration,
            success=success,
            complexity=complexity,
            num_steps=num_steps,
            error=error,
        )

    def get_stats(self) -> Dict:
        """Get routing statistics."""
        return self.performance_db.get_stats()

    def get_routing_summary(self) -> str:
        """Get human-readable routing summary."""
        stats = self.get_stats()

        if not stats:
            return "No routing history yet."

        lines = ["Loop Routing Statistics:", "=" * 50]
        for loop, data in stats.items():
            success_rate = data.get("success_rate", 0) * 100
            avg_dur = data.get("avg_duration", 0)
            total = data.get("total", 0)
            lines.append(
                f"{loop:12} | {total:3} tasks | {success_rate:5.1f}% success | {avg_dur:6.2f}s avg"
            )

        return "\n".join(lines)
