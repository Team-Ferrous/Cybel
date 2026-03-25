"""SAGUARO Sentinel Verifier
Enforces rules against files.
"""

import logging
import os
from typing import Any

from .engines import (
    AESEngine,
    BaseEngine,
    MypyEngine,
    NativeEngine,
    RuffEngine,
    VultureEngine,
)
from .policy import PolicyManager

logger = logging.getLogger(__name__)


class SentinelVerifier:
    """Provide SentinelVerifier support."""
    ENGINE_ORDER = ("native", "ruff", "semantic", "aes", "mypy", "vulture", "graph")

    def __init__(self, repo_path: str, engines: list[str] = None) -> None:
        """Initialize the instance."""
        self.repo_path = os.path.abspath(repo_path)
        self.engines: list[BaseEngine] = []
        self.policy = PolicyManager(self.repo_path)

        # Core engines always available
        engine_map = {
            "native": NativeEngine,
            "ruff": RuffEngine,
            "aes": AESEngine,
            "mypy": MypyEngine,
            "vulture": VultureEngine,
        }

        if engines is None:
            engines = ["native", "ruff", "semantic", "aes"]

        requested = [name.strip() for name in engines if name and name.strip()]
        ordered = self._order_engines(requested)

        for name in ordered:
            if name in engine_map:
                try:
                    self.engines.append(engine_map[name](self.repo_path))
                except Exception as e:
                    logger.warning(f"Failed to initialize engine {name}: {e}")
            elif name == "semantic":
                semantic_cls = self._load_semantic_engine()
                if semantic_cls is None:
                    logger.warning(
                        "Semantic engine unavailable; skipping. "
                        "Install optional semantic dependencies to enable it."
                    )
                    continue
                try:
                    self.engines.append(semantic_cls(self.repo_path))
                except Exception as e:
                    logger.warning(f"Failed to initialize engine {name}: {e}")
            elif name == "graph":
                graph_cls = self._load_graph_engine()
                if graph_cls is None:
                    logger.warning("Graph engine unavailable; skipping.")
                    continue
                try:
                    self.engines.append(graph_cls(self.repo_path))
                except Exception as e:
                    logger.warning(f"Failed to initialize engine {name}: {e}")
            else:
                logger.warning(f"Unknown engine: {name}")

    def verify_all(
        self,
        path_arg: str = ".",
        aal: list[str] | str | None = None,
        domain: list[str] | str | None = None,
        require_trace: bool = False,
        require_evidence: bool = False,
        require_valid_waivers: bool = False,
        change_manifest_path: str | None = None,
        compliance_context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Runs all configured engines and enforces policy."""
        all_violations = []
        verify_context = {
            "aal": aal,
            "domain": domain,
            "require_trace": require_trace,
            "require_evidence": require_evidence,
            "require_valid_waivers": require_valid_waivers,
            "path_arg": path_arg,
            "change_manifest_path": change_manifest_path,
            "compliance_context": compliance_context or {},
        }
        policy_bundle = dict(self.policy.config)
        policy_bundle["verify_context"] = verify_context

        for engine in self.engines:
            # Inject policy
            engine.set_policy(policy_bundle)

            try:
                logger.info(f"Running engine: {engine.__class__.__name__}")
                try:
                    violations = engine.run(path_arg=path_arg)
                except TypeError:
                    # Backward compatibility for engines that haven't adopted path_arg yet.
                    violations = engine.run()
                all_violations.extend(violations)
            except Exception as e:
                logger.error(f"Engine {engine.__class__.__name__} failed: {e}")
                # Fail closed: surface engine failure as a blocking policy violation.
                all_violations.append(
                    {
                        "rule_id": "SENTINEL-ENGINE-FAILURE",
                        "message": f"Engine {engine.__class__.__name__} failed: {e}",
                        "severity": "P0",
                        "closure_level": "blocking",
                        "file": path_arg if path_arg not in {".", "./"} else ".",
                        "line": 0,
                        "domain": ["universal"],
                        "evidence_refs": [],
                        "engine": engine.__class__.__name__,
                    }
                )

        # Apply policy
        scoped_violations = self._filter_violations_for_path(all_violations, path_arg)
        return self.policy.evaluate(scoped_violations)

    @classmethod
    def _order_engines(cls, requested: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []

        for engine_name in cls.ENGINE_ORDER:
            if engine_name in requested and engine_name not in seen:
                ordered.append(engine_name)
                seen.add(engine_name)

        for engine_name in requested:
            if engine_name not in seen:
                ordered.append(engine_name)
                seen.add(engine_name)

        return ordered

    def _filter_violations_for_path(
        self, violations: list[dict[str, Any]], path_arg: str
    ) -> list[dict[str, Any]]:
        excluded_roots = [
            str(item).rstrip("/") + "/"
            for item in self.policy.config.get("excluded_paths", []) or []
        ]
        if not path_arg or path_arg in {".", "./"}:
            return [
                violation
                for violation in violations
                if not self._is_excluded_violation(violation, excluded_roots)
            ]

        target_abs = (
            path_arg
            if os.path.isabs(path_arg)
            else os.path.join(self.repo_path, path_arg)
        )
        target_abs = os.path.abspath(target_abs)

        try:
            rel_target = os.path.relpath(target_abs, self.repo_path).replace("\\", "/")
        except Exception:
            return []

        if rel_target.startswith(".."):
            return []

        if rel_target in {"", "."}:
            return [
                violation
                for violation in violations
                if not self._is_excluded_violation(violation, excluded_roots)
            ]

        is_file_target = os.path.isfile(target_abs)
        prefix = rel_target.rstrip("/") + "/"

        scoped = []
        for violation in violations:
            raw_file = str(violation.get("file", "")).strip()
            if not raw_file:
                continue
            rel_file = raw_file.replace("\\", "/")
            if os.path.isabs(rel_file):
                rel_file = os.path.relpath(rel_file, self.repo_path).replace("\\", "/")

            if rel_file.startswith(".."):
                continue
            if self._is_excluded_violation({"file": rel_file}, excluded_roots):
                continue
            if is_file_target:
                if rel_file == rel_target:
                    scoped.append(violation)
            else:
                if rel_file == rel_target or rel_file.startswith(prefix):
                    scoped.append(violation)

        return scoped

    @staticmethod
    def _is_excluded_violation(
        violation: dict[str, Any], excluded_roots: list[str]
    ) -> bool:
        raw_file = str(violation.get("file", "")).strip().replace("\\", "/")
        if not raw_file:
            return False
        normalized = raw_file.lstrip("./")
        return any(
            normalized == root.rstrip("/") or normalized.startswith(root)
            for root in excluded_roots
        )

    @staticmethod
    def _load_semantic_engine():
        try:
            from .engines.semantic import SemanticEngine

            return SemanticEngine
        except Exception as e:
            logger.warning(f"Unable to import semantic engine: {e}")
            return None

    @staticmethod
    def _load_graph_engine():
        try:
            from .engines.graph import CodeGraphEngine

            return CodeGraphEngine
        except Exception as e:
            logger.warning(f"Unable to import graph engine: {e}")
            return None


# Import at the end to avoid circular imports if any
