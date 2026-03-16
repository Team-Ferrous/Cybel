import os
import time
import venv
from typing import Dict, Optional


class EnvironmentManager:
    """
    Manages the project-local virtual environment and native Saguaro readiness.

    Saguaro is now used in-process from the repository tree (no pip install -e,
    no shelling out to a `saguaro` binary).
    """

    def __init__(self, root_dir: str = "."):
        from config.settings import ENVIRONMENT_CONFIG

        self.root_dir = os.path.abspath(root_dir)
        self.config = ENVIRONMENT_CONFIG

        # Locate anvil root (where this file lives)
        self.anvil_agent_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )

        # Auto-detect dev mode: are we running in anvil's own directory?
        self.is_dev_mode = (
            self.config.get("auto_detect_dev_mode", True)
            and self.root_dir == self.anvil_agent_root
        )

        # Backward/forward compatible config switches
        if self.config.get("use_anvil_venv") is True:
            self.is_dev_mode = True
        if self.config.get("use_granite_venv") is True:
            self.is_dev_mode = True

        # Determine which venv to use
        if self.is_dev_mode:
            self.venv_dir = os.path.join(self.anvil_agent_root, "venv")
        else:
            self.venv_dir = os.path.join(self.root_dir, "venv")

    def get_env_info(self) -> Dict[str, str]:
        """Returns paths to key environment binaries and mode information."""
        if os.name == "nt":
            python_exe = os.path.join(self.venv_dir, "Scripts", "python.exe")
            bin_dir = os.path.join(self.venv_dir, "Scripts")
        else:
            python_exe = os.path.join(self.venv_dir, "bin", "python")
            bin_dir = os.path.join(self.venv_dir, "bin")

        return {
            "root_dir": self.root_dir,
            "venv_dir": self.venv_dir,
            "python": python_exe,
            "bin_dir": bin_dir,
            "saguaro": "native-python-api",
            "anvil_root": self.anvil_agent_root,
            "is_dev_mode": self.is_dev_mode,
            "mode": "development" if self.is_dev_mode else "deployment",
        }

    def ensure_ready(
        self,
        console=None,
        *,
        allow_degraded: bool = True,
    ) -> Dict[str, str]:
        """
        Ensures the venv exists and Saguaro is initialized/indexed natively.
        """
        env_info = self.get_env_info()
        started = time.perf_counter()
        timings = {
            "venv_check_ms": 0.0,
            "saguaro_check_ms": 0.0,
            "environment_ready_ms": 0.0,
        }

        if console and not self.is_dev_mode:
            console.print(
                f"[dim]Mode: {env_info['mode']} | Target: {self.root_dir}[/dim]"
            )

        venv_started = time.perf_counter()
        if not os.path.exists(self.venv_dir):
            if console:
                mode_msg = (
                    "anvil's venv"
                    if self.is_dev_mode
                    else f"local venv at {self.venv_dir}"
                )
                console.print(
                    f"[bold cyan]Creating virtual environment ({mode_msg})...[/bold cyan]"
                )
            venv.create(self.venv_dir, with_pip=True)
        timings["venv_check_ms"] = round(
            (time.perf_counter() - venv_started) * 1000.0,
            3,
        )

        saguaro_started = time.perf_counter()
        saguaro_report = self._ensure_saguaro_ready(
            console=console,
            allow_degraded=allow_degraded,
        )
        timings["saguaro_check_ms"] = round(
            (time.perf_counter() - saguaro_started) * 1000.0,
            3,
        )
        timings["environment_ready_ms"] = round(
            (time.perf_counter() - started) * 1000.0,
            3,
        )
        env_info.update(
            {
                "startup_timings": timings,
                "saguaro_status": saguaro_report.get("status", "unknown"),
                "saguaro_error": saguaro_report.get("error", ""),
                "saguaro_health": saguaro_report.get("health", {}),
                "native_capabilities": self._native_capability_manifest(),
            }
        )
        return env_info

    def _ensure_saguaro_ready(
        self,
        console=None,
        *,
        allow_degraded: bool,
    ) -> Dict[str, object]:
        """Initialize/update Saguaro state via direct Python API."""
        try:
            from saguaro.api import SaguaroAPI

            api = SaguaroAPI(self.root_dir)

            init_result = api.init(force=False)
            if console:
                if init_result.get("status") == "initialized":
                    console.print(
                        "[bold cyan]Initialized Saguaro workspace.[/bold cyan]"
                    )
                elif init_result.get("status") == "already_initialized":
                    console.print(
                        "[bold cyan]Saguaro workspace already initialized.[/bold cyan]"
                    )

            # Snapshot current state, run incremental refresh, then escalate to full reindex only if needed.
            api.chronicle_snapshot()
            index_result = api.index(path=".", force=False)
            health = self._check_index_health(api, fallback=index_result)
            if health["total_entities"] < 100 or health["indexed_files"] < 25:
                if console:
                    console.print(
                        "[yellow]⚠ Saguaro index coverage is low "
                        f"({health['indexed_files']} files, {health['total_entities']} entities). "
                        "Re-indexing...[/yellow]"
                    )
                index_result = self._full_reindex(api)
                health = self._check_index_health(api, fallback=index_result)

            target_file = getattr(self, "_target_file", None)
            if target_file:
                probe = api.query(os.path.basename(target_file), k=5, file=target_file)
                if not probe.get("results"):
                    if console:
                        console.print(
                            f"[yellow]⚠ Target file {target_file} missing from index. Indexing it directly...[/yellow]"
                        )
                    api.index(path=target_file, force=True)

            if console:
                console.print(
                    "[bold cyan]Saguaro index ready "
                    f"({health['indexed_files']} files, "
                    f"{health['total_entities']} entities).[/bold cyan]"
                )
            return {"status": "ready", "health": health}

        except Exception as e:
            if console:
                console.print(
                    "[yellow]Saguaro degraded mode active: "
                    f"{e}[/yellow]"
                )
            if not allow_degraded:
                raise RuntimeError(f"Saguaro strict setup failed: {e}") from e
            return {"status": "degraded", "error": str(e), "health": {}}

    def _check_index_health(
        self, api, fallback: Optional[Dict[str, object]] = None
    ) -> Dict[str, int]:
        report = api.health()
        performance = report.get("performance", {}) if isinstance(report, dict) else {}
        governance = report.get("governance", {}) if isinstance(report, dict) else {}
        fallback = fallback or {}
        fallback_performance = (
            fallback.get("performance", {}) if isinstance(fallback, dict) else {}
        )

        indexed_files = self._first_nonzero_int(
            performance.get("indexed_files"),
            governance.get("total_tracked_files"),
            fallback_performance.get("indexed_files"),
            fallback.get("indexed_files"),
            fallback.get("updated_files"),
            default=0,
        )
        total_entities = self._first_nonzero_int(
            performance.get("indexed_entities"),
            performance.get("total_entities"),
            fallback_performance.get("indexed_entities"),
            fallback_performance.get("total_entities"),
            fallback.get("indexed_entities"),
            fallback.get("total_entities"),
            default=0,
        )

        return {
            "indexed_files": indexed_files,
            "total_entities": total_entities,
        }

    def _full_reindex(self, api) -> Dict[str, int | str | None]:
        return api.index(path=".", force=True)

    def set_target_file(self, file_path: Optional[str]) -> None:
        self._target_file = file_path

    def capture_profile(self) -> Dict[str, object]:
        profile = dict(self.get_env_info())
        profile["native_capabilities"] = self._native_capability_manifest()
        return profile

    @staticmethod
    def _native_capability_manifest() -> Dict[str, object]:
        try:
            from core.native.capability_registry import NativeCapabilityRegistry

            return NativeCapabilityRegistry().build_manifest()
        except Exception as exc:
            return {
                "schema_version": "native_capability_manifest.v1",
                "summary": {
                    "capability_count": 0,
                    "available_count": 0,
                    "degraded_count": 1,
                },
                "capabilities": [
                    {
                        "capability": "registry",
                        "status": "degraded",
                        "fallback_reason": str(exc),
                    }
                ],
            }

    @staticmethod
    def _first_nonzero_int(*values: object, default: int = 0) -> int:
        for value in values:
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                continue
            if parsed > 0:
                return parsed
        return default
