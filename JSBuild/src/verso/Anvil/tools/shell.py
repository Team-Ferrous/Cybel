import os
import subprocess
import time
from typing import Any, Optional


class ShellOps:
    """Executes shell commands with timeout and optional audit timeline logging."""

    def __init__(
        self,
        root_dir: str = ".",
        env_info: Optional[dict] = None,
        max_runtime: int = 30,
        audit_db: Any = None,
        session_id: Optional[str] = None,
    ):
        self.root_dir = os.path.abspath(root_dir)
        self.env_info = env_info or {}
        self.max_runtime = max(1, int(max_runtime))
        self.audit_db = audit_db
        self.session_id = session_id

    def _log_timeline(self, event_type: str, payload: dict) -> None:
        if self.audit_db is None:
            return
        try:
            self.audit_db.log_timeline_event(
                session_id=self.session_id,
                event_type=event_type,
                wall_clock=None,
                monotonic_elapsed_ms=int(time.monotonic() * 1000),
                payload=payload,
            )
        except Exception:
            return

    def run_command(
        self,
        command: str,
        cwd: Optional[str] = None,
        use_docker: bool = False,
        max_runtime: Optional[int] = None,
    ) -> str:
        use_cwd = cwd if cwd else self.root_dir
        timeout = max(1, int(max_runtime)) if max_runtime is not None else self.max_runtime

        self._log_timeline(
            "tool:run_command:start",
            {
                "cwd": use_cwd,
                "use_docker": use_docker,
                "timeout": timeout,
            },
        )

        if use_docker:
            from infrastructure.execution.docker_sandbox import get_docker_sandbox

            sandbox = get_docker_sandbox()
            result = sandbox.execute(command, language="bash")
            output = result.stdout
            if result.stderr:
                output += f"\n[STDERR]\n{result.stderr}"
            self._log_timeline(
                "tool:run_command:end",
                {"status": "ok", "docker": True, "output_chars": len(output)},
            )
            return output.strip() if output.strip() else "(No output)"

        env = os.environ.copy()
        if self.env_info and "bin_dir" in self.env_info:
            bin_dir = self.env_info["bin_dir"]
            env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
            env["VIRTUAL_ENV"] = self.env_info.get("venv_dir", "")
            env["PYTHONPATH"] = self.root_dir

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=use_cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )

            output = result.stdout or ""
            if result.stderr:
                output += f"\n[STDERR]\n{result.stderr}"

            self._log_timeline(
                "tool:run_command:end",
                {
                    "status": "ok",
                    "returncode": result.returncode,
                    "output_chars": len(output),
                },
            )
            return output.strip() if output.strip() else "(No output)"
        except subprocess.TimeoutExpired:
            self._log_timeline(
                "tool:run_command:end",
                {"status": "timeout", "timeout": timeout},
            )
            return f"Error: Command timed out after {timeout} seconds."
        except Exception as exc:
            self._log_timeline(
                "tool:run_command:end",
                {"status": "error", "error": str(exc)},
            )
            return f"Error executing command: {exc}"
