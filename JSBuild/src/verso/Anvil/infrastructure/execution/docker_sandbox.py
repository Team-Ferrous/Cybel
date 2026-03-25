import time
from typing import List
from dataclasses import dataclass

try:
    import docker
except ImportError:  # pragma: no cover - optional dependency in test envs
    docker = None


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    exit_code: int
    success: bool
    duration: float


class DockerSandbox:
    """
    Isolated execution environment for running untrusted code safely.
    Provides resource limits, network isolation, and read-only filesystems.
    """

    def __init__(self):
        try:
            if docker is None:
                raise RuntimeError("docker dependency not installed")
            self.client = docker.from_env()
            self.available = True
        except Exception:
            self.available = False
            # print("Warning: Docker not available. Sandboxing will be disabled.")

    def execute(
        self, code: str, language: str = "python", timeout: int = 30
    ) -> ExecutionResult:
        """
        Execute code in an isolated container.
        """
        if not self.available:
            return ExecutionResult("", "Docker not available", 1, False, 0.0)

        # Select appropriate image
        images = {
            "python": "python:3.11-slim",
            "javascript": "node:20-slim",
            "bash": "debian:bullseye-slim",
        }
        image = images.get(language, "python:3.11-slim")

        # Determine start time
        start_time = time.time()

        try:
            # Create container with resource limits
            container = self.client.containers.run(
                image=image,
                command=self._get_command(code, language),
                detach=True,
                mem_limit="256m",  # 256MB RAM limit
                cpu_period=100000,
                cpu_quota=50000,  # 50% CPU limit
                network_disabled=True,  # No network access
                read_only=True,  # Read-only filesystem
                tmpfs={"/tmp": "size=10m"},  # 10MB temp storage
            )

            try:
                # Wait for completion with timeout
                result = container.wait(timeout=timeout)

                # Get output
                stdout = container.logs(stdout=True, stderr=False).decode()
                stderr = container.logs(stdout=False, stderr=True).decode()

                duration = time.time() - start_time

                return ExecutionResult(
                    stdout=stdout,
                    stderr=stderr,
                    exit_code=result["StatusCode"],
                    success=result["StatusCode"] == 0,
                    duration=duration,
                )
            finally:
                container.remove(force=True)

        except Exception as e:
            return ExecutionResult(
                "",
                f"Sandbox execution error: {str(e)}",
                1,
                False,
                time.time() - start_time,
            )

    def _get_command(self, code: str, language: str) -> List[str]:
        """Convert code to execution command for the specific language."""
        if language == "python":
            return ["python", "-c", code]
        elif language == "javascript":
            return ["node", "-e", code]
        elif language == "bash":
            return ["bash", "-c", code]
        return ["python", "-c", code]


# Singleton
_sandbox = None


def get_docker_sandbox() -> DockerSandbox:
    global _sandbox
    if _sandbox is None:
        _sandbox = DockerSandbox()
    return _sandbox
