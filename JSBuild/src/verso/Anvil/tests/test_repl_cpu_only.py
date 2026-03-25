import importlib
import subprocess
import sys

from config.settings import COCONUT_CONFIG, GPU_CONFIG
from core.reasoning.backends import get_backend
from core.reasoning.backends.native_backend import NativeBackend


GPU_PROBE_COMMANDS = {"nvidia-smi", "rocm-smi", "vulkaninfo", "system_profiler"}


def test_config_is_cpu_only():
    assert GPU_CONFIG["force_cpu"] is True
    assert GPU_CONFIG["enabled"] is False
    assert GPU_CONFIG["n_gpu_layers"] == 0
    assert COCONUT_CONFIG["backend"] == "native"


def test_coconut_auto_backend_is_native():
    backend = get_backend("auto", embedding_dim=64, num_paths=2, steps=1)
    assert isinstance(backend, NativeBackend)


def test_repl_import_does_not_probe_gpu(monkeypatch):
    calls = []

    def fake_run(args, *unused_args, **unused_kwargs):
        cmd = args[0] if isinstance(args, (list, tuple)) and args else str(args)
        calls.append(str(cmd))
        return subprocess.CompletedProcess(args=args, returncode=1, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    sys.modules.pop("cli.repl", None)
    importlib.import_module("cli.repl")

    probed = [cmd for cmd in calls if cmd in GPU_PROBE_COMMANDS]
    assert probed == []
