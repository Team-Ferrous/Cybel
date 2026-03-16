"""Fail-closed model lifecycle helpers for the production QSG models."""

from __future__ import annotations

import subprocess
from typing import Any, Dict, List

from config.settings import PRODUCTION_MODEL_ALLOWLIST
from core.model.model_contract import canonicalize_model_name, resolve_model_contract
from core.ollama_client import DeterministicOllama


class ModelManager:
    """Manage the two supported production models only."""

    def __init__(self, models: List[str] | None = None):
        selected = models or list(PRODUCTION_MODEL_ALLOWLIST.keys())
        self.models = [canonicalize_model_name(model) for model in selected]

    def list_models(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for model in self.models:
            try:
                contract = resolve_model_contract(model)
                rows.append(
                    {
                        "name": model,
                        "digest": contract.expected_digest,
                        "size": contract.blob_size,
                        "path": str(contract.blob_path),
                    }
                )
            except Exception:
                rows.append({"name": model, "digest": None, "size": None, "path": None})
        return rows

    def ensure_models_present(self) -> List[str]:
        missing: List[str] = []
        for model in self.models:
            try:
                resolve_model_contract(model)
            except Exception:
                missing.append(model)
                self.pull_model(model)
        return missing

    def pull_model(self, model_name: str):
        model = canonicalize_model_name(model_name)
        try:
            return subprocess.run(
                ["ollama", "pull", model], capture_output=True, text=True, check=False
            )
        except Exception as exc:
            return str(exc)

    def remove_model(self, model_name: str):
        model = canonicalize_model_name(model_name)
        try:
            return subprocess.run(
                ["ollama", "rm", model], capture_output=True, text=True, check=False
            )
        except Exception as exc:
            return str(exc)

    def warm_up(self, model_name: str) -> str:
        model = canonicalize_model_name(model_name)
        client = DeterministicOllama(model)
        return client.generate("hello")

    def benchmark_model(
        self, model_name: str, prompt: str = "Why is the sky blue?"
    ) -> Dict[str, Any]:
        model = canonicalize_model_name(model_name)
        client = DeterministicOllama(model)
        _ = client.generate(prompt)
        return client.runtime_status()

    def select_best_model(self, task_type: str) -> str:
        _ = task_type
        if "qwen3.5:9b" in self.models:
            return "qwen3.5:9b"
        return self.models[0]
