import os
import yaml
from typing import Any, Dict
from pathlib import Path
from config.settings import MASTER_MODEL, AGENTIC_THINKING
from core.serialization import SerializableMixin  # Import the mixin


class ConfigManager(SerializableMixin):  # Inherit from SerializableMixin
    DEFAULT_CONFIG = {
        "model": MASTER_MODEL,
        "approval_mode": AGENTIC_THINKING.get("approval_mode", "trusted"),
        "max_steps": 25,
        "show_thinking": AGENTIC_THINKING.get("show_thinking", True),
        "temperature": 0.0,
        "seed": 720720,
        "web_search": "enabled",
        "search_engine": "duckduckgo",
    }

    def __init__(self, config_path: str = None, initial_config: Dict[str, Any] = None):
        self.config_path = config_path or os.path.expanduser("~/.anvil/config.yaml")
        if initial_config is None:
            self.config = self.DEFAULT_CONFIG.copy()
            self.load_config()
        else:
            self.config = initial_config.copy()
        self.apply_env_overrides()

    def load_config(self):
        path = Path(self.config_path)
        if path.exists():
            try:
                with open(path, "r") as f:
                    user_config = yaml.safe_load(f) or {}
                    self.config.update(user_config)
            except Exception as e:
                print(f"[WARN] Failed to load config from {self.config_path}: {e}")

    def apply_env_overrides(self):
        if os.environ.get("GRANITE_MODEL"):
            self.config["model"] = os.environ.get("GRANITE_MODEL")
        if os.environ.get("GRANITE_APPROVAL_MODE"):
            self.config["approval_mode"] = os.environ.get("GRANITE_APPROVAL_MODE")

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def set(self, key: str, value: Any):
        self.config[key] = value

    def save(self):
        path = Path(self.config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "w") as f:
                yaml.dump(self.config, f)
        except Exception as e:
            print(f"[ERROR] Failed to save config to {self.config_path}: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the object's state into a dictionary."""
        return {"config_path": self.config_path, "config": self.config}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Deserializes an object from a dictionary."""
        # When deserializing, we pass the 'config' directly to avoid re-loading from file
        # which might overwrite the deserialized state.
        return cls(
            config_path=data.get("config_path"), initial_config=data.get("config")
        )
