try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # Fallback for older Python versions
import tomli_w  # For writing (install via: pip install tomli-w)
from   pathlib import Path

CONFIG_PATH = Path("user_config.toml")

def load_config(path=CONFIG_PATH):
    if not path.exists():
        print("No config found, initializing default config.")
        config = {"user": {"name": "Anonymous", "theme": "dark", "notifications": True}}
        save_config(config)
        return config
    with path.open("rb") as f:
        return tomllib.load(f)

def save_config(config, path=CONFIG_PATH):
    with path.open("wb") as f:
        tomli_w.dump(config, f)

def update_config(config, key, value):
    # Naive flat update (assumes top-level key.subkey format)
    keys = key.split(".")
    subconfig = config
    for k in keys[:-1]:
        subconfig = subconfig.setdefault(k, {})
    subconfig[keys[-1]] = value
    return config
