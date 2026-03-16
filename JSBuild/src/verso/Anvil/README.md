# Anvil
Coding Agent for HighNoon

## Saguaro CLI (Directory-Independent)

Install a launcher shim once:

```bash
python scripts/install_cli_shims.py --force
```

Then run from any working directory:

```bash
saguaro index
saguaro query "authentication flow" --k 5
saguaro unwired --format json
```

Behavior:

- `saguaro` operates on the current directory by default.
- Index/state is isolated per directory in `./.saguaro`.
- Override target explicitly with `--repo /path/to/repo` when needed.


./scripts/run_native_qsg_suite.sh --profile silver

./venv/bin/saguaro corpus create repo_analysis/HighNoon-Language-Framework-Dev --alias validation-highnoon