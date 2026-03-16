from __future__ import annotations

import argparse
import json
import random
import string
import tempfile
import time
from pathlib import Path

from core.aes.rule_registry import AESRuleRegistry


ALPHABET = string.ascii_letters + string.digits + "{}[],:\"' \\n\\t"


def _random_text(rng: random.Random, max_len: int = 256) -> str:
    length = rng.randint(1, max_len)
    return "".join(rng.choice(ALPHABET) for _ in range(length))


def _random_rule(rng: random.Random, i: int) -> dict[str, object]:
    return {
        "id": f"FZ-{i}",
        "section": "fz",
        "text": _random_text(rng, 40),
        "severity": rng.choice(["AAL-0", "AAL-1", "AAL-2", "AAL-3"]),
        "engine": rng.choice(["agent", "native", "semantic", "ruff"]),
        "auto_fixable": rng.choice([True, False]),
        "domain": [rng.choice(["universal", "ml", "hpc", "physics", "quantum"])],
        "language": [rng.choice(["python", "json", "yaml", "c++"])],
    }


def run(seconds: float, seed: int) -> int:
    rng = random.Random(seed)
    deadline = time.monotonic() + seconds
    iterations = 0
    registry = AESRuleRegistry()
    with tempfile.TemporaryDirectory(prefix="aes-fuzz-") as tmp:
        path = Path(tmp) / "rules.json"
        while time.monotonic() < deadline:
            if rng.random() < 0.3:
                payload = _random_text(rng)
                path.write_text(payload, encoding="utf-8")
            else:
                rules = [_random_rule(rng, i) for i in range(rng.randint(1, 4))]
                path.write_text(json.dumps(rules), encoding="utf-8")
            try:
                registry.load(str(path))
            except Exception:
                pass
            iterations += 1
    print(f"fuzz_rule_loading iterations={iterations} seed={seed}")
    return iterations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    run(seconds=max(0.1, args.seconds), seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
