from __future__ import annotations

import argparse
import json
import random
import string
import time


def _random_jsonish(rng: random.Random, max_len: int = 256) -> str:
    alphabet = string.ascii_letters + string.digits + "{}[],:\"' \\n\\t"
    length = rng.randint(1, max_len)
    return "".join(rng.choice(alphabet) for _ in range(length))


def run(seconds: float, seed: int) -> int:
    rng = random.Random(seed)
    deadline = time.monotonic() + seconds
    iterations = 0
    while time.monotonic() < deadline:
        payload = _random_jsonish(rng)
        try:
            json.loads(payload)
        except Exception:
            pass
        iterations += 1
    print(f"fuzz_json_parsing iterations={iterations} seed={seed}")
    return iterations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(seconds=max(0.1, args.seconds), seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
