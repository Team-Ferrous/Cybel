import sys
import os
import time
import shutil

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domains.code_intelligence.saguaro_substrate import SaguaroSubstrate


def benchmark_impact(root_dir, target_file):
    print(f"Benchmarking 'agent_impact' on {target_file} in {root_dir}")
    print(
        f"Tools available: rg={bool(shutil.which('rg'))}, grep={bool(shutil.which('grep'))}"
    )

    substrate = SaguaroSubstrate(root_dir)

    start_time = time.time()
    result = substrate.agent_impact(target_file)
    end_time = time.time()

    duration = end_time - start_time
    print(f"\nDuration: {duration:.4f} seconds")
    print("-" * 40)
    print("Result Preview:")
    print("\n".join(result.splitlines()[:5]))
    if len(result.splitlines()) > 5:
        print(f"... ({len(result.splitlines()) - 5} more lines)")

    return duration


if __name__ == "__main__":
    # Test on core/agent.py (central file, likely many imports)
    target = "core/agent.py"
    benchmark_impact(os.getcwd(), target)
