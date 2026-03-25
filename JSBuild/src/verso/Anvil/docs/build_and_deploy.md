# Build & Deployment Documentation

## 1. Executive Summary

This guide outlines the standard procedures for building, running, and deploying the **Anvil** AI system and its underlying mathematical framework, **HighNoon**, along with the quantum codebase OS, **Saguaro**. The Anvil system is primarily deployed in standalone enterprise environments as an active AI agent.

## 2. Prerequisites

The deployment targets Linux operating systems. The core framework relies heavily on SIMD compute, therefore the host CPU must support Advanced Vector Extensions (AVX2 minimum, AVX-512 or ARM NEON preferred).

-   **Python version:** 3.12+
-   **Compiler:** CMake and a C++17 compatible compiler (e.g., GCC or Clang)
-   **Linter/Formatter:** `ruff`, `black`
-   **Testing:** `pytest` (Python), `bazel`/CMake (C++)

Typical Debian/Ubuntu packages for native benchmark and rebuild work:

```bash
sudo apt-get install -y build-essential cmake pkg-config libnuma-dev libhwloc-dev numactl linux-tools-common
```

On Ubuntu Noble, `cpupower` is provided by `linux-tools-common`, not a separate `linux-cpupower` package. If you also need kernel-matched perf tooling, add:

```bash
sudo apt-get install -y linux-tools-$(uname -r)
```

If you want to compile the optional AMX leaf, use a compiler that accepts `-mamx-tile` and `-mamx-bf16` (GCC 12+ or recent Clang). AMX execution still requires an Intel host with AMX-capable hardware and kernel/xstate support. On the current AMD AVX2 host, AMX may compile into the binary but remains runtime-inactive by design.

## 3. Building Anvil

The project is structured with Python front-end wrappers interacting with heavily optimized C++ backend libraries. Building the project automatically invokes these native compilations.

### 3.1 Standard Compilation
To build and install the native extensions and the Python environment, execute the following from the root of the repository:

```bash
pip install -e .
```

This command acts as a trigger to run the native build script (`core/native/build_native.sh`). It invokes CMake to handle compilation of the HighNoon framework native models, the `native saguaro core` library, and the SIMD mathematical primitives.

### 3.2 Native ISA and AMX Rebuild Controls

The native build entrypoint now accepts explicit environment controls instead of forcing host-tuned flags:

```bash
ANVIL_NATIVE_ISA_BASELINE=avx2 \
ANVIL_ENABLE_AMX=ON \
ANVIL_REQUIRE_AVX2=ON \
ANVIL_ENABLE_HWLOC=ON \
ANVIL_ENABLE_LIBNUMA=ON \
pip install -e .
```

Useful knobs:

- `ANVIL_NATIVE_ISA_BASELINE=avx2` keeps the portable AVX2/FMA baseline used by the benchmark suite.
- `ANVIL_NATIVE_ISA_BASELINE=host` enables `-march=native` host tuning when you want a host-specific binary.
- `ANVIL_ENABLE_AMX=ON|OFF` controls whether the optional AMX wrapper leaf is compiled into the native runtime.
- `ANVIL_NATIVE_EXTRA_CXX_FLAGS="..."` appends extra CMake C++ flags without replacing the checked-in ISA policy.
- `ANVIL_SIMD_FORCE_FLAGS="..."` and `ANVIL_SIMD_EXTRA_FLAGS="..."` control the separate `core/simd/build_simd.sh` library build.

To rebuild the native runtime directly without reinstalling Python packaging:

```bash
source venv/bin/activate
ANVIL_NATIVE_ISA_BASELINE=avx2 ANVIL_ENABLE_AMX=ON ./core/native/build_native.sh
```

For an Intel AMX benchmark host, rebuild with a host-tuned baseline and AMX enabled:

```bash
source venv/bin/activate
ANVIL_NATIVE_ISA_BASELINE=host ANVIL_ENABLE_AMX=ON ./core/native/build_native.sh
```

On AMD hosts, use the AVX2 baseline for benchmarking. Building with `ANVIL_ENABLE_AMX=ON` is allowed, but `native_runtime_amx_available` should remain `false`.

### 3.3 Saguaro Initialization
Before running the agent, the project's codebase must be semantically embedded using Saguaro.

```bash
# 1. Initialize the Saguaro index locally
saguaro init

# 2. Vectorize the codebase
saguaro index --path .
```
This process converts all `.py`, `.cc`, and `.h` routines into deterministic semantic vectors stored inside the `.saguaro/` directory.

## 4. Running Granite (Agent REPL)

The interactive REPL provides a persistent `anvil>` prompt integrating rich trace outputs and immediate subagent delegations. To launch the REPL, use the exposed entrypoint command:

```bash
anvil
```

From here, slash commands (e.g., `/help`, `/plan`) or direct human-language queries can be submitted.

On the first interactive launch, Anvil also checks the managed remediation toolchains used by `saguaro verify --fix`. Lighter profiles are bootstrapped eagerly into `.anvil/toolchains/`, while the heavier `llvm-native` profile is deferred until first use by default. See [managed_toolchains.md](/home/mike/Documents/Github/Anvil/docs/managed_toolchains.md) for the profile model, cache layout, and environment controls.

## 5. Running Autonomous Missions

Systemized missions lacking human interactivity can be launched directly. This triggers the `AgentOrchestrator` to breakdown the objective and delegate tasks programmatically.

```bash
python main.py "Your explicit objective to fulfill"
```

## 6. Saguaro Troubleshooting

If `saguaro index` processes fault or yield `Inhomogeneous shape` output signals:
1.  Verify the `native saguaro core` environment flags (particularly TensorFlow ABI constraints).
2.  Purge the corrupted index: `rm -rf .saguaro`
3.  Execute a clean rebuild: `saguaro index --path .`
