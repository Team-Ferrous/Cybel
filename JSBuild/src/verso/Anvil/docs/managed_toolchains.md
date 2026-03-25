# Managed Toolchains

## Overview

Anvil manages remediation toolchains under [`.anvil/toolchains/`](/home/mike/Documents/Github/Anvil/.anvil/toolchains/) so `saguaro verify --fix` can use consistent language-specific fixers without requiring ad hoc host setup.

The managed toolchain layer does three things:

- reuse system tools when they already satisfy a profile
- reuse cached Anvil-managed tools when they were installed previously
- bootstrap missing tools into `.anvil/toolchains/` when a fixer or first-run startup check needs them

Receipts and fix plans record the resolved toolchain profile, source, and tool paths for every remediation batch.

## Profiles

Current profiles are defined in [toolchains.py](/home/mike/Documents/Github/Anvil/saguaro/sentinel/remediation/toolchains.py).

- `llvm-native`
  Uses the repo's actual native build chain: system `cmake`, `c++`, `cc`, and `g++`, plus managed `clang-format`, `clang-tidy`, and `clang-apply-replacements` for C/C++ remediation.
- `rust-toolchain`
  Uses `cargo` and `rustup`.
- `go-toolchain`
  Manages `go`, `gofmt`, and `gopls`.
- `java-toolchain`
  Manages `javac`, `google-java-format`, and `ktlint`, while reusing a system `java` runtime when present.
- `node-web`
  Reuses system `node` and `npm`, and manages `eslint`, `prettier`, and `stylelint`.
- `config-formatters`
  Manages `taplo`.
- `shell-tooling`
  Manages `shfmt` and `shellcheck`.

## First REPL Run

When a user launches the interactive REPL through `anvil` for the first time, Anvil runs a managed toolchain startup check before constructing the REPL.

That startup check:

- writes a cached report to `.anvil/toolchains/repl_startup_check.json`
- eagerly checks and bootstraps the lighter profiles:
  - `node-web`
  - `config-formatters`
  - `shell-tooling`
  - `go-toolchain`
  - `java-toolchain`
- defers heavy native bootstrap for `llvm-native` until first use by default

The deferred native behavior is intentional because LLVM is much larger than the other profiles and is only needed when C/C++ remediation paths are activated.

## Environment Controls

- `ANVIL_SKIP_TOOLCHAIN_CHECK=1`
  Skips the interactive REPL startup check entirely.
- `ANVIL_BOOTSTRAP_ALL_TOOLCHAINS=1`
  Allows the first interactive REPL run to bootstrap deferred heavy profiles such as `llvm-native`.

## Artifacts

Important local files:

- `.anvil/toolchains/manifest.json`
  Cached record of resolved and bootstrapped profiles.
- `.anvil/toolchains/repl_startup_check.json`
  Cached first-run REPL check report.
- `.anvil/fix_receipts/`
  Per-run remediation receipts and rollback bundles.

These paths are ignored through [`.gitignore`](/home/mike/Documents/Github/Anvil/.gitignore) and should not pollute commits.

## Native Build Alignment

The repo's native build scripts currently use the system GNU/CMake toolchain for compilation:

- [build_native.sh](/home/mike/Documents/Github/Anvil/core/native/build_native.sh)
- [build_simd.sh](/home/mike/Documents/Github/Anvil/core/simd/build_simd.sh)
- [CMakeLists.txt](/home/mike/Documents/Github/Anvil/core/native/CMakeLists.txt)

That means `llvm-native` is a remediation profile, not a forced replacement for the repo's compile toolchain. In practice the profile resolves as a hybrid:

- build with system `cmake` and GNU-style compiler entrypoints already used by the repo
- remediate with `clang-format`, `clang-tidy`, and `clang-apply-replacements`
