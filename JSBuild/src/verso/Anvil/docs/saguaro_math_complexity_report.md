# Saguaro Math Complexity Report

Generated on March 11, 2026 from `MathEngine.parse('.')`.

## Scoring model

Saguaro currently assigns a structural complexity score per extracted math record using:

- operator count
- symbol count
- function call count
- nesting depth
- token count

Complexity bands:

- `low`: score `< 8`
- `medium`: score `8-15`
- `high`: score `>= 16`

Important caveat:

- This is a structural score, not a semantic proof of mathematical difficulty.
- Dense parser tables, regex-heavy config blobs, and large declarative literals can rank very high even when they are not numerical kernels.
- For actual kernel math, the native sections below are the more useful view.

## Repo-wide summary

| Metric | Value |
| --- | ---: |
| Files scanned | 2,473 |
| Math records | 60,145 |
| Code expressions | 51,569 |
| Code comments | 8,572 |
| Markdown equations | 4 |
| Python records | 35,437 |
| C++ records | 13,435 |
| C/C++ header records | 11,269 |
| Total structural score | 1,181,115 |
| Average structural score | 19.64 |
| Max structural score | 9,312 |
| High-complexity records | 20,753 |

## Repo-wide files ranked heaviest to lightest

| Rank | File | Records | Total score | Max score | High-count | Avg score |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `core/native/native_qsg_engine.py` | 1,223 | 24,952 | 1,701 | 654 | 20.40 |
| 2 | `core/unified_chat_loop.py` | 757 | 17,636 | 4,084 | 323 | 23.30 |
| 3 | `audit/runner/benchmark_suite.py` | 664 | 16,487 | 307 | 422 | 24.83 |
| 4 | `saguaro/cli.py` | 879 | 15,085 | 83 | 495 | 17.16 |
| 5 | `saguaro/api.py` | 780 | 14,313 | 145 | 402 | 18.35 |
| 6 | `benchmarks/native_qsg_benchmark.py` | 147 | 13,466 | 8,648 | 65 | 91.61 |
| 7 | `saguaro/parsing/parser.py` | 20 | 9,638 | 9,312 | 11 | 481.90 |
| 8 | `saguaro/services/platform.py` | 117 | 8,259 | 6,185 | 48 | 70.59 |
| 9 | `saguaro/agents/perception.py` | 100 | 8,039 | 6,505 | 35 | 80.39 |
| 10 | `core/native/parallel_generation.py` | 357 | 7,809 | 519 | 191 | 21.87 |
| 11 | `core/native/qsg_forward.py` | 529 | 7,619 | 40 | 154 | 14.40 |
| 12 | `core/native/model_graph.cpp` | 468 | 7,504 | 657 | 104 | 16.03 |
| 13 | `core/campaign/control_plane.py` | 241 | 6,246 | 150 | 137 | 25.92 |
| 14 | `tests/test_saguaro_interface.py` | 317 | 5,853 | 91 | 149 | 18.46 |
| 15 | `audit/runner/native_benchmark_runner.py` | 251 | 5,750 | 560 | 137 | 22.91 |
| 16 | `core/native/model_graph_wrapper.py` | 352 | 5,698 | 250 | 136 | 16.19 |
| 17 | `domains/code_intelligence/saguaro_substrate.py` | 319 | 5,344 | 126 | 131 | 16.75 |
| 18 | `core/agent.py` | 271 | 4,687 | 96 | 96 | 17.30 |
| 19 | `audit/runner/attempt_executor.py` | 139 | 4,452 | 1,353 | 74 | 32.03 |
| 20 | `Saguaro/saguaro/cli.py` | 301 | 4,438 | 50 | 133 | 14.74 |

## Repo-wide top math records

| Rank | File | Line | Kind | Score | Band | Expression |
| ---: | --- | ---: | --- | ---: | --- | --- |
| 1 | `saguaro/parsing/parser.py` | 170 | `code_expression` | 9,312 | `high` | `_TREE_SITTER_BACKENDS = { ... }` |
| 2 | `benchmarks/native_qsg_benchmark.py` | 1710 | `code_expression` | 8,648 | `high` | `pass_count = sum(...) prefill_gaps = [...] decode_gaps = [...]` |
| 3 | `saguaro/agents/perception.py` | 870 | `code_expression` | 6,505 | `high` | `patterns = [(re.compile(...), "class"), ...]` |
| 4 | `saguaro/services/platform.py` | 652 | `code_expression` | 6,185 | `high` | `include_match = re.search(...) ... candidate = ...` |
| 5 | `core/unified_chat_loop.py` | 4876 | `code_expression` | 4,084 | `high` | `chunks = re.split(...) ... deduplicated ...` |
| 6 | `tools/schema.py` | 3 | `code_expression` | 3,190 | `high` | `TOOL_SCHEMAS = { "tools": [...] }` |
| 7 | `saguaro/architecture/topology.py` | 50 | `code_expression` | 2,939 | `high` | `_IMPORT_RE = re.compile(...) ... _normalize_rel_path(...)` |
| 8 | `saguaro/analysis/ffi_scanner.py` | 276 | `code_expression` | 2,313 | `high` | `_PATTERNS = ({ "name": "ctypes_load_library", "regex": re.compile(...) }, ...)` |
| 9 | `saguaro/sentinel/remediation/adapters.py` | 97 | `code_expression` | 2,301 | `high` | `_BARE_EXCEPT_RE = re.compile(...) ... _DOCSTRING_RULES = {...}` |
| 10 | `saguaro/math/engine.py` | 32 | `code_expression` | 2,084 | `high` | `_LINE_COMMENT_RE = re.compile(...) ... _ASSIGNMENT_RE = re.compile(...)` |
| 11 | `saguaro/requirements/extractor.py` | 26 | `code_expression` | 1,815 | `high` | `_SENTENCE_SPLIT_RE = re.compile(...) ... _NORMATIVE_SECTION_HINTS = (...)` |
| 12 | `tests/test_saguaro_aes_engine.py` | 54 | `code_expression` | 1,792 | `high` | `RUNTIME_RULE = { ... }` |

## Native-only summary

This section filters to files under:

- `core/native/`
- `core/simd/`
- `saguaro/native/`

## Native files ranked heaviest to lightest

| Rank | File | Records | Total score | Max score | High-count | Avg score |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `core/native/native_qsg_engine.py` | 1,223 | 24,952 | 1,701 | 654 | 20.40 |
| 2 | `core/native/parallel_generation.py` | 357 | 7,809 | 519 | 191 | 21.87 |
| 3 | `core/native/qsg_forward.py` | 529 | 7,619 | 40 | 154 | 14.40 |
| 4 | `core/native/model_graph.cpp` | 468 | 7,504 | 657 | 104 | 16.03 |
| 5 | `core/native/model_graph_wrapper.py` | 352 | 5,698 | 250 | 136 | 16.19 |
| 6 | `core/native/qsg_parallel_kernels_wrapper.py` | 128 | 3,216 | 164 | 59 | 25.12 |
| 7 | `core/native/runtime_telemetry.py` | 169 | 3,115 | 79 | 126 | 18.43 |
| 8 | `core/simd/hnn_simd_common.h` | 238 | 3,086 | 124 | 46 | 12.97 |
| 9 | `saguaro/native/ops/hnn_simd_common.h` | 240 | 3,013 | 31 | 46 | 12.55 |
| 10 | `core/native/simd_ops_wrapper.py` | 237 | 2,998 | 103 | 44 | 12.65 |
| 11 | `saguaro/native/controllers/hamiltonian_meta_controller.cc` | 111 | 2,837 | 579 | 44 | 25.56 |
| 12 | `core/native/quantized_matmul.cpp` | 185 | 2,527 | 229 | 45 | 13.66 |
| 13 | `core/simd/fused_quls_loss_op.h` | 172 | 2,526 | 137 | 47 | 14.69 |
| 14 | `saguaro/native/ops/fused_qwt_tokenizer_op.cc` | 106 | 2,398 | 661 | 25 | 22.62 |
| 15 | `saguaro/native/ops/unified_attention.py` | 163 | 2,319 | 78 | 48 | 14.23 |
| 16 | `core/native/qsg_parallel_kernels.cpp` | 161 | 2,305 | 100 | 51 | 14.32 |
| 17 | `saguaro/native/ops/neural_kalman_op.cc` | 92 | 2,191 | 306 | 6 | 23.82 |
| 18 | `saguaro/native/ops/train_step_op.cc` | 112 | 2,153 | 290 | 45 | 19.22 |
| 19 | `core/simd/fused_token_shift_op.h` | 23 | 2,135 | 1,760 | 13 | 92.83 |
| 20 | `saguaro/native/ops/fused_token_shift_op.h` | 23 | 2,135 | 1,760 | 13 | 92.83 |

## Native top math records

| Rank | File | Line | Kind | Score | Band | Expression |
| ---: | --- | ---: | --- | ---: | --- | --- |
| 1 | `core/simd/fused_token_shift_op.h` | 271 | `code_expression` | 1,760 | `high` | `inline void token_shift_fft_inplace(float* data, int64_t n, bool inverse = false) { ... }` |
| 2 | `saguaro/native/ops/fused_token_shift_op.h` | 271 | `code_expression` | 1,760 | `high` | `inline void token_shift_fft_inplace(float* data, int64_t n, bool inverse = false) { ... }` |
| 3 | `core/native/native_qsg_engine.py` | 3041 | `code_expression` | 1,701 | `high` | `capabilities = { "backend": self.backend, ... }` |
| 4 | `core/simd/hd_spatial_block_op.h` | 64 | `code_expression` | 1,322 | `high` | `inline void fft_butterfly(float* data, int n, bool inverse = false) { ... }` |
| 5 | `saguaro/native/ops/hd_spatial_block_op.h` | 64 | `code_expression` | 1,322 | `high` | `inline void fft_butterfly(float* data, int n, bool inverse = false) { ... }` |
| 6 | `saguaro/native/ops/fused_reasoning_stack/tt_helpers.cc` | 124 | `code_expression` | 1,194 | `high` | `size_t array_start = descriptor_json.find('[', tt_layers_pos); ...` |
| 7 | `core/simd/fft_utils.h` | 109 | `code_expression` | 1,163 | `high` | `inline void fft_butterfly(float* data, int n, bool inverse = false) { ... }` |
| 8 | `saguaro/native/ops/fft_utils.h` | 109 | `code_expression` | 1,163 | `high` | `inline void fft_butterfly(float* data, int n, bool inverse = false) { ... }` |
| 9 | `core/native/numa_topology.cpp` | 87 | `code_expression` | 1,156 | `high` | `inline int read_int_file(const std::string& path, int fallback = -1) { ... }` |
| 10 | `core/simd/hd_bundle_op.h` | 97 | `code_expression` | 1,135 | `high` | `inline void fft_inplace(float* re, float* im, int n, bool inverse = false) { ... }` |
| 11 | `saguaro/native/ops/hd_bundle_op.h` | 97 | `code_expression` | 1,135 | `high` | `inline void fft_inplace(float* re, float* im, int n, bool inverse = false) { ... }` |
| 12 | `saguaro/native/ops/common/parallel/parallel_backend.h` | 67 | `code_expression` | 1,119 | `high` | `inline Backend BackendFromString(std::string_view raw_value, bool* matched = nullptr) { ... }` |

## Native math observations

- The heaviest native records are concentrated in FFT-style transforms, token-shift kernels, holographic binding utilities, and parallel/native runtime orchestration.
- `core/native/native_qsg_engine.py` dominates total native score because it contains many medium-to-high complexity expressions rather than a single overwhelming kernel.
- The most concentrated kernel-heavy headers are the FFT and holographic families:
  - `core/simd/fused_token_shift_op.h`
  - `core/simd/hd_spatial_block_op.h`
  - `core/simd/fft_utils.h`
  - `core/simd/hd_bundle_op.h`

## Recommended next refinements

- Split the report into:
  - numerical kernels
  - parser/regex/config structures
  - orchestration logic
- Add a symbol-aware aggregation so scores roll up by function or class, not just by file and raw statement.
- Add a `saguaro math report --path ... --format md` command so this report can be regenerated directly from the CLI.
