from __future__ import annotations

from saguaro.math import MathEngine


def test_math_engine_suppresses_config_literals_but_keeps_real_math(tmp_path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.py").write_text(
        "LANGUAGE_HINTS = {\n"
        "    'py': 'python',\n"
        "    'rs': 'rust',\n"
        "}\n"
        "DEFAULTS = ['fast', 'safe']\n",
        encoding="utf-8",
    )
    (src / "kernel.py").write_text(
        "def score(alpha, beta, gamma):\n"
        "    value = alpha * beta + gamma\n"
        "    return value / 2\n",
        encoding="utf-8",
    )

    parsed = MathEngine(str(tmp_path)).parse("src")
    expressions = {item["expression"] for item in parsed["records"]}

    assert "value = alpha * beta + gamma" in expressions
    assert "return value / 2" in expressions
    assert not any("LANGUAGE_HINTS" in item for item in expressions)
    assert not any("DEFAULTS" in item for item in expressions)


def test_math_engine_skips_generated_build_dirs(tmp_path) -> None:
    kernel = tmp_path / "core" / "native" / "kernel.cpp"
    kernel.parent.mkdir(parents=True, exist_ok=True)
    kernel.write_text(
        "inline float dot(float* a, float* b, int i) {\n"
        "  return a[i] * b[i];\n"
        "}\n",
        encoding="utf-8",
    )
    build_log = (
        tmp_path
        / "core"
        / "native"
        / "build-cmake-20260311"
        / "CMakeFiles"
        / "CMakeConfigureLog.yaml"
    )
    build_log.parent.mkdir(parents=True, exist_ok=True)
    build_log.write_text(
        "The system is: Linux - 6.17.0-14-generic - x86_64\n"
        "Compiler: /usr/bin/c++\n",
        encoding="utf-8",
    )

    parsed = MathEngine(str(tmp_path)).parse("core/native")

    assert parsed["files_scanned"] == 1
    assert {item["file"] for item in parsed["records"]} == {"core/native/kernel.cpp"}


def test_math_engine_ignores_cpp_default_argument_signatures(tmp_path) -> None:
    target = tmp_path / "core" / "native" / "kernel.cpp"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "inline void kernel(float* input, float* output, int n, float scale = 1.0f) {\n"
        "  for (int i = 0; i < n; ++i) {\n"
        "    output[i] = input[i] * scale;\n"
        "  }\n"
        "}\n",
        encoding="utf-8",
    )

    parsed = MathEngine(str(tmp_path)).parse("core/native/kernel.cpp")
    expressions = {item["expression"] for item in parsed["records"]}

    assert "output[i] = input[i] * scale" in expressions
    assert not any(item.startswith("inline void kernel(") for item in expressions)


def test_math_engine_ignores_cpp_lambda_bodies(tmp_path) -> None:
    target = tmp_path / "core" / "native" / "kernel.cpp"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "inline void kernel(float* input, float* output, int n) {\n"
        "  auto run = [&](int i) {\n"
        "    output[i] = input[i] * 2.0f;\n"
        "  };\n"
        "  sum += output[0] / 2.0f;\n"
        "}\n",
        encoding="utf-8",
    )

    parsed = MathEngine(str(tmp_path)).parse("core/native/kernel.cpp")
    expressions = {item["expression"] for item in parsed["records"]}

    assert "sum += output[0] / 2.0f" in expressions
    assert not any(item.startswith("auto run = [&]") for item in expressions)
