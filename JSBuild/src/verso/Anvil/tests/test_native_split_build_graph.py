from __future__ import annotations

from pathlib import Path
import re


def test_native_cmake_split_target_graph_includes_compat_boundary() -> None:
    cmake_path = Path("core/native/CMakeLists.txt")
    text = cmake_path.read_text(encoding="utf-8")

    assert "add_library(anvil_kernels_obj OBJECT" in text
    assert re.search(
        r"add_library\(\s*anvil_runtime_core\s+SHARED",
        text,
        flags=re.MULTILINE,
    )
    assert "add_library(anvil_compat_obj OBJECT" in text
    assert re.search(
        r"add_library\(\s*anvil_backend_granite4_tinyh\s+MODULE",
        text,
        flags=re.MULTILINE,
    )
    assert re.search(
        r"add_library\(\s*anvil_backend_qwen35\s+MODULE",
        text,
        flags=re.MULTILINE,
    )
    assert re.search(
        r"add_library\(\s*anvil_native_ops\s+SHARED",
        text,
        flags=re.MULTILINE,
    )
    assert 'OUTPUT_NAME "anvil_native_ops"' in text
    assert 'option(ANVIL_REQUIRE_OPENMP "Require OpenMP for native CPU runtime" ON)' in text
    assert 'option(ANVIL_REQUIRE_AVX2 "Require AVX2-capable x86 CPU features" ON)' in text
    assert "find_package(OpenMP)" in text
    assert "add_compile_options(${OpenMP_CXX_FLAGS})" in text
    assert "ANVIL_REQUIRE_OPENMP is ON but OpenMP was not found." in text
    assert "target_link_libraries(anvil_runtime_core PRIVATE OpenMP::OpenMP_CXX)" in text
    assert "target_link_libraries(anvil_native_ops PRIVATE OpenMP::OpenMP_CXX)" in text
    assert "add_compile_options(-mavx2 -mfma)" in text
    assert "ANVIL_REQUIRE_AVX2 is ON but AVX2/AVX-512 was not detected." in text
    assert "ANVIL_NATIVE_SPLIT_ABI_VERSION=1" in text
    assert (
        '"${ANVIL_SPLIT_ROOT}/compat/native_ops_compat.cpp"' in text
    ), "Split compat source must be part of the CMake graph."
