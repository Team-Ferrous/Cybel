from __future__ import annotations

from pathlib import Path

import saguaro.api as api_module
from saguaro.agents.perception import TracePerception
from saguaro.analysis.ffi_scanner import FFIScanner
from saguaro.api import SaguaroAPI
from saguaro.indexing.backends import NumPyBackend


def test_trace_perception_scans_multiple_ffi_mechanisms(tmp_path: Path) -> None:
    (tmp_path / "bridge.py").write_text(
        "import ctypes\n"
        "lib = ctypes.CDLL('libnative.so')\n"
        "\n"
        "from cffi import FFI\n"
        "ffi = FFI()\n"
        "ffi.dlopen('libffi_target.so')\n",
        encoding="utf-8",
    )
    (tmp_path / "addon.ts").write_text(
        "const ffi = require('ffi-napi');\n",
        encoding="utf-8",
    )

    perception = TracePerception(repo_path=str(tmp_path))
    report = perception.ffi_boundaries(limit=20)

    assert report["status"] == "ok"
    assert report["count"] >= 3
    mechanisms = {row["mechanism"] for row in report["boundaries"]}
    assert {"ctypes", "cffi", "napi"} <= mechanisms


def test_api_ffi_surface_uses_trace_perception(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        api_module,
        "get_backend",
        lambda prefer_tensorflow=True: NumPyBackend(),
    )
    (tmp_path / "bridge.py").write_text(
        "import ctypes\nlib = ctypes.CDLL('libnative.so')\n",
        encoding="utf-8",
    )

    api = SaguaroAPI(repo_path=str(tmp_path))
    report = api.ffi(limit=10)

    assert report["status"] == "ok"
    assert report["count"] >= 1
    assert report["boundaries"][0]["mechanism"] == "ctypes"


def test_scanner_extracts_ctypes_argtypes_and_restype(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "native.cc").write_text("// native impl\n", encoding="utf-8")
    source = (
        "import ctypes\n"
        "lib = ctypes.CDLL('libnative.so')\n"
        "lib.add.argtypes = [ctypes.c_int, ctypes.c_char_p]\n"
        "lib.add.restype = ctypes.c_int\n"
    )
    scanner = FFIScanner(repo_path=str(tmp_path))
    findings = scanner.scan_file("bridge.py", source)

    ctypes_finding = next(
        row for row in findings if row.get("kind") == "ctypes_load_library"
    )
    assert ctypes_finding["type_map"]["host_params"][0]["guest_type"] == "ctypes.c_int"
    assert ctypes_finding["type_map"]["host_return"]["guest_type"] == "ctypes.c_int"
    resolution = ctypes_finding.get("shared_object_resolution") or {}
    assert int(resolution.get("candidate_count", 0)) >= 1
    assert int(resolution.get("existing_count", 0)) >= 1


def test_scanner_extracts_cffi_cdef_signatures(tmp_path: Path) -> None:
    source = (
        "from cffi import FFI\n"
        "ffi = FFI()\n"
        "ffi.cdef('int add(int a, int b);')\n"
        "ffi.dlopen('libffi_target.so')\n"
    )
    scanner = FFIScanner(repo_path=str(tmp_path))
    findings = scanner.scan_file("bridge.py", source)

    cffi_finding = next(row for row in findings if row.get("kind") == "cffi_dlopen")
    extraction = cffi_finding.get("typing_extraction") or {}
    cffi_payload = extraction.get("cffi") or {}
    assert "int add(int a, int b);" in cffi_payload.get("cdef_signatures", [])
    assert cffi_finding["type_map"]["host_return"]["guest_type"] == "int"


def test_trace_perception_ffi_surface_exposes_rich_metadata(tmp_path: Path) -> None:
    (tmp_path / "bridge.py").write_text(
        "import ctypes\n"
        "lib = ctypes.CDLL('libnative.so')\n"
        "lib.add.argtypes = [ctypes.c_int]\n"
        "lib.add.restype = ctypes.c_int\n",
        encoding="utf-8",
    )
    report = TracePerception(repo_path=str(tmp_path)).ffi_boundaries(limit=20)
    row = next(item for item in report["boundaries"] if item["mechanism"] == "ctypes")
    assert row["from_language"] == "python"
    assert row["to_language"] == "native"
    assert row["type_map"]["host_params"][0]["guest_type"] == "ctypes.c_int"


def test_scanner_detects_tensorflow_custom_op_loader(tmp_path: Path) -> None:
    source = (
        "import tensorflow as tf\n"
        "module = tf.load_op_library('libcustom_kernel.so')\n"
    )
    scanner = FFIScanner(repo_path=str(tmp_path))
    findings = scanner.scan_file("bridge.py", source)

    tf_finding = next(
        row for row in findings if row.get("kind") == "tensorflow_load_op_library"
    )
    assert tf_finding["boundary_type"] == "tensorflow.custom_op_load"
    assert tf_finding["shared_object"] == "libcustom_kernel.so"
