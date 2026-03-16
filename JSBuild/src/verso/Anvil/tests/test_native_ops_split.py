from __future__ import annotations

from pathlib import Path

from core.native import native_ops


class _FakeSymbol:
    def __init__(self, value):
        self._value = value
        self.argtypes = None
        self.restype = None

    def __call__(self):
        return self._value


class _FakeLib:
    pass


def test_get_native_library_info_reports_split_abi_metadata(monkeypatch) -> None:
    fake_lib = _FakeLib()
    fake_lib.anvil_native_build_id = _FakeSymbol(b"build-123")
    fake_lib.anvil_native_split_layout = _FakeSymbol(
        b"kernels/runtime_core/backends/compat"
    )
    fake_lib.anvil_native_public_load_target = _FakeSymbol(b"libanvil_native_ops.so")
    fake_lib.anvil_native_runtime_core_target = _FakeSymbol(
        b"libanvil_runtime_core.so"
    )
    fake_lib.anvil_native_split_abi_version = _FakeSymbol(1)
    fake_lib.anvil_native_compat_alias_csv = _FakeSymbol(
        b"libanvil_native_ops.so,libfast_attention.so,libcoconut_bridge.so"
    )

    monkeypatch.setattr(native_ops, "load_native_library", lambda extra=None: fake_lib)
    monkeypatch.setattr(
        native_ops,
        "loaded_library_path",
        lambda extra=None: Path("/tmp/libanvil_native_ops.so"),
    )
    monkeypatch.setattr(native_ops, "_sha256_file", lambda _: "deadbeef")

    info = native_ops.get_native_library_info()

    assert info["native_build_id"] == "build-123"
    assert info["loaded_native_library"] == "/tmp/libanvil_native_ops.so"
    assert info["native_build_sha256"] == "deadbeef"
    assert info["native_split_layout"] == "kernels/runtime_core/backends/compat"
    assert info["native_public_load_target"] == "libanvil_native_ops.so"
    assert info["native_runtime_core_target"] == "libanvil_runtime_core.so"
    assert info["native_split_abi_version"] == 1
    assert info["native_compat_aliases"] == [
        "libanvil_native_ops.so",
        "libfast_attention.so",
        "libcoconut_bridge.so",
    ]


def test_resolve_backend_module_selection_reports_resolution_source() -> None:
    explicit = native_ops.resolve_backend_module_selection(
        backend_name="granite4-tinyh",
        model_name="qwen3.5:9b",
        architecture="qwen35",
        family="qwen",
    )
    inferred = native_ops.resolve_backend_module_selection(
        model_name="qwen3.5:9b",
        architecture="qwen35",
        family="qwen",
    )
    unresolved = native_ops.resolve_backend_module_selection(
        model_name="unknown-model",
        architecture="unknown",
        family="unknown",
    )

    assert explicit["backend_module"] == "granite4_tinyh"
    assert explicit["backend_selection_source"] == "explicit"
    assert explicit["backend_selection_reason"] == "backend_name"

    assert inferred["backend_module"] == "qwen35"
    assert inferred["backend_selection_source"] == "model_name_exact"
    assert inferred["backend_selection_reason"] == "qwen3.5:9b"

    assert unresolved["backend_module"] == ""
    assert unresolved["backend_selection_source"] == "unresolved"
    assert unresolved["backend_selection_reason"] == "no_match"


def test_get_native_backend_info_reports_module_abi_and_candidates(monkeypatch) -> None:
    fake_lib = _FakeLib()
    fake_lib.anvil_backend_qwen35_marker = _FakeSymbol(1)
    fake_lib.anvil_backend_qwen35_name = _FakeSymbol(b"qwen35")
    fake_lib.anvil_backend_qwen35_build_id = _FakeSymbol(b"build-123")
    fake_lib.anvil_backend_qwen35_abi_version = _FakeSymbol(1)

    monkeypatch.setattr(
        native_ops,
        "load_backend_module",
        lambda backend_name: (  # noqa: ARG005
            fake_lib,
            Path("/tmp/libanvil_backend_qwen35.so"),
        ),
    )

    info = native_ops.get_native_backend_info(
        backend_name="",
        model_name="qwen3.5:9b",
        architecture="qwen35",
        family="qwen",
    )

    assert info["backend_module"] == "qwen35"
    assert info["backend_module_loaded"] is True
    assert info["backend_module_library"] == "/tmp/libanvil_backend_qwen35.so"
    assert info["backend_module_marker_symbol"] == "anvil_backend_qwen35_marker"
    assert info["backend_module_marker"] == 1
    assert info["backend_module_name"] == "qwen35"
    assert info["backend_module_build_id"] == "build-123"
    assert info["backend_module_abi_version"] == 1
    assert info["backend_selection_source"] == "model_name_exact"
    assert info["backend_module_candidates"] == [
        str(Path(native_ops.__file__).resolve().parent / "libanvil_backend_qwen35.so"),
        str(
            Path(native_ops.__file__).resolve().parent
            / "build"
            / "libanvil_backend_qwen35.so"
        ),
    ]


def test_get_native_backend_info_preserves_candidates_on_load_error(
    monkeypatch,
) -> None:
    def _raise_not_found(backend_name: str):  # noqa: ARG001
        raise FileNotFoundError("missing module")

    monkeypatch.setattr(native_ops, "load_backend_module", _raise_not_found)

    info = native_ops.get_native_backend_info(
        backend_name="granite4_tinyh",
        model_name="granite4:tiny-h",
        architecture="granite",
        family="granite",
    )

    assert info["backend_module"] == "granite4_tinyh"
    assert info["backend_module_loaded"] is False
    assert info["backend_selection_source"] == "explicit"
    assert info["backend_module_candidates"] == [
        str(
            Path(native_ops.__file__).resolve().parent
            / "libanvil_backend_granite4_tinyh.so"
        ),
        str(
            Path(native_ops.__file__).resolve().parent
            / "build"
            / "libanvil_backend_granite4_tinyh.so"
        ),
    ]
    assert info["backend_module_error"] == "missing module"
