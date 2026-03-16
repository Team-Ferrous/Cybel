import pytest

from core.native.coconut_bridge import (
    CoconutNativeBridge,
    NativeCoconutBridgeUnavailableError,
)


def test_coconut_bridge_strict_mode_fails_fast_when_native_lib_missing(tmp_path):
    missing_lib = tmp_path / "missing_libcoconut_bridge.so"
    with pytest.raises(
        NativeCoconutBridgeUnavailableError,
        match="Native COCONUT bridge library not found",
    ):
        CoconutNativeBridge(lib_path=str(missing_lib), strict_native=True)


def test_coconut_bridge_non_strict_mode_still_requires_native_lib(tmp_path):
    missing_lib = tmp_path / "missing_libcoconut_bridge.so"
    with pytest.raises(
        NativeCoconutBridgeUnavailableError,
        match="Native COCONUT bridge library not found",
    ):
        CoconutNativeBridge(lib_path=str(missing_lib), strict_native=False)
