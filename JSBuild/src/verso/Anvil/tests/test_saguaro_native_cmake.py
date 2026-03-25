from __future__ import annotations

import re
from pathlib import Path


def test_native_cmake_wires_active_saguaro_ops() -> None:
    cmake = Path("Saguaro/saguaro/native/CMakeLists.txt").read_text(encoding="utf-8")
    listed = set(re.findall(r"\b(?:ops|controllers)/[^\s)#]+\.(?:cc|cpp)\b", cmake))

    required = {
        "ops/alphaqubit_correct_op.cc",
        "ops/fused_gqa_op.cc",
        "ops/fused_linear_gqa_op.cc",
        "ops/fused_quls_loss_op.cc",
        "ops/fused_sliding_gqa_op.cc",
        "ops/fused_tpa_op.cc",
        "controllers/hardware_control_client.cc",
    }

    missing = sorted(required - listed)
    assert not missing, (
        "Missing native sources from authoritative "
        f"Saguaro/saguaro/native/CMakeLists.txt: {missing}"
    )
