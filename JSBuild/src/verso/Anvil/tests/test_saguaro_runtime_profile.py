from __future__ import annotations

from saguaro.indexing.auto_scaler import calibrate_runtime_profile, load_runtime_profile


def test_runtime_profile_calibration_persists_selected_layout(tmp_path) -> None:
    payload = calibrate_runtime_profile(
        str(tmp_path),
        cpu_threads=16,
        native_threads=8,
        force=True,
    )

    layout = dict(payload.get("selected_runtime_layout") or {})
    assert layout["cpu_threads"] == 16
    assert layout["native_threads"] == 8
    assert layout["query_threads"] >= 1
    assert layout["max_concurrent_saguaro_sessions"] >= 1
    assert layout["max_parallel_agents"] >= 1
    assert layout["max_parallel_anvil_instances"] >= 1

    loaded = load_runtime_profile(str(tmp_path))
    assert dict(loaded.get("selected_runtime_layout") or {}) == layout
