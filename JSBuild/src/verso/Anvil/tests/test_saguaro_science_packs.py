from saguaro.packs.base import PackManager


def test_pack_manager_enable_and_diagnose(tmp_path):
    (tmp_path / "quantum_solver.py").write_text(
        "hamiltonian = 1\nstatevector = []\n",
        encoding="utf-8",
    )

    manager = PackManager(str(tmp_path))
    enabled = manager.enable("quantum_pack")
    diagnosis = manager.diagnose(".")

    assert enabled["status"] == "ok"
    assert "quantum_pack" in enabled["enabled"]
    quantum = next(item for item in diagnosis["packs"] if item["pack"] == "quantum_pack")
    assert quantum["count"] >= 1


def test_pack_manager_lists_roadmap_initial_pack_set(tmp_path):
    packs = PackManager(str(tmp_path)).list()
    names = {item["name"] for item in packs}

    assert names == {
        "torch_pack",
        "jax_pack",
        "tensorflow_pack",
        "simd_native_pack",
        "physics_pack",
        "chemistry_pack",
        "quantum_pack",
        "cfd_pack",
    }
