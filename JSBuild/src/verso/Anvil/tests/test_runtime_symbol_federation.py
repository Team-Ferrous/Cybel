from core.native.runtime_telemetry import build_runtime_capability_ledger
from core.networking.instance_identity import InstanceRegistry


def test_runtime_symbol_digest_can_be_projected_into_instance_presence(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    registry = InstanceRegistry(anvil_dir=str(tmp_path / ".anvil"), project_root=str(repo))
    ledger = build_runtime_capability_ledger(
        {
            "model": "test-model",
            "digest": "weights-1",
            "native_isa_baseline": "x86_64-avx2",
            "decode_threads": 4,
            "batch_threads": 2,
            "ubatch": 16,
            "backend_module": "native_qsg",
            "backend_module_loaded": True,
        }
    )

    identity = registry.update_presence(runtime_symbol_digest=ledger["capability_digest"])

    assert ledger["capability_digest"]
    assert identity.runtime_symbol_digest == ledger["capability_digest"]
