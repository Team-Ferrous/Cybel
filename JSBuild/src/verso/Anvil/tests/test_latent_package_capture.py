import threading
import time

from core.campaign.state_store import CampaignStateStore
from core.memory.fabric import MemoryFabricStore, MemoryProjector
from core.qsg.config import QSGConfig
from core.qsg.continuous_engine import QSGInferenceEngine, QSGRequest
from core.qsg.latent_bridge import QSGLatentBridge


def _collect_until_chunk(
    engine: QSGInferenceEngine, request_id: str, timeout_s: float = 1.0
):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        chunk = engine.poll(request_id)
        if chunk is None:
            time.sleep(0.001)
            continue
        if chunk.text:
            return chunk
    return None


def test_latent_package_capture_and_restore(tmp_path):
    store = CampaignStateStore(str(tmp_path / "state.db"))
    fabric = MemoryFabricStore(store, storage_root=str(tmp_path / "fabric"))
    projector = MemoryProjector()
    bridge = QSGLatentBridge(fabric, projector)
    memory = fabric.create_memory(
        memory_kind="latent_branch",
        payload_json={"summary": "branch"},
        campaign_id="campaign-1",
        workspace_id="campaign-1",
        summary_text="branch",
    )

    config = QSGConfig(
        continuous_batching_enabled=True,
        batch_wait_timeout_ms=1,
        max_active_requests=2,
        max_pending_requests=4,
        capability_digest="cap-latent",
        delta_watermark={
            "delta_id": "delta-latent",
            "workspace_id": "campaign-1",
            "changed_paths": ["core/native/native_qsg_engine.py"],
        },
    )
    engine = QSGInferenceEngine(
        config=config,
        stream_producer=lambda request: iter(["alpha", "beta"]),
    )
    runner = threading.Thread(target=engine.run_forever, daemon=True)
    runner.start()
    request_id = engine.submit(QSGRequest(prompt="branch"))
    assert _collect_until_chunk(engine, request_id) is not None

    capture = bridge.capture_engine_state(
        engine=engine,
        memory_id=memory.memory_id,
        request_id=request_id,
        summary_text="captured branch state",
        capture_stage="evidence_synthesis",
    )
    assert capture["mode"] == "captured"
    latest_package = fabric.latest_latent_package(memory.memory_id)
    assert latest_package is not None
    compatibility = dict(latest_package.get("compatibility_json") or {})
    assert compatibility["latent_packet_abi_version"] == 3
    assert compatibility["execution_capsule_version"] == 3
    assert compatibility["execution_capsule_id"]
    assert compatibility["capability_digest"] == "cap-latent"
    assert compatibility["delta_watermark"]["delta_id"] == "delta-latent"
    assert compatibility["repo_delta_memory"]["path_count"] == 1
    assert compatibility["tensor_codec"] == "float16"
    assert compatibility["segment_count"] == 2
    assert latest_package["tensor_format"] == "native-f16"

    replay_engine = QSGInferenceEngine(
        config=config,
        stream_producer=lambda request: iter(["resume"]),
    )
    replay_id = bridge.replay(
        engine=replay_engine,
        memory_id=memory.memory_id,
        model_family="qsg-python",
        hidden_dim=1,
    )

    engine.shutdown()
    runner.join(timeout=1.0)
    assert replay_id["restored"] is True
    assert replay_id["memory_tier_decision"]["selected_tier"] == "latent_replay"
    assert replay_id["mission_replay_descriptor"]["capability_digest"] == "cap-latent"
    assert (
        replay_id["mission_replay_descriptor"]["delta_watermark"]["delta_id"]
        == "delta-latent"
    )
    assert replay_engine.metrics_snapshot()["qsg_latent_requests"] == 1
