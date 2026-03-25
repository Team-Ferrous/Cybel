import numpy as np

from core.qsg.runtime_contracts import (
    CapsuleSegmentIndex,
    ExecutionCapsule,
    LatentPacketABI,
    TypedLatentSegment,
    evaluate_qsg_runtime_invariants,
)


def test_typed_latent_capsule_v3_passes_runtime_invariants() -> None:
    segments = [
        TypedLatentSegment(
            segment_id="seg-branch",
            segment_kind="branch_state",
            row_start=0,
            row_count=1,
            hidden_dim=4,
            codec="float16",
        ),
        TypedLatentSegment(
            segment_id="seg-delta",
            segment_kind="repo_delta",
            row_start=1,
            row_count=1,
            hidden_dim=4,
            codec="float16",
        ),
    ]
    index = CapsuleSegmentIndex(segments=segments)
    packet = LatentPacketABI(
        abi_version=3,
        tensor=np.eye(2, 4, dtype=np.float32).tolist(),
        tensor_codec="float16",
        hidden_dim=4,
        capability_digest="cap-1",
        delta_watermark={"delta_id": "delta-1"},
        execution_capsule_id="capsule-1",
        segments=index.as_dict()["segments"],
        segment_count=2,
    )
    capsule = ExecutionCapsule(
        capsule_id="capsule-1",
        request_id="req-1",
        version=3,
        capability_digest="cap-1",
        delta_watermark={"delta_id": "delta-1"},
        hidden_dim=4,
        latent_packet_abi_version=3,
        segment_count=2,
        segment_kinds=["branch_state", "repo_delta"],
        segment_index=index.as_dict()["segments"],
    )

    violations = evaluate_qsg_runtime_invariants(
        runtime_status={
            "qsg_native_runtime_authority": True,
            "qsg_capability_digest": "cap-1",
        },
        latent_packet=packet.as_dict(),
        execution_capsule=capsule.as_dict(),
    )

    assert violations == []


def test_latent_packet_v3_requires_typed_segments() -> None:
    violations = evaluate_qsg_runtime_invariants(
        latent_packet={
            "abi_version": 3,
            "execution_capsule_id": "capsule-1",
            "capability_digest": "cap-1",
            "delta_watermark": {"delta_id": "delta-1"},
            "segments": [],
        },
        execution_capsule={
            "capsule_id": "capsule-1",
            "version": 3,
            "capability_digest": "cap-1",
            "delta_watermark": {"delta_id": "delta-1"},
        },
    )

    assert {item["code"] for item in violations} >= {"latent_packet_segments"}
