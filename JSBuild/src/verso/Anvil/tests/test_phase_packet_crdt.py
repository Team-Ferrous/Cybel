from core.campaign.phase_packet import PhasePacketBuilder


def test_phase_packet_builder_compacts_phase_scope_and_evidence():
    packets = PhasePacketBuilder.build(
        [
            {
                "phase_id": "development",
                "name": "Development",
                "tasks": [
                    {
                        "item_id": "item-1",
                        "repo_scope": ["core/networking", "core/ownership"],
                        "owner_type": "ImplementationEngineerSubagent",
                        "telemetry_contract": {"minimum": ["wall_time", "verification"]},
                        "allowed_writes": ["core/networking"],
                        "required_evidence": ["tests/test_repo_presence.py"],
                        "required_artifacts": ["comparative_reports"],
                        "rollback_criteria": ["peer flapping"],
                        "success_metrics": ["verification_passed"],
                    }
                ],
                "dependencies": ["research"],
                "success_criteria": ["presence stable"],
                "artifact_folder": "artifacts/development",
            }
        ],
        objective="Build connectivity control plane",
    )

    assert packets[0]["phase_id"] == "development"
    assert packets[0]["repo_scope"] == ["core/networking", "core/ownership"]
    assert packets[0]["promotion_gate"]["tasks"] == ["item-1"]
    assert packets[0]["required_evidence"] == ["tests/test_repo_presence.py"]
    assert packets[0]["required_artifacts"] == ["comparative_reports"]
