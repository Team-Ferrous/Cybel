from core.campaign.roadmap_compiler import RoadmapCompiler, RoadmapItem


def test_render_phase_pack_emits_structured_phase_documents():
    items = [
        RoadmapItem(
            item_id="roadmap_feature_1",
            phase_id="architecture",
            title="Resolve API contract",
            type="decision",
            repo_scope=["artifact_store"],
            owner_type="ArchitectureAdjudicatorSubagent",
            success_metrics=["decision:resolved"],
        ),
        RoadmapItem(
            item_id="roadmap_feature_2",
            phase_id="development",
            title="Build implementation",
            type="feature",
            repo_scope=["target"],
            owner_type="ImplementationEngineerSubagent",
            success_metrics=["tests:green"],
        ),
    ]

    rendered = RoadmapCompiler.render_phase_pack(items)

    assert "phase_01.json" in rendered
    assert "phase_04.json" in rendered
    assert "phase_07.json" in rendered
    assert '"phase_id": "questionnaire"' in rendered["phase_04.json"]
    assert '"artifact_folder": "phases/04_questionnaire/"' in rendered["phase_04.json"]
    assert '"Build implementation"' in rendered["phase_07.json"]


class _StateStore:
    def __init__(self) -> None:
        self.items = []

    def record_roadmap_item(self, payload):
        self.items.append(payload)


def test_roadmap_compiler_promotes_comparative_recipe_and_frontier_packets():
    compiler = RoadmapCompiler(_StateStore(), "campaign")

    items = compiler.compile(
        features=[],
        questions=[],
        hypotheses=[],
        repo_dossiers=[
            {
                "repo_id": "analysis_repo",
                "migration_recipes": [
                    {
                        "recipe_id": "analysis_repo:recipe:1",
                        "title": "Native Rewrite token_auth.py into auth.py",
                        "posture": "native_rewrite",
                        "relation_type": "rewrite_candidate",
                        "source_corpus_id": "analysis_repo",
                        "target_corpus_id": "target",
                        "source_path": "token_auth.py",
                        "target_insertion_path": "auth.py",
                        "verification_requirements": ["run affected tests"],
                    }
                ],
                "frontier_packets": [
                    {
                        "packet_id": "analysis_repo:frontier:1",
                        "title": "Experiment native_rewrite for token_auth.py",
                        "priority": 0.84,
                        "posture": "native_rewrite",
                        "source_path": "token_auth.py",
                        "target_path": "auth.py",
                        "recommended_tracks": ["comparative_spike"],
                    }
                ],
            }
        ],
        objective="Promote comparative evidence",
    )

    recipe_item = next(item for item in items if item.type == "comparative_recipe")
    frontier_item = next(item for item in items if item.type == "comparative_frontier")

    assert recipe_item.phase_id == "convergence"
    assert "comparison_backend" in recipe_item.telemetry_contract["minimum"]
    assert "run affected tests" in recipe_item.required_evidence
    assert recipe_item.required_artifacts == ["research", "comparative_reports"]
    assert frontier_item.phase_id == "eid"
    assert frontier_item.metadata["recommended_tracks"] == ["comparative_spike"]

    rendered = RoadmapCompiler.render_phase_pack(items)
    assert '"required_artifacts": [' in rendered["phase_03.json"]
