import sys

from core.campaign.runner import CampaignRunner
from core.campaign.state_store import CampaignStateStore
from core.research.experiment_runner import ExperimentRunner


def make_runner(tmp_path):
    return CampaignRunner(
        config={
            "generated_dir": str(tmp_path / "generated"),
            "custom_dir": str(tmp_path / "custom"),
            "state_dir": str(tmp_path / "state"),
            "workspace_dir": str(tmp_path / "managed"),
            "ledger_db_path": str(tmp_path / "ledger.db"),
        }
    )


def test_managed_campaign_lifecycle(tmp_path):
    runner = make_runner(tmp_path)
    target_repo = tmp_path / "repo"
    target_repo.mkdir()

    created = runner.create_autonomy_campaign(
        name="Autonomy Test",
        objective="Implement an autonomy control plane",
        directives=["Use evidence-backed loops"],
        root_dir=str(target_repo),
    )
    campaign_id = created["campaign_id"]

    attach = runner.attach_repo(
        campaign_id,
        repo_path=str(target_repo),
        role="analysis_local",
    )
    assert attach["role"] == "analysis_local"

    questionnaire = runner.build_questionnaire(campaign_id)
    assert questionnaire["count"] >= 1

    feature_map = runner.build_feature_map(campaign_id)
    assert "Feature Map" not in feature_map["rendered"] or isinstance(
        feature_map["rendered"], str
    )

    roadmap = runner.build_roadmap(campaign_id)
    assert isinstance(roadmap["validation_errors"], list)
    workspace = created["workspace"]
    intake_manifest = (
        tmp_path / "managed" / campaign_id / "phases" / "01_intake" / "manifest.json"
    )
    roadmap_manifest = (
        tmp_path
        / "managed"
        / campaign_id
        / "phases"
        / "06_roadmap_draft"
        / "manifest.json"
    )
    assert intake_manifest.exists()
    assert roadmap_manifest.exists()
    store = CampaignStateStore(workspace["db_path"])
    phase_artifacts = store.list_phase_artifacts(campaign_id)
    assert any(item["phase_id"] == "01_intake" for item in phase_artifacts)
    assert any(item["phase_id"] == "06_roadmap_draft" for item in phase_artifacts)
    assert any(item["artifact_type"] == "roadmap_phase" for item in phase_artifacts)

    artifacts = runner.list_artifacts(campaign_id)
    families = {item["family"] for item in artifacts}
    assert {"intake", "architecture", "feature_map", "roadmap_draft"} <= families

    audit = runner.run_audit(campaign_id)
    assert "summary" in audit

    closure = runner.closure_proof(campaign_id)
    assert "proof" in closure
    assert "closure_allowed" in closure["proof"]


def test_extended_autonomy_workflow_with_acquisition_research_and_eid(tmp_path):
    runner = make_runner(tmp_path)
    target_repo = tmp_path / "target_repo"
    analysis_repo = tmp_path / "analysis_repo"
    target_repo.mkdir()
    analysis_repo.mkdir()
    (analysis_repo / "kernel.cpp").write_text(
        "int main() { return 0; }\n", encoding="utf-8"
    )

    created = runner.create_autonomy_campaign(
        name="Autonomy Full Flow",
        objective="Build a deterministic autonomy operating system",
        directives=["Prioritize telemetry and evidence"],
        root_dir=str(target_repo),
    )
    campaign_id = created["campaign_id"]

    acquired = runner.acquire_repos(campaign_id, repo_specs=[str(analysis_repo)])
    assert acquired["count"] == 1

    research = runner.run_research(
        campaign_id,
        sources=[
            {
                "url": "https://example.test/docs",
                "title": "Telemetry Docs",
                "content": "observability and telemetry coverage guidance",
                "topic": "observability_and_telemetry",
                "summary": "Telemetry guidance for campaign loops",
            }
        ],
    )
    assert research["evaluation"]["replayable"] is True
    assert research["repo_analysis"]["imported_files"] >= 1
    assert research["repo_analysis"]["imported_symbol_traces"] >= 0

    questionnaire = runner.build_questionnaire(campaign_id)
    assert questionnaire["count"] >= 1
    runner.approve_artifact(
        campaign_id, f"{campaign_id}:architecture_questionnaire", state="accepted"
    )

    feature_map = runner.build_feature_map(campaign_id)
    assert feature_map["count"] >= 1
    runner.approve_artifact(campaign_id, f"{campaign_id}:feature_map")

    eid = runner.run_eid(campaign_id)
    assert eid["hypotheses"]
    assert eid["eid"]["innovation_hypotheses"]
    assert eid["eid"]["specialist_packets"]
    assert eid["eid"]["experimental_tracks"]
    assert eid["lane_runs"]
    assert eid["lane_runs"][0]["promotion"]["verdict"] in {"keep", "discard"}

    roadmap = runner.build_roadmap(campaign_id)
    assert isinstance(roadmap["validation_errors"], list)
    assert any(item["phase_id"] == "eid" for item in roadmap["items"])
    assert any(item["type"] == "experiment_lane" for item in roadmap["items"])

    final_roadmap = runner.promote_final_roadmap(campaign_id)
    assert final_roadmap["items"]

    status = runner.get_campaign_status(campaign_id)
    assert status["control_plane"]["runtime_state"] == "DEVELOPMENT"

    artifacts = runner.list_artifacts(campaign_id)
    families = {item["family"] for item in artifacts}
    assert {
        "research",
        "experiments",
        "whitepapers",
        "roadmap_final",
    } <= families

    workspace = created["workspace"]
    store = CampaignStateStore(workspace["db_path"])
    repo_file_docs = store.fetchall(
        """
        SELECT title, repo_context
        FROM research_documents
        WHERE campaign_id = ? AND repo_context LIKE ?
        ORDER BY title ASC
        """,
        (campaign_id, "analysis_local:%"),
    )
    assert any(item["title"] == "kernel.cpp" for item in repo_file_docs)
    traces = store.list_usage_traces(campaign_id)
    assert traces
    convergence = store.list_convergence_checkpoints(campaign_id, "research")
    assert convergence and convergence[0]["metrics"]["repo_files_imported"] >= 1


def test_research_rerun_is_idempotent_for_chunks_and_claims(tmp_path):
    runner = make_runner(tmp_path)
    target_repo = tmp_path / "target_repo"
    analysis_repo = tmp_path / "analysis_repo"
    target_repo.mkdir()
    analysis_repo.mkdir()
    (analysis_repo / "kernel.cpp").write_text(
        "int main() { return 0; }\n", encoding="utf-8"
    )

    created = runner.create_autonomy_campaign(
        name="Autonomy Research Replay",
        objective="Keep research persistence deterministic across reruns",
        directives=["Avoid duplicate chunk ingestion"],
        root_dir=str(target_repo),
    )
    campaign_id = created["campaign_id"]
    runner.acquire_repos(campaign_id, repo_specs=[str(analysis_repo)])

    source = {
        "url": "https://example.test/docs",
        "title": "Telemetry Docs",
        "content": "observability and telemetry coverage guidance",
        "topic": "observability_and_telemetry",
        "summary": "Telemetry guidance for campaign loops",
        "novelty_score": 0.6,
        "confidence": 0.7,
    }

    first = runner.run_research(campaign_id, sources=[source])
    second = runner.run_research(campaign_id, sources=[source])

    assert len(first["claims"]) == len(second["claims"])

    workspace = created["workspace"]
    store = CampaignStateStore(workspace["db_path"])
    chunk_count = store.fetchone(
        """
        SELECT COUNT(*) AS count
        FROM source_chunks
        WHERE campaign_id = ? AND repo_context = ?
        """,
        (campaign_id, "analysis_external:web"),
    )
    claim_count = store.fetchone(
        """
        SELECT COUNT(*) AS count
        FROM research_claims
        WHERE campaign_id = ? AND repo_context = ?
        """,
        (campaign_id, "analysis_external:web"),
    )
    cluster_count = store.fetchone(
        "SELECT COUNT(*) AS count FROM topic_clusters WHERE campaign_id = ?",
        (campaign_id,),
    )
    repo_file_claims = store.fetchone(
        """
        SELECT COUNT(*) AS count
        FROM research_claims
        WHERE campaign_id = ? AND repo_context LIKE ?
        """,
        (campaign_id, "analysis_local:%"),
    )

    assert int(chunk_count["count"]) == 1
    assert int(claim_count["count"]) == 1
    assert int(cluster_count["count"]) == len(second["clusters"])
    assert second["evaluation"]["browser"]["count"] == 1
    assert int(repo_file_claims["count"]) >= 1


def test_experiment_runner_records_metrics_and_timeout(tmp_path):
    store = CampaignStateStore(str(tmp_path / "state.db"))
    runner = ExperimentRunner("campaign-opt", store, cwd=str(tmp_path))

    result = runner.run(
        name="metric_probe",
        commands=[
            {
                "label": "metric_emit",
                "argv": [
                    sys.executable,
                    "-c",
                    "print('latency_ms=12.5\\nreplayability=1')",
                ],
            },
            {
                "label": "timeout_probe",
                "argv": [
                    sys.executable,
                    "-c",
                    "import time; time.sleep(0.2)",
                ],
                "timeout_seconds": 0,
            },
        ],
        default_timeout_seconds=5,
    )

    assert result["summary_metrics"]["aggregate_metrics"]["latency_ms"] == 12.5
    assert result["summary_metrics"]["aggregate_metrics"]["replayability"] == 1.0
    assert any(item["status"] == "timed_out" for item in result["runs"])


def test_experiment_runner_lane_runtime_records_promotion_and_telemetry(tmp_path):
    store = CampaignStateStore(str(tmp_path / "state.db"))
    runner = ExperimentRunner("campaign-lane", store, cwd=str(tmp_path))
    metadata_path = tmp_path / "campaign.json"
    metadata_path.write_text("{}", encoding="utf-8")

    result = runner.run_lane(
        {
            "lane_id": "lane_probe",
            "caller_mode": "eid",
            "lane_type": "bounded_file_experiment",
            "name": "artifact_resume_replay",
            "objective_function": "Verify the shared lane runtime.",
            "commands": [
                {
                    "label": "replayability_probe",
                    "argv": [
                        sys.executable,
                        "-c",
                        "print('replayability=1\\ndeterminism_pass=1\\ncorrectness_pass=1')",
                    ],
                }
            ],
            "telemetry_contract": {
                "required_metrics": [
                    "wall_time_seconds",
                    "command_count",
                    "success_count",
                    "failure_count",
                    "correctness_pass",
                    "determinism_pass",
                ],
                "minimum_success_count": 1,
            },
            "promotion_policy": {"name": "experimental_eid", "minimum_score": 0.0},
            "metadata": {"imported_baseline_metrics": {"determinism_pass": 1.0}},
            "read_only_scope": [str(metadata_path)],
            "allowed_writes": ["artifact_store"],
        }
    )

    assert result["telemetry_check"]["contract_satisfied"] is True
    assert result["promotion"]["verdict"] == "keep"
    assert result["scorecard"]["score"] >= 0.0

    telemetry = store.list_telemetry("campaign-lane")
    assert any(item["telemetry_kind"] == "experiment_lane" for item in telemetry)


def test_experiment_runner_executes_inside_lane_overlay(tmp_path):
    store = CampaignStateStore(str(tmp_path / "state.db"))
    runner = ExperimentRunner("campaign-overlay", store, cwd=str(tmp_path))
    editable = tmp_path / "editable.txt"
    editable.write_text("source\n", encoding="utf-8")

    result = runner.run_lane(
        {
            "lane_id": "lane_overlay",
            "caller_mode": "development",
            "lane_type": "bounded_file_experiment",
            "name": "overlay_write",
            "objective_function": "Edit only inside the lane overlay.",
            "commands": [
                {
                    "label": "mutate_overlay",
                    "argv": [
                        sys.executable,
                        "-c",
                        "from pathlib import Path; Path('editable.txt').write_text('lane\\n', encoding='utf-8'); print('correctness_pass=1\\ndeterminism_pass=1')",
                    ],
                }
            ],
            "editable_scope": [str(editable)],
            "telemetry_contract": {
                "required_metrics": ["correctness_pass", "determinism_pass"],
                "minimum_success_count": 1,
            },
            "promotion_policy": {"minimum_score": 0.0},
            "allowed_writes": ["target"],
        }
    )

    assert editable.read_text(encoding="utf-8") == "source\n"
    assert result["branch_metrics"]["changed_files"] == ["editable.txt"]
