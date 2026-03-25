import json
import subprocess
import sys
from pathlib import Path

from core.campaign.state_store import CampaignStateStore
from core.memory.fabric import (
    LatentPackageRecord,
    MemoryBenchmarkRunner,
    MemoryFabricStore,
    MemoryProjector,
    MemoryRetrievalPlanner,
)


def test_memory_benchmark_runner_scores_retrieval_and_replay(tmp_path):
    store = CampaignStateStore(str(tmp_path / "state.db"))
    fabric = MemoryFabricStore(store, storage_root=str(tmp_path / "fabric"))
    projector = MemoryProjector()
    planner = MemoryRetrievalPlanner(fabric, projector)

    evidence = fabric.create_memory(
        memory_kind="research_claim",
        payload_json={"topic": "replay", "statement": "Latent replay reduces prompt churn."},
        campaign_id="campaign-1",
        workspace_id="campaign-1",
        source_system="test",
        summary_text="Latent replay reduces prompt churn.",
        provenance_json={"source": "unit-test"},
    )
    distractor = fabric.create_memory(
        memory_kind="research_claim",
        payload_json={"topic": "batching", "statement": "Continuous batching increases throughput."},
        campaign_id="campaign-1",
        workspace_id="campaign-1",
        source_system="test",
        summary_text="Continuous batching increases throughput.",
        provenance_json={"source": "unit-test"},
    )
    projector.project_memory(fabric, evidence, include_multivector=True)
    projector.project_memory(fabric, distractor, include_multivector=True)
    fabric.put_latent_package(
        LatentPackageRecord(
            latent_package_id="benchmark-latent",
            memory_id=evidence.memory_id,
            model_family="qsg-python",
            model_revision="v1",
            tokenizer_hash="tok",
            prompt_protocol_hash="almf.v1",
            hidden_dim=2,
            qsg_runtime_version="qsg.v1",
            capture_stage="benchmark",
        ),
        tensor=[[0.3, 0.4]],
    )

    runner = MemoryBenchmarkRunner(fabric, planner)
    summary = runner.run_suite(
        campaign_id="campaign-1",
        cases=[
            {
                "family": "research_recall",
                "query_text": "latent replay prompt churn",
                "expected_memory_ids": [evidence.memory_id],
                "contradiction_memory_ids": [],
                "limit": 2,
            },
            {
                "family": "development_replay",
                "query_text": "latent replay prompt churn",
                "expected_memory_ids": [evidence.memory_id],
                "replay_memory_id": evidence.memory_id,
                "model_family": "qsg-python",
                "hidden_dim": 2,
                "tokenizer_hash": "tok",
                "prompt_protocol_hash": "almf.v1",
                "qsg_runtime_version": "qsg.v1",
                "prompt_reconstruction_tokens": 128,
                "warm_start_tokens": 16,
                "limit": 2,
            },
        ],
    )

    assert summary["core_metrics"]["top_k_evidence_recall"] > 0.0
    assert summary["core_metrics"]["ndcg"] > 0.0
    assert summary["family_metrics"]["development_replay"]["exact_replay_success_rate"] == 1.0
    assert summary["benchmark_gates"]["passed"] is True


def test_almf_benchmark_script_runs_directly_from_repo_root(tmp_path):
    store = CampaignStateStore(str(tmp_path / "state.db"))
    fabric = MemoryFabricStore(store, storage_root=str(tmp_path / "fabric"))
    projector = MemoryProjector()

    evidence = fabric.create_memory(
        memory_kind="research_claim",
        payload_json={"topic": "replay", "statement": "Latent replay reduces prompt churn."},
        campaign_id="campaign-1",
        workspace_id="campaign-1",
        source_system="test",
        summary_text="Latent replay reduces prompt churn.",
        provenance_json={"source": "unit-test"},
    )
    projector.project_memory(fabric, evidence, include_multivector=True)
    fabric.put_latent_package(
        LatentPackageRecord(
            latent_package_id="benchmark-latent",
            memory_id=evidence.memory_id,
            model_family="qsg-python",
            model_revision="v1",
            tokenizer_hash="tok",
            prompt_protocol_hash="almf.v1",
            hidden_dim=2,
            qsg_runtime_version="qsg.v1",
            capture_stage="benchmark",
        ),
        tensor=[[0.3, 0.4]],
    )

    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps(
            [
                {
                    "family": "development_replay",
                    "query_text": "latent replay prompt churn",
                    "expected_memory_ids": [evidence.memory_id],
                    "replay_memory_id": evidence.memory_id,
                    "model_family": "qsg-python",
                    "hidden_dim": 2,
                    "tokenizer_hash": "tok",
                    "prompt_protocol_hash": "almf.v1",
                    "qsg_runtime_version": "qsg.v1",
                    "prompt_reconstruction_tokens": 128,
                    "warm_start_tokens": 16,
                    "limit": 1,
                }
            ]
        ),
        encoding="utf-8",
    )

    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "benchmarks/almf_retrieval_replay_benchmark.py",
            "--db-path",
            str(tmp_path / "state.db"),
            "--campaign-id",
            "campaign-1",
            "--cases",
            str(cases_path),
            "--storage-root",
            str(tmp_path / "fabric"),
            "--out-root",
            str(tmp_path / "runs"),
            "--run-id",
            "almf_direct_script",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["benchmark_gates"]["passed"] is True
    assert payload["family_metrics"]["development_replay"]["exact_replay_success_rate"] == 1.0
