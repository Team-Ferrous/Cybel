from __future__ import annotations

import sys
import numpy as np
import pytest
from pathlib import Path
from subprocess import Popen, run
import time

from core.research.eid_master import EIDMasterLoop
from saguaro.analysis.report import ReportGenerator
from saguaro.api import SaguaroAPI
from saguaro.services.comparative import ComparativeAnalysisService
from saguaro.utils.file_utils import build_corpus_manifest


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_corpus_sessions_support_compare_query_and_slice(tmp_path: Path) -> None:
    target = tmp_path / "target"
    external = tmp_path / "repo_analysis" / "external_auth"

    _write(
        target / "auth.py",
        """
def authenticate_user(token: str) -> bool:
    if not token:
        return False
    return token.startswith("anvil-")
""".strip()
        + "\n",
    )
    _write(
        target / "engine.py",
        """
from auth import authenticate_user

def run(token: str) -> bool:
    return authenticate_user(token)
""".strip()
        + "\n",
    )
    _write(
        external / "token_auth.py",
        """
def authenticate(token: str) -> bool:
    if not token:
        return False
    return token.startswith("ext-")
""".strip()
        + "\n",
    )
    _write(external / "CMakeLists.txt", "project(external_auth)\n")

    api = SaguaroAPI(repo_path=str(target))
    created = api.corpus(action="create", path=str(external), alias="external-auth")
    corpus_id = str(created["session"]["corpus_id"])
    assert corpus_id
    assert created["session"]["telemetry"]["native_indexed"] is True
    assert Path(created["session"]["artifact_paths"]["index_manifest"]).exists()
    assert Path(created["session"]["artifact_paths"]["vectors_metadata"]).exists()
    assert Path(created["session"]["artifact_paths"]["graph"]).exists()
    shown = api.corpus(action="show", corpus_id=corpus_id)
    assert shown["analysis_pack"]["producer"] == "ComparativeAnalysisService.native_index"

    listed = api.corpus(action="list")
    assert listed["count"] >= 1

    federated = api.query("authenticate token", k=5, corpus_ids=[corpus_id])
    assert federated["scope"] == "corpus"
    assert federated["results"]
    assert federated["results"][0]["corpus_id"] == corpus_id

    sliced = api.slice("authenticate", corpus_id=corpus_id, depth=1)
    assert sliced["corpus_id"] == corpus_id
    focus = next(item for item in sliced["content"] if item["role"] == "focus")
    assert "def authenticate" in focus["code"]

    comparison = api.compare(
        target=".",
        candidates=[str(external)],
        top_k=5,
    )
    assert comparison["schema_version"] == "comparative_report.v3"
    assert comparison["candidate_count"] == 1
    candidate = comparison["comparisons"][0]
    assert candidate["migration_recipes"]
    assert candidate["port_ledger"]
    assert comparison["phase_packets"]
    assert comparison["portfolio_leaderboard"]
    assert comparison["artifacts"]["phasepack_path"].endswith(".json")
    assert Path(comparison["artifacts"]["phasepack_path"]).exists()
    assert Path(comparison["artifacts"]["leaderboard_path"]).exists()
    assert Path(comparison["artifacts"]["proof_graphs_path"]).exists()
    assert candidate["candidate"]["corpus_id"] == corpus_id
    assert comparison["artifacts"]["json_path"].endswith(".json")
    assert Path(comparison["artifacts"]["json_path"]).exists()
    assert Path(comparison["artifacts"]["markdown_path"]).exists()
    assert Path(comparison["artifacts"]["port_ledger_path"]).exists()
    assert Path(comparison["artifacts"]["frontier_packets_path"]).exists()
    assert comparison["port_ledger"][0]["ledger_entry_id"]
    assert comparison["port_ledger"][0]["report_id"] == comparison["report_id"]
    assert comparison["port_ledger"][0]["verification_refs"]
    assert candidate["summary"]["rank_fusion"]["top1_top2_margin"] >= 0.0
    ranked = (
        candidate["primary_recommendations"]
        or candidate["secondary_recommendations"]
        or candidate["analogous_mechanisms"]
    )
    assert ranked[0]["proof_graph"]["proof_graph_id"]
    assert ranked[0]["calibrated_confidence"] >= 0.0
    assert comparison["comparisons"][0]["summary"]["comparison_backend"] in {
        "native_cpp",
        "python",
        "python_fallback",
    }

    events = (
        target / ".saguaro" / "reality" / "events.jsonl"
    ).read_text(encoding="utf-8")
    assert "comparative_report" in events


def test_compare_reuses_ready_target_manifest_while_index_lock_is_held(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    external = tmp_path / "repo_analysis" / "external_rules"

    _write(
        target / "rules.py",
        """
def apply_rule(name: str) -> str:
    return name.strip().lower()
""".strip()
        + "\n",
    )
    _write(
        external / "normalize.py",
        """
def normalize_rule(name: str) -> str:
    return name.strip().lower()
""".strip()
        + "\n",
    )

    api = SaguaroAPI(repo_path=str(target))
    api.index(path=".", force=True)

    ready_flag = tmp_path / "lock_ready"
    holder = Popen(
        [
            sys.executable,
            "-c",
            (
                "import pathlib, sys, time\n"
                "from saguaro.storage.locks import RepoLockManager\n"
                "manager = RepoLockManager(sys.argv[1])\n"
                "with manager.acquire('index', mode='exclusive', operation='test_hold'):\n"
                "    pathlib.Path(sys.argv[2]).write_text('ready', encoding='utf-8')\n"
                "    time.sleep(3)\n"
            ),
            str(target / ".saguaro"),
            str(ready_flag),
        ],
    )
    try:
        deadline = time.time() + 5.0
        while not ready_flag.exists():
            if time.time() >= deadline:
                raise AssertionError("lock holder did not become ready in time")
            time.sleep(0.05)

        comparison = api.compare(target=".", candidates=[str(external)], top_k=3)
    finally:
        holder.terminate()
        holder.wait(timeout=5)

    telemetry = comparison["telemetry"]
    assert telemetry["compare_lock_wait_ms"] >= 0.0
    assert telemetry["compare_target_manifest_reuse_hits"] == 1
    assert comparison["candidate_count"] == 1


def test_create_session_uses_lean_runtime_without_saguaro_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target"
    external = tmp_path / "repo_analysis" / "external_runtime"

    _write(
        target / "main.py",
        """
def current() -> str:
    return "target"
""".strip()
        + "\n",
    )
    _write(
        external / "feature.py",
        """
def external_feature(name: str) -> str:
    return name.strip().lower()
""".strip()
        + "\n",
    )

    service = ComparativeAnalysisService(str(target))

    def _boom() -> type[object]:
        raise AssertionError("create_session unexpectedly required SaguaroAPI")

    monkeypatch.setattr(service, "_load_api_cls", _boom)

    created = service.create_session(path=str(external), alias="external-runtime")

    assert created["status"] == "ok"
    assert created["session"]["telemetry"]["native_indexed"] is True
    assert Path(created["session"]["artifact_paths"]["index_manifest"]).exists()
    assert Path(created["session"]["artifact_paths"]["vectors_metadata"]).exists()
    assert (
        service._load_session_pack(str(created["session"]["corpus_id"]))["producer"]
        == "ComparativeAnalysisService.native_index"
    )


def test_create_session_reuses_existing_corpus_for_same_root_path(tmp_path: Path) -> None:
    target = tmp_path / "target"
    external = tmp_path / "repo_analysis" / "external_reuse"

    _write(
        target / "main.py",
        """
def current() -> str:
    return "target"
""".strip()
        + "\n",
    )
    _write(
        external / "feature.py",
        """
def external_feature(value: str) -> str:
    return value.strip()
""".strip()
        + "\n",
    )

    service = ComparativeAnalysisService(str(target))

    first = service.create_session(path=str(external), alias="external-reuse-a")
    second = service.create_session(path=str(external), alias="external-reuse-b")

    assert first["status"] == "ok"
    assert second["status"] == "exists"
    assert second["session"]["root_path"] == str(external.resolve())
    assert second["session"]["corpus_id"] == first["session"]["corpus_id"]
    corpora = service.corpus(include_expired=True)
    matches = [
        session
        for session in corpora["sessions"]
        if session["root_path"] == str(external.resolve())
    ]
    assert len(matches) == 1


def test_screen_candidate_indices_uses_native_overlap_prefilter(tmp_path: Path) -> None:
    service = ComparativeAnalysisService(str(tmp_path))

    class _FakeNativeIndexer:
        def screen_overlap_pairs(self, left_tokens, left_lengths, right_tokens, right_lengths, *, top_k: int, num_threads: int = 0):
            assert top_k == 3
            return (
                np.array([[2, 1, -1]], dtype=np.int32),
                np.array([[1.0, 0.5, 0.0]], dtype=np.float32),
            )

    service._native_indexer = _FakeNativeIndexer()
    screened, backend = service._screen_candidate_indices(
        target_token_rows=[([1, 2], ["one", "two"])],
        candidate_token_rows=[
            ([3], ["three"]),
            ([2], ["two"]),
            ([1, 2], ["one", "two"]),
        ],
        top_k=2,
    )

    assert backend == "native_cpp"
    assert screened == {0: [2, 1]}


def test_eid_prefers_comparative_recipes_over_reuse_candidates() -> None:
    dossier = {
        "repo_id": "analysis_repo",
        "migration_recipes": [
            {
                "title": "Native Rewrite token_auth.py into auth.py",
                "posture": "native_rewrite",
                "source_path": "token_auth.py",
                "target_insertion_path": "auth.py",
            }
        ],
        "port_ledger": [
            {
                "relation_type": "rewrite_candidate",
                "posture": "native_rewrite",
                "source_path": "token_auth.py",
                "target_path": "auth.py",
            }
        ],
        "reuse_candidates": [{"path": "legacy.py"}],
    }

    actions = EIDMasterLoop._repo_dossier_actions([dossier])
    assert actions
    assert actions[0]["posture"] == "native_rewrite"
    assert actions[0]["source_path"] == "token_auth.py"


def test_eid_uses_frontier_packets_when_recipes_are_absent() -> None:
    dossier = {
        "repo_id": "analysis_repo",
        "frontier_packets": [
            {
                "packet_id": "analysis_repo:frontier:1",
                "title": "Experiment native_rewrite for token_auth.py",
                "priority": 0.88,
                "posture": "native_rewrite",
                "source_path": "token_auth.py",
                "target_path": "auth.py",
                "recommended_tracks": ["comparative_spike", "verification_lane"],
                "rationale": "Cross-language alignment warrants a focused experiment.",
            }
        ],
        "reuse_candidates": [{"path": "legacy.py"}],
    }

    actions = EIDMasterLoop._repo_dossier_actions([dossier])

    assert actions
    assert actions[0]["frontier_packet_id"] == "analysis_repo:frontier:1"
    assert actions[0]["recommended_tracks"] == [
        "comparative_spike",
        "verification_lane",
    ]
    assert actions[0]["posture"] == "native_rewrite"


def test_comparative_markdown_export_appends(tmp_path: Path) -> None:
    output = tmp_path / "comparative.md"
    generator = ReportGenerator(str(tmp_path))

    report_one = {
        "report_id": "report-one",
        "generated_at": 1.0,
        "target": {"corpus_id": "target"},
        "candidate_count": 1,
        "port_ledger": [{"ledger_entry_id": "entry-1"}],
        "frontier_packets": [{"packet_id": "packet-1"}],
        "artifacts": {"json_path": "report-one.json"},
        "phase_packets": [
            {
                "phase_id": "development",
                "objective": "reporting",
                "allowed_writes": ["Saguaro/saguaro/analysis/report.py"],
            }
        ],
        "portfolio_leaderboard": [
            {
                "corpus_id": "candidate-a",
                "portfolio_rank_score": 0.91,
                "primary_count": 1,
                "secondary_count": 0,
                "no_port_count": 0,
                "build_truth_depth": "structured",
            }
        ],
        "subsystem_upgrade_summary": [
            {
                "subsystem_id": "saguaro",
                "subsystem_label": "Saguaro",
                "primary_count": 1,
                "secondary_count": 0,
                "recommendation_count": 1,
                "creation_candidate_count": 0,
                "top_feature_families": ["reporting"],
                "top_target_paths": ["Saguaro/saguaro/analysis/report.py"],
            }
        ],
        "telemetry": {"report_evidence_density": 3.0},
        "comparisons": [
            {
                "candidate": {"corpus_id": "candidate-a"},
                "summary": {
                    "common_tech_stack": ["python"],
                    "language_overlap": ["python"],
                    "comparison_backend": "native_cpp",
                    "build_alignment": {
                        "compatible": True,
                        "shared_build_files": ["CMakeLists.txt"],
                    },
                    "capability_delta": {
                        "shared_deep_languages": ["python"],
                        "candidate_only_deep_languages": [],
                        "target_only_deep_languages": [],
                    },
                },
                "report_text": "First comparison",
                "primary_recommendations": [
                    {
                        "feature_families": ["reporting"],
                        "source_path": "token_auth.py",
                        "target_path": "Saguaro/saguaro/analysis/report.py",
                        "posture": "transliteration",
                        "relation_type": "portable_pattern",
                        "relation_score": 0.91,
                        "recommended_subsystem_label": "Saguaro",
                        "subsystem_confidence": 0.94,
                        "subsystem_rationale": "feature families reporting favor Saguaro; target path sits in Saguaro",
                        "why_port": "Porting improves operator reporting and auth flow reuse.",
                        "expected_value_summary": "reporting_gain, capability_gain",
                        "why_target_here": "auth.py is the closest runtime insertion surface.",
                        "evidence": {
                            "shared_feature_tags": ["reporting"],
                            "shared_role_tags": ["service"],
                            "noise_flags": [],
                        },
                    }
                ],
                "secondary_recommendations": [],
                "disparate_opportunities": [],
                "low_signal_relations": [],
                "upgrade_clusters": [
                    {
                        "recommended_subsystem_label": "Saguaro",
                        "feature_families": ["reporting"],
                        "target_path": "Saguaro/saguaro/analysis/report.py",
                        "source_paths": ["token_auth.py"],
                        "source_count": 1,
                        "posture": "transliteration",
                        "dominant_relation_type": "portable_pattern",
                        "top_actionability_score": 0.91,
                        "expected_value": ["reporting_gain", "capability_gain"],
                        "program_summary": "Clustered reporting upgrade",
                    }
                ],
                "subsystem_routing_summary": [
                    {
                        "subsystem_id": "saguaro",
                        "subsystem_label": "Saguaro",
                        "primary_count": 1,
                        "secondary_count": 0,
                        "recommendation_count": 1,
                        "creation_candidate_count": 0,
                        "top_feature_families": ["reporting"],
                        "top_target_paths": ["Saguaro/saguaro/analysis/report.py"],
                    }
                ],
                "candidate_scorecard": {
                    "overall_fit": "high",
                    "top_feature_families": ["reporting"],
                    "recommended_subsystems": ["Saguaro"],
                    "primary_recommendation_count": 1,
                    "secondary_recommendation_count": 0,
                    "low_signal_count": 0,
                },
                "value_realization": {
                    "value_category_counts": {"capability_gain": 1, "reporting_gain": 1}
                },
                "manual_validation_seed": [
                    {
                        "source_path": "token_auth.py",
                        "target_path": "Saguaro/saguaro/analysis/report.py",
                        "feature_families": ["reporting"],
                    }
                ],
                "evidence_quality_summary": {"primary_count": 1, "low_signal_count": 0},
                "analogous_mechanisms": [
                    {
                        "relation_score": 0.91,
                        "relation_type": "portable_pattern",
                        "source_path": "token_auth.py",
                        "target_path": "auth.py",
                        "posture": "transliteration",
                    }
                ],
                "migration_recipes": [
                    {
                        "title": "Transliterate token_auth.py into auth.py",
                        "posture": "transliteration",
                        "target_insertion_path": "Saguaro/saguaro/analysis/report.py",
                        "recommended_subsystem_label": "Saguaro",
                    }
                ],
                "port_ledger": [
                    {
                        "status": "candidate",
                        "relation_type": "portable_pattern",
                        "source_path": "token_auth.py",
                        "target_path": "auth.py",
                        "posture": "transliteration",
                        "relation_score": 0.91,
                    }
                ],
                "frontier_packets": [
                    {
                        "title": "Experiment transliteration for token_auth.py",
                        "priority": 0.91,
                        "recommended_subsystem_label": "Saguaro",
                        "recommended_tracks": ["comparative_spike"],
                    }
                ],
                "overlay_graph": {"nodes": [1, 2], "edges": [1]},
            }
        ],
    }
    report_two = {
        "report_id": "report-two",
        "generated_at": 2.0,
        "target": {"corpus_id": "target"},
        "candidate_count": 1,
        "port_ledger": [],
        "frontier_packets": [],
        "phase_packets": [],
        "portfolio_leaderboard": [],
        "comparisons": [
            {
                "candidate": {"corpus_id": "candidate-b"},
                "summary": {
                    "common_tech_stack": ["cpp"],
                    "language_overlap": ["cpp"],
                    "comparison_backend": "python",
                    "build_alignment": {
                        "compatible": False,
                        "shared_build_files": [],
                    },
                    "capability_delta": {
                        "shared_deep_languages": ["cpp"],
                        "candidate_only_deep_languages": [],
                        "target_only_deep_languages": ["python"],
                    },
                },
                "report_text": "Second comparison",
                "primary_recommendations": [],
                "secondary_recommendations": [],
                "disparate_opportunities": [],
                "low_signal_relations": [],
                "candidate_scorecard": {"overall_fit": "low"},
                "value_realization": {"value_category_counts": {}},
                "manual_validation_seed": [],
                "evidence_quality_summary": {"primary_count": 0, "low_signal_count": 0},
                "analogous_mechanisms": [],
                "migration_recipes": [],
                "port_ledger": [],
                "frontier_packets": [],
                "overlay_graph": {"nodes": [], "edges": []},
            }
        ],
    }

    generator.save_comparative_markdown(report_one, str(output))
    generator.save_comparative_markdown(report_two, str(output))

    content = output.read_text(encoding="utf-8")
    assert "Report ID: report-one" in content
    assert "Report ID: report-two" in content
    assert content.count("# Comparative Port Plan: target") == 2
    assert "\n\n---\n\n" in content
    assert "## Executive Summary" in content
    assert "## Artifacts" in content
    assert "## Subsystem Upgrade Routing" in content
    assert "## Phase Packets" in content
    assert "## Portfolio Leaderboard" in content
    assert "### Candidate Scorecard" in content
    assert "### Subsystem Routing" in content
    assert "### Upgrade Programs" in content
    assert "### Canonical Program Groups" in content
    assert "### Primary Port Recommendations" in content
    assert "### Detailed Migration Recipes" in content
    assert "### Low-Signal and Generic Analogues" in content
    assert "### Port Ledger" in content
    assert "### Frontier Packets" in content
    assert "### Overlay Graph" in content


def test_comparative_markdown_export_writes_per_candidate_reports(tmp_path: Path) -> None:
    output = tmp_path / "comparative_fleet.md"
    generator = ReportGenerator(str(tmp_path))
    report = {
        "report_id": "fleet-report",
        "generated_at": 1.0,
        "target": {"corpus_id": "target"},
        "candidate_count": 2,
        "port_ledger": [],
        "frontier_packets": [],
        "comparisons": [
            {
                "candidate": {
                    "corpus_id": "candidate-a",
                    "root_path": str(tmp_path / "repo_analysis" / "candidate_a"),
                },
                "summary": {
                    "common_tech_stack": ["python"],
                    "language_overlap": ["python"],
                    "comparison_backend": "native_cpp",
                    "build_alignment": {"compatible": True, "shared_build_files": []},
                    "capability_delta": {
                        "shared_deep_languages": ["python"],
                        "candidate_only_deep_languages": [],
                        "target_only_deep_languages": [],
                    },
                },
                "report_text": "Candidate A",
                "primary_recommendations": [],
                "secondary_recommendations": [],
                "disparate_opportunities": [],
                "low_signal_relations": [],
                "candidate_scorecard": {"overall_fit": "medium"},
                "value_realization": {"value_category_counts": {}},
                "manual_validation_seed": [],
                "evidence_quality_summary": {"primary_count": 0, "low_signal_count": 0},
                "analogous_mechanisms": [],
                "migration_recipes": [],
                "port_ledger": [],
                "frontier_packets": [],
                "overlay_graph": {"nodes": [], "edges": []},
            },
            {
                "candidate": {
                    "corpus_id": "candidate-b",
                    "root_path": str(tmp_path / "repo_analysis" / "candidate_b"),
                },
                "summary": {
                    "common_tech_stack": ["cpp"],
                    "language_overlap": ["cpp"],
                    "comparison_backend": "native_cpp",
                    "build_alignment": {"compatible": True, "shared_build_files": []},
                    "capability_delta": {
                        "shared_deep_languages": ["cpp"],
                        "candidate_only_deep_languages": [],
                        "target_only_deep_languages": [],
                    },
                },
                "report_text": "Candidate B",
                "primary_recommendations": [],
                "secondary_recommendations": [],
                "disparate_opportunities": [],
                "low_signal_relations": [],
                "candidate_scorecard": {"overall_fit": "medium"},
                "value_realization": {"value_category_counts": {}},
                "manual_validation_seed": [],
                "evidence_quality_summary": {"primary_count": 0, "low_signal_count": 0},
                "analogous_mechanisms": [],
                "migration_recipes": [],
                "port_ledger": [],
                "frontier_packets": [],
                "overlay_graph": {"nodes": [], "edges": []},
            },
        ],
        "artifacts": {},
        "telemetry": {},
    }

    generator.save_comparative_markdown(report, str(output))

    content = output.read_text(encoding="utf-8")
    assert "## Candidate Reports" in content
    candidate_dir = tmp_path / "comparative_fleet_reports"
    assert candidate_dir.exists()
    assert (candidate_dir / "candidate-a.md").exists()
    assert (candidate_dir / "candidate-b.md").exists()
    assert report["artifacts"]["candidate_markdown_dir"] == str(candidate_dir)
    assert set(report["artifacts"]["candidate_markdown_paths"]) == {
        "candidate-a",
        "candidate-b",
    }


def test_record_tagging_captures_port_relevant_roles(tmp_path: Path) -> None:
    service = ComparativeAnalysisService(str(tmp_path))

    attack_tags = service._tags_for_record(
        "counterfit/core/attacks.py",
        classification="source",
        language="python",
    )
    target_tags = service._tags_for_record(
        "counterfit/core/targets.py",
        classification="source",
        language="python",
    )
    reporting_tags = service._tags_for_record(
        "counterfit/reporting/text.py",
        classification="source",
        language="python",
    )

    assert {"attack", "core_runtime"} <= attack_tags
    assert {"target", "core_runtime"} <= target_tags
    assert {"reporting", "artifact", "report_surface"} <= reporting_tags


def test_compare_emits_tiered_and_disparate_recommendations(tmp_path: Path) -> None:
    target = tmp_path / "target"
    external = tmp_path / "repo_analysis" / "candidate_ops"

    _write(
        target / "core" / "registry.py",
        """
class TargetRegistry:
    def register_target(self, name: str) -> str:
        return name.lower()
""".strip()
        + "\n",
    )
    _write(
        target / "artifacts" / "report_output.py",
        """
def render_report(name: str) -> str:
    return f"report:{name}"
""".strip()
        + "\n",
    )
    _write(
        external / "counterfit" / "core" / "targets.py",
        """
class TargetRegistry:
    def register_target(self, name: str) -> str:
        return name.strip().lower()
""".strip()
        + "\n",
    )
    _write(
        external / "counterfit" / "reporting" / "text.py",
        """
def render_report(name: str) -> str:
    return f"candidate:{name}"
""".strip()
        + "\n",
    )

    service = ComparativeAnalysisService(str(target))
    comparison_report = service.compare(candidates=[str(external)], top_k=8)

    comparison = comparison_report["comparisons"][0]
    promoted = (
        comparison["primary_recommendations"]
        + comparison["secondary_recommendations"]
        + comparison["analogous_mechanisms"]
    )
    families = {
        family for relation in promoted for family in relation.get("feature_families", [])
    }

    assert promoted
    assert comparison["manual_validation_seed"]
    assert "target_registry" in families or "reporting" in families
    assert comparison["evidence_quality_summary"]["low_signal_count"] >= 0
    assert comparison["upgrade_clusters"]
    assert comparison["subsystem_routing_summary"]
    assert {relation.get("recommended_subsystem") for relation in promoted}


def test_subsystem_routing_prefers_saguaro_and_qsg_surfaces(tmp_path: Path) -> None:
    service = ComparativeAnalysisService(str(tmp_path))

    saguaro_route = service._subsystem_routing(
        target_path="Saguaro/saguaro/query/pipeline.py",
        feature_families=["query_engine"],
        shared_role_tags=["service"],
    )
    qsg_route = service._subsystem_routing(
        target_path="core/qsg/continuous_engine.py",
        feature_families=["evaluation_pipeline"],
        shared_role_tags=["core_runtime"],
    )

    assert saguaro_route["recommended_subsystem"] == "saguaro"
    assert saguaro_route["recommended_subsystem_label"] == "Saguaro"
    assert qsg_route["recommended_subsystem"] == "qsg"
    assert qsg_route["recommended_subsystem_label"] == "QSG"


def test_cli_help_exposes_corpus_and_compare_commands() -> None:
    compare = run(
        [sys.executable, "-m", "saguaro.cli", "compare", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert compare.returncode == 0
    assert "--candidate" in compare.stdout
    assert "--fleet-root" in compare.stdout
    assert "--reuse-only" in compare.stdout

    corpus = run(
        [sys.executable, "-m", "saguaro.cli", "corpus", "create", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert corpus.returncode == 0
    assert "--ttl-hours" in corpus.stdout
    assert "--no-quarantine" in corpus.stdout

    corpus_benchmark = run(
        [sys.executable, "-m", "saguaro.cli", "corpus", "benchmark", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert corpus_benchmark.returncode == 0
    assert "--batch-size" in corpus_benchmark.stdout
    assert "--file-batch-size" in corpus_benchmark.stdout


def test_compare_fleet_root_treats_each_top_level_directory_as_repo_and_runs_sequentially(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    fleet_root = tmp_path / "repo_analysis"
    repo_a = fleet_root / "repo_a"
    repo_b = fleet_root / "repo_b"

    _write(
        target / "auth.py",
        """
def authenticate_user(token: str) -> bool:
    return bool(token)
""".strip()
        + "\n",
    )
    _write(
        repo_a / "module.py",
        """
def alpha_feature() -> bool:
    return True
""".strip()
        + "\n",
    )
    _write(
        repo_b / "module.py",
        """
def beta_feature() -> bool:
    return True
""".strip()
        + "\n",
    )

    api = SaguaroAPI(repo_path=str(target))
    result = api.compare(target=".", fleet_root=str(fleet_root), top_k=3)

    assert result["candidate_count"] == 2
    assert result["telemetry"]["fleet_processing_mode"] == "sequential"
    assert len(result["telemetry"]["fleet_processing_order"]) == 2
    candidate_ids = {
        comparison["candidate"]["corpus_id"]
        for comparison in result["comparisons"]
    }
    assert set(result["telemetry"]["fleet_processing_order"]) == candidate_ids


def test_compare_reuse_only_skips_missing_fleet_candidates(tmp_path: Path) -> None:
    target = tmp_path / "target"
    fleet_root = tmp_path / "repo_analysis"
    repo_a = fleet_root / "repo_a"
    repo_b = fleet_root / "repo_b"

    _write(
        target / "auth.py",
        """
def authenticate_user(token: str) -> bool:
    return bool(token)
""".strip()
        + "\n",
    )
    _write(
        repo_a / "module.py",
        """
def alpha_feature() -> bool:
    return True
""".strip()
        + "\n",
    )
    _write(
        repo_b / "module.py",
        """
def beta_feature() -> bool:
    return True
""".strip()
        + "\n",
    )

    api = SaguaroAPI(repo_path=str(target))
    created = api.corpus(action="create", path=str(repo_a), alias="repo-a")
    existing_corpus_id = str(created["session"]["corpus_id"])

    result = api.compare(
        target=".",
        fleet_root=str(fleet_root),
        top_k=3,
        reuse_only=True,
    )

    assert result["candidate_count"] == 1
    assert result["telemetry"]["reuse_only"] is True
    assert result["telemetry"]["skipped_candidate_count"] == 1
    assert result["comparisons"][0]["candidate"]["corpus_id"] == existing_corpus_id
    assert result["telemetry"]["skipped_candidates"][0]["reason"] == "reuse_only_missing_corpus"
    assert result["telemetry"]["skipped_candidates"][0]["path"].endswith("repo_b")


def test_external_corpus_prunes_common_junk_dirs_for_non_git_repo(tmp_path: Path) -> None:
    target = tmp_path / "target"
    external = tmp_path / "repo_analysis" / "external_big"

    _write(
        target / "auth.py",
        """
def authenticate_user(token: str) -> bool:
    return bool(token)
""".strip()
        + "\n",
    )
    _write(
        external / "src" / "main.py",
        """
def useful_feature() -> bool:
    return True
""".strip()
        + "\n",
    )
    _write(
        external / "node_modules" / "huge_vendor.py",
        """
def vendored_noise() -> bool:
    return True
""".strip()
        + "\n",
    )
    _write(
        external / "build" / "generated.py",
        """
def generated_noise() -> bool:
    return True
""".strip()
        + "\n",
    )

    api = SaguaroAPI(repo_path=str(target))
    created = api.corpus(action="create", path=str(external), alias="external-big")
    corpus_id = str(created["session"]["corpus_id"])
    shown = api.corpus(action="show", corpus_id=corpus_id)
    pack = shown["analysis_pack"]
    indexed_paths = {item["path"] for item in pack.get("files", [])}

    assert "src/main.py" in indexed_paths
    assert "node_modules/huge_vendor.py" not in indexed_paths
    assert "build/generated.py" not in indexed_paths


def test_manifest_builder_handles_nested_repo_under_ignored_parent(tmp_path: Path) -> None:
    main_repo = tmp_path / "main"
    nested = main_repo / "repo_analysis" / "external_repo"
    _write(main_repo / ".gitignore", "repo_analysis/\n")
    _write(nested / "shared" / "cpp" / "Diagnostics.cpp", "int meaning() { return 42; }\n")
    _write(nested / "docs" / "codeql" / "qllexer.py", "def lex():\n    return True\n")

    run(["git", "init"], cwd=main_repo, check=True, capture_output=True, text=True)

    manifest = build_corpus_manifest(str(nested))
    indexed = {Path(item).relative_to(nested).as_posix() for item in manifest.files}

    assert "shared/cpp/Diagnostics.cpp" in indexed
    assert "docs/codeql/qllexer.py" in indexed


def test_corpus_benchmark_reports_cold_and_warm_runs(tmp_path: Path) -> None:
    target = tmp_path / "target"
    external = tmp_path / "repo_analysis" / "bench_repo"

    _write(
        target / "auth.py",
        """
def authenticate_user(token: str) -> bool:
    return bool(token)
""".strip()
        + "\n",
    )
    _write(
        external / "src" / "main.py",
        """
def useful_feature() -> bool:
    return True
""".strip()
        + "\n",
    )

    api = SaguaroAPI(repo_path=str(target))
    result = api.corpus(
        action="benchmark",
        path=str(external),
        alias="bench-repo",
        batch_sizes=[64],
        iterations=1,
        reuse_check=True,
    )

    assert result["status"] == "ok"
    assert result["benchmark_kind"] == "corpus_index"
    assert len(result["results"]) == 1
    entry = result["results"][0]
    assert entry["batch_size"] == 64
    assert entry["cold_runs"][0]["status"] == "ok"
    assert entry["cold_runs"][0]["peak_rss_mb"] >= 0.0
    assert entry["cold_summary"]["avg_wall_ms"] is not None
    assert entry["warm_runs"][0]["status"] == "ok"
    assert entry["warm_runs"][0]["session_status"] in {"ok", "exists"}
    assert result["summary"]["fastest_cold"]["batch_size"] == 64


def test_compare_artifact_markdown_defaults_to_repo_root(tmp_path: Path) -> None:
    target = tmp_path / "target"
    external = tmp_path / "repo_analysis" / "external_auth"

    _write(
        target / "auth.py",
        """
def authenticate_user(token: str) -> bool:
    return bool(token)
""".strip()
        + "\n",
    )
    _write(
        external / "token_auth.py",
        """
def authenticate(token: str) -> bool:
    return bool(token)
""".strip()
        + "\n",
    )

    api = SaguaroAPI(repo_path=str(target))
    comparison = api.compare(target=".", candidates=[str(external)], top_k=3)
    markdown_path = Path(comparison["artifacts"]["markdown_path"])

    assert markdown_path.exists()
    assert markdown_path.parent == target / "comparative_reports"
    assert ".saguaro" not in markdown_path.as_posix()


def test_compare_prefers_native_cpp_rewrite_for_feature_level_matches(tmp_path: Path) -> None:
    target = tmp_path / "target"
    external = tmp_path / "repo_analysis" / "codeql_like"

    _write(
        target / "src" / "security" / "vulnerability_report.cpp",
        """
int emit_vulnerability_report() {
    return 1;
}
""".strip()
        + "\n",
    )
    _write(
        external / "ql" / "security" / "vulnerability_report.qll",
        """
module VulnerabilityReport
""".strip()
        + "\n",
    )

    api = SaguaroAPI(repo_path=str(target))
    comparison = api.compare(target=".", candidates=[str(external)], top_k=3)

    result = comparison["comparisons"][0]
    assert result["summary"]["preferred_implementation_language"] == "cpp"
    assert "security_analysis" in result["summary"]["feature_overlap"]
    assert "reporting" in result["summary"]["feature_overlap"]
    assert result["migration_recipes"]
    assert result["migration_recipes"][0]["posture"] == "native_rewrite"
    assert (
        result["migration_recipes"][0]["preferred_implementation_language"] == "cpp"
    )
    assert any(
        "thin wrappers" in invariant
        for invariant in result["migration_recipes"][0]["invariants"]
    )
