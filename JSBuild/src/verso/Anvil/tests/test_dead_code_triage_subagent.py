import json

from core.agents.domain.toolchain.dead_code_triage_subagent import (
    DeadCodeTriageSubagent,
)


def test_parse_deadcode_payload_with_prefixed_logs():
    payload = "INFO startup\n" + json.dumps(
        {
            "threshold": 0.5,
            "count": 1,
            "candidates": [
                {
                    "symbol": "DemoSubagent",
                    "file": "/workspace/demo.py",
                    "confidence": 0.7,
                    "reason": "No static references found",
                    "dynamic_file": False,
                    "public": True,
                }
            ],
        }
    )
    parsed = DeadCodeTriageSubagent.parse_deadcode_payload(payload)
    assert parsed["count"] == 1


def test_classify_subagent_candidate_as_rewire():
    decision = DeadCodeTriageSubagent.classify_candidate(
        {
            "symbol": "SensorFusionSubagent",
            "file": "/repo/core/agents/domain/robotics/sensor_fusion_subagent.py",
            "confidence": 0.7,
            "reason": "No static references found",
            "dynamic_file": False,
            "public": True,
        }
    )
    assert decision.disposition == "rewire_candidate"
    assert decision.suggested_owner == "CampaignDirectorSubagent"


def test_classify_private_candidate_as_likely_obsolete():
    decision = DeadCodeTriageSubagent.classify_candidate(
        {
            "symbol": "_internal_helper",
            "file": "/repo/core/thing.py",
            "confidence": 0.81,
            "reason": "No static references found",
            "dynamic_file": False,
            "public": False,
        }
    )
    assert decision.disposition == "likely_obsolete"


def test_triage_report_buckets_decisions():
    report = {
        "threshold": 0.5,
        "count": 3,
        "candidates": [
            {
                "symbol": "ExampleSubagent",
                "file": "/repo/core/agents/domain/example_subagent.py",
                "confidence": 0.7,
                "reason": "No static references found",
                "dynamic_file": False,
                "public": True,
            },
            {
                "symbol": "_helper",
                "file": "/repo/core/helper.py",
                "confidence": 0.8,
                "reason": "No static references found",
                "dynamic_file": False,
                "public": False,
            },
            {
                "symbol": "mystery",
                "file": "/repo/core/mystery.py",
                "confidence": 0.3,
                "reason": "ambiguous",
                "dynamic_file": False,
                "public": False,
            },
        ],
    }
    triage = DeadCodeTriageSubagent.triage_report(report)
    assert triage["summary"]["rewire_candidate"] == 1
    assert triage["summary"]["likely_obsolete"] == 1
    assert triage["summary"]["needs_manual_review"] == 1
