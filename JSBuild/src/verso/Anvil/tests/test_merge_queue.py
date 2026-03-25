from domains.verification.verification_lane import VerificationLane


class _Verifier:
    def __init__(self, passed: bool) -> None:
        self.passed = passed

    def verify_changes(self, modified_files):
        _ = modified_files
        return {
            "syntax": {"passed": self.passed},
            "lint": {"passed": self.passed},
            "tests": {"passed": self.passed},
            "sentinel": {"passed": self.passed},
            "all_passed": self.passed,
            "runtime_symbols": [],
            "counterexamples": [] if self.passed else ["failing_case"],
        }


def test_merge_queue_gate_blocks_failed_candidate_and_allows_passing_candidate():
    queue = [
        {"lane_id": "lane-a", "result": VerificationLane(_Verifier(False)).run(["core/a.py"])},
        {"lane_id": "lane-b", "result": VerificationLane(_Verifier(True)).run(["core/b.py"])},
    ]

    accepted = [item["lane_id"] for item in queue if not item["result"]["promotion_blocked"]]
    blocked = [item["lane_id"] for item in queue if item["result"]["promotion_blocked"]]

    assert accepted == ["lane-b"]
    assert blocked == ["lane-a"]
