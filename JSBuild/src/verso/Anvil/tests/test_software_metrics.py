from __future__ import annotations

from datetime import datetime, timedelta

from core.analysis.process_metrics import ProcessMetricsTracker
from core.analysis.software_metrics import compute_dsqi, compute_halstead_metrics
from core.analysis.technical_debt_tracker import TechnicalDebtTracker


def test_compute_halstead_metrics_returns_expected_keys() -> None:
    source = "def add(a, b):\n    return a + b\n"
    metrics = compute_halstead_metrics(source)

    assert metrics["volume"] >= 0.0
    assert metrics["difficulty"] >= 0.0
    assert metrics["effort"] >= 0.0


def test_compute_dsqi_returns_structured_score(tmp_path) -> None:
    file_path = tmp_path / "module.py"
    file_path.write_text(
        "def f(x):\n"
        "    if x > 0:\n"
        "        return x\n"
        "    return -x\n",
        encoding="utf-8",
    )
    report = compute_dsqi(str(tmp_path))

    assert 0.0 <= report["score"] <= 1.0
    assert report["functions"] == 1


def test_technical_debt_tracker_accumulates_sqale_dimensions() -> None:
    tracker = TechnicalDebtTracker()
    tracker.add_debt("security", 45)
    tracker.add_debt("testability", 15)

    assert tracker.get_dimension_debt("security") == 45
    assert tracker.total_debt_minutes() == 60
    assert tracker.debt_ratio(120) == 0.5


def test_process_metrics_tracker_computes_mttr_and_rates() -> None:
    tracker = ProcessMetricsTracker()
    now = datetime(2026, 3, 3, 12, 0, 0)
    idx = tracker.record_defect(now, injected_in_change="chg-1")
    tracker.resolve_defect(idx, now + timedelta(hours=4))

    assert tracker.defect_injection_rate(change_count=2) == 0.5
    assert tracker.mttr_hours() == 4.0
    stats = tracker.rca_stats()
    assert stats["total_defects"] == 1.0
    assert stats["resolved_defects"] == 1.0
