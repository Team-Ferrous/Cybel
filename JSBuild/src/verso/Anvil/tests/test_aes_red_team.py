from core.aes.red_team_protocol import RedTeamProtocol


def test_high_aal_requires_red_team_artifacts() -> None:
    validation = RedTeamProtocol().validate(
        artifacts={},
        aal="AAL-0",
        red_team_required=False,
    )

    assert validation.required is True
    assert validation.passed is False
    assert "fmea.json" in validation.missing_artifacts


def test_unresolved_critical_findings_block_closure() -> None:
    artifacts = {
        "fmea.json": {"critical_open": 1},
        "fta_paths.json": {"critical_open": 0},
        "cwe_mapping.json": [
            {"severity": "P0", "resolved": False, "finding": "escape path"}
        ],
        "residual_risk.md": {"critical_open": 0},
    }

    validation = RedTeamProtocol().validate(
        artifacts=artifacts,
        aal="AAL-1",
        red_team_required=True,
    )

    assert validation.passed is False
    assert validation.unresolved_critical_findings


def test_low_aal_without_red_team_requirement_passes() -> None:
    validation = RedTeamProtocol().validate(
        artifacts={},
        aal="AAL-3",
        red_team_required=False,
    )

    assert validation.required is False
    assert validation.passed is True
