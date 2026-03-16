from core.memory.fabric import resolve_memory_backend_profile


def test_enterprise_profile_falls_back_to_sqlite_without_postgres_driver(tmp_path):
    profile = resolve_memory_backend_profile(
        db_path=str(tmp_path / "almf.db"),
        requested_backend="postgres",
        tenant_key="campaign-1",
    )

    assert profile.requested_backend == "postgres"
    assert profile.effective_backend == "sqlite"
    assert profile.driver_name == "sqlite3"
    assert profile.is_fallback is True
    assert "postgres requested" in profile.fallback_reason
