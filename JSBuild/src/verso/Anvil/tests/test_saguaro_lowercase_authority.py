from pathlib import Path


def test_setup_entrypoint_targets_lowercase_saguaro_package() -> None:
    setup_text = Path("setup.py").read_text(encoding="utf-8")

    assert "saguaro=saguaro.cli:main" in setup_text
    assert "Saguaro.cli:main" not in setup_text


def test_compiled_obligations_exclude_vendored_saguaro_tree() -> None:
    obligations = Path("standards/AES_OBLIGATIONS.json").read_text(encoding="utf-8")
    assert '"excluded_reference_roots"' in obligations
    assert "Saguaro/" in obligations
