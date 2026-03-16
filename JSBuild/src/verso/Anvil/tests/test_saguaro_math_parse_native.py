from __future__ import annotations

from saguaro.math import MathEngine


def test_math_parse_prefers_native_cpp_label_for_cpp_ast_paths(
    tmp_path, monkeypatch
) -> None:
    target = tmp_path / "kernel.cc"
    target.write_text("float y = a * b + c;\n", encoding="utf-8")

    engine = MathEngine(str(tmp_path))
    monkeypatch.setattr(engine.parser, "supports_ast_language", lambda lang: lang == "cpp")

    payload = engine.parse(path="kernel.cc")

    assert payload["status"] == "ok"
    assert payload["analysis_engine"] == "native_cpp"
