from __future__ import annotations

import pytest

from saguaro.math import MathEngine
from saguaro.math.languages import supported_languages
from saguaro.parsing.parser import SAGUAROParser


def test_math_engine_covers_parser_language_families(tmp_path) -> None:
    files = {
        "science/solver.jl": "energy = mass * c * c\n",
        "science/stats.r": "score <- total / count\n",
        "numeric/solver.f90": "energy = mass * c * c\n",
        "science/solver.mat": "y = a .* b + c;\n",
        "lang/core.scala": "val energy = mass * c * c\n",
        "contracts/token.sol": "balance = balance + amount;\n",
    }
    for rel_path, content in files.items():
        target = tmp_path / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    parsed = MathEngine(str(tmp_path)).parse(".")
    languages = parsed["summary"]["by_language"]
    expressions = {item["expression"] for item in parsed["records"]}

    assert languages["julia"] >= 1
    assert languages["r"] >= 1
    assert languages["fortran"] >= 1
    assert languages["matlab"] >= 1
    assert languages["scala"] >= 1
    assert languages["solidity"] >= 1
    assert "energy = mass * c * c" in expressions
    assert "score <- total / count" in expressions


def test_math_engine_supported_languages_cover_parser_language_families() -> None:
    parser_languages = set(SAGUAROParser._LIGHTWEIGHT_LANGUAGES) | {  # noqa: SLF001
        "javascript",
        "typescript",
    }

    assert parser_languages <= supported_languages()


@pytest.mark.parametrize(
    ("file_name", "content", "expected_language"),
    [
        ("src/main.zig", "// energy = mass * c * c\n", "zig"),
        ("src/module.nim", "# energy = mass * c * c\n", "nim"),
        ("rtl/core.sv", "-- energy = mass * c * c\n", "hdl"),
        ("legacy/math.pas", "(* energy = mass * c * c *)\n", "pascal"),
        ("pkg/geometry.ads", "-- energy = mass * c * c\n", "ada"),
        ("legacy/payroll.cbl", "*> energy = mass * c * c\n", "cobol"),
        ("web/index.html", "<!-- energy = mass * c * c -->\n", "html"),
        ("web/site.less", "/* ratio = width / height */\n", "css"),
        ("config/app.ini", "; ratio = width / height\n", "config"),
        ("frontend/index.phtml", "// energy = mass * c * c\n", "php"),
        ("bazel/rules.bzl", "# ratio = width / height\n", "bazel"),
        ("logic/engine.clj", ";; energy = mass * c * c\n", "clojure"),
        ("runtime/worker.ex", "# energy = mass * c * c\n", "elixir"),
        ("runtime/worker.erl", "% energy = mass * c * c\n", "erlang"),
        ("lang/model.hs", "-- energy = mass * c * c\n", "haskell"),
        ("lang/model.ml", "(* energy = mass *. c *. c *)\n", "ocaml"),
        ("asm/startup.asm", "; cycles = bytes / lanes\n", "assembly"),
        ("templates/page.jinja2", "{# ratio = width / height #}\n", "jinja"),
        ("docs/paper.tex", "% E = m * c^2\n", "latex"),
        ("infra/main.tf", "# ratio = width / height\n", "hcl"),
        ("ui/Main.qml", "// ratio = width / height\n", "qml"),
        ("editor/init.el", ";; ratio = width / height\n", "emacs_lisp"),
        ("wasm/module.wat", ";; ratio = width / height\n", "wat"),
    ],
)
def test_math_engine_extracts_equations_from_comment_surfaces_for_all_language_families(
    tmp_path,
    file_name,
    content,
    expected_language,
) -> None:
    target = tmp_path / file_name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")

    parsed = MathEngine(str(tmp_path)).parse(".")
    languages = parsed["summary"]["by_language"]

    assert languages[expected_language] >= 1
    assert parsed["count"] >= 1
