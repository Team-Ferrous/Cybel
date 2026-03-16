import json

import pytest

from saguaro.parsing.parser import SAGUAROParser


@pytest.mark.parametrize(
    ("file_name", "expected"),
    [
        ("src/main.zig", "zig"),
        ("src/module.nim", "nim"),
        ("rtl/core.sv", "hdl"),
        ("rtl/core.vhdl", "hdl"),
        ("numeric/solver.f90", "fortran"),
        ("legacy/math.pas", "pascal"),
        ("pkg/geometry.ads", "ada"),
        ("legacy/payroll.cbl", "cobol"),
        ("contracts/token.sol", "solidity"),
        ("web/index.html", "html"),
        ("web/site.less", "css"),
        ("config/app.ini", "config"),
        (".env", "config"),
        ("typing/stub.pyi", "python"),
        ("cython/module.pyx", "python"),
        ("cython/module.pxd", "python"),
        ("frontend/index.phtml", "php"),
        ("build/BUILD.bazel", "bazel"),
        ("build/rules.bzl", "bazel"),
        ("contracts/token.vy", "solidity"),
        ("mobile/App.m", "cpp"),
        ("mobile/App.mm", "cpp"),
        ("science/stats.r", "r"),
        ("science/solver.jl", "julia"),
        ("logic/engine.clj", "clojure"),
        ("runtime/worker.ex", "elixir"),
        ("runtime/worker.erl", "erlang"),
        ("lang/model.hs", "haskell"),
        ("lang/model.ml", "ocaml"),
        ("asm/startup.asm", "assembly"),
        ("science/solver.mat", "matlab"),
        ("lang/core.scala", "scala"),
        ("templates/page.jinja2", "jinja"),
        ("docs/paper.tex", "latex"),
        ("infra/main.tf", "hcl"),
        ("ui/Main.qml", "qml"),
        ("editor/init.el", "emacs_lisp"),
        ("wasm/module.wat", "wat"),
    ],
)
def test_detect_language_expanded_families(file_name, expected):
    parser = SAGUAROParser()
    assert parser._detect_language(file_name, "") == expected  # noqa: SLF001


def test_detect_language_uses_content_for_ambiguous_extensions():
    parser = SAGUAROParser()

    assert (
        parser._detect_language(
            "ios/ViewController.m",
            '#import "UIKit/UIKit.h"\n@interface ViewController : NSObject\n@end\n',
        )
        == "objective_c"
    )
    assert (
        parser._detect_language(
            "logic/rules.pl",
            ":- module(rules, [run/0]).\nrun() :- true.\n",
        )
        == "prolog"
    )


@pytest.mark.parametrize(
    ("file_name", "content", "expected_entity"),
    [
        (
            "src/main.zig",
            'const std = @import("std");\n'
            "pub fn add(a: i32, b: i32) i32 {\n"
            "    return a + b;\n"
            "}\n",
            ("function", "add"),
        ),
        (
            "src/module.nim",
            "import os\n"
            "proc greet*() =\n"
            '  echo "hi"\n',
            ("function", "greet"),
        ),
        (
            "rtl/core.sv",
            "module core(input logic clk);\n"
            "endmodule\n",
            ("module", "core"),
        ),
        (
            "numeric/solver.f90",
            "module solver\n"
            "contains\n"
            "subroutine step()\n"
            "end subroutine step\n"
            "end module solver\n",
            ("function", "step"),
        ),
        (
            "legacy/math.pas",
            "unit MathUtils;\n"
            "interface\n"
            "procedure Compute;\n"
            "implementation\n"
            "end.\n",
            ("module", "MathUtils"),
        ),
        (
            "pkg/geometry.ads",
            "package Geometry is\n"
            "procedure Area;\n"
            "end Geometry;\n",
            ("module", "Geometry"),
        ),
        (
            "legacy/payroll.cbl",
            "IDENTIFICATION DIVISION.\n"
            "PROGRAM-ID. PAYROLL.\n"
            "PROCEDURE DIVISION.\n",
            ("module", "PAYROLL"),
        ),
        (
            "contracts/vault.sol",
            "contract Vault {\n"
            "function deposit() public {}\n"
            "}\n",
            ("class", "Vault"),
        ),
        (
            "web/index.html",
            '<main id="app"><my-card></my-card></main>\n',
            ("element", "main"),
        ),
        (
            "web/site.css",
            ".card { color: red; }\n"
            "@keyframes fade { from { opacity: 0; } to { opacity: 1; } }\n",
            ("selector", ".card"),
        ),
        (
            "config/app.ini",
            "[database]\n"
            "host=localhost\n",
            ("section", "database"),
        ),
        (
            "science/stats.r",
            "score <- function(x) {\n"
            "  x + 1\n"
            "}\n",
            ("function", "score"),
        ),
        (
            "science/solver.jl",
            "module Solver\n"
            "function step(x)\n"
            "  x + 1\n"
            "end\n"
            "end\n",
            ("function", "step"),
        ),
        (
            "logic/engine.clj",
            "(ns app.engine)\n"
            "(defn run [] true)\n",
            ("function", "run"),
        ),
        (
            "runtime/worker.ex",
            "defmodule Worker do\n"
            "  def run(), do: :ok\n"
            "end\n",
            ("module", "Worker"),
        ),
        (
            "runtime/worker.erl",
            "-module(worker).\n"
            "run() -> ok.\n",
            ("module", "worker"),
        ),
        (
            "ios/ViewController.m",
            '#import "UIKit/UIKit.h"\n'
            "@interface ViewController : NSObject\n"
            "- (void)run;\n"
            "@end\n",
            ("class", "ViewController"),
        ),
        (
            "asm/startup.asm",
            "_start:\n"
            "  mov rax, 60\n",
            ("label", "_start"),
        ),
        (
            "science/solver.mat",
            "function y = step(x)\n"
            "y = x + 1;\n"
            "end\n",
            ("function", "step"),
        ),
        (
            "lang/core.scala",
            "object Worker {\n"
            "  def run(): Int = 1\n"
            "}\n",
            ("class", "Worker"),
        ),
        (
            "logic/rules.pl",
            ":- module(rules, [run/0]).\n"
            "run() :- true.\n",
            ("module", "rules"),
        ),
        (
            "templates/page.jinja2",
            "{% block content %}\n"
            "  Hello\n"
            "{% endblock %}\n",
            ("block", "content"),
        ),
        (
            "docs/paper.tex",
            "\\section{Overview}\n"
            "\\newcommand{\\Energy}{E}\n",
            ("section", "Overview"),
        ),
        (
            "infra/main.tf",
            'module "network" {\n'
            '  source = "./modules/network"\n'
            "}\n",
            ("block", "network"),
        ),
        (
            "ui/Main.qml",
            "Item {\n"
            "  function run() {}\n"
            "}\n",
            ("component", "Item"),
        ),
        (
            "editor/init.el",
            "(defun bootstrap () t)\n",
            ("function", "bootstrap"),
        ),
        (
            "wasm/module.wat",
            "(module\n"
            "  (func $run)\n"
            ")\n",
            ("function", "run"),
        ),
    ],
)
def test_parse_file_lightweight_path_for_new_languages(
    tmp_path, file_name, content, expected_entity
):
    file_path = tmp_path / file_name
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")

    parser = SAGUAROParser()
    entities = parser.parse_file(str(file_path))
    typed_names = {(entity.type, entity.name) for entity in entities}

    assert expected_entity in typed_names
    assert any(entity.type == "dependency_graph" for entity in entities)
    assert any(
        entity.type == "file" and entity.name == file_path.name for entity in entities
    )


def test_lightweight_dependency_payload_contains_internal_edges(tmp_path):
    file_path = tmp_path / "src" / "module.nim"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        "proc helper() =\n"
        "  discard\n"
        "proc run() =\n"
        "  helper()\n",
        encoding="utf-8",
    )

    parser = SAGUAROParser()
    entities = parser.parse_file(str(file_path))
    dependency_graph = next(
        entity for entity in entities if entity.type == "dependency_graph"
    )
    payload = json.loads(dependency_graph.content)

    assert payload["exports"]
    assert any(
        edge["from"] == "run" and edge["to"] == "helper" and edge["relation"] == "calls"
        for edge in payload["internal_edges"]
    )


def test_cpp_header_has_explicit_parser_backend_without_generic_fallback(tmp_path):
    file_path = tmp_path / "include" / "feature.hpp"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        '#include "shared.hpp"\n'
        "namespace demo {\n"
        "class Engine {};\n"
        "int compute(int value);\n"
        "}\n",
        encoding="utf-8",
    )

    parser = SAGUAROParser()
    entities = parser.parse_file(str(file_path))
    typed_names = {(entity.type, entity.name) for entity in entities}

    assert ("class", "Engine") in typed_names or ("class", "demo") in typed_names
    assert ("function", "compute") in typed_names
    dependency_graph = next(
        entity for entity in entities if entity.type == "dependency_graph"
    )
    payload = json.loads(dependency_graph.content)
    assert any("shared.hpp" in item for item in payload["imports"])


def test_unknown_language_returns_no_entities_without_implicit_fallback(tmp_path):
    file_path = tmp_path / "notes" / "sample.unknownlang"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("some unsupported syntax\n", encoding="utf-8")

    parser = SAGUAROParser()
    assert parser.parse_file(str(file_path)) == []


@pytest.mark.parametrize(
    "language",
    [
        "shell",
        "scala",
        "ruby",
        "css",
        "html",
        "json",
        "yaml",
        "toml",
        "elixir",
        "erlang",
        "haskell",
        "objective_c",
        "csharp",
    ],
)
def test_runtime_ast_support_surface_expands_beyond_core_languages(language):
    parser = SAGUAROParser()
    assert parser.declares_ast_language(language) is True
    assert parser.supports_ast_language(language) is True


@pytest.mark.parametrize(
    ("file_name", "content", "expected_entity"),
    [
        ("script.sh", "run() {\n  echo hi\n}\n", ("function", "run")),
        ("lib/worker.scala", "object Worker {\n  def run(): Int = 1\n}\n", ("class", "Worker")),
        ("lib/worker.rb", "class Worker\n  def run\n    1\n  end\nend\n", ("class", "Worker")),
        ("styles/site.css", ".card { color: red; }\n", ("selector", ".card")),
        ("web/index.html", "<main id=\"app\"><my-card></my-card></main>\n", ("element", "main")),
        ("config/data.json", "{\"name\": 1, \"inner\": {\"k\": 2}}\n", ("key", "name")),
        ("config/app.toml", "[database]\nhost = \"localhost\"\n", ("section", "database")),
        ("lib/worker.ex", "defmodule Worker do\n  def run(), do: :ok\nend\n", ("module", "Worker")),
        ("lib/worker.erl", "-module(worker).\nrun() -> ok.\n", ("function", "run")),
        ("lang/worker.hs", "module Worker where\nrun :: Int\nrun = 1\n", ("function", "run")),
        ("ios/ViewController.m", '#import "UIKit/UIKit.h"\n@interface ViewController : NSObject\n- (void)run;\n@end\n', ("class", "ViewController")),
    ],
)
def test_expanded_tree_sitter_backends_produce_real_entities(
    tmp_path, file_name, content, expected_entity
):
    file_path = tmp_path / file_name
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")

    parser = SAGUAROParser()
    entities = parser.parse_file(str(file_path))
    typed_names = {(entity.type, entity.name) for entity in entities}

    assert expected_entity in typed_names


def test_declared_ast_language_fails_closed_without_backend_fallback(monkeypatch, tmp_path):
    file_path = tmp_path / "lib" / "worker.scala"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        "object Worker {\n  def run(): Int = 1\n}\n",
        encoding="utf-8",
    )

    parser = SAGUAROParser()
    monkeypatch.setattr(parser, "get_parser", None)
    monkeypatch.setattr(parser, "get_language", None)

    assert parser.parse_file(str(file_path)) == []


def test_tree_sitter_capture_matcher_linear_cases():
    matches = SAGUAROParser._match_tree_sitter_capture_names_python(
        def_starts=[0, 5, 30, 40, 60],
        def_ends=[20, 15, 50, 55, 80],
        def_type_ids=[1, 1, 2, 1, 1],
        name_starts=[2, 8, 32, 42, 65],
        name_ends=[4, 10, 35, 44, 68],
        name_type_ids=[1, 1, 2, 1, 1],
    )

    assert matches == [0, 1, 2, 3, 4]


def test_tree_sitter_capture_matcher_prefers_innermost_nested_def():
    matches = SAGUAROParser._match_tree_sitter_capture_names_python(
        def_starts=[0, 10, 40],
        def_ends=[100, 30, 80],
        def_type_ids=[1, 1, 2],
        name_starts=[12, 50, 90],
        name_ends=[18, 55, 95],
        name_type_ids=[1, 2, 1],
    )

    assert matches == [2, 0, 1]


def test_tree_sitter_capture_matcher_leaves_unmatched_defs():
    matches = SAGUAROParser._match_tree_sitter_capture_names_python(
        def_starts=[0, 20],
        def_ends=[10, 30],
        def_type_ids=[1, 2],
        name_starts=[1],
        name_ends=[3],
        name_type_ids=[1],
    )

    assert matches == [0, -1]
