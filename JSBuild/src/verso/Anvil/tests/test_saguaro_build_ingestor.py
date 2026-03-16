from __future__ import annotations

import json
from pathlib import Path

from saguaro.build_system.ingestor import BuildGraphIngestor


def test_build_ingestor_uses_compile_database_for_target_sources(
    tmp_path: Path,
) -> None:
    source = tmp_path / "src" / "native_core.cc"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("int native_core() { return 1; }\n", encoding="utf-8")

    build_dir = tmp_path / "build"
    build_dir.mkdir()
    compile_commands = [
        {
            "directory": str(tmp_path),
            "command": (
                "/usr/bin/c++ -Iinclude -isystem third_party/include "
                "-c src/native_core.cc "
                "-o build/CMakeFiles/native_core.dir/src/native_core.cc.o"
            ),
            "file": "src/native_core.cc",
            "output": "build/CMakeFiles/native_core.dir/src/native_core.cc.o",
        }
    ]
    (build_dir / "compile_commands.json").write_text(
        json.dumps(compile_commands),
        encoding="utf-8",
    )

    report = BuildGraphIngestor(str(tmp_path)).ingest()

    target = report["targets"]["cmake:native_core"]
    assert report["structured_inputs"]["compile_databases"] == 1
    assert target["language"] == "native"
    assert "src/native_core.cc" in target["sources"]
    assert "include" in target["includes"]
    assert "third_party/include" in target["includes"]
    assert (
        "build/CMakeFiles/native_core.dir/src/native_core.cc.o"
        in target["artifacts"]
    )


def test_build_ingestor_parses_cmake_file_api_targets(tmp_path: Path) -> None:
    reply_dir = tmp_path / "build" / ".cmake" / "api" / "v1" / "reply"
    reply_dir.mkdir(parents=True, exist_ok=True)

    target_json = {
        "name": "anvil_runtime_core",
        "type": "SHARED_LIBRARY",
        "sources": [
            {"path": str(tmp_path / "core" / "native" / "engine.cc")},
            {"path": str(tmp_path / "core" / "native" / "engine.h")},
        ],
        "compileGroups": [
            {
                "includes": [
                    {"path": str(tmp_path / "core" / "native" / "include")},
                ]
            }
        ],
        "dependencies": [{"id": "dep-1"}],
        "artifacts": [{"path": str(tmp_path / "build" / "libanvil_runtime_core.so")}],
    }
    (reply_dir / "target-anvil_runtime_core.json").write_text(
        json.dumps(target_json),
        encoding="utf-8",
    )
    codemodel = {
        "configurations": [
            {
                "targets": [
                    {
                        "id": "tgt-1",
                        "name": "anvil_runtime_core",
                        "jsonFile": "target-anvil_runtime_core.json",
                    },
                    {"id": "dep-1", "name": "anvil_support"},
                ]
            }
        ]
    }
    (reply_dir / "codemodel-v2.json").write_text(
        json.dumps(codemodel),
        encoding="utf-8",
    )
    index = {
        "objects": [
            {
                "kind": "codemodel",
                "jsonFile": "codemodel-v2.json",
            }
        ]
    }
    (reply_dir / "index-1234.json").write_text(json.dumps(index), encoding="utf-8")

    report = BuildGraphIngestor(str(tmp_path)).ingest()

    target = report["targets"]["cmake:anvil_runtime_core"]
    assert report["structured_inputs"]["cmake_file_api_replies"] == 1
    assert target["type"] == "lib"
    assert target["language"] == "native"
    assert "core/native/engine.cc" in target["sources"]
    assert "core/native/engine.h" in target["sources"]
    assert "core/native/include" in target["includes"]
    assert target["deps"] == ["anvil_support"]
    assert "build/libanvil_runtime_core.so" in target["artifacts"]


def test_build_ingestor_skips_null_cmake_file_api_target_payload(tmp_path: Path) -> None:
    reply_dir = tmp_path / "build" / ".cmake" / "api" / "v1" / "reply"
    reply_dir.mkdir(parents=True, exist_ok=True)

    (reply_dir / "target-null.json").write_text("null\n", encoding="utf-8")
    codemodel = {
        "configurations": [
            {
                "targets": [
                    {
                        "id": "tgt-null",
                        "name": "broken_target",
                        "jsonFile": "target-null.json",
                    }
                ]
            }
        ]
    }
    (reply_dir / "codemodel-v2.json").write_text(
        json.dumps(codemodel),
        encoding="utf-8",
    )
    index = {"objects": [{"kind": "codemodel", "jsonFile": "codemodel-v2.json"}]}
    (reply_dir / "index-1234.json").write_text(json.dumps(index), encoding="utf-8")

    report = BuildGraphIngestor(str(tmp_path)).ingest()

    assert report["structured_inputs"]["cmake_file_api_replies"] == 1
    assert "cmake:broken_target" not in report["targets"]
