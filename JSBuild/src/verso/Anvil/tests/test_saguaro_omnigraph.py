from saguaro.omnigraph.store import OmniGraphStore


class _GraphService:
    def load_graph(self):
        return {
            "generated_at": 1700000000,
            "nodes": {
                "node-1": {
                    "id": "node-1",
                    "name": "render_template",
                    "qualified_name": "ui.render_template",
                    "kind": "function",
                    "file": "ui/render.py",
                    "metadata": {"language": "python"},
                },
                "node-2": {
                    "id": "node-2",
                    "name": "page",
                    "qualified_name": "templates/page.jinja2",
                    "kind": "file",
                    "file": "templates/page.jinja2",
                    "metadata": {"language": "jinja"},
                },
                "node-3": {
                    "id": "node-3",
                    "name": "page",
                    "qualified_name": "frontend/page.tsx",
                    "kind": "file",
                    "file": "frontend/page.tsx",
                    "metadata": {"language": "typescript"},
                },
            },
        }


def test_omnigraph_builds_requirement_and_bridge_relations(tmp_path):
    payload = {
        "generation_id": "trace-1",
        "requirements": [
            {
                "id": "REQ-1",
                "file": "README.md",
                "text_norm": "support template bridges",
                "modality": "interface",
                "strength": "MUST",
                "heading_path": ["Top"],
            }
        ],
        "records": [
            {
                "id": "REL-1",
                "requirement_id": "REQ-1",
                "artifact_type": "symbol",
                "artifact_id": "node-1",
                "relation_type": "implements",
                "evidence_types": ["lexical"],
                "confidence": 0.8,
                "verification_state": "verified",
                "notes": [],
            }
        ],
    }

    graph = OmniGraphStore(str(tmp_path), graph_service=_GraphService()).build(
        traceability_payload=payload
    )

    assert graph["summary"]["requirement_count"] == 1
    assert any(
        item["relation_type"] == "bridged_by"
        for item in graph["relations"].values()
    )


def test_omnigraph_discovers_filesystem_template_and_ffi_bridges(tmp_path):
    templates = tmp_path / "templates"
    static = tmp_path / "static"
    src = tmp_path / "src"
    templates.mkdir()
    static.mkdir()
    src.mkdir()

    (templates / "page.jinja2").write_text(
        '<div id="score-root">{{ score }}</div>\n<script src="/static/page.js"></script>\n',
        encoding="utf-8",
    )
    (static / "page.js").write_text(
        'export function hydrateScore() { return document.getElementById("score-root"); }\n',
        encoding="utf-8",
    )
    (src / "bridge.py").write_text(
        'import ctypes\nlib = ctypes.CDLL("libscore.so")\n',
        encoding="utf-8",
    )
    (src / "score.cpp").write_text(
        'extern "C" int native_score(int value) { return value * 2; }\n',
        encoding="utf-8",
    )

    graph = OmniGraphStore(str(tmp_path)).build(traceability_payload={"requirements": [], "records": []})

    assert "file::templates/page.jinja2" in graph["nodes"]
    assert "file::static/page.js" in graph["nodes"]
    assert any(
        item["relation_type"] == "bridged_by"
        and {
            item["src_id"],
            item["dst_id"],
        }
        == {"file::templates/page.jinja2", "file::static/page.js"}
        for item in graph["relations"].values()
    )
    assert any(
        item["relation_type"] == "bridged_by"
        and "ffi_bridge" in item["id"]
        for item in graph["relations"].values()
    )
