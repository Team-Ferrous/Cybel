from saguaro.validation.engine import ValidationEngine


class _GraphService:
    def load_graph(self):
        return {
            "generated_at": 1700000000,
            "nodes": {
                "n1": {
                    "id": "n1",
                    "name": "validate_docs",
                    "qualified_name": "saguaro.docs.validate_docs",
                    "kind": "function",
                    "file": "saguaro/docs.py",
                    "metadata": {"language": "python"},
                },
                "n2": {
                    "id": "n2",
                    "name": "test_validate_docs",
                    "qualified_name": "tests.test_validate_docs",
                    "kind": "test",
                    "file": "tests/test_validate_docs.py",
                    "metadata": {"language": "python"},
                },
            },
        }


def test_validation_engine_reports_requirement_states(tmp_path):
    (tmp_path / "README.md").write_text(
        "# Spec\n\n"
        "- The system MUST validate docs.\n",
        encoding="utf-8",
    )

    report = ValidationEngine(str(tmp_path), graph_service=_GraphService()).validate_docs(".")

    assert report["status"] == "ok"
    assert report["summary"]["count"] == 1
    assert report["requirements"][0]["state"] in {
        "implemented_witnessed",
        "implemented_unwitnessed",
        "partially_implemented",
    }
