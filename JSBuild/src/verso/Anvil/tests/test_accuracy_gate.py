from core.evidence_builder import EvidenceBuilder
from core.hallucination_gate import HallucinationGate


class _FakeAPI:
    def __init__(self, results_map, skeleton_map, repo_root=None):
        self._results_map = results_map
        self._skeleton_map = skeleton_map
        self._repo_root = repo_root

    def query(self, query, k=10):
        return {"results": self._results_map.get(query, [])}

    def skeleton(self, file_path):
        return self._skeleton_map.get(
            file_path,
            {"file_path": file_path, "language": "python", "symbols": []},
        )

    def list_directory(self, path, recursive=False, extensions=None):
        root = self._repo_root
        if root is None:
            return {"entries": []}

        target = root / path if path not in {"", "."} else root
        if recursive:
            entries = []
            for item in sorted(target.rglob("*")):
                if item.is_file() and extensions and item.suffix not in extensions:
                    continue
                entries.append(
                    {
                        "path": item.relative_to(root).as_posix(),
                        "type": "directory" if item.is_dir() else "file",
                    }
                )
            return {"entries": entries}

        entries = []
        for item in sorted(target.iterdir()):
            if item.is_file() and extensions and item.suffix not in extensions:
                continue
            entries.append(
                {
                    "path": item.relative_to(root).as_posix(),
                    "type": "directory" if item.is_dir() else "file",
                }
            )
        return {"entries": entries}


class _FakeSubstrate:
    def __init__(self, api):
        self._api = api


class _FakeSaguaroTools:
    def __init__(self, api):
        self.substrate = _FakeSubstrate(api)


class _FakeRegistry:
    def __init__(self, file_map):
        self._file_map = file_map

    def dispatch(self, tool_name, args):
        if tool_name != "read_file":
            return f"Error: unsupported tool {tool_name}"
        path = args.get("path")
        return self._file_map.get(path, f"Error: file {path} not found")


def test_hallucination_gate_catches_invented_class():
    gate = HallucinationGate()
    response = "The main class is `UnifiedChat`."
    evidence = {
        "file_contents": {
            "core/unified_chat_loop.py": "class UnifiedChatLoop:\n    pass\n"
        },
        "skeletons": {},
        "entities": {},
        "tree_views": {},
    }

    _, violations = gate.validate(response, evidence)
    assert any("UnifiedChat" in violation for violation in violations)


def test_evidence_builder_loads_target_file_and_dependency_graph(tmp_path):
    repo_root = tmp_path

    (repo_root / "core").mkdir(parents=True, exist_ok=True)
    (repo_root / "utils").mkdir(parents=True, exist_ok=True)
    (repo_root / "core" / "main.py").write_text(
        "from utils.helper import helper\n\nclass Main:\n    def run(self):\n        return helper()\n",
        encoding="utf-8",
    )
    (repo_root / "utils" / "helper.py").write_text(
        "def helper():\n    return 1\n",
        encoding="utf-8",
    )

    results_map = {
        "Explain core/main.py": [
            {
                "name": "Main",
                "file": "core/main.py",
                "line": 3,
                "type": "class",
                "score": 0.9,
            },
            {
                "name": "helper",
                "file": "utils/helper.py",
                "line": 1,
                "type": "function",
                "score": 0.8,
            },
        ]
    }
    skeleton_map = {
        "core/main.py": {
            "file_path": "core/main.py",
            "language": "python",
            "symbols": [{"type": "class", "name": "Main", "line_start": 3}],
        },
        "utils/helper.py": {
            "file_path": "utils/helper.py",
            "language": "python",
            "symbols": [{"type": "function", "name": "helper", "line_start": 1}],
        },
    }

    builder = EvidenceBuilder(
        saguaro_tools=_FakeSaguaroTools(
            _FakeAPI(results_map, skeleton_map, repo_root=repo_root)
        ),
        registry=_FakeRegistry(
            {
                "core/main.py": (repo_root / "core" / "main.py").read_text(
                    encoding="utf-8"
                ),
                "utils/helper.py": (repo_root / "utils" / "helper.py").read_text(
                    encoding="utf-8"
                ),
            }
        ),
        console=None,
        repo_root=str(repo_root),
    )

    evidence = builder.build("Explain core/main.py", target_file="core/main.py")

    assert evidence["primary_file"] == "core/main.py"
    assert "core/main.py" in evidence["file_contents"]
    assert "utils/helper.py" in evidence["file_contents"]
    assert "core/main.py" in evidence["imports"]
    assert "utils.helper" in evidence["imports"]["core/main.py"]
    assert evidence["dependency_graph"]["edges"].get("core/main.py")
