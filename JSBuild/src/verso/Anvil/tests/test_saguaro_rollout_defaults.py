from __future__ import annotations

import json
from pathlib import Path

from saguaro.api import SaguaroAPI


def test_health_surfaces_default_query_gateway_and_storage_layout(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "demo.py").write_text("def demo():\n    return 1\n", encoding="utf-8")

    api = SaguaroAPI(repo_path=str(tmp_path))
    api.init(force=True)

    vectors_dir = tmp_path / ".saguaro" / "vectors"
    (vectors_dir / "vectors.bin").write_bytes(b"\x00" * (4 * 4))
    (vectors_dir / "norms.bin").write_bytes(b"\x00" * 4)
    (vectors_dir / "metadata.json").write_text("[]", encoding="utf-8")
    (vectors_dir / "index_meta.json").write_text(
        json.dumps(
            {
                "dim": 8,
                "active_dim": 4,
                "total_dim": 8,
                "storage_dim": 4,
                "count": 0,
                "capacity": 1,
                "version": 4,
                "format": "native_mmap",
                "vector_layout": "active_only",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / ".saguaro" / "index_schema.json").write_text(
        json.dumps({"embedding_schema_version": 3, "repo_path": str(tmp_path)}),
        encoding="utf-8",
    )

    report = api.health()

    assert report["query_gateway"]["status"] == "enabled"
    assert report["query_gateway"]["query_many_available"] is True
    assert report["storage_layout"]["status"] == "ready"
    assert report["storage_layout"]["vector_layout"] == "active_only"
    assert report["storage_layout"]["norms_present"] is True
    assert "runtime_profile" in report
