from __future__ import annotations

from pathlib import Path
import tarfile
from typing import Any

from audit.control_plane.mission import MissionContext
from audit.control_plane.nodes import ArtifactNode
from audit.control_plane.reducers import file_digest, now_iso, question_for_artifact

CAPSULE_MANIFEST_SCHEMA_VERSION = "native_qsg_suite.capsule_manifest.v1"


def build_capsule_manifest(
    mission: MissionContext,
    *,
    artifact_paths: list[Path],
) -> dict[str, Any]:
    artifacts: list[ArtifactNode] = []
    category_counts = {
        "intended": 0,
        "happened": 0,
        "observed": 0,
        "concluded": 0,
        "unresolved": 0,
    }
    for path in artifact_paths:
        relative = path.relative_to(mission.layout.root).as_posix()
        question = question_for_artifact(relative)
        category_counts[question] = category_counts.get(question, 0) + 1
        digest = file_digest(path, root=mission.layout.root)
        artifacts.append(
            ArtifactNode(
                artifact_id=relative,
                question=question,
                path=relative,
                summary=relative,
                required=True,
                exists=bool(digest.get("exists", False)),
                metadata={
                    "bytes": int(digest.get("bytes", 0) or 0),
                    "sha256": str(digest.get("sha256") or ""),
                },
            )
        )
    return {
        "schema_version": CAPSULE_MANIFEST_SCHEMA_VERSION,
        "run_id": mission.run_id,
        "generated_at": now_iso(),
        "categories": category_counts,
        "artifacts": [artifact.to_dict() for artifact in artifacts],
    }


def build_capsule_archive(
    mission: MissionContext,
    *,
    output_path: Path,
    artifact_paths: list[Path],
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output_path, "w:gz") as archive:
        for path in artifact_paths:
            if not path.exists():
                continue
            archive.add(path, arcname=path.relative_to(mission.layout.root))
    return {
        "path": output_path.relative_to(mission.layout.root).as_posix(),
        "artifact_count": sum(1 for path in artifact_paths if path.exists()),
    }
