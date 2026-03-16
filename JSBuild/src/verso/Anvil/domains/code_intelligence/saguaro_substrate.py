import ast
import json
import os
import hashlib
import logging
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from config.settings import CHRONICLE_DIR
from saguaro.api import SaguaroAPI
from saguaro.governor import ContextGovernor
from saguaro.workset import WorksetManager

logger = logging.getLogger(__name__)


class SaguaroSubstrate:
    """
    In-process substrate between Anvil and Saguaro.

    All operations route through `saguaro.api.SaguaroAPI`.
    """

    def __init__(self, root_dir: str = "."):
        self.root_dir = os.path.abspath(root_dir)
        self._api = SaguaroAPI(self.root_dir)

        saguaro_dir = os.path.join(self.root_dir, ".saguaro")
        self.governor = ContextGovernor()
        self.workset_manager = WorksetManager(saguaro_dir, repo_path=self.root_dir)
        self.active_mission_id = None

        self._hd_state_blob: bytes = b""
        self._daemon_watcher = None
        self._daemon_thread: threading.Thread | None = None
        self._daemon_interval = 5

    def _is_index_fresh(self) -> bool:
        tracking = os.path.join(self.root_dir, ".saguaro", "tracking.json")
        if not os.path.exists(tracking):
            return False
        try:
            with open(tracking, "r", encoding="utf-8") as f:
                state = json.load(f)
            return bool(state)
        except Exception:
            return False

    def execute_command(self, command_str: str) -> str:
        """Parse CLI-like substrate commands into API calls."""
        cmd_parts = command_str.strip().split()
        if not cmd_parts:
            return "Error: Empty command."

        action = cmd_parts[0]

        try:
            if action == "skeleton":
                return self.agent_skeleton(cmd_parts[1])

            if action == "slice":
                target = cmd_parts[1]
                depth = 1
                corpus_id = None
                if "--depth" in cmd_parts:
                    i = cmd_parts.index("--depth")
                    if i + 1 < len(cmd_parts):
                        depth = int(cmd_parts[i + 1])
                if "--corpus" in cmd_parts:
                    i = cmd_parts.index("--corpus")
                    if i + 1 < len(cmd_parts):
                        corpus_id = cmd_parts[i + 1]
                # Backward-compatible format: slice path/to/file.py.SymbolName
                if "." in target:
                    maybe_file, maybe_entity = target.rsplit(".", 1)
                    candidate = os.path.join(self.root_dir, maybe_file)
                    if os.path.exists(candidate):
                        return self.agent_slice(
                            maybe_file,
                            maybe_entity,
                            depth=depth,
                            corpus_id=corpus_id,
                        )
                return self.agent_slice(target, depth=depth, corpus_id=corpus_id)

            if action == "impact":
                return self.agent_impact(cmd_parts[1])

            if action == "query":
                k = 5
                scope = "global"
                dedupe_by = "entity"
                recall = None
                breadth = None
                score_threshold = None
                stale_file_bias = None
                cost_budget = None
                corpus_ids = None
                consumed_indexes = {0}
                if "--k" in cmd_parts:
                    idx = cmd_parts.index("--k")
                    if idx + 1 < len(cmd_parts):
                        k = int(cmd_parts[idx + 1])
                        consumed_indexes.update({idx, idx + 1})
                if "--scope" in cmd_parts:
                    idx = cmd_parts.index("--scope")
                    if idx + 1 < len(cmd_parts):
                        scope = str(cmd_parts[idx + 1]).strip().lower()
                        consumed_indexes.update({idx, idx + 1})
                if "--dedupe-by" in cmd_parts:
                    idx = cmd_parts.index("--dedupe-by")
                    if idx + 1 < len(cmd_parts):
                        dedupe_by = str(cmd_parts[idx + 1]).strip().lower()
                        consumed_indexes.update({idx, idx + 1})
                if "--recall" in cmd_parts:
                    idx = cmd_parts.index("--recall")
                    if idx + 1 < len(cmd_parts):
                        recall = str(cmd_parts[idx + 1]).strip().lower()
                        consumed_indexes.update({idx, idx + 1})
                if "--breadth" in cmd_parts:
                    idx = cmd_parts.index("--breadth")
                    if idx + 1 < len(cmd_parts):
                        breadth = int(cmd_parts[idx + 1])
                        consumed_indexes.update({idx, idx + 1})
                if "--score-threshold" in cmd_parts:
                    idx = cmd_parts.index("--score-threshold")
                    if idx + 1 < len(cmd_parts):
                        score_threshold = float(cmd_parts[idx + 1])
                        consumed_indexes.update({idx, idx + 1})
                if "--stale-file-bias" in cmd_parts:
                    idx = cmd_parts.index("--stale-file-bias")
                    if idx + 1 < len(cmd_parts):
                        stale_file_bias = float(cmd_parts[idx + 1])
                        consumed_indexes.update({idx, idx + 1})
                if "--cost-budget" in cmd_parts:
                    idx = cmd_parts.index("--cost-budget")
                    if idx + 1 < len(cmd_parts):
                        cost_budget = str(cmd_parts[idx + 1]).strip().lower()
                        consumed_indexes.update({idx, idx + 1})
                if "--corpus" in cmd_parts:
                    idx = cmd_parts.index("--corpus")
                    if idx + 1 < len(cmd_parts):
                        corpus_ids = [
                            item.strip()
                            for item in str(cmd_parts[idx + 1]).split(",")
                            if item.strip()
                        ]
                        consumed_indexes.update({idx, idx + 1})
                query_text = " ".join(
                    part
                    for idx, part in enumerate(cmd_parts)
                    if idx not in consumed_indexes and not part.startswith("-")
                ).strip()
                return self.agent_query(
                    query_text,
                    k=k,
                    scope=scope,
                    dedupe_by=dedupe_by,
                    recall=recall,
                    breadth=breadth,
                    score_threshold=score_threshold,
                    stale_file_bias=stale_file_bias,
                    cost_budget=cost_budget,
                    corpus_ids=corpus_ids,
                )

            if action == "report":
                return self.agent_report()

            if action == "corpus":
                op = cmd_parts[1] if len(cmd_parts) > 1 else "list"
                path = None
                corpus_id = None
                alias = None
                ttl_hours = 24.0
                trust_level = "medium"
                build_profile = "auto"
                include_expired = "--include-expired" in cmd_parts
                rebuild = "--rebuild" in cmd_parts
                quarantine = "--no-quarantine" not in cmd_parts
                if len(cmd_parts) > 2 and not cmd_parts[2].startswith("-"):
                    if op == "create":
                        path = cmd_parts[2]
                    elif op == "show":
                        corpus_id = cmd_parts[2]
                if "--corpus-id" in cmd_parts:
                    idx = cmd_parts.index("--corpus-id")
                    if idx + 1 < len(cmd_parts):
                        corpus_id = cmd_parts[idx + 1]
                if "--alias" in cmd_parts:
                    idx = cmd_parts.index("--alias")
                    if idx + 1 < len(cmd_parts):
                        alias = cmd_parts[idx + 1]
                if "--ttl-hours" in cmd_parts:
                    idx = cmd_parts.index("--ttl-hours")
                    if idx + 1 < len(cmd_parts):
                        ttl_hours = float(cmd_parts[idx + 1])
                if "--trust-level" in cmd_parts:
                    idx = cmd_parts.index("--trust-level")
                    if idx + 1 < len(cmd_parts):
                        trust_level = cmd_parts[idx + 1]
                if "--build-profile" in cmd_parts:
                    idx = cmd_parts.index("--build-profile")
                    if idx + 1 < len(cmd_parts):
                        build_profile = cmd_parts[idx + 1]
                return self.corpus(
                    action=op,
                    path=path,
                    corpus_id=corpus_id,
                    alias=alias,
                    ttl_hours=ttl_hours,
                    quarantine=quarantine,
                    trust_level=trust_level,
                    build_profile=build_profile,
                    include_expired=include_expired,
                    rebuild=rebuild,
                )

            if action == "compare":
                target = "."
                candidates: list[str] = []
                corpus_ids: list[str] = []
                fleet_root = None
                top_k = 10
                ttl_hours = 72.0
                if "--target" in cmd_parts:
                    idx = cmd_parts.index("--target")
                    if idx + 1 < len(cmd_parts):
                        target = cmd_parts[idx + 1]
                for idx, part in enumerate(cmd_parts):
                    if part == "--candidate" and idx + 1 < len(cmd_parts):
                        candidates.append(cmd_parts[idx + 1])
                    if part == "--corpus" and idx + 1 < len(cmd_parts):
                        corpus_ids.append(cmd_parts[idx + 1])
                if "--fleet-root" in cmd_parts:
                    idx = cmd_parts.index("--fleet-root")
                    if idx + 1 < len(cmd_parts):
                        fleet_root = cmd_parts[idx + 1]
                if "--top-k" in cmd_parts:
                    idx = cmd_parts.index("--top-k")
                    if idx + 1 < len(cmd_parts):
                        top_k = int(cmd_parts[idx + 1])
                if "--ttl-hours" in cmd_parts:
                    idx = cmd_parts.index("--ttl-hours")
                    if idx + 1 < len(cmd_parts):
                        ttl_hours = float(cmd_parts[idx + 1])
                return self.compare(
                    target=target,
                    candidates=candidates,
                    corpus_ids=corpus_ids,
                    fleet_root=fleet_root,
                    top_k=top_k,
                    ttl_hours=ttl_hours,
                )

            if action == "reality":
                op = cmd_parts[1] if len(cmd_parts) > 1 else "twin"
                run_id = None
                limit = 500
                if "--run-id" in cmd_parts:
                    idx = cmd_parts.index("--run-id")
                    if idx + 1 < len(cmd_parts):
                        run_id = str(cmd_parts[idx + 1]).strip() or None
                if "--limit" in cmd_parts:
                    idx = cmd_parts.index("--limit")
                    if idx + 1 < len(cmd_parts):
                        limit = int(cmd_parts[idx + 1])
                return self.reality(action=op, run_id=run_id, limit=limit)

            if action == "verify":
                path = "."
                engines = None
                fix = "--fix" in cmd_parts
                if len(cmd_parts) > 1 and not cmd_parts[1].startswith("-"):
                    path = cmd_parts[1]
                if "--engines" in cmd_parts:
                    idx = cmd_parts.index("--engines")
                    if idx + 1 < len(cmd_parts):
                        engines = cmd_parts[idx + 1]
                return self.verify(path=path, engines=engines, fix=fix)

            if action == "cpu":
                cpu_op = cmd_parts[1] if len(cmd_parts) > 1 else "scan"
                if cpu_op != "scan":
                    return "Error: Unsupported cpu subcommand."
                path = "."
                arch = "x86_64-avx2"
                limit = 20
                if "--path" in cmd_parts:
                    idx = cmd_parts.index("--path")
                    if idx + 1 < len(cmd_parts):
                        path = cmd_parts[idx + 1]
                if "--arch" in cmd_parts:
                    idx = cmd_parts.index("--arch")
                    if idx + 1 < len(cmd_parts):
                        arch = cmd_parts[idx + 1]
                if "--limit" in cmd_parts:
                    idx = cmd_parts.index("--limit")
                    if idx + 1 < len(cmd_parts):
                        limit = int(cmd_parts[idx + 1])
                return self.cpu_scan(path=path, arch=arch, limit=limit)

            if action == "deadcode":
                threshold = 0.5
                low_usage_max_refs = 1
                lang = None
                evidence = False
                runtime_observed = False
                explain = False
                output_format = "text"
                if "--threshold" in cmd_parts:
                    idx = cmd_parts.index("--threshold")
                    if idx + 1 < len(cmd_parts):
                        threshold = float(cmd_parts[idx + 1])
                if "--low-usage-max-refs" in cmd_parts:
                    idx = cmd_parts.index("--low-usage-max-refs")
                    if idx + 1 < len(cmd_parts):
                        low_usage_max_refs = int(cmd_parts[idx + 1])
                if "--lang" in cmd_parts:
                    idx = cmd_parts.index("--lang")
                    if idx + 1 < len(cmd_parts):
                        lang = str(cmd_parts[idx + 1]).strip() or None
                evidence = "--evidence" in cmd_parts
                runtime_observed = "--runtime-observed" in cmd_parts
                explain = "--explain" in cmd_parts
                if "--format" in cmd_parts:
                    idx = cmd_parts.index("--format")
                    if idx + 1 < len(cmd_parts):
                        output_format = str(cmd_parts[idx + 1]).strip().lower()
                return self.deadcode(
                    threshold=threshold,
                    low_usage_max_refs=low_usage_max_refs,
                    lang=lang,
                    evidence=evidence,
                    runtime_observed=runtime_observed,
                    explain=explain,
                    output_format=output_format,
                )

            if action in {"low-usage", "low_usage"}:
                max_refs = 1
                include_tests = "--include-tests" in cmd_parts
                path = None
                limit = None
                output_format = "text"
                if "--max-refs" in cmd_parts:
                    idx = cmd_parts.index("--max-refs")
                    if idx + 1 < len(cmd_parts):
                        max_refs = int(cmd_parts[idx + 1])
                if "--path" in cmd_parts:
                    idx = cmd_parts.index("--path")
                    if idx + 1 < len(cmd_parts):
                        path = str(cmd_parts[idx + 1]).strip() or None
                if "--limit" in cmd_parts:
                    idx = cmd_parts.index("--limit")
                    if idx + 1 < len(cmd_parts):
                        limit = int(cmd_parts[idx + 1])
                if "--format" in cmd_parts:
                    idx = cmd_parts.index("--format")
                    if idx + 1 < len(cmd_parts):
                        output_format = str(cmd_parts[idx + 1]).strip().lower()
                return self.low_usage(
                    max_refs=max_refs,
                    include_tests=include_tests,
                    path=path,
                    limit=limit,
                    output_format=output_format,
                )

            if action == "unwired":
                threshold = 0.55
                min_nodes = 4
                min_files = 2
                include_tests = "--include-tests" in cmd_parts
                include_fragments = "--include-fragments" in cmd_parts
                max_clusters = 20
                refresh_graph = "--no-refresh-graph" not in cmd_parts
                output_format = "text"
                if "--threshold" in cmd_parts:
                    idx = cmd_parts.index("--threshold")
                    if idx + 1 < len(cmd_parts):
                        threshold = float(cmd_parts[idx + 1])
                if "--min-nodes" in cmd_parts:
                    idx = cmd_parts.index("--min-nodes")
                    if idx + 1 < len(cmd_parts):
                        min_nodes = int(cmd_parts[idx + 1])
                if "--min-files" in cmd_parts:
                    idx = cmd_parts.index("--min-files")
                    if idx + 1 < len(cmd_parts):
                        min_files = int(cmd_parts[idx + 1])
                if "--max-clusters" in cmd_parts:
                    idx = cmd_parts.index("--max-clusters")
                    if idx + 1 < len(cmd_parts):
                        max_clusters = int(cmd_parts[idx + 1])
                if "--format" in cmd_parts:
                    idx = cmd_parts.index("--format")
                    if idx + 1 < len(cmd_parts):
                        output_format = str(cmd_parts[idx + 1]).strip().lower()
                return self.unwired(
                    threshold=threshold,
                    min_nodes=min_nodes,
                    min_files=min_files,
                    include_tests=include_tests,
                    include_fragments=include_fragments,
                    max_clusters=max_clusters,
                    refresh_graph=refresh_graph,
                    output_format=output_format,
                )

            if action == "memory":
                return self._memory_command(cmd_parts[1:])

            if action == "read_file":
                if len(cmd_parts) < 2:
                    return "Error: read_file requires a path"
                return json.dumps(self._api.read_file(cmd_parts[1]), indent=2)

            if action == "listdir":
                path = cmd_parts[1] if len(cmd_parts) > 1 else "."
                recursive = "--recursive" in cmd_parts
                return json.dumps(
                    self._api.list_directory(path, recursive=recursive), indent=2
                )

            if action == "module_structure":
                path = cmd_parts[1] if len(cmd_parts) > 1 else "."
                return json.dumps(self._api.module_structure(path), indent=2)

            if action == "chronicle":
                if len(cmd_parts) > 1:
                    op = cmd_parts[1]
                    if op == "diff":
                        return json.dumps(self.create_chronicle_diff(), indent=2)
                    if op == "list":
                        return self.list_chronicles()
                return self.create_chronicle_snapshot()

            if action == "mission_begin":
                desc = cmd_parts[1] if len(cmd_parts) > 1 else "Mission"
                files = cmd_parts[2].split(",") if len(cmd_parts) > 2 else []
                return self.mission_begin(desc, files)

            if action == "mission_end":
                return self.mission_end()

            if action == "index":
                path = cmd_parts[1] if len(cmd_parts) > 1 else "."
                force = "--force" in cmd_parts
                return self.index(path=path, force=force)

            if action == "sync":
                files: list[str] = []
                deleted: list[str] = []
                full = "--full" in cmd_parts
                op = "index"
                peer_id = None
                peer_name = None
                peer_url = None
                auth_token = None
                bundle_path = None
                workspace_id = None
                limit = 1000
                reason = "tool_call"
                if len(cmd_parts) > 1 and not cmd_parts[1].startswith("-"):
                    token = cmd_parts[1].strip().lower()
                    if token in {"serve", "push", "pull", "subscribe"}:
                        op = token
                    elif token == "peer":
                        peer_op = (
                            cmd_parts[2].strip().lower()
                            if len(cmd_parts) > 2 and not cmd_parts[2].startswith("-")
                            else "list"
                        )
                        op = f"peer-{peer_op}"
                if "--files" in cmd_parts:
                    idx = cmd_parts.index("--files")
                    if idx + 1 < len(cmd_parts):
                        files = [p for p in cmd_parts[idx + 1].split(",") if p]
                elif len(cmd_parts) > 1 and not cmd_parts[1].startswith("-"):
                    if op == "index":
                        files = [p for p in cmd_parts[1].split(",") if p]
                if "--deleted" in cmd_parts:
                    idx = cmd_parts.index("--deleted")
                    if idx + 1 < len(cmd_parts):
                        deleted = [p for p in cmd_parts[idx + 1].split(",") if p]
                if "--peer-id" in cmd_parts:
                    idx = cmd_parts.index("--peer-id")
                    if idx + 1 < len(cmd_parts):
                        peer_id = cmd_parts[idx + 1]
                if "--name" in cmd_parts:
                    idx = cmd_parts.index("--name")
                    if idx + 1 < len(cmd_parts):
                        peer_name = cmd_parts[idx + 1]
                if "--url" in cmd_parts:
                    idx = cmd_parts.index("--url")
                    if idx + 1 < len(cmd_parts):
                        peer_url = cmd_parts[idx + 1]
                if "--auth-token" in cmd_parts:
                    idx = cmd_parts.index("--auth-token")
                    if idx + 1 < len(cmd_parts):
                        auth_token = cmd_parts[idx + 1]
                if "--bundle" in cmd_parts:
                    idx = cmd_parts.index("--bundle")
                    if idx + 1 < len(cmd_parts):
                        bundle_path = cmd_parts[idx + 1]
                if "--workspace" in cmd_parts:
                    idx = cmd_parts.index("--workspace")
                    if idx + 1 < len(cmd_parts):
                        workspace_id = cmd_parts[idx + 1]
                if "--limit" in cmd_parts:
                    idx = cmd_parts.index("--limit")
                    if idx + 1 < len(cmd_parts):
                        limit = int(cmd_parts[idx + 1])
                if "--reason" in cmd_parts:
                    idx = cmd_parts.index("--reason")
                    if idx + 1 < len(cmd_parts):
                        reason = cmd_parts[idx + 1]
                return self.sync(
                    changed_files=files,
                    deleted_files=deleted,
                    full=full,
                    reason=reason,
                    action=op,
                    peer_id=peer_id,
                    peer_name=peer_name,
                    peer_url=peer_url,
                    auth_token=auth_token,
                    bundle_path=bundle_path,
                    workspace_id=workspace_id,
                    limit=limit,
                )

            if action == "workspace":
                op = cmd_parts[1] if len(cmd_parts) > 1 else "status"
                limit = 200
                name = None
                workspace_id = None
                against = "main"
                description = ""
                switch = False
                label = "manual"
                if "--limit" in cmd_parts:
                    idx = cmd_parts.index("--limit")
                    if idx + 1 < len(cmd_parts):
                        limit = int(cmd_parts[idx + 1])
                if "--name" in cmd_parts:
                    idx = cmd_parts.index("--name")
                    if idx + 1 < len(cmd_parts):
                        name = cmd_parts[idx + 1]
                if "--workspace" in cmd_parts:
                    idx = cmd_parts.index("--workspace")
                    if idx + 1 < len(cmd_parts):
                        workspace_id = cmd_parts[idx + 1]
                if "--against" in cmd_parts:
                    idx = cmd_parts.index("--against")
                    if idx + 1 < len(cmd_parts):
                        against = cmd_parts[idx + 1]
                if "--description" in cmd_parts:
                    idx = cmd_parts.index("--description")
                    if idx + 1 < len(cmd_parts):
                        description = cmd_parts[idx + 1]
                if "--label" in cmd_parts:
                    idx = cmd_parts.index("--label")
                    if idx + 1 < len(cmd_parts):
                        label = cmd_parts[idx + 1]
                if "--switch" in cmd_parts:
                    switch = True
                if op == "create" and len(cmd_parts) > 2 and not cmd_parts[2].startswith("-"):
                    name = name or cmd_parts[2]
                if op == "switch" and len(cmd_parts) > 2 and not cmd_parts[2].startswith("-"):
                    workspace_id = workspace_id or cmd_parts[2]
                return self.workspace(
                    action=op,
                    limit=limit,
                    name=name,
                    workspace_id=workspace_id,
                    against=against,
                    description=description,
                    switch=switch,
                    label=label,
                )

            if action == "daemon":
                op = cmd_parts[1] if len(cmd_parts) > 1 else "status"
                interval = self._daemon_interval
                lines = 200
                if "--interval" in cmd_parts:
                    idx = cmd_parts.index("--interval")
                    if idx + 1 < len(cmd_parts):
                        interval = int(cmd_parts[idx + 1])
                if "--lines" in cmd_parts:
                    idx = cmd_parts.index("--lines")
                    if idx + 1 < len(cmd_parts):
                        lines = int(cmd_parts[idx + 1])
                return self.daemon(action=op, interval=interval, lines=lines)

            if action == "init":
                result = self._api.init(force="--force" in cmd_parts)
                return json.dumps(result, indent=2)

            if action == "health":
                return json.dumps(self._api.health(), indent=2)

            if action == "doctor":
                return json.dumps(self._api.doctor(), indent=2)

            return (
                f"Error: Unknown Saguaro command '{action}'. "
                "Use: skeleton, slice, impact, query, report, verify, memory, "
                "deadcode, unwired, read_file, listdir, module_structure, chronicle, mission_begin, "
                "mission_end, init, index, sync, workspace, daemon, health, doctor."
            )
        except Exception as e:
            return f"Error executing '{action}': {e}"

    # -------------------------
    # Mission / Workset
    # -------------------------

    def mission_begin(self, description: str, files: List[str]) -> str:
        try:
            ws = self.workset_manager.create_workset(description, files)
            status = self.workset_manager.acquire_lease(ws.id)
            if status["success"]:
                self.active_mission_id = ws.id
                return f"Mission '{description}' started. Workset ID: {ws.id}"
            return f"Error starting mission: {status['message']}"
        except Exception as e:
            return f"Error creating mission workset: {e}"

    def mission_end(self) -> str:
        if not self.active_mission_id:
            return "No active mission to end."

        success = self.workset_manager.release_lease(self.active_mission_id)
        mid = self.active_mission_id
        self.active_mission_id = None
        return (
            f"Mission {mid} ended successfully."
            if success
            else f"Error releasing mission {mid}."
        )

    # -------------------------
    # Holographic state (lightweight)
    # -------------------------

    def get_hd_state(self) -> bytes:
        if self._hd_state_blob:
            return self._hd_state_blob

        # Build deterministic state from chronicle info if available.
        snapshot_info = self._api.chronicle_snapshot(description="Substrate HD State")
        payload = json.dumps(snapshot_info, sort_keys=True).encode("utf-8")
        self._hd_state_blob = payload
        return payload

    def set_hd_state(self, state_bytes: bytes):
        self._hd_state_blob = state_bytes or b""
        logger.info("Holographic state restored (%d bytes)", len(self._hd_state_blob))

    # -------------------------
    # Core SSAI wrappers
    # -------------------------

    def agent_query(
        self,
        query: str,
        k: int = 5,
        scope: str = "global",
        dedupe_by: str = "entity",
        recall: str | None = None,
        breadth: int | None = None,
        score_threshold: float | None = None,
        stale_file_bias: float | None = None,
        cost_budget: str | None = None,
        corpus_ids: list[str] | str | None = None,
    ) -> str:
        try:
            result = self._api.query(
                query,
                k=k,
                scope=scope,
                dedupe_by=dedupe_by,
                recall=recall,
                breadth=breadth,
                score_threshold=score_threshold,
                stale_file_bias=stale_file_bias,
                cost_budget=cost_budget,
                corpus_ids=corpus_ids,
            )
            return self._format_query_results(result)
        except Exception as e:
            return f"Error executing native Saguaro query: {e}"

    def batch_query(
        self,
        queries: list[str],
        *,
        k: int = 5,
        scope: str = "global",
        dedupe_by: str = "entity",
        recall: str | None = None,
        breadth: int | None = None,
        score_threshold: float | None = None,
        stale_file_bias: float | None = None,
        cost_budget: str | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        normalized = [str(item or "").strip() for item in queries if str(item or "").strip()]
        if not normalized:
            return {}
        payloads = self._api.query_many(
            normalized,
            k=k,
            scope=scope,
            dedupe_by=dedupe_by,
            recall=recall or "balanced",
            breadth=breadth or 24,
            score_threshold=score_threshold or 0.0,
            stale_file_bias=stale_file_bias or 0.0,
            cost_budget=cost_budget or "balanced",
        )
        return {
            query: list((payloads.get(query) or {}).get("results", []))
            for query in normalized
        }

    def agent_query_bundle(self, queries: list[str], *, k: int = 5) -> str:
        try:
            payload = self.batch_query(queries, k=k)
        except Exception as exc:
            return f"Error executing native Saguaro query bundle: {exc}"
        lines: list[str] = []
        for query, results in payload.items():
            lines.append(f"Query: {query}")
            if not results:
                lines.append("  No results.")
                continue
            for row in results[:k]:
                score = row.get("score")
                score_text = f"[{float(score):.3f}] " if isinstance(score, (int, float)) else ""
                lines.append(
                    f"  {score_text}{row.get('name', '<unknown>')} ({row.get('type', 'unknown')}) - "
                    f"{row.get('file', '<unknown>')}:{row.get('line', '?')}"
                )
        return "\n".join(lines)

    def query_template_candidates(
        self, task_intent: str, domain: str | None = None, k: int = 5
    ) -> list[dict[str, Any]]:
        """Return structured semantic candidates for AES template selection."""
        intent = (task_intent or "").strip()
        if not intent:
            return []

        parts = ["aes", "template", intent]
        if domain:
            parts.insert(1, domain)
        query_text = " ".join(parts)

        try:
            result = self._api.query(query_text, k=max(1, k))
        except Exception as exc:
            logger.warning("Saguaro template query failed: %s", exc)
            return []

        candidates: list[dict[str, Any]] = []
        for item in result.get("results", []):
            descriptor = " ".join(
                str(item.get(field, ""))
                for field in ("name", "type", "file", "reason", "scope")
            ).lower()
            if "template" not in descriptor and "aes_" not in descriptor:
                if domain and domain.lower() not in descriptor:
                    continue

            raw_score = item.get("score", 0.0)
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                score = 0.0

            candidates.append(
                {
                    "name": item.get("name", ""),
                    "type": item.get("type", ""),
                    "file": item.get("file", ""),
                    "line": item.get("line", 0),
                    "score": score,
                    "reason": item.get("reason", ""),
                }
            )

        return candidates

    def agent_report(self) -> str:
        try:
            report = self._api.report()
            return json.dumps(report, indent=2)
        except Exception as e:
            return f"Error generating Saguaro report: {e}"

    def agent_skeleton(self, relative_path: str) -> str:
        try:
            result = self._api.skeleton(relative_path)
            lines = [
                f"File: {result.get('file_path')}",
                f"Language: {result.get('language')}",
                "",
            ]

            for symbol in result.get("symbols", []):
                sym_type = symbol.get("type", "symbol")
                name = symbol.get("name", "<unknown>")
                line = symbol.get("line_start")
                if line:
                    lines.append(f"- {sym_type} {name} (line {line})")
                else:
                    lines.append(f"- {sym_type} {name}")

            if not result.get("symbols"):
                lines.append("(No structural elements found)")

            constants = result.get("module_constants", []) or []
            if constants:
                lines.append("")
                lines.append("Module Constants:")
                for const in constants:
                    lines.append(
                        f"- {const.get('name', '<constant>')} (line {const.get('line_start', '?')})"
                    )

            graph = result.get("dependency_graph") or {}
            imports = list(graph.get("imports", []) or [])
            exports = list(graph.get("exports", []) or [])
            if imports or exports:
                lines.append("")
                lines.append("Dependency Graph:")
                if imports:
                    lines.append(f"- Imports ({len(imports)}):")
                    for item in imports[:20]:
                        lines.append(f"  - {item}")
                    if len(imports) > 20:
                        lines.append(f"  - ... +{len(imports) - 20} more")
                if exports:
                    lines.append(f"- Exports ({len(exports)}):")
                    for item in exports[:20]:
                        lines.append(f"  - {item}")
                    if len(exports) > 20:
                        lines.append(f"  - ... +{len(exports) - 20} more")

            return "\n".join(lines)
        except Exception as e:
            return f"Error parsing skeleton: {e}"

    def agent_slice(
        self,
        target: str,
        entity_name: str = None,
        depth: int = 1,
        corpus_id: str | None = None,
    ) -> str:
        """
        Backward-compatible signature:
        - agent_slice("symbol", depth=2)
        - agent_slice("path.py", "entity")
        """
        try:
            if entity_name:
                result = self._api.slice(
                    entity_name,
                    depth=depth,
                    file_path=target,
                    corpus_id=corpus_id,
                )
            else:
                result = self._api.slice(target, depth=depth, corpus_id=corpus_id)

            if "error" in result:
                return result.get("suggestion", result["error"])

            focus = next(
                (c for c in result.get("content", []) if c.get("role") == "focus"), None
            )
            if not focus:
                return "No focus content found."
            return focus.get("code", "")
        except Exception as e:
            return f"Error slicing entity: {e}"

    def agent_impact(self, relative_path: str) -> str:
        try:
            report = self._api.impact(relative_path)
            lines = [
                f"Impact Analysis for: {relative_path}",
                f"Target Module: {report.get('module', 'unknown')}",
                f"Impact Score: {report.get('impact_score', 0)}",
                "",
                f"Tests Impacted ({len(report.get('tests_impacted', []))}):",
            ]
            lines.extend(
                [f"- {self._to_rel_path(p)}" for p in report.get("tests_impacted", [])]
            )

            lines.append("")
            lines.append(
                f"Interfaces Impacted ({len(report.get('interfaces_impacted', []))}):"
            )
            lines.extend(
                [
                    f"- {self._to_rel_path(p)}"
                    for p in report.get("interfaces_impacted", [])
                ]
            )

            return "\n".join(lines)
        except Exception as e:
            return f"Error analyzing impact: {e}"

    def verify(self, path: str = ".", engines: str = None, fix: bool = False) -> str:
        try:
            result = self._api.verify(path=path, engines=engines, fix=fix)
            if result.get("status") == "pass":
                lines = ["Sentinel Validation Passed: No violations."]
                if result.get("receipt_summary"):
                    lines.append("")
                    lines.extend(self._format_fix_receipt_summary(result))
                return "\n".join(lines)

            if result.get("status") in {"ok", "error", "timeout", "blocked"} and not result.get("violations"):
                preflight = dict(result.get("preflight") or {})
                lines = [
                    f"Sentinel Verification Status: {result.get('status', 'unknown')}",
                    f"Preflight: {preflight.get('status', 'unknown')}",
                ]
                for issue in preflight.get("issues", []):
                    lines.append(
                        f"[{issue.get('severity', 'info')}] {issue.get('message', '')} ({issue.get('code', 'preflight')})"
                    )
                skipped = list(result.get("skipped_engines") or [])
                if skipped:
                    lines.append(f"Skipped engines: {', '.join(skipped)}")
                for step in result.get("recovery_steps", []) or preflight.get("recovery_steps", []):
                    lines.append(f"Recovery: {step}")
                return "\n".join(lines)

            lines = [
                f"Sentinel Validation Failed: {result.get('count', 0)} violations found.",
            ]
            for v in result.get("violations", []):
                lines.append(
                    f"[{v.get('severity', 'low')}] {v.get('file', '?')}:{v.get('line', '?')} "
                    f"- {v.get('message', '')} ({v.get('rule_id', 'UNKNOWN')})"
                )
            skipped = list(result.get("skipped_engines") or [])
            if skipped:
                lines.append(f"Skipped engines: {', '.join(skipped)}")
            if result.get("receipt_summary"):
                lines.append("")
                lines.extend(self._format_fix_receipt_summary(result))
            return "\n".join(lines)
        except Exception as e:
            return f"Error running verification: {e}"

    def cpu_scan(
        self,
        path: str = ".",
        arch: str = "x86_64-avx2",
        limit: int = 20,
    ) -> str:
        try:
            result = self._api.cpu_scan(path=path, arch=arch, limit=limit)
            return json.dumps(result, indent=2)
        except Exception as e:
            return f"Error running cpu scan: {e}"

    @staticmethod
    def _format_fix_receipt_summary(result: Dict[str, Any]) -> List[str]:
        summary = list(result.get("receipt_summary") or [])
        if not summary:
            return []
        lines = ["Fix receipt summary:"]
        for item in summary:
            lines.append(
                "- "
                f"{item.get('receipt_id', 'receipt')} "
                f"status={item.get('status', 'unknown')} "
                f"toolchain={item.get('toolchain_state', 'unmanaged')} "
                f"reverify={int(item.get('reverification_violations', 0) or 0)} "
                f"rollback={item.get('rollback_bundle_path') or 'none'}"
            )
        return lines

    def deadcode(
        self,
        threshold: float = 0.5,
        low_usage_max_refs: int = 1,
        lang: str | None = None,
        evidence: bool = False,
        runtime_observed: bool = False,
        explain: bool = False,
        output_format: str = "text",
    ) -> str:
        try:
            report = self._api.deadcode(
                threshold=threshold,
                low_usage_max_refs=low_usage_max_refs,
                lang=lang,
                evidence=bool(evidence),
                runtime_observed=bool(runtime_observed),
                explain=bool(explain),
            )
            if output_format == "json":
                return json.dumps(report, indent=2)

            count = int(report.get("count", 0))
            candidates = list(report.get("candidates", []) or [])
            lines: list[str] = []
            if count == 0:
                lines.append("No dead code found with the selected threshold.")
            else:
                lines.append(f"Found {count} candidates:")
                for item in candidates:
                    confidence = float(item.get("confidence", 0.0))
                    symbol = item.get("symbol", "<unknown>")
                    file_path = item.get("file", "<unknown>")
                    lines.append(f"[{confidence:.2f}] {symbol} ({file_path})")
            low_usage = dict(report.get("low_usage") or {})
            low_usage_candidates = list(low_usage.get("candidates") or [])
            if low_usage_candidates:
                lines.append("")
                lines.append(
                    "Low-usage live symbols "
                    f"(max refs={low_usage.get('max_refs', low_usage_max_refs)}):"
                )
                for item in low_usage_candidates[:20]:
                    evidence = dict(item.get("evidence") or {})
                    usage_count = int(evidence.get("usage_count", 0) or 0)
                    references = list(evidence.get("referencing_files") or [])
                    preview = ", ".join(references[:3]) if references else "none"
                    lines.append(
                        f"[{usage_count}] {item.get('symbol', '<unknown>')} "
                        f"({item.get('file', '<unknown>')}) refs: {preview}"
                    )
            return "\n".join(lines)
        except Exception as e:
            return f"Error running deadcode analysis: {e}"

    def low_usage(
        self,
        max_refs: int = 1,
        include_tests: bool = False,
        path: str | None = None,
        limit: int | None = None,
        output_format: str = "text",
    ) -> str:
        try:
            report = self._api.low_usage(
                max_refs=max_refs,
                include_tests=include_tests,
                path=path,
                limit=limit,
            )
            if output_format == "json":
                return json.dumps(report, indent=2)

            count = int(report.get("count", 0))
            candidates = list(report.get("candidates", []) or [])
            if count == 0:
                return "No low-usage live symbols found."

            lines = [
                f"Found {count} low-usage live symbols (max refs={report.get('max_refs', max_refs)}):"
            ]
            if report.get("path_filter"):
                lines.append(f"Path filter: {report['path_filter']}")
            dry_candidates = list(report.get("dry_candidates") or [])
            if dry_candidates:
                lines.append("Top DRY candidates:")
                for item in dry_candidates[:20]:
                    signals = ", ".join(list(item.get("dry_signals") or [])[:3]) or "low-usage"
                    lines.append(
                        f"[{float(item.get('reuse_score', 0.0) or 0.0):.2f}] "
                        f"{item.get('symbol', '<unknown>')} "
                        f"({item.get('file', '<unknown>')}) signals: {signals}"
                    )
                lines.append("")
            areas = list(report.get("areas") or [])
            if areas:
                lines.append("Top areas:")
                for area in areas[:10]:
                    examples = ", ".join(list(area.get("examples") or [])[:3]) or "n/a"
                    lines.append(
                        f"- {area.get('path', '.')}: "
                        f"{int(area.get('count', 0) or 0)} low-usage, "
                        f"{int(area.get('dry_count', 0) or 0)} DRY candidates "
                        f"({examples})"
                    )
                lines.append("")
            lines.append("Low-usage symbols:")
            for item in candidates[:100]:
                evidence = dict(item.get("evidence") or {})
                usage_count = int(evidence.get("usage_count", 0) or 0)
                references = list(evidence.get("referencing_files") or [])
                preview = ", ".join(references[:3]) if references else "none"
                lines.append(
                    f"[{usage_count}] {item.get('symbol', '<unknown>')} "
                    f"({item.get('file', '<unknown>')}) refs: {preview}"
                )
            return "\n".join(lines)
        except Exception as e:
            return f"Error running low-usage analysis: {e}"

    def unwired(
        self,
        threshold: float = 0.55,
        min_nodes: int = 4,
        min_files: int = 2,
        include_tests: bool = False,
        include_fragments: bool = False,
        max_clusters: int = 20,
        refresh_graph: bool = True,
        output_format: str = "text",
    ) -> str:
        try:
            report = self._api.unwired(
                threshold=threshold,
                min_nodes=min_nodes,
                min_files=min_files,
                include_tests=include_tests,
                include_fragments=include_fragments,
                max_clusters=max_clusters,
                refresh_graph=refresh_graph,
            )
            if output_format == "json":
                return json.dumps(report, indent=2)

            summary = dict(report.get("summary") or {})
            clusters = list(report.get("clusters") or [])
            lines = [
                "Unwired Feature Analysis",
                f"Status: {report.get('status', 'unknown')}",
                (
                    "Clusters: "
                    f"{int(summary.get('cluster_count', 0))} "
                    f"(unreachable_nodes={int(summary.get('unreachable_node_count', 0))}, "
                    f"unreachable_files={int(summary.get('unreachable_file_count', 0))})"
                ),
            ]
            for warning in list(report.get("warnings") or []):
                lines.append(f"Warning: {warning}")
            if not clusters:
                lines.append("No unwired clusters matched the selected filters.")
                return "\n".join(lines)
            lines.append("")
            for cluster in clusters:
                lines.append(
                    "- "
                    f"{cluster.get('label', 'Unwired Cluster')} "
                    f"[{cluster.get('classification', 'unknown')}] "
                    f"confidence={float(cluster.get('confidence', 0.0) or 0.0):.2f} "
                    f"nodes={int(cluster.get('node_count', 0) or 0)} "
                    f"files={int(cluster.get('file_count', 0) or 0)} "
                    f"inbound={int(cluster.get('inbound_from_reachable', 0) or 0)}"
                )
            return "\n".join(lines)
        except Exception as e:
            return f"Error running unwired analysis: {e}"

    def index(
        self,
        path: str = ".",
        force: bool = False,
        recursive: bool = True,
        extensions: Optional[List[str]] = None,
    ) -> str:
        try:
            result = self._api.index(path=path, force=force)
            return (
                "Saguaro indexing complete: "
                f"{result.get('indexed_files', 0)} files, "
                f"{result.get('indexed_entities', 0)} entities "
                f"(backend={result.get('backend', 'unknown')})."
            )
        except Exception as e:
            return f"Error indexing repository: {e}"

    def sync(
        self,
        changed_files: Optional[List[str]] = None,
        deleted_files: Optional[List[str]] = None,
        full: bool = False,
        reason: str = "tool_call",
        action: str = "index",
        peer_id: Optional[str] = None,
        peer_name: Optional[str] = None,
        peer_url: Optional[str] = None,
        auth_token: Optional[str] = None,
        bundle_path: Optional[str] = None,
        workspace_id: Optional[str] = None,
        limit: int = 1000,
    ) -> str:
        try:
            changed = [self._to_rel_path(p) for p in self._normalize_repo_paths(changed_files)]
            deleted = [self._to_rel_path(p) for p in self._normalize_repo_paths(deleted_files)]
            result = self._api.sync(
                action=action,
                changed_files=changed,
                deleted_files=deleted,
                full=full,
                reason=reason,
                peer_id=peer_id,
                peer_name=peer_name,
                peer_url=peer_url,
                auth_token=auth_token,
                bundle_path=bundle_path,
                workspace_id=workspace_id,
                limit=limit,
            )
            if action == "index" and isinstance(result, dict):
                receipt = {
                    "timestamp": datetime.now().isoformat(),
                    "reason": reason,
                    "changed_files": changed,
                    "deleted_files": deleted,
                    "indexed_files": int(((result.get("index") or {}).get("indexed_files", 0)) or 0),
                    "indexed_entities": int(((result.get("index") or {}).get("indexed_entities", 0)) or 0),
                    "removed_files": int(((result.get("index") or {}).get("removed_files", 0)) or 0),
                    "updated_files": int(((result.get("index") or {}).get("updated_files", 0)) or 0),
                    "graph_incremental": bool((((result.get("index") or {}).get("graph") or {}).get("incremental", False))),
                }
                self._write_sync_receipt(receipt)
                compat = dict(result)
                compat["sync"] = receipt
                return json.dumps(compat, indent=2)
            return json.dumps(result, indent=2)
        except Exception as e:
            return f"Error syncing index: {e}"

    def workspace(
        self,
        action: str = "status",
        limit: int = 200,
        name: Optional[str] = None,
        workspace_id: Optional[str] = None,
        against: str = "main",
        description: str = "",
        switch: bool = False,
        label: str = "manual",
    ) -> str:
        action = str(action or "status").strip().lower()
        if action == "scan":
            action = "status"
        if action == "sync":
            return self.sync(action="index", full=True, reason="workspace")
        try:
            payload = self._api.workspace(
                action=action,
                name=name,
                workspace_id=workspace_id,
                against=against,
                description=description,
                switch=switch,
                limit=max(1, int(limit or 200)),
                label=label,
            )
            return json.dumps(payload, indent=2)
        except Exception as e:
            return f"Error: workspace operation failed: {e}"

    def daemon(self, action: str = "status", interval: int = 5, lines: int = 200) -> str:
        action = str(action or "status").strip().lower()
        if action == "restart":
            self._api.daemon(action="stop", interval=interval, lines=lines)
            action = "start"
        try:
            payload = self._api.daemon(action=action, interval=interval, lines=lines)
            return json.dumps(payload, indent=2)
        except Exception as e:
            return f"Error: daemon operation failed: {e}"

    def reality(
        self,
        action: str = "twin",
        run_id: str | None = None,
        limit: int = 500,
    ) -> str:
        op = str(action or "twin").strip().lower()
        try:
            if op == "events":
                payload = self._api.reality_events(run_id=run_id, limit=max(1, int(limit or 500)))
            elif op == "graph":
                payload = self._api.reality_graph(run_id=run_id, limit=max(1, int(limit or 500)))
            elif op == "export":
                if not run_id:
                    return "Error: reality export requires --run-id"
                payload = self._api.reality_export(run_id=run_id, limit=max(1, int(limit or 500)))
            else:
                payload = self._api.reality_twin(run_id=run_id, limit=max(1, int(limit or 500)))
            return json.dumps(payload, indent=2)
        except Exception as e:
            return f"Error: reality operation failed: {e}"

    def corpus(
        self,
        action: str = "list",
        *,
        path: str | None = None,
        corpus_id: str | None = None,
        alias: str | None = None,
        ttl_hours: float = 24.0,
        quarantine: bool = True,
        trust_level: str = "medium",
        build_profile: str = "auto",
        include_expired: bool = False,
        rebuild: bool = False,
    ) -> str:
        try:
            payload = self._api.corpus(
                action=action,
                path=path,
                corpus_id=corpus_id,
                alias=alias,
                ttl_hours=ttl_hours,
                quarantine=quarantine,
                trust_level=trust_level,
                build_profile=build_profile,
                include_expired=include_expired,
                rebuild=rebuild,
            )
            return json.dumps(payload, indent=2)
        except Exception as e:
            return f"Error: corpus operation failed: {e}"

    def compare(
        self,
        *,
        target: str = ".",
        candidates: list[str] | None = None,
        corpus_ids: list[str] | None = None,
        fleet_root: str | None = None,
        top_k: int = 10,
        ttl_hours: float = 72.0,
    ) -> str:
        try:
            payload = self._api.compare(
                target=target,
                candidates=candidates,
                corpus_ids=corpus_ids,
                fleet_root=fleet_root,
                top_k=top_k,
                ttl_hours=ttl_hours,
            )
            return json.dumps(payload, indent=2)
        except Exception as e:
            return f"Error: compare operation failed: {e}"

    def health(self) -> Dict[str, Any]:
        return self._api.health()

    def verify_index_coverage(self, target_files: List[str]) -> Dict[str, int]:
        coverage: Dict[str, int] = {}
        for target in target_files:
            try:
                result = self._api.query(os.path.basename(target), k=20, file=target)
            except Exception:
                coverage[target] = 0
                continue
            matches = [
                item
                for item in result.get("results", [])
                if item.get("file", "").endswith(target)
            ]
            coverage[target] = len(matches)
        return coverage

    def resolve_python_symbols(self, relative_path: str) -> List[str]:
        target = os.path.join(self.root_dir, relative_path)
        if not os.path.exists(target):
            return []
        try:
            source = Path(target).read_text(encoding="utf-8")
            tree = ast.parse(source)
        except Exception:
            return []
        return sorted(
            {
                node.name
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            }
        )

    def read_file(
        self, path: str, start_line: int = None, end_line: int = None
    ) -> Dict[str, Any]:
        return self._api.read_file(path, start_line=start_line, end_line=end_line)

    # -------------------------
    # Chronicle
    # -------------------------

    def create_chronicle_snapshot(self, label: str = "manual") -> str:
        os.makedirs(CHRONICLE_DIR, exist_ok=True)

        # Keep legacy file snapshot for compatibility.
        snapshot = {
            "label": label,
            "timestamp": datetime.now().isoformat(),
            "files": {},
        }
        for root, _, files in os.walk(self.root_dir):
            if ".git" in root or "__pycache__" in root:
                continue
            for file in files:
                if file.endswith((".py", ".md", ".yaml", ".json")):
                    fp = os.path.join(root, file)
                    try:
                        with open(fp, "rb") as f:
                            checksum = hashlib.md5(f.read()).hexdigest()
                        snapshot["files"][os.path.relpath(fp, self.root_dir)] = checksum
                    except Exception:
                        continue

        filename = f"chronicle_{label}_{int(datetime.now().timestamp())}.json"
        save_path = os.path.join(CHRONICLE_DIR, filename)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2)

        # Also persist semantic snapshot via API storage.
        self._api.chronicle_snapshot(description=f"{label} snapshot")
        return f"Chronicle snapshot ({label}) saved to {filename}"

    def create_chronicle_diff(self) -> Dict[str, Any]:
        try:
            return self._api.chronicle_diff()
        except Exception as e:
            return {"status": "error", "message": f"chronicle_diff_failed: {e}"}

    def write_chronicle_delta_log(
        self, diff_payload: Dict[str, Any], trace_id: str, task: str
    ) -> str:
        changelog_dir = os.path.join(self.root_dir, "aiChangeLog")
        os.makedirs(changelog_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_trace = str(trace_id or "trace_unknown").replace("/", "_")
        path = os.path.join(changelog_dir, f"{stamp}_{safe_trace}.json")
        payload = {
            "timestamp": datetime.now().isoformat(),
            "trace_id": trace_id or "trace_unknown",
            "task": task or "",
            "chronicle_diff": diff_payload,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return path

    def run_legislation_draft(
        self, reason: str = "phase6_drift_cycle"
    ) -> Dict[str, Any]:
        try:
            from saguaro.legislator import Legislator

            legislator = Legislator(root_dir=self.root_dir)
            yaml_content = legislator.draft_rules()
            output_path = os.path.join(self.root_dir, ".saguaro.rules.draft")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(yaml_content)
            return {
                "status": "ok",
                "reason": reason,
                "draft_path": output_path,
                "generated_at": int(time.time()),
            }
        except Exception as e:
            return {"status": "error", "message": f"legislation_draft_failed: {e}"}

    def list_chronicles(self) -> str:
        if not os.path.exists(CHRONICLE_DIR):
            return "No chronicle snapshots found."
        files = sorted(os.listdir(CHRONICLE_DIR), reverse=True)
        return "\n".join(files) if files else "No snapshots."

    # -------------------------
    # Memory wrapper
    # -------------------------

    def _memory_command(self, args: List[str]) -> str:
        if not args:
            return "Usage: memory [list|read <key>|write <key> <value> [--tier <tier>]]"

        action = args[0]
        tier = "working"
        if "--tier" in args:
            i = args.index("--tier")
            if i + 1 < len(args):
                tier = args[i + 1]

        try:
            if action == "list":
                return json.dumps(self._api.memory_list(), indent=2)

            if action == "read":
                if len(args) < 2:
                    return "Error: memory read requires a key"
                return json.dumps(self._api.memory_read(args[1], tier=tier), indent=2)

            if action in {"write", "store"}:
                if len(args) < 3:
                    return "Error: memory write requires key and value"
                key = args[1]
                value = " ".join(args[2:])
                if "--tier" in value:
                    value = value.split("--tier")[0].strip()
                return json.dumps(
                    self._api.memory_write(key, value, tier=tier), indent=2
                )

            return f"Error: Unknown memory action '{action}'."
        except Exception as e:
            return f"Error accessing memory: {e}"

    # -------------------------
    # Formatting helpers
    # -------------------------

    def _format_query_results(self, result: Dict[str, Any]) -> str:
        lines = [f"Query: '{result.get('query', '')}'"]
        query_plan = dict(result.get("query_plan") or {})
        if query_plan:
            lines.append(
                "Plan: "
                f"recall={query_plan.get('recall', 'balanced')} "
                f"breadth={query_plan.get('breadth', '?')} "
                f"threshold={query_plan.get('score_threshold')} "
                f"budget={query_plan.get('cost_budget', 'balanced')}"
            )
        for item in result.get("results", []):
            lines.append(
                f"[{item.get('rank', '?')}] [{item.get('score', 0.0):.4f}] "
                f"{item.get('name', 'unknown')} ({item.get('type', 'symbol')})"
            )
            lines.append(f"    Path: {item.get('file', '?')}:{item.get('line', '?')}")
            if item.get("corpus_id"):
                lines.append(f"    Corpus: {item.get('corpus_id')}")
            if item.get("reason"):
                lines.append(f"    Why:  {item['reason']}")
            if item.get("scope"):
                lines.append(f"    Scope: {item['scope']}")
            lines.append("")

        return "\n".join(lines).rstrip()

    def _to_rel_path(self, path: str) -> str:
        if not path:
            return path
        if os.path.isabs(path):
            return os.path.relpath(path, self.root_dir)
        return path

    def _normalize_repo_paths(self, paths: Optional[List[str]]) -> list[str]:
        normalized: list[str] = []
        for item in paths or []:
            raw = str(item or "").strip()
            if not raw:
                continue
            full = raw if os.path.isabs(raw) else os.path.join(self.root_dir, raw)
            normalized.append(os.path.abspath(full))
        return sorted(set(normalized))

    def _sync_receipt_path(self) -> str:
        return os.path.join(self.root_dir, ".saguaro", "workspace_sync.json")

    def _write_sync_receipt(self, receipt: Dict[str, Any]) -> None:
        out_path = self._sync_receipt_path()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(receipt, f, indent=2)

    def _load_sync_receipt(self) -> Dict[str, Any]:
        out_path = self._sync_receipt_path()
        if not os.path.exists(out_path):
            return {}
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _workspace_status(self, limit: int = 200) -> Dict[str, Any]:
        tracking_path = os.path.join(self.root_dir, ".saguaro", "tracking.json")
        tracked: list[str] = []
        missing: list[str] = []
        if os.path.exists(tracking_path):
            try:
                with open(tracking_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                if isinstance(state, dict):
                    tracked = sorted(state.keys())
            except Exception:
                tracked = []
        for file_path in tracked:
            if not os.path.exists(file_path):
                missing.append(file_path)

        git_changes: list[str] = []
        git_dirty = False
        try:
            proc = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.root_dir,
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            if proc.returncode == 0:
                for line in (proc.stdout or "").splitlines():
                    if not line.strip():
                        continue
                    git_dirty = True
                    git_changes.append(line.strip())
        except Exception:
            pass

        return {
            "status": "ok",
            "repo_root": self.root_dir,
            "tracked_files": len(tracked),
            "missing_tracked_files": len(missing),
            "missing_sample": [self._to_rel_path(p) for p in missing[:limit]],
            "git_dirty": git_dirty,
            "git_changes": git_changes[:limit],
            "last_sync": self._load_sync_receipt(),
            "daemon": self._daemon_status_payload(),
        }

    def _daemon_status_payload(self) -> Dict[str, Any]:
        running = bool(self._daemon_thread and self._daemon_thread.is_alive())
        return {
            "running": running,
            "interval_seconds": int(self._daemon_interval),
        }

    def _start_daemon(self, interval: int = 5) -> str:
        if self._daemon_thread and self._daemon_thread.is_alive():
            return json.dumps({"status": "ok", "daemon": self._daemon_status_payload()}, indent=2)

        interval = max(1, int(interval or 5))
        self._daemon_interval = interval
        try:
            self._api.init()
            from saguaro.indexing.auto_scaler import get_repo_stats_and_config
            from saguaro.indexing.engine import IndexEngine
            from saguaro.watcher import Watcher

            saguaro_dir = os.path.join(self.root_dir, ".saguaro")
            stats = get_repo_stats_and_config(self.root_dir)
            engine = IndexEngine(self.root_dir, saguaro_dir, stats)
            watcher = Watcher(engine, self.root_dir, interval=interval)
            thread = threading.Thread(target=watcher.start, daemon=True, name="saguaro-watcher")
            thread.start()
            self._daemon_watcher = watcher
            self._daemon_thread = thread
            return json.dumps({"status": "started", "daemon": self._daemon_status_payload()}, indent=2)
        except Exception as e:
            self._daemon_watcher = None
            self._daemon_thread = None
            return f"Error starting daemon: {e}"

    def _stop_daemon(self) -> str:
        watcher = self._daemon_watcher
        thread = self._daemon_thread
        if watcher is not None:
            try:
                watcher.stop()
            except Exception:
                pass
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(2, self._daemon_interval + 1))
        self._daemon_watcher = None
        self._daemon_thread = None
        return json.dumps({"status": "stopped", "daemon": self._daemon_status_payload()}, indent=2)
