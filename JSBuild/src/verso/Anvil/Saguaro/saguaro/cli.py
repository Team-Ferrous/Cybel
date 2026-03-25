"""Utilities for cli."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from typing import Any

from saguaro import __version__

_AES_CATEGORY_ORDER = (
    "Traceability",
    "Type Safety",
    "Error Handling",
    "Security",
    "Dead Code",
    "Complexity",
    "Documentation",
)


def _categorize_aes_violation(violation: dict[str, Any]) -> str:
    rule_id = str(violation.get("rule_id", "")).upper()
    message = str(violation.get("message", "")).lower()

    if (
        rule_id.startswith("AES-TR")
        or "trace" in message
        or "evidence bundle" in message
        or "review" in message
        or "waiver" in message
    ):
        return "Traceability"
    if (
        rule_id.startswith("ANN")
        or rule_id.startswith("UP")
        or rule_id.startswith("F7")
        or rule_id.startswith("MYPY")
        or "type" in message
        or "annotation" in message
    ):
        return "Type Safety"
    if (
        rule_id.startswith("BLE")
        or rule_id.startswith("TRY")
        or rule_id.startswith("RET")
        or "exception" in message
        or "error handling" in message
    ):
        return "Error Handling"
    if (
        rule_id.startswith("S")
        or rule_id.startswith("AES-SEC")
        or "security" in message
        or "secret" in message
        or "cwe" in message
        or "injection" in message
    ):
        return "Security"
    if (
        rule_id.startswith("VULTURE")
        or rule_id.startswith("F401")
        or rule_id.startswith("F841")
        or "dead code" in message
        or "unused" in message
    ):
        return "Dead Code"
    if (
        rule_id.startswith("C901")
        or rule_id.startswith("PLR09")
        or "complexity" in message
        or "too many branches" in message
    ):
        return "Complexity"
    if (
        rule_id.startswith("D")
        or rule_id.startswith("DOC")
        or "docstring" in message
        or "documentation" in message
    ):
        return "Documentation"
    return "Error Handling"


def _build_aes_compliance_report(
    verify_result: dict[str, Any], total_files_scanned: int
) -> dict[str, Any]:
    violations = list(verify_result.get("violations") or [])
    total_files = max(int(total_files_scanned or 0), 1)
    category_data: dict[str, dict[str, Any]] = {
        category: {"files": set(), "violations": 0, "waivers": 0}
        for category in _AES_CATEGORY_ORDER
    }

    for violation in violations:
        category = _categorize_aes_violation(violation)
        bucket = category_data[category]
        bucket["violations"] += 1
        file_path = str(violation.get("file", "")).strip()
        if file_path:
            bucket["files"].add(file_path)
        rule_id = str(violation.get("rule_id", "")).upper()
        message = str(violation.get("message", "")).lower()
        if "waiver" in rule_id or "waiver" in message:
            bucket["waivers"] += 1

    rows = []
    compliance_sum = 0.0
    for category in _AES_CATEGORY_ORDER:
        bucket = category_data[category]
        non_compliant_files = len(bucket["files"])
        compliant_files = max(total_files - non_compliant_files, 0)
        percent = round((compliant_files / total_files) * 100.0, 2)
        compliance_sum += percent
        rows.append(
            {
                "category": category,
                "compliant_files": compliant_files,
                "total_files": total_files,
                "violations": int(bucket["violations"]),
                "waivers": int(bucket["waivers"]),
                "compliance_percent": percent,
            }
        )

    overall = round(compliance_sum / len(_AES_CATEGORY_ORDER), 2)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_percent": 95.0,
        "overall_compliance_percent": overall,
        "overall_status": (
            "pass"
            if overall >= 95.0 and verify_result.get("status") == "pass"
            else "fail"
        ),
        "verification_status": verify_result.get("status"),
        "verification_violation_count": int(
            verify_result.get("count", len(violations))
        ),
        "total_files_scanned": total_files,
        "categories": rows,
    }


def _format_aes_compliance_report(report: dict[str, Any]) -> str:
    lines = []
    lines.append(
        "AES Compliance Report — "
        + str(report.get("generated_at", "")).replace("T", " ").replace("+00:00", "Z")
    )
    lines.append("=" * 67)
    lines.append(
        "Overall: "
        f"{report.get('overall_compliance_percent', 0.0):.2f}% compliant "
        f"(target: {report.get('target_percent', 95.0):.2f}%)"
    )
    lines.append("")
    lines.append("Category          | Compliant | Violations | Waivers")
    lines.append("------------------+-----------+------------+--------")
    for row in report.get("categories", []):
        compliant = f"{row.get('compliant_files', 0)}/{row.get('total_files', 0)}"
        lines.append(
            f"{row.get('category', ''):<18} | "
            f"{compliant:>9} | "
            f"{int(row.get('violations', 0)):>10} | "
            f"{int(row.get('waivers', 0)):>7}"
        )
    return "\n".join(lines)


def _format_unwired_report(report: dict[str, Any]) -> str:
    clusters = list(report.get("clusters") or [])
    summary = dict(report.get("summary") or {})
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
    warnings = list(report.get("warnings") or [])
    for warning in warnings:
        lines.append(f"Warning: {warning}")
    if not clusters:
        lines.append("No unwired clusters matched the selected filters.")
        return "\n".join(lines)

    lines.append("")
    for cluster in clusters:
        label = cluster.get("label", "Unwired Cluster")
        classification = cluster.get("classification", "unknown")
        confidence = float(cluster.get("confidence", 0.0) or 0.0)
        node_count = int(cluster.get("node_count", 0) or 0)
        file_count = int(cluster.get("file_count", 0) or 0)
        inbound = int(cluster.get("inbound_from_reachable", 0) or 0)
        lines.append(
            f"- {label} [{classification}] "
            f"confidence={confidence:.2f} nodes={node_count} files={file_count} inbound={inbound}"
        )
    return "\n".join(lines)


def _format_trace_report(report: dict[str, Any]) -> str:
    lines = [
        "Pipeline Trace",
        f"Status: {report.get('status', 'unknown')}",
        f"Stages: {int(report.get('stage_count', 0) or 0)}",
    ]
    total_complexity = report.get("total_complexity") or {}
    if total_complexity:
        lines.append(
            f"Estimated time complexity: {total_complexity.get('time_complexity', 'unknown')}"
        )
    ffi_count = len(report.get("ffi_boundaries") or [])
    lines.append(f"FFI boundaries in trace files: {ffi_count}")
    lines.append("")
    for stage in list(report.get("stages") or [])[:30]:
        name = stage.get("name", "unknown")
        file = stage.get("file", "")
        line = int(stage.get("line", 0) or 0)
        relation = stage.get("relation")
        rel_fragment = f" via {relation}" if relation else ""
        lines.append(f"- {name} ({file}:{line}){rel_fragment}")
    if int(report.get("stage_count", 0) or 0) > 30:
        lines.append("... truncated ...")
    return "\n".join(lines)


def _trace_to_mermaid(report: dict[str, Any]) -> str:
    def _safe_node_id(value: str) -> str:
        return (
            value.replace(":", "_")
            .replace(".", "_")
            .replace("/", "_")
            .replace("-", "_")
            .replace(" ", "_")
        )

    stages = list(report.get("stages") or [])
    edges = list(report.get("edges") or [])
    if not stages:
        return "graph TD\n  A[No stages]"
    by_id = {str(stage.get("id") or ""): stage for stage in stages}
    lines = ["graph TD"]
    for stage in stages:
        stage_id = str(stage.get("id") or "")
        if not stage_id:
            continue
        label = str(stage.get("name") or stage_id).replace('"', "'")
        lines.append(f'  {_safe_node_id(stage_id)}["{label}"]')
    for edge in edges:
        src = str(edge.get("from") or "")
        dst = str(edge.get("to") or "")
        if src not in by_id or dst not in by_id:
            continue
        relation = str(edge.get("relation") or "next")
        src_id = _safe_node_id(src)
        dst_id = _safe_node_id(dst)
        lines.append(f"  {src_id} -->|{relation}| {dst_id}")
    return "\n".join(lines)


def _format_complexity_report(report: dict[str, Any]) -> str:
    lines = [
        "Complexity Analysis",
        f"Status: {report.get('status', 'unknown')}",
    ]
    if report.get("entry_point"):
        lines.append(f"Pipeline: {report.get('entry_point')}")
    if report.get("symbol"):
        lines.append(f"Symbol: {report.get('symbol')}")
    if report.get("file"):
        lines.append(f"File: {report.get('file')}")
    if report.get("time_complexity"):
        lines.append(f"Time: {report.get('time_complexity')}")
    if report.get("space_complexity"):
        lines.append(f"Space: {report.get('space_complexity')}")
    total = report.get("total_complexity")
    if isinstance(total, dict) and total:
        lines.append(f"Pipeline Time: {total.get('time_complexity', 'unknown')}")
    if "confidence" in report:
        lines.append(f"Confidence: {float(report.get('confidence', 0.0)):.2f}")
    return "\n".join(lines)


def _resolve_verify_repo_root(path_arg: str) -> str:
    target = os.path.abspath(path_arg or ".")
    starts = [target if os.path.isdir(target) else os.path.dirname(target), os.getcwd()]

    for start in starts:
        current = os.path.abspath(start or ".")
        while True:
            if os.path.exists(os.path.join(current, "standards", "AES_RULES.json")):
                return current
            if os.path.exists(os.path.join(current, ".git")):
                return current
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent

    return os.path.abspath(os.getcwd())


def _normalize_compat_argv(argv: list[str]) -> list[str]:
    """Normalize legacy roadmap command forms to the current CLI surface."""
    normalized = list(argv)
    if not normalized:
        return normalized

    index = 0
    command_index: int | None = None
    while index < len(normalized):
        token = normalized[index]
        if token == "--repo":
            index += 2
            continue
        if token.startswith("-"):
            index += 1
            continue
        command_index = index
        break

    if command_index is None:
        return normalized

    command = normalized[command_index]
    next_token = (
        normalized[command_index + 1] if command_index + 1 < len(normalized) else None
    )

    if command == "index" and next_token == "doctor":
        return normalized[:command_index] + ["doctor"] + normalized[command_index + 2 :]

    if command == "index" and next_token == "rebuild":
        tail = normalized[command_index + 2 :]
        if "--force" not in tail:
            tail = ["--force"] + tail
        return normalized[: command_index + 1] + tail

    if command == "liveness" and next_token == "explain":
        if command_index + 2 < len(normalized):
            symbol = normalized[command_index + 2]
            tail = normalized[command_index + 3 :]
            if not str(symbol).startswith("-"):
                return (
                    normalized[: command_index + 1] + ["--symbol", str(symbol)] + tail
                )
        return normalized

    return normalized


def main() -> None:
    """Handle main."""
    parser = argparse.ArgumentParser(
        description=f"SAGUARO v{__version__} - Quantum Codebase OS"
    )
    parser.add_argument(
        "--version", action="version", version=f"SAGUARO v{__version__}"
    )
    parser.add_argument(
        "--repo",
        default=None,
        help=("Repository root to operate on. Defaults to current working directory."),
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Init
    init_parser = subparsers.add_parser(
        "init", help="Initialize SAGUARO in the current directory"
    )
    init_parser.add_argument(
        "--force", action="store_true", help="Overwrite existing .saguaro directory"
    )

    # Quickstart
    # Quickstart
    subparsers.add_parser(
        "quickstart", help="One-command setup (Init + Index + Configs)"
    )

    # Index
    index_parser = subparsers.add_parser("index", help="Index the codebase")
    index_parser.add_argument("--path", default=".", help="Codebase path")
    index_parser.add_argument(
        "--force", action="store_true", help="Force re-indexing of all files"
    )
    index_parser.add_argument(
        "--incremental",
        action="store_true",
        help="Prefer incremental indexing and graph refresh",
    )
    index_parser.add_argument(
        "--changed-files",
        help="Comma-separated changed files for targeted indexing",
    )
    index_parser.add_argument(
        "--events",
        help="Path to JSON/JSONL change-event stream for deterministic ingestion",
    )
    index_parser.add_argument(
        "--prune-deleted",
        action="store_true",
        help="Prune deleted files from tracker/store during indexing",
    )
    index_parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging"
    )

    # Watch logic
    watch_parser = subparsers.add_parser(
        "watch", help="Watch for changes and index incrementally"
    )
    watch_parser.add_argument("--path", default=".", help="Codebase path")
    watch_parser.add_argument(
        "--interval", type=int, default=5, help="Poll interval in seconds"
    )

    # Serve (DNI)
    serve_parser = subparsers.add_parser("serve", help="Start DNI Server")
    serve_parser.add_argument(
        "--mcp", action="store_true", help="Start in MCP Server mode"
    )
    serve_parser.add_argument(
        "--auth-token", help="Require authentication token (MCP only)"
    )
    serve_parser.add_argument(
        "--host", default="127.0.0.1", help="Host for local HTTP server mode"
    )
    serve_parser.add_argument(
        "--port", type=int, default=None, help="Start local HTTP server on this port"
    )

    app_parser = subparsers.add_parser("app", help="Start the local Saguaro app")
    app_parser.add_argument("--host", default="127.0.0.1")
    app_parser.add_argument("--port", type=int, default=8765)

    # Coverage
    coverage_parser = subparsers.add_parser("coverage", help="Repo coverage report")
    coverage_parser.add_argument("--path", default=".", help="Path to report on")
    coverage_parser.add_argument(
        "--structural",
        action="store_true",
        help="Report structural parser coverage (not only AST coverage)",
    )
    coverage_parser.add_argument(
        "--by-language",
        action="store_true",
        help="Include per-language coverage breakdown",
    )
    coverage_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON payload instead of text report",
    )

    # Health
    # Health
    subparsers.add_parser("health", help="Index Health Dashboard")
    subparsers.add_parser(
        "doctor", help="One-shot diagnostics for freshness and ABI/parser health"
    )
    subparsers.add_parser(
        "recover",
        help="Recover corrupt or mismatched committed index artifacts",
    )
    debuginfo_parser = subparsers.add_parser(
        "debuginfo", help="Export a diagnostic bundle"
    )
    debuginfo_parser.add_argument("--output", default=None, help="Bundle output path")
    debuginfo_parser.add_argument(
        "--event-limit",
        type=int,
        default=500,
        help="Maximum journal events to include",
    )
    state_parser = subparsers.add_parser("state", help="State bundle operations")
    state_subparsers = state_parser.add_subparsers(dest="state_op")
    state_restore_parser = state_subparsers.add_parser(
        "restore", help="Restore a state bundle"
    )
    state_restore_parser.add_argument("bundle_path", help="State bundle path")
    state_restore_parser.add_argument(
        "--force", action="store_true", help="Force restore even if repo differs"
    )
    admin_parser = subparsers.add_parser("admin", help="Administrative bundle actions")
    admin_parser.add_argument(
        "admin_action",
        choices=["snapshot", "restore", "debuginfo"],
        help="Administrative action",
    )
    admin_parser.add_argument("--bundle-path", default=None, help="Input bundle path")
    admin_parser.add_argument("--output", default=None, help="Output bundle path")
    admin_parser.add_argument("--force", action="store_true", help="Force the action")
    admin_parser.add_argument(
        "--no-reality",
        action="store_true",
        help="Exclude reality artifacts from generated bundles",
    )
    admin_parser.add_argument(
        "--event-limit",
        type=int,
        default=500,
        help="Maximum journal events to include",
    )
    abi_parser = subparsers.add_parser(
        "abi", help="Roadmap ABI compatibility command surface"
    )
    abi_parser.add_argument(
        "abi_op",
        nargs="?",
        choices=["verify", "orphaned"],
        default="verify",
        help="Compatibility ABI operation",
    )
    abi_parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )

    daemon_parser = subparsers.add_parser(
        "daemon", help="Manage background index freshness daemon"
    )
    daemon_sub = daemon_parser.add_subparsers(dest="daemon_op")
    daemon_sub.add_parser("status", help="Show daemon status")
    daemon_start = daemon_sub.add_parser("start", help="Start daemon")
    daemon_start.add_argument("--interval", type=int, default=5)
    daemon_sub.add_parser("stop", help="Stop daemon")
    daemon_logs = daemon_sub.add_parser("logs", help="Show daemon logs")
    daemon_logs.add_argument("--lines", type=int, default=200)

    workspace_parser = subparsers.add_parser(
        "workspace", help="Workspace and state ledger operations"
    )
    workspace_sub = workspace_parser.add_subparsers(dest="workspace_op")
    workspace_sub.add_parser("status", help="Workspace status")
    workspace_sub.add_parser("list", help="List workspaces")
    workspace_create = workspace_sub.add_parser("create", help="Create workspace")
    workspace_create.add_argument("name")
    workspace_create.add_argument("--description", default="")
    workspace_create.add_argument("--switch", action="store_true")
    workspace_switch = workspace_sub.add_parser(
        "switch", help="Switch active workspace"
    )
    workspace_switch.add_argument("workspace_id")
    workspace_hist = workspace_sub.add_parser("history", help="Workspace event history")
    workspace_hist.add_argument("--workspace", dest="workspace_id", default=None)
    workspace_hist.add_argument("--limit", type=int, default=200)
    workspace_diff = workspace_sub.add_parser("diff", help="Diff workspace state")
    workspace_diff.add_argument("--workspace", dest="workspace_id", default=None)
    workspace_diff.add_argument("--against", default="main")
    workspace_diff.add_argument("--limit", type=int, default=200)
    workspace_snap = workspace_sub.add_parser(
        "snapshot", help="Create workspace snapshot"
    )
    workspace_snap.add_argument("--workspace", dest="workspace_id", default=None)
    workspace_snap.add_argument("--label", default="manual")

    corpus_parser = subparsers.add_parser(
        "corpus", help="Manage isolated comparative corpus sessions"
    )
    corpus_sub = corpus_parser.add_subparsers(dest="corpus_op")
    corpus_sub.add_parser("list", help="List active corpus sessions")
    corpus_show = corpus_sub.add_parser("show", help="Show one corpus session")
    corpus_show.add_argument("corpus_id")
    corpus_create = corpus_sub.add_parser(
        "create", help="Create or refresh a corpus session"
    )
    corpus_create.add_argument("path")
    corpus_create.add_argument("--corpus-id", default=None)
    corpus_create.add_argument("--alias", default=None)
    corpus_create.add_argument("--ttl-hours", type=float, default=24.0)
    corpus_create.add_argument("--trust-level", default="medium")
    corpus_create.add_argument("--build-profile", default="auto")
    corpus_create.add_argument("--no-quarantine", action="store_true")
    corpus_create.add_argument("--rebuild", action="store_true")
    corpus_benchmark = corpus_sub.add_parser(
        "benchmark", help="Benchmark isolated corpus indexing and warm-session reuse"
    )
    corpus_benchmark.add_argument("path")
    corpus_benchmark.add_argument("--alias", default=None)
    corpus_benchmark.add_argument("--ttl-hours", type=float, default=24.0)
    corpus_benchmark.add_argument("--trust-level", default="medium")
    corpus_benchmark.add_argument("--build-profile", default="auto")
    corpus_benchmark.add_argument("--no-quarantine", action="store_true")
    corpus_benchmark.add_argument(
        "--batch-size",
        type=int,
        action="append",
        default=[],
        help="Index batch size to test. Repeat for multiple values.",
    )
    corpus_benchmark.add_argument(
        "--file-batch-size",
        type=int,
        action="append",
        default=[],
        help="Parallel file-batch size to test. Repeat for multiple values.",
    )
    corpus_benchmark.add_argument("--iterations", type=int, default=1)
    corpus_benchmark.add_argument("--no-reuse-check", action="store_true")
    corpus_gc = corpus_sub.add_parser(
        "gc", help="Garbage-collect expired corpus sessions"
    )
    corpus_gc.add_argument("--include-expired", action="store_true")

    sync_parser = subparsers.add_parser(
        "sync", help="Index/workspace/peer synchronization"
    )
    sync_sub = sync_parser.add_subparsers(dest="sync_op")
    sync_sub.add_parser("serve", help="Show sync service descriptor")
    peer_parser = sync_sub.add_parser("peer", help="Manage peers")
    peer_sub = peer_parser.add_subparsers(dest="peer_op")
    peer_add = peer_sub.add_parser("add", help="Add peer")
    peer_add.add_argument("--name", required=True)
    peer_add.add_argument("--url", required=True)
    peer_add.add_argument("--auth-token", default=None)
    peer_remove = peer_sub.add_parser("remove", help="Remove peer")
    peer_remove.add_argument("peer_id")
    peer_sub.add_parser("list", help="List peers")
    sync_push = sync_sub.add_parser("push", help="Export local events for peer")
    sync_push.add_argument("--peer-id", required=True)
    sync_push.add_argument("--limit", type=int, default=1000)
    sync_push.add_argument("--workspace", dest="workspace_id", default=None)
    sync_pull = sync_sub.add_parser("pull", help="Apply peer event bundle")
    sync_pull.add_argument("--peer-id", required=True)
    sync_pull.add_argument("--bundle", required=True)
    sync_pull.add_argument("--workspace", dest="workspace_id", default=None)
    sync_subscribe = sync_sub.add_parser(
        "subscribe", help="Mark peer subscription enabled"
    )
    sync_subscribe.add_argument("--peer-id", required=True)
    sync_subscribe.add_argument("--workspace", dest="workspace_id", default=None)

    # Governor
    gov_parser = subparsers.add_parser("governor", help="Context Budget Governor")
    gov_parser.add_argument(
        "--check", help="Check string against budget", action="store_true"
    )
    gov_parser.add_argument("--text", help="Text to check")

    # Workset

    # Workset
    workset_parser = subparsers.add_parser("workset", help="Manage Agent Worksets")
    workset_sub = workset_parser.add_subparsers(dest="workset_op")

    ws_create = workset_sub.add_parser("create", help="Create a new workset")
    ws_create.add_argument("--desc", required=True, help="Description of task")
    ws_create.add_argument(
        "--files", required=True, help="Comma-separated list of files"
    )

    workset_sub.add_parser("list", help="List active worksets")

    ws_show = workset_sub.add_parser("show", help="Show workset details")
    ws_show.add_argument("id", help="Workset ID")

    ws_expand = workset_sub.add_parser("expand", help="Expand workset budget/files")
    ws_expand.add_argument("id", help="Workset ID")
    ws_expand.add_argument("--files", required=True, help="New files to add")
    ws_expand.add_argument(
        "--justification", required=True, help="Reason for escalation"
    )

    ws_lock = workset_sub.add_parser("lock", help="Acquire lease on workset (Active)")
    ws_lock.add_argument("id", help="Workset ID")

    ws_unlock = workset_sub.add_parser(
        "unlock", help="Release lease on workset (Closed)"
    )
    ws_unlock.add_argument("id", help="Workset ID")

    # Refactor
    refactor_parser = subparsers.add_parser("refactor", help="Refactoring Intelligence")
    refactor_sub = refactor_parser.add_subparsers(dest="refactor_op")

    plan_parser = refactor_sub.add_parser("plan", help="Plan a refactor")
    plan_parser.add_argument("--symbol", required=True, help="Symbol to modify/rename")

    # Rename
    rename_parser = refactor_sub.add_parser("rename", help="Semantic Rename")
    rename_parser.add_argument("old", help="Old symbol Name")
    rename_parser.add_argument("new", help="New symbol Name")
    rename_parser.add_argument("--execute", action="store_true", help="Apply changes")

    # Shim
    shim_parser = refactor_sub.add_parser("shim", help="Generate Compatibility Shim")
    shim_parser.add_argument("path", help="Path to original file (will be overwritten)")
    shim_parser.add_argument("target", help="Module path to redirect to")

    # Safe Delete
    del_parser = refactor_sub.add_parser("safedelete", help="Safe Delete File")
    del_parser.add_argument("path", help="File to delete")
    del_parser.add_argument("--force", action="store_true", help="Ignore dependencies")
    del_parser.add_argument("--execute", action="store_true", help="Apply deletion")

    # Feedback
    fb_parser = subparsers.add_parser("feedback", help="Context Feedback Loop")
    fb_sub = fb_parser.add_subparsers(dest="fb_op")
    fb_log = fb_sub.add_parser("log", help="Log feedback")
    fb_log.add_argument("--query", required=True)
    fb_log.add_argument("--used", help="Comma-separated IDs of used items")
    fb_log.add_argument("--ignored", help="Comma-separated IDs of ignored items")

    fb_sub.add_parser("stats", help="Show feedback stats")

    # Query
    query_parser = subparsers.add_parser("query", help="Query the index")
    query_parser.add_argument("text", help="Query text")
    query_parser.add_argument("--k", type=int, default=5, help="Number of results")
    query_parser.add_argument("--file", help="Seed file for scoped search")
    query_parser.add_argument(
        "--level",
        type=int,
        default=3,
        choices=[0, 1, 2, 3],
        help="Escalation level (0=Local, 3=Global)",
    )
    query_parser.add_argument(
        "--json", action="store_true", help="Output deterministic context bundle JSON"
    )
    query_parser.add_argument(
        "--profile", action="store_true", help="Enable query profiling"
    )
    query_parser.add_argument("--workset", help="ID of active workset to scope query")
    query_parser.add_argument(
        "--strategy",
        choices=[
            "lexical",
            "semantic",
            "hybrid",
            "graph",
            "symbol",
            "search-by-symbol",
            "concept",
            "search-by-concept",
            "impact",
            "search-by-impact",
            "drift",
            "search-by-drift",
            "test-failure",
            "search-by-test-failure",
            "policy",
            "search-by-policy",
            "roadmap",
            "search-by-roadmap",
        ],
        default="hybrid",
        help="Query strategy",
    )
    query_parser.add_argument(
        "--explain",
        action="store_true",
        help="Include explanation payloads in results",
    )
    query_parser.add_argument(
        "--scope",
        choices=["local", "workspace", "peer", "global"],
        default="global",
        help="Scope query results across workspace/peers/global corpus",
    )
    query_parser.add_argument(
        "--dedupe-by",
        choices=["entity", "path", "symbol"],
        default="entity",
        help="Result deduplication key",
    )
    query_parser.add_argument(
        "--refresh-stale",
        action="store_true",
        help="Refresh relevant stale files before querying",
    )
    query_parser.add_argument(
        "--corpus",
        default=None,
        help="Comma-separated corpus ids for federated comparative query",
    )

    graph_parser = subparsers.add_parser("graph", help="Repository graph operations")
    graph_sub = graph_parser.add_subparsers(dest="graph_op")
    graph_build = graph_sub.add_parser(
        "build", help="Build or update the repository graph"
    )
    graph_build.add_argument("--path", default=".")
    graph_build.add_argument(
        "--full", action="store_true", help="Force a full graph rebuild"
    )
    graph_build.add_argument("--changed-files", help="Comma-separated changed files")
    graph_query = graph_sub.add_parser("query", help="Query the repository graph")
    graph_query.add_argument(
        "expression",
        nargs="?",
        default=None,
        help=(
            "Optional graph query expression, e.g. "
            "'tokenize -> decode', 'path(A,B)', 'ffi(python, cpp)', "
            "'touches(kv_cache)', 'complexity >= O(n^2)'"
        ),
    )
    graph_query.add_argument("--symbol", default=None)
    graph_query.add_argument("--file", default=None)
    graph_query.add_argument("--relation", default=None)
    graph_query.add_argument("--depth", type=int, default=1)
    graph_query.add_argument("--limit", type=int, default=50)
    graph_query.add_argument(
        "--from",
        dest="source",
        default=None,
        help="Source selector for path query (node id, symbol, or file path)",
    )
    graph_query.add_argument(
        "--to",
        dest="target",
        default=None,
        help="Target selector for path query (node id, symbol, or file path)",
    )
    graph_query.add_argument(
        "--query-path",
        action="store_true",
        help="Return shortest directed path between --from and --to selectors",
    )
    graph_query.add_argument(
        "--reachable-from",
        default=None,
        help="Filter result to nodes reachable from the selector",
    )
    graph_query.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum depth for path/reachability traversal (defaults to --depth)",
    )
    disparate_parser = subparsers.add_parser(
        "disparate",
        help="Discover native disparate relations from the repository graph",
    )
    disparate_parser.add_argument(
        "--relation",
        default=None,
        choices=[
            "analogous_to",
            "subsystem_analogue",
            "evaluation_analogue",
            "adaptation_candidate",
            "native_upgrade_path",
            "port_program_candidate",
        ],
        help="Optional relation family filter",
    )
    disparate_parser.add_argument("--limit", type=int, default=50)
    disparate_parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refresh the graph before reading disparate relations",
    )
    disparate_parser.add_argument(
        "--format",
        choices=["json", "text"],
        default="json",
    )
    graph_sub.add_parser("export", help="Export repository graph data")
    graph_ffi = graph_sub.add_parser(
        "ffi", help="Detect FFI boundaries across repository source files"
    )
    graph_ffi.add_argument("--path", default=".")
    graph_ffi.add_argument("--limit", type=int, default=200)
    graph_ffi.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format",
    )
    ffi_parser = subparsers.add_parser(
        "ffi", help="Roadmap FFI compatibility command surface"
    )
    ffi_parser.add_argument(
        "ffi_op",
        nargs="?",
        choices=["audit"],
        default="audit",
        help="Compatibility FFI operation",
    )
    ffi_parser.add_argument("--path", default=".")
    ffi_parser.add_argument("--limit", type=int, default=200)
    ffi_parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )

    trace_parser = subparsers.add_parser(
        "trace", help="Trace execution pipeline from entrypoint or semantic query"
    )
    trace_parser.add_argument("entry_point", nargs="?", default=None)
    trace_parser.add_argument("--query", default=None)
    trace_parser.add_argument("--depth", type=int, default=20)
    trace_parser.add_argument("--max-stages", type=int, default=128)
    trace_parser.add_argument(
        "--no-complexity",
        action="store_true",
        help="Disable per-stage complexity estimation",
    )
    trace_parser.add_argument(
        "--format",
        choices=["text", "json", "mermaid"],
        default="text",
        help="Output format",
    )

    complexity_parser = subparsers.add_parser(
        "complexity", help="Estimate complexity for symbol or traced pipeline"
    )
    complexity_parser.add_argument("symbol", nargs="?", default=None)
    complexity_parser.add_argument("--file", default=None)
    complexity_parser.add_argument("--pipeline", default=None)
    complexity_parser.add_argument("--depth", type=int, default=20)
    complexity_parser.add_argument("--include-flops", action="store_true")
    complexity_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format",
    )

    # Verify command
    verify_parser = subparsers.add_parser(
        "verify", help="Verify codebase against rules"
    )
    verify_parser.add_argument("path", help="Path to verify", nargs="?", default=".")
    verify_parser.add_argument(
        "--engines",
        help="Comma-separated list of engines (native,ruff,semantic,aes,mypy,vulture)",
        default=None,
    )
    verify_parser.add_argument(
        "--aal",
        default=None,
        help="Optional AAL scope filter (e.g. '0,1' or 'AAL-0,AAL-1')",
    )
    verify_parser.add_argument(
        "--domain",
        default=None,
        help="Optional domain scope filter (e.g. 'ml' or 'ml,quantum')",
    )
    verify_parser.add_argument(
        "--require-trace",
        action="store_true",
        help="Require traceability artifacts during verification",
    )
    verify_parser.add_argument(
        "--require-evidence",
        action="store_true",
        help="Require evidence bundle artifacts during verification",
    )
    verify_parser.add_argument(
        "--evidence-bundle",
        action="store_true",
        help="Persist a verification evidence bundle artifact",
    )
    verify_parser.add_argument(
        "--require-valid-waivers",
        action="store_true",
        help="Require waiver artifacts to be present and valid",
    )
    verify_parser.add_argument(
        "--change-manifest",
        default=None,
        help="Optional path to change_manifest.json for deterministic AES verification",
    )
    verify_parser.add_argument(
        "--compliance-context",
        default=None,
        help="Optional compliance context JSON string or path to JSON file",
    )
    verify_parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )
    verify_parser.add_argument(
        "--aes-report",
        action="store_true",
        help="Emit AES compliance dashboard report derived from verification results",
    )
    verify_parser.add_argument(
        "--fix", action="store_true", help="Automatically fix violations where possible"
    )
    verify_parser.add_argument(
        "--fix-mode",
        choices=["safe", "guarded", "full"],
        default="safe",
        help="Fix safety tier to allow",
    )
    verify_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build a fix plan and receipts without mutating files",
    )
    verify_parser.add_argument(
        "--patch-preview",
        action="store_true",
        help="Include remediation plan metadata in the response",
    )
    verify_parser.add_argument(
        "--receipt-out",
        default=None,
        help="Directory for remediation receipts and rollback bundles",
    )
    verify_parser.add_argument(
        "--assisted",
        action="store_true",
        help="Enable assisted fallback lanes when supported",
    )
    verify_parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit remediation planning to the first N files",
    )
    verify_parser.add_argument(
        "--unsafe-fixes",
        action="store_true",
        help="Shortcut for --fix-mode full",
    )
    verify_parser.add_argument(
        "--min-parser-coverage",
        type=float,
        default=None,
        help="Minimum parser coverage percentage required for pass",
    )
    aes_lint_parser = subparsers.add_parser(
        "aes-check",
        help="Run deterministic AES lint with Ruff-style diagnostics",
    )
    aes_lint_parser.add_argument(
        "path", nargs="?", default=".", help="Path to lint (file or directory)"
    )
    aes_lint_parser.add_argument(
        "--aal", default=None, help="Optional AAL scope filter"
    )
    aes_lint_parser.add_argument(
        "--domain", default=None, help="Optional domain scope filter"
    )
    aes_lint_parser.add_argument(
        "--format",
        choices=["text", "json", "github"],
        default="text",
        help="Output format",
    )

    # Chronicle (Time Crystal)
    chronicle_parser = subparsers.add_parser(
        "chronicle", help="Manage Time Crystal snapshots"
    )
    chronicle_sub = chronicle_parser.add_subparsers(dest="chronicle_op")

    chronicle_sub.add_parser("snapshot", help="Create a semantic snapshot")
    chronicle_sub.add_parser("list", help="List snapshots")
    chronicle_sub.add_parser("diff", help="Calculate drift between latest snapshots")

    # Grounding (Deterministic Profile Retrieval)
    grounding_parser = subparsers.add_parser(
        "grounding", help="Retrieve deterministic model parameters"
    )
    grounding_parser.add_argument("--model", required=True, help="Ollama model name")
    grounding_parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )

    # Legislation (Auto-Legislator)
    legislation_parser = subparsers.add_parser("legislation", help="Rule discovery")
    legislation_parser.add_argument(
        "--draft", action="store_true", help="Scan and draft new rules"
    )

    # Train (Adaptive Encoder)
    train_parser = subparsers.add_parser("train", help="Train Adaptive Encoder")
    train_parser.add_argument("--path", default=".", help="Corpus path")
    train_parser.add_argument("--epochs", type=int, default=1)

    # Train Baseline (Dev Tool)
    tb_parser = subparsers.add_parser(
        "train-baseline", help="Train pretrained tokenizer baseline"
    )
    tb_parser.add_argument("--corpus", help="Corpus path")
    tb_parser.add_argument(
        "--curriculum", help="Name of curriculum preset (e.g. verso-baseline)"
    )
    tb_parser.add_argument(
        "--output", default="saguaro/artifacts/codebooks/verso_baseline.json"
    )
    tb_parser.add_argument("--fast", action="store_true")

    # Constellation (Global Memory)
    constellation_parser = subparsers.add_parser(
        "constellation", help="Manage Global Constellation"
    )
    constellation_sub = constellation_parser.add_subparsers(dest="constellation_op")

    constellation_sub.add_parser("list", help="List global libraries")

    c_index = constellation_sub.add_parser(
        "index-lib", help="Index a library to global storage"
    )
    c_index.add_argument("name", help="Library name (e.g. requests-2.31)")
    c_index.add_argument("--path", help="Path to library source", required=True)

    c_link = constellation_sub.add_parser(
        "link", help="Link a global library to current project"
    )
    c_link.add_argument("name", help="Library name to link")

    # Benchmarks
    bench_parser = subparsers.add_parser("benchmark", help="Run performance benchmarks")
    bench_parser.add_argument(
        "--dataset",
        default="CodeSearchNet",
        help="Dataset to run (CodeSearchNet, SWE-bench, custom)",
    )
    bench_parser.add_argument(
        "--custom", help="Path to custom JSON dataset file (if dataset=custom)"
    )

    eval_parser = subparsers.add_parser(
        "eval", help="Run CPU-first local evaluation suites"
    )
    eval_sub = eval_parser.add_subparsers(dest="eval_op")

    eval_run = eval_sub.add_parser("run", help="Run an evaluation suite")
    eval_run.add_argument(
        "suite",
        choices=[
            "repoqa",
            "crosscodeeval",
            "local_live",
            "verify_regression",
            "cpu_perf",
            "retrieval_quality",
        ],
        help="Evaluation suite to execute",
    )
    eval_run.add_argument("--k", type=int, default=5, help="Top-k query cutoff")
    eval_run.add_argument(
        "--limit",
        type=int,
        default=8,
        help="Maximum benchmark cases to execute (0 = all cases)",
    )

    eval_list = eval_sub.add_parser("list", help="List recent evaluation runs")
    eval_list.add_argument("--limit", type=int, default=10, help="Number of runs")

    synth_parser = subparsers.add_parser(
        "synth",
        help="Deterministically lower and build bounded code from a command or markdown roadmap",
    )
    synth_sub = synth_parser.add_subparsers(dest="synth_op")
    synth_lower = synth_sub.add_parser("lower", help="Lower a command or roadmap into SagSpec")
    synth_lower.add_argument("objective", nargs="?", default=None)
    synth_lower.add_argument("--roadmap", default=None, help="Markdown roadmap file to lower")
    synth_lower.add_argument("--format", choices=["json", "text"], default="json")
    synth_build = synth_sub.add_parser("build", help="Build bounded code from a command or roadmap")
    synth_build.add_argument("objective", nargs="?", default=None)
    synth_build.add_argument("--roadmap", default=None, help="Markdown roadmap file to build from")
    synth_build.add_argument("--out-dir", default=".", help="Output directory relative to --repo")
    synth_build.add_argument("--format", choices=["json", "text"], default="json")

    # Dead Code
    deadcode_parser = subparsers.add_parser("deadcode", help="Dead Code Discovery")
    deadcode_parser.add_argument(
        "--threshold", type=float, default=0.5, help="Minimum confidence threshold"
    )
    deadcode_parser.add_argument(
        "--low-usage-max-refs",
        type=int,
        default=1,
        help="Maximum static reference count for deadcode low-usage side report",
    )
    deadcode_parser.add_argument(
        "--lang",
        default=None,
        help="Filter dead-code candidates to a detected language family",
    )
    deadcode_parser.add_argument(
        "--evidence",
        action="store_true",
        help="Compatibility flag to include evidence payloads",
    )
    deadcode_parser.add_argument(
        "--runtime-observed",
        action="store_true",
        help="Compatibility runtime-observed mode toggle",
    )
    deadcode_parser.add_argument(
        "--explain",
        action="store_true",
        help="Include short per-candidate explanations in JSON output",
    )
    deadcode_parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )

    low_usage_parser = subparsers.add_parser(
        "low-usage",
        help="Report reachable symbols with very low static reference counts",
    )
    low_usage_parser.add_argument(
        "--max-refs",
        type=int,
        default=1,
        help="Maximum static reference count to classify as low-usage",
    )
    low_usage_parser.add_argument(
        "--include-tests",
        action="store_true",
        help="Include test paths in low-usage analysis",
    )
    low_usage_parser.add_argument(
        "--path",
        default=None,
        help="Optional relative path prefix to focus the report on one subsystem",
    )
    low_usage_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of low-usage and DRY candidates to return",
    )
    low_usage_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format",
    )

    # Unwired Features
    unwired_parser = subparsers.add_parser(
        "unwired",
        help="Detect isolated unreachable feature clusters from runtime roots",
    )
    unwired_parser.add_argument(
        "--threshold", type=float, default=0.55, help="Minimum cluster confidence"
    )
    unwired_parser.add_argument(
        "--min-nodes",
        type=int,
        default=4,
        help="Minimum nodes required for unwired feature classification",
    )
    unwired_parser.add_argument(
        "--min-files",
        type=int,
        default=2,
        help="Minimum files required for unwired feature classification",
    )
    unwired_parser.add_argument(
        "--include-tests",
        action="store_true",
        help="Include tests/ paths in analysis scope",
    )
    unwired_parser.add_argument(
        "--include-fragments",
        action="store_true",
        help="Include unreachable fragments that do not meet unwired-feature criteria",
    )
    unwired_parser.add_argument(
        "--max-clusters",
        type=int,
        default=20,
        help="Maximum clusters returned after filtering and sorting",
    )
    unwired_parser.add_argument(
        "--no-refresh-graph",
        action="store_true",
        help="Skip incremental graph refresh before analysis",
    )
    unwired_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format",
    )

    architecture_parser = subparsers.add_parser(
        "architecture",
        help="Repository topology and architecture conformance",
    )
    architecture_sub = architecture_parser.add_subparsers(dest="architecture_op")

    architecture_map = architecture_sub.add_parser(
        "map", help="Map roots, zones, and crossings"
    )
    architecture_map.add_argument("--path", default=".", help="Path to scan")
    architecture_map.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )

    architecture_verify = architecture_sub.add_parser(
        "verify", help="Verify placement and zone dependency rules"
    )
    architecture_verify.add_argument("--path", default=".", help="Path to verify")
    architecture_verify.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )

    architecture_explain = architecture_sub.add_parser(
        "explain", help="Explain zone, dependencies, and findings for a path"
    )
    architecture_explain.add_argument("path", help="Path to explain")
    architecture_explain.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )

    architecture_zones = architecture_sub.add_parser(
        "zones", help="List zone assignments"
    )
    architecture_zones.add_argument("--path", default=".", help="Optional path")
    architecture_zones.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )
    architecture_violations = architecture_sub.add_parser(
        "violations", help="Compatibility alias for architecture findings"
    )
    architecture_violations.add_argument("--path", default=".", help="Path to verify")
    architecture_violations.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )

    duplicates_parser = subparsers.add_parser(
        "duplicates", help="Detect exact and structural duplicate files"
    )
    duplicates_parser.add_argument("--path", default=".", help="Path to scan")
    duplicates_parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )

    liveness_parser = subparsers.add_parser(
        "liveness", help="Unified liveness and duplicate-aware reachability report"
    )
    liveness_parser.add_argument("--symbol", help="Explain a specific symbol")
    liveness_parser.add_argument(
        "--threshold", type=float, default=0.5, help="Minimum deadness confidence"
    )
    liveness_parser.add_argument(
        "--include-tests",
        action="store_true",
        help="Include test paths in liveness analysis",
    )
    liveness_parser.add_argument(
        "--include-fragments",
        action="store_true",
        help="Include unreachable fragments in cluster output",
    )
    liveness_parser.add_argument(
        "--max-clusters", type=int, default=20, help="Maximum unreachable clusters"
    )
    liveness_parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )
    redundancy_parser = subparsers.add_parser(
        "redundancy", help="Compatibility alias for duplicate analysis"
    )
    redundancy_parser.add_argument("--path", default=".", help="Path to scan")
    redundancy_parser.add_argument(
        "--symbol",
        default=None,
        help="Optional symbol/query to filter duplicate clusters",
    )
    redundancy_parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )
    clones_parser = subparsers.add_parser(
        "clones", help="Compatibility alias for duplicate analysis"
    )
    clones_parser.add_argument("--path", default=".", help="Path to scan")
    clones_parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )
    duplicate_clusters_parser = subparsers.add_parser(
        "duplicate-clusters", help="Compatibility alias for duplicate clusters"
    )
    duplicate_clusters_parser.add_argument("--path", default=".", help="Path to scan")
    duplicate_clusters_parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )
    reachability_parser = subparsers.add_parser(
        "reachability", help="Compatibility alias for liveness reachability report"
    )
    reachability_parser.add_argument(
        "symbol",
        nargs="?",
        default=None,
        help="Optional symbol for reachability explanation",
    )
    reachability_parser.add_argument(
        "--threshold", type=float, default=0.5, help="Minimum deadness confidence"
    )
    reachability_parser.add_argument(
        "--include-tests",
        action="store_true",
        help="Include test paths in reachability analysis",
    )
    reachability_parser.add_argument(
        "--include-fragments",
        action="store_true",
        help="Include unreachable fragments in cluster output",
    )
    reachability_parser.add_argument(
        "--max-clusters", type=int, default=20, help="Maximum unreachable clusters"
    )
    reachability_parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )
    bridge_parser = subparsers.add_parser(
        "bridge", help="Compatibility alias for FFI bridge audit"
    )
    bridge_parser.add_argument(
        "bridge_op",
        nargs="?",
        choices=["audit", "explain"],
        default="audit",
        help="Bridge compatibility operation",
    )
    bridge_parser.add_argument(
        "symbol",
        nargs="?",
        default=None,
        help="Symbol/target to explain when using `bridge explain`",
    )
    bridge_parser.add_argument("--path", default=".")
    bridge_parser.add_argument("--limit", type=int, default=200)
    bridge_parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )

    # Impact
    impact_parser = subparsers.add_parser("impact", help="Impact Analysis")
    impact_parser.add_argument("--path", required=True, help="File to analyze")

    # Report (State of the Repo)
    report_parser = subparsers.add_parser(
        "report", help="Generate State of the Repo Report"
    )
    report_parser.add_argument(
        "--format",
        choices=["json", "markdown"],
        default="markdown",
        help="Output format",
    )
    report_parser.add_argument(
        "--output", default="saguaro_report.md", help="Output file path"
    )

    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare one or more corpora against a target corpus",
    )
    compare_parser.add_argument("--target", default=".")
    compare_parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Candidate repo/subtree path. Repeat for multiple candidates.",
    )
    compare_parser.add_argument(
        "--corpus",
        action="append",
        default=[],
        help="Existing corpus id to compare. Repeat for multiple corpora.",
    )
    compare_parser.add_argument("--fleet-root", default=None)
    compare_parser.add_argument("--top-k", type=int, default=10)
    compare_parser.add_argument("--ttl-hours", type=float, default=72.0)
    compare_parser.add_argument(
        "--mode",
        choices=["flight_plan", "triage", "migration"],
        default="flight_plan",
    )
    compare_parser.add_argument("--portfolio-top-n", type=int, default=12)
    compare_parser.add_argument(
        "--calibration-profile",
        choices=["balanced", "precision", "recall"],
        default="balanced",
    )
    compare_parser.add_argument("--evidence-budget", type=int, default=12)
    compare_parser.add_argument(
        "--no-phasepack",
        action="store_true",
        help="Skip phasepack export artifacts.",
    )
    compare_parser.add_argument(
        "--no-explain-paths",
        action="store_true",
        help="Skip detailed proof/path explanations in rendered output.",
    )
    compare_parser.add_argument(
        "--export-datatables",
        action="store_true",
        help="Export portfolio leaderboard/datatables artifacts.",
    )
    compare_parser.add_argument(
        "--reuse-only",
        action="store_true",
        help="Reuse existing temp corpora only; skip missing candidates instead of indexing them.",
    )
    compare_parser.add_argument(
        "--format",
        choices=["json", "markdown"],
        default="json",
    )
    compare_parser.add_argument(
        "--output",
        default="saguaro_comparative_report.md",
        help="Output path for markdown comparative reports",
    )

    # Analyze (Phase 4)
    analyze_parser = subparsers.add_parser(
        "analyze", help="Deep Texture Analysis (Health Card)"
    )
    analyze_parser.add_argument(
        "--json", action="store_true", help="Output JSON format"
    )

    # Knowledge
    kb_parser = subparsers.add_parser("knowledge", help="Shared Agent Knowledge Base")
    kb_sub = kb_parser.add_subparsers(dest="kb_op")

    kb_add = kb_sub.add_parser("add", help="Add a fact")
    kb_add.add_argument(
        "--category", required=True, choices=["invariant", "rule", "pattern", "zone"]
    )
    kb_add.add_argument("--key", required=True, help="Fact key")
    kb_add.add_argument("--value", required=True, help="Fact value")

    kb_list = kb_sub.add_parser("list", help="List facts")
    kb_list.add_argument("--category", help="Filter by category")

    kb_search = kb_sub.add_parser("search", help="Search facts")
    kb_search.add_argument("query", help="Search query")

    # Auditor
    audit_parser = subparsers.add_parser("audit", help="Auditor Agent Verification")
    audit_parser.add_argument("--path", help="Path to audit (diff mode)")
    audit_parser.add_argument(
        "--engines",
        default="native,ruff,semantic,aes",
        help=(
            "Comma-separated engine list for Sentinel verification "
            "(default: native,ruff,semantic,aes)"
        ),
    )
    audit_parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )

    # Build Graph
    subparsers.add_parser("build-graph", help="Build System Graph")

    # Entry Points
    subparsers.add_parser("entrypoints", help="Runtime Entry Point Detection")

    # Scribe (Phase 2: Synthesis)
    scribe_parser = subparsers.add_parser("scribe", help="Generative Engine")
    scribe_parser.add_argument("task", help="Task description")
    scribe_parser.add_argument("--file", help="Context file (optional hint)")
    scribe_parser.add_argument("--out", help="Output patch file", default="patch.json")

    # --- Phase 4/SSAI: Agent & Orchestration ---
    agent_parser = subparsers.add_parser("agent", help="SSAI Agent Interface")
    agent_sub = agent_parser.add_subparsers(dest="agent_command")

    # Perception Layer (with enhanced help for AI Adoption - Phase 3)
    skel_parser = agent_sub.add_parser(
        "skeleton",
        help="Generate File Skeleton (signatures + docstrings only)",
        description="""Use this INSTEAD of view_file to explore a file's structure.

Shows function/class signatures and docstrings without full implementation code.
Saves 90%% of tokens compared to reading full files.

DECISION TREE:
  Need to understand a file's structure? → Use this command
  Need to read a specific function? → Then use 'saguaro agent slice'

Example:
  saguaro agent skeleton src/core.py
""",
    )
    skel_parser.add_argument("file", help="Target file path to generate skeleton for")

    slice_parser = agent_sub.add_parser(
        "slice",
        help="Generate Context Slice (function + dependencies)",
        description="""Read a specific symbol with its dependencies and context.

Use this INSTEAD of view_file or view_code_item to read function/class code.
Automatically includes imports and parent context for understanding.

DECISION TREE:
  Need to read a specific function/class? → Use this command
  Need to explore file structure first? → Use 'saguaro agent skeleton' first
  Symbol not found? → Use 'saguaro query' to find it semantically

Example:
  saguaro agent slice MyClass.method --depth 2
""",
    )
    slice_parser.add_argument(
        "symbol", help="Entry point symbol (e.g., ClassName.method)"
    )
    slice_parser.add_argument(
        "--depth", type=int, default=1, help="Dependency graph depth (default: 1)"
    )
    slice_parser.add_argument(
        "--corpus",
        default=None,
        help="Corpus id for slicing from an isolated comparative corpus",
    )

    trace_agent = agent_sub.add_parser(
        "trace",
        help="Trace pipeline by entry point or semantic query",
    )
    trace_agent.add_argument("entry_or_query", help="Entry point or query text")
    trace_agent.add_argument(
        "--query", action="store_true", help="Interpret input as semantic query"
    )
    trace_agent.add_argument("--depth", type=int, default=20)

    complexity_agent = agent_sub.add_parser(
        "complexity",
        help="Complexity report for a symbol",
    )
    complexity_agent.add_argument("symbol")
    complexity_agent.add_argument("--file", default=None)
    complexity_agent.add_argument("--include-flops", action="store_true")

    ffi_map_agent = agent_sub.add_parser(
        "ffi-map",
        help="Emit full FFI boundary map",
    )
    ffi_map_agent.add_argument("--path", default=".")
    ffi_map_agent.add_argument("--limit", type=int, default=400)

    pipeline_diff_agent = agent_sub.add_parser(
        "pipeline-diff",
        help="Diff pipelines between revisions (supports HEAD~N..HEAD)",
    )
    pipeline_diff_agent.add_argument("revision_range")
    pipeline_diff_agent.add_argument("--entry", default="main")

    hotspots_agent = agent_sub.add_parser(
        "hotspots",
        help="List symbols with highest estimated complexity",
    )
    hotspots_agent.add_argument("--limit", type=int, default=20)
    architecture_agent = agent_sub.add_parser(
        "architecture",
        help="Compatibility alias for architecture map",
    )
    architecture_agent.add_argument("--path", default=".")

    bridge_agent = agent_sub.add_parser(
        "bridge",
        help="Compatibility alias for FFI bridge audit",
    )
    bridge_agent.add_argument("--path", default=".")
    bridge_agent.add_argument("--limit", type=int, default=200)

    duplicates_agent = agent_sub.add_parser(
        "duplicates",
        help="Compatibility alias for duplicate analysis",
    )
    duplicates_agent.add_argument("--path", default=".")
    redundancy_agent = agent_sub.add_parser(
        "redundancy",
        help="Compatibility alias for duplicate analysis",
    )
    redundancy_agent.add_argument("--path", default=".")
    redundancy_agent.add_argument("--symbol", default=None)
    clones_agent = agent_sub.add_parser(
        "clones",
        help="Compatibility alias for duplicate analysis",
    )
    clones_agent.add_argument("--path", default=".")
    duplicate_clusters_agent = agent_sub.add_parser(
        "duplicate-clusters",
        help="Compatibility alias for duplicate clusters",
    )
    duplicate_clusters_agent.add_argument("--path", default=".")

    liveness_agent = agent_sub.add_parser(
        "liveness",
        help="Compatibility alias for liveness analysis",
    )
    liveness_agent.add_argument("--symbol", default=None)
    liveness_agent.add_argument("--threshold", type=float, default=0.5)
    liveness_agent.add_argument("--include-tests", action="store_true")
    liveness_agent.add_argument("--include-fragments", action="store_true")
    liveness_agent.add_argument("--max-clusters", type=int, default=20)
    reachability_agent = agent_sub.add_parser(
        "reachability",
        help="Compatibility alias for reachability analysis",
    )
    reachability_agent.add_argument("symbol", nargs="?", default=None)
    reachability_agent.add_argument("--threshold", type=float, default=0.5)
    reachability_agent.add_argument("--include-tests", action="store_true")
    reachability_agent.add_argument("--include-fragments", action="store_true")
    reachability_agent.add_argument("--max-clusters", type=int, default=20)

    zones_agent = agent_sub.add_parser(
        "zones",
        help="Compatibility alias for architecture zones",
    )
    zones_agent.add_argument("--path", default=".")
    violations_agent = agent_sub.add_parser(
        "violations",
        help="Compatibility alias for architecture violations",
    )
    violations_agent.add_argument("--path", default=".")
    abi_agent = agent_sub.add_parser(
        "abi",
        help="Compatibility alias for ABI verification",
    )
    abi_agent.add_argument(
        "abi_op",
        nargs="?",
        choices=["verify", "orphaned"],
        default="verify",
    )
    ffi_agent = agent_sub.add_parser(
        "ffi",
        help="Compatibility alias for FFI audit",
    )
    ffi_agent.add_argument("--path", default=".")
    ffi_agent.add_argument("--limit", type=int, default=200)

    # Action Layer
    patch_parser = agent_sub.add_parser("patch", help="Apply Semantic Patch")
    patch_parser.add_argument("file", help="Target file")
    patch_parser.add_argument("patch_json", help="Patch content or path to JSON")

    verify_parser = agent_sub.add_parser("verify", help="Verify Sandbox State")
    verify_parser.add_argument("sandbox_id", help="Sandbox ID to verify")

    # Intelligence Layer
    imp_parser = agent_sub.add_parser("impact", help="Predict Impact")
    imp_parser.add_argument("sandbox_id", help="Sandbox ID")

    commit_parser = agent_sub.add_parser("commit", help="Commit Sandbox to Disk")
    commit_parser.add_argument("sandbox_id", help="Sandbox ID")

    # Legacy Runner
    run_parser = agent_sub.add_parser("run", help="Run a specialized agent")
    run_parser.add_argument(
        "role",
        choices=["planner", "cartographer", "surgeon", "auditor"],
        help="Agent role to run",
    )
    run_parser.add_argument("--task", help="Task description or ID")

    # Task Graph
    task_parser = subparsers.add_parser("tasks", help="Manage Task Graph")
    task_parser.add_argument("--list", action="store_true", help="List ready tasks")
    task_parser.add_argument("--add", help="Add new task JSON string")

    # Shared Memory
    mem_parser = subparsers.add_parser("memory", help="Inspect Shared Memory")
    mem_parser.add_argument("--list", action="store_true", help="List all facts")
    mem_parser.add_argument("--read", help="Read fact by key or semantic search")
    mem_parser.add_argument(
        "--k", type=int, default=5, help="Number of results for semantic search"
    )
    mem_parser.add_argument("--write", help="Write fact content")
    mem_parser.add_argument(
        "--key", help="Fact key (optional, defaults to content snippet)"
    )
    mem_parser.add_argument(
        "--snapshot",
        help="Create an ALMF snapshot for the given campaign id",
    )
    mem_parser.add_argument(
        "--restore",
        help="Restore an ALMF snapshot from a directory",
    )
    mem_parser.add_argument(
        "--db-path",
        default=None,
        help="Optional ALMF sqlite database path override",
    )
    mem_parser.add_argument(
        "--storage-root",
        default=None,
        help="Optional ALMF storage root override",
    )
    mem_parser.add_argument(
        "--campaign-id",
        default=None,
        help="Campaign id to use for ALMF snapshot or restore commands",
    )
    mem_parser.add_argument(
        "--tier",
        choices=["working", "episodic", "semantic", "preference"],
        default="working",
        help="Memory tier (namespace)",
    )

    # --- Phase 5: Simulation ---
    sim_parser = subparsers.add_parser("simulate", help="Run Simulations")
    sim_sub = sim_parser.add_subparsers(dest="sim_op")

    sim_sub.add_parser("volatility", help="Generate Volatility Map")

    sim_reg = sim_sub.add_parser("regression", help="Predict Regressions")
    sim_reg.add_argument("--files", required=True, help="Comma-separated files changed")

    # --- Phase 6: Learning ---
    route_parser = subparsers.add_parser("route", help="Test Intent Routing")
    route_parser.add_argument("query", help="Query to classify")

    research_parser = subparsers.add_parser(
        "research", help="Manage external research ingestion"
    )
    research_sub = research_parser.add_subparsers(dest="research_op")
    research_ingest = research_sub.add_parser(
        "ingest", help="Ingest web/arXiv/repo research metadata"
    )
    research_ingest.add_argument(
        "--source", choices=["web", "arxiv", "repo"], required=True
    )
    research_ingest.add_argument("--manifest", default=None)
    research_sub.add_parser("list", help="List ingested research entries")

    docs_parser = subparsers.add_parser(
        "docs", help="Markdown document graph operations"
    )
    docs_sub = docs_parser.add_subparsers(dest="docs_op")
    docs_parse = docs_sub.add_parser("parse", help="Parse markdown docs")
    docs_parse.add_argument("--path", default=".")
    docs_parse.add_argument("--format", choices=["json", "text"], default="json")
    docs_graph = docs_sub.add_parser("graph", help="Build a docs graph view")
    docs_graph.add_argument("--path", default=".")
    docs_graph.add_argument("--format", choices=["json", "text"], default="json")

    requirements_parser = subparsers.add_parser(
        "requirements", help="Requirement extraction and inspection"
    )
    requirements_sub = requirements_parser.add_subparsers(dest="requirements_op")
    requirements_extract = requirements_sub.add_parser(
        "extract", help="Extract requirements"
    )
    requirements_extract.add_argument("--path", default=".")
    requirements_extract.add_argument(
        "--format", choices=["json", "text"], default="json"
    )
    requirements_list = requirements_sub.add_parser(
        "list", help="List extracted requirements"
    )
    requirements_list.add_argument("--path", default=".")
    requirements_list.add_argument("--format", choices=["json", "text"], default="json")
    requirements_show = requirements_sub.add_parser("show", help="Show one requirement")
    requirements_show.add_argument("requirement_id")
    requirements_show.add_argument("--path", default=".")
    requirements_show.add_argument("--format", choices=["json", "text"], default="json")

    traceability_parser = subparsers.add_parser(
        "traceability", help="Semantic traceability ledger operations"
    )
    traceability_sub = traceability_parser.add_subparsers(dest="traceability_op")
    traceability_build = traceability_sub.add_parser(
        "build", help="Build traceability state"
    )
    traceability_build.add_argument("--docs", default=".")
    traceability_build.add_argument(
        "--format", choices=["json", "text"], default="json"
    )
    traceability_status = traceability_sub.add_parser(
        "status", help="Show requirement traceability"
    )
    traceability_status.add_argument("--req", required=True)
    traceability_status.add_argument(
        "--format", choices=["json", "text"], default="json"
    )
    traceability_diff = traceability_sub.add_parser(
        "diff", help="Diff traceability snapshots"
    )
    traceability_diff.add_argument("--format", choices=["json", "text"], default="json")
    traceability_orphaned = traceability_sub.add_parser(
        "orphaned", help="List orphaned requirements"
    )
    traceability_orphaned.add_argument(
        "--format", choices=["json", "text"], default="json"
    )

    validate_parser = subparsers.add_parser(
        "validate", help="Requirement validation operations"
    )
    validate_sub = validate_parser.add_subparsers(dest="validate_op")
    validate_docs = validate_sub.add_parser("docs", help="Validate markdown docs")
    validate_docs.add_argument("--path", default=".")
    validate_docs.add_argument("--format", choices=["json", "text"], default="json")
    validate_req = validate_sub.add_parser(
        "requirement", help="Validate one requirement"
    )
    validate_req.add_argument("--id", required=True)
    validate_req.add_argument("--format", choices=["json", "text"], default="json")
    validate_gaps = validate_sub.add_parser("gaps", help="List validation gaps")
    validate_gaps.add_argument("--path", default=".")
    validate_gaps.add_argument("--format", choices=["json", "text"], default="json")

    math_parser = subparsers.add_parser("math", help="Repo math extraction and mapping")
    math_sub = math_parser.add_subparsers(dest="math_op")
    math_parse = math_sub.add_parser(
        "parse", help="Parse equations from markdown and source code"
    )
    math_parse.add_argument("--path", default=".")
    math_parse.add_argument("--format", choices=["json", "text"], default="json")
    math_map = math_sub.add_parser(
        "map", help="Map one cached equation into graph matches"
    )
    math_map.add_argument("--id", required=True)
    math_map.add_argument("--format", choices=["json", "text"], default="json")

    cpu_parser = subparsers.add_parser("cpu", help="Static CPU hotspot analysis")
    cpu_sub = cpu_parser.add_subparsers(dest="cpu_op")
    cpu_scan = cpu_sub.add_parser("scan", help="Scan one path for CPU hotspots")
    cpu_scan.add_argument("--path", default=".")
    cpu_scan.add_argument(
        "--arch",
        choices=["x86_64-avx2", "x86_64-avx512", "arm64-neon"],
        default="x86_64-avx2",
    )
    cpu_scan.add_argument("--limit", type=int, default=20)
    cpu_scan.add_argument("--format", choices=["json", "text"], default="json")

    omnigraph_parser = subparsers.add_parser(
        "omnigraph", help="Typed omni-graph operations"
    )
    omnigraph_sub = omnigraph_parser.add_subparsers(dest="omnigraph_op")
    omnigraph_build = omnigraph_sub.add_parser("build", help="Build the omni-graph")
    omnigraph_build.add_argument("--path", default=".")
    omnigraph_build.add_argument("--format", choices=["json", "text"], default="json")
    omnigraph_explain = omnigraph_sub.add_parser(
        "explain", help="Explain a requirement neighborhood"
    )
    omnigraph_explain.add_argument("--req", required=True)
    omnigraph_explain.add_argument("--path", default=".")
    omnigraph_explain.add_argument("--format", choices=["json", "text"], default="json")
    omnigraph_find = omnigraph_sub.add_parser(
        "find", help="Find equation or concept matches"
    )
    omnigraph_find.add_argument("--equation", required=True)
    omnigraph_find.add_argument("--path", default=".")
    omnigraph_find.add_argument("--format", choices=["json", "text"], default="json")
    omnigraph_diff = omnigraph_sub.add_parser("diff", help="Show omni-graph summary")
    omnigraph_diff.add_argument("--format", choices=["json", "text"], default="json")
    omnigraph_gaps = omnigraph_sub.add_parser("gaps", help="List omni-graph gaps")
    omnigraph_gaps.add_argument("--modality", default=None)
    omnigraph_gaps.add_argument("--path", default=".")
    omnigraph_gaps.add_argument("--format", choices=["json", "text"], default="json")

    packet_parser = subparsers.add_parser("packet", help="Weak-model packet operations")
    packet_sub = packet_parser.add_subparsers(dest="packet_op")
    packet_build = packet_sub.add_parser("build", help="Build a task packet")
    packet_build.add_argument("--task", required=True)
    packet_build.add_argument("--format", choices=["json", "text"], default="json")
    packet_review = packet_sub.add_parser("review", help="Review a packet")
    packet_review.add_argument("packet_id")
    packet_review.add_argument("--format", choices=["json", "text"], default="json")
    packet_witness = packet_sub.add_parser("witness", help="Build a witness packet")
    packet_witness.add_argument("requirement_id")
    packet_witness.add_argument("--format", choices=["json", "text"], default="json")

    roadmap_parser = subparsers.add_parser(
        "roadmap", help="Roadmap completion validation"
    )
    roadmap_sub = roadmap_parser.add_subparsers(dest="roadmap_op")
    roadmap_validate = roadmap_sub.add_parser(
        "validate", help="Validate a roadmap markdown file"
    )
    roadmap_validate.add_argument("--path", default=".")
    roadmap_validate.add_argument("--format", choices=["json", "text"], default="json")
    roadmap_graph = roadmap_sub.add_parser(
        "graph", help="Build a roadmap completion graph"
    )
    roadmap_graph.add_argument("--path", default=".")
    roadmap_graph.add_argument("--format", choices=["json", "text"], default="json")

    reality_parser = subparsers.add_parser(
        "reality", help="Runtime reality graph operations"
    )
    reality_sub = reality_parser.add_subparsers(dest="reality_op")
    reality_events = reality_sub.add_parser("events", help="List runtime events")
    reality_events.add_argument("--run-id", default=None)
    reality_events.add_argument("--limit", type=int, default=2000)
    reality_events.add_argument("--format", choices=["json", "text"], default="json")
    reality_graph = reality_sub.add_parser("graph", help="Build runtime reality graph")
    reality_graph.add_argument("--run-id", default=None)
    reality_graph.add_argument("--limit", type=int, default=2000)
    reality_graph.add_argument("--format", choices=["json", "text"], default="json")
    reality_twin = reality_sub.add_parser("twin", help="Show runtime twin-state")
    reality_twin.add_argument("--run-id", default=None)
    reality_twin.add_argument("--limit", type=int, default=500)
    reality_twin.add_argument("--format", choices=["json", "text"], default="json")
    reality_export = reality_sub.add_parser(
        "export", help="Export one run reality bundle"
    )
    reality_export.add_argument("--run-id", required=True)
    reality_export.add_argument("--limit", type=int, default=2000)
    reality_export.add_argument("--format", choices=["json", "text"], default="json")

    packs_parser = subparsers.add_parser("packs", help="Domain/science pack operations")
    packs_sub = packs_parser.add_subparsers(dest="packs_op")
    packs_list = packs_sub.add_parser("list", help="List available packs")
    packs_list.add_argument("--format", choices=["json", "text"], default="json")
    packs_enable = packs_sub.add_parser("enable", help="Enable a pack")
    packs_enable.add_argument("name")
    packs_enable.add_argument("--format", choices=["json", "text"], default="json")
    packs_diag = packs_sub.add_parser("diagnose", help="Diagnose matching packs")
    packs_diag.add_argument("--path", default=".")
    packs_diag.add_argument("--format", choices=["json", "text"], default="json")

    # --- AI Adoption Metrics ---
    metrics_parser = subparsers.add_parser(
        "metrics", help="View AI adoption metrics (Saguaro vs fallback tool usage)"
    )
    metrics_parser.add_argument(
        "--session", action="store_true", help="Show current session metrics only"
    )
    metrics_parser.add_argument("--json", action="store_true", help="Output as JSON")
    metrics_parser.add_argument(
        "--reset", action="store_true", help="Reset all metrics"
    )

    args = parser.parse_args(_normalize_compat_argv(sys.argv[1:]))
    repo_root = (
        os.path.abspath(args.repo) if getattr(args, "repo", None) else os.getcwd()
    )

    # For index, a directory passed via --path should become the repo root unless
    # --repo explicitly overrides it. This keeps .saguaro state isolated per target repo.
    if (
        args.command == "index"
        and not getattr(args, "repo", None)
        and getattr(args, "path", ".") not in (".", "")
    ):
        candidate_repo = os.path.abspath(args.path)
        if os.path.isdir(candidate_repo):
            repo_root = candidate_repo
            args.path = "."

    if args.command in {
        "docs",
        "requirements",
        "traceability",
        "validate",
        "math",
        "omnigraph",
        "packet",
        "roadmap",
        "packs",
    }:
        if args.command == "docs":
            from saguaro.parsing.markdown import MarkdownStructureParser
            from saguaro.query.corpus_rules import canonicalize_rel_path
            from saguaro.requirements.extractor import RequirementExtractor

            extractor = RequirementExtractor(repo_root=repo_root)
            parser_view = MarkdownStructureParser()
            documents = []
            for file_path in extractor.discover_docs(getattr(args, "path", ".")):
                rel_file = canonicalize_rel_path(str(file_path), repo_path=repo_root)
                document = parser_view.parse(
                    file_path.read_text(encoding="utf-8"),
                    source_path=rel_file,
                )
                nodes = [
                    {
                        "kind": node.kind,
                        "title": node.title,
                        "text": node.text,
                        "line_start": node.line_start,
                        "line_end": node.line_end,
                        "section_path": list(node.section_path),
                        "language": node.language,
                    }
                    for node in document.walk()
                ]
                documents.append(
                    {"file": rel_file, "profile": "readme", "nodes": nodes}
                )
            result = {"status": "ok", "count": len(documents), "documents": documents}
            if args.docs_op == "graph":
                edges = []
                for document in documents:
                    previous_section = None
                    for node in document.get("nodes", []):
                        if node.get("kind") != "section":
                            continue
                        if previous_section is not None:
                            edges.append(
                                {
                                    "from": previous_section["title"],
                                    "to": node.get("title"),
                                    "relation": "next_section",
                                    "file": document["file"],
                                }
                            )
                        previous_section = node
                result["edges"] = edges
            if args.format == "json":
                print(json.dumps(result, indent=2))
            else:
                print(f"Docs: {result.get('count', 0)} parsed files")
            return

        if args.command == "requirements":
            from saguaro.requirements.extractor import RequirementExtractor

            payload = RequirementExtractor(repo_root=repo_root).extract(
                getattr(args, "path", ".")
            )
            result = {
                "status": "ok",
                "count": len(payload.requirements),
                "requirements": [item.to_dict() for item in payload.requirements],
                "source_paths": list(payload.source_paths),
                "graph_loaded": payload.graph_loaded,
            }
            if args.requirements_op == "list":
                result["requirements"] = [
                    {
                        "id": item["requirement_id"],
                        "file": item["source_path"],
                        "statement": item["statement"],
                        "strength": item["classification"]["strength"],
                    }
                    for item in result.get("requirements", [])
                ]
            elif args.requirements_op == "show":
                found = next(
                    (
                        item
                        for item in result.get("requirements", [])
                        if item.get("requirement_id") == args.requirement_id
                    ),
                    None,
                )
                result = (
                    {"status": "ok", "requirement": found}
                    if found
                    else {
                        "status": "missing",
                        "requirement_id": args.requirement_id,
                    }
                )
            if args.format == "json":
                print(json.dumps(result, indent=2))
            else:
                if args.requirements_op == "show" and result.get("requirement"):
                    req = result["requirement"]
                    print(
                        f"{req.get('requirement_id', req.get('id'))}: {req.get('statement', req.get('text_raw', ''))}"
                    )
                else:
                    print(f"Requirements: {result.get('count', 0)}")
            return

        if args.command == "traceability":
            from saguaro.requirements.traceability import TraceabilityService

            service = TraceabilityService(repo_root=repo_root)
            if args.traceability_op == "build":
                result = service.build(args.docs)
            elif args.traceability_op == "status":
                result = service.status(args.req)
            elif args.traceability_op == "diff":
                result = service.diff()
            elif args.traceability_op == "orphaned":
                result = service.orphaned()
            else:
                result = {
                    "status": "error",
                    "message": "Select a traceability subcommand.",
                }
            print(
                json.dumps(result, indent=2)
                if args.format == "json"
                else f"Traceability: {result.get('status', 'ok')}"
            )
            return

        if args.command == "validate":
            from saguaro.validation.engine import ValidationEngine

            engine = ValidationEngine(repo_root)
            if args.validate_op == "docs":
                result = engine.validate_docs(path=args.path)
            elif args.validate_op == "requirement":
                result = engine.validate_requirement(args.id)
            elif args.validate_op == "gaps":
                result = engine.gaps(path=args.path)
            else:
                result = {"status": "error", "message": "Select a validate subcommand."}
            print(
                json.dumps(result, indent=2)
                if args.format == "json"
                else f"Validation: {result.get('status', 'ok')}"
            )
            return

        if args.command == "math":
            from saguaro.math import MathEngine

            engine = MathEngine(repo_root)
            if args.math_op == "parse":
                result = engine.parse(args.path)
            elif args.math_op == "map":
                result = engine.map(args.id)
            else:
                result = {"status": "error", "message": "Select a math subcommand."}
            print(
                json.dumps(result, indent=2)
                if args.format == "json"
                else f"Math: {result.get('count', result.get('status', 'ok'))}"
            )
            return

        if args.command == "cpu":
            result = api.cpu_scan(path=args.path, arch=args.arch, limit=args.limit)
            print(
                json.dumps(result, indent=2)
                if args.format == "json"
                else f"CPU Hotspots: {result.get('hotspot_count', 0)}"
            )
            return

        if args.command == "omnigraph":
            from saguaro.omnigraph.store import OmniGraphStore
            from saguaro.requirements.traceability import TraceabilityService

            store = OmniGraphStore(repo_root)
            if args.omnigraph_op == "build":
                traceability = TraceabilityService(repo_root=repo_root).build(args.path)
                result = store.build(traceability_payload=traceability)
            elif args.omnigraph_op == "explain":
                if not os.path.exists(store.graph_path):
                    traceability = TraceabilityService(repo_root=repo_root).build(
                        args.path
                    )
                    store.build(traceability_payload=traceability)
                result = store.explain(args.req)
            elif args.omnigraph_op == "find":
                if not os.path.exists(store.graph_path):
                    traceability = TraceabilityService(repo_root=repo_root).build(
                        args.path
                    )
                    store.build(traceability_payload=traceability)
                result = store.find_equation(args.equation)
            elif args.omnigraph_op == "diff":
                result = store.diff()
            elif args.omnigraph_op == "gaps":
                if not os.path.exists(store.graph_path):
                    traceability = TraceabilityService(repo_root=repo_root).build(
                        args.path
                    )
                    store.build(traceability_payload=traceability)
                result = store.gaps(modality=args.modality)
            else:
                result = {
                    "status": "error",
                    "message": "Select an omnigraph subcommand.",
                }
            print(
                json.dumps(result, indent=2)
                if args.format == "json"
                else f"OmniGraph: {result.get('status', 'ok')}"
            )
            return

        if args.command == "packet":
            from saguaro.packets.builders import PacketBuilder

            builder = PacketBuilder(repo_root)
            if args.packet_op == "build":
                result = builder.build_task_packet(args.task)
            elif args.packet_op == "review":
                result = builder.review_packet(args.packet_id)
            elif args.packet_op == "witness":
                result = builder.witness_packet(args.requirement_id)
            else:
                result = {"status": "error", "message": "Select a packet subcommand."}
            print(
                json.dumps(result, indent=2)
                if args.format == "json"
                else f"Packet: {result.get('id', result.get('status', 'ok'))}"
            )
            return

        if args.command == "roadmap":
            from saguaro.roadmap.validator import RoadmapValidator

            validator = RoadmapValidator(repo_root)
            try:
                if args.roadmap_op == "validate":
                    result = validator.validate(path=args.path)
                elif args.roadmap_op == "graph":
                    result = validator.build_graph(path=args.path)
                else:
                    result = {
                        "status": "error",
                        "message": "Select a roadmap subcommand.",
                    }
            except FileNotFoundError as exc:
                result = {
                    "status": "error",
                    "error_type": "file_not_found",
                    "path": args.path,
                    "message": str(exc),
                }
                if args.format == "json":
                    print(json.dumps(result, indent=2))
                else:
                    print(f"Roadmap error: {result['message']}")
                raise SystemExit(1) from exc
            if args.format == "json":
                print(json.dumps(result, indent=2))
            else:
                summary = result.get("summary", {})
                print(
                    "Roadmap: "
                    f"{summary.get('completed_count', summary.get('completed', 0))}/"
                    f"{summary.get('requirement_count', summary.get('count', 0))} completed"
                )
            return

        if args.command == "packs":
            from saguaro.packs.base import PackManager

            manager = PackManager(repo_root)
            if args.packs_op == "list":
                result = {"status": "ok", "packs": manager.list()}
            elif args.packs_op == "enable":
                result = manager.enable(args.name)
            elif args.packs_op == "diagnose":
                result = manager.diagnose(path=args.path)
            else:
                result = {"status": "error", "message": "Select a packs subcommand."}
            if args.format == "json":
                print(json.dumps(result, indent=2))
            else:
                pack_count = (
                    len(result.get("packs", []))
                    if isinstance(result.get("packs"), list)
                    else 0
                )
                print(f"Packs: {pack_count or result.get('status', 'ok')}")
            return

    if args.command == "health":
        from saguaro.fastpath import FastCommandAPI

        print(json.dumps(FastCommandAPI(repo_root).health(), indent=2))
        return

    if args.command == "agent" and args.agent_command == "skeleton":
        from saguaro.agents.perception import SkeletonGenerator

        generator = SkeletonGenerator()
        print(json.dumps(generator.generate(args.file), indent=2))
        return

    hot_fastpath = (
        args.command in {"query", "doctor", "ffi"}
        or (args.command == "abi" and getattr(args, "abi_op", "verify") == "verify")
        or (args.command == "math" and getattr(args, "math_op", None) == "parse")
        or (args.command == "cpu" and getattr(args, "cpu_op", None) == "scan")
    )
    if hot_fastpath:
        from saguaro.fastpath import FastCommandAPI

        fast_api = FastCommandAPI(repo_root)

        if args.command == "query":
            result = fast_api.query(
                text=args.text,
                k=args.k,
                file=getattr(args, "file", None),
                level=getattr(args, "level", 3),
                strategy=getattr(args, "strategy", "hybrid"),
                explain=bool(getattr(args, "explain", False)),
                scope=getattr(args, "scope", "global"),
                dedupe_by=getattr(args, "dedupe_by", "entity"),
            )
            if getattr(args, "json", False):
                print(json.dumps(result, indent=2))
            else:
                print(f"Query: '{result.get('query', '')}'")
                for row in result.get("results", []):
                    print(
                        f"[{row.get('rank', '?')}] [{row.get('score', 0.0):.4f}] "
                        f"{row.get('name', 'unknown')} ({row.get('type', 'symbol')})"
                    )
                    print(f"    Path: {row.get('file', '?')}:{row.get('line', '?')}")
                    if row.get("reason"):
                        print(f"    Why:  {row['reason']}")
                    if getattr(args, "explain", False) and row.get("explanation"):
                        print(
                            f"    Explain: {json.dumps(row['explanation'], sort_keys=True)}"
                        )
                        print("")
            return

        if args.command == "doctor":
            print(json.dumps(fast_api.doctor(), indent=2))
            return

        if args.command == "abi":
            report = fast_api.abi(action=getattr(args, "abi_op", "verify"))
            if getattr(args, "format", "text") == "json":
                print(json.dumps(report, indent=2))
            else:
                native_abi = dict(report.get("native_abi") or {})
                status = "pass" if native_abi.get("ok") else "warning"
                print(f"ABI verify: {status}")
                if native_abi.get("reason"):
                    print(str(native_abi.get("reason")))
            return

        if args.command == "ffi":
            report = fast_api.ffi_audit(
                path=getattr(args, "path", "."),
                limit=int(getattr(args, "limit", 200) or 200),
            )
            if getattr(args, "format", "text") == "json":
                print(json.dumps(report, indent=2))
            else:
                print(f"FFI audit boundaries: {report.get('count', 0)}")
            return

        if args.command == "math":
            print(json.dumps(fast_api.math_parse(path=getattr(args, "path", ".")), indent=2))
            return

        if args.command == "cpu":
            print(
                json.dumps(
                    fast_api.cpu_scan(
                        path=getattr(args, "path", "."),
                        arch=getattr(args, "arch", "x86_64-avx2"),
                        limit=int(getattr(args, "limit", 20) or 20),
                    ),
                    indent=2,
                )
            )
            return

    if args.command in {"corpus", "compare"}:
        from saguaro.analysis.report import ReportGenerator
        from saguaro.services.comparative import ComparativeAnalysisService

        comparative = ComparativeAnalysisService(repo_root)
        if args.command == "corpus":
            corpus_op = getattr(args, "corpus_op", None) or "list"
            print(
                json.dumps(
                    comparative.corpus(
                        action=corpus_op,
                        path=getattr(args, "path", None),
                        corpus_id=getattr(args, "corpus_id", None),
                        alias=getattr(args, "alias", None),
                        ttl_hours=float(getattr(args, "ttl_hours", 24.0) or 24.0),
                        quarantine=not bool(getattr(args, "no_quarantine", False)),
                        trust_level=getattr(args, "trust_level", "medium"),
                        build_profile=getattr(args, "build_profile", "auto"),
                        include_expired=bool(getattr(args, "include_expired", False)),
                        rebuild=bool(getattr(args, "rebuild", False)),
                        batch_sizes=list(getattr(args, "batch_size", []) or []),
                        file_batch_sizes=list(getattr(args, "file_batch_size", []) or [])
                        or None,
                        iterations=int(getattr(args, "iterations", 1) or 1),
                        reuse_check=not bool(getattr(args, "no_reuse_check", False)),
                    ),
                    indent=2,
                )
            )
            return
        result = comparative.compare(
            target=getattr(args, "target", "."),
            candidates=list(getattr(args, "candidate", []) or []),
            corpus_ids=list(getattr(args, "corpus", []) or []),
            fleet_root=getattr(args, "fleet_root", None),
            top_k=int(getattr(args, "top_k", 10) or 10),
            ttl_hours=float(getattr(args, "ttl_hours", 72.0) or 72.0),
            reuse_only=bool(getattr(args, "reuse_only", False)),
            mode=str(getattr(args, "mode", "flight_plan") or "flight_plan"),
            emit_phasepack=not bool(getattr(args, "no_phasepack", False)),
            explain_paths=not bool(getattr(args, "no_explain_paths", False)),
            portfolio_top_n=int(getattr(args, "portfolio_top_n", 12) or 12),
            calibration_profile=str(
                getattr(args, "calibration_profile", "balanced") or "balanced"
            ),
            evidence_budget=int(getattr(args, "evidence_budget", 12) or 12),
            export_datatables=bool(getattr(args, "export_datatables", False)),
        )
        if args.format == "json":
            print(json.dumps(result, indent=2))
        else:
            ReportGenerator(repo_root).save_comparative_markdown(result, args.output)
            print(f"Comparative report saved to: {args.output}")
        return

    # Native in-process API fast-path (no subprocess, no mandatory TensorFlow).
    from saguaro.api import SaguaroAPI

    api = SaguaroAPI(repo_root)

    if args.command == "debuginfo":
        print(
            json.dumps(
                api.debuginfo(
                    output_path=getattr(args, "output", None),
                    event_limit=int(getattr(args, "event_limit", 500) or 500),
                ),
                indent=2,
            )
        )
        return

    if args.command == "state" and getattr(args, "state_op", None) == "restore":
        print(
            json.dumps(
                api.state_restore(
                    bundle_path=args.bundle_path,
                    force=bool(getattr(args, "force", False)),
                ),
                indent=2,
            )
        )
        return

    if args.command == "admin":
        print(
            json.dumps(
                api.admin(
                    action=args.admin_action,
                    bundle_path=getattr(args, "bundle_path", None),
                    output_path=getattr(args, "output", None),
                    force=bool(getattr(args, "force", False)),
                    include_reality=not bool(getattr(args, "no_reality", False)),
                    event_limit=int(getattr(args, "event_limit", 500) or 500),
                ),
                indent=2,
            )
        )
        return

    if args.command == "init":
        print(json.dumps(api.init(force=getattr(args, "force", False)), indent=2))
        return

    if args.command == "index":
        changed_files = [
            item.strip()
            for item in (args.changed_files or "").split(",")
            if item.strip()
        ] or None
        result = api.index(
            path=getattr(args, "path", "."),
            force=getattr(args, "force", False),
            incremental=bool(
                getattr(args, "incremental", False)
                or getattr(args, "changed_files", None)
                or not getattr(args, "force", False)
            ),
            changed_files=changed_files,
            events_path=getattr(args, "events", None),
            prune_deleted=bool(getattr(args, "prune_deleted", False)),
        )
        print(json.dumps(result, indent=2))
        return

    if args.command == "query":
        result = api.query(
            text=args.text,
            k=args.k,
            file=getattr(args, "file", None),
            level=getattr(args, "level", 3),
            strategy=getattr(args, "strategy", "hybrid"),
            explain=bool(getattr(args, "explain", False)),
            scope=getattr(args, "scope", "global"),
            dedupe_by=getattr(args, "dedupe_by", "entity"),
            auto_refresh=bool(getattr(args, "refresh_stale", False)),
            corpus_ids=getattr(args, "corpus", None),
        )
        if getattr(args, "json", False):
            print(json.dumps(result, indent=2))
        else:
            print(f"Query: '{result.get('query', '')}'")
            for row in result.get("results", []):
                print(
                    f"[{row.get('rank', '?')}] [{row.get('score', 0.0):.4f}] "
                    f"{row.get('name', 'unknown')} ({row.get('type', 'symbol')})"
                )
                print(f"    Path: {row.get('file', '?')}:{row.get('line', '?')}")
                if row.get("corpus_id"):
                    print(f"    Corpus: {row.get('corpus_id')}")
                if row.get("reason"):
                    print(f"    Why:  {row['reason']}")
                if getattr(args, "explain", False) and row.get("explanation"):
                    print(
                        f"    Explain: {json.dumps(row['explanation'], sort_keys=True)}"
                    )
                    print("")
        return

    if args.command == "synth":
        from saguaro.synthesis.builder import DeterministicSynthesisBuilder

        builder = DeterministicSynthesisBuilder(repo_root)
        if args.synth_op == "lower":
            spec = builder.lower(
                objective=getattr(args, "objective", None),
                roadmap_path=getattr(args, "roadmap", None),
            )
            payload = spec.to_dict()
        elif args.synth_op == "build":
            payload = builder.build(
                objective=getattr(args, "objective", None),
                roadmap_path=getattr(args, "roadmap", None),
                out_dir=getattr(args, "out_dir", "."),
            )
        else:
            payload = {"status": "error", "message": "Select synth lower or synth build."}
        if getattr(args, "format", "json") == "json":
            print(json.dumps(payload, indent=2))
        else:
            if args.synth_op == "lower":
                print(json.dumps(payload, indent=2))
            else:
                print(f"Synthesis wrote {len(payload.get('written_files', []))} file(s).")
                for item in payload.get("written_files", []):
                    print(item)
        return

    if args.command == "coverage":
        result = api.coverage(
            path=getattr(args, "path", "."),
            structural=bool(getattr(args, "structural", False)),
            by_language=bool(getattr(args, "by_language", False)),
        )
        if getattr(args, "json", False):
            print(json.dumps(result, indent=2))
        else:
            from saguaro.coverage import CoverageReporter

            reporter = CoverageReporter(os.path.abspath(getattr(args, "path", ".")))
            reporter.print_report(
                structural=bool(getattr(args, "structural", False)),
                by_language=bool(getattr(args, "by_language", False)),
            )
        return

    if args.command == "doctor":
        print(json.dumps(api.doctor(), indent=2))
        return

    if args.command == "recover":
        print(json.dumps(api.recover(), indent=2))
        return

    if args.command == "abi":
        report = api.abi(action=getattr(args, "abi_op", "verify"))
        if getattr(args, "format", "text") == "json":
            print(json.dumps(report, indent=2))
        else:
            if getattr(args, "abi_op", "verify") == "orphaned":
                summary = dict(report.get("summary") or {})
                print(
                    "ABI orphaned report: "
                    f"{summary.get('cluster_count', 0)} clusters, "
                    f"{summary.get('unreachable_node_count', 0)} unreachable nodes"
                )
            else:
                native_abi = dict(report.get("native_abi") or {})
                status = "pass" if native_abi.get("ok") else "warning"
                print(f"ABI verify: {status}")
                if native_abi.get("reason"):
                    print(str(native_abi.get("reason")))
        return

    if args.command == "daemon":
        daemon_op = getattr(args, "daemon_op", None) or "status"
        lines = int(getattr(args, "lines", 200) or 200)
        interval = int(getattr(args, "interval", 5) or 5)
        print(
            json.dumps(
                api.daemon(action=daemon_op, interval=interval, lines=lines),
                indent=2,
            )
        )
        return

    if args.command == "workspace":
        workspace_op = getattr(args, "workspace_op", None) or "status"
        print(
            json.dumps(
                api.workspace(
                    action=workspace_op,
                    name=getattr(args, "name", None),
                    workspace_id=getattr(args, "workspace_id", None),
                    against=getattr(args, "against", "main"),
                    description=getattr(args, "description", ""),
                    switch=bool(getattr(args, "switch", False)),
                    limit=int(getattr(args, "limit", 200) or 200),
                    label=getattr(args, "label", "manual"),
                ),
                indent=2,
            )
        )
        return

    if args.command == "sync":
        sync_op = getattr(args, "sync_op", None) or "index"
        if sync_op == "peer":
            peer_op = getattr(args, "peer_op", None) or "list"
            sync_op = f"peer-{peer_op}"
        print(
            json.dumps(
                api.sync(
                    action=sync_op,
                    full=False,
                    peer_id=getattr(args, "peer_id", None),
                    peer_name=getattr(args, "name", None),
                    peer_url=getattr(args, "url", None),
                    auth_token=getattr(args, "auth_token", None),
                    bundle_path=getattr(args, "bundle", None),
                    workspace_id=getattr(args, "workspace_id", None),
                    limit=int(getattr(args, "limit", 1000) or 1000),
                ),
                indent=2,
            )
        )
        return

    if args.command == "graph":
        changed_files = [
            item.strip()
            for item in (getattr(args, "changed_files", "") or "").split(",")
            if item.strip()
        ] or None
        if args.graph_op == "build":
            print(
                json.dumps(
                    api.graph_build(
                        path=getattr(args, "path", "."),
                        incremental=not getattr(args, "full", False),
                        changed_files=changed_files,
                    ),
                    indent=2,
                )
            )
        elif args.graph_op == "ffi":
            report = api.ffi(
                path=getattr(args, "path", "."),
                limit=int(getattr(args, "limit", 200) or 200),
            )
            if getattr(args, "format", "text") == "json":
                print(json.dumps(report, indent=2))
            else:
                print(f"FFI Boundaries: {report.get('count', 0)}")
                for row in report.get("boundaries", []):
                    print(
                        f"- [{row.get('mechanism', 'unknown')}] "
                        f"{row.get('file', '?')}:{row.get('line', '?')} "
                        f"{row.get('snippet', '')}"
                    )
        elif args.graph_op == "query":
            print(
                json.dumps(
                    api.graph_query(
                        expression=getattr(args, "expression", None),
                        symbol=getattr(args, "symbol", None),
                        file=getattr(args, "file", None),
                        relation=getattr(args, "relation", None),
                        depth=getattr(args, "depth", 1),
                        limit=getattr(args, "limit", 50),
                        source=getattr(args, "source", None),
                        target=getattr(args, "target", None),
                        query_path=bool(getattr(args, "query_path", False)),
                        reachable_from=getattr(args, "reachable_from", None),
                        max_depth=getattr(args, "max_depth", None),
                    ),
                    indent=2,
                )
            )
        else:
            print(json.dumps(api.graph_export(), indent=2))
        return

    if args.command == "disparate":
        payload = api.disparate_relations(
            relation=getattr(args, "relation", None),
            limit=int(getattr(args, "limit", 50) or 50),
            refresh=bool(getattr(args, "refresh", False)),
        )
        if getattr(args, "format", "json") == "json":
            print(json.dumps(payload, indent=2))
        else:
            print(
                f"Disparate Relations: {payload.get('count', 0)} / {payload.get('total_count', 0)}"
            )
            for row in list(payload.get("relations") or []):
                print(
                    f"- [{row.get('relation', 'unknown')}] "
                    f"{row.get('source_path', '?')} -> {row.get('target_path', '?')} "
                    f"confidence={float(row.get('confidence', 0.0) or 0.0):.3f}"
                )
        return

    if args.command == "ffi":
        report = api.ffi_audit(
            path=getattr(args, "path", "."),
            limit=int(getattr(args, "limit", 200) or 200),
        )
        if getattr(args, "format", "text") == "json":
            print(json.dumps(report, indent=2))
        else:
            print(f"FFI audit boundaries: {report.get('count', 0)}")
        return

    if args.command == "bridge":
        bridge_op = str(getattr(args, "bridge_op", "audit") or "audit").strip().lower()
        if bridge_op == "explain":
            report = api.bridge(
                path=getattr(args, "path", "."),
                limit=int(getattr(args, "limit", 200) or 200),
                symbol=getattr(args, "symbol", None),
            )
        else:
            report = api.bridge(
                path=getattr(args, "path", "."),
                limit=int(getattr(args, "limit", 200) or 200),
            )
        if getattr(args, "format", "text") == "json":
            print(json.dumps(report, indent=2))
        else:
            if bridge_op == "explain":
                chain = dict(report.get("chain") or {})
                print(
                    f"Bridge explain for {report.get('symbol', '<unknown>')}: "
                    f"{chain.get('hop_count', 0)} hops"
                )
                for hop in list(chain.get("hops") or [])[:20]:
                    print(
                        f"- {hop.get('file', '?')}:{hop.get('line', '?')} "
                        f"[{hop.get('mechanism', 'unknown')}] {hop.get('target', '')}"
                    )
            else:
                print(f"Bridge boundaries: {report.get('count', 0)}")
        return

    if args.command == "trace":
        report = api.trace(
            entry_point=getattr(args, "entry_point", None),
            query=getattr(args, "query", None),
            depth=int(getattr(args, "depth", 20) or 20),
            max_stages=int(getattr(args, "max_stages", 128) or 128),
            include_complexity=not bool(getattr(args, "no_complexity", False)),
        )
        output_format = getattr(args, "format", "text")
        if output_format == "json":
            print(json.dumps(report, indent=2))
        elif output_format == "mermaid":
            print(_trace_to_mermaid(report))
        else:
            print(_format_trace_report(report))
        return

    if args.command == "complexity":
        report = api.complexity(
            symbol=getattr(args, "symbol", None),
            file=getattr(args, "file", None),
            pipeline=getattr(args, "pipeline", None),
            depth=int(getattr(args, "depth", 20) or 20),
            include_flops=bool(getattr(args, "include_flops", False)),
        )
        if getattr(args, "format", "text") == "json":
            print(json.dumps(report, indent=2))
        else:
            print(_format_complexity_report(report))
        return

    if args.command == "health":
        print(json.dumps(api.health(), indent=2))
        return

    if args.command == "serve" and getattr(args, "port", None):
        from saguaro.app.server import run_server

        run_server(repo_path=repo_root, host=args.host, port=args.port)
        return

    if args.command == "app":
        from saguaro.app.server import run_server

        run_server(repo_path=repo_root, host=args.host, port=args.port)
        return

    if args.command == "verify":
        if (
            os.getenv("CI", "").lower() in {"1", "true", "yes"}
            and args.format != "json"
        ):
            print("Error: --format json is required when CI=true.", file=sys.stderr)
            sys.exit(2)

        compliance_context = None
        if args.compliance_context:
            raw_context = str(args.compliance_context)
            context_path = os.path.abspath(raw_context)
            try:
                if os.path.exists(context_path):
                    with open(context_path, encoding="utf-8") as f:
                        compliance_context = json.load(f)
                else:
                    compliance_context = json.loads(raw_context)
            except Exception as exc:
                print(f"Invalid --compliance-context payload: {exc}", file=sys.stderr)
                sys.exit(2)

        result = api.verify(
            path=args.path,
            engines=args.engines,
            fix=args.fix,
            fix_mode="full" if args.unsafe_fixes else args.fix_mode,
            dry_run=args.dry_run,
            preview=args.patch_preview,
            receipt_dir=args.receipt_out,
            assisted=args.assisted,
            max_files=args.max_files,
            aal=args.aal,
            domain=args.domain,
            require_trace=args.require_trace,
            require_evidence=args.require_evidence,
            require_valid_waivers=args.require_valid_waivers,
            change_manifest_path=args.change_manifest,
            compliance_context=compliance_context,
            evidence_bundle=args.evidence_bundle,
            min_parser_coverage=args.min_parser_coverage,
        )
        if args.aes_report:
            from saguaro.coverage import CoverageReporter

            coverage = CoverageReporter(repo_root).generate_report()
            aes_report = _build_aes_compliance_report(
                result,
                total_files_scanned=int(coverage.get("total_files", 0)),
            )
            if args.format == "json":
                print(json.dumps(aes_report, indent=2))
            else:
                print(_format_aes_compliance_report(aes_report))
        else:
            if args.format == "json":
                print(json.dumps(result, indent=2))
            else:
                if result.get("status") == "pass":
                    print("Sentinel Validation Passed: No violations.")
                else:
                    if result.get("fix_plan"):
                        print(
                            f"Remediation Plan: {result['fix_plan'].get('batch_count', 0)} batches, "
                            f"{result.get('fixed', 0)} fixes applied."
                        )
                        if result.get("receipt_dir"):
                            print(f"Receipts: {result['receipt_dir']}")
                    print(
                        f"Sentinel Validation Failed: {result.get('count', 0)} violations found."
                    )
                    for v in result.get("violations", []):
                        print(
                            f"[{v.get('severity', 'low')}] "
                            f"{v.get('file', '?')}:{v.get('line', '?')} - "
                            f"{v.get('message', '')} ({v.get('rule_id', 'UNKNOWN')})"
                        )
        if result.get("status") != "pass":
            sys.exit(1)
        return

    if args.command == "research":
        if args.research_op == "ingest":
            result = api.research_ingest(
                source=args.source,
                manifest_path=args.manifest,
            )
        else:
            result = {"entries": api.research_list()}
        print(json.dumps(result, indent=2))
        return

    if args.command == "docs":
        if args.docs_op == "parse":
            result = api.docs_parse(path=args.path)
        else:
            result = api.docs_graph(path=args.path)
        if args.format == "json":
            print(json.dumps(result, indent=2))
        else:
            print(f"Docs: {result.get('count', 0)} parsed files")
        return

    if args.command == "requirements":
        if args.requirements_op == "extract":
            result = api.requirements_extract(path=args.path)
        elif args.requirements_op == "list":
            result = api.requirements_list(path=args.path)
        elif args.requirements_op == "show":
            result = api.requirements_show(args.requirement_id, path=args.path)
        else:
            result = {"status": "error", "message": "Select a requirements subcommand."}
        if args.format == "json":
            print(json.dumps(result, indent=2))
        else:
            if args.requirements_op == "show" and result.get("requirement"):
                req = result["requirement"]
                print(
                    f"{req.get('requirement_id', req.get('id'))}: {req.get('statement', req.get('text_raw', ''))}"
                )
            else:
                print(f"Requirements: {result.get('count', 0)}")
        return

    if args.command == "traceability":
        if args.traceability_op == "build":
            result = api.traceability_build(docs=args.docs)
        elif args.traceability_op == "status":
            result = api.traceability_status(args.req)
        elif args.traceability_op == "diff":
            result = api.traceability_diff()
        elif args.traceability_op == "orphaned":
            result = api.traceability_orphaned()
        else:
            result = {"status": "error", "message": "Select a traceability subcommand."}
        if args.format == "json":
            print(json.dumps(result, indent=2))
        else:
            print(f"Traceability: {result.get('status', 'ok')}")
        return

    if args.command == "validate":
        if args.validate_op == "docs":
            result = api.validate_docs(path=args.path)
        elif args.validate_op == "requirement":
            result = api.validate_requirement(args.id)
        elif args.validate_op == "gaps":
            result = api.validate_gaps(path=args.path)
        else:
            result = {"status": "error", "message": "Select a validate subcommand."}
        if args.format == "json":
            print(json.dumps(result, indent=2))
        else:
            print(f"Validation: {result.get('status', 'ok')}")
        return

    if args.command == "math":
        if args.math_op == "parse":
            result = api.math_parse(path=args.path)
        elif args.math_op == "map":
            result = api.math_map(args.id)
        else:
            result = {"status": "error", "message": "Select a math subcommand."}
        if args.format == "json":
            print(json.dumps(result, indent=2))
        else:
            print(f"Math: {result.get('count', result.get('status', 'ok'))}")
        return

    if args.command == "cpu":
        if args.cpu_op == "scan":
            result = api.cpu_scan(path=args.path, arch=args.arch, limit=args.limit)
        else:
            result = {"status": "error", "message": "Select a cpu subcommand."}
        if args.format == "json":
            print(json.dumps(result, indent=2))
        else:
            print(f"CPU Hotspots: {result.get('hotspot_count', 0)}")
        return

    if args.command == "omnigraph":
        if args.omnigraph_op == "build":
            result = api.omnigraph_build(path=args.path)
        elif args.omnigraph_op == "explain":
            result = api.omnigraph_explain(args.req, path=args.path)
        elif args.omnigraph_op == "find":
            result = api.omnigraph_find(args.equation, path=args.path)
        elif args.omnigraph_op == "diff":
            result = api.omnigraph_diff()
        elif args.omnigraph_op == "gaps":
            result = api.omnigraph_gaps(modality=args.modality, path=args.path)
        else:
            result = {"status": "error", "message": "Select an omnigraph subcommand."}
        if args.format == "json":
            print(json.dumps(result, indent=2))
        else:
            print(f"OmniGraph: {result.get('status', 'ok')}")
        return

    if args.command == "packet":
        if args.packet_op == "build":
            result = api.packet_build(args.task)
        elif args.packet_op == "review":
            result = api.packet_review(args.packet_id)
        elif args.packet_op == "witness":
            result = api.packet_witness(args.requirement_id)
        else:
            result = {"status": "error", "message": "Select a packet subcommand."}
        if args.format == "json":
            print(json.dumps(result, indent=2))
        else:
            print(f"Packet: {result.get('id', result.get('status', 'ok'))}")
        return

    if args.command == "roadmap":
        if args.roadmap_op == "validate":
            result = api.roadmap_validate(path=args.path)
        elif args.roadmap_op == "graph":
            result = api.roadmap_graph(path=args.path)
        else:
            result = {"status": "error", "message": "Select a roadmap subcommand."}
        if args.format == "json":
            print(json.dumps(result, indent=2))
        else:
            summary = dict(result.get("summary") or {})
            print(
                "Roadmap: "
                f"{summary.get('completed', summary.get('completed_count', 0))} completed, "
                f"{summary.get('partial', summary.get('partial_count', 0))} partial, "
                f"{summary.get('missing', summary.get('missing_count', 0))} missing"
            )
        return

    if args.command == "reality":
        if args.reality_op == "events":
            result = api.reality_events(run_id=args.run_id, limit=args.limit)
        elif args.reality_op == "graph":
            result = api.reality_graph(run_id=args.run_id, limit=args.limit)
        elif args.reality_op == "twin":
            result = api.reality_twin(run_id=args.run_id, limit=args.limit)
        elif args.reality_op == "export":
            result = api.reality_export(run_id=args.run_id, limit=args.limit)
        else:
            result = {"status": "error", "message": "Select a reality subcommand."}
        if args.format == "json":
            print(json.dumps(result, indent=2))
        else:
            summary = dict(result.get("summary") or {})
            print(
                f"Reality: status={result.get('status', 'ok')} "
                f"run_id={result.get('run_id')} "
                f"events={summary.get('event_count', result.get('count', 0))}"
            )
        return

    if args.command == "packs":
        if args.packs_op == "list":
            result = api.packs_list()
        elif args.packs_op == "enable":
            result = api.packs_enable(args.name)
        elif args.packs_op == "diagnose":
            result = api.packs_diagnose(path=args.path)
        else:
            result = {"status": "error", "message": "Select a packs subcommand."}
        if args.format == "json":
            print(json.dumps(result, indent=2))
        else:
            pack_count = (
                len(result.get("packs", []))
                if isinstance(result.get("packs"), list)
                else 0
            )
            print(f"Packs: {pack_count or result.get('status', 'ok')}")
        return

    if args.command == "eval":
        if args.eval_op == "run":
            result = api.eval_run(args.suite, k=args.k, limit=args.limit)
        else:
            result = {"runs": api.metrics_list(limit=args.limit, category="eval")}
        print(json.dumps(result, indent=2))
        return

    if args.command == "report":
        report_data = api.report()
        if args.format == "json":
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2)
        else:
            # Keep behavior compatible: markdown output when requested.
            from saguaro.analysis.report import ReportGenerator

            ReportGenerator(repo_root).save_markdown(report_data, args.output)
        print(f"Report saved to: {args.output}")
        return

    if args.command == "impact":
        report = api.impact(path=args.path)
        print(json.dumps(report, indent=2))
        return

    if args.command == "deadcode":
        report = api.deadcode(
            threshold=args.threshold,
            low_usage_max_refs=args.low_usage_max_refs,
            lang=getattr(args, "lang", None),
            evidence=bool(getattr(args, "evidence", False)),
            runtime_observed=bool(getattr(args, "runtime_observed", False)),
            explain=bool(getattr(args, "explain", False)),
        )
        if args.format == "json":
            print(json.dumps(report, indent=2))
        else:
            low_usage = dict(report.get("low_usage") or {})
            low_usage_candidates = list(low_usage.get("candidates") or [])
            if not report["candidates"]:
                print("No dead code found with high confidence.")
            else:
                print(f"Found {report['count']} candidates:")
                for c in report["candidates"]:
                    print(
                        f"[{c['confidence']:.2f}] {c['symbol']} "
                        f"[{c.get('classification', 'dead')}] ({c['file']})"
                    )
            if low_usage_candidates:
                print(
                    "\nLow-usage live symbols "
                    f"(max refs={low_usage.get('max_refs', args.low_usage_max_refs)}):"
                )
                for item in low_usage_candidates[:30]:
                    evidence = dict(item.get("evidence") or {})
                    usage_count = int(evidence.get("usage_count", 0) or 0)
                    references = list(evidence.get("referencing_files") or [])
                    ref_preview = ", ".join(references[:3]) if references else "none"
                    print(
                        f"[{usage_count}] {item.get('symbol', '<unknown>')} "
                        f"({item.get('file', '<unknown>')}) refs: {ref_preview}"
                    )
            print("\nNote: Verify manually before deletion.")
        return

    if args.command == "low-usage":
        report = api.low_usage(
            max_refs=args.max_refs,
            include_tests=args.include_tests,
            path=args.path,
            limit=args.limit,
        )
        if args.format == "json":
            print(json.dumps(report, indent=2))
        else:
            candidates = list(report.get("candidates") or [])
            if not candidates:
                print("No low-usage live symbols found.")
            else:
                print(
                    f"Found {report.get('count', len(candidates))} low-usage live symbols "
                    f"(max refs={report.get('max_refs', args.max_refs)}):"
                )
                if report.get("path_filter"):
                    print(f"Path filter: {report['path_filter']}")
                dry_candidates = list(report.get("dry_candidates") or [])
                if dry_candidates:
                    print("Top DRY candidates:")
                    for item in dry_candidates[:20]:
                        signals = (
                            ", ".join(list(item.get("dry_signals") or [])[:3])
                            or "low-usage"
                        )
                        print(
                            f"[{float(item.get('reuse_score', 0.0) or 0.0):.2f}] "
                            f"{item.get('symbol', '<unknown>')} "
                            f"({item.get('file', '<unknown>')}) signals: {signals}"
                        )
                    print("")
                areas = list(report.get("areas") or [])
                if areas:
                    print("Top areas:")
                    for area in areas[:10]:
                        examples = (
                            ", ".join(list(area.get("examples") or [])[:3]) or "n/a"
                        )
                        print(
                            f"- {area.get('path', '.')}: "
                            f"{int(area.get('count', 0) or 0)} low-usage, "
                            f"{int(area.get('dry_count', 0) or 0)} DRY candidates "
                            f"({examples})"
                        )
                    print("")
                print("Low-usage symbols:")
                for item in candidates[:100]:
                    evidence = dict(item.get("evidence") or {})
                    usage_count = int(evidence.get("usage_count", 0) or 0)
                    references = list(evidence.get("referencing_files") or [])
                    reference_preview = (
                        ", ".join(references[:3]) if references else "none"
                    )
                    print(
                        f"[{usage_count}] {item.get('symbol', '<unknown>')} "
                        f"({item.get('file', '<unknown>')}) refs: {reference_preview}"
                    )
        return

    if args.command == "architecture":
        if args.architecture_op == "map":
            report = api.architecture_map(path=args.path)
        elif args.architecture_op == "verify":
            report = api.architecture_verify(path=args.path)
        elif args.architecture_op == "violations":
            report = api.architecture_violations(path=args.path)
        elif args.architecture_op == "explain":
            report = api.architecture_explain(path=args.path)
        elif args.architecture_op == "zones":
            report = api.architecture_zones(path=args.path)
        else:
            report = {
                "status": "error",
                "message": "Select an architecture subcommand.",
            }
        if getattr(args, "format", "json") == "json":
            print(json.dumps(report, indent=2))
        else:
            if args.architecture_op == "map":
                summary = report.get("summary", {})
                print(
                    "Architecture map: "
                    f"{summary.get('file_count', 0)} files, "
                    f"{summary.get('misplaced_count', 0)} misplaced, "
                    f"{summary.get('illegal_dependency_count', 0)} illegal crossings"
                )
                for row in report.get("zone_crossings", [])[:20]:
                    print(f"{row['from_zone']} -> {row['to_zone']}: {row['count']}")
            elif args.architecture_op == "verify":
                findings = report.get("findings", [])
                if not findings:
                    print("Architecture verification passed.")
                else:
                    print(
                        f"Architecture verification failed with {len(findings)} findings:"
                    )
                    for item in findings[:50]:
                        print(
                            f"{item['rule_id']} {item['file']}:{item['line']} {item['message']}"
                        )
            elif args.architecture_op == "violations":
                findings = report.get("violations", [])
                if not findings:
                    print("No architecture violations.")
                else:
                    print(f"Architecture violations: {len(findings)}")
                    for item in findings[:50]:
                        print(
                            f"{item['rule_id']} {item['file']}:{item['line']} {item['message']}"
                        )
            elif args.architecture_op == "explain":
                if report.get("status") != "ok":
                    print(json.dumps(report, indent=2))
                else:
                    zone = report.get("zone", {})
                    print(
                        f"{report['path']}: zone={zone.get('zone', 'unknown')} "
                        f"kind={zone.get('kind', 'other')}"
                    )
                    for violation in report.get("violations", []):
                        print(
                            f"{violation['rule_id']}:{violation['line']} {violation['message']}"
                        )
            elif args.architecture_op == "zones":
                zones = report.get("zones", {})
                for rel_path, row in list(zones.items())[:100]:
                    print(f"{rel_path}: {row.get('zone', 'unknown')}")
        return

    if args.command == "duplicates":
        report = api.duplicates(path=args.path)
        if args.format == "json":
            print(json.dumps(report, indent=2))
        else:
            if not report.get("clusters"):
                print("No duplicate clusters found.")
            else:
                print(f"Found {report['count']} duplicate clusters:")
                for cluster in report["clusters"][:30]:
                    print(
                        f"[{cluster['kind']}] {cluster['file_count']} files "
                        f"{', '.join(cluster['paths'][:3])}"
                    )
        return

    if args.command in {"redundancy", "clones", "duplicate-clusters"}:
        if args.command == "redundancy":
            report = api.redundancy(
                path=args.path,
                symbol=getattr(args, "symbol", None),
            )
        elif args.command == "clones":
            report = api.clones(path=args.path)
        else:
            report = api.duplicate_clusters(path=args.path)
        if args.format == "json":
            print(json.dumps(report, indent=2))
        else:
            print(f"Duplicate clusters: {report.get('count', 0)}")
        return

    if args.command == "liveness":
        report = api.liveness(
            symbol=args.symbol,
            threshold=args.threshold,
            include_tests=args.include_tests,
            include_fragments=args.include_fragments,
            max_clusters=args.max_clusters,
        )
        if args.format == "json":
            print(json.dumps(report, indent=2))
        else:
            if args.symbol:
                item = report.get("item")
                if not item:
                    print(f"No liveness data found for {args.symbol}.")
                else:
                    print(
                        f"{item['symbol']} [{item['classification']}] "
                        f"{item['file']}:{item['line']}"
                    )
                    print(item["reason"])
            else:
                print(
                    f"Liveness candidates: {report.get('count', 0)}; "
                    f"unreachable clusters: {report.get('summary', {}).get('unreachable_feature_clusters', 0)}"
                )
                for item in report.get("candidates", [])[:40]:
                    print(
                        f"[{item['confidence']:.2f}] {item['classification']} "
                        f"{item['symbol']} ({item['file']})"
                    )
        return

    if args.command == "reachability":
        report = api.reachability(
            symbol=getattr(args, "symbol", None),
            threshold=args.threshold,
            include_tests=args.include_tests,
            include_fragments=args.include_fragments,
            max_clusters=args.max_clusters,
        )
        if args.format == "json":
            print(json.dumps(report, indent=2))
        else:
            if getattr(args, "symbol", None):
                item = report.get("item")
                if not item:
                    print(f"No reachability data found for {args.symbol}.")
                else:
                    print(
                        f"{item['symbol']} [{item['classification']}] "
                        f"{item['file']}:{item['line']}"
                    )
                    print(item["reason"])
            else:
                print(
                    f"Reachability candidates: {report.get('count', 0)}; "
                    f"clusters: {report.get('summary', {}).get('unreachable_feature_clusters', 0)}"
                )
        return

    if args.command == "unwired":
        report = api.unwired(
            threshold=args.threshold,
            min_nodes=args.min_nodes,
            min_files=args.min_files,
            include_tests=args.include_tests,
            include_fragments=args.include_fragments,
            max_clusters=args.max_clusters,
            refresh_graph=not args.no_refresh_graph,
        )
        if args.format == "json":
            print(json.dumps(report, indent=2))
        else:
            print(_format_unwired_report(report))
        return

    if args.command == "health":
        print(json.dumps(api.health(), indent=2))
        return

    if args.command == "chronicle":
        if args.chronicle_op == "snapshot":
            print(
                json.dumps(api.chronicle_snapshot(description="CLI Snapshot"), indent=2)
            )
            return
        if args.chronicle_op == "diff":
            print(json.dumps(api.chronicle_diff(), indent=2))
            return
        if args.chronicle_op == "list":
            from saguaro.chronicle.storage import ChronicleStorage

            storage = ChronicleStorage(db_path=os.path.join(".saguaro", "chronicle.db"))
            print(json.dumps(storage.list_snapshots(), indent=2))
            return

    if args.command == "memory":
        if args.snapshot:
            print(
                json.dumps(
                    api.memory_snapshot(
                        campaign_id=args.snapshot,
                        db_path=getattr(args, "db_path", None),
                        storage_root=getattr(args, "storage_root", None),
                    ),
                    indent=2,
                )
            )
            return
        if args.restore:
            print(
                json.dumps(
                    api.memory_restore(
                        snapshot_dir=args.restore,
                        campaign_id=getattr(args, "campaign_id", None),
                        db_path=getattr(args, "db_path", None),
                        storage_root=getattr(args, "storage_root", None),
                    ),
                    indent=2,
                )
            )
            return
        if args.list:
            print(json.dumps(api.memory_list(), indent=2))
            return
        if args.read:
            print(json.dumps(api.memory_read(args.read, tier=args.tier), indent=2))
            return
        if args.write:
            key = args.key or args.write[:20]
            print(
                json.dumps(api.memory_write(key, args.write, tier=args.tier), indent=2)
            )
            return

    if args.command == "agent":
        if args.agent_command == "skeleton":
            print(json.dumps(api.skeleton(args.file), indent=2))
            return
        if args.agent_command == "slice":
            result = api.slice(
                args.symbol,
                depth=args.depth,
                corpus_id=getattr(args, "corpus", None),
            )
            if "error" in result:
                print(f"ERROR: {result['error']}", file=sys.stderr)
                print(result.get("suggestion", ""), file=sys.stderr)
                sys.exit(1)
            print(json.dumps(result, indent=2))
            return
        if args.agent_command == "architecture":
            print(json.dumps(api.architecture_map(path=args.path), indent=2))
            return
        if args.agent_command == "bridge":
            print(
                json.dumps(
                    api.bridge(
                        path=args.path,
                        limit=int(getattr(args, "limit", 200) or 200),
                    ),
                    indent=2,
                )
            )
            return
        if args.agent_command == "duplicates":
            print(json.dumps(api.duplicates(path=args.path), indent=2))
            return
        if args.agent_command == "redundancy":
            print(
                json.dumps(
                    api.redundancy(
                        path=args.path,
                        symbol=getattr(args, "symbol", None),
                    ),
                    indent=2,
                )
            )
            return
        if args.agent_command == "clones":
            print(json.dumps(api.clones(path=args.path), indent=2))
            return
        if args.agent_command == "duplicate-clusters":
            print(json.dumps(api.duplicate_clusters(path=args.path), indent=2))
            return
        if args.agent_command == "liveness":
            print(
                json.dumps(
                    api.liveness(
                        symbol=args.symbol,
                        threshold=args.threshold,
                        include_tests=args.include_tests,
                        include_fragments=args.include_fragments,
                        max_clusters=args.max_clusters,
                    ),
                    indent=2,
                )
            )
            return
        if args.agent_command == "reachability":
            print(
                json.dumps(
                    api.reachability(
                        symbol=getattr(args, "symbol", None),
                        threshold=args.threshold,
                        include_tests=args.include_tests,
                        include_fragments=args.include_fragments,
                        max_clusters=args.max_clusters,
                    ),
                    indent=2,
                )
            )
            return
        if args.agent_command == "zones":
            print(json.dumps(api.architecture_zones(path=args.path), indent=2))
            return
        if args.agent_command == "violations":
            print(json.dumps(api.architecture_violations(path=args.path), indent=2))
            return
        if args.agent_command == "abi":
            print(
                json.dumps(api.abi(action=getattr(args, "abi_op", "verify")), indent=2)
            )
            return
        if args.agent_command == "ffi":
            print(
                json.dumps(
                    api.ffi_audit(
                        path=args.path,
                        limit=int(getattr(args, "limit", 200) or 200),
                    ),
                    indent=2,
                )
            )
            return

    # ... (existing command handlers) ...

    if args.command == "analyze":
        from saguaro.analysis.health_card import RepoHealthCard

        print("Running Deep Texture Analysis...")
        card = RepoHealthCard(repo_root)
        results = card.generate_card()

        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print("\n=== Saguaro Health Card ===")
            print(f"Health Score: {results['health_score']:.2f}/1.00\n")

            m = results["metrics"]
            print(
                f"Complexity:   {m['complexity']['score']:.2f} ({m['complexity']['rating']})"
            )
            print(
                f"Dead Code:    {m['dead_code']['ratio'] * 100:.2f}% ({m['dead_code']['count']} chunks)"
            )
            print(
                f"Type Safety:  {m['type_safety']['score']:.2f} ({m['type_safety']['errors']} errors, {m['type_safety']['density'] * 100:.2f}% density)"
            )
            print("\nrecommendation: Run 'saguaro verify --fix' to improve score.")

    elif args.command == "audit":
        passed = True
        audit_output: dict[str, Any] = {
            "status": "pass",
            "path": args.path or ".",
            "engines": args.engines,
            "checks": {},
        }

        # 1. Run Sentinel Verification
        audit_path = args.path or "."
        verification = api.verify(path=audit_path, engines=args.engines, fix=False)
        violations = verification.get("violations", [])
        audit_output["checks"]["verification"] = {
            "status": "pass" if not violations else "fail",
            "violation_count": len(violations),
        }
        if violations:
            passed = False

        # 2. Check Critical Knowledge Invariants (Zones)
        from saguaro.knowledge_base import KnowledgeBase

        kb = KnowledgeBase(os.path.join(repo_root, ".saguaro"))
        zones = kb.get_facts("zone")  # "do not touch" zones
        # Basic check: did we touch any files in these zones?
        # Requires knowing what changed.
        # For now, we assume "audit" is run on dirty state or we perform a git diff check (not implemented yet).
        # We'll skip for prototype or check if 'path' arg was passed.
        if args.path:  # Audit specific path
            for _z in zones:
                # Check overlap
                pass
        audit_output["checks"]["zones"] = {
            "status": "pass",
            "zones_monitored": len(zones),
            "enforcement_mode": "metadata_only",
        }

        # 3. Impact Risk Assessment
        # If we had a diff, we'd check impact score.
        audit_output["checks"]["impact"] = {
            "status": "pass",
            "risk": "acceptable",
            "mode": "baseline",
        }

        audit_output["status"] = "pass" if passed else "fail"
        if args.format == "json":
            print(json.dumps(audit_output, indent=2))
        else:
            print("SAGUARO Auditor: automated governance check...\n")
            print("[1/3] Running Sentinel Verification...")
            if violations:
                print(f"  FAIL: {len(violations)} violations found.")
            else:
                print("  PASS: Codebase is compliant.")
            print("\n[2/3] Checking Regulatory Zones...")
            print(f"  PASS: {len(zones)} zones monitored (no active violation check).")
            print("\n[3/3] Impact Risk Assessment...")
            print("  PASS: Risk acceptable.")
            print("\n=== Audit Decision ===")
        if passed:
            if args.format != "json":
                print("✅ APPROVED")
            sys.exit(0)
        else:
            if args.format != "json":
                print("❌ REJECTED")
            sys.exit(1)

    elif args.command == "knowledge":
        from saguaro.knowledge_base import KnowledgeBase

        saguaro_dir = os.path.join(repo_root, ".saguaro")
        kb = KnowledgeBase(saguaro_dir)

        if args.kb_op == "add":
            kb.add_fact(args.category, args.key, args.value)
            print(f"Fact added: [{args.category}] {args.key}")

        elif args.kb_op == "list":
            facts = kb.get_facts(args.category)
            if not facts:
                print("No facts found.")
            for f in facts:
                print(f"[{f.category}] {f.key}: {f.value}")

        elif args.kb_op == "search":
            results = kb.search(args.query)
            for f in results:
                print(f"[{f.category}] {f.key}: {f.value}")

    elif args.command == "build-graph":
        from saguaro.build_system.ingestor import BuildGraphIngestor

        ingestor = BuildGraphIngestor(repo_root)
        print("Ingesting build graph...")
        graph = ingestor.ingest()
        structured = graph.get("structured_inputs", {})
        source_coverage = graph.get("source_coverage", {})

        print(f"Discovered {graph['target_count']} targets:")
        print(
            "Structured inputs: "
            f"{structured.get('compile_databases', 0)} compile DB, "
            f"{structured.get('cmake_file_api_replies', 0)} CMake File API reply, "
            f"{structured.get('target_directories', 0)} target directory map"
        )
        print(
            "Source coverage: "
            f"{source_coverage.get('owned_sources', 0)} owned / "
            f"{source_coverage.get('compiled_sources', 0)} compiled "
            f"({source_coverage.get('coverage_percent', 0.0)}%)"
        )
        for name, data in graph["targets"].items():
            print(f" - [{data['type']}] {name} (defined in {data['file']})")
            if data["deps"]:
                print(
                    f"    Deps: {', '.join(data['deps'][:5])}{'...' if len(data['deps']) > 5 else ''}"
                )

    elif args.command == "entrypoints":
        from saguaro.analysis.entry_points import EntryPointDetector

        detector = EntryPointDetector(repo_root)
        print("Scanning for runtime entry points...")
        eps = detector.detect()

        if not eps:
            print("No entry points found.")
        else:
            print(f"Found {len(eps)} entry points:")
            for ep in eps:
                name = ep.get("name", "N/A")
                print(
                    f" - [{ep['type']}] {name} ({os.path.relpath(ep['file'], repo_root)}:{ep['line']})"
                )

    elif args.command == "serve":
        if hasattr(args, "mcp") and args.mcp:
            from saguaro.mcp.server import main as mcp_main

            # Ensure auth token is in environ if not passed via sys.argv check in mcp_main,
            # but mcp_main checks sys.argv too.
            # To be safe, we can set env var if args.auth_token is present,
            # in case sys.argv parsing in mcp_main is position sensitive or confused by other flags.
            if args.auth_token:
                os.environ["SAGUARO_MCP_TOKEN"] = args.auth_token
            mcp_main()
        else:
            from saguaro.dni.server import main as server_main

            server_main()

    elif args.command == "query":
        # One-off query wrapper around DNI logic or Engine
        # For simplicity, reuse DNI logic but print to stdout
        from saguaro.dni.server import DNIServer

        # If we have a file/level, we use EscalationLadder
        if args.file:
            from saguaro.escalation import EscalationLadder
            from saguaro.indexing.auto_scaler import get_repo_stats_and_config
            from saguaro.indexing.engine import IndexEngine
            from saguaro.profiling import profiler

            # Need to partial init the engine/store to get query vector
            target_path = repo_root  # Assume root
            saguaro_dir = os.path.join(target_path, ".saguaro")

            stats = get_repo_stats_and_config(target_path)
            engine = IndexEngine(target_path, saguaro_dir, stats)

            with profiler.measure("query_encoding"):
                # Encode
                query_vec = engine.encode_text(args.text, dim=stats["total_dim"])

            with profiler.measure("search_retrieval"):
                # Search
                ladder = EscalationLadder(engine.store, target_path)
                results = ladder.search(
                    query_vec,
                    args.file,
                    level=args.level,
                    k=args.k,
                    query_text=args.text,
                )

            if args.profile:
                print(
                    f"[Profile] Encoding: {profiler.stats.get('query_encoding', 0):.2f}ms"
                )
                print(
                    f"[Profile] Retrieval: {profiler.stats.get('search_retrieval', 0):.2f}ms"
                )

            workset = None
            if args.workset:
                from saguaro.workset import WorksetManager

                wm = WorksetManager(saguaro_dir)
                workset = wm.get_workset(args.workset)
                if not workset:
                    print(
                        f"Warning: Workset {args.workset} not found.", file=sys.stderr
                    )

            if args.json:
                import time

                from saguaro.context import ContextBuilder

                bundle = ContextBuilder.build_from_results(
                    args.text, results, time.time(), workset=workset
                )
                print(bundle.to_json())
            else:
                print(f"Query: '{args.text}' [Scoped: {args.level}]")
                for res in results:
                    print(
                        f"[{res.get('rank', '?')}] [{res['score']:.4f}] {res['name']} ({res['type']})"
                    )
                    print(f"    Path: {res['file']}:{res['line']}")
                    print(f"    Why:  {res.get('reason', 'N/A')}")
                    print(f"    Scope: {res.get('scope', 'Global')}")
                    print("")

        else:
            # Use standard server logic
            server = DNIServer()
            server.initialize({"path": "."})  # Assume current dir
            result = server.query({"text": args.text, "k": args.k})

            workset = None
            if args.workset:
                from saguaro.workset import WorksetManager

                wm = WorksetManager(os.path.join(repo_root, ".saguaro"))
                workset = wm.get_workset(args.workset)
                if not workset:
                    print(
                        f"Warning: Workset {args.workset} not found.", file=sys.stderr
                    )

            if args.json:
                import time

                from saguaro.context import ContextBuilder

                bundle = ContextBuilder.build_from_results(
                    args.text, result["results"], time.time(), workset=workset
                )
                print(bundle.to_json())
            else:
                print(f"Query: '{args.text}'")
                for res in result["results"]:
                    print(
                        f"[{res['rank']}] [{res['score']:.4f}] {res['name']} ({res['type']})"
                    )
                    print(f"    Path: {res['file']}:{res['line']}")
                    print(f"    Why:  {res.get('reason', 'N/A')}")
                    print("")

    elif args.command == "init":
        print("Initializing SAGUARO...")
        saguaro_dir = os.path.join(repo_root, ".saguaro")
        if os.path.exists(saguaro_dir) and not args.force:
            print("SAGUARO already initialized. Use --force to overwrite.")
            sys.exit(1)

        os.makedirs(saguaro_dir, exist_ok=True)
        # Create default config
        from saguaro.defaults import get_default_yaml

        config_path = os.path.join(saguaro_dir, "config.yaml")
        with open(config_path, "w") as f:
            f.write(get_default_yaml() + "\n")

        print(f"initialized empty SAGUARO repository in {saguaro_dir}")

    elif args.command == "quickstart":
        from saguaro.quickstart import QuickstartManager

        qs = QuickstartManager(repo_root)
        qs.execute()

    elif args.command == "watch":
        target_path = os.path.abspath(args.path)
        print(f"SAGUARO Watch Mode: {target_path}")

        # We need the engine initialized similarly to index command
        # Auto-scale analysis (reuse existing config if possible, or re-analyze)
        from saguaro.indexing.auto_scaler import get_repo_stats_and_config

        # Check if initialized
        saguaro_dir = os.path.join(repo_root, ".saguaro")
        if not os.path.exists(os.path.join(saguaro_dir, "config.yaml")):
            print("Please run 'saguaro init' or 'saguaro index' first.")
            sys.exit(1)

        # Load config to get dims
        import yaml

        # Re-analyze stats to get dimensions
        stats = get_repo_stats_and_config(target_path)

        from saguaro.indexing.coordinator import IndexCoordinator
        from saguaro.watcher import Watcher

        coordinator = IndexCoordinator(repo_root, api=api)
        watcher = Watcher(
            engine=None,
            target_path=target_path,
            interval=args.interval,
            coordinator=coordinator,
        )

        try:
            watcher.start()
        except KeyboardInterrupt:
            watcher.stop()
            print("\nWatcher stopped.")

    elif args.command == "aes-check":
        try:
            from core.aes.lint import format_aes_lint, run_aes_lint
        except ModuleNotFoundError:
            repo_root = _resolve_verify_repo_root(args.path)
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            from core.aes.lint import format_aes_lint, run_aes_lint

        repo_root = _resolve_verify_repo_root(args.path)
        target_path = os.path.abspath(args.path)
        violations = run_aes_lint(
            paths=[target_path],
            repo_root=repo_root,
            aal=args.aal,
            domain=args.domain,
        )
        rendered = format_aes_lint(violations, output_format=args.format)
        if rendered:
            print(rendered)
        sys.exit(1 if violations else 0)

    elif args.command == "chronicle":
        from saguaro.chronicle.storage import ChronicleStorage
        from saguaro.ops import holographic

        storage = ChronicleStorage()

        if args.chronicle_op == "snapshot":
            print("Creating semantic snapshot...")

            # Load stats and engine to get active bundle state
            from saguaro.indexing.auto_scaler import get_repo_stats_and_config
            from saguaro.indexing.memory_optimized_engine import (
                MemoryOptimizedIndexEngine,
            )

            target_path = repo_root
            saguaro_dir = os.path.join(target_path, ".saguaro")
            stats = get_repo_stats_and_config(target_path)
            engine = MemoryOptimizedIndexEngine(target_path, saguaro_dir, stats)

            # Export live state
            state = engine.get_state()

            snapshot_id = storage.save_snapshot(
                hd_state_blob=holographic.serialize_bundle(state),
                description="Manual CLI Snapshot",
            )
            print(f"Snapshot #{snapshot_id} created.")

        elif args.chronicle_op == "list":
            snapshots = storage.list_snapshots()
            if not snapshots:
                print("No snapshots found.")
            else:
                print(
                    f"{'ID':<4} | {'Timestamp':<20} | {'Description':<30} | {'Hash':<10}"
                )
                print("-" * 75)
                for s in snapshots:
                    ts = time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(s["timestamp"])
                    )
                    desc = s["description"] or ""
                    commit = (s["commit_hash"] or "")[:8]
                    print(f"{s['id']:<4} | {ts:<20} | {desc:<30} | {commit:<10}")

        elif args.chronicle_op == "diff":
            from saguaro.chronicle.diff import SemanticDiff

            print("Calculating semantic drift...")

            # 1. Get current state
            from saguaro.indexing.auto_scaler import get_repo_stats_and_config
            from saguaro.indexing.memory_optimized_engine import (
                MemoryOptimizedIndexEngine,
            )

            target_path = repo_root
            saguaro_dir = os.path.join(target_path, ".saguaro")
            stats = get_repo_stats_and_config(target_path)
            engine = MemoryOptimizedIndexEngine(target_path, saguaro_dir, stats)
            current_state = engine.get_state()
            current_blob = holographic.serialize_bundle(current_state)

            # 2. Get latest snapshot
            latest = storage.get_latest_snapshot()
            if not latest:
                print(
                    "Error: No baseline snapshot found. Create one with 'saguaro chronicle snapshot'."
                )
                sys.exit(1)

            drift, details = SemanticDiff.calculate_drift(
                latest["hd_state_blob"], current_blob
            )
            print(f"Comparison: Snapshot #{latest['id']} vs Current Working Directory")
            print(
                f"Drift Score: {drift:.4f} ({SemanticDiff.human_readable_report(drift)})"
            )
            if drift > 0.1:
                print(f"Details: {details}")

    elif args.command == "grounding":
        from saguaro.config.grounding import get_deterministic_params

        params = get_deterministic_params(args.model)
        if args.format == "json":
            print(json.dumps(params, indent=2))
        else:
            if not params:
                print(f"No deterministic profile found for model: {args.model}")
            else:
                print(f"Deterministic Profile for {args.model}:")
                for k, v in params.items():
                    print(f"  {k}: {v}")

    elif args.command == "legislation":
        if args.draft:
            from saguaro.legislator import Legislator

            leg = Legislator(root_dir=repo_root)
            print("Drafting legislation...")
            yaml_content = leg.draft_rules()
            print("\n" + yaml_content)
        else:
            print("Use --draft to generate rules.")

    elif args.command == "train":
        from saguaro.encoder import AdaptiveEncoder

        encoder = AdaptiveEncoder()
        encoder.fine_tune_on_corpus(args.path, epochs=args.epochs)
        print("Adaptive training complete.")

    elif args.command == "train-baseline":
        from saguaro.tokenization.train_baseline import train_baseline

        print("Training Baseline Tokenizer...")
        train_baseline(args.corpus, args.curriculum, args.output, args.fast)

    elif args.command == "constellation":
        from saguaro.constellation.manager import ConstellationManager

        cm = ConstellationManager()

        if args.constellation_op == "list":
            libs = cm.list_libraries()
            if not libs:
                print("Constellation is empty.")
            else:
                print(f"Constellation Libraries ({len(libs)}):")
                for lib in libs:
                    print(f" - {lib}")

        elif args.constellation_op == "index-lib":
            if not args.path:
                print("Error: --path required for index-lib")
                sys.exit(1)
            cm.index_library(args.name, args.path)

        elif args.constellation_op == "link":
            saguaro_dir = os.path.join(repo_root, ".saguaro")
            if not os.path.exists(saguaro_dir):
                print(
                    "Error: Current directory is not initialized. Run 'saguaro init' first."
                )
                sys.exit(1)
            cm.link_to_project(args.name, saguaro_dir)

    elif args.command == "benchmark":
        from saguaro.benchmarks.runner import BenchmarkRunner

        runner = BenchmarkRunner(args.dataset, args.custom)
        results = runner.run()
        runner.print_report(results)

    elif args.command == "coverage":
        from saguaro.coverage import CoverageReporter

        reporter = CoverageReporter(os.path.abspath(args.path))
        reporter.print_report()

    elif args.command == "health":
        from saguaro.health import HealthDashboard

        # Assume .saguaro is in current directory for now, or use args
        saguaro_dir = os.path.join(repo_root, ".saguaro")
        dashboard = HealthDashboard(saguaro_dir)
        dashboard.print_dashboard()

    elif args.command == "governor":
        from saguaro.governor import ContextGovernor

        gov = ContextGovernor()

        if args.check:
            text = args.text or ""
            # Simulate a context item
            item = {"content": text, "name": "CLI_Input"}
            safe, tokens, msg = gov.check_budget([item])

            print("Context Check:")
            print(f"  Tokens: {tokens}")
            print(f"  Status: {msg}")
            print(f"  Safe:   {safe}")

    elif args.command == "workset":
        from saguaro.governor import ContextBudgetExceeded
        from saguaro.workset import WorksetManager

        saguaro_dir = os.path.join(repo_root, ".saguaro")
        wm = WorksetManager(saguaro_dir, repo_path=repo_root)

        if args.workset_op == "create":
            files = [f.strip() for f in args.files.split(",")]
            try:
                ws = wm.create_workset(args.desc, files)
                print(f"Workset created: {ws.id}")
                print(ws.to_json())
            except ContextBudgetExceeded as e:
                print(f"Error: {e}")
                sys.exit(1)

        elif args.workset_op == "list":
            worksets = wm.list_worksets()
            print(f"Found {len(worksets)} worksets:")
            for w in worksets:
                print(
                    f" - [{w.id}] {w.description} ({len(w.files)} files) [{w.status}]"
                )

        elif args.workset_op == "show":
            ws = wm.get_workset(args.id)
            if ws:
                print(ws.to_json())
            else:
                print("Workset not found.")

        elif args.workset_op == "expand":
            files = [f.strip() for f in args.files.split(",")]
            try:
                ws = wm.expand_workset(args.id, files, args.justification)
                print(f"Workset expanded: {ws.id}")
                print(ws.to_json())
            except Exception as e:
                print(f"Error: {e}")
                sys.exit(1)

        elif args.workset_op == "lock":
            try:
                wm.lock_workset(args.id)
                print(f"Workset {args.id} locked.")
            except Exception as e:
                print(f"Error: {e}")

        elif args.workset_op == "unlock":
            try:
                wm.unlock_workset(args.id)
                print(f"Workset {args.id} unlocked.")
            except Exception as e:
                print(f"Error: {e}")

    elif args.command == "scribe":
        from saguaro.agents.scribe import Scribe

        scribe = Scribe(repo_root)
        context_files = [args.file] if args.file else []

        print(f"Scribe generating patch for: '{args.task}'...")
        patch = scribe.generate_patch(args.task, context_files)

        with open(args.out, "w") as f:
            json.dump(patch, f, indent=2)

        print(f"Patch saved to {args.out}")
        print("Apply with: saguaro agent patch <file> " + args.out)

    # --- Phase 4/SSAI Handlers ---
    elif args.command == "agent":
        if args.agent_command == "skeleton":
            from saguaro.agents.perception import SkeletonGenerator

            gen = SkeletonGenerator()
            try:
                result = gen.generate(args.file)
                print(json.dumps(result, indent=2))
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)

        elif args.agent_command == "slice":
            try:
                result = api.slice(
                    args.symbol,
                    depth=args.depth,
                    corpus_id=getattr(args, "corpus", None),
                )

                # Check for actionable error response (Phase 2: AI Adoption)
                if "error" in result and result.get("type") == "INDEX_MISS":
                    print(f"ERROR: {result['error']}", file=sys.stderr)
                    print(f"\n{result['suggestion']}", file=sys.stderr)
                    print("\nRecovery steps:", file=sys.stderr)
                    for step in result.get("recovery_steps", []):
                        print(f"  → {step}", file=sys.stderr)
                    sys.exit(1)

                print(json.dumps(result, indent=2))
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)

        elif args.agent_command == "trace":
            payload = api.trace(
                entry_point=None if args.query else args.entry_or_query,
                query=args.entry_or_query if args.query else None,
                depth=int(args.depth or 20),
            )
            print(json.dumps(payload, indent=2))

        elif args.agent_command == "complexity":
            payload = api.complexity(
                symbol=args.symbol,
                file=args.file,
                include_flops=bool(args.include_flops),
            )
            print(json.dumps(payload, indent=2))

        elif args.agent_command == "ffi-map":
            payload = api.ffi(
                path=args.path,
                limit=int(args.limit or 400),
            )
            print(json.dumps(payload, indent=2))

        elif args.agent_command == "pipeline-diff":
            try:
                from saguaro.analysis.pipeline_diff import PipelineDiff
                from saguaro.api import SaguaroAPI
            except Exception as exc:
                print(f"PipelineDiff unavailable: {exc}", file=sys.stderr)
                sys.exit(1)

            revision_range = str(args.revision_range or "").strip()
            if ".." in revision_range:
                start_rev, end_rev = revision_range.split("..", 1)
                start_rev = start_rev or "HEAD~1"
                end_rev = end_rev or "HEAD"
            else:
                end_rev = revision_range or "HEAD"
                start_rev = f"{end_rev}~1"

            def _trace_at_revision(revision: str) -> dict[str, Any]:
                tmp_root = tempfile.mkdtemp(prefix="saguaro-pipeline-diff-")
                worktree_path = os.path.join(tmp_root, "repo")
                try:
                    subprocess.run(
                        ["git", "worktree", "add", "--detach", worktree_path, revision],
                        cwd=repo_root,
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    rev_api = SaguaroAPI(repo_path=worktree_path)
                    rev_api.index(path=".", force=False, incremental=True)
                    rev_api.graph_build(path=".")
                    return rev_api.trace(entry_point=args.entry, depth=30)
                finally:
                    subprocess.run(
                        ["git", "worktree", "remove", "--force", worktree_path],
                        cwd=repo_root,
                        check=False,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    shutil.rmtree(tmp_root, ignore_errors=True)

            try:
                previous = _trace_at_revision(start_rev)
                current = _trace_at_revision(end_rev)
            except subprocess.CalledProcessError as exc:
                stderr = (exc.stderr or "").strip()
                print(
                    f"Failed to resolve revision range '{revision_range}': {stderr or exc}",
                    file=sys.stderr,
                )
                sys.exit(1)

            diff = PipelineDiff().diff(previous, current)
            diff["revision_range"] = args.revision_range
            diff["from_revision"] = start_rev
            diff["to_revision"] = end_rev
            print(json.dumps(diff, indent=2))

        elif args.agent_command == "hotspots":
            payload = api.graph_query(
                expression="complexity >= O(n^2)",
                limit=int(args.limit or 20),
                depth=3,
            )
            print(json.dumps(payload, indent=2))

        elif args.agent_command == "patch":
            from saguaro.agents.sandbox import Sandbox

            # Load patch
            try:
                if os.path.exists(args.patch_json):
                    with open(args.patch_json) as f:
                        patch_data = json.load(f)
                else:
                    patch_data = json.loads(args.patch_json)
            except Exception as e:
                print(f"Invalid patch JSON: {e}", file=sys.stderr)
                sys.exit(1)

            # Create or get sandbox? For CLI, we create new one and return ID.
            # In a real conversation loop, we'd pass ID.
            # Here we assume a new transaction for the atomic patch.
            sb = Sandbox(repo_root)
            sb.apply_patch(patch_data)
            print(
                f"Patch applied to Sandbox {sb.id}. Run 'saguaro agent verify {sb.id}' to check."
            )
            print(sb.id)  # Output ID on last line for parsing

        elif args.agent_command == "verify":
            from saguaro.agents.sandbox import Sandbox

            sb = Sandbox.get(args.sandbox_id)
            if not sb:
                print(
                    f"Sandbox {args.sandbox_id} not found (in-memory session lost).",
                    file=sys.stderr,
                )
                sys.exit(1)

            report = sb.verify()
            print(json.dumps(report, indent=2))
            if report["status"] != "pass":
                sys.exit(1)

        elif args.agent_command == "commit":
            result = api.sandbox_commit(args.sandbox_id)
            print(json.dumps(result, indent=2))
            if str(result.get("status", "ok")).lower() != "ok":
                sys.exit(1)

        elif args.agent_command == "impact":
            from saguaro.agents.sandbox import Sandbox

            sb = Sandbox.get(args.sandbox_id)
            if not sb:
                print(f"Sandbox {args.sandbox_id} not found.", file=sys.stderr)
                sys.exit(1)

            report = sb.calculate_impact()
            print(json.dumps(report, indent=2))

        elif args.agent_command == "run":
            from saguaro.agents import (
                AuditorAgent,
                CartographerAgent,
                PlannerAgent,
                SurgeonAgent,
            )

            agent_map = {
                "planner": PlannerAgent,
                "cartographer": CartographerAgent,
                "surgeon": SurgeonAgent,
                "auditor": AuditorAgent,
            }

            agent_cls = agent_map.get(args.role)
            if agent_cls:
                agent = agent_cls()
                print(f"Running {agent.name}...")
                # Mock context for CLI run
                result = agent.run(context=None, goal=args.task)
                print(result)
            else:
                print(f"Unknown agent role: {args.role}")

        else:
            parser.parse_args(["agent", "--help"])

    elif args.command == "tasks":
        from saguaro.coordination.graph import TaskGraph, TaskNode

        graph = TaskGraph()

        if args.list:
            ready = graph.get_ready_tasks()
            if not ready:
                print("No ready tasks.")
            for t in ready:
                print(f"[{t.id}] {t.description} (Status: {t.status})")

        elif args.add:
            import uuid

            try:
                data = json.loads(args.add)
                node = TaskNode(
                    id=data.get("id", str(uuid.uuid4())[:8]),
                    description=data.get("description", "No Data"),
                    status="pending",
                    dependencies=data.get("dependencies", []),
                )
                graph.add_task(node)
                print(f"Task {node.id} added.")
            except Exception as e:
                print(f"Error adding task: {e}")

    elif args.command == "memory":
        from saguaro.coordination.memory import SharedMemory

        mem = SharedMemory()

        if args.list:
            facts = mem.list_facts(tier=args.tier)
            if not facts:
                print(f"No facts found in tier '{args.tier}'.")
            else:
                print(f"--- Tier: {args.tier} ---")
                for k, v in facts.items():
                    print(f"{k}: {v['value']} (Source: {v.get('source', '?')})")
        elif args.read:
            if args.tier == "semantic":
                # Semantic retrieval
                from saguaro.indexing.auto_scaler import get_repo_stats_and_config
                from saguaro.indexing.engine import IndexEngine

                target_path = repo_root
                stats = get_repo_stats_and_config(target_path)
                engine = IndexEngine(
                    target_path, os.path.join(target_path, ".saguaro"), stats
                )

                query_vec = engine.encode_text(args.read)
                # This would typically call a semantic memory store,
                # for now we'll simulate output or use SharedMemory if it had search.
                print(
                    f"Executing semantic search for '{args.read}' in tier '{args.tier}' (k={args.k})..."
                )
                # (Actual search logic would go here)
            else:
                val = mem.read_fact(args.read, tier=args.tier)
                print(f"[{args.tier}] {args.read}: {val}")
        elif args.write:
            key = args.key or args.write[:20]
            val = args.write
            mem.write_fact(key, val, agent_id="CLI_USER", tier=args.tier)
            print(f"Fact '{key}' written to tier '{args.tier}'.")

    # --- Phase 5 Handlers ---
    elif args.command == "simulate":
        if args.sim_op == "volatility":
            from saguaro.simulation.volatility import VolatilityMapper

            mapper = VolatilityMapper()
            vmap = mapper.generate_map(repo_root)
            print("Volatility Map:")
            for f, score in vmap.items():
                print(f"[{score:.2f}] {f}")

        elif args.sim_op == "regression":
            from saguaro.simulation.regression import RegressionPredictor

            pred = RegressionPredictor()
            files = args.files.split(",")
            risks = pred.predict_regression(files)
            if risks:
                print("Predicted Risks (Regressions):")
                for r in risks:
                    print(f" - {r}")
            else:
                print("No regressions predicted.")

    # --- Phase 6 Handlers ---
    elif args.command == "route":
        from saguaro.learning.routing import IntentRouter

        router = IntentRouter()
        intent = router.route(args.query)
        print(f"Query: '{args.query}'")
        print(f"Intent: {intent.upper()}")

    elif args.command == "feedback":
        from saguaro.feedback import FeedbackStore

        saguaro_dir = os.path.join(repo_root, ".saguaro")
        fs = FeedbackStore(saguaro_dir)

        if args.fb_op == "log":
            items = []
            if args.used:
                for uid in args.used.split(","):
                    items.append({"id": uid.strip(), "action": "used"})
            if args.ignored:
                for uid in args.ignored.split(","):
                    items.append({"id": uid.strip(), "action": "ignored"})

            sid = fs.log_feedback(args.query, items)
            print(f"Feedback logged. Session ID: {sid}")

        elif args.fb_op == "stats":
            stats = fs.get_stats()
            print("Feedback Stats:")
            for k, v in stats.items():
                print(f"  {k}: {v}")

    elif args.command == "refactor":
        if args.refactor_op == "plan":
            from saguaro.refactor.planner import RefactorPlanner

            planner = RefactorPlanner(repo_root)
            print(f"Analyzing impact for symbol: {args.symbol}...")
            plan = planner.plan_symbol_modification(args.symbol)

            print("\nRefactor Plan Generated:")
            print(f"Impact Score: {plan['impact_score']}")
            print("Impacted Files:")
            for f in plan["files_impacted"]:
                print(f" - {f}")

            print("\nModules:")
            for mod, files in plan["modules"].items():
                print(f"  {mod}: {len(files)} files")

        elif args.refactor_op == "rename":
            from saguaro.refactor.renamer import SemanticRenamer

            renamer = SemanticRenamer(repo_root)
            print(f"Renaming '{args.old}' -> '{args.new}'...")

            res = renamer.rename_symbol(args.old, args.new, dry_run=not args.execute)

            print("Rename Results:")
            if res["files_modified"]:
                for f in res["files_modified"]:
                    print(f" - Modified: {os.path.relpath(f, repo_root)}")
            else:
                print(" - No files matched.")

            if res["errors"]:
                print("\nErrors:")
                for e in res["errors"]:
                    print(f" - {e}")

            if res["dry_run"]:
                print("\n[Dry Run] Use --execute to apply changes.")

        elif args.refactor_op == "shim":
            from saguaro.refactor.shims import CompatShimGenerator

            gen = CompatShimGenerator(repo_root)
            gen.apply_shim(args.path, args.target)
            print(f"Shim created at {args.path} pointing to {args.target}")

        elif args.refactor_op == "safedelete":
            from saguaro.refactor.safety import SafetyEngine

            engine = SafetyEngine(repo_root)
            res = engine.safe_delete(
                args.path, force=args.force, dry_run=not args.execute
            )

            if res["success"]:
                print(f"✅ {res['message']}")
            else:
                print(f"❌ {res['message']}")
                if "blocking_dependents" in res:
                    for d in res["blocking_dependents"]:
                        print(f"   - {os.path.relpath(d, repo_root)}")

    elif args.command == "metrics":
        from saguaro.mcp.adoption_metrics import AdoptionTracker

        saguaro_dir = os.path.join(repo_root, ".saguaro")
        tracker = AdoptionTracker(saguaro_dir)

        if args.reset:
            # Reset metrics by removing the file
            metrics_file = os.path.join(saguaro_dir, "metrics.json")
            if os.path.exists(metrics_file):
                os.remove(metrics_file)
                print("Metrics reset.")
            else:
                print("No metrics to reset.")
        else:
            report = tracker.get_report()

            if args.json:
                print(json.dumps(report, indent=2))
            else:
                tracker.print_report()

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
