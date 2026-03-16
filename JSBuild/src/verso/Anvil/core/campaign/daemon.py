"""Detached campaign worker support."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


class CampaignDaemon:
    """Launch campaign operations in detached local worker processes."""

    def __init__(self, workspace_dir: str) -> None:
        self.workspace_dir = os.path.abspath(workspace_dir)
        self.repo_root = str(Path(__file__).resolve().parents[2])

    def launch_continue(
        self,
        campaign_id: str,
        *,
        environment_profile: dict[str, Any] | None = None,
        detached_lane: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        workspace = Path(self.workspace_dir) / campaign_id
        workspace.mkdir(parents=True, exist_ok=True)
        log_path = workspace / "logs" / "detached_continue.log"
        status_path = workspace / "daemon_status.json"
        status_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        status = {
            "campaign_id": campaign_id,
            "state": "running",
            "started_at": time.time(),
            "log_path": str(log_path),
            "status_path": str(status_path),
            "environment_profile": environment_profile or {},
            "detached_lane": detached_lane or {},
        }
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

        script = (
            "import json, sys, time\n"
            "from core.campaign.runner import CampaignRunner\n"
            "campaign_id = sys.argv[1]\n"
            "workspace_dir = sys.argv[2]\n"
            "status_path = sys.argv[3]\n"
            "runner = CampaignRunner(config={'workspace_dir': workspace_dir})\n"
            "status = json.loads(open(status_path, encoding='utf-8').read())\n"
            "try:\n"
            "    event = runner.continue_autonomy_campaign(campaign_id)\n"
            "    status.update({'state': 'completed', 'completed_at': time.time(), 'result': event})\n"
            "    open(status_path, 'w', encoding='utf-8').write(json.dumps(status, indent=2))\n"
            "    print(json.dumps(event))\n"
            "except Exception as exc:\n"
            "    status.update({'state': 'failed', 'completed_at': time.time(), 'error': str(exc)})\n"
            "    open(status_path, 'w', encoding='utf-8').write(json.dumps(status, indent=2))\n"
            "    raise\n"
        )
        with open(log_path, "a", encoding="utf-8") as handle:
            proc = subprocess.Popen(
                [sys.executable, "-c", script, campaign_id, self.workspace_dir, str(status_path)],
                cwd=self.repo_root,
                stdout=handle,
                stderr=handle,
                start_new_session=True,
            )

        status["pid"] = proc.pid
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        return status

    def status(self, campaign_id: str) -> dict[str, Any]:
        status_path = Path(self.workspace_dir) / campaign_id / "daemon_status.json"
        if not status_path.exists():
            return {"campaign_id": campaign_id, "state": "missing"}
        return json.loads(status_path.read_text(encoding="utf-8"))

    def list_statuses(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        root = Path(self.workspace_dir)
        if not root.exists():
            return items
        for candidate in sorted(root.iterdir()):
            status_path = candidate / "daemon_status.json"
            if status_path.exists():
                items.append(json.loads(status_path.read_text(encoding="utf-8")))
        return items

    def cancel(self, campaign_id: str) -> dict[str, Any]:
        status = self.status(campaign_id)
        pid = int(status.get("pid") or 0)
        if pid > 0 and status.get("state") == "running":
            os.kill(pid, signal.SIGTERM)
            status["state"] = "cancelled"
            status["completed_at"] = time.time()
            status_path = Path(self.workspace_dir) / campaign_id / "daemon_status.json"
            status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        return status

    def tail_log(self, campaign_id: str, *, lines: int = 20) -> str:
        status = self.status(campaign_id)
        log_path = str(status.get("log_path") or "")
        if not log_path or not os.path.exists(log_path):
            return ""
        with open(log_path, encoding="utf-8") as handle:
            payload = handle.readlines()
        return "".join(payload[-max(1, lines) :])
