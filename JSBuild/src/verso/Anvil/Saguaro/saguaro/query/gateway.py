"""Resident query gateway with bounded admission for cross-process Saguaro use."""

from __future__ import annotations

import collections
import contextlib
import json
import os
import signal
import socket
import socketserver
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

from saguaro.indexing.auto_scaler import calibrate_runtime_profile, load_runtime_profile
from saguaro.services.comparative import ComparativeAnalysisService
from saguaro.state.ledger import StateLedger
from saguaro.storage.atomic_fs import atomic_write_json
from queue import Queue

def gateway_socket_path(repo_path: str) -> str:
    root = os.path.abspath(repo_path)
    return os.path.join(root, ".saguaro", "state", "query_gateway.sock")


def gateway_state_path(repo_path: str) -> str:
    root = os.path.abspath(repo_path)
    return os.path.join(root, ".saguaro", "state", "query_gateway.json")


def gateway_log_path(repo_path: str) -> str:
    root = os.path.abspath(repo_path)
    return os.path.join(root, ".saguaro", "state", "query_gateway.log")


def _ensure_state_dir(repo_path: str) -> None:
    os.makedirs(os.path.join(os.path.abspath(repo_path), ".saguaro", "state"), exist_ok=True)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def read_gateway_state(repo_path: str) -> dict[str, Any]:
    path = gateway_state_path(repo_path)
    if not os.path.exists(path):
        return {
            "status": "stopped",
            "socket_path": gateway_socket_path(repo_path),
            "state_path": path,
        }
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle) or {}
    except Exception as exc:
        return {
            "status": "corrupt",
            "socket_path": gateway_socket_path(repo_path),
            "state_path": path,
            "error": str(exc),
        }
    pid = int(payload.get("pid", 0) or 0)
    socket_path = str(payload.get("socket_path") or gateway_socket_path(repo_path))
    running = _pid_alive(pid) and os.path.exists(socket_path)
    payload["status"] = "running" if running else "stopped"
    payload["socket_path"] = socket_path
    payload["state_path"] = path
    return payload


class SessionGovernor:
    """Bounded in-process admission controller used by the resident gateway."""

    def __init__(
        self,
        *,
        active_limit: int,
        queue_limit: int,
        telemetry_cb: Callable[[], None] | None = None,
    ) -> None:
        self.active_limit = max(1, int(active_limit or 1))
        self.queue_limit = max(self.active_limit, int(queue_limit or self.active_limit))
        self.telemetry_cb = telemetry_cb
        self._condition = threading.Condition()
        self._active = 0
        self._queued = 0
        self._rejected = 0

    @property
    def active(self) -> int:
        with self._condition:
            return self._active

    @property
    def queued(self) -> int:
        with self._condition:
            return self._queued

    @property
    def rejected(self) -> int:
        with self._condition:
            return self._rejected

    def acquire(self, *, timeout_seconds: float) -> tuple[bool, str | None]:
        deadline = time.time() + max(0.0, float(timeout_seconds))
        queued_registered = False
        with self._condition:
            if self._queued >= self.queue_limit:
                self._rejected += 1
                self._emit()
                return False, "queue_full"
            self._queued += 1
            queued_registered = True
            self._emit()
            try:
                while self._active >= self.active_limit:
                    remaining = deadline - time.time()
                    if remaining <= 0.0:
                        self._rejected += 1
                        if queued_registered:
                            self._queued = max(0, self._queued - 1)
                            queued_registered = False
                        self._emit()
                        return False, "queue_timeout"
                    self._condition.wait(timeout=remaining)
                self._queued = max(0, self._queued - 1)
                queued_registered = False
                self._active += 1
                self._emit()
                return True, None
            finally:
                if queued_registered:
                    self._queued = max(0, self._queued - 1)
                    self._emit()

    def release(self) -> None:
        with self._condition:
            self._active = max(0, self._active - 1)
            self._emit()
            self._condition.notify()

    def _emit(self) -> None:
        if self.telemetry_cb is not None:
            self.telemetry_cb()


class _GatewayServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 128


class QueryGateway:
    """Resident RPC server for low-latency Saguaro query workloads."""

    def __init__(self, repo_path: str) -> None:
        from saguaro.fastpath import FastCommandAPI

        self.repo_path = os.path.abspath(repo_path)
        self.socket_path = gateway_socket_path(self.repo_path)
        self.state_path = gateway_state_path(self.repo_path)
        profile = load_runtime_profile(self.repo_path)
        selected = dict(profile.get("selected_runtime_layout") or {})
        if not selected:
            profile = calibrate_runtime_profile(self.repo_path)
            selected = dict(profile.get("selected_runtime_layout") or {})
        self._started_at = time.time()
        self._latencies_ms: collections.deque[float] = collections.deque(maxlen=256)
        self._processed = 0
        self._last_error = ""
        self._prewarm_state = "pending"
        self._prewarm_error = ""
        self._api = FastCommandAPI(self.repo_path, use_gateway=False)
        self._state_ledger = StateLedger(self.repo_path)
        self._comparative = ComparativeAnalysisService(
            self.repo_path,
            state_ledger=self._state_ledger,
        )
        self._governor = SessionGovernor(
            active_limit=int(selected.get("max_concurrent_saguaro_sessions", 1) or 1),
            queue_limit=int(
                max(
                    selected.get("queue_depth_target", 1) or 1,
                    selected.get("max_concurrent_saguaro_sessions", 1) or 1,
                )
            ),
            telemetry_cb=self._write_state,
        )
        self._server: _GatewayServer | None = None
        self._queue = Queue()  # or maxsize=queue_limit if you want bounded
        # Start workers (this replaces active_limit)
        for _ in range(self._governor.active_limit):
            threading.Thread(target=self._worker_loop, daemon=True).start()

    def _worker_loop(self):
        while True:
            payload, response_q = self._queue.get()

            started = time.perf_counter()
            try:
                response = self._api.query(
                    text=str(payload.get("text") or ""),
                    k=int(payload.get("k", 5) or 5),
                    file=payload.get("file"),
                    level=int(payload.get("level", 3) or 3),
                    strategy=str(payload.get("strategy") or "hybrid"),
                    explain=bool(payload.get("explain", False)),
                    scope=str(payload.get("scope") or "global"),
                    dedupe_by=str(payload.get("dedupe_by") or "entity"),
                )
                response["gateway"] = read_gateway_state(self.repo_path)
                response_q.put(response)

            except Exception as exc:
                self._last_error = str(exc)
                response_q.put({"status": "error", "message": str(exc)})

            finally:
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                self._latencies_ms.append(elapsed_ms)
                self._processed += 1
                self._queue.task_done()
                self._write_state()

    def _metrics(self) -> dict[str, Any]:
        latencies = sorted(self._latencies_ms)
        def pct(value: float) -> float:
            if not latencies:
                return 0.0
            index = min(len(latencies) - 1, max(0, int(round((value / 100.0) * (len(latencies) - 1)))))
            return round(float(latencies[index]), 3)

        return {
            #"active_sessions": self._governor.active,
            #"queued_sessions": self._governor.queued,
            "active_sessions": self._governor.active,
            "queued_sessions": self._queue.qsize(),
            "rejected_sessions": self._governor.rejected,
            "processed_requests": self._processed,
            "corpus_session_count": int(
                self._state_ledger.list_corpus_sessions().get("count", 0)
            ),
            "p50_ms": pct(50.0),
            "p95_ms": pct(95.0),
            "p99_ms": pct(99.0),
            "last_error": self._last_error,
            "prewarm_state": self._prewarm_state,
            "prewarm_error": self._prewarm_error,
        }

    def _write_state(self) -> None:
        _ensure_state_dir(self.repo_path)
        payload = {
            "status": "running",
            "pid": os.getpid(),
            "repo_path": self.repo_path,
            "socket_path": self.socket_path,
            "started_at": self._started_at,
            "limits": {
                "active_limit": self._governor.active_limit,
                "queue_limit": self._governor.queue_limit,
            },
            "metrics": self._metrics(),
        }
        atomic_write_json(self.state_path, payload, indent=2, sort_keys=True)

    def serve(self) -> None:
        _ensure_state_dir(self.repo_path)
        with contextlib.suppress(FileNotFoundError):
            os.unlink(self.socket_path)
        gateway = self

        class _Handler(socketserver.StreamRequestHandler):
            def handle(self) -> None:
                raw = self.rfile.readline()
                if not raw:
                    return
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except Exception as exc:
                    self._send({"status": "error", "message": str(exc)})
                    return
                response = gateway.handle_request(payload)
                self._send(response)

            def _send(self, payload: dict[str, Any]) -> None:
                blob = (json.dumps(payload) + "\n").encode("utf-8")
                self.wfile.write(blob)

        self._server = _GatewayServer(self.socket_path, _Handler)
        self._write_state()
        threading.Thread(target=self._prewarm_query_runtime, daemon=True).start()

        def _shutdown(_signum: int, _frame: object) -> None:
            if self._server is not None:
                self._server.shutdown()

        signal.signal(signal.SIGTERM, _shutdown)
        signal.signal(signal.SIGINT, _shutdown)
        try:
            self._server.serve_forever()
        finally:
            if self._server is not None:
                self._server.server_close()
            with contextlib.suppress(FileNotFoundError):
                os.unlink(self.socket_path)
            with contextlib.suppress(FileNotFoundError):
                os.remove(self.state_path)

    def handle_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        action = str(payload.get("action") or "status").strip().lower()
        if action == "status":
            self._write_state()
            return {"status": "ok", "gateway": read_gateway_state(self.repo_path)}
        if action == "shutdown":
            threading.Thread(target=self._delayed_shutdown, daemon=True).start()
            return {"status": "ok", "message": "shutdown_requested"}
        if action == "corpus-query":
            response = self._comparative.corpus_query(
                str(payload.get("text") or ""),
                corpus_ids=list(payload.get("corpus_ids") or []),
                k=int(payload.get("k", 5) or 5),
            )
            response["gateway"] = read_gateway_state(self.repo_path)
            return response
        if action != "query":
            return {"status": "error", "message": f"unsupported action: {action}"}

        '''ok, reason = self._governor.acquire(
            timeout_seconds=float(payload.get("timeout_seconds", 30.0) or 30.0)
        )
        if not ok:
            self._write_state()
            return {
                "status": "busy",
                "reason": reason,
                "gateway": read_gateway_state(self.repo_path),
            }
        started = time.perf_counter()
        try:
            response = self._api.query(
                text=str(payload.get("text") or ""),
                k=int(payload.get("k", 5) or 5),
                file=payload.get("file"),
                level=int(payload.get("level", 3) or 3),
                strategy=str(payload.get("strategy") or "hybrid"),
                explain=bool(payload.get("explain", False)),
                scope=str(payload.get("scope") or "global"),
                dedupe_by=str(payload.get("dedupe_by") or "entity"),
            )
            response["gateway"] = read_gateway_state(self.repo_path)
            return response
        except Exception as exc:
            self._last_error = str(exc)
            return {"status": "error", "message": str(exc)}
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self._latencies_ms.append(elapsed_ms)
            self._processed += 1
            self._governor.release()'''
        response_q = Queue(maxsize=1)

        try:
            self._queue.put((payload, response_q), timeout=float(payload.get("timeout_seconds", 30.0)))
        except:
            return {
                "status": "busy",
                "reason": "queue_full",
                "gateway": read_gateway_state(self.repo_path),
            }

        try:
            response = response_q.get(timeout=float(payload.get("timeout_seconds", 30.0)))
            return response
        except:
            return {
                "status": "busy",
                "reason": "timeout",
                "gateway": read_gateway_state(self.repo_path),
            }
            self._write_state()

    def _delayed_shutdown(self) -> None:
        time.sleep(0.05)
        if self._server is not None:
            self._server.shutdown()

    def _prewarm_query_runtime(self) -> None:
        self._prewarm_state = "warming"
        self._write_state()
        try:
            self._api.prime_query_runtime(strategy="hybrid")
        except Exception as exc:
            self._prewarm_state = "error"
            self._prewarm_error = str(exc)
        else:
            self._prewarm_state = "ready"
            self._prewarm_error = ""
        finally:
            self._write_state()


def ensure_gateway_started(repo_path: str, *, wait_seconds: float = 5.0) -> dict[str, Any]:
    repo_root = os.path.abspath(repo_path)
    state = read_gateway_state(repo_root)
    if state.get("status") == "running":
        return state

    _ensure_state_dir(repo_root)
    log_handle = open(gateway_log_path(repo_root), "a", encoding="utf-8")
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "saguaro.query.gateway",
            "--repo",
            repo_root,
        ],
        cwd=repo_root,
        stdout=log_handle,
        stderr=log_handle,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )
    deadline = time.time() + max(0.5, float(wait_seconds))
    while time.time() < deadline:
        state = read_gateway_state(repo_root)
        if state.get("status") == "running":
            return state
        time.sleep(0.05)
    return state


def request_gateway(
    repo_path: str,
    payload: dict[str, Any],
    *,
    start_if_missing: bool = True,
    wait_seconds: float = 5.0,
) -> dict[str, Any]:
    state = (
        ensure_gateway_started(repo_path, wait_seconds=wait_seconds)
        if start_if_missing
        else read_gateway_state(repo_path)
    )
    socket_path = str(state.get("socket_path") or gateway_socket_path(repo_path))
    if state.get("status") != "running" or not os.path.exists(socket_path):
        return {"status": "error", "message": "query_gateway_unavailable", "gateway": state}

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(max(1.0, float(payload.get("timeout_seconds", 30.0) or 30.0) + 1.0))
        sock.connect(socket_path)
        sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))
        chunks: list[bytes] = []
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
            if b"\n" in chunk:
                break
        blob = b"".join(chunks).decode("utf-8").strip()
        return json.loads(blob) if blob else {"status": "error", "message": "empty_response"}
    except Exception as exc:
        return {"status": "error", "message": str(exc), "gateway": state}
    finally:
        with contextlib.suppress(Exception):
            sock.close()


def stop_gateway(repo_path: str) -> dict[str, Any]:
    return request_gateway(
        repo_path,
        {"action": "shutdown", "timeout_seconds": 2.0},
        start_if_missing=False,
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Resident Saguaro query gateway")
    parser.add_argument("--repo", default=".")
    args = parser.parse_args(argv)
    QueryGateway(args.repo).serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
