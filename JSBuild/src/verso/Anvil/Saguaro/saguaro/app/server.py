"""Minimal local HTTP app for Saguaro repository intelligence."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from saguaro.api import SaguaroAPI

APP_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Saguaro Local App</title>
  <style>
    :root { color-scheme: light; --bg:#f3f1ea; --panel:#fffdf8; --line:#d4cab8; --ink:#1f1d18; --muted:#72695b; --accent:#285943; }
    body { margin:0; font-family: "IBM Plex Sans", "Segoe UI", sans-serif; background:linear-gradient(180deg,#f6f0e6 0%,#ede9df 100%); color:var(--ink); }
    header { padding:24px 28px 12px; }
    h1 { margin:0; font-size:28px; letter-spacing:-0.04em; }
    p { color:var(--muted); max-width:72ch; }
    main { display:grid; gap:16px; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); padding:0 28px 28px; }
    section { background:var(--panel); border:1px solid var(--line); border-radius:18px; padding:16px; box-shadow:0 8px 24px rgba(55,45,25,.06); }
    h2 { margin:0 0 10px; font-size:15px; text-transform:uppercase; letter-spacing:.08em; color:var(--accent); }
    pre { white-space:pre-wrap; word-break:break-word; font-size:12px; background:#faf7f1; border-radius:12px; padding:12px; border:1px solid #e4ddcf; }
    input, button, select { font:inherit; }
    .querybar { display:flex; gap:8px; flex-wrap:wrap; }
    .querybar input, .querybar select { flex:1; min-width:160px; border:1px solid var(--line); border-radius:10px; padding:10px 12px; background:#fff; }
    button { border:0; border-radius:10px; padding:10px 14px; background:var(--accent); color:#fff; cursor:pointer; }
  </style>
</head>
<body>
  <header>
    <h1>Saguaro Local Command Center</h1>
    <p>CPU-first repository graph, query, health, verification, and evidence view for the current workspace.</p>
  </header>
  <main>
    <section>
      <h2>Health</h2>
      <pre id="health">Loading…</pre>
    </section>
    <section>
      <h2>Dashboard</h2>
      <pre id="dashboard">Loading…</pre>
    </section>
    <section>
      <h2>Graph</h2>
      <pre id="graph">Loading…</pre>
    </section>
    <section>
      <h2>Query</h2>
      <div class="querybar">
        <input id="queryInput" placeholder="Search by concept, symbol, policy, or subsystem">
        <select id="queryStrategy">
          <option value="hybrid">hybrid</option>
          <option value="symbol">symbol</option>
          <option value="graph">graph</option>
          <option value="semantic">semantic</option>
          <option value="search-by-impact">search-by-impact</option>
          <option value="search-by-policy">search-by-policy</option>
          <option value="search-by-roadmap">search-by-roadmap</option>
        </select>
        <button onclick="runQuery()">Search</button>
      </div>
      <pre id="queryOut">Enter a query.</pre>
    </section>
    <section>
      <h2>Evidence</h2>
      <pre id="evidence">Loading…</pre>
    </section>
    <section>
      <h2>Research</h2>
      <pre id="research">Loading…</pre>
    </section>
    <section>
      <h2>Campaign</h2>
      <pre id="campaign">Loading…</pre>
    </section>
  </main>
  <script>
    async function loadJson(id, path) {
      const res = await fetch(path);
      const data = await res.json();
      document.getElementById(id).textContent = JSON.stringify(data, null, 2);
    }
    async function runQuery() {
      const text = document.getElementById("queryInput").value.trim();
      const strategy = document.getElementById("queryStrategy").value;
      if (!text) return;
      const res = await fetch(`/query?text=${encodeURIComponent(text)}&strategy=${encodeURIComponent(strategy)}&explain=1&k=8`);
      const data = await res.json();
      document.getElementById("queryOut").textContent = JSON.stringify(data, null, 2);
    }
    loadJson("health", "/health");
    loadJson("dashboard", "/app/dashboard");
    loadJson("graph", "/graph?action=query&depth=1&limit=12");
    loadJson("evidence", "/evidence");
    loadJson("research", "/research");
    loadJson("campaign", "/campaign");
  </script>
</body>
</html>
"""


class _SaguaroHandler(BaseHTTPRequestHandler):
    """HTTP handler backed by SaguaroAPI."""

    api: SaguaroAPI

    def _json(self, payload: dict[str, Any], status: int = 200) -> None:
        blob = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:
            return {}

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        params = {key: values[-1] for key, values in parse_qs(parsed.query).items()}
        if parsed.path in {"/", "/app"}:
            body = APP_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        try:
            if parsed.path == "/health":
                self._json(self.api.health())
            elif parsed.path == "/query":
                self._json(
                    self.api.query(
                        text=params.get("text", ""),
                        k=int(params.get("k", "5")),
                        file=params.get("file"),
                        strategy=params.get("strategy", "hybrid"),
                        explain=params.get("explain", "0").lower() in {"1", "true", "yes"},
                    )
                )
            elif parsed.path == "/graph":
                action = params.get("action", "query")
                if action == "build":
                    self._json(self.api.graph_build(path=params.get("path", ".")))
                elif action == "export":
                    self._json(self.api.graph_export())
                else:
                    self._json(
                        self.api.graph_query(
                            symbol=params.get("symbol"),
                            file=params.get("file"),
                            relation=params.get("relation"),
                            depth=int(params.get("depth", "1")),
                            limit=int(params.get("limit", "50")),
                        )
                    )
            elif parsed.path == "/impact":
                self._json(self.api.impact(path=params.get("path", ".")))
            elif parsed.path == "/verify":
                self._json(
                    self.api.verify(
                        path=params.get("path", "."),
                        engines=params.get("engines"),
                        evidence_bundle=params.get("evidence_bundle", "0").lower()
                        in {"1", "true", "yes"},
                        min_parser_coverage=(
                            float(params["min_parser_coverage"])
                            if "min_parser_coverage" in params
                            else None
                        ),
                    )
                )
            elif parsed.path == "/evidence":
                self._json({"bundles": self.api.evidence_list()})
            elif parsed.path == "/research":
                self._json({"entries": self.api.research_list()})
            elif parsed.path == "/metrics":
                self._json({"runs": self.api.metrics_list()})
            elif parsed.path == "/eval":
                self._json(
                    self.api.eval_run(
                        suite=params.get("suite", "cpu_perf"),
                        k=int(params.get("k", "5")),
                        limit=int(params.get("limit", "8")),
                    )
                )
            elif parsed.path == "/campaign":
                self._json(self.api.app_dashboard().get("campaign", {}))
            elif parsed.path == "/app/dashboard":
                self._json(self.api.app_dashboard())
            else:
                self._json({"error": "not_found", "path": parsed.path}, status=404)
        except Exception as exc:  # pragma: no cover - defensive HTTP boundary
            self._json({"error": str(exc), "path": parsed.path}, status=500)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        payload = self._read_json_body()
        try:
            if parsed.path == "/research":
                self._json(
                    self.api.research_ingest(
                        source=payload.get("source", "web"),
                        manifest_path=payload.get("manifest_path"),
                        records=payload.get("records"),
                    )
                )
            elif parsed.path == "/eval":
                self._json(
                    self.api.eval_run(
                        suite=payload.get("suite", "cpu_perf"),
                        k=int(payload.get("k", 5)),
                        limit=int(payload.get("limit", 8)),
                    )
                )
            elif parsed.path == "/verify":
                self._json(self.api.verify(**payload))
            else:
                self._json({"error": "not_found", "path": parsed.path}, status=404)
        except Exception as exc:  # pragma: no cover - defensive HTTP boundary
            self._json({"error": str(exc), "path": parsed.path}, status=500)


def run_server(
    repo_path: str = ".",
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    """Run a local HTTP daemon backed by SaguaroAPI."""

    api = SaguaroAPI(repo_path=repo_path)
    handler = type("SaguaroHandler", (_SaguaroHandler,), {"api": api})
    httpd = ThreadingHTTPServer((host, int(port)), handler)
    print(f"Saguaro local server listening on http://{host}:{int(port)}")
    httpd.serve_forever()
