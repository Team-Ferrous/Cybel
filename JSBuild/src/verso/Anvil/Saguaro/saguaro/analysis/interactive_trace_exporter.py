"""Interactive HTML exporter for pipeline trace payloads."""

from __future__ import annotations

import json
import os
from typing import Any


class InteractiveTraceExporter:
    """Exports trace payloads to a self-contained interactive HTML view."""

    def render_html(
        self,
        *,
        trace_payload: dict[str, Any],
        mermaid: str = "",
        title: str = "Pipeline Trace",
    ) -> str:
        payload_json = json.dumps(trace_payload, indent=2, sort_keys=False)
        stages = trace_payload.get("stages") or []
        stage_rows = "\n".join(
            f"<tr data-stage='{self._escape_html(str(stage.get('name') or ''))}'>"
            f"<td>{int(stage.get('stage_index') or 0)}</td>"
            f"<td>{self._escape_html(str(stage.get('name') or ''))}</td>"
            f"<td>{self._escape_html(str(stage.get('file_path') or ''))}</td>"
            f"<td>{self._escape_html(str((stage.get('complexity') or {}).get('time') or ''))}</td>"
            f"<td>{self._escape_html(', '.join(stage.get('annotations') or []))}</td>"
            "</tr>"
            for stage in stages
            if isinstance(stage, dict)
        )

        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{self._escape_html(title)}</title>
  <style>
    :root {{
      --bg: #f5f6f8;
      --panel: #ffffff;
      --text: #16212f;
      --muted: #5f6b7a;
      --line: #d7dde5;
      --accent: #1764b0;
    }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background: radial-gradient(circle at 12% 8%, #dfeaf9, transparent 48%), var(--bg);
      color: var(--text);
    }}
    .wrap {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 20px;
      display: grid;
      gap: 14px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 14px;
      box-shadow: 0 6px 24px rgba(12, 30, 52, 0.06);
    }}
    h1 {{
      margin: 0 0 10px 0;
      font-size: 1.3rem;
    }}
    .meta {{
      color: var(--muted);
      font-size: 0.9rem;
      margin-bottom: 8px;
    }}
    input {{
      width: 100%;
      max-width: 380px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px 10px;
      font-size: 0.95rem;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
      font-size: 0.9rem;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      text-align: left;
      padding: 8px 6px;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-weight: 600;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      background: #f8fbff;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      max-height: 420px;
      overflow: auto;
      font-size: 0.82rem;
    }}
    .muted {{
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <main class="wrap">
    <section class="card">
      <h1>{self._escape_html(title)}</h1>
      <div class="meta">
        entry={self._escape_html(str(trace_payload.get("entry_point") or ""))},
        stages={len(stages)},
        languages={self._escape_html(', '.join(trace_payload.get("languages_involved") or []))}
      </div>
      <input id="stageFilter" placeholder="Filter stages by name..." />
      <table id="stageTable">
        <thead>
          <tr><th>#</th><th>Stage</th><th>File</th><th>Time</th><th>Annotations</th></tr>
        </thead>
        <tbody>
          {stage_rows}
        </tbody>
      </table>
    </section>
    <section class="card">
      <h2>Mermaid</h2>
      <p class="muted">Rendered as text for portability.</p>
      <pre>{self._escape_html(mermaid)}</pre>
    </section>
    <section class="card">
      <h2>Trace JSON</h2>
      <pre>{self._escape_html(payload_json)}</pre>
    </section>
  </main>
  <script>
    (function () {{
      var input = document.getElementById("stageFilter");
      var rows = Array.prototype.slice.call(document.querySelectorAll("#stageTable tbody tr"));
      function filter() {{
        var q = (input.value || "").toLowerCase().trim();
        rows.forEach(function(row) {{
          var stage = (row.getAttribute("data-stage") || "").toLowerCase();
          row.style.display = !q || stage.indexOf(q) >= 0 ? "" : "none";
        }});
      }}
      input.addEventListener("input", filter);
    }})();
  </script>
</body>
</html>
"""

    def export(
        self,
        *,
        trace_payload: dict[str, Any],
        output_path: str,
        mermaid: str = "",
        title: str = "Pipeline Trace",
    ) -> str:
        html = self.render_html(trace_payload=trace_payload, mermaid=mermaid, title=title)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(html)
        return output_path

    @staticmethod
    def _escape_html(text: str) -> str:
        return (
            str(text or "")
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )


__all__ = ["InteractiveTraceExporter"]
