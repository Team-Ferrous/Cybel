"""Formatting utilities for pipeline traces."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import os
from typing import Any

from saguaro.analysis.pipeline_tracer import PipelineTrace


class TraceOutputFormatter:
    """Formats pipeline traces for agent and CLI consumption."""

    def to_json(self, trace: PipelineTrace) -> str:
        """Serialize a pipeline trace to JSON."""
        payload = self._trace_to_dict(trace)
        return json.dumps(payload, indent=2, sort_keys=False)

    def to_mermaid(self, trace: PipelineTrace) -> str:
        """Render a Mermaid DAG representation of a pipeline trace."""
        lines: list[str] = ["graph LR"]

        for stage in trace.stages:
            title = self._escape_mermaid(stage.name)
            comp = ""
            if isinstance(stage.complexity, dict):
                comp = str(stage.complexity.get("time") or "")
            file_hint = self._escape_mermaid(stage.file_path)
            label_parts = [title]
            if comp:
                label_parts.append(comp)
            if file_hint:
                label_parts.append(file_hint)
            label = "\\n".join(part for part in label_parts if part)
            lines.append(f'    {self._node_token(stage.id)}["{label}"]')

        for edge in trace.edges:
            src = self._node_token(str(edge.get("from") or ""))
            dst = self._node_token(str(edge.get("to") or ""))
            data = self._escape_mermaid(
                str(
                    edge.get("label")
                    or edge.get("condition")
                    or edge.get("data")
                    or edge.get("relation")
                    or ""
                )
            )
            if not src or not dst:
                continue
            if data:
                lines.append(f'    {src} -->|"{data}"| {dst}')
            else:
                lines.append(f"    {src} --> {dst}")

        for idx, boundary in enumerate(trace.ffi_boundaries):
            host_id = self._node_token(str(boundary.get("host_stage_id") or ""))
            guest_id = self._node_token(str(boundary.get("guest_stage_id") or f"ffi_{idx}"))
            mech = self._escape_mermaid(str(boundary.get("mechanism") or "ffi"))
            if host_id and guest_id:
                lines.append(f'    {host_id} -.->|"{mech}"| {guest_id}')

        return "\n".join(lines)

    def to_markdown_table(self, trace: PipelineTrace) -> str:
        """Render a markdown table suitable for reports and PR comments."""
        header = [
            "| # | Stage | File | Lines | Inputs | Outputs | Time | FFI |",
            "|---:|---|---|---:|---|---|---|---|",
        ]
        rows: list[str] = []
        for stage in trace.stages:
            inputs = self._io_summary(stage.inputs)
            outputs = self._io_summary(stage.outputs)
            time_comp = ""
            if isinstance(stage.complexity, dict):
                time_comp = str(stage.complexity.get("time") or "")
            rows.append(
                "| {idx} | {name} | {file} | {start}-{end} | {inp} | {out} | {time} | {ffi} |".format(
                    idx=stage.stage_index,
                    name=self._md_escape(stage.name),
                    file=self._md_escape(stage.file_path),
                    start=stage.start_line,
                    end=stage.end_line,
                    inp=self._md_escape(inputs),
                    out=self._md_escape(outputs),
                    time=self._md_escape(time_comp),
                    ffi="yes" if stage.calls_ffi else "no",
                )
            )

        return "\n".join(header + rows)

    def to_compact(self, trace: PipelineTrace) -> str:
        """Render a token-efficient one-line-per-stage representation."""
        stage_chunks: list[str] = []
        for stage in trace.stages:
            comp = ""
            if isinstance(stage.complexity, dict) and stage.complexity.get("time"):
                comp = f"[{stage.complexity['time']}]"
            ffi = "{ffi}" if stage.calls_ffi else ""
            stage_chunks.append(f"{stage.stage_index}:{stage.name}{comp}{ffi}")

        edge_chunks: list[str] = []
        for edge in trace.edges:
            edge_chunks.append(
                f"{self._short_id(str(edge.get('from') or ''))}->{self._short_id(str(edge.get('to') or ''))}"
            )

        langs = ",".join(trace.languages_involved)
        return (
            f"pipeline={trace.name};entry={trace.entry_point};langs={langs};"
            f"stages={' | '.join(stage_chunks)};edges={' | '.join(edge_chunks)}"
        )

    def to_html(self, trace: PipelineTrace, *, interactive: bool = True) -> str:
        """Render a trace as HTML; optionally includes interactive explorer UI."""
        payload = self._trace_to_dict(trace)
        mermaid = self.to_mermaid(trace)
        if interactive:
            try:
                from saguaro.analysis.interactive_trace_exporter import InteractiveTraceExporter

                return InteractiveTraceExporter().render_html(
                    trace_payload=payload,
                    mermaid=mermaid,
                    title=trace.name,
                )
            except Exception:
                pass

        trace_json = json.dumps(payload, indent=2, sort_keys=False)
        return (
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<title>Pipeline Trace</title></head><body>"
            f"<h1>{self._md_escape(trace.name)}</h1>"
            "<h2>Mermaid</h2><pre>"
            f"{self._md_escape(mermaid)}"
            "</pre><h2>JSON</h2><pre>"
            f"{self._md_escape(trace_json)}"
            "</pre></body></html>"
        )

    def export(
        self,
        trace: PipelineTrace,
        *,
        fmt: str = "json",
        output_path: str | None = None,
        interactive: bool = False,
    ) -> str:
        """Export trace output in ``json``, ``mermaid``, ``markdown``, ``compact``, or ``html``."""
        target_fmt = str(fmt or "json").strip().lower()
        if target_fmt in {"json"}:
            content = self.to_json(trace)
        elif target_fmt in {"mermaid", "mmd"}:
            content = self.to_mermaid(trace)
        elif target_fmt in {"markdown", "md", "table"}:
            content = self.to_markdown_table(trace)
        elif target_fmt in {"compact"}:
            content = self.to_compact(trace)
        elif target_fmt in {"html", "interactive_html"}:
            content = self.to_html(trace, interactive=interactive or target_fmt == "interactive_html")
        else:
            raise ValueError(f"Unsupported trace export format: {fmt}")

        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as handle:
                handle.write(content)
            return output_path
        return content

    def _trace_to_dict(self, trace: PipelineTrace) -> dict[str, Any]:
        if is_dataclass(trace):
            return asdict(trace)
        # Fallback for duck-typed pipeline traces.
        return {
            "name": getattr(trace, "name", ""),
            "entry_point": getattr(trace, "entry_point", ""),
            "stages": [self._stage_to_dict(item) for item in getattr(trace, "stages", [])],
            "edges": list(getattr(trace, "edges", []) or []),
            "total_complexity": getattr(trace, "total_complexity", None),
            "languages_involved": list(getattr(trace, "languages_involved", []) or []),
            "ffi_boundaries": list(getattr(trace, "ffi_boundaries", []) or []),
            "warnings": list(getattr(trace, "warnings", []) or []),
        }

    @staticmethod
    def _stage_to_dict(stage: Any) -> dict[str, Any]:
        if is_dataclass(stage):
            return asdict(stage)
        return {
            "id": getattr(stage, "id", ""),
            "name": getattr(stage, "name", ""),
            "file_path": getattr(stage, "file_path", ""),
            "start_line": getattr(stage, "start_line", 0),
            "end_line": getattr(stage, "end_line", 0),
            "language": getattr(stage, "language", ""),
            "function_signature": getattr(stage, "function_signature", ""),
            "description": getattr(stage, "description", ""),
            "inputs": list(getattr(stage, "inputs", []) or []),
            "outputs": list(getattr(stage, "outputs", []) or []),
            "complexity": getattr(stage, "complexity", None),
            "children": list(getattr(stage, "children", []) or []),
            "calls_ffi": bool(getattr(stage, "calls_ffi", False)),
            "stage_index": int(getattr(stage, "stage_index", 0) or 0),
            "annotations": list(getattr(stage, "annotations", []) or []),
        }

    @staticmethod
    def _escape_mermaid(text: str) -> str:
        return (text or "").replace('"', "'")

    @staticmethod
    def _node_token(raw: str) -> str:
        token = "".join(ch if ch.isalnum() else "_" for ch in (raw or ""))
        if token and token[0].isdigit():
            token = f"n_{token}"
        return token

    @staticmethod
    def _short_id(raw: str) -> str:
        text = raw or ""
        if "::" in text:
            return text.split("::", 1)[1]
        return text

    @staticmethod
    def _io_summary(items: list[dict[str, Any]]) -> str:
        if not items:
            return "-"
        parts = []
        for item in items[:3]:
            name = str(item.get("name") or "?")
            shape = str(item.get("shape") or "").strip()
            if shape:
                parts.append(f"{name}:{shape}")
            else:
                parts.append(name)
        if len(items) > 3:
            parts.append("...")
        return ", ".join(parts)

    @staticmethod
    def _md_escape(text: str) -> str:
        return (text or "").replace("|", "\\|").replace("\n", " ")


__all__ = ["TraceOutputFormatter"]
