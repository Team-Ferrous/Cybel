"""Heuristic tensor/data shape propagation for pipeline traces.

The propagator is intentionally conservative: if shape inference is ambiguous,
it leaves shapes as ``None`` and records only high-confidence guesses.
"""

from __future__ import annotations

import os
import re
from typing import Any

from saguaro.analysis.pipeline_tracer import PipelineTrace


class ShapePropagator:
    """Propagates data shapes through pipeline stages."""

    _TENSOR_HINTS = {
        "tensor",
        "ndarray",
        "array",
        "matrix",
        "logits",
        "embedding",
        "hidden",
        "feature",
    }

    def propagate(self, trace: PipelineTrace, source_files: dict[str, str]) -> None:
        """Annotate pipeline stages with inferred data shapes."""
        if not trace.stages:
            return

        stage_by_id = {stage.id: stage for stage in trace.stages}

        for stage in trace.stages:
            inferred_inputs = self._infer_from_type_hints(stage.function_signature)
            inferred_doc = self._infer_from_docstring(stage.description)

            if inferred_inputs and not stage.inputs:
                stage.inputs = inferred_inputs.get("inputs", [])
            if inferred_inputs and not stage.outputs:
                stage.outputs = inferred_inputs.get("outputs", [])

            if inferred_doc:
                if not stage.inputs:
                    stage.inputs = inferred_doc.get("inputs", [])
                if not stage.outputs:
                    stage.outputs = inferred_doc.get("outputs", [])

            body = self._source_slice_for_stage(stage, source_files)
            if body:
                body_shapes = self._infer_tensor_shapes(body)
                if body_shapes and not stage.outputs:
                    stage.outputs = body_shapes

            # Final fallback: always provide typed placeholders for downstream consumers.
            if not stage.inputs:
                stage.inputs = [{"name": "input", "type": "unknown", "shape": None}]
            if not stage.outputs:
                stage.outputs = [{"name": "output", "type": "unknown", "shape": None}]

        # Edge-driven propagation: carry upstream output shape into downstream input.
        for edge in trace.edges:
            src_id = str(edge.get("from") or "")
            dst_id = str(edge.get("to") or "")
            if src_id not in stage_by_id or dst_id not in stage_by_id:
                continue

            src_stage = stage_by_id[src_id]
            dst_stage = stage_by_id[dst_id]
            if not src_stage.outputs or not dst_stage.inputs:
                continue

            src_primary = src_stage.outputs[0]
            dst_primary = dst_stage.inputs[0]
            if not dst_primary.get("shape") and src_primary.get("shape"):
                dst_primary["shape"] = src_primary.get("shape")
                dst_primary.setdefault("propagated_from", src_stage.name)

            relation = str(edge.get("relation") or "")
            if relation in {"calls", "call", "invokes"}:
                maybe_out = self._propagate_through_matmul(dst_stage.inputs)
                if maybe_out and dst_stage.outputs:
                    if not dst_stage.outputs[0].get("shape"):
                        dst_stage.outputs[0]["shape"] = maybe_out.get("shape")

    def _infer_from_type_hints(self, signature: str) -> dict[str, list[dict[str, Any]]] | None:
        """Infer basic input/output types and shapes from a function signature string."""
        text = (signature or "").strip()
        if not text:
            return None

        open_paren = text.find("(")
        close_paren = text.rfind(")")
        if open_paren < 0 or close_paren < open_paren:
            return None

        args_text = text[open_paren + 1 : close_paren]
        return_text = ""
        if "->" in text[close_paren:]:
            return_text = text.split("->", 1)[1].strip()

        inputs: list[dict[str, Any]] = []
        for raw_arg in [arg.strip() for arg in args_text.split(",") if arg.strip()]:
            if raw_arg in {"self", "cls"}:
                continue
            name, arg_type = self._split_arg(raw_arg)
            inputs.append(
                {
                    "name": name,
                    "type": arg_type,
                    "shape": self._shape_from_text(raw_arg),
                }
            )

        outputs: list[dict[str, Any]] = []
        if return_text:
            outputs.append(
                {
                    "name": "return",
                    "type": return_text,
                    "shape": self._shape_from_text(return_text),
                }
            )

        return {"inputs": inputs, "outputs": outputs}

    def _infer_from_docstring(self, docstring: str) -> dict[str, list[dict[str, Any]]] | None:
        """Infer shapes from docstring/description snippets."""
        text = (docstring or "").strip()
        if not text:
            return None

        inputs: list[dict[str, Any]] = []
        outputs: list[dict[str, Any]] = []

        for line in text.splitlines():
            clean = line.strip(" -\t")
            if not clean:
                continue
            shape = self._shape_from_text(clean)
            lower = clean.lower()

            if "input" in lower or "arg" in lower:
                inputs.append(
                    {
                        "name": self._name_from_doc_line(clean, fallback="input"),
                        "type": self._type_from_text(clean),
                        "shape": shape,
                    }
                )
            if "output" in lower or "return" in lower:
                outputs.append(
                    {
                        "name": self._name_from_doc_line(clean, fallback="output"),
                        "type": self._type_from_text(clean),
                        "shape": shape,
                    }
                )

        if not inputs and not outputs:
            # Generic fallback for shape-like docs without explicit headings.
            generic_shapes = re.findall(r"\[[^\]]+\]|\([^\)]+\)", text)
            if generic_shapes:
                outputs.append(
                    {
                        "name": "output",
                        "type": "tensor",
                        "shape": generic_shapes[-1],
                    }
                )

        if not inputs and not outputs:
            return None
        return {"inputs": inputs, "outputs": outputs}

    def _infer_tensor_shapes(self, function_body: str) -> list[dict[str, Any]]:
        """Infer likely output shapes from operation patterns in source text."""
        body = function_body or ""
        if not body.strip():
            return []

        results: list[dict[str, Any]] = []

        # Matmul-like expressions: often [B, M] @ [M, N] -> [B, N]
        if "@" in body or "matmul" in body:
            out = self._propagate_through_matmul(
                [
                    {"shape": self._shape_from_text(body), "type": "tensor"},
                    {"shape": self._shape_from_text(body), "type": "tensor"},
                ]
            )
            if out:
                results.append({"name": "matmul_out", "type": "tensor", "shape": out.get("shape")})

        # Reshape patterns: x.reshape(B, N, D)
        reshape_match = re.search(r"reshape\(([^\)]*)\)", body)
        if reshape_match:
            args = [part.strip() for part in reshape_match.group(1).split(",") if part.strip()]
            reshaped = self._propagate_through_reshape(
                {"shape": None, "type": "tensor"},
                args,
            )
            if reshaped:
                results.append({"name": "reshape_out", "type": "tensor", "shape": reshaped.get("shape")})

        # Attention hints: output usually keeps [batch, seq, dim]
        if "attention" in body.lower() and not results:
            results.append({"name": "attention_out", "type": "tensor", "shape": "[batch, seq, dim]"})

        return results

    def _propagate_through_matmul(self, input_shapes: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Infer output shape for a 2D matmul if dimensions are compatible."""
        if len(input_shapes) < 2:
            return None
        lhs = self._parse_shape_tokens(str(input_shapes[0].get("shape") or ""))
        rhs = self._parse_shape_tokens(str(input_shapes[1].get("shape") or ""))

        if len(lhs) >= 2 and len(rhs) >= 2:
            return {"shape": f"[{lhs[-2]}, {rhs[-1]}]", "type": "tensor"}
        if lhs and rhs:
            return {"shape": f"[{lhs[0]}, {rhs[-1]}]", "type": "tensor"}
        return None

    def _propagate_through_reshape(
        self, input_shape: dict[str, Any], args: list[Any]
    ) -> dict[str, Any] | None:
        """Infer output shape for reshape-like operations."""
        _ = input_shape
        if not args:
            return None
        dims = [str(a).strip() for a in args if str(a).strip()]
        if not dims:
            return None
        return {"shape": f"[{', '.join(dims)}]", "type": "tensor"}

    @staticmethod
    def _split_arg(raw_arg: str) -> tuple[str, str]:
        if ":" not in raw_arg:
            return raw_arg.strip(), "unknown"
        name, type_text = raw_arg.split(":", 1)
        return name.strip(), type_text.strip() or "unknown"

    def _shape_from_text(self, text: str) -> str | None:
        candidate = (text or "").strip()
        if not candidate:
            return None

        match = re.search(r"\[[^\]]+\]", candidate)
        if match:
            return match.group(0)

        match = re.search(r"\(([^\)]*)\)", candidate)
        if match:
            inner = match.group(1)
            if any(token.strip().isalpha() or token.strip().isdigit() for token in inner.split(",")):
                return f"[{inner}]"

        lowered = candidate.lower()
        if any(hint in lowered for hint in self._TENSOR_HINTS):
            if "sequence" in lowered or "seq" in lowered:
                return "[batch, seq, dim]"
            if "image" in lowered or "conv" in lowered:
                return "[batch, channels, height, width]"
            return "[n, d]"
        return None

    @staticmethod
    def _parse_shape_tokens(shape: str) -> list[str]:
        text = (shape or "").strip()
        if not text:
            return []
        text = text.strip("[]()")
        return [token.strip() for token in text.split(",") if token.strip()]

    @staticmethod
    def _name_from_doc_line(line: str, fallback: str) -> str:
        token = re.split(r"[:\-]", line, maxsplit=1)[0].strip()
        token = re.sub(r"\s+", "_", token.lower())
        token = re.sub(r"[^a-z0-9_]+", "", token)
        return token or fallback

    @staticmethod
    def _type_from_text(text: str) -> str:
        lower = (text or "").lower()
        if "tensor" in lower or "ndarray" in lower:
            return "tensor"
        if "list" in lower:
            return "list"
        if "dict" in lower:
            return "dict"
        if "str" in lower or "text" in lower:
            return "str"
        return "unknown"

    def _source_slice_for_stage(self, stage: Any, source_files: dict[str, str]) -> str:
        file_path = str(getattr(stage, "file_path", "") or "")
        if not file_path:
            return ""

        content = source_files.get(file_path)
        if content is None:
            abs_path = file_path if os.path.isabs(file_path) else os.path.abspath(file_path)
            try:
                with open(abs_path, "r", encoding="utf-8", errors="ignore") as handle:
                    content = handle.read()
            except Exception:
                return ""

        lines = content.splitlines()
        start = max(1, int(getattr(stage, "start_line", 1) or 1))
        end = max(start, int(getattr(stage, "end_line", start) or start))
        return "\n".join(lines[start - 1 : min(end, len(lines))])


__all__ = ["ShapePropagator"]
