"""Math pipeline enrichment for loops, access signatures, and provenance."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re

from saguaro.math.ir import AccessSignature
from saguaro.math.ir import ComplexityReductionHint
from saguaro.math.ir import LayoutState
from saguaro.math.ir import LoopFrame
from saguaro.math.ir import MathIRRecord
from saguaro.math.languages import LanguagePolicy

_ARRAY_ACCESS_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\[([^\]]+)\]")
_ASSIGN_SPLIT_RE = re.compile(r"(?<![=!<>])=(?!=)")
_FOR_LOOP_RE = re.compile(r"\bfor\s*\(([^;]*);([^;]*);([^)]+)\)")
_PYTHON_LOOP_RE = re.compile(r"^\s*for\s+([A-Za-z_][A-Za-z0-9_]*)\s+in\s+(.+):")
_SCALAR_LOOP_RE = re.compile(r"^\s*for\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)")
_FORTRAN_LOOP_RE = re.compile(r"^\s*do\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)", re.I)
_SYMBOL_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NATIVE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".m",
    ".mm",
}
_PYTHON_SUFFIXES = {".py", ".pyi", ".pyx", ".pxd"}


class MathPipeline:
    """Attach execution-shape metadata to extracted math records."""

    def __init__(self, repo_path: str) -> None:
        self.repo_path = Path(repo_path).resolve()
        self._build_targets = self._load_build_targets()

    def enrich_records(
        self,
        records: list[MathIRRecord],
        *,
        text: str,
        source_path: str,
        policy: LanguagePolicy,
    ) -> list[MathIRRecord]:
        lines = text.splitlines()
        for record in records:
            loop_context = self._loop_context(lines, record.line_start, policy)
            record.loop_context = loop_context
            record.access_signatures = self._access_signatures(record, loop_context)
            record.layout_states = self._layout_states(record.access_signatures)
            record.complexity_reduction_hints = self._complexity_reduction_hints(record)
            record.provenance = self._provenance(record, source_path)
        return records

    def _loop_context(
        self,
        lines: list[str],
        line_start: int,
        policy: LanguagePolicy,
    ) -> LoopFrame | None:
        window_start = max(0, line_start - 12)
        candidates = lines[window_start : max(0, line_start - 1)]
        loop_lines = [
            line
            for line in candidates
            if any(re.search(rf"^\s*{re.escape(keyword)}\b", line, re.I) for keyword in policy.loop_keywords)
        ]
        if not loop_lines:
            return None
        loop_line = loop_lines[-1]
        loop_kind = "loop"
        loop_variables: list[str] = []
        bounds_hint = ""
        if match := _FOR_LOOP_RE.search(loop_line):
            loop_kind = "for"
            header = " ".join(match.groups())
            loop_variables = _SYMBOL_RE.findall(match.group(1))
            bounds_hint = match.group(2).strip()
        elif match := _PYTHON_LOOP_RE.search(loop_line):
            loop_kind = "for"
            loop_variables = [match.group(1)]
            bounds_hint = match.group(2).strip()
        elif match := _SCALAR_LOOP_RE.search(loop_line):
            loop_kind = "for"
            loop_variables = [match.group(1)]
            bounds_hint = match.group(2).strip()
        elif match := _FORTRAN_LOOP_RE.search(loop_line):
            loop_kind = "do"
            loop_variables = [match.group(1)]
            bounds_hint = match.group(2).strip()
        else:
            lowered = loop_line.strip().lower()
            loop_kind = next(
                (keyword for keyword in policy.loop_keywords if lowered.startswith(keyword)),
                "loop",
            )
        return LoopFrame(
            loop_kind=loop_kind,
            loop_variables=sorted(set(loop_variables)),
            nesting_depth=len(loop_lines),
            bounds_hint=bounds_hint,
        )

    def _access_signatures(
        self,
        record: MathIRRecord,
        loop_context: LoopFrame | None,
    ) -> list[AccessSignature]:
        loop_vars = set(loop_context.loop_variables if loop_context else [])
        lhs = record.lhs.strip()
        rhs = record.rhs.strip()
        signatures: list[AccessSignature] = []
        seen: set[tuple[str, str, str]] = set()
        for match in _ARRAY_ACCESS_RE.finditer(record.expression):
            base_symbol = match.group(1)
            index_expression = " ".join(match.group(2).split())
            access_kind = "write" if lhs.startswith(base_symbol) and match.start() <= len(lhs) else "read"
            stride_class = self._stride_class(index_expression, loop_vars)
            index_affinity = "loop_bound" if any(var in index_expression for var in loop_vars) else "symbolic"
            reuse_hint = self._reuse_hint(base_symbol, stride_class, record.expression)
            write_mode = "overwrite" if access_kind == "write" and base_symbol not in rhs else "accumulate"
            key = (base_symbol, index_expression, access_kind)
            if key in seen:
                continue
            seen.add(key)
            signatures.append(
                AccessSignature(
                    base_symbol=base_symbol,
                    access_kind=access_kind,
                    index_expression=index_expression,
                    index_affinity=index_affinity,
                    stride_class=stride_class,
                    reuse_hint=reuse_hint,
                    write_mode=write_mode,
                )
            )
        if lhs and not signatures:
            signatures.append(
                AccessSignature(
                    base_symbol=lhs.split("[", 1)[0].strip(),
                    access_kind="write",
                    index_expression="",
                    index_affinity="scalar",
                    stride_class="scalar",
                    reuse_hint="register_resident",
                    write_mode="overwrite" if lhs not in rhs else "accumulate",
                )
            )
        self._annotate_reduction(record, signatures)
        return signatures

    @staticmethod
    def _annotate_reduction(record: MathIRRecord, signatures: list[AccessSignature]) -> None:
        lhs_symbol = record.lhs.split("[", 1)[0].strip()
        recurrence = bool(lhs_symbol and lhs_symbol in record.rhs)
        reduction = record.statement_kind == "compound_assignment" or recurrence
        if record.loop_context is not None:
            record.loop_context.recurrence = recurrence
            record.loop_context.reduction = reduction
            record.loop_context.reduction_symbol = lhs_symbol if reduction else ""
        if recurrence and not any(item.write_mode == "accumulate" for item in signatures):
            for item in signatures:
                if item.access_kind == "write":
                    item.write_mode = "accumulate"

    @staticmethod
    def _stride_class(index_expression: str, loop_vars: set[str]) -> str:
        normalized = index_expression.replace(" ", "")
        if not normalized:
            return "scalar"
        if "(" in normalized and ")" in normalized:
            return "indirect"
        if loop_vars and any(var in normalized for var in loop_vars):
            if re.search(r"[*/%]", normalized):
                return "strided"
            if re.search(r"[+\-]", normalized):
                return "contiguous_offset"
            return "contiguous"
        if _SYMBOL_RE.search(normalized) and not normalized.isdigit():
            return "indirect"
        return "scalar"

    @staticmethod
    def _reuse_hint(base_symbol: str, stride_class: str, expression: str) -> str:
        appearances = expression.count(base_symbol)
        if appearances >= 3:
            return "temporal_reuse"
        if stride_class in {"contiguous", "contiguous_offset"}:
            return "streaming"
        if stride_class == "strided":
            return "cache_sensitive"
        if stride_class == "indirect":
            return "gather_scatter"
        return "register_resident"

    @staticmethod
    def _layout_states(signatures: list[AccessSignature]) -> list[LayoutState]:
        grouped: dict[str, list[AccessSignature]] = defaultdict(list)
        for item in signatures:
            grouped[item.base_symbol].append(item)
        states: list[LayoutState] = []
        for symbol, items in sorted(grouped.items()):
            stride_classes = {item.stride_class for item in items}
            if stride_classes <= {"contiguous", "contiguous_offset", "scalar"}:
                layout = "packed_contiguous"
                contiguous = True
                alias_risk = "low"
            elif "indirect" in stride_classes:
                layout = "gather_scatter"
                contiguous = False
                alias_risk = "high"
            else:
                layout = "strided"
                contiguous = False
                alias_risk = "medium"
            states.append(
                LayoutState(
                    symbol=symbol,
                    layout=layout,
                    contiguous=contiguous,
                    alias_risk=alias_risk,
                )
            )
        return states

    @staticmethod
    def _complexity_reduction_hints(
        record: MathIRRecord,
    ) -> list[ComplexityReductionHint]:
        expression = record.expression.strip()
        lowered = expression.lower()
        hints: list[ComplexityReductionHint] = []
        seen: set[str] = set()

        def add_hint(
            *,
            kind: str,
            summary: str,
            estimated_cost_delta: float,
            confidence: float,
            safe: bool = True,
        ) -> None:
            if kind in seen:
                return
            seen.add(kind)
            hints.append(
                ComplexityReductionHint(
                    kind=kind,
                    summary=summary,
                    estimated_cost_delta=estimated_cost_delta,
                    confidence=confidence,
                    safe=safe,
                )
            )

        if record.loop_context and record.loop_context.reduction:
            add_hint(
                kind="tree_reduce",
                summary="Reformulate scalar accumulation as a tree reduction to lower recurrence pressure.",
                estimated_cost_delta=0.18,
                confidence=0.88,
            )

        if expression.count("max(") + expression.count("min(") >= 2 or lowered.count("std::max") + lowered.count("std::min") >= 2:
            add_hint(
                kind="clamp_simplify",
                summary="Collapse repeated clamp/max/min bookkeeping into a single bounded helper or hoisted clamp.",
                estimated_cost_delta=0.12,
                confidence=0.71,
            )

        call_names = re.findall(r"([A-Za-z_][A-Za-z0-9_:]*)\s*\(", expression)
        repeated_calls = {
            name
            for name in call_names
            if call_names.count(name) > 1 and name not in {"if", "for", "while"}
        }
        if repeated_calls:
            add_hint(
                kind="common_subexpression",
                summary="Hoist repeated function calls or shared scalar terms out of the expression.",
                estimated_cost_delta=0.1,
                confidence=0.67,
            )

        literal_ops = re.findall(r"(?<![A-Za-z0-9_])(?:[0-9]+(?:\.[0-9]+)?f?)(?![A-Za-z0-9_])", expression)
        if len(literal_ops) >= 2 and any(op in expression for op in ("*", "/")):
            add_hint(
                kind="scalar_precompute",
                summary="Precompute repeated scalar coefficients or normalization factors outside the hot path.",
                estimated_cost_delta=0.08,
                confidence=0.63,
            )
        return hints

    def _provenance(self, record: MathIRRecord, source_path: str) -> dict[str, object]:
        rel_path = source_path.replace("\\", "/")
        build_targets = self._build_targets_for_file(rel_path)
        path_lower = rel_path.lower()
        suffix = Path(path_lower).suffix.lower()
        if suffix in _NATIVE_SUFFIXES:
            execution_domain = "native_kernel"
        elif suffix in _PYTHON_SUFFIXES:
            execution_domain = "python_orchestration"
        else:
            execution_domain = "mixed_source"
        return {
            "build_targets": build_targets,
            "execution_domain": execution_domain,
            "ffi_boundary": "native" if execution_domain == "native_kernel" else "python",
            "hotspot_hint": record.complexity.band if record.complexity is not None else "low",
        }

    def _load_build_targets(self) -> dict[str, list[str]]:
        cmake_path = self.repo_path / "core" / "native" / "CMakeLists.txt"
        if not cmake_path.exists():
            return {}
        text = cmake_path.read_text(encoding="utf-8", errors="ignore")
        target_map: dict[str, list[str]] = defaultdict(list)
        current_target = ""
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if match := re.match(r"(?:add_library|add_executable)\s*\(\s*([A-Za-z0-9_]+)", line):
                current_target = match.group(1)
                continue
            if not current_target:
                continue
            for source in re.findall(
                r"([A-Za-z0-9_./-]+\.(?:cpp|cxx|cc|hpp|hxx|hh|py|c|h))",
                line,
            ):
                target_map[source].append(current_target)
        return {key: sorted(set(value)) for key, value in target_map.items()}

    def _build_targets_for_file(self, rel_path: str) -> list[str]:
        basename = Path(rel_path).name
        matches: list[str] = []
        for source, targets in self._build_targets.items():
            if source.endswith(rel_path) or Path(source).name == basename:
                matches.extend(targets)
        return sorted(set(matches))
