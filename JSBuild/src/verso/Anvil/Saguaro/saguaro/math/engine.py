"""Parse mathematical content from repo sources and map it into Saguaro state."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from saguaro.math.ir import MathComplexity
from saguaro.math.ir import MathIRRecord
from saguaro.math.languages import LanguagePolicy
from saguaro.math.languages import policy_for
from saguaro.math.languages import supported_languages
from saguaro.math.pipeline import MathPipeline
from saguaro.omnigraph.store import OmniGraphStore
from saguaro.parsing.parser import SAGUAROParser

_MARKDOWN_EXTENSIONS = {".md", ".mdx"}
_SUPPORTED_SUFFIXES = {
    ".md",
    ".mdx",
    ".py",
    ".pyi",
    ".pyx",
    ".pxd",
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
    ".go",
    ".rs",
    ".java",
    ".scala",
    ".sc",
    ".sol",
    ".vy",
    ".sv",
    ".svh",
    ".v",
    ".vhd",
    ".vhdl",
    ".hdl",
    ".f",
    ".f90",
    ".f95",
    ".f03",
    ".r",
    ".jl",
    ".mat",
}
_SKIP_DIRS = {".git", ".saguaro", ".anvil", "venv", ".venv", "__pycache__"}
_SKIP_DIR_EXACT = _SKIP_DIRS | {"cmakefiles", ".pytest_cache", ".mypy_cache"}
_SKIP_DIR_PREFIXES = ("build", "cmake-build", "dist", "out", "target")
_BLOCK_EQUATION_RE = re.compile(r"\$\$(.+?)\$\$|\\\[(.+?)\\\]", re.DOTALL)
_INLINE_EQUATION_RE = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)")
_SYMBOL_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_FUNCTION_CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_MATH_OPERATOR_RE = re.compile(r"(\*\*|==|!=|<=|>=|<-|[+\-*/%^]|⊙|×|·|⋅|∇|√|σ|⟨|⟩)")
_NON_COMPLEXITY_OPERATORS = {"=", "==", "!=", "<=", ">=", "<-"}
_MATH_SIGNAL_RE = re.compile(
    r"(\*\*|<-|[+\-*/%^]|⊙|×|·|⋅|∇|√|σ|⟨|⟩|[A-Za-z_][A-Za-z0-9_]*\s*\(|[A-Za-z_][A-Za-z0-9_]*\s*=|[A-Za-z_][A-Za-z0-9_]*\s*\[)"
)
_COMMENT_EQUATION_RE = re.compile(r"(=|O\(|⊙|×|·|⋅|∇|√|σ|⟨|⟩|\|\|)")
_STRING_LITERAL_RE = re.compile(r"\"[^\"]*\"|'[^']*'")
_BLOCK_BODY_RE = re.compile(r"\b(if|for|while|switch|case|return)\b|#\s*(if|ifdef|ifndef|else|endif)")
_CPP_LAMBDA_RE = re.compile(r"^\[[^\]]*\]\s*(?:\([^)]*\))?\s*\{")
_COMPLEXITY_BANDS = (
    (16, "high"),
    (8, "medium"),
    (0, "low"),
)


class MathEngine:
    """Provide repo-wide math extraction, complexity scoring, and graph mapping."""

    def __init__(self, repo_path: str) -> None:
        self.repo_path = Path(repo_path).resolve()
        self.base_dir = self.repo_path / ".saguaro" / "math"
        self.cache_path = self.base_dir / "cache.json"
        self.parser = SAGUAROParser()
        self.pipeline = MathPipeline(str(self.repo_path))

    def parse(self, path: str = ".") -> dict[str, Any]:
        records: list[MathIRRecord] = []
        files = self._discover_sources(path)
        file_engines: list[str] = []
        for file_path in files:
            rel_path = self._rel_path(file_path)
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            language = self.parser.detect_language(rel_path, text)
            file_engines.append(self._analysis_engine_for_language(language))
            records.extend(self._extract_from_source(text, rel_path))
        records = self._dedupe_records(records)
        analysis_engine = self._aggregate_analysis_engine(file_engines)
        payload = {
            "status": "ok",
            "schema_version": "saguaro.math.ir.v2",
            "path": self._resolved_output_path(path),
            "files_scanned": len(files),
            "count": len(records),
            "records": [item.to_dict() for item in records],
            "equations": [item.to_dict() for item in records],
            "analysis_engine": analysis_engine,
        }
        payload["summary"] = self._build_summary(records)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        payload["cache_path"] = self.cache_path.as_posix()
        return payload

    def _analysis_engine_for_language(self, language: str) -> str:
        normalized = str(language or "").strip().lower()
        if normalized in {"c", "cpp"} and self.parser.supports_ast_language(normalized):
            return "native_cpp"
        if self.parser.supports_ast_language(normalized):
            return "native_ast"
        return "python_pipeline"

    @staticmethod
    def _aggregate_analysis_engine(file_engines: list[str]) -> str:
        normalized = [str(item or "").strip() for item in file_engines if str(item or "").strip()]
        if not normalized:
            return "python_pipeline"
        unique = sorted(set(normalized))
        if len(unique) == 1:
            return unique[0]
        if "native_cpp" in unique:
            return "native_cpp"
        if "native_ast" in unique:
            return "native_ast"
        return "python_pipeline"

    def map(self, equation_id: str) -> dict[str, Any]:
        payload = self._load_cache()
        items = list(payload.get("records") or payload.get("equations") or [])
        equation = next((item for item in items if item.get("id") == equation_id), None)
        if equation is None:
            return {"status": "missing", "equation_id": equation_id}
        query = " ".join(
            [
                equation.get("expression", ""),
                *list(equation.get("symbols") or []),
            ]
        ).strip()
        matches = OmniGraphStore(str(self.repo_path)).find_equation(query)
        return {
            "status": "ok",
            "equation": equation,
            "matches": matches.get("matches", []),
            "count": len(matches.get("matches", [])),
        }

    def _resolved_output_path(self, path: str) -> str:
        target = Path(path)
        if target.is_absolute():
            return path
        return self._rel_path((self.repo_path / target).resolve())

    def _discover_sources(self, path: str) -> list[Path]:
        root = Path(path)
        root = root if root.is_absolute() else self.repo_path / root
        if root.is_file():
            return [root] if self._is_supported_source(root) else []
        files: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                name for name in dirnames if not self._should_skip_dirname(name)
            ]
            base = Path(dirpath)
            for name in filenames:
                file_path = base / name
                if self._is_supported_source(file_path):
                    files.append(file_path)
        return sorted(files)

    def _is_supported_source(self, file_path: Path) -> bool:
        if self._should_skip_path(file_path):
            return False
        suffix = file_path.suffix.lower()
        if suffix in _SUPPORTED_SUFFIXES:
            return True
        if suffix in _MARKDOWN_EXTENSIONS:
            return True
        return self.parser.detect_language(str(file_path)) in supported_languages()

    def _extract_from_source(self, text: str, source_path: str) -> list[MathIRRecord]:
        language = self.parser.detect_language(source_path, text)
        policy = policy_for(language)
        records: list[MathIRRecord] = []
        if Path(source_path).suffix.lower() in _MARKDOWN_EXTENSIONS or language == "markdown":
            records.extend(
                self._extract_delimited_equations(
                    text,
                    source_path=source_path,
                    language="markdown",
                    source_kind="markdown",
                )
            )
            return records

        records.extend(
            self._extract_delimited_equations(
                text,
                source_path=source_path,
                language=language,
                source_kind="code_comment",
            )
        )
        records.extend(
            self._extract_comment_equations(
                text,
                source_path=source_path,
                language=language,
                policy=policy,
            )
        )
        if policy.language == "config":
            return self._dedupe_records(records)
        code_records = self._extract_code_equations(
            text,
            source_path=source_path,
            language=language,
            policy=policy,
        )
        self.pipeline.enrich_records(
            code_records,
            text=text,
            source_path=source_path,
            policy=policy,
        )
        records.extend(code_records)
        return self._dedupe_records(records)

    def _extract_delimited_equations(
        self,
        text: str,
        *,
        source_path: str,
        language: str,
        source_kind: str,
    ) -> list[MathIRRecord]:
        records: list[MathIRRecord] = []
        matches = list(_BLOCK_EQUATION_RE.finditer(text)) + list(_INLINE_EQUATION_RE.finditer(text))
        for match in matches:
            raw = next((group for group in match.groups() if group), "")
            expression = " ".join(raw.split())
            if not expression:
                continue
            records.append(
                self._build_record(
                    expression,
                    source_path=source_path,
                    language=language,
                    source_kind=source_kind,
                    statement_kind="equation",
                    text=text,
                    start=match.start(),
                    end=match.end(),
                )
            )
        return records

    def _extract_comment_equations(
        self,
        text: str,
        *,
        source_path: str,
        language: str,
        policy: LanguagePolicy,
    ) -> list[MathIRRecord]:
        records: list[MathIRRecord] = []
        for block, offset in self._comment_blocks(text, policy):
            records.extend(
                self._extract_equation_lines(
                    block,
                    source_path=source_path,
                    language=language,
                    source_kind="code_comment",
                    offset=offset,
                )
            )
        for line_no, line in enumerate(text.splitlines(), start=1):
            content = self._line_comment_content(line, policy)
            if not content:
                continue
            if content.lower().startswith(("include ", "pragma ", "noqa", "ruff:")):
                continue
            if self._looks_like_comment_equation(content):
                records.append(
                    self._build_record(
                        content,
                        source_path=source_path,
                        language=language,
                        source_kind="code_comment",
                        statement_kind="comment_equation",
                        line_start=line_no,
                        line_end=line_no,
                    )
                )
        return self._dedupe_records(records)

    def _extract_code_equations(
        self,
        text: str,
        *,
        source_path: str,
        language: str,
        policy: LanguagePolicy,
    ) -> list[MathIRRecord]:
        records: list[MathIRRecord] = []
        for expression, line_start, line_end in self._iter_code_statements(
            self._mask_comment_blocks(text, policy),
            policy,
        ):
            statement = self._classify_statement(expression, policy)
            if statement is None:
                continue
            records.append(
                self._build_record(
                    statement["expression"],
                    source_path=source_path,
                    language=language,
                    source_kind="code_expression",
                    statement_kind=str(statement["statement_kind"]),
                    lhs=str(statement.get("lhs", "")),
                    rhs=str(statement.get("rhs", "")),
                    line_start=line_start,
                    line_end=line_end,
                )
            )
        return self._dedupe_records(records)

    def _extract_equation_lines(
        self,
        block: str,
        *,
        source_path: str,
        language: str,
        source_kind: str,
        offset: int,
    ) -> list[MathIRRecord]:
        records: list[MathIRRecord] = []
        for index, raw_line in enumerate(block.splitlines(), start=1):
            content = raw_line.strip().strip("*").strip()
            if not content or not self._looks_like_comment_equation(content):
                continue
            line_no = offset + index
            records.append(
                self._build_record(
                    content,
                    source_path=source_path,
                    language=language,
                    source_kind=source_kind,
                    statement_kind="comment_equation",
                    line_start=line_no,
                    line_end=line_no,
                )
            )
        return records

    def _build_record(
        self,
        expression: str,
        *,
        source_path: str,
        language: str,
        source_kind: str,
        statement_kind: str,
        lhs: str = "",
        rhs: str = "",
        line_start: int | None = None,
        line_end: int | None = None,
        text: str | None = None,
        start: int | None = None,
        end: int | None = None,
    ) -> MathIRRecord:
        normalized = " ".join(expression.split())
        if text is not None and start is not None and end is not None:
            line_start = text.count("\n", 0, start) + 1
            line_end = max(line_start, text.count("\n", 0, end) + 1)
        assert line_start is not None
        assert line_end is not None
        digest = hashlib.sha1(
            f"{source_path}|{source_kind}|{line_start}|{normalized}".encode("utf-8")
        ).hexdigest()[:12]
        return MathIRRecord(
            id=f"EQ-{digest}".upper(),
            file=source_path,
            expression=normalized,
            normalized_expression=normalized,
            line_start=line_start,
            line_end=line_end,
            symbols=sorted(set(_SYMBOL_RE.findall(normalized))),
            language=language,
            source_kind=source_kind,
            statement_kind=statement_kind,
            lhs=" ".join(lhs.split()),
            rhs=" ".join(rhs.split()),
            complexity=self._score_expression(normalized),
        )

    def _build_summary(self, records: list[MathIRRecord]) -> dict[str, Any]:
        by_kind = Counter(item.source_kind for item in records)
        by_language = Counter(item.language for item in records)
        by_statement_kind = Counter(item.statement_kind for item in records)
        loop_count = sum(1 for item in records if item.loop_context is not None)
        access_count = sum(len(item.access_signatures) for item in records)
        layout_count = sum(len(item.layout_states) for item in records)
        scores = [item.complexity.structural_score for item in records if item.complexity is not None]
        top = sorted(
            records,
            key=lambda item: item.complexity.structural_score if item.complexity is not None else 0,
            reverse=True,
        )[:10]
        return {
            "by_source_kind": dict(sorted(by_kind.items())),
            "by_language": dict(sorted(by_language.items())),
            "by_statement_kind": dict(sorted(by_statement_kind.items())),
            "loop_frame_count": loop_count,
            "access_signature_count": access_count,
            "layout_state_count": layout_count,
            "complexity": {
                "total_structural_score": sum(scores),
                "average_structural_score": round(sum(scores) / len(scores), 2) if scores else 0.0,
                "max_structural_score": max(scores) if scores else 0,
                "high_complexity_count": sum(
                    1 for item in records if item.complexity is not None and item.complexity.band == "high"
                ),
                "top_equations": [
                    {
                        "id": item.id,
                        "file": item.file,
                        "line_start": item.line_start,
                        "expression": item.expression,
                        "source_kind": item.source_kind,
                        "statement_kind": item.statement_kind,
                        "language": item.language,
                        "complexity": item.complexity.to_dict() if item.complexity is not None else {},
                        "loop_context": item.loop_context.to_dict() if item.loop_context is not None else None,
                    }
                    for item in top
                ],
            },
        }

    def _score_expression(self, expression: str) -> MathComplexity:
        operators = [
            token
            for token in _MATH_OPERATOR_RE.findall(expression)
            if token not in _NON_COMPLEXITY_OPERATORS
        ]
        symbols = sorted(set(_SYMBOL_RE.findall(expression)))
        function_calls = _FUNCTION_CALL_RE.findall(expression)
        token_count = len(re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|\S", expression))
        max_nesting_depth = self._max_nesting_depth(expression)
        structural_score = (
            len(operators)
            + len(symbols)
            + (2 * len(function_calls))
            + (2 * max_nesting_depth)
            + max(token_count // 4, 0)
        )
        band = next(level for threshold, level in _COMPLEXITY_BANDS if structural_score >= threshold)
        return MathComplexity(
            operator_count=len(operators),
            symbol_count=len(symbols),
            function_call_count=len(function_calls),
            max_nesting_depth=max_nesting_depth,
            token_count=token_count,
            structural_score=structural_score,
            band=band,
        )

    @staticmethod
    def _max_nesting_depth(expression: str) -> int:
        depth = 0
        max_depth = 0
        for char in expression:
            if char in "([{":
                depth += 1
                max_depth = max(max_depth, depth)
            elif char in ")]}":
                depth = max(depth - 1, 0)
        return max_depth

    @staticmethod
    def _looks_like_comment_equation(text: str) -> bool:
        normalized = " ".join(text.split()).lstrip("-").strip()
        if len(normalized) < 5:
            return False
        if normalized and set(normalized) <= {"=", "-", "*", "_", " "}:
            return False
        if normalized.startswith(("@brief ", "@param ", "@return ", "@tparam ")):
            normalized = normalized.split(" ", 1)[1]
        return bool(_COMMENT_EQUATION_RE.search(normalized))

    def _classify_statement(
        self,
        expression: str,
        policy: LanguagePolicy,
    ) -> dict[str, str] | None:
        normalized = " ".join(expression.split()).rstrip(";")
        if len(normalized) < 3:
            return None
        lowered = normalized.lower()
        if any(lowered.startswith(prefix.lower()) for prefix in policy.ignore_prefixes):
            return None
        if normalized.startswith(("{", "[", "(")) and not _MATH_OPERATOR_RE.search(normalized):
            return None
        for keyword in policy.return_keywords:
            marker = f"{keyword} "
            if lowered.startswith(marker):
                rhs = normalized[len(marker) :].strip()
                if self._rhs_looks_mathy(rhs, policy):
                    return {
                        "expression": normalized,
                        "statement_kind": "return",
                        "lhs": "",
                        "rhs": rhs,
                    }
                return None
        for token in policy.compound_assignment_tokens:
            if token in normalized:
                lhs, rhs = normalized.split(token, 1)
                if self._rhs_looks_mathy(rhs, policy):
                    return {
                        "expression": normalized,
                        "statement_kind": "compound_assignment",
                        "lhs": lhs.strip(),
                        "rhs": rhs.strip(),
                    }
                return None
        assignment = self._split_assignment(normalized, policy)
        if assignment is None:
            return None
        lhs, rhs, assignment_token = assignment
        if not self._rhs_looks_mathy(rhs, policy):
            return None
        statement_kind = "assignment" if assignment_token in {"=", "<-"} else "expression"
        return {
            "expression": normalized,
            "statement_kind": statement_kind,
            "lhs": lhs.strip(),
            "rhs": rhs.strip(),
        }

    def _split_assignment(
        self,
        expression: str,
        policy: LanguagePolicy,
    ) -> tuple[str, str, str] | None:
        for token in policy.assignment_tokens:
            if token == "=":
                offset = self._find_assignment_operator(expression)
                if offset is None:
                    continue
                return expression[:offset], expression[offset + 1 :], token
            if token in expression:
                lhs, rhs = expression.split(token, 1)
                return lhs, rhs, token
        return None

    def _rhs_looks_mathy(self, rhs: str, policy: LanguagePolicy) -> bool:
        normalized = " ".join(_STRING_LITERAL_RE.sub("", rhs).split())
        if len(normalized) < 2:
            return False
        if _CPP_LAMBDA_RE.match(normalized):
            return False
        if "{" in normalized and _BLOCK_BODY_RE.search(normalized):
            return False
        if normalized.startswith(("{", "[")) and ":" in normalized and not _MATH_OPERATOR_RE.search(normalized):
            return False
        if normalized.startswith(("[", "(")) and not _MATH_OPERATOR_RE.search(normalized):
            return False
        if normalized.startswith(("dict(", "list(", "set(")):
            return False
        if normalized in {"true", "false", "none", "null"}:
            return False
        if _MATH_OPERATOR_RE.search(normalized):
            return True
        if "[" in normalized and "]" in normalized:
            return True
        if any(f"{name}(" in normalized for name in policy.math_function_names):
            return True
        symbol_count = len(set(_SYMBOL_RE.findall(normalized)))
        numeric_count = len(re.findall(r"\d+(?:\.\d+)?", normalized))
        return symbol_count + numeric_count >= 3 and "," not in normalized

    def _comment_blocks(
        self,
        text: str,
        policy: LanguagePolicy,
    ) -> list[tuple[str, int]]:
        blocks: list[tuple[str, int]] = []
        for start_marker, end_marker in policy.block_comment_pairs:
            pattern = re.compile(
                re.escape(start_marker) + r"(.*?)" + re.escape(end_marker),
                re.DOTALL,
            )
            for match in pattern.finditer(text):
                raw = match.group(1)
                offset = text.count("\n", 0, match.start())
                blocks.append((raw, offset))
        return blocks

    def _line_comment_content(self, line: str, policy: LanguagePolicy) -> str:
        stripped = line.strip()
        for marker in policy.line_comment_markers:
            if stripped.startswith(marker):
                return stripped[len(marker) :].strip()
        return ""

    def _iter_code_statements(
        self,
        text: str,
        policy: LanguagePolicy,
    ) -> list[tuple[str, int, int]]:
        statements: list[tuple[str, int, int]] = []
        buffer: list[str] = []
        start_line = 0
        depth = 0
        lines = text.splitlines()
        for line_no, raw_line in enumerate(lines, start=1):
            stripped = self._strip_inline_comment(raw_line, policy).strip()
            if not stripped:
                continue
            if not buffer and not self._is_math_statement_start(stripped, policy):
                continue
            if not buffer:
                start_line = line_no
            buffer.append(stripped)
            depth += sum(1 for char in stripped if char in "([{")
            depth -= sum(1 for char in stripped if char in ")]}")
            if self._statement_is_complete(stripped, depth, policy):
                statements.append((" ".join(buffer), start_line, line_no))
                buffer = []
                start_line = 0
                depth = 0
        if buffer:
            statements.append((" ".join(buffer), start_line or 1, len(lines)))
        return statements

    @staticmethod
    def _strip_inline_comment(line: str, policy: LanguagePolicy) -> str:
        result = line
        for marker in policy.line_comment_markers:
            if marker and marker in result:
                result = result.split(marker, 1)[0]
        return result

    def _mask_comment_blocks(self, text: str, policy: LanguagePolicy) -> str:
        masked = text
        for start_marker, end_marker in policy.block_comment_pairs:
            pattern = re.compile(
                re.escape(start_marker) + r"(.*?)" + re.escape(end_marker),
                re.DOTALL,
            )
            masked = pattern.sub(self._mask_match, masked)
        return masked

    @staticmethod
    def _mask_match(match: re.Match[str]) -> str:
        value = match.group(0)
        return "".join("\n" if char == "\n" else " " for char in value)

    def _is_math_statement_start(self, line: str, policy: LanguagePolicy) -> bool:
        normalized = " ".join(line.split())
        lowered = normalized.lower()
        if normalized.endswith("{"):
            return False
        if any(
            lowered.startswith(f"{keyword} ") or lowered.startswith(f"{keyword}(")
            for keyword in policy.loop_keywords
        ):
            return False
        if any(lowered.startswith(f"{keyword} ") for keyword in policy.return_keywords):
            return True
        if any(token in normalized for token in policy.compound_assignment_tokens):
            return True
        return self._split_assignment(normalized, policy) is not None

    @staticmethod
    def _statement_is_complete(line: str, depth: int, policy: LanguagePolicy) -> bool:
        normalized = line.rstrip()
        if policy.statement_terminators:
            return depth <= 0 and normalized.endswith(policy.statement_terminators)
        return depth <= 0 and not normalized.endswith(("\\", ",", "(", "[", "{"))

    @staticmethod
    def _find_assignment_operator(expression: str) -> int | None:
        depth = 0
        quote = ""
        escaped = False
        for index, char in enumerate(expression):
            if quote:
                if escaped:
                    escaped = False
                    continue
                if char == "\\":
                    escaped = True
                    continue
                if char == quote:
                    quote = ""
                continue
            if char in {'"', "'"}:
                quote = char
                continue
            if char in "([{":
                depth += 1
                continue
            if char in ")]}":
                depth = max(depth - 1, 0)
                continue
            if char != "=" or depth > 0:
                continue
            prev_char = expression[index - 1] if index > 0 else ""
            next_char = expression[index + 1] if index + 1 < len(expression) else ""
            if prev_char in "=!<>+-*/%&|^:" or next_char == "=":
                continue
            return index
        return None

    @staticmethod
    def _dedupe_records(records: list[MathIRRecord]) -> list[MathIRRecord]:
        seen: set[tuple[str, int, str, str]] = set()
        deduped: list[MathIRRecord] = []
        for item in records:
            key = (item.file, item.line_start, item.expression, item.source_kind)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _load_cache(self) -> dict[str, Any]:
        if not self.cache_path.exists():
            return {"status": "missing", "records": [], "equations": []}
        return json.loads(self.cache_path.read_text(encoding="utf-8"))

    @staticmethod
    def _should_skip_dirname(name: str) -> bool:
        lowered = name.strip().lower()
        if lowered in _SKIP_DIR_EXACT or lowered.endswith(".egg-info"):
            return True
        return any(
            lowered == prefix
            or lowered.startswith(f"{prefix}-")
            or lowered.startswith(f"{prefix}.")
            for prefix in _SKIP_DIR_PREFIXES
        )

    def _should_skip_path(self, path: Path) -> bool:
        return any(self._should_skip_dirname(part) for part in path.parts[:-1])

    def _rel_path(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.repo_path).as_posix()
        except ValueError:
            return path.resolve().as_posix()
