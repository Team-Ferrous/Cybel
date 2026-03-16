import re
from typing import Any

SIMD_RE = re.compile(r"_mm(128|256|512)_|__m(128|256|512)")
ALIGNMENT_SENSITIVE_SIMD_RE = re.compile(
    r"_mm(?:128|256|512)?_(?:load|store|stream)(?!u)\w*"
)
UNALIGNED_SIMD_RE = re.compile(r"_mm(?:128|256|512)?_(?:load|store|stream)u\w*")
OMP_RE = re.compile(r"#pragma\s+omp\s+parallel")
RAW_ALLOC_RE = re.compile(r"\bnew\s+|\bdelete(?:\[\])?\b")
SMART_PTR_RE = re.compile(r"\b(unique_ptr|shared_ptr|make_unique|make_shared)\b")
NODISCARD_CANDIDATE_RE = re.compile(
    r"^\s*(?!//)(?:(?:inline|virtual|constexpr)\s+)*"
    r"(?P<signature>[A-Za-z_][\w:\s<>,*&]+?)\s+"
    r"(?P<name>[A-Za-z_]\w*)\s*\([^;{}]*\)\s*(?:const\s*)?(?:\{|;)\s*$"
)
C_STYLE_CAST_RE = re.compile(
    r"\(\s*(?:unsigned|signed|const|volatile|struct|class|enum|"
    r"[A-Za-z_]\w*(?:::\w+)*(?:\s*[*&])?)\s*\)\s*[A-Za-z_(]"
)


def _violation(rule_id: str, filepath: str, line: int, message: str) -> dict[str, Any]:
    return {
        "rule_id": rule_id,
        "filepath": filepath,
        "line": line,
        "message": message,
    }


def check_alignment_contracts(source: str, filepath: str) -> list[dict[str, Any]]:
    has_alignment_marker = any(
        token in source for token in ("alignas(", "aligned_alloc", "posix_memalign")
    )
    has_simd = SIMD_RE.search(source) is not None
    has_sensitive_intrinsic = ALIGNMENT_SENSITIVE_SIMD_RE.search(source) is not None
    has_only_explicit_unaligned_intrinsics = (
        has_simd
        and not has_sensitive_intrinsic
        and UNALIGNED_SIMD_RE.search(source) is not None
    )
    if has_simd and not has_alignment_marker and (
        has_sensitive_intrinsic or not has_only_explicit_unaligned_intrinsics
    ):
        return [
            _violation(
                "AES-HPC-2",
                filepath,
                1,
                "SIMD usage detected without alignment contract markers",
            )
        ]
    return []


def check_explicit_omp_clauses(source: str, filepath: str) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for match in OMP_RE.finditer(source):
        line = source[: match.start()].count("\n") + 1
        window = source[match.start() : match.start() + 160]
        if not any(token in window for token in ("shared(", "private(", "reduction(", "schedule(")):
            violations.append(
                _violation(
                    "AES-HPC-3",
                    filepath,
                    line,
                    "OpenMP parallel region missing explicit data/schedule clauses",
                )
            )
    return violations


def check_scalar_reference_impl(source: str, filepath: str) -> list[dict[str, Any]]:
    if SIMD_RE.search(source) and not any(
        token in source.lower() for token in ("scalar", "reference", "oracle")
    ):
        return [
            _violation(
                "AES-HPC-4",
                filepath,
                1,
                "SIMD kernel detected without scalar/reference implementation marker",
            )
        ]
    return []


def check_raii_enforcement(source: str, filepath: str) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for line_no, line in enumerate(source.splitlines(), start=1):
        if line.strip().startswith("//"):
            continue
        if RAW_ALLOC_RE.search(line) and SMART_PTR_RE.search(line) is None:
            violations.append(
                _violation(
                    "AES-CPP-3",
                    filepath,
                    line_no,
                    "Raw new/delete detected without RAII smart-pointer markers",
                )
            )
    return violations


def check_nodiscard_returns(source: str, filepath: str) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for line_no, line in enumerate(source.splitlines(), start=1):
        if "[[nodiscard]]" in line:
            continue
        match = NODISCARD_CANDIDATE_RE.match(line)
        if match is None:
            continue
        signature = match.group("signature")
        if not any(
            marker in signature
            for marker in ("Status", "Result", "Error", "bool", "std::optional")
        ):
            continue
        violations.append(
            _violation(
                "AES-CPP-4",
                filepath,
                line_no,
                "Error-bearing return signature missing [[nodiscard]]",
            )
        )
    return violations


def check_c_style_casts(source: str, filepath: str) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for line_no, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("//"):
            continue
        if any(cast in line for cast in ("static_cast<", "reinterpret_cast<", "const_cast<")):
            continue
        if C_STYLE_CAST_RE.search(line):
            violations.append(
                _violation(
                    "AES-CPP-5",
                    filepath,
                    line_no,
                    "C-style cast detected; use static_cast/reinterpret_cast with rationale",
                )
            )
    return violations
