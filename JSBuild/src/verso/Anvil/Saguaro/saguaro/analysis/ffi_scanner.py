from __future__ import annotations

import ast
from dataclasses import dataclass
import os
import re
from typing import Any


@dataclass(frozen=True, slots=True)
class BoundaryTypeInfo:
    """Canonicalized FFI boundary typing metadata."""

    boundary_type: str
    source_runtime: str
    target_runtime: str
    source_language: str
    target_language: str


class BoundaryTypeMap:
    """Maps scanner pattern/evidence pairs to typed FFI boundaries."""

    _BASE: dict[str, BoundaryTypeInfo] = {
        "ctypes_load_library": BoundaryTypeInfo(
            boundary_type="ctypes.library_load",
            source_runtime="cpython",
            target_runtime="native_abi",
            source_language="python",
            target_language="native",
        ),
        "cffi_dlopen": BoundaryTypeInfo(
            boundary_type="cffi.dlopen",
            source_runtime="cpython",
            target_runtime="native_abi",
            source_language="python",
            target_language="native",
        ),
        "cpp_pybind_module": BoundaryTypeInfo(
            boundary_type="pybind11.module",
            source_runtime="native",
            target_runtime="cpython",
            source_language="cpp",
            target_language="python",
        ),
        "cpp_python_capi": BoundaryTypeInfo(
            boundary_type="python_c_api",
            source_runtime="native",
            target_runtime="cpython",
            source_language="cpp",
            target_language="python",
        ),
        "go_cgo": BoundaryTypeInfo(
            boundary_type="cgo.import_c",
            source_runtime="go",
            target_runtime="native_abi",
            source_language="go",
            target_language="native",
        ),
        "rust_pyo3": BoundaryTypeInfo(
            boundary_type="pyo3.binding",
            source_runtime="native",
            target_runtime="cpython",
            source_language="rust",
            target_language="python",
        ),
        "extern_c_export": BoundaryTypeInfo(
            boundary_type="extern_c_export",
            source_runtime="native",
            target_runtime="native_abi",
            source_language="native",
            target_language="native",
        ),
        "tensorflow_load_op_library": BoundaryTypeInfo(
            boundary_type="tensorflow.custom_op_load",
            source_runtime="tensorflow",
            target_runtime="native_abi",
            source_language="python",
            target_language="native",
        ),
    }

    @classmethod
    def resolve(cls, kind: str, evidence: str) -> BoundaryTypeInfo:
        base = cls._BASE.get(kind)
        if base is None:
            return BoundaryTypeInfo(
                boundary_type="ffi.unknown",
                source_runtime="unknown",
                target_runtime="unknown",
                source_language="unknown",
                target_language="unknown",
            )

        low = (evidence or "").lower()
        if kind == "ctypes_load_library":
            if "pydll" in low:
                return BoundaryTypeInfo(
                    boundary_type="ctypes.pydll",
                    source_runtime=base.source_runtime,
                    target_runtime=base.target_runtime,
                    source_language=base.source_language,
                    target_language=base.target_language,
                )
            if "windll" in low:
                return BoundaryTypeInfo(
                    boundary_type="ctypes.windll",
                    source_runtime=base.source_runtime,
                    target_runtime=base.target_runtime,
                    source_language=base.source_language,
                    target_language=base.target_language,
                )
            if "cdll" in low:
                return BoundaryTypeInfo(
                    boundary_type="ctypes.cdll",
                    source_runtime=base.source_runtime,
                    target_runtime=base.target_runtime,
                    source_language=base.source_language,
                    target_language=base.target_language,
                )
        if kind == "cffi_dlopen":
            if "ffi(" in low and "dlopen" not in low:
                return BoundaryTypeInfo(
                    boundary_type="cffi.constructor",
                    source_runtime=base.source_runtime,
                    target_runtime=base.target_runtime,
                    source_language=base.source_language,
                    target_language=base.target_language,
                )
            if ".verify(" in low or "set_source(" in low:
                return BoundaryTypeInfo(
                    boundary_type="cffi.verify",
                    source_runtime=base.source_runtime,
                    target_runtime=base.target_runtime,
                    source_language=base.source_language,
                    target_language=base.target_language,
                )
        return base


class SharedObjectResolver:
    """Heuristically maps shared object names to probable source/build files."""

    _SOURCE_EXTS = (".cc", ".cpp", ".cxx", ".c", ".h", ".hpp", ".rs", ".go", ".py")
    _SHARED_OBJECT_SUFFIXES = (".so", ".dylib", ".dll", ".pyd", ".wasm")
    _SO_VERSION_RE = re.compile(r"(?P<name>.+?\.so)(?:\.[0-9][0-9.]*)?$")

    @classmethod
    def normalize_shared_object_name(cls, library_hint: str) -> str:
        hint = str(library_hint or "").strip().strip("\"'")
        if not hint:
            return ""
        hint = hint.replace("\\", "/")
        base = os.path.basename(hint)
        if any(base.endswith(ext) for ext in cls._SHARED_OBJECT_SUFFIXES):
            return base
        match = cls._SO_VERSION_RE.match(base)
        if match:
            return str(match.group("name") or "")
        return ""

    @classmethod
    def _shared_object_stem(cls, shared_object: str) -> str:
        stem = cls.normalize_shared_object_name(shared_object)
        for suffix in cls._SHARED_OBJECT_SUFFIXES:
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        if stem.startswith("lib") and len(stem) > 3:
            stem = stem[3:]
        return stem

    @classmethod
    def resolve_metadata(
        cls,
        shared_object: str,
        *,
        rel_file: str = "",
        repo_path: str | None = None,
    ) -> dict[str, Any]:
        so_name = cls.normalize_shared_object_name(shared_object)
        if not so_name:
            return {
                "normalized": "",
                "stem": "",
                "selected": "",
                "candidate_count": 0,
                "existing_count": 0,
                "candidates": [],
            }

        stem = cls._shared_object_stem(so_name)
        base_candidates: dict[str, tuple[str, float]] = {
            so_name: ("shared_object_name", 0.95),
            f"build/{so_name}": ("build_output", 0.9),
            f"dist/{so_name}": ("distribution_artifact", 0.8),
            f"target/release/{so_name}": ("rust_release", 0.85),
            f"target/debug/{so_name}": ("rust_debug", 0.75),
        }

        for ext in cls._SOURCE_EXTS:
            base_candidates[f"src/{stem}{ext}"] = ("source_stem", 0.72)
            base_candidates[f"native/{stem}{ext}"] = ("native_stem", 0.7)
            base_candidates[f"saguaro/native/{stem}{ext}"] = ("saguaro_native_stem", 0.68)
            base_candidates[f"core/native/{stem}{ext}"] = ("core_native_stem", 0.78)

        rel = str(rel_file or "").replace("\\", "/").strip("/")
        if rel:
            folder = os.path.dirname(rel)
            if folder:
                base_candidates[f"{folder}/{so_name}"] = ("relative_to_bridge_file", 0.86)
                base_candidates[f"{folder}/build/{so_name}"] = ("relative_build_artifact", 0.82)
                for ext in cls._SOURCE_EXTS:
                    base_candidates[f"{folder}/{stem}{ext}"] = ("relative_source_stem", 0.66)

        repo_root = os.path.abspath(repo_path or ".")
        candidate_rows: list[dict[str, Any]] = []
        for path, (reason, score) in sorted(base_candidates.items()):
            rel_item = path.replace("\\", "/").lstrip("/")
            abs_item = os.path.abspath(os.path.join(repo_root, rel_item))
            exists = abs_item.startswith(repo_root) and os.path.exists(abs_item)
            candidate_rows.append(
                {
                    "path": rel_item,
                    "exists": bool(exists),
                    "reason": reason,
                    "score": float(score),
                }
            )

        selected = ""
        for row in sorted(
            candidate_rows,
            key=lambda item: (
                not bool(item.get("exists")),
                -float(item.get("score") or 0.0),
                str(item.get("path") or ""),
            ),
        ):
            path = str(row.get("path") or "")
            if path:
                selected = path
                break
        return {
            "normalized": so_name,
            "stem": stem,
            "selected": selected,
            "candidate_count": len(candidate_rows),
            "existing_count": sum(1 for row in candidate_rows if row.get("exists")),
            "candidates": candidate_rows,
        }

    @classmethod
    def resolve_candidates(
        cls,
        shared_object: str,
        *,
        rel_file: str = "",
        repo_path: str | None = None,
    ) -> list[str]:
        metadata = cls.resolve_metadata(
            shared_object,
            rel_file=rel_file,
            repo_path=repo_path,
        )
        return [
            str(row.get("path") or "")
            for row in metadata.get("candidates", [])
            if str(row.get("path") or "")
        ]


class FFIScanner:
    """Detect likely FFI bridge patterns with deterministic confidence scores."""

    _PATTERNS: tuple[dict[str, Any], ...] = (
        {
            "name": "ctypes_load_library",
            "regex": re.compile(
                r"ctypes\.(?:(?P<loader>CDLL|PyDLL|WinDLL|oledll|cdll)\s*\(|(?:cdll|oledll)\.LoadLibrary\s*\()"
            ),
            "extensions": {".py"},
            "direction": "python_to_native",
            "role": "consumer",
            "confidence": 0.96,
        },
        {
            "name": "cffi_dlopen",
            "regex": re.compile(r"(?:cffi\.FFI\s*\(|ffi\.dlopen\s*\()"),
            "extensions": {".py"},
            "direction": "python_to_native",
            "role": "consumer",
            "confidence": 0.9,
        },
        {
            "name": "cpp_pybind_module",
            "regex": re.compile(r"\bPYBIND11_MODULE\s*\("),
            "extensions": {".cc", ".cpp", ".cxx", ".h", ".hpp"},
            "direction": "native_to_python",
            "role": "provider",
            "confidence": 0.97,
        },
        {
            "name": "cpp_python_capi",
            "regex": re.compile(r"#include\s*[<\"]Python\.h[>\"]"),
            "extensions": {".cc", ".cpp", ".c", ".h", ".hpp"},
            "direction": "native_to_python",
            "role": "provider",
            "confidence": 0.82,
        },
        {
            "name": "go_cgo",
            "regex": re.compile(r"^\s*import\s+\"C\"", re.MULTILINE),
            "extensions": {".go"},
            "direction": "go_to_native",
            "role": "consumer",
            "confidence": 0.91,
        },
        {
            "name": "rust_pyo3",
            "regex": re.compile(r"#\[(?:pyfunction|pymodule|pyclass)\]"),
            "extensions": {".rs"},
            "direction": "native_to_python",
            "role": "provider",
            "confidence": 0.9,
        },
        {
            "name": "extern_c_export",
            "regex": re.compile(r"extern\s+\"C\"\s+"),
            "extensions": {".cc", ".cpp", ".cxx", ".c", ".h", ".hpp"},
            "direction": "native_c_abi",
            "role": "provider",
            "confidence": 0.7,
        },
        {
            "name": "tensorflow_load_op_library",
            "regex": re.compile(
                r"(?:tf|tensorflow)\.load_op_library\s*\(|\bload_op_library\s*\("
            ),
            "extensions": {".py"},
            "direction": "tensorflow_to_native",
            "role": "consumer",
            "confidence": 0.97,
        },
    )

    _LIB_HINT = re.compile(r"[\"']([A-Za-z0-9_./\-]+(?:\.[A-Za-z0-9_\-]+)?)")
    _CTYPE_NAME_MAP: dict[str, str] = {
        "c_int": "int",
        "c_long": "int",
        "c_uint": "int",
        "c_short": "int",
        "c_size_t": "int",
        "c_float": "float",
        "c_double": "float",
        "c_bool": "bool",
        "c_char_p": "str|bytes",
        "c_wchar_p": "str",
        "c_void_p": "pointer",
        "c_char": "bytes",
    }

    def __init__(self, repo_path: str | None = None) -> None:
        self.repo_path = os.path.abspath(repo_path or ".")
        self._shared_object_cache: dict[str, dict[str, Any]] = {}

    def scan_file(self, rel_file: str, source: str) -> list[dict[str, Any]]:
        ext = os.path.splitext(rel_file)[1].lower()
        if not source:
            return []

        typing_context = self._extract_typing_context(ext=ext, source=source)
        findings: list[dict[str, Any]] = []
        for pattern in self._PATTERNS:
            if ext not in pattern["extensions"]:
                continue
            regex: re.Pattern[str] = pattern["regex"]
            for index, match in enumerate(regex.finditer(source), start=1):
                line = source.count("\n", 0, match.start()) + 1
                evidence = self._line_excerpt(source, line, fallback_start=match.start())
                library_hint = self._extract_library_hint(evidence)
                if pattern["name"] == "cpp_pybind_module" and not library_hint:
                    pybind = typing_context.get("pybind11") if isinstance(typing_context, dict) else {}
                    module_name = ""
                    if isinstance(pybind, dict):
                        module_name = str(pybind.get("module_name") or "")
                    if module_name:
                        library_hint = f"{module_name}.so"
                typed = self._extract_typed_boundary(pattern_name=pattern["name"], evidence=evidence)
                shared_object = SharedObjectResolver.normalize_shared_object_name(library_hint)
                shared_object_resolution = self._resolve_shared_object(
                    shared_object,
                    rel_file=rel_file,
                )
                typed_extraction = self._typed_extraction_for_kind(
                    pattern_name=str(pattern["name"]),
                    shared_object=shared_object,
                    typing_context=typing_context,
                )
                boundary_type_map = self._build_boundary_type_map(
                    pattern_name=str(pattern["name"]),
                    shared_object=shared_object,
                    typed_extraction=typed_extraction,
                )
                confidence = float(pattern["confidence"])
                if typed_extraction:
                    confidence = min(0.99, confidence + 0.02)
                finding_id = (
                    f"ffi::{rel_file}::{pattern['name']}::{int(line)}::{int(index)}"
                )
                findings.append(
                    {
                        "id": finding_id,
                        "file": rel_file,
                        "line": int(line),
                        "kind": pattern["name"],
                        "direction": pattern["direction"],
                        "role": pattern["role"],
                        "confidence": confidence,
                        "evidence": evidence[:160],
                        "library_hint": library_hint,
                        "boundary_type": typed.boundary_type,
                        "source_runtime": typed.source_runtime,
                        "target_runtime": typed.target_runtime,
                        "source_language": typed.source_language,
                        "target_language": typed.target_language,
                        "typed_boundary": {
                            "type": typed.boundary_type,
                            "source_runtime": typed.source_runtime,
                            "target_runtime": typed.target_runtime,
                            "source_language": typed.source_language,
                            "target_language": typed.target_language,
                        },
                        "type_map": boundary_type_map,
                        "typing_extraction": typed_extraction,
                        "shared_object": shared_object,
                        "shared_object_candidates": SharedObjectResolver.resolve_candidates(
                            shared_object,
                            rel_file=rel_file,
                            repo_path=self.repo_path,
                        ),
                        "shared_object_resolution": shared_object_resolution,
                    }
                )

        findings.sort(
            key=lambda item: (
                str(item.get("file") or ""),
                int(item.get("line") or 0),
                str(item.get("kind") or ""),
                str(item.get("id") or ""),
            )
        )
        return findings

    def _extract_library_hint(self, evidence: str) -> str:
        match = self._LIB_HINT.search(evidence or "")
        if not match:
            return ""
        return str(match.group(1) or "").strip()

    def _resolve_shared_object(self, shared_object: str, *, rel_file: str) -> dict[str, Any]:
        key = f"{rel_file}::{shared_object}"
        cached = self._shared_object_cache.get(key)
        if cached is not None:
            return dict(cached)
        metadata = SharedObjectResolver.resolve_metadata(
            shared_object,
            rel_file=rel_file,
            repo_path=self.repo_path,
        )
        self._shared_object_cache[key] = dict(metadata)
        return metadata

    @staticmethod
    def _line_excerpt(source: str, line: int, *, fallback_start: int = 0) -> str:
        lines = source.splitlines()
        if 1 <= line <= len(lines):
            return str(lines[line - 1]).strip()
        return source[fallback_start : min(len(source), fallback_start + 160)].splitlines()[0].strip()

    def _extract_typing_context(self, *, ext: str, source: str) -> dict[str, Any]:
        if ext == ".py":
            return {
                "ctypes": self._extract_ctypes_typing(source),
                "cffi": self._extract_cffi_typing(source),
            }
        if ext in {".cc", ".cpp", ".cxx", ".h", ".hpp"}:
            return {
                "pybind11": self._extract_pybind_typing(source),
            }
        return {}

    @staticmethod
    def _extract_pybind_typing(source: str) -> dict[str, Any]:
        module_match = re.search(r"\bPYBIND11_MODULE\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)", source)
        module_name = str(module_match.group(1) or "") if module_match else ""
        exports = sorted(
            {
                str(match.group(1) or "")
                for match in re.finditer(r"\.def\s*\(\s*\"([A-Za-z0-9_]+)\"", source)
                if str(match.group(1) or "")
            }
        )
        args = sorted(
            {
                str(match.group(1) or "")
                for match in re.finditer(r"py::arg\s*\(\s*\"([A-Za-z0-9_]+)\"", source)
                if str(match.group(1) or "")
            }
        )
        return_policies = sorted(
            {
                str(match.group(1) or "")
                for match in re.finditer(
                    r"py::return_value_policy::([A-Za-z0-9_]+)",
                    source,
                )
                if str(match.group(1) or "")
            }
        )
        return {
            "module_name": module_name,
            "exports": exports,
            "args": args,
            "return_policies": return_policies,
        }

    def _extract_ctypes_typing(self, source: str) -> dict[str, Any]:
        try:
            tree = ast.parse(source)
        except Exception:
            return {"libraries": {}, "functions": {}, "by_library": {}}

        libraries: dict[str, str] = {}
        functions: dict[str, dict[str, Any]] = {}
        by_library: dict[str, dict[str, dict[str, Any]]] = {}
        loader_names = {
            "ctypes.CDLL",
            "ctypes.PyDLL",
            "ctypes.WinDLL",
            "ctypes.cdll.LoadLibrary",
            "ctypes.oledll.LoadLibrary",
            "CDLL",
            "PyDLL",
            "WinDLL",
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                if isinstance(node.value, ast.Call):
                    call_name = self._node_name(node.value.func)
                    if call_name in loader_names:
                        lib_hint = self._string_arg(node.value)
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                libraries[target.id] = lib_hint
                for target in node.targets:
                    chain = self._attribute_chain(target)
                    if len(chain) < 3:
                        continue
                    field_name = chain[-1]
                    fn_name = chain[-2]
                    lib_name = chain[0]
                    if field_name not in {"argtypes", "restype"}:
                        continue
                    fn_payload = functions.setdefault(
                        fn_name,
                        {"argtypes": [], "restype": "", "library": lib_name},
                    )
                    lib_payload = by_library.setdefault(lib_name, {}).setdefault(
                        fn_name,
                        {"argtypes": [], "restype": ""},
                    )
                    if field_name == "argtypes":
                        argtypes = self._extract_argtype_list(node.value)
                        fn_payload["argtypes"] = argtypes
                        lib_payload["argtypes"] = argtypes
                    else:
                        restype = self._ctypes_type_name(node.value)
                        fn_payload["restype"] = restype
                        lib_payload["restype"] = restype

        return {
            "libraries": libraries,
            "functions": functions,
            "by_library": by_library,
        }

    def _extract_cffi_typing(self, source: str) -> dict[str, Any]:
        try:
            tree = ast.parse(source)
        except Exception:
            return {"ffi_vars": [], "cdef_signatures": [], "dlopen_targets": []}

        ffi_vars: set[str] = set()
        signatures: list[str] = []
        dlopen_targets: list[str] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                call_name = self._node_name(node.value.func)
                if call_name in {"FFI", "cffi.FFI"}:
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            ffi_vars.add(target.id)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                owner_chain = self._attribute_chain(node.func.value)
                owner = owner_chain[0] if owner_chain else ""
                method = str(node.func.attr or "")
                if method == "cdef" and (not owner or owner in ffi_vars):
                    text = self._string_arg(node)
                    if text:
                        signatures.append(text)
                if method == "dlopen" and (not owner or owner in ffi_vars):
                    text = self._string_arg(node)
                    if text:
                        dlopen_targets.append(text)
        return {
            "ffi_vars": sorted(ffi_vars),
            "cdef_signatures": signatures,
            "dlopen_targets": sorted(set(dlopen_targets)),
        }

    def _typed_extraction_for_kind(
        self,
        *,
        pattern_name: str,
        shared_object: str,
        typing_context: dict[str, Any],
    ) -> dict[str, Any]:
        if pattern_name == "ctypes_load_library":
            payload = dict(typing_context.get("ctypes") or {})
            if shared_object:
                normalized = SharedObjectResolver.normalize_shared_object_name(shared_object)
                by_library = payload.get("by_library") if isinstance(payload, dict) else {}
                if isinstance(by_library, dict):
                    narrowed: dict[str, Any] = {}
                    for lib_var, fn_map in by_library.items():
                        lib_hint = str((payload.get("libraries") or {}).get(lib_var) or "")
                        if SharedObjectResolver.normalize_shared_object_name(lib_hint) == normalized:
                            narrowed[lib_var] = fn_map
                    if narrowed:
                        payload["by_library"] = narrowed
            return {"ctypes": payload}
        if pattern_name == "cffi_dlopen":
            return {"cffi": dict(typing_context.get("cffi") or {})}
        if pattern_name == "cpp_pybind_module":
            return {"pybind11": dict(typing_context.get("pybind11") or {})}
        return {}

    def _build_boundary_type_map(
        self,
        *,
        pattern_name: str,
        shared_object: str,
        typed_extraction: dict[str, Any],
    ) -> dict[str, Any] | None:
        if pattern_name == "ctypes_load_library":
            ctypes_payload = typed_extraction.get("ctypes")
            if not isinstance(ctypes_payload, dict):
                return None
            selected_fn: dict[str, Any] | None = None
            for fn in (ctypes_payload.get("functions") or {}).values():
                if isinstance(fn, dict):
                    selected_fn = fn
                    break
            if selected_fn is None:
                by_lib = ctypes_payload.get("by_library")
                if isinstance(by_lib, dict):
                    for fn_map in by_lib.values():
                        if isinstance(fn_map, dict):
                            for fn in fn_map.values():
                                if isinstance(fn, dict):
                                    selected_fn = fn
                                    break
                        if selected_fn is not None:
                            break
            if selected_fn is None:
                return None
            guest_args = [str(arg) for arg in selected_fn.get("argtypes", []) if str(arg)]
            host_params = [
                {
                    "name": f"arg{idx + 1}",
                    "host_type": self._host_type_for_ctype(arg),
                    "guest_type": arg,
                    "shape": "",
                }
                for idx, arg in enumerate(guest_args)
            ]
            guest_ret = str(selected_fn.get("restype") or "void")
            return {
                "host_params": host_params,
                "host_return": {
                    "host_type": self._host_type_for_ctype(guest_ret),
                    "guest_type": guest_ret,
                    "shape": "",
                },
                "memory_model": "copy",
                "ownership": "host-owned",
                "evidence": ["ctypes argtypes/restype extraction"],
                "confidence": 0.72 if host_params or guest_ret else 0.45,
                "shared_object": shared_object,
            }
        if pattern_name == "cffi_dlopen":
            cffi_payload = typed_extraction.get("cffi")
            if not isinstance(cffi_payload, dict):
                return None
            signature = ""
            signatures = list(cffi_payload.get("cdef_signatures") or [])
            if signatures:
                signature = str(signatures[0] or "")
            params: list[dict[str, Any]] = []
            host_return = {"host_type": "unknown", "guest_type": "unknown", "shape": ""}
            match = re.search(
                r"(?P<ret>[A-Za-z_][A-Za-z0-9_\s\*]*)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\((?P<args>[^)]*)\)",
                signature,
            )
            if match:
                args = [item.strip() for item in str(match.group("args") or "").split(",") if item.strip() and item.strip() != "void"]
                params = [
                    {
                        "name": f"arg{idx + 1}",
                        "host_type": "object",
                        "guest_type": arg,
                        "shape": "",
                    }
                    for idx, arg in enumerate(args)
                ]
                ret = str(match.group("ret") or "void").strip()
                host_return = {"host_type": "object", "guest_type": ret, "shape": ""}
            return {
                "host_params": params,
                "host_return": host_return,
                "memory_model": "copy",
                "ownership": "host-owned",
                "evidence": ["cffi cdef signature extraction"] if signature else [],
                "confidence": 0.68 if signature else 0.42,
                "shared_object": shared_object,
            }
        if pattern_name == "cpp_pybind_module":
            pybind = typed_extraction.get("pybind11")
            if not isinstance(pybind, dict):
                return None
            exports = [str(item) for item in pybind.get("exports", []) if str(item)]
            args = [str(item) for item in pybind.get("args", []) if str(item)]
            params = [
                {
                    "name": arg,
                    "host_type": "object",
                    "guest_type": "auto",
                    "shape": "",
                }
                for arg in args
            ]
            return {
                "host_params": params,
                "host_return": {"host_type": "object", "guest_type": "auto", "shape": ""},
                "memory_model": "copy",
                "ownership": "shared",
                "evidence": ["pybind11 export extraction"] if exports else [],
                "confidence": 0.64 if exports else 0.48,
                "exports": exports,
                "shared_object": shared_object,
            }
        return None

    def _host_type_for_ctype(self, ctype: str) -> str:
        token = str(ctype or "").strip()
        if token.startswith("ctypes."):
            token = token[len("ctypes.") :]
        if token.endswith("*"):
            return "pointer"
        return self._CTYPE_NAME_MAP.get(token, "object")

    @staticmethod
    def _attribute_chain(node: ast.AST) -> list[str]:
        out: list[str] = []
        cursor: ast.AST | None = node
        while isinstance(cursor, ast.Attribute):
            out.append(str(cursor.attr or ""))
            cursor = cursor.value
        if isinstance(cursor, ast.Name):
            out.append(str(cursor.id or ""))
        out = [item for item in reversed(out) if item]
        return out

    def _node_name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return str(node.id or "")
        if isinstance(node, ast.Attribute):
            chain = self._attribute_chain(node)
            return ".".join(chain)
        return ""

    @staticmethod
    def _string_arg(call: ast.Call) -> str:
        if not call.args:
            return ""
        first = call.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return str(first.value)
        return ""

    def _extract_argtype_list(self, node: ast.AST) -> list[str]:
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self._ctypes_type_name(item) for item in node.elts if self._ctypes_type_name(item)]
        return []

    def _ctypes_type_name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return str(node.id or "")
        if isinstance(node, ast.Attribute):
            return ".".join(self._attribute_chain(node))
        if isinstance(node, ast.Subscript):
            try:
                return ast.unparse(node)
            except Exception:
                return ""
        if isinstance(node, ast.Constant):
            return str(node.value or "")
        return ""

    @staticmethod
    def _extract_typed_boundary(pattern_name: str, evidence: str) -> BoundaryTypeInfo:
        return BoundaryTypeMap.resolve(kind=pattern_name, evidence=evidence)

    def build_multi_hop_chains(
        self,
        findings: list[dict[str, Any]],
        *,
        max_hops: int = 4,
    ) -> list[dict[str, Any]]:
        """Builds multi-hop chains grouped by shared object/library hints."""
        groups: dict[str, list[dict[str, Any]]] = {}
        for item in findings:
            token = str(item.get("shared_object") or item.get("library_hint") or "").strip().lower()
            if not token:
                continue
            groups.setdefault(token, []).append(item)

        chains: list[dict[str, Any]] = []
        for token, items in sorted(groups.items()):
            ordered = sorted(
                items,
                key=lambda row: (
                    str(row.get("file") or ""),
                    int(row.get("line") or 0),
                    str(row.get("id") or ""),
                ),
            )
            if len(ordered) < 2:
                continue
            hops = ordered[: max(2, max_hops)]
            chains.append(
                {
                    "id": f"ffi_chain::{token}::{len(hops)}",
                    "token": token,
                    "hop_count": len(hops),
                    "hops": [
                        {
                            "id": str(h.get("id") or ""),
                            "file": str(h.get("file") or ""),
                            "line": int(h.get("line") or 0),
                            "role": str(h.get("role") or ""),
                            "boundary_type": str(h.get("boundary_type") or ""),
                            "kind": str(h.get("kind") or ""),
                        }
                        for h in hops
                    ],
                }
            )
        return chains

    def find_chain_for_finding(
        self,
        findings: list[dict[str, Any]],
        *,
        finding_id: str,
        max_hops: int = 4,
    ) -> dict[str, Any] | None:
        """Returns the first multi-hop chain containing ``finding_id``."""
        needle = str(finding_id or "").strip()
        if not needle:
            return None
        for chain in self.build_multi_hop_chains(findings=findings, max_hops=max_hops):
            hop_ids = {str(item.get("id") or "") for item in chain.get("hops") or []}
            if needle in hop_ids:
                return chain
        return None


__all__ = [
    "BoundaryTypeInfo",
    "BoundaryTypeMap",
    "SharedObjectResolver",
    "FFIScanner",
]
