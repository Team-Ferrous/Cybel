from __future__ import annotations

import ast
import difflib
import hashlib
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from core.backup import BackupManager
from saguaro.state.ledger import StateLedger


@dataclass
class FreshnessToken:
    path: str
    mtime_ns: int
    size: int
    inode: int
    content_hash: str
    git_head: Optional[str]
    git_status: Optional[str]
    captured_monotonic_ms: int


class FileOps:
    """Handles file operations with stale-write protection and audit hooks."""

    def __init__(self, root_dir: str = ".", audit_db: Any = None, session_id: Optional[str] = None):
        self.root_dir = os.path.abspath(root_dir)
        self.backup_manager = BackupManager(root_dir)
        self.audit_db = audit_db
        self.session_id = session_id
        self._read_tokens: Dict[str, FreshnessToken] = {}
        self._trace_counter = 0
        self._state_ledger = StateLedger(self.root_dir)

    def _next_trace_id(self) -> str:
        self._trace_counter += 1
        return f"fo-{int(time.time() * 1000)}-{self._trace_counter}"

    def _resolve_path(self, path: str) -> str:
        full_path = os.path.abspath(os.path.join(self.root_dir, path))
        if os.path.commonpath([self.root_dir, full_path]) != self.root_dir:
            raise ValueError(f"Access denied: {path} is outside root directory.")
        return full_path

    def _validate_python(self, content: str) -> Tuple[bool, Optional[str]]:
        try:
            ast.parse(content)
            return True, None
        except SyntaxError as exc:
            return False, str(exc)

    def _hash_bytes(self, data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def _hash_file(self, full_path: str) -> str:
        with open(full_path, "rb") as handle:
            return self._hash_bytes(handle.read())

    def _git_head(self) -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.root_dir,
                capture_output=True,
                text=True,
                check=False,
                timeout=2,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            return None
        return None

    def _git_status_for_path(self, full_path: str) -> Optional[str]:
        rel = os.path.relpath(full_path, self.root_dir)
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain", "--", rel],
                cwd=self.root_dir,
                capture_output=True,
                text=True,
                check=False,
                timeout=2,
            )
            if result.returncode == 0:
                return result.stdout.strip() or "clean"
        except Exception:
            return None
        return None

    def _capture_token(self, full_path: str) -> FreshnessToken:
        st = os.stat(full_path)
        return FreshnessToken(
            path=full_path,
            mtime_ns=int(st.st_mtime_ns),
            size=int(st.st_size),
            inode=int(getattr(st, "st_ino", 0)),
            content_hash=self._hash_file(full_path),
            git_head=self._git_head(),
            git_status=self._git_status_for_path(full_path),
            captured_monotonic_ms=int(time.monotonic() * 1000),
        )

    def _remember_read(self, full_path: str) -> None:
        if os.path.exists(full_path):
            self._read_tokens[full_path] = self._capture_token(full_path)

    def _freshness_check(self, full_path: str) -> Tuple[bool, str, Optional[FreshnessToken], Optional[FreshnessToken]]:
        if not os.path.exists(full_path):
            return True, "new-file", None, None

        previous = self._read_tokens.get(full_path)
        if previous is None:
            return (
                False,
                "Read-before-write policy violation: read the file first to obtain a freshness token.",
                None,
                None,
            )

        current = self._capture_token(full_path)
        if (
            previous.mtime_ns != current.mtime_ns
            or previous.size != current.size
            or previous.inode != current.inode
            or previous.content_hash != current.content_hash
        ):
            return (
                False,
                "Stale file detected: file changed since it was read. Re-read and retry.",
                previous,
                current,
            )

        # Git-aware freshness: if HEAD/index changed for this path since read, require re-read.
        if previous.git_head and current.git_head and previous.git_head != current.git_head:
            return (
                False,
                "Stale file detected: repository HEAD changed for this path since last read.",
                previous,
                current,
            )
        if previous.git_status and current.git_status and previous.git_status != current.git_status:
            return (
                False,
                "Stale file detected: git index/worktree status changed since last read.",
                previous,
                current,
            )

        return True, "fresh", previous, current

    def _log_file_op(
        self,
        *,
        path: str,
        operation: str,
        status: str,
        trace_id: str,
        details: Optional[Dict[str, Any]] = None,
        freshness_token: Optional[str] = None,
    ) -> Optional[int]:
        if self.audit_db is None:
            return None
        try:
            return self.audit_db.log_file_operation(
                session_id=self.session_id,
                path=path,
                operation=operation,
                status=status,
                monotonic_elapsed_ms=int(time.monotonic() * 1000),
                freshness_token=freshness_token,
                details=details,
                trace_id=trace_id,
            )
        except Exception:
            return None

    def _log_file_version(
        self,
        *,
        path: str,
        operation_id: Optional[int],
        hash_before: Optional[str],
        hash_after: Optional[str],
        backup_path: Optional[str],
        trace_id: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.audit_db is None:
            return
        try:
            self.audit_db.log_file_version(
                session_id=self.session_id,
                path=path,
                operation_id=operation_id,
                hash_before=hash_before,
                hash_after=hash_after,
                backup_path=backup_path,
                git_head=self._git_head(),
                trace_id=trace_id,
                details=details,
            )
        except Exception:
            return

    def _repo_rel(self, full_path: str) -> str:
        return os.path.relpath(full_path, self.root_dir).replace("\\", "/")

    def _record_ledger_change(
        self,
        *,
        changed_files: Optional[List[str]] = None,
        deleted_files: Optional[List[str]] = None,
        reason: str,
    ) -> None:
        try:
            self._state_ledger.record_changes(
                changed_files=changed_files or [],
                deleted_files=deleted_files or [],
                reason=reason,
            )
        except Exception:
            return

    def read_file_chunked(self, path: str, chunk_size: int = 8192):
        full_path = self._resolve_path(path)
        with open(full_path, "r", encoding="utf-8", errors="ignore") as handle:
            while True:
                chunk = handle.read(chunk_size)
                if not chunk:
                    break
                yield chunk
        self._remember_read(full_path)

    def read_file_range(self, path: str, start_byte: int = 0, end_byte: Optional[int] = None) -> str:
        full_path = self._resolve_path(path)
        with open(full_path, "rb") as handle:
            handle.seek(start_byte)
            if end_byte is not None:
                data = handle.read(max(0, end_byte - start_byte))
            else:
                data = handle.read()
        self._remember_read(full_path)
        return data.decode("utf-8", errors="ignore")

    def _format_with_line_numbers(self, content: str, first_line_number: int = 1) -> str:
        lines = content.splitlines(keepends=True)
        if not lines:
            return content
        width = max(1, len(str(first_line_number + len(lines) - 1)))
        return "".join(
            f"{first_line_number + idx:>{width}}: {line}" for idx, line in enumerate(lines)
        )

    def read_file(
        self,
        path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        max_chars: Optional[int] = None,
        include_line_numbers: bool = False,
    ) -> str:
        try:
            full_path = self._resolve_path(path)
            if not os.path.exists(full_path):
                return f"Error: File {path} not found."

            with open(full_path, "r", encoding="utf-8", errors="ignore") as handle:
                if start_line or end_line:
                    start = max(1, int(start_line) if start_line else 1)
                    end = int(end_line) if end_line else None
                    selected_lines: List[str] = []
                    for line_number, line in enumerate(handle, start=1):
                        if line_number < start:
                            continue
                        if end is not None and line_number > end:
                            break
                        selected_lines.append(line)
                    content = "".join(selected_lines)
                    first_line = start
                else:
                    if max_chars is not None:
                        content = handle.read(max(0, int(max_chars)))
                    else:
                        content = handle.read()
                    first_line = 1

            self._remember_read(full_path)
            if include_line_numbers:
                return self._format_with_line_numbers(content, first_line)
            return content
        except Exception as exc:
            return f"Error reading file {path}: {exc}"

    def _atomic_write(self, full_path: str, content: str) -> None:
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", delete=False, dir=os.path.dirname(full_path)
        ) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        os.replace(tmp_path, full_path)

    def write_file(self, path: str, content: str) -> str:
        trace_id = self._next_trace_id()
        try:
            full_path = self._resolve_path(path)
            exists = os.path.exists(full_path)

            if full_path.endswith(".py"):
                valid, err = self._validate_python(content)
                if not valid:
                    self._log_file_op(
                        path=path,
                        operation="write",
                        status="blocked",
                        trace_id=trace_id,
                        details={"reason": "syntax", "error": err},
                    )
                    return f"Error: Write blocked due to Syntax Error:\n{err}"

            fresh, reason, previous, _ = self._freshness_check(full_path)
            if exists and not fresh:
                self._log_file_op(
                    path=path,
                    operation="write",
                    status="blocked",
                    trace_id=trace_id,
                    details={"reason": reason},
                    freshness_token=previous.content_hash if previous else None,
                )
                return f"Error: {reason}"

            hash_before = self._hash_file(full_path) if exists else None
            backup_path = self.backup_manager.backup(full_path) if exists else None

            self._atomic_write(full_path, content)
            hash_after = self._hash_file(full_path)
            self._remember_read(full_path)

            op_id = self._log_file_op(
                path=path,
                operation="write",
                status="success",
                trace_id=trace_id,
                details={"bytes": len(content.encode('utf-8'))},
                freshness_token=previous.content_hash if previous else None,
            )
            self._log_file_version(
                path=path,
                operation_id=op_id,
                hash_before=hash_before,
                hash_after=hash_after,
                backup_path=backup_path,
                trace_id=trace_id,
            )
            self._record_ledger_change(
                changed_files=[self._repo_rel(full_path)],
                reason="file_ops.write",
            )

            action = "wrote" if not exists else "updated"
            return f"Successfully {action} {path}"
        except Exception as exc:
            self._log_file_op(
                path=path,
                operation="write",
                status="error",
                trace_id=trace_id,
                details={"error": str(exc)},
            )
            return f"Error writing file: {exc}"

    def edit_file(self, path: str, edits: List[Dict[str, Any]], dry_run: bool = False) -> str:
        trace_id = self._next_trace_id()
        try:
            full_path = self._resolve_path(path)
            if not os.path.exists(full_path):
                return f"Error: File {path} not found."

            fresh, reason, previous, _ = self._freshness_check(full_path)
            if not fresh:
                self._log_file_op(
                    path=path,
                    operation="edit",
                    status="blocked",
                    trace_id=trace_id,
                    details={"reason": reason},
                    freshness_token=previous.content_hash if previous else None,
                )
                return f"Error: {reason}"

            with open(full_path, "r", encoding="utf-8") as handle:
                old_lines = handle.readlines()

            new_lines = old_lines.copy()
            for edit in sorted(edits, key=lambda item: item["start_line"], reverse=True):
                start = max(0, int(edit["start_line"]) - 1)
                end = max(start, int(edit["end_line"]))
                new_content = str(edit.get("new_content", ""))
                if not new_content.endswith("\n") and end < len(old_lines):
                    new_content += "\n"
                new_lines[start:end] = [new_content]

            new_content_str = "".join(new_lines)

            if full_path.endswith(".py"):
                valid, err = self._validate_python(new_content_str)
                if not valid:
                    return f"Error: Edit blocked due to Syntax Error:\n{err}"

            if dry_run:
                diff = difflib.unified_diff(
                    old_lines,
                    new_lines,
                    fromfile=f"a/{path}",
                    tofile=f"b/{path}",
                    lineterm="",
                )
                return "\n".join(diff)

            hash_before = self._hash_file(full_path)
            backup_path = self.backup_manager.backup(full_path)
            self._atomic_write(full_path, new_content_str)
            hash_after = self._hash_file(full_path)
            self._remember_read(full_path)

            op_id = self._log_file_op(
                path=path,
                operation="edit",
                status="success",
                trace_id=trace_id,
                details={"changes": len(edits)},
                freshness_token=previous.content_hash if previous else None,
            )
            self._log_file_version(
                path=path,
                operation_id=op_id,
                hash_before=hash_before,
                hash_after=hash_after,
                backup_path=backup_path,
                trace_id=trace_id,
                details={"changes": len(edits)},
            )
            self._record_ledger_change(
                changed_files=[self._repo_rel(full_path)],
                reason="file_ops.edit",
            )
            return f"Successfully edited {path} ({len(edits)} changes applied)"
        except Exception as exc:
            self._log_file_op(
                path=path,
                operation="edit",
                status="error",
                trace_id=trace_id,
                details={"error": str(exc)},
            )
            return f"Error editing file: {exc}"

    def read_files(
        self,
        paths: List[str],
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        max_chars: Optional[int] = None,
        include_line_numbers: bool = False,
    ) -> str:
        blocks = []
        for path in paths:
            blocks.append(
                f"--- File: {path} ---\n"
                + self.read_file(
                    path,
                    start_line=start_line,
                    end_line=end_line,
                    max_chars=max_chars,
                    include_line_numbers=include_line_numbers,
                )
            )
        return "\n\n".join(blocks)

    def read_files_parallel(
        self,
        paths: List[str],
        max_workers: int = 8,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        max_chars: Optional[int] = None,
        include_line_numbers: bool = False,
    ) -> Dict[str, str]:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results: Dict[str, str] = {}

        def read_single(path: str) -> Tuple[str, str]:
            return (
                path,
                self.read_file(
                    path,
                    start_line=start_line,
                    end_line=end_line,
                    max_chars=max_chars,
                    include_line_numbers=include_line_numbers,
                ),
            )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(read_single, p): p for p in paths}
            for future in as_completed(futures):
                path, content = future.result()
                if not str(content).startswith("Error"):
                    results[path] = content
        return results

    def write_files(self, files: Dict[str, str]) -> str:
        results = [self.write_file(path, content) for path, content in files.items()]
        return "\n".join(results)

    def classify_patch_risk(self, patch: str) -> str:
        changed_files = len([line for line in patch.splitlines() if line.startswith("+++")])
        changed_hunks = len([line for line in patch.splitlines() if line.startswith("@@")])
        if changed_files > 3 or changed_hunks > 8:
            return "high"
        return "low"

    def apply_patch(self, path: str, patch: str) -> str:
        trace_id = self._next_trace_id()
        try:
            full_path = self._resolve_path(path)
            if not os.path.exists(full_path):
                return f"Error: File {path} not found."

            fresh, reason, previous, _ = self._freshness_check(full_path)
            if not fresh:
                self._log_file_op(
                    path=path,
                    operation="apply_patch",
                    status="blocked",
                    trace_id=trace_id,
                    details={"reason": reason},
                    freshness_token=previous.content_hash if previous else None,
                )
                return f"Error: {reason}"

            hash_before = self._hash_file(full_path)
            backup_path = self.backup_manager.backup(full_path)
            risk = self.classify_patch_risk(patch)

            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as tmp:
                tmp.write(patch)
                tmp_path = tmp.name

            try:
                dry_run = subprocess.run(
                    ["patch", "--dry-run", "--fuzz=1", full_path, "-i", tmp_path],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if dry_run.returncode != 0:
                    return f"Patch failed dry-run validation: {dry_run.stderr.strip()}"

                apply_result = subprocess.run(
                    ["patch", "--fuzz=1", full_path, "-i", tmp_path],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if apply_result.returncode != 0:
                    return f"Patch failed: {apply_result.stderr.strip()}"
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

            if full_path.endswith(".py"):
                with open(full_path, "r", encoding="utf-8", errors="ignore") as handle:
                    patched_content = handle.read()
                valid, err = self._validate_python(patched_content)
                if not valid:
                    if backup_path and os.path.exists(backup_path):
                        shutil.copy2(backup_path, full_path)
                    return f"Patch reverted due to syntax error: {err}"

            hash_after = self._hash_file(full_path)
            self._remember_read(full_path)

            op_id = self._log_file_op(
                path=path,
                operation="apply_patch",
                status="success",
                trace_id=trace_id,
                details={"risk": risk},
                freshness_token=previous.content_hash if previous else None,
            )
            self._log_file_version(
                path=path,
                operation_id=op_id,
                hash_before=hash_before,
                hash_after=hash_after,
                backup_path=backup_path,
                trace_id=trace_id,
                details={"risk": risk},
            )
            self._record_ledger_change(
                changed_files=[self._repo_rel(full_path)],
                reason="file_ops.apply_patch",
            )
            return f"Successfully applied patch to {path} (risk={risk})"
        except FileNotFoundError:
            return "Error: 'patch' utility not found."
        except Exception as exc:
            self._log_file_op(
                path=path,
                operation="apply_patch",
                status="error",
                trace_id=trace_id,
                details={"error": str(exc)},
            )
            return f"Error applying patch: {exc}"

    def list_dir(
        self,
        path: str = ".",
        recursive: bool = False,
        max_depth: Optional[int] = None,
        filter_noise: bool = True,
    ) -> List[str]:
        try:
            full_path = self._resolve_path(path)
            if not os.path.isdir(full_path):
                return [f"Error: {path} is not a directory."]

            ignore_patterns = {
                "venv",
                ".git",
                "__pycache__",
                ".saguaro",
                ".pytest_cache",
                ".ruff_cache",
                ".anvil_backups",
                ".idea",
                ".vscode",
                "node_modules",
            }

            results: List[str] = []
            start_depth = full_path.rstrip(os.sep).count(os.sep)
            for root, dirs, files in os.walk(full_path):
                current_depth = root.rstrip(os.sep).count(os.sep) - start_depth
                if not recursive and current_depth > 0:
                    break
                if recursive and max_depth is not None and current_depth >= max_depth:
                    continue

                if filter_noise:
                    dirs[:] = [d for d in dirs if d not in ignore_patterns]

                rel_root = os.path.relpath(root, self.root_dir)
                if rel_root == ".":
                    rel_root = ""

                level_entries = [f"[DIR]  {os.path.join(rel_root, d)}" for d in dirs]
                for filename in files:
                    if filter_noise and any(filename.endswith(ext) for ext in (".pyc", ".pyo", ".so")):
                        continue
                    level_entries.append(f"[FILE] {os.path.join(rel_root, filename)}")

                results.extend(sorted(level_entries))
                if not recursive:
                    break

            return results
        except Exception as exc:
            return [f"Error listing directory: {exc}"]

    def delete_file(self, path: str) -> str:
        trace_id = self._next_trace_id()
        try:
            full_path = self._resolve_path(path)
            if not os.path.exists(full_path):
                return f"Error: File {path} not found."
            if os.path.isdir(full_path):
                return f"Error: {path} is a directory. Use shell commands to remove directories."

            hash_before = self._hash_file(full_path)
            backup_path = self.backup_manager.backup(full_path)
            os.remove(full_path)
            self._read_tokens.pop(full_path, None)

            op_id = self._log_file_op(
                path=path,
                operation="delete",
                status="success",
                trace_id=trace_id,
                details={},
            )
            self._log_file_version(
                path=path,
                operation_id=op_id,
                hash_before=hash_before,
                hash_after=None,
                backup_path=backup_path,
                trace_id=trace_id,
            )
            self._record_ledger_change(
                deleted_files=[self._repo_rel(full_path)],
                reason="file_ops.delete",
            )
            return f"Successfully deleted {path}"
        except Exception as exc:
            self._log_file_op(
                path=path,
                operation="delete",
                status="error",
                trace_id=trace_id,
                details={"error": str(exc)},
            )
            return f"Error deleting file: {exc}"

    def move_file(self, src: str, dst: str) -> str:
        trace_id = self._next_trace_id()
        try:
            full_src = self._resolve_path(src)
            full_dst = self._resolve_path(dst)
            if not os.path.exists(full_src):
                return f"Error: Source {src} not found."
            if os.path.exists(full_dst):
                return f"Error: Destination {dst} already exists."

            os.makedirs(os.path.dirname(full_dst), exist_ok=True)
            os.rename(full_src, full_dst)
            token = self._read_tokens.pop(full_src, None)
            if token is not None:
                self._read_tokens[full_dst] = token

            self._log_file_op(
                path=f"{src} -> {dst}",
                operation="move",
                status="success",
                trace_id=trace_id,
                details={},
            )
            self._record_ledger_change(
                changed_files=[self._repo_rel(full_dst)],
                deleted_files=[self._repo_rel(full_src)],
                reason="file_ops.move",
            )
            return f"Successfully moved {src} to {dst}"
        except Exception as exc:
            self._log_file_op(
                path=f"{src} -> {dst}",
                operation="move",
                status="error",
                trace_id=trace_id,
                details={"error": str(exc)},
            )
            return f"Error moving file: {exc}"

    def list_backups(self, path: str, max_items: int = 20) -> str:
        try:
            full_path = self._resolve_path(path)
            backups = self.backup_manager.list_backups(full_path, max_items=max_items)
            if not backups:
                return f"No backups found for {path}"
            rel_paths = [os.path.relpath(item, self.root_dir) for item in backups]
            return "\n".join(rel_paths)
        except Exception as exc:
            return f"Error listing backups for {path}: {exc}"

    def rollback_file(self, path: str, backup_path: Optional[str] = None) -> str:
        trace_id = self._next_trace_id()
        try:
            full_path = self._resolve_path(path)
            resolved_backup = None
            if backup_path:
                resolved_backup = os.path.abspath(
                    backup_path
                    if os.path.isabs(backup_path)
                    else os.path.join(self.root_dir, backup_path)
                )

            hash_before = self._hash_file(full_path) if os.path.exists(full_path) else None
            restored_from = self.backup_manager.restore(full_path, backup_path=resolved_backup)
            if not restored_from:
                self._log_file_op(
                    path=path,
                    operation="rollback",
                    status="error",
                    trace_id=trace_id,
                    details={"reason": "backup_not_found"},
                )
                return f"Error: No backup available to rollback {path}"

            hash_after = self._hash_file(full_path)
            self._remember_read(full_path)
            op_id = self._log_file_op(
                path=path,
                operation="rollback",
                status="success",
                trace_id=trace_id,
                details={"backup_path": os.path.relpath(restored_from, self.root_dir)},
            )
            self._log_file_version(
                path=path,
                operation_id=op_id,
                hash_before=hash_before,
                hash_after=hash_after,
                backup_path=restored_from,
                trace_id=trace_id,
                details={"action": "rollback"},
            )
            self._record_ledger_change(
                changed_files=[self._repo_rel(full_path)],
                reason="file_ops.rollback",
            )
            return f"Rolled back {path} using backup {os.path.relpath(restored_from, self.root_dir)}"
        except Exception as exc:
            self._log_file_op(
                path=path,
                operation="rollback",
                status="error",
                trace_id=trace_id,
                details={"error": str(exc)},
            )
            return f"Error rolling back {path}: {exc}"

    def get_file_structure(self, path: str) -> Dict[str, Any]:
        try:
            full_path = self._resolve_path(path)
            if not os.path.exists(full_path):
                return {"error": f"File not found: {path}"}
            if not full_path.endswith(".py"):
                return {"error": "Only Python files supported for AST parsing"}

            with open(full_path, "r", encoding="utf-8", errors="ignore") as handle:
                content = handle.read()

            tree = ast.parse(content)
            classes: List[Dict[str, Any]] = []
            functions: List[Dict[str, Any]] = []
            imports: List[str] = []

            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    methods = []
                    for child in node.body:
                        if isinstance(child, ast.FunctionDef):
                            methods.append(
                                {
                                    "name": child.name,
                                    "args": [arg.arg for arg in child.args.args],
                                    "line": child.lineno,
                                }
                            )
                    classes.append(
                        {
                            "name": node.name,
                            "line": node.lineno,
                            "end_line": getattr(node, "end_lineno", node.lineno),
                            "methods": methods,
                        }
                    )
                elif isinstance(node, ast.FunctionDef):
                    functions.append(
                        {
                            "name": node.name,
                            "args": [arg.arg for arg in node.args.args],
                            "line": node.lineno,
                            "end_line": getattr(node, "end_lineno", node.lineno),
                        }
                    )
                elif isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    imports.extend(f"{module}.{alias.name}" for alias in node.names)

            self._remember_read(full_path)
            return {
                "path": path,
                "classes": classes,
                "functions": functions,
                "imports": imports,
                "total_lines": len(content.splitlines()),
            }
        except SyntaxError as exc:
            return {"error": f"Syntax error in {path}: {exc}"}
        except Exception as exc:
            return {"error": f"Error parsing {path}: {exc}"}

    def extract_entity(self, path: str, entity_name: str) -> str:
        try:
            full_path = self._resolve_path(path)
            if not os.path.exists(full_path):
                return f"Error: File {path} not found."

            with open(full_path, "r", encoding="utf-8", errors="ignore") as handle:
                content = handle.read()
            lines = content.splitlines()
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == entity_name:
                    start = node.lineno - 1
                    end = getattr(node, "end_lineno", len(lines))
                    self._remember_read(full_path)
                    return "\n".join(lines[start:end])

            return f"Entity '{entity_name}' not found in {path}."
        except Exception as exc:
            return f"Error extracting entity: {exc}"

    def smart_search(
        self,
        query: str,
        extensions: Optional[List[str]] = None,
        include_content: bool = False,
        max_results: int = 20,
    ) -> List[Dict[str, Any]]:
        if extensions is None:
            extensions = [".py", ".md", ".yaml", ".json"]

        results: List[Dict[str, Any]] = []
        query_lower = query.lower()
        ignore_dirs = {
            "venv",
            ".git",
            "__pycache__",
            ".saguaro",
            "node_modules",
            ".pytest_cache",
            ".ruff_cache",
            ".anvil_backups",
            ".idea",
            ".vscode",
        }

        for root, dirs, files in os.walk(self.root_dir):
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            for filename in files:
                if len(results) >= max_results:
                    return sorted(results, key=lambda item: item.get("score", 0), reverse=True)
                if not any(filename.endswith(ext) for ext in extensions):
                    continue

                rel_path = os.path.relpath(os.path.join(root, filename), self.root_dir)

                if query_lower == filename.lower():
                    results.append({"path": rel_path, "match_type": "exact_filename", "score": 100})
                    continue
                if query_lower in filename.lower():
                    results.append({"path": rel_path, "match_type": "partial_filename", "score": 80})
                    continue

                if filename.endswith(".py"):
                    structure = self.get_file_structure(rel_path)
                    if "error" in structure:
                        continue
                    for cls in structure.get("classes", []):
                        if query_lower == cls["name"].lower():
                            results.append(
                                {
                                    "path": rel_path,
                                    "match_type": "class_definition",
                                    "entity": cls["name"],
                                    "line": cls["line"],
                                    "score": 90,
                                }
                            )
                            break
                    for func in structure.get("functions", []):
                        if query_lower == func["name"].lower():
                            results.append(
                                {
                                    "path": rel_path,
                                    "match_type": "function_definition",
                                    "entity": func["name"],
                                    "line": func["line"],
                                    "score": 85,
                                }
                            )
                            break

                if include_content and len(results) < max_results:
                    try:
                        with open(os.path.join(root, filename), "r", encoding="utf-8", errors="ignore") as handle:
                            text = handle.read(4096)
                        if query_lower in text.lower():
                            results.append(
                                {
                                    "path": rel_path,
                                    "match_type": "content",
                                    "score": 70,
                                }
                            )
                    except Exception:
                        continue

        return sorted(results, key=lambda item: item.get("score", 0), reverse=True)
