from __future__ import annotations

import os
from typing import Any, Dict, List

from core.aes import AALClassifier
from infrastructure.hooks.base import Hook


class AESPreVerifyHook(Hook):
    """Run Sentinel checks after write-capable tools and block strict AAL violations."""

    WRITE_TOOLS = {
        "write_file",
        "edit_file",
        "write_files",
        "apply_patch",
        "move_file",
    }
    STRICT_AAL = {"AAL-0", "AAL-1"}

    def __init__(self, repo_path: str = "."):
        self.repo_path = os.path.abspath(repo_path)
        self.classifier = AALClassifier()

    @property
    def name(self) -> str:
        return "aes_pre_verify"

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        tool_name = str(context.get("tool_name", ""))
        if tool_name not in self.WRITE_TOOLS:
            return context

        targets = self._extract_targets(tool_name, context.get("tool_args", {}))
        from saguaro.sentinel.verifier import SentinelVerifier

        verifier = SentinelVerifier(
            repo_path=self.repo_path,
            engines=["native", "ruff", "semantic", "aes"],
        )

        blocked = False
        all_violations: List[Dict[str, Any]] = []
        files: List[Dict[str, Any]] = []

        for target in targets:
            target_abs = (
                target
                if os.path.isabs(target)
                else os.path.abspath(os.path.join(self.repo_path, target))
            )
            if not os.path.isfile(target_abs):
                continue
            aal = self.classifier.classify_file(target_abs)
            rel_path = os.path.relpath(target_abs, self.repo_path).replace("\\", "/")
            strict = aal in self.STRICT_AAL
            compliance_context = {
                "run_id": "hook-pre-verify",
                "aal": aal,
                "changed_files": [rel_path],
            }
            violations = verifier.verify_all(
                path_arg=target_abs,
                aal=aal,
                require_trace=strict,
                require_evidence=strict,
                require_valid_waivers=strict,
                compliance_context=compliance_context,
            )
            files.append(
                {
                    "path": rel_path,
                    "aal": aal,
                    "violation_count": len(violations),
                }
            )
            all_violations.extend(violations)
            if strict and violations:
                blocked = True

        result = {
            "verified": True,
            "blocked": blocked,
            "files": files,
            "violations": all_violations,
        }
        context["post_write_verification"] = result
        if blocked:
            context["write_blocked"] = True
            context["tool_result"] = (
                "AES GATE: post-write verification failed for AAL-0/AAL-1 target(s)."
            )
        return context

    @staticmethod
    def _extract_targets(tool_name: str, tool_args: Dict[str, Any]) -> List[str]:
        if tool_name in {"write_file", "edit_file", "apply_patch"}:
            path = tool_args.get("path") or tool_args.get("file_path")
            return [path] if isinstance(path, str) and path else []
        if tool_name == "write_files":
            files = tool_args.get("files")
            if isinstance(files, dict):
                return [str(path) for path in files.keys()]
            return []
        if tool_name == "move_file":
            dst = tool_args.get("dst")
            if isinstance(dst, str) and dst:
                return [dst]
        return []
