import ast
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from domains.code_intelligence.saguaro_substrate import SaguaroSubstrate


class SaguaroTools:
    """Tool facade for substrate-backed Saguaro operations."""

    def __init__(self, substrate: SaguaroSubstrate, root_dir: str = "."):
        self.substrate = substrate
        self.root_dir = os.path.abspath(root_dir)

    def skeleton(self, path: str) -> str:
        return self.substrate.agent_skeleton(path)

    def slice(self, target: str) -> str:
        if "." in target:
            # Historical behavior: `file_or_class.symbol`.
            return self.substrate.agent_slice(target)
        return self.substrate.agent_slice(target)

    def compare(
        self,
        *,
        target: str = ".",
        candidates: Optional[List[str]] = None,
        corpus_ids: Optional[List[str]] = None,
        fleet_root: Optional[str] = None,
        top_k: int = 10,
        ttl_hours: float = 72.0,
    ) -> str:
        return self.substrate.compare(
            target=target,
            candidates=candidates,
            corpus_ids=corpus_ids,
            fleet_root=fleet_root,
            top_k=top_k,
            ttl_hours=ttl_hours,
        )

    def corpus(
        self,
        action: str = "list",
        *,
        path: Optional[str] = None,
        corpus_id: Optional[str] = None,
        alias: Optional[str] = None,
        ttl_hours: float = 24.0,
        quarantine: bool = True,
        trust_level: str = "medium",
        build_profile: str = "auto",
        include_expired: bool = False,
        rebuild: bool = False,
    ) -> str:
        return self.substrate.corpus(
            action=action,
            path=path,
            corpus_id=corpus_id,
            alias=alias,
            ttl_hours=ttl_hours,
            quarantine=quarantine,
            trust_level=trust_level,
            build_profile=build_profile,
            include_expired=include_expired,
            rebuild=rebuild,
        )

    def impact(self, path: str) -> str:
        return self.substrate.agent_impact(path)

    def query(
        self,
        query: str,
        k: int = 5,
        scope: str = "global",
        dedupe_by: str = "entity",
        recall: str | None = None,
        breadth: int | None = None,
        score_threshold: float | None = None,
        stale_file_bias: float | None = None,
        cost_budget: str | None = None,
        corpus_ids: Optional[List[str] | str] = None,
    ) -> str:
        kwargs = {
            "k": k,
            "scope": scope,
            "dedupe_by": dedupe_by,
            "recall": recall,
            "breadth": breadth,
            "score_threshold": score_threshold,
            "stale_file_bias": stale_file_bias,
            "cost_budget": cost_budget,
        }
        if corpus_ids is not None:
            kwargs["corpus_ids"] = corpus_ids
        return self.substrate.agent_query(
            query,
            **kwargs,
        )

    def sync(
        self,
        changed_files: Optional[List[str]] = None,
        deleted_files: Optional[List[str]] = None,
        full: bool = False,
        reason: str = "tool_call",
        action: str = "index",
        peer_id: Optional[str] = None,
        peer_name: Optional[str] = None,
        peer_url: Optional[str] = None,
        auth_token: Optional[str] = None,
        bundle_path: Optional[str] = None,
        workspace_id: Optional[str] = None,
        limit: int = 1000,
    ) -> str:
        return self.substrate.sync(
            changed_files=changed_files,
            deleted_files=deleted_files,
            full=full,
            reason=reason,
            action=action,
            peer_id=peer_id,
            peer_name=peer_name,
            peer_url=peer_url,
            auth_token=auth_token,
            bundle_path=bundle_path,
            workspace_id=workspace_id,
            limit=limit,
        )

    def workspace(
        self,
        action: str = "status",
        limit: int = 200,
        name: Optional[str] = None,
        workspace_id: Optional[str] = None,
        against: str = "main",
        description: str = "",
        switch: bool = False,
        label: str = "manual",
    ) -> str:
        return self.substrate.workspace(
            action=action,
            limit=limit,
            name=name,
            workspace_id=workspace_id,
            against=against,
            description=description,
            switch=switch,
            label=label,
        )

    def daemon(self, action: str = "status", interval: int = 5, lines: int = 200) -> str:
        return self.substrate.daemon(action=action, interval=interval, lines=lines)

    def reality(
        self,
        action: str = "twin",
        run_id: str | None = None,
        limit: int = 500,
    ) -> str:
        return self.substrate.reality(action=action, run_id=run_id, limit=limit)

    def doctor(self) -> str:
        return self.substrate.execute_command("doctor")

    def parallel_skeleton_fetch(
        self, paths: List[str], max_workers: int = 8
    ) -> Dict[str, str]:
        results: Dict[str, str] = {}

        def fetch_one(path: str) -> tuple[str, Optional[str]]:
            try:
                skeleton = self.skeleton(path)
                if skeleton and not skeleton.startswith("Error"):
                    return path, skeleton
                return path, None
            except Exception:
                return path, None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_one, p): p for p in paths}
            for future in as_completed(futures):
                path, skeleton = future.result()
                if skeleton:
                    results[path] = skeleton

        return results

    def parse_file_structure(self, path: str) -> Dict[str, Any]:
        full_path = (
            os.path.join(self.root_dir, path) if not os.path.isabs(path) else path
        )

        if not os.path.exists(full_path):
            return {"error": f"File not found: {path}"}

        try:
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            tree = ast.parse(content)

            classes = []
            functions = []
            imports = []

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = [
                        n.name for n in node.body if isinstance(n, ast.FunctionDef)
                    ]
                    classes.append(
                        {"name": node.name, "line": node.lineno, "methods": methods}
                    )
                elif isinstance(node, ast.FunctionDef):
                    if hasattr(node, "col_offset") and node.col_offset == 0:
                        args = [a.arg for a in node.args.args]
                        functions.append(
                            {"name": node.name, "line": node.lineno, "args": args}
                        )
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        imports.append(f"{module}.{alias.name}")

            return {
                "path": path,
                "classes": classes,
                "functions": functions,
                "imports": imports,
                "line_count": len(content.splitlines()),
            }
        except SyntaxError as e:
            return {"error": f"Syntax error parsing {path}: {e}"}
        except Exception as e:
            return {"error": f"Error parsing {path}: {e}"}

    def find_files_by_pattern(
        self, pattern: str, extensions: Optional[List[str]] = None
    ) -> List[str]:
        if extensions is None:
            extensions = [".py"]

        results = []
        pattern_lower = pattern.lower()

        for root, dirs, files in os.walk(self.root_dir):
            dirs[:] = [
                d
                for d in dirs
                if d
                not in {
                    "venv",
                    ".git",
                    "__pycache__",
                    ".saguaro",
                    "node_modules",
                    ".pytest_cache",
                    ".ruff_cache",
                }
            ]

            for f in files:
                if any(f.endswith(ext) for ext in extensions):
                    if pattern_lower in f.lower():
                        rel_path = os.path.relpath(os.path.join(root, f), self.root_dir)
                        results.append(rel_path)

        return results

    def find_entity_definition(self, entity_name: str) -> Optional[Dict[str, Any]]:
        py_files = self.find_files_by_pattern("", extensions=[".py"])

        for file_path in py_files:
            try:
                structure = self.parse_file_structure(file_path)
                if "error" in structure:
                    continue

                for cls in structure.get("classes", []):
                    if cls["name"] == entity_name:
                        return {"file": file_path, "line": cls["line"], "type": "class"}
                    for method in cls.get("methods", []):
                        if method == entity_name:
                            return {
                                "file": file_path,
                                "line": cls["line"],
                                "type": "method",
                                "class": cls["name"],
                            }

                for func in structure.get("functions", []):
                    if func["name"] == entity_name:
                        return {
                            "file": file_path,
                            "line": func["line"],
                            "type": "function",
                        }
            except Exception:
                continue

        return None

    def verify(
        self,
        path: str = ".",
        engines: str = "native,ruff,semantic,aes",
        auto_fix: bool = False,
        preflight_only: bool = False,
        timeout_seconds: float | None = None,
    ) -> str:
        if preflight_only or timeout_seconds is not None:
            result = self.substrate._api.verify(  # noqa: SLF001
                path=path,
                engines=engines,
                fix=auto_fix,
                preflight_only=preflight_only,
                timeout_seconds=timeout_seconds,
            )
            return json.dumps(result, indent=2)
        return self.substrate.verify(path=path, engines=engines, fix=auto_fix)

    def cpu_scan(
        self,
        path: str = ".",
        arch: str = "x86_64-avx2",
        limit: int = 20,
    ) -> str:
        return self.substrate.cpu_scan(path=path, arch=arch, limit=limit)

    def deadcode(
        self,
        threshold: float = 0.5,
        low_usage_max_refs: int = 1,
        lang: str | None = None,
        evidence: bool = False,
        runtime_observed: bool = False,
        explain: bool = False,
        output_format: str = "json",
    ) -> str:
        return self.substrate.deadcode(
            threshold=threshold,
            low_usage_max_refs=low_usage_max_refs,
            lang=lang,
            evidence=evidence,
            runtime_observed=runtime_observed,
            explain=explain,
            output_format=output_format,
        )

    def low_usage(
        self,
        max_refs: int = 1,
        include_tests: bool = False,
        path: str | None = None,
        limit: int | None = None,
        output_format: str = "json",
    ) -> str:
        return self.substrate.low_usage(
            max_refs=max_refs,
            include_tests=include_tests,
            path=path,
            limit=limit,
            output_format=output_format,
        )

    def unwired(
        self,
        threshold: float = 0.55,
        min_nodes: int = 4,
        min_files: int = 2,
        include_tests: bool = False,
        include_fragments: bool = False,
        max_clusters: int = 20,
        refresh_graph: bool = True,
        output_format: str = "json",
    ) -> str:
        return self.substrate.unwired(
            threshold=threshold,
            min_nodes=min_nodes,
            min_files=min_files,
            include_tests=include_tests,
            include_fragments=include_fragments,
            max_clusters=max_clusters,
            refresh_graph=refresh_graph,
            output_format=output_format,
        )

    def report(self) -> str:
        return self.substrate.agent_report()

    def memory(
        self,
        action: str,
        key: str = None,
        value: str = None,
        tier: str = "working",
        tags: List[str] = None,
        query: str = None,
    ) -> str:
        # Keep text-based command compatibility via substrate command parser.
        if action == "list":
            return self.substrate.execute_command("memory list")

        if action == "read":
            if not key:
                return "Error: 'read' action requires 'key'."
            return self.substrate.execute_command(f"memory read {key} --tier {tier}")

        if action in {"write", "store"}:
            if not key or value is None:
                return "Error: 'write/store' action requires 'key' and 'value'."
            result = self.substrate._api.memory_write(
                key=key, value=str(value), tier=tier
            )
            return json.dumps(result, indent=2)

        if action == "recall":
            if query:
                # API memory is keyed storage; best-effort fallback to list and filter.
                payload = self.substrate.execute_command("memory list")
                try:
                    data = json.loads(payload)
                    matches = []
                    tiers = data.get("tiers", {})
                    for t_name, facts in tiers.items():
                        for k, entry in facts.items():
                            val = str(entry.get("value", ""))
                            if (
                                query.lower() in k.lower()
                                or query.lower() in val.lower()
                            ):
                                matches.append({"tier": t_name, "key": k, "value": val})
                    return json.dumps({"matches": matches}, indent=2)
                except Exception:
                    return payload
            if key:
                return self.substrate.execute_command(
                    f"memory read {key} --tier {tier}"
                )
            return "Error: 'recall' action requires 'query' or 'key'."

        return f"Error: Unknown action '{action}'."
