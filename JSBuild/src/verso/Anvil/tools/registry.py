import fnmatch
import os
import re
import subprocess
from typing import Any, Dict, List
from tools.file_ops import FileOps
from tools.shell import ShellOps
from tools.saguaro_tools import SaguaroTools
from domains.code_intelligence.saguaro_substrate import SaguaroSubstrate
from core.upgrade import UpgradeManager
from core.skills import SkillManager
from tools.web_search import search_web
from tools.web_fetch import fetch_url
from tools.grep import grep
from tools.glob import glob_files
from tools.think import think
from tools.notify_user import notify_user
from tools.verify import (
    verify_all,
    verify_aes_tool,
    verify_syntax_tool,
    verify_lint_tool,
    verify_types_tool,
    run_tests_tool,
    run_tests_suspended_tool,
)
from shared_kernel.event_store import get_event_store
from tools.visualize import visualize_code
from tools.delegate import DelegateTool
from tools.subagent_tool import ExecuteSubagentTaskTool
from tools.browser import browser_visit, browser_click, browser_screenshot
from tools.memory_bank import update_memory_bank
from tools.lsp import lsp_get_definition, lsp_find_references, lsp_get_diagnostics
from tools.debug import run_with_debugger, self_heal_diagnose
from tools.git_ops import GitOperationsManager
from tools.analysis import analyze_codebase_tool
from tools.specialized_search import (
    search_arxiv,
    search_finance,
    search_scholar,
    search_news,
    search_patents,
)
from tools.arxiv_search import fetch_arxiv_paper
from tools.forum_search import (
    search_hackernews,
    search_reddit,
    search_stackoverflow,
)
from domains.tools.resource_manager import ToolResourceManager
from domains.tools.validation import ToolResultValidator
from infrastructure.providers.mcp_provider import get_mcp_provider
from infrastructure.execution.docker_sandbox import get_docker_sandbox


class ToolRegistry:
    def __init__(
        self,
        root_dir=".",
        console: Any = None,
        brain: Any = None,
        semantic_engine: Any = None,
        agent: Any = None,
    ):
        self.console = console
        self.brain = brain
        self.semantic_engine = semantic_engine
        self.agent = agent
        audit_db = getattr(agent, "audit_db", None)
        session_id = getattr(agent, "session_id", None)
        self.file_ops = FileOps(root_dir, audit_db=audit_db, session_id=session_id)
        max_runtime = 30
        if agent is not None and hasattr(agent, "approval_manager"):
            max_runtime = getattr(
                agent.approval_manager.policy, "max_command_runtime", 30
            )
        self.shell_ops = ShellOps(
            root_dir,
            env_info=getattr(agent, "env_info", {}),
            max_runtime=max_runtime,
            audit_db=audit_db,
            session_id=session_id,
        )
        self.saguaro = SaguaroSubstrate(root_dir)
        self.saguaro_tools = SaguaroTools(self.saguaro)
        self.upgrade_manager = UpgradeManager(root_dir)
        self.skill_manager = SkillManager()
        self.delegate_tool = DelegateTool(console=self.console, brain=self.brain)
        self.subagent_tool = ExecuteSubagentTaskTool(agent=self.agent)
        self.resource_manager = ToolResourceManager()
        self.validator = ToolResultValidator()
        self.mcp = get_mcp_provider()
        self.sandbox = get_docker_sandbox()
        self.git_ops = GitOperationsManager(root_dir)

        # Initialize MCP connections in background if needed
        # (Usually done during agent startup, but we'll ensure availability)

        self.tools = {
            "read_file": self.file_ops.read_file,
            "write_file": self.file_ops.write_file,
            "edit_file": self.file_ops.edit_file,
            "read_files": self.file_ops.read_files,
            "write_files": self.file_ops.write_files,
            "list_dir": self.file_ops.list_dir,
            "delete_file": self.file_ops.delete_file,
            "move_file": self.file_ops.move_file,
            "list_backups": self.file_ops.list_backups,
            "rollback_file": self.file_ops.rollback_file,
            "apply_patch": self.file_ops.apply_patch,
            "run_command": self.shell_ops.run_command,
            "glob": glob_files,
            "saguaro_query": self.saguaro_tools.query,
            "skeleton": self.saguaro_tools.skeleton,
            "slice": self.saguaro_tools.slice,
            "impact": self.saguaro_tools.impact,
            "query": self.saguaro_tools.query,
            "verify": self.saguaro_tools.verify,
            "cpu_scan": self.saguaro_tools.cpu_scan,
            "deadcode": self.saguaro_tools.deadcode,
            "low_usage": self.saguaro_tools.low_usage,
            "memory": self.saguaro_tools.memory,
            "saguaro_sync": self.saguaro_tools.sync,
            "saguaro_workspace": self.saguaro_tools.workspace,
            "saguaro_daemon": self.saguaro_tools.daemon,
            "saguaro_doctor": self.saguaro_tools.doctor,
            "upgrade": self._upgrade_tool,
            "delegate": self.delegate_tool.execute,
            "execute_subagent_task": self.subagent_tool.execute,
            "activate_skill": self._activate_skill,
            # Canonical web research tools for specialist agents.
            "web_search": search_web,
            "web_fetch": fetch_url,
            # Compatibility aliases (non-canonical names kept for older callers).
            "search_web": search_web,
            "fetch_url": fetch_url,
            # Agentic Thinking Tools
            "think": think,
            "notify_user": notify_user,
            "verify_all": verify_all,
            "verify_aes": verify_aes_tool,
            "verify_syntax": verify_syntax_tool,
            "verify_lint": verify_lint_tool,
            "verify_types": verify_types_tool,
            "run_tests": run_tests_tool,
            "run_tests_suspended": run_tests_suspended_tool,
            "visualize": visualize_code,
            "semantic_search": self._semantic_search_tool,
            "grep": grep,
            "grep_search": self._grep_search_tool,
            "find_by_name": self._find_by_name_tool,
            "lsp_definition": lsp_get_definition,
            "lsp_references": lsp_find_references,
            "lsp_diagnostics": lsp_get_diagnostics,
            "debug": run_with_debugger,
            "self_heal_diagnose": self_heal_diagnose,
            "git_create_branch": self._git_create_branch_tool,
            "git_smart_commit": self._git_smart_commit_tool,
            "git_create_pr": self._git_create_pr_tool,
            "browser_visit": browser_visit,
            "browser_click": browser_click,
            "browser_screenshot": browser_screenshot,
            "update_memory_bank": update_memory_bank,
            "analyze_codebase": self._analyze_codebase_tool,
            "saguaro_index": self._saguaro_index_tool,
            "search_arxiv": search_arxiv,
            "fetch_arxiv_paper": fetch_arxiv_paper,
            "search_finance": search_finance,
            "search_scholar": search_scholar,
            "search_news": search_news,
            "search_patents": search_patents,
            "search_reddit": search_reddit,
            "search_hackernews": search_hackernews,
            "search_stackoverflow": search_stackoverflow,
            "export_audit": lambda output_path=".anvil/audit_export.json": (
                self.agent.history.export_audit(output_path)
                if self.agent and hasattr(self.agent, "history")
                else "Error: history unavailable."
            ),
        }

    def _upgrade_tool(self, action="check"):
        if action == "check":
            avail, msg = self.upgrade_manager.check_updates()
            return f"Status: {msg}"
        elif action == "perform":
            return self.upgrade_manager.update()
        return "Error: Unknown action."

    def _activate_skill(self, skill_name: str):
        skill = self.skill_manager.get_skill(skill_name)
        if not skill:
            return f"Error: Skill '{skill_name}' not found. Available skills: {[s.name for s in self.skill_manager.skills.values()]}"

        return f"ACTIVATING SKILL: {skill.name}\nDESCRIPTION: {skill.description}\n\nINSTRUCTIONS:\n{skill.content}"

    def _semantic_search_tool(self, query: str, k: int = 5):
        """Compatibility alias for Saguaro semantic query."""
        return self.saguaro_tools.query(query, k=k)

    def _grep_search_tool(
        self,
        pattern: str,
        path: str = ".",
        is_regex: bool = False,
        file_pattern: str = "*",
    ) -> str:
        search_root = os.path.abspath(os.path.join(self.file_ops.root_dir, path))
        if not os.path.exists(search_root):
            return f"Error: path not found: {path}"

        rg_cmd = ["rg", "-n", "--color", "never"]
        if not is_regex:
            rg_cmd.append("-F")
        if file_pattern:
            rg_cmd.extend(["-g", file_pattern])
        rg_cmd.extend([pattern, search_root])
        try:
            proc = subprocess.run(
                rg_cmd,
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode in {0, 1}:
                output = (proc.stdout or "").strip()
                return output or "No matches found."
        except FileNotFoundError:
            pass

        # Fallback when ripgrep is unavailable.
        matches = []
        for root, dirs, files in os.walk(search_root):
            dirs[:] = [
                d for d in dirs if d not in {".git", "venv", "__pycache__", ".saguaro"}
            ]
            for filename in files:
                if file_pattern and not fnmatch.fnmatch(filename, file_pattern):
                    continue
                abs_path = os.path.join(root, filename)
                try:
                    with open(
                        abs_path, "r", encoding="utf-8", errors="ignore"
                    ) as handle:
                        for line_no, line in enumerate(handle, start=1):
                            hay = line if is_regex else line.lower()
                            needle = pattern if is_regex else pattern.lower()
                            matched = (
                                bool(re.search(pattern, line))
                                if is_regex
                                else needle in hay
                            )
                            if matched:
                                rel = os.path.relpath(abs_path, self.file_ops.root_dir)
                                matches.append(f"{rel}:{line_no}:{line.rstrip()}")
                except Exception:
                    continue
        return "\n".join(matches) if matches else "No matches found."

    def _find_by_name_tool(
        self, pattern: str, path: str = ".", recursive: bool = True
    ) -> str:
        search_root = os.path.abspath(os.path.join(self.file_ops.root_dir, path))
        if not os.path.exists(search_root):
            return f"Error: path not found: {path}"
        matches = []
        if recursive:
            for root, dirs, files in os.walk(search_root):
                dirs[:] = [
                    d
                    for d in dirs
                    if d not in {".git", "venv", "__pycache__", ".saguaro"}
                ]
                for filename in files:
                    if fnmatch.fnmatch(filename, pattern):
                        abs_path = os.path.join(root, filename)
                        matches.append(
                            os.path.relpath(abs_path, self.file_ops.root_dir)
                        )
        else:
            for filename in os.listdir(search_root):
                abs_path = os.path.join(search_root, filename)
                if os.path.isfile(abs_path) and fnmatch.fnmatch(filename, pattern):
                    matches.append(os.path.relpath(abs_path, self.file_ops.root_dir))
        return "\n".join(sorted(matches)) if matches else "No files matched."

    def _analyze_codebase_tool(self, topic: str):
        """Bridge to autonomous code analysis loop."""
        return analyze_codebase_tool(topic, agent_repl=self.agent)

    def _saguaro_index_tool(self, path: str = "."):
        """Force indexing of the workspace."""
        try:
            self.saguaro.index(path)
            if self.semantic_engine:
                self.semantic_engine._indexed = True
            return f"Successfully indexed repository at {path}."
        except Exception as e:
            return f"Error indexing repository: {e}"

    def _git_create_branch_tool(self, feature_name: str):
        try:
            return self.git_ops.create_feature_branch(feature_name)
        except Exception as e:
            return f"Error creating branch: {e}"

    def _git_smart_commit_tool(self, files, message: str):
        try:
            if isinstance(files, str):
                files = [f.strip() for f in files.split(",") if f.strip()]
            if not isinstance(files, list) or not files:
                return (
                    "Error: files must be a non-empty list or comma-separated string."
                )
            return self.git_ops.smart_commit(files, message)
        except Exception as e:
            return f"Error creating commit: {e}"

    def _git_create_pr_tool(self, title: str, body: str, base: str = "main"):
        try:
            return self.git_ops.create_pull_request(title, body, base=base)
        except Exception as e:
            return f"Error creating pull request: {e}"

    def _mcp_tool_handler(self, server_name: str, tool_name: str, **kwargs):
        """Generic handler for MCP tools."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        result = loop.run_until_complete(
            self.mcp.call_tool(server_name, tool_name, kwargs)
        )
        return str(result)

    def register_mcp_tools(self):
        """Dynamically registers tools from connected MCP servers."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        loop.run_until_complete(self.mcp.connect_all())
        all_tools = loop.run_until_complete(self.mcp.list_all_tools())

        for server_name, server_tools in all_tools.items():
            for tool in server_tools:
                # Format: mcp_brave_search_web_search
                mcp_name = f"mcp_{server_name.replace('-', '_')}_{tool.name}"
                self.tools[mcp_name] = (
                    lambda t=tool, s=server_name, **kw: self._mcp_tool_handler(
                        s, t.name, **kw
                    )
                )

    def dispatch(self, tool_name, arguments):
        """
        Executes a tool by name with given arguments.
        """
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found."

        modifying_tools = {
            "write_file",
            "edit_file",
            "write_files",
            "apply_patch",
            "delete_file",
            "move_file",
            "rollback_file",
        }
        direct_write_without_agent = tool_name in modifying_tools and self.agent is None
        if (
            tool_name in modifying_tools
            and self.agent is not None
            and not getattr(self.agent, "_active_tool_execution", False)
        ):
            return (
                "Error: Write-capable tools must execute via BaseAgent._execute_tool "
                "to enforce post-write verification hooks."
            )

        try:
            func = self.tools[tool_name]

            # --- RESOURCE MANAGEMENT (Budgets & Timeouts) ---
            # We use a helper to run the sync function with a timeout
            import asyncio

            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            result = loop.run_until_complete(
                self.resource_manager.execute_with_budget(tool_name, func, arguments)
            )

            # --- VALIDATION ---
            val_error = self.validator.validate(tool_name, result)
            if val_error:
                result = val_error

            if direct_write_without_agent and not str(result).startswith("Error"):
                blocking = self._verify_direct_write(tool_name, arguments)
                if blocking:
                    rule_list = ", ".join(
                        sorted({item.get("rule_id", "UNKNOWN") for item in blocking})
                    )
                    return (
                        "AES GATE: direct write failed post-write verification "
                        f"({len(blocking)} blocking findings: {rule_list})."
                    )

            # --- EVENT SOURCING ---
            try:
                get_event_store().emit(
                    event_type="TOOL_EXECUTED",
                    source=getattr(self.agent, "name", "unknown_agent"),
                    payload={
                        "tool": tool_name,
                        "args": str(arguments)[:1000],  # Truncate args if too large
                        "success": (
                            True if not str(result).startswith("Error") else False
                        ),
                    },
                    metadata={"elapsed": None},
                )
            except Exception:
                pass

            return result
        except TypeError as e:
            return f"Error calling '{tool_name}': {str(e)}"
        except Exception as e:
            return f"Error executing '{tool_name}': {str(e)}"

    def _extract_write_targets(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> List[str]:
        if tool_name in {
            "write_file",
            "edit_file",
            "apply_patch",
            "delete_file",
            "rollback_file",
        }:
            path = arguments.get("path") or arguments.get("file_path")
            return [path] if isinstance(path, str) and path else []
        if tool_name == "write_files":
            files = arguments.get("files")
            if isinstance(files, dict):
                return [str(path) for path in files.keys()]
            return []
        if tool_name == "move_file":
            dst = arguments.get("dst")
            return [dst] if isinstance(dst, str) and dst else []
        return []

    def _verify_direct_write(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        try:
            from saguaro.sentinel.verifier import SentinelVerifier
        except Exception:
            return []

        targets = self._extract_write_targets(tool_name, arguments)
        if not targets:
            return []

        verifier = SentinelVerifier(
            repo_path=self.file_ops.root_dir, engines=["native", "aes"]
        )
        blocking: List[Dict[str, Any]] = []
        for target in targets:
            try:
                violations = verifier.verify_all(path_arg=target)
            except Exception:
                continue
            for item in violations:
                severity = str(item.get("severity", "")).upper()
                closure = str(item.get("closure_level", "")).lower()
                if closure == "blocking" or severity in {"P0", "P1", "ERROR"}:
                    blocking.append(item)

        return blocking

    def get_schemas(self):
        """
        Returns the tool schemas from tools/schema.py.
        """
        from tools.schema import TOOL_SCHEMAS

        return TOOL_SCHEMAS
