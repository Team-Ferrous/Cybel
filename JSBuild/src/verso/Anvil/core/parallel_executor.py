"""
Parallel Tool Execution with Saguaro Integration

Executes independent tools simultaneously for massive speedup.
Uses Saguaro for intelligent parallelization decisions.
"""

import concurrent.futures
import logging
import os
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from saguaro.indexing.auto_scaler import calibrate_runtime_profile, load_runtime_profile

from core.context_compression import (
    ensure_context_updates_arg,
    extract_context_updates,
)
from domains.code_intelligence.saguaro_substrate import SaguaroSubstrate

logger = logging.getLogger(__name__)


class ParallelToolExecutor:
    """
    Execute tools in parallel when safe to do so.
    Uses Saguaro to detect dependencies and plan execution.
    """

    def __init__(
        self,
        registry,
        semantic_engine,
        console,
        approval_manager=None,
        tool_executor: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
    ):
        self.registry = registry
        self.semantic_engine = semantic_engine
        self.console = console
        self.approval_manager = approval_manager
        self.tool_executor = tool_executor
        self.max_workers = self._resolve_max_workers()

    def _resolve_max_workers(self) -> int:
        repo_root = os.getcwd()
        profile = calibrate_runtime_profile(repo_root)
        layout = dict(profile.get("selected_runtime_layout") or {})
        configured = int(layout.get("max_parallel_agents", 0) or 0)
        if configured > 0:
            return max(1, configured)
        loaded = load_runtime_profile(repo_root)
        layout = dict(loaded.get("selected_runtime_layout") or {})
        configured = int(layout.get("max_parallel_agents", 0) or 0)
        if configured > 0:
            return max(1, configured)
        return 8

    def execute_tools(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute tools in parallel when possible.

        Uses Saguaro to:
        1. Detect which tools can run in parallel (no dependencies)
        2. Group tools into execution waves
        3. Execute each wave in parallel

        Args:
            tool_calls: List of tool dicts with 'tool' and 'args'

        Returns:
            List of results in original order
        """
        if not tool_calls:
            return []

        # Analyze dependencies using Saguaro
        execution_plan = self._build_execution_plan(tool_calls)

        self.console.print(
            f"[cyan]Executing {len(tool_calls)} tools in {len(execution_plan)} wave(s)[/cyan]"
        )

        all_results = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=self.console,
            transient=True,  # Hide Progress bar after completion to keep terminal clean
        ) as progress:
            for wave_idx, wave in enumerate(execution_plan):
                description = (
                    f"Wave {wave_idx + 1}/{len(execution_plan)} ({len(wave)} tools)"
                )
                task = progress.add_task(description, total=len(wave))

                # Log to the file as well
                logger.info(
                    f"Executing tool wave {wave_idx + 1}: {[tc[1].get('tool') for tc in wave]}"
                )

                wave_results = self._execute_wave(wave, progress, task)
                for idx, result in wave_results:
                    all_results[idx] = result
                progress.update(task, completed=len(wave))

        # Return results in original order
        return [all_results[i] for i in range(len(tool_calls))]

    def _build_execution_plan(
        self, tool_calls: List[Dict[str, Any]]
    ) -> List[List[Tuple[int, Dict]]]:
        """
        Build execution plan with waves of parallel-safe tools.

        Uses Saguaro to detect dependencies:
        - Read operations can always run in parallel
        - Write operations to different files can run in parallel
        - Write operations to same file must be sequential
        - Tools that depend on previous results must wait

        Returns:
            List of waves, each wave is list of (index, tool_call) tuples
        """
        # Simple dependency analysis
        waves = []
        current_wave = []
        written_files = set()
        read_files = set()

        for idx, tool_call in enumerate(tool_calls):
            tool_name = tool_call.get("tool")
            args = tool_call.get("args", {})

            # Check if this tool can run in current wave
            can_run_now = self._can_run_in_wave(
                tool_name, args, written_files, read_files
            )

            if can_run_now:
                current_wave.append((idx, tool_call))

                # Track file operations
                if tool_name in ["write_file", "edit_file"]:
                    written_files.add(args.get("file_path"))
                elif tool_name == "read_file":
                    read_files.add(args.get("file_path"))
            else:
                # Start new wave
                if current_wave:
                    waves.append(current_wave)
                    current_wave = []

                # Add this tool to new wave
                current_wave.append((idx, tool_call))
                written_files = (
                    {args.get("file_path")}
                    if tool_name in ["write_file", "edit_file"]
                    else set()
                )
                read_files = (
                    {args.get("file_path")} if tool_name == "read_file" else set()
                )

        # Add final wave
        if current_wave:
            waves.append(current_wave)

        return waves

    def _can_run_in_wave(
        self, tool_name: str, args: Dict, written_files: set, read_files: set
    ) -> bool:
        """
        Determine if a tool can run in the current wave.

        Rules:
        - Read-only tools can always run in parallel
        - Write to different files can run in parallel
        - Write to file that's been read/written must wait
        """
        file_path = args.get("file_path")

        # Read operations
        if tool_name in ["read_file", "saguaro_query", "query", "skeleton", "slice"]:
            # Can run if file hasn't been written in this wave
            if file_path and file_path in written_files:
                return False
            return True

        # Write operations
        if tool_name in ["write_file", "edit_file"]:
            # Can't run if file already accessed in this wave
            if file_path and (file_path in written_files or file_path in read_files):
                return False
            return True

        # Other tools (web search, etc.) can run in parallel
        return True

    def _execute_wave(
        self, wave: List[Tuple[int, Dict]], progress, task_id
    ) -> List[Tuple[int, Dict]]:
        """
        Execute a wave of tools in parallel.
        """
        results = []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            # Submit all tools in wave
            futures = {
                executor.submit(self._execute_single_tool, tool_call): (idx, tool_call)
                for idx, tool_call in wave
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                idx, tool_call = futures[future]
                try:
                    result = future.result()
                    results.append((idx, result))
                except Exception as e:
                    # Return error result
                    results.append(
                        (
                            idx,
                            {
                                "tool": tool_call.get("tool"),
                                "args": tool_call.get("args"),
                                "result": f"Error: {str(e)}",
                                "success": False,
                                "context_updates": [],
                            },
                        )
                    )

        return results

    def _execute_single_tool(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single tool and return result dict.
        """
        tool_name = tool_call.get("tool")
        tool_args = dict(tool_call.get("args", {}) or {})
        ensure_context_updates_arg(tool_args)
        context_updates = extract_context_updates(tool_args)

        try:
            # Check with approval manager if available
            if self.approval_manager and not self.approval_manager.request_approval(
                tool_name, tool_args
            ):
                return {
                    "tool": tool_name,
                    "args": tool_args,
                    "result": f"Tool '{tool_name}' execution was denied by the user.",
                    "success": False,
                    "context_updates": context_updates,
                }

            if self.tool_executor is not None:
                result = self.tool_executor(tool_name, tool_args)
            else:
                result = self.registry.dispatch(tool_name, tool_args)

            return {
                "tool": tool_name,
                "args": tool_args,
                "result": result,
                "success": True,
                "context_updates": context_updates,
            }

        except Exception as e:
            return {
                "tool": tool_name,
                "args": tool_args,
                "result": f"Error: {str(e)}",
                "success": False,
                "context_updates": context_updates,
            }


class SaguaroParallelSearch:
    """
    Leverage Saguaro for massively parallel semantic searches.
    """

    def __init__(self, saguaro_tools, console):
        self.saguaro_tools = saguaro_tools
        self.console = console
        self._broker = SaguaroQueryBroker(
            self.saguaro_tools.substrate,
            batch_window_ms=int(
                os.getenv("SAGUARO_QUERY_BROKER_WINDOW_MS", "2") or 2
            ),
            max_batch_size=int(
                os.getenv("SAGUARO_QUERY_BROKER_MAX_BATCH", "32") or 32
            ),
        )

    def multi_query_search(
        self, queries: List[str], k: int = 5
    ) -> Dict[str, List[str]]:
        """
        Execute multiple semantic searches in parallel.

        Args:
            queries: List of search queries
            k: Number of results per query

        Returns:
            Dict mapping query -> list of file paths
        """
        results = {}
        futures = {
            query: self._broker.submit(query, k)
            for query in queries
            if str(query or "").strip()
        }

        for query, future in futures.items():
            try:
                results[query] = future.result(timeout=10.0)[:k]
            except Exception as e:
                self.console.print(f"[red]Search failed for '{query}': {e}[/red]")
                results[query] = []

        return results

    def _run_query(self, query: str, k: int) -> List[str]:
        result = self.saguaro_tools.substrate._api.query(query, k=k)
        paths = []
        for item in result.get("results", []):
            file_path = item.get("file")
            if file_path:
                paths.append(file_path)
        return paths

    def parallel_skeleton_fetch(self, file_paths: List[str]) -> Dict[str, str]:
        """
        Fetch code skeletons for multiple files in parallel.

        Uses Saguaro's skeleton feature to get structure without content.
        Much faster than reading full files.
        """
        from tools.saguaro_tools import SaguaroTools
        from domains.code_intelligence.saguaro_substrate import SaguaroSubstrate

        saguaro = SaguaroSubstrate()
        saguaro_tools = SaguaroTools(saguaro)

        skeletons = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(saguaro_tools.skeleton, fp): fp for fp in file_paths
            }

            for future in concurrent.futures.as_completed(futures):
                file_path = futures[future]
                try:
                    skeleton = future.result()
                    skeletons[file_path] = skeleton
                except Exception as e:
                    skeletons[file_path] = f"Error: {e}"

        return skeletons


class SaguaroQueryBroker:
    """Micro-batch concurrent queries over one shared Saguaro substrate."""

    def __init__(
        self,
        substrate: SaguaroSubstrate,
        *,
        batch_window_ms: int = 2,
        max_batch_size: int = 32,
    ) -> None:
        self.substrate = substrate
        self.batch_window_seconds = max(0.0, float(batch_window_ms) / 1000.0)
        self.max_batch_size = max(1, int(max_batch_size))
        self._condition = threading.Condition()
        self._pending: List[Tuple[str, int, concurrent.futures.Future[List[str]]]] = []
        self._stats = {
            "batches": 0,
            "queue_depth_max": 0,
            "queries_routed": 0,
            "batch_size_max": 0,
        }
        self._thread = threading.Thread(
            target=self._run,
            name="saguaro-query-broker",
            daemon=True,
        )
        self._thread.start()

    def submit(self, query: str, k: int) -> concurrent.futures.Future[List[str]]:
        future: concurrent.futures.Future[List[str]] = concurrent.futures.Future()
        with self._condition:
            self._pending.append((str(query), int(k), future))
            self._stats["queue_depth_max"] = max(
                int(self._stats["queue_depth_max"]),
                len(self._pending),
            )
            self._condition.notify()
        return future

    def snapshot_stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._pending:
                    self._condition.wait()
                self._condition.wait(timeout=self.batch_window_seconds)
                batch = self._pending[: self.max_batch_size]
                del self._pending[: len(batch)]
            if not batch:
                continue
            self._dispatch_batch(batch)

    def _dispatch_batch(
        self,
        batch: List[Tuple[str, int, concurrent.futures.Future[List[str]]]],
    ) -> None:
        self._stats["batches"] += 1
        self._stats["queries_routed"] += len(batch)
        self._stats["batch_size_max"] = max(
            int(self._stats["batch_size_max"]),
            len(batch),
        )
        max_k = max(k for _query, k, _future in batch)
        queries = [query for query, _k, _future in batch]
        try:
            payload = self.substrate.batch_query(queries, k=max_k)
        except Exception as exc:
            for _query, _k, future in batch:
                future.set_exception(exc)
            return
        for query, k, future in batch:
            rows = payload.get(query, [])
            future.set_result(
                [str(row.get("file")) for row in rows[:k] if row.get("file")]
            )
