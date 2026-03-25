"""
Multi-Agent Evidence Gatherer

Distributes evidence gathering across multiple parallel agents with independent context windows.
Like Claude Code/Antigravity - avoids single-context overflow by delegating to subagents.

Architecture:
- Splits large file sets into chunks that fit in 300k tokens
- Spawns parallel agents, each with their own 200k context window
- Uses COCONUT multi-path reasoning to aggregate findings
- Returns unified evidence dict for synthesis
"""

import os
import concurrent.futures
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)
import numpy as np
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from config.settings import ORCHESTRATION_CONFIG


class MultiAgentEvidenceGatherer:
    """
    Distributes evidence gathering across multiple agents with independent contexts.

    Each agent:
    - Has its own 200k context window
    - Analyzes a subset of files
    - Returns structured findings

    COCONUT aggregates findings into unified evidence.
    """

    def __init__(self, brain, console, registry):
        self.brain = brain
        self.console = console
        self.registry = registry

        # Context budget per agent
        self.agent_context_limit = 300000

        # Max parallel agents (avoid overwhelming system)
        self.max_agents = int(
            max(1, ORCHESTRATION_CONFIG.get("max_parallel_agents", 4))
        )
        
        # Path entanglement: Shared quantum-inspired state for coherent multi-agent reasoning
        # Each agent's reasoning contributes to this shared entangled state
        self._entangled_state = None
        self._entanglement_dim = 512
        self._entanglement_lock = None  # Thread-safe state updates

    def gather_evidence(
        self,
        query: str,
        candidate_files: List[str],
        complexity_profile: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Parallel evidence gathering using multiple agents.

        Args:
            query: The user's question/task
            candidate_files: List of file paths to analyze

        Returns:
            Dict with:
                - agent_summaries: List of summaries from each agent
                - files_analyzed: Total files analyzed
                - file_contents: Dict of file_path -> content/skeleton
                - coconut_refined: Aggregated embedding
                - all_results: Raw results from all agents
        """
        guidance = self._resolve_gathering_guidance(complexity_profile)
        effective_context_limit = guidance.get(
            "agent_context_limit", self.agent_context_limit
        )
        configured_parallel_cap = int(
            max(1, ORCHESTRATION_CONFIG.get("max_parallel_agents", self.max_agents))
        )
        effective_max_agents = min(
            guidance.get("max_agents", self.max_agents),
            configured_parallel_cap,
        )

        # 1. Split files into chunks that fit in agent context windows
        file_groups = self._chunk_files_by_tokens(
            candidate_files, context_limit=effective_context_limit
        )
        if not file_groups:
            return {
                "agent_summaries": [],
                "files_analyzed": 0,
                "file_contents": {},
                "all_results": [],
                "adaptive_allocation": guidance,
            }

        # Limit to max_agents
        if len(file_groups) > effective_max_agents:
            self.console.print(
                f"  [yellow]⚠ Limiting to {effective_max_agents} agents (had {len(file_groups)} groups)[/yellow]"
            )
            # Merge groups to fit max_agents
            file_groups = self._merge_groups(file_groups, effective_max_agents)

        logger.info(
            f"Gathering evidence with {len(file_groups)} agents for query: {query}"
        )
        self.console.print(
            f"  [cyan]→ Spawning {len(file_groups)} analysis agents in parallel...[/cyan]"
        )

        # Initialize path entanglement for coherent multi-agent reasoning
        import threading
        self._entanglement_lock = threading.Lock()
        self._initialize_entangled_state(query, len(file_groups))

        # 2. Spawn agents
        agent_results = []
        execution_mode = ORCHESTRATION_CONFIG.get("execution_mode", "parallel")
        latent_depth = int(max(1, int(getattr(complexity_profile, "coconut_depth", 1) or 1)))
        logger.debug(f"Multi-agent execution mode: {execution_mode}")
        self.console.print(
            f"  [cyan]→ Spawning {len(file_groups)} analysis agents in {execution_mode} mode...[/cyan]"
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=self.console,
        ) as progress:
            task_id = progress.add_task(
                f"Analyzing with {len(file_groups)} agents", total=len(file_groups)
            )

            if execution_mode == "sequential":
                for idx, group in enumerate(file_groups):
                    try:
                        result = self._analyze_file_group(
                            query,
                            group,
                            idx,
                            context_limit=effective_context_limit,
                            latent_depth=latent_depth,
                        )
                        agent_results.append(result)
                        progress.update(task_id, advance=1)
                    except Exception as e:
                        self.console.print(f"    [red]✗ Agent failed: {e}[/red]")
                        progress.update(task_id, advance=1)
            else:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=max(1, min(len(file_groups), effective_max_agents))
                ) as executor:
                    futures = [
                        executor.submit(
                            self._analyze_file_group,
                            query,
                            group,
                            idx,
                            effective_context_limit,
                            latent_depth,
                        )
                        for idx, group in enumerate(file_groups)
                    ]

                    # Collect results as they complete
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            result = future.result()
                            agent_results.append(result)
                            progress.update(task_id, advance=1)
                        except Exception as e:
                            self.console.print(f"    [red]✗ Agent failed: {e}[/red]")
                            progress.update(task_id, advance=1)

        if not agent_results:
            self.console.print("  [red]✗ All agents failed[/red]")
            return {
                "agent_summaries": [],
                "files_analyzed": 0,
                "file_contents": {},
                "all_results": [],
            }

        self.console.print(
            f"  [green]✓ {len(agent_results)} agents completed analysis[/green]"
        )

        # 3. COCONUT aggregation (multi-path consensus)
        logger.info(f"Aggregating results from {len(agent_results)} agents")
        aggregated = self._coconut_aggregate(
            query, agent_results, complexity_profile=complexity_profile
        )
        aggregated["adaptive_allocation"] = guidance

        return aggregated

    def _analyze_file_group(
        self,
        query: str,
        files: List[str],
        agent_id: int,
        context_limit: Optional[int] = None,
        latent_depth: int = 1,
    ) -> Dict[str, Any]:
        """
        Single agent analyzes its file subset.

        This runs in its own thread with independent context.

        Args:
            query: The question to answer
            files: Subset of files for this agent
            agent_id: Agent identifier

        Returns:
            Dict with summary, findings, and file contents
        """
        # Read files for this agent
        file_contents = {}
        total_tokens = 0
        files_loaded = 0

        limit = int(context_limit or self.agent_context_limit)

        for file_path in files:
            try:
                # Check token budget
                if total_tokens > limit:
                    break

                # Try full read first
                content = self.registry.dispatch("read_file", {"file_path": file_path})

                if content and not content.startswith("Error"):
                    tokens = len(content) // 4  # Rough estimate

                    # If file too large, use skeleton
                    if tokens > 15000:
                        from tools.saguaro_tools import SaguaroTools
                        from domains.code_intelligence.saguaro_substrate import SaguaroSubstrate

                        saguaro = SaguaroSubstrate()
                        saguaro_tools = SaguaroTools(saguaro)

                        skeleton = saguaro_tools.skeleton(file_path)
                        if skeleton and not skeleton.startswith("Error"):
                            file_contents[file_path] = f"[SKELETON]\n{skeleton}"
                            total_tokens += len(skeleton) // 4
                        else:
                            # Truncate if skeleton fails
                            file_contents[file_path] = content[:10000]
                            total_tokens += 2500
                    else:
                        file_contents[file_path] = content
                        total_tokens += tokens

                    files_loaded += 1

            except Exception as e:
                self.console.print(
                    f"      [dim]Agent {agent_id}: Failed to read {file_path}: {e}[/dim]"
                )

        # Build analysis prompt
        files_text = "\n\n".join(
            [
                f"### {path}\n```\n{content[:5000]}\n```"
                for path, content in list(file_contents.items())[:10]
            ]
        )

        prompt = f"""Analyze these files to answer the question: "{query}"

Files analyzed ({files_loaded} files):
{files_text}

Provide a structured analysis:
1. Key findings relevant to the question
2. Important code patterns or implementations found
3. Files that are most relevant
4. Any missing information needed

Be concise but thorough."""

        # Generate analysis using brain
        messages = [
            {
                "role": "system",
                "content": "You are a code analysis agent. Analyze the provided files and extract key information.",
            },
            {"role": "user", "content": prompt},
        ]

        # Stream response
        analysis = ""
        try:
            for chunk in self.brain.stream_chat(
                messages, max_tokens=4000, temperature=0.0
            ):
                analysis += chunk
        except Exception as e:
            analysis = f"Error during analysis: {e}"

        agent_embedding: Optional[np.ndarray] = None
        # Update entangled state with this agent's reasoning
        try:
            agent_embedding = np.asarray(
                self.brain.embeddings(analysis[:1000]), dtype=np.float32
            ).reshape(-1)
            self._update_entangled_state(agent_id, agent_embedding)
        except Exception as e:
            logger.debug(f"Agent {agent_id} entanglement update failed: {e}")
        latent_payload = self._build_latent_payload(
            embedding=agent_embedding,
            depth_used=latent_depth,
        )
        latent_state = latent_payload.get("state")

        return {
            "agent_id": agent_id,
            "files": files,
            "files_loaded": files_loaded,
            "summary": analysis,
            "file_contents": file_contents,
            "query": query,
            "tokens_used": total_tokens,
            "latent": latent_payload,
            "latent_state": latent_state,
            "latent_tool_signals": [],
            "latent_reinjections": 0,
        }

    @staticmethod
    def _build_latent_payload(
        embedding: Optional[np.ndarray], depth_used: int
    ) -> Dict[str, Any]:
        state: Optional[List[float]] = None
        if embedding is not None:
            arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
            if arr.size > 0 and np.isfinite(arr).all():
                norm = float(np.linalg.norm(arr))
                if norm > 1e-8:
                    arr = arr / norm
                state = [float(v) for v in arr]
        return {
            "state": state,
            "state_dim": int(len(state) if state else 0),
            "reinjections": 0,
            "tool_signals": [],
            "depth_used": int(max(1, depth_used)),
            "seeded_from_master": False,
        }

    def _coconut_aggregate(
        self,
        query: str,
        agent_results: List[Dict[str, Any]],
        complexity_profile: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Use COCONUT multi-path reasoning to aggregate agent findings.

        Each agent's result becomes a "thought path" in COCONUT.
        Amplitude scoring selects the most relevant findings.

        Uses native COCONUT bridge if available for faster processing.

        Args:
            query: Original query
            agent_results: List of results from each agent

        Returns:
            Aggregated evidence dict
        """
        self.console.print(
            "  [dim]→ COCONUT multi-path aggregation of agent findings...[/dim]"
        )

        # Collect all file contents from all agents
        all_file_contents = {}
        for result in agent_results:
            all_file_contents.update(result["file_contents"])

        # Collect summaries
        agent_summaries = [r["summary"] for r in agent_results]

        # Try COCONUT aggregation with native bridge fallback
        refined = None
        paths = None  # Initialize paths
        amplitudes = None  # Initialize amplitudes
        aggregation_method = "none"

        target_dim = 512
        num_agents = max(1, len(agent_results))
        configured_paths = getattr(complexity_profile, "coconut_paths", None)
        if isinstance(configured_paths, (int, float)):
            num_paths = int(max(1, min(12, int(configured_paths))))
        else:
            num_paths = min(num_agents, 12)

        try:
            # Convert each agent summary to embedding
            embeddings_list = []
            for summary in agent_summaries:
                try:
                    emb = self.brain.embeddings(
                        summary[:1000]
                    )  # Truncate long summaries
                    embeddings_list.append(emb)
                except Exception:
                    # Fallback: zero embedding
                    embeddings_list.append([0.0] * 512)

            # Convert to numpy array
            embeddings = np.array(embeddings_list, dtype=np.float32)

            # Ensure 2D shape [batch, dim]
            if len(embeddings.shape) == 1:
                embeddings = embeddings.reshape(1, -1)

            # Pad or truncate to 512 dims
            if embeddings.shape[1] != 512:
                if embeddings.shape[1] > 512:
                    embeddings = embeddings[:, :512]
                else:
                    pad_width = ((0, 0), (0, 512 - embeddings.shape[1]))
                    embeddings = np.pad(embeddings, pad_width, mode="constant")

            mean_embedding = np.mean(embeddings, axis=0, keepdims=True)
            query_emb = self.brain.embeddings(query[:500])
            query_emb = np.array([query_emb], dtype=np.float32)
            if query_emb.shape[1] != target_dim:
                if query_emb.shape[1] > target_dim:
                    query_emb = query_emb[:, :target_dim]
                else:
                    pad_width = ((0, 0), (0, target_dim - query_emb.shape[1]))
                    query_emb = np.pad(query_emb, pad_width, mode="constant")
            from core.native.coconut_bridge import CoconutNativeBridge

            bridge = CoconutNativeBridge(strict_native=True)

            # Use native ops for multi-path reasoning
            # Expand paths from mean embedding
            paths = bridge.expand_paths(mean_embedding, num_paths, noise_scale=0.05)

            # Score paths against context (query embedding)
            amplitudes = bridge.score_paths(paths, query_emb)
            amplitudes = np.asarray(amplitudes, dtype=np.float32).reshape(-1)
            if amplitudes.size == 0:
                raise RuntimeError("Native COCONUT returned empty amplitudes.")
            amplitudes = self._stable_softmax(amplitudes)

            # Aggregate paths using amplitude weighting
            refined = bridge.aggregate_paths(paths, amplitudes)
            refined = np.asarray(refined, dtype=np.float32)

            aggregation_method = "native_bridge"
            self.console.print(
                "  [green]✓ COCONUT native bridge aggregation complete[/green]"
            )

        except Exception as e:
            raise RuntimeError(
                "Native COCONUT aggregation failed; Python fallback path is disabled."
            ) from e

        return {
            "agent_summaries": agent_summaries,
            "files_analyzed": sum(r["files_loaded"] for r in agent_results),
            "file_contents": all_file_contents,
            "coconut_refined": refined,
            "aggregation_method": aggregation_method,
            "all_results": agent_results,
            "total_agents": len(agent_results),
            "total_tokens_used": sum(r.get("tokens_used", 0) for r in agent_results),
            "coconut_paths": paths,  # New: return paths
            "coconut_amplitudes": amplitudes,  # New: return amplitudes
            "entanglement_correlation": self._get_entangled_correlation(),  # Path entanglement
        }

    def _chunk_files_by_tokens(
        self, files: List[str], context_limit: Optional[int] = None
    ) -> List[List[str]]:
        """
        Split files into groups that fit in agent context windows.

        Uses file size estimation to avoid token overflow.

        Args:
            files: List of file paths

        Returns:
            List of file groups
        """
        chunks = []
        current_chunk = []
        current_tokens = 0
        limit = int(context_limit or self.agent_context_limit)

        for file_path in files:
            # Estimate tokens from file size (4 chars/token heuristic)
            try:
                if os.path.exists(file_path):
                    size = os.path.getsize(file_path)
                    # Account for skeleton compression (~50% of full file)
                    estimated_tokens = (size // 4) // 2
                else:
                    estimated_tokens = 5000  # Default estimate
            except Exception:
                estimated_tokens = 5000

            # Check if adding this file would exceed limit
            if (
                current_tokens + estimated_tokens > limit
                and current_chunk
            ):
                # Start new chunk
                chunks.append(current_chunk)
                current_chunk = [file_path]
                current_tokens = estimated_tokens
            else:
                current_chunk.append(file_path)
                current_tokens += estimated_tokens

        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _merge_groups(
        self, groups: List[List[str]], max_groups: int
    ) -> List[List[str]]:
        """
        Merge file groups to fit within max_agents limit.

        Args:
            groups: Original file groups
            max_groups: Maximum number of groups

        Returns:
            Merged groups
        """
        if len(groups) <= max_groups:
            return groups

        # Calculate how many groups to merge together
        merge_factor = (len(groups) + max_groups - 1) // max_groups

        merged = []
        for i in range(0, len(groups), merge_factor):
            # Flatten merged groups
            merged_group = []
            for group in groups[i : i + merge_factor]:
                merged_group.extend(group)
            merged.append(merged_group)

        return merged

    def _resolve_gathering_guidance(
        self, complexity_profile: Optional[Any]
    ) -> Dict[str, int]:
        """
        Resolve adaptive gatherer limits from an optional complexity profile.

        Supported inputs:
        - Adaptive profile fields: subagent_count, recommended_context_budget
        - Legacy score profile: score (1-10)
        """
        max_agents = int(max(1, self.max_agents))
        context_limit = int(max(20000, self.agent_context_limit))

        if complexity_profile is not None:
            suggested_agents = getattr(complexity_profile, "subagent_count", None)
            if suggested_agents is None:
                score = getattr(complexity_profile, "score", None)
                if isinstance(score, (int, float)):
                    scaled = float(score) / 10.0
                    suggested_agents = 1 + int(
                        round(max(0.0, min(1.0, scaled)) * (self.max_agents - 1))
                    )

            if isinstance(suggested_agents, (int, float)):
                max_agents = int(max(1, min(self.max_agents, int(suggested_agents))))

            recommended_budget = getattr(
                complexity_profile, "recommended_context_budget", None
            )
            if isinstance(recommended_budget, (int, float)) and recommended_budget > 0:
                per_agent = int(recommended_budget // max(1, max_agents))
                context_limit = int(
                    max(20000, min(int(self.agent_context_limit), per_agent))
                )

        return {
            "max_agents": int(max_agents),
            "agent_context_limit": int(context_limit),
        }

    @staticmethod
    def _stable_softmax(values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return np.asarray([], dtype=np.float32)
        if not np.isfinite(arr).all():
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        shifted = arr - float(np.max(arr))
        exp = np.exp(shifted)
        denom = float(np.sum(exp))
        if denom <= 1e-12:
            return np.full((arr.size,), 1.0 / float(arr.size), dtype=np.float32)
        return (exp / denom).astype(np.float32)

    @staticmethod
    def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
        arr = np.asarray(matrix, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms <= 1e-8, 1.0, norms)
        return arr / norms

    def _initialize_entangled_state(self, query: str, num_agents: int):
        """
        Initialize shared entangled state for coherent multi-agent reasoning.
        
        The entangled state is a superposition of agent reasoning paths.
        Each agent's analysis collapses part of this superposition,
        but coherence is maintained through shared amplitude correlations.
        
        Args:
            query: The query embedding serves as the initial state
            num_agents: Number of agents to entangle
        """
        try:
            # Get query embedding as basis
            query_emb = self.brain.embeddings(query[:500])
            query_emb = np.array(query_emb, dtype=np.float32)
            
            # Pad to entanglement dimension
            if len(query_emb) < self._entanglement_dim:
                query_emb = np.pad(query_emb, (0, self._entanglement_dim - len(query_emb)))
            elif len(query_emb) > self._entanglement_dim:
                query_emb = query_emb[:self._entanglement_dim]
            
            # Initialize entangled state as tensor product of agent subspaces
            # Each agent gets a "slot" in the entangled state
            self._entangled_state = np.tile(query_emb, (num_agents, 1))
            
            # Add random phase to create superposition (agents will collapse this)
            phases = np.random.randn(num_agents, self._entanglement_dim) * 0.1
            self._entangled_state += phases
            
            # Normalize to unit sphere
            norms = np.linalg.norm(self._entangled_state, axis=1, keepdims=True)
            self._entangled_state = self._entangled_state / (norms + 1e-8)
            
            logger.debug(f"Initialized entangled state for {num_agents} agents")
            
        except Exception as e:
            logger.warning(f"Failed to initialize entanglement: {e}")
            self._entangled_state = None

    def _update_entangled_state(self, agent_id: int, agent_embedding: np.ndarray):
        """
        Update the shared entangled state with an agent's reasoning.
        
        This creates correlation between agents' findings through
        amplitude interference in the shared state.
        
        Args:
            agent_id: The agent's index
            agent_embedding: The agent's analysis embedding
        """
        if self._entangled_state is None or self._entanglement_lock is None:
            return
            
        try:
            with self._entanglement_lock:
                # Normalize agent embedding
                agent_emb = np.array(agent_embedding, dtype=np.float32)
                if len(agent_emb) < self._entanglement_dim:
                    agent_emb = np.pad(agent_emb, (0, self._entanglement_dim - len(agent_emb)))
                elif len(agent_emb) > self._entanglement_dim:
                    agent_emb = agent_emb[:self._entanglement_dim]
                
                norm = np.linalg.norm(agent_emb)
                if norm > 1e-8:
                    agent_emb = agent_emb / norm
                
                # Quantum-inspired interference: agent collapses its slot
                # but the interference pattern affects other slots
                if agent_id < len(self._entangled_state):
                    # Direct update to agent's slot
                    self._entangled_state[agent_id] = agent_emb
                    
                    # Coherent interference with other slots (scaled by 1/sqrt(N))
                    n_agents = len(self._entangled_state)
                    interference_strength = 0.1 / np.sqrt(n_agents)
                    for i in range(n_agents):
                        if i != agent_id:
                            # Add interference term (maintains entanglement)
                            self._entangled_state[i] += interference_strength * agent_emb
                    
                    # Re-normalize all states
                    norms = np.linalg.norm(self._entangled_state, axis=1, keepdims=True)
                    self._entangled_state = self._entangled_state / (norms + 1e-8)
                    
        except Exception as e:
            logger.debug(f"Entanglement update failed for agent {agent_id}: {e}")

    def _get_entangled_correlation(self) -> Optional[np.ndarray]:
        """
        Get the correlation matrix from the entangled state.
        
        This shows how strongly each agent's reasoning is correlated
        with other agents, enabling coherent synthesis.
        
        Returns:
            Correlation matrix [N x N] or None if not available
        """
        if self._entangled_state is None:
            return None
            
        try:
            # Compute pairwise cosine similarities (correlation)
            # Shape: [N x N]
            dots = self._entangled_state @ self._entangled_state.T
            return dots
        except Exception:
            return None
