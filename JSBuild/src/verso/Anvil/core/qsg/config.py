from dataclasses import dataclass
from typing import Any


@dataclass
class QSGConfig:
    """Configuration for Quantum-Inspired Speculative Generation pipeline."""

    bond_dim: int = 32
    coherence_range: int = 64
    grover_iterations: int = 3
    jacobi_iterations: int = 2
    hopfield_beta: float = 15.0
    temperature: float = 1.0
    top_k: int = 50
    use_coconut_reasoning: bool = False

    # COCONUT enhancements
    use_coconut: bool = True  # Enable COCONUT multi-path reasoning
    use_grover: bool = True  # Enable Grover amplitude amplification
    coconut_paths: int = 8  # Number of parallel reasoning paths
    use_self_consistency: bool = (
        True  # Enable DeepSeek-R1 style self-consistency verification
    )

    # Speculative settings
    speculative_drafts: int = 4
    acceptance_threshold: float = 0.8

    # Continuous batching (disabled by default for backward compatibility)
    continuous_batching_enabled: bool = False
    max_active_requests: int = 64
    max_pending_requests: int = 4096
    scheduler_policy: str = "fcfs"
    batch_wait_timeout_ms: int = 2
    max_batch_state_rows: int = 8192
    max_prefill_rows_per_iteration: int = 1024
    state_page_rows: int = 128
    state_compaction_soft_threshold: float = 0.18
    state_compaction_hard_threshold: float = 0.30
    semantic_resonance_timeout_ms: int = 4
    continuous_interleaved_streams: bool = False

    # Parallel-first planner controls
    parallel_prompt_lookup_enabled: bool = True
    parallel_jacobi_lookahead_enabled: bool = True
    medusa_head_enabled: bool = True
    medusa_head_max_draft_tokens: int = 4
    medusa_head_top_k: int = 8
    medusa_head_acceptance_floor: float = 0.20
    hydra_head_enabled: bool = True
    hydra_head_max_draft_tokens: int = 4
    hydra_head_top_k: int = 8
    hydra_head_acceptance_floor: float = 0.22
    hydra_head_blend_alpha: float = 0.55
    parallel_ssd_bridge_enabled: bool = True
    parallel_replacement_enabled: bool = True
    parallel_replacement_max_tree_width: int = 4
    parallel_replacement_acceptance_floor: float = 0.20
    parallel_replacement_max_draft_tokens: int = 6
    parallel_ar_recovery_enabled: bool = True
    block_diffusion_enabled: bool = True
    block_diffusion_force: bool = False

    # Runtime contract controls
    native_runtime_authority: bool = True
    capability_digest: str = ""
    performance_envelope_path: str = ""
    controller_frontier_mode: str = "adaptive"
    controller_drift_mode: str = "adaptive"
    memory_tier_policy_mode: str = "adaptive"
    memory_prompt_cache_hit_threshold: float = 0.40
    memory_latent_replay_threshold: float = 0.60
    repo_delta_memory_window: int = 8
    execution_capsule_version: int = 2
    latent_packet_abi_version: int = 2
    delta_watermark: dict[str, Any] | None = None
    lineage_prefix_reuse_enabled: bool = True
    reasoning_lane_default: str = "strict"
