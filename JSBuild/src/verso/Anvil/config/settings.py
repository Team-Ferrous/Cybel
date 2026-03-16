import os

# --- AES ROLLOUT FLAGS ---
AES_CONFIG = {
    "enforcement_enabled": os.getenv("AES_ENFORCEMENT_ENABLED", "1") == "1",
    "block_high_aal": os.getenv("AES_BLOCK_HIGH_AAL", "1") == "1",
    "prompt_contract_required": os.getenv("AES_PROMPT_CONTRACT_REQUIRED", "1") == "1",
    "sentinel_engine_required": os.getenv("AES_SENTINEL_ENGINE_REQUIRED", "1") == "1",
}

AES_ENFORCEMENT_ENABLED = AES_CONFIG["enforcement_enabled"]
AES_BLOCK_HIGH_AAL = AES_CONFIG["block_high_aal"]
AES_PROMPT_CONTRACT_REQUIRED = AES_CONFIG["prompt_contract_required"]
AES_SENTINEL_ENGINE_REQUIRED = AES_CONFIG["sentinel_engine_required"]

# --- GPU CONFIGURATION ---
GPU_CONFIG = {
    "enabled": False,
    "n_gpu_layers": 0,
    "main_gpu": 0,
    "tensor_split": None,
    "force_cpu": True,
}

# --- COCONUT CONFIGURATION ---
COCONUT_CONFIG = {
    "backend": "native",
    "embedding_dim": "auto",
    "use_fft": True,
    "persistent_freq_state": True,
    "deterministic": True,
    "context_budget": 400000,
    "max_paths": 12,
    "max_steps": 8,
}

# --- DYNAMIC COCONUT CONFIGURATION ---
DYNAMIC_COCONUT_CONFIG = {
    "adaptive_min_steps": 1,
    "adaptive_max_steps": 8,
    "adaptive_entropy_threshold": 0.30,
    "adaptive_confidence_threshold": 0.85,
    "max_subagent_slots": 4,
}

# --- MODEL CONFIGURATION ---
# Based on "Agentic Coding Roadmap Strategy"
MASTER_MODEL = "granite4:tiny-h"  # The "Context Holder"
# Recommended for complex agentic tasks: "qwen2.5-coder:7b" or "granite-3.1-dense:8b"
SUB_MODEL = "granite4:tiny-h"  # The "Transient Worker"

# --- OLLAMA CONNECTION ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_LOAD_METHOD = "qsg"
QSG_CONFIG = {
    "bond_dim": 32,
    "grover_iterations": 2,
    "jacobi_iterations": 2,
    "speculative_drafts": 4,
    "use_coconut_reasoning": True,
    "use_grover": True,
}
CONTINUOUS_BATCHING_CONFIG = {
    "enabled": os.getenv("ANVIL_CONTINUOUS_BATCHING_ENABLED", "0") == "1",
    "max_active_requests": int(
        os.getenv("ANVIL_CONTINUOUS_BATCHING_MAX_ACTIVE", "4")
    ),
    "max_pending_requests": int(
        os.getenv("ANVIL_CONTINUOUS_BATCHING_MAX_PENDING", "4096")
    ),
    "scheduler_policy": os.getenv(
        "ANVIL_CONTINUOUS_BATCHING_SCHEDULER", "fcfs"
    ).strip()
    or "fcfs",
    "batch_wait_timeout_ms": int(
        os.getenv("ANVIL_CONTINUOUS_BATCHING_BATCH_WAIT_MS", "2")
    ),
    "semantic_poll_timeout_ms": int(
        os.getenv("ANVIL_CONTINUOUS_BATCHING_POLL_MS", "4")
    ),
}
MODEL_PROFILES = {
    "granite4:tiny-h": {
        "chat_template": "granite",
        "propagator_strategy": "mlp",
        "coconut_mode": "logits_proxy",
        "grover_iterations": 2,
        "coconut_paths": 8,
        "speculative_enabled": False,
        "spec_num_candidates": 4,
        "spec_max_draft_length": 4,
        "spec_acceptance_threshold": 0.65,
    },
    "qwen3.5:9b": {
        "chat_template": "chatml",
        "propagator_strategy": "mlp",
        "coconut_mode": "logits_proxy",
        "grover_iterations": 1,
        "coconut_paths": 8,
        "speculative_enabled": False,
        "spec_num_candidates": 3,
        "spec_max_draft_length": 3,
        "spec_acceptance_threshold": 0.7,
    },
    "qwen3.5:4b": {
        "chat_template": "chatml",
        "propagator_strategy": "mlp",
        "coconut_mode": "logits_proxy",
        "grover_iterations": 1,
        "coconut_paths": 8,
        "speculative_enabled": False,
        "spec_num_candidates": 3,
        "spec_max_draft_length": 3,
        "spec_acceptance_threshold": 0.7,
    },
}

# Strict native production scope. These digests must match local Ollama manifests.
PRODUCTION_MODEL_POLICY = {
    "granite4:tiny-h": {
        "expected_manifest_digest": "sha256:566b725534ea0e9824f844abe6a78e1ab6f7357f1efb549be94908cb681513bb",
        "expected_model_digest": "sha256:491ba81786c46a345a5da9a60cdb9f9a3056960c8411dd857153c194b1f91313",
        "quant_variant": "manifest-pinned",
    },
    "qwen3.5:9b": {
        "expected_manifest_digest": "sha256:6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7",
        "expected_model_digest": "sha256:dec52a44569a2a25341c4e4d3fee25846eed4f6f0b936278e3a3c900bb99d37c",
        "quant_variant": "manifest-pinned",
    },
    "qwen3.5:4b": {
        "expected_manifest_digest": "sha256:2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd",
        "expected_model_digest": "sha256:81fb60c7daa80fc1123380b98970b320ae233409f0f71a72ed7b9b0d62f40490",
        "quant_variant": "manifest-pinned",
    },
}

SUPPORTED_PRODUCTION_MODELS = tuple(PRODUCTION_MODEL_POLICY.keys())

PRODUCTION_MODEL_ALLOWLIST = {
    "granite4:tiny-h": {
        "digest": "sha256:491ba81786c46a345a5da9a60cdb9f9a3056960c8411dd857153c194b1f91313",
        "template": "granite",
    },
    "qwen3.5:9b": {
        "digest": "sha256:dec52a44569a2a25341c4e4d3fee25846eed4f6f0b936278e3a3c900bb99d37c",
        "template": "chatml",
    },
    "qwen3.5:4b": {
        "digest": "sha256:81fb60c7daa80fc1123380b98970b320ae233409f0f71a72ed7b9b0d62f40490",
        "template": "chatml",
    },
}

# Qwen3.5 sampler presets (aligned with Qwen guidance and local tuning).
QWEN35_SAMPLING_PROFILES = {
    "instruct_deterministic": {
        "temperature": 0.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
    },
    "thinking_general": {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
    },
    "thinking_coding": {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
    },
    "instruct_general": {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
    },
    "instruct_reasoning": {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
    },
    "instruct_reasoning_official": {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 40,
        "min_p": 0.0,
        "presence_penalty": 2.0,
        "repetition_penalty": 1.0,
    },
}

GRANITE4_SAMPLING_PROFILES = {
    "coding_deterministic": {
        "temperature": 0.0,
        "top_p": 0.9,
        "top_k": 40,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
    },
    "coding_balanced": {
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 40,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
    },
    "research_balanced": {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "min_p": 0.0,
        "presence_penalty": 0.5,
        "repetition_penalty": 1.0,
    },
}

# --- DETERMINISTIC PARAMETERS (CRITICAL) ---
# Derived from "Saguaro Enhancement for Deterministic AI"
# "Infinitesimal parameter tuning" to bypass short-circuiting.
# OPTIMIZED: Reduced from 200K to adaptive sizing for CPU performance
GENERATION_PARAMS = {
    "temperature": 0.0,  # Standard deterministic
    "seed": 720720,
    "repeat_penalty": 1.1,  # Increased to prevent loops
    "top_p": 0.9,
    "top_k": 40,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
    "granite4_sampling_profile": "coding_balanced",
    # Bias Qwen toward the more coherent native preset by default.
    "qwen35_sampling_profile": "instruct_deterministic",
    "num_ctx": 400000,  # Expanded benchmark context
    # Native graph context controls (separate from high-level agent `num_ctx`).
    # Keep aligned with agent context so runtime does not silently clamp below 400k.
    "native_ctx_default": 400000,
    "native_ctx_cap": 400000,
    "granite4_native_ctx_default": 400000,
    "granite4_native_ctx_cap": 400000,
    "qwen35_native_ctx_default": 400000,
    "qwen35_native_ctx_cap": 400000,
    "num_predict": 8192,  # Expanded prediction window
    "num_batch": 512,  # Optimized batch size
    "use_mlock": False,  # Disabled for OOM safety on limited VRAM
    "keep_alive": -1,
}

# --- AGENTIC THINKING CONFIGURATION ---
# Enhanced loop and thinking system settings
AGENTIC_THINKING = {
    "thinking_budget": 300000,  # Maximum token budget for deep thinking
    "coconut_enabled": True,  # Enable CoCoNut latent reasoning
    "coconut_use_gpu": False,
    "coconut_verbosity": "concise",  # "concise" or "verbose" - controls detail in reasoning insights
    "enhanced_loop_enabled": True,  # Use enhanced loop for complex tasks
    "auto_verify": True,  # Auto-run verification after execution
    "require_plan_approval": False,  # Require user approval for plans
    "max_verification_attempts": 3,  # Max backtrack cycles
    "show_thinking": True,  # Display thinking blocks
    "compact_task_view": False,  # Compact vs expanded task panel
    "logging_enabled": True,  # Enable unified logging to .anvil/logs/granite.log
    # Multi-agent coherence bus (35% reduction in agent communication overhead)
    # Uses quantum-inspired state transfer for efficient agent handoffs
    "use_coherence_bus": True,  # Enable coherence bus for multi-agent
    "coherence_bus_agents": 6,  # Max agents in entangled mesh
    "coherence_refresh_interval": 50,  # Refresh entanglement every N transfers
}

# --- MULTI-AGENT OWNERSHIP CONFIGURATION ---
OWNERSHIP_CONFIG = {
    # Disabled by default. Enable explicitly for coordinated multi-agent edits.
    "enabled": os.getenv("OWNERSHIP_ENABLED", "0") == "1",
    "lease_ttl_seconds": 300,
    "heartbeat_interval_seconds": 30,
    "auto_release_on_completion": True,
    "default_mode": "exclusive",
    "allow_collaborative_mode": True,
}

# --- CROSS-INSTANCE COLLABORATION CONFIGURATION ---
COLLABORATION_CONFIG = {
    # Disabled by default. Local/offline execution must work without peers.
    "enabled": os.getenv("COLLABORATION_ENABLED", "0") == "1",
    "discovery_method": "auto",  # auto, mdns, filesystem, rendezvous
    "transport_provider": "filesystem",  # in_memory, filesystem
    "rendezvous_url": None,
    "listen_port": 0,  # 0 == auto-assign
    "tls_enabled": True,
    "context_sharing_level": "summary",  # full, summary, metadata_only, none
    "auto_negotiate": True,
    "overlap_threshold": 0.75,
    "user_approval_required": True,
}

# --- CAMPAIGN ORCHESTRATOR CONFIGURATION ---
CAMPAIGN_CONFIG = {
    # Campaign orchestration is designed to run standalone by default.
    "enabled": True,
    "campaigns_dir": ".anvil/campaigns",
    "generated_dir": ".anvil/campaigns/generated",
    "specs_dir": ".anvil/campaigns/specs",
    "custom_dir": ".anvil/campaigns/custom",
    "state_dir": ".anvil/campaigns/state",
    "reports_dir": ".anvil/campaigns/reports",
    "ledger_db_path": ".anvil/campaigns/campaign_ledger.db",
    "max_phases": 20,
    "max_retries_per_phase": 3,
    "default_context_per_phase": 5000000,
    "ledger_max_tokens": 50000,
    "auto_save_state": True,
    "phase_timeout_sec": 1800,
    # Fail fast for deterministic behavior unless explicitly overridden.
    "halt_on_failure": True,
}

# --- ARTIFACT CONFIGURATION ---
ARTIFACT_CONFIG = {
    "base_dir": ".anvil/artifacts",
    "archive_on_complete": True,
    "save_thinking_chains": True,
}

# --- ORCHESTRATION CONFIG ---
ORCHESTRATION_CONFIG = {
    # Options: "parallel", "sequential"
    # Forces the MultiAgentEvidenceGatherer to run subagents sequentially.
    "execution_mode": os.getenv("ANVIL_SUBAGENT_EXECUTION_MODE", "sequential"),
    "max_parallel_agents": int(os.getenv("ANVIL_MAX_PARALLEL_SUBAGENTS", "4")),
}

# --- ENVIRONMENT CONFIGURATION ---
# Controls where virtual environments and Saguaro are installed
ENVIRONMENT_CONFIG = {
    # If True, uses anvil's venv (for self-development)
    # If False, creates venv in target repo (for working on other projects)
    "use_anvil_venv": False,
    # Always install Saguaro from anvil's bundled copy
    "saguaro_source": "bundled",  # "bundled" or "pip"
    # Auto-detect mode: if cwd == anvil root, enable dev mode
    "auto_detect_dev_mode": True,
}

# --- RESPONSE CONFIGURATION ---
RESPONSE_CONFIG = {
    "max_synthesis_tokens": 2000,
    "max_thinking_tokens_for_simple_questions": 500,
    "compression_threshold_chars": 3000,
    "enable_deduplication": True,
}

# --- PERFORMANCE OPTIMIZATIONS ---
PERFORMANCE_CONFIG = {
    # Native QSG runtime
    "native_qsg_engine": True,
    "parallel_decode": True,
    "parallel_width": 4,
    "parallel_prompt_lookup_enabled": True,
    "parallel_jacobi_lookahead_enabled": True,
    "medusa_head_enabled": True,
    "medusa_head_max_draft_tokens": 4,
    "medusa_head_top_k": 8,
    "medusa_head_acceptance_floor": 0.20,
    "hydra_head_enabled": True,
    "hydra_head_max_draft_tokens": 4,
    "hydra_head_top_k": 8,
    "hydra_head_acceptance_floor": 0.22,
    "hydra_head_blend_alpha": 0.55,
    "parallel_ssd_bridge_enabled": True,
    "parallel_replacement_enabled": True,
    "parallel_replacement_max_tree_width": 4,
    "parallel_replacement_acceptance_floor": 0.20,
    "parallel_replacement_max_draft_tokens": 6,
    "parallel_replacement_top_k": 8,
    "parallel_ar_recovery_enabled": True,
    "block_diffusion_enabled": True,
    "block_diffusion_force": False,
    "block_diffusion_min_new_tokens": 96,
    "block_diffusion_min_prompt_tokens": 256,
    "block_diffusion_block_size_tokens": 16,
    "block_diffusion_denoise_iterations": 2,
    "block_diffusion_acceptance_floor": 0.18,
    # Incremental KV cache (reuses attention states with prefix caching)
    "incremental_kv_cache": True,
    # Semantic KV cache (1M+ token contexts via holographic compression)
    # Uses Saguaro fused_coconut_crystallize_op for compression
    # Compresses distant context into 64 semantic crystals, keeps recent 8K uncompressed
    "semantic_kv_cache": True,  # Enable for massive contexts (experimental)
    # PagedAttention KV cache (CPU cache line aligned, reduced fragmentation)
    # Allocates memory in 64-token pages for better cache utilization
    "paged_kv_cache": True,  # Enable for better memory efficiency
    # Hybrid Transformer-SSM (2x speed for 100K+ contexts)
    # Uses Mamba SSM to compress distant context beyond 16K threshold
    "hybrid_ssm": False,  # Disabled in strict native-only mode
    # CPU-optimized speculative decoding (uses COCONUT for drafting)
    # No separate draft model - uses multi-path reasoning as candidates
    "cpu_speculative_decode": False,  # Disabled by default for strict native-only stability
    "spec_num_candidates": 4,  # Number of candidate tokens
    "spec_max_draft_length": 4,  # Max tokens to draft ahead
    "spec_acceptance_threshold": 0.68,  # Baseline; model profiles may override
    # Disable speculative path for the session after repeated full rejections.
    "spec_disable_after_rejections": 2,
    # Strict native-only QSG enforcement:
    # - no NumPy COCONUT bridge fallback
    # - no speculative top-k fallback
    # - no standard-generation fallback ladder
    "strict_native_qsg": os.getenv("ANVIL_STRICT_NATIVE_QSG", "1") == "1",
    "strict_coconut_bridge": os.getenv("ANVIL_STRICT_COCONUT_BRIDGE", "1") == "1",
    "strict_speculative_decode": os.getenv("ANVIL_STRICT_SPECULATIVE_DECODE", "1")
    == "1",
    "strict_logits_processor": os.getenv("ANVIL_STRICT_LOGITS_PROCESSOR", "1") == "1",
    # TimeCrystal context stabilization (native graph drift policy)
    "timecrystal_context_stabilizer": os.getenv("ANVIL_TC_STABILIZER", "1") == "1",
    "timecrystal_mode": os.getenv("ANVIL_TC_MODE", "aggressive_staged"),
    "timecrystal_block_size_tokens": int(os.getenv("ANVIL_TC_BLOCK_SIZE", "128") or "128"),
    "timecrystal_update_interval_tokens": int(
        os.getenv("ANVIL_TC_UPDATE_INTERVAL", "64") or "64"
    ),
    "timecrystal_prune_interval_tokens": 128,
    "timecrystal_preserve_head_tokens": 256,
    "timecrystal_preserve_recent_tokens": int(
        os.getenv("ANVIL_TC_PRESERVE_RECENT", "8192") or "8192"
    ),
    "timecrystal_min_active_tokens": 16384,
    "timecrystal_damp_threshold": float(
        os.getenv("ANVIL_TC_DAMP_THRESHOLD", "0.35") or "0.35"
    ),
    "timecrystal_prune_threshold": float(
        os.getenv("ANVIL_TC_PRUNE_THRESHOLD", "0.72") or "0.72"
    ),
    "timecrystal_damping_strength": 1.2,
    "timecrystal_hysteresis": 0.05,
    "timecrystal_overhead_target_pct": float(
        os.getenv("ANVIL_TC_OVERHEAD_TARGET_PCT", "15") or "15"
    ),
    "timecrystal_overhead_max_pct": float(
        os.getenv("ANVIL_TC_OVERHEAD_MAX_PCT", "20") or "20"
    ),
    "timecrystal_control_interval_tokens": 64,
    "timecrystal_overhead_window_tokens": 128,
    "timecrystal_recovery_tokens": 256,
    # Latent space steering (25% better intent alignment)
    # Uses HD gradient projection for orthogonal steering vectors
    "latent_steering": True,  # Enable for better intent following
    "steering_strength": 0.25,  # Steering strength (0.0-1.0)
    # Adaptive context sizing (8K-200K based on complexity)
    "adaptive_context": True,
    # Tool schema lazy loading (reduce prompt size by 15K tokens)
    "tool_lazy_loading": True,
    # Fast attention kernels (AVX2/AVX512)
    "fast_attention_kernels": True,
    # Speculative decoding (disabled - slower in practice on CPU)
    "speculative_decoding": False,
    "draft_model": "tinyllama:1.1b",
    "num_draft_tokens": 4,
    # Performance monitoring
    "enable_perf_monitoring": True,
}

# --- SAGUARO SYSTEM PATHS ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHRONICLE_DIR = os.path.join(DATA_DIR, "chronicle")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, ARTIFACT_CONFIG["base_dir"])

# Ensure directories exist
os.makedirs(CHRONICLE_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(CAMPAIGN_CONFIG["generated_dir"], exist_ok=True)
os.makedirs(CAMPAIGN_CONFIG["specs_dir"], exist_ok=True)
os.makedirs(CAMPAIGN_CONFIG["custom_dir"], exist_ok=True)
os.makedirs(CAMPAIGN_CONFIG["state_dir"], exist_ok=True)
os.makedirs(CAMPAIGN_CONFIG["reports_dir"], exist_ok=True)
