from __future__ import annotations

from typing import Any

NATIVE_SHARED_OBJECT = "libanvil_native_ops.so"

FFI_BOUNDARY_MANIFEST: dict[str, dict[str, Any]] = {
    "core/native/fast_attention_wrapper.py": {
        "shared_object": NATIVE_SHARED_OBJECT,
        "symbols": ["fused_attention_f32", "fused_attention_mqa_f32"],
    },
    "core/native/model_graph_wrapper.py": {
        "shared_object": NATIVE_SHARED_OBJECT,
        "symbols": [
            "create_model_graph_v2",
            "graph_forward_token",
            "graph_forward_token_id",
            "graph_forward_token_ids",
            "graph_set_drift_config",
            "graph_get_last_drift_snapshot",
            "graph_get_drift_config",
            "graph_reset",
            "graph_reset_perf_stats",
            "graph_get_position",
            "graph_get_perf_stats",
            "graph_copy_last_hidden",
            "graph_create_execution_checkpoint",
            "graph_restore_execution_checkpoint",
            "graph_destroy_execution_checkpoint",
            "graph_forward_token_id_to_exit",
            "graph_continue_from_hidden",
            "destroy_model_graph",
            "graph_forward_layer",
            "graph_forward_head",
            "graph_set_layer_weights",
            "graph_set_embedding_weights",
            "graph_set_embedding_weights_quantized",
            "graph_set_head_weights",
            "graph_set_head_weights_quantized",
            "graph_set_layer_weights_quantized",
            "graph_set_layer_extras",
            "graph_set_qwen_mrope_config",
            "graph_set_qwen_hybrid_config",
        ],
    },
    "core/native/native_kv_cache_wrapper.py": {
        "shared_object": NATIVE_SHARED_OBJECT,
        "symbols": [
            "kv_cache_append_batch",
            "kv_cache_flash_attention",
            "kv_cache_snapshot_prefix",
            "kv_cache_restore_prefix",
            "kv_cache_release_snapshot",
            "kv_cache_get_metrics",
        ],
    },
    "core/native/native_ops.py": {
        "shared_object": NATIVE_SHARED_OBJECT,
        "symbols": [
            "anvil_native_build_id",
            "anvil_native_compat_alias_csv",
            "anvil_native_optional_isa_leaves",
            "anvil_native_public_load_target",
            "anvil_native_split_layout",
            "anvil_native_runtime_core_target",
            "anvil_native_split_abi_version",
            "anvil_native_isa_baseline",
            "anvil_compiled_with_amx",
            "anvil_runtime_amx_available",
        ],
    },
    "core/native/native_tokenizer.py": {
        "shared_object": NATIVE_SHARED_OBJECT,
        "symbols": [
            "anvil_native_tokenizer_create",
            "anvil_native_tokenizer_destroy",
            "anvil_native_tokenizer_encode",
            "anvil_native_tokenizer_decode",
            "anvil_native_tokenizer_get_suppressed_token_count",
            "anvil_native_tokenizer_get_suppressed_tokens",
        ],
    },
    "core/native/qsg_parallel_kernels_wrapper.py": {
        "shared_object": NATIVE_SHARED_OBJECT,
        "symbols": [
            "qsg_autoregressive_generate",
            "qsg_masked_diffusion_draft",
            "qsg_block_diffusion_draft",
            "qsg_eagle_replacement_draft",
            "qsg_hydra_head_draft",
        ],
    },
    "core/native/qsg_state_kernels_wrapper.py": {
        "shared_object": NATIVE_SHARED_OBJECT,
        "symbols": [
            "qsg_state_gather_rows",
            "qsg_state_scatter_rows",
            "qsg_state_clone_cow",
            "qsg_state_compact",
            "qsg_state_weighted_merge",
            "qsg_latent_encode_f16",
            "qsg_latent_decode_f16",
        ],
    },
    "core/native/quantized_matmul_wrapper.py": {
        "shared_object": NATIVE_SHARED_OBJECT,
        "symbols": [
            "quantized_matvec",
            "quantized_matmul",
            "quantized_matvec_q8_0",
            "quantized_matmul_q8_0",
            "quantized_matvec_q4_k",
            "quantized_matmul_q4_k",
            "quantized_matvec_q6_k",
            "quantized_matmul_q6_k",
        ],
    },
    "core/native/simd_ops_wrapper.py": {
        "shared_object": NATIVE_SHARED_OBJECT,
        "symbols": [
            "simd_matmul_f32",
            "simd_matvec_f32",
            "simd_rmsnorm_f32",
            "simd_swiglu_f32",
            "simd_softmax_f32",
            "simd_sanitize_logits_f32",
            "simd_argmax_f32",
            "simd_score_token_f32",
        ],
    },
}


def resolve_ffi_boundary(file_path: str) -> dict[str, Any]:
    return dict(FFI_BOUNDARY_MANIFEST.get(str(file_path).strip(), {}))
