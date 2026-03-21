"""SAGUARO Native Ops Wrapper
Loads the compiled C++ operations from _saguaro_core.so.

When TensorFlow or native ops are unavailable, critical helper functions fall back
to pure-Python/NumPy implementations so indexing can still complete.
"""

import contextlib
import os
import sys

import numpy as np

try:
    import tensorflow as tf
except Exception:
    tf = None

# Do not clobber caller-provided OpenMP settings. When no policy is supplied,
# default to the visible CPU set instead of force-clamping to one thread.
if "OMP_NUM_THREADS" not in os.environ:
    default_threads = os.getenv("SAGUARO_OMP_NUM_THREADS_DEFAULT", "").strip()
    if not default_threads:
        if hasattr(os, "sched_getaffinity"):
            with contextlib.suppress(Exception):
                default_threads = str(max(1, len(os.sched_getaffinity(0))))
        if not default_threads:
            default_threads = str(max(1, int(os.cpu_count() or 1)))
    os.environ["OMP_NUM_THREADS"] = default_threads
import ctypes
import inspect
import logging

logger = logging.getLogger(__name__)
_DEBUG_NATIVE_OPS = os.getenv("SAGUARO_NATIVE_DEBUG", "0") == "1"


def _debug_log(message: str) -> None:
    """Emit optional native-loader debug logs without polluting stdout."""
    if _DEBUG_NATIVE_OPS:
        print(message, file=sys.stderr)


_module = None
_quantum_embedding_op = None
_quantum_embedding_backward_op = None
_fused_qwt_tokenizer_op = None
_time_crystal_step_op = None
_fused_coconut_bfs_op = None
_fused_text_tokenize_op = None
_fused_text_tokenize_batch_op = None
_superword_trie_create_op = None
_holographic_bundle_op = None
_crystallize_memory_op = None
_modern_hopfield_retrieve_op = None
_streaming_ngram_count_create_op = None
_streaming_ngram_count_op = None
_streaming_ngram_count_export_op = None
_superword_trie_insert_op = None
_superword_trie_build_from_table_op = None
_init_holographic_store_op = None


def load_saguaro_core():
    global _module
    global _quantum_embedding_op
    global _quantum_embedding_backward_op
    global _fused_qwt_tokenizer_op
    global _time_crystal_step_op
    global _fused_coconut_bfs_op
    global _fused_text_tokenize_op
    global _fused_text_tokenize_batch_op
    global _superword_trie_create_op
    global _holographic_bundle_op
    global _crystallize_memory_op
    global _modern_hopfield_retrieve_op
    global _streaming_ngram_count_create_op
    global _streaming_ngram_count_op
    global _streaming_ngram_count_export_op
    global _superword_trie_insert_op
    global _superword_trie_build_from_table_op
    global _init_holographic_store_op

    # Only return early if EVERYTHING is loaded
    if (
        _module is not None
        and _quantum_embedding_op is not None
        and _fused_qwt_tokenizer_op is not None
    ):
        return _module

    if tf is None:
        logger.warning("TensorFlow unavailable; native SAGUARO ops disabled.")
        return None

    try:
        mod = None
        if _module is not None:
            mod = _module
        else:
            # Load logic
            current_dir = os.path.dirname(os.path.abspath(__file__))
            proposal_dir = os.path.dirname(os.path.dirname(current_dir))

            search_paths = [
                os.path.join(proposal_dir, "build", "_saguaro_core.so"),
                os.path.join(proposal_dir, "_saguaro_core.so"),
                os.path.join(current_dir, "_saguaro_core.so"),
                os.path.join(
                    os.path.dirname(current_dir), "_saguaro_core.so"
                ),  # saguaro/ dir
                "_saguaro_core.so",
            ]

            lib_path = None
            for path in search_paths:
                if os.path.exists(path):
                    lib_path = path
                    break

            if not lib_path:
                lib_path = "_saguaro_core.so"

            _debug_log(f"DEBUG: Loading SAGUARO Core from: {lib_path}")

            # CRITICAL FIX: Promote libtensorflow_framework to RTLD_GLOBAL
            # This allows the saguaro_core plugin (built with undefined symbols) to resolve against the host TF.
            try:
                lib_dir = tf.sysconfig.get_lib()
                fw_lib_name = "libtensorflow_framework.so.2"  # Linux default
                # Try to find it roughly
                fw_lib_path = os.path.join(lib_dir, fw_lib_name)
                if not os.path.exists(fw_lib_path):
                    # Try typical variations or heuristic
                    fw_lib_path = os.path.join(lib_dir, "libtensorflow_framework.so")

                if os.path.exists(fw_lib_path):
                    # Check current flags? No, just force it.
                    ctypes.CDLL(fw_lib_path, mode=ctypes.RTLD_GLOBAL)
                    _debug_log(
                        f"DEBUG: Promoted {fw_lib_path} to RTLD_GLOBAL for plugin compatibility."
                    )
            except Exception as e:
                _debug_log(
                    f"DEBUG: Could not promote framework lib to RTLD_GLOBAL: {e}"
                )

            try:
                mod = TEO.load_custom_op((lib_path)
                _debug_log(f"DEBUG: Loaded module attributes: {dir(mod)}")
            except tf.errors.AlreadyExistsError:
                _debug_log("DEBUG: SAGUARO Core already loaded (AlreadyExistsError).")
                # If already loaded, we can't easily get the module object if we didn't store it.
                # But normally we check _module first. If _module is None but library is loaded,
                # it means another import loaded it. We need to find it or re-import.
                # However, standard practice is to just look up ops in the current graph/context if possible,
                # or try to import the module that loaded it.
                # Since we can't get the module object from the error, and we need it for getattr,
                # we might have a problem.
                logger.warning(
                    "SAGUARO Core ops already registered (AlreadyExistsError). Assuming available in tf.raw_ops or previously loaded module."
                )
                mod = None  # Ensure defined
                # We don't have the module object, but ops should be registered.
                # We can't access them via 'mod.OpName'.
                # We must fallback to looking them up via tf.raw_ops or assume they are unavailable via this wrapper
                # and we have to rely on the fact they are in the graph?
                # But our wrappers below use `getattr(mod, ...)`
                # If mod is None, we need an alternative.
                pass

        if mod:
            # Bind locally first
            q_emb = getattr(mod, "QuantumEmbeddingForward", None)
            if q_emb is None:
                q_emb = getattr(mod, "quantum_embedding_forward", None)
            if q_emb is None:
                q_emb = getattr(mod, "QuantumEmbedding", None)

            qwt = getattr(mod, "fused_qwt_tokenizer", None)
            if qwt is None:
                qwt = getattr(mod, "FusedQwtTokenizer", None)

            tc_step = getattr(mod, "time_crystal_step", None)
            if tc_step is None:
                tc_step = getattr(mod, "TimeCrystalStep", None)

            coconut = getattr(mod, "fused_coconut_bfs", None)
            if coconut is None:
                coconut = getattr(mod, "FusedCoconutBfs", None)

            tok = getattr(mod, "fused_text_tokenize", None)
            if tok is None:
                tok = getattr(mod, "FusedTextTokenize", None)
            if tok is None:
                tok = getattr(mod, "SAGUAROTextTokenize", None)
            if tok is None:
                tok = getattr(mod, "saguaro_text_tokenize", None)

            tok_batch = getattr(mod, "SAGUAROTextTokenizeBatch", None)
            if tok_batch is None:
                tok_batch = getattr(mod, "saguaro_text_tokenize_batch", None)
            if tok_batch is None:
                tok_batch = getattr(mod, "FusedTextTokenizeBatch", None)
            if tok_batch is None:
                tok_batch = getattr(mod, "fused_text_tokenize_batch", None)

            trie = getattr(mod, "superword_trie_create", None)
            if trie is None:
                trie = getattr(mod, "SuperwordTrieCreate", None)
            if trie is None:
                trie = getattr(mod, "SAGUAROTrieCreate", None)
            if trie is None:
                trie = getattr(mod, "saguaro_trie_create", None)

            # New Ops
            snc_create = getattr(mod, "StreamingNgramCountCreate", None)
            snc_run = getattr(mod, "StreamingNgramCount", None)
            snc_export = getattr(mod, "StreamingNgramCountExport", None)
            trie_insert = getattr(mod, "SuperwordTrieInsert", None)
            trie_build = getattr(mod, "SuperwordTrieBuildFromTable", None)

            _holographic_bundle_op = getattr(mod, "SAGUAROHolographicBundle", None)
            if _holographic_bundle_op is None:
                _holographic_bundle_op = getattr(
                    mod, "saguaro_holographic_bundle", None
                )
            if _holographic_bundle_op is None:
                _holographic_bundle_op = getattr(mod, "HolographicBundle", None)
            if _holographic_bundle_op is None:
                _holographic_bundle_op = getattr(mod, "holographic_bundle", None)

            _crystallize_memory_op = getattr(mod, "crystallize_memory", None)
            if _crystallize_memory_op is None:
                _crystallize_memory_op = getattr(mod, "CrystallizeMemory", None)

            _modern_hopfield_retrieve_op = getattr(
                mod, "modern_hopfield_retrieve", None
            )
            if _modern_hopfield_retrieve_op is None:
                _modern_hopfield_retrieve_op = getattr(
                    mod, "ModernHopfieldRetrieve", None
                )

            _module = mod

        else:
            # Mod is None (AlreadyExistsError). Fallback to tf.raw_ops
            # tf.raw_ops contains generated wrappers for registered ops.
            # They usually use the CamelCase registered name.
            logger.info("Falling back to tf.raw_ops lookup.")

            q_emb = getattr(tf.raw_ops, "QuantumEmbeddingForward", None)
            if q_emb is None:
                q_emb = getattr(tf.raw_ops, "QuantumEmbedding", None)

            q_emb_grad = getattr(tf.raw_ops, "QuantumEmbeddingBackward", None)
            if q_emb_grad:
                _quantum_embedding_backward_op = q_emb_grad

            qwt = getattr(tf.raw_ops, "FusedQwtTokenizer", None)
            tc_step = getattr(tf.raw_ops, "TimeCrystalStep", None)
            coconut = getattr(tf.raw_ops, "FusedCoconutBfs", None)

            tok = getattr(tf.raw_ops, "FusedTextTokenize", None)
            if tok is None:
                tok = getattr(tf.raw_ops, "SAGUAROTextTokenize", None)

            tok_batch = getattr(tf.raw_ops, "FusedTextTokenizeBatch", None)
            if tok_batch is None:
                tok_batch = getattr(tf.raw_ops, "SAGUAROTextTokenizeBatch", None)

            trie = getattr(tf.raw_ops, "SuperwordTrieCreate", None)
            if trie is None:
                trie = getattr(tf.raw_ops, "SAGUAROTrieCreate", None)

            snc_create = getattr(tf.raw_ops, "StreamingNgramCountCreate", None)
            snc_run = getattr(tf.raw_ops, "StreamingNgramCount", None)
            snc_export = getattr(tf.raw_ops, "StreamingNgramCountExport", None)
            trie_insert = getattr(tf.raw_ops, "SuperwordTrieInsert", None)
            trie_build = getattr(tf.raw_ops, "SuperwordTrieBuildFromTable", None)
            init_store = getattr(tf.raw_ops, "InitHolographicStore", None)
            if init_store:
                _init_holographic_store_op = init_store

            _holographic_bundle_op = getattr(tf.raw_ops, "HolographicBundle", None)
            if _holographic_bundle_op is None:
                _holographic_bundle_op = getattr(
                    tf.raw_ops, "SAGUAROHolographicBundle", None
                )
            _crystallize_memory_op = getattr(tf.raw_ops, "CrystallizeMemory", None)
            _modern_hopfield_retrieve_op = getattr(
                tf.raw_ops, "ModernHopfieldRetrieve", None
            )

            if _holographic_bundle_op:
                # Ensure we set a flag that we are using raw ops?
                # Raw ops argument handling might be stricter?
                pass
            else:
                logger.warning("Could not find HolographicBundle even in tf.raw_ops")

        # Atomically(ish) update globals - globals already declared at function start
        if True:  # always try to update if we found something
            if q_emb:
                _quantum_embedding_op = q_emb

            # Load Backward Op
            q_emb_grad = getattr(mod, "QuantumEmbeddingBackward", None)
            if q_emb_grad is None:
                q_emb_grad = getattr(mod, "quantum_embedding_backward", None)
            if q_emb_grad:
                _quantum_embedding_backward_op = q_emb_grad

            if qwt:
                _fused_qwt_tokenizer_op = qwt
            if tc_step:
                _time_crystal_step_op = tc_step
            if coconut:
                _fused_coconut_bfs_op = coconut
            if tok:
                _fused_text_tokenize_op = tok
            if tok_batch:
                _fused_text_tokenize_batch_op = tok_batch
            if trie:
                _superword_trie_create_op = trie

            if snc_create:
                _streaming_ngram_count_create_op = snc_create
            if snc_run:
                _streaming_ngram_count_op = snc_run
            if snc_export:
                _streaming_ngram_count_export_op = snc_export
            if trie_insert:
                _superword_trie_insert_op = trie_insert
            if trie_build:
                _superword_trie_build_from_table_op = trie_build

            init_store = getattr(mod, "init_holographic_store", None)
            if init_store is None:
                init_store = getattr(mod, "InitHolographicStore", None)
            if init_store:
                _init_holographic_store_op = init_store

            # These are assigned earlier in the function but need to persist
            # The earlier assignments already updated these globals

            # Print status of key ops
            _debug_log(
                f"DEBUG: _holographic_bundle_op bound to: {_holographic_bundle_op}"
            )
            try:
                _debug_log(
                    f"DEBUG: _holographic_bundle_op signature: {inspect.signature(_holographic_bundle_op)}"
                )
            except Exception:
                _debug_log("DEBUG: Could not inspect signature")
            _debug_log(
                f"DEBUG: _fused_text_tokenize_batch_op bound to: {_fused_text_tokenize_batch_op}"
            )
            with contextlib.suppress(Exception):
                _debug_log(
                    f"DEBUG: _modern_hopfield_retrieve_op signature: {inspect.signature(_modern_hopfield_retrieve_op)}"
                )

        _debug_log("DEBUG: SAGUARO Core loaded (or re-verified)")
        logger.info("SAGUARO Core loaded (or re-verified)")

    except Exception as e:
        logger.error(f"Failed to load SAGUARO Core: {e}")
        # Don't raise, let it proceed with None ops so fallback can work?
        # raise e
        pass

    return _module


# ... (rest of file)

# Cached empty resource handle
_empty_resource_handle = None


def _get_empty_resource_handle():
    global _empty_resource_handle
    if _empty_resource_handle is None:
        if _superword_trie_create_op is not None:
            _empty_resource_handle = _superword_trie_create_op(
                container="", shared_name="saguaro_empty_trie_default"
            )
        else:
            # Fallback
            pass

    return _empty_resource_handle


def init_holographic_store(
    embeddings=None, token_keys=None, container="", shared_name="", name=None
):
    """Initializes the holographic associative memory store."""
    load_saguaro_core()

    # Sensible defaults for bootstrapping if not provided
    # Standard Granite embedding dimension is typically 1024 or 2048, but tiny-h is 384 or something similar.
    # We use tf.zeros if seeding is not explicitly required.
    if embeddings is None:
        embeddings = tf.zeros([1, 64], dtype=tf.float32)
    if token_keys is None:
        token_keys = tf.zeros([1, 64], dtype=tf.float32)

    return _init_holographic_store_op(
        embeddings=embeddings, token_keys=token_keys, name=name
    )


def quantum_embedding(
    input_tensor,
    token_keys,
    holographic_store=None,
    vocab_size=50257,
    hd_dim=8192,
    num_bundles=None,
    name=None,
):
    """Quantum-enhanced embedding lookup via holographic unbinding.

    Args:
        input_tensor: Token IDs [batch, seq_len] or [seq_len]
        token_keys: Token-specific orthogonal keys [vocab_size, dim]
        holographic_store: Bundled representations [num_bundles, dim]
        vocab_size: Vocabulary size
        hd_dim: Embedding dimension
        num_bundles: Number of holographic bundles (inferred from store if None)
        name: Op name

    Returns:
        Embeddings [batch, seq_len, dim] or [seq_len, dim]
    """
    load_saguaro_core()

    if holographic_store is None:
        holographic_store = _get_empty_resource_handle()
        if num_bundles is None:
            num_bundles = 4  # Default
    else:
        # Infer num_bundles from store shape
        if num_bundles is None:
            store_shape = holographic_store.shape
            if store_shape.rank is not None and store_shape.rank > 0:
                num_bundles = store_shape[0]
                if num_bundles is None:
                    num_bundles = tf.shape(holographic_store)[0]
            else:
                num_bundles = tf.shape(holographic_store)[0]

    # Ensure num_bundles is an int for the C++ op attribute
    if isinstance(num_bundles, tf.Tensor):
        num_bundles = 4  # Fallback if dynamic
    else:
        num_bundles = int(num_bundles)

    # Dynamic Op Resolution to prevent tf.function capture issues
    op_func = _quantum_embedding_op
    if op_func is None:
        mod = load_saguaro_core()
        op_func = getattr(
            mod,
            "QuantumEmbeddingForward",
            getattr(mod, "quantum_embedding_forward", None),
        )
        if op_func is None:
            # Last ditch attempt via tf.raw_ops
            op_func = getattr(
                tf.raw_ops,
                "QuantumEmbeddingForward",
                getattr(tf.raw_ops, "QuantumEmbedding", None),
            )

    if op_func is None:
        raise RuntimeError("Could not resolve QuantumEmbeddingForward op")

    return op_func(
        token_ids=input_tensor,
        token_keys=token_keys,
        holographic_store=holographic_store,
        vocab_size=vocab_size,
        dim=hd_dim,
        num_bundles=num_bundles,
        name=name,
    )


def _quantum_embedding_grad(op, grad):
    """Gradient for QuantumEmbeddingForward.

    Passes holographic_store to backward op for dynamic shape inference.
    This enables training with arbitrary store sizes instead of hardcoded num_bundles=4.
    """
    load_saguaro_core()

    # Forward Op Inputs:
    # 0: token_ids
    # 1: holographic_store
    # 2: token_keys
    token_ids = op.inputs[0]
    holographic_store = op.inputs[1]
    token_keys = op.inputs[2]

    vocab_size = op.get_attr("vocab_size")
    dim = op.get_attr("dim")
    # num_bundles is now inferred from holographic_store shape, not an attribute

    # Backward Op now takes holographic_store for dynamic shape inference
    grad_store = _quantum_embedding_backward_op(
        grad_output=grad,
        token_ids=token_ids,
        token_keys=token_keys,
        holographic_store=holographic_store,  # Pass store for dynamic shape inference
        vocab_size=vocab_size,
        dim=dim,
    )

    # Return gradients for Forward Inputs: [ids, store, keys]
    # We only have gradient for store.
    return [None, grad_store, None]


with contextlib.suppress(Exception):
    tf.RegisterGradient("QuantumEmbeddingForward")(_quantum_embedding_grad)


def fused_qwt_tokenizer(
    input_text,
    levels=3,
    low_pass_filter=None,
    high_pass_filter=None,
    mask=None,
    evolution_time=1.0,
    ctqw_steps=10,
    name=None,
):
    load_saguaro_core()
    # Fix: Provide defaults if None, and pass all required args to C++ op
    if low_pass_filter is None:
        # Dummy filter for gradients (assuming [2] for Haar wavelet)
        low_pass_filter = tf.constant([0.707, 0.707], dtype=tf.float32)
    if high_pass_filter is None:
        high_pass_filter = tf.constant([0.707, -0.707], dtype=tf.float32)
    if mask is None:
        # Dummy mask [1, 128]
        mask = tf.ones([1, 128], dtype=tf.float32)

    return _fused_qwt_tokenizer_op(
        input_text,
        num_wavelet_levels=levels,
        low_pass_filter=low_pass_filter,
        high_pass_filter=high_pass_filter,
        mask=mask,
        evolution_time=evolution_time,
        ctqw_steps=ctqw_steps,
        name=name,
    )


def time_crystal_step(state, feedback, name=None):
    """Experimental: Time Crystal Dynamics step.
    .. warning:: This op is experimental and may change.
    """
    load_saguaro_core()
    return _time_crystal_step_op(state, feedback, name=name)


def fused_coconut_bfs(adjacency_matrix, start_node, name=None):
    """Experimental: Coherent Coconut BFS.
    .. warning:: This op is experimental and may change.
    """
    load_saguaro_core()
    return _fused_coconut_bfs_op(adjacency_matrix, start_node, name=name)


def fused_text_tokenize(
    input_text,
    trie_handle=None,
    max_length=512,
    byte_offset=32,
    add_special_tokens=True,
    name=None,
):
    load_saguaro_core()
    if _fused_text_tokenize_op is None:
        raise RuntimeError("FusedTextTokenize op not available in this build")

    if trie_handle is None:
        trie_handle = _get_empty_resource_handle()

    return _fused_text_tokenize_op(
        input_text=input_text,
        trie_handle=trie_handle,
        byte_offset=byte_offset,
        add_special_tokens=add_special_tokens,
        max_length=max_length,
        name=name,
    )


def fused_text_tokenize_batch(
    input_texts,
    trie_handle=None,
    max_length=512,
    byte_offset=32,
    add_special_tokens=True,
    num_threads=0,
    name=None,
):
    load_saguaro_core()
    # If op is missing or fails, use fallback
    use_fallback = False
    if _fused_text_tokenize_batch_op is None:
        use_fallback = True
    else:
        try:
            if trie_handle is None:
                trie_handle = _get_empty_resource_handle()

            return _fused_text_tokenize_batch_op(
                input_texts=input_texts,
                trie_handle=trie_handle,
                byte_offset=byte_offset,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                num_threads=num_threads,
                name=name,
            )
        except Exception as e:
            logger.warning(f"Native tokenization failed ({e}), using NumPy fallback.")
            use_fallback = True

    if use_fallback:
        # Pure-Python deterministic tokenizer fallback.
        if tf is None:
            texts = [str(t) for t in input_texts]
        else:
            try:
                texts = [
                    t.decode("utf-8", errors="ignore") for t in input_texts.numpy()
                ]
            except Exception:
                texts = [str(t) for t in input_texts]

        token_rows = []
        lengths = []

        for text in texts:
            # Simple byte-level tokenization.
            raw = text.encode("utf-8", errors="ignore")
            if not raw:
                ids = [1]
            else:
                ids = [int(b) + int(byte_offset) for b in raw[:max_length]]

            if add_special_tokens:
                ids = [2] + ids + [3]
                ids = ids[:max_length]

            lengths.append(len(ids))
            if len(ids) < max_length:
                ids = ids + [0] * (max_length - len(ids))
            token_rows.append(ids[:max_length])

        token_np = np.asarray(token_rows, dtype=np.int32)
        lengths_np = np.asarray(lengths, dtype=np.int32)

        if tf is None:
            return token_np, lengths_np
        return tf.constant(token_np), tf.constant(lengths_np)
    return None


def holographic_bundle(vectors, name=None):
    load_saguaro_core()
    if _holographic_bundle_op is not None:
        return _holographic_bundle_op(vectors=vectors, name=name)

    # Fallback: simple mean bundle.
    if tf is not None:
        t = tf.convert_to_tensor(vectors, dtype=tf.float32)
        return tf.reduce_mean(t, axis=0)
    arr = np.asarray(vectors, dtype=np.float32)
    return np.mean(arr, axis=0, dtype=np.float32)


def crystallize_memory(knowledge, importance, threshold=0.5, name=None):
    load_saguaro_core()
    if _crystallize_memory_op is not None:
        return _crystallize_memory_op(
            knowledge=knowledge, importance=importance, threshold=threshold, name=name
        )

    # Fallback: thresholded masking.
    if tf is not None:
        k = tf.convert_to_tensor(knowledge, dtype=tf.float32)
        i = tf.convert_to_tensor(importance, dtype=tf.float32)
        mask = tf.cast(i >= threshold, tf.float32)
        return k * mask
    k = np.asarray(knowledge, dtype=np.float32)
    i = np.asarray(importance, dtype=np.float32)
    return np.where(i >= float(threshold), k, 0.0).astype(np.float32)


def modern_hopfield_retrieve(query, memory, beta=1.0, name=None):
    load_saguaro_core()
    if _modern_hopfield_retrieve_op is not None:
        return _modern_hopfield_retrieve_op(query, memory, beta=beta, name=name)

    # Fallback: attention-style retrieval.
    if tf is not None:
        q = tf.convert_to_tensor(query, dtype=tf.float32)
        m = tf.convert_to_tensor(memory, dtype=tf.float32)
        logits = beta * tf.matmul(q, m, transpose_b=True)
        w = tf.nn.softmax(logits, axis=-1)
        return tf.matmul(w, m)

    q = np.asarray(query, dtype=np.float32)
    m = np.asarray(memory, dtype=np.float32)
    logits = float(beta) * (q @ m.T)
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(logits)
    weights = exp / (np.sum(exp, axis=-1, keepdims=True) + 1e-9)
    return (weights @ m).astype(np.float32)


def streaming_ngram_count_create(
    min_ngram=2, max_ngram=5, container="", shared_name="", name=None
):
    load_saguaro_core()
    return _streaming_ngram_count_create_op(
        min_ngram=min_ngram,
        max_ngram=max_ngram,
        container=container,
        shared_name=shared_name,
        name=name,
    )


def streaming_ngram_count(handle, tokens, lengths, name=None):
    load_saguaro_core()
    return _streaming_ngram_count_op(handle, tokens, lengths, name=name)


def streaming_ngram_count_export(handle, min_frequency=5, max_count=10000, name=None):
    load_saguaro_core()
    return _streaming_ngram_count_export_op(
        handle, min_frequency=min_frequency, max_count=max_count, name=name
    )


def superword_trie_insert(handle, ngram, superword_id, name=None):
    load_saguaro_core()
    return _superword_trie_insert_op(handle, ngram, superword_id, name=name)


def superword_trie_build_from_table(
    handle, ngram_offsets, ngram_tokens, superword_ids, name=None
):
    load_saguaro_core()
    return _superword_trie_build_from_table_op(
        handle=handle,
        ngram_offsets=ngram_offsets,
        ngram_tokens=ngram_tokens,
        superword_ids=superword_ids,
        name=name,
    )
