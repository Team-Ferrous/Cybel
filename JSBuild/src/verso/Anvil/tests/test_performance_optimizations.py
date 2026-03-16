"""
Comprehensive test suite for Anvil performance optimizations.

Tests:
1. Incremental KV cache
2. Adaptive context manager
3. Performance monitoring
4. Fast attention kernels
5. End-to-end speedup measurements
"""

import pytest
import numpy as np
import time
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestIncrementalKVCache:
    """Test incremental KV cache functionality."""

    def test_import(self):
        """Test that incremental KV cache can be imported."""
        from core.native.incremental_kv_cache import IncrementalKVCache

        assert IncrementalKVCache is not None

    def test_basic_operations(self):
        """Test basic KV cache operations."""
        from core.native.incremental_kv_cache import IncrementalKVCache

        # Mock llm context (would be llama_cpp context in reality)
        class MockContext:
            pass

        class MockLLM:
            def __init__(self):
                self.reset_calls = 0

            def reset(self):
                self.reset_calls += 1

        mock_ctx = MockContext()
        mock_llm = MockLLM()
        cache = IncrementalKVCache(mock_ctx, max_seq_len=1024, llm_obj=mock_llm)

        # Test initialization
        assert cache.current_pos == 0
        assert cache.valid_until == 0
        assert cache.max_seq_len == 1024

        # Test position tracking
        cache.advance_position(10)
        assert cache.current_pos == 10
        assert cache.valid_until == 10

        # Test reset
        cache.reset()
        assert mock_llm.reset_calls == 1
        assert cache.current_pos == 0
        assert cache.valid_until == 0


class TestAdaptiveContext:
    """Test adaptive context manager."""

    def test_import(self):
        """Test that adaptive context can be imported."""
        from core.adaptive_context import AdaptiveContextManager

        assert AdaptiveContextManager is not None

    def test_complexity_analysis(self):
        """Test complexity tier detection."""
        from core.adaptive_context import AdaptiveContextManager

        manager = AdaptiveContextManager()

        # Simple query
        tier = manager.analyze_complexity("what is the codebase structure?")
        assert tier == "simple"

        # Medium query
        tier = manager.analyze_complexity("write a function to parse JSON")
        assert tier == "medium"

        # Complex query
        tier = manager.analyze_complexity("implement a new authentication system")
        assert tier == "complex"

        # Extreme query
        tier = manager.analyze_complexity("migrate the entire codebase to TypeScript")
        assert tier == "extreme"

    def test_generation_params(self):
        """Test that generation params are correctly sized."""
        from core.adaptive_context import AdaptiveContextManager

        manager = AdaptiveContextManager()

        # Simple tier
        params = manager.get_generation_params("simple")
        assert params["num_ctx"] == 8192
        assert params["num_predict"] == 4096

        # Medium tier
        params = manager.get_generation_params("medium")
        assert params["num_ctx"] == 32768
        assert params["num_predict"] == 16384

        # Complex tier
        params = manager.get_generation_params("complex")
        assert params["num_ctx"] == 131072
        assert params["num_predict"] == 65536

    def test_context_estimation(self):
        """Test token count estimation."""
        from core.adaptive_context import AdaptiveContextManager

        manager = AdaptiveContextManager()

        system_prompt = "You are a helpful assistant." * 100
        user_input = "Help me write code"

        tokens = manager.estimate_context_size(system_prompt, user_input)
        assert tokens > 0
        assert tokens < 10000  # Reasonable bounds

    def test_compression(self):
        """Test context compression."""
        from core.adaptive_context import ContextCompressor

        compressor = ContextCompressor()

        context_parts = {
            "system": "System prompt",
            "tools": '{"name": "tool1", "schema": ' + '{"x": "y"}' * 1000 + "}",
            "files": "\n".join([f"file{i}.py" for i in range(100)]),
        }

        compressed = compressor.compress_to_fit(context_parts, max_tokens=500)

        # Check that compression happened
        assert len(compressed["tools"]) < len(context_parts["tools"])
        assert "Available tools:" in compressed["tools"]


class TestPerformanceMonitoring:
    """Test performance monitoring system."""

    def test_import(self):
        """Test that performance monitor can be imported."""
        from core.performance_monitor import PerformanceMonitor

        assert PerformanceMonitor is not None

    def test_logging(self):
        """Test metric logging."""
        from core.performance_monitor import PerformanceMonitor

        monitor = PerformanceMonitor(persist_path=None)

        # Log some metrics
        monitor.log(tokens=100, elapsed=2.0, ctx_size=8192, tier="simple")
        monitor.log(tokens=200, elapsed=5.0, ctx_size=32768, tier="medium")

        stats = monitor.get_stats()

        assert stats["total_requests"] == 2
        assert stats["total_tokens"] == 300
        assert abs(stats["total_time"] - 7.0) < 0.01
        assert stats["avg_throughput"] > 0

    def test_tier_stats(self):
        """Test per-tier statistics."""
        from core.performance_monitor import PerformanceMonitor

        monitor = PerformanceMonitor(persist_path=None)

        monitor.log(100, 2.0, 8192, tier="simple")
        monitor.log(150, 3.0, 8192, tier="simple")
        monitor.log(200, 5.0, 32768, tier="medium")

        tier_stats = monitor.get_stats_by_tier()

        assert "simple" in tier_stats
        assert "medium" in tier_stats
        assert tier_stats["simple"]["count"] == 2
        assert tier_stats["medium"]["count"] == 1

    def test_report_generation(self):
        """Test report generation."""
        from core.performance_monitor import PerformanceMonitor

        monitor = PerformanceMonitor(persist_path=None)
        monitor.log(100, 2.0, 8192, tier="simple")

        report = monitor.get_report()

        assert "PERFORMANCE REPORT" in report
        assert "Total Requests" in report
        assert "Avg Throughput" in report


class TestFastAttention:
    """Test fast attention kernels (if available)."""

    def test_import(self):
        """Test that fast attention can be imported."""
        try:
            from core.native.fast_attention_wrapper import FastAttention

            assert FastAttention is not None
        except ImportError:
            pytest.skip("Fast attention library not built")

    def test_availability(self):
        """Test availability check."""
        try:
            from core.native.fast_attention_wrapper import FastAttention

            attn = FastAttention()
            # Will be False until library is built
            assert isinstance(attn.available, bool)
        except ImportError:
            pytest.skip("Fast attention library not built")

    @pytest.mark.skipif(
        not os.path.exists(
            "/home/mike/Documents/granite-agent/core/native/libfast_attention.so"
        ),
        reason="Fast attention library not built",
    )
    def test_attention_computation(self):
        """Test attention computation (requires built library)."""
        from core.native.fast_attention_wrapper import FastAttention

        attn = FastAttention()

        if not attn.available:
            pytest.skip("Fast attention library not loaded")

        # Test with small matrices
        batch_heads = 2
        seq_len = 16
        head_dim = 64

        Q = np.random.randn(batch_heads, seq_len, head_dim).astype(np.float32)
        K = np.random.randn(batch_heads, seq_len, head_dim).astype(np.float32)
        V = np.random.randn(batch_heads, seq_len, head_dim).astype(np.float32)

        # Compute attention
        out = attn.compute_attention(Q, K, V)

        # Check output shape
        assert out.shape == (batch_heads, seq_len, head_dim)

        # Check that output is not all zeros
        assert np.abs(out).sum() > 0

    @pytest.mark.skipif(
        not os.path.exists(
            "/home/mike/Documents/granite-agent/core/native/libfast_attention.so"
        ),
        reason="Fast attention library not built",
    )
    def test_attention_performance(self):
        """Benchmark attention computation."""
        from core.native.fast_attention_wrapper import FastAttention

        attn = FastAttention()

        if not attn.available:
            pytest.skip("Fast attention library not loaded")

        batch_heads = 8
        seq_len = 128
        head_dim = 64

        Q = np.random.randn(batch_heads, seq_len, head_dim).astype(np.float32)
        K = np.random.randn(batch_heads, seq_len, head_dim).astype(np.float32)
        V = np.random.randn(batch_heads, seq_len, head_dim).astype(np.float32)

        # Warmup
        for _ in range(5):
            attn.compute_attention(Q, K, V)

        # Benchmark
        start = time.time()
        iterations = 100
        for _ in range(iterations):
            attn.compute_attention(Q, K, V)
        elapsed = time.time() - start

        avg_time = elapsed / iterations
        print(f"\nAverage attention time: {avg_time * 1000:.2f} ms")

        # Should be reasonably fast (< 10ms for this size)
        assert avg_time < 0.01  # 10ms


class TestHopfieldOptimization:
    """Test optimized Hopfield network."""

    def test_import(self):
        """Test that Hopfield vocab can be imported."""
        from core.qsg.hopfield_vocab import HopfieldVocab, create_hopfield_vocab

        assert HopfieldVocab is not None
        assert create_hopfield_vocab is not None

    def test_factory_function(self):
        """Test factory function creates correct implementation."""
        from core.qsg.hopfield_vocab import create_hopfield_vocab

        vocab_size = 1000
        dim = 64
        vocab_embeddings = np.random.randn(vocab_size, dim).astype(np.float32)

        hopfield = create_hopfield_vocab(vocab_embeddings, beta=1.0)

        # Should return some implementation
        assert hopfield is not None
        assert hasattr(hopfield, "get_token_probs")

    def test_token_probs(self):
        """Test token probability computation."""
        from core.qsg.hopfield_vocab import HopfieldVocab

        vocab_size = 100
        dim = 32
        vocab_embeddings = np.random.randn(vocab_size, dim).astype(np.float32)

        hopfield = HopfieldVocab(vocab_embeddings, beta=1.0)

        query = np.random.randn(1, dim).astype(np.float32)
        probs = hopfield.get_token_probs(query)

        # Check shape
        assert probs.shape == (1, vocab_size)

        # Check that it's a valid probability distribution
        assert np.all(probs >= 0)
        assert np.abs(np.sum(probs, axis=-1) - 1.0) < 1e-5


class TestEndToEndPerformance:
    """End-to-end performance tests."""

    @pytest.mark.skipif(
        os.getenv("SKIP_SLOW_TESTS") == "1",
        reason="Slow test - set SKIP_SLOW_TESTS=0 to run",
    )
    def test_agent_throughput(self):
        """Test actual agent throughput (requires models)."""
        try:
            from core.agent import BaseAgent

            # Create agent with optimizations
            agent = BaseAgent(output_format="json")

            if agent.perf_monitor is None:
                pytest.skip("Performance monitoring not enabled")

            # Simple query
            start = time.time()
            result = agent.simple_chat("What is 2+2?")
            elapsed = time.time() - start

            # Check that we got a response
            assert len(result) > 0

            # Check performance was logged
            stats = agent.perf_monitor.get_stats()
            assert stats["total_requests"] >= 1

            print(f"\nAgent response time: {elapsed:.2f}s")
            print(f"Throughput: {stats['avg_throughput']:.2f} tokens/sec")

        except ImportError as e:
            pytest.skip(f"Agent dependencies not available: {e}")


class TestConfigIntegration:
    """Test configuration integration."""

    def test_performance_config_exists(self):
        """Test that performance config is defined."""
        from config.settings import PERFORMANCE_CONFIG

        assert isinstance(PERFORMANCE_CONFIG, dict)
        assert "incremental_kv_cache" in PERFORMANCE_CONFIG
        assert "adaptive_context" in PERFORMANCE_CONFIG
        assert "tool_lazy_loading" in PERFORMANCE_CONFIG

    def test_optimized_generation_params(self):
        """Test that generation params are optimized."""
        from config.settings import GENERATION_PARAMS

        assert GENERATION_PARAMS["num_ctx"] <= 32768
        assert GENERATION_PARAMS["num_predict"] <= 16384
        assert "num_batch" in GENERATION_PARAMS
        assert GENERATION_PARAMS["num_batch"] >= 128


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
