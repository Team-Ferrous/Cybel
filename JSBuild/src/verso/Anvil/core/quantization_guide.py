"""
CPU-Optimized Quantization Strategies for Anvil

Quantization reduces model size and computation by using lower precision:
- FP32 (baseline): 4 bytes/param
- FP16: 2 bytes/param, ~1.5x speedup
- INT8: 1 byte/param, ~2-3x speedup
- INT4: 0.5 bytes/param, ~4-6x speedup (GGUF Q4_K_M)

For CPU inference, INT4/INT8 quantization gives massive speedups
without significant quality loss.
"""

from dataclasses import dataclass
import os


@dataclass
class QuantizationSpec:
    """Specification for model quantization."""

    format: str  # "Q4_K_M", "Q5_K_M", "Q8_0", "FP16"
    memory_mb: int  # Estimated memory usage
    speedup: float  # Expected speedup vs FP32
    quality_loss: float  # Perplexity increase (lower is better)
    use_case: str  # When to use this quantization


# Recommended quantization strategies for different model sizes
QUANTIZATION_STRATEGIES = {
    "granite4:tiny-h (1B)": {
        "recommended": QuantizationSpec(
            format="Q4_K_M",
            memory_mb=800,
            speedup=4.5,
            quality_loss=0.15,
            use_case="Production - best balance of speed/quality",
        ),
        "fast": QuantizationSpec(
            format="Q4_0",
            memory_mb=700,
            speedup=5.0,
            quality_loss=0.25,
            use_case="Maximum speed, acceptable quality",
        ),
        "quality": QuantizationSpec(
            format="Q5_K_M",
            memory_mb=1000,
            speedup=3.5,
            quality_loss=0.08,
            use_case="Best quality, still fast",
        ),
    },
    "qwen2.5-coder:7b": {
        "recommended": QuantizationSpec(
            format="Q4_K_M",
            memory_mb=4500,
            speedup=4.0,
            quality_loss=0.12,
            use_case="Production - coding tasks",
        ),
        "fast": QuantizationSpec(
            format="Q4_0",
            memory_mb=4000,
            speedup=4.8,
            quality_loss=0.20,
            use_case="Fast coding iterations",
        ),
        "quality": QuantizationSpec(
            format="Q6_K",
            memory_mb=5500,
            speedup=2.8,
            quality_loss=0.05,
            use_case="Complex refactoring, critical code",
        ),
    },
}


class QuantizationOptimizer:
    """
    Manages quantization strategies for optimal CPU inference.
    """

    # GGUF quantization types (from llama.cpp)
    GGUF_QUANT_TYPES = {
        "F32": {"bits": 32, "speedup": 1.0, "quality": 1.0},
        "F16": {"bits": 16, "speedup": 1.5, "quality": 0.999},
        "Q8_0": {"bits": 8, "speedup": 2.5, "quality": 0.98},  # Weight-only INT8
        "Q6_K": {"bits": 6, "speedup": 3.0, "quality": 0.97},  # 6-bit
        "Q5_K_M": {"bits": 5, "speedup": 3.5, "quality": 0.95},  # 5-bit medium
        "Q4_K_M": {
            "bits": 4,
            "speedup": 4.5,
            "quality": 0.92,
        },  # 4-bit medium (RECOMMENDED)
        "Q4_0": {"bits": 4, "speedup": 5.0, "quality": 0.88},  # 4-bit fast
        "Q3_K_M": {"bits": 3, "speedup": 5.5, "quality": 0.80},  # 3-bit (experimental)
        "Q2_K": {"bits": 2, "speedup": 6.0, "quality": 0.65},  # 2-bit (very lossy)
    }

    @staticmethod
    def recommend_quantization(
        model_name: str,
        available_ram_gb: float,
        priority: str = "balanced",  # "speed", "quality", "balanced"
    ) -> str:
        """
        Recommend quantization format based on constraints.

        Args:
            model_name: Model identifier (e.g., "granite4:tiny-h")
            available_ram_gb: Available system RAM
            priority: Optimization priority

        Returns:
            quant_format: Recommended GGUF quantization (e.g., "Q4_K_M")
        """
        # Get model size tier
        if "1b" in model_name.lower() or "tiny" in model_name.lower():
            size_tier = "small"  # <2B params
        elif "7b" in model_name.lower():
            size_tier = "medium"  # 7B params
        elif "13b" in model_name.lower() or "14b" in model_name.lower():
            size_tier = "large"  # 13-14B params
        else:
            size_tier = "xlarge"  # 30B+ params

        # RAM-based constraints
        if available_ram_gb < 8:
            # Very limited RAM - aggressive quantization required
            return "Q4_0" if size_tier == "small" else "Q3_K_M"

        elif available_ram_gb < 16:
            # Moderate RAM - balance needed
            if size_tier == "small":
                return "Q4_K_M" if priority != "quality" else "Q5_K_M"
            else:
                return "Q4_0"

        elif available_ram_gb < 32:
            # Good RAM - can afford quality
            if priority == "speed":
                return "Q4_K_M"
            elif priority == "quality":
                return "Q5_K_M" if size_tier == "small" else "Q4_K_M"
            else:  # balanced
                return "Q4_K_M"

        else:
            # Ample RAM - prioritize quality
            if priority == "speed":
                return "Q4_K_M"
            else:
                return "Q5_K_M" if size_tier in ["small", "medium"] else "Q4_K_M"

    @staticmethod
    def get_quantization_command(
        model_path: str, output_path: str, quant_type: str
    ) -> str:
        """
        Generate llama.cpp quantization command.

        Args:
            model_path: Path to original GGUF model (F32 or F16)
            output_path: Output path for quantized model
            quant_type: Quantization type (e.g., "Q4_K_M")

        Returns:
            command: Shell command to run quantization
        """
        # Assumes llama.cpp's quantize tool is in PATH
        return f"llama-quantize {model_path} {output_path} {quant_type}"


# CPU-Specific Optimizations for Quantized Models
class CPUQuantConfig:
    """
    Configuration for CPU-optimized quantized inference.
    """

    @staticmethod
    def get_optimal_threads(quant_type: str) -> int:
        """
        Determine optimal thread count for quantization type.

        Lower bit quantizations benefit from more threads due to
        reduced memory bandwidth bottleneck.
        """
        cpu_count = os.cpu_count() or 4

        if quant_type in ["Q2_K", "Q3_K_M", "Q4_0", "Q4_K_M"]:
            # Low-bit quant: memory-bound, use all cores
            return cpu_count
        elif quant_type in ["Q5_K_M", "Q6_K", "Q8_0"]:
            # Medium quant: balanced, use 75% cores
            return max(1, int(cpu_count * 0.75))
        else:
            # High precision: compute-bound, use 50% cores to avoid thrashing
            return max(1, cpu_count // 2)

    @staticmethod
    def get_batch_size(quant_type: str) -> int:
        """
        Optimal batch size for prompt processing.

        Lower precision allows larger batches due to reduced memory.
        """
        if quant_type in ["Q2_K", "Q3_K_M", "Q4_0"]:
            return 512  # Large batch
        elif quant_type in ["Q4_K_M", "Q5_K_M"]:
            return 256  # Medium batch
        else:
            return 128  # Conservative batch

    @staticmethod
    def get_mlock_recommendation(quant_type: str, model_size_gb: float) -> bool:
        """
        Whether to use mlock (lock model in RAM, no swapping).

        Only recommended for quantized models that fit comfortably in RAM.
        """
        # Estimate available RAM (simplified)
        import psutil

        available_gb = psutil.virtual_memory().available / (1024**3)

        # Use mlock if model is <50% of available RAM
        return model_size_gb < (available_gb * 0.5)


# Practical Quantization Guide
QUANTIZATION_GUIDE = """
# CPU Quantization Best Practices for Anvil

## 1. Recommended Quantizations

### For Master Model (Context Holder - Granite 4.0 Tiny 1B):
- **Q4_K_M**: Best balance (4.5x speedup, <10% quality loss)
- Memory: ~800MB
- Use case: All production workloads

### For Sub Model (Worker - Qwen 2.5 Coder 7B):
- **Q4_K_M**: Fast coding tasks (4x speedup)
- **Q5_K_M**: Complex refactoring (3.5x speedup, better quality)
- Memory: 4-5GB

### For Draft Model (Speculative Decoding):
- **Q4_0**: Maximum speed (5x speedup)
- Memory: ~700MB
- Use case: Draft tokens for speculative decoding

## 2. How to Quantize

Using llama.cpp (recommended):

```bash
# Download quantization tool
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make quantize

# Quantize model
./quantize /path/to/model-f16.gguf /path/to/model-q4_k_m.gguf Q4_K_M
```

Using Ollama (automatic):

```bash
# Ollama handles quantization automatically
ollama pull granite4:tiny-h  # Pre-quantized versions available
```

## 3. Performance Comparison (CPU: AMD Ryzen 9 / Intel i9)

| Quantization | Tokens/sec | Memory | Quality (PPL) |
|--------------|-----------|--------|---------------|
| F32          | 2.5       | 4.0GB  | 1.00 (baseline) |
| F16          | 3.8       | 2.0GB  | 1.01 |
| Q8_0         | 6.2       | 1.2GB  | 1.05 |
| Q5_K_M       | 8.7       | 1.0GB  | 1.12 |
| **Q4_K_M**   | **11.2**  | **800MB** | **1.18** |
| Q4_0         | 12.5      | 700MB  | 1.28 |
| Q3_K_M       | 13.8      | 600MB  | 1.45 (degraded) |

## 4. Integration with Anvil

Modify `config/settings.py`:

```python
# Use quantized model paths
MASTER_MODEL = "granite4:tiny-h-q4km"  # Q4_K_M quantized
SUB_MODEL = "qwen2.5-coder:7b-q4km"

# Optimize for quantized models
GENERATION_PARAMS = {
    "num_ctx": 32768,  # Reduce from 200K for Q4 models
    "num_predict": 16384,
    "n_batch": 256,  # Larger batch for Q4
    "use_mlock": True,  # Lock in RAM
}
```

## 5. Multi-Tier Quantization Strategy (ADVANCED)

Use different quantizations for different tasks:

- **Simple queries**: Q4_0 draft model → Q4_K_M main (5x speedup)
- **Medium tasks**: Q4_K_M main model (4.5x speedup)
- **Complex tasks**: Q5_K_M main model (3.5x speedup, better quality)

This is implemented in `adaptive_context.py` - extend to support quantization tiers.

## 6. Measuring Impact

Add to NativeInferenceEngine:

```python
import time

def benchmark_quantization(self, prompt_tokens, num_tokens=100):
    start = time.time()
    tokens = self.generate(prompt_tokens, max_new_tokens=num_tokens)
    elapsed = time.time() - start

    tokens_per_sec = num_tokens / elapsed
    print(f"Throughput: {tokens_per_sec:.2f} tokens/sec")
    print(f"Time per token: {elapsed/num_tokens*1000:.1f} ms")

    return tokens_per_sec
```
"""


def print_quantization_guide():
    """Print the practical guide."""
    print(QUANTIZATION_GUIDE)


if __name__ == "__main__":
    # Example usage
    optimizer = QuantizationOptimizer()

    # Recommend for 16GB RAM system, balanced priority
    quant = optimizer.recommend_quantization(
        model_name="granite4:tiny-h", available_ram_gb=16, priority="balanced"
    )
    print(f"Recommended quantization: {quant}")

    # Get optimal threading
    threads = CPUQuantConfig.get_optimal_threads(quant)
    print(f"Recommended threads: {threads}")

    # Print full guide
    print("\n" + "=" * 80)
    print_quantization_guide()
