from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import benchmarks.native_qsg_benchmark as native_bench


def _fake_run_once(
    model: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    sampling_profile: str | None = None,
    override_params: dict | None = None,
    run_index: int = 0,
) -> native_bench.BenchmarkResult:
    del prompt
    del temperature
    del sampling_profile
    del override_params
    generate_seconds = 0.5
    new_tokens = int(max_new_tokens)
    return native_bench.BenchmarkResult(
        model=model,
        threads=2,
        max_new_tokens=new_tokens,
        digest="sha256:test",
        temperature=0.0,
        run_index=run_index,
        context_length=2048,
        load_seconds=0.01,
        generate_seconds=generate_seconds,
        first_token_latency_seconds=0.02,
        prefill_throughput_tps=200.0,
        decode_throughput_tps=16.0,
        decode_threads=2,
        batch_threads=2,
        kv_used_cells=12,
        new_tokens=new_tokens,
        tokens_per_second=float(new_tokens) / generate_seconds,
        sample_text=f"{model}-ok",
        printable_ratio=1.0,
        ascii_ratio=1.0,
        word_count=2,
        measurement_valid=True,
    )


def _evaluate_gate(rows: list[dict], min_tps: float) -> dict:
    mean_tps = statistics.mean(float(row["tokens_per_second"]) for row in rows)
    return {
        "passed": bool(mean_tps >= float(min_tps)),
        "mean_tps": float(mean_tps),
        "min_tps": float(min_tps),
    }


def test_perf_regression_harness_emits_structured_metrics_json(monkeypatch, capsys) -> None:
    monkeypatch.setattr(native_bench, "_run_once", _fake_run_once)
    monkeypatch.setattr(
        native_bench,
        "parse_args",
        lambda: argparse.Namespace(
            model=["stub-model"],
            prompt="x",
            max_new_tokens=8,
            temperature=0.0,
            sampling_profile=None,
            top_p=None,
            top_k=None,
            min_p=None,
            presence_penalty=None,
            repetition_penalty=None,
            runs=2,
            json=True,
            json_out=None,
            markdown_out=None,
            min_decode_tps=None,
            max_ttft_ms=None,
            min_printable_ratio=0.95,
            max_repeated_4gram_ratio=0.15,
            max_repeated_8gram_ratio=0.05,
            min_ascii_ratio=None,
            min_word_count=None,
            max_rss_growth_mb=None,
            require_utf8=False,
            require_measurement_valid=False,
            require_openmp=False,
            require_avx2=False,
            require_mmap=False,
        ),
    )

    rc = native_bench.main()
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list)
    assert len(payload) == 2
    assert {item["run_index"] for item in payload} == {0, 1}
    required = {
        "model",
        "threads",
        "max_new_tokens",
        "digest",
        "context_length",
        "generate_seconds",
        "runtime_total_seconds",
        "runtime_prefill_seconds",
        "runtime_decode_seconds",
        "first_token_latency_seconds",
        "ttft_ms",
        "prefill_throughput_tps",
        "effective_prefill_throughput_tps",
        "decode_throughput_tps",
        "decode_threads",
        "batch_threads",
        "template_name",
        "active_thread_mode",
        "kv_used_cells",
        "new_tokens",
        "tokens_per_second",
        "prompt_cache_hit_ratio",
        "printable_ratio",
        "measurement_valid",
    }
    for row in payload:
        assert required.issubset(row.keys())


def test_perf_regression_gate_pass_fail_behavior() -> None:
    rows = [
        {"tokens_per_second": 16.0},
        {"tokens_per_second": 18.0},
        {"tokens_per_second": 20.0},
    ]

    pass_gate = _evaluate_gate(rows, min_tps=15.0)
    fail_gate = _evaluate_gate(rows, min_tps=25.0)

    assert pass_gate["passed"] is True
    assert fail_gate["passed"] is False
    assert pass_gate["mean_tps"] == fail_gate["mean_tps"]


def test_perf_regression_harness_fails_when_release_gate_is_missed(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(native_bench, "_run_once", _fake_run_once)
    monkeypatch.setattr(
        native_bench,
        "parse_args",
        lambda: argparse.Namespace(
            model=["stub-model"],
            prompt="x",
            max_new_tokens=8,
            temperature=0.0,
            sampling_profile=None,
            top_p=None,
            top_k=None,
            min_p=None,
            presence_penalty=None,
            repetition_penalty=None,
            runs=1,
            json=False,
            json_out=None,
            markdown_out=None,
            min_decode_tps=32.0,
            max_ttft_ms=None,
            min_printable_ratio=0.95,
            max_repeated_4gram_ratio=0.15,
            max_repeated_8gram_ratio=0.05,
            min_ascii_ratio=None,
            min_word_count=None,
            max_rss_growth_mb=None,
            require_utf8=False,
            require_measurement_valid=False,
            require_openmp=False,
            require_avx2=False,
            require_mmap=False,
        ),
    )

    rc = native_bench.main()

    assert rc == 1
    assert "failures:" in capsys.readouterr().out


def test_perf_regression_harness_writes_signoff_artifacts(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    monkeypatch.setattr(native_bench, "_run_once", _fake_run_once)
    json_out = tmp_path / "native-qsg.json"
    markdown_out = tmp_path / "native-qsg.md"
    monkeypatch.setattr(
        native_bench,
        "parse_args",
        lambda: argparse.Namespace(
            model=["stub-model"],
            prompt="x",
            max_new_tokens=8,
            temperature=0.0,
            sampling_profile=None,
            top_p=None,
            top_k=None,
            min_p=None,
            presence_penalty=None,
            repetition_penalty=None,
            runs=1,
            json=False,
            json_out=str(json_out),
            markdown_out=str(markdown_out),
            min_decode_tps=None,
            max_ttft_ms=None,
            min_printable_ratio=0.95,
            max_repeated_4gram_ratio=0.15,
            max_repeated_8gram_ratio=0.05,
            min_ascii_ratio=None,
            min_word_count=None,
            max_rss_growth_mb=None,
            require_utf8=False,
            require_measurement_valid=False,
            require_openmp=False,
            require_avx2=False,
            require_mmap=False,
        ),
    )

    rc = native_bench.main()
    _ = capsys.readouterr()

    assert rc == 0
    assert json.loads(json_out.read_text(encoding="utf-8"))[0]["model"] == "stub-model"
    assert "Native QSG Benchmark Report" in markdown_out.read_text(encoding="utf-8")


def test_perf_regression_harness_can_gate_on_basic_coherence_shape(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(native_bench, "_run_once", _fake_run_once)
    monkeypatch.setattr(
        native_bench,
        "parse_args",
        lambda: argparse.Namespace(
            model=["stub-model"],
            prompt="x",
            max_new_tokens=8,
            temperature=0.0,
            sampling_profile=None,
            top_p=None,
            top_k=None,
            min_p=None,
            presence_penalty=None,
            repetition_penalty=None,
            runs=1,
            json=False,
            json_out=None,
            markdown_out=None,
            min_decode_tps=None,
            max_ttft_ms=None,
            min_printable_ratio=0.95,
            max_repeated_4gram_ratio=0.15,
            max_repeated_8gram_ratio=0.05,
            min_ascii_ratio=0.95,
            min_word_count=3,
            max_rss_growth_mb=None,
            require_utf8=False,
            require_measurement_valid=False,
            require_openmp=False,
            require_avx2=False,
            require_mmap=False,
        ),
    )

    rc = native_bench.main()

    assert rc == 1
    assert "word_count" in capsys.readouterr().out


def test_run_once_restores_env_and_derives_missing_runtime_metrics(monkeypatch) -> None:
    original_env = {
        "ANVIL_NUM_THREADS": "7",
        "ANVIL_NUM_THREADS_DECODE": "6",
        "ANVIL_NUM_THREADS_BATCH": "5",
        "ANVIL_NUM_UBATCH": "4",
    }
    for key, value in original_env.items():
        monkeypatch.setenv(key, value)

    class FakeEngine:
        def __init__(self, model: str, context_length: int) -> None:
            del model
            del context_length
            self.num_threads_decode = 3
            self.num_threads_batch = 2
            self.num_threads = 3

        def prepare_prompt_tokens(self, prompt: str) -> list[int]:
            return [11, 12, 13, 14]

        def generate(self, prompt_tokens: list[int], **kwargs) -> list[int]:
            del kwargs
            return list(prompt_tokens) + [21, 22]

        def get_runtime_status(self) -> dict:
            return {
                "context_length": 4096,
                "prefill_seconds": 2.0,
                "decode_seconds": 4.0,
                "ttft_seconds": 2.5,
                "prompt_cache_hits": 2,
                "prompt_cache_misses": 1,
                "prompt_cache_reused_tokens": 1,
                "template_name": "chatml",
                "active_thread_mode": "split",
            }

        def detokenize(self, tokens: list[int]) -> str:
            del tokens
            return "hello world"

        def close(self) -> None:
            return None

    monkeypatch.setattr(native_bench, "NativeQSGEngine", FakeEngine)

    result = native_bench._run_once(
        model="stub-model",
        prompt="abc",
        max_new_tokens=2,
        temperature=0.0,
        run_index=0,
        context_length=4096,
        decode_threads=8,
        batch_threads=4,
        ubatch=1,
    )

    assert result.runtime_prefill_seconds == 2.0
    assert result.runtime_decode_seconds == 4.0
    assert result.runtime_total_seconds == 0.0
    assert result.prefill_throughput_tps == 2.0
    assert result.effective_prefill_tokens == 3
    assert result.effective_prefill_throughput_tps == 1.5
    assert result.decode_throughput_tps == 0.5
    assert result.prompt_cache_lookups == 3
    assert result.prompt_cache_hit_ratio == 2.0 / 3.0
    assert result.prompt_cache_reuse_ratio == 0.25
    assert result.template_name == "chatml"
    assert result.active_thread_mode == "split"
    assert result.measurement_source == "runtime_partial"
    assert result.requested_decode_threads == 8
    assert result.requested_batch_threads == 4
    assert result.requested_ubatch == 1

    for key, value in original_env.items():
        assert native_bench.os.environ.get(key) == value
