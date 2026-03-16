from __future__ import annotations

import os

from benchmarks import continuous_qsg_benchmark as benchmark


def test_run_matrix_exports_native_thread_budget(monkeypatch):
    captured: dict[str, object] = {}

    class _FakeAdapter:
        def __init__(self, model, config):
            del model
            for key in (
                "ANVIL_NUM_THREADS",
                "ANVIL_NUM_THREADS_DECODE",
                "ANVIL_NUM_THREADS_BATCH",
                "ANVIL_NUM_UBATCH",
            ):
                captured[key] = os.environ.get(key)
            captured["max_active_requests"] = config.max_active_requests
            captured["batch_wait_timeout_ms"] = config.batch_wait_timeout_ms
            captured["max_prefill_rows_per_iteration"] = (
                config.max_prefill_rows_per_iteration
            )
            captured["state_page_rows"] = config.state_page_rows
            captured["continuous_interleaved_streams"] = (
                config.continuous_interleaved_streams
            )

        def get_runtime_status(self):
            return {"continuous_batching": {"ok": True}}

    def _fake_run_request(adapter, *, request_id, prompt, max_new_tokens):
        del adapter, request_id, prompt, max_new_tokens
        return benchmark.RequestResult(
            request_id="req-1",
            concurrency=1,
            prompt_class="short",
            scheduler_policy="fcfs",
            output_text="ok",
            generated_token_estimate=1,
        )

    monkeypatch.setattr(benchmark, "OllamaQSGAdapter", _FakeAdapter)
    monkeypatch.setattr(benchmark, "_run_request", _fake_run_request)

    result = benchmark._run_matrix(
        model="granite4:tiny-h",
        prompt_class="short",
        concurrency=1,
        max_active_requests=3,
        scheduler_policy="fcfs",
        state_page_rows=64,
        batch_wait_timeout_ms=1,
        max_prefill_rows_per_iteration=512,
        continuous_interleaved_streams=True,
        max_new_tokens=8,
        context_length=1024,
        decode_threads=8,
        batch_threads=6,
        ubatch=32,
    )

    assert result.metrics_snapshot == {"ok": True}
    assert captured["ANVIL_NUM_THREADS"] == "8"
    assert captured["ANVIL_NUM_THREADS_DECODE"] == "8"
    assert captured["ANVIL_NUM_THREADS_BATCH"] == "6"
    assert captured["ANVIL_NUM_UBATCH"] == "32"
    assert captured["max_active_requests"] == 3
    assert captured["batch_wait_timeout_ms"] == 1
    assert captured["max_prefill_rows_per_iteration"] == 512
    assert captured["state_page_rows"] == 64
    assert captured["continuous_interleaved_streams"] is True
