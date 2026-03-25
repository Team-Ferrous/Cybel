from __future__ import annotations

from saguaro.synthesis.cache import SynthesisCache


def test_synthesis_cache_reuses_records_by_spec_digest_and_runtime_context() -> None:
    cache = SynthesisCache()
    record = cache.put(
        spec_digest="spec-1",
        capability_digest="cap-1",
        repo_delta_id="delta-1",
        counterexamples=[{"reason": "seed"}],
        proof_refs=["proof.json"],
    )

    loaded = cache.get(
        spec_digest="spec-1",
        capability_digest="cap-1",
        repo_delta_id="delta-1",
    )

    assert loaded is not None
    assert loaded.cache_key == record.cache_key
    assert cache.stats()["proof_reuse_hits"] == 1

