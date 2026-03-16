# ALMF Memory Contract

## Purpose

This contract defines the stable surfaces for the Anvil Latent Memory Fabric (ALMF).
Both the local-first SQLite profile and the enterprise scale-out profile must honor
the same canonical object schema, projection semantics, replay compatibility rules,
snapshot format, and audit requirements.

## Canonical Object Rules

- Every memory object is canonical first and projection-derived second.
- Every memory object must have:
  - `memory_id`
  - `campaign_id`
  - `memory_kind`
  - `payload_json`
  - `provenance_json`
  - `canonical_hash`
  - `retention_class`
  - `sensitivity_class`
- Derived projections and latent packages must reference a canonical `memory_id`.

## Backend Contract

- Local-first profile:
  - canonical metadata in SQLite via `CampaignStateStore`
  - retrieval indexes on the local filesystem
  - latent blobs in deterministic `safetensors`
- Enterprise profile:
  - may request Postgres metadata
  - may request object-store blobs
  - must fall back to SQLite when Postgres drivers or DSNs are unavailable
  - must preserve the same ALMF schema and snapshot compatibility

## Snapshot Contract

- Snapshot schema version: `almf.snapshot.v1`
- Required coverage:
  - canonical memory metadata
  - aliases
  - edges
  - read and feedback audit logs
  - retrieval index artifacts
  - latent blobs
- Restore must support campaign-scoped remapping.

## Replay Contract

- No latent package may exist without a canonical owner memory object.
- Every latent package must retain:
  - model family and revision
  - tokenizer hash
  - prompt protocol hash
  - runtime version
  - dimensional metadata
  - checksum
- Replay must be compatibility-gated before restore.

## Governance Contract

- Memory reads and replays are auditable.
- Retention policy is explicit per memory kind.
- Sensitivity classification must be present.
- Provenance must be complete before promotion of high-value artifacts.
- Contradictions must be surfaced before roadmap promotion.

## Benchmark Contract

- Benchmark schema version: `almf.benchmark.v1`
- Core metrics:
  - `top_k_evidence_recall`
  - `contradiction_recall`
  - `ndcg`
  - `answer_grounding_score`
  - `false_memory_rate`
  - `exact_replay_success_rate`
  - `degraded_replay_success_rate`
  - `warm_start_token_savings`
- Benchmark gates are required before migration or replay promotion.
