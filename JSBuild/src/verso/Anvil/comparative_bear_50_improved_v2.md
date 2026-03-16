# Comparative Port Plan: target-4fab4804

Report ID: comparative_1773473739_5c47c5a3
Generated: Sat Mar 14 01:35:39 2026
Candidate count: 1
Port ledger count: 50
Frontier packet count: 5
Creation ledger count: 1
Native migration program count: 2

## Executive Summary

This markdown is optimized for deep port planning rather than fleet triage. It surfaced 10 primary recommendations and 0 secondary opportunities across 1 candidate comparisons, while preserving low-signal analogues in separate sections.

## Artifacts

- best_of_breed_path: /home/mike/Documents/Github/Anvil/.saguaro/comparative/reports/comparative_1773473739_5c47c5a3.best_of_breed.json
- creation_ledger_path: /home/mike/Documents/Github/Anvil/.saguaro/comparative/reports/comparative_1773473739_5c47c5a3.creation_ledger.json
- frontier_packets_path: /home/mike/Documents/Github/Anvil/.saguaro/comparative/reports/comparative_1773473739_5c47c5a3.frontier_packets.json
- json_path: /home/mike/Documents/Github/Anvil/.saguaro/comparative/reports/comparative_1773473739_5c47c5a3.json
- markdown_path: /home/mike/Documents/Github/Anvil/comparative_reports/comparative_1773473739_5c47c5a3.md
- native_programs_path: /home/mike/Documents/Github/Anvil/.saguaro/comparative/reports/comparative_1773473739_5c47c5a3.native_programs.json
- port_ledger_path: /home/mike/Documents/Github/Anvil/.saguaro/comparative/reports/comparative_1773473739_5c47c5a3.port_ledger.json

## Telemetry

- aggregate_candidate_count: 50
- comparative_boot_ms: 12115.172
- comparative_frontier_acceptance_rate: 0.4
- comparative_program_count: 2
- compare_lock_wait_ms: 0.011
- compare_target_cache_bytes: 919159
- compare_target_manifest_reuse_hits: 1
- corpus_session_count: 2
- creation_candidate_count: 1
- feature_winner_count: 2
- fleet_parallel_workers: 1
- fleet_processing_mode: batched
- fleet_processing_order: ['repo-analysis-bear-master-30b1b0c2']
- fleet_queue_depth: 1
- fleet_repo_count: 1
- migration_recipe_lowering_count: 2
- native_rewrite_promotion_count: 2
- pair_candidates_after_filter: 12544
- pair_candidates_before_filter: 12544
- per_candidate_compare_ms: [11670.56]
- primary_recommendation_count: 10
- report_compile_ms: 34.394
- report_evidence_density: 66.0
- reuse_only: False
- secondary_recommendation_count: 0
- skipped_candidate_count: 0
- skipped_candidates: []
- subsystem_upgrade_count: 2
- target_signature_build_ms: 61.983
- target_twin_reuse_hits: 0
- winner_confidence_distribution: {'medium': 1, 'high': 1}
- wrapper_budget_count: 2

## Best of Breed

- rust: repo-analysis-bear-master-30b1b0c2 [score=0.900] rust -> feature/rust (Saguaro)
- artifact_output: repo-analysis-bear-master-30b1b0c2 [score=0.790] bear/src/output/clang/converter.rs -> core/artifacts/manager.py (Anvil)

## Subsystem Upgrade Routing

- Anvil: primary=10, secondary=0, creation=0
  Top families: artifact_output
  Target zones: core/artifacts/manager.py
- Saguaro: primary=0, secondary=0, creation=1
  Top families: rust
  Target zones: none

## Creation Ledger

- rust [language_enablement] priority=0.900 repo-analysis-bear-master-30b1b0c2 shows credible rust support beyond the target's current truth matrix. (Saguaro)

## Native Migration Programs

- rust [priority=0.900] rust -> feature/rust (Saguaro)
- artifact_output [priority=0.790] bear/src/output/clang/converter.rs -> core/artifacts/manager.py (Anvil)

## Candidate: repo-analysis-bear-master-30b1b0c2

repo-analysis-bear-master-30b1b0c2 best aligns with target-4fab4804 through artifact_output on bear/src/output/clang/converter.rs -> core/artifacts/manager.py with posture native_rewrite. Current best landing zone is in Anvil. Leading migration themes: artifact_output.

### Candidate Scorecard
- Overall fit: high
- Top feature families: artifact_output
- Primary recommendations: 10
- Secondary recommendations: 0
- Low-signal relations: 40
- Preferred implementation language: cpp
- Common tech stack: artifact, build, c, config, config_surface, doc, formatter, markdown, native, report_surface, reporting, secondary_surface, session, source, test, test_harness, toml, yaml
- Language overlap: c, markdown, toml, yaml
- Feature overlap: artifact_output
- Candidate feature gaps: none
- Recommended subsystems: Anvil
- Comparison backend: native_cpp

### Build Alignment
- Compatible: False
- Shared build files: none
- Build fingerprint depth: shallow

### Capability Delta
- Shared deep languages: c, toml, yaml
- Candidate-only deep languages: rust
- Target-only deep languages: cmake, cpp, json, python, shell, typescript
- Pair candidates before filter: 12544
- Pair candidates after filter: 12544

### Evidence Quality
- low_signal_count: 40
- pair_candidates_after_filter: 12544
- pair_candidates_before_filter: 12544
- pair_screen_backend: native_cpp
- primary_count: 10
- secondary_count: 0

### Subsystem Routing

- Anvil: primary=10, secondary=0, creation=0
  Top families: artifact_output
  Target zones: core/artifacts/manager.py
- Saguaro: primary=0, secondary=0, creation=1
  Top families: rust
  Target zones: none

### Upgrade Programs

- Anvil :: artifact_output -> core/artifacts/manager.py
  Sources (10): bear/src/output/clang/converter.rs, bear/src/output/clang/filter_duplicates.rs, bear/src/output/clang/filter_sources.rs, bear/src/output/clang/format.rs, bear/src/output/clang/mod.rs, bear/src/output/formats.rs, bear/src/output/json.rs, bear/src/output/writers.rs
  Posture: native_rewrite | Relation: reporting_analogue | Actionability: 0.790
  Value: reporting_gain, performance_gain
  Summary: Porting bear/src/output/clang/converter.rs would strengthen artifact_output and deliver reporting_gain, performance_gain.

### Primary Port Recommendations

#### 1. artifact_output
- Source: bear/src/output/clang/converter.rs
- Target: core/artifacts/manager.py
- Subsystem: Anvil [confidence=0.960]
- Posture: native_rewrite | Relation: reporting_analogue | Score: 0.790
- Why port: Porting bear/src/output/clang/converter.rs would strengthen artifact_output and deliver reporting_gain, performance_gain.
- What Anvil gets: reporting_gain, performance_gain
- Why this target: core/artifacts/manager.py is the nearest target surface because it shares artifact, report_surface. This routes into Anvil because feature families artifact_output favor Anvil; shared roles artifact, report_surface align with control_plane; target path core/artifacts/manager.py sits in Anvil.
- Subsystem rationale: feature families artifact_output favor Anvil; shared roles artifact, report_surface align with control_plane; target path core/artifacts/manager.py sits in Anvil
- Evidence: features=artifact_output; roles=artifact, report_surface; noise=none

#### 2. artifact_output
- Source: bear/src/output/clang/filter_duplicates.rs
- Target: core/artifacts/manager.py
- Subsystem: Anvil [confidence=0.960]
- Posture: native_rewrite | Relation: reporting_analogue | Score: 0.790
- Why port: Porting bear/src/output/clang/filter_duplicates.rs would strengthen artifact_output and deliver reporting_gain, performance_gain.
- What Anvil gets: reporting_gain, performance_gain
- Why this target: core/artifacts/manager.py is the nearest target surface because it shares artifact, report_surface. This routes into Anvil because feature families artifact_output favor Anvil; shared roles artifact, report_surface align with control_plane; target path core/artifacts/manager.py sits in Anvil.
- Subsystem rationale: feature families artifact_output favor Anvil; shared roles artifact, report_surface align with control_plane; target path core/artifacts/manager.py sits in Anvil
- Evidence: features=artifact_output; roles=artifact, report_surface; noise=none

#### 3. artifact_output
- Source: bear/src/output/clang/filter_sources.rs
- Target: core/artifacts/manager.py
- Subsystem: Anvil [confidence=0.960]
- Posture: native_rewrite | Relation: reporting_analogue | Score: 0.790
- Why port: Porting bear/src/output/clang/filter_sources.rs would strengthen artifact_output and deliver reporting_gain, performance_gain.
- What Anvil gets: reporting_gain, performance_gain
- Why this target: core/artifacts/manager.py is the nearest target surface because it shares artifact, report_surface. This routes into Anvil because feature families artifact_output favor Anvil; shared roles artifact, report_surface align with control_plane; target path core/artifacts/manager.py sits in Anvil.
- Subsystem rationale: feature families artifact_output favor Anvil; shared roles artifact, report_surface align with control_plane; target path core/artifacts/manager.py sits in Anvil
- Evidence: features=artifact_output; roles=artifact, report_surface; noise=none

#### 4. artifact_output
- Source: bear/src/output/clang/format.rs
- Target: core/artifacts/manager.py
- Subsystem: Anvil [confidence=0.960]
- Posture: native_rewrite | Relation: reporting_analogue | Score: 0.790
- Why port: Porting bear/src/output/clang/format.rs would strengthen artifact_output and deliver reporting_gain, performance_gain.
- What Anvil gets: reporting_gain, performance_gain
- Why this target: core/artifacts/manager.py is the nearest target surface because it shares artifact, report_surface. This routes into Anvil because feature families artifact_output favor Anvil; shared roles artifact, report_surface align with control_plane; target path core/artifacts/manager.py sits in Anvil.
- Subsystem rationale: feature families artifact_output favor Anvil; shared roles artifact, report_surface align with control_plane; target path core/artifacts/manager.py sits in Anvil
- Evidence: features=artifact_output; roles=artifact, report_surface; noise=none

#### 5. artifact_output
- Source: bear/src/output/clang/mod.rs
- Target: core/artifacts/manager.py
- Subsystem: Anvil [confidence=0.960]
- Posture: native_rewrite | Relation: reporting_analogue | Score: 0.790
- Why port: Porting bear/src/output/clang/mod.rs would strengthen artifact_output and deliver reporting_gain, performance_gain.
- What Anvil gets: reporting_gain, performance_gain
- Why this target: core/artifacts/manager.py is the nearest target surface because it shares artifact, report_surface. This routes into Anvil because feature families artifact_output favor Anvil; shared roles artifact, report_surface align with control_plane; target path core/artifacts/manager.py sits in Anvil.
- Subsystem rationale: feature families artifact_output favor Anvil; shared roles artifact, report_surface align with control_plane; target path core/artifacts/manager.py sits in Anvil
- Evidence: features=artifact_output; roles=artifact, report_surface; noise=none

#### 6. artifact_output
- Source: bear/src/output/formats.rs
- Target: core/artifacts/manager.py
- Subsystem: Anvil [confidence=0.960]
- Posture: native_rewrite | Relation: reporting_analogue | Score: 0.790
- Why port: Porting bear/src/output/formats.rs would strengthen artifact_output and deliver reporting_gain, performance_gain.
- What Anvil gets: reporting_gain, performance_gain
- Why this target: core/artifacts/manager.py is the nearest target surface because it shares artifact, report_surface. This routes into Anvil because feature families artifact_output favor Anvil; shared roles artifact, report_surface align with control_plane; target path core/artifacts/manager.py sits in Anvil.
- Subsystem rationale: feature families artifact_output favor Anvil; shared roles artifact, report_surface align with control_plane; target path core/artifacts/manager.py sits in Anvil
- Evidence: features=artifact_output; roles=artifact, report_surface; noise=none

#### 7. artifact_output
- Source: bear/src/output/json.rs
- Target: core/artifacts/manager.py
- Subsystem: Anvil [confidence=0.960]
- Posture: native_rewrite | Relation: reporting_analogue | Score: 0.790
- Why port: Porting bear/src/output/json.rs would strengthen artifact_output and deliver reporting_gain, performance_gain.
- What Anvil gets: reporting_gain, performance_gain
- Why this target: core/artifacts/manager.py is the nearest target surface because it shares artifact, report_surface. This routes into Anvil because feature families artifact_output favor Anvil; shared roles artifact, report_surface align with control_plane; target path core/artifacts/manager.py sits in Anvil.
- Subsystem rationale: feature families artifact_output favor Anvil; shared roles artifact, report_surface align with control_plane; target path core/artifacts/manager.py sits in Anvil
- Evidence: features=artifact_output; roles=artifact, report_surface; noise=none

#### 8. artifact_output
- Source: bear/src/output/writers.rs
- Target: core/artifacts/manager.py
- Subsystem: Anvil [confidence=0.960]
- Posture: native_rewrite | Relation: reporting_analogue | Score: 0.790
- Why port: Porting bear/src/output/writers.rs would strengthen artifact_output and deliver reporting_gain, performance_gain.
- What Anvil gets: reporting_gain, performance_gain
- Why this target: core/artifacts/manager.py is the nearest target surface because it shares artifact, report_surface. This routes into Anvil because feature families artifact_output favor Anvil; shared roles artifact, report_surface align with control_plane; target path core/artifacts/manager.py sits in Anvil.
- Subsystem rationale: feature families artifact_output favor Anvil; shared roles artifact, report_surface align with control_plane; target path core/artifacts/manager.py sits in Anvil
- Evidence: features=artifact_output; roles=artifact, report_surface; noise=none

#### 9. artifact_output
- Source: bear/src/output/mod.rs
- Target: core/artifacts/manager.py
- Subsystem: Anvil [confidence=0.960]
- Posture: native_rewrite | Relation: reporting_analogue | Score: 0.775
- Why port: Porting bear/src/output/mod.rs would strengthen artifact_output and deliver reporting_gain, performance_gain.
- What Anvil gets: reporting_gain, performance_gain
- Why this target: core/artifacts/manager.py is the nearest target surface because it shares artifact, report_surface. This routes into Anvil because feature families artifact_output favor Anvil; shared roles artifact, report_surface align with control_plane; target path core/artifacts/manager.py sits in Anvil.
- Subsystem rationale: feature families artifact_output favor Anvil; shared roles artifact, report_surface align with control_plane; target path core/artifacts/manager.py sits in Anvil
- Evidence: features=artifact_output; roles=artifact, report_surface; noise=none

#### 10. artifact_output
- Source: bear/src/output/statistics.rs
- Target: core/artifacts/manager.py
- Subsystem: Anvil [confidence=0.960]
- Posture: native_rewrite | Relation: reporting_analogue | Score: 0.775
- Why port: Porting bear/src/output/statistics.rs would strengthen artifact_output and deliver reporting_gain, performance_gain.
- What Anvil gets: reporting_gain, performance_gain
- Why this target: core/artifacts/manager.py is the nearest target surface because it shares artifact, report_surface. This routes into Anvil because feature families artifact_output favor Anvil; shared roles artifact, report_surface align with control_plane; target path core/artifacts/manager.py sits in Anvil.
- Subsystem rationale: feature families artifact_output favor Anvil; shared roles artifact, report_surface align with control_plane; target path core/artifacts/manager.py sits in Anvil
- Evidence: features=artifact_output; roles=artifact, report_surface; noise=none

### Disparate / Non-Obvious Opportunities

- No non-obvious opportunities were promoted above the low-signal threshold.

### Secondary Opportunities

- No secondary opportunities.

### Feature Synthesis

- artifact_output [winner_candidate] score=0.790 bear/src/output/clang/converter.rs -> core/artifacts/manager.py

### Detailed Migration Recipes
- Native Rewrite artifact_output from bear/src/output/clang/converter.rs into core/artifacts/manager.py
  Target: core/artifacts/manager.py | Subsystem: Anvil | Posture: native_rewrite | Implementation tier: python
  Why port: Porting bear/src/output/clang/converter.rs would strengthen artifact_output and deliver reporting_gain, performance_gain.
  Why here: core/artifacts/manager.py is the nearest target surface because it shares artifact, report_surface. This routes into Anvil because feature families artifact_output favor Anvil; shared roles artifact, report_surface align with control_plane; target path core/artifacts/manager.py sits in Anvil.
  Expected value: reporting_gain, performance_gain
  Verification: saguaro verify . --engines native,ruff,semantic, run affected tests
- Native Rewrite artifact_output from bear/src/output/clang/filter_duplicates.rs into core/artifacts/manager.py
  Target: core/artifacts/manager.py | Subsystem: Anvil | Posture: native_rewrite | Implementation tier: python
  Why port: Porting bear/src/output/clang/filter_duplicates.rs would strengthen artifact_output and deliver reporting_gain, performance_gain.
  Why here: core/artifacts/manager.py is the nearest target surface because it shares artifact, report_surface. This routes into Anvil because feature families artifact_output favor Anvil; shared roles artifact, report_surface align with control_plane; target path core/artifacts/manager.py sits in Anvil.
  Expected value: reporting_gain, performance_gain
  Verification: saguaro verify . --engines native,ruff,semantic, run affected tests
- Native Rewrite artifact_output from bear/src/output/clang/filter_sources.rs into core/artifacts/manager.py
  Target: core/artifacts/manager.py | Subsystem: Anvil | Posture: native_rewrite | Implementation tier: python
  Why port: Porting bear/src/output/clang/filter_sources.rs would strengthen artifact_output and deliver reporting_gain, performance_gain.
  Why here: core/artifacts/manager.py is the nearest target surface because it shares artifact, report_surface. This routes into Anvil because feature families artifact_output favor Anvil; shared roles artifact, report_surface align with control_plane; target path core/artifacts/manager.py sits in Anvil.
  Expected value: reporting_gain, performance_gain
  Verification: saguaro verify . --engines native,ruff,semantic, run affected tests
- Native Rewrite artifact_output from bear/src/output/clang/format.rs into core/artifacts/manager.py
  Target: core/artifacts/manager.py | Subsystem: Anvil | Posture: native_rewrite | Implementation tier: python
  Why port: Porting bear/src/output/clang/format.rs would strengthen artifact_output and deliver reporting_gain, performance_gain.
  Why here: core/artifacts/manager.py is the nearest target surface because it shares artifact, report_surface. This routes into Anvil because feature families artifact_output favor Anvil; shared roles artifact, report_surface align with control_plane; target path core/artifacts/manager.py sits in Anvil.
  Expected value: reporting_gain, performance_gain
  Verification: saguaro verify . --engines native,ruff,semantic, run affected tests
- Native Rewrite artifact_output from bear/src/output/clang/mod.rs into core/artifacts/manager.py
  Target: core/artifacts/manager.py | Subsystem: Anvil | Posture: native_rewrite | Implementation tier: python
  Why port: Porting bear/src/output/clang/mod.rs would strengthen artifact_output and deliver reporting_gain, performance_gain.
  Why here: core/artifacts/manager.py is the nearest target surface because it shares artifact, report_surface. This routes into Anvil because feature families artifact_output favor Anvil; shared roles artifact, report_surface align with control_plane; target path core/artifacts/manager.py sits in Anvil.
  Expected value: reporting_gain, performance_gain
  Verification: saguaro verify . --engines native,ruff,semantic, run affected tests
- Native Rewrite artifact_output from bear/src/output/formats.rs into core/artifacts/manager.py
  Target: core/artifacts/manager.py | Subsystem: Anvil | Posture: native_rewrite | Implementation tier: python
  Why port: Porting bear/src/output/formats.rs would strengthen artifact_output and deliver reporting_gain, performance_gain.
  Why here: core/artifacts/manager.py is the nearest target surface because it shares artifact, report_surface. This routes into Anvil because feature families artifact_output favor Anvil; shared roles artifact, report_surface align with control_plane; target path core/artifacts/manager.py sits in Anvil.
  Expected value: reporting_gain, performance_gain
  Verification: saguaro verify . --engines native,ruff,semantic, run affected tests
- Native Rewrite artifact_output from bear/src/output/json.rs into core/artifacts/manager.py
  Target: core/artifacts/manager.py | Subsystem: Anvil | Posture: native_rewrite | Implementation tier: python
  Why port: Porting bear/src/output/json.rs would strengthen artifact_output and deliver reporting_gain, performance_gain.
  Why here: core/artifacts/manager.py is the nearest target surface because it shares artifact, report_surface. This routes into Anvil because feature families artifact_output favor Anvil; shared roles artifact, report_surface align with control_plane; target path core/artifacts/manager.py sits in Anvil.
  Expected value: reporting_gain, performance_gain
  Verification: saguaro verify . --engines native,ruff,semantic, run affected tests
- Native Rewrite artifact_output from bear/src/output/writers.rs into core/artifacts/manager.py
  Target: core/artifacts/manager.py | Subsystem: Anvil | Posture: native_rewrite | Implementation tier: python
  Why port: Porting bear/src/output/writers.rs would strengthen artifact_output and deliver reporting_gain, performance_gain.
  Why here: core/artifacts/manager.py is the nearest target surface because it shares artifact, report_surface. This routes into Anvil because feature families artifact_output favor Anvil; shared roles artifact, report_surface align with control_plane; target path core/artifacts/manager.py sits in Anvil.
  Expected value: reporting_gain, performance_gain
  Verification: saguaro verify . --engines native,ruff,semantic, run affected tests
- Native Rewrite artifact_output from bear/src/output/mod.rs into core/artifacts/manager.py
  Target: core/artifacts/manager.py | Subsystem: Anvil | Posture: native_rewrite | Implementation tier: python
  Why port: Porting bear/src/output/mod.rs would strengthen artifact_output and deliver reporting_gain, performance_gain.
  Why here: core/artifacts/manager.py is the nearest target surface because it shares artifact, report_surface. This routes into Anvil because feature families artifact_output favor Anvil; shared roles artifact, report_surface align with control_plane; target path core/artifacts/manager.py sits in Anvil.
  Expected value: reporting_gain, performance_gain
  Verification: saguaro verify . --engines native,ruff,semantic, run affected tests
- Native Rewrite artifact_output from bear/src/output/statistics.rs into core/artifacts/manager.py
  Target: core/artifacts/manager.py | Subsystem: Anvil | Posture: native_rewrite | Implementation tier: python
  Why port: Porting bear/src/output/statistics.rs would strengthen artifact_output and deliver reporting_gain, performance_gain.
  Why here: core/artifacts/manager.py is the nearest target surface because it shares artifact, report_surface. This routes into Anvil because feature families artifact_output favor Anvil; shared roles artifact, report_surface align with control_plane; target path core/artifacts/manager.py sits in Anvil.
  Expected value: reporting_gain, performance_gain
  Verification: saguaro verify . --engines native,ruff,semantic, run affected tests

### Value Realization
- performance_gain: 10
- reporting_gain: 10

### Port Ledger
- candidate reporting_analogue bear/src/output/clang/converter.rs -> core/artifacts/manager.py [native_rewrite] score=0.790
- candidate reporting_analogue bear/src/output/clang/filter_duplicates.rs -> core/artifacts/manager.py [native_rewrite] score=0.790
- candidate reporting_analogue bear/src/output/clang/filter_sources.rs -> core/artifacts/manager.py [native_rewrite] score=0.790
- candidate reporting_analogue bear/src/output/clang/format.rs -> core/artifacts/manager.py [native_rewrite] score=0.790
- candidate reporting_analogue bear/src/output/clang/mod.rs -> core/artifacts/manager.py [native_rewrite] score=0.790
- candidate reporting_analogue bear/src/output/formats.rs -> core/artifacts/manager.py [native_rewrite] score=0.790
- candidate reporting_analogue bear/src/output/json.rs -> core/artifacts/manager.py [native_rewrite] score=0.790
- candidate reporting_analogue bear/src/output/writers.rs -> core/artifacts/manager.py [native_rewrite] score=0.790
- candidate reporting_analogue bear/src/output/mod.rs -> core/artifacts/manager.py [native_rewrite] score=0.775
- candidate reporting_analogue bear/src/output/statistics.rs -> core/artifacts/manager.py [native_rewrite] score=0.775

### Frontier Packets
- Port artifact_output from bear/src/output/clang/converter.rs [priority=0.790, Anvil] tracks=comparative_spike, verification_lane
- Port artifact_output from bear/src/output/clang/filter_duplicates.rs [priority=0.790, Anvil] tracks=comparative_spike, verification_lane
- Port artifact_output from bear/src/output/clang/filter_sources.rs [priority=0.790, Anvil] tracks=comparative_spike, verification_lane
- Port artifact_output from bear/src/output/clang/format.rs [priority=0.790, Anvil] tracks=comparative_spike, verification_lane
- Port artifact_output from bear/src/output/clang/mod.rs [priority=0.790, Anvil] tracks=comparative_spike, verification_lane

### Creation Ledger
- rust [language_enablement] priority=0.900 repo-analysis-bear-master-30b1b0c2 shows credible rust support beyond the target's current truth matrix.

### Low-Signal and Generic Analogues
- [0.322] disparate_mechanism_candidate :: intercept-preload/src/session.rs -> Saguaro/saguaro/state/ledger.py (native_rewrite)
- [0.268] performance_upgrade_path :: bear/src/args.rs -> Saguaro/build_secure.sh (native_rewrite)
- [0.250] performance_upgrade_path :: bear/src/modes/mod.rs -> Saguaro/build_secure.sh (native_rewrite)
- [0.229] analogous_mechanism :: bear/src/args.rs -> core/simd/mps_isometry_helpers.h (native_rewrite)
- [0.229] analogous_mechanism :: intercept-preload/src/c/shim.c -> Saguaro/build_secure.sh (native_rewrite)
- [0.239] analogous_mechanism :: intercept-preload/src/implementation.rs -> Saguaro/build_secure.sh (native_rewrite)
- [0.226] analogous_mechanism :: platform-checks/src/lib.rs -> Saguaro/build_secure.sh (native_rewrite)
- [0.237] analogous_mechanism :: bear/src/context.rs -> Saguaro/build_secure.sh (native_rewrite)
- [0.220] analogous_mechanism :: bear/src/args.rs -> core/simd/fused_text_tokenizer_op.h (native_rewrite)
- [0.231] analogous_mechanism :: bear/src/intercept/tcp.rs -> Saguaro/build_secure.sh (native_rewrite)
- [0.229] analogous_mechanism :: bear/src/semantic/interpreters/compilers/arguments.rs -> Saguaro/build_secure.sh (native_rewrite)
- [0.214] analogous_mechanism :: bear/src/semantic/interpreters/compilers/arguments.rs -> core/simd/fused_text_tokenizer_op.h (native_rewrite)
- [0.213] analogous_mechanism :: bear/src/modes/mod.rs -> core/simd/hyperdimensional_embedding_op.h (native_rewrite)
- [0.213] analogous_mechanism :: bear/src/modes/mod.rs -> core/simd/mps_isometry_helpers.h (native_rewrite)
- [0.211] analogous_mechanism :: bear/src/args.rs -> core/simd/quantum_embedding_op.h (native_rewrite)
- [0.220] analogous_mechanism :: bear/src/intercept/mod.rs -> Saguaro/build_secure.sh (native_rewrite)
- [0.207] analogous_mechanism :: bear/src/args.rs -> core/simd/fused_coconut_bfs_op.h (native_rewrite)
- [0.207] analogous_mechanism :: intercept-preload/src/implementation.rs -> core/simd/mps_isometry_helpers.h (native_rewrite)
- [0.204] analogous_mechanism :: bear/src/args.rs -> core/simd/hyperdimensional_embedding_op.h (native_rewrite)
- [0.204] analogous_mechanism :: bear/src/modes/mod.rs -> core/simd/fused_text_tokenizer_op.h (native_rewrite)

### Manual Validation Checklist
- [ ] bear/src/output/clang/converter.rs -> core/artifacts/manager.py features=artifact_output
- [ ] bear/src/output/clang/filter_duplicates.rs -> core/artifacts/manager.py features=artifact_output
- [ ] bear/src/output/clang/filter_sources.rs -> core/artifacts/manager.py features=artifact_output
- [ ] bear/src/output/clang/format.rs -> core/artifacts/manager.py features=artifact_output
- [ ] bear/src/output/clang/mod.rs -> core/artifacts/manager.py features=artifact_output
- [ ] bear/src/output/formats.rs -> core/artifacts/manager.py features=artifact_output
- [ ] bear/src/output/json.rs -> core/artifacts/manager.py features=artifact_output
- [ ] bear/src/output/writers.rs -> core/artifacts/manager.py features=artifact_output
- [ ] bear/src/output/mod.rs -> core/artifacts/manager.py features=artifact_output
- [ ] bear/src/output/statistics.rs -> core/artifacts/manager.py features=artifact_output

### Overlay Graph
- Nodes: 40
- Edges: 50
