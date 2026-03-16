# Comparative Port Plan: target-4fab4804

Report ID: comparative_1773473631_3304bbfe
Generated: Sat Mar 14 01:33:51 2026
Candidate count: 1
Port ledger count: 79
Frontier packet count: 5
Creation ledger count: 1
Native migration program count: 6

## Executive Summary

This markdown is optimized for deep port planning rather than fleet triage. It surfaced 19 primary recommendations and 7 secondary opportunities across 1 candidate comparisons, while preserving low-signal analogues in separate sections.

## Artifacts

- best_of_breed_path: /home/mike/Documents/Github/Anvil/.saguaro/comparative/reports/comparative_1773473631_3304bbfe.best_of_breed.json
- creation_ledger_path: /home/mike/Documents/Github/Anvil/.saguaro/comparative/reports/comparative_1773473631_3304bbfe.creation_ledger.json
- frontier_packets_path: /home/mike/Documents/Github/Anvil/.saguaro/comparative/reports/comparative_1773473631_3304bbfe.frontier_packets.json
- json_path: /home/mike/Documents/Github/Anvil/.saguaro/comparative/reports/comparative_1773473631_3304bbfe.json
- markdown_path: /home/mike/Documents/Github/Anvil/comparative_reports/comparative_1773473631_3304bbfe.md
- native_programs_path: /home/mike/Documents/Github/Anvil/.saguaro/comparative/reports/comparative_1773473631_3304bbfe.native_programs.json
- port_ledger_path: /home/mike/Documents/Github/Anvil/.saguaro/comparative/reports/comparative_1773473631_3304bbfe.port_ledger.json

## Telemetry

- aggregate_candidate_count: 79
- comparative_boot_ms: 26331.956
- comparative_frontier_acceptance_rate: 1.2
- comparative_program_count: 6
- compare_lock_wait_ms: 0.011
- compare_target_cache_bytes: 919159
- compare_target_manifest_reuse_hits: 1
- corpus_session_count: 2
- creation_candidate_count: 1
- feature_winner_count: 6
- fleet_parallel_workers: 1
- fleet_processing_mode: batched
- fleet_processing_order: ['counterfit-main-3e8c38f5']
- fleet_queue_depth: 1
- fleet_repo_count: 1
- migration_recipe_lowering_count: 6
- native_rewrite_promotion_count: 6
- pair_candidates_after_filter: 18432
- pair_candidates_before_filter: 18432
- per_candidate_compare_ms: [25874.744]
- primary_recommendation_count: 19
- report_compile_ms: 0.0
- report_evidence_density: 111.0
- reuse_only: False
- secondary_recommendation_count: 7
- skipped_candidate_count: 0
- skipped_candidates: []
- subsystem_upgrade_count: 3
- target_signature_build_ms: 65.043
- target_twin_reuse_hits: 0
- winner_confidence_distribution: {'high': 2, 'medium': 4}
- wrapper_budget_count: 6

## Best of Breed

- artifact_output: counterfit-main-3e8c38f5 [score=0.842] counterfit/core/output.py -> core/artifacts/manager.py (Anvil)
- attack_orchestration: counterfit-main-3e8c38f5 [score=0.802] counterfit/core/attacks.py -> core/campaign/control_plane.py (Anvil)
- target_registry: counterfit-main-3e8c38f5 [score=0.794] counterfit/core/targets.py -> core/campaign/repo_registry.py (Anvil)
- framework_adapter: counterfit-main-3e8c38f5 [score=0.680] counterfit/core/frameworks.py -> core/campaign/tooling_factory.py (Anvil)
- optimization: counterfit-main-3e8c38f5 [score=0.617] counterfit/core/optimize.py -> core/qsg/continuous_engine.py (QSG)
- reporting: counterfit-main-3e8c38f5 [score=0.588] counterfit/core/reporting.py -> Saguaro/saguaro/analysis/report.py (Saguaro)

## Subsystem Upgrade Routing

- Anvil: primary=19, secondary=5, creation=1
  Top families: target_registry, attack_orchestration, artifact_output, framework_adapter
  Target zones: core/campaign/repo_registry.py, core/artifacts/manager.py, core/campaign/control_plane.py, cli/commands/thinking.py
- QSG: primary=0, secondary=1, creation=0
  Top families: optimization
  Target zones: core/qsg/continuous_engine.py
- Saguaro: primary=0, secondary=1, creation=0
  Top families: reporting
  Target zones: Saguaro/saguaro/analysis/report.py

## Creation Ledger

- framework_adapter [feature_creation] priority=0.700 counterfit-main-3e8c38f5 exposes framework_adapter capability that is absent from target-4fab4804 and should be created as a native primitive. (Anvil)

## Native Migration Programs

- artifact_output [priority=0.842] counterfit/core/output.py -> core/artifacts/manager.py (Anvil)
- attack_orchestration [priority=0.802] counterfit/core/attacks.py -> core/campaign/control_plane.py (Anvil)
- target_registry [priority=0.794] counterfit/core/targets.py -> core/campaign/repo_registry.py (Anvil)
- framework_adapter [priority=0.680] counterfit/core/frameworks.py -> core/campaign/tooling_factory.py (Anvil)
- optimization [priority=0.617] counterfit/core/optimize.py -> core/qsg/continuous_engine.py (QSG)
- reporting [priority=0.588] counterfit/core/reporting.py -> Saguaro/saguaro/analysis/report.py (Saguaro)

## Candidate: counterfit-main-3e8c38f5

counterfit-main-3e8c38f5 best aligns with target-4fab4804 through artifact_output on counterfit/core/output.py -> core/artifacts/manager.py with posture native_rewrite. Current best landing zone is in Anvil. Leading migration themes: artifact_output, attack_orchestration, framework_adapter.

### Candidate Scorecard
- Overall fit: high
- Top feature families: target_registry, attack_orchestration, artifact_output, framework_adapter, optimization, reporting
- Primary recommendations: 19
- Secondary recommendations: 7
- Low-signal relations: 53
- Preferred implementation language: cpp
- Common tech stack: adapter, artifact, build, cli, config, config_surface, core_runtime, doc, entrypoint, formatter, framework, json, markdown, module_init, optimizer, orchestration, python, report_surface, reporting, secondary_surface, security, service, source, state, target, test, test_harness, yaml
- Language overlap: json, markdown, python, unknown, yaml
- Feature overlap: artifact_output, attack_orchestration, framework_adapter, optimization, reporting, target_registry
- Candidate feature gaps: framework_adapter
- Recommended subsystems: Anvil, QSG, Saguaro
- Comparison backend: native_cpp

### Build Alignment
- Compatible: True
- Shared build files: requirements.txt, setup.py
- Build fingerprint depth: shallow

### Capability Delta
- Shared deep languages: json, python, yaml
- Candidate-only deep languages: none
- Target-only deep languages: c, cmake, cpp, shell, toml, typescript
- Pair candidates before filter: 18432
- Pair candidates after filter: 18432

### Evidence Quality
- low_signal_count: 53
- pair_candidates_after_filter: 18432
- pair_candidates_before_filter: 18432
- pair_screen_backend: native_cpp
- primary_count: 19
- secondary_count: 7

### Subsystem Routing

- Anvil: primary=19, secondary=5, creation=1
  Top families: target_registry, attack_orchestration, artifact_output, framework_adapter
  Target zones: core/campaign/repo_registry.py, core/artifacts/manager.py, core/campaign/control_plane.py, cli/commands/thinking.py
- QSG: primary=0, secondary=1, creation=0
  Top families: optimization
  Target zones: core/qsg/continuous_engine.py
- Saguaro: primary=0, secondary=1, creation=0
  Top families: reporting
  Target zones: Saguaro/saguaro/analysis/report.py

### Upgrade Programs

- Anvil :: artifact_output -> core/artifacts/manager.py
  Sources (4): counterfit/core/output.py, counterfit/reporting/image.py, counterfit/reporting/tabular.py, counterfit/reporting/text.py
  Posture: native_rewrite | Relation: reporting_analogue | Actionability: 0.842
  Value: reporting_gain
  Summary: Porting counterfit/core/output.py would strengthen artifact_output and deliver reporting_gain.
- Anvil :: attack_orchestration -> core/campaign/control_plane.py
  Sources (4): counterfit/core/attacks.py, counterfit/frameworks/textattack/textattack.py, counterfit/frameworks/textattack/scripts/generate_config.py, examples/terminal/core/state.py
  Posture: native_rewrite | Relation: disparate_mechanism_candidate | Actionability: 0.802
  Value: operator_experience_gain
  Summary: Porting counterfit/core/attacks.py would strengthen attack_orchestration and deliver operator_experience_gain.
- Anvil :: target_registry -> core/campaign/repo_registry.py
  Sources (11): counterfit/core/targets.py, counterfit/targets/movie_reviews.py, counterfit/targets/cart_pole/DCPW.py, counterfit/targets/cart_pole/cart_pole.py, counterfit/targets/cart_pole/cart_pole_initstate.py, counterfit/targets/creditfraud.py, counterfit/targets/digits_keras.py, counterfit/targets/digits_mlp.py
  Posture: native_rewrite | Relation: feature_gap_candidate | Actionability: 0.794
  Value: capability_gain, coverage_gain
  Summary: Porting counterfit/core/targets.py would strengthen target_registry and deliver capability_gain, coverage_gain.
- Anvil :: framework_adapter -> core/campaign/tooling_factory.py
  Sources (1): counterfit/core/frameworks.py
  Posture: native_rewrite | Relation: disparate_mechanism_candidate | Actionability: 0.680
  Value: capability_gain, coverage_gain
  Summary: Porting counterfit/core/frameworks.py would strengthen framework_adapter and deliver capability_gain, coverage_gain.
- Anvil :: attack_orchestration -> cli/commands/thinking.py
  Sources (4): examples/terminal/core/state.py, examples/terminal/core/terminal.py, examples/terminal/core/config.py, examples/terminal/commands/show.py
  Posture: native_rewrite | Relation: orchestration_analogue | Actionability: 0.630
  Value: operator_experience_gain
  Summary: Porting examples/terminal/core/state.py would strengthen attack_orchestration and deliver operator_experience_gain.
- QSG :: optimization -> core/qsg/continuous_engine.py
  Sources (1): counterfit/core/optimize.py
  Posture: native_rewrite | Relation: disparate_mechanism_candidate | Actionability: 0.617
  Value: operator_experience_gain
  Summary: Porting counterfit/core/optimize.py would strengthen optimization and deliver operator_experience_gain.
- Saguaro :: reporting -> Saguaro/saguaro/analysis/report.py
  Sources (1): counterfit/core/reporting.py
  Posture: native_rewrite | Relation: feature_gap_candidate | Actionability: 0.588
  Value: reporting_gain
  Summary: Porting counterfit/core/reporting.py would strengthen reporting and deliver reporting_gain.

### Primary Port Recommendations

#### 1. artifact_output
- Source: counterfit/core/output.py
- Target: core/artifacts/manager.py
- Subsystem: Anvil [confidence=0.960]
- Posture: native_rewrite | Relation: reporting_analogue | Score: 0.842
- Why port: Porting counterfit/core/output.py would strengthen artifact_output and deliver reporting_gain.
- What Anvil gets: reporting_gain
- Why this target: core/artifacts/manager.py is the nearest target surface because it shares artifact, core_runtime, service. This routes into Anvil because feature families artifact_output favor Anvil; shared roles artifact, core_runtime, service align with control_plane; target path core/artifacts/manager.py sits in Anvil.
- Subsystem rationale: feature families artifact_output favor Anvil; shared roles artifact, core_runtime, service align with control_plane; target path core/artifacts/manager.py sits in Anvil
- Evidence: features=artifact_output; roles=artifact, core_runtime, service; noise=none

#### 2. artifact_output
- Source: counterfit/reporting/image.py
- Target: core/artifacts/manager.py
- Subsystem: Anvil [confidence=0.960]
- Posture: native_rewrite | Relation: reporting_analogue | Score: 0.820
- Why port: Porting counterfit/reporting/image.py would strengthen artifact_output and deliver reporting_gain.
- What Anvil gets: reporting_gain
- Why this target: core/artifacts/manager.py is the nearest target surface because it shares artifact, report_surface. This routes into Anvil because feature families artifact_output favor Anvil; shared roles artifact, report_surface align with control_plane; target path core/artifacts/manager.py sits in Anvil.
- Subsystem rationale: feature families artifact_output favor Anvil; shared roles artifact, report_surface align with control_plane; target path core/artifacts/manager.py sits in Anvil
- Evidence: features=artifact_output; roles=artifact, report_surface; noise=none

#### 3. attack_orchestration
- Source: counterfit/core/attacks.py
- Target: core/campaign/control_plane.py
- Subsystem: Anvil [confidence=0.960]
- Posture: native_rewrite | Relation: disparate_mechanism_candidate | Score: 0.802
- Why port: Porting counterfit/core/attacks.py would strengthen attack_orchestration and deliver operator_experience_gain.
- What Anvil gets: operator_experience_gain
- Why this target: core/campaign/control_plane.py is the nearest target surface because it shares core_runtime, service. This routes into Anvil because feature families attack_orchestration favor Anvil; shared roles core_runtime, service align with control_plane; target path core/campaign/control_plane.py sits in Anvil.
- Subsystem rationale: feature families attack_orchestration favor Anvil; shared roles core_runtime, service align with control_plane; target path core/campaign/control_plane.py sits in Anvil
- Evidence: features=attack_orchestration; roles=core_runtime, service; noise=none

#### 4. artifact_output
- Source: counterfit/reporting/tabular.py
- Target: core/artifacts/manager.py
- Subsystem: Anvil [confidence=0.960]
- Posture: native_rewrite | Relation: reporting_analogue | Score: 0.802
- Why port: Porting counterfit/reporting/tabular.py would strengthen artifact_output and deliver reporting_gain.
- What Anvil gets: reporting_gain
- Why this target: core/artifacts/manager.py is the nearest target surface because it shares artifact, report_surface. This routes into Anvil because feature families artifact_output favor Anvil; shared roles artifact, report_surface align with control_plane; target path core/artifacts/manager.py sits in Anvil.
- Subsystem rationale: feature families artifact_output favor Anvil; shared roles artifact, report_surface align with control_plane; target path core/artifacts/manager.py sits in Anvil
- Evidence: features=artifact_output; roles=artifact, report_surface; noise=none

#### 5. artifact_output
- Source: counterfit/reporting/text.py
- Target: core/artifacts/manager.py
- Subsystem: Anvil [confidence=0.960]
- Posture: native_rewrite | Relation: reporting_analogue | Score: 0.802
- Why port: Porting counterfit/reporting/text.py would strengthen artifact_output and deliver reporting_gain.
- What Anvil gets: reporting_gain
- Why this target: core/artifacts/manager.py is the nearest target surface because it shares artifact, report_surface. This routes into Anvil because feature families artifact_output favor Anvil; shared roles artifact, report_surface align with control_plane; target path core/artifacts/manager.py sits in Anvil.
- Subsystem rationale: feature families artifact_output favor Anvil; shared roles artifact, report_surface align with control_plane; target path core/artifacts/manager.py sits in Anvil
- Evidence: features=artifact_output; roles=artifact, report_surface; noise=none

#### 6. target_registry
- Source: counterfit/core/targets.py
- Target: core/campaign/repo_registry.py
- Subsystem: Anvil [confidence=0.960]
- Posture: native_rewrite | Relation: disparate_mechanism_candidate | Score: 0.794
- Why port: Porting counterfit/core/targets.py would strengthen target_registry and deliver capability_gain, coverage_gain.
- What Anvil gets: capability_gain, coverage_gain
- Why this target: core/campaign/repo_registry.py is the nearest target surface because it shares core_runtime, service. This routes into Anvil because feature families target_registry favor Anvil; shared roles core_runtime, service align with control_plane; target path core/campaign/repo_registry.py sits in Anvil.
- Subsystem rationale: feature families target_registry favor Anvil; shared roles core_runtime, service align with control_plane; target path core/campaign/repo_registry.py sits in Anvil
- Evidence: features=target_registry; roles=core_runtime, service; noise=none

#### 7. target_registry
- Source: counterfit/targets/movie_reviews.py
- Target: core/campaign/repo_registry.py
- Subsystem: Anvil [confidence=0.960]
- Posture: native_rewrite | Relation: feature_gap_candidate | Score: 0.730
- Why port: Porting counterfit/targets/movie_reviews.py would strengthen target_registry and deliver capability_gain, coverage_gain.
- What Anvil gets: capability_gain, coverage_gain
- Why this target: core/campaign/repo_registry.py is the nearest target surface because it shares target_registry. This routes into Anvil because feature families target_registry favor Anvil; target path core/campaign/repo_registry.py sits in Anvil.
- Subsystem rationale: feature families target_registry favor Anvil; target path core/campaign/repo_registry.py sits in Anvil
- Evidence: features=target_registry; roles=none; noise=none

#### 8. target_registry
- Source: counterfit/targets/cart_pole/DCPW.py
- Target: core/campaign/repo_registry.py
- Subsystem: Anvil [confidence=0.960]
- Posture: native_rewrite | Relation: feature_gap_candidate | Score: 0.723
- Why port: Porting counterfit/targets/cart_pole/DCPW.py would strengthen target_registry and deliver capability_gain, coverage_gain.
- What Anvil gets: capability_gain, coverage_gain
- Why this target: core/campaign/repo_registry.py is the nearest target surface because it shares target_registry. This routes into Anvil because feature families target_registry favor Anvil; target path core/campaign/repo_registry.py sits in Anvil.
- Subsystem rationale: feature families target_registry favor Anvil; target path core/campaign/repo_registry.py sits in Anvil
- Evidence: features=target_registry; roles=none; noise=none

#### 9. attack_orchestration
- Source: counterfit/frameworks/textattack/textattack.py
- Target: core/campaign/control_plane.py
- Subsystem: Anvil [confidence=0.960]
- Posture: native_rewrite | Relation: feature_gap_candidate | Score: 0.722
- Why port: Porting counterfit/frameworks/textattack/textattack.py would strengthen attack_orchestration and deliver operator_experience_gain.
- What Anvil gets: operator_experience_gain
- Why this target: core/campaign/control_plane.py is the nearest target surface because it shares attack_orchestration. This routes into Anvil because feature families attack_orchestration favor Anvil; target path core/campaign/control_plane.py sits in Anvil.
- Subsystem rationale: feature families attack_orchestration favor Anvil; target path core/campaign/control_plane.py sits in Anvil
- Evidence: features=attack_orchestration; roles=none; noise=none

#### 10. target_registry
- Source: counterfit/targets/cart_pole/cart_pole.py
- Target: core/campaign/repo_registry.py
- Subsystem: Anvil [confidence=0.960]
- Posture: native_rewrite | Relation: feature_gap_candidate | Score: 0.714
- Why port: Porting counterfit/targets/cart_pole/cart_pole.py would strengthen target_registry and deliver capability_gain, coverage_gain.
- What Anvil gets: capability_gain, coverage_gain
- Why this target: core/campaign/repo_registry.py is the nearest target surface because it shares target_registry. This routes into Anvil because feature families target_registry favor Anvil; target path core/campaign/repo_registry.py sits in Anvil.
- Subsystem rationale: feature families target_registry favor Anvil; target path core/campaign/repo_registry.py sits in Anvil
- Evidence: features=target_registry; roles=none; noise=none

#### 11. target_registry
- Source: counterfit/targets/cart_pole/cart_pole_initstate.py
- Target: core/campaign/repo_registry.py
- Subsystem: Anvil [confidence=0.960]
- Posture: native_rewrite | Relation: feature_gap_candidate | Score: 0.714
- Why port: Porting counterfit/targets/cart_pole/cart_pole_initstate.py would strengthen target_registry and deliver capability_gain, coverage_gain.
- What Anvil gets: capability_gain, coverage_gain
- Why this target: core/campaign/repo_registry.py is the nearest target surface because it shares target_registry. This routes into Anvil because feature families target_registry favor Anvil; target path core/campaign/repo_registry.py sits in Anvil.
- Subsystem rationale: feature families target_registry favor Anvil; target path core/campaign/repo_registry.py sits in Anvil
- Evidence: features=target_registry; roles=none; noise=none

#### 12. target_registry
- Source: counterfit/targets/creditfraud.py
- Target: core/campaign/repo_registry.py
- Subsystem: Anvil [confidence=0.960]
- Posture: native_rewrite | Relation: feature_gap_candidate | Score: 0.714
- Why port: Porting counterfit/targets/creditfraud.py would strengthen target_registry and deliver capability_gain, coverage_gain.
- What Anvil gets: capability_gain, coverage_gain
- Why this target: core/campaign/repo_registry.py is the nearest target surface because it shares target_registry. This routes into Anvil because feature families target_registry favor Anvil; target path core/campaign/repo_registry.py sits in Anvil.
- Subsystem rationale: feature families target_registry favor Anvil; target path core/campaign/repo_registry.py sits in Anvil
- Evidence: features=target_registry; roles=none; noise=none

### Disparate / Non-Obvious Opportunities

- examples/terminal/core/state.py -> cli/commands/thinking.py [orchestration_analogue, native_rewrite, Anvil, score=0.630]
  Why it is non-obvious: Role alignment is stronger than lexical overlap.
  Why it still matters: Porting examples/terminal/core/state.py would strengthen attack_orchestration and deliver operator_experience_gain.
- counterfit/core/optimize.py -> core/qsg/continuous_engine.py [disparate_mechanism_candidate, native_rewrite, QSG, score=0.617]
  Why it is non-obvious: Role alignment is stronger than lexical overlap.
  Why it still matters: Porting counterfit/core/optimize.py would strengthen optimization and deliver operator_experience_gain.
- examples/terminal/core/terminal.py -> cli/commands/thinking.py [orchestration_analogue, native_rewrite, Anvil, score=0.608]
  Why it is non-obvious: Role alignment is stronger than lexical overlap.
  Why it still matters: Porting examples/terminal/core/terminal.py would strengthen attack_orchestration and deliver operator_experience_gain.
- examples/terminal/core/state.py -> core/campaign/control_plane.py [disparate_mechanism_candidate, native_rewrite, Anvil, score=0.590]
  Why it is non-obvious: Role alignment is stronger than lexical overlap.
  Why it still matters: Porting examples/terminal/core/state.py would strengthen attack_orchestration and deliver operator_experience_gain.
- examples/terminal/core/config.py -> cli/commands/thinking.py [orchestration_analogue, native_rewrite, Anvil, score=0.586]
  Why it is non-obvious: Role alignment is stronger than lexical overlap.
  Why it still matters: Porting examples/terminal/core/config.py would strengthen attack_orchestration and deliver operator_experience_gain.
- examples/terminal/commands/show.py -> cli/commands/thinking.py [orchestration_analogue, native_rewrite, Anvil, score=0.579]
  Why it is non-obvious: Role alignment is stronger than lexical overlap.
  Why it still matters: Porting examples/terminal/commands/show.py would strengthen attack_orchestration and deliver operator_experience_gain.

### Secondary Opportunities

- examples/terminal/core/state.py -> cli/commands/thinking.py [native_rewrite, Anvil, score=0.630]
  Value: operator_experience_gain
  Rationale: Porting examples/terminal/core/state.py would strengthen attack_orchestration and deliver operator_experience_gain.
- counterfit/core/optimize.py -> core/qsg/continuous_engine.py [native_rewrite, QSG, score=0.617]
  Value: operator_experience_gain
  Rationale: Porting counterfit/core/optimize.py would strengthen optimization and deliver operator_experience_gain.
- examples/terminal/core/terminal.py -> cli/commands/thinking.py [native_rewrite, Anvil, score=0.608]
  Value: operator_experience_gain
  Rationale: Porting examples/terminal/core/terminal.py would strengthen attack_orchestration and deliver operator_experience_gain.
- examples/terminal/core/state.py -> core/campaign/control_plane.py [native_rewrite, Anvil, score=0.590]
  Value: operator_experience_gain
  Rationale: Porting examples/terminal/core/state.py would strengthen attack_orchestration and deliver operator_experience_gain.
- counterfit/core/reporting.py -> Saguaro/saguaro/analysis/report.py [native_rewrite, Saguaro, score=0.588]
  Value: reporting_gain
  Rationale: Porting counterfit/core/reporting.py would strengthen reporting and deliver reporting_gain.
- examples/terminal/core/config.py -> cli/commands/thinking.py [native_rewrite, Anvil, score=0.586]
  Value: operator_experience_gain
  Rationale: Porting examples/terminal/core/config.py would strengthen attack_orchestration and deliver operator_experience_gain.
- examples/terminal/commands/show.py -> cli/commands/thinking.py [native_rewrite, Anvil, score=0.579]
  Value: operator_experience_gain
  Rationale: Porting examples/terminal/commands/show.py would strengthen attack_orchestration and deliver operator_experience_gain.

### Feature Synthesis

- artifact_output [winner_candidate] score=0.842 counterfit/core/output.py -> core/artifacts/manager.py
- attack_orchestration [winner_candidate] score=0.802 counterfit/core/attacks.py -> core/campaign/control_plane.py
- framework_adapter [winner_candidate] score=0.680 counterfit/core/frameworks.py -> core/campaign/tooling_factory.py
- optimization [winner_candidate] score=0.617 counterfit/core/optimize.py -> core/qsg/continuous_engine.py
- reporting [winner_candidate] score=0.588 counterfit/core/reporting.py -> Saguaro/saguaro/analysis/report.py
- target_registry [winner_candidate] score=0.794 counterfit/core/targets.py -> core/campaign/repo_registry.py

### Detailed Migration Recipes
- Native Rewrite artifact_output from counterfit/core/output.py into core/artifacts/manager.py
  Target: core/artifacts/manager.py | Subsystem: Anvil | Posture: native_rewrite | Implementation tier: python
  Why port: Porting counterfit/core/output.py would strengthen artifact_output and deliver reporting_gain.
  Why here: core/artifacts/manager.py is the nearest target surface because it shares artifact, core_runtime, service. This routes into Anvil because feature families artifact_output favor Anvil; shared roles artifact, core_runtime, service align with control_plane; target path core/artifacts/manager.py sits in Anvil.
  Expected value: reporting_gain
  Verification: saguaro verify . --engines native,ruff,semantic, run affected tests
- Native Rewrite artifact_output from counterfit/reporting/image.py into core/artifacts/manager.py
  Target: core/artifacts/manager.py | Subsystem: Anvil | Posture: native_rewrite | Implementation tier: python
  Why port: Porting counterfit/reporting/image.py would strengthen artifact_output and deliver reporting_gain.
  Why here: core/artifacts/manager.py is the nearest target surface because it shares artifact, report_surface. This routes into Anvil because feature families artifact_output favor Anvil; shared roles artifact, report_surface align with control_plane; target path core/artifacts/manager.py sits in Anvil.
  Expected value: reporting_gain
  Verification: saguaro verify . --engines native,ruff,semantic, run affected tests
- Native Rewrite attack_orchestration from counterfit/core/attacks.py into core/campaign/control_plane.py
  Target: core/campaign/control_plane.py | Subsystem: Anvil | Posture: native_rewrite | Implementation tier: python
  Why port: Porting counterfit/core/attacks.py would strengthen attack_orchestration and deliver operator_experience_gain.
  Why here: core/campaign/control_plane.py is the nearest target surface because it shares core_runtime, service. This routes into Anvil because feature families attack_orchestration favor Anvil; shared roles core_runtime, service align with control_plane; target path core/campaign/control_plane.py sits in Anvil.
  Expected value: operator_experience_gain
  Verification: saguaro verify . --engines native,ruff,semantic, run affected tests
- Native Rewrite artifact_output from counterfit/reporting/tabular.py into core/artifacts/manager.py
  Target: core/artifacts/manager.py | Subsystem: Anvil | Posture: native_rewrite | Implementation tier: python
  Why port: Porting counterfit/reporting/tabular.py would strengthen artifact_output and deliver reporting_gain.
  Why here: core/artifacts/manager.py is the nearest target surface because it shares artifact, report_surface. This routes into Anvil because feature families artifact_output favor Anvil; shared roles artifact, report_surface align with control_plane; target path core/artifacts/manager.py sits in Anvil.
  Expected value: reporting_gain
  Verification: saguaro verify . --engines native,ruff,semantic, run affected tests
- Native Rewrite artifact_output from counterfit/reporting/text.py into core/artifacts/manager.py
  Target: core/artifacts/manager.py | Subsystem: Anvil | Posture: native_rewrite | Implementation tier: python
  Why port: Porting counterfit/reporting/text.py would strengthen artifact_output and deliver reporting_gain.
  Why here: core/artifacts/manager.py is the nearest target surface because it shares artifact, report_surface. This routes into Anvil because feature families artifact_output favor Anvil; shared roles artifact, report_surface align with control_plane; target path core/artifacts/manager.py sits in Anvil.
  Expected value: reporting_gain
  Verification: saguaro verify . --engines native,ruff,semantic, run affected tests
- Native Rewrite target_registry from counterfit/core/targets.py into core/campaign/repo_registry.py
  Target: core/campaign/repo_registry.py | Subsystem: Anvil | Posture: native_rewrite | Implementation tier: python
  Why port: Porting counterfit/core/targets.py would strengthen target_registry and deliver capability_gain, coverage_gain.
  Why here: core/campaign/repo_registry.py is the nearest target surface because it shares core_runtime, service. This routes into Anvil because feature families target_registry favor Anvil; shared roles core_runtime, service align with control_plane; target path core/campaign/repo_registry.py sits in Anvil.
  Expected value: capability_gain, coverage_gain
  Verification: saguaro verify . --engines native,ruff,semantic, run affected tests
- Native Rewrite target_registry from counterfit/targets/movie_reviews.py into core/campaign/repo_registry.py
  Target: core/campaign/repo_registry.py | Subsystem: Anvil | Posture: native_rewrite | Implementation tier: python
  Why port: Porting counterfit/targets/movie_reviews.py would strengthen target_registry and deliver capability_gain, coverage_gain.
  Why here: core/campaign/repo_registry.py is the nearest target surface because it shares target_registry. This routes into Anvil because feature families target_registry favor Anvil; target path core/campaign/repo_registry.py sits in Anvil.
  Expected value: capability_gain, coverage_gain
  Verification: saguaro verify . --engines native,ruff,semantic, run affected tests
- Native Rewrite target_registry from counterfit/targets/cart_pole/DCPW.py into core/campaign/repo_registry.py
  Target: core/campaign/repo_registry.py | Subsystem: Anvil | Posture: native_rewrite | Implementation tier: python
  Why port: Porting counterfit/targets/cart_pole/DCPW.py would strengthen target_registry and deliver capability_gain, coverage_gain.
  Why here: core/campaign/repo_registry.py is the nearest target surface because it shares target_registry. This routes into Anvil because feature families target_registry favor Anvil; target path core/campaign/repo_registry.py sits in Anvil.
  Expected value: capability_gain, coverage_gain
  Verification: saguaro verify . --engines native,ruff,semantic, run affected tests
- Native Rewrite attack_orchestration from counterfit/frameworks/textattack/textattack.py into core/campaign/control_plane.py
  Target: core/campaign/control_plane.py | Subsystem: Anvil | Posture: native_rewrite | Implementation tier: python
  Why port: Porting counterfit/frameworks/textattack/textattack.py would strengthen attack_orchestration and deliver operator_experience_gain.
  Why here: core/campaign/control_plane.py is the nearest target surface because it shares attack_orchestration. This routes into Anvil because feature families attack_orchestration favor Anvil; target path core/campaign/control_plane.py sits in Anvil.
  Expected value: operator_experience_gain
  Verification: saguaro verify . --engines native,ruff,semantic, run affected tests
- Native Rewrite target_registry from counterfit/targets/cart_pole/cart_pole.py into core/campaign/repo_registry.py
  Target: core/campaign/repo_registry.py | Subsystem: Anvil | Posture: native_rewrite | Implementation tier: python
  Why port: Porting counterfit/targets/cart_pole/cart_pole.py would strengthen target_registry and deliver capability_gain, coverage_gain.
  Why here: core/campaign/repo_registry.py is the nearest target surface because it shares target_registry. This routes into Anvil because feature families target_registry favor Anvil; target path core/campaign/repo_registry.py sits in Anvil.
  Expected value: capability_gain, coverage_gain
  Verification: saguaro verify . --engines native,ruff,semantic, run affected tests

### Value Realization
- capability_gain: 12
- coverage_gain: 12
- operator_experience_gain: 9
- reporting_gain: 5

### Port Ledger
- candidate reporting_analogue counterfit/core/output.py -> core/artifacts/manager.py [native_rewrite] score=0.842
- candidate reporting_analogue counterfit/reporting/image.py -> core/artifacts/manager.py [native_rewrite] score=0.820
- candidate disparate_mechanism_candidate counterfit/core/attacks.py -> core/campaign/control_plane.py [native_rewrite] score=0.802
- candidate reporting_analogue counterfit/reporting/tabular.py -> core/artifacts/manager.py [native_rewrite] score=0.802
- candidate reporting_analogue counterfit/reporting/text.py -> core/artifacts/manager.py [native_rewrite] score=0.802
- candidate disparate_mechanism_candidate counterfit/core/targets.py -> core/campaign/repo_registry.py [native_rewrite] score=0.794
- candidate feature_gap_candidate counterfit/targets/movie_reviews.py -> core/campaign/repo_registry.py [native_rewrite] score=0.730
- candidate feature_gap_candidate counterfit/targets/cart_pole/DCPW.py -> core/campaign/repo_registry.py [native_rewrite] score=0.723
- candidate feature_gap_candidate counterfit/frameworks/textattack/textattack.py -> core/campaign/control_plane.py [native_rewrite] score=0.722
- candidate feature_gap_candidate counterfit/targets/cart_pole/cart_pole.py -> core/campaign/repo_registry.py [native_rewrite] score=0.714

### Frontier Packets
- Port artifact_output from counterfit/core/output.py [priority=0.842, Anvil] tracks=comparative_spike, verification_lane
- Port artifact_output from counterfit/reporting/image.py [priority=0.820, Anvil] tracks=comparative_spike, verification_lane
- Port attack_orchestration from counterfit/core/attacks.py [priority=0.802, Anvil] tracks=comparative_spike, verification_lane
- Port artifact_output from counterfit/reporting/tabular.py [priority=0.802, Anvil] tracks=comparative_spike, verification_lane
- Port artifact_output from counterfit/reporting/text.py [priority=0.802, Anvil] tracks=comparative_spike, verification_lane

### Creation Ledger
- framework_adapter [feature_creation] priority=0.700 counterfit-main-3e8c38f5 exposes framework_adapter capability that is absent from target-4fab4804 and should be created as a native primitive.

### Low-Signal and Generic Analogues
- [0.477] reporting_analogue :: counterfit/reporting/image.py -> core/artifacts/manager.py (native_rewrite)
- [0.476] disparate_mechanism_candidate :: counterfit/core/core.py -> Saguaro/saguaro/native/runtime/arch.py (native_rewrite)
- [0.473] disparate_mechanism_candidate :: counterfit/core/logger.py -> Saguaro/saguaro/native/runtime/arch.py (native_rewrite)
- [0.470] disparate_mechanism_candidate :: counterfit/core/options.py -> Saguaro/saguaro/native/runtime/arch.py (native_rewrite)
- [0.417] orchestration_analogue :: examples/terminal/commands/load.py -> cli/commands/thinking.py (native_rewrite)
- [0.417] orchestration_analogue :: examples/terminal/commands/reload.py -> cli/commands/thinking.py (native_rewrite)
- [0.404] orchestration_analogue :: examples/terminal/commands/docs.py -> cli/commands/thinking.py (native_rewrite)
- [0.404] orchestration_analogue :: examples/terminal/commands/exit.py -> cli/commands/thinking.py (native_rewrite)
- [0.404] orchestration_analogue :: examples/terminal/commands/info.py -> cli/commands/thinking.py (native_rewrite)
- [0.404] orchestration_analogue :: examples/terminal/commands/run.py -> cli/commands/thinking.py (native_rewrite)
- [0.404] orchestration_analogue :: examples/terminal/commands/save.py -> cli/commands/thinking.py (native_rewrite)
- [0.396] orchestration_analogue :: examples/terminal/commands/interact.py -> cli/commands/thinking.py (native_rewrite)
- [0.396] orchestration_analogue :: examples/terminal/commands/new.py -> cli/commands/thinking.py (native_rewrite)
- [0.396] orchestration_analogue :: examples/terminal/commands/scan.py -> cli/commands/thinking.py (native_rewrite)
- [0.396] orchestration_analogue :: examples/terminal/commands/use.py -> cli/commands/thinking.py (native_rewrite)
- [0.388] orchestration_analogue :: examples/terminal/commands/predict.py -> cli/commands/thinking.py (native_rewrite)
- [0.383] orchestration_analogue :: examples/terminal/terminal.py -> cli/commands/thinking.py (native_rewrite)
- [0.380] orchestration_analogue :: examples/terminal/commands/list.py -> cli/commands/thinking.py (native_rewrite)
- [0.345] disparate_mechanism_candidate :: examples/terminal/core/config.py -> core/campaign/control_plane.py (native_rewrite)
- [0.342] feature_gap_candidate :: counterfit/frameworks/art/art.py -> core/campaign/tooling_factory.py (native_rewrite)

### Manual Validation Checklist
- [ ] counterfit/core/output.py -> core/artifacts/manager.py features=artifact_output
- [ ] counterfit/reporting/image.py -> core/artifacts/manager.py features=artifact_output
- [ ] counterfit/core/attacks.py -> core/campaign/control_plane.py features=attack_orchestration
- [ ] counterfit/reporting/tabular.py -> core/artifacts/manager.py features=artifact_output
- [ ] counterfit/reporting/text.py -> core/artifacts/manager.py features=artifact_output
- [ ] counterfit/core/targets.py -> core/campaign/repo_registry.py features=target_registry
- [ ] counterfit/targets/movie_reviews.py -> core/campaign/repo_registry.py features=target_registry
- [ ] counterfit/targets/cart_pole/DCPW.py -> core/campaign/repo_registry.py features=target_registry
- [ ] counterfit/frameworks/textattack/textattack.py -> core/campaign/control_plane.py features=attack_orchestration
- [ ] counterfit/targets/cart_pole/cart_pole.py -> core/campaign/repo_registry.py features=target_registry
- [ ] counterfit/targets/cart_pole/cart_pole_initstate.py -> core/campaign/repo_registry.py features=target_registry
- [ ] counterfit/targets/creditfraud.py -> core/campaign/repo_registry.py features=target_registry
- [ ] counterfit/targets/digits_keras.py -> core/campaign/repo_registry.py features=target_registry
- [ ] counterfit/targets/digits_mlp.py -> core/campaign/repo_registry.py features=target_registry
- [ ] counterfit/targets/satellite.py -> core/campaign/repo_registry.py features=target_registry
- [ ] counterfit/targets/movie_reviews/movie-reviews-scores-full.csv -> core/campaign/repo_registry.py features=target_registry
- [ ] counterfit/targets/movie_reviews/movie-reviews-scores.csv -> core/campaign/repo_registry.py features=target_registry
- [ ] counterfit/frameworks/textattack/scripts/generate_config.py -> core/campaign/control_plane.py features=attack_orchestration
- [ ] counterfit/core/frameworks.py -> core/campaign/tooling_factory.py features=framework_adapter
- [ ] examples/terminal/core/state.py -> cli/commands/thinking.py features=attack_orchestration
- [ ] counterfit/core/optimize.py -> core/qsg/continuous_engine.py features=optimization
- [ ] examples/terminal/core/terminal.py -> cli/commands/thinking.py features=attack_orchestration
- [ ] examples/terminal/core/state.py -> core/campaign/control_plane.py features=attack_orchestration
- [ ] counterfit/core/reporting.py -> Saguaro/saguaro/analysis/report.py features=reporting
- [ ] examples/terminal/core/config.py -> cli/commands/thinking.py features=attack_orchestration
- [ ] examples/terminal/commands/show.py -> cli/commands/thinking.py features=attack_orchestration

### Overlay Graph
- Nodes: 72
- Edges: 79
