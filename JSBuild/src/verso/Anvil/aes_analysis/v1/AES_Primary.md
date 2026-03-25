# The Anvil Engineering Standard

## Preamble

### Purpose and scope

The Anvil Engineering Standard (AES) is the governing engineering law for any code, configuration, data-contract, build artifact, or operational change **written, modified, reviewed, audited, or maintained** by the Anvil multi-agent coding system and its semantic intelligence layer, Saguaro. AES is designed to make high-assurance engineering practical in fast-moving research and development by enforcing **proof-oriented correctness at boundaries and invariants**, while enabling iteration velocity inside well-defined change envelopes. fileciteturn0file0 fileciteturn0file2 citeturn25search0turn25search3turn2search1

AES is prescriptive: it defines what Anvil **MUST** do to prevent defects by construction, to produce evidence of correctness, and to maintain long-term repository integrity under changing personnel, changing infrastructure, and changing scientific goals. The standard is intended to be executable as governance: rules are written so they can be checked by automation (static analysis, CI, coverage tooling, telemetry, and repository policy). citeturn22search2turn13search3turn23search0turn25search2

### Domains governed

AES governs software and engineering artifacts in these domains:

- General software engineering: Python 3.12+ and C++17 systems programming (libraries, services, tooling, CLIs, build systems).
- Deep learning and machine learning: data pipelines, training loops, evaluation, inference/serving, metric integrity, reproducibility.
- Quantum computing: circuits, variational/hybrid algorithms, noise-aware compilation, error mitigation/correction-aware workflows.
- Physics simulation: numerical methods, solvers, conservation laws, symmetry enforcement, verification of invariants.
- High-performance computing: CPU-first design, SIMD/AVX2 kernels, OpenMP/threading, memory alignment, cache discipline, deterministic performance envelopes. fileciteturn0file3 citeturn20search1turn20search0turn17search4turn17search17

### How to read AES

AES uses the normative requirement keywords defined for technical specifications: **MUST**, **MUST NOT**, **SHALL**, **SHALL NOT**, **SHOULD**, **SHOULD NOT**, **MAY**. Only uppercase usage is normative. citeturn27search0turn27search2

AES rules fall into three categories:

- **Core mandates**: apply to all code and artifacts unless explicitly scoped.
- **Assurance-scoped mandates**: requirements vary by assurance level (defined below).
- **Domain-scoped mandates**: apply when domain markers are present (ML, Quantum, Physics, HPC). fileciteturn0file3

### Authority, tailoring, and waivers

AES is the default and controlling standard. Project-local style guides MAY exist, but Anvil MUST treat AES as higher priority whenever a conflict exists. fileciteturn0file3

A rule waiver is permitted only if all of the following are true:

- The waiver is tied to a single change-set (no open-ended waivers).
- The waiver states the alternative control (how the risk is mitigated).
- The waiver is approved at the review level required by the component’s assurance level.
- The waiver is recorded in a machine-readable waiver registry and expires automatically (time-based or version-based expiry). citeturn25search2turn6search0turn5search0turn13search1

Rationale: waiver discipline prevents “permanent exception rot,” preserves traceability, and ensures risk decisions are auditable. citeturn25search3turn6search8turn13search17

## Severity and assurance framework

### Unified assurance model

AES unifies consequence-based assurance ladders into one system called **Anvil Assurance Level (AAL)**. AAL is defined by the **consequence of failure**, not by language, performance, or developer intent. This makes the assurance model compatible with safety-critical reasoning (hazard analysis, fault analysis), aviation-style verification closure (coverage and traceability), and security control baselines. fileciteturn0file0 citeturn25search1turn1search3turn5search0turn22search2

**AAL levels**

| AAL | Name | Typical consequence if faulty | Examples (non-exhaustive) |
|---|---|---|---|
| AAL-0 | Catastrophic | Silent corruption, unsafe actuation/control behavior, irrecoverable mission/experiment loss, or security boundary compromise | SIMD kernels used in production pipelines; gradient update kernels; auth/crypto; irreversible experiment execution controllers |
| AAL-1 | Critical | Wrong scientific result, wrong model outputs, numerical instability, quantum circuit miscompilation causing invalid conclusions | training loops; inference core; physics solver core; quantum circuit execution path |
| AAL-2 | Major | Localized incorrectness, partial outage, significant performance regression, incorrect configuration affecting runs | config systems; schedulers; ETL/data preprocessing; internal services |
| AAL-3 | Minor/Informational | Documentation defects, style violations, low-impact scripts; failures are nuisance-level | docs; examples; notebooks; non-critical scripts |

This table is an AES artifact derived from consequence-based assurance principles; it is intentionally conservative and designed for auditable classification. fileciteturn0file0 citeturn25search1turn1search3turn5search2turn5search3

### Classification rules

**AES-AAL-1 (mandatory classification):** Saguaro MUST tag every file and every change-set with:
- language(s) (e.g., python, cpp),
- domain marker(s) (ml, quantum, physics, hpc),
- AAL target,
- hot-path status (hot, warm, cold) derived from profiling/telemetry. fileciteturn0file3 citeturn2search0turn13search3

Rationale: automatic tagging makes enforcement feasible at repository scale and prevents “high-criticality code drifting into low-process zones.” citeturn25search3turn2search2turn6search8

**AES-AAL-2 (risk-based escalation):** Any component MUST be escalated by at least one AAL level if:
- the run is non-replayable or economically irrecoverable (e.g., expensive cluster run, limited quantum hardware slots),
- the component is part of a chain of custody or provenance boundary (build, release, signing),
- the component is part of a security boundary (authz, secrets, network ingress). citeturn6search15turn13search1turn22search2turn5search0

Rationale: irreversibility and boundary roles amplify impact beyond local defect radius. citeturn13search17turn22search12turn5search0

### Verification obligations by AAL

**AES-VRF-1 (verification is evidence):** Verification is not a test count. Verification is **evidence of claims**, and evidence MUST be linked to the claim via traceability. citeturn25search3turn2search1turn6search0

**AES-VRF-2 (traceability closure):** For AAL-0 through AAL-2, changes MUST include an updated trace chain:
Requirement → Design decision → Code → Test/Analysis → Recorded result (CI run ID, report, or signed artifact). citeturn25search3turn25search2turn13search17turn22search12

**AES-VRF-3 (structural coverage):** Structural coverage requirements MUST be satisfied as follows:
- AAL-0: 100% statement + decision coverage, and MC/DC coverage for safety- or mission-critical decision logic; uncovered or extraneous code is forbidden.
- AAL-1: 100% statement + decision coverage on changed code; MC/DC required for safety-critical decision logic and strongly recommended elsewhere.
- AAL-2: statement coverage on changed code; decision coverage required for risk-bearing branching logic.
- AAL-3: smoke tests + doc CI checks; coverage is advisory. citeturn1search3turn1search2turn6search8turn25search1

Rationale: higher consequence requires stronger structural argument that all logic has been exercised, and that no untraceable code exists. citeturn1search3turn25search3turn2search1

**AES-VRF-4 (independence):** AAL-0 components MUST receive independent review and independent verification activities (separate agent or human reviewer not involved in authoring). AAL-1 SHOULD do so when feasible. citeturn25search5turn24search7turn6search0

Rationale: independence reduces confirmation bias and is a core high-assurance assurance mechanism. citeturn25search15turn25search5turn6search0

**AES-VRF-5 (fault analysis):** AAL-0 MUST include FMEA and fault tree analysis for the change’s failure pathways; AAL-1 MUST include at least one of (FMEA, FTA) for material changes, and both for safety/security boundaries. citeturn26search0turn26search1turn26search6turn1search1

Rationale: combining inductive (FMEA) and deductive (FTA) reasoning systematically exposes failure modes and propagation chains that tests alone miss. citeturn26search1turn26search14turn26search6

### Assurance decision tree

AES requires a deterministic classification path.

**AES-AAL-DEC-1:** If a change can cause silent scientific corruption, security compromise, or irrecoverable run loss → classify as AAL-0.

**AES-AAL-DEC-2:** Else if a change can materially change model output correctness, circuit correctness, or simulation stability → AAL-1.

**AES-AAL-DEC-3:** Else if a change can degrade performance, reliability, or config correctness but is recoverable → AAL-2.

**AES-AAL-DEC-4:** Else documentation/examples/non-critical scripts → AAL-3. fileciteturn0file0 citeturn25search1turn5search0turn2search0

Rationale: deterministic classification prevents under-classification and enables consistent automated enforcement. citeturn6search0turn25search6turn13search3

## Architecture and universal engineering mandates

This section defines rules that apply across languages and domains, with emphasis on software architecture as a first-class artifact.

### Architectural governance

**AES-ARC-1 (architecture is mandatory):** Any repository with AAL-0 or AAL-1 code MUST maintain an architecture description that includes:
- component boundaries and responsibilities,
- dependency direction rules (allowed edges),
- data contracts and stability policy,
- concurrency model and determinism policy,
- failure domains and recovery strategy,
- observability plan (signals, metrics, logs, traces). citeturn6search1turn5search3turn13search3turn2search0

Rationale: architecture is the only scalable way to prevent “accidental complexity” and cross-component coupling in large repositories. citeturn6search1turn5search7turn15search4

**AES-ARC-2 (component interfaces):** Component boundaries MUST be expressed as explicit interfaces:
- stable types/contracts,
- versioning policy,
- error contract,
- performance envelope (latency and memory bound),
- security assumptions. citeturn16search1turn5search0turn15search7turn2search20

Rationale: in systems with many consumers, every observable behavior becomes depended upon; interfaces must be explicit to prevent accidental contracts. citeturn15search7turn16search1

**AES-ARC-3 (compatibility policy):** AES adopts a “strict send, strict receive” policy at safety/security boundaries and a “strict send, tolerant receive” policy only for explicitly versioned and fuzz-tested protocol interfaces. citeturn16search6turn16search12turn23search2turn5search0

Rationale: tolerant parsing without controls causes long-term interoperability and security failures; tolerance must be paired with explicit versioning and adversarial testing. citeturn16search3turn16search12turn4search7

### Traceability and lifecycle evidence

**AES-TRC-1 (no orphan code):** Every code artifact MUST trace back to at least one requirement or a declared infrastructure purpose. Orphan code MUST NOT exist in AAL-0/AAL-1. citeturn25search3turn1search3turn25search2

**AES-TRC-2 (bidirectional traceability):** For AAL-0 through AAL-2, traceability MUST be bidirectional:
- From each requirement to code and verification evidence.
- From every code path and test back to an owning requirement or infrastructure declaration. citeturn25search3turn25search19turn1search3

Rationale: bidirectional traceability ensures “only what is required is built,” eliminates surplus behavior, and reduces misinterpretation during refinement. citeturn25search3turn25search19turn1search3

**AES-TRC-3 (machine-readable traces):** Traces MUST be machine-readable (manifest, IDs in code comments, or structured metadata). Human-only trace narratives are insufficient for AAL-1+. citeturn25search2turn13search3turn6search0

### Complexity and performance discipline

**AES-CPLX-1 (complexity budgets):** Each function in AAL-0/AAL-1 MUST stay below:
- cyclomatic complexity ≤ 10, except with formal justification and expanded tests;
- ≤ 60 executable lines per function (excluding doc/comments), except with design-review waiver. citeturn6search0turn15search4turn2search2

Rationale: high complexity defeats coverage closure, increases defect density, and blocks effective review. citeturn2search2turn6search0turn15search4

**AES-CPLX-2 (algorithmic complexity in hot paths):** For any code where telemetry shows ≥ 5% contribution to wall-time or CPU, the dominant time complexity MUST be documented, and average complexity MUST NOT exceed O(n) unless:
- a scaling plot demonstrates safety margin,
- a design review approves the tradeoff,
- a benchmark gate enforces the bound. citeturn2search0turn2search4turn17search12turn15search4

Rationale: performance regressions are reliability failures at scale; they must be controlled by evidence, not intuition. citeturn2search0turn2search20turn17search4

### Error-handling philosophy

AES combines three validated practices into one policy:

1) errors are first-class values and must be handled explicitly,  
2) processes may fail fast at well-defined crash boundaries with supervision/restart,  
3) masking symptoms is prohibited in high-assurance code. citeturn24search1turn24search2turn2search1

**AES-ERR-1 (explicit error contracts):** Public APIs MUST declare an error contract:
- success output types,
- error types/codes,
- what is retryable vs terminal,
- which invariants are enforced and how violations surface. citeturn24search1turn16search2turn8search10turn7search3

**AES-ERR-2 (no silent failure):** Catch-all swallowing (bare `except`, `catch(...)`, ignoring error returns) MUST NOT appear in AAL-0/AAL-1 and SHOULD NOT appear anywhere. citeturn24search1turn8search10turn4search9

**AES-ERR-3 (no exceptions for control flow):** Exceptions MAY be used for exceptional conditions and invariant violations, but MUST NOT be used as ordinary control flow in hot paths. citeturn7search3turn8search0turn24search2

**AES-ERR-4 (crash domains):** “Let it crash” is permitted only at explicitly declared crash boundaries where:
- state is transactional or reconstructable,
- a supervisor/restart mechanism exists,
- crash events are observable and alertable,
- the boundary is tested with fault injection. citeturn24search2turn13search3turn2search0turn23search2

Rationale: explicit error values prevent ambiguity; controlled crash domains eliminate error masking while preserving simplicity and recoverability. citeturn24search1turn24search2turn2search1

### Documentation and knowledge architecture

**AES-DOC-1 (Diátaxis discipline):** Documentation MUST be organized into four distinct forms:
- Tutorials (learning-oriented),
- How-to guides (task-oriented),
- Reference (complete contracts),
- Explanations (conceptual/architectural rationale). citeturn13search0turn13search4turn7search0

**AES-DOC-2 (API contracts):** Public APIs MUST be callable from docs alone (docstrings/comments must specify: purpose, inputs, outputs, error behavior, side-effects, complexity, and determinism policy). citeturn7search3turn6search0turn13search12

**AES-DOC-3 (doc-code sync):** Any AAL-1+ change that alters externally observable behavior MUST update the corresponding reference docs in the same change-set unless the docs are generated from contracts. citeturn13search12turn15search2turn25search3

Rationale: separating doc intents reduces ambiguity and makes documentation maintainable under frequent change. citeturn13search0turn13search12

### Security baseline and supply chain integrity

**AES-SEC-1 (security verification levels):** Any network-reachable or user-reachable component MUST meet an application security verification baseline, with stricter requirements for higher-risk components. citeturn4search2turn5search0turn22search2

**AES-SEC-2 (CWE prevention):** AAL-1+ code MUST explicitly mitigate the dominant weakness classes in the current high-risk weakness set relevant to its platform (injection, out-of-bounds, missing authorization, deserialization, SSRF, etc.). citeturn4search7turn4search3turn5search0

**AES-SEC-3 (dependency pinning and hashes):** Dependencies MUST be pinned and integrity-checked:
- Python packages: use hash-checking mode with pinned versions and expected hashes for production builds.
- C/C++ deps: vendored with verified provenance or locked via toolchain lockfile and cryptographic verification. citeturn22search0turn13search17turn13search1turn5search0

**AES-SEC-4 (provenance and signing):** AAL-0/AAL-1 release artifacts MUST include supply-chain provenance and signature evidence (signing + transparency log inclusion) and SHOULD include SBOMs. citeturn13search1turn13search10turn22search12turn22search6

**AES-SEC-5 (secure development lifecycle):** Secure development practices MUST include threat modeling for boundaries, code review for security controls, and automated scanning for known vulnerability patterns. citeturn22search2turn5search0turn4search6

Rationale: modern systems fail through supply chain compromise and common weakness classes; integrity controls must be preventative and auditable. citeturn22search2turn13search5turn4search7turn5search0

### Concurrency safety and determinism

**AES-CONC-1 (no data races):** Any shared mutable state MUST be:
- protected by synchronization, or
- made atomic, or
- removed by design (immutability, message passing, process isolation). citeturn8search9turn8search6turn23search21turn6search1

**AES-CONC-2 (document lock discipline):** AAL-1+ code MUST document:
- which locks guard which data,
- lock acquisition ordering,
- thread ownership or thread-local guarantees. citeturn6search1turn8search2turn23search21

**AES-CONC-3 (deterministic execution policy):** Systems that claim deterministic behavior MUST:
- fix sources of nondeterminism (unordered iteration, non-seeded RNG, race-dependent reductions),
- pin versions of numerical libraries,
- record determinism parameters (thread count, compiler flags, hardware assumptions). citeturn6search1turn11search4turn17search4turn20search1

Rationale: data races are undefined behavior in C/C++ and produce untestable failures; determinism is essential for debugging, reproducibility, and safety arguments. citeturn8search9turn8search6turn6search5turn17search4

## Language standards

AES is intentionally language-selective: it requires choosing the right tool for the job, not defaulting to a single language.

### Language selection rubric

**AES-LANG-1 (choose language by constraints):** Anvil MUST choose implementation language based on:
- performance envelope (latency/throughput),
- memory-layout control needs,
- safety/security risk,
- required ecosystem maturity,
- integration surface (bindings, deployment). citeturn2search0turn8search0turn20search15turn17search1

**AES-LANG-2 (C++ as the performance and correctness core):** For hot paths, kernels, and invariant-critical systems components, C++17 SHOULD be the default unless a justified alternative provides equal control and verification surface. Python SHOULD serve as orchestration and wrapper layer, not as the computational core for AAL-0 hot paths. citeturn17search17turn20search0turn20search2turn11search8

Rationale: CPU performance discipline, memory control, and UB avoidance are not reliably achievable in dynamic languages for AAL-0 hot paths; wrappers enable usability without sacrificing kernel integrity. citeturn17search17turn8search11turn20search2turn20search15

### Python 3.12+ standard

#### Style and structure

**AES-PY-1 (PEP 8 layout):** Python code MUST follow canonical style conventions and consistent formatting. citeturn7search0turn20search1

**AES-PY-2 (imports):** Imports MUST be explicit and ordered (stdlib → third-party → local). Wildcard imports MUST NOT be used. citeturn7search0turn7search3

#### Typing and contracts

**AES-PY-3 (type annotations required):** All AAL-1+ Python MUST use type annotations for public functions, classes, and module-level variables. Anvil MUST maintain type-checking in CI for this code. citeturn7search1turn7search2turn7search5

**AES-PY-4 (structured data contracts):** Structured inputs/outputs MUST use typed data models (dataclasses/pydantic-equivalent) rather than untyped dictionaries as primary contracts, except at raw I/O boundaries. citeturn7search1turn16search1

Rationale: static typing enables scalable refactoring, reduces interface ambiguity, and supports automated verification in large repositories. citeturn7search1turn15search2

#### Safety and correctness

**AES-PY-5 (resource safety):** Resources (files, locks, sockets) MUST be managed with context managers. citeturn7search3turn20search1

**AES-PY-6 (no dynamic execution):** `eval`/`exec` MUST NOT be used in AAL-0/AAL-1 and SHOULD NOT be used elsewhere except inside sandboxed tooling with explicit input controls. citeturn4search7turn5search0

Example: explicit contracts and validation at a boundary

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

@dataclass(frozen=True)
class Batch:
    logits: "torch.Tensor"   # shape: (B, C), dtype float32/float16
    targets: "torch.Tensor"  # shape: (B,), dtype int64

def validate_batch(batch: Batch) -> None:
    if batch.logits.ndim != 2:
        raise ValueError(f"logits must be rank-2 (B,C); got {batch.logits.shape}")
    if batch.targets.ndim != 1:
        raise ValueError(f"targets must be rank-1 (B,); got {batch.targets.shape}")
    if batch.logits.shape[0] != batch.targets.shape[0]:
        raise ValueError("Batch size mismatch between logits and targets")
```

Rationale: boundary validation prevents silent shape/type corruption propagating into training or inference. citeturn25search1turn4search7turn12search1

### C and C++17 standard

#### Undefined behavior elimination

**AES-CPP-1 (UB is forbidden):** C/C++ code in AAL-0/AAL-1 MUST NOT contain constructs that can trigger undefined behavior (out-of-bounds, data races, signed overflow, invalid aliasing, misaligned access). citeturn8search6turn8search9turn23search0turn4search5

**AES-CPP-2 (sanitizers required):** Debug/CI builds for AAL-1+ C/C++ MUST run with sanitizers appropriate to the component (ASan/UBSan; TSan where concurrency exists) and MUST treat sanitizer findings as defects. citeturn23search8turn23search0turn23search21

Rationale: UB can make bugs invisible to tests because compilers optimize assuming UB never happens; sanitizer instrumentation makes whole classes of defects observable. citeturn8search6turn23search0turn4search5

#### Resource management

**AES-CPP-3 (RAII everywhere):** Owning resources MUST be managed via RAII; raw `new/delete` MUST NOT appear in application code. citeturn8search0turn8search4turn8search19

**AES-CPP-4 (ownership clarity):** API signatures MUST encode ownership:
- owning types: `unique_ptr` or explicit owner wrapper,
- non-owning: references or non-owning pointers with documented lifetime,
- shared ownership is permitted only with explicit justification. citeturn8search4turn8search11

#### Type safety and API correctness

**AES-CPP-5 (casts):** C-style casts MUST NOT be used. Narrowing conversions MUST be explicit and justified. citeturn8search0turn8search6turn4search5

**AES-CPP-6 (`[[nodiscard]]` for defect-prone returns):** Functions returning status/error-bearing values MUST be `[[nodiscard]]` or enforce checked usage via types. citeturn8search10turn8search1

#### Include discipline and exceptions

**AES-CPP-7 (include order):** Includes MUST follow a deterministic order to avoid hidden dependencies and build instability. citeturn8search1

**AES-CPP-8 (exceptions policy):** Exceptions are permitted but MUST follow one consistent policy per library:
- either “exceptions enabled with RAII and invariants,” or
- “exceptions disabled; explicit status/expected types.” citeturn8search0turn8search1turn24search1

Rationale: mixed exception policies across boundaries cause undefined termination and untestable error paths. citeturn8search0turn8search1

### Quantum circuit coding standard

AES treats quantum programs as **hardware-attached programs**: circuit correctness includes transpilation, mapping, and noise model assumptions.

**AES-QC-1 (target-aware design):** Circuits MUST be designed with an explicit target backend (or simulator) and compilation constraints. Running an “ideal-only” circuit on hardware without transpilation/mapping is a defect. citeturn9search0turn9search4turn9search7

**AES-QC-2 (named parameters):** Variational and parameterized circuits MUST use named parameters; magic-number angles MUST NOT appear in AAL-1+ circuits. citeturn9search10turn9search2

**AES-QC-3 (encoding contract):** Hybrid classical–quantum interfaces MUST document:
- encoding scheme,
- normalization constraints,
- measurement mapping and aggregation,
- shot budget and statistical confidence policy. citeturn9search10turn9search1turn0search1

**AES-QC-4 (transpilation is part of the artifact):** Every runnable circuit artifact MUST include:
- original circuit,
- transpiled circuit,
- transpiler settings/pass pipeline,
- mapping (logical → physical qubits),
- basis gate set,
- noise model if simulated. citeturn9search0turn9search3turn9search1

Rationale: transpilation changes circuit depth, routing, and gate fidelity; without recording it, experiments are non-reproducible and conclusions are not defensible. citeturn9search0turn9search4turn0search17

Example: parameterized circuit (Cirq-style)

```python
import cirq
import sympy as sp

q = cirq.LineQubit.range(2)
theta = sp.Symbol("theta_layer1")

circuit = cirq.Circuit(
    cirq.H(q[0]),
    cirq.CNOT(q[0], q[1]),
    cirq.rx(theta)(q[0]),
    cirq.measure(*q, key="m"),
)
```

## Domain standards

### Deep learning and machine learning

AES treats ML correctness as **a pipeline of invariants**: data → preprocessing → model → training step → optimizer update → artifact → evaluation → deployment. Failures anywhere can silently poison results.

#### Training loop integrity

**AES-ML-1 (step invariants):** Every training step in AAL-0/AAL-1 MUST enforce:
- finite loss (no NaN/Inf),
- finite gradients (no NaN/Inf),
- bounded gradient norms (configurable thresholds),
- optimizer state validity (finite moments, non-negative variances),
- deterministic logging of step metadata. citeturn11search3turn0search7turn12search2turn2search0

Rationale: mixed precision and large-scale training frequently fail via non-finite gradients and numerical issues; detecting these at the step boundary prevents silent model corruption. citeturn11search3turn11search8turn0search7

Example: minimal gradient health gate (framework-agnostic pseudocode)

```python
def training_step(batch) -> float:
    loss = forward(batch)
    assert is_finite(loss), "loss is non-finite"
    grads = backward(loss)
    assert all_finite(grads), "non-finite gradients"
    assert grad_norm(grads) <= max_norm, "gradient explosion"
    optimizer.step(grads)
    return float(loss)
```

#### Mixed-precision and numerical stability

**AES-ML-2 (mixed precision requires stability controls):** If FP16/BF16 is used:
- maintain FP32 master weights OR an equivalent stability mechanism,
- apply loss scaling (static or dynamic),
- ensure accumulation operations occur in sufficient precision (often FP32),
- gate updates on finiteness checks. citeturn11search3turn11search8turn0search7turn11search3

Rationale: half precision has limited range; loss scaling and FP32 accumulation are established techniques to preserve gradient information and prevent underflow/overflow. citeturn11search3turn11search8turn11search3

#### Gradient verification and correctness auditing

**AES-ML-3 (gradient audits):** AAL-0/AAL-1 training code MUST include at least one gradient verification mode:
- finite-difference checks on small synthetic cases,
- analytic vs numerical gradient comparison for custom ops,
- invariant checks on gradient flow (no disconnected params, no unexpected zero grads). citeturn0search8turn0search16turn0search12

Rationale: gradient computation bugs can “train” while optimizing the wrong objective; numerical checks provide an independent signal of correctness. citeturn0search8turn0search16

#### Data pipeline validation

**AES-ML-4 (data is code):** Training and serving data MUST be tested like code:
- schema validation,
- distribution drift checks,
- training-serving skew checks,
- slice-based metric evaluation for critical cohorts. citeturn12search3turn12search0turn12search1

Rationale: production ML failures often arise from data shift and pipeline skew rather than model code; mature ML practice treats data contracts as first-class. citeturn12search1turn12search2turn12search3

#### Reproducibility and artifact integrity

**AES-ML-5 (reproducible runs):** AAL-1+ training MUST record:
- code commit hash,
- dependency lock hashes,
- random seeds and determinism settings,
- dataset version + data hash,
- hardware platform summary,
- evaluation metrics and thresholds. citeturn13search17turn22search0turn2search0turn12search2

Rationale: without complete provenance, model results cannot be audited, reproduced, or trusted under iteration. citeturn13search17turn22search12turn12search2

### Quantum computing

AES treats quantum workflows as **probabilistic experiments under noise**, requiring explicit statistical discipline and recording of compilation states.

#### Noise-aware programming

**AES-Q-1 (noise model required for claims):** Any experimental claim about circuit performance MUST specify:
- device/backend identity,
- calibration epoch if hardware,
- noise model if simulated,
- shot count and confidence interpretation. citeturn9search1turn9search0turn0search17

Rationale: quantum results are sensitive to noise and compilation; without these details, results are not comparable or reproducible. citeturn9search1turn9search4turn0search17

#### Classical–quantum interfaces

**AES-Q-2 (interface normalization):** Classical→quantum encodings MUST assert normalization and bounds; quantum→classical decoding MUST specify estimator bias/variance expectations and aggregation (mean, median, trimming). citeturn9search10turn9search1turn0search15

Rationale: encoding/decoding is the dominant source of silent bugs in hybrid workflows and drives both correctness and statistical efficiency. citeturn0search15turn9search10

#### Error correction and mitigation posture

**AES-Q-3 (QEC awareness):** When relevant, code MUST distinguish:
- error mitigation techniques (noise model-based, post-processing),
- error correction codes (stabilizers, syndrome extraction, decoding),
- assumptions about logical error rate and overhead. citeturn0search1turn0search5turn0search13

Rationale: mitigation and correction have different guarantees; conflating them leads to invalid system-level planning and conclusions. citeturn0search1turn0search13

### Physics simulation

AES treats simulation correctness as **structure preservation**: invariants, conservation laws, and symmetry constraints must be enforced and verified.

#### Conservation laws and discretization integrity

**AES-PHY-1 (conservation claims require discrete checks):** If the modeled system conserves mass/momentum/energy (or other invariants), the simulation MUST:
- select a numerical method aligned with the conservation form,
- include discrete conservation checks (per step and over horizon),
- quantify drift and define acceptable thresholds. citeturn19search4turn19search13turn18search0

Rationale: conservation is often lost at the discretization level; enforcing discrete conservation prevents long-run divergence and unphysical artifacts. citeturn19search13turn18search2turn18search0

#### Symmetry enforcement

**AES-PHY-2 (symmetry is a contract):** If the underlying physics is invariant under a symmetry group (translation, rotation, gauge-like invariance), the code MUST:
- encode symmetry explicitly (by formulation or constraints),
- test symmetry by transforming initial conditions and comparing conserved quantities,
- document which symmetries are preserved vs broken by approximation. citeturn18search1turn18search0turn18search2

Rationale: symmetries correspond to conserved quantities; breaking them often breaks the scientific meaning of the simulation. citeturn18search1turn18search0

#### Numerical stability and round-off control

**AES-PHY-3 (floating-point discipline):** Simulation code MUST:
- treat IEEE-754 special values (NaN/Inf) as defect indicators unless explicitly modeled,
- use numerically stable formulations for sums and reductions in sensitive computations,
- avoid catastrophic cancellation when possible (reformulate or use compensated methods). citeturn11search4turn10search8turn11search0turn10search4

Rationale: floating-point round-off is a primary cause of silent drift and instability in long-running simulations. citeturn11search0turn11search10turn11search4

Example: Kahan compensated summation (C++17)

```cpp
double kahan_sum(const double* x, size_t n) {
    double sum = 0.0;
    double c = 0.0; // compensation
    for (size_t i = 0; i < n; ++i) {
        double y = x[i] - c;
        double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    return sum;
}
```

### High-performance computing

AES treats performance as a reliability constraint: predictable performance prevents overload, queue waste, and unstable systems.

#### CPU-first and SIMD-first rules

**AES-HPC-1 (CPU-first by default):** Hot paths MUST be optimized for CPU-first execution unless a GPU/accelerator path is explicitly required and justified. CPU baselines MUST exist even when accelerators are used. fileciteturn0file0 citeturn2search0turn17search1turn17search17

Rationale: CPU-first baselines preserve portability, provide a correctness oracle, and reduce dependency on specialized hardware availability. citeturn17search1turn20search15

**AES-HPC-2 (SIMD discipline):** For vectorizable kernels, code MUST:
- use SoA or vector-friendly layout when beneficial,
- align data and document alignment assumptions,
- avoid hidden aliasing preventing vectorization,
- record SIMD utilization metrics. citeturn17search17turn17search9turn17search8turn17search4

Rationale: cache and vector unit efficiency are dominant on modern CPUs; alignment and layout control are prerequisite for stable performance. citeturn17search17turn17search9turn17search4

Example: OpenMP SIMD alignment declaration

```cpp
#pragma omp simd aligned(x, y : 32)
for (int i = 0; i < n; ++i) {
    y[i] = a * x[i] + y[i];
}
```

#### Memory alignment and cache discipline

**AES-HPC-3 (cache-aware layout):** AAL-0/AAL-1 HPC code MUST document:
- cache-line assumptions where relevant,
- avoidance of false sharing (padding or partitioning),
- contiguous access patterns in hot loops. citeturn17search3turn17search17turn2search0

Rationale: cache-line contention and false sharing can destroy throughput and cause saturation failures; explicit discipline keeps performance predictable. citeturn17search3turn2search0

#### Threading rules

**AES-HPC-4 (thread safety):** Multi-threaded code MUST pass:
- correctness tests under thread count variation,
- race detection where feasible,
- determinism checks when determinism is required. citeturn23search21turn8search9turn17search4turn6search1

Rationale: concurrency defects are often workload-dependent and do not reproduce consistently without dedicated tooling. citeturn23search21turn8search9turn23search0

## Verification, testing, and quality evidence

### Test taxonomy

AES defines required test classes; the applicable set depends on AAL.

- **Unit tests**: smallest isolated behaviors; deterministic; high signal.
- **Integration tests**: component composition; contracts; serialization; boundary behavior.
- **Property-based tests**: invariants across broad input domains.
- **Fuzz tests**: adversarial input generation for parsers/interfaces.
- **Benchmark tests**: performance envelopes; regression gates.
- **Statistical tests**: reliability certification for probabilistic behavior (ML, Monte Carlo, noisy quantum). citeturn23search3turn23search2turn2search1turn12search2

Rationale: modern defect profiles include adversarial inputs, integration boundary mismatches, and performance regressions; a single test type cannot cover all. citeturn23search2turn2search0turn12search2

### Coverage and structural verification

**AES-TST-1 (coverage as a gate):** Coverage is a merge gate for AAL-0/AAL-1; uncovered logic must be explained, proven unreachable, or removed. citeturn1search3turn6search8turn15search2

**AES-TST-2 (no extraneous code):** Any code not traceable to a requirement is considered extraneous and MUST be removed or explicitly deactivated with justification and verification. citeturn1search3turn25search3turn25search19

Rationale: untraceable code expands attack surface, hides defects, and violates high-assurance completion criteria. citeturn1search3turn25search3turn5search0

### Property-based testing requirements

**AES-PBT-1:** AAL-1+ components with mathematical invariants (numerical kernels, parsers, transformations) MUST include property-based tests for core invariants. citeturn23search3turn19search4turn18search0

Rationale: property-based tests discover edge cases that example-based tests systematically miss. citeturn23search3turn2search1

### Fuzzing requirements

**AES-FUZZ-1:** Any AAL-1+ component that parses, deserializes, or ingests untrusted or semi-trusted input MUST have fuzz tests and MUST treat fuzz findings as defects. citeturn23search2turn4search7turn5search0

Rationale: fuzzing is a proven method to uncover memory safety and parser logic defects with low marginal cost once integrated. citeturn23search2turn23search8turn4search9

### ML-specific testing framework

AES adopts a four-bucket view of ML system readiness:

- data/feature tests,
- model correctness tests,
- infrastructure tests,
- monitoring tests. citeturn12search2turn12search0

**AES-ML-TST-1 (data tests):** Schema, anomalies, drift, skew. citeturn12search3turn12search1

**AES-ML-TST-2 (model tests):** slice metrics, bias checks where relevant, regression vs golden checkpoints. citeturn12search0turn12search2

**AES-ML-TST-3 (infra tests):** reproducible builds, deterministic evaluation harness, rollback support. citeturn12search2turn13search17turn22search2

### Quantum-specific verification

**AES-Q-TST-1 (simulator validation):** Any hardware-bound circuit MUST have a simulator-equivalent validation harness, including:
- ideal simulation for functional structure,
- noisy simulation aligned to the target noise model when making performance claims. citeturn9search1turn9search5turn9search4

Rationale: simulators provide controlled conditions to distinguish algorithmic defects from hardware noise artifacts. citeturn9search1turn9search5

## Observability, telemetry, and reliability operations

AES treats observability as mandatory evidence infrastructure.

### Golden signals

AES adopts four universal “golden signals” and adapts them per domain:

- latency,
- throughput,
- errors,
- saturation. citeturn2search0turn2search4

**AES-OBS-1 (mandatory golden signals):** Every AAL-1+ service/job MUST emit golden signals with consistent tags including component, version, run_id, and AAL. citeturn13search3turn2search0turn2search20

### Domain signal mapping

- ML training: step_time (p50/p95/p99), tokens/sec or samples/sec, non-finite counts, gradient norm outliers, GPU/CPU utilization, memory RSS.
- HPC kernels: cycles/element, GFLOPS, cache miss rate proxies, SIMD lane utilization, thread saturation.
- Quantum: compile time, circuit depth, transpilation delta, shot throughput, error counts, calibration epoch, noise model version.
- Physics: energy drift, invariant residuals, solver iteration counts, stability flags. citeturn2search0turn9search4turn18search2turn11search4

### Chronicle protocol

**AES-CHR-1 (baseline → change → result):** Any potential performance- or correctness-impacting change in AAL-0/AAL-1 hot paths MUST follow:
1) Baseline: record pre-change golden signals,
2) Change: implement and test,
3) Result: record post-change signals,
4) Attach: bind the metrics delta to the change record. fileciteturn0file0 citeturn2search0turn15search2turn13search3

Rationale: performance and reliability must be controlled quantitatively; baselines prevent “silent regression” and support continuous improvement. citeturn2search2turn2search0turn14search0

### Structured logging and tracing

**AES-OBS-2 (structured logs):** Logs MUST be structured and correlated to trace/span context where applicable. citeturn13search3turn13search7turn13search19

**AES-OBS-3 (telemetry standards):** Implementations SHOULD use a unified telemetry framework to emit traces, metrics, and logs with consistent resource tags. citeturn13search3turn13search11

Rationale: correlated telemetry reduces mean time to detect and diagnose failures and is required for disciplined SLO/error budget operations. citeturn2search20turn13search19

### SLOs and error budgets

**AES-SRE-1:** Any user-facing or pipeline-critical component MUST define SLOs and operate with error budgets; repeated budget burns MUST trigger toil reduction or reliability work. citeturn2search20turn2search0turn2search8

Rationale: SLOs align engineering effort with user impact and prevent reliability collapse under growth. citeturn2search20turn2search0

## Governance, review, anti-patterns, and appendices

### AI agent workflow governance

AES assumes Anvil can generate code; therefore, governance must constrain generation.

**AES-AG-1 (self-verification is mandatory):** Before proposing a change, Anvil MUST:
- classify AAL,
- list affected invariants,
- generate tests for new/changed behavior,
- run static analysis and relevant linters,
- produce a trace entry linking requirement → change → evidence. fileciteturn0file2 citeturn15search2turn25search3turn23search0

**AES-AG-2 (red-team protocol):** For AAL-0/AAL-1, Anvil MUST invoke an internal red-team pass that attempts to:
- find invariants violations,
- identify top failure modes (FMEA),
- construct at least one fault tree path to catastrophic outcome,
- map relevant security weakness classes,
- confirm regression gates, coverage closure, and provenance controls. fileciteturn0file2 citeturn26search0turn26search6turn4search7turn1search3

Rationale: autonomous generation increases throughput of defects unless paired with adversarial self-audit; structured red-teaming catches systematic blind spots. citeturn2search1turn15search4turn4search7

### Code review and change management

**AES-REV-1 (review scope):** Review MUST prioritize:
1) architecture and design correctness,
2) safety/security invariants,
3) correctness and tests,
4) readability and maintainability. citeturn15search4turn15search2

**AES-REV-2 (review standard):** Code review MUST ensure repository health improves over time. Reviewers MUST block merges that introduce new technical debt in AAL-0/AAL-1 unless explicitly accepted with a remediation plan. citeturn15search2turn14search1turn14search0

**AES-REV-3 (comment protocol):** Reviews SHOULD use a structured comment format that distinguishes blocking vs non-blocking feedback to reduce churn. citeturn15search1turn15search5

Rationale: consistent review discipline is the primary defense against systemic quality decay in large codebases. citeturn15search2turn14search0

### Anti-patterns registry

The following patterns are explicitly forbidden in AAL-0/AAL-1 unless waived with mitigations.

#### Universal anti-patterns

- Silent exception swallowing (bare catch/except). citeturn4search9turn24search1  
- Untested error paths. citeturn1search3turn2search1  
- Untraceable code (“mystery features”). citeturn25search3turn1search3  
- Dependency upgrades without lock+hash+regression gate. citeturn22search0turn13search17  
- Ad-hoc “fix by clamp” masking numerical failures rather than addressing root cause. citeturn2search1turn11search0  

#### ML anti-patterns

- Training without non-finite gradient checks in mixed precision. citeturn11search3turn0search7  
- Using unstable formulations (`exp` directly on large logits) in loss code. citeturn10search7turn11search0  

#### Quantum anti-patterns

- Running circuits on hardware without recorded transpilation/mapping state. citeturn9search0turn9search4  
- Magic-number angles in variational layers without naming and documentation. citeturn9search10turn9search2  

#### HPC anti-patterns

- Shared mutable arrays updated by multiple threads without clear partitioning or atomics (data races). citeturn8search9turn23search21  
- “Optimizations” using UB (e.g., signed overflow assumptions, aliasing violations). citeturn8search6turn23search0turn4search5  

Rationale: these patterns are historically correlated with high-severity failures and are inconsistent with high-assurance evidence closure. citeturn4search7turn1search3turn23search0turn25search3

### Appendices

#### Quick-reference checklists (AAL-1+)

Change-set checklist:

- AAL classification recorded. citeturn25search6turn6search0  
- Requirement/design links updated and machine-readable. citeturn25search2turn25search3  
- Tests added/updated; coverage gate satisfied. citeturn1search3turn6search8  
- Security controls reviewed for relevant weakness classes. citeturn4search7turn5search0  
- Chronicle protocol metrics attached for hot paths. citeturn2search0turn13search3  
- Provenance/signing/SBOM updated for release artifacts. citeturn13search1turn22search6turn22search12  

#### Severity decision cheat-sheet

If it can silently corrupt results, breach security boundaries, or waste irrecoverable compute → AAL-0. citeturn25search1turn5search0turn2search0

If it can produce wrong scientific output or unstable systems but is recoverable → AAL-1. citeturn12search2turn9search4turn18search2

If it is operational/config/performance but localized → AAL-2. citeturn2search0turn5search0

Docs/examples → AAL-3. citeturn13search0turn7search0

#### Glossary of synthesized terms

- **AAL**: consequence-based assurance level used by AES.
- **Crash boundary**: an explicitly declared boundary where fail-fast termination is permitted because restart and state recovery are designed and tested.
- **Chronicle protocol**: baseline → change → result telemetry binding for correctness/performance evidence.
- **Extraneous code**: code not traceable to a requirement (including dead code); forbidden in high-assurance components. citeturn1search3turn25search3  
- **Golden signals**: latency, throughput, errors, saturation monitoring primitives. citeturn2search0  
- **Noise-aware compilation**: transpilation, routing, and optimization using explicit backend constraints and noise assumptions. citeturn9search0turn9search1  
- **SBOM**: software bill of materials representing component composition and provenance. citeturn22search6turn22search10

