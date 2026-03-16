# The Anvil Engineering Standard

Version: AES v2 (Anvil-Saguaro Edition). This standard is a unified, prescriptive rulebook for AI-assisted and human-assisted systems engineering across general programming, deep learning, quantum computing, physics simulation, and high-performance computing. fileciteturn0file0

## Preamble

### Purpose and scope

The Anvil Engineering Standard (AES) defines **mandatory** engineering rules for designing, implementing, verifying, operating, and evolving software created or modified by the Anvil coding agent and its human collaborators. These rules are written to produce **high-assurance, evidence-backed** code and architectures while remaining usable in fast-moving research and product work. fileciteturn0file0 fileciteturn0file2 citeturn0search0turn2search42turn11search4turn13search4

AES applies to:
- new projects and new modules,
- changes to existing repositories,
- emergency fixes (with tightened post-facto evidence requirements),
- code generation, refactoring, and audits performed by Anvil and Saguaro-assisted workflows. citeturn13search4turn16search1turn19search1

### Domains governed

AES governs (and explicitly couples) these domains:

- **General Software Engineering** (Python 3.12+, C++17+, systems programming, secure coding, lifecycle processes). fileciteturn0file4turn0file5  
- **Deep Learning / Machine Learning** (training loops, gradient integrity, numerical stability, data/model validation, monitoring). citeturn6search1turn24search3turn24search45  
- **Quantum Computing** (circuit construction, compilation/transpilation discipline, noise-aware workflows, error mitigation and error correction concepts, hybrid classical–quantum interfaces). fileciteturn0file1 citeturn7search2turn7search7turn8search0turn7search0  
- **Physics Simulation** (numerical methods, invariants, conservation laws, symmetry enforcement, validation against physics). citeturn4search2turn5search47  
- **High-Performance Computing** (CPU-first design, SIMD/AVX2-first kernels, OpenMP-style threading discipline, memory alignment & cache behavior). fileciteturn0file5 citeturn25search1  

### How to read this standard

AES uses RFC-style normative keywords. Interpreting “MUST / MUST NOT / SHOULD / MAY” consistently is required for automation and enforcement. citeturn20search0

- **Mandatory**: rules using **MUST / SHALL / MUST NOT / SHALL NOT**. Violations require an approved waiver (see “Tailoring and waivers”).  
- **Advisory**: rules using **SHOULD / SHOULD NOT**. Deviations must be documented with a short rationale in the change record.  
- **Aspirational**: rules using **MAY** or described as “target state.” These are roadmap items and do not block merges unless explicitly elevated by project policy.

### Operating model: Anvil + Saguaro semantic layer

Anvil operates as a multi-agent system. Saguaro is the semantic code intelligence layer that:
- classifies code and changes by criticality and domain,
- enforces tracing, testing, and telemetry obligations,
- performs automated audits (complexity, security, correctness, performance regression),
- compiles evidence into a machine-verifiable change record. fileciteturn0file0 citeturn19search1

### Tailoring and waivers

AES is strict by default. Tailoring is allowed **only** when justified by risk, scope, and lifecycle constraints, and when accompanied by compensating controls.

**AES-POL-TAIL-1 (waiver policy):** Any deviation from a **MUST** requirement MUST be:
1) explicitly documented as a waiver,  
2) time-bounded (expiry date or milestone),  
3) paired with compensating controls,  
4) reviewed by an independent reviewer for high-criticality levels.  
Rationale: high-assurance standards require controlled tailoring and documented relief processes to preserve systematic integrity. citeturn0search46turn2search43turn11search4

## Severity and assurance framework

### Core concept: consequence-based assurance levels

AES uses a single unified severity model called **AAL: Anvil Assurance Level**. AAL is consequence-based and intentionally conservative. It aligns to safety-critical and high-assurance practice by requiring increasing verification evidence as consequences rise. citeturn0search46turn2search43turn11search0turn11search7

AAL is applied per:
- file/module,
- build artifact,
- runtime component,
- and *each change-set* (patch) affecting that scope. fileciteturn0file0

### The AAL scale

| AAL | Priority band | Typical consequence if faulty | Examples |
|---|---|---|---|
| **AAL-0** | **P0** | Catastrophic or irrecoverable harm (safety, mission, or large-scale data integrity); systemic corruption; unrecoverable model or computation results in a way that cannot be detected post-hoc | SIMD kernels; numerical integrators in core simulation loops; crypto primitives; quantum compilation/runtime control; safety-critical control surfaces |
| **AAL-1** | **P1** | Severe correctness or reliability failure that is recoverable but expensive; silent scientific invalidation; major security exposure | Python training loops; loss/optimizer implementations; core inference engine; distributed orchestration; quantum experiment orchestration |
| **AAL-2** | **P2** | Localized failure; service degradation; moderate security issue; incorrect non-core metrics | configuration tooling; ETL jobs; data labeling; internal dashboards; experiment runners |
| **AAL-3** | **P3** | Minor defects, documentation problems, non-critical scripts | docs; examples; notebooks; tutorial code |

Rationale: consequence-based classification is the foundation for systematic assurance scaling and objective-based verification. citeturn0search46turn2search43turn16search2turn13search4

### Mandatory classification and tagging

**AES-AAL-CLASS-1 (mandatory tagging):** Saguaro MUST tag every file and every change-set with:
- language(s),
- domain marker(s): `{general, ml, quantum, physics, hpc}`,
- target AAL,
- hot-path status `{hot, warm, cold}`, derived from profiling/telemetry when executable.  
Rationale: large repositories require automated classification to make enforcement scalable and auditable. fileciteturn0file0 citeturn16search4turn17search1

### Severity decision triggers

A change MUST be elevated to the **highest** applicable AAL when any of the following hold:

- it affects correctness of a conserved quantity / invariant relied upon by downstream scientific conclusions,  
- it changes numerical stability properties, floating-point behavior, or non-finite handling,  
- it changes threading, vectorization, memory layout, or synchronization,  
- it changes security boundaries, authentication, authorization, secrets handling, or dependency supply chain,  
- it changes compilation/transpilation for quantum circuits or changes hardware execution parameters,  
- it affects observability evidence used for go/no-go or rollback,  
- it modifies public APIs or any call surface used by third parties or multiple internal teams (implicit interface risk). citeturn22search4turn16search2turn24search45turn25search1turn7search7turn13search4

### Verification obligations by AAL

**AES-AAL-VER-1 (verification table):** AAL determines minimum verification evidence:

| Evidence obligation | AAL-0 | AAL-1 | AAL-2 | AAL-3 |
|---|---:|---:|---:|---:|
| Bidirectional traceability (req→design→code→test→result and reverse) | MUST | MUST | SHOULD | MAY |
| Independent review | MUST (2 reviewers or 1 independent + specialist) | MUST (1 independent) | SHOULD | MAY |
| Structural coverage | MUST (includes decision logic independence for safety logic) | MUST (statement+branch for critical modules) | SHOULD | MAY |
| Zero dead code policy | MUST | MUST | SHOULD | MAY |
| Fuzz/property-based testing for attack surfaces and parsers | MUST | SHOULD | SHOULD | MAY |
| Reproducibility evidence (deterministic harness, pinned deps) | MUST | MUST | SHOULD | MAY |
| Performance baseline + regression gate for hot paths | MUST | MUST | SHOULD | MAY |
| Security verification level (AES-SVL, defined below) | SVL-3 | SVL-2 | SVL-1 | SVL-0 |

Rationale: objective-based evidence scaling is a proven approach for high-assurance engineering (traceability, coverage adequacy, and independent verification). citeturn2search43turn1search6turn13search4turn10search1turn16search1

## Universal coding mandates

### Architecture mandates

AES is architecture-forward: code quality is a *byproduct* of design clarity, explicit constraints, and continuously verified architecture.

**AES-ARCH-1 (architecture dimensions):** Every production component MUST have an Architecture Block that identifies:
- chosen architectural style (or hybrid),
- required architecture characteristics (quality attributes),
- logical components and boundaries,
- architecture decisions and constraints (what is forbidden).  
Rationale: architecture is the combination of style, characteristics, components, and decisions; making this explicit improves change safety and trade-off decision quality. fileciteturn0file2

**AES-ARCH-2 (trade-off record):** Any decision that meaningfully changes a quality attribute (latency, determinism, safety, security, evolvability, cost) MUST include a short **trade-off record** with at least: decision, alternatives, impacts, rollback plan.  
Rationale: architectural work is primarily trade-off analysis among competing concerns. fileciteturn0file2

**AES-ARCH-3 (fitness functions):** Every AAL-0/AAL-1 system MUST encode its non-negotiable architecture constraints as executable checks (“fitness functions”), run in CI. Examples: no cyclic dependencies, layering rules, forbidden imports, ABI boundaries, determinism checks.  
Rationale: code-level metrics require interpretation; fitness functions automate architecture governance and preserve constraints as systems evolve. fileciteturn0file2

### Language selection policy

AES is not “Python-first.” AES is **risk- and hot-path-first**.

**AES-LANG-1 (default split):**
- **C++** (or similarly low-level, compiled languages) MUST be used for **hot** paths, core kernels, memory- and concurrency-sensitive code, and all AAL-0 performance-critical components.  
- **Python** SHOULD be used for orchestration, experimentation, configuration, and thin wrappers around stable C++ cores.  
Rationale: compiled kernels provide predictable performance and control over memory/layout; scripting layers accelerate iteration while keeping high-assurance compute in analyzable kernels. fileciteturn0file5turn0file6

**AES-LANG-2 (wrapper contract):** Wrapper layers MUST be shallow and declarative:
- no business logic duplication across Python and C++,  
- stable ABI/API boundary,  
- binary compatibility strategy documented,  
- serialization formats versioned.  
Rationale: duplicated logic multiplies defect surfaces and breaks traceability; explicit boundaries preserve evolvability. fileciteturn0file2turn0file6 citeturn22search4turn20search0

### Traceability and “reason-to-exist” rule

**AES-TRC-1 (bidirectional traceability):** For AAL-0/AAL-1, every change MUST link:
- requirement or intent (ticket/spec/issue),
- design notes or decision record,
- code locations,
- tests added/updated,
- verification results (CI run IDs, metrics deltas).  
Rationale: bidirectional traceability ensures everything is implemented and tested, and everything in the system has a justification. citeturn2search43turn1search5turn16search2

**AES-TRC-2 (no mystery features):** Code MUST NOT ship without an explicit requirement/intent tag.  
Rationale: untraceable behavior becomes untestable, unreviewable, and unsafe to modify. citeturn2search43turn13search4

### Complexity and performance discipline

**AES-CPLX-1 (complexity bounds):**
- O(1), O(log n), and O(n) are preferred.  
- O(n²)+ in hot paths is prohibited unless proven non-hot by profiling and guarded by regression tests.  
Rationale: predictable complexity is foundational for latency and throughput SLOs and avoids hidden saturation collapse points. citeturn16search2turn17search1turn17search3

**AES-PERF-1 (hot-path evidence):** Any claim of “faster,” “more efficient,” or “optimized” in AAL-0/AAL-1 MUST include:
- baseline measurements,
- post-change measurements,
- and a variance-aware comparison (p50/p95/p99 or equivalent).  
Rationale: quantitative verification is required for credible performance claims and continuous improvement. citeturn16search2turn16search0turn18search1

### Error-handling philosophy

AES synthesizes three compatible principles into one:

- **errors are explicit values** within components,  
- **crash/abort is acceptable only at carefully defined boundaries**,  
- **recovery is structured and owned by supervisors/orchestrators**, not scattered ad hoc. citeturn23search0turn21search47turn10search1

**AES-ERR-1 (no silent failure):** Silent exception swallowing is forbidden. Any caught exception MUST either:
- be converted to an explicit error return value,
- be rethrown,
- or trigger a controlled fail-fast at a boundary (see AES-ERR-3).  
Rationale: hidden failures destroy observability and make correctness unverifiable. citeturn10search1turn17search1turn23search3

**AES-ERR-2 (explicit error surfaces):** All public functions that can fail MUST expose failure explicitly via one of:
- `Result<T, E>`-style return types in C++ (preferred),
- `(value, error)`-style returns in Python,
- or exceptions only when *construction cannot produce a meaningful object* (constructors / module initialization), AND only when the local language ecosystem mandates it.  
Rationale: explicit error channels force call sites to acknowledge failure and enable systematic testing of error paths. citeturn23search2turn23search1turn10search1

**AES-ERR-3 (supervision boundaries):** AAL-0/AAL-1 systems MUST define “supervisor boundaries” where fail-fast is permitted (e.g., process-level restart, job-level retry, circuit-run abort). Inside a boundary, code MUST be defensive and explicit; across a boundary, the system MAY crash to trigger restart with correct logging and backoff.  
Rationale: structured supervision isolates faults and prevents partial corruption; this is the core of “let it crash” used safely. citeturn21search47turn16search1turn17search1

**AES-ERR-4 (error context):** Errors MUST carry enough context to be actionable (operation, key parameters, component, correlation IDs), but MUST NOT leak secrets.  
Rationale: diagnosing failures requires context; security requires minimizing sensitive exposure. citeturn23search3turn13search4turn9search47

### Documentation requirements

AES documentation MUST be organized by user needs in four forms:
- Tutorial (learning-oriented),
- How-to (task-oriented),
- Reference (complete API/interface),
- Explanation (conceptual/trade-off).  
Rationale: documentation is most effective when structured around distinct user intents and content types. citeturn21search5turn21search3

**AES-DOC-1 (public interface documentation):** Any declared public API MUST have:
- reference docs (parameters, types, errors, invariants),
- at least one how-to for common workflows,
- and an explanation section for dangerous or non-obvious trade-offs.  
Rationale: declared APIs become contracts; incomplete docs create implicit, accidental contracts. citeturn20search0turn22search4turn21search5

### Security baseline

AES defines **SVL: Security Verification Level**, a three-tier scale:

- **SVL-1**: baseline controls for all networked or user-input code.  
- **SVL-2**: strong controls for sensitive data, internal services, and ML pipelines.  
- **SVL-3**: maximum assurance for AAL-0/AAL-1 security boundaries, crypto, authN/Z, secrets, supply chain.

Rationale: tiered verification is necessary to scale security rigor with risk. citeturn9search47turn13search4turn12search0

**AES-SEC-1 (weakness prevention):** Code MUST include proactive prevention for top-ranked weakness classes relevant to the language and surface, including:
- memory safety (bounds checks, UAF prevention),
- injection prevention,
- authZ and authN correctness,
- resource exhaustion controls,
- secrets handling.  
Rationale: recurring weakness patterns dominate real-world exploitation; prevention rules are more reliable than after-the-fact detection alone. citeturn12search0turn10search0turn10search4turn9search47

**AES-SEC-2 (secure development practices):** AAL-1+ repositories MUST implement secure development practices including:
- threat modeling or misuse-case review for exposed surfaces,
- vulnerability handling process,
- dependency review and provenance checks.  
Rationale: secure SDLC practices reduce vulnerability rates and mitigate supply-chain risk. citeturn13search4turn14search5turn14search1

### Supply chain and dependency governance

**AES-SUP-1 (pinned dependencies):** All non-dev dependencies MUST be pinned (lockfiles or exact versions), with upgrade PRs including:
- changelog review,
- compatibility notes,
- security scan output,
- rollback plan.  
Rationale: unpinned dependencies create uncontrolled behavior drift and increase compromise risk. citeturn13search4turn14search5

**AES-SUP-2 (provenance):** AAL-1+ release artifacts MUST publish build provenance sufficient to verify “built as expected,” and SHOULD progress toward higher provenance levels over time.  
Rationale: provenance enables detection of tampering and supports supply-chain incident response. citeturn14search5turn13search4

**AES-SUP-3 (artifact signing):** AAL-1+ release artifacts SHOULD be signed, and verification SHOULD be automated in CI/CD.  
Rationale: cryptographic signing plus transparency logs improve integrity verification and auditability. citeturn14search1turn14search4

**AES-SUP-4 (SBOM):** AAL-1+ releases MUST generate a Software Bill of Materials (SBOM) and attach it to the release record.  
Rationale: SBOMs enable vulnerability response, license compliance, and dependency inventory. citeturn14search0turn14search6

### Concurrency safety

**AES-CON-1 (data race prohibition):** Data races are forbidden. Shared mutable state MUST be protected by:
- ownership discipline,
- locks with documented lock order,
- atomics with documented memory ordering,
- or message passing.  
Rationale: data races create nondeterministic, often unreproducible failures that cannot be reliably tested away. citeturn10search6turn0file5

**AES-CON-2 (signal and interruption discipline):** System-level interruptions (e.g., `EINTR`) MUST be handled correctly in system code; retry loops MUST be bounded and observable (metrics).  
Rationale: correct handling of interrupted system calls and OS errors prevents rare, catastrophic failure modes in production. fileciteturn0file5 citeturn10search1

### API design and evolution

AES adopts two truths simultaneously:

- be strict and well-formed in what you emit,  
- expect variance and malice in what you accept, but without accepting ambiguity that breaks safety. citeturn21search7turn21search4

**AES-API-1 (explicit contracts):** Every public API MUST define:
- input validation rules,
- output schema invariants,
- error schema,
- versioning strategy.  
Rationale: consumers depend on observable behavior, including accidental behavior, so contracts must be explicit and guarded. citeturn22search4turn20search0turn23search3

**AES-API-2 (compatibility discipline):** Backward-incompatible changes MUST:
- increment a declared major version,
- ship a migration guide,
- provide a deprecation window,
- and include compatibility tests if feasible.  
Rationale: version semantics communicate compatibility; immutable released versions prevent “silent rewrites.” citeturn20search0turn22search4

## Language-specific standards

### Python standards

**AES-PY-1 (role of Python):** Python SHOULD be used as orchestration, configuration, data plumbing, and wrapper glue around high-criticality C++ kernels, not as the primary location of high-performance compute.  
Rationale: keeping hot paths out of dynamic runtime environments reduces performance variance and improves analyzability. fileciteturn0file6

**AES-PY-2 (typing requirement):** All AAL-1+ Python modules MUST use type annotations for public APIs and critical internal boundaries.  
Rationale: type annotations convert classes of runtime errors into build-time (CI) failures and improve maintainability. citeturn19search2

**AES-PY-3 (numerical stability by construction):** For ML/numerical code in Python:
- prefer library primitives that explicitly provide stabilized computations (`logsumexp`, `logaddexp`, stable softmax),  
- forbid naive `exp`/`softmax` implementations in production code.  
Rationale: numerically stabilized primitives prevent overflow/underflow and silent NaNs. citeturn24search3turn24search1turn24search45

**AES-PY-4 (exceptions):** Python exceptions MAY be used internally, but any AAL-1+ module boundary MUST translate exceptions into:
- structured errors, OR  
- a controlled crash at an explicit supervisor boundary (job/process).  
Rationale: explicit failure channels + structured supervision reduce undefined partial states. citeturn21search47turn23search0turn10search1

### C/C++ standards

**AES-CPP-1 (no undefined behavior):** AAL-0/AAL-1 C/C++ MUST be written so that undefined behavior is provably absent under stated preconditions.  
Rationale: undefined behavior invalidates compiler assumptions, makes testing unreliable, and can create security vulnerabilities. citeturn9search3turn9search4turn10search0

**AES-CPP-2 (resource safety):** Resource management MUST be exception-safe and leak-free (RAII discipline), whether exceptions are enabled or not.  
Rationale: resource leaks and partial initialization are dominant root causes in long-running systems. citeturn10search2turn11search4turn19search4

**AES-CPP-3 (exceptions policy):**
- For AAL-0/AAL-1, exceptions MUST NOT cross module or ABI boundaries.  
- Within a tightly controlled component, exceptions MAY be used only if the component’s error model and callers are explicitly exception-safe, and CI enforces it.  
Rationale: uncontrolled exception propagation breaks integration assumptions and complicates verification across large codebases. citeturn19search4turn10search2turn2search43

**AES-CPP-4 (integer safety):** Integer operations MUST not overflow unless explicitly intended and guarded (checked arithmetic).  
Rationale: overflow is a common root cause for memory corruption and exploitation. citeturn10search0turn10search4turn12search0

**AES-CPP-5 (alignment discipline):** Any use of aligned SIMD loads MUST guarantee required alignment or use unaligned-safe intrinsics.  
Rationale: aligned intrinsics can fault or silently degrade; correctness requires explicit alignment control. citeturn25search1turn25search0

### Quantum circuit standards

AES defines circuit discipline independent of library brand, then maps to common runtime behaviors.

**AES-QC-1 (immutability discipline):** Circuit elements that are defined as immutable by the chosen framework MUST NOT be mutated; transformations MUST create new objects or use supported transformation APIs.  
Rationale: mutating “immutable” circuit structures violates library invariants and leads to subtle correctness failures. citeturn8search1

**AES-QC-2 (no hidden compilation):** Circuit compilation/transformation MUST be explicit and user-initiated. No tool or wrapper is allowed to silently rewrite circuits.  
Rationale: “what you see is what you get” preserves experimental intent and supports reproducibility. citeturn8search1

**AES-QC-3 (hardware compatibility):** Hardware-bound circuits MUST use only supported qubits/gates/connectivity, and compilation MUST report:
- depth change,
- two-qubit gate count change,
- routing (SWAP) overhead.  
Rationale: mapping and routing dominate fidelity under noise; compilation deltas are essential evidence. citeturn7search2turn7search5turn7search7turn8search0

## Domain-specific standards

### Deep learning and machine learning standards

**AES-ML-1 (training loop integrity):** Training loops at AAL-1 MUST:
- record seeds and determinism settings,
- detect and fail-fast on non-finite loss/gradients,
- log gradient norms and outlier detection,
- version data, code, and config in a single run record.  
Rationale: ML systems are difficult to specify a priori; testing + monitoring + reproducibility are required to reduce ML technical debt. citeturn6search1turn16search0turn24search45

**AES-ML-2 (numerical stability mandates):** ML code MUST use stable numerics for:
- softmax/log-softmax (subtract max / logsumexp),
- log probabilities (logaddexp/logsumexp),
- reductions where cancellation is plausible (compensated summation or numerically stable algorithms).  
Rationale: stable transforms eliminate overflow/underflow and reduce catastrophic cancellation risk in deep stacks. citeturn24search45turn24search3turn5search47

**AES-ML-3 (gradient correctness verification):** For any new optimizer, loss, custom autograd, or fused kernel:
- MUST include gradient checks (finite differences vs autodiff) on small randomized cases,
- MUST include non-finite tests and extreme-value tests.  
Rationale: gradient integrity failures can silently invalidate training; small randomized checks catch classes of errors not found by end-to-end metrics. citeturn6search1turn24search45

**AES-ML-4 (data validation):** Any AAL-1 data pipeline MUST enforce:
- schema validation,
- drift detection,
- training-serving skew checks when serving exists.  
Rationale: drift/skew are common production failure modes; explicit validation reduces silent degradation. citeturn6search3turn6search0turn6search1

**AES-ML-5 (model versioning and reproducibility):** AAL-1 models MUST be reproducible enough to:
- rebuild from pinned dependencies and recorded inputs/config,
- re-run evaluation deterministically (or with bounded stochastic variance reported),
- compare to golden checkpoints with defined acceptance thresholds.  
Rationale: reproducibility is a prerequisite for scientific validity and safe iteration. citeturn6search1turn14search5turn16search1

#### Numerical stability code examples

Stable log-softmax (Python)

```python
import torch

def log_softmax_stable(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # Uses numerically stabilized logsumexp
    return logits - torch.logsumexp(logits, dim=dim, keepdim=True)
```

Rationale: `logsumexp` is explicitly stabilized; subtract-max/logsumexp stabilization avoids overflow/underflow. citeturn24search3turn24search45

Compensated summation (C++)

```cpp
// Kahan-style compensated summation for improved accuracy
double kahan_sum(const double* x, size_t n) {
    double sum = 0.0;
    double c = 0.0;  // compensation
    for (size_t i = 0; i < n; ++i) {
        double y = x[i] - c;
        double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    return sum;
}
```

Rationale: floating-point arithmetic requires explicit error control for long reductions; compensated summation reduces loss of significance. citeturn5search47turn4search2

### Quantum computing standards

**AES-Q-1 (noise-aware execution):** Hardware-bound quantum workflows MUST track and record:
- backend identity,
- compilation settings and optimization level,
- calibration epoch (or equivalent),
- noise model version when using noisy simulation or mitigation.  
Rationale: noise and compilation choices can dominate observed outcomes; recording them is required for reproducibility and correct attribution. citeturn7search7turn7search4turn7search0

**AES-Q-2 (simulator parity):** Any circuit executed on hardware MUST have:
- an ideal simulator validation for functional correctness, and  
- a noise-aware simulator or mitigation plan when performance claims are made.  
Rationale: simulators distinguish algorithmic defects from hardware noise artifacts. fileciteturn0file1 citeturn7search0turn8search5

**AES-Q-3 (depth and routing minimization):** Circuit design MUST minimize:
- depth,
- two-qubit gates,
- routing SWAP overhead,  
and MUST report these metrics pre/post compilation.  
Rationale: depth and two-qubit operations compound error; routing overhead is often the primary fidelity killer. citeturn7search2turn8search0turn8search1

**AES-Q-4 (error mitigation disclosure):** If error mitigation is used, results MUST report:
- mitigation method,
- overhead (shots or circuit variants),
- whether bias is expected or not.  
Rationale: mitigation can improve results but may introduce bias and significant overhead; disclosure is required for scientific validity. citeturn7search0turn0search46

### Physics simulation standards

**AES-PHY-1 (invariant ledger):** Every physics simulator MUST define an “invariant ledger” listing:
- conserved quantities (energy, momentum, charge, norm, etc.),
- symmetries (translation, rotation, gauge constraints),
- expected tolerances and drift budgets by integrator/time step.  
Rationale: explicit invariants enable automated correctness checks and prevent silent physics invalidation. citeturn4search2turn5search47

**AES-PHY-2 (conservation tests):** AAL-0/AAL-1 simulation code MUST include tests that measure:
- invariant residuals over time,
- sensitivity to time step and solver tolerances,
- regression against known analytic or benchmark solutions where available.  
Rationale: error propagation and rounding accumulation can silently destroy correctness; tests must measure invariants directly. citeturn4search2turn5search47

### High-performance computing standards

**AES-HPC-1 (CPU-first / SIMD-first):** All hot paths MUST be designed CPU-first:
- scalar reference implementation first (correct and tested),
- then vectorized kernel with identical semantics,
- then threaded scaling.  
Rationale: staged optimization preserves correctness and enables differential testing against a trusted baseline. citeturn16search2turn25search1

**AES-HPC-2 (alignment and layout):** SIMD code MUST:
- explicitly control alignment (`alignas`, aligned allocators),
- define struct layout to prevent false sharing,
- document cache-line assumptions for shared counters and per-thread buffers.  
Rationale: misalignment can fault or degrade; false sharing creates saturation and latency collapse. citeturn25search1turn0file5

**AES-HPC-3 (deterministic parallel semantics):** Parallel floating-point reductions MUST document:
- whether strict reproducibility is required,
- what ordering is assumed,
- and what error bounds are expected.  
Rationale: parallel reduction order changes rounding; reproducibility demands explicit policy. citeturn5search47turn4search2

## Verification and testing framework

### Test taxonomy

AES requires a layered test portfolio:

- **Unit tests**: pure logic, deterministic, fast.  
- **Integration tests**: component boundaries, I/O, serialization, concurrency boundaries.  
- **Property-based tests**: invariants across randomized inputs (especially for parsers, math kernels).  
- **Fuzz tests**: hostile inputs for exposed surfaces.  
- **Benchmark tests**: performance regression protection for hot paths.  
Rationale: diversified testing is required because each method catches different defect classes; fuzzing is particularly effective for memory safety and parser bugs. citeturn10search1turn10search4turn16search2

### Coverage and adequacy rules

**AES-TST-1 (coverage adequacy concept):** Coverage numbers are not sufficient alone; coverage is used to assess *test adequacy* and detect untested logic.  
Rationale: structural coverage exists to ensure the test set is adequate for critical logic. citeturn1search5turn1search6turn2search43

**AES-TST-2 (high-criticality coverage):** AAL-0 decision logic with safety or security consequences MUST demonstrate decision logic independence coverage for boolean decisions (MC/DC-style adequacy).  
Rationale: independence coverage is required for the highest criticality to ensure each condition can independently affect decisions. citeturn1search6turn2search43

### ML-specific testing buckets

Anvil enforces four ML readiness buckets:

- **Data tests** (schema, anomalies, drift/skew), citeturn6search3turn6search0  
- **Model tests** (slice-based metrics, regression vs golden checkpoints), citeturn6search1  
- **Infrastructure tests** (reproducible builds, deterministic eval harness), citeturn14search5turn6search1  
- **Monitoring tests** (telemetry presence and alert hooks). citeturn17search1turn15search0  

Rationale: ML systems require tests and monitoring beyond traditional software due to data dependence and behavior specification difficulty. citeturn6search1

### Quantum-specific testing

**AES-Q-TST-1 (ideal validation):** All circuits MUST pass ideal simulation tests for structural correctness.  
Rationale: ideal simulation isolates algorithmic defects from noise. fileciteturn0file1

**AES-Q-TST-2 (noise-aware validation):** When making hardware performance claims, workflows MUST test against a documented noise model or mitigation strategy.  
Rationale: noise dominates NISQ-era outcomes; performance must be conditioned on noise assumptions. citeturn7search0turn8search5turn8search0

## Observability, telemetry, and operations governance

### Golden signals and domain mapping

AES requires four universal health signals:
- latency,
- throughput,
- errors,
- saturation.  
Rationale: focusing on these four signals provides strong coverage for user-facing incidents and capacity collapse. citeturn17search1turn16search2

**AES-OBS-1 (mandatory emission):** Every AAL-1+ service/job MUST emit golden signals with consistent tags including:
- component,
- version,
- run_id,
- AAL,
- domain marker.  
Rationale: consistent telemetry enables correlation, diagnosis, and error-budget governance. citeturn15search0turn17search1turn16search2

### Chronicle protocol

**AES-CHR-1 (baseline → change → result):** Any AAL-0/AAL-1 hot-path change MUST follow Chronicle:
1) record pre-change golden signals baseline,  
2) implement change + tests,  
3) record post-change signals,  
4) bind delta to change record (PR/commit).  
Rationale: baselining prevents silent regressions and enables quantitative improvement loops. citeturn16search1turn17search1turn18search1

### Structured logs and tracing

**AES-OBS-2 (structured logs):** Logs MUST be structured and correlate to trace/span context when applicable (request IDs, run IDs).  
Rationale: correlation reduces mean time to detect and diagnose failures. citeturn15search3turn15search0

### SLOs and error budgets

**AES-SRE-1 (SLOs):** Any user-facing or pipeline-critical component MUST define SLOs with clear SLIs and measurement windows.  
Rationale: SLOs set explicit expectations and allow objective alerting and prioritization. citeturn16search2turn16search4

**AES-SRE-2 (error budgets):** Components with SLOs MUST operate with error budgets and an enforcement policy (freeze rules, postmortems, escalation).  
Rationale: error budgets balance reliability with innovation and provide a control mechanism for change velocity. citeturn16search1turn16search4turn16search3

### Agent workflow governance

**AES-AG-1 (code generation guardrails):** For AAL-0/AAL-1, Anvil MUST:
- produce trace links (intent/design/tests),
- produce a verification plan,
- run static analysis + tests,
- and summarize evidence in the change record.  
Rationale: autonomous agents must attach verifiable evidence to maintain trust and auditability. citeturn13search4turn19search1

**AES-AG-2 (red-team protocol):** For AAL-0/AAL-1, Anvil MUST run a red-team checklist on the change:
- complexity check,
- failure mode analysis,
- security weakness scan,
- regression gates,
- rollback plan.  
Rationale: adversarial review reduces blind spots and prevents “defect masking,” emphasizing root-cause fixes. citeturn16search1turn12search0turn10search1

### Code review and change management

**AES-REV-1 (every-line review):** Reviewers SHOULD examine every human-written line they are assigned and explicitly state scope when partial.  
Rationale: thorough review prevents hidden defects and improves readability for future maintainers. citeturn19search1

**AES-REV-2 (review gating by AAL):**
- AAL-0: MUST have independent review + specialist review where relevant (security, concurrency, numerics).  
- AAL-1: MUST have independent review.  
Rationale: independence reduces confirmation bias and increases defect detection for critical systems. citeturn2search43turn19search1

**AES-REV-3 (comment protocol):** Reviews SHOULD use structured tags for machine readability, e.g.:
- `blocker:` must fix,
- `issue:` should fix,
- `nit:` optional,  
- `question:` clarify.  
Rationale: consistent comment formats reduce churn and enable automated review analytics. citeturn19search5turn19search1

### Anti-patterns registry

The following patterns are forbidden in AAL-0/AAL-1 unless waived with compensating controls:

- Silent exception swallowing. citeturn10search1  
- Unchecked standard library or system-call error returns. citeturn10search1turn0file5  
- Undefined behavior or reliance on unspecified behavior in C/C++. citeturn9search3turn10search0  
- Naive softmax / log-softmax implementations using raw exponentials without stabilization. citeturn24search45turn24search3  
- Dependency upgrades without pinned versions and provenance evidence. citeturn14search5turn13search4  
- API changes that alter observable behavior without explicit contract/version bump and backward-compatibility plan. citeturn22search4turn20search0  
- Circuit execution on hardware without recorded compilation settings and depth/two-qubit deltas. citeturn7search2turn7search7  

Rationale: these failures are high-frequency root causes of security incidents, scientific invalidation, and reliability collapse. citeturn12search0turn6search1turn17search1

### Appendices

#### Quick-reference checklists

**AAL-0 merge checklist (minimum):**
- Traceability complete (intent/design/code/tests/results). citeturn2search43  
- Independent review complete; scope declared. citeturn19search1  
- Structural adequacy checks for decision logic. citeturn1search6  
- Determinism + non-finite tests (numerics/ML). citeturn24search45turn6search1  
- Chronicle baseline and regression gate for hot paths. citeturn16search1turn17search1  
- SBOM + provenance attached for releases. citeturn14search6turn14search5  

#### Severity decision tree (text)

- If change affects SIMD/hot kernel, low-level memory layout, concurrency safety, or invariant enforcement → AAL-0. citeturn25search1turn10search6turn4search2  
- Else if change affects training loop correctness, optimizer/loss/gradient math, quantum hardware execution workflow → AAL-1. citeturn6search1turn7search7turn24search45  
- Else if change affects tooling/config/pipelines and is operationally significant → AAL-2. citeturn13search4turn16search1  
- Else docs/examples/non-critical scripts → AAL-3. citeturn21search5  

#### Glossary of synthesized terms

- **AAL (Anvil Assurance Level):** consequence-based criticality tier controlling evidence requirements. citeturn0search46turn2search43  
- **Chronicle protocol:** baseline→change→result evidence binding for performance/correctness. citeturn16search1turn17search1  
- **Fitness function:** executable rule that enforces architecture constraints continuously. fileciteturn0file2  
- **SVL (Security Verification Level):** AES security verification tier aligned to risk. citeturn9search47turn13search4turn12search0  

#### Source backbone for this edition

This edition is additionally informed by core software architecture, systems programming, and quantum computing references used as high-density engineering guidance. fileciteturn0file2turn0file5turn0file1turn0file6turn0file4turn0file3