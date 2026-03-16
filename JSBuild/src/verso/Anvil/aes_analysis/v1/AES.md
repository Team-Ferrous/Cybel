# The Anvil Engineering Standard (AES)

**A unified, high-assurance engineering rulebook for Anvil + Saguaro across systems, ML, quantum, physics simulation, and HPC**
**Version:** 2.0 (Unified Synthesis) • **Status:** Normative • **Applies to:** all repositories touched by Anvil

> **Prime Directive:** *Correctness is not a feature. It is an invariant you must continually prove you have not broken.*

---

## Table of Contents

1. [PREAMBLE](#1-preamble)
2. [SEVERITY & ASSURANCE FRAMEWORK](#2-severity--assurance-framework)
3. [UNIVERSAL CODING MANDATES](#3-universal-coding-mandates)
4. [LANGUAGE-SPECIFIC STANDARDS](#4-language-specific-standards)

   * [4a. Python Standards](#4a-python-standards)
   * [4b. C/C++ Standards](#4b-cc-standards)
   * [4c. Quantum Circuit Standards (Qiskit/Cirq patterns)](#4c-quantum-circuit-standards-qiskitcirq-patterns)
5. [DOMAIN-SPECIFIC STANDARDS](#5-domain-specific-standards)

   * [5a. Deep Learning / ML Standards](#5a-deep-learning--ml-standards)
   * [5b. Quantum Computing Standards](#5b-quantum-computing-standards)
   * [5c. Physics Simulation Standards](#5c-physics-simulation-standards)
   * [5d. High-Performance Computing Standards](#5d-high-performance-computing-standards)
6. [VERIFICATION & TESTING FRAMEWORK](#6-verification--testing-framework)
7. [OBSERVABILITY & TELEMETRY](#7-observability--telemetry)
8. [AGENT WORKFLOW GOVERNANCE](#8-agent-workflow-governance)
9. [CODE REVIEW & CHANGE MANAGEMENT](#9-code-review--change-management)
10. [ANTI-PATTERNS REGISTRY](#10-anti-patterns-registry)
11. [APPENDICES](#11-appendices)

---

## 1. PREAMBLE

### 1.1 Purpose and scope

AES defines **binding** engineering constraints and evidence requirements for code **written, modified, reviewed, audited, or maintained** by the Anvil multi-agent system and Saguaro semantic intelligence layer across large repositories. AES is designed so that if followed strictly, Anvil’s output achieves **flight-software-grade assurance** while staying practical for research velocity. 

AES governs:

* **Software architecture** (system decomposition, boundaries, contracts, determinism)
* **Implementation** (correctness, safety, security, performance)
* **Verification** (coverage, fault analysis, evidence closure)
* **Operations** (telemetry, SLOs, regressions, incident readiness)
* **Agent governance** (autonomy constraints, self-verification, red-team protocols)

### 1.2 Domains governed

AES applies to:

* **General software engineering** (Python 3.12+, C++17+, systems programming)
* **Deep Learning / ML** (training loops, gradient integrity, numerical stability, reproducibility)
* **Quantum computing** (circuits, VQCs, hybrid loops, error mitigation/correction awareness)
* **Physics simulation** (numerical methods, invariants, conservation laws, symmetry enforcement)
* **HPC** (CPU-first design, SIMD/AVX2 discipline, OpenMP, cache + alignment, determinism)

### 1.3 How to read this document

AES uses RFC 2119 keywords **MUST**, **MUST NOT**, **SHALL**, **SHOULD**, **MAY**. ([IETF Datatracker][1])
Rule classes:

* **Mandatory**: MUST / MUST NOT / SHALL — violation is a defect; merge is blocked unless waived
* **Advisory**: SHOULD — deviation requires documented rationale
* **Aspirational**: MAY — optional improvements when cost-effective

**Waivers:** Any deviation from a MUST/MUST NOT requires:

1. written rationale, 2) risk acceptance, 3) approval proportional to assurance level, 4) a time-bounded follow-up ticket.

---

## 2. SEVERITY & ASSURANCE FRAMEWORK

### 2.1 Unified assurance levels (AAL)

AES uses a single four-tier ladder, evaluated **per change** (not per repo). The selected AAL MUST be recorded in the change description and linked into traceability. 

| **AAL** | **Name** | **Existing Severity**rance intent** |
|---|---|---:|---|---|
| **AAL-0** | Catastrophic | P0 | SIMD kernels, allocators, schedulers, crypto, circuit compilers, “ground truth” integrators | Failures can silently corrupt results or cause systemic loss |
| **AAL-1** | Critical | P1 | Training loops, autograd extensions, hybrid quantum-classical orchestration, poisoning-prone ingestion | Failures produce wrong scientific/operational outcomes |
| **AAL-2** | Major | P2 | Tooling, config loaders, non-hot-path libraries, batch pipelines | Failures are localized/recoverable but costly |
| **AAL-3** | Minor | P3 | Docs, notebooks, non-executable references | Low risk; correctness still matters |

This mapping is the operational synthesis Anvil already uses and MUST be preserved. 

### 2.2 AAL selection rules (strictest-winsat the **highest** AAL triggered by any condition:

1. **Silent correctness loss**: plausible defects can corrupt model weights, circuit outputs, physical invariants, or benchmark truth → **≥ AAL-1**
2. **Hazard linkage**: touches hazards / hazard controls / safety mitigations → **≥ AAL-1** (often AAL-0)
3. **Attack surface**: parsing, deserialization, network boundaries, plugin execution, dependency acquisition → **≥ AAL-1**
4. **Hot path**: materially impacts latency/throughput/saturation (SIMD/OpenMP/allocators/schedulers) → **AAL-0**

These rules are mandatory because performance-critical and correctness-critical code have outsized blast radius. 

### 2.3 Verification requirements per AAL (gntrol | **AAL-0** | **AAL-1** | **AAL-2** | **AAL-3** |

|---|---:|---:|---:|---:|
| Traceability (req→design→code→test→result) | MUST | MUST | SHOULD | MAY |
| Independent review (not the author) | MUST | MUST | SHOULD | MAY |
| Structural coverage | MUST include MC/DC where decision logic matters | MUST include branch coverage | SHOULD | MAY |
| Dead/extraneous code | MUST NOT exist | MUST NOT exist | SHOULD NOT | MAY |
| Static analysis (security + UB + concurrency + numeric) | MUST | MUST | SHOULD | MAY |
| Fault analysis (FMEA/FTA) | MUST | SHOULD | MAY | MAY |
| Performance baselines + regression gates | MUST | MUST | SHOULD | MAY |
| Provenance (build inputs, process, outputs) | MUST | MUST | SHOULD | MAY |

This is the minimum; projects MAY exceed it. 

### 2.4 Downgrade prohibition (assurance inhT inherit a lower AAL than any component it can:

* corrupt (data/control coupling),
* starve (resource coupling),
* impersonate (authz/authn coupling),
* or silently influence (implicit contract coupling).

**Exception:** demonstrable, enforced isolation (process boundary + validated serialization contract + resource limits + denial containment).

---

## 3. UNIVERSAL CODING MANDATES

### 3.1 Architecture is a first-class safety mechanism

**AES-ARCH-1 (Component Boundaries):** Systems MUST be decomposed into **components** with explicit contracts: inputs, outputs, errors, invariants, performance envelope, and telemetry obligations.
**Rationale:** Clear contracts reduce hidden coupling and make verification feasible at scale.

**AES-ARCH-2 (Layering & Dependency Direction):** Dependencies MUST flow from **high-level orchestration → domain logic → low-level kernels**, never the reverse. Hot kernels MUST NOT depend on orchestration frameworks.
**Rationale:** Prevents architectural contamination and keeps critical paths analyzable.

**AES-ARCH-3 (Determinism Zones):** Each subsystem MUST declare a **Determinism Mode**:

* **D0 (Strict):** bitwise deterministic (preferred for AAL-0 truth kernels and validation harnesses)
* **D1 (Statistical):** deterministic given seed + environment manifest
* **D2 (Best-effort):** allowed only for AAL-2/AAL-3 experimentation

Determinism Mode MUST be enforced by tests and by runtime checks where applicable.
**Rationale:** Prevents irreproducible science and un-debuggable regressions.

**AES-ARCH-4 (Configuration as Code):** All behavior-affecting configuration MUST be:

* typed (schema),
* versioned,
* validated,
* and traceable to experiments/releases.

No “mystery knobs”.
**Rationale:** Config drift is a primary cause of silent behavior change.

---

### 3.2 Traceability & evidence closure (non-negotiable)

**AES-TR-1 (End-to-end Traceability):** All non-trivial code MUST be traceable:
**requirement → design decision → code change → test(s) → verification result(s)**. 

**AES-TR-2 (Evidence Closure):** Verification e is:

* archived,
* reproducible,
* replayable by an independent agent,
* and linked to the change record. 

---

### 3.3 Cleanroom mandate: prevent defectS-CR-ROOT-1 (No Symptom Masking):** Code MUST NOT “hide” defects via:

* broad exception catches,
* silent fallbacks,
* disabling checks,
* “ignore and continue” patterns.

Instead, fix root cause and add a regression test that fails without the fix. 

**AES-CR-ROOT-2 (Operational Distributions):T cover representative operational distributions; probabilistic failure claims MUST include confidence assumptions. 

---

### 3.4 Zero dead/extraneous code polic Code):** Shipped artifacts MUST contain **no extraneous code** (untraceable to a requirement) and **no dead code**. If structural coverage reveals unexecuted code, Anvil MUST resolve by **testing, correcting requirements, or removing code**. 

---

### 3.5 Complexity & performance discipomplexity Bound):** Hot paths MUST be **O(1), O(log n), or O(n)**. **O(n²)+** in hot paths MUST NOT be introduced unless a bounded-`n` proof is documented and enforced at runtime. 

**AES-PERF-2 (Chronicle Protocol is mandatorhanges MUST follow:
**Baseline → Change → Result → Delta (logged and gated)**. 
**Rationale:** Golden-signal regressions ar([Google SRE][2])

**AES-CPLX-1 (Cyclomatic Complexity):** AAL-0/AAL-1 functions SHOULD be ≤ 10 cyclomatic complexity; exceeding requires decomposition or MC/DC justification.
**Rationale:** High complexity makes structural coverage and review unreliable.

---

### 3.6 Unified error-handling philosophy (explicit + fail-fast + structured)

**AES-ERR-1 (Recoverable errors are values):** Recoverable failures MUST be surfaced explicitly in return types (or typed result objects) and MUST NOT be silently ignored. 

**AES-ERR-2 (Invariant violations crash fast):** When internal invariants are violated, the component SHOULD fail loudly an at a higher supervision boundary, not by burying failures. 

**AES-ERR-3 (Structured errors):** Errors MUST carry structured context enabling classification and observability. ERR-4 (Deterministic error contract):** All public APIs MUST document: what can fail, how failure is surfaced, and whailure. 

**AES-ERR-5 (Error paths tested):** Error paths MUST be tested; AAL-0/AAL-1 require realistic fault injection (I/O errors, 

---

### 3.7 Documentation requirements (Diátaxis)

**AES-DOC-1 (Diátaxis partition):** Documentation MUST be organized as *ce / Explanation** and MUST NOT mix categories.  ([diataxis.fr][3])

**AES-DOC-2 (API reference completeness):** Every public API MUST specify:
inputs, outputs, invariants,inism mode, and performance envelope (if relevant). 

---

### 3.8 Security baseline (risk-proportional, control-catalog aligned)

**AES-SEC-1 (Verification levels):** Components MUST meet a baseline security verification level; Assurance.  ([OWASP][4])

**AES-SEC-2 (CWE Top 25 defense):** AAL-0/AAL-1 MUST explicitly mitigate the relevant CWE Top 25 weakness classes for their surface area. ([CWE][5])

**AES-SEC-3 (Control catalog alignment):** Security control selection MUST be aligned to a recognized control catalog (access control, audit, comms protection, configuration management, etc.).  ([NIST Computer Security Resource Center][6])le boundaries):** Parsing, deserialization, templating, and format-string handling MUST be designed to prevent injection and memory safety hazards. 

---

### 3.9 Dependency management & supply chain integrity

**AES-SUP-1 (Pin + integritT be pinned; AAL-0/AAL-1 MUST use integrity verification (hashes / lockfiles). 

**AES-SUP-2 (Provenance):** Build outputs MUST be traceable via provenance (what built it, how, with which inputs). ([SLSA][7])

arency):** High-assurance releases SHOULD be signed and recorded in transparency-log style systems. ([OpenSSF][8])

---

### 3.10 Concurrency safety

**AES-CONC-1 (No data races):* threads without synchronization is forbidden.
**AES-CONC-2 (Lock order):** If multiple locks are used, they MUST follow a consistent global ordering.
**AES-CONC-3 (Deterministic concurrency):** AAL-0 truth kernels MUST NOT depend on scheduler nondeterminism; use deterministic reduction strategies.

---

### 3.11 Language selection policy (C++ first for kernels, Python as orchestration)

**AES-LANG-1 (Fit-for-purpose language):** Anvil MUST choose implementation language based on constraints, not convenience:

* **C++ (primary)** for: hot paths, kernels, simulation cores, parsers, schedulers, memory-sensitive systems, deterministic compute.
* **Python (orchestration)** for: experiment wiring, configuration, pipeline composition, testing harnesses, thin wrappers around native kernels.
* **Other languages MAY be used** if they dominate the requirement constraints (e.g., Rust for memory-safe systems tooling, Go for service glue), but must still satisfy AES evidence rules.

**AES-LANG-2 (No Python hot loops):** If Python code contributes materially to hot-path time, it MUST be moved to a vectorized library path or a native C++ extension.

---

## 4. LANGUAGE-SPECIFIC STANDARDS

## 4a. Python Standards

### 4a.1 Typing is mandatory (especially at boundaries)

**AES-PY-1 (Type annotations):** All public Python APIs MUST provide type annotations. AAL-0/AAL-1 code MUST pass strict type checking; `Any` is forbidden except at explicitly documented I/O boundaries. 

**AES-PY-2 (Structured data contracts):** Public-facing structured data MUST use typed schemas (dataclasses / pydantic-style models). Plain `dict` MUST NOT be the primary contract in AAL-0/AAL-1.

### 4a.2 Safety patterns

**AES-PY-3 (No mutable defaults):** Mutable default arguments are forbidden. 

**AES-PY-4 (No dynamic execution):** `eval()` / `exec()` are forbidden outside explicitly sandboxed contexts.

### n (explicit)

Prefer `Result[T, E]`-style returns for recoverable errors at system boundaries; raise exceptions for programmer errors/invariant violations.

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Generic, TypeVar, Union, Literal

T = TypeVar(":contentReference[oaicite:58]{index=58}ass(frozen=True)
class Ok(Generic[T]):
    value: T

@dataclass(frozen=True)
class Err(Generic[E]):
    error: E

Result = Union[Ok[T], Err[E]]

@dataclass(frozen=True)
class ParseError:
    kind: Literal["invalid_format", "out_of_range"]
    msg: str

def parse_lr(s: str) -> Result[float, ParseError]:
    try:
        x = float(s)
    except ValueError:
        return Err(ParseError("invalid_format", f"not a float: {s!r}"))
    if not (0.0 < x <= 10.0):
        return Err(ParseError("out_of_range", f"lr out of range: {x}"))
    return Ok(x)
```

---

## 4b. C/C++ Standards

### 4b.1 No undefined behavior (UB) in high assurance code

**AES-CPP-1 (UB is forbidden):** In AAL-0/AAL-1 C/C++, constructs that can trigger UB MUST NOT appear (OOB, signed overflow, data races, invalid casts). Static analysis + sanitizers are mandatory gates.
**Rationale:** UB can invalidate tests because compilers optimize assuming UB never occurs.

### 4b.2 Resource management (RAII-only)

**AES-CPP-2 (RAII required):** Ownership MUST be expressed via RAII; raw `new/delete` are forbidden in application code.
**Rationale:** RAII prevents leaks and makes failure paths safe.

### 4b.3 Type safety mandates

**AES-CPP-3 (No void* polymorphism):** `void*`, C-style unions, and `std::any` are forbidden for polymorphic state in high assurance code; use `std::variant` with exhaustive visitors. 

````cpp
#include <variant>
#include <string>
#include <type_traits>

using SensorData = std::variant<double, int, std::string>;

void process_sensor_telemetry(const SensorData& data) {
  std::visit([](auto&& arg) {
    using T = std::decay_t<decltype(arg)>;
    if constexpr (std::is_same_v<T, double>) {
      // ...
    } else if constexpr (std::is_same_v<T, int>) {
      // ...
    } else if constexpr (std::is_same_v<T, std::string>) {
      /:contentReference[oaicite:60]{index=60}``
:contentReference[oaicite:61]{index=61}

### 4b.4 “Error values” in C++
**AES-CPP-4 (nodiscard errors):** Functions returning error-bearing values MUST be `[[nodiscard]]`.  
Prefer `std::expected<T, E>`-style patterns (or an equivalent) for recoverable failures.

---

## 4c. Quantum Circuit Standards (Qiskit/Cirq patterns)

### 4c.1 Circuits are inspectable artifacts
**AES-QC-1 (Circuit objects + inspection):** Circuits MUST be explicit circuit objects and MUST be inspected after op:contentReference[oaicite:62]{index=62}detect semantic drift. :contentReference[oaicite:63]{index=63} :contentReference[oaicite:64]{index=64}

### 4c.2 No magic angles
**AES-QC-2 (Named parameters):** Parameterized circuits MUST use named parameters; magic-number angles are forbidden.

### 4c.3 Noise-aware workflow is mandatory for hardware
**AES-QC-3 (Noise-aware validation):** Any NISQ-targeted circuit MUST run:
1) ideal simulation, 2) noisy simulation with documented noise model, 3) hardware run (or explicit waiver). :contentReference[oaicite:65]{index=65} :contentReference[oaicite:66]{index=66}:contentReference[oaicite:67]{index=67}c.4 Hybrid loop determinism
**AES-QC-4 (Replay checkpoints):** Hybrid classical–quantum loops MUST record per-step: parameters, seeds, backend settings, transpiler settings, and compiled circuit artifacts for replay in AAL-0/AAL-1. :contentReference[oaicite:68]{index=68}

---

## 5. DOMAIN-SPECIFIC STANDARDS

## 5a. Deep Learning / ML Standards

### 5a.1 Training loop integrity (AAL-1 minimum; AAL-0 for core kernels)
**AES-ML-1 (Gradient health checks):** Training loops MUST check for:
-:contentReference[oaicite:69]{index=69}e gradients,
- exploding norms,
- optimizer state anomalies,
before each optimizer step.

**AES-ML-2 (Gradient correctness verification):** Custom gradients MUST be verified via gradcheck-style finite differences before distributed training. :contentReference[oaicite:70]{index=70} :contentReference[oaicite:71]{index=71}

### 5a.2 Numerical stability rules
**:contentReference[oaicite:72]{index=72}ions):** Operations prone to catastrophic cancellation MUST use stabilized forms (e.g., LogSumExp, Kahan summation). :contentReference[oaicite:73]{index=73}

**Stable softmax (required pattern):**
```python
import torch

def stable_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # REQUIRED: subtract max for numerical stability
    m = torch.amax(x, dim=dim, keepdim=True)
    z = x - m
    ez = torch.exp(z)
    return ez / torch.sum(ez, dim=dim, keepdim=True)
````

**Kahan summation (rehpp
struct KahanSum {
double sum{0.0};
double c{0.0};
void add(double x) {
double y = x - c;
double t = sum + y;
c = (t - sum) - y;
sum = t;
}
};

````
IEEE 754 def:contentReference[oaicite:76]{index=76}ities you must design around. :contentReference[oaicite:77]{index=77}

### 5a.3 Data pipeline validation
**AES-ML-4 (Schema + anomaly checks):** Training data MUST be validated at ingest:
types, shapes, ranges, missingness, label leakage tests, distribution drift vs baseline.

### 5a.4 Reproducibility & versioning
**AES-ML-5 (Repro manifest):** Every training run MUST emit a manifest:
- code version,
- dependency lock hashes,
- dataset version + checksum,
- seeds,
- hardware + driver stack,
- determinism mode,
- compiled kernels versions.

**AES-ML-6 (Train/serve parity):** Feature extraction MUST be shared between training and serving to prevent skew. :contentReference[oaicite:78]{index=78} :contentReference[oaicite:79]{index=79}

---

## 5b. Quantum Computing Standards

### 5b.1 Circuit design discipline
**AES-QC-ARCH-1 (Encoding contract):** Classical→quantum encoding (angle/amplitude/basis) MUST be documented with normalization requirements and failure modes.

### 5b.2 Noise-aware programming
**AES-QC-NOISE-1 (Backend-aware transpilation):** Compilation MUST include a hardware-specific pass manager/routing strategy and must avoid known bad qubits/links when calibration data indicates. :contentReference[oaicite:80]{index=80} :contentReference[oaicite:81]{index=81}

### 5b.3 Classical–quantum interface rules
**AES-QC-IF-1 (Deterministic checkpoints):** Hybrid loops MUST checkpoint every iteration (parameters + circuit + transpile s:contentReference[oaicite:82]{index=82}:contentReference[oaicite:83]{index=83}

### 5b.4 “Fault tolerant” claims are constrained
**AES-QC-QEC-1 (No false fault tolerance):** Claims of fault tolerance MUST state QEC assumptions and MUST NOT present NISQ results as fault-tolerant performance. :contentReference[oaicite:84]{index=84}

---

## 5c. Physics Simulation Standards

### 5c.1 Conservation laws are runtime-verified invariants
**AES-PHYS-1 (Invariant monitors):** Simulations MUST implement monitors for:
- mass/charge conservation (where applicable),
- energy conservation bounds (as appropriate),
- momentum/angular momentum,
- constraint sat:contentReference[oaicite:85]{index=85}ree fields).

Monitors MUST be telemetry signals and regression-gated.

### 5c.2 Symmetry enforcement
**AES-PHYS-2 (Symmetry by construction):** If the physics has symmetry (translation, rotation, gauge), the discretization SHOULD preserve it; if not feasible, the violation must be measured and bounded.

### 5c.3 Nu:contentReference[oaicite:86]{index=86}eria
**AES-PHYS-3 (Integrator correctness):** For Hamiltonian systems requiring long-time energy behavior, non-symplectic integrators are forbidden; symplectic methods are required. :contentReference[oaicite:87]{index=87}

-:contentReference[oaicite:88]{index=88}omputing Standards

### 5d.1 CPU-first / SIMD-first
**AES-HPC-1 (CPU-first hot paths):** Hot paths MUST be optimized for CPU first; SIMD vectorization is required when the operation is data-parallel and dominates time.

### 5d.2 Memory alignment and cache discipline
**AES-HPC-2 (Alignment contracts):** If aligned loads are used, the pointer alignment MUST be enforced and expressed as a contract (types, allocators, assertions). :contentReference[oaicite:89]{index=89}

```cpp
#include <immintrin.h>
#include <cstdlib>
#include <new>

struct AlignedBuffer {
  float* p;
  explicit AlignedBuffer(size_t n, size_t alignment = 32) {
    const size_t bytes = n * sizeof(float);
    p = static_cast<float*>(std::aligned_alloc(alignment, ((bytes + alignment - 1) / alignment) * alignment));
    if (!p) throw std::bad_alloc{};
  }
  ~AlignedBuffer() { std::free(p); }
  AlignedBuffer(const Al:contentReference[oaicite:90]{index=90}gnedBuffer& operator=(const AlignedBuffer&) = delete;
};

__m256 load8_aligned(const float* ptr) {
  // Contract: ptr must be 32-byte aligned.
  return _mm256_load_ps(ptr);
}
````



### 5d.3 OpenMP and determinism

**AES-HPC-3 (Explicit OpenMP clauses):** OpenMP pragmas MUST specify scheduling and thread counts explicitly in hot loops, and reductions MUST be analyzed for floating-point associativity and determinism. ([OpenMP][9])

---

## 6. VERIFICATION & TESTING FRAMEWORK

### 6.1 Test taxonomy (mandatory selection by risk)

AES defines these test categories:

* **Unit** (function/class isolation)
* **Integration** (module interactions)
* **Property-based** (invariants; Hypothesis/rapidcheck)
* **Fuzz** (input exploration; sanitizers required)
* **Benchmark** (performance regression)
* **Mutation** (test effectiveness)
* **End-to-end** (full pipeline)

The AAL gating matrix MUST include appropriate categories; AAL-0/AAL-1 require property + benchmark at minimum. 

### 6.2 Coverage requirements (hard gin statement + branch + MC/DC (where decision logic matters).

**AAL-1:** MUST maintain statement + branch.
Coverage regressions are merge-blocking. 

### 6.3 ML-specific testing (production-grade rubric)

MLta tests, model tests, infrastructure tests, and monitoring tests, consistent with production readiness thinking.  ([Google Research][10])

Minimum ML gates:

* single-batch overfit test
* gradient flow test (every parameter receives gradient)
* serialization round-trip equivalence within tolerance
* restart from checkpoint reproducibility

### 6.4 Quantum-specific testing

* ideal simulator expected outputs
* noisy simulation with documented noise model
* parameter sweep tests to detect identity/aliasing bugs  ([Google Quantum AI][11])

### 6.5 Statistical evidence (wh)

For probabilistic systems (ML, Monte Carlo, noisy quantum), claims of improvement MUST include:

* baseline vs treatment,
* confidence intervals,
* and stated assumptions.

---

## 7. OBSERVABILITY & TELEMETRY

### 7.1domain

Anvil MUST instrument the four Golden Signals: **latency, throughput, errors, saturation**.  ([Google SRE][2])

Domain examples (minimum required):
egrator_step_time, circuit_compile_time, p50/p95/p99

* **Throughput:** tokens/sec, GFLOPS, samples/sec, shots/sec
* **Errors:** NaNs/Infs, gradient failures, OOB faults, decode errors
* **Saturation:** thread utilization, cache miss rate, RSS, telemetry buffer pressure

### 7.2 Structured telemetry (OpenTelemetry-aligned)

**AES-OBS-1 (Structured logs):** Logging MUST be structured and MUST propagate trace context across boundaries.  ([OpenTelemetry][12])

### 7.3 Alerting thresholds (minimMUST be blocked if:

* error rate increases beyond threshold,
* p95 latency regresses beyond threshold,
* saturation crosses safe operating bounds,
* or non-finite events occur in training/simulation.

### 7.4 Chronicle protocol (performance + reliability evidence)

Chronicle is mandatory before/after telemetry logging for AAL-0/AAL-1 changes. 

---

## 8. AGENT WORKFLOW GOVERNANCE

### 8.1 Autonomy constraints

Anvil’s autonomy MUST be bounded by immutable governance, because agre and attack modes. 

### 8.2 Mandatory self-verification (AAL-0/AAL-1)

Before proposing a merge, Anvil MUST complete and attach:

* type checks,
* error contract audit,
* dead code scan,
* complexity bounds,
* security boundary review,
* numerical stability review,
* tests + coverage,
* traceability,
* FMEA top failure modes,
* Chronicle baseline/results. 

### 8.3 Red-team protocol (adversarial self-audit)

AAL-0/AAL-1 autonomous generation MUST trigger an adversarial “red team” agent that attempts to break correctness, safety, and securitynges MUST be rejected. 

### 8.4 Irreversible actions require explicit safeguards

Destructive repo actions MUST require human confirmation or a rollback window.

---

## 9. CODE REVIEW & CHANGE MANAGEMENT

### 9.1 Review requirements per AAL

Minimum review gates (can be stricter per repo):

* **AAL-0:** 2 independent reviewers + Anvil self-review; all gates pass; red-team log required
* **AAL-1:** 1 independent reviewer + Anvil self-review; all gates pass
* **AAL-2:** omous merge if all gates pass
* **AAL-3:** Anvil review; lints and doc checks should pass

(These are aligned with the existing Anvil review gating practice.) tandard
Reviews MUST use Conventional Comments labels (issue/suggestion/nitpick/question/thought) for clarity and automation. ([Conventional Comments][13])

### 9.3 Change risk scoring (CRS)

Each change MUST compute CRS (blast radius heuristic); high CRS triggers stronger gates (red-team, fuzz, extended coverage). 

# commit discipline

* Public APIs MUST follow semantic versioning rules. ([Semantic Versioning][14])
* Commits SHOULD follow Conventional Commits to enable automated release tooling. ([conventionalcommits.org][15])

---

## 10. ANTI-PATTERNS REGISTRY

**Anvil Mate these patterns.**

### 10.1 Universal “never do this”

* **Silent error swallowing** (`except: pass`, ignoring error returns). 
* **Shipping uncovered/untraceable code in AAL-0/AAL-1**. 
* **Unsafe parsing/deserialization on exposed boundaries**. 
* **Disabling linters/type-checkers inline instead of fixing root cause**. 

### 10.2 ML-specific anti-patterns

* **Custom gradients without gradcheck evidence**. 
* **Training on unvalidated data** (no schema/anomaly checks). *Silent NaN propagation** (no finite assertions). 

### 10.3 Quantum-specific anti-patterns

* **Not inspecting transpiled circuits; assuming compilation preserves semantics**. 
* **Claiming fault tolerance without explicit QEC assumptions**. 

### 10.4 HPC/systems anti-patterns

* **Assuming alignment and  on unknown pointers**. 
* **Parallel floating reductions without determinism/stability analysis**. 

---

## 11. APPENDICES

### Appendix A — Quick-reference checklists

**AAL-0 minimum merge checklist (must attach evidence):**

* Traceability chain complete (req→design→code→test→result) 
* MC/DC where decision logic matters; uncovered code resolved 
* Static analysis clean (UB/security/concurrency); CWE risks assessed
* Chronicle protocol complete (baseline vs after metrics)
* Supply chain integrity (pinned deps, provenance) ndependent review signoff 

### Appendix B — Seonal)

A practical decision tree is required for agent classification and defaults. ppendix C — Glossary (selected)

* **AAL:** Anvil Assurance Level (
* **Chronicle Protocol:** ce logging. 
* **Extraneous code:** Code not traceable to a requirement; includes dead code. ### Appendix D — Source anchors (for follow-up research)
  The AES synimum) these publicly accessible anchors:
* Software engineering lifecycle requirements and structure ([nodis3.gsfc.nasa.gov][16]) coverage practice and MC/DC expectations ([ldra.com][17])fication levels and requirements framing ([OWASP][4])
* CWE Top 25 weakness taxonomy ([CWE][5])
* Golden signals monitoring doctrine ([Google SRE][2])xis documentation taxonomy ([diataxis.fr][3])
* Sgning transparency ([SLSA][7])
* Quany simulation + transpilation APIs ([Google Quantum AI][18])h12
* ML production readiness testing rubric ([Google Research][10])n5search1
* IEEE 754 floatingnding) ([Arm Developer][19])

---

## Notes on lineage (internal)

This AES v2.0 is the **unified synthesis** and evolutiontandard drafts you provided, preserving the operational backbone (AAL hierarchy, Golden Signals, traotocol, cleanroom mindset, red-team protocol) while strengthening archpolicy (C++-first for kernels, Python for orchestration), and domain rigor.   



[1]: https://datatracker.ietf.org/doc/html/rfc2119?utm_source=chatgpt.com "RFC 2119 - Key words for use in RFCs to Indicate ..."
[2]: https://sre.google/sre-book/monitoring-distributed-systems/?utm_source=chatgpt.com "Chapter 6 - Monitoring Distributed Systems"
[3]: https://diataxis.fr/start-here/?utm_source=chatgpt.com "Start here - Diátaxis in five minutes"
[4]: https://owasp.org/www-project-application-security-verification-standard/?utm_source=chatgpt.com "OWASP Application Security Verification Standard (ASVS)"
[5]: https://cwe.mitre.org/top25/?utm_source=chatgpt.com "CWE Top 25 Most Dangerous Software Weaknesses"
[6]: https://csrc.nist.gov/pubs/sp/800/53/r5/upd1/final?utm_source=chatgpt.com "SP 800-53 Rev. 5, Security and Privacy Controls"
[7]: https://slsa.dev/?utm_source=chatgpt.com "SLSA • Supply-chain Levels for Software Artifacts"
[8]: https://openssf.org/community/sigstore/?utm_source=chatgpt.com "Sigstore – Open Source Security Foundation"
[9]: https://www.openmp.org/wp-content/uploads/openmp-examples-5.1.pdf?utm_source=chatgpt.com "OpenMP Application Programming Interface Examples"
[10]: https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/?utm_source=chatgpt.com "The ML Test Score: A Rubric for ML Production Readiness ..."
[11]: https://quantumai.google/cirq/simulate/noisy_simulation?utm_source=chatgpt.com "Noisy Simulation | Cirq"
[12]: https://opentelemetry.io/docs/specs/otel/logs/?utm_source=chatgpt.com "OpenTelemetry Logging"
[13]: https://conventionalcomments.org/?utm_source=chatgpt.com "Conventional Comments"
[14]: https://semver.org/?utm_source=chatgpt.com "Semantic Versioning 2.0.0 | Semantic Versioning"
[15]: https://www.conventionalcommits.org/en/v1.0.0/?utm_source=chatgpt.com "Conventional Commits"
[16]: https://nodis3.gsfc.nasa.gov/displayDir.cfm?c=7150&s=2D&t=NPR&utm_source=chatgpt.com "NPR 7150.2D - main - NODIS Library - NASA"
[17]: https://ldra.com/ldra-blog/do-178c-structural-coverage-analysis/?utm_source=chatgpt.com "DO-178C & Structural Coverage Analysis"
[18]: https://quantumai.google/cirq/google/best_practices?utm_source=chatgpt.com "Best practices | Cirq"
[19]: https://developer.arm.com/documentation/den0042/latest/Floating-Point/Floating-point-basics-and-the-IEEE-754-standard?utm_source=chatgpt.com "Floating-point basics and the IEEE-754 standard"
