The Anvil Engineering Standard
Preamble
Purpose and scope
The Anvil Engineering Standard defines mandatory engineering rules, verification expectations, and
governance controls for Anvil (an AI coding agent) operating with a semantic code intelligence layer
(“Saguaro”) across large repositories. Its purpose is to produce code and engineering artifacts that are
demonstrably correct, secure, and maintainable under safety-style evidence, while remaining practical for
fast-moving research and development. 1
This standard is explicitly cross-domain: it covers general software engineering (Python 3.12+ and C++17
systems programming), deep learning and machine learning, quantum computing (circuits, hybrid
workflows, and error-correction-aware programming), physics simulation (numerical methods and
invariants), and high-performance computing (CPU-first, SIMD/AVX2 discipline, OpenMP, cache and memory
alignment). 2
How to read and apply this standard
This document uses the normative keywords MUST, MUST NOT, SHALL, SHOULD, and MAY per the IETF
requirement-language convention. Any deviation from a MUST/MUST NOT requirement requires explicit
documented rationale, risk acceptance, and approval commensurate with the code’s assurance level. 3
Three classifications are used throughout:
• Mandatory: “MUST / SHALL / MUST NOT”. Required unless formally waived with traceable risk
acceptance.
• Advisory: “SHOULD / SHOULD NOT”. Strongly recommended; deviations require justification in code
review.
• Aspirational: “MAY”. Optional improvements; adopted when economically justified.
Rationale is provided for major requirements in a single line to keep the standard scannable while
preserving “why.” 4
1Authoritative sources synthesized
This standard unifies principles from aerospace software assurance and traceability requirements, avionics
development assurance, reliability engineering, safety lifecycle frameworks, secure coding standards, and
modern reliability/observability practice:
• National Aeronautics and Space Administration 5 foundational requirements cover systematic
software assurance, software safety, independent verification and validation, software classification,
and bi-directional traceability. 6
• Federal Aviation Administration 7 -aligned avionics safety practice contributes Development
Assurance Level thinking, structural coverage expectations (including MC/DC for the highest
criticality), and explicit treatment of extraneous/dead and deactivated code discovered by coverage
analysis. 8
• Google 9 Site Reliability Engineering practice contributes Golden Signals, SLO-error-budget
governance, and toil reduction principles. 10
• International Business Machines Corporation 11 Cleanroom software engineering contributes
defect prevention, correctness-focused development, and statistical certification concepts. 12
• CMMI Institute 13 contributes quantitative process management and continuous improvement
expectations for high-maturity engineering organizations. 14
• Fédération Internationale de l'Automobile 15 telemetry and data-acquisition regulation contributes
real-time telemetry discipline, controlled logging buffers, and independent oversight access
concepts. 16
• JEDEC Solid State Technology Association 17 contributes statistical rigor, confidence-driven
qualification, and “no failures with confidence bounds” reasoning. 18
• MISRA 19 and the Software Engineering Institute 20 CERT secure coding standards contribute
undefined-behavior avoidance, defensive coding, and vulnerability prevention rules. 21
• International Organization for Standardization 22 , International Electrotechnical Commission 23 ,
and Institute of Electrical and Electronics Engineers 24 contribute functional safety lifecycle
concepts and software lifecycle / SQA process expectations. 25
• Open Web Application Security Project 26 , MITRE Corporation 27 , and National Institute of
Standards and Technology 28 contribute security verification levels, weakness taxonomies, and a
catalog of security/privacy controls. 29
• Modern language and ecosystem standards provide concrete style and correctness guidance:
Python PEP style and typing, Google style guides, and the C++ Core Guidelines. 30
• Domain engineering references include: Google’s ML Test Score and Rules of ML, and official Qiskit
and Cirq documentation for circuit construction and compilation discipline. 31
• Numerical reliability is grounded in IEEE‑754-oriented practice and established guidance on floating
point error and stability analysis. 32
• Supply-chain security is grounded in SLSA provenance/levels and Sigstore’s design for signing and
transparency. 33
Severity and assurance framework
Unified Anvil assurance levels
Anvil uses a unified risk and assurance model that merges avionics DAL thinking, functional-safety severity
stratification, and the project’s existing P0–P3 severity matrix into one operational framework. The goal is to
2ensure verification effort is proportional to consequences and exposure, while keeping rules consistent
across domains. 34
Anvil Assurance Level (AAL) is evaluated per change, not per repository. The AAL chosen for a change
MUST be recorded in the change description and traceable to the impacted requirements and hazards,
using the repository’s traceability mechanism. 35
Anvil
Assurance
Level
AAL‑0
(Catastrophic)
AAL‑1
(Critical)
AAL‑2 (Major)
Existing
severityTypical repository
scope
P0SIMD kernels,
memory
allocators,
schedulers,
safety/security
gates,
cryptography,
circuit compilers,
numerical
integrators used
for “ground
truth,” and any
code that can
silently corrupt
results
P1Core training
loops, autograd
extensions,
quantum-
classical
orchestration,
data ingestion
that can poison
training, high-
impact services
P2Tooling,
configuration
loaders, model
architecture
experiments,
non-hot-path
libraries, batch
pipelines
DAL
analogyISO 26262
analogyIEC
61508
analogySecurity
verification
analogy
DAL‑AASIL D-
style
highest
rigor
context
(risk-based
functional
safety)SIL 4-
style
highest
rigor
contextOWASP ASVS
Level 3 “high
value/high
assurance/
high safety”
apps
DAL‑BASIL C-
style
contextSIL 3-
style
contextASVS Level 2
“most
applications”
DAL‑CASIL B-
style
contextSIL 2-
style
contextASVS Level 1
“first steps/
portfolio
view”
3Anvil
Assurance
Level
AAL‑3 (Minor)
Existing
severityTypical repository
scopeDAL
analogyISO 26262
analogyIEC
61508
analogySecurity
verification
analogy
P3Docs, notebooks,
non-executable
reference
material; pure
metadata
changesDAL‑D/
EASIL A/QM
contextSIL 1/
QM
contextBaseline
security
hygiene
Rationale: DAL-style stratification ties assurance to failure impact; functional safety frameworks emphasize
risk-driven safety lifecycle selection; ASVS explicitly defines verification levels for different risk profiles. 36
AAL selection criteria
A change MUST be classified at the highest AAL triggered by any of these conditions:
• Silent correctness loss: If plausible defects can silently corrupt scientific conclusions, model
weights, circuit outputs, or physical invariants, the change is at least AAL‑1; if outcomes can be
safety- or mission-catastrophic (e.g., foundational kernels, core verification infrastructure), it is
AAL‑0. 37
• Hazard linkage: If the change touches hazards, hazard controls, hazard mitigations, or safety-critical
behavior, it MUST be at least AAL‑0 or AAL‑1 and MUST maintain bi-directional traceability between
software requirements and hazards. 38
• Attack surface: If the change introduces or modifies a network boundary, deserialization, plugin
execution, dependency acquisition, or any interface that can be exploited, it MUST be at least AAL‑1
and meet the security verification baseline. 39
• Performance-critical hot path: If the change touches code where latency/throughput/saturation
are first-order constraints (SIMD, OpenMP parallel loops, allocators, schedulers), it MUST be at least
AAL‑0. 40
Rationale: NASA requires hazard-linked traceability; DO‑178 practice ties the highest assurance to the most
severe failure conditions; SRE ties reliability work to user-impact metrics and overload risk. 41
Verification required per AAL
All AAL levels require basic engineering hygiene (lint, formatting, unit tests where applicable), but higher
levels require stronger evidence, independence, coverage, and statistical rigor. 42
Evidence / control
Traceability
(req→design→code→test→result)
Independent review
AAL‑0AAL‑1AAL‑2AAL‑3
MUST, end-to-endMUSTSHOULDMAY
MUST (not author;
independent
competence)MUSTSHOULDMAY
4Evidence / controlAAL‑0AAL‑1AAL‑2AAL‑3
Structural coverageMUST include MC/
DC where branching
logic is safety/
liveness-criticalMUST include
branch coverage
(or justified
alternatives)SHOULDMAY
Dead/extraneous codeMUST NOT exist in
shipped artifact;
MUST be resolved if
found by coverageMUST NOT exist
in shipped
artifactSHOULD
NOTMAY
Static analysisMUST (security + UB
+ concurrency +
numerical checks)MUSTSHOULDMAY
FMEA/FTA for changeMUSTSHOULDMAYMAY
Performance baselines and
regression gatesMUSTMUSTSHOULDMAY
Crypto / supply chain provenanceMUST show
provenance for
produced artifactsMUSTSHOULDMAY
Rationale: NASA explicitly requires bi-directional traceability and hazard linkage; FAA-aligned guidance
asserts MC/DC and structural coverage as required for Level A and treats extraneous/dead code discovered
through structural coverage as an issue requiring resolution; Cleanroom and JEDEC emphasize defect
prevention and statistical evidence rather than anecdotes. 43
Universal coding mandates
Traceability and evidence closure
Requirement: All non-trivial code MUST be traceable as: requirement → design decision → code change
→ test(s) → verification result(s). For AAL‑0/AAL‑1 changes, this trace MUST be explicit and machine-
navigable (e.g., requirement IDs in code comments, structured metadata, or repository-standard linkage).
35
Requirement: Where software requirements link to hazards, Anvil MUST maintain bi-directional traceability
between software requirements and software-related hazards and controls. 44
Requirement: Verification is not complete until evidence is archived and reproducible (test logs, seeds,
build provenance, performance traces). Evidence MUST be sufficiently detailed that an independent agent
can replay the verification and obtain matching conclusions. 45
Rationale: NASA explicitly mandates maintaining bi-directional traceability across requirements, hazards,
design, code, verification, and nonconformances; SLSA requires provenance to understand what built an
artifact and with what inputs. 46
5Cleanroom defect prevention and “no symptom masking”
Requirement: Code MUST be engineered to prevent defects at the source. Anvil MUST NOT “mask” defects
with broad exception catches, silent fallbacks, or disabling checks; instead, it MUST isolate root cause,
implement a corrective fix, and add a regression test that would fail without the fix. 47
Requirement: For AAL‑0/AAL‑1 components, tests MUST demonstrate correctness on representative
operational distributions (not only developer-curated examples). Where failures are probabilistic, tests
MUST quantify confidence and assumptions. 48
Rationale: Cleanroom explicitly prioritizes defect prevention and statistical certification over debugging;
NASA’s standards emphasize systematic assurance and IV&V; JEDEC-style qualification thinking uses
confidence and sampling assumptions. 49
Zero dead/extraneous code
Requirement: Shipped artifacts MUST contain no extraneous code (code not traceable to a requirement)
and no dead code (code present due to design error and not traceable). If coverage analysis reveals
unexecuted code, Anvil MUST resolve it by: (a) adding tests derived from requirements, (b) correcting
requirements, or (c) removing the code, as appropriate. 50
Rationale: FAA-aligned DO‑178C differences guidance explicitly treats extraneous code as a superset of
dead code (not traceable to a requirement) and ties its resolution to structural coverage analysis resolution
activities. 51
Complexity and performance discipline
Requirement: For any hot path (defined as contributing materially to latency, throughput, or saturation),
algorithms MUST be O(1), O(log n), or O(n) in the dominant dimension(s). O(n²) or worse in hot paths
MUST NOT be introduced unless accompanied by a formally documented proof that the effective n is
strictly bounded and enforced at runtime.
52
Requirement: Performance changes MUST follow the Chronicle protocol: (1) establish baseline metrics, (2)
apply change, (3) record post-change metrics, and (4) attach results to the change record. For AAL‑0/AAL‑1,
Chronicle results are gating. 53
Rationale: SRE’s Golden Signals explicitly treat latency/traffic/errors/saturation as the minimal monitoring
basis, and release policy is often governed by objective SLO/error budget evidence; FIA data acquisition
rules reflect strict telemetry and logging discipline with regulated buffers and oversight access. 54
Error handling philosophy
Anvil uses a unified error-handling philosophy that merges three ideas:
• Explicit errors for recoverable conditions (Go’s “errors are values”): recoverable failure MUST be
surfaced in return types (or typed result objects) and MUST NOT be silently ignored. 55
6• Crash fast for invariant violations (“let it crash” in a supervised environment): when an internal
invariant is violated, the system SHOULD fail loudly and early rather than continue in a corrupted
state; recovery MUST be handled at a higher level with supervision/restart or transactional rollback,
not by burying failures. 56
• Typed, structured errors (Rust’s Result): errors MUST carry structured context that supports
programmatic handling, classification, and observability. 57
Mandatory rule: All public APIs MUST document and implement a deterministic error contract: what errors
can occur, how they are surfaced, and what state guarantees hold on failure. 58
Mandatory rule: Error paths MUST be tested. For AAL‑0/AAL‑1, tests MUST demonstrate correctness for
failure-handling logic under realistic fault injection (I/O errors, NaNs/non-finites, timeouts, empty data
windows). 59
Rationale: The Zen of Python explicitly warns “errors should never pass silently”; Erlang supervision trees
formalize fault containment and restart; Rust explicitly distinguishes recoverable errors and panics; NASA
and SRE practice both require reliable failure detection and operational monitoring. 60
Documentation requirements
Requirement: Documentation MUST be written in a Diátaxis-compatible structure: tutorials (learning),
how-to guides (task execution), reference (complete API/contracts), and explanation (design rationale).
Every public API MUST have reference documentation that fully specifies inputs, outputs, errors, invariants,
and side effects. 61
Rationale: Diátaxis provides a durable documentation taxonomy; PEP 257 explicitly expects docstrings to
summarize behavior and document arguments, return values, side effects, and exceptions. 61
Security baseline
Requirement: Every component MUST implement a baseline security posture proportional to AAL. At
minimum:
• AAL‑0/AAL‑1 MUST map security requirements to ASVS Level 2 or Level 3, as applicable, and MUST
explicitly mitigate CWE Top 25 weakness classes relevant to the component’s surface area. 62
• Security controls selection MUST align with an explicit control catalog and risk-management framing
(NIST SP 800‑53-style). 63
Requirement: Input boundaries MUST be treated as hostile. Parsing, deserialization, templating, and
format-string handling MUST be designed to prevent injection-style vulnerabilities and memory safety
hazards. 64
Rationale: ASVS provides requirements for secure development and verification; CWE Top 25 identifies
common high-impact weakness patterns; NIST SP 800‑53 provides a catalog of controls for organizational
risk protection. 29
7Dependency management and supply chain integrity
Requirement: Dependencies MUST be pinned to known versions for reproducibility, and for AAL‑0/AAL‑1
MUST be integrity-verified via hashes or equivalent cryptographic mechanisms. For Python packaging,
hash-checking mode MUST be used for deployment-grade installs whenever feasible. 65
Requirement: Build outputs MUST be traceable via provenance. For AAL‑0/AAL‑1 artifacts, Anvil MUST
produce and store SLSA-style provenance describing what built the artifact, what process was used, and
what inputs were used. 66
Requirement: Releases MUST support verification (signing and tamper evidence). For AAL‑0/AAL‑1 releases,
artifacts SHOULD be signed and recorded in a transparency-log style mechanism consistent with Sigstore’s
model. 67
Rationale: pip explicitly supports forced hash-checking mode; SLSA defines levels and provenance to
incrementally improve supply chain security; Sigstore is designed around signing and transparency to
reduce reliance on long-lived private keys. 68
Language and domain standards
Python standards
Style and readability
• Python code MUST follow PEP 8 formatting and naming conventions unless the repository’s
formatter enforces a stricter rule. 69
• Public modules, classes, and functions MUST have docstrings following PEP 257 content
expectations; docstrings SHOULD be Google-style when used with Sphinx/Napoleon. 70
Rationale: PEP 8 and PEP 257 define conventions for readability and interface documentation.
71
Typing and contracts
• All public Python APIs MUST provide type annotations using the PEP 484 framework, including return
types; module-level and class-level fields SHOULD be annotated per PEP 526. 72
• For AAL‑0/AAL‑1 Python, Anvil MUST run a static type checker in CI and MUST fix or justify all type
errors (no “ignore everything” configs).
73
Rationale: PEP 484 aims to enable static analysis and safer refactoring; NASA requires systematic
engineering evidence and controlled processes. 73
Error handling
• Public functions MUST return structured error results (e.g., Result[T, E] pattern) or raise
documented exceptions; they MUST NOT silently return sentinel values like None on error unless
None is a valid success value and the API contract states it unambiguously.
8
74• Logging MUST be structured, and error logs MUST include stable event identifiers and context fields
needed for triage. 75
Rationale: “Errors should never pass silently” is a Python design maxim; modern observability expects
structured telemetry for analysis. 76
Numerical safety
• Any numeric code that can produce non-finite values (NaN/Inf) MUST check and surface non-finites
at boundaries (inputs, outputs, and key invariants), and MUST treat unexpected non-finites as errors
in AAL‑0/AAL‑1. 77
• Summations over large sequences SHOULD use numerically stable strategies (pairwise /
compensated summation) when accuracy matters, and MUST document error expectations. 78
Rationale: IEEE‑754 defines NaN/Inf behavior and rounding; Goldberg and Higham explain why rounding
and conditioning matter and how stable algorithms reduce error growth. 79
Concrete example: explicit contract + typed errors
from __future__ import annotations
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar
T = TypeVar("T")
@dataclass(frozen=True)
class Error:
code: str
# stable machine-readable identifier
message: str
# human-readable summary
context: dict[str, object]
@dataclass(frozen=True)
class Result(Generic[T]):
value: Optional[T] = None
error: Optional[Error] = None
def unwrap(self) -> T:
if self.error is not None:
raise RuntimeError(f"{self.error.code}: {self.error.message}
({self.error.context})")
assert self.value is not None
return self.value
9Rationale: Explicit failure channels prevent silent error loss and enable structured observability and
verification. 80
C and C++ standards
Core C++ safety and resource discipline
• C++ MUST follow modern safety profiles: RAII for all resources, bounded access, and lifetime safety;
raw owning pointers MUST NOT exist in AAL‑0/AAL‑1 code. 81
• The codebase MUST define and enforce a single error-handling strategy for C++ modules (status
returns, expected<T,E> -style results, or exceptions with strict boundaries). Performance-critical
or safety-critical components SHOULD avoid exceptions and prefer explicit results to reduce
uncontrolled stack unwinding risk. 82
Rationale: The C++ Core Guidelines emphasize RAII and tool-enforced safety profiles; DO‑178-derived
practice demands explicit verification and predictable behavior under high assurance. 83
Undefined behavior and defensive coding
• For C/C++ code, undefined behavior MUST be treated as a security and correctness defect, not a
“performance trick.” For AAL‑0/AAL‑1, UB-sanitizers and static analysis MUST be part of CI. 84
• Buffer bounds, integer overflow/wraparound, and uninitialized reads MUST be prevented by
construction (safe types, checked arithmetic, explicit initialization) rather than “best effort.” 85
Rationale: MISRA’s aim is to prevent unsafe/undefined behaviors; CERT rules explicitly address uninitialized
memory and other UB sources; CWE Top 25 includes out-of-bounds and memory safety weaknesses as
high-impact classes. 86
Includes, naming, and style
• C++ code SHOULD follow the Google C++ Style Guide for include order and consistent naming unless
the repository defines stricter conventions. 87
Rationale: Consistent conventions reduce review overhead and improve long-term maintainability in large
codebases. 88
SIMD and alignment (HPC-critical)
• SIMD paths MUST explicitly define alignment assumptions and MUST use aligned allocation or safe
unaligned loads consistently with measured performance impact. Any intrinsic that requires
alignment MUST NOT be used on potentially unaligned memory. 89
• Production SIMD kernels MUST ship with a scalar reference implementation used for correctness
cross-checks and differential testing. 90
Rationale: SIMD programming guidance emphasizes alignment constraints and careful load/store strategy;
high-assurance practice requires an oracle/reference path to validate optimized implementations. 91
Concrete example: aligned allocation + AVX2-safe load strategy
10#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <immintrin.h>
#include <stdexcept>
struct AlignedBuffer {
float* p = nullptr;
std::size_t n = 0;
explicit AlignedBuffer(std::size_t count) : n(count) {
// 32-byte boundary for AVX2-friendly alignment.
std::size_t bytes = n * sizeof(float);
std::size_t alignment = 32;
// C++17 doesn't standardize aligned_alloc portability everywhere; wrap per
platform in real code.
p = static_cast<float*>(std::aligned_alloc(alignment, ((bytes + alignment -
1) / alignment) * alignment));
if (!p) throw std::bad_alloc{};
}
~AlignedBuffer() { std::free(p); }
AlignedBuffer(const AlignedBuffer&) = delete;
AlignedBuffer& operator=(const AlignedBuffer&) = delete;
};
__m256 load8_aligned(const float* ptr) {
// Contract: ptr must be 32-byte aligned.
return _mm256_load_ps(ptr);
}
Rationale: Alignment discipline avoids undefined behavior and enables predictable performance; Intel
intrinsics references and SIMD teaching materials emphasize alignment as a primary issue. 89
Quantum circuit standards
Anvil’s quantum standards focus on correctness under compilation/transpilation, explicit noise awareness,
and rigorous separation of classical-quantum responsibilities.
Circuit construction, representation, and inspection
• Circuits MUST be represented as explicit circuit objects (not ad-hoc arrays of gates) and MUST be
inspectable after optimization/transpilation passes. Anvil MUST review optimized circuits for
unintended consequences and semantic drift. 92
• Parameterized circuits MUST define stable parameter ordering and explicit bindings; parameter
sweeps MUST be represented using library primitives (e.g., Cirq sweeps). 93
11Rationale: Cirq best practices explicitly recommend inspecting the circuit after optimization; IBM’s circuit
documentation formalizes circuit construction patterns. 94
Noise-aware programming and simulation
• Any code intended for NISQ hardware MUST include a noise-aware validation step: (a) ideal
simulator results, (b) noisy simulation using a documented noise model, and (c) a hardware run (or
explicit waiver) with measurement-error considerations. 95
• Noise models MUST be documented as part of traceability evidence: what Kraus/channel model (or
backend noise model) was assumed and why. 96
Rationale: Cirq explicitly documents noisy simulation via operator-sum/Kraus representations; IBM provides
guidance for running circuits on hardware and emphasizes measurement/primitive choices. 97
Gradients and hybrid optimization integrity
• When implementing variational quantum algorithms, gradients MUST be computed either via
validated parameter-shift rules or via a documented alternative with correctness evidence. 98
• Hybrid classical-quantum loops MUST have deterministic checkpoints: parameters, random seeds,
backend/transpiler settings, and compiled circuit artifacts per step MUST be recorded for replay at
AAL‑0/AAL‑1. 99
Rationale: Parameter-shift rules are a primary gradient mechanism in variational quantum algorithms;
reproducible compilation settings are necessary because transpilation is not always semantically trivial
across hardware constraints. 100
Quantum error correction awareness
• Quantum programs that claim “fault-tolerant readiness” MUST explicitly state assumptions about
quantum error correction (QEC) and encode/logical mapping; they MUST NOT present NISQ results
as fault-tolerant performance. 101
Rationale: IBM’s QEC discussions emphasize that QEC encodes information into physical qubits to protect
against errors and distinguishes near-term codes and fault-tolerant pathways. 101
Concrete example: Cirq optimization inspection checkpoint
import cirq
import sympy
q = cirq.LineQubit(0)
theta = sympy.Symbol("theta")
circuit = cirq.Circuit(
cirq.rx(theta)(q),
cirq.measure(q, key="m"),
)
12# Compile / optimize (transformers), then inspect
optimized = cirq.optimize_for_target_gateset(circuit, gateset=cirq.CZPowGate)
print("Original:\n", circuit)
print("Optimized:\n", optimized)
Rationale: Cirq best practices emphasize inspecting the circuit after optimization to ensure compilation did
not introduce unintended consequences. 102
Verification, testing, and observability
Test taxonomy
All code MUST fit into a test taxonomy. Each component MUST define which of the following are required
and what evidence constitutes closure:
• Unit tests (logic-level contracts)
• Integration tests (boundary contracts: filesystem, GPU/CPU kernels, quantum backends)
• Property-based tests (invariant enforcement)
• Fuzz tests (parser/decoder/interface robustness)
• Performance/benchmark tests (latency/throughput/saturation regressions)
• Statistical tests (for probabilistic robustness and confidence-based claims)
Rationale: NASA mandates systematic lifecycle evidence and verification results; SRE practice identifies
minimal operational signals and uses objective policy gates; Cleanroom emphasizes statistical testing and
certification. 103
Coverage requirements and structural coverage
• Coverage MUST be meaningful and aligned with assurance level. For AAL‑0, structural coverage
MUST be strong enough to validate decision logic, including MC/DC for safety-critical or liveness-
critical branching. 104
• Coverage analysis MUST trigger a “no extraneous code” workflow: uncovered code must be tested,
justified as deactivated with verified non-execution, or removed. 51
Rationale: FAA-aligned practice requires MC/DC for Level A and treats structural coverage as a check on
requirements-based testing; DO‑178C differences guidance expands and clarifies extraneous/dead code
resolution. 8
ML-specific verification
ML test score alignment
• Production ML systems SHOULD implement ML Test Score-style tests across data, model, and
infrastructure readiness, including monitoring/rollback needs. AAL‑1 ML pipelines MUST implement
them unless waived with explicit risks. 105
13Rationale: The ML Test Score explicitly presents a rubric of tests and monitoring needs to quantify ML
production readiness and reduce technical debt. 105
Data pipeline validation
• Training and serving data MUST be validated against an explicit schema: descriptive statistics,
inferred/declared schema, anomaly detection, drift/skew checks. 106
Rationale: TensorFlow Data Validation explicitly supports statistics, schema inference, and anomaly
detection, including drift and training-serving skew checks. 107
Reproducibility and determinism
• Training runs MUST record all sources of randomness, determinism flags, and environment versions.
For PyTorch-based systems, deterministic algorithm mode MUST be used for AAL‑1 unless
performance constraints are formally justified. 108
Rationale: PyTorch documentation explicitly supports deterministic algorithm enforcement and describes
randomness sources affecting reproducibility. 109
Gradient integrity
• Custom autograd functions and gradients MUST be validated with finite-difference gradcheck (or
equivalent) for AAL‑1/AAL‑0 ML kernels. 110
• Gradient clipping MUST be explicit and MUST specify behavior on non-finite gradients (error if non-
finite for high assurance). 111
Rationale: PyTorch provides gradcheck/gradgradcheck for analytical vs numerical gradient validation and
clip_grad_norm_ for standardized gradient clipping with non-finite error options. 112
Quantum-specific verification
• Quantum results MUST be validated against simulators for small circuits, and where noise matters,
by noisy simulation. 113
• Circuit transpilation/optimization MUST be followed by semantic checks: invariants, expected
distributions/expectation values, and gate counts/depth effects must be recorded. 114
Rationale: Cirq best practices and simulation docs emphasize inspection and simulation; IBM transpiler docs
formalize optimization-level behavior. 115
Observability and telemetry
Golden Signals as the universal overlay
Every production-relevant subsystem MUST emit metrics that map to the four Golden Signals: latency,
traffic/throughput, errors, saturation. The exact metric names differ by domain, but the taxonomy MUST
be preserved. 116
14Rationale: The SRE book explicitly frames dashboards around the four Golden Signals and treats them as
the minimal useful monitoring set. 116
Domain-specific Golden Signals mapping
• HPC kernels: latency = kernel runtime; throughput = FLOP/s or items/s; errors = incorrect outputs /
NaNs; saturation = SIMD lane utilization, memory bandwidth pressure. 117
• ML training: latency = step_time; throughput = samples/s or tokens/s; errors = non-finite loss/
gradients; saturation = GPU/CPU utilization and memory. 118
• Quantum execution: latency = circuit compile + run time; throughput = shots/s or circuits/s; errors =
invalid results / calibration drift; saturation = queue depth, backend utilization. 119
• Telemetry-intensive systems: logs and event buffers MUST be controlled and auditable; data
access for oversight MUST be supported. 120
Rationale: FIA technical regulations explicitly require access to logged data and real-time telemetry data and
constrain buffer clearing and oversight access, illustrating rigorous telemetry governance. 16
OpenTelemetry standardization
• Telemetry SHOULD be emitted using OpenTelemetry-compatible structured logs, metrics, and traces,
including semantic conventions that allow cross-service correlation. 121
Rationale: OpenTelemetry defines cross-language observability primitives and semantic conventions for
correlation. 121
Error budgets as change gates
• Services with defined SLOs MUST implement error budgets, and AAL‑0/AAL‑1 releases SHOULD be
gated by error budget burn rate policy (slow/stop feature rollouts when budgets are exhausted). 122
Rationale: Google’s SRE workbook explicitly defines error budgets as 1 − SLO and frames error budget policy
as the operational decision tool for balancing reliability and velocity. 123
Agent workflow governance and change control
Rules for AI coding agents
Generation guardrails
• Anvil MUST NOT introduce changes without an explicit task objective and acceptance criteria;
acceptance criteria MUST be translated into tests or verification steps. 124
• For AAL‑0/AAL‑1, Anvil MUST perform a self-audit that includes: traceability check, structural
coverage check plan, security review against CWE Top 25 relevant classes, and regression/
performance gates per Chronicle. 125
Rationale: NASA requires traceability and recorded evidence; SRE requires objective monitoring and policy
gates; CWE Top 25 enumerates common impactful weakness patterns. 126
15Independence and review
• AAL‑0 changes MUST be reviewed by at least one reviewer who is not the author agent and who is
competent in the domain (HPC, ML, quantum, safety/security). 127
• Reviewers MUST apply the “code health over time” standard and require tests commensurate with
risk. 128
Rationale: NASA and high-assurance practice emphasizes independent verification; Google’s engineering
practices frame code review as improving overall code health over time. 129
Change size and reversibility
• Changes SHOULD be small and reversible; large refactors MUST be decomposed into invariant-
preserving steps, each with tests and Chronicle baselines. 130
Rationale: Google’s author guide emphasizes small changes; Cleanroom emphasizes incremental
development with certification feedback on increments. 131
Change risk assessment and technical debt governance
Risk assessment
• Every AAL‑0/AAL‑1 change MUST include a brief risk analysis: failure mode description, detection
mechanism, and mitigation. For safety-relevant software, this must include hazard linkage. 132
Rationale: NASA explicitly requires hazard traceability; functional safety standards are risk-driven and
require lifecycle thinking. 133
Technical debt management
• Debt MUST be tracked explicitly and categorized using a structured model (SQALE-style
quantification + Fowler-style quadrant classification) for prioritization. 134
• Debt that impacts correctness, safety, or security MUST NOT be scheduled as “later” without a formal
risk acceptance and a time-bounded mitigation plan. 135
Rationale: SQALE frames technical debt management as definition, calculation, and monitoring; Fowler’s
quadrant frames debt as deliberate/inadvertent and reckless/prudent. 136
API stability rules
• Public APIs MUST be treated as contracts. Observable behavior changes MUST be considered
breaking unless explicitly versioned and communicated. 137
• Inputs SHOULD be validated robustly, but the “be liberal in what you accept” principle MUST be
applied with security awareness: be conservative in what you emit, and validate what you accept;
avoid permissiveness that leads to ambiguity or attack surface expansion. 138
16Rationale: Hyrum’s Law warns that users depend on all observable behaviors; Semantic Versioning defines
compatibility signaling; RFC 1122 explicitly states the robustness principle and warns one misbehaving host
can impact many. 139
Anti-patterns and appendices
Anti-patterns registry
The following patterns are explicitly forbidden unless approved via AAL-appropriate waiver:
Universal “never do this”
• Silent error swallowing: broad except: pass , ignoring error returns, or logging without
surfacing failure. 140
• Shipping uncovered/untraceable code in AAL‑0/AAL‑1 artifacts. 141
• Nondeterministic training without recording randomness sources, especially for AAL‑1 ML
systems. 109
• Unsafe input parsing (format strings, injection-prone templating, unsafe deserialization) on
exposed boundaries. 142
HPC/Systems-specific
• Assuming alignment without enforcing it, then using aligned-load intrinsics on unknown pointers.
143
• Parallel reductions without associativity/precision analysis for floating point sums where
determinism and numeric stability matter. 144
Quantum-specific
• Failing to inspect optimized/transpiled circuits and treating compilation as semantics-preserving
by default. 145
• Claiming fault tolerance without explicit QEC assumptions. 101
ML-specific
• Custom gradients without gradcheck evidence. 110
• Training on unvalidated data without schema/anomaly checks.
107
Rationale: These anti-patterns correspond to known failure modes under high assurance: silent corruption,
unverifiable behavior, and common high-impact weaknesses. 146
Quick-reference checklists
AAL‑0 change checklist (minimum)
AAL‑0 changes MUST not be merged unless all items are satisfied and evidence is attached:
• Traceability chain complete (req→design→code→test→result) and hazards linked where relevant.
44
17• Structural coverage plan includes MC/DC where decision logic matters; uncovered code resolved
(test/remove/justify deactivated). 8
• Static analysis clean (UB, security, concurrency); top CWE risks assessed and mitigated. 147
• Chronicle protocol: baseline vs after metrics for latency/throughput/errors/saturation. 148
• Supply chain: pinned deps, integrity checks, provenance for built artifacts. 65
• Independent review completed, with explicit signoff. 149
ML training loop integrity checklist (AAL‑1)
• Deterministic configuration documented; seeds and environment versions recorded;
nondeterminism waivers documented. 109
• Data validation: schema, anomalies, drift/skew checks. 107
• Gradient integrity: gradcheck for custom gradients; non-finite detection in loss/gradients; clipping
behavior specified. 150
• Monitoring: Golden Signals mapped for training infrastructure; alerts defined for non-finites and
throughput collapse. 148
Glossary of synthesized terms
• AAL (Anvil Assurance Level): Unified assurance tier used to select verification and governance
requirements. 151
• Chronicle protocol: Mandatory baseline→change→result logging for performance and operational
metrics, aligned with SRE monitoring disciplines and telemetry governance analogies. 152
• Extraneous code: Code not traceable to a system or software requirement; a superset that includes
dead code. 153
• Toil: Manual, repetitive, automatable work tied to running a production service that scales linearly
with service growth. 154
• Provenance: Verifiable information describing where, when, and how an artifact was produced,
used for supply chain integrity. 155
1
6
17
https://standards.nasa.gov/standard/NASA/NASA-STD-87398
https://standards.nasa.gov/standard/NASA/NASA-STD-87398
2
https://nodis3.gsfc.nasa.gov/displayDir.cfm?c=7150&s=2D&t=NPR
https://nodis3.gsfc.nasa.gov/displayDir.cfm?c=7150&s=2D&t=NPR
3
https://datatracker.ietf.org/doc/html/rfc2119
https://datatracker.ietf.org/doc/html/rfc2119
4
https://standards.ieee.org/ieee/730/5284/
https://standards.ieee.org/ieee/730/5284/
5
92
https://quantum.cloud.ibm.com/docs/guides/construct-circuits
https://quantum.cloud.ibm.com/docs/guides/construct-circuits
https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-
technical-debt-reduction/
7
27
31
105
https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/
188
34 36 104 151 https://www.faa.gov/sites/faa.gov/files/aircraft/air_cert/design_approvals/air_software/
AR-01-18_MCDC.pdf
https://www.faa.gov/sites/faa.gov/files/aircraft/air_cert/design_approvals/air_software/AR-01-18_MCDC.pdf
9
91
143
https://acl.inf.ethz.ch/teaching/fastcode/2022/slides/07-simd-avx.pdf
https://acl.inf.ethz.ch/teaching/fastcode/2022/slides/07-simd-avx.pdf
10
40
52
53
54
116
117
118
148
https://sre.google/sre-book/monitoring-distributed-systems/
152
https://sre.google/sre-book/monitoring-distributed-systems/
11
12
13
47
48
49
90
https://www.cs.toronto.edu/~chechik/courses07/csc410/mills.pdf
https://www.cs.toronto.edu/~chechik/courses07/csc410/mills.pdf
14
https://cmmiinstitute.com/learning/appraisals/levels
https://cmmiinstitute.com/learning/appraisals/levels
15
128
149
https://google.github.io/eng-practices/review/reviewer/standard.html
https://google.github.io/eng-practices/review/reviewer/standard.html
https://www.fia.com/sites/default/files/documents/fia_2025_formula_1_technical_regulations_-
_issue_03_-_2025-04-07.pdf
16
120
https://www.fia.com/sites/default/files/documents/fia_2025_formula_1_technical_regulations_-_issue_03_-_2025-04-07.pdf
18
https://nepp.nasa.gov/workshops/etw2023/talks/15-JUN-THU/1000_Chen_20230009005.pdf
https://nepp.nasa.gov/workshops/etw2023/talks/15-JUN-THU/1000_Chen_20230009005.pdf
19
75
121
https://opentelemetry.io/docs/concepts/semantic-conventions/
https://opentelemetry.io/docs/concepts/semantic-conventions/
20
101
https://www.ibm.com/quantum/blog/large-scale-ftqc
https://www.ibm.com/quantum/blog/large-scale-ftqc
21
84
86
https://misra.org.uk/misra-c/
https://misra.org.uk/misra-c/
22
154
https://sre.google/sre-book/eliminating-toil/
https://sre.google/sre-book/eliminating-toil/
23
85
147
https://wiki.sei.cmu.edu/confluence/display/c/EXP37-C.
%2BCall%2Bfunctions%2Bwith%2Bthe%2Bcorrect%2Bnumber%2Band%2Btype%2Bof%2Barguments?
focusedCommentId=87153239
https://wiki.sei.cmu.edu/confluence/display/c/EXP37-C.
%2BCall%2Bfunctions%2Bwith%2Bthe%2Bcorrect%2Bnumber%2Band%2Btype%2Bof%2Barguments?
focusedCommentId=87153239
24
63
135
https://csrc.nist.gov/pubs/sp/800/53/r5/upd1/final
https://csrc.nist.gov/pubs/sp/800/53/r5/upd1/final
25
https://www.iso.org/standard/68383.html
https://www.iso.org/standard/68383.html
26
35
37
38
41
42
43
44
45
46
59
103
124
125
126
127
129
132
133
146
https://nodis3.gsfc.nasa.gov/
displayDir.cfm?Internal_ID=N_PR_7150_002D_&page_name=Chapter3
https://nodis3.gsfc.nasa.gov/displayDir.cfm?Internal_ID=N_PR_7150_002D_&page_name=Chapter3
1928 50 51 141 153 https://www.faa.gov/sites/faa.gov/files/aircraft/air_cert/design_approvals/air_software/
differences_tool.pdf
https://www.faa.gov/sites/faa.gov/files/aircraft/air_cert/design_approvals/air_software/differences_tool.pdf
29
https://owasp.org/www-project-application-security-verification-standard/
https://owasp.org/www-project-application-security-verification-standard/
30
69
71
https://peps.python.org/pep-0008/
https://peps.python.org/pep-0008/
32
78
https://www.itu.dk/~sestoft/bachelor/IEEE754_article.pdf
https://www.itu.dk/~sestoft/bachelor/IEEE754_article.pdf
33
https://slsa.dev/spec/v1.0/
https://slsa.dev/spec/v1.0/
39
64
142
https://cwe.mitre.org/top25/
https://cwe.mitre.org/top25/
55
80
https://go.dev/blog/errors-are-values
https://go.dev/blog/errors-are-values
56
https://www.erlang.org/doc/system/design_principles.html
https://www.erlang.org/doc/system/design_principles.html
57
https://doc.rust-lang.org/book/ch09-00-error-handling.html
https://doc.rust-lang.org/book/ch09-00-error-handling.html
58
70
https://peps.python.org/pep-0257/
https://peps.python.org/pep-0257/
60
74
76
140
https://peps.python.org/pep-0020/
https://peps.python.org/pep-0020/
61
https://diataxis.fr/explanation/
https://diataxis.fr/explanation/
62
https://devguide.owasp.org/en/06-verification/01-guides/03-asvs/
https://devguide.owasp.org/en/06-verification/01-guides/03-asvs/
65
68
https://pip.pypa.io/en/stable/topics/secure-installs/
https://pip.pypa.io/en/stable/topics/secure-installs/
66
155
https://slsa.dev/spec/v1.0/levels
https://slsa.dev/spec/v1.0/levels
67
https://docs.sigstore.dev/cosign/signing/overview/
https://docs.sigstore.dev/cosign/signing/overview/
72
73
https://peps.python.org/pep-0484/
https://peps.python.org/pep-0484/
https://developer.arm.com/documentation/den0042/latest/Floating-Point/Floating-point-basics-and-
the-IEEE-754-standard
77
79
https://developer.arm.com/documentation/den0042/latest/Floating-Point/Floating-point-basics-and-the-IEEE-754-standard
2081
83
https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines
https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines
82
87
88
https://google.github.io/styleguide/cppguide.html
https://google.github.io/styleguide/cppguide.html
89
https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
93
https://quantumai.google/cirq/simulate/params
https://quantumai.google/cirq/simulate/params
94
102
114
115
145
https://quantumai.google/cirq/google/best_practices
https://quantumai.google/cirq/google/best_practices
95
96
97
113
https://quantumai.google/cirq/simulate/noisy_simulation
https://quantumai.google/cirq/simulate/noisy_simulation
98
100
https://quantum-journal.org/papers/q-2022-03-30-677/
https://quantum-journal.org/papers/q-2022-03-30-677/
99
119
https://quantum.cloud.ibm.com/docs/api/qiskit/transpiler
https://quantum.cloud.ibm.com/docs/api/qiskit/transpiler
106
https://www.tensorflow.org/tfx/data_validation/get_started
https://www.tensorflow.org/tfx/data_validation/get_started
107
https://www.tensorflow.org/tfx/guide/tfdv
https://www.tensorflow.org/tfx/guide/tfdv
108
109
https://docs.pytorch.org/docs/stable/notes/randomness.html
https://docs.pytorch.org/docs/stable/notes/randomness.html
110
112
150
https://docs.pytorch.org/docs/stable/generated/torch.autograd.gradcheck.gradcheck.html
https://docs.pytorch.org/docs/stable/generated/torch.autograd.gradcheck.gradcheck.html
111
https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html
https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html
122
123
https://sre.google/workbook/error-budget-policy/
https://sre.google/workbook/error-budget-policy/
130
131
https://google.github.io/eng-practices/review/developer/
https://google.github.io/eng-practices/review/developer/
https://www.agilealliance.org/wp-content/uploads/2016/01/SQALE-Meaningful-Insights-into-your-
Technical-Debt.pdf
134
136
https://www.agilealliance.org/wp-content/uploads/2016/01/SQALE-Meaningful-Insights-into-your-Technical-Debt.pdf
137
139
https://www.hyrumslaw.com/
https://www.hyrumslaw.com/
138
https://datatracker.ietf.org/doc/html/rfc1122
https://datatracker.ietf.org/doc/html/rfc1122
144
https://www.openmp.org/spec-html/5.2/openmpsu52.html
https://www.openmp.org/spec-html/5.2/openmpsu52.html
21