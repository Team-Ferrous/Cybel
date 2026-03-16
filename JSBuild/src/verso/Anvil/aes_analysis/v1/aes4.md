The Anvil Engineering Standard (AES)
Version 1.0 — Unified Coding Standards and Governance for the Anvil Agent
"Correctness is not a property you add to software. It is a property you prove you have not destroyed." —
Adapted from IBM Cleanroom methodology
TABLE OF CONTENTS
1. Preamble
2. Severity & Assurance Framework
3. Universal Coding Mandates
4. Language-Specific Standards
5. Domain-Specific Standards
6. Verification & Testing Framework
7. Observability & Telemetry
8. Agent Workflow Governance
9. Code Review & Change Management
10. Anti-Patterns Registry
11. Appendices
1. PREAMBLE
1.1 Purpose
The Anvil Engineering Standard (AES) is the governing law for all code written, reviewed, audited, or
maintained by the Anvil AI coding agent and its semantic intelligence layer, Saguaro. It synthesizes the strictest
applicable requirements from NASA flight software standards, FAA avionics certification, automotive safety,
functional safety, secure coding, numerical computing, and ML/quantum engineering into a single, coherent,
prescriptive rulebook.
AES exists because high-assurance and high-velocity are not opposites. NASA builds software that works
perfectly for decades. Deep learning research demands rapid iteration. This standard bridges that gap by
applying NASA-grade discipline to the interfaces and invariants of a system while permitting agility in the
implementation details.1.2 Domains Governed
AES governs all code in the following domains operated by Anvil:
DomainPrimary LanguagesKey Concerns
General Software EngineeringPython 3.12+, C++17Correctness, security, maintainability
Deep Learning / MLPython, CUDA (via wrappers)Gradient integrity, numerical stability, reproducibility
Quantum ComputingPython (Qiskit/Cirq), C++Circuit correctness, noise-awareness, hybrid
interfaces
Physics SimulationC++17, PythonConservation laws, numerical accuracy, symmetry
High-PerformanceC++17, SIMD/AVX2,Throughput, memory discipline, determinism
ComputingOpenMP
1.3 How to Read This Document
All requirements use RFC 2119 language:
MUST / SHALL / MUST NOT / SHALL NOT — Mandatory. Violation is a defect. No exceptions
without written exemption.
SHOULD / SHOULD NOT — Strong recommendation. Deviation requires documented rationale.
MAY — Optional guidance. Use when appropriate.
Each major rule includes a [Source:] tag citing the originating standard(s). When multiple standards agree, the
strictest interpretation governs.
Severity tagging: Rules marked [P0] through [P3] indicate at which severity level the rule becomes
mandatory (P0 = all, P3 = docs only). See Section 2.
2. SEVERITY & ASSURANCE FRAMEWORK
2.1 Unified Severity Levels
AES synthesizes DAL (DO-178C), ASIL (ISO 26262), and SIL (IEC 61508) into a single Anvil Priority Level
(APL) system. The mapping is approximate; the Anvil definitions govern.APL
Name
APL-
Catastrophic
DO-178CISO 26262IEC 61508
AnalogAnalogAnalog
DAL-AASIL DSIL 4
0
Consequence of Failure
Loss of training run integrity, silent data
corruption, security breach, gradient explosion in
production
APL-
Critical
DAL-B
ASIL C
SIL 3
1
Wrong model outputs, numerical instability,
quantum circuit error, simulation divergence
APL-
Marginal
DAL-C
ASIL B
SIL 2
2
Performance degradation, incorrect config, partial
test failure
APL-
Informational
DAL-D
ASIL A
SIL 1
3
Documentation errors, style violations, advisory
warnings
Assignment rules:
SIMD kernels and gradient computation loops: APL-0
Training loops, model inference, quantum circuit execution, physics simulation core: APL-1
Configuration management, tooling, data preprocessing: APL-2
Documentation, examples, non-critical scripts: APL-3
2.2 Verification Requirements by APL
RequirementAPL-0APL-1APL-2APL-3
Formal design reviewMUSTMUSTSHOULDMAY
Full traceability matrixMUSTMUSTMUSTSHOULD
MC/DC structural coverageMUSTMUSTSHOULD—
Independent V&VMUSTSHOULD——
FMEA analysisMUSTMUSTSHOULD—
FTA (Fault Tree Analysis)MUSTSHOULD——
Property-based testsMUSTMUSTSHOULD—
Fuzz testingMUSTSHOULD——RequirementAPL-0APL-1APL-2APL-3
Mutation testingMUSTSHOULD——
Performance baseline delta ≤5%MUSTMUSTSHOULD—
Security audit (OWASP ASVS L2+)MUSTMUSTSHOULD—
2.3 Severity Decision Tree
Is the code on the critical computational path (gradients, SIMD, circuit gates)?
YES → APL-0
NO → Does incorrect output cause wrong model results or simulation divergence?
YES → APL-1
NO → Does it affect system configuration or data pipelines?
YES → APL-2
NO → APL-3
3. UNIVERSAL CODING MANDATES
These rules apply to all code regardless of language, domain, or APL level.
3.1 Traceability [P0]
AES-T1: Every code artifact MUST be traceable to a requirement. The chain is: Requirement → Design Decision
→ Implementation → Test → Verification Result . No code SHALL exist without a traceable origin. [Source: NASA
NPR 7150.2D, DO-178C §11.9]
AES-T2: Traceability MUST be maintained in machine-readable form (e.g., # AES-REQ-042 comments, JIRA
links, or a TRACEABILITY.toml manifest). [Source: NASA-STD-8739.8B]
AES-T3: Dead code MUST NOT exist in APL-0 or APL-1 components. Dead code is a DO-178C structural
coverage violation and indicates untested execution paths. [Source: DO-178C §6.4.4.2]
3.2 Complexity Bounds [P0]
AES-C1: All hot-path functions MUST have documented time complexity. O(1) or O(log n) are preferred. O(n)
is acceptable. O(n²) or worse in hot paths is prohibited without an explicit APL-waiver signed off by review.
[Source: CMMI Level 5 performance baselines, Google SRE capacity planning]
AES-C2: Cyclomatic complexity MUST NOT exceed 10 per function in APL-0/APL-1 code. SHOULD NOT
exceed 15 in APL-2. [Source: IEEE 730, IBM Cleanroom]AES-C3: Functions MUST NOT exceed 60 lines of executable code (excluding docstrings and blank lines). A
function that approaches this limit SHOULD be decomposed. [Source: Google Style Guides, C++ Core
Guidelines F.2]
3.3 Error Handling Philosophy [P0]
AES synthesizes IBM Cleanroom (fix root cause), Go (explicit errors), and Rust (Result typing) into one
philosophy: Errors are values. Every error path is a code path. Every code path is a test case.
AES-E1: All public functions MUST declare their error contract explicitly. In Python, this means return type
annotations including Optional , Result , or raising documented exceptions. In C++, this means [[nodiscard]]
return types or std::expected . [Source: CERT C++, Google Style Guide, Rust philosophy]
AES-E2: Exceptions MAY be used for genuinely exceptional conditions (e.g., OOM, hardware fault). They
MUST NOT be used for control flow. [Source: C++ Core Guidelines E.3, MISRA C++ Rule 15-0-1]
AES-E3: All error paths MUST be covered by tests. An untested error path is equivalent to dead code (see
AES-T3). [Source: DO-178C MC/DC, IBM Cleanroom]
AES-E4: The Cleanroom mandate applies universally: Fix root cause. Never mask symptoms. except: pass ,
swallowed exceptions, and silent failure modes are forbidden. [Source: IBM Cleanroom defect prevention]
AES-E5: At APL-0 and APL-1 boundaries (e.g., entering a training loop, executing a quantum circuit), all
inputs MUST be validated before processing begins. Validation failures MUST abort with a structured error, not
silently degrade. [Source: CERT C, ISO 26262 §8]
3.4 Documentation Requirements [P0]
AES adopts the Diátaxis framework: every documentation artifact is one of Tutorial, How-To, Reference, or
Explanation. Mixed-purpose documentation SHALL NOT be created.
AES-D1: All public functions, classes, and modules MUST have docstrings that document: purpose, parameters
(with types), return value, raised exceptions, and any side effects. [Source: PEP 257, Google Python Style
Guide]
AES-D2: All non-obvious algorithmic choices MUST include an explanation comment citing the source (paper,
textbook, or standard). Example: # Kahan compensated sum — Higham 2002 §4.3 [Source: IEEE 730]
AES-D3: TODO comments MUST include an owner and a ticket reference: # TODO(owner): AES-123 —
description . Unowned TODOs are treated as APL-3 defects. [Source: Google Code Review practices]
3.5 Security Baseline [P0]
AES-S1: All code MUST comply with OWASP ASVS Level 1 at minimum. APL-0/APL-1 components MUST
comply with Level 2. [Source: OWASP ASVS, NIST SP 800-53 SA-11]
AES-S2: All external inputs MUST be validated and sanitized before use. Buffer overflows, integer overflows,
and format string vulnerabilities (CWE Top 25) MUST be prevented by construction, not by post-hoc patching.[Source: CERT C/C++, CWE-120, CWE-190, CWE-134]
AES-S3: All dependencies MUST be pinned to exact versions with cryptographic hash verification.
Dependency updates MUST trigger a full regression gate. Supply chain provenance MUST be verifiable (SLSA
Level 2+ for APL-0/APL-1). [Source: NIST SP 800-53 SA-12, SLSA framework]
AES-S4: No credentials, API keys, or secrets SHALL appear in source code or version history. All secrets
MUST be injected via environment variables or secrets management systems. [Source: OWASP ASVS V2,
CWE-312]
AES-S5: Cryptographic operations MUST use approved algorithms from NIST SP 800-131A. No MD5, SHA-
1, or DES. [Source: NIST SP 800-53 SC-13]
3.6 Naming and Structure [P1]
AES-N1: Names MUST be self-documenting. Single-letter variables are forbidden outside mathematical
contexts (loop indices, linear algebra). In ML code, mathematical variable names ( W , b , grad ) are permitted
when the corresponding mathematical formulation is documented. [Source: C++ Core Guidelines NL.1, Google
Style Guides]
AES-N2: All modules MUST have a single, stated responsibility (Single Responsibility Principle). A module
that does two things is a design defect. [Source: SOLID principles, AUTOSAR Adaptive component model]
4. LANGUAGE-SPECIFIC STANDARDS
4a. Python Standards
4a.1 Type System
AES-PY1: All Python code MUST use type annotations (PEP 484, PEP 526). mypy --strict MUST pass with
zero errors for APL-0/APL-1 code. Any is forbidden except at I/O boundaries where types are genuinely
unknown, and MUST be documented. [Source: PEP 484, Google Python Style Guide]
python# REQUIRED: Full type annotation
def compute_loss(
logits: torch.Tensor,
targets: torch.Tensor,
reduction: str = "mean"
) -> torch.Tensor:
"""Compute cross-entropy loss with explicit shape contract.
Args:
logits: Shape (B, C) float32 unnormalized log-probabilities.
targets: Shape (B,) int64 class indices in [0, C).
reduction: One of 'mean', 'sum', 'none'.
Returns:
Scalar loss if reduction in ('mean','sum'), else shape (B,).
Raises:
ValueError: If logits/targets shapes are incompatible.
"""
if logits.ndim != 2 or targets.ndim != 1:
raise ValueError(
f"Expected logits (B,C) and targets (B,), "
f"got {logits.shape} and {targets.shape}"
)
...
AES-PY2: dataclasses or pydantic MUST be used for structured data. Plain dict is forbidden as a function's
primary data contract. [Source: Google Python Style Guide]
4a.2 Imports and Module Structure
AES-PY3: Imports MUST be ordered: stdlib → third-party → local, with blank lines between groups. Wildcard
imports ( from x import * ) are forbidden in all APL levels. [Source: PEP 8, Google Python Style Guide]
AES-PY4: Circular imports MUST NOT exist. If they arise, the design is wrong and MUST be refactored.
[Source: Google Python Style Guide]
4a.3 Python Safety Patterns
AES-PY5: eval() and exec() are forbidden outside explicitly sandboxed environments. [Source: CERT
Python, CWE-95]
AES-PY6: All file operations MUST use context managers ( with statement). Resource leaks are APL-2
defects. [Source: SEI CERT Python, OWASP]AES-PY7: Mutable default arguments MUST NOT be used. Use None sentinel with interior initialization.
python
# FORBIDDEN
def add_item(item: str, items: list[str] = []) -> None: ...
# REQUIRED
def add_item(item: str, items: list[str] | None = None) -> None:
if items is None:
items = []
...
4b. C/C++ Standards
4b.1 Resource Management
AES-CPP1: All resource acquisition MUST use RAII. Raw new / delete are forbidden in application code.
Use std::unique_ptr , std::shared_ptr , or stack allocation. [Source: C++ Core Guidelines R.1, MISRA C++ Rule
18-0-1]
AES-CPP2: std::shared_ptr SHOULD be avoided in hot paths due to atomic reference count overhead. Prefer
std::unique_ptr or raw borrowed pointers (with documented lifetime). [Source: C++ Core Guidelines R.20, HPC
performance discipline]
4b.2 Type Safety
AES-CPP3: C-style casts are forbidden. Use static_cast , reinterpret_cast (with comment justifying necessity),
const_cast (forbidden except at FFI boundaries), dynamic_cast (SHOULD be avoided; forbidden in APL-0 hot
paths). [Source: C++ Core Guidelines ES.49, MISRA C++ Rule 5-2-4]
AES-CPP4: [[nodiscard]] MUST be applied to all functions that return error codes or values whose discard
constitutes a defect. [Source: C++ Core Guidelines F.9, CERT C++]
AES-CPP5: Integer arithmetic MUST be checked for overflow at APL-0/APL-1. Use std::numeric_limits
checks or SafeInt libraries. [Source: CERT C++ INT30-C, CWE-190]
4b.3 Undefined Behavior Prevention
AES-CPP6: All code MUST compile clean with -Wall -Wextra -Wpedantic -Wundefined-behavior -
fsanitize=address,undefined in debug builds. APL-0 code MUST also pass clang -analyze . [Source: MISRA C++,
CERT C++]
AES-CPP7: Uninitialized variables are forbidden. All variables MUST be initialized at declaration. [Source:
MISRA C++ Rule 8-5-1, C++ Core Guidelines ES.20]
4b.4 Include and Namespace DisciplineAES-CPP8: Include order MUST be: related header → C system headers → C++ standard library → third-
party → project headers. [Source: Google C++ Style Guide]
AES-CPP9: using namespace std; is forbidden in header files and SHOULD NOT appear in implementation
files. Use explicit namespacing. [Source: C++ Core Guidelines SF.7, Google C++ Style Guide]
4c. Quantum Circuit Standards
AES-QC1: All quantum circuits MUST specify the target backend and its noise model before design begins. A
circuit designed for an ideal simulator and run on hardware without transpilation is a defect. [Source: IBM
Qiskit Best Practices]
AES-QC2: Circuit parameterization MUST use named parameters (e.g., ParameterVector ). Magic-number
angles in circuits are forbidden. [Source: Cirq patterns, IBM Qiskit]
python
# FORBIDDEN: magic angle
circuit.rx(0.785398, qubit)
# REQUIRED: named parameter with documented meaning
theta = Parameter('theta_ry') # Rotation angle in variational layer l
circuit.ry(theta, qubit)
AES-QC3: All quantum-classical interfaces MUST document the encoding scheme (e.g., amplitude encoding,
angle encoding, basis encoding) and its normalization requirements. [Source: IBM Qiskit Best Practices]
5. DOMAIN-SPECIFIC STANDARDS
5a. Deep Learning / ML Standards
5a.1 Training Loop Integrity [APL-0]
AES-ML1: Every training loop MUST implement gradient health checks before each optimizer step. A training
loop without gradient monitoring is an APL-0 defect. Required checks:
python# MANDATORY: gradient health gate before optimizer.step()
def check_gradient_health(model: nn.Module, step: int) -> None:
for name, param in model.named_parameters():
if param.grad is None:
continue
grad = param.grad
if not torch.isfinite(grad).all():
raise GradientCorruptionError(
f"Non-finite gradient in {name} at step {step}: "
f"nan={grad.isnan().sum()}, inf={grad.isinf().sum()}"
)
grad_norm = grad.norm().item()
if grad_norm > GRADIENT_NORM_THRESHOLD:
log_warning(f"Large gradient norm in {name}: {grad_norm:.3e}")
[Source: ML Engineering Best Practices, Google ML Test Score §model-training]
AES-ML2: Gradient clipping MUST be applied at APL-0 training runs unless explicitly disabled with
documented rationale. Max gradient norm MUST be logged as a telemetry signal. [Source: ML Engineering
Best Practices]
AES-ML3: Training loss MUST be validated as finite after every step. The training loop MUST abort, log, and
alert on the first non-finite loss. Silent NaN propagation is forbidden. [Source: IBM Cleanroom defect
prevention, Google ML Test Score]
AES-ML4: Every training run MUST produce a reproducibility manifest containing: random seeds (Python,
NumPy, framework), software versions (exact), hardware description, and hyperparameters. Runs without
manifests are not verifiable. [Source: ML Engineering Best Practices, CMMI Level 5]
5a.2 Numerical Stability [APL-0]
AES-ML5: Log-sum-exp computations MUST use numerically stable implementations. Direct exp() without
stabilization is forbidden in APL-0/APL-1 code.
python# FORBIDDEN: numerically unstable
def softmax_bad(x: torch.Tensor) -> torch.Tensor:
return torch.exp(x) / torch.exp(x).sum()
# REQUIRED: numerically stable (subtract max)
def softmax_stable(x: torch.Tensor) -> torch.Tensor:
x_shifted = x - x.max(dim=-1, keepdim=True).values
exp_x = torch.exp(x_shifted)
return exp_x / exp_x.sum(dim=-1, keepdim=True)
[Source: Higham "Accuracy and Stability of Numerical Algorithms" §4, IEEE 754]
AES-ML6: Mixed-precision training (FP16/BF16) MUST use loss scaling. The loss scaler state MUST be
checkpointed alongside model weights. [Source: IEEE 754, ML Engineering Best Practices]
AES-ML7: All floating-point accumulation in APL-0 code SHOULD use Kahan compensated summation or
pairwise summation for vectors larger than 10⁴ elements. [Source: Higham §4.3, IEEE 754]
python
def kahan_sum(values: list[float]) -> float:
"""Kahan compensated summation. Source: Higham 2002 §4.3."""
total = 0.0
compensation = 0.0
for v in values:
y = v - compensation
t = total + y
compensation = (t - total) - y
total = t
return total
5a.3 Data Pipeline Validation [APL-1]
AES-ML8: Data pipelines MUST validate: schema (column types, shapes), range checks (no out-of-
distribution outliers beyond 5σ without flagging), and referential integrity. Failed validation MUST halt pipeline
execution with structured diagnostics. [Source: Google ML Test Score §data-tests]
AES-ML9: Train/validation/test splits MUST be verified as non-overlapping using example-level hashing
before any training begins. Data leakage is an APL-1 defect. [Source: Google ML Test Score]
AES-ML10: All dataset preprocessing MUST be versioned alongside model code. A model is not reproducible
without its exact preprocessing pipeline. [Source: ML Engineering Best Practices, CMMI Level 5]
5a.4 Model Validation [APL-1]AES-ML11: Every model MUST pass a "sanity sweep": overfit a single batch to near-zero loss within a
reasonable step budget. A model that cannot overfit one batch has a bug, not a generalization problem. [Source:
ML Engineering Best Practices, IBM Cleanroom]
AES-ML12: Evaluation MUST be performed on a held-out test set that was invisible during all development
(including hyperparameter search). Validation-set hyperparameter selection is acceptable; test-set selection is an
APL-1 integrity violation. [Source: ML Engineering Best Practices]
5b. Quantum Computing Standards
5b.1 Circuit Design Discipline [APL-1]
AES-QT1: Every quantum algorithm MUST have a documented classical simulation path for correctness
verification. Circuits that cannot be simulated classically (due to qubit count) MUST instead have a parametric
test suite covering all structural branches. [Source: IBM Qiskit Best Practices, DO-178C structural coverage]
AES-QT2: Circuit depth MUST be tracked as an explicit metric. Unnecessary depth (redundant gates, missing
optimization passes) is an APL-2 defect. Transpilation MUST be run with optimization_level ≥ 2 for hardware
execution. [Source: IBM Qiskit Best Practices]
AES-QT3: Barren plateau diagnostics MUST be run for variational circuits with more than 10 parameters
before full-scale training. [Source: Quantum ML research practice]
5b.2 Noise-Aware Programming [APL-1]
AES-QT4: All circuits intended for noisy hardware execution MUST be validated against a noise model before
hardware submission. The noise model MUST include: gate error rates (1-qubit and 2-qubit), readout error, and
T1/T2 decoherence. [Source: IBM Qiskit Best Practices]
AES-QT5: Quantum error mitigation techniques (zero-noise extrapolation, probabilistic error cancellation)
MUST be documented when applied, including their assumptions and their impact on the effective sample
complexity. [Source: IBM Qiskit Best Practices, research best practices]
5b.3 Classical-Quantum Interface [APL-1]
AES-QT6: Classical post-processing of quantum measurement results MUST validate that shot counts are
sufficient for the required statistical confidence. Minimum shots MUST be computed as shots ≥ 1 / (ε² · δ)
where ε is the desired precision and δ is the failure probability. [Source: quantum complexity theory, JEDEC
statistical rigor]
AES-QT7: Parameter shift gradients MUST be validated against finite-difference estimates on small circuits
before deployment in a VQC training loop. Discrepancy threshold: |∂_shift - ∂_fd| / max(|∂_shift|, 1e-8) < 1e-3 .
[Source: quantum ML best practices, AES-ML5]
5c. Physics Simulation Standards
5c.1 Conservation Law Verification [APL-0]AES-PS1: Every physics simulation MUST implement conservation law monitors. For mechanical systems:
total energy, momentum, and angular momentum. For electromagnetic systems: charge and flux. Violation
thresholds MUST be set before simulation begins, and violations MUST trigger structured alerts. [Source:
Numerical Methods literature, NASA flight simulation practice]
cpp
struct ConservationMonitor {
double initial_energy;
double tolerance_fraction; // e.g., 1e-6 for APL-0
void check(double current_energy, size_t step) const {
const double relative_error =
std::abs(current_energy - initial_energy) / std::abs(initial_energy);
if (relative_error > tolerance_fraction) {
throw ConservationViolationError{
"Energy conservation violated at step " + std::to_string(step) +
": relative error = " + std::to_string(relative_error)
};
}
}
};
AES-PS2: Numerical integrators MUST be validated against known analytical solutions before use in
production. The validation MUST cover the full expected simulation time range. [Source: Higham, NASA
verification practice]
5c.2 Symmetry Enforcement [APL-1]
AES-PS3: Symmetries present in the physical system (time-reversal, gauge invariance, Lorentz covariance)
MUST be verified numerically in tests. A test that breaks known symmetry is a sign-off blocker. [Source:
physics simulation best practices]
AES-PS4: Numerical method selection MUST be documented with: order of accuracy, stability region,
dissipation and dispersion properties, and the CFL condition if applicable. Changing numerical methods
requires re-running the full validation suite. [Source: Numerical Methods for Conservation Laws, Leveque]
5c.3 Floating-Point Discipline [APL-0]
AES-PS5: Simulation code MUST use IEEE 754 double precision (float64) unless a documented rationale for
float32 is provided and the impact on conservation law tolerance is quantified. [Source: IEEE 754, Higham]
AES-PS6: Catastrophic cancellation MUST be avoided. Subtraction of nearly-equal numbers in critical paths
MUST use alternative formulations (e.g., (a-b)(a+b) instead of a²-b² when a≈b ). [Source: Higham §1.7]5d. High-Performance Computing Standards
5d.1 CPU-First, SIMD-First Design [APL-0]
AES-HPC1: All hot-path numerical kernels MUST be designed for CPU execution first, with SIMD
vectorization as the primary optimization strategy. GPU offload is secondary and MUST be justified by
profiling data. [Source: HPC engineering practice, Agner Fog optimization guides]
AES-HPC2: SIMD code MUST target AVX2 as the baseline. AVX-512 SHOULD be supported but MUST
have a runtime fallback to AVX2. Kernels MUST be benchmarked on both paths. [Source: Intel Intrinsics
Guide, HPC engineering practice]
cpp
// REQUIRED pattern: runtime dispatch
void vector_multiply(float* a, const float* b, size_t n) {
#if defined(__AVX2__)
vector_multiply_avx2(a, b, n);
#elif defined(__SSE4_2__)
vector_multiply_sse42(a, b, n);
#else
vector_multiply_scalar(a, b, n);
#endif
}
AES-HPC3: All SIMD kernels MUST include a scalar reference implementation that produces bit-identical
results (within IEEE 754 rounding). The scalar path MUST be tested independently. [Source: DO-178C
structural coverage, HPC verification practice]
5d.2 Memory Alignment and Cache Discipline [APL-0]
AES-HPC4: Data structures used in SIMD hot paths MUST be aligned to at least 32 bytes (AVX2
requirement). Use alignas(32) for stack arrays and std::aligned_alloc for heap. Unaligned SIMD loads in hot
paths are forbidden. [Source: Intel Architecture Manual, HPC engineering]
cpp
// REQUIRED: 32-byte aligned allocation for AVX2
alignas(32) float weight_buffer[SIMD_WIDTH];
// For heap allocation:
float* data = static_cast<float*>(std::aligned_alloc(32, n * sizeof(float)));
if (!data) throw std::bad_alloc{};AES-HPC5: Hot-path data structures MUST be analyzed for cache locality. SoA (Structure of Arrays) MUST
be preferred over AoS (Array of Structures) when accessing individual fields in tight loops. [Source: Mike
Acton CppCon 2014, HPC cache optimization]
AES-HPC6: False sharing MUST be prevented in multi-threaded hot paths by padding shared data structures to
cache line boundaries (64 bytes). [Source: Intel VTune documentation, CMMI performance discipline]
5d.3 Concurrency and Threading [APL-0]
AES-HPC7: All parallelized code MUST be analyzed for data races using ThreadSanitizer (TSan) in CI. A
TSan-reported data race is an APL-0 defect with immediate halt. [Source: CERT C++ CON32-C, C++ Core
Guidelines CP.1]
AES-HPC8: OpenMP pragmas MUST specify all relevant clauses explicitly: schedule , num_threads ,
reduction , and data-sharing ( private , shared , firstprivate ). Implicit data sharing is forbidden. [Source:
MISRA C++ adapted, OpenMP specification]
cpp
// FORBIDDEN: implicit data sharing
#pragma omp parallel for
for (int i = 0; i < n; ++i) { result += a[i]; }
// REQUIRED: explicit clauses
#pragma omp parallel for schedule(static) reduction(+:result) \
num_threads(THREAD_COUNT)
for (int i = 0; i < n; ++i) { result += a[i]; }
AES-HPC9: Lock-free algorithms MUST have a documented proof of correctness (or citation of a peer-
reviewed proof) and MUST be validated under TSan with at minimum 10⁸ operations. [Source: C++ Core
Guidelines CP.20, CERT C++]
6. VERIFICATION & TESTING FRAMEWORK
6.1 Test Taxonomy
Every component MUST have tests at each applicable level:LevelDescriptionAPL-0APL-1APL-2APL-3
UnitFunction/class in isolationMUSTMUSTMUSTSHOULD
IntegrationModule interactionsMUSTMUSTSHOULDMAY
Property-basedInvariant testing (Hypothesis)MUSTMUSTSHOULD—
FuzzInput space explorationMUSTSHOULD——
BenchmarkPerformance regressionMUSTMUSTSHOULD—
MutationVerify test efficacyMUSTSHOULD——
End-to-endFull pipelineMUSTMUSTSHOULDMAY
6.2 Coverage Requirements
AES-V1: APL-0 code MUST achieve 100% statement coverage, 100% branch coverage, and Modified
Condition/Decision Coverage (MC/DC) — as mandated by DO-178C DAL-A. [Source: DO-178C §6.4.4.2]
AES-V2: APL-1 code MUST achieve 100% statement coverage and 100% branch coverage. [Source: DO-178C
DAL-B]
AES-V3: Coverage MUST be measured in CI on every commit. Coverage regressions at APL-0/APL-1 are a
build-breaking defect. [Source: Google SRE error budgets, CMMI Level 5]
AES-V4: Test files MUST be co-located with implementation or in a mirrored test tree. Tests MUST be named
test_<module_name> . [Source: Google Python Style Guide, IEEE 730]
6.3 ML-Specific Testing [APL-1]
AES adopts the Google ML Test Score framework. The following MUST pass before any ML model is
promoted:
Data Tests:
Schema validation (types, shapes, ranges)
Distribution drift detection vs. baseline
Train/val/test non-overlap verification (AES-ML9)
Model Tests:
Single-batch overfit test (AES-ML11)
Gradient flow test: every parameter receives a gradientLoss decrease monotonicity on known-good data
Output shape and type contracts
Infrastructure Tests:
Training restarts from checkpoint produce identical results
Model serialization round-trips preserve outputs within float32 precision
Inference latency is within P95 SLO
6.4 Quantum-Specific Testing [APL-1]
AES-QV1: All quantum circuits MUST be tested on a noise-free simulator first, verifying known outputs for
standard input states. [Source: IBM Qiskit Best Practices]
AES-QV2: Noise model tests MUST verify that error mitigation techniques improve output fidelity on the
target noise model by a quantified margin. [Source: IBM Qiskit Best Practices]
AES-QV3: Parametric circuits MUST be tested at parameter values {0, π/4, π/2, π, 3π/2, 2π} to detect angle-
wrapping bugs and identity-gate conditions. [Source: IBM Qiskit circuit testing practice]
6.5 The Chronicle Protocol
Every APL-0/APL-1 change MUST follow the Chronicle Protocol:
1. Baseline: Record all Golden Signal metrics (see §7) before the change.
2. Change: Implement and test the change.
3. Result: Record all Golden Signal metrics after the change.
4. Delta: Log the delta. If any signal regresses beyond threshold, block merge.
The Chronicle log is a permanent artifact, stored alongside the change record.
7. OBSERVABILITY & TELEMETRY
7.1 Golden Signals (Adapted per Domain)
AES adapts Google SRE's Golden Signals (Latency, Throughput, Errors, Saturation) to each domain:SignalGeneral / MLQuantumPhysics SimHPC
Latencystep_timecircuit_submit →integration_step_timekernel_time,
(P50/P95/P99), TTFTresult_latencytokens/sec,shots/sec, circuits/sec
Throughput
dispatch_latency
steps/sec, particles/sec
samples/sec
Errors
Saturation
GFLOPS, GB/s
memory bandwidth
non-finite loss rate,gate error rate, readoutconservation violationSIMD fault rate, OOM
gradient failureserror raterateevents
GPU/CPU util, RSS,QPU queue depth,memory pressure, solverthread utilization, L3
VRAMclassical pre/postiterationscache miss rate
overhead
7.2 Mandatory Metrics [APL-0/APL-1]
All APL-0 and APL-1 systems MUST export the following via structured logging (OpenTelemetry-compatible
format):
json
{
"timestamp": "ISO-8601",
"apl_level": 0,
"component": "training_loop",
"step": 1234,
"metrics": {
"loss": 0.234,
"loss_finite": true,
"grad_norm": 0.892,
"grad_norm_clipped": false,
"step_time_ms": 42.1,
"throughput_samples_per_sec": 1024,
"memory_rss_mb": 8192,
"thread_utilization": 0.94
}
}7.3 Alerting Thresholds
ConditionThresholdAction
Non-finite lossAny occurrenceIMMEDIATE HALT + alert
Gradient norm > 100Sustained 3+ stepsWarning + log
Step time P99 > 2× P50Any stepWarning + investigate
Conservation violation> toleranceHALT simulation
SIMD coverage < 80%After kernel changeFail benchmark gate
Test coverage regressionAny APL-0/APL-1Block merge
7.4 FIA-Inspired Saturation Monitoring
Borrowing from F1 telemetry discipline, all APL-0 components MUST track compound saturation score — a
weighted combination of resource utilization signals that indicates overall system stress. When compound
saturation exceeds 85%, the system MUST begin graceful degradation (shedding non-critical work) rather than
failing catastrophically.
8. AGENT WORKFLOW GOVERNANCE
These rules govern Anvil (and Saguaro) specifically as AI coding agents operating on production repositories.
8.1 Code Generation Guardrails [APL-0]
AES-AG1: Anvil MUST NOT generate code that circumvents or disables any monitoring, alerting, health
check, or safety gate defined in this standard. An agent disabling its own oversight is a categorical violation.
[Source: AI safety principles, NASA-STD-8739.8B systematic assurance]
AES-AG2: Anvil MUST NOT commit directly to main / master branches for APL-0 or APL-1 components. All
changes MUST go through the review gate defined in §9. [Source: DO-178C change control, Google SRE
change management]
AES-AG3: When generating numerical algorithms, Anvil MUST cite the source (paper, textbook, standard) in
the generated code comment. Generated algorithms without provenance are APL-2 defects. [Source: NASA
NPR 7150.2D traceability]
AES-AG4: Anvil MUST NOT generate code containing hardcoded credentials, API keys, or personally
identifying information. [Source: AES-S4, OWASP ASVS]AES-AG5: Before generating any APL-0 code, Anvil MUST produce and log an FMEA summary identifying
the top 3 failure modes for the proposed implementation. [Source: NASA NPR 7150.2D FMEA requirement]
8.2 Self-Verification Requirements [APL-0]
AES-AG6: Anvil MUST self-verify generated code using the following checklist before presenting it for
review:
SELF-VERIFICATION CHECKLIST (mandatory for APL-0/APL-1):
[ ] Type annotations complete and mypy-clean
[ ] All error paths handled and documented
[ ] No dead code introduced
[ ] Complexity bounds documented
[ ] Security: no input validation bypasses, no hardcoded secrets
[ ] Numerical: no direct exp() without stabilization, no catastrophic cancellation
[ ] Tests: unit tests generated for all public functions
[ ] Traceability: requirement reference included
[ ] FMEA: top-3 failure modes identified
[ ] Chronicle: baseline metrics recorded
8.3 Red-Team Protocol [APL-0]
Before any APL-0 change is proposed for review, Anvil MUST run the Red-Team Protocol:
1. Complexity attack: Can the change be made O(n) worse by adversarial input? If yes, bound it.
2. FMEA sweep: List all single points of failure introduced.
3. FTA (Fault Tree Analysis): Identify all combinatorial failure conditions.
4. OWASP sweep: Check CWE Top 25 applicability.
5. Regression gate: Does the change break any existing test? Does it reduce coverage?
6. Numerical check (ML/Physics/Quantum): Are all floating-point operations stable?
Red-team findings MUST be documented in the change record. Unresolved critical findings block merge.
8.4 Saguaro Semantic Layer Discipline
Saguaro MUST treat its code understanding outputs as probabilistic, not authoritative. When Saguaro identifies
a pattern (e.g., "this is a training loop"), it MUST emit a confidence score. Downstream decisions MUST
degrade gracefully when confidence is below threshold.
AES-AG7: Saguaro MUST NOT make irreversible changes (delete files, push to main, deploy) autonomously.
Irreversible actions require explicit human confirmation or a time-delayed rollback window. [Source: AI safety,
NASA systematic assurance]9. CODE REVIEW & CHANGE MANAGEMENT
9.1 Review Requirements by APL
APLReviewers RequiredAutomated GatesMerge Conditions
APL-02 humans + Anvil self-reviewAll CI gates, Chronicle protocol, Red-team logAll must PASS
APL-11 human + Anvil self-reviewCI gates, coverage checkAll must PASS
APL-21 human OR Anvil reviewCI gatesMust PASS
APL-3Anvil review onlyLinterSHOULD PASS
9.2 Review Comment Standards
AES adopts Conventional Comments (conventionalcomments.org) for all review feedback:
suggestion: — Optional improvement
nitpick: — Minor, non-blocking
issue: — Blocking defect
question: — Needs clarification before merge
thought: — Non-blocking observation
AES-CR1: Every issue: comment MUST reference an AES rule number or an external standard. Comments
without rationale are noise. [Source: Google Code Review practices]
AES-CR2: Reviews MUST evaluate design, not just implementation. If the design is wrong, fix the design, not
the code. [Source: IBM Cleanroom principle]
9.3 Change Risk Assessment
Every proposed change MUST be assigned a Change Risk Score (CRS):FactorScore
Touches APL-0 component+3
Modifies public API+2
Changes data format or schema+2
New dependency added+2
Modifies test framework or CI+2
No new tests added+2
APL-2 component+1
APL-3 component+0
CRS ≥ 5: Full Red-Team Protocol required.
CRS 3–4: FMEA summary required.
CRS ≤ 2: Standard review.
9.4 Regression Gates
AES-RG1: No change MAY merge if it causes any of the following:
Test coverage regression at APL-0/APL-1
Performance regression > 5% on any Golden Signal benchmark
Any new TSan or ASan finding
Any new mypy --strict error (Python APL-0/APL-1)
A broken build or test failure
10. ANTI-PATTERNS REGISTRY
10.1 Universal Anti-Patterns (All Domains)
Anti-PatternWhy ForbiddenAES Rule
except: pass or bare exceptSilently swallows errors, violates CleanroomAES-E4
Mutable global stateCreates invisible data races and makes testing impossibleAES-N2Anti-PatternWhy ForbiddenAES Rule
Magic numbers without constantsUndocumented assumptions that will be wrongAES-D2
Copy-paste code (DRY violation)Defect in one copy is silently absent in othersAES-C3
print() debugging in committed codeUnstructured observability, log pollutionAES §7.2
Disabling linters/type-checkers inlineHiding defects rather than fixing themAES-E4
Untested public functionsViolates Cleanroom and DO-178CAES-V1
Hardcoded credentialsCritical security vulnerabilityAES-S4
10.2 ML-Specific Anti-Patterns
Anti-PatternWhy ForbiddenAES Rule
Training without gradient health checksSilent NaN propagation poisons model silentlyAES-ML1
loss = loss.mean() without finite checkNaN in any sample corrupts whole batchAES-ML3
Evaluating on test set during developmentData leakage, invalid performance claimsAES-ML12
torch.manual_seed() but not numpy.random.seed()False reproducibilityAES-ML4
Direct softmax(x) without stabilizationOverflow/underflow on large logitsAES-ML5
Loading weights without shape validationSilent model corruptionAES-E5
Treating NaN loss as a "learning rate problem"NaN is a bug, not a hyperparameterAES-E4
10.3 Quantum Computing Anti-Patterns
Anti-PatternWhy ForbiddenAES Rule
Hardcoded gate anglesUndocumented circuit semanticsAES-QC2
No noise model validation before hardware runCircuit designed for ideal conditions will failAES-QT4
Ignoring transpilation optimizationSub-optimal circuit depth → increased errorAES-QT2
Asserting statevector equality without toleranceFloating-point identity is not circuit identityAES-ML5
Not validating shot sufficiencyMisleading expectation value estimatesAES-QT610.4 HPC Anti-Patterns
Anti-PatternWhy ForbiddenAES Rule
Unaligned SIMD loads in hot pathsPerformance penalty or segfaultAES-HPC4
Implicit OpenMP data sharingSilent data racesAES-HPC8
AoS layout for SIMD-heavy loopsPoor SIMD utilizationAES-HPC5
std::shared_ptr in tight loopsAtomic refcount is O(n) contentionAES-CPP2
False sharing across threadsKills cache coherence performanceAES-HPC6
GPU offload without CPU profilingPremature optimization, may be slowerAES-HPC1
10.5 C++ Anti-Patterns
Anti-PatternWhy ForbiddenAES Rule
C-style cast (float)xBypasses type system, hides errorsAES-CPP3
Raw new / deleteResource leaks, exception-unsafeAES-CPP1
using namespace std; in headersPollutes all including translation unitsAES-CPP9
Uninitialized variablesUndefined behavior per C++ standardAES-CPP7
Discarding [[nodiscard]] return valuesSilent failure modeAES-CPP4
11. APPENDICES
Appendix A: Quick-Reference Checklist
Pre-Commit Checklist (All APLs):[ ] Type annotations complete
[ ] Docstrings on all public APIs
[ ] No dead code
[ ] No magic numbers (use named constants)
[ ] No hardcoded credentials
[ ] All error paths handled explicitly
[ ] Tests written for all new public functions
[ ] Traceability comment included (AES-REQ-xxx)
[ ] CI passes locally (lint, type-check, tests)
Additional for APL-0/APL-1:
[ ] FMEA summary in change record
[ ] Red-Team Protocol executed and logged
[ ] Chronicle baseline metrics recorded
[ ] MC/DC coverage maintained (APL-0) or branch coverage (APL-1)
[ ] Performance benchmark delta within 5%
[ ] Security: OWASP ASVS L2 sweep done
[ ] Reproducibility manifest updated (ML)
[ ] Gradient health checks present (ML training loops)
[ ] Conservation law monitors active (physics sim)
[ ] Noise model validation done (quantum)
Appendix B: AES Severity Decision Tree (Detailed)
START: What does this code do?
1. Does it compute gradients, run SIMD kernels, or execute
safety-critical mathematical operations?
YES ─────────────────────────────────────────► APL-0
2. Does incorrect output produce wrong model predictions,
failed quantum circuit results, or simulation divergence?
YES ─────────────────────────────────────────► APL-1
3. Does it configure the system, process data pipelines,
or manage experiment infrastructure?
YES ─────────────────────────────────────────► APL-2
4. Is it documentation, examples, or non-critical tooling?
YES ─────────────────────────────────────────► APL-3When in doubt: assign the HIGHER severity level.
Classification errors toward leniency are defects.
Appendix C: Glossary
TermDefinition
APLAnvil Priority Level — unified severity classification (APL-0 through APL-3)
AESAnvil Engineering Standard — this document
Chronicle ProtocolMandatory before/after metrics logging for all APL-0/APL-1 changes
CleanroomIBM methodology: fix root cause, never mask symptoms; correctness by prevention
CRSChange Risk Score — numeric risk assessment for proposed changes
DALDevelopment Assurance Level — DO-178C's severity classification
FTAFault Tree Analysis — top-down failure analysis method
FMEAFailure Mode and Effects Analysis — bottom-up failure analysis method
Golden SignalsLatency, Throughput, Errors, Saturation — Google SRE's core four metrics
MC/DCModified Condition/Decision Coverage — DO-178C structural coverage criterion
Red-Team ProtocolAnvil's mandatory adversarial self-review before APL-0 changes
SaguaroAnvil's semantic code intelligence layer
SIMDSingle Instruction, Multiple Data — vector processing paradigm
SLOService Level Objective — agreed performance target
VQCVariational Quantum Circuit — parameterized quantum circuit used in optimization
Appendix D: Source Standards Cross-Reference
AES Rule PrefixPrimary Source Standards
AES-T (Traceability)NASA NPR 7150.2D, NASA-STD-8739.8B, DO-178C
AES-C (Complexity)CMMI Level 5, IEEE 730, IBM Cleanroom
AES-E (Error Handling)CERT C/C++, IBM Cleanroom, C++ Core GuidelinesAES Rule PrefixPrimary Source Standards
AES-D (Documentation)Diátaxis, IEEE 730, Google Style Guides
AES-S (Security)OWASP ASVS, NIST SP 800-53, CWE Top 25
AES-PY (Python)PEP 484/526/8, Google Python Style Guide, SEI CERT
AES-CPP (C++)C++ Core Guidelines, MISRA C++, CERT C++, Google C++ Style
AES-QC/QT (Quantum)IBM Qiskit Best Practices, Cirq patterns
AES-ML (Machine Learning)Google ML Test Score, ML Engineering Best Practices
AES-PS (Physics Simulation)IEEE 754, Higham, NASA simulation practice
AES-HPC (HPC)Intel Architecture Guide, OpenMP spec, CERT C++
AES-V (Verification)DO-178C, JEDEC, IEEE 730, CMMI Level 5
AES-AG (Agent Governance)NASA-STD-8739.8B, DO-178C, AI safety principles
End of Anvil Engineering Standard (AES) v1.0
This document SHALL be treated as APL-1 code: subject to version control, review, and the Chronicle protocol
on every update. The version history of AES is itself a compliance artifact.