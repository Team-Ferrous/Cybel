The Anvil Engineering Standard (AES)
1. PREAMBLE
1.1 Purpose and scope
The Anvil Engineering Standard (AES) defines binding engineering, coding, verification, and
governance rules for all software produced or modified by the Anvil multi‑agent system and its
semantic intelligence layer, Saguaro. It unifies high ‑assurance aerospace and safety standards
with modern ML, quantum computing, physics simulation, and HPC practice.[1] [2] [3] [4] [5] [6]
AES is designed so that code written in compliance can achieve assurance comparable to NASA
and avionics software, while remaining usable in fast‑moving deep learning and quantum R&D
environments.[3] [7] [1]
1.2 Domains governed
AES applies to:
General software engineering in Python 3.12+ and C/C++17, including systems and tooling.
Deep Learning / Machine Learning systems: data pipelines, training loops, evaluation,
serving.[8] [9]
Quantum computing: quantum circuits, variational algorithms (VQCs), hybrid classical–
quantum workflows, and error mitigation/correction.[10] [11] [12]
Physics simulation and numerical methods: PDE solvers, conservation laws,
symmetry‑preserving integrators.[13] [14]
High ‑Performance Computing: CPU‑first design, SIMD/AVX2 kernels, OpenMP or
equivalent threading, cache‑aware data layouts.[5] [15]
Whenever AES conflicts with local style guides or project conventions, AES SHALL take
precedence for Anvil‑generated, reviewed, or modified code.
1.3 Normative language and rule classes
AES uses RFC 2119 terminology:
MUST / SHALL: mandatory; Anvil is not allowed to merge code that violates these without
a documented waiver tied to a specific change.
MUST NOT / SHALL NOT: prohibited.
SHOULD: strong recommendation; violations require justification.MAY: optional guidance.
Every section is normative unless explicitly labeled “Rationale” or “Example”.
AES introduces three categories of rules:
Mandatory Core Rules – apply to all repositories and severities.
Severity‑Scoped Rules – vary by AES Assurance Level (Section 2).
Domain‑Scoped Rules – apply only when relevant domain markers (ML, Quantum, Physics,
HPC) are present.
Saguaro MUST tag each file and change with its language(s), domain(s), and AES Assurance
Level to enable correct rule application.[2] [16]
2. SEVERITY & ASSURANCE FRAMEWORK
2.1 Unified assurance levels
AES unifies DO‑178C DAL levels, ISO 26262 ASIL levels, and IEC 61508 SIL levels into a single AES
Assurance Level (AES‑L) scale from 0–4.[4] [17] [3]
AES‑LTypical impact if faultyApprox. mappingL4 – CriticalCatastrophic physical or
large‑scale financial damage;
irrecoverable scientific loss; core
security breachDO‑178C DAL A /
ISO 26262 ASIL
D / IEC 61508
SIL 4L3 – HighSevere but bounded impact; major
degradation of core experiments
or systemsDAL B–C / ASIL
B–C / SIL 2–3L2 – MediumLocalized failures; recoverable;
mainly affects productivity or
localized correctnessDAL D / ASIL A /
SIL 1L1 – LowNon‑safety‑critical; failures are
nuisances, not hazardsDAL E / QMExperimental notebooks,
prototypes, one‑off research
scripts. [3] [4]
L0 –
InformationalDocumentation, comments,
non‑executable artifacts—Markdown docs, diagrams, design
notes.
Existing P0–P3 severity tags MUST be mapped as:
P0 ⇒ AES‑L4
P1 ⇒ AES‑L3
P2 ⇒ AES‑L2
P3 ⇒ AES‑L0/L1 (depending on whether code executes)
Examples
SIMD/HPC kernels used in
production risk‑sensitive models;
safety‑critical controllers;
cryptographic or auth modules. [3]
[4] [17]
Training loops for major models;
core data pipelines; quantum
compiler passes used for
production experiments. [3] [18] [5]
Internal tools, configuration
generators, dashboards feeding
decisions, non‑critical services. [19]
[5]Rationale: DO‑178C, ISO 26262, and IEC 61508 all tie process rigor to consequence of failure; AES
unifies this into one ladder for all domains.[17] [3] [4]
2.2 Verification requirements by level
Structural coverage and traceability requirements per level are:
AES‑LRequirements
coverageStructural coverage
targetExtra constraints
L4100% of high‑ and
low‑level
requirements100% MC/DC + 100%
statement + decision
coverageZero dead code; full
requirement→design→code→test→result traceability;
independent review/IV&V. [3] [20] [2]
L3100% requirementsStatement + decision
coverage; robustness
tests for abnormal
inputsNo untraceable code; rigorous fault injection where
relevant. [3] [19]
L2All “must not fail
silently” featuresStatement coverage
on changed code;
sampling MC/DC for
risky logicAutomated regression suite; static analysis clean for
high‑risk patterns. [3] [5]
L1Best‑effort; smoke
testsLine or branch
coverage on main
flowsFast feedback prioritized over completeness.
L0N/AN/ADocs must remain consistent with code; doc CI
checks SHOULD be in place. [21]
Rationale: DO‑178C mandates increasing structural coverage with DAL severity and forbids
uncovered, non ‑deactivated code for high levels; NASA and IEC 61508 require bi‑directional
traceability and safety‑lifecycle discipline.[20] [6] [2] [3] [5]
2.3 Assigning AES levels
Anvil MUST assign AES‑L based on:
1. Hazard analysis / risk assessment: severity × exposure × controllability or detectability,
adapting ISO 26262 HARA and IEC 61508 risk analysis to software components.[18] [4] [17]
2. Role in control loops:
Any code directly controlling actuators, critical HPC kernels, or security boundaries
MUST be at least L3; if compromise or failure can propagate widely, it MUST be L4.
3. Scientific or business irreversibility:
Irrecoverable experiment runs, stateful training on very expensive clusters, or
non ‑replayable quantum experiments MAY be escalated one level.
A simple decision tree MUST be provided in Appendix B; Saguaro MUST implement it as a
rule‑based classifier and suggest a default AES‑L which a human can override only with
rationale.3. UNIVERSAL CODING MANDATES
3.1 Defect prevention and cleanroom mindset
1. Formal prevention over patching
For L3–L4 code, designs MUST prefer constructs known to avoid entire classes of
defects (e.g., RAII, bounds‑checked APIs, immutability) over ad‑hoc defensive checks.
[22] [23]
Statistical testing MUST be used to demonstrate reliability for L3–L4 components
whose behavior is probabilistic (ML, Monte‑Carlo, noisy quantum).[24] [23] [22]
Rationale: Cleanroom and NASA standards emphasize defect prevention, formal
verification, and statistical testing to certify reliability instead of relying on debugging
alone.[23] [1] [22]
2. Zero undefined behavior
In C/C++, any construct that can trigger language‑level undefined or “erroneous”
behavior (out‑of‑bounds, signed overflow, data races, invalid pointer casts, etc.) MUST
NOT appear in L3–L4 code; L1–L2 MUST have static‑analysis checks and documented
exceptions only.[25] [26] [27] [28]
Rationale: MISRA and CERT both put “no undefined behavior” as an overarching rule,
because compilers may optimize under the assumption that it never occurs, making bugs
invisible to tests and dangerous in production.[26] [29] [25]
3.2 Traceability
1. Bi‑directional traceability
For AES‑L2 and above, every requirement MUST be traceable to:
One or more design artifacts
One or more code artifacts
One or more verification artifacts (tests, proofs, analyses)
Recorded verification results linked to CI runs or formal reports [30] [31] [2] [5]
Traceability MUST be bi‑directional: from requirement to code/tests and from any code
or test back to at least one requirement (or explicitly designated “supporting
infrastructure”).[30] [2] [5]
2. Tool‑enforced links
Saguaro MUST ensure that each merged change references at least one requirement ID
(or explicitly declares “no requirement change”; e.g., refactor only) and updates trace
links if it adds, modifies, or deprecates behaviors.[2] [30]
Rationale: NASA NPR 7150.2 and IEC 61508 demand bi‑directional traceability to ensure
every requirement is implemented and verified, and that there is no surplus or orphan code.
[6] [16] [5] [30] [2]3.3 Complexity bounds
1. Algorithmic complexity
Hot paths (any code where telemetry shows ≥5% of CPU or wall‑time) MUST use
algorithms with average complexity no worse than
;
or worse in hot
paths is prohibited unless proved negligible and explicitly waived.[7] [32]
For L3–L4, any
+ algorithm MUST be justified in a design review and accompanied
by empirical scaling plots to show safety margins.
2. Code complexity
A single function or method MUST NOT have:
More than one responsibility at AES‑L3–L4.
Cyclomatic complexity above a configured threshold (e.g., 10–15), unless
refactoring is demonstrably impossible; in such cases, exhaustive tests and MC/DC
analysis MUST be provided.[20] [3]
Rationale: High structural and algorithmic complexity directly undermines MC/DC
coverage, maintainability, and defect containment; CMMI Level 5 demands quantitative
control of process and quality metrics.[32] [3] [7] [20]
3.4 Error handling philosophy
AES combines three philosophies:
Explicit error values (Go, Rust Result) for most public APIs.
Fail fast & restart (“let it crash”) at well‑defined process or service boundaries.[33] [34]
No masking in L3–L4: safety‑critical code MUST NOT continue silently after detecting
invariant violations.[1] [3] [23]
Rules:
1. Function contracts
Public functions and methods in Python and C++ MUST surface failure explicitly:
Python: return Ok[T]/Err[E]‑style results or raise well‑typed exceptions; do not
return ambiguous “magic values”.
C++: return status enums/small error types, or use expected<T,E>/similar;
exceptions MAY be disabled in core libraries, following Google C++ style.[35] [36]
2. No silent swallowing
Catch ‑all handlers (catch (...), bare except:) are forbidden in L3–L4 and SHOULD NOT
appear elsewhere; handlers MUST either:
Correct and log at WARN/INFO, or
Escalate by rethrowing or by propagating an error value with structured context.
[37] [38] [36]
3. Crash domainsIt is acceptable—and sometimes preferred—for a process or worker to terminate on
unrecoverable internal errors, provided:
State is either transactional or reconstructable.
A supervisor or orchestration layer restarts the process (Erlang‑style supervision
tree concept).[34] [39] [33]
Rationale: Erlang’s “let it crash” with supervision trees yields highly reliable systems while
keeping code simple; combined with explicit error types and full test coverage, it avoids
masked errors yet keeps complexity manageable.[40] [33] [34]
3.5 Documentation requirements
1. Diátaxis structure
Project docs MUST be organized according to Diátaxis: Tutorials, How‑to guides,
Reference, Explanation.[41] [21]
For L3–L4 systems, each major subsystem MUST have:
At least one tutorial for common tasks
At least one how‑to per key operation mode
Complete API reference
Explanations of core design and trade‑offs
2. API documentation
All public APIs MUST have language‑appropriate docstrings or comments sufficient to
call them without reading implementation (Google Python Style / C++ style).[38] [36]
3. Doc–code synchronization
Any L2+ code change that affects externally observable behavior MUST update
reference docs in the same change list unless the docs are generated from
types/signatures automatically.[42] [41]
Rationale: Diátaxis and Google documentation guidance show that separating learning
material, task recipes, reference, and explanations produces clearer documentation that is
easier to maintain.[21] [38] [41]
3.6 Security baseline
1. Security levels
By default, all network‑reachable or user‑reachable components MUST meet at least
OWASP ASVS Level 2 requirements; components responsible for authentication,
authorization, or high ‑value data MUST meet Level 3.[43] [44] [45]
2. CWE Top 25 defense
AES‑L2+ code MUST demonstrate mitigations against the current CWE Top 25 classes
relevant to its stack (injection, out‑of‑bounds, missing auth, deserialization, SSRF, etc.).
[46] [47] [48]Dependency and SAST tools MUST be configured to flag CWE Top 25 weaknesses as
blocking for L3–L4.
3. NIST SP 800‑53 alignment
Design and code reviews MUST consider at least:
Access Control (AC): enforce least privilege.
Audit & Accountability (AU): ensure logs for critical operations.
System & Communications Protection (SC): secure channels and input validation.
Configuration Management (CM): managed, versioned configs.[49] [50]
Rationale: OWASP ASVS defines three graded security assurance levels; NIST SP 800‑53
control families provide a comprehensive catalog of security and privacy controls for
applications and infrastructure.[45] [50] [43] [49]
3.7 Concurrency safety
1. No unstructured sharing
Shared mutable state across threads without synchronization is forbidden; C++ code
MUST use atomics, mutexes, or higher‑level concurrency primitives; Python code
MUST use process‑level concurrency or thread‑safe structures.[51] [52] [15]
2. Race‑free design
For L3–L4, concurrency policies (which data are guarded by which locks, or which are
thread‑local/immutable) MUST be documented, and static or dynamic thread‑safety
analysis SHOULD be run where available.[53] [51]
3. Avoid deadlocks
When multiple locks are required, they MUST be acquired in a consistent global order;
deadlock‑prone patterns (nested locks without consistent order) MUST NOT appear in
L3–L4 code.[52] [15]
Rationale: Data races in C/C++ are undefined behavior and can produce arbitrarily incorrect
results; static thread‑safety analysis and disciplined locking make concurrent code
analysable and testable.[26] [51] [52] [53]
4. LANGUAGE‑SPECIFIC STANDARDS
4a. Python Standards
1. Style and layout
Code MUST follow PEP 8 for layout and naming (79–100 character lines, snake_case
functions, CamelCase classes, constants in UPPER_SNAKE_CASE).[54] [55] [56] [37]
Imports MUST be explicit and ordered: stdlib, third‑party, local; wildcards (from x
import *) are forbidden in L2+.[37] [38]
2. TypingPublic functions, methods, and module‑level variables in L2+ code MUST have PEP
484/526‑style type annotations; mypy or equivalent MUST run in CI on such code.[57]
[55] [37]
3. Error handling
Bare except: is forbidden; catch the minimal specific exception needed.[38] [37]
Code MUST NOT use exceptions for control flow in hot paths; instead, structure loops
and conditionals to avoid frequent exceptions.
4. Memory and resources
Use context managers (with) for resources (files, network, locks) to guarantee cleanup,
per PEP 8 recommendations.[37]
5. Security
Avoid dynamic execution of untrusted input (eval, exec); any usage MUST be guarded
by strict sanitization and isolated contexts.
When adapting CERT Java and C++ rules, ensure:
Input validation and encoding to prevent injection wherever Python builds queries
or shell commands.
Safe integer usage in Python C‑extensions or array indexing.[29] [58]
6. Example (docstring and typing)
from typing import Sequence
def normalize_probabilities(values: Sequence[float]) -> list[float]:
"""Return a numerically stable probability vector.
Args:
values: Unnormalized, non-negative scores.
Returns:
A list of probabilities summing to 1.0.
Raises:
ValueError: If any value is negative or all values are zero.
"""
if not values:
raise ValueError("values must be non-empty")
if any(v < 0 for v in values):
raise ValueError("values must be non-negative")
total = sum(values)
if total == 0.0:
raise ValueError("sum of values is zero")
return [v / total for v in values]4b. C/C++ Standards
1. Style
Follow the C++ Core Guidelines and Google C++ Style Guide for naming, layout,
includes, and header guards; use namespaces instead of global symbols, #pragma once
or include guards in every header.[59] [60] [36]
2. Resource management (RAII)
All owning resources (heap memory, file descriptors, locks, etc.) MUST be managed by
RAII types (std::unique_ptr, std::shared_ptr, custom RAII wrappers).[61] [62] [63]
Raw new/delete are forbidden in L3–L4, and SHOULD NOT appear elsewhere outside of
RAII wrappers.[62] [59]
3. MISRA / CERT alignment
L3–L4 C/C++ MUST comply with a MISRA‑like safe subset: no dangerous casts, strict
pointer aliasing discipline, and no reliance on implementation ‑defined behavior
without documentation.[64] [25] [26]
CERT rules on integers, memory, and strings MUST be observed: no unchecked integer
arithmetic for sizes, no unsafe string functions, and explicit bounds on allocations.[27]
[65] [29]
4. Exceptions
For performance‑critical or embedded‑like subsystems, exceptions MAY be disabled
following Google’s “no exceptions” practice; such code MUST return error codes or
result types and MUST NOT throw from library or API code.[36] [35]
5. Example (RAII and error status)
class FileReader {
public:
static StatusOr<FileReader> Open(const std::string& path);
Status ReadAll(std::string* out) const;
private:
explicit FileReader(int fd) : fd_(fd) {}
int fd_;
};
StatusOr<FileReader> FileReader::Open(const std::string& path) {
int fd = ::open(path.c_str(), O_RDONLY);
if (fd < 0) {
return Status::FromErrno(errno, "Failed to open file");
}
return FileReader(fd);
}
Status FileReader::ReadAll(std::string* out) const {
// RAII for buffer, etc.
}4c. Quantum Circuit Standards (Qiskit/Cirq)
1. Transpilation discipline
All circuits targeting hardware MUST be transpiled for a specific backend using an
explicit optimization and resilience level; circuits MUST only use basis gates supported
by the backend.[11] [12] [10]
For production experiments, transpilation configuration (optimization level, layout
strategy, noise‑aware mapping) MUST be versioned and tied into traceability.
2. Noise‑aware programming
Circuits MUST be optimized for depth and two‑qubit gate count to reduce decoherence;
Qiskit transpiler levels 1–2 are preferred for fast iteration, with level 3 reserved for final
runs where added compile time is acceptable.[66] [10] [11]
When error mitigation is used (ZNE, PEC, readout mitigation), circuits and calibration
data MUST be logged and tied to their measurement results.[12]
3. Error correction and layout
When using logical qubits or surface codes, code distances, syndrome extraction
schedules, and decoding algorithms MUST be documented and versioned; circuits
MUST respect topology and constraints of the code (e.g., lattice connectivity).[67]
4. Example (Qiskit transpilation)
from qiskit import QuantumCircuit, transpile
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()
# Backend-specific transpilation for a target device
transpiled = transpile(
qc,
backend=backend,
optimization_level=2, # depth and noise-aware optimizations
)
5. DOMAIN‑SPECIFIC STANDARDS
5a. Deep Learning / ML Standards
5a.1 Training loop integrity
1. No hidden state mutation
Training loops MUST make all sources of randomness explicit (seeds, RNG streams)
and ensure that the same configuration yields reproducible behavior for L3–L4
experiments (given fixed libraries and hardware).[68] [9] [8]2. Gradient and loss checks
Every L2+ training loop MUST:
Detect NaNs/Infs in losses and gradients and abort or automatically back off
(smaller learning rate, gradient clipping) with metrics logged.
Log gradient norms and their statistics periodically.[9] [69] [8]
3. No silent divergence
If any iteration produces non ‑finite gradients or parameters, the system MUST mark
the run as failed unless an explicit, tested recovery mechanism is in place.
Rationale: Google’s ML Test Score and reliability guidance emphasize explicit testing for
NaNs, input distributions, feature ranges, and reproducibility, as well as monitoring for
numerical pathologies during training and serving.[69] [8] [68] [9]
5a.2 Data pipeline validation
1. Schema and distribution checks
Data pipelines MUST validate:
Schema compatibility (field names, types, shapes).
Value ranges and enums for categorical features.
Distribution shifts relative to training baselines (simple statistics for L2; richer drift
detection for L3–L4).[70] [8] [9]
2. Test categories
ML pipelines MUST adopt tests across four ML Test Score categories:
Data tests
Model tests
ML infrastructure tests
Monitoring tests [8] [68] [70]
5a.3 Model versioning and reproducibility
1. Versioned artifacts
Each model version MUST be traceable to:
Exact training code revision
Data snapshot / dataset version
Hyperparameters
Random seeds and environment (libraries, CUDA, etc.) [9] [70] [8]
2. Reproducible training
For L3–L4 models, training MUST be repeatable to within specified tolerances (e.g.,
within a small delta in metrics) by replaying configuration and data.5a.4 Numerical stability in ML
1. Stable operations
Probability computations MUST use numerically stable formulations:
Log‑sum‑exp for softmax normalization.
Normalization and clipping for gradient and loss computations.[71] [72] [73]
Summations over large tensors SHOULD use strategies equivalent to pairwise or
compensated summation when statistics are critical (e.g., large reductions for loss or
metrics).[72] [74] [71]
2. Mixed precision
When using mixed precision, scaling factors, loss scaling, and casting strategies MUST
be explicitly tested for overflow/underflow and integrated into the telemetry (e.g.,
counts of overflowed gradients).
5b. Quantum Computing Standards
5b.1 Circuit design discipline
1. Topology‑aware design
Logical circuits SHOULD be designed with anticipated coupling maps in mind
(nearest‑neighbor, heavy‑hex lattices, etc.), reducing SWAP overhead during
transpilation.[10] [11] [12]
2. Depth and gate counts
L3–L4 quantum workloads MUST:
Track depth, two‑qubit gate count, and measurement count as primary cost
metrics.
Set upper bounds per experiment type and enforce them in CI.[75] [10]
5b.2 Noise‑aware programming and error mitigation
1. Noise modeling
For simulators, circuits MUST specify noise models (e.g., depolarizing, amplitude
damping) that approximate target hardware; simulations MUST test sensitivity of
algorithms to noise variations.[76] [77] [67]
2. Error mitigation controls
When using ZNE or PEC, the number of scaled noise levels, extrapolation methods,
and validation circuits (e.g., GHZ, idle sequences) MUST be documented and tested.[66]
[12]5b.3 Classical–quantum interface
1. Deterministic orchestration
Hybrid loops (e.g., VQCs) MUST treat quantum execution as a pure function of classical
parameters; no hidden dependence on global mutable state is allowed.
2. Sampling and confidence
For each measured quantity (e.g., energy estimate), the number of shots and resulting
confidence intervals MUST be recorded; stopping criteria MUST be based on
confidence or variance thresholds.[78] [76] [66]
5c. Physics Simulation Standards
5c.1 Conservation laws and invariants
1. Invariant monitoring
For conservation laws (mass, momentum, energy, charge), simulations MUST monitor
discrete totals and flux balances over time and log deviations; acceptable tolerances
MUST be stated and tested.[79] [14] [13]
2. Entropy and stability
Schemes SHOULD be entropy‑stable or well‑balanced when solving conservation laws
with shocks or steady states, following modern numerical methods literature (e.g.,
entropy‑stable WENO or DG schemes).[14] [76] [13] [78]
5c.2 Numerical method selection
1. Method criteria
For each solver, the following MUST be documented:
Order of accuracy
Stability conditions (e.g., CFL constraints)
Known failure modes (e.g., Gibbs phenomena near shocks)
2. Verification tests
Simulations MUST include:
Method of manufactured solutions (MMS) or equivalent for verifying convergence
rates.
Benchmark problems with known analytical or high ‑accuracy reference solutions.
[79] [76] [13]5d. High‑Performance Computing Standards
5d.1 CPU‑first / SIMD‑first rules
1. Hot‑path identification
Profiling MUST identify hot kernels; these MUST be implemented with careful control
of memory layout, vectorization, and branching.
2. SIMD discipline
L3–L4 performance‑critical code MUST:
Use compiler vectorization reports and intrinsics or libraries when needed.
Align data to natural SIMD widths.
Avoid unpredictable branches inside tight loops.
5d.2 Memory alignment and cache discipline
1. Data locality
Arrays SHOULD be laid out contiguously and iterated with unit stride for cache
efficiency; AoS vs SoA layouts must be chosen based on vectorization and access
patterns.
2. False sharing
Shared structures across threads MUST be padded or partitioned to avoid false sharing
on cache lines.
5d.3 Concurrency and threading
1. Deterministic parallel sections
For L3–L4, parallel regions (OpenMP, TBB, custom threads) MUST be structured such
that reductions and updates are deterministic or at least numerically robust to
reordering using stable summation methods for floating point.[74] [71] [72]
2. Thread safety and affinity
Thread pools, pinning, and NUMA layouts SHOULD be tuned for predictable
performance; observability MUST include per‑thread CPU utilization and cache miss
metrics.
6. VERIFICATION & TESTING FRAMEWORK
6.1 Test taxonomy
Anvil recognizes the following test types:
Unit tests – small, deterministic tests per function or class.
Integration tests – verify interactions between components or services.Property‑based tests – assert invariants across randomly generated inputs.
Fuzz tests – feed randomized, possibly malformed inputs to find robustness issues.
Benchmark tests – measure time, memory, and throughput, tied into golden performance
baselines.
Statistical/operational tests – Cleanroom‑style statistical testing to estimate failure rates
under realistic usage distributions.[80] [22] [23]
6.2 Coverage requirements
Coverage targets per AES‑L are given in Section 2.2; implementation details:
Statement and branch coverage MUST be measured with tooling and reported in CI.[81] [3]
MC/DC (Modified Condition/Decision Coverage) MUST be applied to all safety‑critical
decisions in L4 code and SHOULD be considered for key decisions in L3.[3] [20]
Uncovered code MUST be justified as deactivated, unreachable under all configurations, or
removed as dead code.[81] [3]
Rationale: DO‑178C and NASA standards require MC/DC coverage for high ‑criticality software
because it provides strong assurance that all logical combinations driving decisions have been
exercised.[20] [3]
6.3 ML‑specific testing
In addition to conventional tests:
Data tests: schema validation, distribution checks, known ‑corrupt samples, drift detection.
[68] [8] [9]
Model tests: golden ‑set evaluation, slice performance, invariants such as monotonicity or
symmetry where applicable.[82] [8] [68]
Infrastructure tests: pipeline continuity tests, configuration compatibility, correct feature
parity between training and serving.[8] [68] [9]
Monitoring tests: ensure telemetry pipelines can detect shifts in performance and data.[83]
[9] [8]
6.4 Quantum‑specific testing
Simulator vs. hardware: circuits MUST first be validated on an ideal simulator, then on a
noisy simulator with calibrated noise models, before being deployed to hardware.[77] [76]
[66]
Noise model validation: calibration circuits (T1/T2, randomized benchmarking, GHZ) MUST
be used to validate noise models periodically.[76] [77] [66]
Error mitigation validation: error‑mitigated results MUST be cross‑checked against known
small examples and sanity bounds.7. OBSERVABILITY & TELEMETRY
7.1 Golden Signals per domain
The four Golden Signals—latency, traffic, errors, saturation—MUST form the basis of
observability across all domains.[84] [85] [86] [87]
AES standardizes them as:
Latency – step time, TTFT, p50/p95/p99 response; per‑epoch time for training; circuit
execution time for quantum; timestep cost for simulations.
Traffic – tokens/sec, samples/sec, GFLOPS, circuits/sec, cells updated/sec.
Errors – NaNs/Infs, gradient failures, exceptions, failed jobs, non ‑convergent iterations,
hardware timeouts.
Saturation – CPU utilization, thread utilization, SIMD coverage, memory/RSS,
GPU/accelerator utilization.[85] [84] [9]
Rationale: Google SRE and follow‑on literature show that these four signals give early warning
for most production issues and align closely with user experience and capacity planning.[86] [87]
[84] [85]
7.2 OpenTelemetry and trace correlation
All services and long‑running jobs MUST emit structured logs, metrics, and traces using
OpenTelemetry or compatible tooling.[88] [89] [90] [91]
Logs MUST include trace and span identifiers so they can be correlated with traces and
metrics automatically.[89] [90] [88]
Histograms SHOULD be used to capture latency distributions (for p95/p99) instead of only
averages.[91] [88]
7.3 F1‑inspired telemetry
Hot training runs and quantum experiments MUST expose high ‑granularity telemetry
analogous to F1 telemetry: fine‑grained internal metrics that allow real‑time adjustments
and post‑run analysis of compounding effects (e.g., gradient norms, learning‑rate
schedules, hardware temperature, memory pressure).[92] [93] [94] [95]
8. AGENT WORKFLOW GOVERNANCE
8.1 Anvil’s obligations
For each change, Anvil MUST:
1. Determine AES‑L and domain tags.
2. Enforce all mandatory AES rules relevant to that level and domain.
3. Ensure:Lint, type checks, and static analysis pass for L2+.
Tests and coverage meet the level’s thresholds (or are explicitly waived).
Traceability links are updated.
Anvil MUST refuse to auto‑merge changes that break core AES constraints (e.g., undefined
behavior, missing traceability, coverage shortfalls for L3–L4) without explicit human waiver.
8.2 Code generation guardrails
When generating or refactoring code, Anvil MUST:
Prefer safe language subsets (MISRA/CERT‑aligned patterns for C/C++; PEP 8 + typing for
Python).[25] [29] [26] [38] [37]
Avoid introducing new dependencies without:
License compatibility check.
Security and supply‑chain assessment (see Section 3.6 and 8.3).
Generated code MUST include:
Tests appropriate to severity and domain.
Docstrings or comments matching Diátaxis categories for public APIs.
8.3 Supply chain and dependency management
Dependencies MUST be pinned for reproducibility and scanned for vulnerabilities.
For critical pipelines, build and artifact provenance MUST comply with SLSA Level 2+ where
feasible: tamper‑resistant, signed provenance generated by the build service itself, ideally
using Sigstore or equivalent tooling.[96] [97] [98] [99]
Rationale: SLSA and Sigstore provide frameworks and tools for generating and verifying
provenance metadata to guard against supply‑chain attacks; Level 2+ introduces signed,
tamper‑resistant provenance.[97] [98] [99] [96]
8.4 Self‑verification and red ‑teaming
Before completing an autonomous task, Anvil MUST:
Run a self‑review: re‑analyze code and tests it just generated using independent
prompts/models where possible.
Execute a red‑team checklist:
Complexity checks against AES limits.
FMEA/FTA‑style reasoning for critical paths.[19] [5] [2]
OWASP ASVS checklists for exposed endpoints.[100] [43] [45]
Regression gate: confirm that Chronicle protocol metrics show no regression.9. CODE REVIEW & CHANGE MANAGEMENT
9.1 Review requirements by level
L4: At least:
One domain expert reviewer
One independent assurance or safety reviewer
All comments resolved before merge [101] [1] [3]
L3: At least one peer engineer plus an owner of the affected module.[102] [103] [101]
L2–L1: At least one peer review; reviewers MAY be from the same team.
L0 (docs): Review by a maintainer for structure and correctness.
Rationale: Google’s review practices show that multiple roles (peer, code owner, language/style
readability) yield both correctness and maintainability while remaining scalable.[104] [105] [103]
[101] [102]
9.2 Change risk assessment
Each change MUST be tagged with:
Impacted AES‑L components.
Type of change (algorithmic, configuration, refactor, documentation).
Risk rating based on:
Potential blast radius
Complexity
History of instability in that area
Higher‑risk changes MUST be split into smaller CLs where possible, following Google’s
guidance that small, coherent change lists make reviews more effective.[103] [101] [102]
9.3 Regression gates
CI MUST run:
All relevant unit and integration tests.
Coverage measurement for changed code.
Static analysis and security scans.
Performance benchmarks for hot paths, compared against a baseline Chronicle for
regressions beyond thresholds.
Changes that degrade performance or reliability beyond agreed budgets MUST NOT be merged
without a documented SLO/SLA adjustment and justification.[87] [84] [85] [9]10. ANTI‑PATTERNS REGISTRY
This section lists patterns that are explicitly forbidden or heavily discouraged in AES‑governed
code.
10.1 General anti‑patterns (all domains)
Dead code: Unreachable branches, unused functions, and obsolete flags MUST be
removed; DO‑178C and NASA standards treat dead code as a coverage and safety risk.[81] [3]
[20]
Global mutable state: Except for process‑wide configuration and logging, global mutable
variables are forbidden in L3–L4 and strongly discouraged elsewhere.
“Magic constants” without explanation: Non ‑obvious numeric constants MUST have
named constants and comments.
Ad‑hoc security: Custom crypto, encoding, or auth mechanisms are forbidden; standard
libraries and audited implementations MUST be used.
10.2 Python anti‑patterns
Bare except: or catching broad exceptions without rethrowing; masks real issues and
violates PEP 8 guidance.[38] [37]
Dynamic monkey‑patching of core libraries in production code; allowed only in controlled
testing contexts.
Use of eval/exec on untrusted input; violates secure coding guidelines and opens injection
vulnerabilities.[50] [45]
10.3 C/C++ anti‑patterns
Manual memory management with raw pointers instead of RAII or smart pointers.[61] [62]
[59]
Pointer aliasing tricks and unsafe casts violating MISRA and CERT recommendations.[64]
[29] [25] [26]
Silent integer overflow in length/size calculations; these are known to lead to buffer
overflows.[106] [107] [65] [108] [27]
Sharing data across threads without documented synchronization (data races).[51] [52] [53]
10.4 ML anti‑patterns
Training on test or validation data, or using validation metrics to tune hyperparameters
without a separate held‑out test set.[68] [8]
Relying solely on training loss/accuracy with no monitoring for drift or slice disparities in
production.[109] [82] [9] [8]
Ignoring NaNs/Infs and continuing training by replacing them with zeros or clipping
silently.10.5 Quantum anti‑patterns
Ignoring the device’s qubit connectivity and error rates when preparing circuits; leads to
unnecessarily deep, noisy circuits.[110] [11] [12] [10]
Running large‑scale circuits on hardware without prior validation on simulators and noise
models.[77] [66] [76]
Using error mitigation (ZNE, PEC) without calibrating its effect or reporting effective
sample complexity and confidence.[12] [66]
10.6 Physics/HPC anti‑patterns
Not checking or logging conservation law violations in systems that are supposed to
conserve quantities.[13] [14] [79]
Relying on naive summation of large floating‑point arrays where sign or magnitude
cancellation is important; use compensated or pairwise summation.[73] [71] [72] [74]
Introducing thread‑unsafe global accumulators or reductions; leads to data races and
non ‑deterministic results.[15] [52]
11. APPENDICES (OVERVIEW)
11.1 Quick‑reference checklists (indicative structure)
Each project SHOULD maintain tailored checklists derived from AES. At minimum:
General AES checklist:
AES‑L assigned and documented?
Traceability links updated?
Lint, type checks, and static analysis clean?
Tests written and coverage within thresholds?
Observability (metrics, logs, traces) added or updated?
ML checklist:
Data schema tests?
NaN/Inf detection in training?
Drift monitoring and slice tests?
Model versioning metadata captured?
Quantum checklist:
Circuit depth and gate counts within bounds?
Simulated vs. hardware tests?
Noise model and mitigation documented?
Physics/HPC checklist:Invariants monitored and within tolerances?
Convergence or MMS tests?
SIMD and threading verified with profiling?
11.2 Severity decision tree (high‑level)
A decision tree SHOULD include branches such as:
Does failure lead to safety, security, or large‑scale financial or scientific harm?
Yes → L4
No → Next question
Does it control or feed high ‑impact ML/quantum experiments or services?
Yes → L3
No → Next
Is it operational tooling or infrastructure?
Yes → L2
No → L1 or L0 depending on executability
This tree MUST be encoded into Saguaro’s classification logic for consistent AES‑L assignment.
11.3 Glossary of synthesized terms
AES‑L – Anvil Engineering Standard Assurance Level (0–4).
Golden Signals – Latency, traffic, errors, saturation metrics used to characterize system
health.[84] [85] [86] [87]
Chronicle protocol – The requirement to log baseline metrics, changes, and new metrics for
each significant change to enable before/after comparison.
Red‑team protocol – Structured attempts by Anvil or humans to find weaknesses through
complexity checks, FMEA/FTA, OWASP reviews, and regression gates.[19] [45] [2]
Cleanroom – Software process emphasizing formal specification, human verification, and
statistical testing to achieve certifiable reliability.[111] [22] [23] [80]
SLSA – Supply‑chain Levels for Software Artifacts; a framework for build and provenance
security.[98] [99] [96]
If Anvil follows this Anvil Engineering Standard rigorously—including severity classification,
universal mandates, domain ‑specific rules, and governance workflows—it will produce
software whose quality, safety, and reliability are aligned with the expectations of NASA flight
software and modern high ‑assurance ML and quantum systems, while still enabling rapid
research and iteration.[4] [5] [1] [2] [3] [9] [8]
⁂
1. https://standards.nasa.gov/sites/default/files/standards/NASA/B/0/NASA-STD-87398-Revision-B.pdf2. https://nodis3.gsfc.nasa.gov/displayDir.cfm?t=NPR&c=7150&s=2D
3. https://www.rapitasystems.com/do178c-testing
4. https://www.parasoft.com/learning-center/iso-26262/what-is/
5. https://www.qa-systems.com/solutions/iec-61508/
6. https://en.wikipedia.org/wiki/IEC_61508
7. https://www.wibas.com/cmmi/maturity-level-5-optimizing-cmmi-dev
8. https://research.google.com/pubs/archive/45742.pdf
9. https://docs.cloud.google.com/architecture/framework/perspectives/ai-ml/reliability
10. https://qiskit.qotlabs.org/docs/guides/transpile
11. https://quantum.cloud.ibm.com/docs/api/qiskit/transpiler
12. https://quantum.cloud.ibm.com/learning/courses/quantum-computing-in-practice/running-quantum-circuits
13. https://epubs.siam.org/doi/book/10.1137/1.9781611975109
14. https://www.math.umd.edu/~tadmor/references/files/Puppo Semplice Entropy PhysProc2011.pdf
15. https://en.cppreference.com/w/cpp/atomic.html
16. https://quality.arc42.org/standards/iso12207
17. https://www.perforce.com/blog/qac/what-iec-61508-safety-integrity-levels-sils
18. https://learn.arm.com/learning-paths/automotive/openadkit2_safetyisolation/1c_iso26262/
19. https://www.einfochips.com/blog/road-vehicles-functional-safety-a-software-developers-perspective/
20. https://swehb.nasa.gov/plugins/viewsource/viewpagesrc.action?pageId=146539500
21. https://diataxis.fr
22. http://www.cs.toronto.edu/~chechik/courses07/csc410/mills.pdf
23. https://en.wikipedia.org/wiki/Cleanroom_software_engineering
24. https://www.ti.com/support-quality/reliability/reliability-testing.html
25. https://www.mathworks.com/help/bugfinder/ref/misrac2023rule4.1.3.html
26. https://learn.adacore.com/courses/SPARK_for_the_MISRA_C_Developer/chapters/07_undefined_behavior.ht
ml
27. https://www.sei.cmu.edu/asset_files/Presentation/2011_017_001_51345.pdf
28. https://pvs-studio.com/en/blog/posts/cpp/1136/
29. https://abougouffa.github.io/awesome-coding-standards/sei-cert-c-2016.pdf
30. https://nasa.github.io/progpy/npr7150.html
31. https://www.mathworks.com/content/dam/mathworks/mathworks-dot-com/images/responsive/supporting/
solutions/aerospace-defense/certification-standards/poster-mathworks-nasa-npr7150-2d-workflow.pdf
32. https://www.6sigma.us/process-improvement/capability-maturity-model-integration-cmmi/
33. https://www.linkedin.com/posts/ahmadalnawayseh_the-let-it-crash-philosophy-which-is-most-activity-7309
685709685116928-ukcn
34. https://www.cs.tufts.edu/cs/21/notes/error_handling/index.html35. https://stackoverflow.com/questions/5184115/google-c-style-guides-no-exceptions-rule-stl
36. https://google.github.io/styleguide/cppguide.html
37. https://peps.python.org/pep-0008/
38. https://google.github.io/styleguide/pyguide.html
39. https://erlang.org/pipermail/erlang-questions/2011-January/055531.html
40. https://oneuptime.com/blog/post/2026-02-02-elixir-supervisors-fault-tolerance/view
41. https://idratherbewriting.com/blog/what-is-diataxis-documentation-framework
42. https://blog.sequinstream.com/we-fixed-our-documentation-with-the-diataxis-framework/
43. https://codific.com/owasp-asvs-a-comprehensive-overview/
44. https://nearshore-it.eu/articles/owasp-asvs/
45. https://github.com/OWASP/ASVS/blob/master/4.0/en/0x03-Using-ASVS.md
46. https://www.helpnetsecurity.com/2024/11/21/cwe-top-25-most-dangerous-software-weaknesses/
47. https://www.sans.org/top25-software-errors
48. https://cwe.mitre.org/top25/
49. https://www.isms.online/nist/nist-sp-800-53/
50. https://www.kiuwan.com/blog/how-nist-sp-800-53-revision-5-affects-application-security/
51. https://www.sei.cmu.edu/blog/thread-safety-analysis-in-c-and-c/
52. https://isocpp.org/wiki/faq/cpp11-library-concurrency
53. https://research.google.com/pubs/archive/42958.pdf
54. https://www.geeksforgeeks.org/python/pep-8-coding-style-guide-python/
55. https://www.educative.io/blog/python-pep8-tutorial
56. https://algomaster.io/learn/python/pep8-style-guide
57. https://stackoverflow.com/questions/65768215/type-annotations-with-google-style-docstrings
58. https://www.sei.cmu.edu/blog/cert-c-secure-coding-guidelines/
59. https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines
60. https://opentitan.org/book/doc/contributing/style_guides/c_cpp_coding_style.html
61. https://en.cppreference.com/w/cpp/language/raii.html
62. https://www.linkedin.com/pulse/c-core-guidelines-rules-resource-management-rainer-grimm
63. https://www.thecodedmessage.com/posts/raii/
64. https://stackoverflow.com/questions/78372004/undefined-behavior-with-pointer-casts-in-c99-and-misra-c
2012
65. https://cwe.mitre.org/data/definitions/680.html
66. https://arxiv.org/pdf/2507.01195.pdf
67. https://quantumzeitgeist.com/qiskit-in-practice/
68. https://github.com/full-stack-deep-learning/course-gitbook/blob/master/course-content/testing-and-deploy
ment/ml-test-score.md69. https://developers.google.com/machine-learning/guides/rules-of-ml
70. https://docs.cloud.google.com/architecture/ml-on-gcp-best-practices
71. https://www.tuhh.de/ti3/paper/rump/Ru05d.pdf
72. https://ogilab.w.waseda.jp/ogita/math/doc/2008_RuOgOi_01.pdf
73. https://nhigham.com/wp-content/uploads/2021/04/high21m.pdf
74. https://epubs.siam.org/doi/10.1137/0914050
75. https://quantum.cloud.ibm.com/docs/tutorials/hello-world
76. https://aimath.org/WWN/balancelaws/balancelaws.pdf
77. https://arxiv.org/html/2506.03636v1
78. https://arxiv.org/abs/1408.6817
79. https://link.aps.org/doi/10.1103/PhysRevB.77.165123
80. https://ntrs.nasa.gov/api/citations/19820016143/downloads/19820016143.pdf
81. https://ldra.com/ldra-blog/do-178c-structural-coverage-analysis/
82. https://www.datadoghq.com/blog/ml-model-monitoring-in-production-best-practices/
83. https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-r
eduction/
84. https://www.solarwinds.com/sre-best-practices/golden-signals
85. https://www.motadata.com/blog/golden-signals-monitoring-sre-metrics/
86. https://firehydrant.com/blog/4-sre-golden-signals-what-they-are-and-why-they-matter/
87. https://sre.google/sre-book/monitoring-distributed-systems/
88. https://www.dnsstuff.com/opentelemetry-overview-traces-metrics-logs
89. https://www.dash0.com/knowledge/logs-metrics-and-traces-observability
90. https://opentelemetry.io/docs/concepts/signals/logs/
91. https://opentelemetry.io/docs/specs/otel/metrics/
92. https://www.catapult.com/blog/f1-data-analysis-transforming-performance
93. https://community.cadence.com/cadence_blogs_8/b/life-at-cadence/posts/formula-1-how-f1-teams-use-tele
metry-control-analytics-to-go-faster
94. https://www.pass4sure.com/blog/how-data-analytics-is-driving-the-future-of-formula-one-racing/
95. https://www.scichart.com/blog/realtime-telemetry-datavisualisation-formulaone-motorsport/
96. https://www.kusari.dev/learning-center/slsa-supply-chain-levels-for-software-artifacts
97. https://www.infoq.com/news/2025/08/provenance/
98. https://github.blog/security/supply-chain-security/slsa-3-compliance-with-github-actions/
99. https://edu.chainguard.dev/compliance/slsa/what-is-slsa/
100. https://devguide.owasp.org/en/06-verification/01-guides/03-asvs/
101. https://abseil.io/resources/swe-book/html/ch09.html
102. https://google.github.io/eng-practices/review/103. https://github.com/google/eng-practices
104. https://bssw.io/items/google-guidance-on-code-review
105. https://slab.com/library/templates/google-code-review/
106. https://www.securecoding.com/blog/integer-overflow-attack-and-prevention/
107. https://users.cs.utah.edu/~regehr/papers/overflow12.pdf
108. https://www.blackduck.com/static-analysis-tools-sast/sei-cert.html
109. https://cloud.google.com/blog/products/devops-sre/applying-sre-principles-to-your-mlops-pipelines
110. https://pubsonline.informs.org/doi/10.1287/ijoc.2024.0551
111. http://www.ijarse.com/images/fullpdf/1489828422_D817ijarse.pdf
112. https://swehb.nasa.gov/plugins/viewsource/viewpagesrc.action?pageId=135331967
113. https://s3vi.ndc.nasa.gov/ssri-kb/static/resources/nasa-std-8739.8a.pdf
114. https://innovationspace.ansys.com/knowledge/forums/topic/an-introduction-to-do-178c/
115. https://ldra.com/npr7150-2d/
116. https://sparta.aerospace.org/countermeasures/nasabpg/MI-SOFT-01
117. https://standards.nasa.gov/standard/NASA/NASA-STD-87398
118. https://nodis3.gsfc.nasa.gov/displayAll.cfm?Internal_ID=N_PR_7150_002D_&page_name=ALL
119. https://real-time-consulting.com/case-study/3956/
120. https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3100r5.pdf
121. https://www.perforce.com/blog/qac/what-is-iso-26262
122. https://www.sei.cmu.edu/library/cert-c-secure-coding-standard/
123. https://www.dnv.us/services/functional-safety-for-automotive-iso-26262-86905/
124. https://www.cortex.io/post/making-sre-metrics-work-for-your-team
125. https://www.theoris.com/from-chaos-to-control-cmmi-and-the-power-of-process-improvement/
126. https://www.intertek.com/electrical/standards/iec-61508/
127. https://www.trigyn.com/insights/what-cmmi-dev-level-5
128. https://www.jamasoftware.com/requirements-management-guide/industrial-manufacturing-development/fu
nctional-safety-made-simple-a-guide-to-iec-61508-for-manufacturing/
129. https://ca.indeed.com/career-advice/career-development/cmmi-maturity-levels
130. https://www.geeksforgeeks.org/software-engineering/software-engineering-cleanroom-testing/
131. https://www.nxp.com/products/nxp-product-information/quality/product-qualification:QUALITY__QUALIF
132. https://www.lumissil.com/assets/pdf/support/reliability assurance/Qualification Test Method and Acceptance
Criteria.pdf
133. https://www.pentest-limited.com/services/penetration-testing-services/owasp-asvs/
134. https://ptacts.uspto.gov/ptacts/public-informations/petitions/1541825/download-documents?artifactId=Y-Px
PacZ40fhnMDD7m7afzlyQPrcLieH3UYDZu7sBx5OJfXY-c5NcQM135. https://e.chipanalog.com/Public/Uploads/uploadfile/files/20240621/ReliabilityTestReportCAIS206XCAIS209
XV1.1.pdf
136. https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html
137. https://docs.idmod.org/projects/doc-guidance/en/latest/docstrings.html
138. https://cs.stanford.edu/people/nick/py/python-style-basics.html
139. https://joshdimella.com/blog/python-docstring-formats-best-practices
140. https://www.reddit.com/r/rust/comments/y1dwg6/blog_post_my_perspective_on_raii_and_memory/
141. https://guiquanz.gitbooks.io/google-cc-style-guide/ebook/Naming.html
142. https://primo.ai/index.php/ML_Test_Score
143. https://www.giskard.ai/knowledge/how-did-the-idea-of-giskard-ai-emerge-1-n
144. https://drake.mit.edu/styleguide/cppguide.html
145. https://dl.acm.org/doi/abs/10.1137/0914050
146. https://oneuptime.com/blog/post/2026-02-06-opentelemetry-traces-vs-metrics-vs-logs/view
147. https://www.youtube.com/watch?v=MvX5OUK-tbE
148. https://slsa.dev/spec/v1.1/faq
149. https://www.bleepingcomputer.com/news/security/mitre-shares-2025s-top-25-most-dangerous-software-
weaknesses/
150. https://industrialcyber.co/nist/nist-enhances-sp-800-53-controls-to-improve-cybersecurity-and-software-
maintenance-reduce-cyber-risks/
151. https://www.parasoft.com/blog/theres-no-good-reason-to-ignore-cert-c/
152. https://docs.legato.io/18_05/ccodingStdsSecurity.html
153. https://www.syteca.com/en/solutions/meeting-compliance-requirements/nist-compliance
154. https://www.reddit.com/r/cybersecurity/comments/1pkx1up/mitre_shares_2025s_top_25_most_dangerous_
software/
155. https://www.saltycloud.com/isora-grc/solutions/nist-800-53-compliance-software/
156. https://standards.ieee.org/ieee/730/5284/
157. https://en.wikipedia.org/wiki/ISO/IEC_12207
158. https://dl.acm.org/doi/10.5555/3408352.3408539
159. https://www.cs.utep.edu/isalamah/courses/5387/IEEE-Std-730-2002.pdf
160. http://retis.sssup.it/~a.biondi/papers/DATE25.pdf
161. https://asq.org/quality-resources/articles/an-introduction-to-the-new-ieee-730-standard-on-software-qualit
y-assurance?id=b4d1d01a493649afb6cc68732d68b206
162. https://blog.pacificcert.com/iso-iec-ieee-12207-standardizing-software-lifecycle-processes/
163. https://arxiv.org/pdf/1912.01367.pdf
164. http://sqgne.org/presentations/2011-12/Rakitin-Feb-2012.pdf
165. https://standards.ieee.org/ieee/12207/5672/
166. https://www.autosar.org/fileadmin/standards/R22-11/AP/AUTOSAR_EXP_SWArchitecture.pdf167. https://www.yegor256.com/pdf/ieee-730-2014.pdf
168. https://standards.ieee.org/ieee/12207-2/10353/
169. https://www.autosar.org/fileadmin/standards/R21-11/AP/AUTOSAR_EXP_SWArchitecture.pdf
170. https://www.cutter.com/article/managing-technical-debt-sqale-method-490726
171. https://dl.acm.org/doi/pdf/10.5555/2666036.2666042
172. https://www.agilealliance.org/wp-content/uploads/2016/01/SQALE-Meaningful-Insights-into-your-Technical-
Debt.pdf
173. https://diataxis.fr/start-here/
174. https://securityreviewer.atlassian.net/wiki/spaces/KC/pages/426091/SQALE
175. https://github.com/evildmp/diataxis-documentation-framework/blob/main/reference-explanation.rst
176. https://www.sonarsource.com/blog/sqale-the-ultimate-quality-model-to-assess-technical-debt/
177. https://www.reddit.com/r/F1Technical/comments/16ti6jg/what_types_of_telemetry_do_f1_teams_have_and_u
se/
178. https://stackoverflow.com/questions/4393197/erlangs-let-it-crash-philosophy-applicable-elsewhere
179. https://stanford-cs242.github.io/f18/lectures/06-2-concurrency.html
180. https://discuss.pennylane.ai/t/circuit-executing-with-qiskit-aer-noise-model-very-slow/2946