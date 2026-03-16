# Anvil Connectivity, Ownership, and Federated Worktree Roadmap

Status: proposed
Authoring mode: inventive research
Date: 2026-03-11
Repository: `Anvil`
Primary focus: interconnectivity, file ownership awareness, session and campaign awareness, and company-server federation across parallel Anvil instances working on the same project
Roadmap posture: mechanism-first, repo-grounded, validator-friendly
Target line of effort split: approximately 45% practical / 55% moonshot

---

## 1. First-Principles Framing

### 1.1 What the system fundamentally is

[Repo-grounded observation]

Anvil is not merely a chat interface with tools.

Anvil is already a partially-formed code operating system with:

- an interactive front door in `cli/repl.py`
- a large execution and governance core in `core/unified_chat_loop.py`
- campaign orchestration in `core/campaign/control_plane.py`
- isolated lane workspaces in `core/campaign/worktree_manager.py`
- a local message fabric in `core/subagent_communication.py`
- local collaboration signals in `core/collaboration/task_announcer.py` and `core/collaboration/negotiation.py`
- local and peer identity plumbing in `core/networking/instance_identity.py`, `core/networking/peer_discovery.py`, and `core/networking/peer_transport.py`
- file-level ownership semantics in `core/ownership/file_ownership.py`
- event and ledger infrastructure in `shared_kernel/event_store.py` and `saguaro/state/ledger.py`
- memory substrate in `core/memory/project_memory.py` and `core/memory/fabric/*`
- verification and roadmap extraction in `saguaro/roadmap/validator.py`

That means the system already behaves more like a control plane than a single agent.

### 1.2 What the system appears to be trying to become

[Inference from repo structure]

Anvil appears to be trying to become a federated engineering runtime where:

- specialized loops cooperate across phases instead of only across prompts
- Saguaro provides not just search but operational state, traceability, and verification
- campaign work products become governed artifacts instead of ad hoc chat residue
- file ownership, workset scope, and verification evidence shape what agents are allowed to do
- the agent can scale from solo use to coordinated engineering programs

The missing center of gravity is not “more agents.”

The missing center of gravity is a shared causal model of repo intent, repo state, repo ownership, and repo promotion.

### 1.3 What this system should become if engineered to elite standards

[Synthesis]

If built to NASA mission-control or Formula 1 pit-wall standards, Anvil plus Saguaro should operate as:

- a distributed mission control for code
- a causal ledger for who planned, changed, verified, promoted, and invalidated what
- a real-time workload allocator aware of file claims, symbol interference, validation state, and machine capability
- a semantic merge and arbitration fabric that reasons about intent and program behavior, not just text
- a company-wide repo coordination layer where local instances, detached campaign daemons, and server-side architect services share one operational truth

### 1.4 The core constraints that actually matter

[Synthesis]

The real constraints are:

- correctness beats raw parallelism because code merges can silently corrupt behavior
- intent preservation matters as much as textual convergence because two clean textual changes can still semantically interfere
- latency matters on the inner loop, so federation must not stall local editing or inference
- bandwidth matters because synchronizing full chat context between many agents is impossible at scale
- trust boundaries matter because local peers, company servers, and open-source contributors cannot share the same write powers
- crash recovery matters because leases, locks, and partially-promoted deltas will fail in the real world
- human legibility matters because the system must explain why an ownership denial, task reassignment, or merge arbitration occurred
- verification must be promotion-gating, not advisory

### 1.5 The design thesis for this roadmap

[Synthesis]

The right move is not to bolt a Slack-like collaboration layer onto Anvil.

The right move is to promote the repo itself into a replicated control surface with:

- identity
- causality
- leases
- intent packets
- semantic conflict prediction
- validation-gated delta distribution
- capability-aware scheduling
- explicit trust zones

That lets hundreds of agents cooperate without pretending that eventually-consistent text alone is good enough.

---

## 2. External Research Scan

### 2.1 What was researched

[External research]

Research was widened across:

- papers and arXiv work on semantic merge and semantic conflict detection
- local-first and CRDT systems
- practitioner docs on collaborative sync engines
- version-control systems with virtual branches and workspace overlays
- ownership and review governance systems
- distributed systems techniques for ordering and leader selection
- remote artifact sharing and cache systems
- practitioner discussions on the limits of CODEOWNERS and raw CRDT convergence

### 2.2 High-signal patterns and why they matter here

#### R1. Local-first software as a control-plane stance

Source:

- Ink & Switch, “Local-first software”  
  https://www.inkandswitch.com/local-first/

Why it matters:

- The local-first position is not just about offline UX.
- It says the local copy is primary and sync is a first-class protocol.
- For Anvil, every local instance should continue operating even when the company server is slow or absent.
- That strongly supports a design where state is authored locally, durably logged locally, and later reconciled with higher tiers.

Key transfer:

- local-first for plans, claims, and evidence
- server-assisted, not server-dependent

#### R2. Automerge shows how to replicate intent-bearing state, not just files

Source:

- Automerge docs, “Welcome to Automerge”  
  https://automerge.org/docs/hello/

Observed pattern:

- multiple devices can update local state independently, even offline
- sync later converges
- change history remains inspectable
- the library is network-agnostic

Why it matters:

- Roadmap packets, ownership offers, task assignments, and session summaries are much better CRDT targets than source code files themselves
- This strongly suggests a dual strategy:
- CRDT for intent-bearing metadata
- stricter guarded merge flow for source code

#### R3. Yjs shows that presence and awareness are separate from document convergence

Source:

- Yjs docs, “Introduction”  
  https://docs.yjs.dev/

Observed pattern:

- collaborative apps need both shared state and awareness/presence
- the network can be provider-agnostic
- order of update delivery need not matter for convergence

Why it matters:

- Anvil currently has almost no true presence model beyond peer discovery and task announcements
- Presence should include:
- who is alive
- what phase they are in
- what they claim
- what they are editing
- what verification state they hold
- what they are waiting on

#### R4. Figma chose a custom multiplayer model because generic OT was too complex for its problem space

Source:

- Figma engineering, “How Figma’s multiplayer technology works”  
  https://www.figma.com/blog/how-figmas-multiplayer-technology-works/

Observed pattern:

- Figma explicitly rejected stock OT for its own custom model because it was overly complex for the actual object graph they needed
- live collaboration removed export/sync/email loops

Why it matters:

- Anvil should not naively “use CRDT everywhere”
- code collaboration is not a plain-text shared editor problem
- Anvil needs a custom replicated object model for:
- claims
- plans
- change capsules
- validation state
- promotion gates

This is a major architectural permission: a domain-specific sync model is the right answer.

#### R5. Linear’s sync engine validates that speed and coherence require a dedicated sync subsystem

Source:

- Linear engineering, “Scaling the Linear Sync Engine”  
  https://linear.app/now/scaling-the-linear-sync-engine

Why it matters:

- Linear’s example shows that sync quality becomes a product moat
- For Anvil, coordination quality will determine whether multi-instance development feels magical or disastrous
- Sync cannot be a side-effect of chat or git polling
- It needs to be a named subsystem with budgets, telemetry, recovery paths, and protocol contracts

#### R6. GitHub CODEOWNERS is useful but branch-scoped and structurally limited

Sources:

- GitHub Docs, “About code owners”  
  https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners
- Reddit discussion on distributed OWNERS files  
  https://www.reddit.com/r/github/comments/16gc4o8

Observed pattern:

- GitHub uses one effective `CODEOWNERS` file per branch
- ownership lives on the base branch of a PR
- practitioners complain that a single CODEOWNERS file becomes limiting and bottleneck-prone for monorepos

Why it matters:

- Anvil should not model ownership as a static CODEOWNERS replacement
- It needs layered ownership:
- static policy ownership
- campaign ownership
- lease ownership
- symbol ownership
- validation ownership

Static ownership is governance.

Dynamic ownership is concurrency control.

Anvil needs both.

#### R7. GitButler proves that parallel branch lanes need an explicit workspace union model

Sources:

- GitButler docs, “Parallel Branches”  
  https://docs.gitbutler.com/features/branch-management/virtual-branches
- GitButler docs, “Workspace Branch”  
  https://docs.gitbutler.com/workspace-branch

Observed pattern:

- multiple branches can be applied in one working directory
- the tool maintains a synthetic workspace branch that represents the merged union
- stock Git needs protection hooks because it does not understand this mode

Why it matters:

- The user request for Saguaro “to actually have a worktree like GitHub does that traces and tracks changes” maps cleanly to this idea
- Anvil should maintain an explicit workspace-union artifact for multi-agent work
- `core/campaign/worktree_manager.py` already has lane overlays
- the missing pieces are:
- a causal workspace manifest
- per-lane ownership linkage
- promotion guards
- traceable union views

#### R8. SemanticMerge shows why file-level merge is too weak for refactoring-heavy collaboration

Source:

- SemanticMerge intro guide  
  https://www.semanticmerge.com/documentation/intro-guide/semanticmerge-intro-guide

Observed pattern:

- structure-aware merge works on code trees, not only line positions
- moved methods and refactors can be tracked independently of their textual location
- method-by-method merge beats whole-file conflict handling in many cases

Why it matters:

- Anvil’s current file ownership is valuable but too coarse
- symbol-aware claims and AST-aware arbitration would dramatically increase safe concurrency
- this aligns with Saguaro’s semantic ambitions better than text-only locks

#### R9. Bazel remote caching shows how validated build/test outputs can be shared aggressively

Source:

- Bazel docs, “Remote Caching”  
  https://bazel.build/remote/caching

Observed pattern:

- a remote cache reuses outputs from another user’s build
- the cache is content-addressable
- teams can read-only or read-write gate cache access
- concurrent source mutation during build is a known failure mode

Why it matters:

- once one Anvil instance has validated a patch set, other instances should not re-run identical work blindly
- Anvil needs a validation artifact cache and promotion capsule store
- this is especially important for company-server and large-fleet scenarios

#### R10. Hybrid Logical Clocks are the right mental model for ordering without atomic clocks

Source:

- CockroachDB glossary, “Hybrid Logical Clock (HLC) Timestamps”  
  https://www.cockroachlabs.com/glossary/distributed-db/hybrid-logical-clock-hlc-timestamps/

Observed pattern:

- distributed systems need ordered events
- HLC combines physical and logical components
- it is a practical alternative when Spanner-style atomic clocks are unavailable

Why it matters:

- Anvil does not need true global time
- It needs stable causal ordering for:
- claim grants
- plan promotions
- delta relays
- architect decisions
- verification gates

`saguaro/state/ledger.py` already has clocks and watermarks.

That should be extended into a proper hybrid causal watermark model.

#### R11. Verified three-way merge and SafeMerge show semantic conflict-freedom is a real, formal target

Sources:

- arXiv, “Verifying Semantic Conflict-Freedom in Three-Way Program Merges”  
  https://arxiv.org/abs/1802.06551
- Microsoft Research summary, “Verified Three-Way Program Merge”  
  https://www.microsoft.com/en-us/research/publication/verified-three-way-program-merge/

Observed pattern:

- textual merge tools can create behavior bugs even when text conflicts are resolved
- semantic conflict-freedom can be defined and checked compositionally

Why it matters:

- Architect arbitration should not stop at “these files overlap”
- it should aim to estimate whether the concurrent changes are semantically interfering
- Saguaro is unusually well positioned for this because it already has code graph, impact, and semantic verification surfaces

#### R12. Newer semantic-conflict research reinforces that static analysis can detect hidden interference

Source:

- arXiv, “Detecting Semantic Conflicts using Static Analysis”  
  https://arxiv.org/abs/2310.04269

Why it matters:

- this supports building a Saguaro semantic conflict radar rather than relying on merge failures
- detection should happen before promotion, not after the code lands

#### R13. MergeBERT and neural merge work are provocative but should remain advisory

Source:

- arXiv, “Program Merge Conflict Resolution via Neural Transformers”  
  https://arxiv.org/abs/2109.00084

Why it matters:

- neural merge can propose resolutions
- it should not become the source of truth for repo promotion
- in this roadmap, neural arbitration is an assistant to the Architect, not the commit authority

#### R14. etcd-style elections remain the cleanest practical model for an Architect leader

Source:

- etcd concurrency package docs  
  https://pkg.go.dev/go.etcd.io/etcd/client/v3/concurrency

Observed pattern:

- one session leads at a time
- the leader can proclaim new values
- observers can watch ordered leadership changes
- leases and TTL matter

Why it matters:

- a company-server Architect or repo-local lead should be elected, not assumed
- leadership should be revocable, observable, and recoverable

### 2.3 Cross-industry analogies worth stealing

[External inspiration plus synthesis]

#### Formula 1 pit wall

- the driver does not hold the global race picture
- the pit wall fuses telemetry, tire state, weather, and rivals
- strategy changes are issued with low latency and high confidence

Transfer to Anvil:

- local instances keep executing
- the Architect sees fleet-wide telemetry and reassigns work based on changing repo state

#### NASA mission control

- every subsystem has an owner
- anomalies get traced through a causal chain
- no promotion to a risky state occurs without explicit evidence

Transfer to Anvil:

- every claimed file, symbol, and roadmap lane should have ownership, evidence, and rollback criteria

#### Air traffic control

- traffic is deconflicted through reserved corridors, timing, and authority tiers
- not every plane negotiates directly with every other plane

Transfer to Anvil:

- hundreds of agents cannot pairwise negotiate safely
- a control plane and trust-zone hierarchy are mandatory

### 2.4 Research conclusions that materially change the design space

[Synthesis]

The research suggests:

- use CRDTs for metadata and plan state, not raw source as the primary integration primitive
- use causal watermarks and leases for authority and freshness
- use structure-aware merge and semantic conflict prediction for code
- use remote cache and content-addressable artifacts for validated outputs
- separate presence, intent, ownership, and source promotion into distinct but linked layers

---

## 3. Repo Grounding Summary

### 3.1 Commands and tools used

[Repo-grounded observation]

Primary repo-grounding was run from the repo virtual environment with:

- `source venv/bin/activate`
- `./venv/bin/saguaro entrypoints`
- `./venv/bin/saguaro build-graph`
- `./venv/bin/saguaro query "..." --k ...`
- `./venv/bin/saguaro agent skeleton ...`
- `./venv/bin/saguaro agent slice ... --depth 2`
- `./venv/bin/saguaro impact --path ...`

Observed caveat:

- `./venv/bin/saguaro health` repeatedly stalled after native/TensorFlow startup logs during this session
- several broad semantic queries also stalled
- targeted queries, skeletons, slices, entrypoints, build graph, and impacts were successful
- fallback file inspection was used only after documenting the degraded Saguaro behavior

### 3.2 Entry points and user-facing loops inspected

[Repo-grounded observation]

From `./venv/bin/saguaro entrypoints`:

- `cli/repl.py`
- `main.py`
- `anvil.py`
- `saguaro/cli.py`
- `saguaro/mcp/server.py`
- `saguaro/dni/server.py`

Why this matters:

- there is already more than one runtime surface
- the system is capable of headless and service-like expansion

### 3.3 Core user loop and prompt assembly surfaces

[Repo-grounded observation]

`cli/repl.py` shows that the REPL already wires together:

- `core.networking.instance_identity.InstanceRegistry`
- `core.networking.peer_discovery.PeerDiscovery`
- `core.networking.peer_transport.PeerTransport`
- `core.collaboration.task_announcer.TaskAnnouncer`
- `core.collaboration.context_sharing.ContextShareProtocol`
- `core.collaboration.negotiation.CollaborationNegotiator`
- `core.ownership.file_ownership.FileOwnershipRegistry`
- `core.memory.project_memory.ProjectMemory`
- `core.campaign.runner.CampaignRunner`
- `shared_kernel.event_store.get_event_store`

This is critical.

It means the desired connectivity story is not alien to the repo.

It is underintegrated, not absent.

`core/prompts/system_prompt_builder.py` contains `SystemPromptBuilder.build`, which is a clear insertion point for:

- ownership posture
- live peer count
- claim conflicts
- current campaign and workspace state
- trust-zone restrictions

### 3.4 Local transport and coordination surfaces

[Repo-grounded observation]

`core/subagent_communication.py` already contains:

- `OWNERSHIP_TOPICS`
- `MessageBus`
- `CoordinationProtocol.handoff`
- `CoordinationProtocol.barrier_sync`
- `CoordinationProtocol.request_response`
- trace segment capture and export

Current limitation:

- this bus is local-process and memory-resident
- the API shape is good
- the transport durability and federation are not

### 3.5 Existing collaboration mechanisms

[Repo-grounded observation]

`core/collaboration/task_announcer.py` currently:

- announces tasks via transport
- reduces tasks to `instruction` plus `context_files`
- detects overlap via token similarity and shared-file count

`core/collaboration/negotiation.py` currently:

- creates proposals
- emits accepted or counter-proposal events
- merges plans mainly by deduping tasks and tagging shared files

Interpretation:

- collaboration exists, but it is pre-semantic
- it does not yet understand campaign phase, proof obligations, symbol overlap, or machine capability

### 3.6 Ownership surfaces already present

[Repo-grounded observation]

`core/ownership/file_ownership.py` is much more mature than the current roadmap file implied.

`FileOwnershipRegistry.claim_files` already supports:

- `instance_id`
- `ownership_crdt`
- `sync_protocol`
- `repo_policy_resolver`
- `campaign_id`
- `phase_id`
- `task_id`
- `access_mode`
- TTL-based leases
- heartbeats
- conflict publication
- event emission

Important implication:

- the repo already knows that ownership is multi-instance, policy-gated, and synchronizable
- the missing work is to make the CRDT, sync protocol, and policy layer authoritative across actual peers

### 3.7 Networking surfaces already present

[Repo-grounded observation]

`core/networking/instance_identity.py` already models:

- an `AnvilInstance`
- project hash
- instance heartbeat
- listen address

`core/networking/peer_discovery.py` already contains:

- `MDNSDiscovery`
- `RendezvousDiscovery`
- `FileSystemDiscovery`
- `PeerDiscovery`

`core/networking/peer_transport.py` already contains:

- `PeerMessage`
- `PeerConnection`
- a transport registry
- point-to-point send and broadcast callbacks

Interpretation:

- the repo has discovery and transport placeholders
- they are not yet elevated into a company-safe, durable, validated federation layer

### 3.8 Campaign, worktree, and lane surfaces

[Repo-grounded observation]

`core/campaign/worktree_manager.py` already implements:

- isolated lane preparation
- overlay workspace seeding
- per-lane baseline hashes
- changed file detection
- direct promotion from lane workspace to repo root

`core/campaign/control_plane.py` already pulls together:

- specialist registry
- loop scheduler
- telemetry
- roadmap compiler
- verification lane
- memory fabric
- research runtime
- state ledger
- event store

This is the strongest architectural clue in the entire repo.

Anvil already has a proto-control-plane.

The roadmap should extend it outward, not invent a separate parallel architecture.

### 3.9 State and event durability surfaces

[Repo-grounded observation]

`shared_kernel/event_store.py` already provides:

- durable SQLite event logging
- run export
- replay tape export
- checkpoints
- mission capsules
- safety cases

`saguaro/state/ledger.py` already provides:

- workspaces
- snapshots
- delta watermarks
- changesets
- peer add/remove
- `sync_push`
- `sync_pull`
- `sync_subscribe`
- repo file records
- filesystem comparison

Interpretation:

- the real missing primitive is not “some way to trace changes”
- that primitive already exists in an initial form
- the missing step is to make the ledger the authoritative heartbeat of federated worktree state

### 3.10 Memory surfaces that can be repurposed for campaign awareness

[Repo-grounded observation]

`core/memory/project_memory.py` is small and coarse.

By contrast, `core/memory/fabric/store.py`, `core/memory/fabric/retrieval_planner.py`, and `core/memory/fabric/projectors.py` already provide:

- stored memory objects
- aliases and graph edges
- embeddings
- multivectors
- latent packages
- read and feedback records
- retrieval planning
- projector-derived embeddings and bundles

Interpretation:

- campaign memory awareness should be built on the fabric, not the simple project memory file
- instances should remember not only facts but coordination history and arbitration outcomes

### 3.11 Telemetry and governance surfaces

[Repo-grounded observation]

`core/campaign/telemetry.py` currently gives:

- lightweight spans
- summarization

`core/unified_chat_loop.py` already integrates:

- governance checkpoints
- evidence phases
- verification
- telemetry recorders
- runtime control policy
- context compression and dynamic tooling

Interpretation:

- telemetry exists but is too thin for fleet orchestration
- governance exists and should become the hard gate for promotion across instances

### 3.12 Native/Python boundary

[Repo-grounded observation]

Targeted query results surfaced `core/native/native_qsg_engine.py` and related native QSG code.

This matters because:

- the native engine already produces dense runtime telemetry and capabilities
- that telemetry should feed scheduler decisions for multi-instance balancing
- heavy semantic analysis can be routed to the strongest node instead of assumed local

### 3.13 Where the architecture is strongest

[Repo-grounded observation]

Strong areas:

- campaign control plane
- state ledger and event store
- worktree isolation
- ownership data model
- REPL wiring to collaboration/networking primitives
- roadmap validation and traceability

### 3.14 Where the architecture is underexploited or thin

[Repo-grounded observation]

Thin areas:

- true network transport durability
- cross-instance causal ordering
- semantic rather than lexical overlap detection
- source promotion governance across peers
- presence, awareness, and trust-zone modeling
- symbol-level ownership
- validated artifact distribution
- split-brain recovery

### 3.15 Grounding conclusion

[Synthesis]

The repo does not need a brand-new collaboration story.

It needs:

- a stricter shared state model
- better federation of existing primitives
- semantic arbitration instead of lexical negotiation
- explicit authority and promotion paths

---

## 4. Hidden Assumptions Limiting the Design Space

[Synthesis]

1. An Anvil instance is the unit of work.

Reality:

- the real unit of work is a claimable, verifiable change capsule over files, symbols, tests, and evidence.

2. File ownership is enough.

Reality:

- file ownership is the safety floor, not the concurrency ceiling.

3. Sync means copying code.

Reality:

- sync should usually mean relaying intent, state deltas, proof packets, and validated artifacts first.

4. If Git can merge it, the system is safe.

Reality:

- semantic interference can remain after a clean textual merge.

5. Roadmaps are documents.

Reality:

- roadmaps should be executable coordination objects with phase packets, gates, and ownership bindings.

6. One master agent can coordinate any fleet size.

Reality:

- large fleets need hierarchical or elected control, not a single immortal master.

7. The company server should own all truth.

Reality:

- local instances must remain authoritative for local progress until promotion.

8. CRDT convergence equals user-intent convergence.

Reality:

- CRDTs provide deterministic convergence, not correct human-intent resolution.

9. Ownership is governance metadata.

Reality:

- ownership is also a live concurrency control mechanism.

10. Promotion should occur at file granularity.

Reality:

- promotion should be capsule- and verification-granular.

---

## 5. Candidate Implementation Phases

### Candidate 01. Repo Presence Mesh

- Name: Repo Presence Mesh
- Suggested `phase_id`: `intake`
- Core insight: turn instance heartbeat from an incidental side effect into a first-class repo-presence plane.
- External inspiration or analogy: Yjs awareness, local-first presence, air-traffic radar.
- Why it fits Saguaro and Anvil specifically: `InstanceRegistry` and `PeerDiscovery` already exist, but they do not project campaign, lane, or claim state.
- Exact places in this codebase where it could wire in: `core/networking/instance_identity.py`, `core/networking/peer_discovery.py`, `core/networking/peer_transport.py`, `cli/repl.py`, `cli/commands/agent.py`.
- Existing primitives it can reuse: project hash, heartbeat, listen address, peer browse, REPL command surfaces.
- New primitive, data flow, or subsystem needed: `core/connectivity/repo_presence.py` with heartbeat envelopes, capability descriptors, phase state, and trust-zone tags.
- `repo_scope`: `core/networking/*`, `core/connectivity/*`, `cli/repl.py`, `cli/commands/agent.py`.
- `owning_specialist_type`: `distributed_runtime_engineer`
- `allowed_writes`: `core/networking/*.py`, `core/connectivity/*.py`, `cli/repl.py`, `tests/test_repo_presence.py`.
- `telemetry_contract`: emit `presence.heartbeat.sent`, `presence.heartbeat.missed`, `presence.peer.joined`, and `presence.peer.stale` with repo hash, instance id, and campaign id.
- `required_evidence`: two local instances and one detached campaign daemon appear in the same repo-presence table with live phase and capability metadata.
- `rollback_criteria`: false peer resurrection or stale peers remain visible longer than configured TTL.
- `promotion_gate`: `pytest tests/test_repo_presence.py tests/test_collaboration_commands.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- `success_criteria`: every connected instance can answer who is present, what they are doing, and whether they are promotable.
- Why this creates value: it eliminates blind parallelism.
- Why this creates moat: most agent systems stop at chat visibility, not repo-native presence.
- Main risk or failure mode: noisy peer flapping on unstable networks.
- Smallest credible first experiment: extend `PeersCommand` to show phase, claims, and lane count from a local LAN-only heartbeat bus.
- Confidence level: 0.87

### Candidate 02. Causal Repo State Ledger

- Name: Causal Repo State Ledger
- Suggested `phase_id`: `research`
- Core insight: use the existing Saguaro state ledger as the authority for workspace snapshots, claim watermarks, and promotion order.
- External inspiration or analogy: hybrid logical clocks, append-only event logs, mission timeline control.
- Why it fits Saguaro and Anvil specifically: `saguaro/state/ledger.py` already tracks workspaces, deltas, peers, and sync methods.
- Exact places in this codebase where it could wire in: `saguaro/state/ledger.py`, `shared_kernel/event_store.py`, `core/campaign/control_plane.py`, `core/campaign/worktree_manager.py`.
- Existing primitives it can reuse: workspaces, snapshots, delta watermarks, sync push/pull, event store checkpoints.
- New primitive, data flow, or subsystem needed: HLC-style causal watermark and repo-state epochs for claims, deltas, and promotions.
- `repo_scope`: `saguaro/state/*.py`, `shared_kernel/event_store.py`, `core/campaign/control_plane.py`, `tests/test_state_ledger.py`.
- `owning_specialist_type`: `state_systems_engineer`
- `allowed_writes`: `saguaro/state/*.py`, `shared_kernel/*.py`, `tests/test_state_ledger.py`, `tests/test_ledger_federation.py`.
- `telemetry_contract`: emit `ledger.watermark.advanced`, `ledger.peer.reconciled`, `ledger.divergence.detected`, and `ledger.replay.rebuilt`.
- `required_evidence`: two instances independently record deltas and reconcile them into a shared ordered ledger without losing claim or workspace lineage.
- `rollback_criteria`: any replay can produce a different workspace ordering from the original causal stream.
- `promotion_gate`: `pytest tests/test_state_ledger.py tests/test_ledger_federation.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- `success_criteria`: every promotion and denial can be causally explained from ledger state alone.
- Why this creates value: it turns sync from guesswork into evidence-backed ordering.
- Why this creates moat: a causal repo ledger is far deeper than generic multi-agent messaging.
- Main risk or failure mode: clock drift or malformed reconciliation logic creating duplicate or missing events.
- Smallest credible first experiment: attach a watermark and peer-origin envelope to existing `sync_push` and `sync_pull`.
- Confidence level: 0.91

### Candidate 03. Fencing-Token File Ownership

- Name: Fencing-Token File Ownership
- Suggested `phase_id`: `feature_map`
- Core insight: file claims need fencing tokens and lease epochs so stale leaders cannot keep writing after failover.
- External inspiration or analogy: Kleppmann-style fencing for distributed locks, etcd leases.
- Why it fits Saguaro and Anvil specifically: `FileOwnershipRegistry` already has CRDT and sync hooks, TTLs, and heartbeats.
- Exact places in this codebase where it could wire in: `core/ownership/file_ownership.py`, `core/ownership/ownership_models.py`, `core/subagent_communication.py`, `saguaro/state/ledger.py`.
- Existing primitives it can reuse: TTL leases, heartbeats, denied file records, sync protocol callbacks, ownership topics.
- New primitive, data flow, or subsystem needed: monotonic fencing tokens bound to ledger epochs and claim grants.
- `repo_scope`: `core/ownership/*`, `core/subagent_communication.py`, `saguaro/state/ledger.py`, `tests/test_file_ownership.py`, `tests/test_ownership_crdt.py`.
- `owning_specialist_type`: `concurrency_engineer`
- `allowed_writes`: `core/ownership/*.py`, `saguaro/state/ledger.py`, `tests/test_file_ownership.py`, `tests/test_ownership_crdt.py`, `tests/test_ownership_fencing.py`.
- `telemetry_contract`: emit `ownership.claim.granted`, `ownership.claim.denied`, `ownership.fence.revoked`, and `ownership.lease.expired`.
- `required_evidence`: stale instance writes are rejected after lease takeover even if that instance still thinks it owns the file.
- `rollback_criteria`: any stale claimant can overwrite a newer claimant after fence change.
- `promotion_gate`: `pytest tests/test_file_ownership.py tests/test_ownership_crdt.py tests/test_ownership_fencing.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- `success_criteria`: file ownership becomes crash-tolerant rather than advisory.
- Why this creates value: it prevents the exact “two instances fighting over changes” failure mode the user described.
- Why this creates moat: few coding-agent systems implement real fencing semantics.
- Main risk or failure mode: aggressive lease expiry causing unnecessary churn.
- Smallest credible first experiment: add epoch numbers to `claim_files` and reject writes if local epoch is older than ledger epoch.
- Confidence level: 0.93

### Candidate 04. Symbol Ownership Graph

- Name: Symbol Ownership Graph
- Suggested `phase_id`: `analysis_upgrade`
- Core insight: claims should expand from file-only to symbol-aware when Saguaro can resolve semantic ownership safely.
- External inspiration or analogy: SemanticMerge, AST-aware merge, structure-aware lock domains.
- Why it fits Saguaro and Anvil specifically: the repo already uses Saguaro slices and has semantic graph infrastructure.
- Exact places in this codebase where it could wire in: `core/ownership/file_ownership.py`, `saguaro/omnigraph/*`, `domains/code_intelligence/saguaro_substrate.py`, `core/campaign/worktree_manager.py`.
- Existing primitives it can reuse: file claims, graph refs, slices, impact analysis, roadmap validator graph extraction.
- New primitive, data flow, or subsystem needed: symbol claim map keyed by file path plus symbol identity plus claim mode.
- `repo_scope`: `core/ownership/*`, `saguaro/omnigraph/*`, `domains/code_intelligence/*`, `tests/test_symbol_ownership.py`.
- `owning_specialist_type`: `semantic_runtime_engineer`
- `allowed_writes`: `core/ownership/*.py`, `saguaro/omnigraph/*.py`, `domains/code_intelligence/*.py`, `tests/test_symbol_ownership.py`.
- `telemetry_contract`: emit `ownership.symbol.claimed`, `ownership.symbol.conflict`, and `ownership.symbol.promoted`.
- `required_evidence`: two instances can concurrently edit disjoint methods in one file without triggering a file-level deadlock.
- `rollback_criteria`: symbol claims produce false safety and allow a semantically interfering merge.
- `promotion_gate`: `pytest tests/test_symbol_ownership.py tests/test_file_ownership.py` plus `./venv/bin/saguaro impact --path core/ownership/file_ownership.py`.
- `success_criteria`: safe concurrency improves on refactor-heavy files.
- Why this creates value: it unlocks parallelism without throwing away safety.
- Why this creates moat: symbol-aware ownership tied to semantic analysis is unusually differentiated.
- Main risk or failure mode: symbol identity drift under refactors.
- Smallest credible first experiment: symbol claims only for Python function and class definitions surfaced by Saguaro slices.
- Confidence level: 0.71

### Candidate 05. Intent Announcement Graph

- Name: Intent Announcement Graph
- Suggested `phase_id`: `research`
- Core insight: overlap detection should compare structured intent packets, not just tokenized instructions.
- External inspiration or analogy: contract-net protocols, blackboard systems, requirements traceability.
- Why it fits Saguaro and Anvil specifically: `TaskAnnouncer` and `CollaborationNegotiator` already exist, but are shallow.
- Exact places in this codebase where it could wire in: `core/collaboration/task_announcer.py`, `core/collaboration/negotiation.py`, `core/campaign/control_plane.py`, `core/subagent_communication.py`.
- Existing primitives it can reuse: proposals, overlap results, shared files, message bus topics, campaign task packets.
- New primitive, data flow, or subsystem needed: structured intent packets containing goals, expected files, expected symbols, proof obligations, and dependency edges.
- `repo_scope`: `core/collaboration/*.py`, `core/campaign/*.py`, `core/subagent_communication.py`, `tests/test_intent_graph.py`.
- `owning_specialist_type`: `planner_systems_engineer`
- `allowed_writes`: `core/collaboration/*.py`, `core/campaign/*.py`, `tests/test_intent_graph.py`, `tests/test_collaboration_commands.py`.
- `telemetry_contract`: emit `intent.packet.announced`, `intent.overlap.detected`, `intent.dependency.derived`, and `intent.packet.stale`.
- `required_evidence`: overlapping tasks are classified with reason codes beyond lexical similarity.
- `rollback_criteria`: packet richness collapses task announcement latency or overwhelms context budgets.
- `promotion_gate`: `pytest tests/test_intent_graph.py tests/test_collaboration_commands.py` and `./venv/bin/saguaro build-graph`.
- `success_criteria`: conflicts become explainable in terms of intent, not token overlap.
- Why this creates value: it makes arbitration actionable.
- Why this creates moat: traceable intent packets are a durable coordination primitive.
- Main risk or failure mode: over-structuring before the repo has enough extraction accuracy.
- Smallest credible first experiment: extend `TaskAnnouncer._task_to_dict` with `phase_id`, `campaign_id`, `context_symbols`, and `verification_targets`.
- Confidence level: 0.88

### Candidate 06. Architect Arbitration Plane

- Name: Architect Arbitration Plane
- Suggested `phase_id`: `roadmap_draft`
- Core insight: overlapping instances need an elected architect layer that can merge plans, split ownership, and issue binding decisions.
- External inspiration or analogy: NASA flight director, etcd election, Formula 1 race strategist.
- Why it fits Saguaro and Anvil specifically: `CampaignControlPlane` is already a proto-kernel; it should become architect-aware rather than replaced.
- Exact places in this codebase where it could wire in: `core/campaign/control_plane.py`, `core/campaign/runner.py`, `core/subagent_communication.py`, `core/networking/peer_transport.py`, `cli/repl.py`.
- Existing primitives it can reuse: task packets, control plane transitions, message bus handoff, peer transport, event store.
- New primitive, data flow, or subsystem needed: `core/architect/architect_plane.py` with election, arbitration journal, and binding assignment API.
- `repo_scope`: `core/campaign/*.py`, `core/architect/*.py`, `core/networking/*.py`, `tests/test_architect_arbitration.py`.
- `owning_specialist_type`: `systems_architect`
- `allowed_writes`: `core/campaign/*.py`, `core/architect/*.py`, `cli/repl.py`, `tests/test_architect_arbitration.py`, `tests/test_campaign_runner.py`.
- `telemetry_contract`: emit `architect.elected`, `architect.arbitration.started`, `architect.assignment.issued`, and `architect.assignment.overridden`.
- `required_evidence`: when two instances propose conflicting edits, the elected architect emits a merged execution plan with ownership rebalancing.
- `rollback_criteria`: split-brain leadership or non-deterministic assignment results.
- `promotion_gate`: `pytest tests/test_architect_arbitration.py tests/test_campaign_runner.py tests/test_campaign_control_kernel.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- `success_criteria`: overlapping work no longer devolves into last-writer-wins chaos.
- Why this creates value: it answers the user’s requested “software architect layer.”
- Why this creates moat: coordinated plan arbitration is harder than multi-agent chat and much more defensible.
- Main risk or failure mode: leadership election complexity outpacing near-term value.
- Smallest credible first experiment: single-machine elected architect among multiple local REPL instances using a shared ledger prefix.
- Confidence level: 0.82

### Candidate 07. Validation-Gated Delta Relay

- Name: Validation-Gated Delta Relay
- Suggested `phase_id`: `development`
- Core insight: validated changes should stream to other instances immediately, but only after passing explicit gates.
- External inspiration or analogy: Bazel remote cache, CI merge queues, staged rollout.
- Why it fits Saguaro and Anvil specifically: `CampaignWorktreeManager.promote` already copies changed files back; it needs a relay and gate around that action.
- Exact places in this codebase where it could wire in: `core/campaign/worktree_manager.py`, `core/campaign/control_plane.py`, `shared_kernel/event_store.py`, `saguaro/state/ledger.py`.
- Existing primitives it can reuse: baseline hashes, changed file detection, promotion, verification lane, event export.
- New primitive, data flow, or subsystem needed: delta capsules containing diff manifest, validation proof, and affected-path signatures.
- `repo_scope`: `core/campaign/*.py`, `shared_kernel/*.py`, `saguaro/state/*.py`, `tests/test_delta_distribution.py`.
- `owning_specialist_type`: `promotion_pipeline_engineer`
- `allowed_writes`: `core/campaign/*.py`, `shared_kernel/event_store.py`, `saguaro/state/ledger.py`, `tests/test_delta_distribution.py`.
- `telemetry_contract`: emit `delta.capsule.built`, `delta.capsule.verified`, `delta.capsule.relayed`, and `delta.capsule.rejected`.
- `required_evidence`: after one instance validates a lane, another instance hydrates the promoted files without re-resolving the same delta manually.
- `rollback_criteria`: unverified changes propagate or verified changes hydrate inconsistently.
- `promotion_gate`: `pytest tests/test_delta_distribution.py tests/test_campaign_control_kernel.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- `success_criteria`: other instances work on fresh validated code rather than stale local state.
- Why this creates value: it addresses the user’s “updated into other Anvil instances immediately” requirement.
- Why this creates moat: most agent tools have no validated delta propagation at all.
- Main risk or failure mode: excessive churn if deltas relay too frequently.
- Smallest credible first experiment: relay only phase-complete or test-passing lane promotions.
- Confidence level: 0.89

### Candidate 08. Phase Packet CRDT

- Name: Phase Packet CRDT
- Suggested `phase_id`: `roadmap_draft`
- Core insight: roadmap and campaign state should converge as structured packets rather than freehand markdown alone.
- External inspiration or analogy: Automerge repo docs, local-first metadata replication.
- Why it fits Saguaro and Anvil specifically: `RoadmapCompiler`, `RoadmapValidator`, and `CampaignControlPlane.create_task_packet` already treat phases as structured objects.
- Exact places in this codebase where it could wire in: `core/campaign/roadmap_compiler.py`, `saguaro/roadmap/validator.py`, `core/campaign/control_plane.py`, `core/campaign/phase_packet.py`.
- Existing primitives it can reuse: phase ids, phase packets, task graph, validator traceability.
- New primitive, data flow, or subsystem needed: replicated phase packet store with stable field-level merges and conflict annotations.
- `repo_scope`: `core/campaign/*.py`, `saguaro/roadmap/*.py`, `tests/test_phase_packet_crdt.py`.
- `owning_specialist_type`: `roadmap_systems_engineer`
- `allowed_writes`: `core/campaign/*.py`, `saguaro/roadmap/*.py`, `tests/test_phase_packet_crdt.py`, `tests/test_campaign_roadmap_phase_pack.py`.
- `telemetry_contract`: emit `phase_packet.updated`, `phase_packet.conflicted`, `phase_packet.reconciled`, and `phase_packet.promoted`.
- `required_evidence`: two instances can update the same roadmap phase packet without corrupting required fields.
- `rollback_criteria`: packet convergence produces validator-invalid phase state.
- `promotion_gate`: `pytest tests/test_phase_packet_crdt.py tests/test_campaign_roadmap_phase_pack.py tests/test_saguaro_roadmap_validator.py`.
- `success_criteria`: roadmap collaboration becomes machine-mergeable without losing markdown renderability.
- Why this creates value: it removes roadmap thrash between parallel instances.
- Why this creates moat: executable CRDT roadmap packets are unusual and defensible.
- Main risk or failure mode: markdown and packet views drifting apart.
- Smallest credible first experiment: maintain hidden YAML/JSON sidecar phase packets that render back into markdown.
- Confidence level: 0.74

### Candidate 09. Campaign Awareness Memory Projection

- Name: Campaign Awareness Memory Projection
- Suggested `phase_id`: `research`
- Core insight: instances need durable memory of peer decisions, denied claims, arbitration outcomes, and verified capsules.
- External inspiration or analogy: mission logs, blackboard memory, incident timeline systems.
- Why it fits Saguaro and Anvil specifically: `MemoryFabricStore` already supports objects, edges, reads, feedback, and latent packages.
- Exact places in this codebase where it could wire in: `core/memory/fabric/store.py`, `core/memory/fabric/retrieval_planner.py`, `core/memory/fabric/projectors.py`, `core/campaign/control_plane.py`, `cli/repl.py`.
- Existing primitives it can reuse: memory objects, read records, feedback, projectors, retrieval planner, project memory.
- New primitive, data flow, or subsystem needed: campaign coordination memory schema with instance, claim, and verdict objects.
- `repo_scope`: `core/memory/fabric/*.py`, `core/campaign/control_plane.py`, `cli/repl.py`, `tests/test_campaign_memory_projection.py`.
- `owning_specialist_type`: `memory_systems_engineer`
- `allowed_writes`: `core/memory/fabric/*.py`, `core/campaign/*.py`, `cli/repl.py`, `tests/test_campaign_memory_projection.py`.
- `telemetry_contract`: emit `memory.coord.ingested`, `memory.coord.retrieved`, `memory.coord.stale`, and `memory.coord.feedback`.
- `required_evidence`: a new instance joining a campaign can reconstruct the current coordination picture from memory plus ledger.
- `rollback_criteria`: stale memory causes bad claim routing or stale prompt posture.
- `promotion_gate`: `pytest tests/test_campaign_memory_projection.py tests/test_memory_governance.py tests/test_memory_snapshot_restore.py`.
- `success_criteria`: late-joining instances stop behaving like amnesiac clones.
- Why this creates value: session and campaign awareness become durable, not ephemeral.
- Why this creates moat: persistent coordination memory compounds over time.
- Main risk or failure mode: retrieval quality under high memory volume.
- Smallest credible first experiment: ingest architect verdicts and ownership denials into the memory fabric.
- Confidence level: 0.86

### Candidate 10. Semantic Conflict Radar

- Name: Semantic Conflict Radar
- Suggested `phase_id`: `analysis_upgrade`
- Core insight: detect semantically dangerous overlaps before promotion using Saguaro graph, impact, and static interference heuristics.
- External inspiration or analogy: SafeMerge, semantic conflict static analysis, race prediction.
- Why it fits Saguaro and Anvil specifically: Saguaro already has impact, graph, verify, and code slicing.
- Exact places in this codebase where it could wire in: `saguaro/state/ledger.py`, `domains/code_intelligence/saguaro_substrate.py`, `core/campaign/control_plane.py`, `core/collaboration/negotiation.py`.
- Existing primitives it can reuse: `impact --path`, slices, verify engines, roadmap validator graph refs.
- New primitive, data flow, or subsystem needed: semantic interference score over concurrent delta capsules.
- `repo_scope`: `domains/code_intelligence/*`, `core/campaign/*.py`, `core/collaboration/*.py`, `tests/test_semantic_conflict_radar.py`.
- `owning_specialist_type`: `program_analysis_engineer`
- `allowed_writes`: `domains/code_intelligence/*.py`, `core/campaign/*.py`, `core/collaboration/*.py`, `tests/test_semantic_conflict_radar.py`.
- `telemetry_contract`: emit `conflict.semantic.scored`, `conflict.semantic.blocked`, and `conflict.semantic.false_positive`.
- `required_evidence`: concurrent deltas with clean textual merge but unsafe behavioral interference are flagged before promotion.
- `rollback_criteria`: the radar blocks too many harmless merges without useful explanation.
- `promotion_gate`: `pytest tests/test_semantic_conflict_radar.py tests/test_campaign_control_kernel.py` and `./venv/bin/saguaro impact --path core/campaign/worktree_manager.py`.
- `success_criteria`: architect decisions become semantically informed.
- Why this creates value: fewer hidden regressions from parallel work.
- Why this creates moat: semantic interference handling is a genuine deep primitive.
- Main risk or failure mode: too many false positives on dynamic or framework-heavy code.
- Smallest credible first experiment: score only shared call sites and shared mutated public APIs.
- Confidence level: 0.76

### Candidate 11. Lease Heartbeat Recovery Drills

- Name: Lease Heartbeat Recovery Drills
- Suggested `phase_id`: `deep_test_audit`
- Core insight: every coordination primitive must survive crashes, process death, and half-applied promotions.
- External inspiration or analogy: chaos engineering, failure drills, air-traffic failover.
- Why it fits Saguaro and Anvil specifically: ownership leases, worktree lanes, and ledger sync already expose expiry and replay surfaces.
- Exact places in this codebase where it could wire in: `core/ownership/file_ownership.py`, `saguaro/state/ledger.py`, `core/campaign/worktree_manager.py`, `tests/test_state_ledger.py`.
- Existing primitives it can reuse: TTLs, heartbeats, reapers, checkpoints, workspace snapshots.
- New primitive, data flow, or subsystem needed: crash-simulation harness for claims, promotions, and sync.
- `repo_scope`: `core/ownership/*`, `saguaro/state/*.py`, `core/campaign/*.py`, `tests/test_connectivity_chaos.py`.
- `owning_specialist_type`: `resilience_engineer`
- `allowed_writes`: `core/ownership/*.py`, `saguaro/state/*.py`, `tests/test_connectivity_chaos.py`.
- `telemetry_contract`: emit `recovery.replay.started`, `recovery.replay.completed`, `recovery.orphan.cleaned`, and `recovery.split_brain.detected`.
- `required_evidence`: killing a leader during ownership or promotion does not leave permanent stuck claims or orphaned lanes.
- `rollback_criteria`: replay cannot determine authoritative owner or promoted state.
- `promotion_gate`: `pytest tests/test_connectivity_chaos.py tests/test_state_ledger.py tests/test_file_ownership.py`.
- `success_criteria`: failure becomes routine rather than catastrophic.
- Why this creates value: the coordination layer becomes trustworthy enough for real teams.
- Why this creates moat: crash-hard collaboration is rare in agent tooling.
- Main risk or failure mode: chaos harness complexity outgrows coverage value.
- Smallest credible first experiment: inject leader death between claim grant and relay acknowledgment.
- Confidence level: 0.90

### Candidate 12. Merge Queue with Verification Lanes

- Name: Merge Queue with Verification Lanes
- Suggested `phase_id`: `convergence`
- Core insight: all cross-instance promotions should flow through a verification queue that can batch, replay, and prove safety.
- External inspiration or analogy: CI merge queues, release trains, pit-lane release control.
- Why it fits Saguaro and Anvil specifically: `VerificationLane` and `CampaignControlPlane.create_verification_lane` already exist.
- Exact places in this codebase where it could wire in: `domains/verification/verification_lane.py`, `core/campaign/control_plane.py`, `core/campaign/worktree_manager.py`, `shared_kernel/event_store.py`.
- Existing primitives it can reuse: verification lanes, completion proof, event store, lane workspace state.
- New primitive, data flow, or subsystem needed: repo-level promotion queue with queue order, conflict groups, and proof capsules.
- `repo_scope`: `domains/verification/*.py`, `core/campaign/*.py`, `shared_kernel/*.py`, `tests/test_merge_queue.py`.
- `owning_specialist_type`: `release_systems_engineer`
- `allowed_writes`: `domains/verification/*.py`, `core/campaign/*.py`, `shared_kernel/*.py`, `tests/test_merge_queue.py`.
- `telemetry_contract`: emit `queue.item.enqueued`, `queue.item.batched`, `queue.item.promoted`, and `queue.item.rejected`.
- `required_evidence`: multiple validated deltas can be ordered and promoted without reintroducing stale conflicts.
- `rollback_criteria`: queue starvation or promotion ordering inversions.
- `promotion_gate`: `pytest tests/test_merge_queue.py tests/test_campaign_timeline_and_verification_lane.py`.
- `success_criteria`: repo promotion becomes deterministic under load.
- Why this creates value: concurrency stops being synonymous with merge roulette.
- Why this creates moat: promotion discipline scales to company server and public-repo modes.
- Main risk or failure mode: queue latency harms local developer flow.
- Smallest credible first experiment: queue only deltas touching shared files or shared symbols.
- Confidence level: 0.84

### Candidate 13. Capability-Aware Work Scheduler

- Name: Capability-Aware Work Scheduler
- Suggested `phase_id`: `feature_map`
- Core insight: work should route based on machine strength, not only ownership.
- External inspiration or analogy: cluster schedulers, pit-wall tire and pace strategy, heterogeneous compute routing.
- Why it fits Saguaro and Anvil specifically: native QSG runtime exposes hardware-sensitive capabilities and telemetry.
- Exact places in this codebase where it could wire in: `core/native/native_qsg_engine.py`, `core/campaign/control_plane.py`, `core/networking/instance_identity.py`, `core/subagent_communication.py`.
- Existing primitives it can reuse: runtime capabilities, instance identity, task packets, message handoff.
- New primitive, data flow, or subsystem needed: capability vector and placement rules for heavy analysis, verification, and indexing tasks.
- `repo_scope`: `core/native/*.py`, `core/campaign/*.py`, `core/networking/*.py`, `tests/test_capability_scheduler.py`.
- `owning_specialist_type`: `runtime_scheduler_engineer`
- `allowed_writes`: `core/native/*.py`, `core/campaign/*.py`, `core/networking/*.py`, `tests/test_capability_scheduler.py`.
- `telemetry_contract`: emit `scheduler.task.placed`, `scheduler.task.migrated`, `scheduler.task.declined`, and `scheduler.capability.updated`.
- `required_evidence`: expensive semantic jobs route to the strongest available node while low-latency editing stays local.
- `rollback_criteria`: placement churn or latency regressions on interactive tasks.
- `promotion_gate`: `pytest tests/test_capability_scheduler.py tests/test_campaign_runner.py`.
- `success_criteria`: fleet throughput improves without increasing user-visible lag.
- Why this creates value: local laptops stop doing server-class work they are bad at.
- Why this creates moat: capability-aware code-intelligence scheduling is hard to reproduce.
- Main risk or failure mode: noisy or inaccurate runtime capability reports.
- Smallest credible first experiment: advertise a simple `analysis_capacity` and `verification_capacity` score in presence heartbeats.
- Confidence level: 0.80

### Candidate 14. Repo Digital Twin

- Name: Repo Digital Twin
- Suggested `phase_id`: `analysis_upgrade`
- Core insight: maintain a live model of hotspots, active claims, volatility, and pending promotions as one federated operational picture.
- External inspiration or analogy: digital twins, mission dashboards, race telemetry walls.
- Why it fits Saguaro and Anvil specifically: control plane, impact, roadmap validator graph, and state ledger already cover most raw inputs.
- Exact places in this codebase where it could wire in: `core/campaign/control_plane.py`, `saguaro/state/ledger.py`, `saguaro/roadmap/validator.py`, `core/campaign/telemetry.py`.
- Existing primitives it can reuse: task graph, timeline, telemetry, ledger deltas, graph refs.
- New primitive, data flow, or subsystem needed: `core/connectivity/repo_twin.py` that synthesizes operational state and risk maps.
- `repo_scope`: `core/campaign/*.py`, `core/connectivity/*.py`, `saguaro/state/*.py`, `saguaro/roadmap/*.py`, `tests/test_repo_twin.py`.
- `owning_specialist_type`: `systems_observability_engineer`
- `allowed_writes`: `core/campaign/*.py`, `core/connectivity/*.py`, `saguaro/state/*.py`, `tests/test_repo_twin.py`.
- `telemetry_contract`: emit `repo_twin.updated`, `repo_twin.hotspot.detected`, and `repo_twin.risk.band.changed`.
- `required_evidence`: operators can see active claims, blocked promotions, semantic conflict risk, and stale peers from one view.
- `rollback_criteria`: twin becomes stale enough to mislead scheduling or arbitration.
- `promotion_gate`: `pytest tests/test_repo_twin.py tests/test_campaign_timeline_and_verification_lane.py`.
- `success_criteria`: the repo becomes operationally legible at team scale.
- Why this creates value: humans and architect agents get the same current picture.
- Why this creates moat: digital-twin visibility for code operations is rare.
- Main risk or failure mode: overbuilding dashboards before control logic is mature.
- Smallest credible first experiment: render a console-only twin from ledger, ownership snapshot, and queue state.
- Confidence level: 0.78

### Candidate 15. Ownership-Aware Prompt Assembly

- Name: Ownership-Aware Prompt Assembly
- Suggested `phase_id`: `development`
- Core insight: the model should know current claim constraints and peer intent before it proposes edits.
- External inspiration or analogy: cockpit briefing before takeoff, mission rules of engagement.
- Why it fits Saguaro and Anvil specifically: `SystemPromptBuilder.build` and `UnifiedChatLoop` already support dynamic prompt context.
- Exact places in this codebase where it could wire in: `core/prompts/system_prompt_builder.py`, `core/unified_chat_loop.py`, `cli/repl.py`, `core/ownership/file_ownership.py`.
- Existing primitives it can reuse: prompt builder, shared context, runtime status, ownership snapshot.
- New primitive, data flow, or subsystem needed: prompt directives for live claims, architect decisions, and trust-zone posture.
- `repo_scope`: `core/prompts/*.py`, `core/unified_chat_loop.py`, `cli/repl.py`, `tests/test_connectivity_prompt_context.py`.
- `owning_specialist_type`: `prompt_runtime_engineer`
- `allowed_writes`: `core/prompts/*.py`, `core/unified_chat_loop.py`, `cli/repl.py`, `tests/test_connectivity_prompt_context.py`.
- `telemetry_contract`: emit `prompt.connectivity.context.refreshed` and `prompt.connectivity.conflict_injected`.
- `required_evidence`: the model explicitly avoids proposing writes into fenced files or recommends negotiation when appropriate.
- `rollback_criteria`: prompt bloat degrades response quality or latency.
- `promotion_gate`: `pytest tests/test_connectivity_prompt_context.py tests/test_collaboration_commands.py`.
- `success_criteria`: coordination posture becomes proactive instead of reactive.
- Why this creates value: fewer doomed edit attempts.
- Why this creates moat: prompt assembly becomes repo-operational rather than generic.
- Main risk or failure mode: stale context causing overconstrained prompts.
- Smallest credible first experiment: inject top five active conflicting claims into prompt context when editing is requested.
- Confidence level: 0.88

### Candidate 16. Trust Zones and Policy Rings

- Name: Trust Zones and Policy Rings
- Suggested `phase_id`: `intake`
- Core insight: local peers, company servers, and open-source contributors need different coordination rights and promotion paths.
- External inspiration or analogy: zero-trust networking, protected branches, contributor quarantine.
- Why it fits Saguaro and Anvil specifically: `repo_policy_resolver` is already a hook in file ownership.
- Exact places in this codebase where it could wire in: `core/ownership/file_ownership.py`, `core/campaign/control_plane.py`, `saguaro/state/ledger.py`, `cli/commands/agent.py`.
- Existing primitives it can reuse: repo policy resolver, campaign state getter, verification lane, roadmap validator.
- New primitive, data flow, or subsystem needed: trust-zone model with `local`, `campaign`, `company`, and `external` rings.
- `repo_scope`: `core/ownership/*.py`, `core/campaign/*.py`, `saguaro/state/*.py`, `tests/test_governance_trust_zones.py`.
- `owning_specialist_type`: `governance_engineer`
- `allowed_writes`: `core/ownership/*.py`, `core/campaign/*.py`, `saguaro/state/*.py`, `tests/test_governance_trust_zones.py`.
- `telemetry_contract`: emit `policy.zone.assigned`, `policy.zone.denied`, and `policy.zone.escalated`.
- `required_evidence`: external contributors can propose deltas but cannot self-promote into protected repo lanes.
- `rollback_criteria`: policy rings become so rigid that legitimate collaboration stalls.
- `promotion_gate`: `pytest tests/test_governance_trust_zones.py tests/test_ownership_defaults.py`.
- `success_criteria`: the same architecture can serve local, company, and open-source modes safely.
- Why this creates value: it directly addresses the user’s company and public repo concern.
- Why this creates moat: multi-tier trust-aware agent collaboration is difficult and strategic.
- Main risk or failure mode: policy sprawl.
- Smallest credible first experiment: add a zone flag to presence and deny direct promotion from `external`.
- Confidence level: 0.92

### Candidate 17. Validated Artifact CAS

- Name: Validated Artifact Content-Addressable Store
- Suggested `phase_id`: `development`
- Core insight: once a proof, test run, or built delta exists, peers should consume it by hash instead of recomputing it.
- External inspiration or analogy: Bazel CAS, remote cache, build artifact stores.
- Why it fits Saguaro and Anvil specifically: event store, lane hashes, and memory fabric already support durable payload references.
- Exact places in this codebase where it could wire in: `shared_kernel/event_store.py`, `core/campaign/worktree_manager.py`, `core/campaign/control_plane.py`, `core/memory/fabric/store.py`.
- Existing primitives it can reuse: baseline hashes, exported runs, checkpoints, latent package storage.
- New primitive, data flow, or subsystem needed: content-addressable capsule store for validation outputs and patch bundles.
- `repo_scope`: `shared_kernel/*.py`, `core/campaign/*.py`, `core/memory/fabric/*.py`, `tests/test_validation_cas.py`.
- `owning_specialist_type`: `artifact_systems_engineer`
- `allowed_writes`: `shared_kernel/*.py`, `core/campaign/*.py`, `core/memory/fabric/*.py`, `tests/test_validation_cas.py`.
- `telemetry_contract`: emit `artifact.capsule.stored`, `artifact.capsule.fetched`, `artifact.capsule.cache_hit`, and `artifact.capsule.invalidated`.
- `required_evidence`: repeated verification on another instance can reuse a proof capsule when inputs and environment signatures match.
- `rollback_criteria`: invalid cache hits or environment drift cause false trust.
- `promotion_gate`: `pytest tests/test_validation_cas.py tests/test_campaign_closure_safety_case.py`.
- `success_criteria`: duplicate work drops sharply in multi-instance scenarios.
- Why this creates value: company fleets stop wasting cycles on identical validation.
- Why this creates moat: verified artifact reuse tied to agent promotion is not commodity.
- Main risk or failure mode: environment signatures being too weak.
- Smallest credible first experiment: reuse test result capsules only when file hashes and target environment match exactly.
- Confidence level: 0.85

### Candidate 18. Causal Replay Cockpit

- Name: Causal Replay Cockpit
- Suggested `phase_id`: `deep_test_audit`
- Core insight: every ownership fight and architect decision should be replayable as a causal tape.
- External inspiration or analogy: flight data recorder, race replay, distributed trace viewer.
- Why it fits Saguaro and Anvil specifically: event store exports replay tapes; ledger tracks deltas; roadmap validator already emits graphs.
- Exact places in this codebase where it could wire in: `shared_kernel/event_store.py`, `saguaro/state/ledger.py`, `core/subagent_communication.py`, `cli/repl.py`.
- Existing primitives it can reuse: replay tape export, trace segments, checkpoints, list events.
- New primitive, data flow, or subsystem needed: replay browser and causal blame graph for coordination incidents.
- `repo_scope`: `shared_kernel/*.py`, `saguaro/state/*.py`, `core/subagent_communication.py`, `cli/repl.py`, `tests/test_causal_replay.py`.
- `owning_specialist_type`: `observability_engineer`
- `allowed_writes`: `shared_kernel/*.py`, `saguaro/state/*.py`, `core/subagent_communication.py`, `tests/test_causal_replay.py`.
- `telemetry_contract`: emit `replay.generated`, `replay.expanded`, and `replay.counterfactual.requested`.
- `required_evidence`: operators can reconstruct why a file was denied, reassigned, or promoted.
- `rollback_criteria`: replay omits critical edges or diverges from actual event order.
- `promotion_gate`: `pytest tests/test_causal_replay.py tests/test_state_ledger.py`.
- `success_criteria`: debugging multi-instance behavior stops depending on raw logs.
- Why this creates value: operational issues become diagnosable.
- Why this creates moat: replayable agent coordination is a serious differentiator.
- Main risk or failure mode: too much recorded volume without summarization.
- Smallest credible first experiment: generate a text timeline for one file claim conflict from event store plus ledger.
- Confidence level: 0.83

### Candidate 19. LAN Plus Company Broker Federation

- Name: LAN Plus Company Broker Federation
- Suggested `phase_id`: `intake`
- Core insight: local collaboration and company-server collaboration should share one protocol with different rendezvous backends.
- External inspiration or analogy: Yjs network agnosticism, client/server plus peer mesh.
- Why it fits Saguaro and Anvil specifically: `PeerDiscovery` already has MDNS, filesystem, and rendezvous concepts.
- Exact places in this codebase where it could wire in: `core/networking/peer_discovery.py`, `core/networking/peer_transport.py`, `core/networking/instance_identity.py`, `cli/repl.py`.
- Existing primitives it can reuse: MDNS discovery, filesystem discovery, rendezvous stub, instance registry.
- New primitive, data flow, or subsystem needed: transport provider layer with LAN mesh, company broker, and offline spool modes.
- `repo_scope`: `core/networking/*.py`, `cli/repl.py`, `tests/test_transport_providers.py`.
- `owning_specialist_type`: `network_runtime_engineer`
- `allowed_writes`: `core/networking/*.py`, `cli/repl.py`, `tests/test_transport_providers.py`.
- `telemetry_contract`: emit `transport.provider.selected`, `transport.peer.latency`, `transport.peer.partitioned`, and `transport.mode.degraded`.
- `required_evidence`: the same repo can federate on local LAN or via company broker without changing the coordination contract.
- `rollback_criteria`: provider abstraction leaks and produces inconsistent semantics.
- `promotion_gate`: `pytest tests/test_transport_providers.py tests/test_repo_presence.py`.
- `success_criteria`: one protocol, multiple deployment modes.
- Why this creates value: the architecture scales from two local instances to company infrastructure.
- Why this creates moat: deployment-flexible coordination reduces adoption friction.
- Main risk or failure mode: protocol complexity rising too early.
- Smallest credible first experiment: LAN discovery and broker discovery share the same presence envelope schema.
- Confidence level: 0.86

### Candidate 20. AST-Aware Change Capsules

- Name: AST-Aware Change Capsules
- Suggested `phase_id`: `development`
- Core insight: source deltas should be shipped as structured change capsules with symbol moves and semantic intent, not only file diffs.
- External inspiration or analogy: SemanticMerge, AST differencing, refactoring-aware merge.
- Why it fits Saguaro and Anvil specifically: Saguaro and the roadmap validator already operate on code structure and graph refs.
- Exact places in this codebase where it could wire in: `core/campaign/worktree_manager.py`, `domains/code_intelligence/saguaro_substrate.py`, `saguaro/state/ledger.py`, `shared_kernel/event_store.py`.
- Existing primitives it can reuse: baseline hashes, changed files, slices, graph refs, event capsules.
- New primitive, data flow, or subsystem needed: capsule serializer storing file diff, symbol map, impact set, and verification witnesses.
- `repo_scope`: `core/campaign/*.py`, `domains/code_intelligence/*.py`, `saguaro/state/*.py`, `tests/test_change_capsules.py`.
- `owning_specialist_type`: `merge_systems_engineer`
- `allowed_writes`: `core/campaign/*.py`, `domains/code_intelligence/*.py`, `saguaro/state/*.py`, `tests/test_change_capsules.py`.
- `telemetry_contract`: emit `capsule.ast.built`, `capsule.ast.replayed`, and `capsule.ast.refactor_detected`.
- `required_evidence`: a refactor-heavy delta can still be reasoned about without collapsing to file-level text conflict only.
- `rollback_criteria`: capsule build time becomes too expensive for normal development flow.
- `promotion_gate`: `pytest tests/test_change_capsules.py tests/test_campaign_roadmap_phase_pack.py`.
- `success_criteria`: promotions retain semantic structure.
- Why this creates value: better arbitration and replay.
- Why this creates moat: structured source capsules bridge version control and semantic analysis.
- Main risk or failure mode: language coverage gaps.
- Smallest credible first experiment: capsuleize only Python class and function edits.
- Confidence level: 0.79

### Candidate 21. Virtual Workspace Union Branch

- Name: Virtual Workspace Union Branch
- Suggested `phase_id`: `development`
- Core insight: maintain an explicit synthetic workspace manifest representing the union of all active validated lanes.
- External inspiration or analogy: GitButler workspace branch.
- Why it fits Saguaro and Anvil specifically: `CampaignWorktreeManager` already prepares overlay lanes but does not materialize a durable union view.
- Exact places in this codebase where it could wire in: `core/campaign/worktree_manager.py`, `saguaro/state/ledger.py`, `core/campaign/control_plane.py`, `cli/repl.py`.
- Existing primitives it can reuse: workspace dir, changed files, baseline hashes, snapshot and workspace status.
- New primitive, data flow, or subsystem needed: synthetic workspace manifest and optional union worktree for operator inspection.
- `repo_scope`: `core/campaign/*.py`, `saguaro/state/*.py`, `cli/repl.py`, `tests/test_virtual_workspace_union.py`.
- `owning_specialist_type`: `workspace_engineer`
- `allowed_writes`: `core/campaign/*.py`, `saguaro/state/*.py`, `cli/repl.py`, `tests/test_virtual_workspace_union.py`.
- `telemetry_contract`: emit `workspace.union.updated`, `workspace.union.diverged`, and `workspace.union.rebuilt`.
- `required_evidence`: operators can inspect the currently-active union of validated branches without guessing which lane state is authoritative.
- `rollback_criteria`: union view diverges from actual promoted lane set.
- `promotion_gate`: `pytest tests/test_virtual_workspace_union.py tests/test_campaign_runner.py`.
- `success_criteria`: multi-lane repo state is visible and reproducible.
- Why this creates value: parallel work stops hiding behind opaque lane folders.
- Why this creates moat: explicit union workspace semantics are powerful and rare.
- Main risk or failure mode: confusion between union view and promotable truth.
- Smallest credible first experiment: generate a read-only manifest before generating a writable union.
- Confidence level: 0.81

### Candidate 22. Architect Council

- Name: Architect Council
- Suggested `phase_id`: `convergence`
- Core insight: at company scale, one architect instance becomes a bottleneck; use a hierarchical council with scoped authority.
- External inspiration or analogy: command hierarchy, distributed control rooms, federated schedulers.
- Why it fits Saguaro and Anvil specifically: control plane already understands phases, repos, and specialization; it can be extended to repo, campaign, and org scopes.
- Exact places in this codebase where it could wire in: `core/campaign/control_plane.py`, `core/architect/architect_plane.py`, `core/networking/peer_transport.py`, `saguaro/state/ledger.py`.
- Existing primitives it can reuse: task packets, telemetry, arbitration journal, peer discovery.
- New primitive, data flow, or subsystem needed: scoped architect roles for repo lead, campaign lead, and org lead.
- `repo_scope`: `core/campaign/*.py`, `core/architect/*.py`, `core/networking/*.py`, `tests/test_architect_council.py`.
- `owning_specialist_type`: `federation_architect`
- `allowed_writes`: `core/campaign/*.py`, `core/architect/*.py`, `core/networking/*.py`, `tests/test_architect_council.py`.
- `telemetry_contract`: emit `architect.scope.claimed`, `architect.scope.escalated`, and `architect.scope.delegated`.
- `required_evidence`: repo-level architect decisions can escalate to company policy or org-wide resource decisions without losing traceability.
- `rollback_criteria`: authority ambiguity between architect tiers.
- `promotion_gate`: `pytest tests/test_architect_council.py tests/test_architect_arbitration.py`.
- `success_criteria`: coordination still works when the fleet grows by orders of magnitude.
- Why this creates value: the architecture remains viable beyond a small team.
- Why this creates moat: hierarchical multi-agent control is a durable systems moat.
- Main risk or failure mode: overengineering before adoption.
- Smallest credible first experiment: repo architect plus optional company architect, no deeper hierarchy initially.
- Confidence level: 0.68

### Candidate 23. Promotion Proof Packets

- Name: Promotion Proof Packets
- Suggested `phase_id`: `deep_test_audit`
- Core insight: every promoted delta should carry a machine-checkable proof packet linking code refs, tests, graph refs, and verification commands.
- External inspiration or analogy: release attestation, supply-chain provenance, mission certification artifacts.
- Why it fits Saguaro and Anvil specifically: roadmap validator already extracts traceability and completion evidence.
- Exact places in this codebase where it could wire in: `saguaro/roadmap/validator.py`, `core/campaign/control_plane.py`, `shared_kernel/event_store.py`, `domains/verification/verification_lane.py`.
- Existing primitives it can reuse: safety cases, completion proof, validator requirements, verification lane.
- New primitive, data flow, or subsystem needed: promotion proof schema attached to each delta capsule and queue item.
- `repo_scope`: `saguaro/roadmap/*.py`, `core/campaign/*.py`, `shared_kernel/*.py`, `domains/verification/*.py`, `tests/test_promotion_proofs.py`.
- `owning_specialist_type`: `verification_engineer`
- `allowed_writes`: `saguaro/roadmap/*.py`, `core/campaign/*.py`, `shared_kernel/*.py`, `tests/test_promotion_proofs.py`.
- `telemetry_contract`: emit `proof.packet.created`, `proof.packet.missing`, and `proof.packet.verified`.
- `required_evidence`: each promoted item can explain exactly which code, tests, and commands justified promotion.
- `rollback_criteria`: packets are generated but not trustworthy enough to reconstruct promotion basis.
- `promotion_gate`: `pytest tests/test_promotion_proofs.py tests/test_saguaro_roadmap_validator.py tests/test_campaign_closure_safety_case.py`.
- `success_criteria`: promotion becomes auditable by machine and human.
- Why this creates value: company-scale trust improves dramatically.
- Why this creates moat: proof-oriented promotion is unusually strong operational discipline.
- Main risk or failure mode: proof packet friction on early developer flow.
- Smallest credible first experiment: require proof packets only for cross-instance or protected-path promotions.
- Confidence level: 0.89

### Candidate 24. Semantic Race Strategy Engine

- Name: Semantic Race Strategy Engine
- Suggested `phase_id`: `convergence`
- Core insight: workload splitting should optimize for semantic independence, critical path, and machine capability, not naïve 50/50 file counts.
- External inspiration or analogy: Formula 1 race strategy, portfolio schedulers, resource-aware planning.
- Why it fits Saguaro and Anvil specifically: `CampaignControlPlane`, capability discovery, ownership, and semantic conflict radar together can support true strategy.
- Exact places in this codebase where it could wire in: `core/campaign/control_plane.py`, `core/campaign/telemetry.py`, `core/architect/architect_plane.py`, `core/native/native_qsg_engine.py`.
- Existing primitives it can reuse: task graph, telemetry spans, capability data, ownership state, verification queue.
- New primitive, data flow, or subsystem needed: objective function for balancing risk, freshness, compute fit, and promotion urgency.
- `repo_scope`: `core/campaign/*.py`, `core/architect/*.py`, `core/native/*.py`, `tests/test_strategy_engine.py`.
- `owning_specialist_type`: `optimization_engineer`
- `allowed_writes`: `core/campaign/*.py`, `core/architect/*.py`, `core/native/*.py`, `tests/test_strategy_engine.py`.
- `telemetry_contract`: emit `strategy.plan.generated`, `strategy.rebalance.issued`, and `strategy.regret.measured`.
- `required_evidence`: workload splits improve cycle time and lower conflict rates compared with static equal division.
- `rollback_criteria`: strategy churn causes instability or user confusion.
- `promotion_gate`: `pytest tests/test_strategy_engine.py tests/test_architect_arbitration.py`.
- `success_criteria`: the fleet feels coordinated rather than crowded.
- Why this creates value: it converts raw concurrency into real throughput.
- Why this creates moat: intelligent, semantically aware load balancing is a high-end systems capability.
- Main risk or failure mode: reward function gaming or poor proxy metrics.
- Smallest credible first experiment: rebalance only when queue depth and conflict risk exceed thresholds.
- Confidence level: 0.72

### Candidate 25. Open-Source Quarantine Lanes

- Name: Open-Source Quarantine Lanes
- Suggested `phase_id`: `development`
- Core insight: public or low-trust contributions should land in quarantined lanes that cannot touch protected promotion paths directly.
- External inspiration or analogy: protected branches, security review quarantine, patch review sandboxes.
- Why it fits Saguaro and Anvil specifically: lane workspaces, trust zones, and verification lanes already exist as primitives.
- Exact places in this codebase where it could wire in: `core/campaign/worktree_manager.py`, `core/campaign/control_plane.py`, `core/ownership/file_ownership.py`, `domains/verification/verification_lane.py`.
- Existing primitives it can reuse: lane preparation, repo policy resolver, verification lanes, completion proof.
- New primitive, data flow, or subsystem needed: quarantine lane class with stricter promotion contract and mandatory architect review.
- `repo_scope`: `core/campaign/*.py`, `core/ownership/*.py`, `domains/verification/*.py`, `tests/test_quarantine_lanes.py`.
- `owning_specialist_type`: `security_governance_engineer`
- `allowed_writes`: `core/campaign/*.py`, `core/ownership/*.py`, `domains/verification/*.py`, `tests/test_quarantine_lanes.py`.
- `telemetry_contract`: emit `quarantine.lane.created`, `quarantine.lane.escalated`, and `quarantine.lane.rejected`.
- `required_evidence`: external contributions are isolated, reviewable, and promotable only through protected gates.
- `rollback_criteria`: legitimate external work becomes too costly to integrate.
- `promotion_gate`: `pytest tests/test_quarantine_lanes.py tests/test_governance_trust_zones.py`.
- `success_criteria`: open-source or vendor collaboration becomes safe enough to support.
- Why this creates value: it broadens deployment modes without collapsing trust.
- Why this creates moat: one coordination architecture can serve private and public repos.
- Main risk or failure mode: operator fatigue from too many quarantine reviews.
- Smallest credible first experiment: mark quarantine lanes read-only until proof packet and architect approval exist.
- Confidence level: 0.88

### Candidate 26. Runtime-Symbol Federation

- Name: Runtime-Symbol Federation
- Suggested `phase_id`: `analysis_upgrade`
- Core insight: native runtime capabilities and symbol closures should influence coordination decisions, not stay trapped in one process.
- External inspiration or analogy: fleet capability maps, binary compatibility tables.
- Why it fits Saguaro and Anvil specifically: the roadmap validator already reasons about runtime symbols, and the native engine surfaces capabilities.
- Exact places in this codebase where it could wire in: `core/native/native_qsg_engine.py`, `saguaro/roadmap/validator.py`, `core/networking/instance_identity.py`, `core/campaign/control_plane.py`.
- Existing primitives it can reuse: runtime symbol resolver, native capability construction, presence heartbeats.
- New primitive, data flow, or subsystem needed: runtime capability digest and symbol closure advertisement per instance.
- `repo_scope`: `core/native/*.py`, `saguaro/roadmap/*.py`, `core/networking/*.py`, `tests/test_runtime_symbol_federation.py`.
- `owning_specialist_type`: `runtime_integration_engineer`
- `allowed_writes`: `core/native/*.py`, `saguaro/roadmap/*.py`, `core/networking/*.py`, `tests/test_runtime_symbol_federation.py`.
- `telemetry_contract`: emit `runtime.symbol.digest.published` and `runtime.capability.compatibility.checked`.
- `required_evidence`: a peer can decide whether another node is eligible for a native-heavy workload without blind dispatch.
- `rollback_criteria`: symbol digest mismatches lead to bad placements.
- `promotion_gate`: `pytest tests/test_runtime_symbol_federation.py tests/test_qsg_state_kernels.py`.
- `success_criteria`: native-heavy jobs route only to compatible peers.
- Why this creates value: fewer failed offloads and better hardware use.
- Why this creates moat: this ties native runtime knowledge into agent orchestration.
- Main risk or failure mode: unstable capability digests after environment changes.
- Smallest credible first experiment: advertise architecture, kernel mode, and context profile only.
- Confidence level: 0.77

---

## 6. Critical Pressure Test

### 6.1 Elegant but likely wrong

[Synthesis]

These ideas are elegant but easy to over-romanticize:

- Phase Packet CRDT as the immediate answer for all roadmap collaboration
- Architect Council before a simpler single-architect plane is proven
- Symbol Ownership Graph before claim semantics, refactor identity, and conflict radar are mature

Why they may be wrong:

- CRDTs solve field convergence, not human-intent correctness
- multi-tier control introduces authority complexity very early
- symbol claims can create false confidence if semantic interference detection is weak

### 6.2 Ugly but strategically powerful

[Synthesis]

These ideas are less glamorous but strategically potent:

- fencing-token file ownership
- causal repo state ledger
- validation-gated delta relay
- trust zones and quarantine lanes
- virtual workspace union manifest

Why they matter:

- they create operational discipline
- they constrain the failure modes that destroy trust first
- they make more ambitious layers possible later

### 6.3 Ideas likely to fail unless the repo gains one key primitive

[Synthesis]

Missing primitive: authoritative delta capsule schema.

Without this:

- architect arbitration cannot reason about concurrent work robustly
- semantic conflict radar lacks a normalized unit of analysis
- delta relay cannot be safely gated
- promotion proof packets become ad hoc

The strongest candidate architecture depends on delta capsules as the standard transport unit.

### 6.4 Where external research conflicts with engineering reality

[Synthesis]

CRDT research says convergence is easy.

Practitioner discussions remind us:

- deterministic merge is not the same as correct merge
- domain semantics matter
- human review remains necessary in ambiguous edits

That means:

- use CRDTs for plan state
- do not use unconstrained CRDTs as the final code-promotion authority

### 6.5 Where the repo itself is missing leverage

[Repo-grounded observation]

Thin leverage points today:

- `PeerTransport` is too lightweight to serve as the durable federation layer
- `TaskAnnouncer` overlap is too shallow for real arbitration
- `CampaignTelemetry` is too thin to drive fleet decisions
- `CampaignWorktreeManager.promote` is too blunt for proof-oriented promotion

### 6.6 Failure scenarios this roadmap must survive

[Synthesis]

- two instances claim one file and one crashes
- two instances edit different symbols in one file and a refactor moves them
- one instance validates a delta and another hydrates stale pre-gate state
- architect leader dies mid-assignment
- company server is unreachable but local work must continue
- open-source external contribution attempts to modify protected paths
- proof packets say “safe” but semantic conflict radar flags interference

---

## 7. Synthesis

### 7.1 Strongest overall ideas

[Synthesis]

The strongest overall ideas are:

1. Causal Repo State Ledger
2. Fencing-Token File Ownership
3. Architect Arbitration Plane
4. Validation-Gated Delta Relay
5. Trust Zones and Policy Rings

### 7.2 Best balance of novelty and plausibility

[Synthesis]

Best balance:

- Causal Repo State Ledger
- Fencing-Token File Ownership
- Validation-Gated Delta Relay
- Campaign Awareness Memory Projection

These are novel enough to matter and plausible enough to ship.

### 7.3 Most feasible now

[Synthesis]

Most feasible now:

- Fencing-Token File Ownership

Reason:

- the code already exposes CRDT hooks, sync hooks, and policy hooks
- this is closer to completion than invention

### 7.4 Biggest long-term moat bet

[Synthesis]

Biggest long-term moat:

- Architect Arbitration Plane backed by Semantic Conflict Radar and Promotion Proof Packets

Reason:

- many tools can message multiple agents
- very few can arbitrate overlapping engineering work with semantic evidence and promotion discipline

### 7.5 Cleanest unification with current repo architecture

[Synthesis]

Cleanest unification:

- Causal Repo State Ledger plus Validation-Gated Delta Relay

Reason:

- they extend `saguaro/state/ledger.py`, `shared_kernel/event_store.py`, and `core/campaign/worktree_manager.py`
- they align with the repo’s existing control plane rather than splitting it

### 7.6 Best first prototype

[Synthesis]

Prototype first:

- Presence Mesh
- Causal Ledger
- Fencing Ownership

That prototype would already demonstrate:

- peer awareness
- safe claim semantics
- recoverable coordination

### 7.7 Conviction ranking

[Synthesis]

High conviction:

- Candidate 02. Causal Repo State Ledger
- Candidate 03. Fencing-Token File Ownership
- Candidate 07. Validation-Gated Delta Relay
- Candidate 16. Trust Zones and Policy Rings
- Candidate 09. Campaign Awareness Memory Projection

Medium conviction:

- Candidate 06. Architect Arbitration Plane
- Candidate 10. Semantic Conflict Radar
- Candidate 21. Virtual Workspace Union Branch
- Candidate 23. Promotion Proof Packets

Speculative but strategically interesting:

- Candidate 08. Phase Packet CRDT
- Candidate 14. Repo Digital Twin
- Candidate 22. Architect Council
- Candidate 24. Semantic Race Strategy Engine

---

## 8. Implementation Program

### Phase 1

- `phase_id`: `intake`
- Phase title: Repo Presence Mesh and Trust-Zone Bootstrap
- Objective: establish a shared presence plane so every Anvil instance can discover peers, expose capability and trust metadata, and participate in the same repo coordination protocol.
- Dependencies: none
- Repo scope: `core/networking/instance_identity.py`, `core/networking/peer_discovery.py`, `core/networking/peer_transport.py`, `core/ownership/file_ownership.py`, `cli/repl.py`, `cli/commands/agent.py`
- Owning specialist type: `distributed_runtime_engineer`
- Allowed writes: `core/networking/*.py`, `core/ownership/*.py`, `cli/repl.py`, `cli/commands/agent.py`, `tests/test_repo_presence.py`, `tests/test_transport_providers.py`, `tests/test_governance_trust_zones.py`
- Telemetry contract: publish `presence.heartbeat.sent`, `presence.peer.joined`, `transport.provider.selected`, `policy.zone.assigned`, and `policy.zone.denied`
- Required evidence: three instances in one repo show live phase, trust ring, and capability summary through CLI without manual coordination
- Rollback criteria: peer flapping, trust-zone false denies, or incompatible discovery semantics between LAN and broker mode
- Promotion gate: `pytest tests/test_repo_presence.py tests/test_transport_providers.py tests/test_governance_trust_zones.py tests/test_collaboration_commands.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- Success criteria: the repo has a coherent live instance map and unsafe actors are structurally separated before any write coordination occurs
- Exact wiring points: `cli/repl.py` REPL initialization, `core/networking/instance_identity.py` identity payloads, `core/networking/peer_discovery.py` provider selection, `core/networking/peer_transport.py` provider abstraction, `core/ownership/file_ownership.py` trust ring policy checks
- Deliverables: repo presence envelopes, transport providers, trust-zone policy model, enriched `/peers` command output
- Tests: `tests/test_repo_presence.py`, `tests/test_transport_providers.py`, `tests/test_governance_trust_zones.py`, `tests/test_collaboration_commands.py`
- Verification commands: `pytest tests/test_repo_presence.py tests/test_transport_providers.py tests/test_governance_trust_zones.py tests/test_collaboration_commands.py`, `./venv/bin/saguaro entrypoints`, `./venv/bin/saguaro build-graph`, `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- Exit criteria: presence and trust metadata become reliable enough to support authoritative coordination decisions

### Phase 2

- `phase_id`: `research`
- Phase title: Causal Repo Ledger and Coordination Memory
- Objective: promote Saguaro state ledger plus memory fabric into the authoritative causal model for workspace state, peer deltas, coordination history, and late-joiner hydration
- Dependencies: `intake`
- Repo scope: `saguaro/state/ledger.py`, `shared_kernel/event_store.py`, `core/memory/fabric/store.py`, `core/memory/fabric/retrieval_planner.py`, `core/memory/fabric/projectors.py`, `core/campaign/control_plane.py`
- Owning specialist type: `state_systems_engineer`
- Allowed writes: `saguaro/state/*.py`, `shared_kernel/*.py`, `core/memory/fabric/*.py`, `core/campaign/*.py`, `tests/test_state_ledger.py`, `tests/test_ledger_federation.py`, `tests/test_campaign_memory_projection.py`
- Telemetry contract: publish `ledger.watermark.advanced`, `ledger.peer.reconciled`, `memory.coord.ingested`, and `memory.coord.retrieved`
- Required evidence: a newly started instance can reconstruct current workspace, active claims, and current coordination posture from ledger plus memory
- Rollback criteria: replay divergence, duplicate events, or stale coordination recall
- Promotion gate: `pytest tests/test_state_ledger.py tests/test_ledger_federation.py tests/test_campaign_memory_projection.py tests/test_memory_snapshot_restore.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- Success criteria: every major coordination event can be replayed, queried, and retrieved as structured history
- Exact wiring points: `saguaro/state/ledger.py` for HLC-like watermarks and peer reconciliation, `shared_kernel/event_store.py` for event witness capture, `core/memory/fabric/store.py` for coordination objects, `core/campaign/control_plane.py` for ingest and retrieval
- Deliverables: causal watermark model, ledger federation API, campaign coordination memory schema, late-joiner hydration path
- Tests: `tests/test_state_ledger.py`, `tests/test_ledger_federation.py`, `tests/test_campaign_memory_projection.py`, `tests/test_memory_snapshot_restore.py`
- Verification commands: `pytest tests/test_state_ledger.py tests/test_ledger_federation.py tests/test_campaign_memory_projection.py tests/test_memory_snapshot_restore.py`, `./venv/bin/saguaro impact --path saguaro/state/ledger.py`, `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- Exit criteria: coordination state becomes causal, durable, and queryable enough to power ownership and architect decisions

### Phase 3

- `phase_id`: `feature_map`
- Phase title: Fenced Ownership Graph
- Objective: harden ownership from advisory file claims into fencing-token, lease-aware, and eventually symbol-aware concurrency control
- Dependencies: `intake`, `research`
- Repo scope: `core/ownership/file_ownership.py`, `core/ownership/ownership_models.py`, `core/subagent_communication.py`, `saguaro/state/ledger.py`, `domains/code_intelligence/saguaro_substrate.py`
- Owning specialist type: `concurrency_engineer`
- Allowed writes: `core/ownership/*.py`, `core/subagent_communication.py`, `saguaro/state/ledger.py`, `domains/code_intelligence/*.py`, `tests/test_file_ownership.py`, `tests/test_ownership_crdt.py`, `tests/test_ownership_fencing.py`, `tests/test_symbol_ownership.py`
- Telemetry contract: publish `ownership.claim.granted`, `ownership.fence.revoked`, `ownership.symbol.claimed`, and `ownership.lease.expired`
- Required evidence: stale claimants are fenced off, while safe disjoint symbol edits can proceed in the same file
- Rollback criteria: stale writers can still mutate repo state or symbol claims create unsafe false negatives
- Promotion gate: `pytest tests/test_file_ownership.py tests/test_ownership_crdt.py tests/test_ownership_fencing.py tests/test_symbol_ownership.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- Success criteria: ownership denials and grants are deterministic, explainable, and crash-tolerant
- Exact wiring points: `FileOwnershipRegistry.claim_files`, `FileOwnershipRegistry.can_access`, `MessageBus.publish` on ownership topics, `saguaro/state/ledger.py` claim epochs, semantic substrate for symbol resolution
- Deliverables: fencing tokens, lease epoch checks, symbol claim schema, ownership denial reason codes
- Tests: `tests/test_file_ownership.py`, `tests/test_ownership_crdt.py`, `tests/test_ownership_fencing.py`, `tests/test_symbol_ownership.py`
- Verification commands: `pytest tests/test_file_ownership.py tests/test_ownership_crdt.py tests/test_ownership_fencing.py tests/test_symbol_ownership.py`, `./venv/bin/saguaro impact --path core/ownership/file_ownership.py`, `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- Exit criteria: ownership is strong enough to stop write fights without freezing legitimate parallelism

### Phase 4

- `phase_id`: `roadmap_draft`
- Phase title: Architect Arbitration Kernel and Phase Packet Convergence
- Objective: add an elected architect layer that arbitrates overlapping plans, merges roadmap intent packets, and issues binding workload splits with traceable reasons
- Dependencies: `research`, `feature_map`
- Repo scope: `core/campaign/control_plane.py`, `core/campaign/runner.py`, `core/collaboration/task_announcer.py`, `core/collaboration/negotiation.py`, `core/campaign/roadmap_compiler.py`, `saguaro/roadmap/validator.py`
- Owning specialist type: `systems_architect`
- Allowed writes: `core/campaign/*.py`, `core/collaboration/*.py`, `core/architect/*.py`, `saguaro/roadmap/*.py`, `tests/test_architect_arbitration.py`, `tests/test_phase_packet_crdt.py`, `tests/test_campaign_roadmap_phase_pack.py`
- Telemetry contract: publish `architect.elected`, `architect.arbitration.started`, `intent.overlap.detected`, and `phase_packet.conflicted`
- Required evidence: when two instances generate incompatible plans, the architect emits one merged execution contract with changed ownership and handoff instructions
- Rollback criteria: split-brain architect leadership or validator-invalid merged phase packets
- Promotion gate: `pytest tests/test_architect_arbitration.py tests/test_phase_packet_crdt.py tests/test_campaign_roadmap_phase_pack.py tests/test_saguaro_roadmap_validator.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- Success criteria: plan collision becomes a solvable protocol event rather than a human rescue job
- Exact wiring points: `TaskAnnouncer.detect_overlap`, `CollaborationNegotiator.synthesize_plan`, `CampaignControlPlane.create_task_packet`, `RoadmapCompiler.render_phase_pack`, `saguaro/roadmap/validator.py`
- Deliverables: architect election, structured intent packets, merged phase packet store, arbitration journal
- Tests: `tests/test_architect_arbitration.py`, `tests/test_phase_packet_crdt.py`, `tests/test_campaign_roadmap_phase_pack.py`, `tests/test_saguaro_roadmap_validator.py`
- Verification commands: `pytest tests/test_architect_arbitration.py tests/test_phase_packet_crdt.py tests/test_campaign_roadmap_phase_pack.py tests/test_saguaro_roadmap_validator.py`, `./venv/bin/saguaro roadmap validate --path anvil_connectivity_roadmap.md --format json`, `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- Exit criteria: parallel roadmap generation and task planning become mergeable and architect-governed

### Phase 5

- `phase_id`: `development`
- Phase title: Validation-Gated Delta Relay and Virtual Workspace Union
- Objective: ship validated change capsules between instances, maintain a synthetic union workspace view, and distribute verified artifacts by content hash
- Dependencies: `feature_map`, `roadmap_draft`
- Repo scope: `core/campaign/worktree_manager.py`, `core/campaign/control_plane.py`, `domains/verification/verification_lane.py`, `shared_kernel/event_store.py`, `core/memory/fabric/store.py`, `saguaro/state/ledger.py`
- Owning specialist type: `promotion_pipeline_engineer`
- Allowed writes: `core/campaign/*.py`, `domains/verification/*.py`, `shared_kernel/*.py`, `core/memory/fabric/*.py`, `saguaro/state/*.py`, `tests/test_delta_distribution.py`, `tests/test_virtual_workspace_union.py`, `tests/test_validation_cas.py`, `tests/test_merge_queue.py`
- Telemetry contract: publish `delta.capsule.verified`, `delta.capsule.relayed`, `workspace.union.updated`, and `artifact.capsule.cache_hit`
- Required evidence: a delta that has passed verification on one instance hydrates on peers as a validated capsule, not as blind file copy
- Rollback criteria: hydration inconsistency, stale union manifest, or untrusted artifact reuse
- Promotion gate: `pytest tests/test_delta_distribution.py tests/test_virtual_workspace_union.py tests/test_validation_cas.py tests/test_merge_queue.py tests/test_campaign_timeline_and_verification_lane.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- Success criteria: peers rapidly converge on validated repo state without redoing identical work or stepping on stale branches
- Exact wiring points: `CampaignWorktreeManager.prepare`, `CampaignWorktreeManager.changed_files`, `CampaignWorktreeManager.promote`, `CampaignControlPlane.create_verification_lane`, `EventStore.export_run`, memory fabric artifact storage, ledger delta registration
- Deliverables: delta capsules, promotion proof hooks, virtual workspace union manifest, validation artifact CAS, merge queue
- Tests: `tests/test_delta_distribution.py`, `tests/test_virtual_workspace_union.py`, `tests/test_validation_cas.py`, `tests/test_merge_queue.py`, `tests/test_campaign_timeline_and_verification_lane.py`
- Verification commands: `pytest tests/test_delta_distribution.py tests/test_virtual_workspace_union.py tests/test_validation_cas.py tests/test_merge_queue.py tests/test_campaign_timeline_and_verification_lane.py`, `./venv/bin/saguaro impact --path core/campaign/worktree_manager.py`, `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- Exit criteria: validated work flows through the fleet with explicit proofs and reproducible workspace views

### Phase 6

- `phase_id`: `analysis_upgrade`
- Phase title: Semantic Conflict Radar and Repo Twin
- Objective: score interference between concurrent capsules, expose a repo digital twin, and feed architect decisions with semantic risk rather than lexical overlap
- Dependencies: `research`, `development`
- Repo scope: `domains/code_intelligence/saguaro_substrate.py`, `core/campaign/control_plane.py`, `core/campaign/telemetry.py`, `core/connectivity/repo_twin.py`, `saguaro/roadmap/validator.py`
- Owning specialist type: `program_analysis_engineer`
- Allowed writes: `domains/code_intelligence/*.py`, `core/campaign/*.py`, `core/connectivity/*.py`, `saguaro/roadmap/*.py`, `tests/test_semantic_conflict_radar.py`, `tests/test_repo_twin.py`, `tests/test_runtime_symbol_federation.py`
- Telemetry contract: publish `conflict.semantic.scored`, `conflict.semantic.blocked`, `repo_twin.hotspot.detected`, and `runtime.symbol.digest.published`
- Required evidence: architect decisions can explain not just that two tasks overlap, but why their concurrent promotion is behaviorally risky
- Rollback criteria: conflict scoring becomes noisy enough to block normal flow or the twin becomes misleadingly stale
- Promotion gate: `pytest tests/test_semantic_conflict_radar.py tests/test_repo_twin.py tests/test_runtime_symbol_federation.py tests/test_campaign_control_kernel.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- Success criteria: multi-instance scheduling and arbitration become semantically informed
- Exact wiring points: Saguaro substrate impact paths, control-plane risk maps, campaign telemetry aggregation, runtime symbol digest publication, roadmap validator graph summaries
- Deliverables: semantic conflict scorer, repo twin synthesizer, runtime capability digest, risk-aware architect hints
- Tests: `tests/test_semantic_conflict_radar.py`, `tests/test_repo_twin.py`, `tests/test_runtime_symbol_federation.py`, `tests/test_campaign_control_kernel.py`
- Verification commands: `pytest tests/test_semantic_conflict_radar.py tests/test_repo_twin.py tests/test_runtime_symbol_federation.py tests/test_campaign_control_kernel.py`, `./venv/bin/saguaro impact --path core/campaign/control_plane.py`, `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- Exit criteria: the system can distinguish harmless concurrency from dangerous semantic interference often enough to change scheduling policy

### Phase 7

- `phase_id`: `convergence`
- Phase title: Fleet-Scale Architecting, Chaos Recovery, and Promotion Proofs
- Objective: harden the whole stack for company-server and open-source scale through chaos drills, scoped architect leadership, promotion proof packets, and merge queue governance
- Dependencies: `development`, `analysis_upgrade`
- Repo scope: `core/architect/architect_plane.py`, `core/campaign/control_plane.py`, `shared_kernel/event_store.py`, `saguaro/state/ledger.py`, `domains/verification/verification_lane.py`, `saguaro/roadmap/validator.py`
- Owning specialist type: `federation_architect`
- Allowed writes: `core/architect/*.py`, `core/campaign/*.py`, `shared_kernel/*.py`, `saguaro/state/*.py`, `domains/verification/*.py`, `saguaro/roadmap/*.py`, `tests/test_connectivity_chaos.py`, `tests/test_architect_council.py`, `tests/test_promotion_proofs.py`
- Telemetry contract: publish `recovery.split_brain.detected`, `architect.scope.escalated`, `proof.packet.verified`, and `queue.item.promoted`
- Required evidence: the coordination layer survives leader death, stale claims, delayed peers, and protected-path contributions while preserving explainable promotion history
- Rollback criteria: chaos tests uncover unresolved split-brain or non-replayable promotion decisions
- Promotion gate: `pytest tests/test_connectivity_chaos.py tests/test_architect_council.py tests/test_promotion_proofs.py tests/test_merge_queue.py tests/test_state_ledger.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- Success criteria: the architecture is safe enough for company-server deployment and extensible enough for public-repo modes
- Exact wiring points: architect scopes and delegation, event-store replay, ledger failover semantics, verification lane proof attachments, roadmap validator completion graph linkage
- Deliverables: architect hierarchy, chaos harness, promotion proof packets, fleet-scale governance cookbook
- Tests: `tests/test_connectivity_chaos.py`, `tests/test_architect_council.py`, `tests/test_promotion_proofs.py`, `tests/test_merge_queue.py`, `tests/test_state_ledger.py`
- Verification commands: `pytest tests/test_connectivity_chaos.py tests/test_architect_council.py tests/test_promotion_proofs.py tests/test_merge_queue.py tests/test_state_ledger.py`, `./venv/bin/saguaro roadmap validate --path anvil_connectivity_roadmap.md --format json`, `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- Exit criteria: the system can coordinate parallel engineering work across local peers and company infrastructure without silent ownership corruption or promotion ambiguity

---

## 9. Implementation Contract

The following items are normative and intended to be traceable by roadmap validation tooling.

- The system shall implement repo-presence envelopes through `core/networking/instance_identity.py`, `core/networking/peer_discovery.py`, and `core/networking/peer_transport.py`, exposed in `cli/repl.py` and `cli/commands/agent.py`, tested by `tests/test_repo_presence.py`, `tests/test_transport_providers.py`, and `tests/test_collaboration_commands.py`, and verified with `pytest tests/test_repo_presence.py tests/test_transport_providers.py tests/test_collaboration_commands.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- The system shall enforce trust-zone-aware coordination through `core/ownership/file_ownership.py` and `core/campaign/control_plane.py`, tested by `tests/test_governance_trust_zones.py` and `tests/test_ownership_defaults.py`, and verified with `pytest tests/test_governance_trust_zones.py tests/test_ownership_defaults.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- The system shall extend `saguaro/state/ledger.py` and `shared_kernel/event_store.py` with causal watermark and peer-reconciliation semantics that can replay authoritative repo state, tested by `tests/test_state_ledger.py` and `tests/test_ledger_federation.py`, and verified with `pytest tests/test_state_ledger.py tests/test_ledger_federation.py` and `./venv/bin/saguaro impact --path saguaro/state/ledger.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- The system shall ingest coordination history into `core/memory/fabric/store.py`, `core/memory/fabric/retrieval_planner.py`, and `core/memory/fabric/projectors.py`, integrated via `core/campaign/control_plane.py`, tested by `tests/test_campaign_memory_projection.py` and `tests/test_memory_snapshot_restore.py`, and verified with `pytest tests/test_campaign_memory_projection.py tests/test_memory_snapshot_restore.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- The system shall harden file claims with fencing tokens and lease epochs through `core/ownership/file_ownership.py`, `core/ownership/ownership_models.py`, and `saguaro/state/ledger.py`, tested by `tests/test_file_ownership.py`, `tests/test_ownership_crdt.py`, and `tests/test_ownership_fencing.py`, and verified with `pytest tests/test_file_ownership.py tests/test_ownership_crdt.py tests/test_ownership_fencing.py` and `./venv/bin/saguaro impact --path core/ownership/file_ownership.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- The system shall implement symbol-aware claim arbitration through `core/ownership/file_ownership.py`, `domains/code_intelligence/saguaro_substrate.py`, and `saguaro/omnigraph` integration points, tested by `tests/test_symbol_ownership.py`, and verified with `pytest tests/test_symbol_ownership.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- The system shall replace lexical-only task overlap with structured intent packets through `core/collaboration/task_announcer.py`, `core/collaboration/negotiation.py`, and `core/campaign/control_plane.py`, tested by `tests/test_intent_graph.py` and `tests/test_collaboration_commands.py`, and verified with `pytest tests/test_intent_graph.py tests/test_collaboration_commands.py` and `./venv/bin/saguaro build-graph`.
- The system shall implement an elected architect arbitration kernel through `core/architect/architect_plane.py`, `core/campaign/control_plane.py`, `core/campaign/runner.py`, and `core/subagent_communication.py`, tested by `tests/test_architect_arbitration.py` and `tests/test_campaign_runner.py`, and verified with `pytest tests/test_architect_arbitration.py tests/test_campaign_runner.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- The system shall converge roadmap collaboration as structured phase packets through `core/campaign/roadmap_compiler.py`, `core/campaign/phase_packet.py`, and `saguaro/roadmap/validator.py`, tested by `tests/test_phase_packet_crdt.py`, `tests/test_campaign_roadmap_phase_pack.py`, and `tests/test_saguaro_roadmap_validator.py`, and verified with `pytest tests/test_phase_packet_crdt.py tests/test_campaign_roadmap_phase_pack.py tests/test_saguaro_roadmap_validator.py` and `./venv/bin/saguaro roadmap validate --path anvil_connectivity_roadmap.md --format json`.
- The system shall relay only verified delta capsules through `core/campaign/worktree_manager.py`, `core/campaign/control_plane.py`, `domains/verification/verification_lane.py`, `shared_kernel/event_store.py`, and `saguaro/state/ledger.py`, tested by `tests/test_delta_distribution.py`, `tests/test_merge_queue.py`, and `tests/test_campaign_timeline_and_verification_lane.py`, and verified with `pytest tests/test_delta_distribution.py tests/test_merge_queue.py tests/test_campaign_timeline_and_verification_lane.py` and `./venv/bin/saguaro impact --path core/campaign/worktree_manager.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- The system shall expose a synthetic union workspace view through `core/campaign/worktree_manager.py`, `saguaro/state/ledger.py`, and `core/campaign/control_plane.py`, tested by `tests/test_virtual_workspace_union.py`, and verified with `pytest tests/test_virtual_workspace_union.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- The system shall store reusable verification artifacts in a content-addressable capsule store through `shared_kernel/event_store.py`, `core/memory/fabric/store.py`, and `core/campaign/control_plane.py`, tested by `tests/test_validation_cas.py` and `tests/test_campaign_closure_safety_case.py`, and verified with `pytest tests/test_validation_cas.py tests/test_campaign_closure_safety_case.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- The system shall score semantic interference before promotion through `domains/code_intelligence/saguaro_substrate.py`, `core/campaign/control_plane.py`, and `core/collaboration/negotiation.py`, tested by `tests/test_semantic_conflict_radar.py` and `tests/test_campaign_control_kernel.py`, and verified with `pytest tests/test_semantic_conflict_radar.py tests/test_campaign_control_kernel.py` and `./venv/bin/saguaro impact --path core/campaign/control_plane.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- The system shall synthesize a repo digital twin through `core/connectivity/repo_twin.py`, `core/campaign/telemetry.py`, `saguaro/state/ledger.py`, and `saguaro/roadmap/validator.py`, tested by `tests/test_repo_twin.py` and `tests/test_saguaro_roadmap_validator.py`, and verified with `pytest tests/test_repo_twin.py tests/test_saguaro_roadmap_validator.py` and `./venv/bin/saguaro build-graph` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- The system shall publish runtime capability and symbol digests through `core/native/native_qsg_engine.py`, `core/networking/instance_identity.py`, and `saguaro/roadmap/validator.py`, tested by `tests/test_runtime_symbol_federation.py` and `tests/test_qsg_state_kernels.py`, and verified with `pytest tests/test_runtime_symbol_federation.py tests/test_qsg_state_kernels.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- The system shall inject live coordination posture into model execution through `core/prompts/system_prompt_builder.py`, `core/unified_chat_loop.py`, and `cli/repl.py`, tested by `tests/test_connectivity_prompt_context.py`, and verified with `pytest tests/test_connectivity_prompt_context.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- The system shall generate promotion proof packets through `saguaro/roadmap/validator.py`, `core/campaign/control_plane.py`, `shared_kernel/event_store.py`, and `domains/verification/verification_lane.py`, tested by `tests/test_promotion_proofs.py`, `tests/test_campaign_closure_safety_case.py`, and `tests/test_saguaro_roadmap_validator.py`, and verified with `pytest tests/test_promotion_proofs.py tests/test_campaign_closure_safety_case.py tests/test_saguaro_roadmap_validator.py` and `./venv/bin/saguaro roadmap validate --path anvil_connectivity_roadmap.md --format json`.
- The system shall survive leader death, stale leases, and replay recovery through `core/architect/architect_plane.py`, `saguaro/state/ledger.py`, `core/ownership/file_ownership.py`, and `shared_kernel/event_store.py`, tested by `tests/test_connectivity_chaos.py`, `tests/test_architect_council.py`, and `tests/test_state_ledger.py`, and verified with `pytest tests/test_connectivity_chaos.py tests/test_architect_council.py tests/test_state_ledger.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.

---

## 10. Reference Register

[External research]

- [R1] Ink & Switch, “Local-first software”  
  https://www.inkandswitch.com/local-first/
- [R2] Automerge Docs, “Welcome to Automerge”  
  https://automerge.org/docs/hello/
- [R3] Yjs Docs, “Introduction”  
  https://docs.yjs.dev/
- [R4] Figma Engineering, “How Figma’s multiplayer technology works”  
  https://www.figma.com/blog/how-figmas-multiplayer-technology-works/
- [R5] Linear Engineering, “Scaling the Linear Sync Engine”  
  https://linear.app/now/scaling-the-linear-sync-engine
- [R6] GitHub Docs, “About code owners”  
  https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners
- [R7] GitButler Docs, “Parallel Branches”  
  https://docs.gitbutler.com/features/branch-management/virtual-branches
- [R8] GitButler Docs, “Workspace Branch”  
  https://docs.gitbutler.com/workspace-branch
- [R9] SemanticMerge Docs, “Intro guide”  
  https://www.semanticmerge.com/documentation/intro-guide/semanticmerge-intro-guide
- [R10] Bazel Docs, “Remote Caching”  
  https://bazel.build/remote/caching
- [R11] Cockroach Labs, “Hybrid Logical Clock (HLC) Timestamps”  
  https://www.cockroachlabs.com/glossary/distributed-db/hybrid-logical-clock-hlc-timestamps/
- [R12] arXiv, “Verifying Semantic Conflict-Freedom in Three-Way Program Merges”  
  https://arxiv.org/abs/1802.06551
- [R13] Microsoft Research, “Verified Three-Way Program Merge”  
  https://www.microsoft.com/en-us/research/publication/verified-three-way-program-merge/
- [R14] arXiv, “Detecting Semantic Conflicts using Static Analysis”  
  https://arxiv.org/abs/2310.04269
- [R15] arXiv, “Program Merge Conflict Resolution via Neural Transformers”  
  https://arxiv.org/abs/2109.00084
- [R16] etcd concurrency package docs  
  https://pkg.go.dev/go.etcd.io/etcd/client/v3/concurrency
- [R17] Practitioner discussion on distributed ownership metadata  
  https://www.reddit.com/r/github/comments/16gc4o8
- [R18] Practitioner discussion on CRDT limits and intent mismatch  
  https://www.reddit.com/r/programming/comments/ydyqr4

---

## 11. Short Decision

[Synthesis]

Do not treat “connectivity” as chat sync.

Treat it as a federated repo control plane with:

- presence
- causal ledgering
- fencing ownership
- architect arbitration
- validation-gated delta relay
- semantic conflict scoring
- proof-oriented promotion

That is the architecture most aligned with the repo you already have, the failures you explicitly want to avoid, and the strongest external patterns that survive contact with real engineering systems.
