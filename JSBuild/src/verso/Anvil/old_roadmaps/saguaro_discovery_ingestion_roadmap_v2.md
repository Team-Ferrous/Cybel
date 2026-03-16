# Saguaro Discovery and Ingestion Roadmap V2

> Generated: 2026-03-13
> Workspace: `/home/mike/Documents/Github/Anvil`
> Standard: NASA and Formula 1 grade systems engineering
> Precondition: execute this after the comparative and deep-analysis foundation in `saguaro_deep_analysis_roadmap.md`
> Intent: upgrade Saguaro from a passive comparative-analysis substrate into an authenticated, budgeted, evidence-first discovery engine that can query the actual GitHub API, pull relevant repos into a controlled temporary analysis hangar, compare them against Anvil and Saguaro, and promote findings into campaign and roadmap artifacts

---

## 1. First-Principles Framing

### What the system fundamentally is

Saguaro plus Anvil is already a code-intelligence operating system with four live layers:

- `[Repo]` A code understanding substrate in `Saguaro/saguaro/*` with indexing, graph analysis, slicing, impact, runtime-symbol resolution, and roadmap validation.
- `[Repo]` A managed campaign and research runtime in `core/campaign/*`, `core/research/*`, and `core/dare/*`.
- `[Repo]` A telemetry and governance plane in `core/telemetry/*`, `core/native/runtime_telemetry.py`, `Saguaro/saguaro/roadmap/validator.py`, and `shared_kernel/event_store.py`.
- `[Repo]` A native-performance boundary in `Saguaro/src/*`, `Saguaro/CMakeLists.txt`, and `core/native/*`.

This means the right mental model is not "add a repo-search feature." The right mental model is "add a reconnaissance layer to an existing mission-control stack."

### What it appears to be trying to become

The repo is clearly leaning toward an autonomous engineering research loop:

- `[Repo]` `core/campaign/control_plane.py` already acquires repos, stores dossiers, runs research, drives EID, emits artifacts, and promotes roadmaps.
- `[Repo]` `core/research/repo_acquisition.py` already clones or snapshots repos and immediately builds structured analysis packs.
- `[Repo]` `Saguaro/saguaro/services/comparative.py` already creates isolated corpus sessions and generates cross-repo comparison reports.
- `[Repo]` `core/dare/web_research.py` already talks to `api.github.com/search/repositories`, but today it does so unauthenticated, shallowly ranked, and outside the stronger campaign control model.

From first principles, the system wants to become:

- an authenticated repo reconnaissance engine,
- a controlled acquisition hangar,
- a comparative evidence compiler,
- and an R&D planner that can justify why an external repo matters before any code is copied, adapted, or ignored.

### Direct answer to the user's core question

Yes. Saguaro can be set up so that after the deep-analysis roadmap is in place, it can query the actual GitHub API, discover repos similar to the codebase or the user's request, and pull them into a temporary `repo_analysis` workflow.

The strong version is not "clone straight into `repo_analysis/` and hope."

The strong version is:

- authoritative storage in the campaign cache via `core/campaign/repo_cache.py`,
- repo registration and write-policy control via `core/campaign/repo_registry.py`,
- a sessionized, read-only mirror under `repo_analysis/discovery_sessions/<campaign_id>/`,
- and isolated Saguaro corpus sessions created by `Saguaro/saguaro/services/comparative.py`.

That is the architectural center of gravity for this roadmap.

### Core constraints

These are the real constraints; everything else is secondary:

- `[Repo]` The primary corpus must remain clean. `repo_analysis/**` is already excluded by policy and shall stay excluded from the default index.
- `[Repo]` External repos must be immutable by default. Current repo roles and write policies already support that posture.
- `[Repo]` Discovery must use authenticated GitHub API access. Unauthenticated search is not acceptable as the primary production path.
- `[Repo]` Discovery must emit provenance, rate-limit, and ranking evidence, not just a list of URLs.
- `[Repo]` Any repo pulled for analysis must be traceable to a user request, a discovery query, or a follow-on frontier expansion event.
- `[Repo]` Compare before promote. No external code should become implementation input without a comparative report, arbitration posture, and promotion gate.
- `[Repo]` The CLI contract must work from the repo venv without fragile path assumptions. During this investigation, several Saguaro commands required `PYTHONPATH=.` because the raw `./venv/bin/saguaro` path otherwise raised `ModuleNotFoundError: No module named 'core'`.

### Engineering standard

- `[Inference]` NASA analogy: discovery is a reconnaissance vehicle, not mission control. It scouts, samples, and reports.
- `[Inference]` Formula 1 analogy: the discovery hangar is where outside parts are inspected, weighed, and wind-tunnel tested before touching the race car.
- `[Inference]` Cyber threat-intel analogy: every external repo needs provenance, trust, confidence, and retention metadata.

---

## 2. External Research Scan

### What I researched

Per request, the design space was widened through:

- `[External]` official GitHub REST, GraphQL, and code-search documentation,
- `[External]` Git and GitHub engineering material on partial clone and sparse checkout,
- `[External]` dependency-graph and package-intelligence sources such as Libraries.io and deps.dev,
- `[External]` provenance and historical source archives such as Software Heritage and GH Archive,
- `[External]` code-search and graph-retrieval systems such as Zoekt and GraphRAG,
- `[External]` repository-level reasoning and similarity papers such as RepoHyper, MetaSim, RANGER, and CodeRAG-Bench,
- `[External]` practitioner signal from current repository-graph MCP and code-context discussions.

### Most relevant external findings

#### 2.1 GitHub API realities

- `[External]` GitHub REST search is separately rate-limited and search results are not the same thing as a full repository inventory. It is appropriate for candidate generation, not authoritative completeness. [R1][R2]
- `[External]` GitHub GraphQL is the better fit for precision metadata fetches and bulk hydration after candidate generation because it reduces over-fetch and exposes a points-based rate model. [R3]
- `[External]` GitHub code search is useful as a targeted signal, but it only searches the default branch and applies content constraints, so it should be treated as one ranking signal rather than the discovery source of truth. [R4]
- `[External]` If we want "the actual GitHub API," the robust production choice is authenticated requests via a GitHub App installation token or a fine-grained PAT. Unauthenticated REST search is a degraded fallback path only. [R2][R5][R6]

Why it matters here:

- `[Synthesis]` The discovery engine shall separate candidate generation from candidate hydration.
- `[Synthesis]` Search endpoints are the scout. GraphQL and repo metadata endpoints are the pit wall telemetry feed.

#### 2.2 Authenticated access should be GitHub App first

- `[External]` GitHub Apps provide installation-scoped authentication and stronger operational boundaries than user PATs. [R6]
- `[External]` Fine-grained PATs are simpler to bootstrap and can be acceptable for single-user flows, but they are a weaker long-term control model than installation tokens. [R5]

Why it matters here:

- `[Synthesis]` The roadmap shall prefer a token broker that supports:
  - GitHub App installation tokens for campaign-safe production discovery,
  - fine-grained PATs for developer-local discovery,
  - explicit degraded mode for unauthenticated search only when the authenticated path is unavailable and the reason is logged.

#### 2.3 Bandwidth and storage discipline are mandatory

- `[External]` Blobless partial clone and sparse checkout materially change the economics of repo ingestion and are built for exactly this kind of large-repo selective analysis. [R7][R8][R9]

Why it matters here:

- `[Synthesis]` `core/campaign/repo_cache.py` currently does `git clone --depth 1` only.
- `[Synthesis]` That is not sufficient for a fleet-scale discovery engine.
- `[Synthesis]` The acquisition hangar shall use partial clone plus selective materialization by default.

#### 2.4 Dependency signals widen the design space beyond keyword search

- `[External]` Libraries.io and deps.dev provide package and dependency intelligence that GitHub search alone does not. [R10][R11]
- `[External]` GH Archive exposes ecosystem motion over time, which matters for maintenance quality, star velocity, and active evolution rather than static popularity. [R12]

Why it matters here:

- `[Synthesis]` Similar repositories are not just "same keywords" repositories.
- `[Synthesis]` Similar repositories can be:
  - sharing the same dependency ecosystem,
  - solving the same performance problem,
  - being co-mentioned in issue and PR ecosystems,
  - or sitting near the same maintenance and adoption frontier.

#### 2.5 Provenance and historical identity can become moat

- `[External]` Software Heritage and SWHID-style provenance show how code identity and lineage can survive across forks and mirrors. [R13]

Why it matters here:

- `[Synthesis]` Long term, Saguaro should not just know "where a repo came from."
- `[Synthesis]` It should know lineage, fork ancestry, archive fingerprints, and whether a candidate is a trivial fork versus a meaningfully divergent implementation.

#### 2.6 Lexical prefilter plus semantic rerank is still the right shape

- `[External]` Zoekt remains a strong example of very fast lexical prefiltering over massive code inventories. [R14]
- `[External]` GraphRAG shows the value of graph-structured retrieval once candidate context is narrowed. [R15]

Why it matters here:

- `[Synthesis]` The right discovery shape is:
  - cheap lexical or metadata scout,
  - multi-signal candidate ranker,
  - graph-aware rerank for the finalists,
  - then acquisition.

#### 2.7 Repo-level reasoning research is converging on graph-plus-context

- `[External]` RepoHyper and related repository-level reasoning work push toward graph-aware retrieval over a repository's structure, not flat chunk stuffing. [R16]
- `[External]` CodeRAG-Bench and RANGER are useful not because they hand us a turnkey implementation, but because they reinforce that repository tasks collapse when retrieval and structure are weak. [R17][R18]
- `[External]` MetaSim is a useful reminder that repository similarity is multi-factor and should not be reduced to stars or topic labels. [R19]

Why it matters here:

- `[Synthesis]` Saguaro already has comparative sessions and repo dossiers.
- `[Synthesis]` The next step is not to imitate generic repo RAG.
- `[Synthesis]` The next step is to fuse ranking, acquisition, and comparative analysis into one operational loop.

#### 2.8 Practitioner signal

- `[External]` Recent practitioner interest around code-graph MCP and repository-context servers suggests that graph-shaped code context is becoming operationally relevant, not purely academic. [R20]

Why it matters here:

- `[Synthesis]` Anvil should not stop at "search GitHub and clone."
- `[Synthesis]` It should aim for a graph-native discovery and comparison pipeline while the field is still early.

### Why this changes the design space

- `[Synthesis]` The discovery engine should be GitHub-App capable from day one.
- `[Synthesis]` The repo cache should become a budgeted acquisition hangar rather than a simple clone directory.
- `[Synthesis]` `repo_analysis/` should become a mirrored session surface, not the source of truth.
- `[Synthesis]` The strongest moat is not API wiring. It is authenticated discovery plus provenance plus comparative evidence plus promotion control.

### Reference index

- `[R1]` GitHub REST search docs: <https://docs.github.com/en/rest/search/search#about-search>
- `[R2]` GitHub REST rate limits: <https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api>
- `[R3]` GitHub GraphQL rate limits and query limits: <https://docs.github.com/en/graphql/overview/rate-limits-and-query-limits-for-the-graphql-api>
- `[R4]` GitHub code search syntax: <https://docs.github.com/en/search-github/github-code-search/understanding-github-code-search-syntax>
- `[R5]` Fine-grained PAT docs: <https://docs.github.com/en/github/authenticating-to-github/creating-a-personal-access-token>
- `[R6]` GitHub App authentication docs: <https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app>
- `[R7]` GitHub blog on partial clone and shallow clone: <https://github.blog/open-source/git/get-up-to-speed-with-partial-clone-and-shallow-clone/>
- `[R8]` `git clone` docs: <https://git-scm.com/docs/git-clone>
- `[R9]` `git sparse-checkout` docs: <https://git-scm.com/docs/git-sparse-checkout>
- `[R10]` Libraries.io API: <https://libraries.io/api>
- `[R11]` deps.dev API docs: <https://docs.deps.dev/api/>
- `[R12]` GH Archive: <https://gharchive.org/>
- `[R13]` Software Heritage graph docs: <https://docs.softwareheritage.org/devel/swh-graph/>
- `[R14]` Zoekt repository: <https://github.com/sourcegraph/zoekt>
- `[R15]` GraphRAG: <https://microsoft.github.io/graphrag/>
- `[R16]` RepoHyper: <https://arxiv.org/abs/2503.09089>
- `[R17]` CodeRAG-Bench: <https://github.com/kiliczsh/coderag-bench>
- `[R18]` RANGER: <https://arxiv.org/html/2402.10009v1>
- `[R19]` MetaSim: <https://2024.esec-fse.org/details/fse-2024-research-papers/87/MetaSim-Using-Metadata-and-Text-Semantics-for-Repository-Similarity-Computation>
- `[R20]` CodeGraphContext discussion: <https://www.reddit.com/r/mcp/comments/1lolk31/codegraphcontext_repo_to_symbol_graph/>

---

## 3. Repo Grounding Summary

### Commands executed from the repo venv

All repo inspection was run from the repository venv after `source venv/bin/activate`.

Executed commands:

```bash
./venv/bin/saguaro health
PYTHONPATH=. ./venv/bin/saguaro entrypoints
PYTHONPATH=. ./venv/bin/saguaro build-graph
PYTHONPATH=. ./venv/bin/saguaro verify . --engines native,ruff,semantic --format json
PYTHONPATH=. ./venv/bin/saguaro impact --path core/campaign/control_plane.py
PYTHONPATH=. ./venv/bin/saguaro impact --path core/dare/web_research.py
PYTHONPATH=. ./venv/bin/saguaro query "prompt assembly orchestration planning loops" --k 5
PYTHONPATH=. ./venv/bin/saguaro query "memory systems telemetry runtime signals governance verification" --k 5
PYTHONPATH=. ./venv/bin/saguaro query "github repository analysis ingestion workspace comparative analysis" --k 8
PYTHONPATH=. ./venv/bin/saguaro agent skeleton core/dare/repo_ingestion.py
PYTHONPATH=. ./venv/bin/saguaro agent skeleton core/campaign/workspace.py
PYTHONPATH=. ./venv/bin/saguaro agent skeleton core/campaign/repo_registry.py
PYTHONPATH=. ./venv/bin/saguaro agent skeleton core/campaign/retrieval_policy.py
PYTHONPATH=. ./venv/bin/saguaro agent skeleton core/campaign/runner.py
PYTHONPATH=. ./venv/bin/saguaro agent skeleton core/research/analysis_pack.py
PYTHONPATH=. ./venv/bin/saguaro agent skeleton core/unified_chat_loop.py
PYTHONPATH=. ./venv/bin/saguaro agent skeleton Saguaro/saguaro/services/comparative.py
PYTHONPATH=. ./venv/bin/saguaro agent slice RepoIngestionEngine._analyze_single_repo --depth 2
PYTHONPATH=. ./venv/bin/saguaro agent slice CampaignRepoRegistry.register_repo --depth 2
PYTHONPATH=. ./venv/bin/saguaro agent slice RetrievalPolicyEngine.decide --depth 2
```

### Observed command behavior

- `[Repo]` `./venv/bin/saguaro health` worked directly and reported a healthy index, healthy native ABI, active query gateway, and high graph coverage.
- `[Repo]` `./venv/bin/saguaro entrypoints`, `build-graph`, and `verify` initially failed without `PYTHONPATH=.`, raising `ModuleNotFoundError: No module named 'core'` from `Saguaro/saguaro/services/comparative.py`.
- `[Repo]` With `PYTHONPATH=.`, `entrypoints`, `build-graph`, `impact`, `query`, and `verify` succeeded.
- `[Repo]` `PYTHONPATH=. ./venv/bin/saguaro verify . --engines native,ruff,semantic --format json` returned `status: pass` with no violations and high confidence posture.

### Code paths inspected and why they matter

#### 3.1 Acquisition and repo-control spine

- `[Repo]` `core/campaign/control_plane.py`
  - Real center of gravity.
  - Already owns `attach_repo()`, `acquire_repos()`, `run_research()`, `build_feature_map()`, `build_roadmap()`, `promote_final_roadmap()`, and `ensure_repo_dossier_brief()`.
  - The discovery engine should feed this file, not bypass it.

- `[Repo]` `core/campaign/repo_cache.py`
  - Already provides immutable local snapshots and remote clone support.
  - Today it uses `git clone --depth 1` only.
  - This is the correct place to add partial clone, sparse checkout, and discovery cache metadata.

- `[Repo]` `core/campaign/repo_registry.py`
  - Already encodes repo roles and deterministic write policies.
  - `analysis_external` and `analysis_local` are first-class roles already.
  - This should become the authoritative repo policy ledger for discovery-acquired repos.

- `[Repo]` `core/campaign/workspace.py`
  - Already exposes repo family directories, record directories, and repo cache directories.
  - This is the right substrate for a sessionized `repo_analysis` mirror.

#### 3.2 Existing comparative substrate

- `[Repo]` `Saguaro/saguaro/services/comparative.py`
  - Already creates isolated corpus sessions with TTL, quarantine, trust level, and manifest storage.
  - Already exposes `create_session()`, `corpus_query()`, `slice_symbol()`, and `compare()`.
  - This is not hypothetical. The comparative backbone already exists.

- `[Repo]` `core/research/analysis_pack.py`
  - Already emits `repo_dossier.v1`, build fingerprints, capability matrices, semantic inventory, reuse candidates, and risk signals.
  - This is a strong primitive to reuse, not a thing to replace.

- `[Repo]` `core/dare/repo_ingestion.py`
  - Already turns a local repo into a `RepoProfile` and writes analysis knowledge.
  - `_cross_repo_analysis()` already tries to call `ComparativeAnalysisService`.
  - Discovery should widen and harden this path rather than invent a second ingestion engine.

#### 3.3 Retrieval and research routing

- `[Repo]` `core/campaign/retrieval_policy.py`
  - Already encodes a Saguaro-first retrieval budget router with telemetry emission and repo dossier reuse logic.
  - This is exactly where authenticated GitHub fallback should become explicit and inspectable.

- `[Repo]` `core/campaign/control_plane.py:run_research()`
  - Already records retrieval routes, utility summaries, frontier candidates, and stop-proof artifacts.
  - This is the strongest landing zone for authenticated external discovery.

- `[Repo]` `core/telemetry/black_box.py`
  - Already records run events, tool plans, tool results, verification results, and performance snapshots.
  - Discovery and acquisition telemetry should plug into this rather than inventing an ad hoc log format.

#### 3.4 Existing GitHub path that needs upgrading

- `[Repo]` `core/dare/web_research.py`
  - Already contains `GitHubSearchSource`.
  - It currently calls `https://api.github.com/search/repositories` with `requests.get(...)` and no auth.
  - It also clones top-ranked GitHub repos and deep-ingests them locally.
  - This is the seed of the requested capability, but it is not strong enough yet.

#### 3.5 Prompt, loop, and user-facing integration points

- `[Repo]` `core/unified_chat_loop.py`
  - Huge orchestration surface with research delegation, repo-analysis delegation, evidence capture, and governance checkpoints.
  - Discovery results should become structured evidence here, not loose notes.

- `[Repo]` `core/prompts/prompt_manager.py`
  - Already constructs master prompt contracts and runtime contracts.
  - Comparative discovery mode should surface here as an explicit contract, not hidden lore.

### Existing primitives that are strongest

- `[Repo]` Auth-adjacent environment precedent already exists in `config/mcp_servers.yaml` through `GITHUB_TOKEN`.
- `[Repo]` Comparative corpora already exist in `Saguaro/saguaro/services/comparative.py`.
- `[Repo]` Repo acquisition and dossier generation already exist in `core/research/repo_acquisition.py` and `core/research/analysis_pack.py`.
- `[Repo]` Campaign and research routing already exist in `core/campaign/control_plane.py`.
- `[Repo]` Retrieval budget telemetry already exists in `core/campaign/retrieval_policy.py`.
- `[Repo]` Flight-recorder style telemetry already exists in `core/telemetry/black_box.py`.

### Where the architecture is strongest

- `[Repo]` There is already a campaign-safe place to register and isolate external repos.
- `[Repo]` There is already a comparative service with TTL and quarantine.
- `[Repo]` There is already a repo dossier abstraction.
- `[Repo]` There is already a retrieval and stop-proof loop.

### Where it is underexploited or thin

- `[Repo]` GitHub access is unauthenticated and not treated as a first-class subsystem.
- `[Repo]` Acquisition lacks partial clone, sparse checkout, and rate-limit awareness.
- `[Repo]` Discovery provenance is not first-class.
- `[Repo]` `repo_analysis/` is currently a loose folder, not a governed session mirror.
- `[Repo]` The Saguaro CLI bootstrap path is too fragile for a discovery-heavy workflow if `PYTHONPATH=.` remains required.

---

## 4. Hidden Assumptions

1. `[Inference]` Keyword search is enough to find relevant repos.
2. `[Inference]` Repo popularity is a good proxy for repo relevance.
3. `[Inference]` Repos should be cloned directly into `repo_analysis/` as the source of truth.
4. `[Inference]` Unauthenticated GitHub REST search is acceptable for production discovery.
5. `[Inference]` The best discovery source is GitHub search alone rather than a fusion of GitHub metadata, dependencies, and comparative signals.
6. `[Inference]` Every discovered repo deserves a full clone and deep analysis.
7. `[Inference]` Search results do not need strong provenance because the user can inspect them manually.
8. `[Inference]` Rate limits and token scopes are an implementation detail rather than a control-plane concern.
9. `[Inference]` Discovery can live inside DARE alone instead of becoming a first-class campaign primitive.
10. `[Inference]` A temporary analysis repo is harmless even if its lifecycle is unmanaged.

---

## 5. Candidate Implementation Phases

The set below mixes practical work and moonshots with an approximate 70/30 split.

### C01. Authenticated GitHub Access Layer

- Suggested `phase_id`: `intake`
- Class: `practical`
- Core insight: discovery shall use the actual GitHub API with authentication, not the current unauthenticated `requests.get(...)` call in `core/dare/web_research.py`.
- External inspiration or analogy: GitHub App installation tokens and scoped API governance. [R2][R5][R6]
- Why it fits Saguaro and Anvil specifically: `core/dare/web_research.py` already proves the value path; it just needs to be made real and governed.
- Exact places in this codebase where it could wire in: `core/dare/web_research.py`, `core/campaign/control_plane.py`, `config/mcp_servers.yaml`, `Saguaro/saguaro/discovery/github_client.py`
- Existing primitives it can reuse: campaign state store, event store, repo acquisition, env var injection precedent
- New primitive, data flow, or subsystem needed: authenticated GitHub client with App-token and fine-grained PAT modes
- `repo_scope`: `["target", "analysis_external", "artifact_store"]`
- `owning_specialist_type`: `["discovery-engineering"]`
- `allowed_writes`: `["core/dare/web_research.py", "core/campaign/control_plane.py", "config/mcp_servers.yaml", "Saguaro/saguaro/discovery/github_client.py"]`
- `telemetry_contract`: `{"minimum": ["github_auth_mode", "github_api_calls", "github_rate_limit_remaining", "github_secondary_limit_events"]}`
- `required_evidence`: `["authenticated search transcript", "rate-limit header capture", "token-mode audit"]`
- `rollback_criteria`: `["token leakage risk", "undocumented degraded mode", "search succeeds only unauthenticated"]`
- `promotion_gate`: authenticated search works with GitHub App or PAT and records provenance
- `success_criteria`: no production discovery path depends on unauthenticated GitHub search
- Why this creates value: it turns the request into a real system contract
- Why this creates moat: operationally correct auth and budget control are harder than raw API usage
- Main risk or failure mode: secret handling is bolted on instead of designed
- Smallest credible first experiment: implement authenticated repository search for one query and verify with `tests/test_dare_web_research.py`
- Confidence level: high

### C02. GitHub Token Broker and Secret Policy

- Suggested `phase_id`: `intake`
- Class: `practical`
- Core insight: the discovery engine shall not read ad hoc env vars from random code paths; it needs one token broker and one policy surface.
- External inspiration or analogy: mission-control credential staging
- Why it fits Saguaro and Anvil specifically: campaign control already centralizes other operational decisions
- Exact places in this codebase where it could wire in: `core/campaign/control_plane.py`, `core/campaign/state_store.py`, `Saguaro/saguaro/discovery/github_client.py`, `config/mcp_servers.yaml`
- Existing primitives it can reuse: state store, metadata capture, existing `GITHUB_TOKEN` env reference
- New primitive, data flow, or subsystem needed: broker that resolves `GITHUB_APP_ID`, `GITHUB_APP_PRIVATE_KEY`, `GITHUB_INSTALLATION_ID`, and `GITHUB_TOKEN`
- `repo_scope`: `["artifact_store"]`
- `owning_specialist_type`: `["security-platform"]`
- `allowed_writes`: `["core/campaign/control_plane.py", "core/campaign/state_store.py", "Saguaro/saguaro/discovery/github_client.py", "config/mcp_servers.yaml"]`
- `telemetry_contract`: `{"minimum": ["github_auth_mode", "github_token_source", "secret_resolution_failures"]}`
- `required_evidence`: `["secret policy doc", "redacted telemetry sample", "token resolution tests"]`
- `rollback_criteria`: `["token material persisted to artifacts", "fallback to plaintext config"]`
- `promotion_gate`: auth mode is inspectable without exposing secrets
- `success_criteria`: campaign logs know whether discovery used App, PAT, or degraded mode
- Why this creates value: safe auth becomes operable
- Why this creates moat: disciplined secret handling is part of enterprise viability
- Main risk or failure mode: App flow complexity delays initial adoption
- Smallest credible first experiment: PAT-only broker first, App path immediately scaffolded
- Confidence level: high

### C03. Discovery Provenance Envelope

- Suggested `phase_id`: `research`
- Class: `practical`
- Core insight: every discovered repo needs a signed-style evidence envelope containing query, signal weights, API source, rate-limit state, and acquisition reason.
- External inspiration or analogy: threat-intel provenance packets
- Why it fits Saguaro and Anvil specifically: `AnalysisPackBuilder` already emits evidence envelopes for repo dossiers
- Exact places in this codebase where it could wire in: `core/research/analysis_pack.py`, `core/campaign/control_plane.py`, `core/campaign/repo_registry.py`, `Saguaro/saguaro/discovery/provenance.py`
- Existing primitives it can reuse: dossier evidence envelope, artifact publication, state store JSON rows
- New primitive, data flow, or subsystem needed: `discovery_candidate.v1` and `discovery_envelope.v1`
- `repo_scope`: `["analysis_external", "artifact_store"]`
- `owning_specialist_type`: `["evidence-governance"]`
- `allowed_writes`: `["core/research/analysis_pack.py", "core/campaign/control_plane.py", "core/campaign/repo_registry.py", "Saguaro/saguaro/discovery/provenance.py"]`
- `telemetry_contract`: `{"minimum": ["discovery_candidate_count", "provenance_missing_count", "candidate_to_acquisition_ratio"]}`
- `required_evidence`: `["candidate envelope examples", "artifact linkage", "promotion trace"]`
- `rollback_criteria`: `["candidate records exist without provenance", "provenance fields drift across code paths"]`
- `promotion_gate`: every acquired repo has a linked discovery envelope
- `success_criteria`: no repo enters analysis without a why
- Why this creates value: discovery becomes auditable
- Why this creates moat: provenance compounds into long-term trust and learning
- Main risk or failure mode: provenance is collected but not consumed
- Smallest credible first experiment: attach one discovery envelope to one acquired repo
- Confidence level: high

### C04. Multi-Signal Candidate Ranker

- Suggested `phase_id`: `research`
- Class: `practical`
- Core insight: rank by a blend of text match, dependency overlap, language fit, activity, structural clues, and request alignment.
- External inspiration or analogy: MetaSim and repo similarity research. [R19]
- Why it fits Saguaro and Anvil specifically: `run_research()` already computes frontier utilities and stop proofs
- Exact places in this codebase where it could wire in: `core/campaign/control_plane.py`, `core/campaign/retrieval_policy.py`, `Saguaro/saguaro/discovery/ranker.py`, `core/dare/web_research.py`
- Existing primitives it can reuse: retrieval utility context, repo dossier features, analysis pack fingerprints
- New primitive, data flow, or subsystem needed: weighted ranker with per-signal explanation
- `repo_scope`: `["analysis_external", "artifact_store"]`
- `owning_specialist_type`: `["discovery-intelligence"]`
- `allowed_writes`: `["core/campaign/control_plane.py", "core/campaign/retrieval_policy.py", "Saguaro/saguaro/discovery/ranker.py", "core/dare/web_research.py"]`
- `telemetry_contract`: `{"minimum": ["ranker_signal_breakdown", "top_k_acceptance_rate", "ranker_precision_review_score"]}`
- `required_evidence`: `["ranked candidate list", "signal audit", "manual relevance review"]`
- `rollback_criteria`: `["ranker explanation missing", "top-k quality below baseline"]`
- `promotion_gate`: top 10 contains materially better repos than search-by-stars alone
- `success_criteria`: compute is focused on strong candidates
- Why this creates value: better discovery quality with the same API budget
- Why this creates moat: ranking intelligence is harder to copy than API wiring
- Main risk or failure mode: no labeled relevance set
- Smallest credible first experiment: text plus stars plus language plus dependency overlap
- Confidence level: high

### C05. Dependency and Package Signal Ingestion

- Suggested `phase_id`: `research`
- Class: `practical`
- Core insight: discovery should consider shared package ecosystems and dependency neighborhoods, not just repo text.
- External inspiration or analogy: Libraries.io and deps.dev. [R10][R11]
- Why it fits Saguaro and Anvil specifically: `AnalysisPackBuilder` already extracts build files and tech stack
- Exact places in this codebase where it could wire in: `core/research/analysis_pack.py`, `Saguaro/saguaro/discovery/dependency_sources.py`, `core/campaign/control_plane.py`
- Existing primitives it can reuse: tech stack, build fingerprint, repo dossier
- New primitive, data flow, or subsystem needed: dependency-signal fetcher for package ecosystems
- `repo_scope`: `["analysis_external", "artifact_store"]`
- `owning_specialist_type`: `["dependency-intelligence"]`
- `allowed_writes`: `["core/research/analysis_pack.py", "Saguaro/saguaro/discovery/dependency_sources.py", "core/campaign/control_plane.py"]`
- `telemetry_contract`: `{"minimum": ["dependency_signal_hits", "ecosystem_overlap_score", "dependency_api_calls"]}`
- `required_evidence`: `["dependency overlap reports", "ecosystem mapping examples"]`
- `rollback_criteria`: `["dependency signals overfit package names", "external API budget spikes"]`
- `promotion_gate`: dependency-aware ranking changes at least one top-k result materially
- `success_criteria`: similar repos can be found even when terminology differs
- Why this creates value: it widens the discovery aperture
- Why this creates moat: ecosystem intelligence is underused in code discovery
- Main risk or failure mode: dependency names are noisy proxies for real mechanism fit
- Smallest credible first experiment: Python and CMake ecosystems only
- Confidence level: medium-high

### C06. Discovery Hangar and `repo_analysis` Mirror

- Suggested `phase_id`: `research`
- Class: `practical`
- Core insight: authoritative acquisition shall live in the campaign cache; `repo_analysis/` shall become a readable mirror of selected sessions.
- External inspiration or analogy: aircraft hangar plus flight-line presentation
- Why it fits Saguaro and Anvil specifically: `CampaignWorkspace.repo_cache_dir()` and repo-family directories already exist
- Exact places in this codebase where it could wire in: `core/campaign/workspace.py`, `core/campaign/repo_cache.py`, `core/campaign/repo_registry.py`, `Saguaro/saguaro/discovery/session_mirror.py`
- Existing primitives it can reuse: repo cache, record dirs, repo roles, immutable snapshots
- New primitive, data flow, or subsystem needed: mirror manifest under `repo_analysis/discovery_sessions/<campaign_id>/`
- `repo_scope`: `["analysis_external"]`
- `owning_specialist_type`: `["workspace-operations"]`
- `allowed_writes`: `["core/campaign/workspace.py", "core/campaign/repo_cache.py", "core/campaign/repo_registry.py", "Saguaro/saguaro/discovery/session_mirror.py"]`
- `telemetry_contract`: `{"minimum": ["mirrored_repo_count", "mirror_refresh_ms", "mirror_staleness_seconds"]}`
- `required_evidence`: `["mirror manifest", "authoritative-cache to mirror mapping", "cleanup log"]`
- `rollback_criteria`: `["repo_analysis becomes source of truth", "mirror drifts from cache without detection"]`
- `promotion_gate`: mirrored session appears in `repo_analysis` with authoritative record linkage
- `success_criteria`: user can inspect temp repos in `repo_analysis` without compromising control
- Why this creates value: it answers the exact requested workflow
- Why this creates moat: controlled human-readable mirrors are better than raw hidden caches
- Main risk or failure mode: duplicate storage and cleanup drift
- Smallest credible first experiment: symlink or manifest-only mirror for one session
- Confidence level: high

### C07. Partial Clone and Sparse Materialization

- Suggested `phase_id`: `research`
- Class: `practical`
- Core insight: discovery acquisition should start blobless and materialize only what ranking or analysis needs.
- External inspiration or analogy: Git partial clone and sparse checkout. [R7][R8][R9]
- Why it fits Saguaro and Anvil specifically: `RepoCache.clone_remote_repo()` is a small, central choke point
- Exact places in this codebase where it could wire in: `core/campaign/repo_cache.py`, `core/research/repo_acquisition.py`, `Saguaro/saguaro/discovery/github_client.py`
- Existing primitives it can reuse: repo cache, snapshot logic, revision fingerprinting
- New primitive, data flow, or subsystem needed: acquisition modes `blobless`, `sparse`, `full`
- `repo_scope`: `["analysis_external"]`
- `owning_specialist_type`: `["repo-acquisition"]`
- `allowed_writes`: `["core/campaign/repo_cache.py", "core/research/repo_acquisition.py", "Saguaro/saguaro/discovery/github_client.py"]`
- `telemetry_contract`: `{"minimum": ["clone_mode", "clone_wall_ms", "materialized_bytes", "checkout_expansion_count"]}`
- `required_evidence`: `["before/after bandwidth comparison", "clone mode traces", "materialization logs"]`
- `rollback_criteria`: `["partial clone breaks downstream analysis", "unexpected full fetches"]`
- `promotion_gate`: blobless acquisition reduces initial bytes without preventing analysis-pack generation
- `success_criteria`: discovery can scale beyond a handful of full clones
- Why this creates value: cost and speed control
- Why this creates moat: operational scaling is where many discovery systems fail
- Main risk or failure mode: tools silently trigger full checkout
- Smallest credible first experiment: blobless clone plus selective checkout of build files and top candidate paths
- Confidence level: high

### C08. License, Trust, and Quarantine Gate

- Suggested `phase_id`: `research`
- Class: `practical`
- Core insight: discovery needs a pre-ingestion gate for license, fork status, archived status, and trust posture.
- External inspiration or analogy: supply-chain intake inspection
- Why it fits Saguaro and Anvil specifically: repo roles and write policies already exist
- Exact places in this codebase where it could wire in: `core/campaign/repo_registry.py`, `core/campaign/control_plane.py`, `Saguaro/saguaro/discovery/github_client.py`, `Saguaro/saguaro/services/comparative.py`
- Existing primitives it can reuse: trust_level, quarantine, repo role metadata
- New primitive, data flow, or subsystem needed: intake gate verdict record
- `repo_scope`: `["analysis_external", "artifact_store"]`
- `owning_specialist_type`: `["governance-intake"]`
- `allowed_writes`: `["core/campaign/repo_registry.py", "core/campaign/control_plane.py", "Saguaro/saguaro/discovery/github_client.py", "Saguaro/saguaro/services/comparative.py"]`
- `telemetry_contract`: `{"minimum": ["license_unknown_count", "archived_repo_count", "fork_filter_count", "trust_level_distribution"]}`
- `required_evidence`: `["gate verdicts", "blocked candidate examples"]`
- `rollback_criteria`: `["high-risk repos reach deep analysis without a gate", "gate blocks almost everything due to bad defaults"]`
- `promotion_gate`: candidate acquisition emits intake verdicts before clone or session creation
- `success_criteria`: risky repos are explicit, not silent
- Why this creates value: safer external ingestion
- Why this creates moat: governance-heavy discovery is enterprise-grade
- Main risk or failure mode: overly conservative filtering reduces useful discovery
- Smallest credible first experiment: block archived repos and flag fork-heavy results
- Confidence level: high

### C09. Authenticated GraphQL Hydration Plane

- Suggested `phase_id`: `research`
- Class: `practical`
- Core insight: REST search should seed candidates, but GraphQL should hydrate the finalist set with exact metadata, topics, branches, and rich fields.
- External inspiration or analogy: scout car plus telemetry truck
- Why it fits Saguaro and Anvil specifically: the discovery stack needs a two-stage design to stay within rate limits
- Exact places in this codebase where it could wire in: `Saguaro/saguaro/discovery/github_client.py`, `core/campaign/control_plane.py`
- Existing primitives it can reuse: ranker, state store, discovery provenance envelope
- New primitive, data flow, or subsystem needed: candidate hydration batcher
- `repo_scope`: `["analysis_external", "artifact_store"]`
- `owning_specialist_type`: `["api-orchestration"]`
- `allowed_writes`: `["Saguaro/saguaro/discovery/github_client.py", "core/campaign/control_plane.py"]`
- `telemetry_contract`: `{"minimum": ["graphql_points_used", "hydrated_candidate_count", "hydration_latency_ms"]}`
- `required_evidence`: `["REST seed vs GraphQL hydrated payload comparison"]`
- `rollback_criteria`: `["hydration cost outweighs value", "GraphQL queries become brittle"]`
- `promotion_gate`: top candidates are hydrated without redundant endpoint spray
- `success_criteria`: discovery records contain rich metadata with bounded API cost
- Why this creates value: precision without waste
- Why this creates moat: correct API choreography matters at scale
- Main risk or failure mode: GraphQL query sprawl
- Smallest credible first experiment: hydrate top 10 REST results only
- Confidence level: high

### C10. Shallow Fingerprint Gate

- Suggested `phase_id`: `analysis_upgrade`
- Class: `practical`
- Core insight: many repos can be rejected after a lightweight build fingerprint, capability matrix, and semantic inventory pass.
- External inspiration or analogy: pit-lane weight and aero check before full teardown
- Why it fits Saguaro and Anvil specifically: `AnalysisPackBuilder` already emits exactly the raw ingredients needed
- Exact places in this codebase where it could wire in: `core/research/analysis_pack.py`, `Saguaro/saguaro/services/comparative.py`, `core/campaign/control_plane.py`
- Existing primitives it can reuse: repo dossier, fingerprint, capability matrix, semantic inventory
- New primitive, data flow, or subsystem needed: shallow accept/reject gate for deep session creation
- `repo_scope`: `["analysis_external", "artifact_store"]`
- `owning_specialist_type`: `["comparative-triage"]`
- `allowed_writes`: `["core/research/analysis_pack.py", "Saguaro/saguaro/services/comparative.py", "core/campaign/control_plane.py"]`
- `telemetry_contract`: `{"minimum": ["shallow_reject_count", "deep_promotion_rate", "triage_false_negative_rate"]}`
- `required_evidence`: `["accepted/rejected candidate review set"]`
- `rollback_criteria`: `["good candidates consistently rejected", "gate duplicates ranker without new value"]`
- `promotion_gate`: shallow triage removes obvious low-fit repos before full comparison
- `success_criteria`: deep analysis budget is protected
- Why this creates value: cost discipline
- Why this creates moat: triage quality shapes overall system quality
- Main risk or failure mode: crude heuristics cause strategic misses
- Smallest credible first experiment: reject repos with language and build mismatch plus poor semantic overlap
- Confidence level: high

### C11. Comparative Session Autopilot

- Suggested `phase_id`: `analysis_upgrade`
- Class: `practical`
- Core insight: once a candidate passes shallow gating, the system should automatically create a quarantined comparative session and compare it against the target root or target subtree.
- External inspiration or analogy: automatic dyno run after intake
- Why it fits Saguaro and Anvil specifically: `ComparativeAnalysisService.compare()` already exists
- Exact places in this codebase where it could wire in: `Saguaro/saguaro/services/comparative.py`, `core/campaign/control_plane.py`, `core/research/repo_acquisition.py`
- Existing primitives it can reuse: create_session, compare, repo acquisition
- New primitive, data flow, or subsystem needed: autopilot policy for target/candidate pairing
- `repo_scope`: `["target", "analysis_external", "artifact_store"]`
- `owning_specialist_type`: `["comparative-autopilot"]`
- `allowed_writes`: `["Saguaro/saguaro/services/comparative.py", "core/campaign/control_plane.py", "core/research/repo_acquisition.py"]`
- `telemetry_contract`: `{"minimum": ["comparative_session_count", "comparison_compile_ms", "session_reuse_rate"]}`
- `required_evidence`: `["session manifests", "comparative reports", "session cleanup log"]`
- `rollback_criteria`: `["session explosion", "duplicate comparisons", "stale session reuse"]`
- `promotion_gate`: accepted candidates automatically yield comparative artifacts
- `success_criteria`: discovery does not stop at cloning
- Why this creates value: it closes the loop from scouting to evidence
- Why this creates moat: automation quality compounds
- Main risk or failure mode: too many auto-comparisons dilute attention
- Smallest credible first experiment: autopilot only for top 3 candidates
- Confidence level: high

### C12. Discovery Telemetry and Black Box Replay

- Suggested `phase_id`: `analysis_upgrade`
- Class: `practical`
- Core insight: discovery should emit replayable telemetry into the same black-box and event-store model as the rest of the campaign runtime.
- External inspiration or analogy: flight recorder discipline
- Why it fits Saguaro and Anvil specifically: `core/telemetry/black_box.py` already does exactly this for run events
- Exact places in this codebase where it could wire in: `core/telemetry/black_box.py`, `core/campaign/control_plane.py`, `core/campaign/retrieval_policy.py`, `Saguaro/saguaro/discovery/github_client.py`
- Existing primitives it can reuse: event store, flight recorder, retrieval policy telemetry
- New primitive, data flow, or subsystem needed: discovery event types `candidate_found`, `candidate_rejected`, `candidate_acquired`, `rate_limit_hit`
- `repo_scope`: `["artifact_store"]`
- `owning_specialist_type`: `["telemetry-assurance"]`
- `allowed_writes`: `["core/telemetry/black_box.py", "core/campaign/control_plane.py", "core/campaign/retrieval_policy.py", "Saguaro/saguaro/discovery/github_client.py"]`
- `telemetry_contract`: `{"minimum": ["discovery_event_count", "rate_limit_hit_count", "candidate_funnel_dropoff"]}`
- `required_evidence`: `["replay export", "event taxonomy", "sample run trace"]`
- `rollback_criteria`: `["discovery path becomes opaque", "events emitted without campaign linkage"]`
- `promotion_gate`: one discovery run is fully replayable
- `success_criteria`: no important discovery decision exists only in logs or memory
- Why this creates value: fast debugging and defensible ops
- Why this creates moat: high-fidelity operational history is rare
- Main risk or failure mode: event spam
- Smallest credible first experiment: record search, rank, clone, compare, cleanup only
- Confidence level: high

### C13. Frontier Expansion Controller

- Suggested `phase_id`: `eid`
- Class: `practical`
- Core insight: comparative findings should recursively improve discovery queries and candidate targeting.
- External inspiration or analogy: adaptive reconnaissance loop
- Why it fits Saguaro and Anvil specifically: `run_research()` already tracks frontier candidates and stop proof
- Exact places in this codebase where it could wire in: `core/campaign/control_plane.py`, `core/dare/web_research.py`, `Saguaro/saguaro/discovery/ranker.py`
- Existing primitives it can reuse: frontier history, stop proof, retrieval routes
- New primitive, data flow, or subsystem needed: frontier-expansion policy using comparative evidence
- `repo_scope`: `["artifact_store", "analysis_external"]`
- `owning_specialist_type`: `["frontier-planning"]`
- `allowed_writes`: `["core/campaign/control_plane.py", "core/dare/web_research.py", "Saguaro/saguaro/discovery/ranker.py"]`
- `telemetry_contract`: `{"minimum": ["frontier_expansion_count", "yield_per_round", "stop_proof_reason"]}`
- `required_evidence`: `["frontier history", "improved query derivations"]`
- `rollback_criteria`: `["frontier loops endlessly", "novelty collapses after first round"]`
- `promotion_gate`: second-round queries are measurably better than first-round generic queries
- `success_criteria`: discovery becomes a learning loop
- Why this creates value: better results over time
- Why this creates moat: adaptive discovery becomes cumulative advantage
- Main risk or failure mode: self-reinforcing local maxima
- Smallest credible first experiment: expand only from top comparative relations and repo dossier tags
- Confidence level: medium-high

### C14. Repo Dossier Fusion Into Retrieval Policy

- Suggested `phase_id`: `eid`
- Class: `practical`
- Core insight: once external repos are ingested, repo dossier evidence should influence later browser/API fetch decisions and task-packet generation.
- External inspiration or analogy: memory-backed pit-wall briefing
- Why it fits Saguaro and Anvil specifically: `RetrievalPolicyEngine` and `ensure_repo_dossier_brief()` already exist
- Exact places in this codebase where it could wire in: `core/campaign/retrieval_policy.py`, `core/campaign/control_plane.py`, `core/research/eid_master.py`
- Existing primitives it can reuse: repo dossier brief, retrieval decision, memory fabric read recording
- New primitive, data flow, or subsystem needed: discovery dossier priority and reuse weighting
- `repo_scope`: `["artifact_store", "analysis_external", "target"]`
- `owning_specialist_type`: `["retrieval-architecture"]`
- `allowed_writes`: `["core/campaign/retrieval_policy.py", "core/campaign/control_plane.py", "core/research/eid_master.py"]`
- `telemetry_contract`: `{"minimum": ["repo_dossier_reuse_count", "artifact_cache_reuse_count", "browser_fetch_avoided_count"]}`
- `required_evidence`: `["before/after routing decisions", "dossier reuse examples"]`
- `rollback_criteria`: `["dossiers become stale and still dominate routing"]`
- `promotion_gate`: dossier presence measurably reduces redundant external fetches
- `success_criteria`: discovery becomes a cumulative evidence system, not a stateless crawler
- Why this creates value: lower cost and higher continuity
- Why this creates moat: memory-backed repo discovery is stronger than repeated cold search
- Main risk or failure mode: stale dossier bias
- Smallest credible first experiment: boost previously successful repo families in ranker explanations
- Confidence level: high

### C15. Campaign Operator Surface for Discovery

- Suggested `phase_id`: `development`
- Class: `practical`
- Core insight: the user should be able to start a discovery campaign from CLI, REPL, or control-plane APIs with explicit auth mode and acquisition budget.
- External inspiration or analogy: mission-control console
- Why it fits Saguaro and Anvil specifically: operator surfaces already exist across `saguaro`, REPL, and campaign runner
- Exact places in this codebase where it could wire in: `Saguaro/saguaro/cli.py`, `cli/repl.py`, `core/campaign/runner.py`, `core/unified_chat_loop.py`
- Existing primitives it can reuse: runner, control plane, operator status, artifact listing
- New primitive, data flow, or subsystem needed: discovery commands and review commands
- `repo_scope`: `["target", "analysis_external", "artifact_store"]`
- `owning_specialist_type`: `["operator-experience"]`
- `allowed_writes`: `["Saguaro/saguaro/cli.py", "cli/repl.py", "core/campaign/runner.py", "core/unified_chat_loop.py"]`
- `telemetry_contract`: `{"minimum": ["discovery_cli_invocations", "discovery_repl_invocations", "operator_abort_count"]}`
- `required_evidence`: `["CLI transcript", "REPL transcript", "help output"]`
- `rollback_criteria`: `["operator surface hides auth state", "budget defaults are unsafe"]`
- `promotion_gate`: one command creates a discovery session and one command promotes candidates into acquisition
- `success_criteria`: discovery is usable without hand-running shell and curl steps
- Why this creates value: operational reach
- Why this creates moat: integrated UX matters
- Main risk or failure mode: too many knobs too early
- Smallest credible first experiment: hidden experimental CLI subcommand first
- Confidence level: high

### C16. Benchmark and Golden Relevance Harness

- Suggested `phase_id`: `deep_test_audit`
- Class: `practical`
- Core insight: discovery quality needs a golden set of known-good requests, known-good repos, and known-bad repos.
- External inspiration or analogy: wind-tunnel baseline dataset; RepoHyper-style benchmark discipline. [R16][R17][R18]
- Why it fits Saguaro and Anvil specifically: this repo already values benchmarks and runtime gates
- Exact places in this codebase where it could wire in: `benchmarks/`, `tests/`, `core/campaign/control_plane.py`, `Saguaro/saguaro/discovery/ranker.py`
- Existing primitives it can reuse: benchmark suite patterns, impact traces, roadmap validation
- New primitive, data flow, or subsystem needed: golden discovery corpus and eval runner
- `repo_scope`: `["artifact_store"]`
- `owning_specialist_type`: `["benchmark-engineering"]`
- `allowed_writes`: `["benchmarks/discovery_frontier_benchmark.py", "tests/test_discovery_ranker.py", "core/campaign/control_plane.py", "Saguaro/saguaro/discovery/ranker.py"]`
- `telemetry_contract`: `{"minimum": ["precision_at_5", "precision_at_10", "wasted_clone_rate"]}`
- `required_evidence`: `["benchmark run output", "golden-set definition", "error analysis"]`
- `rollback_criteria`: `["no benchmark truth set exists", "benchmark drift untracked"]`
- `promotion_gate`: ranker must beat baseline search-by-stars on the golden set
- `success_criteria`: discovery quality becomes measurable
- Why this creates value: disciplined improvement
- Why this creates moat: benchmark-backed iteration moves faster
- Main risk or failure mode: gold set too narrow
- Smallest credible first experiment: 10 handcrafted requests relevant to Anvil/Saguaro
- Confidence level: high

### C17. Repo Hyper-Embedding Fleet Index

- Suggested `phase_id`: `analysis_upgrade`
- Class: `moonshot`
- Core insight: over time, each repo session should produce a compact repo-level embedding for fleet-wide similarity search before full comparison.
- External inspiration or analogy: RepoHyper plus native vector search. [R16]
- Why it fits Saguaro and Anvil specifically: native vector infrastructure already exists
- Exact places in this codebase where it could wire in: `core/research/analysis_pack.py`, `Saguaro/saguaro/storage/native_vector_store.py`, `Saguaro/saguaro/services/comparative.py`
- Existing primitives it can reuse: semantic inventory, build fingerprint, vector store
- New primitive, data flow, or subsystem needed: repo-level embedding schema and fleet ANN index
- `repo_scope`: `["analysis_external", "artifact_store"]`
- `owning_specialist_type`: `["semantic-systems"]`
- `allowed_writes`: `["core/research/analysis_pack.py", "Saguaro/saguaro/storage/native_vector_store.py", "Saguaro/saguaro/services/comparative.py"]`
- `telemetry_contract`: `{"minimum": ["repo_embedding_count", "fleet_ann_latency_ms", "embedding_rank_gain"]}`
- `required_evidence`: `["embedding ablation", "fleet search review"]`
- `rollback_criteria`: `["opaque embeddings replace explicit evidence", "ranking degrades"]`
- `promotion_gate`: repo-level ANN search improves shortlist quality on difficult requests
- `success_criteria`: discovery scales across a growing external repo fleet
- Why this creates value: long-term retrieval speed and breadth
- Why this creates moat: repo-level semantic fleet memory is defensible
- Main risk or failure mode: poor labels and poor explainability
- Smallest credible first experiment: embedding as rerank-only, never source of truth
- Confidence level: medium

### C18. Temporal Maintenance Oracle

- Suggested `phase_id`: `eid`
- Class: `moonshot`
- Core insight: repo quality should factor in maintenance trajectory, not just static stars and README polish.
- External inspiration or analogy: GH Archive time-series telemetry. [R12]
- Why it fits Saguaro and Anvil specifically: frontier selection should reward alive, relevant repos
- Exact places in this codebase where it could wire in: `Saguaro/saguaro/discovery/ranker.py`, `core/campaign/control_plane.py`
- Existing primitives it can reuse: utility scoring, stop proof, frontier priority
- New primitive, data flow, or subsystem needed: activity-health model over repo event history
- `repo_scope`: `["artifact_store"]`
- `owning_specialist_type`: `["temporal-intelligence"]`
- `allowed_writes`: `["Saguaro/saguaro/discovery/ranker.py", "core/campaign/control_plane.py"]`
- `telemetry_contract`: `{"minimum": ["maintenance_score", "event_freshness_days", "rank_change_from_temporal_signal"]}`
- `required_evidence`: `["maintenance trend examples", "temporal rerank review"]`
- `rollback_criteria`: `["recent activity overwhelms technical fit", "API cost too high"]`
- `promotion_gate`: temporal signal improves ranking for at least one real discovery run
- `success_criteria`: dead or drifting repos are filtered more intelligently
- Why this creates value: better long-term fit
- Why this creates moat: time-aware repo selection is strategically powerful
- Main risk or failure mode: noisy event data
- Smallest credible first experiment: use recent push and release cadence only
- Confidence level: medium

### C19. Motif Foundry and Capability Atoms

- Suggested `phase_id`: `eid`
- Class: `moonshot`
- Core insight: repeated mechanisms discovered across external repos should be abstracted into reusable capability atoms and patterns.
- External inspiration or analogy: spare-parts catalog plus design language
- Why it fits Saguaro and Anvil specifically: EID already reasons about innovation hypotheses and experimental tracks
- Exact places in this codebase where it could wire in: `core/research/eid_master.py`, `Saguaro/saguaro/services/comparative.py`, `core/research/analysis_pack.py`
- Existing primitives it can reuse: comparative relations, repo dossiers, EID hypothesis promotion
- New primitive, data flow, or subsystem needed: motif registry keyed by mechanism rather than repo
- `repo_scope`: `["artifact_store", "analysis_external", "target"]`
- `owning_specialist_type`: `["innovation-architecture"]`
- `allowed_writes`: `["core/research/eid_master.py", "Saguaro/saguaro/services/comparative.py", "core/research/analysis_pack.py"]`
- `telemetry_contract`: `{"minimum": ["motif_count", "motif_promotion_rate", "cross_repo_motif_frequency"]}`
- `required_evidence`: `["motif registry entries", "cross-repo support"]`
- `rollback_criteria`: `["motifs become vague slogans", "no implementation consequence"]`
- `promotion_gate`: one motif is promoted into a concrete roadmap item
- `success_criteria`: the system learns mechanisms, not just repos
- Why this creates value: durable technical insight
- Why this creates moat: capability atoms are strong long-term memory
- Main risk or failure mode: abstraction before evidence
- Smallest credible first experiment: extract motifs only from top 3 comparative reports
- Confidence level: medium

### C20. Patch Wind Tunnel

- Suggested `phase_id`: `convergence`
- Class: `moonshot`
- Core insight: before promoting code reuse or native rewrite, simulate the likely blast radius and compatibility cost.
- External inspiration or analogy: wind tunnel plus hardware-in-the-loop
- Why it fits Saguaro and Anvil specifically: impact analysis and campaign verification already exist
- Exact places in this codebase where it could wire in: `Saguaro/saguaro/analysis/impact.py`, `core/campaign/control_plane.py`, `core/research/eid_master.py`
- Existing primitives it can reuse: impact, verification lane, comparative reports
- New primitive, data flow, or subsystem needed: pre-promotion simulation verdict
- `repo_scope`: `["target", "artifact_store"]`
- `owning_specialist_type`: `["integration-simulation"]`
- `allowed_writes`: `["Saguaro/saguaro/analysis/impact.py", "core/campaign/control_plane.py", "core/research/eid_master.py"]`
- `telemetry_contract`: `{"minimum": ["predicted_blast_radius", "simulation_vs_actual_delta"]}`
- `required_evidence`: `["simulation artifact", "actual comparison after implementation"]`
- `rollback_criteria`: `["simulation quality is poor enough to mislead promotion"]`
- `promotion_gate`: one promoted candidate includes simulation output
- `success_criteria`: comparative adoption comes with forecast, not guesswork
- Why this creates value: safer ambitious moves
- Why this creates moat: pre-integration simulation is high leverage
- Main risk or failure mode: false precision
- Smallest credible first experiment: touched-file and touched-test forecast only
- Confidence level: medium-low

---

## 6. Critical Pressure Test

### Elegant but likely wrong

1. `[Inference]` Going GraphQL-only from the start. REST search plus GraphQL hydration is more realistic.
2. `[Inference]` Treating repo embeddings as a first-line source of truth before provenance, ranker signals, and comparative reports are solid.
3. `[Inference]` Cloning everything into `repo_analysis/` because it feels simple. It is operationally wrong.
4. `[Inference]` Assuming GitHub App auth is "just credentials." It is a control-plane and developer-experience problem too.

### Ugly but strategically powerful

1. `[Repo+Inference]` Fixing the `PYTHONPATH=.` bootstrap requirement is not glamorous, but discovery-heavy workflows will be brittle until it is resolved.
2. `[Repo+Inference]` Turning `repo_analysis/` into a mirror rather than a primary store is less flashy than "auto-ingest repos," but much stronger.
3. `[Repo+Inference]` Rate-limit capture, ETag reuse, and token-broker telemetry are plumbing. They are also what makes the system operable at all.
4. `[Repo+Inference]` A shallow fingerprint gate feels conservative, but it is what protects the deep-analysis budget.

### Likely to fail because the repo is missing a key primitive

1. `[Repo+Inference]` Fleet-wide semantic repo embeddings will be weak until there is a labeled discovery quality harness.
2. `[Repo+Inference]` Long-horizon frontier learning will be noisy until discovery provenance and candidate outcomes are stored consistently.
3. `[Repo+Inference]` Patch wind-tunnel simulation will be weak until comparative relation quality and integration evidence are stable.

### What this means for the current user ask

The strongest near-term implementation is:

- authenticated GitHub API client,
- rank-and-hydrate discovery flow,
- blobless acquisition into campaign cache,
- `repo_analysis` session mirror,
- automatic comparative session creation,
- and campaign artifacts plus roadmap-ready evidence.

That is technically plausible now because the repo already contains most of the supporting scaffolding.

---

## 7. Synthesis

### Strongest overall ideas

1. `C01` Authenticated GitHub Access Layer
2. `C03` Discovery Provenance Envelope
3. `C06` Discovery Hangar and `repo_analysis` Mirror
4. `C07` Partial Clone and Sparse Materialization
5. `C11` Comparative Session Autopilot

### Best balance of novelty and plausibility

1. `C06` Discovery Hangar and `repo_analysis` Mirror
2. `C04` Multi-Signal Candidate Ranker
3. `C11` Comparative Session Autopilot
4. `C14` Repo Dossier Fusion Into Retrieval Policy

### Most feasible now

`C01` plus `C06` plus `C07`

Reason:

- `[Repo]` the weak unauthenticated path already exists,
- `[Repo]` repo caching and comparative sessions already exist,
- `[Repo]` the biggest gap is disciplined authenticated wiring and operator-safe storage.

### Biggest long-term moat bet

`C17` Repo Hyper-Embedding Fleet Index plus `C19` Motif Foundry

Reason:

- `[Synthesis]` that is how discovery becomes a compounding research engine rather than a nicer search tool.

### Cleanest unification with the current codebase

Route discovery through:

- `core/dare/web_research.py` for source integration,
- `core/campaign/control_plane.py` for orchestration,
- `core/campaign/repo_cache.py` for acquisition,
- `Saguaro/saguaro/services/comparative.py` for isolated sessions,
- and `core/campaign/retrieval_policy.py` for budgeted reuse and telemetry.

### Prototype first

Prototype order by conviction:

1. authenticated GitHub search and GraphQL hydration
2. campaign-cache acquisition plus `repo_analysis` mirror
3. shallow fingerprint gate
4. comparative session autopilot
5. discovery provenance plus replay telemetry

---

## 8. Implementation Program

### Phase `intake`

- Phase title: Authenticated GitHub Discovery Substrate
- `phase_id`: `intake`
- Objective: replace unauthenticated GitHub search with a governed authenticated discovery client and token broker
- Dependencies: none
- Repo scope: `["artifact_store", "analysis_external"]`
- Owning specialist type: `["discovery-engineering", "security-platform"]`
- Allowed writes: `["core/dare/web_research.py", "core/campaign/control_plane.py", "config/mcp_servers.yaml", "Saguaro/saguaro/discovery/github_client.py", "Saguaro/saguaro/discovery/auth.py"]`
- Telemetry contract: `{"minimum": ["github_auth_mode", "github_api_calls", "github_rate_limit_remaining", "github_token_source", "github_secondary_limit_events"]}`
- Required evidence: `["authenticated search transcript", "redacted auth telemetry", "token-broker regression tests"]`
- Rollback criteria: `["secret leakage", "silent unauthenticated fallback", "auth path not campaign-visible"]`
- Promotion gate: authenticated GitHub App or PAT search works end to end and degraded mode is explicit
- Success criteria: no default discovery path uses unauthenticated search
- Exact wiring points: `core/dare/web_research.py`, `core/campaign/control_plane.py`, `config/mcp_servers.yaml`, `Saguaro/saguaro/discovery/github_client.py`, `Saguaro/saguaro/discovery/auth.py`
- Deliverables: authenticated client, token broker, rate-limit capture, degraded-mode policy
- Tests: `tests/test_dare_web_research.py`, `tests/test_campaign_control_kernel.py`, `tests/test_operator_surface_contract.py`, `tests/test_entrypoints.py`
- Verification commands:
  - `pytest tests/test_dare_web_research.py tests/test_campaign_control_kernel.py tests/test_operator_surface_contract.py tests/test_entrypoints.py -q`
  - `PYTHONPATH=. ./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
  - `PYTHONPATH=. ./venv/bin/saguaro impact --path core/dare/web_research.py`
- Exit criteria: discovery auth mode is explicit, secure, and test-covered

### Phase `research`

- Phase title: Discovery Hangar and Candidate Ranking
- `phase_id`: `research`
- Objective: build the candidate ranker, dependency-aware discovery signals, and controlled acquisition hangar with `repo_analysis` mirroring
- Dependencies: `["intake"]`
- Repo scope: `["analysis_external", "artifact_store"]`
- Owning specialist type: `["discovery-intelligence", "repo-acquisition", "workspace-operations"]`
- Allowed writes: `["core/campaign/repo_cache.py", "core/campaign/workspace.py", "core/campaign/repo_registry.py", "core/research/repo_acquisition.py", "Saguaro/saguaro/discovery/ranker.py", "Saguaro/saguaro/discovery/dependency_sources.py", "Saguaro/saguaro/discovery/session_mirror.py"]`
- Telemetry contract: `{"minimum": ["clone_mode", "materialized_bytes", "top_k_acceptance_rate", "dependency_signal_hits", "mirrored_repo_count", "candidate_to_acquisition_ratio"]}`
- Required evidence: `["ranked candidate review", "cache-to-mirror manifest", "bandwidth comparison", "dependency overlap sample"]`
- Rollback criteria: `["repo_analysis becomes authoritative", "full clones become default", "ranker cannot beat naive baseline"]`
- Promotion gate: top candidates are mirrored into `repo_analysis` via cache-backed sessions and shallow triage
- Success criteria: the user can inspect temporary discovery repos without losing cache authority or budget control
- Exact wiring points: `core/campaign/repo_cache.py`, `core/campaign/workspace.py`, `core/campaign/repo_registry.py`, `core/research/repo_acquisition.py`, `Saguaro/saguaro/discovery/ranker.py`, `Saguaro/saguaro/discovery/dependency_sources.py`, `Saguaro/saguaro/discovery/session_mirror.py`
- Deliverables: partial-clone acquisition modes, discovery mirror, dependency signal fetcher, explainable ranker
- Tests: `tests/test_repo_campaign_subagent.py`, `tests/test_campaign_retrieval_budget_router.py`, `tests/test_discovery_ranker.py`, `tests/test_repo_cache_partial_clone.py`
- Verification commands:
  - `pytest tests/test_repo_campaign_subagent.py tests/test_campaign_retrieval_budget_router.py tests/test_discovery_ranker.py tests/test_repo_cache_partial_clone.py -q`
  - `PYTHONPATH=. ./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
  - `PYTHONPATH=. ./venv/bin/saguaro impact --path core/campaign/control_plane.py`
- Exit criteria: acquisition is efficient, mirrored, and provenance-backed

### Phase `analysis_upgrade`

- Phase title: Comparative Session Autopilot and Triage
- `phase_id`: `analysis_upgrade`
- Objective: automatically create quarantined comparative sessions for accepted discovery candidates and reject weak candidates via shallow fingerprinting
- Dependencies: `["research"]`
- Repo scope: `["target", "analysis_external", "artifact_store"]`
- Owning specialist type: `["comparative-triage", "comparative-autopilot", "telemetry-assurance"]`
- Allowed writes: `["core/research/analysis_pack.py", "core/campaign/control_plane.py", "Saguaro/saguaro/services/comparative.py", "core/telemetry/black_box.py", "Saguaro/saguaro/discovery/provenance.py"]`
- Telemetry contract: `{"minimum": ["shallow_reject_count", "comparative_session_count", "comparison_compile_ms", "rate_limit_hit_count", "candidate_funnel_dropoff"]}`
- Required evidence: `["accepted/rejected triage set", "comparative reports", "replay trace", "session manifests"]`
- Rollback criteria: `["session explosion", "weak candidates consume deep-analysis budget", "discovery replay incomplete"]`
- Promotion gate: one discovery run yields accepted candidates, comparative sessions, and replayable telemetry
- Success criteria: discovery naturally terminates in comparative evidence
- Exact wiring points: `core/research/analysis_pack.py`, `core/campaign/control_plane.py`, `Saguaro/saguaro/services/comparative.py`, `core/telemetry/black_box.py`, `Saguaro/saguaro/discovery/provenance.py`
- Deliverables: shallow gate, autopilot compare policy, discovery events in flight recorder, candidate envelopes
- Tests: `tests/test_repo_campaign_subagent.py`, `tests/test_campaign_control_kernel.py`, `tests/test_discovery_provenance.py`, `tests/test_saguaro_comparative.py`
- Verification commands:
  - `pytest tests/test_repo_campaign_subagent.py tests/test_campaign_control_kernel.py tests/test_discovery_provenance.py tests/test_saguaro_comparative.py -q`
  - `PYTHONPATH=. ./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
  - `PYTHONPATH=. ./venv/bin/saguaro impact --path core/campaign/control_plane.py`
- Exit criteria: accepted candidates become comparative artifacts with no manual glue

### Phase `eid`

- Phase title: Frontier Learning and Repo-Dossier Reuse
- `phase_id`: `eid`
- Objective: feed comparative outcomes back into the research frontier and route later work using repo-dossier memory and discovery outcomes
- Dependencies: `["analysis_upgrade"]`
- Repo scope: `["artifact_store", "analysis_external", "target"]`
- Owning specialist type: `["frontier-planning", "retrieval-architecture"]`
- Allowed writes: `["core/campaign/control_plane.py", "core/campaign/retrieval_policy.py", "core/research/eid_master.py", "Saguaro/saguaro/discovery/ranker.py"]`
- Telemetry contract: `{"minimum": ["frontier_expansion_count", "yield_per_round", "repo_dossier_reuse_count", "browser_fetch_avoided_count"]}`
- Required evidence: `["frontier history", "query evolution", "routing delta", "dossier reuse samples"]`
- Rollback criteria: `["frontier loops without yield", "stale dossier bias dominates routing"]`
- Promotion gate: second-round discovery improves from first-round evidence and later research runs reuse repo dossiers materially
- Success criteria: the system learns which repo families actually help
- Exact wiring points: `core/campaign/control_plane.py`, `core/campaign/retrieval_policy.py`, `core/research/eid_master.py`, `Saguaro/saguaro/discovery/ranker.py`
- Deliverables: frontier-expansion controller, dossier-aware reuse weighting, EID discovery evidence bridge
- Tests: `tests/test_campaign_research_frontier.py`, `tests/test_campaign_eid_portfolio_scheduler.py`, `tests/test_campaign_retrieval_budget_router.py`
- Verification commands:
  - `pytest tests/test_campaign_research_frontier.py tests/test_campaign_eid_portfolio_scheduler.py tests/test_campaign_retrieval_budget_router.py -q`
  - `PYTHONPATH=. ./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
  - `PYTHONPATH=. ./venv/bin/saguaro impact --path core/campaign/control_plane.py`
- Exit criteria: discovery becomes a closed-loop evidence engine rather than one-shot search

### Phase `development`

- Phase title: Operator Surface and Discovery Contracts
- `phase_id`: `development`
- Objective: expose discovery sessions, auth mode, acquisition budgets, and promotion review through CLI, REPL, and chat-loop surfaces
- Dependencies: `["analysis_upgrade", "eid"]`
- Repo scope: `["target", "analysis_external", "artifact_store"]`
- Owning specialist type: `["operator-experience"]`
- Allowed writes: `["Saguaro/saguaro/cli.py", "cli/repl.py", "core/campaign/runner.py", "core/unified_chat_loop.py", "core/prompts/prompt_manager.py"]`
- Telemetry contract: `{"minimum": ["discovery_cli_invocations", "discovery_repl_invocations", "operator_abort_count", "promotion_review_count"]}`
- Required evidence: `["CLI help output", "REPL transcript", "prompt contract update", "operator review flow"]`
- Rollback criteria: `["unsafe defaults", "auth state hidden from user", "promotion flow bypassed"]`
- Promotion gate: discovery is operable through user-facing surfaces without shell-only workarounds
- Success criteria: discovery is a first-class user capability
- Exact wiring points: `Saguaro/saguaro/cli.py`, `cli/repl.py`, `core/campaign/runner.py`, `core/unified_chat_loop.py`, `core/prompts/prompt_manager.py`
- Deliverables: discovery CLI, review CLI, REPL actions, prompt contract additions
- Tests: `tests/test_interactive_cli.py`, `tests/test_cli_repl_startup.py`, `tests/test_campaign_runner.py`, `tests/test_unified_chat_loop.py`
- Verification commands:
  - `pytest tests/test_interactive_cli.py tests/test_cli_repl_startup.py tests/test_campaign_runner.py tests/test_unified_chat_loop.py -q`
  - `PYTHONPATH=. ./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
  - `PYTHONPATH=. ./venv/bin/saguaro entrypoints`
- Exit criteria: discovery is visible, inspectable, and promotion-gated

### Phase `deep_test_audit`

- Phase title: Discovery Reliability and CLI Bootstrap Hardening
- `phase_id`: `deep_test_audit`
- Objective: harden discovery with a golden relevance harness, replay audits, and a bootstrap fix so discovery commands do not require manual `PYTHONPATH=.` intervention
- Dependencies: `["development"]`
- Repo scope: `["artifact_store", "target"]`
- Owning specialist type: `["benchmark-engineering", "comparative-assurance"]`
- Allowed writes: `["Saguaro/saguaro/bootstrap.py", "Saguaro/saguaro/cli.py", "benchmarks/discovery_frontier_benchmark.py", "tests/test_saguaro_cli_bootstrap.py", "tests/test_discovery_ranker.py"]`
- Telemetry contract: `{"minimum": ["precision_at_5", "precision_at_10", "wasted_clone_rate", "cli_bootstrap_failure_count"]}`
- Required evidence: `["benchmark output", "bootstrap regression test", "replay audit sample"]`
- Rollback criteria: `["bootstrap remains fragile", "ranker quality is not measurable"]`
- Promotion gate: benchmark harness exists and Saguaro discovery commands run from venv without manual path patching
- Success criteria: discovery is reliable enough for repeated use
- Exact wiring points: `Saguaro/saguaro/bootstrap.py`, `Saguaro/saguaro/cli.py`, `benchmarks/discovery_frontier_benchmark.py`, `tests/test_saguaro_cli_bootstrap.py`, `tests/test_discovery_ranker.py`
- Deliverables: bootstrap fix, benchmark harness, discovery eval suite
- Tests: `tests/test_saguaro_cli_bootstrap.py`, `tests/test_discovery_ranker.py`, `tests/test_entrypoints.py`, `tests/test_saguaro_roadmap_validator.py`
- Verification commands:
  - `pytest tests/test_saguaro_cli_bootstrap.py tests/test_discovery_ranker.py tests/test_entrypoints.py tests/test_saguaro_roadmap_validator.py -q`
  - `PYTHONPATH=. ./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
  - `./venv/bin/saguaro health`
- Exit criteria: discovery is benchmarked, replayable, and toolchain-safe

---

## 9. Implementation Contract

- The system shall implement authenticated GitHub discovery through `core/dare/web_research.py`, `core/campaign/control_plane.py`, `Saguaro/saguaro/discovery/github_client.py`, and `Saguaro/saguaro/discovery/auth.py`, tested by `tests/test_dare_web_research.py`, `tests/test_campaign_control_kernel.py`, and `tests/test_entrypoints.py`, and verified with `pytest tests/test_dare_web_research.py tests/test_campaign_control_kernel.py tests/test_entrypoints.py -q`, `PYTHONPATH=. ./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`, and `PYTHONPATH=. ./venv/bin/saguaro impact --path core/dare/web_research.py`.

- The system shall support GitHub App installation tokens and fine-grained PAT fallback through `Saguaro/saguaro/discovery/auth.py`, `config/mcp_servers.yaml`, `core/campaign/control_plane.py`, and `core/campaign/state_store.py`, tested by `tests/test_github_auth_broker.py`, `tests/test_campaign_control_kernel.py`, and `tests/test_operator_surface_contract.py`, and verified with `pytest tests/test_github_auth_broker.py tests/test_campaign_control_kernel.py tests/test_operator_surface_contract.py -q` and `PYTHONPATH=. ./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.

- The system shall persist discovery provenance for every accepted candidate through `Saguaro/saguaro/discovery/provenance.py`, `core/campaign/control_plane.py`, `core/campaign/repo_registry.py`, and `core/research/analysis_pack.py`, tested by `tests/test_discovery_provenance.py`, `tests/test_repo_campaign_subagent.py`, and `tests/test_campaign_state_store_phase_artifacts.py`, and verified with `pytest tests/test_discovery_provenance.py tests/test_repo_campaign_subagent.py tests/test_campaign_state_store_phase_artifacts.py -q`, `PYTHONPATH=. ./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`, and `PYTHONPATH=. ./venv/bin/saguaro roadmap validate --path saguaro_discovery_ingestion_roadmap_v2.md --format json`.

- The system shall rank discovery candidates with explainable multi-signal scoring through `Saguaro/saguaro/discovery/ranker.py`, `core/campaign/retrieval_policy.py`, `core/campaign/control_plane.py`, and `Saguaro/saguaro/discovery/dependency_sources.py`, tested by `tests/test_discovery_ranker.py`, `tests/test_campaign_retrieval_budget_router.py`, and `tests/test_campaign_research_frontier.py`, and verified with `pytest tests/test_discovery_ranker.py tests/test_campaign_retrieval_budget_router.py tests/test_campaign_research_frontier.py -q` and `PYTHONPATH=. ./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.

- The system shall acquire external repos through cache-backed partial clone and selective materialization in `core/campaign/repo_cache.py`, `core/research/repo_acquisition.py`, and `Saguaro/saguaro/discovery/github_client.py`, tested by `tests/test_repo_cache_partial_clone.py`, `tests/test_repo_campaign_subagent.py`, and `tests/test_dare_web_research.py`, and verified with `pytest tests/test_repo_cache_partial_clone.py tests/test_repo_campaign_subagent.py tests/test_dare_web_research.py -q`, `PYTHONPATH=. ./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`, and `PYTHONPATH=. ./venv/bin/saguaro impact --path core/campaign/control_plane.py`.

- The system shall materialize discovery sessions into a governed `repo_analysis` mirror through `core/campaign/workspace.py`, `core/campaign/repo_cache.py`, `core/campaign/repo_registry.py`, and `Saguaro/saguaro/discovery/session_mirror.py`, tested by `tests/test_repo_analysis_mirror.py`, `tests/test_repo_campaign_subagent.py`, and `tests/test_campaign_runner.py`, and verified with `pytest tests/test_repo_analysis_mirror.py tests/test_repo_campaign_subagent.py tests/test_campaign_runner.py -q` and `PYTHONPATH=. ./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.

- The system shall reject weak discovery candidates before deep comparison through `core/research/analysis_pack.py`, `Saguaro/saguaro/services/comparative.py`, and `core/campaign/control_plane.py`, tested by `tests/test_discovery_triage.py`, `tests/test_saguaro_comparative.py`, and `tests/test_campaign_control_kernel.py`, and verified with `pytest tests/test_discovery_triage.py tests/test_saguaro_comparative.py tests/test_campaign_control_kernel.py -q`, `PYTHONPATH=. ./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`, and `PYTHONPATH=. ./venv/bin/saguaro impact --path core/campaign/control_plane.py`.

- The system shall automatically create quarantined comparative sessions for accepted candidates through `Saguaro/saguaro/services/comparative.py`, `core/research/repo_acquisition.py`, and `core/campaign/control_plane.py`, tested by `tests/test_saguaro_comparative.py`, `tests/test_repo_campaign_subagent.py`, and `tests/test_campaign_control_kernel.py`, and verified with `pytest tests/test_saguaro_comparative.py tests/test_repo_campaign_subagent.py tests/test_campaign_control_kernel.py -q`, `PYTHONPATH=. ./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`, and `PYTHONPATH=. ./venv/bin/saguaro impact --path core/campaign/control_plane.py`.

- The system shall emit replayable discovery telemetry through `core/telemetry/black_box.py`, `core/campaign/control_plane.py`, `core/campaign/retrieval_policy.py`, and `Saguaro/saguaro/discovery/github_client.py`, tested by `tests/test_discovery_black_box.py`, `tests/test_campaign_retrieval_budget_router.py`, and `tests/test_development_replay.py`, and verified with `pytest tests/test_discovery_black_box.py tests/test_campaign_retrieval_budget_router.py tests/test_development_replay.py -q` and `PYTHONPATH=. ./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.

- The system shall feed discovery outcomes back into frontier planning and repo-dossier reuse through `core/campaign/control_plane.py`, `core/campaign/retrieval_policy.py`, `core/research/eid_master.py`, and `Saguaro/saguaro/discovery/ranker.py`, tested by `tests/test_campaign_research_frontier.py`, `tests/test_campaign_eid_portfolio_scheduler.py`, and `tests/test_campaign_retrieval_budget_router.py`, and verified with `pytest tests/test_campaign_research_frontier.py tests/test_campaign_eid_portfolio_scheduler.py tests/test_campaign_retrieval_budget_router.py -q`, `PYTHONPATH=. ./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`, and `PYTHONPATH=. ./venv/bin/saguaro impact --path core/campaign/control_plane.py`.

- The system shall expose discovery through CLI, REPL, and campaign control surfaces in `Saguaro/saguaro/cli.py`, `cli/repl.py`, `core/campaign/runner.py`, `core/unified_chat_loop.py`, and `core/prompts/prompt_manager.py`, tested by `tests/test_interactive_cli.py`, `tests/test_cli_repl_startup.py`, `tests/test_campaign_runner.py`, and `tests/test_unified_chat_loop.py`, and verified with `pytest tests/test_interactive_cli.py tests/test_cli_repl_startup.py tests/test_campaign_runner.py tests/test_unified_chat_loop.py -q`, `PYTHONPATH=. ./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`, and `PYTHONPATH=. ./venv/bin/saguaro entrypoints`.

- The system shall harden the Saguaro CLI bootstrap for discovery workflows through `Saguaro/saguaro/bootstrap.py` and `Saguaro/saguaro/cli.py`, tested by `tests/test_saguaro_cli_bootstrap.py` and `tests/test_entrypoints.py`, and verified with `pytest tests/test_saguaro_cli_bootstrap.py tests/test_entrypoints.py -q`, `./venv/bin/saguaro health`, and `PYTHONPATH=. ./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
