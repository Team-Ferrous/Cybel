import json
import hashlib
import os
import re
import time
from typing import List, Dict, Any, Optional, Set
import numpy as np
from rich.panel import Panel
from core.agent import BaseAgent
from core.agents.prompt_loader import SpecialistPromptLoader
from core.thinking import ThinkingParser
from core.context import ContextManager
from core.subagent_communication import MessageType, Priority, OWNERSHIP_TOPICS
from core.utils.logger import get_logger

logger = get_logger(__name__)

UNIVERSAL_SPECIALIST_RESEARCH_TOOLS = {
    "web_search",
    "web_fetch",
    "search_arxiv",
    "fetch_arxiv_paper",
    "search_hackernews",
    "search_stackoverflow",
    "search_reddit",
}

SAGUARO_TOOL_NAMES = {
    "saguaro_query",
    "query",
    "skeleton",
    "slice",
    "impact",
    "verify",
    "deadcode",
    "saguaro_index",
    "saguaro_sync",
    "saguaro_workspace",
    "saguaro_daemon",
    "saguaro_doctor",
}


class SubAgent(BaseAgent):
    """
    Unified base class for specialized subagents.
    Provides latent alignment, tool filtering, and mission-focused execution.

    Features:
    - Shared Latent Space: Reuses the parent's brain (KV cache/model) for efficiency.
    - Tool Filtering: Restricts subagent to a specific subset of tools.
    - Result Extraction: Extracts clean summaries from model outputs for tool-based integration.
    """

    system_prompt: str = ""
    tools: List[str] = []
    prompt_profile: str = "default"
    specialist_prompt_key: str = ""
    sovereign_build_policy_block: str = ""
    sovereign_build_policy_enabled: bool = False

    def __init__(
        self,
        task: str,
        parent_name: str = "Master",
        brain=None,
        console=None,
        parent_agent: Optional[BaseAgent] = None,
        quiet: bool = False,
        message_bus: Optional[Any] = None,
        ownership_registry: Optional[Any] = None,
        complexity_profile: Optional[Any] = None,
        context_budget: Optional[int] = None,
        coconut_context_vector: Optional[np.ndarray] = None,
        coconut_depth: Optional[int] = None,
        prompt_profile: Optional[str] = None,
        specialist_prompt_key: Optional[str] = None,
        sovereign_build_policy_block: Optional[str] = None,
        sovereign_build_policy_enabled: Optional[bool] = None,
        prompt_injection: Optional[str] = None,
    ):
        """
        Initialize the subagent.

        Args:
            task: The mission for this subagent.
            parent_name: Name of the agent that launched this one.
            brain: Shared model instance.
            console: Shared console instance.
            parent_agent: Optional full agent instance for deep latent alignment.
            quiet: If True, suppress console output during execution.
        """
        # Latent Alignment: Prefer parent_agent for context sharing, fallback to explicit brain/console
        self.parent_agent = parent_agent
        used_brain = brain or (parent_agent.brain if parent_agent else None)
        used_console = console or (parent_agent.console if parent_agent else None)

        # Combined name for logging/tracing
        name = f"{parent_name}:{self.__class__.__name__}"
        super().__init__(name=name, brain=used_brain, console=used_console)

        self.task = task
        self.parent_name = parent_name
        self.quiet = quiet
        self.message_bus = message_bus
        self.ownership_registry = ownership_registry
        self.complexity_profile = complexity_profile
        self._coconut_context_vector = coconut_context_vector
        self._original_quiet = self.console.quiet if self.console else False
        self._prompt_loader = SpecialistPromptLoader()
        self.prompt_profile = (
            str(
                prompt_profile if prompt_profile is not None else self.prompt_profile
            ).strip()
            or "default"
        )
        self.specialist_prompt_key = str(
            specialist_prompt_key
            if specialist_prompt_key is not None
            else self.specialist_prompt_key
        ).strip()
        self._instance_sovereign_policy_block = str(
            sovereign_build_policy_block or ""
        ).strip()
        default_policy_enabled = bool(self.sovereign_build_policy_enabled)
        if sovereign_build_policy_enabled is None:
            self.sovereign_build_policy_enabled = default_policy_enabled
        else:
            self.sovereign_build_policy_enabled = bool(sovereign_build_policy_enabled)
        self._prompt_injections: List[str] = []
        if prompt_injection:
            self._append_prompt_injection(prompt_injection)
        model_name = getattr(self.brain, "model_name", "") or ""
        self.is_small_model = any(
            tag in model_name.lower()
            for tag in ["tiny", "mini", "small", "1b", "2b", "3b"]
        )
        self._task_complexity = None
        try:
            from core.reasoning.complexity_analyzer import TaskComplexityAnalyzer

            self._task_complexity = TaskComplexityAnalyzer().analyze(task)
        except Exception:
            self._task_complexity = None

        if any(tag in model_name.lower() for tag in ["tiny", "mini", "1b", "2b"]):
            self.max_autonomous_steps = 6
            self.thinking_budget = 60000
        elif any(tag in model_name.lower() for tag in ["small", "3b", "7b", "8b"]):
            self.max_autonomous_steps = 12
            self.thinking_budget = 160000
        else:
            self.max_autonomous_steps = 20
            self.thinking_budget = 300000
        if self._task_complexity is not None:
            self.max_autonomous_steps = min(
                20,
                max(
                    self.max_autonomous_steps, self._task_complexity.max_steps_per_agent
                ),
            )
        self.files_read = set()

        # Share context-heavy components if available
        if parent_agent:
            self.semantic_engine = parent_agent.semantic_engine
            self.project_context = parent_agent.project_context
            self.token_manager = getattr(parent_agent, "token_manager", None)

        from config.settings import AGENTIC_THINKING

        dynamic_coconut = bool(
            self._task_complexity is not None and self._task_complexity.subagent_coconut
        )
        self.coconut_enabled = bool(
            getattr(complexity_profile, "subagent_coconut", False) or dynamic_coconut
        )
        self.show_thinking = AGENTIC_THINKING.get("show_thinking", True)
        self.thinking_budget = min(
            self.thinking_budget, AGENTIC_THINKING.get("thinking_budget", 300000)
        )
        self.context_budget = int(context_budget or 250000)
        self.subagent_context_manager = ContextManager(
            max_tokens=self.context_budget,
            system_prompt_tokens=max(800, int(self.context_budget * 0.05)),
        )

        # Initialize approval manager for subagent
        from core.approval import ApprovalMode

        approval_mode_str = AGENTIC_THINKING.get("approval_mode", "trusted")
        self.approval_manager.set_mode(ApprovalMode(approval_mode_str))

        # Filter tools to the restricted set
        all_schemas = self.registry.get_schemas().get("tools", [])
        if self.tools:
            allowed = set(self.tools)
            allowed.update({"grep_search", "find_by_name"})
            allowed.update(SAGUARO_TOOL_NAMES)
            allowed.update(UNIVERSAL_SPECIALIST_RESEARCH_TOOLS)
            self.tool_schemas = [s for s in all_schemas if s["name"] in allowed]
        else:
            self.tool_schemas = all_schemas

        if self.message_bus is not None:
            try:
                subscriptions = ["guidance", f"guidance.{self.name}"]
                for topic in OWNERSHIP_TOPICS:
                    if topic not in subscriptions:
                        subscriptions.append(topic)
                self.message_bus.register_agent(
                    self.name,
                    subscriptions=subscriptions,
                    metadata={"role": "subagent"},
                )
            except Exception:
                pass
        self._owned_files: Set[str] = set()

        self._tool_intent_classifier = None
        self._result_adapter = None
        self._latent_state = None
        depth_override = coconut_depth
        if depth_override is None:
            try:
                depth_override = int(
                    getattr(complexity_profile, "coconut_depth", 0) or 0
                )
            except Exception:
                depth_override = None
        self._dynamic_coconut_depth = int(
            max(
                1,
                depth_override
                or getattr(complexity_profile, "coconut_steps", 0)
                or getattr(self._task_complexity, "coconut_depth", 2)
                or 2,
            )
        )
        if self.coconut_enabled and self.brain is not None:
            try:
                from core.reasoning.tool_intent_classifier import ToolIntentClassifier
                from core.reasoning.result_adapter import ResultEmbeddingAdapter

                allowed_tools = [
                    tool.get("name", "")
                    for tool in self.tool_schemas
                    if isinstance(tool, dict) and tool.get("name")
                ]
                embedding_dim = self._infer_embedding_dim()
                threshold = 0.76 if self.is_small_model else 0.82
                self._tool_intent_classifier = ToolIntentClassifier(
                    embedding_dim=embedding_dim,
                    tool_names=allowed_tools,
                    threshold=threshold,
                )
                self._result_adapter = ResultEmbeddingAdapter(
                    brain=self.brain,
                    embedding_dim=embedding_dim,
                    residual_weight=0.72,
                )
            except Exception as exc:
                logger.debug("SubAgent latent pipeline initialization skipped: %s", exc)

    def _build_system_prompt(self) -> str:
        return self._build_specialized_system_prompt()

    def _build_specialized_system_prompt(self) -> str:
        if self.is_small_model:
            return self._build_compact_prompt()
        return self._build_full_prompt()

    def _append_prompt_injection(self, text: Optional[str]) -> None:
        injection = str(text or "").strip()
        if injection:
            self._prompt_injections.append(injection)

    def _resolve_sovereign_policy_block(self) -> str:
        class_policy = str(
            getattr(self, "sovereign_build_policy_block", "") or ""
        ).strip()
        if class_policy and self._instance_sovereign_policy_block:
            return f"{class_policy}\n\n{self._instance_sovereign_policy_block}".strip()
        return class_policy or self._instance_sovereign_policy_block

    def _resolve_expertise_prompt(self) -> str:
        role_addendum = (
            self.system_prompt
            or "You are an expert autonomous sub-agent. Produce concrete evidence."
        )
        if self._prompt_injections:
            role_addendum = f"{role_addendum}\n\n## Prompt Injection\n" + "\n\n".join(
                p for p in self._prompt_injections if p
            )
        composed = self._prompt_loader.compose(
            role_addendum=role_addendum,
            prompt_profile=self.prompt_profile,
            specialist_prompt_key=self.specialist_prompt_key,
            include_sovereign_policy=self.sovereign_build_policy_enabled,
            sovereign_policy_block=self._resolve_sovereign_policy_block(),
        )
        return composed or role_addendum

    def _build_common_evidence_envelope_defaults(self) -> Dict[str, Any]:
        return {
            "schema_version": "phase1",
            "fallback_mode": None,
            "saguaro_failures": [],
            "prompt_profile": self.prompt_profile,
            "specialist_prompt_key": self.specialist_prompt_key or None,
            "sovereign_policy": {
                "enabled": bool(self.sovereign_build_policy_enabled),
                "policy_block_present": bool(self._resolve_sovereign_policy_block()),
            },
        }

    @staticmethod
    def _is_tool_execution_error(result: Any) -> bool:
        text = str(result or "").strip().lower()
        if not text:
            return False
        return (
            text.startswith("error")
            or text.startswith("aes gate:")
            or "traceback" in text
        )

    def _build_aes_subagent_payload(self) -> tuple[str, str]:
        candidate_files = sorted(
            set(self.files_read)
            | set(getattr(self.parent_agent, "files_read", set()) or set())
            | set(getattr(self.parent_agent, "files_edited", set()) or set())
        )
        payload, contract = self.prompt_manager.aes_builder.build_subagent_prompt(
            role=self.__class__.__name__,
            task_files=candidate_files or None,
            task_text=self.task,
        )
        contract_block = self.prompt_manager.format_prompt_contract(contract)
        return contract_block, payload

    def _build_full_prompt(self) -> str:
        tools_json = json.dumps(self.tool_schemas, indent=2)
        expertise = self._resolve_expertise_prompt()
        tool_names = [t.get("name", "unknown") for t in self.tool_schemas]
        tools_summary = ", ".join(tool_names)
        contract_block, aes_payload = self._build_aes_subagent_payload()

        return f"""You are {self.name}, an elite autonomous specialist.

# AES PROMPT CONTRACT
{contract_block}

# AES GOVERNANCE PAYLOAD
{aes_payload}

# MISSION
{self.task}

# EXPERTISE
{expertise}

# COGNITIVE CAPABILITIES
- **200K Context Window**: Full access to large codebases without truncation
- **COCONUT Reasoning**: Latent thought exploration for deep analysis
- **Grover Amplification**: Quantum-inspired solution refinement
- **Saguaro Integration**: Semantic code intelligence as ground truth
- **Thinking Budget**: {self.thinking_budget:,} tokens for deep reasoning

# AVAILABLE TOOLS
You have access to the following tools: {tools_summary}

Full tool schemas:
{tools_json}

# CRITICAL: TOOL CALL FORMAT (MANDATORY)
You MUST use this EXACT format to call tools. DO NOT describe tools in natural language.

CORRECT FORMAT:
<thinking>
I need grounded evidence from the codebase. I will begin with Saguaro semantic discovery.
</thinking>
<tool_call>
{{"name": "saguaro_query", "arguments": {{"query": "core execution flow", "k": 5}}}}
</tool_call>

WRONG (DO NOT DO THIS):
"1. I will scan the directory first"
"First, let's grep for it"
"I'll use semantic_search if Saguaro is vague"

# IMMEDIATE ACTION REQUIRED
Your FIRST response MUST include at least ONE <tool_call> block.
Do NOT just describe what you plan to do - CALL THE TOOL.

# SAGUARO-FIRST PROTOCOLS
1. **saguaro_query(query)** - ALWAYS use first for repository discovery.
2. **skeleton(file)** - Use after saguaro_query to understand file structure.
3. **slice(entity)** - Use for specific code entities (classes, functions) once located.
4. **read_file(file)** - Use alongside Saguaro for grounding when concrete implementation detail is needed.
5. **Forbidden fallback behavior** - Do NOT use grep, semantic_search, glob, or list_dir for primary discovery.
6. **Fallback exception** - If Saguaro returns too few results or explicit errors, you MAY use `grep_search` or `find_by_name`.

# THOROUGHNESS PROTOCOL
- Do NOT finish until you have seen the actual source code of the relevant logic.
- If you find a reference to a class or function, you MUST find its definition using `saguaro_query`, `skeleton`, or `slice`.
- "I think X works like Y" is NOT acceptable. You must state "I have verified X works like Y by reading [file]".

# TECHNOLOGY STACK DETECTION PROTOCOL
Before concluding which frameworks or architectures are used, you MUST:
1. **Identify Fingerprint Files**: Search for `requirements.txt`, `package.json`, `go.mod`, `pom.xml`, etc.
2. **Verify Entry Points**: Use `skeleton` on suspected main files to verify framework-specific imports.
3. **Ignore Noise**: Treat directories like `venv`, `node_modules`, or `build` as external dependencies.
4. **Admit Uncertainty**: If no clear evidence is found, state "Undetermined". NO GUESSING.

# EVIDENCE VERIFICATION PROTOCOL (MANDATORY)
Before concluding your analysis, you MUST:

1. **Read Core Files**: Use `read_file` or `slice` on at least 2-3 key files you identified
2. **Extract Evidence**: For each claim about code, cite with format: `ComponentName` in `path/file.py:L123`
3. **Admit Gaps**: If you cannot find evidence, state "I could not locate [X] in the indexed files"

FAILURE MODE: Analysis based ONLY on file paths or skeleton summaries is INCOMPLETE.
You MUST use `read_file` or `slice` on critical files before producing your final analysis.

Begin your mission now. Your FIRST message MUST contain a <tool_call> block.
"""

    def _build_compact_prompt(self) -> str:
        tools_json = json.dumps(self.tool_schemas, indent=2)
        contract_block, aes_payload = self._build_aes_subagent_payload()
        expertise = self._resolve_expertise_prompt()
        return f"""You are {self.name}. Task: {self.task}

AES PROMPT CONTRACT:
{contract_block}

AES PAYLOAD:
{aes_payload}

RULES:
1. Reply with a <tool_call> block immediately.
2. ALWAYS start with saguaro_query.
3. Use read_file alongside skeleton/slice whenever concrete implementation details are required.
4. Do NOT use grep, semantic_search, glob, or list_dir for primary discovery.
5. If Saguaro returns too few results or errors, you MAY use grep_search/find_by_name.
6. Every claim must come from code you actually read.

EXPERTISE:
{expertise}

Tools:
{tools_json}

Call a tool now."""

    def _build_oneshot_messages(self) -> List[Dict[str, str]]:
        if self.is_small_model:
            return [
                {"role": "user", "content": "How does the config system work?"},
                {
                    "role": "assistant",
                    "content": '<tool_call>\n{"name": "saguaro_query", "arguments": {"query": "configuration system", "k": 3}}\n</tool_call>',
                },
                {
                    "role": "tool",
                    "content": "[1] ConfigManager (class) - config/settings.py:15",
                },
                {
                    "role": "assistant",
                    "content": '<tool_call>\n{"name": "skeleton", "arguments": {"path": "config/settings.py"}}\n</tool_call>',
                },
                {
                    "role": "tool",
                    "content": "class ConfigManager:\n  def load()...\n  def get()...\n  def set()...",
                },
            ]
        return [
            {"role": "user", "content": "How does the authentication system work?"},
            {
                "role": "assistant",
                "content": '<thinking>\nI need to find the authentication code semantically before reading implementations.\n</thinking>\n<tool_call>\n{"name": "saguaro_query", "arguments": {"query": "authentication system", "k": 5}}\n</tool_call>',
            },
            {
                "role": "tool",
                "content": "[1] AuthManager (class) - core/auth.py:42\n[2] login_handler (function) - api/routes.py:156",
            },
        ]

    def _default_first_tool_call(
        self, strategy: Optional[str] = None
    ) -> Dict[str, Any]:
        file_paths = re.findall(r"[\w./-]+\.(?:py|cc|cpp|h|js|ts|md)", self.task)
        existing_file_paths = []
        for path in file_paths:
            candidates = [path, os.path.join(os.getcwd(), path)]
            if any(os.path.exists(candidate) for candidate in candidates):
                existing_file_paths.append(path)
        if existing_file_paths:
            selected_paths = list(dict.fromkeys(existing_file_paths))[:3]
            forced_tool = "read_file" if len(selected_paths) == 1 else "read_files"
            self._last_default_tool_features = {
                "strategy": strategy or "default",
                "file_paths": selected_paths,
                "tech_terms": [],
                "query_text": " ".join(selected_paths),
                "forced_tool": forced_tool,
            }
            if forced_tool == "read_file":
                return {
                    "name": "read_file",
                    "arguments": {"path": selected_paths[0]},
                }
            return {"name": "read_files", "arguments": {"paths": selected_paths}}
        tech_terms = re.findall(
            r"\b[A-Z][a-zA-Z0-9_]*[A-Z][a-zA-Z0-9_]*\b|\b[a-z]+_[a-z0-9_]+\b",
            self.task,
        )
        strategy_prefix = {
            "structure": ["repository", "architecture", "module", "structure"],
            "implementation": ["implementation", "code", "logic", "function"],
            "integration": ["integration", "dependency", "imports", "call graph"],
        }.get(strategy or "", [])
        query_parts = strategy_prefix + file_paths[:2] + tech_terms[:3]
        query_text = (
            " ".join(query_parts) if query_parts else " ".join(self.task.split())[:100]
        )
        self._last_default_tool_features = {
            "strategy": strategy or "default",
            "file_paths": file_paths[:2],
            "tech_terms": tech_terms[:3],
            "query_text": query_text,
            "forced_tool": "saguaro_query",
        }
        return {"name": "saguaro_query", "arguments": {"query": query_text, "k": 10}}

    def _infer_embedding_dim(self) -> int:
        if self._result_adapter is not None:
            return int(self._result_adapter.embedding_dim)
        if not hasattr(self.brain, "embeddings"):
            return 4096
        try:
            probe = np.asarray(self.brain.embeddings(self.task[:128]), dtype=np.float32)
            if probe.ndim == 0:
                return 4096
            if probe.ndim == 1:
                return int(max(8, probe.shape[0]))
            return int(max(8, probe.reshape(probe.shape[0], -1).shape[1]))
        except Exception:
            return 4096

    def _compute_latent_state(self, text: str) -> Optional[np.ndarray]:
        projected: Optional[np.ndarray] = None
        if self._result_adapter is not None:
            projected_arr = self._result_adapter.project_text(text)
            if projected_arr.size > 0:
                projected = projected_arr

        context_seed = self._coerce_latent_matrix(
            self._coconut_context_vector,
            target_dim=(
                projected.shape[1]
                if projected is not None
                else self._infer_embedding_dim()
            ),
        )
        if projected is None:
            return context_seed
        if context_seed is None:
            return projected

        merged = 0.65 * context_seed + 0.35 * projected
        norms = np.linalg.norm(merged, axis=1, keepdims=True)
        norms = np.where(norms <= 1e-8, 1.0, norms)
        return merged / norms

    def _coerce_latent_matrix(
        self, value: Any, target_dim: int
    ) -> Optional[np.ndarray]:
        if value is None:
            return None
        try:
            arr = np.asarray(value, dtype=np.float32)
        except Exception:
            return None
        if arr.size == 0:
            return None
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        if arr.shape[1] > target_dim:
            arr = arr[:, :target_dim]
        elif arr.shape[1] < target_dim:
            pad = np.zeros((arr.shape[0], target_dim - arr.shape[1]), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=1)
        if not np.isfinite(arr).all():
            return None
        return arr

    @staticmethod
    def _serialize_latent_state(state: Optional[np.ndarray]) -> Optional[List[float]]:
        if state is None:
            return None
        arr = np.asarray(state, dtype=np.float32).reshape(-1)
        if arr.size == 0 or not np.isfinite(arr).all():
            return None
        return [float(v) for v in arr]

    def _build_latent_tool_call(
        self, signal: Any, mission_input: str
    ) -> Optional[Dict[str, Any]]:
        tool_name = str(getattr(signal, "tool_name", "") or "").strip()
        if not tool_name:
            return None
        allowed = {s.get("name") for s in self.tool_schemas if isinstance(s, dict)}
        if tool_name not in allowed:
            tool_name = "saguaro_query" if "saguaro_query" in allowed else ""
        if not tool_name:
            return None

        task_files = re.findall(
            r"[\w./-]+\.(?:py|cc|cpp|h|hpp|js|ts|md|json|yaml|yml)",
            mission_input,
        )
        primary_file = task_files[0] if task_files else None
        if tool_name == "saguaro_query":
            return {
                "name": "saguaro_query",
                "arguments": {"query": mission_input[:220], "k": 8},
            }
        if tool_name in {"read_file", "skeleton"} and primary_file:
            key = "path" if tool_name == "skeleton" else "file_path"
            return {"name": tool_name, "arguments": {key: primary_file}}
        if tool_name == "find_by_name" and primary_file:
            return {
                "name": "find_by_name",
                "arguments": {"pattern": os.path.basename(primary_file)},
            }
        return {"name": tool_name, "arguments": {}}

    def _isolated_inference(self, mission_input: str) -> Dict[str, Any]:
        parent_history = self.history
        from cli.history import ConversationHistory

        isolated_history = ConversationHistory(history_file=None)
        isolated_history.add_message("user", mission_input)
        self.history = isolated_history

        subagent_system_prompt = self._build_specialized_system_prompt()

        try:
            step_count = 0
            final_response = ""
            tool_stats = {}
            tool_trace = []
            evidence_envelope = self._build_common_evidence_envelope_defaults()
            saguaro_failures: List[Dict[str, Any]] = []
            fallback_mode = evidence_envelope.get("fallback_mode")
            latent_tool_signals: List[Dict[str, Any]] = []
            latent_reinjections = 0
            start_time = time.time()
            duplicate_fingerprints: Dict[str, int] = {}
            max_duplicate_threshold = 2
            consecutive_no_tool_steps = 0
            tool_strategy = getattr(self, "tool_strategy", None)
            latent_state = self._compute_latent_state(mission_input)
            self._latent_state = latent_state

            while step_count < self.max_autonomous_steps:
                step_count += 1
                logger.info(
                    f"SubAgent {self.name} starting step {step_count}/{self.max_autonomous_steps}"
                )
                self._publish_progress(
                    event="step_start",
                    payload={
                        "step": step_count,
                        "max_steps": self.max_autonomous_steps,
                        "task": self.task[:180],
                    },
                )
                messages = isolated_history.get_messages()
                oneshot = self._build_oneshot_messages()

                chat_messages = (
                    [{"role": "system", "content": subagent_system_prompt}]
                    + oneshot
                    + messages
                )
                prompt_chars = sum(
                    len(str(m.get("content", ""))) for m in chat_messages
                )
                logger.info(
                    "subagent.step.start name=%s step=%d/%d messages=%d prompt_chars=%d strategy=%s",
                    self.name,
                    step_count,
                    self.max_autonomous_steps,
                    len(chat_messages),
                    prompt_chars,
                    tool_strategy or "default",
                )

                recovery_hint = ""
                if (
                    chat_messages
                    and chat_messages[-1]["role"] == "tool"
                    and "Error" in chat_messages[-1]["content"]
                ):
                    recovery_hint = (
                        "\n[SYSTEM NOTE: Previous tool failed. Adjust strategy.]"
                    )
                guidance_hint = self._consume_master_guidance()
                if guidance_hint:
                    recovery_hint += f"\n[SYSTEM GUIDANCE] {guidance_hint}"

                # NOTE: COCONUT latent reasoning is handled at master agent level
                # Subagents focus on tool execution efficiency, not parallel path exploration

                latent_signal = None
                latent_short_circuit = False
                if (
                    self.coconut_enabled
                    and self._tool_intent_classifier is not None
                    and latent_state is not None
                ):
                    try:
                        allowed_tools = [
                            schema.get("name")
                            for schema in self.tool_schemas
                            if isinstance(schema, dict)
                        ]
                        latent_signal = self._tool_intent_classifier.detect(
                            hidden_state=latent_state,
                            context_text=mission_input,
                            allowed_tools=allowed_tools,
                        )
                    except Exception as exc:
                        logger.debug("Latent tool intent detection skipped: %s", exc)

                latent_intent_window = max(1, min(3, int(self._dynamic_coconut_depth)))
                if latent_signal is not None and step_count <= latent_intent_window:
                    forced_call = self._build_latent_tool_call(
                        latent_signal, mission_input
                    )
                    if forced_call:
                        full_response = (
                            "<tool_call>\n"
                            + json.dumps(forced_call, ensure_ascii=True)
                            + "\n</tool_call>"
                        )
                        latent_short_circuit = True
                        logger.info(
                            "subagent.latent_tool_short_circuit name=%s tool=%s confidence=%.3f",
                            self.name,
                            getattr(latent_signal, "tool_name", ""),
                            float(getattr(latent_signal, "confidence", 0.0)),
                        )
                        latent_tool_signals.append(
                            {
                                "step": int(step_count),
                                "tool": str(getattr(latent_signal, "tool_name", "")),
                                "confidence": float(
                                    getattr(latent_signal, "confidence", 0.0)
                                ),
                            }
                        )

                try:
                    if not latent_short_circuit:
                        full_response = self._stream_response(
                            chat_messages, assistant_prefix="<thinking>" + recovery_hint
                        )
                except Exception as e:
                    self.console.print(f"[red]Inference error: {e}[/red]")
                    break

                isolated_history.add_message("assistant", full_response)
                final_response = full_response

                response_fingerprint = hashlib.sha1(
                    full_response[:2000].encode("utf-8", errors="ignore")
                ).hexdigest()[:12]
                duplicate_fingerprints[response_fingerprint] = (
                    duplicate_fingerprints.get(response_fingerprint, 0) + 1
                )
                fingerprint_hits = duplicate_fingerprints[response_fingerprint]
                if fingerprint_hits > max_duplicate_threshold:
                    logger.warning(
                        "subagent.duplicate_output name=%s hash=%s hits=%d",
                        self.name,
                        response_fingerprint,
                        fingerprint_hits,
                    )
                    self.console.print(
                        "[bold red]⚠ Detected repetitive output. Breaking loop.[/bold red]"
                    )
                    break

                tool_calls = self._extract_tool_calls(full_response)
                is_done = self._check_if_done(full_response)
                logger.info(
                    "subagent.step.parse name=%s step=%d tool_calls=%d is_done=%s fingerprint=%s hits=%d",
                    self.name,
                    step_count,
                    len(tool_calls),
                    is_done,
                    response_fingerprint,
                    fingerprint_hits,
                )

                # NEW: Detect when model outputs natural language tool descriptions instead of actual calls
                if not tool_calls and step_count == 1:
                    # Check for common patterns of describing tools instead of calling them
                    nl_tool_patterns = [
                        r"\d+\.\s*[`'\"]?(saguaro_query|skeleton|slice|read_file)[`'\"]?",
                        r"(?:first|let's|i will|i'll)\s+(?:use|call|run)\s+[`'\"]?(saguaro_query|skeleton|slice|read_file)",
                        r"(?:use|call|run)\s+the?\s+[`'\"]?(saguaro_query|skeleton|slice|read_file)[`'\"]?\s+(?:tool|command)",
                    ]
                    response_lower = full_response.lower()
                    is_nl_description = any(
                        re.search(p, response_lower) for p in nl_tool_patterns
                    )

                    if is_nl_description:
                        logger.warning(
                            "Model output natural language tool description instead of tool call - injecting retry hint"
                        )
                        self.console.print(
                            "[yellow]⚠ Model described tools in text instead of calling them. Retrying with format enforcement...[/yellow]"
                        )

                        # Add a strong format reminder to history
                        retry_hint = """
[SYSTEM ERROR] Your previous response described tools in natural language instead of calling them.

You MUST use this EXACT XML format to call tools:

<tool_call>
{"name": "saguaro_query", "arguments": {"query": "relevant code for the task", "k": 5}}
</tool_call>

DO NOT describe what you will do. CALL THE TOOL NOW using the <tool_call> tags.
"""
                        isolated_history.add_message("tool", retry_hint)
                        continue  # Retry with the hint added

                # FAIL-SAFE: If step 1 still produces no tool call, force a sensible default
                if not tool_calls and step_count == 1:
                    forced_call = self._default_first_tool_call(strategy=tool_strategy)
                    forced_tool = forced_call.get("name", "saguaro_query")
                    logger.warning(
                        "Step 1 produced no tool calls - forcing default tool=%s strategy=%s query=%s features=%s",
                        forced_tool,
                        tool_strategy or "default",
                        forced_call.get("arguments", {}).get("query", ""),
                        getattr(self, "_last_default_tool_features", {}),
                    )
                    self.console.print(
                        f"[yellow]⚠ Forcing default tool call: {forced_tool}[/yellow]"
                    )
                    tool_calls = [forced_call]

                if not tool_calls:
                    consecutive_no_tool_steps += 1
                    if self._result_adapter is not None and full_response:
                        try:
                            latent_state = self._result_adapter.inject(
                                tool_result=full_response,
                                pre_tool_state=latent_state,
                                tool_name="assistant_response",
                            )
                            self._latent_state = latent_state
                            latent_reinjections += 1
                        except Exception as exc:
                            logger.debug(
                                "Latent assistant re-injection skipped: %s", exc
                            )
                    self._publish_progress(
                        event="no_tool_call",
                        payload={
                            "step": step_count,
                            "consecutive_no_tool_steps": consecutive_no_tool_steps,
                        },
                    )
                    if self.message_bus is not None and consecutive_no_tool_steps >= 2:
                        try:
                            self.message_bus.send(
                                sender=self.name,
                                recipient=self.parent_name + ":master",
                                message_type=MessageType.COORDINATION,
                                payload={
                                    "guidance": (
                                        "Need clarification: model is not emitting tool calls. "
                                        "Please reinforce concrete tool usage."
                                    )
                                },
                                priority=Priority.HIGH,
                            )
                        except Exception:
                            pass
                    if is_done or consecutive_no_tool_steps >= 3:
                        break
                    continue

                consecutive_no_tool_steps = 0

                for tool_call in tool_calls:
                    name = tool_call["name"]
                    self._publish_progress(
                        event="tool_start",
                        payload={"step": step_count, "tool": name},
                    )
                    tool_stats[name] = tool_stats.get(name, 0) + 1
                    trace_entry = {
                        "step": step_count,
                        "tool": name,
                        "is_saguaro": name in SAGUARO_TOOL_NAMES,
                        "timestamp": time.time(),
                        "status": "pending",
                    }
                    tool_trace.append(trace_entry)
                    if name == "delegate_to_subagent":
                        result = self._handle_delegation(tool_call, isolated_history)
                    else:
                        result = self._execute_tool(tool_call)
                    trace_entry["status"] = (
                        "error" if self._is_tool_execution_error(result) else "ok"
                    )
                    trace_entry["result_preview"] = str(result)[:200]
                    if trace_entry["is_saguaro"] and trace_entry["status"] == "error":
                        fallback_mode = "fallback_static_scan"
                        trace_entry["failure_tag"] = "saguaro_failure"
                        trace_entry["fallback_mode"] = fallback_mode
                        failure_record = {
                            "step": int(step_count),
                            "tool": name,
                            "error": str(result)[:500],
                            "failure_tag": "saguaro_failure",
                            "fallback_mode": fallback_mode,
                        }
                        saguaro_failures.append(failure_record)
                        logger.warning(
                            "subagent.saguaro_failure name=%s step=%d tool=%s fallback_mode=%s error=%s",
                            self.name,
                            step_count,
                            name,
                            fallback_mode,
                            str(result)[:160],
                        )
                    if self._result_adapter is not None:
                        try:
                            latent_state = self._result_adapter.inject(
                                tool_result=str(result),
                                pre_tool_state=latent_state,
                                tool_name=name,
                            )
                            self._latent_state = latent_state
                            latent_reinjections += 1
                        except Exception as exc:
                            logger.debug(
                                "Latent tool result re-injection skipped: %s", exc
                            )

                    if self.output_format != "json":
                        from rich.panel import Panel

                        full_visibility_tools = {
                            "read_file",
                            "read_files",
                            "slice",
                            "skeleton",
                            "query",
                            "saguaro_query",
                        }
                        should_truncate_display = (
                            name not in full_visibility_tools and len(result) > 10000
                        )
                        display_result = (
                            result[:10000]
                            + "\n\n[bold yellow]... (output truncated for brevity) ...[/bold yellow]"
                            if should_truncate_display
                            else result
                        )
                        self.console.print(
                            Panel(
                                display_result,
                                title=f"Tool Output: {name}",
                                border_style="green",
                                expand=False,
                            )
                        )

                    self._record_tool_result_message(
                        name,
                        tool_call.get("arguments", {}),
                        result,
                    )
                    self._publish_progress(
                        event="tool_end",
                        payload={
                            "step": step_count,
                            "tool": name,
                            "result_preview": str(result)[:180],
                        },
                    )
                    self._post_shared_finding(
                        step_count=step_count,
                        tool=name,
                        result=result,
                    )

            duration = (time.time() - start_time) * 1000
            from core.response_utils import clean_response

            cleaned_response = clean_response(final_response)
            # Add a machine-readable progress signal for master agent monitoring
            progress_marker = f"\n\n[SUBAGENT_PROGRESS: {step_count} steps, {len(self.files_read)} files]"
            cleaned_response += progress_marker
            saguaro_ratio = sum(1 for t in tool_trace if t["is_saguaro"]) / max(
                len(tool_trace), 1
            )
            if saguaro_ratio < 0.6:
                logger.warning(
                    "SubAgent %s Saguaro usage ratio below threshold: %.0f%%",
                    self.name,
                    saguaro_ratio * 100.0,
                )
            self._publish_progress(
                event="completed",
                payload={
                    "steps": step_count,
                    "files_read": len(self.files_read),
                    "saguaro_ratio": saguaro_ratio,
                },
            )

            serialized_latent = self._serialize_latent_state(self._latent_state)
            latent_payload = {
                "state": serialized_latent,
                "state_dim": len(serialized_latent) if serialized_latent else 0,
                "reinjections": int(latent_reinjections),
                "tool_signals": latent_tool_signals,
                "depth_used": int(self._dynamic_coconut_depth),
                "seeded_from_master": self._coconut_context_vector is not None,
            }
            evidence_envelope.update(
                {
                    "fallback_mode": fallback_mode,
                    "saguaro_failures": saguaro_failures,
                    "tool_trace_count": len(tool_trace),
                    "files_read_count": len(self.files_read),
                }
            )
            return {
                "response": cleaned_response,
                "stats": {
                    "steps": step_count,
                    "tool_calls": tool_stats,
                    "duration_ms": int(duration),
                    "tool_trace": tool_trace,
                    "saguaro_ratio": saguaro_ratio,
                    "duplicate_fingerprints": duplicate_fingerprints,
                    "fallback_mode": fallback_mode,
                    "saguaro_failures": saguaro_failures,
                },
                "files_read": list(self.files_read),
                "latent": latent_payload,
                "latent_state": serialized_latent,
                "latent_tool_signals": latent_tool_signals,
                "latent_reinjections": int(latent_reinjections),
                "evidence_envelope": evidence_envelope,
                "error": None,
            }

        finally:
            self.history = parent_history

    def run(self, mission_override: Optional[str] = None, **kwargs) -> Any:
        if self.quiet:
            self.console.quiet = True
        try:
            self.console.print(
                Panel(
                    f"[bold magenta]SubAgent Launched[/bold magenta]\nClass: {self.__class__.__name__}\nTask: {self.task}",
                    border_style="magenta",
                    title="Delegated Mission",
                )
            )
            mission_input = mission_override or self.task
            logger.info(
                f"SubAgent mission start: {self.name} | Task: {self.task[:100]}..."
            )
            runtime_prompt_profile = kwargs.pop("prompt_profile", None)
            if runtime_prompt_profile is not None:
                self.prompt_profile = str(runtime_prompt_profile).strip() or "default"
            runtime_prompt_key = kwargs.pop("specialist_prompt_key", None)
            if runtime_prompt_key is not None:
                self.specialist_prompt_key = str(runtime_prompt_key).strip()
            runtime_sovereign_enabled = kwargs.pop(
                "sovereign_build_policy_enabled", None
            )
            if runtime_sovereign_enabled is not None:
                self.sovereign_build_policy_enabled = bool(runtime_sovereign_enabled)
            runtime_sovereign_block = kwargs.pop("sovereign_build_policy_block", None)
            if runtime_sovereign_block:
                self._instance_sovereign_policy_block = str(
                    runtime_sovereign_block
                ).strip()
            runtime_prompt_injection = kwargs.pop("prompt_injection", None)
            if runtime_prompt_injection:
                self._append_prompt_injection(runtime_prompt_injection)
            if kwargs:
                context_str = "\n".join([f"- {k}: {v}" for k, v in kwargs.items()])
                mission_input = f"Context:\n{context_str}\n\nMission: {mission_input}"

            result = self._isolated_inference(mission_input)
            response_text = (
                result.get("response", "") if isinstance(result, dict) else str(result)
            )
            summary = self._extract_summary(response_text)

            logger.info(
                f"SubAgent mission complete: {self.name} | Steps: {result.get('stats', {}).get('steps', 0)}"
            )
            return {
                "summary": summary,
                "full_response": response_text,
                "iterations": (
                    result.get("stats", {}).get("steps", 0)
                    if isinstance(result, dict)
                    else 0
                ),
                "files_read": (
                    result.get("files_read", []) if isinstance(result, dict) else []
                ),
                "latent": (
                    result.get("latent", {}) if isinstance(result, dict) else {}
                ),
                "latent_state": (
                    result.get("latent_state") if isinstance(result, dict) else None
                ),
                "latent_tool_signals": (
                    result.get("latent_tool_signals", [])
                    if isinstance(result, dict)
                    else []
                ),
                "latent_reinjections": (
                    int(result.get("latent_reinjections", 0))
                    if isinstance(result, dict)
                    else 0
                ),
                "evidence_envelope": (
                    result.get(
                        "evidence_envelope",
                        self._build_common_evidence_envelope_defaults(),
                    )
                    if isinstance(result, dict)
                    else self._build_common_evidence_envelope_defaults()
                ),
                "prompt_profile": self.prompt_profile,
                "specialist_prompt_key": self.specialist_prompt_key,
                "sovereign_build_policy_enabled": bool(
                    self.sovereign_build_policy_enabled
                ),
            }
        finally:
            if self.quiet:
                self.console.quiet = self._original_quiet

    def _handle_delegation(self, tool_call: Dict, history) -> str:
        args = tool_call.get("arguments", {})
        subagent_type = args.get("subagent_type", "researcher")
        task = args.get("task", "Investigate further")

        from rich.tree import Tree

        delegation_tree = Tree(f"[cyan]{self.__class__.__name__}[/cyan]")
        delegation_tree.add(f"[yellow]↳ {subagent_type.title()}Subagent[/yellow]").add(
            f"[dim]{task[:50]}...[/dim]"
        )
        self.console.print(
            Panel(delegation_tree, title="Delegation", border_style="magenta")
        )

        from core.agents.specialists import (
            SpecialistRegistry,
            build_specialist_subagent,
            route_specialist,
        )

        registry = SpecialistRegistry()
        hint_domains = set(self.domain_detector.detect_from_description(task))
        hint_domains.add("research")
        routing = route_specialist(
            registry=registry,
            objective=task,
            requested_role=str(subagent_type or ""),
            aal="AAL-3",
            domains=sorted(hint_domains),
            question_type="research",
            repo_roles=["analysis_local"],
        )
        selected_role = routing.primary_role
        prompt_key = registry.prompt_key_for_role(selected_role)
        try:
            agent = build_specialist_subagent(
                role=selected_role,
                task=task,
                parent_name=self.name,
                brain=self.brain,
                console=self.console,
                parent_agent=self,
                message_bus=self.message_bus,
                complexity_profile=self.complexity_profile,
                context_budget=max(8000, int(self.context_budget * 0.8)),
                coconut_context_vector=self._latent_state,
                coconut_depth=self._dynamic_coconut_depth,
                prompt_profile=self.prompt_profile,
                specialist_prompt_key=prompt_key or self.specialist_prompt_key,
                sovereign_build_policy_block=self._resolve_sovereign_policy_block(),
                sovereign_build_policy_enabled=self.sovereign_build_policy_enabled,
                prompt_injection=(
                    f"Delegated from {self.__class__.__name__} via subagent_type='{subagent_type}'.\n"
                    f"Routing reasons: {', '.join(routing.reasons) if routing.reasons else 'none'}."
                ),
            )
            logger.info(f"Delegating to {selected_role}: {task[:50]}...")
            res = agent.run()
            return res.get("summary", str(res)) if isinstance(res, dict) else str(res)
        except Exception as e:
            return f"Delegation failed: {str(e)}"

    def _check_if_done(self, response: str) -> bool:
        markers = [
            "task complete",
            "mission complete",
            "analysis complete",
            "final answer:",
        ]
        return any(m in response.lower() for m in markers)

    def _extract_summary(self, content: str) -> str:
        if not content:
            return ""
        cleaned = ThinkingParser.remove_thinking_blocks(content)
        cleaned = re.sub(r"<tool_call>.*?</tool_call>", "", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r"\[SYSTEM:\s*Streaming loop terminated\.\]", "", cleaned)
        cleaned = re.sub(r"\[SYSTEM:\s*Thinking loop terminated\.\]", "", cleaned)
        cleaned = re.sub(r"\[SUBAGENT_PROGRESS:[^\]]+\]", "", cleaned)
        cleaned = re.sub(
            r'^\s*\{\s*"name"\s*:\s*"[a-zA-Z0-9_]+"\s*,\s*"arguments"\s*:\s*\{.*\}\s*\}\s*$',
            "",
            cleaned,
            flags=re.DOTALL,
        )
        from core.response_utils import clean_response

        cleaned = clean_response(cleaned).strip()
        return (
            cleaned if len(cleaned) > 10 else "Analysis completed (no summary output)."
        )

    def _publish_progress(
        self, event: str, payload: Optional[Dict[str, Any]] = None
    ) -> None:
        if self.message_bus is None:
            return
        try:
            enriched = {"agent": self.name, "event": event}
            if payload:
                enriched.update(payload)
            self.message_bus.publish(
                topic="progress",
                sender=self.name,
                payload=enriched,
                priority=Priority.NORMAL,
                metadata={
                    "run_id": getattr(self, "current_mission_id", None)
                    or getattr(self.parent_agent, "current_mission_id", None),
                },
            )
        except Exception:
            pass

    def _post_shared_finding(self, step_count: int, tool: str, result: str) -> None:
        if self.message_bus is None:
            return
        if tool not in {
            "saguaro_query",
            "skeleton",
            "slice",
            "read_file",
            "read_files",
            "grep_search",
            "find_by_name",
        }:
            return
        payload = {
            "step": step_count,
            "tool": tool,
            "preview": str(result)[:240],
            "timestamp": time.time(),
        }
        try:
            self.message_bus.set_shared_context(f"scratchpad.{self.name}", payload)
        except Exception:
            pass

    def _consume_master_guidance(self) -> str:
        if self.message_bus is None:
            return ""
        guidance: List[str] = []
        while True:
            msg = self.message_bus.receive(self.name)
            if msg is None:
                break
            if msg.message_type in {
                MessageType.COORDINATION,
                MessageType.REQUEST,
                MessageType.RESPONSE,
            }:
                text = str((msg.payload or {}).get("guidance", "")).strip()
                if text:
                    guidance.append(text)
        if not guidance:
            shared = self.message_bus.get_shared_context(f"guidance.{self.name}")
            if isinstance(shared, str) and shared.strip():
                guidance.append(shared.strip())
        return " | ".join(guidance[:2])

    def _execute_tool(self, tool_call: dict, retries: int = 3, delay: int = 2) -> str:
        name = tool_call.get("name")
        args = tool_call.get("arguments", {})

        # Track file reads for synthesis visibility
        if name in ["read_file", "skeleton", "slice", "view_file", "inspect_file"]:
            path = args.get("file_path") or args.get("path") or args.get("AbsolutePath")
            if path:
                self.files_read.add(path)
        elif name == "read_files":
            for path in args.get("paths", []) or []:
                if path:
                    self.files_read.add(path)

        write_targets = self._extract_write_targets(name, args)
        if self.ownership_registry is not None and write_targets:
            for target in write_targets:
                decision = self.ownership_registry.can_access(
                    agent_id=self.name,
                    file_path=target,
                    access_type="write",
                )
                if not decision.allowed:
                    return self._handle_access_denied(target, decision)
            self._owned_files.update(write_targets)

        return super()._execute_tool(tool_call, retries, delay)

    def _handle_access_denied(self, path: str, decision: Any) -> str:
        reason = getattr(decision, "reason", "access_denied")
        owner = getattr(
            getattr(decision, "owner_info", None), "owner_agent_id", "unknown"
        )
        message = f"WRITE_ACCESS_DENIED path={path} owner={owner} reason={reason}"
        logger.warning("%s", message)

        if self.message_bus is not None:
            try:
                guidance = (
                    "Ownership conflict detected. "
                    f"Path '{path}' owned by '{owner}'. Reason: {reason}."
                )
                self.message_bus.send(
                    sender=self.name,
                    recipient=self.parent_name + ":master",
                    message_type=MessageType.OWNERSHIP_DENIED,
                    payload={
                        "path": path,
                        "owner": owner,
                        "reason": reason,
                        "can_negotiate": bool(
                            getattr(decision, "can_negotiate", False)
                        ),
                        "guidance": guidance,
                    },
                    priority=Priority.HIGH,
                    metadata={
                        "run_id": getattr(self, "current_mission_id", None)
                        or getattr(self.parent_agent, "current_mission_id", None),
                        "files": [path],
                    },
                )
            except Exception:
                pass

        if bool(getattr(decision, "can_negotiate", False)):
            return message + " negotiable=true"
        return message
