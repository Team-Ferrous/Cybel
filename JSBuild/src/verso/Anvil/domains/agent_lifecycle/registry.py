from typing import Dict, Any

# Categories: Core Development (8), Security (5), Code Intelligence (6), QA (5), Research (4), DevOps (4), Specialized (30+)

AGENT_REGISTRY = {
    # --- Core Development ---
    "planner": {
        "class": "PlannerSubagent",
        "tools": ["skeleton", "slice", "semantic_search", "grep"],
        "system_prompt": "You are a strategic planning specialist. You analyze requirements and decompose them into actionable steps.",
    },
    "coder": {
        "class": "CoderSubagent",
        "tools": ["read_file", "write_file", "edit_file", "apply_patch"],
        "system_prompt": "You are an elite coding specialist. Your focus is on writing clean, efficient, and robust code.",
    },
    "reviewer": {
        "class": "ReviewerSubagent",
        "tools": ["read_file", "skeleton", "analyze_codebase"],
        "system_prompt": "You are a senior code reviewer. You look for bugs, style violations, and architectural inconsistencies.",
    },
    "tester": {
        "class": "TesterSubagent",
        "tools": ["read_file", "write_file", "run_tests", "verify_syntax"],
        "system_prompt": "You are a quality assurance specialist. You design and implement tests to ensure code correctness.",
    },
    "debugger": {
        "class": "DebuggerSubagent",
        "tools": ["read_file", "grep", "debug", "run_command"],
        "system_prompt": "You are a debugging expert. You can track down the most elusive bugs and propose targeted fixes.",
    },
    # --- Code Intelligence ---
    "architect": {
        "class": "ArchitectSubagent",
        "tools": ["skeleton", "impact", "visualize"],
        "system_prompt": "You are a software architect. You analyze system structure and high-level component interactions.",
    },
    "security_auditor": {
        "class": "SecuritySubagent",
        "tools": ["read_file", "grep", "analyze_codebase"],
        "system_prompt": "You are a security expert. You scan code for vulnerabilities and security hotspots.",
    },
    # --- Specialized ---
    "frontend": {
        "class": "FrontendSubagent",
        "tools": ["read_file", "write_file", "browser_visit"],
        "system_prompt": "You are a frontend development expert specialized in UI/UX and web standards.",
    },
    "backend": {
        "class": "BackendSubagent",
        "tools": ["read_file", "write_file", "run_command"],
        "system_prompt": "You are a backend development expert focused on APIs, databases, and server-side logic.",
    },
    "documentation": {
        "class": "DocSubagent",
        "tools": ["read_file", "write_file", "list_dir"],
        "system_prompt": "You are a technical writer. You create clear, comprehensive, and helpful documentation.",
    },
}


def get_agent_config(agent_type: str) -> Dict[str, Any]:
    """Returns the configuration for a specific specialist agent type."""
    return AGENT_REGISTRY.get(
        agent_type,
        {
            "tools": [],
            "system_prompt": "You are a general-purpose autonomous subagent.",
        },
    )
