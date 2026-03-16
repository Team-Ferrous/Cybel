from core.context_compression import inject_context_updates_into_all

TOOL_SCHEMAS = {
    "tools": [
        {
            "name": "list_dir",
            "description": "List contents of a directory. Supports recursive listing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to list",
                        "default": ".",
                    },
                    "recursive": {"type": "boolean", "default": False},
                    "max_depth": {"type": "integer"},
                    "filter_noise": {
                        "type": "boolean",
                        "default": True,
                        "description": "If true, filters out noisy directories like venv, .git, etc.",
                    },
                },
            },
        },
        {
            "name": "delete_file",
            "description": "Delete a file safely (maintains a backup).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path of file to delete"}
                },
                "required": ["path"],
            },
        },
        {
            "name": "move_file",
            "description": "Move or rename a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "src": {"type": "string", "description": "Source path"},
                    "dst": {"type": "string", "description": "Destination path"},
                },
                "required": ["src", "dst"],
            },
        },
        {
            "name": "list_backups",
            "description": "List available backups for a file (newest first).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Target file path"},
                    "max_items": {"type": "integer", "default": 20},
                },
                "required": ["path"],
            },
        },
        {
            "name": "rollback_file",
            "description": "Restore a file from backup (latest by default).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Target file path"},
                    "backup_path": {
                        "type": "string",
                        "description": "Optional backup file path to restore from",
                    },
                },
                "required": ["path"],
            },
        },
        {
            "name": "read_file",
            "description": "Read file contents. Returns full text by default. Use start_line/end_line for targeted windows and include_line_numbers for easier review.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative file path",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Optional start line (1-indexed)",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Optional end line (inclusive)",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Optional hard character cap for raw file read",
                    },
                    "include_line_numbers": {
                        "type": "boolean",
                        "default": False,
                        "description": "Prefix output lines with 1-indexed line numbers",
                    },
                },
                "required": ["path"],
            },
        },
        {
            "name": "write_file",
            "description": "Write or overwrite a file with given content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
        {
            "name": "edit_file",
            "description": "Apply line-based edits to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "edits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "start_line": {"type": "integer"},
                                "end_line": {"type": "integer"},
                                "new_content": {"type": "string"},
                            },
                        },
                    },
                },
                "required": ["path", "edits"],
            },
        },
        {
            "name": "search_code",
            "description": "Semantic search for code. Uses Saguaro query search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query",
                    },
                    "k": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
        {
            "name": "run_command",
            "description": "Execute a shell command and return output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "cwd": {"type": "string", "description": "Working directory"},
                    "use_docker": {
                        "type": "boolean",
                        "default": False,
                        "description": "Run command inside a secure Docker container",
                    },
                    "max_runtime": {
                        "type": "integer",
                        "description": "Optional max runtime in seconds for this command",
                    },
                },
                "required": ["command"],
            },
        },
        {
            "name": "saguaro_query",
            "description": "SAGUARO CORE: Semantic code discovery. Always use this first to locate relevant files, modules, classes, and functions by meaning before reading code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query describing the code or behavior to find",
                    },
                    "k": {"type": "integer", "default": 5},
                    "scope": {
                        "type": "string",
                        "enum": ["local", "workspace", "peer", "global"],
                        "default": "global",
                    },
                    "dedupe_by": {
                        "type": "string",
                        "enum": ["entity", "path", "symbol"],
                        "default": "entity",
                    },
                    "recall": {
                        "type": "string",
                        "enum": ["fast", "balanced", "high", "exhaustive"],
                        "default": "balanced",
                    },
                    "breadth": {
                        "type": "integer",
                        "description": "Resolved candidate breadth budget for ANN and rerank stages",
                    },
                    "score_threshold": {
                        "type": "number",
                        "description": "Optional minimum final ranking score",
                    },
                    "stale_file_bias": {
                        "type": "number",
                        "description": "Bias stale-file handling from -1.0 to 1.0",
                    },
                    "cost_budget": {
                        "type": "string",
                        "enum": ["cheap", "balanced", "generous"],
                        "default": "balanced",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "skeleton",
            "description": "SAGUARO CORE: Get lightning-fast file structure (classes, functions) without full content. Use this as your PRIMARY tool for orientation in large files. Far more efficient than read_file for understanding API surfaces.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the file",
                    }
                },
                "required": ["path"],
            },
        },
        {
            "name": "slice",
            "description": "SAGUARO CORE: Extract a specific function or class implementation plus its immediate local dependencies. Use this for deep code reading once you've identified a target via skeleton. Example: 'slice: core/agent.py.BaseAgent'",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Format: filename.Object",
                    }
                },
                "required": ["target"],
            },
        },
        {
            "name": "verify",
            "description": "Run Saguaro verification engines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "default": "."},
                    "engines": {
                        "type": "string",
                        "default": "native,ruff,semantic,aes",
                    },
                    "auto_fix": {"type": "boolean", "default": False},
                    "preflight_only": {"type": "boolean", "default": False},
                    "timeout_seconds": {"type": "number"},
                },
            },
        },
        {
            "name": "cpu_scan",
            "description": "Run Saguaro's static CPU hotspot scan over one file or directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File or directory to scan",
                        "default": ".",
                    },
                    "arch": {
                        "type": "string",
                        "enum": ["x86_64-avx2", "x86_64-avx512", "arm64-neon"],
                        "default": "x86_64-avx2",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                        "description": "Maximum number of hotspots to return",
                    },
                },
            },
        },
        {
            "name": "deadcode",
            "description": "Run Saguaro dead-code analysis and return candidates as text or JSON.",
            "parameters": {
                "type": "object",
                "properties": {
                    "threshold": {
                        "type": "number",
                        "default": 0.5,
                        "description": "Minimum confidence threshold for candidates.",
                    },
                    "low_usage_max_refs": {
                        "type": "integer",
                        "default": 1,
                        "description": "Max static refs for deadcode low-usage side report.",
                    },
                    "lang": {
                        "type": "string",
                        "description": "Compatibility language selector.",
                    },
                    "evidence": {"type": "boolean", "default": False},
                    "runtime_observed": {"type": "boolean", "default": False},
                    "explain": {"type": "boolean", "default": False},
                    "output_format": {
                        "type": "string",
                        "enum": ["text", "json"],
                        "default": "json",
                    },
                },
            },
        },
        {
            "name": "low_usage",
            "description": "Report reachable symbols with very low static usage counts to identify DRY/refactor opportunities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_refs": {
                        "type": "integer",
                        "default": 1,
                        "description": "Maximum static reference count to classify as low-usage.",
                    },
                    "include_tests": {
                        "type": "boolean",
                        "default": False,
                    },
                    "path": {
                        "type": "string",
                        "description": "Optional path prefix to focus the report on one subsystem.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Optional maximum number of candidates to return.",
                    },
                    "output_format": {
                        "type": "string",
                        "enum": ["text", "json"],
                        "default": "json",
                    },
                },
            },
        },
        {
            "name": "unwired",
            "description": "Detect isolated unreachable feature clusters from entrypoints.",
            "parameters": {
                "type": "object",
                "properties": {
                    "threshold": {
                        "type": "number",
                        "default": 0.55,
                        "description": "Minimum confidence threshold for returned clusters.",
                    },
                    "min_nodes": {
                        "type": "integer",
                        "default": 4,
                        "description": "Minimum nodes for unwired_feature classification.",
                    },
                    "min_files": {
                        "type": "integer",
                        "default": 2,
                        "description": "Minimum files for unwired_feature classification.",
                    },
                    "include_tests": {
                        "type": "boolean",
                        "default": False,
                    },
                    "include_fragments": {
                        "type": "boolean",
                        "default": False,
                    },
                    "max_clusters": {
                        "type": "integer",
                        "default": 20,
                    },
                    "refresh_graph": {
                        "type": "boolean",
                        "default": True,
                        "description": "Refresh graph incrementally before analysis.",
                    },
                    "output_format": {
                        "type": "string",
                        "enum": ["text", "json"],
                        "default": "json",
                    },
                },
            },
        },
        {
            "name": "saguaro_sync",
            "description": "Incrementally sync Saguaro index/graph after file changes so downstream agents read fresh semantics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "index",
                            "serve",
                            "peer-add",
                            "peer-remove",
                            "peer-list",
                            "push",
                            "pull",
                            "subscribe",
                        ],
                        "default": "index",
                    },
                    "changed_files": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "deleted_files": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "full": {"type": "boolean", "default": False},
                    "reason": {"type": "string", "default": "tool_call"},
                    "peer_id": {"type": "string"},
                    "peer_name": {"type": "string"},
                    "peer_url": {"type": "string"},
                    "auth_token": {"type": "string"},
                    "bundle_path": {"type": "string"},
                    "workspace_id": {"type": "string"},
                    "limit": {"type": "integer", "default": 1000},
                },
            },
        },
        {
            "name": "saguaro_workspace",
            "description": "Inspect tracked-vs-working-tree status like a semantic worktree overlay.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "status",
                            "scan",
                            "sync",
                            "list",
                            "create",
                            "switch",
                            "history",
                            "diff",
                            "snapshot",
                        ],
                        "default": "status",
                    },
                    "limit": {"type": "integer", "default": 200},
                    "name": {"type": "string"},
                    "workspace_id": {"type": "string"},
                    "against": {"type": "string", "default": "main"},
                    "description": {"type": "string", "default": ""},
                    "switch": {"type": "boolean", "default": False},
                    "label": {"type": "string", "default": "manual"},
                },
            },
        },
        {
            "name": "saguaro_daemon",
            "description": "Manage the background Saguaro watcher daemon for automatic indexing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["start", "stop", "restart", "status", "logs"],
                        "default": "status",
                    },
                    "interval": {"type": "integer", "default": 5},
                    "lines": {"type": "integer", "default": 200},
                },
            },
        },
        {
            "name": "saguaro_doctor",
            "description": "Run one-shot diagnostics across backend, ABI compatibility, parser coverage, duplicate trees, and freshness.",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "report",
            "description": "SAGUARO SPECIAL: Generate a multi-page high-fidelity 'State of the Repo' report. Use this for broad architectural discovery or when entering a completely new workspace.",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "memory",
            "description": "Access agentic memory tiers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["list", "read", "write"]},
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                    "tier": {"type": "string", "default": "working"},
                },
                "required": ["action"],
            },
        },
        {
            "name": "export_audit",
            "description": "Export immutable audit bundle for current session.",
            "parameters": {
                "type": "object",
                "properties": {
                    "output_path": {
                        "type": "string",
                        "default": ".anvil/audit_export.json",
                    }
                },
            },
        },
        {
            "name": "upgrade",
            "description": "Check for updates or self-upgrade the agent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["check", "perform"],
                        "default": "check",
                    }
                },
            },
        },
        {
            "name": "delegate",
            "description": "Delegate a complex sub-task to a specialized sub-agent. Use this for research, analysis, or isolated implementation steps.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The specific task description for the sub-agent.",
                    },
                    "quiet": {
                        "type": "boolean",
                        "description": "Whether the sub-agent should run silently without printing to the console.",
                        "default": False,
                    },
                },
                "required": ["task"],
            },
        },
        {
            "name": "activate_skill",
            "description": "Activate a registered skill by loading its instructions and context. Use this to enable specialized capabilities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "Name of the skill to activate (e.g. 'security-auditor')",
                    }
                },
                "required": ["skill_name"],
            },
        },
        {
            "name": "web_search",
            "description": "Canonical specialist research search tool. Use for broad external discovery across docs, standards, and technical references.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {
                        "type": "integer",
                        "default": 5,
                        "description": "Number of results to return",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "read_files",
            "description": "Read multiple files at once. Returns full content for each file by default.",
            "parameters": {
                "type": "object",
                "properties": {
                    "paths": {"type": "array", "items": {"type": "string"}},
                    "start_line": {
                        "type": "integer",
                        "description": "Optional start line (1-indexed) for each file",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Optional end line (inclusive) for each file",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Optional hard character cap per file",
                    },
                    "include_line_numbers": {
                        "type": "boolean",
                        "default": False,
                        "description": "Prefix output lines with 1-indexed line numbers",
                    },
                },
                "required": ["paths"],
            },
        },
        {
            "name": "write_files",
            "description": "Write multiple files at once.",
            "parameters": {
                "type": "object",
                "properties": {
                    "files": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                    }
                },
                "required": ["files"],
            },
        },
        {
            "name": "apply_patch",
            "description": "Apply a unified diff patch to a file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "patch": {"type": "string"}},
                "required": ["path", "patch"],
            },
        },
        {
            "name": "grep",
            "description": "Search for a pattern in files. Uses ripgrep if available.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string", "default": "."},
                    "is_regex": {"type": "boolean", "default": False},
                    "file_pattern": {"type": "string", "default": "*"},
                },
                "required": ["pattern"],
            },
        },
        {
            "name": "grep_search",
            "description": "Literal/regex code search fallback when semantic search returns too few results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string", "default": "."},
                    "is_regex": {"type": "boolean", "default": False},
                    "file_pattern": {"type": "string", "default": "*.py"},
                },
                "required": ["pattern"],
            },
        },
        {
            "name": "find_by_name",
            "description": "Find files by filename glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string", "default": "."},
                    "recursive": {"type": "boolean", "default": True},
                },
                "required": ["pattern"],
            },
        },
        {
            "name": "glob",
            "description": "Find files matching a pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string", "default": "."},
                    "recursive": {"type": "boolean", "default": True},
                },
                "required": ["pattern"],
            },
        },
        {
            "name": "web_fetch",
            "description": "Canonical specialist research fetch tool. Retrieve a specific URL after discovery and return normalized markdown content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"}
                },
                "required": ["url"],
            },
        },
        {
            "name": "think",
            "description": "Pause to think and reflect on the current task progress. Use when uncertain, after unexpected results, or before complex decisions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": "What the agent is thinking about or reasoning through",
                    },
                    "question": {
                        "type": "string",
                        "description": "Optional question to answer through this thinking",
                    },
                    "type": {
                        "type": "string",
                        "enum": [
                            "understanding",
                            "planning",
                            "reasoning",
                            "reflection",
                            "correction",
                        ],
                        "default": "reasoning",
                        "description": "Type of thinking being performed",
                    },
                },
                "required": ["thought"],
            },
        },
        {
            "name": "notify_user",
            "description": "Send a message to the user. Use for updates, requesting review, or asking questions. This is the only way to communicate during active tasks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message to send to the user",
                    },
                    "paths_to_review": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "File paths that the user should review",
                    },
                    "blocked_on_user": {
                        "type": "boolean",
                        "default": False,
                        "description": "If true, agent will wait for user response before continuing",
                    },
                    "notification_type": {
                        "type": "string",
                        "enum": ["info", "warning", "error", "success", "question"],
                        "default": "info",
                        "description": "Type of notification",
                    },
                },
                "required": ["message"],
            },
        },
        {
            "name": "verify_all",
            "description": "Run all verification tools (syntax, lint, types, tests) and return aggregated results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "default": ".",
                        "description": "Directory to verify",
                    }
                },
            },
        },
        {
            "name": "verify_syntax",
            "description": "Check Python files for syntax errors.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "default": ".",
                        "description": "Directory or file to check",
                    }
                },
            },
        },
        {
            "name": "verify_lint",
            "description": "Run linter (ruff/flake8) on the codebase.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "default": ".",
                        "description": "Directory to check",
                    }
                },
            },
        },
        {
            "name": "verify_types",
            "description": "Run type checker (mypy) on the codebase.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "default": ".",
                        "description": "Directory to check",
                    }
                },
            },
        },
        {
            "name": "run_tests",
            "description": "Execute test suite with pytest.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "default": ".",
                        "description": "Directory containing tests",
                    },
                    "verbose": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include verbose output",
                    },
                },
            },
        },
        {
            "name": "run_tests_suspended",
            "description": "Run test suite with automatic model suspension to free RAM. On CPU-only systems this evicts model weights (1-5+ GB) before running pytest and reloads them afterward.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "default": ".",
                        "description": "Directory containing tests",
                    },
                    "verbose": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include verbose output",
                    },
                    "force_suspend": {
                        "type": "boolean",
                        "default": True,
                        "description": "Always suspend model (True) or only when RAM is tight (False)",
                    },
                },
            },
        },
        {
            "name": "visualize",
            "description": "Generate a Mermaid diagram for a file or process.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File to visualize"},
                    "diagram_type": {
                        "type": "string",
                        "enum": ["class", "flow"],
                        "default": "class",
                    },
                },
                "required": ["path"],
            },
        },
        {
            "name": "semantic_search",
            "description": "Deprecated compatibility alias for `saguaro_query`. Routes to the same Saguaro semantic index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query",
                    },
                    "k": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
        {
            "name": "lsp_definition",
            "description": "Find the definition of a symbol (class, function, variable).",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "The symbol name to find",
                    }
                },
                "required": ["symbol"],
            },
        },
        {
            "name": "lsp_references",
            "description": "Find all usages of a symbol across the codebase.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "The symbol name to search for",
                    }
                },
                "required": ["symbol"],
            },
        },
        {
            "name": "lsp_diagnostics",
            "description": "Get lint, syntax, and type errors for a path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "default": ".",
                        "description": "Path to check",
                    }
                },
            },
        },
        {
            "name": "debug",
            "description": "Run a command with debugging enabled. Captures rich stack traces and local state on failure for self-healing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to run",
                    }
                },
                "required": ["command"],
            },
        },
        {
            "name": "browser_visit",
            "description": "Visit a website and read its text content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to visit"}
                },
                "required": ["url"],
            },
        },
        {
            "name": "browser_click",
            "description": "Simulate a click on a web element by its selector.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "CSS selector to click",
                    }
                },
                "required": ["selector"],
            },
        },
        {
            "name": "browser_screenshot",
            "description": "Simulate capturing a screenshot of the current page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name for the screenshot file",
                        "default": "screenshot",
                    }
                },
            },
        },
        {
            "name": "update_memory_bank",
            "description": "Updates the GRANITE.md project memory bank with new architecture, conventions, or lessons learned.",
            "parameters": {
                "type": "object",
                "properties": {
                    "section": {
                        "type": "string",
                        "enum": ["Architecture", "Conventions", "Lessons Learned"],
                        "description": "The section to update",
                    },
                    "content": {
                        "type": "string",
                        "description": "The information to add to the section",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["append", "overwrite"],
                        "default": "append",
                        "description": "Whether to append to or overwrite the section",
                    },
                },
                "required": ["section", "content"],
            },
        },
        {
            "name": "analyze_codebase",
            "description": "Trigger a semi-autonomous subagent to deep-research a specific topic in the codebase. Use this for complex questions that require tracing multiple files and dependencies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The research goal or question to answer.",
                    }
                },
                "required": ["topic"],
            },
        },
        {
            "name": "execute_subagent_task",
            "description": "Executes a specific, atomic task using a transient SubAgent with a designated role.",
            "parameters": {
                "type": "object",
                "properties": {
                    "role": {
                        "type": "string",
                        "description": "The role for the sub-agent (e.g., researcher, architect, implementer, validator).",
                    },
                    "task": {
                        "type": "string",
                        "description": "The specific and detailed instructions for the task.",
                    },
                    "aal": {
                        "type": "string",
                        "description": "Optional AES assurance level override for the subagent task.",
                    },
                    "domains": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional AES domain tags assigned to the task.",
                    },
                    "compliance": {
                        "type": "object",
                        "description": "Optional compliance context (trace_id, evidence_bundle_id, waiver_ids, red_team_required).",
                    },
                },
                "required": ["role", "task"],
            },
        },
        {
            "name": "search_arxiv",
            "description": "Primary specialist research tool for peer-reviewed and preprint paper discovery on arXiv.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for papers",
                    },
                    "max_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
        {
            "name": "fetch_arxiv_paper",
            "description": "Primary specialist research tool to fetch one arXiv paper by id with metadata and excerpt for evidence capture.",
            "parameters": {
                "type": "object",
                "properties": {
                    "arxiv_id": {
                        "type": "string",
                        "description": "The arXiv identifier, for example 2501.12345",
                    }
                },
                "required": ["arxiv_id"],
            },
        },
        {
            "name": "search_finance",
            "description": "Fetch real-time financial data and news for a ticker symbol (e.g., AAPL, TSLA, BTC-USD).",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "The stock or crypto ticker symbol",
                    }
                },
                "required": ["symbol"],
            },
        },
        {
            "name": "search_scholar",
            "description": "Search for academic papers and citations on Google Scholar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for scholarly articles",
                    },
                    "max_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
        {
            "name": "search_news",
            "description": "Search for recent news articles. Useful for staying current on events.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "News topic search query",
                    },
                    "max_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
        {
            "name": "search_reddit",
            "description": "Primary specialist research tool for community signal gathering from Reddit discussion threads.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Discussion topic to search for",
                    },
                    "subreddits": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional subreddit filter list",
                    },
                    "sort": {"type": "string", "default": "relevance"},
                    "time_filter": {"type": "string", "default": "year"},
                },
                "required": ["query"],
            },
        },
        {
            "name": "search_hackernews",
            "description": "Primary specialist research tool for engineering discussion discovery from Hacker News.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Discussion topic to search for",
                    },
                    "max_results": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
        },
        {
            "name": "search_stackoverflow",
            "description": "Primary specialist research tool for implementation patterns, diagnostics, and accepted fixes from Stack Overflow.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Question topic to search for",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional Stack Overflow tags",
                    },
                    "max_results": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
        },
        {
            "name": "search_patents",
            "description": "Search for patents on Google Patents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Patent search query"},
                    "max_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
        {
            "name": "subagent_research",
            "description": "DELEGATION: Spawns an elite Researcher subagent to perform comprehensive web and codebase research. Use this for open-ended questions or 'how-to' investigations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "research_query": {
                        "type": "string",
                        "description": "The specific research topic or question.",
                    }
                },
                "required": ["research_query"],
            },
        },
        {
            "name": "subagent_analyze",
            "description": "DELEGATION: Spawns an elite Repository Analysis subagent to map out architecture, dependencies, and critical paths. Use this for deep codebase understanding.",
            "parameters": {
                "type": "object",
                "properties": {
                    "root_dir": {
                        "type": "string",
                        "description": "Root directory to analyze.",
                        "default": ".",
                    }
                },
            },
        },
        {
            "name": "subagent_debug",
            "description": "DELEGATION: Spawns an elite Debugging subagent to perform root cause analysis and implement fixes for complex bugs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "error_log": {
                        "type": "string",
                        "description": "The error log or description of the bug.",
                    }
                },
                "required": ["error_log"],
            },
        },
        {
            "name": "subagent_implement",
            "description": "DELEGATION: Spawns an elite Implementation subagent to build new features or perform large-scale refactors based on a spec.",
            "parameters": {
                "type": "object",
                "properties": {
                    "spec": {
                        "type": "string",
                        "description": "The detailed implementation specification.",
                    }
                },
                "required": ["spec"],
            },
        },
        {
            "name": "saguaro_index",
            "description": "SAGUARO ADMIN: Force a fresh indexing of the workspace. Use this after significant file additions or deletions to ensure the semantic engine is sync'd.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "default": ".",
                        "description": "Root path to index.",
                    }
                },
            },
        },
    ]
}

TOOL_SCHEMAS["tools"] = inject_context_updates_into_all(TOOL_SCHEMAS["tools"])
