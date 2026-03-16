"""
Specialized subagents for delegated tasks.

Each subagent:
- Shares the brain (DeterministicOllama) with master - no memory duplication
- Has its own 200k context window
- Specialized for specific task type
- Returns structured results to master
"""

from core.subagents.file_analyst import FileAnalysisSubagent

__all__ = ["FileAnalysisSubagent"]
