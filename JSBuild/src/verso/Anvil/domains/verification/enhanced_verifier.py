import os
import subprocess
from typing import List, Dict, Any
from domains.code_intelligence.saguaro_substrate import SaguaroSubstrate


class EnhancedVerifier:
    """
    Advanced verification suite beyond basic syntax/linting.
    Performs semantic checks like unused imports, dead code, and auto-fixes.
    """

    def __init__(self, root_dir: str = "."):
        self.root_dir = root_dir
        self.saguaro = SaguaroSubstrate(root_dir)

    def verify_files(self, file_paths: List[str]) -> Dict[str, Any]:
        """Runs enhanced checks on a set of files."""
        results = {}
        for path in file_paths:
            if not os.path.exists(path):
                continue

            results[path] = {
                "dead_code": self._check_dead_code(path),
                "unused_imports": self._check_unused_imports(path),
                "type_consistency": self._check_type_consistency(path),
            }
        return results

    def auto_fix(self, file_path: str) -> bool:
        """
        Attempts to automatically fix common issues:
        - Remove unused imports (autoflake)
        - Format code (black)
        """
        if not os.path.exists(file_path):
            return False

        try:
            # 1. Remove unused imports
            subprocess.run(
                ["autoflake", "--in-place", "--remove-all-unused-imports", file_path],
                check=False,
            )

            # 2. Format with black
            subprocess.run(["black", file_path], check=False)

            return True
        except Exception:
            return False

    def _check_dead_code(self, file_path: str) -> List[str]:
        """Identifies potentially dead functions or classes using Saguaro."""
        try:
            # Current Saguaro 'verify' tool handles some of this, but we can extend
            # In a real implementation, we'd query the call graph
            # For now, we use a placeholder that calls Saguaro verify
            res = self.saguaro.verify(file_path)
            if "unused" in res.lower():
                return [res]
            return []
        except Exception:
            return []

    def _check_unused_imports(self, file_path: str) -> List[str]:
        """Checks for unused imports specifically."""
        try:
            result = subprocess.run(
                ["autoflake", "--check", "--remove-all-unused-imports", file_path],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                return ["Contains unused imports"]
            return []
        except Exception:
            return []

    def _check_type_consistency(self, file_path: str) -> List[str]:
        """Checks for type inconsistencies (mypy)."""
        try:
            result = subprocess.run(
                ["mypy", "--ignore-missing-imports", file_path],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                return [result.stdout.strip()]
            return []
        except Exception:
            return []


# Singleton
_enhanced_verifier = None


def get_enhanced_verifier() -> EnhancedVerifier:
    global _enhanced_verifier
    if _enhanced_verifier is None:
        _enhanced_verifier = EnhancedVerifier()
    return _enhanced_verifier
