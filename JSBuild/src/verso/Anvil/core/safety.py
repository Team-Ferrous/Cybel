from urllib.parse import urlparse
from typing import List, Optional


class NetworkGuard:
    """
    Enforces network allow/block lists for the agent.
    """

    DEFAULT_ALLOW_LIST = [
        "docs.python.org",
        "github.com",
        "pypi.org",
        "stackoverflow.com",
        "npmjs.com",
        "duckduckgo.com",
    ]

    def __init__(self, allow_list: Optional[List[str]] = None):
        self.allow_list = allow_list or self.DEFAULT_ALLOW_LIST

    def is_allowed(self, url: str) -> bool:
        """Check if a URL is in the allow list."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if not domain:
                return False

            # Simple check: domain or subdomain
            for allowed in self.allow_list:
                if domain == allowed or domain.endswith("." + allowed):
                    return True
            return False
        except Exception:
            return False


_guard = NetworkGuard()


def check_url_safety(url: str) -> None:
    """Raises PermissionError if URL is not safe."""
    if not _guard.is_allowed(url):
        raise PermissionError(
            f"Access to '{url}' is restricted by network safety policy."
        )
