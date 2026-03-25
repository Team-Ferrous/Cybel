import requests

try:
    import html2text
except ImportError:  # pragma: no cover - optional dependency in test envs
    html2text = None
from core.safety import check_url_safety


def fetch_url(url: str) -> str:
    """
    Fetches content from a URL and converts it to Markdown.
    """
    try:
        check_url_safety(url)
    except PermissionError as pe:
        return str(pe)

    headers = {
        "User-Agent": "AnvilAgent/1.0 (AI Assistant; +https://github.com/project/anvil)"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # Check content type
        content_type = response.headers.get("Content-Type", "")

        if "text/html" in content_type:
            if html2text is None:
                return response.text
            converter = html2text.HTML2Text()
            converter.ignore_links = False
            converter.ignore_images = True
            converter.ignore_tables = False
            markdown = converter.handle(response.text)
            return markdown

        elif "text/plain" in content_type or "application/json" in content_type:
            return response.text

        else:
            return f"Unsupported content type: {content_type}. Content preview:\n{response.text[:500]}..."

    except requests.exceptions.RequestException as e:
        return f"Error fetching URL {url}: {str(e)}"
