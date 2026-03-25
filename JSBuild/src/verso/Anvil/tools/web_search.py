import json
import requests

try:
    import lxml.html as lxml_html
except ImportError:  # pragma: no cover - optional dependency in test envs
    lxml_html = None

try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:  # pragma: no cover - optional dependency in test envs
        DDGS = None


def _scrape_fallback(query: str, max_results: int = 5) -> list:
    """Fallback scraping using html.duckduckgo.com"""
    print(f"DEBUG: Starting fallback scrape for '{query}'")
    try:
        url = "https://html.duckduckgo.com/html/"
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/115.0",
            "Referer": "https://html.duckduckgo.com/",
        }
        data = {"q": query}

        resp = requests.post(url, data=data, headers=headers, timeout=10)
        resp.raise_for_status()

        if lxml_html is None:
            return []
        tree = lxml_html.fromstring(resp.content)

        results = []
        found_nodes = tree.cssselect(".result")
        print(f"DEBUG: Found {len(found_nodes)} .result nodes")

        for result in found_nodes:
            title_node = result.cssselect(".result__title .result__a")
            snippet_node = result.cssselect(".result__snippet")

            if title_node:
                title = title_node[0].text_content().strip()
                href = title_node[0].get("href")
                snippet = snippet_node[0].text_content().strip() if snippet_node else ""

                results.append({"title": title, "href": href, "body": snippet})

            if len(results) >= max_results:
                break

        return results
    except Exception as e:
        print(f"Fallback scraping error: {e}")
        return []


from core.safety import check_url_safety


def search_web(query: str, max_results: int = 5) -> str:
    """
    Performs a web search using DuckDuckGo.
    Returns a JSON string of results.
    """
    try:
        check_url_safety("https://duckduckgo.com")
    except PermissionError as pe:
        return str(pe)

    results = []

    # Try Library First
    if DDGS is not None:
        try:
            with DDGS() as ddgs:
                # text() returns a generator
                for r in ddgs.text(query, max_results=max_results):
                    results.append(r)
        except Exception as e:
            print(f"DDGS Library Error: {e}")

    # Fallback if empty
    if not results:
        print("DEBUG: DDGS library returned no results or failed, attempting fallback.")
        results = _scrape_fallback(query, max_results)

    if not results:
        return "No results found."

    return json.dumps(results, indent=2)
