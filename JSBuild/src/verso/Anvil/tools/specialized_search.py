import json
try:
    import arxiv
except ImportError:  # pragma: no cover - optional dependency in test envs
    arxiv = None

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - optional dependency in test envs
    yf = None

try:
    from scholarly import scholarly
except ImportError:
    scholarly = None

try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None


def search_arxiv(query: str, max_results: int = 5) -> str:
    """
    Search for scientific papers on arXiv.
    """
    if arxiv is None:
        return "arXiv library not installed."
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance
        )

        results = []
        for result in client.results(search):
            results.append(
                {
                    "title": result.title,
                    "url": result.entry_id,
                    "pdf_url": result.pdf_url,
                    "summary": result.summary.replace("\n", " "),
                    "authors": [a.name for a in result.authors],
                    "published": result.published.strftime("%Y-%m-%d"),
                    "categories": result.categories,
                }
            )

        if not results:
            return "No arXiv results found."

        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Error searching arXiv: {str(e)}"


def search_finance(symbol: str) -> str:
    """
    Get financial information for a given ticker symbol (e.g., AAPL, BTC-USD).
    """
    data = {"symbol": symbol}
    ticker_success = False

    if yf is not None:
        try:
        # Some yfinance symbols benefit from .T or .L etc, but we'll take what user gives
            ticker = yf.Ticker(symbol)
            info = ticker.info

            if info and "longName" in info:
                data.update(
                    {
                        "name": info.get("longName"),
                        "currentPrice": info.get("currentPrice")
                        or info.get("regularMarketPrice"),
                        "currency": info.get("currency"),
                        "marketCap": info.get("marketCap"),
                        "dayHigh": info.get("dayHigh"),
                        "dayLow": info.get("dayLow"),
                        "volume": info.get("volume"),
                        "summary": (
                            info.get("longBusinessSummary", "")[:300] + "..."
                            if info.get("longBusinessSummary")
                            else ""
                        ),
                    }
                )
                ticker_success = True
        except Exception as e:
            data["ticker_error"] = str(e)

    # Get recent news via DuckDuckGo
    news_results = []
    if DDGS:
        try:
            with DDGS() as ddgs:
                # Use symbol + " stock news" for better targeting
                for r in ddgs.news(f"{symbol} stock news", max_results=5):
                    news_results.append(
                        {
                            "title": r.get("title"),
                            "url": r.get("url"),
                            "source": r.get("source"),
                            "date": r.get("date"),
                            "body": r.get("body"),
                        }
                    )
        except Exception:
            pass

    data["recent_news"] = news_results

    if not ticker_success and not news_results:
        return (
            f"Error fetching finance data for {symbol}: No information or news found."
        )

    return json.dumps(data, indent=2)


def search_scholar(query: str, max_results: int = 5) -> str:
    """
    Search for academic papers on Google Scholar.
    """
    if not scholarly:
        # Fallback to DuckDuckGo site search
        if DDGS:
            try:
                with DDGS() as ddgs:
                    results = []
                    for r in ddgs.text(
                        f"site:scholar.google.com {query}", max_results=max_results
                    ):
                        results.append(r)
                    return json.dumps(results, indent=2)
            except Exception as e:
                return f"Scholar search failed: {str(e)}"
        return "Scholarly library not installed and no fallback available."

    try:
        search_query = scholarly.search_pubs(query)
        results = []
        for _ in range(max_results):
            try:
                pub = next(search_query)
                results.append(
                    {
                        "title": pub["bib"].get("title"),
                        "author": pub["bib"].get("author"),
                        "pub_year": pub["bib"].get("pub_year"),
                        "venue": pub["bib"].get("venue"),
                        "abstract": pub["bib"].get("abstract"),
                        "url": pub.get("pub_url"),
                    }
                )
            except StopIteration:
                break

        if not results:
            return "No Google Scholar results found."

        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Error searching Google Scholar: {str(e)}"


def search_books(query: str, max_results: int = 5) -> str:
    """
    Search for books and papers on Anna's Archive.
    """
    if not DDGS:
        return "Search engine not available."

    try:
        # Check if the DDGS library we have supports annasarchive
        # In this specific env, it seems to be 'Dux Distributed Global Search'
        with DDGS() as ddgs:
            # We need to call it specifically if it's the custom one
            if hasattr(ddgs, "books"):
                results = ddgs.books(
                    query, max_results=max_results, backend="annasarchive"
                )
            else:
                # Standard duckduckgo_search doesn't have books() but we might be able to use text with a filter
                results = list(
                    ddgs.text(
                        f"site:annas-archive.org {query}", max_results=max_results
                    )
                )

            return json.dumps(results, indent=2)
    except Exception as e:
        return f"Error searching books: {str(e)}"


def search_news(query: str, max_results: int = 5) -> str:
    """
    Search for recent news articles.
    """
    if not DDGS:
        return "Search engine not available."
    try:
        with DDGS() as ddgs:
            results = []
            for r in ddgs.news(query, max_results=max_results):
                results.append(r)
            return json.dumps(results, indent=2)
    except Exception as e:
        return f"News search failed: {str(e)}"


def search_patents(query: str, max_results: int = 5) -> str:
    """
    Search for patents on Google Patents.
    """
    if not DDGS:
        return "Search engine not available."
    try:
        with DDGS() as ddgs:
            results = []
            # Use site:patents.google.com
            for r in ddgs.text(
                f"site:patents.google.com {query}", max_results=max_results
            ):
                results.append(r)
            return json.dumps(results, indent=2)
    except Exception as e:
        return f"Patent search failed: {str(e)}"
