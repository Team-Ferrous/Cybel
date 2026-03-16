"""Forum and community search helpers for DARE."""

from __future__ import annotations

import json

import requests

USER_AGENT = "AnvilDARE/1.0"


def _get_json(url: str, params: dict[str, object] | None = None) -> dict[str, object]:
    response = requests.get(
        url,
        params=params or {},
        headers={"User-Agent": USER_AGENT},
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


def search_reddit(
    query: str,
    subreddits: list[str] | None = None,
    sort: str = "relevance",
    time_filter: str = "year",
) -> str:
    """Search Reddit via the public JSON endpoint."""
    subreddit_clause = ""
    if subreddits:
        subreddit_clause = (
            " (" + " OR ".join(f"subreddit:{item}" for item in subreddits) + ")"
        )
    payload = _get_json(
        "https://www.reddit.com/search.json",
        params={
            "q": f"{query}{subreddit_clause}",
            "sort": sort,
            "t": time_filter,
            "limit": 10,
            "restrict_sr": False,
            "raw_json": 1,
        },
    )
    items = []
    for child in payload.get("data", {}).get("children", []):
        data = child.get("data", {})
        items.append(
            {
                "title": data.get("title"),
                "subreddit": data.get("subreddit"),
                "url": f"https://www.reddit.com{data.get('permalink', '')}",
                "score": data.get("score"),
                "created_utc": data.get("created_utc"),
                "selftext": data.get("selftext", "")[:1000],
            }
        )
    return json.dumps(items, indent=2)


def search_hackernews(query: str, max_results: int = 10) -> str:
    """Search Hacker News via Algolia."""
    payload = _get_json(
        "https://hn.algolia.com/api/v1/search",
        params={"query": query, "hitsPerPage": max_results},
    )
    items = []
    for hit in payload.get("hits", []):
        items.append(
            {
                "title": hit.get("title") or hit.get("story_title"),
                "url": hit.get("url") or hit.get("story_url"),
                "author": hit.get("author"),
                "points": hit.get("points"),
                "created_at": hit.get("created_at"),
                "comment_text": (hit.get("comment_text") or "")[:1000],
            }
        )
    return json.dumps(items, indent=2)


def search_stackoverflow(
    query: str,
    tags: list[str] | None = None,
    max_results: int = 10,
) -> str:
    """Search Stack Overflow via Stack Exchange API."""
    params: dict[str, object] = {
        "order": "desc",
        "sort": "relevance",
        "intitle": query,
        "site": "stackoverflow",
        "pagesize": max_results,
        "filter": "!nNPvSNVqHq",
    }
    if tags:
        params["tagged"] = ";".join(tags)
    payload = _get_json("https://api.stackexchange.com/2.3/search", params=params)
    items = []
    for item in payload.get("items", []):
        items.append(
            {
                "title": item.get("title"),
                "url": item.get("link"),
                "score": item.get("score"),
                "is_answered": item.get("is_answered"),
                "tags": item.get("tags", []),
                "creation_date": item.get("creation_date"),
            }
        )
    return json.dumps(items, indent=2)
