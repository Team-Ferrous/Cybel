"""Dedicated arXiv search helpers for DARE."""

from __future__ import annotations

import io
import json

try:
    import arxiv
except ImportError:  # pragma: no cover - optional dependency in test envs
    arxiv = None
import requests

try:
    from pdfminer.high_level import extract_text
except ImportError:  # pragma: no cover - optional dependency
    extract_text = None


def search_arxiv(
    query: str,
    max_results: int = 10,
    sort_by: str = "relevance",
    categories: list[str] | None = None,
) -> str:
    """Search arXiv for papers and return structured JSON."""
    if arxiv is None:
        return json.dumps({"error": "arXiv library not installed."}, indent=2)
    criterion = {
        "relevance": arxiv.SortCriterion.Relevance,
        "submitted": arxiv.SortCriterion.SubmittedDate,
        "updated": arxiv.SortCriterion.LastUpdatedDate,
    }.get(sort_by, arxiv.SortCriterion.Relevance)
    category_filter = ""
    if categories:
        category_filter = (
            " AND (" + " OR ".join(f"cat:{item}" for item in categories) + ")"
        )
    search = arxiv.Search(
        query=f"{query}{category_filter}",
        max_results=max_results,
        sort_by=criterion,
    )
    client = arxiv.Client()
    results = []
    for paper in client.results(search):
        results.append(
            {
                "id": paper.get_short_id(),
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "summary": paper.summary.replace("\n", " "),
                "published": paper.published.strftime("%Y-%m-%d"),
                "updated": paper.updated.strftime("%Y-%m-%d"),
                "pdf_url": paper.pdf_url,
                "entry_id": paper.entry_id,
                "categories": list(paper.categories),
            }
        )
    return json.dumps(results, indent=2)


def fetch_arxiv_paper(arxiv_id: str) -> str:
    """Fetch arXiv metadata and best-effort paper text summary."""
    if arxiv is None:
        return json.dumps({"error": "arXiv library not installed."}, indent=2)
    search = arxiv.Search(id_list=[arxiv_id], max_results=1)
    client = arxiv.Client()
    paper = next(client.results(search), None)
    if paper is None:
        return json.dumps({"error": f"Paper not found: {arxiv_id}"}, indent=2)

    text_excerpt = ""
    if paper.pdf_url and extract_text is not None:
        try:
            response = requests.get(paper.pdf_url, timeout=30)
            response.raise_for_status()
            text = extract_text(io.BytesIO(response.content))
            text_excerpt = " ".join(text.split())[:4000]
        except requests.RequestException:
            text_excerpt = ""
        except OSError:
            text_excerpt = ""

    payload = {
        "id": paper.get_short_id(),
        "title": paper.title,
        "authors": [author.name for author in paper.authors],
        "summary": paper.summary.replace("\n", " "),
        "published": paper.published.strftime("%Y-%m-%d"),
        "updated": paper.updated.strftime("%Y-%m-%d"),
        "pdf_url": paper.pdf_url,
        "entry_id": paper.entry_id,
        "categories": list(paper.categories),
        "text_excerpt": text_excerpt,
    }
    return json.dumps(payload, indent=2)
