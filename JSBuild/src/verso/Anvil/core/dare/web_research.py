"""External research and competitor intelligence for DARE."""

from __future__ import annotations

import json
import os
import subprocess
from typing import Dict, List, Optional

import requests

from core.dare.deep_analyzer import DeepAnalyzer
from core.dare.knowledge_base import DareKnowledgeBase
from core.dare.models import (
    CompetitiveReport,
    CompetitorProfile,
    ResearchReport,
    ResearchSource,
    RepoProfile,
)
from core.dare.repo_ingestion import RepoIngestionEngine
from tools.arxiv_search import fetch_arxiv_paper, search_arxiv
from tools.forum_search import search_hackernews, search_reddit, search_stackoverflow
from tools.specialized_search import search_scholar
from tools.web_search import search_web


def _parse_results(payload: str) -> List[dict]:
    try:
        parsed = json.loads(payload)
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else [parsed]


class ArxivSource:
    def search(self, topic: str, max_results: int = 5) -> List[ResearchSource]:
        results = _parse_results(search_arxiv(topic, max_results=max_results))
        return [
            ResearchSource(
                source_type="arxiv",
                title=item.get("title", "Untitled"),
                url=item.get("entry_id") or item.get("pdf_url") or "",
                summary=item.get("summary", ""),
                confidence="high",
                published=item.get("published"),
                metadata=item,
            )
            for item in results
        ]


class GitHubSearchSource:
    def search(self, topic: str, max_results: int = 5) -> List[ResearchSource]:
        try:
            response = requests.get(
                "https://api.github.com/search/repositories",
                params={
                    "q": topic,
                    "sort": "stars",
                    "order": "desc",
                    "per_page": max_results,
                },
                timeout=20,
                headers={"Accept": "application/vnd.github+json"},
            )
            response.raise_for_status()
            payload = response.json()
        except Exception:
            return []
        items = payload.get("items", [])
        return [
            ResearchSource(
                source_type="github",
                title=item.get("full_name", "unknown"),
                url=item.get("html_url", ""),
                summary=item.get("description") or "",
                confidence="medium",
                published=item.get("updated_at"),
                metadata={
                    "stargazers_count": item.get("stargazers_count"),
                    "language": item.get("language"),
                    "default_branch": item.get("default_branch"),
                },
            )
            for item in items
        ]


class ForumAggregatorSource:
    def search(self, topic: str, max_results: int = 5) -> List[ResearchSource]:
        combined: List[ResearchSource] = []
        for source_type, payload in [
            ("reddit", _parse_results(search_reddit(topic, time_filter="year"))),
            (
                "hackernews",
                _parse_results(search_hackernews(topic, max_results=max_results)),
            ),
            (
                "stackoverflow",
                _parse_results(search_stackoverflow(topic, max_results=max_results)),
            ),
        ]:
            for item in payload[:max_results]:
                combined.append(
                    ResearchSource(
                        source_type=source_type,
                        title=item.get("title", "Untitled"),
                        url=item.get("url") or item.get("link") or "",
                        summary=item.get("selftext")
                        or item.get("comment_text")
                        or json.dumps(item)[:600],
                        confidence=(
                            "medium" if source_type != "stackoverflow" else "high"
                        ),
                        published=item.get("created_at") or item.get("creation_date"),
                        metadata=item,
                    )
                )
        return combined


class DocumentationSource:
    def search(self, topic: str, max_results: int = 5) -> List[ResearchSource]:
        results = _parse_results(
            search_web(f"{topic} official documentation", max_results=max_results)
        )
        return [
            ResearchSource(
                source_type="docs",
                title=item.get("title", "Untitled"),
                url=item.get("href") or item.get("url") or "",
                summary=item.get("body") or "",
                confidence="medium",
                metadata=item,
            )
            for item in results
        ]


class ScholarSource:
    def search(self, topic: str, max_results: int = 5) -> List[ResearchSource]:
        results = _parse_results(search_scholar(topic, max_results=max_results))
        return [
            ResearchSource(
                source_type="scholar",
                title=item.get("title", "Untitled"),
                url=item.get("url") or "",
                summary=item.get("abstract") or "",
                confidence="medium",
                published=item.get("pub_year"),
                metadata=item,
            )
            for item in results
        ]


class WebResearchEngine:
    """Autonomous research with multiple public data sources."""

    def __init__(
        self,
        knowledge_base: DareKnowledgeBase,
        workspace_root: str = ".",
    ):
        self.kb = knowledge_base
        self.workspace_root = os.path.abspath(workspace_root)
        self.deep_analyzer = DeepAnalyzer(self.kb)
        self.sources = {
            "arxiv": ArxivSource(),
            "github": GitHubSearchSource(),
            "forums": ForumAggregatorSource(),
            "docs": DocumentationSource(),
            "scholar": ScholarSource(),
        }

    def research_topic(self, topic: str, depth: str = "deep") -> ResearchReport:
        rounds = {"broad": 1, "deep": 2, "exhaustive": 3}.get(depth, 2)
        max_results = {"broad": 3, "deep": 5, "exhaustive": 8}.get(depth, 5)
        seen_urls: set[str] = set()
        sources: List[ResearchSource] = []
        for _ in range(rounds):
            for source_name in self._source_order_for_depth(depth):
                for item in self.sources[source_name].search(
                    topic, max_results=max_results
                ):
                    if item.url in seen_urls:
                        continue
                    seen_urls.add(item.url)
                    sources.append(item)

        findings = [f"{item.source_type}: {item.title}" for item in sources[:8]]
        coverage = self._coverage(sources)
        gaps = self._gaps(coverage)
        report = ResearchReport(
            topic=topic,
            summary=self._summarize(topic, sources, gaps),
            sources=sources,
            findings=findings,
            gaps=gaps,
            coverage=coverage,
            novelty_score=self._novelty_score(coverage, gaps),
            exhausted=self._is_exhausted(coverage, gaps),
            metadata={
                "depth": depth,
                "source_order": self._source_order_for_depth(depth),
            },
        )
        self.kb.store(
            category="research",
            topic=topic,
            content=self._report_to_markdown(report),
            source="web-research",
            confidence="high" if len(sources) >= 5 else "medium",
            tags=["research", depth],
            metadata=report.to_dict(),
        )
        return report

    def competitive_analysis(
        self,
        domain: str,
        our_repo: RepoProfile | None = None,
    ) -> CompetitiveReport:
        competitor_sources = self.sources["github"].search(domain, max_results=5)
        competitors: List[CompetitorProfile] = []
        acquired_repos = self._ingest_github_repos(
            domain,
            competitor_sources,
            max_repos=2,
        )
        acquired_by_url = {item["url"]: item for item in acquired_repos}
        for item in competitor_sources:
            acquired = acquired_by_url.get(item.url, {})
            competitors.append(
                CompetitorProfile(
                    name=item.title,
                    url=item.url,
                    summary=item.summary,
                    stars=item.metadata.get("stargazers_count"),
                    local_path=acquired.get("local_path"),
                    metadata=acquired,
                ),
            )
        forum_items = self.sources["forums"].search(domain, max_results=5)
        gaps = self._extract_competitive_gaps(competitors, forum_items, our_repo)
        recommendations = [
            "Register downloaded competitor repos as read-only campaign resources.",
            "Convert repeated community pain points into hypothesis tests.",
        ]
        if acquired_repos:
            recommendations.append(
                f"Deep-ingested {len(acquired_repos)} GitHub repos for file-by-file comparison."
            )
        report = CompetitiveReport(
            domain=domain,
            summary=f"Competitive scan covered {len(competitors)} repos and {len(forum_items)} forum threads.",
            competitors=competitors,
            gaps=gaps,
            recommendations=recommendations,
            metadata={
                "our_repo": our_repo.repo_name if our_repo else None,
                "acquired_repos": acquired_repos,
            },
        )
        self.kb.store(
            category="competitors",
            topic=domain,
            content=self._competitive_to_markdown(report),
            source="competitive-analysis",
            confidence="medium",
            tags=["competitive", "github", "forums"],
            metadata=report.to_dict(),
        )
        return report

    def research_until_satisfied(
        self,
        objective: str,
        min_sources: int = 5,
        min_confidence: str = "medium",
    ) -> ResearchReport:
        minimum = {"low": 0, "medium": 1, "high": 2}.get(min_confidence.lower(), 1)
        frontier = self._initial_frontier(objective)
        seen_urls: set[str] = set()
        sources: List[ResearchSource] = []
        frontier_history: List[Dict[str, object]] = []
        acquired_repos: List[Dict[str, object]] = []
        low_yield_rounds = 0
        iteration = 0

        while frontier and iteration < 8:
            task = frontier.pop(0)
            source_name = str(task["source"])
            query = str(task["query"])
            max_results = int(task["max_results"])
            results = self.sources[source_name].search(query, max_results=max_results)
            new_sources = 0
            for item in results:
                if item.url in seen_urls:
                    continue
                seen_urls.add(item.url)
                sources.append(item)
                new_sources += 1

            if source_name == "github":
                acquired = self._ingest_github_repos(
                    objective,
                    results,
                    max_repos=int(task.get("repo_limit", 2)),
                )
                existing_urls = {item["url"] for item in acquired_repos}
                for record in acquired:
                    if record["url"] not in existing_urls:
                        acquired_repos.append(record)

            coverage = self._coverage(sources)
            gaps = self._gaps(coverage)
            exhausted = self._is_exhausted(coverage, gaps)
            high_confidence_count = sum(
                1
                for item in sources
                if self._confidence_rank(item.confidence) >= minimum
            )
            frontier_history.append(
                {
                    "iteration": iteration + 1,
                    "source": source_name,
                    "query": query,
                    "new_sources": new_sources,
                    "coverage": dict(coverage),
                    "gaps": list(gaps),
                    "acquired_repos": len(acquired_repos),
                }
            )
            if new_sources == 0:
                low_yield_rounds += 1
            else:
                low_yield_rounds = 0

            if (
                len(sources) >= min_sources
                and high_confidence_count >= min_sources
                and exhausted
            ) or low_yield_rounds >= 2:
                break

            frontier.extend(
                self._expand_frontier(
                    objective,
                    frontier,
                    coverage,
                    gaps,
                    iteration + 1,
                )
            )
            iteration += 1

        coverage = self._coverage(sources)
        gaps = self._gaps(coverage)
        findings = [f"{item.source_type}: {item.title}" for item in sources[:8]]
        report = ResearchReport(
            topic=objective,
            summary=self._summarize(objective, sources, gaps),
            sources=sources,
            findings=findings,
            gaps=gaps,
            coverage=coverage,
            novelty_score=self._novelty_score(coverage, gaps),
            exhausted=self._is_exhausted(coverage, gaps) or low_yield_rounds >= 2,
            metadata={
                "mode": "frontier",
                "frontier_history": frontier_history,
                "acquired_repos": acquired_repos,
                "min_confidence": min_confidence,
            },
        )
        self.kb.store(
            category="research",
            topic=f"{objective}_frontier",
            content=self._report_to_markdown(report),
            source="web-research-frontier",
            confidence="high" if len(sources) >= min_sources else "medium",
            tags=["research", "frontier", "autonomous"],
            metadata=report.to_dict(),
        )
        return report

    def download_competitor_repo(
        self,
        repo_url: str,
        dest_root: Optional[str] = None,
    ) -> Optional[str]:
        dest_root = dest_root or os.path.join(
            self.workspace_root, ".anvil", "dare", "competitors"
        )
        os.makedirs(dest_root, exist_ok=True)
        slug = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
        destination = os.path.join(dest_root, slug)
        if os.path.isdir(destination):
            return destination
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, destination],
                check=True,
                capture_output=True,
                text=True,
            )
            return destination
        except Exception:
            return None

    def fetch_paper_detail(self, arxiv_id: str) -> dict:
        try:
            return json.loads(fetch_arxiv_paper(arxiv_id))
        except Exception:
            return {"error": f"Unable to fetch paper detail for {arxiv_id}"}

    @staticmethod
    def _source_order_for_depth(depth: str) -> List[str]:
        if depth == "broad":
            return ["docs", "github", "forums", "arxiv"]
        if depth == "exhaustive":
            return ["docs", "github", "forums", "arxiv", "scholar"]
        return ["docs", "github", "forums", "scholar", "arxiv"]

    def _initial_frontier(self, objective: str) -> List[Dict[str, object]]:
        return [
            {"source": "docs", "query": objective, "max_results": 4, "repo_limit": 0},
            {"source": "github", "query": objective, "max_results": 5, "repo_limit": 2},
            {"source": "forums", "query": objective, "max_results": 4, "repo_limit": 0},
            {"source": "arxiv", "query": objective, "max_results": 3, "repo_limit": 0},
        ]

    def _expand_frontier(
        self,
        objective: str,
        frontier: List[Dict[str, object]],
        coverage: Dict[str, int],
        gaps: List[str],
        iteration: int,
    ) -> List[Dict[str, object]]:
        pending = {(str(item["source"]), str(item["query"])) for item in frontier}
        additions: List[Dict[str, object]] = []

        def add(source: str, query: str, max_results: int, repo_limit: int = 0) -> None:
            key = (source, query)
            if key in pending:
                return
            pending.add(key)
            additions.append(
                {
                    "source": source,
                    "query": query,
                    "max_results": max_results,
                    "repo_limit": repo_limit,
                }
            )

        if coverage.get("docs", 0) == 0:
            add("docs", f"{objective} official documentation", 5)
        if coverage.get("github", 0) == 0:
            add("github", f"{objective} implementation", 6, repo_limit=2)
        if coverage.get("forums", 0) == 0:
            add("forums", f"{objective} benchmark OR performance OR pain points", 5)
        if coverage.get("arxiv", 0) == 0:
            add("arxiv", f"{objective} algorithm", 4)
        if coverage.get("scholar", 0) == 0 and iteration >= 1:
            add("scholar", f"{objective} research paper", 4)
        if any("Fewer than five distinct sources" in gap for gap in gaps):
            add("github", f"{objective} framework", 5, repo_limit=1)
            add("docs", f"{objective} architecture guide", 4)
        if iteration >= 2:
            add("forums", f"{objective} alternatives", 4)
            add("github", f"{objective} optimized implementation", 4, repo_limit=1)
        return additions

    @staticmethod
    def _confidence_rank(confidence: str) -> int:
        return {"low": 0, "medium": 1, "high": 2}.get(str(confidence).lower(), 0)

    def _ingest_github_repos(
        self,
        objective: str,
        sources: List[ResearchSource],
        *,
        max_repos: int,
    ) -> List[Dict[str, object]]:
        ranked = sorted(
            (
                (self._github_relevance(objective, item), item)
                for item in sources
                if item.source_type == "github" and item.url
            ),
            key=lambda pair: pair[0],
            reverse=True,
        )
        acquired: List[Dict[str, object]] = []
        for score, source in ranked:
            if len(acquired) >= max_repos or score < 0.35:
                continue
            local_path = self.download_competitor_repo(source.url)
            if not local_path:
                continue
            ingestion = RepoIngestionEngine(
                [local_path],
                self.kb,
                repo_roles={local_path: "analysis_external"},
            )
            profiles = ingestion.ingest_all()
            profile = profiles.get(local_path)
            if profile is None:
                continue
            deep_report = self.deep_analyzer.analyze(profile)
            source.metadata["local_path"] = local_path
            source.metadata["relevance_score"] = score
            acquired.append(
                {
                    "title": source.title,
                    "url": source.url,
                    "local_path": local_path,
                    "relevance_score": score,
                    "file_count": profile.file_count,
                    "loc": profile.loc,
                    "analysis_path": deep_report.repo_path,
                }
            )
        return acquired

    @staticmethod
    def _github_relevance(objective: str, source: ResearchSource) -> float:
        objective_terms = {
            token.lower()
            for token in objective.replace("/", " ").replace("-", " ").split()
            if len(token) > 2
        }
        haystack = f"{source.title} {source.summary}".lower()
        lexical = len([term for term in objective_terms if term in haystack]) / max(
            len(objective_terms), 1
        )
        stars = float(source.metadata.get("stargazers_count") or 0)
        star_score = min(stars / 5000.0, 1.0)
        description_bonus = 0.15 if source.summary else 0.0
        return round((lexical * 0.6) + (star_score * 0.25) + description_bonus, 2)

    @staticmethod
    def _coverage(sources: List[ResearchSource]) -> Dict[str, int]:
        coverage: Dict[str, int] = {}
        for item in sources:
            coverage[item.source_type] = coverage.get(item.source_type, 0) + 1
        return coverage

    @staticmethod
    def _gaps(coverage: Dict[str, int]) -> List[str]:
        desired = {"docs", "github", "forums", "arxiv"}
        missing = sorted(item for item in desired if coverage.get(item, 0) == 0)
        gaps = [f"No coverage from {item}" for item in missing]
        if sum(coverage.values()) < 5:
            gaps.append("Fewer than five distinct sources were collected.")
        return gaps

    @staticmethod
    def _novelty_score(coverage: Dict[str, int], gaps: List[str]) -> float:
        spread = min(len(coverage), 5) / 5.0
        penalty = min(len(gaps), 4) * 0.1
        return max(0.0, round(spread - penalty, 2))

    @staticmethod
    def _is_exhausted(coverage: Dict[str, int], gaps: List[str]) -> bool:
        return len(coverage) >= 3 and len(gaps) <= 1

    @staticmethod
    def _summarize(topic: str, sources: List[ResearchSource], gaps: List[str]) -> str:
        return (
            f"DARE gathered {len(sources)} sources for '{topic}' across "
            f"{len({item.source_type for item in sources})} source classes. "
            f"Open gaps: {len(gaps)}."
        )

    @staticmethod
    def _report_to_markdown(report: ResearchReport) -> str:
        lines = ["# Research Report", "", report.summary, "", "## Sources"]
        for source in report.sources:
            lines.append(f"- **{source.source_type}** [{source.title}]({source.url})")
        lines.extend(["", "## Findings"])
        lines.extend([f"- {item}" for item in report.findings] or ["- none"])
        lines.extend(["", "## Gaps"])
        lines.extend([f"- {item}" for item in report.gaps] or ["- none"])
        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def _extract_competitive_gaps(
        competitors: List[CompetitorProfile],
        forum_items: List[ResearchSource],
        our_repo: RepoProfile | None,
    ) -> List[str]:
        gaps: List[str] = []
        if our_repo and "simd" not in our_repo.tech_stack:
            gaps.append(
                "Primary repo is not explicitly advertising SIMD-oriented execution paths."
            )
        if not competitors:
            gaps.append(
                "No obvious GitHub competitors were found through unauthenticated search."
            )
        for item in forum_items[:3]:
            text = f"{item.title} {item.summary}".lower()
            if "documentation" in text:
                gaps.append(
                    "Community discussion indicates documentation remains a differentiator."
                )
            if "benchmark" in text or "performance" in text:
                gaps.append("Benchmark transparency is a recurring community concern.")
        return sorted(set(gaps))

    @staticmethod
    def _competitive_to_markdown(report: CompetitiveReport) -> str:
        lines = ["# Competitive Analysis", "", report.summary, "", "## Competitors"]
        for competitor in report.competitors:
            lines.append(f"- [{competitor.name}]({competitor.url})")
        lines.extend(["", "## Gaps"])
        lines.extend([f"- {item}" for item in report.gaps] or ["- none"])
        lines.extend(["", "## Recommendations"])
        lines.extend([f"- {item}" for item in report.recommendations] or ["- none"])
        return "\n".join(lines).rstrip() + "\n"
