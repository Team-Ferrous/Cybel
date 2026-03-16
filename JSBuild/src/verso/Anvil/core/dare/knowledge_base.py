"""Persistent markdown-backed knowledge base for DARE artifacts."""

from __future__ import annotations

import json
import os
import re
import sqlite3
from typing import Any, Dict, Iterable, List, Optional

from core.dare.models import KBEntry, utc_now_iso


CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}


class DareKnowledgeBase:
    """
    Persistent, structured markdown knowledge base.

    DARE artifacts are stored under:
        .anvil/dare/knowledge/<category>/<topic>.md
    """

    def __init__(self, root_dir: str = ".", campaign_id: Optional[str] = None):
        self.root_dir = os.path.abspath(root_dir)
        self.campaign_id = campaign_id
        self.kb_dir = os.path.join(self.root_dir, ".anvil", "dare", "knowledge")
        self.db_path = os.path.join(self.kb_dir, "knowledge.db")
        os.makedirs(self.kb_dir, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS entries (
                path TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                topic TEXT NOT NULL,
                source TEXT NOT NULL,
                confidence TEXT NOT NULL,
                created TEXT NOT NULL,
                updated TEXT NOT NULL,
                tags_json TEXT NOT NULL,
                dependencies_json TEXT NOT NULL,
                campaign_id TEXT,
                metadata_json TEXT NOT NULL,
                content TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts
            USING fts5(path, category, topic, source, content, tags, tokenize='porter')
            """
        )
        self._conn.commit()

    @staticmethod
    def _slugify(value: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
        return cleaned or "entry"

    def _frontmatter(self, entry: KBEntry) -> str:
        frontmatter = {
            "source": entry.source,
            "confidence": entry.confidence,
            "created": entry.created,
            "updated": entry.updated,
            "tags": entry.tags,
            "dependencies": entry.dependencies,
            "campaign_id": entry.campaign_id,
            "metadata": entry.metadata,
        }
        return "---\n" + json.dumps(frontmatter, indent=2, sort_keys=True) + "\n---\n\n"

    def store(
        self,
        category: str,
        topic: str,
        content: str,
        source: str,
        confidence: str,
        tags: List[str],
        dependencies: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        campaign_id: Optional[str] = None,
    ) -> str:
        """Store a knowledge artifact and return the markdown file path."""
        category_slug = self._slugify(category)
        topic_slug = self._slugify(topic)
        category_dir = os.path.join(self.kb_dir, category_slug)
        os.makedirs(category_dir, exist_ok=True)
        path = os.path.join(category_dir, f"{topic_slug}.md")
        now = utc_now_iso()
        existing = self.get_entry(path)
        created = existing.created if existing else now
        entry = KBEntry(
            category=category_slug,
            topic=topic,
            path=path,
            content=content.strip(),
            source=source,
            confidence=(confidence or "low").lower(),
            created=created,
            updated=now,
            tags=sorted(set(tags or [])),
            dependencies=sorted(set(dependencies or [])),
            campaign_id=campaign_id or self.campaign_id,
            metadata=dict(metadata or {}),
        )

        with open(path, "w", encoding="utf-8") as handle:
            handle.write(self._frontmatter(entry))
            handle.write(entry.content.rstrip() + "\n")

        self._conn.execute(
            """
            INSERT INTO entries (
                path, category, topic, source, confidence, created, updated,
                tags_json, dependencies_json, campaign_id, metadata_json, content
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                category=excluded.category,
                topic=excluded.topic,
                source=excluded.source,
                confidence=excluded.confidence,
                updated=excluded.updated,
                tags_json=excluded.tags_json,
                dependencies_json=excluded.dependencies_json,
                campaign_id=excluded.campaign_id,
                metadata_json=excluded.metadata_json,
                content=excluded.content
            """,
            (
                path,
                entry.category,
                entry.topic,
                entry.source,
                entry.confidence,
                entry.created,
                entry.updated,
                json.dumps(entry.tags),
                json.dumps(entry.dependencies),
                entry.campaign_id,
                json.dumps(entry.metadata, sort_keys=True),
                entry.content,
            ),
        )
        self._conn.execute("DELETE FROM entries_fts WHERE path = ?", (path,))
        self._conn.execute(
            """
            INSERT INTO entries_fts (path, category, topic, source, content, tags)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (path, entry.category, entry.topic, entry.source, entry.content, " ".join(entry.tags)),
        )
        self._conn.commit()
        return path

    def get_entry(self, path: str) -> Optional[KBEntry]:
        row = self._conn.execute(
            "SELECT * FROM entries WHERE path = ?",
            (path,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_entry(row)

    def list_entries(
        self,
        category: Optional[str] = None,
        campaign_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[KBEntry]:
        query = "SELECT * FROM entries"
        params: List[Any] = []
        clauses: List[str] = []
        if category:
            clauses.append("category = ?")
            params.append(self._slugify(category))
        active_campaign = campaign_id if campaign_id is not None else self.campaign_id
        if active_campaign:
            clauses.append("(campaign_id = ? OR campaign_id IS NULL)")
            params.append(active_campaign)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY updated DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_entry(row) for row in rows]

    def query(
        self,
        query: str,
        category: Optional[str] = None,
        min_confidence: str = "low",
        limit: int = 10,
        campaign_id: Optional[str] = None,
    ) -> List[KBEntry]:
        """Search across stored knowledge using FTS with a LIKE fallback."""
        active_campaign = campaign_id if campaign_id is not None else self.campaign_id
        minimum = CONFIDENCE_ORDER.get(str(min_confidence).lower(), 0)
        clauses: List[str] = []
        params: List[Any] = []
        if category:
            clauses.append("e.category = ?")
            params.append(self._slugify(category))
        if active_campaign:
            clauses.append("(e.campaign_id = ? OR e.campaign_id IS NULL)")
            params.append(active_campaign)

        matches_all = not query or query.strip() in {"*", "all"}
        if matches_all:
            sql = "SELECT e.* FROM entries e"
            if clauses:
                sql += " WHERE " + " AND ".join(clauses)
            sql += " ORDER BY e.updated DESC LIMIT ?"
            params.append(limit)
        else:
            sql = """
                SELECT e.*
                FROM entries e
                JOIN entries_fts f ON f.path = e.path
                WHERE entries_fts MATCH ?
            """
            fts_query = " ".join(part for part in re.split(r"\s+", query.strip()) if part)
            params = [fts_query] + params
            if clauses:
                sql += " AND " + " AND ".join(clauses)
            sql += " ORDER BY rank LIMIT ?"
            params.append(limit)

        try:
            rows = self._conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            like = f"%{query}%"
            sql = "SELECT * FROM entries WHERE (topic LIKE ? OR content LIKE ?)"
            params = [like, like]
            if category:
                sql += " AND category = ?"
                params.append(self._slugify(category))
            if active_campaign:
                sql += " AND (campaign_id = ? OR campaign_id IS NULL)"
                params.append(active_campaign)
            sql += " ORDER BY updated DESC LIMIT ?"
            params.append(limit)
            rows = self._conn.execute(sql, params).fetchall()

        entries = [self._row_to_entry(row) for row in rows]
        return [
            entry
            for entry in entries
            if CONFIDENCE_ORDER.get(entry.confidence, 0) >= minimum
        ][:limit]

    def get_context_for_task(
        self,
        task_description: str,
        budget_tokens: int = 8000,
        category: Optional[str] = None,
        campaign_id: Optional[str] = None,
    ) -> str:
        """Build a compact context block from the most relevant entries."""
        entries = self.query(
            task_description,
            category=category,
            limit=12,
            campaign_id=campaign_id,
        )
        chunks: List[str] = []
        max_words = max(1, int(budget_tokens * 0.75))
        used_words = 0
        for entry in entries:
            chunk = "\n".join(
                [
                    f"## {entry.topic}",
                    f"- Category: {entry.category}",
                    f"- Source: {entry.source}",
                    f"- Confidence: {entry.confidence}",
                    f"- Tags: {', '.join(entry.tags) if entry.tags else 'none'}",
                    "",
                    entry.content.strip(),
                ]
            ).strip()
            words = chunk.split()
            if used_words + len(words) > max_words and chunks:
                break
            if used_words + len(words) > max_words:
                words = words[: max_words - used_words]
                chunk = " ".join(words)
            chunks.append(chunk)
            used_words += len(words)
        return "\n\n".join(chunks)

    def get_full_report(
        self,
        category: Optional[str] = None,
        campaign_id: Optional[str] = None,
    ) -> str:
        """Compile knowledge into a single markdown report."""
        entries = self.list_entries(category=category, campaign_id=campaign_id)
        lines = ["# DARE Knowledge Report", ""]
        if category:
            lines.extend([f"Category: `{self._slugify(category)}`", ""])
        for entry in entries:
            lines.extend(
                [
                    f"## {entry.topic}",
                    f"- Path: `{os.path.relpath(entry.path, self.root_dir)}`",
                    f"- Source: {entry.source}",
                    f"- Confidence: {entry.confidence}",
                    f"- Updated: {entry.updated}",
                    "",
                    entry.content.strip(),
                    "",
                ]
            )
        return "\n".join(lines).rstrip() + "\n"

    def merge_findings(
        self,
        sources: List[str],
        output_topic: str,
        category: str = "synthesis",
        source_label: str = "dare-merge",
        confidence: str = "medium",
        tags: Optional[List[str]] = None,
    ) -> str:
        """Cross-reference multiple KB entries into a synthesis document."""
        entries: List[KBEntry] = []
        for source in sources:
            candidate = self.get_entry(source)
            if candidate is not None:
                entries.append(candidate)
                continue
            entries.extend(self.query(source, limit=1))
        lines = [f"# {output_topic}", ""]
        for entry in entries:
            lines.extend(
                [
                    f"## {entry.topic}",
                    f"Source: {entry.source}",
                    entry.content.strip(),
                    "",
                ]
            )
        merged_content = "\n".join(lines).strip() + "\n"
        self.store(
            category=category,
            topic=output_topic,
            content=merged_content,
            source=source_label,
            confidence=confidence,
            tags=tags or ["synthesis"],
            dependencies=[entry.path for entry in entries],
        )
        return merged_content

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> KBEntry:
        return KBEntry(
            category=row["category"],
            topic=row["topic"],
            path=row["path"],
            content=row["content"],
            source=row["source"],
            confidence=row["confidence"],
            created=row["created"],
            updated=row["updated"],
            tags=json.loads(row["tags_json"]),
            dependencies=json.loads(row["dependencies_json"]),
            campaign_id=row["campaign_id"],
            metadata=json.loads(row["metadata_json"]),
        )

    def close(self) -> None:
        self._conn.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
