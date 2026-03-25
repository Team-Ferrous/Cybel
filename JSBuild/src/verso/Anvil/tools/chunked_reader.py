"""
Chunked File Reader

Progressive file reading with semantic-guided chunk selection.
Enables analyzing large files without overwhelming context windows.
"""

from typing import List, Dict, Any, Optional, Tuple
import os


class ChunkedFileReader:
    """
    Reads large files in configurable chunks with semantic guidance.

    Features:
    - Read files progressively in chunks (default 1000 lines)
    - Semantic-guided chunk selection (find relevant sections)
    - Token-aware loading (respect budgets)
    - Integration with grep results (load only matched sections)
    """

    def __init__(self, chunk_size: int = 1000, overlap: int = 50):
        """
        Initialize chunked reader.

        Args:
            chunk_size: Lines per chunk
            overlap: Lines to overlap between chunks for context continuity
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def read_chunks(
        self, file_path: str, max_chunks: Optional[int] = None, start_line: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Read file in chunks.

        Args:
            file_path: Path to file
            max_chunks: Maximum number of chunks to read (None = all)
            start_line: Line to start from (0-indexed)

        Returns:
            List of chunk dicts with:
                - content: Chunk text
                - start_line: Starting line number
                - end_line: Ending line number
                - file_path: Source file
        """
        if not os.path.exists(file_path):
            return []

        chunks = []

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            total_lines = len(lines)
            current_line = start_line

            while current_line < total_lines:
                if max_chunks and len(chunks) >= max_chunks:
                    break

                # Calculate chunk boundaries
                end_line = min(current_line + self.chunk_size, total_lines)

                # Extract chunk
                chunk_lines = lines[current_line:end_line]
                chunk_content = "".join(chunk_lines)

                chunks.append(
                    {
                        "content": chunk_content,
                        "start_line": current_line + 1,  # 1-indexed for display
                        "end_line": end_line,
                        "file_path": file_path,
                        "lines_count": len(chunk_lines),
                    }
                )

                # Move to next chunk with overlap
                current_line += self.chunk_size - self.overlap

        except Exception as e:
            print(f"Error reading chunks from {file_path}: {e}")

        return chunks

    def read_section(
        self, file_path: str, start_line: int, end_line: int, context_lines: int = 10
    ) -> Optional[str]:
        """
        Read a specific section of a file with context.

        Useful for loading sections around grep matches.

        Args:
            file_path: Path to file
            start_line: Starting line (1-indexed)
            end_line: Ending line (1-indexed)
            context_lines: Additional context lines before/after

        Returns:
            Section content with context, or None if error
        """
        if not os.path.exists(file_path):
            return None

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            # Convert to 0-indexed
            start_idx = max(0, start_line - 1 - context_lines)
            end_idx = min(len(lines), end_line + context_lines)

            section_lines = lines[start_idx:end_idx]
            return "".join(section_lines)

        except Exception as e:
            print(f"Error reading section from {file_path}: {e}")
            return None

    def read_around_matches(
        self, file_path: str, match_lines: List[int], context_lines: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Read sections around grep match lines.

        Merges overlapping sections to avoid duplication.

        Args:
            file_path: Path to file
            match_lines: List of line numbers with matches (1-indexed)
            context_lines: Context lines around each match

        Returns:
            List of section dicts with content and line ranges
        """
        if not match_lines:
            return []

        # Sort and deduplicate matches
        match_lines = sorted(set(match_lines))

        # Merge overlapping regions
        regions = []
        current_start = match_lines[0] - context_lines
        current_end = match_lines[0] + context_lines

        for line in match_lines[1:]:
            region_start = line - context_lines
            region_end = line + context_lines

            if region_start <= current_end:
                # Overlapping - extend current region
                current_end = max(current_end, region_end)
            else:
                # Non-overlapping - save current and start new
                regions.append((current_start, current_end))
                current_start = region_start
                current_end = region_end

        # Add final region
        regions.append((current_start, current_end))

        # Read each region
        sections = []
        for start, end in regions:
            content = self.read_section(file_path, max(1, start), end, context_lines=0)
            if content:
                sections.append(
                    {
                        "content": content,
                        "start_line": max(1, start),
                        "end_line": end,
                        "file_path": file_path,
                        "match_lines": [m for m in match_lines if start <= m <= end],
                    }
                )

        return sections

    def estimate_relevance(
        self, chunk_content: str, query: str, keywords: Optional[List[str]] = None
    ) -> float:
        """
        Estimate relevance of a chunk to the query.

        Simple keyword-based scoring for chunk prioritization.

        Args:
            chunk_content: Content of the chunk
            query: The search query
            keywords: Optional additional keywords to match

        Returns:
            Relevance score (0.0 to 1.0)
        """
        import re

        score = 0.0
        content_lower = chunk_content.lower()
        query_lower = query.lower()

        # Extract query terms
        query_terms = re.findall(r"\b[a-z_][a-z0-9_]{2,}\b", query_lower)

        # Score based on query term frequency
        for term in query_terms:
            count = content_lower.count(term)
            if count > 0:
                score += min(count * 0.1, 0.5)  # Max 0.5 per term

        # Score based on additional keywords
        if keywords:
            for keyword in keywords:
                if keyword.lower() in content_lower:
                    score += 0.2

        # Normalize to 0-1 range
        return min(score, 1.0)

    def read_file_with_budget(
        self,
        file_path: str,
        token_budget: int,
        query: Optional[str] = None,
        prioritize_relevant: bool = True,
    ) -> Tuple[str, int]:
        """
        Read file respecting token budget.

        Loads chunks progressively, prioritizing relevant sections if query provided.

        Args:
            file_path: Path to file
            token_budget: Maximum tokens to use
            query: Optional query for relevance scoring
            prioritize_relevant: If True, load most relevant chunks first

        Returns:
            (content, tokens_used)
        """
        from core.token_budget import TokenBudgetManager

        budget_mgr = TokenBudgetManager(token_budget)

        # Get all chunks
        chunks = self.read_chunks(file_path)

        if not chunks:
            return "", 0

        # Score chunks by relevance if query provided
        if query and prioritize_relevant:
            scored_chunks = []
            for chunk in chunks:
                relevance = self.estimate_relevance(chunk["content"], query)
                scored_chunks.append((relevance, chunk))

            # Sort by relevance (highest first)
            scored_chunks.sort(reverse=True, key=lambda x: x[0])
            chunks = [chunk for _, chunk in scored_chunks]

        # Load chunks until budget exhausted
        loaded_chunks = []
        for chunk in chunks:
            content = chunk["content"]
            tokens = budget_mgr.count_tokens(content)

            if budget_mgr.fits_in_budget(tokens):
                loaded_chunks.append(chunk)
                budget_mgr.allocate(tokens, f"chunk_{chunk['start_line']}")
            else:
                # Budget exhausted
                break

        # Combine chunks (sort by line number for correct order)
        loaded_chunks.sort(key=lambda x: x["start_line"])

        combined_content = "\n".join(
            [
                f"# Lines {chunk['start_line']}-{chunk['end_line']}\n{chunk['content']}"
                for chunk in loaded_chunks
            ]
        )

        return combined_content, budget_mgr.allocated


def read_large_file_progressively(
    file_path: str,
    query: str,
    token_budget: int = 10000,
    grep_matches: Optional[List[int]] = None,
) -> str:
    """
    Convenience function for progressive large file reading.

    Args:
        file_path: Path to file
        query: Search query for relevance scoring
        token_budget: Maximum tokens to use
        grep_matches: Optional list of line numbers from grep

    Returns:
        Loaded content within budget
    """
    reader = ChunkedFileReader()

    if grep_matches:
        # Load sections around grep matches
        sections = reader.read_around_matches(file_path, grep_matches, context_lines=20)

        # Combine sections
        combined = "\n\n".join(
            [
                f"# Lines {sec['start_line']}-{sec['end_line']} (matched lines: {sec['match_lines']})\n{sec['content']}"
                for sec in sections
            ]
        )

        # Truncate if over budget
        from core.token_budget import TokenBudgetManager, smart_truncate

        budget_mgr = TokenBudgetManager(token_budget)
        tokens = budget_mgr.count_tokens(combined)

        if tokens > token_budget:
            combined = smart_truncate(combined, token_budget)

        return combined
    else:
        # Progressive relevance-based loading
        content, tokens_used = reader.read_file_with_budget(
            file_path, token_budget, query=query, prioritize_relevant=True
        )

        return content
