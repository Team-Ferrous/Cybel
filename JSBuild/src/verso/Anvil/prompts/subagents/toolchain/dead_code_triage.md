# Dead Code Triage Addendum

Mission: Run Saguaro deadcode analysis, classify each candidate as either rewireable or obsolete, and provide concrete follow-up actions.

Operating sequence:
1. Call deadcode with JSON output and record threshold plus candidate count.
2. Classify each candidate into one of: rewire_candidate, likely_obsolete, or needs_manual_review.
3. For rewire candidates, provide the target wiring surface (registry, hook, plugin, API export, or tests).
4. For likely obsolete candidates, provide deletion preconditions and validation checks.

Research baseline:
- Use web_search, web_fetch, search_arxiv, and fetch_arxiv_paper when external rationale is required.

Deliverables:
- Deadcode triage ledger with symbol, path, confidence, disposition, rationale, and next action.
- Summary counts by disposition and confidence band.
- Risk notes for false-positive-prone patterns (dynamic loading, reflection, registration-by-side-effect).
