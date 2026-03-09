# Test Questions for Large Document Set

## Document Overview
- 25 interconnected documents
- Each document 3-6 pages
- Extensive cross-references between documents
- Total content: ~100+ pages

## Test Questions

### Level 1: Single Document (Easy)
```bash
uv run explore --task "Look in data/large_acquisition/. What is the total purchase price?"
uv run explore --task "Look in data/large_acquisition/. Who is the CTO and what is their retention bonus?"
uv run explore --task "Look in data/large_acquisition/. What patents does the company own?"
```

### Level 2: Cross-Reference Required (Medium)
```bash
uv run explore --task "Look in data/large_acquisition/. What customer consents are required and what is their status?"
uv run explore --task "Look in data/large_acquisition/. What is the total litigation exposure and is it covered by insurance?"
uv run explore --task "Look in data/large_acquisition/. How is the purchase price being paid and what are the escrow terms?"
```

### Level 3: Multi-Document Synthesis (Hard)
```bash
uv run explore --task "Look in data/large_acquisition/. What are all the conditions that must be satisfied before closing and what is the status of each?"
uv run explore --task "Look in data/large_acquisition/. Provide a complete picture of MegaCorp's relationship with the company - revenue, contract terms, consent status, and any risks."
uv run explore --task "Look in data/large_acquisition/. What are all the financial terms of this deal including adjustments, escrow, earnouts, and stock?"
```

### Level 4: Deep Cross-Reference (Expert)
```bash
uv run explore --task "Look in data/large_acquisition/. Trace all references to the Legal Opinion Letter - what documents cite it and what opinions does it provide?"
uv run explore --task "Look in data/large_acquisition/. Create a complete picture of IP assets - patents, trademarks, assignments, and any related risks or litigation."
uv run explore --task "Look in data/large_acquisition/. What happens after closing? List all post-closing obligations, their timelines, and related documents."
```
