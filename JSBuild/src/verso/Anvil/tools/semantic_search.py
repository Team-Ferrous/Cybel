def semantic_search_tool(query: str, k: int = 5, engine=None) -> str:
    """
    Compatibility alias for Saguaro semantic query.
    """
    if engine and hasattr(engine, "substrate"):
        results = engine.substrate.agent_query(query, k=k)
        if "Error" in results:
            return f"Semantic Search (Saguaro) Error: {results}"
        return f"Semantic Search Results (Saguaro Q-COS):\n{results}"

    return (
        "Error: semantic_search is deprecated in strict grounding mode. "
        "Use saguaro_query."
    )
