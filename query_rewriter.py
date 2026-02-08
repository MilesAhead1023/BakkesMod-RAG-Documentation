"""
Query Rewriting & Expansion
============================
Rewrites user queries with domain-specific synonyms and terminology.
"""

import os
from typing import Optional
from llama_index.core.llms import LLM


class QueryRewriter:
    """Rewrites queries to improve retrieval using domain knowledge."""

    # BakkesMod-specific terminology mappings
    DOMAIN_SYNONYMS = {
        # Core concepts
        "plugin": ["mod", "extension", "addon", "modification"],
        "hook": ["attach", "subscribe", "listen", "register", "bind"],
        "event": ["callback", "trigger", "signal", "notification"],

        # Game entities
        "car": ["vehicle", "CarWrapper"],
        "ball": ["BallWrapper", "soccer ball", "game ball"],
        "player": ["PlayerController", "PRI", "PlayerReplicationInfo"],
        "goal": ["score", "GoalEvent"],

        # UI concepts
        "GUI": ["ImGui", "interface", "menu", "window"],
        "settings": ["config", "configuration", "options", "preferences"],
        "render": ["draw", "display", "paint", "show"],

        # SDK classes
        "GameWrapper": ["gameWrapper", "game wrapper", "game interface"],
        "ServerWrapper": ["server", "game server"],
        "CameraWrapper": ["camera", "view"],

        # Actions
        "create": ["initialize", "instantiate", "make", "build"],
        "get": ["retrieve", "fetch", "access", "obtain"],
        "set": ["assign", "update", "modify", "change"],
        "load": ["initialize", "start", "onLoad"],
        "unload": ["cleanup", "onUnload", "shutdown"],
    }

    def __init__(self, llm: Optional[LLM] = None, use_llm: bool = True):
        """
        Initialize query rewriter.

        Args:
            llm: Language model for intelligent rewriting (optional)
            use_llm: Whether to use LLM for rewriting (default: True)
        """
        self.llm = llm
        self.use_llm = use_llm and llm is not None

    def expand_with_synonyms(self, query: str) -> str:
        """
        Expand query with domain-specific synonyms.

        Args:
            query: Original user query

        Returns:
            Expanded query with synonyms
        """
        expanded_terms = []
        query_lower = query.lower()

        # Check each domain term
        for term, synonyms in self.DOMAIN_SYNONYMS.items():
            if term.lower() in query_lower:
                # Add original term
                expanded_terms.append(term)
                # Add top 2 synonyms
                expanded_terms.extend(synonyms[:2])

        if expanded_terms:
            # Add synonyms to original query
            synonym_text = " ".join(expanded_terms)
            return f"{query} ({synonym_text})"

        return query

    def rewrite_with_llm(self, query: str) -> str:
        """
        Use LLM to intelligently rewrite query for better retrieval.

        Args:
            query: Original user query

        Returns:
            Rewritten query optimized for BakkesMod SDK
        """
        if not self.use_llm:
            return query

        prompt = f"""You are a BakkesMod SDK expert. Rewrite this query to be more specific and include relevant technical terms from the BakkesMod SDK.

Original query: {query}

Rewrite the query to:
1. Use correct BakkesMod terminology (e.g., "hook event" → "HookEvent with gameWrapper")
2. Include relevant class names (e.g., "car" → "CarWrapper")
3. Expand abbreviations (e.g., "GUI" → "ImGui interface")
4. Keep it concise (1-2 sentences max)

Rewritten query:"""

        try:
            response = self.llm.complete(prompt)
            rewritten = response.text.strip()

            # Fallback to synonym expansion if LLM fails
            if not rewritten or len(rewritten) < 5:
                return self.expand_with_synonyms(query)

            return rewritten

        except Exception as e:
            print(f"[WARNING] LLM rewriting failed: {e}, using synonym expansion")
            return self.expand_with_synonyms(query)

    def rewrite(self, query: str) -> str:
        """
        Rewrite query using best available method.

        Args:
            query: Original user query

        Returns:
            Rewritten/expanded query
        """
        if self.use_llm:
            return self.rewrite_with_llm(query)
        else:
            return self.expand_with_synonyms(query)


# Convenience function
def rewrite_query(query: str, llm: Optional[LLM] = None, use_llm: bool = True) -> str:
    """
    Rewrite a query for better retrieval.

    Args:
        query: Original user query
        llm: Optional LLM for intelligent rewriting
        use_llm: Whether to use LLM (default: True if llm provided)

    Returns:
        Rewritten query
    """
    rewriter = QueryRewriter(llm=llm, use_llm=use_llm)
    return rewriter.rewrite(query)


if __name__ == "__main__":
    # Test synonym expansion (no LLM needed)
    print("\n=== Query Rewriting Examples ===\n")

    rewriter = QueryRewriter(use_llm=False)

    test_queries = [
        "How do I hook events?",
        "How to create a plugin?",
        "Access car data",
        "Set up GUI",
        "Get player stats",
    ]

    for query in test_queries:
        expanded = rewriter.expand_with_synonyms(query)
        print(f"Original: {query}")
        print(f"Expanded: {expanded}")
        print()

    print("=" * 80)
    print("NOTE: LLM-based rewriting requires API key and model initialization")
    print("=" * 80)
