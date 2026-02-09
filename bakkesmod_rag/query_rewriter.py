"""
Query Rewriting & Expansion
============================
Rewrites user queries with domain-specific synonyms and terminology.
"""

from typing import Optional


class QueryRewriter:
    """Rewrites queries to improve retrieval using domain knowledge."""

    DOMAIN_SYNONYMS = {
        "plugin": ["mod", "extension", "addon", "modification"],
        "hook": ["attach", "subscribe", "listen", "register", "bind"],
        "event": ["callback", "trigger", "signal", "notification"],
        "car": ["vehicle", "CarWrapper"],
        "ball": ["BallWrapper", "soccer ball", "game ball"],
        "player": ["PlayerController", "PRI", "PlayerReplicationInfo"],
        "goal": ["score", "GoalEvent"],
        "GUI": ["ImGui", "interface", "menu", "window"],
        "settings": ["config", "configuration", "options", "preferences"],
        "render": ["draw", "display", "paint", "show"],
        "GameWrapper": ["gameWrapper", "game wrapper", "game interface"],
        "ServerWrapper": ["server", "game server"],
        "CameraWrapper": ["camera", "view"],
        "create": ["initialize", "instantiate", "make", "build"],
        "get": ["retrieve", "fetch", "access", "obtain"],
        "set": ["assign", "update", "modify", "change"],
        "load": ["initialize", "start", "onLoad"],
        "unload": ["cleanup", "onUnload", "shutdown"],
    }

    def __init__(self, llm=None, use_llm: bool = True):
        """Initialize query rewriter.

        Args:
            llm: Language model for intelligent rewriting (optional).
            use_llm: Whether to use LLM for rewriting (default: True).
        """
        self.llm = llm
        self.use_llm = use_llm and llm is not None

    def expand_with_synonyms(self, query: str) -> str:
        """Expand query with domain-specific synonyms.

        Args:
            query: Original user query.

        Returns:
            Expanded query with synonyms.
        """
        expanded_terms = []
        query_lower = query.lower()

        for term, synonyms in self.DOMAIN_SYNONYMS.items():
            if term.lower() in query_lower:
                expanded_terms.append(term)
                expanded_terms.extend(synonyms[:2])

        if expanded_terms:
            synonym_text = " ".join(expanded_terms)
            return f"{query} ({synonym_text})"

        return query

    def rewrite_with_llm(self, query: str) -> str:
        """Use LLM to intelligently rewrite query for better retrieval.

        Args:
            query: Original user query.

        Returns:
            Rewritten query optimized for BakkesMod SDK.
        """
        if self.llm is None:
            return query

        prompt = (
            "You are a BakkesMod SDK expert. Rewrite this query to be more "
            "specific and include relevant technical terms from the BakkesMod SDK.\n\n"
            f"Original query: {query}\n\n"
            "Rewrite the query to:\n"
            '1. Use correct BakkesMod terminology (e.g., "hook event" -> "HookEvent with gameWrapper")\n'
            '2. Include relevant class names (e.g., "car" -> "CarWrapper")\n'
            '3. Expand abbreviations (e.g., "GUI" -> "ImGui interface")\n'
            "4. Keep it concise (1-2 sentences max)\n\n"
            "Rewritten query:"
        )

        try:
            response = self.llm.complete(prompt)
            rewritten = response.text.strip()

            if not rewritten or len(rewritten) < 5:
                return self.expand_with_synonyms(query)

            return rewritten

        except Exception:
            return self.expand_with_synonyms(query)

    def rewrite(self, query: str, force_llm: bool = False) -> str:
        """Rewrite query using best available method.

        Args:
            query: Original user query.
            force_llm: If True, use LLM rewriting even if not configured.
                Falls back to synonym expansion if no LLM available.

        Returns:
            Rewritten/expanded query.
        """
        if force_llm and self.llm is not None:
            return self.rewrite_with_llm(query)
        if self.use_llm:
            return self.rewrite_with_llm(query)
        else:
            return self.expand_with_synonyms(query)
