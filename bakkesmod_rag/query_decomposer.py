"""
Query Decomposition
====================
Breaks complex multi-part queries into focused sub-questions,
retrieves answers for each, and merges them into a single coherent response.

Uses a hybrid approach:
1. **Rule-based splitting** (free, instant) — splits on conjunctions,
   semicolons, numbered lists.
2. **LLM-powered decomposition** (fallback) — for complex queries where
   rule-based splitting fails to find clear split points.
"""

from __future__ import annotations

import re
import logging
from typing import List, Optional

logger = logging.getLogger("bakkesmod_rag.query_decomposer")

# Patterns that indicate separable sub-questions
_SPLIT_PATTERNS = [
    r"\band\b\s+(?:also\s+)?(?:how|what|where|when|why|can|do|does|is|are)\b",
    r";\s+",
    r"\?\s+",
    r"\b(?:also|additionally|furthermore)\b[,]?\s+",
]
_SPLIT_RE = re.compile("|".join(_SPLIT_PATTERNS), re.IGNORECASE)

# Numbered list pattern: "1. ... 2. ..." or "1) ... 2) ..."
_NUMBERED_RE = re.compile(r"(?:^|\n)\s*\d+[.)]\s+", re.MULTILINE)

# Heuristics for complexity detection
_QUESTION_WORDS = re.compile(
    r"\b(?:how|what|where|when|why|can|does|do|is|are|which|should)\b",
    re.IGNORECASE,
)


class QueryDecomposer:
    """Decomposes complex queries into focused sub-questions.

    Args:
        llm: LlamaIndex LLM instance for LLM-powered decomposition.
        max_sub_queries: Maximum number of sub-queries allowed.
        complexity_threshold: Character-length threshold above which
            a query is considered potentially complex.
        enable_decomposition: Master toggle; when *False*, ``decompose``
            always returns the original query as-is.
    """

    def __init__(
        self,
        llm=None,
        max_sub_queries: int = 4,
        complexity_threshold: int = 80,
        enable_decomposition: bool = True,
    ) -> None:
        self.llm = llm
        self.max_sub_queries = max_sub_queries
        self.complexity_threshold = complexity_threshold
        self.enabled = enable_decomposition

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decompose(self, query: str) -> List[str]:
        """Break *query* into independent sub-questions.

        Returns a list with **at least one** element (the original query
        if no decomposition applies).

        Args:
            query: The user's natural-language question.

        Returns:
            List of sub-question strings (1 … ``max_sub_queries``).
        """
        if not self.enabled or not query or not query.strip():
            return [query]

        # 1. Try rule-based split first (free, instant)
        sub_queries = self._rule_based_split(query)
        if len(sub_queries) > 1:
            logger.info(
                "Rule-based decomposition: %d sub-queries from: %s",
                len(sub_queries),
                query[:80],
            )
            return sub_queries[: self.max_sub_queries]

        # 2. Check if query is complex enough for LLM decomposition
        if self._is_complex(query) and self.llm is not None:
            sub_queries = self._llm_split(query)
            if len(sub_queries) > 1:
                logger.info(
                    "LLM decomposition: %d sub-queries from: %s",
                    len(sub_queries),
                    query[:80],
                )
                return sub_queries[: self.max_sub_queries]

        # 3. No decomposition needed — return original
        return [query]

    # ------------------------------------------------------------------
    # Rule-based splitting
    # ------------------------------------------------------------------

    def _rule_based_split(self, query: str) -> List[str]:
        """Split on conjunctions, semicolons, question marks, numbered lists.

        Args:
            query: The user's query.

        Returns:
            List of sub-queries (may be length 1 if no split found).
        """
        # Check for numbered lists first (e.g., "1. ... 2. ...")
        numbered_parts = _NUMBERED_RE.split(query)
        numbered_parts = [p.strip().rstrip("?").strip() for p in numbered_parts if p.strip()]
        if len(numbered_parts) > 1:
            return [p + "?" if not p.endswith("?") else p for p in numbered_parts if len(p) > 5]

        # Split on conjunction patterns
        parts = _SPLIT_RE.split(query)
        parts = [p.strip().rstrip("?").strip() for p in parts if p.strip()]
        if len(parts) > 1:
            cleaned = []
            for p in parts:
                if len(p) > 10:
                    if not p.endswith("?"):
                        p = p + "?"
                    cleaned.append(p)
            if len(cleaned) > 1:
                return cleaned

        return [query]

    # ------------------------------------------------------------------
    # Complexity heuristic
    # ------------------------------------------------------------------

    def _is_complex(self, query: str) -> bool:
        """Determine whether the query is complex enough to warrant LLM decomposition.

        Heuristics:
        - Length exceeds ``complexity_threshold``
        - Contains multiple question words (how…what…)
        - Contains multiple BakkesMod API terms

        Args:
            query: The user's query.

        Returns:
            True if the query appears complex.
        """
        if len(query) < self.complexity_threshold:
            return False

        question_word_count = len(_QUESTION_WORDS.findall(query))
        if question_word_count >= 2:
            return True

        # Multiple distinct BakkesMod API terms
        api_terms = [
            "GameWrapper", "ServerWrapper", "CarWrapper", "BallWrapper",
            "PlayerController", "HookEvent", "ImGui", "CVar", "onLoad",
            "onUnload", "plugin", "hook", "render", "canvas",
        ]
        api_hits = sum(1 for t in api_terms if t.lower() in query.lower())
        if api_hits >= 2:
            return True

        return False

    # ------------------------------------------------------------------
    # LLM-powered splitting
    # ------------------------------------------------------------------

    def _llm_split(self, query: str) -> List[str]:
        """Use the LLM to decompose a complex query.

        Falls back to ``[query]`` if the LLM call fails or returns
        unusable output.

        Args:
            query: The user's query.

        Returns:
            List of sub-queries.
        """
        prompt = (
            "You are a BakkesMod SDK documentation assistant. "
            "Break the following complex question into independent, "
            "self-contained sub-questions that can each be answered separately.\n\n"
            "Rules:\n"
            "- Return one sub-question per line\n"
            "- Each sub-question must be answerable on its own\n"
            f"- Maximum {self.max_sub_queries} sub-questions\n"
            "- If the question is already simple, return it unchanged\n"
            "- Do NOT add numbering or bullet points\n\n"
            f"Question: {query}\n\n"
            "Sub-questions:"
        )

        try:
            response = self.llm.complete(prompt)
            text = response.text.strip()

            lines = [
                line.strip().lstrip("0123456789.-) ").strip()
                for line in text.splitlines()
                if line.strip() and len(line.strip()) > 5
            ]

            if not lines:
                return [query]

            # Ensure each line ends with '?'
            cleaned = []
            for line in lines:
                if not line.endswith("?"):
                    line = line + "?"
                cleaned.append(line)

            return cleaned[: self.max_sub_queries]

        except Exception as e:
            logger.warning("LLM decomposition failed: %s", e)
            return [query]

    # ------------------------------------------------------------------
    # Result merging
    # ------------------------------------------------------------------

    @staticmethod
    def merge_sub_answers(
        sub_queries: List[str],
        sub_answers: List[str],
        llm=None,
    ) -> str:
        """Merge multiple sub-query answers into one coherent response.

        If an LLM is provided, uses it to synthesize. Otherwise,
        concatenates with section headers.

        Args:
            sub_queries: The sub-question strings.
            sub_answers: The answer for each sub-question (same order).
            llm: Optional LLM for intelligent synthesis.

        Returns:
            A single merged answer string.
        """
        if len(sub_answers) == 1:
            return sub_answers[0]

        if llm is not None:
            return QueryDecomposer._llm_merge(sub_queries, sub_answers, llm)

        # Fallback: structured concatenation
        parts = []
        for q, a in zip(sub_queries, sub_answers):
            parts.append(f"**{q}**\n{a}")
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _llm_merge(
        sub_queries: List[str],
        sub_answers: List[str],
        llm,
    ) -> str:
        """Use LLM to synthesize sub-answers into a coherent response.

        Args:
            sub_queries: Sub-question strings.
            sub_answers: Corresponding answers.
            llm: LlamaIndex LLM instance.

        Returns:
            Synthesized answer string.
        """
        qa_pairs = "\n\n".join(
            f"Q: {q}\nA: {a}" for q, a in zip(sub_queries, sub_answers)
        )

        prompt = (
            "You are a BakkesMod SDK documentation assistant. "
            "I asked a complex question that was broken into sub-questions. "
            "Below are the sub-questions and their answers.\n\n"
            f"{qa_pairs}\n\n"
            "Now synthesize these into a single, coherent, well-organized "
            "answer that addresses the original question completely. "
            "Do not repeat information. Be concise and technical."
        )

        try:
            response = llm.complete(prompt)
            merged = response.text.strip()
            if merged and len(merged) > 10:
                return merged
        except Exception as e:
            logger.warning("LLM merge failed, using concatenation: %s", e)

        # Fallback to concatenation
        parts = []
        for q, a in zip(sub_queries, sub_answers):
            parts.append(f"**{q}**\n{a}")
        return "\n\n---\n\n".join(parts)
