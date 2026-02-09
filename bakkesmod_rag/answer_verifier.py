"""
Answer Verification
===================
Verifies that generated answers are grounded in the retrieved source
documents. Uses a two-tier approach:

1. **Embedding check** (fast, free): cosine similarity between answer
   and source embeddings.
2. **LLM check** (borderline only): asks the LLM whether the answer
   is supported by the sources.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

logger = logging.getLogger("bakkesmod_rag.answer_verifier")


@dataclass
class VerificationResult:
    """Result of answer verification against source documents.

    Attributes:
        grounded: Whether the answer is supported by sources.
        grounding_score: Cosine similarity score (0.0–1.0).
        confidence_penalty: Amount to subtract from confidence.
        warning: User-facing warning if answer appears ungrounded.
        ungrounded_claims: Specific claims not found in sources.
    """

    grounded: bool = True
    grounding_score: float = 1.0
    confidence_penalty: float = 0.0
    warning: Optional[str] = None
    ungrounded_claims: List[str] = field(default_factory=list)


class AnswerVerifier:
    """Verifies answers are grounded in retrieved source documents.

    Args:
        embed_model: Embedding model for cosine similarity check.
        llm: LLM for borderline verification (optional).
        grounded_threshold: Score above which answer is considered grounded.
        borderline_threshold: Score below which answer is considered ungrounded
            (between this and grounded_threshold triggers LLM check).
        borderline_penalty: Confidence penalty for borderline-ungrounded.
        ungrounded_penalty: Confidence penalty for clearly ungrounded.
        enabled: Master toggle for verification.
    """

    def __init__(
        self,
        embed_model=None,
        llm=None,
        grounded_threshold: float = 0.75,
        borderline_threshold: float = 0.55,
        borderline_penalty: float = 0.15,
        ungrounded_penalty: float = 0.30,
        enabled: bool = True,
    ) -> None:
        self.embed_model = embed_model
        self.llm = llm
        self.grounded_threshold = grounded_threshold
        self.borderline_threshold = borderline_threshold
        self.borderline_penalty = borderline_penalty
        self.ungrounded_penalty = ungrounded_penalty
        self.enabled = enabled

    def verify(
        self,
        answer: str,
        source_nodes: list,
        query: str = "",
    ) -> VerificationResult:
        """Verify that the answer is grounded in source documents.

        Args:
            answer: The generated answer text.
            source_nodes: Retrieved source nodes (LlamaIndex NodeWithScore).
            query: The original query (for LLM context).

        Returns:
            VerificationResult with grounding assessment.
        """
        if not self.enabled:
            return VerificationResult()

        if not answer or not answer.strip():
            return VerificationResult()

        if not source_nodes:
            return VerificationResult(
                grounded=False,
                grounding_score=0.0,
                confidence_penalty=self.ungrounded_penalty,
                warning="No source documents were retrieved to verify this answer.",
            )

        # Step 1: Embedding check (fast)
        embedding_score = self._embedding_check(answer, source_nodes)

        if embedding_score >= self.grounded_threshold:
            return VerificationResult(
                grounded=True,
                grounding_score=embedding_score,
            )

        if embedding_score < self.borderline_threshold:
            return VerificationResult(
                grounded=False,
                grounding_score=embedding_score,
                confidence_penalty=self.ungrounded_penalty,
                warning=(
                    "This answer may not be fully supported by the documentation. "
                    "Please verify the information independently."
                ),
            )

        # Step 2: Borderline — escalate to LLM if available
        if self.llm is not None:
            return self._llm_check(answer, source_nodes, query, embedding_score)

        # No LLM available — treat borderline as cautionary
        return VerificationResult(
            grounded=True,
            grounding_score=embedding_score,
            confidence_penalty=self.borderline_penalty,
            warning=(
                "This answer has moderate grounding in the documentation. "
                "Some details may need verification."
            ),
        )

    def _embedding_check(self, answer: str, source_nodes: list) -> float:
        """Calculate cosine similarity between answer and source embeddings.

        Args:
            answer: The answer text.
            source_nodes: Source nodes with text content.

        Returns:
            Average cosine similarity (0.0–1.0).
        """
        if not self.embed_model:
            return 1.0  # No embed model — assume grounded

        try:
            answer_embedding = self.embed_model.get_text_embedding(answer)

            source_texts = []
            for node in source_nodes:
                text = getattr(node, "text", None)
                if text is None and hasattr(node, "node"):
                    text = getattr(node.node, "text", None)
                if text and text.strip():
                    source_texts.append(text)

            if not source_texts:
                return 0.0

            similarities = []
            for source_text in source_texts:
                source_embedding = self.embed_model.get_text_embedding(source_text)
                sim = self._cosine_similarity(answer_embedding, source_embedding)
                similarities.append(sim)

            return float(np.mean(similarities))

        except Exception as e:
            logger.warning("Embedding check failed: %s", e)
            return 1.0  # Fail open — don't block the answer

    def _llm_check(
        self,
        answer: str,
        source_nodes: list,
        query: str,
        embedding_score: float,
    ) -> VerificationResult:
        """Use LLM to verify borderline answers.

        Args:
            answer: The answer text.
            source_nodes: Source nodes for context.
            query: Original query.
            embedding_score: Pre-computed embedding similarity.

        Returns:
            VerificationResult from LLM analysis.
        """
        source_texts = []
        for node in source_nodes:
            text = getattr(node, "text", None)
            if text is None and hasattr(node, "node"):
                text = getattr(node.node, "text", None)
            if text and text.strip():
                source_texts.append(text[:500])  # Truncate for prompt

        combined_sources = "\n---\n".join(source_texts[:5])

        prompt = (
            "You are a fact-checking assistant. Given the source documents and "
            "the answer below, determine if the answer is fully supported by "
            "the sources.\n\n"
            f"Question: {query}\n\n"
            f"Source Documents:\n{combined_sources}\n\n"
            f"Answer: {answer}\n\n"
            "Respond with JSON only:\n"
            '{"grounded": true/false, "unsupported_claims": ["claim1", ...]}\n\n'
            "If fully grounded, set unsupported_claims to an empty list."
        )

        try:
            response = self.llm.complete(prompt)
            text = response.text.strip()

            # Try to extract JSON from response
            result = self._parse_llm_json(text)

            is_grounded = result.get("grounded", True)
            claims = result.get("unsupported_claims", [])

            if is_grounded:
                return VerificationResult(
                    grounded=True,
                    grounding_score=embedding_score,
                )

            return VerificationResult(
                grounded=False,
                grounding_score=embedding_score,
                confidence_penalty=self.borderline_penalty,
                warning=(
                    "Some parts of this answer may not be directly supported "
                    "by the documentation. Please verify key details."
                ),
                ungrounded_claims=claims if isinstance(claims, list) else [],
            )

        except Exception as e:
            logger.warning("LLM verification failed: %s", e)
            # Fail open — apply mild penalty but don't block
            return VerificationResult(
                grounded=True,
                grounding_score=embedding_score,
                confidence_penalty=self.borderline_penalty * 0.5,
            )

    @staticmethod
    def _parse_llm_json(text: str) -> dict:
        """Extract JSON from LLM response text.

        Handles cases where LLM wraps JSON in markdown code blocks.

        Args:
            text: Raw LLM response text.

        Returns:
            Parsed dict.
        """
        # Strip markdown code blocks
        if "```" in text:
            lines = text.split("```")
            for block in lines:
                block = block.strip()
                if block.startswith("json"):
                    block = block[4:].strip()
                if block.startswith("{"):
                    try:
                        return json.loads(block)
                    except json.JSONDecodeError:
                        continue

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try finding JSON in text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass

        return {"grounded": True, "unsupported_claims": []}

    @staticmethod
    def _cosine_similarity(a: list, b: list) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            a: First embedding vector.
            b: Second embedding vector.

        Returns:
            Cosine similarity (0.0–1.0 for normalized embeddings).
        """
        a_arr = np.array(a)
        b_arr = np.array(b)
        dot = np.dot(a_arr, b_arr)
        norm_a = np.linalg.norm(a_arr)
        norm_b = np.linalg.norm(b_arr)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))
