"""
Feedback Store
==============
Persists code generation history and user feedback to enable learning
from past generations. Successful patterns become few-shot examples
for future prompts; common error→fix pairs help avoid recurring
mistakes.

Storage: JSON file at ``.cache/feedback/generations.json``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("bakkesmod_rag.feedback_store")


@dataclass
class FeedbackEntry:
    """A single code generation record with feedback.

    Attributes:
        id: Unique identifier for this generation.
        timestamp: Unix timestamp of the generation.
        description: What the user asked for.
        features: Detected feature flags (e.g. ``["event_hooks", "imgui"]``).
        generated_files: Dict mapping filename to content.
        validation_errors: Errors found during validation.
        validation_warnings: Warnings found during validation.
        fix_iterations: Number of fix attempts made.
        accepted: Whether the user accepted this generation.
        user_edits: Description of what the user changed (optional).
        compile_attempted: Whether compilation was attempted.
        compile_success: Whether compilation succeeded.
        compiler_errors: Compiler error messages (if any).
    """

    id: str = ""
    timestamp: float = 0.0
    description: str = ""
    features: List[str] = field(default_factory=list)
    generated_files: Dict[str, str] = field(default_factory=dict)
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)
    fix_iterations: int = 0
    accepted: bool = False
    user_edits: Optional[str] = None
    compile_attempted: bool = False
    compile_success: bool = False
    compiler_errors: List[str] = field(default_factory=list)


class FeedbackStore:
    """Persists and queries code generation feedback for learning.

    Stores all generation attempts as :class:`FeedbackEntry` records
    in a JSON file. Provides methods to retrieve successful patterns
    for few-shot prompting and common error→fix pairs for avoidance.

    Args:
        feedback_dir: Directory to store the feedback JSON file.
        max_entries: Maximum entries to keep (oldest are pruned).
    """

    FILENAME = "generations.json"

    def __init__(
        self,
        feedback_dir: str = ".cache/feedback",
        max_entries: int = 500,
    ) -> None:
        self.feedback_dir = Path(feedback_dir)
        self.max_entries = max_entries
        self._entries: List[FeedbackEntry] = []
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_generation(
        self,
        description: str,
        features: List[str],
        generated_files: Dict[str, str],
        validation_errors: Optional[List[str]] = None,
        validation_warnings: Optional[List[str]] = None,
        fix_iterations: int = 0,
        accepted: bool = False,
        user_edits: Optional[str] = None,
        compile_attempted: bool = False,
        compile_success: bool = False,
        compiler_errors: Optional[List[str]] = None,
    ) -> str:
        """Record a code generation attempt.

        Args:
            description: User's plugin description.
            features: Detected feature flags.
            generated_files: Generated source files.
            validation_errors: Validation errors found.
            validation_warnings: Validation warnings found.
            fix_iterations: Number of LLM fix iterations.
            accepted: Whether user accepted the output.
            user_edits: What the user changed (if any).
            compile_attempted: Whether compilation was attempted.
            compile_success: Whether compilation succeeded.
            compiler_errors: Compiler error messages.

        Returns:
            Unique ID for this generation record.
        """
        import time

        entry = FeedbackEntry(
            id=str(uuid.uuid4())[:8],
            timestamp=time.time(),
            description=description,
            features=features,
            generated_files=generated_files,
            validation_errors=validation_errors or [],
            validation_warnings=validation_warnings or [],
            fix_iterations=fix_iterations,
            accepted=accepted,
            user_edits=user_edits,
            compile_attempted=compile_attempted,
            compile_success=compile_success,
            compiler_errors=compiler_errors or [],
        )

        self._entries.append(entry)

        # Prune if over limit (remove oldest)
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries:]

        self._save()
        logger.info("Recorded generation %s: %s", entry.id, description[:60])
        return entry.id

    def update_feedback(
        self,
        generation_id: str,
        accepted: bool,
        user_edits: Optional[str] = None,
    ) -> bool:
        """Update feedback for an existing generation.

        Args:
            generation_id: ID returned by :meth:`record_generation`.
            accepted: Whether user accepted the output.
            user_edits: What the user changed.

        Returns:
            True if the entry was found and updated.
        """
        for entry in self._entries:
            if entry.id == generation_id:
                entry.accepted = accepted
                entry.user_edits = user_edits
                self._save()
                return True
        return False

    def get_successful_patterns(
        self,
        target_features: Optional[List[str]] = None,
        max_results: int = 3,
    ) -> List[dict]:
        """Retrieve accepted generations as few-shot examples.

        Returns the most recent accepted generations, optionally
        filtered by feature overlap with the target description.

        Args:
            target_features: Feature flags to match against (optional).
            max_results: Maximum patterns to return.

        Returns:
            List of dicts with ``description``, ``features``, and
            ``code_snippet`` (first 500 chars of implementation).
        """
        accepted = [e for e in self._entries if e.accepted]

        if target_features:
            # Score by feature overlap
            def overlap_score(entry: FeedbackEntry) -> int:
                return len(set(entry.features) & set(target_features))

            accepted.sort(key=overlap_score, reverse=True)

        # Most recent first (within same overlap)
        accepted = accepted[:max_results * 2]
        accepted.sort(key=lambda e: e.timestamp, reverse=True)
        accepted = accepted[:max_results]

        patterns = []
        for entry in accepted:
            # Find the main implementation file
            impl_content = ""
            for fname, content in entry.generated_files.items():
                if fname.endswith(".cpp") and fname not in ("pch.cpp", "GuiBase.cpp"):
                    impl_content = content[:500]
                    break

            patterns.append({
                "description": entry.description,
                "features": entry.features,
                "code_snippet": impl_content,
            })

        return patterns

    def get_common_fixes(self, max_results: int = 5) -> List[dict]:
        """Aggregate recurring error→fix patterns from past generations.

        Looks at entries that had validation/compiler errors and were
        eventually accepted (meaning the fix loop or user fixed them).

        Args:
            max_results: Maximum fix patterns to return.

        Returns:
            List of dicts with ``error_pattern`` and ``frequency``.
        """
        error_counter: Counter = Counter()

        for entry in self._entries:
            for err in entry.validation_errors:
                # Normalise to a pattern (strip file-specific info)
                pattern = self._normalise_error(err)
                error_counter[pattern] += 1

            for err in entry.compiler_errors:
                pattern = self._normalise_error(err)
                error_counter[pattern] += 1

        fixes = []
        for pattern, count in error_counter.most_common(max_results):
            fixes.append({
                "error_pattern": pattern,
                "frequency": count,
            })

        return fixes

    def get_feedback_stats(self) -> dict:
        """Calculate summary statistics from the feedback store.

        Returns:
            Dict with ``total_generations``, ``acceptance_rate``,
            ``avg_fix_iterations``, ``compile_success_rate``, and
            ``common_features``.
        """
        total = len(self._entries)
        if total == 0:
            return {
                "total_generations": 0,
                "acceptance_rate": 0.0,
                "avg_fix_iterations": 0.0,
                "compile_success_rate": 0.0,
                "common_features": [],
            }

        accepted = sum(1 for e in self._entries if e.accepted)
        avg_fixes = sum(e.fix_iterations for e in self._entries) / total

        compiled = [e for e in self._entries if e.compile_attempted]
        compile_success = (
            sum(1 for e in compiled if e.compile_success) / len(compiled)
            if compiled else 0.0
        )

        # Most common features
        feature_counter: Counter = Counter()
        for entry in self._entries:
            for f in entry.features:
                feature_counter[f] += 1

        return {
            "total_generations": total,
            "acceptance_rate": accepted / total,
            "avg_fix_iterations": avg_fixes,
            "compile_success_rate": compile_success,
            "common_features": feature_counter.most_common(5),
        }

    def format_few_shot_prompt(
        self,
        target_features: Optional[List[str]] = None,
    ) -> str:
        """Format successful patterns as a few-shot prompt section.

        Args:
            target_features: Features to match against.

        Returns:
            Formatted string ready to insert into an LLM prompt,
            or empty string if no patterns available.
        """
        patterns = self.get_successful_patterns(target_features)
        if not patterns:
            return ""

        lines = [
            "EXAMPLES OF PREVIOUSLY SUCCESSFUL PLUGINS:",
            "",
        ]
        for i, p in enumerate(patterns, 1):
            lines.append(f"Example {i}: {p['description']}")
            lines.append(f"Features: {', '.join(p['features'])}")
            if p["code_snippet"]:
                lines.append(f"Code pattern:\n```cpp\n{p['code_snippet']}\n```")
            lines.append("")

        common_fixes = self.get_common_fixes(3)
        if common_fixes:
            lines.append("COMMON MISTAKES TO AVOID:")
            for fix in common_fixes:
                lines.append(
                    f"- {fix['error_pattern']} "
                    f"(occurred {fix['frequency']} times)"
                )
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load entries from JSON file."""
        filepath = self.feedback_dir / self.FILENAME
        if not filepath.exists():
            self._entries = []
            return

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            self._entries = [
                FeedbackEntry(**entry) for entry in data
            ]
            logger.debug(
                "Loaded %d feedback entries from %s",
                len(self._entries), filepath,
            )
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.warning("Failed to load feedback store: %s", e)
            self._entries = []

    def _save(self) -> None:
        """Save entries to JSON file."""
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.feedback_dir / self.FILENAME

        try:
            data = [asdict(entry) for entry in self._entries]
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error("Failed to save feedback store: %s", e)

    @staticmethod
    def _normalise_error(error_msg: str) -> str:
        """Normalise an error message for pattern aggregation.

        Strips file paths, line numbers, and specific identifiers to
        find the underlying error pattern.

        Args:
            error_msg: Raw error message string.

        Returns:
            Normalised pattern string.
        """
        # Remove file paths
        normalised = re.sub(r"[\w/\\]+\.(cpp|h|hpp)", "<file>", error_msg)
        # Remove line/position numbers
        normalised = re.sub(r"(line|position|at)\s+\d+", r"\1 N", normalised)
        # Remove specific identifiers in quotes
        normalised = re.sub(r"'[^']+?'", "'<id>'", normalised)
        return normalised.strip()
