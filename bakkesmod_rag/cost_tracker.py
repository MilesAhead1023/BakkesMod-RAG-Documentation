"""
Cost Tracking and Budget Management
====================================
Real-time cost tracking for RAG operations with budget alerts.
"""

import json
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime, timedelta
from threading import Lock


class CostTracker:
    """Tracks API costs in real-time with budget management."""

    def __init__(self, config=None, persistence_path: Optional[Path] = None):
        """Initialize cost tracker.

        Args:
            config: RAGConfig instance (optional, uses defaults if None).
            persistence_path: Path to persist cost data.
        """
        if config is not None:
            self.cost_config = config.cost
        else:
            self.cost_config = None

        self.persistence_path = persistence_path or Path(".cache/cost_tracker.json")
        self.persistence_path.parent.mkdir(parents=True, exist_ok=True)

        self.costs: Dict = self._load_costs()
        self.lock = Lock()
        self._tokenizers = None

    @property
    def track_costs(self) -> bool:
        if self.cost_config is not None:
            return self.cost_config.track_costs
        return True

    @property
    def daily_budget_usd(self) -> Optional[float]:
        if self.cost_config is not None:
            return self.cost_config.daily_budget_usd
        return None

    @property
    def alert_threshold_pct(self) -> float:
        if self.cost_config is not None:
            return self.cost_config.alert_threshold_pct
        return 80.0

    @property
    def tokenizers(self):
        """Lazy-load tokenizers on first use."""
        if self._tokenizers is None:
            try:
                import tiktoken

                self._tokenizers = {
                    "gpt-4": tiktoken.encoding_for_model("gpt-4"),
                    "gpt-3.5": tiktoken.encoding_for_model("gpt-3.5-turbo"),
                }
            except Exception:
                self._tokenizers = {}
        return self._tokenizers

    def _load_costs(self) -> Dict:
        """Load cost history from disk."""
        if self.persistence_path.exists():
            try:
                with open(self.persistence_path, "r") as f:
                    return json.load(f)
            except Exception:
                pass

        return {
            "total": 0.0,
            "by_date": {},
            "by_provider": {},
            "by_operation": {},
        }

    def _save_costs(self):
        """Persist costs to disk."""
        try:
            with open(self.persistence_path, "w") as f:
                json.dump(self.costs, f, indent=2)
        except Exception:
            pass

    def count_tokens(self, text: str, model: str = "gpt-4") -> int:
        """Count tokens in text for a given model."""
        if not self.tokenizers:
            return len(text) // 4

        encoder = self.tokenizers.get(model, self.tokenizers.get("gpt-4"))
        if not encoder:
            return len(text) // 4

        return len(encoder.encode(text))

    def track_embedding(self, num_tokens: int, provider: str = "openai", model: str = "text-embedding-3-small"):
        """Track embedding API cost."""
        if not self.track_costs:
            return

        cost_per_1m = 0.02  # text-embedding-3-small default
        if self.cost_config:
            cost_per_1m = self.cost_config.openai_embedding_cost
        cost = (num_tokens / 1_000_000) * cost_per_1m

        self._record_cost(cost, provider, "embedding", model)

    def track_llm_call(
        self,
        input_tokens: int,
        output_tokens: int,
        provider: str,
        model: str,
    ):
        """Track LLM API cost."""
        if not self.track_costs:
            return

        # Determine cost rates
        if provider == "openai":
            input_cost = 0.15
            output_cost = 0.60
            if self.cost_config:
                input_cost = self.cost_config.openai_gpt4o_mini_input
                output_cost = self.cost_config.openai_gpt4o_mini_output
        elif provider == "anthropic":
            input_cost = 3.0
            output_cost = 15.0
            if self.cost_config:
                input_cost = self.cost_config.anthropic_claude_sonnet_input
                output_cost = self.cost_config.anthropic_claude_sonnet_output
        elif provider == "gemini":
            input_cost = 0.075
            output_cost = 0.30
            if self.cost_config:
                input_cost = self.cost_config.gemini_flash_input
                output_cost = self.cost_config.gemini_flash_output
        elif provider in ("openrouter", "deepseek"):
            # Free tier — no cost
            return
        else:
            return

        cost = (input_tokens / 1_000_000) * input_cost + (
            output_tokens / 1_000_000
        ) * output_cost

        self._record_cost(cost, provider, "llm", model)

    def _record_cost(self, cost: float, provider: str, operation: str, model: str):
        """Record a cost entry."""
        with self.lock:
            self.costs["total"] += cost

            today = datetime.now().strftime("%Y-%m-%d")
            if today not in self.costs["by_date"]:
                self.costs["by_date"][today] = 0.0
            self.costs["by_date"][today] += cost

            if provider not in self.costs["by_provider"]:
                self.costs["by_provider"][provider] = 0.0
            self.costs["by_provider"][provider] += cost

            op_key = f"{provider}:{operation}:{model}"
            if op_key not in self.costs["by_operation"]:
                self.costs["by_operation"][op_key] = 0.0
            self.costs["by_operation"][op_key] += cost

            self._check_budget()
            self._save_costs()

    def _check_budget(self):
        """Check if daily budget is exceeded."""
        if not self.daily_budget_usd:
            return

        today = datetime.now().strftime("%Y-%m-%d")
        daily_cost = self.costs["by_date"].get(today, 0.0)

        threshold = self.daily_budget_usd * (self.alert_threshold_pct / 100)

        if daily_cost >= self.daily_budget_usd:
            raise RuntimeError(
                f"Daily budget exceeded: ${daily_cost:.4f} / ${self.daily_budget_usd:.2f}"
            )
        elif daily_cost >= threshold:
            print(
                f"Budget Alert: ${daily_cost:.4f} / ${self.daily_budget_usd:.2f} "
                f"({(daily_cost / self.daily_budget_usd) * 100:.1f}%)"
            )

    def get_daily_cost(self, date: Optional[str] = None) -> float:
        """Get cost for a specific date (or today)."""
        date = date or datetime.now().strftime("%Y-%m-%d")
        return self.costs["by_date"].get(date, 0.0)

    def reset_daily(self, date: Optional[str] = None):
        """Reset costs for a specific date (or today).

        Args:
            date: Date string in ``YYYY-MM-DD`` format; defaults to today.
        """
        date = date or datetime.now().strftime("%Y-%m-%d")
        with self.lock:
            self.costs["by_date"][date] = 0.0
            self._save_costs()

    def get_report(self, days: int = 7) -> str:
        """Generate a cost report for the last N days."""
        lines = ["=" * 60]
        lines.append("Cost Report")
        lines.append("=" * 60)
        lines.append(f"Total All-Time: ${self.costs['total']:.4f}")
        lines.append("")

        lines.append(f"Last {days} Days:")
        for i in range(days):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            cost = self.costs["by_date"].get(date, 0.0)
            lines.append(f"  {date}: ${cost:.4f}")
        lines.append("")

        lines.append("By Provider:")
        for provider, cost in sorted(
            self.costs["by_provider"].items(), key=lambda x: x[1], reverse=True
        ):
            lines.append(f"  {provider}: ${cost:.4f}")
        lines.append("")

        lines.append("Top 5 Operations:")
        sorted_ops = sorted(
            self.costs["by_operation"].items(), key=lambda x: x[1], reverse=True
        )[:5]
        for op, cost in sorted_ops:
            lines.append(f"  {op}: ${cost:.4f}")

        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Redis-backed cost tracker (multi-process safe)
# ---------------------------------------------------------------------------

class RedisCostTracker(CostTracker):
    """CostTracker with Redis-backed atomic token counters.

    Inherits all CostTracker methods but uses Redis INCR for cost
    accumulation — making it safe to use from multiple worker processes.

    Falls back to in-memory/file counters if Redis is unavailable.
    """

    def __init__(
        self,
        config=None,
        redis_url: str = "redis://localhost:6379",
        persistence_path: Optional[Path] = None,
    ):
        super().__init__(config=config, persistence_path=persistence_path)
        self._redis = None

        try:
            import redis as _redis
            client = _redis.from_url(redis_url, socket_connect_timeout=2)
            client.ping()
            self._redis = client
            import logging as _logging
            _logging.getLogger("bakkesmod_rag.cost_tracker").info(
                "RedisCostTracker connected: %s", redis_url
            )
        except Exception as e:
            import logging as _logging
            _logging.getLogger("bakkesmod_rag.cost_tracker").warning(
                "RedisCostTracker falling back to in-memory: %s", e
            )

    def _record_cost(self, cost: float, provider: str, operation: str, model: str):
        """Record cost atomically in Redis, falling back to in-memory."""
        if self._redis is not None:
            try:
                today = datetime.now().strftime("%Y-%m-%d")
                # Store costs as integer microdollars for atomic INCR
                microdollars = int(cost * 1_000_000)
                self._redis.incrby("rag:cost:total", microdollars)
                self._redis.incrby(f"rag:cost:date:{today}", microdollars)
                self._redis.incrby(f"rag:cost:provider:{provider}", microdollars)
                self._redis.incrby(
                    f"rag:cost:op:{provider}:{operation}:{model}", microdollars
                )
                return
            except Exception as e:
                import logging as _logging
                _logging.getLogger("bakkesmod_rag.cost_tracker").warning(
                    "Redis cost record failed, using in-memory: %s", e
                )

        # Fall back to in-memory
        super()._record_cost(cost, provider, operation, model)

    def get_daily_cost(self, date: Optional[str] = None) -> float:
        """Get daily cost from Redis (converted from microdollars)."""
        if self._redis is None:
            return super().get_daily_cost(date)
        date = date or datetime.now().strftime("%Y-%m-%d")
        try:
            raw = self._redis.get(f"rag:cost:date:{date}")
            return int(raw or 0) / 1_000_000
        except Exception:
            return super().get_daily_cost(date)


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_tracker_instance: Optional[CostTracker] = None


def get_tracker(config=None) -> CostTracker:
    """Return the global ``CostTracker`` singleton.

    Creates one on first call. Passing *config* on the first call
    configures the tracker; subsequent calls return the cached instance.

    Args:
        config: Optional ``RAGConfig`` instance.

    Returns:
        The shared ``CostTracker`` instance.
    """
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = CostTracker(config=config)
    return _tracker_instance
