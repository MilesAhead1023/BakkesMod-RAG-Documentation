"""
Cost Tracking and Budget Management
====================================
Real-time cost tracking for RAG operations with budget alerts.
"""

import time
import json
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime, timedelta
from threading import Lock
import tiktoken
from config import get_config


class CostTracker:
    """Tracks API costs in real-time with budget management."""
    
    def __init__(self, persistence_path: Optional[Path] = None):
        self.config = get_config().cost
        self.persistence_path = persistence_path or Path("./.cache/cost_tracker.json")
        self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.costs: Dict[str, Dict] = self._load_costs()
        self.lock = Lock()
        
        # Lazy-load tokenizers (only when needed)
        self._tokenizers = None
    
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
            except Exception as e:
                print(f"⚠️  Could not load tiktoken: {e}. Token counting will use approximation.")
                self._tokenizers = {}
        return self._tokenizers
    
    def _load_costs(self) -> Dict:
        """Load cost history from disk."""
        if self.persistence_path.exists():
            try:
                with open(self.persistence_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️  Could not load cost history: {e}")
        
        return {
            "total": 0.0,
            "by_date": {},
            "by_provider": {},
            "by_operation": {}
        }
    
    def _save_costs(self):
        """Persist costs to disk."""
        try:
            with open(self.persistence_path, 'w') as f:
                json.dump(self.costs, f, indent=2)
        except Exception as e:
            print(f"⚠️  Could not save cost history: {e}")
    
    def count_tokens(self, text: str, model: str = "gpt-4") -> int:
        """Count tokens in text for a given model."""
        if not self.tokenizers:
            # Fallback: approximate as 4 chars per token
            return len(text) // 4
        
        encoder = self.tokenizers.get(model, self.tokenizers.get("gpt-4"))
        if not encoder:
            return len(text) // 4
        
        return len(encoder.encode(text))
    
    def track_embedding(self, num_tokens: int, provider: str = "openai", model: str = "text-embedding-3-large"):
        """Track embedding API cost."""
        if not self.config.track_costs:
            return
        
        # Calculate cost
        cost_per_1m = self.config.openai_embedding_cost
        cost = (num_tokens / 1_000_000) * cost_per_1m
        
        self._record_cost(cost, provider, "embedding", model)
    
    def track_llm_call(
        self,
        input_tokens: int,
        output_tokens: int,
        provider: str,
        model: str
    ):
        """Track LLM API cost."""
        if not self.config.track_costs:
            return
        
        # Determine cost rates
        if provider == "openai":
            if "gpt-4o-mini" in model:
                input_cost = self.config.openai_gpt4o_mini_input
                output_cost = self.config.openai_gpt4o_mini_output
            else:
                # Default to gpt-4o-mini rates
                input_cost = self.config.openai_gpt4o_mini_input
                output_cost = self.config.openai_gpt4o_mini_output
        elif provider == "anthropic":
            input_cost = self.config.anthropic_claude_sonnet_input
            output_cost = self.config.anthropic_claude_sonnet_output
        elif provider == "gemini":
            input_cost = self.config.gemini_flash_input
            output_cost = self.config.gemini_flash_output
        else:
            print(f"⚠️  Unknown provider: {provider}")
            return
        
        # Calculate total cost
        cost = (input_tokens / 1_000_000) * input_cost + (output_tokens / 1_000_000) * output_cost
        
        self._record_cost(cost, provider, "llm", model)
    
    def _record_cost(self, cost: float, provider: str, operation: str, model: str):
        """Record a cost entry."""
        with self.lock:
            # Update totals
            self.costs["total"] += cost
            
            # Update by date
            today = datetime.now().strftime("%Y-%m-%d")
            if today not in self.costs["by_date"]:
                self.costs["by_date"][today] = 0.0
            self.costs["by_date"][today] += cost
            
            # Update by provider
            if provider not in self.costs["by_provider"]:
                self.costs["by_provider"][provider] = 0.0
            self.costs["by_provider"][provider] += cost
            
            # Update by operation
            op_key = f"{provider}:{operation}:{model}"
            if op_key not in self.costs["by_operation"]:
                self.costs["by_operation"][op_key] = 0.0
            self.costs["by_operation"][op_key] += cost
            
            # Check budget
            self._check_budget()
            
            # Save to disk
            self._save_costs()
    
    def _check_budget(self):
        """Check if daily budget is exceeded."""
        if not self.config.daily_budget_usd:
            return
        
        today = datetime.now().strftime("%Y-%m-%d")
        daily_cost = self.costs["by_date"].get(today, 0.0)
        
        threshold = self.config.daily_budget_usd * (self.config.alert_threshold_pct / 100)
        
        if daily_cost >= self.config.daily_budget_usd:
            print(f"🚨 BUDGET EXCEEDED: ${daily_cost:.4f} / ${self.config.daily_budget_usd:.2f}")
            raise RuntimeError("Daily budget exceeded")
        elif daily_cost >= threshold:
            print(f"⚠️  Budget Alert: ${daily_cost:.4f} / ${self.config.daily_budget_usd:.2f} ({(daily_cost/self.config.daily_budget_usd)*100:.1f}%)")
    
    def get_daily_cost(self, date: Optional[str] = None) -> float:
        """Get cost for a specific date (or today)."""
        date = date or datetime.now().strftime("%Y-%m-%d")
        return self.costs["by_date"].get(date, 0.0)
    
    def get_report(self, days: int = 7) -> str:
        """Generate a cost report for the last N days."""
        lines = ["=" * 60]
        lines.append("Cost Report")
        lines.append("=" * 60)
        lines.append(f"Total All-Time: ${self.costs['total']:.4f}")
        lines.append("")
        
        # Last N days
        lines.append(f"Last {days} Days:")
        for i in range(days):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            cost = self.costs["by_date"].get(date, 0.0)
            lines.append(f"  {date}: ${cost:.4f}")
        lines.append("")
        
        # By provider
        lines.append("By Provider:")
        for provider, cost in sorted(self.costs["by_provider"].items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  {provider}: ${cost:.4f}")
        lines.append("")
        
        # Top operations
        lines.append("Top 5 Operations:")
        sorted_ops = sorted(self.costs["by_operation"].items(), key=lambda x: x[1], reverse=True)[:5]
        for op, cost in sorted_ops:
            lines.append(f"  {op}: ${cost:.4f}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


# Global singleton
_tracker: Optional[CostTracker] = None


def get_tracker() -> CostTracker:
    """Get the global cost tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = CostTracker()
    return _tracker


if __name__ == "__main__":
    # Test cost tracking
    tracker = get_tracker()
    
    # Simulate some operations
    tracker.track_embedding(1000, "openai")
    tracker.track_llm_call(500, 200, "anthropic", "claude-3-5-sonnet-20240620")
    tracker.track_llm_call(300, 100, "openai", "gpt-4o-mini")
    
    print(tracker.get_report(days=7))
