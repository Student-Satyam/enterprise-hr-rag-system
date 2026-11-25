"""Structured JSON logging and thread-safe runtime metrics with USD & INR cost tracking."""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)

# Current USD to INR conversion rate (Nov 2026)
USD_TO_INR: float = 92.68

# Groq per-1M token pricing (USD): (input_rate_per_1m, output_rate_per_1m)
MODEL_PRICING_PER_1M: dict[str, tuple[float, float]] = {
    "openai/gpt-oss-20b": (0.075, 0.30),
    "openai/gpt-oss-120b": (0.15, 0.60),
    "llama-3.1-8b-instant": (0.05, 0.08),
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "semantic-cache": (0.0, 0.0),
    "idk-fallback": (0.0, 0.0),
    "guardrail-blocked": (0.0, 0.0),
    "error": (0.0, 0.0),
}


def estimate_cost_usd(model_used: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate USD inference cost from prompt and completion token counts."""
    in_rate, out_rate = MODEL_PRICING_PER_1M.get(model_used, (0.075, 0.30))
    cost = (prompt_tokens / 1_000_000) * in_rate + (completion_tokens / 1_000_000) * out_rate
    return round(cost, 8)


def usd_to_inr(cost_usd: float) -> float:
    """Convert USD cost into Indian Rupees (INR)."""
    return round(cost_usd * USD_TO_INR, 6)


class MetricsTracker:
    """Thread-safe metrics tracker for requests, cache hits, latency, tokens, cost (USD/INR), and fallbacks."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.total_requests: int = 0
        self.cache_hits: int = 0
        self.guardrail_blocks: int = 0
        self.errors: int = 0
        self.fallback_count: int = 0
        self.total_latency_ms: float = 0.0
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_cost_usd: float = 0.0
        self.total_cost_inr: float = 0.0
        self.model_usage: dict[str, int] = defaultdict(int)

    def record(
        self,
        latency_ms: float,
        model_used: str,
        cache_hit: bool = False,
        blocked: bool = False,
        error: bool = False,
        fallback_used: bool = False,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        """Record metrics for one request."""
        cost_inr = usd_to_inr(cost_usd)
        with self._lock:
            self.total_requests += 1
            self.total_latency_ms += latency_ms
            if cache_hit:
                self.cache_hits += 1
            if blocked:
                self.guardrail_blocks += 1
            if error:
                self.errors += 1
            if fallback_used:
                self.fallback_count += 1

            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_cost_usd = round(self.total_cost_usd + cost_usd, 8)
            self.total_cost_inr = round(self.total_cost_inr + cost_inr, 6)
            self.model_usage[model_used] += 1

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-ready snapshot of live system metrics."""
        with self._lock:
            avg_latency = (
                round(self.total_latency_ms / self.total_requests, 2)
                if self.total_requests > 0
                else 0.0
            )
            cache_hit_rate = (
                round((self.cache_hits / self.total_requests) * 100, 1)
                if self.total_requests > 0
                else 0.0
            )
            return {
                "total_requests": self.total_requests,
                "cache_hits": self.cache_hits,
                "cache_hit_rate_pct": cache_hit_rate,
                "guardrail_blocks": self.guardrail_blocks,
                "errors": self.errors,
                "fallback_count": self.fallback_count,
                "avg_latency_ms": avg_latency,
                "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
                "total_cost_usd": self.total_cost_usd,
                "total_cost_inr": self.total_cost_inr,
                "model_usage": dict(self.model_usage),
            }