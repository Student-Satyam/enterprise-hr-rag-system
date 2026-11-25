"""Complexity-based query router selecting between fast and high-reasoning Groq models."""

import logging
import re

from src.config import get_settings

logger = logging.getLogger(__name__)

# Indicators that a query requires cross-document comparison, synthesis, or multi-step reasoning
COMPLEX_QUERY_KEYWORDS = {
    "compare", "comparison", "difference", "differences", "versus", "vs",
    "contrast", "across", "all handbooks", "all policies", "both",
    "summarize", "summary", "explain in detail", "step by step",
    "eligibility criteria", "exceptions", "calculate", "how does",
    "why does", "pros and cons", "between",
}


def classify_complexity(query: str) -> str:
    """Classify query as 'simple' or 'complex' using fast deterministic heuristics."""
    lowered = query.lower().strip()
    word_count = len(re.findall(r"\w+", lowered))

    # Multiple question marks or long multi-clause questions need deeper reasoning
    if lowered.count("?") > 1 or word_count >= 18:
        return "complex"

    # Explicit comparison or synthesis keywords trigger the stronger model
    if any(kw in lowered for kw in COMPLEX_QUERY_KEYWORDS):
        return "complex"

    return "simple"


def route_query(query: str) -> str:
    """Return the optimal Groq model ID based on query complexity."""
    settings = get_settings()
    complexity = classify_complexity(query)

    if complexity == "complex":
        chosen_model = settings.complex_llm_model
    else:
        chosen_model = settings.fast_llm_model

    logger.info("Router classified query as '%s' -> model='%s'", complexity, chosen_model)
    return chosen_model