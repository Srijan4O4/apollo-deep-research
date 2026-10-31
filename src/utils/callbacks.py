"""LLM callback handler for token-usage tracking and budget enforcement.

Provides :class:`TokenTrackingCallback` — a LangChain-compatible callback
handler that accumulates token counts across every LLM invocation and can
raise :class:`TokenBudgetExceeded` when a configurable ceiling is breached.
"""

from __future__ import annotations

import threading
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from src.utils.logging import get_logger

logger = get_logger(__name__)


# ── Exceptions ─────────────────────────────────────────────────────────────


class TokenBudgetExceeded(Exception):
    """Raised when cumulative token usage exceeds the allowed budget."""

    def __init__(self, used: int, budget: int) -> None:
        self.used = used
        self.budget = budget
        super().__init__(
            f"Token budget exceeded: {used:,} tokens used against a "
            f"budget of {budget:,}.",
        )


# ── Pricing Table ──────────────────────────────────────────────────────────

# Costs are expressed in USD per 1 000 tokens (prompt / completion).
_MODEL_PRICING: dict[str, dict[str, float]] = {
    "gemini-2.5-flash": {
        "prompt": 0.000_15,
        "completion": 0.000_60,
    },
    "gpt-4o": {
        "prompt": 0.002_50,
        "completion": 0.010_00,
    },
}


# ── Callback Handler ──────────────────────────────────────────────────────


class TokenTrackingCallback(BaseCallbackHandler):
    """Accumulates prompt, completion and total token counts.

    Thread-safe via an internal :class:`threading.Lock` so the handler can
    be shared across concurrent chains.
    """

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0

    # ── LangChain hook ─────────────────────────────────────────────────

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Aggregate token usage reported by the LLM provider."""
        if response.llm_output is None:
            return

        usage: dict[str, int] | None = response.llm_output.get("token_usage")
        if usage is None:
            # Some providers nest under "usage" instead.
            usage = response.llm_output.get("usage")

        if not usage:
            return

        prompt = usage.get("prompt_tokens", 0)
        completion = usage.get("completion_tokens", 0)
        total = usage.get("total_tokens", prompt + completion)

        with self._lock:
            self.prompt_tokens += prompt
            self.completion_tokens += completion
            self.total_tokens += total

        logger.debug(
            "llm_tokens_recorded",
            prompt_tokens=prompt,
            completion_tokens=completion,
            cumulative_total=self.total_tokens,
        )

    # ── Public helpers ─────────────────────────────────────────────────

    def get_total_tokens(self) -> int:
        """Return the cumulative number of tokens consumed so far."""
        with self._lock:
            return self.total_tokens

    def get_cost_estimate(self, model: str) -> float:
        """Estimate the cumulative USD cost for a given *model*.

        Parameters
        ----------
        model:
            Model identifier (e.g. ``"gemini-2.5-flash"`` or ``"gpt-4o"``).

        Returns
        -------
        float
            Estimated cost in USD.  Returns ``0.0`` if the model is not in
            the internal pricing table.
        """
        pricing = _MODEL_PRICING.get(model)
        if pricing is None:
            logger.warning("unknown_model_pricing", model=model)
            return 0.0

        with self._lock:
            prompt_cost = (self.prompt_tokens / 1_000) * pricing["prompt"]
            completion_cost = (self.completion_tokens / 1_000) * pricing["completion"]

        return round(prompt_cost + completion_cost, 6)

    def check_budget(self, max_tokens: int) -> bool:
        """Assert that cumulative usage is within *max_tokens*.

        Parameters
        ----------
        max_tokens:
            The hard token ceiling.

        Returns
        -------
        bool
            ``True`` when usage is within budget.

        Raises
        ------
        TokenBudgetExceeded
            If the cumulative token count exceeds *max_tokens*.
        """
        used = self.get_total_tokens()
        if used > max_tokens:
            raise TokenBudgetExceeded(used=used, budget=max_tokens)
        return True

    def reset(self) -> None:
        """Zero-out all counters (useful between test runs)."""
        with self._lock:
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.total_tokens = 0
