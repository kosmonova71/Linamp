from __future__ import annotations

import logging
from typing import ClassVar

from .config import MAX_FRAME_BUDGET

logger = logging.getLogger("milkdrop_renderer")

# MAX_PER_POINT_BUDGET is the separate per-wave per-point allowance; see
# FrameBudget docstring. Both budgets are documented here so they stay
# discoverable together.


# --- Expression-level budget ------------------------------------
class LoopBudget:
    """Total iteration allowance shared by every loop in one expression.

    ``LoopExpr`` clamps each loop to ``max_iterations`` independently, so
    nesting multiplies the real cost. One budget object is created per
    top-level expression evaluation and threaded down through the loop body;
    the first loop that overspends marks it exhausted and all deeper/sibling
    loops become no-ops.

    Single-threaded by design; counters are plain ints.
    """

    __slots__ = ("_remaining", "exhausted", "total")

    def __init__(self, total: int) -> None:
        if total < 0:
            raise ValueError("LoopBudget total must be non-negative")
        if total < 0:
            raise ValueError("LoopBudget total must be non-negative")
        self.total: int = max(0, int(total))
        self._remaining: int = self.total
        self.exhausted: bool = self._remaining == 0

    def take(self, count: int) -> int:
        """Reserve up to *count* iterations, returning how many are allowed.

        Partial grants are returned when the remaining budget is smaller
        than the request. Once exhausted, subsequent calls return 0.
        Negative *count* is rejected (raises ValueError).
        """
        if count < 0:
            raise ValueError("LoopBudget.take count must be non-negative")
        if self.exhausted or count == 0:
            return 0
        allowed = min(count, self._remaining)
        self._remaining -= allowed
        if self._remaining == 0:
            self.exhausted = True
        return allowed

    @property
    def remaining(self) -> int:
        return self._remaining

    @property
    def usage(self) -> tuple[int, int, float]:
        """(consumed, total, fraction)."""
        consumed = self.total - self._remaining
        fraction = (consumed / self.total) if self.total else 1.0
        return consumed, self.total, fraction

    def __bool__(self) -> bool:
        return not self.exhausted

    def __repr__(self) -> str:
        return (
            f"LoopBudget(remaining={self._remaining}, "
            f"exhausted={self.exhausted})"
        )


# --- Frame-level budget ----------------------------------------
class FrameBudget:
    """Aggregate per-frame evaluation budget shared across all VM expressions.

    Unlike ``LoopBudget`` (created per top-level expression and only tracks
    loop iterations inside that expression), ``FrameBudget`` spans the
    *entire* frame: ``per_frame_init``, ``per_frame``, every wave's
    ``per_frame`` and every shape's ``per_frame``.

    The wave *per-point* phase is budgeted separately (see
    :data:`MAX_PER_POINT_BUDGET`). Charging per-point work against this
    budget previously exhausted the shared allowance on ordinary presets,
    truncating wave geometry and skipping the shape phase entirely.

    This budget prevents a preset that contains many top-level expressions
    or many shape layers from consuming unbounded CPU even when every
    individual expression stays within its own limits.

    Work is accounted in abstract "evaluation units" (one unit per
    expression evaluation by default, or a caller-specified amount for
    heavier work). When the budget is exhausted, remaining expressions
    become no-ops and a single warning is emitted so the problem is visible
    rather than silently clamped.

    Single-threaded by design; counters are plain ints.
    """

    __slots__ = (
        "_label",
        "_remaining",
        "_warn",
        "_warned",
        "consumed",
        "exhausted",
        "total",
    )

    DEFAULT_FRAME_BUDGET: ClassVar[int] = MAX_FRAME_BUDGET

    def __init__(self, total: int = MAX_FRAME_BUDGET, *, warn: bool = True, label: str = "Frame evaluation") -> None:
        if total < 0:
            raise ValueError("FrameBudget total must be non-negative")
        self.total: int = max(0, int(total))
        self._remaining: int = self.total
        self.consumed: int = 0
        self.exhausted: bool = False
        self._warned: bool = False
        self._warn: bool = bool(warn)
        self._label: str = label

    def consume(self, amount: int = 1) -> bool:
        """Account for *amount* units of work.

        Returns ``True`` only when the full amount fits and has been
        deducted. If the request exceeds the remaining budget the remaining
        budget is spent, the budget is marked exhausted, and ``False`` is
        returned (the caller must treat the work as not permitted).

        A warning is logged the first time the budget becomes exhausted
        in a frame. Negative *amount* is treated as a no-op and returns ``False``.
        """
        if amount < 0:
            return False
        if self.exhausted or amount == 0:
            return False

        if amount > self._remaining:
            self.consumed += self._remaining
            self._remaining = 0
            self.exhausted = True
            self._emit_exhaustion_warning()
            return False

        self._remaining -= amount
        self.consumed += amount
        return True

    def reset(self, total: int | None = None) -> None:
        """Reset the budget for a new frame.

        If *total* is supplied it becomes the new frame limit; otherwise
        the previous total is reused. Invariant: remaining + consumed == total.
        """
        if total is not None:
            if total < 0:
                raise ValueError("FrameBudget total must be non-negative")
            self.total = max(0, int(total))
        self._remaining = self.total
        self.consumed = 0
        self.exhausted = False
        self._warned = False

    @property
    def remaining(self) -> int:
        return self._remaining

    @property
    def used(self) -> int:
        """How many units have been consumed so far this frame."""
        return self.consumed

    @property
    def fraction_used(self) -> float:
        """Fraction of the total budget already spent (0.0 – 1.0)."""
        if self.total == 0:
            return 1.0
        return self.consumed / self.total

    @property
    def usage(self) -> tuple[int, int, float]:
        """(consumed, total, fraction)."""
        return self.consumed, self.total, self.fraction_used

    def __bool__(self) -> bool:
        return not self.exhausted

    def _emit_exhaustion_warning(self) -> None:
        if self._warned or not self._warn:
            return
        logger.warning(
            "%s budget exhausted (total=%d, consumed=%d). "
            "Remaining expressions will be no-ops.",
            self._label, self.total, self.consumed,
        )
        self._warned = True

    def __repr__(self) -> str:
        return (
            f"FrameBudget(total={self.total}, remaining={self._remaining}, "
            f"consumed={self.consumed}, exhausted={self.exhausted}, "
            f"fraction_used={self.fraction_used:.3f})"
        )
