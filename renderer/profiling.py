from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from typing import Self, cast

logger = logging.getLogger("milkdrop_renderer")


class FrameProfiler:
    """Singleton frame / stage profiler for the MilkDrop render path.

    Enable with ``MILKDROP_PROFILE=1``.  Print a summary every
    ``MILKDROP_PROFILE_INTERVAL`` frames (default 300).

    Typical usage inside the render loop::

        with profile_stage("warp"):
            ...
        with profile_stage("comp"):
            ...
        profile_end_frame()
    """

    _instance: FrameProfiler | None = None
    _creation_lock = threading.Lock()

    def __new__(cls) -> Self:
        with cls._creation_lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._init()
                cls._instance = inst
            assert cls._instance is not None
            return cast(Self, cls._instance)

    def _init(self) -> None:
        self._data_lock = threading.Lock()
        self._timings: dict[str, deque[float]] = {}
        self._counters: dict[str, deque[float]] = {}
        self._memory_counters: dict[str, deque[float]] = {}
        self._frame_count = 0
        self._last_print = 0
        self._frame_timestamps: deque[float] = deque(maxlen=300)
        self._last_frame_time: float | None = None
        self._enabled = os.getenv("MILKDROP_PROFILE", "0") == "1"
        self._print_interval = int(os.getenv("MILKDROP_PROFILE_INTERVAL", "300"))
        self._max_samples = 1000  # keep last N frames of data

    # ------------------------------------------------------------------
    # Public control
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = bool(value)

    def reset(self) -> None:
        with self._data_lock:
            self._timings.clear()
            self._counters.clear()
            self._memory_counters.clear()
            self._frame_count = 0
            self._last_print = 0
            self._frame_timestamps.clear()
            self._last_frame_time: float | None = None

    def end_frame(self) -> None:
        """Call once at the end of every rendered frame.

        Advances the frame counter and triggers a report when the interval
        is reached.  Safe to call when profiling is disabled.
        """
        if not self._enabled:
            return
        with self._data_lock:
            self._frame_count += 1
            if self._frame_count - self._last_print >= self._print_interval:
                self._last_print = self._frame_count
                self._print_report_unlocked()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, name: str, delta_ms: float) -> None:
        if not self._enabled:
            return
        with self._data_lock:
            if name not in self._timings:
                self._timings[name] = deque(maxlen=self._max_samples)
            self._timings[name].append(delta_ms)

    def record_value(self, name: str, value: float) -> None:
        """Record a scalar metric (bytes uploaded, FBO count, etc.)."""
        if not self._enabled:
            return
        with self._data_lock:
            if name not in self._counters:
                self._counters[name] = deque(maxlen=self._max_samples)
            self._counters[name].append(float(value))

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @staticmethod
    def _percentile(sorted_samples: list[float], pct: float) -> float:
        if not sorted_samples:
            return 0.0
        n = len(sorted_samples)
        if n == 1:
            return sorted_samples[0]
        k = (n - 1) * pct
        f = int(k)
        c = min(f + 1, n - 1)
        if f == c:
            return sorted_samples[f]
        return sorted_samples[f] + (sorted_samples[c] - sorted_samples[f]) * (k - f)

    def get_stats(self) -> dict[str, dict[str, float]]:
        """Return min / median / p95 / avg / max / samples for every series."""
        with self._data_lock:
            stats: dict[str, dict[str, float]] = {}
            for name, samples in self._timings.items():
                if not samples:
                    continue
                sorted_s = sorted(samples)
                n = len(sorted_s)
                stats[name] = {
                    "min": sorted_s[0],
                    "median": self._percentile(sorted_s, 0.5),
                    "p95": self._percentile(sorted_s, 0.95),
                    "avg": sum(samples) / n,
                    "max": sorted_s[-1],
                    "samples": float(n),
                }
            for name, values in self._counters.items():
                if not values:
                    continue
                sorted_v = sorted(values)
                n = len(sorted_v)
                stats[name] = {
                    "min": sorted_v[0],
                    "median": self._percentile(sorted_v, 0.5),
                    "p95": self._percentile(sorted_v, 0.95),
                    "avg": sum(values) / n,
                    "max": sorted_v[-1],
                    "samples": float(n),
                }
            return stats

    def _print_report(self) -> None:
        """Print the profile report (acquires the data lock)."""
        with self._data_lock:
            self._print_report_unlocked()

    def _print_report_unlocked(self) -> None:
        lines = ["\n=== MilkDrop Performance Profile ==="]
        total_avg = 0.0
        for name, samples in sorted(self._timings.items()):
            if not samples:
                continue
            sorted_s = sorted(samples)
            n = len(sorted_s)
            avg = sum(samples) / n
            median = self._percentile(sorted_s, 0.5)
            p95 = self._percentile(sorted_s, 0.95)
            mx = sorted_s[-1]
            lines.append(
                f"  {name:<30} avg={avg:>7.2f}ms  median={median:>7.2f}ms  "
                f"p95={p95:>7.2f}ms  max={mx:>7.2f}ms  n={n}"
            )
            total_avg += avg
        for name, values in sorted(self._counters.items()):
            if not values:
                continue
            sorted_v = sorted(values)
            n = len(sorted_v)
            avg = sum(values) / n
            mx = sorted_v[-1]
            lines.append(
                f"  {name:<30} avg={avg:>10.1f}  max={mx:>10.1f}  n={n}"
            )
        lines.append(f"  {'TOTAL (sum of stage avgs)':<30} {total_avg:>7.2f}ms")
        lines.append(f"  frames since last report: {self._frame_count - self._last_print + self._print_interval}")
        lines.append("=" * 50)
        logger.info("\n".join(lines))


# ---------------------------------------------------------------------------
# Module-level helpers (zero-cost when disabled)
# ---------------------------------------------------------------------------

_FRAME_PROFILER: FrameProfiler | None = None


def _profiler() -> FrameProfiler:
    global _FRAME_PROFILER
    if _FRAME_PROFILER is None:
        _FRAME_PROFILER = FrameProfiler()
    return _FRAME_PROFILER


class _NullProfileStage:
    __slots__ = ()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> bool:
        return False


_NULL = _NullProfileStage()


class _ProfileStageContext:
    __slots__ = ("_name", "_t0")

    def __init__(self, name: str) -> None:
        self._name = name
        self._t0 = 0.0

    def __enter__(self) -> Self:
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> bool:
        dt_ms = (time.perf_counter() - self._t0) * 1000.0
        _profiler().record(self._name, dt_ms)
        return False


def profile_stage(name: str) -> _ProfileStageContext | _NullProfileStage:
    """Context manager for timing a render stage.

    When profiling is disabled a shared no-op instance is returned so the
    hot path pays neither object construction nor timer calls.
    """
    if not _profiler().enabled:
        return _NULL
    return _ProfileStageContext(name)


def profile_end_frame() -> None:
    """Mark the end of a frame (advances the frame counter / may print)."""
    _profiler().end_frame()


def profile_record_value(name: str, value: float) -> None:
    """Convenience for non-timing metrics."""
    _profiler().record_value(name, value)


def profile_get_stats() -> dict[str, dict[str, float]]:
    return _profiler().get_stats()


def profile_reset() -> None:
    _profiler().reset()
