from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import threading
from collections import OrderedDict
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any, ClassVar

import numpy as np

from resources import (
    VAR_REGISTRY,
    VARIABLES,
)

from .budgets import (
    FrameBudget,
)
from .config import (
    _VM_EVAL_TIMEOUT,
    ASSIGN_RE,
    LOOP_START_RE,
    MAX_CONTINUATION_LENGTH,
    MAX_FRAME_BUDGET,
    MAX_FRAME_BUDGET_INIT,
    MAX_INIT_LOOP_ITERATIONS,
    MAX_PER_POINT_BUDGET,
    PER_POINT_BUDGET_INSTRUCTION_FACTOR,
    PER_POINT_BUDGET_SLACK,
    PROJECT_ROOT,
    SHAPE_RE,
    WAVE_RE,
    PresetBudgetExhausted,
    PresetCompileError,
    _compound_assign_re,
    _monotonic,
    _split_milk_eqs,
    _strip_inline_comments,
)
from .gl import (
    GPUUniforms,
)
from .shaders import (
    CompiledExpr,
    LoopExpr,
    ShapeProgram,
    WaveProgram,
    WaveSamples,
    _append_compiled_expression,
    _compile_expression_list,
    _compile_shape,
    _compile_wave,
)
from .state import (
    PresetState,
)

logger = logging.getLogger("milkdrop_renderer")


# --- Audio buffer -------------------------------------------------
class LatestAudioBuffer:
    """Latest-value audio buffer for lock-free single-writer/multi-reader.

    This is a simple SPSC (single-producer, single-consumer) buffer that
    stores only the latest published value.  The consumer reads the latest
    snapshot atomically under a lock.  Unlike a ring buffer, it does not
    preserve history — it only guarantees the reader sees the most recent
    complete write.

    The previous name ``LockFreeAudioBuffer`` was misleading because the
    implementation uses a ``threading.Lock`` internally.  ``LatestAudioBuffer``
    more accurately describes the semantics: readers always get the latest
    published frame, and slow readers simply miss intermediate frames.
    """

    __slots__ = ("_latest", "_lock")

    def __init__(self):
        self._latest: dict | None = None
        self._lock = threading.Lock()

    def publish(self, data: dict):
        with self._lock:
            self._latest = data

    def consume(self) -> dict | None:
        """Return a defensive copy of the latest data.

        Returns a shallow copy so the caller cannot mutate the stored
        snapshot.  Returns ``None`` if no data has been published yet.
        """
        with self._lock:
            if self._latest is None:
                return None
            # Defensive shallow copy to prevent external mutation
            return dict(self._latest)


LockFreeAudioBuffer = LatestAudioBuffer


# --- Preset health scoring ----------------------------------------
def apply_preset_parse_health(health, preset):
    """Apply parse-error penalties to a PresetHealthScore from a MilkPreset."""
    if health is None or preset is None:
        return
    for err in getattr(preset, "parse_errors", []):
        cat = err.get("category", "unsupported_syntax")
        if cat in ("recoverable_formatting",):
            health.add_warning(f"parse: {cat} (line {err.get('line', '?')})")
        else:
            health.add_error(f"parse: {cat} (line {err.get('line', '?')})")


class PresetHealthScore:
    __slots__ = ("errors", "score", "unsupported_funcs", "warnings")

    def __init__(self):
        self.score = 100.0
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.unsupported_funcs: list[str] = []

    def add_warning(self, msg: str):
        self.warnings.append(msg)
        self.score = max(0.0, self.score - 5.0)

    def add_error(self, msg: str):
        self.errors.append(msg)
        self.score = max(0.0, self.score - 20.0)

    def is_healthy(self) -> bool:
        return self.score >= 50.0 and not self.errors


# --- On-disk preset cache -------------------------------------------
class PresetCache:
    CACHE_DIR: Path = PROJECT_ROOT / ".preset_cache"
    CACHE_VERSION = 3

    def __init__(self):
        self.CACHE_DIR.mkdir(exist_ok=True)
        self._cache_index: dict[str, Path] = {}
        self._load_index()

    def _load_index(self):
        idx_file = self.CACHE_DIR / "index.json"
        try:
            if idx_file.exists():
                data = json.loads(idx_file.read_text())
                self._cache_index = {
                    k: Path(v) for k, v in data.get("entries", {}).items()
                }
        except (OSError, json.JSONDecodeError):
            self._cache_index = {}

    def _save_index(self):
        idx_file = self.CACHE_DIR / "index.json"
        try:
            data = {
                "version": self.CACHE_VERSION,
                "entries": {k: str(v) for k, v in self._cache_index.items()},
            }
            idx_file.write_text(json.dumps(data, indent=2))
        except (OSError, TypeError, ValueError) as e:
            logger.debug(f"Failed to save cache index: {e}")

    def _cache_path(self, source_path: str) -> Path:
        h = hashlib.sha256(str(source_path).encode()).hexdigest()[:16]
        return self.CACHE_DIR / f"{h}.mdc"

    def get(self, source_path: str, mtime: float) -> dict[str, Any] | None:
        cp = self._cache_path(source_path)
        try:
            if not cp.exists():
                return None
            if cp.stat().st_mtime < mtime:
                return None
            with open(cp) as f:
                data = json.load(f)
            if data.get("version") != self.CACHE_VERSION:
                return None
            if data.get("source_mtime", 0.0) != mtime:
                return None
            return data  # type: ignore
        except (OSError, json.JSONDecodeError, ValueError):
            return None

    def put(self, source_path: str, mtime: float, compiled_data: dict):
        cp = self._cache_path(source_path)
        try:
            compiled_data["version"] = self.CACHE_VERSION
            compiled_data["source_mtime"] = mtime
            with open(cp, "w") as f:
                json.dump(compiled_data, f)
            try:
                os.utime(cp, (mtime, mtime))
            except (OSError, ValueError) as e:
                logger.debug(f"Failed to set cache file mtime: {e}")
            self._cache_index[str(source_path)] = cp
            self._save_index()
        except (OSError, TypeError, ValueError) as e:
            logger.debug(f"Failed to save compiled preset to cache: {e}")

    def invalidate(self, source_path: str):
        cp = self._cache_path(source_path)
        try:
            cp.unlink(missing_ok=True)
            self._cache_index.pop(str(source_path), None)
            self._save_index()
        except OSError as exc:
            logger.debug("Exception in try block: %s", exc)


PRESET_CACHE = PresetCache()


# --- Preset container ----------------------------------------------
class MilkPreset:
    """Complete MilkDrop preset container with all equation types."""

    def __init__(self):
        self.params = {}
        self.per_frame_init = []
        self.per_frame = []
        self.per_pixel = []
        self.per_vertex = []
        self.waves = {}
        self.shapes = {}
        self.lines = []
        self.genome = None
        self.source_file: str | None = None
        self.parse_errors: list[dict] = []

    def get_stats(self):
        """Get preset statistics for debugging."""
        return {
            "params": len(self.params),
            "per_frame": len(self.per_frame),
            "per_pixel": len(self.per_pixel),
            "per_vertex": len(self.per_vertex),
            "waves": len(self.waves),
            "shapes": len(self.shapes),
        }

    def get(self, key, default=None):
        """Dictionary-style access to preset attributes."""
        return getattr(self, key, default)

    def copy(self) -> MilkPreset:
        """Create a deep copy of this preset."""
        new_preset = MilkPreset()
        new_preset.params = self.params.copy()
        new_preset.per_frame_init = self.per_frame_init.copy()
        new_preset.per_frame = self.per_frame.copy()
        new_preset.per_pixel = self.per_pixel.copy()
        new_preset.per_vertex = self.per_vertex.copy()
        new_preset.waves = {k: v.copy() for k, v in self.waves.items()}
        new_preset.shapes = {k: v.copy() for k, v in self.shapes.items()}
        new_preset.lines = self.lines[:]
        new_preset.genome = self.genome
        new_preset.source_file = self.source_file
        new_preset.parse_errors = self.parse_errors[:]
        return new_preset

    def to_dict(self):
        """Convert to plain dict for PresetProgram and dict-style consumers."""
        return {
            "params": self.params,
            "per_frame_init": self.per_frame_init,
            "per_frame": self.per_frame,
            "per_pixel": self.per_pixel,
            "per_vertex": self.per_vertex,
            "waves": self.waves,
            "shapes": self.shapes,
            "lines": self.lines,
        }


# --- Equation helpers ----------------------------------------------
def _eq_bucket_of(key: str) -> str:
    """Return the equation bucket a preset key belongs to.

    Wrapped equations are written across several incrementing keys
    (``per_frame_1``, ``per_frame_2``, ...), so the trailing index is dropped to
    decide whether two keys address the same bucket.  Keys that are not indexed
    equation buckets (``warp_1``, ``comp_1``, ``fDecay``) keep their full name so
    they never compare equal to an equation bucket.
    """
    return re.sub(r"_\d+$", "", key.strip())


def _split_loop_body(body_str: str) -> list:
    """Split a loop body on ';' stripping inline comments per item.

    Comment stripping happens at the parsing stage so a commented body
    item (``loop(3, a=1; //note; b=2;)``) never reaches the compiler
    as a bare ``//...`` line.
    """
    result = []
    for part in str(body_str).split(";"):
        part = _strip_inline_comments(part).strip()
        if part:
            result.append(part)
    if not result:
        logger.debug(
            "Loop body is empty after comment stripping (body_str=%r)", body_str
        )
    return result


# --- Continuation / incompleteness helpers (refactored) -----------
def _paren_depth(s: str) -> int:
    """Count unbalanced parentheses.
    
    Returns the net balance: +1 for '(', -1 for ')'. This is used to track
    whether expressions/loops have unclosed parentheses.
    """
    if not isinstance(s, str):
        return 0
    depth = 0
    for c in s:
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
    return depth


def _strip_trailing_semis_commas(s: str) -> str:
    """Strip trailing semicolons and commas without touching parens."""
    if not s:
        return s
    t = s
    while t and t[-1] in ";,":
        t = t[:-1]
    return t


def _is_balanced_parens(val: str) -> bool:
    """Check if parentheses are balanced in *val*."""
    return val.count("(") == val.count(")")


def _expr_ends_with_operator(s: str) -> bool:
    """True if expression ends with a binary/operator char indicating continuation."""
    if not s:
        return False
    t = s.rstrip()
    if not t:
        return False
    return t[-1] in ("+", "-", "*", "/", ",", "^", "&", "|", "<", ">", "=", "!", "~", "(")


def _is_bare_closer_line(s: str) -> bool:
    """True if line consists only of ) ; , chars."""
    if not isinstance(s, str):
        return False
    t = s.strip()
    if not t:
        return False
    # Only closers
    return all(c in ");," for c in t)


def _is_incomplete_expr(val: str) -> bool:
    """Check if a MilkDrop expression value appears to be incomplete."""
    if not val or not val.strip():
        return True
    cleaned = _strip_inline_comments(val).rstrip()
    if not cleaned:
        return True
    if cleaned.strip() in (")", ");"):
        return False
    if not _is_balanced_parens(cleaned):
        return True
    stripped = cleaned.rstrip()
    if not stripped:
        return True
    # If balanced and ends with ) it's usually complete; old behavior
    # was overly aggressive - keep conservative for operators/open parens
    last_char = stripped[-1]
    if last_char == ")":
        return False  # balanced parens ending with ) is complete
    return _expr_ends_with_operator(stripped)


def _next_line_is_continuation(next_line: object, joined_val: str) -> bool:
    """Check if *next_line* is a bare expression fragment continuing *joined_val*.

    Some MilkDrop presets split a single assignment across consecutive
    ``per_frame_N`` lines when the RHS expression is too long to fit on one
    line.  Example from "Martin - QBikal - Surface Turbulence"::

        per_frame_46=      megabuf (oy*ylen+ox) = megabuf (oy*ylen+ox)
        per_frame_47=                           - 2*30/fps*sqrt(amp)*above(...)

    The first line's value ``"megabuf (oy*ylen+ox) = megabuf (oy*ylen+ox)"``
    ends with a balanced ``)`` and :func:`_is_incomplete_expr` therefore
    reports it as complete.  However the second line's value,
    ``"- 2*30/fps*sqrt(amp)*above(...)"``, has no ``=`` of its own and starts
    with a binary operator — it is a bare continuation fragment, not a
    standalone equation.

    This function detects that situation so the continuation-joining loops in
    :func:`parse_milk_file_robust`, :func:`_preprocess_loop_blocks` (Place 2,
    inside-loop-body) and :func:`_preprocess_loop_blocks` (Place 3,
    regular-equation) also fire when the current expression merely *looks*
    complete.

    The check is deliberately conservative: it only returns ``True`` when the
    fragment genuinely cannot stand alone as an equation (no top-level ``=``
    in the value), preventing false positives on legitimate single-line
    assignments such as ``per_frame_N=x = -x;``.
    """
    if not isinstance(next_line, str):
        return False
    if not next_line:
        return False
    stripped = next_line.strip()
    if not stripped or stripped.startswith(("//", "[")):
        return False
    if not ASSIGN_RE.match(stripped):
        # Non-key=value line (e.g. a bare ``- expr`` fragment that has
        # already been stripped of its ``per_frame_N=`` prefix by the
        # loop-body scanner).  If it has no '=' and starts with a binary
        # arithmetic operator, it is a continuation fragment, not a
        # standalone equation — the join must happen.
        if "=" not in stripped:
            return stripped[0] in ("+", "-", "*", "/", "%")
        return False
    # Extract the value after the first '=' (i.e. the part after key=).
    nval = stripped.split("=", 1)[1]
    nval = _strip_inline_comments(nval).strip()
    if not nval:
        return False
    # A value that itself contains '=' is a complete assignment on its own
    # (e.g. "xind = xind+1"); it is NOT a continuation fragment.
    if "=" in nval:
        return False
    # A value starting with a binary arithmetic operator is a fragment that
    # must be appended to the previous line's expression.
    return nval[0] in ("+", "-", "*", "/", "%")


def _should_join_continuation(joined: str, peek: object) -> bool:
    """Determine if *peek* should be joined to *joined* as a continuation."""
    if not isinstance(peek, str):
        return False
    ps = peek.strip()
    if not ps:
        return False
    # Inside unbalanced parens - be permissive for call continuations
    if _paren_depth(joined) > 0:
        if ps == ")" or ps.endswith((");", ")")):
            return True
        if ASSIGN_RE.match(ps) or _compound_assign_re.match(ps):
            return True
        if LOOP_START_RE.search(ps):
            return True
    # Bare binary operator fragment
    if "=" not in ps and ps and ps[0] in ("+", "-", "*", "/", "%"):
        return True
    if _next_line_is_continuation(peek, joined):
        return True
    return bool(_is_incomplete_expr(joined))


# --- Shared loop parsing helpers -----------------------------------
def _extract_loop_header_info(s: str):
    """Extract loop count string and comma index from a loop header string."""
    count_match = LOOP_START_RE.search(s)
    if count_match is None:
        return None, None
    after_paren = s.find("(", count_match.start())
    comma_idx = s.find(",", after_paren)
    if comma_idx is None or comma_idx <= after_paren:
        return "0", after_paren
    count_str = s[after_paren + 1 : comma_idx].strip()
    return count_str, comma_idx


def _is_self_closing_loop_header(s: str) -> bool:
    if not isinstance(s, str):
        return False
    return s.count("(") > 0 and s.count("(") == s.count(")")


def _strip_trailing_closing_parens_for_loop(s: str) -> str:
    """Strip trailing ')' only while parens remain unbalanced (for loop headers)."""
    if not isinstance(s, str):
        return s
    t = s
    while t and t[-1] == ")" and _paren_depth(t) > 0:
        t = t[:-1]
    return t


def _count_unmatched_closes(s: str) -> int:
    """Count how many trailing ')' close unmatched parens."""
    if not isinstance(s, str) or not s:
        return 0
    trailing = len(s) - len(s.rstrip(")"))
    if trailing <= 0:
        return 0
    pd = _paren_depth(s)
    if pd >= 0:
        return 0
    return min(trailing, -pd)


# --- Cache payload ----------------------------------------------
def _build_preset_cache_data(program: PresetProgram, base_params: dict) -> dict:
    # The ``body_signature`` is a stable hash of every equation body the
    # preset compiles (per_frame_init/per_frame/per_pixel/per_vertex/wave/
    # shape).  Caching a PresetProgram is only safe if the evolved preset's
    # equation bodies are byte-identical to the raw preset's — today this is
    # guaranteed because :class:`EvolutionEngine.evolve` only jitters scalar
    # params.  If evolution ever starts mutating equation bodies, this hash
    # will diverge and the cache hit path will be skipped automatically.
    body_parts = (
        [e.source for e in program.per_frame_init]
        + [e.source for e in program.per_frame]
        + [e.source for e in program.per_pixel]
        + [e.source for e in program.per_vertex]
    )
    for w in program.waves:
        body_parts.extend(e.source for e in w.per_frame)
        body_parts.extend(e.source for e in w.per_point)
    for s in program.shapes:
        body_parts.extend(e.source for e in s.per_frame)
        body_parts.extend(e.source for e in s.per_frame_init)
    body_signature = hashlib.sha256("\n".join(body_parts).encode("utf-8")).hexdigest()
    data = {
        "per_frame_init": [e.source for e in program.per_frame_init],
        "per_frame": [e.source for e in program.per_frame],
        "per_pixel": [e.source for e in program.per_pixel],
        "per_vertex": [e.source for e in program.per_vertex],
        "waves": [
            {
                "per_frame": [e.source for e in w.per_frame],
                "per_point": [e.source for e in w.per_point],
                "per_frame_init": [e.source for e in getattr(w, "per_frame_init", [])],
                "enabled": w.enabled,
                "mode": getattr(w, "mode", 0),
                "samples": getattr(w, "samples", 512),
                "bSpectrum": getattr(w, "bSpectrum", True),
                "bUseDots": getattr(w, "bUseDots", False),
                "bDrawThick": getattr(w, "bDrawThick", False),
                "additive": getattr(w, "additive", True),
                "r": getattr(w, "r", 1.0),
                "g": getattr(w, "g", 1.0),
                "b": getattr(w, "b", 1.0),
                "a": getattr(w, "a", 1.0),
                "x": getattr(w, "x", 0.5),
                "y": getattr(w, "y", 0.5),
                "num_instances": getattr(w, "num_instances", 1),
                "instance_separator": getattr(w, "instance_separator", 0.0),
            }
            for w in program.waves
        ],
        "shapes": [
            {
                "per_frame": [e.source for e in s.per_frame],
                "per_frame_init": [e.source for e in s.per_frame_init],
                "enabled": s.enabled,
                "sides": getattr(s, "sides", 3),
                "additive": getattr(s, "additive", False),
                "thickOutline": getattr(s, "thick_outline", False),
                "textured": getattr(s, "textured", False),
                "r": getattr(s, "r", 1.0),
                "g": getattr(s, "g", 1.0),
                "b": getattr(s, "b", 1.0),
                "a": getattr(s, "a", 1.0),
                "r2": getattr(s, "r2", 0.0),
                "g2": getattr(s, "g2", 0.0),
                "b2": getattr(s, "b2", 0.0),
                "a2": getattr(s, "a2", 0.0),
                "x": getattr(s, "x", 0.5),
                "y": getattr(s, "y", 0.5),
                "rad": getattr(s, "rad", 0.1),
                "ang": getattr(s, "ang", 0.0),
                "tex_ang": getattr(s, "tex_ang", 0.0),
                "tex_zoom": getattr(s, "tex_zoom", 1.0),
                "border_r": getattr(s, "border_r", 1.0),
                "border_g": getattr(s, "border_g", 1.0),
                "border_b": getattr(s, "border_b", 1.0),
                "border_a": getattr(s, "border_a", 1.0),
                "border_size": getattr(s, "border_size", 0.01),
                "num_instances": getattr(s, "num_instances", 1),
                "instance_separator": getattr(s, "instance_separator", 0.0),
            }
            for s in program.shapes
        ],
        "base_params": base_params,
        "body_signature": body_signature,
    }
    return data


# --- Asynchronous preset compilation -----------------------------
class AsyncCompileResult:
    """Result of an async preset compilation.

    Exactly one of ``program`` or ``error`` is set when the result is
    terminal (success or failure).  A result is only stored in the cache
    *after* compilation has completed, so a missing cache entry means
    "not yet compiled" or "evicted" — never a failure.

    ``preset_dict`` is the raw parsed .milk dictionary the worker produced
    while compiling. It is exposed so the caller can perform further CPU
    work (e.g. evolution) without re-parsing the file on the main thread.

    ``submission_epoch`` is the epoch at which this compilation was submitted.
    It allows the caller to reject stale results from a previous epoch.
    """

    __slots__ = ("error", "preset_dict", "program", "status", "submission_epoch")

    def __init__(
        self,
        program: PresetProgram | None = None,
        error: str | None = None,
        status: str = "pending",
        preset_dict: dict | MilkPreset | None = None,
        submission_epoch: int = 0,
    ):
        self.program = program
        self.error = error
        self.status = status
        self.preset_dict = preset_dict
        self.submission_epoch = submission_epoch

    @property
    def is_success(self) -> bool:
        return self.status == "success" and self.program is not None

    @property
    def is_failure(self) -> bool:
        return self.status == "failed"


class AsyncPresetCompiler:
    """Background preloader that produces a :class:`PresetProgram` for a
    preset without blocking the GL thread.

    The preloader is keyed by ``(path, mtime)`` so that a stale cached
    result from a previous build can never be confused with a fresh one.
    The main preset loader consults :attr:`get_result` before its
    synchronous parse/compile path, eliminating the double-compile that
    used previously to occur when the user landed on a preset that the
    preloader had already handled.
    """

    _queue: ClassVar[Queue] = Queue(maxsize=2)
    _worker_thread: ClassVar[threading.Thread | None] = None
    _stop_event: ClassVar[threading.Event] = threading.Event()
    # Keyed by (path, mtime) to prevent stale entries returning on a
    # preset file that has been edited on disk between submission and load.
    _results: ClassVar[OrderedDict[tuple[str, float], AsyncCompileResult]] = (
        OrderedDict()
    )
    _lock: ClassVar[threading.Lock] = threading.Lock()
    _active: ClassVar[bool] = False

    # Submission epoch.  Each call to stop() bumps this value; the worker
    # discards any queue item whose recorded epoch does not match the
    # current one.  This is what lets a restarted worker skip obsolete
    # requests that were enqueued before the stop (e.g. after a context
    # loss or a burst of failed compiles) instead of processing them ahead
    # of newer submissions.
    _submission_epoch: ClassVar[int] = 0

    # Maximum number of compiled presets to keep in the async cache.
    # Entries are evicted in LRU order (least-recently inserted first).
    MAX_PRELOADED_PRESETS: ClassVar[int] = 32

    @classmethod
    def start(cls):
        with cls._lock:
            if cls._worker_thread is not None and cls._worker_thread.is_alive():
                return
            cls._active = True
            cls._stop_event.clear()
            cls._worker_thread = threading.Thread(target=cls._worker, daemon=True, name="AsyncPresetCompiler")
            cls._worker_thread.start()


    @classmethod
    def stop(cls):
        cls._stop_event.set()
        cls._active = False
        cls._submission_epoch += 1
        with cls._lock:
            cls._results.clear()
        while True:
            try:
                cls._queue.get_nowait()
            except Empty:
                break
            except Exception:  # noqa: BLE001
                break
        worker = cls._worker_thread
        if worker is not None and worker.is_alive():
            worker.join(timeout=2.0)
            if worker.is_alive():
                logger.warning(
                    "AsyncPresetCompiler worker did not stop within 2s timeout"
                )
                cls._worker_thread = worker
                return
        cls._worker_thread = None


    @classmethod
    def submit(
        cls,
        preset_path: str,
        preset_dict: dict | MilkPreset | None = None,
        base_params: dict | None = None,
    ) -> PresetProgram | None:
        try:
            mtime = os.path.getmtime(preset_path)
        except OSError:
            mtime = 0.0
        try:
            cls._queue.put_nowait(
                (
                    preset_path,
                    mtime,
                    preset_dict,
                    base_params or {},
                    cls._submission_epoch,
                )
            )
        except Full:
            logger.debug(
                "Async preset queue full; dropping prefetch for %s", preset_path
            )
        return None

    @classmethod
    def _worker(cls):
        while True:
            try:
                item = cls._queue.get(timeout=0.5)
            except Empty:
                if cls._stop_event.is_set():
                    return
                continue
            except Exception as e:  # noqa: BLE001
                logger.debug("AsyncPresetCompiler queue error: %s", e)
                if cls._stop_event.is_set():
                    return
                continue
            if item is None:
                if cls._stop_event.is_set():
                    return
                continue
            if len(item) == 4:
                preset_path, mtime, preset_dict, _base_params = item
                submission_epoch = 0
            else:
                (
                    preset_path,
                    mtime,
                    preset_dict,
                    _base_params,
                    submission_epoch,
                ) = item
            if submission_epoch != cls._submission_epoch:
                logger.debug(
                    "Dropping obsolete async preset submission for %s "
                    "(epoch %d != current %d)",
                    preset_path,
                    submission_epoch,
                    cls._submission_epoch,
                )
                continue
            try:
                parsed: dict | MilkPreset = (
                    preset_dict
                    if preset_dict is not None
                    else parse_milk_file(preset_path)
                )
                program = PresetProgram(preset_dict=parsed)
                result = AsyncCompileResult(
                    program=program,
                    preset_dict=parsed,
                    status="success",
                    submission_epoch=submission_epoch,
                )
            except (PresetCompileError, SyntaxError, MemoryError, RuntimeError) as e:
                logger.warning(f"Failed to compile preset {preset_path}: {e}")
                result = AsyncCompileResult(
                    error=str(e), status="failed", submission_epoch=submission_epoch
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Failed to compile preset {preset_path}: {e}")
                result = AsyncCompileResult(
                    error=str(e), status="failed", submission_epoch=submission_epoch
                )
            with cls._lock:
                if submission_epoch != cls._submission_epoch:
                    logger.debug(
                        "Dropping stale async compile result for %s "
                        "(epoch %d != current %d)",
                        preset_path,
                        submission_epoch,
                        cls._submission_epoch,
                    )
                else:
                    cls._results[(preset_path, mtime)] = result
                    while len(cls._results) > cls.MAX_PRELOADED_PRESETS:
                        cls._results.popitem(last=False)

    @classmethod
    def get_result(
        cls, preset_path: str, mtime: float | None = None
    ) -> AsyncCompileResult | None:
        """Pop a previously-cached compile result for ``preset_path``.

        If ``mtime`` is supplied the cache is only honored when it matches
        the stored mtime, preventing a stale result from being returned
        after the file has been edited.

        Results are also validated against the current ``_submission_epoch``
        to reject stale entries from a previous worker lifecycle.

        Returns the :class:`AsyncCompileResult` on success, or ``None``
        when the result is absent (not yet compiled or evicted) or when the
        async compilation previously failed (in which case the caller falls
        back to a synchronous compile).

        When ``mtime`` is ``None`` the lookup is best-effort: the first
        matching success entry for ``preset_path`` is returned regardless of
        its stored mtime.  This is used by callers that either do not have
        reliable file metadata or that intentionally accept a stale cached
        program in exchange for avoiding a synchronous recompile.  The epoch
        check still applies, so entries from a previous worker lifecycle are
        rejected.  Callers that need to detect a file change after a
        best-effort hit should compare the result's ``source_mtime`` (if
        present) against the current file mtime before committing.
        """
        with cls._lock:
            if mtime is None:
                for key in list(cls._results.keys()):
                    if key[0] == preset_path:
                        result = cls._results.pop(key)
                        if (
                            result.is_success
                            and result.submission_epoch == cls._submission_epoch
                        ):
                            return result
                return None
            result = cls._results.pop((preset_path, mtime), None)  # type: ignore[arg-type]
            if result is None:
                pass
            elif result.is_success and result.submission_epoch == cls._submission_epoch:
                return result
            elif result.is_failure:
                logger.debug(
                    "Async compile previously failed for %s: %s",
                    preset_path,
                    result.error,
                )
                return result
            stale = [k for k in cls._results if k[0] == preset_path and k[1] != mtime]
            for k in stale:
                cls._results.pop(k, None)
            return None


# --- Compiled preset program and virtual machine ------------------
class PresetProgram:
    """A complete preset program with all execution phases compiled.
    Organizes equations by execution phase:
    - per_frame_init: Runs once at preset load/reset
    - per_frame: Runs every frame
    - per_pixel: Runs per-pixel (GPU via shader)
    - waves: Per-wave equations
    - shapes: Per-shape equations
    - warp: Per-pixel warp shader code (backtick-delimited GLSL/HLSL)
    - composite: Per-pixel composite shader code (backtick-delimited GLSL/HLSL)
    """

    def __init__(
        self,
        preset_dict: dict | MilkPreset | None = None,
        from_cache: dict | None = None,
    ):
        self.per_frame_init: list[CompiledExpr] = []
        self.per_frame: list[CompiledExpr] = []
        self.per_pixel: list[CompiledExpr] = []
        self.per_vertex: list[CompiledExpr] = []
        self.waves: list[WaveProgram] = []
        self.shapes: list[ShapeProgram] = []
        self.warp_code: str = ""
        self.comp_code: str = ""
        if from_cache is not None:
            self._compile_from_cache(from_cache)
        elif preset_dict is not None:
            self._compile_from_dict(preset_dict)

    def _compile_from_cache(self, cache_data: dict):
        """Restore compiled expressions from cache data."""
        for src in _preprocess_loop_blocks(cache_data.get("per_frame_init", [])):
            _append_compiled_expression(self.per_frame_init, src, "per_frame_init")
        for src in _preprocess_loop_blocks(cache_data.get("per_frame", [])):
            _append_compiled_expression(self.per_frame, src, "per_frame")
        for src in _preprocess_loop_blocks(cache_data.get("per_pixel", [])):
            _append_compiled_expression(self.per_pixel, src, "per_pixel")
        for src in _preprocess_loop_blocks(cache_data.get("per_vertex", [])):
            _append_compiled_expression(self.per_vertex, src, "per_vertex")
        for wd in cache_data.get("waves", [])[:4]:
            wp = _compile_wave(wd)
            if wp:
                self.waves.append(wp)
        for sd in cache_data.get("shapes", [])[:4]:
            sp = _compile_shape(sd)
            if sp:
                self.shapes.append(sp)

    def _compile_from_dict(self, preset_dict: dict | MilkPreset):
        """Compile all equations from parsed preset dictionary."""
        per_frame_list = _preprocess_loop_blocks(preset_dict.get("per_frame", []))
        self.per_frame = _compile_expression_list(per_frame_list)
        per_pixel_list = _preprocess_loop_blocks(preset_dict.get("per_pixel", []))
        self.per_pixel = _compile_expression_list(per_pixel_list, True)
        per_vertex_list = _preprocess_loop_blocks(preset_dict.get("per_vertex", []))
        self.per_vertex = _compile_expression_list(per_vertex_list, True)
        per_frame_init_list = _preprocess_loop_blocks(
            preset_dict.get("per_frame_init", [])
        )
        self.per_frame_init = _compile_expression_list(per_frame_init_list)
        waves = preset_dict.get("waves", {})
        if waves is None:
            waves = {}
        waves_list = list(waves.values()) if isinstance(waves, dict) else waves
        for wave_data in waves_list[:4]:
            wp = _compile_wave(wave_data)
            if wp:
                self.waves.append(wp)
        shapes = preset_dict.get("shapes", {})
        if shapes is None:
            shapes = {}
        shapes_list = list(shapes.values()) if isinstance(shapes, dict) else shapes
        for shape_data in shapes_list[:4]:
            sp = _compile_shape(shape_data)
            if sp:
                self.shapes.append(sp)

    @staticmethod
    def _extract_warp_comp_code(parsed: MilkPreset) -> tuple[str, str]:
        """Extract warp_ and comp_ backtick-delimited shader code from a parsed preset.

        The parser stores lines like ``warp_1=``gsl code```` and ``comp_2=``gsl code````
        in ``parsed.lines``.  We scan those lines and collect the backtick-delimited
        shader bodies into two strings keyed by their prefix.
        """
        warp_parts: list[str] = []
        comp_parts: list[str] = []
        inside_warp = False
        inside_comp = False
        current_key: str | None = None
        current_val: list[str] = []

        def flush():
            nonlocal inside_warp, inside_comp, current_key, current_val
            if not current_val:
                return
            body = "\n".join(current_val).strip()
            if not body:
                return
            prefix = current_key or ""
            if prefix.startswith("warp_"):
                warp_parts.append(body)
            elif prefix.startswith("comp_"):
                comp_parts.append(body)
            current_key = None
            current_val = []

        for line in getattr(parsed, "lines", []) or []:
            stripped = line.strip()
            if not stripped:
                continue

            # Track backtick-delimited blocks: ``code``
            # A line starting a block has `` at the beginning after strip
            if stripped.startswith("``"):
                # starting a block
                inside_warp = current_key is not None and current_key.startswith("warp_") if current_key else False
                inside_comp = current_key is not None and current_key.startswith("comp_") if current_key else False
                current_val = []
                continue
            if stripped.endswith("``") and (not inside_warp and not inside_comp):
                # potential end of block on its own line
                # check if there's content before the closing ``
                inner = stripped[:-2].strip()
                if inner:
                    current_val.append(inner)
                flush()
                inside_warp = False
                inside_comp = False
                current_key = None
                current_val = []
                continue

            # Process as assignment key = value
            if "=" in stripped:
                key, val = stripped.split("=", 1)
                key = key.strip()
                val = val.strip().rstrip(";")
                # Check if val starts a backtick block
                if val.startswith("`"):
                    # The code after = is backtick-delimited
                    # Extract content between first and last backtick
                    # Handle multi-line: val may be `code\nmore code`
                    inner = val[1:]  # strip leading backtick
                    if inner.endswith("`"):
                        inner = inner[:-1]  # strip trailing backtick
                    else:
                        # multi-line block - the backtick may be on its own line
                        # collect until we find closing backtick
                        current_val.append(inner)
                        # Don't flush yet - wait for more lines
                        continue

                # Regular assignment (not backtick) - check for warp_/comp_ keys
                if key.startswith(("warp_", "comp_")):
                    # This is a non-backtick assignment, but we already handle backtick
                    # For now, just record the key and skip
                    current_key = key
                    # Don't flush existing val since this is a new key
                    flush()
                    continue

            # If we're inside a backtick block, accumulate lines
            if inside_warp or inside_comp:
                current_val.append(stripped)
                # Check if this line contains the closing backtick
                if "``" in stripped:
                    flush()
                    inside_warp = False
                    inside_comp = False
                    current_key = None
                    current_val = []

        # Final flush
        flush()

        warp_code = "\n\n".join(warp_parts) if warp_parts else ""
        comp_code = "\n\n".join(comp_parts) if comp_parts else ""
        return warp_code, comp_code

    def get_stats(self) -> dict:
        """Get compilation statistics."""
        return {
            "per_frame_init": len(self.per_frame_init),
            "per_frame": len(self.per_frame),
            "per_pixel": len(self.per_pixel),
            "per_vertex": len(self.per_vertex),
            "waves": len(self.waves),
            "shapes": len(self.shapes),
        }


class PresetVM:
    """The main preset virtual machine.
    Executes preset programs with proper deterministic order:
    1. per_frame_init (on load/reset)
    2. per_frame (every frame)
    3. per_pixel (GPU - handled separately)
    4. waves/shapes (during render)
    """

    def __init__(self, program: PresetProgram, preset_params: dict | None = None):
        self.program = program
        self.preset_params = dict(preset_params or {})
        self.state = PresetState()
        self.init_run = False
        self._wave_init_done = [False] * len(getattr(program, "waves", []) or [])
        self._shape_init_done = [False] * len(getattr(program, "shapes", []) or [])
        self.base_params = {}
        for vname, vdef in sorted(VARIABLES.items(), key=lambda x: x[1].index):
            if vdef.category in ("animation", "wave"):
                self.base_params[vname] = vdef.default
        self.base_params["wave_mode"] = VARIABLES["wave_mode"].default
        for vname, vdef in sorted(VARIABLES.items(), key=lambda x: x[1].index):
            if vname in ("echo_zoom", "echo_alpha", "echo_orient"):
                self.base_params[vname] = vdef.default
        if preset_params:
            for key, value in preset_params.items():
                if isinstance(value, (int, float)):
                    kl = key.lower()
                    if kl in VARIABLES:
                        self.base_params[kl] = float(value)
                        self.state.set(kl, value)
            for preset_key, state_key in (
                ("nWaveMode", "wave_mode"),
                ("fWaveAlpha", "wave_a"),
                ("fWaveScale", "fwavescale"),
                ("fWaveSmoothing", "fwavesmoothing"),
                ("fWaveParam", "fwaveparam"),
            ):
                if state_key not in preset_params and preset_key in preset_params:
                    value = preset_params[preset_key]
                    if isinstance(value, (int, float)):
                        self.base_params[state_key] = float(value)
                        self.state.set(state_key, value)
        for key in list(self.base_params.keys()):
            if str(key).lower().startswith("q"):
                raise AssertionError(
                    f"PresetVM.base_params must not include q-registers, found: {key!r}"
                )
        self._dirty_base_keys: set = set()
        self._last_base_values: dict = dict(self.base_params)
        # Classic (audio) waveform samples.  These hold the live audio
        # waveform drawn by the base waveform path and by waves that do not
        # define per-point equations.
        self.wave_x = np.zeros(512, dtype=np.float32)
        self.wave_y = np.zeros(512, dtype=np.float32)
        # Per-wave sample geometry produced by the per-point equations.
        # One buffer per wave: a single shared buffer made every wave of a
        # multi-wave preset render the geometry of whichever wave was
        # evaluated last.
        self.wave_buffers: list[WaveSamples] = []
        # Aggregate per-frame budget that is checked across all VM expressions.
        self._frame_budget = FrameBudget(MAX_FRAME_BUDGET)
        self._frame_budget_init = FrameBudget(MAX_FRAME_BUDGET_INIT)
        # The wave per-point phase runs against its own budget: it performs
        # one evaluation per sample point per equation, so its cost scales
        # with the preset content rather than with the number of top-level
        # expressions.  See MAX_PER_POINT_BUDGET.
        self._per_point_budget = FrameBudget(self._per_point_budget_total(), warn=False, label="Per-point")
        # Whether a truncated per-point phase has already been reported for
        # this preset.  A preset that is always too slow would otherwise log
        # a warning on every frame.
        self._per_point_truncation_reported = False
        # Whether the shared frame budget exhaustion that truncated the
        # wave/shape phases has already been reported for this preset.
        # One-time so a pathological preset does not spam the log.
        self._frame_budget_truncation_reported = False

    def reset(self):
        """Reset VM state (for preset reload)."""
        self.state = PresetState()
        for k, v in self.base_params.items():
            self.state.set(k, v)
        self._last_base_values = dict(self.base_params)
        self._dirty_base_keys.clear()
        self.init_run = False
        self._wave_init_done = [False] * len(getattr(self.program, "waves", []) or [])
        self._shape_init_done = [False] * len(getattr(self.program, "shapes", []) or [])
        self._frame_budget.reset()
        self._frame_budget_init.reset()
        self._per_point_budget.reset()
        self._per_point_truncation_reported = False
        self._frame_budget_truncation_reported = False
        for buf in self.wave_buffers:
            buf.valid_points = 0

    def run_frame(
        self,
        time: float,
        frame: int,
        audio_levels: Any,
        width: float = 1.0,
        height: float = 1.0,
        dt: float | None = None,
        fps: float = 60.0,
    ) -> PresetState:
        """Execute one full frame of the preset."""
        try:
            return self._evaluate_frame_refactored(
                time=time,
                frame=frame,
                audio_levels=audio_levels,
                width=width,
                height=height,
                dt=dt,
                fps=fps,
            )
        except PresetBudgetExhausted as exc:
            logger.warning(
                "Preset VM frame budget exhausted; keeping the last known-good "
                "state (%s)",
                exc,
            )
            return self.state

    def _apply_dirty_base(self):
        """Apply dirty base params to state."""
        s = self.state
        changed_keys = self._dirty_base_keys
        if changed_keys:
            bp = self.base_params
            lv = self._last_base_values
            for key in changed_keys:
                val = bp[key]
                if lv.get(key) != val:
                    s.set(key, val)
                    lv[key] = val
            changed_keys.clear()
        else:
            for key, val in self.base_params.items():
                if self._last_base_values.get(key) != val:
                    s.set(key, val)
                    self._last_base_values[key] = val


    def _eval_expr_list(self, exprs, state, frame_budget=None, max_iterations=None, phase="expr"):
        if not exprs:
            return True
        for expr in exprs:
            try:
                if max_iterations is not None:
                    expr.eval(state, max_iterations=max_iterations, frame_budget=frame_budget)
                else:
                    expr.eval(state, frame_budget=frame_budget)
                expr.runtime_failures = 0
            except PresetBudgetExhausted:
                return False
            except (
                PresetCompileError,
                SyntaxError,
                RuntimeError,
                ValueError,
                TypeError,
                ZeroDivisionError,
                OverflowError,
            ) as e:
                if getattr(expr, "bytecode_prog", "unset") is None:
                    continue
                logger.exception(
                    "%s failed: %s",
                    phase,
                    getattr(expr, "source", "<unknown>"),
                    exc_info=e,
                )
                expr.runtime_failures = getattr(expr, "runtime_failures", 0) + 1
                if expr.runtime_failures >= 3:
                    expr.bytecode_prog = None
        return True

    def _sync_shape_program_from_state(self, shape, state):
        # Copy animatable outputs back to the compiled shape for rendering.
        # Use custom-shape-prefixed names when present in state; avoid clobbering
        # compiled defaults from unrelated global state.
        mapping = {
            "enabled":"enabled","sides":"sides","additive":"additive","thick_outline":"thick_outline","textured":"textured",
            "r":"r","g":"g","b":"b","a":"a","r2":"r2","g2":"g2","b2":"b2","a2":"a2",
            "x":"x","y":"y","rad":"rad","ang":"ang","tex_ang":"tex_ang","tex_zoom":"tex_zoom",
            "border_r":"border_r","border_g":"border_g","border_b":"border_b","border_a":"border_a","border_size":"border_size",
            "num_instances":"num_inst","instance":"instance",
            "t1":"t1","t2":"t2","t3":"t3","t4":"t4","t5":"t5","t6":"t6","t7":"t7","t8":"t8",
        }
        for attr, state_name in mapping.items():
            if not hasattr(shape, attr):
                continue
            try:
                val = state.get(state_name)
            except Exception:  # noqa: BLE001, S112
                continue
            try:
                cur = getattr(shape, attr)
                if isinstance(cur, bool):
                    setattr(shape, attr, bool(float(val)))
                elif isinstance(cur, int) and not isinstance(cur, bool):
                    iv = int(float(val))
                    if attr == "sides":
                        iv = min(max(iv, 0), 4096)
                    elif attr == "num_instances":
                        iv = min(max(iv, 1), 1024)
                    setattr(shape, attr, iv)
                else:
                    fv = float(val)
                    if math.isfinite(fv):
                        setattr(shape, attr, fv)
            except Exception:  # noqa: BLE001, S110
                pass

    def _sync_wave_program_from_state(self, wave, state):
        # Custom waves have no dedicated per-wave state namespace in this VM;
        # only copy obviously wave-specific outputs that equations commonly set.
        # Never clobber compiled structural fields (samples, mode flags, per_point)
        # from global defaults, which breaks per-point evaluation/rendering.
        for name in ("r", "g", "b", "a", "x", "y",
                     "t1","t2","t3","t4","t5","t6","t7","t8"):
            try:
                val = state.get(name)
            except Exception:  # noqa: BLE001, S112
                continue
            try:
                fv = float(val)
                if math.isfinite(fv):
                    setattr(wave, name, fv)
            except Exception:  # noqa: BLE001, S110
                pass

    def _eval_waves_init(self, s, _budget):
        if not self.program.waves:
            return
        if len(self._wave_init_done) != len(self.program.waves):
            self._wave_init_done = [False] * len(self.program.waves)
        for i, wave in enumerate(self.program.waves):
            if self._wave_init_done[i]:
                continue
            init_exprs = getattr(wave, "per_frame_init", None) or []
            if init_exprs:
                ws = s.shallow_clone()
                ws.vars["wave_index"] = i
                try:
                    self._eval_expr_list(init_exprs, ws, frame_budget=_budget, max_iterations=MAX_INIT_LOOP_ITERATIONS, phase=f"Wave {i} init")
                finally:
                    ws.vars.pop("wave_index", None)
                # init writes back q/t/regs and wave params via main state copy pattern is complex;
                # apply wave params and shared registers.
                self._sync_wave_program_from_state(wave, ws)
                s.q = list(ws.q)
                s.t_vars = list(ws.t_vars)
            self._wave_init_done[i] = True

    def _eval_shapes_init(self, s, _budget):
        if not self.program.shapes:
            return
        if len(self._shape_init_done) != len(self.program.shapes):
            self._shape_init_done = [False] * len(self.program.shapes)
        for i, shape in enumerate(self.program.shapes):
            if self._shape_init_done[i]:
                continue
            init_exprs = getattr(shape, "per_frame_init", None) or []
            if init_exprs:
                ss = s.shallow_clone_for_shapes()
                ss.vars["shape_index"] = i
                try:
                    self._eval_expr_list(init_exprs, ss, frame_budget=_budget, max_iterations=MAX_INIT_LOOP_ITERATIONS, phase=f"Shape {i} init")
                finally:
                    ss.vars.pop("shape_index", None)
                self._sync_shape_program_from_state(shape, ss)
                s.q = list(ss.q)
                s.t_vars = list(ss.t_vars)
            self._shape_init_done[i] = True

    def _eval_per_frame_init(self, s, _budget):
        if self.init_run:
            return _budget
        self._frame_budget_init.reset()
        _budget_init = self._frame_budget_init
        init_state = s.shallow_clone()
        init_ok = True
        if _budget_init.exhausted and not self.init_run:
            logger.warning(
                "Frame budget already exhausted before per_frame_init "
                "(consumed=%d/%d)",
                _budget_init.consumed,
                _budget_init.total,
            )
        for expr in self.program.per_frame_init:
            try:
                expr.eval(
                    init_state,
                    max_iterations=MAX_INIT_LOOP_ITERATIONS,
                    frame_budget=_budget_init,
                )
            except PresetBudgetExhausted:
                init_ok = False
                break
            except (
                PresetCompileError,
                SyntaxError,
                RuntimeError,
                ValueError,
                TypeError,
                ZeroDivisionError,
                OverflowError,
            ) as e:
                if getattr(expr, "bytecode_prog", "unset") is None:
                    init_ok = False
                    continue
                logger.exception(
                    "Per-frame-init expression failed: %s",
                    getattr(expr, "source", "<unknown>"),
                    exc_info=e,
                )
                expr.runtime_failures += 1
                if expr.runtime_failures >= 3:
                    expr.bytecode_prog = None
                init_ok = False
        if init_ok:
            for expr in self.program.per_frame_init:
                expr.runtime_failures = 0
            s.vars = init_state.vars
            s.vars_array = init_state.vars_array
            s.q = init_state.q
            s.t_vars = init_state.t_vars
            s.megabuf = init_state.megabuf
            s.zoom = init_state.zoom
            s.rot = init_state.rot
            s.warp = init_state.warp
            s.cx = init_state.cx
            s.cy = init_state.cy
            s.dx = init_state.dx
            s.dy = init_state.dy
            s.sx = init_state.sx
            s.sy = init_state.sy
            s.decay = init_state.decay
            s.rand_seed = init_state.rand_seed
            self.init_run = True
            self._eval_waves_init(s, _budget_init)
            self._eval_shapes_init(s, _budget_init)
        return _budget_init


    def _eval_per_frame(self, s, _budget, _wall_deadline):
        executed_count = 0
        _wall_timeout = False
        for expr in self.program.per_frame:
            if _wall_timeout or _monotonic() > _wall_deadline:
                if not _wall_timeout:
                    _wall_timeout = True
                    logger.warning(
                        "VM evaluation wall-clock timeout (%.2fs) reached during "
                        "per_frame; %d expressions executed, %d remaining skipped",
                        _VM_EVAL_TIMEOUT,
                        executed_count,
                        len(self.program.per_frame) - executed_count,
                    )
                break
            try:
                expr.eval(s, frame_budget=_budget)
                expr.runtime_failures = 0
                executed_count += 1
            except PresetBudgetExhausted:
                break
            except (
                PresetCompileError,
                SyntaxError,
                RuntimeError,
                ValueError,
                TypeError,
                ZeroDivisionError,
                OverflowError,
            ) as e:
                if getattr(expr, "bytecode_prog", "unset") is None:
                    continue
                logger.exception(
                    "Per-frame expression failed: %s",
                    getattr(expr, "source", "<unknown>"),
                    exc_info=e,
                )
                expr.runtime_failures += 1
                if expr.runtime_failures >= 3:
                    expr.bytecode_prog = None
        return _wall_timeout


    def _eval_waves(self, s, _budget, _wall_deadline, _wall_timeout):
        for i, wave in enumerate(self.program.waves):
            if _wall_timeout or _monotonic() > _wall_deadline:
                if not _wall_timeout:
                    _wall_timeout = True
                    logger.warning(
                        "VM evaluation wall-clock timeout (%.2fs) reached during "
                        "wave %d setup; remaining waves skipped",
                        _VM_EVAL_TIMEOUT,
                        i,
                    )
                break
            if _budget.exhausted:
                self._log_frame_budget_truncation(
                    "Frame budget exhausted at wave %d; remaining waves and "
                    "shapes skipped",
                    i,
                )
                break
            s.vars["wave_index"] = i
            try:
                for expr in wave.per_frame:
                    try:
                        expr.eval(s, frame_budget=_budget)
                        expr.runtime_failures = 0
                    except PresetBudgetExhausted:
                        break
                    except (
                        PresetCompileError,
                        SyntaxError,
                        RuntimeError,
                        ValueError,
                        TypeError,
                        ZeroDivisionError,
                        OverflowError,
                    ) as e:
                        if getattr(expr, "bytecode_prog", "unset") is None:
                            continue
                        logger.exception(
                            "Per-wave expression failed: wave_index=%s expr=%s",
                            i,
                            getattr(expr, "source", "<unknown>"),
                            exc_info=e,
                        )
                        expr.runtime_failures += 1
                        if expr.runtime_failures >= 3:
                            expr.bytecode_prog = None
                self._sync_wave_program_from_state(wave, s)
            finally:
                s.vars.pop("wave_index", None)
        return _wall_timeout


    def _eval_shapes(self, s, _budget, _wall_deadline, _wall_timeout):
        for i, shape in enumerate(self.program.shapes):
            if _wall_timeout or _monotonic() > _wall_deadline:
                if not _wall_timeout:
                    _wall_timeout = True
                    logger.warning(
                        "VM evaluation wall-clock timeout (%.2fs) reached during "
                        "shape %d; remaining shapes skipped",
                        _VM_EVAL_TIMEOUT,
                        i,
                    )
                break
            if not shape.per_frame:
                continue
            if _budget.exhausted:
                self._log_frame_budget_truncation(
                    "Frame budget exhausted at shape %d; remaining shapes skipped", i
                )
                break
            s.vars["shape_index"] = i
            try:
                shape_state = s.shallow_clone_for_shapes()
                shape_state.vars["shape_index"] = i
                for expr in shape.per_frame:
                    try:
                        expr.eval(shape_state, frame_budget=_budget)
                        expr.runtime_failures = 0
                    except PresetBudgetExhausted:
                        break
                    except (
                        PresetCompileError,
                        SyntaxError,
                        RuntimeError,
                        ValueError,
                        TypeError,
                        ZeroDivisionError,
                        OverflowError,
                    ) as e:
                        if getattr(expr, "bytecode_prog", "unset") is None:
                            continue
                        logger.exception(
                            "Per-shape expression failed: shape_index=%s expr=%s",
                            i,
                            getattr(expr, "source", "<unknown>"),
                            exc_info=e,
                        )
                        expr.runtime_failures += 1
                        if expr.runtime_failures >= 3:
                            expr.bytecode_prog = None
                self._sync_shape_program_from_state(shape, shape_state)
                for _name, idx in VAR_REGISTRY.items():
                    s.vars_array[idx] = float(shape_state.vars_array[idx])
                    if _name in PresetState._SHAPE_STATE_ATTRS:
                        try:
                            setattr(s, _name, float(shape_state.vars_array[idx]))
                        except Exception as exc:  # noqa: BLE001
                            logger.debug("Exception in try block: %s", exc)
                s.q = list(shape_state.q)
                s.t_vars = list(shape_state.t_vars)
            finally:
                s.vars.pop("shape_index", None)
        return _wall_timeout


    def _evaluate_frame_refactored(
        self,
        time: float,
        frame: int,
        audio_levels: Any,
        width: float = 1.0,
        height: float = 1.0,
        dt: float | None = None,
        fps: float = 60.0,
    ) -> PresetState:
        s = self.state
        self._apply_dirty_base()
        self._frame_rand_seed = s.rand_seed
        s.begin_frame()
        s.update_time(time, frame, dt=dt, fps=fps)
        s.update_audio(audio_levels)
        s.update_resolution(width, height)
        if not self.init_run:
            _budget = self._frame_budget_init
        else:
            _budget = self._frame_budget
            self._frame_budget.reset()
        self._per_point_budget.reset()
        _wall_start = _monotonic()
        _wall_deadline = _wall_start + _VM_EVAL_TIMEOUT
        _wall_timeout = False
        _budget = self._eval_per_frame_init(s, _budget)
        _wall_timeout = self._eval_per_frame(s, _budget, _wall_deadline)
        _wall_timeout = self._eval_waves(s, _budget, _wall_deadline, _wall_timeout)
        _pp_budget = self._per_point_budget
        # Per-point work is reported via _log_per_point_truncation; suppress
        # generic FrameBudget exhaustion warnings for ad-hoc/test budgets.
        try:
            _pp_budget._warn = False
            _pp_budget._label = "Per-point"
        except Exception:  # noqa: BLE001, S110
            pass
        # Create the per-wave buffers up front so the snapshot below covers
        # every wave: a buffer created during the phase would otherwise have
        # no snapshot to roll back to.
        for wave_idx in range(len(self.program.waves)):
            self._wave_buffer(wave_idx)
        # Only waves that actually run per-point equations can be modified
        # by the phase, so only those need a rollback snapshot.  Copying
        # every wave's six sample arrays each frame was pure overhead for
        # the common case where no wave uses per-point equations.
        _wave_snapshot: list[WaveSamples | None] = [None] * len(self.wave_buffers)
        for wave_idx, wave in enumerate(self.program.waves):
            if wave_idx >= len(self.wave_buffers):
                break
            if not (getattr(wave, "per_point", None) or []):
                continue
            _wave_snapshot[wave_idx] = self.wave_buffers[wave_idx].copy()
        _per_point_truncated = False
        _truncated_wave = -1
        _truncated_points = 0
        for i, wave in enumerate(self.program.waves):
            if not getattr(wave, "enabled", True):
                continue
            if _pp_budget.exhausted:
                self._log_per_point_truncation(
                    "Per-point budget exhausted; wave %d per_point and the "
                    "remaining waves are skipped",
                    i,
                )
                _per_point_truncated = True
                _truncated_wave = i
                break
            per_point_count = len(wave.per_point)
            if per_point_count > 0:
                num_samples = int(getattr(wave, "samples", 512))
                num_samples = min(num_samples, len(self.wave_x), len(self.wave_y))
                if num_samples > 0:
                    points_done = 0
                    for point_idx in range(num_samples):
                        if _wall_timeout or _monotonic() > _wall_deadline:
                            if not _wall_timeout:
                                _wall_timeout = True
                                logger.warning(
                                    "VM evaluation wall-clock timeout (%.2fs) "
                                    "reached during wave %d per_point sample "
                                    "%d/%d; remaining samples skipped",
                                    _VM_EVAL_TIMEOUT,
                                    i,
                                    point_idx,
                                    num_samples,
                                )
                            _per_point_truncated = True
                            break
                        if _pp_budget.exhausted:
                            self._log_per_point_truncation(
                                "Per-point budget exhausted at wave %d per_point "
                                "sample %d/%d",
                                i,
                                point_idx,
                                num_samples,
                            )
                            _per_point_truncated = True
                            break
                        sample_val = (point_idx + 0.5) / num_samples
                        try:
                            self.run_wave_per_point(
                                i,
                                point_idx,
                                sample_val,
                                num_samples,
                                frame_budget=_pp_budget,
                            )
                        except PresetBudgetExhausted:
                            # Budget exhausted mid-sample: skip the remaining
                            # samples for this wave and roll the whole wave
                            # phase back below.
                            self._log_per_point_truncation(
                                "Per-point budget exhausted at wave %d per_point "
                                "sample %d/%d",
                                i,
                                point_idx,
                                num_samples,
                            )
                            _per_point_truncated = True
                            break
                        points_done = point_idx + 1
                    if _per_point_truncated:
                        _truncated_wave = i
                        _truncated_points = points_done
                        break
                    self._wave_buffer(i).valid_points = num_samples
        if _per_point_truncated:
            if any(buf.valid_points > 0 for buf in self.wave_buffers):
                # Restore the last known-good wave geometry.  A partially
                # evaluated wave would otherwise be drawn as a mixture of
                # this frame's and the previous frame's samples.
                for buf, snapshot in zip(self.wave_buffers, _wave_snapshot):
                    if snapshot is not None:
                        buf.restore(snapshot)
            elif _truncated_points > 0:
                # Nothing complete to fall back on yet (first frame): keep
                # the prefix that was computed so the wave is still drawn
                # instead of disappearing entirely.
                for buf in self.wave_buffers:
                    buf.valid_points = 0
                if 0 <= _truncated_wave < len(self.program.waves):
                    self._wave_buffer(_truncated_wave).valid_points = _truncated_points
        for i, shape in enumerate(self.program.shapes):
            if _wall_timeout or _monotonic() > _wall_deadline:
                if not _wall_timeout:
                    _wall_timeout = True
                    logger.warning(
                        "VM evaluation wall-clock timeout (%.2fs) reached during "
                        "shape %d; remaining shapes skipped",
                        _VM_EVAL_TIMEOUT,
                        i,
                    )
                break
            if not shape.per_frame:
                continue
            if _budget.exhausted:
                self._log_frame_budget_truncation(
                    "Frame budget exhausted at shape %d; remaining shapes skipped", i
                )
                break
            s.vars["shape_index"] = i
            try:
                shape_state = s.shallow_clone_for_shapes()
                shape_state.vars["shape_index"] = i
                for expr in shape.per_frame:
                    try:
                        expr.eval(shape_state, frame_budget=_budget)
                        expr.runtime_failures = 0
                    except PresetBudgetExhausted:
                        # Budget exhausted mid-shape: skip remaining shapes.
                        break
                    except (
                        PresetCompileError,
                        SyntaxError,
                        RuntimeError,
                        ValueError,
                        TypeError,
                        ZeroDivisionError,
                        OverflowError,
                    ) as e:
                        if getattr(expr, "bytecode_prog", "unset") is None:
                            continue
                        logger.exception(
                            "Per-shape expression failed: shape_index=%s expr=%s",
                            i,
                            getattr(expr, "source", "<unknown>"),
                            exc_info=e,
                        )
                        expr.runtime_failures += 1
                        if expr.runtime_failures >= 3:
                            expr.bytecode_prog = None
                for _name, idx in VAR_REGISTRY.items():
                    s.vars_array[idx] = float(shape_state.vars_array[idx])
                    if _name in PresetState._SHAPE_STATE_ATTRS:
                        try:
                            setattr(s, _name, float(shape_state.vars_array[idx]))
                        except Exception as exc:  # noqa: BLE001
                            logger.debug("Exception in try block: %s", exc)
                s.q = list(shape_state.q)
                s.t_vars = list(shape_state.t_vars)
            finally:
                s.vars.pop("shape_index", None)
        return s

    def get_gpu_uniforms(self) -> GPUUniforms:
        """Get all GPU uniforms as a typed dataclass for fast attribute access."""
        s = self.state
        u = GPUUniforms(
            zoom=s.get("zoom", 1.0),
            rot=s.get("rot", 0.0),
            warp=s.get("warp", 0.0),
            cx=s.get("cx", 0.5),
            cy=s.get("cy", 0.5),
            dx=s.get("dx", 0.0),
            dy=s.get("dy", 0.0),
            sx=s.get("sx", 1.0),
            sy=s.get("sy", 1.0),
            decay=s.get("decay", 0.9),
            wave_r=s.get("wave_r", 0.0),
            wave_g=s.get("wave_g", 0.0),
            wave_b=s.get("wave_b", 0.0),
            wave_a=s.get("wave_a", 0.0),
            wave_x=s.get("wave_x", 0.5),
            wave_y=s.get("wave_y", 0.5),
            bass=s.get("bass", 0.0),
            mid=s.get("mid", 0.0),
            treb=s.get("treb", 0.0),
            bass_att=s.get("bass_att", 0.0),
            mid_att=s.get("mid_att", 0.0),
            treb_att=s.get("treb_att", 0.0),
            beat=s.get("beat", 0.0),
            beat_flash=s.get("beat_flash", 0.0),
            rms=s.get("rms", 0.0),
            peak=s.get("peak", 0.0),
            time=s.get("time", 0.0),
            frame=float(s.get("frame")),
            fps=s.get("fps", 60.0),
            dt=s.get("dt", 0.0),
            x=s.get("x", 0.0),
            y=s.get("y", 0.0),
            rad=s.get("rad", 0.0),
            ang=s.get("ang", 0.0),
            width=s.get("width", 1.0),
            height=s.get("height", 1.0),
            aspect=s.get("aspect", 1.0),
            w=s.get("w", 1.0),
            h=s.get("h", 1.0),
            q=list(s.q),
            t=list(s.t_vars),
            echo_zoom=s.get("echo_zoom", 1.0),
            echo_alpha=s.get("echo_alpha", 0.0),
            echo_orient=int(s.get("echo_orient", 0)),
        )
        return u

    def _per_point_budget_total(self) -> int:
        """Size the per-point budget from the work the phase has to perform.

        Every sample point evaluates every per-point equation of every wave
        once, so the number of evaluations is known before the phase runs.
        The budget covers those evaluations plus the per-instruction charge
        (one unit per 256 bytecode instructions) and is capped by
        :data:`MAX_PER_POINT_BUDGET` so a pathological preset cannot run
        unbounded; the wall-clock timeout remains the hard CPU bound.
        """
        evaluations = 0
        for wave in self.program.waves:
            if not getattr(wave, "enabled", True):
                continue
            per_point = getattr(wave, "per_point", None) or []
            if not per_point:
                continue
            samples = int(getattr(wave, "samples", 512) or 512)
            samples = min(samples, len(self.wave_x), len(self.wave_y))
            evaluations += max(0, samples) * len(per_point)
        if evaluations <= 0:
            return 0
        total = evaluations * PER_POINT_BUDGET_INSTRUCTION_FACTOR
        total += PER_POINT_BUDGET_SLACK
        return min(total, MAX_PER_POINT_BUDGET)

    def _log_per_point_truncation(self, message: str, *args) -> None:
        """Report a truncated per-point phase.

        The first truncation of a preset is a warning — it means the wave
        geometry is incomplete.  Later truncations of the same preset are
        demoted to debug output: a preset that is always too slow would
        otherwise log a warning on every frame.
        """
        if self._per_point_truncation_reported:
            logger.debug(message, *args)
        else:
            logger.warning(message, *args)
            self._per_point_truncation_reported = True

    def _log_frame_budget_truncation(self, message: str, *args) -> None:
        """Report a wave/shape phase truncated by the shared frame budget.

        When the aggregate per-frame budget is exhausted mid-frame, waves
        and shapes may be skipped.  The first such truncation is a warning;
        subsequent ones are demoted to debug so a pathological preset does
        not spam the log at frame rate.
        """
        if self._frame_budget_truncation_reported:
            logger.debug(message, *args)
        else:
            logger.warning(message, *args)
            self._frame_budget_truncation_reported = True

    def _wave_buffer(self, wave_idx: int) -> WaveSamples:
        """Return the sample buffer of wave *wave_idx*, creating it if needed."""
        while len(self.wave_buffers) <= wave_idx:
            self.wave_buffers.append(WaveSamples(len(self.wave_x)))
        return self.wave_buffers[wave_idx]

    def run_wave_per_point(
        self,
        wave_idx: int,
        point_idx: int,
        sample_val: float,
        num_points: int,
        frame_budget: FrameBudget | None = None,
    ):
        """Run per-point equations for a wave.
        This updates the wave's sample buffer for the given point.

        When *frame_budget* is provided it is forwarded to each per-point
        bytecode expression so that heavy wave equations charge against the
        per-point budget.
        """
        if wave_idx >= len(self.program.waves):
            return
        if not (0 <= point_idx < len(self.wave_x)):
            return
        s = self.state
        wave = self.program.waves[wave_idx]
        # Initialize per-point inputs (MilkDrop exposes sample/value/point semantics)
        s.vars["wave_index"] = wave_idx
        s.vars["point"] = point_idx
        s.vars["sample"] = sample_val
        s.vars["value1"] = sample_val
        s.vars["value2"] = sample_val
        s.vars["num_points"] = num_points
        # Seed per-point color/position from wave/per-frame state so equations can modify them.
        s.set("x", s.get("x", 0.5))
        s.set("y", s.get("y", 0.5))
        for expr in wave.per_point:
            try:
                expr.eval(s, frame_budget=frame_budget)
                expr.runtime_failures = 0
            except PresetBudgetExhausted:
                return
            except (
                PresetCompileError,
                SyntaxError,
                RuntimeError,
                ValueError,
                TypeError,
                ZeroDivisionError,
                OverflowError,
            ) as e:
                logger.exception(
                    "Per-point expression failed: wave_index=%s point=%s expr=%s",
                    wave_idx,
                    point_idx,
                    getattr(expr, "source", "<unknown>"),
                    exc_info=e,
                )
                expr.runtime_failures += 1
                if expr.runtime_failures >= 3:
                    expr.bytecode_prog = None
        samples = self._wave_buffer(wave_idx)
        x = s.get("x", 0.5)
        y = s.get("y", 0.5)
        r = s.get("r", s.get("wave_r", getattr(wave, "r", 1.0)))
        g = s.get("g", s.get("wave_g", getattr(wave, "g", 1.0)))
        b = s.get("b", s.get("wave_b", getattr(wave, "b", 1.0)))
        a = s.get("a", s.get("wave_a", getattr(wave, "a", 1.0)))
        samples.x[point_idx] = 0.5 if not math.isfinite(float(x)) else float(x)
        samples.y[point_idx] = 0.5 if not math.isfinite(float(y)) else float(y)
        samples.r[point_idx] = 1.0 if not math.isfinite(float(r)) else float(r)
        samples.g[point_idx] = 1.0 if not math.isfinite(float(g)) else float(g)
        samples.b[point_idx] = 1.0 if not math.isfinite(float(b)) else float(b)
        samples.a[point_idx] = 1.0 if not math.isfinite(float(a)) else float(a)


# --- Loop preprocessing ------------------------------------------
def _extract_nested_loop(expr_list, start_idx, header):
    """Consume a nested loop beginning at *start_idx* and return ``(LoopExpr, next_idx)``.

    The outer loop's paren accounting already treats the nested loop's lines as
    plain body text, so this walks the nested loop's own extent independently:
    its header, its statements, and the line carrying its closing paren.
    """
    sub: list = [header]
    # Unclosed parens the nested loop still owns after its header line. The
    # loop's own "(" is already balanced by the ")" a self-closing header ends
    # with, so _paren_depth is exactly the outstanding count here.
    depth = _paren_depth(header)
    j = start_idx + 1
    while depth > 0 and j < len(expr_list):
        nxt = expr_list[j]
        if not isinstance(nxt, str):
            j += 1
            continue
        nxt_s = nxt.strip()
        nxt_opens = len(LOOP_START_RE.findall(nxt_s))
        nxt_delta = _paren_depth(nxt_s) - nxt_opens
        depth += nxt_opens + nxt_delta
        if depth <= 0:
            # Only this line's unmatched closing parens belong to the loop; a
            # balanced trailing ")" is part of a body statement.
            trailing = len(nxt_s) - len(nxt_s.rstrip(")"))
            strip = min(max(0, -(nxt_opens + nxt_delta)), trailing)
            nxt_s = nxt_s[: len(nxt_s) - strip] if strip else nxt_s
            nxt_s = nxt_s.rstrip(";").rstrip(",").strip()
        else:
            # Not the closing line: strip only the statement terminator. A
            # trailing comma must survive -- it is what separates a nested
            # "loop( n," header's count from its body, and dropping it made
            # the count parse as "0" so the loop body never ran.
            nxt_s = nxt_s.rstrip(";").strip()
        if nxt_s:
            sub.append(nxt_s)
        j += 1
    nested = _preprocess_loop_blocks(sub)
    if not nested or not isinstance(nested[0], LoopExpr):
        # Unterminated nested loop: keep the statements rather than dropping
        # them, matching how a malformed outer loop is handled.
        return sub, j
    return nested[0], j


def _preprocess_loop_blocks(expr_list):
    """Convert loop/while blocks in an expression list to LoopExpr objects.

    Handles nested loops and multi-line equation continuations inside loops.
    """
    result = []
    i = 0
    while i < len(expr_list):
        item = expr_list[i]
        if isinstance(item, LoopExpr):
            preprocessed_body = _preprocess_loop_blocks(item.raw_body)
            result.append(LoopExpr(item.count_expr, preprocessed_body))
            i += 1
            continue
        if not isinstance(item, str):
            result.append(item)
            i += 1
            continue
        s = item.strip()

        # Handle loop/while blocks
        count_match = LOOP_START_RE.search(s)
        if count_match is not None and _paren_depth(s[:count_match.start()]) == 0:
            after_paren = s.find("(", count_match.start())
            comma_idx = s.find(",", after_paren)
            count_str = (
                s[after_paren + 1 : comma_idx].strip()
                if comma_idx > after_paren
                else "0"
            )
            # Check if the loop header line is self-closing (e.g., "loop(3, a=1; b=2;)")
            # If the last ")" in the header line closes the loop, extract body from it
            body_eqs = []
            # A loop header whose last statement continues on the next line.
            # Initialised here so the self-closing branch below cannot leave
            # it unbound when the body is scanned.
            pending_header: str | None = None
            if s.count("(") > 0 and s.count("(") == s.count(")"):
                # Self-closing: extract body between comma and closing paren
                rest = (
                    s[comma_idx + 1 :]
                    if comma_idx > after_paren
                    else s[after_paren + 1 :]
                )
                closing = rest.rfind(")")
                if closing > 0:
                    body_str = rest[:closing]
                    body_eqs.extend(_split_loop_body(body_str))
                    depth = 0
                    i += 1
                else:
                    depth = 1
                    i += 1
            else:
                # Multi-line: extract any body content from the header line
                rest = s[comma_idx + 1 :] if comma_idx > after_paren else ""
                header_rest = rest.rstrip(",;")
                while (
                    header_rest
                    and header_rest[-1] == ")"
                    and _paren_depth(header_rest) > 0
                ):
                    header_rest = header_rest[:-1]
                header_rest = header_rest.strip().rstrip(";").strip()
                if header_rest:
                    parts = _split_loop_body(header_rest)
                    for part in parts:
                        if _is_incomplete_expr(part):
                            pending_header = part
                            break
                        body_eqs.append(part)
                    else:
                        pending_header = None
                else:
                    pending_header = None
                depth = 1
                i += 1
            while i < len(expr_list) and depth > 0:
                inner = expr_list[i]
                if isinstance(inner, LoopExpr):
                    if pending_header:
                        body_eqs.append(pending_header)
                        pending_header = None
                    body_eqs.append(inner)
                    i += 1
                    continue
                if not isinstance(inner, str):
                    i += 1
                    continue
                inner_s = inner.strip().rstrip(";").strip()
                if pending_header:
                    inner_s = pending_header + " " + _strip_inline_comments(inner_s)
                    pending_header = None
                loop_opens = len(LOOP_START_RE.findall(inner_s))
                paren_delta = _paren_depth(inner_s)
                if loop_opens > 0:
                    paren_delta = max(0, paren_delta - loop_opens)
                closes_loop = depth + loop_opens + paren_delta <= 0
                if loop_opens > 0 and not closes_loop:
                    # A nested loop opens on this body line. Recurse over its
                    # whole extent instead of leaving the bare "loop(n,"
                    # header in the body: CompiledExpr rejected it ("Expression
                    # missing assignment: 'loop( nn,'") and the inner loop's
                    # statements then ran exactly once instead of n times.
                    nested, i = _extract_nested_loop(expr_list, i, inner_s)
                    body_eqs.append(nested)
                    continue
                depth += loop_opens + paren_delta
                if closes_loop:
                    # This line closes the loop - process any body content before
                    # closing.  Strip only the ')' that closes the loop, not every
                    # trailing ')' (a body statement may legitimately end in one).
                    trailing = len(inner_s) - len(inner_s.rstrip(")"))
                    # Only this line's *unmatched* closing parens can close the
                    # loop; a balanced line keeps its own trailing ')'.
                    strip = min(max(0, -paren_delta), trailing)
                    inner_s = inner_s[: len(inner_s) - strip] if strip else inner_s
                    while inner_s and inner_s[-1] == ";":
                        inner_s = inner_s[:-1]
                    inner_s = inner_s.strip()
                    if inner_s:
                        body_eqs.extend(_split_loop_body(inner_s))
                    i += 1
                    continue
                # Join continuation lines inside the loop. A peek that opens a
                # new statement (an assignment) or a nested loop ends this
                # expression; anything else is a wrapped fragment of it and is
                # joined.  _next_line_is_continuation also catches the
                # Surface-Turbulence pattern: a complete-looking line whose
                # value ends with a balanced ')' but whose successor is a bare
                # fragment starting with a binary operator. That is what lets a
                # body statement like
                #   hit = below( sqrt( .. ) +
                #       gmegabuf(index2 + 4)
                #     )
                #     * above( sqrt( .. ) );
                # survive intact instead of being cut at the first ")".
                joined = inner_s
                while i + 1 < len(expr_list) and (
                    _is_incomplete_expr(joined)
                    or _next_line_is_continuation(expr_list[i + 1], joined)
                ):
                    peek = expr_list[i + 1]
                    if isinstance(peek, LoopExpr):
                        break
                    if not isinstance(peek, str):
                        break
                    peek_s = peek.strip()
                    if not peek_s:
                        break
                    # When inside an open function call (e.g. exec3( or exec2(),
                    # assignment lines, standalone ')' closers, and even
                    # loop() arguments that appear inside the call are part of
                    # the call, not separate statements — keep joining so the
                    # full call is preserved as one body_eq instead of being
                    # fragmented into bare "exec3(" / ")" fragments that
                    # trigger "Expression missing assignment" warnings.
                    if _paren_depth(joined) > 0:
                        if peek_s == ")" or peek_s.endswith((");", ")")):
                            joined = joined + " " + _strip_inline_comments(peek_s)
                            i += 1
                            continue
                        if ASSIGN_RE.match(peek_s) or _compound_assign_re.match(peek_s):
                            joined = joined + " " + _strip_inline_comments(peek_s)
                            i += 1
                            continue
                        if LOOP_START_RE.search(peek_s):
                            joined = joined + " " + _strip_inline_comments(peek_s)
                            i += 1
                            continue
                    if LOOP_START_RE.search(peek_s):
                        break
                    if ASSIGN_RE.match(peek_s) or _compound_assign_re.match(peek_s):
                        break
                    joined = joined + " " + _strip_inline_comments(peek_s)
                    i += 1
                body_eqs.append(joined)
                i += 1
            result.append(LoopExpr(count_str, body_eqs))
            continue

        # Join continuation lines for regular equations
        joined = s
        while i + 1 < len(expr_list) and (
            _is_incomplete_expr(joined)
            or _next_line_is_continuation(expr_list[i + 1], joined)
        ):
            peek = expr_list[i + 1]
            if isinstance(peek, LoopExpr):
                break
            if not isinstance(peek, str):
                break
            peek_s = peek.strip()
            if not peek_s:
                break
            # When inside an open function call (e.g. exec3( or exec2(), keep
            # joining through ')' closers, assignment lines, and loop()
            # arguments — the full call is preserved instead of being
            # fragmented.
            if _paren_depth(joined) > 0:
                if peek_s == ")" or peek_s.endswith((");", ")")):
                    joined = joined + " " + _strip_inline_comments(peek_s)
                    i += 1
                    continue
                if ASSIGN_RE.match(peek_s) or _compound_assign_re.match(peek_s):
                    joined = joined + " " + _strip_inline_comments(peek_s)
                    i += 1
                    continue
                if LOOP_START_RE.search(peek_s):
                    joined = joined + " " + _strip_inline_comments(peek_s)
                    i += 1
                    continue
            if (peek_s == ")" or peek_s.endswith((");", ")")) and not _next_line_is_continuation(peek_s, joined)):
                # A continuation fragment may legitimately end with ')' —
                # don't abandon it if _next_line_is_continuation confirms it
                # is a bare continuation fragment.
                break
            if LOOP_START_RE.search(peek_s):
                break
            if ASSIGN_RE.match(peek_s) or _compound_assign_re.match(peek_s):
                break
            joined = joined + " " + _strip_inline_comments(peek_s)
            i += 1
        result.append(joined)
        i += 1
    return result


# --- Preset file parsing ---------------------------------------
def parse_milk_file_robust(path: str) -> MilkPreset:
    """Robust MilkDrop preset parser with full wavecode/shape support.
    Handles the inconsistent INI-like format of .milk files.
    Tolerant of malformed lines and encoding issues.
    Supports multi-line equations, loop blocks, and inline comments.
    """
    preset = MilkPreset()
    preset.source_file = path
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            all_lines = [line.rstrip("\n") for line in f]

        i = 0
        while i < len(all_lines):
            line = all_lines[i].strip()
            i += 1

            if not line or line.startswith(("//", "[")):
                preset.lines.append(line)
                continue

            preset.lines.append(line)

            if not ASSIGN_RE.match(line):
                continue

            try:
                key, val = line.split("=", 1)
                key, val = key.strip(), val.strip().rstrip(";")
                if not val:
                    continue

                # Handle loop/while blocks
                # Skip loop detection for warp_/comp_ shader code: the `while`
                # and `loop` keywords inside backtick-delimited GLSL/HLSL are
                # shader constructs, not MilkDrop loop statements. Matching them
                # here hijacks the subsequent comp_/warp_ lines as a fake loop
                # body and feeds backtick fragments to CompiledExpr.
                # Also skip when loop( is inside another function call (e.g.
                # exec3(loop(...))) — only treat loop/while as a block when
                # at paren depth 0 (start of the expression value).
                if not key.startswith(("comp_", "warp_")) and (
                    count_match := LOOP_START_RE.search(val)
                ) is not None and _paren_depth(val[:count_match.start()]) == 0:
                    after_paren = val.find("(", count_match.start())
                    comma_idx = val.find(",", after_paren)
                    count_str = (
                        val[after_paren + 1 : comma_idx].strip()
                        if comma_idx > after_paren
                        else "0"
                    )
                    body_eqs = []
                    # Check if the loop header is self-closing (balanced parens)
                    header_paren_delta = _paren_depth(val)
                    if header_paren_delta == 0 and comma_idx > after_paren:
                        # Self-closing: extract body from the header line
                        rest = val[comma_idx + 1 :]
                        closing = rest.rfind(")")
                        if closing > 0:
                            body_str = rest[:closing]
                            body_eqs.extend(_split_loop_body(body_str))
                    else:
                        # Multi-line loop: scan subsequent lines for body
                        depth = header_paren_delta if header_paren_delta > 0 else 1
                        # Extract body content from the header line itself (after comma)
                        if comma_idx > after_paren:
                            header_rest = val[comma_idx + 1 :].rstrip(",;")
                            # Strip trailing ')' ONLY when they belong to the
                            # loop's own '(' -- i.e. the expression still has
                            # unclosed parens.  A balanced expression like
                            # "bd_treb*(bd_src==2)" keeps its closing ')', which
                            # is part of the math, not a loop terminator.
                            # Stripping it left "bd_src==2" without its ')' and
                            # produced a corrupted, un-compiled equation.
                            while (
                                header_rest
                                and header_rest[-1] == ")"
                                and _paren_depth(header_rest) > 0
                            ):
                                header_rest = header_rest[:-1]
                            header_rest = header_rest.strip().rstrip(";")
                            if header_rest:
                                for p in header_rest.split(";"):
                                    p = p.strip()
                                    if p:
                                        body_eqs.append(p)
                        while i < len(all_lines) and depth > 0:
                            body_line = all_lines[i].strip()
                            i += 1
                            if not body_line:
                                continue
                            # A body line that opens a nested loop (e.g.
                            # "loop(3, x = sin(") does not match ASSIGN_RE
                            # because it starts with the loop keyword, not a
                            # variable name.  Without this guard such lines
                            # were silently dropped, breaking nested-loop
                            # headers whose body content continues on the
                            # next line.  _preprocess_loop_blocks handles the
                            # continuation and structure once the raw line is
                            # in body_eqs.
                            if LOOP_START_RE.match(body_line):
                                if body_line:
                                    body_eqs.append(body_line)
                                loop_opens = len(LOOP_START_RE.findall(body_line))
                                paren_delta = _paren_depth(body_line)
                                if loop_opens > 0:
                                    paren_delta = max(0, paren_delta - loop_opens)
                                depth += loop_opens + paren_delta
                                if depth <= 0:
                                    break
                                i += 1
                                continue
                            if not ASSIGN_RE.match(body_line):
                                continue
                            _bkey, bval = body_line.split("=", 1)
                            # Shader code lines (warp_/comp_) use backtick-delimited
                            # GLSL/HLSL, not MilkDrop equations.  They must never be
                            # absorbed into a MilkDrop loop body.
                            if _bkey.strip().startswith(("comp_", "warp_")):
                                continue
                            bval = bval.strip().rstrip(";")
                            # Strip only the trailing ')' characters that close
                            # *this* loop.  Blanket-stripping every trailing ')'
                            # corrupted any body statement that legitimately ends
                            # in a paren -- "tmpv = gmegabuf(n+3)" lost its ')'
                            # and no longer compiled.
                            trailing = len(bval) - len(bval.rstrip(")"))
                            paren_delta = _paren_depth(bval) if bval else 0
                            # Only the *unmatched* closing parens of this line can
                            # close the loop.  A line whose parens are balanced
                            # ("bd_ampl = gmegabuf(n)") must keep its ')'; stripping
                            # it truncated the statement and broke compilation.
                            close_count = min(max(0, -paren_delta), trailing)
                            bval = (
                                bval[: len(bval) - close_count] if close_count else bval
                            )
                            bval = bval.rstrip(";").strip()
                            if bval:
                                bval = _strip_inline_comments(bval)
                                if bval:
                                    body_eqs.append(bval)
                            loop_opens = len(LOOP_START_RE.findall(bval)) if bval else 0
                            paren_delta = _paren_depth(bval) if bval else 0
                            if loop_opens > 0:
                                paren_delta = max(0, paren_delta - loop_opens)
                            depth += loop_opens + paren_delta - close_count
                            # Preserve closing ')' tokens so _preprocess_loop_blocks
                            # can find where nested loops end.  parse_milk_file_robust
                            # already balanced close_count against depth above, but
                            # the individual ')' characters were stripped from bval.
                            # _preprocess_loop_blocks scans body_eqs linearly and
                            # needs explicit ')' items to detect nested loop closing --
                            # without them it absorbs the entire remainder into the
                            # innermost loop, causing "Loop iteration budget exhausted"
                            # warnings.  The ')' that closes *this* loop (depth<=0)
                            # is omitted: it is a loop terminator, not body content.
                            if depth > 0:
                                for _ in range(close_count):
                                    body_eqs.append(")")
                            if depth <= 0:
                                break
                    # A nested ``loop(`` inside the body is collected as a raw
                    # string here; _preprocess_loop_blocks turns it into a real
                    # nested LoopExpr.  Without this the inner header stayed a
                    # bare unbalanced "loop (3," string and never compiled.
                    body_eqs = _preprocess_loop_blocks(body_eqs)
                    loop_expr = LoopExpr(count_str, body_eqs)
                    if key.startswith("per_frame_init"):
                        preset.per_frame_init.append(loop_expr)
                    elif key.startswith("per_frame"):
                        preset.per_frame.append(loop_expr)
                    elif key.startswith("per_pixel"):
                        preset.per_pixel.append(loop_expr)
                    elif key.startswith("per_vertex"):
                        preset.per_vertex.append(loop_expr)
                    elif key.startswith(("wavecode_", "wave_")):
                        m = WAVE_RE.match(key)
                        if m:
                            wave_idx, param_name = int(m.group(1)), m.group(2)
                            preset.waves.setdefault(
                                wave_idx,
                                {"enabled": True, "per_frame": [], "per_point": [], "per_frame_init": []},
                            )
                            # Normalize numbered equation names.
                            if param_name.startswith("per_point"):
                                param_name = "per_point"
                            elif param_name.startswith("per_frame_init") or param_name == "init" or (param_name.startswith("init") and param_name[4:].isdigit()):
                                param_name = "per_frame_init"
                            elif param_name.startswith("per_frame"):
                                param_name = "per_frame"
                            if param_name in ("per_frame", "per_point", "per_frame_init"):
                                preset.waves[wave_idx].setdefault("per_frame_init", [])
                                preset.waves[wave_idx][param_name].append(loop_expr)
                            else:
                                preset.waves[wave_idx][param_name] = loop_expr
                    elif key.startswith(("shape_", "shapecode_")):
                        m = SHAPE_RE.match(key)
                        if m:
                            shape_idx, param_name = int(m.group(1)), m.group(2)
                            preset.shapes.setdefault(
                                shape_idx,
                                {
                                    "enabled": True,
                                    "per_frame": [],
                                    "per_frame_init": [],
                                },
                            )
                            # Normalize numbered equation names.
                            if param_name.startswith("per_frame_init") or param_name == "init" or (param_name.startswith("init") and param_name[4:].isdigit()):
                                param_name = "per_frame_init"
                            elif param_name.startswith("per_frame"):
                                param_name = "per_frame"
                            if param_name in ("per_frame", "per_frame_init"):
                                preset.shapes[shape_idx].setdefault("per_frame_init", [])
                                preset.shapes[shape_idx][param_name].append(loop_expr)
                            else:
                                preset.shapes[shape_idx][param_name] = loop_expr
                    continue

                # Join continuation lines
                joined_val = _strip_inline_comments(val)
                is_shader_param = key.startswith(("comp_", "warp_"))
                if not joined_val:
                    # The whole value was an inline comment (``per_pixel_3=//foo``).
                    # Nothing is left to continue, and treating the empty string
                    # as "incomplete" would make the joiner below swallow every
                    # following assignment -- dragging whole ``warp_*``/``comp_*``
                    # GLSL bodies into this equation bucket.
                    continue
                if re.fullmatch(r"[A-Za-z_]\w*\s*=", joined_val):
                    # A nested assignment header left with no right-hand side,
                    # e.g. the malformed
                    # ``per_frame_init_9=per_frame_1=//Thanks`` in a few presets.
                    # There is no expression to compile or continue from.
                    continue
                while i < len(all_lines) and (
                    _is_incomplete_expr(joined_val)
                    or _next_line_is_continuation(all_lines[i], joined_val)
                ):
                    # Each iteration re-tests the whole accumulated string, so
                    # an unterminated equation (unbalanced paren, ``loop(2,``
                    # with no close) that swallows every following same-bucket
                    # line is quadratic in the number of lines. The cap keeps
                    # that bounded; such an expression can never compile anyway.
                    if len(joined_val) > MAX_CONTINUATION_LENGTH:
                        break
                    next_line = all_lines[i].strip()
                    if not next_line or next_line.startswith(("//", "[")):
                        break
                    # A line made only of closing punctuation belongs to the
                    # enclosing construct, not to this expression.  A line
                    # that merely *ends* with ");" is the continuation of the
                    # expression (e.g. "x = if (a,0, if (b,1, 2));" split
                    # across lines) and must be joined, not abandoned.
                    # The exception is a bare closer while this expression
                    # still has unclosed parens of its own: there it *is* the
                    # continuation, and abandoning it truncated expressions
                    # such as "... + gmegabuf(index2 + 4)\n)\n* above(..."
                    # mid-argument.
                    if (
                        next_line
                        and not (set(next_line) - set(");,"))
                        and _paren_depth(joined_val) <= 0
                    ):
                        break
                    if not ASSIGN_RE.match(next_line):
                        cleaned = _strip_inline_comments(next_line)
                        if cleaned:
                            joined_val = joined_val + " " + cleaned
                        i += 1
                        continue
                    if is_shader_param:
                        break
                    _nkey, nval = next_line.split("=", 1)
                    _nkey = _nkey.strip()
                    # An equation continuation can never legitimately run into a
                    # different *bucket*: a ``warp_*``/``comp_*`` key holds a GLSL
                    # body, not a continuation of a ``per_frame_N`` equation.
                    # (The numeric suffix may differ -- wrapped equations are
                    # written as ``per_frame_1``/``per_frame_2``/``per_frame_3``
                    # -- so compare the key with its trailing index removed.)
                    if _eq_bucket_of(_nkey) != _eq_bucket_of(key):
                        break
                    # Keep the statement terminator: a joined line can carry a
                    # second statement ("a = 1; b = 2") and _split_milk_eqs
                    # needs the ";" to find the boundary again.  Only the
                    # inline comment is stripped here.
                    nval = _strip_inline_comments(nval.strip())
                    if not nval:
                        i += 1
                        continue
                    joined_val = joined_val + " " + nval
                    i += 1

                eqs = _split_milk_eqs(joined_val)
                for eq in eqs:
                    eq = eq.strip()
                    if not eq or eq.startswith("//"):
                        continue

                    if not ASSIGN_RE.match(eq) and not key.startswith(
                        ("per_frame", "per_pixel", "per_vertex", "wave", "shape")
                    ):
                        try:
                            if isinstance(eq, str) and (
                                eq.lstrip().startswith("`") or ASSIGN_RE.match(eq)
                            ):
                                preset.params[key] = eq
                            else:
                                preset.params[key] = float(eq)
                        except (ValueError, TypeError):
                            preset.params[key] = eq
                        continue
                    if key.startswith("per_frame_init"):
                        preset.per_frame_init.append(eq)
                    elif key.startswith("per_frame"):
                        preset.per_frame.append(eq)
                    elif key.startswith("per_pixel"):
                        preset.per_pixel.append(eq)
                    elif key.startswith("per_vertex"):
                        preset.per_vertex.append(eq)
                    elif key.startswith(("wavecode_", "wave_")):
                        m = WAVE_RE.match(key)
                        if m:
                            wave_idx, param_name = int(m.group(1)), m.group(2)
                            preset.waves.setdefault(
                                wave_idx,
                                {"enabled": True, "per_frame": [], "per_point": [], "per_frame_init": []},
                            )
                            # Normalize numbered equation names.
                            if param_name.startswith("per_point"):
                                param_name = "per_point"
                            elif param_name.startswith("per_frame_init") or param_name == "init" or (param_name.startswith("init") and param_name[4:].isdigit()):
                                param_name = "per_frame_init"
                            elif param_name.startswith("per_frame"):
                                param_name = "per_frame"
                            if param_name in ("per_frame", "per_point", "per_frame_init"):
                                preset.waves[wave_idx].setdefault("per_frame_init", [])
                                preset.waves[wave_idx][param_name].append(eq)
                            else:
                                preset.waves[wave_idx][param_name] = eq
                        elif key in {"wave_r", "wave_g", "wave_b", "wave_x", "wave_y"}:
                            try:
                                preset.params[key] = float(eq)
                            except ValueError:
                                preset.params[key] = eq
                    elif key.startswith(("shape_", "shapecode_")):
                        m = SHAPE_RE.match(key)
                        if m:
                            shape_idx, param_name = int(m.group(1)), m.group(2)
                            preset.shapes.setdefault(
                                shape_idx,
                                {
                                    "enabled": True,
                                    "per_frame": [],
                                    "per_frame_init": [],
                                },
                            )
                            # Normalize numbered equation names.
                            if param_name.startswith("per_frame_init") or param_name == "init" or (param_name.startswith("init") and param_name[4:].isdigit()):
                                param_name = "per_frame_init"
                            elif param_name.startswith("per_frame"):
                                param_name = "per_frame"
                            if param_name in ("per_frame", "per_frame_init"):
                                preset.shapes[shape_idx].setdefault("per_frame_init", [])
                                preset.shapes[shape_idx][param_name].append(eq)
                            else:
                                preset.shapes[shape_idx][param_name] = eq
                    else:
                        if eq.startswith(("`", "'")):
                            preset.params[key] = val
                        elif ASSIGN_RE.match(eq):
                            preset.params[key] = eq
                        else:
                            try:
                                preset.params[key] = float(eq)
                            except ValueError:
                                preset.params[key] = eq
            except Exception as e:  # noqa: BLE001
                if isinstance(e, ValueError):
                    category = "recoverable_formatting"
                elif isinstance(e, SyntaxError):
                    category = "invalid_equation"
                elif isinstance(e, (AttributeError, TypeError, IndexError)):
                    category = "parser_bug"
                else:
                    category = "unsupported_syntax"
                preset.parse_errors.append(
                    {"line": i, "error": str(e), "category": category}
                )
                logger.debug(
                    "Preset parse error line %d [%s]: %s",
                    i,
                    category,
                    e,
                )
                continue
        return preset
    except Exception as e:  # noqa: BLE001
        logger.error(f"[Preset Parse Error] Failed to parse preset file: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        return preset


parse_milk_file = parse_milk_file_robust
