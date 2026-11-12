from __future__ import annotations

import logging
import os
from typing import Protocol, runtime_checkable

import numpy as np

from resources import (
    T_LOOKUP,
    VAR_REGISTRY,
    VARIABLES,
)

from .config import (
    _VAR_KIND_ARRAY,
    _VAR_KIND_DICT,
    _VAR_KIND_Q,
    _VAR_KIND_REG,
    _VAR_KIND_T,
)

logger = logging.getLogger("milkdrop_renderer")


# --- Register mapping and state adapters -------------------------
def _reg_var_index(name: str) -> int | None:
    """Return the ``reg00``-``reg99`` index of *name*, or ``None``.

    Mirrors the lookup performed by :meth:`PresetState.get`/:meth:`PresetState.set`:
    the name must be exactly ``reg`` followed by two digits.
    """
    if len(name) != 5 or not name.startswith("reg"):
        return None
    if not name[3:].isdigit():
        return None
    idx = int(name[3:])
    if 0 <= idx < 100:
        return idx
    return None


class _StateVarsProxy:
    """Adapter for states that do not expose a ``vars`` dictionary.

    The storage-specialised opcodes address ``state.vars`` directly.  A
    duck-typed state (a test double, or a caller with its own state object)
    may only implement ``get``/``set``; this adapter keeps the specialised
    opcodes working for those by forwarding to the same accessors the
    generic opcodes use.
    """

    __slots__ = ("_state",)

    def __init__(self, state):
        self._state = state

    def get(self, name, default=0.0):
        return self._state.get(name, default)

    def __getitem__(self, name):
        return self._state.get(name, 0.0)

    def __setitem__(self, name, value):
        self._state.set(name, value)


class _StateGlobalRegsProxy:
    """Adapter mirroring :class:`_StateVarsProxy` for the global registers."""

    __slots__ = ("_state",)

    def __init__(self, state):
        self._state = state

    def __getitem__(self, idx):
        return self._state.get(f"reg{idx:02d}", 0.0)

    def __setitem__(self, idx, value):
        self._state.set(f"reg{idx:02d}", value)


# ---------------------------------------------------------------------------
# Duck-typed state protocol
# ---------------------------------------------------------------------------
@runtime_checkable
class StateLike(Protocol):
    """Minimal interface required by the bytecode VM for evaluation.

    ``PresetState`` is the canonical implementation; minimal duck-typed
    state objects (``_MiniState`` in tests, or a caller with its own
    state object) need only provide the attributes and methods below.

    The optional ``vars`` and ``_global_regs`` attributes are accessed
    via ``getattr`` in :meth:`BytecodeProgram.execute` with adapter-proxy
    fallbacks (:class:`_StateVarsProxy`, :class:`_StateGlobalRegsProxy`),
    so they are **not** required members of this protocol — a state that
    only implements ``get``/``set`` is sufficient for expressions that do
    not touch megabuf or the ``rand`` function.

    Similarly, ``megabuf_get``/``megabuf_set`` and ``rand_seed`` are only
    needed when the executed bytecode actually references those features;
    the VM resolves them through ``getattr`` so their absence does not
    cause failures for simpler expressions.
    """

    vars_array: np.ndarray
    q: list[float]
    t_vars: list[float]

    def get(self, name: str, default: float = 0.0) -> float: ...
    def set(self, name: str, value: float) -> None: ...


# --- Preset variable state ----------------------------------------
class PresetState:
    """Stateful container for all preset variables.
    Maintains persistent state across frames including:
    - All variables (zoom, rot, warp, etc.)
    - Previous frame's variables (for temporal calculations)
    - q1-q32 registers (bridged from preset to GPU)
    - t1-t8 bridge variables (shared with custom waves/shapes)
    - megabuf (1M float array)
    - reg00-reg99 global registers (shared across presets)
    - Random seed (for deterministic rand())
    """

    MEGABUF_SIZE_MAX = 1048576
    MEGABUF_SIZE = min(
        int(os.getenv("MILKDROP_MEGABUF_SIZE", "262144")), MEGABUF_SIZE_MAX
    )
    _CLAMP_ZOOM = os.getenv("MILKDROP_CLAMP_ZOOM", "0") == "1"
    STATE_ATTRS = frozenset(VARIABLES.keys())
    _SHAPE_STATE_ATTRS = frozenset(
        set(VARIABLES.keys()) | {"rand_seed", "megabuf_version"}
    )
    # Instance attributes that exist on a preset state.  ``PresetState.set``
    # mirrors a write into the matching attribute, so the bytecode compiler
    # keeps routing these names through the generic opcodes instead of the
    # storage-specialised fast path.  ``tests/test_preset_state_variable_storage.py``
    # keeps this list in sync with the live instance.
    ATTRIBUTE_NAMES = frozenset(
        {
            "zoom",
            "rot",
            "warp",
            "cx",
            "cy",
            "dx",
            "dy",
            "sx",
            "sy",
            "decay",
            "vars",
            "vars_array",
            "prev_array",
            "prev_vars",
            "q",
            "t_vars",
            "megabuf",
            "megabuf_version",
            "rand_seed",
        }
    )

    def __init__(self):
        self.zoom = 1.0
        self.rot = 0.0
        self.warp = 0.0
        self.cx = 0.5
        self.cy = 0.5
        self.dx = 0.0
        self.dy = 0.0
        self.sx = 1.0
        self.sy = 1.0
        self.decay = 0.9
        self.vars: dict[str, float] = {}
        self.vars_array = np.zeros(1024, dtype=np.float32)
        self.prev_array = np.zeros(1024, dtype=np.float32)
        self.prev_vars = {}
        self.q = [0.0] * 32
        self.t_vars = [0.0] * 8
        self.megabuf = None
        self.megabuf_version = 0
        self.rand_seed = 1
        # Global registers (reg00-reg99) - now instance-owned so multiple
        # MilkdropRenderer instances don't share registers.
        self._global_regs = np.zeros(100, dtype=np.float32)
        # Memoised name -> storage resolution (see _resolve_var).  Shared with
        # clones: the mapping is a pure function of the name.
        self._var_res: dict[str, tuple[int, int, str, int]] = {}
        # Names that exist as instance attributes.  ``PresetState.set`` mirrors
        # a write into the attribute when the attribute holds a number, so the
        # fast path has to know which names qualify.  The attribute set is
        # fixed once the instance is built, hence it is captured here.
        self._attr_names = PresetState.ATTRIBUTE_NAMES
        for name, idx in VAR_REGISTRY.items():
            self.vars_array[idx] = float(getattr(self, name, 0.0))
        self._init_base_vars()

    def _init_base_vars(self):
        """Initialize base MilkDrop variables with defaults from VARIABLES registry."""
        for vname, vdef in VARIABLES.items():
            self.set(vname, vdef.default)

    def begin_frame(self):
        """Called at start of each frame to save previous state."""
        self.prev_vars = self.vars.copy()
        self.prev_array[:] = self.vars_array

    def _resolve_var(self, name: str) -> tuple[int, int, str, int]:
        """Resolve *name* to its storage location and cache the result.

        The lookup order mirrors :meth:`get`/:meth:`set` exactly:
        ``reg00``-``reg99``, ``q1``-``q32``, ``t1``-``t8``, the fixed
        ``VARIABLES`` table and finally the free-form ``vars`` dictionary.

        Returns a ``(kind, index, lower_name, extra)`` tuple.  ``extra`` is the
        ``vars_array`` index for ``t`` variables (``t1``-``t8`` are part of the
        VARIABLES table, so writes to them are mirrored there as well) and -1
        otherwise.
        """
        name_lower = name.lower()
        kind = _VAR_KIND_DICT
        index = 0
        extra = -1
        if (
            len(name_lower) == 5
            and name_lower.startswith("reg")
            and name_lower[3:].isdigit()
        ):
            reg_idx = int(name_lower[3:])
            if 0 <= reg_idx < 100:
                kind = _VAR_KIND_REG
                index = reg_idx
        if kind == _VAR_KIND_DICT and (
            len(name_lower) <= 3 and name_lower[0] == "q" and name_lower[1:].isdigit()
        ):
            q_idx = int(name_lower[1:]) - 1
            if 0 <= q_idx < 32:
                kind = _VAR_KIND_Q
                index = q_idx
        if kind == _VAR_KIND_DICT:
            t_idx = T_LOOKUP.get(name_lower)
            if t_idx is not None:
                kind = _VAR_KIND_T
                index = t_idx
                extra = VAR_REGISTRY.get(name_lower, -1)
            else:
                array_idx = VAR_REGISTRY.get(name_lower)
                if array_idx is not None:
                    kind = _VAR_KIND_ARRAY
                    index = array_idx
        res = (kind, index, name_lower, extra)
        self._var_res[name] = res
        return res

    def get(self, name: str, default: float = 0.0) -> float:
        """Get variable value, supporting q1-q32, t1-t8 syntax and reg00-reg99 global registers."""
        res = self._var_res.get(name)
        if res is None:
            res = self._resolve_var(name)
        kind, index, lower_name, _extra = res  # unpack once for faster access
        if kind == _VAR_KIND_DICT:
            return float(self.vars.get(lower_name, default))
        if kind == _VAR_KIND_ARRAY:
            return float(self.vars_array[index])
        if kind == _VAR_KIND_Q:
            return float(self.q[index])
        if kind == _VAR_KIND_T:
            return float(self.t_vars[index])
        return float(self._global_regs[index])

    def set(self, name: str, value: float):
        """Set variable value, supporting q1-q32 syntax and reg00-reg99 global registers."""
        value = float(value)
        res = self._var_res.get(name)
        if res is None:
            res = self._resolve_var(name)
        kind, index, lower_name, _extra = res  # unpack once for faster access
        if kind == _VAR_KIND_REG:
            self._global_regs[index] = value
            return
        if lower_name == "zoom" and PresetState._CLAMP_ZOOM:
            value = max(value, 0.001)
        if kind == _VAR_KIND_ARRAY:
            self.vars_array[index] = float(value)
        elif kind == _VAR_KIND_T and _extra >= 0:
            self.vars_array[_extra] = float(value)
        if lower_name in self._attr_names:
            val = getattr(self, lower_name, None)
            if isinstance(val, (int, float, np.number)):
                setattr(self, lower_name, float(value))
        if kind == _VAR_KIND_Q:
            self.q[index] = value
            return
        if kind == _VAR_KIND_T:
            self.t_vars[index] = value
            return
        self.vars[lower_name] = value

    def _ensure_megabuf(self):
        """Lazy allocate megabuf on first access."""
        if self.megabuf is None:
            tex_size = getattr(self, "MEGABUF_SIZE", 262144)
            self.megabuf = np.zeros(tex_size, dtype=np.float32)

    def megabuf_get(self, idx: float) -> float:
        """Get value from megabuf at index."""
        self._ensure_megabuf()
        i = int(idx)
        tex_size = getattr(self, "megabuf_tex_size", self.MEGABUF_SIZE)
        megabuf = self.megabuf
        if megabuf is None:
            return 0.0
        if 0 <= i < tex_size:
            return float(megabuf[i])
        return 0.0

    def megabuf_set(self, idx: float, value: float):
        """Set value in megabuf at index."""
        self._ensure_megabuf()
        i = int(idx)
        tex_size = getattr(self, "megabuf_tex_size", self.MEGABUF_SIZE)
        megabuf = self.megabuf
        if megabuf is None:
            return
        if 0 <= i < tex_size:
            value = float(value)
            megabuf[i] = float(value)
            self.megabuf_version = (self.megabuf_version + 1) & 0xFFFFFFFF

    def update_audio(self, audio_levels):
        """Update audio-related variables from external source.

        ``rms`` and ``peak`` are forwarded so advanced presets and the
        enhancement layer can access overall loudness information that is
        not captured by the three-band (bass/mid/treb) model alone.
        """
        bass = audio_levels.get("bass", 0.0)
        bass_att = audio_levels.get("bass_att", 0.0)
        mid = audio_levels.get("mid", 0.0)
        mid_att = audio_levels.get("mid_att", 0.0)
        treb = audio_levels.get("treb", 0.0)
        treb_att = audio_levels.get("treb_att", 0.0)
        self.set("bass", bass)
        self.set("bass_att", bass_att)
        self.set("mid", mid)
        self.set("mid_att", mid_att)
        self.set("treb", treb)
        self.set("treb_att", treb_att)
        # MilkDrop presets commonly read vol/vol_att (overall loudness).
        vol = audio_levels.get("vol")
        if vol is None:
            vol = (float(bass or 0.0) + float(mid or 0.0) + float(treb or 0.0)) / 3.0
        vol_att = audio_levels.get("vol_att")
        if vol_att is None:
            vol_att = (float(bass_att or 0.0) + float(mid_att or 0.0) + float(treb_att or 0.0)) / 3.0
        self.set("vol", vol)
        self.set("vol_att", vol_att)
        self.set("beat", audio_levels.get("beat", 0.0))
        self.set("beat_flash", audio_levels.get("beat_flash", 0.0))
        self.set("rms", audio_levels.get("rms", 0.0))
        self.set("peak", audio_levels.get("peak", 0.0))

    def update_time(
        self, time: float, frame: int, fps: float = 60.0, dt: float | None = None
    ):
        """Update time-related variables.

        ``dt`` is the per-frame delta-time (seconds since last frame), as
        produced by :meth:`MilkdropRenderer._advance_frame_clock`.  When
        provided (the normal render path), it is stored under the ``dt``
        MilkDrop variable so presets can perform frame-rate-independent
        integration (``dec = 1 - 6*dt``, ``tt = tt + dt*0.1``, physics, etc.).
        When ``None`` (legacy callers / tests), ``dt`` is left at its
        previous value so it is never silently zeroed mid-animation.

        ``fps`` is recomputed from ``dt`` when available so presets that use
        ``fps`` (e.g. ``dec_s = 1 - 0.015*30/fps``) reflect the real frame
        rate rather than a hard-coded 60.
        """
        self.set("time", time)
        self.set("t", time)
        self.set("frame", frame)
        if dt is not None and dt > 0.0:
            self.set("dt", dt)
            self.set("fps", 1.0 / dt)
        else:
            self.set("fps", fps)

    def update_resolution(self, width: float, height: float):
        """Update resolution-related variables."""
        self.set("width", width)
        self.set("height", height)
        self.set("w", width)
        self.set("h", height)
        self.set("aspect", width / height if height > 0 else 1.0)
        self.set("aspectx", width / height if width < height else 1.0)
        self.set("aspecty", height / width if height < width else 1.0)

    def _copy_core_state_to(self, clone: PresetState):
        clone.zoom = self.zoom
        clone.rot = self.rot
        clone.warp = self.warp
        clone.cx = self.cx
        clone.cy = self.cy
        clone.dx = self.dx
        clone.dy = self.dy
        clone.sx = self.sx
        clone.sy = self.sy
        clone.decay = self.decay
        clone.rand_seed = self.rand_seed
        clone.megabuf_version = self.megabuf_version
        clone.vars = self.vars.copy()
        clone.vars_array = self.vars_array.copy()
        clone.q = list(self.q)
        clone.t_vars = list(self.t_vars)
        clone.megabuf = self.megabuf
        clone._global_regs = self._global_regs
        # The name -> storage resolution is a pure function of the name, so
        # the cache can be shared with the clone.
        clone._var_res = self._var_res
        clone._attr_names = self._attr_names

    def shallow_clone_for_shapes(self) -> PresetState:
        """Lightweight clone for shape execution.
        Copies only mutable fields that shapes modify:
        - vars_dict (shape can write custom variables)
        - vars_array (motion vars: zoom, rot, warp, etc.)
        - q registers
        - rand_seed (deterministic per-frame rand)
        Does NOT copy megabuf (shared reference to avoid 4MB/mem-bandwidth copy).
        Does NOT copy prev_vars/prev_array (not needed for shapes).
        """
        clone = PresetState.__new__(PresetState)
        self._copy_core_state_to(clone)
        clone.prev_vars = {}
        clone.prev_array = np.zeros(1024, dtype=np.float32)
        return clone

    def shallow_clone(self) -> PresetState:
        """Full shallow clone with separate state arrays."""
        clone = PresetState.__new__(PresetState)
        self._copy_core_state_to(clone)
        clone.prev_vars = self.prev_vars.copy()
        clone.prev_array = self.prev_array.copy()
        return clone
