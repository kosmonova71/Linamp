from __future__ import annotations

import logging
import math
import re

import numpy as np

logger = logging.getLogger("milkdrop_renderer")

# GLSL pattern regexes (extracted from gl_state during refactoring)
_ALIAS_DECL_RE = re.compile(r'^\s*alias\s+')
_GLSL_RESERVED_RENAME_RE = re.compile(r'^\s*[a-zA-Z_][a-zA-Z0-9_]*\s+float\s*')
_GLSL_RESERVED_RENAMES = {
    "gl_FragColor", "gl_FragData", "gl_FragDepth", "gl_Position",
    "gl_VertexID", "gl_InstanceID", "gl_Normal", "gl_FrontFacing",
}
_PIXEL_CODE_IDENT_RE = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')
_PIXEL_CODE_KNOWN_IDENTS = {"gl_FragColor", "gl_FragData"}
_PIXEL_LOCALS_DECLARED = set()

# GLSL local register aliases injected into the main visual fragment shader.
# q1..q32 alias u_q[1..32] (u_q has 33 elements, index 0 unused), t1..t8 alias
# u_t[1..8] (u_t has 9 elements), reg00..reg99 alias u_regs[0..99].
_Q_LOCALS = "\n".join(f"float q{i} = u_q[{i}];" for i in range(1, 33))
_T_LOCALS = "\n".join(f"float t{i} = u_t[{i}];" for i in range(1, 9))
_REG_LOCALS = "\n".join(f"float reg{i:02d} = u_regs[{i}];" for i in range(100))

_DEFAULT_SCALARS = {
    "zoom": 1.0, "rot": 0.0, "warp": 0.0,
    "cx": 0.5, "cy": 0.5, "dx": 0.0, "dy": 0.0,
    "sx": 1.0, "sy": 1.0, "decay": 0.9,
    "wave_r": 0.0, "wave_g": 0.0, "wave_b": 0.0, "wave_a": 0.0,
    "wave_x": 0.5, "wave_y": 0.5,
    "bass": 0.0, "mid": 0.0, "treb": 0.0,
    "bass_att": 0.0, "mid_att": 0.0, "treb_att": 0.0,
    "beat": 0.0, "beat_flash": 0.0, "rms": 0.0, "peak": 0.0,
    "time": 0.0, "frame": 0.0, "fps": 60.0, "dt": 0.0,
    "x": 0.0, "y": 0.0, "rad": 0.0, "ang": 0.0,
    "width": 1.0, "height": 1.0, "aspect": 1.0, "w": 1.0, "h": 1.0,
    "blend": 0.0,
    "echo_zoom": 1.0, "echo_alpha": 0.0, "echo_orient": 0,
}

# Track valid scalar uniform names for init validation
_VALID_SCALAR_NAMES = frozenset(_DEFAULT_SCALARS.keys())

def _as_finite_float(value: object, name: str) -> float:
    """Coerce a value to a finite float, rejecting NaN, infinity, strings, and None.

    Raises:
        TypeError: If value is a string or None.
        ValueError: If value is NaN or infinity.
    """
    if isinstance(value, str) or value is None:
        raise TypeError(f"Uniform '{name}' cannot be assigned a {type(value).__name__} value")
    v = float(value)  # type: ignore[reportArgumentType]
    if math.isnan(v) or math.isinf(v):
        raise ValueError(f"Uniform '{name}' cannot be assigned NaN or infinity")
    return v


class GPUUniforms:
    """Mutable GPU uniform state container with dirty-field tracking.

    This is deliberately a plain mutable state object, not a dataclass.
    Fields are assigned every frame and dirty tracking avoids redundant
    uniform uploads.

    Call :meth:`clear_changed` after uploading uniforms to the GPU to
    reset the dirty-tracking sets so that only subsequently-mutated
    fields are returned by :meth:`get_changed` on the next frame.

    Typical frame usage::

        uniforms.time = t
        uniforms.bass = b
        uniforms.q = q_vals          # or set_q / set_q_bulk
        changed = uniforms.get_changed()
        # upload only the keys in `changed`
        uniforms.clear_changed()

    q / t registers are tracked per index: :meth:`set_q` / :meth:`set_t`
    and bulk q / t assignments record exactly which indices changed.
    :meth:`get_changed` reports only the actually-written q / t prefix,
    so stale registers are not re-uploaded every frame.
    """

    _MISSING = object()

    # Dirty tracking attributes (set dynamically in __init__/methods)
    _force_full: bool
    _changed_fields: set[str]
    _changed_q: set[int]
    _changed_t: set[int]
    _q: list[float]
    _t: list[float]
    _cached_full: dict | None  # Cached to_dict() snapshot; invalidated on any mutation

    zoom: float = 1.0
    rot: float = 0.0
    warp: float = 0.0
    cx: float = 0.5
    cy: float = 0.5
    dx: float = 0.0
    dy: float = 0.0
    sx: float = 1.0
    sy: float = 1.0
    decay: float = 0.9
    wave_r: float = 0.0
    wave_g: float = 0.0
    wave_b: float = 0.0
    wave_a: float = 0.0
    wave_x: float = 0.5
    wave_y: float = 0.5
    bass: float = 0.0
    mid: float = 0.0
    treb: float = 0.0
    bass_att: float = 0.0
    mid_att: float = 0.0
    treb_att: float = 0.0
    beat: float = 0.0
    beat_flash: float = 0.0
    rms: float = 0.0
    peak: float = 0.0
    time: float = 0.0
    frame: float = 0.0
    fps: float = 60.0
    dt: float = 0.0
    x: float = 0.0
    y: float = 0.0
    rad: float = 0.0
    ang: float = 0.0
    width: float = 1.0
    height: float = 1.0
    aspect: float = 1.0
    w: float = 1.0
    h: float = 1.0
    blend: float = 0.0
    echo_zoom: float = 1.0
    echo_alpha: float = 0.0
    echo_orient: int = 0

    def __init__(self, **kwargs):
        object.__setattr__(self, "_force_full", True)
        object.__setattr__(self, "_changed_fields", set())
        object.__setattr__(self, "_changed_q", set())
        object.__setattr__(self, "_changed_t", set())
        object.__setattr__(self, "_q", [0.0] * 32)
        object.__setattr__(self, "_t", [0.0] * 8)
        object.__setattr__(self, "_cached_full", None)

        for name, value in kwargs.items():
            setattr(self, name, value)

    def reset(self) -> None:
        """Restore all uniforms to defaults and clear dirty tracking."""
        for name, default in _DEFAULT_SCALARS.items():
            setattr(self, name, default)
        object.__setattr__(self, "_changed_fields", set())
        self._changed_q.clear()
        self._changed_t.clear()
        object.__setattr__(self, "_q", [0.0] * 32)
        object.__setattr__(self, "_t", [0.0] * 8)
        object.__setattr__(self, "_force_full", True)
        object.__setattr__(self, "_cached_full", None)

    def _apply_qt(self, name: str, value) -> None:
        """Apply q or t values with shape validation.

        Rejects multidimensional arrays. Only 1D lists/tuples/arrays are accepted.
        Values exceeding the register count (32 for q, 8 for t) are truncated.
        """
        arr = np.asarray(value, dtype=float)

        # Validate shape: must be 1-dimensional
        if arr.ndim != 1:
            raise ValueError(
                f"{name} must be a 1-dimensional array, got {arr.ndim}-dimensional "
                f"array with shape {arr.shape}"
            )

        # Truncate or zero-pad to the correct size
        if name == "q":
            target_size = 32
        else:  # t
            target_size = 8

        # Convert to list and handle sizing
        values = arr.tolist()

        if len(values) > target_size:
            # Truncate: keep only the first `target_size` values
            values = values[:target_size]
        elif len(values) < target_size:
            # Zero-pad remaining entries
            values.extend([0.0] * (target_size - len(values)))

        # Assign the values
        target = self._q if name == "q" else self._t
        for i, v in enumerate(values):
            target[i] = v

        # Mark the changed indices
        changed_indices = set(range(min(len(arr), target_size)))
        if name == "q":
            self._changed_q.update(changed_indices)
        else:
            self._changed_t.update(changed_indices)

        object.__setattr__(self, "_cached_full", None)

    def _set_named_uniform(self, name: str, value: object) -> None:
        """Set a named uniform scalar with validation and dirty tracking.

        Rejects None and strings explicitly, coerces other numerics to float
        early, and marks the field changed only when the value actually
        differs from the current value.
        """
        value = _as_finite_float(value, name)

        # Special handling for echo_orient: mask to 2-bit value
        if name == "echo_orient":
            value = int(value) & 3

        old = getattr(self, name, self._MISSING)

        # Check if value is different from old - if same, don't mark dirty
        # But still assign the value to ensure type consistency
        object.__setattr__(self, name, value)

        # Mark as changed only if it's a tracked scalar and value actually changed
        if name in _VALID_SCALAR_NAMES and (old is self._MISSING or old != value):
            self._changed_fields.add(name)

    @property
    def q(self) -> tuple:
        return tuple(self._q)

    @q.setter
    def q(self, value) -> None:
        self._apply_qt("q", value)

    @property
    def t(self) -> tuple:
        return tuple(self._t)

    @t.setter
    def t(self, value) -> None:
        self._apply_qt("t", value)

    def __setattr__(self, name: str, value: object) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return

        if name in ("q", "t"):
            self._apply_qt(name, value)
            return

        if name not in _VALID_SCALAR_NAMES:
            # Allow future non-tracked attributes without exploding
            object.__setattr__(self, name, value)
            object.__setattr__(self, "_cached_full", None)
            return

        value = _as_finite_float(value, name)
        if name == "echo_orient":
            value = int(value) & 3

        old = getattr(self, name, self._MISSING)
        object.__setattr__(self, name, value)

        if old is self._MISSING or old != value:
            self._changed_fields.add(name)
        object.__setattr__(self, "_cached_full", None)

    def set_q(self, index: int, value: float) -> None:
        if not 0 <= index < 32:
            raise IndexError(f"q index {index} out of range [0, 32)")
        v = _as_finite_float(value, f"q{index + 1}")
        self._q[index] = v
        self._changed_q.add(index)
        object.__setattr__(self, "_cached_full", None)

    def set_t(self, index: int, value: float) -> None:
        if not 0 <= index < 8:
            raise IndexError(f"t index {index} out of range [0, 8)")
        v = _as_finite_float(value, f"t{index + 1}")
        self._t[index] = v
        self._changed_t.add(index)
        object.__setattr__(self, "_cached_full", None)

    def set_q_bulk(self, values) -> None:
        """Set all q registers from a sequence, with index-level tracking.

        Accepts a list, tuple, or numpy array of float values.
        Values are truncated or zero-padded to exactly 32 entries.
        All 32 indices are marked as changed for efficient upload.
        """
        arr = np.asarray(values, dtype=float)
        if arr.ndim != 1:
            raise ValueError(f"q values must be 1-dimensional, got {arr.ndim}-dimensional array")
        if arr.size > 32:
            arr = arr[:32]
        elif arr.size < 32:
            padded = np.zeros(32, dtype=float)
            padded[: arr.size] = arr
            arr = padded
        for i in range(32):
            self._q[i] = float(arr[i])
        # Mark all 32 indices as changed
        self._changed_q.update(range(32))
        object.__setattr__(self, "_cached_full", None)

    def set_t_bulk(self, values) -> None:
        """Set all t registers from a sequence, with index-level tracking.

        Accepts a list, tuple, or numpy array of float values.
        Values are truncated or zero-padded to exactly 8 entries.
        All 8 indices are marked as changed for efficient upload.
        """
        arr = np.asarray(values, dtype=float)
        if arr.ndim != 1:
            raise ValueError(f"t values must be 1-dimensional, got {arr.ndim}-dimensional array")
        if arr.size > 8:
            arr = arr[:8]
        elif arr.size < 8:
            padded = np.zeros(8, dtype=float)
            padded[: arr.size] = arr
            arr = padded
        for i in range(8):
            self._t[i] = float(arr[i])
        # Mark all 8 indices as changed
        self._changed_t.update(range(8))
        object.__setattr__(self, "_cached_full", None)


    def to_dict(self) -> dict:
        cached = self._cached_full
        if cached is not None:
            return cached

        d = {name: getattr(self, name) for name in _VALID_SCALAR_NAMES}
        d.update((f"q{i + 1}", self._q[i]) for i in range(32))
        d.update((f"t{i + 1}", self._t[i]) for i in range(8))
        object.__setattr__(self, "_cached_full", d)
        return d

    def get_changed(self) -> dict:
        if not self.is_dirty:
            return {}

        if self._force_full:
            return self.to_dict()

        # Early exit if nothing changed
        if not self._changed_fields and not self._changed_q and not self._changed_t:
            return {}

        result = {name: getattr(self, name) for name in self._changed_fields}
        result.update((f"q{i + 1}", self._q[i]) for i in self._changed_q)
        result.update((f"t{i + 1}", self._t[i]) for i in self._changed_t)

        return result

    def clear_changed(self) -> None:
        object.__setattr__(self, "_changed_fields", set())
        self._changed_q.clear()
        self._changed_t.clear()
        object.__setattr__(self, "_q_len", 0)
        object.__setattr__(self, "_t_len", 0)
        object.__setattr__(self, "_force_full", False)
        object.__setattr__(self, "_cached_full", None)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("GPUUniforms dirty tracking cleared")

    def update(self, **kwargs) -> None:
        """Convenience bulk assignment; every kwarg goes through setattr so
        validation and dirty tracking are applied consistently."""
        for k, v in kwargs.items():
            setattr(self, k, v)

    def set_matrix(self, name: str, matrix: list[list[float]] | np.ndarray) -> None:
        """Set a 4x4 matrix uniform with validation and dirty tracking.

        Args:
            name: The uniform variable name.
            matrix: A 4x4 matrix as a list of lists or numpy array, stored in
                  column-major order (OpenGL convention).
        """
        arr = np.asarray(matrix, dtype=float)
        if arr.shape != (4, 4):
            raise ValueError(f"Matrix must be 4x4, got shape {arr.shape}")
        # Flatten in column-major order for GLSL
        flat = arr.flatten(order="F").tolist()
        if name in _VALID_SCALAR_NAMES:
            # For tracked scalars, only accept the first component
            object.__setattr__(self, name, flat[0])
        else:
            object.__setattr__(self, name, flat)
        object.__setattr__(self, "_cached_full", None)

    @property
    def is_dirty(self) -> bool:
        """Return True if any uniform has been modified and hasn't been cleared."""
        return self._force_full or bool(
            self._changed_fields or
            self._changed_q or
            self._changed_t
        )

    def __repr__(self) -> str:
        return (
            f"GPUUniforms(dirty={self._force_full}, "
            f"fields_changed={len(self._changed_fields)}, "
            f"q_changed={len(self._changed_q)}, "
            f"t_changed={len(self._changed_t)})"
        )
