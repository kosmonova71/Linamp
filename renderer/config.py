"""Shared constants, limits, regexes, shader sources, and equation helpers.

Ownership
---------
This module is the renderer package's central repository for:

* **Paths & environment** - project root and process-level configuration.
* **GL/rendering constants** - enums-as-ints, shader sources, uniform tables.
* **Budgets** - iteration/step limits that keep malformed presets from hanging
  or exhausting frame time.
* **Regexes** - precompiled patterns for preset parsing and equation analysis.
* **Variable kinds** - the :class:`VarKind` storage-class enum (with historic
  ``_VAR_KIND_*`` integer aliases preserved for compatibility).
* **Return-type tables** - builtin/function return-type mappings for the
  expression compiler.
* **Errors** - preset compilation and budget exceptions.
* **Crash log** - best-effort, size-rotated diagnostic logging.
* **Equation helpers** - small text utilities used by parsing and compilation.

Future extraction plan
----------------------
The four equation helpers (``_comma_splits_statement``, ``_split_milk_eqs``,
``_strip_inline_comments``, ``_is_metadata_or_comment_expression``) and their
supporting regexes (``_BRANCH_TARGET_RE``, ``_METADATA_LHS_RE``,
``LOOP_START_RE``) can be moved into ``equation_utils.py`` later.  To avoid
breaking current imports, ``renderer.config`` should continue to re-export
those names and ``renderer.__init__`` should keep forwarding them.

Logging policy
--------------
This module intentionally does **not** call ``logging.basicConfig()`` at import
time.  Configuring the root logger is too aggressive for a library-style
module; the application entry point is expected to configure logging once at
startup.  We only obtain the named logger used for renderer diagnostics.
"""

from __future__ import annotations

import keyword
import logging
import os
import re
import sys
import time
import types
from enum import IntEnum
from pathlib import Path
from typing import TypeVar

import gi
from OpenGL import GL

# ---------------------------------------------------------------------------
# Paths & Environment
# ---------------------------------------------------------------------------

try:
    gi.require_version("Gdk", "4.0")
    gi.require_version("Gtk", "4.0")
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst  # type: ignore[reportMissingModuleSource]

    Gst.init(None)
except ValueError as exc:
    sys.stderr.write(
        f"milkdrop_renderer: required GI versions unavailable "
        f"(Gdk 4.0, Gtk 4.0, Gst 1.0): {exc}\n"
    )
    sys.exit(1)

#: Monotonic clock alias captured at module level (``time`` is commonly
#: shadowed by local parameters in render/VM paths).
_monotonic = time.monotonic

logger = logging.getLogger("milkdrop_renderer")

#: Repository root.  The renderer used to live in the repository root and
#: located its assets with ``Path(__file__).parent``; inside the package that
#: path points at ``renderer/``, so the root is resolved explicitly.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

#: Shared free type variable used by generic helpers in other modules
#: (e.g. ``TextureLRU.get``); defined once so modules share a single TypeVar.
_T = TypeVar("_T")  # noqa: PYI018

#: Enable verbose VM debug output via ``MILKDROP_VM_DEBUG=1|true|yes``.
VM_DEBUG: bool = os.environ.get("MILKDROP_VM_DEBUG", "").lower() in (
    "1",
    "true",
    "yes",
)

# ---------------------------------------------------------------------------
# GL / Rendering
# ---------------------------------------------------------------------------

# PyOpenGL enums are union-typed; snapshot as plain ints for static analysis
# and for safe bitwise/offset arithmetic in render paths.
_GL_TEXTURE0: int = int(GL.GL_TEXTURE0)
_GL_COLOR_BUFFER_BIT: int = int(GL.GL_COLOR_BUFFER_BIT)
_GL_DEPTH_BUFFER_BIT: int = int(GL.GL_DEPTH_BUFFER_BIT)

#: Audio spectrum bin count consumed by waveform/spectrum paths.
SPECTRUM_BINS: int = 64

#: Raw waveform sample buffer size expected by the renderer.
WAVEFORM_BUFFER_SAMPLES: int = 576

#: Megabuf backing size (clamped to 1M entries) for preset ``megabuf``/``gmegabuf``.
MEGABUF_SIZE: int = min(
    int(os.getenv("MILKDROP_MEGABUF_SIZE", "262144")), 1_048_576
)

#: Maximum polygon side count for custom shapes (guards geometry explosion).
MAX_SHAPE_SIDES: int = 4096

#: Maximum number of audio-waveform samples drawn in a single wave line-strip.
#: Presets without per-point equations draw the live audio waveform; only the
#: slots actually populated by ``_populate_waveform_data`` should be drawn so
#: trailing zero-padded entries (which map to NDC (-1, -1)) do not spike the
#: line strip into the corner.
WAVE_MAX_SAMPLES: int = 512

#: Line width for classic waveform / per-point wave line strips.
#: Matches the MilkDrop default of 2 px used by the reference visualiser.
WAVE_LINE_WIDTH: float = 2.0

#: Opaque black clear colour used by render passes and presentation fallback.
#: Renderer policy, not preset semantics.
CLEAR_COLOR_BLACK: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

#: Consecutive render-failure threshold before requesting GL reinitialization.
#: Protects the app from tight failure loops when the GL context is unhealthy.
CONTEXT_HEALTH_FAILURE_THRESHOLD: int = 5

#: Fallback procedural shader frequencies (R/G/B) when no preset shader is active.
FALLBACK_SHADER_FREQ_R: float = 0.7
FALLBACK_SHADER_FREQ_G: float = 0.9
FALLBACK_SHADER_FREQ_B: float = 1.1

_SCREEN_BLIT_VERT_SRC: str = """
#version 330 core
layout(location = 0) in vec2 a_position;
layout(location = 1) in vec2 a_texcoord;
out vec2 v_uv;
void main() {
    v_uv = a_texcoord;
    gl_Position = vec4(a_position, 0.0, 1.0);
}
"""

_SCREEN_BLIT_FRAG_SRC: str = """
#version 330 core
in vec2 v_uv;
out vec4 frag_color;
uniform sampler2D u_texture;
void main() {
    frag_color = texture(u_texture, v_uv);
}
"""

_TRANSITION_BLEND_FRAG_SRC: str = """
#version 330 core
in vec2 v_uv;
out vec4 frag_color;
uniform sampler2D u_from_texture;
uniform sampler2D u_to_texture;
uniform float u_progress;  // 0.0 = from, 1.0 = to
void main() {
    vec4 from_color = texture(u_from_texture, v_uv);
    vec4 to_color = texture(u_to_texture, v_uv);
    float eased = smoothstep(0.0, 1.0, u_progress);
    frag_color = mix(from_color, to_color, eased);
}
"""

_WAVE_SHAPE_VERT_SRC: str = """
#version 330 core
layout(location = 0) in vec2 a_position;
layout(location = 1) in vec4 a_color;
uniform float u_point_size;
out vec4 v_color;
void main() {
    v_color = a_color;
    gl_PointSize = u_point_size;
    gl_Position = vec4(a_position, 0.0, 1.0);
}
"""

_WAVE_SHAPE_FRAG_SRC: str = """
#version 330 core
in vec4 v_color;
out vec4 frag_color;
void main() {
    frag_color = v_color;
}
"""

_COMPOSITE_UNIFORM_NAMES: tuple[str, ...] = (
    "u_original",
    "u_resolution",
    "u_time",
    "u_bloom_temporal_mix",
    "u_bloom_weight",
    "u_bloom_intensity",
    "u_ray_count",
    "u_light_pos",
    "u_ray_decay",
    "u_ray_step_weight",
    "u_chromatic_aberration",
    "u_chromatic_beat",
    # Audio-reactive enhancement uniforms
    "u_audio_bass",
    "u_audio_bass_att",
    "u_audio_mid",
    "u_audio_treb",
    "u_audio_beat",
    "u_audio_beat_flash",
    "u_audio_rms",
    "u_audio_peak",
    "u_brightness_boost",
    "u_vignette_strength",
    "u_vignette_smooth",
    "u_vignette_color",
    "u_lift",
    "u_gamma",
    "u_gain",
    "u_saturation",
    "u_contrast",
    "u_light_tint_strength",
    "u_audio_vignette_breath",
    "u_lens_distortion",
)

# ---------------------------------------------------------------------------
# Budgets
# ---------------------------------------------------------------------------

#: Maximum iterations for preset init ``loop``/``while`` blocks.
#: Protects startup from non-terminating or excessively expensive init code.
MAX_INIT_LOOP_ITERATIONS: int = 5000

#: Maximum iterations for per-frame ``loop``/``while`` blocks.
#: Prevents a single preset frame from stalling the render loop.
MAX_FRAME_LOOP_ITERATIONS: int = 1000

#: Maximum number of loop constructs allowed in one expression.
#: Limits parser/VM expansion from pathological nested/concatenated loops.
MAX_LOOPS_PER_EXPRESSION: int = 8

#: Hard bytecode step ceiling for a single VM execution context.
#: Guards against runaway bytecode even when higher-level budgets are misused.
MAX_BYTECODE_STEPS: int = 1 << 16

#: Maximum continuation length accepted while joining wrapped preset lines.
#: Prevents memory blowups from malformed presets with unbounded continuations.
MAX_CONTINUATION_LENGTH: int = 32 * 1024

#: Maximum attempts to install/compile a preset before giving up.
#: Bounds recovery churn when a preset or GL state repeatedly fails to install.
PRESET_INSTALL_MAX_ATTEMPTS: int = 5

#: Default per-frame VM instruction budget.
#: Keeps normal preset evaluation within real-time frame constraints.
MAX_FRAME_BUDGET: int = 50_000

#: Higher VM instruction budget for init-time evaluation.
#: Allows heavier one-time setup without using the same tight frame cap.
MAX_FRAME_BUDGET_INIT: int = 100_000

#: Base per-point evaluation budget for wave/shape point equations.
#: Prevents expensive per-sample scripts from collapsing frame rate.
MAX_PER_POINT_BUDGET: int = 200_000

#: Slack added to per-point budgets for small fixed overhead.
PER_POINT_BUDGET_SLACK: int = 8_192

#: Multiplier applied to instruction estimates when deriving per-point budgets.
PER_POINT_BUDGET_INSTRUCTION_FACTOR: int = 4

#: Interval (in VM steps) between cooperative budget/timeout checks.
#: Balances responsiveness to budget exhaustion against check overhead.
_BUDGET_CHECK_INTERVAL: int = 256

#: Wall-clock timeout (seconds) for VM evaluation safety checks.
#: Acts as a backstop when step-based accounting cannot detect all hangs.
_VM_EVAL_TIMEOUT: float = 1.0

# ---------------------------------------------------------------------------
# Regexes
# ---------------------------------------------------------------------------

ASSIGN_RE: re.Pattern[str] = re.compile(
    r"^\s*([A-Za-z_]\w*|g?megabuf\s*\(.*\))\s*="
)

_compound_assign_re: re.Pattern[str] = re.compile(
    r"^\s*([A-Za-z_]\w*|g?megabuf\s*\(.*\))\s*[+\-*/%^]="
)

WAVE_RE: re.Pattern[str] = re.compile(r"wave(?:code)?_(\d+)_([a-zA-Z_]\w*)")

SHAPE_RE: re.Pattern[str] = re.compile(
    r"(?:shapecode|shape)_(\d+)_([a-zA-Z_]\w*)"
)

RE_MEGABUF_LHS: re.Pattern[str] = re.compile(
    r"^g?megabuf\s*\(.*\)$", re.IGNORECASE
)

RE_STRICT_IDENTIFIER: re.Pattern[str] = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

# ``and``, ``or``, ``not`` are MilkDrop/EEL boolean operators (not Python
# keywords in preset semantics) and ``if``/``else`` appear in MilkDrop's
# ternary/conditional forms; renaming them would break translation/parsing.
# All other Python keywords are suffixed with ``_`` to produce safe identifiers.
_PYTHON_KEYWORD_RENAME: dict[str, str] = {
    k: f"{k}_"
    for k in keyword.kwlist
    if k not in ("and", "or", "not", "if", "else")
}

_BRANCH_TARGET_RE: re.Pattern[str] = re.compile(
    r"^([A-Za-z_]\w*(?:\s*\([^()]*\))?)\s*(?:[-+*/]?=)(?!=)\s*\S", re.DOTALL
)

_METADATA_LHS_RE: re.Pattern[str] = re.compile(
    r"^(?:fRating|PSVERSION|MILKDROP_[A-Za-z0-9_]*)$",
    re.IGNORECASE,
)

LOOP_START_RE: re.Pattern[str] = re.compile(
    r"\b(?:loop|while)\s*\(", re.IGNORECASE
)

# ---------------------------------------------------------------------------
# Variable Kinds
# ---------------------------------------------------------------------------


class VarKind(IntEnum):
    """Storage class for preset variables resolved by name.

    The integer values are stable and match the historic ``_VAR_KIND_*``
    constants so existing call sites and serialized assumptions remain valid.
    """

    DICT = 0  # free-form ``vars`` dictionary
    ARRAY = 1  # ``vars_array`` (the fixed VARIABLES table)
    Q = 2  # ``q1``-``q32`` registers
    T = 3  # ``t1``-``t8`` bridge variables
    REG = 4  # ``reg00``-``reg99`` global registers


# Historic aliases kept for compatibility with existing call sites/imports.
_VAR_KIND_DICT = int(VarKind.DICT)
_VAR_KIND_ARRAY = int(VarKind.ARRAY)
_VAR_KIND_Q = int(VarKind.Q)
_VAR_KIND_T = int(VarKind.T)
_VAR_KIND_REG = int(VarKind.REG)

# ---------------------------------------------------------------------------
# Return-Type Tables
# ---------------------------------------------------------------------------

BUILTIN_RETURN_TYPES: types.MappingProxyType[str, str] = (
    types.MappingProxyType(
        {
            "lum": "float",
            "length": "float",
            "distance": "float",
            "dot": "float",
            "sin": "float",
            "cos": "float",
            "tan": "float",
            "asin": "float",
            "acos": "float",
            "atan": "float",
            "atan2": "float",
            "sinh": "float",
            "cosh": "float",
            "tanh": "float",
            "pow": "float",
            "exp": "float",
            "exp2": "float",
            "log": "float",
            "log2": "float",
            "sqrt": "float",
            "inversesqrt": "float",
            "sign": "float",
            "floor": "float",
            "ceil": "float",
            "trunc": "float",
            "round": "float",
            "normalize": "vec3",
            "reflect": "vec3",
            "refract": "vec3",
            "cross": "vec3",
            "texture": "vec4",
            "texture2D": "vec4",
            "textureLod": "vec4",
            "textureGrad": "vec4",
            "tex2D": "vec4",
            "tex3D": "vec4",
            "GetPixel": "vec3",
            "getpixel": "vec3",
            "GetBlur1": "vec3",
            "getblur1": "vec3",
            "GetBlur2": "vec3",
            "getblur2": "vec3",
            "GetBlur3": "vec3",
            "getblur3": "vec3",
            "above": "float",
            "below": "float",
            "equal": "float",
            "above_equal": "float",
            "band": "float",
            "rand": "float",
            # NOTE: ``mul`` is intentionally absent. ``mul(a, b)`` lowers to GLSL
            # ``a * b``, whose result type depends on operand shapes (component-
            # wise, vector/matrix rules). A fixed ``float`` return would silently
            # misclassify valid non-scalar forms; infer from arguments instead.
        }
    )
)

_FUNCTION_RETURN_TYPES: types.MappingProxyType[str, str] = (
    types.MappingProxyType(
        {
            "below_equal": "float",
            "not_equal": "float",
            "wrap": "float",
            "cwrap": "float",
            "sigmoid": "float",
            "log10": "float",
            "triangle": "float",
            "sqr": "float",
            "int": "float",
            "frac": "float",
            "megabuf": "float",
            "gmegabuf": "float",
            "bor": "float",
            "bnot": "float",
            "pulse": "float",
            "select": "float",
            "invsqrt": "float",
            "fmod": "float",
            "fact": "float",
            "nchoosek": "float",
            "saturate": "float",
            "lerp": "float",
            "getblur1": "vec3",
            "getblur2": "vec3",
            "getblur3": "vec3",
            "getpixel": "vec3",
            # NSEEL-internal comparison/boolean aliases share scalar float return
            # types with their public counterparts.
            "_aboeq": "float",
            "_beleq": "float",
            "_above": "float",
            "_below": "float",
            "_equal": "float",
            "_nteq": "float",
            "_not": "float",
        }
    )
)

_EXPR_RETURN_TYPES: types.MappingProxyType[str, str] = types.MappingProxyType(
    {
        name.lower(): return_type
        for name, return_type in {**_FUNCTION_RETURN_TYPES, **BUILTIN_RETURN_TYPES}.items()
    }
)

#: Parse categories that indicate a preset cannot be compiled safely.
FATAL_PARSE_CATEGORIES: frozenset[str] = frozenset(
    {"parser_bug", "unsupported_syntax", "invalid_equation"}
)

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class PresetCompileError(Exception):
    """Raised when a preset expression or shader fails to compile."""


class PresetBudgetExhausted(Exception):
    """Raised when the per-frame VM evaluation budget is exhausted.

    This is **not** a compilation error — the preset is valid, but it would
    consume more instruction steps than the frame budget allows.  The exception
    is raised inside :meth:`BytecodeProgram.execute` and propagates up to
    :meth:`PresetVM.run_frame`, which catches it and skips remaining expressions
    for the frame while preserving last known-good variable state.
    """

    def __init__(
        self,
        message: str = "",
        *,
        steps_consumed: int = 0,
        steps_remaining: int = 0,
    ) -> None:
        super().__init__(message)
        self.steps_consumed = steps_consumed
        self.steps_remaining = steps_remaining


# ---------------------------------------------------------------------------
# Crash Log
# ---------------------------------------------------------------------------

CRASH_LOG_MAX_SIZE: int = 1024 * 1024  # 1 MB
CRASH_LOG_BACKUP_COUNT: int = 3


def _write_crash_log(entry: str) -> None:
    """Best-effort append of *entry* to the crash log with size rotation.

    Rotation keeps ``CRASH_LOG_BACKUP_COUNT`` numbered backups.  Errors are
    swallowed so crash logging never interferes with rendering/recovery.
    """
    try:
        crash_log = PROJECT_ROOT / "linamp_crash.log"
        crash_log.parent.mkdir(parents=True, exist_ok=True)

        # Rotate if current log exceeds max size.
        if crash_log.exists() and crash_log.stat().st_size >= CRASH_LOG_MAX_SIZE:
            for i in range(CRASH_LOG_BACKUP_COUNT - 1, 0, -1):
                src = crash_log.with_suffix(f".{i}.log")
                dst = crash_log.with_suffix(f".{i + 1}.log")
                if src.exists():
                    dst.unlink(missing_ok=True)
                    src.rename(dst)
            if crash_log.exists():
                crash_log.rename(crash_log.with_suffix(".1.log"))

        text = entry if entry.endswith("\n") else f"{entry}\n"
        with crash_log.open("a", encoding="utf-8") as f:
            f.write(text)
    except OSError:
        pass  # Best effort; never raise from crash logging.


# ---------------------------------------------------------------------------
# Equation Helpers
# ---------------------------------------------------------------------------
# These helpers are shared by preset parsing and expression compilation.
# They can be extracted to ``equation_utils.py`` later; keep re-exporting
# them from this module (and ``renderer.__init__``) to preserve imports.


def _comma_splits_statement(s: str, i: int) -> bool:
    rest = s[i + 1 :]
    return bool(re.match(r"\s*(?:g?megabuf\s*\([^)]*\)|[A-Za-z_][\w]*)\s*=", rest))


def _split_milk_eqs(val: str) -> list[str]:
    """Split expression value by semicolons without breaking backtick strings.

    A ``;`` only separates two statements at parenthesis depth 0.  Presets may
    place statement separators inside argument lists (e.g.
    ``if(beat, timediff = time - time_st; time_st = time, timediff)``); splitting
    on those semicolons would tear a single statement apart.
    """
    result: list[str] = []
    current: list[str] = []
    in_backtick = False
    depth = 0
    for _i, ch in enumerate(val):
        if ch == "`":
            in_backtick = not in_backtick
            current.append(ch)
        elif (
            ch == ";" or (ch == "," and _comma_splits_statement(val, _i))
        ) and not in_backtick and depth <= 0:
            part = "".join(current).strip()
            if part:
                result.append(part)
            current = []
        else:
            if not in_backtick:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth = max(0, depth - 1)
            current.append(ch)
    part = "".join(current).strip()
    if part:
        result.append(part)
    return result


def _strip_inline_comments(val: str) -> str:
    """Remove ``//`` comments from a MilkDrop equation value."""
    result: list[str] = []
    in_backtick = False
    i = 0
    while i < len(val):
        c = val[i]
        if c == "`":
            in_backtick = not in_backtick
            result.append(c)
            i += 1
        elif c == "/" and i + 1 < len(val) and val[i + 1] == "/" and not in_backtick:
            break
        else:
            result.append(c)
            i += 1
    return "".join(result).rstrip()


def _is_metadata_or_comment_expression(expr_str: str) -> bool:
    """Return ``True`` when *expr_str* is a comment or metadata assignment.

    The check is syntax-aware: inline ``//`` comments are stripped first, and
    only the assignment target (before the first ``=``) is matched against
    metadata key patterns.  An equation that merely references a metadata
    variable (e.g. ``vol = fRating * 0.5``) is still compiled, while a leaked
    metadata line (``fRating=2.0``) is dropped.
    """
    cleaned = _strip_inline_comments(str(expr_str)).strip()
    if not cleaned or cleaned.startswith("//"):
        return True
    lhs = cleaned.split("=", 1)[0].strip()
    return bool(_METADATA_LHS_RE.match(lhs))

# Configuration constants (extracted from gl_state during refactoring)
AUDIO_SNAPSHOT_AGE_THRESHOLD: float = 1.0
TEXTURE_UNIT_MEGABUF: int = 1024
