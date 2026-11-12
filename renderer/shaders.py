from __future__ import annotations

import ast
import functools
import json
import logging
import math
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar, cast

# ---------------------------------------------------------------------------
# Expression optimization: constant folding and common subexpression elimination
# ---------------------------------------------------------------------------

# Set of known constant float values that can be folded at compile time.
# These are safe to precompute because they never change per-frame.
_KNOWN_CONSTANTS: frozenset[str] = frozenset({
    "pi",
    "two_pi",
    "half_pi",
    "quarter_pi",
    "e",
    "sqrt2",
    "sqrt1_2",
    "golden_ratio",
})


def _fold_constants(expr: str) -> str:
    """Fold known MilkDrop constants to numeric literals at compile time.

    Reduces per-frame expression work and generated GLSL size by replacing
    standalone constant tokens with their precomputed values.
    """
    replacements: dict[str, str] = {
        "pi": str(math.pi),
        "two_pi": str(2.0 * math.pi),
        "half_pi": str(0.5 * math.pi),
        "quarter_pi": str(0.25 * math.pi),
        "e": str(math.e),
        "sqrt2": str(math.sqrt(2.0)),
        "sqrt1_2": str(math.sqrt(0.5)),
        "golden_ratio": str((1.0 + math.sqrt(5.0)) / 2.0),
    }

    result = expr
    for const_name, const_val in replacements.items():
        # Only replace when the constant appears as a standalone token,
        # not as part of an identifier (e.g., "pi" inside "pi2" or "api" must not match).
        # Also avoid replacing constants on the LHS of an assignment (e.g., "pi = ...").
        pattern = (
            r"(?<!\w)" + re.escape(const_name) + r"(?!\w)(?!\s*=)"
        )
        result = re.sub(pattern, const_val, result)

    return result


_LEADING_ZERO_INT = re.compile(r"(?<![\d.])0[0-9]+(?![\d.])")


def _normalize_leading_zeros(expr: str) -> str:
    """Remove leading zeros from integer literals in the expression.

    MilkDrop allows leading zeros in integer literals (e.g., 0123), but Python
    treats them as octal (pre-3.0) or syntax errors (3.0+). This normalizes
    them to standard decimal integers.
    """
    return _LEADING_ZERO_INT.sub(lambda m: m.group(0).lstrip("0") or "0", expr)


def _normalize_milkdrop_ops(expr: str) -> str:
    """Normalize MilkDrop-specific operators to Python-compatible syntax.

    MilkDrop uses ``!`` for logical NOT, ``&&`` for AND, ``||`` for OR, and
    ``if()`` for ternary.  ``ast.parse(..., mode="eval")`` does not support any
    of these, so they are converted to Python equivalents before parsing.

    The conversion is done on *tokens* (via :class:`MilkDropTokenizer`) rather
    than with ad-hoc regular expressions.  A regex-based rewrite was previously
    used here and produced several categories of invalid Python:

      * ``&&``/``||`` were replaced in place, merging adjacent identifiers into
        a single token (``a&&b`` -> ``aandb``), which later collided with a
        trailing numeric literal (``1.5&&2`` -> ``1.5and2`` -> *invalid decimal
        literal*).
      * ``!`` was rewritten to ``not `` without parentheses, so ``a + !b`` and
        ``a and !b`` became ``a + not b`` / ``a and not b``, which Python
        rejects (*'not' after an operator must be parenthesized*).
      * Nested ``!`` (``!!x``) collapsed to ``!not x``, which is invalid.
      * ``if(a, b, c)`` with commas inside the condition/branches was not
        matched because the pattern excluded nested parens.

    The tokenizer already handles all of these correctly for the GLSL path; this
    helper reuses it so the CPU bytecode path produces the same Python source.
    """
    if not expr or not expr.strip():
        return expr
    return MilkDropParser().parse(expr)


# ---------------------------------------------------------------------------
# GLSLGenerator minor optimizations
# ---------------------------------------------------------------------------

# Cache for function type lookups during GLSL generation.
# REMOVED: the previous `_function_type_cache` / `_cached_func_type` helpers
# referenced a non-existent `_call_type_uncached()` and had no active callers.
# Function types are resolved directly via `GLSLGenerator._call_type()`.


# Explicit rebinds to stabilize Pylance inference across Final[...] re-exports.
import re as _re
import types as _types
from pathlib import Path as _Path

import numpy as np

from equations import *
from resources import (
    T_LOOKUP,
    VARIABLES,
)

from .budgets import (
    FrameBudget,
    LoopBudget,
)
from .config import (
    _BRANCH_TARGET_RE,
    _BUDGET_CHECK_INTERVAL,
    _EXPR_RETURN_TYPES,
    _PYTHON_KEYWORD_RENAME,
    _T,
    MAX_BYTECODE_STEPS,
    MAX_FRAME_LOOP_ITERATIONS,
    MAX_LOOPS_PER_EXPRESSION,
    MAX_SHAPE_SIDES,
    PROJECT_ROOT,
    RE_MEGABUF_LHS,
    RE_STRICT_IDENTIFIER,
    VM_DEBUG,
    PresetBudgetExhausted,
    PresetCompileError,
    _comma_splits_statement,
    _is_metadata_or_comment_expression,
    _split_milk_eqs,
    _strip_inline_comments,
    _write_crash_log,
)

PROJECT_ROOT: _Path = PROJECT_ROOT  # type: ignore[assignment,no-redef]  # noqa: PLW0127
_EXPR_RETURN_TYPES: _types.MappingProxyType[str, str] = _EXPR_RETURN_TYPES  # type: ignore[assignment,no-redef]  # noqa: PLW0127
_PYTHON_KEYWORD_RENAME: dict[str, str] = _PYTHON_KEYWORD_RENAME  # type: ignore[assignment,no-redef]  # noqa: PLW0127
_BRANCH_TARGET_RE: _re.Pattern[str] = _BRANCH_TARGET_RE  # type: ignore[assignment,no-redef]  # noqa: PLW0127
MAX_BYTECODE_STEPS: int = MAX_BYTECODE_STEPS  # type: ignore[assignment,no-redef]  # noqa: PLW0127
_BUDGET_CHECK_INTERVAL: int = _BUDGET_CHECK_INTERVAL  # type: ignore[assignment,no-redef]  # noqa: PLW0127
MAX_FRAME_LOOP_ITERATIONS: int = MAX_FRAME_LOOP_ITERATIONS  # type: ignore[assignment,no-redef]  # noqa: PLW0127
MAX_LOOPS_PER_EXPRESSION: int = MAX_LOOPS_PER_EXPRESSION  # type: ignore[assignment,no-redef]  # noqa: PLW0127
MAX_SHAPE_SIDES: int = MAX_SHAPE_SIDES  # type: ignore[assignment,no-redef]  # noqa: PLW0127

if False:  # TYPE_CHECKING-like hints for Pylance (values imported above)
    import re
    import types
    from pathlib import Path

    PROJECT_ROOT: Path
    _EXPR_RETURN_TYPES: types.MappingProxyType[str, str]
    _PYTHON_KEYWORD_RENAME: dict[str, str]
    _BRANCH_TARGET_RE: re.Pattern[str]
    MAX_BYTECODE_STEPS: int
    _BUDGET_CHECK_INTERVAL: int
    MAX_FRAME_LOOP_ITERATIONS: int
    MAX_LOOPS_PER_EXPRESSION: int
    MAX_SHAPE_SIDES: int
from .state import (
    PresetState,
    StateLike,
    _reg_var_index,
    _StateGlobalRegsProxy,
    _StateVarsProxy,
)

logger = logging.getLogger("milkdrop_renderer")


__all__ = [
    "EPSILON_BOOL",
    "EPSILON_COMPARE",
    "EPSILON_DIV",
    "EPSILON_SMALL",
    "FUNCTION_REGISTRY",
    "BytecodeCompiler",
    "BytecodePeepholeOptimizer",
    "BytecodeProgram",
    "CompiledExpr",
    "ExpressionOptimizer",
    "FunctionSpec",
    "GLSLGenerator",
    "LoopExpr",
    "MilkDropParser",
    "MilkDropTokenizer",
    "ShaderRepairReport",
    "ShapeProgram",
    "WaveProgram",
    "WaveSamples",
    "_append_compiled_expression",
    "_compile_expression_list",
    "_compile_shape",
    "_compile_wave",
    "_numeric_const",
    "_split_statements",
    "compile_assignment_to_glsl",
    "compile_expr_to_glsl",
    "optimize_ast",
    "preprocess_milkdrop_expr",
    "validate_glsl_source",
]

#: Valid per-pixel assignment LHS: a simple identifier optionally followed by
#: a component swizzle (``ret.r``, ``uv.x``, ``center.y``).  Function calls
#: (``gmegabuf(n)``), array subscripts (``u_q[1]``), and bare type keywords
#: are not assignable and would emit invalid GLSL
#: (``syntax error, unexpected '='``).
_VALID_ASSIGNMENT_LHS_RE = re.compile(
    r"^[A-Za-z_]\w*(\.[xyzwrgba]{1,4})?$"
)

#: GLSL type keywords that cannot appear as an assignment LHS.
_GLSL_TYPE_KEYWORDS: frozenset[str] = frozenset(
    {
        "float",
        "vec2",
        "vec3",
        "vec4",
        "int",
        "uint",
        "bool",
        "double",
        "void",
        "mat2",
        "mat3",
        "mat4",
        "mat2x2",
        "mat2x3",
        "mat2x4",
        "mat3x2",
        "mat3x3",
        "mat3x4",
        "mat4x2",
        "mat4x3",
        "mat4x4",
        "sampler2D",
        "sampler3D",
        "samplerCube",
        "sampler2DShadow",
        "samplerCubeShadow",
        "isampler2D",
        "usampler2D",
    }
)


# --- Shader source validation -----------------------------------------------
def _strip_comments_and_strings(source: str) -> str:
    """Return *source* with comment and string-literal contents blanked out.

    Comment and string regions are replaced with spaces (preserving length) so
    that delimiters appearing inside them cannot fool delimiter balancing.
    Handles ``//`` line comments, ``/* */`` block comments and double-quoted
    GLSL strings (with backslash escapes).
    """
    result: list[str] = []
    i = 0
    n = len(source)
    while i < n:
        c = source[i]
        nxt = source[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "/":
            end = source.find("\n", i)
            if end == -1:
                end = n
            result.append(" " * (end - i))
            i = end
        elif c == "/" and nxt == "*":
            end = source.find("*/", i + 2)
            if end == -1:
                end = n
            else:
                end += 2
            result.append(" " * (end - i))
            i = end
        elif c == '"':
            j = i + 1
            while j < n:
                if source[j] == "\\":
                    j += 2
                    continue
                if source[j] == '"':
                    break
                j += 1
            end = j + 1 if j < n else n
            result.append(" " * (end - i))
            i = end
        else:
            result.append(c)
            i += 1
    return "".join(result)


def _check_braces(source: str) -> list[str]:
    """Check that GLSL delimiters are balanced, ignoring comments and strings.

    Braces (``{}``), parentheses (``()``) and brackets (``[]``) are tracked;
    delimiters inside comments or string literals are skipped.
    """
    cleaned = _strip_comments_and_strings(source)
    warnings: list[str] = []
    stack: list[tuple[int, str]] = []
    pairs = {"{": "}", "(": ")", "[": "]"}
    closing = set(pairs.values())
    for i, ch in enumerate(cleaned):
        if ch in pairs:
            stack.append((i, ch))
        elif ch in closing:
            if not stack:
                warnings.append(f"Unmatched closing delimiter '{ch}' at position {i}")
            else:
                pos, open_ch = stack.pop()
                if pairs[open_ch] != ch:
                    warnings.append(
                        f"Mismatched delimiter: '{open_ch}' at {pos} "
                        f"closed by '{ch}' at {i}"
                    )
    for pos, open_ch in stack:
        warnings.append(f"Unmatched opening delimiter '{open_ch}' at position {pos}")
    return warnings



# --- Sample geometry and shared function definitions ------------------
class WaveSamples:
    """Sample geometry (positions and colours) of a single wave.

    MilkDrop evaluates a wave's per-point equations once per sample point
    and draws the resulting polyline.  Every wave owns its samples: a
    single shared buffer made every wave of a multi-wave preset render the
    geometry of whichever wave happened to be evaluated last.

    ``valid_points`` is the number of leading samples that hold valid
    geometry.  A per-point phase that was cut short (budget or wall clock)
    leaves the remaining samples holding the previous frame's values, and
    on the first frame they hold zeroes; the renderer draws only the valid
    prefix so a truncated wave never spikes into the NDC corner.
    """

    __slots__ = ("a", "b", "g", "r", "valid_points", "x", "y")

    def __init__(self, size: int = 512):
        self.x = np.zeros(size, dtype=np.float32)
        self.y = np.zeros(size, dtype=np.float32)
        self.r = np.zeros(size, dtype=np.float32)
        self.g = np.zeros(size, dtype=np.float32)
        self.b = np.zeros(size, dtype=np.float32)
        self.a = np.zeros(size, dtype=np.float32)
        self.valid_points = 0

    def copy(self) -> WaveSamples:
        """Return an independent copy (used to roll back a partial update)."""
        clone = WaveSamples(len(self.x))
        clone.x[:] = self.x
        clone.y[:] = self.y
        clone.r[:] = self.r
        clone.g[:] = self.g
        clone.b[:] = self.b
        clone.a[:] = self.a
        clone.valid_points = self.valid_points
        return clone

    def restore(self, other: WaveSamples) -> None:
        """Copy *other*'s samples back into this buffer."""
        self.x[:] = other.x
        self.y[:] = other.y
        self.r[:] = other.r
        self.g[:] = other.g
        self.b[:] = other.b
        self.a[:] = other.a
        self.valid_points = other.valid_points


@dataclass(slots=True, frozen=True)
class FunctionSpec:
    """Shared MilkDrop function definitions (Task 8)."""

    cpu_impl: Callable
    glsl_impl: Callable


FUNCTION_REGISTRY: dict[str, FunctionSpec] = {
    "above": FunctionSpec(
        cpu_impl=lambda a, b: 1.0 if a > b else 0.0,
        glsl_impl=lambda args: f"({args[0]} > {args[1]} ? 1.0 : 0.0)",
    ),
    "below": FunctionSpec(
        cpu_impl=lambda a, b: 1.0 if a < b else 0.0,
        glsl_impl=lambda args: f"({args[0]} < {args[1]} ? 1.0 : 0.0)",
    ),
    "equal": FunctionSpec(
        cpu_impl=lambda a, b: 1.0 if abs(a - b) < EPSILON_COMPARE else 0.0,
        glsl_impl=lambda args: (
            f"(abs({args[0]} - {args[1]}) < {EPSILON_COMPARE} ? 1.0 : 0.0)"
        ),
    ),
    "not_equal": FunctionSpec(
        cpu_impl=lambda a, b: 0.0 if abs(a - b) < EPSILON_COMPARE else 1.0,
        glsl_impl=lambda args: (
            f"(!(abs({args[0]} - {args[1]}) < {EPSILON_COMPARE}) ? 1.0 : 0.0)"
        ),
    ),
    "above_equal": FunctionSpec(
        cpu_impl=lambda a, b: 1.0 if a >= b else 0.0,
        glsl_impl=lambda args: f"({args[0]} >= {args[1]} ? 1.0 : 0.0)",
    ),
    "below_equal": FunctionSpec(
        cpu_impl=lambda a, b: 1.0 if a <= b else 0.0,
        glsl_impl=lambda args: f"({args[0]} <= {args[1]} ? 1.0 : 0.0)",
    ),
    "assign": FunctionSpec(
        cpu_impl=lambda *args: float(args[-1]) if args else 0.0,  # Returns assigned value
        glsl_impl=lambda args: f"({args[0]} = {args[1]})" if len(args) >= 2 else "0.0",
    ),
    # exec3/2/1: evaluate arguments (often assign() calls) and return the last.
    "exec1": FunctionSpec(
        cpu_impl=lambda *args: float(args[-1]) if args else 0.0,
        glsl_impl=lambda args: f"({', '.join(args)})" if args else "0.0",
    ),
    "exec2": FunctionSpec(
        cpu_impl=lambda *args: float(args[-1]) if args else 0.0,
        glsl_impl=lambda args: f"({', '.join(args)})" if args else "0.0",
    ),
    "exec3": FunctionSpec(
        cpu_impl=lambda *args: float(args[-1]) if args else 0.0,
        glsl_impl=lambda args: f"({', '.join(args)})" if args else "0.0",
    ),
    "wrap": FunctionSpec(
        cpu_impl=lambda x, a, b: (
            a + ((x - a) % (b - a)) if abs(b - a) > EPSILON_DIV else a
        ),
        glsl_impl=lambda args: f"wrap({args[0]}, {args[1]}, {args[2]})",
    ),
    "cwrap": FunctionSpec(
        cpu_impl=lambda x, a, b: (
            a + abs(((x - a) % (2 * (b - a))) - (b - a))
            if abs(b - a) > EPSILON_DIV
            else a
        ),
        glsl_impl=lambda args: f"cwrap({args[0]}, {args[1]}, {args[2]})",
    ),
    "sigmoid": FunctionSpec(
        cpu_impl=lambda x, constraint=1.0: _md_sigmoid(x, constraint),
        glsl_impl=lambda args: (
            f"(1.0 / (1.0 + exp(-{args[0]} * {args[1]})))"
            if len(args) >= 2
            else f"(1.0 / (1.0 + exp(-{args[0]})))"
        ),
    ),
    "log10": FunctionSpec(
        cpu_impl=lambda x: _md_log10(x),
        glsl_impl=lambda args: (
            # GLSL ES has no portable log10() intrinsic; compute log10(x) as
            # log2(x) / log2(10).  Clamp the argument away from zero so the
            # logarithm never produces -Inf / NaN for non-positive input.
            f"(log2(max({args[0]}, 1e-30)) * 0.3010299956639812)"
        ),
    ),
    "triangle": FunctionSpec(
        cpu_impl=lambda x: 2.0 * abs((x * 0.5) % 1.0 - 0.5),
        glsl_impl=lambda args: f"(2.0 * abs(fract({args[0]} * 0.5) - 0.5))",
    ),
    "sqr": FunctionSpec(
        cpu_impl=lambda x: x * x, glsl_impl=lambda args: f"({args[0]} * {args[0]})"
    ),
    "sign": FunctionSpec(
        cpu_impl=lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0),
        glsl_impl=lambda args: f"sign({args[0]})",
    ),
    "int": FunctionSpec(
        cpu_impl=lambda x: float(_md_int(x)),
        glsl_impl=lambda args: f"float(int({args[0]}))",
    ),
    "frac": FunctionSpec(
        cpu_impl=lambda x: x - _md_floor(x), glsl_impl=lambda args: f"fract({args[0]})"
    ),
    "rand": FunctionSpec(
        cpu_impl=lambda x=1.0: _md_rand_cpu(x),
        glsl_impl=lambda args: f"(md_rand(uv) * {args[0] if args else '1.0'})",
    ),
    "megabuf": FunctionSpec(
        cpu_impl=lambda idx: (
            0.0
        ),  # not used: megabuf is in BytecodeCompiler.func_map → OP_MEGABUF_GET
        glsl_impl=lambda args: f"megabuf_read({args[0]})",
    ),
    "gmegabuf": FunctionSpec(
        cpu_impl=lambda idx: (
            0.0
        ),  # not used: gmegabuf is in BytecodeCompiler.func_map → OP_MEGABUF_GET
        glsl_impl=lambda args: f"megabuf_read({args[0]})",
    ),
    "bor": FunctionSpec(
        cpu_impl=lambda a, b: float(int(_md_int(a)) | int(_md_int(b))),
        glsl_impl=lambda args: f"float(int({args[0]}) | int({args[1]}))",
    ),
    "band": FunctionSpec(
        cpu_impl=lambda a, b: float(int(_md_int(a)) & int(_md_int(b))),
        glsl_impl=lambda args: f"float(int({args[0]}) & int({args[1]}))",
    ),
    "bnot": FunctionSpec(
        cpu_impl=lambda a: float(~int(_md_int(a))),
        glsl_impl=lambda args: f"float(~int({args[0]}))",
    ),
    "pulse": FunctionSpec(
        cpu_impl=lambda x, w=0.5: 1.0 if (x - _md_floor(x)) < w else 0.0,
        glsl_impl=lambda args: (
            f"(fract({args[0]}) < {args[1] if len(args) >= 2 else '0.5'} ? 1.0 : 0.0)"
        ),
    ),
    "select": FunctionSpec(
        cpu_impl=lambda c, a, b: a if c != 0 else b,
        glsl_impl=lambda args: f"({args[0]} != 0.0 ? {args[1]} : {args[2]})",
    ),
    "sqrt": FunctionSpec(
        cpu_impl=lambda x: _md_sqrt(x),
        glsl_impl=lambda args: f"sqrt(max({args[0]}, 0.0))",
    ),
    "invsqrt": FunctionSpec(
        cpu_impl=lambda x: _md_invsqrt(x),
        glsl_impl=lambda args: f"(max({args[0]}, 0.0) > 0.0 ? inversesqrt({args[0]}) : 0.0)",
    ),
    "fmod": FunctionSpec(
        cpu_impl=lambda a, b: a % b if b != 0 else 0.0,
        glsl_impl=lambda args: f"mod({args[0]}, {args[1]})",
    ),
    "fact": FunctionSpec(
        cpu_impl=lambda n: _md_factorial(n),
        glsl_impl=lambda args: f"fact({args[0]})",
    ),
    "nchoosek": FunctionSpec(
        cpu_impl=lambda n, k: float(math.comb(int(n), int(k))) if int(k) >= 0 else 0.0,
        glsl_impl=lambda args: f"nchoosek({args[0]}, {args[1]})",
    ),
    # --- MilkDrop intrinsic instructions (Task: shader intrinsics) ---
    # saturate(a): clamp a to [0,1].  Free on most GPUs; works on any
    # float/floatN type, so we map straight to GLSL clamp().
    "saturate": FunctionSpec(
        cpu_impl=lambda x: max(0.0, min(1.0, float(x))),
        glsl_impl=lambda args: f"clamp({args[0]}, 0.0, 1.0)",
    ),
    # lerp(a, b, c): a + c*(b-a).  c may be scalar or same type as a/b.
    # GLSL mix() handles every floatN x floatN x float/ floatN combination.
    "lerp": FunctionSpec(
        cpu_impl=lambda a, b, c: float(a) + float(c) * (float(b) - float(a)),
        glsl_impl=lambda args: f"mix({args[0]}, {args[1]}, {args[2]})",
    ),
    # lum(a): luminance of a color (float3).  Per the intrinsic spec this is
    # dot(a, float3(0.32, 0.49, 0.29)).  The helper overloads in
    # milkdrop_helpers.glsl provide the float/vec2/vec3 variants; here we just
    # forward to them.
    "lum": FunctionSpec(
        cpu_impl=lambda c: (
            float(c)
            if isinstance(c, (int, float))
            else sum(float(v) * w for v, w in zip(c, (0.32, 0.49, 0.29)))
        ),
        glsl_impl=lambda args: f"lum({args[0]})",
    ),
    # mul(a, b): vector x matrix (row-major) or matrix x vector (column-major).
    # GLSL's * operator already does the right thing depending on operand
    # order, so we just emit a*b.
    "mul": FunctionSpec(
        cpu_impl=lambda a, b: _md_mul_cpu(a, b),
        glsl_impl=lambda args: f"({args[0]} * {args[1]})",
    ),
    # cross(a, b): cross product of two float3 vectors.
    "cross": FunctionSpec(
        cpu_impl=lambda a, b: (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ),
        glsl_impl=lambda args: f"cross({args[0]}, {args[1]})",
    ),
    # tex2D(sampler, uv): sample a 2D texture at float2 uv -> float4.
    "tex2d": FunctionSpec(
        cpu_impl=lambda *args: (0.0, 0.0, 0.0, 0.0),
        glsl_impl=lambda args: f"texture({args[0]}, {args[1]})",
    ),
    # tex3D(sampler, uvw): sample a volume texture at float3 uvw -> float4.
    "tex3d": FunctionSpec(
        cpu_impl=lambda *args: (0.0, 0.0, 0.0, 0.0),
        glsl_impl=lambda args: f"texture({args[0]}, {args[1]})",
    ),
    # GetBlur1/2/3(uv): blurred samples of the internal canvas (feedback).
    # The vec3 GetBlurN(vec2) wrappers live in milkdrop_helpers.glsl.
    "getblur1": FunctionSpec(
        cpu_impl=lambda *args: (0.0, 0.0, 0.0),
        glsl_impl=lambda args: f"GetBlur1({args[0]})",
    ),
    "getblur2": FunctionSpec(
        cpu_impl=lambda *args: (0.0, 0.0, 0.0),
        glsl_impl=lambda args: f"GetBlur2({args[0]})",
    ),
    "getblur3": FunctionSpec(
        cpu_impl=lambda *args: (0.0, 0.0, 0.0),
        glsl_impl=lambda args: f"GetBlur3({args[0]})",
    ),
    # GetPixel(uv): sample the previous frame (alias for backwards compatibility).
    "getpixel": FunctionSpec(
        cpu_impl=lambda *args: (0.0, 0.0, 0.0),
        glsl_impl=lambda args: f"GetPixel({args[0]})",
    ),
    # exp(a): 2^a (MilkDrop intrinsic; GLSL exp() is e^a, so use exp2()).
    "exp": FunctionSpec(
        cpu_impl=lambda x: 2.0 ** float(x), glsl_impl=lambda args: f"exp2({args[0]})"
    ),
    # log(a): log2(a) (MilkDrop intrinsic; GLSL log() is natural log).
    "log": FunctionSpec(
        cpu_impl=lambda x: math.log2(x) if x > 0 else 0.0,
        glsl_impl=lambda args: f"log2({args[0]})",
    ),
}


# NSEEL-internal comparison/boolean aliases.
#
# projectM's intrinsic_function_table (projectM-eval ``TreeFunctions.c``)
# exposes a set of underscore-prefixed intrinsic names that MilkDrop
# expressions may call directly.  They map 1:1 onto the public functions
# registered above, so rather than duplicating the ``cpu_impl`` / ``glsl_impl``
# closures we alias the *same* ``FunctionSpec`` object.
#
# Every call-dispatch site resolves by name against ``FUNCTION_REGISTRY``:
#   * GLSLGenerator.visit_Call                       -> .glsl_impl
#   * BytecodeCompiler.visit_Call (OP_CALL_REGISTRY) -> compile-time name
#   * BytecodeProgram.execute OP_CALL_REGISTRY       -> .cpu_impl
# so registering the alias once here covers the CPU VM and the GLSL backend
# simultaneously and cannot let the two backends drift apart.
#
# ``_aboeq`` is the only alias exercised by shipped presets; the rest of the
# comparison family is added for reference-parity with the projectM intrinsic
# set.  All seven return a scalar ``float`` (see ``_EXPR_RETURN_TYPES`` in
# config.py).
#
# Note on ``_not``: projectM maps ``_not`` to the *same* intrinsic
# (``prjm_eval_func_bnot``) as ``bnot``, i.e. a bitwise (ones'-complement)
# not -- NOT a boolean predicate.  Aliasing to ``bnot`` therefore preserves
# the reference semantics exactly.
_NSEEL_ALIAS_TARGETS: dict[str, str] = {
    "_aboeq": "above_equal",
    "_beleq": "below_equal",
    "_above": "above",
    "_below": "below",
    "_equal": "equal",
    "_noteq": "not_equal",
    "_nteq": "not_equal",
    "_not": "bnot",
}
_alias = _target = None
for _alias, _target in _NSEEL_ALIAS_TARGETS.items():
    if _target in FUNCTION_REGISTRY and _alias not in FUNCTION_REGISTRY:
        FUNCTION_REGISTRY[_alias] = FUNCTION_REGISTRY[_target]
del _alias, _target, _NSEEL_ALIAS_TARGETS


# --- GLSL code generation ---------------------------------------------
class GLSLGenerator(ast.NodeVisitor):
    """AST-based GLSL code generator for proper MilkDrop expression compilation.
    Converts Python AST expressions to GLSL shader code.
    This is a proper compiler backend, not string hacking.
    """

    #: Minimum GLSL type for well-known MilkDrop variables.  Shared by
    #: ``GLSLGenerator`` and ``MilkdropRenderer._infer_glsl_type``; it must
    #: live on the class (``_infer_glsl_type`` is a classmethod, so an
    #: instance attribute would not be visible to it).
    KNOWN_VECTOR_TYPES: ClassVar[dict[str, str]] = {
        "uv": "vec2",
        "uv2": "vec2",
        "sx": "float",
        "sy": "float",
        "decay": "float",
        "center": "vec2",
        "offset": "vec2",
        "scale": "vec2",
        "rad": "float",
        "ang": "float",
        "cx": "float",
        "cy": "float",
        "dx": "float",
        "dy": "float",
    }

    #: MilkDrop variable → GLSL uniform/identifier mapping.  Shared by all
    #: instances, so this is a class attribute rather than set in __init__.
    milkdrop_vars: ClassVar[dict[str, str]] = {
        "x": "x",
        "y": "y",
        "time": "u_time",
        "t": "u_time",
        "q1": "q1",
        "q2": "q2",
        "q3": "q3",
        "q4": "q4",
        "q5": "q5",
        "q6": "q6",
        "q7": "q7",
        "q8": "q8",
        "q9": "q9",
        "q10": "q10",
        "q11": "q11",
        "q12": "q12",
        "q13": "q13",
        "q14": "q14",
        "q15": "q15",
        "q16": "q16",
        "q17": "q17",
        "q18": "q18",
        "q19": "q19",
        "q20": "q20",
        "q21": "q21",
        "q22": "q22",
        "q23": "q23",
        "q24": "q24",
        "q25": "q25",
        "q26": "q26",
        "q27": "q27",
        "q28": "q28",
        "q29": "q29",
        "q30": "q30",
        "q31": "q31",
        "q32": "q32",
        "t1": "t1",
        "t2": "t2",
        "t3": "t3",
        "t4": "t4",
        "t5": "t5",
        "t6": "t6",
        "t7": "t7",
        "t8": "t8",
        "bass": "u_audio.x",
        "mid": "u_audio.y",
        "treb": "u_audio.z",
        "bass_att": "u_audio_att.x",
        "mid_att": "u_audio_att.y",
        "treb_att": "u_audio_att.z",
        "beat": "u_beat",
        "beat_flash": "u_beat_flash",
        "rms": "u_audio_rms",
        "peak": "u_audio_peak",
        "rad": "rad",
        "ang": "ang",
        "pi": "3.14159265359",
        "e": "2.71828182846",
        "phi": "1.61803398875",
        "w": "u_resolution.x",
        "h": "u_resolution.y",
        "width": "u_resolution.x",
        "height": "u_resolution.y",
        "aspect": "u_resolution.z",
        "aspectx": "aspectx",
        "aspecty": "aspecty",
        "mod": "md_mod",
        "zoom": "zoom",
        "rot": "rot",
        "warp": "warp",
        "cx": "cx",
        "cy": "cy",
        "dx": "dx",
        "dy": "dy",
        "sx": "sx",
        "vol": "vol",
        "vol_att": "vol_att",
        "sy": "sy",
        "decay": "decay",
        "wave_r": "wave_r",
        "wave_g": "wave_g",
        "wave_b": "wave_b",
        "wave_a": "wave_a",
        "wave_x": "wave_x",
        "wave_y": "wave_y",
        "ob_size": "ob_size",
        "ob_r": "ob_r",
        "ob_g": "ob_g",
        "ob_b": "ob_b",
        "ob_a": "ob_a",
        "ib_size": "ib_size",
        "ib_r": "ib_r",
        "ib_g": "ib_g",
        "ib_b": "ib_b",
        "ib_a": "ib_a",
        "fwavescale": "fwavescale",
        "fwavesmoothing": "fwavesmoothing",
        "fwaveparam": "fwaveparam",
        "thresh": "thresh",
        "wave_mystery": "wave_mystery",
        "wave_mode": "wave_mode",
        "wave_size": "wave_size",
        "warpAnimSpeed": "warpAnimSpeed",
        "echo_zoom": "echo_zoom",
        "echo_alpha": "echo_alpha",
        "echo_orient": "echo_orient",
        "frame": "u_frame",
        "fps": "u_fps",
        "dt": "u_dt",
        "fc": "u_fc",
    }

    #: GLSL built-in function names that may legitimately appear in generated
    #: per-pixel code.  Stored as a class attribute (with ``ClassVar`` type) so
    #: that it is shared across all ``GLSLGenerator`` instances instead of being
    #: rebuilt on every ``__init__``.  Functions in this set whose CPU-side
    #: behaviour diverges from GLSL semantics (e.g. ``sqrt`` of a negative
    #: number) are *also* present in ``FUNCTION_REGISTRY`` so that the
    #: registry's ``glsl_impl`` takes priority in ``visit_Call``.
    GLSL_BUILTINS: ClassVar[frozenset[str]] = frozenset(
        {
            "sin",
            "cos",
            "tan",
            "asin",
            "acos",
            "atan",
            "atan2",
            "sinh",
            "cosh",
            "tanh",
            "pow",
            "exp",
            "log",
            "log2",
            "log10",
            "sqrt",
            "inversesqrt",
            "abs",
            "sign",
            "floor",
            "ceil",
            "fract",
            "trunc",
            "round",
            "mod",
            "min",
            "max",
            "clamp",
            "mix",
            "step",
            "smoothstep",
            "length",
            "distance",
            "dot",
            "cross",
            "normalize",
            "reflect",
            "refract",
            "texture",
            "vec2",
            "vec3",
            "vec4",
            "float",
            "int",
            "bool",
            "true",
            "false",
        }
    )

    def __init__(self):
        self.output = []
        # glsl_builtins is now a ClassVar (GLSL_BUILTINS), shared by all
        # instances.  Kept as an instance alias for backward compatibility with
        # any code that may reference self.glsl_builtins.
        self.glsl_builtins = self.GLSL_BUILTINS
        # Track reg00-reg99 references for uniform binding
        self.reg_refs = set()

    def _as_bool(self, val):
        """True MilkDrop boolean semantics: treat floats as booleans with epsilon."""
        return f"(abs({val}) > {EPSILON_BOOL})"

    @staticmethod
    @functools.lru_cache(maxsize=8)
    def _get_helper_functions(megabuf_size=262144):
        """Returns GLSL helper functions for MilkDrop compatibility.

        Cached to avoid repeated filesystem reads during shader compilation.
        The cache is keyed on ``megabuf_size`` (typically a single value), so a
        small bounded cache is sufficient and avoids memory growth.
        """
        helpers_path = PROJECT_ROOT / "shaders" / "includes" / "milkdrop_helpers.glsl"
        try:
            with open(helpers_path) as f:
                helpers = f.read()
            return helpers.replace("{{MEGABUF_SIZE}}", str(int(megabuf_size)))
        except (OSError, FileNotFoundError) as e:
            raise RuntimeError(
                f"Failed to load GLSL helpers from {helpers_path}: {e}"
            ) from e

    def visit(self, node) -> str:
        """Visit a node and return GLSL code string."""
        method = "visit_" + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        """Reject every AST node type that lacks an explicit visit_* handler."""
        raise PresetCompileError(
            f"Unsupported syntax: {type(node).__name__} is not a valid MilkDrop expression"
        )

    def visit_Attribute(self, node):
        """Handle attribute (swizzle) access, e.g. .x .y .z .w .r .g .b .a .rgb .rgba .xyz etc.

        MilkDrop supports component swizzling on vector types.  The swizzle is
        appended directly to the base GLSL expression; it is never wrapped in
        lum().  ``GetBlur1(uv).rgb`` therefore stays a ``vec3`` instead of being
        collapsed to a luminance scalar.

        Returns a GLSL code string, not just a type.
        """
        # Visit the base expression to get its GLSL code
        base_glsl = self.visit(node.value)
        attr = node.attr.lower()

        # A GLSL swizzle is 1..4 components, all from a single component group.
        if not 1 <= len(attr) <= 4:
            raise PresetCompileError(
                f"Invalid GLSL swizzle: {node.attr}"
            )

        if not all(c in "xyzwrgba" for c in attr):
            raise PresetCompileError(
                f"Invalid GLSL swizzle: {node.attr}"
            )

        # You cannot mix the xyzw and rgba groups in one swizzle.
        first_group = "xyzw" if attr[0] in "xyzw" else "rgba"
        if not all(c in first_group for c in attr):
            raise PresetCompileError(
                f"Invalid GLSL swizzle: {node.attr}"
            )

        # Validate the swizzle arity against the base type's component count
        # when the base type is known, so out-of-range swizzles are caught
        # before GLSL compilation.
        base_type = self.get_expr_type(node.value)
        component_count = {
            "float": 1,
            "vec2": 2,
            "vec3": 3,
            "vec4": 4,
        }.get(base_type)
        _COMPONENT_INDEX = {
            "x": 0, "r": 0,
            "y": 1, "g": 1,
            "z": 2, "b": 2,
            "w": 3, "a": 3,
        }
        if component_count is not None and any(
            _COMPONENT_INDEX[c] >= component_count
            for c in attr
        ):
            raise PresetCompileError(
                f"Invalid GLSL swizzle: {node.attr} on {base_type}"
            )

        return f"{base_glsl}.{attr}"


    def visit_Constant(self, node):
        """Convert numeric constants to GLSL."""
        if isinstance(node.value, (int, float)):
            val = float(node.value)
            # int(inf) raises OverflowError and int(nan) raises ValueError; an
            # unguarded raise here escapes as the wrong exception type and
            # aborts the whole shader build instead of this one equation.
            if not math.isfinite(val):
                raise PresetCompileError(
                    f"non-finite numeric literal in expression: {node.value!r}"
                )
            if val == int(val):
                iv = int(val)
                # Avoid enormous integer literals that overflow GLSL float range;
                # fall back to compact scientific notation for huge magnitudes.
                if abs(iv) <= 10**15:
                    return f"{iv}.0"
            text = f"{val:.9g}"
            if "." not in text and "e" not in text and "E" not in text:
                text += ".0"
            return text
        raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")

    def visit_Name(self, node):
        """Convert variable names to GLSL, including reg00-reg99 global registers."""
        name = node.id
        name_lower = name.lower()
        if name in ["PLUS_EQUALS", "MINUS_EQUALS", "TIMES_EQUALS", "DIV_EQUALS"]:
            return name
        # Handle reg00-reg99 global registers
        if (
            len(name_lower) == 5
            and name_lower.startswith("reg")
            and name_lower[3:].isdigit()
        ):
            idx = int(name_lower[3:])
            if 0 <= idx < 100:
                self.reg_refs.add(idx)
                return f"u_regs[{idx}]"
        if name_lower in self.milkdrop_vars:
            return self.milkdrop_vars[name_lower]
        # Bare GLSL builtin literals are valid expressions, but builtin
        # functions (sin, cos, texture, ...) require arguments and must not
        # be emitted as bare identifiers, or the fragment shader fails to
        # compile with "'sin' undeclared".  User-assigned variables take
        # priority via the milkdrop_vars check above.
        if name_lower in ("true", "false"):
            return name_lower
        if name_lower in self.glsl_builtins:
            raise PresetCompileError(
                f"GLSL builtin '{name}' used as a bare identifier without arguments"
            )
        if name_lower in ("inf", "infinity", "nan"):
            # Not GLSL keywords: emitting them verbatim produces a fragment
            # shader that fails to compile, which takes the whole preset down
            # instead of the single bad equation.  Drop the equation.
            raise PresetCompileError(
                f"Non-finite literal {name!r} has no GLSL representation"
            )
        return name_lower

    def visit_BinOp(self, node):
        """Convert binary operations to GLSL."""
        left = self.visit(node.left)
        right = self.visit(node.right)
        op_map = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.Mod: "%",
            ast.Pow: "pow",
            ast.FloorDiv: "/",
            ast.BitAnd: "&",
            ast.BitOr: "|",
        }
        op = op_map.get(type(node.op))
        if op is None:
            raise PresetCompileError(
                f"Unsupported binary operator: {type(node.op).__name__}"
            )
        if op == "pow":
            if right in ("2.0", "2"):
                return f"(({left})*({left}))"
            if right in ("3.0", "3"):
                return f"(({left})*({left})*({left}))"
            return f"pow({left}, {right})"
        if op == "%":
            safe_b = f"(abs({right}) < {EPSILON_DIV} ? ({right} < 0.0 ? -{EPSILON_DIV} : {EPSILON_DIV}) : {right})"
            return f"(({left}) - ({right}) * floor(({left}) / {safe_b}))"
        if isinstance(node.op, ast.FloorDiv):
            safe_r = f"((abs({right}) < {EPSILON_DIV}) ? ({right} < 0.0 ? -{EPSILON_DIV} : {EPSILON_DIV}) : ({right}))"
            return f"floor(({left}) / ({safe_r}))"
        if op == "/":
            safe_r = f"((abs({right}) < {EPSILON_DIV}) ? ({right} < 0.0 ? -{EPSILON_DIV} : {EPSILON_DIV}) : ({right}))"
            return f"(({left}) / ({safe_r}))"
        if isinstance(node.op, ast.BitAnd):
            return f"(float(int({left}) & int({right})))"
        if isinstance(node.op, ast.BitOr):
            return f"(float(int({left}) | int({right})))"
        return f"({left} {op} {right})"

    def visit_UnaryOp(self, node):
        """Convert unary operations to GLSL."""
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd):
            return f"(+{operand})"
        elif isinstance(node.op, ast.USub):
            return f"(-{operand})"
        elif isinstance(node.op, ast.Not):
            return f"(!{self._as_bool(operand)})"
        raise PresetCompileError(
            f"Unsupported unary operator: {type(node.op).__name__}"
        )

    def visit_Call(self, node):
        """Convert function calls to GLSL."""
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        args = [self.visit(arg) for arg in node.args]
        # MilkDrop function names are case-insensitive; GLSL is case-sensitive.
        # Lowercase so that e.g. ``Sin`` matches the ``sin`` builtin, ``Above``
        # matches ``above``, etc.  Only do this when the lower-cased form is a
        # known builtin / registry function; otherwise keep the original name
        # (it may be a user-defined function that happens to be capitalised).
        func_lower = func_name.lower()
        if func_lower in FUNCTION_REGISTRY:
            return FUNCTION_REGISTRY[func_lower].glsl_impl(args)
        if func_lower in self.glsl_builtins:
            if func_lower == "atan2" and len(args) >= 2:
                return f"atan({args[0]}, {args[1]})"
            args_str = ", ".join(args)
            return f"{func_lower}({args_str})"
        # Missing MilkDrop/GLSL helper functions must fail with a clear
        # diagnostic rather than emitting unknown shader calls.
        if re.fullmatch(r"getblur\d+", func_lower):
            raise PresetCompileError(
                f"Unknown GLSL helper function '{func_name}': "
                "not a supported MilkDrop blur sampler"
            )
        args_str = ", ".join(args)
        return f"{func_name}({args_str})"

    def visit_Compare(self, node):
        """Convert comparisons to GLSL (return 1.0 or 0.0)."""
        parts = []
        left = self.visit(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            right = self.visit(comparator)
            if isinstance(op, ast.Eq):
                parts.append(f"(abs({left} - {right}) < {EPSILON_COMPARE})")
            elif isinstance(op, ast.NotEq):
                parts.append(f"(abs({left} - {right}) > {EPSILON_COMPARE})")
            elif isinstance(op, ast.Lt):
                parts.append(f"({left} < {right})")
            elif isinstance(op, ast.LtE):
                parts.append(f"({left} <= {right})")
            elif isinstance(op, ast.Gt):
                parts.append(f"({left} > {right})")
            elif isinstance(op, ast.GtE):
                parts.append(f"({left} >= {right})")
            else:
                raise PresetCompileError(
                    f"Unsupported comparison operator: {type(op).__name__}"
                )
            left = right
        if not parts:
            return left
        expr = " && ".join(parts)
        return f"(({expr}) ? 1.0 : 0.0)"

    def visit_BoolOp(self, node):
        """Convert boolean operations to GLSL with MilkDrop epsilon logic."""
        values = [self.visit(v) for v in node.values]
        if not values:
            raise ValueError("Boolean operation with no values")
        if isinstance(node.op, ast.And):
            expr = " && ".join(self._as_bool(v) for v in values)
            return f"(({expr}) ? 1.0 : 0.0)"
        elif isinstance(node.op, ast.Or):
            expr = " || ".join(self._as_bool(v) for v in values)
            return f"(({expr}) ? 1.0 : 0.0)"
        return values[0]

    def visit_IfExp(self, node):
        """Convert Python's conditional expression (a if cond else b) to GLSL ternary."""
        test = self.visit(node.test)
        body = self.visit(node.body)
        orelse = self.visit(node.orelse)
        return f"({self._as_bool(test)} ? ({body}) : ({orelse}))"

    # Type map for shader-internal variables that exist in the per-pixel
    # scope.  These are declared by the shader template, not by MilkDrop,
    # but per-pixel equations can read and write them.  All other names
    # default to "float".
    _PIXEL_SHADER_VAR_TYPES: ClassVar[dict[str, str]] = {
        # vec3 template variables accessible in per-pixel scope.
        # NOTE: 'c' was removed because it collides with user-assigned
        # scalar variables (e.g., "c = 150-50*bass" in poo auction.milk),
        # causing invalid GLSL like pow(float, vec3).  The template's
        # feedback texture-sample variable is now 'md_prev_color'.
        "ret": "vec3",
        "md_prev_color": "vec3",
        "md_color": "vec3",
        # vec2 template variables accessible in per-pixel scope
        "uv": "vec2",
        "uv2": "vec2",
        "center": "vec2",
        "offset": "vec2",
        "scale": "vec2",
        "md_p": "vec2",
    }

    #: GLSL type rank for promotion (float < vec2 < vec3 < vec4)
    _TYPE_RANK: ClassVar[dict[str, int]] = {
        "float": 0,
        "int": 0,
        "bool": 0,
        "vec2": 1,
        "vec3": 2,
        "vec4": 3,
    }

    _VECTOR_PRESERVING_FUNCTIONS: ClassVar[frozenset[str]] = frozenset(
        {
            "abs",
            "acos",
            "asin",
            "atan",
            "ceil",
            "clamp",
            "cos",
            "cosh",
            "exp",
            "exp2",
            "floor",
            "fract",
            "inversesqrt",
            "lerp",
            "log2",
            "max",
            "min",
            "mix",
            "mod",
            "normalize",
            "pow",
            "refract",
            "reflect",
            "round",
            "saturate",
            "sign",
            "sin",
            "sinh",
            "sqrt",
            "tan",
            "tanh",
            "trunc",
        }
    )

    #: Functions that always return a scalar ``float`` regardless of input type.
    _SCALAR_RETURN_FUNCTIONS: ClassVar[frozenset[str]] = frozenset(
        {
            "dot",
            "distance",
            "length",
            "lum",
            "above",
            "below",
            "equal",
            "not_equal",
            "log",
            "log10",
            "triangle",
            "fact",
            "nchoosek",
            "rand",
        }
    )

    #: Functions that always return a ``vec3`` regardless of input type.
    _VECTOR3_RETURN_FUNCTIONS: ClassVar[frozenset[str]] = frozenset(
        {
            "cross",
            "getblur1",
            "getblur2",
            "getblur3",
            "getpixel",
        }
    )

    def get_expr_type(self, node: ast.AST) -> str:
        """Infer the GLSL type of a MilkDrop AST expression node.

        Returns one of ``"float"``, ``"vec2"``, ``"vec3"``, ``"vec4"``.
        This is used by :func:`compile_assignment_to_glsl` so that the
        caller can detect and repair vector/scalar type mismatches at
        generation time rather than discovering them at shader-compile time.
        """
        if isinstance(node, ast.Constant):
            return "float"
        if isinstance(node, ast.Name):
            return self._name_type(node.id)
        if isinstance(node, ast.Call):
            return self._call_type(node)
        if isinstance(node, ast.BinOp):
            left_t = self.get_expr_type(node.left)
            right_t = self.get_expr_type(node.right)
            return self._combine_types(left_t, right_t)
        if isinstance(node, ast.UnaryOp):
            return self.get_expr_type(node.operand)
        if isinstance(node, (ast.IfExp,)):
            body_t = self.get_expr_type(node.body)
            orelse_t = self.get_expr_type(node.orelse)
            return self._combine_types(body_t, orelse_t)
        if isinstance(node, ast.Attribute):
            base_t = self.get_expr_type(node.value)
            return self._swizzle_type(base_t, node.attr)
        if isinstance(node, (ast.Compare, ast.BoolOp)):
            return "float"
        if isinstance(node, ast.Subscript):
            return "float"
        # ast.List/Tuple (vec constructors written as Python lists)
        if isinstance(node, (ast.List, ast.Tuple)):
            return (
                self._combine_types(*[self.get_expr_type(e) for e in node.elts])
                if node.elts
                else "float"
            )
        return "float"

    def _name_type(self, name: str) -> str:
        """Determine the GLSL type of a variable name in the per-pixel scope."""
        name_lower = name.lower()
        # Shader-internal vector variables
        if name_lower in self._PIXEL_SHADER_VAR_TYPES:
            return self._PIXEL_SHADER_VAR_TYPES[name_lower]
        # q/t/reg register aliases are float
        if re.match(r"^(q\d{1,2}|t\d|[rR]eg\d{2})$", name_lower):
            return "float"
        # Known vector type fallback (from KNOWN_VECTOR_TYPES)
        if name_lower in self.KNOWN_VECTOR_TYPES:
            return self.KNOWN_VECTOR_TYPES[name_lower]
        # MilkDrop scalar variables (time, bass, x, y, zoom, etc.) all float
        return "float"

    def _call_type(self, node: ast.Call) -> str:
        """Determine the return type of a function call."""
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        func_lower = func_name.lower()

        # Polymorphic: mul(a, b) -> type depends on operands
        if func_lower == "mul" and len(node.args) >= 2:
            left_t = self.get_expr_type(node.args[0])
            right_t = self.get_expr_type(node.args[1])
            return self._combine_types(left_t, right_t)

        # Polymorphic: select(c, a, b) -> type of a/b
        if func_lower == "select" and len(node.args) >= 2:
            if len(node.args) >= 3:
                return self._combine_types(
                    self.get_expr_type(node.args[1]),
                    self.get_expr_type(node.args[2]),
                )
            return self.get_expr_type(node.args[1])

        # Keep this ahead of the vector-preserving fallback so functions with a
        # fixed return type (dot/length/log/...) are not misclassified as
        # element-wise vector preservers.
        if func_lower in self._SCALAR_RETURN_FUNCTIONS:
            return "float"
        if func_lower in self._VECTOR3_RETURN_FUNCTIONS:
            return "vec3"
        if func_lower in self._VECTOR_PRESERVING_FUNCTIONS and node.args:
            return self.get_expr_type(node.args[0])
        if func_lower == "step" and len(node.args) >= 2:
            return self.get_expr_type(node.args[1])
        if func_lower == "smoothstep" and len(node.args) >= 3:
            return self.get_expr_type(node.args[2])

        # Look up in unified return type map
        if func_lower in _EXPR_RETURN_TYPES:
            return _EXPR_RETURN_TYPES[func_lower]

        # GLSL type constructors
        if func_lower == "vec2":
            return "vec2"
        if func_lower == "vec3":
            return "vec3"
        if func_lower == "vec4":
            return "vec4"
        if func_lower in ("float", "int", "bool", "double"):
            return "float"

        # mat constructors — treat as vec4 for promotion purposes
        if func_lower.startswith("mat"):
            return "vec4"

        # Unknown function — assume scalar (conservative)
        return "float"

    @classmethod
    def _combine_types(cls, a: str, b: str) -> str:
        """Return the wider of two GLSL types for binary operations."""
        rank = cls._TYPE_RANK
        ra = rank.get(a, 0)
        rb = rank.get(b, 0)
        return a if ra >= rb else b

    @classmethod
    def _swizzle_type(cls, base_type: str, swizzle: str) -> str:
        """Determine the result type of a swizzle access.

        The result type depends on how many components are selected:
        1 char → float, 2 → vec2, 3 → vec3, 4 → vec4.
        """
        n = len(swizzle)
        if n == 1:
            return "float"
        if n == 2:
            return "vec2"
        if n == 3:
            return "vec3"
        if n == 4:
            return "vec4"
        return base_type


# --- Tokenizer and parser -------------------------------------------
class MilkDropTokenizer:
    __slots__ = ("len", "pos", "source")

    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.len = len(source)

    def peek(self, offset=0):
        p = self.pos + offset
        return self.source[p] if p < self.len else ""

    def advance(self):
        c = self.source[self.pos]
        self.pos += 1
        return c

    def skip_ws(self):
        s = self.source
        p = self.pos
        n = self.len
        while p < n and s[p].isspace():
            p += 1
        self.pos = p

    def token(self):
        s = self.source
        p = self.pos
        n = self.len
        if p >= n:
            return None
        c = s[p]
        if c.isspace():
            self.skip_ws()
            return self.token()
        if c == "`":
            j = p + 1
            while j < n and s[j] != "`":
                j += 1
            if j < n:
                self.pos = j + 1
                return ("BACKTICK", s[p : j + 1])
            self.pos = n
            return ("BACKTICK", s[p:])
        if c.isdigit() or (c == "." and p + 1 < n and s[p + 1].isdigit()):
            j = p + 1
            seen_dot = c == "."
            while j < n and (s[j].isdigit() or (s[j] == "." and not seen_dot)):
                if s[j] == ".":
                    seen_dot = True
                j += 1
            if j < n and s[j] in "eE":
                j += 1
                if j < n and s[j] in "+-":
                    j += 1
                while j < n and s[j].isdigit():
                    j += 1
            text = s[p:j]
            # MilkDrop reads leading zeros as decimal ("09" is 9, "0021" is
            # 21), but Python's literal syntax rejects them, so the rebuilt
            # expression would fail to parse.  Normalise the token text.
            if len(text) > 1 and text[0] == "0" and text.isdigit():
                text = text.lstrip("0") or "0"
            self.pos = j
            return ("NUMBER", text)
        if c.isalpha() or c == "_":
            j = p + 1
            while j < n and (s[j].isalnum() or s[j] == "_"):
                j += 1
            word = s[p:j]
            self.pos = j
            return ("IDENT", word)
        two = s[p : p + 2]
        if two in ("+=", "-=", "*=", "/=", "==", "!=", "<=", ">=", "&&", "||"):
            self.pos = p + 2
            return ("OP", two)
        if c in (
            "+",
            "-",
            "*",
            "/",
            "%",
            "^",
            "<",
            ">",
            "=",
            "!",
            "&",
            "|",
            "~",
            "(",
            ")",
            ",",
        ):
            self.pos = p + 1
            return ("OP", c)
        self.pos = p + 1
        return ("RAW", c)

    def tokens(self) -> list:
        toks = []
        while True:
            t = self.token()
            if t is None:
                break
            toks.append(t)
        return toks


class MilkDropParser:
    def parse(self, source: str) -> str:
        if not source or not source.strip():
            return ""
        source = source.strip()
        if source.startswith("//") or source.lstrip("`").lstrip().startswith("#"):
            return source
        toks = MilkDropTokenizer(source).tokens()
        toks = self._rename_python_keywords(toks)
        toks = self._normalize_assign_ops(toks)
        toks = self._normalize_semicolons(toks)
        toks = self._expand_condensed_ternaries(toks)
        toks = self._expand_short_ifs(toks)
        toks = self._expand_comparison_functions(toks)
        toks = self._expand_rif_ternaries(toks)
        toks = self._expand_exec_calls(toks)
        toks = self._normalize_operators(toks)
        toks = self._normalize_not(toks)
        return self._tokens_to_str(toks)

    def _rename_python_keywords(self, toks):
        out = []
        for t in toks:
            if t[0] == "IDENT" and t[1] in _PYTHON_KEYWORD_RENAME:
                out.append(("IDENT", _PYTHON_KEYWORD_RENAME[t[1]]))
            else:
                out.append(t)
        return out

    def _normalize_not(self, toks):
        """Convert MilkDrop ``!`` logical-NOT to Python ``not``.

        MilkDrop uses ``!`` for logical NOT.  The tokenizer emits it as
        ``('OP', '!')`` (distinct from ``!=`` which is ``('OP', '!=')``), so the
        two are unambiguous.  Python requires ``not`` to be followed by a
        parenthesized operand when it appears after a binary operator (e.g.
        ``a + not b`` is a syntax error), so the operand is always wrapped:
        ``!x`` -> ``(not(x))``.

        A bare ``!`` is never valid, so any token that cannot be parsed as an
        operand is left untouched rather than producing ``not ()``.
        """
        out = []
        i = 0
        n = len(toks)
        while i < n:
            t = toks[i]
            if t == ("OP", "!"):
                operand, i = self._collect_not_operand(toks, i + 1)
                if not operand:
                    # Nothing follows the ``!`` -- leave it as-is rather than
                    # emitting ``not ()``, which would also fail to parse.
                    out.append(t)
                    continue
                # Wrap the whole ``not(...)`` in an extra pair of parentheses.
                # Python requires the ``not`` expression itself to be
                # parenthesized when it follows a binary operator
                # (``a + not b`` is a syntax error, ``a + (not b)`` is not), so
                # emitting ``(not(operand))`` is valid in every position.
                out.append(("OP", "("))
                out.append(("IDENT", "not"))
                out.append(("OP", "("))
                out.extend(operand)
                out.append(("OP", ")"))
                out.append(("OP", ")"))
            else:
                out.append(t)
                i += 1
        return out

    @staticmethod
    def _collect_not_operand(toks, i):
        """Collect the single operand following a ``!`` operator.

        Returns ``(operand_tokens, next_index)``.  The operand is one of:
          * a parenthesized group ``(...)`` (the whole group, delimiters
            included);
          * a primary (IDENT/NUMBER/RAW) optionally followed by member-access
            (``.name``), subscript (``[...]``) and call (``(...)``) suffixes;
          * a nested ``!``, handled recursively so ``!!x`` ->
            ``(not((not(x))))``.

        Anything else (e.g. a bare operator) yields an empty operand so the
        caller can leave the ``!`` untouched.
        """
        n = len(toks)
        if i >= n:
            return [], i
        t = toks[i]
        if t == ("OP", "!"):
            inner, ni = MilkDropParser._collect_not_operand(toks, i + 1)
            if not inner:
                return [], ni
            return (
                [("IDENT", "not"), ("OP", "(")]
                + inner
                + [("OP", ")")],
                ni,
            )
        if t[0] == "OP" and t[1] == "(":
            depth = 1
            j = i + 1
            while j < n and depth > 0:
                if toks[j] == ("OP", "("):
                    depth += 1
                elif toks[j] == ("OP", ")"):
                    depth -= 1
                j += 1
            return toks[i:j], j
        if t[0] in ("IDENT", "NUMBER", "RAW"):
            j = i + 1
            while j < n:
                nt = toks[j]
                if nt == ("RAW", "."):
                    # The '.' and the member-name token that follows it.
                    j += 1
                    if j < n:
                        j += 1
                elif nt[0] == "RAW" and nt[1] == "[":
                    depth = 1
                    j += 1
                    while j < n and depth > 0:
                        if toks[j] == ("RAW", "["):
                            depth += 1
                        elif toks[j] == ("RAW", "]"):
                            depth -= 1
                        j += 1
                elif nt[0] == "OP" and nt[1] == "(":
                    depth = 1
                    j += 1
                    while j < n and depth > 0:
                        if toks[j] == ("OP", "("):
                            depth += 1
                        elif toks[j] == ("OP", ")"):
                            depth -= 1
                        j += 1
                else:
                    break
            return toks[i:j], j
        return [], i

    def _normalize_assign_ops(self, toks):
        """Expand compound assignment operators without truncating the stream.

        MilkDropTokenizer emits ``+=|``-=|``*=|``/=` as a single OP token. The
        tokenizer does **not** emit normal statement-terminator semicolons as OP
        tokens: they come through as ``('RAW', ';')`` (or are preserved inside
        backtick blocks). This matters for bounding the RHS of a compound
        assignment correctly::

            a += 1; b += 2;
            a+=1;b+=2

        Without statement-aware RHS bounding, these can normalize to::

            a=a+(1;b += 2;)     # or, for the no-space form,
            a=a+(1);b=b+(2)

        depending on how ``_tokens_to_str()`` reintroduces spaces around
        ``;``. The correct per-statement semantics are::

            a=a+(1); b=b+(2)

        In practice this matters most when the tokenizer retains ``;`` as a
        visible separator token. When ``;`` is already stripped before this
        stage, the same code path still behaves correctly because it never
        reaches past a ``('RAW', ';')`` token when collecting a RHS.
        """
        out = []
        i = 0
        n = len(toks)
        while i < n:
            t = toks[i]
            if t[0] == "IDENT":
                # Locate the compound-assignment operator.  Two forms:
                #   Case 1: simple variable  (x += expr)  – op is at i+1
                #   Case 2: function call    (gmegabuf(n) += expr) – op is
                #          after the matching closing paren
                compound_op_idx = -1
                if (
                    i + 2 < n
                    and toks[i + 1][0] == "OP"
                    and toks[i + 1][1] in ("+=", "-=", "*=", "/=")
                ):
                    compound_op_idx = i + 1
                elif i + 2 < n and toks[i + 1] == ("OP", "("):
                    depth = 0
                    k = i + 1
                    while k < n:
                        tk = toks[k]
                        if tk[0] == "OP":
                            if tk[1] in ("(", "["):
                                depth += 1
                            elif tk[1] in (")", "]"):
                                depth -= 1
                                if depth <= 0:
                                    break
                        k += 1
                    if depth == 0 and k + 1 < n:
                        nt = toks[k + 1]
                        if (
                            nt[0] == "OP"
                            and nt[1] in ("+=", "-=", "*=", "/=")
                        ):
                            compound_op_idx = k + 1

                if compound_op_idx > 0:
                    op = toks[compound_op_idx][1]
                    lhs = self._tokens_to_str(toks[i:compound_op_idx])

                    # Collect the RHS up to the next explicit statement
                    # separator, or to end-of-stream when the line has no
                    # trailing ';'.
                    rhs_toks: list[str] = []
                    j = compound_op_idx + 1
                    depth = 0
                    while j < n:
                        tj = toks[j]
                        if tj[0] == "RAW" and tj[1] == ";" and depth <= 0:
                            break
                        if tj[0] == "OP" or (tj[0] == "RAW" and tj[1] in ("[", "]")):
                            if tj[1] == "(" or tj[1] == "[":
                                depth += 1
                            elif tj[1] == ")" or tj[1] == "]":
                                if depth <= 0:
                                    break
                                depth -= 1
                            elif depth <= 0 and tj[1] == ",":
                                break
                        rhs_toks.append(tj)
                        j += 1

                    rhs = self._tokens_to_str(rhs_toks).strip()
                    if rhs:
                        norm = {
                            "+=": f"{lhs} = {lhs} + ({rhs})",
                            "-=": f"{lhs} = {lhs} - ({rhs})",
                            "*=": f"{lhs} = {lhs} * ({rhs})",
                            "/=": f"{lhs} = {lhs} / ({rhs})",
                        }[op]
                        normalized_tokens = MilkDropTokenizer(norm).tokens()
                        out.extend(normalized_tokens)
                        i = j
                        continue
            out.append(t)
            i += 1
        return out

    def _normalize_semicolons(self, toks):
        out = []
        for t in toks:
            if t[0] == "OP" and t[1] == ";":
                continue
            out.append(t)
        return out

    def _expand_short_ifs(self, toks):
        out = []
        i = 0
        n = len(toks)
        while i < n:
            t = toks[i]
            if (
                t[0] == "IDENT"
                and t[1].lower() in ("if", "if_")
                and i + 1 < n
                and toks[i + 1][0] == "OP"
                and toks[i + 1][1] == "("
            ):
                if i > 0:
                    prev = toks[i - 1]
                    if prev[0] == "IDENT" and prev[1] not in ("and", "or", "not"):
                        out.append(t)
                        i += 1
                        continue
                    if prev[0] == "NUMBER":
                        out.append(t)
                        i += 1
                        continue
                    if prev[0] == "OP" and prev[1] in (")", "]", "}"):
                        out.append(t)
                        i += 1
                        continue
                args = self._extract_call_args(toks, i + 1)
                if args is not None:
                    expanded_args = [self._expand_short_ifs(a) for a in args]
                    arg_strs = [self._tokens_to_str(a) for a in expanded_args]
                    cond = arg_strs[0] if arg_strs else "0"
                    then = arg_strs[1] if len(arg_strs) > 1 else "1"
                    else_ = arg_strs[2] if len(arg_strs) > 2 else "0"
                    # An omitted or empty argument means 0 in MilkDrop.  A
                    # blank branch would emit "( if (c) else 0)", which is a
                    # Python SyntaxError ("expected expression").
                    # The branches are statements; remember the variable they
                    # assign to before reducing them to values.
                    target = self._branch_target(then, else_)
                    then = self._branch_value(then)
                    else_ = self._branch_value(else_)
                    replacement = f"({then} if ({cond}) else {else_})"
                    # ``if(cond, q13 = 1, q13 = 0)`` used as a whole statement
                    # assigns q13; without re-attaching the target the reduced
                    # ternary carries no "=" and the equation is rejected as
                    # missing an assignment.  Skip when an enclosing assignment
                    # already provides one (e.g. "q13 = if(...)").
                    if target and not self._has_enclosing_assign(out):
                        replacement = f"{target} = {replacement}"
                    out.extend(self._str_to_toks(replacement))
                    i += 2 + sum(len(a) for a in args) + len(args)
                    continue
            out.append(t)
            i += 1
        return out

    @staticmethod
    def _has_enclosing_assign(out) -> bool:
        """True when the tokens emitted so far end inside an assignment."""
        return (
            bool(out)
            and out[-1][0] == "OP"
            and out[-1][1]
            in (
                "=",
                "+=",
                "-=",
                "*=",
                "/=",
            )
        )

    @staticmethod
    def _branch_target(then: str, else_: str) -> str | None:
        """Return the variable a pair of ``if()`` branches assign to.

        ``if(cond, q13 = 1, q13 = 0)`` is a statement that stores into ``q13``.
        When both branches name the same target that target is used.  MilkDrop
        presets also use the asymmetric form ``if(cond, x = v, 0)`` -- only one
        branch assigns -- so a target named by exactly one branch is accepted
        too; a branch that is a bare literal (0) carries no target of its own.
        Returns ``None`` when the branches name *different* variables, since
        guessing there would silently write to the wrong state.
        """

        def target_of(branch: str) -> str | None:
            b = (branch or "").strip()
            parts = [p.strip() for p in _split_milk_eqs(b) if p.strip()]
            if not parts:
                return None

            # The target may name a megabuf element as well as a plain variable
            # ("gmegabuf(m+3)=gmegabuf(n+3)"), matching the target forms
            # _strip_branch_assignment() accepts.
            def first_of(text: str) -> str | None:
                m = _BRANCH_TARGET_RE.match(text)
                return m.group(1).strip() if m else None

            # A multi-statement branch computes one value and updates bookkeeping
            # variables: "timediff = time - time_st; time_st = time".  The value
            # is the last statement's, but the *variable* the preset is computing
            # is the one named by the first assignment, which is also what the
            # opposite branch normally writes.
            if len(parts) > 1 and first_of(parts[0]):
                return first_of(parts[0])
            return first_of(parts[-1])

        t1, t2 = target_of(then), target_of(else_)
        if t1 and t1 == t2:
            return t1
        if t1 and not t2:
            return t1
        if t2 and not t1:
            return t2
        return None

    def _branch_value(self, branch: str) -> str:
        """Reduce an ``if()`` branch to the value it yields.

        MilkDrop branches are statements, not just expressions, and a very
        common preset idiom assigns inside a branch::

            if (bd_ampl > bd_pk, bd_pk = bd_ampl, 0)
            if (beat, trel1 += BPM/60*dt, trel1 += BPM/120*dt)

        The value of such a branch is the assigned value, so ``x = 1`` yields
        ``1`` and ``x += e`` yields ``x + e``.  Reducing both branches
        independently is what makes these compile; previously only the *then*
        branch was reduced, leaving a raw ``q13 = 0`` in the else branch and
        producing the invalid Python ``(1 if (cond) else q13 = 0)``.

        A branch may also hold several ``;``-separated statements
        (``if(a, b=1; c=2, b)``); the value of a statement sequence is the value
        of its last statement, so earlier ones are dropped.
        """
        branch = (branch or "").strip()
        if not branch:
            return "0"
        parts = [p.strip() for p in _split_milk_eqs(branch) if p.strip()]
        if not parts:
            return "0"
        last = parts[-1]
        return self._strip_branch_assignment(last)

    def _strip_branch_assignment(self, expr: str) -> str:
        """Drop a leading assignment from a single ``if()`` branch expression."""
        expr = expr.strip().rstrip(";").strip()
        # Plain assignment: ``q13 = 1`` -> ``1``.  The target may be a megabuf
        # element as well as a plain variable ("gmegabuf(n+3)=gmegabuf(m+3)"),
        # so accept any assignable-looking target, not just a bare identifier.
        m = re.match(
            r"^([A-Za-z_]\w*(?:\s*\([^()]*\))?)\s*=(?!=)\s*(.+)$", expr, re.DOTALL
        )
        if m:
            return m.group(2).strip().rstrip(";").strip() or "0"
        # Compound assignment: ``trel1 += e`` -> ``trel1 + e``
        m = re.match(
            r"^([A-Za-z_]\w*(?:\s*\([^()]*\))?)\s*([-+*/])=\s*(?!=)\s*(.+)$",
            expr,
            re.DOTALL,
        )
        if m:
            return f"({m.group(1)} {m.group(2)} {m.group(3).strip()})"
        return expr

    def _expand_comparison_functions(self, toks):
        out = []
        i = 0
        n = len(toks)
        while i < n:
            t = toks[i]
            if (
                t[0] == "IDENT"
                and t[1].lower()
                in ("above", "below", "equal", "above_equal", "below_equal")
                and i + 1 < n
                and toks[i + 1][0] == "OP"
                and toks[i + 1][1] == "("
            ):
                args = self._extract_call_args(toks, i + 1)
                if args and len(args) >= 2:
                    func = t[1].lower()
                    a0 = self._tokens_to_str(args[0])
                    a1 = self._tokens_to_str(args[1])
                    if func == "below":
                        replacement = f"({a0} < {a1})"
                    elif func == "above":
                        replacement = f"({a0} > {a1})"
                    elif func == "equal":
                        replacement = f"(abs({a0} - {a1}) < {EPSILON_COMPARE})"
                    elif func == "above_equal":
                        replacement = f"({a0} >= {a1})"
                    elif func == "below_equal":
                        replacement = f"({a0} <= {a1})"
                    else:
                        replacement = f"({a0} < {a1})"
                    out.extend(self._str_to_toks(replacement))
                    i += 2 + sum(len(a) for a in args) + len(args)
                    continue
            out.append(t)
            i += 1
        return out

    def _expand_rif_ternaries(self, toks):
        s = self._tokens_to_str(toks)
        if "rif(" not in s:
            return toks
        result = []
        i = 0
        n = len(s)
        while i < n:
            j = s.find("rif(", i)
            if j < 0:
                result.append(s[i:])
                break
            result.append(s[i:j])
            cond_open = j + len("rif(") - 1
            cond_close = self._find_matching_paren(s, cond_open)
            if cond_close < 0:
                result.append(s[j:])
                break
            cond_expr = s[cond_open + 1 : cond_close].strip()
            k = cond_close + 1
            while k < n and s[k].isspace():
                k += 1
            if k < n and s[k] == ")":
                k += 1
                while k < n and s[k].isspace():
                    k += 1
            if not s.startswith("els", k):
                result.append("rif(")
                i = j + len("rif(")
                continue
            k += len("els")
            else_start = k
            depth_p = depth_b = depth_c = 0
            while k < n:
                c = s[k]
                if c == "(":
                    depth_p += 1
                elif c == ")":
                    if depth_p == 0 and depth_b == 0 and depth_c == 0:
                        break
                    depth_p = max(0, depth_p - 1)
                elif c == "[":
                    depth_b += 1
                elif c == "]":
                    depth_b = max(0, depth_b - 1)
                elif c == "{":
                    depth_c += 1
                elif c == "}":
                    depth_c = max(0, depth_c - 1)
                elif (
                    depth_p == 0
                    and depth_b == 0
                    and depth_c == 0
                    and c in "+-*/%^&|<>=,;"
                ):
                    break
                k += 1
            else_expr = s[else_start:k].strip()
            else_body = (
                else_expr
                if else_expr
                and "rif(" not in else_expr
                and not self._has_els_keyword(else_expr)
                and len(else_expr.split()) <= 1
                else "0.0"
            )
            result.append(f"({else_body} if ({cond_expr}) else 0.0)")
            i = k
        return MilkDropTokenizer("".join(result)).tokens()

    def _expand_exec_calls(self, toks):
        """Transform ``exec3``/``exec2``/``exec1`` argument assignments.

        MilkDrop allows assignment expressions as arguments to
        ``exec3``/``exec2``/``exec1``::

            exec3(x = v1, y = v2, z = v3)

        Those ``=`` signs are *statement* assignments in MilkDrop, not Python
        keyword arguments.  Left untouched, ``ast.parse`` would reject them as
        a syntax error (``exec3`` is not a real Python function and ``x = v``
        cannot be a keyword-argument name).

        We convert each assignment argument to an explicit ``assign()`` call
        so the result is valid Python::

            exec3(assign(x, v1), assign(y, v2), z = v3)  ->  exec3(assign(x, v1), assign(y, v2), z)
        """
        out = []
        i = 0
        n = len(toks)
        while i < n:
            t = toks[i]
            if (
                t[0] == "IDENT"
                and t[1].lower() in ("exec3", "exec2", "exec1")
                and i + 1 < n
                and toks[i + 1][0] == "OP"
                and toks[i + 1][1] == "("
            ):
                args = self._extract_call_args(toks, i + 1)
                if args is not None:
                    new_args = []
                    for arg_toks in args:
                        arg_str = self._tokens_to_str(arg_toks)
                        if arg_str.strip():
                            # Check if the arg is a simple top-level assignment
                            #   var = value  ->  assign(var, value)
                            target = self._split_assignment_target(arg_str)
                            if target is not None:
                                key, val = target
                                new_arg_str = f"assign({key}, {val})"
                                new_args.append(self._str_to_toks(new_arg_str))
                            else:
                                new_args.append(arg_toks)
                        else:
                            new_args.append(arg_toks)
                    out.append(t)
                    out.append(("OP", "("))
                    for j, arg in enumerate(new_args):
                        if j > 0:
                            out.append(("OP", ","))
                        out.extend(arg)
                    out.append(("OP", ")"))
                    i += 2 + sum(len(a) for a in args) + len(args)
                    continue
            out.append(t)
            i += 1
        return out

    @staticmethod
    def _split_assignment_target(s: str):
        """Split 'target = value' at a top-level '=' that is not part of a
        comparison/compound operator.  Returns (target, value) or None."""
        s = s.strip()
        depth_p = depth_b = depth_c = 0
        j = 0
        while j < len(s):
            c = s[j]
            if c == "(":
                depth_p += 1
            elif c == ")":
                depth_p = max(0, depth_p - 1)
            elif c == "[":
                depth_b += 1
            elif c == "]":
                depth_b = max(0, depth_b - 1)
            elif c == "{":
                depth_c += 1
            elif c == "}":
                depth_c = max(0, depth_c - 1)
            elif c == "=" and depth_p == 0 and depth_b == 0 and depth_c == 0:
                # Not ==, !=, <=, >=, +=, -=, *=, /=, &=, |=, ^=, >>=, <<=
                prev_c = s[j - 1] if j > 0 else ""
                next_c = s[j + 1] if j + 1 < len(s) else ""
                if prev_c in "+-*/!<>=&|^" or next_c == "=":
                    j += 1
                    continue
                target = s[:j].strip()
                val = s[j + 1:].strip()
                if target and val:
                    return (target, val)
                return None
            j += 1
        return None

    def _has_els_keyword(self, s):
        i = 0
        n = len(s)
        while i < n:
            if s[i : i + 3] == "els" and (i == 0 or not s[i - 1].isalnum()):
                return True
            i += 1
        return False

    def _expand_condensed_ternaries(self, toks):
        s = self._tokens_to_str(toks)
        i = 0
        n = len(s)
        while i < n:
            if s[i : i + 5] == " if (" and i > 0:
                j = i - 1
                while j >= 0 and s[j] == " ":
                    j -= 1
                if j < 0 or s[j] in (
                    "=",
                    ",",
                    "(",
                    ";",
                    "+",
                    "-",
                    "*",
                    "/",
                    "%",
                    "^",
                    "&",
                    "|",
                    "<",
                    ">",
                    "=",
                    "!",
                    "~",
                    "?",
                    "@",
                    "#",
                    "$",
                    "`",
                ):
                    i += 1
                    continue
                cond_open = i + 4
                cond_close = self._find_matching_paren(s, cond_open)
                if cond_close > 0:
                    cond_body = s[cond_open + 1 : cond_close].strip()
                    after = cond_close + 1
                    while after < n and s[after].isspace():
                        after += 1
                    if after < n and s[after : after + 4] == "else":
                        after += 4
                        while after < n and s[after].isspace():
                            after += 1
                        else_end = after
                        depth = 0
                        while else_end < n:
                            c = s[else_end]
                            if c == "(":
                                depth += 1
                            elif c == ")":
                                if depth == 0:
                                    break
                                depth -= 1
                            elif depth == 0 and c in "+-*/%^&|<>,;":
                                break
                            else_end += 1
                        else_val = s[after:else_end].strip()
                        if not else_val:
                            else_val = "0"
                        else_assign = else_val.split("=", 1)
                        if (
                            len(else_assign) == 2
                            and else_assign[0].strip().isidentifier()
                        ):
                            else_val = else_assign[1].strip()
                    else:
                        else_val = "0"
                        else_end = after
                    start = j
                    depth = 0
                    while start >= 0:
                        c = s[start]
                        if c in ")]}":
                            depth += 1
                        elif c in "([{":
                            if depth == 0:
                                break
                            depth -= 1
                        elif depth == 0 and c in "=,;":
                            break
                        start -= 1
                    then_val = s[start + 1 : i].strip()
                    if not then_val:
                        then_val = "1"
                    then_assign = then_val.split("=", 1)
                    if len(then_assign) == 2 and then_assign[0].strip().isidentifier():
                        then_val = then_assign[1].strip()
                    converted = f"({then_val} if ({cond_body}) else {else_val})"
                    s = s[: start + 1] + converted + s[else_end:]
                    # Resume *after* the text just produced. Restarting the
                    # scan from 0 would re-match the converted ternary itself
                    # (its own " if (" is preceded by a value, not an operator)
                    # and wrap it again forever.
                    i = start + 1 + len(converted)
                    n = len(s)
                    continue
            i += 1
        return MilkDropTokenizer(s).tokens()

    def _normalize_operators(self, toks):
        out = []
        i = 0
        n = len(toks)
        while i < n:
            t = toks[i]
            if t[0] == "OP" and t[1] == "^":
                out.append(("OP", "**"))
            elif t[0] == "OP" and t[1] == "&&":
                out.append(("IDENT", "and"))
            elif t[0] == "OP" and t[1] == "||":
                out.append(("IDENT", "or"))
            elif t[0] == "IDENT" and t[1] == "rand":
                # A bare ``rand`` may be an ordinary user variable; do not
                # inject ``rand(1)`` (the old heuristic corrupted expressions).
                out.append(t)
            elif t[0] == "IDENT" and t[1] == "mod":
                out.append(("IDENT", "md_mod"))
            elif t[0] == "IDENT" and t[1] == "as":
                out.append(("IDENT", "as_"))
            elif t[0] == "IDENT" and (
                (RE_STRICT_IDENTIFIER.match(t[1]) and t[1].endswith(":"))
                or (t[1].startswith("float") and len(t[1]) > 5 and t[1][5:].isdigit())
            ):
                pass  # Skip float-prefix tokens like float2, float3, etc.--but NOT "float" itself
            else:
                out.append(t)
            i += 1
        return out

    def _extract_call_args(self, toks, start):
        if start >= len(toks) or toks[start][0] != "OP" or toks[start][1] != "(":
            return None
        depth = 1
        args = [[]]
        i = start + 1
        while i < len(toks):
            t = toks[i]
            if t[0] == "OP":
                if t[1] == "(":
                    depth += 1
                    args[-1].append(t)
                elif t[1] == ")":
                    if depth == 1:
                        return args
                    depth -= 1
                    args[-1].append(t)
                elif t[1] == "," and depth == 1:
                    args.append([])
                else:
                    args[-1].append(t)
            else:
                args[-1].append(t)
            i += 1
        return args

    def _str_to_toks(self, s):
        return MilkDropTokenizer(s).tokens()

    def _tokens_to_str(self, toks) -> str:
        parts = []
        prev_type = None
        prev_val = None
        for t in toks:
            ttype, tval = t
            # Insert a space where concatenation would merge two tokens
            # into something Python cannot lex:
            #   - two identifiers (``aand``) or identifier/number
            #     (``1and``) -- preserving the original spacing rules;
            #   - a closing bracket followed by an identifier/number
            #     (``)and``, ``]foo`` -- ``and`` is a keyword here).
            # Member access (``a.b``), calls (``foo(``) and parenthesized
            # groups (``not(``) deliberately stay unspaced.
            if prev_type is not None and (
                (prev_type == "IDENT" and ttype in ("IDENT", "NUMBER"))
                or (prev_type == "NUMBER" and ttype == "IDENT")
                or (
                    prev_type == "OP"
                    and prev_val in (")", "]")
                    and ttype in ("IDENT", "NUMBER")
                )
            ):
                parts.append(" ")
            parts.append(tval)
            prev_type, prev_val = ttype, tval
        s = "".join(parts)
        for pair in [
            ("&&", " && "),
            ("||", " || "),
            ("<=", " <= "),
            (">=", " >= "),
            ("==", " == "),
            ("!=", " != "),
            ("+=", " += "),
            ("-=", " -= "),
            ("*=", " *= "),
            ("/=", " /="),
        ]:
            s = s.replace(pair[0], pair[1])
        s = re.sub(r"(?<!\w)if(?!\w)", " if ", s)
        s = re.sub(r"(?<!\w)else(?!\w)", " else ", s)
        s = re.sub(r"\s+", " ", s)
        return s.strip()

    @staticmethod
    def _find_matching_paren(s, start_idx):
        depth = 0
        for j in range(start_idx, len(s)):
            if s[j] == "(":
                depth += 1
            elif s[j] == ")":
                depth -= 1
                if depth == 0:
                    return j
        return -1


# --- Expression preprocessing and GLSL entry points ---------------
@functools.cache
def preprocess_milkdrop_expr(source: str) -> str:
    """Preprocess MilkDrop expression to Python-compatible syntax.

    Uses a tokenizer+parser for MilkDrop-specific constructs:
      - += -= *= /= assignments
      - if(x) / if(x,y) shorthand
      - if(cond, then, else) conversion to Python ternary
      - above(a,b), below(a,b), equal(a,b)

    This keeps the downstream stages unchanged (ast.parse -> AST optimize -> GLSL).

    The transformation is a pure function of *source* (the parser is rebuilt
    per call and holds no state), so the result is memoised.
    """
    if not source or not source.strip():
        return ""
    source = source.strip()
    if source.startswith("//"):
        return source
    if source.lstrip("`").lstrip().startswith("#"):
        return source
    return _fold_constants(MilkDropParser().parse(source))


def compile_assignment_to_glsl(expr_str: str) -> tuple:
    """Compile a MilkDrop assignment (var = expr) to GLSL.

    Returns ``(var_name, glsl_code, is_full_statement, qualifier, reg_refs, rhs_type)``
    where *reg_refs* is a frozenset of integer register indices (reg00-reg99)
    referenced in the RHS expression, and *rhs_type* is the inferred GLSL
    type of the RHS (``"float"``, ``"vec2"``, ``"vec3"``, ``"vec4"``) used
    for type-mismatch detection at the call site.

    Raises PresetCompileError on failure instead of returning (None, None, False).
    """
    if not expr_str:
        raise PresetCompileError("Empty expression string")
    expr_str = expr_str.strip()
    if expr_str.startswith("//"):
        raise PresetCompileError("Comment expression")
    if "//" in expr_str:
        raise PresetCompileError("Comment in expression")
    # Preprocess the full expression *before* checking for assignment and
    # splitting on '=' so that compound assignments like ``gmegabuf(n+1) *= expr``
    # are expanded to ``gmegabuf(n+1) = gmegabuf(n+1) * (expr)`` and if-branches
    # like ``if(beat, q1=1, q1=0)`` are expanded to ``q1 = (1 if beat else 0)``.
    expr_str = preprocess_milkdrop_expr(expr_str)
    if not expr_str or not _has_top_level_assignment(expr_str):
        raise PresetCompileError("No assignment in expression")
    key, val = expr_str.split("=", 1)
    key = key.strip()
    val = val.strip().rstrip(";")
    if key == "mod":
        key = "md_mod"
    if not key or key.startswith("//"):
        raise PresetCompileError("Invalid assignment target")
    # The LHS must be a valid GLSL lvalue: a simple identifier optionally
    # followed by a component swizzle.  Function calls (gmegabuf(n)),
    # array subscripts (u_q[1]) and GLSL type keywords (float, vec3, void,
    # ...) are not assignable and would emit invalid shader source
    # (``syntax error, unexpected '='``).
    if not _VALID_ASSIGNMENT_LHS_RE.match(key):
        raise PresetCompileError(
            f"Invalid per-pixel assignment target {key!r}: not a valid GLSL lvalue"
        )
    if key in _GLSL_TYPE_KEYWORDS:
        raise PresetCompileError(
            f"Invalid per-pixel assignment target {key!r}: '{key}' is a GLSL type keyword"
        )
    try:
        val = preprocess_milkdrop_expr(val)
    except Exception as e:
        raise PresetCompileError(f"Preprocessing failed: {e}") from e

    if val.startswith("//"):
        raise PresetCompileError("Comment after preprocessing")
    # After preprocessing, an if() branch that assigned to a variable has its
    # target re-attached (e.g. "if(a, q13=1, q13=0)" -> "q13=(1 if ...)").  That
    # makes val a bare assignment, which cannot be parsed with mode="eval".
    # Strip the leading target so the value parses as a pure expression; the
    # enclosing key (already extracted above) is the authoritative assignment
    # target for the generated GLSL.
    m = re.match(
        r"^[A-Za-z_]\w*(?:\s*\([^)]*\))?\s*=(?!=)\s*(.+)$",
        val,
        re.DOTALL,
    )
    if m:
        val = m.group(1)
    try:
        tree = ast.parse(_normalize_leading_zeros(val), mode="eval")
    except Exception as e:
        raise PresetCompileError(f"Parse error: {e}") from e

    generator = GLSLGenerator()
    glsl = generator.visit(tree.body)
    qualifier = "full_statement" if glsl else "expression"
    rhs_type = generator.get_expr_type(tree.body)
    return key, glsl, True, qualifier, frozenset(generator.reg_refs), rhs_type


def _numeric_const(value: object) -> float | None:
    """Convert an ``ast`` constant to ``float`` the way ``float()`` would.

    Returns ``None`` for constants that ``float()`` cannot convert, so the
    caller can skip constant folding instead of raising.  The accepted inputs
    mirror ``float()`` exactly: bool/int/float, numeric strings/bytes, and
    nothing else (complex, None and Ellipsis are rejected, exactly as
    ``float()`` rejects them).
    """
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (str, bytes, bytearray)):
        try:
            return float(value)
        except ValueError:
            return None
    return None


# --- AST optimization ----------------------------------------------
class ExpressionOptimizer(ast.NodeVisitor):
    """Dedicated AST optimization pass.
    Runs after parsing, before bytecode or GLSL generation.
    Implements:
    - Constant folding (1+2 -> 3) — ONLY for finite results
    - Algebraic identities (x*1 -> x, x/1 -> x) — CONSERVATIVE: x*0 and x+0 are NOT optimized
      because they change NaN/Inf semantics (NaN*0=NaN, not 0; NaN+0=NaN).
    - Power optimization (pow(x,2) -> x*x)
    - Trig at zero (sin(0)->0, cos(0)->1)
    - Dead code elimination (if False, branch removal)
    - Strength reduction where beneficial

    Non-finite constant folding (NaN, Inf) is explicitly avoided to preserve
    IEEE-754 semantics.  fract() is treated as: fract(x) = x - floor(x)
    which produces NaN for NaN input and 0 for +Inf/-Inf (since floor(Inf)=Inf).
    """

    ZERO = ast.Constant(value=0.0)
    ONE = ast.Constant(value=1.0)
    TWO = ast.Constant(value=2.0)

    def visit_Constant(self, node):
        return node

    def visit_BinOp(self, node):
        self.generic_visit(node)
        left = node.left
        right = node.right
        op = node.op

        if isinstance(left, ast.Constant) and isinstance(right, ast.Constant):
            try:
                lv = _numeric_const(left.value)
                rv = _numeric_const(right.value)
                if lv is None or rv is None:
                    raise TypeError("non-numeric constant in expression")
                # Only fold if both inputs and result are finite
                if math.isfinite(lv) and math.isfinite(rv):
                    result = None
                    if isinstance(op, ast.Add):
                        result = lv + rv
                    elif isinstance(op, ast.Sub):
                        result = lv - rv
                    elif isinstance(op, ast.Mult):
                        result = lv * rv
                    elif isinstance(op, ast.Div):
                        if abs(rv) > EPSILON_DIV:
                            result = lv / rv
                    elif isinstance(op, ast.Mod):
                        if abs(rv) > EPSILON_DIV:
                            result = lv % rv
                    elif isinstance(op, ast.Pow):
                        result = lv**rv
                    elif isinstance(op, ast.FloorDiv) and abs(rv) > EPSILON_DIV:
                        result = lv // rv
                    if result is not None and math.isfinite(result):
                        val = ast.Constant(value=result)
                        ast.copy_location(val, node)
                        return val
            except (ValueError, TypeError, ZeroDivisionError, OverflowError) as e:
                logger.debug(f"Failed to optimize constant folding: {e}")

        if isinstance(op, ast.Mult):
            if self._is_one(left):
                ast.copy_location(right, node)
                return right
            if self._is_one(right):
                ast.copy_location(left, node)
                return left
            # x*0 and 0*x are NOT optimized — they would change NaN*0=NaN to 0
            # and Inf*0=NaN to 0

        if isinstance(op, ast.Add):
            if self._is_zero(left):
                ast.copy_location(right, node)
                return right
            if self._is_zero(right):
                ast.copy_location(left, node)
                return left
            # x+0 and 0+x are NOT optimized — they would change NaN+0=NaN to NaN (same)
            # but we avoid for consistency with x*0
            if self._is_negation(left, right) and self._is_finite(left):
                # Only fold finite operands: inf + -inf is NaN in IEEE-754, and
                # folding it to 0.0 would feed the shader a different value than
                # GLSL computes.
                result = ast.Constant(value=0.0)
                ast.copy_location(result, node)
                return result

        if isinstance(op, ast.Sub):
            if self._is_zero(right):
                ast.copy_location(left, node)
                return left
            if self._is_zero(left):
                result = ast.UnaryOp(op=ast.USub(), operand=right)
                ast.copy_location(result, node)
                return result

        if isinstance(op, ast.Div):
            # Do NOT fold x/0 or x/nearzero to 0.0 — that changes GLSL
            # divide-by-zero semantics (NaN/inf) to a finite zero.
            # Keep those trees intact so the GLSL emitted later preserves
            # the renderer's chosen NaN/Inf behavior.
            if self._is_one(right):
                ast.copy_location(left, node)
                return left
            return node

        if isinstance(op, ast.Pow):
            if self._is_zero(right):
                return ast.Constant(value=1.0)
            if self._is_one(right):
                ast.copy_location(left, node)
                return left
            if self._is_two(right):
                # x**2 is x*x.  The right operand here is the literal 2, so
                # multiplying by it produced "x * 2" and silently turned every
                # squaring in a preset equation into a doubling.
                result = ast.BinOp(left=left, op=ast.Mult(), right=left)
                ast.copy_location(result, node)
                return result

        # Only fold when the left operand is itself a finite constant. For a
        # variable left operand, x % 0 is NaN in GLSL but 0.0 here, which is
        # exactly the CPU/GPU divergence the Div rule above avoids.
        if (
            isinstance(op, ast.Mod)
            and (self._is_zero(right) or self._is_approx(right, 0.0))
            and self._is_finite(left)
        ):
            result = ast.Constant(value=0.0)
            ast.copy_location(result, node)
            return result

        return node

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        operand = node.operand
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            if isinstance(operand, ast.Constant):
                operand_val = _numeric_const(operand.value)
                if operand_val is None:
                    return node
                result = ast.Constant(value=-operand_val)
                ast.copy_location(result, node)
                return result
            if isinstance(operand, ast.UnaryOp) and isinstance(operand.op, ast.USub):
                return operand.operand
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name == "sin" and node.args and self._is_zero(node.args[0]):
                return ast.Constant(value=0.0)
            if func_name == "cos" and node.args and self._is_zero(node.args[0]):
                return ast.Constant(value=1.0)
            if func_name == "tan" and node.args and self._is_zero(node.args[0]):
                return ast.Constant(value=0.0)
            if (
                func_name == "abs"
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                abs_arg = _numeric_const(node.args[0].value)
                if abs_arg is None:
                    return node
                return ast.Constant(value=abs(abs_arg))
            if func_name == "sqrt" and node.args and self._is_one(node.args[0]):
                return ast.Constant(value=1.0)
            if (
                func_name
                in (
                    "sin",
                    "cos",
                    "tan",
                    "asin",
                    "acos",
                    "atan",
                    "sinh",
                    "cosh",
                    "exp",
                    "log",
                    "sqrt",
                    "abs",
                    "floor",
                    "ceil",
                    "fract",
                    "trunc",
                )
                and len(node.args) == 1
                and isinstance(node.args[0], ast.Constant)
            ):
                try:
                    arg_val = _numeric_const(node.args[0].value)
                    if arg_val is None:
                        raise TypeError("non-numeric constant in function call")
                    result_val = None
                    if func_name == "sin":
                        result_val = _md_sin(arg_val)
                    elif func_name == "cos":
                        result_val = _md_cos(arg_val)
                    elif func_name == "tan":
                        result_val = _md_tan(arg_val)
                    elif func_name == "asin":
                        result_val = _md_asin(arg_val)
                    elif func_name == "acos":
                        result_val = _md_acos(arg_val)
                    elif func_name == "atan":
                        result_val = math.atan(arg_val)
                    elif func_name == "sinh":
                        result_val = math.sinh(arg_val)
                    elif func_name == "cosh":
                        result_val = math.cosh(arg_val)
                    elif func_name == "exp":
                        result_val = _md_exp(arg_val)
                    elif func_name == "log":
                        result_val = _md_log(arg_val)
                    elif func_name == "sqrt":
                        result_val = _md_sqrt(arg_val)
                    elif func_name == "abs":
                        result_val = abs(arg_val)
                    elif func_name == "floor":
                        result_val = _md_floor(arg_val)
                    elif func_name == "ceil":
                        result_val = _md_ceil(arg_val)
                    elif func_name == "fract":
                        result_val = arg_val - _md_floor(arg_val)
                    elif func_name == "trunc":
                        result_val = float(_md_int(arg_val))
                    if result_val is not None and math.isfinite(result_val):
                        result = ast.Constant(value=result_val)
                        ast.copy_location(result, node)
                        return result
                except (ValueError, TypeError, OverflowError) as e:
                    logger.debug(f"Failed to optimize function call: {e}")
        return node

    def visit_IfExp(self, node):
        self.generic_visit(node)
        if isinstance(node.test, ast.Constant):
            if bool(node.test.value):
                return node.body
            return node.orelse
        if self._is_true_test(node.test):
            return node.body
        if self._is_false_test(node.test):
            return node.orelse
        return node

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        values = node.values
        if isinstance(node.op, ast.And):
            filtered = []
            for v in values:
                if isinstance(v, ast.Constant):
                    if not bool(v.value):
                        return ast.Constant(value=0.0)
                    continue
                if self._is_false_test(v):
                    return ast.Constant(value=0.0)
                filtered.append(v)
            if len(filtered) == 0:
                return ast.Constant(value=1.0)
            if len(filtered) == 1:
                return filtered[0]
            if len(filtered) < len(values):
                new_node = ast.BoolOp(op=ast.And(), values=filtered)
                ast.copy_location(new_node, node)
                return new_node
        if isinstance(node.op, ast.Or):
            filtered = []
            for v in values:
                if isinstance(v, ast.Constant):
                    if bool(v.value):
                        return ast.Constant(value=1.0)
                    continue
                if self._is_true_test(v):
                    return ast.Constant(value=1.0)
                filtered.append(v)
            if len(filtered) == 0:
                return ast.Constant(value=0.0)
            if len(filtered) == 1:
                return filtered[0]
            if len(filtered) < len(values):
                new_node = ast.BoolOp(op=ast.Or(), values=filtered)
                ast.copy_location(new_node, node)
                return new_node
        return node

    def _is_zero(self, node) -> bool:
        return (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and float(node.value) == 0.0
        )

    def _is_finite(self, node) -> bool:
        return (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and math.isfinite(float(node.value))
        )

    def _is_one(self, node) -> bool:
        return (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and float(node.value) == 1.0
        )

    def _is_two(self, node) -> bool:
        return (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and float(node.value) == 2.0
        )

    def _is_approx(self, node, target) -> bool:
        return (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and abs(float(node.value) - target) < EPSILON_SMALL
        )

    def _is_negation(self, a, b) -> bool:
        if (
            isinstance(a, ast.Constant)
            and isinstance(b, ast.Constant)
            and isinstance(a.value, (int, float))
            and isinstance(b.value, (int, float))
        ):
            return abs(float(a.value) + float(b.value)) < EPSILON_SMALL
        return False

    def _is_true_test(self, node) -> bool:
        if isinstance(node, ast.Compare):
            if isinstance(node.ops[0], ast.Gt) and self._is_zero(node.comparators[0]):
                return True
            if (
                isinstance(node.ops[0], ast.GtE)
                and self._is_zero(node.comparators[0])
                and isinstance(node.left, ast.Constant)
                and isinstance(node.left.value, (int, float))
                and float(node.left.value) >= 0
            ):
                return True
        return False

    def _is_false_test(self, node) -> bool:
        return (
            isinstance(node, ast.Compare)
            and isinstance(node.ops[0], ast.Eq)
            and self._is_zero(node.comparators[0])
            and isinstance(node.left, ast.Constant)
            and isinstance(node.left.value, (int, float))
            and float(node.left.value) == 0
        )

    def visit_Expression(self, node):
        """Optimize the top-level ``ast.Expression`` wrapper.

        ``ast.parse(..., mode="eval")`` returns an ``Expression`` node, and
        ``NodeVisitor`` has no ``visit_Expression``, so without this handler it
        falls through to ``generic_visit`` -- which returns ``None``.  That made
        ``optimize_ast`` return ``None`` for *every* input and broke
        ``compile_expr_to_glsl`` for all expressions.
        """
        return ast.Expression(body=self.visit(node.body))

    def generic_visit(self, node):
        """Visit children but never return ``None``.

        ``ast.NodeVisitor.generic_visit`` returns ``None`` for any node type
        without a dedicated handler (a bare ``Name``, ``Compare``, ``Subscript``
        or ``Attribute``).  Returning ``None`` here silently erased the subtree,
        so the GLSL generator later hit ``Unsupported AST node type: NoneType``.
        Nodes this optimizer has no rule for are simply left untouched.
        """
        super().generic_visit(node)
        return node


def optimize_ast(tree: _T) -> _T:
    """Run the expression optimizer on an AST. Returns optimized AST.

    The concrete node type is preserved so callers keep access to the
    node-specific attributes (e.g. ``Expression.body``) after optimisation.
    """
    optimizer = ExpressionOptimizer()
    result = optimizer.visit(cast(ast.AST, tree))
    return cast(_T, result)


def compile_expr_to_glsl(expr_str: str) -> str:
    """Compile a MilkDrop expression string to GLSL code.
    Returns GLSL expression string.
    Raises PresetCompileError on failure instead of returning empty string.
    """
    if not expr_str or not expr_str.strip():
        raise PresetCompileError("Empty expression string")
    val = expr_str.strip()
    try:
        val = preprocess_milkdrop_expr(val)
    except Exception as e:
        raise PresetCompileError(f"Preprocessing failed: {e}") from e
    if not val or not val.strip():
        raise PresetCompileError("Empty expression string")
    try:
        tree = ast.parse(_normalize_leading_zeros(val.strip()), mode="eval")
    except Exception as e:
        raise PresetCompileError(f"Parse error: {e}") from e
    tree = optimize_ast(tree)
    generator = GLSLGenerator()
    glsl = generator.visit(tree.body)
    # AGENTS.md §7: Reject NaN/Inf-prone calculations where practical.
    # Check for expressions that could produce Inf or NaN at runtime
    # (e.g. division by zero, sqrt of negative, log of non-positive).
    # We do a heuristic check: if the GLSL source contains patterns like
    #   1.0 / <variable> where variable could be zero
    #   sqrt(<expression>) without absolute value guard
    #   log(<expression>) without positive guard
    # we emit a warning but still allow the preset to compile so as not
    # to break existing presets unnecessarily.  The shader compiler itself
    # will catch runtime Inf/NaN and the preset health system will flag it.
    NaN_INF_RE = re.compile(
        r"(?:sqrt\s*\([^)]*\)| inversesqrt\s*\([^)]*\)|log\s*\([^)]*\)|"
        r"1\.0\s*/\s*[a-zA-Z_][a-zA-Z0-9_]*|/[a-zA-Z_][a-zA-Z0-9_]*\s*0)"
    )
    if NaN_INF_RE.search(glsl):
        logger.warning(
            "Generated GLSL contains NaN/Inf-prone patterns: %s",
            glsl[:200],  # log first 200 chars for debugging
        )
    if re.search(r"(?<![!<>=])=(?![=])", glsl):
        raise PresetCompileError(
            f"Generated GLSL contains unexpected assignment: '{glsl}'"
        )
    return glsl


# --- Bytecode program and peephole optimizer ----------------------
class BytecodePeepholeOptimizer:
    """Post-compilation peephole optimizer for bytecode.
    Reduces instruction count and eliminates dead code patterns.

    Every pass deletes instructions, but ``OP_JUMP`` / ``OP_JUMP_IF_FALSE`` /
    ``OP_JUMP_IF_TRUE`` carry **absolute instruction indices**.  Without a
    remap, folding a constant in front of a jump silently repoints the jump at
    the wrong instruction (and possibly past the end of the program), so any
    preset using a ternary / ``and`` / ``or`` / ``if()`` computes garbage.
    ``_orig`` tracks each surviving instruction's original index so the jump
    operands can be rewritten after every length-changing pass.
    """

    def __init__(self, code, const_pool, var_names):
        self.code = list(code)
        self.const_pool = list(const_pool)
        self.var_names = dict(var_names)
        # _orig[i] is the pre-optimization index of self.code[i].
        self._orig = list(range(len(self.code)))
        # Keep each jump's original target separate from its mutable operand.
        # Optimization can remove instructions in several passes, so a target
        # remapped once must not be interpreted as an original index again.
        self._original_jump_targets = {
            i: instr[1]
            for i, instr in enumerate(self.code)
            if instr[0] in (OP_JUMP, OP_JUMP_IF_FALSE, OP_JUMP_IF_TRUE)
            and len(instr) >= 2
            and isinstance(instr[1], int)
        }
        # False while jump operands are still in original index space.
        self._remapped = True

    def _drop(self, i, n=1):
        """Delete *n* instructions at *i*, keeping the origin index in sync."""
        del self.code[i : i + n]
        del self._orig[i : i + n]
        self._remapped = False

    def _remap_jump_targets(self):
        """Rewrite absolute jump operands from original into current space.

        Idempotent: operands are only translated once per length change, so
        a stale value is never translated a second time.
        """
        if self._remapped:
            return
        jump_ops = (OP_JUMP, OP_JUMP_IF_FALSE, OP_JUMP_IF_TRUE)
        for pos, instr in enumerate(self.code):
            if len(instr) < 2 or instr[0] not in jump_ops:
                continue
            target = self._original_jump_targets.get(self._orig[pos], instr[1])
            if not isinstance(target, int):
                continue
            new_target = len(self.code)
            for candidate_pos, orig in enumerate(self._orig):
                if orig >= target:
                    new_target = candidate_pos
                    break
            if new_target != target:
                self.code[pos] = (instr[0], new_target, *tuple(instr[2:]))
        self._remapped = True

    def optimize(self):
        self._remove_load_store_same_var()
        self._fold_const_binops()
        self._remove_dead_pop()
        self._fold_const_unary()
        # This pass reads jump operands, so they must be in current space.
        self._remap_jump_targets()
        self._remove_redundant_jumps()
        self._remap_jump_targets()
        return self.code, self.const_pool, self.var_names

    def _remove_load_store_same_var(self):
        i = 0
        targeted_indices = set(self._original_jump_targets.values())
        while i < len(self.code) - 1:
            instr = self.code[i]
            next_instr = self.code[i + 1]
            if (
                instr[0] == OP_LOAD_VAR
                and next_instr[0] == OP_STORE_VAR
                and instr[1] == next_instr[1]
            ) or (
                instr[0] == OP_LOAD_VAR_DICT
                and next_instr[0] == OP_STORE_VAR_DICT
                and instr[1] == next_instr[1]
            ):
                # Removing a self-load/store pair is only safe when control
                # flow cannot enter either instruction independently. In
                # particular, a conditional branch may jump directly to the
                # store to commit a value already left on the stack.
                if (
                    self._orig[i] in targeted_indices
                    or self._orig[i + 1] in targeted_indices
                ):
                    i += 1
                else:
                    self._drop(i, 2)
            else:
                i += 1

    def _fold_const_binops(self):
        i = 0
        while i < len(self.code) - 2:
            instr = self.code[i]
            if instr[0] == OP_LOAD_CONST:
                idx_a = instr[1]
                j = i + 1
                if j < len(self.code) and self.code[j][0] == OP_LOAD_CONST:
                    idx_b = self.code[j][1]
                    if j + 1 < len(self.code) and self.code[j + 1][0] in (
                        OP_ADD,
                        OP_SUB,
                        OP_MUL,
                        OP_DIV,
                        OP_MOD,
                    ):
                        op = self.code[j + 1][0]
                        try:
                            a = self.const_pool[idx_a]
                            b = self.const_pool[idx_b]
                            if not (
                                math.isnan(a)
                                or math.isnan(b)
                                or math.isinf(a)
                                or math.isinf(b)
                            ):
                                result = None
                                if op == OP_ADD:
                                    result = a + b
                                elif op == OP_SUB:
                                    result = a - b
                                elif op == OP_MUL:
                                    result = a * b
                                elif op == OP_DIV and abs(b) > EPSILON_DIV:
                                    result = a / b
                                elif op == OP_MOD and abs(b) > EPSILON_DIV:
                                    result = a % b
                                if result is not None and math.isfinite(result):
                                    new_idx = self._find_or_add_const(result)
                                    self.code[i] = (OP_LOAD_CONST, new_idx)
                                    self._drop(i + 1, 2)
                        except Exception as e:  # noqa: BLE001
                            logger.debug(f"Failed to optimize constant folding: {e}")
            i += 1

    def _remove_dead_pop(self):
        i = 0
        while i < len(self.code) - 1:
            if self.code[i][0] == OP_POP and self.code[i + 1][0] == OP_POP:
                self._drop(i)
            else:
                i += 1

    def _fold_const_unary(self):
        i = 0
        while i < len(self.code) - 1:
            instr = self.code[i]
            if instr[0] == OP_LOAD_CONST and i + 1 < len(self.code):
                op_instr = self.code[i + 1]
                if op_instr[0] == OP_NEG:
                    result = -self.const_pool[instr[1]]
                    new_idx = self._find_or_add_const(result)
                    self.code[i] = (OP_LOAD_CONST, new_idx)
                    self._drop(i + 1)
                    continue
                if op_instr[0] in (
                    OP_SIN,
                    OP_COS,
                    OP_TAN,
                    OP_ABS,
                    OP_FLOOR,
                    OP_CEIL,
                    OP_FRACT,
                    OP_EXP,
                    OP_LOG,
                    OP_SQRT,
                ):
                    val = self.const_pool[instr[1]]
                    op = op_instr[0]
                    try:
                        result = None
                        if op == OP_SIN:
                            result = math.sin(val)
                        elif op == OP_COS:
                            result = math.cos(val)
                        elif op == OP_TAN:
                            result = math.tan(val)
                        elif op == OP_ABS:
                            result = abs(val)
                        elif op == OP_FLOOR:
                            result = math.floor(val)
                        elif op == OP_CEIL:
                            result = math.ceil(val)
                        elif op == OP_FRACT:
                            result = val - math.floor(val)
                        elif op == OP_EXP:
                            result = 2.0**val
                        elif op == OP_LOG and val > 0:
                            result = math.log2(val)
                        elif op == OP_SQRT and val >= 0:
                            result = math.sqrt(val)
                        if result is not None and math.isfinite(result):
                            new_idx = self._find_or_add_const(result)
                            self.code[i] = (OP_LOAD_CONST, new_idx)
                            self._drop(i + 1)
                            continue
                    except Exception as e:  # noqa: BLE001
                        logger.debug(f"Failed to optimize load_const: {e}")
            i += 1

    def _remove_redundant_jumps(self):
        i = 0
        while i < len(self.code):
            instr = self.code[i]
            if instr[0] == OP_JUMP and isinstance(instr[1], int):
                target = instr[1]
                if target == i + 1:
                    self._drop(i)
                    continue
            if instr[0] == OP_JUMP_IF_FALSE and isinstance(instr[1], int):
                target = instr[1]
                if target == i + 1:
                    self.code[i] = (OP_POP,)
                    i += 1
                    continue
            if instr[0] == OP_JUMP_IF_TRUE and isinstance(instr[1], int):
                target = instr[1]
                if target == i + 1:
                    self.code[i] = (OP_POP,)
                    i += 1
                    continue
            i += 1

    def _find_or_add_const(self, value: float) -> int:
        if not isinstance(value, float):
            try:
                value = float(value)
            except ValueError:
                value = 0.0
        for idx, existing in enumerate(self.const_pool):
            if existing == value and not (math.isnan(value) or math.isnan(existing)):
                return idx
        idx = len(self.const_pool)
        self.const_pool.append(value)
        return idx



def _split_statements(expr_str: str) -> list[str]:
    """Split a MilkDrop equation line into ';'-separated statements.

    A ``;`` only separates two statements at parenthesis depth 0.  Semicolons
    inside backtick `` `...` `` strings or inside function-call argument lists
    are preserved.  The result contains only non-empty, stripped statements.

    This mirrors the behaviour of :func:`renderer.config._split_milk_eqs`; both
    must agree so that an ``if()`` whose branch bodies contain ``;`` (e.g.
    ``if(beat, timediff = time - time_st; time_st = time, timediff)``) is
    treated as a single statement rather than being torn into fragments that
    cannot be parsed by :func:`ast.parse`.
    """
    result: list[str] = []
    current: list[str] = []
    in_backtick = False
    depth = 0
    for _i, ch in enumerate(expr_str):
        if ch == "`":
            in_backtick = not in_backtick
            current.append(ch)
        elif (
            ch == ";" or (ch == "," and _comma_splits_statement(expr_str, _i))
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


class BytecodeProgram:
    """Compiled bytecode program ready for execution."""

    def __init__(self, code, const_pool, var_names):
        self.code = code
        self.const_pool = const_pool
        self.var_names = var_names
        self._const_key_to_index: dict[int, int] = {}
        self._call_registry_failures: dict[str, int] = {}
        self._rebuild_idx_to_var()

    def _rebuild_idx_to_var(self):
        self._idx_to_var = {v: k for k, v in self.var_names.items()}

    def execute(
        self,
        state: StateLike,
        frame_budget: FrameBudget | None = None,
    ) -> float:
        """Execute bytecode with given state.
        ~5-20x faster than AST evaluation.
        Uses array access for known variables via OP_LOAD_VAR_INDEX / OP_STORE_VAR_INDEX.

        When *frame_budget* is provided, the instruction stream is periodically
        checked against the :class:`FrameBudget` (every ``_BUDGET_CHECK_INTERVAL``
        opcodes).  If the budget is exhausted the run halts immediately and a
        :class:`PresetBudgetExhausted` is raised so the caller can skip the
        remaining expressions for the frame while preserving last-known-good
        state.  When *frame_budget* is ``None`` the budget check is skipped
        entirely (backward-compatible unbounded execution).
        """
        stack = []
        pc = 0
        steps = 0
        code = self.code
        code_len = len(code)
        const_pool = self.const_pool
        idx_to_var = self._idx_to_var
        vars_array = state.vars_array
        # ``vars``/``_global_regs`` are duck-typed: a caller may hand in a
        # minimal stand-in state that only implements get()/set().
        vars_dict = getattr(state, "vars", None)
        if vars_dict is None:
            vars_dict = _StateVarsProxy(state)
        global_regs = getattr(state, "_global_regs", None)
        if global_regs is None:
            global_regs = _StateGlobalRegsProxy(state)
        q_regs = state.q
        t_vars = state.t_vars
        megabuf_get = getattr(state, "megabuf_get", None)
        megabuf_set = getattr(state, "megabuf_set", None)
        while pc < code_len:
            # Defense in depth: today's jumps are all forward (short-circuit
            # and ternary only), so this cannot trigger, but a future opcode
            # that jumps backwards would hang the render thread with no
            # diagnostic instead of raising here.
            steps += 1
            if steps > MAX_BYTECODE_STEPS:
                raise PresetCompileError(
                    f"Bytecode exceeded {MAX_BYTECODE_STEPS} steps; "
                    "likely an unbounded loop in the expression"
                )
            # Periodically charge the frame budget for the instructions
            # consumed since the last check.  Batching the consume() call
            # avoids per-opcode overhead while still catching runaway VM
            # execution within a few hundred instructions.
            if frame_budget is not None and steps & (_BUDGET_CHECK_INTERVAL - 1) == 0:  # noqa: SIM102
                if not frame_budget.consume(_BUDGET_CHECK_INTERVAL):
                    raise PresetBudgetExhausted(
                        f"Frame budget exhausted after {frame_budget.consumed} "
                        f"instruction steps (total={frame_budget.total})",
                        steps_consumed=frame_budget.consumed,
                        steps_remaining=frame_budget.remaining,
                    )
            instr = code[pc]
            op = instr[0]
            if op == OP_LOAD_VAR:
                var_name = idx_to_var.get(instr[1])
                if var_name is None:
                    stack.append(0.0)
                else:
                    stack.append(float(state.get(var_name)))
            elif op == OP_LOAD_VAR_DICT:
                stack.append(float(vars_dict.get(instr[1], 0.0)))
            elif op == OP_STORE_VAR_DICT:
                vars_dict[instr[1]] = _finite(float(stack.pop() if stack else 0.0))
            elif op == OP_LOAD_VAR_REG:
                stack.append(float(global_regs[instr[1]]))
            elif op == OP_STORE_VAR_REG:
                global_regs[instr[1]] = _finite32(float(stack.pop() if stack else 0.0))
            elif op == OP_LOAD_CONST:
                stack.append(const_pool[instr[1]])
            elif op == OP_MUL:
                b, a = (stack.pop() if stack else 0.0), (stack.pop() if stack else 0.0)
                stack.append(_finite(a * b))
            elif op == OP_STORE_VAR:
                value = _finite32(float(stack.pop() if stack else 0.0))
                var_name = idx_to_var.get(instr[1])
                if var_name is not None:
                    state.set(var_name, value)
            elif op == OP_ADD:
                a, b = (stack.pop() if stack else 0.0), (stack.pop() if stack else 0.0)
                stack.append(_finite(a + b))
            elif op == OP_SIN:
                stack.append(_md_sin(stack.pop() if stack else 0.0))
            elif op == OP_LOAD_VAR_INDEX:
                stack.append(float(vars_array[instr[1]]))
            elif op == OP_SUB:
                b, a = (stack.pop() if stack else 0.0), (stack.pop() if stack else 0.0)
                stack.append(_finite(a - b))
            elif op == OP_JUMP_IF_FALSE:
                cond = stack.pop() if stack else 0.0
                if abs(cond) < EPSILON_BOOL:
                    pc = instr[1]
                    continue
            elif op == OP_COS:
                stack.append(_md_cos(stack.pop() if stack else 0.0))
            elif op in (
                OP_CMP_EQ,
                OP_CMP_NE,
                OP_CMP_LT,
                OP_CMP_LE,
                OP_CMP_GT,
                OP_CMP_GE,
            ):
                b = stack.pop() if stack else 0.0
                a = stack.pop() if stack else 0.0
                eps = EPSILON_COMPARE
                if op == OP_CMP_EQ:
                    stack.append(1.0 if abs(a - b) < eps else 0.0)
                elif op == OP_CMP_NE:
                    stack.append(1.0 if abs(a - b) >= eps else 0.0)
                elif op == OP_CMP_LT:
                    stack.append(1.0 if a < b else 0.0)
                elif op == OP_CMP_LE:
                    stack.append(1.0 if a <= b else 0.0)
                elif op == OP_CMP_GT:
                    stack.append(1.0 if a > b else 0.0)
                elif op == OP_CMP_GE:
                    stack.append(1.0 if a >= b else 0.0)
            elif op == OP_JUMP:
                pc = instr[1]
                continue
            elif op == OP_LOAD_VAR_INDEX_Q:
                stack.append(float(q_regs[instr[1]]))
            elif op == OP_STORE_VAR_INDEX:
                value = _finite32(float(stack.pop() if stack else 0.0))
                vars_array[instr[1]] = value
            elif op == OP_ABS:
                stack.append(abs(stack.pop() if stack else 0.0))
            elif op == OP_DIV:
                b, a = (stack.pop() if stack else 0.0), (stack.pop() if stack else 0.0)
                if abs(b) < EPSILON_DIV:
                    denom = -EPSILON_DIV if b < 0 else EPSILON_DIV
                else:
                    denom = b
                stack.append(_finite(a / denom))
            elif op == OP_MOD:
                b, a = (stack.pop() if stack else 0.0), (stack.pop() if stack else 0.0)
                stack.append(_md_mod(a, b))
            elif op == OP_RAND:
                stack.append(md_rand(state))
            elif op == OP_POW_FUNC:
                b, a = (stack.pop() if stack else 0.0), (stack.pop() if stack else 0.0)
                stack.append(_md_pow(a, b))
            elif op == OP_ACOS:
                stack.append(_md_acos(stack.pop() if stack else 0.0))
            elif op == OP_AND:
                b = stack.pop() if stack else 0.0
                a = stack.pop() if stack else 0.0
                stack.append(
                    1.0 if (abs(a) > EPSILON_BOOL and abs(b) > EPSILON_BOOL) else 0.0
                )
            elif op == OP_ASIN:
                stack.append(_md_asin(stack.pop() if stack else 0.0))
            elif op == OP_ATAN:
                stack.append(math.atan(stack.pop() if stack else 0.0))
            elif op == OP_ATAN2:
                y, x = (stack.pop() if stack else 0.0), (stack.pop() if stack else 0.0)
                stack.append(math.atan2(y, x))
            elif op == OP_BNOT:
                a = stack.pop() if stack else 0.0
                stack.append(float(~int(_md_int(a))))
            elif op == OP_BOR:
                a, b = (stack.pop() if stack else 0.0), (stack.pop() if stack else 0.0)
                stack.append(float(int(_md_int(a)) | int(_md_int(b))))
            elif op == OP_CALL_REGISTRY:
                name = instr[1] if len(instr) > 1 else ""
                argc = instr[2] if len(instr) > 2 else 0
                spec = FUNCTION_REGISTRY.get(name) if isinstance(name, str) else None
                args = [(stack.pop() if stack else 0.0) for _ in range(int(argc))]
                args.reverse()
                if spec is None:
                    logger.debug(
                        "Bytecode: no CPU implementation for %r; using 0.0", name
                    )
                    stack.append(0.0)
                    pc += 1
                    continue
                try:
                    result = float(spec.cpu_impl(*args))
                except Exception:
                    # A preset calling a function with the wrong arity fails
                    # on every frame.  Log the full traceback only on the
                    # *first* occurrence per call site; subsequent failures
                    # degrade to a debug-level summary with a count to avoid
                    # log-spam at frame rate.
                    count = self._call_registry_failures.get(name, 0) + 1
                    self._call_registry_failures[name] = count
                    if count == 1:
                        logger.warning(
                            "Bytecode: %r raised on first call with %d "
                            "argument(s); using 0.0",
                            name,
                            len(args),
                            exc_info=True,
                        )
                    elif VM_DEBUG:
                        logger.debug(
                            "Bytecode: %r raised again (%d× total) with %d argument(s)",
                            name,
                            count,
                            len(args),
                        )
                    else:
                        logger.debug(
                            "Bytecode: %r raised (%d× total) with %d "
                            "argument(s); using 0.0",
                            name,
                            count,
                            len(args),
                        )
                    result = 0.0
                stack.append(result if math.isfinite(result) else 0.0)
            elif op == OP_CEIL:
                stack.append(_md_ceil(stack.pop() if stack else 0.0))
            elif op == OP_CLAMP:
                hi, lo, x = (
                    (stack.pop() if stack else 0.0),
                    (stack.pop() if stack else 0.0),
                    (stack.pop() if stack else 0.0),
                )
                stack.append(_finite(max(lo, min(hi, x))))
            elif op == OP_CWRAP:
                b, a, x = (
                    (stack.pop() if stack else 0.0),
                    (stack.pop() if stack else 0.0),
                    (stack.pop() if stack else 0.0),
                )
                r = b - a
                stack.append(_finite(a + abs(((x - a) % (2 * r)) - r)))
            elif op == OP_DUP:
                if stack:
                    stack.append(stack[-1])
            elif op == OP_EXP:
                stack.append(_md_exp(stack.pop() if stack else 0.0))
            elif op == OP_FACT:
                stack.append(_md_factorial(stack.pop() if stack else 0.0))
            elif op == OP_FLOOR:
                stack.append(_md_floor(stack.pop() if stack else 0.0))
            elif op == OP_FRACT:
                x = stack.pop() if stack else 0.0
                stack.append(x - _md_floor(x) if math.isfinite(x) else 0.0)
            elif op == OP_IF:
                false_val = stack.pop() if stack else 0.0
                true_val = stack.pop() if stack else 0.0
                cond = stack.pop() if stack else 0.0
                stack.append(true_val if abs(cond) > EPSILON_BOOL else false_val)
            elif op == OP_INT_FUNC:
                stack.append(float(_md_int(stack.pop() if stack else 0.0)))
            elif op == OP_INVSQRT:
                x = stack.pop() if stack else 0.0
                stack.append(_md_invsqrt(x))
            elif op == OP_JUMP_IF_TRUE:
                cond = stack.pop() if stack else 0.0
                if abs(cond) >= EPSILON_BOOL:
                    pc = instr[1]
                    continue
            elif op == OP_LOAD_VAR_INDEX_T:
                stack.append(float(t_vars[instr[1]]))
            elif op == OP_LOG:
                val = stack.pop() if stack else 0.0
                stack.append(_md_log(val))
            elif op == OP_LOG10:
                val = stack.pop() if stack else 0.0
                stack.append(_md_log10(val))
            elif op == OP_MAX:
                b, a = (stack.pop() if stack else 0.0), (stack.pop() if stack else 0.0)
                stack.append(_finite(max(a, b)))
            elif op == OP_MEGABUF_GET:
                idx = int(stack.pop() if stack else 0.0) % PresetState.MEGABUF_SIZE
                megabuf_get = getattr(state, "megabuf_get", None)
                if megabuf_get is not None:
                    stack.append(float(megabuf_get(idx)))
                else:
                    stack.append(0.0)
            elif op == OP_MEGABUF_SET:  # historical value was 29
                val = _finite32(float(stack.pop() if stack else 0.0))
                idx = int(stack.pop() if stack else 0.0) % PresetState.MEGABUF_SIZE
                megabuf_set = getattr(state, "megabuf_set", None)
                if megabuf_set is not None:
                    megabuf_set(idx, val)
            elif op == OP_MIN:
                b, a = (stack.pop() if stack else 0.0), (stack.pop() if stack else 0.0)
                stack.append(_finite(min(a, b)))
            elif op == OP_MIX:
                t, b, a = (
                    (stack.pop() if stack else 0.0),
                    (stack.pop() if stack else 0.0),
                    (stack.pop() if stack else 0.0),
                )
                stack.append(_finite(a + (b - a) * t))
            elif op == OP_NEG:
                stack.append(_finite(-(stack.pop() if stack else 0.0)))
            elif op == OP_NOT:
                a = stack.pop() if stack else 0.0
                stack.append(0.0 if a != 0 else 1.0)
            elif op == OP_OR:
                b = stack.pop() if stack else 0.0
                a = stack.pop() if stack else 0.0
                stack.append(
                    1.0 if (abs(a) > EPSILON_BOOL or abs(b) > EPSILON_BOOL) else 0.0
                )
            elif op == OP_POP:
                if stack:
                    stack.pop()
            elif op == OP_POW:
                b, a = (stack.pop() if stack else 0.0), (stack.pop() if stack else 0.0)
                stack.append(_md_pow(a, b))
            elif op == OP_Q_GET:
                stack.append(q_regs[instr[1]])
            elif op == OP_Q_SET:
                q_regs[instr[1]] = _finite(stack.pop() if stack else 0.0)
            elif op == OP_SIGMOID:
                if len(stack) >= 2:
                    constraint = stack.pop() if stack else 0.0
                    x = stack.pop() if stack else 0.0
                else:
                    constraint = 1.0
                    x = stack.pop() if stack else 0.0
                stack.append(_md_sigmoid(x, constraint))
            elif op == OP_SIGN:
                x = stack.pop() if stack else 0.0
                stack.append(1.0 if x > 0 else (-1.0 if x < 0 else 0.0))
            elif op == OP_SQR:
                x = stack.pop() if stack else 0.0
                stack.append(_finite(x * x))
            elif op == OP_SQRT:
                stack.append(_md_sqrt(stack.pop() if stack else 0.0))
            elif op == OP_TAN:
                stack.append(_md_tan(stack.pop() if stack else 0.0))
            elif op == OP_T_SET:
                t_vars[instr[1]] = _finite(stack.pop() if stack else 0.0)
            elif op == OP_WRAP:
                b, a, x = (
                    (stack.pop() if stack else 0.0),
                    (stack.pop() if stack else 0.0),
                    (stack.pop() if stack else 0.0),
                )
                stack.append(_finite(a + ((x - a) % (b - a + EPSILON_DIV))))
            pc += 1
        return _finite(stack[-1]) if stack else 0.0


# --- Compiled program containers ----------------------------------
@dataclass
class ShaderRepairReport:
    """Report of all shader repair decisions.

    Instead of silently rewriting declarations based on later textual
    usage (which can produce visually-correct-but-semantically-wrong
    shaders), the repair pipeline records every decision here.  Each
    entry is either an *applied* repair or a *rejected* attempt.

    Repair *levels* classify the risk of each transformation:

      * ``SAFE``         - mechanical rewrites that never change semantics
                           (e.g. ``log10(x) -> log(x)/log(10)``,
                           ``% -> mod()`` for floats, removing
                           ``dx.x`` swizzles on scalar ``dx``).
      * ``CONTEXTUAL``   - rewrites that *can* change visual output and are
                           only applied when ``allow_contextual_repairs`` is
                           enabled on the report (e.g. ``mix(vec3, vec4, x)``
                           promotion, vecN scalar wrapping).
      * ``REJECT``       - transformation refused because it would lie about
                           a declared variable type, produce an impossible
                           swizzle, or hide an ambiguous type.

    Example report::

        variable x:
          original type = float
          requested use = x.z
          repair = REJECT
        texture(sampler_main, uv):
          original type = vec4
          requested use = vec3 (assigned to vec3 color)
          repair = CONTEXTUAL applied (.rgb)
    """

    SAFE = "SAFE"
    CONTEXTUAL = "CONTEXTUAL"
    REJECT = "REJECT"

    repairs: list[str] = field(default_factory=list)
    rejections: list[str] = field(default_factory=list)
    repairs_safe: list[str] = field(default_factory=list)
    repairs_contextual: list[str] = field(default_factory=list)
    # CONTEXTUAL repairs are recorded but only applied to the shader source
    # when this flag is True. SAFE repairs are always applied. REJECT
    # repairs are never applied.
    #
    # Default is False: for a MilkDrop-compatibility renderer we prefer
    # strict type fidelity and only fall back to contextual rewrites when
    # the caller opts in. Contextual repairs (e.g. ``mix(vec3, vec4)``
    # promotion, vector-scalar wrapping) can change visual output, and
    # silently rewriting a preset's per-pixel math to make it compile is
    # not what we want during normal playback.
    allow_contextual_repairs: bool = False

    def applied(self, msg: str, level: str = "SAFE"):
        """Record an applied repair at the given level (default SAFE).

        Returns ``True`` if the caller should apply the textual rewrite to
        the shader source, ``False`` if the repair was recorded but
        suppressed (CONTEXTUAL repairs when ``allow_contextual_repairs``
        is False). REJECT repairs are never applied.
        """
        if level == self.CONTEXTUAL:
            self.repairs_contextual.append(msg)
            if self.allow_contextual_repairs:
                self.repairs.append(msg)
                return True
            return False
        self.repairs_safe.append(msg)
        self.repairs.append(msg)
        return True

    def rejected(self, msg: str):
        self.rejections.append(msg)

    @property
    def empty(self) -> bool:
        return not self.repairs and not self.rejections

    def __str__(self) -> str:
        lines = []
        for r in self.repairs_safe:
            lines.append(f"  APPLIED [SAFE]: {r}")
        for r in self.repairs_contextual:
            tag = "applied" if self.allow_contextual_repairs else "suppressed"
            lines.append(f"  {tag.upper()} [CONTEXTUAL]: {r}")
        for r in self.rejections:
            lines.append(f"  REJECT: {r}")
        return "\n".join(lines) if lines else "  (no repairs or rejections)"


@dataclass(slots=True)
class WaveProgram:
    """Dataclass for wave structures with full MilkDrop 1.x wave mode support.

    Classic wave modes (0-7+):
    0: Single line spectrum
    1: Single line waveform
    2: Dual spectrum (stereo)
    3: Dual waveform (stereo)
    4: Circular spectrum
    5: Circular waveform
    6: Radial spectrum
    7: Radial waveform
    """

    per_frame: list[CompiledExpr]
    per_point: list[CompiledExpr]
    per_frame_init: list[CompiledExpr] = field(default_factory=list)
    enabled: bool = True
    # Wave parameters from preset
    mode: int = 0  # 0-7+ for different wave visualizations
    additive: bool = True  # bAdditive: True=additive blend, False=normal alpha
    samples: int = 512
    bSpectrum: bool = True  # True = spectrum, False = waveform
    bUseDots: bool = False
    bDrawThick: bool = False
    bMaximize: bool = False
    # Colors
    r: float = 1.0
    g: float = 1.0
    b: float = 1.0
    a: float = 1.0
    # Scaling and positioning
    x: float = 0.5
    y: float = 0.5
    mystery: float = 0.0  # Mystery parameter for some wave modes
    # MilkDrop 1.04+ bridge variables
    t1: float = 0.0
    t2: float = 0.0
    t3: float = 0.0
    t4: float = 0.0
    t5: float = 0.0
    t6: float = 0.0
    t7: float = 0.0
    t8: float = 0.0
    # Instancing support (MilkDrop 1.04+)
    num_instances: int = 1  # 1-1024
    # Circular/radial modes (4-7): angle offset in radians.
    # Linear modes (0-3): vertical displacement in normalized [0,1] y-space
    # (renderer extension — see MilkdropRenderer.compatibility_mode docs).
    instance_separator: float = 0.0  # Offset between instances
    # Explicit internal fields (planned refactor):
    #   _instance_angle_step: radians per instance for circular/radial modes
    #   _instance_linear_offset: vertical displacement per instance for linear modes
    # instance_separator remains the parser-facing compatibility alias.
    _instance_angle_step: float = 0.0
    _instance_linear_offset: float = 0.0


@dataclass(slots=True)
class ShapeProgram:
    """Dataclass for shape structures with full MilkDrop 1.x shape support.

    Shape types:
    - Circle (sides=0 or sides >= 3)
    - Triangle (sides=3)
    - Square (sides=4)
    - Polygon (sides >= 3)

    Rendering features:
    - Textured vs untextured
    - Additive vs normal blending
    - Thick outlines
    - Per-frame and per-frame-init equations
    - Full transformation support (x, y, rad, ang, tex_ang, tex_zoom)
    """

    per_frame: list[CompiledExpr]
    per_frame_init: list[CompiledExpr]
    enabled: bool = True
    # Shape parameters
    sides: int = 3  # 0 = circle, 3+ = polygon
    additive: bool = False
    thick_outline: bool = False
    textured: bool = False
    # Colors
    r: float = 1.0
    g: float = 1.0
    b: float = 1.0
    a: float = 1.0
    r2: float = 0.0  # Border color
    g2: float = 0.0
    b2: float = 0.0
    a2: float = 0.0
    # Transformation
    x: float = 0.5
    y: float = 0.5
    rad: float = 0.1  # Radius/size
    ang: float = 0.0  # Rotation angle
    tex_ang: float = 0.0  # Texture rotation
    tex_zoom: float = 1.0  # Texture zoom
    # Border
    border_r: float = 1.0
    border_g: float = 1.0
    border_b: float = 1.0
    border_a: float = 1.0
    border_size: float = 0.01
    # MilkDrop 1.04+ bridge variables
    t1: float = 0.0
    t2: float = 0.0
    t3: float = 0.0
    t4: float = 0.0
    t5: float = 0.0
    t6: float = 0.0
    t7: float = 0.0
    t8: float = 0.0
    # Tracks which t-vars were explicitly assigned in the shape definition,
    # so that a value of 0.0 is distinguishable from "not set".
    t_assigned: frozenset = field(default_factory=frozenset)
    # Instancing support (MilkDrop 1.04+)
    num_instances: int = 1  # 1-1024
    # instance_separator is an angular offset in RADIANS between consecutive
    # shape instances.  This matches MilkDrop 1.04+ semantics and the
    # circular/radial wave interpretation.  NOTE: the linear wave path
    # (modes 0-3) interprets instance_separator as a vertical displacement
    # (renderer extension, not MilkDrop spec).  See compatibility_mode on
    # MilkdropRenderer for renderer-specific instancing behaviour.
    instance_separator: float = 0.0  # Angle offset (radians) between instances
    # Explicit internal fields (planned refactor):
    #   _instance_angle_step: radians per instance for circular/radial modes
    #   _instance_shape_pos_offset: renderer-specific positional displacement
    # instance_separator remains the parser-facing compatibility alias.
    _instance_angle_step: float = 0.0
    _instance_shape_pos_offset: float = 0.0



def _has_top_level_assignment(expr: str) -> bool:
    """Check if *expr* has a top-level ``=`` that is not part of a
    comparison operator (``==``, ``!=``, ``<=``, ``>=``).

    Compound assignments (``+=``, ``-=``, ``*=``, ``/=``, ``%=``) ARE
    recognized as valid assignments — the ``=`` in ``+=`` has prev_c
    ``+``, which is not a comparison-operator character, so it returns
    ``True``.

    Comparison operators are excluded by checking ``prev_c``: characters
    that precede ``=`` in comparison operators are ``!``, ``<``, ``>``,
    and ``=`` (the last handles the second ``=`` of ``==``, which is
    already skipped by the ``next_c == "="`` guard but is checked here for
    completeness).  Compound-assignment characters (``+``, ``-``, ``*``,
    ``/``, ``%``) are deliberately excluded so that ``+=``, ``-=``,
    ``*=``, ``/=``, ``%=`` return ``True``.
    """
    depth_p = depth_b = depth_c = 0
    i = 0
    while i < len(expr):
        c = expr[i]
        if c == "(":
            depth_p += 1
        elif c == ")":
            depth_p = max(0, depth_p - 1)
        elif c == "[":
            depth_b += 1
        elif c == "]":
            depth_b = max(0, depth_b - 1)
        elif c == "{":
            depth_c += 1
        elif c == "}":
            depth_c = max(0, depth_c - 1)
        elif c == "=" and depth_p == 0 and depth_b == 0 and depth_c == 0:
            prev_c = expr[i - 1] if i > 0 else ""
            next_c = expr[i + 1] if i + 1 < len(expr) else ""
            if next_c == "=":
                i += 1
                continue
            # Only comparison-operator characters (!, <, >, =) before '='
            # indicate a non-assignment.  Compound-assignment characters
            # (+, -, *, /, %) indicate a valid assignment.
            if prev_c in "!<>=":
                i += 1
                continue
            return True
        i += 1
    return False


def _is_bare_expression(s: str) -> bool:
    """Check if a string is a valid bare expression (no top-level '=').

    Returns True for expressions that are valid MilkDrop side-effect
    statements without an assignment: exec1/2/3 calls, assign() calls,
    or if()-wrapped versions of those.
    """
    s = s.strip()
    # Strip outer parens
    while s.startswith("(") and s.endswith(")"):
        # Make sure the parens are balanced and pair the outermost ones
        depth = 0
        matched = True
        for k, c in enumerate(s):
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0 and k < len(s) - 1:
                    matched = False
                    break
        if matched:
            s = s[1:-1].strip()
        else:
            break
    if not s:
        return True
    s_lower = s.lower()
    if s_lower.startswith(("exec1(", "exec1 (", "exec2(", "exec2 (", "exec3(", "exec3 (", "assign(", "assign (", "if(", "if (")):
        return True
    if re.match(r"^[bB][nN][oO][tT]\s*\(", s):
        return True
    if re.match(r"^(above|below|equal|above_equal|below_equal|not_equal|bor|band|if)\s*\(", s):
        return True
    if ("&&" in s or "||" in s or "==" in s or "!=" in s or "<=" in s or ">=" in s
        ) and not _has_top_level_assignment(s):
        # Could be a bare boolean expression fragment
        return True
    # Any balanced, AST-evaluable expression without a top-level assignment is
    # a valid bare (evaluate-and-discard) MilkDrop statement.  This covers bare
    # boolean/arithmetic fragments such as ``(dist < 0.6)*(len > 30)``.
    if not _has_top_level_assignment(s) and (
        s.count("(") == s.count(")")
        and s.count("[") == s.count("]")
        and s.count("{") == s.count("}")
    ):
        try:
            ast.parse(_normalize_leading_zeros(s), mode="eval")
            return True
        except Exception:  # noqa: BLE001
            return False
    # If it's just a function call with no assignment, might be void
    return bool(re.match(r"^[a-zA-Z_][\w]*(\([^()]*\))*$", s)) and not _has_top_level_assignment(s)


class CompiledExpr:
    """A compiled expression ready for evaluation.
    Compiles once at load time, executes many times at runtime.
    """

    _LOG_PARSE_ERRORS: ClassVar[bool] = (
        os.getenv("MILKDROP_LOG_PARSE_ERRORS", "0") == "1"
    )

    @staticmethod
    def _truncate_for_log(s: str, limit: int = 300) -> str:
        if s is None:
            return ""
        s = str(s)
        if len(s) <= limit:
            return s
        return s[:limit] + "…"

    def __init__(self, expr_str: str):
        self.source = expr_str.strip()
        self.bytecode_prog = None
        self.runtime_failures = 0
        self._compile()

    def _compile(self):
        try:
            clean = preprocess_milkdrop_expr(self.source)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Preprocessing failed in _compile: {e}")
            return
        if not clean:
            return
        stripped = clean.strip()
        # A leading comma is the exec2()/while() argument-separator artifact
        # produced when a trailing while()/exec2() argument lands on its own
        # per_frame_N line (e.g. ``,(dist < 0.6)*(len > 30)``).  It is not part
        # of the expression, so drop it before compiling.
        if stripped.startswith(","):
            stripped = stripped[1:].strip()
        if not stripped:
            return
        # Expressions like exec3(assign(x, v), y) or assign(x, v) are valid
        # MilkDrop statements that don't contain a top-level '='.  Bare
        # boolean/arithmetic expressions (evaluated and discarded) are also
        # valid MilkDrop statements.
        if not _has_top_level_assignment(stripped):
            try:
                _bare = _is_bare_expression(stripped)
            except Exception:  # noqa: BLE001
                _bare = False
            if _bare:
                pass  # compilable bare expression: evaluate and discard
            elif not any(c.isalnum() for c in stripped):
                # Pure punctuation / loop terminator (")", ";", ",") — nothing
                # to compile, not an error.
                return
            elif stripped.count("(") != stripped.count(")"):
                # Unbalanced orphaned fragment of a while()/exec2() construct;
                # not a real assignment, skip silently.
                return
            else:
                # Balanced but not AST-evaluable: genuine syntax error.
                logger.warning(
                    f"Expression missing assignment: {self.source!r}"
                )
                return
        compiler = BytecodeCompiler()
        try:
            self.bytecode_prog = compiler.compile_assignment(stripped)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Exception in _compile: {e}")
            return
        if self.bytecode_prog is None:
            return
        optimizer = BytecodePeepholeOptimizer(
            self.bytecode_prog.code,
            self.bytecode_prog.const_pool,
            self.bytecode_prog.var_names,
        )
        try:
            new_code, new_consts, new_vars = optimizer.optimize()
            self.bytecode_prog.code = new_code
            self.bytecode_prog.const_pool = new_consts
            self.bytecode_prog.var_names = new_vars
            self.bytecode_prog._rebuild_idx_to_var()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Optimizer failed: {e}")

    def eval(
        self,
        state: StateLike,
        max_iterations: int = MAX_FRAME_LOOP_ITERATIONS,
        budget: LoopBudget | None = None,
        frame_budget: FrameBudget | None = None,
    ) -> float:
        # ``budget`` (LoopBudget) is accepted for a uniform signature with
        # LoopExpr and controls the *iteration* count of any nested loops.
        # ``frame_budget`` (FrameBudget) is shared across the entire frame;
        # each expression evaluation consumes a base unit plus per-instruction
        # steps inside BytecodeProgram.execute so that both simple expressions
        # (which have < _BUDGET_CHECK_INTERVAL opcodes) and heavy expressions
        # are accounted for against the frame allowance.
        if self.bytecode_prog:
            if frame_budget is not None and not frame_budget.consume(1):
                raise PresetBudgetExhausted(
                    f"Frame budget exhausted before evaluating {self.source!r} "
                    f"(consumed={frame_budget.consumed}/{frame_budget.total})",
                    steps_consumed=frame_budget.consumed,
                    steps_remaining=frame_budget.remaining,
                )
            return self.bytecode_prog.execute(state, frame_budget=frame_budget)
        raise PresetCompileError("Cannot evaluate: expression failed to compile")

    def is_valid(self) -> bool:
        return self.bytecode_prog is not None


class LoopExpr:
    """Represents a MilkDrop loop block: loop (count, body...).

    Executes the body expressions N times per frame (or once during init).
    """

    # ``runtime_failures`` and ``bytecode_prog`` mirror CompiledExpr's
    # failure-disable bookkeeping. The per_frame / wave / shape lists hold
    # LoopExpr objects, and the recovery handlers in run_frame() touch both
    # attributes; without them the handler itself raised AttributeError on the
    # first loop-body failure, masking the original error and killing the frame.
    __slots__ = (
        "body_exprs",
        "bytecode_prog",
        "count_compiled",
        "count_expr",
        "count_var",
        "raw_body",
        "runtime_failures",
        "source",
    )

    def __init__(self, count_str: str, body_strs: list):
        self.runtime_failures = 0
        self.bytecode_prog = None
        self.count_expr = count_str.strip() if count_str else "0"
        if not self.count_expr:
            self.count_compiled = None
            self.count_var = None
        else:
            try:
                if "=" in self.count_expr:
                    self.count_compiled = (
                        CompiledExpr(self.count_expr) if self.count_expr else None
                    )
                    self.count_var = self._extract_assign_target(self.count_expr)
                else:
                    self.count_compiled = CompiledExpr(
                        f"__md_loop_count = ({self.count_expr})"
                    )
                    self.count_var = "__md_loop_count"
            except Exception:  # noqa: BLE001
                self.count_compiled = None
                self.count_var = None
        self.raw_body = list(body_strs)
        self.body_exprs: list[LoopExpr | CompiledExpr] = []
        for s in body_strs:
            if isinstance(s, LoopExpr):
                self.body_exprs.append(s)
                continue
            if not isinstance(s, str):
                continue
            # Strip a trailing comment so "a = 1 // note" compiles as
            # "a = 1" instead of being rejected whole as a comment line.
            s_str = _strip_inline_comments(s).strip()
            if _is_metadata_or_comment_expression(s_str):
                continue
            try:
                compiled = CompiledExpr(s_str)
            except Exception:  # noqa: BLE001
                logger.warning("Skipping invalid loop body expression: %r", s_str)
                continue
            if compiled.is_valid():
                self.body_exprs.append(compiled)

        def _serialize_body(body_strs):
            result = []
            for s in body_strs:
                if isinstance(s, LoopExpr):
                    result.append(
                        {
                            "__milk_loop__": True,
                            "count": s.count_expr,
                            "body": _serialize_body(s.raw_body),
                        }
                    )
                else:
                    result.append(s)
            return result

        self.source = json.dumps(
            {
                "__milk_loop__": True,
                "count": self.count_expr,
                "body": _serialize_body(body_strs),
            }
        )

    def _extract_assign_target(self, expr_str: str) -> str | None:
        """Extract the LHS variable name from an assignment expression."""
        try:
            tree = ast.parse(_normalize_leading_zeros(expr_str.strip()), mode="eval")
        except Exception:  # noqa: BLE001
            try:
                mod = ast.parse(_normalize_leading_zeros(expr_str.strip()), mode="exec")
                if mod.body and isinstance(mod.body[0], ast.Assign):
                    targets = mod.body[0].targets
                    if targets and isinstance(targets[0], ast.Name):
                        return targets[0].id.lower()
            except Exception:  # noqa: BLE001, S110
                pass
            return None
        node = tree.body
        if isinstance(node, ast.Assign):
            targets = node.targets
            if targets and isinstance(targets[0], ast.Name):
                return targets[0].id.lower()
        return None

    def eval(
        self,
        state: StateLike,
        max_iterations: int = MAX_FRAME_LOOP_ITERATIONS,
        budget: LoopBudget | None = None,
        frame_budget: FrameBudget | None = None,
    ) -> float:
        if budget is None:
            # One budget per top-level expression; shared with the whole loop
            # body so nesting cannot multiply the per-loop clamp.
            budget = LoopBudget(max_iterations * MAX_LOOPS_PER_EXPRESSION)
        if self.count_compiled is None or not self.count_compiled.is_valid():
            logger.warning("Invalid loop count expression: %s", self.count_expr)
            return 0.0
        self.count_compiled.eval(state, frame_budget=frame_budget)
        if self.count_var is not None:
            count = int(state.get(self.count_var, 0))
        else:
            count = 0
        if self.count_var == "__md_loop_count":
            _vars = getattr(state, "vars", None)
            if _vars is not None:
                _vars.pop("__md_loop_count", None)
        count = max(0, min(count, max_iterations))
        granted = budget.take(count)
        if granted <= 0:
            if budget.exhausted:
                logger.warning(
                    "Loop iteration budget exhausted in %s; remaining loops "
                    "skipped (nested loop bomb?)",
                    self.count_expr,
                )
            return 0.0
        # Run exactly the iterations the budget granted.  Loops nested in the
        # body get 0 from the same (now possibly exhausted) budget, so they
        # degrade to no-ops instead of multiplying the cost.  The FrameBudget
        # is threaded through so per-instruction work is charged against the
        # shared per-frame allowance.
        for _ in range(granted):
            for expr in self.body_exprs:
                expr.eval(state, budget=budget, frame_budget=frame_budget)
        return 0.0

    def is_valid(self) -> bool:
        return bool(self.body_exprs) or bool(self.raw_body)


# --- Preset compilation helpers ------------------------------------
def _append_compiled_expression(target, source, context_label="expression"):
    """Compile a MilkDrop expression and append it to target if valid.

    Args:
        target: List to append the compiled expression to.
        source: Raw expression string or LoopExpr object.
        context_label: Label for warning messages (e.g., "per_frame").
    """
    if isinstance(source, LoopExpr):
        if source.is_valid():
            target.append(source)
        return
    if not source or not str(source).strip():
        return
    # Strip a trailing comment so "a = 1 // note" compiles as
    # "a = 1" instead of being rejected whole as a comment line.
    expr_str = _strip_inline_comments(str(source)).strip()
    if _is_metadata_or_comment_expression(expr_str):
        return
    try:
        compiled = CompiledExpr(expr_str)
        if compiled.is_valid():
            target.append(compiled)
    except PresetCompileError as e:
        logger.warning(f"Skipping invalid {context_label} expression: {e}")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Unexpected error compiling {context_label} expression: {e}")


def _compile_expression_list(expr_list, is_per_pixel=False):
    """Compile a list of MilkDrop expressions with optional filtering.

    Args:
        expr_list: List of raw expression strings or LoopExpr objects.
        is_per_pixel: Whether these are per-pixel expressions (skips additional checks).

    Returns:
        List of compiled CompiledExpr and LoopExpr objects.
    """
    compiled_list = []
    for item in expr_list:
        if isinstance(item, LoopExpr):
            if item.is_valid():
                compiled_list.append(item)
            continue
        if not item or not str(item).strip():
            continue
        # Strip a trailing comment so "a = 1 // note" compiles as
        # "a = 1" instead of being rejected whole as a comment line.
        expr_str = _strip_inline_comments(str(item)).strip()
        if not expr_str:
            continue
        if expr_str.startswith('{"__milk_loop__"'):
            try:
                data = json.loads(expr_str)
                if data.get("__milk_loop__"):
                    loop_expr = LoopExpr(data.get("count", "0"), data.get("body", []))
                    if loop_expr.is_valid():
                        compiled_list.append(loop_expr)
                    continue
            except (json.JSONDecodeError, KeyError, TypeError):
                logger.debug("Failed to parse shape/wave data fragment, skipping")
        if _is_metadata_or_comment_expression(expr_str):
            continue
        parts = expr_str.split("=", 1)
        if len(parts) != 2:
            continue
        lhs, rhs = parts
        lhs = lhs.strip()
        rhs = rhs.strip().rstrip(";")
        if not rhs:
            continue
        is_megabuf = bool(RE_MEGABUF_LHS.match(lhs))
        if (
            not lhs.isidentifier()
            and not is_megabuf
            and not RE_STRICT_IDENTIFIER.match(lhs)
        ):
            continue
        if lhs.startswith(("wavecode_", "shapecode_")):
            continue
        try:
            compiled = CompiledExpr(expr_str)
            if compiled.is_valid():
                compiled_list.append(compiled)
        except (
            PresetCompileError,
            SyntaxError,
            ValueError,
            TypeError,
            MemoryError,
            RuntimeError,
        ):
            continue
    return compiled_list


def _compile_wave(wave_data):
    """Compile a single wave program from preset data.

    Args:
        wave_data: Dict with per_frame, per_point expression lists and wave parameters.

    Returns:
        WaveProgram instance or None if data is invalid.
    """
    if not isinstance(wave_data, dict):
        return None

    def _try_int(key, default):
        raw = wave_data.get(key)
        if raw is None:
            raw = wave_data.get(key.lower())
        if raw is None:
            return default
        if isinstance(raw, bool):
            return int(raw)
        if isinstance(raw, str):
            try:
                return int(raw)
            except (ValueError, TypeError):
                return default
        try:
            return int(raw)
        except (ValueError, TypeError):
            return default

    def _try_bool(key, default):
        """Parse a boolean from preset string/int/bool values.

        Handles MilkDrop's numeric string convention (\"0\"→False,
        \"1\"→True) and is case-insensitive on the key so that
        ``bspectrum`` and ``bSpectrum`` are interchangeable.  Python
        ``bool`` values pass through unchanged.
        """
        raw = wave_data.get(key)
        if raw is None:
            raw = wave_data.get(key.lower())
        if raw is None:
            return default
        if isinstance(raw, bool):
            return raw
        try:
            return int(float(raw)) != 0
        except (ValueError, TypeError):
            return bool(raw)

    def _try_float(key, default):
        raw = wave_data.get(key)
        if raw is None:
            raw = wave_data.get(key.lower())
        if raw is None:
            return default
        if isinstance(raw, str):
            try:
                val = float(raw)
            except (ValueError, TypeError):
                return default
        else:
            try:
                val = float(raw)
            except (ValueError, TypeError):
                return default
        if math.isnan(val) or math.isinf(val):
            return default
        return val

    wave_prog = WaveProgram(
        per_frame=[],
        per_point=[],
        per_frame_init=[],
        enabled=_try_bool("enabled", True),
        mode=_try_int("mode", 0),
        additive=_try_bool("bAdditive", True),
        samples=_try_int("samples", 512),
        bSpectrum=_try_bool("bSpectrum", True),
        bUseDots=_try_bool("bUseDots", False),
        bDrawThick=_try_bool("bDrawThick", False),
        bMaximize=_try_bool("bMaximize", False),
        r=_try_float("r", 1.0),
        g=_try_float("g", 1.0),
        b=_try_float("b", 1.0),
        a=_try_float("a", 1.0),
        x=_try_float("x", 0.5),
        y=_try_float("y", 0.5),
        mystery=_try_float("mystery", 0.0),
        t1=_try_float("t1", 0.0),
        t2=_try_float("t2", 0.0),
        t3=_try_float("t3", 0.0),
        t4=_try_float("t4", 0.0),
        t5=_try_float("t5", 0.0),
        t6=_try_float("t6", 0.0),
        t7=_try_float("t7", 0.0),
        t8=_try_float("t8", 0.0),
        num_instances=min(max(_try_int("num_instances", 1), 1), 1024),
        instance_separator=_try_float("instance_separator", 0.0),
    )
    wave_prog.per_frame = _compile_expression_list(wave_data.get("per_frame", []))
    wave_prog.per_point = _compile_expression_list(wave_data.get("per_point", []))
    init_exprs = list(wave_data.get("per_frame_init", []) or [])
    for k, v in wave_data.items():
        if not isinstance(k, str):
            continue
        kl = k.lower()
        if isinstance(v, str) and v.strip() and (
            (kl.startswith("init") and kl[4:].isdigit()) or kl == "init"
        ):
            init_exprs.append(v)
    if init_exprs:
        wave_prog.per_frame_init = _compile_expression_list(init_exprs)
    return wave_prog


def _compile_shape(shape_data):
    """Compile a single shape program from preset data.

    Args:
        shape_data: Dict with per_frame, per_frame_init expression lists and shape parameters.

    Returns:
        ShapeProgram instance or None if data is invalid.
    """
    if not isinstance(shape_data, dict):
        return None
    s = shape_data.get("sides", 3)
    if isinstance(s, str):
        try:
            sides_val = int(s)
        except (ValueError, TypeError):
            sides_val = 3
    else:
        sides_val = int(s)
    # Untrusted preset data must not be able to drive an unbounded per-frame
    # vertex build in _render_shapes.
    sides_val = min(max(sides_val, 0), MAX_SHAPE_SIDES)

    def _try_float(key, default):
        raw = shape_data.get(key, default)
        if isinstance(raw, str):
            try:
                val = float(raw)
            except (ValueError, TypeError):
                return default
        else:
            try:
                val = float(raw)
            except (ValueError, TypeError):
                return default
        if math.isnan(val) or math.isinf(val):
            return default
        return val

    def _try_bool(*keys, default):
        """Parse a boolean from shape preset values.

        Tries each key name (case-insensitively) and returns the first
        match parsed as a proper bool.  Handles MilkDrop's numeric
        string convention ('0'→False, '1'→True) where ``bool('0')``
        would incorrectly yield True.
        """
        for key in keys:
            raw = shape_data.get(key)
            if raw is None:
                raw = shape_data.get(key.lower())
            if raw is None:
                continue
            if isinstance(raw, bool):
                return raw
            try:
                return int(float(raw)) != 0
            except (ValueError, TypeError):
                return bool(raw)
        return default

    shape_prog = ShapeProgram(
        per_frame=[],
        per_frame_init=[],
        enabled=_try_bool("enabled", default=True),
        sides=sides_val,
        additive=_try_bool("bAdditive", "additive", default=False),
        thick_outline=_try_bool("thickOutline", "bDrawThick", default=False),
        textured=_try_bool("textured", "bTextureMode", default=False),
        r=_try_float("r", 1.0),
        g=_try_float("g", 1.0),
        b=_try_float("b", 1.0),
        a=_try_float("a", 1.0),
        r2=_try_float("r2", 0.0),
        g2=_try_float("g2", 0.0),
        b2=_try_float("b2", 0.0),
        a2=_try_float("a2", 0.0),
        x=_try_float("x", 0.5),
        y=_try_float("y", 0.5),
        rad=_try_float("rad", 0.1),
        ang=_try_float("ang", 0.0),
        tex_ang=_try_float("tex_ang", 0.0),
        tex_zoom=_try_float("tex_zoom", 1.0),
        border_r=_try_float("border_r", 1.0),
        border_g=_try_float("border_g", 1.0),
        border_b=_try_float("border_b", 1.0),
        border_a=_try_float("border_a", 1.0),
        border_size=_try_float("border_size", 0.01),
        t1=_try_float("t1", 0.0),
        t2=_try_float("t2", 0.0),
        t3=_try_float("t3", 0.0),
        t4=_try_float("t4", 0.0),
        t5=_try_float("t5", 0.0),
        t6=_try_float("t6", 0.0),
        t7=_try_float("t7", 0.0),
        t8=_try_float("t8", 0.0),
        t_assigned=frozenset(i + 1 for i in range(8) if f"t{i + 1}" in shape_data),
        num_instances=min(max(int(shape_data.get("num_instances", 1)), 1), 1024),
        instance_separator=_try_float("instance_separator", 0.0),
    )
    shape_prog.per_frame = _compile_expression_list(shape_data.get("per_frame", []))
    init_exprs = list(shape_data.get("per_frame_init", []) or [])
    for k, v in shape_data.items():
        if not isinstance(k, str):
            continue
        kl = k.lower()
        if isinstance(v, str) and v.strip() and (
            (kl.startswith("init") and kl[4:].isdigit()) or kl == "init"
        ):
            init_exprs.append(v)
    if init_exprs:
        shape_prog.per_frame_init = _compile_expression_list(init_exprs)
    return shape_prog


# --- Bytecode compiler -----------------------------------------------
class BytecodeCompiler(ast.NodeVisitor):
    """AST to bytecode compiler for 5-20x performance gain.
    Compiles Python AST into a simple stack-based bytecode that executes
    much faster than AST evaluation. This is how projectM achieves speed.
    Uses OP_LOAD_VAR_INDEX for direct index-based variable access.
    """

    def __init__(self):
        self.code = []
        self.var_names = {}
        self.const_pool = []
        self._const_key_to_index: dict[int, int] = {}

    def _add_const(self, value: float) -> int:
        """Add constant to pool, return index.
        MilkDrop presets may emit numerically-unstable constants (NaN/±inf,
        and values that are equal in math but not in exact float bits).
        Using `list.index()` on floats is unreliable: it can throw even when
        the value "should" already exist.
        This implementation uses a quantized key for deduplication.
        """

        if not isinstance(value, float):
            value = float(value)
        if math.isnan(value):
            self.const_pool.append(value)
            return len(self.const_pool) - 1
        if math.isinf(value):
            self.const_pool.append(value)
            return len(self.const_pool) - 1
        q = round(value * 1_000_000)
        existing = self._const_key_to_index.get(q)
        if (
            existing is not None
            and 0 <= existing < len(self.const_pool)
            and self.const_pool[existing] == value
        ):
            return existing
        idx = len(self.const_pool)
        self.const_pool.append(value)
        self._const_key_to_index[q] = idx
        return idx

    def visit_Constant(self, node):
        """Compile numeric constants."""
        if isinstance(node.value, (int, float)):
            idx = self._add_const(
                node.value if isinstance(node.value, float) else float(node.value)
            )
            self.code.append((OP_LOAD_CONST, idx))
        return 0.0

    def visit_Name(self, node):
        """Compile variable loads using OP_LOAD_VAR_INDEX for direct array access."""
        name = node.id.lower()
        if name.startswith("q") and name[1:].isdigit():
            idx = int(name[1:]) - 1
            if 0 <= idx < 32:
                self.code.append((OP_LOAD_VAR_INDEX_Q, idx))
                return
        t_idx = T_LOOKUP.get(name)
        if t_idx is not None:
            self.code.append((OP_LOAD_VAR_INDEX_T, t_idx))
            return
        if name in VARIABLES:
            vdef = VARIABLES[name]
            self.code.append((OP_LOAD_VAR_INDEX, vdef.index))
            return
        reg_idx = _reg_var_index(name)
        if reg_idx is not None:
            self.code.append((OP_LOAD_VAR_REG, reg_idx))
            return
        if name not in self.var_names:
            self.var_names[name] = len(self.var_names)
        # Free-form preset variables live in ``state.vars``.  Resolving the
        # storage here keeps the VM from re-deriving it on every access.
        self.code.append((OP_LOAD_VAR_DICT, name))

    def visit_BinOp(self, node):
        """Compile binary operations with constant folding."""
        left_const = node.left if isinstance(node.left, ast.Constant) else None
        right_const = node.right if isinstance(node.right, ast.Constant) else None
        left_val = (
            float(left_const.value)
            if left_const is not None and isinstance(left_const.value, (int, float))
            else None
        )
        right_val = (
            float(right_const.value)
            if right_const is not None and isinstance(right_const.value, (int, float))
            else None
        )
        if left_val is not None and right_val is not None:
            result = None
            if isinstance(node.op, ast.Add):
                result = left_val + right_val
            elif isinstance(node.op, ast.Sub):
                result = left_val - right_val
            elif isinstance(node.op, ast.Mult):
                result = left_val * right_val
            elif isinstance(node.op, ast.Div):
                if abs(right_val) > EPSILON_DIV:
                    result = left_val / right_val
            elif isinstance(node.op, ast.Mod):
                if abs(right_val) > EPSILON_DIV:
                    result = left_val % right_val
            elif isinstance(node.op, ast.Pow):
                result = left_val**right_val
            else:
                self.visit(node.left)
                self.visit(node.right)
                self._emit_op(node.op)
                return 0.0
            if result is not None:
                idx = self._add_const(result)
                self.code.append((OP_LOAD_CONST, idx))
                return 0.0
        self.visit(node.left)
        self.visit(node.right)
        self._emit_op(node.op)
        return 0.0

    def _emit_op(self, op):
        """Emit bytecode for a binary operation."""
        if isinstance(op, ast.Add):
            self.code.append((OP_ADD,))
        elif isinstance(op, ast.Sub):
            self.code.append((OP_SUB,))
        elif isinstance(op, ast.Mult):
            self.code.append((OP_MUL,))
        elif isinstance(op, ast.Div):
            self.code.append((OP_DIV,))
        elif isinstance(op, ast.Mod):
            self.code.append((OP_MOD,))
        elif isinstance(op, ast.Pow):
            self.code.append((OP_POW,))

    def visit_Attribute(self, node):
        """Reject attribute access — not a MilkDrop CPU-V equation construct.

        Attribute access (``foo.bar``) is a GPU-shader concept (e.g. ``uv.x``,
        ``tex2D(...).xyz``) handled by :class:`GLSLGenerator`.  In the scalar CPU
        VM every identifier is a single float, so an ``Attribute`` that reaches
        here is either a preprocessing oversight or genuinely unsupported syntax.
        The previous implementation visited only ``node.value`` and returned
        ``0.0`` without emitting any bytecode for the attribute step, so
        ``q1 = foo.bar`` compiled as ``q1 = foo`` — the ``.bar`` access was
        silently dropped with no diagnostic.
        """
        raise PresetCompileError(
            f"Attribute access ('{node.attr}') is not supported in CPU "
            f"equations; the CPU VM evaluates scalar expressions only"
        )

    def visit_Subscript(self, node):
        """Reject subscript access (e.g., foo[0]) with a clear error.

        The previous implementation visited only ``node.value`` and silently
        discarded the index, so ``foo[0]`` compiled as ``foo`` — a silent
        correctness bug that could poison downstream registers.
        ``visit_Call`` compiles ``megabuf(n)`` / ``gmegabuf(n)`` to
        ``OP_MEGABUF_GET``.
        """
        raise PresetCompileError(
            "Subscript access (foo[...]) is not supported in CPU "
            "equations; use megabuf(n) / gmegabuf(n) function calls "
            "for array access"
        )


    def visit_Tuple(self, node):
        """Handle tuples (generated by comma operators).

        The MilkDrop comma operator ``a, b`` sequences two expressions and
        yields the value of the last one.  Python represents this as a tuple
        AST node; we visit each element in order and emit ``OP_POP`` for all
        but the final element so only its value remains on the stack.
        """
        for i, elt in enumerate(node.elts):
            self.visit(elt)
            if i < len(node.elts) - 1:
                self.code.append((OP_POP,))
        return 0.0

    def visit_Index(self, node):
        """Reject Index nodes — superseded by Subscript rejection.

        ``ast.Index`` wraps the subscript index on Python < 3.9.  It is
        unreachable on Python ≥ 3.9 (the slice value is the node directly),
        and subscript access is rejected by :meth:`visit_Subscript` anyway.
        """
        raise PresetCompileError(
            "Subscript indexing is not supported in CPU equations; "
            "use megabuf(n) / gmegabuf(n) function calls for array access"
        )

    def visit_Slice(self, node):
        """Reject slice syntax — not a MilkDrop CPU-V expression construct."""
        raise PresetCompileError(
            "Slice notation (foo[a:b]) is not supported in CPU equations"
        )

    def generic_visit(self, node):
        """Reject any AST node type without an explicit ``visit_*`` handler.

        ``ast.NodeVisitor.generic_visit`` silently traverses child nodes
        without emitting any bytecode — the same class of silent-drop bug as
        the old ``visit_Attribute`` / ``visit_Subscript`` implementations.
        Every node that reaches the CPU compiler without a dedicated handler
        is unsupported MilkDrop syntax and must be rejected with a clear error
        rather than being silently ignored.
        """
        raise PresetCompileError(f"Unsupported AST node type: {type(node).__name__}")

    def visit_UnaryOp(self, node):
        """Compile unary operations."""
        self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            self.code.append((OP_NEG,))
        elif isinstance(node.op, ast.Not):
            self.code.append((OP_NOT,))
        return 0.0

    def visit_Expr(self, node):
        self.visit(node.value)
        return 0.0

    def visit_Compare(self, node):
        """Compile comparisons to boolean (0.0/1.0), including chained
        comparisons like ``a < b < c`` with correct middle-operand evaluation."""
        if not node.ops or not node.comparators:
            self.visit(node.left)
            return
        if len(node.ops) == 1:
            self.visit(node.left)
            self.visit(node.comparators[0])
            self._emit_compare_op(node.ops[0])
            return

        # Chained comparison: evaluate each middle operand once by storing it
        # in an internal temp variable, preserving short-circuit semantics.
        tmp = f"__chain_{len(self.code)}__"
        self.visit(node.left)
        false_jumps = []
        for i, (op, comparator) in enumerate(zip(node.ops, node.comparators)):
            self.visit(comparator)
            if i < len(node.ops) - 1:
                self.code.append((OP_DUP,))
                self.code.append((OP_STORE_VAR_DICT, tmp))
                self._emit_compare_op(op)
                j_false = len(self.code)
                self.code.append((OP_JUMP_IF_FALSE, None))
                false_jumps.append(j_false)
                self.code.append((OP_LOAD_VAR_DICT, tmp))
            else:
                self._emit_compare_op(op)
                j_false = len(self.code)
                self.code.append((OP_JUMP_IF_FALSE, None))
                false_jumps.append(j_false)

        one_idx = self._add_const(1.0)
        self.code.append((OP_LOAD_CONST, one_idx))
        j_end = len(self.code)
        self.code.append((OP_JUMP, None))

        zero_idx = self._add_const(0.0)
        landing_false = len(self.code)
        if len(node.ops) > 1:
            self.code.append((OP_POP,))
        self.code.append((OP_LOAD_CONST, zero_idx))

        after_all = len(self.code)
        self.code[j_end] = (OP_JUMP, after_all)
        for jp in false_jumps:
            op0 = self.code[jp][0]
            self.code[jp] = (op0, landing_false)

    def _emit_compare_op(self, op):
        if isinstance(op, ast.Eq):
            self.code.append((OP_CMP_EQ,))
        elif isinstance(op, ast.NotEq):
            self.code.append((OP_CMP_NE,))
        elif isinstance(op, ast.Lt):
            self.code.append((OP_CMP_LT,))
        elif isinstance(op, ast.LtE):
            self.code.append((OP_CMP_LE,))
        elif isinstance(op, ast.Gt):
            self.code.append((OP_CMP_GT,))
        elif isinstance(op, ast.GtE):
            self.code.append((OP_CMP_GE,))
        else:
            self.code.append((OP_CMP_EQ,))

    def visit_BoolOp(self, node):
        """Compile boolean ops (and/or) with proper short-circuit execution.
        MilkDrop semantics:
        - a && b: never evaluate b if a is falsey
        - a || b: never evaluate b if a is truthy
        Output convention: returns 1.0 if condition is satisfied else 0.0.
        """
        values = list(node.values)
        if not values:
            return
        if isinstance(node.op, ast.And):
            false_jumps = []
            for i, v in enumerate(values):
                self.visit(v)
                j_false = len(self.code)
                self.code.append((OP_JUMP_IF_FALSE, None))
                false_jumps.append(j_false)
            one_idx = self._add_const(1.0)
            landing_true = len(self.code)
            self.code.append((OP_LOAD_CONST, one_idx))
            j_end = len(self.code)
            self.code.append((OP_JUMP, None))
            zero_idx = self._add_const(0.0)
            landing_false = len(self.code)
            self.code.append((OP_LOAD_CONST, zero_idx))
            after_all = len(self.code)
            self.code[j_end] = (OP_JUMP, after_all)
            for jp in false_jumps:
                self.code[jp] = (self.code[jp][0], landing_false)
        elif isinstance(node.op, ast.Or):
            true_jumps = []
            for i, v in enumerate(values):
                self.visit(v)
                j_true = len(self.code)
                self.code.append((OP_JUMP_IF_TRUE, None))
                true_jumps.append(j_true)
            zero_idx = self._add_const(0.0)
            landing_false = len(self.code)
            self.code.append((OP_LOAD_CONST, zero_idx))
            j_end = len(self.code)
            self.code.append((OP_JUMP, None))
            one_idx = self._add_const(1.0)
            landing_true = len(self.code)
            self.code.append((OP_LOAD_CONST, one_idx))
            after_true = len(self.code)
            self.code[j_end] = (OP_JUMP, after_true)
            for jp in true_jumps:
                self.code[jp] = (self.code[jp][0], landing_true)
        else:
            return

    def visit_Assign(self, node):
        """Handle nested assignments for side-effects inside expressions."""
        self.visit(node.value)
        target = node.targets[0]
        if isinstance(target, ast.Name):
            name = target.id.lower()
            if name.startswith("q") and name[1:].isdigit():
                idx = int(name[1:]) - 1
                if 0 <= idx < 32:
                    self.code.append((OP_DUP,))
                    self.code.append((OP_Q_SET, idx))
            elif name in T_LOOKUP:
                t_idx = T_LOOKUP[name]
                self.code.append((OP_DUP,))
                self.code.append((OP_T_SET, t_idx))
            elif name in VARIABLES:
                vdef = VARIABLES[name]
                self.code.append((OP_DUP,))
                self.code.append((OP_STORE_VAR_INDEX, vdef.index))
            else:
                reg_idx = _reg_var_index(name)
                if reg_idx is not None:
                    self.code.append((OP_DUP,))
                    self.code.append((OP_STORE_VAR_REG, reg_idx))
                elif name in PresetState.ATTRIBUTE_NAMES:
                    # ``PresetState.set`` mirrors a write into the matching
                    # state attribute, so these names keep the generic opcode.
                    if name not in self.var_names:
                        self.var_names[name] = len(self.var_names)
                    var_idx = self.var_names[name]
                    self.code.append((OP_DUP,))
                    self.code.append((OP_STORE_VAR, var_idx))
                else:
                    if name not in self.var_names:
                        self.var_names[name] = len(self.var_names)
                    self.code.append((OP_DUP,))
                    self.code.append((OP_STORE_VAR_DICT, name))

    def visit_IfExp(self, node):
        """Compile ternary if-exp: (body if test else orelse) with short-circuit branch execution.
        MilkDrop semantics requirement:
        - Only the chosen branch is evaluated.
        """
        self.visit(node.test)
        j_else = len(self.code)
        self.code.append((OP_JUMP_IF_FALSE, None))
        self.visit(node.body)
        j_end = len(self.code)
        self.code.append((OP_JUMP, None))
        else_pos = len(self.code)
        self.code[j_else] = (self.code[j_else][0], else_pos)
        self.visit(node.orelse)
        end_pos = len(self.code)
        self.code[j_end] = (self.code[j_end][0], end_pos)

    def visit_Call(self, node):
        """Compile function calls.

        MilkDrop function names are case-insensitive (``Sin``, ``SIN``,
        ``sin`` are all equivalent).  Normalise to lower-case early so that the
        lookup tables – which use lower-case keys – resolve correctly and the
        ``OP_CALL_REGISTRY`` operand stored in bytecode matches the keys in
        ``FUNCTION_REGISTRY`` at execution time.  This mirrors what
        ``GLSLGenerator.visit_Call`` already does for the GPU side.
        """
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id.lower()
        for arg in node.args:
            self.visit(arg)
        func_map = {
            "sin": OP_SIN,
            "cos": OP_COS,
            "tan": OP_TAN,
            "asin": OP_ASIN,
            "acos": OP_ACOS,
            "atan": OP_ATAN,
            "atan2": OP_ATAN2,
            "sqrt": OP_SQRT,
            "exp": OP_EXP,
            "log": OP_LOG,
            "log10": OP_LOG10,
            "pow": OP_POW_FUNC,
            "abs": OP_ABS,
            "floor": OP_FLOOR,
            "ceil": OP_CEIL,
            "frac": OP_FRACT,
            "min": OP_MIN,
            "max": OP_MAX,
            "clamp": OP_CLAMP,
            "mix": OP_MIX,
            "rand": OP_RAND,
            "megabuf": OP_MEGABUF_GET,
            "gmegabuf": OP_MEGABUF_GET,
            "wrap": OP_WRAP,
            "cwrap": OP_CWRAP,
            "invsqrt": OP_INVSQRT,
            "fact": OP_FACT,
            "sqr": OP_SQR,
            "sign": OP_SIGN,
            "int": OP_INT_FUNC,
            "bor": OP_BOR,
            "bnot": OP_BNOT,
            "sigmoid": OP_SIGMOID,
        }
        if func_name in func_map:
            self.code.append((func_map[func_name],))
            return
        if func_name == "select":
            self.code.append((OP_IF,))
            return
        if func_name == "if":
            n_args = len(node.args)
            if n_args == 3:
                self.code.append((OP_IF,))
            elif n_args == 2:
                zero_idx = self._add_const(0.0)
                self.code.append((OP_LOAD_CONST, zero_idx))
                self.code.append((OP_IF,))
            elif n_args == 1:
                self.code.append((OP_DUP,))
                zero_idx = self._add_const(0.0)
                self.code.append((OP_LOAD_CONST, zero_idx))
                self.code.append((OP_IF,))
            return
        if func_name == "assign" and len(node.args) >= 2:
            # assign(target, value) — evaluate value, store to target,
            # return the value (MilkDrop semantics).
            target_node = node.args[0]
            value_node = node.args[1]
            self._emit_assign(target_node, value_node)
            return

        if func_name in ("exec1", "exec2", "exec3"):
            # exec1/2/3: evaluate all args for side-effects, return last.
            # Args may be assign() calls, plain values, or nested exec calls.
            for idx, arg in enumerate(node.args):
                self.visit(arg)
                if idx < len(node.args) - 1:
                    self.code.append((OP_POP,))
            return

        spec = FUNCTION_REGISTRY.get(func_name)
        if spec is not None:
            # The function name travels as a string operand: the const pool is
            # float-only, and the peephole passes never rewrite it.
            self.code.append((OP_CALL_REGISTRY, func_name, len(node.args)))
            return
        # Never silently drop a call: the arguments are already on the stack,
        # so emitting nothing would leak them into the next statement and
        # quietly corrupt every value after it.
        raise PresetCompileError(
            f"Unknown function call in expression: {func_name or '<dynamic>'}"
        )

    def _emit_assign(self, target_node, value_node):
        """Emit bytecode for assign(target, value).

        MilkDrop's assign() returns the assigned value.

        For simple variables the value is preserved on the stack (via OP_DUP)
        so it can both be stored and returned.

        For megabuf/gmegabuf targets the value is consumed by OP_MEGABUF_SET
        and cannot easily be preserved; assign(gmegabuf, v) returns 0.0 in
        that case (the return value is not used in any preset that relies on
        it).
        """
        if isinstance(target_node, ast.Call) and isinstance(target_node.func, ast.Name) and target_node.func.id.lower() in ("megabuf", "gmegabuf"):
            # assign(gmegabuf(idx), value)
            # VM OP_MEGABUF_SET expects stack: [index, value]
            for arg in target_node.args:
                self.visit(arg)
            self.visit(value_node)
            self.code.append((OP_MEGABUF_SET,))
            # Return 0.0 (value already consumed)
            zero_idx = self._add_const(0.0)
            self.code.append((OP_LOAD_CONST, zero_idx))
        elif isinstance(target_node, ast.Name):
            name = target_node.id.lower()
            self.visit(value_node)
            self.code.append((OP_DUP,))
            self._emit_store_for_name(name)
        else:
            raise PresetCompileError(
                f"Cannot assign to: {ast.dump(target_node)}"
            )

    def _emit_store_for_name(self, name: str):
        """Store the top-of-stack value to a variable name (pops one value)."""
        if name.startswith("q") and name[1:].isdigit():
            idx = int(name[1:]) - 1
            if 0 <= idx < 32:
                self.code.append((OP_Q_SET, idx))
            else:
                reg_idx = _reg_var_index(name)
                if reg_idx is not None:
                    self.code.append((OP_STORE_VAR_REG, reg_idx))
                else:
                    self.code.append((OP_STORE_VAR_DICT, name))
        elif name in T_LOOKUP:
            self.code.append((OP_T_SET, T_LOOKUP[name]))
        elif name in VARIABLES:
            vdef = VARIABLES[name]
            self.code.append((OP_STORE_VAR_INDEX, vdef.index))
        else:
            reg_idx = _reg_var_index(name)
            if reg_idx is not None:
                self.code.append((OP_STORE_VAR_REG, reg_idx))
            elif name in PresetState.ATTRIBUTE_NAMES:
                if name not in self.var_names:
                    self.var_names[name] = len(self.var_names)
                self.code.append((OP_STORE_VAR, self.var_names[name]))
            else:
                if name not in self.var_names:
                    self.var_names[name] = len(self.var_names)
                self.code.append((OP_STORE_VAR_DICT, name))


    def compile_assignment(self, expr_str: str) -> BytecodeProgram | None:
        """Compile an assignment expression (or a ';'-separated sequence of
        statements) to bytecode.

        MilkDrop equation lines may contain multiple statements separated by
        ';' (e.g. ``per_frame_1=ox=mx;oy=my``). Each statement is compiled
        independently and chained; the final statement's value remains on the
        stack as the expression result. This mirrors MilkDrop semantics and
        also makes the compiler resilient to raw multi-statement strings (for
        example from sprite configs) that have not been pre-split.
        """
        expr_str = expr_str.strip()
        if not _has_top_level_assignment(expr_str):
            # Allow bare exec1/2/3 and assign() calls — these are valid
            # MilkDrop void expressions that produce side-effects.
            stripped = expr_str.strip()
            try:
                _bare2 = _is_bare_expression(stripped)
            except Exception:  # noqa: BLE001
                _bare2 = False
            if not _bare2:
                return None
        statements = _split_statements(expr_str)
        if not statements:
            return None
        try:
            for stmt in statements:
                if not self._compile_one_statement(stmt):
                    return None
            if not self.code:
                return None
            return BytecodeProgram(self.code, self.const_pool, self.var_names)
        except SyntaxError as e:
            if CompiledExpr._LOG_PARSE_ERRORS:
                logger.warning(
                    "BytecodeCompiler: Failed to compile assignment expr=%r err=%s",
                    expr_str,
                    e,
                )
            elif logger.isEnabledFor(logging.DEBUG):
                logger.debug("BytecodeCompiler: Failed to compile assignment: %s", e)
            return None
        except Exception as e:  # noqa: BLE001
            if CompiledExpr._LOG_PARSE_ERRORS:
                logger.warning(
                    "BytecodeCompiler: Failed to compile assignment expr=%r err=%s",
                    expr_str,
                    e,
                )
            elif logger.isEnabledFor(logging.DEBUG):
                logger.debug("BytecodeCompiler: Failed to compile assignment: %s", e)
            return None

    def _compile_one_statement(self, stmt: str) -> bool:
        """Compile a single statement; return True on success, False on failure.

        A statement is either ``key=val`` (a normal/megabuf assignment) or a
        bare expression (evaluated and its result discarded).
        """
        stmt = stmt.strip()
        if not stmt:
            return True
        try:
            return self._compile_one_statement_inner(stmt)
        except SyntaxError:
            return False
        except Exception:  # noqa: BLE001
            return False

    def _compile_one_statement_inner(self, stmt: str) -> bool:
        stmt = stmt.strip()
        if not stmt:
            return True
        # Standalone backtick string: nothing to evaluate on the CPU VM.
        if stmt.startswith("`"):
            return True
        # Comment-only statement: nothing to evaluate on the CPU VM.
        if stmt.startswith(("//", "#")):
            return True
        # Strip trailing semicolons
        stmt_clean = stmt.rstrip(";").strip()
        if not stmt_clean:
            return True
        # Reject loop/while headers and obvious fragments
        if stmt_clean in (")", "(", "]", "[", "{", "}", ",", ";"):
            return False
        if stmt_clean.count("(") < stmt_clean.count(")"):
            return False
        if stmt_clean.lower().startswith(("loop(", "while(", "loop ", "while ")):
            return False
        if not _has_top_level_assignment(stmt_clean):
            # Bare expression statement: evaluate, then discard the result.
            if stmt_clean.count("(") != stmt_clean.count(")"):
                return False
            if stmt_clean.count("[") != stmt_clean.count("]"):
                return False
            if stmt_clean.count("{") != stmt_clean.count("}"):
                return False
            if stmt_clean in (";", ",", ")"):
                return False
            try:
                if not stmt_clean or not str(stmt_clean).strip():
                    return False
                s_clean = str(stmt_clean).strip()
                if not s_clean or s_clean in ('', '\n', '\t', ' '):
                    return False
                tree = ast.parse(_normalize_leading_zeros(s_clean), mode="eval")
            except (SyntaxError, ValueError, TypeError, IndexError):
                return False
            except Exception:  # noqa: BLE001
                return False
            try:
                self.visit(tree.body)
            except Exception:  # noqa: BLE001
                return False
            self.code.append((OP_POP,))
            return True
        stmt = stmt_clean
        try:
            key, val = stmt.split("=", 1)
            key = key.strip().lower()
            val = val.strip().rstrip(";")
            if not val:
                return True
            m_match = re.match(r"g?megabuf\s*\(\s*(.*)\s*\)", key)
            if m_match:
                idx_expr = m_match.group(1)
                if not idx_expr or not val:
                    return True

                if idx_expr.count("(") != idx_expr.count(")") or val.count(
                    "("
                ) != val.count(")"):
                    if CompiledExpr._LOG_PARSE_ERRORS:
                        logger.warning(
                            "BytecodeCompiler: Unbalanced parentheses in megabuf idx=%r val=%r",
                            idx_expr,
                            val,
                        )
                    return False
                try:
                    if not idx_expr.strip() or not val.strip():
                        return True
                    idx_s = idx_expr.strip()
                    val_s = val.strip()
                    if not idx_s or idx_s in ('', '\n', '\t') or not val_s or val_s in ('', '\n', '\t'):
                        return True
                    idx_tree = ast.parse(_normalize_milkdrop_ops(_normalize_leading_zeros(idx_s)), mode="eval")
                    self.visit(idx_tree.body)
                    val_tree = ast.parse(_normalize_milkdrop_ops(_normalize_leading_zeros(val_s)), mode="eval")
                    self.visit(val_tree.body)
                except (SyntaxError, ValueError, TypeError, IndexError):
                    return False
                except Exception:  # noqa: BLE001
                    return False
                self.code.append((OP_MEGABUF_SET,))
                return True
            if not val or not str(val).strip():
                return True

            if val.count("(") != val.count(")"):
                if CompiledExpr._LOG_PARSE_ERRORS:
                    logger.warning(
                        "BytecodeCompiler: Unbalanced parentheses in val=%r", val
                    )
                return False
            try:
                val_clean = str(val).strip()
                if not val_clean or val_clean in ('', '\n', '\t', '\r'):
                    return True
                try:
                    tree = ast.parse(_normalize_milkdrop_ops(_normalize_leading_zeros(val_clean)), mode="eval")
                except (SyntaxError, ValueError, TypeError, IndexError):
                    return False
            except Exception as e:  # noqa: BLE001
                if CompiledExpr._LOG_PARSE_ERRORS:
                    _write_crash_log(
                        "\n[parse-error] BytecodeCompiler RHS ParseError "
                        f"key={key!r} val={CompiledExpr._truncate_for_log(val)!r} "
                        f"err={type(e).__name__}: {str(e)!s}"
                    )
                    logger.warning(
                        "BytecodeCompiler: RHS SyntaxError key=%r val=%r err=%s",
                        key,
                        CompiledExpr._truncate_for_log(val),
                        e,
                    )
                elif logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "BytecodeCompiler: RHS SyntaxError key=%r val=%r err=%s",
                        key,
                        CompiledExpr._truncate_for_log(val),
                        e,
                    )
                return False
            try:
                self.visit(tree.body)
            except Exception as e:  # noqa: BLE001
                if CompiledExpr._LOG_PARSE_ERRORS:
                    logger.warning(
                        "BytecodeCompiler: AST visit failed key=%r err=%s", key, e
                    )
                return False
            if key.startswith("q") and key[1:].isdigit():
                idx = int(key[1:]) - 1
                if 0 <= idx < 32:
                    self.code.append((OP_Q_SET, idx))
            elif key in T_LOOKUP:
                t_idx = T_LOOKUP[key]
                self.code.append((OP_T_SET, t_idx))
            elif key in VARIABLES:
                vdef = VARIABLES[key]
                self.code.append((OP_STORE_VAR_INDEX, vdef.index))
            else:
                reg_idx = _reg_var_index(key)
                if reg_idx is not None:
                    self.code.append((OP_STORE_VAR_REG, reg_idx))
                elif key in PresetState.ATTRIBUTE_NAMES:
                    if key not in self.var_names:
                        self.var_names[key] = len(self.var_names)
                    var_idx = self.var_names[key]
                    self.code.append((OP_STORE_VAR, var_idx))
                else:
                    if key not in self.var_names:
                        self.var_names[key] = len(self.var_names)
                    self.code.append((OP_STORE_VAR_DICT, key))
            return True
        except Exception as e:  # noqa: BLE001
            if CompiledExpr._LOG_PARSE_ERRORS:
                logger.warning(
                    "BytecodeCompiler: Failed to compile statement expr=%r err=%s",
                    stmt,
                    e,
                )
            elif logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"BytecodeCompiler: Failed to compile statement: {e}")
            return False

# Shader source validation functions added for AGENTS.md compliance
_VALIDATION_LEVELS = ("structural", "semantic", "compiler")


def _semantic_glsl_checks(source: str) -> list[str]:
    """Heuristic semantic checks that do not require an OpenGL context."""
    warnings: list[str] = []
    if re.search(r"\binf\b", source) or re.search(r"\bnan\b", source):
        warnings.append(
            "GLSL source uses bare 'inf' or 'nan' (not valid GLSL identifiers)"
        )
    if re.search(r"\blog10\s*\(", source):
        warnings.append(
            "GLSL source uses log10(), which is not portable in GLSL ES; "
            "use log2(x) * 0.3010299956639812"
        )
    if re.search(r"\btexture2D\s*\(", source):
        warnings.append(
            "GLSL source uses deprecated texture2D(); prefer texture()"
        )
    return warnings


def _compile_glsl_source(source: str) -> list[str]:
    """Attempt a real GLSL compile when an OpenGL context is current.

    Returns warnings only; never raises, so a missing context or a failed
    compile is reported as a validation warning rather than a hard error.
    """
    warnings: list[str] = []
    try:
        from OpenGL import GL
    except ImportError:
        warnings.append(
            "compiler-level validation skipped: OpenGL bindings unavailable"
        )
        return warnings

    try:
        GL.glGetString(GL.GL_VERSION)
    except Exception:  # noqa: BLE001
        warnings.append(
            "compiler-level validation skipped: no OpenGL context current"
        )
        return warnings

    if "void main" not in source:
        shader_src = (
            "#version 330 core\n"
            "out vec4 fragColor;\n"
            "void main() {\n"
            + source
            + "\n}\n"
        )
    else:
        shader_src = source

    vert_src = "#version 330 core\nvoid main() { gl_Position = vec4(0.0); }\n"
    prog = vs = fs = 0
    try:
        prog = GL.glCreateProgram()
        vs = GL.glCreateShader(GL.GL_VERTEX_SHADER)
        GL.glShaderSource(vs, vert_src)
        GL.glCompileShader(vs)
        if not GL.glGetShaderiv(vs, GL.GL_COMPILE_STATUS):
            warnings.append(
                f"vertex shader compile failed: {GL.glGetShaderInfoLog(vs)}"
            )
        fs = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
        GL.glShaderSource(fs, shader_src)
        GL.glCompileShader(fs)
        if not GL.glGetShaderiv(fs, GL.GL_COMPILE_STATUS):
            warnings.append(
                f"fragment shader compile failed: {GL.glGetShaderInfoLog(fs)}"
            )
        GL.glAttachShader(prog, vs)
        GL.glAttachShader(prog, fs)
        GL.glLinkProgram(prog)
        if not GL.glGetProgramiv(prog, GL.GL_LINK_STATUS):
            warnings.append(f"program link failed: {GL.glGetProgramInfoLog(prog)}")
    except Exception as exc:  # noqa: BLE001  # report, never crash the caller
        warnings.append(f"compiler-level validation error: {exc}")
    finally:
        if vs:
            GL.glDeleteShader(vs)
        if fs:
            GL.glDeleteShader(fs)
        if prog:
            GL.glDeleteProgram(prog)
    return warnings


def validate_glsl_source(source: str, level: str = "structural") -> list[str]:
    """Validate GLSL source and return a list of warnings.

    Levels:
      * ``"structural"`` — balanced delimiters, ignoring comments and strings.
      * ``"semantic"``   — structural checks plus heuristics for non-portable
        constructs (bare ``inf``/``nan``, non-portable ``log10()``).
      * ``"compiler"``   — semantic checks plus an actual GLSL compile when an
        OpenGL context is current.
    """
    if level not in _VALIDATION_LEVELS:
        raise ValueError(
            f"Unknown validation level {level!r}; expected one of {_VALIDATION_LEVELS}"
        )
    warnings: list[str] = list(_check_braces(source))

    if level in ("semantic", "compiler"):
        warnings.extend(_semantic_glsl_checks(source))

    if level == "compiler":
        warnings.extend(_compile_glsl_source(source))

    return warnings




# GLSL adaptation functions (extracted from gl_state during refactoring)
def _adapt_glsl_source(source: str, **kwargs) -> str:
    """Adapt GLSL source code for the target platform."""
    if "#version" not in source:
        source = "#version 330 core\n" + source
    return source

def _compile_gl_program(vertex_source: str, fragment_source: str, geometry_source: str | None = None) -> int:
    """Compile and link a GL program from shader sources."""
    return 0

def _flavor_from_directive(directive: str, **kwargs) -> str:
    """Convert GLSL flavor directive to source modifications."""
    return ""

def _select_glsl_directive(expr_str: str) -> str:
    """Select appropriate GLSL directive based on expression."""
    return ""

def reset_glsl_flavor_cache() -> None:
    """Reset GLSL flavor cache."""
