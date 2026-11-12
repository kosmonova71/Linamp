"""Safe numeric backend for the MilkDrop expression VM.

Safety contract (total functions):
- No helper raises on bad numeric input and no helper returns NaN or Inf.
- Non-finite inputs are normalized via finite-float guards; domain errors
  return a documented safe default.
- Default safe result is 0.0 for most math/domain failures.
- Exceptions to the default: ``_md_factorial`` returns 1.0 for invalid/
  negative/non-finite inputs (and clamps large inputs); ``_md_sigmoid``
  returns 0.5 for NaN/unexpected failures and saturates to 0/1 for ±Inf;
  ``_md_bool`` follows MilkDrop logical truthiness (zero false, non-zero true,
  NaN -> 0.0, Inf -> 1.0); ``_md_pow(0, 0)`` returns 1.0.
- ``_md_mod`` uses integer remainder semantics (truncated division) matching
  MilkDrop % operator.
- ``_md_band``, ``_md_bor``, ``_md_bnot`` implement integer bitwise operations
  distinct from logical ``&&``, ``||``, ``!``.
- ``_md_mix`` clamps the blend factor ``a`` to [0, 1] for safety and
  deterministic preset behavior; this is intentional and documented here.
- Randomness is explicit: pass a state object with ``rand_seed`` to
  ``md_rand`` / ``_md_rand_cpu`` when evaluation must be isolated or
  re-entrant. A module-level fallback state exists only for legacy callers.
"""

from __future__ import annotations

import math
import sys
from typing import Any

EPSILON_COMPARE = 0.00001
EPSILON_DIV = 1e-7
EPSILON_BOOL = 0.00001
EPSILON_SMALL = 1e-6
_INT_EPS = 1e-12

MAX_SAFE_FLOAT = float(sys.float_info.max)
_SAFE_FACT_MAX = 170
_MAX_FLOAT32 = 3.4028234663852886e38

OP_LOAD_CONST = 1
OP_LOAD_VAR = 2
OP_STORE_VAR = 3
OP_LOAD_VAR_INDEX = 200
OP_LOAD_VAR_INDEX_Q = 201
OP_STORE_VAR_INDEX = 202
OP_LOAD_VAR_INDEX_T = 203
OP_ADD = 4
OP_SUB = 5
OP_MUL = 6
OP_DIV = 7
OP_MOD = 8
OP_POW = 9
OP_NEG = 10
OP_SIN = 11
OP_COS = 12
OP_TAN = 13
OP_ASIN = 14
OP_ACOS = 15
OP_ATAN = 16
OP_ATAN2 = 17
OP_SQRT = 18
OP_ABS = 19
OP_FLOOR = 20
OP_CEIL = 21
OP_FRACT = 22
OP_MIN = 23
OP_MAX = 24
OP_CLAMP = 25
OP_MIX = 26
OP_RAND = 27
OP_MEGABUF_GET = 28
OP_MEGABUF_SET = 29
OP_Q_GET = 30
OP_Q_SET = 31
OP_POP = 50
OP_DUP = 51
OP_CMP_EQ = 32
OP_CMP_NE = 33
OP_CMP_LT = 34
OP_CMP_LE = 35
OP_CMP_GT = 36
OP_CMP_GE = 37
OP_AND = 38
OP_OR = 39
OP_IF = 40
OP_NOT = 41
OP_EXP = 42
OP_LOG = 43
OP_POW_FUNC = 44
OP_JUMP = 60
OP_JUMP_IF_FALSE = 61
OP_JUMP_IF_TRUE = 62
OP_WRAP = 100
OP_CWRAP = 101
OP_INVSQRT = 102
OP_FACT = 103
OP_NCHOOSEK = 104
OP_SQR = 105
OP_SIGN = 106
OP_INT_FUNC = 107
OP_LOG10 = 108
OP_BOR = 109
OP_BNOT = 110
OP_BAND = 111
OP_SIGMOID = 112
OP_T_GET = 113
OP_T_SET = 114
OP_CALL_REGISTRY = 115
OP_LOAD_VAR_DICT = 116
OP_STORE_VAR_DICT = 117
OP_LOAD_VAR_REG = 118
OP_STORE_VAR_REG = 119

_md_floor_cached = math.floor
_md_ceil_cached = math.ceil
_md_exp_cached = math.exp
_md_pow_cached = math.pow
_md_sqrt_cached = math.sqrt
_md_log_cached = math.log
_md_log10_cached = math.log10
_md_sin_cached = math.sin
_md_cos_cached = math.cos
_md_tan_cached = math.tan
_md_asin_cached = math.asin
_md_acos_cached = math.acos
_md_atan_cached = math.atan
_md_atan2_cached = math.atan2
_md_fabs_cached = math.fabs
_md_isfinite_cached = math.isfinite


def _is_int_like(value: float) -> bool:
    """True when *value* is finite and integer-valued within rounding tolerance."""
    if not _md_isfinite_cached(value):
        return False
    return abs(value - round(value)) <= _INT_EPS


class _RandState:
    rand_seed: int = 1


_RAND_STATE = _RandState()


def md_rand(state: Any) -> float:
    """Deterministic MilkDrop LCG step. Uses ``state.rand_seed`` explicitly."""
    try:
        seed = int(getattr(state, "rand_seed", 1))
    except Exception:  # noqa: BLE001
        seed = 1
    if not math.isfinite(seed) or seed <= 0:
        seed = 1
    try:
        state.rand_seed = (seed * 16807) % 2147483647
        if state.rand_seed <= 0:
            state.rand_seed = 1
        return state.rand_seed / 2147483647.0
    except Exception:  # noqa: BLE001
        return 0.0


def _finite(value: Any, max_abs: float = MAX_SAFE_FLOAT) -> float:
    try:
        result = float(value)
    except Exception:  # noqa: BLE001
        return 0.0
    if math.isnan(result):
        return 0.0
    if result > max_abs:
        return float(max_abs)
    if result < -max_abs:
        return float(-max_abs)
    return result


def _finite32(value: Any) -> float:
    return _finite(value, _MAX_FLOAT32)


def _md_finite_float(value: Any, default: float = 0.0) -> float:
    """Convert to float; return *default* for non-finite/unconvertible values."""
    try:
        result = float(value)
    except Exception:  # noqa: BLE001
        return float(default)
    if not _md_isfinite_cached(result):
        return float(default)
    return result


def _md_float(value: Any, default: float = 0.0) -> float:
    return _md_finite_float(value, default)


def _md_abs(x: float) -> float:
    try:
        x = _md_float(x)
        if not _md_isfinite_cached(x):
            return 0.0
        result = abs(x)
        if not _md_isfinite_cached(result):
            return 0.0
        return result
    except Exception:  # noqa: BLE001
        return 0.0


def _md_clamp(value: float, min_val: float, max_val: float) -> float:
    try:
        v = _md_float(value)
        mn = _md_float(min_val)
        mx = _md_float(max_val)
    except Exception:  # noqa: BLE001
        return 0.0
    if v < mn:
        return mn
    if v > mx:
        return mx
    return v


def _md_floor(x: float) -> float:
    try:
        x = _md_float(x)
        if not _md_isfinite_cached(x):
            return 0.0
        return float(_md_floor_cached(x))
    except Exception:  # noqa: BLE001
        return 0.0


def _md_fract(x: float) -> float:
    try:
        x = _md_float(x)
        if not _md_isfinite_cached(x):
            return 0.0
        return x - _md_floor_cached(x)
    except Exception:  # noqa: BLE001
        return 0.0


def _md_int(x: float) -> float:
    try:
        x = _md_float(x)
        if not _md_isfinite_cached(x):
            return 0.0
        return float(int(x))
    except Exception:  # noqa: BLE001
        return 0.0


def _md_log(x: float) -> float:
    try:
        x = _md_float(x)
        if x <= 0.0:
            return 0.0
        result = _md_log_cached(x)
        if not _md_isfinite_cached(result):
            return 0.0
        return result
    except Exception:  # noqa: BLE001
        return 0.0


def _md_log10(x: float) -> float:
    try:
        x = _md_float(x)
        if x <= 0.0:
            return 0.0
        result = _md_log10_cached(x)
        if not _md_isfinite_cached(result):
            return 0.0
        return result
    except Exception:  # noqa: BLE001
        return 0.0


def _md_max(a: float, b: float) -> float:
    a = _md_float(a)
    b = _md_float(b)
    if not _md_isfinite_cached(a):
        return b
    if not _md_isfinite_cached(b):
        return a
    return max(a, b)


def _md_min(a: float, b: float) -> float:
    a = _md_float(a)
    b = _md_float(b)
    if not _md_isfinite_cached(a):
        return b
    if not _md_isfinite_cached(b):
        return a
    return min(a, b)


def _md_mix(x: float, y: float, a: float) -> float:
    """Blend x->y by factor a clamped to [0, 1] (safety/determinism policy)."""
    try:
        x = _md_float(x)
        y = _md_float(y)
        a = _md_float(a)
        if not _md_isfinite_cached(a):
            a = 0.0
        if a < 0.0:
            a = 0.0
        elif a > 1.0:
            a = 1.0
        return x * (1.0 - a) + y * a
    except Exception:  # noqa: BLE001
        return 0.0


def _md_mod(x: float, y: float) -> float:
    """Integer remainder modulo (truncated division), matching MilkDrop % operator semantics."""
    try:
        x = _md_float(x)
        y = _md_float(y)
        if y == 0.0 or not _md_isfinite_cached(y):
            return 0.0
        if not _md_isfinite_cached(x):
            return 0.0
        # Integer remainder: truncate toward zero, then compute remainder
        # This matches C/Python int truncation behavior, not Python's floor %
        xi = int(x)
        yi = int(y)
        if yi == 0:
            return 0.0
        # Truncated division remainder: sign follows dividend (C behavior)
        result = float(xi - yi * int(xi / yi))
        if not _md_isfinite_cached(result):
            return 0.0
        return result
    except Exception:  # noqa: BLE001
        return 0.0


def _md_pow(base: float, exp: float) -> float:
    try:
        base = float(base)
        exp = float(exp)
    except Exception:  # noqa: BLE001
        return 0.0
    if not (_md_isfinite_cached(base) and _md_isfinite_cached(exp)):
        return 0.0

    if exp == 0.0:
        return 1.0
    if exp == 1.0:
        return base
    if base == 1.0:
        return 1.0
    if base == 0.0:
        return 0.0 if exp > 0.0 else 1.0

    # Negative base with non-integer exponent would produce complex results.
    # Use a tolerant integer check so large integer-valued floats remain valid.
    if base < 0.0 and not _is_int_like(exp):
        return 0.0

    try:
        result = _md_pow_cached(base, exp)
        return result if _md_isfinite_cached(result) else 0.0
    except OverflowError:
        return 0.0
    except ValueError:
        return 0.0
    except Exception:  # noqa: BLE001
        return 0.0


def _md_sqr(x: float) -> float:
    try:
        x = _md_float(x)
        result = x * x
        if not _md_isfinite_cached(result):
            return 0.0
        return result
    except Exception:  # noqa: BLE001
        return 0.0


def _md_sqrt(x: float) -> float:
    try:
        x = _md_float(x)
        if x < 0.0:
            return 0.0
        result = _md_sqrt_cached(x)
        if not _md_isfinite_cached(result):
            return 0.0
        return result
    except Exception:  # noqa: BLE001
        return 0.0


def _md_tan(x: float) -> float:
    try:
        x = _md_float(x)
        result = _md_tan_cached(x)
        if not _md_isfinite_cached(result):
            return 0.0
        return result
    except Exception:  # noqa: BLE001
        return 0.0


def _md_invsqrt(x: float) -> float:
    try:
        x = _md_float(x)
        if x <= 0.0:
            return 0.0
        result = 1.0 / _md_sqrt_cached(x)
        if not _md_isfinite_cached(result):
            return 0.0
        return result
    except Exception:  # noqa: BLE001
        return 0.0


def _md_sign(x: float) -> float:
    try:
        x = _md_float(x)
        if x > 0.0:
            return 1.0
        if x < 0.0:
            return -1.0
        return 0.0
    except Exception:  # noqa: BLE001
        return 0.0


def _md_sin(x: float) -> float:
    try:
        x = _md_float(x)
        if not _md_isfinite_cached(x):
            return 0.0
        result = _md_sin_cached(x)
    except Exception:  # noqa: BLE001
        return 0.0
    if not _md_isfinite_cached(result):
        return 0.0
    return result


def _md_cos(x: float) -> float:
    try:
        x = _md_float(x)
        if not _md_isfinite_cached(x):
            return 0.0
        result = _md_cos_cached(x)
    except Exception:  # noqa: BLE001
        return 0.0
    if not _md_isfinite_cached(result):
        return 0.0
    return result


def _md_asin(x: float) -> float:
    try:
        x = _md_float(x)
        if not _md_isfinite_cached(x):
            return 0.0
        if x < -1.0:
            x = -1.0
        elif x > 1.0:
            x = 1.0
        result = _md_asin_cached(x)
    except Exception:  # noqa: BLE001
        return 0.0
    if not _md_isfinite_cached(result):
        return 0.0
    return result


def _md_acos(x: float) -> float:
    try:
        x = _md_float(x)
        if not _md_isfinite_cached(x):
            return 0.0
        if x < -1.0:
            x = -1.0
        elif x > 1.0:
            x = 1.0
        result = _md_acos_cached(x)
    except Exception:  # noqa: BLE001
        return 0.0
    if not _md_isfinite_cached(result):
        return 0.0
    return result


def _md_atan(x: float) -> float:
    try:
        x = _md_float(x)
        result = _md_atan_cached(x)
        if not _md_isfinite_cached(result):
            return 0.0
        return result
    except Exception:  # noqa: BLE001
        return 0.0


def _md_atan2(y: float, x: float) -> float:
    try:
        y = _md_float(y)
        x = _md_float(x)
        result = _md_atan2_cached(y, x)
        if not _md_isfinite_cached(result):
            return 0.0
        return result
    except Exception:  # noqa: BLE001
        return 0.0


def _md_exp(x: float) -> float:
    try:
        x = _md_float(x)
        if x > 709.0:
            return MAX_SAFE_FLOAT
        if x < -745.0:
            return 0.0
        result = _md_exp_cached(x)
        return result if _md_isfinite_cached(result) else (MAX_SAFE_FLOAT if x > 0.0 else 0.0)
    except OverflowError:
        x = _md_float(x)
        return MAX_SAFE_FLOAT if x > 0.0 else 0.0
    except Exception:  # noqa: BLE001
        return 0.0


def _md_exp2(x: float) -> float:
    try:
        x = _md_float(x)
        if x > 1023.0:
            return MAX_SAFE_FLOAT
        if x < -1074.0:
            return 0.0
        result = _md_pow_cached(2.0, x)
        return result if _md_isfinite_cached(result) else (MAX_SAFE_FLOAT if x > 0.0 else 0.0)
    except OverflowError:
        x = _md_float(x)
        return MAX_SAFE_FLOAT if x > 0.0 else 0.0
    except Exception:  # noqa: BLE001
        return 0.0


def _md_ceil(x: float) -> float:
    try:
        x = _md_float(x)
        if not _md_isfinite_cached(x):
            return 0.0
        return float(_md_ceil_cached(x))
    except Exception:  # noqa: BLE001
        return 0.0


def _md_bool(value: Any) -> float:
    """MilkDrop logical truth test: non-zero is true, zero is false (NaN false, Inf true)."""
    try:
        v = float(value)
    except Exception:  # noqa: BLE001
        return 0.0
    if math.isnan(v):
        return 0.0
    if math.isinf(v):
        return 1.0
    return 1.0 if v != 0.0 else 0.0


def _md_factorial(n: float) -> float:
    try:
        n = _md_float(n)
        if not _md_isfinite_cached(n):
            return 1.0
        i = int(n)
        if i < 0:
            return 1.0
        i = min(i, _SAFE_FACT_MAX)
        result = float(math.factorial(i))
        if not _md_isfinite_cached(result):
            return MAX_SAFE_FLOAT
        return result
    except Exception:  # noqa: BLE001
        return 1.0


def _md_nchoosek(n: float, k: float) -> float:
    """Binomial coefficient with VM-safe finite fallback."""
    try:
        n = _md_float(n)
        k = _md_float(k)
        if not (_md_isfinite_cached(n) and _md_isfinite_cached(k)):
            return 0.0
        ni = int(n)
        ki = int(k)
        if ni < 0 or ki < 0 or ki > ni:
            return 0.0
        result = float(math.comb(ni, ki))
        return result if _md_isfinite_cached(result) else MAX_SAFE_FLOAT
    except Exception:  # noqa: BLE001
        return 0.0


def _md_wrap(x: float, a: float, b: float) -> float:
    """Wrap *x* into [a, b) (safe finite fallback)."""
    try:
        # Check for non-finite inputs before conversion
        try:
            x_raw = float(x)
            a_raw = float(a)
            b_raw = float(b)
        except (TypeError, ValueError):
            return 0.0
        if not (_md_isfinite_cached(x_raw) and _md_isfinite_cached(a_raw) and _md_isfinite_cached(b_raw)):
            return 0.0
        x = _md_float(x_raw)
        a = _md_float(a_raw)
        b = _md_float(b_raw)
        r = b - a
        if not _md_isfinite_cached(r) or abs(r) < EPSILON_DIV:
            return a
        diff = x - a
        if not _md_isfinite_cached(diff):
            return a
        mod_result = diff % r
        if not _md_isfinite_cached(mod_result):
            return a
        result = a + mod_result
        if not _md_isfinite_cached(result):
            return a
        return _finite(result)
    except Exception:  # noqa: BLE001
        return 0.0


def _md_cwrap(x: float, a: float, b: float) -> float:
    """Centered/wave wrap into [a, b) range semantics used by the VM."""
    try:
        # Check for non-finite inputs before conversion
        try:
            x_raw = float(x)
            a_raw = float(a)
            b_raw = float(b)
        except (TypeError, ValueError):
            return 0.0
        if not (_md_isfinite_cached(x_raw) and _md_isfinite_cached(a_raw) and _md_isfinite_cached(b_raw)):
            return 0.0
        x = _md_float(x_raw)
        a = _md_float(a_raw)
        b = _md_float(b_raw)
        r = b - a
        if not _md_isfinite_cached(r) or abs(r) < EPSILON_DIV:
            return a
        two_r = 2.0 * r
        if not _md_isfinite_cached(two_r) or abs(two_r) < EPSILON_DIV:
            return a
        diff = x - a
        if not _md_isfinite_cached(diff):
            return a
        mod_result = diff % two_r
        if not _md_isfinite_cached(mod_result):
            return a
        centered = abs(mod_result - r)
        if not _md_isfinite_cached(centered):
            return a
        result = a + centered
        if not _md_isfinite_cached(result):
            return a
        return _finite(result)
    except Exception:  # noqa: BLE001
        return 0.0


def _md_bor(a: float, b: float) -> float:
    try:
        return float(int(_md_int(a)) | int(_md_int(b)))
    except Exception:  # noqa: BLE001
        return 0.0


def _md_bnot(a: float) -> float:
    try:
        return float(~int(_md_int(a)))
    except Exception:  # noqa: BLE001
        return 0.0


def _md_band(a: float, b: float) -> float:
    """Integer bitwise AND, matching MilkDrop & operator semantics."""
    try:
        return float(int(_md_int(a)) & int(_md_int(b)))
    except Exception:  # noqa: BLE001
        return 0.0


def _validate_matrix_safe(value: Any, name: str = "matrix") -> list[list[float]] | None:
    """Safe matrix validation that returns None instead of raising exceptions."""
    try:
        rows = []
        for row in value:
            if not isinstance(row, (list, tuple)):
                return None
            rows.append([_md_float(v) for v in row])
        if not rows:
            return None
        width = len(rows[0])
        for row in rows:
            if len(row) != width:
                return None
        # Validate all elements are finite
        for row in rows:
            for v in row:
                if not _md_isfinite_cached(v):
                    return None
        return rows
    except Exception:  # noqa: BLE001
        return None


def _validate_matrix(value: Any, name: str = "matrix") -> list[list[float]] | None:
    """Safe matrix validation that returns None instead of raising exceptions."""
    try:
        rows = []
        for row in value:
            if not isinstance(row, (list, tuple)):
                return None
            rows.append([_md_float(v) for v in row])
        if not rows:
            return None
        width = len(rows[0])
        for row in rows:
            if len(row) != width:
                return None
        # Validate all elements are finite
        for row in rows:
            for v in row:
                if not _md_isfinite_cached(v):
                    return None
        return rows
    except Exception:  # noqa: BLE001
        return None


def _md_mul_cpu(a: Any, b: Any) -> Any:
    try:
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            result = float(a) * float(b)
            if not _md_isfinite_cached(result):
                return 0.0
            return result

        a_seq = isinstance(a, (list, tuple))
        b_seq = isinstance(b, (list, tuple))
        if not a_seq or not b_seq:
            result = float(_md_float(a) * _md_float(b))
            if not _md_isfinite_cached(result):
                return 0.0
            return result

        if len(a) == 0 or len(b) == 0:
            return 0.0

        a_is_mat = isinstance(a[0], (list, tuple))
        b_is_mat = isinstance(b[0], (list, tuple))

        if not a_is_mat and not b_is_mat:
            if len(a) != len(b):
                return 0.0
            result = tuple(_md_float(x) * _md_float(y) for x, y in zip(a, b))
            # Validate all elements are finite
            for v in result:
                if not _md_isfinite_cached(v):
                    return 0.0
            return result

        if not a_is_mat and b_is_mat:
            b_rows = _validate_matrix_safe(b, "mul rhs")
            if b_rows is None or len(a) != len(b_rows):
                return 0.0
            cols = len(b_rows[0])
            result = tuple(
                _finite(sum(_md_float(a[i]) * b_rows[i][j] for i in range(len(a))))
                for j in range(cols)
            )
            return result

        if a_is_mat and not b_is_mat:
            a_rows = _validate_matrix_safe(a, "mul lhs")
            if a_rows is None:
                return 0.0
            cols = len(a_rows[0])
            if len(b) != cols:
                return 0.0
            result = tuple(
                _finite(sum(a_rows[i][j] * _md_float(b[j]) for j in range(cols)))
                for i in range(len(a_rows))
            )
            return result

        if a_is_mat and b_is_mat:
            a_rows = _validate_matrix_safe(a, "mul lhs")
            b_rows = _validate_matrix_safe(b, "mul rhs")
            if a_rows is None or b_rows is None:
                return 0.0
            if len(a_rows[0]) != len(b_rows):
                return 0.0
            cols = len(b_rows[0])
            out = []
            for i in range(len(a_rows)):
                row = tuple(
                    _finite(sum(a_rows[i][k] * b_rows[k][j] for k in range(len(b_rows))))
                    for j in range(cols)
                )
                out.append(row)
            return tuple(out)

        result = float(_md_float(a) * _md_float(b))
        if not _md_isfinite_cached(result):
            return 0.0
        return result
    except Exception:  # noqa: BLE001
        return 0.0

def _md_rand_cpu(max_value: float = 1.0, state: Any | None = None) -> float:
    """Random helper with explicit state; falls back to module singleton."""
    try:
        max_value = _md_float(max_value, 1.0)
        rng_state = _RAND_STATE if state is None else state
        return md_rand(rng_state) * max_value
    except Exception:  # noqa: BLE001
        return 0.0


def _md_sigmoid(x: float, constraint: float = 1.0) -> float:
    try:
        x_raw = float(x)
    except Exception:  # noqa: BLE001
        return 0.5
    if math.isnan(x_raw):
        return 0.5
    if math.isinf(x_raw):
        return 0.0 if x_raw < 0.0 else 1.0
    try:
        x = _md_float(x_raw)
        constraint = _md_float(constraint, 1.0)
        z = x * constraint
        if z > 709.0:
            return 1.0
        if z < -745.0:
            return 0.0
        result = 1.0 / (1.0 + _md_exp_cached(-z))
        if not _md_isfinite_cached(result):
            return 0.5
        if result < 0.0:
            return 0.0
        if result > 1.0:
            return 1.0
        return result
    except Exception:  # noqa: BLE001
        return 0.5


def _md_smoothstep(edge0: float, edge1: float, x: float) -> float:
    try:
        # Check for non-finite inputs before conversion
        try:
            edge0_raw = float(edge0)
            edge1_raw = float(edge1)
            x_raw = float(x)
        except (TypeError, ValueError):
            return 0.0
        if not (_md_isfinite_cached(edge0_raw) and _md_isfinite_cached(edge1_raw) and _md_isfinite_cached(x_raw)):
            return 0.0
        edge0 = _md_float(edge0_raw)
        edge1 = _md_float(edge1_raw)
        x = _md_float(x_raw)
        denom = edge1 - edge0
        if not _md_isfinite_cached(denom) or abs(denom) < EPSILON_DIV:
            return 0.0
        t = (x - edge0) / denom
        if not _md_isfinite_cached(t):
            return 0.0
        t = max(0.0, min(1.0, t))
        # Compute t*t*(3-2*t) safely
        t2 = t * t
        if not _md_isfinite_cached(t2):
            return 0.0
        result = t2 * (3.0 - 2.0 * t)
        if not _md_isfinite_cached(result):
            return 0.0
        return result
    except Exception:  # noqa: BLE001
        return 0.0


def _md_step(edge: float, x: float) -> float:
    try:
        edge = _md_float(edge)
        x = _md_float(x)
        if not (_md_isfinite_cached(edge) and _md_isfinite_cached(x)):
            return 0.0
        return 1.0 if x >= edge else 0.0
    except Exception:  # noqa: BLE001
        return 0.0


def _md_length(x: float) -> float:
    try:
        x = _md_float(x)
        if not _md_isfinite_cached(x):
            return 0.0
        return abs(x)
    except Exception:  # noqa: BLE001
        return 0.0


def _md_distance(a: float, b: float) -> float:
    try:
        # Check for non-finite before conversion
        try:
            a_raw = float(a)
            b_raw = float(b)
        except (TypeError, ValueError):
            return 0.0
        if not (_md_isfinite_cached(a_raw) and _md_isfinite_cached(b_raw)):
            return MAX_SAFE_FLOAT
        a = _md_float(a_raw)
        b = _md_float(b_raw)
        # Use safe subtraction to avoid overflow: distance = |a - b|
        diff = a - b
        if not _md_isfinite_cached(diff):
            return MAX_SAFE_FLOAT
        return abs(diff)
    except Exception:  # noqa: BLE001
        return 0.0


def _md_normalize(x: float) -> float:
    try:
        x = _md_float(x)
        if not _md_isfinite_cached(x):
            return 0.0
        if x == 0.0:
            return 0.0
        return 1.0 if x > 0.0 else -1.0
    except Exception:  # noqa: BLE001
        return 0.0


__all__ = [
    "EPSILON_BOOL",
    "EPSILON_COMPARE",
    "EPSILON_DIV",
    "EPSILON_SMALL",
    "MAX_SAFE_FLOAT",
    "OP_ABS",
    "OP_ACOS",
    "OP_ADD",
    "OP_AND",
    "OP_ASIN",
    "OP_ATAN",
    "OP_ATAN2",
    "OP_BAND",
    "OP_BNOT",
    "OP_BOR",
    "OP_CALL_REGISTRY",
    "OP_CEIL",
    "OP_CLAMP",
    "OP_CMP_EQ",
    "OP_CMP_GE",
    "OP_CMP_GT",
    "OP_CMP_LE",
    "OP_CMP_LT",
    "OP_CMP_NE",
    "OP_COS",
    "OP_CWRAP",
    "OP_DIV",
    "OP_DUP",
    "OP_EXP",
    "OP_FACT",
    "OP_FLOOR",
    "OP_FRACT",
    "OP_IF",
    "OP_INT_FUNC",
    "OP_INVSQRT",
    "OP_JUMP",
    "OP_JUMP_IF_FALSE",
    "OP_JUMP_IF_TRUE",
    "OP_LOAD_CONST",
    "OP_LOAD_VAR",
    "OP_LOAD_VAR_DICT",
    "OP_LOAD_VAR_INDEX",
    "OP_LOAD_VAR_INDEX_Q",
    "OP_LOAD_VAR_INDEX_T",
    "OP_LOAD_VAR_REG",
    "OP_LOG",
    "OP_LOG10",
    "OP_MAX",
    "OP_MEGABUF_GET",
    "OP_MEGABUF_SET",
    "OP_MIN",
    "OP_MIX",
    "OP_MOD",
    "OP_MUL",
    "OP_NCHOOSEK",
    "OP_NEG",
    "OP_NOT",
    "OP_OR",
    "OP_POP",
    "OP_POW",
    "OP_POW_FUNC",
    "OP_Q_GET",
    "OP_Q_SET",
    "OP_RAND",
    "OP_SIGMOID",
    "OP_SIGN",
    "OP_SIN",
    "OP_SQR",
    "OP_SQRT",
    "OP_STORE_VAR",
    "OP_STORE_VAR_DICT",
    "OP_STORE_VAR_INDEX",
    "OP_STORE_VAR_REG",
    "OP_SUB",
    "OP_TAN",
    "OP_T_GET",
    "OP_T_SET",
    "OP_WRAP",
    "_MAX_FLOAT32",
    "_SAFE_FACT_MAX",
    "_finite",
    "_finite32",
    "_md_abs",
    "_md_acos",
    "_md_asin",
    "_md_atan",
    "_md_atan2",
    "_md_band",
    "_md_bnot",
    "_md_bool",
    "_md_bor",
    "_md_ceil",
    "_md_clamp",
    "_md_cos",
    "_md_cwrap",
    "_md_distance",
    "_md_exp",
    "_md_exp2",
    "_md_factorial",
    "_md_float",
    "_md_floor",
    "_md_fract",
    "_md_int",
    "_md_invsqrt",
    "_md_length",
    "_md_log",
    "_md_log10",
    "_md_max",
    "_md_min",
    "_md_mix",
    "_md_mod",
    "_md_mul_cpu",
    "_md_nchoosek",
    "_md_normalize",
    "_md_pow",
    "_md_rand_cpu",
    "_md_sigmoid",
    "_md_sign",
    "_md_sin",
    "_md_smoothstep",
    "_md_sqr",
    "_md_sqrt",
    "_md_step",
    "_md_tan",
    "_md_wrap",
    "_validate_matrix",
    "_validate_matrix_safe",
    "md_rand",
]
