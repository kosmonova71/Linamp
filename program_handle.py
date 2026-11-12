"""ProgramHandle: transactional OpenGL program handle with generation safety.

Each handle captures the program id, a context *generation* (frozen at
creation and recording which GL context the handle belongs to), and a
separate *location_generation* counter (bumped whenever the underlying
program is recompiled so cached uniform locations can be invalidated).

When the program is recreated (shader recompile, preset switch) the renderer
calls :meth:`invalidate`, which bumps ``location_generation`` and clears the
cached locations.  Any upload attempt carrying an older ``location_generation``
is rejected, preventing stale uniform locations from one program object
being silently applied to another.

The context ``generation`` is never mutated after creation.  It serves as
external bookkeeping so tests and diagnostics can confirm a handle belongs to
the current GL context.  Context-level protection — preventing a handle from a
destroyed/recreated context being reused — is enforced by the renderer at the
handle-table level: the cache key includes the context generation and the
entire handle table is cleared when the context is recreated
(``_bump_program_context_generation``), so a stale handle can never be
returned to a caller in the first place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from OpenGL import GL

from gl_state import _gl_finite


class _GLState(Protocol):
    """Minimal protocol for the state manager used by upload methods."""

    program: int

    def use_program(self, program: int, force: bool = False) -> None: ...


def _upload_uniform_value(loc: int, value: float | bool | np.ndarray | list | tuple) -> None:
    """Dispatch the correct ``glUniform*`` call based on *value* type.

    Scalar ints/bools go to ``glUniform1i``; scalar floats and numpy
    scalars go to ``glUniform1f``.  Sequences (``list``/``tuple``) and
    ``np.ndarray`` of length 2/3/4 are uploaded as vector uniforms.

    **Important:** For plain ``float[]`` array uniforms (e.g. ``float u_q[33]``,
    the q/t/reg batches) you **must** use :meth:`ProgramHandle.upload_uniform_1fv`.
    Multi-element values that are not true vec2/vec3/vec4 raise ``ValueError``.
    """
    if isinstance(value, bool):
        GL.glUniform1i(loc, 1 if value else 0)
    elif isinstance(value, (int, np.integer)):
        GL.glUniform1i(loc, int(value))
    elif isinstance(value, (float, np.floating)):
        GL.glUniform1f(loc, _gl_finite(value))
    elif isinstance(value, np.ndarray):
        n = int(value.size)
        if n == 2:
            GL.glUniform2f(
                loc, _gl_finite(float(value.flat[0])), _gl_finite(float(value.flat[1]))
            )
        elif n == 3:
            GL.glUniform3f(
                loc,
                _gl_finite(float(value.flat[0])),
                _gl_finite(float(value.flat[1])),
                _gl_finite(float(value.flat[2])),
            )
        elif n == 4:
            GL.glUniform4f(
                loc,
                _gl_finite(float(value.flat[0])),
                _gl_finite(float(value.flat[1])),
                _gl_finite(float(value.flat[2])),
                _gl_finite(float(value.flat[3])),
            )
        else:
            raise ValueError(
                f"upload_uniform: length-{n} array is not a scalar/vec2/3/4. "
                "Use upload_uniform_1fv for float[] uniforms."
            )
    elif isinstance(value, (list, tuple)):
        # Convert list/tuple to numpy for consistent handling
        arr = np.array(value, dtype=np.float32)
        n = int(arr.size)
        if n == 1:
            GL.glUniform1f(loc, _gl_finite(float(arr[0])))
        elif n == 2:
            GL.glUniform2f(loc, _gl_finite(float(arr[0])), _gl_finite(float(arr[1])))
        elif n == 3:
            GL.glUniform3f(
                loc, _gl_finite(float(arr[0])), _gl_finite(float(arr[1])), _gl_finite(float(arr[2]))
            )
        elif n == 4:
            GL.glUniform4f(
                loc,
                _gl_finite(float(arr[0])),
                _gl_finite(float(arr[1])),
                _gl_finite(float(arr[2])),
                _gl_finite(float(arr[3])),
            )
        else:
            raise ValueError(
                f"upload_uniform: length-{n} array is not a scalar/vec2/3/4. "
                "Use upload_uniform_1fv for float[] uniforms."
            )
    else:
        raise TypeError(
            f"upload_uniform: unsupported type {type(value).__name__}. "
            "Use scalars, lists, tuples, or np.ndarray."
        )


@dataclass(slots=True)
class ProgramHandle:
    """Transactional program handle with generation-based uniform safety.

    Instead of passing raw ``program_id`` + ``uniform_location`` pairs
    through the render path (which allows a stale location from program A
    to be used after program B becomes active), every uniform upload goes
    through a handle that captures the program id AND a generation counter.

    When the underlying program is recreated (shader recompile, preset
    switch), the renderer calls :meth:`invalidate`, which bumps
    ``location_generation``.  Any upload attempt using an older
    ``location_generation`` is rejected, eliminating the remaining class of::

        program A
        location from A
        program B becomes active
        old location accidentally reused

    bugs without requiring callers to track program identity themselves.
    """

    program_id: int = 0
    # Frozen-at-creation GL-context generation the handle belongs to.
    # Never mutated after the handle is created; used for external
    # bookkeeping (tests, diagnostics) to verify context ownership.
    # Staleness *within* a context is guarded by location_generation,
    # not by generation — context-level protection is enforced by the
    # renderer's handle-table key system.
    generation: int = 0
    location_generation: int = 0
    uniform_locations: dict[str, int] = field(default_factory=dict)
    retired: bool = False
    # Scratch buffer reused across upload_uniform_1fv calls so the
    # per-frame q/t/reg batch uploads avoid a numpy allocation on every
    # render frame.  Pre-sized to the largest expected batch (100 global
    # registers) and grown on demand if needed.
    sanitize_buf: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.sanitize_buf is None:
            self.sanitize_buf = np.empty(100, dtype=np.float32)

    def valid(self) -> bool:
        """A handle is valid when it tracks a real program and is not retired."""
        return self.program_id > 0 and not self.retired

    def __bool__(self) -> bool:
        """Mirror valid() so ``if handle:`` is safe and consistent."""
        return self.valid()

    def matches_generation(self, expected: int) -> bool:
        return self.valid() and self.location_generation == expected

    def snapshot(self) -> tuple[int, int]:
        """Return ``(program_id, location_generation)`` for later checks.

        Useful when a caller wants to capture generation before a batch of
        work and later pass ``expected_generation`` to the upload methods.
        """
        return self.program_id, self.location_generation

    def invalidate(self) -> None:
        """Bump the location-generation so all cached locations become stale.

        ``generation`` is frozen at creation time and records the GL context
        generation the handle belongs to.  ``location_generation`` is a
        separate monotonic counter bumped whenever the underlying program is
        recompiled, so staleness of cached uniform locations is tracked
        independently from context identity.
        """
        self.location_generation += 1
        self.uniform_locations.clear()

    def retire(self) -> None:
        """Mark this handle as retired so future uploads are rejected.

        Called by the owning renderer whenever the underlying GL program is
        queued for deletion or otherwise permanently invalidated.
        """
        self.retired = True
        self.uniform_locations.clear()

    def delete(self) -> None:
        """Delete the underlying GL program and retire the handle."""
        if self.program_id > 0 and not self.retired:
            try:
                GL.glDeleteProgram(self.program_id)
            except Exception as exc:  # noqa: BLE001 - best-effort GL program deletion
                import logging

                logging.getLogger(__name__).debug("glDeleteProgram failed: %s", exc)
        # Always clear the id so valid() stays false even if retire() is
        # later changed or the exception path left residual state.
        self.program_id = 0
        self.retire()

    def get_location(self, name: str) -> int:
        if self.retired:
            return -1
        return self.uniform_locations.get(name, -1)

    def set_location(self, name: str, loc: int) -> None:
        if self.retired:
            return
        self.uniform_locations[name] = loc

    def upload_uniform(
        self,
        name: str,
        value: Any,
        gl_state: _GLState,
        expected_generation: int | None = None,
    ) -> bool:
        """Upload a uniform using this handle's cached location.

        Returns ``True`` on success, ``False`` if the location is unknown or
        the handle has been invalidated (location-generation mismatch).

        .. note::

            **Program-binding side-effect.**  When the currently-bound program
            differs from ``self.program_id``, this method calls
            ``gl_state.use_program(self.program_id, force=True)`` to bind the
            correct program before uploading.  This guarantees the uniform
            lands in the right program even if a stale binding is active.
            The :class:`GLStateManager` caches the binding, so after the first
            upload within a batch the early-out
            (``gl_state.program == self.program_id``) skips the redundant
            ``glUseProgram`` on every subsequent call.  Callers that batch
            many uniforms on the same handle therefore pay the program bind
            once and then run through the fast path.

        .. warning::

            For plain ``float[]`` array uniforms use :meth:`upload_uniform_1fv`
            instead.  ``upload_uniform`` only accepts scalars and true
            vec2/vec3/vec4 values; longer sequences raise ``ValueError``.
        """
        if not self.valid():
            return False
        if expected_generation is not None and expected_generation != self.location_generation:
            return False
        loc = self.get_location(name)
        if loc < 0:
            return False
        if gl_state.program != self.program_id:
            gl_state.use_program(self.program_id, force=True)
        _upload_uniform_value(loc, value)
        return True

    def upload_uniform_1fv(
        self,
        name: str,
        values: float | bool | np.ndarray | list | tuple,
        gl_state: _GLState,
        expected_generation: int | None = None,
    ) -> bool:
        """Upload a float array via ``glUniform1fv``.

        ``upload_uniform`` only accepts scalars and true vec2/vec3/vec4.
        This method always uses ``glUniform1fv`` with per-element NaN/Inf
        sanitisation, which is the correct call for :decl:`float[]` arrays
        (e.g. ``float u_q[33]``, the q/t/reg batches).

        Works for ``list``, ``tuple``, ``np.ndarray`` and any iterable of
        numeric values (including generators).
        """
        if not self.valid():
            return False
        if expected_generation is not None and expected_generation != self.location_generation:
            return False
        loc = self.get_location(name)
        if loc < 0:
            return False
        if gl_state.program != self.program_id:
            gl_state.use_program(self.program_id, force=True)

        # Always materialise once so generators / one-shot iterables work.
        if isinstance(values, np.ndarray):
            src = values.astype(np.float32, copy=False).ravel()
        else:
            try:
                src = np.asarray(values, dtype=np.float32).ravel()
            except (TypeError, ValueError):
                if isinstance(values, (float, int, bool, np.floating, np.integer)):
                    src = np.array([_gl_finite(float(values))], dtype=np.float32)
                else:
                    src = np.array(
                        [_gl_finite(float(v)) for v in values], dtype=np.float32
                    )

        count = int(src.size)
        if count == 0:
            return True

        if count > 10_000:
            import logging

            logging.getLogger(__name__).warning(
                "upload_uniform_1fv: unusually large count=%d for %r", count, name
            )

        buf = self.sanitize_buf
        if buf is None or buf.size < count:
            buf = np.empty(max(count, 100), dtype=np.float32)
            self.sanitize_buf = buf

        np.copyto(buf[:count], src, casting="unsafe")
        np.copyto(buf[:count], 0.0, where=~np.isfinite(buf[:count]))
        GL.glUniform1fv(loc, count, buf[:count])
        return True
