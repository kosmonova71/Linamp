"""GL state management: resource ownership, state caching, shaders.

Extracted from ``milkdrop_renderer`` to centralize OpenGL state ownership,
FBO validation, shader compilation, and the state-guard context manager.

Dependencies are strictly downward: gl_state -> resources, gl_state ->
equations, gl_state -> debug_budget.  No module below may import gl_state.

Threading
---------
``GLStateManager`` cache mutations and all GL calls that go through it must
happen on the single render thread that owns the active context.  Deletion
queues are locked; the state cache is not.
"""

from __future__ import annotations

import contextlib
import logging
import math
import re
import threading
from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np
from OpenGL import GL
from OpenGL import error as _GLPyError

logger = logging.getLogger(__name__)

# When True, validate each queued GL object with glIs*() before deletion.
# Disabled by default to avoid per-delete driver synchronization overhead.
DEBUG_GL_DELETE_VALIDATION = False


def _gl_finite(value: Any) -> float:
    """Coerce a non-finite or non-numeric value to 0.0 before it reaches a GL uniform.

    The preset VM deliberately does not clamp arithmetic, so an equation such
    as ``exp(2000) * exp(2000)`` stores ``inf``.  Uploading that makes the
    fragment shader evaluate ``inf - inf`` -> NaN, which several drivers
    handle on a drastically slower path (or drop the context), and a whole
    ``glUniform1fv`` register batch is poisoned by a single bad value.

    ``ValueError`` (e.g. ``float("not a number")``) and ``TypeError``
    (e.g. ``float(None)``) are also caught and treated as zero — audio
    levels sourced from an external audio pipeline may arrive as strings
    or ``None`` and must never crash the render thread.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return v if math.isfinite(v) else 0.0


def _int_gl_id(value: Any) -> int:
    """Coerce a PyOpenGL resource-ID return value to a plain ``int``.

    ``glGenTextures``, ``glGenFramebuffers`` and similar functions may
    return a scalar ``int`` or an array-like object (``list`` /
    ``tuple`` / ``ndarray``) depending on the PyOpenGL version and
    profile.  This helper normalises every case to ``int`` so callers —
    including ``except`` handlers where the ``isinstance`` narrowing
    may not have run — always receive a definite ``int``.
    """
    if isinstance(value, (list, tuple, np.ndarray)):
        return int(value[0])
    return int(value)


def _delete_gl(delete_fn: Callable[..., object], obj: int) -> None:
    """Delete a single GL object, accepting PyOpenGL's variant call signatures.

    PyOpenGL bindings differ by GL function and version:
    - ``glDeleteProgram(obj)`` / ``glDeleteShader(obj)`` / ``glDeleteSync(obj)``
      take a single name.
    - ``glDeleteTextures/Buffers/Framebuffers/VertexArrays/Queries`` may take
      ``(obj)``, ``([obj])``, or ``(1, [obj])`` and may raise ``TypeError``,
      ``ValueError`` or ``ctypes.ArgumentError`` on a signature mismatch.
    """
    import ctypes as _ctypes

    # Normalize to a plain Python int; ctypes/PyOpenGL can reject numpy scalars.
    try:
        obj = int(obj)
    except Exception:  # noqa: BLE001 - best-effort normalization for ctypes/PyOpenGL
        logger.debug("Failed to normalize GL object id %r to int", obj)
    obj_i = _ctypes.c_uint(int(obj)).value

    # Single-name delete functions must not be called with batch signatures;
    # on some PyOpenGL/ctypes combinations that produces confusing ArgumentError
    # failures instead of falling through cleanly.
    name = getattr(delete_fn, "__name__", "") or ""
    single_name = name in {
        "glDeleteProgram",
        "glDeleteShader",
        "glDeleteSync",
        "DeleteProgram",
        "DeleteShader",
        "DeleteSync",
    }
    if single_name:
        for arg in (obj_i, obj, _ctypes.c_uint(obj_i)):
            try:
                delete_fn(arg)  # type: ignore[arg-type]
                return
            except (TypeError, ValueError, _ctypes.ArgumentError):
                continue

    last_exc: Exception | None = None
    # Batch-delete functions: try spec-canonical (n, [ids]), list-only, then
    # bare scalar as a fallback.
    candidates = ((1, [obj_i]), ([obj_i],), (obj_i,), (1, [obj]), ([obj],), (obj,))
    for args in candidates:
        try:
            delete_fn(*args)  # type: ignore[arg-type]
            return
        except (TypeError, ValueError, _ctypes.ArgumentError) as exc:
            last_exc = exc
            continue
    logger.warning(
        "GL delete %r(%s) failed after all signatures: %s", delete_fn, obj_i, last_exc
    )
    if last_exc is not None:
        raise last_exc


def _upload_uniform_value(loc: int, value) -> None:
    """Dispatch the correct ``glUniform*`` call based on *value* type.

    All float values are run through :func:`_gl_finite` to guard against
    NaN/Inf poisoning a batch upload — a single ``inf`` in a
    ``glUniform1fv`` register batch poisons every subsequent register on
    some drivers.
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
            # Array upload — sanitize every element before the batch call.
            # Use vectorized numpy ops instead of a Python-level
            # _gl_finite(float(v)) loop over each element.
            arr = np.asarray(value, dtype=np.float32).ravel()
            if not arr.flags.writeable:
                arr = arr.copy()
            np.copyto(arr, 0.0, where=~np.isfinite(arr))
            sanitized = arr
            if n % 4 == 0:
                GL.glUniform4fv(loc, n // 4, sanitized)
            elif n % 3 == 0:
                GL.glUniform3fv(loc, n // 3, sanitized)
            elif n % 2 == 0:
                GL.glUniform2fv(loc, n // 2, sanitized)
            else:
                GL.glUniform1fv(loc, n, sanitized)
    elif isinstance(value, (list, tuple)):
        n = len(value)
        if n == 2:
            GL.glUniform2f(loc, _gl_finite(value[0]), _gl_finite(value[1]))
        elif n == 3:
            GL.glUniform3f(
                loc, _gl_finite(value[0]), _gl_finite(value[1]), _gl_finite(value[2])
            )
        elif n == 4:
            GL.glUniform4f(
                loc,
                _gl_finite(value[0]),
                _gl_finite(value[1]),
                _gl_finite(value[2]),
                _gl_finite(value[3]),
            )
        else:
            # Array upload — sanitize every element before the batch call.
            # Same vectorized approach as the ndarray branch above.
            arr = np.asarray(value, dtype=np.float32).ravel()
            if not arr.flags.writeable:
                arr = arr.copy()
            np.copyto(arr, 0.0, where=~np.isfinite(arr))
            if n % 4 == 0:
                GL.glUniform4fv(loc, n // 4, arr)
            elif n % 3 == 0:
                GL.glUniform3fv(loc, n // 3, arr)
            elif n % 2 == 0:
                GL.glUniform2fv(loc, n // 2, arr)
            else:
                GL.glUniform1fv(loc, n, arr)


class GLResourceOwner:
    """Per-renderer GL resource lifecycle owner.

    Each ``MilkdropRenderer`` instance owns its own :class:`GLResourceOwner`,
    preventing cross-renderer contamination of deletion queues, render-thread
    identity, and the ``_in_render_pass`` flag.

    Every GL object must hold an explicit :class:`GLResourceOwner` reference;
    the legacy ``get_default()`` escape hatch has been removed so that no
    GL resource can accidentally fall back to a process-wide shared owner.
    """

    _DELETE_FUNCS: ClassVar[
        dict[str, tuple[Callable[[int], bool], Callable[[int], None], str | None]]
    ] = {
        # resource-owner deletion — all GL object deletions flow through
        # queue_delete() + flush_deletes() so they are deferred to safe frames
        # and skipped on context invalidation.
        # glIs*() returns GLboolean (int 0/1); wrap in bool() so the callable
        # satisfies the declared Callable[[int], bool] signature and the
        # truthiness-tested ``if not valid:`` in flush_deletes() reads correctly.
        "program": (
            lambda id: bool(GL.glIsProgram(id)),
            lambda obj: _delete_gl(GL.glDeleteProgram, obj),
            "invalidate_program",
        ),
        "texture": (
            lambda id: bool(GL.glIsTexture(id)),
            lambda obj: _delete_gl(GL.glDeleteTextures, obj),
            "invalidate_texture",
        ),
        "fbo": (
            lambda id: bool(GL.glIsFramebuffer(id)),
            lambda obj: _delete_gl(GL.glDeleteFramebuffers, obj),
            "invalidate_framebuffer",
        ),
        "vao": (
            lambda id: bool(GL.glIsVertexArray(id)),
            lambda obj: _delete_gl(GL.glDeleteVertexArrays, obj),
            "invalidate_vertex_array",
        ),
        "buffer": (
            lambda id: bool(GL.glIsBuffer(id)),
            lambda obj: _delete_gl(GL.glDeleteBuffers, obj),
            "invalidate_buffer",
        ),
        "query": (
            lambda id: bool(GL.glIsQuery(id)),
            lambda obj: _delete_gl(GL.glDeleteQueries, obj),
            None,
        ),
        # Sync objects are owned by the same deferred-deletion queue so a
        # fence is never destroyed while a frame still references it.
        "sync": (
            lambda id: bool(GL.glIsSync(id)),
            lambda obj: _delete_gl(GL.glDeleteSync, obj),
            None,
        ),
    }

    def __init__(self, context_is_current: Callable[[], bool] | None = None):
        self._delete_queues: dict[str, list] = {}
        self._delete_queues_lock: threading.Lock = threading.Lock()
        self._render_thread_id: int | None = None
        self._in_render_pass: bool = False
        self._state_manager: GLStateManager | None = None
        self._program_retire_hook: Callable | None = None
        self._context_is_current = context_is_current
        # Live timer-query IDs allocated via create_query(); discarded on
        # context invalidation (numeric names are then meaningless).
        self._live_queries: set[int] = set()
        # Per-owner GLSL dialect cache. Preferred over the process-wide
        # fallback so multi-context / multi-renderer setups do not share a
        # stale dialect after one context is recreated.
        self._glsl_flavor: tuple[str, bool, tuple[int, int]] | None = None

    def enter_render_context(self):
        self._render_thread_id = threading.current_thread().ident

    def exit_render_context(self):
        self._render_thread_id = None

    def invalidate_context(self, reason="context_destroyed"):
        """Drop all cached GL state after a context is destroyed/recreated.

        .. warning::

           **Invariant — only call this from genuine GL context lifecycle
           paths.**  After a GL context loss and recreation, old numeric
           object names (texture/FBO/program/VAO/buffer IDs) are no longer
           valid in the new context.  ``flush_deletes()`` would see
           ``glIs*() == False`` for every stale ID and — **before** the fix
           to ``flush_deletes`` — retain them indefinitely.  Even **with**
           that fix, the deletion queues may still contain IDs whose
           validity is unknown, and the render-thread identity and
           ``_in_render_pass`` flag are stale.

           Calling this method **while the old context is still alive** is
           catastrophic: the queues (which reference live, valid GL object
           names) are discarded and those objects can never be deleted,
           leaking GPU memory until the process exits.  It must therefore
           only be invoked from actual context teardown/recreation paths
           (``on_realize`` re-realize, ``_reinitialize_gl_resources``
           during context-loss recovery), never from mid-frame reset or
           error handlers.

        The ``reason`` argument is a free-form label recorded for
        diagnostics so callers must self-document whether the call is a
        *legitimate* context-loss invalidation (``"context_lost"``,
        ``"realize"``) or a suspicious one (``"error_recovery"``,
        ``"frame_reset"``) that would violate the invariant above.

        This method clears the deletion queues, resets the render-thread
        identity, and clears the render-pass flag so the owner starts from a
        clean slate in the new context.  It must be called **before** any new
        resources are created or deletions queued.
        """
        if self._in_render_pass:
            if reason not in ("context_lost", "realize"):
                logger.error(
                    "GLResourceOwner.invalidate_context(reason=%r) called while "
                    "a render pass is in flight — rejecting invalidation "
                    "unless this is a genuine context loss/realize event.",
                    reason,
                )
                return
            logger.warning(
                "GLResourceOwner.invalidate_context(reason=%r) called while "
                "a render pass is in flight — this discards deletion queues "
                "and may leak live GL objects if the old context is still "
                "alive.",
                reason,
            )
        logger.debug("GLResourceOwner.invalidate_context(reason=%r)", reason)
        with self._delete_queues_lock:
            self._delete_queues.clear()
            self._live_queries.clear()
        self._render_thread_id = None
        self._in_render_pass = False
        self._glsl_flavor = None
        if self._state_manager is not None:
            self._state_manager.reset()
        # Stale program objects and the process-wide GLSL dialect cache must
        # not outlive the context that produced them.
        GLProgram.clear_registry()
        reset_glsl_flavor_cache()

    def stats(self) -> dict[str, Any]:
        """Return a snapshot of the current deletion queues and resource sets.

        Useful for debugging resource leaks: callers can compare successive
        snapshots to see which IDs have been created, queued, or retained.
        """
        with self._delete_queues_lock:
            queues = {k: set(v) for k, v in self._delete_queues.items()}
            live_queries = set(self._live_queries)
        return {
            "delete_queues": queues,
            "live_queries": live_queries,
            "render_thread_id": (
                {self._render_thread_id} if self._render_thread_id else set()
            ),
            "glsl_flavor": self._glsl_flavor,
        }

    def _get_delete_queue(self, resource_type: str) -> list:
        if resource_type not in self._delete_queues:
            self._delete_queues[resource_type] = []
        return self._delete_queues[resource_type]

    def create_query(self) -> int:
        """Allocate a single timer query object and track it as live.

        ``glGenQueries`` does not mutate bindable GL state, so it does not
        require a state guard.  The ID is recorded in ``_live_queries`` so
        leaks are visible via :meth:`stats` and so context invalidation can
        drop the tracking set.

        Prefer :meth:`release_query` (or ``queue_delete("query", id)``) when
        the query is no longer needed.  On context loss, tracked IDs are
        discarded without a GL delete — the names are invalid in the new
        context.
        """
        qid = _int_gl_id(GL.glGenQueries(1))
        if qid > 0:
            with self._delete_queues_lock:
                self._live_queries.add(qid)
        return qid

    def release_query(self, gl_id: int) -> None:
        """Queue a timer query for deletion and drop it from the live set.

        Equivalent to ``queue_delete("query", gl_id)`` with explicit live-set
        maintenance.  Safe to call with ``<= 0`` or an already-released ID.
        """
        self.queue_delete("query", gl_id)

    def queue_delete(self, resource_type: str, gl_id: int):
        gl_id = int(gl_id)
        if gl_id <= 0:
            return
        if resource_type not in self._DELETE_FUNCS:
            raise ValueError(
                f"Unsupported GL resource type for deletion: {resource_type!r}"
            )
        with self._delete_queues_lock:
            if resource_type == "query":
                self._live_queries.discard(gl_id)
            q = self._get_delete_queue(resource_type)
            if gl_id not in q:
                q.append(gl_id)
                retire_hook = getattr(self, "_program_retire_hook", None)
                if resource_type == "program" and retire_hook is not None:
                    retire_hook(gl_id)

    def flush_deletes(self):
        """Flush the deletion queues for this owner.

        GL resources can only be safely deleted from the thread that owns
        the active GL context (and from within a context that's current).
        Worker threads queue deletes via :meth:`queue_delete`; the actual
        deletion must happen on the render thread.

        Crucially, the queue is **not** cleared before attempting deletion.
        If the GL context is not current or a driver operation fails, the
        resource remains in the queue for a future successful flush.  This
        prevents the VRAM-leak / resource-loss scenario that occurred with
        the previous clear-then-delete approach during context loss or
        ``on_unrealize``.

        A ``glIs*() == False`` result means the object is already gone —
        either it was already deleted, or (after a context loss/recreation)
        the numeric name no longer refers to a valid object in the new
        context.  In **both** cases the ID is dropped from the queue so that
        stale names do not linger and risk being reused by OpenGL for
        new resources.  Only IDs whose ``glDelete*`` call itself raises are
        retained for a future retry.

        In debug mode (``DEBUG_GL_DELETE_VALIDATION`` True), each ID is
        validated with ``glIs*`` before deletion.  In normal mode, IDs are
        deleted directly to avoid the CPU/driver synchronization cost of
        repeated validation queries during rapid preset switching.
        """
        current_id = threading.current_thread().ident
        if self._render_thread_id is not None and current_id != self._render_thread_id:
            raise RuntimeError(
                "GL deletion attempted off render thread "
                f"(current={current_id}, render={self._render_thread_id})"
            )
        if self._context_is_current is None:
            logger.warning(
                "Skipping GL deletion flush because no current-context "
                "validator is configured"
            )
            return
        try:
            if not self._context_is_current():
                logger.debug("Skipping GL deletion flush without its GL context")
                return
        except Exception:
            logger.exception("Unable to verify the GL context before deletion")
            return
        validate = DEBUG_GL_DELETE_VALIDATION
        with self._delete_queues_lock:
            for resource_type, q in list(self._delete_queues.items()):
                if not q:
                    continue
                entry = self._DELETE_FUNCS.get(resource_type)
                if entry is None:
                    raise ValueError(
                        f"Unsupported GL resource type in delete queue: {resource_type}"
                    )
                validate_fn, delete_fn, invalidate_attr = entry
                remaining = []
                for single_id in q:
                    if validate:
                        valid = False
                        isvalid_failed = False
                        try:
                            valid = validate_fn(single_id)
                        except GL.GLError as e:
                            logger.warning(
                                f"Failed to validate {resource_type} {single_id}: {e}"
                            )
                            isvalid_failed = True
                        if not valid:
                            if isvalid_failed:
                                remaining.append(single_id)
                            continue
                    try:
                        delete_fn(single_id)
                        if self._state_manager is not None and invalidate_attr is not None:
                            getattr(self._state_manager, invalidate_attr)(single_id)
                    except (GL.GLError, _GLPyError.Error, TypeError, ValueError) as e:
                        logger.warning(f"Failed to delete {resource_type} {single_id}: {e}")
                        remaining.append(single_id)
                q[:] = remaining


TEXTURE_UNITS = {
    "primary": 0,
    "texture": 1,
    "noise_lq": 2,
    "noise_mq": 3,
    "noise_hq": 4,
    "megabuf": 7,
}
TEXTURE_UNIT_MEGABUF = TEXTURE_UNITS["megabuf"]

# Ownership record for the reserved texture units.  This table is
# descriptive: it documents which pass owns which unit so that binds stay
# conflict-free.  It is not consulted at runtime — every bind passes an
# explicit unit number.
TEXTURE_UNIT_RESERVATIONS = {
    TEXTURE_UNITS["primary"]: "primary/feedback/bloom-source/transition",
    TEXTURE_UNITS["texture"]: "preset-texture/bloom-prevframe",
    TEXTURE_UNITS["noise_lq"]: "noise_lq",
    TEXTURE_UNITS["noise_mq"]: "noise_mq",
    TEXTURE_UNITS["noise_hq"]: "noise_hq",
    TEXTURE_UNITS["megabuf"]: "megabuf",
}


# gl-state-guard: approved
class GLStateManager:
    """Cached OpenGL state for one renderer / context.

    All cache mutations and GL calls that go through this manager must run on
    the single render thread that owns the active context.  The cache itself
    is not locked.

    Prefer ``GLResourceOwner.invalidate_context`` for full context teardown;
    ``invalidate_context`` on this class only resets the local cache.
    """

    __slots__ = (
        "_active_texture_unit",
        "_clear_color",
        "_default_fb_unpresentable",
        "_owner",
        "array_buffer",
        "blend",
        "blend_dst",
        "blend_eq_alpha",
        "blend_eq_rgb",
        "blend_src",
        "cull_face",
        "depth_test",
        "element_array_buffer",
        "fbo_draw",
        "fbo_read",
        "line_width",
        "max_combined_texture_units",
        "pixel_pack_buffer",
        "program",
        "scissor",
        "scissor_test",
        "texture",
        "texture_unit",
        "textures_per_unit",
        "unpack_alignment",
        "vao",
        "viewport",
    )

    def __init__(self, owner: GLResourceOwner):
        self._owner = owner
        if owner is not None:
            owner._state_manager = self
        self.program = 0
        self.vao = 0
        self.texture = 0
        self.texture_unit = -1
        # Map of texture unit -> {target: texture_id}, so multiple texture targets
        # (e.g. GL_TEXTURE_2D and GL_TEXTURE_BUFFER) can be bound simultaneously
        # on the same unit. The single ``texture`` /\n        # ``texture_unit`` slots remain the "most recently bound" view used
        # by callers that don't care which unit is active.
        self.textures_per_unit: dict[int, dict[int, int]] = {}
        # Currently bound ARRAY_BUFFER. VAOs cache their own element-array
        # binding, but the wave/shape rendering paths still bind an
        # ARRAY_BUFFER directly when updating interleaved vertex data; we
        # track it so redundant binds are skipped and snapshot/restore
        # returns the buffer to its previous state.
        self.array_buffer = 0
        self.element_array_buffer = 0
        self._active_texture_unit = 0
        self.fbo_read = 0
        self.fbo_draw = 0
        self.blend = False
        self.blend_src = int(GL.GL_SRC_ALPHA)
        self.blend_dst = int(GL.GL_ONE)
        self.blend_eq_rgb = int(GL.GL_FUNC_ADD)
        self.blend_eq_alpha = int(GL.GL_FUNC_ADD)
        self.depth_test = False
        self.cull_face = False
        self.scissor = (0, 0, 0, 0)
        self.scissor_test = False
        self.viewport = (0, 0, 0, 0)
        self.line_width = 1.0
        # Cached GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS so callers that hand out
        # texture units (e.g. custom samplers) never allocate past the limit.
        # Queried lazily during snapshot(); falls back to 8 if the query fails.
        self.max_combined_texture_units = 8
        self._clear_color = (0.0, 0.0, 0.0, 1.0)
        self.pixel_pack_buffer = 0
        self.unpack_alignment = 4
        # False = default framebuffer assumed presentable until a real draw
        # to FBO 0 fails.  True = a draw to FBO 0 returned
        # GL_INVALID_FRAMEBUFFER_OPERATION and the default FB should be
        # skipped.  Context-level property, reset on realize/unrealize.
        self._default_fb_unpresentable: bool = False

    def reset(self):
        self.program = 0
        self.vao = 0
        self.texture = 0
        self.texture_unit = -1
        self.textures_per_unit = {}
        self.array_buffer = 0
        self.element_array_buffer = 0
        self._active_texture_unit = 0
        self.fbo_read = 0
        self.fbo_draw = 0
        self.pixel_pack_buffer = 0
        self.unpack_alignment = 4
        self.blend = False
        self.blend_src = int(GL.GL_SRC_ALPHA)
        self.blend_dst = int(GL.GL_ONE)
        self.blend_eq_rgb = int(GL.GL_FUNC_ADD)
        self.blend_eq_alpha = int(GL.GL_FUNC_ADD)
        self.depth_test = False
        self.cull_face = False
        self.scissor = (0, 0, 0, 0)
        self.scissor_test = False
        self.viewport = (0, 0, 0, 0)
        self.line_width = 1.0
        self.max_combined_texture_units = 8
        self._clear_color = (0.0, 0.0, 0.0, 1.0)
        self._default_fb_unpresentable = False

    def invalidate_context(self, reason: str = "context_destroyed") -> None:
        """Reset cached state only.

        Full context-loss teardown (delete queues, render-thread identity,
        program registry, GLSL dialect cache) must go through
        ``GLResourceOwner.invalidate_context``.  This method is a thin
        cache reset kept for callers that already hold only the state
        manager reference.
        """
        logger.debug("GLStateManager.invalidate_context(reason=%r)", reason)
        self.reset()

    # --- Snapshot/restore methods for _gl_state_guard ---

    def snapshot(self) -> dict | None:
        """Snapshot the actual driver state (expensive, queries OpenGL).

        Only ``GL_TEXTURE_BINDING_2D`` (and the active unit) is read back.
        Non-2D targets (e.g. ``GL_TEXTURE_BUFFER`` for megabuf) are **not**
        reconstructed from the driver.  Callers that bind non-2D targets
        inside a strict ``_gl_state_guard`` must re-bind them after the
        guard exits; ``restore_validated`` will not put them back.
        """
        try:
            try:
                self.max_combined_texture_units = int(
                    GL.glGetIntegerv(GL.GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS)
                )
            except Exception:  # noqa: BLE001 - driver/query may be unavailable
                logger.debug("Unable to query GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS")
            return {
                "program": GL.glGetIntegerv(GL.GL_CURRENT_PROGRAM),
                "vao": GL.glGetIntegerv(GL.GL_VERTEX_ARRAY_BINDING),
                "array_buffer": GL.glGetIntegerv(GL.GL_ARRAY_BUFFER_BINDING),
                "element_array_buffer": GL.glGetIntegerv(GL.GL_ELEMENT_ARRAY_BUFFER_BINDING),
                "texture_2d": GL.glGetIntegerv(GL.GL_TEXTURE_BINDING_2D),
                "active_texture": GL.glGetIntegerv(GL.GL_ACTIVE_TEXTURE) - GL.GL_TEXTURE0,
                "fbo_draw": GL.glGetIntegerv(GL.GL_DRAW_FRAMEBUFFER_BINDING),
                "fbo_read": GL.glGetIntegerv(GL.GL_READ_FRAMEBUFFER_BINDING),
                "blend": GL.glGetBooleanv(GL.GL_BLEND)[0],
                "blend_src": GL.glGetIntegerv(GL.GL_BLEND_SRC),
                "blend_dst": GL.glGetIntegerv(GL.GL_BLEND_DST),
                "blend_eq_rgb": GL.glGetIntegerv(GL.GL_BLEND_EQUATION_RGB),
                "blend_eq_alpha": GL.glGetIntegerv(GL.GL_BLEND_EQUATION_ALPHA),
                "depth_test": GL.glGetBooleanv(GL.GL_DEPTH_TEST)[0],
                "cull_face": GL.glGetBooleanv(GL.GL_CULL_FACE)[0],
                "scissor_test": GL.glGetBooleanv(GL.GL_SCISSOR_TEST)[0],
                "scissor": tuple(GL.glGetIntegerv(GL.GL_SCISSOR_BOX)),
                "viewport": tuple(GL.glGetIntegerv(GL.GL_VIEWPORT)),
                "line_width": GL.glGetFloatv(GL.GL_LINE_WIDTH)[0],
                "pixel_pack_buffer": GL.glGetIntegerv(GL.GL_PIXEL_PACK_BUFFER_BINDING),
                "unpack_alignment": GL.glGetIntegerv(GL.GL_UNPACK_ALIGNMENT),
                "clear_color": tuple(GL.glGetFloatv(GL.GL_COLOR_CLEAR_VALUE)),
            }
        except Exception:
            logger.debug("GL state snapshot failed", exc_info=True)
            return None

    def cheap_snapshot(self) -> dict:
        """Snapshot only the manager's tracked cache (fast, no driver queries)."""
        return {
            "program": self.program,
            "vao": self.vao,
            "array_buffer": self.array_buffer,
            "element_array_buffer": self.element_array_buffer,
            "texture": self.texture,
            "texture_unit": self.texture_unit,
            "textures_per_unit": {k: v.copy() for k, v in self.textures_per_unit.items()},
            "fbo_draw": self.fbo_draw,
            "fbo_read": self.fbo_read,
            "blend": self.blend,
            "blend_src": self.blend_src,
            "blend_dst": self.blend_dst,
            "blend_eq_rgb": self.blend_eq_rgb,
            "blend_eq_alpha": self.blend_eq_alpha,
            "depth_test": self.depth_test,
            "cull_face": self.cull_face,
            "scissor_test": self.scissor_test,
            "scissor": self.scissor,
            "viewport": self.viewport,
            "line_width": self.line_width,
            "pixel_pack_buffer": self.pixel_pack_buffer,
            "unpack_alignment": self.unpack_alignment,
            "clear_color": self._clear_color,
        }

    def restore_validated(self, snap: dict) -> None:
        """Restore state from a full driver snapshot, validating against cache."""
        if snap is None:
            self.restore_fast(self.cheap_snapshot())
            return
        # Restore driver state
        if "program" in snap:
            GL.glUseProgram(snap["program"])
            self.program = snap["program"]
        if "vao" in snap:
            GL.glBindVertexArray(snap["vao"])
            self.vao = snap["vao"]
        if "array_buffer" in snap:
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, snap["array_buffer"])
            self.array_buffer = snap["array_buffer"]
        if "element_array_buffer" in snap:
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, snap["element_array_buffer"])
            self.element_array_buffer = snap["element_array_buffer"]
        if "active_texture" in snap:
            GL.glActiveTexture(GL.GL_TEXTURE0 + snap["active_texture"])
            self._active_texture_unit = snap["active_texture"]
        if "texture_2d" in snap:
            GL.glBindTexture(GL.GL_TEXTURE_2D, snap["texture_2d"])
            self.texture = snap["texture_2d"]
            self.texture_unit = snap.get("active_texture", 0)
        if "fbo_draw" in snap:
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, snap["fbo_draw"])
            self.fbo_draw = snap["fbo_draw"]
        if "fbo_read" in snap:
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, snap["fbo_read"])
            self.fbo_read = snap["fbo_read"]
        if "blend" in snap:
            if snap["blend"]:
                GL.glEnable(GL.GL_BLEND)
            else:
                GL.glDisable(GL.GL_BLEND)
            self.blend = snap["blend"]
        if "blend_src" in snap and "blend_dst" in snap:
            GL.glBlendFuncSeparate(
                snap["blend_src"],
                snap["blend_dst"],
                snap.get("blend_src_alpha", snap["blend_src"]),
                snap.get("blend_dst_alpha", snap["blend_dst"]),
            )
            self.blend_src = snap["blend_src"]
            self.blend_dst = snap["blend_dst"]
        if "blend_eq_rgb" in snap and "blend_eq_alpha" in snap:
            GL.glBlendEquationSeparate(snap["blend_eq_rgb"], snap["blend_eq_alpha"])
            self.blend_eq_rgb = snap["blend_eq_rgb"]
            self.blend_eq_alpha = snap["blend_eq_alpha"]
        if "depth_test" in snap:
            if snap["depth_test"]:
                GL.glEnable(GL.GL_DEPTH_TEST)
            else:
                GL.glDisable(GL.GL_DEPTH_TEST)
            self.depth_test = snap["depth_test"]
        if "cull_face" in snap:
            if snap["cull_face"]:
                GL.glEnable(GL.GL_CULL_FACE)
            else:
                GL.glDisable(GL.GL_CULL_FACE)
            self.cull_face = snap["cull_face"]
        if "scissor_test" in snap:
            if snap["scissor_test"]:
                GL.glEnable(GL.GL_SCISSOR_TEST)
            else:
                GL.glDisable(GL.GL_SCISSOR_TEST)
            self.scissor_test = snap["scissor_test"]
        if "scissor" in snap:
            GL.glScissor(*snap["scissor"])
            self.scissor = snap["scissor"]
        if "viewport" in snap:
            GL.glViewport(*snap["viewport"])
            self.viewport = snap["viewport"]
        if "line_width" in snap:
            GL.glLineWidth(snap["line_width"])
            self.line_width = snap["line_width"]
        if "pixel_pack_buffer" in snap:
            GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, snap["pixel_pack_buffer"])
            self.pixel_pack_buffer = snap["pixel_pack_buffer"]
        if "unpack_alignment" in snap:
            GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, snap["unpack_alignment"])
            self.unpack_alignment = snap["unpack_alignment"]
        if "clear_color" in snap:
            GL.glClearColor(*snap["clear_color"])
            self._clear_color = snap["clear_color"]
        # Restore textures_per_unit
        if "textures_per_unit" in snap:
            for unit, targets in snap["textures_per_unit"].items():
                if unit != self._active_texture_unit:
                    GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
                    self._active_texture_unit = unit
                for target, tid in targets.items():
                    GL.glBindTexture(target, tid)
                    self.textures_per_unit.setdefault(unit, {})[target] = tid

    def restore_fast(self, snap: dict) -> None:
        """Restore state from a cheap snapshot (cache only, no driver queries)."""
        if snap is None:
            return
        self.program = snap.get("program", 0)
        self.vao = snap.get("vao", 0)
        self.array_buffer = snap.get("array_buffer", 0)
        self.element_array_buffer = snap.get("element_array_buffer", 0)
        self.texture = snap.get("texture", 0)
        self.texture_unit = snap.get("texture_unit", -1)
        self.textures_per_unit = {
            k: v.copy() for k, v in snap.get("textures_per_unit", {}).items()
        }
        self.fbo_draw = snap.get("fbo_draw", 0)
        self.fbo_read = snap.get("fbo_read", 0)
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fbo_read)
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo_draw)
        self.blend = snap.get("blend", False)
        self.blend_src = snap.get("blend_src", int(GL.GL_SRC_ALPHA))
        self.blend_dst = snap.get("blend_dst", int(GL.GL_ONE))
        self.blend_eq_rgb = snap.get("blend_eq_rgb", int(GL.GL_FUNC_ADD))
        self.blend_eq_alpha = snap.get("blend_eq_alpha", int(GL.GL_FUNC_ADD))
        self.depth_test = snap.get("depth_test", False)
        self.cull_face = snap.get("cull_face", False)
        self.scissor_test = snap.get("scissor_test", False)
        self.scissor = snap.get("scissor", (0, 0, 0, 0))
        self.viewport = snap.get("viewport", (0, 0, 0, 0))
        self.line_width = snap.get("line_width", 1.0)
        self.pixel_pack_buffer = snap.get("pixel_pack_buffer", 0)
        self.unpack_alignment = snap.get("unpack_alignment", 4)
        self._clear_color = snap.get("clear_color", (0.0, 0.0, 0.0, 1.0))

    # --- State-binding methods (track cache to skip redundant GL calls) ---

    def active_texture(self, unit: int) -> None:
        """Set the active texture unit."""
        if self._active_texture_unit != unit:
            GL.glActiveTexture(int(GL.GL_TEXTURE0) + int(unit))
            self._active_texture_unit = unit

    def bind_texture(self, texture: int, unit: int = 0, target: int = GL.GL_TEXTURE_2D) -> None:
        """Bind a texture to a texture unit, skipping redundant binds."""
        if unit != self._active_texture_unit:
            self.active_texture(unit)
        unit_textures = self.textures_per_unit.setdefault(unit, {})
        if unit_textures.get(target) != texture:
            GL.glBindTexture(target, texture)
            unit_textures[target] = texture
        # Update the "most recently bound" view for callers that don't track units
        self.texture = texture
        self.texture_unit = unit

    def bind_array_buffer(self, buffer: int) -> None:
        """Bind an array buffer, skipping redundant binds."""
        if self.array_buffer != buffer:
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, buffer)
            self.array_buffer = buffer

    def bind_element_array_buffer(self, buffer: int) -> None:
        """Bind an element array buffer, skipping redundant binds."""
        if self.element_array_buffer != buffer:
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, buffer)
            self.element_array_buffer = buffer

    def bind_vertex_array(self, vao: int) -> None:
        """Bind a vertex array object, skipping redundant binds."""
        if self.vao != vao:
            GL.glBindVertexArray(vao)
            self.vao = vao

    def bind_draw_fbo(self, fbo: int, force: bool = False) -> None:
        """Bind a draw framebuffer, skipping redundant binds."""
        if force or self.fbo_draw != fbo:
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, fbo)
            self.fbo_draw = fbo

    def bind_read_fbo(self, fbo: int) -> None:
        """Bind a read framebuffer, skipping redundant binds."""
        if self.fbo_read != fbo:
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, fbo)
            self.fbo_read = fbo

    def bind_framebuffer(self, target: int, fbo: int) -> None:
        """Bind a framebuffer to a target (DRAW/READ/BOTH)."""
        if target == GL.GL_DRAW_FRAMEBUFFER:
            self.bind_draw_fbo(fbo)
        elif target == GL.GL_READ_FRAMEBUFFER:
            self.bind_read_fbo(fbo)
        else:
            GL.glBindFramebuffer(target, fbo)
            if target in (GL.GL_FRAMEBUFFER, GL.GL_DRAW_FRAMEBUFFER):
                self.fbo_draw = fbo
            if target in (GL.GL_FRAMEBUFFER, GL.GL_READ_FRAMEBUFFER):
                self.fbo_read = fbo

    def use_program(self, program: int, force: bool = False) -> None:
        """Use a shader program, skipping redundant binds."""
        if force or self.program != program:
            GL.glUseProgram(program)
            self.program = program

    def set_viewport(self, x: int, y: int, width: int, height: int) -> None:
        """Set the viewport, skipping redundant calls."""
        vp = (x, y, width, height)
        if self.viewport != vp:
            GL.glViewport(x, y, width, height)
            self.viewport = vp

    def set_clear_color(self, r: float, g: float, b: float, a: float) -> None:
        """Set the clear color, skipping redundant calls."""
        color = (r, g, b, a)
        if self._clear_color != color:
            GL.glClearColor(r, g, b, a)
            self._clear_color = color

    def clear(self, mask: int) -> None:
        """Clear the current framebuffer."""
        GL.glClear(mask)

    def set_blend_func(
        self, src: int, dst: int, src_alpha: int | None = None, dst_alpha: int | None = None
    ) -> None:
        """Set the blend function, skipping redundant calls.

        Blend equations are independent and are left unchanged; call
        :meth:`set_blend_equation` separately when a non-``GL_FUNC_ADD``
        equation is required.
        """
        if src_alpha is None:
            src_alpha = src
        if dst_alpha is None:
            dst_alpha = dst
        if self.blend_src != src or self.blend_dst != dst:
            GL.glBlendFuncSeparate(src, dst, src_alpha, dst_alpha)
            self.blend_src = src
            self.blend_dst = dst

    def set_blend_equation(self, eq_rgb: int, eq_alpha: int | None = None) -> None:
        """Set the blend equation, skipping redundant calls."""
        if eq_alpha is None:
            eq_alpha = eq_rgb
        if self.blend_eq_rgb != eq_rgb or self.blend_eq_alpha != eq_alpha:
            GL.glBlendEquationSeparate(eq_rgb, eq_alpha)
            self.blend_eq_rgb = eq_rgb
            self.blend_eq_alpha = eq_alpha

    def enable_blend(self) -> None:
        """Enable blending."""
        if not self.blend:
            GL.glEnable(GL.GL_BLEND)
            self.blend = True

    def disable_blend(self) -> None:
        """Disable blending."""
        if self.blend:
            GL.glDisable(GL.GL_BLEND)
            self.blend = False

    def set_line_width(self, width: float) -> None:
        """Set the line width, skipping redundant calls."""
        if self.line_width != width:
            GL.glLineWidth(width)
            self.line_width = width

    def buffer_data(
        self, target: int, size: int, data: object | None, usage: int
    ) -> None:
        """Upload buffer data."""
        GL.glBufferData(target, size, data, usage)

    def buffer_sub_data(self, target: int, offset: int, size: int, data: object) -> None:
        """Update a subregion of a buffer."""
        GL.glBufferSubData(target, offset, size, data)

    # --- Invalidation helpers for resource deletion ---

    def invalidate_buffer(self, buffer: int) -> None:
        """Invalidate cached buffer state after deletion."""
        if self.array_buffer == buffer:
            self.array_buffer = 0
        if self.element_array_buffer == buffer:
            self.element_array_buffer = 0

    def invalidate_vertex_array(self, vao: int) -> None:
        """Invalidate cached VAO state after deletion."""
        if self.vao == vao:
            self.vao = 0

    def invalidate_program(self, program: int) -> None:
        """Invalidate cached program state after deletion."""
        if self.program == program:
            self.program = 0

    def invalidate_texture(self, texture: int) -> None:
        """Invalidate cached texture state after deletion."""
        if self.texture == texture:
            self.texture = 0
            self.texture_unit = -1
        # Also clean up from textures_per_unit
        for unit_dict in self.textures_per_unit.values():
            for target, tid in list(unit_dict.items()):
                if tid == texture:
                    del unit_dict[target]

    def invalidate_framebuffer(self, fbo: int) -> None:
        """Invalidate cached framebuffer state after deletion."""
        if self.fbo_draw == fbo:
            self.fbo_draw = 0
        if self.fbo_read == fbo:
            self.fbo_read = 0


@contextlib.contextmanager
def _gl_state_guard(gl_state, owner: GLResourceOwner | None = None, strict=True):
    """State-preserving guard for raw GL resource-creation regions.

    Architectural invariant
    -----------------------
    Every raw OpenGL state mutation must either be inside this guard (or
    another equivalent state-snapshot/restore boundary) or must update the
    :class:`GLStateManager` cache explicitly.  There is no third option.

    This is a hard rule: the renderer mixes a managed state manager with
    raw GL calls, resource creation, and direct framebuffer manipulation.
    Without strict enforcement, the manager's cache silently diverges from
    the driver, causing later short-circuited binds and rendering corruption.

    Resource creation (texture / FBO / framebuffer setup) necessarily issues raw
    ``glBindTexture`` / ``glActiveTexture`` calls that mutate the *actual* driver
    state without informing any :class:`GLStateManager` cache.  If such a region
    runs while a render pass expects a particular binding to be resident, the
    manager can later short-circuit a required bind because its cache no longer
    matches the driver.

    This guard snapshots the *actual* driver state (read back from OpenGL) on
    entry and restores it on exit — including through exceptions — so resource
    creation can never leak binding state into the surrounding render pipeline.
    The manager's cache is rebuilt during restore to stay coherent with the
    driver.

    When *gl_state* is ``None`` the guard is a no-op (used by callers that run
    before the state manager is wired up).

    Parameters
    ----------
    gl_state:
        The :class:`GLStateManager` to snapshot/restore.  May be ``None``.
    strict:
        When ``True`` (default), perform a full driver-state snapshot via
        :meth:`GLStateManager.snapshot` — correct but expensive (many
        ``glGetIntegerv`` + ``glActiveTexture`` round-trips).  Use this in
        resource-creation, resize, and preset-compile contexts.

        When ``False``, use :meth:`GLStateManager.cheap_snapshot` instead,
        which reads only the manager's tracked cache (no driver queries).
        This is safe for the 60 FPS render path but relies on the manager's
        cache being in sync with the driver.

    owner:
        The :class:`GLResourceOwner` to use for the ``_in_render_pass`` check.
        Must be passed explicitly — pass ``self._gl_resource_owner`` from the
        renderer for per-instance isolation.

    Render-path enforcement:
        If ``_gl_state_guard`` is entered with ``strict=True`` while
        the owner's ``_in_render_pass`` is ``True``, debug builds raise
        ``RuntimeError`` while optimized builds emit a warning.  The full
        snapshot still runs (it is correct), but the enforcement surfaces
        accidental creep of expensive queries into the frame loop so it
        can be downgraded to ``strict=False``.
    """
    if gl_state is None:
        yield
        return
    if strict and owner is not None and owner._in_render_pass:
        if __debug__:
            raise RuntimeError(
                "_gl_state_guard(strict=True) entered during render pass — "
                "use strict=False for 60 FPS-safe state preservation"
            )
        logger.warning(
            "_gl_state_guard(strict=True) entered during render pass — "
            "consider strict=False for 60 FPS-safe state preservation"
        )
    if strict:
        snap = gl_state.snapshot()
        if snap is None:
            snap = gl_state.cheap_snapshot()
    else:
        snap = gl_state.cheap_snapshot()
    try:
        yield
    finally:
        if strict:
            gl_state.restore_validated(snap)
        else:
            gl_state.restore_fast(snap)


# gl-state-guard: end-approved


class GLProgram:
    """Compiled vertex+fragment program with optional resource-owner lifecycle.

    Composition only: holds an optional :class:`GLResourceOwner` reference for
    deferred deletion.  Does **not** inherit the owner (programs are resources,
    not deletion-queue owners).

    After a GL context loss, call :meth:`clear_registry` (done automatically by
    ``GLResourceOwner.invalidate_context``) so stale numeric IDs are not reused.
    """

    _registry: ClassVar[dict[int, GLProgram]] = {}

    def __init__(
        self,
        vert_src: str = "",
        frag_src: str = "",
        owner: GLResourceOwner | None = None,
    ):
        self._id: int = 0
        self._vert_src = vert_src
        self._frag_src = frag_src
        self._uniform_locs: dict[str, int] = {}
        self._linked: bool = False
        self._active: bool = False
        self._retired: bool = False
        self._generation: int = 0
        self._owner = owner
        if vert_src and frag_src:
            self._compile_and_link(vert_src, frag_src)
            if self._id > 0:
                GLProgram._registry[self._id] = self

    @classmethod
    def clear_registry(cls) -> None:
        """Mark every registered program retired and drop the registry.

        Must be called after context loss/recreation so callers cannot keep
        using numeric IDs that are no longer valid in the new context.
        """
        for prog in list(cls._registry.values()):
            prog._retired = True
            prog._active = False
            prog._linked = False
            prog._id = 0
            prog._generation += 1
        cls._registry.clear()

    @property
    def active(self) -> bool:
        return self._active

    @active.setter
    def active(self, value: bool) -> None:
        self._active = bool(value)

    @property
    def retired(self) -> bool:
        return self._retired

    def retire(self, generation: int = 0) -> None:
        self._retired = True
        self._active = False
        self._generation = max(self._generation, generation)
        GLProgram._registry.pop(self._id, None)

    def _compile_and_link(self, vert_src: str, frag_src: str) -> bool:
        prog_id = _compile_gl_program(
            vert_src, frag_src, name="GLProgram", owner=self._owner
        )
        if prog_id > 0:
            self._id = prog_id
            self._linked = True
            GLProgram._registry[self._id] = self
            return True
        return False

    @property
    def id(self) -> int:
        return self._id

    def __int__(self) -> int:
        return self._id

    def __bool__(self) -> bool:
        return self._linked and self._id > 0

    def cleanup(self):
        if self._id > 0:
            GLProgram._registry.pop(self._id, None)
            if self._owner is not None:
                self._owner.queue_delete("program", self._id)
            self._id = 0
            self._linked = False
            self._active = False
            self._retired = True
            self._generation += 1



def _build_register_spec() -> tuple[str, str, str, str, str, str, frozenset, frozenset]:
    """Return the authoritative register uniform block, local aliases, and
    declared-variable set shared by the preset VM, equation evaluator,
    uniform uploader, and GLSL emitter.
    """
    q_uniforms = "uniform float u_q[33];"
    q_locals = "\n        ".join(f"float q{i} = u_q[{i}];" for i in range(1, 33))
    t_uniforms = "uniform float u_t[9];"
    t_locals = "\n        ".join(f"float t{i} = u_t[{i}];" for i in range(1, 9))
    reg_uniforms = "uniform float u_regs[100];"
    reg_locals = "\n        ".join(
        f"float reg{i:02d} = u_regs[{i}];" for i in range(100)
    )
    declared = frozenset(
        {
            *{f"q{i}" for i in range(1, 33)},
            *{f"t{i}" for i in range(1, 9)},
            *{f"reg{i:02d}" for i in range(100)},
            *{f"u_q[{i}]" for i in range(1, 33)},
            *{f"u_t[{i}]" for i in range(1, 9)},
            *{f"u_regs[{i}]" for i in range(100)},
        }
    )
    # Pixel-local variables declared in the main visual shader
    pixel_locals_declared = frozenset(
        {
            "uv",
            "uv2",
            "x",
            "y",
            "sx",
            "sy",
            "decay",
            "center",
            "offset",
            "scale",
            "rad",
            "ang",
            "aspect",
            "aspectx",
            "aspecty",
            "zoom",
            "rot",
            "warp",
            "cx",
            "cy",
            "dx",
            "dy",
            # Classic MilkDrop per-pixel color component outputs.  ``r``/``g``/``b``
            # are aliases for the per-pixel color (equivalent to ``ret.r/g/b``) and
            # must be recognized as declared locals so presets that write them
            # actually affect the output color instead of silently dropping the
            # assignment (they would otherwise be auto-declared as unused custom
            # locals).
            "r",
            "g",
            "b",
        }
    )
    # Silence unused variable warnings - these are used in generated GLSL code
    _ = q_locals, t_locals, reg_locals
    return (
        q_uniforms,
        q_locals,
        t_uniforms,
        t_locals,
        reg_uniforms,
        reg_locals,
        declared,
        pixel_locals_declared,
    )


(
    _Q_UNIFORMS,
    _Q_LOCALS,
    _T_UNIFORMS,
    _T_LOCALS,
    _REG_UNIFORMS,
    _REG_LOCALS,
    _REG_DECLARED,
    _PIXEL_LOCALS_DECLARED,
) = _build_register_spec()


# --- GLSL dialect adaptation -------------------------------------------------
#
# GtkGLArea hands us whatever context GDK negotiated, which is very often an
# OpenGL ES context (Wayland/GTK4 default) whose GLSL compiler stops at
# ``#version 320 es``.  Shader sources are authored once with
# ``#version 330 core`` because that is the flavour the math in
# ``milkdrop_helpers.glsl`` is written against, so every compiled program is
# adapted here instead of duplicating version strings at each call site.
#
# Two adaptations are required for ESSL:
#   1. the ``#version`` directive is replaced with the detected dialect;
#   2. ``layout(location = N) in`` is illegal for vertex inputs in ESSL, so
#      the qualifier is stripped and the location is bound by name with
#      ``glBindAttribLocation`` before linking.

# ``layout(location = N) in <type> <name>;`` as authored in the inline sources.
_ATTRIB_LOCATION_RE = re.compile(
    r"layout\s*\(\s*location\s*=\s*(\d+)\s*\)\s*"
    r"in\s+([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*;"
)
_VERSION_DIRECTIVE_RE = re.compile(r"^[ \t]*#version[^\n]*", re.MULTILINE)

# ``float q1 = u_q[1];`` — the register alias form produced by
# ``_build_register_spec``.
_ALIAS_DECL_RE = re.compile(r"^float\s+([A-Za-z_]\w*)\s*=")

# ---------------------------------------------------------------------------
# Identifier-scanning helpers for the per-pixel GLSL emission
# ---------------------------------------------------------------------------

# GLSL built-in function / type names, GLSL keywords, and template-local
# identifiers that may legitimately appear in the generated per-pixel code.
# These must never be treated as custom user-declared locals because they
# are either already declared (built-in / template) or are uniforms
# (u_*-prefixed, handled separately).
_PIXEL_CODE_KNOWN_IDENTS: frozenset[str] = frozenset(
    {
        # GLSL declaration / control-flow keywords
        "void",
        "main",
        "if",
        "else",
        "for",
        "while",
        "return",
        "break",
        "continue",
        "in",
        "out",
        "inout",
        "uniform",
        "const",
        "struct",
        "precision",
        "highp",
        "mediump",
        "lowp",
        # Scalar / vector / matrix / sampler types
        "vec2",
        "vec3",
        "vec4",
        "mat2",
        "mat3",
        "mat4",
        "mat2x2",
        "mat3x3",
        "mat4x4",
        "float",
        "int",
        "uint",
        "bool",
        "double",
        "sampler2D",
        "samplerCube",
        "sampler3D",
        "isampler2D",
        "usampler2D",
        "true",
        "false",
        # GLSL built-in functions (also in GLSLGenerator.glsl_builtins, but kept
        # here for completeness in case the generator set is extended later)
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
        "exp2",
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
        "roundEven",
        "mod",
        "modf",
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
        "faceforward",
        "texture",
        "texture2D",
        "texture3D",
        "textureCube",
        "texelFetch",
        "textureSize",
        "textureGrad",
        "textureOffset",
        "dFdx",
        "dFdy",
        "fwidth",
        # Fragment-shader template locals (fragColor case-insensitive)
        "fragColor",
        "FragColor",
        "fragTexCoord",
        "gl_FragCoord",
        "gl_FrontFacing",
        "gl_PointCoord",
        "uv",
        "uv2",
        "uv_w",
        "ret",
        "md_prev_color",
        "md_color",
        "md_p",
        "md_s",
        "md_c",
        "center",
        "offset",
        "scale",
        "prev",
        "tex_sample",
        "sample_uv",
        # MilkDrop helper functions emitted by milkdrop_helpers.glsl
        "megabuf_read",
        "invsqrt",
        "fact",
        # Misc
        "varying",
        "attribute",
    }
)
# GLSL reserved keywords that MilkDrop presets use as variable names.
# These cannot appear as identifiers in GLSL, so they are renamed to
# safe ``md_``-prefixed locals in the generated per-pixel code.  Only
# the subset that MilkDrop presets actually use as variable names is
# listed; type names (``float``, ``vec3`` …) and function names are
# handled separately by the GLSLGenerator and must not be touched here.
_GLSL_RESERVED_RENAMES: dict[str, str] = {
    "in": "md_in",
    "out": "md_out",
    "inout": "md_inout",
    "uniform": "md_uniform",
    "const": "md_const",
    "struct": "md_struct",
    "void": "md_void",
    "attribute": "md_attribute",
    "varying": "md_varying",
}
# Pre-compiled pattern: match any reserved keyword as a whole word.
_GLSL_RESERVED_RENAME_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _GLSL_RESERVED_RENAMES) + r")\b"
)
# Regex matching identifiers that are NOT immediately followed by `(`
# (i.e. not a function-call or type-constructor name).  Used to scan the
# generated per-pixel GLSL so that function names like ``sin`` or
# ``megabuf_read`` are not mistaken for undeclared user variables.
_PIXEL_CODE_IDENT_RE = re.compile(r"\b([A-Za-z_]\w*)\b(?!\s*\()")

# Only a version/profile complaint is worth retrying under another dialect; a
# genuine syntax or type error in a preset body must be reported as-is.
_VERSION_ERROR_RE = re.compile(
    r"not supported|#version|version\s*['\"]?\d|no such version|unsupported version",
    re.IGNORECASE,
)

# Fallback dialects, most capable first, used when the detected one rejects a
# source outright (some drivers under-report their GLSL version).
_FALLBACK_GLSL_DIRECTIVES: tuple[str, ...] = (
    "#version 320 es",
    "#version 300 es",
    "#version 330 core",
)

_GLSL_FLAVOR_CACHE: tuple[str, bool, tuple[int, int]] | None = None


def _decode_gl_string(value: object) -> str:
    """Return a GL string query result as ``str`` (PyOpenGL may return bytes)."""
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="ignore")
    if value is None:
        return ""
    return str(value)


def _parse_gl_version(text: str, *, is_es: bool = False) -> tuple[int, int] | None:
    """Extract the first ``major.minor`` pair from a GL version string."""
    match = re.search(r"(\d+)\.(\d+)", text)
    if match is None:
        return None
    major = int(match.group(1))
    minor = int(match.group(2))
    if is_es and minor >= 10:
        # GLSL ES versions in strings like "OpenGL ES GLSL ES 3.10" use decimal
        # minors but some drivers report them differently; normalize two-digit
        # minor encodings (e.g., 10 -> 1, 20 -> 2).
        minor //= 10
    return major, minor


def _select_glsl_directive(version: tuple[int, int] | None, is_es: bool) -> str:
    """Pick the highest ``#version`` directive the context is known to accept."""
    if is_es:
        if version is not None and version < (3, 0):
            # ESSL 1.00 cannot express ``in``/``out`` or integer uniforms; the
            # program will fail to compile, but with a directive that at least
            # states the truth about the context.
            return "#version 100"
        if version is not None and version < (3, 2):
            return f"#version {version[0]}{version[1]}0 es"
        return "#version 320 es"
    if version is not None and version < (3, 3):
        if version < (3, 0):
            return "#version 150"
        return f"#version {version[0]}{version[1]}0 core"
    return "#version 330 core"


def _flavor_from_directive(directive: str) -> tuple[str, bool, tuple[int, int]]:
    """Derive ``(directive, is_es, version)`` from a ``#version`` directive.

    Used when pinning a fallback dialect so the cached tuple describes the
    directive that actually compiled rather than the one the context claimed.
    """
    is_es = directive.endswith(" es")
    match = re.search(r"(\d+)", directive.lstrip("#version").strip())
    if match is None:
        return directive, is_es, (3, 2) if is_es else (3, 3)
    # GLSL encodes the version as three digits: major, then a two-digit minor
    # ("320" -> 3.2, "100" -> 1.0, "450" -> 4.5).
    digits = match.group(1)
    major = int(digits[0])
    minor = int(digits[1:3]) if len(digits) >= 3 else 0
    return directive, is_es, (major, minor)


def _fragment_highp_supported() -> bool:
    """True when ``highp`` is actually available in fragment shaders.

    ``highp`` is optional in ESSL fragment shaders; many mobile-class drivers
    report 0 for the high-float range and reject a shader that uses it.
    """
    # ``bool()`` guard first: PyOpenGL exposes a sentinel callable whose
    # ``__bool__`` is False when the underlying entry point is not available,
    # and calling it raises ``NullFunctionError``.  Short-circuit before the
    # call so the exception path is never hit on a healthy driver.
    fn = GL.glGetShaderPrecisionFormat
    if not bool(fn):
        return True
    try:
        fmt = fn(GL.GL_FRAGMENT_SHADER, GL.GL_HIGH_FLOAT)
        return int(fmt[0]) > 0
    except Exception:
        # Unavailable entry point or no current context: assume highp, which
        # is mandatory in desktop GLSL and honoured by every GLES 3.x driver
        # that exposes the query at all.
        logger.debug("Fragment highp precision query failed; assuming supported", exc_info=True)
        return True


def detect_glsl_flavor(
    owner: GLResourceOwner | None = None,
) -> tuple[str, bool, tuple[int, int]]:
    """Return ``(directive, is_es, version)`` for the current GL context.

    Preference order for the cache:

    1. *owner*'s ``_glsl_flavor`` when *owner* is provided and already set.
    2. The process-wide ``_GLSL_FLAVOR_CACHE`` (legacy single-context path).
    3. Fresh detection from ``GL_SHADING_LANGUAGE_VERSION`` / ``GL_VERSION``.

    When a fresh dialect is detected it is stored on *owner* (if given) and
    in the process-wide cache.  ``GLResourceOwner.invalidate_context`` clears
    both the owner field and the process cache via
    :func:`reset_glsl_flavor_cache`.

    Multi-context / multi-renderer setups should pass the owning
    ``GLResourceOwner`` so dialects do not leak across contexts.
    """
    global _GLSL_FLAVOR_CACHE
    if owner is not None and owner._glsl_flavor is not None:
        return owner._glsl_flavor
    if _GLSL_FLAVOR_CACHE is not None:
        if owner is not None:
            owner._glsl_flavor = _GLSL_FLAVOR_CACHE
        return _GLSL_FLAVOR_CACHE
    shading = ""
    gl_version = ""
    try:
        shading = _decode_gl_string(GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION))
    except (GL.GLError, _GLPyError.Error, AttributeError, TypeError):
        shading = ""
    try:
        gl_version = _decode_gl_string(GL.glGetString(GL.GL_VERSION))
    except (GL.GLError, _GLPyError.Error, AttributeError, TypeError):
        gl_version = ""
    is_es = bool(re.search(r"\bES\b", shading)) or "OpenGL ES" in gl_version
    version = _parse_gl_version(shading, is_es=is_es) or _parse_gl_version(
        gl_version, is_es=is_es
    )
    if version is None:
        # No usable version string: assume the ESSL 3.20 ceiling, which is the
        # most conservative dialect GDK ever hands us.
        is_es = True
        version = (3, 2)
    directive = _select_glsl_directive(version, is_es)
    flavor = (directive, is_es, version)
    _GLSL_FLAVOR_CACHE = flavor
    if owner is not None:
        owner._glsl_flavor = flavor
    logger.info(
        "GLSL dialect: %s (is_es=%s, version=%d.%d, GLSL=%r, GL=%r)",
        directive,
        is_es,
        version[0],
        version[1],
        shading,
        gl_version,
    )
    return flavor


def reset_glsl_flavor_cache(owner: GLResourceOwner | None = None) -> None:
    """Drop cached GLSL dialect(s) so the next compile re-detects them.

    Clears the process-wide cache always.  When *owner* is provided, also
    clears that owner's ``_glsl_flavor``.  ``GLResourceOwner.invalidate_context``
    already nulls the owner field before calling this with no argument.
    """
    global _GLSL_FLAVOR_CACHE
    _GLSL_FLAVOR_CACHE = None
    if owner is not None:
        owner._glsl_flavor = None


def _adapt_glsl_source(
    source: str, directive: str, *, is_vertex: bool
) -> tuple[str, list[tuple[int, str]]]:
    """Rewrite *source* for *directive*.

    Returns the adapted source and the ``(location, attribute name)`` pairs
    that must be bound with ``glBindAttribLocation`` before linking.  Binding
    by name is a no-op for drivers that honour ``layout(location = ...)`` and
    is required for ESSL, which has no location qualifier on vertex inputs.
    """
    bindings: list[tuple[int, str]] = []
    adapted = source
    if is_vertex:

        def _strip_qualifier(match: re.Match[str]) -> str:
            bindings.append((int(match.group(1)), match.group(3)))
            return f"in {match.group(2)} {match.group(3)};"

        adapted = _ATTRIB_LOCATION_RE.sub(_strip_qualifier, adapted)
    version_index = -1
    if _VERSION_DIRECTIVE_RE.search(adapted):
        adapted = _VERSION_DIRECTIVE_RE.sub(directive, adapted, count=1)
        version_index = next(
            (
                i
                for i, line in enumerate(adapted.split("\n"))
                if line.strip().startswith("#version")
            ),
            -1,
        )
    else:
        adapted = directive + "\n" + adapted.lstrip()
        version_index = 0
    if directive.endswith(" es") and not is_vertex:
        qualifier = "highp" if _fragment_highp_supported() else "mediump"
        lines = adapted.split("\n")
        precision_lines = {
            "float": f"precision {qualifier} float;",
            "int": f"precision {qualifier} int;",
            # GLSL ES has no implicit default precision for samplerBuffer.
            "samplerBuffer": f"precision {qualifier} samplerBuffer;",
        }
        missing_precisions = [
            declaration
            for value_type, declaration in precision_lines.items()
            if not re.search(
                rf"^[ \t]*precision[ \t]+(?:lowp|mediump|highp)[ \t]+{value_type}\s*;",
                adapted,
                re.MULTILINE,
            )
        ]
        if missing_precisions:
            lines[version_index + 1 : version_index + 1] = missing_precisions
            adapted = "\n".join(lines)
    return adapted, bindings


# Patterns that extract a 1-based source line number from common driver
# info-log formats (NVIDIA, Mesa, AMD, ANGLE, Apple).
# Tried in order per log line; more specific forms come first.
_GLSL_INFO_LOG_LINE_RES: tuple[re.Pattern[str], ...] = (
    # NVIDIA: "0(123) : error ..."  — shader number, then (line)
    re.compile(r"^\s*\d+\s*\(\s*(\d+)\s*\)"),
    # Mesa / common: "0:123(45): error ..." or "0:123: error"
    re.compile(r"^\s*\d+\s*:\s*(\d+)\b"),
    # AMD / some ES: "ERROR: 0:123: ..."
    re.compile(r"ERROR\s*:\s*\d+\s*:\s*(\d+)\s*:", re.IGNORECASE),
    # "ERROR: 123: ..."
    re.compile(r"ERROR\s*:\s*(\d+)\s*:", re.IGNORECASE),
    # "(123) :"
    re.compile(r"\((\d+)\)\s*:"),
    # "Error at line 123" / "error line 123"
    re.compile(r"[Ee]rror\s+(?:at\s+)?[Ll]ine\s+(\d+)"),
    # Generic ":123: error|warning"
    re.compile(r":(\d+)\s*:\s*(?:error|warning)\b", re.IGNORECASE),
)


def _decode_info_log(value: object) -> str:
    """Normalise a ``glGet*InfoLog`` result to a stripped ``str``."""
    if value is None:
        return ""
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="ignore").strip()
    return str(value).strip()


def _glsl_error_line_numbers(info_log: str) -> list[int]:
    """Extract unique 1-based source line numbers referenced in *info_log*."""
    found: list[int] = []
    seen: set[int] = set()
    for raw_line in info_log.splitlines():
        for pattern in _GLSL_INFO_LOG_LINE_RES:
            match = pattern.search(raw_line)
            if match is None:
                continue
            try:
                n = int(match.group(1))
            except (ValueError, IndexError):
                continue
            if n > 0 and n not in seen:
                seen.add(n)
                found.append(n)
            break
    return found


def _format_glsl_source_excerpt(
    source: str, line_numbers: list[int], *, context: int = 3, max_excerpts: int = 4
) -> str:
    """Return annotated source excerpts around the reported error lines.

    Line numbers are 1-based and refer to *source* after adaptation (the same
    text that was uploaded with ``glShaderSource``).
    """
    if not source or not line_numbers:
        return ""
    src_lines = source.splitlines()
    n = len(src_lines)
    chunks: list[str] = []
    covered: set[int] = set()
    for lineno in line_numbers[:max_excerpts]:
        start = max(1, lineno - context)
        end = min(n, lineno + context)
        if all(i in covered for i in range(start, end + 1)):
            continue
        block: list[str] = []
        for i in range(start, end + 1):
            covered.add(i)
            mark = ">>>" if i == lineno else "   "
            text = src_lines[i - 1] if 1 <= i <= n else ""
            block.append(f"{mark} {i:4d} | {text}")
        chunks.append("\n".join(block))
    return "\n\n".join(chunks)


def _format_glsl_compile_error(
    stage: str,
    info_log: str,
    source: str,
    *,
    directive: str = "",
    name: str = "",
) -> str:
    """Build a human-readable GLSL compile/link failure message.

    Includes the driver info log, the ``#version`` directive in use, and
    source excerpts around any line numbers the driver reported.
    """
    info_log = (info_log or "").strip() or "(no driver info log)"
    header_parts = [f"{stage} failed"]
    if name:
        header_parts.append(f"[{name}]")
    if directive:
        header_parts.append(f"under {directive}")
    header = " ".join(header_parts)

    line_numbers = _glsl_error_line_numbers(info_log)
    excerpt = _format_glsl_source_excerpt(source, line_numbers) if source else ""

    parts = [header, info_log]
    if line_numbers:
        parts.append(
            "Reported source line(s): " + ", ".join(str(n) for n in line_numbers)
        )
    if excerpt:
        parts.append("Source context:\n" + excerpt)
    return "\n".join(parts)


# --- Shader compilation cache --------------------------------------------------
# Cache compiled GL programs keyed by (vertex_shader_hash, fragment_shader_hash,
# directive) to avoid redundant compilations during preset switching.
# Max 64 entries to keep memory bounded; uses FIFO eviction.
_shader_compile_cache: dict[tuple[str, str, str], tuple[int, str]] = {}
_shader_compile_cache_maxsize = 64


def _shader_source_hash(source: str) -> str:
    """Compute a quick hash for shader source for cache keying."""
    import hashlib
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def _compile_gl_program_cached(
    vert_src: str,
    frag_src: str,
    directive: str,
    *,
    name: str = "",
) -> tuple[int, str]:
    """Compile+link a GL program with caching.
    
    Returns ``(program id, error message)``; the id is 0 on failure.
    Cache key is based on shader source hashes and directive to avoid
    redundant compilations during preset switching.
    """
    vhash = _shader_source_hash(vert_src)
    fhash = _shader_source_hash(frag_src)
    cache_key = (vhash, fhash, directive)

    # Check cache first
    if cache_key in _shader_compile_cache:
        cached_prog, cached_err = _shader_compile_cache[cache_key]
        if cached_prog > 0 and GL.glIsProgram(cached_prog):
            return cached_prog, cached_err
        # Cache entry is stale (program deleted), remove it
        del _shader_compile_cache[cache_key]

    # Compile fresh via the original helper
    result = _compile_gl_program_once(vert_src, frag_src, directive, name=name)

    # Store in cache (evict if too large, FIFO)
    if len(_shader_compile_cache) >= _shader_compile_cache_maxsize:
        first_key = next(iter(_shader_compile_cache))
        del _shader_compile_cache[first_key]
    _shader_compile_cache[cache_key] = result

    return result


def _compile_gl_program_once(
    vert_src: str,
    frag_src: str,
    directive: str,
    *,
    name: str = "",
) -> tuple[int, str]:
    """Compile+link one program under *directive*.

    Returns ``(program id, error message)``; the id is 0 on failure and the
    message is a formatted report (driver log + source context when available).
    """
    vert, vert_bindings = _adapt_glsl_source(vert_src, directive, is_vertex=True)
    frag, _ = _adapt_glsl_source(frag_src, directive, is_vertex=False)
    # Use the same text that was uploaded so reported line numbers match.
    vert_uploaded = vert.lstrip()
    frag_uploaded = frag.lstrip()
    vs = fs = prog = 0
    try:
        vs = GL.glCreateShader(GL.GL_VERTEX_SHADER)
        GL.glShaderSource(vs, vert_uploaded)
        GL.glCompileShader(vs)
        if not GL.glGetShaderiv(vs, GL.GL_COMPILE_STATUS):
            err = _decode_info_log(GL.glGetShaderInfoLog(vs))
            raise RuntimeError(
                _format_glsl_compile_error(
                    "Vertex shader compilation",
                    err,
                    vert_uploaded,
                    directive=directive,
                    name=name,
                )
            )
        fs = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
        GL.glShaderSource(fs, frag_uploaded)
        GL.glCompileShader(fs)
        if not GL.glGetShaderiv(fs, GL.GL_COMPILE_STATUS):
            err = _decode_info_log(GL.glGetShaderInfoLog(fs))
            raise RuntimeError(
                _format_glsl_compile_error(
                    "Fragment shader compilation",
                    err,
                    frag_uploaded,
                    directive=directive,
                    name=name,
                )
            )
        prog = GL.glCreateProgram()
        GL.glAttachShader(prog, vs)
        GL.glAttachShader(prog, fs)
        for location, attr_name in vert_bindings:
            GL.glBindAttribLocation(prog, location, attr_name.encode("utf-8"))
        GL.glLinkProgram(prog)
        if not GL.glGetProgramiv(prog, GL.GL_LINK_STATUS):
            err = _decode_info_log(GL.glGetProgramInfoLog(prog))
            # Link errors rarely include usable source line numbers; still
            # attach both adapted sources for diagnosis.
            combined = (
                "// ---- vertex ----\n"
                + vert_uploaded
                + "\n// ---- fragment ----\n"
                + frag_uploaded
            )
            raise RuntimeError(
                _format_glsl_compile_error(
                    "Shader program linking",
                    err,
                    combined,
                    directive=directive,
                    name=name,
                )
            )
        return int(prog) if prog is not None else 0, ""
    except Exception as e:
        # A fragment/vertex compile or link failure must never escape this
        # wrapper and crash the renderer: reject the candidate program and
        # let the caller keep the previously working preset.  Catch every
        # exception type here because some drivers raise non-GL errors
        # (e.g. from glGetShaderInfoLog on a lost context) that the narrower
        # tuple below would let propagate.
        logger.debug("GL program compilation/link failed", exc_info=True)
        message = str(e)
        if prog and int(prog) > 0:
            # Intentional direct delete: the program never left this helper
            # and was never registered with a GLResourceOwner.
            GL.glDeleteProgram(prog)
        return 0, message
    finally:
        # Shader objects are compile-local and never queued through the owner.
        if vs:
            GL.glDeleteShader(vs)
        if fs:
            GL.glDeleteShader(fs)


def _compile_gl_program(
    vert_src: str,
    frag_src: str,
    name: str = "shader",
    owner: GLResourceOwner | None = None,
) -> int:
    """Compile and link a vertex+fragment shader program.

    Returns the program ID, or 0 on failure.  When *owner* is provided the
    GLSL dialect cache is read from / written to that owner so multi-context
    setups stay isolated.
    """
    directive, _is_es, version = detect_glsl_flavor(owner)
    candidates = [directive]
    candidates.extend(d for d in _FALLBACK_GLSL_DIRECTIVES if d != directive)
    last_error = ""
    for index, candidate in enumerate(candidates):
        prog, error = _compile_gl_program_once(
            vert_src, frag_src, candidate, name=name
        )
        if prog > 0:
            if index:
                # The context under-reported its GLSL version; pin the dialect
                # that actually worked so later compiles skip the probe.
                # Derive is_es/version from the winning directive so the cached
                # tuple stays self-consistent -- keeping the originally detected
                # values would report a stale dialect in every later log line.
                global _GLSL_FLAVOR_CACHE
                flavor = _flavor_from_directive(candidate)
                _GLSL_FLAVOR_CACHE = flavor
                if owner is not None:
                    owner._glsl_flavor = flavor
                logger.info(
                    "Shader %s compiled under fallback dialect %s "
                    "(context reported %d.%d)",
                    name,
                    candidate,
                    version[0],
                    version[1],
                )
            return prog
        last_error = error
        if not _VERSION_ERROR_RE.search(error):
            break
    # Log the full formatted report (may be multi-line with source context).
    logger.error(
        "Shader compilation failed (%s) after trying: %s\n%s",
        name,
        ", ".join(candidates),
        last_error or "(no error detail)",
    )
    return 0
