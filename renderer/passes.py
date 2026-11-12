from __future__ import annotations

import ctypes
import logging
import math
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from OpenGL import GL
from OpenGL import error as _GLPyError

from gl_state import (
    _compile_gl_program,
    _gl_finite,
    _gl_state_guard,
    _int_gl_id,
)
from program_handle import (
    ProgramHandle,
)
from resources import (
    EnhancementLayer,
    PresetLoadResult,
    RenderTarget,
    TransitionPhase,
)

from .base import AudioLevels, RendererBase
from .config import (
    _COMPOSITE_UNIFORM_NAMES,
    _GL_COLOR_BUFFER_BIT,
    _GL_DEPTH_BUFFER_BIT,
    _SCREEN_BLIT_FRAG_SRC,
    _SCREEN_BLIT_VERT_SRC,
    _TRANSITION_BLEND_FRAG_SRC,
    _WAVE_SHAPE_FRAG_SRC,
    _WAVE_SHAPE_VERT_SRC,
    PresetBudgetExhausted,
)
from .presets import (
    PresetProgram,
    PresetVM,
    parse_milk_file_robust,
)

logger = logging.getLogger("milkdrop_renderer")


@contextmanager
def _gl_state_guard_deferred_flush(gl_state, owner=None, strict=True):
    """Like :func:`_gl_state_guard` but defers ``flush_deletes`` to after restore.

    ``queue_delete()`` is safe inside the guard (it only appends to a list),
    but ``flush_deletes()`` issues ``glDeleteTextures``/``glDeleteFramebuffers``
    which can free IDs that appear in the cheap_snapshot taken at guard entry.
    If those IDs are freed *before* ``restore()`` runs, the restore rebinds
    freed texture names and raises ``GL_INVALID_OPERATION`` (1282).

    This wrapper calls ``flush_deletes()`` in the ``finally`` block, ensuring
    it always runs *after* the guard's restore, even on ``return``/``raise``.
    """
    try:
        with _gl_state_guard(gl_state, owner=owner, strict=strict):
            yield
    finally:
        if owner is not None:
            owner.flush_deletes()


class RenderPassesMixin(RendererBase):
    """Render-pass, feedback, transition and presentation methods.

    These methods own the per-frame render graph: feedback, waves,
    compositing and preset transitions. The mixin is driven by
    ``MilkdropRenderer`` with a current GL context.
    """

    def _render_shapes(self) -> None:
        """Render custom MilkDrop shapes.

        Shapes are tessellated on CPU from compiled ShapeProgram parameters
        (updated by per-frame equations) and drawn via the shared
        wave/shape program (position+color, no texture transform for now).
        Supports filled polygon/circle, border, additive blend, instances.
        """
        if not getattr(self, "preset_vm", None):
            return
        vm = self.preset_vm
        program = getattr(vm, "program", None)
        shapes = getattr(program, "shapes", None) or []
        if not shapes:
            return
        if not self._ensure_wave_shape_resources():
            return
        shape_program = int(getattr(self, "_wave_shape_program", 0) or 0)
        shape_vbo = int(getattr(self, "shape_vbos", [0])[0] if getattr(self, "shape_vbos", None) else 0)
        shape_vao = int(getattr(self, "shape_vaos", [0])[0] if getattr(self, "shape_vaos", None) else 0)
        if not (shape_program and GL.glIsProgram(shape_program) and shape_vbo and shape_vao):
            return

        multipass = getattr(getattr(self, "_render_graph", None), "multipass", None)
        if multipass is None:
            return
        mp = multipass
        width = getattr(mp, "width", 0) or 0
        height = getattr(mp, "height", 0) or 0
        if width <= 0 or height <= 0:
            return
        shape_fbo = int(mp.fbo[mp.write_fbo_idx]) if getattr(mp, "fbo", None) else 0
        if not shape_fbo:
            return

        self._gl_state.bind_draw_fbo(shape_fbo, force=True)
        self._gl_state.set_viewport(0, 0, width, height)
        self._gl_state.use_program(shape_program)
        self._gl_state.bind_vertex_array(shape_vao)
        self._gl_state.bind_array_buffer(shape_vbo)
        self._gl_state.enable_blend()
        point_size_location = GL.glGetUniformLocation(shape_program, "u_point_size")

        aspect = float(width) / float(height) if height else 1.0
        for shape in shapes:
            try:
                if not bool(getattr(shape, "enabled", True)):
                    continue
                a_fill = float(getattr(shape, "a", 0.0) or 0.0)
                a2_fill = float(getattr(shape, "a2", 0.0) or 0.0)
                border_a = float(getattr(shape, "border_a", 0.0) or 0.0)
                if a_fill <= 0.0 and a2_fill <= 0.0 and border_a <= 0.0:
                    continue
                additive = bool(getattr(shape, "additive", False))
                self._gl_state.set_blend_func(GL.GL_SRC_ALPHA, GL.GL_ONE if additive else GL.GL_ONE_MINUS_SRC_ALPHA)
                thick = bool(getattr(shape, "thick_outline", False))
                self._gl_state.set_line_width(2.0 if thick else 1.0)
                if point_size_location >= 0:
                    GL.glUniform1f(point_size_location, 1.0)

                sides = int(getattr(shape, "sides", 4) or 0)
                sides = max(sides, 0)
                circle = (sides == 0) or (sides > 64)
                segs = 64 if circle else max(3, min(int(sides), 4096))
                num_instances = min(max(int(getattr(shape, "num_instances", 1) or 1), 1), 1024)
                inst_sep = float(getattr(shape, "instance_separator", 0.0) or 0.0)
                base_x = float(getattr(shape, "x", 0.5) or 0.5)
                base_y = float(getattr(shape, "y", 0.5) or 0.5)
                base_rad = float(getattr(shape, "rad", 0.1) or 0.1)
                base_ang = float(getattr(shape, "ang", 0.0) or 0.0)
                # MilkDrop shape coords are in preset space [0,1]; radius is relative.
                if not (math.isfinite(base_x) and math.isfinite(base_y) and math.isfinite(base_rad) and math.isfinite(base_ang)):
                    continue
                base_rad = max(0.0, min(abs(base_rad), 4.0))

                r1,g1,b1 = float(getattr(shape,"r",1.0)), float(getattr(shape,"g",1.0)), float(getattr(shape,"b",1.0))
                r2,g2,b2 = float(getattr(shape,"r2",r1)), float(getattr(shape,"g2",g1)), float(getattr(shape,"b2",b1))
                for inst in range(num_instances):
                    cx = base_x
                    cy = base_y
                    ang = base_ang
                    if num_instances > 1:
                        # Approximate MilkDrop 2 instance behavior:
                        # linear displacement plus angular rotation when separator used.
                        tinst = inst / float(max(1, num_instances - 1))
                        cx += inst_sep * (tinst - 0.5) * 0.0  # reserved; presets vary
                        ang += inst_sep * inst
                    # Build vertices in normalized [0,1] then to NDC
                    verts = np.zeros((segs + 2, 6), dtype=np.float32)
                    # center uses inner color
                    verts[0, 0] = (cx * 2.0 - 1.0)
                    verts[0, 1] = (1.0 - cy * 2.0)
                    verts[0, 2:6] = (r1, g1, b1, max(0.0, min(a_fill, 1.0)))
                    rx = base_rad
                    ry = base_rad * aspect
                    for i in range(segs + 1):
                        th = ang + (2.0 * math.pi * i / float(segs))
                        px = cx + math.cos(th) * rx
                        py = cy + math.sin(th) * ry
                        verts[i + 1, 0] = px * 2.0 - 1.0
                        verts[i + 1, 1] = 1.0 - py * 2.0
                        # simple radial gradient inner->outer
                        if segs > 0:
                            mix = i / float(segs)
                        else:
                            mix = 1.0
                        rr = r1 + (r2 - r1) * mix
                        gg = g1 + (g2 - g1) * mix
                        bb = b1 + (b2 - b1) * mix
                        aa = max(0.0, min(a_fill + (a2_fill - a_fill) * mix, 1.0))
                        verts[i + 1, 2:6] = (rr, gg, bb, aa)
                    if a_fill > 0.0 or a2_fill > 0.0:
                        self._gl_state.buffer_data(GL.GL_ARRAY_BUFFER, verts.nbytes, verts, GL.GL_DYNAMIC_DRAW)
                        GL.glDrawArrays(GL.GL_TRIANGLE_FAN, 0, verts.shape[0])

                    if border_a > 0.0:
                        bverts = np.zeros((segs + 1, 6), dtype=np.float32)
                        br = float(getattr(shape,"border_r",1.0)); bg = float(getattr(shape,"border_g",1.0)); bb2 = float(getattr(shape,"border_b",1.0))
                        for i in range(segs + 1):
                            th = ang + (2.0 * math.pi * i / float(segs))
                            px = cx + math.cos(th) * rx
                            py = cy + math.sin(th) * ry
                            bverts[i, 0] = px * 2.0 - 1.0
                            bverts[i, 1] = 1.0 - py * 2.0
                            bverts[i, 2:6] = (br, bg, bb2, max(0.0, min(border_a, 1.0)))
                        self._gl_state.buffer_data(GL.GL_ARRAY_BUFFER, bverts.nbytes, bverts, GL.GL_DYNAMIC_DRAW)
                        self._gl_state.set_line_width(3.0 if thick else 1.5)
                        GL.glDrawArrays(GL.GL_LINE_STRIP, 0, bverts.shape[0])
            except Exception:
                logger.debug("Shape render failed for one shape", exc_info=True)
                continue

        self._check_draw_error("shapes")
        self._gl_state.disable_blend()
        self._gl_state.bind_vertex_array(0)


    def _ensure_feedback_black_texture(self) -> int:
        """Return a complete, initialized 1x1 black texture.

        The main visual pass samples the previous feedback frame; on the
        first frame, after a preset switch or after a resize there is no
        valid previous frame.  Binding texture 0 there would sample the
        default texture object, which is incomplete (undefined content).
        A dedicated 1x1 RGBA black texture gives the shader a
        well-defined, complete render target that reads as black
        everywhere under CLAMP_TO_EDGE.

        Created lazily on the GL-owning thread (the render callback),
        matching the screen-blit resource lifecycle.
        """
        tex = getattr(self, "_feedback_black_tex", 0)
        if tex and int(tex) > 0 and GL.glIsTexture(int(tex)):
            return int(tex)
        owner = getattr(self, "_gl_resource_owner", None)
        old_tex = int(tex) if tex else 0
        candidate = 0
        try:
            black = np.zeros((1, 1, 4), dtype=np.uint8)
            candidate = _int_gl_id(GL.glGenTextures(1))
            if candidate <= 0:
                raise RuntimeError("OpenGL returned an invalid feedback texture ID")
            # Resource-creation region: the raw texture calls below
            # must not leak binding state into the render pass, so
            # they run inside the state guard (cheap snapshot — this
            # is the lazily-created fallback on the render path).
            with _gl_state_guard(
                self._gl_state,
                owner=owner,
                strict=False,
            ):
                self._gl_state.active_texture(0)
                self._gl_state.bind_texture(candidate, unit=0)
                GL.glTexImage2D(
                    GL.GL_TEXTURE_2D,
                    0,
                    GL.GL_RGBA8,
                    1,
                    1,
                    0,
                    GL.GL_RGBA,
                    GL.GL_UNSIGNED_BYTE,
                    black,
                )
                GL.glTexParameteri(
                    GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR
                )
                GL.glTexParameteri(
                    GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR
                )
                GL.glTexParameteri(
                    GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE
                )
                GL.glTexParameteri(
                    GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE
                )
            if not GL.glIsTexture(candidate):
                raise RuntimeError(
                    f"Feedback black texture {candidate} was not created successfully"
                )
            if old_tex > 0 and owner is not None:
                owner.queue_delete("texture", old_tex)
        except Exception:
            if candidate > 0 and owner is not None:
                try:
                    owner.queue_delete("texture", candidate)
                except Exception:
                    logger.exception(
                        "Failed to queue incomplete feedback texture %d for deletion",
                        candidate,
                    )
            logger.exception("Failed to create feedback black texture")
            raise

        self._feedback_black_tex = candidate
        return candidate

    # ------------------------------------------------------------------ #
    #  Resource replacement helper                                        #
    # ------------------------------------------------------------------ #
    def replace_resource(self, kind: str, attr_name: str, new_obj) -> None:
        """Delete the old GL resource held by *attr_name* and publish *new_obj*.

        Used by the ``_ensure_*`` resource constructors to avoid the
        copy-paste pattern of "read old → queue_delete → setattr" that
        was duplicated in ``_ensure_screen_blit_resources`` and
        ``_ensure_wave_shape_resources``.

        The old resource (if any) is queued for deferred deletion through
        ``_gl_resource_owner`` and ``flush_deletes`` is **not** called
        here — the caller batches the flush after all replacements for the
        frame are queued, matching the existing ownership contract.

        Args:
            kind: Resource kind string understood by ``GLResourceOwner``
                  (e.g. ``"program"``, ``"vao"``, ``"buffer"``,
                  ``"texture"``, ``"fbo"``).
            attr_name: Instance attribute name holding the current (old)
                       resource handle (int).
            new_obj: The replacement handle to install.
        """
        owner = getattr(self, "_gl_resource_owner", None)
        old = int(getattr(self, attr_name, 0) or 0)
        if old > 0 and owner is not None:
            try:
                owner.queue_delete(kind, old)
            except (GL.GLError, TypeError, ValueError):
                logger.debug(
                    "Failed to queue replaced %s (id=%s) deletion", kind, old
                )
        setattr(self, attr_name, new_obj)

    def _ensure_screen_blit_resources(self) -> bool:
        """Ensure the screen blit program and VAO are created.

        Returns True if resources are ready, False on failure.
        """
        if (
            self._screen_blit_program
            and GL.glIsProgram(self._screen_blit_program)
            and self._screen_blit_vao
            and GL.glIsVertexArray(self._screen_blit_vao)
        ):
            return True
        prog = 0
        vao = 0
        vbo = 0
        try:
            # Compile screen blit program
            prog = _compile_gl_program(
                _SCREEN_BLIT_VERT_SRC,
                _SCREEN_BLIT_FRAG_SRC,
                name="screen_blit",
            )
            if prog <= 0:
                logger.error("Failed to compile screen blit program")
                return False
            self.replace_resource("program", "_screen_blit_program", prog)
            # Create VAO/VBO for fullscreen quad
            vao_arr = np.zeros(1, dtype=np.uint32)
            GL.glGenVertexArrays(1, vao_arr)
            vao = int(vao_arr[0])
            self.replace_resource("vao", "_screen_blit_vao", vao)
            vbo_arr = np.zeros(1, dtype=np.uint32)
            GL.glGenBuffers(1, vbo_arr)
            vbo = int(vbo_arr[0])
            self.replace_resource("buffer", "_screen_blit_vbo", vbo)
            # Setup fullscreen quad: position.xy, texcoord.xy
            vertices = np.array(
                [
                    -1.0,
                    -1.0,
                    0.0,
                    0.0,
                    1.0,
                    -1.0,
                    1.0,
                    0.0,
                    -1.0,
                    1.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ],
                dtype=np.float32,
            )
            self._gl_state.bind_vertex_array(self._screen_blit_vao)
            self._gl_state.bind_array_buffer(self._screen_blit_vbo)
            self._gl_state.buffer_data(
                GL.GL_ARRAY_BUFFER,
                vertices.nbytes,
                vertices,
                GL.GL_STATIC_DRAW,
            )
            # Position attribute (location = 0)
            GL.glEnableVertexAttribArray(0)
            GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, False, 16, ctypes.c_void_p(0))
            # Texcoord attribute (location = 1)
            GL.glEnableVertexAttribArray(1)
            GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, False, 16, ctypes.c_void_p(8))
            self._gl_state.bind_vertex_array(0)
            self._gl_state.bind_array_buffer(0)
            self._gl_resource_owner.flush_deletes()
            return True
        except Exception:
            logger.exception("Failed to create screen blit resources")
            # Clean up partial allocations
            if vao > 0:
                self._gl_resource_owner.queue_delete("vao", vao)
            if vbo > 0:
                self._gl_resource_owner.queue_delete("buffer", vbo)
            if prog > 0:
                self._gl_resource_owner.queue_delete("program", prog)
            self._screen_blit_program = 0
            self._screen_blit_vao = 0
            self._screen_blit_vbo = 0
            self._gl_resource_owner.flush_deletes()
            return False

    def _ensure_wave_shape_resources(self) -> bool:
        """Create the wave/shape shader program and shared vertex buffers.

        The wave and shape renderers use ``_wave_shape_program``, ``wave_vao``,
        ``wave_vbo``, ``shape_vaos`` and ``shape_vbos`` which are declared and
        initialised in :meth:`__init__` so the GL-resource lifecycle is explicit
        and the render path can never dereference an undefined attribute before
        the first creation.

        This method compiles a single position+colour passthrough program
        and allocates one reusable interleaved VAO/VBO per primitive kind.
        The vertex data itself is uploaded per frame from the live VM
        state, so the buffers only need to exist -- they do not need to
        be sized ahead of time.

        The wave pass and the shape pass both call this, and the frame
        pipeline calls it once more, so without a guard the liveness
        validation (five ``glIs*`` driver round-trips) ran three times per
        frame.  A successful validation is memoised for the current frame;
        :meth:`invalidate_context` clears it whenever the resources are torn
        down.
        """
        frame_id = int(getattr(self, "_current_frame_id", 0))
        if (
            bool(getattr(self, "_wave_shape_resources_valid", False))
            and int(getattr(self, "_wave_shape_resources_valid_frame", -1)) == frame_id
        ):
            return True
        try:
            if not (
                getattr(self, "_wave_shape_program", None)
                and GL.glIsProgram(int(self._wave_shape_program))
            ):
                prog = _compile_gl_program(
                    _WAVE_SHAPE_VERT_SRC,
                    _WAVE_SHAPE_FRAG_SRC,
                    name="wave_shape",
                )
                if prog <= 0:
                    logger.error("Failed to compile wave/shape program")
                    return False
                self.replace_resource("program", "_wave_shape_program", int(prog))

            if not (
                getattr(self, "wave_vao", 0)
                and GL.glIsVertexArray(int(self.wave_vao))
                and getattr(self, "wave_vbo", 0)
                and GL.glIsBuffer(int(self.wave_vbo))
            ):
                vao_arr = np.zeros(1, dtype=np.uint32)
                GL.glGenVertexArrays(1, vao_arr)
                wave_vao = int(vao_arr[0])
                vbo_arr = np.zeros(1, dtype=np.uint32)
                GL.glGenBuffers(1, vbo_arr)
                wave_vbo = int(vbo_arr[0])
                self._gl_state.bind_vertex_array(wave_vao)
                self._gl_state.bind_array_buffer(wave_vbo)
                # Interleaved: position (vec2) + colour (vec4) = 6 floats.
                self._setup_interleaved_vao_attribs(wave_vbo)
                self._gl_state.bind_vertex_array(0)
                self._gl_state.bind_array_buffer(0)
                self.replace_resource("vao", "wave_vao", wave_vao)
                self.replace_resource("buffer", "wave_vbo", wave_vbo)

            if not (
                getattr(self, "shape_vaos", None)
                and len(self.shape_vaos) > 0
                and GL.glIsVertexArray(int(self.shape_vaos[0]))
                and getattr(self, "shape_vbos", None)
                and len(self.shape_vbos) > 0
                and GL.glIsBuffer(int(self.shape_vbos[0]))
            ):
                vao = np.zeros(1, dtype=np.uint32)
                GL.glGenVertexArrays(1, vao)
                shape_vao = int(vao[0])
                vbo = np.zeros(1, dtype=np.uint32)
                GL.glGenBuffers(1, vbo)
                shape_vbo = int(vbo[0])
                self._gl_state.bind_vertex_array(shape_vao)
                self._gl_state.bind_array_buffer(shape_vbo)
                self._setup_interleaved_vao_attribs(shape_vbo)
                self._gl_state.bind_vertex_array(0)
                self._gl_state.bind_array_buffer(0)
                # Retire any previously allocated shape VAOs/VBOs before
                # replacing the lists.  These are public list attributes.
                for old_vao in getattr(self, "shape_vaos", []):
                    if old_vao and int(old_vao) > 0:
                        self._gl_resource_owner.queue_delete("vao", int(old_vao))
                for old_vbo in getattr(self, "shape_vbos", []):
                    if old_vbo and int(old_vbo) > 0:
                        self._gl_resource_owner.queue_delete("buffer", int(old_vbo))
                self.shape_vaos = [shape_vao]
                self.shape_vbos = [shape_vbo]
            self._gl_resource_owner.flush_deletes()
            self._wave_shape_resources_valid = True
            self._wave_shape_resources_valid_frame = frame_id
            return True
        except Exception:
            logger.exception("Failed to create wave/shape resources")
            self._wave_shape_resources_valid = False
            return False

    def _ensure_wave_resources(self) -> bool:
        """Delegate to _ensure_wave_shape_resources for backwards-compat naming.

        Some call-sites and tests reference ``_ensure_wave_resources``; this
        method forwards to the canonical :meth:`_ensure_wave_shape_resources`.
        """
        return self._ensure_wave_shape_resources()

    def _ensure_transition_blend_program(self) -> bool:
        """Ensure the transition blend program is compiled.

        Returns True on success, False on failure.
        """
        if self._transition_blend_program and GL.glIsProgram(
            self._transition_blend_program
        ):
            return True
        try:
            prog = _compile_gl_program(
                _SCREEN_BLIT_VERT_SRC,
                _TRANSITION_BLEND_FRAG_SRC,
                name="transition_blend",
            )
            if prog <= 0:
                logger.error("Failed to compile transition blend program")
                return False
            self._transition_blend_program = prog
            return True
        except Exception:
            logger.exception("Failed to create transition blend program")
            return False

    def _setup_transition_target_fbo(self, width: int, height: int) -> bool:
        """Create (or recreate) a dedicated FBO+texture pair for rendering
        the target preset during the RENDER_TARGET phase.

        Two ping-pong texture/FBO pairs are created:

        * ``target_fbo``/``target_tex`` — the *write* target on the current
          frame.
        * ``target_prev_fbo``/``target_prev_tex`` — the *read* source for
          ``u_prev_frame`` on the next frame, so the target preset
          accumulates feedback history instead of always sampling black.

        Returns True on success.
        """
        try:
            width, height = int(width), int(height)
        except (TypeError, ValueError, OverflowError):
            logger.error("Invalid transition target dimensions: %r x %r", width, height)
            return False
        if width <= 0 or height <= 0:
            logger.error("Transition target dimensions must be positive: %d x %d", width, height)
            return False

        with _gl_state_guard_deferred_flush(
            self._gl_state, owner=self._gl_resource_owner, strict=False
        ):
            old_fbo = getattr(self._transition_resources, "target_fbo", 0)
            old_tex = getattr(self._transition_resources, "target_tex", 0)
            old_prev_fbo = getattr(self._transition_resources, "target_prev_fbo", 0)
            old_prev_tex = getattr(self._transition_resources, "target_prev_tex", 0)
            candidate_tex = 0
            candidate_fbo = 0
            candidate_prev_tex = 0
            candidate_prev_fbo = 0
            try:
                candidate_tex = _int_gl_id(GL.glGenTextures(1))
                if candidate_tex <= 0:
                    raise RuntimeError("OpenGL returned an invalid transition target texture ID")

                self._gl_state.bind_texture(candidate_tex, unit=0)
                GL.glTexImage2D(
                    GL.GL_TEXTURE_2D,
                    0,
                    GL.GL_RGBA8,
                    width,
                    height,
                    0,
                    GL.GL_RGBA,
                    GL.GL_UNSIGNED_BYTE,
                    None,
                )
                GL.glTexParameteri(
                    GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR
                )
                GL.glTexParameteri(
                    GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR
                )
                GL.glTexParameteri(
                    GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE
                )
                GL.glTexParameteri(
                    GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE
                )

                candidate_fbo = _int_gl_id(GL.glGenFramebuffers(1))
                if candidate_fbo <= 0:
                    raise RuntimeError("OpenGL returned an invalid transition target framebuffer ID")

                self._gl_state.bind_draw_fbo(candidate_fbo, force=True)
                GL.glFramebufferTexture2D(
                    GL.GL_FRAMEBUFFER,
                    GL.GL_COLOR_ATTACHMENT0,
                    GL.GL_TEXTURE_2D,
                    candidate_tex,
                    0,
                )
                status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
                if status != GL.GL_FRAMEBUFFER_COMPLETE:
                    logger.error("Transition target FBO incomplete: status=%s", status)
                    self._gl_resource_owner.queue_delete("texture", candidate_tex)
                    self._gl_resource_owner.queue_delete("fbo", candidate_fbo)
                    # Reset the state cache since the FBO creation failed
                    self._gl_state.bind_draw_fbo(0, force=True)
                    return False

                self._gl_state.set_clear_color(0.0, 0.0, 0.0, 0.0)
                self._gl_state.clear(GL.GL_COLOR_BUFFER_BIT)

                # Create the ping-pong "previous" texture/FBO so the
                # target preset can sample its own feedback history
                # (u_prev_frame) on subsequent frames instead of always
                # sampling the solid black texture.
                candidate_prev_tex = _int_gl_id(GL.glGenTextures(1))
                if candidate_prev_tex <= 0:
                    raise RuntimeError("OpenGL returned an invalid transition previous-frame texture ID")

                self._gl_state.bind_texture(candidate_prev_tex, unit=0)
                GL.glTexImage2D(
                    GL.GL_TEXTURE_2D,
                    0,
                    GL.GL_RGBA8,
                    width,
                    height,
                    0,
                    GL.GL_RGBA,
                    GL.GL_UNSIGNED_BYTE,
                    None,
                )
                GL.glTexParameteri(
                    GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR
                )
                GL.glTexParameteri(
                    GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR
                )
                GL.glTexParameteri(
                    GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE
                )
                GL.glTexParameteri(
                    GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE
                )

                candidate_prev_fbo = _int_gl_id(GL.glGenFramebuffers(1))
                if candidate_prev_fbo <= 0:
                    raise RuntimeError("OpenGL returned an invalid transition previous-frame framebuffer ID")

                self._gl_state.bind_draw_fbo(candidate_prev_fbo, force=True)
                GL.glFramebufferTexture2D(
                    GL.GL_FRAMEBUFFER,
                    GL.GL_COLOR_ATTACHMENT0,
                    GL.GL_TEXTURE_2D,
                    candidate_prev_tex,
                    0,
                )
                status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
                if status != GL.GL_FRAMEBUFFER_COMPLETE:
                    logger.error(
                        "Transition target prev FBO incomplete: status=%s", status
                    )
                    self._gl_resource_owner.queue_delete("texture", candidate_tex)
                    self._gl_resource_owner.queue_delete("fbo", candidate_fbo)
                    self._gl_resource_owner.queue_delete("texture", candidate_prev_tex)
                    self._gl_resource_owner.queue_delete("fbo", candidate_prev_fbo)
                    # Reset the state cache since the FBO creation failed
                    self._gl_state.bind_draw_fbo(0, force=True)
                    return False

                self._gl_state.set_clear_color(0.0, 0.0, 0.0, 0.0)
                self._gl_state.clear(GL.GL_COLOR_BUFFER_BIT)
            except Exception:
                candidate_tex = _int_gl_id(candidate_tex)
                candidate_fbo = _int_gl_id(candidate_fbo)
                candidate_prev_tex = _int_gl_id(candidate_prev_tex)
                candidate_prev_fbo = _int_gl_id(candidate_prev_fbo)
                if candidate_tex > 0:
                    self._gl_resource_owner.queue_delete("texture", candidate_tex)
                if candidate_fbo > 0:
                    self._gl_resource_owner.queue_delete("fbo", candidate_fbo)
                if candidate_prev_tex > 0:
                    self._gl_resource_owner.queue_delete("texture", candidate_prev_tex)
                if candidate_prev_fbo > 0:
                    self._gl_resource_owner.queue_delete("fbo", candidate_prev_fbo)
                raise

            self._transition_resources.target_fbo = candidate_fbo
            self._transition_resources.target_tex = candidate_tex
            self._transition_resources.target_prev_fbo = candidate_prev_fbo
            self._transition_resources.target_prev_tex = candidate_prev_tex
            self._transition_resources.target_prev_valid = False
            self._transition_resources.width = width
            self._transition_resources.height = height

            if old_fbo and int(old_fbo) > 0:
                self._gl_resource_owner.queue_delete("fbo", int(old_fbo))
            if old_tex and int(old_tex) > 0:
                self._gl_resource_owner.queue_delete("texture", int(old_tex))
            if old_prev_fbo and int(old_prev_fbo) > 0:
                self._gl_resource_owner.queue_delete("fbo", int(old_prev_fbo))
            if old_prev_tex and int(old_prev_tex) > 0:
                self._gl_resource_owner.queue_delete("texture", int(old_prev_tex))
            return True

    # ------------------------------------------------------------------ #
    #  Temporary preset-state swap                                        #
    # ------------------------------------------------------------------ #
    @contextmanager
    def _temp_main_visual_state(self, preset_program, preset_vm):
        """Snapshot and restore main-visual state for a temporary switch.

        Used by :meth:`_render_target_to_texture_continuous` to render the
        *target* preset into the transition FBO without permanently clobbering
        the *source* preset's render state.  The following fields are saved
        and restored atomically:

        * ``preset_program`` / ``preset_vm``
        * ``_main_visual_program``
        * ``_main_visual_shader_source``
        * ``_main_visual_reg_refs``
        * ``_main_visual_has_per_pixel``
        * ``_main_visual_uniforms_refreshed``
        * ``_transition_target_sampler_paths``

        This replaces the fragile inline save/restore block that mutated long-
        lived attributes directly inside a ``try/finally``, which was prone
        to drift (new attributes added to the snapshot list but omitted from
        the restore) and to early-return bugs.
        """
        saved = (
            getattr(self, "preset_program", None),
            getattr(self, "preset_vm", None),
            getattr(self, "_main_visual_program", 0),
            getattr(self, "_main_visual_shader_source", ""),
            getattr(self, "_main_visual_reg_refs", set()),
            getattr(self, "_main_visual_has_per_pixel", False),
            getattr(self, "_main_visual_uniforms_refreshed", False),
            getattr(self, "_transition_target_sampler_paths", {}),
        )
        try:
            self.preset_program = preset_program
            self.preset_vm = preset_vm
            yield
        finally:
            (
                self.preset_program,
                self.preset_vm,
                self._main_visual_program,
                self._main_visual_shader_source,
                self._main_visual_reg_refs,
                self._main_visual_has_per_pixel,
                self._main_visual_uniforms_refreshed,
                self._transition_target_sampler_paths,
            ) = saved

    def _render_target_to_texture_continuous(self, width: int, height: int) -> bool:
        """Render the target preset into the transition target FBO, continuously.

        This method:
        - Does NOT retire the target shader program (it's reused each frame)
        - Uses direct-write blending (Issue 4 fix)
        - Properly handles the target program compilation once, then reuses it

        This allows the target preset's equations (zoom, rot, per-frame, waves)
        to continue evolving during the BLEND phase for proper MilkDrop-style
        transitions.
        """
        fbo = self._transition_resources.target_fbo
        if not fbo or fbo == 0:
            return False

        pending_program = getattr(self, "_transition_pending_program", None)
        pending_vm = getattr(self, "_transition_pending_vm", None)
        if pending_program is None:
            return False

        target_prog_id = 0  # resolved inside the context manager

        with self._temp_main_visual_state(pending_program, pending_vm):
            try:
                # Compile target shader if not already compiled
                target_prog_id = int(
                    self._transition_resources.target_program_id or 0
                )
                if not (target_prog_id > 0 and GL.glIsProgram(target_prog_id)):
                    prog = self._compile_main_visual_shader(pending_program)
                    if not prog or prog == 0:
                        logger.error(
                            "Failed to compile target preset main visual shader"
                        )
                        return False
                    target_prog_id = int(prog)
                    # Store the target program ID for reuse across frames
                    self._transition_resources.target_program_id = target_prog_id
                    self._transition_resources.target_program = pending_program
                    # Refresh uniform locations FOR THE TARGET PROGRAM explicitly.
                    self._refresh_main_visual_uniform_locations(
                        program_id=target_prog_id
                    )
                    # Set up transition target sampler paths for custom textures
                    try:
                        # Ensure a ProgramHandle cache entry exists for the
                        # target program; the handle is fetched again later
                        # where it is actually used for uniform uploads.
                        self._get_or_create_program_handle(target_prog_id)
                        self._transition_target_sampler_paths = {}
                        for k, v in getattr(
                            self, "_prepared_custom_texture_paths", {}
                        ).items():
                            self._transition_target_sampler_paths[k] = v
                    except Exception:  # noqa: BLE001
                        self._transition_target_sampler_paths = {}

                # Run target preset VM to update per-frame state.
                # Pass the pending/target VM and target program explicitly so
                # the entire VM + uniform upload path operates on the target
                # objects without relying on self.preset_vm / self.preset_program.
                try:
                    self._run_preset_vm(
                        vm=pending_vm, program_id=target_prog_id
                    )
                except Exception:
                    logger.exception(
                        "Target preset VM failed during transition render"
                    )

                # Render to target FBO
                self._gl_state.bind_draw_fbo(fbo, force=True)
                self._check_fbo_complete("transition target")
                self._gl_state.set_clear_color(0.0, 0.0, 0.0, 0.0)
                self._gl_state.clear(GL.GL_COLOR_BUFFER_BIT)
                self._gl_state.set_viewport(0, 0, width, height)
                self._gl_state.use_program(target_prog_id)
                # The VM path uploads these uniforms after evaluation. Retry
                # only when that upload did not complete for this frame.
                if not self._vm_uniforms_are_current(target_prog_id):
                    self._upload_vm_uniforms(
                        program_id=target_prog_id, vm=pending_vm
                    )

                # Bind previous frame texture.  On the first render of the
                # transition target there is no valid previous frame, so bind
                # the 1x1 black fallback texture instead of texture 0 (the
                # default texture object is incomplete and yields undefined
                # samples in OpenGL ES).  On subsequent frames, sample the
                # ping-pong "previous" target texture so the target preset
                # accumulates feedback history (motion trails, ``f`` decay)
                # just like a standalone MilkDrop preset.
                self._gl_state.active_texture(0)
                if getattr(self, "opengl_ready", False):
                    prev_tex = getattr(
                        self._transition_resources, "target_prev_tex", 0
                    )
                    prev_valid = getattr(
                        self._transition_resources, "target_prev_valid", False
                    )
                    if prev_tex and int(prev_tex) > 0 and prev_valid:
                        self._gl_state.bind_texture(int(prev_tex), unit=0)
                    else:
                        self._gl_state.bind_texture(
                            self._ensure_feedback_black_texture(), unit=0
                        )
                    # After successfully binding the previous frame texture,
                    # mark it as valid so subsequent frames will sample it
                    # instead of the black fallback, enabling feedback history
                    # accumulation (motion trails, ``f`` decay) during the transition.
                    if prev_tex and int(prev_tex) > 0:
                        self._transition_resources.target_prev_valid = True
                else:
                    self._gl_state.bind_texture(0, unit=0)
                handle = self._get_or_create_program_handle(target_prog_id)
                handle.upload_uniform("u_prev_frame", 0, self._gl_state)

                # Bind preset texture using custom sampler system
                tex = self._resolve_transition_target_texture()
                if tex and int(tex) > 0 and GL.glIsTexture(int(tex)):
                    self._gl_state.active_texture(1)
                    self._gl_state.bind_texture(int(tex), unit=1)
                    handle.upload_uniform("u_texture", 1, self._gl_state)
                else:
                    handle.upload_uniform("u_texture", 0, self._gl_state)

                # Use direct-write blending (Issue 4 fix) - same as main visual
                self._gl_state.disable_blend()
                self._gl_state.bind_vertex_array(self._composite_vao)
                GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
                self._check_draw_error("transition target")
                self._gl_state.bind_vertex_array(0)
                # Render the target preset's waves and shapes into the target
                # FBO so the transition shows the full visual, not just the
                # main visual shader output.  These passes are best-effort: a
                # failure must not abort the transition (the main visual pass
                # has already produced a valid target texture).
                # Wave/shape rendering uses self.preset_vm / self.preset_vm.program,
                # which _temp_main_visual_state has already swapped to the target
                # VM and program, so the correct per-frame and per-point equations
                # execute for the target preset.
                try:
                    if hasattr(self, "_render_waves"):
                        self._render_waves(fbo_override=fbo)
                except Exception:
                    logger.debug(
                        "Target wave rendering skipped during transition",
                        exc_info=True,
                    )
                # Re-bind the target FBO after wave rendering (which may have
                # rebound the GL state internally) so shapes draw into the same
                # target texture.
                self._gl_state.bind_draw_fbo(fbo, force=True)
                try:
                    if hasattr(self, "_render_shapes"):
                        self._render_shapes()
                except Exception:
                    logger.debug(
                        "Target shape rendering skipped during transition",
                        exc_info=True,
                    )
                # Restore the bound FBO to the target so the texture is complete
                # and ready for the transition blend pass.
                self._gl_state.bind_draw_fbo(fbo, force=True)

                # Ping-pong swap: the texture we just rendered into (target_tex)
                # becomes the "previous" frame for the next call, and vice-versa.
                # After the swap, the freshly rendered content lives in
                # target_prev_tex, so update to_tex accordingly so the blend
                # phase reads the latest render rather than the stale swap pair.
                transition = self._transition_resources
                (
                    transition.target_fbo,
                    transition.target_prev_fbo,
                ) = (
                    transition.target_prev_fbo,
                    transition.target_fbo,
                )
                (
                    transition.target_tex,
                    transition.target_prev_tex,
                ) = (
                    transition.target_prev_tex,
                    transition.target_tex,
                )
                transition.target_prev_valid = True

                # Pre-bind textures that won't change during the blend pass
                # to avoid redundant binds in later passes
                src_tex = getattr(self, "_current_source_tex", 0)
                if src_tex and int(src_tex) > 0:
                    self._gl_state.active_texture(0)
                    self._gl_state.bind_texture(int(src_tex), unit=0)
                else:
                    self._gl_state.bind_texture(0, unit=0)
                preset_tex = self._resolve_transition_target_texture()
                if preset_tex and int(preset_tex) > 0 and GL.glIsTexture(int(preset_tex)):
                    self._gl_state.active_texture(1)
                    self._gl_state.bind_texture(int(preset_tex), unit=1)
                else:
                    self._gl_state.bind_texture(0, unit=1)

                transition.to_tex = transition.target_prev_tex
                return True
            except Exception:
                logger.exception(
                    "Failed to render target preset to texture (continuous)"
                )
                return False

    def _resolve_transition_target_texture(self) -> int:
        """Resolve the best texture for the transition target preset.

        Uses ONLY the target preset's explicit custom-sampler declarations
        (``_transition_target_sampler_paths``).  It deliberately does NOT
        invent a texture from the global ``_available_textures_by_stem``
        fallback, because the transition target is a specific preset and must
        render with its own declared sampler.
        """
        sampler_paths = getattr(self, "_transition_target_sampler_paths", {})
        for sampler_name, tex_path in sampler_paths.items():
            if not sampler_name.startswith("sampler_"):
                continue
            if tex_path is None:
                continue
            if isinstance(tex_path, int):
                tex_id = tex_path
            else:
                tex_id = self.loaded_textures.get(tex_path, 0)
            if tex_id and int(tex_id) > 0 and GL.glIsTexture(int(tex_id)):
                return int(tex_id)
        # No explicit target-sampler resolved a valid texture — do not invent one.
        return 0

    def _retire_transition_target_program(self) -> None:
        """Retire the cached continuous transition target program, if any.

        The continuous transition path caches the target program in
        ``self._transition_resources.target_program_id`` and reuses it across BLEND frames.
        Before starting a new transition (or before tearing down transition
        resources), this cached program must be explicitly retired so there is
        exactly one owner for that GL program.

        Retiring here also marks the corresponding ProgramHandle as retired so
        stale uniform uploads are rejected even if the GL object deletion is
        deferred.
        """
        target_prog_id = int(self._transition_resources.target_program_id or 0)
        if not (target_prog_id > 0):
            return
        if not GL.glIsProgram(target_prog_id):
            self._transition_resources.target_program_id = 0
            return
        try:
            # _retire_program retires the ProgramHandle, invalidates the GL
            # state program cache and queues deletion via the resource owner,
            # keeping a single ownership path for program retirement.
            self._retire_program(target_prog_id)
            self._gl_resource_owner.flush_deletes()
        except Exception:  # noqa: BLE001
            logger.debug(
                "Failed to retire transition target program %s", target_prog_id
            )
        finally:
            self._transition_resources.target_program_id = 0

    def _start_transition(
        self, target_path: str, request_id: int | None = None
    ) -> PresetLoadResult:
        """Start a real visual transition from current preset to target.

        State machine:
        CAPTURE_SOURCE -> PREPARE_TARGET -> RENDER_TARGET -> BLEND -> COMMIT -> RETIRE_OLD

        The pending preset request is claimed under the preset request lock so
        _consume_pending_preset()/load_milk_preset() cannot race it.

        A new transition must not reuse a previous transition's cached target
        program. If one exists and is still live, retire it here so there is a
        single explicit owner for that program.

        Transition capture must use a clearly defined previous/current frame index
        (curr_fbo_idx invariant: previous frame FBO -> current frame FBO -> effects).
        """
        lock = getattr(self, "_preset_request_lock", None)
        if lock is not None:
            lock.acquire()
        try:
            if request_id is not None and request_id != getattr(self, "_preset_request_id", 0):
                return PresetLoadResult.STALE
            phase = getattr(self, "_transition_phase", None)
            if phase is not None and getattr(phase, "value", phase) != "idle":
                return PresetLoadResult.PENDING
        finally:
            if lock is not None:
                lock.release()

        if not self._ensure_screen_blit_resources():
            return PresetLoadResult.FAILED
        if not self._ensure_transition_blend_program():
            return PresetLoadResult.FAILED
        self._retire_transition_target_program()
        src_tex = getattr(self, "_current_source_tex", 0)
        if not src_tex or src_tex == 0:
            logger.debug("No valid source frame for transition, installing immediately")
            return self._install_preset_immediately(target_path, request_id=request_id)

        if lock is not None:
            lock.acquire()
        try:
            if request_id is not None and request_id != getattr(self, "_preset_request_id", 0):
                return PresetLoadResult.STALE
            phase = getattr(self, "_transition_phase", None)
            if phase is not None and getattr(phase, "value", phase) != "idle":
                return PresetLoadResult.PENDING
            # Rule #3: Capture using defined ping-pong buffer indices
            self._transition_phase = TransitionPhase.CAPTURE_SOURCE
            self._transition_state.active = False
            self._transition_state.start_time = time.monotonic()
            self._transition_state.progress = 0.0
            self._transition_state.from_valid = True
            self._transition_state.from_generation = getattr(self, "_current_frame_id", 0)
            self._transition_resources.from_tex = src_tex
            self._transition_target_path = target_path
            # Reset transition target failure tracking for the new transition.
            self._consecutive_transition_target_failures = 0
            # Snapshot the request id so commit paths can detect stale transitions.
            self._transition_request_id = (
                int(request_id)
                if request_id is not None
                else int(getattr(self, "_preset_request_id", 0) or 0)
            )
            # Claim the pending request so _consume_pending_preset() skips it.
            self._pending_preset_request = None
        finally:
            if lock is not None:
                lock.release()
        logger.info("Starting transition to: %s", target_path)
        return PresetLoadResult.PENDING

    def _tick_transition(self) -> None:
        """Advance one step of the preset transition state machine.

        Source capture, target preparation and initial rendering happen once.
        During BLEND the target is rendered every frame so its equations and
        feedback continue evolving. The visible blend clock starts only after
        target preparation and its first successful render.
        """
        phase = self._transition_phase
        if phase == TransitionPhase.IDLE:
            return

        now = time.monotonic()

        if phase == TransitionPhase.CAPTURE_SOURCE:
            # Source captured, move to target preparation
            self._transition_phase = TransitionPhase.PREPARE_TARGET
            return

        if phase == TransitionPhase.PREPARE_TARGET:
            # Parse and compile the target preset
            target_path = getattr(self, "_transition_target_path", None)
            if target_path is None:
                self._fail_transition_preserving_state(reason="no target path")
                return
            try:
                raw_preset = parse_milk_file_robust(target_path)
                preset_dict = (
                    raw_preset.to_dict() if hasattr(raw_preset, "to_dict") else {}
                )
                self._transition_pending_program = PresetProgram(
                    preset_dict=preset_dict
                )
                self._transition_pending_vm = PresetVM(self._transition_pending_program)
                self._transition_phase = TransitionPhase.RENDER_TARGET
            except Exception:
                logger.exception("Failed to prepare transition target")
                self._fail_transition_preserving_state(
                    reason="target preparation failed"
                )
            return

        if phase == TransitionPhase.RENDER_TARGET:
            # Initial render of the target preset to its dedicated texture.
            # This is also done once to validate the target before entering blend.
            if self._transition_pending_program is None:
                self._fail_transition_preserving_state(reason="no pending program")
                return
            mp = getattr(getattr(self, "_render_graph", None), "multipass", None)
            width = mp.width if mp else getattr(self, "fbo_width", 1920)
            height = mp.height if mp else getattr(self, "fbo_height", 1080)
            if not self._setup_transition_target_fbo(width, height):
                logger.error("Failed to create transition target FBO")
                self._fail_transition_preserving_state(
                    reason="target FBO creation failed"
                )
                return
            try:
                _render_ok = self._render_target_to_texture_continuous(width, height)
            except PresetBudgetExhausted as exc:
                logger.warning("Transition target render budget exhausted: %s", exc)
                _render_ok = False
            if not _render_ok:
                logger.error("Failed to render target preset to texture")
                self._fail_transition_preserving_state(reason="target render failed")
                return
            # _render_target_to_texture_continuous performs a ping-pong
            # swap of the target/prev texture pairs, so after the swap the
            # freshly rendered content lives in target_prev_tex and
            # target_prev_valid is True.  When the method is mocked (no
            # swap) we fall back to target_tex so tests remain accurate.
            if self._transition_resources.target_prev_valid:
                self._transition_resources.to_tex = (
                    self._transition_resources.target_prev_tex
                )
            else:
                self._transition_resources.to_tex = (
                    self._transition_resources.target_tex
                )
            # to_tex is also updated inside _render_target_to_texture_continuous
            # during the BLEND phase for subsequent frames.
            self._transition_state.to_valid = True
            self._transition_state.active = True
            # Start the blend clock NOW.
            self._transition_state.start_time = now
            self._transition_state.progress = 0.0
            self._transition_phase = TransitionPhase.BLEND
            logger.info(
                "Transition started: %s -> %s (duration=%.2fs)",
                getattr(self, "current_milk_file", "unknown"),
                getattr(self, "_transition_target_path", "unknown"),
                getattr(self, "_transition_duration", 0.0),
            )
            return

        if phase == TransitionPhase.BLEND:
            # Re-render the target preset every frame during blend so that
            # evolving equations (zoom, rot, per-frame, waves) continue to
            # animate during the visible transition. This is essential for
            # proper MilkDrop-style transitions.
            mp = getattr(getattr(self, "_render_graph", None), "multipass", None)
            width = mp.width if mp else getattr(self, "fbo_width", 1920)
            height = mp.height if mp else getattr(self, "fbo_height", 1080)
            if self._transition_pending_program is not None:
                target_ok = self._render_target_to_texture_continuous(
                    width, height
                )
                if not target_ok:
                    # The target failed to re-render this frame.  The blend
                    # continues with the last valid target texture (sampled by
                    # u_prev_frame or to_tex), so a one-off failure is not
                    # fatal.  But persistent failures indicate the target preset
                    # is broken and should be surfaced distinctly from a
                    # transient hiccup.
                    self._consecutive_transition_target_failures = (
                        getattr(
                            self,
                            "_consecutive_transition_target_failures", 0
                        )
                        + 1
                    )
                    if self._consecutive_transition_target_failures == 1:
                        logger.warning(
                            "Transition target failed to re-render during "
                            "BLEND (consecutive=%d); showing last valid "
                            "target frame",
                            self._consecutive_transition_target_failures
                        )
                else:
                    if self._consecutive_transition_target_failures > 0:
                        logger.info(
                            "Transition target re-rendered successfully "
                            "after %d consecutive failure(s)",
                            self._consecutive_transition_target_failures
                        )
                    self._consecutive_transition_target_failures = 0
            else:
                self._consecutive_transition_target_failures = 0
            # Update blend progress
            elapsed = max(0.0, now - self._transition_state.start_time)
            try:
                duration = float(self._transition_duration)
            except (TypeError, ValueError, OverflowError):
                duration = 0.0
            if not np.isfinite(duration) or duration <= 0.0:
                duration = 1e-6
            progress = min(elapsed / self._transition_duration, 1.0)
            self._transition_state.progress = progress
            if progress >= 0.5 and getattr(self, "_transition_logged_half", False) is False:
                self._transition_logged_half = True
                logger.info(
                    "Transition %.0f%% complete (%s -> %s)",
                    progress * 100.0,
                    getattr(self, "current_milk_file", "unknown"),
                    getattr(self, "_transition_target_path", "unknown"),
                )
            if progress >= 1.0:
                self._commit_transition_preset()
                self._transition_phase = TransitionPhase.COMMIT
                logger.info(
                    "Transition complete: %s -> %s",
                    getattr(self, "current_milk_file", "unknown"),
                    getattr(self, "_transition_target_path", "unknown"),
                )
            return

        if phase == TransitionPhase.COMMIT:
            self._transition_phase = TransitionPhase.RETIRE_OLD
            return

        if phase == TransitionPhase.RETIRE_OLD:
            self._cleanup_transition_resources()
            self._transition_state.active = False
            self._transition_state.from_valid = False
            self._transition_state.to_valid = False
            self._transition_resources.from_tex = 0
            self._transition_resources.to_tex = 0
            self._transition_phase = TransitionPhase.IDLE
            self._consecutive_transition_target_failures = 0
            self._transition_logged_half = False
            logger.info("Transition complete")
            return

    def _fail_transition_preserving_state(self, reason: str = "") -> None:
        """Handle transition failure by staying on the current preset.

        Instead of silently losing the transition mid-flight, this
        method:
        1. Preserves the current preset (remains active)
        2. Cleans up transition resources
        3. Resets the transition state machine to IDLE

        The requested target preset is dropped; a failed transition
        never installs the target behind the user's back.
        """
        logger.warning("Transition failed (%s), staying on current preset", reason)

        # Clean up transition resources first
        self._cleanup_transition_resources()

        # Reset transition state; the current preset stays active.
        self._transition_phase = TransitionPhase.IDLE
        self._transition_state.active = False
        self._transition_state.from_valid = False
        self._transition_state.to_valid = False
        self._transition_resources.from_tex = 0
        self._transition_resources.to_tex = 0
        self._transition_resources.target_program_id = 0
        self._transition_pending_program = None
        self._transition_pending_vm = None
        self._transition_pending_preset_dict = {}
        self._transition_target_path = None
        self._transition_request_id = 0
        self._consecutive_transition_target_failures = 0
        self._transition_logged_half = False

    def _commit_transition_preset(self) -> bool:
        """Commit the target preset as the active preset.

        Called at the end of the BLEND phase.  Atomically swaps:
        - preset_program / preset_vm
        - main visual shader program (compiled from target preset)
        - current_milk_file / preset_name / preset_eqs

        If anything fails here, the old preset remains intact.

        IMPORTANT: This must install the target's main visual shader, not just
        the preset_program. Without this, the renderer would have:
        preset_program = NEW, but main_visual_program = OLD shader.

        Architecture — promote, don't recompile:
        The continuous transition has already compiled the target shader into
        ``self._transition_resources.target_program_id`` and refreshed its uniform locations.
        We *promote* that program to be the active ``_main_visual_program``
        rather than compiling a second copy.  This avoids:
        1. Wasted compilation time.
        2. The risk that the second compile differs from the first.
        3. The window where ``_main_visual_program`` points to a freshly
           compiled program whose uniform locations haven't been refreshed.
        """
        # A transition is stale when a newer preset request has superseded the
        # one this transition was started for.  _start_transition() snapshots the
        # request id; if it no longer matches the renderer's current
        # _preset_request_id, a newer load has arrived and we must abandon this
        # transition rather than commit a superseded target.
        if getattr(self, "_transition_request_id", 0) != getattr(
            self, "_preset_request_id", 0
        ):
            self._cleanup_transition_resources()
            self._transition_phase = TransitionPhase.IDLE
            self._transition_state.active = False
            self._transition_state.from_valid = False
            self._transition_state.to_valid = False
            self._transition_resources.target_program_id = 0
            self._transition_pending_program = None
            self._transition_pending_vm = None
            self._transition_pending_preset_dict = {}
            self._transition_target_path = None
            self._transition_request_id = 0
            logger.info("Transition abandoned: superseded by a newer preset request")
            return False
        try:
            # Check if the transition target texture was invalidated during the
            # BLEND phase.  If the target failed to render, we should not commit
            # a broken preset — stay on the current preset instead.
            if not self._transition_state.to_valid:
                logger.warning(
                    "Transition target texture was degraded during BLEND; "
                    "staying on current preset"
                )
                self._fail_transition_preserving_state(reason="target texture degraded")
                return False

            old_main_visual = getattr(self, "_main_visual_program", 0)
            candidate_program = self._transition_pending_program
            # Snapshot the target program that was already compiled during
            # the continuous transition rendering phase.
            target_prog_id = int(self._transition_resources.target_program_id or 0)

            if candidate_program is not None and getattr(self, "opengl_ready", False):
                if target_prog_id > 0 and GL.glIsProgram(target_prog_id):
                    # Promote the already-compiled target program.
                    # Uniform locations were already refreshed during the
                    # continuous transition rendering in
                    # _render_target_to_texture_continuous.
                    pass
                else:
                    # Fallback: compile if the target program wasn't
                    # pre-compiled (e.g. first-time transition or recovery).
                    prog = self._compile_main_visual_shader(
                        candidate_program
                    )
                    if not prog or prog == 0:
                        raise RuntimeError(
                            "Failed to compile target preset shader "
                            "during transition commit"
                        )
                    target_prog_id = int(prog)
                    # Refresh locations for the program that is about to
                    # become active.  Passing program_id explicitly ensures
                    # the cache is populated for the *target* program, not
                    # the outgoing source program.
                    self._refresh_main_visual_uniform_locations(
                        program_id=target_prog_id
                    )

            # --- SWAP: atomically commit all preset-owned state ---
            # All assignments below are pure attribute writes with no GL
            # calls and no exception surface, so the renderer never observes
            # a half-swapped preset.  _main_visual_program is set to the
            # TARGET program (promoted or newly compiled) in a single step.
            self._main_visual_program = (
                target_prog_id if target_prog_id > 0 else old_main_visual
            )
            # Mark uniform locations stale so the first render frame after
            # the swap re-resolves them against the new program.  This is
            # safe because _refresh_main_visual_uniform_locations() checks
            # the current _main_visual_program value.
            self._main_visual_uniforms_refreshed = False
            self.preset_program = candidate_program
            self.preset_vm = self._transition_pending_vm
            target_path = self._transition_target_path
            self.current_milk_file = target_path
            if target_path is not None:
                self.preset_name = Path(target_path).stem
            if candidate_program is not None:
                # Load raw preset dict for preset_eqs (used for save/export).
                # A .milk file is INI-ish text, not JSON, so json.load always
                # failed here and silently cleared the equations; parse it the
                # same way the immediate path does.
                preset_eqs: dict = {}
                if target_path is not None:
                    try:
                        preset_eqs = parse_milk_file_robust(
                            target_path
                        ).to_dict()
                    except Exception:
                        logging.getLogger(__name__).debug(
                            "Could not re-read equations for %s",
                            target_path,
                            exc_info=True,
                        )
                self.preset_eqs = preset_eqs
            mp = getattr(
                getattr(self, "_render_graph", None), "multipass", None
            )
            if mp is not None:
                mp.has_valid_frame = False

            # --- RETIRE: delete old program after swap ---
            # This runs only if the swap succeeded.  A failure to retire the
            # old program must NOT roll back the swap — the new preset is
            # already valid.  The deferred delete queue will eventually free
            # the GL object even if we return here.
            if (
                old_main_visual
                and int(old_main_visual) > 0
                and GL.glIsProgram(int(old_main_visual))
                and int(old_main_visual) != target_prog_id
            ):
                try:
                    self._retire_program(int(old_main_visual))
                    self._gl_resource_owner.flush_deletes()
                except (GL.GLError, _GLPyError.Error) as exc:
                    # GL context death during retirement is non-fatal to the
                    # new preset — the old program's GL object may already
                    # be gone.  Log and continue; the renderer keeps running
                    # with the new (valid) preset.
                    logger.warning(
                        "GL error retiring old main visual program %s "
                        "during transition: %s",
                        old_main_visual,
                        exc,
                    )
            # --- CLEANUP: clear transition bookkeeping ---
            # Clear the transition target program ID so
            # _cleanup_transition_resources does not delete the program we
            # just promoted to active.
            self._transition_resources.target_program_id = 0
            logger.info(
                "Transition preset committed: %s",
                self._transition_target_path,
            )
            # The candidate program/VM are now the active preset.  Drop the
            # pending transition references so later abort/cleanup paths
            # cannot delete (or double-own) the promoted program.
            self._transition_pending_program = None
            self._transition_pending_vm = None
            return True
        except (GL.GLError, _GLPyError.Error) as exc:
            # Distinguish GL-context-dead errors from recoverable ones.
            # A context-loss or out-of-memory here means the GL state is
            # unreliable — request a full rebuild rather than silently
            # staying on the old preset with corrupted state.
            err_code = getattr(exc, "err", None)
            if err_code is not None and int(err_code) in (
                GL.GL_INVALID_OPERATION,
                GL.GL_OUT_OF_MEMORY,
            ):
                logger.error(
                    "GL context unhealthy during transition commit "
                    "(err=0x%x); requesting full reinitialization",
                    int(err_code),
                )
                self._recovery.request_gl_reinit()
            else:
                logger.exception(
                    "GL error during transition commit (recoverable)"
                )
            return False
        except Exception:
            logger.exception("Failed to commit transition preset")
            return False

    def _cleanup_transition_resources(self) -> None:
        """Delete dedicated transition render target FBO/texture/program."""
        target_fbo = getattr(self._transition_resources, "target_fbo", 0)
        target_tex = getattr(self._transition_resources, "target_tex", 0)
        target_prev_fbo = getattr(self._transition_resources, "target_prev_fbo", 0)
        target_prev_tex = getattr(self._transition_resources, "target_prev_tex", 0)
        if target_fbo and int(target_fbo) > 0:
            self._gl_resource_owner.queue_delete("fbo", int(target_fbo))
        if target_tex and int(target_tex) > 0:
            self._gl_resource_owner.queue_delete("texture", int(target_tex))
        if target_prev_fbo and int(target_prev_fbo) > 0:
            self._gl_resource_owner.queue_delete("fbo", int(target_prev_fbo))
        if target_prev_tex and int(target_prev_tex) > 0:
            self._gl_resource_owner.queue_delete("texture", int(target_prev_tex))
        self._gl_resource_owner.flush_deletes()
        self._transition_resources.target_fbo = 0
        self._transition_resources.target_tex = 0
        self._transition_resources.target_prev_fbo = 0
        self._transition_resources.target_prev_tex = 0
        self._transition_resources.target_prev_valid = False
        self._retire_transition_target_program()

    def _render_transition_blend(self, width: int, height: int) -> bool:
        """Render the transition blend between source and target.

        This is called during the BLEND phase to draw the blended result.
        Uses from_tex (old preset frame) and to_tex (new preset frame).

        Uses ProgramHandle for cached uniform locations to avoid repeated
        glGetUniformLocation calls and potential invalid uniform operations.

        Returns True only when the blended frame was drawn successfully.
        """
        if self._transition_phase != TransitionPhase.BLEND:
            return False
        screen_fbo = getattr(self, "_screen_fbo_for_frame", None)
        if screen_fbo is None:
            raise RuntimeError(
                "Transition blend requires a captured GTK presentation framebuffer"
            )
        screen_fbo = int(screen_fbo)

        prog = self._transition_blend_program
        if not prog or not GL.glIsProgram(prog):
            return False

        # Get or create ProgramHandle for cached uniform locations
        handle = self._get_or_create_transition_blend_handle(int(prog))

        # Reuse the exact framebuffer GTK supplied at the start of this frame.
        self._gl_state.bind_draw_fbo(screen_fbo, force=True)
        self._check_fbo_complete("transition blend")
        self._gl_state.set_viewport(0, 0, width, height)
        self._gl_state.use_program(int(prog))
        # Bind source texture (old preset) to unit 0
        from_tex = self._transition_resources.from_tex
        if from_tex and from_tex > 0 and GL.glIsTexture(from_tex):
            self._gl_state.active_texture(0)
            self._gl_state.bind_texture(from_tex, unit=0)
            handle.upload_uniform("u_from_texture", 0, self._gl_state)
        # Bind target texture (new preset, rendered to dedicated FBO) to unit 1.
        to_tex = self._transition_resources.to_tex
        target_tex_valid = to_tex and to_tex > 0 and GL.glIsTexture(to_tex)
        if not target_tex_valid:
            # The target texture is invalid — the target preset failed to render.
            # Mark the transition as degraded so the commit logic can decide
            # whether to proceed.  Fall back to the source texture only if it
            # is valid; otherwise leave to_tex at 0 so the blend draws nothing
            # rather than silently blending old→old.
            self._transition_state.to_valid = False
            logger.warning(
                "Transition: target texture invalid (to_tex=%s), marking transition degraded",
                to_tex,
            )
            if from_tex and from_tex > 0 and GL.glIsTexture(from_tex):
                to_tex = from_tex
            else:
                to_tex = 0
        if to_tex and to_tex > 0 and GL.glIsTexture(to_tex):
            self._gl_state.active_texture(1)
            self._gl_state.bind_texture(to_tex, unit=1)
            handle.upload_uniform("u_to_texture", 1, self._gl_state)
        # Set blend progress
        handle.upload_uniform(
            "u_progress", self._transition_state.progress, self._gl_state
        )
        # Set transition effect
        handle.upload_uniform(
            "transition_effect", self._transition_state.effect, self._gl_state
        )
        # Set pixel dissolve parameters
        handle.upload_uniform(
            "u_pixel_size", self._transition_state.pixel_size, self._gl_state
        )
        handle.upload_uniform(
            "u_pixel_shape", self._transition_state.pixel_shape, self._gl_state
        )
        # Disable blending for direct draw
        self._gl_state.disable_blend()
        # Draw fullscreen quad
        self._gl_state.bind_vertex_array(self._screen_blit_vao)
        if not self._screen_draw_allowed():
            self._gl_state.bind_vertex_array(0)
            return False
        if not self._safe_draw_to_current_fbo("transition blend"):
            self._gl_state.bind_vertex_array(0)
            return False
        self._check_draw_error("transition blend")
        self._gl_state.bind_vertex_array(0)
        return True

    def _get_or_create_transition_blend_handle(self, prog_id: int) -> ProgramHandle:
        """Get or create a ProgramHandle for the transition blend program.

        The transition blend program is compiled once by
        :meth:`_ensure_transition_blend_program` using
        ``_TRANSITION_BLEND_FRAG_SRC``; this method does not recompile it.
        It only caches the uniform locations (``u_from_texture``,
        ``u_to_texture``, ``u_progress``, ``transition_effect``) on a
        :class:`ProgramHandle` so repeated uploads avoid repeated
        ``glGetUniformLocation`` calls.

        The handle table is keyed by the context generation + program id, the
        same scheme used by :meth:`_get_or_create_program_handle`, so stale
        handles are invalidated automatically when the GL context is
        recreated.
        """
        handles = getattr(self, "_program_handles", None)
        if handles is None:
            handles = {}
            self._program_handles = handles
        generation = int(getattr(self, "_program_context_generation", 0))
        key = (generation, int(prog_id))
        handle = handles.get(key)
        if handle is not None and handle.valid():
            return handle
        handle = ProgramHandle(program_id=int(prog_id), generation=generation)
        for name in (
            "u_from_texture",
            "u_to_texture",
            "u_progress",
            "transition_effect",
        ):
            loc = GL.glGetUniformLocation(int(prog_id), name)
            if loc >= 0:
                handle.set_location(name, loc)
        handles[key] = handle
        return handle

    def _screen_draw_allowed(self) -> bool:
        """False once a draw to FBO 0 has actually failed on this driver.

        Mesa's EGL/GLX backends report ``GL_FRAMEBUFFER_UNDEFINED`` for the
        default framebuffer even though it is complete by definition and
        fully usable, so the status alone is not a skip signal.  This guard
        is only set after a real ``glDrawArrays`` to FBO 0 returns an error;
        until then every draw is attempted so a working setup presents
        normally.
        """
        if not self._gl_state._default_fb_unpresentable:
            return True
        try:
            return int(GL.glGetIntegerv(GL.GL_DRAW_FRAMEBUFFER_BINDING)) != 0
        except (GL.GLError, _GLPyError.Error, TypeError, ValueError):
            return True

    def _default_framebuffer_presentable(self) -> bool:
        """Whether the default framebuffer is presentable (can be drawn to).

        Delegates to the GL state manager's ``_default_fb_unpresentable``
        flag, which is a context-level property.  The flag is False until a
        real draw to FBO 0 fails, True after such a failure, and is reset to
        False on realize/unrealize/context-loss.

        Mesa's EGL/GLX backends report ``GL_FRAMEBUFFER_UNDEFINED`` for the
        window's default framebuffer even though it is complete by definition
        and fully usable, so the status alone is not a skip signal.  Only an
        actual ``glDrawArrays`` to FBO 0 returning
        ``GL_INVALID_FRAMEBUFFER_OPERATION`` sets the flag; until then every
        draw is attempted so a working setup presents normally.
        """
        return not self._gl_state._default_fb_unpresentable

    def _blit_texture(
        self,
        src_tex: int,
        dst_fbo: int,
        width: int,
        height: int,
        clear_dst: bool = False,
    ) -> bool:
        """Blit src_tex onto dst_fbo as a full-screen quad.

        Uses the dedicated screen blit program (``_screen_blit_program``)
        for direct texture copy.  This is architecturally separate from the
        compositor, so strict MilkDrop mode is not indirectly dependent on
        a non-strict/compositor resource.

        Uses ProgramHandle for cached uniform locations (Issue 9 fix).

        Returns early without drawing if no valid blit program is available
        - drawing a fullscreen quad with an arbitrary current GL program is
        unsafe.

        When the destination is the window's default framebuffer (FBO 0) and
        the driver reports it as undefined, the draw is attempted and the
        resulting error inspected.  Mesa's EGL/GLX backends report
        ``GL_FRAMEBUFFER_UNDEFINED`` for the default FB even though it is
        complete by definition and fully usable; pre-emptively skipping on
        that status alone would black out every frame on a perfectly working
        setup.  Only a real ``GL_INVALID_FRAMEBUFFER_OPERATION`` (or other
        GL error) is treated as a genuine skip signal.

        Args:
            src_tex: Source texture ID to blit from.
            dst_fbo: Destination framebuffer ID to blit to.
            width: Viewport width.
            height: Viewport height.
            clear_dst: If True, clear the destination FBO before drawing.
                For a full-screen opaque copy, clearing is redundant since
                the entire destination will be overwritten. In a layered
                pipeline, clearing can erase content from other passes.
                Defaults to False.

        Returns:
            True if the draw was issued successfully, False if it was
            skipped due to missing resources or a GL draw failure.
        """
        # Ensure screen blit resources are available
        if not self._ensure_screen_blit_resources():
            logger.warning(
                "_blit_texture: screen blit resources not available, skipping draw"
            )
            return False
        prog = self._screen_blit_program
        if not prog or not GL.glIsProgram(prog):
            logger.warning(
                "_blit_texture: no valid screen blit program available, skipping draw"
            )
            return False
        self._gl_state.bind_draw_fbo(dst_fbo, force=True)
        self._check_fbo_complete("screen blit destination")
        # Only clear if explicitly requested. For a full-screen opaque copy,
        # clearing is redundant and can erase content from other passes in
        # a layered pipeline.
        if clear_dst:
            self._gl_state.set_clear_color(0.0, 0.0, 0.0, 0.0)
            try:
                self._gl_state.clear(GL.GL_COLOR_BUFFER_BIT)
            except (GL.GLError, _GLPyError.Error) as exc:
                try:
                    bound = int(GL.glGetIntegerv(GL.GL_DRAW_FRAMEBUFFER_BINDING))
                except (GL.GLError, _GLPyError.Error, TypeError, ValueError):
                    bound = 0
                self._on_screen_draw_failed(exc, bound, "_blit_texture clear")
                return False
        self._gl_state.active_texture(0)
        self._gl_state.bind_texture(int(src_tex), unit=0)
        self._gl_state.set_viewport(0, 0, width, height)
        self._gl_state.use_program(prog, force=True)
        # Use ProgramHandle for cached uniform location (Issue 9 fix)
        handle = self._get_or_create_program_handle(int(prog))
        handle.upload_uniform("u_texture", 0, self._gl_state)
        # Disable blending for a direct copy
        self._gl_state.disable_blend()
        self._gl_state.bind_vertex_array(self._screen_blit_vao)
        # Attempt the draw.  Mesa's EGL/GLX backends report
        # GL_FRAMEBUFFER_UNDEFINED for the default framebuffer (FBO 0) even
        # though it is complete by definition and fully usable; the status
        # alone is not proof the draw will fail, so we issue it and inspect
        # the resulting error.  Only a genuine GL_INVALID_FRAMEBUFFER_OPERATION
        # (or another GL error) is treated as a skip signal.  PyOpenGL's
        # error checker raises GLError synchronously from glDrawArrays, so
        # the failure is caught here rather than queried afterwards.
        try:
            GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
        except (GL.GLError, _GLPyError.Error) as exc:
            self._on_screen_draw_failed(exc, int(dst_fbo), "_blit_texture")
            self._gl_state.bind_vertex_array(0)
            return False
        self._gl_state.bind_vertex_array(0)
        return True

    def _render_feedback_pass(self) -> None:
        """Accumulate current frame with decay into multipass textures.

        When the main visual pass has already run (it samples the previous frame
        with decay), the feedback accumulation is handled inside that shader.
        This pass then becomes a no-op to avoid overwriting the main visual
        output in the same buffer.

        Uses GLStateManager consistently to avoid state cache incoherence.
        """
        multipass = getattr(self, "_render_graph", None)
        if multipass is None or multipass.multipass is None:
            return
        mp = multipass.multipass
        if len(mp.fbo) < 2 or len(mp.fbo_tex) < 2:
            return
        width = mp.width
        height = mp.height
        if width <= 0 or height <= 0:
            return
        # The main visual pass writes to fbo[write_fbo_idx] and sets
        # _current_source_tex.  Feedback (decay + sample previous frame) is
        # already handled by the main visual shader, so we skip the redundant
        # copy that would overwrite the rendered result.
        main_visual_program = int(getattr(self, "_main_visual_program", 0) or 0)
        if main_visual_program > 0 and GL.glIsProgram(main_visual_program):
            return

        prev_tex = int(getattr(self, "_current_source_tex", 0) or 0)
        if prev_tex > 0 and not GL.glIsTexture(prev_tex):
            logger.warning(
                "Feedback source texture %d is invalid; preserving multipass state",
                prev_tex,
            )
            return
        # Determine which FBO texture index the previous texture corresponds to.
        # This allows us to compute the correct write index for ping-pong buffering.
        source_idx = next(
            (
                index
                for index, texture in enumerate(mp.fbo_tex)
                if prev_tex and int(texture or 0) == prev_tex
            ),
            int(mp.write_fbo_idx),
        )
        if source_idx not in (0, 1):
            raise RuntimeError(f"Invalid feedback source index: {source_idx}")
        write_idx = 1 - source_idx
        write_fbo = int(mp.fbo[write_idx] or 0)
        write_tex = int(mp.fbo_tex[write_idx] or 0)
        if write_fbo <= 0 or write_tex <= 0:
            raise RuntimeError(f"Feedback target {write_idx} is invalid")
        # Prevent sampling the texture we're about to write into.
        if prev_tex and prev_tex == write_tex:
            raise RuntimeError("Feedback pass cannot sample its draw attachment")

        self._gl_state.bind_draw_fbo(write_fbo, force=True)
        self._check_fbo_complete("feedback")
        self._gl_state.set_clear_color(0.0, 0.0, 0.0, 0.0)
        self._gl_state.clear(GL.GL_COLOR_BUFFER_BIT)
        if prev_tex > 0:
            # Use GLStateManager for texture binding to keep state cache coherent
            self._gl_state.active_texture(0)
            self._gl_state.bind_texture(prev_tex, unit=0)
            self._gl_state.enable_blend()
            self._gl_state.set_blend_func(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
            self._gl_state.set_blend_equation(GL.GL_FUNC_ADD, GL.GL_FUNC_ADD)
            self._gl_state.set_viewport(0, 0, width, height)
            if not self._blit_texture(prev_tex, write_fbo, width, height):
                logger.warning(
                    "Feedback blit failed; preserving multipass source texture %s",
                    prev_tex,
                )
                return
        mp.write_fbo_idx = write_idx
        self._current_source_tex = write_tex
        mp.has_valid_frame = True

    def _render_warp_pass(self) -> None:
        """Apply warp distortion: read from current FBO, write to other.

        Uses GLStateManager consistently for all state management.
        """
        multipass = getattr(self, "_render_graph", None)
        if multipass is None or multipass.multipass is None:
            return
        mp = multipass.multipass
        read_idx = int(getattr(mp, "write_fbo_idx", 0) or 0)
        write_idx = 1 - read_idx
        fbo = getattr(mp, "fbo", None)
        fbo_tex = getattr(mp, "fbo_tex", None)
        if not (fbo and fbo[write_idx] and fbo[write_idx] > 0):
            return
        if not (fbo_tex and fbo_tex[write_idx] and fbo_tex[write_idx] > 0):
            return
        if not (fbo and fbo[read_idx] and fbo[read_idx] > 0):
            return
        if not (fbo_tex and fbo_tex[read_idx] and fbo_tex[read_idx] > 0):
            return
        width = int(getattr(mp, "width", 0) or 0)
        height = int(getattr(mp, "height", 0) or 0)
        if width <= 0 or height <= 0:
            return
        self._gl_state.bind_draw_fbo(fbo[write_idx], force=True)
        self._check_fbo_complete("warp")
        self._gl_state.set_clear_color(0.0, 0.0, 0.0, 0.0)
        self._gl_state.clear(GL.GL_COLOR_BUFFER_BIT)
        warp_prog = getattr(self, "_last_warp_program", 0)
        if warp_prog and int(warp_prog) > 0 and GL.glIsProgram(int(warp_prog)):
            src_tex = mp.fbo_tex[read_idx]
            # Use GLStateManager for texture binding
            self._gl_state.active_texture(0)
            self._gl_state.bind_texture(int(src_tex), unit=0)
            self._gl_state.enable_blend()
            self._gl_state.set_blend_func(GL.GL_ONE, GL.GL_ONE_MINUS_SRC_ALPHA)
            self._gl_state.set_viewport(0, 0, width, height)
            self._gl_state.use_program(int(warp_prog))
            # Use ProgramHandle for cached uniform locations
            handle = self._get_or_create_program_handle(int(warp_prog))
            handle.upload_uniform("u_tex", 0, self._gl_state)
            warp_val = 0.0
            if hasattr(self, "preset_vm") and self.preset_vm is not None:
                warp_val = float(self.preset_vm.state.get("warp", 0.0))
            elif hasattr(self, "warp"):
                warp_val = float(getattr(self, "warp", 0.0))
            handle.upload_uniform("u_warp", warp_val, self._gl_state)
            bj = 0.0
            if hasattr(self, "preset_vm") and self.preset_vm is not None:
                bj = float(self.preset_vm.state.get("bass_att", 0.0))
            handle.upload_uniform("u_bj", bj, self._gl_state)
            self._gl_state.bind_vertex_array(self._composite_vao)
            GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
            self._check_draw_error("warp")
            self._gl_state.bind_vertex_array(0)
            # Commit the FBO swap so subsequent passes (waves, shapes) and the
            # presentation stage read from the buffer this pass just wrote to.
            # The blend state enabled above (GL_ONE, GL_ONE_MINUS_SRC_ALPHA) is a
            # deliberate film-feedback mode for this pass; it must be reset to
            # the default so the additive wave/shape blend func applied later is
            # the one that takes effect, not a stale leftover.
            self._gl_state.disable_blend()
            mp.write_fbo_idx = write_idx
            self._current_source_tex = mp.fbo_tex[write_idx]
            mp.has_valid_frame = True
        else:
            if not self._blit_texture(
                mp.fbo_tex[read_idx], mp.fbo[write_idx], width, height
            ):
                logger.warning("Warp fallback blit failed; retaining the previous frame")
                return
            mp.write_fbo_idx = write_idx
            self._current_source_tex = mp.fbo_tex[write_idx]
            mp.has_valid_frame = True

    def _render_waves(self, fbo_override: int = 0) -> None:
        """Render wave geometry into the target FBO.

        IMPORTANT: Do NOT clear the FBO before drawing waves.
        The main visual pass has already rendered the preset into this FBO.
        Clearing would erase the preset, leaving only waves visible.

        Two rendering paths:

        * **Custom waves** — when the preset defines ``program.waves``, each
          WaveProgram is rendered with its own parameters (mode, color, etc.).
          A wave that defines per-point equations is drawn from its own
          sample buffer (``vm.wave_buffers``), which holds the geometry the
          per-point evaluation produced; a wave without per-point equations
          draws the live audio waveform, exactly like the classic base
          waveform.
        * **Classic base waveform** — when no custom waves are defined, the
          renderer draws the MilkDrop classic waveform from ``vm.wave_x`` /
          ``vm.wave_y`` (populated from live audio by
          :meth:`_populate_waveform_data`) using the scalar ``wave_r`` /
          ``wave_g`` / ``wave_b`` / ``wave_a`` preset parameters.

        Only the sample points that hold valid geometry are drawn: a
        per-point phase that was cut short leaves the remaining points
        holding the previous frame's samples, and trailing zero-padded
        slots map to NDC (-1, -1) and would spike the line strip into the
        corner.

        Args:
            fbo_override: When non-zero, draw to this FBO instead of the
                multipass write slot.  Used by the transition target render
                path so the target preset's waves are drawn into the
                transition target texture, not the regular multipass buffer.
        """
        if not hasattr(self, "preset_vm") or not self.preset_vm:
            logger.debug("Skipping wave rendering: preset_vm is not available")
            return
        # Ensure wave/shape GL resources exist before rendering.
        if hasattr(self, "_ensure_wave_resources") and not self._ensure_wave_resources():
            logger.debug("Wave/shape GL resources unavailable; skipping wave render")
            return
        # Ensure wave/shape GL resources exist before rendering.
        if (
            getattr(self, "_render_graph", None) is not None
            and hasattr(self, "_ensure_wave_resources")
            and not self._ensure_wave_resources()
        ):
            logger.debug("Wave/shape GL resources unavailable; skipping wave render")
            return
        # Unbind any VAO carried over from previous passes so wave drawing
        # starts from a clean vertex-array state.
        self._gl_state.bind_vertex_array(0)
        vm = self.preset_vm
        program = vm.program
        multipass = getattr(self, "_render_graph", None)
        if multipass is None or multipass.multipass is None:
            return
        mp = multipass.multipass
        width = mp.width
        height = mp.height
        if width <= 0 or height <= 0:
            return
        # When fbo_override is provided (transition target path), use it as the
        # draw target; otherwise fall back to the active multipass slot.  The
        # active multipass slot can be 0 when a framebuffer failed to build;
        # binding 0 would draw the waves over the presented window.
        if fbo_override and int(fbo_override) > 0:
            wave_fbo = int(fbo_override)
        else:
            wave_fbo = int(mp.fbo[mp.write_fbo_idx]) if mp.fbo else 0
        if not (wave_fbo and wave_fbo > 0):
            return
        self._gl_state.bind_draw_fbo(wave_fbo, force=True)
        self._check_fbo_complete("waves")
        # DO NOT CLEAR HERE - the main visual pass has already rendered
        # the preset into this FBO. Clearing would erase the preset.
        # Each wave's blend mode is set per-wave below: additive
        # (GL_SRC_ALPHA, GL_ONE) when wave.additive is True, or normal
        # alpha blending (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA) when
        # False (MilkDrop bAdditive=0).

        # Populate classic waveform sample arrays from live audio so that
        # presets without custom waves still show an audio-reactive waveform.
        # A custom wave that defines no per-point equations draws the same
        # live audio waveform; waves with per-point equations use their own
        # per-wave sample buffers, so the audio data never overwrites them.
        custom_waves = [
            wave
            for wave in (getattr(program, "waves", None) or [])
            if getattr(wave, "enabled", True)
        ]
        needs_audio_wave = not custom_waves or any(
            not len(getattr(w, "per_point", []) or [])
            for w in custom_waves
        )
        if needs_audio_wave:
            valid_samples = self._populate_waveform_data(vm)
        else:
            valid_samples = 0

        wave_program = int(self._wave_shape_program or 0)
        if not wave_program or not GL.glIsProgram(wave_program):
            return
        wave_vbo = int(getattr(self, "wave_vbo", 0) or 0)
        if not wave_vbo:
            return
        wave_vao = int(getattr(self, "wave_vao", 0) or 0)
        if not wave_vao:
            return

        self._gl_state.enable_blend()
        # Blend function is set per-wave below based on wave.additive.
        self._gl_state.set_viewport(0, 0, width, height)
        self._gl_state.use_program(wave_program)
        # Default line width (per-wave overrides below via bDrawThick).
        self._gl_state.set_line_width(1.0)
        point_size_location = GL.glGetUniformLocation(wave_program, "u_point_size")
        self._gl_state.bind_vertex_array(wave_vao)
        self._gl_state.bind_array_buffer(wave_vbo)

        # The wave VAO's attribute layout was configured once when the VAO
        # was created (see _ensure_wave_shape_resources): attr 0 is a vec2
        # position at offset 0 and attr 1 is a vec4 colour at offset 8.
        # Re-asserting it every frame issued four redundant GL calls and
        # invalidated the driver's VAO caches.

        # Read classic waveform colour from VM state scalars.  These are
        # the MilkDrop wave_r / wave_g / wave_b / wave_a preset parameters.
        state = getattr(vm, "state", None)
        if state is not None:
            wave_r = float(state.get("wave_r", 0.0))
            wave_g = float(state.get("wave_g", 0.0))
            wave_b = float(state.get("wave_b", 0.0))
            wave_a = float(state.get("wave_a", 1.0))
            wave_mode = int(state.get("wave_mode", 0))
        else:
            wave_r, wave_g, wave_b = 1.0, 1.0, 1.0
            wave_a = 1.0
            wave_mode = 0

        # If all colour channels are zero the waveform would be invisible —
        # fall back to a sensible default so audio reactivity is visible.
        if wave_r == 0.0 and wave_g == 0.0 and wave_b == 0.0:
            wave_r, wave_g, wave_b = 1.0, 1.0, 1.0

        # wave_a defaults to 0.0 in the VM state; with additive blending
        # (SRC_ALPHA, ONE) alpha 0 makes the waveform invisible.  Floor to
        # 1.0 so audio-reactive classic waves are always visible.
        if wave_a <= 0.0:
            wave_a = 1.0

        if custom_waves:
            wave_buffers = getattr(vm, "wave_buffers", None)
            for wave_idx, wave in enumerate(custom_waves):
                if not getattr(wave, "enabled", True):
                    continue
                # Respect the per-wave additive flag (MilkDrop bAdditive).
                # Additive waves use GL_SRC_ALPHA, GL_ONE; non-additive
                # waves use normal alpha blending GL_SRC_ALPHA,
                # GL_ONE_MINUS_SRC_ALPHA.
                if getattr(wave, "additive", True):
                    self._gl_state.set_blend_func(GL.GL_SRC_ALPHA, GL.GL_ONE)
                else:
                    self._gl_state.set_blend_func(
                        GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA
                    )
                # MilkDrop wave flags:
                #   bUseDots  → draw GL_POINTS instead of GL_LINE_STRIP
                #   bDrawThick → use a wider line
                #   mode      → 0=line, 1=waveform, 2=dual spectrum, 3=dual
                #               waveform, 4+=circular/radial
                b_use_dots = bool(getattr(wave, "bUseDots", False))
                b_draw_thick = bool(getattr(wave, "bDrawThick", False))
                wave_mode = int(getattr(wave, "mode", 0))
                if b_use_dots:
                    primitive = GL.GL_POINTS
                else:
                    primitive = GL.GL_LINE_STRIP
                self._gl_state.set_line_width(2.0 if b_draw_thick else 1.0)
                is_point_primitive = primitive == GL.GL_POINTS
                GL.glUniform1f(
                    point_size_location,
                    2.0 if is_point_primitive and b_draw_thick else 1.0,
                )
                has_per_point = len(getattr(wave, "per_point", []) or []) > 0
                if (
                    has_per_point
                    and wave_buffers is not None
                    and wave_idx < len(wave_buffers)
                ):
                    # Per-point geometry: only the samples that hold valid
                    # geometry are drawn.  A per-point phase that was
                    # truncated (budget or wall clock) leaves the remaining
                    # samples holding the previous frame's values — or, on
                    # the first frame, zeroes — and drawing those would spike
                    # the line strip into the NDC corner.
                    samples = wave_buffers[wave_idx]
                    npts = min(len(samples.x), samples.valid_points)
                    if npts <= 0:
                        continue
                    wx = samples.x[:npts]
                    wy = samples.y[:npts]
                    wr = np.asarray(samples.r[:npts], dtype=np.float32)
                    wg = np.asarray(samples.g[:npts], dtype=np.float32)
                    wb = np.asarray(samples.b[:npts], dtype=np.float32)
                    wa = np.asarray(samples.a[:npts], dtype=np.float32)
                    # If per-point colors are all zero, fall back to wave defaults
                    if (
                        not (wr == 0.0).all()
                        or not (wg == 0.0).all()
                        or not (wb == 0.0).all()
                    ):
                        pass  # use per-point colors
                    else:
                        wr = np.full(npts, float(getattr(wave, "r", wave_r)))
                        wg = np.full(npts, float(getattr(wave, "g", wave_g)))
                        wb = np.full(npts, float(getattr(wave, "b", wave_b)))
                        wa = np.full(npts, float(getattr(wave, "a", wave_a)))
                else:
                    # A wave without per-point equations draws the live
                    # audio waveform, exactly like the classic base
                    # waveform.  Only the populated samples are drawn:
                    # trailing zero-padded slots map to NDC (-1, -1) and
                    # would spike the line strip into the corner.
                    npts = min(
                        512,
                        len(vm.wave_x),
                        len(vm.wave_y),
                        valid_samples if valid_samples > 0 else 512,
                    )
                    if npts <= 0:
                        continue
                    wx = vm.wave_x[:npts]
                    wy = vm.wave_y[:npts]
                    wr = np.full(npts, float(getattr(wave, "r", wave_r)))
                    wg = np.full(npts, float(getattr(wave, "g", wave_g)))
                    wb = np.full(npts, float(getattr(wave, "b", wave_b)))
                    wa = np.full(npts, float(getattr(wave, "a", wave_a)))
                # Floor zero alpha to ensure visibility
                wa = np.where(wa <= 0.0, wave_a, wa)
                verts = np.zeros((npts, 6), dtype=np.float32)
                verts[:, 0] = (wx.astype(np.float64) * 2.0 - 1.0).astype(np.float32)
                verts[:, 1] = (wy.astype(np.float64) * 2.0 - 1.0).astype(np.float32)
                verts[:, 2] = wr
                verts[:, 3] = wg
                verts[:, 4] = wb
                verts[:, 5] = wa
                self._gl_state.buffer_data(
                    GL.GL_ARRAY_BUFFER, verts.nbytes, verts, GL.GL_DYNAMIC_DRAW
                )
                GL.glDrawArrays(primitive, 0, npts)
                # WaveProgram mode 2 (dual spectrum) and mode 3 (dual waveform)
                # render the same geometry a second time to approximate the
                # left/right stereo channels.  Use the same primitive type so
                # that bUseDots is respected for dual modes too.
                if wave_mode in (2, 3):
                    GL.glDrawArrays(primitive, 0, npts)
        else:
            # Classic base waveform: draw from VM audio sample arrays using
            # the scalar wave_r/g/b/a preset parameters for colour.
            # Use only the samples actually populated by _populate_waveform_data;
            # trailing zero-padded slots map to NDC (-1, -1) and would spike the
            # line strip into the corner.
            # The classic MilkDrop waveform always uses additive blending.
            self._gl_state.set_blend_func(GL.GL_SRC_ALPHA, GL.GL_ONE)
            npts = (
                valid_samples
                if valid_samples > 0
                else min(512, len(vm.wave_x), len(vm.wave_y))
            )
            if npts > 0:
                wx = vm.wave_x[:npts]
                wy = vm.wave_y[:npts]
                preset_params = getattr(vm, "preset_params", {})
                try:
                    wave_scale = float(preset_params.get("fWaveScale", 1.0))
                except (TypeError, ValueError):
                    wave_scale = 1.0
                wy = 0.5 + (wy - 0.5) * wave_scale
                verts = np.zeros((npts, 6), dtype=np.float32)
                verts[:, 0] = (wx.astype(np.float64) * 2.0 - 1.0).astype(np.float32)
                verts[:, 1] = (wy.astype(np.float64) * 2.0 - 1.0).astype(np.float32)
                verts[:, 2] = wave_r
                verts[:, 3] = wave_g
                verts[:, 4] = wave_b
                verts[:, 5] = wave_a
                self._gl_state.buffer_data(
                    GL.GL_ARRAY_BUFFER, verts.nbytes, verts, GL.GL_DYNAMIC_DRAW
                )
                try:
                    base_wave_dots = bool(int(preset_params.get("bWaveDots", 0)))
                    base_wave_thick = bool(int(preset_params.get("bWaveThick", 0)))
                    base_wave_additive = bool(
                        int(preset_params.get("bAdditiveWaves", 1))
                    )
                except (TypeError, ValueError):
                    base_wave_dots = False
                    base_wave_thick = False
                    base_wave_additive = True
                self._gl_state.set_line_width(2.0 if base_wave_thick else 1.0)
                self._gl_state.set_blend_func(
                    GL.GL_SRC_ALPHA,
                    GL.GL_ONE if base_wave_additive else GL.GL_ONE_MINUS_SRC_ALPHA,
                )
                if base_wave_dots or wave_mode == 1:
                    classic_primitive = GL.GL_POINTS
                elif wave_mode == 2:
                    classic_primitive = GL.GL_LINE_LOOP
                else:
                    classic_primitive = GL.GL_LINE_STRIP
                is_point_primitive = classic_primitive == GL.GL_POINTS
                GL.glUniform1f(
                    point_size_location,
                    2.0 if is_point_primitive and base_wave_thick else 1.0,
                )
                GL.glDrawArrays(classic_primitive, 0, npts)
        # ------------------------------------------------------------------
        # Post-wave cleanup: restore blend state and VAO binding so this pass
        # does not leak blend factors or a bound VAO into the shape,
        # warp, or presentation passes that follow.  _render_waves enabled
        # blend above (with per-wave additive or normal blend func) to
        # composite waves over the main visual; it must disable it afterward
        # so subsequent passes start from a clean state.  _render_target_to_texture_continuous sets
        # the same precedent (disable_blend + bind_vertex_array(0) after draw).
        self._check_draw_error("waves")
        self._gl_state.disable_blend()
        self._gl_state.bind_vertex_array(0)

    def _compute_enhancement_uniforms(
        self, enhancement: EnhancementLayer, audio_levels: AudioLevels | dict[str, object]
    ) -> dict[str, object]:
        """Compute audio-reactive enhancement uniform values for the composite pass.

        Reads the EnhancementLayer response objects and the current audio
        levels to produce a dict of uniform-name -> value pairs suitable for
        direct upload via ProgramHandle.upload_uniform().

        When the enhancement layer is disabled (e.g. STRICT_MILKDROP mode),
        all values are returned as neutrals so no renderer-owned tuning is
        applied — the composite shader becomes a pass-through.

        State updated:
            ``self._brightness_boost_level`` — decays toward zero each frame
            and is kicked up on a beat, per BrightnessResponse.
        """
        # Neutral defaults: identity transforms in the shader.
        uniforms: dict[str, object] = {
            "u_chromatic_beat": 0.0,
            "u_brightness_boost": 0.0,
            "u_bloom_intensity": 1.0,
            "u_vignette_strength": 0.0,
            "u_vignette_smooth": 2.0,
            "u_vignette_color": (0.0, 0.0, 0.0),
            "u_lift": (0.0, 0.0, 0.0),
            "u_gamma": (1.0, 1.0, 1.0),
            "u_gain": (1.0, 1.0, 1.0),
            "u_saturation": 1.0,
            "u_contrast": 1.0,
            "u_light_tint_strength": 0.0,
            "u_audio_vignette_breath": 0.0,
        }

        if not enhancement.enabled:
            return uniforms

        brightness = enhancement.brightness
        bloom_resp = enhancement.bloom
        composite = enhancement.composite

        bass = _gl_finite(audio_levels.get("bass", 0.0))
        mid = _gl_finite(audio_levels.get("mid", 0.0))
        treb = _gl_finite(audio_levels.get("treb", 0.0))
        bass_att = _gl_finite(audio_levels.get("bass_att", 0.0))
        beat = _gl_finite(audio_levels.get("beat", 0.0))

        # --- Chromatic aberration beat component (coefficient; the shader modulates by beat_flash) ---
        uniforms["u_chromatic_beat"] = _gl_finite(composite.chromatic_beat)

        # --- Brightness boost (stateful decay + beat kick) ---
        if brightness.enabled:
            self._brightness_boost_level *= brightness.decay
            if beat > brightness.beat_threshold:
                beat_floor_boost = brightness.beat_floor - 1.0
                boost = beat_floor_boost + beat * brightness.beat_boost
                self._brightness_boost_level = max(self._brightness_boost_level, boost)
            uniforms["u_brightness_boost"] = _gl_finite(
                self._brightness_boost_level
            )
        else:
            self._brightness_boost_level *= getattr(brightness, "decay", 1.0)
            uniforms["u_brightness_boost"] = 0.0

        # --- Bloom intensity (audio-reactive) ---
        if bloom_resp.enabled:
            bloom_intensity = bloom_resp.intensity_base
            bloom_intensity += bass * bloom_resp.intensity_bass
            bloom_intensity += beat * bloom_resp.intensity_beat
            uniforms["u_bloom_intensity"] = _gl_finite(bloom_intensity)

        # --- Vignetting (audio-reactive) ---
        vignette_strength = composite.vignette_base
        vignette_strength += treb * composite.vignette_treb
        vignette_strength -= bass * composite.vignette_bass_neg
        vignette_strength = max(
            composite.vignette_min, min(composite.vignette_max, vignette_strength)
        )
        uniforms["u_vignette_strength"] = _gl_finite(vignette_strength)
        tuning = enhancement.tuning
        uniforms["u_vignette_smooth"] = _gl_finite(float(tuning.vignette_smooth))
        # Configurable vignette color (supports dark or colored vignettes).
        uniforms["u_vignette_color"] = composite.vignette_color

        # --- Audio-reactive vignette breath ---
        # Bass attack transients cause the vignette to pulse inward,
        # creating a subtle "breathing" effect synced to the kick drum.
        uniforms["u_audio_vignette_breath"] = _gl_finite(bass_att * 0.3)

        # --- Color grading: lift ---
        lift_val = composite.lift_base + mid * composite.lift_mid
        lift_val = _gl_finite(lift_val)
        uniforms["u_lift"] = (lift_val, lift_val, lift_val)

        # --- Color grading: gamma ---
        gamma_val = composite.gamma_base
        gamma_val += bass * composite.gamma_bass
        gamma_val += treb * composite.gamma_treb
        gamma_val = max(0.01, _gl_finite(gamma_val))  # guard against division by zero
        uniforms["u_gamma"] = (gamma_val, gamma_val, gamma_val)

        # --- Color grading: gain ---
        gain_val = composite.gain_base + beat * composite.gain_beat
        gain_val = _gl_finite(gain_val)
        uniforms["u_gain"] = (gain_val, gain_val, gain_val)

        # --- Saturation ---
        saturation = composite.sat_base
        saturation += mid * composite.sat_mid
        saturation += beat * composite.sat_beat
        saturation = max(0.0, min(composite.sat_max, saturation))
        uniforms["u_saturation"] = _gl_finite(saturation)

        # --- Contrast ---
        contrast = composite.contrast_base + bass * composite.contrast_bass
        contrast += beat * composite.contrast_beat
        contrast = min(composite.contrast_max, contrast)
        uniforms["u_contrast"] = _gl_finite(contrast)

        # --- Light tint strength (warm glow from bloom, configurable) ---
        uniforms["u_light_tint_strength"] = _gl_finite(
            composite.light_tint_strength
        )

        return uniforms

    def _render_composite_pass(self, width: int, height: int) -> bool:
        """Render through the compositor shader with bloom/feedback.

        Returns True when the compositor draw was issued, False when a
        required resource (program, source texture, VAO) is missing or
        the draw failed -- the caller must not mark the frame as
        presented in that case.
        """
        enhancement = getattr(self, "enhancement", EnhancementLayer(enabled=True))
        tuning = enhancement.tuning
        src_tex = getattr(self, "_current_source_tex", 0)
        if not (src_tex and GL.glIsTexture(int(src_tex))):
            self._gl_state.set_clear_color(0.0, 0.0, 0.0, 1.0)
            self._gl_state.clear(_GL_COLOR_BUFFER_BIT | _GL_DEPTH_BUFFER_BIT)
            return False
        if not enhancement.enabled or not tuning.enabled:
            return self._blit_texture(
                int(src_tex),
                int(self._gl_state.fbo_draw),
                width,
                height,
            )

        shader = getattr(self, "_composite_shader", None)
        prog = int(shader) if shader else 0
        if not prog or not GL.glIsProgram(prog):
            self._gl_state.set_clear_color(0.0, 0.0, 0.0, 1.0)
            self._gl_state.clear(_GL_COLOR_BUFFER_BIT | _GL_DEPTH_BUFFER_BIT)
            return False
        composite_vao = getattr(self, "_composite_vao", 0)
        if not (composite_vao and GL.glIsVertexArray(int(composite_vao))):
            self._gl_state.set_clear_color(0.0, 0.0, 0.0, 1.0)
            self._gl_state.clear(_GL_COLOR_BUFFER_BIT | _GL_DEPTH_BUFFER_BIT)
            return False
        self._gl_state.use_program(prog)
        self._gl_state.set_clear_color(*self.CLEAR_COLOR_BLACK)
        self._gl_state.clear(GL.GL_COLOR_BUFFER_BIT)
        # Bind source texture through the state manager.
        self._gl_state.active_texture(0)
        self._gl_state.bind_texture(int(src_tex), unit=0)
        # Upload uniforms through the composite ProgramHandle so every
        # glUniform* call is generation-validated and location-cached,
        # consistent with the main-visual and transition shaders.
        _prog = int(prog)
        handle = self._get_or_create_composite_handle(_prog)
        handle.upload_uniform("u_original", 0, self._gl_state)
        handle.upload_uniform(
            "u_resolution", (float(width), float(height)), self._gl_state
        )
        handle.upload_uniform(
            "u_time", getattr(self, "_current_time", 0.0), self._gl_state
        )
        handle.upload_uniform(
            "u_bloom_temporal_mix", float(tuning.bloom_temporal_mix), self._gl_state
        )
        handle.upload_uniform(
            "u_bloom_weight", float(tuning.bloom_weight_small), self._gl_state
        )
        handle.upload_uniform("u_ray_count", int(tuning.ray_count), self._gl_state)
        handle.upload_uniform(
            "u_light_pos",
            (float(tuning.light_pos_x), float(tuning.light_pos_y)),
            self._gl_state,
        )
        handle.upload_uniform(
            "u_ray_decay", float(tuning.ray_decay), self._gl_state
        )
        handle.upload_uniform(
            "u_ray_step_weight", float(tuning.ray_contribution), self._gl_state
        )
        handle.upload_uniform(
            "u_chromatic_aberration", float(tuning.chromatic_step), self._gl_state
        )
        handle.upload_uniform(
            "u_lens_distortion", float(tuning.lens_distortion), self._gl_state
        )

        # --- Audio-reactive EnhancementLayer uniforms ---
        audio_levels = self._get_vm_audio_levels(vm=None)

        # Upload raw audio levels so the shader can modulate effects per-frame
        # (e.g. chromatic aberration pulse, vignette breath).
        handle.upload_uniform(
            "u_audio_bass",
            _gl_finite(audio_levels.get("bass", 0.0)),
            self._gl_state,
        )
        handle.upload_uniform(
            "u_audio_bass_att",
            _gl_finite(audio_levels.get("bass_att", 0.0)),
            self._gl_state,
        )
        handle.upload_uniform(
            "u_audio_mid",
            _gl_finite(audio_levels.get("mid", 0.0)),
            self._gl_state,
        )
        handle.upload_uniform(
            "u_audio_treb",
            _gl_finite(audio_levels.get("treb", 0.0)),
            self._gl_state,
        )
        handle.upload_uniform(
            "u_audio_beat",
            _gl_finite(audio_levels.get("beat", 0.0)),
            self._gl_state,
        )
        handle.upload_uniform(
            "u_audio_beat_flash",
            _gl_finite(audio_levels.get("beat_flash", 0.0)),
            self._gl_state,
        )
        handle.upload_uniform(
            "u_audio_rms",
            _gl_finite(audio_levels.get("rms", 0.0)),
            self._gl_state,
        )
        handle.upload_uniform(
            "u_audio_peak",
            _gl_finite(audio_levels.get("peak", 0.0)),
            self._gl_state,
        )

        # Compute all audio-reactive enhancement parameters from the
        # EnhancementLayer response objects and upload them as uniforms.
        enh_uniforms = self._compute_enhancement_uniforms(
            enhancement, audio_levels
        )
        for uniform_name, value in enh_uniforms.items():
            handle.upload_uniform(uniform_name, value, self._gl_state)

        self._gl_state.bind_vertex_array(int(composite_vao))
        self._check_fbo_complete("composite")
        return self._safe_draw_to_current_fbo("composite pass")

    def _get_or_create_composite_handle(self, prog_id: int) -> ProgramHandle:
        """Get or create a ProgramHandle for the composite shader.

        Caches uniform locations for the compositor's tuning uniforms
        (u_original, u_resolution, u_time, etc.) to avoid repeated
        glGetUniformLocation calls.  Uses the same context-generation-aware
        handle table as other programs so stale locations are
        automatically invalidated on context recreation.
        """
        handle = self._get_or_create_program_handle(prog_id)
        # Pre-cache locations the first time we see this program.
        if not handle.uniform_locations:
            for name in _COMPOSITE_UNIFORM_NAMES:
                loc = GL.glGetUniformLocation(prog_id, name)
                if loc >= 0:
                    handle.set_location(name, loc)
        return handle

    def _render_composite_shader_source(self) -> str:
        """Generate the compositor shader source with tuning and audio-reactive uniforms.

        The shader applies (in order):
        1. Lens distortion (barrel/pincushion UV warping from CompositorTuning)
        2. Base texture sample (at distorted UV)
        3. Audio-reactive chromatic aberration (bass widens the fringing;
           beat transients add a sharp CA pulse modulated by beat_flash and
           audio peak for extra transient punch)
        4. Volumetric light-ray bloom (ray marching)
        5. Audio-reactive bloom intensity boost (bass/beat)
        6. Warm light tint from bloom (configurable strength, uses bloom luminance)
        6. Audio-reactive brightness boost on beat
        7. Vignetting (audio-reactive strength, configurable smooth exponent,
           configurable color, bass-attack "breath" pulse)
        8. Color grading: lift / gamma / gain (Photoshop-style control)
        9. Saturation control (mix toward luminance, audio-reactive)
        10. Contrast control (pivot around 0.5, bass + beat reactive)
        All enhancement values are computed on the CPU from the EnhancementLayer
        response objects and current audio snapshot, then uploaded as uniforms.
        The raw audio levels (bass, bass_att, mid, treb, beat, beat_flash, rms)
        are also uploaded so the shader can modulate effects per-frame.
        """
        return """#version 330 core
in vec2 fragTexCoord;
out vec4 FragColor;
uniform sampler2D u_original;
uniform vec2 u_resolution;
uniform float u_time;
uniform float u_dt;
uniform float u_fps;
uniform float u_bloom_temporal_mix;
uniform float u_bloom_weight;
uniform float u_bloom_intensity;
uniform int u_ray_count;
uniform vec2 u_light_pos;
uniform float u_chromatic_aberration;
uniform float u_chromatic_beat;
uniform float u_lens_distortion;
uniform float u_ray_decay;
uniform float u_ray_step_weight;
uniform float u_audio_bass;
uniform float u_audio_bass_att;
uniform float u_audio_mid;
uniform float u_audio_treb;
uniform float u_audio_beat;
uniform float u_audio_beat_flash;
uniform float u_audio_rms;
uniform float u_audio_peak;
uniform float u_brightness_boost;
uniform float u_vignette_strength;
uniform float u_vignette_smooth;
uniform vec3 u_vignette_color;
uniform vec3 u_lift;
uniform vec3 u_gamma;
uniform vec3 u_gain;
uniform float u_saturation;
uniform float u_contrast;
uniform float u_light_tint_strength;
uniform float u_audio_vignette_breath;
void main() {
    vec2 uv = fragTexCoord;
    vec2 direction = uv - 0.5;
    float dist = length(direction);

    // --- Lens distortion (barrel / pincushion) ---
    // Radial UV warping driven by CompositorTuning.lens_distortion.
    // A positive value creates a barrel (fisheye) effect; negative creates
    // pincushion.  The magnitude scales with squared distance so the effect
    // is strongest at screen corners and zero at centre.
    float distortion = 1.0 + u_lens_distortion * dist * dist;
    vec2 distorted_uv = 0.5 + direction * distortion;
    vec3 color = texture(u_original, distorted_uv).rgb;

    // --- Audio-reactive chromatic aberration ---
    // Bass widens the red/cyan fringing; beat transients add a sharp pulse;
    // peak (RMS-style transient) adds an extra punch on sharp attacks.
    float ca_mod = u_chromatic_aberration * (1.0 + u_audio_bass * 0.5);
    ca_mod += u_chromatic_beat * u_audio_beat_flash;
    ca_mod += u_chromatic_beat * 0.5 * u_audio_peak;
    ca_mod = min(ca_mod, u_chromatic_aberration * 3.0);
    vec2 off = direction * ca_mod;
    vec3 front = texture(u_original, distorted_uv + off).rgb;
    vec3 back = texture(u_original, distorted_uv - off).rgb;

    // --- Volumetric light-ray bloom ---
    vec3 bloom = vec3(0.0);
    bloom = (front + back) * 0.5 * u_bloom_temporal_mix;
    vec2 light_pos = u_light_pos;
    vec3 light_color = vec3(1.0, 0.9, 0.7);
    vec2 dir_to_light = light_pos - distorted_uv;
    float dist_to_light = length(dir_to_light);
    vec2 ray_dir = dir_to_light / max(dist_to_light, 0.001);
    for (int i = 0; i<int(u_ray_count); i++) {
        float t = (float(i) + 0.5) / float(u_ray_count);
        vec2 sample_uv = distorted_uv + ray_dir * t * u_ray_decay;
        bloom += texture(u_original, sample_uv).rgb * u_ray_step_weight;
    }

    // Apply audio-reactive bloom intensity (bass/beat boost the glow)
    color = color + bloom * u_bloom_weight * u_bloom_intensity;
    // Warm light tint from bloom — uses configurable strength and bloom
    // luminance (not just the red channel) so the tint is scene-independent.
    float bloom_luma = dot(bloom, vec3(0.299, 0.587, 0.114));
    color += light_color * bloom_luma * u_light_tint_strength;

    // --- Audio-reactive brightness boost (beat flash) ---
    color *= (1.0 + u_brightness_boost);

    // --- Vignetting ---
    // The vignette strength pulses inward on bass attacks (kick drum "breath").
    float vignette = 1.0 - pow(dist * (u_vignette_strength + u_audio_vignette_breath), u_vignette_smooth);
    vignette = clamp(vignette, 0.0, 1.0);
    color = mix(color * u_vignette_color, color, vignette);

    // --- Color grading: lift / gamma / gain ---
    // Lift pulls shadows toward a pivot; gamma adjusts midtones; gain scales highlights.
    color = color * (1.0 - u_lift) + u_lift * 0.5;
    color = pow(color, 1.0 / u_gamma);
    color = color * u_gain;

    // --- Saturation (mix toward luminance) ---
    float luma = dot(color, vec3(0.299, 0.587, 0.114));
    color = mix(vec3(luma), color, u_saturation);

    // --- Contrast (pivot around 0.5) ---
    color = (color - 0.5) * u_contrast + 0.5;

    color = clamp(color, 0.0, 1.0);
    FragColor = vec4(color, 1.0);
}
"""

    def _get_transition_target(self, target) -> RenderTarget:
        """Return a RenderTarget for the transition's destination."""
        target_fbo = target.fbo
        target_tex = target.texture
        width = target.width
        height = target.height
        return RenderTarget(
            fbo=target_fbo, texture=target_tex, width=width, height=height
        )

    def _abort_transition(self, reason: str = ""):
        """Cancel the current transition and restore normal rendering state.

        This is the single entry point for transition cancellation. It resets
        state flags, clears pending capture state, and marks the transition as
        inactive.  It does NOT delete transition GL resources — those are
        allocated once per framebuffer size and reused across preset switches
        (see :meth:`_setup_transition_textures`).  Resources are only freed
        during resize (via :meth:`_cleanup_fbos`) or full GL teardown.
        """
        if reason:
            logger.debug("[TRANSITION] Aborting: %s", reason)
        self._transition_resources.capture_pending = False
        self._transition_resources.capture_pending_frame_id = -1
        ts = self._transition_state
        ts.active = False
        ts.progress = 0.0
        ts.from_valid = False
        ts.to_valid = False
        ts.from_generation = -1
        ts.to_generation = -1
        ts.resource_generation = -1
        self._transition_logged_half = False

    def _abort_transition_and_reset(self, reason: str = ""):
        """Cancel the current transition and fully reset all transition state.

        Unlike _abort_transition(), this method also:
        - Retires the target program (prevents shader leaks)
        - Cleans up target FBO/texture resources
        - Clears pending transition program/VM references
        - Resets transition phase to IDLE

        Use this when a new preset request supersedes an active transition.
        """
        if reason:
            logger.debug("[TRANSITION] Full reset: %s", reason)

        # Retire the target program via the single authoritative int stored on
        # self._transition_resources.target_program_id.  Never dig into the PresetProgram
        # object for a GL name — PresetProgram has no program_id attribute and
        # doing so silently skipped retirement, leaking shaders on every
        # supersedence transition.
        self._retire_transition_target_program()

        # Retire the target VM
        self._transition_pending_vm = None
        # Clear the pending program object reference
        self._transition_pending_program = None

        # Clean up target FBO/texture resources
        self._cleanup_transition_resources()

        # Reset all transition state
        self._transition_phase = TransitionPhase.IDLE
        self._transition_target_path = None
        self._transition_resources.from_tex = 0
        self._transition_resources.to_tex = 0
        self._abort_transition(reason=reason)
        self._gl_resource_owner.flush_deletes()

    def _capture_transition_source(self, curr_idx=0):
        logger.debug("[TRANSITION] _capture_transition_source not implemented; aborting")
        self._abort_transition_preserving_state()

    def _prepare_target_preset_frame(self):
        logger.debug("[TRANSITION] _prepare_target_preset_frame not implemented; aborting")
        self._abort_transition_preserving_state()
        return PresetLoadResult.FAILED

    def _abort_transition_preserving_state(self):
        self._abort_transition_and_reset(reason="preserving state")

    def _render_target_preset_frame(self):
        pass

    def _blend_transition_frames(self, progress):
        pass

    def _revert_transition_preserving_state(self):
        self._abort_transition_and_reset(reason="reverting")

    def _retire_old_preset_resources(self):
        self._retire_transition_target_program()
