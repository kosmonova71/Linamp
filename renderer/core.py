from __future__ import annotations

# Rule #6: Atomic preset switching - never destroy the current working preset
# before the replacement has been validated. Correct sequence:
# parse new preset -> build VM state -> generate shader -> compile/link ->
# validate required resources -> commit new preset -> begin transition -> release old resources.
# On failure: new preset rejected, current preset continues, is_transitioning remains
# false, capture_pending remains false.
import ctypes
import logging
import math
import os
import random
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar, Self

import gi
import numpy as np

gi.require_version("Gdk", "4.0")
gi.require_version("Gtk", "4.0")
gi.require_version("Gst", "1.0")
from gi.repository import Gdk, Gtk  # type: ignore[reportMissingModuleSource]
from OpenGL import GL
from OpenGL import error as _GLPyError

from gl_state import (
    _PIXEL_CODE_IDENT_RE,
    _PIXEL_CODE_KNOWN_IDENTS,
    _PIXEL_LOCALS_DECLARED,
    _Q_LOCALS,
    _REG_LOCALS,
    _T_LOCALS,
    _compile_gl_program,
    _gl_state_guard,
    _int_gl_id,
)
from program_handle import (
    ProgramHandle,
)
from resources import (
    AUTO_SWITCH_MIN_INTERVAL,
    EnhancementLayer,
    FrameResult,
    MultipassResources,
    PendingPresetRequest,
    PresetLoadResult,
    RendererConfig,
    RendererMode,
    RenderGraph,
    TransitionPhase,
)

from .base import AudioLevels
from .config import MEGABUF_SIZE, PRESET_INSTALL_MAX_ATTEMPTS
from .passes import (
    RenderPassesMixin,
)
from .presets import (
    AsyncPresetCompiler,
    PresetProgram,
    PresetVM,
    parse_milk_file_robust,
)
from .shaders import (
    GLSLGenerator,
    compile_assignment_to_glsl,
)
from .textures import (
    AsyncTextureLoadMixin,
    TextureLRU,
)

logger = logging.getLogger("milkdrop_renderer")


def _monotonic() -> float:
    """Monotonic clock wrapper for beat timing and audio analysis.

    Using a single indirection keeps all timing calls grep-able and makes it
    easy to audit that no wall-clock source leaks into beat gating.
    """
    return time.monotonic()


# --- Background worker shutdown --------------------------------
def shutdown_compile_pool() -> None:
    """Shut down the shared preset compile pool. Call on application exit.

    No-op placeholder: the current async preset preloader does not use a
    separate compile pool. Kept for API compatibility with callers that
    expect this function to exist.
    """


def shutdown_async_preset_compiler() -> None:
    """Stop the shared async preset preloader. Call on application exit.

    The preloader is intentionally implemented as a process-wide singleton
    (``AsyncPresetCompiler``); stopping it from per-renderer cleanup would
    prevent every other renderer from submitting background work. The
    application is responsible for invoking this function once at exit,
    alongside :func:`shutdown_compile_pool`.
    """
    AsyncPresetCompiler.stop()


# --- The renderer widget ---------------------------------------
class MilkdropRenderer(RenderPassesMixin, Gtk.GLArea, AsyncTextureLoadMixin):
    """OpenGL-based visualizer implementing Milkdrop-style equations and feedback."""

    # GTK.GLArea owns the presentation framebuffer during its render callback.
    # Do not assume that framebuffer is FBO 0: GTK4 may bind a window-system
    # FBO/backbuffer before invoking the callback.  ``SCREEN_FBO`` is retained
    # only as a fallback for tests/direct render calls outside GLArea::render.
    # The actual presentation framebuffer is captured at render start with
    # glGetIntegerv(GL_FRAMEBUFFER_BINDING).
    SCREEN_FBO: int = 0
    curr_fbo_idx: int = 0
    _last_beat_time: float = -float("inf")
    DEBUG_GL_VALIDATION = False  # Set True for debug GL validation or per-frame FBO/shader validation
    _current_source_tex: int = 0
    _AUDIO_LEVEL_KEYS: ClassVar[tuple[str, ...]] = (
        "bass",
        "bass_att",
        "mid",
        "mid_att",
        "treb",
        "treb_att",
        "vol",
        "vol_att",
        "beat",
        "beat_flash",
        "rms",
        "peak",
    )

    # --- Shape rendering constants (Issue 14: extracted magic numbers) ---
    SHAPE_CIRCLE_SEGMENTS: int = 64
    SHAPE_THICK_OUTLINE_WIDTH: float = 3.0
    SHAPE_THIN_OUTLINE_WIDTH: float = 1.0
    SHAPE_POLYGON_OUTLINE_WIDTH: float = 2.0
    # Renderer-defined t5/t6 scale factor for instance scaling.
    # Not part of the MilkDrop spec; applied only when t5/t6 are explicitly
    # assigned in the shape definition.
    SHAPE_T_SCALE_FACTOR: float = 0.1
    # Renderer-defined positional displacement per instance (degrees → NDC
    # scale).  Not part of the MilkDrop spec.
    SHAPE_INSTANCE_POS_SCALE: float = 0.01

    # GPU safety clamp for zoom uniforms.  This is a renderer policy limit to
    # prevent pathological values from crashing the shader; it is intentionally
    # wider than the evolutionary DNA bounds.
    GPU_SAFE_ZOOM_BOUNDS: tuple[float, float] = (0.1, 10.0)

    # --- GLSL type inference (classmethods, no instance or GL needed) ---
    @classmethod
    def _wider_glsl_type(cls, a: str, b: str) -> str:
        """Return the wider of two GLSL types (float < vec2 < vec3 < vec4)."""
        rank: dict[str, int] = GLSLGenerator._TYPE_RANK  # type: ignore[attr-defined]
        ra = rank.get(a, 0)
        rb = rank.get(b, 0)
        return a if ra >= rb else b

    @classmethod
    def _base_glsl_type(cls, name: str) -> str:
        """Base GLSL type for a MilkDrop variable name (default float)."""
        name_lower = name.lower()
        if name_lower in GLSLGenerator._PIXEL_SHADER_VAR_TYPES:  # type: ignore[attr-defined]
            return GLSLGenerator._PIXEL_SHADER_VAR_TYPES[name_lower]  # type: ignore[attr-defined]
        if name_lower in GLSLGenerator.KNOWN_VECTOR_TYPES:
            return GLSLGenerator.KNOWN_VECTOR_TYPES[name_lower]
        return "float"

    @classmethod
    def _trailing_swizzle(cls, expr: str) -> str:
        """Return the trailing component swizzle of an expression, if any."""
        idx = expr.rfind(".")
        if idx == -1 or idx == len(expr) - 1:
            return ""
        suffix = expr[idx + 1 :]
        if suffix and all(c in "xyzwrgbaXYZWRGBA" for c in suffix):
            return suffix
        return ""

    @classmethod
    def _infer_glsl_type(cls, name: str, expr: str) -> str:
        """Infer the GLSL type of a MilkDrop variable reference.

        ``name`` determines the base type; ``expr`` (e.g. ``"uv.xy"``) may
        carry a trailing swizzle that refines the result.  A swizzle that
        references the fourth component ('w'/'a') implies vec4; any other
        swizzle implies vec2.
        """
        base = cls._base_glsl_type(name)
        swizzle = cls._trailing_swizzle(expr)
        if not swizzle:
            return base
        low = swizzle.lower()
        if "w" in low or "a" in low:
            return "vec4"
        return "vec2"

    # --- Renderer policy constants ---
    # Clear colours used across multiple render paths.  These are renderer
    # decisions, not preset semantics.
    CLEAR_COLOR_GREY: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)

    # Audio-response smoothing factors (renderer policy, not preset).
    BEAT_SMOOTHING_FACTOR: float = 0.93
    BEAT_SMOOTHING_LEAK: float = 0.07
    BEAT_FLASH_DECAY: float = 0.82
    BEAT_FLASH_VISUAL_DECAY: float = 0.75

    # Fitness metric scaling and penalties (renderer policy).
    FITNESS_WEIGHT_LUM_STD: float = 3.0
    FITNESS_WEIGHT_COLOR_VAR: float = 5.0
    FITNESS_WEIGHT_EDGE_DENSITY: float = 5.0
    FITNESS_WEIGHT_TEMPORAL_DIFF: float = 5.0
    FITNESS_WEIGHT_ENTROPY_DIVISOR: float = 4.0
    FITNESS_BLACK_PIXEL_PENALTY: float = 0.25

    # When True (default), renderer-specific geometry scaling is applied
    # (SHAPE_T_SCALE_FACTOR, SHAPE_INSTANCE_POS_SCALE).  When False, shape
    # instancing uses pure MilkDrop semantics with no renderer-defined
    # scaling.  This flag exists to support strict MilkDrop compatibility
    # modes where renderer-specific geometry must not masquerade as authored
    # preset behaviour.
    compatibility_mode: bool = True

    _PRESET_OWNED_FIELDS: ClassVar[tuple[str, ...]] = (
        "shader_program",
        "program",
        "_last_warp_program",
        "_warp_program_ref",
        "_composite_program_ref",
        "_main_visual_program",
        "_main_visual_shader_source",
        "_main_visual_reg_refs",
        "_main_visual_has_per_pixel",
        "_main_visual_uniforms_refreshed",
        "preset_composite_program",
        "preset_program",
        "preset_vm",
        "preset_eqs",
        "current_milk_file",
        "preset_name",
        "pixel_shader_body",
        "vertex_shader_body",
        "_last_loaded_raw_preset",
        "warp_shader_code",
        "mesh_vao",
        "mesh_vbo",
        "mesh_texcoord_vbo",
        "mesh_ebo",
        "mesh_index_count",
        "mesh_x",
        "mesh_y",
        "_uniforms_locked_to",
        "_uniform_cache_generation",
        "_program_context_generation",
        "_program_generations",
        "_program_handles",
        "_active_program_handle",
        "q_uniform_locs",
        "noise_tex_locs",
        "custom_sampler_locs",
        "custom_texsize_locs",
    )
    _PRESET_CACHE_FIELDS: ClassVar[tuple[str, ...]] = (
        "_shader_cache",
        "_shader_source_cache",
        "_shader_program_refcount",
    )
    _PRESET_SNAPSHOT_SENTINEL = object()

    def __init__(self, *, auto_switch: bool = True, **kwargs):
        super().__init__(**kwargs)
        self._init_renderer_state(auto_switch=auto_switch)
        self._init_async_texture_loader()
        self._connect_glarea_signals()

    @classmethod
    def _build_enhancement_layer(
        cls, config: RendererConfig, mode: RendererMode
    ) -> EnhancementLayer:
        """Build the enhancement layer for the given renderer mode.

        Strict milkdrop mode forces the enhancement layer off regardless of
        the config value.  Enhanced mode passes the config's own layer
        through unchanged.
        """
        if mode == RendererMode.STRICT_MILKDROP:
            return EnhancementLayer(enabled=False)
        return config.enhancement

    def _init_renderer_state(self, *, auto_switch: bool = True) -> None:
        self.auto_switch = bool(auto_switch)
        self.auto_switch_interval_seconds = RendererConfig().auto_switch_interval_seconds
        self._auto_switch_current_path = None
        self._auto_switch_next_at = None
        self._auto_switch_recent: list[str] = []
        self._program_handles: dict[tuple[int, int], ProgramHandle] = {}
        # GL state and resource ownership are created eagerly so that
        # on_realize -> _reinitialize_gl_resources can find them without
        # special-casing.  The GLStateManager is intentionally created here
        # (not in _reinitialize_gl_resources) because it is a pure-Python
        # cache object with no GL dependencies; it only becomes meaningful
        # once the GL context is current.
        from gl_state import GLResourceOwner, GLStateManager
        self._gl_resource_owner = GLResourceOwner(
            context_is_current=self._is_gl_context_current
        )
        self.loaded_textures = TextureLRU(owner=self._gl_resource_owner)
        self._gl_state = GLStateManager(self._gl_resource_owner)
        # Transition state machine
        try:
            from resources import TransitionPhase, TransitionResources, TransitionState
            self._transition_phase = TransitionPhase.IDLE
            self._transition_state = TransitionState()
            self._transition_resources = TransitionResources()
            self._transition_duration = 1.0
        except Exception:  # noqa: BLE001, S110
            pass
        # Ensure preset attributes exist (some tests create detached instances)
        if not hasattr(self, "preset_program"):
            self.preset_program: PresetProgram | None = None
        if not hasattr(self, "preset_vm"):
            self.preset_vm: PresetVM | None = None
        # Common render state
        if not hasattr(self, "strict_milkdrop_mode"):
            self.strict_milkdrop_mode = True
        if not hasattr(self, "enable_motion_blur"):
            self.enable_motion_blur = False
        if not hasattr(self, "enable_lens_distortion"):
            self.enable_lens_distortion = False
        if not hasattr(self, "enable_kaleidoscope"):
            self.enable_kaleidoscope = False
        if not hasattr(self, "enable_glitch"):
            self.enable_glitch = False
        try:
            from resources import EnhancementLayer
            if not hasattr(self, "enhancement"):
                self.enhancement = EnhancementLayer(enabled=False)
        except Exception:  # noqa: BLE001, S110
            pass
        if not hasattr(self, "opengl_ready"):
            self.opengl_ready = False
        if not hasattr(self, "_current_time"):
            self._current_time = 0.0
        if not hasattr(self, "_current_frame_id"):
            self._current_frame_id = 0
        if not hasattr(self, "_current_dt"):
            self._current_dt = 1.0 / 60.0
        if not hasattr(self, "_consecutive_render_failures"):
            self._consecutive_render_failures = 0
        if not hasattr(self, "_frame_failure_events"):
            self._frame_failure_events = []
        if not hasattr(self, "_current_render_phase"):
            self._current_render_phase = "idle"
        if not hasattr(self, "_latest_audio_snapshot"):
            self._latest_audio_snapshot = None
        # Rule #3: Ping-pong FBO index — tracks which frame buffer is 'previous'
        # and which is 'current'. At frame start: prev_idx = self.curr_fbo_idx,
        # curr_idx = 1 - prev_idx. Render into curr_idx, read from prev_idx,
        # then commit self.curr_fbo_idx = curr_idx only after the full pipeline
        # has completed.
        if not hasattr(self, "curr_fbo_idx"):
            self.curr_fbo_idx = 0
        # Ensure curr_fbo_idx is always a valid integer index (0 or 1)
        self.curr_fbo_idx = int(self.curr_fbo_idx) % 2

        # Debug FBO validation — check that all renderer-owned FBOs are complete.
        # Enable with DEBUG_GL_VALIDATION = True in the class declaration.
        if getattr(self, "DEBUG_GL_VALIDATION", False) and hasattr(self, "_gl_resource_owner"):
            try:
                gl_state = getattr(self, "_gl_state", None)
                owner = self._gl_resource_owner
                for obj_type, ids in getattr(owner, "_debug_fbo_ids", {}).items():
                    for fid in ids:
                        if fid > 0:
                            if gl_state is not None:
                                gl_state.bind_framebuffer(GL.GL_FRAMEBUFFER, fid)
                            else:
                                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fid)  # gl-state-guard: inline-approved
                            status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
                            if status != GL.GL_FRAMEBUFFER_COMPLETE:
                                logger.warning(
                                    "Debug FBO validation: FBO %s (%s) is incomplete (status=%s)",
                                    obj_type,
                                    fid,
                                    status,
                                )
                # Reset to default draw FBO
                default_fbo = int(getattr(self, "SCREEN_FBO", 0) or 0)
                if gl_state is not None:
                    gl_state.bind_framebuffer(GL.GL_FRAMEBUFFER, default_fbo)
                else:
                    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, default_fbo)  # gl-state-guard: inline-approved
            except Exception:
                logger.debug("Debug FBO validation query failed", exc_info=True)

        # Rule #12: Resize handling — ensure FBOs are valid after dimensions change.
        # If multipass FBOs were recreated, validate them now.
        if hasattr(self, "_render_graph") and self._render_graph is not None:
            mp = getattr(self._render_graph, "multipass", None)
            if mp is not None:
                for i in range(2):
                    fbo_id = mp.fbo[i] if hasattr(mp, "fbo") and i < len(mp.fbo) else 0
                    if fbo_id > 0:
                        try:
                            gl_state = getattr(self, "_gl_state", None)
                            if gl_state is not None:
                                gl_state.bind_framebuffer(GL.GL_FRAMEBUFFER, fbo_id)
                            else:
                                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo_id)  # gl-state-guard: inline-approved
                            status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
                            if status != GL.GL_FRAMEBUFFER_COMPLETE:
                                logger.warning(
                                    "Resize FBO validation: multipass FBO %d incomplete (status=%s)",
                                    fbo_id,
                                    status,
                                )
                        except Exception:  # noqa: BLE001
                            logger.debug("FBO status query failed during resize validation")
                        finally:
                            # Reset to default draw FBO
                            gl_state = getattr(self, "_gl_state", None)
                            default_fbo = int(getattr(self, "SCREEN_FBO", 0) or 0)
                            if gl_state is not None:
                                gl_state.bind_framebuffer(GL.GL_FRAMEBUFFER, default_fbo)
                            else:
                                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, default_fbo)  # gl-state-guard: inline-approved
        if not hasattr(self, "_audio_snapshot_lock"):
            import threading
            self._audio_snapshot_lock = threading.RLock()
        # More attributes expected by tests
        for attr, default in [
            ("_program_context_generation", 0),
            ("_uniform_cache_generation", 0),
            ("_uniforms_locked_to", None),
            ("_vm_uniforms_upload_frame", -1),
            ("_last_frame_time", None),
            ("_frame_failure_events", []),
            ("_realize_failed", False),
            ("_gl_context", None),
            ("_gl_context_lost", False),
            ("_program_handles", {}),
        ]:
            if not hasattr(self, attr):
                self.__setattr__(attr, default)
        for attr, default in [
            ("_preset_request_lock", None),
            ("_preset_install_attempts", {}),
            ("_pending_preset_request", None),
            ("_preset_request_id", 0),
        ]:
            import threading

            if attr == "_preset_request_lock" and getattr(self, attr, None) is None:
                self.__setattr__(attr, threading.RLock())
            elif not hasattr(self, attr):
                self.__setattr__(attr, default)
        # Initialize render graph related attributes if missing
        if not hasattr(self, "_render_graph"):
            self._render_graph = None
        if not hasattr(self, "_composite_shader"):
            self._composite_shader = 0
        if not hasattr(self, "_composite_program"):
            self._composite_program = 0
        if not hasattr(self, "_current_source_tex"):
            self._current_source_tex = 0
        if not hasattr(self, "_has_valid_frame"):
            self._has_valid_frame = False
        if not hasattr(self, "_multipass_fbos"):
            self._multipass_fbos = {}
        if not hasattr(self, "_multipass_textures"):
            self._multipass_textures = {}
        if not hasattr(self, "_presented_tex_valid"):
            self._presented_tex_valid = False
        if not hasattr(self, "_last_present_success"):
            self._last_present_success = 0.0
        if not hasattr(self, "_last_present_failure"):
            self._last_present_failure = 0.0
        if not hasattr(self, "_pending_preset_request"):
            self._pending_preset_request = None
        if not hasattr(self, "_create_gl_texture_func"):
            self._create_gl_texture_func = self._create_gl_texture_default
        if not hasattr(self, "_audio_pcm_lock"):
            import threading

            self._audio_pcm_lock = threading.RLock()
        if not hasattr(self, "_recovery"):
            try:
                from .recovery import RecoveryManager
                self._recovery = RecoveryManager(self)  # type: ignore[type-assignment]
            except Exception:  # noqa: BLE001
                self._recovery = None  # type: ignore[assignment]
        # Screen blit/composite VAOs
        for attr in ["_screen_blit_program", "_screen_blit_vao", "_screen_blit_vbo", "_transition_blend_program", "_wave_shape_program", "wave_vao", "wave_vbo", "_composite_vao"]:
            if not hasattr(self, attr):
                setattr(self, attr, 0)
        if not hasattr(self, "shape_vaos"):
            self.shape_vaos = []
        if not hasattr(self, "shape_vbos"):
            self.shape_vbos = []

    @classmethod
    def create_detached(cls, *, auto_switch: bool = True) -> Self:
        """Create a renderer without constructing the GTK widget.

        ``Gtk.GLArea`` construction requires an initialized GTK display and
        aborts the process outright in headless environments, which would
        make the renderer's Python state machine untestable without a
        display.  This factory allocates the instance and runs the same
        state initialization as ``__init__`` while skipping both the
        GObject construction and the signal wiring.

        Instances created this way cannot render and must never be added to
        a widget hierarchy; they exist for logic-level testing and for
        tooling that needs the renderer's state without a GL context.
        """
        instance = cls.__new__(cls)
        instance._init_renderer_state(auto_switch=auto_switch)
        instance._init_async_texture_loader()
        return instance  # type: ignore[return-value]

    def _is_gl_context_current(self) -> bool:
        try:
            context = getattr(self, "_gl_context", None)
            return context is not None and Gdk.GLContext.get_current() is context
        except Exception:  # noqa: BLE001
            return False

    def _connect_glarea_signals(self) -> None:
        """Connect the Gtk4 GLArea lifecycle signals.

        Gtk.GLArea in the installed GTK build exposes no
        set_render_function(), so the realize/unrealize/resize/map
        signals are connected explicitly.  Rendering is implemented by
        overriding Gtk.GLArea.do_render.  The on_* handlers below are
        written with the signal-handler signature (area, context), which is
        what these signals deliver.  Without these connections the lifecycle
        handlers never run, so GL resources are not initialized or released.
        """
        if hasattr(self, "on_realize"):
            self.connect("realize", self.on_realize)
        elif hasattr(self, "do_realize"):
            try:
                # In GTK4, do_realize is a vfunc; connecting might not be needed
                pass
            except Exception:  # noqa: BLE001, S110
                pass
        if hasattr(self, "on_unrealize"):
            self.connect("unrealize", self.on_unrealize)
        if hasattr(self, "on_resize"):
            self.connect("resize", self.on_resize)
        if hasattr(self, "on_map"):
            self.connect("map", self.on_map)
        if hasattr(self, "on_unmap"):
            self.connect("unmap", self.on_unmap)

    def on_resize(self, _area: Any, width: int, height: int) -> None:
        """Defer transition teardown to the render path where GL is current."""
        phase = getattr(self, "_transition_phase", None)
        phase_value = getattr(phase, "value", phase)
        if phase_value not in (None, "idle"):
            self._pending_transition_reset_reason = "resize invalidates transition"

    def on_realize(self, *args):
        try:
            self.make_current()
            error = self.get_error()
            if error is not None:
                raise RuntimeError(f"Could not make GLArea context current: {error}")
            self._gl_context = Gdk.GLContext.get_current()
            if self._gl_context is None:
                raise RuntimeError("GLArea realized without a current GL context")
            if not self._reinitialize_gl_resources():
                raise RuntimeError("_reinitialize_gl_resources failed")
            self.queue_render()
        except Exception:
            self.opengl_ready = False
            self._realize_failed = True
            logger.exception("Failed to initialize renderer after GLArea realize")

    def on_map(self, *args, **kwargs):
        try:
            logger.debug("GLArea: Map event")
            if getattr(self, "opengl_ready", False):
                self.queue_render()
        except Exception:  # noqa: BLE001, S110
            pass

    def on_unmap(self, *args, **kwargs):
        """on_unmap: GLArea unmapped from widget hierarchy. No GL teardown."""
        try:
            logger.debug("GLArea: Unmap event")
        except Exception:  # noqa: BLE001, S110
            pass

    def _reinitialize_gl_resources(self) -> bool:
        """Rebuild GL resources after context acquisition.

        Precondition: the caller has already made the GL context current.
        This method must NOT acquire or release the GL context itself.
        """
        if getattr(self, "_realize_in_progress", False):
            return False
        try:
            gl_state = getattr(self, "_gl_state", None)
            if gl_state is None:
                # _gl_state should have been initialized in __init__ via
                # _init_renderer_state. If it's None, something went wrong
                # with initialization; return False so the caller can handle
                # the failure gracefully (on_realize sets opengl_ready=False).
                self.opengl_ready = False
                return False
            if not self._ensure_screen_blit_resources():
                return False
            if not self._ensure_wave_shape_resources():
                return False
            self._main_visual_vao = self._screen_blit_vao
            self._composite_vao = self._screen_blit_vao
            self.opengl_ready = True
            width = int(self.get_allocated_width() * self.get_scale_factor())
            height = int(self.get_allocated_height() * self.get_scale_factor())
            if width > 1 and height > 1:
                self._setup_multipass_fbos(width, height)
            return True
        except Exception:
            self.opengl_ready = False
            logger.exception("Failed to recreate renderer GL resources")
            return False

    def _setup_multipass_fbos(self, width: int, height: int) -> None:
        """Create and atomically replace the two feedback render targets."""
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid multipass dimensions: {width}x{height}")

        old_graph = getattr(self, "_render_graph", None)
        old_mp = getattr(old_graph, "multipass", None)
        if old_mp is not None and (old_mp.width, old_mp.height) == (width, height):
            return

        textures: list[int] = []
        fbos: list[int] = []
        owner = getattr(self, "_gl_resource_owner", None)
        try:
            with _gl_state_guard(
                self._gl_state, owner=owner, strict=False
            ):
                for index in range(2):
                    tex = _int_gl_id(GL.glGenTextures(1))
                    if tex <= 0:
                        return
                    textures.append(tex)
                    GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
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
                        GL.GL_TEXTURE_2D,
                        GL.GL_TEXTURE_MIN_FILTER,
                        GL.GL_LINEAR,
                    )
                    GL.glTexParameteri(
                        GL.GL_TEXTURE_2D,
                        GL.GL_TEXTURE_MAG_FILTER,
                        GL.GL_LINEAR,
                    )
                    GL.glTexParameteri(
                        GL.GL_TEXTURE_2D,
                        GL.GL_TEXTURE_WRAP_S,
                        GL.GL_CLAMP_TO_EDGE,
                    )
                    GL.glTexParameteri(
                        GL.GL_TEXTURE_2D,
                        GL.GL_TEXTURE_WRAP_T,
                        GL.GL_CLAMP_TO_EDGE,
                    )
                    fbo = _int_gl_id(GL.glGenFramebuffers(1))
                    if fbo <= 0:
                        raise RuntimeError(
                            f"OpenGL returned an invalid multipass FBO {index}"
                        )
                    fbos.append(fbo)
                    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
                    GL.glFramebufferTexture2D(
                        GL.GL_FRAMEBUFFER,
                        GL.GL_COLOR_ATTACHMENT0,
                        GL.GL_TEXTURE_2D,
                        tex,
                        0,
                    )
                    status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
                    if status != GL.GL_FRAMEBUFFER_COMPLETE:
                        raise RuntimeError(
                            f"Multipass FBO {index} incomplete: status={status}"
                        )
                    GL.glClear(GL.GL_COLOR_BUFFER_BIT)

            replacement = MultipassResources(
                fbo=fbos.copy(),
                fbo_tex=textures.copy(),
                width=width,
                height=height,
            )
        except Exception:
            if owner is not None:
                for tex in textures:
                    owner.queue_delete("texture", tex)
                for fbo in fbos:
                    owner.queue_delete("fbo", fbo)
                owner.flush_deletes()
            raise

        if old_graph is None:
            old_graph = RenderGraph()
        old_graph.multipass = replacement
        old_graph.width = width
        old_graph.height = height
        self._render_graph = old_graph
        self._multipass_fbos = fbos.copy()
        self._multipass_textures = textures.copy()
        self._current_source_tex = textures[0]
        self._has_valid_frame = False
        self._validated_fbo_ids = set()
        self.fbo_width = width
        self.fbo_height = height

        if owner is not None:
            if old_mp is not None and bool(getattr(old_mp, "owned", True)):
                for tex in old_mp.fbo_tex:
                    if tex:
                        owner.queue_delete("texture", int(tex))
                for fbo in old_mp.fbo:
                    if fbo:
                        owner.queue_delete("fbo", int(fbo))
            owner.flush_deletes()

    def _create_noise_texture(self, size: int):  # gl-state-guard: approved
        try:
            import numpy as _np
            from OpenGL import GL

            tex = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
            # Generate simple noise
            rng = _np.random.RandomState(12345)
            data = (rng.rand(size, size, 4) * 255).astype(_np.uint8)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, size, size, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, data)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
            return tex
        except Exception:  # noqa: BLE001
            return 0  # gl-state-guard: end-approved

    def _create_gl_texture_default(
        self,
        data,
        width: int,
        height: int,
        internal_format: int,
        format: int,
        *,
        texture_path: str | None = None,
    ) -> int:
        """Upload *data* to a fresh GL texture and return its ID.

        This is the default implementation backing
        ``_create_gl_texture_func``.  It is invoked on the GL-owning
        thread by the async texture loader (see
        :class:`AsyncTextureLoadMixin`) once a decoded image has been
        drained from the worker queue.  The upload is wrapped in the
        renderer's GL state guard so binding state stays coherent with
        :class:`GLStateManager`, matching the pattern used by
        :meth:`_ensure_feedback_black_texture` and
        :meth:`_setup_multipass_fbos`.

        Returns 0 on any failure so callers can treat it as a no-op.
        """
        from OpenGL import GL

        owner = getattr(self, "_gl_resource_owner", None)
        candidate = 0
        try:
            candidate = _int_gl_id(GL.glGenTextures(1))
            if candidate <= 0:
                raise RuntimeError(
                    "OpenGL returned an invalid texture ID for upload"
                )
            with _gl_state_guard(self._gl_state, owner=owner, strict=False):
                self._gl_state.active_texture(0)
                self._gl_state.bind_texture(candidate, unit=0)
                GL.glTexImage2D(
                    GL.GL_TEXTURE_2D,
                    0,
                    internal_format,
                    width,
                    height,
                    0,
                    format,
                    GL.GL_UNSIGNED_BYTE,
                    data,
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
                    f"Texture {candidate} was not created successfully"
                )
            return int(candidate)
        except Exception:
            logger.exception(
                "Failed to upload texture%s",
                f" {texture_path}" if texture_path else "",
            )
            if candidate > 0 and owner is not None:
                try:
                    owner.queue_delete("texture", candidate)
                    owner.flush_deletes()
                except Exception:  # noqa: BLE001, S110
                    pass
            return 0

    def _bind_blur_noise_samplers(
        self,
        program_id: int | None = None,
        *,
        source_tex: int | None = None,
        update_blur: bool = True,
    ) -> None:
        try:
            from OpenGL import GL  # gl-state-guard: approved

            self._ensure_noise_textures()

            if getattr(self, "_blur1_tex", 0):
                GL.glActiveTexture(GL.GL_TEXTURE1)
                GL.glBindTexture(GL.GL_TEXTURE_2D, int(self._blur1_tex))
            if getattr(self, "_blur2_tex", 0):
                GL.glActiveTexture(GL.GL_TEXTURE2)
                GL.glBindTexture(GL.GL_TEXTURE_2D, int(self._blur2_tex))
            if getattr(self, "_blur3_tex", 0):
                GL.glActiveTexture(GL.GL_TEXTURE3)
                GL.glBindTexture(GL.GL_TEXTURE_2D, int(self._blur3_tex))

            if getattr(self, "_noise_lq_tex", 0):
                GL.glActiveTexture(GL.GL_TEXTURE4)
                GL.glBindTexture(GL.GL_TEXTURE_2D, int(self._noise_lq_tex))
            if getattr(self, "_noise_mq_tex", 0):
                GL.glActiveTexture(GL.GL_TEXTURE5)
                GL.glBindTexture(GL.GL_TEXTURE_2D, int(self._noise_mq_tex))
            if getattr(self, "_noise_hq_tex", 0):
                GL.glActiveTexture(GL.GL_TEXTURE6)
                GL.glBindTexture(GL.GL_TEXTURE_2D, int(self._noise_hq_tex))

            prog = int(program_id) if program_id else 0
            if prog and GL.glIsProgram(prog):
                try:
                    loc = GL.glGetUniformLocation(prog, "u_blur1")
                    if loc >= 0:
                        GL.glUniform1i(loc, 1)
                    loc = GL.glGetUniformLocation(prog, "u_blur2")
                    if loc >= 0:
                        GL.glUniform1i(loc, 2)
                    loc = GL.glGetUniformLocation(prog, "u_blur3")
                    if loc >= 0:
                        GL.glUniform1i(loc, 3)
                    loc = GL.glGetUniformLocation(prog, "iNoiseLQ")
                    if loc >= 0:
                        GL.glUniform1i(loc, 4)
                    loc = GL.glGetUniformLocation(prog, "iNoiseMQ")
                    if loc >= 0:
                        GL.glUniform1i(loc, 5)
                    loc = GL.glGetUniformLocation(prog, "iNoiseHQ")
                    if loc >= 0:
                        GL.glUniform1i(loc, 6)
                except Exception:  # noqa: BLE001, S110
                    pass  # gl-state-guard: end-approved
            GL.glActiveTexture(GL.GL_TEXTURE0)
        except Exception:  # noqa: BLE001, S110
            pass

    def _prepare_pass_draw(self, program_id, *, source_tex=None, update_blur=True) -> bool:  # gl-state-guard: approved
        try:
            from OpenGL import GL

            prog = int(program_id) if program_id else 0
            if not prog or not GL.glIsProgram(prog):
                return False
            GL.glUseProgram(prog)
            src = int(source_tex) if source_tex is not None else int(getattr(self, "_current_source_tex", 0) or 0)
            if src:
                GL.glActiveTexture(GL.GL_TEXTURE0)
                GL.glBindTexture(GL.GL_TEXTURE_2D, src)
                try:
                    loc = GL.glGetUniformLocation(prog, "u_prev_frame")
                    if loc >= 0:
                        GL.glUniform1i(loc, 0)
                except Exception:  # noqa: BLE001, S110
                    pass
            self._bind_blur_noise_samplers(prog, source_tex=src, update_blur=update_blur)
            try:
                if hasattr(self, "_upload_vm_uniforms"):
                    vm = getattr(self, "preset_vm", None)
                    self._upload_vm_uniforms(program_id=prog, vm=vm)
            except Exception:  # noqa: BLE001, S110
                pass
            return True
        except Exception:  # noqa: BLE001
            return False  # gl-state-guard: end-approved

    def _render_warp_pass_alt(self) -> bool:  # gl-state-guard: approved
        try:
            prog = getattr(self, "_last_warp_program", 0)
            if not prog:
                return False
            from OpenGL import GL
            if not GL.glIsProgram(int(prog)):
                return False
            if not self._prepare_pass_draw(prog):
                return False
            # full screen draw
            vao = getattr(self, "_composite_vao", None) or getattr(self, "_main_visual_vao", None)
            if not vao:
                return False
            GL.glBindVertexArray(int(vao))
            GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
            GL.glBindVertexArray(0)
            return True
        except Exception:  # noqa: BLE001
            return False  # gl-state-guard: end-approved


    def _setup_interleaved_vao_attribs(self, vbo_id: int) -> None:
        """Configure the wave/shape VBO layout: vec2 position + vec4 color."""
        self._gl_state.bind_array_buffer(int(vbo_id))
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(
            0, 2, GL.GL_FLOAT, False, 6 * ctypes.sizeof(ctypes.c_float), None
        )
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(
            1,
            4,
            GL.GL_FLOAT,
            False,
            6 * ctypes.sizeof(ctypes.c_float),
            ctypes.c_void_p(2 * ctypes.sizeof(ctypes.c_float)),
        )

    def _check_fbo_complete(self, label: str = "") -> None:
        """Raise when the currently bound user framebuffer is incomplete."""
        if getattr(self, "_gl_context_lost", False):
            raise RuntimeError(
                f"Cannot validate {label or 'render target'}: GL context is lost"
            )

        draw_fbo = int(GL.glGetIntegerv(GL.GL_DRAW_FRAMEBUFFER_BINDING))
        if draw_fbo == 0:
            # Some Mesa EGL/GLX drivers report GL_FRAMEBUFFER_UNDEFINED for
            # the window-system framebuffer even when it is usable.
            return

        status = int(GL.glCheckFramebufferStatus(GL.GL_DRAW_FRAMEBUFFER) or 0)
        if status != int(GL.GL_FRAMEBUFFER_COMPLETE):
            raise RuntimeError(
                f"Framebuffer {draw_fbo} for {label or 'render target'} "
                f"is incomplete: status=0x{status:04x}"
            )

    def _check_draw_error(self, context: str = "") -> None:
        error = GL.glGetError()
        try:
            error_code = int(error or 0)
            no_error = int(GL.GL_NO_ERROR)
        except (TypeError, ValueError):
            return
        if error_code != no_error:
            # A draw to the default framebuffer (binding 0) that fails with
            # GL_INVALID_FRAMEBUFFER_OPERATION marks the default FB as
            # un-presentable so later screen draws can skip gracefully.
            bound = 0
            try:
                bound = int(GL.glGetIntegerv(GL.GL_DRAW_FRAMEBUFFER_BINDING) or 0)
            except Exception:  # noqa: BLE001, S110
                pass
            if (
                error_code == int(GL.GL_INVALID_FRAMEBUFFER_OPERATION)
                and bound == 0
                and hasattr(self, "_gl_state")
                and self._gl_state is not None
            ):
                try:
                    self._gl_state._default_fb_unpresentable = True
                except Exception:  # noqa: BLE001, S110
                    pass
                return
            raise RuntimeError(
                f"OpenGL draw failed in {context or 'render pass'}: 0x{error_code:04x}"
            )

    def _vm_uniforms_are_current(self, program_id: int) -> bool:
        return (
            int(getattr(self, "_vm_uniforms_upload_frame", -1))
            == int(getattr(self, "_current_frame_id", 0))
            and int(program_id) > 0
        )

    def _safe_draw_to_current_fbo(self, context: str = "") -> bool:
        vao = int(
            getattr(self, "_composite_vao", 0)
            or getattr(self, "_screen_blit_vao", 0)
            or 0
        )
        if vao <= 0 or not GL.glIsVertexArray(vao):
            raise RuntimeError(f"Fullscreen VAO is invalid in {context or 'draw'}")
        self._gl_state.bind_vertex_array(vao)
        try:
            GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
            self._check_draw_error(context)
            return True
        except (GL.GLError, _GLPyError.Error):
            logger.exception("Fullscreen draw failed in %s", context)
            return False
        finally:
            self._gl_state.bind_vertex_array(0)

    def on_unrealize(self, *args):
        # Stop the evolution worker before invalidating GL context so the
        # worker cannot race against GL handle teardown.
        evo = getattr(self, "evo", None)
        if evo is not None:
            try:
                evo.stop_worker()
            except Exception:  # noqa: BLE001, S110
                pass
            try:
                evo.invalidate_context("glarea_unrealized")
            except Exception:  # noqa: BLE001, S110
                pass
        try:
            recovery = getattr(self, "_recovery", None)
            if recovery is not None and hasattr(recovery, "reset"):
                try:
                    recovery.reset()
                except Exception:  # noqa: BLE001, S110
                    pass
            try:
                self._stop_texture_load_thread()
            except Exception:  # noqa: BLE001, S110
                pass
        except Exception:  # noqa: BLE001, S110
            pass
        try:
            with self._current_gl_context():
                try:
                    self._cleanup_gl_resources()
                except Exception:  # noqa: BLE001, S110
                    pass
        except Exception:
            logger.exception(
                "Could not release GL resources during GLArea unrealize; "
                "discarding stale context handles"
            )
            try:
                if hasattr(self, "_gl_resource_owner") and self._gl_resource_owner is not None:
                    self._gl_resource_owner.invalidate_context("glarea_unrealized")
            except Exception:  # noqa: BLE001, S110
                pass
            try:
                if hasattr(self, "_gl_state") and self._gl_state is not None:
                    self._gl_state.invalidate_context("glarea_unrealized")
            except Exception:  # noqa: BLE001, S110
                pass
        finally:
            self.opengl_ready = False
            self._gl_context = None
            self._gl_context_lost = True
            try:
                self._program_context_generation = int(getattr(self, "_program_context_generation", 0)) + 1
            except Exception:  # noqa: BLE001
                self._program_context_generation = 0
            try:
                self._program_handles = {}
            except Exception:  # noqa: BLE001, S110
                pass
            try:
                self.invalidate_context("unrealize")
            except Exception:  # noqa: BLE001, S110
                pass

    def do_render(self, context):
        return self.on_render(context=context)

    def on_render(self, area=None, context=None):
        if not self.get_realized() or not self.opengl_ready:
            return False

        width = int(self.get_allocated_width() * self.get_scale_factor())
        height = int(self.get_allocated_height() * self.get_scale_factor())
        if width <= 1 or height <= 1:
            return False

        result = self._render_frame(width, height)
        succeeded = result is FrameResult.SUCCESS
        return succeeded

    def _capture_presentation_fbo(self) -> int:
        """Capture the GTK presentation framebuffer for this frame.

        Per AGENTS.md rule #4, GTK's drawable framebuffer is not necessarily
        OpenGL framebuffer 0.  Query the actual draw framebuffer binding at
        the start of the render callback so every subsequent pass targets the
        real presentation surface rather than assuming FBO 0.

        Important: We do NOT unconditionally default to FBO 0 when the query
        returns 0, because some drivers report 0 for the window-system
        framebuffer even when it is the valid presentation surface.
        """
        try:
            fbo = int(GL.glGetIntegerv(GL.GL_FRAMEBUFFER_BINDING))
        except (GL.GLError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "Could not capture GTK presentation framebuffer"
            ) from exc
        self._screen_fbo_for_frame = fbo
        return fbo

    def _render_frame_impl(self, width: int, height: int) -> None:
        """Render one frame of the MilkDrop visualizer.

        This method implements the full per-frame render graph:
        - capture the GTK presentation framebuffer
        - process pending texture uploads
        - tick the transition state machine
        - render the main visual with feedback
        - render waves and shapes
        - composite and present the result

        The render graph follows the invariant:
        previous frame FBO -> current frame FBO -> effects -> transition/composite -> GTK drawable
        """
        self._current_render_phase = "setup"
        owner = getattr(self, "_gl_resource_owner", None)
        if owner is not None:
            owner.enter_render_context()
        try:
            screen_fbo = self._capture_presentation_fbo()
            self._screen_fbo_for_frame = screen_fbo
            gl_state = getattr(self, "_gl_state", None)
            if gl_state is not None:
                self._gl_state.bind_draw_fbo(screen_fbo, force=True)
            pending_transition_reset = getattr(self, "_pending_transition_reset_reason", None)
            if pending_transition_reset:
                self._pending_transition_reset_reason = None
                abort = getattr(self, "_abort_transition_and_reset", None)
                if callable(abort):
                    abort(reason=pending_transition_reset)
            if hasattr(self, "_process_pending_preset"):
                self._process_pending_preset()
            graph = getattr(self, "_render_graph", None)
            multipass = getattr(graph, "multipass", None)
            if (
                multipass is None
                or multipass.width != width
                or multipass.height != height
            ):
                self._setup_multipass_fbos(width, height)

            self._process_texture_load_queue()
            now = time.monotonic()
            self._advance_frame_clock(now)
            self._tick_transition()
            self._check_auto_preset_switch(now)
            self._current_render_phase = "preset_vm"
            self._run_preset_vm(width=width, height=height)

            main_visual_prog = int(getattr(self, "_main_visual_program", 0) or 0)
            if main_visual_prog <= 0 and self.preset_program is not None:
                main_visual_prog = self._compile_main_visual_shader(self.preset_program)
                if main_visual_prog <= 0:
                    raise RuntimeError(
                        "Could not compile the active preset's main visual shader"
                    )
                self._main_visual_program = main_visual_prog
                # Rule #7: Invalidate uniform location cache when shader changes.
                # Without this, stale location caches from the previous shader cause
                # uniforms to upload to non-existent locations, producing a black screen.
                self._main_visual_uniforms_refreshed = False
                self._refresh_main_visual_uniform_locations(main_visual_prog)
            self._current_render_phase = "main_visual"
            main_visual_active = main_visual_prog > 0 and GL.glIsProgram(main_visual_prog)
            if main_visual_prog > 0 and GL.glIsProgram(main_visual_prog):
                self._render_main_visual_pass(width, height)

            self._current_render_phase = "feedback"
            if hasattr(self, "_render_warp_pass") and not main_visual_active:
                self._render_warp_pass()
            self._render_shapes()
            self._current_render_phase = "warp"
            if hasattr(self, "_ensure_wave_resources"):
                self._ensure_wave_resources()
            self._render_waves()
            self._current_render_phase = "waves"

            self._current_render_phase = "shapes"
            self._gl_state.bind_draw_fbo(screen_fbo, force=True)
            self._check_fbo_complete("screen blit destination")
            transition_phase = getattr(self, "_transition_phase", None)
            phase = getattr(transition_phase, "value", transition_phase)
            if phase == "blend":
                self._current_render_phase = "composite"
                self._check_fbo_complete("composite")
                if not self._render_transition_blend(width, height):
                    raise RuntimeError("Transition blend/presentation pass failed")
            else:
                self._check_fbo_complete("composite")
                if not self._render_composite_pass(width, height):
                    raise RuntimeError("Final composite/presentation pass failed")
            # Rule #3: Advance ping-pong FBO index after the full render pipeline
            # has completed. previous frame -> current frame -> effects -> presentation.
            prev_idx = self.curr_fbo_idx
            curr_idx = 1 - prev_idx
            self.curr_fbo_idx = curr_idx
            self._current_render_phase = "blit"
        finally:
            if owner is not None:
                owner.exit_render_context()
        self._current_render_phase = "fitness_analysis"

    def _check_auto_preset_switch(self, now: float | None = None) -> None:
        """Queue a smooth switch when the active preset's display interval expires."""
        if not getattr(self, "auto_switch", False):
            self._auto_switch_current_path = None
            self._auto_switch_next_at = None
            return

        current_path = getattr(self, "current_milk_file", None)
        available = list(getattr(self, "available_milk_files", ()) or ())
        if not current_path or len(available) < 2:
            self._auto_switch_current_path = None
            self._auto_switch_next_at = None
            return

        now = time.monotonic() if now is None else float(now)
        try:
            interval = float(
                getattr(self, "auto_switch_interval_seconds", 30.0)
            )
        except (TypeError, ValueError):
            interval = RendererConfig().auto_switch_interval_seconds
        if not math.isfinite(interval) or interval < AUTO_SWITCH_MIN_INTERVAL:
            interval = AUTO_SWITCH_MIN_INTERVAL

        current_key = os.path.normcase(os.path.normpath(os.fspath(current_path)))
        tracked_key = getattr(self, "_auto_switch_current_path", None)
        if current_key != tracked_key:
            self._auto_switch_current_path = current_key
            self._auto_switch_next_at = now + interval
            return

        phase = getattr(self, "_transition_phase", None)
        if getattr(phase, "value", phase) not in (None, "idle"):
            # A transition phase represents work that must actually happen
            # (capture, target render, blend, commit, retirement). Never mutate
            # it here merely to unblock auto-switching: doing so can skip those
            # operations and retire resources that are still in use.
            return
        if getattr(self, "_pending_preset_request", None) is not None:
            return

        next_at = getattr(self, "_auto_switch_next_at", None)
        if next_at is None:
            self._auto_switch_next_at = now + interval
            return
        if now < next_at:
            return

        candidates = [
            preset_path
            for preset_path in available
            if os.path.normcase(os.path.normpath(os.fspath(preset_path)))
            != current_key
        ]
        if not candidates:
            return

        available_keys = {
            os.path.normcase(os.path.normpath(os.fspath(preset_path)))
            for preset_path in available
        }
        recent = [
            key
            for key in getattr(self, "_auto_switch_recent", [])
            if key in available_keys and key != current_key
        ]
        max_recent = min(32, max(0, len(available) - 2))
        if len(recent) > max_recent:
            recent = recent[-max_recent:]
        self._auto_switch_recent = recent

        recent_set = set(recent)
        fresh_candidates = [
            preset_path
            for preset_path in candidates
            if os.path.normcase(os.path.normpath(os.fspath(preset_path)))
            not in recent_set
        ]
        if not fresh_candidates:
            fresh_candidates = candidates

        target_path = random.choice(fresh_candidates)
        target_key = os.path.normcase(os.path.normpath(os.fspath(target_path)))
        if target_key in recent:
            recent.remove(target_key)
        recent.append(target_key)
        if len(recent) > max_recent:
            recent = recent[-max_recent:]
        self._auto_switch_recent = recent

        self._auto_switch_next_at = now + interval
        result = self.load_milk_preset(target_path, smooth_transition=True)
        if result is PresetLoadResult.FAILED:
            logger.error("Automatic preset switch failed for %s", target_path)

    def _render_frame(self, width: int, height: int) -> FrameResult:
        """Render one frame and keep failures visible to the GLArea callback."""
        if not getattr(self, "opengl_ready", False):
            return FrameResult.FAILED
        try:
            self._render_frame_impl(width, height)
        except Exception as exc:
            self._consecutive_render_failures = getattr(
                self, "_consecutive_render_failures", 0
            ) + 1
            phase = getattr(self, "_current_render_phase", "main_visual")
            recovery_action = "consecutive_failure_counted"
            self._record_frame_failure(phase, exc, recovery_action)
            if self._consecutive_render_failures == 10 or (
                self._consecutive_render_failures > 10
                and self._consecutive_render_failures % 120 == 0
            ):
                logger.error(
                    "Render failed %d consecutive times; requesting recovery (phase=%s)",
                    self._consecutive_render_failures,
                    phase,
                )
                recovery = getattr(self, "_recovery", None)
                if recovery is not None:
                    try:
                        recovery.request_shader_recovery()
                        logger.error("entering recovery (phase=%s)", phase)
                    except Exception:
                        logger.exception("Shader recovery request failed")
                else:
                    logger.error("Shader recovery unavailable; renderer remains in failed state")
            elif self._consecutive_render_failures == 1:
                logger.exception("Render frame failed (phase=%s)", phase)
            elif self._consecutive_render_failures % 60 == 0:
                logger.error(
                    "Render has failed for %d consecutive frames; latest error: %s (phase=%s)",
                    self._consecutive_render_failures,
                    exc,
                    phase,
                )
            return FrameResult.FAILED

        self._consecutive_render_failures = 0
        return FrameResult.SUCCESS

    def _ensure_render_continuity(self) -> None:
        """Ensure the render loop continues even after errors.

        This method is called after each frame to reset the consecutive
        failure counter and ensure queue_render is called if appropriate.
        It guards against render failures that might otherwise stop the
        animation loop.
        """
        # If we had consecutive failures, we should still try to render
        # the next frame. The render loop will self-correct if the
        # underlying issues are fixed.

    def _advance_frame_clock(self, now: float | None = None) -> float:
        """Advance preset time using a bounded monotonic wall-clock delta."""
        if now is None:
            now = time.monotonic()
        last = getattr(self, "_last_frame_time", None)
        if last is None:
            dt = 1.0 / 60.0
        else:
            dt = min(max(float(now) - float(last), 1.0 / 240.0), 0.1)
        self._last_frame_time = max(float(now), float(last)) if last is not None else float(now)
        self._current_dt = dt
        self._current_time = float(getattr(self, "_current_time", 0.0)) + dt
        self._current_frame_id = int(getattr(self, "_current_frame_id", 0)) + 1
        self._vm_uniforms_upload_frame = -1
        return dt

    def _upload_vm_uniforms(
        self,
        program_id: int | None = None,
        vm: object | None = None,
    ) -> None:
        prog = int(program_id) if program_id is not None else int(getattr(self, "_main_visual_program", 0))
        if prog <= 0:
            return
        if not hasattr(self, "_gl_state") or self._gl_state is None:
            return
        try:
            handle = self._get_or_create_program_handle(prog)
        except Exception:
            logger.exception("Failed to get program handle for VM uniform upload")
            return
        gl_state = self._gl_state
        # --- time-related uniforms (always fresh per frame) ---
        try:
            handle.upload_uniform("u_time", getattr(self, "_current_time", 0.0), gl_state)
        except Exception:
            logger.exception("Failed to upload u_time uniform")
        try:
            handle.upload_uniform("u_dt", getattr(self, "_current_dt", 1.0 / 60.0), gl_state)
        except Exception:
            logger.exception("Failed to upload u_dt uniform")
        try:
            dt = float(getattr(self, "_current_dt", 1.0 / 60.0) or 1.0 / 60.0)
            handle.upload_uniform("u_fps", 1.0 / max(dt, 1e-6), gl_state)
        except Exception:
            logger.exception("Failed to upload u_fps uniform")
        try:
            handle.upload_uniform("u_frame", float(getattr(self, "_current_frame_id", 0)), gl_state)
        except Exception:
            logger.exception("Failed to upload u_frame uniform")
        # --- dirty tracking for VM uniforms (avoid redundant GPU uploads) ---
        if not hasattr(self, "_vm_dirty"):
            object.__setattr__(self, "_vm_dirty", set())
        if not hasattr(self, "_vm_last_uploaded"):
            object.__setattr__(self, "_vm_last_uploaded", {})
        vm_obj = vm if vm is not None else getattr(self, "preset_vm", None)
        s = getattr(vm_obj, "state", None) if vm_obj is not None else None
        if s is None:
            # No VM state — upload zeros for critical registers so the shader
            # doesn't see garbage/uninitialized values that could produce black.
            try:
                handle.upload_uniform_1fv(
                    "u_q", np.full(33, 0.0, dtype=np.float32), gl_state
                )
            except Exception:
                logger.exception("Failed to upload u_q (zero fallback)")
            try:
                handle.upload_uniform_1fv(
                    "u_t", np.full(9, 0.0, dtype=np.float32), gl_state
                )
            except Exception:
                logger.exception("Failed to upload u_t (zero fallback)")
            return

        def upload(name: str, value: object) -> None:
            try:
                handle.upload_uniform(name, value, gl_state)
            except Exception:
                logger.exception("Failed to upload %s uniform", name)

        def value(name: str, default: float = 0.0) -> float:
            try:
                return float(s.get(name, default))
            except (TypeError, ValueError, AttributeError):
                return float(default)

        multipass = getattr(getattr(self, "_render_graph", None), "multipass", None)
        try:
            width = float(getattr(multipass, "width", 1) or 1)
            height = float(getattr(multipass, "height", 1) or 1)
        except (TypeError, ValueError):
            width = float(value("width", 1))
            height = float(value("height", 1))
        upload(
            "u_resolution",
            (
                width,
                height,
                width / height,
                width / height if width < height else 1.0,
            ),
        )
        upload("u_aspect", height / width if height < width else 1.0)
        upload("u_audio", (value("bass"), value("mid"), value("treb")))
        upload(
            "u_audio_att",
            (value("bass_att"), value("mid_att"), value("treb_att")),
        )
        upload("u_beat", float(s.get("beat", 0.0)))
        upload("u_beat_flash", float(s.get("beat_flash", 0.0)))
        upload("u_audio_rms", value("rms"))
        upload("u_audio_peak", value("peak"))
        upload("u_zoom", value("zoom", 1.0))
        upload("u_rot", value("rot"))
        upload("u_warp", value("warp"))
        upload("u_decay", value("decay", 0.9))
        upload("u_center", (value("cx", 0.5), value("cy", 0.5)))
        upload("u_offset", (value("dx"), value("dy")))
        upload("u_scale", (value("sx", 1.0), value("sy", 1.0)))
        upload("u_rand_seed", float(getattr(s, "rand_seed", 0)))
        upload("u_fc", float(getattr(self, "_current_frame_id", 0)))

        try:
            handle.upload_uniform_1fv(
                "u_q", np.asarray([0.0, *getattr(s, "q", [])], dtype=np.float32), gl_state
            )
            handle.upload_uniform_1fv(
                "u_t",
                np.asarray([0.0, *getattr(s, "t_vars", [])], dtype=np.float32),
                gl_state,
            )
            regs = getattr(s, "_global_regs", np.zeros(100, dtype=np.float32))
            handle.upload_uniform_1fv(
                "u_regs", np.asarray(regs, dtype=np.float32), gl_state
            )
        except Exception:
            logger.exception("Failed to upload MilkDrop register arrays")

        snapshot = self._copy_audio_snapshot()
        spectrum = snapshot.get("spectrum", snapshot.get("_spectrum")) if snapshot else None
        if spectrum is not None:
            try:
                values = np.asarray(spectrum, dtype=np.float32).reshape(-1)
                bands = np.zeros(64, dtype=np.float32)
                if values.size:
                    # Use evenly-spaced indices across the spectrum
                    indices = np.linspace(0, values.size - 1, num=64).astype(int)
                    bands[:] = values[indices]
                # Upload audio bands - these are raw band levels for the current frame
                handle.upload_uniform_1fv("u_audio_bands", bands, gl_state)
            except Exception:
                logger.exception("Failed to upload spectrum uniform")
        else:
            # No audio data available for this frame - upload zeros to avoid
            # stale values from previous frames persisting indefinitely.
            handle.upload_uniform_1fv("u_audio_bands", np.zeros(64, dtype=np.float32), gl_state)
        self._vm_uniforms_upload_frame = int(
            getattr(self, "_current_frame_id", 0)
        )

    @contextmanager
    def _current_gl_context(self):
        self.make_current()
        try:
            yield
        finally:
            self._unmake_current()

    def _unmake_current(self):
        try:
            Gdk.GLContext.clear_current()
            return
        except Exception:
            logger.debug("Gdk.GLContext.clear_current failed", exc_info=True)
        try:
            gl_state = getattr(self, "_gl_state", None)
            if gl_state is not None:
                gl_state.use_program(0)
            else:
                GL.glUseProgram(0)  # gl-state-guard: inline-approved
        except Exception:  # noqa: BLE001, S110
            pass
        return

    def make_current(self):
        try:
            return super().make_current()
        except Exception:
            try:
                ctx = getattr(self, '_gl_context', None)
                if ctx is not None and hasattr(ctx, 'make_current'):
                    try:
                        ctx.make_current()
                        return
                    except Exception:  # noqa: BLE001, S110
                        pass
            except Exception:  # noqa: BLE001, S110
                pass
            raise

    def _upload_q_registers(self, *args, **kwargs):
        try:
            return self._upload_vm_uniforms(*args, **kwargs)
        except Exception:  # noqa: BLE001
            return None

    def _upload_t_registers(self, *args, **kwargs):
        try:
            return self._upload_vm_uniforms(*args, **kwargs)
        except Exception:  # noqa: BLE001
            return None

    def _upload_reg_registers(self, full: bool = True, *args, **kwargs):
        try:
            return self._upload_vm_uniforms(*args, **kwargs)
        except Exception:  # noqa: BLE001
            return None


    def _cleanup_gl_resources(self, full: bool = True):
        if full is False:
            raise RuntimeError("forbidden")
        try:
            if hasattr(self, "_gl_state") and self._gl_state is not None:
                try:
                    self._gl_state.bind_framebuffer(GL.GL_FRAMEBUFFER, 0)
                except Exception:  # noqa: BLE001, S110
                    pass
        except Exception:  # noqa: BLE001, S110
            pass
        try:
            self._stop_texture_load_thread()
        except Exception:  # noqa: BLE001, S110
            pass
        return True

    def _lookup_program_handle(self, program_id: int):
        generation = int(getattr(self, "_program_context_generation", 0))
        return getattr(self, "_program_handles", {}).get((generation, int(program_id)))

    def _program_handle_key(self, program_id: int):
        generation = int(getattr(self, "_program_context_generation", 0))
        return (generation, int(program_id))

    def _get_or_create_program_handle(self, program_id: int):
        from program_handle import ProgramHandle

        pid = int(program_id)
        generation = int(getattr(self, "_program_context_generation", 0))
        handles = getattr(self, "_program_handles", None)
        if handles is None:
            handles = {}
            self._program_handles = handles
        key = (generation, pid)
        handle = handles.get(key)
        if handle is not None and getattr(handle, "valid", lambda: False)():
            return handle
        # If handle exists but is retired/invalid, create fresh
        handle = ProgramHandle(program_id=pid, generation=generation)
        handles[key] = handle
        return handle

    def _bump_program_context_generation(self) -> None:
        self._program_context_generation = int(getattr(self, "_program_context_generation", 0)) + 1
        if hasattr(self, "_program_handles"):
            self._program_handles = {}
        self._active_program_handle = None
        self._uniforms_locked_to = None
        self._uniform_cache_generation = int(getattr(self, "_uniform_cache_generation", 0)) + 1

    def invalidate_context(self, reason: str = "context_destroyed"):
        try:
            self._program_handles = {}
        except Exception:  # noqa: BLE001, S110
            pass
        try:
            self._active_program_handle = None
        except Exception:  # noqa: BLE001, S110
            pass
        try:
            self._uniforms_locked_to = None
        except Exception:  # noqa: BLE001, S110
            pass
        try:
            self._program_context_generation = int(getattr(self, "_program_context_generation", 0)) + 1
        except Exception:  # noqa: BLE001, S110
            pass
        try:
            self._uniform_cache_generation = int(getattr(self, "_uniform_cache_generation", 0)) + 1
        except Exception:  # noqa: BLE001, S110
            pass
        try:
            if hasattr(self, "_gl_resource_owner") and self._gl_resource_owner is not None:
                try:
                    self._gl_resource_owner.invalidate_context(reason)
                except Exception:  # noqa: BLE001, S110
                    pass
        except Exception:  # noqa: BLE001, S110
            pass
        try:
            if hasattr(self, "_gl_state") and self._gl_state is not None:
                try:
                    self._gl_state.invalidate_context(reason)
                except Exception:  # noqa: BLE001, S110
                    pass
        except Exception:  # noqa: BLE001, S110
            pass
        return True

    def _assert_strict_pipeline_clean(self) -> None:
        """Assert that no non-strict effects are enabled in STRICT_MILKDROP mode.

        In strict mode, motion blur, lens distortion, kaleidoscope, glitch,
        and the enhancement layer must all be disabled.  Non-strict modes
        are a no-op.
        """
        if not getattr(self, "strict_milkdrop_mode", True):
            return
        if getattr(self, "enable_motion_blur", False):
            raise AssertionError("strict milkdrop mode: motion blur is not allowed")
        if getattr(self, "enable_lens_distortion", False):
            raise AssertionError("strict milkdrop mode: lens distortion is not allowed")
        if getattr(self, "enable_kaleidoscope", False):
            raise AssertionError("strict milkdrop mode: kaleidoscope is not allowed")
        if getattr(self, "enable_glitch", False):
            raise AssertionError("strict milkdrop mode: glitch is not allowed")
        enhancement = getattr(self, "enhancement", None)
        if enhancement is not None and getattr(enhancement, "enabled", False):
            raise AssertionError(
                "strict milkdrop mode: EnhancementLayer must be disabled"
            )

    def _retire_program(self, prog_id: int, force: bool = False) -> None:
        prog_id = int(prog_id)
        if prog_id <= 0:
            return
        handles = getattr(self, "_program_handles", None)
        if isinstance(handles, dict):
            for key in [k for k in list(handles.keys()) if k[1] == prog_id]:
                h = handles.pop(key, None)
                if h is not None and hasattr(h, "retire"):
                    try:
                        h.retire()
                    except Exception:  # noqa: BLE001, S110
                        pass
                    if hasattr(h, "uniform_locations"):
                        try:
                            h.uniform_locations = {}
                        except Exception:  # noqa: BLE001, S110
                            pass
        gl_state = getattr(self, "_gl_state", None)
        if gl_state is not None and hasattr(gl_state, "invalidate_program"):
            try:
                gl_state.invalidate_program(prog_id)
            except Exception:  # noqa: BLE001, S110
                pass
        owner = getattr(self, "_gl_resource_owner", None)
        if owner is not None and hasattr(owner, "queue_delete"):
            try:
                owner.queue_delete("program", prog_id)
            except Exception:  # noqa: BLE001, S110
                pass



    def _record_frame_failure(self, phase, error, recovery_action=None):
        events = getattr(self, "_frame_failure_events", None)
        if events is None:
            events = []
            self._frame_failure_events = events
        events.append(
            {
                "phase": phase,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "recovery_action": recovery_action,
                "consecutive_failures": getattr(self, "_consecutive_render_failures", 0),
            }
        )
        if len(events) > 16:
            self._frame_failure_events = events[-16:]

    def _on_screen_draw_failed(self, err, bound: int | object, context: str = "") -> None:
        logger.error("Screen draw failed in %s: %s", context, err)
        try:
            bound_val = int(bound) if isinstance(bound, (int, float, str, bytes, bytearray)) else 0
        except Exception:  # noqa: BLE001
            bound_val = 0
        if bound_val == 0 and self._gl_state is not None:
            self._gl_state._default_fb_unpresentable = True

    def _fix_shader_vector_mismatches(self, code: str, report=None):
        try:
            import re
            try:
                from renderer.shaders import ShaderRepairReport
            except Exception:  # noqa: BLE001
                ShaderRepairReport = None
            
            if report is None and ShaderRepairReport is not None:
                report = ShaderRepairReport(allow_contextual_repairs=True)
            elif report is None:
                class DummyReport:
                    def __init__(self):
                        self.repairs = []
                        self.rejections = []
                        self.repairs_safe = []
                        self.repairs_contextual = []
                        self.allow_contextual_repairs = True
                    def applied(self, msg, level="SAFE"):
                        if level == "CONTEXTUAL":
                            self.repairs_contextual.append(msg)
                            if self.allow_contextual_repairs:
                                self.repairs.append(msg)
                                return True
                            return False
                        self.repairs_safe.append(msg)
                        self.repairs.append(msg)
                        return True
                    def rejected(self, msg):
                        self.rejections.append(msg)
                report = DummyReport()
            
            # Fix spaces before swizzles
            code = re.sub(r'\)\s+\.([rgba]+|[xyzw]+)', r').\1', code)
            code = re.sub(r'([a-zA-Z0-9_]+)\s+\.([rgba]+|[xyzw]+)', r'\1.\2', code)
            code = re.sub(r'(texture\([^)]+\))\s+\.([xyzwrgba]+)', r'\1.\2', code)
            code = re.sub(r'(GetBlur[123]\([^)]+\))\s+\.([xyzw])', r'\1.\2', code)
            code = re.sub(r'(GetPixel\([^)]+\))\s+\.([xyzw])', r'\1.\2', code)
            
            # Fix float declared but used as vec2 from swizzles like .xy, .zw
            # Look for float var assigned to something with .xy/.zw/.xyz
            for m in re.finditer(r'float\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([^;]+);', code):
                var = m.group(1)
                rhs = m.group(2)
                if re.search(r'\.xy|\.zw|\.xz|\.yz', rhs) and not re.search(r'\.x\b|\.y\b|\.z\b|\.w\b(?![a-z])', rhs.split('.')[0] if False else rhs):  # rough
                    pass
            # More specific
            code = re.sub(r'float\s+(corr|d|[a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([^;]*\.xy[^;]*);', 
                         lambda m: f'vec2 {m.group(1)} = {m.group(2)};', code)
            code = re.sub(r'float\s+(corr|d|[a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([^;]*\.zw[^;]*);', 
                         lambda m: f'vec2 {m.group(1)} = {m.group(2)};', code)
            
            # Fix pow with scalar exponent
            def fix_pow_vec(m):
                vec_call = m.group(1)
                exp = m.group(2).strip()
                if re.match(r'^vec[2-4]\s*\(', exp):
                    return m.group(0)
                if vec_call.strip().startswith('vec3'):
                    result = f'pow({vec_call}, vec3({exp}))'
                    report.applied("pow vec3 scalar exponent wrapped (CONTEXTUAL)", "CONTEXTUAL")
                    return result
                elif vec_call.strip().startswith('vec2'):
                    result = f'pow({vec_call}, vec2({exp}))'
                    report.applied("pow vec2 scalar exponent wrapped (CONTEXTUAL)", "CONTEXTUAL")
                    return result
                elif vec_call.strip().startswith('vec4'):
                    result = f'pow({vec_call}, vec4({exp}))'
                    report.applied("pow vec4 scalar exponent wrapped (CONTEXTUAL)", "CONTEXTUAL")
                    return result
                return m.group(0)
            
            code = re.sub(r'pow\(\s*(vec3\([^)]+\))\s*,\s*([0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?)\s*\)', fix_pow_vec, code)
            code = re.sub(r'pow\(\s*(vec2\([^)]+\))\s*,\s*([0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?)\s*\)', fix_pow_vec, code)
            code = re.sub(r'pow\(\s*(vec4\([^)]+\))\s*,\s*([0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?)\s*\)', fix_pow_vec, code)
            
            # GetBlur/texture blur used as scalar multiplied by vec
            code_before = code
            code = re.sub(r'(texture\(u_blur[123][^)]*\))\s*\*\s*(vec[234])', 
                         lambda m: m.group(1) + '.x * ' + m.group(2), code)
            code = re.sub(r'(GetBlur[123]\([^)]+\))\s*\*\s*(vec[234])', 
                         lambda m: m.group(1) + '.x * ' + m.group(2), code)
            if code != code_before:
                report.applied("GetBlur used as scalar multiplied by vector: added .x swizzle", "CONTEXTUAL")
            
            # texture() assigned to vec3 without .rgb
            def fix_texture_assign(m):
                prefix = m.group(1)
                tex_call = m.group(2)
                if re.search(r'\.[rgba]+|\.[xyzw]+', tex_call):
                    return m.group(0)
                result = f'{prefix} {tex_call}.rgb;'
                report.applied("texture() assigned to vec3 without .rgb: added .rgb", "CONTEXTUAL")
                return result
            
            code = re.sub(r'(ret\s*=|color\s*=|fragColor\s*=|FragColor\s*=|vec3\s+[a-zA-Z_][a-zA-Z0-9_]*\s*=)\s*(texture\([^)]+\))\s*;', fix_texture_assign, code)
            code = re.sub(r'(ret\s*=|color\s*=|FragColor\s*=|vec3\s+[a-zA-Z_][a-zA-Z0-9_]*\s*=)\s*(tex2D\([^)]+\))\s*;', 
                         lambda m: (m.group(1) + ' ' + m.group(2).replace('tex2D', 'texture') + '.rgb;'), code)
            
            # Bool arithmetic wrapped
            code = re.sub(r'\(\s*([a-zA-Z_][a-zA-Z0-9_]*\s*(?:>|<|>=|<=|==|!=)\s*[^()]+?)\s*\)\s*\*\s*vec([234])', 
                         r'(float(\1)) * vec\2', code)
            
            # log10 repair
            if re.search(r'\blog10\s*\(', code):
                code2 = re.sub(r'\blog10\s*\(([^()]+)\)', r'(log(\1)/log(10.0))', code)
                if code2 != code:
                    report.applied("log10 repaired to log(x)/log(10.0)", "CONTEXTUAL")
                    code = code2
            
            # GetBlur in float assignment
            code = re.sub(r'float\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*GetBlur([123])\(([^()]+)\)\s*;', 
                         r'float \1 = lum(GetBlur\2(\3));', code)
            code = re.sub(r'float\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(texture\(u_blur[123][^)]*\))\s*;', 
                         r'float \1 = lum(\2);', code)
            code = re.sub(r'float\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([^;]*GetBlur[123]\([^)]+\)[^;]*);', 
                         lambda m: f'float {m.group(1)} = lum({m.group(2)});' if 'lum(' not in m.group(2) else m.group(0), code)
            
            # float var with z/y/w swizzle - reject
            for m in re.finditer(r'float\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=', code):
                var = m.group(1)
                if re.search(r'\b' + re.escape(var) + r'\s*\.\s*[yzw]', code):
                    report.rejected(f"float variable '{var}' accessed with swizzle")
            
            # dx/dy parens balance fix
            lines = code.split('\n')
            new_lines = []
            for line in lines:
                if re.match(r'\s*(dx|dy)\s*=', line):
                    open_c = line.count('(')
                    close_c = line.count(')')
                    if open_c > close_c:
                        missing = open_c - close_c
                        stripped = line.rstrip()
                        if stripped.endswith(';'):
                            line = stripped[:-1] + ')' * missing + ';'
                        else:
                            line = stripped + ')' * missing
                new_lines.append(line)
            code = '\n'.join(new_lines)
            
            report.repairs.append("shader repair complete")
            
            return code, report
        except Exception as exc:  # noqa: BLE001
            logger.debug("Shader repair failed: %s", exc)
            try:
                if report is not None:
                    report.repairs.append("shader repair complete")
            except Exception:  # noqa: BLE001, S110
                pass
            return code, report



    def _prepare_pass_shader_body(self, raw_code: str) -> str:
        if not raw_code:
            return ""
        try:
            code = str(raw_code)
            # Do basic HLSL->GLSL translation similar to _translate_hlsl_fragment
            import re

            defines = {}

            def process_define(m):
                name = m.group(1)
                val = m.group(2).strip()
                defines[name] = val
                return ''

            code = re.sub(r'#define\s+(\w+)\s+(.+?)(?:\n|$)', process_define, code)
            for name, val in defines.items():
                code = re.sub(r'\b' + re.escape(name) + r'\b', val, code)

            code = re.sub(r'sampler2D\s+(\w+);', r'uniform sampler2D \1;', code)
            code = re.sub(r'sampler3D\s+(\w+);', r'uniform sampler3D \1;', code)
            code = re.sub(r'samplerCube\s+(\w+);', r'uniform samplerCube \1;', code)

            code = code.replace('sampler_main', 'u_prev_frame')
            code = code.replace('sampler_fc_main', 'u_prev_frame')
            code = code.replace('sampler_bw_main', 'u_prev_frame')
            code = code.replace('sampler_noise_hq', 'iNoiseHQ')
            code = code.replace('sampler_noise_lq', 'iNoiseLQ')
            code = code.replace('sampler_noise_mq', 'iNoiseMQ')
            code = code.replace('ret', 'color')
            code = re.sub(r'\btex2D\s*\(', 'texture(', code)
            code = re.sub(r'\btex2Dlod\s*\(', 'textureLod(', code)
            code = re.sub(r'\btexCUBE\s*\(', 'texture(', code)
            code = code.replace('lerp(', 'mix(')
            code = code.replace('frac(', 'fract(')
            code = re.sub(r'\batan2\s*\(', 'atan(', code)
            code = code.replace('ddx(', 'dFdx(')
            code = code.replace('ddy(', 'dFdy(')
            code = code.replace('fmod(', 'mod(')

            code = code.replace('float2', 'vec2')
            code = code.replace('float3', 'vec3')
            code = code.replace('float4', 'vec4')
            code = code.replace('half2', 'vec2')
            code = code.replace('half3', 'vec3')
            code = code.replace('half4', 'vec4')

            # map common blur names
            code = code.replace('sampler_blur1', 'u_blur1')
            code = code.replace('sampler_blur2', 'u_blur2')
            code = code.replace('sampler_blur3', 'u_blur3')
            code = code.replace('GetBlur1', 'GetBlur1')
            code = code.replace('GetBlur2', 'GetBlur2')
            code = code.replace('GetBlur3', 'GetBlur3')

            code = self._map_sampler_names(code)
            return code
        except Exception:  # noqa: BLE001
            return str(raw_code)

    def _pass_shader_uniforms_block(self) -> str:
        return """uniform sampler2D u_prev_frame;
uniform sampler2D u_blur1;
uniform sampler2D u_blur2;
uniform sampler2D u_blur3;
uniform sampler2D iNoiseLQ;
uniform sampler2D iNoiseMQ;
uniform sampler2D iNoiseHQ;
uniform vec2 u_resolution;
uniform float u_time;
uniform float u_dt;
uniform float u_fps;
uniform float u_audio_bass;
uniform float u_audio_mid;
uniform float u_audio_treb;
uniform float u_audio_bass_att;
uniform float u_audio_mid_att;
uniform float u_audio_treb_att;
uniform float u_audio_beat;
uniform float u_audio_beat_flash;
uniform float u_audio_rms;
uniform float u_audio_peak;
uniform vec4 u_audio;
uniform float u_q[33];
uniform float u_t[9];
uniform float u_regs[100];
uniform float u_zoom;
uniform float u_rot;
uniform float u_warp;
uniform float u_decay;
uniform vec2 u_center;
uniform vec2 u_offset;
uniform vec2 u_scale;
uniform float u_rand_seed;"""

    def _pass_shader_locals_block(self) -> str:
        return """    vec2 uv = v_uv;
    float time = u_time;
    float bass = u_audio_bass;
    float mid = u_audio_mid;
    float treb = u_audio_treb;
    float bass_att = u_audio_bass_att;
    float mid_att = u_audio_mid_att;
    float treb_att = u_audio_treb_att;
    float beat = u_audio_beat;
    float beat_flash = u_audio_beat_flash;
    vec4 audio = u_audio;
    for(int i=1;i<33;i++) { /* q/t/regs available via uniforms */ }
    vec3 color = vec3(0.0);"""

    def _build_translated_pass_shader(self, raw_code: str, *, pass_name: str) -> str:
        try:
            body = self._prepare_pass_shader_body(raw_code)
            if not body:
                body = "color = texture(u_prev_frame, uv).rgb;"

            try:
                helpers = GLSLGenerator._get_helper_functions(MEGABUF_SIZE)
            except Exception:  # noqa: BLE001
                try:
                    helpers = GLSLGenerator._get_helper_functions(MEGABUF_SIZE)
                except Exception:  # noqa: BLE001
                    helpers = ""

            if "FragColor" not in body and "gl_FragColor" not in body:
                epilogue = "\n    FragColor = vec4(color, 1.0);\n"
            else:
                body = body.replace("gl_FragColor", "FragColor")
                epilogue = ""

            return f"""#version 330 core
precision highp float;
// Auto-generated {pass_name} pass (HLSL → GLSL)
{self._pass_shader_uniforms_block()}
in vec2 v_uv;
out vec4 FragColor;
{helpers}
void main() {{
{self._pass_shader_locals_block()}
    // ---- preset {pass_name} body ----
    {body}
{epilogue}}}"""
        except Exception:
            logger.exception("Failed to build translated pass shader")
            return """#version 330 core
precision highp float;
uniform sampler2D u_prev_frame;
in vec2 v_uv;
out vec4 FragColor;
void main() {
    vec3 color = texture(u_prev_frame, v_uv).rgb;
    FragColor = vec4(color, 1.0);
}"""

    def _translate_hlsl_fragment(self, code: str) -> str:
        """Translate HLSL fragment code to GLSL."""
        try:
            import re
            # Handle sampler aliases from #define
            defines = {}
            def process_define(m):
                name = m.group(1)
                val = m.group(2).strip()
                defines[name] = val
                return ''
            code = re.sub(r'#define\s+(\w+)\s+(.+?)(?:\n|$)', process_define, code)
            for name, val in defines.items():
                code = re.sub(r'\b' + re.escape(name) + r'\b', val, code)
            
            # Handle sampler declarations
            code = re.sub(r'sampler2D\s+(\w+);', r'uniform sampler2D \1;', code)
            code = re.sub(r'sampler3D\s+(\w+);', r'uniform sampler3D \1;', code)
            code = re.sub(r'samplerCube\s+(\w+);', r'uniform samplerCube \1;', code)
            
            # Map common names
            code = code.replace('sampler_main', 'u_prev_frame')
            code = code.replace('sampler_fc_main', 'u_prev_frame')
            code = code.replace('sampler_bw_main', 'u_prev_frame')
            code = code.replace('sampler_noise_hq', 'iNoiseHQ')
            code = code.replace('sampler_noise_lq', 'iNoiseLQ')
            code = code.replace('sampler_noise_mq', 'iNoiseMQ')
            code = code.replace('ret', 'color')
            code = re.sub(r'\btex2D\s*\(', 'texture(', code)
            code = re.sub(r'\btex2Dlod\s*\(', 'textureLod(', code)
            code = re.sub(r'\btexCUBE\s*\(', 'texture(', code)
            code = code.replace('lerp(', 'mix(')
            code = code.replace('frac(', 'fract(')
            code = re.sub(r'\batan2\s*\(', 'atan(', code)
            code = code.replace('ddx(', 'dFdx(')
            code = code.replace('ddy(', 'dFdy(')
            code = code.replace('fmod(', 'mod(')
            
            code = code.replace('float2', 'vec2')
            code = code.replace('float3', 'vec3')
            code = code.replace('float4', 'vec4')
            code = code.replace('half2', 'vec2')
            code = code.replace('half3', 'vec3')
            code = code.replace('half4', 'vec4')
            
            return code
        except Exception:  # noqa: BLE001
            return code



    def _build_main_visual_shader_source(self, program=None) -> tuple[str, str]:
        """Compile per-pixel preset equations into the feedback shader."""
        equations = list(getattr(program, "per_pixel", []) or [])
        statements: list[str] = []
        lhs_names: set[str] = set()
        for equation in equations:
            source = str(getattr(equation, "source", equation)).strip()
            if not source:
                continue
            try:
                lhs, rhs, _is_statement, _qualifier, _refs, rhs_type = (
                    compile_assignment_to_glsl(source)
                )
            except Exception as exc:
                raise ValueError(
                    f"Could not compile per-pixel equation {source!r}: {exc}"
                ) from exc
            lhs = self._map_sampler_names(lhs)
            rhs = self._map_sampler_names(rhs)
            lhs_type = GLSLGenerator._PIXEL_SHADER_VAR_TYPES.get(lhs, "float")  # type: ignore[attr-defined]
            # Avoid double-wrapping: visit_Attribute already wraps vec3 swizzles in lum()
            if lhs_type == "float" and rhs_type.startswith("vec") and not rhs.startswith("lum("):
                rhs = f"lum({rhs})"
            elif lhs_type.startswith("vec") and rhs_type == "float":
                rhs = f"{lhs_type}({rhs})"
            statements.append(f"{lhs} = {rhs};")
            lhs_names.add(lhs)
            if lhs == "ret":
                statements.extend(
                    ("r = ret.r;", "g = ret.g;", "b = ret.b;")
                )

        code = "\n        ".join(statements)
        known = set(_PIXEL_CODE_KNOWN_IDENTS)
        known.update(_PIXEL_LOCALS_DECLARED)
        known.update(
            name
            for name in (
                "uv_w", "ret", "md_prev_color", "md_color", "md_p", "x", "y",
                "zoom", "rot", "warp", "decay", "cx", "cy", "dx", "dy",
                "sx", "sy", "rad", "ang", "aspect", "aspectx", "aspecty",
                "center", "offset", "scale", "r", "g", "b",
            )
        )
        known.update(f"q{i}" for i in range(1, 33))
        known.update(f"t{i}" for i in range(1, 9))
        known.update(f"reg{i:02d}" for i in range(100))
        known.update(
            {
                "u_dt", "u_fps", "u_time", "u_frame", "u_q", "u_t",
                "u_regs", "u_resolution", "u_center", "u_offset", "u_scale",
                "u_zoom", "u_rot", "u_warp", "u_decay", "u_audio",
                "u_audio_att", "u_audio_bands", "u_beat", "u_beat_flash",
                "u_audio_rms", "u_audio_peak", "u_rand_seed", "u_fc",
                "u_aspect", "u_prev_frame", "u_texture", "u_megabuf",
            }
        )
        custom_names = sorted(
            {
                name
                for name in _PIXEL_CODE_IDENT_RE.findall(code)
                if name not in known and not name.startswith("u_")
            }
        )
        custom_declarations = "\n        ".join(
            f"float {name} = 0.0;" for name in custom_names
        )

        pixel_color = (
            f"""
        {custom_declarations}
        vec3 ret = mix(md_prev_color, vec3(0.0), decay);
        float r = ret.r;
        float g = ret.g;
        float b = ret.b;
        {code}
        vec3 md_color = vec3(r, g, b);
"""
            if equations
            else """
        vec3 md_color = mix(md_prev_color, vec3(0.0), decay);
"""
        )
        vertex = """#version 330 core
layout(location = 0) in vec2 a_pos;
out vec2 v_uv;
void main() {
    v_uv = a_pos * 0.5 + 0.5;
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
"""
        # Uniform declarations must come before helper functions so that helper
        # functions (md_rand, megabuf_read, GetBlurN, GetPixel) can reference
        # u_rand_seed, u_megabuf, u_prev_frame, u_resolution etc. without
        # "undeclared" errors from the GLSL ES compiler.
        helpers = GLSLGenerator._get_helper_functions(MEGABUF_SIZE)

        frag = f"""#version 330 core
precision highp float;
uniform float u_dt;
uniform float u_fps;
uniform float u_time;
uniform float u_frame;
uniform float u_q[33];
uniform float u_t[9];
uniform float u_regs[100];
uniform vec4 u_resolution;
uniform vec2 u_center;
uniform vec2 u_offset;
uniform vec2 u_scale;
uniform float u_zoom;
uniform float u_rot;
uniform float u_warp;
uniform float u_decay;
uniform vec3 u_audio;
uniform vec3 u_audio_att;
uniform float u_audio_bands[64];
uniform float u_beat;
uniform float u_beat_flash;
uniform float u_audio_rms;
uniform float u_audio_peak;
uniform float u_rand_seed;
uniform float u_fc;
uniform float u_aspect;
uniform sampler2D u_prev_frame;
uniform samplerBuffer u_megabuf;
{"uniform sampler2D u_texture;" if not getattr(self, "strict_milkdrop_mode", True) else ""}
    in vec2 v_uv;
    out vec4 FragColor;
    {helpers}
    void main() {{
        vec2 uv = v_uv;
    vec2 uv2 = uv;
    float x = uv.x;
    float y = uv.y;
    {"vec3 tex_sample = texture(u_texture, uv).rgb;" if not getattr(self, "strict_milkdrop_mode", True) else ""}
    float zoom = u_zoom;
    float rot = u_rot;
    float warp = u_warp;
    float decay = u_decay;
    float cx = u_center.x;
    float cy = u_center.y;
    float dx = u_offset.x;
    float dy = u_offset.y;
    float sx = u_scale.x;
    float sy = u_scale.y;
    float rad = length(uv - u_center);
    float ang = atan(uv.y - u_center.y, uv.x - u_center.x);
    float aspect = u_resolution.z;
    float aspectx = (u_resolution.x < u_resolution.y) ? u_resolution.x / u_resolution.y : 1.0;
    float aspecty = (u_resolution.y < u_resolution.x) ? u_resolution.y / u_resolution.x : 1.0;
    {_Q_LOCALS}
    {_T_LOCALS}
    {_REG_LOCALS}
    vec2 center = vec2(cx, cy);
    vec2 offset = vec2(dx, dy);
    vec2 scale = vec2(sx, sy);
    vec2 md_p = vec2(x, y) - center;
    md_p *= scale;
    md_p = mat2(cos(rot), -sin(rot), sin(rot), cos(rot)) * md_p;
    md_p /= max(abs(zoom), 0.0001);
    vec2 uv_w = fract(md_p + center + offset);
    vec3 md_prev_color = texture(u_prev_frame, uv_w).rgb;
    {pixel_color}
    FragColor = vec4(md_color, 1.0);
}}
"""
        return vertex, frag

    def _refresh_main_visual_uniform_locations(
        self, program_id: int | None = None
    ) -> None:
        """Refresh cached locations for every dynamic main-visual uniform."""
        prog = int(program_id or getattr(self, "_main_visual_program", 0) or 0)
        if prog <= 0:
            return
        handle = self._get_or_create_program_handle(prog)
        handle.invalidate()
        for name in (
            "u_time", "u_frame", "u_dt", "u_fps", "u_resolution",
            "u_audio", "u_audio_att", "u_audio_bands", "u_zoom", "u_rot",
            "u_warp", "u_center", "u_offset", "u_scale", "u_decay",
            "u_beat", "u_beat_flash", "u_audio_rms", "u_audio_peak",
            "u_rand_seed", "u_fc", "u_aspect", "u_prev_frame", "u_texture",
            "u_megabuf", "u_q", "u_t", "u_regs",
        ):
            location = GL.glGetUniformLocation(prog, name)
            if location >= 0:
                handle.set_location(name, int(location))
        for i in range(1, 33):
            location = GL.glGetUniformLocation(prog, f"u_q[{i}]")
            if location >= 0:
                handle.set_location(f"u_q[{i}]", int(location))
        for i in range(1, 9):
            location = GL.glGetUniformLocation(prog, f"u_t[{i}]")
            if location >= 0:
                handle.set_location(f"u_t[{i}]", int(location))
        for i in range(100):
            location = GL.glGetUniformLocation(prog, f"u_regs[{i}]")
            if location >= 0:
                handle.set_location(f"u_regs[{i}]", int(location))
        self._main_visual_uniforms_refreshed = True

    def _refresh_shader_uniform_locations(
        self, program_id: int | None = None
    ) -> None:
        """Refresh uniform locations for the primary preset shader."""
        prog = int(program_id or getattr(self, "_main_visual_program", 0) or 0)
        if prog <= 0:
            return
        handle = self._get_or_create_program_handle(prog)
        handle.invalidate()
        for name in (
            "u_time", "u_frame", "u_dt", "u_fps", "u_resolution",
            "u_audio", "u_audio_att", "u_audio_bands", "u_zoom", "u_rot",
            "u_warp", "u_center", "u_offset", "u_scale", "u_decay",
            "u_beat", "u_beat_flash", "u_audio_rms", "u_audio_peak",
            "u_rand_seed", "u_fc", "u_aspect", "u_prev_frame", "u_texture",
            "u_megabuf", "u_q", "u_t", "u_regs",
        ):
            location = GL.glGetUniformLocation(prog, name)
            if location >= 0:
                handle.set_location(name, int(location))
        for i in range(1, 33):
            location = GL.glGetUniformLocation(prog, f"u_q[{i}]")
            if location >= 0:
                handle.set_location(f"u_q[{i}]", int(location))
        for i in range(1, 9):
            location = GL.glGetUniformLocation(prog, f"u_t[{i}]")
            if location >= 0:
                handle.set_location(f"u_t[{i}]", int(location))
        for i in range(100):
            location = GL.glGetUniformLocation(prog, f"u_regs[{i}]")
            if location >= 0:
                handle.set_location(f"u_regs[{i}]", int(location))

    def _compile_main_visual_shader(self, candidate_program=None) -> int:
        """Compile a candidate preset shader without replacing the active one."""
        if candidate_program is None:
            candidate_program = getattr(self, "preset_program", None)
        if candidate_program is None:
            logger.error("Cannot compile main visual shader without a preset")
            return 0
        try:
            vertex, fragment = self._build_main_visual_shader_source(candidate_program)
            program_id = _compile_gl_program(
                vertex, fragment, name="main_visual"
            )
            if program_id <= 0:
                logger.error(
                    "Main visual shader compilation failed; generated fragment:\n%s",
                    fragment,
                )
                return 0
            self._refresh_main_visual_uniform_locations(program_id)
            return int(program_id)
        except Exception:
            logger.exception("Failed to build main visual shader")
            return 0

    def load_milk_preset(
        self, milk_path: str | os.PathLike[str], smooth_transition: bool = True
    ) -> PresetLoadResult:
        """Queue a preset until GL is current, or install it transactionally."""
        path = str(milk_path)
        lock = getattr(self, "_preset_request_lock", None)
        if lock is not None:
            lock.acquire()
        try:
            request_id = int(getattr(self, "_preset_request_id", 0) or 0) + 1
            self._preset_request_id = request_id
            if not getattr(self, "opengl_ready", False):
                self._pending_preset_request = PendingPresetRequest(
                    request_id=request_id,
                    path=path,
                    smooth_transition=bool(smooth_transition),
                )
                return PresetLoadResult.PENDING
            smooth = bool(smooth_transition) and getattr(self, "preset_program", None) is not None
        finally:
            if lock is not None:
                lock.release()
        # Parse/compile/GL work happens outside the request lock; commit/start
        # paths recheck request freshness before replacing active state.
        if smooth:
            return self._start_transition(path, request_id=request_id)
        return self._install_preset_immediately(path, request_id=request_id)

    def _process_pending_preset(self) -> None:
        request = getattr(self, "_pending_preset_request", None)
        if request is None or not getattr(self, "opengl_ready", False):
            return
        self._consume_pending_preset()

    def _consume_pending_preset(self) -> None:
        lock = getattr(self, "_preset_request_lock", None)
        if lock is not None:
            lock.acquire()
        try:
            request = getattr(self, "_pending_preset_request", None)
            if request is None:
                return
            phase = getattr(self, "_transition_phase", None)
            if phase is not None and getattr(phase, "value", phase) != "idle":
                return
            if request.request_id != getattr(self, "_preset_request_id", 0):
                return
            limit = PRESET_INSTALL_MAX_ATTEMPTS
            attempts = getattr(self, "_preset_install_attempts", {})
            attempts[request.request_id] = attempts.get(request.request_id, 0) + 1
            self._preset_install_attempts = attempts
            if attempts[request.request_id] > limit:
                self._pending_preset_request = None
                self._preset_install_attempts.pop(request.request_id, None)
                return
            snapshot = self._snapshot_preset_state()
            path = request.path
            smooth = bool(request.smooth_transition) and getattr(self, "preset_program", None) is not None
            req_id = request.request_id
        finally:
            if lock is not None:
                lock.release()

        if smooth:
            result = self._start_transition(path, request_id=req_id)
        else:
            result = self._install_preset_immediately(path, req_id)

        if lock is not None:
            lock.acquire()
        try:
            attempts = getattr(self, "_preset_install_attempts", {})
            count = attempts.get(req_id, 0)
            if result is PresetLoadResult.INSTALLED:
                if getattr(self, "_pending_preset_request", None) is request:
                    self._pending_preset_request = None
                attempts.pop(req_id, None)
            elif result is PresetLoadResult.FAILED:
                self._restore_preset_state(snapshot)
                if count > PRESET_INSTALL_MAX_ATTEMPTS and getattr(self, "_pending_preset_request", None) is request:
                    self._pending_preset_request = None
                    attempts.pop(req_id, None)
            elif result is PresetLoadResult.PENDING:
                # Transition claimed the request; keep attempts for bookkeeping.
                pass
            elif result is PresetLoadResult.STALE:
                if getattr(self, "_pending_preset_request", None) is request:
                    self._pending_preset_request = None
                attempts.pop(req_id, None)
            self._preset_install_attempts = attempts
        finally:
            if lock is not None:
                lock.release()

    def _snapshot_preset_state(self) -> dict[str, object]:
        sentinel = self._PRESET_SNAPSHOT_SENTINEL
        fields = (*self._PRESET_OWNED_FIELDS, *self._PRESET_CACHE_FIELDS)
        snapshot: dict[str, object] = {}
        for name in fields:
            value = getattr(self, name, sentinel)
            if value is sentinel:
                continue
            if isinstance(value, (dict, set, list)):
                import copy

                snapshot[name] = copy.deepcopy(value)
            else:
                snapshot[name] = value
        return snapshot

    def _restore_preset_state(self, snapshot: dict[str, object]) -> None:
        sentinel = self._PRESET_SNAPSHOT_SENTINEL
        for name, value in snapshot.items():
            if value is sentinel:
                if hasattr(self, name):
                    delattr(self, name)
            else:
                setattr(self, name, value)

    def _install_preset_immediately(
        self,
        milk_path: str | os.PathLike[str],
        request_id: int | None = None,
    ) -> PresetLoadResult:
        """Parse and compile a candidate before replacing the active preset."""
        if (
            request_id is not None
            and getattr(self, "_preset_request_id", 0)
            and request_id != self._preset_request_id
        ):
            return PresetLoadResult.STALE
        path = str(milk_path)
        try:
            # Resolve through the compatibility module so callers and existing
            # integrations that patch the public parser keep working.
            import milkdrop_renderer

            parser = getattr(
                milkdrop_renderer, "parse_milk_file_robust", parse_milk_file_robust
            )
            parsed = parser(path)
            preset_program = PresetProgram(parsed.to_dict())
            # Extract warp_ and comp_ backtick-delimited shader code from the parsed preset
            warp_code, comp_code = PresetProgram._extract_warp_comp_code(parsed)
            preset_program.warp_code = warp_code
            preset_program.comp_code = comp_code
            preset_vm = PresetVM(preset_program, getattr(parsed, "params", {}))
            _vertex, fragment = self._build_main_visual_shader_source(preset_program)
        except Exception:
            logger.exception("Failed to parse preset %s", path)
            return PresetLoadResult.FAILED

        candidate_id = 0
        if getattr(self, "opengl_ready", False):
            try:
                candidate_id = self._compile_main_visual_shader(preset_program)
            except Exception:
                logger.exception("Failed to compile preset shader %s", path)
                self._is_transitioning = False
                self._capture_pending = False
                return PresetLoadResult.FAILED
            if candidate_id <= 0 or not GL.glIsProgram(candidate_id):
                if candidate_id > 0:
                    self._retire_program(candidate_id)
                self._is_transitioning = False
                self._capture_pending = False
                logger.error("Preset rejected because its shader did not compile: %s", path)
                return PresetLoadResult.FAILED
        old_program = int(getattr(self, "_main_visual_program", 0) or 0)
        self.preset_program = preset_program
        self.preset_vm = preset_vm
        self._main_visual_program = candidate_id
        self._main_visual_shader_source = fragment
        self._main_visual_has_per_pixel = bool(preset_program.per_pixel)
        self._main_visual_reg_refs = set()
        self.current_milk_file = path
        self.preset_name = Path(path).stem
        self.preset_eqs = dict(getattr(parsed, "params", {}) or {})
        # --- Compile warp and composite shader code from the preset ---
        self._compile_warp_comp_shaders(preset_program, warp_code, comp_code)
        self._is_transitioning = False
        self._capture_pending = False
        multipass = getattr(getattr(self, "_render_graph", None), "multipass", None)
        if multipass is not None:
            multipass.has_valid_frame = False
            multipass.write_fbo_idx = 0
        self._current_source_tex = (
            int(multipass.fbo_tex[0])
            if multipass is not None and multipass.fbo_tex
            else 0
        )
        if old_program > 0 and old_program != candidate_id:
            self._retire_program(old_program)
        logger.info(
            "Installed preset %s: per_frame=%d per_pixel=%d waves=%d shapes=%d",
            self.preset_name,
            len(preset_program.per_frame),
            len(preset_program.per_pixel),
            len(preset_program.waves),
            len(preset_program.shapes),
        )
        return PresetLoadResult.INSTALLED

    def _compile_warp_comp_shaders(
        self, preset_program: PresetProgram, warp_code: str, comp_code: str
    ) -> None:
        """Compile the preset's warp_ and comp_ backtick-delimited shader code.

        The warp shader is compiled into a program used by ``_render_warp_pass``,
        which reads the current frame and applies warp distortion.  The composite
        shader is compiled into a program used by ``_render_composite_pass``,
        which applies bloom/feedback/enhancement effects.

        If either code string is empty, translation is incomplete, or compilation
        fails, the corresponding program is left at 0 and the render pass falls
        back to a blit.
        """
        vert = (
            "#version 330 core\n"
            "layout(location = 0) in vec2 a_pos;\n"
            "out vec2 v_uv;\n"
            "void main() {\n"
            "    v_uv = a_pos * 0.5 + 0.5;\n"
            "    gl_Position = vec4(a_pos, 0.0, 1.0);\n"
            "}\n"
        )

        # --- Warp shader ---
        warp_prog = 0
        if str(warp_code or "").strip():
            try:
                frag = self._translate_warp_code(warp_code)
                if not str(frag or "").strip():
                    logger.info("Warp shader unavailable after translation; using blit fallback")
                else:
                    warp_prog = _compile_gl_program(vert, frag, name="warp")
                    if warp_prog <= 0 or not GL.glIsProgram(warp_prog):
                        logger.warning(
                            "Warp shader compilation failed; warp pass will fall back to blit"
                        )
                        warp_prog = 0
            except Exception:
                logger.exception("Failed to compile warp shader")
                warp_prog = 0
        self._last_warp_program = warp_prog

        # --- Composite shader ---
        comp_prog = 0
        if str(comp_code or "").strip():
            try:
                frag = self._translate_comp_code(comp_code)
                if not str(frag or "").strip():
                    logger.info("Composite shader unavailable after translation; using blit fallback")
                else:
                    comp_prog = _compile_gl_program(vert, frag, name="composite")
                    if comp_prog <= 0 or not GL.glIsProgram(comp_prog):
                        logger.warning(
                            "Composite shader compilation failed; composite pass will fall back to blit"
                        )
                        comp_prog = 0
            except Exception:
                logger.exception("Failed to compile composite shader")
                comp_prog = 0
        self._composite_shader = comp_prog

    def _looks_like_hlsl_shader(self, code: str) -> bool:
        if not code:
            return False
        sample = str(code)
        tokens = (
            "tex2D", "tex2Dlod", "sampler_state", "float2", "float3", "float4",
            "half2", "half3", "half4", "mul(", "lerp(", "saturate(",
        )
        return any(token in sample for token in tokens)

    def _translate_preset_shader_code(self, code: str, *, kind: str) -> str:
        """Translate preset HLSL shader snippets to GLSL when possible.

        If translation cannot produce GLSL-ready output, return an empty string
        so callers fall back to a safe blit instead of compiling broken shaders.
        """
        source = str(code or "").strip()
        if not source:
            return ""
        translated = self._translate_hlsl_fragment(source)
        translated = self._map_sampler_names(translated)
        try:
            translated, _report = self._fix_shader_vector_mismatches(translated)
        except Exception:
            logger.debug("%s shader repair skipped due to error", kind, exc_info=True)
        if self._looks_like_hlsl_shader(translated):
            logger.warning(
                "Skipping %s shader: HLSL translation incomplete; falling back to blit",
                kind,
            )
            return ""
        return translated

    def _translate_warp_code(self, warp_code: str) -> str:
        """Translate a MilkDrop ``warp_`` HLSL body into a complete GLSL fragment shader."""
        try:
            return self._build_translated_pass_shader(warp_code, pass_name="warp")
        except Exception:
            logger.exception("Warp HLSL→GLSL translation failed; using pass-through")
            try:
                return self._build_translated_pass_shader("", pass_name="warp")
            except Exception:  # noqa: BLE001
                return self._translate_preset_shader_code(warp_code, kind="warp")

    def _translate_comp_code(self, comp_code: str) -> str:
        """Translate a MilkDrop ``comp_`` HLSL body into a complete GLSL fragment shader."""
        try:
            return self._build_translated_pass_shader(comp_code, pass_name="comp")
        except Exception:
            logger.exception("Composite HLSL→GLSL translation failed; using pass-through")
            try:
                return self._build_translated_pass_shader("", pass_name="comp")
            except Exception:  # noqa: BLE001
                return self._translate_preset_shader_code(comp_code, kind="composite")

    def _populate_waveform_data(self, vm):
        # Populate classic waveform arrays for audio-reactive waves
        if vm is None:
            return 0
        wave_x = getattr(vm, "wave_x", None)
        wave_y = getattr(vm, "wave_y", None)
        if wave_x is None or wave_y is None:
            return 0
        max_pts = len(wave_x) if hasattr(wave_x, "__len__") else 512
        max_pts = min(max_pts, len(wave_y)) if hasattr(wave_y, "__len__") else max_pts
        if max_pts <= 0:
            return 0

        snapshot = self._copy_audio_snapshot()
        waveform = None
        if isinstance(snapshot, dict):
            waveform = snapshot.get("waveform", snapshot.get("_waveform"))
        if waveform is None:
            audio_data = getattr(self, "audio_data", None)
            if audio_data is None or len(audio_data) == 0:
                wave_x[:max_pts] = np.linspace(0.0, 1.0, max_pts, dtype=np.float32)
                wave_y[:max_pts] = 0.5
                return max_pts
            samples = np.asarray(audio_data, dtype=np.float32).ravel()
            npts = min(max_pts, min(len(samples), 512))
            if npts <= 0:
                wave_x[:max_pts] = np.linspace(0.0, 1.0, max_pts, dtype=np.float32)
                wave_y[:max_pts] = 0.5
                return max_pts
            wave_x[:npts] = np.linspace(0.0, 1.0, npts, dtype=np.float32)
            wave_y[:npts] = np.clip(0.5 + 0.4 * samples[:npts], 0.0, 1.0)
            if npts < max_pts:
                wave_x[npts:max_pts] = 0.0
                wave_y[npts:max_pts] = 0.0
            return npts

        samples = np.asarray(waveform, dtype=np.float32).ravel()
        samples = np.where(np.isfinite(samples), samples, 0.0)
        npts = min(max_pts, len(samples))
        if npts <= 0:
            wave_x[:max_pts] = np.linspace(0.0, 1.0, max_pts, dtype=np.float32)
            wave_y[:max_pts] = 0.5
            return max_pts
        wave_x[:npts] = np.linspace(0.0, 1.0, npts, dtype=np.float32)
        wave_y[:npts] = np.clip(0.5 + 0.4 * samples[:npts], 0.0, 1.0)
        if npts < max_pts:
            wave_x[npts:max_pts] = 0.0
            wave_y[npts:max_pts] = 0.0
        return npts

    def _render_main_visual_pass(
        self, width: int | None = None, height: int | None = None
    ) -> ProgramHandle | None:
        program_id = int(getattr(self, "_main_visual_program", 0) or 0)
        if program_id <= 0 or not GL.glIsProgram(program_id):
            raise RuntimeError(f"Main visual program is invalid: {program_id}")
        graph = getattr(self, "_render_graph", None)
        multipass = getattr(graph, "multipass", None)
        if multipass is None:
            raise RuntimeError("Main visual pass has no multipass feedback targets")
        width = int(width if width is not None else multipass.width)
        height = int(height if height is not None else multipass.height)
        if width != multipass.width or height != multipass.height:
            raise RuntimeError(
                "Main visual target dimensions are stale: "
                f"{multipass.width}x{multipass.height} != {width}x{height}"
            )
        read_idx = int(multipass.write_fbo_idx)
        if read_idx not in (0, 1):
            raise RuntimeError(f"Invalid feedback read index: {read_idx}")
        write_idx = 1 - read_idx
        write_fbo = int(multipass.fbo[write_idx] or 0)
        write_tex = int(multipass.fbo_tex[write_idx] or 0)
        if write_fbo <= 0 or write_tex <= 0:
            raise RuntimeError(f"Invalid feedback write target at index {write_idx}")

        if not getattr(self, "_main_visual_uniforms_refreshed", False):
            self._refresh_main_visual_uniform_locations(program_id)
        handle = self._get_or_create_program_handle(program_id)
        self._gl_state.bind_draw_fbo(write_fbo, force=True)
        self._check_fbo_complete("main visual")
        self._gl_state.set_viewport(0, 0, width, height)
        self._gl_state.set_clear_color(0.0, 0.0, 0.0, 0.0)
        self._gl_state.clear(GL.GL_COLOR_BUFFER_BIT)
        self._gl_state.use_program(program_id)

        if multipass.has_valid_frame:
            feedback_tex = int(multipass.fbo_tex[read_idx] or 0)
            if feedback_tex <= 0 or not GL.glIsTexture(feedback_tex):
                raise RuntimeError(
                    f"Previous feedback texture {feedback_tex} is invalid"
                )
        else:
            feedback_tex = self._ensure_feedback_black_texture()
        self._gl_state.active_texture(0)
        self._gl_state.bind_texture(feedback_tex, unit=0)
        handle.upload_uniform("u_prev_frame", 0, self._gl_state)
        vm = getattr(self, "preset_vm", None)
        if vm is not None:
            # Rule #9: Update all dynamic uniforms every render frame
            # Per AGENTS.md: never leave animation uniforms permanently initialized
            # at startup; each render frame must upload fresh values.
            self._upload_vm_uniforms(program_id=program_id, vm=vm)
            # _upload_vm_uniforms is the canonical per-frame upload path.
            # Do not upload a second set of defaults here: that can overwrite
            # live VM values (especially q/t registers and audio uniforms).

        vao = int(
            getattr(self, "_main_visual_vao", 0)
            or getattr(self, "_composite_vao", 0)
            or 0
        )
        if vao <= 0 or not GL.glIsVertexArray(vao):
            raise RuntimeError("Main visual VAO is invalid")
        self._gl_state.bind_vertex_array(vao)
        try:
            GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
            self._check_draw_error("main visual")
        except (GL.GLError, _GLPyError.Error) as exc:
            logger.error("Main visual draw failed: %s", exc)
            raise
        finally:
            self._gl_state.bind_vertex_array(0)

        multipass.write_fbo_idx = write_idx
        multipass.has_valid_frame = True
        self._current_source_tex = write_tex
        self._has_valid_frame = True
        return handle

    @staticmethod
    def _coerce_audio_scalar(value, default: float = 0.0) -> float:
        if value is None:
            return float(default)
        try:
            fv = float(value)
        except (TypeError, ValueError):
            return float(default)
        if math.isnan(fv) or math.isinf(fv):
            return float(default)
        return fv

    def _sanitise_audio_array(self, value):
        try:
            arr = np.asarray(value, dtype=np.float32)
        except Exception:  # noqa: BLE001
            return None
        if arr.size == 0:
            return np.zeros(0, dtype=np.float32)
        arr = arr.copy()
        arr[~np.isfinite(arr)] = 0.0
        return arr.astype(np.float32, copy=False)

    def _publish_audio_snapshot(self, snapshot: dict[str, object]) -> None:
        if not isinstance(snapshot, dict):
            return
        snapshot = dict(snapshot)
        for key in self._AUDIO_LEVEL_KEYS:
            snapshot.setdefault(key, 0.0)
        try:
            if not snapshot.get("vol"):
                snapshot["vol"] = (self._coerce_audio_scalar(snapshot.get("bass", 0.0)) + self._coerce_audio_scalar(snapshot.get("mid", 0.0)) + self._coerce_audio_scalar(snapshot.get("treb", 0.0))) / 3.0
            if not snapshot.get("vol_att"):
                snapshot["vol_att"] = (self._coerce_audio_scalar(snapshot.get("bass_att", 0.0)) + self._coerce_audio_scalar(snapshot.get("mid_att", 0.0)) + self._coerce_audio_scalar(snapshot.get("treb_att", 0.0))) / 3.0
        except Exception:  # noqa: BLE001, S110
            pass
        snapshot["_timestamp"] = _monotonic()
        for name, canon in (("spectrum", "_spectrum"), ("waveform", "_waveform")):
            candidate = snapshot.get(name, snapshot.get(canon))
            if candidate is None:
                continue
            cleaned = self._sanitise_audio_array(candidate)
            if cleaned is None:
                continue
            snapshot[name] = cleaned
            snapshot[canon] = cleaned
        try:
            lock = getattr(self, "_audio_snapshot_lock", None)
            if lock is None:
                import threading
                lock = threading.RLock()
                self._audio_snapshot_lock = lock
            with lock:
                self._latest_audio_snapshot = snapshot
            vm = getattr(self, "preset_vm", None)
            state = getattr(vm, "state", None) if vm is not None else None
            if state is not None and hasattr(state, "update_audio"):
                state.update_audio(snapshot)
        except Exception:  # noqa: BLE001
            self._latest_audio_snapshot = snapshot

    def _analyse_pcm_to_levels(self, pcm, sample_rate=44100.0):
        """Analyse PCM audio into bass/mid/treb/beat/rms/peak levels.

        Beat gating uses the monotonic clock so that NTP or manual wall-clock
        adjustments cannot permanently lock out beat detection.
        """
        import numpy as np

        now_time = _monotonic()
        try:
            arr = np.asarray(pcm, dtype=np.float32).reshape(-1)
        except Exception:  # noqa: BLE001
            arr = np.zeros(1, dtype=np.float32)
        if arr.size == 0:
            arr = np.zeros(1, dtype=np.float32)
        rms = float(np.sqrt(np.mean(arr ** 2))) if arr.size else 0.0
        peak = float(np.max(np.abs(arr))) if arr.size else 0.0
        # Simple band split: bass = low-freq energy, treb = high-freq energy.
        bass = float(np.mean(np.abs(arr[: max(1, arr.size // 4)]))) if arr.size else 0.0
        treb = float(np.mean(np.abs(arr[-max(1, arr.size // 4):]))) if arr.size else 0.0
        mid = float(np.mean(np.abs(arr))) if arr.size else 0.0
        prev = 0.0
        try:
            prev = float(getattr(self, "_pcm_beat_level", 0.0) or 0.0)
        except Exception:  # noqa: BLE001
            prev = 0.0
        smooth = float(getattr(self, "BEAT_SMOOTHING_FACTOR", 0.93) or 0.93)
        smooth = min(max(smooth, 0.0), 0.999)
        raw_beat = 1.0 if (rms > 0.15 and now_time - float(getattr(self, "_last_beat_time", 0.0) or 0.0) > 0.1) else 0.0
        if raw_beat > 0.0:
            self._last_beat_time = now_time
            beat = 1.0
        else:
            beat = prev * smooth
        self._pcm_beat_level = beat
        return {
            "bass": bass,
            "mid": mid,
            "treb": treb,
            "rms": rms,
            "peak": peak,
            "beat": beat,
            "beat_flash": raw_beat,
        }

    def update_audio(self, audio_levels, channels=2):
        try:
            if isinstance(audio_levels, dict):
                out: dict[str, object] = {key: 0.0 for key in self._AUDIO_LEVEL_KEYS}
                for key, value in audio_levels.items():
                    if value is None:
                        continue
                    if key in ("spectrum", "_spectrum", "waveform", "_waveform", "pcm"):
                        out[key] = value
                        continue
                    try:
                        import numpy as _np
                        if isinstance(value, _np.ndarray):
                            out[key] = value
                            continue
                    except Exception:  # noqa: BLE001, S110
                        pass
                    out[key] = self._coerce_audio_scalar(value)
                if "spectrum" in audio_levels and audio_levels["spectrum"] is not None:
                    arr = self._sanitise_audio_array(audio_levels["spectrum"])
                    if arr is not None:
                        out["_spectrum"] = arr
                if "waveform" in audio_levels and audio_levels["waveform"] is not None:
                    arr = self._sanitise_audio_array(audio_levels["waveform"])
                    if arr is not None:
                        out["_waveform"] = arr
                if "_spectrum" in audio_levels and audio_levels["_spectrum"] is not None:
                    arr = self._sanitise_audio_array(audio_levels["_spectrum"])
                    if arr is not None:
                        out["_spectrum"] = arr
                if "_waveform" in audio_levels and audio_levels["_waveform"] is not None:
                    arr = self._sanitise_audio_array(audio_levels["_waveform"])
                    if arr is not None:
                        out["_waveform"] = arr
                self._publish_audio_snapshot(out)
                return

            if isinstance(audio_levels, (list, tuple, np.ndarray)):
                arr = np.asarray(audio_levels, dtype=np.float32)
                if arr.ndim == 0:
                    return
                if channels == 1:
                    if not hasattr(self, "audio_data") or self.audio_data is None:
                        self.audio_data = np.zeros(1024, dtype=np.float32)
                    data = arr.reshape(-1)
                    lock = getattr(self, "_audio_pcm_lock", None)
                    if lock is None:
                        lock = threading.RLock()
                        self._audio_pcm_lock = lock
                    with lock:
                        n = min(data.size, self.audio_data.size)
                        if n > 0:
                            self.audio_data[-n:] = data[:n]
                    return

                if not hasattr(self, "audio_data_stereo") or self.audio_data_stereo is None:
                    self.audio_data_stereo = np.zeros((1024, 2), dtype=np.float32)
                if not hasattr(self, "audio_data") or self.audio_data is None:
                    self.audio_data = np.zeros(1024, dtype=np.float32)
                data = np.asarray(audio_levels, dtype=np.float32).reshape(-1)
                if data.size % 2 != 0:
                    data = data[:-1]
                if data.size == 0:
                    return
                stereo = data.reshape(-1, 2)
                lock = getattr(self, "_audio_pcm_lock", None)
                if lock is None:
                    lock = threading.RLock()
                    self._audio_pcm_lock = lock
                with lock:
                    n = min(stereo.shape[0], self.audio_data_stereo.shape[0])
                    if n > 0:
                        self.audio_data_stereo[-n:] = stereo[-n:]
                    mono = (stereo[:, 0] + stereo[:, 1]) * 0.5
                    n_mono = min(mono.size, self.audio_data.size)
                    if n_mono > 0:
                        self.audio_data[-n_mono:] = mono[:n_mono]
                return

            # Scalar or unsupported payloads are ignored; preserve the last snapshot.
            return
        except Exception:  # noqa: BLE001, S110
            pass

    def _copy_audio_snapshot(self) -> dict[str, object] | None:
        lock = getattr(self, "_audio_snapshot_lock", None)
        if lock is not None:
            try:
                with lock:
                    snapshot = getattr(self, "_latest_audio_snapshot", None)
                    return dict(snapshot) if isinstance(snapshot, dict) else None
            except Exception:  # noqa: BLE001
                snapshot = getattr(self, "_latest_audio_snapshot", None)
                return dict(snapshot) if isinstance(snapshot, dict) else None
        snapshot = getattr(self, "_latest_audio_snapshot", None)
        return dict(snapshot) if isinstance(snapshot, dict) else None

    def _get_vm_audio_levels(self, vm: object | None = None) -> AudioLevels:
        # Prefer latest audio snapshot if available
        snapshot = self._copy_audio_snapshot()
        if snapshot is not None:
            out: AudioLevels = {}
            for k in ("bass", "mid", "treb", "bass_att", "mid_att", "treb_att", "beat", "beat_flash", "rms", "peak"):
                if k in snapshot and snapshot.get(k) is not None:
                    try:
                        raw = snapshot.get(k)
                        if raw is None:
                            continue
                        try:
                            v = float(raw)  # type: ignore[arg-type]
                        except (TypeError, ValueError):
                            out[k] = 0.0
                            continue
                        if math.isnan(v) or math.isinf(v):
                            v = 0.0
                        out[k] = v
                    except Exception:  # noqa: BLE001
                        out[k] = 0.0
            return out
        # Fallback to vm state
        vm_obj = vm if vm is not None else getattr(self, "preset_vm", None)
        if vm_obj is None:
            return {}
        try:
            state = getattr(vm_obj, "state", None)
            out: AudioLevels = {}
            for k in ("bass", "mid", "treb", "bass_att", "mid_att", "treb_att", "beat", "beat_flash", "rms", "peak"):
                v = None
                if state is not None:
                    if hasattr(state, "get"):
                        try:
                            v = state.get(k)
                        except Exception:  # noqa: BLE001
                            v = None
                    elif isinstance(state, dict):
                        v = state.get(k)
                if v is not None:
                    try:
                        fv = float(v)
                        if math.isnan(fv) or math.isinf(fv):
                            fv = 0.0
                        out[k] = fv
                    except Exception:  # noqa: BLE001
                        out[k] = 0.0
            return out
        except Exception:  # noqa: BLE001
            return {}
    def _validate_glsl_source(self, glsl_source: str) -> bool:
        import re

        # Check for balanced braces
        brace_count = glsl_source.count('{') - glsl_source.count('}')
        if brace_count != 0:
            logger.warning("GLSL source has unbalanced braces: count=%s", brace_count)
            return False

        # Check for supported math functions
        supported_math_funcs = {
            'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh',
            'exp', 'log', 'log10', 'sqrt', 'abs', 'floor', 'ceil', 'mod', 'pow',
            'min', 'max', 'sign', 'inversesqrt'
        }

        func_calls = re.findall(r'(\w+)\s*\(', glsl_source)
        for func in func_calls:
            if func not in supported_math_funcs and not func.startswith('u_') and not func.startswith('v_'):
                logger.warning("Unknown GLSL function: %s in preset shader", func)

        # Check for potential NaN/Inf issues
        if re.search(r'sqrt\s*\(\s*[^)]*-\s*[^)]*\)', glsl_source):
            logger.warning("GLSL source has sqrt of potentially negative expression")

        return True

    def _validate_compiled_program(self, program_handle, vm_state: PresetVM) -> bool:
        from OpenGL import GL

        # Check linking status
        status = GL.glGetProgramiv(program_handle, GL.GL_LINK_STATUS)
        if not status:
            log = GL.glGetProgramInfoLog(program_handle)
            logger.error("Shader program linking failed: %s", log)
            return False

        # Log program info for debugging
        num_active_uniforms = GL.glGetProgramiv(program_handle, GL.GL_ACTIVE_UNIFORMS)
        logger.debug("Compiled program has %s active uniforms", num_active_uniforms)

        return True

    def _update_vm_uniforms_from_vm(
        self, program_handle: ProgramHandle, vm_state: PresetVM
    ) -> None:
        """Update renderer uniforms from PresetVM state.

        Rule #8: Keep q1..q32, zoom, rot, warp, cx, cy, dx, dy, and other
        MilkDrop variables semantically consistent between CPU VM and generated
        GPU code.

        Args:
            program_handle: The active program's ProgramHandle (used for
                generation-safe uniform uploads).
            vm_state: The PresetVM state containing the preset variables.
        """
        gl_state = self._gl_state
        # Rule #8: Update q1..q32 registers
        q_vars = getattr(vm_state, 'q', [])
        if q_vars:
            try:
                import numpy as np
                program_handle.upload_uniform_1fv(
                    'u_q', np.asarray([0.0, *q_vars], dtype=np.float32), gl_state
                )
            except Exception:  # noqa: BLE001
                logger.debug("Failed to upload q uniforms")

        # Rule #8: Update zoom, rot, warp, cx, cy, dx, dy
        # u_zoom/u_rot/u_warp are declared as scalar `uniform float` in the
        # shader, so they are uploaded via upload_uniform (glUniform1f),
        # NOT upload_uniform_1fv (which expects an array and would call
        # len() on the scalar — TypeError: object of type 'float' has no len()).
        program_handle.upload_uniform(
            'u_zoom', float(getattr(vm_state, 'zoom', 1.0)), gl_state
        )
        program_handle.upload_uniform(
            'u_rot', float(getattr(vm_state, 'rot', 0.0)), gl_state
        )
        program_handle.upload_uniform(
            'u_warp', float(getattr(vm_state, 'warp', 0.0)), gl_state
        )
        program_handle.upload_uniform(
            'u_center',
            (float(getattr(vm_state, 'cx', 0.5)), float(getattr(vm_state, 'cy', 0.5))),
            gl_state,
        )
        program_handle.upload_uniform(
            'u_offset',
            (float(getattr(vm_state, 'dx', 0.0)), float(getattr(vm_state, 'dy', 0.0))),
            gl_state,
        )

        # Rule #10: Update audio-related uniforms
        audio_bands = getattr(vm_state, 'audio_bands', None)
        if audio_bands is not None:
            try:
                import numpy as np
                bands = np.asarray(audio_bands, dtype=np.float32)
                if bands.size > 0:
                    program_handle.upload_uniform_1fv(
                        'u_audio_bands', bands, gl_state
                    )
            except Exception:  # noqa: BLE001
                logger.debug("Failed to upload audio bands")

        # Rule #11: Update decay and brightness
        # u_decay/u_brightness are scalar `uniform float` in the shader
        # (see shaders/composite_shader.frag), so use upload_uniform
        # (glUniform1f), not upload_uniform_1fv (which expects an array).
        decay = float(getattr(vm_state, 'decay', 0.9))
        program_handle.upload_uniform('u_decay', decay, gl_state)

        brightness = float(getattr(vm_state, 'brightness', 1.0))
        program_handle.upload_uniform('u_brightness', brightness, gl_state)

    def _begin_preset_transition(self, preset_data: dict, program_handle, vm_state):
        """Begin the preset transition from old to new preset.

        Rule #6: This is called only after the new preset has been successfully
        compiled and validated. The old preset continues rendering until the
        transition completes.

        Sequence (Rule #6):
        - Set the new program as the current program
        - Update VM state uniforms (q1..q32, zoom, rot, warp, cx, cy, dx, dy, etc.)
        - Mark that we're transitioning
        - Initialize transition state
        """
        # Set the new program as the current program
        self._current_program = program_handle

        # Update VM state uniforms
        self._update_vm_uniforms_from_vm(program_handle, vm_state)

        # Mark that we're transitioning
        self._is_transitioning = True
        self._capture_pending = False

        # Initialize transition state
        self._transition_phase = TransitionPhase.CAPTURE_SOURCE
        self._transition_state.active = True
        self._transition_state.start_time = time.monotonic()
        self._transition_state.progress = 0.0

        logger.info("Beginning preset transition to: %s", preset_data.get("name", "unknown"))

    def _map_sampler_names(self, code: str) -> str:
        try:
            import re
            # Replace whole identifiers only
            code = re.sub(r'\bsampler_main\b', 'u_prev_frame', code)
            code = re.sub(r'\bsampler_pic\b', 'sampler_prayerwheel', code)
            return code
        except Exception:  # noqa: BLE001
            return code

    def _fix_shader_vector_mismatches_v2(self, code: str, report=None):
        try:
            import re
            # Fix pow(vecN, scalar) -> pow(vecN, vecN(scalar))
            pattern = r'pow\(\s*(vec[2-4]\([^()]*(?:\([^()]*\)[^()]*)*\))\s*,\s*([^)]+?)\s*\)'
            def fix_pow2(m):
                vec_call = m.group(1)
                exp = m.group(2).strip()
                if re.match(r'^vec[2-4]\s*\(', exp):
                    return m.group(0)
                if vec_call.strip().startswith('vec2'):
                    return f'pow({vec_call}, vec2({exp}))'
                elif vec_call.strip().startswith('vec3'):
                    return f'pow({vec_call}, vec3({exp}))'
                elif vec_call.strip().startswith('vec4'):
                    return f'pow({vec_call}, vec4({exp}))'
                return m.group(0)
            code = re.sub(pattern, fix_pow2, code)
            return code, report
        except Exception:  # noqa: BLE001
            return code, report
