"""Shared member declarations for the renderer's mixin classes.

The renderer implementation is split across several modules.  The concrete
``MilkdropRenderer`` class in :mod:`renderer.core` owns the runtime
implementation of the members declared here, while the mixins in
:mod:`renderer.passes` and :mod:`renderer.textures` reach those members
through this base class so that cross-module access resolves statically.

The declarations are limited to the members the mixins actually use.  Pure
contract methods are marked abstract so an incomplete subclass fails at
instantiation time rather than silently doing nothing at runtime.
``_run_preset_vm`` is intentionally the only concrete method: it centralizes
the common "VM → uniforms" path (AGENTS.md §8: audio/input → VM state →
shader uniforms → GPU) so mixins and the core share one implementation.
"""

from __future__ import annotations

import logging
import os
from abc import abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Self, TypedDict

logger = logging.getLogger(__name__)

from gl_state import GLResourceOwner, GLStateManager
from program_handle import ProgramHandle
from resources import (
    PendingPresetRequest,
    PresetLoadResult,
    TransitionPhase,
    TransitionResources,
    TransitionState,
)

from .config import CLEAR_COLOR_BLACK
from .textures import TextureLRU

if TYPE_CHECKING:
    from .presets import PresetProgram, PresetVM
    from .recovery import RecoveryManager


class AudioLevels(TypedDict, total=False):
    """Audio level snapshot consumed by the VM and GPU uniforms.

    Values are normalized/processed levels published by the audio pipeline.
    All keys are optional because snapshots may be partially populated.
    """

    bass: float
    mid: float
    treb: float
    bass_att: float
    mid_att: float
    treb_att: float
    beat: float
    beat_flash: float
    rms: float
    peak: float


class RendererBase:
    """Attribute and method contract shared by the renderer mixins.

    Pure contract methods carry ``@abstractmethod`` markers; incomplete
    concrete subclasses fail at instantiation.  This class avoids
    ``abc.ABC``/``ABCMeta`` so it composes cleanly with GTK/GObject metaclasses
    in the final renderer class.
    """

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        abstracts = {
            name
            for name in dir(cls)
            if getattr(getattr(cls, name, None), "__isabstractmethod__", False)
        }
        if abstracts:
            cls.__abstractmethods__ = frozenset(abstracts)
        elif "__abstractmethods__" in cls.__dict__:
            del cls.__abstractmethods__

    def __new__(cls, *args: object, **kwargs: object) -> Self:
        abstracts = getattr(cls, "__abstractmethods__", frozenset())
        if abstracts:
            raise TypeError(
                f"Can't instantiate abstract class {cls.__name__} "
                f"with abstract methods {', '.join(sorted(abstracts))}"
            )
        return super().__new__(cls)

    # --- GL / resource ownership ------------------------------------------
    _gl_resource_owner: GLResourceOwner
    _gl_state: GLStateManager

    # --- Texture cache ----------------------------------------------------
    loaded_textures: TextureLRU

    # --- Preset program / VM handles --------------------------------------
    _pending_preset_request: PendingPresetRequest | None
    _transition_pending_program: PresetProgram | None
    _transition_pending_vm: PresetVM | None
    _transition_pending_preset_dict: Mapping[str, object]
    _transition_target_path: str | None
    preset_vm: PresetVM | None
    preset_program: PresetProgram | None

    # --- Runtime frame state ----------------------------------------------
    _current_time: float
    _current_frame_id: int
    _current_dt: float

    # --- Transition state -------------------------------------------------
    _transition_state: TransitionState
    _transition_resources: TransitionResources
    _transition_phase: TransitionPhase
    _transition_duration: float

    # --- EnhancementLayer audio-reactive state ---------------------------
    # Enhancement-related; a narrower mixin base may own these later.
    _brightness_boost_level: float

    # --- GL handles -------------------------------------------------------
    _composite_vao: int
    _screen_blit_program: int
    _screen_blit_vao: int
    _screen_blit_vbo: int
    _transition_blend_program: int
    _wave_shape_program: int
    wave_vao: int
    wave_vbo: int
    shape_vaos: list[int]
    shape_vbos: list[int]

    # --- Renderer policy constants ----------------------------------------
    CLEAR_COLOR_BLACK: ClassVar[tuple[float, float, float, float]] = CLEAR_COLOR_BLACK

    # --- Recovery state ---------------------------------------------------
    _recovery: RecoveryManager

    # --- Methods implemented by MilkdropRenderer --------------------------
    @classmethod
    def create_headless(cls, *, auto_switch: bool = True) -> RendererBase:
        """Create a headless renderer instance for logic-level use."""
        raise NotImplementedError

    def _init_renderer_state(self, *, auto_switch: bool = True) -> None:
        """Initialize renderer-owned runtime state."""
        fn = getattr(super(), "_init_renderer_state", None)
        if callable(fn):
            fn(auto_switch=auto_switch)

    def _init_async_texture_loader(self) -> None:
        """Initialize async texture loader state."""
        fn = getattr(super(), "_init_async_texture_loader", None)
        if callable(fn):
            fn()

    @abstractmethod
    def _check_fbo_complete(self, label: str = "") -> None:
        """Validate that the currently-bound framebuffer object is complete.

        Only user-created FBOs are checked — the default window framebuffer
        (FBO 0) is owned by the window system and may report
        ``GL_FRAMEBUFFER_UNDEFINED`` on some Mesa EGL/GLX backends even though
        it is valid and usable.  See AGENTS.md §3 (FBO invariants).

        Concrete implementation provided by ``MilkdropRenderer``.
        """
        ...

    @abstractmethod
    def _check_draw_error(self, context: str = "") -> None:
        """Check for GL errors after a draw call in *context*.

        Concrete implementation provided by ``MilkdropRenderer``.
        """
        ...

    @abstractmethod
    def _compile_main_visual_shader(
        self, candidate_program: PresetProgram | None = None
    ) -> int:
        """Compile and link the main visual shader from *candidate_program*.

        Returns the GL program id, or 0 on failure.  Must validate required
        resources before returning so the caller can safely use the program.

        Concrete implementation provided by ``MilkdropRenderer``.
        """
        ...

    @abstractmethod
    def _get_or_create_program_handle(self, program_id: int) -> ProgramHandle:
        """Return (or lazily create) the cached ProgramHandle for *program_id*.

        The handle caches uniform locations keyed by ``location_generation``
        so that stale locations from a previous program are never reused
        after a shader recompile.

        Concrete implementation provided by ``MilkdropRenderer``.
        """
        ...

    @abstractmethod
    def _get_vm_audio_levels(self, vm: PresetVM | None = None) -> AudioLevels:
        """Extract current audio levels, preferring ``_latest_audio_snapshot``.

        The renderer atomically publishes an audio snapshot (bass, mid, treb,
        bass_att, mid_att, treb_att, beat, beat_flash, rms, peak) under
        ``_audio_snapshot_lock`` via :meth:`_publish_audio_snapshot`.
        ``_get_vm_audio_levels`` reads that snapshot so the VM and GPU receive
        a consistent, non-partially-updated set of levels (AGENTS.md §10 keeps
        raw band level, smoothed level, attack response, beat detection, and
        beat flash as separate concepts).  When the snapshot is stale or
        absent, the method falls back to the VM state (or an empty dict).

        Concrete implementation provided by ``MilkdropRenderer``.
        """
        ...

    @abstractmethod
    def _install_preset_immediately(
        self, milk_path: str | os.PathLike[str] | Any, request_id: int | None = None
    ) -> PresetLoadResult:
        """Parse, compile and install a preset synchronously (atomic preset switch).

        The previous preset is never destroyed before the replacement has been
        validated (AGENTS.md §6).  Returns a :class:`PresetLoadResult`
        describing the outcome; on failure the current preset continues
        rendering and transition state is left inactive.

        Concrete implementation provided by ``MilkdropRenderer``.
        """
        ...

    @abstractmethod
    def _on_screen_draw_failed(
        self, err: BaseException | object, bound: int | object, context: str = ""
    ) -> None:
        """Handle a draw failure during a render pass in *context*.

        Concrete implementation provided by ``MilkdropRenderer``.
        """
        ...

    @abstractmethod
    def _populate_waveform_data(self, vm: PresetVM) -> int:
        """Upload the waveform audio data buffer to the shader for audio reactivity.

        Concrete implementation provided by ``MilkdropRenderer``.
        """
        ...

    @abstractmethod
    def _refresh_main_visual_uniform_locations(
        self, program_id: int | None = None
    ) -> None:
        """Re-resolve uniform locations for the current main visual program.

        Called after a shader recompile so cached locations match the new
        program.  Sets ``_main_visual_uniforms_refreshed`` to True.

        Concrete implementation provided by ``MilkdropRenderer``.
        """
        ...

    @abstractmethod
    def _vm_uniforms_are_current(self, program_id: int) -> bool:
        """Return True when VM uniforms have been uploaded for *program_id* this frame.

        Concrete implementation provided by ``MilkdropRenderer``.
        """
        ...

    @abstractmethod
    def _retire_program(self, prog_id: int, force: bool = False) -> None:
        """Queue a program for deferred deletion, releasing cached handles.

        Concrete implementation provided by ``MilkdropRenderer``.
        """
        ...

    def _run_preset_vm(
        self,
        vm: PresetVM | None = None,
        program_id: int | None = None,
        dt: float | None = None,
        width: float = 1.0,
        height: float = 1.0,
    ) -> None:
        """Execute one frame of the preset VM and upload uniforms to the GPU.

        Concrete implementation provided here; subclasses should not override
        unless they have a strong reason.  This is intentionally the only
        non-abstract method in the base so the common "VM → uniforms" path
        stays centralized (AGENTS.md §8).

        Guards on ``vm`` and ``program_id`` — returns early when both are None
        (the normal pre-preset path).  When either is provided (including the
        transition target path which passes ``program_id`` directly), the VM
        ``run_frame`` is invoked and uniforms are uploaded to the active
        program.  PresetVM.run_frame requires ``dt``; the call site always
        supplies it (defaulting from ``_current_dt``).

        Errors during VM execution or uniform upload are logged but do not
        crash the render loop.  The renderer preserves the last known-good
        state and continues rendering.
        """
        vm_obj = vm if vm is not None else self.preset_vm
        if vm_obj is None:
            if program_id is None and vm is None and getattr(self, "preset_program", None) is None:
                logger.debug("_run_preset_vm no-op: no VM or program available")
            return
        dt_val = dt if dt is not None else getattr(self, "_current_dt", 1.0 / 60.0)
        try:
            vm_obj.run_frame(
                time=self._current_time,
                frame=self._current_frame_id,
                audio_levels=self._get_vm_audio_levels(vm_obj),
                width=width,
                height=height,
                dt=dt_val,
            )
        except Exception:
            logger.exception("Preset VM execution failed")
            return
        try:
            self._upload_vm_uniforms(program_id=program_id, vm=vm_obj)
        except Exception:
            logger.exception("Preset VM uniform upload failed")

    @abstractmethod
    def _safe_draw_to_current_fbo(self, context: str = "") -> bool:
        """Draw the fullscreen quad to the currently-bound FBO, returning success.

        Concrete implementation provided by ``MilkdropRenderer``.
        """
        ...

    @abstractmethod
    def _setup_interleaved_vao_attribs(self, vbo_id: int) -> None:
        """Configure interleaved vertex attributes (position + texcoord) for the fullscreen quad.

        Concrete implementation provided by ``MilkdropRenderer``.
        """
        ...

    @abstractmethod
    def _upload_vm_uniforms(
        self,
        program_id: int | None = None,
        vm: PresetVM | object | None = None,
    ) -> None:
        """Upload all VM-derived uniforms to the shader.

        AGENTS.md §9 requires that dynamic uniforms are updated every render
        frame — time, audio bands, beat state, resolution, feedback textures,
        and preset parameters must all be current.

        Concrete implementation provided by ``MilkdropRenderer``.
        """
        ...
