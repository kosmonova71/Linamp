"""Pure data types for the MilkDrop renderer.

This module owns the **state and configuration dataclasses** that have no
dependency on OpenGL, GTK, or the renderer instance:

* audio-reactive visual tuning (BrightnessResponse, FeedbackResponse, …),
* CompositorTuning / EnhancementLayer (renderer-owned post-processing),
* ResourceOwnership (OWNED/BORROWED ownership metadata for the render graph),
* TextureAnimationSpeedUnits, RenderTarget, AudioUniformTuning,
* RendererMode / RendererConfig / RendererSettings (renderer configuration),
* VariableDef / VARIABLES / VAR_REGISTRY / Q_LOOKUP / T_LOOKUP (VM variable registry),
* MultipassResources / PresentedResources / BloomResources /
  AdvancedResources / RenderGraph / RenderPassContext (render-graph containers),
* TransitionState / TransitionResources (transition state),
* PendingPresetRequest / RetryState (preset load bookkeeping),
* FrameResult / TransitionPhase / FrameContext (frame and transition enums),
* RecoveryState / RecoveryOperation / PresetLoadResult / TextureLoadState (state enums),
* ProgramResources (compiled-program resource bundle).

The module is intentionally free of GPU / windowing side-effects at import
time: importing it must never touch the GPU, open a window, or parse a
preset file.  The only work performed at import is pure construction of the
VARIABLES / VAR_REGISTRY lookup tables.

``milkdrop_renderer`` re-exports every name listed in ``__all__`` so that
existing ``from milkdrop_renderer import X`` references resolve unchanged.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from renderer.presets import PresetProgram

if TYPE_CHECKING:
    from renderer.presets import PresetProgram

__all__ = [
    # Constants
    "AUTO_SWITCH_DEFAULT_DURATION",
    "AUTO_SWITCH_MIN_INTERVAL",
    "FRAME_TIME_BASE",
    "Q_LOOKUP",
    "T_LOOKUP",
    # Variable registry
    "VARIABLES",
    "VAR_REGISTRY",
    "AdvancedResources",
    "AudioUniformTuning",
    "BloomResources",
    "BloomResponse",
    "BlurResponse",
    # Audio-reactive response types
    "BrightnessResponse",
    "CompositeResponse",
    # Compositor / enhancement
    "CompositorTuning",
    "EnhancementLayer",
    "FeedbackResponse",
    "FrameContext",
    # Frame / phase enums
    "FrameResult",
    "GlitchResponse",
    "KaleidoscopeResponse",
    "LensResponse",
    "MotionBlurResponse",
    # Render-graph resources
    "MultipassResources",
    # Preset-load bookkeeping
    "PendingPresetRequest",
    "PresentedResources",
    "PresetLoadResult",
    # Compiled program resources
    "ProgramResources",
    "RecoveryOperation",
    # Recovery enums
    "RecoveryState",
    "RenderGraph",
    "RenderPassContext",
    # Texture / render target / tuning
    "RenderTarget",
    "RendererConfig",
    # Renderer config
    "RendererMode",
    "RendererSettings",
    # Ownership & units
    "ResourceOwnership",
    "RetryState",
    "TextureAnimationSpeedUnits",
    "TextureLoadState",
    "TransitionPhase",
    "TransitionResources",
    # Transition
    "TransitionState",
    "VariableDef",
]

# ---------------------------------------------------------------------------
# Renderer policy constants
# ---------------------------------------------------------------------------

#: Minimum interval (seconds) between automatic preset switches.
AUTO_SWITCH_MIN_INTERVAL: float = 0.1

#: Default duration (seconds) for the preset transition blend animation.
AUTO_SWITCH_DEFAULT_DURATION: float = 1.5  # seconds for preset transition blend

#: Frame-normalization baseline for evolutionary texture animation.
#: ``_update_texture_animation`` scales per-frame deltas by dt / FRAME_TIME_BASE
#: so animation speed is effectively tied to a 60 FPS reference rather than
#: racing or crawling at non-standard refresh rates.
FRAME_TIME_BASE: float = 1.0 / 60.0  # seconds, exact 60 FPS reference

# ---------------------------------------------------------------------------
# Audio-reactive response data classes
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class BrightnessResponse:
    """Audio-reactive brightness behaviour driven by the beat."""

    enabled: bool = True
    decay: float = 0.93  # per-frame multiplicative decay toward baseline
    beat_boost: float = 0.4  # brightness = baseline + beat * beat_boost
    beat_threshold: float = 0.5  # beat level that triggers the beat floor
    beat_floor: float = 1.4  # minimum brightness while a strong beat is active


@dataclass(slots=True, frozen=True)
class FeedbackResponse:
    """Audio-reactive feedback decay (bass tightens the persistence)."""

    enabled: bool = True
    bass_dip: float = 0.12  # feedback_amount -= bass * bass_dip
    min_amount: float = 0.80
    max_amount: float = 0.99


@dataclass(slots=True, frozen=True)
class BloomResponse:
    """Audio-reactive bloom intensity."""

    enabled: bool = True
    threshold: float = 0.25
    intensity_base: float = 1.2
    intensity_bass: float = 0.8
    intensity_beat: float = 0.7


@dataclass(slots=True, frozen=True)
class BlurResponse:
    """Audio-reactive blur radius per bloom octave."""

    enabled: bool = True
    radii: tuple[float, ...] = (2.0, 5.0, 12.0)
    audio_boost: float = 3.0  # radius += bass * audio_boost * (octave + 1)


@dataclass(slots=True, frozen=True)
class MotionBlurResponse:
    """Audio-reactive motion-blur strength."""

    enabled: bool = True
    audio_exponent: float = 1.5
    audio_scale: float = 4.0


@dataclass(slots=True, frozen=True)
class LensResponse:
    """Audio-reactive lens-distortion strength."""

    enabled: bool = True
    audio_exponent: float = 2.0
    audio_scale: float = 3.0


@dataclass(slots=True, frozen=True)
class KaleidoscopeResponse:
    """Audio-reactive kaleidoscope spin speed."""

    enabled: bool = True
    base_speed: float = 0.01
    bass_speed: float = 0.1
    treb_speed: float = 0.05


@dataclass(slots=True, frozen=True)
class GlitchResponse:
    """Audio-triggered glitch intensity bump."""

    enabled: bool = True
    audio_boost: float = 0.5  # extra intensity when bass exceeds the trigger threshold


@dataclass(slots=True, frozen=True)
class CompositeResponse:
    """Final compositor audio->look response (bloom mix, chromatic, vignette, grade)."""

    enabled: bool = True
    bloom_intensity_base: float = 0.8
    bloom_intensity_bass: float = 1.0
    bloom_intensity_beat: float = 0.6
    chromatic_base: float = 0.25
    chromatic_bass: float = 0.2
    chromatic_beat: float = 0.15
    chromatic_max: float = 0.6
    vignette_base: float = 0.35
    vignette_treb: float = 0.12
    vignette_bass_neg: float = 0.05
    vignette_min: float = 0.25
    vignette_max: float = 0.55
    vignette_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    lift_base: float = 0.02
    lift_mid: float = 0.03
    gamma_base: float = 0.92
    gamma_bass: float = 0.04
    gamma_treb: float = 0.02
    gain_base: float = 0.95
    gain_beat: float = 0.1
    sat_base: float = 0.9
    sat_mid: float = 0.35
    sat_beat: float = 0.1
    sat_max: float = 1.3
    contrast_base: float = 1.1
    contrast_bass: float = 0.15
    contrast_beat: float = 0.1
    contrast_max: float = 1.4
    light_tint_strength: float = 0.35


@dataclass(slots=True, frozen=True)
class CompositorTuning:
    """Renderer-owned visual tuning for the generic composite pass.

    These values control how the renderer blends bloom octaves, the temporal
    accumulation against the reservoir, the volumetric light-ray scattering
    (ray count, ray decay, light position), the lens distortion strength, and
    the chromatic aberration offset.  They are *not* derived from any preset's
    authored equations - they are the renderer's own post-processing touches
    that were previously hard-coded inside the composite shader.

    Preset-authoritative visuals: keep ``enabled`` True (so the renderer applies
    its house style) but retune individual constants here, or set ``enabled``
    False to get a neutral composite that does not impose any renderer style.
    When ``EnhancementLayer.enabled`` is False the composite uniforms are filled
    with neutral values so no renderer-owned tuning is applied at all.
    """

    enabled: bool = True
    bloom_weight_small: float = 0.8
    bloom_weight_medium: float = 0.5
    bloom_weight_large: float = 0.25
    bloom_temporal_mix: float = 0.15
    ray_step_scale: float = 1.0 / 32.0
    ray_decay: float = 0.96
    ray_contribution: float = 0.4
    # Number of light-ray iterations.  Previously hard-coded to 32 in the
    # shader; now a data-driven uniform so the pass cost scales with the
    # configured value instead of always running a fixed 32-iteration loop.
    ray_count: int = 32
    light_pos_x: float = 0.5
    light_pos_y: float = 0.5
    #: Lens-distortion strength applied in the composite pass (barrel/pincushion
    #: UV distortion).  Now wired into the composite shader via u_lens_distortion.
    lens_distortion: float = 0.03
    chromatic_step: float = 0.02
    glow_scale: float = 0.005
    beat_glow_scale: float = 10.0
    vig_strength: float = 0.8
    #: Exponent for the vignette falloff curve (pow(dist, vignette_smooth)).
    #: 1.0 = linear, 2.0 = quadratic (smoother edges), higher = sharper fall-off.
    #: Replaces the previously hard-coded 2.0 so the vignette curve shape is
    #: data-driven and can be retuned or disabled without touching shader code.
    vignette_smooth: float = 2.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "ray_count",
            max(1, min(128, int(self.ray_count))),
        )


@dataclass(slots=True, frozen=True)
class EnhancementLayer:
    """Optional renderer-owned audio->visual response layer.

    Presets define visual *behaviour* (warp, zoom, waves, shapes, authored
    shaders); audio is delivered to them as preset variables (bass/mid/treb/
    beat).  This layer is the *optional* set of audio-reactive touches that the
    renderer used to bake in directly (brightness beating, feedback bass dip,
    bloom/blur boosts, chromatic aberration, ...).  It is pure data: a preset
    or user settings bundle can enable/disable or retune each response without
    touching renderer code.  Set ``enabled`` (or any sub-response) to ``False``
    to get pure preset-driven visuals.
    """

    enabled: bool = True
    brightness: BrightnessResponse = field(default_factory=BrightnessResponse)
    feedback: FeedbackResponse = field(default_factory=FeedbackResponse)
    bloom: BloomResponse = field(default_factory=BloomResponse)
    blur: BlurResponse = field(default_factory=BlurResponse)
    motion_blur: MotionBlurResponse = field(default_factory=MotionBlurResponse)
    lens: LensResponse = field(default_factory=LensResponse)
    kaleidoscope: KaleidoscopeResponse = field(default_factory=KaleidoscopeResponse)
    glitch: GlitchResponse = field(default_factory=GlitchResponse)
    composite: CompositeResponse = field(default_factory=CompositeResponse)
    tuning: CompositorTuning = field(default_factory=CompositorTuning)


class ResourceOwnership(enum.Enum):
    """Ownership of GL resources in the render graph.

    OWNED: The container allocated these GL objects and is responsible for
           deleting them on cleanup/retirement.
    BORROWED: The container merely references persistent/live GL objects
              (e.g. presented_src_fbo, same-size multipass FBOs reused across
              a rebuild). _cleanup_render_graph must never gl-delete borrowed
              objects.

    Bool compatibility: ``bool(ResourceOwnership.OWNED) == True``,
    ``bool(ResourceOwnership.BORROWED) == False``. Constructors accept either
    the enum or a bool for backward compatibility with existing tests.
    """

    OWNED = True
    BORROWED = False

    def __bool__(self) -> bool:
        return self.value


class TextureAnimationSpeedUnits(enum.Enum):
    """Explicit units for texture scroll / animation speeds.

    UNITS_PER_REFERENCE_FRAME: the legacy 60-FPS-normalized convention.
        A speed of ``0.02`` advances the texture by ``0.02`` per reference
        frame (1/60 s), with automatic scaling so the visual velocity stays
        constant across 30/60/120/144 Hz.
    UNITS_PER_SECOND: true physical velocity.  A speed of ``0.02`` advances
        the texture by ``0.02`` per *second* of wall-clock time.
    """

    UNITS_PER_REFERENCE_FRAME = "units_per_reference_frame"
    UNITS_PER_SECOND = "units_per_second"


@dataclass(slots=True, frozen=True)
class RenderTarget:
    """Immutable per-frame render destination.

    Bundles the FBO and its color-attachment texture so the render pipeline
    can return a single value instead of continuously updating separate
    ``target_fbo`` / ``target_tex`` / ``final_tex`` / ``current_source_tex`` /
    ``completed_fbo`` / ``completed_tex`` fields.
    """

    fbo: int = 0
    texture: int = 0
    width: int = 0
    height: int = 0

    def valid(self) -> bool:
        return self.fbo > 0 and self.texture > 0 and self.width > 0 and self.height > 0


@dataclass(slots=True, frozen=True)
class AudioUniformTuning:
    """Centralized tuning constants for audio-derived GPU uniforms."""

    beat_derived_scale: float = 0.5
    bpm_normalization: float = 200.0


class RendererMode(enum.Enum):
    """Explicit renderer operating mode.

    STRICT_MILKDROP: only MilkDrop 1.x authored behaviour; no renderer-owned
        post-processing, no renderer extensions.
    ENHANCED_MILKDROP: MilkDrop visuals plus the optional EnhancementLayer
        responses (bloom, blur, motion blur, lens distortion, etc.).
    EVOLUTIONARY: full renderer visual layer including auto-randomization
        and texture animation.
    """

    STRICT_MILKDROP = "strict_milkdrop"
    ENHANCED_MILKDROP = "enhanced_milkdrop"
    EVOLUTIONARY = "evolutionary"


@dataclass(slots=True, frozen=True)
class RendererConfig:
    preset_name: str = "default"
    auto_switch: bool = True
    auto_switch_interval_seconds: float = 30.0
    bloom_enabled: bool = True
    motion_blur_enabled: bool = False
    lens_distortion_enabled: bool = False
    kaleidoscope_enabled: bool = False
    glitch_enabled: bool = False
    texture_blend: float = 0.5
    texture_zoom: float = 1.0
    warp_amount: float = 0.0
    warp_type: int = 0
    feedback_amount: float = 0.95
    brightness: float = 0.8
    beat_flash: float = 0.0
    enhancement: EnhancementLayer = field(default_factory=EnhancementLayer)
    texture_enhancement_enabled: bool = False
    texture_animation_speed_units: TextureAnimationSpeedUnits = (
        TextureAnimationSpeedUnits.UNITS_PER_REFERENCE_FRAME
    )
    texture_bass_boost_multiplier: float = 2.0
    texture_offset_wrap_range: float = 10.0
    renderer_mode: RendererMode = RendererMode.STRICT_MILKDROP
    audio_tuning: AudioUniformTuning = field(default_factory=AudioUniformTuning)

    def __post_init__(self) -> None:
        if self.renderer_mode == RendererMode.STRICT_MILKDROP:
            object.__setattr__(self, "bloom_enabled", False)
            object.__setattr__(self, "motion_blur_enabled", False)
            object.__setattr__(self, "lens_distortion_enabled", False)
            object.__setattr__(self, "kaleidoscope_enabled", False)
            object.__setattr__(self, "glitch_enabled", False)
            object.__setattr__(self, "texture_enhancement_enabled", False)
            object.__setattr__(self, "enhancement", EnhancementLayer(enabled=False))
        # Clamp the auto-switch interval at the data-model boundary so a
        # misconfigured 0/negative/non-numeric value cannot enqueue a switch
        # on every housekeeping tick.  _check_auto_preset_switch also clamps
        # defensively for any value that bypasses this constructor.
        try:
            iv = float(self.auto_switch_interval_seconds)
        except (TypeError, ValueError):
            iv = 30.0
        if not math.isfinite(iv) or iv < AUTO_SWITCH_MIN_INTERVAL:
            iv = AUTO_SWITCH_MIN_INTERVAL
        object.__setattr__(self, "auto_switch_interval_seconds", iv)
        for _name in ("texture_blend", "brightness", "feedback_amount"):
            try:
                _v = float(getattr(self, _name))
            except (TypeError, ValueError):
                _v = 0.0
            if not math.isfinite(_v):
                _v = 0.0
            _v = max(0.0, min(1.0, _v))
            object.__setattr__(self, _name, _v)
        if not isinstance(self.texture_animation_speed_units, TextureAnimationSpeedUnits):
            object.__setattr__(
                self,
                "texture_animation_speed_units",
                TextureAnimationSpeedUnits.UNITS_PER_REFERENCE_FRAME,
            )


@dataclass(slots=True)
class RendererSettings:
    """Legacy wrapper around :class:`RendererConfig`.

    ``RendererSettings`` is currently a one-field passthrough: the only
    consumer in :mod:`linamp` previously read ``visualizer._settings.renderer.preset_name``
    and wrote a fresh ``RendererSettings`` back - but the MilkDrop renderer
    never reads ``_settings`` itself, so that assignment was dead code and
    has now been removed.

    ``RendererSettings`` is kept constructible (and is exercised by
    ``tests/test_compositor_tuning.py::test_renderer_settings_serializable``)
    for backward compatibility with any external consumer that may still
    import it.  Prefer instantiating :class:`RendererConfig` directly; new
    code should not depend on ``RendererSettings`` and it is a candidate
    for removal in a future cleanup pass.
    """

    renderer: RendererConfig = field(default_factory=RendererConfig)


# ---------------------------------------------------------------------------
# VM variable registry
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class VariableDef:
    name: str
    default: float = 0.0
    index: int = 0
    gpu_uniform: bool = False
    category: str = "general"
    min_val: float = -1e30
    max_val: float = 1e30

    def clamp(self, value: float) -> float:
        return max(self.min_val, min(self.max_val, value))


VARIABLES: dict[str, VariableDef] = {}
_VAR_INDEX_COUNTER = [0]


def _var(
    name: str,
    default: float = 0.0,
    gpu: bool = False,
    category: str = "general",
    min_val: float = -1e30,
    max_val: float = 1e30,
) -> VariableDef:
    idx = _VAR_INDEX_COUNTER[0]
    _VAR_INDEX_COUNTER[0] += 1
    v = VariableDef(
        name=name,
        default=default,
        index=idx,
        gpu_uniform=gpu,
        category=category,
        min_val=min_val,
        max_val=max_val,
    )
    VARIABLES[name] = v
    return v


VAR_REGISTRY: dict[str, int] = {}

_ANIMATION_VARIABLES = [
    ("zoom", 1.0),
    ("rot", 0.0),
    ("warp", 0.0),
    ("cx", 0.5),
    ("cy", 0.5),
    ("dx", 0.0),
    ("dy", 0.0),
    ("sx", 1.0),
    ("sy", 1.0),
    ("decay", 0.9),
    ("wave_r", 0.0),
    ("wave_g", 0.0),
    ("wave_b", 0.0),
    ("wave_a", 0.0),
    ("wave_x", 0.5),
    ("wave_y", 0.5),
]

for _n, _d in _ANIMATION_VARIABLES:
    _var(_n, _d, gpu=True, category="animation")

_audio_vars = [
    ("bass", 0.0),
    ("mid", 0.0),
    ("treb", 0.0),
    ("bass_att", 0.0),
    ("mid_att", 0.0),
    ("treb_att", 0.0),
    ("vol", 0.0),
    ("vol_att", 0.0),
    ("beat", 0.0),
    ("beat_flash", 0.0),
    ("rms", 0.0),
    ("peak", 0.0),
]
for _n, _d in _audio_vars:
    _var(_n, _d, gpu=True, category="audio")

_time_vars = [
    ("time", 0.0),
    ("frame", 0),
    ("fps", 60.0),
    ("dt", 0.0),
]
for _n, _d in _time_vars:
    _var(_n, _d, gpu=True, category="time")

_res_vars = [
    ("x", 0.0),
    ("y", 0.0),
    ("rad", 0.0),
    ("ang", 0.0),
    ("width", 1.0),
    ("height", 1.0),
    ("aspect", 1.0),
    ("aspectx", 1.0),
    ("aspecty", 1.0),
    ("w", 1.0),
    ("h", 1.0),
]
for _n, _d in _res_vars:
    _var(_n, _d, gpu=True, category="resolution")

_wave_vars = [
    ("ob_size", 0.0),
    ("ob_r", 0.0),
    ("ob_g", 0.0),
    ("ob_b", 0.0),
    ("ob_a", 0.0),
    ("ib_size", 0.0),
    ("ib_r", 0.0),
    ("ib_g", 0.0),
    ("ib_b", 0.0),
    ("ib_a", 0.0),
    # MilkDrop 1.x wave scaling parameters (used by wave rendering)
    ("fwavescale", 1.0),
    ("fwavesmoothing", 0.5),
    ("fwaveparam", 0.0),
]
for _n, _d in _wave_vars:
    _var(_n, _d, gpu=True, category="wave")

_other_vars = [
    ("thresh", 0.0),
    ("wave_mystery", 0.0),
    ("wave_mode", 0),
    ("wave_size", 1.0),
    ("warpAnimSpeed", 0.0),
    ("t", 0.0),
    ("pi", math.pi),
    ("e", math.e),
    ("phi", 1.618033988749895),
]
for _n, _d in _other_vars:
    _var(_n, _d, gpu=False, category="other")

echo_vars = [
    ("echo_zoom", 1.0),
    ("echo_alpha", 0.0),
    ("echo_orient", 0),
]
for _n, _d in echo_vars:
    _var(_n, _d, gpu=False, category="echo")

_t_bridge_vars = [
    ("t1", 0.0),
    ("t2", 0.0),
    ("t3", 0.0),
    ("t4", 0.0),
    ("t5", 0.0),
    ("t6", 0.0),
    ("t7", 0.0),
    ("t8", 0.0),
]
for _n, _d in _t_bridge_vars:
    _var(_n, _d, gpu=False, category="wave")

VAR_REGISTRY = {name: vdef.index for name, vdef in VARIABLES.items()}

Q_LOOKUP = {f"q{i + 1}": i for i in range(32)}
T_LOOKUP = {f"t{i + 1}": i for i in range(8)}


# ---------------------------------------------------------------------------
# Render-graph resource containers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TransitionState:
    """Single authoritative container for all transition-related state.

    Consolidating these nine independent attributes into one object prevents
    state desynchronization when a preset switch fails partway through.

    Snapshot contract
    -----------------
    ``_snapshot_transition_state`` captures every field on this object
    plus the capture-phase flags (``capture_pending`` /
    ``capture_pending_frame_id``) that live on :class:`TransitionResources`.
    The snapshot dict is the single source of truth for rollback: every
    key written during snapshotting has a matching restore in
    ``_restore_transition_state``.

    The *transition target* - i.e. the destination preset whose GL
    resources (textures, FBOs, shaders) are being prepared - is **not**
    part of the snapshot.  The target is recomputed from the renderer's
    pending-preset request (``_pending_preset_request``) on each frame
    during the capture phase.  If the capture fails and the state is
    rolled back, the target request is discarded by the caller before
    the next frame, so the snapshot remains self-contained and free of
    dangling references to partially-created GL resources.
    """

    generation: int = 0
    from_valid: bool = False
    to_valid: bool = False
    from_generation: int = -1
    to_generation: int = -1
    resource_generation: int = -1
    start_time: float = 0.0
    progress: float = 0.0
    active: bool = False
    effect: str = "crossfade"
    pixel_size: float = 1.0  # Pixel size for dissolve effect
    pixel_shape: int = 0      # 0=square, 1=circle, 2=triangle


@dataclass(slots=True)
class TransitionResources:
    """Authoritative container for all transition GL resources.

    Consolidating the six scattered texture/FBO attributes plus capture
    flags into one object eliminates the opportunity for desynchronization
    between ``transition_from_tex``, ``transition_to_tex``, ``transition_fbo``,
    ``capture_pending``, etc.

    The ``target_fbo`` / ``target_tex`` pair is a dedicated render target
    for the *new* preset during the RENDER_TARGET phase.  The old preset
    continues to render to the multipass FBOs (``from_tex``), while the
    new preset renders to ``target_fbo`` so the BLEND phase can sample
    both independently.
    """

    from_tex: int = 0
    to_tex: int = 0
    fbo_tex: int = 0
    fbo: int = 0
    from_fbo: int = 0
    to_fbo: int = 0
    # Dedicated render target for the new preset during RENDER_TARGET.
    target_fbo: int = 0
    target_tex: int = 0
    # Ping-pong buffer for the *previous* target frame so the target
    # preset's ``u_prev_frame`` samples its own evolving history instead
    # of always sampling a solid black texture.  This enables MilkDrop
    # feedback (``f`` variable, motion blur, trails) to build up during
    # the visible transition, not just during the initial capture.
    target_prev_fbo: int = 0
    target_prev_tex: int = 0
    target_prev_valid: bool = False
    # GL program ID of the *new* preset during RENDER_TARGET.
    target_program_id: int = 0
    # Python-side program object (PresetProgram) backing target_program_id.
    target_program: PresetProgram | None = None
    capture_pending: bool = False
    capture_pending_frame_id: int = -1
    width: int = 0
    height: int = 0
    # See MultipassResources.owned for the ownership contract.
    owned: ResourceOwnership = ResourceOwnership.OWNED



@dataclass(slots=True)
class MultipassResources:
    fbo: list[int] = field(default_factory=lambda: [0, 0])
    fbo_tex: list[int] = field(default_factory=lambda: [0, 0])
    write_fbo_idx: int = 0
    last_presented_fbo: int = 0
    width: int = 0
    height: int = 0
    has_valid_frame: bool = False
    owned: ResourceOwnership = ResourceOwnership.OWNED



@dataclass(slots=True)
class PresentedResources:
    tex: int = 0
    fbo: int = 0
    width: int = 0
    height: int = 0
    tex_valid: bool = False
    owned: ResourceOwnership = ResourceOwnership.OWNED



@dataclass(slots=True)
class BloomResources:
    tex: int = 0
    fbo: int = 0
    blur_tex: list[list[int]] = field(default_factory=list)
    blur_fbo: list[list[int]] = field(default_factory=list)
    owned: ResourceOwnership = ResourceOwnership.OWNED



@dataclass(slots=True)
class AdvancedResources:
    motion_blur_tex: int = 0
    motion_blur_fbo: int = 0
    lens_tex: int = 0
    lens_fbo: int = 0
    bloom_reservoir_tex: int = 0
    bloom_reservoir_fbo: int = 0
    kaleidoscope_tex: int = 0
    kaleidoscope_fbo: int = 0
    owned: ResourceOwnership = ResourceOwnership.OWNED



@dataclass(slots=True)
class RenderGraph:
    multipass: MultipassResources | None = None
    presented: PresentedResources | None = None
    transition: TransitionResources | None = None
    bloom: BloomResources | None = None
    advanced: AdvancedResources | None = None
    width: int = 0
    height: int = 0


@dataclass(slots=True)
class RenderPassContext:
    """Per-frame render state passed to each render-pass function.

    This replaces the sprawling local-variable block in ``_render_frame``
    with an explicit, typed container.  Each pass receives a context and
    returns either a new context (on success) or ``None`` (on abort).

    The context is intentionally NOT frozen: passes mutate it to thread
    state forward (e.g. ``current_source_tex`` advances after each effect
    pass).  Callers must treat the context as single-threaded per-frame
    state; no two passes share a context.
    """

    width: int = 0
    height: int = 0
    screen_fbo: int = 0
    frame_time: float = 0.0
    dt: float = 0.0
    curr_idx: int = 0
    prev_frame_tex: int = 0
    current_source_tex: int = 0
    final_tex: int = 0
    target_fbo: int = 0
    target_tex: int = 0
    composite_write_idx: int = 0
    audio_levels: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    preset_composite_ran: bool = False
    completed_target: RenderTarget | None = None


# ---------------------------------------------------------------------------
# Preset-load bookkeeping and frame/transition enums
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PendingPresetRequest:
    """Authoritative pending preset request with cancellation support.

    Only the newest request may reach ``_install_preset_immediately``.  Older
    requests are discarded at the atomic-commit point by comparing their
    ``request_id`` against the renderer's current ``_preset_request_id``.
    """

    request_id: int
    path: str
    smooth_transition: bool


@dataclass(slots=True)
class RetryState:
    """Bookkeeping for a deferred (GLib-idle) preset retry.

    Each retry is tied to a specific ``request_id`` so that a stale retry
    from a superseded request cannot be confused with a fresh load for the
    same preset path.
    """

    request_id: int
    milk_path: str
    smooth_transition: bool
    first_request_ms: int
    attempt: int = 0


class FrameResult(enum.Enum):
    """Render frame outcome.

    SUCCESS: the frame rendered and was presented.
    RECOVERED: the frame failed but recovery code painted the previous frame.
    FAILED: the frame could not be rendered at all; GTK may continue.
    """

    SUCCESS = "success"
    RECOVERED = "recovered"
    FAILED = "failed"


class TransitionPhase(enum.Enum):
    """States for the real preset transition state machine.

    IDLE: No transition active.
    CAPTURE_SOURCE: Capture the current frame as the source texture.
    PREPARE_TARGET: Parse and compile the new preset.
    RENDER_TARGET: Render the new preset to its own texture.
    BLEND: Blend between source and target textures.
    COMMIT: Finalize the transition and retire old resources.
    RETIRE_OLD: Clean up old preset resources.
    """

    IDLE = "idle"
    CAPTURE_SOURCE = "capture_source"
    PREPARE_TARGET = "prepare_target"
    RENDER_TARGET = "render_target"
    BLEND = "blend"
    COMMIT = "commit"
    RETIRE_OLD = "retire_old"


@dataclass(slots=True)
class FrameContext:
    """Frame-local values passed between render stages."""

    width: int
    height: int
    screen_fbo: int = 0
    dt: float = 0.0
    frame_result: FrameResult = FrameResult.SUCCESS
    frame_time: float = 0.0
    params: dict[str, Any] = field(default_factory=dict)
    curr_idx: int = 0
    prev_frame_tex: int = 0
    audio_levels: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    eh: EnhancementLayer | None = None
    current_time: float = 0.0
    current_source_tex: int = 0
    composite_write_idx: int = 0
    final_tex: int = 0
    bloom_w: int = 0
    bloom_h: int = 0
    target_fbo: int = 0
    target_tex: int = 0
    completed_target: RenderTarget | None = None


class RecoveryState(enum.Enum):
    """Current recovery *operation*.

    A single :class:`RecoveryState` value always describes the operation
    currently executing (or FAILED).  Pending work is tracked separately by
    :class:`RecoveryManager` via ``pending_gl_reinit`` /
    ``pending_shader_recovery`` so that a new request arriving during an
    existing retry is represented independently of the running operation.
    """

    IDLE = "idle"
    REINIT_RUNNING = "reinit_running"
    SHADER_RUNNING = "shader_running"
    FAILED = "failed"


class RecoveryOperation(enum.Enum):
    """Which recovery operation is currently running."""

    NONE = "none"
    GL_REINIT = "gl_reinit"
    SHADER_RECOVERY = "shader_recovery"


class PresetLoadResult(enum.Enum):
    """Result of loading a MilkDrop preset.

    INSTALLED: preset was successfully compiled and installed.
    PENDING:   preset compilation is still running; a retry is scheduled.
    FAILED:    preset failed to compile or load.
    STALE:     preset load was superseded by a newer request.
    """

    INSTALLED = "installed"
    PENDING = "pending"
    FAILED = "failed"
    STALE = "stale"

    def is_success(self) -> bool:
        """Return True for results that indicate the preset was accepted."""
        return self in (PresetLoadResult.INSTALLED, PresetLoadResult.PENDING)


class TextureLoadState(enum.Enum):
    """Explicit lifecycle state for an asynchronously-loaded texture.

    Replaces the previous implicit model where a texture's state was
    inferred from which data structure (``_texture_load_queue``,
    ``_texture_load_inflight``, ``_pending_texture_uploads``,
    ``loaded_textures``) currently held it.  An explicit state makes the
    lifecycle unambiguous and lets callers ask for the current state
    without inspecting four separate collections.

    States
    ------
    UNREQUESTED: path has never been enqueued.
    LOADING: disk decode is in progress on the worker thread.
    GPU_PENDING: decoded image data is queued for GL upload.
    UPLOADING: the render thread is currently performing the GL upload.
    READY: texture is fully uploaded and ``loaded_textures`` holds the ID.
    FAILED: the worker or upload path could not produce a texture.
    """

    UNREQUESTED = "unrequested"
    LOADING = "loading"
    GPU_PENDING = "gpu_pending"
    UPLOADING = "uploading"
    READY = "ready"
    FAILED = "failed"


@dataclass(slots=True)
class ProgramResources:
    """Bundle of GL resources produced by preset shader compilation.

    Returned by ``_compile_main_visual_shader`` (and the fake-compile helper
    used in tests).  Each render path stage that needs a preset program reads
    ``program_id`` (the GL program name), ``shader_source`` (the generated
    GLSL for diagnostics), ``sampler_bindings`` (logical-name -> uniform-name
    mapping), ``texsize_bindings`` (logical-name -> texsize-uniform mapping),
    ``texture_paths`` (logical-name -> file path for custom textures), and
    ``texsize_paths`` (logical-name -> texsize file path).
    """

    program_id: int = 0
    shader_source: str = ""
    sampler_bindings: dict[str, Any] = field(default_factory=dict)
    texsize_bindings: dict[str, Any] = field(default_factory=dict)
    texture_paths: dict[str, Any] = field(default_factory=dict)
    texsize_paths: dict[str, Any] = field(default_factory=dict)
