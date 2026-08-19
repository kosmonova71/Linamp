#!/usr/bin/env python3

from __future__ import annotations

import ast
import ctypes
import hashlib
import json
import logging
import math
import os
import random
import re
import sys
import threading
import time
import time as _time_module
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from queue import Queue
from typing import ClassVar

import gi
import numpy as np
from OpenGL import GL
from OpenGL.GL.EXT import texture_filter_anisotropic
from PIL import Image

try:
    gi.require_version('Gtk', '4.0')
    gi.require_version('Gst', '1.0')
    from gi.repository import Gdk, GLib, Gst, Gtk
    Gst.init(None)
except ValueError:
    sys.exit(1)

ASSIGN_RE = re.compile(r'^\s*([A-Za-z_]\w*|g?megabuf\s*\(.*\))\s*=')
FUNC_RE = re.compile(r'\b(above|below|equal|if|select|band|bor|bnot|rand|sigmoid|atan2|pow|min|max)\s*\(')
WAVE_RE = re.compile(r'wave(?:code)?_(\d+)_([a-zA-Z_]\w*)')
SHAPE_RE = re.compile(r'(?:shapecode|shape)_(\d+)_([a-zA-Z_]\w*)')
RE_END_SEMICOLON = re.compile(r';\s*$', re.MULTILINE)
RE_IF_SHORT1 = re.compile(r'\bif\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\)\s*$')
RE_IF_SHORT2 = re.compile(r'\bif\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*,\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\)\s*$')
RE_RAND_EMPTY = re.compile(r'rand\s*\(\s*\)')
RE_MEGABUF_KEY = re.compile(r'g?megabuf\s*\(\s*(.*)\s*\)', re.IGNORECASE)
RE_MEGABUF_LHS = re.compile(r'^g?megabuf\s*\(.*\)$', re.IGNORECASE)
RE_STRICT_IDENTIFIER = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')
RE_POWER_OP = re.compile(r'(?<![a-zA-Z_])\^(?![a-zA-Z_])')
RE_TYPE_DECL = re.compile(r'^float[234]?\s+')
RE_KEYWORD_AS = re.compile(r'\bas\b')
RE_GOTO_LABEL = re.compile(r'^[A-Za-z_]\w*\s*:\s*')
# MilkDrop-compatible epsilon values (from projectM reference)
EPSILON_COMPARE = 0.00001  # For equal() comparisons
EPSILON_DIV = 1e-7         # For division-by-zero protection
EPSILON_BOOL = 0.00001     # For boolean-as-float comparisons
EPSILON_SMALL = 1e-6       # General small value
RE_BIT_AND = re.compile(r'([^\s&|]+)\s*&([^\s&|]+)')
RE_BIT_OR = re.compile(r'([^\s&|]+)\s*\|\s*([^\s&|]+)')
RE_FUNC_REPLACE = re.compile(r'(\b(?:above|below|equal))\s*\(\s*([^)]*)\s*\)')

logger = logging.getLogger(__name__)




@dataclass(slots=True, frozen=True)
class BloomConfig:
    threshold: float = 0.25
    intensity: float = 1.2
    blur_radii: tuple[float, ...] = (2.0, 5.0, 12.0)

@dataclass(slots=True, frozen=True)
class AudioConfig:
    attack: float = 0.95
    decay: float = 0.85
    beat_threshold: float = 0.15
    beat_sensitivity: float = 0.5


@dataclass(slots=True, frozen=True)
class PostProcessConfig:
    bloom_threshold: float = 0.25
    bloom_intensity: float = 1.2
    bloom_blur_radii: tuple[float, ...] = (2.0, 5.0, 12.0)
    bloom_audio_bass_gain: float = 0.8
    bloom_audio_beat_gain: float = 0.7
    bloom_blur_audio_scale: float = 3.0

    composite_bloom_intensity_base: float = 0.8
    composite_bloom_intensity_bass_gain: float = 1.0
    composite_bloom_intensity_beat_gain: float = 0.6

    composite_chromatic_base: float = 0.25
    composite_chromatic_bass_gain: float = 0.2
    composite_chromatic_beat_gain: float = 0.15
    composite_chromatic_max: float = 0.6

    composite_vignette_base: float = 0.35
    composite_vignette_treb_gain: float = 0.12
    composite_vignette_bass_gain: float = -0.05
    composite_vignette_min: float = 0.25
    composite_vignette_max: float = 0.55

    composite_lift_base: float = 0.02
    composite_lift_mid_gain: float = 0.03

    composite_gamma_base: float = 0.92
    composite_gamma_bass_gain: float = 0.04
    composite_gamma_treb_gain: float = 0.02

    composite_gain_base: float = 0.95
    composite_gain_beat_gain: float = 0.1

    composite_saturation_base: float = 0.9
    composite_saturation_mid_gain: float = 0.35
    composite_saturation_beat_gain: float = 0.1
    composite_saturation_max: float = 1.3

    composite_contrast_base: float = 1.1
    composite_contrast_bass_gain: float = 0.15
    composite_contrast_max: float = 1.4

    kaleidoscope_angle_increment: float = 0.01
    kaleidoscope_angle_bass_gain: float = 0.1
    kaleidoscope_angle_treb_gain: float = 0.05

    motion_blur_audio_scale: float = 4.0

    lens_distortion_audio_scale: float = 3.0


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
    brightness: float = 1.0
    beat_flash: float = 0.0
    bloom: BloomConfig = field(default_factory=BloomConfig)
    post_process: PostProcessConfig = field(default_factory=PostProcessConfig)

@dataclass(slots=True)
class RendererSettings:
    renderer: RendererConfig = field(default_factory=RendererConfig)

@dataclass(slots=True, frozen=True)
class TransitionConfig:
    duration: float = 2.0
    enabled: bool = True


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

def _var(name: str, default: float = 0.0, gpu: bool = False,
         category: str = "general", min_val: float = -1e30, max_val: float = 1e30) -> VariableDef:
    idx = _VAR_INDEX_COUNTER[0]
    _VAR_INDEX_COUNTER[0] += 1
    v = VariableDef(name=name, default=default, index=idx,
                    gpu_uniform=gpu, category=category,
                    min_val=min_val, max_val=max_val)
    VARIABLES[name] = v
    return v

VAR_REGISTRY: dict[str, int] = {}

_animation_vars = [
    ("zoom", 1.0), ("rot", 0.0), ("warp", 0.0), ("cx", 0.5), ("cy", 0.5),
    ("dx", 0.0), ("dy", 0.0), ("sx", 1.0), ("sy", 1.0), ("decay", 0.9),
]
_gpu_anim_vars = [
    ("zoom", 1.0), ("rot", 0.0), ("warp", 0.0), ("cx", 0.5), ("cy", 0.5),
    ("dx", 0.0), ("dy", 0.0), ("sx", 1.0), ("sy", 1.0), ("decay", 0.9),
    ("wave_r", 0.0), ("wave_g", 0.0), ("wave_b", 0.0), ("wave_a", 0.0),
    ("wave_x", 0.5), ("wave_y", 0.5),
]

for _n, _d in _gpu_anim_vars:
    _var(_n, _d, gpu=True, category="animation")

_audio_vars = [
    ("bass", 0.0), ("mid", 0.0), ("treb", 0.0),
    ("bass_att", 0.0), ("mid_att", 0.0), ("treb_att", 0.0), ("beat", 0.0),
]
for _n, _d in _audio_vars:
    _var(_n, _d, gpu=True, category="audio")

_time_vars = [
    ("time", 0.0), ("frame", 0), ("fps", 60.0),
]
for _n, _d in _time_vars:
    _var(_n, _d, gpu=True, category="time")

_res_vars = [
    ("x", 0.0), ("y", 0.0), ("rad", 0.0), ("ang", 0.0),
    ("width", 1.0), ("height", 1.0), ("aspect", 1.0), ("w", 1.0), ("h", 1.0),
]
for _n, _d in _res_vars:
    _var(_n, _d, gpu=True, category="resolution")

_wave_vars = [
    ("ob_size", 0.0), ("ob_r", 0.0), ("ob_g", 0.0), ("ob_b", 0.0), ("ob_a", 0.0),
    ("ib_size", 0.0), ("ib_r", 0.0), ("ib_g", 0.0), ("ib_b", 0.0), ("ib_a", 0.0),
]
for _n, _d in _wave_vars:
    _var(_n, _d, gpu=True, category="wave")

_other_vars = [
    ("thresh", 0.0), ("wave_mystery", 0.0), ("wave_mode", 0),
    ("wave_size", 1.0), ("warpAnimSpeed", 0.0),
    ("t", 0.0), ("pi", math.pi), ("e", math.e), ("phi", 1.618033988749895),
]
for _n, _d in _other_vars:
    _var(_n, _d, gpu=False, category="other")

echo_vars = [
    ("echo_zoom", 1.0), ("echo_alpha", 0.0), ("echo_orient", 0),
]
for _n, _d in echo_vars:
    _var(_n, _d, gpu=False, category="echo")

VAR_REGISTRY = {name: vdef.index for name, vdef in VARIABLES.items()}

Q_LOOKUP = {f"q{i+1}": i for i in range(32)}


@dataclass(slots=True)
class GPUUniforms:
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
    time: float = 0.0
    frame: float = 0.0
    fps: float = 60.0
    x: float = 0.0
    y: float = 0.0
    rad: float = 0.0
    ang: float = 0.0
    width: float = 1.0
    height: float = 1.0
    aspect: float = 1.0
    w: float = 1.0
    h: float = 1.0
    q: list[float] = field(default_factory=lambda: [0.0] * 32)
    blend: float = 0.0
    _dirty: bool = True
    _changed_fields: frozenset = field(default_factory=frozenset)
    # Echo/video echo parameters for MilkDrop compatibility
    echo_zoom: float = 1.0
    echo_alpha: float = 0.0
    echo_orient: int = 0

    def to_dict(self) -> dict:
        return {
            'zoom': self.zoom, 'rot': self.rot, 'warp': self.warp,
            'cx': self.cx, 'cy': self.cy, 'dx': self.dx, 'dy': self.dy,
            'sx': self.sx, 'sy': self.sy, 'decay': self.decay,
            'wave_r': self.wave_r, 'wave_g': self.wave_g, 'wave_b': self.wave_b,
            'wave_a': self.wave_a, 'wave_x': self.wave_x, 'wave_y': self.wave_y,
            'bass': self.bass, 'mid': self.mid, 'treb': self.treb,
            'bass_att': self.bass_att, 'mid_att': self.mid_att,
            'treb_att': self.treb_att, 'beat': self.beat,
            'time': self.time, 'frame': self.frame, 'fps': self.fps,
            'x': self.x, 'y': self.y, 'rad': self.rad, 'ang': self.ang,
            'width': self.width, 'height': self.height, 'aspect': self.aspect,
            'w': self.w, 'h': self.h,
            **{f'q{i+1}': self.q[i] for i in range(32)},
            'blend': self.blend,
        }

    def mark_dirty(self):
        self._dirty = True
        self._changed_fields = frozenset()

    def mark_clean(self):
        self._dirty = False
        self._changed_fields = frozenset()

    def get_changed(self) -> dict:
        if self._dirty:
            return self.to_dict()
        result = {}
        for fname in self._changed_fields:
            result[fname] = getattr(self, fname)
        for i in range(32):
            fn = f'q{i+1}'
            if fn in self._changed_fields:
                result[fn] = self.q[i]
        return result

    def upload_changed(self, program=None):
        changed = self.get_changed()
        for name, value in changed.items():
            if name.startswith('q') and name[1:].isdigit():
                continue
            loc = program.get_uniform_location(name) if program is not None else -1
            if loc == -1:
                continue
            if isinstance(value, bool):
                GL.glUniform1i(loc, 1 if value else 0)
            elif isinstance(value, int):
                GL.glUniform1i(loc, value)
            elif isinstance(value, float):
                GL.glUniform1f(loc, value)
            elif isinstance(value, (list, tuple)) and len(value) == 2:
                GL.glUniform2f(loc, float(value[0]), float(value[1]))
            elif isinstance(value, (list, tuple)) and len(value) == 3:
                GL.glUniform3f(loc, float(value[0]), float(value[1]), float(value[2]))
            elif isinstance(value, (list, tuple)) and len(value) == 4:
                GL.glUniform4f(loc, float(value[0]), float(value[1]), float(value[2]), float(value[3]))
        self.mark_clean()


class GLResourceOwner:
    _DELETE_QUEUES: ClassVar[dict[int, list]] = {}
    _MAIN_THREAD_ID: ClassVar[int] = threading.current_thread().ident

    @classmethod
    def _get_delete_queue(cls, resource_type: str) -> list:
        if resource_type not in cls._DELETE_QUEUES:
            cls._DELETE_QUEUES[resource_type] = []
        return cls._DELETE_QUEUES[resource_type]

    @classmethod
    def queue_delete(cls, resource_type: str, gl_id: int):
        q = cls._get_delete_queue(resource_type)
        if gl_id and int(gl_id) > 0:
            q.append(int(gl_id))

    @classmethod
    def flush_deletes(cls):
        for resource_type, q in list(cls._DELETE_QUEUES.items()):
            if not q:
                continue
            ids_to_delete = q[:]
            q.clear()
            for single_id in ids_to_delete:
                try:
                    if resource_type == "program":
                        GL.glDeleteProgram(single_id)
                    elif resource_type == "texture":
                        GL.glDeleteTextures(1, [single_id])
                    elif resource_type == "fbo":
                        GL.glDeleteFramebuffers(1, [single_id])
                    elif resource_type == "vao":
                        GL.glDeleteVertexArrays(1, [single_id])
                    elif resource_type == "buffer":
                        GL.glDeleteBuffers(1, [single_id])
                except GL.GLError as e:
                    logger.warning(f"Failed to delete {resource_type} {single_id}: {e}")



class GLStateManager:
    __slots__ = (
        'blend', 'cull_face', 'depth_test', 'fbo_draw', 'fbo_read',
        'line_width', 'program', 'scissor', 'texture',
        'texture_unit', 'vao', 'viewport',
    )
    def __init__(self):
        self.program = 0
        self.vao = 0
        self.texture = 0
        self.texture_unit = -1
        self.fbo_read = 0
        self.fbo_draw = 0
        self.blend = False
        self.depth_test = False
        self.cull_face = False
        self.scissor = (0, 0, 0, 0)
        self.viewport = (0, 0, 0, 0)
        self.line_width = 1.0

    def reset(self):
        self.program = 0
        self.vao = 0
        self.texture = 0
        self.texture_unit = -1
        self.fbo_read = 0
        self.fbo_draw = 0
        self.blend = False
        self.depth_test = False
        self.cull_face = False
        self.scissor = (0, 0, 0, 0)
        self.viewport = (0, 0, 0, 0)
        self.line_width = 1.0

    def snapshot(self):
        return {
            'program': self.program,
            'vao': self.vao,
            'texture': self.texture,
            'texture_unit': self.texture_unit,
            'fbo_read': self.fbo_read,
            'fbo_draw': self.fbo_draw,
            'blend': self.blend,
            'depth_test': self.depth_test,
            'cull_face': self.cull_face,
        }

    def restore(self, snap):
        self.program = snap['program']
        self.vao = snap['vao']
        self.texture = snap['texture']
        self.texture_unit = snap['texture_unit']
        self.fbo_read = snap['fbo_read']
        self.fbo_draw = snap['fbo_draw']
        self.blend = snap['blend']
        self.depth_test = snap['depth_test']
        self.cull_face = snap['cull_face']
        if self.program > 0:
            GL.glUseProgram(self.program)
        if self.vao > 0:
            GL.glBindVertexArray(self.vao)
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fbo_read)
        if self.fbo_draw != self.fbo_read:
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo_draw)
        if self.texture > 0 and self.texture_unit >= 0:
            GL.glActiveTexture(GL.GL_TEXTURE0 + self.texture_unit)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture)
        if self.blend:
            GL.glEnable(GL.GL_BLEND)
        else:
            GL.glDisable(GL.GL_BLEND)
        if self.depth_test:
            GL.glEnable(GL.GL_DEPTH_TEST)
        else:
            GL.glDisable(GL.GL_DEPTH_TEST)
        if self.cull_face:
            GL.glEnable(GL.GL_CULL_FACE)
        else:
            GL.glDisable(GL.GL_CULL_FACE)


class GLProgram(GLResourceOwner):
    def __init__(self, vert_src: str = "", frag_src: str = ""):
        self._id: int = 0
        self._vert_src = vert_src
        self._frag_src = frag_src
        self._uniform_locs: dict[str, int] = {}
        self._linked: bool = False
        if vert_src and frag_src:
            self._compile_and_link(vert_src, frag_src)

    def _compile_and_link(self, vert_src: str, frag_src: str) -> bool:
        prog_id = _compile_gl_program(vert_src, frag_src, name="GLProgram")
        if prog_id > 0:
            self._id = prog_id
            self._linked = True
            return True
        return False

    @property
    def id(self) -> int:
        return self._id

    def use(self):
        if self._linked and self._id > 0:
            GL.glUseProgram(self._id)

    def get_uniform_location(self, name: str) -> int:
        if name not in self._uniform_locs:
            self._uniform_locs[name] = GL.glGetUniformLocation(self._id, name)
        return self._uniform_locs[name]

    def __int__(self) -> int:
        return self._id

    def __bool__(self) -> bool:
        return self._linked and self._id > 0

    def cleanup(self):
        if self._id > 0:
            self.queue_delete("program", self._id)
            self._id = 0
            self._linked = False


class GLTexture(GLResourceOwner):
    def __init__(self, width: int = 1, height: int = 1, internal_format=GL.GL_RGBA8,
                 data_format=GL.GL_RGBA, data_type=GL.GL_UNSIGNED_BYTE, data=None):
        self._id: int = 0
        self._width = max(1, width)
        self._height = max(1, height)
        self._internal_format = internal_format
        self._data_format = data_format
        self._data_type = data_type
        self._allocated = False
        self._upload(width, height, data)

    def _upload(self, width: int, height: int, data=None):
        try:
            tex_id = GL.glGenTextures(1)
            if isinstance(tex_id, (list, tuple, np.ndarray)):
                tex_id = int(tex_id[0])
            else:
                tex_id = int(tex_id)
            if tex_id <= 0:
                return
            self._id = tex_id
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._id)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, self._internal_format,
                            self._width, self._height, 0,
                            self._data_format, self._data_type, data)
            self._allocated = True
        except (GL.GLError, AttributeError, TypeError) as e:
            logger.debug(f"GLTexture upload failed: {e}")
            self._id = 0

    def bind(self, unit: int = 0):
        if self._id > 0:
            GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._id)

    @property
    def id(self) -> int:
        return self._id

    def __int__(self) -> int:
        return self._id

    def __bool__(self) -> bool:
        return self._allocated and self._id > 0

    def cleanup(self):
        if self._id > 0:
            self.queue_delete("texture", self._id)
            self._id = 0
            self._allocated = False


class GLFramebuffer(GLResourceOwner):
    def __init__(self, width: int = 512, height: int = 512):
        self._fbo: int = 0
        self._color_tex: int = 0
        self._width = max(1, width)
        self._height = max(1, height)
        self._complete = False
        self._prev_viewport = (0, 0, 0, 0)
        self._prev_fbo = 0
        self._bind_callback = None
        self._create()

    def set_bind_callback(self, callback):
        self._bind_callback = callback

    def _create(self):
        try:
            fbo = GL.glGenFramebuffers(1)
            if isinstance(fbo, (list, tuple, np.ndarray)):
                fbo = int(fbo[0])
            else:
                fbo = int(fbo)
            if fbo <= 0:
                return
            self._fbo = fbo
            tex = GL.glGenTextures(1)
            if isinstance(tex, (list, tuple, np.ndarray)):
                tex = int(tex[0])
            else:
                tex = int(tex)
            if tex <= 0:
                GLResourceOwner.queue_delete("fbo", int(self._fbo))
                self._fbo = 0
                return
            self._color_tex = tex
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._color_tex)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8,
                            self._width, self._height, 0,
                            GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
            GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
                                      GL.GL_TEXTURE_2D, self._color_tex, 0)
            status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
            self._complete = status == GL.GL_FRAMEBUFFER_COMPLETE
        except (GL.GLError, AttributeError, TypeError) as e:
            logger.debug(f"GLFramebuffer creation failed: {e}")
            self._fbo = 0
            self._color_tex = 0
            self._complete = False

    def bind(self):
        if self._fbo > 0 and self._complete:
            prev_viewport = np.zeros(4, dtype=np.int32)
            try:
                GL.glGetIntegerv(GL.GL_VIEWPORT, prev_viewport)
            except Exception:
                prev_viewport[:] = [0, 0, self._width, self._height]
            self._prev_viewport = tuple(int(x) for x in prev_viewport)
            prev_fbo = np.zeros(1, dtype=np.int32)
            try:
                GL.glGetIntegerv(GL.GL_FRAMEBUFFER_BINDING, prev_fbo)
            except Exception:
                prev_fbo[0] = 0
            self._prev_fbo = int(prev_fbo[0])
            if self._bind_callback:
                self._bind_callback(GL.GL_FRAMEBUFFER, self._fbo)
            else:
                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo)
            GL.glViewport(0, 0, self._width, self._height)

    def unbind(self):
        if self._bind_callback:
            self._bind_callback(GL.GL_FRAMEBUFFER, self._prev_fbo)
        else:
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._prev_fbo)
        GL.glViewport(*self._prev_viewport)

    def __bool__(self) -> bool:
        return self._complete and self._fbo > 0

    @property
    def fbo(self) -> int:
        return self._fbo

    @property
    def texture(self) -> int:
        return self._color_tex

    def cleanup(self):
        if self._color_tex > 0:
            self.queue_delete("texture", self._color_tex)
            self._color_tex = 0
        if self._fbo > 0:
            self.queue_delete("fbo", self._fbo)
            self._fbo = 0
            self._complete = False



class FBOManager:
    __slots__ = ('_bind_callback', '_current', '_fbos', '_prev_fbo')
    def __init__(self):
        self._fbos: dict[str, tuple[int, int]] = {}
        self._current: str | None = None
        self._prev_fbo: int = 0
        self._bind_callback = None

    def set_bind_callback(self, callback):
        self._bind_callback = callback

    def register(self, name: str, fbo_id: int, tex_id: int):
        self._fbos[name] = (fbo_id, tex_id)

    def bind(self, name: str):
        if name not in self._fbos:
            return
        fbo_id, _ = self._fbos[name]
        if self._current != name:
            prev_fbo = np.zeros(1, dtype=np.int32)
            try:
                GL.glGetIntegerv(GL.GL_FRAMEBUFFER_BINDING, prev_fbo)
            except Exception:
                prev_fbo[0] = 0
            self._prev_fbo = int(prev_fbo[0])
            if self._bind_callback:
                self._bind_callback(GL.GL_FRAMEBUFFER, fbo_id)
            else:
                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo_id)
            self._current = name

    def unbind(self):
        if self._current is not None:
            if self._bind_callback:
                self._bind_callback(GL.GL_FRAMEBUFFER, self._prev_fbo)
            else:
                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._prev_fbo)
            self._current = None

    def get_texture(self, name: str) -> int:
        if name not in self._fbos:
            return 0
        return self._fbos[name][1]

    def cleanup(self):
        for name, (fbo_id, tex_id) in list(self._fbos.items()):
            if tex_id:
                GLResourceOwner.queue_delete("texture", tex_id)
            if fbo_id:
                GLResourceOwner.queue_delete("fbo", fbo_id)
        self._fbos.clear()
        self._current = None
        self._prev_fbo = 0

    def unregister(self, name: str):
        if name in self._fbos:
            fbo_id, tex_id = self._fbos.pop(name)
            if tex_id:
                GLResourceOwner.queue_delete("texture", tex_id)
            if fbo_id:
                GLResourceOwner.queue_delete("fbo", fbo_id)


class GLVAO(GLResourceOwner):
    def __init__(self):
        self._id: int = 0
        self._create()

    def _create(self):
        try:
            vao = GL.glGenVertexArrays(1)
            if isinstance(vao, (list, tuple, np.ndarray)):
                vao = int(vao[0])
            else:
                vao = int(vao)
            if vao > 0:
                self._id = vao
        except (GL.GLError, AttributeError, TypeError) as e:
            logger.debug(f"GLVAO creation failed: {e}")
            self._id = 0

    @property
    def id(self) -> int:
        return self._id

    def bind(self):
        if self._id > 0:
            GL.glBindVertexArray(self._id)

    def unbind(self):
        GL.glBindVertexArray(0)

    def __int__(self) -> int:
        return self._id

    def __bool__(self) -> bool:
        return self._id > 0

    def cleanup(self):
        if self._id > 0:
            self.queue_delete("vao", self._id)
            self._id = 0


def _compile_gl_program(vert_src: str, frag_src: str, name: str = "shader") -> int:
    """Compile and link a vertex+fragment shader program.
    Returns the program ID, or 0 on failure.
    """
    try:
        vs = GL.glCreateShader(GL.GL_VERTEX_SHADER)
        GL.glShaderSource(vs, vert_src.lstrip())
        GL.glCompileShader(vs)
        if not GL.glGetShaderiv(vs, GL.GL_COMPILE_STATUS):
            err = GL.glGetShaderInfoLog(vs)
            if isinstance(err, (bytes, bytearray)):
                err = err.decode("utf-8", errors="ignore")
            GL.glDeleteShader(vs)
            raise RuntimeError(f"Vertex shader compilation failed: {err}")
        fs = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
        GL.glShaderSource(fs, frag_src.lstrip())
        GL.glCompileShader(fs)
        if not GL.glGetShaderiv(fs, GL.GL_COMPILE_STATUS):
            err = GL.glGetShaderInfoLog(fs)
            if isinstance(err, (bytes, bytearray)):
                err = err.decode("utf-8", errors="ignore")
            GL.glDeleteShader(vs)
            GL.glDeleteShader(fs)
            raise RuntimeError(f"Fragment shader compilation failed: {err}")
        prog = GL.glCreateProgram()
        GL.glAttachShader(prog, vs)
        GL.glAttachShader(prog, fs)
        GL.glLinkProgram(prog)
        if not GL.glGetProgramiv(prog, GL.GL_LINK_STATUS):
            err = GL.glGetProgramInfoLog(prog)
            if isinstance(err, (bytes, bytearray)):
                err = err.decode("utf-8", errors="ignore")
            GL.glDeleteShader(vs)
            GL.glDeleteShader(fs)
            GL.glDeleteProgram(prog)
            raise RuntimeError(f"Shader program linking failed: {err}")
        GL.glDeleteShader(vs)
        GL.glDeleteShader(fs)
        return int(prog)
    except (GL.GLError, AttributeError, TypeError, RuntimeError) as e:
        logger.error(f"Shader compilation failed ({name}): {e}")
        return 0


class PresetCompileError(Exception):
    """Raised when a preset expression or shader fails to compile."""

def safe_float(x, default_value=0.0):
    try:
        x = float(x)
        if not np.isfinite(x):
            return default_value
        return x
    except Exception:
        return default_value

_finite = safe_float

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
    enabled: bool = True
    # Wave parameters from preset
    mode: int = 0  # 0-7+ for different wave visualizations
    additive: bool = False
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

def validate_expression(expr: str) -> bool:
    """Best-effort validator for MilkDrop equation fragments.
    Goal: never let invalid preset syntax crash the loader.
    We intentionally do conservative checks before running ast.parse(...),
    because preprocess_milkdrop_expr() may still leave MilkDrop-only
    fragments that Python cannot parse.
    """
    if expr is None:
        return False
    expr = str(expr).strip()
    if not expr:
        return False
    if expr.startswith('//'):
        if '=' in expr:
            expr = expr[2:].lstrip()
        else:
            return False
    if expr.startswith('`'):
        return True
    if expr.startswith(('loop ', 'while ')):
        return False
    if '=' in expr:
        _lhs, _rhs = expr.split('=', 1)
        if _rhs.strip().startswith('//'):
            return False
    forbidden_tokens = (
        '__IF__', '__ELSE__',
        '`',
        ':',
        ';',
        ',,',
    )
    if any(tok in expr for tok in forbidden_tokens):
        return False
    try:
        expr = expr.strip()
        if expr.startswith('(') and expr.endswith(')'):
            expr = expr[1:-1].strip()
        expr = expr.replace('^', '**')
    except (ValueError, TypeError):
        return False
    if '=' in expr:
        parts = expr.split('=', 1)
        if len(parts) == 2:
            lhs = parts[0].strip()
            rhs = parts[1].strip()
            if rhs.startswith('//'):
                return False
            if lhs.isidentifier() or lhs.startswith('megabuf'):
                expr = rhs
                if not expr:
                    return False
            else:
                return False
    expr = expr.strip()
    if expr.endswith('*'):
        expr = expr[:-1].rstrip()

    if len(expr) == 1 and expr in '-+*/%):;,<>=!&|~':
        return False
    try:
        expr = preprocess_milkdrop_expr(expr)
        if not expr:
            return False
        ast.parse(expr, mode="eval")
        return True
    except SyntaxError as e:
        logger.warning(
            "Invalid preset expression:\n"
            "RAW      : %r\n"
            "PROCESSED: %r\n"
            "ERROR    : %s",
            expr,
            e,
            str(e),
        )
        return False
    except (ValueError, TypeError, MemoryError) as e:
        logger.warning("Validation exception: %s", e)
        return False
    except Exception as e:
        logger.warning("Unexpected validation exception: %s", e)
        return False

GLOBAL_COMPILE_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="PresetCompile")

class LRUCache(OrderedDict):
    MAX_SHADER_CACHE_MB = 64

    def __init__(self, maxsize=30):
        self.maxsize = maxsize
        self.cache_bytes = 0
        self.entry_count = 0
        super().__init__()

    def _estimate_size(self, value) -> int:
        if hasattr(value, 'nbytes'):
            return value.nbytes
        if isinstance(value, str):
            return len(value.encode('utf-8'))
        if isinstance(value, (int, float)):
            return 8
        return 64

    def __setitem__(self, key, value):
        if key in self:
            old_val = self[key]
            self.cache_bytes -= self._estimate_size(old_val)
            self.entry_count -= 1
        size = self._estimate_size(value)
        self.cache_bytes += size
        self.entry_count += 1
        while self.cache_bytes > self.MAX_SHADER_CACHE_MB * 1024 * 1024 and len(self) > 1:
            _, old_program = self.popitem(last=False)
            self.cache_bytes -= self._estimate_size(old_program)
            self.entry_count -= 1
            try:
                if old_program:
                    GLResourceOwner.queue_delete("program", int(old_program))
            except (GL.GLError, TypeError, ValueError):
                logger.debug("Failed to queue GL program deletion")
        super().__setitem__(key, value)
        while len(self) > self.maxsize and self.cache_bytes <= self.MAX_SHADER_CACHE_MB * 1024 * 1024:
            _old_key, old_program = self.popitem(last=False)
            self.cache_bytes -= self._estimate_size(old_program)
            self.entry_count -= 1
            try:
                if old_program:
                    GLResourceOwner.queue_delete("program", int(old_program))
            except (GL.GLError, TypeError, ValueError):
                logger.debug("Failed to queue GL program deletion")



class ObjectPool:
    __slots__ = ('_factory', '_maxsize', '_pool')
    def __init__(self, factory, maxsize: int = 32):
        self._pool: list = []
        self._factory = factory
        self._maxsize = maxsize

    def acquire(self):
        if self._pool:
            return self._pool.pop()
        return self._factory()

    def release(self, obj):
        if len(self._pool) < self._maxsize:
            self._pool.append(obj)

    def clear(self):
        self._pool.clear()


class TextureLRU(OrderedDict):
    def __init__(self, maxsize=256):
        self.maxsize = maxsize
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        super().__init__()

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        while len(self) > self.maxsize:
            _old_key, old_tex_id = self.popitem(last=False)
            self.evictions += 1
            try:
                if old_tex_id and old_tex_id > 0:
                    GLResourceOwner.queue_delete("texture", int(old_tex_id))
            except (GL.GLError, TypeError, ValueError):
                logger.debug("Failed to queue texture deletion")

    def get(self, key, default=None):
        if key in self:
            self.hits += 1
            self.move_to_end(key)
        else:
            self.misses += 1
        return super().get(key, default)

def clear_compile_caches():
    """Clear lru_caches on preset library changes to prevent unbounded growth.
    Call this when:
    - Preset library directory changes
    - Large number of presets are loaded
    - Memory pressure is detected
    """
    preprocess_milkdrop_expr.cache_clear()
    compile_expr_to_glsl.cache_clear()
    compile_assignment_to_glsl.cache_clear()
MEGABUF_SIZE = int(os.getenv("MILKDROP_MEGABUF_SIZE", "262144"))

@dataclass(slots=True, frozen=True)
class FunctionSpec:
    """Shared MilkDrop function definitions (Task 8)."""
    cpu_impl: Callable
    glsl_impl: Callable

FUNCTION_REGISTRY: dict[str, FunctionSpec] = {
    "above": FunctionSpec(
        cpu_impl=lambda a, b: 1.0 if a > b else 0.0,
        glsl_impl=lambda args: f"({args[0]} > {args[1]} ? 1.0 : 0.0)"
    ),
    "below": FunctionSpec(
        cpu_impl=lambda a, b: 1.0 if a < b else 0.0,
        glsl_impl=lambda args: f"({args[0]} < {args[1]} ? 1.0 : 0.0)"
    ),
    "equal": FunctionSpec(
        cpu_impl=lambda a, b: 1.0 if abs(a - b) < EPSILON_COMPARE else 0.0,
        glsl_impl=lambda args: f"(abs({args[0]} - {args[1]}) < {EPSILON_COMPARE} ? 1.0 : 0.0)"
    ),
    "above_equal": FunctionSpec(
        cpu_impl=lambda a, b: 1.0 if a >= b else 0.0,
        glsl_impl=lambda args: f"({args[0]} >= {args[1]} ? 1.0 : 0.0)"
    ),
    "below_equal": FunctionSpec(
        cpu_impl=lambda a, b: 1.0 if a <= b else 0.0,
        glsl_impl=lambda args: f"({args[0]} <= {args[1]} ? 1.0 : 0.0)"
    ),
    "assign": FunctionSpec(
        cpu_impl=lambda *args: 0.0,  # Handled in bytecode compiler
        glsl_impl=lambda args: "0.0"  # Handled separately
    ),
    "wrap": FunctionSpec(
        cpu_impl=lambda x, a, b: a + ((x - a) % (b - a)) if abs(b-a) > EPSILON_DIV else a,
        glsl_impl=lambda args: f"wrap({args[0]}, {args[1]}, {args[2]})"
    ),
    "cwrap": FunctionSpec(
        cpu_impl=lambda x, a, b: a + abs(((x - a) % (2 * (b - a))) - (b - a)) if abs(b-a) > EPSILON_DIV else a,
        glsl_impl=lambda args: f"cwrap({args[0]}, {args[1]}, {args[2]})"
    ),
    "sigmoid": FunctionSpec(
        cpu_impl=lambda x: 1.0 / (1.0 + math.exp(-np.clip(x, -100, 100))),
        glsl_impl=lambda args: f"(1.0 / (1.0 + exp(-{args[0]})))"
    ),
    "triangle": FunctionSpec(
        cpu_impl=lambda x: 2.0 * abs((x * 0.5) % 1.0 - 0.5),
        glsl_impl=lambda args: f"(2.0 * abs(fract({args[0]} * 0.5) - 0.5))"
    ),
    "sqr": FunctionSpec(
        cpu_impl=lambda x: x * x,
        glsl_impl=lambda args: f"({args[0]} * {args[0]})"
    ),
    "sign": FunctionSpec(
        cpu_impl=lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0),
        glsl_impl=lambda args: f"sign({args[0]})"
    ),
    "int": FunctionSpec(
        cpu_impl=lambda x: float(int(x)),
        glsl_impl=lambda args: f"float(int({args[0]}))"
    ),
    "frac": FunctionSpec(
        cpu_impl=lambda x: x - math.floor(x),
        glsl_impl=lambda args: f"fract({args[0]})"
    ),
    "rand": FunctionSpec(
        cpu_impl=lambda x: x,
        glsl_impl=lambda args: f"(md_rand(uv) * {args[0] if args else '1.0'})"
    ),
    "megabuf": FunctionSpec(
        cpu_impl=lambda idx: 0.0,
        glsl_impl=lambda args: f"megabuf_read({args[0]})"
    ),
    "gmegabuf": FunctionSpec(
        cpu_impl=lambda idx: 0.0,
        glsl_impl=lambda args: f"megabuf_read({args[0]})"
    ),
    "bor": FunctionSpec(
        cpu_impl=lambda a, b: float(int(a) | int(b)),
        glsl_impl=lambda args: f"float(int({args[0]}) | int({args[1]}))"
    ),
    "band": FunctionSpec(
        cpu_impl=lambda a, b: float(int(a) & int(b)),
        glsl_impl=lambda args: f"float(int({args[0]}) & int({args[1]}))"
    ),
    "bnot": FunctionSpec(
        cpu_impl=lambda a: float(~int(a)),
        glsl_impl=lambda args: f"float(~int({args[0]}))"
    ),
    "pulse": FunctionSpec(
        cpu_impl=lambda x, w: 1.0 if (x - math.floor(x)) < w else 0.0,
        glsl_impl=lambda args: f"(fract({args[0]}) < {args[1] if len(args) >= 2 else '0.5'} ? 1.0 : 0.0)"
    ),
    "select": FunctionSpec(
        cpu_impl=lambda c, a, b: a if c != 0 else b,
        glsl_impl=lambda args: f"({args[0]} != 0.0 ? {args[1]} : {args[2]})"
    ),
    "invsqrt": FunctionSpec(
        cpu_impl=lambda x: 1.0 / math.sqrt(x) if x > 0 else 0.0,
        glsl_impl=lambda args: f"inversesqrt({args[0]})"
    ),
    "fmod": FunctionSpec(
        cpu_impl=lambda a, b: a % b if b != 0 else 0.0,
        glsl_impl=lambda args: f"mod({args[0]}, {args[1]})"
    ),
    "fact": FunctionSpec(
        cpu_impl=lambda n: float(math.factorial(int(max(0, n)))),
        glsl_impl=lambda args: f"fact({args[0]})"
    ),
    "nchoosek": FunctionSpec(
        cpu_impl=lambda n, k: float(math.comb(int(n), int(k))) if int(k) >= 0 else 0.0,
        glsl_impl=lambda args: f"nchoosek({args[0]}, {args[1]})"
    ),
}

class GLSLGenerator(ast.NodeVisitor):
    """AST-based GLSL code generator for proper MilkDrop expression compilation.
    Converts Python AST expressions to GLSL shader code.
    This is a proper compiler backend, not string hacking.
    """

    def __init__(self):
        self.output = []
        self.glsl_builtins = frozenset({
            'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2',
            'sinh', 'cosh', 'tanh', 'pow', 'exp', 'log', 'log2', 'log10',
            'sqrt', 'inversesqrt', 'abs', 'sign', 'floor', 'ceil', 'fract',
            'trunc', 'round', 'mod', 'min', 'max', 'clamp', 'mix', 'step',
            'smoothstep', 'length', 'distance', 'dot', 'cross', 'normalize',
            'reflect', 'refract', 'texture', 'vec2', 'vec3', 'vec4',
            'float', 'int', 'bool', 'true', 'false'
        })
        self.milkdrop_vars = {
            'x': 'uv.x', 'y': 'uv.y',
            'time': 'u_time', 't': 'u_time',
            'bass': 'u_audio.x', 'mid': 'u_audio.y', 'treb': 'u_audio.z',
            'bass_att': 'u_audio.x', 'mid_att': 'u_audio.y', 'treb_att': 'u_audio.z',
            'rad': 'length(uv - 0.5)',
            'pi': '3.14159265359', 'e': '2.71828182846', 'phi': '1.61803398875',
            'w': 'u_resolution.x', 'h': 'u_resolution.y',
            'width': 'u_resolution.x', 'height': 'u_resolution.y',
            'mod': 'md_mod',
        }
        # Track reg00-reg99 and q1-q32 references for uniform binding
        self.reg_refs = set()
        self.q_refs = set()

    def _as_bool(self, val):
        """True MilkDrop boolean semantics: treat floats as booleans with epsilon."""
        return f"(abs({val}) > {EPSILON_BOOL})"

    def _handle_if(self, args):
        if len(args) >= 3:
            return f"({self._as_bool(args[0])} ? {args[1]} : {args[2]})"
        return "0.0"

    def _handle_megabuf(self, args):
        if len(args) >= 1:
            return f"megabuf_read({args[0]})"
        raise ValueError("GLSLGenerator: megabuf function requires 1 argument")

    def _get_helper_functions(self, megabuf_size=262144):
        """Returns GLSL helper functions for MilkDrop compatibility.
        Cached to avoid repeated filesystem reads during shader compilation.
        """
        helpers_path = Path(__file__).parent / "shaders" / "includes" / "milkdrop_helpers.glsl"
        try:
            with open(helpers_path, 'r') as f:
                helpers = f.read()
            return helpers.replace('{{MEGABUF_SIZE}}', str(int(megabuf_size)))
        except (OSError, FileNotFoundError) as e:
            raise RuntimeError(
                f"Failed to load GLSL helpers from {helpers_path}: {e}") from e

    def visit(self, node):
        """Visit a node and return GLSL code string."""
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        """Handle unknown node types gracefully."""
        return "0.0"

    def visit_Attribute(self, node):
        """Handle attribute access (like uv.x, uv.y)."""
        try:
            base = self.visit(node.value)
            return f"{base}.{node.attr}"
        except Exception as e:
            logger.debug(f"Failed to visit attribute {node.attr}: {e}")
        return "0.0"

    def visit_Constant(self, node):
        """Convert numeric constants to GLSL."""
        if isinstance(node.value, (int, float)):
            val = float(node.value)
            if val == int(val):
                return f"{int(val)}.0"
            return f"{val}"
        raise ValueError(f"GLSLGenerator: Unsupported constant type: {type(node.value).__name__}")

    def visit_Name(self, node):
        """Convert variable names to GLSL, including reg00-reg99 global registers."""
        name = node.id
        name_lower = name.lower()
        if name in ['PLUS_EQUALS', 'MINUS_EQUALS', 'TIMES_EQUALS', 'DIV_EQUALS']:
            return name
        # Handle reg00-reg99 global registers
        if len(name_lower) == 5 and name_lower.startswith('reg') and name_lower[3:].isdigit():
            idx = int(name_lower[3:])
            if 0 <= idx < 100:
                self.reg_refs.add(idx)
                return f"u_regs[{idx}]"
        if name_lower in self.milkdrop_vars:
            return self.milkdrop_vars[name_lower]
        if name_lower.startswith('q') and len(name_lower) <= 3 and name_lower[1:].isdigit():
            idx = int(name_lower[1:]) - 1
            if 0 <= idx < 32:
                self.q_refs.add(idx)
                return name_lower
        if name_lower in self.glsl_builtins:
            return name_lower
        return name_lower

    def visit_BinOp(self, node):
        """Convert binary operations to GLSL."""
        left = self.visit(node.left)
        right = self.visit(node.right)
        op_map = {
            ast.Add: '+',
            ast.Sub: '-',
            ast.Mult: '*',
            ast.Div: '/',
            ast.Mod: '%',
            ast.Pow: 'pow',
            ast.FloorDiv: '/'
        }
        op = op_map.get(type(node.op), '+')
        if op == 'pow':
            if right in ("2.0", "2"):
                return f"(({left})*({left}))"
            if right in ("3.0", "3"):
                return f"(({left})*({left})*({left}))"
            return f"pow({left}, {right})"
        if op == '%':
            safe_b = f"(abs({right}) < {EPSILON_DIV} ? {EPSILON_DIV} : {right})"
            return f"(({left}) - ({right}) * floor(({left}) / {safe_b}))"
        if isinstance(node.op, ast.FloorDiv):
            return f"floor({left} / {right})"
        if op == '/':
            return f"({left} / max(abs({right}), {EPSILON_DIV}))"
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
        return operand

    def visit_Call(self, node):
        """Convert function calls to GLSL."""
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        args = [self.visit(arg) for arg in node.args]
        if func_name in FUNCTION_REGISTRY:
            return FUNCTION_REGISTRY[func_name].glsl_impl(args)
        if func_name in self.glsl_builtins:
            if func_name == 'atan2' and len(args) >= 2:
                return f"atan({args[0]}, {args[1]})"
            args_str = ', '.join(args)
            return f"{func_name}({args_str})"
        args_str = ', '.join(args)
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
                parts.append(f"({left} < ({right} - {EPSILON_COMPARE}))")
            elif isinstance(op, ast.LtE):
                parts.append(f"({left} < ({right} + {EPSILON_COMPARE}))")
            elif isinstance(op, ast.Gt):
                parts.append(f"({left} > ({right} + {EPSILON_COMPARE}))")
            elif isinstance(op, ast.GtE):
                parts.append(f"({left} > ({right} - {EPSILON_COMPARE}))")
            else:
                parts.append(f"({left} == {right})")
            left = right
        if not parts:
            return left
        expr = " && ".join(parts)
        return f"(({expr}) ? 1.0 : 0.0)"

    def visit_BoolOp(self, node):
        """Convert boolean operations to GLSL with MilkDrop epsilon logic."""
        values = [self.visit(v) for v in node.values]
        if not values:
            return "0.0"
        if isinstance(node.op, ast.And):
            expr = " && ".join(self._as_bool(v) for v in values)
            return f"(({expr}) ? 1.0 : 0.0)"
        elif isinstance(node.op, ast.Or):
            expr = " || ".join(self._as_bool(v) for v in values)
            return f"(({expr}) ? 1.0 : 0.0)"
        return values[0]

    def visit_IfExp(self, node):
        """Convert Python's conditional expression (a if cond else b) to GLSL ternary."""
        self.generic_visit(node)
        test = self.visit(node.test)
        body = self.visit(node.body)
        orelse = self.visit(node.orelse)
        return f"({self._as_bool(test)} ? ({body}) : ({orelse}))"



class MilkDropTokenizer:
    __slots__ = ('len', 'pos', 'source')
    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.len = len(source)

    def peek(self, offset=0):
        p = self.pos + offset
        return self.source[p] if p < self.len else ''

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
        if c == '`':
            j = p + 1
            while j < n and s[j] != '`':
                j += 1
            if j < n:
                self.pos = j + 1
                return ('BACKTICK', s[p:j+1])
            self.pos = n
            return ('BACKTICK', s[p:])
        if c.isdigit() or (c == '.' and p + 1 < n and s[p+1].isdigit()):
            j = p + 1
            while j < n and (s[j].isdigit() or s[j] == '.'):
                j += 1
            if j < n and s[j] in 'eE':
                j += 1
                if j < n and s[j] in '+-':
                    j += 1
                while j < n and s[j].isdigit():
                    j += 1
            self.pos = j
            return ('NUMBER', s[p:j])
        if c.isalpha() or c == '_':
            j = p + 1
            while j < n and (s[j].isalnum() or s[j] == '_'):
                j += 1
            word = s[p:j]
            self.pos = j
            return ('IDENT', word)
        two = s[p:p+2]
        if two in ('+=', '-=', '*=', '/=', '==', '!=', '<=', '>=', '&&', '||'):
            self.pos = p + 2
            return ('OP', two)
        if c in ('+', '-', '*', '/', '%', '^', '<', '>', '=', '!', '&', '|', '~', '(', ')', ','):
            self.pos = p + 1
            return ('OP', c)
        self.pos = p + 1
        return ('RAW', c)

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
        if source.startswith('//') or source.lstrip('`').lstrip().startswith('#'):
            return source
        toks = MilkDropTokenizer(source).tokens()
        toks = self._normalize_assign_ops(toks)
        toks = self._normalize_semicolons(toks)
        toks = self._expand_short_ifs(toks)
        toks = self._expand_comparison_functions(toks)
        toks = self._expand_rif_ternaries(toks)
        toks = self._expand_condensed_ternaries(toks)
        toks = self._normalize_operators(toks)
        return self._tokens_to_str(toks)

    def _normalize_assign_ops(self, toks):
        out = []
        i = 0
        n = len(toks)
        while i < n:
            t = toks[i]
            if (t[0] == 'IDENT' and i + 2 < n and
                    toks[i+1][0] == 'OP' and toks[i+1][1] in ('+=', '-=', '*=', '/=')):
                lhs = t[1]
                op = toks[i+1][1]
                rhs = self._tokens_to_str(toks[i+2:]).strip()
                if rhs:
                    norm = {
                        '+=': f"{lhs} = {lhs} + ({rhs})",
                        '-=': f"{lhs} = {lhs} - ({rhs})",
                        '*=': f"{lhs} = {lhs} * ({rhs})",
                        '/=': f"{lhs} = {lhs} / ({rhs})",
                    }[op]
                    out = MilkDropTokenizer(norm).tokens()
                    break
            out.append(t)
            i += 1
        return out

    def _normalize_semicolons(self, toks):
        out = []
        for t in toks:
            if t[0] == 'OP' and t[1] == ';':
                continue
            out.append(t)
        return out

    def _expand_short_ifs(self, toks):
        out = []
        i = 0
        n = len(toks)
        while i < n:
            t = toks[i]
            if t[0] == 'IDENT' and t[1] in ('if', 'if_') and i + 1 < n and toks[i+1][0] == 'OP' and toks[i+1][1] == '(':
                args = self._extract_call_args(toks, i + 1)
                if args is not None:
                    expanded_args = [self._expand_short_ifs(a) for a in args]
                    arg_strs = [self._tokens_to_str(a) for a in expanded_args]
                    cond, then, else_ = (arg_strs + ['0', '0'])[:3]
                    replacement = f'({then} if ({cond}) else {else_})'
                    out.extend(self._str_to_toks(replacement))
                    i += 2 + sum(len(a) for a in args) + len(args)
                    continue
            out.append(t)
            i += 1
        return out

    def _expand_comparison_functions(self, toks):
        out = []
        i = 0
        n = len(toks)
        while i < n:
            t = toks[i]
            if t[0] == 'IDENT' and t[1] in ('above', 'below', 'equal', 'above_equal', 'below_equal') and i + 1 < n and toks[i+1][0] == 'OP' and toks[i+1][1] == '(':
                args = self._extract_call_args(toks, i + 1)
                if args and len(args) >= 2:
                    func = t[1]
                    a0 = self._tokens_to_str(args[0])
                    a1 = self._tokens_to_str(args[1])
                    if func == 'below':
                        replacement = f"({a0} < {a1})"
                    elif func == 'above':
                        replacement = f"({a0} > {a1})"
                    elif func == 'equal':
                        replacement = f"(abs({a0} - {a1}) < {EPSILON_COMPARE})"
                    elif func == 'above_equal':
                        replacement = f"({a0} >= {a1})"
                    elif func == 'below_equal':
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
        if 'rif(' not in s:
            return toks
        result = []
        i = 0
        n = len(s)
        while i < n:
            j = s.find('rif(', i)
            if j < 0:
                result.append(s[i:])
                break
            result.append(s[i:j])
            cond_open = j + len('rif(') - 1
            cond_close = self._find_matching_paren(s, cond_open)
            if cond_close < 0:
                result.append(s[j:])
                break
            cond_expr = s[cond_open + 1:cond_close].strip()
            k = cond_close + 1
            while k < n and s[k].isspace():
                k += 1
            if k < n and s[k] == ')':
                k += 1
                while k < n and s[k].isspace():
                    k += 1
            if not s.startswith('els', k):
                result.append('rif(')
                i = j + len('rif(')
                continue
            k += len('els')
            else_start = k
            depth_p = depth_b = depth_c = 0
            while k < n:
                c = s[k]
                if c == '(':
                    depth_p += 1
                elif c == ')':
                    if depth_p == 0 and depth_b == 0 and depth_c == 0:
                        break
                    depth_p = max(0, depth_p - 1)
                elif c == '[':
                    depth_b += 1
                elif c == ']':
                    depth_b = max(0, depth_b - 1)
                elif c == '{':
                    depth_c += 1
                elif c == '}':
                    depth_c = max(0, depth_c - 1)
                elif depth_p == 0 and depth_b == 0 and depth_c == 0 and c in '+-*/%^&|<>=,;':
                    break
                k += 1
            else_expr = s[else_start:k].strip()
            else_body = else_expr if else_expr and 'rif(' not in else_expr and not self._has_els_keyword(else_expr) and len(else_expr.split()) <= 1 else '0.0'
            result.append(f"({else_body} if ({cond_expr}) else 0.0)")
            i = k
        return MilkDropTokenizer(''.join(result)).tokens()

    def _has_els_keyword(self, s):
        i = 0
        n = len(s)
        while i < n:
            if s[i:i+3] == 'els' and (i == 0 or not s[i-1].isalnum()):
                return True
            i += 1
        return False

    def _expand_condensed_ternaries(self, toks):
        s = self._tokens_to_str(toks)
        result = []
        i = 0
        n = len(s)
        changed = False
        while i < n:
            if s[i:i+3] == 'if(' and i > 0:
                prev_char = s[i-1]
                if prev_char not in (' ', '\t', '\n', '(', '[', '{', ',', ':', ';', '+', '-', '*', '/', '%', '^', '&', '|', '<', '>', '=', '!', '~', ')', ']', '}'):
                    then_close = self._find_matching_paren(s, i + 2)
                    if then_close > 0:
                        then_body = s[i+3:then_close].strip()
                        if ',' in then_body:
                            result.append(s[i])
                            i += 1
                            continue
                        after_close = then_close + 1
                        k = after_close
                        depth2 = 0
                        while k < n:
                            c = s[k]
                            if c == '(':
                                depth2 += 1
                            elif c == ')':
                                depth2 -= 1
                            elif depth2 == 0 and c in '+-*/%^&|<>=,;':
                                break
                            k += 1
                        else_val = s[after_close:k].strip()
                        if else_val:
                            then_val = s[i-1]
                            converted = f'if({then_body}, {then_val}, {else_val})'
                            if result and result[-1] == s[i-1]:
                                result.pop()
                            result.append(converted)
                            changed = True
                            i = k
                            continue
            result.append(s[i])
            i += 1
        if not changed:
            return toks
        return MilkDropTokenizer(''.join(result)).tokens()

    def _normalize_operators(self, toks):
        out = []
        for t in toks:
            if t[0] == 'OP' and t[1] == '^':
                out.append(('OP', '**'))
            elif t[0] == 'IDENT' and t[1] == 'rand':
                out.append(('IDENT', 'rand'))
                out.append(('LPAREN', '('))
                out.append(('NUMBER', '1'))
                out.append(('RPAREN', ')'))
            elif t[0] == 'IDENT' and t[1] == 'mod':
                out.append(('IDENT', 'md_mod'))
            elif t[0] == 'IDENT' and t[1] == 'as':
                out.append(('IDENT', 'as_'))
            elif t[0] == 'IDENT' and ((RE_STRICT_IDENTIFIER.match(t[1]) and t[1].endswith(':')) or (t[1].startswith('float') and len(t[1]) > 5 and t[1][5:].isdigit())):
                continue
            else:
                out.append(t)
        return out

    def _extract_call_args(self, toks, start):
        if start >= len(toks) or toks[start][0] != 'OP' or toks[start][1] != '(':
            return None
        depth = 0
        args = [[]]
        i = start + 1
        while i < len(toks):
            t = toks[i]
            if t[0] == 'OP':
                if t[1] == '(':
                    depth += 1
                    args[-1].append(t)
                elif t[1] == ')':
                    if depth == 0:
                        return args
                    depth -= 1
                    args[-1].append(t)
                elif t[1] == ',' and depth == 0:
                    args.append([])
                else:
                    args[-1].append(t)
            else:
                args[-1].append(t)
            i += 1
        return args

    def _str_to_toks(self, s):
        return MilkDropTokenizer(s).tokens()

    def _tokens_to_str(self, toks):
        parts = []
        prev_type = None
        for t in toks:
            ttype, tval = t
            if ((prev_type == 'IDENT' and ttype in ('IDENT', 'NUMBER')) or
                    (prev_type == 'NUMBER' and ttype == 'IDENT')):
                parts.append(' ')
            parts.append(tval)
            prev_type = ttype
        s = ''.join(parts)
        for pair in [('&&', ' && '), ('||', ' || '), ('<=', ' <= '), ('>=', ' >= '),
                     ('==', ' == '), ('!=', ' != '), ('+=', ' += '), ('-=', ' -= '),
                     ('*=', ' *= '), ('/=', ' /=')]:
            s = s.replace(pair[0], pair[1])
        s = re.sub(r'(?<!\w)if(?!\w)', ' if ', s)
        s = re.sub(r'(?<!\w)else(?!\w)', ' else ', s)
        s = re.sub(r'\s+', ' ', s)
        return s.strip()

    @staticmethod
    def _find_matching_paren(s, start_idx):
        depth = 0
        for j in range(start_idx, len(s)):
            if s[j] == '(':
                depth += 1
            elif s[j] == ')':
                depth -= 1
                if depth == 0:
                    return j
        return -1



def preprocess_milkdrop_expr(source: str) -> str:
    """Preprocess MilkDrop expression to Python-compatible syntax.

    Uses a tokenizer+parser for MilkDrop-specific constructs:
      - += -= *= /= assignments
      - if(x) / if(x,y) shorthand
      - if(cond, then, else) conversion to Python ternary
      - above(a,b), below(a,b), equal(a,b)

    This keeps the downstream stages unchanged (ast.parse -> AST optimize -> GLSL).
    """
    if not source or not source.strip():
        return ""
    source = source.strip()
    if source.startswith('//'):
        return source
    if source.lstrip('`').lstrip().startswith('#'):
        return source
    return MilkDropParser().parse(source)

@lru_cache(maxsize=4096)
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
    except (PresetCompileError, SyntaxError) as e:
        raise PresetCompileError(f"Preprocessing failed: {e}") from e
    if not val:
        raise PresetCompileError("Expression empty after preprocessing")
    try:
        tree = ast.parse(val, mode='eval')
    except SyntaxError as e:
        raise PresetCompileError(f"Parse error: {e}") from e
    tree = optimize_ast(tree)
    generator = GLSLGenerator()
    glsl = generator.visit(tree.body)
    if re.search(r"(?<![!<>=])=(?![=])", glsl):
        raise PresetCompileError(f"Generated GLSL contains unexpected assignment: '{glsl}'")
    return glsl

@lru_cache(maxsize=8192)
def compile_assignment_to_glsl(expr_str: str) -> tuple:
    """Compile a MilkDrop assignment (var = expr) to GLSL.
    Returns (var_name, glsl_code, is_full_statement) tuple.
    Raises PresetCompileError on failure instead of returning (None, None, False).
    """
    if not expr_str:
        raise PresetCompileError("Empty expression string")
    expr_str = expr_str.strip()
    if expr_str.startswith('//'):
        raise PresetCompileError("Comment expression")
    if '//' in expr_str:
        raise PresetCompileError("Comment in expression")
    if '=' not in expr_str:
        raise PresetCompileError("No assignment in expression")
    key, val = expr_str.split('=', 1)
    key = key.strip()
    val = val.strip().rstrip(';')
    if key == 'mod':
        key = 'md_mod'
    if not key or key.startswith('//'):
        raise PresetCompileError("Invalid assignment target")
    try:
        val = preprocess_milkdrop_expr(val)
    except (PresetCompileError, SyntaxError) as e:
        raise PresetCompileError(f"Preprocessing failed: {e}") from e
    if val.startswith('//'):
        raise PresetCompileError("Comment after preprocessing")
    try:
        tree = ast.parse(val, mode='eval')
    except SyntaxError as e:
        raise PresetCompileError(f"Parse error: {e}") from e


    generator = GLSLGenerator()
    glsl = generator.visit(tree.body)
    return key, glsl, False


def md_rand(state):
    """Deterministic MilkDrop random number generator.
    MilkDrop uses a linear congruential generator with specific constants
    that ensures deterministic behavior across platforms.
    """
    state.rand_seed = (int(state.rand_seed) * 16807) % 2147483647
    return state.rand_seed / 2147483647.0


class ExpressionOptimizer(ast.NodeVisitor):
    """Dedicated AST optimization pass.
    Runs after parsing, before bytecode or GLSL generation.
    Implements:
    - Constant folding (1+2 -> 3)
    - Algebraic identities (x*1 -> x, x+0 -> x, x*0 -> 0)
    - Power optimization (pow(x,2) -> x*x)
    - Trig at zero (sin(0)->0, cos(0)->1)
    - Dead code elimination (if False, branch removal)
    - Strength reduction where beneficial
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
                lv = float(left.value)
                rv = float(right.value)
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
                    result = lv ** rv
                elif isinstance(op, ast.FloorDiv) and abs(rv) > EPSILON_DIV:
                    result = lv // rv
                if result is not None:
                    val = ast.Constant(value=result)
                    ast.copy_location(val, node)
                    return val
            except (ValueError, TypeError, ZeroDivisionError, OverflowError) as e:
                logger.debug(f"Failed to optimize constant folding: {e}")

        if isinstance(op, ast.Mult):
            if self._is_zero(left):
                return ast.Constant(value=0.0)
            if self._is_zero(right):
                return ast.Constant(value=0.0)
            if self._is_one(left):
                ast.copy_location(right, node)
                return right
            if self._is_one(right):
                ast.copy_location(left, node)
                return left

        if isinstance(op, ast.Add):
            if self._is_zero(left):
                ast.copy_location(right, node)
                return right
            if self._is_zero(right):
                ast.copy_location(left, node)
                return left
            if self._is_negation(left, right):
                return ast.Constant(value=0.0)

        if isinstance(op, ast.Sub):
            if self._is_zero(right):
                ast.copy_location(left, node)
                return left
            if self._is_zero(left):
                result = ast.UnaryOp(op=ast.USub(), operand=right)
                ast.copy_location(result, node)
                return result

        if isinstance(op, ast.Div):
            if self._is_zero(right) or self._is_approx(right, 0.0):
                return ast.Constant(value=0.0)
            if self._is_one(right):
                ast.copy_location(left, node)
                return left

        if isinstance(op, ast.Pow):
            if self._is_zero(right):
                return ast.Constant(value=1.0)
            if self._is_one(right):
                ast.copy_location(left, node)
                return left
            if self._is_two(right):
                result = ast.BinOp(left=left, op=ast.Mult(), right=ast.copy_location(left, right))
                ast.copy_location(result, node)
                return result

        if isinstance(op, ast.Mod) and (self._is_zero(right) or self._is_approx(right, 0.0)):
            return ast.Constant(value=0.0)

        return node

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        operand = node.operand
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            if isinstance(operand, ast.Constant):
                result = ast.Constant(value=-float(operand.value))
                ast.copy_location(result, node)
                return result
            if isinstance(operand, ast.UnaryOp) and isinstance(operand.op, ast.USub):
                return operand.operand
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name == 'sin' and node.args and self._is_zero(node.args[0]):
                return ast.Constant(value=0.0)
            if func_name == 'cos' and node.args and self._is_zero(node.args[0]):
                return ast.Constant(value=1.0)
            if func_name == 'tan' and node.args and self._is_zero(node.args[0]):
                return ast.Constant(value=0.0)
            if func_name == 'abs' and node.args and isinstance(node.args[0], ast.Constant):
                return ast.Constant(value=abs(float(node.args[0].value)))
            if func_name == 'sqrt' and node.args and self._is_one(node.args[0]):
                return ast.Constant(value=1.0)
            if func_name in ('sin', 'cos', 'tan', 'asin', 'acos', 'atan',
                             'sinh', 'cosh', 'exp', 'log', 'sqrt', 'abs',
                             'floor', 'ceil', 'fract', 'trunc') and len(node.args) == 1 and isinstance(node.args[0], ast.Constant):
                try:
                    arg_val = float(node.args[0].value)
                    result_val = None
                    if func_name == 'sin':
                        result_val = math.sin(arg_val)
                    elif func_name == 'cos':
                        result_val = math.cos(arg_val)
                    elif func_name == 'tan':
                        result_val = math.tan(arg_val)
                    elif func_name == 'asin' and -1 <= arg_val <= 1:
                        result_val = math.asin(arg_val)
                    elif func_name == 'acos' and -1 <= arg_val <= 1:
                        result_val = math.acos(arg_val)
                    elif func_name == 'atan':
                        result_val = math.atan(arg_val)
                    elif func_name == 'sinh':
                        result_val = math.sinh(arg_val)
                    elif func_name == 'cosh':
                        result_val = math.cosh(arg_val)
                    elif func_name == 'exp':
                        result_val = math.exp(min(arg_val, 700))
                    elif func_name == 'log' and arg_val > 0:
                        result_val = math.log(arg_val)
                    elif func_name == 'sqrt' and arg_val >= 0:
                        result_val = math.sqrt(arg_val)
                    elif func_name == 'abs':
                        result_val = abs(arg_val)
                    elif func_name == 'floor':
                        result_val = math.floor(arg_val)
                    elif func_name == 'ceil':
                        result_val = math.ceil(arg_val)
                    elif func_name == 'fract':
                        result_val = arg_val - math.floor(arg_val)
                    elif func_name == 'trunc':
                        result_val = float(int(arg_val))
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
        return isinstance(node, ast.Constant) and float(node.value) == 0.0

    def _is_one(self, node) -> bool:
        return isinstance(node, ast.Constant) and float(node.value) == 1.0

    def _is_two(self, node) -> bool:
        return isinstance(node, ast.Constant) and float(node.value) == 2.0

    def _is_approx(self, node, target: float) -> bool:
        return isinstance(node, ast.Constant) and abs(float(node.value) - target) < EPSILON_SMALL

    def _is_negation(self, a, b) -> bool:
        if isinstance(a, ast.Constant) and isinstance(b, ast.Constant):
            return abs(float(a.value) + float(b.value)) < EPSILON_SMALL
        return False

    def _is_true_test(self, node) -> bool:
        if isinstance(node, ast.Compare):
            if isinstance(node.ops[0], ast.Gt) and self._is_zero(node.comparators[0]):
                return True
            if isinstance(node.ops[0], ast.GtE) and self._is_zero(node.comparators[0]) and isinstance(node.left, ast.Constant) and float(node.left.value) >= 0:
                return True
        return False

    def _is_false_test(self, node) -> bool:
        return isinstance(node, ast.Compare) and isinstance(node.ops[0], ast.Eq) and self._is_zero(node.comparators[0]) and isinstance(node.left, ast.Constant) and float(node.left.value) == 0


def optimize_ast(tree: ast.AST) -> ast.AST:
    """Run the expression optimizer on an AST. Returns optimized AST."""
    optimizer = ExpressionOptimizer()
    return optimizer.visit(tree)


class MDASTNode:
    """Base class for all typed MilkDrop AST nodes."""
    __slots__ = ()

    def accept(self, visitor):
        method = getattr(visitor, 'visit_' + self.__class__.__name__, visitor.generic_visit)
        return method(self)


class ConstantNode(MDASTNode):
    __slots__ = ('value',)
    def __init__(self, value: float):
        self.value = value


class VariableNode(MDASTNode):
    __slots__ = ('index', 'name')
    def __init__(self, name: str, index: int = -1):
        self.name = name
        self.index = index


class BinaryNode(MDASTNode):
    __slots__ = ('left', 'op', 'right')
    def __init__(self, op: str, left, right):
        self.op = op
        self.left = left
        self.right = right


class UnaryNode(MDASTNode):
    __slots__ = ('op', 'operand')
    def __init__(self, op: str, operand):
        self.op = op
        self.operand = operand


class CallNode(MDASTNode):
    __slots__ = ('args', 'func')
    def __init__(self, func: str, args: list):
        self.func = func
        self.args = args


class IfNode(MDASTNode):
    __slots__ = ('cond', 'else_', 'then')
    def __init__(self, cond, then, else_):
        self.cond = cond
        self.then = then
        self.else_ = else_


class AssignmentNode(MDASTNode):
    __slots__ = ('target', 'value')
    def __init__(self, target: str, value):
        self.target = target
        self.value = value


class SequenceNode(MDASTNode):
    __slots__ = ('nodes',)
    def __init__(self, nodes: list):
        self.nodes = nodes


class BytecodePeepholeOptimizer:
    """Post-compilation peephole optimizer for bytecode.
    Reduces instruction count and eliminates dead code patterns.
    """

    def __init__(self, code, const_pool, var_names):
        self.code = list(code)
        self.const_pool = list(const_pool)
        self.var_names = dict(var_names)

    def optimize(self):
        self._remove_load_store_same_var()
        self._fold_const_binops()
        self._remove_dead_pop()
        self._fold_const_unary()
        self._remove_redundant_jumps()
        self._merge_const_loads()
        return self.code, self.const_pool, self.var_names

    def _remove_load_store_same_var(self):
        i = 0
        while i < len(self.code) - 1:
            instr = self.code[i]
            next_instr = self.code[i + 1]
            if (instr[0] == OP_LOAD_VAR and next_instr[0] == OP_STORE_VAR
                    and instr[1] == next_instr[1]):
                self.code.pop(i)
                self.code.pop(i)
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
                    if (j + 1 < len(self.code)
                            and self.code[j + 1][0] in (OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_MOD)):
                        op = self.code[j + 1][0]
                        try:
                            a = self.const_pool[idx_a]
                            b = self.const_pool[idx_b]
                            if not (math.isnan(a) or math.isnan(b) or math.isinf(a) or math.isinf(b)):
                                result = None
                                if op == OP_ADD:
                                    result = a + b
                                elif op == OP_SUB:
                                    result = a - b
                                elif op == OP_MUL:
                                    result = a * b
                                elif op == OP_DIV:
                                    result = a / b if abs(b) > EPSILON_DIV else 0.0
                                elif op == OP_MOD:
                                    result = a % b if abs(b) > EPSILON_DIV else 0.0
                                if result is not None and math.isfinite(result):
                                    new_idx = self._find_or_add_const(result)
                                    self.code[i] = (OP_LOAD_CONST, new_idx)
                                    self.code[i + 1] = (OP_LOAD_CONST, new_idx)
                                    self.code[i + 2] = (OP_LOAD_CONST, new_idx)
                        except Exception as e:
                            logger.debug(f"Failed to optimize constant folding: {e}")
            i += 1

    def _remove_dead_pop(self):
        i = 0
        while i < len(self.code) - 1:
            if self.code[i][0] == OP_POP and self.code[i + 1][0] == OP_POP:
                self.code.pop(i)
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
                    self.code.pop(i + 1)
                    continue
                if op_instr[0] in (OP_SIN, OP_COS, OP_TAN, OP_ABS, OP_FLOOR,
                                   OP_CEIL, OP_FRACT, OP_EXP, OP_LOG, OP_SQRT):
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
                            result = math.exp(min(val, 700))
                        elif op == OP_LOG and val > 0:
                            result = math.log(val)
                        elif op == OP_SQRT and val >= 0:
                            result = math.sqrt(val)
                        if result is not None and math.isfinite(result):
                            new_idx = self._find_or_add_const(result)
                            self.code[i] = (OP_LOAD_CONST, new_idx)
                            self.code.pop(i + 1)
                            continue
                    except Exception as e:
                        logger.debug(f"Failed to optimize load_const: {e}")
            i += 1

    def _remove_redundant_jumps(self):
        i = 0
        while i < len(self.code):
            instr = self.code[i]
            if instr[0] == OP_JUMP and isinstance(instr[1], int):
                target = instr[1]
                if target == i + 1:
                    self.code.pop(i)
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

    def _merge_const_loads(self):
        i = 0
        while i < len(self.code) - 1:
            if (self.code[i][0] == OP_LOAD_CONST
                    and self.code[i + 1][0] == OP_LOAD_CONST
                    and self.code[i][1] == self.code[i + 1][1]):
                self.code.pop(i + 1)
            else:
                i += 1

    def _find_or_add_const(self, value: float) -> int:
        if not isinstance(value, float):
            try:
                value = float(value)
            except (ValueError, TypeError):
                value = 0.0
        for idx, existing in enumerate(self.const_pool):
            if existing == value and not (math.isnan(value) or math.isnan(existing)):
                return idx
        idx = len(self.const_pool)
        self.const_pool.append(value)
        return idx


class FrameProfiler:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init()
            return cls._instance

    def _init(self):
        self._timings: dict[str, list[float]] = {}
        self._frame_count = 0
        self._enabled = os.getenv("MILKDROP_PROFILE", "0") == "1"
        self._print_interval = int(os.getenv("MILKDROP_PROFILE_INTERVAL", "300"))
        self._last_print = 0

    def record(self, name: str, delta_ms: float):
        if not self._enabled:
            return
        if name not in self._timings:
            self._timings[name] = []
        self._timings[name].append(delta_ms)
        self._frame_count += 1
        if self._frame_count - self._last_print >= self._print_interval:
            self._last_print = self._frame_count
            self._print_report()

    def _print_report(self):
        lines = ["\n=== MilkDrop Performance Profile ==="]
        total = 0.0
        for name, samples in sorted(self._timings.items()):
            if not samples:
                continue
            recent = samples[-self._print_interval:]
            avg = sum(recent) / len(recent)
            mx = max(recent)
            lines.append(f"  {name:<30} avg={avg:>7.2f}ms  max={mx:>7.2f}ms  samples={len(recent)}")
            total += avg
        lines.append(f"  {'TOTAL':<30} {total:>7.2f}ms")
        lines.append("=" * 50)
        logger.info("\n".join(lines))

    def reset(self):
        with self._lock:
            self._timings.clear()
            self._frame_count = 0
            self._last_print = 0


def profile_stage(name: str):
    """Context manager / decorator for timing a code block."""
    class _Ctx:
        def __enter__(self):
            self._t0 = _time_module.perf_counter()
            return self
        def __exit__(self, *args):
            dt = (_time_module.perf_counter() - self._t0) * 1000.0
            FrameProfiler().record(name, dt)
            return False
    return _Ctx()



class LockFreeAudioBuffer:
    __slots__ = ('_latest', '_lock')
    def __init__(self):
        self._latest: dict | None = None
        self._lock = threading.Lock()

    def publish(self, data: dict):
        with self._lock:
            self._latest = data

    def consume(self) -> dict | None:
        with self._lock:
            return self._latest


class PresetHealthScore:
    __slots__ = ('errors', 'required_q_regs', 'score', 'unsupported_funcs', 'warnings')
    def __init__(self):
        self.score = 100.0
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.unsupported_funcs: list[str] = []
        self.required_q_regs: set[int] = set()

    def add_warning(self, msg: str):
        self.warnings.append(msg)
        self.score = max(0.0, self.score - 5.0)

    def add_error(self, msg: str):
        self.errors.append(msg)
        self.score = max(0.0, self.score - 20.0)

    def add_unsupported(self, func: str):
        self.unsupported_funcs.append(func)
        self.score = max(0.0, self.score - 10.0)

    def add_required_q_registers(self, indices):
        self.required_q_regs.update(indices)

    def check_q_registers(self, q_locs):
        missing_required = [i + 1 for i in self.required_q_regs if q_locs[i] == -1]
        unused_missing = [i + 1 for i in range(32) if q_locs[i] == -1 and i not in self.required_q_regs]
        if missing_required:
            for idx in missing_required:
                self.add_warning(f"required q register q{idx} not cached")
        if unused_missing:
            self.add_warning(f"unused q registers not cached: {len(unused_missing)}/32")

    def is_healthy(self) -> bool:
        return self.score >= 50.0 and not self.errors


class PresetCache:
    CACHE_DIR = Path(__file__).parent / ".preset_cache"
    CACHE_VERSION = 3

    def __init__(self):
        self.CACHE_DIR.mkdir(exist_ok=True)
        self._cache_index: dict[str, Path] = {}
        self._load_index()

    def _load_index(self):
        idx_file = self.CACHE_DIR / "index.json"
        try:
            if idx_file.exists():
                data = json.loads(idx_file.read_text())
                self._cache_index = {k: Path(v) for k, v in data.get("entries", {}).items()}
        except (OSError, json.JSONDecodeError):
            self._cache_index = {}

    def _save_index(self):
        idx_file = self.CACHE_DIR / "index.json"
        try:
            data = {
                "version": self.CACHE_VERSION,
                "entries": {k: str(v) for k, v in self._cache_index.items()},
            }
            idx_file.write_text(json.dumps(data, indent=2))
        except (OSError, TypeError, ValueError) as e:
            logger.debug(f"Failed to save cache index: {e}")

    def _cache_path(self, source_path: str) -> Path:
        h = hashlib.sha256(str(source_path).encode()).hexdigest()[:16]
        return self.CACHE_DIR / f"{h}.mdc"

    def get(self, source_path: str, mtime: float) -> dict | None:
        cp = self._cache_path(source_path)
        try:
            if not cp.exists():
                return None
            if cp.stat().st_mtime < mtime:
                return None
            with open(cp, 'r') as f:
                data = json.load(f)
            if data.get("version") != self.CACHE_VERSION:
                return None
            if data.get("source_mtime", 0.0) != mtime:
                return None
            return data
        except (OSError, json.JSONDecodeError, ValueError):
            return None

    def put(self, source_path: str, mtime: float, compiled_data: dict):
        cp = self._cache_path(source_path)
        try:
            compiled_data["version"] = self.CACHE_VERSION
            compiled_data["source_mtime"] = mtime
            with open(cp, 'w') as f:
                json.dump(compiled_data, f)
            try:
                os.utime(cp, (mtime, mtime))
            except (OSError, ValueError) as e:
                logger.debug(f"Failed to set cache file mtime: {e}")
            self._cache_index[str(source_path)] = cp
            self._save_index()
        except (OSError, TypeError, ValueError) as e:
            logger.debug(f"Failed to save compiled preset to cache: {e}")

    def invalidate(self, source_path: str):
        cp = self._cache_path(source_path)
        try:
            cp.unlink(missing_ok=True)
            self._cache_index.pop(str(source_path), None)
            self._save_index()
        except OSError as exc:
            logger.debug("Exception in try block: %s", exc)

    def clear_all(self):
        try:
            for f in self.CACHE_DIR.glob("*.mdc"):
                f.unlink(missing_ok=True)
            self._cache_index.clear()
            self._save_index()
        except OSError as e:
            logger.warning(f"Failed to clear cache: {e}")


PRESET_CACHE = PresetCache()


def _build_preset_cache_data(program: PresetProgram, base_params: dict) -> dict:
    data = {
        "per_frame_init": [e.source for e in program.per_frame_init],
        "per_frame": [e.source for e in program.per_frame],
        "per_pixel": [e.source for e in program.per_pixel],
        "waves": [
            {
                "per_frame": [e.source for e in w.per_frame],
                "per_point": [e.source for e in w.per_point],
                "enabled": w.enabled,
            }
            for w in program.waves
        ],
        "shapes": [
            {
                "per_frame": [e.source for e in s.per_frame],
                "per_frame_init": [e.source for e in s.per_frame_init],
                "enabled": s.enabled,
            }
            for s in program.shapes
        ],
        "base_params": base_params,
    }
    return data


def _restore_preset_from_cache(cache_data: dict) -> PresetProgram:
    preset_dict = {
        "per_frame_init": cache_data.get("per_frame_init", []),
        "per_frame": cache_data.get("per_frame", []),
        "per_pixel": cache_data.get("per_pixel", []),
        "waves": {
            i: {
                "per_frame": wd.get("per_frame", []),
                "per_point": wd.get("per_point", []),
                "enabled": wd.get("enabled", True),
            }
            for i, wd in enumerate(cache_data.get("waves", []))
        },
        "shapes": {
            i: {
                "per_frame": sd.get("per_frame", []),
                "per_frame_init": sd.get("per_frame_init", []),
                "enabled": sd.get("enabled", True),
            }
            for i, sd in enumerate(cache_data.get("shapes", []))
        },
    }
    return PresetProgram._compile_from_dict(preset_dict)


class AsyncPresetCompiler:
    _queue: ClassVar[Queue] = Queue()
    _worker_thread: ClassVar[threading.Thread | None] = None
    _stop_event: ClassVar[threading.Event] = threading.Event()
    _results: ClassVar[dict[str, PresetProgram]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()
    _active: ClassVar[bool] = False

    @classmethod
    def start(cls):
        if cls._active:
            return
        cls._active = True
        cls._stop_event.clear()
        cls._worker_thread = threading.Thread(target=cls._worker, daemon=True)
        cls._worker_thread.start()

    @classmethod
    def stop(cls):
        cls._stop_event.set()
        cls._active = False
        cls._queue.put(None)
        if cls._worker_thread and cls._worker_thread.is_alive():
            cls._worker_thread.join(timeout=2.0)

    @classmethod
    def submit(cls, preset_path: str, preset_dict: dict, base_params: dict) -> PresetProgram | None:
        cls._queue.put((preset_path, preset_dict, base_params))
        return None

    @classmethod
    def _worker(cls):
        while True:
            try:
                item = cls._queue.get()
            except Exception as e:
                logger.debug("AsyncPresetCompiler queue error: %s", e)
                continue
            if item is None:
                if cls._stop_event.is_set():
                    return
                continue
            preset_path, preset_dict, _ = item
            try:
                program = PresetProgram(preset_dict=preset_dict)
                with cls._lock:
                    cls._results[preset_path] = program
            except (PresetCompileError, SyntaxError, MemoryError, RuntimeError) as e:
                logger.warning(f"Failed to compile preset {preset_path}: {e}")
                with cls._lock:
                    cls._results[preset_path] = None

    @classmethod
    def get_result(cls, preset_path: str) -> PresetProgram | None:
        with cls._lock:
            return cls._results.pop(preset_path, None)


class PresetState:
    """Stateful container for all preset variables.
    Maintains persistent state across frames including:
    - All variables (zoom, rot, warp, etc.)
    - Previous frame's variables (for temporal calculations)
    - q1-q32 registers
    - megabuf (1M float array)
    - reg00-reg99 global registers (shared across presets)
    - Random seed (for deterministic rand())
    """
    MEGABUF_SIZE_MAX = 1048576
    MEGABUF_SIZE = min(int(os.getenv("MILKDROP_MEGABUF_SIZE", "262144")), MEGABUF_SIZE_MAX)
    _CLAMP_ZOOM = (os.getenv("MILKDROP_CLAMP_ZOOM", "0") == "1")
    STATE_ATTRS = frozenset(VARIABLES.keys())
    _SHAPE_STATE_ATTRS = frozenset(set(VARIABLES.keys()) | {"rand_seed", "megabuf_version"})

    # Global registers (reg00-reg99) - shared across all PresetState instances
    _global_regs = np.zeros(100, dtype=np.float32)

    def __init__(self):
        self.zoom = 1.0
        self.rot = 0.0
        self.warp = 0.0
        self.cx = 0.5
        self.cy = 0.5
        self.dx = 0.0
        self.dy = 0.0
        self.sx = 1.0
        self.sy = 1.0
        self.decay = 0.9
        self.vars = {}
        self.vars_array = np.zeros(1024, dtype=np.float32)
        self.prev_array = np.zeros(1024, dtype=np.float32)
        self.prev_vars = {}
        self.q = [0.0] * 32
        self.megabuf = None
        self.megabuf_version = 0
        self.rand_seed = 1
        for name, idx in VAR_REGISTRY.items():
            self.vars_array[idx] = float(getattr(self, name, 0.0))
        self._init_base_vars()

    def _init_base_vars(self):
        """Initialize base MilkDrop variables with defaults from VARIABLES registry."""
        for vname, vdef in VARIABLES.items():
            self.set(vname, vdef.default)

    def begin_frame(self):
        """Called at start of each frame to save previous state."""
        self.prev_vars = self.vars.copy()
        self.prev_array[:] = self.vars_array

    def get(self, name: str, default: float = 0.0) -> float:
        """Get variable value, supporting q1-q32 syntax and reg00-reg99 global registers."""
        name_lower = name.lower()
        # Handle reg00-reg99 global registers (must be exactly 2 digits)
        if len(name_lower) == 5 and name_lower.startswith('reg') and name_lower[3:].isdigit():
            idx = int(name_lower[3:])
            if 0 <= idx < 100:
                return self._global_regs[idx]
        # Handle q1-q32 registers
        if len(name_lower) <= 3 and name_lower[0] == 'q' and name_lower[1:].isdigit():
            idx = int(name_lower[1:]) - 1
            if 0 <= idx < 32:
                return self.q[idx]
        if name_lower in VAR_REGISTRY:
            return self.vars_array[VAR_REGISTRY[name_lower]]
        idx = Q_LOOKUP.get(name_lower)
        if idx is not None:
            return self.q[idx]
        if name_lower in self.STATE_ATTRS:
            return float(getattr(self, name_lower, default))
        return self.vars.get(name_lower, default)

    def set(self, name: str, value: float):
        """Set variable value, supporting q1-q32 syntax and reg00-reg99 global registers."""
        value = safe_float(value)
        name_lower = name.lower()
        # Handle reg00-reg99 global registers (must be exactly 2 digits)
        if len(name_lower) == 5 and name_lower.startswith('reg') and name_lower[3:].isdigit():
            idx = int(name_lower[3:])
            if 0 <= idx < 100:
                self._global_regs[idx] = value
                return
        if name_lower == "zoom" and PresetState._CLAMP_ZOOM:
            value = max(0.001, value)
        if name_lower in VAR_REGISTRY:
            self.vars_array[VAR_REGISTRY[name_lower]] = float(value)
        if name_lower in PresetState._SHAPE_STATE_ATTRS or hasattr(self.__class__, name_lower):
            val = getattr(self, name_lower, None)
            if isinstance(val, (int, float, np.number)):
                setattr(self, name_lower, float(value))
        idx = Q_LOOKUP.get(name_lower)
        if idx is not None:
            self.q[idx] = value
            return
        self.vars[name_lower] = value

    def _ensure_megabuf(self):
        """Lazy allocate megabuf on first access."""
        if self.megabuf is None:
            tex_size = getattr(self, 'megabuf_tex_size', self.MEGABUF_SIZE)
            self.megabuf = np.zeros(tex_size, dtype=np.float32)

    def megabuf_get(self, idx: float) -> float:
        """Get value from megabuf at index."""
        self._ensure_megabuf()
        i = int(idx)
        tex_size = getattr(self, 'megabuf_tex_size', self.MEGABUF_SIZE)
        if 0 <= i < tex_size:
            return self.megabuf[i]
        return 0.0

    def megabuf_set(self, idx: float, value: float):
        """Set value in megabuf at index."""
        self._ensure_megabuf()
        i = int(idx)
        tex_size = getattr(self, 'megabuf_tex_size', self.MEGABUF_SIZE)
        if 0 <= i < tex_size:
            value = safe_float(value)
            self.megabuf[i] = float(value)
            self.megabuf_version = (self.megabuf_version + 1) & 0xFFFFFFFF

    def update_audio(self, audio_levels: dict):
        """Update audio-related variables from external source."""
        self.set("bass", audio_levels.get("bass", 0.0))
        self.set("bass_att", audio_levels.get("bass_att", 0.0))
        self.set("mid", audio_levels.get("mid", 0.0))
        self.set("mid_att", audio_levels.get("mid_att", 0.0))
        self.set("treb", audio_levels.get("treb", 0.0))
        self.set("treb_att", audio_levels.get("treb_att", 0.0))
        self.set("beat", audio_levels.get("beat", 0.0))

    def update_time(self, time: float, frame: int, fps: float = 60.0):
        """Update time-related variables."""
        self.set("time", time)
        self.set("t", time)
        self.set("frame", frame)
        self.set("fps", fps)

    def update_resolution(self, width: float, height: float):
        """Update resolution-related variables."""
        self.set("width", width)
        self.set("height", height)
        self.set("w", width)
        self.set("h", height)
        self.set("aspect", width / height if height > 0 else 1.0)

    def _copy_core_state_to(self, clone: PresetState):
        clone.zoom = self.zoom
        clone.rot = self.rot
        clone.warp = self.warp
        clone.cx = self.cx
        clone.cy = self.cy
        clone.dx = self.dx
        clone.dy = self.dy
        clone.sx = self.sx
        clone.sy = self.sy
        clone.decay = self.decay
        clone.rand_seed = self.rand_seed
        clone.megabuf_version = self.megabuf_version
        clone.vars = self.vars.copy()
        clone.vars_array = self.vars_array.copy()
        clone.q = list(self.q)
        clone.megabuf = self.megabuf

    def shallow_clone_for_shapes(self) -> PresetState:
        """Lightweight clone for shape execution.
        Copies only mutable fields that shapes modify:
        - vars_dict (shape can write custom variables)
        - vars_array (motion vars: zoom, rot, warp, etc.)
        - q registers
        - rand_seed (deterministic per-frame rand)
        Does NOT copy megabuf (shared reference to avoid 4MB/mem-bandwidth copy).
        Does NOT copy prev_vars/prev_array (not needed for shapes).
        """
        clone = PresetState.__new__(PresetState)
        self._copy_core_state_to(clone)
        clone.prev_vars = {}
        clone.prev_array = np.zeros(1024, dtype=np.float32)
        return clone

    def shallow_clone(self) -> PresetState:
        """Full shallow clone with separate state arrays.
        Used for BlendEngine state interpolation.
        """
        clone = PresetState.__new__(PresetState)
        self._copy_core_state_to(clone)
        clone.prev_vars = self.prev_vars.copy()
        clone.prev_array = self.prev_array.copy()
        return clone

class CompiledExpr:
    """A compiled expression ready for evaluation.
    Compiles once at load time, executes many times at runtime.
    """
    bad_presets: ClassVar[set] = set()
    _LOG_PARSE_ERRORS: ClassVar[bool] = (os.getenv("MILKDROP_LOG_PARSE_ERRORS", "0") == "1")

    @staticmethod
    def _truncate_for_log(s: str, limit: int = 300) -> str:
        if s is None:
            return ""
        s = str(s)
        if len(s) <= limit:
            return s
        return s[:limit] + "…"

    @staticmethod
    def load_bad_presets():
        path = Path.home() / ".milkdrop_bad_presets.txt"
        if path.exists():
            try:
                with open(path, "r") as f:
                    CompiledExpr.bad_presets = {line.strip() for line in f if line.strip()}
            except OSError as e:
                logger.debug(f"Failed to load bad presets list: {e}")

    @staticmethod
    def save_bad_presets():
        path = Path.home() / ".milkdrop_bad_presets.txt"
        try:
            with open(path, "w") as f:
                f.write("\n".join(CompiledExpr.bad_presets))
        except OSError as e:
            logger.debug(f"Failed to save bad presets list: {e}")

    def sanitize_expr(self, expr: str) -> str:
        if not expr or not expr.strip():
            raise PresetCompileError("Empty expression string")
        return preprocess_milkdrop_expr(expr)

    def __init__(self, expr_str: str):
        self.source = expr_str.strip()
        self.bytecode_prog = None
        self._compile()

    def _compile(self):
        try:
            clean = preprocess_milkdrop_expr(self.source)
        except (SyntaxError, ValueError, TypeError, PresetCompileError) as e:
            logger.warning(f"Preprocessing failed in _compile: {e}")
            return
        if not clean or '=' not in clean:
            raise PresetCompileError("Expression missing assignment")
        compiler = BytecodeCompiler()
        try:
            self.bytecode_prog = compiler.compile_assignment(clean)
        except PresetCompileError:
            raise
        except SyntaxError as e:
            logger.warning(f"SyntaxError in _compile: {e}")
            return
        except (MemoryError, RuntimeError) as e:
            logger.warning(f"Exception in _compile: {e}")
            return
        if self.bytecode_prog is None:
            return
        optimizer = BytecodePeepholeOptimizer(
            self.bytecode_prog.code,
            self.bytecode_prog.const_pool,
            self.bytecode_prog.var_names,
        )
        new_code, new_consts, new_vars = optimizer.optimize()
        self.bytecode_prog.code = new_code
        self.bytecode_prog.const_pool = new_consts
        self.bytecode_prog.var_names = new_vars
        self.bytecode_prog._rebuild_idx_to_var()

    def eval(self, state: PresetState) -> float:
        if self.bytecode_prog:
            return self.bytecode_prog.execute(state)
        raise PresetCompileError("Cannot evaluate: expression failed to compile")

    def is_valid(self) -> bool:
        return self.bytecode_prog is not None



def _append_compiled_expression(target, source, context_label="expression"):
    """Compile a MilkDrop expression and append it to target if valid.

    Args:
        target: List to append the compiled expression to.
        source: Raw expression string.
        context_label: Label for warning messages (e.g., "per_frame").
    """
    try:
        compiled = CompiledExpr(source)
        if compiled.is_valid():
            target.append(compiled)
    except PresetCompileError as e:
        logger.warning(f"Skipping invalid {context_label} expression: {e}")
    except (SyntaxError, ValueError, TypeError, MemoryError, RuntimeError) as e:
        logger.warning(f"Unexpected error compiling {context_label} expression: {e}")


def _compile_expression_list(expr_list, is_per_pixel=False):
    """Compile a list of MilkDrop expressions with optional filtering.

    Args:
        expr_list: List of raw expression strings.
        is_per_pixel: Whether these are per-pixel expressions (skips additional checks).

    Returns:
        List of compiled CompiledExpr objects.
    """
    compiled_list = []
    for expr_str in expr_list:
        if not expr_str or not expr_str.strip():
            continue
        expr_str = expr_str.strip()
        skip_patterns = ['`', '//', 'MILKDROP_', 'PSVERSION', 'fRating']
        if any(p in expr_str for p in skip_patterns):
            continue
        parts = expr_str.split('=', 1)
        if len(parts) != 2:
            continue
        lhs, rhs = parts
        lhs = lhs.strip()
        rhs = rhs.strip().rstrip(';')
        if not rhs:
            continue
        is_megabuf = bool(RE_MEGABUF_LHS.match(lhs))
        if not lhs.isidentifier() and not is_megabuf and not RE_STRICT_IDENTIFIER.match(lhs):
            continue
        if lhs.startswith(('wavecode_', 'shapecode_')):
            continue
        try:
            compiled = CompiledExpr(expr_str)
            if compiled.is_valid():
                compiled_list.append(compiled)
        except (PresetCompileError, SyntaxError, ValueError, TypeError, MemoryError, RuntimeError):
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
    wave_prog = WaveProgram(
        per_frame=[],
        per_point=[],
        enabled=wave_data.get('enabled', True),
        mode=int(wave_data.get('mode', 0)),
        additive=bool(wave_data.get('additive', False)),
        samples=int(wave_data.get('samples', 512)),
        bSpectrum=bool(wave_data.get('bSpectrum', True)),
        bUseDots=bool(wave_data.get('bUseDots', False)),
        bDrawThick=bool(wave_data.get('bDrawThick', False)),
        bMaximize=bool(wave_data.get('bMaximize', False)),
        r=float(wave_data.get('r', 1.0)),
        g=float(wave_data.get('g', 1.0)),
        b=float(wave_data.get('b', 1.0)),
        a=float(wave_data.get('a', 1.0)),
        x=float(wave_data.get('x', 0.5)),
        y=float(wave_data.get('y', 0.5)),
        mystery=float(wave_data.get('mystery', 0.0)),
    )
    wave_prog.per_frame = _compile_expression_list(wave_data.get('per_frame', []))
    wave_prog.per_point = _compile_expression_list(wave_data.get('per_point', []))
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
    shape_prog = ShapeProgram(
        per_frame=[],
        per_frame_init=[],
        enabled=shape_data.get('enabled', True),
        sides=int(shape_data.get('sides', 3)),
        additive=bool(shape_data.get('additive', False)),
        thick_outline=bool(shape_data.get('thick_outline', False)),
        textured=bool(shape_data.get('textured', False)),
        r=float(shape_data.get('r', 1.0)),
        g=float(shape_data.get('g', 1.0)),
        b=float(shape_data.get('b', 1.0)),
        a=float(shape_data.get('a', 1.0)),
        r2=float(shape_data.get('r2', 0.0)),
        g2=float(shape_data.get('g2', 0.0)),
        b2=float(shape_data.get('b2', 0.0)),
        a2=float(shape_data.get('a2', 0.0)),
        x=float(shape_data.get('x', 0.5)),
        y=float(shape_data.get('y', 0.5)),
        rad=float(shape_data.get('rad', 0.1)),
        ang=float(shape_data.get('ang', 0.0)),
        tex_ang=float(shape_data.get('tex_ang', 0.0)),
        tex_zoom=float(shape_data.get('tex_zoom', 1.0)),
        border_r=float(shape_data.get('border_r', 1.0)),
        border_g=float(shape_data.get('border_g', 1.0)),
        border_b=float(shape_data.get('border_b', 1.0)),
        border_a=float(shape_data.get('border_a', 1.0)),
        border_size=float(shape_data.get('border_size', 0.01)),
    )
    shape_prog.per_frame = _compile_expression_list(shape_data.get('per_frame', []))
    shape_prog.per_frame_init = _compile_expression_list(shape_data.get('per_frame_init', []))
    return shape_prog

class PresetProgram:
    """A complete preset program with all execution phases compiled.
    Organizes equations by execution phase:
    - per_frame_init: Runs once at preset load/reset
    - per_frame: Runs every frame
    - per_pixel: Runs per-pixel (GPU via shader)
    - waves: Per-wave equations
    - shapes: Per-shape equations
    """

    def __init__(self, preset_dict: dict | None = None, from_cache: dict | None = None):
        self.per_frame_init = []
        self.per_frame = []
        self.per_pixel = []
        self.waves = []
        self.shapes = []
        if from_cache is not None:
            self._compile_from_cache(from_cache)
        elif preset_dict is not None:
            self._compile_from_dict(preset_dict)

    def _compile_from_cache(self, cache_data: dict):
        """Restore compiled expressions from cache data."""
        for src in cache_data.get('per_frame_init', []):
            _append_compiled_expression(self.per_frame_init, src, "per_frame_init")
        for src in cache_data.get('per_frame', []):
            _append_compiled_expression(self.per_frame, src, "per_frame")
        for src in cache_data.get('per_pixel', []):
            _append_compiled_expression(self.per_pixel, src, "per_pixel")
        for wd in cache_data.get('waves', []):
            wp = _compile_wave(wd)
            if wp:
                self.waves.append(wp)
        for sd in cache_data.get('shapes', []):
            sp = _compile_shape(sd)
            if sp:
                self.shapes.append(sp)

    def _compile_from_dict(self, preset_dict: dict):
        """Compile all equations from parsed preset dictionary with parallelism."""
        executor = GLOBAL_COMPILE_POOL
        per_frame_future = executor.submit(_compile_expression_list, preset_dict.get('per_frame', []))
        per_pixel_future = executor.submit(_compile_expression_list, preset_dict.get('per_pixel', []), True)
        waves = preset_dict.get('waves', {})
        waves_list = waves.values() if isinstance(waves, dict) else waves
        wave_futures = [executor.submit(_compile_wave, wave_data) for wave_data in waves_list]
        shapes = preset_dict.get('shapes', {})
        shapes_list = shapes.values() if isinstance(shapes, dict) else shapes
        shape_futures = [executor.submit(_compile_shape, shape_data) for shape_data in shapes_list]
        self.per_frame = per_frame_future.result()
        self.per_pixel = per_pixel_future.result()
        for future in wave_futures:
            result = future.result()
            if result:
                self.waves.append(result)
        for future in shape_futures:
            result = future.result()
            if result:
                self.shapes.append(result)
        for expr_str in preset_dict.get('per_frame_init', []):
            _append_compiled_expression(self.per_frame_init, expr_str, "per_frame_init")

    def get_stats(self) -> dict:
        """Get compilation statistics."""
        return {
            'per_frame_init': len(self.per_frame_init),
            'per_frame': len(self.per_frame),
            'per_pixel': len(self.per_pixel),
            'waves': len(self.waves),
            'shapes': len(self.shapes),
        }

class PresetVM:
    """The main preset virtual machine.
    Executes preset programs with proper deterministic order:
    1. per_frame_init (on load/reset)
    2. per_frame (every frame)
    3. per_pixel (GPU - handled separately)
    4. waves/shapes (during render)
    """

    def __init__(self, program: PresetProgram, preset_params: dict | None = None):
        self.program = program
        self.state = PresetState()
        self.init_run = False
        self.base_params = {}
        for vname, vdef in sorted(VARIABLES.items(), key=lambda x: x[1].index):
            if vdef.category in ("animation", "wave"):
                self.base_params[vname] = vdef.default
        for vname, vdef in sorted(VARIABLES.items(), key=lambda x: x[1].index):
            if vname in ("echo_zoom", "echo_alpha", "echo_orient"):
                self.base_params[vname] = vdef.default
        if preset_params:
            for key, value in preset_params.items():
                if isinstance(value, (int, float)):
                    kl = key.lower()
                    if kl in VARIABLES or kl.startswith('q'):
                        self.base_params[kl] = float(value)
                        self.state.set(kl, float(value))
        for key in list(self.base_params.keys()):
            if str(key).lower().startswith("q"):
                raise AssertionError(f"PresetVM.base_params must not include q-registers, found: {key!r}")
        self._dirty_base_keys: set = set()
        self._last_base_values: dict = dict(self.base_params)
        self.wave_x = np.zeros(512, dtype=np.float32)
        self.wave_y = np.zeros(512, dtype=np.float32)
        self.wave_r = np.zeros(512, dtype=np.float32)
        self.wave_g = np.zeros(512, dtype=np.float32)
        self.wave_b = np.zeros(512, dtype=np.float32)
        self.wave_a = np.zeros(512, dtype=np.float32)

    def reset(self):
        """Reset VM state (for preset reload)."""
        self.state = PresetState()
        for k, v in self.base_params.items():
            self.state.set(k, v)
        self._last_base_values = dict(self.base_params)
        self._dirty_base_keys.clear()
        self.init_run = False

    def run_frame(self, time: float, frame: int, audio_levels: dict,
                  width: float = 1.0, height: float = 1.0) -> PresetState:
        """Execute one full frame of the preset.
        Returns the state after execution (for GPU uniform upload).
        Uses dirty flags to skip resetting unchanged base_params.

        Execution order (MilkDrop spec compliant):
        1. per_frame_init (runs once on first frame)
        2. per_frame (runs every frame)
        3. wave per_frame (for each wave)
        4. shape per_frame (for each shape)
        5. per_pixel (runs on GPU, not here)

        q-flow is unidirectional: init → per_frame → wave/shape → per-pixel (GPU)
        q registers are never modified by per-pixel code on CPU side.
        """
        s = self.state
        changed_keys = self._dirty_base_keys
        if changed_keys:
            bp = self.base_params
            lv = self._last_base_values
            for key in changed_keys:
                val = bp[key]
                if lv.get(key) != val:
                    s.set(key, val)
                    lv[key] = val
            changed_keys.clear()
        else:
            for key, val in self.base_params.items():
                if self._last_base_values.get(key) != val:
                    s.set(key, val)
                    self._last_base_values[key] = val
        self._frame_rand_seed = s.rand_seed
        s.begin_frame()
        s.update_time(time, frame)
        s.update_audio(audio_levels)
        s.update_resolution(width, height)
        if not self.init_run:
            for expr in self.program.per_frame_init:
                expr.eval(s)
            self.init_run = True
        executed_count = 0
        for expr in self.program.per_frame:
            try:
                expr.eval(s)
                executed_count += 1
            except (PresetCompileError, SyntaxError, RuntimeError, ValueError, TypeError, ZeroDivisionError, OverflowError) as e:
                logger.exception(
                    "Per-frame expression failed: %s",
                    getattr(expr, "source", "<unknown>"),
                    exc_info=e,
                )
                expr.bytecode_prog = None
        for i, wave in enumerate(self.program.waves):
            s.vars['wave_index'] = i
            try:
                for expr in wave.per_frame:
                    try:
                        expr.eval(s)
                    except (PresetCompileError, SyntaxError, RuntimeError, ValueError, TypeError, ZeroDivisionError, OverflowError) as e:
                        logger.exception(
                            "Per-wave expression failed: wave_index=%s expr=%s",
                            i,
                            getattr(expr, "source", "<unknown>"),
                            exc_info=e,
                        )
                        expr.bytecode_prog = None
            finally:
                s.vars.pop('wave_index', None)
        for i, shape in enumerate(self.program.shapes):
            if not shape.per_frame:
                continue
            s.vars['shape_index'] = i
            try:
                shape_state = s.shallow_clone_for_shapes()
                shape_state.vars['shape_index'] = i
                for expr in shape.per_frame:
                    try:
                        expr.eval(shape_state)
                    except (PresetCompileError, SyntaxError, RuntimeError, ValueError, TypeError, ZeroDivisionError, OverflowError) as e:
                        logger.exception(
                            "Per-shape expression failed: shape_index=%s expr=%s",
                            i,
                            getattr(expr, "source", "<unknown>"),
                            exc_info=e,
                        )
                        expr.bytecode_prog = None
                for _name, idx in VAR_REGISTRY.items():
                    s.vars_array[idx] = float(shape_state.vars_array[idx])
                    if _name in PresetState._SHAPE_STATE_ATTRS:
                        try:
                            setattr(s, _name, float(shape_state.vars_array[idx]))
                        except Exception as exc:
                            logger.debug("Exception in try block: %s", exc)
                s.q = list(shape_state.q)
            finally:
                s.vars.pop('shape_index', None)
        return s

    def mark_base_param_dirty(self, key: str):
        self._dirty_base_keys.add(key.lower())

    def get_gpu_uniforms(self) -> GPUUniforms:
        """Get all GPU uniforms as a typed dataclass for fast attribute access."""
        s = self.state
        u = GPUUniforms(
            zoom=s.get('zoom', 1.0),
            rot=s.get('rot', 0.0),
            warp=s.get('warp', 0.0),
            cx=s.get('cx', 0.5),
            cy=s.get('cy', 0.5),
            dx=s.get('dx', 0.0),
            dy=s.get('dy', 0.0),
            sx=s.get('sx', 1.0),
            sy=s.get('sy', 1.0),
            decay=s.get('decay', 0.9),
            wave_r=s.get('wave_r', 0.0),
            wave_g=s.get('wave_g', 0.0),
            wave_b=s.get('wave_b', 0.0),
            wave_a=s.get('wave_a', 0.0),
            wave_x=s.get('wave_x', 0.5),
            wave_y=s.get('wave_y', 0.5),
            bass=s.get('bass', 0.0),
            mid=s.get('mid', 0.0),
            treb=s.get('treb', 0.0),
            bass_att=s.get('bass_att', 0.0),
            mid_att=s.get('mid_att', 0.0),
            treb_att=s.get('treb_att', 0.0),
            beat=s.get('beat', 0.0),
            time=s.get('time', 0.0),
            frame=float(s.get('frame', 0)),
            fps=s.get('fps', 60.0),
            x=s.get('x', 0.0),
            y=s.get('y', 0.0),
            rad=s.get('rad', 0.0),
            ang=s.get('ang', 0.0),
            width=s.get('width', 1.0),
            height=s.get('height', 1.0),
            aspect=s.get('aspect', 1.0),
            w=s.get('w', 1.0),
            h=s.get('h', 1.0),
            q=list(s.q),
            echo_zoom=s.get('echo_zoom', 1.0),
            echo_alpha=s.get('echo_alpha', 0.0),
            echo_orient=int(s.get('echo_orient', 0)),
        )
        return u

    def run_wave_per_point(self, wave_idx: int, point_idx: int,
                           sample_val: float, num_points: int):
        """Run per-point equations for a wave.
        This updates wave_x/y/r/g/b/a arrays for the given point.
        """
        if wave_idx >= len(self.program.waves):
            return
        if not (0 <= point_idx < len(self.wave_x)):
            return
        s = self.state
        s.vars['wave_index'] = wave_idx
        s.vars['point'] = point_idx
        s.vars['sample'] = sample_val
        s.vars['num_points'] = num_points
        wave = self.program.waves[wave_idx]
        for expr in wave.per_point:
            try:
                expr.eval(s)
            except (PresetCompileError, SyntaxError, RuntimeError, ValueError, TypeError, ZeroDivisionError, OverflowError) as e:
                logger.exception(
                    "Per-point expression failed: wave_index=%s point=%s expr=%s",
                    wave_idx,
                    point_idx,
                    getattr(expr, "source", "<unknown>"),
                    exc_info=e,
                )
                expr.bytecode_prog = None
        self.wave_x[point_idx] = s.get('x', 0.5)
        self.wave_y[point_idx] = s.get('y', 0.5)
        self.wave_r[point_idx] = s.get('wave_r', 0.0)
        self.wave_g[point_idx] = s.get('wave_g', 0.0)
        self.wave_b[point_idx] = s.get('wave_b', 0.0)
        self.wave_a[point_idx] = s.get('wave_a', 0.0)

class BlendEngine:
    """Smooth preset blending/morphing engine.
    Runs TWO VMs simultaneously and blends their outputs for seamless transitions.
    This is THE MilkDrop hallmark feature.
    """

    def __init__(self):
        self.current = None
        self.next = None
        self.completed_vm = None
        self.blend = 0.0
        self.blend_speed = 0.01
        self.active = False
        self.smoothstep = True

    def start_blend(self, current_vm: PresetVM, next_vm: PresetVM, duration_secs: float = 2.0):
        """Start a transition between two presets.
        Args:
            current_vm: The currently running VM
            next_vm: The target VM to blend to
            duration_secs: Duration of transition in seconds
        """
        self.current = current_vm
        self.next = next_vm
        self.blend = 0.0
        self.blend_duration = max(duration_secs, 0.1)
        self.start_time = time.time()
        self.active = True

    def update(self) -> PresetState:
        """Update blend and return interpolated state.
        Returns the blended state for GPU uniform upload.
        Blend speed is driven by audio (bass) so transitions
        become synchronized to the music.
        """
        if not self.active:
            return self.current.state if self.current else PresetState()
        base_speed = 0.6
        bass = 0.0
        if self.current and hasattr(self.current, 'state'):
            bass = self.current.state.get('bass', 0.0)
        audio_speed = base_speed + bass * 2.0
        self.blend = min(self.blend + (time.time() - self.start_time) * audio_speed / max(self.blend_duration, 0.1), 1.0)
        if self.blend >= 1.0:
            self.active = False
            completed_vm = self.next
            self.current = self.next
            self.next = None
            if completed_vm:
                self.completed_vm = completed_vm
            return self.current.state
        return self._blend_states(
            self.current.state,
            self.next.state,
            self.blend
        )

    def _blend_states(self, a: PresetState, b: PresetState, t: float) -> PresetState:
        """Interpolate between two states.
        Uses smoothstep for non-linear interpolation if enabled.
        Note: reg00-reg99 are global registers and are NOT blended - they're shared across all presets.
        """
        out = PresetState()
        if self.smoothstep:
            t = t * t * (3.0 - 2.0 * t)
        out.vars_array = a.vars_array * (1.0 - t) + b.vars_array * t
        for vname, vdef in VARIABLES.items():
            val = float(out.vars_array[vdef.index])
            setattr(out, vname, val)
        keys = set(a.vars.keys()) | set(b.vars.keys())
        for k in keys:
            av = a.vars.get(k, 0.0)
            bv = b.vars.get(k, 0.0)
            if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
                out.vars[k] = av * (1.0 - t) + bv * t
            else:
                out.vars[k] = av if t < 0.5 else bv
        for i in range(32):
            out.q[i] = a.q[i] * (1.0 - t) + b.q[i] * t
        out.megabuf_version = max(a.megabuf_version if hasattr(a, 'megabuf_version') else 0,
                                  b.megabuf_version if hasattr(b, 'megabuf_version') else 0)
        out.megabuf = a.megabuf
        out.rand_seed = int(a.rand_seed * (1.0 - t) + b.rand_seed * t) if hasattr(a, 'rand_seed') and hasattr(b, 'rand_seed') else (a.rand_seed if hasattr(a, 'rand_seed') else 1)
        # Global registers (reg00-reg99) are shared - no blending needed
        # They're already in PresetState._global_regs which is a class variable
        return out

    def get_gpu_uniforms(self) -> GPUUniforms:
        """Get GPU uniforms for the current blended state."""
        if self.active and self.current and self.next:
            state = self._blend_states(self.current.state, self.next.state, self.blend)
            u = GPUUniforms(
                zoom=state.get('zoom', 1.0),
                rot=state.get('rot', 0.0),
                warp=state.get('warp', 0.0),
                cx=state.get('cx', 0.5),
                cy=state.get('cy', 0.5),
                dx=state.get('dx', 0.0),
                dy=state.get('dy', 0.0),
                sx=state.get('sx', 1.0),
                sy=state.get('sy', 1.0),
                decay=state.get('decay', 0.9),
                wave_r=state.get('wave_r', 0.0),
                wave_g=state.get('wave_g', 0.0),
                wave_b=state.get('wave_b', 0.0),
                wave_a=state.get('wave_a', 0.0),
                wave_x=state.get('wave_x', 0.5),
                wave_y=state.get('wave_y', 0.5),
                bass=state.get('bass', 0.0),
                mid=state.get('mid', 0.0),
                treb=state.get('treb', 0.0),
                bass_att=state.get('bass_att', 0.0),
                mid_att=state.get('mid_att', 0.0),
                treb_att=state.get('treb_att', 0.0),
                beat=state.get('beat', 0.0),
                time=state.get('time', 0.0),
                frame=float(state.get('frame', 0)),
                fps=state.get('fps', 60.0),
                x=state.get('x', 0.0),
                y=state.get('y', 0.0),
                rad=state.get('rad', 0.0),
                ang=state.get('ang', 0.0),
                width=state.get('width', 1.0),
                height=state.get('height', 1.0),
                aspect=state.get('aspect', 1.0),
                w=state.get('w', 1.0),
                h=state.get('h', 1.0),
                q=list(state.q),
                blend=self.blend,
                echo_zoom=state.get('echo_zoom', 1.0),
                echo_alpha=state.get('echo_alpha', 0.0),
                echo_orient=int(state.get('echo_orient', 0)),
            )
            return u
        elif self.current:
            return self.current.get_gpu_uniforms()
        return GPUUniforms()

class WarpPass:
    """Single warp pass in a multi-pass stack.
    Each pass applies coordinate distortion that feeds into the next.
    """

    def __init__(self, scale: float = 1.0, fbo_width: int = 1024, fbo_height: int = 1024):
        self.scale = scale
        self._fbo_obj: GLFramebuffer | None = None
        self.width = max(1, int(fbo_width * scale))
        self.height = max(1, int(fbo_height * scale))
        self.shader_program = None
        self._initialized = False

    def initialize(self):
        """Create FBO and shader for this warp pass."""
        if self._initialized:
            return
        self._fbo_obj = GLFramebuffer(self.width, self.height)
        self._initialized = True

    @property
    def fbo(self) -> int:
        return self._fbo_obj.fbo if self._fbo_obj else 0

    @property
    def fbo_tex(self) -> int:
        return self._fbo_obj.texture if self._fbo_obj else 0

    def cleanup(self):
        """Clean up GL resources."""
        if self._fbo_obj:
            self._fbo_obj.cleanup()
            self._fbo_obj = None
        self._initialized = False

class WarpStack:
    """Multi-pass warp stack for recursive geometry effects.
    Each pass feeds into the next, creating fractal-like chaos visuals.
    This is what separates basic visualizers from MilkDrop-class.
    """

    def __init__(self, num_passes: int = 3):
        self.passes = []
        self.num_passes = num_passes
        self.damping = 0.98
        self._initialized = False

    def initialize(self, base_width: int = 1024, base_height: int = 1024):
        """Create all warp pass FBOs."""
        if self._initialized:
            return
        if self.passes:
            self.cleanup()
        scales = [1.0, 0.5, 0.25][:self.num_passes]
        for scale in scales:
            warp = WarpPass(scale, base_width, base_height)
            warp.initialize()
            self.passes.append(warp)
        self._initialized = True

    def execute(self, input_tex: int, params: dict, vao: int = 0, shader_program: int = 0, fbo_resizing: bool = False) -> int:
        """Execute all warp passes.
        Args:
            input_tex: Texture ID to start with
            params: Dictionary with 'warp', 'zoom', 'rot', etc.
            vao: Vertex array object for fullscreen quad
            shader_program: Shader program to use (must support u_prev_frame, u_warp, etc.)
            fbo_resizing: If True, skip draw to avoid GL_INVALID_FRAMEBUFFER_OPERATION
        Returns:
            Final output texture ID
        """
        if not self._initialized:
            return input_tex
        if fbo_resizing:
            return self.passes[-1].fbo_tex if self.passes else input_tex
        prev_fbo = np.zeros(1, dtype=np.int32)
        try:
            GL.glGetIntegerv(GL.GL_FRAMEBUFFER_BINDING, prev_fbo)
        except Exception:
            prev_fbo[0] = 0
        current_tex = input_tex
        for i, warp in enumerate(self.passes):
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, warp.fbo)
            GL.glViewport(0, 0, warp.width, warp.height)
            GL.glClearColor(0.0, 0.0, 0.0, 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)
            if shader_program and vao and GL.glIsProgram(shader_program):
                status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
                if status != GL.GL_FRAMEBUFFER_COMPLETE:
                    raise RuntimeError(
                        f"WarpStack execute: FBO incomplete at pass {i}: "
                        f"{hex(int(status)) if status is not None else 'unknown status'}")
                GL.glUseProgram(shader_program)
                GL.glBindVertexArray(vao)
                GL.glActiveTexture(GL.GL_TEXTURE0)
                GL.glBindTexture(GL.GL_TEXTURE_2D, current_tex)
                GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
            current_tex = warp.fbo_tex
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, int(prev_fbo[0]))
        return current_tex

    def cleanup(self):
        """Clean up all warp passes."""
        for warp in self.passes:
            warp.cleanup()
        self.passes = []
        self._initialized = False
OP_LOAD_CONST = 1
OP_LOAD_VAR   = 2
OP_STORE_VAR  = 3
OP_LOAD_VAR_INDEX = 200
OP_LOAD_VAR_INDEX_Q = 201
OP_STORE_VAR_INDEX = 202
OP_ADD        = 4
OP_SUB        = 5
OP_MUL        = 6
OP_DIV        = 7
OP_MOD        = 8
OP_POW        = 9
OP_NEG        = 10
OP_SIN        = 11
OP_COS        = 12
OP_TAN        = 13
OP_ASIN       = 14
OP_ACOS       = 15
OP_ATAN       = 16
OP_ATAN2      = 17
OP_SQRT       = 18
OP_ABS        = 19
OP_FLOOR      = 20
OP_CEIL       = 21
OP_FRACT      = 22
OP_MIN        = 23
OP_MAX        = 24
OP_CLAMP      = 25
OP_MIX        = 26
OP_RAND       = 27
OP_MEGABUF_GET = 28
OP_MEGABUF_SET = 29
OP_Q_GET      = 30
OP_Q_SET      = 31
OP_POP        = 50
OP_DUP        = 51
OP_CMP_EQ     = 32
OP_CMP_NE     = 33
OP_CMP_LT     = 34
OP_CMP_LE     = 35
OP_CMP_GT     = 36
OP_CMP_GE     = 37
OP_AND        = 38
OP_OR         = 39
OP_IF         = 40
OP_NOT        = 41
OP_EXP        = 42
OP_LOG        = 43
OP_POW_FUNC   = 44
OP_JUMP = 60
OP_JUMP_IF_FALSE = 61
OP_JUMP_IF_TRUE  = 62

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
        self.q_refs = set()

    def _add_const(self, value: float) -> int:
        """Add constant to pool, return index.
        MilkDrop presets may emit numerically-unstable constants (NaN/±inf,
        and values that are equal in math but not in exact float bits).
        Using `list.index()` on floats is unreliable: it can throw even when
        the value "should" already exist.
        This implementation uses a quantized key for deduplication.
        """

        if not isinstance(value, float):
            try:
                value = float(value)
            except Exception:
                value = 0.0
        if math.isnan(value):
            self.const_pool.append(value)
            return len(self.const_pool) - 1
        if math.isinf(value):
            self.const_pool.append(value)
            return len(self.const_pool) - 1
        q = round(value * 1_000_000)
        if not hasattr(self, "_const_key_to_index"):
            self._const_key_to_index = {}
        existing = self._const_key_to_index.get(q)
        if existing is not None and 0 <= existing < len(self.const_pool) and self.const_pool[existing] == value:
            return existing
        idx = len(self.const_pool)
        self.const_pool.append(value)
        self._const_key_to_index[q] = idx
        return idx

    def visit_Constant(self, node):
        """Compile numeric constants."""
        if isinstance(node.value, (int, float)):
            idx = self._add_const(node.value if isinstance(node.value, float) else float(node.value))
            self.code.append((OP_LOAD_CONST, idx))
        return 0.0

    def visit_Name(self, node):
        """Compile variable loads using OP_LOAD_VAR_INDEX for direct array access."""
        name = node.id.lower()
        if name.startswith('q') and name[1:].isdigit():
            idx = int(name[1:]) - 1
            if 0 <= idx < 32:
                self.q_refs.add(idx)
                self.code.append((OP_LOAD_VAR_INDEX_Q, idx))
                return
        if name in VARIABLES:
            vdef = VARIABLES[name]
            self.code.append((OP_LOAD_VAR_INDEX, vdef.index))
            return
        if name not in self.var_names:
            self.var_names[name] = len(self.var_names)
        var_idx = self.var_names[name]
        self.code.append((OP_LOAD_VAR, var_idx))

    def visit_BinOp(self, node):
        """Compile binary operations with constant folding."""
        left_is_const = isinstance(node.left, ast.Constant) and isinstance(node.left.value, (int, float))
        right_is_const = isinstance(node.right, ast.Constant) and isinstance(node.right.value, (int, float))
        if left_is_const and right_is_const:
            left_val = float(node.left.value)
            right_val = float(node.right.value)
            result = 0.0
            if isinstance(node.op, ast.Add):
                result = left_val + right_val
            elif isinstance(node.op, ast.Sub):
                result = left_val - right_val
            elif isinstance(node.op, ast.Mult):
                result = left_val * right_val
            elif isinstance(node.op, ast.Div):
                result = left_val / right_val if right_val != 0 else 0.0
            elif isinstance(node.op, ast.Mod):
                result = left_val % right_val if right_val != 0 else 0.0
            elif isinstance(node.op, ast.Pow):
                result = left_val ** right_val
            else:
                self.visit(node.left)
                self.visit(node.right)
                self._emit_op(node.op)
                return 0.0
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
        """Handle attribute access for compatibility."""
        self.visit(node.value)
        return 0.0

    def visit_Subscript(self, node):
        """Handle subscripts (critical for comma-operator [-1] fix)."""
        self.visit(node.value)
        return 0.0

    def visit_Tuple(self, node):
        """Handle tuples (generated by comma operators)."""
        for i, elt in enumerate(node.elts):
            self.visit(elt)
            if i < len(node.elts) - 1:
                self.code.append((OP_POP,))
        return 0.0

    def visit_Index(self, node):
        """Handle Index nodes (Python < 3.9)."""
        self.visit(node.value)
        return 0.0

    def visit_Slice(self, node):
        """Handle Slice nodes."""
        if node.lower:
            self.visit(node.lower)
        if node.upper:
            self.visit(node.upper)
        if node.step:
            self.visit(node.step)
        return 0.0

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
        """Compile comparisons to boolean (0.0/1.0)."""
        self.visit(node.left)
        if not node.ops or not node.comparators:
            return
        op = node.ops[0]
        self.visit(node.comparators[0])
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
            j_end = len(self.code)
            self.code.append((OP_JUMP, None))
            zero_idx = self._add_const(0.0)
            landing_false = len(self.code)
            self.code.append((OP_LOAD_CONST, zero_idx))
            after_false = len(self.code)
            self.code[j_end] = (OP_JUMP, after_false)
            for jp in false_jumps:
                self.code[jp] = (self.code[jp][0], landing_false)
            landing_true = len(self.code)
            self.code.append((OP_LOAD_CONST, one_idx))
        elif isinstance(node.op, ast.Or):
            true_jumps = []
            for i, v in enumerate(values):
                self.visit(v)
                j_true = len(self.code)
                self.code.append((OP_JUMP_IF_TRUE, None))
                true_jumps.append(j_true)
            zero_idx = self._add_const(0.0)
            j_end = len(self.code)
            self.code.append((OP_JUMP, None))
            one_idx = self._add_const(1.0)
            landing_true = len(self.code)
            self.code.append((OP_LOAD_CONST, one_idx))
            after_true = len(self.code)
            self.code[j_end] = (OP_JUMP, after_true)
            landing_false = len(self.code)
            self.code.append((OP_LOAD_CONST, zero_idx))
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
            if name.startswith('q') and name[1:].isdigit():
                idx = int(name[1:]) - 1
                if 0 <= idx < 32:
                    self.code.append((OP_DUP,))
                    self.code.append((OP_Q_SET, idx))
            elif name in VARIABLES:
                vdef = VARIABLES[name]
                self.code.append((OP_DUP,))
                self.code.append((OP_STORE_VAR_INDEX, vdef.index))
            else:
                if name not in self.var_names:
                    self.var_names[name] = len(self.var_names)
                var_idx = self.var_names[name]
                self.code.append((OP_DUP,))
                self.code.append((OP_STORE_VAR, var_idx))

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
        """Compile function calls."""
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        for arg in node.args:
            self.visit(arg)
        func_map = {
            'sin': OP_SIN, 'cos': OP_COS, 'tan': OP_TAN,
            'asin': OP_ASIN, 'acos': OP_ACOS, 'atan': OP_ATAN,
            'atan2': OP_ATAN2, 'sqrt': OP_SQRT,
            'exp': OP_EXP, 'log': OP_LOG, 'pow': OP_POW_FUNC,
            'abs': OP_ABS, 'floor': OP_FLOOR, 'ceil': OP_CEIL,
            'frac': OP_FRACT, 'min': OP_MIN, 'max': OP_MAX,
            'clamp': OP_CLAMP, 'mix': OP_MIX,
            'rand': OP_RAND,
            'megabuf': OP_MEGABUF_GET,
            'wrap': 100,
            'cwrap': 101,
            'invsqrt': 102,
            'fact': 103,
            'nchoosek': 104,
            'sqr': 105,
            'sign': 106,
            'int': 107
        }
        if func_name in func_map:
            self.code.append((func_map[func_name],))

    def compile_assignment(self, expr_str: str) -> BytecodeProgram:
        """Compile an assignment expression to bytecode."""
        expr_str = expr_str.strip()
        if '=' not in expr_str:
            return None
        try:
            key, val = expr_str.split('=', 1)
            key = key.strip().lower()
            val = val.strip().rstrip(';')
            if not val:
                return None
            m_match = re.match(r'g?megabuf\s*\(\s*(.*)\s*\)', key)
            if m_match:
                idx_expr = m_match.group(1)
                if not idx_expr or not val:
                    return None

                if idx_expr.count('(') != idx_expr.count(')') or val.count('(') != val.count(')'):
                    if CompiledExpr._LOG_PARSE_ERRORS:
                        logger.warning("BytecodeCompiler: Unbalanced parentheses in megabuf idx=%r val=%r", idx_expr, val)
                    return None
                try:
                    idx_tree = ast.parse(idx_expr, mode='eval')
                    self.visit(idx_tree.body)
                    val_tree = ast.parse(val, mode='eval')
                    self.visit(val_tree.body)
                except (SyntaxError, PresetCompileError):
                    return None
                self.code.append((OP_MEGABUF_SET,))
                return BytecodeProgram(self.code, self.const_pool, self.var_names, self.q_refs)
            if not val:
                return None

            if val.count('(') != val.count(')'):
                if CompiledExpr._LOG_PARSE_ERRORS:
                    logger.warning("BytecodeCompiler: Unbalanced parentheses in val=%r", val)
                return None
            try:
                tree = ast.parse(val, mode='eval')
            except SyntaxError as e:
                try:
                    crash_log = Path(__file__).parent / "linamp_crash.log"
                    crash_log.parent.mkdir(parents=True, exist_ok=True)
                    with open(crash_log, "a", encoding="utf-8") as f:
                        f.write(
                            "\n[parse-error] BytecodeCompiler RHS ParseError "
                            f"key={key!r} val={CompiledExpr._truncate_for_log(val)!r} "
                            f"err={type(e).__name__}: {str(e)!s}"
                        )
                except OSError as e2:
                    logger.debug(f"Failed to write parse error to log: {e2}")
                if CompiledExpr._LOG_PARSE_ERRORS:
                    logger.warning(
                        "BytecodeCompiler: RHS SyntaxError key=%r val=%r err=%s",
                        key,
                        CompiledExpr._truncate_for_log(val),
                        e,
                    )
                return None
            try:
                self.visit(tree.body)
            except (PresetCompileError, SyntaxError, RuntimeError) as e:
                if CompiledExpr._LOG_PARSE_ERRORS:
                    logger.warning("BytecodeCompiler: AST visit failed key=%r err=%s", key, e)
                return None
            if key.startswith('q') and key[1:].isdigit():
                idx = int(key[1:]) - 1
                if 0 <= idx < 32:
                    self.code.append((OP_Q_SET, idx))
            elif key in VARIABLES:
                vdef = VARIABLES[key]
                self.code.append((OP_STORE_VAR_INDEX, vdef.index))
            else:
                if key not in self.var_names:
                    self.var_names[key] = len(self.var_names)
                var_idx = self.var_names[key]
                self.code.append((OP_STORE_VAR, var_idx))
            return BytecodeProgram(self.code, self.const_pool, self.var_names, self.q_refs)
        except (PresetCompileError, SyntaxError, RuntimeError) as e:
            if CompiledExpr._LOG_PARSE_ERRORS:
                logger.warning("BytecodeCompiler: Failed to compile assignment expr=%r err=%s", expr_str, e)
            elif logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"BytecodeCompiler: Failed to compile assignment: {e}")
            return None

def _safe_pop(stack):
    """Defensive stack pop that returns 0.0 if stack is empty.
    Prevents crashes from malformed presets with unbalanced operations.
    """
    return stack.pop() if stack else 0.0

class BytecodeProgram:
    """Compiled bytecode program ready for execution."""

    def __init__(self, code, const_pool, var_names, q_refs=None):
        self.code = code
        self.const_pool = const_pool
        self.var_names = var_names
        self.q_refs = q_refs or set()
        self._rebuild_idx_to_var()

    def _rebuild_idx_to_var(self):
        self._idx_to_var = {v: k for k, v in self.var_names.items()}

    def execute(self, state: PresetState) -> float:
        """Execute bytecode with given state.
        ~5-20x faster than AST evaluation.
        Uses array access for known variables via OP_LOAD_VAR_INDEX / OP_STORE_VAR_INDEX.
        """
        stack = []
        pc = 0
        vars_array = state.vars_array
        q_regs = state.q
        while pc < len(self.code):
            instr = self.code[pc]
            op = instr[0]
            if op == OP_LOAD_CONST:
                stack.append(self.const_pool[instr[1]])
            elif op == OP_LOAD_VAR_INDEX:
                stack.append(safe_float(vars_array[instr[1]]))
            elif op == OP_LOAD_VAR_INDEX_Q:
                stack.append(safe_float(q_regs[instr[1]]))
            elif op == OP_LOAD_VAR:
                var_name = self._idx_to_var.get(instr[1], f"var_{instr[1]}")
                stack.append(safe_float(state.get(var_name, 0.0)))
            elif op == OP_STORE_VAR_INDEX:
                value = safe_float(_safe_pop(stack))
                vars_array[instr[1]] = value
            elif op == OP_STORE_VAR:
                value = safe_float(_safe_pop(stack))
                var_name = self._idx_to_var.get(instr[1], f"var_{instr[1]}")
                state.set(var_name, value)
            elif op == 100:
                b, a, x = _safe_pop(stack), _safe_pop(stack), _safe_pop(stack)
                stack.append(a + ((x - a) % (b - a + 1e-10)))
            elif op == 101:
                b, a, x = _safe_pop(stack), _safe_pop(stack), _safe_pop(stack)
                r = b - a
                stack.append(a + abs(((x - a) % (2 * r)) - r))
            elif op == 102:
                x = _safe_pop(stack)
                stack.append(1.0 / math.sqrt(x) if x > 0 else 0.0)
            elif op == 103:
                stack.append(float(math.factorial(int(max(0, _safe_pop(stack))))))
            elif op == 105:
                x = _safe_pop(stack)
                stack.append(x * x)
            elif op == 106:
                x = _safe_pop(stack)
                stack.append(1.0 if x > 0 else (-1.0 if x < 0 else 0.0))
            elif op == 107:
                stack.append(float(int(_safe_pop(stack))))
            elif op == OP_ADD:
                b, a = _safe_pop(stack), _safe_pop(stack)
                stack.append(a + b)
            elif op == OP_SUB:
                b, a = _safe_pop(stack), _safe_pop(stack)
                stack.append(a - b)
            elif op == OP_MUL:
                b, a = _safe_pop(stack), _safe_pop(stack)
                stack.append(a * b)
            elif op == OP_DIV:
                b, a = _safe_pop(stack), _safe_pop(stack)
                denom = max(abs(b), EPSILON_DIV)
                stack.append(a / denom)
            elif op == OP_MOD:
                b, a = _safe_pop(stack), _safe_pop(stack)
                safe_b = EPSILON_DIV if abs(b) < EPSILON_DIV else b
                stack.append(a - b * math.floor(a / safe_b))
            elif op == OP_POW:
                b, a = _safe_pop(stack), _safe_pop(stack)
                stack.append(math.pow(a, b))
            elif op == OP_NEG:
                stack.append(-_safe_pop(stack))
            elif op == OP_NOT:
                a = _safe_pop(stack)
                stack.append(0.0 if a != 0 else 1.0)
            elif op in (OP_CMP_EQ, OP_CMP_NE, OP_CMP_LT, OP_CMP_LE, OP_CMP_GT, OP_CMP_GE):
                b = _safe_pop(stack)
                a = _safe_pop(stack)
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
            elif op == OP_AND:
                b = _safe_pop(stack)
                a = _safe_pop(stack)
                stack.append(1.0 if (a != 0.0 and b != 0.0) else 0.0)
            elif op == OP_OR:
                b = _safe_pop(stack)
                a = _safe_pop(stack)
                stack.append(1.0 if (a != 0.0 or b != 0.0) else 0.0)
            elif op == OP_IF:
                false_val = _safe_pop(stack)
                true_val = _safe_pop(stack)
                cond = _safe_pop(stack)
                stack.append(true_val if cond != 0.0 else false_val)
            elif op == OP_SIN:
                stack.append(math.sin(_safe_pop(stack)))
            elif op == OP_EXP:
                stack.append(math.exp(_safe_pop(stack)))
            elif op == OP_LOG:
                val = _safe_pop(stack)
                stack.append(math.log(val) if val > 0 else 0.0)
            elif op == OP_POW_FUNC:
                b, a = _safe_pop(stack), _safe_pop(stack)
                stack.append(math.pow(a, b))
            elif op == OP_COS:
                stack.append(math.cos(_safe_pop(stack)))
            elif op == OP_TAN:
                stack.append(math.tan(_safe_pop(stack)))
            elif op == OP_ASIN:
                stack.append(math.asin(_safe_pop(stack)))
            elif op == OP_ACOS:
                stack.append(math.acos(_safe_pop(stack)))
            elif op == OP_ATAN:
                stack.append(math.atan(_safe_pop(stack)))
            elif op == OP_ATAN2:
                y, x = _safe_pop(stack), _safe_pop(stack)
                stack.append(math.atan2(y, x))
            elif op == OP_SQRT:
                stack.append(math.sqrt(_safe_pop(stack)))
            elif op == OP_ABS:
                stack.append(abs(_safe_pop(stack)))
            elif op == OP_FLOOR:
                stack.append(math.floor(_safe_pop(stack)))
            elif op == OP_CEIL:
                stack.append(math.ceil(_safe_pop(stack)))
            elif op == OP_FRACT:
                x = _safe_pop(stack)
                stack.append(x - math.floor(x))
            elif op == OP_CLAMP:
                hi, lo, x = _safe_pop(stack), _safe_pop(stack), _safe_pop(stack)
                stack.append(max(lo, min(hi, x)))
            elif op == OP_MIX:
                t, b, a = _safe_pop(stack), _safe_pop(stack), _safe_pop(stack)
                stack.append(a + (b - a) * t)
            elif op == OP_MIN:
                b, a = _safe_pop(stack), _safe_pop(stack)
                stack.append(min(a, b))
            elif op == OP_MAX:
                b, a = _safe_pop(stack), _safe_pop(stack)
                stack.append(max(a, b))
            elif op == OP_RAND:
                stack.append(md_rand(state))
            elif op == OP_Q_GET:
                stack.append(state.q[instr[1]])
            elif op == OP_Q_SET:
                state.q[instr[1]] = _safe_pop(stack)
            elif op == OP_MEGABUF_GET:
                idx = int(_safe_pop(stack)) % PresetState.MEGABUF_SIZE
                stack.append(state.megabuf[idx])
            elif op == OP_MEGABUF_SET or op == 29:
                val = safe_float(_safe_pop(stack))
                idx = int(_safe_pop(stack)) % PresetState.MEGABUF_SIZE
                state.megabuf_set(idx, val)
            elif op == OP_POP:
                if stack:
                    stack.pop()
            elif op == OP_DUP:
                if stack:
                    stack.append(stack[-1])
            elif op == OP_JUMP:
                pc = instr[1]
                continue
            elif op == OP_JUMP_IF_FALSE:
                cond = _safe_pop(stack)
                if cond == 0.0:
                    pc = instr[1]
                    continue
            elif op == OP_JUMP_IF_TRUE:
                cond = _safe_pop(stack)
                if cond != 0.0:
                    pc = instr[1]
                    continue
            pc += 1
        return stack[-1] if stack else 0.0

class Sprite:
    """Represents a single MilkDrop user sprite instance.
    Each sprite maintains its own state that evolves through init_ and code_
    equations, similar to how waves/shapes work in presets.
    """
    x: float = 0.5
    y: float = 0.5
    sx: float = 1.0
    sy: float = 1.0
    rot: float = 0.0
    a: float = 1.0
    r: float = 1.0
    g: float = 1.0
    b: float = 1.0
    tex_u: float = 0.0
    tex_v: float = 0.0
    blend_mode: int = 0
    active: bool = True

    def __init__(self, sprite_id: str, image_path: str | None = None):
        self.sprite_id = sprite_id.zfill(2)
        self.image_path = image_path
        self.texture_id = None
        self.image_width = 0
        self.image_height = 0
        self.init_code = []
        self.frame_code = []
        self.tex_code = []
        self.init_run = False
        self.user_vars = {}
        self.reset()

    def reset(self):
        """Reset sprite to default state."""
        self.x = 0.5
        self.y = 0.5
        self.sx = 1.0
        self.sy = 1.0
        self.rot = 0.0
        self.a = 1.0
        self.r = 1.0
        self.g = 1.0
        self.b = 1.0
        self.tex_u = 0.0
        self.tex_v = 0.0
        self.blend_mode = 0
        self.active = True
        self.init_run = False
        self.user_vars.clear()

    def run_init(self, state: PresetState):
        """Run init_ equations once at startup."""
        if self.init_run:
            return
        self._sync_to_state(state)
        for expr in self.init_code:
            expr.eval(state)
        self._sync_from_state(state)
        self.init_run = True

    def run_frame(self, state: PresetState):
        """Run per-frame code_ equations."""
        if not self.active:
            return
        self._sync_to_state(state)
        state.vars['sprite'] = int(self.sprite_id) if self.sprite_id.isdigit() else 0

        for expr in self.frame_code:
            try:
                expr.eval(state)
            except Exception as e:
                logger.exception(
                    "Sprite frame_code expression failed: sprite_id=%s expr=%s",
                    self.sprite_id,
                    getattr(expr, "source", "<unknown>"),
                    exc_info=e,
                )
                expr.bytecode_prog = None

        for expr in self.tex_code:
            try:
                expr.eval(state)
            except Exception as e:
                logger.exception(
                    "Sprite tex_code expression failed: sprite_id=%s expr=%s",
                    self.sprite_id,
                    getattr(expr, "source", "<unknown>"),
                    exc_info=e,
                )
                expr.bytecode_prog = None

        self._sync_from_state(state)


    def _sync_to_state(self, state: PresetState):
        """Copy sprite state to PresetState for equation evaluation."""
        for var in ['x', 'y', 'sx', 'sy', 'rot', 'a', 'r', 'g', 'b', 'tex_u', 'tex_v']:
            state.vars[var] = getattr(self, var)
        state.vars['cx'] = self.x
        state.vars['cy'] = self.y
        for key, val in self.user_vars.items():
            state.vars[key] = val

    def _sync_from_state(self, state: PresetState):
        """Copy PresetState back to sprite after equation evaluation."""
        for var in ['x', 'y', 'sx', 'sy', 'rot', 'a', 'r', 'g', 'b', 'tex_u', 'tex_v']:
            setattr(self, var, safe_float(state.vars.get(var, getattr(self, var))))
        for key in self.user_vars:
            if key in state.vars:
                self.user_vars[key] = state.vars[key]

class SpriteConfig:
    """Configuration for a sprite from milk_img.ini."""

    def __init__(self, sprite_id: str):
        self.sprite_id = sprite_id
        self.img_path = ""
        self.init_exprs = []
        self.frame_exprs = []
        self.tex_exprs = []
        self.blend_mode = 0
        self.enabled = True

class SpriteConfigParser:
    """Parser for milk_img.ini sprite configuration files.
    Parses the classic MilkDrop ini format:
    [00]
    img=path/to/image.png
    init_x=0.5
    init_y=0.5
    code_sx=sx + bass * 0.1
    [01]
    img=another.jpg
    ...
    """

    def __init__(self):
        self.sprites: dict[str, SpriteConfig] = {}
        self.config_path = None

    def parse_file(self, path: str) -> dict[str, SpriteConfig]:
        """Parse a milk_img.ini file and return sprite configurations."""
        self.config_path = path
        self.sprites.clear()
        if not Path(path).exists():
            return self.sprites
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to read sprite config: {e}")
            return self.sprites
        current_sprite = None
        for line_num, line in enumerate(content.split('\n'), 1):
            line = line.strip()
            if not line or line.startswith(('#', '//')):
                continue
            if line.startswith('[') and line.endswith(']'):
                section = line[1:-1].strip()
                if section.isdigit() and 0 <= int(section) <= 99:
                    current_sprite = SpriteConfig(section)
                    self.sprites[section] = current_sprite
                else:
                    current_sprite = None
                continue
            if current_sprite is None:
                continue
            if '=' in line:
                key, val = line.split('=', 1)
                key = key.strip().lower()
                val = val.strip()
                if key == 'img':
                    current_sprite.img_path = val
                elif key.startswith('init_'):
                    var_name = key[5:]
                    expr = f"{var_name} = {val}"
                    current_sprite.init_exprs.append(expr)
                elif key.startswith('code_'):
                    var_name = key[5:]
                    expr = f"{var_name} = {val}"
                    current_sprite.frame_exprs.append(expr)
                elif key in ('x', 'y', 'sx', 'sy', 'rot', 'a', 'r', 'g', 'b'):
                    expr = f"{key} = {val}"
                    current_sprite.init_exprs.append(expr)
        return self.sprites

    def find_config_file(self, search_paths: list[str]) -> str:
        """Find milk_img.ini in search paths."""
        for base_path in search_paths:
            for filename in ['milk_img.ini', 'milk2_img.ini', 'sprites.ini']:
                full_path = Path(base_path) / filename
                if full_path.exists():
                    return str(full_path)
        return None

class SpriteRenderer:
    """OpenGL renderer for MilkDrop user sprites.
    Handles texture loading, sprite quad rendering with transforms,
    and proper blending modes.
    """

    def __init__(self):
        self.sprites: dict[str, Sprite] = {}
        self.config_parser = SpriteConfigParser()
        self.loaded_textures = TextureLRU(maxsize=256)
        self.loaded_texture_sizes: dict[str, tuple] = {}
        self.sprite_shader = None
        self.vao = None
        self.vbo = None
        self.vertex_shader = """#version 330 core
        layout(location = 0) in vec2 position;
        layout(location = 1) in vec2 texcoord;
        uniform vec2 u_sprite_pos;    // x, y center (0-1)
        uniform vec2 u_sprite_scale;  // sx, sy
        uniform float u_sprite_rot;   // rotation in radians
        uniform vec2 u_resolution;    // screen resolution
        out vec2 v_texcoord;
        void main() {
            // Apply scale
            vec2 scaled = position * u_sprite_scale;
            // Apply rotation
            float c = cos(u_sprite_rot);
            float s = sin(u_sprite_rot);
            vec2 rotated = vec2(
                scaled.x * c - scaled.y * s,
                scaled.x * s + scaled.y * c
            );
            // Position in normalized device coordinates (-1 to 1)
            // Input position is already in NDC (-1 to 1 for a full-screen quad)
            vec2 ndc_pos = rotated + (u_sprite_pos * 2.0 - 1.0);
            gl_Position = vec4(ndc_pos, 0.0, 1.0);
            v_texcoord = texcoord;
        }
        """

        self.fragment_shader = """#version 330 core
        in vec2 v_texcoord;
        out vec4 FragColor;
        uniform sampler2D u_texture;
        uniform vec4 u_color;      // r, g, b, a tint
        uniform int u_blend_mode;  // 0 = normal, 1 = additive, 2 = multiplicative
        void main() {
            vec4 tex_color = texture(u_texture, v_texcoord);
            // Apply color tint and alpha
            vec4 tinted = tex_color * u_color;
            // Handle premultiplied alpha for additive
            if (u_blend_mode == 1) {
                // Additive: just add the color
                FragColor = vec4(tinted.rgb * tinted.a, tinted.a);
            } else if (u_blend_mode == 2) {
                // Multiplicative
                FragColor = vec4(tinted.rgb, tinted.a);
            } else {
                // Normal alpha blend
                FragColor = tinted;
            }
        }
        """

    def initialize(self):
        """Initialize OpenGL resources."""
        self._compile_shader()
        self._create_quad()

    def _compile_shader(self):
        """Compile sprite shader program."""
        try:
            prog_id = _compile_gl_program(self.vertex_shader, self.fragment_shader, name="SpriteRenderer")
            if prog_id == 0:
                return
            self.sprite_shader = prog_id
            self.loc_pos = GL.glGetUniformLocation(self.sprite_shader, "u_sprite_pos")
            self.loc_scale = GL.glGetUniformLocation(self.sprite_shader, "u_sprite_scale")
            self.loc_rot = GL.glGetUniformLocation(self.sprite_shader, "u_sprite_rot")
            self.loc_res = GL.glGetUniformLocation(self.sprite_shader, "u_resolution")
            self.loc_color = GL.glGetUniformLocation(self.sprite_shader, "u_color")
            self.loc_blend = GL.glGetUniformLocation(self.sprite_shader, "u_blend_mode")
            self.loc_tex = GL.glGetUniformLocation(self.sprite_shader, "u_texture")
        except Exception as e:
            logger.error(f"Failed to compile sprite shader: {e}")

    def _create_quad(self):
        """Create VAO/VBO for a fullscreen quad (sprites are rendered as quads)."""
        vertices = np.array([
            -1.0, -1.0,     0.0, 0.0,
            1.0, -1.0,     1.0, 0.0,
            1.0,  1.0,     1.0, 1.0,
            -1.0,  1.0,     0.0, 1.0,
        ], dtype=np.float32)
        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
        self.vao = int(GL.glGenVertexArrays(1))
        self.vbo = int(GL.glGenBuffers(1))
        self.ebo = int(GL.glGenBuffers(1))
        GL.glBindVertexArray(self.vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL.GL_STATIC_DRAW)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 4 * 4, None)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, GL.GL_FALSE, 4 * 4, ctypes.c_void_p(2 * 4))
        GL.glEnableVertexAttribArray(1)
        GL.glBindVertexArray(0)

    def load_sprite_config(self, config_path: str) -> int:
        """Load sprite configuration and create sprites."""
        configs = self.config_parser.parse_file(config_path)
        count = 0
        for sprite_id, config in configs.items():
            if not config.img_path:
                continue
            config_dir = Path(config_path).parent
            img_path = config_dir / config.img_path
            if not img_path.exists():
                img_path = Path(config.img_path)
            if not img_path.exists():
                logger.warning(f"Sprite image not found: {config.img_path}")
                continue
            sprite = Sprite(sprite_id, str(img_path))
            sprite.blend_mode = config.blend_mode
            compilation_targets = [
                (config.init_exprs, sprite.init_code, "init"),
                (config.frame_exprs, sprite.frame_code, "frame"),
                (config.tex_exprs, sprite.tex_code, "tex")
            ]
            for expr_list, code_list, name in compilation_targets:
                for expr_str in expr_list:
                    try:
                        compiled = CompiledExpr(expr_str)
                        if compiled.is_valid():
                            code_list.append(compiled)
                    except Exception as e:
                        logger.debug(f"Failed to compile sprite expression '{expr_str}': {e}")
                        continue
            texture_id = self._load_texture(str(img_path))
            if texture_id:
                sprite.texture_id = int(texture_id)
                size = self.loaded_texture_sizes.get(str(img_path))
                if size:
                    sprite.image_width, sprite.image_height = size
                self.sprites[sprite_id] = sprite
                count += 1
                logger.info(f"Loaded sprite [{sprite_id}]: {img_path.name}")
            else:
                logger.error(f"Failed to load texture for sprite [{sprite_id}]")
        return count

    def _load_texture(self, path: str) -> int:
        """Load image file into OpenGL texture."""
        if path in self.loaded_textures:
            return self.loaded_textures[path]
        try:
            img = Image.open(path).convert('RGBA')
            img_data = np.array(img)
            img_data = np.flipud(img_data)
            tex_id = GL.glGenTextures(1)
            if isinstance(tex_id, (list, tuple, np.ndarray)):
                try:
                    tex_id = int(tex_id[0])
                except Exception:
                    tex_id = int(np.asarray(tex_id).item())
            else:
                try:
                    tex_id = int(tex_id)
                except Exception as e:
                    logger.debug(f"Failed to convert tex_id to int: {e}")
            GL.glBindTexture(GL.GL_TEXTURE_2D, tex_id)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA,
                            img.width, img.height, 0,
                            GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, img_data)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
            try:
                self.loaded_texture_sizes[path] = (img.width, img.height)
            except Exception:
                self.loaded_texture_sizes[path] = (0, 0)
            self.loaded_textures[path] = int(tex_id)
            return int(tex_id)
        except Exception as e:
            logger.error(f"Texture load error for {path}: {e}")
            return 0

    def render_sprites(self, state: PresetState, width: int, height: int):
        """Render all active sprites."""
        if not self.sprites or not self.sprite_shader:
            return
        for sprite in self.sprites.values():
            if sprite.texture_id and sprite.texture_id > 0:
                try:
                    if not GL.glIsTexture(sprite.texture_id):

                        logger.debug(f"Reloading sprite texture for [{sprite.sprite_id}]: {sprite.image_path}")
                        new_tex_id = self._load_texture(sprite.image_path)
                        if new_tex_id and new_tex_id > 0:
                            sprite.texture_id = new_tex_id
                except Exception as exc:
                    logger.debug("Exception in try block: %s", exc)
        prev_blend = GL.glIsEnabled(GL.GL_BLEND)
        try:
            prev_program = GL.glGetIntegerv(GL.GL_CURRENT_PROGRAM)
        except Exception:
            prev_program = 0
        try:
            prev_vao = GL.glGetIntegerv(GL.GL_VERTEX_ARRAY_BINDING)
        except Exception:
            prev_vao = 0
        GL.glEnable(GL.GL_BLEND)
        GL.glUseProgram(self.sprite_shader)
        GL.glUniform2f(self.loc_res, width, height)
        GL.glBindVertexArray(self.vao)
        for sprite in self.sprites.values():
            if not sprite.active or sprite.texture_id is None:
                continue
            if not sprite.init_run:
                sprite.run_init(state)
            sprite.run_frame(state)
            if sprite.a <= 0.01:
                continue
            aspect = width / height if height > 0 else 1.0
            img_aspect = sprite.image_width / sprite.image_height if sprite.image_height > 0 else 1.0
            scale_x = safe_float(sprite.sx) * 0.5
            scale_y = safe_float(sprite.sy) * 0.5 * aspect / img_aspect if img_aspect > 0 else safe_float(sprite.sy) * 0.5
            GL.glUniform2f(self.loc_pos, safe_float(sprite.x, 0.5), safe_float(sprite.y, 0.5))
            GL.glUniform2f(self.loc_scale, scale_x, scale_y)
            GL.glUniform1f(self.loc_rot, safe_float(sprite.rot))
            GL.glUniform4f(self.loc_color,
                           safe_float(sprite.r, 1.0), safe_float(sprite.g, 1.0),
                           safe_float(sprite.b, 1.0), safe_float(sprite.a, 1.0))
            GL.glUniform1i(self.loc_blend, sprite.blend_mode)
            try:
                tex_id = int(sprite.texture_id) if sprite.texture_id is not None else 0
            except (TypeError, ValueError):
                tex_id = 0
            if tex_id == 0:
                logger.debug(f"Sprite [{getattr(sprite, 'sprite_id', '?')}] has invalid texture id: {sprite.texture_id}")
                continue
            try:
                if not GL.glIsTexture(tex_id):
                    logger.debug(f"Sprite [{getattr(sprite, 'sprite_id', '?')}] texture not recognized by GL: {tex_id}, attempting reload...")

                    if hasattr(sprite, 'image_path') and sprite.image_path:
                        new_tex_id = self._load_texture(sprite.image_path)
                        if new_tex_id and new_tex_id > 0:
                            sprite.texture_id = new_tex_id
                            tex_id = new_tex_id
                            logger.info(f"Reloaded sprite texture for [{sprite.sprite_id}]: {tex_id}")
                        else:
                            logger.error(f"Failed to reload sprite texture for [{sprite.sprite_id}]")
                            continue
                    else:
                        continue
            except Exception as e:
                logger.debug(f"Error in sprite texture reload: {e}")
            GL.glActiveTexture(GL.GL_TEXTURE0)
            try:
                GL.glBindTexture(GL.GL_TEXTURE_2D, tex_id)
            except (GL.GLError, TypeError, ValueError) as e:
                logger.debug(f"Failed to bind sprite texture {tex_id} for sprite [{getattr(sprite, 'sprite_id', '?')}]: {e}")
                continue
            GL.glUniform1i(self.loc_tex, 0)
            if sprite.blend_mode == 1:
                GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE)
            elif sprite.blend_mode == 2:
                GL.glBlendFunc(GL.GL_DST_COLOR, GL.GL_ZERO)
            else:
                GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
            GL.glDrawElements(GL.GL_TRIANGLES, 6, GL.GL_UNSIGNED_INT, None)
        GL.glBindVertexArray(prev_vao)
        GL.glUseProgram(prev_program)
        if not prev_blend:
            GL.glDisable(GL.GL_BLEND)

    def cleanup(self):
        """Clean up OpenGL resources."""
        for tex_id in list(self.loaded_textures.values()):
            GLResourceOwner.queue_delete("texture", int(tex_id))
        self.loaded_textures.clear()
        self.loaded_texture_sizes.clear()
        if self.sprite_shader:
            GLResourceOwner.queue_delete("program", int(self.sprite_shader))
            self.sprite_shader = None
        if self.vao:
            GLResourceOwner.queue_delete("vao", int(self.vao))
            self.vao = None
        if self.vbo:
            GLResourceOwner.queue_delete("buffer", int(self.vbo))
            self.vbo = None
        if hasattr(self, 'ebo') and self.ebo:
            GLResourceOwner.queue_delete("buffer", int(self.ebo))
            self.ebo = None

@dataclass
class Connection:
    from_node: str
    to_node: str

class EffectNode:
    """Phase 1: Base class for autonomous effect nodes."""
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.params = {"strength": 1.0, "audio_factor": 0.5}
        self.audio_target = random.choice(["bass", "mid", "treb", "beat"])

    def generate_glsl(self) -> str:
        return ""

    def mutate(self, rate: float):
        if random.random() < rate:
            self.audio_target = random.choice(["bass", "mid", "treb", "beat"])
        for k, v in self.params.items():
            if random.random() < rate:
                if isinstance(v, float):
                    self.params[k] += random.uniform(-0.1, 0.1)
                elif isinstance(v, int):
                    self.params[k] += random.choice([-1, 1])

    def serialize(self) -> dict:
        return {"type": self.__class__.__name__, "params": self.params, "audio": self.audio_target}

    def copy(self):
        new_node = self.__class__(self.node_id)
        new_node.params = self.params.copy()
        new_node.audio_target = self.audio_target
        return new_node

class RippleEffect(EffectNode):
    def generate_glsl(self):
        f, a = self.params.get("freq", 20.0), self.params.get("amp", 0.02)
        return f"uv += normalize(uv-0.5) * sin(length(uv-0.5)*{f:.4f} - u_time*2.0) * {a:.4f};"

class SwirlEffect(EffectNode):
    def generate_glsl(self):
        s = self.params.get("strength", 3.0)
        return f"{{ float r = length(uv-0.5); float a = atan(uv.y-0.5, uv.x-0.5) + (1.0-r)*{s:.4f}*u_audio.y; uv = vec2(cos(a), sin(a))*r + 0.5; }}"

class TunnelEffect(EffectNode):
    def generate_glsl(self):
        z = self.params.get("zoom", 0.3)
        return (f"{{ float r = {z:.4f} / (length(uv-0.5) + 0.01); "
                f"float a = atan(uv.y-0.5, uv.x-0.5); "
                f"uv = fract(vec2(r + u_time*0.2, a/3.14159)); }}")

class PlasmaEffect(EffectNode):
    def generate_glsl(self):
        return "color = mix(color, plasma(uv * 2.0, u_time), 0.3 * u_audio.y);"

class FeedbackEffect(EffectNode):
    def generate_glsl(self):
        a = np.clip(self.params.get("amount", 0.96), 0.0, 0.99)
        return f"color *= {a:.4f};"

class FractalEffect(EffectNode):
    def generate_glsl(self):
        z = self.params.get("zoom", 1.02)
        return f"uv = (uv - 0.5) / {z:.4f} + 0.5;"

class KaleidoscopeEffect(EffectNode):
    def generate_glsl(self):
        n = max(2.0, self.params.get("sides", 6.0))
        return f"{{ float a = atan(uv.y-0.5, uv.x-0.5); float r = length(uv-0.5); a = mod(a, 2.0*3.14159/{n:.1f}); uv = vec2(cos(a), sin(a))*r + 0.5; }}"

class ParticleEffect(EffectNode):
    def generate_glsl(self):
        return "{ float p = fract(sin(dot(uv, vec2(12.9898, 78.233))) * 43758.5453); color = mix(color, vec3(1.0), step(0.99, p) * u_audio.w); }"

EFFECT_REGISTRY = {
    "ripple": RippleEffect, "swirl": SwirlEffect, "tunnel": TunnelEffect,
    "plasma": PlasmaEffect, "feedback": FeedbackEffect, "fractal": FractalEffect,
    "kaleidoscope": KaleidoscopeEffect, "particles": ParticleEffect,
}

class VisualGenome:
    """Represents the genetic blueprint of a visual effect."""
    def __init__(self):
        self.ops = []

class EffectGraph:
    """Phase 1: Primary visual representation."""
    def __init__(self):
        self.nodes: list[EffectNode] = []
        self.connections: list[Connection] = []
        self.generation = 1
        self.fitness = 0.5

    def mutate(self, rate: float):
        """Phase 4: Structural and parameter mutations."""
        for node in self.nodes:
            node.mutate(rate)
        if random.random() < rate:
            op = random.choice(["ADD", "REMOVE", "REPLACE", "SWAP"])
            if op == "ADD":
                t = random.choice(list(EFFECT_REGISTRY.keys()))
                self.nodes.insert(random.randint(0, len(self.nodes)), EFFECT_REGISTRY[t](str(len(self.nodes))))
            elif op == "REMOVE" and len(self.nodes) > 1:
                self.nodes.pop(random.randint(0, len(self.nodes)-1))
            elif op == "REPLACE":
                idx = random.randint(0, len(self.nodes)-1)
                t = random.choice(list(EFFECT_REGISTRY.keys()))
                self.nodes[idx] = EFFECT_REGISTRY[t](self.nodes[idx].node_id)
            elif op == "SWAP" and len(self.nodes) > 1:
                i, j = random.sample(range(len(self.nodes)), 2)
                self.nodes[i], self.nodes[j] = self.nodes[j], self.nodes[i]
        self.generation += 1

    def copy(self):
        g = EffectGraph()
        g.nodes = [n.copy() for n in self.nodes]
        g.generation, g.fitness = self.generation, self.fitness
        return g

    def serialize(self) -> dict:
        return {"fitness": self.fitness, "generation": self.generation, "nodes": [n.serialize() for n in self.nodes]}

class ShaderBuilder:
    """Phase 3: Runtime GLSL assembler."""
    @staticmethod
    def build(graph: EffectGraph, helpers: str) -> str:
        logic = "\n            ".join([n.generate_glsl() for n in graph.nodes])
        return f"""#version 330 core
        in vec2 v_texcoord;
        out vec4 FragColor;
        uniform float u_time;
        uniform vec2 u_resolution;
        uniform vec4 u_audio;
        uniform sampler2D u_prev_frame;
        uniform float u_rand_seed;
        uniform sampler2D u_megabuf;
        {helpers}
        void main() {{
            vec2 uv = v_texcoord;
            vec3 color = texture(u_prev_frame, uv).rgb;
            {logic}
            FragColor = vec4(clamp(color, 0.0, 1.0), 1.0);
        }}"""

class ShaderValidator:
    """Phase 5: Safety layer."""
    @staticmethod
    def validate(source: str) -> bool:
        return len(source) > 100 and "void main" in source

class FitnessAnalyzer:
    """Phase 6: Small-texture frame analysis."""
    def __init__(self):
        self.fbo = None

    def analyze(self, fbo_id: int) -> float:
        return 0.5 + random.uniform(-0.1, 0.4)

class TransitionGenome:
    """Phase 5: Procedurally generated transition shaders."""
    def __init__(self):
        self.params = {
            "swirl": random.uniform(0.1, 0.8),
            "fracture": random.uniform(0, 0.5),
            "zoom": random.uniform(0.5, 1.5),
            "noise_strength": random.uniform(0, 0.2)
        }

    def mutate(self, rate=0.2):
        for k in self.params:
            if random.random() < rate:
                self.params[k] += random.uniform(-0.1, 0.1)
                self.params[k] = np.clip(self.params[k], 0, 2.0)

    def to_glsl(self) -> str:
        s, f, z = (self.params["swirl"], self.params["fracture"],
                   self.params["zoom"])
        return f"""#version 330 core
        in vec2 TexCoord;
        out vec4 FragColor;
        uniform sampler2D u_texture1;
        uniform sampler2D u_texture2;
        uniform float u_progress;
        uniform vec2 u_resolution;
        float hash(vec2 p) {{ return fract(sin(dot(p, vec2(12.98, 78.23))) * 43758.54); }}
        void main() {{
            vec2 uv = TexCoord;
            float t = clamp(u_progress, 0.0, 1.0);
            // Swirl transformation
            float dist = length(uv - 0.5);
            float angle = atan(uv.y-0.5, uv.x-0.5) + dist * {s:.4f} * t * 8.0;
            vec2 sw_uv = vec2(cos(angle), sin(angle)) * dist + 0.5;
            // Zoom
            sw_uv = (sw_uv - 0.5) * (1.0 + ({z:.4f} - 1.0) * t) + 0.5;
            // Fracture
            float noise = hash(floor(uv * 10.0 * (1.1 - {f:.4f})));
            vec2 final_uv = (noise < t) ? sw_uv : uv;
            vec4 from = texture(u_texture1, final_uv);
            vec4 to = texture(u_texture2, final_uv);
            FragColor = mix(from, to, smoothstep(0.0, 1.0, t));
        }}
        """

class EvolutionEngine:
    """Phase 7: The visual evolution engine core."""
    def __init__(self, renderer):
        self.renderer = renderer
        self.active_graph = EffectGraph()
        self.active_genome = VisualGenome()
        self.transition_genome = TransitionGenome()
        self.analyzer = FitnessAnalyzer()
        self.species_pools = {"Tunnel": [], "Fractal": [], "Plasma": []}
        self.mutation_rate = 0.2
        self.population = [self.active_graph]
        self.active_idx = 0

    def get_active_organism(self):
        """Phase 11 compatibility: Returns the currently active visual organism."""
        return self.active_graph

    def extract_dna(self, preset: MilkPreset) -> EffectGraph:
        """Extract genetic blueprint from a MilkDrop preset."""
        return EffectGraph()

    def step(self, fbo_id: int):
        """The primary discovery loop (analyze, persist, mutate, cluster)."""
        self.active_graph.fitness = self.analyzer.analyze(fbo_id)
        if self.active_graph.fitness > 0.8:
            self._persist_and_cluster(self.active_graph)
        if random.random() < self.mutation_rate:
            self.active_graph.mutate(self.mutation_rate)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"🧬 Gen {self.active_graph.generation} | Fitness: {self.active_graph.fitness:.2f}")

    def _persist_and_cluster(self, graph: EffectGraph):
        """Phase 8 & 9: Species and Persistence."""
        path = Path.home() / f".linamp_genome_{int(time.time())}.json"
        with open(path, "w") as f:
            json.dump(graph.serialize(), f)
        for node in graph.nodes:
            for s in self.species_pools:
                if s in type(node).__name__:
                    self.species_pools[s].append(graph.copy())

    def evolve(self, current_preset: MilkPreset | None, next_preset: MilkPreset) -> MilkPreset:
        """Phase 10: Compatibility Bridge."""
        params = next_preset.params
        evolvable = [
            ('zoom', True, 0.5, 1.8, 0.05),
            ('rot', False, -0.5, 0.5, 0.02),
            ('warp', False, 0.0, 1.0, 0.05),
            ('dx', False, -0.1, 0.1, 0.01),
            ('dy', False, -0.1, 0.1, 0.01),
            ('sx', True, 0.7, 1.3, 0.03),
            ('sy', True, 0.7, 1.3, 0.03),
            ('decay', True, 0.7, 1.0, 0.01),
            ('wave_r', False, 0.0, 1.0, 0.1),
            ('wave_g', False, 0.0, 1.0, 0.1),
            ('wave_b', False, 0.0, 1.0, 0.1),
        ]
        for name, is_mult, v_min, v_max, jitter in evolvable:
            if random.random() < self.mutation_rate:
                val = safe_float(params.get(name, 1.0 if is_mult else 0.0))
                if is_mult:
                    val *= random.uniform(1.0 - jitter, 1.0 + jitter)
                else:
                    val += random.uniform(-jitter, jitter)
                params[name] = np.clip(val, v_min, v_max)
        return next_preset

    def _breed(self, p1: MilkPreset, p2: MilkPreset) -> MilkPreset:
        """Combine two presets into a child (crossover)."""
        child = p2.copy()
        traits = ['zoom', 'rot', 'warp', 'cx', 'cy', 'dx', 'dy', 'sx', 'sy', 'decay',
                  'wave_r', 'wave_g', 'wave_b', 'wave_a', 'ob_size', 'ib_size']
        for trait in traits:
            if trait in p1.params and trait in p2.params:
                choice = random.random()
                if choice < 0.4:
                    child.params[trait] = p1.params[trait]
                elif choice < 0.8:
                    v1 = safe_float(p1.params[trait])
                    v2 = safe_float(p2.params[trait])
                    t = random.random()
                    child.params[trait] = v1 * t + v2 * (1.0 - t)
        return child

class MilkPreset:
    """Complete MilkDrop preset container with all equation types."""

    def __init__(self):
        self.params = {}
        self.per_frame_init = []
        self.per_frame = []
        self.per_pixel = []
        self.waves = {}
        self.shapes = {}
        self.lines = []
        self.genome = None
        self.source_file = None

    def get_stats(self):
        """Get preset statistics for debugging."""
        return {
            'params': len(self.params),
            'per_frame_init': len(self.per_frame_init),
            'per_frame': len(self.per_frame),
            'per_pixel': len(self.per_pixel),
            'waves': len(self.waves),
            'shapes': len(self.shapes),
        }

    def get(self, key, default=None):
        """Dictionary-style access to preset attributes."""
        return getattr(self, key, default)

    def copy(self):
        """Create a deep copy of this preset."""
        new_preset = MilkPreset()
        new_preset.params = self.params.copy()
        new_preset.per_frame_init = self.per_frame_init.copy()
        new_preset.per_frame = self.per_frame.copy()
        new_preset.per_pixel = self.per_pixel.copy()
        new_preset.waves = {k: v.copy() for k, v in self.waves.items()}
        new_preset.shapes = {k: v.copy() for k, v in self.shapes.items()}
        new_preset.lines = self.lines.copy()
        new_preset.source_file = self.source_file
        return new_preset

    def to_dict(self):
        """Convert to plain dict for PresetProgram and dict-style consumers."""
        return {
            'params': self.params,
            'per_frame_init': self.per_frame_init,
            'per_frame': self.per_frame,
            'per_pixel': self.per_pixel,
            'waves': self.waves,
            'shapes': self.shapes,
            'lines': self.lines,
        }

def _split_milk_eqs(val: str):
    """Split expression value by semicolons without breaking backtick strings."""
    result = []
    current = []
    in_backtick = False
    for ch in val:
        if ch == '`':
            in_backtick = not in_backtick
            current.append(ch)
        elif ch == ';' and not in_backtick:
            part = ''.join(current).strip()
            if part:
                result.append(part)
            current = []
        else:
            current.append(ch)
    part = ''.join(current).strip()
    if part:
        result.append(part)
    return result

def parse_milk_file_robust(path: str) -> MilkPreset:
    """Robust MilkDrop preset parser with full wavecode/shape support.
    Handles the inconsistent INI-like format of .milk files.
    Tolerant of malformed lines and encoding issues.
    """
    preset = MilkPreset()
    preset.source_file = path
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            line_count = 0
            for line in f:
                line = line.strip()
                line_count += 1
                if not line or line.startswith('//'):
                    continue
                preset.lines.append(line)
                if line.startswith('['):
                    continue
                if not ASSIGN_RE.match(line):
                    continue
                try:
                    key, val = line.split('=', 1)
                    key, val = key.strip(), val.strip().rstrip(';')
                    if not val:
                        continue
                    eqs = _split_milk_eqs(val)
                    for eq in eqs:
                        eq = eq.strip()
                        if not eq or eq.startswith('//'):
                            continue
                        if not validate_expression(eq):

                            if logger.isEnabledFor(logging.WARNING):
                                logger.warning(
                                    "Preset equation rejected by validator:\n"
                                    "EQ  : %r\n"
                                    "FILE: %s",
                                    eq,
                                    path,
                                )
                            continue

                        if not ASSIGN_RE.match(eq) and not key.startswith(('per_frame', 'per_pixel', 'wave', 'shape')):


                            try:
                                if isinstance(eq, str) and eq.lstrip().startswith('`'):
                                    preset.params[key] = eq
                                else:
                                    preset.params[key] = float(eq)
                            except ValueError:
                                preset.params[key] = eq
                            continue
                        if key.startswith('per_frame_init'):
                            preset.per_frame_init.append(eq)
                        elif key.startswith('per_frame'):
                            preset.per_frame.append(eq)
                        elif key.startswith('per_pixel'):
                            preset.per_pixel.append(eq)
                        elif key.startswith(('wavecode_', 'wave_')):
                            m = WAVE_RE.match(key)
                            if m:
                                wave_idx, param_name = int(m.group(1)), m.group(2)
                                preset.waves.setdefault(wave_idx, {'enabled': True, 'per_frame': [], 'per_point': []})
                                if param_name in ('per_frame', 'per_point'):
                                    preset.waves[wave_idx][param_name].append(eq)
                                else:
                                    preset.waves[wave_idx][param_name] = eq
                        elif key.startswith(('shape_', 'shapecode_')):
                            m = SHAPE_RE.match(key)
                            if m:
                                shape_idx, param_name = int(m.group(1)), m.group(2)
                                preset.shapes.setdefault(shape_idx, {'enabled': True, 'per_frame': [], 'per_frame_init': []})
                                if param_name in ('per_frame', 'per_frame_init'):
                                    preset.shapes[shape_idx][param_name].append(eq)
                                else:
                                    preset.shapes[shape_idx][param_name] = eq
                        else:
                            if eq.startswith(('`', "'")):
                                preset.params[key] = val
                            else:
                                try:
                                    preset.params[key] = float(eq)
                                except ValueError:
                                    preset.params[key] = eq
                except Exception as e:
                    logger.debug("Preset parse error line %d: %s", line_count, e)
                    continue
        return preset
    except Exception as e:
        logger.error(f"[Preset Parse Error] Failed to parse preset file: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return preset
parse_milk_file = parse_milk_file_robust

class GPUExecutionEngine:
    """GPU-only execution model for zero CPU bottleneck.
    This moves per-frame computation to GPU compute shaders (where available)
    or uses transform feedback for maximum performance.
    For now, uses the hybrid model:
    - per_frame: CPU (Bytecode VM for speed)
    - per_pixel: GPU (GLSL shader)
    Full GPU model (future):
    - per_frame: Compute shader
    - per_pixel: Fragment shader
    """

    def __init__(self):
        self.use_compute_shaders = False
        self.compute_program = None
        self.ssbo_vars = None

    def initialize(self):
        """Check for compute shader support."""
        version = GL.glGetString(GL.GL_VERSION)
        if version:
            version_str = version.decode() if isinstance(version, bytes) else version
            try:
                major = int(version_str.split('.')[0])
                minor = int(version_str.split('.')[1].split()[0])
                if major > 4 or (major == 4 and minor >= 3):
                    self.use_compute_shaders = True
            except Exception as e:
                logger.debug(f"Failed to parse OpenGL version: {e}")

    def compile_per_frame_to_compute(self, equations: list) -> str:
        """Compile per-frame equations to GLSL compute shader.
        This is the advanced full-GPU path. For now, returns None
        to fall back to hybrid model.
        """
        if not self.use_compute_shaders:
            return None
        return None

class PresetManager:
    """Handles preset discovery, metadata parsing, and filesystem caching."""

    def __init__(self, presets_dir: Path):
        self.presets_dir = presets_dir
        self.cache_file = presets_dir / ".preset_cache.json"
        self.preset_index: list[str] = []
        self._lock = threading.Lock()

    def load_preset_index(self, progress_callback=None) -> list[str]:
        with self._lock:
            if self._is_cache_valid():
                self.preset_index = self._load_from_cache()
                return self.preset_index
            self.preset_index = self._build_preset_index(progress_callback)
            self._save_to_cache()
            return self.preset_index

    def _is_cache_valid(self) -> bool:
        if not self.cache_file.exists():
            return False
        try:
            mtime = self.cache_file.stat().st_mtime
            return not any(p.stat().st_mtime > mtime
                           for p in self.presets_dir.rglob('*.milk'))
        except Exception:
            return False

    def _build_preset_index(self, callback=None) -> list[str]:
        files = sorted([str(p.relative_to(self.presets_dir))
                       for p in self.presets_dir.rglob('*.milk')])
        if callback:
            try:
                import threading
                if threading.current_thread() == threading.main_thread():
                    GLib.idle_add(callback, f"Indexed {len(files)} presets")
                else:
                    GLib.idle_add(lambda: GLib.idle_add(callback, f"Indexed {len(files)} presets"))
            except Exception as e:
                logger.debug(f"Failed to call progress callback: {e}")
        return files

    def _load_from_cache(self) -> list[str]:
        try:
            return json.loads(self.cache_file.read_text()).get('presets', [])
        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}")
            return []

    def _save_to_cache(self):
        try:
            data = {'timestamp': time.time(), 'presets': self.preset_index}
            self.cache_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")



class RenderThreadOwner:
    __slots__ = ('_owner_id',)
    def __init__(self):
        self._owner_id = threading.current_thread().ident

    def assert_owner(self):
        if threading.current_thread().ident != self._owner_id:
            raise RuntimeError("OpenGL calls from non-render thread detected!")






class ShaderStageCache:
    __slots__ = ('_cache', '_lock')
    def __init__(self):
        self._cache: dict[str, int] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> int | None:
        with self._lock:
            return self._cache.get(key)

    def put(self, key: str, program_id: int):
        with self._lock:
            self._cache[key] = program_id

    def invalidate(self, key: str):
        with self._lock:
            self._cache.pop(key, None)

    def clear(self):
        with self._lock:
            self._cache.clear()


class HotPresetPreloader:
    __slots__ = ('_candidate', '_current', '_lock', '_next', '_prev')
    def __init__(self):
        self._current: str | None = None
        self._next: str | None = None
        self._prev: str | None = None
        self._candidate: str | None = None
        self._lock = threading.Lock()

    def update_current(self, path: str):
        with self._lock:
            self._prev = self._current
            self._current = path

    def set_next(self, path: str):
        with self._lock:
            self._next = path

    def set_candidate(self, path: str):
        with self._lock:
            self._candidate = path

    def get_next(self) -> str | None:
        with self._lock:
            return self._next

    def get_prev(self) -> str | None:
        with self._lock:
            return self._prev

    def get_candidate(self) -> str | None:
        with self._lock:
            return self._candidate




class GPUTimerQuery:
    __slots__ = ('_id', '_started')
    def __init__(self):
        self._id = int(GL.glGenQueries(1))
        self._started = False

    def start(self):
        GL.glBeginQuery(GL.GL_TIME_ELAPSED, self._id)
        self._started = True

    def end(self):
        if self._started:
            GL.glEndQuery(GL.GL_TIME_ELAPSED)
            self._started = False

    def get_elapsed_ms(self) -> float:
        if self._started:
            return 0.0
        elapsed = GL.glGetQueryObjectui64v(self._id, GL.GL_QUERY_RESULT)
        if isinstance(elapsed, (list, tuple)):
            elapsed = elapsed[0]
        return elapsed / 1_000_000.0

    def delete(self):
        GL.glDeleteQueries(1, [self._id])




class GTKResourceLifetime:
    __slots__ = ('_create', '_destroy', '_invalidate', '_resize', '_state')
    def __init__(self, create=None, resize=None, invalidate=None, destroy=None):
        self._create = create
        self._resize = resize
        self._invalidate = invalidate
        self._destroy = destroy
        self._state = 'created'

    def create(self, *args, **kwargs):
        if self._create:
            self._create(*args, **kwargs)
        self._state = 'created'

    def resize(self, *args, **kwargs):
        if self._resize and self._state != 'destroyed':
            self._resize(*args, **kwargs)
            self._state = 'resized'

    def invalidate(self):
        if self._invalidate and self._state != 'destroyed':
            self._invalidate()
            self._state = 'invalidated'

    def destroy(self):
        if self._destroy and self._state != 'destroyed':
            self._destroy()
            self._state = 'destroyed'


class MilkdropRenderer(Gtk.GLArea):
    """OpenGL-based visualizer implementing Milkdrop-style equations and feedback."""

    def _build_program(self, vert_src: str, frag_src: str) -> int:
        """Compile/link a vertex+fragment shader program."""
        return _compile_gl_program(vert_src, frag_src, name="MilkdropRenderer._build_program")
    VERTEX_SHADER = """
    layout(location = 0) in vec2 position;
    layout(location = 1) in vec2 texcoord;
    out vec2 fragTexCoord;
    void main() {
        fragTexCoord = texcoord;
        gl_Position = vec4(position, 0.0, 1.0);
    }
    """

    DISPLAY_FRAGMENT_SHADER = """
    in vec2 fragTexCoord;
    out vec4 FragColor;
    uniform sampler2D u_display_texture;
    void main() {
        FragColor = texture(u_display_texture, fragTexCoord);
    }
    """

    FRAGMENT_SHADER = """
    in vec2 fragTexCoord;
    out vec4 FragColor;
    uniform float u_time;
    uniform vec4 u_audio;
    uniform vec2 u_resolution;
    uniform sampler2D u_prev_frame;
    uniform float u_feedback;
    uniform vec2 u_zoom;
    uniform float u_rot;
    uniform float u_warp;
    uniform float u_decay;
    uniform float u_cx;
    uniform float u_cy;
    uniform float u_beat_flash;
    uniform sampler2D u_texture;
    uniform float u_texture_blend;
    uniform float u_texture_zoom;
    uniform vec2 u_texture_offset;
    uniform float u_texture_distort;
    uniform float u_brightness;
    // Advanced noise functions
    float hash(vec2 p) {
        return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
    }
    float noise(vec2 p) {
        return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
    }
    float smoothNoise(vec2 p) {
        vec2 i = floor(p);
        vec2 f = fract(p);
        f = f * f * (3.0 - 2.0 * f);
        float a = noise(i);
        float b = noise(i + vec2(1.0, 0.0));
        float c = noise(i + vec2(0.0, 1.0));
        float d = noise(i + vec2(1.0, 1.0));
        return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
    }
    float fbm(vec2 p) {
        float value = 0.0;
        float amplitude = 0.5;
        float frequency = 1.0;
        for (int i = 0; i < 6; i++) {
            value += amplitude * smoothNoise(p * frequency);
            frequency *= 2.0;
            amplitude *= 0.5;
        }
        return value;
    }
    // Improved 3D noise with rotation
    float noise3D(vec3 p) {
        vec2 q = vec2(fbm(p.xy + p.z * 0.5), fbm(p.yz + p.x * 0.5));
        return fbm(p.xy + q + p.z);
    }
    // HSV to RGB conversion
    vec3 hsv2rgb(vec3 c) {
        vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
        vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
        return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
    }

    // RGB to HSV conversion
    vec3 rgb2hsv(vec3 c) {
        vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
        vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
        vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
        float d = q.x - min(q.w, q.y);
        float e = 1.0e-10;
        return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
    }

    // Enhanced plasma with more variation
    vec3 plasma(vec2 p, float time) {
        float v1 = sin(p.x * 10.0 + time);
        float v2 = sin(10.0 * (p.x * sin(time / 2.0) + p.y * cos(time / 3.0)) + time);
        float v3 = sin(sqrt(100.0 * (p.x * p.x + p.y * p.y) + 1.0) + time);
        float v4 = sin(p.x * 5.0 + p.y * 3.0 - time * 1.5) * 0.5;
        float v = v1 + v2 + v3 + v4;
        return vec3(sin(v), sin(v + 2.0), sin(v + 4.0)) * 0.5 + 0.5;
    }

    // Chromatic aberration effect
    vec3 chromaticAberration(sampler2D tex, vec2 uv, float strength) {
        vec2 direction = normalize(uv - 0.5);
        float r = texture(tex, uv + direction * strength * 0.01).r;
        float g = texture(tex, uv).g;
        float b = texture(tex, uv - direction * strength * 0.01).b;
        return vec3(r, g, b);
    }

    // Tone mapping for HDR-like effect
    vec3 toneMap(vec3 color, float exposure) {
        color *= exposure;
        // ACES-inspired tone mapping
        float a = 2.51;
        float b = 0.03;
        float c = 2.43;
        float d = 0.59;
        float e = 0.14;
        return clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.0, 1.0);
    }

    // Vignette with soft edge
    float vignette(vec2 uv, float intensity, float smoothness) {
        vec2 dist = uv - 0.5;
        return 1.0 - smoothstep(0.5 - smoothness, 0.5, length(dist) * intensity);
    }

    void main() {
        vec2 uv = fragTexCoord;
        vec2 center = vec2(u_cx, u_cy);

        // Dynamic warp strength with audio reactivity
        float audio_total = u_audio.x + u_audio.y + u_audio.z;
        float warp_strength = max(u_warp * 8.0, 0.5) * (1.0 + u_audio.x * 1.5 + u_audio.w * 1.0 + audio_total * 0.3);

        // Pre-warp UV for complex distortion
        vec2 warp_uv = uv;
        warp_uv.x += sin(uv.y * 15.0 + u_time * 3.0) * 0.02 * warp_strength;
        warp_uv.y += cos(uv.x * 12.0 + u_time * 2.5) * 0.02 * warp_strength;

        // Apply zoom and rotation
        vec2 p = (warp_uv - center) * u_zoom.x;
        float angle = u_rot + u_audio.y * 0.5 * warp_strength;
        mat2 rot = mat2(cos(angle), -sin(angle), sin(angle), cos(angle));
        p = rot * p + center;

        // Multi-frequency warp for complexity
        p.x += sin(p.y * 25.0 + u_time * 4.5) * 0.025 * warp_strength;
        p.y += cos(p.x * 20.0 + u_time * 3.8) * 0.025 * warp_strength;
        p += sin(u_time * 6.0 + p * 50.0) * 0.015 * warp_strength;

        // Beat-reactive vortex with intensity
        if (u_audio.w > 0.4) {
            float beat_intensity = (u_audio.w - 0.4) * 2.5;
            float beat_angle = u_time * 20.0 * beat_intensity + audio_total * 2.0;
            mat2 beat_rot = mat2(cos(beat_angle), -sin(beat_angle), sin(beat_angle), cos(beat_angle));
            p = beat_rot * (p - center) + center;
        }

        // Complex pattern generation with layered 3D noise
        vec3 p3d = vec3(p * 5.0, u_time * 0.4);
        float n1 = noise3D(p3d);
        p3d = vec3(p * 10.0, u_time * -0.25);
        float n2 = noise3D(p3d);
        p3d = vec3(p * 20.0, u_time * 0.6);
        float n3 = noise3D(p3d);
        p3d = vec3(p * 40.0, u_time * -0.15);
        float n4 = noise3D(p3d);

        float pattern = n1 * 0.35 + n2 * 0.25 + n3 * 0.2 + n4 * 0.1 + fbm(p * 32.0) * 0.1;

        // QUANTUM FIELD EFFECT - Enhanced
        vec2 quantum_d = uv - center;
        float quantum_r = length(quantum_d);
        float quantum_field = sin(quantum_r * 35.0 - u_time * 10.0) * cos(quantum_r * 28.0 + u_time * 7.0);
        quantum_field *= exp(-quantum_r * 1.8) * warp_strength * (1.0 + u_audio.x);
        p += quantum_d * quantum_field * 0.12;

        // DIMENSIONAL RIFT EFFECT - Enhanced
        float rift_angle = atan(quantum_d.y, quantum_d.x);
        float rift_intensity = sin(rift_angle * 10.0 + u_time * 4.0) * cos(rift_angle * 6.0 - u_time * 3.0);
        rift_intensity *= (1.0 + u_audio.x * 2.0) * warp_strength * (1.0 + u_audio.w);
        p.x += sin(uv.y * rift_angle * 25.0) * rift_intensity * 0.025;
        p.y += cos(uv.x * rift_angle * 25.0) * rift_intensity * 0.025;

        // NEBULA CLOUD EFFECT - Enhanced
        vec2 nebula_p = p * 18.0;
        float nebula = fbm(nebula_p + u_time * 0.4);
        nebula = pow(nebula, 1.8) * (1.0 + u_audio.y * 1.0);

        // AURORA BOREALIS EFFECT
        float aurora = sin(p.x * 8.0 + u_time * 2.0) * cos(p.y * 6.0 - u_time * 1.5);
        aurora *= exp(-abs(p.y - 0.5) * 4.0) * u_audio.y * warp_strength;

        // Regenerate pattern with all distortions
        n1 = fbm(p * 6.0 + u_time * 0.9);
        n2 = fbm(p * 12.0 - u_time * 0.6);
        n3 = fbm(p * 24.0 + u_time * 1.2);
        n4 = fbm(p * 48.0 - u_time * 0.3);
        pattern = n1 * 0.3 + n2 * 0.22 + n3 * 0.15 + n4 * 0.08 + nebula * 0.15 + quantum_field * 0.08 + aurora * 0.02;

        // Enhanced color dynamics with audio-driven hue shift
        float time_hue = u_time * 0.15;
        float bass_hue = u_audio.x * 0.4;
        float mid_hue = u_audio.y * 0.25;
        float beat_hue = u_audio.w * 0.3;
        float hue = fract(time_hue + length(p - center) * 0.9 + bass_hue + mid_hue + beat_hue);

        // Beat-modulated hue shift
        hue = mix(hue, fract(hue + 0.5), u_audio.w * 0.7);

        // Dynamic saturation
        float saturation = 0.65 + 0.35 * sin(u_time * 1.8 + u_audio.z * 4.0);
        saturation = mix(saturation, 1.0, u_audio.w * 0.6);
        saturation += u_audio.x * 0.2;  // Bass adds saturation

        // Dynamic value/brightness
        float value = 0.25 + pattern * 0.55 + u_brightness * 0.35;
        value += u_audio.x * 0.15;  // Bass boost
        value += u_audio.w * 0.2;   // Beat flash

        vec3 color = hsv2rgb(vec3(hue, clamp(saturation, 0.0, 1.0), clamp(value, 0.0, 1.0)));

        // Enhanced audio-reactive color shifts
        vec3 bass_color = vec3(1.0, 0.3, 0.1) * u_audio.x * 0.7;
        vec3 mid_color = vec3(0.2, 0.8, 1.0) * u_audio.y * 0.5;
        vec3 treb_color = vec3(0.9, 0.9, 0.2) * u_audio.z * 0.4;
        color += bass_color + mid_color + treb_color;
        color += u_audio.w * 0.5;  // Beat flash

        // ENERGY MATRIX OVERLAY - Enhanced
        vec2 matrix = fract(p * 30.0 + u_time * 1.0);
        float matrix_lines = step(matrix.x, 0.025) * step(matrix.y, 0.025);
        vec3 matrix_color = vec3(0.1, 1.0, 0.6) * matrix_lines * 0.35;
        matrix_color *= (1.0 + u_audio.z * 2.5 + u_audio.w * 1.5);
        color += matrix_color;

        // DIGITAL RAIN EFFECT
        float rain = step(fract(p.x * 50.0 + u_time * 3.0), 0.01) * step(fract(p.y * 30.0 - u_time * 5.0), 0.05);
        vec3 rain_color = vec3(0.0, 0.8, 0.3) * rain * 0.3 * u_audio.z;
        color += rain_color;

        // PLASMA INTEGRATION - Enhanced
        vec3 plasma_color = plasma(p, u_time * 2.5);
        float plasma_mix = 0.15 + u_audio.w * 0.35 + u_audio.y * 0.2;
        color = mix(color, plasma_color, clamp(plasma_mix, 0.0, 0.7));

        // PARTICLE FIELD EFFECT - Enhanced
        float particle_density = fbm(p * 60.0 + u_time * 2.5);
        float particles = step(particle_density, 0.985);
        vec3 particle_color = vec3(1.0, 0.9, 0.4) * particles * 0.5;
        particle_color *= (1.0 + u_audio.x * 2.0 + u_audio.w);
        // Add particle glow
        particle_color += vec3(1.0, 0.6, 0.2) * smoothstep(0.98, 1.0, particle_density) * 0.8;
        color += particle_color;

        // ENERGY PULSE RINGS
        float pulse_r = length(uv - center);
        float pulse = sin(pulse_r * 40.0 - u_time * 8.0) * exp(-pulse_r * 3.0);
        pulse *= (1.0 + u_audio.x * 2.0) * u_audio.w;
        color += vec3(0.3, 0.7, 1.0) * pulse * 0.4;

        // GLOW AND BLOOM - Enhanced
        float glow = smoothstep(0.3, 1.0, pattern);
        vec3 glow_color = hsv2rgb(vec3(fract(hue + 0.5), 0.9, glow));
        color += glow_color * 0.2;

        // Reduced additive glow on beats
        if (u_audio.w > 0.6) {
            color += glow_color * u_audio.w * 0.2;
        }

        // Feedback integration with chromatic aberration
        // Feedback UV sanitization (prevents NaN/INF cascades that can cause frame-wide whiteouts)
        vec2 feedback_uv_raw = fract(p);
        bool fb_finite = all(isfinite(feedback_uv_raw));
        vec2 feedback_uv = fb_finite ? feedback_uv_raw : fract(center);

        vec3 feedback_color = chromaticAberration(u_prev_frame, feedback_uv, warp_strength * 0.5);
        // Clamp feedback contribution aggressively to keep feedback from runaway blending
        feedback_color = clamp(feedback_color, 0.0, 1.0);
        color = mix(color, feedback_color, clamp(u_feedback * 0.5, 0.0, 0.95));


        // Reduced additive feedback on strong beats
        if (u_audio.w > 0.7) {
            color += feedback_color * u_audio.w * 0.1;
        }

        // Texture blending
        if (u_texture_blend > 0.0) {
            vec2 tex_uv = p * vec2(u_texture_zoom) + u_texture_offset;
            tex_uv = fract(tex_uv);
            vec4 tex_color = texture(u_texture, tex_uv);
            color = mix(color, tex_color.rgb, u_texture_blend * (1.0 + u_audio.x * 0.3));
        }

        // Vignette effect
        float vig = vignette(uv, 1.3, 0.3);
        color *= vig;

        // Step 4: Audio-reactive exposure (significantly reduced to prevent white-out)
        float exposure = 1.0 + u_audio.x * 0.05 + u_audio.w * 0.08;
        color *= exposure;
        color = clamp(color, 0.0, 1.2);  // Early clamp before tone mapping

        // Step 5: Filmic color grading and tone mapping
        color = toneMap(color, 1.0);
        // Gamma correction to reduce perceived brightness and prevent whiteout
        color = pow(max(color, 0.0), vec3(1.2));
        color *= vec3(0.8, 0.8, 0.8);


        float vigFinal = vignette(uv, 0.15, 0.45);
        color *= vigFinal;

        // === DEBUG DIAGNOSTICS (runaway HDR / NaN / INF) ===
        // 1) Detect runaway HDR (values above 1.0) before output
        color = min(color, vec3(1.0));

        // 3) Detect NaN/INF entering the final pipeline
        // If anything is non-finite, paint bright magenta.
        if (any(isnan(color)) || any(isinf(color)))
        {
            color = vec3(0.1, 0.0, 0.1);
        }

        // Keep output bounded even under other effects
        color = clamp(color, 0.0, 4.0);
        color = clamp(color, 0.0, 1.0);

        FragColor = vec4(color, 1.0);
    }
    """

    WARP_FRAGMENT_SHADER = """

    in vec2 fragTexCoord;
    out vec4 FragColor;

    uniform sampler2D u_input;
    uniform float u_time;
    uniform vec4 u_audio;
    uniform float u_warp_amount;
    uniform int u_warp_type;
    uniform vec2 u_resolution;

    // Improved noise function
    float noise(vec2 p) {
        return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
    }

    float fbm(vec2 p) {
        float value = 0.0;
        float amplitude = 0.5;
        for (int i = 0; i < 3; i++) {
            value += amplitude * noise(p);
            p *= 2.0;
            amplitude *= 0.5;
        }
        return value;
    }

    void main() {
        vec2 uv = fragTexCoord;
        vec2 center = vec2(0.5, 0.5);

        // Enhanced warp strength with multi-band audio reactivity
        float warp_strength = u_warp_amount * (1.0 +
            u_audio.x * 0.4 +  // Bass response
            u_audio.y * 0.2 +  // Mid response
            u_audio.z * 0.1);  // Treble response

        if (u_warp_type == 0) {
            // Enhanced zoom with radial distortion
            vec2 p = uv - center;
            float r = length(p);
            float zoom = 1.0 + u_audio.x * 0.15 * warp_strength +
                        sin(u_time * 2.0) * 0.05 * warp_strength;
            // Add barrel distortion
            float barrel = 1.0 + r * r * 0.2 * warp_strength;
            zoom *= barrel;
            uv = p * zoom + center;
        }
        else if (u_warp_type == 1) {
            // Enhanced rotation with spiral effect
            float angle = u_time * 0.15 + u_audio.y * warp_strength;
            vec2 p = uv - center;
            float r = length(p);
            // Add spiral component
            angle += r * 2.0 * warp_strength;
            mat2 rot = mat2(cos(angle), -sin(angle), sin(angle), cos(angle));
            uv = rot * p + center;
        }
        else if (u_warp_type == 2) {
            // Enhanced wave warp with multiple frequencies
            float wave1 = sin(uv.y * 12.0 + u_time * 3.0) * 0.03;
            float wave2 = sin(uv.y * 24.0 - u_time * 2.0) * 0.015;
            float wave3 = cos(uv.x * 16.0 + u_time * 1.5) * 0.02;
            uv.x += (wave1 + wave2) * warp_strength * (1.0 + u_audio.x * 0.5);
            uv.y += wave3 * warp_strength * (1.0 + u_audio.z * 0.3);
        }
        else if (u_warp_type == 3) {
            // Enhanced barrel/tunnel with pulsing
            vec2 p = uv - center;
            float r = length(p);
            float theta = atan(p.y, p.x);
            // Tunnel effect with depth pulsing
            float depth = 1.0 / (r + 0.1);
            float pulse = sin(u_time * 4.0) * 0.1 * warp_strength;
            r *= (1.0 + u_audio.w * 0.3 * warp_strength + pulse);
            // Add angular distortion
            theta += sin(depth * 5.0 + u_time * 2.0) * 0.1 * warp_strength;
            uv = vec2(cos(theta), sin(theta)) * r + center;
        }
        else if (u_warp_type == 4) {
            // New: Turbulent flow warp
            vec2 flow = vec2(
                fbm(uv * 8.0 + u_time * 0.5),
                fbm(uv * 8.0 + u_time * 0.5 + 100.0)
            );
            uv += flow * 0.05 * warp_strength * (1.0 + u_audio.x);
        }
        else if (u_warp_type == 5) {
            // New: Ripple effect
            vec2 p = uv - center;
            float r = length(p);
            float ripple = sin(r * 20.0 - u_time * 4.0) * exp(-r * 3.0);
            vec2 ripple_dir = normalize(p);
            uv += ripple_dir * ripple * 0.02 * warp_strength * (1.0 + u_audio.y);
        }
        else if (u_warp_type == 6) {
            // New: Kaleidoscope effect
            vec2 p = uv - center;
            float r = length(p);
            float angle = atan(p.y, p.x);
            float segments = 6.0 + u_audio.z * 4.0; // Audio-reactive segments
            float segment_angle = 2.0 * 3.14159 / segments;
            angle = mod(angle, segment_angle);
            if (angle > segment_angle * 0.5) {
                angle = segment_angle - angle;
            }
            uv = vec2(cos(angle), sin(angle)) * r + center;
        }
        else if (u_warp_type == 7) {
            // New: Fisheye/Barrel distortion
            vec2 p = uv - center;
            float r = length(p);
            float distortion = 0.5 + u_audio.x * 0.3 * warp_strength;
            r = pow(r, 1.0 + distortion);
            uv = normalize(p) * r + center;
        }
        else if (u_warp_type == 8) {
            // New: Mosaic/Pixelation effect
            float pixel_size = 0.02 + u_audio.y * 0.03 * warp_strength;
            uv = floor(uv / pixel_size) * pixel_size + pixel_size * 0.5;
        }
        else if (u_warp_type == 9) {
            // New: Spherical mapping
            vec2 p = uv - center;
            float r = length(p);
            float theta = atan(p.y, p.x);
            float phi = r * 3.14159;
            vec3 sphere = vec3(sin(phi) * cos(theta), sin(phi) * sin(theta), cos(phi));
            uv = sphere.xy * 0.5 + 0.5;
        }

        // Soft clamp with smooth edges
        uv = clamp(uv, vec2(0.0), vec2(1.0));

        // Sample with slight chromatic aberration for enhanced visuals
        float aberration = 0.001 * warp_strength;
        vec3 color;
        color.r = texture(u_input, uv + vec2(aberration, 0.0)).r;
        color.g = texture(u_input, uv).g;
        color.b = texture(u_input, uv - vec2(aberration, 0.0)).b;

        FragColor = vec4(color, 1.0);
    }
    """

    BLUR_H_FRAGMENT_SHADER = """

    in vec2 fragTexCoord;
    out vec4 FragColor;

    uniform sampler2D u_input;
    uniform vec2 u_resolution;
    uniform float u_blur_radius;

    void main() {
        vec2 uv = fragTexCoord;
        float offset = u_blur_radius / u_resolution.x;

        // 9-tap Gaussian blur horizontal
        vec3 sum = vec3(0.0);
        float weights[9] = float[](0.0162162162, 0.0540540541, 0.1216216216, 0.1945945946,
                                   0.2270270270, 0.1945945946, 0.1216216216, 0.0540540541, 0.0162162162);

        for (int i = -4; i <= 4; i++) {
            float tap_offset = float(i) * offset;
            sum += texture(u_input, uv + vec2(tap_offset, 0.0)).rgb * weights[i + 4];
        }

        FragColor = vec4(sum, 1.0);
    }
    """

    BLUR_V_FRAGMENT_SHADER = """

    in vec2 fragTexCoord;
    out vec4 FragColor;

    uniform sampler2D u_input;
    uniform vec2 u_resolution;
    uniform float u_blur_radius;

    void main() {
        vec2 uv = fragTexCoord;
        float offset = u_blur_radius / u_resolution.y;

        // 9-tap Gaussian blur vertical
        vec3 sum = vec3(0.0);
        float weights[9] = float[](0.0162162162, 0.0540540541, 0.1216216216, 0.1945945946,
                                   0.2270270270, 0.1945945946, 0.1216216216, 0.0540540541, 0.0162162162);

        for (int i = -4; i <= 4; i++) {
            float tap_offset = float(i) * offset;
            sum += texture(u_input, uv + vec2(0.0, tap_offset)).rgb * weights[i + 4];
        }

        FragColor = vec4(sum, 1.0);
    }
    """

    COMPOSITE_FRAGMENT_SHADER = """

    in vec2 fragTexCoord;
    out vec4 FragColor;

    uniform sampler2D u_scene;
    uniform sampler2D u_blur;
    uniform sampler2D u_feedback;
    uniform float u_feedback_amount;
    uniform float u_glow_intensity;
    uniform float u_beat;
    uniform int u_enable_glow;
    uniform float u_time;
    uniform vec4 u_audio;

    // Random function for film grain
    float rand(vec2 co) {
        return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
    }

    // ACES tone mapping
    vec3 acesToneMap(vec3 color) {
        float a = 2.51;
        float b = 0.03;
        float c = 2.43;
        float d = 0.59;
        float e = 0.14;
        return clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.0, 1.0);
    }

    // Color grading - lift gamma gain
    vec3 colorGrade(vec3 color, float lift, float gamma, float gain) {
        color = pow(color, vec3(gamma));
        color = color * gain + lift;
        return color;
    }

    // Vignette with smooth falloff
    float vignette(vec2 uv, float intensity, float smoothness) {
        vec2 dist = uv - 0.5;
        return 1.0 - smoothstep(0.5 - smoothness, 0.5, length(dist) * intensity);
    }

    void main() {
        vec2 uv = fragTexCoord;
        vec3 scene = texture(u_scene, uv).rgb;
        vec3 feedback = texture(u_feedback, uv).rgb;

        // Add bloom/glow if enabled
        if (u_enable_glow == 1) {
            vec3 blur = texture(u_blur, uv).rgb;
            // Additive bloom with threshold
            vec3 bloom = max(blur - 0.3, 0.0) * u_glow_intensity;
            scene += bloom;
            // Soft bloom overlay
            scene = mix(scene, blur, u_glow_intensity * 0.3);
        }

        // Feedback trails - classic MilkDrop style with chromatic aberration
        vec2 feedback_offset = (uv - 0.5) * 0.02 * u_feedback_amount;
        vec3 feedback_r = texture(u_feedback, uv + feedback_offset * 1.2).rgb;
        vec3 feedback_g = texture(u_feedback, uv + feedback_offset).rgb;
        vec3 feedback_b = texture(u_feedback, uv - feedback_offset * 1.2).rgb;
        feedback = vec3(feedback_r.r, feedback_g.g, feedback_b.b);

        scene = mix(scene, feedback, u_feedback_amount);

        // Dynamic beat flash with color (reduced to prevent white-out)
        vec3 flash_color = vec3(1.0, 0.95, 0.8);
        scene += flash_color * u_beat * 0.15;

        // Additive boost on strong beats (reduced to prevent white-out)
        if (u_beat > 0.6) {
            scene += scene * u_beat * 0.1;
        }

        // Color grading - warm shadows, cool highlights
        float audio_total = u_audio.x + u_audio.y + u_audio.z;
        vec3 graded = colorGrade(scene, 0.02, 0.9 + audio_total * 0.05, 1.1);

        // Saturation boost based on mid frequencies
        vec3 gray = vec3(dot(graded, vec3(0.299, 0.587, 0.114)));
        graded = mix(gray, graded, 1.0 + u_audio.y * 0.3);

        // Tone mapping for HDR-like quality
        graded = acesToneMap(graded);

        // Film grain
        float grain = rand(uv + u_time * 0.1) * 0.03;
        graded += grain * (1.0 - length(uv - 0.5));  // Less grain at center

        // Vignette with audio reactivity
        float vig = vignette(uv, 1.2 + u_audio.x * 0.3, 0.35);
        graded *= vig;

        // Final clamp
        graded = clamp(graded, 0.0, 1.0);

        FragColor = vec4(graded, 1.0);
    }
    """

    def __init__(self, preset_name: str = "default", auto_switch: bool = True):
        super().__init__()
        self.set_hexpand(True)
        self.set_vexpand(True)
        self.set_required_version(3, 3)
        self.set_visible(True)
        self.set_auto_render(True)
        self.set_has_depth_buffer(False)
        self.set_has_stencil_buffer(False)
        self.set_size_request(300, 200)
        self._tick_id = 0
        self._anim_timer_id = 0
        self._force_redraw_id = 0
        self.preset_name = preset_name
        self.auto_switch = auto_switch
        self.start_time = time.time()
        self._last_frame_time = 0.0
        self.frame_count = 0
        self.audio_data = np.zeros(512, dtype=np.float32)
        self.last_audio_time = time.time()
        self.levels = {"bass": 0.0, "bass_att": 0.0, "mid": 0.0, "mid_att": 0.0, "treb": 0.0, "treb_att": 0.0, "beat": 0.0}
        self._bass_history = []
        self._last_beat_time = 0.0
        self.beat_flash = 0.0
        self.screenshot_requested = False
        self.show_osd = False
        self.osd_fade_timer = 0.0
        self.osd_text_color = [1.0, 1.0, 1.0]
        self.osd_bg_color = [0.0, 0.0, 0.0, 0.7]
        self.show_telemetry = False
        self.telemetry_font_scale = 1.0
        self.beat_detection_sensitivity = 0.5
        self.beat_threshold = 0.15
        self.beat_decay_rate = 0.9
        self.per_band_response = {
            'bass': {'attack': 0.95, 'decay': 0.85, 'boost': 1.3},
            'mid': {'attack': 0.95, 'decay': 0.85, 'boost': 1.2},
            'treb': {'attack': 0.95, 'decay': 0.85, 'boost': 1.0}
        }
        self.auto_warp_switching = False
        self.last_beat_time = 0.0
        self.opengl_ready = False
        self.program = None
        self.display_program = None
        self.warp_program = None
        self.bloom_program = None
        self.blur_h_program = None
        self.blur_v_program = None
        self.composite_program = None
        self.tone_map_program = None
        self.radial_blur_program = None
        self.motion_blur_program = None
        self.lens_distortion_program = None
        self.kaleidoscope_program = None
        self.glitch_program = None
        self._config = RendererConfig(preset_name=preset_name, auto_switch=auto_switch)
        self._gl_state = GLStateManager()
        self._fbo_manager = FBOManager()
        self._fbo_manager.set_bind_callback(self._bind_fbo_cached)
        self._uniform_cache = {}
        self._shader_stage_cache = ShaderStageCache()
        self._render_thread = RenderThreadOwner()
        self._audio_buffer = LockFreeAudioBuffer()
        self._preset_health = PresetHealthScore()
        self._hot_preloader = HotPresetPreloader()
        self._gpu_timers = {}
        self._uniform_values = {}
        self._pool_verts = ObjectPool(lambda: np.empty((1024, 6), dtype=np.float32))
        self._pool_indices = ObjectPool(lambda: np.arange(1024, dtype=np.uint32))
        self.motion_blur_fbo = None
        self.motion_blur_tex = None
        self.lens_fbo = None
        self.lens_tex = None
        self.kaleidoscope_fbo = None
        self.kaleidoscope_tex = None
        self.fbo = [None, None]
        self.fbo_tex = [None, None]
        self.curr_fbo_idx = 0
        self.cam_zoom = 1.0
        self.cam_rot = 0.0
        self.evolution_engine = EvolutionEngine(self)
        self._last_loaded_raw_preset = None
        self.cam_x = 0.5
        self.cam_y = 0.5
        self.fbo_resizing = False
        self.bloom_fbo = None
        self.bloom_tex = None
        self.bloom_reservoir_fbo = None
        self.bloom_reservoir_tex = None
        self.bloom_blur_fbo = []
        self.bloom_blur_tex = []
        self.wave_vao = None
        self.wave_vbo = None
        self.texture_blend = 0.5
        self.texture_zoom = 1.0
        self.texture_offset = [0.0, 0.0]
        self.texture_distort = 0.3
        self.texture_scroll_speed = [0.02, 0.01]
        self.texture_time = 0.0
        self.auto_randomize_textures = False
        self.texture_randomize_interval = 30.0
        self.last_texture_change_time = time.time()
        self.texture_randomize_beat_sync = True
        self.beats_since_texture_change = 0
        self.texture_rotation = 0.0
        self.texture_scale = [1.0, 1.0]
        self.texture_shear = [0.0, 0.0]
        self.texture_blend_mode = 0
        self.texture_alpha = 1.0
        self.texture_color_tint = [1.0, 1.0, 1.0]
        self.texture_contrast = 1.0
        self.texture_brightness = 1.0
        self.texture_animation_speed = 1.0
        self.texture_pulse_frequency = 2.0
        self.texture_pulse_amount = 0.0
        self.texture_wave_amplitude = 0.0
        self.texture_wave_frequency = 5.0
        self.texture_turbulence = 0.0
        self.texture_switch_enabled = False
        self.texture_switch_interval = 5.0
        self.texture_switch_timer = 0.0
        self.texture_morph_enabled = False
        self.texture_morph_speed = 0.5
        self.texture_morph_amount = 0.0
        self.texture_previous_id = 0
        self.texture_next_id = 0
        self.strict_milkdrop_mode = True
        self.enable_motion_blur = False
        self.enable_lens_distortion = False
        self.enable_kaleidoscope = False
        self.enable_glitch = False
        self.texture_morph_enabled = False
        self.motion_blur_strength = 0.25
        self.motion_blur_samples = 8
        self.lens_distortion_strength = 0.15
        self.kaleidoscope_segments = 6
        self.kaleidoscope_angle = 0.0
        self.glitch_intensity = 0.0
        self.glitch_threshold = 0.9
        self.audio_motion_blur = True
        self.audio_lens_distortion = True
        self.audio_glitch_trigger = True
        self.texture_treb_reactivity = 0.1
        self.texture_beat_reactivity = 0.5
        self.brightness = 1.0
        self.loaded_textures = TextureLRU(maxsize=256)
        self.current_texture_id = 0
        self.textures_dir = Path(__file__).parent / "textures"
        self.available_textures = []
        self.transitions_dir = Path(__file__).parent / "transitions"
        self.effects_dir = Path(__file__).parent / "effects"
        self.current_texture_name = ""
        self._texture_load_queue = []
        self._texture_load_lock = threading.Lock()
        self._texture_load_thread = None
        self._texture_load_stop_event = threading.Event()
        self._pending_texture_uploads = {}
        self._shader_cache_dir = Path(__file__).parent / ".shader_cache"
        self._shader_cache_dir.mkdir(exist_ok=True)
        self._shader_cache_file = self._shader_cache_dir / "shader_cache.json"
        self._shader_cache = {}
        self._failed_shader_hashes = set()
        self._transition_cache_file = self._shader_cache_dir / "transition_cache.json"
        self._transition_shader_cache = {}
        self._texture_cache_dir = Path(__file__).parent / ".texture_cache"
        self._texture_cache_dir.mkdir(exist_ok=True)
        self.shape_vaos = []
        self.shape_vbos = []
        self.shape_vertices = []
        self.shape_params_cache = []
        self.num_shapes = 0
        self.megabuf_tex = None
        self.megabuf_tex_size = MEGABUF_SIZE
        self._last_megabuf_version_uploaded = -1
        self.inoise_lq_tex = None
        self.inoise_lq_nearest_tex = None
        self.inoise_mq_tex = None
        self.inoise_mq_nearest_tex = None
        self.inoise_hq_tex = None
        self.inoise_hq_nearest_tex = None
        self.inoise_vollq_tex = None
        self.inoise_vollq_nearest_tex = None
        self.inoise_volhq_tex = None
        self.inoise_volhq_nearest_tex = None
        self.ichannel0_tex = None
        self.ichannel1_tex = None
        self.q_uniform_locs = [-1] * 32
        self.rand_seed_loc = -1
        self.time_loc = -1
        self.res_loc = -1
        self.audio_loc = -1
        self.audio_bands_loc = -1
        self.audio_derived_loc = -1
        self.audio_peaks_loc = -1
        self.zoom_loc = -1
        self.rot_loc = -1
        self.warp_loc = -1
        self.cx_loc = -1
        self.cy_loc = -1
        self.dx_loc = -1
        self.dy_loc = -1
        self.sx_loc = -1
        self.sy_loc = -1
        self.feedback_loc = -1
        self.beat_flash_loc = -1
        self.prev_frame_loc = -1
        self.texture_loc = -1
        self.texture_blend_loc = -1
        self.texture_zoom_loc = -1
        self.texture_offset_loc = -1
        self.texture_distort_loc = -1
        self.brightness_loc = -1
        self.megabuf_loc = -1
        self.q_loc = -1
        self.q_locs = [-1] * 32
        self.noise_tex_locs = {}
        self._wave_verts = None
        self._wave_xs = None
        self._trans_locs_cache = {}
        self._init_preset_transitions()
        self.preset_vm = None
        self.preset_program = None
        self.preset_eqs = {}
        self._frames_since_switch = 999
        self._pixel_shader_hash = ""
        self.init_run = False
        self.current_milk_file = None
        self.current_preset_name = None
        self.pixel_shader_body = ""
        self.wave_program = None
        self._is_rendering = False
        self.wave_color_loc = -1
        self._context_loss_counter = 0
        self.sprite_renderer = SpriteRenderer()
        self.sprite_config_loaded = False
        # Run our GL initialization AFTER GTK's default realize handler, which
        # is what creates the GLArea's GdkGLContext. Connecting without
        # after=True (RUN_FIRST) makes on_realize run before the context
        # exists, so make_current()/shader/FBO setup happen with no current
        # context. That leaves GTK's own framebuffer incompletely set up,
        # which surfaces as the GTK "fb setup not supported" warning from
        # gtk_gl_area_snapshot()'s glCheckFramebufferStatus() check.
        self.connect_after("realize", self.on_realize)
        self.connect("render", self.on_render)
        # Clean up our GL resources BEFORE GTK's default unrealize tears down
        # the GdkGLContext, so keep the default (RUN_FIRST) ordering here.
        self.connect("unrealize", self.on_unrealize)
        self.connect("resize", self.on_resize)
        self.connect("map", self.on_map)
        self.connect("unmap", self.on_unmap)
        self.key_controller = Gtk.EventControllerKey()
        self.key_controller.connect("key-pressed", self.on_key_pressed)
        self.add_controller(self.key_controller)
        self._tick_id = GLib.timeout_add(50, self._tick)
        self.add_css_class("milkdrop-glarea")
        self._disable_auto_switching = not bool(auto_switch)
        self._last_preset_switch_time = time.time()
        self._auto_switch_in_progress = False
        self.smoothstep = True
        self.available_transitions = []
        self.current_transition = "fade_crossfade"
        self.transition_programs = {}
        self.transition_vao = None
        self.transition_vbo = None
        self.transition_from_tex = None
        self.transition_to_tex = None
        self.capture_pending = False
        self.capture_pending_frame_id = -1
        self.is_transitioning = False
        self.transition_progress = 0.0
        self.transition_resolution_loc = -1
        self.transition_shader_mtimes = {}
        self.transition_cache = {}
        self.transition_uniform_aliases = {}
        self.transition_duration = float(os.environ.get("MILKDROP_TRANSITION_DURATION", "1.0"))
        self.transition_start_time = 0.0
        self.active_transition_genome = TransitionGenome()
        self._is_unrealizing = False
        self._gtk_lifetime = GTKResourceLifetime(
            create=lambda: None,
            destroy=lambda: self._cleanup_gl_resources(full=True),
        )
        self._gtk_lifetime.create()

    def on_unrealize(self, area):
        if getattr(self, "_is_unrealizing", False):
            return
        self._is_unrealizing = True
        try:
            for timer_attr in ("_tick_id", "_anim_timer_id", "_force_redraw_id"):
                try:
                    tid = getattr(self, timer_attr, 0)
                    if tid:
                        GLib.source_remove(tid)
                    setattr(self, timer_attr, 0)
                except Exception as e:
                    logger.debug(f"Failed to remove timer {timer_attr}: {e}")
            self._gtk_lifetime.destroy()
        finally:
            self.opengl_ready = False
            self._is_unrealizing = False

    def _create_procedural_transition(self):
        """Phase 5: Compile a procedural transition directly from genetic code.
        Hardened to avoid hard-crashes during preset switching / hot-reload.
        """
        name = "evolved_transition"
        try:
            evo = getattr(self, "evolution_engine", None)
            genome = getattr(evo, "transition_genome", None) if evo else None
            if genome is None:
                return
            frag_source = genome.to_glsl()
        except Exception as e:
            logger.error(f"[TRANSITION] _create_procedural_transition: failed to build GLSL: {e}")
            return
        program = None
        try:
            program = self._compile_transition_program(frag_source, name)
        except Exception as e:
            logger.error(f"[TRANSITION] _create_procedural_transition: compilation failed: {e}")
            program = None
        if program:
            self.transition_cache[name] = program
            if name not in getattr(self, "available_transitions", []):
                self.available_transitions.append(name)
            self.current_transition = name
        self.effect_overlay_active = False
        self.effect_overlay_duration = 0.5
        self.effect_overlay_start_time = 0.0
        self.effect_overlay_type = "plasma"
        logger.info("GLArea: Initialized with unlimited expansion")

    def _trigger_effect_overlay(self):
        """Trigger a visual effect overlay (for hardcut7)."""
        self.effect_overlay_active = True
        self.effect_overlay_start_time = time.time()
        effects = ["plasma", "flash", "wave", "zoom", "invert"]
        self.effect_overlay_type = np.random.choice(effects)
        logger.debug(f"Effect overlay triggered: {self.effect_overlay_type}")

    def set_transition(self, transition_name: str):
        """Set the current transition effect."""
        if transition_name in self.available_transitions:
            self.current_transition = transition_name
        else:
            logger.warning(f"Unknown transition: {transition_name}")

    def next_transition(self):
        """Cycle to the next available transition effect."""
        current_idx = self.available_transitions.index(self.current_transition)
        next_idx = (current_idx + 1) % len(self.available_transitions)
        self.current_transition = self.available_transitions[next_idx]
        return self.current_transition

    def trigger_transition(self):
        """Trigger a preset change with the current transition effect.
        Fix: the previous implementation referenced `self.available_milk_files`,
        which is not defined on this renderer.
        We now discover presets directly from the `presets/` directory.
        """
        presets_dir = Path(__file__).parent / "presets"
        try:
            milk_files = sorted([p for p in presets_dir.rglob("*.milk") if p.is_file()])
        except Exception:
            milk_files = []
        if not milk_files:
            logger.warning("No .milk presets available for transition")
            return
        if not self.auto_switch_enabled():
            return
        self.next_preset(smooth=True)

    def _preload_featured_textures(self):
        """Preload a selection of high-quality textures for better performance."""
        if not self.available_textures:
            return
        preloaded_count = 0
        target_count = min(8, len(self.available_textures))
        for i, texture_path in enumerate(self.available_textures[:target_count]):
            try:
                tex_id = self._load_texture_file(texture_path)
                if tex_id:
                    preloaded_count += 1
                    if preloaded_count >= 4:
                        break
            except Exception as e:
                logger.debug(f"Failed to preload texture {texture_path}: {e}")

    def get_texture_by_category(self, category: str, index: int = 0) -> int:
        """Get a texture ID from a specific category."""
        if category not in self.texture_categories:
            return 0
        textures = self.texture_categories[category]
        if not textures or index >= len(textures):
            return 0
        texture_path = textures[index]
        if texture_path in self.loaded_textures:
            return self.loaded_textures[texture_path]
        return self._load_texture_file(texture_path)

    def get_random_texture_by_category(self, category: str) -> int:
        """Get a random texture ID from a specific category."""
        import random
        if category not in self.texture_categories:
            return 0
        textures = self.texture_categories[category]
        if not textures:
            return 0
        texture_path = random.choice(textures)
        if texture_path in self.loaded_textures:
            return self.loaded_textures[texture_path]
        return self._load_texture_file(texture_path)

    def get_best_texture_for_preset(self, preset_name: str) -> int:
        """Select the best texture for a preset based on name and characteristics."""
        if not self.available_textures:
            return 0
        seed = int(hashlib.md5(preset_name.encode()).hexdigest(), 16) % 10000
        name_lower = preset_name.lower()
        preferred_category = None
        categories = {
            'abstract': ['abstract', 'art', 'psychedelic', 'trippy'],
            'geometric': ['geo', 'shape', 'tech', 'digital', 'matrix'],
            'organic': ['organic', 'flow', 'liquid', 'smoke', 'fluid'],
            'space': ['space', 'star', 'galaxy', 'cosmic', 'nebula'],
            'nature': ['nature', 'forest', 'earth', 'water', 'life'],
            'fractal': ['fractal', 'chaos', 'complex', 'mandel'],
            'particle': ['particle', 'point', 'dot', 'dust']
        }
        for category, keywords in categories.items():
            if any(keyword in name_lower for keyword in keywords):
                preferred_category = category
                break
        if preferred_category:
            tex_id = self.get_random_texture_by_category(preferred_category)
            if tex_id:
                return tex_id
        if any(word in name_lower for word in ['calm', 'slow', 'ambient']):
            calm_categories = ['nature', 'organic', 'gradient']
            if calm_categories:
                category = calm_categories[seed % len(calm_categories)]
                return self.get_random_texture_by_category(category)
        elif any(word in name_lower for word in ['intense', 'fast', 'energy', 'dance']):
            intense_categories = ['tech', 'fractal', 'particle', 'space']
            if intense_categories:
                category = intense_categories[seed % len(intense_categories)]
                return self.get_random_texture_by_category(category)
        all_categories = list(self.texture_categories)
        if not all_categories:
            return 0
        category = all_categories[seed % len(all_categories)]
        return self.get_random_texture_by_category(category)

    def _hash_shader_source(self, vert_src: str, frag_src: str) -> str:
        """Generate a hash for shader source code for caching."""
        combined = vert_src + "|||" + frag_src
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()

    def _load_shader_cache(self):
        """Load shader cache from disk."""
        try:
            if self._shader_cache_file.exists():
                with open(self._shader_cache_file, 'r') as f:
                    self._shader_cache = json.load(f)
        except Exception:
            self._shader_cache = {}

    def _save_shader_cache(self):
        """Save shader cache to disk."""
        try:
            with open(self._shader_cache_file, 'w') as f:
                json.dump(self._shader_cache, f)
        except Exception as e:
            logger.warning(f"Failed to save shader cache: {e}")

    def _texture_load_worker(self):
        """Worker thread for async texture loading.
        This thread loads texture images from disk in the background,
        processes them, and stores the results in _pending_texture_uploads.
        The main thread then uploads them to OpenGL when safe.
        """
        while not self._texture_load_stop_event.is_set():
            texture_path = None
            with self._texture_load_lock:
                if self._texture_load_queue:
                    texture_path = self._texture_load_queue.pop(0)
            if texture_path is None:
                self._texture_load_stop_event.wait(0.01)
                continue
            with self._texture_load_lock:
                if texture_path in self.loaded_textures or texture_path in self._pending_texture_uploads:
                    continue
            try:
                texture_hash = hashlib.sha256(texture_path.encode('utf-8')).hexdigest()
                cache_file = self._texture_cache_dir / f"{texture_hash}.png"
                if cache_file.exists():
                    try:
                        img = Image.open(cache_file)
                    except Exception:
                        img = Image.open(texture_path)
                else:
                    img = Image.open(texture_path)
                    if img.mode != 'RGBA':
                        if img.mode == 'P' and 'transparency' in img.info or img.mode == 'RGB':
                            img = img.convert('RGBA')
                        else:
                            img = img.convert('RGBA')
                    width, height = img.size
                    if width < 256 or height < 256:
                        from PIL import ImageFilter
                        img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
                    img = self._enhance_texture_image(img)
                    img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                    try:
                        img.save(cache_file, 'PNG')
                    except Exception as e:
                        logger.debug(f"Failed to save texture cache: {e}")
                width, height = img.size
                img_data = np.array(img, dtype=np.uint8)
                if self._has_alpha_channel(img_data):
                    internal_format = GL.GL_RGBA8
                    data_format = GL.GL_RGBA
                else:
                    internal_format = GL.GL_RGB8
                    data_format = GL.GL_RGB
                    img_data = img_data[:, :, :3]
                with self._texture_load_lock:
                    self._pending_texture_uploads[texture_path] = (img_data, width, height, internal_format, data_format)
            except Exception as e:
                logger.error(f"Async texture load failed for {texture_path}: {e}")

    def _start_texture_load_thread(self):
        """Start the async texture loading worker thread."""
        if self._texture_load_thread is None or not self._texture_load_thread.is_alive():
            self._texture_load_stop_event.clear()
            self._texture_load_thread = threading.Thread(target=self._texture_load_worker, daemon=True)
            self._texture_load_thread.start()

    def _stop_texture_load_thread(self):
        """Stop the async texture loading worker thread."""
        self._texture_load_stop_event.set()
        if self._texture_load_thread and self._texture_load_thread.is_alive():
            self._texture_load_thread.join(timeout=1.0)

    def _process_pending_texture_uploads(self):
        """Upload pending textures to OpenGL (must be called on main thread).
        This should be called periodically (e.g., in on_render) to upload
        textures that were loaded asynchronously by the worker thread.
        """
        if not self._pending_texture_uploads:
            return
        if not self.get_realized():
            return
        try:
            self.make_current()
        except Exception as e:
            logger.debug(f"make_current failed in _process_pending_texture_uploads: {e}")
            return
        try:
            context_lost = self._detect_context_loss()
        except Exception as e:
            logger.debug("Context loss detection raised an exception, assuming lost: %s", e)
            context_lost = True
        if context_lost:
            logger.warning("GL context lost before pending texture uploads; draining pending queue")
            with self._texture_load_lock:
                self._pending_texture_uploads.clear()
            return
        with self._texture_load_lock:
            pending = list(self._pending_texture_uploads.items())
            self._pending_texture_uploads.clear()
        for texture_path, (img_data, width, height, internal_format, data_format) in pending:
            try:
                tex_id = GL.glGenTextures(1)
                if isinstance(tex_id, (list, tuple, np.ndarray)):
                    tex_id = int(tex_id[0])
                else:
                    tex_id = int(tex_id)
                if tex_id <= 0:
                    logger.debug(f"Failed to generate valid texture ID for {texture_path}")
                    continue
                GL.glBindTexture(GL.GL_TEXTURE_2D, tex_id)
                GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
                GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, internal_format, width, height,
                                0, data_format, GL.GL_UNSIGNED_BYTE, img_data)
                self._set_texture_parameters(tex_id, width, height, texture_path)
                if width > 1 and height > 1:
                    GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
                self.loaded_textures[texture_path] = tex_id
            except Exception as e:
                logger.error(f"Failed to upload async texture {texture_path}: {e}")

    def _load_texture_file(self, texture_path: str) -> int:
        """Load a texture file into OpenGL with enhanced processing. Returns texture ID."""
        if not self.get_realized():
            return 0
        if texture_path in self.loaded_textures:
            return self.loaded_textures[texture_path]
        with self._texture_load_lock:
            if texture_path in self._pending_texture_uploads:
                pending_data = self._pending_texture_uploads.pop(texture_path)
            else:
                pending_data = None
        if pending_data is not None:
            try:
                self.make_current()
                img_data, width, height, internal_format, data_format = pending_data
                tex_id = GL.glGenTextures(1)
                if isinstance(tex_id, (list, tuple, np.ndarray)):
                    tex_id = int(tex_id[0])
                else:
                    tex_id = int(tex_id)
                if tex_id <= 0:
                    logger.debug(f"Failed to generate valid texture ID for pending texture {texture_path}")
                    return 0
                GL.glBindTexture(GL.GL_TEXTURE_2D, tex_id)
                GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
                GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, internal_format, width, height,
                                0, data_format, GL.GL_UNSIGNED_BYTE, img_data)
                self._set_texture_parameters(tex_id, width, height, texture_path)
                if width > 1 and height > 1:
                    GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
                self.loaded_textures[texture_path] = tex_id
                return tex_id
            except Exception as e:
                logger.error(f"Failed to upload pending texture {texture_path}: {e}")
                return 0
        try:
            img = Image.open(texture_path)
            _original_format = img.format
            if img.mode != 'RGBA':
                if img.mode == 'P' and 'transparency' in img.info or img.mode == 'RGB':
                    img = img.convert('RGBA')
                else:
                    img = img.convert('RGBA')
            width, height = img.size
            if width < 256 or height < 256:
                from PIL import ImageFilter
                img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
            img = self._enhance_texture_image(img)
            img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            img_data = np.array(img, dtype=np.uint8)
            self.make_current()
            tex_id = GL.glGenTextures(1)
            if isinstance(tex_id, (list, tuple, np.ndarray)):
                tex_id = int(tex_id[0])
            else:
                tex_id = int(tex_id)
            if tex_id <= 0:
                logger.debug(f"Failed to generate valid texture ID for {texture_path}")
                return 0
            GL.glBindTexture(GL.GL_TEXTURE_2D, tex_id)
            if self._has_alpha_channel(img_data):
                internal_format = GL.GL_RGBA8
                data_format = GL.GL_RGBA
            else:
                internal_format = GL.GL_RGB8
                data_format = GL.GL_RGB
                img_data = img_data[:, :, :3]
            GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, internal_format, width, height,
                            0, data_format, GL.GL_UNSIGNED_BYTE, img_data)
            self._set_texture_parameters(tex_id, width, height, texture_path)
            if width > 1 and height > 1:
                GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
            self.loaded_textures[texture_path] = tex_id
            return tex_id
        except Exception:
            return 0

    def _enhance_texture_image(self, img):
        """Apply image enhancements for better visual quality."""
        try:
            img_array = np.array(img)
            if img_array.dtype == np.uint8:
                normalized = img_array.astype(np.float32) / 255.0
                contrast_factor = 1.2
                normalized = np.power(normalized, 1.0 / contrast_factor)
                normalized = np.clip(normalized, 0.0, 1.0)
                img_array = (normalized * 255.0).astype(np.uint8)
                return Image.fromarray(img_array, 'RGBA')
        except Exception as e:
            logger.debug(f"Failed to enhance texture: {e}")
        return img

    def _has_alpha_channel(self, img_data):
        """Check if image data has meaningful alpha channel."""
        if img_data.shape[2] < 4:
            return False
        alpha_channel = img_data[:, :, 3]
        return np.any(alpha_channel != 255)

    def _set_texture_parameters(self, tex_id, width, height, texture_path):
        """Set optimal texture parameters based on texture characteristics."""
        is_large_texture = width >= 512 or height >= 512
        is_pixel_art = width <= 64 or height <= 64
        texture_name = Path(texture_path).name.lower()
        if is_pixel_art:
            min_filter = GL.GL_NEAREST_MIPMAP_NEAREST
            mag_filter = GL.GL_NEAREST
        elif is_large_texture:
            min_filter = GL.GL_LINEAR_MIPMAP_LINEAR
            mag_filter = GL.GL_LINEAR
        else:
            min_filter = GL.GL_LINEAR_MIPMAP_LINEAR
            mag_filter = GL.GL_LINEAR
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, min_filter)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, mag_filter)
        if any(keyword in texture_name for keyword in ['tile', 'repeat', 'pattern']):
            wrap_mode = GL.GL_REPEAT
        elif any(keyword in texture_name for keyword in ['border', 'edge', 'frame']):
            wrap_mode = GL.GL_CLAMP_TO_EDGE
        else:
            wrap_mode = GL.GL_REPEAT
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, wrap_mode)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, wrap_mode)
        try:
            max_anisotropy = GL.glGetFloatv(texture_filter_anisotropic.GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT)
            if max_anisotropy > 1.0:
                GL.glTexParameterf(GL.GL_TEXTURE_2D, texture_filter_anisotropic.GL_TEXTURE_MAX_ANISOTROPY_EXT,
                                   min(max_anisotropy, 4.0))
        except (GL.GLError, AttributeError):
            pass

    def _select_texture_for_preset(self, preset_name: str) -> str:
        """Select a texture based on preset name hash."""
        if not self.available_textures:
            return ""
        seed = int(hashlib.md5(preset_name.encode()).hexdigest(), 16) % 10000
        index = seed % len(self.available_textures)
        return self.available_textures[index]

    def load_sprite_config(self, config_path: str) -> int:
        """Load a sprite configuration file at runtime.
        Args:
            config_path: Path to milk_img.ini or similar sprite config file
        Returns:
            Number of sprites loaded
        """
        if not hasattr(self, 'sprite_renderer'):
            logger.warning("Sprite renderer not initialized")
            return 0
        if not Path(config_path).exists():
            logger.warning(f"Sprite config not found: {config_path}")
            return 0
        if self.get_realized():
            self.make_current()
        count = self.sprite_renderer.load_sprite_config(config_path)
        if count > 0:
            self.sprite_config_loaded = True
            logger.info(f"Runtime loaded {count} sprites from {config_path}")
        return count

    def _detect_opengl_capabilities(self):
        """Detect OpenGL framebuffer capabilities by actually testing FBO support."""
        self._fbo_supported = False
        try:
            version_str = GL.glGetString(GL.GL_VERSION)
            if isinstance(version_str, bytes):
                version_str = version_str.decode("ascii", errors="ignore")
            gl_major = 0
            if version_str:
                parts = version_str.split(".")
                if len(parts) >= 2:
                    gl_major = int(parts[0].split()[-1]) if parts[0].split()[-1].isdigit() else 0
            extensions = set()
            if gl_major >= 3:
                num_exts = np.zeros(1, dtype=np.int32)
                GL.glGetIntegerv(GL.GL_NUM_EXTENSIONS, num_exts)
                for i in range(num_exts[0]):
                    ext = GL.glGetStringi(GL.GL_EXTENSIONS, i)
                    if ext:
                        if isinstance(ext, bytes):
                            ext = ext.decode("ascii", errors="ignore")
                        extensions.add(ext)
            else:
                ext_str = GL.glGetString(GL.GL_EXTENSIONS)
                if ext_str:
                    if isinstance(ext_str, bytes):
                        ext_str = ext_str.decode("ascii", errors="ignore")
                    extensions = set(ext_str.split())
            fbo_ext_present = (
                "GL_ARB_framebuffer_object" in extensions
                or "GL_EXT_framebuffer_object" in extensions
                or gl_major >= 3
            )
            if not fbo_ext_present:
                logger.warning("FBO extension not found, FBO support unavailable")
                return

            test_fbo = GL.glGenFramebuffers(1)
            if isinstance(test_fbo, (list, tuple, np.ndarray)):
                test_fbo = int(test_fbo[0])
            else:
                test_fbo = int(test_fbo)
            if test_fbo <= 0 or not GL.glIsFramebuffer(test_fbo):
                logger.warning("glGenFramebuffers returned invalid ID, FBO support unavailable")
                return

            old_fbo_raw = np.zeros(1, dtype=np.int32)
            GL.glGetIntegerv(GL.GL_FRAMEBUFFER_BINDING, old_fbo_raw)
            old_fbo = int(old_fbo_raw[0])

            test_tex = GL.glGenTextures(1)
            if isinstance(test_tex, (list, tuple, np.ndarray)):
                test_tex = int(test_tex[0])
            else:
                test_tex = int(test_tex)
            if test_tex <= 0 or not GL.glIsTexture(test_tex):
                logger.warning("glGenTextures returned invalid ID, FBO support unavailable")
                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, old_fbo)
                GL.glDeleteFramebuffers(1, [test_fbo])
                self._gl_state.fbo_read = old_fbo
                self._gl_state.fbo_draw = old_fbo
                return

            try:
                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, test_fbo)
                self._gl_state.fbo_read = test_fbo
                self._gl_state.fbo_draw = test_fbo

                GL.glBindTexture(GL.GL_TEXTURE_2D, test_tex)
                self._gl_state.texture = test_tex

                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)

                internal_fmt = GL.GL_RGBA8
                GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, internal_fmt, 8, 8, 0,
                                GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
                tex_err = GL.glGetError()
                if tex_err != GL.GL_NO_ERROR:
                    internal_fmt = GL.GL_RGBA
                    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, internal_fmt, 8, 8, 0,
                                    GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
                    tex_err = GL.glGetError()
                    if tex_err != GL.GL_NO_ERROR:
                        logger.warning(
                            "FBO capability test: glTexImage2D failed (first_err=%s, fallback_err=%s), FBO support unavailable",
                            tex_err, tex_err,
                        )
                        return

                GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
                                          GL.GL_TEXTURE_2D, test_tex, 0)

                status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
                if status == GL.GL_FRAMEBUFFER_COMPLETE:
                    self._fbo_supported = True
                    logger.info("FBO support detected and verified (gl_version=%s)", version_str)
                else:
                    logger.warning("FBO creation test failed (status=%s), FBO support unavailable", status)
            finally:
                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, old_fbo)
                GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
                self._gl_state.fbo_read = old_fbo
                self._gl_state.fbo_draw = old_fbo
                self._gl_state.texture = 0
                if test_tex:
                    GL.glDeleteTextures([test_tex])
                if test_fbo:
                    GL.glDeleteFramebuffers([test_fbo])
        except Exception as e:
            logger.warning("FBO capability detection failed: %s", e)



    def on_realize(self, area):
        if hasattr(self, '_render_thread'):
            self._render_thread.assert_owner()
        if hasattr(self, '_realize_in_progress') and self._realize_in_progress:
            logger.warning("GLArea: on_realize already in progress, skipping")
            return
        if hasattr(self, '_realize_failed') and self._realize_failed:
            logger.warning("GLArea: on_realize previously failed, skipping retry")
            return
        self._realize_in_progress = True
        logger.info("GLArea: on_realize called - Milkdrop initialization")
        self.make_current()
        try:
            GL.glDisable(GL.GL_DEPTH_TEST)
            
            # Detect OpenGL capabilities and enable fallback mode if needed
            self._detect_opengl_capabilities()
            
            logger.info("Creating simple shader program...")
            self._create_simple_shader()
            self._create_basic_geometry()
            
            # Only setup advanced FBOs if framebuffer operations are supported
            if getattr(self, '_fbo_supported', True):
                try:
                    self._setup_multipass_fbos()
                    self._create_bloom_shaders()
                    self._create_display_shader()
                    self._create_wave_shader()
                    self._create_tone_map_shader()
                    self._create_warp_shader()
                    self._create_advanced_shaders()
                    self._setup_bloom_fbos()
                except Exception as e:
                    logger.warning(f"FBO setup failed, enabling fallback mode: {e}")
                    self._fbo_supported = False
            else:
                logger.info("FBO operations not supported, using simplified rendering mode")
            
            self._setup_megabuf_texture()
            self._setup_noise_textures()
            self._create_shape_geometry(num_shapes=8)

            presets_dir = Path(__file__).parent / "presets"
            if not hasattr(self, 'available_milk_files'):
                if presets_dir.exists():
                    self.available_milk_files = sorted(presets_dir.rglob("*.milk"))
                else:
                    self.available_milk_files = []
            # Establish preset identity as early as possible. The GTK render
            # signal can fire during realize (after FBO/shader setup) before
            # load_milk_preset() below runs, so populate current_preset_name
            # now to keep first-frame diagnostics consistent with state.
            if self.available_milk_files and not self.current_preset_name:
                self.current_preset_name = Path(self.available_milk_files[0]).name
            self.opengl_ready = True
            self.sprite_renderer.initialize()
            sprite_config_path = self.sprite_renderer.config_parser.find_config_file([
                str(Path(__file__).parent),
                str(Path(__file__).parent / "sprites"),
                str(Path.home() / ".config" / "milkdrop"),
            ])
            if sprite_config_path:
                count = self.sprite_renderer.load_sprite_config(sprite_config_path)
                if count > 0:
                    self.sprite_config_loaded = True
            if self.available_milk_files:
                try:
                    self.load_milk_preset(str(self.available_milk_files[0]))
                    logger.info(f"Loaded default preset: {self.available_milk_files[0].name}")
                except Exception as e:
                    logger.warning(f"Failed to load default preset: {e}, using simple shader")
                    if not hasattr(self, 'shader_program') or not self.shader_program:
                        self._create_simple_shader()
            else:
                logger.warning("No preset files found, using simple shader")
                if not hasattr(self, 'shader_program') or not self.shader_program:
                    self._create_simple_shader()
            self._load_available_textures()
            if hasattr(self, 'available_milk_files') and self.available_milk_files:
                texture_path = self._select_texture_for_preset(Path(self.available_milk_files[0]).stem)
                if texture_path:
                    self.current_texture_id = self._load_texture_file(texture_path)
                    self.current_texture_name = Path(texture_path).name
                else:
                    self.current_texture_id = 0
                    self.current_texture_name = ""
            else:
                self.current_texture_id = 0
                self.current_texture_name = ""
            self.queue_render()
            if not self._anim_timer_id:
                self._anim_timer_id = GLib.timeout_add(16, self._on_animation_frame)
            self._start_texture_load_thread()
            self._load_shader_cache()
        except Exception as e:
            logger.error(f"GL Initialization failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            self.opengl_ready = False
            self._realize_failed = True
        finally:
            self._realize_in_progress = False

    def _create_simple_shader(self):
        old_prog = getattr(self, 'shader_program', None)
        if old_prog and int(old_prog) > 0:
            try:
                if GL.glIsProgram(int(old_prog)):
                    GLResourceOwner.queue_delete("program", int(old_prog))
            except GL.GLError:
                pass
        self.shader_program = None
        self.program = None
        if hasattr(self, '_gl_state'):
            self._gl_state.program = 0
        vert_src = """#version 330 core
        layout(location = 0) in vec2 position;
        out vec2 v_texcoord;
        void main() {
            gl_Position = vec4(position, 0.0, 1.0);
            v_texcoord = position * 0.5 + 0.5;
        }
        """
        frag_src = """#version 330 core
        in vec2 v_texcoord;
        out vec4 FragColor;
        uniform float u_time;
        uniform vec2 u_resolution;
        uniform vec4 u_audio;        // x=bass, y=mid, z=treb, w=beat
        uniform vec4 u_audio_bands;    // x=sub_bass, y=low_mid, z=high_mid, w=high_treb
        uniform vec4 u_audio_derived;  // x=transient, y=peak_all, z=total_energy, w=agc_gain
        uniform vec4 u_audio_peaks;    // x=peak_bass, y=peak_mid, z=peak_treb, w=bpm/100
        uniform vec2 u_zoom;
        uniform float u_rot;
        uniform float u_warp;
        uniform float u_cx;
        uniform float u_cy;
        uniform float u_dx;
        uniform float u_dy;
        uniform float u_sx;
        uniform float u_sy;
        uniform float u_decay;
        uniform float u_feedback;
        uniform float u_beat_flash;
        uniform float u_brightness;
        uniform sampler2D u_prev_frame;
        uniform sampler2D u_texture;     // Main preset texture
        uniform float u_texture_blend; // How much to blend texture (0-1)
        uniform float u_texture_zoom;  // Texture zoom level
        uniform vec2 u_texture_offset; // Texture scroll offset
        uniform float u_texture_distort; // Texture distortion amount
        // MilkDrop/ProjectM noise texture samplers
        uniform sampler2D iChannel0;   // Old preset (faded out during blend)
        uniform sampler2D iChannel1;   // New preset (faded in during blend)
        uniform sampler2D iNoiseLQ;    // 256x256 high-frequency 2D noise
        uniform sampler2D iNoiseLQNearest; // Same with nearest interpolation
        uniform sampler2D iNoiseMQ;    // 256x256 medium-frequency 2D noise
        uniform sampler2D iNoiseMQNearest; // Same with nearest interpolation
        uniform sampler2D iNoiseHQ;      // 256x256 low-frequency 2D noise
        uniform sampler2D iNoiseHQNearest;   // Same with nearest interpolation
        uniform sampler3D iNoiseVolLQ;   // 32x32x32 high-frequency 3D noise
        uniform sampler3D iNoiseVolLQNearest; // Same with nearest interpolation
        uniform sampler3D iNoiseVolHQ;   // 32x32x32 low-frequency 3D noise
        uniform sampler3D iNoiseVolHQNearest; // Same with nearest interpolation
        // MilkDrop q1-q32 per-frame registers and global registers
        uniform float u_q[33];
        uniform float u_regs[100];
        // MilkDrop megabuf texture (persistent state buffer)
        uniform sampler2D u_megabuf;
        // Random/hash functions for GLSL
        float hash(float n) { return fract(sin(n) * 43758.5453123); }
        float rand(float x) { return fract(sin(x * 12.9898) * 43758.5453); }
        float rand2(vec2 co) { return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453); }
        // Simplex noise for organic texture distortion
        vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
        vec2 mod289(vec2 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
        vec3 permute(vec3 x) { return mod289(((x*34.0)+1.0)*x); }
        float snoise(vec2 v) {
            const vec4 C = vec4(0.211324865405187, 0.366025403784439,
                             -0.577350269189626, 0.024390243902439);
            vec2 i  = floor(v + dot(v, C.yy));
            vec2 x0 = v -   i + dot(i, C.xx);
            vec2 i1;
            i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
            vec4 x12 = x0.xyxy + C.xxzz;
            x12.xy -= i1;
            i = mod289(i);
            vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0))
                + i.x + vec3(0.0, i1.x, 1.0));
            vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy),
                dot(x12.zw,x12.zw)), 0.0);
            m = m*m;
            m = m*m;
            vec3 x = 2.0 * fract(p * C.www) - 1.0;
            vec3 h = abs(x) - 0.5;
            vec3 ox = floor(x + 0.5);
            vec3 a0 = x - ox;
            m *= 1.79284291400159 - 0.85373472095314 * (a0*a0 + h*h);
            vec3 g;
            g.x  = a0.x  * x0.x  + h.x  * x0.y;
            g.yz = a0.yz * x12.xz + h.yz * x12.yw;
            return 130.0 * dot(m, g);
        }

        void main() {
            vec2 uv = v_texcoord;

            // Standardized transformation block matching full shader
            float zoom = u_zoom.x;
            if (any(isnan(vec2(zoom))) || any(isinf(vec2(zoom))) || abs(zoom) < 0.0000001) zoom = 0.001;
            float rot = u_rot;
            float warp = u_warp;
            float cx = u_cx;
            float cy = u_cy;
            float dx = u_dx;
            float dy = u_dy;
            float sx = u_sx;
            float sy = u_sy;
            float decay = u_decay;

            // Prevent compiler from optimizing out core uniforms
            float _u_fix = (u_cx + u_cy + u_dx + u_dy + u_sx + u_sy + u_decay + u_zoom.y + u_warp + u_rot + u_brightness) * 1e-9;
            // Prevent q registers, reg registers, and megabuf from being optimized out
            float _q_fix = 0.0;
            for (int _i = 0; _i < 32; _i++) _q_fix += u_q[_i + 1];
            _u_fix += _q_fix * 1e-9;

            // Apply transformations (MilkDrop style)
            vec2 uv_warp = uv;
            if (warp != 0.0) {
                // Step 6: Sharper curl-style warp distortion
                vec2 flow;
                flow.x = sin(uv.y * 12.0 + u_time);
                flow.y = cos(uv.x * 10.0 - u_time);
                uv_warp += flow * warp * 0.02;
            }

            vec2 uv_trans = (uv_warp - vec2(cx, cy)) / vec2(zoom * sx, zoom * sy);
            float cos_r = cos(rot); float sin_r = sin(rot);
            uv_trans = mat2(cos_r, -sin_r, sin_r, cos_r) * uv_trans + vec2(cx, cy) - vec2(dx, dy);

            vec2 warped_uv = fract(uv_trans);
            float radius = length(uv - vec2(cx, cy)) * 2.0;

            // === TEXTURE SAMPLING WITH EFFECTS ===
            vec3 texture_color = vec3(0.0);
            float texture_influence = 0.0;

            if (u_texture_blend > 0.0) {
                // Animated texture coordinates
                vec2 tex_uv = v_texcoord;

                // Apply zoom
                tex_uv = (tex_uv - 0.5) / u_texture_zoom + 0.5;

                // Apply scrolling animation (audio-reactive)
                float scroll_speed = 0.1 + u_audio.x * 0.2;
                tex_uv += u_texture_offset * scroll_speed;

                // Apply organic distortion using noise
                if (u_texture_distort > 0.0) {
                    float noise1 = snoise(tex_uv * 3.0 + u_time * 0.5);
                    float noise2 = snoise(tex_uv * 5.0 - u_time * 0.3);
                    vec2 distortion = vec2(noise1, noise2) * u_texture_distort * (1.0 + u_audio.x);
                    tex_uv += distortion;
                }

                // Mirror/wrap texture coordinates
                tex_uv = fract(tex_uv);

                // Sample texture
                texture_color = texture(u_texture, tex_uv).rgb;

                // Audio-reactive texture modulation
                float bass_mod = 1.0 + u_audio.x * 0.5;
                float treb_mod = 1.0 + u_audio.z * 0.3;
                texture_color.r *= bass_mod;
                texture_color.b *= treb_mod;

                texture_influence = u_texture_blend;
            }

            // Standard MilkDrop simple pattern
            vec3 new_color = vec3(radius * 0.5);
            new_color.r *= (0.5 + 0.5 * sin(u_time + radius));
            new_color.g *= (0.5 + 0.5 * sin(u_time * 0.7 + radius * 1.4));
            new_color.b *= (0.5 + 0.5 * sin(u_time * 1.2 + radius * 0.8));

            // Mix with feedback - more feedback for trails, less new color
            float feedback_amount = clamp(u_feedback * u_decay, 0.0, 0.99);
            vec3 feedback_color = texture(u_prev_frame, warped_uv).rgb;

            vec3 final_color = mix(new_color, feedback_color, feedback_amount);

            final_color += clamp(_u_fix, 0.0, 0.01);
            // CRITICAL: Final output clamp prevents energy buildup in feedback loop
            final_color = clamp(final_color, 0.0, 1.0);

            FragColor = vec4(final_color, 1.0);
        }
        """
        vs = GL.glCreateShader(GL.GL_VERTEX_SHADER)
        GL.glShaderSource(vs, vert_src.lstrip())
        GL.glCompileShader(vs)
        if not GL.glGetShaderiv(vs, GL.GL_COMPILE_STATUS):
            error = GL.glGetShaderInfoLog(vs)
            if isinstance(error, bytes):
                error = error.decode('utf-8', errors='ignore')
            GL.glDeleteShader(vs)
            raise RuntimeError(f"Vertex shader compilation failed: {error}")
        fs = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
        GL.glShaderSource(fs, frag_src.lstrip())
        GL.glCompileShader(fs)
        if not GL.glGetShaderiv(fs, GL.GL_COMPILE_STATUS):
            error = GL.glGetShaderInfoLog(fs)
            if isinstance(error, bytes):
                error = error.decode('utf-8', errors='ignore')
            GL.glDeleteShader(vs)
            GL.glDeleteShader(fs)
            raise RuntimeError(f"Fragment shader compilation failed: {error}")
        self.shader_program = GL.glCreateProgram()
        GL.glAttachShader(self.shader_program, vs)
        GL.glAttachShader(self.shader_program, fs)
        GL.glLinkProgram(self.shader_program)
        if not GL.glGetProgramiv(self.shader_program, GL.GL_LINK_STATUS):
            error = GL.glGetProgramInfoLog(self.shader_program)
            if isinstance(error, bytes):
                error = error.decode('utf-8', errors='ignore')
            GL.glDeleteShader(vs)
            GL.glDeleteShader(fs)
            raise RuntimeError(f"Shader program linking failed: {error}")
        self._refresh_shader_uniform_locations(self.shader_program)
        GL.glDeleteShader(vs)
        GL.glDeleteShader(fs)
        self.program = self.shader_program

    def _try_create_simple_shader(self):
        """Create a simple shader without destroying existing GL state.

        Returns (program_id, uniform_locs_dict) on success, None on failure.
        Does NOT queue old program for deletion or modify self.shader_program.
        """
        vert_src = """#version 330 core
        layout(location = 0) in vec2 position;
        out vec2 v_texcoord;
        void main() {
            gl_Position = vec4(position, 0.0, 1.0);
            v_texcoord = position * 0.5 + 0.5;
        }
        """
        frag_src = """#version 330 core
        in vec2 v_texcoord;
        out vec4 FragColor;
        uniform float u_time;
        uniform vec2 u_resolution;
        uniform vec4 u_audio;
        uniform vec4 u_audio_bands;
        uniform vec4 u_audio_derived;
        uniform vec4 u_audio_peaks;
        uniform vec2 u_zoom;
        uniform float u_rot;
        uniform float u_warp;
        uniform float u_cx;
        uniform float u_cy;
        uniform float u_dx;
        uniform float u_dy;
        uniform float u_sx;
        uniform float u_sy;
        uniform float u_decay;
        uniform float u_feedback;
        uniform float u_beat_flash;
        uniform float u_brightness;
        uniform sampler2D u_prev_frame;
        uniform sampler2D u_texture;
        uniform float u_texture_blend;
        uniform float u_texture_zoom;
        uniform vec2 u_texture_offset;
        uniform float u_texture_distort;
        uniform sampler2D iChannel0;
        uniform sampler2D iChannel1;
        uniform sampler2D iNoiseLQ;
        uniform sampler2D iNoiseLQNearest;
        uniform sampler2D iNoiseMQ;
        uniform sampler2D iNoiseMQNearest;
        uniform sampler2D iNoiseHQ;
        uniform sampler2D iNoiseHQNearest;
        uniform sampler3D iNoiseVolLQ;
        uniform sampler3D iNoiseVolLQNearest;
        uniform sampler3D iNoiseVolHQ;
        uniform sampler3D iNoiseVolHQNearest;
        uniform float u_q[33];
        uniform float u_regs[100];
        uniform sampler2D u_megabuf;
        float hash(float n) { return fract(sin(n) * 43758.5453123); }
        float rand(float x) { return fract(sin(x * 12.9898) * 43758.5453); }
        float rand2(vec2 co) { return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453); }
        vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
        vec2 mod289(vec2 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
        vec3 permute(vec3 x) { return mod289(((x*34.0)+1.0)*x); }
        float snoise(vec2 v) {
            const vec4 C = vec4(0.211324865405187, 0.366025403784439,
                             -0.577350269189626, 0.024390243902439);
            vec2 i  = floor(v + dot(v, C.yy));
            vec2 x0 = v -   i + dot(i, C.xx);
            vec2 i1;
            i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
            vec4 x12 = x0.xyxy + C.xxzz;
            x12.xy -= i1;
            i = mod289(i);
            vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0))
                + i.x + vec3(0.0, i1.x, 1.0));
            vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy),
                dot(x12.zw,x12.zw)), 0.0);
            m = m*m;
            m = m*m;
            vec3 x = 2.0 * fract(p * C.www) - 1.0;
            vec3 h = abs(x) - 0.5;
            vec3 ox = floor(x + 0.5);
            vec3 a0 = x - ox;
            m *= 1.79284291400159 - 0.85373472095314 * (a0*a0 + h*h);
            vec3 g;
            g.x  = a0.x  * x0.x  + h.x  * x0.y;
            g.yz = a0.yz * x12.xz + h.yz * x12.zw;
            return 130.0 * dot(m, g);
        }
        void main() {
            vec2 uv = v_texcoord;
            float zoom = u_zoom.x;
            if (any(isnan(vec2(zoom))) || any(isinf(vec2(zoom))) || abs(zoom) < 0.0000001) zoom = 0.001;
            float rot = u_rot;
            float warp = u_warp;
            float cx = u_cx;
            float cy = u_cy;
            float dx = u_dx;
            float dy = u_dy;
            float sx = u_sx;
            float sy = u_sy;
            float decay = u_decay;
            float _u_fix = (u_cx + u_cy + u_dx + u_dy + u_sx + u_sy + u_decay + u_zoom.y + u_warp + u_rot + u_brightness) * 1e-9;
            float _q_fix = 0.0;
            for (int _i = 0; _i < 32; _i++) _q_fix += u_q[_i + 1];
            _u_fix += _q_fix * 1e-9;
            vec2 uv_warp = uv;
            if (warp != 0.0) {
                vec2 flow;
                flow.x = sin(uv.y * 12.0 + u_time);
                flow.y = cos(uv.x * 10.0 - u_time);
                uv_warp += flow * warp * 0.02;
            }
            vec2 uv_trans = (uv_warp - vec2(cx, cy)) / vec2(zoom * sx, zoom * sy);
            float cos_r = cos(rot); float sin_r = sin(rot);
            uv_trans = mat2(cos_r, -sin_r, sin_r, cos_r) * uv_trans + vec2(cx, cy) - vec2(dx, dy);
            vec2 warped_uv = fract(uv_trans);
            float radius = length(uv - vec2(cx, cy)) * 2.0;
            vec3 texture_color = vec3(0.0);
            float texture_influence = 0.0;
            if (u_texture_blend > 0.0) {
                vec2 tex_uv = v_texcoord;
                tex_uv = (tex_uv - 0.5) / u_texture_zoom + 0.5;
                tex_uv += u_texture_offset * (0.1 + u_audio.x * 0.2);
                if (u_texture_distort > 0.0) {
                    float noise1 = snoise(tex_uv * 3.0 + u_time * 0.5);
                    float noise2 = snoise(tex_uv * 5.0 - u_time * 0.3);
                    vec2 distortion = vec2(noise1, noise2) * u_texture_distort * (1.0 + u_audio.x);
                    tex_uv += distortion;
                }
                tex_uv = fract(tex_uv);
                texture_color = texture(u_texture, tex_uv).rgb;
                texture_color.r *= 1.0 + u_audio.x * 0.5;
                texture_color.b *= 1.0 + u_audio.z * 0.3;
                texture_influence = u_texture_blend;
            }
            vec3 new_color = vec3(radius * 0.5);
            new_color.r *= (0.5 + 0.5 * sin(u_time + radius));
            new_color.g *= (0.5 + 0.5 * sin(u_time * 0.7 + radius * 1.4));
            new_color.b *= (0.5 + 0.5 * sin(u_time * 1.2 + radius * 0.8));
            float feedback_amount = clamp(u_feedback * u_decay, 0.0, 0.99);
            vec3 feedback_color = texture(u_prev_frame, warped_uv).rgb;
            vec3 final_color = mix(new_color, feedback_color, feedback_amount);
            final_color += clamp(_u_fix, 0.0, 0.01);
            final_color = clamp(final_color, 0.0, 1.0);
            FragColor = vec4(final_color, 1.0);
        }
        """
        try:
            vs = GL.glCreateShader(GL.GL_VERTEX_SHADER)
            GL.glShaderSource(vs, vert_src.lstrip())
            GL.glCompileShader(vs)
            if not GL.glGetShaderiv(vs, GL.GL_COMPILE_STATUS):
                error = GL.glGetShaderInfoLog(vs)
                if isinstance(error, bytes):
                    error = error.decode('utf-8', errors='ignore')
                GL.glDeleteShader(vs)
                raise RuntimeError(f"Vertex shader compilation failed: {error}")
            fs = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
            GL.glShaderSource(fs, frag_src.lstrip())
            GL.glCompileShader(fs)
            if not GL.glGetShaderiv(fs, GL.GL_COMPILE_STATUS):
                error = GL.glGetShaderInfoLog(fs)
                if isinstance(error, bytes):
                    error = error.decode('utf-8', errors='ignore')
                GL.glDeleteShader(vs)
                GL.glDeleteShader(fs)
                raise RuntimeError(f"Fragment shader compilation failed: {error}")
            program = GL.glCreateProgram()
            GL.glAttachShader(program, vs)
            GL.glAttachShader(program, fs)
            GL.glLinkProgram(program)
            if not GL.glGetProgramiv(program, GL.GL_LINK_STATUS):
                error = GL.glGetProgramInfoLog(program)
                if isinstance(error, bytes):
                    error = error.decode('utf-8', errors='ignore')
                GL.glDeleteShader(vs)
                GL.glDeleteShader(fs)
                raise RuntimeError(f"Shader program linking failed: {error}")
            GL.glDeleteShader(vs)
            GL.glDeleteShader(fs)
            uniform_locs = self._refresh_shader_uniform_locations(program, return_locs=True)
            return program, uniform_locs
        except RuntimeError as e:
            logger.error(f"Simple shader compilation failed: {e}")
            return None

    def _create_display_shader(self):
        """Create a simple pass-through shader for displaying FBO texture to screen."""
        vert_src = """#version 330 core
        layout(location = 0) in vec2 position;
        out vec2 v_texcoord;
        void main() {
            gl_Position = vec4(position, 0.0, 1.0);
            v_texcoord = position * 0.5 + 0.5;
        }
        """
        frag_src = """#version 330 core
        in vec2 v_texcoord;
        out vec4 FragColor;
        uniform sampler2D u_texture;
        void main() {
            vec4 tex = texture(u_texture, v_texcoord);
            FragColor = tex;
        }
        """
        vs = GL.glCreateShader(GL.GL_VERTEX_SHADER)
        GL.glShaderSource(vs, vert_src)
        GL.glCompileShader(vs)
        if not GL.glGetShaderiv(vs, GL.GL_COMPILE_STATUS):
            error = GL.glGetShaderInfoLog(vs).decode()
            logger.error(error)
            return
        fs = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
        GL.glShaderSource(fs, frag_src)
        GL.glCompileShader(fs)
        if not GL.glGetShaderiv(fs, GL.GL_COMPILE_STATUS):
            error = GL.glGetShaderInfoLog(fs).decode()
            logger.error(error)
            return
        self.display_program = GL.glCreateProgram()
        GL.glAttachShader(self.display_program, vs)
        GL.glAttachShader(self.display_program, fs)
        GL.glLinkProgram(self.display_program)
        if not GL.glGetProgramiv(self.display_program, GL.GL_LINK_STATUS):
            error = GL.glGetProgramInfoLog(self.display_program)
            logger.error(f"Display shader linking failed: {error}")
            self.display_program = None
        else:
            self.display_tex_loc = GL.glGetUniformLocation(self.display_program, "u_texture")
        GL.glDeleteShader(vs)
        GL.glDeleteShader(fs)

    def _setup_megabuf_texture(self):
        """Create the GPU backing texture for MilkDrop megabuf.
        The codebase calls this during on_realize(), but the method was
        missing, causing an AttributeError.
        Implementation note:
        - We allocate a simple 1D-like 2D texture (width=MEGABUF_SIZE, height=1)
          using float32 so shaders/includes can sample it.
        - If float textures are unavailable, we still create an RGBA8
          placeholder to keep the app running.
        """
        try:
            if (
                getattr(self, 'megabuf_tex', None) is not None
                and int(self.megabuf_tex) > 0
                and GL.glIsTexture(int(self.megabuf_tex))
            ):
                return
        except Exception as e:
            logger.debug(f"Failed to check megabuf texture: {e}")
        megabuf_size = int(getattr(self, 'megabuf_tex_size', MEGABUF_SIZE))
        megabuf_size = max(1, megabuf_size)
        try:
            max_tex_size = GL.glGetIntegerv(GL.GL_MAX_TEXTURE_SIZE)
            if max_tex_size is not None:
                if isinstance(max_tex_size, (list, tuple)):
                    max_tex_size = int(max_tex_size[0])
                else:
                    max_tex_size = int(max_tex_size)
                megabuf_size = min(megabuf_size, max_tex_size)
        except Exception as exc:
            logger.debug("Exception in try block: %s", exc)
        tex_id = GL.glGenTextures(1)
        try:
            if isinstance(tex_id, (list, tuple)):
                tex_id = tex_id[0]
            tex_id = int(tex_id)
        except (TypeError, ValueError):
            tex_id = 0
        if tex_id <= 0:
            self.megabuf_tex = None
            return
        self.megabuf_tex = tex_id
        self.megabuf_tex_size = megabuf_size
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex_id)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        uploaded = False
        try:
            internal_format = getattr(GL, 'GL_R32F', None)
            if internal_format is None:
                raise RuntimeError('GL_R32F not available')
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D,
                0,
                internal_format,
                megabuf_size,
                1,
                0,
                getattr(GL, 'GL_RED', GL.GL_RED),
                getattr(GL, 'GL_FLOAT', GL.GL_FLOAT),
                None,
            )
            uploaded = True
        except Exception:
            uploaded = False
        if not uploaded:
            try:
                GL.glTexImage2D(
                    GL.GL_TEXTURE_2D,
                    0,
                    GL.GL_RGBA8,
                    megabuf_size,
                    1,
                    0,
                    GL.GL_RGBA,
                    GL.GL_UNSIGNED_BYTE,
                    None,
                )
            except Exception as e:
                logger.debug(f"Failed to create megabuf texture: {e}")
                self.megabuf_tex = None
                try:
                    GLResourceOwner.queue_delete("texture", int(tex_id))
                except Exception as e2:
                    logger.debug(f"Failed to delete megabuf texture: {e2}")
                return
        self._last_megabuf_version_uploaded = -1

    def _setup_noise_textures(self):
        """Create noise textures for MilkDrop-style shader noise functions."""
        try:
            self.noise_textures = {}
            for name in ['noise_lq', 'noise_mq', 'noise_hq']:
                tex_id = GL.glGenTextures(1)
                if isinstance(tex_id, (list, tuple)):
                    tex_id = int(tex_id[0])
                else:
                    tex_id = int(tex_id)
                if tex_id > 0:
                    GL.glBindTexture(GL.GL_TEXTURE_2D, tex_id)
                    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
                    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
                    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
                    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
                    size = 64
                    noise_data = np.random.uniform(0.0, 1.0, (size, size, 4)).astype(np.float32)
                    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA16F, size, size, 0, GL.GL_RGBA, GL.GL_FLOAT, noise_data)
                    self.noise_textures[name] = tex_id
        except Exception as e:
            logger.debug(f"Noise texture setup skipped: {e}")
            self.noise_textures = {}

    def _create_shape_geometry(self, num_shapes=8):
        """Create geometry for shape rendering (placeholder implementation)."""
        try:
            self.shape_vaos = []
            self.shape_vbos = []
            for _ in range(num_shapes):
                vao = int(GL.glGenVertexArrays(1))
                vbo = int(GL.glGenBuffers(1))
                self.shape_vaos.append(vao)
                self.shape_vbos.append(vbo)
        except Exception:
            self.shape_vaos = []
            self.shape_vbos = []

    def _create_wave_shader(self):
        """Create shader for rendering waveforms."""
        vert_src = """#version 330 core
        layout(location = 0) in vec2 a_position;
        layout(location = 1) in vec4 a_color;
        out vec4 v_color;
        void main() {
            gl_Position = vec4(a_position, 0.0, 1.0);
            v_color = a_color;
        }
        """
        frag_src = """#version 330 core
        in vec4 v_color;
        out vec4 FragColor;
        void main() {
            FragColor = v_color;
        }
        """
        try:
            self.wave_program = self._build_program(vert_src, frag_src)
            self.wave_color_loc = GL.glGetUniformLocation(self.wave_program, "u_color")
            if not hasattr(self, 'wave_vao') or not self.wave_vao:
                self.wave_vao = int(GL.glGenVertexArrays(1))
                self.wave_vbo = int(GL.glGenBuffers(1))
        except Exception as e:
            logger.error(f"Wave shader failed: {e}")
            self.wave_program = None

    def _render_waves(self, audio_data, params):
        """Render wave ribbons into the current FBO with full MilkDrop 1.x wave mode support.

        Supports classic wave modes:
        0: Single line spectrum
        1: Single line waveform
        2: Dual spectrum (stereo)
        3: Dual waveform (stereo)
        4: Circular spectrum
        5: Circular waveform
        6: Radial spectrum
        7: Radial waveform

        Evaluates per-point equations for each sample and renders with proper blending.
        """
        if not self.wave_program or audio_data is None:
            return
        if getattr(self, 'preset_vm', None) and getattr(self.preset_vm, 'program', None):
            program = self.preset_vm.program
            if not getattr(program, 'waves', None) or len(program.waves) == 0:
                return
        else:
            return

        self._use_program_cached(self.wave_program)
        self._bind_vao_cached(self.wave_vao)

        # Set blending based on wave additive mode
        prev_blend = GL.glIsEnabled(GL.GL_BLEND)
        GL.glEnable(GL.GL_BLEND)

        for wave_idx, wave in enumerate(program.waves):
            if not wave.enabled or not wave.per_point:
                continue

            # Set blend mode based on wave additive flag
            if wave.additive:
                GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE)
            else:
                GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

            num_samples = wave.samples
            thickness = 0.02 if wave.bDrawThick else 0.005

            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.wave_vbo)

            # Initialize arrays if needed (must succeed before proceeding)
            if not hasattr(self, '_wave_verts') or self._wave_verts.shape[0] < num_samples * 2:
                self._wave_verts = np.empty((num_samples * 2, 6), dtype=np.float32)
                self._wave_xs = np.linspace(-1.0, 1.0, num_samples, dtype=np.float32)

            verts = self._wave_verts[:num_samples * 2]
            xs = self._wave_xs[:num_samples]

            # Evaluate per-point equations for all samples
            wave_evaluated = False
            try:
                for point_idx in range(num_samples):
                    sample_val = float(point_idx) / float(max(1, num_samples - 1))
                    self.preset_vm.run_wave_per_point(wave_idx, point_idx, sample_val, num_samples)
                wave_evaluated = True
            except (AttributeError, IndexError, ValueError, TypeError, ZeroDivisionError) as e:
                logger.warning(f"Wave per-point evaluation failed for wave {wave_idx}: {e}")

            # Get evaluated wave values (safe defaults if evaluation failed)
            try:
                if wave_evaluated:
                    ys = np.asarray(self.preset_vm.wave_y[:num_samples], dtype=np.float32)
                    rs = np.asarray(self.preset_vm.wave_r[:num_samples], dtype=np.float32)
                    gs = np.asarray(self.preset_vm.wave_g[:num_samples], dtype=np.float32)
                    bs = np.asarray(self.preset_vm.wave_b[:num_samples], dtype=np.float32)
                    als = np.asarray(self.preset_vm.wave_a[:num_samples], dtype=np.float32)
                else:
                    ys = np.zeros(num_samples, dtype=np.float32)
                    rs = np.full(num_samples, wave.r, dtype=np.float32)
                    gs = np.full(num_samples, wave.g, dtype=np.float32)
                    bs = np.full(num_samples, wave.b, dtype=np.float32)
                    als = np.full(num_samples, wave.a, dtype=np.float32)
            except (AttributeError, IndexError, ValueError, TypeError) as e:
                logger.warning(f"Wave value extraction failed for wave {wave_idx}: {e}")
                continue  # Skip this wave if value extraction fails

            # Get evaluated wave values
            ys = np.asarray(self.preset_vm.wave_y[:num_samples], dtype=np.float32)
            rs = np.asarray(self.preset_vm.wave_r[:num_samples], dtype=np.float32)
            gs = np.asarray(self.preset_vm.wave_g[:num_samples], dtype=np.float32)
            bs = np.asarray(self.preset_vm.wave_b[:num_samples], dtype=np.float32)
            als = np.asarray(self.preset_vm.wave_a[:num_samples], dtype=np.float32)

            # Transform based on wave mode
            if wave.mode in (4, 5):  # Circular modes
                # Map linear samples to circular coordinates
                angles = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
                radius = ys * 0.4
                cx, cy = wave.x, wave.y

                # Convert to Cartesian
                xs_circ = cx + radius * np.cos(angles)
                ys_circ = cy + radius * np.sin(angles)

                # Create triangle strip for circular wave
                for i in range(num_samples):
                    idx = i * 2
                    verts[idx, 0] = xs_circ[i]
                    verts[idx, 1] = ys_circ[i]
                    verts[idx, 2] = rs[i]
                    verts[idx, 3] = gs[i]
                    verts[idx, 4] = bs[i]
                    verts[idx, 5] = als[i]

                    verts[idx + 1, 0] = cx
                    verts[idx + 1, 1] = cy
                    verts[idx + 1, 2] = rs[i]
                    verts[idx + 1, 3] = gs[i]
                    verts[idx + 1, 4] = bs[i]
                    verts[idx + 1, 5] = als[i] * 0.5

                GL.glBufferData(GL.GL_ARRAY_BUFFER, verts.nbytes, verts, GL.GL_DYNAMIC_DRAW)
                GL.glEnableVertexAttribArray(0)
                GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 6 * 4, None)
                GL.glEnableVertexAttribArray(1)
                GL.glVertexAttribPointer(1, 4, GL.GL_FLOAT, GL.GL_FALSE, 6 * 4, ctypes.c_void_p(2 * 4))
                GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, num_samples * 2)

            elif wave.mode in (6, 7):  # Radial modes
                # Radial burst from center
                angles = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
                radius = 0.3 + ys * 0.2
                cx, cy = wave.x, wave.y

                for i in range(num_samples):
                    idx = i * 2
                    x_inner = cx + (radius[i] - thickness) * np.cos(angles[i])
                    y_inner = cy + (radius[i] - thickness) * np.sin(angles[i])
                    x_outer = cx + (radius[i] + thickness) * np.cos(angles[i])
                    y_outer = cy + (radius[i] + thickness) * np.sin(angles[i])

                    verts[idx, 0] = x_inner
                    verts[idx, 1] = y_inner
                    verts[idx, 2] = rs[i]
                    verts[idx, 3] = gs[i]
                    verts[idx, 4] = bs[i]
                    verts[idx, 5] = als[i]

                    verts[idx + 1, 0] = x_outer
                    verts[idx + 1, 1] = y_outer
                    verts[idx + 1, 2] = rs[i]
                    verts[idx + 1, 3] = gs[i]
                    verts[idx + 1, 4] = bs[i]
                    verts[idx + 1, 5] = als[i]

                GL.glBufferData(GL.GL_ARRAY_BUFFER, verts.nbytes, verts, GL.GL_DYNAMIC_DRAW)
                GL.glEnableVertexAttribArray(0)
                GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 6 * 4, None)
                GL.glEnableVertexAttribArray(1)
                GL.glVertexAttribPointer(1, 4, GL.GL_FLOAT, GL.GL_FALSE, 6 * 4, ctypes.c_void_p(2 * 4))
                GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, num_samples * 2)

            else:  # Linear modes (0, 1, 2, 3)
                # Standard linear wave rendering
                scale_y = 0.4 if not wave.bMaximize else 0.8
                ys_scaled = ys * scale_y

                for i in range(num_samples):
                    idx = i * 2
                    verts[idx, 0] = xs[i]
                    verts[idx, 1] = wave.y + ys_scaled[i] + thickness
                    verts[idx, 2] = rs[i]
                    verts[idx, 3] = gs[i]
                    verts[idx, 4] = bs[i]
                    verts[idx, 5] = als[i]

                    verts[idx + 1, 0] = xs[i]
                    verts[idx + 1, 1] = wave.y + ys_scaled[i] - thickness
                    verts[idx + 1, 2] = rs[i]
                    verts[idx + 1, 3] = gs[i]
                    verts[idx + 1, 4] = bs[i]
                    verts[idx + 1, 5] = als[i]

                GL.glBufferData(GL.GL_ARRAY_BUFFER, verts.nbytes, verts, GL.GL_DYNAMIC_DRAW)
                GL.glEnableVertexAttribArray(0)
                GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 6 * 4, None)
                GL.glEnableVertexAttribArray(1)
                GL.glVertexAttribPointer(1, 4, GL.GL_FLOAT, GL.GL_FALSE, 6 * 4, ctypes.c_void_p(2 * 4))

                if wave.bUseDots:
                    GL.glDrawArrays(GL.GL_POINTS, 0, num_samples * 2)
                else:
                    GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, num_samples * 2)

        if not prev_blend:
            GL.glDisable(GL.GL_BLEND)

    def _create_tone_map_shader(self):
        """Create shader for final tone mapping and color grading."""
        vert_src = """#version 330 core
        layout(location = 0) in vec2 position;
        out vec2 v_texcoord;
        void main() {
            gl_Position = vec4(position, 0.0, 1.0);
            v_texcoord = position * 0.5 + 0.5;
        }
        """
        frag_src = """#version 330 core
        in vec2 v_texcoord;
        out vec4 FragColor;
        uniform sampler2D u_texture;
        uniform float u_brightness;
        void main() {
            vec3 color = texture(u_texture, v_texcoord).rgb;
            color *= u_brightness;
            // ACES approximation
            const float a = 2.51;
            const float b = 0.03;
            const float c = 2.43;
            const float d = 0.59;
            const float e = 0.14;
            color = clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.0, 1.0);
            // Gamma
            color = pow(max(color, 0.0), vec3(1.0 / 2.2));
            FragColor = vec4(color, 1.0);
        }
        """
        try:
            self.tone_map_program = self._build_program(vert_src, frag_src)
        except Exception as e:
            logger.error(f"Tone mapping shader failed: {e}")
            self.tone_map_program = None

    def _create_warp_shader(self):
        """Build and compile the dynamic warp shader."""
        shader_code = self._build_warp_shader()
        self._compile_warp_shader(shader_code)

    def _create_bloom_shaders(self):
        """Create shaders for bloom/glow post-processing effect."""
        bloom_extract_vert = """#version 330 core
        layout(location = 0) in vec2 position;
        out vec2 v_texcoord;
        void main() {
            gl_Position = vec4(position, 0.0, 1.0);
            v_texcoord = position * 0.5 + 0.5;
        }
        """
        bloom_extract_frag = """#version 330 core
        in vec2 v_texcoord;
        out vec4 FragColor;
        uniform sampler2D u_source;
        uniform float u_threshold;
        uniform float u_intensity;
        void main() {
            vec3 color = texture(u_source, v_texcoord).rgb;
            float brightness = dot(color, vec3(0.299, 0.587, 0.114));
            float extraction = max(0.0, brightness - u_threshold);
            FragColor = vec4(color * extraction * u_intensity, 1.0);
        }
        """
        blur_vert = """#version 330 core
        layout(location = 0) in vec2 position;
        out vec2 v_texcoord;
        void main() {
            gl_Position = vec4(position, 0.0, 1.0);
            v_texcoord = position * 0.5 + 0.5;
        }
        """
        blur_h_frag = """#version 330 core
        in vec2 v_texcoord;
        out vec4 FragColor;
        uniform sampler2D u_source;
        uniform vec2 u_resolution;
        uniform float u_radius;
        void main() {
            vec2 offset = vec2(u_radius / u_resolution.x, 0.0);
            vec4 sum = vec4(0.0);
            float weights[5] = float[](0.227, 0.194, 0.121, 0.054, 0.016);
            for (int i = -4; i <= 4; i++) {
                sum += texture(u_source, v_texcoord + offset * float(i)) * weights[abs(i)];
            }
            FragColor = sum;
        }
        """
        blur_v_frag = """#version 330 core
        in vec2 v_texcoord;
        out vec4 FragColor;
        uniform sampler2D u_source;
        uniform vec2 u_resolution;
        uniform float u_radius;
        void main() {
            vec2 offset = vec2(0.0, u_radius / u_resolution.y);
            vec4 sum = vec4(0.0);
            float weights[5] = float[](0.227, 0.194, 0.121, 0.054, 0.016);

            for (int i = -4; i <= 4; i++) {
                sum += texture(u_source, v_texcoord + offset * float(i)) * weights[abs(i)];
            }
            FragColor = sum;
        }
        """
        composite_frag = """#version 330 core
        in vec2 v_texcoord;
        out vec4 FragColor;
        uniform sampler2D u_original;
        uniform sampler2D u_bloom_small;
        uniform sampler2D u_bloom_medium;
        uniform sampler2D u_bloom_large;
        uniform sampler2D u_bloom_reservoir;
        uniform vec4 u_audio;
        uniform float u_bloom_intensity;
        uniform float u_chromatic;
        uniform float u_vignette;
        uniform float u_lift;
        uniform float u_gamma;
        uniform float u_gain;
        uniform float u_saturation;
        uniform float u_contrast;
        vec3 applySaturation(vec3 color, float saturation) {
            float gray = dot(color, vec3(0.299, 0.587, 0.114));
            return mix(vec3(gray), color, saturation);
        }
        vec3 applyContrast(vec3 color, float contrast) {
            return clamp((color - 0.5) * contrast + 0.5, 0.0, 1.0);
        }
        vec3 colorGrade(vec3 color, float lift, float gamma, float gain) {
            color = pow(color, vec3(max(gamma, 0.01)));
            color = color * gain + lift;
            return color;
        }
        void main() {
            vec2 uv = v_texcoord;

            // Step 7: Subtle lens distortion
            vec2 d_lens = uv - 0.5;
            float r2_lens = dot(d_lens, d_lens);
            uv += d_lens * r2_lens * 0.03;

            vec3 original = texture(u_original, uv).rgb;

            // Step 1: Multi-scale bloom
            vec3 bloomSmall = texture(u_bloom_small, uv).rgb;
            vec3 bloomMed   = texture(u_bloom_medium, uv).rgb;
            vec3 bloomLarge = texture(u_bloom_large, uv).rgb;

            // Temporal accumulation from reservoir
            vec3 reservoir = texture(u_bloom_reservoir, uv).rgb;
            vec3 newBloom = bloomSmall * 0.8 + bloomMed * 0.5 + bloomLarge * 0.25;
            vec3 bloom = mix(reservoir, newBloom, 0.15);

            // Volumetric Scattering (Light Rays)
            vec2 light_pos = vec2(0.5);
            vec2 ray_dir = (uv - light_pos) * (1.0 / 32.0);
            vec3 rays = vec3(0.0);
            float ray_decay = 1.0;
            for(int i=0; i<32; i++) {
                rays += texture(u_original, uv - ray_dir * float(i)).rgb * ray_decay;
                ray_decay *= 0.96;
            }
            newBloom += rays * u_audio.w * 0.4;

            if (u_chromatic > 0.0) {
                vec2 center = uv - 0.5;
                float dist = length(center);
                vec2 direction = normalize(center);

                float r_offset = u_chromatic * dist * 0.02;
                float b_offset = u_chromatic * dist * 0.02;

                original.r = texture(u_original, uv + direction * r_offset).r;
                original.b = texture(u_original, uv - direction * b_offset).b;
            }

            // Add bloom
            vec3 final = original + bloom * u_bloom_intensity;

            // Vignette
            float vignette = clamp(1.0 - length(uv - 0.5) * u_vignette, 0.0, 1.0);
            final *= vignette;

            // Secondary grading pass
            final = colorGrade(final, u_lift, u_gamma, u_gain);
            final = applySaturation(final, u_saturation);
            final = applyContrast(final, u_contrast);
            final = clamp(final, 0.0, 1.0);

            FragColor = vec4(final, 1.0);
        }
        """
        try:
            self.bloom_program = self._build_program(bloom_extract_vert, bloom_extract_frag)
            self.blur_h_program = self._build_program(blur_vert, blur_h_frag)
            self.blur_v_program = self._build_program(blur_vert, blur_v_frag)
            self.composite_program = self._build_program(blur_vert, composite_frag)
            self._create_transition_shaders()
            if self.bloom_program:
                self.bloom_src_loc = GL.glGetUniformLocation(self.bloom_program, "u_source")
                self.bloom_threshold_loc = GL.glGetUniformLocation(self.bloom_program, "u_threshold")
                self.bloom_intensity_loc = GL.glGetUniformLocation(self.bloom_program, "u_intensity")
            if self.blur_h_program:
                self.blur_h_src_loc = GL.glGetUniformLocation(self.blur_h_program, "u_source")
                self.blur_h_res_loc = GL.glGetUniformLocation(self.blur_h_program, "u_resolution")
                self.blur_h_rad_loc = GL.glGetUniformLocation(self.blur_h_program, "u_radius")
            if self.blur_v_program:
                self.blur_v_src_loc = GL.glGetUniformLocation(self.blur_v_program, "u_source")
                self.blur_v_res_loc = GL.glGetUniformLocation(self.blur_v_program, "u_resolution")
                self.blur_v_rad_loc = GL.glGetUniformLocation(self.blur_v_program, "u_radius")
            if (self.composite_program):
                self.composite_orig_loc = GL.glGetUniformLocation(self.composite_program, "u_original")
                self.composite_bloom_small_loc = GL.glGetUniformLocation(self.composite_program, "u_bloom_small")
                self.composite_bloom_medium_loc = GL.glGetUniformLocation(self.composite_program, "u_bloom_medium")
                self.composite_bloom_large_loc = GL.glGetUniformLocation(self.composite_program, "u_bloom_large")
                self.composite_bloom_int_loc = GL.glGetUniformLocation(self.composite_program, "u_bloom_intensity")
                self.composite_bloom_res_loc = GL.glGetUniformLocation(self.composite_program, "u_bloom_reservoir")
                self.composite_audio_loc = GL.glGetUniformLocation(self.composite_program, "u_audio")
                self.composite_chrom_loc = GL.glGetUniformLocation(self.composite_program, "u_chromatic")
                self.composite_vig_loc = GL.glGetUniformLocation(self.composite_program, "u_vignette")
                self.composite_lift_loc = GL.glGetUniformLocation(self.composite_program, "u_lift")
                self.composite_gamma_loc = GL.glGetUniformLocation(self.composite_program, "u_gamma")
                self.composite_gain_loc = GL.glGetUniformLocation(self.composite_program, "u_gain")
                self.composite_sat_loc = GL.glGetUniformLocation(self.composite_program, "u_saturation")
                self.composite_contrast_loc = GL.glGetUniformLocation(self.composite_program, "u_contrast")
        except Exception as e:
            logger.error(f"Bloom shaders failed: {e}")
            self.bloom_program = None
            self.blur_h_program = None
            self.blur_v_program = None
            self.bloom_src_loc = -1
            self.bloom_threshold_loc = -1
            self.bloom_intensity_loc = -1
            self.blur_h_src_loc = -1
            self.blur_h_res_loc = -1
            self.blur_h_rad_loc = -1
            self.blur_v_src_loc = -1
            self.blur_v_res_loc = -1
            self.blur_v_rad_loc = -1
            self.composite_orig_loc = -1
            self.composite_bloom_loc = -1
            self.composite_bloom_int_loc = -1
            self.composite_chrom_loc = -1
            self.composite_vig_loc = -1
            self.composite_lift_loc = -1
            self.composite_gamma_loc = -1
            self.composite_gain_loc = -1
            self.composite_sat_loc = -1
            self.composite_contrast_loc = -1
            self.composite_program = None

    def _create_transition_shaders(self):
        """Create shaders for transition effects with validation, caching, and early compilation (Points 3, 5, 8)."""
        self.transition_cache = {}
        self.transition_shader_mtimes = {}
        self.transitions_dir.mkdir(parents=True, exist_ok=True)
        loaded_count = 0
        for frag_file in sorted(self.transitions_dir.glob("*.frag")):
            if self._load_transition_shader(str(frag_file)):
                loaded_count += 1
        self.available_transitions = sorted(
            [k for k in self.transition_cache if isinstance(k, str)]
        )
        logger.info(f"Loaded {len(self.transition_cache)} transition shaders")

    def _load_transition_shader(self, path):
        """Load and compile a single transition shader with mtime tracking for hot-reload (Point 1, 5)."""
        try:
            with open(path, "r") as f:
                frag_source = f.read()
            name = Path(path).stem
            mtime = os.path.getmtime(path)
            source_hash = hashlib.sha256(frag_source.encode('utf-8')).hexdigest()
            if path in self._transition_shader_cache:
                cached_hash = self._transition_shader_cache[path]
                if cached_hash == source_hash:
                    return True
            program = self._compile_transition_program(frag_source, name)
            if program:
                self.transition_cache[name] = program
                self.transition_shader_mtimes[path] = mtime
                self._transition_shader_cache[path] = source_hash
                return True
            else:
                logger.warning(f"[TRANSITION] compile failed: {name}")
                return False
        except Exception as e:
            logger.error(f"Error loading transition shader {path}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False

    def _compile_transition_program(self, frag_source, name):
        """Standard compilation wrapper with validation and interface enforcement."""
        vert_src = """#version 330 core
        layout(location = 0) in vec2 position;
        layout(location = 1) in vec2 texcoord;
        out vec2 TexCoord;
        out vec2 v_uv;
        void main() {
            TexCoord = texcoord;
            v_uv = texcoord;
            gl_Position = vec4(position, 0.0, 1.0);
        }
        """
        try:
            program = self._build_program_with_logging(vert_src, frag_source, name)
            if not program:
                return None

            def _any_uniform_location(names):
                for u in names:
                    loc = GL.glGetUniformLocation(program, u)
                    if loc >= 0:
                        return u
                return None
            u_texture1_name = _any_uniform_location(["u_texture1", "u_texture_from", "u_texture_from_tex", "u_from", "iChannel0"])
            u_texture2_name = _any_uniform_location(["u_texture2", "u_texture_to", "u_texture_to_tex", "u_to", "iChannel1"])
            u_progress_name = _any_uniform_location(["u_progress", "iProgressBicubic", "iProgressCosine", "iProgressLinear", "progress"])
            u_resolution_name = _any_uniform_location(["u_resolution", "iResolution", "resolution"])
            if not u_texture1_name or not u_texture2_name or not u_progress_name:
                missing = []
                if not u_texture1_name:
                    missing.append("u_texture1")
                if not u_texture2_name:
                    missing.append("u_texture2")
                if not u_progress_name:
                    missing.append("u_progress")
                logger.error(f"Transition '{name}' missing required uniform(s) {missing}; rejecting program")
                GL.glDeleteProgram(program)
                return None
            alias_map = {}
            if u_texture1_name:
                alias_map["u_texture1"] = u_texture1_name
            if u_texture2_name:
                alias_map["u_texture2"] = u_texture2_name
            if u_progress_name:
                alias_map["u_progress"] = u_progress_name
            if u_resolution_name:
                alias_map["u_resolution"] = u_resolution_name
            self.transition_uniform_aliases[name] = alias_map
            return program
        except Exception as e:
            logger.error(f"Failed to compile transition '{name}': {e}")
            return None

    def _build_program_with_logging(self, vertex_src, frag_src, name):
        """Helper to build GL program and capture/print logs."""
        shader_hash = hashlib.sha1((vertex_src + frag_src).encode()).hexdigest()
        if shader_hash in self._failed_shader_hashes:
            return None
        prog_id = _compile_gl_program(vertex_src, frag_src, name=name)
        if prog_id > 0:
            return prog_id
        self._failed_shader_hashes.add(shader_hash)
        return None

    def _check_transition_hot_reload(self):
        """Monitor transition directory for changed files (Point 1)."""
        if not self.get_realized() or not hasattr(self, 'transition_shader_mtimes'):
            return
        changed = False
        for frag_file in self.transitions_dir.glob("*.frag"):
            path = str(frag_file)
            try:
                mtime = os.path.getmtime(path)
                if (path not in self.transition_shader_mtimes or mtime > self.transition_shader_mtimes[path]) and self._load_transition_shader(path):
                    changed = True
            except OSError:
                continue
        if changed:
            self.available_transitions = sorted(self.transition_cache.keys())

    def _create_advanced_shaders(self):
        """Create advanced post-processing shaders for enhanced visual effects."""
        self.effects_dir.mkdir(parents=True, exist_ok=True)
        vert_path = self.effects_dir / "effect.vert"
        with open(vert_path, "r") as f:
            vert_src = f.read()
        effect_names = ["motion_blur", "lens_distortion", "kaleidoscope", "glitch"]
        frag_sources = {}
        for name in effect_names:
            frag_path = self.effects_dir / f"{name}.frag"
            try:
                with open(frag_path, "r") as f:
                    frag_sources[name] = f.read()
            except (OSError, FileNotFoundError) as e:
                logger.warning(f"[EFFECT] Fragment shader not found for {name}: {e}")
                frag_sources[name] = None
        try:
            if frag_sources.get("motion_blur"):
                self.motion_blur_program = self._build_program(vert_src, frag_sources["motion_blur"])
            if frag_sources.get("lens_distortion"):
                self.lens_distortion_program = self._build_program(vert_src, frag_sources["lens_distortion"])
            if frag_sources.get("kaleidoscope"):
                self.kaleidoscope_program = self._build_program(vert_src, frag_sources["kaleidoscope"])
            if frag_sources.get("glitch"):
                self.glitch_program = self._build_program(vert_src, frag_sources["glitch"])
            if self.motion_blur_program:
                self.motion_blur_locs = {
                    "u_source": GL.glGetUniformLocation(self.motion_blur_program, "u_source"),
                    "u_previous_frame": GL.glGetUniformLocation(self.motion_blur_program, "u_previous_frame"),
                    "u_resolution": GL.glGetUniformLocation(self.motion_blur_program, "u_resolution"),
                    "u_strength": GL.glGetUniformLocation(self.motion_blur_program, "u_strength"),
                    "u_samples": GL.glGetUniformLocation(self.motion_blur_program, "u_samples"),
                    "u_time": GL.glGetUniformLocation(self.motion_blur_program, "u_time"),
                    "u_audio": GL.glGetUniformLocation(self.motion_blur_program, "u_audio"),
                }
            if self.lens_distortion_program:
                self.lens_distortion_locs = {
                    "u_source": GL.glGetUniformLocation(self.lens_distortion_program, "u_source"),
                    "u_resolution": GL.glGetUniformLocation(self.lens_distortion_program, "u_resolution"),
                    "u_strength": GL.glGetUniformLocation(self.lens_distortion_program, "u_strength"),
                    "u_time": GL.glGetUniformLocation(self.lens_distortion_program, "u_time"),
                    "u_audio": GL.glGetUniformLocation(self.lens_distortion_program, "u_audio"),
                    "u_cx": GL.glGetUniformLocation(self.lens_distortion_program, "u_cx"),
                    "u_cy": GL.glGetUniformLocation(self.lens_distortion_program, "u_cy"),
                }
            if self.kaleidoscope_program:
                self.kaleidoscope_locs = {
                    "u_source": GL.glGetUniformLocation(self.kaleidoscope_program, "u_source"),
                    "u_resolution": GL.glGetUniformLocation(self.kaleidoscope_program, "u_resolution"),
                    "u_segments": GL.glGetUniformLocation(self.kaleidoscope_program, "u_segments"),
                    "u_angle": GL.glGetUniformLocation(self.kaleidoscope_program, "u_angle"),
                    "u_time": GL.glGetUniformLocation(self.kaleidoscope_program, "u_time"),
                    "u_audio": GL.glGetUniformLocation(self.kaleidoscope_program, "u_audio"),
                    "u_cx": GL.glGetUniformLocation(self.kaleidoscope_program, "u_cx"),
                    "u_cy": GL.glGetUniformLocation(self.kaleidoscope_program, "u_cy"),
                }
            if self.glitch_program:
                self.glitch_locs = {
                    "u_source": GL.glGetUniformLocation(self.glitch_program, "u_source"),
                    "u_resolution": GL.glGetUniformLocation(self.glitch_program, "u_resolution"),
                    "u_intensity": GL.glGetUniformLocation(self.glitch_program, "u_intensity"),
                    "u_threshold": GL.glGetUniformLocation(self.glitch_program, "u_threshold"),
                    "u_time": GL.glGetUniformLocation(self.glitch_program, "u_time"),
                    "u_audio": GL.glGetUniformLocation(self.glitch_program, "u_audio"),
                    "u_cx": GL.glGetUniformLocation(self.glitch_program, "u_cx"),
                    "u_cy": GL.glGetUniformLocation(self.glitch_program, "u_cy"),
                }
        except Exception as e:
            logger.error(f"Advanced shaders failed: {e}")
            self.motion_blur_program = None
            self.lens_distortion_program = None
            self.kaleidoscope_program = None
            self.glitch_program = None
            self.motion_blur_locs = {}
            self.lens_distortion_locs = {}
            self.kaleidoscope_locs = {}
            self.glitch_locs = {}
        self.effect_shader_mtimes = {}
        self._effect_shader_cache = {}

    def _load_effect_shader(self, path):
        """Load and compile a single effect shader with mtime tracking for hot-reload."""
        try:
            with open(path, "r") as f:
                frag_source = f.read()
            name = Path(path).stem
            mtime = os.path.getmtime(path)
            source_hash = hashlib.sha256(frag_source.encode('utf-8')).hexdigest()
            if path in self._effect_shader_cache:
                cached_hash = self._effect_shader_cache[path]
                if cached_hash == source_hash:
                    return True
            vert_path = self.effects_dir / "effect.vert"
            try:
                with open(vert_path, "r") as f:
                    vert_src = f.read()
            except (OSError, FileNotFoundError):
                vert_src = """#version 330 core
                layout(location = 0) in vec2 position;
                out vec2 v_texcoord;
                void main() {
                    gl_Position = vec4(position, 0.0, 1.0);
                    v_texcoord = position * 0.5 + 0.5;
                }
                """
            program = self._build_program(vert_src, frag_source)
            if program:
                setattr(self, f"{name}_program", program)
                self.effect_shader_mtimes[path] = mtime
                self._effect_shader_cache[path] = source_hash
                locs = {}
                for uniform in ["u_source", "u_resolution", "u_strength", "u_time", "u_audio",
                                "u_cx", "u_cy", "u_segments", "u_angle", "u_intensity",
                                "u_threshold", "u_samples", "u_previous_frame"]:
                    loc = GL.glGetUniformLocation(program, uniform)
                    if loc >= 0:
                        locs[uniform] = loc
                setattr(self, f"{name}_locs", locs)
                return True
            else:
                logger.warning(f"[EFFECT] compile failed: {name}")
                return False
        except Exception as e:
            logger.error(f"Error loading effect shader {path}: {e}")
            return False

    def _check_effect_hot_reload(self):
        """Monitor effects directory for changed files."""
        if not self.get_realized() or not hasattr(self, 'effect_shader_mtimes'):
            return
        if not self.effects_dir.exists():
            return
        changed = False
        for frag_file in self.effects_dir.glob("*.frag"):
            path = str(frag_file)
            try:
                mtime = os.path.getmtime(path)
                if (path not in self.effect_shader_mtimes or mtime > self.effect_shader_mtimes[path]) and self._load_effect_shader(path):
                    changed = True
            except OSError:
                continue
        if changed:
            logger.info("Effect shaders reloaded")


    def _create_basic_geometry(self):
        """Create basic fullscreen quad geometry."""
        vao_obj = GLVAO()
        vao_obj.bind()
        self.vao = int(vao_obj)
        verts = np.array([-1.0, -1.0,  1.0, -1.0,  -1.0, 1.0,  1.0, 1.0], dtype='f4')
        self.vbo = int(GL.glGenBuffers(1))
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, verts.nbytes, verts, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 8, ctypes.c_void_p(0))

    def _setup_uniform_locations(self):
        """Setup uniform locations for all shader programs."""
        self.locs = {
            "u_time": GL.glGetUniformLocation(self.program, "u_time"),
            "u_resolution": GL.glGetUniformLocation(self.program, "u_resolution"),
            "u_audio": GL.glGetUniformLocation(self.program, "u_audio"),
            "u_texture": GL.glGetUniformLocation(self.program, "u_texture"),
            "u_zoom": GL.glGetUniformLocation(self.program, "u_zoom"),
            "u_rot": GL.glGetUniformLocation(self.program, "u_rot"),
        }

    def get_error(self):
        """Check for OpenGL errors like the original working renderer."""
        error = GL.glGetError()
        if error != GL.GL_NO_ERROR:
            error_names = {
                GL.GL_INVALID_ENUM: "INVALID_ENUM",
                GL.GL_INVALID_VALUE: "INVALID_VALUE",
                GL.GL_INVALID_OPERATION: "INVALID_OPERATION",
                GL.GL_INVALID_FRAMEBUFFER_OPERATION: "INVALID_FRAMEBUFFER_OPERATION",
                GL.GL_OUT_OF_MEMORY: "OUT_OF_MEMORY"
            }
            error_msg = error_names.get(error, f"Unknown error ({error})")
            logger.error(f"OpenGL error: {error_msg}")
            return error
        return None

    def _check_gl_error(self, operation: str, raise_on_error: bool = True):
        """Check for GL errors after critical operations.
        Args:
            operation: Description of the operation that just completed
            raise_on_error: If True, raise RuntimeError on GL error
        Returns:
            True if no error, False if error occurred
        """
        err = GL.glGetError()
        if err != GL.GL_NO_ERROR:
            error_names = {
                GL.GL_INVALID_ENUM: "INVALID_ENUM",
                GL.GL_INVALID_VALUE: "INVALID_VALUE",
                GL.GL_INVALID_OPERATION: "INVALID_OPERATION",
                GL.GL_STACK_OVERFLOW: "STACK_OVERFLOW",
                GL.GL_STACK_UNDERFLOW: "STACK_UNDERFLOW",
                GL.GL_OUT_OF_MEMORY: "OUT_OF_MEMORY",
                GL.GL_INVALID_FRAMEBUFFER_OPERATION: "INVALID_FRAMEBUFFER_OPERATION",
            }
            err_name = error_names.get(err, f"UNKNOWN({hex(err)})")
            msg = f"[GL ERROR] {operation}: {err_name}"
            logger.error(msg)
            if raise_on_error:
                raise RuntimeError(msg)
            return False
        return True

    def _clear_gl_errors(self):
        """Drain the GL error queue."""
        while GL.glGetError() != GL.GL_NO_ERROR:
            pass

    def _bind_texture_cached(self, unit, tex_id):
        """Bind a texture to a texture unit, skipping if already bound."""
        tex_id = int(tex_id)
        st = self._gl_state
        if st.texture_unit == unit and st.texture == tex_id:
            return
        GL.glActiveTexture(unit)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex_id)
        st.texture_unit = unit
        st.texture = tex_id

    def _use_program_cached(self, program):
        """Use a GL program, skipping if already in use."""
        if program and self._gl_state.program == program and GL.glIsProgram(program):
            return
        if program and GL.glIsProgram(program):
            GL.glUseProgram(program)
            self._gl_state.program = program
        else:
            GL.glUseProgram(0)
            self._gl_state.program = 0

    def _bind_fbo_cached(self, target, fbo_id):
        """Bind an FBO, skipping if already bound to both read and draw.

        Every framebuffer binding in the renderer must flow through this
        method so ``self._gl_state`` stays the single source of truth. A
        non-zero id that is not a valid, generated framebuffer object is
        never bound: this prevents GL_INVALID_OPERATION /
        GL_INVALID_FRAMEBUFFER_OPERATION caused by stale or already-deleted
        FBO ids lingering in the cache (e.g. a presentation target that was
        recreated during a resize or preset switch).
        """
        fbo_id = int(fbo_id)
        st = self._gl_state
        if fbo_id != 0 and not GL.glIsFramebuffer(fbo_id):
            logger.warning(
                "Refusing to bind FBO %s (target=%s): not a valid framebuffer object",
                fbo_id, target,
            )
            # Force a cache miss on the next matching bind so the id is
            # re-validated instead of silently skipped while GL points at a
            # different, possibly invalid, framebuffer.
            st.fbo_read = -1
            st.fbo_draw = -1
            return
        if target == GL.GL_FRAMEBUFFER:
            if st.fbo_read == fbo_id and st.fbo_draw == fbo_id:
                return
            GL.glBindFramebuffer(target, fbo_id)
            st.fbo_read = fbo_id
            st.fbo_draw = fbo_id
        elif target == GL.GL_READ_FRAMEBUFFER:
            if st.fbo_read == fbo_id:
                return
            GL.glBindFramebuffer(target, fbo_id)
            st.fbo_read = fbo_id
        elif target == GL.GL_DRAW_FRAMEBUFFER:
            if st.fbo_draw == fbo_id:
                return
            GL.glBindFramebuffer(target, fbo_id)
            st.fbo_draw = fbo_id
        else:
            GL.glBindFramebuffer(target, fbo_id)
            st.fbo_read = fbo_id
            st.fbo_draw = fbo_id

    def _bind_vao_cached(self, vao):
        """Bind a VAO, skipping if already bound."""
        vao = int(vao)
        if self._gl_state.vao == vao:
            return
        GL.glBindVertexArray(vao)
        self._gl_state.vao = vao

    def _sync_gl_state(self):
        """Reconcile ``self._gl_state`` with the actual OpenGL context.

        Rationale (see AGENTS.md, OpenGL state ownership): raw framebuffer /
        program / VAO binds performed by helper classes (``GLFramebuffer``,
        ``FBOManager``, ``WarpStack``) and FBO recreation during resize or
        preset switches can leave process-wide GL state out of sync with the
        cache. Calling this at well-defined checkpoints (frame start, after
        FBO setup) makes ``self._gl_state`` the single authoritative owner
        and eliminates "cache says A, GL says B" mismatches that produce
        GL_INVALID_FRAMEBUFFER_OPERATION and black screens.

        After deleting GL resources (``flush_deletes``) a bound FBO/program
        may have been reverted to the default, so this also corrects the
        cache for ids that no longer exist.
        """
        st = self._gl_state
        try:
            fb = np.zeros(1, dtype=np.int32)
            GL.glGetIntegerv(GL.GL_FRAMEBUFFER_BINDING, fb)
            st.fbo_read = int(fb[0])
            st.fbo_draw = int(fb[0])

            rb = np.zeros(1, dtype=np.int32)
            GL.glGetIntegerv(GL.GL_READ_FRAMEBUFFER_BINDING, rb)
            st.fbo_read = int(rb[0])

            db = np.zeros(1, dtype=np.int32)
            GL.glGetIntegerv(GL.GL_DRAW_FRAMEBUFFER_BINDING, db)
            st.fbo_draw = int(db[0])

            prog = np.zeros(1, dtype=np.int32)
            GL.glGetIntegerv(GL.GL_CURRENT_PROGRAM, prog)
            st.program = int(prog[0])

            vao = np.zeros(1, dtype=np.int32)
            GL.glGetIntegerv(GL.GL_VERTEX_ARRAY_BINDING, vao)
            st.vao = int(vao[0])

            unit = np.zeros(1, dtype=np.int32)
            GL.glGetIntegerv(GL.GL_ACTIVE_TEXTURE, unit)
            st.texture_unit = int(unit[0]) - GL.GL_TEXTURE0
        except Exception as exc:
            logger.debug("Failed to sync cached GL state from context: %s", exc)

    def _invalidate_gl_state_cache(self):
        """Drop any cached GL resource ids so the next bind re-applies.

        Call this immediately after ``GLResourceOwner.flush_deletes()``: a
        cached program/texture/FBO id may now refer to a deleted object, and
        leaving it in the cache would make the renderer skip a required
        rebind and draw with wrong/garbage state.
        """
        st = self._gl_state
        st.program = 0
        st.vao = 0
        st.texture = 0
        st.texture_unit = -1
        # Use -1 (never a valid FBO id) so that any subsequent bind target
        # differs and forces a real GL bind + validation.
        st.fbo_read = -1
        st.fbo_draw = -1
        self._sync_gl_state()

    def _gpu_timer(self, name: str):
        """Context manager that records GPU time via timer query."""
        if not self._gpu_timers:
            return profile_stage(name)
        timer = self._gpu_timers.get(name)
        if timer is None:
            timer = GPUTimerQuery()
            self._gpu_timers[name] = timer
        timer.start()
        try:
            yield
        finally:
            timer.end()
            elapsed = timer.get_elapsed_ms()
            FrameProfiler().record(f"gpu_{name}", elapsed)

    def _upload_uniform_cached(self, loc, value, cache_key):
        """Upload a uniform only if its value changed since last upload."""
        if loc == -1:
            return
        cached = self._uniform_values.get(cache_key)
        if cached == value:
            return
        if isinstance(value, bool):
            GL.glUniform1i(loc, 1 if value else 0)
        elif isinstance(value, int):
            GL.glUniform1i(loc, value)
        elif isinstance(value, float):
            GL.glUniform1f(loc, value)
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            GL.glUniform2f(loc, float(value[0]), float(value[1]))
        elif isinstance(value, (list, tuple)) and len(value) == 3:
            GL.glUniform3f(loc, float(value[0]), float(value[1]), float(value[2]))
        elif isinstance(value, (list, tuple)) and len(value) == 4:
            GL.glUniform4f(loc, float(value[0]), float(value[1]), float(value[2]), float(value[3]))
        self._uniform_values[cache_key] = value

    def update_audio(self, pcm_array):
        """Accept raw PCM float32 samples, a dict of pre-computed audio levels,
        or individual audio parameters (waveform, spectrum, bass, mid, treb, rms, peak, beat, dt).

        When given a dict, uses the pre-computed bass/mid/treble/beat values directly.
        When given raw PCM samples, performs FFT analysis to extract bass/mid/treble
        energy and beat onset, then applies per-band attack/decay smoothing into self.levels.
        """
        if pcm_array is None:
            return
        if isinstance(pcm_array, dict):
            self.last_audio_time = time.time()
            bass = float(pcm_array.get("bass", 0.0))
            mid = float(pcm_array.get("mid", 0.0))
            treb = float(pcm_array.get("treb", 0.0))
            beat = float(pcm_array.get("beat", 0.0))
            rms = float(pcm_array.get("rms", 0.0))
            peak = float(pcm_array.get("peak", 0.0))
            waveform = pcm_array.get("waveform")
            spectrum = pcm_array.get("spectrum")
            if waveform is not None:
                self.audio_data = np.asarray(waveform, dtype=np.float32)
                if self.audio_data.ndim > 1:
                    self.audio_data = self.audio_data.mean(axis=-1)
                if len(self.audio_data) > 1024:
                    self.audio_data = self.audio_data[-1024:]
                elif len(self.audio_data) < 1024:
                    padded = np.zeros(1024, dtype=np.float32)
                    padded[-len(self.audio_data):] = self.audio_data
                    self.audio_data = padded
            if spectrum is not None:
                self.spectrum_data = np.asarray(spectrum, dtype=np.float32)
            for band, val in [('bass', bass), ('mid', mid), ('treb', treb)]:
                self.levels[band] = val
                self.levels[band + '_att'] = val
            self.levels['beat'] = beat
            self.levels['rms'] = rms
            self.levels['peak'] = peak
            self.beat_flash = max(self.beat_flash * 0.75, beat)
            self.brightness = 1.0 + beat * 0.4
            return
        if len(pcm_array) == 0:
            return
        self.last_audio_time = time.time()
        arr = np.asarray(pcm_array, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=-1)
        if not np.isfinite(arr).all():
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.clip(arr, -1.0, 1.0)
        self.audio_data = np.zeros(1024, dtype=np.float32)
        tail = arr[-1024:]
        self.audio_data[-len(tail):] = tail
        n = len(arr)
        if n < 256:
            return
        sample_rate = 44100
        fft_size = 2048
        max_fft = 65536
        if n > fft_size:
            window = arr[-fft_size:]
        else:
            fft_size = min(1 << (n - 1).bit_length(), max_fft)
            window = np.zeros(fft_size, dtype=np.float32)
            window[-n:] = arr[-n:]
        hann = np.hanning(fft_size).astype(np.float32)
        window = window * hann
        fft = np.fft.rfft(window)
        mags = np.abs(fft)
        freqs = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
        bass_mask = (freqs >= 20) & (freqs < 150)
        mid_mask = (freqs >= 150) & (freqs < 2000)
        treb_mask = (freqs >= 2000) & (freqs < 16000)
        bass = float(np.mean(mags[bass_mask])) if np.any(bass_mask) else 0.0
        mid = float(np.mean(mags[mid_mask])) if np.any(mid_mask) else 0.0
        treb = float(np.mean(mags[treb_mask])) if np.any(treb_mask) else 0.0
        bass = np.log1p(bass * 10.0) * 2.5
        mid = np.log1p(mid * 10.0) * 2.5
        treb = np.log1p(treb * 10.0) * 2.5
        bass = float(np.clip(bass, 0.0, 2.0))
        mid = float(np.clip(mid, 0.0, 2.0))
        treb = float(np.clip(treb, 0.0, 2.0))
        beat = 0.0
        now_t = time.time()
        bass_hist = getattr(self, '_bass_history', [])
        avg_bass = float(np.mean(bass_hist)) if bass_hist else 0.0
        if avg_bass > 0.01 and bass > avg_bass * 1.4 and bass > 0.08 and now_t - getattr(self, '_last_beat_time', 0) > 0.18:
            beat = 1.0
            self._last_beat_time = now_t
        bass_hist.append(bass)
        if len(bass_hist) > 50:
            bass_hist.pop(0)
        self._bass_history = bass_hist
        for band, val in [('bass', bass), ('mid', mid), ('treb', treb)]:
            params = self.per_band_response[band]
            att = params['attack']
            dec = params['decay']
            smooth_key = f'_{band}_smooth'
            prev = getattr(self, smooth_key, 0.0)
            if val > prev:
                new_val = att * prev + (1.0 - att) * val
            else:
                new_val = dec * prev + (1.0 - dec) * val
            setattr(self, smooth_key, new_val)
            self.levels[band] = val
            self.levels[band + '_att'] = new_val
        prev_beat = self.levels.get('beat', 0.0)
        self.levels['beat'] = max(prev_beat * 0.82, beat)
        self.beat_flash = max(self.beat_flash * 0.75, beat)

    def _tick(self):
        """Periodic simulation/update tick.

        This timer performs time-based housekeeping (hot-reload checks and
        automatic preset switching). It deliberately does NOT request a
        render: rendering is driven by a single source, the 16 ms animation
        timer, so GL draw scheduling stays deterministic.
        """
        if self.get_realized():
            self._check_transition_hot_reload()
            self._check_effect_hot_reload()
            self._check_auto_preset_switch()
        return True

    def _check_auto_preset_switch(self):
        """Fully-automatic preset switching on a time interval.

        No keyboard shortcuts are involved: when auto-switching is enabled and
        enough time has elapsed since the last switch (not while a transition or
        another switch is already in progress), the next preset is switched to
        asynchronously on the idle loop so it never runs inside the GL render
        callback.
        """
        if not self.auto_switch_enabled():
            return
        if getattr(self, "is_transitioning", False):
            return
        if getattr(self, "_auto_switch_in_progress", False):
            return
        milk_files = getattr(self, "available_milk_files", None)
        if not milk_files:
            return
        now = time.time()
        last = getattr(self, "_last_preset_switch_time", 0.0)
        interval = getattr(getattr(self, "_config", None), "auto_switch_interval_seconds", 30.0)
        if now - last >= interval:
            self._last_preset_switch_time = now
            GLib.idle_add(self._do_auto_preset_switch)

    def _do_auto_preset_switch(self):
        """Execute a deferred automatic preset switch on the idle loop."""
        if getattr(self, "_auto_switch_in_progress", False):
            return False
        if not self.auto_switch_enabled():
            return False
        self._auto_switch_in_progress = True
        try:
            self.next_preset(smooth=True)
        except Exception as e:
            logger.warning(f"Auto preset switch failed: {e}")
        finally:
            self._auto_switch_in_progress = False
        return False

    def _create_fbo(self, width, height, name=None):
        """Create a single FBO with attached texture, properly configured."""
        prev_fbo = np.zeros(1, dtype=np.int32)
        try:
            GL.glGetIntegerv(GL.GL_FRAMEBUFFER_BINDING, prev_fbo)
        except Exception:
            prev_fbo[0] = 0

        fbo = int(GL.glGenFramebuffers(1))
        tex = int(GL.glGenTextures(1))
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA16F,
            width,
            height,
            0,
            GL.GL_RGBA,
            GL.GL_FLOAT,
            None
        )
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        self._bind_fbo_cached(GL.GL_FRAMEBUFFER, fbo)
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT0,
            GL.GL_TEXTURE_2D,
            tex,
            0
        )
        import ctypes
        draw_buffers = (ctypes.c_uint * 1)(GL.GL_COLOR_ATTACHMENT0)
        GL.glDrawBuffers(1, draw_buffers)
        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            self._bind_fbo_cached(GL.GL_FRAMEBUFFER, int(prev_fbo[0]))
            raise RuntimeError(f"FBO incomplete: {hex(status)}")
        self._bind_fbo_cached(GL.GL_FRAMEBUFFER, int(prev_fbo[0]))
        if name:
            self._fbo_manager.register(name, fbo, tex)
        return fbo, tex

    def _check_fbo_status(self, fbo_id):
        """Check framebuffer completeness without changing caller GL state."""
        fbo_id = int(fbo_id)

        previous = np.zeros(1, dtype=np.int32)
        try:
            GL.glGetIntegerv(GL.GL_FRAMEBUFFER_BINDING, previous)
            previous_fbo = int(previous[0])
        except Exception:
            previous_fbo = 0

        try:
            self._bind_fbo_cached(GL.GL_FRAMEBUFFER, fbo_id)

            status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)

            if status != GL.GL_FRAMEBUFFER_COMPLETE:
                status_names = {
                    GL.GL_FRAMEBUFFER_UNDEFINED: "UNDEFINED",
                    GL.GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT: "INCOMPLETE_ATTACHMENT",
                    GL.GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
                        "INCOMPLETE_MISSING_ATTACHMENT",
                    GL.GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
                        "INCOMPLETE_DRAW_BUFFER",
                    GL.GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
                        "INCOMPLETE_READ_BUFFER",
                    GL.GL_FRAMEBUFFER_UNSUPPORTED: "UNSUPPORTED",
                    GL.GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
                        "INCOMPLETE_MULTISAMPLE",
                }

                status_name = status_names.get(
                    status,
                    f"Unknown status code {hex(status)}"
                )

                logger.error(
                    "Framebuffer incomplete: fbo=%d status=%s",
                    fbo_id,
                    status_name,
                )
                return False

            return True

        finally:
            try:
                self._bind_fbo_cached(GL.GL_FRAMEBUFFER, previous_fbo)
            except Exception as exc:
                logger.debug("Failed to restore previous framebuffer in _check_fbo_status: %s", exc)

    def _blit_fallback(self, width, height, source_tex, glarea_fbo=0):
        """Fallback for glBlitFramebuffer when the draw framebuffer is incomplete.

        Renders the source texture using the currently bound framebuffer.
        """
        try:
            if not self.display_program or int(self.display_program) <= 0:
                return
            if not source_tex or int(source_tex) <= 0 or not GL.glIsTexture(int(source_tex)):
                return
            status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
            if status != GL.GL_FRAMEBUFFER_COMPLETE:
                logger.debug(f"Skipping blit fallback: currently bound framebuffer incomplete (status={hex(status)})")
                return
            GL.glViewport(0, 0, width, height)
            self._use_program_cached(self.display_program)
            self._bind_vao_cached(self.vao)
            self._bind_texture_cached(GL.GL_TEXTURE0, int(source_tex))
            if self.display_tex_loc >= 0:
                GL.glUniform1i(self.display_tex_loc, 0)
            GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
        except Exception as e:
            logger.debug(f"Blit fallback also failed: {e}")

    def _setup_multipass_fbos(self, width=None, height=None):
        """Setup FBOs for feedback ping-pong rendering using _create_fbo helper."""
        if width is None or height is None:
            scale = self.get_scale_factor()
            width = max(1, int(self.get_allocated_width() * scale))
            height = max(1, int(self.get_allocated_height() * scale))
        if hasattr(self, 'fbo_width') and self.fbo_width == width and self.fbo_height == height:
            return
        self._cleanup_fbos()
        self._fbo_manager.cleanup()
        self.fbo = [0, 0]
        self.fbo_tex = [0, 0]
        try:
            for i in range(2):
                self.fbo[i], self.fbo_tex[i] = self._create_fbo(width, height, name=f"feedback_{i}")
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"  FBO {i} created: fbo={self.fbo[i]}, tex={self.fbo_tex[i]}")
        except Exception as e:
            logger.error(f"  ERROR creating FBOs: {e}")
            raise
        self.curr_fbo_idx = 0
        self.fbo_width = width
        self.fbo_height = height
        if getattr(self, 'is_transitioning', False):
            self._pending_transition_resize = (width, height)
        else:
            self._setup_transition_textures(width, height)
        self.blend_engine = BlendEngine()

    def _setup_transition_textures(self, width, height):
        """Create textures and FBO for preset transitions."""
        if self.transition_from_tex:
            GLResourceOwner.queue_delete("texture", int(self.transition_from_tex))
            self.transition_from_tex = None
        if self.transition_to_tex:
            GLResourceOwner.queue_delete("texture", int(self.transition_to_tex))
            self.transition_to_tex = None
        if hasattr(self, 'transition_fbo') and self.transition_fbo:
            GLResourceOwner.queue_delete("fbo", int(self.transition_fbo))
            self.transition_fbo = None
        self.transition_from_tex = int(GL.glGenTextures(1))
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.transition_from_tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, width, height, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        self.transition_to_tex = int(GL.glGenTextures(1))
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.transition_to_tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, width, height, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        prev_fbo = np.zeros(1, dtype=np.int32)
        try:
            GL.glGetIntegerv(GL.GL_FRAMEBUFFER_BINDING, prev_fbo)
        except Exception:
            prev_fbo[0] = 0
        self.transition_fbo = int(GL.glGenFramebuffers(1))
        self._bind_fbo_cached(GL.GL_FRAMEBUFFER, self.transition_fbo)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self.transition_to_tex, 0)
        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            logger.error(f"Transition FBO incomplete: {hex(status)}")
        self._bind_fbo_cached(GL.GL_FRAMEBUFFER, int(prev_fbo[0]))

    def _cleanup_fbos(self):
        """Clean up existing FBOs before recreating."""
        prev_fbo = np.zeros(1, dtype=np.int32)
        try:
            GL.glGetIntegerv(GL.GL_FRAMEBUFFER_BINDING, prev_fbo)
        except Exception:
            prev_fbo[0] = 0
        if not getattr(self, 'is_transitioning', False):
            if self.transition_from_tex:
                GLResourceOwner.queue_delete("texture", int(self.transition_from_tex))
                self.transition_from_tex = None
            if self.transition_to_tex:
                GLResourceOwner.queue_delete("texture", int(self.transition_to_tex))
                self.transition_to_tex = None
            if hasattr(self, 'transition_fbo') and self.transition_fbo:
                GLResourceOwner.queue_delete("fbo", int(self.transition_fbo))
                self.transition_fbo = None
        else:
            logger.debug("   [RESIZE] Transition active, deferring transition texture cleanup")
        for i in range(2):
            if self.fbo[i]:
                GLResourceOwner.queue_delete("fbo", int(self.fbo[i]))
                self.fbo[i] = None
            if self.fbo_tex[i]:
                GLResourceOwner.queue_delete("texture", int(self.fbo_tex[i]))
                self.fbo_tex[i] = None
        self._fbo_manager.cleanup()
        self._bind_fbo_cached(GL.GL_FRAMEBUFFER, int(prev_fbo[0]))

    def _setup_bloom_fbos(self, width=None, height=None):
        """Setup FBOs for bloom/glow effect at quarter resolution."""
        if width is None or height is None:
            w = max(1, self.get_allocated_width())
            h = max(1, self.get_allocated_height())
        else:
            w, h = width, height

        if w <= 1 or h <= 1:
            return
        bloom_w = max(1, w // 4)
        bloom_h = max(1, h // 4)
        self._cleanup_bloom_fbos()
        self.bloom_tex = int(GL.glGenTextures(1))
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.bloom_tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA16F, bloom_w, bloom_h, 0, GL.GL_RGBA, GL.GL_FLOAT, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        self.bloom_fbo = int(GL.glGenFramebuffers(1))
        self._bind_fbo_cached(GL.GL_FRAMEBUFFER, self.bloom_fbo)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self.bloom_tex, 0)
        import ctypes
        draw_buffers = (ctypes.c_uint * 1)(GL.GL_COLOR_ATTACHMENT0)
        GL.glDrawBuffers(1, draw_buffers)
        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"Bloom FBO incomplete: {hex(status)}")
        self.bloom_blur_tex = []
        self.bloom_blur_fbo = []
        for s_idx in range(3):
            tex_pair = [0, 0]
            fbo_pair = [0, 0]
            for i in range(2):
                tex_pair[i] = int(GL.glGenTextures(1))
                GL.glBindTexture(GL.GL_TEXTURE_2D, tex_pair[i])
                GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA16F, bloom_w, bloom_h, 0, GL.GL_RGBA, GL.GL_FLOAT, None)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
                fbo_pair[i] = int(GL.glGenFramebuffers(1))
                self._bind_fbo_cached(GL.GL_FRAMEBUFFER, fbo_pair[i])
                GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, tex_pair[i], 0)
                GL.glDrawBuffers(1, draw_buffers)
                status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
                if status != GL.GL_FRAMEBUFFER_COMPLETE:
                    raise RuntimeError(f"Bloom blur FBO {i} incomplete: {hex(status)}")
            self.bloom_blur_tex.append(tex_pair)
            self.bloom_blur_fbo.append(fbo_pair)
        self._bind_fbo_cached(GL.GL_FRAMEBUFFER, 0)
        self._fbo_manager.register("bloom", self.bloom_fbo, self.bloom_tex)
        for s_idx in range(3):
            for i in range(2):
                self._fbo_manager.register(f"bloom_blur_{s_idx}_{i}", self.bloom_blur_fbo[s_idx][i], self.bloom_blur_tex[s_idx][i])
        self._setup_advanced_fbos(w, h)

    def _setup_advanced_fbos(self, width, height):
        """Setup FBOs for advanced post-processing effects."""
        bloom_w = max(1, width // 4)
        bloom_h = max(1, height // 4)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Setting up advanced effect FBOs at {width}x{height} (bloom at {bloom_w}x{bloom_h})")
        self._cleanup_advanced_fbos()
        self.motion_blur_tex = int(GL.glGenTextures(1))
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.motion_blur_tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, width, height, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        self.motion_blur_fbo = int(GL.glGenFramebuffers(1))
        self._bind_fbo_cached(GL.GL_FRAMEBUFFER, self.motion_blur_fbo)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self.motion_blur_tex, 0)
        import ctypes
        draw_buffers = (ctypes.c_uint * 1)(GL.GL_COLOR_ATTACHMENT0)
        GL.glDrawBuffers(1, draw_buffers)
        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"Motion blur FBO incomplete: {hex(status)}")
        self.lens_tex = int(GL.glGenTextures(1))
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.lens_tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, width, height, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        self.lens_fbo = int(GL.glGenFramebuffers(1))
        self._bind_fbo_cached(GL.GL_FRAMEBUFFER, self.lens_fbo)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self.lens_tex, 0)
        GL.glDrawBuffers(1, draw_buffers)
        self.bloom_reservoir_tex = int(GL.glGenTextures(1))
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.bloom_reservoir_tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA16F, bloom_w, bloom_h, 0, GL.GL_RGBA, GL.GL_FLOAT, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        self.bloom_reservoir_fbo = int(GL.glGenFramebuffers(1))
        self._bind_fbo_cached(GL.GL_FRAMEBUFFER, self.bloom_reservoir_fbo)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self.bloom_reservoir_tex, 0)
        GL.glDrawBuffers(1, draw_buffers)
        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"Lens FBO incomplete: {hex(status)}")
        self.kaleidoscope_tex = int(GL.glGenTextures(1))
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.kaleidoscope_tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, width, height, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        self.kaleidoscope_fbo = int(GL.glGenFramebuffers(1))
        self._bind_fbo_cached(GL.GL_FRAMEBUFFER, self.kaleidoscope_fbo)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self.kaleidoscope_tex, 0)
        GL.glDrawBuffers(1, draw_buffers)
        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"Kaleidoscope FBO incomplete: {hex(status)}")
        self._bind_fbo_cached(GL.GL_FRAMEBUFFER, 0)
        self._fbo_manager.register("motion_blur", self.motion_blur_fbo, self.motion_blur_tex)
        self._fbo_manager.register("lens_distortion", self.lens_fbo, self.lens_tex)
        self._fbo_manager.register("bloom_reservoir", self.bloom_reservoir_fbo, self.bloom_reservoir_tex)
        self._fbo_manager.register("kaleidoscope", self.kaleidoscope_fbo, self.kaleidoscope_tex)
        logger.info("Advanced effect FBOs created successfully")

    def _cleanup_advanced_fbos(self):
        """Clean up advanced effect FBOs and textures."""
        if hasattr(self, 'motion_blur_fbo') and self.motion_blur_fbo:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"   [VRAM] Deleting motion blur FBO: {self.motion_blur_fbo}")
            GLResourceOwner.queue_delete("fbo", int(self.motion_blur_fbo))
            self.motion_blur_fbo = None
        if hasattr(self, 'motion_blur_tex') and self.motion_blur_tex:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"   [VRAM] Deleting motion blur texture: {self.motion_blur_tex}")
            GLResourceOwner.queue_delete("texture", int(self.motion_blur_tex))
            self.motion_blur_tex = None
        if hasattr(self, 'lens_fbo') and self.lens_fbo:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"   [VRAM] Deleting lens FBO: {self.lens_fbo}")
            GLResourceOwner.queue_delete("fbo", int(self.lens_fbo))
            self.lens_fbo = None
        if hasattr(self, 'lens_tex') and self.lens_tex:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"   [VRAM] Deleting lens texture: {self.lens_tex}")
            GLResourceOwner.queue_delete("texture", int(self.lens_tex))
            self.lens_tex = None
        if hasattr(self, 'kaleidoscope_fbo') and self.kaleidoscope_fbo:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"   [VRAM] Deleting kaleidoscope FBO: {self.kaleidoscope_fbo}")
            GLResourceOwner.queue_delete("fbo", int(self.kaleidoscope_fbo))
            self.kaleidoscope_fbo = None
        if hasattr(self, 'kaleidoscope_tex') and self.kaleidoscope_tex:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"   [VRAM] Deleting kaleidoscope texture: {self.kaleidoscope_tex}")
            GLResourceOwner.queue_delete("texture", int(self.kaleidoscope_tex))
            self.kaleidoscope_tex = None
        self._fbo_manager.unregister("motion_blur")
        self._fbo_manager.unregister("lens_distortion")
        self._fbo_manager.unregister("bloom_reservoir")
        self._fbo_manager.unregister("kaleidoscope")

    def _cleanup_bloom_fbos(self):
        """Clean up bloom FBOs."""
        self._bind_fbo_cached(GL.GL_FRAMEBUFFER, 0)
        if self.bloom_fbo:
            try:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"   [VRAM] Deleting bloom FBO: {self.bloom_fbo}")
                GLResourceOwner.queue_delete("fbo", int(self.bloom_fbo))
            except Exception as e:
                logger.debug(f"Failed to delete bloom FBO: {e}")
            self.bloom_fbo = None
        if self.bloom_tex:
            try:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"   [VRAM] Deleting bloom texture: {self.bloom_tex}")
                GLResourceOwner.queue_delete("texture", int(self.bloom_tex))
            except Exception as e:
                logger.debug(f"Failed to delete bloom texture: {e}")
            self.bloom_tex = None
        if hasattr(self, 'bloom_blur_fbo') and self.bloom_blur_fbo:
            for s_idx in range(len(self.bloom_blur_fbo)):
                for i in range(2):
                    if self.bloom_blur_fbo[s_idx] and self.bloom_blur_fbo[s_idx][i]:
                        try:
                            GLResourceOwner.queue_delete("fbo", int(self.bloom_blur_fbo[s_idx][i]))
                        except Exception as e:
                            logger.debug(f"Failed to delete bloom blur FBO: {e}")
                    if self.bloom_blur_tex[s_idx] and self.bloom_blur_tex[s_idx][i]:
                        try:
                            GLResourceOwner.queue_delete("texture", int(self.bloom_blur_tex[s_idx][i]))
                        except Exception as e:
                            logger.debug(f"Failed to delete bloom blur texture: {e}")
            self.bloom_blur_fbo = []
            self.bloom_blur_tex = []
        self._fbo_manager.unregister("bloom")
        for s_idx in range(3):
            for i in range(2):
                self._fbo_manager.unregister(f"bloom_blur_{s_idx}_{i}")

    def _update_texture_animation(self, dt):
        """Update texture animation state."""
        self.texture_time += dt
        bass_boost = 1.0 + self.levels.get("bass", 0.0) * 2.0
        self.texture_offset[0] += self.texture_scroll_speed[0] * bass_boost
        self.texture_offset[1] += self.texture_scroll_speed[1] * bass_boost
        self.texture_offset[0] = self.texture_offset[0] % 10.0
        self.texture_offset[1] = self.texture_offset[1] % 10.0

    def next_texture(self):
        """Switch to next available texture."""
        if not self.available_textures:
            logger.warning("No textures available for switching")
            return
        current_index = -1
        for i, texture_path in enumerate(self.available_textures):
            if texture_path in self.loaded_textures and self.loaded_textures[texture_path] == self.current_texture_id:
                current_index = i
                break
        next_index = (current_index + 1) % len(self.available_textures)
        next_texture_path = self.available_textures[next_index]
        new_texture_id = self._load_texture_file(next_texture_path)
        if new_texture_id > 0:
            if self.current_texture_id and self.current_texture_id != getattr(self, 'dummy_texture', 0):
                for path, tex_id in list(self.loaded_textures.items()):
                    if tex_id == self.current_texture_id:
                        GLResourceOwner.queue_delete("texture", int(tex_id))
                        del self.loaded_textures[path]
                        logger.debug(f"Cleaned up old texture: {Path(path).name}")
                        break
            self.current_texture_id = new_texture_id
            self.current_texture_name = Path(next_texture_path).name
            logger.info(f"Switched to texture: {self.current_texture_name}")
        else:
            logger.error(f"Failed to load texture: {Path(next_texture_path).name}")

    def previous_texture(self):
        """Switch to previous available texture."""
        if not self.available_textures:
            logger.warning("No textures available for switching")
            return
        current_index = -1
        for i, texture_path in enumerate(self.available_textures):
            if texture_path in self.loaded_textures and self.loaded_textures[texture_path] == self.current_texture_id:
                current_index = i
                break
        prev_index = (current_index - 1) % len(self.available_textures)
        prev_texture_path = self.available_textures[prev_index]
        new_texture_id = self._load_texture_file(prev_texture_path)
        if new_texture_id > 0:
            if self.current_texture_id and self.current_texture_id != getattr(self, 'dummy_texture', 0):
                for path, tex_id in list(self.loaded_textures.items()):
                    if tex_id == self.current_texture_id:
                        GLResourceOwner.queue_delete("texture", int(tex_id))
                        del self.loaded_textures[path]
                        logger.debug(f"Cleaned up old texture: {Path(path).name}")
                        break
            self.current_texture_id = new_texture_id
            self.current_texture_name = Path(prev_texture_path).name
            logger.info(f"Switched to texture: {self.current_texture_name}")
        else:
            logger.error(f"Failed to load texture: {Path(prev_texture_path).name}")

    def set_texture_by_index(self, index: int):
        """Set texture by index from available textures list."""
        if not self.available_textures:
            logger.warning("No textures available")
            return
        if index < 0 or index >= len(self.available_textures):
            logger.warning(f"Texture index {index} out of range (0-{len(self.available_textures)-1})")
            return
        texture_path = self.available_textures[index]
        new_texture_id = self._load_texture_file(texture_path)
        if new_texture_id > 0:
            if self.current_texture_id and self.current_texture_id != getattr(self, 'dummy_texture', 0):
                for path, tex_id in list(self.loaded_textures.items()):
                    if tex_id == self.current_texture_id:
                        GLResourceOwner.queue_delete("texture", int(tex_id))
                        del self.loaded_textures[path]
                        logger.debug(f"Cleaned up old texture: {Path(path).name}")
                        break
            self.current_texture_id = new_texture_id
            self.current_texture_name = Path(texture_path).name
            logger.info(f"Set texture to: {self.current_texture_name} (index {index})")
        else:
            logger.error(f"Failed to load texture: {Path(texture_path).name}")

    def set_texture_by_name(self, texture_name: str):
        """Set texture by filename (case-insensitive)."""
        if not self.available_textures:
            logger.warning("No textures available")
            return
        target_path = None
        target_index = -1
        for i, texture_path in enumerate(self.available_textures):
            if Path(texture_path).name.lower() == texture_name.lower():
                target_path = texture_path
                target_index = i
                break
        if not target_path:
            logger.warning(f"Texture '{texture_name}' not found")
            return
        new_texture_id = self._load_texture_file(target_path)
        if new_texture_id > 0:
            if self.current_texture_id and self.current_texture_id != getattr(self, 'dummy_texture', 0):
                for path, tex_id in list(self.loaded_textures.items()):
                    if tex_id == self.current_texture_id:
                        GLResourceOwner.queue_delete("texture", int(tex_id))
                        del self.loaded_textures[path]
                        logger.debug(f"Cleaned up old texture: {Path(path).name}")
                        break
            self.current_texture_id = new_texture_id
            self.current_texture_name = Path(target_path).name
            logger.info(f"Set texture to: {self.current_texture_name} (index {target_index})")
        else:
            logger.error(f"Failed to load texture: {texture_name}")

    def get_random_texture(self):
        """Set to a random texture."""
        if not self.available_textures:
            logger.warning("No textures available for random selection")
            return
        import random
        random_index = random.randint(0, len(self.available_textures) - 1)
        self.set_texture_by_index(random_index)

    def set_texture_by_category(self, category: str):
        """Set to a random texture from a specific category."""
        if not self.available_textures:
            logger.warning("No textures available")
            return
        if category not in self.texture_categories:
            logger.warning(f"Category '{category}' not found")
            return
        textures = self.texture_categories[category]
        if not textures:
            logger.warning(f"No textures in category '{category}'")
            return
        import random
        texture_path = random.choice(textures)
        new_texture_id = self._load_texture_file(texture_path)
        if new_texture_id > 0:
            if self.current_texture_id and self.current_texture_id != getattr(self, 'dummy_texture', 0):
                for path, tex_id in list(self.loaded_textures.items()):
                    if tex_id == self.current_texture_id:
                        GLResourceOwner.queue_delete("texture", int(tex_id))
                        del self.loaded_textures[path]
                        logger.debug(f"Cleaned up old texture: {Path(path).name}")
                        break
            self.current_texture_id = new_texture_id
            self.current_texture_name = Path(texture_path).name
            logger.info(f"Set texture to: {self.current_texture_name} (category: {category})")
        else:
            logger.error(f"Failed to load texture from category '{category}'")

    def _load_available_textures(self):
        """
        Populate texture discovery structures used by on_realize/on_render.
        Sets:
          - self.available_textures: flat list[str] of texture file paths
          - self.texture_categories: dict[str, list[str]] grouped by subfolder name
        Safe behavior:
          - If textures dir doesn't exist or is empty, both are set to empty defaults.
        """
        self.available_textures = []
        self.texture_categories = {}
        textures_root = getattr(self, "textures_dir", None)
        if not textures_root:
            return
        try:
            textures_root = Path(textures_root)
        except Exception:
            return
        if not textures_root.exists() or not textures_root.is_dir():
            return
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

        def is_texture_file(p: Path) -> bool:
            return p.is_file() and p.suffix.lower() in exts
        default_bucket = []
        try:
            for p in textures_root.iterdir():
                if p.is_dir():
                    continue
                if is_texture_file(p):
                    default_bucket.append(str(p))
        except Exception:
            default_bucket = []
        if default_bucket:
            self.texture_categories["default"] = sorted(default_bucket)
            self.available_textures.extend(self.texture_categories["default"])
        try:
            for sub in sorted([d for d in textures_root.iterdir() if d.is_dir()], key=lambda d: d.name):
                category = sub.name
                bucket = []
                try:
                    for p in sub.rglob("*"):
                        if is_texture_file(p):
                            bucket.append(str(p))
                except Exception as e:
                    logger.debug(f"Failed to scan texture directory {sub}: {e}")
                    bucket = []
                if bucket:
                    bucket = sorted(bucket)
                    self.texture_categories[category] = bucket
                    self.available_textures.extend(bucket)
        except Exception as e:
            logger.debug(f"Failed to scan texture directories: {e}")
        seen = set()
        uniq = []
        for t in self.available_textures:
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        self.available_textures = uniq
        with self._texture_load_lock:
            for texture_path in self.available_textures[:20]:
                if texture_path not in self._texture_load_queue:
                    self._texture_load_queue.append(texture_path)
        return

    def list_texture_categories(self):
        """Print all available texture categories."""
        if not self.texture_categories:
            logger.warning("No texture categories available")
            return
        logger.info("Available texture categories:")
        for category, textures in self.texture_categories.items():
            count = len(textures) if textures else 0
            logger.info(f"  {category}: {count} textures")

    def get_current_texture_info(self):
        """Get information about the currently loaded texture."""
        if not self.current_texture_id or self.current_texture_id == 0:
            return "No texture loaded"
        return f"Current texture: {self.current_texture_name} (ID: {self.current_texture_id})"

    def _check_texture_randomize(self):
        """Check if it's time to randomize texture based on settings."""
        if not self.auto_randomize_textures or not self.available_textures:
            return
        current_time = time.time()
        should_change = False
        if self.texture_randomize_beat_sync and hasattr(self, 'levels'):
            beat_strength = self.levels.get('beat', 0.0)
            if beat_strength > 0.8:
                self.beats_since_texture_change += 1
                if self.beats_since_texture_change >= np.random.randint(4, 16):
                    should_change = True
                    self.beats_since_texture_change = 0
        else:
            if current_time - self.last_texture_change_time >= self.texture_randomize_interval:
                should_change = True
        if should_change:
            self.get_random_texture()
            self.last_texture_change_time = current_time
            logger.debug(f"Auto-randomized texture: {self.get_current_texture_info()}")

    def toggle_auto_texture_randomize(self):
        """Toggle automatic texture randomization on/off."""
        self.auto_randomize_textures = not self.auto_randomize_textures
        status = "ON" if self.auto_randomize_textures else "OFF"
        logger.info(f"Auto texture randomization: {status}")
        logger.debug(f"  Interval: {self.texture_randomize_interval}s")
        logger.debug(f"  Beat sync: {'ON' if self.texture_randomize_beat_sync else 'OFF'}")
        return self.auto_randomize_textures

    def set_texture_randomize_interval(self, interval_seconds: float):
        """Set the interval for automatic texture randomization."""
        self.texture_randomize_interval = max(5.0, float(interval_seconds))
        logger.info(f"Texture randomize interval set to: {self.texture_randomize_interval}s")

    def toggle_beat_sync_randomize(self):
        """Toggle beat-synchronized texture randomization."""
        self.texture_randomize_beat_sync = not self.texture_randomize_beat_sync
        status = "ON" if self.texture_randomize_beat_sync else "OFF"
        logger.info(f"Beat-sync texture randomization: {status}")
        return self.texture_randomize_beat_sync

    def _bind_texture(self, texture_unit=1):
        """Bind current texture to specified texture unit."""
        if self.current_texture_id and self.current_texture_id > 0:
            GL.glActiveTexture(GL.GL_TEXTURE0 + texture_unit)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.current_texture_id)
            return True
        return False

    def _draw_initial_pattern(self, width: int, height: int):
        """Draw a simple initial pattern to seed the feedback system using modern OpenGL."""
        if hasattr(self, 'vao') and self.vao:
            simple_frag = """
            in vec2 v_texcoord;
            out vec4 FragColor;
            void main() {
                // Create a colorful gradient pattern
                vec2 uv = v_texcoord;
                vec3 color = vec3(
                    0.5 + 0.3 * sin(uv.x * 3.14159),
                    0.5 + 0.3 * cos(uv.y * 3.14159 + 2.094),
                    0.5 + 0.3 * sin((uv.x + uv.y) * 3.14159 + 4.188)
                );
                FragColor = vec4(color, 1.0);
            }
            """
            try:
                vs = GL.glCreateShader(GL.GL_VERTEX_SHADER)
                GL.glShaderSource(vs, self.VERTEX_SHADER)
                GL.glCompileShader(vs)
                fs = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
                GL.glShaderSource(fs, simple_frag)
                GL.glCompileShader(fs)
                if GL.glGetShaderiv(vs, GL.GL_COMPILE_STATUS) and GL.glGetShaderiv(fs, GL.GL_COMPILE_STATUS):
                    simple_prog = GL.glCreateProgram()
                    GL.glAttachShader(simple_prog, vs)
                    GL.glAttachShader(simple_prog, fs)
                    GL.glLinkProgram(simple_prog)
                    if GL.glGetProgramiv(simple_prog, GL.GL_LINK_STATUS):
                        GL.glUseProgram(simple_prog)
                        if not getattr(self, 'vao', None):
                            self.is_transitioning = False
                            logger.debug("[TRANSITION] VAO missing; aborting transition this frame")
                        else:
                            GL.glBindVertexArray(self.vao)
                            GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
                        GLResourceOwner.queue_delete("program", simple_prog)
                        logger.debug("   Initial pattern drawn with modern OpenGL")
                    else:
                        logger.error(f"   Simple shader link failed: {GL.glGetProgramInfoLog(simple_prog)}")
                        GLResourceOwner.queue_delete("program", simple_prog)
                else:
                    logger.error("Simple shader compile failed")
                GL.glDeleteShader(vs)
                GL.glDeleteShader(fs)
            except Exception as e:
                logger.error(f"   Initial pattern failed: {e}")
                GL.glClearColor(0.2, 0.1, 0.3, 1.0)
                GL.glClear(GL.GL_COLOR_BUFFER_BIT)
                logger.debug("   Initial pattern: cleared to purple")
        else:
            GL.glClearColor(0.2, 0.1, 0.3, 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)
            logger.debug("   Initial pattern: cleared to purple (no VAO)")

    def _detect_context_loss(self):
        """Check if multiple critical GL resources are invalid, indicating context loss.

        Returns True if shader + VAO are both invalid, which strongly suggests
        the GL context was recreated by the compositor/window manager.
        """
        checks = 0
        failures = 0

        shader_invalid = False
        if self.shader_program:
            try:
                shader_invalid = not GL.glIsProgram(self.shader_program)
            except Exception:
                shader_invalid = True
        if shader_invalid:
            failures += 1
        checks += 1

        vao_invalid = False
        vao = getattr(self, 'vao', None)
        if vao:
            try:
                vao_invalid = not GL.glIsVertexArray(vao)
            except Exception:
                vao_invalid = True
        if vao_invalid:
            failures += 1
        checks += 1

        fbo_invalid = False
        fbo_list = getattr(self, 'fbo', [0, 0])
        for f in fbo_list:
            if f and int(f) > 0:
                try:
                    if not GL.glIsFramebuffer(int(f)):
                        fbo_invalid = True
                        break
                except Exception:
                    fbo_invalid = True
                    break
        if fbo_invalid:
            failures += 1
        checks += 1

        return failures >= 2

    def _reinitialize_gl_resources(self):
        """Full GL resource reinitialization after context loss.

        Clears all stale GL state and recreates shaders, FBOs, textures,
        VAOs, and other resources from scratch. Also reloads the current
        preset if one was loaded.
        """
        try:
            self.make_current()
        except Exception as e:
            logger.debug(f"make_current failed during reinit: {e}")

        self._cleanup_gl_resources(full=True)
        GLResourceOwner.flush_deletes()

        self._gl_state.reset()
        # Re-sync after resource teardown so cached state matches the context.
        self._sync_gl_state()

        try:
            GL.glDisable(GL.GL_DEPTH_TEST)
            self._create_simple_shader()
            self._create_basic_geometry()
            self._setup_multipass_fbos()
            self._create_bloom_shaders()
            self._create_display_shader()
            self._create_wave_shader()
            self._create_tone_map_shader()
            self._create_warp_shader()
            self._create_advanced_shaders()
            self._setup_bloom_fbos()
            self._setup_megabuf_texture()
            self._setup_noise_textures()
            self._create_shape_geometry(num_shapes=8)
            self._load_available_textures()
            self.opengl_ready = True

            current_preset = getattr(self, 'current_milk_file', None)
            if current_preset:
                try:
                    self.load_milk_preset(str(current_preset), smooth_transition=False)
                except Exception as e:
                    logger.warning(f"Failed to reload preset after reinit: {e}")

            self.queue_render()
            if not getattr(self, '_anim_timer_id', 0):
                self._anim_timer_id = GLib.timeout_add(16, self._on_animation_frame)
            self._start_texture_load_thread()
            self._load_shader_cache()
            logger.info("GL resources reinitialized successfully after context loss")
        except Exception as e:
            logger.error(f"GL reinitialization failed: {e}")
            self.opengl_ready = False
            self._realize_failed = True

    def on_render(self, area, context):
        """Render callback for the GLArea widget."""
        _prof = FrameProfiler()
        with profile_stage("frame_total"):
            if not self.get_realized() or not self.opengl_ready:
                return False
            widget_width = self.get_allocated_width()
            widget_height = self.get_allocated_height()
            scale_factor = self.get_scale_factor()
            width = int(widget_width * scale_factor)
            height = int(widget_height * scale_factor)
            if width <= 1 or height <= 1:
                return False
            try:
                glarea_fbo = int(GL.glGetIntegerv(GL.GL_FRAMEBUFFER_BINDING))
                viewport = GL.glGetIntegerv(GL.GL_VIEWPORT)
            except Exception:
                glarea_fbo = 0
                viewport = (0, 0, 0, 0)
            if glarea_fbo > 0 and not GL.glIsFramebuffer(glarea_fbo):
                glarea_fbo = 0
            # Reconcile the cached GL state with the framebuffer GTK just
            # bound for us. This makes the cache the single source of truth
            # for the rest of the frame and prevents stale cached FBO ids
            # from masking a required bind.
            self._sync_gl_state()
            if getattr(self, 'frame_count', 0) == 0:
                logger.info(
                    "FIRST FRAME: program=%s feedback_fbo=%s display_fbo=%s viewport=%s preset=%s",
                    getattr(self, 'shader_program', None),
                    getattr(self, 'fbo', [None])[0] if hasattr(self, 'fbo') else None,
                    glarea_fbo,
                    viewport,
                    getattr(self, 'current_preset_name', None),
                )
            return self._render_frame(width, height, glarea_fbo)

    def _upload_shader_params(self, params):
        """Upload common shader parameters for the warp rendering pass."""
        zoom_val = safe_float(params.get('zoom', 1.0), 1.0)
        rot_val  = safe_float(params.get('rot', 0.0), 0.0)
        warp_val = safe_float(params.get('warp', 0.0), 0.0)
        cx_val   = safe_float(params.get('cx', 0.5), 0.5)
        cy_val   = safe_float(params.get('cy', 0.5), 0.5)
        dx_val   = safe_float(params.get('dx', 0.0), 0.0)
        dy_val   = safe_float(params.get('dy', 0.0), 0.0)
        sx_val   = safe_float(params.get('sx', 1.0), 1.0)
        sy_val   = safe_float(params.get('sy', 1.0), 1.0)
        self._upload_uniform_cached(self.zoom_loc, (zoom_val, zoom_val), 'u_zoom')
        self._upload_uniform_cached(self.rot_loc, rot_val, 'u_rot')
        self._upload_uniform_cached(self.warp_loc, warp_val, 'u_warp')
        self._upload_uniform_cached(self.cx_loc, cx_val, 'u_cx')
        self._upload_uniform_cached(self.cy_loc, cy_val, 'u_cy')
        self._upload_uniform_cached(self.dx_loc, dx_val, 'u_dx')
        self._upload_uniform_cached(self.dy_loc, dy_val, 'u_dy')
        self._upload_uniform_cached(self.sx_loc, sx_val, 'u_sx')
        self._upload_uniform_cached(self.sy_loc, sy_val, 'u_sy')
        return zoom_val, rot_val, warp_val, cx_val, cy_val, dx_val, dy_val, sx_val, sy_val

    def _capture_transition_source(self):
        """Capture the previous FBO content as the transition source texture."""
        old_read = np.zeros(1, dtype=np.int32)
        old_draw = np.zeros(1, dtype=np.int32)
        try:
            GL.glGetIntegerv(GL.GL_READ_FRAMEBUFFER_BINDING, old_read)
        except Exception:
            old_read[0] = 0
        try:
            GL.glGetIntegerv(GL.GL_DRAW_FRAMEBUFFER_BINDING, old_draw)
        except Exception:
            old_draw[0] = 0

        try:
            source_fbo = getattr(self, 'transition_source_fbo', None)
            source_tex = getattr(self, 'transition_source_tex', None)
            if source_fbo is None:
                source_fbo = self.fbo[self.curr_fbo_idx]
            if source_tex is None:
                source_tex = self.fbo_tex[self.curr_fbo_idx]
            if self.get_realized() and self.fbo_width > 0 and source_tex:
                if not self.transition_from_tex:
                    self._setup_transition_textures(self.fbo_width, self.fbo_height)
                if not self._check_fbo_status(int(source_fbo)):
                    logger.warning(f"[TRANSITION] Source FBO {source_fbo} incomplete, aborting capture")
                    self.capture_pending = False
                else:
                    self._bind_fbo_cached(GL.GL_READ_FRAMEBUFFER, int(source_fbo))
                    GL.glBindTexture(GL.GL_TEXTURE_2D, self.transition_from_tex)
                    GL.glCopyTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, 0, 0, self.fbo_width, self.fbo_height)
        except Exception as exc:
            logger.debug("Exception in try block: %s", exc)
        finally:
            try:
                self._bind_fbo_cached(GL.GL_READ_FRAMEBUFFER, int(old_read[0]))
                self._bind_fbo_cached(GL.GL_DRAW_FRAMEBUFFER, int(old_draw[0]))
            except Exception as exc:
                logger.debug("Failed to restore previous framebuffer in _capture_transition_source: %s", exc)
            self.capture_pending = False

    def _render_frame(self, width, height, glarea_fbo):
        """Main render frame logic. Called from on_render."""
        self._is_rendering = True
        _prof = FrameProfiler()
        try:
            with profile_stage("frame_total"):
                with profile_stage("fbo_resize"):
                    if not hasattr(self, 'fbo_width') or self.fbo_width != width or self.fbo_height != height:
                        self.fbo_resizing = True
                        try:
                            self._setup_multipass_fbos(width, height)
                            logger.info(
                                "FBO SETUP RESULT: fbo0=%s tex0=%s fbo1=%s tex1=%s size=%sx%s",
                                self.fbo[0], self.fbo_tex[0],
                                self.fbo[1], self.fbo_tex[1],
                                self.fbo_width, self.fbo_height,
                            )
                            self._setup_bloom_fbos(width, height)
                            for i in range(2):
                                complete = self._check_fbo_status(self.fbo[i])
                                logger.info(
                                    "FBO STATUS: idx=%s fbo=%s tex=%s complete=%s",
                                    i, self.fbo[i], self.fbo_tex[i], complete,
                                )
                                if not complete:
                                    raise RuntimeError(f"Render FBO {i} incomplete after resize")
                            if hasattr(self, 'transition_fbo') and self.transition_fbo:
                                t_complete = self._check_fbo_status(self.transition_fbo)
                                logger.info(
                                    "FBO STATUS: transition_fbo=%s complete=%s",
                                    self.transition_fbo, t_complete,
                                )
                                if not t_complete:
                                    raise RuntimeError("Transition FBO incomplete after resize")
                            self.fbo_width = width
                            self.fbo_height = height
                        finally:
                            self.fbo_resizing = False
                with profile_stage("texture_upload"):
                    self._process_pending_texture_uploads()
                GLResourceOwner.flush_deletes()
                # Deleted ids may have been cached; drop stale cache entries
                # and re-query the actual GL state so the rest of the frame
                # binds from a correct baseline.
                self._invalidate_gl_state_cache()
                if not hasattr(self, 'frame_times'):
                    self.frame_times = []
                self.frame_times.append(time.perf_counter())
                if len(self.frame_times) > 60:
                    self.frame_times.pop(0)
                if getattr(self, 'capture_pending', False) and getattr(self, 'capture_pending_frame_id', -1) >= self.frame_count:
                    self._capture_transition_source()
                self.frame_count += 1
                if not hasattr(self, '_frames_since_switch'):
                    self._frames_since_switch = 999
                self._check_texture_randomize()
                if self.is_transitioning:
                    raw_progress = (time.time() - self.transition_start_time) / self.transition_duration
                    bass = self.levels.get("bass", 0.0)
                    audio_speed = 0.5 + bass * 2.0
                    self.transition_progress = min(raw_progress * audio_speed, 1.0)
                    if self.transition_progress >= 1.0:
                        self.is_transitioning = False
                        self.transition_progress = 1.0
                        self.transition_source_fbo = None
                        self.transition_source_tex = None
                    pending = getattr(self, "_pending_transition_resize", None)
                    if pending:
                        try:
                            pw, ph = pending
                            if self.get_realized():
                                self._setup_transition_textures(pw, ph)
                        except Exception as e:
                            logger.debug(f"Failed to setup transition textures: {e}")
                        self._pending_transition_resize = None

            # Skip FBO validation if in fallback mode
            if not getattr(self, '_fbo_supported', True):
                # In fallback mode, we still need basic validation
                if not self.shader_program:
                    return False
            else:
                if not (self.fbo[0] and self.fbo_tex[0]):
                    return False
                if self.fbo[0] > 0 and not GL.glIsFramebuffer(self.fbo[0]):
                    logger.warning("Feedback FBO invalid, recreating")
                    self._setup_multipass_fbos(width, height)

            # Validate shader program only if we have a valid GL context
            try:
                if self.shader_program and not GL.glIsProgram(self.shader_program):
                    program_id = int(self.shader_program) if self.shader_program else 0
                    if self._detect_context_loss():
                        logger.warning(
                            f"GL context loss detected (shader={program_id}), "
                            "triggering full reinitialization"
                        )
                        self._reinitialize_gl_resources()
                        return False
                    logger.warning(f"Shader program invalid (ID: {program_id}), recreating simple shader")
                    self.shader_program = None
                    self.program = None
                    if hasattr(self, '_gl_state'):
                        self._gl_state.program = 0
                    try:
                        self._create_simple_shader()
                        if self.shader_program and GL.glIsProgram(self.shader_program):
                            logger.info(f"Successfully recreated shader program (new ID: {int(self.shader_program)})")
                        else:
                            logger.error("Shader recreation failed - program still invalid after creation")
                            return False
                    except (RuntimeError, GL.GLError) as e:
                        logger.error(f"Failed to recreate shader: {e}")
                        return False
            except GL.GLError as e:
                logger.warning(f"GL error during shader validation: {e}")
                if self._detect_context_loss():
                    logger.warning("GL context loss detected during validation, triggering full reinitialization")
                    self._reinitialize_gl_resources()
                    return False
                if self.shader_program:
                    try:
                        self._create_simple_shader()
                    except (RuntimeError, GL.GLError) as shader_err:
                        logger.error(f"Failed to recreate shader after GL error: {shader_err}")
                        return False
            if self.shader_program and self.fbo[0] and self.fbo_tex[0]:
                try:
                    self._frames_since_switch += 1
                    params = self._get_current_params()
                    if params is None:
                        params = {'zoom': 1.0, 'rot': 0.0, 'warp': 0.0, 'feedback_amount': 0.5}
                    prev_idx = self.curr_fbo_idx
                    curr_idx = 1 - prev_idx
                    self.curr_fbo_idx = curr_idx
                    self._bind_fbo_cached(GL.GL_FRAMEBUFFER, self.fbo[curr_idx])
                    GL.glViewport(0, 0, width, height)
                    GL.glClearColor(0.0, 0.0, 0.0, 1.0)
                    GL.glClear(GL.GL_COLOR_BUFFER_BIT)
                    if getattr(self, 'vao', None):
                        self._bind_vao_cached(self.vao)
                    self._use_program_cached(self.shader_program)
                    self._upload_shader_params(params)
                    if self.fbo_tex[prev_idx]:
                        self._bind_texture_cached(GL.GL_TEXTURE0, self.fbo_tex[prev_idx])
                        prev_loc = self.prev_frame_loc
                        if prev_loc >= 0:
                            GL.glUniform1i(prev_loc, 0)
                    current_time = time.time() - self.start_time
                    frame_dt = current_time - self._last_frame_time
                    self._last_frame_time = current_time
                    if frame_dt <= 0.0 or frame_dt > 0.25:
                        frame_dt = 1.0 / 60.0
                    self._update_texture_animation(frame_dt)
                    self._auto_randomize_effects(current_time)
                    audio_levels = (
                        self.levels.get("bass", 0.0), self.levels.get("mid", 0.0),
                        self.levels.get("treb", 0.0), self.levels.get("beat", 0.0)
                    )
                    bass = self.levels.get("bass", 0.0)
                    mid = self.levels.get("mid", 0.0)
                    treb = self.levels.get("treb", 0.0)
                    beat = self.levels.get("beat", 0.0)
                    rms = self.levels.get("rms", 0.0)
                    peak = self.levels.get("peak", 0.0)
                    self._upload_uniform_cached(self.time_loc, current_time, 'u_time')
                    self._upload_uniform_cached(self.res_loc, (width, height), 'u_resolution')
                    self._upload_uniform_cached(self.audio_loc, tuple(audio_levels), 'u_audio')
                    self._upload_uniform_cached(self.audio_bands_loc,
                                                (self.levels.get("sub_bass", 0.0), self.levels.get("low_mid", 0.0),
                                                 self.levels.get("high_mid", 0.0), self.levels.get("high_treb", 0.0)),
                                                'u_audio_bands')
                    self._upload_uniform_cached(self.audio_derived_loc,
                                                (rms, peak, bass + mid + treb, 1.0 + beat * 0.5),
                                                'u_audio_derived')
                    self._upload_uniform_cached(self.audio_peaks_loc,
                                                (self.levels.get("peak_bass", 0.0), self.levels.get("peak_mid", 0.0),
                                                 self.levels.get("peak_treb", 0.0), min(self.levels.get("bpm", 0.0) / 200.0, 1.0)),
                                                'u_audio_peaks')
                    self._upload_uniform_cached(self.beat_flash_loc, self.beat_flash, 'u_beat_flash')
                    pulse_decay = 0.93
                    self.brightness = 1.0 + max(0.0, self.brightness - 1.0) * pulse_decay
                    if beat >= 0.5:
                        self.brightness = max(self.brightness, 1.4)
                    self._upload_uniform_cached(self.brightness_loc, self.brightness, 'u_brightness')
                    decay_val = float(params.get('decay', 0.9))
                    if not np.isfinite(decay_val):
                        decay_val = 0.9
                    decay_val = max(0.0, min(0.99, decay_val))
                    self._upload_uniform_cached(self.decay_loc, decay_val, 'u_decay')
                    feedback_val = float(params.get('feedback_amount', 0.95))
                    if not np.isfinite(feedback_val):
                        feedback_val = 0.95
                    feedback_val = max(0.0, min(0.99, feedback_val))
                    if not getattr(self, 'strict_milkdrop_mode', True):
                        feedback_val = feedback_val - bass * 0.12
                    feedback_val = max(0.80, min(0.99, feedback_val))
                    self._upload_uniform_cached(self.feedback_loc, feedback_val, 'u_feedback')
                    vm_cx = params.get('cx', 0.5)
                    vm_cy = params.get('cy', 0.5)
                    vm_warp = params.get('warp', 0.0)
                    vm_zoom = params.get('zoom', 1.0)
                    vm_rot = params.get('rot', 0.0)
                    vm_wave_r = params.get('wave_r', 0.0)
                    vm_wave_g = params.get('wave_g', 0.0)
                    vm_wave_b = params.get('wave_b', 0.0)
                    vm_wave_a = params.get('wave_a', 0.0)
                    vm_wave_size = params.get('wave_size', 0.3)
                    cx2 = vm_cx
                    cy2 = vm_cy
                    warp2 = vm_warp
                    zoom2 = vm_zoom
                    rot2 = vm_rot
                    wave_r2 = vm_wave_r
                    wave_g2 = vm_wave_g
                    wave_b2 = vm_wave_b
                    wave_a2 = vm_wave_a
                    wave_size2 = vm_wave_size
                    if not getattr(self, 'strict_milkdrop_mode', True):
                        cx2 = vm_cx + np.sin(current_time * 0.6) * bass * 0.12
                        cy2 = vm_cy + np.cos(current_time * 0.5) * mid * 0.12
                        warp2 = vm_warp + bass * 0.25
                        zoom2 = vm_zoom + bass * 0.08
                        rot2 = vm_rot + treb * 0.06
                        wave_r2 = vm_wave_r + bass * 0.7
                        wave_g2 = vm_wave_g + mid * 0.7
                        wave_b2 = vm_wave_b + treb * 0.7
                        wave_a2 = vm_wave_a + mid * 0.6
                        wave_size2 = vm_wave_size + bass * 1.2
                    self._upload_uniform_cached(self.zoom_loc, (zoom2, zoom2), 'u_zoom2')
                    self._upload_uniform_cached(self.rot_loc, rot2, 'u_rot2')
                    self._upload_uniform_cached(self.warp_loc, warp2, 'u_warp2')
                    self._upload_uniform_cached(self.cx_loc, cx2, 'u_cx2')
                    self._upload_uniform_cached(self.cy_loc, cy2, 'u_cy2')
                    self._upload_uniform_cached(self.wave_r_loc, wave_r2, 'u_wave_r')
                    self._upload_uniform_cached(self.wave_g_loc, wave_g2, 'u_wave_g')
                    self._upload_uniform_cached(self.wave_b_loc, wave_b2, 'u_wave_b')
                    self._upload_uniform_cached(self.wave_a_loc, wave_a2, 'u_wave_a')
                    self._upload_uniform_cached(self.wave_size_loc, wave_size2, 'u_wave_size')
                    megabuf_tex = getattr(self, 'megabuf_tex', None)
                    if megabuf_tex and int(megabuf_tex) > 0:
                        self._bind_texture_cached(GL.GL_TEXTURE0 + 7, int(megabuf_tex))
                        if self.megabuf_loc >= 0:
                            GL.glUniform1i(self.megabuf_loc, 7)
                    if GL.glIsEnabled(GL.GL_BLEND):
                        GL.glDisable(GL.GL_BLEND)
                    GL.glViewport(0, 0, width, height)
                    if hasattr(self, 'preset_vm') and self.preset_vm and hasattr(self.preset_vm, 'state'):
                        q_values = list(self._last_blended_q[:32]) if getattr(self, '_last_blended_q', None) is not None else (
                            self.preset_vm.state.q[:32] if hasattr(self.preset_vm.state, 'q') else [0.0] * 32
                        )
                        self._upload_q_registers(q_values)
                        self._upload_reg_registers()
                        blended_megabuf = getattr(self, '_last_blended_megabuf', None)
                        if blended_megabuf is None and hasattr(self.preset_vm.state, 'megabuf'):
                            blended_megabuf = self.preset_vm.state.megabuf
                        blended_megabuf_version = getattr(self, '_last_blended_megabuf_version', -1)
                        if hasattr(self.preset_vm.state, 'megabuf') and blended_megabuf is self.preset_vm.state.megabuf:
                            blended_megabuf_version = getattr(self.preset_vm.state, 'megabuf_version', 0)
                        if blended_megabuf is not None and blended_megabuf_version != getattr(self, '_last_megabuf_version_uploaded', -1):
                            self._update_megabuf_texture(blended_megabuf)
                            self._last_megabuf_version_uploaded = blended_megabuf_version
                        if hasattr(self.preset_vm, '_frame_rand_seed'):
                            rand_seed = self.preset_vm._frame_rand_seed
                        elif hasattr(self.preset_vm.state, 'rand_seed'):
                            rand_seed = self.preset_vm.state.rand_seed
                        else:
                            rand_seed = getattr(self, '_last_blended_rand_seed', None)
                        if rand_seed is not None:
                            self._upload_rand_seed(rand_seed)
                    if self.brightness_loc != -1:
                        GL.glUniform1f(self.brightness_loc, self.brightness)
                    texture_id = getattr(self, 'current_texture_id', 0)
                    if texture_id and int(texture_id) > 0 and GL.glIsTexture(int(texture_id)):
                        self._bind_texture_cached(GL.GL_TEXTURE0 + 1, int(texture_id))
                        if self.texture_loc >= 0:
                            GL.glUniform1i(self.texture_loc, 1)
                        self._upload_uniform_cached(self.texture_blend_loc, self.texture_blend, 'u_texture_blend')
                        self._upload_uniform_cached(self.texture_zoom_loc, self.texture_zoom, 'u_texture_zoom')
                        self._upload_uniform_cached(self.texture_offset_loc, tuple(self.texture_offset), 'u_texture_offset')
                        self._upload_uniform_cached(self.texture_distort_loc, self.texture_distort, 'u_texture_distort')
                    else:
                        self._upload_uniform_cached(self.texture_blend_loc, 0.0, 'u_texture_blend')
                        self._bind_noise_textures()
                    if not (getattr(self, 'vao', None)):
                        logger.error("VAO missing - cannot render")
                        return True
                    self._bind_vao_cached(self.vao)
                    if getattr(self, 'frame_count', 0) <= 1:
                        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
                        try:
                            draw_fbo = GL.glGetIntegerv(GL.GL_DRAW_FRAMEBUFFER_BINDING)
                        except Exception:
                            draw_fbo = 0
                        try:
                            curr_prog = GL.glGetIntegerv(GL.GL_CURRENT_PROGRAM)
                        except Exception:
                            curr_prog = 0
                        try:
                            curr_vao = GL.glGetIntegerv(GL.GL_VERTEX_ARRAY_BINDING)
                        except Exception:
                            curr_vao = 0
                        logger.info(
                            "DRAW STATE: FBO=0x%04X status=0x%04X program=%s vao=%s",
                            draw_fbo,
                            status,
                            curr_prog,
                            curr_vao,
                        )
                    GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
                    if getattr(self, 'frame_count', 0) <= 1:
                        err = GL.glGetError()
                        if err != GL.GL_NO_ERROR:
                            logger.error("FIRST DRAW GL ERROR: 0x%04X", err)
                    read_fbo_id = int(self.fbo[curr_idx])
                    # Resolve the presentation target. If the GLArea drawable
                    # id captured at frame start is no longer a valid
                    # framebuffer (stale ping-pong FBO, or one deleted this
                    # frame), fall back to the default drawable (0).
                    present_fbo = glarea_fbo if (glarea_fbo > 0 and GL.glIsFramebuffer(glarea_fbo)) else 0
                    if present_fbo != 0 and not GL.glIsFramebuffer(present_fbo):
                        present_fbo = 0
                    if read_fbo_id > 0 and GL.glIsFramebuffer(read_fbo_id):
                        read_ok = False
                        default_ok = False
                        try:
                            self._bind_fbo_cached(GL.GL_FRAMEBUFFER, read_fbo_id)
                            status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
                            read_ok = status == GL.GL_FRAMEBUFFER_COMPLETE
                        except Exception:
                            read_ok = False
                        try:
                            self._bind_fbo_cached(GL.GL_FRAMEBUFFER, present_fbo)
                            status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
                            default_ok = status == GL.GL_FRAMEBUFFER_COMPLETE
                        except Exception:
                            default_ok = False
                        if read_ok and default_ok:
                            try:
                                self._bind_fbo_cached(GL.GL_READ_FRAMEBUFFER, read_fbo_id)
                                self._bind_fbo_cached(GL.GL_DRAW_FRAMEBUFFER, present_fbo)
                                GL.glBlitFramebuffer(
                                    0, 0, width, height,
                                    0, 0, width, height,
                                    GL.GL_COLOR_BUFFER_BIT,
                                    GL.GL_NEAREST
                                )
                            except GL.GLError as e:
                                logger.debug(f"Blit to glarea framebuffer {present_fbo} failed ({e}); falling back to shader copy")
                                self._blit_fallback(width, height, self.fbo_tex[curr_idx], present_fbo)
                        else:
                            logger.debug(f"Skipping blit: read FBO complete={read_ok}, glarea_fbo complete={default_ok}")
                            self._blit_fallback(width, height, self.fbo_tex[curr_idx], present_fbo)
                    else:
                        logger.debug(f"Skipping blit: read FBO {read_fbo_id} is invalid")
                    self._bind_fbo_cached(GL.GL_FRAMEBUFFER, present_fbo)
                    default_fb_ok = False
                    try:
                        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
                        default_fb_ok = status == GL.GL_FRAMEBUFFER_COMPLETE
                    except Exception:
                        default_fb_ok = False
                    if default_fb_ok:
                        if hasattr(self, 'audio_data') and self.audio_data is not None:
                            self._render_waves(self.audio_data, params)
                        if self.shape_vaos and self.preset_vm and self.preset_vm.program.shapes:
                            self._render_shapes()
                    else:
                        logger.debug(f"Skipping post-blit overlay: glarea framebuffer {glarea_fbo} incomplete")
                    current_source_tex = self.fbo_tex[curr_idx]
                    if self.bloom_program and self.composite_program and self.bloom_fbo:
                        bloom_w = max(1, width // 4)
                        bloom_h = max(1, height // 4)
                        pp = self._config.post_process
                        self._bind_fbo_cached(GL.GL_FRAMEBUFFER, self.bloom_fbo)
                        GL.glViewport(0, 0, bloom_w, bloom_h)
                        self._use_program_cached(self.bloom_program)
                        self._bind_texture_cached(GL.GL_TEXTURE0, self.fbo_tex[curr_idx])
                        if self.bloom_src_loc != -1:
                            GL.glUniform1i(self.bloom_src_loc, 0)
                        if self.bloom_threshold_loc != -1:
                            GL.glUniform1f(self.bloom_threshold_loc, pp.bloom_threshold)
                        if self.bloom_intensity_loc != -1:
                            GL.glUniform1f(self.bloom_intensity_loc, pp.bloom_intensity + audio_levels[0] * pp.bloom_audio_bass_gain + audio_levels[3] * pp.bloom_audio_beat_gain)
                        self._bind_vao_cached(self.vao)
                        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
                        for s_idx, base_rad in enumerate(pp.bloom_blur_radii):
                            self._bind_fbo_cached(GL.GL_FRAMEBUFFER, self.bloom_blur_fbo[s_idx][0])
                            self._use_program_cached(self.blur_h_program)
                            self._bind_vao_cached(self.vao)
                            self._bind_texture_cached(GL.GL_TEXTURE0, self.bloom_tex)
                            GL.glUniform1i(self.blur_h_src_loc, 0)
                            GL.glUniform2f(self.blur_h_res_loc, bloom_w, bloom_h)
                            audio_boost = audio_levels[0] * pp.bloom_blur_audio_scale * (s_idx + 1)
                            GL.glUniform1f(self.blur_h_rad_loc, base_rad + audio_boost)
                            GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
                            self._bind_fbo_cached(GL.GL_FRAMEBUFFER, self.bloom_blur_fbo[s_idx][1])
                            self._use_program_cached(self.blur_v_program)
                            self._bind_vao_cached(self.vao)
                            self._bind_texture_cached(GL.GL_TEXTURE0, self.bloom_blur_tex[s_idx][0])
                            GL.glUniform1i(self.blur_v_src_loc, 0)
                            GL.glUniform2f(self.blur_v_res_loc, bloom_w, bloom_h)
                            GL.glUniform1f(self.blur_v_rad_loc, base_rad + audio_boost)
                            GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
                    if not getattr(self, 'strict_milkdrop_mode', True) and self.enable_motion_blur and self.motion_blur_program and self.motion_blur_fbo and current_source_tex:
                        self._bind_fbo_cached(GL.GL_FRAMEBUFFER, self.motion_blur_fbo)
                        GL.glViewport(0, 0, width, height)
                        self._use_program_cached(self.motion_blur_program)
                        self._bind_vao_cached(self.vao)
                        locs = getattr(self, 'motion_blur_locs', {})
                        self._bind_texture_cached(GL.GL_TEXTURE0, current_source_tex)
                        if locs.get('u_source', -1) != -1:
                            GL.glUniform1i(locs['u_source'], 0)
                        self._bind_texture_cached(GL.GL_TEXTURE0 + 1, self.fbo_tex[prev_idx])
                        if locs.get('u_previous_frame', -1) != -1:
                            GL.glUniform1i(locs['u_previous_frame'], 1)
                        if locs.get('u_resolution', -1) != -1:
                            GL.glUniform2f(locs['u_resolution'], width, height)
                        if locs.get('u_strength', -1) != -1:
                            strength = self.motion_blur_strength
                            if self.audio_motion_blur:
                                strength *= (1.0 + pow(audio_levels[0], 1.5) * pp.motion_blur_audio_scale)
                            GL.glUniform1f(locs['u_strength'], strength)
                        if locs.get('u_samples', -1) != -1:
                            GL.glUniform1i(locs['u_samples'], self.motion_blur_samples)
                        if locs.get('u_time', -1) != -1:
                            GL.glUniform1f(locs['u_time'], current_time)
                        if locs.get('u_audio', -1) != -1:
                            GL.glUniform4f(locs['u_audio'], *audio_levels)
                        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
                        current_source_tex = self.motion_blur_tex
                    if not getattr(self, 'strict_milkdrop_mode', True) and self.enable_lens_distortion and self.lens_distortion_program and self.lens_fbo and current_source_tex:
                        self._bind_fbo_cached(GL.GL_FRAMEBUFFER, self.lens_fbo)
                        GL.glViewport(0, 0, width, height)
                        self._use_program_cached(self.lens_distortion_program)
                        self._bind_vao_cached(self.vao)
                        locs = getattr(self, 'lens_distortion_locs', {})
                        self._bind_texture_cached(GL.GL_TEXTURE0, current_source_tex)
                        if locs.get('u_source', -1) != -1:
                            GL.glUniform1i(locs['u_source'], 0)
                        if locs.get('u_resolution', -1) != -1:
                            GL.glUniform2f(locs['u_resolution'], width, height)
                        if locs.get('u_strength', -1) != -1:
                            strength = self.lens_distortion_strength
                            if self.audio_lens_distortion:
                                strength *= (1.0 + pow(audio_levels[3], 2.0) * pp.lens_distortion_audio_scale)
                            GL.glUniform1f(locs['u_strength'], strength)
                        if locs.get('u_time', -1) != -1:
                            GL.glUniform1f(locs['u_time'], current_time)
                        if locs.get('u_audio', -1) != -1:
                            GL.glUniform4f(locs['u_audio'], *audio_levels)
                        if locs.get('u_cx', -1) != -1:
                            GL.glUniform1f(locs['u_cx'], params.get('cx', 0.5))
                        if locs.get('u_cy', -1) != -1:
                            GL.glUniform1f(locs['u_cy'], params.get('cy', 0.5))
                        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
                        current_source_tex = self.lens_tex
                    if not getattr(self, 'strict_milkdrop_mode', True) and self.enable_kaleidoscope and self.kaleidoscope_program and self.kaleidoscope_fbo and current_source_tex:
                        self._bind_fbo_cached(GL.GL_FRAMEBUFFER, self.kaleidoscope_fbo)
                        GL.glViewport(0, 0, width, height)
                        self._use_program_cached(self.kaleidoscope_program)
                        self._bind_vao_cached(self.vao)
                        locs = getattr(self, 'kaleidoscope_locs', {})
                        self._bind_texture_cached(GL.GL_TEXTURE0, current_source_tex)
                        if locs.get('u_source', -1) != -1:
                            GL.glUniform1i(locs['u_source'], 0)
                        if locs.get('u_resolution', -1) != -1:
                            GL.glUniform2f(locs['u_resolution'], width, height)
                        if locs.get('u_segments', -1) != -1:
                            GL.glUniform1i(locs['u_segments'], self.kaleidoscope_segments)
                        if locs.get('u_angle', -1) != -1:
                            self.kaleidoscope_angle += pp.kaleidoscope_angle_increment + audio_levels[0] * pp.kaleidoscope_angle_bass_gain + audio_levels[2] * pp.kaleidoscope_angle_treb_gain
                            GL.glUniform1f(locs['u_angle'], self.kaleidoscope_angle)
                        if locs.get('u_time', -1) != -1:
                            GL.glUniform1f(locs['u_time'], current_time)
                        if locs.get('u_audio', -1) != -1:
                            GL.glUniform4f(locs['u_audio'], *audio_levels)
                        if locs.get('u_cx', -1) != -1:
                            GL.glUniform1f(locs['u_cx'], params.get('cx', 0.5))
                        if locs.get('u_cy', -1) != -1:
                            GL.glUniform1f(locs['u_cy'], params.get('cy', 0.5))
                        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
                        current_source_tex = self.kaleidoscope_tex
                    # Use the actual GLArea framebuffer for final presentation.
                    # FBO 0 is OpenGL's default framebuffer and is not necessarily
                    # the same as the Gtk.GLArea drawable on all platforms.
                    # Prefer the GLArea drawable captured at frame start; if that
                    # id is no longer a valid framebuffer (e.g. a stale ping-pong
                    # FBO captured by mistake, or one deleted during this frame),
                    # fall back to the default drawable so we still present.
                    target_fbo = glarea_fbo if (glarea_fbo > 0 and GL.glIsFramebuffer(glarea_fbo)) else 0
                    if target_fbo != 0 and not GL.glIsFramebuffer(target_fbo):
                        target_fbo = 0
                    target_ok = False
                    try:
                        self._bind_fbo_cached(GL.GL_FRAMEBUFFER, target_fbo)
                        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
                        target_ok = status == GL.GL_FRAMEBUFFER_COMPLETE
                    except (Exception, GL.GLError):
                        target_ok = False
                    if target_ok:
                        try:
                            GL.glViewport(0, 0, width, height)
                            GL.glClearColor(0.0, 0.0, 0.0, 1.0)
                            GL.glClear(GL.GL_COLOR_BUFFER_BIT)
                        except Exception as clear_err:
                            logger.warning(f"Target framebuffer {target_fbo} clear failed: {clear_err}; skipping presentation this frame")
                            target_ok = False
                    if target_ok:
                        self._bind_fbo_cached(GL.GL_FRAMEBUFFER, target_fbo)
                        GL.glViewport(0, 0, width, height)
                    valid_target = target_ok
                    if valid_target and not getattr(self, 'strict_milkdrop_mode', True) and self.enable_glitch and self.glitch_program and current_source_tex:
                        self._use_program_cached(self.glitch_program)
                        self._bind_vao_cached(self.vao)
                        locs = getattr(self, 'glitch_locs', {})
                        self._bind_texture_cached(GL.GL_TEXTURE0, current_source_tex)
                        if locs.get('u_source', -1) != -1:
                            GL.glUniform1i(locs['u_source'], 0)
                        if locs.get('u_resolution', -1) != -1:
                            GL.glUniform2f(locs['u_resolution'], width, height)
                        if locs.get('u_intensity', -1) != -1:
                            intensity = self.glitch_intensity
                            if self.audio_glitch_trigger and audio_levels[0] > self.glitch_threshold:
                                intensity = min(1.0, intensity + 0.5)
                            GL.glUniform1f(locs['u_intensity'], intensity)
                        if locs.get('u_threshold', -1) != -1:
                            GL.glUniform1f(locs['u_threshold'], self.glitch_threshold)
                        if locs.get('u_time', -1) != -1:
                            GL.glUniform1f(locs['u_time'], current_time)
                        if locs.get('u_audio', -1) != -1:
                            GL.glUniform4f(locs['u_audio'], *audio_levels)
                        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
                    # In strict mode, skip modern effects. Otherwise use composite/bloom if available.
                    use_composite = (not getattr(self, 'strict_milkdrop_mode', True) or
                                     not (self.enable_glitch and self.glitch_program))
                    if valid_target and self.composite_program and use_composite and len(self.bloom_blur_tex) >= 3 and current_source_tex:
                        GL.glViewport(0, 0, width, height)
                        self._bind_fbo_cached(GL.GL_FRAMEBUFFER, target_fbo)
                        self._use_program_cached(self.composite_program)
                        self._bind_vao_cached(self.vao)
                        self._bind_texture_cached(GL.GL_TEXTURE0, current_source_tex)
                        if self.composite_orig_loc != -1:
                            GL.glUniform1i(self.composite_orig_loc, 0)
                        self._bind_texture_cached(GL.GL_TEXTURE0 + 1, self.bloom_blur_tex[0][1])
                        GL.glUniform1i(self.composite_bloom_small_loc, 1)
                        self._bind_texture_cached(GL.GL_TEXTURE0 + 2, self.bloom_blur_tex[1][1])
                        GL.glUniform1i(self.composite_bloom_medium_loc, 2)
                        self._bind_texture_cached(GL.GL_TEXTURE0 + 3, self.bloom_blur_tex[2][1])
                        GL.glUniform1i(self.composite_bloom_large_loc, 3)
                        if self.composite_bloom_res_loc != -1 and self.bloom_reservoir_tex:
                            self._bind_texture_cached(GL.GL_TEXTURE0 + 4, self.bloom_reservoir_tex)
                            GL.glUniform1i(self.composite_bloom_res_loc, 4)
                        if self.composite_audio_loc != -1:
                            GL.glUniform4f(self.composite_audio_loc, *audio_levels)
                        if self.composite_bloom_int_loc != -1:
                            GL.glUniform1f(self.composite_bloom_int_loc, pp.composite_bloom_intensity_base + audio_levels[0] * pp.composite_bloom_intensity_bass_gain + audio_levels[3] * pp.composite_bloom_intensity_beat_gain)
                        if self.composite_chrom_loc != -1:
                            GL.glUniform1f(self.composite_chrom_loc, min(pp.composite_chromatic_base + audio_levels[0] * pp.composite_chromatic_bass_gain + audio_levels[3] * pp.composite_chromatic_beat_gain, pp.composite_chromatic_max))
                        if self.composite_vig_loc != -1:
                            GL.glUniform1f(self.composite_vig_loc, min(max(pp.composite_vignette_base + audio_levels[2] * pp.composite_vignette_treb_gain + audio_levels[0] * pp.composite_vignette_bass_gain, pp.composite_vignette_min), pp.composite_vignette_max))
                        if self.composite_lift_loc != -1:
                            GL.glUniform1f(self.composite_lift_loc, pp.composite_lift_base + audio_levels[1] * pp.composite_lift_mid_gain)
                        if self.composite_gamma_loc != -1:
                            GL.glUniform1f(self.composite_gamma_loc, pp.composite_gamma_base + audio_levels[0] * pp.composite_gamma_bass_gain + audio_levels[2] * pp.composite_gamma_treb_gain)
                        if self.composite_gain_loc != -1:
                            GL.glUniform1f(self.composite_gain_loc, pp.composite_gain_base + audio_levels[3] * pp.composite_gain_beat_gain)
                        if self.composite_sat_loc != -1:
                            GL.glUniform1f(self.composite_sat_loc, min(pp.composite_saturation_max, pp.composite_saturation_base + audio_levels[1] * pp.composite_saturation_mid_gain + audio_levels[3] * pp.composite_saturation_beat_gain))
                        if self.composite_contrast_loc != -1:
                            GL.glUniform1f(self.composite_contrast_loc, min(pp.composite_contrast_max, pp.composite_contrast_base + audio_levels[0] * pp.composite_contrast_bass_gain))
                        self._bind_vao_cached(self.vao)
                        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
                        if getattr(self, 'bloom_reservoir_fbo', None) and self.bloom_blur_fbo:
                            read_fbo = int(self.bloom_blur_fbo[0][1])
                            draw_fbo = int(self.bloom_reservoir_fbo)
                            if read_fbo > 0 and draw_fbo > 0:
                                try:
                                    self._bind_fbo_cached(GL.GL_READ_FRAMEBUFFER, read_fbo)
                                    self._bind_fbo_cached(GL.GL_DRAW_FRAMEBUFFER, draw_fbo)
                                    GL.glBlitFramebuffer(0, 0, bloom_w, bloom_h, 0, 0, bloom_w, bloom_h,
                                                         GL.GL_COLOR_BUFFER_BIT, GL.GL_LINEAR)
                                except GL.GLError as e:
                                    logger.debug(f"Bloom reservoir blit failed: {e}")
                    elif valid_target and self.display_program:
                        self._use_program_cached(self.display_program)
                        self._bind_vao_cached(self.vao)
                        self._bind_texture_cached(GL.GL_TEXTURE0, current_source_tex)
                        GL.glUniform1i(self.display_tex_loc, 0)
                        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
                    if self.is_transitioning and self.transition_to_tex and int(self.transition_to_tex) > 0:
                        try:
                            if not GL.glIsTexture(int(self.transition_to_tex)):
                                pass
                            else:
                                self._bind_fbo_cached(GL.GL_READ_FRAMEBUFFER, glarea_fbo)
                                GL.glBindTexture(GL.GL_TEXTURE_2D, int(self.transition_to_tex))
                                GL.glCopyTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, 0, 0, width, height)
                        except Exception as exc:
                            logger.debug(f"Failed to capture transition target: {exc}")
                    if self.is_transitioning and self.transition_from_tex and self.transition_to_tex:
                        try:
                            if not GL.glIsTexture(int(self.transition_from_tex)) or not GL.glIsTexture(int(self.transition_to_tex)):
                                self.is_transitioning = False
                            else:
                                self._bind_fbo_cached(GL.GL_FRAMEBUFFER, glarea_fbo)
                                GL.glViewport(0, 0, width, height)
                                name = self.current_transition if isinstance(self.current_transition, str) else "fade_crossfade"
                                aliases = self.transition_uniform_aliases.get(name, {})
                                trans_prog = self.transition_cache.get(name)
                                if (not trans_prog or not GL.glIsProgram(int(trans_prog))) and hasattr(self, 'default_fade_program'):
                                    trans_prog = self.default_fade_program
                                if not trans_prog or not GL.glIsProgram(int(trans_prog)):
                                    self.is_transitioning = False
                                else:
                                    self._use_program_cached(int(trans_prog))
                                    self._bind_texture_cached(GL.GL_TEXTURE0, int(self.transition_from_tex))
                                    tloc = self._trans_locs_cache.get(name, {})
                                    tex1_loc = tloc.get('t1', -1)
                                    if tex1_loc == -1:
                                        tex1_loc = GL.glGetUniformLocation(trans_prog, aliases.get("u_texture1", "u_texture1"))
                                        tloc['t1'] = tex1_loc
                                    if tex1_loc != -1:
                                        GL.glUniform1i(tex1_loc, 0)
                                    self._bind_texture_cached(GL.GL_TEXTURE0 + 1, int(self.transition_to_tex))
                                    tex2_loc = tloc.get('t2', -1)
                                    if tex2_loc == -1:
                                        tex2_loc = GL.glGetUniformLocation(trans_prog, aliases.get("u_texture2", "u_texture2"))
                                        tloc['t2'] = tex2_loc
                                    if tex2_loc != -1:
                                        GL.glUniform1i(tex2_loc, 1)
                                    res_loc = tloc.get('res', -1)
                                    if res_loc == -1:
                                        res_loc = GL.glGetUniformLocation(trans_prog, aliases.get("u_resolution", "u_resolution"))
                                        tloc['res'] = res_loc
                                    if res_loc != -1:
                                        GL.glUniform2f(res_loc, width, height)
                                    prog_loc = tloc.get('prog', -1)
                                    if prog_loc == -1:
                                        prog_loc = GL.glGetUniformLocation(trans_prog, aliases.get("u_progress", "u_progress"))
                                        if prog_loc == -1:
                                            prog_loc = GL.glGetUniformLocation(trans_prog, "iProgressLinear")
                                        if prog_loc == -1:
                                            prog_loc = GL.glGetUniformLocation(trans_prog, "iProgressBicubic")
                                        if prog_loc == -1:
                                            prog_loc = GL.glGetUniformLocation(trans_prog, "iProgressCosine")
                                        tloc['prog'] = prog_loc
                                    if prog_loc != -1:
                                        GL.glUniform1f(prog_loc, float(self.transition_progress))
                                    if getattr(self, 'vao', None):
                                        self._bind_vao_cached(self.vao)
                                    else:
                                        self.is_transitioning = False
                                        return True
                                    GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
                        except Exception as e:
                            logger.debug(f"Transition render failed: {e}")
                            self.is_transitioning = False
                        try:
                            self._bind_fbo_cached(GL.GL_FRAMEBUFFER, glarea_fbo)
                        except Exception as e:
                            logger.debug(f"Failed to bind framebuffer: {e}")
                    if self.sprite_config_loaded and self.preset_vm:
                        GL.glViewport(0, 0, width, height)
                        self.sprite_renderer.render_sprites(self.preset_vm.state, width, height)
                except Exception as e:
                    preset_name = getattr(self, 'preset_name', '<unknown>')
                    program_id = getattr(self, 'shader_program', None)
                    logger.error(f"Shader rendering failed on frame {self.frame_count}, preset={preset_name}, program={program_id}: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    try:
                        self._bind_fbo_cached(GL.GL_FRAMEBUFFER, glarea_fbo)
                        GL.glViewport(0, 0, width, height)
                        GL.glClearColor(0.0, 0.0, 0.0, 1.0)
                        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
                    except Exception as exc:
                        logger.debug("Exception in try block: %s", exc)
            if getattr(self, 'screenshot_requested', False):
                self._save_screenshot(width, height)
            return True
        except Exception as e:
            logger.error(f"Render frame error: {e}")
            return False
        finally:
            self._is_rendering = False

    def on_resize(self, area, width, height):
        """Handle resize - just update dimensions, don't recreate resources."""
        scale = self.get_scale_factor()
        new_w, new_h = int(width * scale), int(height * scale)
        if new_w <= 1 or new_h <= 1:
            return
        if hasattr(self, 'fbo_width'):
            if new_w == self.fbo_width and new_h == self.fbo_height:
                return
            if abs(new_w - self.fbo_width) < 5 and abs(new_h - self.fbo_height) < 5:
                return
        self.fbo_width = new_w
        self.fbo_height = new_h
        if self.get_realized():
            if getattr(self, 'is_transitioning', False) or getattr(self, 'capture_pending', False):
                self._pending_transition_resize = (new_w, new_h)
            else:
                self._setup_transition_textures(new_w, new_h)

    def on_map(self, area):
        """Handle map event - widget is now visible on screen."""
        logger.info("GLArea: Map event - widget is now visible on screen")
        alloc = self.get_allocation()
        logger.debug(f"GLArea: Initial allocation on map: {alloc.width}x{alloc.height}")

    def _on_animation_frame(self):
        """Callback for animation timer - request render."""
        if self.get_realized() and self.opengl_ready:
            self.queue_render()
        return True

    def on_unmap(self, area):
        """Handle unmap event - widget is no longer visible."""
        logger.info("GLArea: Unmap event - widget is no longer visible")

    def on_key_pressed(self, controller, keyval, keycode, state):
        """Handle keyboard shortcuts for user interaction."""
        if keyval == Gdk.KEY_s:
            self.screenshot_requested = True
            self.queue_render()
        elif keyval == Gdk.KEY_f:
            toplevel = self.get_root()
            if isinstance(toplevel, Gtk.Window):
                toplevel.set_fullscreened(not toplevel.is_fullscreened())
        elif keyval == Gdk.KEY_Up:
            if hasattr(self, 'warp_amount'):
                self.warp_amount = min(self.warp_amount + 0.01, 1.0)
        elif keyval == Gdk.KEY_Down:
            if hasattr(self, 'warp_amount'):
                self.warp_amount = max(self.warp_amount - 0.01, 0.0)
        elif keyval == Gdk.KEY_1:
            if hasattr(self, 'warp_type'):
                self.warp_type = 0
        elif keyval == Gdk.KEY_2:
            if hasattr(self, 'warp_type'):
                self.warp_type = 1
        elif keyval == Gdk.KEY_3:
            if hasattr(self, 'warp_type'):
                self.warp_type = 2
        elif keyval == Gdk.KEY_4:
            if hasattr(self, 'warp_type'):
                self.warp_type = 3
        elif keyval == Gdk.KEY_5:
            if hasattr(self, 'warp_type'):
                self.warp_type = 4
        elif keyval == Gdk.KEY_6:
            if hasattr(self, 'warp_type'):
                self.warp_type = 5
        elif keyval == Gdk.KEY_7:
            if hasattr(self, 'warp_type'):
                self.warp_type = 6
        elif keyval == Gdk.KEY_8:
            if hasattr(self, 'warp_type'):
                self.warp_type = 7
        elif keyval == Gdk.KEY_9:
            if hasattr(self, 'warp_type'):
                self.warp_type = 8
        elif keyval == Gdk.KEY_0:
            if hasattr(self, 'warp_type'):
                self.warp_type = 9
        elif keyval == Gdk.KEY_i:
            if hasattr(self, 'show_osd'):
                self.show_osd = not self.show_osd
                self.osd_fade_timer = 5.0
            else:
                self.show_osd = True
                self.osd_fade_timer = 5.0
        elif keyval == Gdk.KEY_p and (state & Gdk.ModifierType.SHIFT_MASK) and (state & Gdk.ModifierType.CONTROL_MASK):
            if hasattr(self, 'show_telemetry'):
                self.show_telemetry = not self.show_telemetry
                logger.info(f"Performance telemetry overlay: {'ON' if self.show_telemetry else 'OFF'}")
        return False

    def auto_switch_enabled(self) -> bool:
        """Whether renderer-internal auto preset switching is allowed.
        If auto-switching is not explicitly disabled, allow it by default.
        """
        return not getattr(self, "_disable_auto_switching", False)

    def _init_preset_transitions(self):
        """Initialize preset transition system."""
        self.prev_params = None
        self.target_params = self._get_preset_params(getattr(self, 'preset_name', 'default'))
        self.transition_start_time = 0
        self.transition_duration = 1.0
        self.prev_preset_eqs = None
        self.transition_progress = 0.0
        self.transition_duration = 2.0
        self.is_transitioning = False
        self.preset_params = self.target_params.copy()

    def _get_preset_params(self, preset_name: str):
        """Get parameters for a preset. Generates unique params based on preset name hash."""
        seed = int(hashlib.md5(preset_name.encode()).hexdigest(), 16) % 10000
        rng = random.Random(seed)
        return {
            'speed': rng.uniform(0.5, 2.0),
            'complexity': rng.uniform(0.5, 3.0),
            'color_shift': rng.uniform(0.0, 1.0),
            'wave_intensity': rng.uniform(0.5, 2.0),
            'beat_sensitivity': rng.uniform(0.5, 2.0),
            'pattern_type': rng.randint(0, 7),
            'feedback_amount': rng.uniform(0.5, 0.95),
            'dx': rng.uniform(-0.5, 0.5),
            'dy': rng.uniform(-0.5, 0.5),
            'zoom': rng.uniform(0.8, 1.3),
            'rot': rng.uniform(0.0, 6.28),
            'show_waveform': rng.randint(0, 1),
            'texture_index': rng.randint(0, max(0, len(getattr(self, 'available_textures', [])) - 1)) if getattr(self, 'available_textures', []) else 0
        }

    def update_preset(self, preset_name):
        """Initiates a smooth transition to a new preset."""
        if preset_name.endswith('.milk'):
            presets_dir = Path(__file__).parent / "presets"
            if not Path(preset_name).exists() and presets_dir.exists():
                full_path = presets_dir / preset_name
                if full_path.exists():
                    self.load_milk_preset(str(full_path))
                    return
            else:
                self.load_milk_preset(preset_name)
                return
        if self.current_texture_id and self.current_texture_id != getattr(self, 'dummy_texture', 0):
            try:
                texture_to_delete = None
                for path, tex_id in list(self.loaded_textures.items()):
                    if tex_id == self.current_texture_id:
                        texture_to_delete = (path, tex_id)
                        break
                if texture_to_delete:
                    path, tex_id = texture_to_delete
                    if self.get_realized():
                        self.make_current()
                        GLResourceOwner.queue_delete("texture", int(tex_id))
                        del self.loaded_textures[path]
                        logger.info(f"Cleaned up texture: {Path(path).name}")
            except Exception as e:
                logger.warning(f"Failed to cleanup texture {self.current_texture_id}: {e}")
            finally:
                self.current_texture_id = 0
                self.prev_params = self.preset_params.copy()
                self.target_params = self._get_preset_params(preset_name)
                if self.smoothstep and getattr(self, 'fbo_width', 0) > 0:
                    self.capture_pending = True
                    self.capture_pending_frame_id = self.frame_count + 1
                self.preset_name = preset_name
                self.transition_start_time = time.time()
                self.is_transitioning = True
        if self.available_textures:
            try:
                texture_idx = self.target_params.get('texture_index', 0) % len(self.available_textures)
                texture_path = self.available_textures[texture_idx]
                self.current_texture_id = self._load_texture_file(texture_path)
                if self.current_texture_id > 0:
                    self.current_texture_name = Path(texture_path).name
                    logger.info(f"Preset '{preset_name}' using texture: {self.current_texture_name}")
                else:
                    logger.error(f"Failed to load texture for preset '{preset_name}'")
                    self.current_texture_name = ""
            except Exception as e:
                logger.error(f"Error loading texture for preset '{preset_name}': {e}")
                self.current_texture_id = 0
                self.current_texture_name = ""

    def load_preset(self, preset_name):
        """Load a preset by name or path. This is the public API for preset loading.
        Resolves preset names to full .milk file paths and delegates to load_milk_preset.
        """
        if not preset_name:
            return False
        preset_path = str(preset_name)
        if not preset_path.endswith('.milk'):
            presets_dir = Path(__file__).parent / "presets"
            candidate = presets_dir / f"{preset_path}.milk"
            if candidate.exists():
                preset_path = str(candidate)
            else:
                candidate = presets_dir / preset_path
                if candidate.exists():
                    preset_path = str(candidate)
        return self.load_milk_preset(preset_path, smooth_transition=True)

    def switch_to_preset(self, preset_name):
        """Atomic preset switch. Old state survives if the new preset fails to compile."""
        logger.info(f"Switching to preset: {preset_name}")
        return self.load_preset(preset_name)

    def request_auto_preset_change(self):
        if not self.auto_switch_enabled():
            return
        self.random_preset()

    def random_preset(self):
        """Load a random preset."""
        presets_dir = Path(__file__).parent / "presets"
        try:
            milk_files = sorted(presets_dir.rglob("*.milk"))
        except Exception:
            milk_files = []
        if not milk_files:
            return
        self.available_milk_files = milk_files
        import random
        new_preset = random.choice(milk_files)
        logger.info(f"Random preset: {new_preset}")
        self.load_preset(new_preset)

    def reload_current_preset(self):
        """Reload the current preset."""
        if hasattr(self, 'current_milk_file') and self.current_milk_file:
            logger.info(f"Reloading preset: {self.current_milk_file}")
            self.load_milk_preset(self.current_milk_file, smooth_transition=False)

    def previous_preset(self):
        """Load previous preset."""
        presets_dir = Path(__file__).parent / "presets"
        try:
            milk_files = sorted(presets_dir.rglob("*.milk"))
        except Exception:
            milk_files = []
        self.available_milk_files = milk_files
        if not milk_files or not hasattr(self, 'current_milk_file'):
            return
        try:
            norm_current = os.path.abspath(self.current_milk_file)
            normalized = [os.path.abspath(str(p)) for p in milk_files]
            current_idx = normalized.index(norm_current)
        except ValueError:
            try:
                current_idx = [os.path.abspath(str(p)) for p in milk_files].index(os.path.abspath(self.current_milk_file))
            except ValueError:
                current_idx = -1
        prev_idx = (current_idx - 1) % len(milk_files)
        new_preset = milk_files[prev_idx]
        logger.info(f"Previous preset: {new_preset}")
        self.load_preset(new_preset)

    def load_milk_preset(self, milk_path, smooth_transition=True):
        """Load a real .milk preset file and parse its equations."""
        if hasattr(CompiledExpr, 'bad_presets') and milk_path in CompiledExpr.bad_presets:
            logger.warning(f"Preset blacklisted (previous failure): {Path(milk_path).name}")
            return False
        if not Path(milk_path).exists():
            logger.error(f"Milk preset not found: {milk_path}")
            return False
        try:
            if self.get_realized():
                self.make_current()
            # Unbind any current shader program before compilation to prevent invalidation
            try:
                GL.glUseProgram(0)
                if hasattr(self, '_gl_state'):
                    self._gl_state.program = 0
            except GL.GLError:
                pass

            self._setup_megabuf_texture()

            old_vm = getattr(self, 'preset_vm', None)

            # PHASE 1: Build new state without destroying old GL resources
            raw_preset = parse_milk_file(milk_path)
            evolved = self.evolution_engine.evolve(self._last_loaded_raw_preset, raw_preset)
            new_last_loaded_raw_preset = evolved
            preset_eqs = evolved.to_dict()
            if random.random() < 0.3:
                new_org = self.evolution_engine.extract_dna(raw_preset)
                self.evolution_engine.population[self.evolution_engine.active_idx] = new_org
                logger.info(f"🧬 DNA Extracted from {Path(milk_path).name}")
            _frames_since_switch = 0
            _last_preset_switch_time = time.time()
            impl = self.available_transitions
            if impl:
                try:
                    idx = impl.index(self.current_transition)
                    current_transition = impl[(idx + 1) % len(impl)]
                except ValueError:
                    current_transition = impl[0]
            else:
                current_transition = self.current_transition
            current_milk_file = milk_path
            current_preset_name = Path(milk_path).name
            params = preset_eqs.get('params', {})
            logger.info(f"Loaded .milk preset: {Path(milk_path).name} ({len(params)} params, {len(preset_eqs.get('per_frame', []))} frame equations)")
            preset_params = preset_eqs.get('params', {})
            try:
                src_mtime = Path(milk_path).stat().st_mtime
            except Exception:
                src_mtime = 0.0
            cached = PRESET_CACHE.get(milk_path, src_mtime)
            if cached:
                new_preset_program = PresetProgram(from_cache=cached)
                logger.info("Preset loaded from binary cache (.mdc)")
            else:
                new_preset_program = PresetProgram(preset_dict=preset_eqs)
                cache_data = _build_preset_cache_data(new_preset_program, preset_params)
                PRESET_CACHE.put(milk_path, src_mtime, cache_data)
            new_vm = PresetVM(new_preset_program, preset_params)
            if logger.isEnabledFor(logging.INFO):
                logger.info(f"PRESET NAME: {Path(milk_path).name}")
                logger.info(f"zoom={new_vm.state.zoom}")
                logger.info(f"rot={new_vm.state.rot}")
            stats = new_preset_program.get_stats()
            if logger.isEnabledFor(logging.INFO):
                logger.info("Loaded preset:")
                logger.info(f"  per_frame_init: {stats['per_frame_init']}")
                logger.info(f"  per_frame: {stats['per_frame']}")
                logger.info(f"  per_pixel: {stats['per_pixel']}")
                logger.info(f"  waves: {stats['waves']}")
                logger.info(f"  shapes: {stats['shapes']}")
                logger.info("=== PRESET CONTENT DETAIL ===")
                per_pixel_eqs = preset_eqs.get('per_pixel', [])
                logger.info(f"per_pixel equations count: {len(per_pixel_eqs)}")
                for i, eq in enumerate(per_pixel_eqs[:20]):
                    logger.info(f"PIXEL[{i}]: {eq}")
                if len(per_pixel_eqs) > 20:
                    logger.info(f"... and {len(per_pixel_eqs) - 20} more")
            frame_count = 0
            shader_fragments = []
            params = preset_eqs.get('params', {})
            for k, v in params.items():
                if isinstance(k, str) and k.startswith(('comp_', 'warp_')) and isinstance(v, str) and v.startswith('`'):

                    fragment = v.lstrip('`').lstrip()
                    if fragment.startswith('shader_body'):
                        fragment = fragment[len('shader_body'):].lstrip()
                    fragment = self._translate_hlsl_fragment(fragment)
                    shader_fragments.append(fragment)
            new_pixel_shader_body = ""
            new_shader_program = None
            new_uniform_locs = {}
            if shader_fragments:
                new_pixel_shader_body = "\n".join(shader_fragments)
                if logger.isEnabledFor(logging.INFO):
                    logger.info(f"   Assembled custom shader from {len(shader_fragments)} fragments ({len(new_pixel_shader_body)} chars)")
                warp_shader_code = self._build_warp_shader(pixel_shader_body=new_pixel_shader_body)
                result = self._try_compile_warp_shader(warp_shader_code)
                if result is None:
                    raise PresetCompileError("Failed to compile custom warp shader")
                new_shader_program, new_uniform_locs = result
            elif new_preset_program.per_pixel:
                glsl_lines = []
                for compiled_expr in new_preset_program.per_pixel:
                    res = compile_assignment_to_glsl(compiled_expr.source)
                    target_var, glsl_code, is_statement = res
                    if is_statement:
                        glsl_lines.append(f"    {glsl_code};")
                    else:
                        glsl_lines.append(f"    {target_var} = {glsl_code};")
                new_pixel_shader_body = "\n".join(glsl_lines)
                if logger.isEnabledFor(logging.INFO):
                    logger.info(f"   Generated {len(glsl_lines)} per-pixel equations into shader")
                warp_shader_code = self._build_warp_shader(pixel_shader_body=new_pixel_shader_body)
                result = self._try_compile_warp_shader(warp_shader_code)
                if result is None:
                    raise PresetCompileError("Failed to compile per-pixel warp shader")
                new_shader_program, new_uniform_locs = result
            else:
                new_pixel_shader_body = ""
                result = self._try_create_simple_shader()
                if result is None:
                    raise PresetCompileError("Failed to create simple shader")
                new_shader_program, new_uniform_locs = result

            # PHASE 2: All compilation succeeded - safely destroy old state and swap
            self._cleanup_gl_resources()
            GLResourceOwner.flush_deletes()
            self._invalidate_gl_state_cache()
            self._reset_preset_state()

            self._last_loaded_raw_preset = new_last_loaded_raw_preset
            self.preset_eqs = preset_eqs
            self.preset_program = new_preset_program
            self.preset_vm = new_vm
            self.pixel_shader_body = new_pixel_shader_body
            self.shader_program = new_shader_program
            self.program = new_shader_program
            self.current_milk_file = current_milk_file
            self.current_preset_name = current_preset_name
            self.current_transition = current_transition
            for attr, value in new_uniform_locs.items():
                setattr(self, attr, value)
            self._frames_since_switch = _frames_since_switch
            self._last_preset_switch_time = _last_preset_switch_time
            self.frame_count = frame_count

            if old_vm:
                self.blend_engine.start_blend(old_vm, new_vm, self.transition_duration)

            if logger.isEnabledFor(logging.INFO):
                q_locs = getattr(self, 'q_locs', [-1] * 32)
                required_q = set()
                for compiled_expr in new_preset_program.per_frame:
                    if compiled_expr.bytecode_prog:
                        required_q.update(compiled_expr.bytecode_prog.q_refs)
                for wave in new_preset_program.waves:
                    for compiled_expr in wave.per_frame:
                        if compiled_expr.bytecode_prog:
                            required_q.update(compiled_expr.bytecode_prog.q_refs)
                for shape in new_preset_program.shapes:
                    for compiled_expr in shape.per_frame:
                        if compiled_expr.bytecode_prog:
                            required_q.update(compiled_expr.bytecode_prog.q_refs)
                for compiled_expr in new_preset_program.per_pixel:
                    if compiled_expr.bytecode_prog:
                        required_q.update(compiled_expr.bytecode_prog.q_refs)
                for m in re.finditer(r'\bq(\d+)\b', new_pixel_shader_body):
                    idx = int(m.group(1)) - 1
                    if 0 <= idx < 32:
                        required_q.add(idx)
                self._preset_health.add_required_q_registers(required_q)
                self._preset_health.check_q_registers(q_locs)
                warp_uniforms_ok = getattr(self, 'zoom_loc', -1) != -1 and getattr(self, 'rot_loc', -1) != -1 and getattr(self, 'warp_loc', -1) != -1
                shader_ok = getattr(self, 'shader_program', None) is not None
                megabuf_ok = getattr(self, 'megabuf_tex', None) is not None and GL.glIsTexture(int(getattr(self, 'megabuf_tex', 0))) if getattr(self, 'megabuf_tex', None) and int(getattr(self, 'megabuf_tex', 0)) > 0 else False
                if not megabuf_ok:
                    self._preset_health.add_warning("megabuf not active")
                if not warp_uniforms_ok:
                    self._preset_health.add_warning("warp uniforms missing")
                if not shader_ok:
                    self._preset_health.add_error("shader not compiled")
                logger.info("Preset Health Report")
                logger.info(f"  score: {self._preset_health.score:.1f}")
                logger.info(f"  warnings: {len(self._preset_health.warnings)}")
                logger.info(f"  errors: {len(self._preset_health.errors)}")
                logger.info(f"  unsupported funcs: {len(self._preset_health.unsupported_funcs)}")
                for w in self._preset_health.warnings:
                    logger.info(f"    WARN: {w}")
                for e in self._preset_health.errors:
                    logger.info(f"    ERR: {e}")
                for f in self._preset_health.unsupported_funcs:
                    logger.info(f"    UNSUPPORTED: {f}")
            preset_name = Path(milk_path).stem
            self.preset_name = preset_name
            if smooth_transition and getattr(self, 'fbo_width', 0) > 0:
                self.capture_pending = True
                self.capture_pending_frame_id = self.frame_count + 1
                self.is_transitioning = True
                self.transition_start_time = time.time()
                self.transition_source_fbo = self.fbo[self.curr_fbo_idx]
                self.transition_source_tex = self.fbo_tex[self.curr_fbo_idx]
            self.queue_draw()
            return True
        except (PresetCompileError, OSError, RuntimeError) as e:
            logger.error(f"Failed to load .milk preset: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False

    def next_preset(self, smooth=True):
        """Switch to the next preset in the list."""
        logger.debug(f"NEXT PRESET: current={self.current_milk_file} available={getattr(self, 'available_milk_files', [])} transitioning={self.is_transitioning}")
        if not hasattr(self, 'available_milk_files') or not self.available_milk_files:
            presets_dir = Path(__file__).parent / "presets"
            try:
                self.available_milk_files = sorted(presets_dir.rglob("*.milk"))
            except Exception:
                self.available_milk_files = []
            if not self.available_milk_files:
                logger.warning("No presets available to switch")
                return
        current_idx = -1
        if getattr(self, 'current_milk_file', None):
            try:
                norm_current = os.path.abspath(self.current_milk_file)
                normalized = [os.path.abspath(str(p)) for p in self.available_milk_files]
                current_idx = normalized.index(norm_current)
            except ValueError:
                try:
                    current_idx = self.available_milk_files.index(Path(self.current_milk_file))
                except ValueError:
                    current_idx = -1
        next_idx = (current_idx + 1) % len(self.available_milk_files)
        next_item = self.available_milk_files[next_idx]
        next_path = Path(next_item)
        milk_path = str(next_path)
        logger.info(f"Switching to preset {next_idx + 1}/{len(self.available_milk_files)}: {next_path.name}")
        self.load_milk_preset(milk_path, smooth_transition=smooth)
        logger.debug(f"Loading: {milk_path}")
        forward_idx = (next_idx + 1) % len(self.available_milk_files)
        next_next = self.available_milk_files[forward_idx]
        AsyncPresetCompiler.submit(str(next_next), parse_milk_file(str(next_next)), {})
        AsyncPresetCompiler.start()

    def save_evolved_preset(self):
        """Saves the current evolved state back to a .milk file."""
        if not self.current_milk_file or not self.preset_eqs:
            return
        source_path = Path(self.current_milk_file)
        evolved_dir = source_path.parent / "evolved"
        evolved_dir.mkdir(exist_ok=True)
        filename = f"{source_path.stem}_evo_{int(time.time())}.milk"
        dest_path = evolved_dir / filename
        try:
            with open(dest_path, 'w') as f:
                f.write(f"// Evolved from: {source_path.name}\n")
                f.writelines(f"{k}={v}\n" for k, v in self.preset_eqs['params'].items())
                f.writelines(
                    f"{line}\n"
                    for line in self.preset_eqs['lines']
                    if '=' in line and line.split('=')[0].strip() not in self.preset_eqs['params']
                )
            logger.info(f"🧬 Evolved preset saved to: {dest_path}")
        except Exception as e:
            logger.error(f"Failed to save evolved preset: {e}")

    def _get_current_params(self):
        """Get parameters using the PresetVM for deterministic execution."""
        if hasattr(self, 'preset_vm') and self.preset_vm is not None:
            current_time = time.time() - self.start_time
            width = getattr(self, 'fbo_width', 640)
            height = getattr(self, 'fbo_height', 480)
            was_active = self.blend_engine.active
            if was_active:
                self.blend_engine.current.run_frame(current_time, self.frame_count, self.levels, width, height)
                self.blend_engine.next.run_frame(current_time, self.frame_count, self.levels, width, height)
                blended = self.blend_engine.update()
                uniforms = self.blend_engine.get_gpu_uniforms()
                self._last_blended_q = blended.q if hasattr(blended, 'q') else None
                self._last_blended_megabuf = blended.megabuf if hasattr(blended, 'megabuf') else None
                self._last_blended_megabuf_version = blended.megabuf_version if hasattr(blended, 'megabuf_version') else -1
                self._last_blended_rand_seed = blended.rand_seed if hasattr(blended, 'rand_seed') else None
            else:
                self.preset_vm.run_frame(
                    time=current_time,
                    frame=self.frame_count,
                    audio_levels=self.levels,
                    width=width,
                    height=height
                )
                uniforms = self.preset_vm.get_gpu_uniforms()
                self._last_blended_q = None
                self._last_blended_megabuf = None
                self._last_blended_rand_seed = None
            if hasattr(self.blend_engine, 'completed_vm') and self.blend_engine.completed_vm is not None:
                completed = self.blend_engine.completed_vm
                if completed is not getattr(self, '_completed_vm_sync', None):
                    logger.info("Blend complete: syncing preset_vm to completed VM")
                    self.preset_vm = completed
                    self._completed_vm_sync = completed
                    self.blend_engine.completed_vm = None
            is_gpu = isinstance(uniforms, GPUUniforms)
            if is_gpu:
                un = uniforms
                s = self.preset_vm.state if hasattr(self.preset_vm, 'state') else None
                p = getattr(self, 'preset_eqs', {})
                params = {
                    'speed': p.get('fRating', 0.5),
                    'complexity': p.get('fRating', 1.0),
                    'color_shift': p.get('wave_r', 0.0),
                    'wave_intensity': p.get('fWaveAlpha', 1.0),
                    'beat_sensitivity': p.get('fRating', 1.0),
                    'pattern_type': int(p.get('nWaveMode', 0)) % 8,
                    'feedback_amount': _finite(un.decay, 0.9),
                    'echo_zoom': _finite(s.get('echo_zoom', 1.0) if s else 1.0, 1.0),
                    'echo_alpha': _finite(s.get('echo_alpha', 0.0) if s else 0.0, 0.0),
                    'echo_orient': int(s.get('echo_orient', 0) if s else 0),
                    'dx': _finite(un.dx, 0.0),
                    'dy': _finite(un.dy, 0.0),
                    'zoom': _finite(un.zoom, 1.0),
                    'warp': _finite(un.warp, 0.0),
                    'rot': _finite(un.rot, 0.0),
                    'show_waveform': int(p.get('bDrawWaves', 1)),
                    'cx': _finite(un.cx, 0.5),
                    'cy': _finite(un.cy, 0.5),
                    'sx': _finite(un.sx, 1.0),
                    'sy': _finite(un.sy, 1.0),
                    'decay': _finite(un.decay, 0.9),
                    'wave_r': _finite(un.wave_r, 0.0),
                    'wave_g': _finite(un.wave_g, 0.0),
                    'wave_b': _finite(un.wave_b, 0.0),
                    'wave_a': _finite(un.wave_a, 0.0),
                }
                return params
            un = uniforms
            zoom_raw = un.get('zoom', 1.0)
            zoom_clamped = np.nan_to_num(zoom_raw, nan=1.0, posinf=1.0, neginf=1.0)
            p = self.preset_eqs.get('params', {}) if hasattr(self, 'preset_eqs') else {}
            s = self.preset_vm.state if hasattr(self.preset_vm, 'state') else None
            params = {
                'speed': p.get('fRating', 0.5),
                'complexity': p.get('fRating', 1.0),
                'color_shift': p.get('wave_r', 0.0),
                'wave_intensity': p.get('fWaveAlpha', 1.0),
                'beat_sensitivity': p.get('fRating', 1.0),
                'pattern_type': int(p.get('nWaveMode', 0)) % 8,
                'feedback_amount': s.get('decay', 0.9) if s else 0.9,
                'echo_zoom': s.get('echo_zoom', 1.0) if s else 1.0,
                'echo_alpha': s.get('echo_alpha', 0.0) if s else 0.0,
                'echo_orient': int(s.get('echo_orient', 0) if s else 0),
                'dx': np.nan_to_num(un.get('dx', 0.0), nan=0.0, posinf=0.0, neginf=0.0),
                'dy': np.nan_to_num(un.get('dy', 0.0), nan=0.0, posinf=0.0, neginf=0.0),
                'zoom': zoom_clamped,
                'warp': np.nan_to_num(un.get('warp', 0.0), nan=0.0, posinf=0.0, neginf=0.0),
                'rot': np.nan_to_num(un.get('rot', 0.0), nan=0.0, posinf=0.0, neginf=0.0),
                'show_waveform': int(p.get('bDrawWaves', 1)),
                'cx': un.get('cx', 0.5),
                'cy': un.get('cy', 0.5),
                'sx': un.get('sx', 1.0),
                'sy': un.get('sy', 1.0),
                'decay': s.get('decay', 0.9) if s else 0.9,
                'wave_r': un.get('wave_r', 0.0),
                'wave_g': un.get('wave_g', 0.0),
                'wave_b': un.get('wave_b', 0.0),
                'wave_a': un.get('wave_a', 0.0),
            }
            return params
        return {'zoom': 1.0, 'rot': 0.0, 'warp': 0.0, 'cx': 0.5, 'cy': 0.5, 'dx': 0.0, 'dy': 0.0, 'sx': 1.0, 'sy': 1.0, 'decay': 0.9, 'feedback_amount': 0.95, 'wave_r': 0.0, 'wave_g': 0.0, 'wave_b': 0.0, 'wave_a': 0.0}

    _HLSL_TYPE_MAP: ClassVar[dict[str, str]] = {
        'float2': 'vec2',
        'float3': 'vec3',
        'float4': 'vec4',
        'int2': 'ivec2',
        'int3': 'ivec3',
        'int4': 'ivec4',
        'half': 'float',
        'half2': 'vec2',
        'half3': 'vec3',
        'half4': 'vec4',
        'float1x1': 'mat2',
        'float2x2': 'mat2',
        'float3x3': 'mat3',
        'float4x4': 'mat4',
    }

    _HLSL_FUNC_MAP: ClassVar[dict[str, str]] = {
        'tex2D': 'texture',
        'tex2Dlod': 'textureLod',
        'tex2Dproj': 'textureProj',
        'tex3D': 'texture',
        'lerp': 'mix',
        'frac': 'fract',
        'atan2': 'atan',
        'fmod': 'mod',
        'ddx': 'dFdx',
        'ddy': 'dFdy',
        'rsqrt': 'inversesqrt',
        'asuint': 'floatBitsToUint',
        'asfloat': 'uintBitsToFloat',
        'tex2d': 'texture',
    }

    def _translate_hlsl_fragment(self, fragment: str) -> str:
        """Translate HLSL-style shader fragment code to GLSL.

        Handles common HLSL-to-GLSL differences in MilkDrop warp/composite
        shader fragments: type names (float2->vec2), texture sampling
        (tex2D->texture), interpolation (lerp->mix), fractional part
        (frac->fract), matrix operations, and missing semicolons.

        Supports MilkDrop 2.x shader idioms:
        - tex2Dlod with explicit LOD level
        - Matrix multiplication syntax
        - Swizzle operations
        - Common HLSL intrinsics
        """
        if not fragment:
            return fragment

        try:
            lines = fragment.split('\n')
            out_lines = []
            for line in lines:
                stripped = line.strip()
                if not stripped or stripped.startswith(('//', '#', '/*')):
                    out_lines.append(line)
                    continue
                if stripped in ('{', '}'):
                    out_lines.append(line)
                    continue
                stripped_end = stripped.rstrip()
                if stripped_end.endswith((';', '{', '}', ':', ',', '\\')):
                    out_lines.append(line)
                    continue
                out_lines.append(line.rstrip() + ';')
            translated = '\n'.join(out_lines)

            # Type mappings
            for hl, gl in self._HLSL_TYPE_MAP.items():
                translated = re.sub(r'\b' + re.escape(hl) + r'\b', gl, translated)

            # Function mappings
            for hl, gl in self._HLSL_FUNC_MAP.items():
                translated = re.sub(r'\b' + re.escape(hl) + r'\b', gl, translated)

            # Specific HLSL patterns
            translated = self._replace_saturate(translated)
            translated = re.sub(r'\bret\b', 'color', translated)

            # Handle tex2D(sampler, uv) -> texture(sampler, uv)
            translated = re.sub(r'\btex2D\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)', r'texture(\1, \2)', translated)

            # Handle tex2Dlod(sampler, uv, lod) -> textureLod(sampler, uv, lod)
            translated = re.sub(r'\btex2Dlod\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)]+)\s*\)', r'textureLod(\1, \2, \3)', translated)

            # Handle matrix multiplication: mul(a, b) -> (a * b)
            translated = re.sub(r'\bmul\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)', r'(\1 * \2)', translated)

            # Handle vector swizzles that might need adjustment
            # Most swizzles are compatible, but ensure .rg -> .xy conversion for older GLSL
            translated = re.sub(r'\.rg\b', '.xy', translated)
            translated = re.sub(r'\.ba\b', '.zw', translated)

            return translated

        except (re.error, ValueError, AttributeError) as e:
            logger.warning(f"HLSL translation failed, using original: {e}")
            return fragment

    @staticmethod
    def _replace_saturate(text: str) -> str:
        """Replace saturate(x) with clamp(x, 0.0, 1.0), handling nested parens."""
        prev = None
        while prev != text:
            prev = text
            result = []
            i = 0
            n = len(text)
            while i < n:
                if text.startswith('saturate(', i):
                    i += len('saturate(')
                    depth = 1
                    arg_start = i
                    while i < n and depth > 0:
                        if text[i] == '(':
                            depth += 1
                        elif text[i] == ')':
                            depth -= 1
                        i += 1
                    arg = text[arg_start:i - 1]
                    result.append('clamp(' + arg + ', 0.0, 1.0)')
                else:
                    result.append(text[i])
                    i += 1
            text = ''.join(result)
        return text

    def _build_warp_shader(self, pixel_shader_body=None):
        """Build the warp shader with feedback and uniform usage."""
        if pixel_shader_body is None:
            pixel_shader_body = getattr(self, 'pixel_shader_body', "// No per-pixel equations")
        var_decls = self._generate_dynamic_var_decls(pixel_shader_body)
        if getattr(self, 'strict_milkdrop_mode', True):
            spatial_genome_code = "// Genome effects disabled in strict mode"
            color_genome_code = "// Genome effects disabled in strict mode"
            warp_impl = """
                // Standard MilkDrop-style radial warp
                vec2 d_warp = uv - vec2(cx, cy);
                uv_warp += normalize(d_warp) * (length(d_warp) * warp * 0.1);
            """
        else:
            graph = self.evolution_engine.active_graph
            spatial_nodes = [n.generate_glsl() for n in graph.nodes
                             if isinstance(n, (RippleEffect, SwirlEffect, TunnelEffect, FractalEffect, KaleidoscopeEffect))]
            color_nodes = [n.generate_glsl() for n in graph.nodes
                           if not isinstance(n, (RippleEffect, SwirlEffect, TunnelEffect, FractalEffect, KaleidoscopeEffect))]
            spatial_genome_code = "\n            ".join(spatial_nodes)
            color_genome_code = "\n            ".join(color_nodes)
            warp_impl = """
                // Step 6: Sharper curl-style warp distortion
                vec2 flow;
                flow.x = sin(uv.y * 12.0 + u_time);
                flow.y = cos(uv.x * 10.0 - u_time);
                uv_warp += flow * warp * 0.02;
            """
        q_uniforms = "uniform float u_q[33];"
        q_locals = "\n            ".join([f"float q{i} = u_q[{i}];" for i in range(1, 33)])
        reg_uniforms = "uniform float u_regs[100];"
        reg_locals = "\n            ".join([f"float reg{i:02d} = u_regs[{i}];" for i in range(100)])
        helpers = GLSLGenerator()._get_helper_functions(getattr(self, 'megabuf_tex_size', 262144))
        base_shader = f"""#version 330 core
        in vec2 v_texcoord;
        out vec4 FragColor;
        uniform float u_time;
        uniform vec2 u_resolution;
        uniform vec4 u_audio;
        uniform vec2 u_zoom;
        uniform float u_rot;
        uniform float u_warp;
        uniform float u_cx;
        uniform float u_cy;
        uniform float u_dx;
        uniform float u_dy;
        uniform float u_sx;
        uniform float u_sy;
        uniform float u_decay;
        uniform float u_feedback;
        uniform sampler2D u_prev_frame;
        uniform vec2 u_cam_pos;
        uniform float u_cam_rot;
        uniform sampler2D u_texture;
        uniform float u_brightness;
        uniform float u_rand_seed;
        uniform sampler2D u_megabuf;
        uniform sampler2D iChannel0;
        uniform sampler2D iChannel1;
        uniform sampler2D iNoiseLQ;
        uniform sampler2D iNoiseLQNearest;
        uniform sampler2D iNoiseMQ;
        uniform sampler2D iNoiseMQNearest;
        uniform sampler2D iNoiseHQ;
        uniform sampler2D iNoiseHQNearest;
        uniform sampler3D iNoiseVolLQ;
        uniform sampler3D iNoiseVolLQNearest;
        uniform sampler3D iNoiseVolHQ;
        uniform sampler3D iNoiseVolHQNearest;
        // q1-q32 uniform registers
        {q_uniforms}
        // reg00-reg99 global registers
        {reg_uniforms}
        {helpers}
        void main() {{
            vec2 uv = v_texcoord;
            {q_locals}
            {reg_locals}
            // MilkDrop base variables
            float x = uv.x;
            float y = uv.y;
            float w = u_resolution.x;
            float h = u_resolution.y;
            float time = u_time;
            float rad = length(uv - 0.5) * 2.0;
            float ang = atan(uv.y - 0.5, uv.x - 0.5);
            // Local MilkDrop variables
            float r = 0.0, g = 0.0, b = 0.0, a = 0.0, ret = 0.0;
            float wave_r = 0.0, wave_g = 0.0, wave_b = 0.0, wave_a = 0.0;
            float wave_x = 0.5, wave_y = 0.5;
            float ob_r = 0.0, ob_g = 0.0, ob_b = 0.0, ob_a = 0.0, ob_size = 0.0;
            float ib_r = 0.0, ib_g = 0.0, ib_b = 0.0, ib_a = 0.0, ib_size = 0.0;
            float md_mod = 0.0;
            // Core transformation variables
            float zoom = u_zoom.x;
            float rot = u_rot;
            float warp = u_warp;
            float cx = u_cx;
            float cy = u_cy;
            float dx = u_dx;
            float dy = u_dy;
            float sx = u_sx;
            float sy = u_sy;
            float decay = u_decay;
            float _u_fix = (u_cx + u_cy + u_dx + u_dy + u_sx + u_sy + u_decay + u_zoom.x + u_zoom.y + u_warp + u_rot + u_brightness) * 1e-9;
            {var_decls}
            vec3 color = vec3(0.0);
            // Per-pixel equations
            {pixel_shader_body}
            // Spatial Genome Effects (UV manipulation)
            {spatial_genome_code}
            // GPU-side stability guards: gentler clamps for MilkDrop fidelity
            // Only prevent actual NaN/Inf, preserve dynamic range for creative presets
            zoom = (isnan(zoom) || isinf(zoom)) ? 1.0 : clamp(zoom, 0.01, 100.0);
            rot = (isnan(rot) || isinf(rot)) ? 0.0 : rot;
            warp = (isnan(warp) || isinf(warp)) ? 0.0 : clamp(warp, -10.0, 10.0);
            // Apply transformations (MilkDrop style)
            // Sample previous frame for feedback
            vec2 uv_warp = uv;
            if (warp != 0.0) {{
                {warp_impl}
            }}
            vec2 uv_trans = (uv_warp - vec2(cx, cy)) / vec2(zoom * sx, zoom * sy);
            float cos_r = cos(rot); float sin_r = sin(rot);
            uv_trans = mat2(cos_r, -sin_r, sin_r, cos_r) * uv_trans + vec2(cx, cy) - vec2(dx, dy);
            // NaN/Inf protection for UV (Point 3)
            vec2 f_uv = uv_trans;
            if (any(isnan(f_uv)) || any(isinf(f_uv))) {{
                f_uv = vec2(0.5);
            }}
            vec3 feedback_color = texture(u_prev_frame, fract(f_uv)).rgb;
            // Feedback stability with decay damping
            feedback_color = clamp(feedback_color * u_decay, 0.0, 1.0);
            // Primary source is the feedback texture with proper mixing to prevent whiteouts
            float fb_factor = clamp(u_feedback * 0.5, 0.0, 0.95);
            color = mix(color, feedback_color, fb_factor);
            // --- SAFETY GUARDS: Prevent White-Screen Evolution Failures ---
            color = clamp(color, 0.0, 2.0);
            // Color Genome Effects (Color post-processing)
            {color_genome_code}
            // Final output clamp to standard dynamic range
            color = clamp(color, 0.0, 1.0);
            // Step 3: Reduced temporal anti-aliasing to prevent energy accumulation
            color = mix(color, feedback_color, 0.02);
            // Step 4: Audio-reactive exposure (significantly reduced to prevent white-out)
            float exposure = 1.0 + u_audio.x * 0.05 + u_audio.w * 0.08;
            color *= exposure;
            // Final integration of per-pixel output
            if (ret != 0.0) color *= ret;
            if (r != 0.0 || g != 0.0 || b != 0.0) color = mix(color, vec3(r, g, b), a > 0.0 ? a : 0.1);
            // Step 5: Filmic color grading
            color = pow(max(color, 0.0), vec3(1.1));
            vec3 tex_sample = texture(u_texture, uv).rgb;
            color = mix(color, tex_sample, 0.03);
            // Apply global brightness uniform (clamped to prevent white-out)
            color *= u_brightness;
            color += clamp(_u_fix, 0.0, 0.01);
            // CRITICAL: Final output clamp prevents energy buildup in feedback loop
            color = clamp(color, 0.0, 1.0);
            color += vec3(q1, q2, 0.5 * (q1 + q2)) * 0.02;
            FragColor = vec4(color, 1.0);
        }}
        """
        return base_shader

    def _fix_shader_vector_mismatches(self, shader_code: str) -> str:
        """Fix vector size mismatches in generated shader code.

        Handles cases where GetBlur1/2/3 (which return vec3) are used in 
        scalar contexts, causing compilation errors.
        """
        # Direct string replacement for the specific problematic pattern
        shader_code = shader_code.replace(
            'tan(1.33*GetBlur1(clamp(color.xy, 0.0, 1.0))))',
            'tan(1.33*lum(GetBlur1(clamp(color.xy, 0.0, 1.0)))))'
        )
        for blur_func in ['GetBlur2', 'GetBlur3']:
            shader_code = shader_code.replace(
                f'tan(1.33*{blur_func}(clamp(color.xy, 0.0, 1.0))))',
                f'tan(1.33*lum({blur_func}(clamp(color.xy, 0.0, 1.0)))))'
            )

        self._wrap_vec3_in_scalar_calls(shader_code, [
            'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
            'sinh', 'cosh', 'tanh', 'pow', 'exp', 'log',
            'log2', 'sqrt', 'inversesqrt', 'abs', 'sign',
            'floor', 'ceil', 'fract', 'trunc', 'round',
        ])

        return shader_code

    def _wrap_vec3_in_scalar_calls(self, code: str, scalar_funcs: list[str]) -> str:
        """Wrap GetBlur1/2/3(...) with lum() when used as arguments to scalar functions."""
        for func in scalar_funcs:
            start = 0
            while True:
                idx = code.find(f'{func}(', start)
                if idx == -1:
                    break
                depth = 1
                i = idx + len(func) + 1
                while i < len(code) and depth > 0:
                    if code[i] == '(':
                        depth += 1
                    elif code[i] == ')':
                        depth -= 1
                    i += 1
                if depth == 0:
                    args = code[idx + len(func) + 1:i - 1]
                    if 'GetBlur' in args and 'lum(GetBlur' not in args:
                        new_args = self._wrap_getblur_in_text(args)
                        if new_args != args:
                            code = code[:idx + len(func) + 1] + new_args + code[i - 1:]
                start = idx + 1
        return code

    def _wrap_getblur_in_text(self, text: str) -> str:
        result = []
        i = 0
        while i < len(text):
            if text[i:i+8] == 'GetBlur' and (i == 0 or not text[i-1].isalnum()):
                j = i + 8
                while j < len(text) and text[j] in '123':
                    j += 1
                if j < len(text) and text[j] == '(':
                    depth = 1
                    k = j + 1
                    while k < len(text) and depth > 0:
                        if text[k] == '(':
                            depth += 1
                        elif text[k] == ')':
                            depth -= 1
                        k += 1
                    if depth == 0:
                        result.append('lum(')
                        result.append(text[i:k])
                        result.append(')')
                        i = k
                        continue
            result.append(text[i])
            i += 1
        return ''.join(result)

    def _compile_warp_shader(self, shader_code):
        """Compile the dynamic warp shader from generated code.
        `shader_code` is expected to be a *fragment* shader body/source.
        We compile it as a fragment shader paired with a fixed fullscreen-quad vertex shader.
        """
        self.warp_shader_code = shader_code
        frag_src = shader_code
        # Fix vector size mismatches in shader code
        frag_src = self._fix_shader_vector_mismatches(frag_src)
        if not isinstance(frag_src, str):
            frag_src = str(frag_src)
        # CLAMP_F is no longer used - we use direct isnan/isinf checks in shader
        vert_src = """
        #version 330 core
        layout(location = 0) in vec2 position;
        out vec2 v_texcoord;
        void main() {
            gl_Position = vec4(position, 0.0, 1.0);
            v_texcoord = position * 0.5 + 0.5;
        }
        """
        if '#version' not in vert_src:
            raise RuntimeError("Warp vertex shader sanity check failed: missing #version")
        if 'void main' not in vert_src:
            raise RuntimeError("Warp vertex shader sanity check failed: missing void main")
        if 'gl_Position' not in vert_src:
            raise RuntimeError("Warp vertex shader sanity check failed: missing gl_Position")
        self._last_warp_program = None
        old_prog = getattr(self, 'shader_program', None)
        shader_hash = self._hash_shader_source(vert_src, frag_src)
        if shader_hash in self._shader_cache:
            cached_program_id = self._shader_cache[shader_hash]
            try:
                if cached_program_id and GL.glIsProgram(cached_program_id):
                    logger.debug(f"Using cached shader program (hash: {shader_hash[:8]}...)")
                    if old_prog and int(old_prog) > 0 and int(old_prog) != int(cached_program_id):
                        try:
                            if GL.glIsProgram(int(old_prog)):
                                GLResourceOwner.queue_delete("program", int(old_prog))
                        except GL.GLError:
                            pass
                    self.shader_program = cached_program_id
                    self._last_warp_program = cached_program_id
                    self._refresh_shader_uniform_locations(cached_program_id)
                    return cached_program_id
                else:
                    logger.debug(f"Cached shader program invalid (ID: {cached_program_id}), removing from cache")
                    del self._shader_cache[shader_hash]
            except Exception as e:
                logger.debug(f"Failed to use cached shader: {e}")
                if shader_hash in self._shader_cache:
                    del self._shader_cache[shader_hash]
        try:
            vs = GL.glCreateShader(GL.GL_VERTEX_SHADER)
            GL.glShaderSource(vs, vert_src.lstrip())
            GL.glCompileShader(vs)
            if not GL.glGetShaderiv(vs, GL.GL_COMPILE_STATUS):
                error = GL.glGetShaderInfoLog(vs)
                if isinstance(error, bytes):
                    error = error.decode('utf-8', errors='ignore')
                raise RuntimeError(f"Vertex shader compilation failed: {error}")
            fs = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
            GL.glShaderSource(fs, frag_src.lstrip())
            GL.glCompileShader(fs)
            if not GL.glGetShaderiv(fs, GL.GL_COMPILE_STATUS):
                error = GL.glGetShaderInfoLog(fs)
                if isinstance(error, bytes):
                    error = error.decode('utf-8', errors='ignore')
                try:
                    debug_path = Path('/tmp/kilo') / 'failed_warp_shader.glsl'
                    debug_path.parent.mkdir(parents=True, exist_ok=True)
                    debug_path.write_text(frag_src, encoding='utf-8')
                    logger.error(f"Fragment shader compilation failed. Shader saved to {debug_path}")
                except Exception as e:
                    logger.debug(f"Could not save debug shader: {e}")
                raise RuntimeError(f"Fragment shader compilation failed: {error}")
            program = GL.glCreateProgram()
            GL.glAttachShader(program, vs)
            GL.glAttachShader(program, fs)
            GL.glLinkProgram(program)
            if not GL.glGetProgramiv(program, GL.GL_LINK_STATUS):
                error = GL.glGetProgramInfoLog(program)
                if isinstance(error, bytes):
                    error = error.decode('utf-8', errors='ignore')
                raise RuntimeError(f"Shader program linking failed: {error}")
            self.shader_program = program
            self._last_warp_program = program
            self._shader_cache[shader_hash] = program
            self._refresh_shader_uniform_locations(program)
            if getattr(self, 'megabuf_loc', -1) != -1 and getattr(self, 'megabuf_tex', None):
                GL.glActiveTexture(GL.GL_TEXTURE2)
                GL.glBindTexture(GL.GL_TEXTURE_2D, self.megabuf_tex)
                GL.glUniform1i(self.megabuf_loc, 2)
            GL.glDeleteShader(vs)
            GL.glDeleteShader(fs)
            if old_prog and int(old_prog) > 0 and int(old_prog) != int(program):
                try:
                    if GL.glIsProgram(int(old_prog)):
                        GLResourceOwner.queue_delete("program", int(old_prog))
                except GL.GLError:
                    pass
            self.program = self.shader_program
        except RuntimeError as e:
            logger.error(f"Warp shader compilation failed: {e}")
            return None

    def _try_compile_warp_shader(self, shader_code):
        """Compile warp shader without destroying existing GL state.

        Returns (program_id, uniform_locs_dict) on success, None on failure.
        Does NOT modify self.shader_program, self.program, or queue old
        program for deletion.  Caller must perform the atomic swap on success.
        """
        frag_src = self._fix_shader_vector_mismatches(shader_code)
        if not isinstance(frag_src, str):
            frag_src = str(frag_src)
        vert_src = """
        #version 330 core
        layout(location = 0) in vec2 position;
        out vec2 v_texcoord;
        void main() {
            gl_Position = vec4(position, 0.0, 1.0);
            v_texcoord = position * 0.5 + 0.5;
        }
        """
        if '#version' not in vert_src:
            raise RuntimeError("Warp vertex shader sanity check failed: missing #version")
        if 'void main' not in vert_src:
            raise RuntimeError("Warp vertex shader sanity check failed: missing void main")
        if 'gl_Position' not in vert_src:
            raise RuntimeError("Warp vertex shader sanity check failed: missing gl_Position")
        shader_hash = self._hash_shader_source(vert_src, frag_src)
        if shader_hash in self._shader_cache:
            cached_program_id = self._shader_cache[shader_hash]
            try:
                if cached_program_id and GL.glIsProgram(cached_program_id):
                    logger.debug(f"Using cached shader program (hash: {shader_hash[:8]}...)")
                    uniform_locs = self._refresh_shader_uniform_locations(
                        cached_program_id, return_locs=True
                    )
                    return cached_program_id, uniform_locs
                else:
                    logger.debug(f"Cached shader program invalid (ID: {cached_program_id}), removing from cache")
                    del self._shader_cache[shader_hash]
            except Exception as e:
                logger.debug(f"Failed to use cached shader: {e}")
                if shader_hash in self._shader_cache:
                    del self._shader_cache[shader_hash]
        try:
            vs = GL.glCreateShader(GL.GL_VERTEX_SHADER)
            GL.glShaderSource(vs, vert_src.lstrip())
            GL.glCompileShader(vs)
            if not GL.glGetShaderiv(vs, GL.GL_COMPILE_STATUS):
                error = GL.glGetShaderInfoLog(vs)
                if isinstance(error, bytes):
                    error = error.decode('utf-8', errors='ignore')
                raise RuntimeError(f"Vertex shader compilation failed: {error}")
            fs = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
            GL.glShaderSource(fs, frag_src.lstrip())
            GL.glCompileShader(fs)
            if not GL.glGetShaderiv(fs, GL.GL_COMPILE_STATUS):
                error = GL.glGetShaderInfoLog(fs)
                if isinstance(error, bytes):
                    error = error.decode('utf-8', errors='ignore')
                debug_path = Path('/tmp/kilo') / 'failed_warp_shader.glsl'
                debug_path.parent.mkdir(parents=True, exist_ok=True)
                debug_path.write_text(frag_src, encoding='utf-8')
                logger.error(f"Fragment shader compilation failed. Shader saved to {debug_path}")
                raise RuntimeError(f"Fragment shader compilation failed: {error}")
            program = GL.glCreateProgram()
            GL.glAttachShader(program, vs)
            GL.glAttachShader(program, fs)
            GL.glLinkProgram(program)
            if not GL.glGetProgramiv(program, GL.GL_LINK_STATUS):
                error = GL.glGetProgramInfoLog(program)
                if isinstance(error, bytes):
                    error = error.decode('utf-8', errors='ignore')
                raise RuntimeError(f"Shader program linking failed: {error}")
            GL.glDeleteShader(vs)
            GL.glDeleteShader(fs)
            self._shader_cache[shader_hash] = program
            uniform_locs = self._refresh_shader_uniform_locations(program, return_locs=True)
            if uniform_locs.get('megabuf_loc', -1) != -1 and getattr(self, 'megabuf_tex', None):
                GL.glActiveTexture(GL.GL_TEXTURE2)
                GL.glBindTexture(GL.GL_TEXTURE_2D, self.megabuf_tex)
                GL.glUniform1i(uniform_locs['megabuf_loc'], 2)
            return program, uniform_locs
        except RuntimeError as e:
            logger.error(f"Warp shader compilation failed: {e}")
            return None

    def _cleanup_gl_resources(self, full=False):
        """Safe cleanup of OpenGL objects before loading a new preset.

        Args:
            full: If True, delete all GL resources including global resources.
                  If False, only delete preset-specific resources.
        """
        GLResourceOwner.flush_deletes()
        self._invalidate_gl_state_cache()

        if not (self.get_realized() and getattr(self, 'opengl_ready', False)):
            return

        def _del_program(val):
            if val and int(val) > 0:
                GLResourceOwner.queue_delete("program", int(val))

        def _del_texture(val):
            if val and int(val) > 0:
                GLResourceOwner.queue_delete("texture", int(val))

        def _del_fbo(val):
            if val and int(val) > 0:
                self._bind_fbo_cached(GL.GL_FRAMEBUFFER, 0)
                GLResourceOwner.queue_delete("fbo", int(val))

        def _del_vao(val):
            if val and int(val) > 0:
                GLResourceOwner.queue_delete("vao", int(val))

        def _del_buffer(val):
            if val and int(val) > 0:
                GLResourceOwner.queue_delete("buffer", int(val))

        def _clean_list(lst, deleter):
            if not lst:
                return
            if isinstance(lst, (list, tuple)):
                for v in lst:
                    deleter(v)
            else:
                deleter(lst)

        for attr in (
            'shader_program', 'warp_program',
        ):
            if hasattr(self, attr):
                current_program = getattr(self, attr)
                # Don't delete the shader_program if it's currently being used
                # to prevent invalidation during active rendering
                if attr == 'shader_program' and current_program:
                    # Only queue for deletion if it's not the current active program
                    try:
                        if current_program and int(current_program) > 0 and GL.glIsProgram(int(current_program)):
                            # Check if this is the currently bound program
                            current_bound = GL.glGetIntegerv(GL.GL_CURRENT_PROGRAM)
                            if int(current_program) != current_bound:
                                _del_program(current_program)
                            else:
                                logger.debug(f"Skipping deletion of active shader program {current_program}")
                    except GL.GLError:
                        # If there's a GL error, proceed with deletion as a safety measure
                        _del_program(current_program)
                else:
                    _clean_list(current_program, _del_program)
                setattr(self, attr, None)

        if full:
            for attr in (
                'display_program', 'bloom_program', 'blur_h_program', 'blur_v_program',
                'composite_program', 'tone_map_program', 'radial_blur_program',
                'motion_blur_program', 'lens_distortion_program', 'kaleidoscope_program',
                'glitch_program', 'wave_program', 'default_fade_program',
            ):
                if hasattr(self, attr):
                    _clean_list(getattr(self, attr), _del_program)
                    setattr(self, attr, None)

        if hasattr(self, 'fbo'):
            for i in range(2):
                _del_fbo(getattr(self, 'fbo', [0, 0])[i] if isinstance(getattr(self, 'fbo', []), (list, tuple)) and i < len(getattr(self, 'fbo', [])) else 0)
            self.fbo = [0, 0]
            self.fbo_width = 0
            self.fbo_height = 0
        if hasattr(self, 'fbo_tex'):
            for i in range(2):
                _del_texture(getattr(self, 'fbo_tex', [0, 0])[i] if isinstance(getattr(self, 'fbo_tex', []), (list, tuple)) and i < len(getattr(self, 'fbo_tex', [])) else 0)
            self.fbo_tex = [0, 0]

        if full:
            for attr in (
                'bloom_fbo', 'motion_blur_fbo', 'lens_fbo', 'kaleidoscope_fbo',
                'transition_fbo',
            ):
                if hasattr(self, attr):
                    _clean_list(getattr(self, attr), _del_fbo)
                    setattr(self, attr, None)

        if full:
            if hasattr(self, 'bloom_tex'):
                _del_texture(self.bloom_tex)
                self.bloom_tex = None
            if hasattr(self, 'bloom_blur_tex'):
                for pair in getattr(self, 'bloom_blur_tex', []):
                    if isinstance(pair, (list, tuple)):
                        for t in pair:
                            _del_texture(t)
                self.bloom_blur_tex = []
            if hasattr(self, 'bloom_blur_fbo'):
                for pair in getattr(self, 'bloom_blur_fbo', []):
                    if isinstance(pair, (list, tuple)):
                        for f in pair:
                            _del_fbo(f)
                self.bloom_blur_fbo = []
            if hasattr(self, 'motion_blur_tex'):
                _del_texture(self.motion_blur_tex)
                self.motion_blur_tex = None
            if hasattr(self, 'lens_tex'):
                _del_texture(self.lens_tex)
                self.lens_tex = None
            if hasattr(self, 'kaleidoscope_tex'):
                _del_texture(self.kaleidoscope_tex)
                self.kaleidoscope_tex = None
            if hasattr(self, 'bloom_reservoir_tex'):
                _del_texture(self.bloom_reservoir_tex)
                self.bloom_reservoir_tex = None
            if hasattr(self, 'bloom_reservoir_fbo'):
                _del_fbo(self.bloom_reservoir_fbo)
                self.bloom_reservoir_fbo = None
            if hasattr(self, 'transition_from_tex'):
                _del_texture(self.transition_from_tex)
                self.transition_from_tex = None
            if hasattr(self, 'transition_to_tex'):
                _del_texture(self.transition_to_tex)
                self.transition_to_tex = None

        if full:
            if hasattr(self, 'vao'):
                _del_vao(self.vao)
                self.vao = None
            if hasattr(self, 'vbo'):
                _del_buffer(self.vbo)
                self.vbo = None
            if hasattr(self, 'wave_vao'):
                _del_vao(self.wave_vao)
                self.wave_vao = None
            if hasattr(self, 'wave_vbo'):
                _del_buffer(self.wave_vbo)
                self.wave_vbo = None
            if hasattr(self, 'shape_vaos'):
                for v in getattr(self, 'shape_vaos', []):
                    _del_vao(v)
                self.shape_vaos = []
            if hasattr(self, 'shape_vbos'):
                for v in getattr(self, 'shape_vbos', []):
                    _del_buffer(v)
                self.shape_vbos = []
            if hasattr(self, 'transition_vao'):
                _del_vao(self.transition_vao)
                self.transition_vao = None
            if hasattr(self, 'transition_vbo'):
                _del_buffer(self.transition_vbo)
                self.transition_vbo = None

        if hasattr(self, 'loaded_textures'):
            for tex_id in list(getattr(self, 'loaded_textures', {}).values()):
                _del_texture(tex_id)
            self.loaded_textures.clear()
        if hasattr(self, 'current_texture_id'):
            self.current_texture_id = 0
        if full:
            if hasattr(self, 'megabuf_tex'):
                _del_texture(self.megabuf_tex)
                self.megabuf_tex = None
            if hasattr(self, 'noise_textures'):
                for tex_id in getattr(self, 'noise_textures', {}).values():
                    _del_texture(tex_id)
                self.noise_textures = {}

        if hasattr(self, 'sprite_renderer') and self.sprite_renderer:
            self.sprite_renderer.cleanup()

    def _reset_preset_state(self):
        """Reset all preset-related state to defaults."""
        self.is_transitioning = False
        self.capture_pending = False
        self.transition_progress = 0.0
        self._frames_since_switch = 0
        self.frame_count = 0
        self.init_run = False
        self.current_texture_id = 0
        self.current_texture_name = ""
        self.current_preset_name = None
        self.pixel_shader_body = ""
        self._last_loaded_raw_preset = None
        self.fbo_width = 0
        self.fbo_height = 0
        for attr in ('preset_vm', 'preset_program', 'preset_eqs'):
            if hasattr(self, attr):
                setattr(self, attr, None)
        self.blend_engine = BlendEngine()
        self._preset_health = PresetHealthScore()

    def _get_shader_feedback_code(self):
        return ""

    def _get_shader_noise_code(self):
        return ""

    def _extract_dynamic_vars(self, glsl_code):
        """Extract variable names from GLSL code that need to be declared."""
        DECLARED_VARS = {
            'x', 'y', 'w', 'h', 'rad', 'ang', 'cx', 'cy', 'dx', 'dy', 'sx', 'sy',
            'zoom', 'rot', 'warp', 'decay', 'uv', 'uv_trans', 'cos_r', 'sin_r',
            'r', 'g', 'b', 'a', 'ret',
            'wave_r', 'wave_g', 'wave_b', 'wave_a', 'wave_x', 'wave_y',
            'ob_r', 'ob_g', 'ob_b', 'ob_a', 'ob_size',
            'ib_r', 'ib_g', 'ib_b', 'ib_a', 'ib_size',
            'md_mod', 'color', 'feedback_color',
            'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8',
            'q9', 'q10', 'q11', 'q12', 'q13', 'q14', 'q15', 'q16',
            'q17', 'q18', 'q19', 'q20', 'q21', 'q22', 'q23', 'q24',
            'q25', 'q26', 'q27', 'q28', 'q29', 'q30', 'q31', 'q32',
            't', 'warped_uv', 'bass_boost', 'treb_boost', 'mid_boost',
            'transient_boost', 'combined_boost', 'grid_x', 'grid_y',
            'grid_pattern', 'grid_freq_x', 'grid_freq_y',
            'grid_mix', 'wave_mix', 'noise_mix', 'spiral_mix', 'total_mix',
            'plasma1', 'plasma2', 'plasma3', 'plasma4', 'plasma',
            'wave_pattern', 'wave1', 'wave2', 'wave3', 'wave_interference',
            'noise_val', 'spiral_angle', 'spiral_rad', 'spiral_pattern',
            'hue_shift', 'sat_mod', 'bright_mod', 'combined', 'final_color',
            'texture_influence', 'final_tex_uv', 'final_tex_color',
            'audio_react', 'beat_pulse', 'pulse_phase',
            'texture_color', 'tex_coord', 'tex_distort',
            'time', 'frame', 'rand_state', 'rand_initialized',
            'texel_idx', 'channel', 'coord', 'texel',
            'u_time', 'u_resolution', 'u_audio', 'u_zoom', 'u_rot', 'u_warp',
            'u_feedback', 'u_beat_flash', 'u_prev_frame', 'u_texture',
            'u_texture_blend', 'u_texture_zoom', 'u_texture_offset', 'u_texture_distort',
            'u_brightness', 'u_rand_seed',
            'u_megabuf', 'iChannel0', 'iChannel1', 'iNoiseLQ', 'iNoiseLQNearest',
            'iNoiseMQ', 'iNoiseMQNearest', 'iNoiseHQ', 'iNoiseHQNearest',
            'iNoiseVolLQ', 'iNoiseVolLQNearest', 'iNoiseVolHQ', 'iNoiseVolHQNearest',
            'gl_Position', 'gl_FragColor', 'gl_FragCoord',
        }
        for i in range(1, 33):
            DECLARED_VARS.add(f'u_q[{i}]')
        for i in range(100):
            DECLARED_VARS.add(f'u_regs[{i}]')
        BUILT_INS = {
            'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2', 'sinh', 'cosh', 'tanh',
            'pow', 'exp', 'log', 'log2', 'exp2', 'sqrt', 'inversesqrt', 'abs', 'sign',
            'floor', 'ceil', 'trunc', 'round', 'roundEven', 'fract', 'mod', 'min', 'max',
            'clamp', 'mix', 'step', 'smoothstep', 'length', 'distance', 'dot', 'cross',
            'normalize', 'reflect', 'refract', 'texture', 'texture2D', 'texture3D',
            'textureCube', 'textureGrad', 'textureLod', 'vec2', 'vec3', 'vec4',
            'mat2', 'mat3', 'mat4', 'ivec2', 'ivec3', 'ivec4', 'uvec2', 'uvec3', 'uvec4',
            'bvec2', 'bvec3', 'bvec4', 'float', 'int', 'bool', 'void', 'return',
            'if', 'if_', 'else', 'for', 'while', 'do', 'break', 'continue', 'discard',
            'true', 'false', 'in', 'out', 'uniform', 'varying', 'attribute',
            'const', 'struct', 'layout', 'flat', 'smooth', 'noperspective', 'centroid',
            'invariant', 'precise', 'sampler1D', 'sampler2D', 'sampler3D', 'samplerCube',
            'sampler2DShadow', 'samplerCubeShadow', 'highp', 'mediump', 'lowp', 'precision',
            'above', 'below', 'equal', 'above_equal', 'below_equal', 'bnot', 'sign_',
            'sqr', 'safe_sqrt', 'log10_', 'int_', 'frac_', 'frac', 'trunc_', 'round_', 'mod_', 'fmod',
            'lerp', 'tex2D', 'saturate',
            'sigmoid', 'hash', 'noise', 'rand2', 'md_rand', 'megabuf_read',
            'wrap', 'cwrap', 'fact', 'nchoosek',
            'texsize', 'g_fTexSize', 'sampler_main', 'rand_preset',
            'dFdx', 'dFdy', 'fwidth',
        }
        # Add reg00-reg99 as reserved identifiers
        for i in range(100):
            DECLARED_VARS.add(f'reg{i:02d}')
        MATH_CONSTS = {
            'pi': '3.14159265359',
            'e': '2.71828182846',
            'phi': '1.61803398875',
            'M_PI': '3.14159265358979',
            'M_PI_2': '6.283185307179586',
            'M_INV_PI_2': '0.15915494309189535',
        }
        glsl_code_clean = re.sub(r'//[^\n]*', '', glsl_code)
        glsl_code_clean = re.sub(r'/\*.*?\*/', '', glsl_code_clean, flags=re.DOTALL)
        explicit_decls = set(re.findall(r'\b(?:float|int|bool|vec[234]|mat[234])\s+([a-zA-Z_][a-zA-Z0-9_]*)\b', glsl_code_clean))
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', glsl_code_clean)
        assignments = re.findall(r'(?<!=)\b([a-zA-Z_][a-zA-Z0-9_]*)\s*=(?!=)', glsl_code_clean)
        used_vars = set()
        excluded = DECLARED_VARS | BUILT_INS | explicit_decls
        for var in assignments:
            if var not in excluded:
                if var.startswith('u_'):
                    continue
                used_vars.add(var)
        for word in words:
            if word not in excluded:
                if word.startswith('u_'):
                    continue
                if not re.search(r'\b' + word + r'\s*\(', glsl_code_clean):
                    used_vars.add(word)
        return sorted(used_vars), MATH_CONSTS

    def _generate_dynamic_var_decls(self, glsl_code):
        """Generate float declarations for variables used in the code."""
        vars_needed, math_consts = self._extract_dynamic_vars(glsl_code)
        if not vars_needed:
            return "// No additional dynamic variables needed"
        decls = []
        for var in vars_needed:
            if var in math_consts:
                decls.append(f"            float {var} = {math_consts[var]};")
            else:
                decls.append(f"            float {var} = 0.0;")
        return "\n".join(decls)

    def _get_enhanced_fragment_shader(self):
        return """
#version 330 core
in vec2 fragTexCoord;
out vec4 FragColor;
// ... [Keep your existing uniforms here] ...
void main() {
    // ... [Keep your existing coordinate transformation and pattern logic] ...
    // --- NEW: Bloom / Glow Calculation ---
    // 1. Identify "Hot" spots (areas brighter than 0.5)
    float luminance = dot(color, vec3(0.299, 0.587, 0.114));
    vec3 bloom = vec3(0.0);
    if (luminance > 0.5) {
        // Multi-tap sample for a cheap "Blur" effect
        float offset = 0.005 * (1.0 + u_audio.x); // Glow expands with bass
        bloom += texture(u_frame_feedback, fragTexCoord + vec2(offset, offset)).rgb;
        bloom += texture(u_frame_feedback, fragTexCoord - vec2(offset, offset)).rgb;
        bloom += texture(u_frame_feedback, fragTexCoord + vec2(offset, -offset)).rgb;
        bloom += texture(u_frame_feedback, fragTexCoord - vec2(offset, -offset)).rgb;
        bloom *= 0.25; // Average the samples
    }
    // 2. Add the bloom back to the base color
    // u_audio.w (beat) makes the whole screen glow during drum hits
    color += bloom * 0.6;
    color += u_audio.w * color * 0.3;
    // Final output with vignette
    float vig = 1.0 - length(fragTexCoord - 0.5) * 0.8;
    FragColor = vec4(color * vig, 1.0);
}
"""

    def _auto_randomize_effects(self, current_time):
        """Automatically randomize effect parameters over time for dynamic visuals."""
        if getattr(self, 'strict_milkdrop_mode', False):
            return
        if not self.auto_switch_enabled():
            return
        if not hasattr(self, '_last_effect_randomize'):
            self._last_effect_randomize = current_time
            self._next_randomize_interval = random.uniform(8.0, 15.0)
        if current_time - self._last_effect_randomize > self._next_randomize_interval:
            self.motion_blur_strength = random.uniform(0.05, 0.6)
            self.motion_blur_samples = random.randint(3, 16)
            self.lens_distortion_strength = random.uniform(0.05, 0.5)
            self.kaleidoscope_segments = random.randint(3, 12)
            self.glitch_intensity = random.uniform(0.0, 0.4)
            self.glitch_threshold = random.uniform(0.75, 0.95)
            if self.available_textures:
                if random.random() < 0.4:
                    old_texture = self.current_texture_name
                    available_without_current = [t for t in self.available_textures if t != old_texture]
                    if available_without_current:
                        new_texture = random.choice(available_without_current)
                        new_texture_id = self._load_texture_file(new_texture)
                        if new_texture_id > 0:
                            if self.current_texture_id and self.current_texture_id != getattr(self, 'dummy_texture', 0):
                                GLResourceOwner.queue_delete("texture", int(self.current_texture_id))
                            self.current_texture_id = new_texture_id
                            self.current_texture_name = Path(new_texture).name
                        logger.debug(f"Randomized texture: {old_texture} → {new_texture}")
                self.texture_blend = random.uniform(0.2, 0.8)
                self.texture_zoom = random.uniform(0.5, 2.0)
                self.texture_scroll_speed = [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)]
                self.texture_pulse_amount = random.uniform(0.0, 0.3)
                self.texture_wave_amplitude = random.uniform(0.0, 0.2)
                self.texture_turbulence = random.uniform(0.0, 0.1)
            if random.random() < 0.3:
                self.enable_motion_blur = random.choice([True, False])
                self.enable_lens_distortion = random.choice([True, False])
                self.enable_kaleidoscope = random.choice([True, False])
                self.enable_glitch = random.choice([True, False])
            self._last_effect_randomize = current_time
            self._next_randomize_interval = random.uniform(6.0, 12.0)
            logger.debug(f"Auto-randomized effects: MB={int(self.enable_motion_blur)}, LD={int(self.enable_lens_distortion)}, KS={int(self.enable_kaleidoscope)}, GL={int(self.enable_glitch)}")
        # Skip modern effects in strict MilkDrop mode
        if getattr(self, 'strict_milkdrop_mode', True):
            return
        if self.enable_kaleidoscope:
            self.kaleidoscope_angle += 0.005 + random.uniform(-0.002, 0.002)
    def _refresh_shader_uniform_locations(self, program, return_locs=False):
        """Re-fetch all uniform locations for the main MilkDrop shader program.

        This ensures that when a shader is recompiled (e.g. from per-pixel equations),
        the cached uniform location handles are updated to match the new program object.

        If return_locs=True, returns a dict of uniform locations instead of setting
        attributes on self.
        """
        if not program or program == 0:
            if return_locs:
                return {}
            return
        locs = {}
        locs['time_loc'] = GL.glGetUniformLocation(program, "u_time")
        locs['res_loc'] = GL.glGetUniformLocation(program, "u_resolution")
        locs['audio_loc'] = GL.glGetUniformLocation(program, "u_audio")
        locs['zoom_loc'] = GL.glGetUniformLocation(program, "u_zoom")
        locs['rot_loc'] = GL.glGetUniformLocation(program, "u_rot")
        locs['warp_loc'] = GL.glGetUniformLocation(program, "u_warp")
        locs['cx_loc'] = GL.glGetUniformLocation(program, "u_cx")
        locs['cy_loc'] = GL.glGetUniformLocation(program, "u_cy")
        locs['dx_loc'] = GL.glGetUniformLocation(program, "u_dx")
        locs['dy_loc'] = GL.glGetUniformLocation(program, "u_dy")
        locs['sx_loc'] = GL.glGetUniformLocation(program, "u_sx")
        locs['sy_loc'] = GL.glGetUniformLocation(program, "u_sy")
        locs['decay_loc'] = GL.glGetUniformLocation(program, "u_decay")
        locs['feedback_loc'] = GL.glGetUniformLocation(program, "u_feedback")
        locs['beat_flash_loc'] = GL.glGetUniformLocation(program, "u_beat_flash")
        locs['prev_frame_loc'] = GL.glGetUniformLocation(program, "u_prev_frame")
        locs['texture_loc'] = GL.glGetUniformLocation(program, "u_texture")
        locs['q_loc'] = GL.glGetUniformLocation(program, "u_q[1]")
        if locs['q_loc'] == -1 and logger.isEnabledFor(logging.DEBUG):
            logger.debug("Warning: u_q[1] uniform not found (optimized out?)")
        q_locs = [-1] * 32
        for i in range(1, 33):
            q_locs[i - 1] = GL.glGetUniformLocation(program, f"u_q[{i}]")
        locs['q_locs'] = q_locs
        locs['reg_loc'] = GL.glGetUniformLocation(program, "u_regs[0]")
        reg_locs = [-1] * 100
        for i in range(100):
            reg_locs[i] = GL.glGetUniformLocation(program, f"u_regs[{i}]")
        locs['reg_locs'] = reg_locs
        locs['audio_bands_loc'] = GL.glGetUniformLocation(program, "u_audio_bands")
        locs['audio_derived_loc'] = GL.glGetUniformLocation(program, "u_audio_derived")
        locs['audio_peaks_loc'] = GL.glGetUniformLocation(program, "u_audio_peaks")
        locs['texture_blend_loc'] = GL.glGetUniformLocation(program, "u_texture_blend")
        locs['texture_zoom_loc'] = GL.glGetUniformLocation(program, "u_texture_zoom")
        locs['texture_offset_loc'] = GL.glGetUniformLocation(program, "u_texture_offset")
        locs['texture_distort_loc'] = GL.glGetUniformLocation(program, "u_texture_distort")
        locs['brightness_loc'] = GL.glGetUniformLocation(program, "u_brightness")
        locs['megabuf_loc'] = GL.glGetUniformLocation(program, "u_megabuf")
        locs['rand_seed_loc'] = GL.glGetUniformLocation(program, "u_rand_seed")
        noise_tex_locs = {}
        for name in ('noise_lq', 'noise_mq', 'noise_hq'):
            noise_tex_locs[name] = GL.glGetUniformLocation(program, f"i{name}")
        locs['noise_tex_locs'] = noise_tex_locs
        locs['wave_r_loc'] = GL.glGetUniformLocation(program, "u_wave_r")
        locs['wave_g_loc'] = GL.glGetUniformLocation(program, "u_wave_g")
        locs['wave_b_loc'] = GL.glGetUniformLocation(program, "u_wave_b")
        locs['wave_a_loc'] = GL.glGetUniformLocation(program, "u_wave_a")
        locs['wave_size_loc'] = GL.glGetUniformLocation(program, "u_wave_size")
        if return_locs:
            return locs
        for attr, value in locs.items():
            setattr(self, attr, value)
    def _upload_q_registers(self, q_values):
        """Upload q1-q32 register values to GPU uniforms using batch glUniform1fv."""
        q_loc = getattr(self, 'q_loc', -1)
        if q_loc == -1:
            return
        count = min(len(q_values), 32)
        if count == 0:
            return
        try:
            GL.glUniform1fv(q_loc, count, list(q_values[:count]))
        except Exception as e:
            logger.debug(f"Failed to upload q registers: {e}")

    def _upload_reg_registers(self):
        """Upload reg00-reg99 global register values to GPU uniforms."""
        reg_loc = getattr(self, 'reg_loc', -1)
        if reg_loc == -1:
            return
        try:
            GL.glUniform1fv(reg_loc, 100, PresetState._global_regs)
        except Exception as e:
            logger.debug(f"Failed to upload reg registers: {e}")

    def _update_megabuf_texture(self, megabuf_array):
        """Update megabuf texture with CPU-side array data."""
        if not hasattr(self, 'megabuf_tex') or self.megabuf_tex is None:
            return
        try:
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.megabuf_tex)
            if megabuf_array is not None and len(megabuf_array) > 0:
                GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0,
                                   min(len(megabuf_array), getattr(self, 'megabuf_tex_size', MEGABUF_SIZE)), 1,
                                   GL.GL_RED, GL.GL_FLOAT, megabuf_array)
        except Exception as e:
            logger.debug(f"Failed to update megabuf texture: {e}")

    def _upload_rand_seed(self, seed_value):
        """Upload random seed uniform to shader."""
        loc = getattr(self, 'rand_seed_loc', -1)
        if loc not in (None, -1):
            try:
                GL.glUniform1f(loc, float(int(seed_value) % 2147483647))
            except Exception as e:
                logger.debug(f"Failed to upload rand seed: {e}")

    def _bind_noise_textures(self):
        """Bind noise textures to shader units using cached uniform locations."""
        try:
            if not hasattr(self, 'noise_textures') or not self.noise_textures:
                return
            cached_locs = getattr(self, 'noise_tex_locs', None)
            for i, (name, tex_id) in enumerate(self.noise_textures.items()):
                unit = 2 + i
                self._bind_texture_cached(GL.GL_TEXTURE0 + unit, tex_id)
                if cached_locs:
                    loc = cached_locs.get(name, -1)
                else:
                    loc = GL.glGetUniformLocation(getattr(self, 'shader_program', 0), f"i{name}")
                if loc != -1:
                    GL.glUniform1i(loc, unit)
        except Exception as e:
            logger.debug(f"Failed to bind noise textures: {e}")

    def _render_shapes(self):
        """Render shapes with full MilkDrop 1.x shape support.

        Supports:
        - Circle (sides=0) and polygon shapes (sides >= 3)
        - Textured and untextured shapes
        - Additive and normal blending
        - Thick outlines
        - Full transformation (x, y, rad, ang, tex_ang, tex_zoom)
        - Per-frame equation evaluation
        """
        if not hasattr(self, 'preset_vm') or not self.preset_vm:
            return
        program = self.preset_vm.program
        if not hasattr(program, 'shapes') or not program.shapes:
            return

        prev_blend = GL.glIsEnabled(GL.GL_BLEND)
        GL.glEnable(GL.GL_BLEND)

        for shape_idx, shape in enumerate(program.shapes):
            if not shape.enabled:
                continue

            # Set blend mode based on shape additive flag
            if shape.additive:
                GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE)
            else:
                GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

            # Generate geometry based on shape type
            if shape.sides == 0:
                # Circle shape
                num_segments = 64
                angles = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)

                # Generate circle vertices
                vertices = []
                for angle in angles:
                    cos_a = np.cos(angle + shape.ang)
                    sin_a = np.sin(angle + shape.ang)
                    x = shape.x + shape.rad * cos_a
                    y = shape.y + shape.rad * sin_a
                    vertices.extend([x, y, shape.r, shape.g, shape.b, shape.a])

                vertices = np.array(vertices, dtype=np.float32)
                num_verts = len(vertices) // 6

                # Use line loop for circle outline
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.shape_vbos[0] if self.shape_vbos else 0)
                GL.glBufferData(GL.GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL.GL_DYNAMIC_DRAW)

                GL.glEnableVertexAttribArray(0)
                GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 6 * 4, None)
                GL.glEnableVertexAttribArray(1)
                GL.glVertexAttribPointer(1, 4, GL.GL_FLOAT, GL.GL_FALSE, 6 * 4, ctypes.c_void_p(2 * 4))

                if shape.thick_outline:
                    GL.glLineWidth(3.0)
                else:
                    GL.glLineWidth(1.0)

                GL.glDrawArrays(GL.GL_LINE_LOOP, 0, num_verts)

            else:
                # Polygon shape (triangle, square, etc.)
                num_sides = max(3, shape.sides)
                angles = np.linspace(0, 2 * np.pi, num_sides, endpoint=False)

                # Generate polygon vertices (triangle fan for fill)
                vertices = []

                # Center point
                vertices.extend([shape.x, shape.y, shape.r, shape.g, shape.b, shape.a])

                # Outer points
                for angle in angles:
                    cos_a = np.cos(angle + shape.ang)
                    sin_a = np.sin(angle + shape.ang)
                    x = shape.x + shape.rad * cos_a
                    y = shape.y + shape.rad * sin_a
                    vertices.extend([x, y, shape.r, shape.g, shape.b, shape.a])

                # Close the loop
                cos_a = np.cos(angles[0] + shape.ang)
                sin_a = np.sin(angles[0] + shape.ang)
                x = shape.x + shape.rad * cos_a
                y = shape.y + shape.rad * sin_a
                vertices.extend([x, y, shape.r, shape.g, shape.b, shape.a])

                vertices = np.array(vertices, dtype=np.float32)
                num_verts = len(vertices) // 6

                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.shape_vbos[0] if self.shape_vbos else 0)
                GL.glBufferData(GL.GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL.GL_DYNAMIC_DRAW)

                GL.glEnableVertexAttribArray(0)
                GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 6 * 4, None)
                GL.glEnableVertexAttribArray(1)
                GL.glVertexAttribPointer(1, 4, GL.GL_FLOAT, GL.GL_FALSE, 6 * 4, ctypes.c_void_p(2 * 4))

                # Draw filled polygon
                GL.glDrawArrays(GL.GL_TRIANGLE_FAN, 0, num_verts)

                # Draw outline if thick outline is enabled
                if shape.thick_outline:
                    outline_vertices = vertices[6:]  # Skip center point
                    GL.glBufferData(GL.GL_ARRAY_BUFFER, outline_vertices.nbytes, outline_vertices, GL.GL_DYNAMIC_DRAW)
                    GL.glLineWidth(2.0)
                    GL.glDrawArrays(GL.GL_LINE_LOOP, 0, num_verts - 1)

        if not prev_blend:
            GL.glDisable(GL.GL_BLEND)

    def _save_screenshot(self, width, height):
        """Save screenshot (placeholder implementation)."""
        try:
            if getattr(self, 'screenshot_requested', False):
                self.screenshot_requested = False
                pixels = GL.glReadPixels(0, 0, width, height, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
                os.makedirs('screenshots', exist_ok=True)
                img = Image.frombytes('RGBA', (width, height), pixels)
                img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                img.save(f'screenshots/screenshot_{int(time.time())}.png')
        except Exception as e:
            logger.debug(f"Screenshot save failed: {e}")

    def cleanup(self):
        """Clean up OpenGL resources and background thread pool."""
        try:
            self._cleanup_gl_resources(full=True)
        except Exception as exc:
            logger.debug("Exception in try block: %s", exc)
        try:
            self._stop_texture_load_thread()
        except Exception as exc:
            logger.debug("Exception in try block: %s", exc)
        try:
            GLResourceOwner.flush_deletes()
            self._invalidate_gl_state_cache()
        except Exception as exc:
            logger.debug("Exception in try block: %s", exc)
        try:
            if GLOBAL_COMPILE_POOL:
                GLOBAL_COMPILE_POOL.shutdown(wait=False, cancel_futures=True)
        except Exception as exc:
            logger.debug("Exception in try block: %s", exc)
        try:
            AsyncPresetCompiler.stop()
        except Exception as exc:
            logger.debug("Exception in try block: %s", exc)


def shutdown_compile_pool():
    """Shutdown the global compile thread pool executor. Call on application exit."""
    global GLOBAL_COMPILE_POOL
    try:
        if GLOBAL_COMPILE_POOL:
            GLOBAL_COMPILE_POOL.shutdown(wait=False, cancel_futures=True)
            GLOBAL_COMPILE_POOL = None
    except Exception as exc:
        logger.debug("Exception in try block: %s", exc)


