from __future__ import annotations

import copy
import ctypes
import json
import logging
import random
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from OpenGL import GL

from gl_state import (
    GLResourceOwner,
    GLStateManager,
    _compile_gl_program,
    _gl_state_guard,
)

from .config import (
    MAX_INIT_LOOP_ITERATIONS,
)
from .presets import (
    MilkPreset,
    PresetVM,
)
from .state import (
    PresetState,
)
from .textures import (
    AsyncTextureLoadMixin,
    DecodedTexture,
    TextureLRU,
)

if TYPE_CHECKING:
    from .core import MilkdropRenderer

logger = logging.getLogger("milkdrop_renderer")


# --- User sprite rendering ------------------------------------------
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
        self.texture_id: int | None = None
        self.image_width = 0
        self.image_height = 0
        self.init_code: list = []
        self.frame_code: list = []
        self.tex_code: list = []
        self.init_run = False
        self.user_vars: dict[str, float] = {}
        self._init_time = 0.0
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
        self._init_time = state.vars.get("time", 0.0)
        init_state = state.shallow_clone()
        self._sync_to_state(init_state)
        init_ok = True
        for expr in self.init_code:
            try:
                expr.eval(init_state, max_iterations=MAX_INIT_LOOP_ITERATIONS)
            except Exception as e:
                logger.exception(
                    "Sprite init_code expression failed: sprite_id=%s expr=%s",
                    self.sprite_id,
                    getattr(expr, "source", "<unknown>"),
                    exc_info=e,
                )
                expr.runtime_failures += 1
                if expr.runtime_failures >= 3:
                    expr.bytecode_prog = None
                init_ok = False
        if init_ok:
            for expr in self.init_code:
                expr.runtime_failures = 0
            self._sync_from_state(init_state)
            self.init_run = True

    def run_frame(self, state: PresetState):
        """Run per-frame code_ equations."""
        if not self.active:
            return
        self._sync_to_state(state)
        state.vars["sprite"] = int(self.sprite_id) if self.sprite_id.isdigit() else 0
        if self.init_run:
            current_time = state.vars.get("time", 0.0)
            state.vars["img_timer"] = max(0.0, current_time - self._init_time)
        for expr in self.frame_code:
            try:
                expr.eval(state)
                expr.runtime_failures = 0
            except Exception as e:
                logger.exception(
                    "Sprite frame_code expression failed: sprite_id=%s expr=%s",
                    self.sprite_id,
                    getattr(expr, "source", "<unknown>"),
                    exc_info=e,
                )
                expr.runtime_failures += 1
                if expr.runtime_failures >= 3:
                    expr.bytecode_prog = None

        for expr in self.tex_code:
            try:
                expr.eval(state)
                expr.runtime_failures = 0
            except Exception as e:
                logger.exception(
                    "Sprite tex_code expression failed: sprite_id=%s expr=%s",
                    self.sprite_id,
                    getattr(expr, "source", "<unknown>"),
                    exc_info=e,
                )
                expr.runtime_failures += 1
                if expr.runtime_failures >= 3:
                    expr.bytecode_prog = None

        self._sync_from_state(state)

    def _sync_to_state(self, state: PresetState):
        """Copy sprite state to PresetState for equation evaluation."""
        for var in ["x", "y", "sx", "sy", "rot", "a", "r", "g", "b", "tex_u", "tex_v"]:
            state.vars[var] = getattr(self, var)
        state.vars["cx"] = self.x
        state.vars["cy"] = self.y
        for key, val in self.user_vars.items():
            state.vars[key] = val

    def _sync_from_state(self, state: PresetState):
        """Copy PresetState back to sprite after equation evaluation."""
        for var in ["x", "y", "sx", "sy", "rot", "a", "r", "g", "b", "tex_u", "tex_v"]:
            value = state.vars.get(var)
            if value is None:
                value = getattr(self, var)
            setattr(self, var, float(value))
        for key in self.user_vars:
            if key in state.vars:
                self.user_vars[key] = state.vars[key]
        # Preserve new vars created by equations.
        known = {"x", "y", "sx", "sy", "rot", "a", "r", "g", "b", "tex_u", "tex_v"}
        self.user_vars.update({k: v for k, v in state.vars.items() if k not in known and k not in self.user_vars})


class SpriteConfig:
    """Configuration for a sprite from milk_img.ini."""

    def __init__(self, sprite_id: str):
        self.sprite_id = sprite_id
        self.img_path = ""
        self.init_exprs: list[str] = []
        self.frame_exprs: list[str] = []
        self.tex_exprs: list[str] = []
        self.blend_mode = 0
        self.enabled = True

    def to_sprite(self, compile_expr=None) -> Sprite:
        """Turn config into a live Sprite instance without duplicating wiring logic."""
        sprite = Sprite(self.sprite_id, self.img_path)

        if compile_expr is not None:
            for expr_str in self.init_exprs:
                try:
                    expr = compile_expr(expr_str)
                    if expr is not None:
                        sprite.init_code.append(expr)
                except Exception as e:  # noqa: BLE001
                    logger.debug(
                        "Sprite init expr compile failed: sprite_id=%s err=%s",
                        self.sprite_id,
                        e,
                    )
            for expr_str in self.frame_exprs:
                try:
                    expr = compile_expr(expr_str)
                    if expr is not None:
                        sprite.frame_code.append(expr)
                except Exception as e:  # noqa: BLE001
                    logger.debug(
                        "Sprite frame expr compile failed: sprite_id=%s err=%s",
                        self.sprite_id,
                        e,
                    )
            for expr_str in self.tex_exprs:
                try:
                    expr = compile_expr(expr_str)
                    if expr is not None:
                        sprite.tex_code.append(expr)
                except Exception as e:  # noqa: BLE001
                    logger.debug(
                        "Sprite tex expr compile failed: sprite_id=%s err=%s",
                        self.sprite_id,
                        e,
                    )
        else:
            # If no compiler provided, store raw strings for later compilation
            sprite.init_code = list(self.init_exprs)
            sprite.frame_code = list(self.frame_exprs)
            sprite.tex_code = list(self.tex_exprs)

        sprite.blend_mode = int(self.blend_mode) if self.blend_mode is not None else 0
        sprite.active = bool(self.enabled)
        return sprite


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

    def clear(self):
        """Clear all parsed sprites."""
        self.sprites.clear()
        self.config_path = None

    def load(self, path: str | Path) -> dict[str, SpriteConfig]:
        """Load and parse a milk_img.ini file. Returns parsed sprites dict."""
        self.clear()
        self.config_path = Path(path)
        return self.parse(self.config_path)

    def parse(self, path: str | Path) -> dict[str, SpriteConfig]:
        """Parse a milk_img.ini file."""
        self.sprites = {}
        config_file = Path(path)
        if not config_file.exists():
            logger.debug("Sprite config not found: %s", config_file)
            return self.sprites
        if not config_file.is_file():
            logger.debug("Sprite config not a file: %s", config_file)
            return self.sprites

        try:
            with open(config_file, encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to read sprite config %s: %s", config_file, e)
            return self.sprites

        current_id = None
        current_cfg: SpriteConfig | None = None

        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith((";", "#")):
                continue
            # Section header
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1].strip()
                # Normalise numeric section ids to two digits
                if section.isdigit():
                    current_id = section.zfill(2)
                else:
                    # Try to extract digits
                    import re

                    m = re.match(r"(\d+)", section)
                    if m:
                        current_id = m.group(1).zfill(2)
                    else:
                        current_id = section[:2].zfill(2) if section else None
                if current_id is not None:
                    current_cfg = SpriteConfig(current_id)
                    self.sprites[current_id] = current_cfg
                else:
                    current_cfg = None
                continue

            if "=" not in line or current_cfg is None:
                continue

            key, value = line.split("=", 1)
            key = key.strip().lower()
            value = value.strip()

            if key == "img":
                current_cfg.img_path = value
            elif key.startswith("init_"):
                # init_x, init_y etc as init expressions
                current_cfg.init_exprs.append(f"{key[5:]}={value}")
            elif key.startswith("code_"):
                # code_sx style
                current_cfg.frame_exprs.append(f"{key[5:]}={value}")
            elif key.startswith("tex_"):
                current_cfg.tex_exprs.append(f"{key[4:]}={value}")
            elif key in ("blend", "blendmode", "blend_mode"):
                try:
                    current_cfg.blend_mode = int(float(value))
                except Exception:  # noqa: BLE001, S110
                    pass
            elif key in ("enabled", "enable"):
                v = value.lower()
                current_cfg.enabled = v not in ("0", "false", "no", "off")
            elif key in ("init",):
                if value:
                    current_cfg.init_exprs.append(value)
            elif key in ("code", "frame"):
                if value:
                    current_cfg.frame_exprs.append(value)
            elif key in ("tex",) and value:
                current_cfg.tex_exprs.append(value)

        logger.info("Parsed sprite config %s: %d sprites", config_file, len(self.sprites))
        return self.sprites


class SpriteRenderer(AsyncTextureLoadMixin):
    """OpenGL renderer for MilkDrop user sprites.
    Handles texture loading, sprite quad rendering with transforms,
    and proper blending modes.

    The renderer is *stateless* with respect to OpenGL state: every call
    that touches ``glEnable(GL_BLEND)``, ``glUseProgram``,
    ``glBindVertexArray``, ``glBindTexture`` or ``glBlendFunc`` goes
    through the supplied :class:`GLStateManager` so the renderer's cache
    of GL state stays coherent with the actual driver state. Restoring
    raw GL bindings here used to desynchronise the manager and cause
    intermittent visual bugs.
    """

    def __init__(
        self,
        gl_state: GLStateManager | None = None,
        create_gl_texture_func: Callable | None = None,
    ):
        self.sprites: dict[str, Sprite] = {}
        self.config_parser = SpriteConfigParser()
        self.texture_enhancement_enabled = False
        _owner = getattr(gl_state, "_owner", None) if gl_state is not None else None
        self._owner = _owner
        self.loaded_textures = TextureLRU(maxsize=256, owner=_owner)
        self.sprite_shader = None
        self.vao = None
        self.vbo = None
        self._gl_state = gl_state
        self._create_gl_texture_func = create_gl_texture_func
        self._sprite_texture_reload_queue: list[tuple] = []
        self._sprite_texture_reload_pending: dict[str, list] = {}
        self._init_async_texture_loader()
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
            prog_id = _compile_gl_program(
                self.vertex_shader, self.fragment_shader, name="SpriteRenderer"
            )
            if prog_id == 0:
                return
            self.sprite_shader = prog_id
            self.loc_pos = GL.glGetUniformLocation(self.sprite_shader, "u_sprite_pos")
            self.loc_scale = GL.glGetUniformLocation(
                self.sprite_shader, "u_sprite_scale"
            )
            self.loc_rot = GL.glGetUniformLocation(self.sprite_shader, "u_sprite_rot")
            self.loc_res = GL.glGetUniformLocation(self.sprite_shader, "u_resolution")
            self.loc_color = GL.glGetUniformLocation(self.sprite_shader, "u_color")
            self.loc_blend = GL.glGetUniformLocation(self.sprite_shader, "u_blend_mode")
            self.loc_tex = GL.glGetUniformLocation(self.sprite_shader, "u_texture")
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to compile sprite shader: {e}")

    def _create_quad(self):
        """Create VAO/VBO for a fullscreen quad (sprites are rendered as quads)."""
        gl_state = self._gl_state
        if gl_state is None:
            logger.error("Cannot create sprite quad without a GL state manager")
            return
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
                1.0,
                1.0,
                1.0,
                1.0,
                -1.0,
                1.0,
                0.0,
                1.0,
            ],
            dtype=np.float32,
        )
        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
        self.vao = int(GL.glGenVertexArrays(1))
        self.vbo = int(GL.glGenBuffers(1))
        self.ebo = int(GL.glGenBuffers(1))
        gl_state.bind_vertex_array(self.vao)
        gl_state.bind_array_buffer(self.vbo)
        # gl-state-guard: inline-approved — resource creation (quad VBO).
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL.GL_STATIC_DRAW
        )
        gl_state.bind_element_array_buffer(self.ebo)
        # gl-state-guard: inline-approved — resource creation (quad EBO).
        GL.glBufferData(
            GL.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL.GL_STATIC_DRAW
        )
        # gl-state-guard: inline-approved — resource creation (quad VAO attribs).
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 4 * 4, None)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(
            1, 2, GL.GL_FLOAT, GL.GL_FALSE, 4 * 4, ctypes.c_void_p(2 * 4)
        )
        GL.glEnableVertexAttribArray(1)
        gl_state.bind_vertex_array(0)
        gl_state.bind_array_buffer(0)

    def _upload_one_decoded_texture(
        self, result: DecodedTexture, replace_existing: bool = False
    ) -> bool:
        """Upload a single decoded texture to GL. GL thread only."""
        if not getattr(self, "opengl_ready", False) and replace_existing:
            return False
        path = result.get("path")
        image = result.get("image")
        if not path or image is None:
            return False
        old_tex = self.loaded_textures.get(path) if replace_existing else None
        if self._create_gl_texture_func is None:
            logger.debug("No _create_gl_texture_func; dropping decoded %s", path)
            return False
        try:
            img_data = np.array(image)
            img_data = np.flipud(img_data)
            tex_id = self._create_gl_texture_func(
                img_data,
                image.width,
                image.height,
                GL.GL_RGBA8,
                GL.GL_RGBA,
                texture_path=path,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Texture upload error for %s: %s", path, exc)
            return False
        if old_tex is not None and old_tex != int(tex_id) and self._owner is not None:
            try:
                self._owner.queue_delete("texture", old_tex)
            except Exception:  # noqa: BLE001, S110
                pass
        self.loaded_textures[path] = int(tex_id)
        self.loaded_texture_sizes[path] = (image.width, image.height)
        self._mark_texture_uploaded(path)
        return True

    def _upload_decoded_textures(self) -> None:
        """Upload textures decoded by the worker thread.  GL thread only.

        The worker only decodes; ``_create_gl_texture_func`` (and therefore
        every GL call) stays on the thread that owns the context.
        """
        results = self._drain_texture_load_results()
        if not results:
            return
        for result in results:
            path = result["path"]
            if path in self.loaded_textures:
                self._mark_texture_uploaded(path)
                continue
            self._upload_one_decoded_texture(result, replace_existing=False)

    def _process_texture_load_queue(self) -> None:
        """Drain decoded textures, upload them to GL, release replaced ones."""
        if not getattr(self, "opengl_ready", False):
            return
        results = self._drain_texture_load_results()
        if not results:
            return
        for result in results:
            self._upload_one_decoded_texture(result, replace_existing=True)

    def _process_sprite_texture_reloads(self):
        """Process queued sprite texture reloads (must be called on main thread)."""
        self._upload_decoded_textures()
        # Drain any pending entries whose textures have since become available.
        for path in list(self._sprite_texture_reload_pending.keys()):
            if path in self.loaded_textures:
                tex_id = self.loaded_textures[path]
                for spr in self._sprite_texture_reload_pending.pop(path):
                    spr.texture_id = tex_id
        if not self._sprite_texture_reload_queue:
            return
        for sprite, image_path in self._sprite_texture_reload_queue:
            if image_path in self.loaded_textures:
                sprite.texture_id = self.loaded_textures[image_path]
                logger.info(f"Reloaded sprite texture for [{sprite.sprite_id}]")
            else:
                with self._texture_load_lock:
                    if (
                        image_path not in self._pending_texture_uploads
                        and image_path not in self._texture_load_inflight
                        and image_path not in self._texture_load_queue
                    ):
                        self._texture_load_queue.append(image_path)
                    if image_path not in self._sprite_texture_reload_pending:
                        self._sprite_texture_reload_pending[image_path] = []
                    self._sprite_texture_reload_pending[image_path].append(sprite)
                logger.debug(f"Sprite texture queued for async reload: {image_path}")
        self._sprite_texture_reload_queue.clear()

    def cleanup(self):
        """Clean up OpenGL resources."""
        owner = self._owner
        if owner is None:
            # Without an owner there is no deletion queue to hand the GL
            # objects to; drop the local references and leave the driver-side
            # objects to the context teardown.
            self.loaded_textures.clear()
            self.loaded_texture_sizes.clear()
            self.sprite_shader = None
            self.vao = None
            self.vbo = None
            self.ebo = None
            return
        for tex_id in list(self.loaded_textures.values()):
            owner.queue_delete("texture", int(tex_id))
        self.loaded_textures.clear()
        self.loaded_texture_sizes.clear()
        if self.sprite_shader:
            owner.queue_delete("program", int(self.sprite_shader))
            self.sprite_shader = None
        if self.vao:
            owner.queue_delete("vao", int(self.vao))
            self.vao = None
        if self.vbo:
            owner.queue_delete("buffer", int(self.vbo))
            self.vbo = None
        if hasattr(self, "ebo") and self.ebo:
            owner.queue_delete("buffer", int(self.ebo))
            self.ebo = None


# --- Effect graph -------------------------------------------------
@dataclass
class Connection:
    from_node: str
    to_node: str

    def serialize(self) -> dict:
        return {"from": self.from_node, "to": self.to_node}

    def copy(self):
        return Connection(self.from_node, self.to_node)


class EffectNode:
    """Phase 1: Base class for autonomous effect nodes."""

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.params = {"strength": 1.0, "audio_factor": 0.5}
        self.audio_target = random.choice(["bass", "mid", "treb", "beat"])

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
        return {
            "type": self.__class__.__name__,
            "params": self.params,
            "audio": self.audio_target,
        }

    def copy(self):
        new_node = self.__class__(self.node_id)
        new_node.params = self.params.copy()
        new_node.audio_target = self.audio_target
        return new_node


class PresetDNAEffect(EffectNode):
    """Captures the visual DNA of a MilkDrop preset as an effect node.

    Stores the preset's key evolvable parameters (zoom, rot, warp, sx, sy,
    decay, wave colors) so the evolution engine can track which presets
    produced visually interesting results and breed from high-fitness seeds.
    """

    _DNA_PARAMS = (
        "zoom",
        "rot",
        "warp",
        "cx",
        "cy",
        "dx",
        "dy",
        "sx",
        "sy",
        "decay",
        "wave_r",
        "wave_g",
        "wave_b",
        "wave_a",
    )

    def __init__(self, node_id: str, preset: MilkPreset | None = None):
        super().__init__(node_id)
        self.params = {}
        self.preset: MilkPreset | None = None
        self.source_file = None
        self.num_waves = 0
        self.num_shapes = 0
        self.num_per_pixel_eqs = 0
        self.num_per_vertex_eqs = 0
        if preset is not None:
            self.preset = preset
            params = getattr(preset, "params", {})
            for p in self._DNA_PARAMS:
                if p in params and params[p] is not None:
                    try:
                        self.params[p] = float(params[p])
                    except (ValueError, TypeError):
                        pass
            self.source_file = getattr(preset, "source_file", None)
            self.num_waves = len(getattr(preset, "waves", {}))
            self.num_shapes = len(getattr(preset, "shapes", {}))
            self.num_per_pixel_eqs = len(getattr(preset, "per_pixel", []))
            self.num_per_vertex_eqs = len(getattr(preset, "per_vertex", []))

    _DNA_BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {
        "zoom": (0.5, 1.8),
        "rot": (-0.5, 0.5),
        "warp": (0.0, 1.0),
        "cx": (0.0, 1.0),
        "cy": (0.0, 1.0),
        "dx": (-0.1, 0.1),
        "dy": (-0.1, 0.1),
        "sx": (0.7, 1.3),
        "sy": (0.7, 1.3),
        "decay": (0.7, 1.0),
        "wave_r": (0.0, 1.0),
        "wave_g": (0.0, 1.0),
        "wave_b": (0.0, 1.0),
        "wave_a": (0.0, 1.0),
    }
    DNA_ZOOM_BOUNDS = _DNA_BOUNDS["zoom"]

    _DNA_MUTATION_SCALE: ClassVar[dict[str, float]] = {
        "zoom": 0.4,
        "rot": 0.6,
        "warp": 0.7,
        "cx": 0.8,
        "cy": 0.8,
        "dx": 1.0,
        "dy": 1.0,
        "sx": 0.6,
        "sy": 0.6,
        "decay": 0.3,
        "wave_r": 1.2,
        "wave_g": 1.2,
        "wave_b": 1.2,
        "wave_a": 0.5,
    }

    def mutate(self, rate: float):
        if random.random() < rate * 0.5:
            self.audio_target = random.choice(["bass", "mid", "treb", "beat"])
        for k, v in list(self.params.items()):
            scale = self._DNA_MUTATION_SCALE.get(k, 1.0)
            if random.random() < rate * scale:
                if isinstance(v, float):
                    step = 0.08 * scale
                    self.params[k] = v + random.uniform(-step, step)
                elif isinstance(v, int):
                    self.params[k] = v + random.choice([-1, 1])
                bounds = self._DNA_BOUNDS.get(k)
                if bounds is not None:
                    lo, hi = bounds
                    self.params[k] = max(lo, min(hi, self.params[k]))

    def serialize(self) -> dict:
        base = super().serialize()
        base.update(
            {
                "source_file": self.source_file,
                "num_waves": self.num_waves,
                "num_shapes": self.num_shapes,
                "num_per_pixel_eqs": self.num_per_pixel_eqs,
                "num_per_vertex_eqs": self.num_per_vertex_eqs,
            }
        )
        return base

    def copy(self):
        """Deep-copy this DNA node, preserving source/structural metadata.

        The inherited :meth:`EffectNode.copy` constructs a new node using only
        ``node_id`` and copies ``params`` / ``audio_target``.  For a DNA node
        that loses the structural metadata (source_file, wave/shape/equation
        counts), which is exactly the information a real population
        cloning/breeding cycle would need.  Override here so a cloned organism
        remains a faithful copy of its parent.
        """
        new = PresetDNAEffect(self.node_id)
        new.params = self.params.copy()
        new.preset = self.preset
        new.audio_target = self.audio_target
        new.source_file = self.source_file
        new.num_waves = self.num_waves
        new.num_shapes = self.num_shapes
        new.num_per_pixel_eqs = self.num_per_pixel_eqs
        new.num_per_vertex_eqs = self.num_per_vertex_eqs
        return new


class EffectGraph:
    """Phase 1: Primary visual representation."""

    def __init__(self):
        self.nodes: list[EffectNode] = []
        self.connections: list[Connection] = []
        self.generation = 0
        self.fitness = 0.5

    def mutate(self, rate: float):
        """Parameter-only mutation for evolution.

        Only mutates node parameters; structural mutations (ADD/REMOVE/REPLACE/SWAP)
        are removed since the evolutionary system now uses the MilkDrop shader
        pipeline with parameter jittering instead of generating separate GLSL.
        """
        for node in self.nodes:
            node.mutate(rate)

    def copy(self) -> EffectGraph:
        g = EffectGraph()
        g.nodes = [n.copy() for n in self.nodes]
        g.connections = [c.copy() for c in self.connections]
        g.fitness = self.fitness
        g.generation = self.generation
        return g

    def serialize(self) -> dict:
        return {
            "fitness": self.fitness,
            "generation": self.generation,
            "nodes": [n.serialize() for n in self.nodes],
            "connections": [c.serialize() for c in self.connections],
        }


# --- Fitness analysis --------------------------------------------
@dataclass(frozen=True)
class FitnessSnapshot:
    """Immutable, atomic record of the most recently analyzed frame.

    Publishing metrics and fitness as a single frozen snapshot under the lock
    eliminates the temporal-ordering divergence that occurred when
    ``_last_metrics`` and ``_last_fitness`` were tracked by separate counters.
    """

    frame_id: int
    fitness: float
    metrics: dict
    organism_id: int = 0
    generation: int = 0
    preset_generation: int = 0


@dataclass(frozen=True)
class EvaluationToken:
    """Immutable identity token for an in-flight fitness evaluation.

    A snapshot is only valid for the organism it was captured for.  Tagging
    every queued analysis with ``frame_id``, ``organism_id``, ``generation``
    and ``preset_generation`` lets :meth:`EvolutionEngine.step` require an
    exact match before mutating, so a fitness result from organism A can never
    be applied to organism B when ``active_idx`` advances while the generation
    stays the same, and a stale result from a previous preset install is
    rejected even if its frame_id has since advanced.
    """

    frame_id: int
    organism_id: int
    generation: int
    preset_generation: int = 0


@dataclass
class PendingPBO:
    pbo_id: int
    fence: int | None
    frame_id: int
    organism_id: int
    generation: int
    preset_generation: int
    epoch: int = 0

    @classmethod
    def empty_slot(
        cls, pbo_id: int, epoch: int, preset_generation: int = 0
    ) -> PendingPBO:
        """Create an empty slot with the current epoch.

        Used when a PBO has been consumed and its fence deleted, so the
        slot's identity resets to the current epoch for audit clarity.
        """
        return cls(
            pbo_id=pbo_id,
            fence=None,
            frame_id=0,
            organism_id=0,
            generation=0,
            preset_generation=preset_generation,
            epoch=epoch,
        )


class FitnessAnalyzer:
    """Phase 6: Small-texture frame analysis with async PBO readback."""

    ANALYSIS_SIZE = 64

    # Fitness metric scaling and penalties (renderer policy).
    FITNESS_WEIGHT_LUM_STD: float = 3.0
    FITNESS_WEIGHT_COLOR_VAR: float = 5.0
    FITNESS_WEIGHT_EDGE_DENSITY: float = 5.0
    FITNESS_WEIGHT_TEMPORAL_DIFF: float = 5.0
    FITNESS_WEIGHT_ENTROPY_DIVISOR: float = 4.0
    FITNESS_BLACK_PIXEL_PENALTY: float = 0.25

    def __init__(self, renderer: MilkdropRenderer | None = None):
        self.renderer = renderer
        self.fbo = None
        self.tex = None
        self._pbo_ring: list[PendingPBO] = []
        self._last_fitness = 0.5
        self._last_snapshot: FitnessSnapshot | None = None
        self._owner: GLResourceOwner | None = None
        self._analysis_frames_dropped = 0
        self._metric_queue_dropped = 0
        self._lock = threading.Lock()
        self._metric_queue: Queue = Queue(maxsize=8)
        self._worker_thread = None
        self._worker_started = False
        self._stop_event = threading.Event()
        self._epoch = 0
        if renderer is not None:
            self._owner = getattr(renderer, "_gl_resource_owner", None)
        self._start_worker()

    def _start_worker(self):
        """Start the background worker thread if not already running."""
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._stop_event.clear()
            self._metric_queue = Queue(maxsize=8)
            self._worker_thread = threading.Thread(target=self._worker, daemon=True)
            self._worker_thread.start()
            self._worker_started = True

    def shutdown(self):
        """Stop the worker thread, then release GL resources.

        Enforces the ownership contract: the worker thread is stopped *first*
        and joined so that it cannot touch any GL resource that
        :meth:`cleanup` proceeds to queue for deletion.  Call this on final
        application shutdown or renderer destruction, not on GL context
        loss/recreation (use :meth:`invalidate_context` for that).

        The worker reference is only cleared once the thread has actually
        terminated.  If the thread refuses to die within the join window
        the reference is retained so that a subsequent ``_start_worker()`` cannot
        spawn a second worker alongside a still-living one.
        """
        self._stop_event.set()
        self._worker_started = False
        worker = self._worker_thread
        if worker is not None and worker.is_alive():
            worker.join(timeout=2.0)
            if worker.is_alive():
                logger.warning(
                    "FitnessAnalyzer worker did not stop within 2s shutdown window"
                )
                self._worker_thread = worker
        # Drain any pending metric-queue items the worker may have enqueued
        # before it stopped, so we don't leave stale data behind.
        try:
            while True:
                self._metric_queue.get_nowait()
        except Empty:
            pass
        self.cleanup()
        self._worker_thread = None

    def invalidate_context(self, reason="context_destroyed"):
        """Drop every GL object reference after a context loss/recreation.

        Stale PBO names, GLsync handles, and pending frame IDs belong to the
        destroyed context and must never be reused in the new one.  This
        clears the PBO ring (without queuing deletions for objects whose
        ownership is now meaningless) and resets the frame counter so the
        next analysis pass starts from a clean slate.

        Must only be called from genuine GL context lifecycle paths (see
        :meth:`GLResourceOwner.invalidate_context` for the full invariant).
        The ``reason`` label is recorded for diagnostics.

        Does NOT stop the worker thread - it will be restarted on next
        analysis via _start_worker_if_needed().
        """
        logger.debug("FitnessAnalyzer.invalidate_context(reason=%r)", reason)
        self.fbo = None
        self.tex = None
        self._pbo_ring.clear()

    def _worker(self):
        # Structural invariant: this worker must be CPU-only.  It processes
        # pixel buffers that the render thread already read back via PBOs,
        # so it must never touch any GL resource.  Assert that it is NOT
        # running on the render thread (the thread that owns the GL context)
        # so that a future regression that adds a GL call here fails loudly
        # instead of racing against context teardown during on_unrealize.
        owner = self._owner
        render_thread_id = owner._render_thread_id if owner is not None else None
        if render_thread_id is not None:
            assert threading.current_thread().ident != render_thread_id, (
                "FitnessAnalyzer worker must be CPU-only and never run on the render thread"
            )
        while True:
            try:
                item = self._metric_queue.get(timeout=0.5)
            except Empty:
                if self._stop_event.is_set():
                    return
                continue
            if item is None:
                if self._stop_event.is_set():
                    return
                continue
            frame_id, pixels, organism_id, generation, preset_generation, epoch = item
            with self._lock:
                if epoch != self._epoch:
                    continue
                prev = self._last_snapshot
                # The temporal baseline must belong to the SAME organism.
                # _last_snapshot can be left over from a previous organism
                # (the evolutionary loop deliberately switches through the
                # population), so comparing against it would reward or
                # penalize an organism based on how different it is from
                # its predecessor rather than its own temporal behaviour.
                # Also require the temporal baseline to match preset_generation.
                if (
                    prev is not None
                    and prev.organism_id == organism_id
                    and abs(prev.generation - generation) <= 1
                    and prev.preset_generation == preset_generation
                ):
                    prev_lum_mean = prev.metrics["lum_mean"]
                else:
                    prev_lum_mean = None
            _, metrics, score = self._compute_metrics(pixels, prev_lum_mean)
            with self._lock:
                if epoch != self._epoch:
                    continue
                prev = self._last_snapshot
                if prev is None or frame_id >= prev.frame_id:
                    self._last_snapshot = FitnessSnapshot(
                        frame_id=frame_id,
                        fitness=score,
                        metrics=metrics,
                        organism_id=organism_id,
                        generation=generation,
                        preset_generation=preset_generation,
                    )
                    self._last_fitness = score

    def _ensure_resources(self):
        if self.fbo and GL.glIsFramebuffer(self.fbo):
            return
        owner = self._owner
        renderer = self.renderer
        if owner is None or renderer is None:
            raise RuntimeError(
                "Fitness analysis requires a renderer with a GL resource owner"
            )
        if owner._render_thread_id is not None:
            assert threading.current_thread().ident == owner._render_thread_id, (
                "_ensure_resources must be called from the render thread"
            )
        fbo = tex = 0
        try:
            with _gl_state_guard(renderer._gl_state, owner=owner):
                fbo = int(GL.glGenFramebuffers(1))
                tex = int(GL.glGenTextures(1))
                GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
                GL.glTexImage2D(
                    GL.GL_TEXTURE_2D,
                    0,
                    GL.GL_RGBA8,
                    self.ANALYSIS_SIZE,
                    self.ANALYSIS_SIZE,
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
                    raise RuntimeError(f"Analysis FBO incomplete: 0x{status:X}")
            self.fbo = fbo
            self.tex = tex
            fbo = tex = 0
        finally:
            if fbo:
                owner.queue_delete("fbo", fbo)
            if tex:
                owner.queue_delete("texture", tex)

    def _get_pbo(self, index: int) -> int:
        while len(self._pbo_ring) <= index:
            self._pbo_ring.append(
                PendingPBO(int(GL.glGenBuffers(1)), None, 0, 0, 0, 0, self._epoch)
            )
        return self._pbo_ring[index].pbo_id

    def _compute_metrics(self, pixels, prev_lum_mean: float | None = None):
        rgba = pixels.astype(np.float32) / 255.0
        r, g, b = rgba[..., 0], rgba[..., 1], rgba[..., 2]
        # BT.601 luminance coefficients (mathematical standard, not renderer policy).
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        lum_mean = float(np.mean(lum, keepdims=False))
        lum_std = float(np.std(lum))
        color_var = float(np.mean([np.var(r, keepdims=False), np.var(g, keepdims=False), np.var(b, keepdims=False)], keepdims=False))
        dx = np.abs(np.diff(rgba, axis=1))
        dy = np.abs(np.diff(rgba, axis=0))
        edge_density = float(0.5 * (np.mean(dx, keepdims=False) + np.mean(dy, keepdims=False)))
        hist, _ = np.histogram(lum, bins=16, range=(0, 1))
        hist = hist.astype(np.float32) + 1e-6
        hist /= hist.sum()
        entropy = float(-np.sum(hist * np.log2(hist)))
        black_frac = float(np.mean(lum < 0.05, keepdims=False))
        temporal_diff = (
            abs(lum_mean - prev_lum_mean) if prev_lum_mean is not None else 0.0
        )
        score = 0.0
        # Normalise the positive weights so they sum to 1.0.  Each component
        # is already clamped to [0, 1] by the min() wrapper, so five equal
        # weights of 0.2 give a maximum positive score of 1.0 before the
        # black-pixel penalty.  The previous 0.15 each capped the ceiling at
        # 0.75, which made the persistence threshold unreachable.
        score += min(1.0, lum_std * self.FITNESS_WEIGHT_LUM_STD) * 0.2
        score += min(1.0, color_var * self.FITNESS_WEIGHT_COLOR_VAR) * 0.2
        score += min(1.0, edge_density * self.FITNESS_WEIGHT_EDGE_DENSITY) * 0.2
        score += min(1.0, entropy / self.FITNESS_WEIGHT_ENTROPY_DIVISOR) * 0.2
        score += min(1.0, temporal_diff * self.FITNESS_WEIGHT_TEMPORAL_DIFF) * 0.2
        score -= black_frac * self.FITNESS_BLACK_PIXEL_PENALTY
        score = max(0.0, min(1.0, score))
        return (
            lum_mean,
            {
                "lum_mean": lum_mean,
                "lum_std": lum_std,
                "color_var": color_var,
                "edge_density": edge_density,
                "entropy": entropy,
                "black_frac": black_frac,
                "temporal_diff": temporal_diff,
            },
            score,
        )

    def latest_snapshot(self) -> FitnessSnapshot | None:
        """Return the most recent fitness snapshot, or ``None`` if no
        analysis has completed yet.

        Each snapshot is tagged with the ``frame_id`` and the
        ``organism_id`` / ``generation`` of the organism being evaluated.
        Callers (e.g. :meth:`EvolutionEngine.step`) should compare
        ``frame_id`` — or the organism/generation tags — against their last
        evaluated value to avoid mutating an organism based on stale or
        from-a-different-organism fitness data.
        """
        with self._lock:
            return self._last_snapshot

    def cleanup(self):
        """Release GL resources only. Does not stop the worker thread.

        **Ownership contract**: the caller must guarantee that the worker
        thread is not running (or cannot touch GL resources) before invoking
        this method.  On final shutdown, call :meth:`shutdown` instead, which
        stops the worker first and then delegates here.

        Validates GLsync objects with ``glIsSync`` before queueing for
        deletion, preventing stale or already-deleted sync handles from
        entering the flush queue.
        """
        owner = self._owner
        if owner is None:
            self.fbo = None
            self.tex = None
            self._pbo_ring = []
            return
        if self.fbo:
            owner.queue_delete("fbo", self.fbo)
            self.fbo = None
        if self.tex:
            owner.queue_delete("texture", self.tex)
            self.tex = None
        for entry in self._pbo_ring:
            pbo_id = entry.pbo_id
            fence = entry.fence
            if pbo_id:
                owner.queue_delete("buffer", pbo_id)
            if fence is not None and GL.glIsSync(fence):
                owner.queue_delete("sync", fence)
        self._pbo_ring = []


# --- Evolution engine -------------------------------------------
class EvolutionEngine:
    """Phase 7: The visual evolution engine core.

    Simplified to only handle parameter evolution via the MilkDrop shader pipeline.
    No longer generates separate GLSL shaders.
    """

    def __init__(self, renderer):
        self.renderer = renderer
        self.active_graph = EffectGraph()
        self.analyzer = FitnessAnalyzer(renderer)
        self.mutation_rate = 0.2
        self.population = [self.active_graph]
        self.active_idx = 0
        self.generation = 0
        self._last_evaluated_frame = -1
        self._evaluated_organism = None
        # Population-based evolution bookkeeping.
        self._organisms_evaluated_this_gen = 0
        # Generation advancement is controlled entirely by the evaluation
        # counter: the population advances once every member has been
        # evaluated.  The previous fixed-frame interval (120 frames) was
        # dead bookkeeping — it was never consulted by the advance guard —
        # and has been removed so the algorithm's real control flow is the
        # only thing that matters.
        # Keep the top N organisms as elites when forming the next generation.
        self._elitism_count = 2
        # Minimum population size for the evolutionary cycle to operate.
        # When the population is below this threshold, mutated copies of the
        # active organism are seeded so that rank/elites/crossover can run
        # instead of stalling on a single organism.
        self._min_population = 4
        self._max_population = 12

    def _apply_generation(self, population: list[EffectGraph]) -> None:
        self.generation += 1
        for org in population:
            org.generation = self.generation

    def apply_dna_to_preset_vm(self, preset_vm: PresetVM) -> None:
        """Apply the active organism's DNA to the live PresetVM.

        This is the single authoritative bridge between the evolutionary
        system and the rendered phenotype:

            active organism (EffectGraph)
                ↓ DNA parameters
                ↓
                PresetVM.base_params
                ↓
                GLSL uniforms
                ↓
                render

        Without this call, mutations accumulate in ``active_graph`` while
        the rendered picture continues to follow whatever preset was last
        loaded from disk — the evolutionary system would be observing and
        mutating metadata rather than evolving the actual rendered organism.

        The DNA graph carries a single :class:`PresetDNAEffect` node whose
        wrapped preset holds the evolvable numeric traits (zoom, rot, warp,
        sx, sy, decay, wave_*).  We copy those values into
        ``preset_vm.base_params`` so the next ``run_frame`` call emits them
        as GPU uniforms.  Values are clamped to the same bounds used by
        :meth:`evolve` so a runaway mutation cannot produce a broken frame.
        """
        node = next(
            (n for n in self.active_graph.nodes if isinstance(n, PresetDNAEffect)),
            None,
        )
        if node is None:
            return
        # Read from node.params (the mutated DNA) — NOT from node.preset.params
        # (the original unmutated preset).  EffectNode.mutate modifies
        # node.params in place, so reading the preset's params here would
        # silently discard every mutation and leave the renderer showing the
        # unmodified preset.  See the DNA→PresetVM bridge docstring above.
        params = getattr(node, "params", None)
        if not params:
            return
        bounds = PresetDNAEffect._DNA_BOUNDS
        for name, value in params.items():
            if not isinstance(value, (int, float)):
                continue
            if isinstance(value, bool):
                continue
            lo, hi = bounds.get(name, (-1e30, 1e30))
            clamped = float(max(lo, min(hi, value)))
            preset_vm.base_params[name] = clamped
            preset_vm._dirty_base_keys.add(name)

    def current_evaluation_token(self, frame_id: int | None = None) -> EvaluationToken:
        """Capture the identity of the organism currently under evaluation.

        The fitness pipeline is asynchronous: a frame is rendered, queued for
        analysis, and the result arrives many frames later.  Between queue and
        consumption the active organism may change.  Callers that queue an
        analysis must capture the *expected* organism atomically via this
        method so the snapshot can be matched exactly before mutation.

        ``frame_id`` must be supplied by the caller — it is the renderer's
        current ``global_frame`` at the moment the frame was rendered.  Using
        ``self._last_evaluated_frame`` here would be wrong: that field tracks
        the last fitness result *consumed* by :meth:`step`, not the frame
        currently being queued for analysis.  A stale token would cause
        :meth:`step` to reject a perfectly valid snapshot because its
        ``frame_id`` would never advance past the already-consumed value.
        """
        if frame_id is None:
            raise ValueError(
                "current_evaluation_token requires an explicit frame_id from "
                "the caller (the renderer's global_frame at render time)."
            )
        return EvaluationToken(
            frame_id=frame_id,
            organism_id=self.active_idx,
            generation=self.active_graph.generation,
            preset_generation=self.renderer.preset_switch_generation,
        )

    def step(self, frame_id: int | None = None):
        """The primary discovery loop (analyze, persist, record, advance).

        No-ops when *frame_id* is invalid (0 or negative) — this can happen
        during shutdown or before the first frame completes.

        When *frame_id* is ``None``, the snapshot's own ``frame_id`` is
        used, which allows callers to invoke ``step()`` without tracking
        the render-side frame counter.

        Fitness is evaluated from an asynchronous snapshot tagged with the
        frame id of the analysed frame.  If the most recent snapshot belongs
        to a frame we have already processed (i.e. it is stale, or it belongs
        to a different organism/generation), ``step`` returns early without
        mutating, so the evolution loop never acts on out-of-date fitness.

        Validation order is strict: the snapshot is rejected *before* any
        bookkeeping advances, so a stale or mismatched result can never leave
        ``_last_evaluated_frame`` pointing past a frame that was actually
        discarded.

        The active organism is recorded in ``_evaluated_organism`` so the
        population loop can later rank, retain elites, and breed.  After the
        current organism's fitness is recorded, ``step`` advances to the next
        unevaluated organism in the population so the engine actually cycles
        through multiple candidates instead of mutating a single organism
        forever.
        """
        snapshot = self.analyzer.latest_snapshot()
        if snapshot is None:
            return
        if frame_id is None:
            frame_id = snapshot.frame_id
        if not frame_id or frame_id <= 0:
            return
        # Validate first, consume second.
        if snapshot.frame_id <= self._last_evaluated_frame:
            return
        if snapshot.organism_id != self.active_idx:
            return
        if snapshot.generation and snapshot.generation != self.active_graph.generation:
            return
        if snapshot.preset_generation != self.renderer.preset_switch_generation:
            return
        self._last_evaluated_frame = snapshot.frame_id
        self.active_graph.fitness = snapshot.fitness
        self._organisms_evaluated_this_gen += 1
        # Record the evaluated organism for the next population cycle.
        self._evaluated_organism = self.active_graph
        # The positive fitness components are normalised to sum to 1.0 in
        # FitnessAnalyzer._compute_metrics (five 0.2 weights, each clamped to
        # [0,1]), so the reachable ceiling is 1.0 before the black-pixel
        # penalty.  A threshold of 0.8 is therefore attainable; the previous
        # 0.8 was unreachable because the old 0.15 weights capped the ceiling
        # at 0.75.
        if self.active_graph.fitness > 0.8:
            self._persist_and_cluster(self.active_graph)
        # Do NOT mutate the organism that was just evaluated.  Mutation is
        # reserved for creating the next generation in
        # :meth:`_maybe_advance_generation`.  Mutating here would assign a
        # fitness to organism N and then silently turn it into organism N+1,
        # causing the ranking system to treat an untested mutation as if it
        # had the parent's proven fitness.
        self._maybe_advance_generation()
        # If the generation did not advance (not all organisms evaluated yet),
        # move to the next organism in the current generation so the render
        # loop actually cycles through the population.
        if self._evaluated_organism is self.active_graph and self.active_idx + 1 < len(
            self.population
        ):
            self.active_idx += 1
            self.active_graph = self.population[self.active_idx]
            logger.debug("Organism switch: idx=%d gen=%d", self.active_idx, self.generation)

    def _maybe_advance_generation(self):
        """Advance the population when every organism has been evaluated.
        """
        # Only advance when every organism in the current generation has been
        # evaluated.
        if self._organisms_evaluated_this_gen < len(self.population):
            return
        self._organisms_evaluated_this_gen = 0

        # Seed first so ranking sees a full population
        while len(self.population) < self._min_population:
            parent = random.choice(self.population)
            clone = parent.copy()
            clone.mutate(self.mutation_rate)
            clone.fitness = float("-inf")
            clone.generation = self.generation
            self.population.append(clone)
        # Cap population size
        self.population = self.population[: getattr(self, "_max_population", 12)]

        ranked = sorted(self.population, key=lambda g: g.fitness, reverse=True)
        elites = ranked[: max(1, self._elitism_count)]
        next_pop: list[EffectGraph] = [e.copy() for e in elites]

        while len(next_pop) < len(self.population):
            p1 = random.choice(elites)
            p2 = random.choice(ranked)
            child = self._breed_graphs(p1, p2)
            if random.random() < self.mutation_rate:
                child.mutate(self.mutation_rate)
            next_pop.append(child)

        self.population = next_pop
        self._apply_generation(self.population)
        self._evaluated_organism = None
        self.active_idx = 0
        self.active_graph = self.population[0]
        logger.info(
            "🧬 Generation advanced: pop=%d elites=%d best=%.2f gen=%d",
            len(self.population),
            len(elites),
            self.active_graph.fitness,
            self.generation,
        )

    def _materialize_dna_preset(self, node: PresetDNAEffect) -> MilkPreset:
        """Create a preset copy with DNA-mutated params overlaid.

        The active organism's ``node.params`` are the authoritative evolved
        phenotype.  Overlay only the DNA-authoritative parameters onto a copy
        of ``node.preset`` so that crossover and breeding operate on the
        mutated values rather than the stale disk-loaded preset.
        """
        if node.preset is None:
            raise RuntimeError("DNA node has no source preset")
        preset = node.preset.copy()
        for name in PresetDNAEffect._DNA_PARAMS:
            if name in node.params:
                preset.params[name] = node.params[name]
        return preset

    def _breed_graphs(self, p1: EffectGraph, p2: EffectGraph) -> EffectGraph:
        """Crossover two EffectGraphs into a child.

        The graphs carry a single PresetDNAEffect node.  Materialize each
        node's mutated params onto a preset copy before breeding so that
        crossover operates on the evolved phenotype, not the stale
        disk-loaded preset.
        """
        n1 = next((n for n in p1.nodes if isinstance(n, PresetDNAEffect)), None)
        n2 = next((n for n in p2.nodes if isinstance(n, PresetDNAEffect)), None)
        if n1 is None or n2 is None:
            winner = p1 if p1.fitness >= p2.fitness else p2
            child = winner.copy()
            return child
        p1_mat = self._materialize_dna_preset(n1)
        p2_mat = self._materialize_dna_preset(n2)
        child_preset = self._breed(p1_mat, p2_mat)
        child = EffectGraph()
        child.nodes.append(PresetDNAEffect("dna", child_preset))
        child.fitness = 0.5
        return child

    def _persist_and_cluster(self, graph: EffectGraph):
        """Persist high-fitness genome to disk."""
        try:
            genomes_dir = Path.home() / ".linamp" / "genomes"
            genomes_dir.mkdir(parents=True, exist_ok=True)
            ts = int(time.time())
            gen = getattr(graph, "generation", self.generation)
            fitness = getattr(graph, "fitness", 0.0)
            suffix = uuid.uuid4().hex[:6]
            path = genomes_dir / f"genome_g{gen}_f{fitness:.3f}_{ts}_{suffix}.json"
            data = graph.serialize()
            data["_meta"] = {
                "generation": gen,
                "fitness": fitness,
                "timestamp": ts,
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)),
            }
            with open(path, "w") as f:
                json.dump(data, f)
            logger.debug("Persisted genome to %s", path)
        except Exception as e:  # noqa: BLE001
            logger.debug("Failed to persist genome: %s", e)

    def evolve(
        self, current_preset: MilkPreset | None, next_preset: MilkPreset
    ) -> MilkPreset:
        """Phase 10: Compatibility Bridge.

        Returns a deep-copy of *next_preset* with jitter applied to evolvable
        parameters.  The caller's object is never mutated — this is critical
        because presets may be cached or reused across the loader pool.
        """
        evolved = copy.deepcopy(next_preset)
        params = evolved.params
        evolvable = [
            ("zoom", True, 0.5, 1.8, 0.05),
            ("rot", False, -0.5, 0.5, 0.02),
            ("warp", False, 0.0, 1.0, 0.05),
            ("dx", False, -0.1, 0.1, 0.01),
            ("dy", False, -0.1, 0.1, 0.01),
            ("sx", True, 0.7, 1.3, 0.03),
            ("sy", True, 0.7, 1.3, 0.03),
            ("decay", True, 0.7, 1.0, 0.01),
            ("wave_r", False, 0.0, 1.0, 0.1),
            ("wave_g", False, 0.0, 1.0, 0.1),
            ("wave_b", False, 0.0, 1.0, 0.1),
        ]
        for name, is_mult, v_min, v_max, jitter in evolvable:
            if name not in params or params[name] is None:
                continue
            if random.random() < self.mutation_rate:
                val = float(params[name])
                if is_mult:
                    val *= random.uniform(1.0 - jitter, 1.0 + jitter)
                else:
                    val += random.uniform(-jitter, jitter)
                params[name] = np.clip(val, v_min, v_max)
        return evolved

    def _breed(self, p1: MilkPreset, p2: MilkPreset) -> MilkPreset:
        """Combine two presets into a child (crossover)."""
        child = p2.copy()
        traits = [
            "zoom",
            "rot",
            "warp",
            "cx",
            "cy",
            "dx",
            "dy",
            "sx",
            "sy",
            "decay",
            "wave_r",
            "wave_g",
            "wave_b",
            "wave_a",
            "ob_size",
            "ib_size",
        ]
        for trait in traits:
            if trait in p1.params and trait in p2.params:
                if p1.params[trait] is None or p2.params[trait] is None:
                    continue
                choice = random.random()
                if choice < 0.4:
                    child.params[trait] = p1.params[trait]
                elif choice < 0.8:
                    v1 = float(p1.params[trait])
                    v2 = float(p2.params[trait])
                    t = random.random()
                    child.params[trait] = v1 * t + v2 * (1.0 - t)
        return child

    def cleanup(self):
        """Tear down owned runtime resources permanently.

        Stops the fitness analyzer worker thread and releases GL resources.
        Call this only on final application shutdown or renderer destruction.
        """
        self.stop_worker()
        self.analyzer.cleanup()

    def stop_worker(self):
        """Stop the fitness analyzer worker thread without releasing GL resources.

        This is the correct call from :meth:`MilkdropRenderer.on_unrealize`:
        the worker must be terminated *before* GL handles are invalidated so
        that the ownership contract is structural rather than conventional.

        The worker is CPU-only by construction (it only processes pixel
        buffers already read back by the render thread via PBOs).  That
        invariant is asserted in :meth:`FitnessAnalyzer._worker` so that a
        future regression that adds a GL call to the worker fails loudly
        instead of racing against context teardown.
        """
        self.analyzer.shutdown()

    def invalidate_context(self, reason="context_destroyed"):
        """Invalidate GL resources after context loss/recreation.

        Must only be called from genuine GL context lifecycle paths (see
        :meth:`GLResourceOwner.invalidate_context` for the full invariant).
        The ``reason`` label is forwarded to the analyzer and recorded for
        diagnostics.

        Does not stop the worker thread - it will be restarted on next use.
        """
        logger.debug("EvolutionEngine.invalidate_context(reason=%r)", reason)
        self.analyzer.invalidate_context(reason=reason)
        self._last_evaluated_frame = -1
