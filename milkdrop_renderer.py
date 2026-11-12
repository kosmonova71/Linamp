#!/usr/bin/env python3
"""Compatibility shim for the split MilkDrop renderer package.

The renderer implementation now lives in the :mod:`renderer` package
(:mod:`renderer.core`, :mod:`renderer.presets`, :mod:`renderer.shaders`,
etc.).  This module re-exports the public API so that existing
``from milkdrop_renderer import X`` references keep working unchanged —
both in application code (``linamp.core``) and in tests that read
``milkdrop_renderer.__file__`` / ``milkdrop_renderer.source_text()``.
"""

from __future__ import annotations

import renderer  # ensures the package is bound in this namespace
from renderer import *  # re-export renderer.__all__
from renderer import (  # explicit for Pylance / IDE resolution
    MilkdropRenderer,
    RendererConfig,
    RendererSettings,
    shutdown_async_preset_compiler,
    shutdown_compile_pool,
    source_text,
)

__all__ = list(renderer.__all__) + [  # type: ignore[reportUnsupportedDunderAll]
    "MilkdropRenderer",
    "RendererConfig",
    "RendererSettings",
    "shutdown_async_preset_compiler",
    "shutdown_compile_pool",
    "source_text",
]