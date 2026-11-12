from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import gi

gi.require_version("Gdk", "4.0")
gi.require_version("Gtk", "4.0")
from gi.repository import Gdk, GdkPixbuf, GLib, Gtk

from ...core import *
from ...media import *
from ...models import *
from ...podcasts import *
from ...radio import *


class CoverArtWidget(Gtk.Box):
    def __init__(self, player: Any, size: int = 150) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self.player = player
        self.size = size
        self.current_cover_path: str | None = None
        self.loading = False
        self.set_size_request(size, size)
        self.set_halign(Gtk.Align.CENTER)
        self.set_valign(Gtk.Align.CENTER)
        self.image = Gtk.Image()
        self.image.set_size_request(size, size)
        self.image.set_halign(Gtk.Align.CENTER)
        self.image.set_valign(Gtk.Align.CENTER)
        self.image.set_icon_size(Gtk.IconSize.LARGE)
        self.append(self.image)
        self.add_css_class("cover-art-widget")
        self.add_css_class("cover-square")
        self.show_placeholder()

    def show_placeholder(self) -> None:
        self.current_cover_path = None
        self.image.set_from_icon_name("folder-music")
        self.image.set_pixel_size(self.size // 2)
        self.loading = False

    def _create_texture_from_bytes(self, bytes_data: bytes) -> Gdk.Texture | None:
        """Create a Gdk.Texture from raw image bytes, scaled to self.size."""
        try:
            loader = GdkPixbuf.PixbufLoader()
            loader.write(bytes_data)
            loader.close()
            pixbuf = loader.get_pixbuf()
            if not pixbuf:
                return None
            width = pixbuf.get_width()
            height = pixbuf.get_height()
            if width > height:
                scale = self.size / width
            else:
                scale = self.size / height
            new_width = int(width * scale)
            new_height = int(height * scale)
            scaled = pixbuf.scale_simple(
                new_width, new_height, GdkPixbuf.InterpType.BILINEAR
            )
            if not scaled:
                return None
            success, png_bytes = scaled.save_to_bufferv("png", [], [])
            if not success:
                return None
            return Gdk.Texture.new_from_bytes(GLib.Bytes.new(png_bytes))
        except (GLib.GError, OSError, TypeError):
            return None

    def load_cover(self, cover_art: CoverArt | None, audio_path: str = "") -> None:
        if not cover_art:
            self.show_placeholder()
            return
        if cover_art.path and os.path.exists(cover_art.path):
            try:
                texture: Gdk.Texture | None = Gdk.Texture.new_from_filename(
                    cover_art.path
                )
                self.image.set_from_paintable(texture)
                self.image.set_size_request(self.size, self.size)
                self.image.set_halign(Gtk.Align.FILL)
                self.image.set_valign(Gtk.Align.FILL)
                self.current_cover_path = cover_art.path
            except GLib.GError:
                if cover_art.bytes_data:
                    texture = self._create_texture_from_bytes(cover_art.bytes_data)
                    if texture:
                        self.image.set_from_paintable(texture)
                        self.current_cover_path = audio_path
                    else:
                        self.show_placeholder()
                else:
                    self.show_placeholder()
        elif cover_art.bytes_data:
            texture = self._create_texture_from_bytes(cover_art.bytes_data)
            if texture:
                self.image.set_from_paintable(texture)
                self.current_cover_path = audio_path
            else:
                self.show_placeholder()
        else:
            self.show_placeholder()

    def set_shape(self, shape: str) -> None:
        """Apply cover art shape class."""
        for cls in ("cover-square", "cover-circle", "cover-rounded"):
            self.remove_css_class(cls)
        if shape not in ("square", "circle", "rounded"):
            shape = "square"
        self.add_css_class(f"cover-{shape}")

    def set_size(self, size: int) -> None:
        if size < 50:
            size = 50
        elif size > 800:
            size = 800
        self.size = size
        self.set_size_request(size, size)
        self.image.set_size_request(size, size)

    def load_cover_from_path(self, cover_path: str | Path) -> None:
        if not cover_path or not os.path.exists(cover_path):
            self.show_placeholder()
            return
        try:
            texture = Gdk.Texture.new_from_filename(str(cover_path))
        except GLib.GError:
            logging.getLogger(__name__).debug(
                "Cover art could not be loaded: %s", cover_path, exc_info=True
            )
            self.show_placeholder()
            return
        self.image.set_from_paintable(texture)
        self.image.set_size_request(self.size, self.size)
        self.image.set_halign(Gtk.Align.FILL)
        self.image.set_valign(Gtk.Align.FILL)
        self.current_cover_path = str(cover_path)

    def clear(self) -> None:
        self.current_cover_path = None
        self.show_placeholder()

    def cleanup(self) -> None:
        self.current_cover_path = None
        self.show_placeholder()
