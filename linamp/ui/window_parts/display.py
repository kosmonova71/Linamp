"""WinampWindow mixin: Cover art, status line and button state updates."""

from __future__ import annotations

import gi

from .base import WindowBase

gi.require_version("Gdk", "4.0")
gi.require_version("Gtk", "4.0")
gi.require_version("Gst", "1.0")
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from gi.repository import Gdk, GLib, Gst, Gtk  # type: ignore[reportMissingModuleSource]

from ...core import (
    COVER_CACHE_DIR,
    LOGGER,
    THEME_FILE,
)
from ...media import (
    CoverArt,
    _resize_cover_for_display,
    download_cover_for_audio,
    extract_cover_art,
)


class DisplayMixin(WindowBase):
    """Cover art, status line and button state updates.

    The methods are split out of ``WinampWindow`` so the window class
    stays readable; they keep operating on the window instance and
    share its playback, playlist and settings state.
    """

    def update_display(self):
        is_radio = getattr(self, "_is_radio_stream", False)
        if hasattr(self, "player") and self.player and self.playing:
            if is_radio:
                elapsed = (
                    int(time.time() - self._radio_start_time)
                    if hasattr(self, "_radio_start_time")
                    else 0
                )
                mins = elapsed // 60
                secs = elapsed % 60
                live_str = f"LIVE {mins}:{secs:02d}"
                if hasattr(self, "player_tab") and self.player_tab:
                    if (
                        hasattr(self.player_tab, "time_label")
                        and self.player_tab.time_label
                    ):
                        self.player_tab.time_label.set_text(live_str)
                        self.player_tab.time_label.add_css_class("time-label")
                    if (
                        hasattr(self.player_tab, "progress")
                        and self.player_tab.progress
                    ):
                        self.player_tab.progress.set_value(0)
                if hasattr(self, "header_nowplaying") and self.header_nowplaying:
                    title = (
                        self._radio_stream_title
                        if hasattr(self, "_radio_stream_title")
                        else "Radio"
                    )
                    self.header_nowplaying.set_text(f"{title}  •  {live_str}")
            else:
                try:
                    success, position = self.player.query_position(Gst.Format.TIME)
                    if not success:
                        position = 0
                    success, duration = self.player.query_duration(Gst.Format.TIME)
                    if not success:
                        duration = 0
                except (AttributeError, GLib.Error, TypeError):
                    return True
                position_sec = position // Gst.SECOND
                duration_sec = duration // Gst.SECOND
                pos_str = f"{position_sec // 60}:{position_sec % 60:02d}"
                dur_str = f"{duration_sec // 60}:{duration_sec % 60:02d}"
                if hasattr(self, "player_tab") and self.player_tab:
                    if (
                        hasattr(self.player_tab, "time_label")
                        and self.player_tab.time_label
                    ):
                        self.player_tab.time_label.set_text(f"{pos_str} / {dur_str}")
                        self.player_tab.time_label.add_css_class("time-label")
                    if (
                        hasattr(self.player_tab, "progress")
                        and self.player_tab.progress
                    ):
                        if duration_sec > 0 and position_sec >= 0:
                            fraction = position_sec / duration_sec
                            fraction = max(0.0, min(1.0, fraction))
                            current_value = self.player_tab.progress.get_value()
                            if abs(fraction * 100 - current_value) > 0.1:
                                self.player_tab.progress.set_value(fraction * 100)
                                if fraction > 0.75:
                                    self.player_tab.progress.add_css_class("near-end")
                                    self.player_tab.progress.remove_css_class("halfway")
                                elif fraction > 0.5:
                                    self.player_tab.progress.remove_css_class(
                                        "near-end"
                                    )
                                    self.player_tab.progress.add_css_class("halfway")
                                else:
                                    self.player_tab.progress.remove_css_class(
                                        "near-end"
                                    )
                                    self.player_tab.progress.remove_css_class("halfway")
                        else:
                            self.player_tab.progress.set_value(0)
                    if (
                        hasattr(self.player_tab, "track_label")
                        and self.player_tab.track_label
                    ):
                        self.player_tab.track_label.add_css_class("playing-indicator")
                if (
                    hasattr(self, "header_nowplaying")
                    and self.header_nowplaying
                    and self.playlist
                    and 0 <= self.current_track < len(self.playlist)
                ):
                    current = self.playlist[self.current_track]
                    title = getattr(current, "title", "") or "Playing"
                    self.header_nowplaying.set_text(
                        f"{title}  •  {pos_str} / {dur_str}"
                    )
        status_parts = []
        if self.repeat_mode != "none":
            status_parts.append(f"Repeat: {self.repeat_mode.upper()}")
        if self.shuffle_mode:
            status_parts.append("SHUFFLE")
        if not self.auto_play_next:
            status_parts.append("NO AUTO-NEXT")
        if self.beat_aware_enabled:
            status_parts.append("BEAT-AWARE")
        status_text = " | ".join(status_parts) if status_parts else "Ready"
        if self.playing:
            status_text = f"🎵 {status_text}"
            if hasattr(self, "status_bar"):
                self.status_bar.add_css_class("playing")
        else:
            if hasattr(self, "status_bar"):
                self.status_bar.remove_css_class("playing")
                if hasattr(self, "header_nowplaying") and self.header_nowplaying:
                    if self.playlist and 0 <= self.current_track < len(self.playlist):
                        current = self.playlist[self.current_track]
                        title = getattr(current, "title", "") or "Ready"
                        self.header_nowplaying.set_text(title)
                    else:
                        self.header_nowplaying.set_text("Ready")
        self.set_status_message(status_text)
        return True

    def update_status_display(self):
        status_parts = []
        if self.repeat_mode != "none":
            status_parts.append(f"Repeat: {self.repeat_mode.upper()}")
        if self.shuffle_mode:
            status_parts.append("SHUFFLE")
        if not self.auto_play_next:
            status_parts.append("NO AUTO-NEXT")
        if self.beat_aware_enabled:
            status_parts.append("BEAT-AWARE")
        if self.auto_download_covers:
            status_parts.append("AUTO-DL")
        status_text = " | ".join(status_parts) if status_parts else "Ready"
        GLib.idle_add(self.set_status_message, status_text)
        self.update_button_states()

    def update_button_states(self):
        if not hasattr(self, "player_tab"):
            return
        if hasattr(self.player_tab, "repeat_btn"):
            if self.repeat_mode == "none":
                self.player_tab.repeat_btn.remove_css_class("active")
                self.player_tab.repeat_btn.set_tooltip_text("Repeat: Off")
            elif self.repeat_mode == "one":
                self.player_tab.repeat_btn.add_css_class("active")
                self.player_tab.repeat_btn.set_tooltip_text("Repeat: One Track")
            else:
                self.player_tab.repeat_btn.add_css_class("active")
                self.player_tab.repeat_btn.set_tooltip_text("Repeat: All")
        if hasattr(self.player_tab, "shuffle_btn"):
            if self.shuffle_mode:
                self.player_tab.shuffle_btn.add_css_class("active")
                self.player_tab.shuffle_btn.set_tooltip_text("Shuffle: On")
            else:
                self.player_tab.shuffle_btn.remove_css_class("active")
                self.player_tab.shuffle_btn.set_tooltip_text("Shuffle: Off")
        if hasattr(self.player_tab, "beat_btn"):
            if self.beat_aware_enabled:
                self.player_tab.beat_btn.add_css_class("active")
                self.player_tab.beat_btn.set_tooltip_text("Beat Detection: On")
            else:
                self.player_tab.beat_btn.remove_css_class("active")
                self.player_tab.beat_btn.set_tooltip_text("Beat Detection: Off")
        if hasattr(self.player_tab, "download_btn"):
            if self.auto_download_covers:
                self.player_tab.download_btn.add_css_class("active")
                self.player_tab.download_btn.set_tooltip_text(
                    "Auto Download Covers: On"
                )
            else:
                self.player_tab.download_btn.remove_css_class("active")
                self.player_tab.download_btn.set_tooltip_text(
                    "Auto Download Covers: Off"
                )
        if hasattr(self.player_tab, "autonext_btn"):
            if self.auto_play_next:
                self.player_tab.autonext_btn.add_css_class("active")
                self.player_tab.autonext_btn.set_tooltip_text("Auto Next: On")
            else:
                self.player_tab.autonext_btn.remove_css_class("active")
                self.player_tab.autonext_btn.set_tooltip_text("Auto Next: Off")
        if hasattr(self.player_tab, "cover_btn"):
            if self.settings.show_cover_art:
                self.player_tab.cover_btn.add_css_class("active")
                self.player_tab.cover_btn.set_tooltip_text("Cover Art: On")
            else:
                self.player_tab.cover_btn.remove_css_class("active")
                self.player_tab.cover_btn.set_tooltip_text("Cover Art: Off")

    def _load_cover_art_for_track(self, item):
        if not hasattr(self, "player_tab") or not hasattr(
            self.player_tab, "cover_art_widget"
        ):
            return
        cover_widget = self.player_tab.cover_art_widget
        current_audio_path = cover_widget.current_cover_path
        if current_audio_path:
            if current_audio_path.startswith(str(COVER_CACHE_DIR)):
                try:
                    cache_filename = Path(current_audio_path).name
                    cache_key = cache_filename.split(".")[0]
                    item_cache_key = CoverArt.get_cache_key(item.path)
                    if item_cache_key is not None and cache_key == item_cache_key:
                        return
                except (OSError, ValueError):
                    pass
            elif current_audio_path == item.path:
                return
        allow_download = getattr(self, "auto_download_covers", False)
        self._cover_generation = getattr(self, "_cover_generation", 0) + 1
        cover_generation = self._cover_generation
        if self._cover_executor is None:
            self._cover_executor = ThreadPoolExecutor(max_workers=2)
        pending = getattr(self, "_cover_future", None)
        if pending is not None and not pending.done():
            return

        def apply_cover(_cover, _gen=cover_generation, _path=item.path):
            """Publish *cover* on the main loop, re-checking shutdown state.
            The generation/cleanup guards cannot be evaluated on the worker
            thread: full_cleanup can run between the check here and the idle
            dispatch, after which cover_art_widget.cleanup() has torn the tab
            down.  Re-evaluate inside the idle, as the logo and artwork paths
            already do.
            """
            if (
                _gen != getattr(self, "_cover_generation", 0)
                or getattr(self, "_cleanup_in_progress", False)
                or getattr(self, "_cleanup_complete", False)
            ):
                return False
            try:
                cover_widget.load_cover(_cover, _path)
            except Exception:
                logging.getLogger(__name__).debug(
                    "Cover display failed for %s", _path, exc_info=True
                )
            return False

        def extract_and_load(_gen=cover_generation, _path=item.path):
            cover = None
            try:
                cover, _cache_path = extract_cover_art(_path)
            except Exception as e:
                logging.getLogger(__name__).debug(
                    "Cover lookup failed for %s: %s", _path, e, exc_info=True
                )
                cover = None
            if (
                cover
                and _gen == getattr(self, "_cover_generation", 0)
                and not getattr(self, "_cleanup_in_progress", False)
                and not getattr(self, "_cleanup_complete", False)
            ):
                GLib.idle_add(
                    apply_cover, _resize_cover_for_display(cover), _gen, _path
                )
                return
            if allow_download:
                try:
                    cover = download_cover_for_audio(_path)
                except Exception as e:
                    logging.getLogger(__name__).debug(
                        "Cover download failed for %s: %s", _path, e, exc_info=True
                    )
                    cover = None
                if (
                    cover
                    and _gen == getattr(self, "_cover_generation", 0)
                    and not getattr(self, "_cleanup_in_progress", False)
                    and not getattr(self, "_cleanup_complete", False)
                ):
                    GLib.idle_add(apply_cover, cover, _gen, _path)
                    return
            if (
                _gen == getattr(self, "_cover_generation", 0)
                and not getattr(self, "_cleanup_in_progress", False)
                and not getattr(self, "_cleanup_complete", False)
            ):
                GLib.idle_add(apply_cover, None, _gen, _path)

        try:
            self._cover_future = self._cover_executor.submit(extract_and_load)
        except RuntimeError:
            return

    def resize_to_cover_art(self):
        if hasattr(self, "player_tab") and hasattr(self.player_tab, "cover_art_widget"):
            cover_widget = self.player_tab.cover_art_widget
            if cover_widget and hasattr(cover_widget, "size"):
                size = cover_widget.size
                padding = 280
                self.set_default_size(size + padding, size + padding)

    def apply_responsive_layout(self):
        window_width = (
            self.current_window_width if hasattr(self, "current_window_width") else 600
        )
        if window_width <= 400:
            cover_size = 200
        elif window_width <= 500:
            cover_size = 100
        elif window_width <= 600:
            cover_size = 150
        else:
            cover_size = 250
        if self.is_compact_mode:
            if hasattr(self.player_tab, "volume_scale"):
                self.player_tab.volume_scale.set_size_request(120, -1)
            for btn in [
                self.player_tab.prev_btn,
                self.player_tab.play_btn,
                self.player_tab.pause_btn,
                self.player_tab.stop_btn,
                self.player_tab.next_btn,
            ]:
                btn.set_size_request(28, 24)
                btn.set_margin_start(2)
                btn.set_margin_end(2)
            if hasattr(self.playlist_tab, "search_entry"):
                self.playlist_tab.search_entry.set_margin_start(4)
                self.playlist_tab.search_entry.set_margin_end(4)
                self.playlist_tab.search_entry.set_size_request(-1, 32)
            if hasattr(self.player_tab, "cover_art_widget"):
                compact_size = max(60, cover_size // 2)
                self.player_tab.cover_art_widget.set_size(compact_size)
                cover_size = compact_size
            if hasattr(self.player_tab, "progress"):
                self.player_tab.progress.set_size_request(-1, 20)
        elif self.current_window_width < 800:
            if hasattr(self.player_tab, "volume_scale"):
                self.player_tab.volume_scale.set_size_request(160, -1)
            for btn in [
                self.player_tab.prev_btn,
                self.player_tab.play_btn,
                self.player_tab.pause_btn,
                self.player_tab.stop_btn,
                self.player_tab.next_btn,
            ]:
                btn.set_size_request(36, 30)
                btn.set_margin_start(4)
                btn.set_margin_end(4)
            if hasattr(self.playlist_tab, "search_entry"):
                self.playlist_tab.search_entry.set_margin_start(8)
                self.playlist_tab.search_entry.set_margin_end(8)
                self.playlist_tab.search_entry.set_size_request(-1, 36)
            if hasattr(self.player_tab, "cover_art_widget"):
                medium_size = max(100, cover_size - 30)
                self.player_tab.cover_art_widget.set_size(medium_size)
                cover_size = medium_size
            if hasattr(self.player_tab, "progress"):
                self.player_tab.progress.set_size_request(-1, 24)
        else:
            if hasattr(self.player_tab, "volume_scale"):
                self.player_tab.volume_scale.set_size_request(200, -1)
            for btn in [
                self.player_tab.prev_btn,
                self.player_tab.play_btn,
                self.player_tab.pause_btn,
                self.player_tab.stop_btn,
                self.player_tab.next_btn,
            ]:
                btn.set_size_request(44, 36)
                btn.set_margin_start(6)
                btn.set_margin_end(6)
            if hasattr(self.playlist_tab, "search_entry"):
                self.playlist_tab.search_entry.set_margin_start(16)
                self.playlist_tab.search_entry.set_margin_end(16)
                self.playlist_tab.search_entry.set_size_request(-1, 40)
            if self.current_window_width >= 1200:
                large_size = min(350, cover_size + 100)
            elif self.current_window_width >= 1000:
                large_size = min(300, cover_size + 50)
            else:
                large_size = cover_size
            if hasattr(self.player_tab, "cover_art_widget"):
                cover_widget = self.player_tab.cover_art_widget
                current_track = None
                if hasattr(self, "current_track") and 0 <= self.current_track < len(
                    self.playlist
                ):
                    current_track = self.playlist[self.current_track]
                if current_track is not None:
                    cached_cover_path = CoverArt.get_cached_cover_path(
                        current_track.path
                    )
                    if cached_cover_path and cached_cover_path.exists():
                        cover_widget.load_cover_from_path(cached_cover_path)
                self.player_tab.cover_art_widget.set_size(large_size)
                cover_size = large_size
        if hasattr(self.player_tab, "progress"):
            self.player_tab.progress.set_size_request(-1, 28)
        if self.is_compact_mode:
            self.main_box.add_css_class("compact-layout")
            if hasattr(self, "stack"):
                self.stack.add_css_class("compact-stack")
        else:
            self.main_box.remove_css_class("compact-layout")
            if hasattr(self, "stack"):
                self.stack.remove_css_class("compact-stack")

    def on_window_size_changed(self, widget, pspec):
        width = widget.get_width()
        if width <= 0:
            width = self.current_window_width
        was_compact = self.is_compact_mode
        if width < 600:
            self.is_compact_mode = True
        else:
            self.is_compact_mode = False
        if was_compact != self.is_compact_mode or width != self.current_window_width:
            self.apply_responsive_layout()
        self.current_window_width = width

    def set_status_message(self, message):
        self.status_label.set_label(message)

    def apply_xmms_css(self):
        """Load the bundled theme without making startup depend on its presence."""
        display = Gdk.Display.get_default()
        if display is None or not THEME_FILE.is_file():
            LOGGER.warning("GTK theme file not found: %s", THEME_FILE)
            return False
        css_provider = getattr(self, "_xmms_css_provider", None)
        if css_provider is None:
            css_provider = Gtk.CssProvider()
            try:
                css_provider.load_from_path(str(THEME_FILE))
            except GLib.Error as exc:
                LOGGER.warning("Failed to load GTK theme: %s", exc)
                return False
            self._xmms_css_provider = css_provider
        Gtk.StyleContext.add_provider_for_display(
            display, css_provider, Gtk.STYLE_PROVIDER_PRIORITY_USER
        )
        return False
