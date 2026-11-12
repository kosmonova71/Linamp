from __future__ import annotations

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import gi

gi.require_version("Gdk", "4.0")
gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk

from ...core import *
from ...media import *
from ...models import *
from ...podcasts import *
from ...radio import *
from .. import _create_labeled_icon_button

mutagen: Any = None
File: Any = None
ID3: Any = None
MP3: Any = None
FLAC: Any = None
MP4: Any = None

try:
    import mutagen as _mutagen
    from mutagen import File as _File
    from mutagen.flac import FLAC as _FLAC
    from mutagen.id3 import ID3 as _ID3
    from mutagen.mp3 import MP3 as _MP3
    from mutagen.mp4 import MP4 as _MP4

    mutagen = _mutagen
    File = _File
    ID3 = _ID3
    MP3 = _MP3
    FLAC = _FLAC
    MP4 = _MP4
except ImportError:
    pass


class LyricsTab(Gtk.Box):
    _executor: ThreadPoolExecutor | None = None

    def __init__(self, player: Any) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self.player = player
        self.add_css_class("lyrics-tab")
        self._lyrics_manager = LyricsManager()
        self._executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="LyricsTab"
        )
        self._current_audio_path = ""
        self._current_source = "none"
        self._is_synced = False
        self._fetch_in_progress = False
        self._cleanup_complete = False
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        main_box.set_margin_top(8)
        main_box.set_margin_bottom(8)
        main_box.set_margin_start(8)
        main_box.set_margin_end(8)
        self.append(main_box)
        toolbar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        toolbar.set_halign(Gtk.Align.CENTER)
        main_box.append(toolbar)
        self.source_label = Gtk.Label(label="Source: none")
        self.source_label.add_css_class("lyrics-source-label")
        self.source_label.set_halign(Gtk.Align.START)
        self.source_label.set_hexpand(True)
        toolbar.append(self.source_label)
        self.synced_badge = Gtk.Label(label="")
        self.synced_badge.add_css_class("lyrics-synced-badge")
        self.synced_badge.set_visible(False)
        toolbar.append(self.synced_badge)
        fetch_btn = _create_labeled_icon_button(
            "Fetch Online", "network-workgroup-symbolic"
        )
        fetch_btn.add_css_class("lyrics-action-btn")
        fetch_btn.connect("clicked", self._on_fetch_online)
        toolbar.append(fetch_btn)
        reload_btn = _create_labeled_icon_button("Reload", "view-refresh-symbolic")
        reload_btn.add_css_class("lyrics-action-btn")
        reload_btn.connect("clicked", self._on_reload)
        toolbar.append(reload_btn)
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_hexpand(True)
        scrolled.set_vexpand(True)
        scrolled.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        main_box.append(scrolled)
        self.text_view = Gtk.TextView()
        self.text_view.set_editable(False)
        self.text_view.set_cursor_visible(False)
        self.text_view.set_wrap_mode(Gtk.WrapMode.WORD)
        self.text_view.set_left_margin(12)
        self.text_view.set_right_margin(12)
        self.text_view.set_top_margin(12)
        self.text_view.set_bottom_margin(12)
        self.text_view.add_css_class("lyrics-text-view")
        scrolled.set_child(self.text_view)
        self.buffer = self.text_view.get_buffer()
        self._show_placeholder()

    def cleanup(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        self._cleanup_complete = True

    def _show_placeholder(self) -> None:
        self.buffer.set_text("No lyrics loaded.\nPlay a track to load lyrics.")
        self.source_label.set_label("Source: none")
        self.synced_badge.set_visible(False)
        self._current_source = "none"
        self._is_synced = False

    def _on_fetch_online(self, button: Gtk.Button) -> None:
        if not self._current_audio_path:
            return
        item = self._get_current_item()
        if not item:
            return
        request_path = self._current_audio_path
        self._fetch_in_progress = True
        self.source_label.set_label("Source: fetching...")
        self.synced_badge.set_visible(False)

        def _task():
            artist = self._get_artist(item)
            title = self._clean_lyrics_title(
                item.title or os.path.splitext(os.path.basename(item.path))[0]
            )
            album = self._get_album(item)
            duration = self._get_duration(item)
            text, is_synced = self._lyrics_manager._get_lrclib_lyrics(
                artist, title, album, duration
            )
            source = "lrclib" if text else "none"
            cache_key = self._lyrics_manager._cache_key(item.path, artist, title)
            self._lyrics_manager._cache_set(
                cache_key,
                {"text": text, "source": source, "synced": is_synced},
            )
            return text, source, is_synced

        def _on_done(future):
            self._fetch_in_progress = False
            if request_path != getattr(self, "_current_audio_path", ""):
                return False
            if getattr(self, "_cleanup_in_progress", False):
                return False
            if getattr(self, "_cleanup_complete", False):
                return False
            if getattr(self.player, "_cleanup_in_progress", False):
                return False
            if getattr(self.player, "_cleanup_complete", False):
                return False
            try:
                text, source, is_synced = future.result()
            except Exception as exc:  # noqa: BLE001 - future.result() may raise any worker error
                logging.getLogger(__name__).debug("Lyrics fetch failed: %s", exc)
                text, source, is_synced = "", "none", False
            self._current_source = source
            self._is_synced = is_synced
            self._update_display(text, source, is_synced)
            return False

        if self._executor is None:
            return
        future = self._executor.submit(_task)
        future.add_done_callback(lambda f: GLib.idle_add(_on_done, f))

    def _on_reload(self, button: Gtk.Button) -> None:
        item = self._get_current_item()
        if not item:
            return
        self.load_lyrics(item)

    @staticmethod
    def _clean_lyrics_title(title: str) -> str:
        if not title:
            return ""
        title = os.path.splitext(title)[0]
        title = re.sub(r"^(\d+)[\s\.\-_]+", "", title)
        parts = re.split(r"\s*-\s*", title)
        parts = [p for p in parts if not re.match(r"^\d+$", p.strip())]
        if len(parts) > 2:
            return parts[-1].strip()
        if len(parts) == 2:
            last = parts[-1].strip()
            if re.search(
                r"\b(Vol\.?|Volume|Pt\.?|Part|Compilation|Anthology|Collection|Series)\b",
                last,
                re.IGNORECASE,
            ):
                return parts[0].strip()
        return title.strip()

    def load_lyrics(self, item: Any) -> None:
        request_path = item.path
        self._current_audio_path = item.path
        self.source_label.set_label("Source: loading...")
        self.synced_badge.set_visible(False)
        self.buffer.set_text("Loading lyrics...")

        def _task():
            artist = self._get_artist(item)
            title = self._clean_lyrics_title(
                item.title or os.path.splitext(os.path.basename(item.path))[0]
            )
            album = self._get_album(item)
            duration = self._get_duration(item)
            return self._lyrics_manager._fetch_lyrics(
                item.path, artist, title, album, duration
            )

        def _on_done(future):
            if request_path != getattr(self, "_current_audio_path", ""):
                return
            if getattr(self.player, "_cleanup_in_progress", False):
                return
            if getattr(self, "_cleanup_complete", False):
                return
            if getattr(self.player, "_cleanup_complete", False):
                return
            try:
                text, source, is_synced = future.result()
            except Exception as exc:  # noqa: BLE001 - future.result() may raise any worker error
                logging.getLogger(__name__).debug("Lyrics fetch failed: %s", exc)
                text, source, is_synced = "", "none", False
            self._current_source = source
            self._is_synced = is_synced
            self._update_display(text, source, is_synced)

        if self._executor is None:
            return
        future = self._executor.submit(_task)
        future.add_done_callback(lambda f: GLib.idle_add(_on_done, f))

    def _update_display(self, text: str, source: str, is_synced: bool) -> None:
        self.source_label.set_label(f"Source: {source}")
        if is_synced:
            self.synced_badge.set_label("Synced")
            self.synced_badge.set_visible(True)
        else:
            self.synced_badge.set_visible(False)
        if text:
            self.buffer.set_text(text)
        else:
            self.buffer.set_text("No lyrics available for this track.")
            self.source_label.set_label("Source: none")

    def _get_current_item(self) -> Any:
        if hasattr(
            self.player, "current_track"
        ) and 0 <= self.player.current_track < len(self.player.playlist):
            return self.player.playlist[self.player.current_track]
        return None

    def _get_artist(self, item: Any) -> str:
        if mutagen is None:
            return ""
        try:
            audio = File(item.path)
            if audio is None:
                return ""
            if isinstance(audio, MP3):
                frame = audio.tags.get("TPE1") if audio.tags else None
                if frame:
                    text = getattr(frame, "text", None)
                    if text:
                        return str(text[0]).strip()
            elif isinstance(audio, FLAC):
                values = audio.get("artist")
                if values:
                    return str(values[0]).strip()
            elif isinstance(audio, MP4):
                values = audio.get("\xa9ART")
                if values:
                    return str(values[0]).strip()
        except (MutagenError, OSError, AttributeError) as exc:
            logging.getLogger(__name__).debug("Artist lookup failed: %s", exc)
        return ""

    def _get_album(self, item: Any) -> str:
        if mutagen is None:
            return ""
        try:
            audio = File(item.path)
            if audio is None:
                return ""
            if isinstance(audio, MP3):
                frame = audio.tags.get("TALB") if audio.tags else None
                if frame:
                    text = getattr(frame, "text", None)
                    if text:
                        return str(text[0]).strip()
            elif isinstance(audio, FLAC):
                values = audio.get("album")
                if values:
                    return str(values[0]).strip()
            elif isinstance(audio, MP4):
                values = audio.get("\xa9alb")
                if values:
                    return str(values[0]).strip()
        except (MutagenError, OSError, AttributeError) as exc:
            logging.getLogger(__name__).debug("Album lookup failed: %s", exc)
        return ""

    def _get_duration(self, item: Any) -> int:
        duration = getattr(item, "duration", 0)
        if duration and duration > 0:
            return int(duration)
        if mutagen is None:
            return 0
        try:
            audio = File(item.path)
            if (
                audio is not None
                and hasattr(audio, "info")
                and hasattr(audio.info, "length")
            ):
                return int(audio.info.length)
        except (MutagenError, OSError, AttributeError) as exc:
            logging.getLogger(__name__).debug("Duration lookup failed: %s", exc)
        return 0
