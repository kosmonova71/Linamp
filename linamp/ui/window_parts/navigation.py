"""WinampWindow mixin: Track navigation, shuffle order and playback toggles."""

from __future__ import annotations

import gi

from .base import WindowBase

gi.require_version("Gdk", "4.0")
gi.require_version("Gtk", "4.0")
gi.require_version("Gst", "1.0")
import random

from gi.repository import Gtk

from ...core import (
    COVER_DOWNLOAD_AVAILABLE,
)


class NavigationMixin(WindowBase):
    """Track navigation, shuffle order and playback toggles.

    The methods are split out of ``WinampWindow`` so the window class
    stays readable; they keep operating on the window instance and
    share its playback, playlist and settings state.
    """

    def play_track(self, index):
        with self._playlist_lock:
            if not 0 <= index < len(self.playlist):
                return False
            item = self.playlist[index]
            self._capture_podcast_position()
            self._stop_podcast_position_tracking()
            self._is_radio_stream = False
            old_track = self.current_track
            self.current_track = index
        if not self.play_file(item.path, item.title):
            with self._playlist_lock:
                self.current_track = old_track
            return False
        if self.shuffle_mode and index in self.shuffled_indices:
            self.shuffle_position = self.shuffled_indices.index(index)
        if hasattr(self, "playlist_tab") and hasattr(
            self.playlist_tab, "selection_model"
        ):
            filtered_pos = self._get_filtered_position(index)
            if filtered_pos is not None:
                self.playlist_tab.selection_model.set_selected(filtered_pos)
            else:
                self.playlist_tab.selection_model.set_selected(
                    Gtk.INVALID_LIST_POSITION
                )
            if hasattr(self.playlist_tab, "refresh_playback_indicator"):
                self.playlist_tab.refresh_playback_indicator()
        self._load_cover_art_for_track(item)
        if hasattr(self, "lyrics_tab"):
            self.lyrics_tab.load_lyrics(item)
        return True

    def on_prev(self, button):
        self.play_previous_track()

    def play_previous_track(self):
        if not self.playlist:
            return
        prev_index = self.get_previous_track_index()
        if prev_index is not None:
            self.play_track(prev_index)

    def get_previous_track_index(self):
        if not self.playlist:
            return None
        if self.shuffle_mode:
            return self.get_previous_shuffled_index()
        else:
            if self.current_track > 0:
                return self.current_track - 1
            elif self.repeat_mode == "all":
                return len(self.playlist) - 1
            else:
                return None

    def get_previous_shuffled_index(self):
        if not self.shuffled_indices:
            self.regenerate_shuffle_list()
        if self.shuffle_position > 0:
            self.shuffle_position -= 1
            return self.shuffled_indices[self.shuffle_position]
        elif self.repeat_mode == "all":
            self.shuffle_position = len(self.shuffled_indices) - 1
            return self.shuffled_indices[self.shuffle_position]
        else:
            return None

    def toggle_repeat_mode(self):
        modes = ["none", "one", "all"]
        current_index = modes.index(self.repeat_mode)
        self.repeat_mode = modes[(current_index + 1) % len(modes)]
        self.update_status_display()
        self.auto_save_settings()

    def toggle_shuffle_mode(self):
        self.shuffle_mode = not self.shuffle_mode
        if self.shuffle_mode:
            self.regenerate_shuffle_list()
            if self.current_track >= 0:
                try:
                    self.shuffle_position = self.shuffled_indices.index(
                        self.current_track
                    )
                except ValueError:
                    self.shuffle_position = -1
        self.update_status_display()
        self.auto_save_settings()

    def toggle_auto_play_next(self):
        self.auto_play_next = not self.auto_play_next
        self.update_status_display()
        self.auto_save_settings()

    def toggle_beat_aware(self):
        self.beat_aware_enabled = not self.beat_aware_enabled
        self.update_status_display()
        self.auto_save_settings()

    def toggle_auto_download_covers(self):
        if not COVER_DOWNLOAD_AVAILABLE:
            self.set_status_message(
                "Cover download requires requests and PIL libraries"
            )
            return
        self.auto_download_covers = not self.auto_download_covers
        status = "enabled" if self.auto_download_covers else "disabled"
        self.set_status_message(f"Auto download covers: {status}")
        self.update_status_display()
        self.auto_save_settings()
        if (
            self.auto_download_covers
            and self.current_track >= 0
            and self.current_track < len(self.playlist)
        ):
            current_item = self.playlist[self.current_track]
            self._load_cover_art_for_track(current_item)

    def toggle_show_cover_art(self):
        self.settings.show_cover_art = not self.settings.show_cover_art
        self.player_tab.cover_art_widget.set_visible(self.settings.show_cover_art)
        if self.settings.show_cover_art and 0 <= self.current_track < len(
            self.playlist
        ):
            current_item = self.playlist[self.current_track]
            self._load_cover_art_for_track(current_item)
        status = "enabled" if self.settings.show_cover_art else "disabled"
        self.set_status_message(f"Cover art: {status}")
        self.update_status_display()
        self.auto_save_settings()

    def on_next(self, button):
        self.play_next_track()

    def play_next_track(self):
        if not self.playlist:
            return
        next_index = self.get_next_track_index()
        if next_index is not None:
            self.play_track(next_index)

    def get_next_track_index(self):
        if not self.playlist:
            return None
        if self.shuffle_mode:
            return self.get_next_shuffled_index()
        else:
            if self.current_track < len(self.playlist) - 1:
                return self.current_track + 1
            elif self.repeat_mode == "all":
                return 0
            else:
                return None

    def get_next_shuffled_index(self):
        if not self.shuffled_indices:
            self.regenerate_shuffle_list()
        if self.shuffle_position < len(self.shuffled_indices) - 1:
            self.shuffle_position += 1
            return self.shuffled_indices[self.shuffle_position]
        elif self.repeat_mode == "all":
            self.regenerate_shuffle_list()
            self.shuffle_position = 0
            return self.shuffled_indices[0]
        else:
            return None

    def regenerate_shuffle_list(self):
        with self._playlist_lock:
            self.original_indices = list(range(len(self.playlist)))
            self.shuffled_indices = self.original_indices.copy()
            random.shuffle(self.shuffled_indices)
            self.shuffle_position = -1

    def _regenerate_shuffle_list_if_needed(self) -> None:
        """Regenerate shuffle list after any structural playlist change
        when shuffle mode is active.
        """
        with self._playlist_lock:
            if not self.shuffle_mode:
                return
            self.regenerate_shuffle_list()
            if self.current_track >= 0:
                try:
                    self.shuffle_position = self.shuffled_indices.index(
                        self.current_track
                    )
                except ValueError:
                    self.shuffle_position = -1
