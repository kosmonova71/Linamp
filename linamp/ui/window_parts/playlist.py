"""WinampWindow mixin: Playlist storage, scanning and display updates."""

from __future__ import annotations

import gi

from .base import WindowBase

gi.require_version("Gdk", "4.0")
gi.require_version("Gtk", "4.0")
gi.require_version("Gst", "1.0")
import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path

from gi.repository import GLib

from ...core import (
    AUDIO_EXTENSIONS,
    MAX_REPORTED_SCAN_ERRORS,
    PLAYLIST_FILE,
    _atomic_write_json,
    _get_audio_duration,
)
from ...models import (
    PlaylistItem,
)


class PlaylistMixin(WindowBase):
    """Playlist storage, scanning and display updates.

    The methods are split out of ``WinampWindow`` so the window class
    stays readable; they keep operating on the window instance and
    share its playback, playlist and settings state.
    """

    def _invalidate_playlist_index(self) -> None:
        with self._playlist_lock:
            self._playlist_index = {}

    def _get_playlist_index(self) -> dict[str, int]:
        with self._playlist_lock:
            if not self._playlist_index or len(self._playlist_index) != len(
                self.playlist
            ):
                self._playlist_index = {
                    item.id: i for i, item in enumerate(self.playlist)
                }
            return self._playlist_index

    def clear_playlist(self) -> bool:
        with self._playlist_lock:
            was_current = 0 <= self.current_track < len(self.playlist)
            self._clear_playlist_store()
            self.playlist.clear()
            self.current_track = -1
            self._invalidate_playlist_index()
            self._regenerate_shuffle_list_if_needed()
        if was_current:
            self.on_stop(None)
        self.save_playlist()
        return True

    def remove_playlist_index(self, index: int) -> bool:
        with self._playlist_lock:
            if index < 0 or index >= len(self.playlist):
                return False
            was_current = index == self.current_track
            self.playlist.pop(index)
            need_stop = self._fix_current_track_after_remove(index, was_current)
            self._invalidate_playlist_index()
            self._update_playlist_display()
        if need_stop:
            self.on_stop(None)
        self.save_playlist()
        return True

    def _fix_current_track_after_remove(
        self, removed_idx: int, was_current: bool, skip_shift: bool = False
    ) -> bool:
        """Fix current track after removing a playlist item.
        Returns True if on_stop() should be called (was_current was True).
        Must be called while holding ``self._playlist_lock``.
        """
        if was_current:
            self.current_track = -1
            return True
        if not 0 <= self.current_track < len(self.playlist):
            self.current_track = 0 if self.playlist else -1
            return False
        if not skip_shift and removed_idx < self.current_track:
            self.current_track -= 1
        if self.current_track >= len(self.playlist):
            self.current_track = len(self.playlist) - 1
        if self.current_track < 0 and self.playlist:
            self.current_track = 0
        return False

    def add_to_playlist(self, file_paths):
        added_items = []
        with self._playlist_lock:
            for path in file_paths:
                try:
                    normalized_path = os.path.abspath(os.path.expanduser(str(path)))
                    if "\0" in normalized_path:
                        continue
                    real_path = os.path.realpath(normalized_path)
                    if not os.path.exists(real_path):
                        continue
                    if not os.path.isfile(real_path):
                        continue
                    if real_path.startswith(("/proc/", "/sys/")):
                        continue
                except (OSError, ValueError):
                    continue
                item = PlaylistItem(
                    path=normalized_path,
                    title=os.path.basename(normalized_path),
                    duration=_get_audio_duration(real_path),
                )
                self.playlist.append(item)
                added_items.append(item)
            playlist_tab = getattr(self, "playlist_tab", None)
            if playlist_tab is not None and added_items:
                self._refresh_playlist_store()
                if hasattr(playlist_tab, "update_statistics"):
                    playlist_tab.update_statistics()
                self._invalidate_playlist_index()
            self._regenerate_shuffle_list_if_needed()
        self.save_playlist()

    def add_folder_to_playlist(self, folder_path):
        try:
            normalized_path = os.path.abspath(os.path.expanduser(str(folder_path)))
            if "\0" in normalized_path:
                return
            if not os.path.exists(normalized_path) or not os.path.isdir(
                normalized_path
            ):
                return
        except (OSError, ValueError):
            return
        GLib.idle_add(
            self.set_status_message,
            f"Scanning folder: {os.path.basename(normalized_path)}...",
        )
        with self._folder_scan_lock:
            previous = getattr(self, "_folder_scan_cancel", None)
            if previous is not None:
                previous.set()
            self._folder_scan_generation += 1
            generation = self._folder_scan_generation
            self._folder_scan_cancel = threading.Event()
        cancel = self._folder_scan_cancel

        def scan_folder():
            added_items = []
            errors = []
            try:
                for root, dirs, files in os.walk(normalized_path, followlinks=False):
                    if cancel.is_set():
                        return
                    dirs[:] = [d for d in dirs if not d.startswith(".")]
                    for file in files:
                        if cancel.is_set():
                            return
                        try:
                            file_path = os.path.join(root, file)
                            file_ext = os.path.splitext(file)[1].lower()
                            if file_ext in AUDIO_EXTENSIONS:
                                rel_path = os.path.relpath(file_path, normalized_path)
                                title = rel_path if rel_path != file_path else file
                                item = PlaylistItem(
                                    path=file_path,
                                    title=title,
                                    duration=_get_audio_duration(file_path),
                                )
                                added_items.append(item)
                        except (OSError, ValueError) as e:
                            errors.append(f"{file}: {e}")
            except (OSError, PermissionError) as e:
                errors.append(f"Scan failed: {e}")
            if cancel.is_set():
                return
            GLib.idle_add(
                self._on_folder_scan_complete, added_items, errors, generation
            )

        def tracked_scan_folder():
            try:
                scan_folder()
            finally:
                with self._folder_scan_lock:
                    self._folder_scan_threads.discard(threading.current_thread())

        thread = threading.Thread(
            target=tracked_scan_folder, daemon=True, name="FolderScan"
        )
        with self._folder_scan_lock:
            self._folder_scan_threads.add(thread)
        thread.start()

    def _on_folder_scan_complete(self, added_items, errors=None, generation=None):
        if generation is not None and generation != self._folder_scan_generation:
            return False
        if getattr(self, "_cleanup_complete", False):
            return False
        if getattr(self, "_cleanup_in_progress", False):
            return False
        with self._playlist_lock:
            if added_items:
                self.playlist.extend(added_items)
                playlist_tab = getattr(self, "playlist_tab", None)
                if playlist_tab is not None:
                    self._refresh_playlist_store()
                    if hasattr(playlist_tab, "update_statistics"):
                        playlist_tab.update_statistics()
                    self._invalidate_playlist_index()
                self._regenerate_shuffle_list_if_needed()
                self.save_playlist()
            else:
                GLib.idle_add(self.set_status_message, "No audio files found in folder")
        if errors:
            for error in errors[:MAX_REPORTED_SCAN_ERRORS]:
                logging.getLogger(__name__).warning("Folder scan error: %s", error)
            total_errors = len(errors)
            error_summary = "; ".join(errors[:MAX_REPORTED_SCAN_ERRORS])
            if total_errors > MAX_REPORTED_SCAN_ERRORS:
                error_summary += (
                    f"; ... and {total_errors - MAX_REPORTED_SCAN_ERRORS} more"
                )
            if added_items:
                GLib.idle_add(
                    self.set_status_message,
                    f"Added {len(added_items)} tracks to playlist "
                    f"({total_errors} errors: {error_summary})",
                )
            else:
                GLib.idle_add(
                    self.set_status_message,
                    f"Folder scan completed with {total_errors} errors: "
                    f"{error_summary}",
                )
        return False

    def _cancel_folder_scans(self) -> None:
        with self._folder_scan_lock:
            self._folder_scan_generation += 1
            self._folder_scan_cancel = threading.Event()
            self._folder_scan_cancel.set()
            threads = list(self._folder_scan_threads)
        deadline = time.monotonic() + 1.0
        for thread in threads:
            if thread is threading.current_thread():
                continue
            thread.join(timeout=max(0.0, deadline - time.monotonic()))

    def save_playlist(self, filepath: str | None = None) -> bool:
        try:
            if not hasattr(self, "playlist"):
                return False
            if not filepath:
                filepath = str(PLAYLIST_FILE)
            else:
                filepath = os.path.abspath(os.path.expanduser(filepath))
            with self._playlist_lock:
                playlist_snapshot = list(self.playlist)
            playlist_data = []
            for item in playlist_snapshot:
                if (
                    hasattr(item, "to_dict")
                    and callable(item.to_dict)
                    and hasattr(item, "path")
                    and item.path
                    and os.path.exists(item.path)
                ):
                    playlist_data.append(item.to_dict())
            return _atomic_write_json(Path(filepath), playlist_data, mode=0o600)
        except (OSError, TypeError) as e:
            logging.getLogger(__name__).debug(
                "Failed to save playlist: %s", e, exc_info=True
            )
            return False

    def _clear_playlist_store(self) -> bool:
        with self._playlist_lock:
            playlist_tab = getattr(self, "playlist_tab", None)
            if playlist_tab is None or not hasattr(
                playlist_tab, "playlist_store"
            ):
                return False
            store = playlist_tab.playlist_store
            store.freeze_notify()
            try:
                store.splice(0, store.get_n_items(), [])
            except (AttributeError, TypeError):
                while store.get_n_items() > 0:
                    store.remove(store.get_n_items() - 1)
            store.thaw_notify()
            return True

    def _build_playlist_display_names(self) -> list[str]:
        """Build display strings (title|id) from self.playlist.
        Must be called while holding ``self._playlist_lock``.
        """
        display_names: list[str] = []
        for item in self.playlist:
            if isinstance(item, PlaylistItem):
                display_names.append(f"{item.get_display_name()}|{item.id}")
            elif isinstance(item, dict):
                title = item.get("title") or os.path.basename(item.get("path", ""))
                path = item.get("path", "")
                track_id = (
                    item.get("id")
                    or hashlib.md5(f"{path}:{title}".encode()).hexdigest()[:32]
                )
                display_names.append(f"{title}|{track_id}")
            else:
                s = str(item)
                track_id = hashlib.md5(s.encode()).hexdigest()[:32]
                display_names.append(f"{s}|{track_id}")
        return display_names

    def _refresh_playlist_store(self) -> None:
        """Atomically rebuild the entire playlist_store from self.playlist.
        Always performs a full clear + repopulate so the store is guaranteed
        to match ``self.playlist`` exactly — no incremental splice can leave
        stale entries after a sort, shuffle, or reorder.
        Must be called while holding ``self._playlist_lock``.
        """
        playlist_tab = getattr(self, "playlist_tab", None)
        if playlist_tab is None or not hasattr(
            playlist_tab, "playlist_store"
        ):
            return
        store = playlist_tab.playlist_store
        display_names = self._build_playlist_display_names()
        store.freeze_notify()
        try:
            store.splice(0, store.get_n_items(), display_names)
        except (AttributeError, TypeError):
            while store.get_n_items() > 0:
                store.remove(store.get_n_items() - 1)
            for name in display_names:
                store.append(name)
        store.thaw_notify()

    def _update_playlist_display(self) -> bool:
        with self._playlist_lock:
            playlist_tab = getattr(self, "playlist_tab", None)
            if playlist_tab is None or not hasattr(
                playlist_tab, "playlist_store"
            ):
                return False
            if not self.playlist:
                self._clear_playlist_store()
                return True
            self._refresh_playlist_store()
            return True

    def load_playlist(self, filepath: str | None = None) -> bool:
        if not filepath:
            playlist_dir = os.path.expanduser("~/.config/linamp")
            filepath = os.path.join(playlist_dir, "playlist.json")
        else:
            filepath = os.path.abspath(os.path.expanduser(filepath))
        if not os.path.exists(filepath):
            return False
        if not os.access(filepath, os.R_OK):
            return False
        try:
            with open(filepath, encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    return False
            if not isinstance(data, list):
                return False
            if len(data) > 100000:
                return False
        except OSError:
            return False
        playlist = []
        invalid_items = 0
        missing_files = []
        for i, item_data in enumerate(data, 1):
            if not isinstance(item_data, dict):
                invalid_items += 1
                continue
            try:
                item = PlaylistItem.from_dict(item_data)
                if item.exists():
                    playlist.append(item)
                else:
                    missing_files.append(item.path)
                    invalid_items += 1
            except (TypeError, ValueError):
                invalid_items += 1
        if invalid_items > 0:
            msg_parts = []
            if invalid_items - len(missing_files) > 0:
                msg_parts.append(
                    f"{invalid_items - len(missing_files)} invalid entr{'y' if invalid_items - len(missing_files) == 1 else 'ies'}"
                )
            if missing_files:
                msg_parts.append(
                    f"{len(missing_files)} missing file{'s' if len(missing_files) != 1 else ''}"
                )
            self.set_status_message(
                f"Playlist loaded with warnings: {', '.join(msg_parts)}"
            )
        if self.playing:
            self.stop()
        with self._playlist_lock:
            self.playlist = playlist
            self.current_track = -1
            self._invalidate_playlist_index()
            self._regenerate_shuffle_list_if_needed()
            playlist_tab = getattr(self, "playlist_tab", None)
            if playlist_tab is not None:
                self._clear_playlist_store()
                self._update_playlist_display()
        return len(playlist) > 0

    def cleanup_invalid_tracks(self):
        with self._playlist_lock:
            original_count = len(self.playlist)
            playlist_snapshot = list(self.playlist)
            current_path = None
            if 0 <= self.current_track < original_count:
                try:
                    current_path = os.path.realpath(
                        self.playlist[self.current_track].path
                    )
                except (OSError, TypeError, ValueError):
                    current_path = None
        valid_tracks = [item for item in playlist_snapshot if item.exists()]
        if len(valid_tracks) == original_count:
            return
        was_current = current_path is not None and not any(
            os.path.realpath(item.path) == current_path for item in valid_tracks
        )
        with self._playlist_lock:
            self.playlist = valid_tracks
            if was_current:
                self.current_track = -1
            elif current_path is not None:
                self.current_track = next(
                    (
                        i
                        for i, item in enumerate(self.playlist)
                        if os.path.realpath(item.path) == current_path
                    ),
                    -1,
                )
            else:
                self.current_track = 0 if self.playlist else -1
            self._invalidate_playlist_index()
            self._update_playlist_display()
            self._regenerate_shuffle_list_if_needed()
        if was_current:
            self.on_stop(None)
        self.save_playlist()
        self.set_status_message(
            f"Removed {original_count - len(valid_tracks)} invalid tracks"
        )

    def focus_playlist_search(self):
        stack = getattr(self, "stack", None)
        if stack is not None:
            stack.set_visible_child_name("playlist")
        playlist_tab = getattr(self, "playlist_tab", None)
        if (
            playlist_tab is not None
            and hasattr(playlist_tab, "search_entry")
            and playlist_tab.search_entry
        ):
            playlist_tab.search_entry.grab_focus()

    def _get_filtered_position(self, playlist_index: int) -> int | None:
        """Map a playlist index to a position in the filtered selection model.
        Returns None if the item is not visible in the current filter
        (selection should be cleared). Otherwise returns the filtered
        position (>= 0) in the filtered model.
        """
        if not hasattr(self, "playlist_tab") or self.playlist_tab is None:
            return playlist_index
        tab = self.playlist_tab
        if not hasattr(tab, "filter_model") or tab.filter_model is None:
            return playlist_index
        try:
            target_id = self.playlist[playlist_index].id
        except (IndexError, AttributeError):
            return playlist_index
        n = tab.filter_model.get_n_items()
        for pos in range(n):
            item = tab.filter_model.get_item(pos)
            if item is None:
                continue
            try:
                raw = item.get_string() if hasattr(item, "get_string") else str(item)
                parts = raw.split("|", 1)
                item_id = parts[1] if len(parts) > 1 else None
                if item_id == target_id:
                    return pos
            except (AttributeError, ValueError):
                continue
        return None
