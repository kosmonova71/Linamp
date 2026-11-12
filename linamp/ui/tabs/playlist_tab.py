from __future__ import annotations

import logging
import os
import random
import tempfile
import threading
from pathlib import Path
from typing import Any

import gi

gi.require_version("Gdk", "4.0")
gi.require_version("Gtk", "4.0")
from gi.repository import Gio, GLib, Gtk

from ...core import *
from ...core import METADATA_FIXER_AVAILABLE, MetadataFixer
from ...media import *
from ...models import *
from ...podcasts import *
from ...radio import *
from .. import _create_labeled_icon_button


class PlaylistTab(Gtk.Box):
    def __init__(self, player: Any) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self.player = player
        self.add_css_class("playlist-tab")
        self.playlist_store = Gtk.StringList()
        self.filtered_store: Any = None
        self.filter_model: Any = None
        self._search_debounce_id: int | None = None
        self._playlist_by_id: dict[str, PlaylistItem] = {}
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        main_box.set_margin_top(16)
        main_box.set_margin_bottom(16)
        main_box.set_margin_start(16)
        main_box.set_margin_end(16)
        self.append(main_box)
        search_section = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        main_box.append(search_section)
        search_header = Gtk.Label(label="Search Playlist")
        search_header.add_css_class("section-header")
        search_header.set_halign(Gtk.Align.START)
        search_section.append(search_header)
        search_container = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        search_section.append(search_container)
        self.search_entry = Gtk.Entry()
        self.search_entry.set_placeholder_text("Type to filter playlist...")
        self.search_entry.set_hexpand(True)
        self.search_entry.add_css_class("search-entry")
        self.search_entry.connect("changed", self.on_search_changed)
        search_container.append(self.search_entry)
        clear_search_btn = _create_labeled_icon_button("Clear", "edit-clear-symbolic")
        clear_search_btn.connect("clicked", self.on_clear_search)
        search_container.append(clear_search_btn)
        self.search_entry.connect(
            "activate", lambda _e: self.filter_model.set_filter(self.filter_model.get_filter())
        )
        toolbar_section = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        main_box.append(toolbar_section)
        toolbar_header = Gtk.Label(label="Playlist Actions")
        toolbar_header.add_css_class("section-header")
        toolbar_header.set_halign(Gtk.Align.START)
        toolbar_section.append(toolbar_header)
        main_toolbar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        main_toolbar.set_homogeneous(True)
        toolbar_section.append(main_toolbar)
        add_btn = _create_labeled_icon_button("Add Files", "list-add-symbolic")
        add_btn.connect("clicked", self.on_add_files)
        main_toolbar.append(add_btn)
        add_folder_btn = _create_labeled_icon_button(
            "Add Folder", "folder-open-symbolic"
        )
        add_folder_btn.connect("clicked", self.on_add_folder)
        main_toolbar.append(add_folder_btn)
        remove_btn = _create_labeled_icon_button("Remove", "list-remove-symbolic")
        remove_btn.connect("clicked", self.on_remove)
        main_toolbar.append(remove_btn)
        clear_btn = _create_labeled_icon_button("Clear All", "edit-delete-symbolic")
        clear_btn.connect("clicked", self.on_clear)
        main_toolbar.append(clear_btn)
        shuffle_btn = _create_labeled_icon_button(
            "Shuffle", "media-playlist-shuffle-symbolic"
        )
        shuffle_btn.connect("clicked", self.on_shuffle)
        main_toolbar.append(shuffle_btn)
        secondary_toolbar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        secondary_toolbar.set_homogeneous(True)
        toolbar_section.append(secondary_toolbar)
        sort_dropdown = Gtk.DropDown.new_from_strings(
            ["Default Order", "By Title", "By Path", "By Duration"]
        )
        sort_dropdown.add_css_class("sort-dropdown")
        sort_dropdown.connect("notify::selected", self.on_sort_changed)
        secondary_toolbar.append(sort_dropdown)
        self.sort_dropdown = sort_dropdown
        import_btn = _create_labeled_icon_button("Import", "document-open-symbolic")
        import_btn.connect("clicked", self.on_import_m3u)
        secondary_toolbar.append(import_btn)
        export_btn = _create_labeled_icon_button("Export", "document-save-symbolic")
        export_btn.connect("clicked", self.on_export_m3u)
        secondary_toolbar.append(export_btn)
        remove_dups_btn = _create_labeled_icon_button(
            "Remove Dups", "view-refresh-symbolic"
        )
        remove_dups_btn.connect("clicked", self.on_remove_duplicates)
        secondary_toolbar.append(remove_dups_btn)
        if METADATA_FIXER_AVAILABLE:
            fix_tags_btn = _create_labeled_icon_button(
                "Complete Tags", "document-edit-symbolic"
            )
            fix_tags_btn.set_tooltip_text(
                "Automatically complete missing ID3/metadata tags from filenames"
            )
            fix_tags_btn.connect("clicked", self.on_fix_tags)
            secondary_toolbar.append(fix_tags_btn)
            self.fix_tags_btn: Gtk.Button | None = fix_tags_btn
        else:
            self.fix_tags_btn = None
        stats_container = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        stats_container.set_halign(Gtk.Align.END)
        stats_container.set_margin_top(8)
        toolbar_section.append(stats_container)
        self.stats_label = Gtk.Label(label="0 tracks")
        self.stats_label.add_css_class("stats-label")
        stats_container.append(self.stats_label)
        playlist_container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        playlist_container.set_margin_top(12)
        main_box.append(playlist_container)
        playlist_header = Gtk.Label(label="Track List")
        playlist_header.add_css_class("section-header")
        playlist_header.set_halign(Gtk.Align.START)
        playlist_container.append(playlist_header)
        scrolled = Gtk.ScrolledWindow(
            hscrollbar_policy=Gtk.PolicyType.AUTOMATIC,
            vscrollbar_policy=Gtk.PolicyType.AUTOMATIC,
        )
        scrolled.add_css_class("playlist-scrolled")
        scrolled.set_hexpand(True)
        scrolled.set_vexpand(True)
        playlist_container.append(scrolled)
        self.sort_model = Gtk.SortListModel(model=self.playlist_store)
        self.filter_model = Gtk.FilterListModel(model=self.sort_model)
        self.selection_model = Gtk.SingleSelection(model=self.filter_model)
        self.column_view = Gtk.ColumnView(model=self.selection_model)
        self.column_view.add_css_class("playlist-view")
        self.column_view.set_hexpand(True)
        self.column_view.set_vexpand(True)
        self.column_view.set_size_request(-1, 180)
        factory = Gtk.SignalListItemFactory()
        factory.connect("setup", self._on_factory_setup)
        factory.connect("bind", self._on_factory_bind)
        column = Gtk.ColumnViewColumn(title="Tracks", factory=factory)
        self.column_view.append_column(column)
        scrolled.set_child(self.column_view)
        self.selection_model.connect("selection-changed", self.on_selection_changed)
        self.column_view.connect("activate", self.on_row_activated)

    def on_add_files(self, button):
        dialog = Gtk.FileDialog(title="Add Files")
        dialog.set_modal(True)
        audio_filter = Gtk.FileFilter()
        audio_filter.set_name("Audio files")
        audio_filter.add_mime_type("audio/*")
        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(audio_filter)
        dialog.set_filters(filters)
        dialog.open_multiple(self.player, None, self._on_files_selected)

    def _on_files_selected(self, dialog, result):
        try:
            files = dialog.open_multiple_finish(result)
        except GLib.Error:
            return
        file_paths = [file.get_path() for file in files if file.get_path() is not None]
        if file_paths:
            self.player.add_to_playlist(file_paths)
        else:
            self.player.set_status_message("No files were selected.")

    def on_add_folder(self, button):
        dialog = Gtk.FileDialog(title="Add Music Folder")
        dialog.set_modal(True)
        dialog.select_folder(self.player, None, self._on_folder_selected)

    def _on_folder_selected(self, dialog, result):
        try:
            folder = dialog.select_folder_finish(result)
        except GLib.Error:
            return
        if folder:
            folder_path = folder.get_path()
            if folder_path:
                self.player.add_folder_to_playlist(folder_path)
            else:
                self.player.set_status_message(
                    "Folder selection did not return a valid path."
                )

    def on_remove(self, button):
        position = self.selection_model.get_selected()
        if position == Gtk.INVALID_LIST_POSITION:
            return
        item = self.selection_model.get_item(position)
        if item is None:
            return
        stored_string = item.get_string() if hasattr(item, "get_string") else str(item)
        track_id = self._resolve_track_id(stored_string)
        if track_id is None:
            return
        with self.player._playlist_lock:
            idx = self.player._get_playlist_index().get(track_id)
            if idx is not None:
                self.player.remove_playlist_index(idx)
                self.update_statistics()

    def on_clear(self, button):
        self.player.clear_playlist()
        self.update_statistics()

    def _on_factory_setup(self, factory, list_item):
        label = Gtk.Label()
        label.set_halign(Gtk.Align.START)
        label.set_ellipsize(3)  # PANGO_ELLIPSIZE_END
        label.set_hexpand(True)
        list_item.set_child(label)

    def _parse_stored_string(self, raw: str) -> tuple[str, str]:
        if "|" in raw:
            title, track_id = raw.split("|", 1)
            return title, track_id
        return raw, ""

    def _resolve_track_id(self, raw: str) -> str | None:
        title, track_id = self._parse_stored_string(raw)
        if track_id:
            return track_id
        with self.player._playlist_lock:
            for playlist_item in getattr(self.player, "playlist", []):
                if playlist_item.title == title or playlist_item.path.endswith(title):
                    return getattr(playlist_item, "id", None)
        return None

    def _on_factory_bind(self, factory, list_item):
        label = list_item.get_child()
        item = list_item.get_item()
        if label is None or item is None:
            return
        raw = item.get_string() if hasattr(item, "get_string") else str(item)
        title, track_id = self._parse_stored_string(raw)
        label.remove_css_class("playlist-current")
        current_id = None
        is_playing = bool(getattr(self.player, "playing", False))
        with self.player._playlist_lock:
            idx = getattr(self.player, "current_track", -1)
            playlist = getattr(self.player, "playlist", [])
            if 0 <= idx < len(playlist):
                current = playlist[idx]
                current_id = getattr(current, "id", None)
        if track_id and current_id and track_id == current_id:
            label.add_css_class("playlist-current")
            label.set_text(f"{'▶ ' if is_playing else '■ '}{title}")
        else:
            label.set_text(title)

    def on_selection_changed(self, selection, position, n_items):
        # Selection is UI-only; activation/double-click starts playback.
        return

    def on_search_changed(self, entry):
        search_text = (entry.get_text() or "").strip().lower()
        if self._search_debounce_id is not None:
            self.player._safe_source_remove(self._search_debounce_id)
            self._search_debounce_id = None

        def apply_filter():
            if not search_text:
                self.filter_model.set_filter(None)
            else:
                filter_obj = Gtk.CustomFilter.new(
                    lambda item: self._filter_func(item, search_text)
                )
                self.filter_model.set_filter(filter_obj)
            self._search_debounce_id = None
            return False

        self._search_debounce_id = self.player._register_timer(120, apply_filter)

    def _filter_func(self, item, search_text):
        if item is None:
            return False
        raw = item.get_string() if hasattr(item, "get_string") else str(item)
        title, track_id = (raw.split("|", 1) + [""])[:2] if "|" in raw else (raw, "")
        terms = [t for t in search_text.split() if t]
        if not terms:
            return True
        haystacks = [title.lower()]
        if track_id:
            with self.player._playlist_lock:
                pitem = self._playlist_by_id.get(track_id)
            if pitem is not None:
                haystacks.append((pitem.path or "").lower())
        for term in terms:
            if not any(term in h for h in haystacks):
                return False
        return True

    def on_clear_search(self, button):
        self.search_entry.set_text("")
        self.filter_model.set_filter(None)

    def on_shuffle(self, button):
        with self.player._playlist_lock:
            if not self.player.playlist:
                return
            random.shuffle(self.player.playlist)
            self.player._invalidate_playlist_index()
            self.player._update_playlist_display()
            self.player.save_playlist()

    def on_sort_changed(self, dropdown, pspec):
        selected = dropdown.get_selected()
        with self.player._playlist_lock:
            if selected == 1:
                self.player.playlist.sort(key=lambda x: x.title.lower())
            elif selected == 2:
                self.player.playlist.sort(key=lambda x: x.path.lower())
            elif selected == 3:
                self.player.playlist.sort(key=lambda x: x.duration)
            if selected != 0:
                self.player._invalidate_playlist_index()
                self.player._update_playlist_display()
                self.player.save_playlist()

    def update_statistics(self):
        playlist = getattr(self.player, "playlist", []) or []
        total_tracks = len(playlist)
        by_id = {}
        total_duration = 0
        known_durations = 0
        for item in playlist:
            if isinstance(item, PlaylistItem):
                if item.id:
                    by_id[item.id] = item
                dur = item.duration or 0
            elif isinstance(item, dict):
                tid = item.get("id")
                if tid:
                    try:
                        by_id[tid] = PlaylistItem.from_dict(item)
                    except (TypeError, ValueError):
                        pass
                dur = item.get("duration", 0) or 0
            else:
                continue
            try:
                dur_int = int(dur)
            except (TypeError, ValueError):
                dur_int = 0
            if dur_int > 0:
                total_duration += dur_int
                known_durations += 1
        self._playlist_by_id = by_id
        track_word = "track" if total_tracks == 1 else "tracks"
        if total_duration > 0:
            hours = total_duration // 3600
            minutes = (total_duration % 3600) // 60
            seconds = total_duration % 60
            if hours:
                duration_text = f"{hours}h {minutes}m"
            elif minutes:
                duration_text = f"{minutes}m {seconds}s"
            else:
                duration_text = f"{seconds}s"
            suffix = ""
            if 0 < known_durations < total_tracks:
                suffix = " (known)"
            stats_text = f"{total_tracks} {track_word} • {duration_text}{suffix}"
        else:
            stats_text = f"{total_tracks} {track_word}"
        self.stats_label.set_text(stats_text)

    def refresh_playback_indicator(self):
        if getattr(self.player, "_cleanup_in_progress", False) or getattr(
            self.player, "_cleanup_complete", False
        ):
            return False
        if hasattr(self.player, "_update_playlist_display"):
            with self.player._playlist_lock:
                self.player._update_playlist_display()
        return False

    def on_import_m3u(self, button):
        dialog = Gtk.FileDialog(title="Import M3U Playlist")
        dialog.set_modal(True)
        m3u_filter = Gtk.FileFilter()
        m3u_filter.set_name("M3U playlist files")
        m3u_filter.add_mime_type("audio/x-mpegurl")
        m3u_filter.add_pattern("*.m3u")
        m3u_filter.add_pattern("*.M3U")
        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(m3u_filter)
        dialog.set_filters(filters)
        dialog.open(self.player, None, self._on_m3u_import_selected)

    def _on_m3u_import_selected(self, dialog, result):
        try:
            file = dialog.open_finish(result)
        except GLib.Error:
            return
        if file:
            filepath = file.get_path()
            if filepath:
                self._import_m3u_file(filepath)

    def _import_m3u_file(self, filepath):
        try:
            with open(filepath, encoding="utf-8") as f:
                lines = f.readlines()
        except (OSError, UnicodeError):
            return
        m3u_dir = os.path.realpath(os.path.dirname(os.path.abspath(filepath)))
        imported_files = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if not os.path.isabs(line):
                line = os.path.join(m3u_dir, line)
            resolved_path = os.path.realpath(os.path.abspath(os.path.expanduser(line)))
            try:
                if os.path.commonpath((m3u_dir, resolved_path)) != m3u_dir:
                    continue
            except ValueError:
                continue
            if _validate_file_path(resolved_path, allow_symlinks=False):
                ext = os.path.splitext(resolved_path)[1].lower()
                if ext in AUDIO_EXTENSIONS:
                    imported_files.append(resolved_path)
        if imported_files:
            self.player.add_to_playlist(imported_files)

    def on_export_m3u(self, button):
        if not self.player.playlist:
            return
        dialog = Gtk.FileDialog(title="Export M3U Playlist")
        dialog.set_modal(True)
        m3u_filter = Gtk.FileFilter()
        m3u_filter.set_name("M3U playlist files")
        m3u_filter.add_mime_type("audio/x-mpegurl")
        m3u_filter.add_pattern("*.m3u")
        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(m3u_filter)
        dialog.set_filters(filters)
        dialog.set_initial_name("playlist.m3u")
        dialog.save(self.player, None, self._on_m3u_export_selected)

    def _on_m3u_export_selected(self, dialog, result):
        try:
            file = dialog.save_finish(result)
        except GLib.Error:
            return
        if file:
            filepath = file.get_path()
            if filepath:
                if not filepath.lower().endswith(".m3u"):
                    filepath += ".m3u"
                self._export_m3u_file(filepath)

    def _export_m3u_file(self, filepath):
        def _one_line(value: str) -> str:
            return str(value).replace("\r", " ").replace("\n", " ")

        lines = ["#EXTM3U"]
        for item in self.player.playlist:
            path = getattr(item, "path", "")
            if not path or not os.path.exists(path):
                continue
            duration = int(getattr(item, "duration", 0) or 0)
            title = getattr(item, "title", "") or os.path.splitext(os.path.basename(path))[0]
            if duration > 0:
                lines.append(f"#EXTINF:{duration},{_one_line(title)}")
            artist = getattr(item, "artist", "")
            if artist:
                lines.append(f"#EXTARTIST:{_one_line(artist)}")
            album = getattr(item, "album", "")
            if album:
                lines.append(f"#EXTALBUM:{_one_line(album)}")
            genre = getattr(item, "genre", "")
            if genre:
                lines.append(f"#EXTGENRE:{_one_line(genre)}")
            track_number = getattr(item, "track_number", 0) or 0
            if track_number > 0:
                lines.append(f"#EXTTRACK:{int(track_number)}")
            total_tracks = getattr(item, "total_tracks", 0) or 0
            if total_tracks > 0:
                lines.append(f"#EXTTOTAL:{int(total_tracks)}")
            description = getattr(item, "description", "")
            if description:
                lines.append(f"#EXTDESC:{_one_line(description)}")
            lines.append(_one_line(path))
        dest_dir = os.path.dirname(filepath) or "."
        fd, tmp = tempfile.mkstemp(
            dir=dest_dir, prefix=os.path.basename(filepath) + ".", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, filepath)
        except OSError:
            try:
                if os.path.exists(tmp):
                    os.unlink(tmp)
            except OSError:
                pass
            logging.getLogger(__name__).exception("M3U export failed for %s", filepath)
            if hasattr(self.player, "set_status_message"):
                self.player.set_status_message(f"Export failed: {filepath}")

    def on_remove_duplicates(self, button):
        with self.player._playlist_lock:
            if not self.player.playlist:
                return
            seen_paths = set()
            unique_playlist = []
            for item in self.player.playlist:
                try:
                    norm_path = os.path.realpath(item.path)
                except (OSError, TypeError, ValueError):
                    continue
                if norm_path in seen_paths:
                    continue
                seen_paths.add(norm_path)
                unique_playlist.append(item)
            if len(unique_playlist) == len(self.player.playlist):
                return
            current_path = None
            if 0 <= self.player.current_track < len(self.player.playlist):
                try:
                    current_path = os.path.realpath(
                        self.player.playlist[self.player.current_track].path
                    )
                except (OSError, TypeError, ValueError):
                    current_path = None
            was_current = current_path is not None and current_path not in seen_paths
            self.player.playlist = unique_playlist
            if was_current:
                self.player.current_track = -1
            elif current_path is not None:
                self.player.current_track = next(
                    (
                        i
                        for i, item in enumerate(self.player.playlist)
                        if os.path.realpath(item.path) == current_path
                    ),
                    -1,
                )
            else:
                self.player.current_track = 0 if self.player.playlist else -1
            self.player._invalidate_playlist_index()
            self.player._update_playlist_display()
            self.player._regenerate_shuffle_list_if_needed()
            self.update_statistics()
        if was_current:
            self.player.on_stop(None)
        self.player.save_playlist()

    def on_row_activated(self, column_view, position):
        item = self.selection_model.get_item(position)
        if item is None:
            return
        stored_string = item.get_string() if hasattr(item, "get_string") else str(item)
        track_id = self._resolve_track_id(stored_string)
        if track_id is None:
            return
        with self.player._playlist_lock:
            idx = self.player._get_playlist_index().get(track_id)
        if idx is not None:
            self.player.play_track(idx)

    def on_fix_tags(self, button):
        if not hasattr(self.player, "playlist") or not self.player.playlist:
            self.player.set_status_message("No tracks in playlist to fix")
            return
        if not METADATA_FIXER_AVAILABLE:
            self.player.set_status_message("Metadata fixer not available")
            return
        self._run_fix_tags()

    def _run_fix_tags(self):
        if getattr(self, "_fix_tags_in_progress", False):
            self.player.set_status_message("Tag fixing already in progress")
            return
        if not self.player.playlist:
            return
        file_paths = [
            item.path
            for item in self.player.playlist
            if hasattr(item, "path") and item.path and os.path.exists(item.path)
        ]
        if not file_paths:
            self.player.set_status_message("No valid files to fix")
            return
        self._fix_tags_in_progress = True
        self.player.set_status_message(
            f"Completing ID3 tags for {len(file_paths)} tracks..."
        )

        def fix_in_background():
            try:
                fixer = MetadataFixer()
                results = fixer.fix_files_threaded(file_paths)
            except Exception:
                logging.getLogger(__name__).exception("Tag fixing failed")
                GLib.idle_add(self._on_fix_tags_complete, 0, len(file_paths), 0, [])
                return
            updated = sum(1 for r in results if r.updated_fields)
            failed = sum(1 for r in results if not r.success)
            skipped = len(results) - updated - failed
            GLib.idle_add(self._on_fix_tags_complete, updated, failed, skipped, results)

        thread = threading.Thread(target=fix_in_background, daemon=True)
        thread.start()

    def _on_fix_tags_complete(
        self, updated: int, failed: int, skipped: int, results: list
    ):
        self._on_fix_tags_complete_impl(updated, failed, skipped, results)

    def _on_fix_tags_complete_impl(
        self, updated: int, failed: int, skipped: int, results: list
    ):
        self._fix_tags_in_progress = False
        if getattr(self.player, "_cleanup_in_progress", False) or getattr(
            self.player, "_cleanup_complete", False
        ):
            return False
        self.player.set_status_message(
            f"ID3 tags completed: {updated} updated, {failed} failed/skipped"
        )
        if hasattr(self.player, "_update_playlist_display"):
            self.player._update_playlist_display()
        current_path = None
        if self.player.current_track >= 0 and self.player.current_track < len(
            self.player.playlist
        ):
            current_item = self.player.playlist[self.player.current_track]
            if hasattr(current_item, "path"):
                current_path = current_item.path
        if current_path:
            for result in results:
                if (
                    result.success
                    and result.new_metadata
                    and result.file_path == current_path
                ):
                    new_title = result.new_metadata.get("title", "")
                    if new_title and hasattr(self.player, "player_tab"):
                        display = f"{new_title} ({Path(current_path).name})"
                        GLib.idle_add(
                            self.player.player_tab.track_label.set_text, display
                        )
                        GLib.idle_add(self.player.set_title, f"LinAmp - {new_title}")
                    break
