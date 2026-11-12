from __future__ import annotations

import logging
import time
from pathlib import Path

import gi

gi.require_version("Gdk", "4.0")
gi.require_version("Gtk", "4.0")
from gi.repository import Gdk, GLib, Gst, Gtk, Pango

from ...core import *
from ...core import requests
from ...media import *
from ...models import *
from ...podcasts import *
from ...radio import *
from .. import _create_labeled_icon_button, _create_modal_form


class RadioTab(Gtk.Box):
    def __init__(self, player) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self.player = player
        self.add_css_class("radio-tab")
        self.set_spacing(8)
        self.set_margin_top(8)
        self.set_margin_bottom(8)
        self.set_margin_start(8)
        self.set_margin_end(8)
        self._is_playing_radio = False
        self._radio_buffer_timer: int = 0
        self._radio_buffer_percent: int = 0
        self._live_timer_id: int = 0
        self._search_debounce_id: int | None = None
        self._stations = load_radio_stations()
        self._filtered_stations: list[dict[str, str]] = []
        self._stream_tag_update_id = None
        try:
            preload_radio_logos(self._stations)
        except OSError as e:
            logging.getLogger(__name__).debug("Radio logo preload failed: %s", e)
        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        header.set_halign(Gtk.Align.START)
        self.append(header)
        title_label = Gtk.Label(label="Internet Radio")
        title_label.add_css_class("section-header")
        title_label.set_halign(Gtk.Align.START)
        header.append(title_label)
        self.live_indicator = Gtk.Label(label="")
        self.live_indicator.add_css_class("live-indicator")
        self.live_indicator.set_halign(Gtk.Align.END)
        header.append(self.live_indicator)
        main_layout = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.append(main_layout)
        toolbar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        main_layout.append(toolbar)
        add_btn = _create_labeled_icon_button("Add Stream", "list-add-symbolic")
        add_btn.connect("clicked", self.on_add_stream)
        toolbar.append(add_btn)
        remove_btn = _create_labeled_icon_button("Remove", "list-remove-symbolic")
        remove_btn.connect("clicked", self.on_remove_station)
        toolbar.append(remove_btn)
        defaults_btn = _create_labeled_icon_button(
            "Load Defaults", "document-open-symbolic"
        )
        defaults_btn.connect("clicked", self.on_load_defaults)
        toolbar.append(defaults_btn)
        refresh_btn = _create_labeled_icon_button("Refresh", "view-refresh-symbolic")
        refresh_btn.connect("clicked", self.on_refresh)
        toolbar.append(refresh_btn)
        search_container = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        main_layout.append(search_container)
        self.search_entry = Gtk.Entry()
        self.search_entry.set_placeholder_text("Filter stations...")
        self.search_entry.set_hexpand(True)
        self.search_entry.add_css_class("search-entry")
        self.search_entry.connect("changed", self.on_search_changed)
        search_container.append(self.search_entry)
        self._genre_list = self._build_genre_list()
        self.genre_filter = Gtk.DropDown.new_from_strings(self._genre_list)
        self.genre_filter.add_css_class("sort-dropdown")
        self.genre_filter.set_selected(0)
        self.genre_filter.connect("notify::selected", self.on_genre_filter_changed)
        search_container.append(self.genre_filter)
        station_container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        station_container.set_margin_top(8)
        main_layout.append(station_container)
        station_header = Gtk.Label(label="Stations")
        station_header.add_css_class("section-header")
        station_header.set_halign(Gtk.Align.START)
        station_container.append(station_header)
        scrolled = Gtk.ScrolledWindow(
            hscrollbar_policy=Gtk.PolicyType.AUTOMATIC,
            vscrollbar_policy=Gtk.PolicyType.AUTOMATIC,
        )
        scrolled.add_css_class("radio-scrolled")
        scrolled.set_hexpand(True)
        scrolled.set_vexpand(True)
        station_container.append(scrolled)
        self.store = Gtk.StringList()
        self._populate_store()
        self.selection_model = Gtk.SingleSelection(model=self.store)
        self.column_view = Gtk.ColumnView(model=self.selection_model)
        self.column_view.add_css_class("radio-view")
        self.column_view.set_hexpand(True)
        self.column_view.set_vexpand(True)
        factory = Gtk.SignalListItemFactory()
        factory.connect("setup", self._on_station_factory_setup)
        factory.connect("bind", self._on_station_factory_bind)
        col = Gtk.ColumnViewColumn(title="Stations", factory=factory)
        self.column_view.append_column(col)
        self.column_view.connect("activate", self.on_station_activated)
        scrolled.set_child(self.column_view)
        self.selection_model.connect("selection-changed", self.on_selection_changed)

    def _build_genre_list(self) -> list[str]:
        """Build a sorted list of unique genres from the station collection."""
        genres = {"All Genres"}
        for station in self._stations:
            genre = station.get("genre", "").strip()
            if genre:
                genres.add(genre)
        sorted_genres = sorted(g for g in genres if g != "All Genres")
        return ["All Genres"] + sorted_genres

    def _refresh_genre_filter(self) -> None:
        """Rebuild the genre dropdown when the station list changes."""
        if not hasattr(self, "genre_filter"):
            return
        self._genre_list = self._build_genre_list()
        old_filter = self.genre_filter
        self.genre_filter = Gtk.DropDown.new_from_strings(self._genre_list)
        self.genre_filter.add_css_class("sort-dropdown")
        self.genre_filter.set_selected(0)
        self.genre_filter.connect("notify::selected", self.on_genre_filter_changed)
        search_container = self.search_entry.get_parent()
        if search_container and isinstance(search_container, Gtk.Box):
            if old_filter is not None and old_filter is not self.genre_filter:
                search_container.remove(old_filter)
            search_container.append(self.genre_filter)

    def _populate_store(self):
        self.store.splice(0, self.store.get_n_items())
        self._filtered_stations = []
        filter_text = (
            self.search_entry.get_text().lower()
            if hasattr(self, "search_entry")
            else ""
        )
        genre_idx = (
            self.genre_filter.get_selected() if hasattr(self, "genre_filter") else 0
        )
        genre = (
            self._genre_list[genre_idx]
            if genre_idx < len(self._genre_list)
            else "All Genres"
        )
        for station in self._stations:
            name = station.get("name", "")
            station_genre = station.get("genre", "")
            url = station.get("url", "")
            if (
                filter_text
                and filter_text not in name.lower()
                and filter_text not in url.lower()
            ):
                continue
            if genre != "All Genres" and station_genre.lower() != genre.lower():
                continue
            display = f"{name}  ·  {station_genre}" if station_genre else name
            self.store.append(display)
            self._filtered_stations.append(station)

    def _on_station_factory_setup(self, factory, list_item):
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        row.set_margin_top(4)
        row.set_margin_bottom(4)
        row.set_margin_start(8)
        row.set_margin_end(8)
        logo = Gtk.Image()
        logo.set_size_request(40, 40)
        logo.set_pixel_size(40)
        logo.add_css_class("radio-station-logo")
        logo.set_from_icon_name("audio-x-generic")
        row.append(logo)
        text_col = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        text_col.set_hexpand(True)
        text_col.set_valign(Gtk.Align.CENTER)
        title_label = Gtk.Label(label="")
        title_label.set_hexpand(True)
        title_label.set_halign(Gtk.Align.START)
        title_label.set_ellipsize(Pango.EllipsizeMode.END)
        title_label.add_css_class("radio-station-name")
        text_col.append(title_label)
        genre_label = Gtk.Label(label="")
        genre_label.add_css_class("radio-station-genre")
        genre_label.set_halign(Gtk.Align.START)
        genre_label.set_ellipsize(Pango.EllipsizeMode.END)
        text_col.append(genre_label)
        row.append(text_col)
        play_btn = Gtk.Button()
        play_btn.add_css_class("flat")
        play_btn.set_tooltip_text("Play station")
        play_btn.set_valign(Gtk.Align.CENTER)
        play_img = Gtk.Image.new_from_icon_name("media-playback-start-symbolic")
        play_btn.set_child(play_img)
        play_btn.connect("clicked", self._on_station_play_clicked, list_item)
        row.append(play_btn)
        list_item.set_child(row)

    def _on_station_factory_bind(self, factory, list_item):
        item = list_item.get_item()
        if not item:
            return
        position = list_item.get_position()
        station_data = None
        if (
            position is not None
            and position >= 0
            and position < len(self._filtered_stations)
        ):
            station_data = self._filtered_stations[position]
        if not station_data:
            return
        row = list_item.get_child()
        if not row or not isinstance(row, Gtk.Box):
            return
        logo = row.get_first_child()
        text_col = logo.get_next_sibling() if logo else None
        title_label = text_col.get_first_child() if text_col else None
        genre_label = title_label.get_next_sibling() if title_label else None
        if title_label and isinstance(title_label, Gtk.Label):
            title_label.set_text(station_data.get("name", ""))
        if genre_label and isinstance(genre_label, Gtk.Label):
            genre_label.set_text(station_data.get("genre", ""))
        if logo and isinstance(logo, Gtk.Image):
            logo_url = station_data.get("logo", "")
            self._apply_station_logo(logo, logo_url)

    def _apply_station_logo(self, image_widget: Gtk.Image, logo_url: str) -> None:
        """Load a station logo into a Gtk.Image (cached; async-safe via idle)."""
        if not logo_url:
            image_widget.set_from_icon_name("audio-x-generic")
            return
        expected_url = logo_url
        try:
            setattr(image_widget, "_expected_logo_url", expected_url)  # noqa: B010
        except AttributeError:
            pass

        def _set_from_path(path_str: str) -> bool:
            try:
                texture = Gdk.Texture.new_from_filename(path_str)
                image_widget.set_from_paintable(texture)
                image_widget.set_size_request(40, 40)
            except (GLib.Error, OSError):
                image_widget.set_from_icon_name("audio-x-generic")
            return False

        def _guarded_set_from_path(path_str: str) -> bool:
            current_expected = getattr(image_widget, "_expected_logo_url", None)
            if current_expected != expected_url:
                return False
            if getattr(self, "_cleanup_complete", False) or getattr(
                self, "_cleanup_in_progress", False
            ):
                return False
            _set_from_path(path_str)
            return False

        local = Path(logo_url).expanduser()
        if local.is_file() and local.is_absolute():
            try:
                real = local.resolve()
            except OSError:
                real = local
            blocked = False
            for private in (
                PODCASTS_CONFIG_DIR,
                PODCAST_EPISODES_DIR,
                COVER_CACHE_DIR,
                LYRICS_CACHE_DIR,
                RADIO_LOGO_CACHE_DIR,
            ):
                try:
                    if private != private.parent and real.is_relative_to(private):
                        blocked = True
                except (OSError, ValueError):
                    blocked = True
            if not blocked and real.suffix.lower() in _LOCAL_IMAGE_SUFFIXES:
                _set_from_path(str(real))
                return
        cache = _radio_logo_cache_path(logo_url)
        if cache.exists():
            _set_from_path(str(cache))
            return
        image_widget.set_from_icon_name("audio-x-generic")

        def _download():
            try:
                path = fetch_radio_logo(logo_url)
            except (requests.RequestException, OSError):
                logging.getLogger(__name__).debug(
                    "Radio logo fetch failed for %s", logo_url, exc_info=True
                )
                return
            if path and path.exists():
                GLib.idle_add(_guarded_set_from_path, str(path))

        executor = _get_download_executor()
        if executor is None:
            return
        try:
            executor.submit(_download)
        except RuntimeError:
            logging.getLogger(__name__).debug(
                "Download executor unavailable; skipping station logo"
            )

    def on_selection_changed(self, selection, position, n_items):
        pass

    def on_search_changed(self, entry):
        def apply_filter():
            self._populate_store()
            return False

        if self._search_debounce_id:
            self.player._safe_source_remove(self._search_debounce_id)
        self._search_debounce_id = self.player._register_timer(200, apply_filter)

    def on_genre_filter_changed(self, dropdown, pspec):
        self._populate_store()

    def on_station_activated(self, column_view, position):
        if position < 0 or position >= len(self._filtered_stations):
            return
        station = self._filtered_stations[position]
        self._play_station(station)

    def _on_station_play_clicked(self, button, list_item):
        position = list_item.get_position()
        if (
            position is not None
            and position >= 0
            and position < len(self._filtered_stations)
        ):
            self._play_station(self._filtered_stations[position])

    def _play_station(self, station):
        if not station or not station.get("url"):
            return
        url = station["url"]
        name = station.get("name", url)
        self._stop_radio_ui()
        if hasattr(self.player, "play_uri"):
            self.player.play_uri(url, name)
            self._is_playing_radio = True
            self._radio_start_time = time.time()
            self._start_live_indicator()
            self.player._register_timer(0, self._start_buffer_timer)
            logo_url = station.get("logo") or ""
            if logo_url:
                self.player._cover_generation = (
                    getattr(self.player, "_cover_generation", 0) + 1
                )
                logo_generation = self.player._cover_generation

                def _show_logo():
                    path = fetch_radio_logo(logo_url)

                    def _apply_logo():
                        if logo_generation != getattr(
                            self.player, "_cover_generation", 0
                        ):
                            return False
                        if getattr(self.player, "_cleanup_in_progress", False):
                            return False
                        if getattr(self.player, "_cleanup_complete", False):
                            return False
                        cover = getattr(
                            getattr(self.player, "player_tab", None),
                            "cover_art_widget",
                            None,
                        )
                        if path and cover and hasattr(cover, "load_cover_from_path"):
                            cover.load_cover_from_path(str(path))
                        return False

                    GLib.idle_add(_apply_logo)
                    return False

                executor = _get_download_executor()
                if executor is not None:
                    try:
                        executor.submit(_show_logo)
                    except RuntimeError:
                        logging.getLogger(__name__).debug(
                            "Station logo not scheduled: executor unavailable"
                        )

    def _stop_buffer_timer(self):
        """Remove the radio buffering monitor. Safe to call multiple times."""
        if self._radio_buffer_timer:
            self.player._safe_source_remove(self._radio_buffer_timer)
            self._radio_buffer_timer = 0

    def _start_buffer_timer(self):
        if self._radio_buffer_timer:
            return False
        self._radio_buffer_timer = self.player._register_timer(2000, self._check_buffer)
        return False

    def _check_buffer(self):
        if not self._is_playing_radio:
            self._radio_buffer_timer = 0
            return False
        buffering = False
        percent = 0
        gst_player = None
        try:
            gst_player = getattr(self.player, "player", None)
            if gst_player is not None:
                query = Gst.Query.new_buffering(Gst.Format.TIME)
                if gst_player.query(query):
                    _, percent = query.parse_buffering_percent()
                    self._radio_buffer_percent = percent
                    buffering = percent < 100
        except (GLib.Error, AttributeError, TypeError) as exc:
            logging.getLogger(__name__).debug("Radio buffering query failed: %s", exc)
        if buffering:
            return True
        if gst_player is not None:
            try:
                _change, state, _pending = gst_player.get_state(0)
            except (AttributeError, GLib.Error, TypeError) as exc:
                logging.getLogger(__name__).debug(
                    "Radio buffering state query failed: %s", exc
                )
                return True
            if state != Gst.State.PLAYING:
                self._radio_buffer_timer = 0
                return False
            return True
        self._radio_buffer_timer = 0
        return False

    def _start_live_indicator(self):
        if self._live_timer_id:
            return
        if not self._is_playing_radio or not hasattr(self, "_radio_start_time"):
            return
        self._live_timer_id = self.player._register_timer(
            1000, self._update_live_indicator
        )

    def _stop_radio_ui(self):
        if self._live_timer_id:
            self.player._safe_source_remove(self._live_timer_id)
            self._live_timer_id = 0
        self._stop_buffer_timer()
        self._is_playing_radio = False
        self.live_indicator.set_text("")

    def _update_live_indicator(self):
        if not self._is_playing_radio:
            self._live_timer_id = 0
            GLib.idle_add(self.live_indicator.set_text, "")
            return False
        elapsed = (
            int(time.time() - self._radio_start_time)
            if hasattr(self, "_radio_start_time")
            else 0
        )
        mins = elapsed // 60
        secs = elapsed % 60
        self.live_indicator.set_text(f"LIVE {mins}:{secs:02d}")
        return True

    def on_refresh(self, button):
        self._stations = load_radio_stations()
        self._populate_store()
        self.player.set_status_message("Radio stations refreshed")

    def on_load_defaults(self, button):
        self._stations = DEFAULT_RADIO_STATIONS.copy()
        save_radio_stations(self._stations)
        self._populate_store()
        self.player.set_status_message("Default stations loaded")

    def on_add_stream(self, button):
        root = self.get_root()
        dialog, entries, cancel_button, add_button = _create_modal_form(
            "Add Radio Stream",
            root if isinstance(root, Gtk.Window) else None,
            ["Station name", "Genre (optional)", "http:// or https:// stream URL"],
            "Add",
        )
        name_entry, genre_entry, url_entry = entries

        def close_dialog(*_args):
            dialog.close()

        def on_add(*_args):
            name = name_entry.get_text().strip()
            genre = genre_entry.get_text().strip()
            url = url_entry.get_text().strip()
            if not url or not _validate_url(url):
                self.player.set_status_message(
                    "Invalid URL — must start with http:// or https://"
                )
                url_entry.grab_focus()
                return
            if not name:
                name = url
            self._stations.append({"name": name, "genre": genre, "url": url})
            save_radio_stations(self._stations)
            self._refresh_genre_filter()
            self._populate_store()
            self.player.set_status_message(f"Added station: {name}")
            dialog.close()

        cancel_button.connect("clicked", close_dialog)
        add_button.connect("clicked", on_add)
        url_entry.connect("activate", lambda *_args: add_button.emit("clicked"))
        dialog.connect("close-request", lambda *_args: False)
        dialog.present()

    def on_remove_station(self, button):
        selected = self.selection_model.get_selected()
        if selected is None:
            self.player.set_status_message("No station selected")
            return
        if selected < 0 or selected >= len(self._filtered_stations):
            return
        station = self._filtered_stations[selected]
        self._stations.remove(station)
        save_radio_stations(self._stations)
        self._refresh_genre_filter()
        self._populate_store()
        name = station.get("name", "Unknown")
        self.player.set_status_message(f"Removed station: {name}")
