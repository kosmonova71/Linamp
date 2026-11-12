from __future__ import annotations

import gi

gi.require_version("Gdk", "4.0")
gi.require_version("Gtk", "4.0")
gi.require_version("Gst", "1.0")
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from gi.repository import Gdk, Gio, GLib, Gst, Gtk, Pango  # type: ignore[reportMissingModuleSource]

from ..core import (
    APP_ID,
    APP_NAME,
    CACHE_DIR,
    CONFIG_DIR,
    LOGGER,
    MILKDROP_AVAILABLE,
    PERIODIC_SAVE_INTERVAL,
    THEME_FILE,
    UPDATE_DISPLAY_INTERVAL,
    gi,
    prune_all_caches,
)
from ..models import (
    PlayerSettings,
    PlaylistItem,
)
from .tabs import (
    EqualizerTab,
    LyricsTab,
    PlayerTab,
    PlaylistTab,
    PodcastsTab,
    RadioTab,
    VisualizerTab,
)
from .window_parts.audio_sink import AudioSinkMixin
from .window_parts.display import DisplayMixin
from .window_parts.navigation import NavigationMixin
from .window_parts.playback import PlaybackMixin
from .window_parts.playlist import PlaylistMixin
from .window_parts.podcasts import PodcastsMixin
from .window_parts.radio import RadioMixin
from .window_parts.settings import SettingsMixin


class WinampWindow(  # type: ignore[misc]
    SettingsMixin,
    PlaylistMixin,
    PodcastsMixin,
    PlaybackMixin,
    RadioMixin,
    AudioSinkMixin,
    DisplayMixin,
    NavigationMixin,
    Gtk.ApplicationWindow,
):
    """Main application window: GTK4 UI + GStreamer playback + playlist/radio/podcasts.
    Playback invariants
    -------------------
    1. **Single authority.** ``_prepare_playback`` is the *only* method that
       sets the playbin URI and transitions to ``PLAYING``.  Every playback
       entry point (``play_file``, ``play_uri``, ``play_podcast_episode``,
       ``_attempt_recovery``, radio reconnect) must route through it.
    2. **Generation tokens.** ``self._playback_generation`` is an ever-incrementing
       counter.  It is bumped **before** any pipeline mutation that abandons
       the current track (``_handle_eos``, ``_advance_after_error``).  Late
       callbacks (verify timers, recovery timers, EOS handlers) capture the
       generation at scheduling time and become no-ops if it has changed,
       preventing stale callbacks from restarting an old URI.
    3. **Radio reconnect.** Uses a triple-guard: ``_radio_reconnect_token``
       (bumped on every invalidation), ``_playback_generation`` (bumped on
       track change/stop), and ``_radio_reconnect_pending`` (single-flight
       flag).  The reconnect is a single atomic ``STOP → NULL → URI →
       PLAYING`` sequence inside one timer callback, eliminating overlapping
       playback attempts.  After repeated CDN failures the original station
       URL is re-resolved via ``_resolve_radio_url_async``.
    4. **Audio sink lifecycle.** ``on_stop`` retains the audio sink
       (``release_audio_sink=False``) to preserve the visualizer/equalizer
       sink between tracks.  Sink lifecycle is coordinated through
       ``_ensure_audio_sink`` / ``_release_audio_sink``."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)
        self.set_title("LinAmp")
        self.add_css_class("linamp-window")
        self.header_bar = Gtk.HeaderBar()
        self.header_bar.add_css_class("app-header")
        self.set_titlebar(self.header_bar)
        title_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        title_box.add_css_class("title-box")
        title_label = Gtk.Label(label="LinAmp")
        title_label.add_css_class("app-title")
        title_label.set_halign(Gtk.Align.START)
        self.header_nowplaying = Gtk.Label(label="Ready")
        self.header_nowplaying.add_css_class("app-subtitle")
        self.header_nowplaying.set_halign(Gtk.Align.START)
        self.header_nowplaying.set_ellipsize(Pango.EllipsizeMode.END)
        title_box.append(title_label)
        title_box.append(self.header_nowplaying)
        self.header_bar.set_title_widget(title_box)
        search_btn = Gtk.Button()
        search_btn.set_tooltip_text("Search playlist")
        search_btn.add_css_class("icon-button")
        search_btn.set_child(Gtk.Image.new_from_icon_name("system-search-symbolic"))
        search_btn.connect("clicked", lambda *_: self.focus_playlist_search())
        self.header_bar.pack_end(search_btn)
        self.set_resizable(True)
        self._active_timers: set[int] = set()
        self._active_timers_lock = threading.Lock()
        self._playlist_lock = threading.RLock()
        self._sink_lock = threading.Lock()
        self._folder_scan_threads: set[threading.Thread] = set()
        self._folder_scan_lock = threading.Lock()
        self._folder_scan_generation = 0
        self._download_lock = threading.Lock()
        self.playlist: list[PlaylistItem] = []
        self._playlist_index: dict[str, int] = {}
        self.current_track = -1
        self.shuffled_indices: list[int] = []
        self.shuffle_position = 0
        self._is_muted = False
        self._previous_volume = 1.0
        self._is_radio_stream = False
        self._active_playback_is_remote = False
        self._radio_source_uri = ""
        self._radio_stream_uri = ""
        self._radio_stream_title = ""
        self._radio_start_time = 0.0
        self._radio_reconnect_pending = False
        self._radio_reconnect_token = 0
        self._radio_reconnect_failures = 0
        self._playback_generation = 0
        self._playback_verify_timer_id: int | None = None
        self._playback_verify_attempts = 0
        self._playback_recovery_attempts = 0
        self._playback_gave_up = False
        self._playback_deadline_reported = False
        self._user_stopped = False
        self._desired_play_state = Gst.State.PLAYING
        self._active_podcast_episode = None
        self._active_podcast_feed = None
        self._active_podcast_feed_id = ""
        self._active_podcast_episodes = None
        self._podcast_position_timer_id = 0
        self._podcast_position_generation = 0
        self._podcast_position_last_save = 0.0
        self._podcast_resume_attempted = False
        self._podcast_art_generation = 0
        self._podcast_sync_generation = 0
        self._bus_watch_id = None
        self._main_bus_signal_watch_attached = False
        self._main_bus_sync_handler_attached = False
        self._internal_buses_signal_watches_attached: list[int] = []
        self.is_compact_mode = False
        self.current_window_width = 650
        self.settings = PlayerSettings()
        self._cleanup_in_progress: bool = False
        self._cleanup_complete: bool = False
        self.player: Any = None
        self.equalizer: Any = None
        self._internal_buses: list[Any] = []
        self.playing: bool = False
        self._download_executor: ThreadPoolExecutor | None = None
        self._cover_executor: ThreadPoolExecutor | None = None
        self._persist_executor: ThreadPoolExecutor | None = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="Persist"
        )
        self._podcast_index_generation: int = 0
        self._fs_executor: ThreadPoolExecutor | None = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="FS"
        )
        self._idle: Any = GLib.idle_add
        self._auto_save_timer: int = 0
        self.load_settings()
        self.auto_play_next = self.settings.auto_play_next
        if self.settings.window_size[0] > 0 and self.settings.window_size[1] > 0:
            self.set_default_size(*self.settings.window_size)
        self.shuffle_mode = self.settings.shuffle_mode
        self.repeat_mode = self.settings.repeat_mode
        self.beat_aware_enabled = self.settings.beat_aware_enabled
        self.beat_threshold = self.settings.beat_threshold
        self.auto_download_covers = self.settings.auto_download_covers
        self.milkdrop_auto_switch = self.settings.milkdrop_auto_switch
        self.connect("close-request", self.on_close_request)
        self.connect("notify::default-width", self.on_window_size_changed)
        self.connect("notify::default-height", self.on_window_size_changed)
        self.connect("notify::width", self.on_window_size_changed)
        self.connect("notify::height", self.on_window_size_changed)
        self.setup_player()
        self.apply_xmms_css()
        self.main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.main_box.add_css_class("main-container")
        self.main_box.set_hexpand(True)
        self.main_box.set_vexpand(True)
        self.main_box.set_halign(Gtk.Align.FILL)
        self.main_box.set_valign(Gtk.Align.FILL)
        self.set_child(self.main_box)
        self.navigation_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.navigation_box.set_hexpand(True)
        self.navigation_box.set_vexpand(True)
        self.navigation_box.set_halign(Gtk.Align.FILL)
        self.navigation_box.set_valign(Gtk.Align.FILL)
        self.stack_switcher = Gtk.StackSwitcher()
        self.stack_switcher.set_halign(Gtk.Align.CENTER)
        self.stack_switcher.set_margin_bottom(6)
        self.stack = Gtk.Stack()
        self.stack.set_transition_type(Gtk.StackTransitionType.SLIDE_LEFT_RIGHT)
        self.stack.set_hexpand(True)
        self.stack.set_vexpand(True)
        self.stack.set_halign(Gtk.Align.FILL)
        self.stack.set_valign(Gtk.Align.FILL)
        self.navigation_box.append(self.stack_switcher)
        self.navigation_box.append(self.stack)
        self.main_box.append(self.navigation_box)
        self.player_tab = PlayerTab(self)
        self.equalizer_tab = EqualizerTab(self)
        self.playlist_tab = PlaylistTab(self)
        self.available_presets: list[str] = []
        self._preset_scan_started = False
        self.visualizer_tab = None
        self._visualizer_error: Exception | None = None
        if not MILKDROP_AVAILABLE:
            self._visualizer_error = ImportError(
                "MilkdropRenderer is not available — visualization disabled"
            )
            self.logger.warning(str(self._visualizer_error))
        else:
            try:
                self.visualizer_tab = VisualizerTab(self, [])
            except (ImportError, AttributeError, RuntimeError) as e:
                self._visualizer_error = e
                self.logger.exception("VisualizerTab creation failed")
        if self.visualizer_tab is not None:
            GLib.idle_add(self._start_preset_scan)
        for tab in (self.player_tab, self.equalizer_tab, self.playlist_tab):
            tab.set_hexpand(True)
            tab.set_vexpand(True)
        self.stack.add_titled(self.player_tab, "player", "Player")
        self.stack.add_titled(self.equalizer_tab, "equalizer", "Equalizer")
        self.stack.add_titled(self.playlist_tab, "playlist", "Playlist")
        if self.visualizer_tab is not None:
            self.stack.add_titled(self.visualizer_tab, "visualizer", "Visualizer")
        self.radio_tab = RadioTab(self)
        self.radio_tab.set_hexpand(True)
        self.radio_tab.set_vexpand(True)
        self.radio_tab.set_halign(Gtk.Align.FILL)
        self.radio_tab.set_valign(Gtk.Align.FILL)
        self.stack.add_titled(self.radio_tab, "radio", "Radio")
        self.podcasts_tab = PodcastsTab(self)
        self.podcasts_tab.set_hexpand(True)
        self.podcasts_tab.set_vexpand(True)
        self.podcasts_tab.set_halign(Gtk.Align.FILL)
        self.podcasts_tab.set_valign(Gtk.Align.FILL)
        self.stack.add_titled(self.podcasts_tab, "podcasts", "Podcasts")
        self.lyrics_tab = LyricsTab(self)
        self.lyrics_tab.set_hexpand(True)
        self.lyrics_tab.set_vexpand(True)
        self.lyrics_tab.set_halign(Gtk.Align.FILL)
        self.lyrics_tab.set_valign(Gtk.Align.FILL)
        self.stack.add_titled(self.lyrics_tab, "lyrics", "Lyrics")
        self.stack_switcher.set_stack(self.stack)
        self.stack.set_visible_child_name("player")

        def on_visible_child_changed(stack, _pspec):
            page = stack.get_visible_child()
            if page is self.visualizer_tab and hasattr(
                self.visualizer_tab, "_rendering_enabled"
            ):
                self._set_visualizer_activity(True)
            elif hasattr(self.visualizer_tab, "_rendering_enabled"):
                self._set_visualizer_activity(False)
            if page is self.lyrics_tab and 0 <= self.current_track < len(self.playlist):
                self.lyrics_tab.load_lyrics(self.playlist[self.current_track])

        self.stack.connect("notify::visible-child", on_visible_child_changed)
        GLib.idle_add(self.apply_responsive_layout)
        self.player_tab.play_btn.connect("clicked", self.on_play)
        self.player_tab.pause_btn.connect("clicked", self.on_pause)
        self.player_tab.stop_btn.connect("clicked", self.on_stop)
        self.player_tab.prev_btn.connect("clicked", self.on_prev)
        self.player_tab.next_btn.connect("clicked", self.on_next)
        self.player_tab.volume_scale.connect("value-changed", self.on_volume_changed)
        self.player_tab.repeat_btn.connect(
            "clicked", lambda b: self.toggle_repeat_mode()
        )
        self.player_tab.shuffle_btn.connect(
            "clicked", lambda b: self.toggle_shuffle_mode()
        )
        self.player_tab.beat_btn.connect("clicked", lambda b: self.toggle_beat_aware())
        self.player_tab.download_btn.connect(
            "clicked", lambda b: self.toggle_auto_download_covers()
        )
        self.player_tab.autonext_btn.connect(
            "clicked", lambda b: self.toggle_auto_play_next()
        )
        self.status_bar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        self.status_bar.set_size_request(-1, 28)
        self.status_bar.add_css_class("statusbar")
        self.status_label = Gtk.Label()
        self.status_label.set_halign(Gtk.Align.START)
        self.status_label.set_hexpand(True)
        self.status_label.set_margin_start(12)
        self.status_label.set_margin_end(12)
        self.status_bar.append(self.status_label)
        self.main_box.append(self.status_bar)
        self.setup_drag_drop()
        self._register_timer(500, self._apply_equalizer_settings_delayed)
        self._register_timer(1000, self._apply_ui_settings_delayed)
        self._register_timer(UPDATE_DISPLAY_INTERVAL, self.update_display)
        self._register_timer(PERIODIC_SAVE_INTERVAL, self.periodic_auto_save)
        try:
            self._register_timer(300000, prune_all_caches)
        except Exception:
            self.logger.exception("Failed to register cache prune timer")
        prune_all_caches()
        self.load_playlist()
        if (
            hasattr(self.settings, "last_played_track")
            and self.settings.last_played_track
        ):
            self._resume_last_played()

    def _start_preset_scan(self):
        if (
            getattr(self, "_preset_scan_started", False)
            or getattr(self, "_cleanup_in_progress", False)
            or not MILKDROP_AVAILABLE
        ):
            return False
        self._preset_scan_started = True

        def _scan():
            # The bundled MilkDrop preset collection may live either next to
            # this module (``linamp/ui/presets``, the installed-package layout)
            # or at the repository root (``presets/``, the editable/dev
            # layout — see pyproject [tool.pyest] pythonpath=["."]).  Prefer the
            # in-package directory when present and fall back to the repo root
            # so the scan never silently resolves to an empty (missing) path,
            # which leaves the visualizer with no preset and a static screen.
            module_dir = Path(__file__).parent
            candidate_dirs = [
                module_dir / "presets",  # installed package
                module_dir.parents[1] / "presets",  # repository root bundle
            ]
            presets_dir = next(
                (d for d in candidate_dirs if d.is_dir()), candidate_dirs[0]
            )
            try:
                presets = [
                    str(path) for path in presets_dir.rglob("*.milk") if path.is_file()
                ]
                error = None
            except OSError as e:
                presets = []
                error = e
            self.logger.debug(
                "MilkDrop preset scan: dir=%s found=%d error=%s",
                presets_dir,
                len(presets),
                error,
            )
            GLib.idle_add(self._apply_preset_scan, presets, error)

        threading.Thread(target=_scan, daemon=True, name="MilkDropPresetScan").start()
        return False

    def _apply_preset_scan(self, presets, error=None):
        if getattr(self, "_cleanup_in_progress", False) or getattr(
            self, "_cleanup_complete", False
        ):
            return False
        self.available_presets = presets
        visualizer_tab = getattr(self, "visualizer_tab", None)
        if error is not None:
            self.logger.error(
                "MilkDrop preset scan failed: %s",
                error,
                exc_info=(type(error), error, error.__traceback__),
            )
            if (
                visualizer_tab is not None
                and hasattr(visualizer_tab, "visualizer")
                and visualizer_tab.visualizer is not None
            ):
                label = Gtk.Label(
                    label=f"Preset scan failed: {type(error).__name__}: {error}"
                )
                label.add_css_class("placeholder-label")
                visualizer_tab.append(label)
            return False
        if visualizer_tab is not None and hasattr(
            visualizer_tab, "set_available_presets"
        ):
            visualizer_tab.set_available_presets(presets)
        return False

    def _register_timer(self, interval, callback):
        """Register a GLib timeout that stops on unexpected callback failures.
        Callbacks are expected to catch their own anticipated exceptions
        (network timeouts, transient parse errors, etc.). This wrapper
        guards the GTK main loop against truly unexpected failures,
        logs them at exception level for debugging, and stops the timer
        so programming errors do not become permanent error loops.
        """
        timer_id: list[int | None] = [None]

        def _wrapper():
            try:
                result = callback()
            except Exception:
                self.logger.exception(
                    "Timer callback failed: %s",
                    getattr(callback, "__name__", type(callback).__name__),
                )
                result = False
            if not result and timer_id[0] is not None:
                with self._active_timers_lock:
                    self._active_timers.discard(timer_id[0])
            return result

        timer_id[0] = GLib.timeout_add(interval, _wrapper)
        with self._active_timers_lock:
            self._active_timers.add(timer_id[0])
        return timer_id[0]

    def cleanup(self):
        """Deprecated alias for :meth:`full_cleanup`.
        This used to be a second, partial teardown that disconnected the bus
        and nulled the pipeline but never released the audio sink, the
        equalizer or the internal-bus watches, so it diverged silently from
        the real shutdown path (``do_shutdown`` -> ``full_cleanup``).  Two
        teardown implementations for one lifecycle is how the audio sink
        leaks; keep exactly one.
        """
        self._update_settings_from_state()
        self.save_settings()
        self.full_cleanup()

    def setup_drag_drop(self):
        self.dnd = Gtk.DropTarget.new(Gio.File, Gdk.DragAction.COPY)
        self.dnd.connect("drop", self.on_file_dropped)
        self.add_controller(self.dnd)

    def on_file_dropped(self, drop, file, x, y):
        path = file.get_path()
        if path:
            self.add_to_playlist([path])
            if not self.playing and self.playlist:
                self.play_track(0)
        return True


class LinAmpApp(Gtk.Application):
    def __init__(self) -> None:
        super().__init__(application_id=APP_ID)
        self.win: Any = None
        gtk_settings = Gtk.Settings.get_default()
        if gtk_settings is not None:
            # ``gtk-application-prefer-dark-theme`` is deprecated since GTK 4.4
            # and emits a runtime warning; prefer ``gtk-interface-color-scheme``
            # when it is available and fall back for older GTK releases.
            if gtk_settings.find_property("gtk-interface-color-scheme") is not None:
                gtk_settings.set_property("gtk-interface-color-scheme", "dark")
            else:
                gtk_settings.set_property(
                    "gtk-application-prefer-dark-theme", True
                )

    def do_activate(self):
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            for _sub in ("radio_logos", "podcast_art", "covers", "lyrics"):
                (CACHE_DIR / _sub).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            LOGGER.warning("Could not create LinAmp config/cache directories: %s", exc)
        if not self.win:
            self.win = WinampWindow(application=self, title=APP_NAME)
            self.win.present()
            self.win.set_visible(True)
            self.win.grab_focus()
            self.win._register_timer(500, self.apply_dark_theme_css)
        else:
            self.win.present()

    def apply_dark_theme_css(self):
        """Apply the bundled theme if it is available."""
        display = Gdk.Display.get_default()
        if display is None or not THEME_FILE.is_file():
            LOGGER.warning("GTK theme file not found: %s", THEME_FILE)
            return False
        css_provider = Gtk.CssProvider()
        try:
            css_provider.load_from_path(str(THEME_FILE))
        except GLib.Error as exc:
            LOGGER.warning("Failed to load GTK theme: %s", exc)
            return False
        Gtk.StyleContext.add_provider_for_display(
            display, css_provider, Gtk.STYLE_PROVIDER_PRIORITY_USER
        )
        return False

    def do_shutdown(self):
        if hasattr(self, "win") and self.win is not None:
            self.win.full_cleanup()
        Gtk.Application.do_shutdown(self)

    def do_startup(self):
        Gtk.Application.do_startup(self)
        self.setup_menu()
        self.set_accels_for_action("app.resize_to_cover_art", ["<Control>r"])
        self.set_accels_for_action("app.play_pause", ["space", "p"])
        self.set_accels_for_action("app.stop", ["s"])
        self.set_accels_for_action("app.prev_track", ["b", "z"])
        self.set_accels_for_action("app.next_track", ["n", "x"])
        # Volume shortcuts
        vol_up_action = Gio.SimpleAction.new("volume_up", None)
        vol_up_action.connect("activate", lambda *_: self.win.adjust_volume(5) if hasattr(self, "win") and self.win and hasattr(self.win, "adjust_volume") else None)
        vol_down_action = Gio.SimpleAction.new("volume_down", None)
        vol_down_action.connect("activate", lambda *_: self.win.adjust_volume(-5) if hasattr(self, "win") and self.win and hasattr(self.win, "adjust_volume") else None)
        self.add_action(vol_up_action)
        self.add_action(vol_down_action)
        self.set_accels_for_action("app.volume_up", ["<Control>Up", "<Control>plus"])
        self.set_accels_for_action("app.volume_down", ["<Control>Down", "<Control>minus"])
        if len(sys.argv) > 1:
            if not self.win:
                self.win = WinampWindow(application=self, title=APP_NAME)
            self.win.add_to_playlist(sys.argv[1:])
            if self.win.playlist:
                self.win.cleanup_invalid_tracks()
                if self.win.playlist:
                    self.win.play_track(0)

    def setup_menu(self):
        open_action = Gio.SimpleAction.new("open", None)
        open_action.connect("activate", self.on_open)
        open_folder_action = Gio.SimpleAction.new("open_folder", None)
        open_folder_action.connect("activate", self.on_open_folder)
        add_to_playlist_action = Gio.SimpleAction.new("add_to_playlist", None)
        add_to_playlist_action.connect("activate", self.on_add_to_playlist)
        add_folder_to_playlist_action = Gio.SimpleAction.new(
            "add_folder_to_playlist", None
        )
        add_folder_to_playlist_action.connect(
            "activate", self.on_add_folder_to_playlist
        )
        cleanup_action = Gio.SimpleAction.new("cleanup_invalid", None)
        cleanup_action.connect("activate", self.on_cleanup_invalid)
        resize_cover_action = Gio.SimpleAction.new("resize_to_cover_art", None)
        resize_cover_action.connect("activate", self.on_resize_to_cover_art)
        quit_action = Gio.SimpleAction.new("quit", None)
        quit_action.connect("activate", lambda *_: self.quit())
        play_action = Gio.SimpleAction.new("play_pause", None)
        play_action.connect("activate", lambda *_: self.win.on_play(None) if hasattr(self, "win") and self.win else None)
        next_action = Gio.SimpleAction.new("next_track", None)
        next_action.connect("activate", lambda *_: self.win.play_next_track() if hasattr(self, "win") and self.win else None)
        prev_action = Gio.SimpleAction.new("prev_track", None)
        prev_action.connect("activate", lambda *_: self.win.play_previous_track() if hasattr(self, "win") and self.win else None)
        stop_action = Gio.SimpleAction.new("stop", None)
        stop_action.connect("activate", lambda *_: self.win.on_stop(None) if hasattr(self, "win") and self.win else None)
        vol_up_action = Gio.SimpleAction.new("volume_up", None)
        vol_up_action.connect("activate", lambda *_: self.win.adjust_volume(5) if hasattr(self, "win") and self.win and hasattr(self.win, "adjust_volume") else None)
        vol_down_action = Gio.SimpleAction.new("volume_down", None)
        vol_down_action.connect("activate", lambda *_: self.win.adjust_volume(-5) if hasattr(self, "win") and self.win and hasattr(self.win, "adjust_volume") else None)
        self.add_action(open_action)
        self.add_action(open_folder_action)
        self.add_action(add_to_playlist_action)
        self.add_action(add_folder_to_playlist_action)
        self.add_action(cleanup_action)
        self.add_action(resize_cover_action)
        self.add_action(quit_action)
        self.add_action(play_action)
        self.add_action(next_action)
        self.add_action(prev_action)
        self.add_action(stop_action)
        self.add_action(vol_up_action)
        self.add_action(vol_down_action)
        # Playback controls
        play_action = Gio.SimpleAction.new("play_pause", None)
        play_action.connect("activate", lambda *_: self.win.on_play(None) if hasattr(self, "win") and self.win else None)
        next_action = Gio.SimpleAction.new("next_track", None)
        next_action.connect("activate", lambda *_: self.win.play_next_track() if hasattr(self, "win") and self.win else None)
        prev_action = Gio.SimpleAction.new("prev_track", None)
        prev_action.connect("activate", lambda *_: self.win.play_previous_track() if hasattr(self, "win") and self.win else None)
        stop_action = Gio.SimpleAction.new("stop", None)
        stop_action.connect("activate", lambda *_: self.win.on_stop(None) if hasattr(self, "win") and self.win else None)
        self.add_action(play_action)
        self.add_action(next_action)
        self.add_action(prev_action)
        self.add_action(stop_action)
        menu = Gio.Menu()
        file_menu = Gio.Menu()
        file_menu.append("Open File", "app.open")
        file_menu.append("Open Folder", "app.open_folder")
        file_menu.append("Add to Playlist", "app.add_to_playlist")
        file_menu.append("Add Folder to Playlist", "app.add_folder_to_playlist")
        file_menu.append("Cleanup Invalid Tracks", "app.cleanup_invalid")
        file_menu.append("Resize Window to Cover Art", "app.resize_to_cover_art")
        file_menu.append("Quit", "app.quit")
        menu.append_submenu("File", file_menu)
        playback_menu = Gio.Menu()
        playback_menu.append("Play/Pause", "app.play_pause")
        playback_menu.append("Stop", "app.stop")
        playback_menu.append("Previous", "app.prev_track")
        playback_menu.append("Next", "app.next_track")
        menu.append_submenu("Playback", playback_menu)
        playback_menu = Gio.Menu()
        playback_menu.append("Play/Pause", "app.play_pause")
        playback_menu.append("Stop", "app.stop")
        playback_menu.append("Previous", "app.prev_track")
        playback_menu.append("Next", "app.next_track")
        menu.append_submenu("Playback", playback_menu)
        self.set_menubar(menu)

    def on_open(self, action, param):
        dialog = Gtk.FileDialog()
        dialog.set_title("Open Audio File")
        dialog.set_modal(True)
        audio_filter = Gtk.FileFilter()
        audio_filter.set_name("Audio files")
        audio_filter.add_mime_type("audio/*")
        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(audio_filter)
        dialog.set_filters(filters)
        dialog.open(self.win, None, self._on_open_finish, None)

    def on_add_to_playlist(self, action, param):
        dialog = Gtk.FileDialog()
        dialog.set_title("Add to Playlist")
        dialog.set_modal(True)
        audio_filter = Gtk.FileFilter()
        audio_filter.set_name("Audio files")
        audio_filter.add_mime_type("audio/*")
        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(audio_filter)
        dialog.set_filters(filters)
        dialog.open_multiple(self.win, None, self._on_add_to_playlist_finish, None)

    def on_open_folder(self, action, param):
        dialog = Gtk.FileDialog()
        dialog.set_title("Open Music Folder")
        dialog.set_modal(True)
        dialog.select_folder(self.win, None, self._on_open_folder_finish, None)

    def on_add_folder_to_playlist(self, action, param):
        dialog = Gtk.FileDialog()
        dialog.set_title("Add Music Folder to Playlist")
        dialog.set_modal(True)
        dialog.select_folder(
            self.win, None, self._on_add_folder_to_playlist_finish, None
        )

    def _on_open_finish(self, dialog, result, user_data):
        try:
            file = dialog.open_finish(result)
        except GLib.Error:
            return
        if file:
            self.win.clear_playlist()
            self._handle_playlist_addition(file.get_path(), is_folder=False)
            if self.win.playlist:
                self.win.play_track(0)

    def _on_add_to_playlist_finish(self, dialog, result, user_data):
        was_empty = not self.win.playlist
        try:
            files = dialog.open_multiple_finish(result)
        except GLib.Error:
            files = None
        if files:
            for file in files:
                file_path = file.get_path()
                if file_path and os.path.isfile(file_path):
                    self.win.add_to_playlist([file_path])
            if was_empty and self.win.playlist:
                self.win.play_track(0)

    def _on_open_folder_finish(self, dialog, result, user_data):
        try:
            folder = dialog.select_folder_finish(result)
        except GLib.Error:
            return
        if folder:
            self.win.clear_playlist()
            self._handle_playlist_addition(folder.get_path(), is_folder=True)
            if self.win.playlist:
                self.win.play_track(0)

    def _on_add_folder_to_playlist_finish(self, dialog, result, user_data):
        try:
            folder = dialog.select_folder_finish(result)
        except GLib.Error:
            return
        if folder:
            self._handle_playlist_addition(folder.get_path(), is_folder=True)

    def _handle_playlist_addition(self, file_path, is_folder=False):
        if not file_path or not hasattr(self, "win") or not self.win:
            return False
        if is_folder:
            self.win.add_folder_to_playlist(file_path)
        else:
            self.win.add_to_playlist([file_path])
        return True

    def on_cleanup_invalid(self, action, param):
        if hasattr(self, "win") and self.win:
            return self.win.cleanup_invalid_tracks()

    def on_resize_to_cover_art(self, action, param):
        if hasattr(self.win, "cover_art_widget") and self.win.cover_art_widget:
            self.win.resize_to_cover_art()


def main() -> int:
    logging.basicConfig(
        level=os.environ.get("LINAMP_LOG_LEVEL", "WARNING").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app = LinAmpApp()
    return int(app.run(sys.argv) or 0)


if __name__ == "__main__":
    sys.exit(main())


def window_source() -> str:
    """Return the concatenated source of the window package.

    The window class is composed from the mixin modules in
    ``linamp/ui/window_parts``; source-inspection assertions therefore have
    to look at the window module *and* every mixin module.
    """
    package_dir = Path(__file__).resolve().parent
    parts = [Path(__file__).resolve()]
    parts.extend(sorted((package_dir / "window_parts").glob("*.py")))
    return "\n".join(part.read_text(encoding="utf-8") for part in parts)


__all__ = ["LinAmpApp", "WinampWindow", "main"]
