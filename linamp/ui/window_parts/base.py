"""Shared member declarations for the window mixin classes.

The main window is composed from focused mixin modules (see
``linamp/ui/window_parts``).  The concrete ``WinampWindow`` class owns the
runtime implementation of the members declared here, while the mixins reach
those members through this base class so that cross-module access resolves
statically.

Every method declared here is implemented by one of the mixins; the base
implementations only exist so that an incomplete refactor fails loudly
instead of silently doing nothing.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from ...models import PlayerSettings, PlaylistItem
from ...podcasts import PodcastEpisode

if TYPE_CHECKING:
    from gi.repository import Gtk  # type: ignore[reportMissingModuleSource]

    WindowBase = Gtk.ApplicationWindow
else:
    try:
        from gi.repository import Gtk

        class WindowBase(Gtk.ApplicationWindow):
            """Attribute and method contract shared by the window mixins."""
    except ImportError:

        class WindowBase:
            """Attribute and method contract shared by the window mixins."""
    """Attribute and method contract shared by the window mixins."""

    # --- Window state -----------------------------------------------------
    logger: logging.Logger
    settings: PlayerSettings
    player: Any
    equalizer: Any
    audio_sink: Any
    playing: bool

    # --- Playlist ---------------------------------------------------------
    current_track: int
    _playlist_lock: Any
    _playlist_index: dict[str, int]
    _folder_scan_lock: Any
    _folder_scan_threads: set[threading.Thread]
    _folder_scan_generation: int

    # --- Executors --------------------------------------------------------
    _sink_lock: Any
    _cover_executor: ThreadPoolExecutor | None
    _download_executor: ThreadPoolExecutor | None
    _persist_executor: ThreadPoolExecutor | None
    _fs_executor: ThreadPoolExecutor | None

    # --- Playback ---------------------------------------------------------
    _bus_watch_id: int | None
    _vis_caps_filter: Any
    _playback_verify_timer_id: int | None

    # --- Playback members owned by other mixins ---------------------------
    # Declared here so PlaybackMixin (and the other mixins) can reference
    # them statically without reportAttributeAccessIssue. The concrete
    # window class / owning mixin provides the runtime value.
    _active_timers: set[int]
    _active_timers_lock: threading.Lock
    _radio_stream_uri: str

    # --- UI tabs ----------------------------------------------------------
    # Set in ``WinampWindow.__init__``; declared here so every mixin can
    # reference them statically without ``reportAttributeAccessIssue``.
    # Tabs may be ``None`` when the underlying feature is unavailable.
    player_tab: Any
    equalizer_tab: Any
    playlist_tab: Any
    visualizer_tab: Any
    radio_tab: Any
    podcasts_tab: Any
    lyrics_tab: Any
    header_nowplaying: Any

    # --- Display state ----------------------------------------------------
    current_window_width: int
    is_compact_mode: bool
    _radio_start_time: float
    _radio_stream_title: str
    status_bar: Any
    status_label: Any
    main_box: Any
    stack: Any

    # --- Playback settings (mirrored from PlayerSettings) -----------------
    playlist: list[PlaylistItem]
    auto_play_next: bool
    shuffle_mode: bool
    repeat_mode: str
    beat_aware_enabled: bool
    beat_threshold: float
    auto_download_covers: bool
    milkdrop_auto_switch: bool

    # --- Methods implemented by the mixins --------------------------------
    def _capture_podcast_position(self) -> None:
        raise NotImplementedError

    def play_file(self, filepath, title=None) -> bool:
        raise NotImplementedError

    def _get_filtered_position(self, playlist_index: int) -> int | None:
        raise NotImplementedError

    def _load_cover_art_for_track(self, item: Any) -> None:
        raise NotImplementedError

    def update_status_display(self) -> None:
        raise NotImplementedError

    def auto_save_settings(self) -> None:
        raise NotImplementedError

    def set_status_message(self, message: str) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def on_stop(self, button: Any) -> None:
        raise NotImplementedError

    def save_settings(self) -> None:
        raise NotImplementedError

    def _update_settings_from_state(self) -> None:
        raise NotImplementedError

    def _register_timer(self, interval: int, callback: Any) -> Any:
        raise NotImplementedError

    def _safe_source_remove(self, timer_id: Any) -> bool:
        raise NotImplementedError

    def _set_playback_state(self, is_playing: bool) -> None:
        raise NotImplementedError

    def _prepare_playback(
        self,
        uri: str,
        title: str = "",
        is_radio: bool = False,
        podcast_episode: PodcastEpisode | None = None,
    ) -> bool:
        raise NotImplementedError

    def _invalidate_radio_reconnect(self) -> None:
        raise NotImplementedError

    def _schedule_radio_reconnect(self) -> None:
        raise NotImplementedError

    def _regenerate_shuffle_list_if_needed(self) -> None:
        raise NotImplementedError

    def _cancel_folder_scans(self) -> None:
        raise NotImplementedError

    def _flush_persist_futures(self, timeout: float) -> bool:
        raise NotImplementedError

    def _stop_podcast_position_tracking(self, persist: bool = True) -> None:
        raise NotImplementedError

    def _ensure_audio_sink(self) -> bool | None:
        raise NotImplementedError

    def _shutdown_download_workers(self) -> None:
        raise NotImplementedError

    def save_settings_on_track_change(self) -> None:
        raise NotImplementedError

    def _attempt_podcast_resume(self, episode: Any) -> None:
        raise NotImplementedError

    def set_title(self, title: str) -> None:
        try:
            from gi.repository import Gtk  # type: ignore[reportMissingModuleSource]
            return Gtk.ApplicationWindow.set_title(self, title)
        except Exception as e:
            self.logger.debug("set_title failed: %s", e, exc_info=True)
        if hasattr(super(), 'set_title'):
            return super().set_title(title)
        raise NotImplementedError

    def get_application(self) -> Any:
        try:
            from gi.repository import Gtk  # type: ignore[reportMissingModuleSource]
            return Gtk.ApplicationWindow.get_application(self)
        except Exception as e:
            self.logger.debug("get_application failed: %s", e, exc_info=True)
        if hasattr(super(), 'get_application'):
            return super().get_application()
        raise NotImplementedError

    def set_default_size(self, width: int, height: int) -> None:
        try:
            from gi.repository import Gtk  # type: ignore[reportMissingModuleSource]
            return Gtk.ApplicationWindow.set_default_size(self, width, height)
        except Exception as e:
            self.logger.debug("set_default_size failed: %s", e, exc_info=True)
        if hasattr(super(), 'set_default_size'):
            return super().set_default_size(width, height)
        raise NotImplementedError

    def get_width(self) -> int:
        try:
            from gi.repository import Gtk  # type: ignore[reportMissingModuleSource]
            return Gtk.ApplicationWindow.get_width(self)
        except Exception as e:
            self.logger.debug("get_width failed: %s", e, exc_info=True)
        if hasattr(super(), 'get_width'):
            return super().get_width()
        raise NotImplementedError

    def get_height(self) -> int:
        try:
            from gi.repository import Gtk  # type: ignore[reportMissingModuleSource]
            return Gtk.ApplicationWindow.get_height(self)
        except Exception as e:
            self.logger.debug("get_height failed: %s", e, exc_info=True)
        if hasattr(super(), 'get_height'):
            return super().get_height()
        raise NotImplementedError

    def get_default_size(self) -> tuple[int, int]:
        try:
            from gi.repository import Gtk  # type: ignore[reportMissingModuleSource]
            return Gtk.ApplicationWindow.get_default_size(self)
        except Exception as e:
            self.logger.debug("get_default_size failed: %s", e, exc_info=True)
        if hasattr(super(), 'get_default_size'):
            return super().get_default_size()
        raise NotImplementedError
