"""WinampWindow mixin: GStreamer pipeline setup, playback state and recovery."""

from __future__ import annotations

import gi

from .base import WindowBase

gi.require_version("Gdk", "4.0")
gi.require_version("Gtk", "4.0")
gi.require_version("Gst", "1.0")
import logging
import os
import time
from urllib.parse import urlparse

from gi.repository import GLib, Gst

from ...core import (
    DEFAULT_VOLUME,
    PLAYBACK_RECOVERY_MAX_ATTEMPTS,
    PLAYBACK_VERIFY_INTERVAL_MS,
    PLAYBACK_VERIFY_MAX_ATTEMPTS,
    _shutdown_download_executor,
    _validate_file_path,
    _validate_url,
)
from ...models import (
    PodcastEpisode,
)


class PlaybackMixin(WindowBase):
    """GStreamer pipeline setup, playback state and recovery.

    The methods are split out of ``WinampWindow`` so the window class
    stays readable; they keep operating on the window instance and
    share its playback, playlist and settings state.
    """

    def setup_player(self):
        self.player = Gst.ElementFactory.make("playbin", "player")
        if not self.player:
            GLib.idle_add(self.set_status_message, "GStreamer initialization failed")
            return
        self._set_playback_state(False)
        self.equalizer = None
        self.audio_sink = None
        self._audio_sample_rate = 44100.0
        self._audio_channels = 2
        self._vis_caps_filter = None
        self._ensure_audio_sink()
        volume = max(0.0, min(1.0, getattr(self.settings, "volume", DEFAULT_VOLUME)))
        self.player.set_property("volume", volume)
        self.player.connect("source-setup", self._on_source_setup)
        self._setup_player_bus()

    def _on_source_setup(self, playbin, source):
        """Configure source element for HTTP streams.

        For **radio** streams, disables automatic redirect following in
        souphttpsrc.  This prevents DNS-rebinding attacks where the source
        could follow redirects to private addresses.  Redirecting is then
        gated by ``_validate_radio_redirect`` in the bus sync handler, which
        validates each redirect destination before allowing it through.

        For **podcast** episodes, the initial audio URL was already
        pre-validated via ``_validate_url_dns`` before playback started, and
        podcast audio URLs commonly redirect through tracking CDNs (e.g.
        swap.fm → byspotify → podtrac → simplecast).  We therefore leave
        ``automatic-redirect`` enabled so souphttpsrc follows redirects
        automatically, and the bus sync handler validates each redirect
        message as a safety net without blocking the streaming pipeline.

        Disabling the property is the *only* thing that actually stops the
        redirect: dropping the element message in the bus sync handler removes
        it from the bus, but souphttpsrc has already been handed the new
        Location and connects to it regardless.  So a source element that does
        not accept the property is refused instead of being allowed to follow
        redirects unvalidated.
        """
        is_radio = getattr(self, "_is_radio_stream", False)
        is_podcast = getattr(self, "_active_podcast_episode", None) is not None
        if not is_radio and not is_podcast:
            return
        if not getattr(self, "_active_playback_is_remote", False):
            return
        # Podcast audio URLs were pre-validated and commonly redirect through
        # tracking CDNs; allow automatic redirect following so playback works.
        # Radio streams still need redirects disabled for SSRF protection.
        if is_podcast and not is_radio:
            return
        if not hasattr(source, "set_property"):
            self._refuse_unvalidated_redirects(source)
            return
        try:
            source.set_property("automatic-redirect", False)
        except (TypeError, GLib.Error) as exc:
            logging.getLogger(__name__).error(
                "Could not disable automatic-redirect on %s (%s)",
                type(source).__name__,
                exc,
            )
            self._refuse_unvalidated_redirects(source)
            return
        try:
            still_following = bool(source.get_property("automatic-redirect"))
        except (AttributeError, TypeError, GLib.Error):
            return
        if still_following:
            logging.getLogger(__name__).error(
                "Source %s still follows redirects after configuration",
                type(source).__name__,
            )
            self._refuse_unvalidated_redirects(source)

    def _refuse_unvalidated_redirects(self, source):
        """Stop playback: this source would follow redirects unvalidated."""
        logging.getLogger(__name__).error(
            "Refusing stream: %s cannot disable redirect following, so its "
            "redirects would bypass URL validation",
            type(source).__name__,
        )
        GLib.idle_add(
            self.set_status_message, "Refusing stream: source cannot disable redirects"
        )
        GLib.idle_add(self._stop_pipeline_for_unsafe_source)

    def _stop_pipeline_for_unsafe_source(self):
        player = getattr(self, "player", None)
        if player is None:
            self._set_playback_state(False)
            return False
        try:
            player.set_state(Gst.State.NULL)
        except (AttributeError, GLib.Error):
            pass
        self._set_playback_state(False)
        return False

    def _setup_player_bus(self):
        player = getattr(self, "player", None)
        if player is None:
            return
        bus = player.get_bus()
        if bus:
            if self._main_bus_signal_watch_attached:
                bus.remove_signal_watch()
                self._main_bus_signal_watch_attached = False
            if self._main_bus_sync_handler_attached:
                try:
                    bus.set_sync_handler(None)
                except (AttributeError, TypeError, GLib.Error):
                    pass
                self._main_bus_sync_handler_attached = False
            if hasattr(self, "_bus_watch_id") and self._bus_watch_id is not None:
                bus.disconnect(self._bus_watch_id)
                self._bus_watch_id = None
            bus.set_sync_handler(self._on_bus_sync_message, None)
            self._main_bus_sync_handler_attached = True
            bus.add_signal_watch()
            self._main_bus_signal_watch_attached = True
            self._bus_watch_id = bus.connect("message", self.on_bus_message)

    def cleanup_timers(self):
        if hasattr(self, "_active_timers"):
            with self._active_timers_lock:
                timer_ids = list(self._active_timers)
            for timer_id in timer_ids:
                self._safe_source_remove(timer_id)
            with self._active_timers_lock:
                self._active_timers.clear()

    def full_cleanup(self):
        """Tear down all resources: timers, executors, GStreamer pipeline, tabs.
        Idempotent and re-entrant: a guard flag prevents double-execution,
        and ``_playback_generation`` is bumped first so any in-flight
        recovery/reconnect callbacks become no-ops before resources are
        released.
        Explicitly drains:
          * Folder scan threads (``_cancel_folder_scans``)
          * Podcast position timer (``_stop_podcast_position_tracking``)
          * Radio reconnect / UI (``_invalidate_radio_reconnect`` + ``_stop_radio_ui``)
          * All registered GLib timers (``cleanup_timers`` / ``_active_timers``)
          * Visualizer tab (``visualizer_tab.cleanup()``)
          * Lyrics / podcasts tabs (``.cleanup()``)
          * Cover art display cache (``cover_art_widget.cleanup()``)
          * GStreamer pipeline → ``NULL`` then ``cleanup_pipeline``
          * Persist futures (``_flush_persist_futures``)
          * Download / persist / fs executors (``_shutdown_download_workers``)
          * Shared download executor (``_shutdown_download_executor``)
        """
        if getattr(self, "_cleanup_complete", False):
            return
        if getattr(self, "_cleanup_in_progress", False):
            return
        self._cleanup_in_progress = True
        try:
            self._run_full_cleanup()
        finally:
            self._cleanup_in_progress = False

    def _run_full_cleanup(self) -> None:
        self._playback_generation = getattr(self, "_playback_generation", 0) + 1
        try:
            self._update_settings_from_state()
            self.save_settings()
        except Exception as e:  # noqa: BLE001
            logging.getLogger(__name__).debug("Settings save on shutdown failed: %s", e)
        self._cancel_folder_scans()
        self._stop_podcast_position_tracking()
        self._invalidate_radio_reconnect()
        if hasattr(self, "radio_tab") and self.radio_tab:
            self.radio_tab._stop_radio_ui()
        self._set_playback_state(False)
        self.cleanup_timers()
        if (
            hasattr(self, "visualizer_tab")
            and self.visualizer_tab
            and hasattr(self.visualizer_tab, "cleanup")
        ):
            self.visualizer_tab.cleanup()
        if hasattr(self, "player") and self.player:
            try:
                self.player.set_state(Gst.State.NULL)
            except (AttributeError, GLib.Error):
                pass
            self.cleanup_pipeline(self.player, release_audio_sink=True)
            self.player = None
        if hasattr(self, "lyrics_tab") and self.lyrics_tab:
            self.lyrics_tab.cleanup()
        if hasattr(self, "podcasts_tab") and self.podcasts_tab:
            self.podcasts_tab.cleanup()
        if hasattr(self, "equalizer") and self.equalizer:
            self.equalizer.set_state(Gst.State.NULL)
            self.equalizer = None
        if hasattr(self, "player_tab") and hasattr(self.player_tab, "cover_art_widget"):
            self.player_tab.cover_art_widget.cleanup()
        if hasattr(self, "_cover_executor") and self._cover_executor:
            try:
                self._cover_executor.shutdown(wait=False, cancel_futures=True)
            except Exception as e:  # noqa: BLE001
                logging.getLogger(__name__).debug(
                    "Cover executor shutdown failed: %s", e
                )
            self._cover_executor = None
        self._flush_persist_futures(timeout=1.0)
        self._shutdown_download_workers()
        _shutdown_download_executor()
        self._cover_generation = getattr(self, "_cover_generation", 0) + 1
        self._podcast_sync_generation = getattr(self, "_podcast_sync_generation", 0) + 1
        self._clear_playback_verify_timer()
        self._cleanup_complete = True

    def on_close_request(self, window):
        app = self.get_application()
        if app is not None:
            app.quit()
        return True

    def _set_playback_state(self, is_playing: bool) -> None:
        self.playing = is_playing
        if hasattr(self, "player_tab") and hasattr(
            self.player_tab, "update_playback_state"
        ):
            self.player_tab.update_playback_state(is_playing)
        playlist_tab = getattr(self, "playlist_tab", None)
        if playlist_tab is not None and hasattr(
            playlist_tab, "refresh_playback_indicator"
        ):
            try:
                playlist_tab.refresh_playback_indicator()
            except Exception:
                logging.getLogger(__name__).debug(
                    "Failed to refresh playback indicator", exc_info=True
                )

    def _clear_playback_verify_timer(self) -> None:
        """Cancel the playback-verify watchdog and forget its id.
        The id is the only thing ``on_play`` uses to decide whether a verify
        cycle is already running, so it must be cleared on *every* exit path.
        Leaving it set after the source already stopped made ``on_play`` a
        silent no-op for the rest of the session.
        """
        timer_id = getattr(self, "_playback_verify_timer_id", None)
        if timer_id:
            self._safe_source_remove(timer_id)
        self._playback_verify_timer_id = None

    def _request_playback_state(self, state, generation) -> None:
        """Queue a state change that must not outlive *generation*.
        ``set_state(PLAYING)`` is deferred so it runs from the main loop
        rather than inside the verify timer, but the pipeline may be stopped,
        replaced or torn down before the idle runs.  Without these guards a
        queued idle can resurrect playback the user just stopped, or raise on
        ``self.player is None`` after ``full_cleanup``.
        """

        def _apply() -> bool:
            if generation != getattr(self, "_playback_generation", 0):
                return False
            if getattr(self, "_user_stopped", False):
                return False
            if (
                state == Gst.State.PLAYING
                and getattr(self, "_desired_play_state", Gst.State.PLAYING)
                != Gst.State.PLAYING
            ):
                return False
            player = getattr(self, "player", None)
            if player is None:
                return False
            try:
                player.set_state(state)
            except (AttributeError, GLib.Error, TypeError):
                pass
            return False

        GLib.idle_add(_apply)

    def _verify_playback_state(
        self, stream_id, title, generation, podcast_episode=None, _attempt=0
    ):
        if generation != getattr(self, "_playback_generation", 0):
            self._clear_playback_verify_timer()
            return False
        if not hasattr(self, "player") or not self.player:
            self._clear_playback_verify_timer()
            return False
        self._playback_verify_timer_id = None
        attempts = getattr(self, "_playback_verify_attempts", 0) + 1
        self._playback_verify_attempts = attempts
        if attempts > PLAYBACK_VERIFY_MAX_ATTEMPTS:
            self._clear_playback_verify_timer()
            if getattr(self, "_playback_deadline_reported", False):
                return False
            self._playback_deadline_reported = True
            logging.getLogger(__name__).warning(
                "Playback did not reach PLAYING within %d ms for %s; giving up",
                PLAYBACK_VERIFY_MAX_ATTEMPTS * PLAYBACK_VERIFY_INTERVAL_MS,
                stream_id or "unknown stream",
            )
            GLib.idle_add(self.set_status_message, "Playback failed to start")
            self._handle_playback_start_failure(stream_id, generation)
            return False
        try:
            change_return, current_state, _pending_state = self.player.get_state(0)
        except (AttributeError, GLib.Error, TypeError):
            self._clear_playback_verify_timer()
            return False
        if change_return == Gst.StateChangeReturn.FAILURE:
            self._clear_playback_verify_timer()
            GLib.idle_add(self._handle_playback_start_failure, stream_id, generation)
            return False
        if change_return in (
            Gst.StateChangeReturn.ASYNC,
            Gst.StateChangeReturn.NO_PREROLL,
        ):
            self._clear_playback_verify_timer()
            self._playback_verify_timer_id = self._register_timer(
                PLAYBACK_VERIFY_INTERVAL_MS,
                lambda: self._verify_playback_state(
                    stream_id, title, generation, podcast_episode
                ),
            )
            return False
        if current_state == Gst.State.PLAYING:
            self._clear_playback_verify_timer()
            self._playback_recovery_attempts = 0
            self._radio_reconnect_failures = 0
            self._set_playback_state(True)
            if hasattr(self, "save_settings_on_track_change"):
                self.save_settings_on_track_change()
            if podcast_episode is not None:
                self._attempt_podcast_resume(podcast_episode)
            is_radio = getattr(self, "_is_radio_stream", False)
            if is_radio and getattr(self, "_radio_stream_uri", None) == stream_id:
                track_name = self._radio_stream_title or title or "Radio Stream"
                if hasattr(self, "player_tab") and hasattr(
                    self.player_tab, "track_label"
                ):
                    GLib.idle_add(self.player_tab.track_label.set_text, track_name)
                GLib.idle_add(self.set_title, f"LinAmp - {track_name}")
            else:
                track_name = title or (
                    os.path.basename(stream_id) if stream_id else "Unknown Track"
                )
                if hasattr(self, "player_tab") and hasattr(
                    self.player_tab, "track_label"
                ):
                    GLib.idle_add(self.player_tab.track_label.set_text, track_name)
                GLib.idle_add(self.set_title, f"LinAmp - {track_name}")
            visualizer_tab = getattr(self, "visualizer_tab", None)
            visualizer_visible = (
                hasattr(self, "stack")
                and self.stack.get_visible_child() is visualizer_tab
            )
            if visualizer_tab is not None:
                self._set_visualizer_activity(visualizer_visible)
            return False
        if current_state == Gst.State.READY:
            if (
                getattr(self, "_desired_play_state", Gst.State.PLAYING)
                != Gst.State.PLAYING
            ):
                self._clear_playback_verify_timer()
                return False
            if not getattr(self, "_is_radio_stream", False):
                try:
                    if self.player.get_state(0)[1] in (
                        Gst.State.PAUSED,
                        Gst.State.PLAYING,
                    ):
                        self.player.seek_simple(
                            Gst.Format.TIME,
                            Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT,
                            0,
                        )
                except (AttributeError, GLib.Error, TypeError) as e:
                    logging.getLogger(__name__).debug(
                        "Playback restart seek failed: %s", e
                    )
            self._clear_playback_verify_timer()
            self._request_playback_state(Gst.State.PLAYING, generation)
            self._playback_verify_timer_id = self._register_timer(
                PLAYBACK_VERIFY_INTERVAL_MS,
                lambda: self._verify_playback_state(
                    stream_id, title, generation, podcast_episode
                ),
            )
        elif current_state == Gst.State.PAUSED:
            if (
                getattr(self, "_desired_play_state", Gst.State.PLAYING)
                != Gst.State.PLAYING
            ):
                self._clear_playback_verify_timer()
                return False
            self._clear_playback_verify_timer()
            self._request_playback_state(Gst.State.PLAYING, generation)
            self._playback_verify_timer_id = self._register_timer(
                PLAYBACK_VERIFY_INTERVAL_MS,
                lambda: self._verify_playback_state(
                    stream_id, title, generation, podcast_episode
                ),
            )
        else:
            self._clear_playback_verify_timer()
        return False

    def _handle_playback_start_failure(self, stream_id, generation):
        """Give up on a start attempt, retrying only a bounded number of times.
        ``_prepare_playback`` returns True whenever ``set_state(PLAYING)``
        reports ASYNC, which is exactly what a truncated or unsupported file
        does before failing during preroll.  The bus ERROR handler then called
        recovery at the *same* generation forever, so one bad track restarted
        in an endless loop and never advanced to the next one.
        """
        if getattr(self, "_user_stopped", False):
            return False
        if generation is not None and generation != getattr(
            self, "_playback_generation", 0
        ):
            return False
        if getattr(self, "_is_radio_stream", False):
            GLib.idle_add(self._schedule_radio_reconnect)
            return False
        if not stream_id:
            GLib.idle_add(self._advance_after_error)
            return False
        if getattr(self, "_playback_gave_up", False):
            return False
        attempts = getattr(self, "_playback_recovery_attempts", 0) + 1
        self._playback_recovery_attempts = attempts
        if attempts > PLAYBACK_RECOVERY_MAX_ATTEMPTS:
            self._playback_gave_up = True
            logging.getLogger(__name__).error(
                "Giving up on %s after %d recovery attempts",
                stream_id,
                attempts - 1,
            )
            GLib.idle_add(self.set_status_message, "Playback failed - skipping track")
            GLib.idle_add(self._advance_after_error)
            return False
        GLib.idle_add(self._attempt_recovery, stream_id, generation)
        return False

    def play_file(self, filepath, title=None):
        if not _validate_file_path(filepath):
            return False
        if not hasattr(self, "player") or not self.player:
            return False
        self._user_stopped = False
        self._desired_play_state = Gst.State.PLAYING
        self._capture_podcast_position()
        self._stop_podcast_position_tracking()
        self._is_radio_stream = False
        if (
            hasattr(self, "_playback_verify_timer_id")
            and self._playback_verify_timer_id
        ):
            self._safe_source_remove(self._playback_verify_timer_id)
            self._playback_verify_timer_id = None
        self._playback_generation = getattr(self, "_playback_generation", 0) + 1
        self._invalidate_radio_reconnect()
        self._ensure_audio_sink()
        uri = Gst.filename_to_uri(os.path.abspath(filepath))
        if uri is None:
            return False
        if not self._prepare_playback(
            uri, title or os.path.basename(filepath), is_radio=False
        ):
            return False
        if hasattr(self, "player_tab") and hasattr(self.player_tab, "track_label"):
            GLib.idle_add(
                self.player_tab.track_label.set_text,
                f"Loading: {os.path.basename(filepath)}...",
            )
        return True

    def play_uri(self, uri, title="Radio Stream"):
        if not uri:
            GLib.idle_add(self.set_status_message, "Invalid stream URL")
            return False
        if not hasattr(self, "player") or not self.player:
            return False
        self._user_stopped = False
        self._desired_play_state = Gst.State.PLAYING
        self._set_playback_state(False)
        self._ensure_audio_sink()
        self._stop_podcast_position_tracking()
        self._is_radio_stream = True
        self._radio_source_uri = uri
        self._radio_stream_title = title
        self._radio_start_time = time.time()
        if (
            hasattr(self, "_playback_verify_timer_id")
            and self._playback_verify_timer_id
        ):
            self._safe_source_remove(self._playback_verify_timer_id)
            self._playback_verify_timer_id = None
        self._playback_generation = getattr(self, "_playback_generation", 0) + 1
        self._invalidate_radio_reconnect()
        if hasattr(self, "player_tab") and hasattr(self.player_tab, "track_label"):
            GLib.idle_add(
                self.player_tab.track_label.set_text, f"Resolving: {title}..."
            )
        generation = getattr(self, "_playback_generation", 0)

        def on_resolved(resolved_uri):
            if (
                generation != getattr(self, "_playback_generation", 0)
                or getattr(self, "_user_stopped", False)
                or not getattr(self, "_is_radio_stream", False)
            ):
                return
            if not resolved_uri or not _validate_url(resolved_uri):
                GLib.idle_add(self.set_status_message, "Invalid or unsafe stream URL")
                self._is_radio_stream = False
                self._set_playback_state(False)
                return
            self._radio_stream_uri = resolved_uri
            if hasattr(self, "player_tab") and hasattr(self.player_tab, "track_label"):
                GLib.idle_add(
                    self.player_tab.track_label.set_text, f"Connecting: {title}..."
                )
            self._ensure_audio_sink()
            if not self._prepare_playback(resolved_uri, title, is_radio=True):
                return

        self._resolve_radio_url_async(uri, on_resolved)
        return True

    def on_pause(self, button):
        if (
            hasattr(self, "_playback_verify_timer_id")
            and self._playback_verify_timer_id
        ):
            self._safe_source_remove(self._playback_verify_timer_id)
            self._playback_verify_timer_id = None
        self._pause_podcast_position_tracking()
        self._desired_play_state = Gst.State.PAUSED
        self._set_visualizer_activity(False)
        if self.player:
            self.player.set_state(Gst.State.PAUSED)

    def on_play(self, button):
        """Toggle playback or resume a paused track."""
        self._desired_play_state = Gst.State.PLAYING
        if self.player:
            try:
                state = self.player.get_state(0)[1]
            except (AttributeError, GLib.Error, TypeError):
                state = Gst.State.NULL
            if state == Gst.State.PLAYING:
                self.on_pause(button)
                return
            if state == Gst.State.PAUSED:
                self._clear_playback_verify_timer()
                self._playback_verify_attempts = 0
                self.player.set_state(Gst.State.PLAYING)
                self._resume_podcast_position_tracking()
                generation = self._playback_generation
                current_uri = self.player.get_property("uri") if self.player else ""
                self._playback_verify_timer_id = self._register_timer(
                    PLAYBACK_VERIFY_INTERVAL_MS,
                    lambda: self._verify_playback_state(current_uri, "", generation),
                )
                visualizer_tab = getattr(self, "visualizer_tab", None)
                visualizer_visible = (
                    hasattr(self, "stack")
                    and self.stack.get_visible_child() is visualizer_tab
                )
                if visualizer_tab is not None:
                    self._set_visualizer_activity(visualizer_visible)
                return
        pending = getattr(self, "_playback_verify_timer_id", None)
        if pending:
            with self._active_timers_lock:
                live = pending in self._active_timers
            if live:
                return
            self._playback_verify_timer_id = None
        self._set_playback_state(False)
        if self.current_track >= 0 and self.current_track < len(self.playlist):
            self.play_track(self.current_track)
        elif self.playlist:
            self.play_track(0)
        else:
            GLib.idle_add(
                self.set_status_message, "No tracks in playlist - add files first"
            )

    def stop(self):
        self.on_stop(None)

    def on_stop(self, button):
        if hasattr(self, "_cleanup_in_progress") and self._cleanup_in_progress:
            return
        self._user_stopped = True
        self._playback_generation = getattr(self, "_playback_generation", 0) + 1
        self._clear_playback_verify_timer()
        self._capture_podcast_position()
        self._stop_podcast_position_tracking()
        self._invalidate_radio_reconnect()
        self._is_radio_stream = False
        if hasattr(self, "radio_tab") and self.radio_tab is not None:
            self.radio_tab._stop_radio_ui()
        if hasattr(self, "player") and self.player is not None:
            self.player.set_state(Gst.State.NULL)
            self.cleanup_pipeline(self.player, release_audio_sink=False)
        self._set_visualizer_activity(False)
        self._set_playback_state(False)
        if hasattr(self, "player_tab"):
            self.player_tab.time_label.set_text("0:00 / 0:00")
            self.player_tab.progress.set_value(0)
            self.player_tab.track_label.set_text("No track playing")
            if hasattr(self.player_tab, "cover_art_widget"):
                self.player_tab.cover_art_widget.clear()
        playlist_tab = getattr(self, "playlist_tab", None)
        if playlist_tab is not None and hasattr(
            playlist_tab, "refresh_playback_indicator"
        ):
            try:
                playlist_tab.refresh_playback_indicator()
            except Exception:
                logging.getLogger(__name__).debug(
                    "Failed to refresh playback indicator", exc_info=True
                )
        self.set_title("LinAmp - XMMS Style")
        self.save_settings_on_stop()

    def on_volume_changed(self, scale):
        if not hasattr(self, "player") or not self.player:
            return
        volume = scale.get_value() / 100.0
        if hasattr(self, "player_tab") and hasattr(self.player_tab, "volume_label"):
            self.player_tab.volume_label.set_label(f"{int(scale.get_value())}%")
        self.player.set_property("volume", volume)
        self.settings.volume = volume

    def adjust_volume(self, delta):
        if not hasattr(self, "player_tab") or not hasattr(self.player_tab, "volume_scale"):
            return
        scale = self.player_tab.volume_scale
        new_val = max(0, min(100, scale.get_value() + delta))
        scale.set_value(new_val)
        self.on_volume_changed(scale)
        volume = new_val / 100.0
        if hasattr(self, "_is_muted") and self._is_muted and volume > 0 and hasattr(self.player_tab, "_update_volume_button_icon"):
            self._is_muted = False
            self._previous_volume = volume
            self.player_tab._update_volume_button_icon()
        elif hasattr(self, "_is_muted") and not self._is_muted and hasattr(self.player_tab, "_update_volume_button_icon"):
            self.player_tab._update_volume_button_icon()
        self.auto_save_settings()

    def on_mute_toggle(self, button):
        if not hasattr(self, "player") or not self.player:
            return
        current_volume = self.player.get_property("volume")
        if hasattr(self, "_is_muted") and self._is_muted:
            self.player.set_property("volume", self._previous_volume)
            self._is_muted = False
        elif current_volume > 0:
            self._previous_volume = current_volume
            self.player.set_property("volume", 0.0)
            self._is_muted = True
        else:
            self._previous_volume = 1.0
            self.player.set_property("volume", 1.0)
            self._is_muted = False
        if hasattr(self, "player_tab") and hasattr(
            self.player_tab, "_update_volume_button_icon"
        ):
            self.player_tab._update_volume_button_icon()
        actual_volume = self.player.get_property("volume")
        if hasattr(self, "player_tab") and hasattr(self.player_tab, "volume_scale"):
            self.player_tab.volume_scale.set_value(actual_volume * 100)

    def cleanup_pipeline(self, player, release_audio_sink=False):
        if player is None:
            return
        bus = player.get_bus()
        if bus and release_audio_sink:
            if getattr(self, "_main_bus_signal_watch_attached", False):
                try:
                    bus.remove_signal_watch()
                except (TypeError, GLib.Error):
                    pass
                self._main_bus_signal_watch_attached = False
            if getattr(self, "_main_bus_sync_handler_attached", False):
                try:
                    bus.set_sync_handler(None)
                except (AttributeError, TypeError, GLib.Error):
                    pass
                self._main_bus_sync_handler_attached = False
            if getattr(self, "_bus_watch_id", None) is not None:
                try:
                    bus.disconnect(self._bus_watch_id)
                except (TypeError, GLib.Error):
                    pass
                self._bus_watch_id = None
        if release_audio_sink:
            self._release_audio_sink()

    def on_bus_message(self, bus, message):
        try:
            self._on_bus_message(bus, message)
        except Exception:
            logging.getLogger(__name__).exception("Bus message handler failed")

    def _on_bus_message(self, bus, message):
        if message.type == Gst.MessageType.ERROR:
            if getattr(self, "_user_stopped", False):
                return
            err, _debug = message.parse_error()
            if getattr(self, "_active_podcast_episode", None) is not None:
                generation = getattr(self, "_playback_generation", 0)
                GLib.idle_add(self._handle_podcast_error, err, generation)
                return
            if getattr(self, "_is_radio_stream", False):
                error_generation = getattr(self, "_playback_generation", 0)

                def _reconnect_if_current() -> bool:
                    if error_generation != getattr(self, "_playback_generation", 0):
                        return False
                    self._schedule_radio_reconnect()
                    return False

                GLib.idle_add(_reconnect_if_current)
                return
            error_msg = str(err.message).lower()
            current_uri = self.player.get_property("uri") if self.player else None
            generation = getattr(self, "_playback_generation", 0)

            def _stop_if_current() -> bool:
                if generation != getattr(self, "_playback_generation", 0):
                    return False
                if self.player:
                    try:
                        self.player.set_state(Gst.State.NULL)
                    except (AttributeError, GLib.Error):
                        pass
                return False

            GLib.idle_add(_stop_if_current)

            def _recover_once() -> bool:
                if generation != getattr(self, "_playback_generation", 0):
                    return False
                self._clear_playback_verify_timer()
                self._handle_playback_start_failure(current_uri, generation)
                return False

            self._register_timer(100, _recover_once)
            if "internal" in error_msg and "stream" in error_msg:
                GLib.idle_add(
                    self.set_status_message, "Internal stream error - recovering..."
                )
            else:
                GLib.idle_add(self.set_status_message, "Playback error - recovering...")
            return
        elif message.type == Gst.MessageType.EOS:
            if getattr(self, "_active_podcast_episode", None) is not None:
                generation = getattr(self, "_playback_generation", 0)
                GLib.idle_add(self._handle_podcast_eos, generation)
            elif getattr(self, "_is_radio_stream", False):
                generation = getattr(self, "_playback_generation", 0)

                def _reconnect_if_current() -> bool:
                    if generation != getattr(self, "_playback_generation", 0):
                        return False
                    self._schedule_radio_reconnect()
                    return False

                GLib.idle_add(_reconnect_if_current)
            else:
                eos_generation = getattr(self, "_playback_generation", 0)
                GLib.idle_add(self._handle_eos, eos_generation)
        elif message.type == Gst.MessageType.TAG and getattr(self, "_is_radio_stream", False):
                tag_list = message.parse_tag()
                if tag_list:
                    for key in ["title", "StreamTitle", "icy-name", "icy-url"]:
                        success, value = tag_list.get_string(key)
                        if success and value:
                            GLib.idle_add(self._on_radio_tag, key, value)
                            break
        return

    def _on_bus_sync_message(self, bus, message, user_data=None):
        try:
            return self._handle_bus_sync_message(bus, message, user_data)
        except Exception:
            logging.getLogger(__name__).exception("Bus sync handler failed")
            return Gst.BusSyncReply.PASS

    def _handle_bus_sync_message(self, bus, message, user_data=None):
        if message.type == Gst.MessageType.STATE_CHANGED and message.src == getattr(
            self, "player", None
        ):
            try:
                _old_state, new_state, _pending_state = message.parse_state_changed()
            except (AttributeError, GLib.Error, TypeError):
                return Gst.BusSyncReply.PASS
            generation = getattr(self, "_playback_generation", 0)

            def _apply_state(is_playing: bool) -> bool:
                if generation != getattr(self, "_playback_generation", 0):
                    return False
                self._set_playback_state(is_playing)
                return False

            GLib.idle_add(_apply_state, new_state == Gst.State.PLAYING)
            return Gst.BusSyncReply.PASS
        is_radio = getattr(self, "_is_radio_stream", False)
        is_podcast = getattr(self, "_active_podcast_episode", None) is not None
        if not is_radio and not is_podcast or message.type != Gst.MessageType.ELEMENT:
            return Gst.BusSyncReply.PASS
        structure = message.get_structure()
        if structure is None or structure.name != "redirect":
            return Gst.BusSyncReply.PASS
        if not self._validate_radio_redirect(message):
            generation = getattr(self, "_playback_generation", 0)
            logging.getLogger(__name__).warning(
                "HTTP redirect blocked by validation — stopping pipeline"
            )

            def _stop_if_current() -> bool:
                if generation != getattr(self, "_playback_generation", 0):
                    return False
                player = getattr(self, "player", None)
                if player is None:
                    return False
                player.set_state(Gst.State.NULL)
                return False

            GLib.idle_add(_stop_if_current)
            return Gst.BusSyncReply.DROP
        # Safe redirect: when automatic-redirect is disabled (radio streams),
        # souphttpsrc did not follow the redirect automatically.  Set the
        # validated new location on the source element so it can retry.
        source = message.src
        if source is not None and hasattr(source, "set_property"):
            try:
                num_entries = message.get_num_redirect_entries()
                for index in range(num_entries):
                    redirect_url, _tags, _entry = message.parse_redirect_entry(index)
                    if redirect_url and _validate_url(redirect_url):
                        source.set_property("location", redirect_url)
                        break
            except (AttributeError, GLib.Error, TypeError, ValueError):
                pass
        return Gst.BusSyncReply.PASS

    def _handle_eos(self, generation=None):
        if getattr(self, "_user_stopped", False):
            return False
        if generation is not None and generation != getattr(
            self, "_playback_generation", 0
        ):
            return False
        if getattr(self, "_desired_play_state", Gst.State.PLAYING) != Gst.State.PLAYING:
            logging.getLogger(__name__).debug(
                "EOS while paused - not advancing to the next track"
            )
            self._set_playback_state(False)
            return False
        player = getattr(self, "player", None)
        if player is None:
            self._set_playback_state(False)
            return False
        if getattr(self, "_is_radio_stream", False):
            self._set_playback_state(False)
            self._schedule_radio_reconnect()
            return False
        self._playback_generation = getattr(self, "_playback_generation", 0) + 1
        if getattr(self, "_active_podcast_episode", None) is not None:
            self._capture_podcast_position()
        ret = player.set_state(Gst.State.NULL)
        if ret == Gst.StateChangeReturn.FAILURE:
            logging.getLogger(__name__).warning("EOS: set_state(NULL) failed: %s", ret)
        self._set_playback_state(False)
        if self.auto_play_next:
            if self.repeat_mode == "one":
                self.play_track(self.current_track)
            elif self.repeat_mode == "all":
                self.play_next_track()
            else:
                next_index = self.get_next_track_index()
                if next_index is not None:
                    self.play_track(next_index)
                else:
                    self.on_stop(None)
        else:
            self.on_stop(None)

    @staticmethod
    def _query_playbin_state_summary(player) -> str:
        """Return a human-readable summary of the playbin's current state."""
        if player is None:
            return "unknown"
        try:
            change_return, current, pending = player.get_state(0)
        except (AttributeError, TypeError, GLib.Error):
            return "unknown"
        return (
            f"change_return={change_return.name.lower()} "
            f"current={current.name.lower()} "
            f"pending={pending.name.lower()}"
        )

    def _prepare_playback(
        self,
        uri: str,
        title: str = "",
        is_radio: bool = False,
        podcast_episode: PodcastEpisode | None = None,
    ) -> bool:
        """Single authority for URI/state transitions.
        Called by all playback entry points (play_file, play_uri,
        play_podcast_episode, _attempt_recovery, radio reconnect).
        Sequence:
            STOP/NULL → set URI → PLAYING → verify timer
        Every other callback requests playback via this method
        rather than directly manipulating the pipeline.
        """
        if not uri:
            return False
        self._user_stopped = False
        self._set_playback_state(False)
        player = getattr(self, "player", None)
        if player is None:
            return False
        try:
            self._active_playback_is_remote = urlparse(uri).scheme in ("http", "https")
        except (TypeError, ValueError):
            self._active_playback_is_remote = False
        try:
            player.set_state(Gst.State.NULL)
        except (AttributeError, GLib.Error):
            pass
        try:
            player.set_property("uri", uri)
            player.set_state(Gst.State.READY)
        except (AttributeError, GLib.Error):
            pass
        ret = player.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            self._set_playback_state(False)
            GLib.idle_add(self.set_status_message, "Failed to start playback")
            return False
        generation = self._playback_generation
        self._playback_verify_attempts = 0
        self._playback_recovery_attempts = 0
        self._playback_gave_up = False
        self._playback_deadline_reported = False
        self._clear_playback_verify_timer()
        self._playback_verify_timer_id = self._register_timer(
            PLAYBACK_VERIFY_INTERVAL_MS * 2,
            lambda: self._verify_playback_state(
                uri, title, generation, podcast_episode
            ),
        )
        if not is_radio:
            visualizer_tab = getattr(self, "visualizer_tab", None)
            visualizer_visible = (
                hasattr(self, "stack")
                and self.stack.get_visible_child() is visualizer_tab
            )
            if visualizer_tab is not None:
                self._set_visualizer_activity(visualizer_visible)
        return True

    def _attempt_recovery(self, uri, generation=None):
        if getattr(self, "_user_stopped", False):
            return False
        if generation is not None and generation != getattr(
            self, "_playback_generation", 0
        ):
            return False
        self._set_playback_state(False)
        if not uri:
            GLib.idle_add(
                self.set_status_message, "Recovery failed - trying next track"
            )
            self._advance_after_error()
            return False
        if generation is not None and generation != getattr(
            self, "_playback_generation", 0
        ):
            return False
        if getattr(self, "_user_stopped", False):
            return False
        if not getattr(self, "player", None):
            return False
        self._ensure_audio_sink()
        if not self._prepare_playback(uri, "", is_radio=False):
            self._advance_after_error()
        return False

    def _advance_after_error(self):
        self._playback_generation = getattr(self, "_playback_generation", 0) + 1
        if not self.auto_play_next:
            self.on_stop(None)
            return
        if not self.playlist:
            self.on_stop(None)
            return
        if self.current_track < 0:
            next_index = 0
        elif self.current_track < len(self.playlist) - 1:
            next_index = self.current_track + 1
        elif self.repeat_mode == "all":
            next_index = 0
        else:
            self.on_stop(None)
            return
        GLib.idle_add(lambda: (self.play_track(next_index), False)[1])

    def _seek_to_position(self, position):
        if hasattr(self, "player") and self.player and self.playing:
            seek_pos = int(position * Gst.SECOND)
            self.player.seek_simple(Gst.Format.TIME, Gst.SeekFlags.FLUSH, seek_pos)
        return False

    def _resume_last_played(self):
        if not self.settings.last_played_track:
            return
        track_index = None
        for i, item in enumerate(self.playlist):
            if item.path == self.settings.last_played_track:
                track_index = i
                break
        if (
            track_index is not None
            and self.play_track(track_index)
            and self.settings.last_played_position > 2.0
        ):
            self._register_timer(
                1000,
                lambda pos=self.settings.last_played_position: self._seek_to_position(
                    pos
                ),
            )

    def _set_visualizer_activity(self, enabled: bool) -> None:
        """Enable or disable the visualizer's periodic render/analysis timers.
        ``VisualizerTab._rendering_enabled`` is the single gate for both the
        16 ms render timer and the 33 ms audio-analysis timer.  The
        stack-switch handler and the play/pause/stop paths own that flag;
        routing every change through this helper is what keeps a stopped or
        paused pipeline from leaving a render loop running for the rest of
        the session.
        """
        tab = getattr(self, "visualizer_tab", None)
        if tab is None:
            return
        tab._rendering_enabled = bool(enabled)
        if enabled:
            tab._start_audio_consume_timer()
        else:
            tab._stop_audio_consume_timer()

    def _safe_source_remove(self, timer_id):
        if timer_id is None or timer_id == 0:
            return True
        with self._active_timers_lock:
            if timer_id not in self._active_timers:
                return False
            self._active_timers.discard(timer_id)
        try:
            GLib.source_remove(timer_id)
            return True
        except (TypeError, ValueError, GLib.Error):
            return False
