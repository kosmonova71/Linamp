"""WinampWindow mixin: Podcast feeds, episode playback and position persistence."""

from __future__ import annotations

import gi

from .base import WindowBase

gi.require_version("Gdk", "4.0")
gi.require_version("Gtk", "4.0")
gi.require_version("Gst", "1.0")
import copy
import logging
import os
import threading
import time
from concurrent.futures import wait
from typing import Any
from urllib.parse import urlparse

from gi.repository import GLib, Gst

from ...core import (
    _DeferredPersistEntry,
    _get_download_executor,
    _linamp_attr,
    _validate_file_path,
    _validate_url,
)
from ...core import (
    _validate_url_dns as _core_validate_url_dns,
)
from ...podcasts import (
    load_feed_episodes,
    save_feed_episodes,
    save_podcast_state,
)


class PodcastsMixin(WindowBase):
    """Podcast feeds, episode playback and position persistence.

    The methods are split out of ``WinampWindow`` so the window class
    stays readable; they keep operating on the window instance and
    share its playback, playlist and settings state.
    """

    def _podcast_episodes_for_active_feed(self):
        feed_id = getattr(self, "_active_podcast_feed_id", "")
        podcasts_tab = getattr(self, "podcasts_tab", None)
        if feed_id and podcasts_tab is not None:
            selected_feed = getattr(podcasts_tab, "selected_feed", None)
            if (
                getattr(selected_feed, "feed_id", "") == feed_id
                and getattr(podcasts_tab, "episodes", None) is not None
            ):
                return podcasts_tab.episodes
            for candidate in getattr(podcasts_tab, "feeds", []):
                if getattr(candidate, "feed_id", "") == feed_id:
                    return getattr(candidate, "episodes", [])
        feed = getattr(self, "_active_podcast_feed", None)
        if feed is not None and getattr(feed, "feed_id", "") == feed_id:
            return getattr(feed, "episodes", [])
        if feed_id:
            return load_feed_episodes(feed_id)
        return None

    def _persist_podcast_position(self):
        snapshot = self._capture_podcast_snapshot()
        if snapshot is None:
            return
        feed_id, episodes, feeds = snapshot
        persist = getattr(self, "_save_podcast_snapshot", None)
        if persist is not None:
            try:
                persist(feed_id, episodes, feeds)
                self._podcast_position_last_save = time.time()
                return
            except Exception:  # noqa: BLE001
                logging.getLogger(__name__).debug(
                    "Unified podcast position persist failed; falling back"
                )
        save_feed_episodes(feed_id, episodes)
        self._podcast_position_last_save = time.time()

    @staticmethod
    def _log_download_failure(context: str, url: str, error: Exception) -> None:
        """Log a podcast/radio download failure (called from worker threads)."""
        logging.getLogger(__name__).warning(
            "Download failed [%s] %s: %s", context, url, error
        )

    def _query_current_position(self) -> int:
        """Return current playback position in GStreamer nanoseconds."""
        player = getattr(self, "player", None)
        if player is None:
            return 0
        try:
            success, position = player.query_position(Gst.Format.TIME)
            if not success:
                return 0
            return int(position)
        except (AttributeError, TypeError, ValueError, GLib.Error):
            return 0

    def _submit_persist(self, worker, key: Any = None) -> Any:
        """Submit a persist worker with optional key-based coalescing.
        When ``_persist_coalesced_pending`` is True, workers are staged in
        ``_pending_persist_workers`` keyed by *key* (superseding any prior
        worker for the same key) without execution.  When False, staged
        workers are flushed first, then the new worker is submitted directly.
        """
        if getattr(self, "_persist_coalesced_pending", False):
            lock = getattr(self, "_download_lock", None)
            if lock is None:
                lock = threading.Lock()
            with lock:
                if not hasattr(self, "_pending_persist_workers"):
                    self._pending_persist_workers = {}
                self._pending_persist_workers[key] = worker
            return None
        pending = getattr(self, "_pending_persist_workers", None)
        if pending:
            self._run_coalesced_persist()
        return self._do_persist_submit(worker)

    def _do_persist_submit(self, worker) -> Any:
        """Submit *worker* to the persist executor and track the future."""
        executor = getattr(self, "_persist_executor", None)
        if executor is None:
            return None
        revision = getattr(self, "_persist_revision", 0)
        future = executor.submit(self._persist_worker, worker, revision)
        lock = getattr(self, "_download_lock", None)

        def _on_done(fut):
            futures = getattr(self, "_persist_futures", None)
            if futures is None:
                return
            if lock is not None:
                with lock:
                    futures.discard(fut)
            else:
                futures.discard(fut)

        if lock is not None:
            with lock:
                if not hasattr(self, "_persist_futures"):
                    self._persist_futures = set()
                self._persist_futures.add(future)
        else:
            if not hasattr(self, "_persist_futures"):
                self._persist_futures = set()
            self._persist_futures.add(future)
        future.add_done_callback(_on_done)
        return future

    def _run_coalesced_persist(self) -> None:
        """Flush all staged persist workers to the executor."""
        lock = getattr(self, "_download_lock", None)
        if lock is None:
            lock = threading.Lock()
        with lock:
            self._persist_coalesced_pending = False
            if not hasattr(self, "_pending_persist_workers"):
                return
            pending = dict(self._pending_persist_workers)
            self._pending_persist_workers.clear()
        for worker in pending.values():
            self._do_persist_submit(worker)

    def _persist_worker(self, worker, revision: int) -> Any:
        """Run a persist worker, returning its result.
        The *revision* parameter is a compatibility shim for callers that
        pass a generation guard; supersession is already enforced by the
        key-coalescing in ``_submit_persist`` so only the latest worker per
        key is ever submitted.
        """
        return worker()

    def _flush_persist_futures(self, timeout: float) -> bool:
        """Drain all persist futures within *timeout* seconds.
        Returns True if the queue drained completely, False on timeout.
        """
        lock = getattr(self, "_download_lock", None)
        if lock is None:
            lock = threading.Lock()
        deadline = time.monotonic() + timeout
        while True:
            with lock:
                futures = list(getattr(self, "_persist_futures", set()))
                pending = bool(getattr(self, "_pending_persist_workers", None))
                if not futures and not pending:
                    self._persist_flush_complete = True
                    return True
            if pending or getattr(self, "_persist_coalesced_pending", False):
                self._run_coalesced_persist()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._persist_flush_complete = False
                return False
            wait_time = min(0.1, remaining)
            if futures:
                wait(futures, timeout=wait_time)

    def _podcast_feeds(self) -> list:
        """Return the live podcast feed list owned by the podcasts tab.
        The feed list lives on ``PodcastsTab``; ``WinampWindow`` itself has no
        ``feeds`` attribute.  Reading ``self.feeds`` here silently yields ``[]``
        and would rewrite ``podcasts.json`` as an empty index, wiping every
        subscription, so the lookup must go through the tab.
        """
        tab = getattr(self, "podcasts_tab", None)
        feeds = getattr(tab, "feeds", None)
        if isinstance(feeds, list):
            return feeds
        return []

    def _capture_podcast_snapshot(self) -> tuple[str, list, list] | None:
        """Capture current podcast playback state for persistence.
        Returns ``(feed_id, episodes_copy, feeds_deepcopy)`` or ``None`` when
        no podcast is active.
        """
        if not getattr(self, "player", None):
            return None
        context = getattr(self, "_podcast_context", None)
        if context:
            episode = context[1] if len(context) > 1 else None
            feed = context[2] if len(context) > 2 else None
        else:
            episode = getattr(self, "_active_podcast_episode", None)
            feed = getattr(self, "_active_podcast_feed", None)
        if feed is None:
            return None
        feed_id = getattr(feed, "feed_id", "") or getattr(
            self, "_active_podcast_feed_id", ""
        )
        position_ns = getattr(self, "_query_current_position", lambda: 0)()
        if position_ns and episode is not None:
            position_seconds = max(0, int(position_ns / 1_000_000_000))
            episode.position = position_seconds
            if (episode.duration > 0 and position_seconds >= episode.duration - 5) or (
                episode.duration <= 0 and position_seconds >= 30
            ):
                episode.played = True
        episodes = getattr(self, "_active_podcast_episodes", None)
        if episodes is None:
            episodes = getattr(feed, "episodes", [])
        if not episodes and episode is not None:
            episodes = [episode]
        episodes_copy = [copy.copy(ep) for ep in episodes]
        feeds = self._podcast_feeds()
        feeds_deepcopy = copy.deepcopy(feeds)
        return (feed_id, episodes_copy, feeds_deepcopy)

    def _snapshot_feeds_for_persist(self) -> list:
        """Return a deep copy of the current feeds list for persistence."""
        return copy.deepcopy(self._podcast_feeds())

    def _save_podcast_snapshot(
        self, fid: str, episodes, feeds, generation: int | None = None
    ) -> None:
        """Submit a unified persist of episode data + feed index.
        The snapshot is a deep copy taken on the UI thread, so it can go stale
        before the single-worker persist queue drains it -- for example when a
        feed is removed in between.  The captured *generation* is re-checked
        immediately before the write so a superseded snapshot can never
        resurrect a removed feed or drop a newly added one.
        """
        gen = (
            getattr(self, "_podcast_index_generation", 0)
            if generation is None
            else generation
        )

        def _worker() -> None:
            if gen != getattr(self, "_podcast_index_generation", 0):
                return
            if not save_podcast_state(feeds, fid, episodes):
                logging.getLogger(__name__).error(
                    "Podcast state persist failed for feed %s", fid
                )

        self._submit_persist(_worker, key=f"podcast_episodes_{fid}")

    def _invalidate_podcast_persists(self, feed_id: str = "") -> None:
        """Invalidate staged/queued podcast index persists for *feed_id*.
        Called when the feed index is mutated outside the position-persist
        path (feed add/remove) so queued snapshots stop short of the disk.
        """
        self._podcast_index_generation = (
            getattr(self, "_podcast_index_generation", 0) + 1
        )
        pending = getattr(self, "_pending_persist_workers", None)
        if not pending:
            return
        lock = getattr(self, "_download_lock", None) or threading.Lock()
        with lock:
            for key in [
                k
                for k in pending
                if isinstance(k, str)
                and k.startswith("podcast_episodes_")
                and (not feed_id or k == f"podcast_episodes_{feed_id}")
            ]:
                pending.pop(key, None)

    def _idle_run_deferred(self, entry) -> bool:
        """GLib idle callback wrapper for deferred persist."""
        self._run_deferred_persist(entry)
        return False

    def _run_deferred_persist(self, entry) -> None:
        """Run a deferred persist entry, unregistering it first."""
        self._unregister_deferred_persist(entry.callback, entry.snapshot)
        try:
            entry.callback(entry.snapshot)
        except Exception:
            logging.getLogger(__name__).debug(
                "Deferred persist callback failed", exc_info=True
            )

    def _schedule_deferred_persist(self, callback, snapshot) -> Any:
        """Schedule a deferred persist callback to run on idle."""
        entry = _DeferredPersistEntry(callback, snapshot)
        pending = getattr(self, "_pending_deferred_persist", None)
        if pending is None:
            pending = []
            self._pending_deferred_persist = pending
        pending.append(entry)
        idle_add = getattr(self, "_idle", None)
        if idle_add is not None:
            idle_cb = getattr(self, "_idle_run_deferred", None)
            if idle_cb is not None:
                idle_add(idle_cb, entry)
        return entry

    def _unregister_deferred_persist(self, callback, snapshot) -> None:
        """Remove a matching deferred persist entry from the registry."""
        pending = getattr(self, "_pending_deferred_persist", None)
        if not pending:
            return
        for entry in list(pending):
            if entry.callback == callback and entry.snapshot == snapshot:
                pending.remove(entry)

    def _drain_deferred_persist(self) -> None:
        """Run all pending deferred persist callbacks synchronously."""
        pending = getattr(self, "_pending_deferred_persist", None)
        if not pending:
            return
        for entry in list(pending):
            self._run_deferred_persist(entry)

    def _shutdown_download_workers(self) -> None:
        """Shut down download, fs, and persist executors."""
        flush_complete = getattr(self, "_persist_flush_complete", False)
        executor = getattr(self, "_persist_executor", None)
        if executor is not None:
            if flush_complete:
                try:
                    executor.shutdown(wait=True)
                except Exception as e:  # noqa: BLE001
                    self.logger.debug(
                        "Error shutting down persist executor (flush): %s", e
                    )
            else:
                try:
                    executor.shutdown(wait=False, cancel_futures=True)
                except TypeError:
                    executor.shutdown(wait=False)
                except Exception as e:  # noqa: BLE001
                    self.logger.debug("Error shutting down persist executor: %s", e)
        self._persist_executor = None
        download_exec = getattr(self, "_download_executor", None)
        if download_exec is not None:
            try:
                download_exec.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                download_exec.shutdown(wait=False)
            except Exception as e:  # noqa: BLE001
                self.logger.debug("Error shutting down download executor: %s", e)
        self._download_executor = None
        fs_exec = getattr(self, "_fs_executor", None)
        if fs_exec is not None:
            try:
                fs_exec.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                fs_exec.shutdown(wait=False)
            except Exception as e:  # noqa: BLE001
                self.logger.debug("Error shutting down fs executor: %s", e)
        self._fs_executor = None

    def _stop_podcast_position_tracking(self, persist=True):
        timer_id = getattr(self, "_podcast_position_timer_id", 0)
        if timer_id:
            self._safe_source_remove(timer_id)
            self._podcast_position_timer_id = 0
        if persist:
            self._persist_podcast_position()
        self._active_podcast_episode = None
        self._active_podcast_feed = None
        self._active_podcast_feed_id = ""
        self._active_podcast_episodes = None
        self._podcast_position_generation = 0
        self._podcast_resume_attempted = False

    def _pause_podcast_position_tracking(self):
        timer_id = getattr(self, "_podcast_position_timer_id", 0)
        if timer_id:
            self._safe_source_remove(timer_id)
            self._podcast_position_timer_id = 0

    def _resume_podcast_position_tracking(self):
        episode = getattr(self, "_active_podcast_episode", None)
        if episode is None:
            return
        if getattr(self, "_podcast_position_timer_id", 0):
            return
        self._podcast_position_generation = getattr(self, "_playback_generation", 0)
        self._podcast_position_last_save = 0.0
        self._podcast_position_timer_id = self._register_timer(
            1000, self._update_podcast_position
        )

    def _start_podcast_position_tracking(self):
        self._stop_podcast_position_tracking(persist=False)
        self._podcast_position_generation = getattr(self, "_playback_generation", 0)
        self._podcast_position_last_save = 0.0
        self._podcast_resume_attempted = False
        self._podcast_position_timer_id = self._register_timer(
            1000, self._update_podcast_position
        )

    def _update_podcast_position(self):
        episode = getattr(self, "_active_podcast_episode", None)
        if episode is None:
            self._podcast_position_timer_id = 0
            return False
        if getattr(self, "_podcast_position_generation", 0) != getattr(
            self, "_playback_generation", 0
        ):
            self._podcast_position_timer_id = 0
            return False
        player = getattr(self, "player", None)
        if not player:
            self._podcast_position_timer_id = 0
            return False
        try:
            _change_return, current_state, _pending_state = player.get_state(0)
        except (AttributeError, GLib.Error, TypeError):
            self._podcast_position_timer_id = 0
            return False
        if _change_return == Gst.StateChangeReturn.FAILURE:
            self._podcast_position_timer_id = 0
            return False
        if current_state != Gst.State.PLAYING:
            self._podcast_position_timer_id = 0
            return False
        try:
            success, position = player.query_position(Gst.Format.TIME)
            if not success:
                return True
            position_seconds = max(0, int(position / Gst.SECOND))
            episode.position = position_seconds
            if (episode.duration > 0 and position_seconds >= episode.duration - 5) or (
                episode.duration <= 0 and position_seconds >= 30
            ):
                episode.played = True
            now = time.time()
            if (
                self._podcast_position_last_save == 0.0
                or now - self._podcast_position_last_save >= 5
            ) and hasattr(self, "_persist_podcast_position"):
                self._persist_podcast_position()
        except (AttributeError, GLib.Error, TypeError, ValueError):
            pass
        return True

    def _capture_podcast_position(self):
        episode = getattr(self, "_active_podcast_episode", None)
        if episode is None or not getattr(self, "player", None):
            return
        try:
            success, position = self.player.query_position(Gst.Format.TIME)
            if not success:
                return
            position_seconds = max(0, int(position / Gst.SECOND))
            episode.position = position_seconds
            if (episode.duration > 0 and position_seconds >= episode.duration - 5) or (
                episode.duration <= 0 and position_seconds >= 30
            ):
                episode.played = True
        except (AttributeError, GLib.Error, TypeError, ValueError):
            pass

    def _attempt_podcast_resume(self, episode):
        if (
            episode is None
            or episode.position <= 2
            or self._podcast_resume_attempted
            or not getattr(self, "player", None)
        ):
            return
        self._podcast_resume_attempted = True
        try:
            self.player.seek_simple(
                Gst.Format.TIME,
                Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT,
                int(episode.position * Gst.SECOND),
            )
        except (AttributeError, GLib.Error, TypeError, ValueError):
            pass

    def _handle_podcast_eos(self, generation=None):
        if generation is not None and generation != getattr(
            self, "_playback_generation", 0
        ):
            return False
        self._playback_generation = getattr(self, "_playback_generation", 0) + 1
        self._capture_podcast_position()
        self._stop_podcast_position_tracking()
        self._set_playback_state(False)
        if getattr(self, "player", None):
            self.player.set_state(Gst.State.NULL)
        if hasattr(self, "set_status_message"):
            self.set_status_message("Podcast playback finished")
        return False

    def _handle_podcast_error(self, err, generation=None):
        if generation is not None and generation != getattr(
            self, "_playback_generation", 0
        ):
            return
        self._capture_podcast_position()
        self._stop_podcast_position_tracking()
        self._set_playback_state(False)
        if getattr(self, "player", None):
            self.player.set_state(Gst.State.NULL)
        if hasattr(self, "set_status_message"):
            self.set_status_message(f"Podcast playback failed: {err}")
        return

    def play_podcast_episode(self, episode, feed=None):
        if episode is None or not getattr(episode, "audio_url", ""):
            return False
        audio_path = getattr(episode, "downloaded_path", "")
        if audio_path and _validate_file_path(audio_path, allow_symlinks=False):
            uri = Gst.filename_to_uri(os.path.abspath(audio_path))
            return self._start_podcast_episode_playback(episode, feed, uri)
        uri = episode.audio_url
        parsed = urlparse(uri)
        if parsed.scheme not in ("http", "https"):
            return False
        if not _validate_url(uri):
            GLib.idle_add(
                self.set_status_message,
                "Podcast audio URL blocked: invalid destination",
            )
            return False
        generation = getattr(self, "_playback_generation", 0)
        _validate_url_dns = _linamp_attr("_validate_url_dns", _core_validate_url_dns)
        _get_executor = _linamp_attr("_get_download_executor", _get_download_executor)

        def _validate_then_start():
            try:
                allowed = _validate_url_dns(uri)
            except Exception:
                logging.getLogger(__name__).exception(
                    "Podcast audio URL validation failed"
                )
                allowed = False
            GLib.idle_add(
                self._on_podcast_url_validated, episode, feed, uri, generation, allowed
            )

        executor = _get_executor()
        if executor is None:
            GLib.idle_add(
                self.set_status_message,
                "Podcast audio URL blocked: validation unavailable",
            )
            return False
        try:
            executor.submit(_validate_then_start)
        except RuntimeError:
            logging.getLogger(__name__).exception(
                "Could not schedule podcast audio URL validation"
            )
            GLib.idle_add(
                self.set_status_message,
                "Podcast audio URL blocked: validation unavailable",
            )
            return False
        return True

    def _on_podcast_url_validated(self, episode, feed, uri, generation, allowed):
        """Continue episode playback once the audio URL has been resolved."""
        if generation != getattr(self, "_playback_generation", 0):
            return
        if not allowed:
            self.set_status_message("Podcast audio URL blocked: invalid destination")
            return
        self._start_podcast_episode_playback(episode, feed, uri)

    def _start_podcast_episode_playback(self, episode, feed, uri):
        if not uri:
            return False
        if not hasattr(self, "player") or not self.player:
            return False
        self._desired_play_state = Gst.State.PLAYING
        self._user_stopped = False
        self._set_playback_state(False)
        self._capture_podcast_position()
        self._stop_podcast_position_tracking()
        self._active_podcast_episode = episode
        self._active_podcast_feed = feed
        self._active_podcast_feed_id = getattr(feed, "feed_id", "")
        self._active_podcast_episodes = self._podcast_episodes_for_active_feed()
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
        if hasattr(self, "player_tab") and hasattr(self.player_tab, "track_label"):
            GLib.idle_add(
                self.player_tab.track_label.set_text,
                f"Loading: {getattr(episode, 'title', 'Podcast')}...",
            )
        if not self._prepare_playback(
            uri,
            getattr(episode, "title", "Podcast"),
            is_radio=False,
            podcast_episode=episode,
        ):
            return False
        self._start_podcast_position_tracking()
        return True
