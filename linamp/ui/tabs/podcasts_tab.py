from __future__ import annotations

import gi

gi.require_version("Gdk", "4.0")
gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk, Pango  # noqa: I001

# Wildcard imports re-export the public API of each module.  Because several
# of those modules also ``import logging`` and ``from xml.etree import
# ElementTree as ET``, Pylance can lose track of the explicit stdlib imports
# that were previously bound — even though those names are *not* in any
# module's ``__all__`` and therefore are not actually overwritten at runtime.
# Following the same "restore-after-star-import" convention used in
# ``linamp/__init__.py``, we re-import the names below so that they are the
# final, authoritative binding in static analysis as well as at runtime.
from ...core import *
from ...media import *
from ...models import *
from ...podcasts import *
from ...radio import *
from ...core import _linamp_attr

import copy
import logging
import threading
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from urllib.error import URLError

from .. import _create_labeled_icon_button, _create_modal_form


class PodcastsTab(Gtk.Box):
    """GTK tab for podcast subscription management and episode playback."""

    def __init__(self, player):
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self.player = player
        self.feeds = load_podcasts()
        self.selected_feed = None
        self.episodes = []
        self._filtered_episodes = []
        self.sync_threads = {}
        self._feed_sync_tokens: dict[str, int] = {}
        self._feed_sync_lock = threading.Lock()
        self._shutting_down = False
        self._setup_ui()
        self._populate_feeds()
        if not self.feeds:
            self._default_feed_timer_id = self.player._register_timer(
                5000, self._add_default_feed
            )

    def _schedule_sync_callback(self, callback, *args) -> bool:
        if self._shutting_down:
            return False
        if getattr(self.player, "_cleanup_in_progress", False):
            return False
        if getattr(self.player, "_cleanup_complete", False):
            return False
        GLib.idle_add(callback, *args)
        return False

    def cleanup(self):
        """Stop background sync threads and the default-feed timer."""
        self._shutting_down = True
        timer_id = getattr(self, "_default_feed_timer_id", 0)
        if timer_id and hasattr(self.player, "_safe_source_remove"):
            self.player._safe_source_remove(timer_id)
            self._default_feed_timer_id = 0
        with self._feed_sync_lock:
            threads = list(self.sync_threads.values())
            self.sync_threads.clear()
        for thread in threads:
            if thread is not threading.current_thread():
                thread.join(timeout=1.0)

    def _setup_ui(self):
        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        header.set_margin_top(8)
        header.set_margin_bottom(8)
        header.set_margin_start(8)
        header.set_margin_end(8)
        self.append(header)
        title_label = Gtk.Label(label="Podcasts")
        title_label.add_css_class("section-header")
        title_label.set_halign(Gtk.Align.START)
        header.append(title_label)
        toolbar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        toolbar.set_margin_start(8)
        toolbar.set_margin_end(8)
        toolbar.set_margin_bottom(8)
        self.append(toolbar)
        add_btn = _create_labeled_icon_button("Add Feed", "list-add-symbolic")
        add_btn.connect("clicked", self.on_add_feed)
        toolbar.append(add_btn)
        remove_btn = _create_labeled_icon_button("Remove Feed", "list-remove-symbolic")
        remove_btn.connect("clicked", self.on_remove_feed)
        toolbar.append(remove_btn)
        sync_selected_btn = _create_labeled_icon_button(
            "Sync Selected", "view-refresh-symbolic"
        )
        sync_selected_btn.connect("clicked", self.on_sync_selected)
        toolbar.append(sync_selected_btn)
        sync_all_btn = _create_labeled_icon_button(
            "Sync All", "emblem-synchronizing-symbolic"
        )
        sync_all_btn.connect("clicked", self.on_sync_all)
        toolbar.append(sync_all_btn)
        paned = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        paned.set_hexpand(True)
        paned.set_vexpand(True)
        paned.set_margin_start(8)
        paned.set_margin_end(8)
        paned.set_margin_bottom(8)
        self.append(paned)
        left_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        left_box.set_size_request(250, -1)
        paned.set_start_child(left_box)
        feed_header = Gtk.Label(label="Subscriptions")
        feed_header.add_css_class("section-header")
        feed_header.set_halign(Gtk.Align.START)
        feed_header.set_margin_bottom(6)
        left_box.append(feed_header)
        feed_scrolled = Gtk.ScrolledWindow(
            hscrollbar_policy=Gtk.PolicyType.AUTOMATIC,
            vscrollbar_policy=Gtk.PolicyType.AUTOMATIC,
        )
        feed_scrolled.add_css_class("radio-scrolled")
        feed_scrolled.set_hexpand(True)
        feed_scrolled.set_vexpand(True)
        left_box.append(feed_scrolled)
        self.feed_store = Gtk.StringList()
        self.feed_selection = Gtk.SingleSelection(model=self.feed_store)
        self.feed_list = Gtk.ListView(model=self.feed_selection)
        self.feed_list.add_css_class("radio-view")
        feed_factory = Gtk.SignalListItemFactory()
        feed_factory.connect("setup", self._on_feed_factory_setup)
        feed_factory.connect("bind", self._on_feed_factory_bind)
        self.feed_list.set_factory(feed_factory)
        feed_scrolled.set_child(self.feed_list)
        self.feed_selection.connect("selection-changed", self.on_feed_selection_changed)
        right_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        paned.set_end_child(right_box)
        episode_header = Gtk.Label(label="Episodes")
        episode_header.add_css_class("section-header")
        episode_header.set_halign(Gtk.Align.START)
        episode_header.set_margin_bottom(6)
        right_box.append(episode_header)
        filter_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        filter_box.set_margin_bottom(6)
        right_box.append(filter_box)
        self.episode_filter = Gtk.Entry()
        self.episode_filter.set_placeholder_text("Filter episodes...")
        self.episode_filter.set_hexpand(True)
        self.episode_filter.add_css_class("search-entry")
        self.episode_filter.connect("changed", self.on_episode_filter_changed)
        filter_box.append(self.episode_filter)
        episode_scrolled = Gtk.ScrolledWindow(
            hscrollbar_policy=Gtk.PolicyType.AUTOMATIC,
            vscrollbar_policy=Gtk.PolicyType.AUTOMATIC,
        )
        episode_scrolled.add_css_class("radio-scrolled")
        episode_scrolled.set_hexpand(True)
        episode_scrolled.set_vexpand(True)
        right_box.append(episode_scrolled)
        self.episode_store = Gtk.StringList()
        self.episode_selection = Gtk.SingleSelection(model=self.episode_store)
        self.episode_list = Gtk.ListView(model=self.episode_selection)
        self.episode_list.add_css_class("radio-view")
        episode_factory = Gtk.SignalListItemFactory()
        episode_factory.connect("setup", self._on_episode_factory_setup)
        episode_factory.connect("bind", self._on_episode_factory_bind)
        self.episode_list.set_factory(episode_factory)
        episode_scrolled.set_child(self.episode_list)
        self.episode_list.connect("activate", self.on_episode_activated)
        episode_toolbar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        episode_toolbar.set_margin_top(6)
        right_box.append(episode_toolbar)
        mark_played_btn = _create_labeled_icon_button(
            "Mark Played", "emblem-ok-symbolic"
        )
        mark_played_btn.connect("clicked", self.on_mark_played)
        episode_toolbar.append(mark_played_btn)
        mark_unplayed_btn = _create_labeled_icon_button(
            "Mark Unplayed", "emblem-default-symbolic"
        )
        mark_unplayed_btn.connect("clicked", self.on_mark_unplayed)
        episode_toolbar.append(mark_unplayed_btn)

    def _on_feed_factory_setup(self, factory, list_item):
        label = Gtk.Label()
        label.set_halign(Gtk.Align.START)
        label.set_margin_start(6)
        label.set_margin_end(6)
        label.set_margin_top(3)
        label.set_margin_bottom(3)
        list_item.set_child(label)

    def _on_feed_factory_bind(self, factory, list_item):
        label = list_item.get_child()
        string_obj = list_item.get_item()
        label.set_text(string_obj.get_string())

    def _on_episode_factory_setup(self, factory, list_item):
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        box.set_margin_start(6)
        box.set_margin_end(6)
        box.set_margin_top(3)
        box.set_margin_bottom(3)
        played_indicator = Gtk.Image()
        played_indicator.set_from_icon_name("emblem-ok-symbolic")
        played_indicator.set_pixel_size(12)
        played_indicator.set_visible(False)
        box.append(played_indicator)
        label = Gtk.Label()
        label.set_halign(Gtk.Align.START)
        label.set_hexpand(True)
        label.set_ellipsize(Pango.EllipsizeMode.END)
        box.append(label)
        duration_label = Gtk.Label()
        duration_label.add_css_class("dim-label")
        box.append(duration_label)
        list_item.set_child(box)

    def _on_episode_factory_bind(self, factory, list_item):
        box = list_item.get_child()
        string_obj = list_item.get_item()
        parts = string_obj.get_string().rsplit("|", 2)
        if len(parts) == 3:
            title, duration, played = parts[0], parts[1], parts[2] == "True"
            played_indicator = box.get_first_child()
            played_indicator.set_visible(played)
            label = played_indicator.get_next_sibling()
            label.set_text(title)
            duration_label = label.get_next_sibling()
            duration_label.set_text(duration)

    def _populate_feeds(self):
        # Preserve the current selection across the store rebuild by feed_id.
        # During splice, selection-changed fires with INVALID position, which
        # clears self.selected_feed in on_feed_selection_changed. We restore
        # the selection afterwards to avoid losing the user's place.
        saved_feed_id = getattr(self.selected_feed, "feed_id", "")

        self.feed_store.splice(0, self.feed_store.get_n_items())
        for feed in self.feeds:
            self.feed_store.append(feed.title)

        # Restore selection to the same feed (by feed_id), or default to 0.
        new_selected = 0
        if saved_feed_id:
            for idx, feed in enumerate(self.feeds):
                if feed.feed_id == saved_feed_id:
                    new_selected = idx
                    break

        if self.feeds and new_selected < len(self.feeds):
            self.feed_selection.set_selected(new_selected)
            # Guard against GTK deduplicating selection-changed when the
            # position didn't visibly change across the splice.  Without this,
            # self.selected_feed stays None (cleared by the splice handler) and
            # the episode list stays empty, making it appear the user cannot
            # select feeds.
            if (
                self.selected_feed is None
                or getattr(self.selected_feed, "feed_id", "")
                != self.feeds[new_selected].feed_id
            ):
                self.selected_feed = self.feeds[new_selected]
                self.episodes = load_feed_episodes(self.selected_feed.feed_id)
                self._populate_episodes()
        elif not self.feeds:
            self.selected_feed = None
            self.episodes = []
            self._populate_episodes()

    def _populate_episodes(self):
        self.episode_store.splice(0, self.episode_store.get_n_items())
        self._filtered_episodes = []
        filter_text = self.episode_filter.get_text().lower()
        for episode in self.episodes:
            title = episode.title
            if filter_text and filter_text not in title.lower():
                continue
            duration_str = self._format_duration(episode.duration)
            self._filtered_episodes.append(episode)
            display = f"{title}|{duration_str}|{episode.played}"
            self.episode_store.append(display)

    def _format_duration(self, seconds: int) -> str:
        if not seconds:
            return ""
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        if hours:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        return f"{minutes}:{secs:02d}"

    def on_feed_selection_changed(self, selection, position, n_items):
        selected = self.feed_selection.get_selected()
        if (
            (
                selected == Gtk.INVALID_LIST_POSITION
                or selected < 0
                or selected >= len(self.feeds)
            )
            and position is not None
            and 0 <= position < len(self.feeds)
        ):
            selected = position
        if selected is not None and 0 <= selected < len(self.feeds):
            self.selected_feed = self.feeds[selected]
            self.episodes = load_feed_episodes(self.selected_feed.feed_id)
            self._populate_episodes()
        else:
            self.selected_feed = None
            self.episodes = []
            self._populate_episodes()

    def on_episode_filter_changed(self, entry):
        self._populate_episodes()

    def on_episode_activated(self, list_view, position):
        if not (
            self.selected_feed
            and position >= 0
            and position < len(self._filtered_episodes)
        ):
            return
        episode = self._filtered_episodes[position]
        if not episode.audio_url or not hasattr(self.player, "play_podcast_episode"):
            return
        if not self.player.play_podcast_episode(episode, self.selected_feed):
            return
        if not episode.artwork_url:
            return
        self.player._podcast_art_generation += 1
        generation = self.player._podcast_art_generation

        def _show_artwork():
            try:
                path = fetch_podcast_artwork(episode.artwork_url)
            except Exception as exc:  # noqa: BLE001
                logging.getLogger(__name__).debug(
                    "Podcast artwork fetch failed: %s", exc
                )
                return False
            if generation != getattr(self.player, "_podcast_art_generation", 0):
                return False
            if getattr(self.player, "_cleanup_in_progress", False):
                return False
            cover = getattr(
                getattr(self.player, "player_tab", None),
                "cover_art_widget",
                None,
            )
            if path and cover and hasattr(cover, "load_cover_from_path"):

                def _apply_artwork():
                    if generation != getattr(self.player, "_podcast_art_generation", 0):
                        return False
                    if getattr(self.player, "_cleanup_in_progress", False):
                        return False
                    if getattr(self.player, "_cleanup_complete", False):
                        return False
                    cover.load_cover_from_path(str(path))
                    return False

                GLib.idle_add(_apply_artwork)
            return False

        if self.player._cover_executor is None:
            self.player._cover_executor = ThreadPoolExecutor(max_workers=2)
        try:
            self.player._cover_executor.submit(_show_artwork)
        except RuntimeError:
            pass

    def on_add_feed(self, button):
        root = self.get_root()
        dialog, entries, cancel_button, add_button = _create_modal_form(
            "Add Podcast Feed",
            root if isinstance(root, Gtk.Window) else None,
            ["RSS or Atom feed URL", "Custom title (optional)"],
            "Add & Sync",
        )
        url_entry, title_entry = entries

        def close_dialog(*_args):
            dialog.close()

        def on_add(*_args):
            url = url_entry.get_text().strip()
            custom_title = title_entry.get_text().strip()
            if not url:
                self.player.set_status_message("URL is required")
                url_entry.grab_focus()
                return
            if not _validate_url(url):
                self.player.set_status_message(
                    "Invalid feed URL — use a public http:// or https:// URL"
                )
                url_entry.grab_focus()
                return
            self._add_and_sync_feed(url, custom_title)
            dialog.close()

        cancel_button.connect("clicked", close_dialog)
        add_button.connect("clicked", on_add)
        url_entry.connect("activate", lambda *_args: add_button.emit("clicked"))
        dialog.connect("close-request", lambda *_args: False)
        dialog.present()

    def _add_and_sync_feed(self, feed_url: str, custom_title: str = ""):
        if not hasattr(self.player, "status_label"):
            return
        feed_id = PodcastFeed.generate_feed_id(feed_url)
        token = self._feed_sync_tokens.get(feed_id, 0) + 1
        self._feed_sync_tokens[feed_id] = token

        def _sync_worker():
            if self._shutting_down:
                return
            try:
                existing_feed = next(
                    (f for f in list(self.feeds) if f.feed_id == feed_id), None
                )
                new_feed = sync_feed(feed_url, existing_feed)
                if custom_title:
                    new_feed.title = custom_title
                self._schedule_sync_callback(
                    self._apply_synced_feed_main_thread,
                    token,
                    feed_id,
                    new_feed,
                    existing_feed is not None,
                )
            except (ValueError, URLError, OSError, ET.ParseError) as e:
                self._schedule_sync_callback(
                    self._post_status_if_current,
                    feed_id,
                    token,
                    f"Sync failed: {e}",
                )
            except Exception:
                logging.getLogger(__name__).exception("Sync failed")
                self._schedule_sync_callback(
                    self._post_status_if_current,
                    feed_id,
                    token,
                    "Sync failed: unexpected error",
                )

        thread = threading.Thread(target=_sync_worker, daemon=True, name="PodcastSync")
        if not self._claim_feed_sync_slot(feed_url, thread):
            return
        thread.start()
        if hasattr(self.player, "status_label"):
            self.player.set_status_message("Syncing feed...")

    def _post_status_if_current(self, feed_id: str, token: int, message: str) -> bool:
        """Queue a status message only if this sync is still the current one.
        The success path already guards on generation and cleanup state; this
        helper applies the same guard to the error path so a sync that fails
        after shutdown begins cannot queue a GTK callback against a destroyed
        player, and so a stale/slow sync cannot overwrite a newer sync's
        status with an obsolete error message.
        """
        if self._shutting_down:
            return False
        if self._feed_sync_tokens.get(feed_id, 0) != token:
            return False
        if getattr(self.player, "_cleanup_in_progress", False):
            return False
        if getattr(self.player, "_cleanup_complete", False):
            return False
        if hasattr(self.player, "status_label"):
            self.player.set_status_message(message)
        return False

    def _apply_synced_feed_main_thread(
        self, token: int, feed_id: str, new_feed: PodcastFeed, is_update: bool
    ):
        if self._shutting_down:
            return False
        if self._feed_sync_tokens.get(feed_id, 0) != token:
            return False
        if getattr(self.player, "_cleanup_in_progress", False):
            return False
        if getattr(self.player, "_cleanup_complete", False):
            return False
        self._apply_synced_feed(feed_id, new_feed, is_update)
        return False

    def _apply_synced_feed(self, feed_id: str, new_feed: PodcastFeed, is_update: bool):
        with self._feed_sync_lock:
            self._apply_synced_feed_locked(feed_id, new_feed, is_update)

    def _apply_synced_feed_locked(
        self, feed_id: str, new_feed: PodcastFeed, is_update: bool
    ):
        """Install *new_feed* into the in-memory feed list.

        ``is_update`` is recorded for callers but does not change the
        merge rule: an existing subscription with the same ``feed_id`` is
        replaced in place (preserving list order), otherwise the feed is
        appended.  Both branches previously implemented this rule; the
        non-update branch re-iterated ``self.feeds`` inside an outer
        loop, which worked but was needlessly convoluted and shadowed the
        loop variable.
        """
        for idx, feed in enumerate(self.feeds):
            if feed.feed_id == feed_id:
                self.feeds[idx] = new_feed
                break
        else:
            self.feeds.append(new_feed)
        invalidate = getattr(
            getattr(self, "player", None), "_invalidate_podcast_persists", None
        )
        if callable(invalidate):
            invalidate(new_feed.feed_id)
        if not save_podcast_state(self.feeds, new_feed.feed_id, new_feed.episodes):
            logging.getLogger(__name__).error(
                "Failed to persist synced feed %s", new_feed.feed_id
            )
            if hasattr(self.player, "status_label"):
                self.player.set_status_message(f"Failed to save: {new_feed.title}")
        if self.selected_feed and self.selected_feed.feed_id == feed_id:
            self.selected_feed = new_feed
            self.episodes = list(new_feed.episodes)
            self._populate_episodes()
        self._populate_feeds()
        if hasattr(self.player, "status_label"):
            self.player.set_status_message(f"Synced: {new_feed.title}")

    def on_remove_feed(self, button):
        selected = self.feed_selection.get_selected()
        if selected is None or selected < 0 or selected >= len(self.feeds):
            if hasattr(self.player, "status_label"):
                self.player.set_status_message("No feed selected")
            return
        feed = self.feeds[selected]
        player = getattr(self, "player", None)
        invalidate = getattr(player, "_invalidate_podcast_persists", None)
        if callable(invalidate):
            invalidate(feed.feed_id)
        flush = getattr(player, "_flush_persist_futures", None)
        if callable(flush):
            flush(1.0)
        self.feeds.pop(selected)
        # Resolve remove_podcast_feed through the live ``linamp`` module so
        # that tests which monkeypatch linamp.remove_podcast_feed are honoured
        # instead of the local binding captured by ``from ...podcasts import *``.
        # The optional ``sync_tokens`` argument is not consumed by
        # remove_podcast_feed (the journal + index rewrite is self-contained),
        # and this tab tracks per-feed sync *generations* (ints) rather than
        # token lists, so no token payload is passed here; in-flight workers are
        # invalidated via ``_invalidate_podcast_persists`` above.
        _remove = _linamp_attr("remove_podcast_feed", remove_podcast_feed)
        if not _remove(feed.feed_id):
            logging.getLogger(__name__).error(
                "Failed to remove podcast feed %s", feed.feed_id
            )
            self.feeds.insert(selected, feed)
            if hasattr(self.player, "status_label"):
                self.player.set_status_message("Failed to remove feed")
            return
        self._feed_sync_tokens[feed.feed_id] = (
            self._feed_sync_tokens.get(feed.feed_id, 0) + 1
        )
        self.selected_feed = None
        self.episodes = []
        self._populate_feeds()
        self._populate_episodes()
        if hasattr(self.player, "status_label"):
            self.player.set_status_message(f"Removed feed: {feed.title}")

    def on_sync_selected(self, button):
        selected = self.feed_selection.get_selected()
        if selected is None or selected < 0 or selected >= len(self.feeds):
            if hasattr(self.player, "status_label"):
                self.player.set_status_message("No feed selected")
            return
        feed = self.feeds[selected]
        self._sync_feed(feed)

    def on_sync_all(self, button):
        if not self.feeds:
            if hasattr(self.player, "status_label"):
                self.player.set_status_message("No feeds to sync")
            return
        for feed in self.feeds:
            self._sync_feed(feed)
        if hasattr(self.player, "status_label"):
            self.player.set_status_message(f"Syncing {len(self.feeds)} feeds...")

    def _sync_feed(self, feed: PodcastFeed):
        feed_url = feed.feed_url
        feed_id = feed.feed_id
        token = self._feed_sync_tokens.get(feed_id, 0) + 1
        self._feed_sync_tokens[feed_id] = token

        def _sync_worker():
            if self._shutting_down:
                return
            try:
                existing_feed = feed
                new_feed = sync_feed(feed_url, existing_feed)
                self._schedule_sync_callback(
                    self._apply_synced_feed_main_thread,
                    token,
                    feed_id,
                    new_feed,
                    True,
                )
            except (ValueError, URLError, OSError, ET.ParseError) as e:
                self._schedule_sync_callback(
                    self._post_status_if_current,
                    feed_id,
                    token,
                    f"Sync failed for {feed.title}: {e}",
                )
            except Exception:
                logging.getLogger(__name__).exception("Sync failed for %s", feed.title)
                self._schedule_sync_callback(
                    self._post_status_if_current,
                    feed_id,
                    token,
                    f"Sync failed for {feed.title}: unexpected error",
                )
            finally:
                with self._feed_sync_lock:
                    if self.sync_threads.get(feed_url) is threading.current_thread():
                        self.sync_threads.pop(feed_url, None)

        thread = threading.Thread(target=_sync_worker, daemon=True, name="PodcastSync")
        if not self._claim_feed_sync_slot(feed_url, thread):
            return
        thread.start()

    def _claim_feed_sync_slot(self, feed_url: str, thread: threading.Thread) -> bool:
        """Register *thread* as the sync for *feed_url*, or reject the start.
        Repeatedly hitting "Refresh" used to overwrite the dict entry and leave
        the previous thread running and unjoinable, so each click added another
        full feed fetch with nothing bounding the total.  One live sync per feed
        is the cap; a second request for the same feed is simply ignored.
        """
        with self._feed_sync_lock:
            if self._shutting_down:
                return False
            existing = self.sync_threads.get(feed_url)
            if existing is not None and existing.is_alive():
                return False
            self.sync_threads[feed_url] = thread
            return True

    def on_mark_played(self, button):
        selected = self.episode_selection.get_selected()
        if selected is None or selected < 0 or selected >= len(self._filtered_episodes):
            if hasattr(self.player, "status_label"):
                self.player.set_status_message("No episode selected")
            return
        episode = self._filtered_episodes[selected]
        episode.played = True
        episode.position = 0
        self._persist_episode_state()
        self._populate_episodes()
        if hasattr(self.player, "status_label"):
            self.player.set_status_message(f"Marked as played: {episode.title}")

    def on_mark_unplayed(self, button):
        selected = self.episode_selection.get_selected()
        if selected is None or selected < 0 or selected >= len(self._filtered_episodes):
            if hasattr(self.player, "status_label"):
                self.player.set_status_message("No episode selected")
            return
        episode = self._filtered_episodes[selected]
        episode.played = False
        episode.position = 0
        self._persist_episode_state()
        self._populate_episodes()
        if hasattr(self.player, "status_label"):
            self.player.set_status_message(f"Marked as unplayed: {episode.title}")

    def _persist_episode_state(self) -> None:
        """Submit the current episode list through the unified writer.
        Routes the mark-played/mark-unplayed mutations through
        ``WinampWindow._save_podcast_snapshot`` so the episode file and the
        embedded copy in ``podcasts.json`` are updated in the same
        transactional generation.  Falls back to the legacy direct write if
        the unified path is unavailable (e.g. during tests), but never
        raises: the UI must remain responsive.
        """
        if not self.selected_feed:
            return
        feed_id = self.selected_feed.feed_id
        episodes_snapshot = [copy.copy(ep) for ep in self.episodes]
        feeds_snapshot = []
        for feed in self.feeds:
            if feed.feed_id == feed_id:
                feeds_snapshot.append(
                    PodcastFeed(
                        feed_id=feed.feed_id,
                        title=feed.title,
                        author=feed.author,
                        description=feed.description,
                        artwork_url=feed.artwork_url,
                        homepage=feed.homepage,
                        feed_url=feed.feed_url,
                        last_updated=feed.last_updated,
                        episodes=list(self.episodes),
                    )
                )
            else:
                feeds_snapshot.append(feed)
        persist = getattr(self.player, "_save_podcast_snapshot", None)
        if persist is not None:
            try:
                persist(feed_id, episodes_snapshot, feeds_snapshot)
                return
            except Exception:  # noqa: BLE001
                logging.getLogger(__name__).debug(
                    "Unified episode persist failed; falling back"
                )
        save_feed_episodes(feed_id, episodes_snapshot)

    def _add_default_feed(self):
        """Add NPR Up First as the default starter feed."""
        if getattr(self.player, "_cleanup_in_progress", False) or getattr(
            self.player, "_cleanup_complete", False
        ):
            return False
        if not hasattr(self.player, "status_label"):
            return True
        npr_feed_url = "https://feeds.npr.org/510289/podcast.xml"
        self._add_and_sync_feed(npr_feed_url, "NPR Up First")
        return False
