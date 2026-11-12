"""WinampWindow mixin: Radio stream resolution, redirect validation and reconnect."""

from __future__ import annotations

import gi

from .base import WindowBase

gi.require_version("Gdk", "4.0")
gi.require_version("Gtk", "4.0")
gi.require_version("Gst", "1.0")
import logging
from urllib.parse import urlparse

from gi.repository import GLib, Gst

from ...core import (
    _DNS_VALIDATION_CACHE,
    _DNS_VALIDATION_CACHE_LOCK,
    RADIO_RECONNECT_MAX_FAILURES,
    REQUESTS_AVAILABLE,
    _get_download_executor,
    _get_shared_session,
    _linamp_attr,
    _parse_ip_literal,
    _safe_http_get,
    _validate_url,
)
from ...core import (
    _validate_url_dns as _core_validate_url_dns,
)


class RadioMixin(WindowBase):
    """Radio stream resolution, redirect validation and reconnect.

    The methods are split out of ``WinampWindow`` so the window class
    stays readable; they keep operating on the window instance and
    share its playback, playlist and settings state.
    """

    def _resolve_radio_url_async(self, uri, callback):
        """Resolve radio URL asynchronously with redirect validation.
        Uses :func:`_safe_http_get` with a controlled GET request to follow
        redirects and validate each hop.  GET is used instead of HEAD because
        some radio servers do not support HEAD requests.
        Old resolution jobs become no-ops once the playback generation
        changes (a new track/radio station was selected), so the callback
        is never invoked for stale jobs.
        """
        generation = getattr(self, "_playback_generation", 0)
        _validate_url_dns = _linamp_attr("_validate_url_dns", _core_validate_url_dns)
        _http_get = _linamp_attr("_safe_http_get", _safe_http_get)
        _get_sess = _linamp_attr("_get_shared_session", _get_shared_session)
        _get_executor = _linamp_attr("_get_download_executor", _get_download_executor)

        def worker():
            try:
                if not REQUESTS_AVAILABLE:
                    GLib.idle_add(callback, None)
                    return
                sess = _get_sess()
                if not sess:
                    GLib.idle_add(callback, None)
                    return
                if (
                    generation != getattr(self, "_playback_generation", 0)
                    or getattr(self, "_user_stopped", False)
                    or not getattr(self, "_is_radio_stream", False)
                ):
                    return
                if not _validate_url_dns(uri):
                    GLib.idle_add(callback, None)
                    return
                resp = _http_get(
                    uri,
                    session=sess,
                    timeout=10.0,
                    stream=True,
                    method="GET",
                )
                if resp is None:
                    GLib.idle_add(callback, None)
                    return
                try:
                    final_url = getattr(resp, "url", None)
                finally:
                    resp.close()
                if (
                    not final_url
                    or generation != getattr(self, "_playback_generation", 0)
                    or getattr(self, "_user_stopped", False)
                    or not getattr(self, "_is_radio_stream", False)
                    or not _validate_url_dns(final_url)
                ):
                    GLib.idle_add(callback, None)
                    return
                GLib.idle_add(callback, final_url)
            except Exception:
                logging.getLogger(__name__).exception("Radio URL resolve failed")
                GLib.idle_add(callback, None)

        if getattr(self, "_cleanup_in_progress", False) or getattr(
            self, "_cleanup_complete", False
        ):
            return
        executor = _get_executor()
        if executor is None:
            GLib.idle_add(callback, None)
            return
        try:
            executor.submit(worker)
        except RuntimeError:
            logging.getLogger(__name__).debug(
                "Radio URL resolve not scheduled: download executor unavailable"
            )
            GLib.idle_add(callback, None)

    def _validate_radio_redirect(self, message) -> bool:
        """Validate redirect destination URLs in bus redirect messages.

        For radio streams, ``automatic-redirect`` is disabled on the source
        so that souphttpsrc does not follow redirects unvalidated.  Any
        redirect message that reaches the bus is therefore unexpected and
        must be validated before we allow it through (which lets
        ``_handle_bus_sync_message`` set the new location on the source).

        For podcast episodes, ``automatic-redirect`` remains enabled because
        the initial URL was pre-validated via ``_validate_url_dns`` and podcast
        audio URLs commonly redirect through tracking CDNs.  The redirect
        message is informational in this case; we validate it as a safety net.

        Validation is non-blocking: ``_validate_url`` performs static checks
        (scheme, hostname presence, loopback rejection, and IP-literal
        public-address verification).  For hostname-based redirects we also
        consult the DNS validation cache so that hosts already resolved during
        pre-playback validation are trusted immediately.
        """
        try:
            num_entries = message.get_num_redirect_entries()
            if num_entries == 0:
                return True
            for index in range(num_entries):
                redirect_url, _tags, _entry = message.parse_redirect_entry(index)
                if not redirect_url:
                    continue
                # Static structural validation: rejects non-http schemes,
                # localhost/loopback, and private/link-local IP literals.
                if not _validate_url(redirect_url):
                    logging.getLogger(__name__).warning(
                        "Unsafe redirect blocked: %s", redirect_url
                    )
                    if hasattr(self, "set_status_message"):
                        GLib.idle_add(
                            self.set_status_message, "Blocked unsafe redirect"
                        )
                    return False
                # For hostname-based redirects (not bare IP literals), consult
                # the DNS validation cache.  Hostnames already resolved during
                # pre-playback validation (e.g. the podcast CDN) will have a
                # cached verdict; if the cache says the host is unsafe, reject.
                parsed = urlparse(redirect_url)
                hostname = parsed.hostname
                if hostname and _parse_ip_literal(hostname) is None:
                    cache_key = hostname.lower()
                    with _DNS_VALIDATION_CACHE_LOCK:
                        cached = _DNS_VALIDATION_CACHE.get(cache_key)
                    if cached is not None and not cached[1]:
                        logging.getLogger(__name__).warning(
                            "Redirect to DNS-blocked host: %s", redirect_url
                        )
                        if hasattr(self, "set_status_message"):
                            GLib.idle_add(
                                self.set_status_message,
                                "Blocked unsafe redirect",
                            )
                        return False
            return True
        except (AttributeError, GLib.Error, TypeError, ValueError):
            logging.getLogger(__name__).warning("Radio redirect message parsing failed")
            return False

    def _invalidate_radio_reconnect(self):
        self._radio_reconnect_pending = False
        self._radio_reconnect_token = getattr(self, "_radio_reconnect_token", 0) + 1
        self._radio_reconnect_failures = 0

    def _schedule_radio_reconnect(self):
        """Schedule a radio reconnect with atomic STOP → NULL → URI → PLAYING sequence.
        This replaces the previous multi-callback approach with a single timer that
        performs the entire reconnect sequence atomically, preventing overlapping
        playback attempts.
        After repeated failures with the current resolved stream URL (e.g. an
        expired CDN endpoint), the original station URL is re-resolved to obtain
        a fresh stream endpoint.
        """
        if not getattr(self, "_is_radio_stream", False) or not getattr(
            self, "player", None
        ):
            return
        if getattr(self, "_user_stopped", False):
            return
        if getattr(self, "_radio_reconnect_pending", False):
            return
        failures = getattr(self, "_radio_reconnect_failures", 0)
        if failures >= RADIO_RECONNECT_MAX_FAILURES:
            self._radio_reconnect_failures = 0
            if hasattr(self, "set_status_message"):
                GLib.idle_add(self.set_status_message, "Station unavailable")
            self._set_playback_state(False)
            self._invalidate_radio_reconnect()
            return
        failures = self._radio_reconnect_failures = failures + 1
        source_uri = getattr(self, "_radio_source_uri", "")
        stream_uri = getattr(self, "_radio_stream_uri", "")
        if not stream_uri and not source_uri:
            return
        self._set_playback_state(False)
        generation = getattr(self, "_playback_generation", 0)
        self._radio_reconnect_pending = True
        self._radio_reconnect_token = getattr(self, "_radio_reconnect_token", 0) + 1
        token = self._radio_reconnect_token
        title = getattr(self, "_radio_stream_title", "Radio Stream")
        reconnect_id = None

        def _do_atomic_reconnect() -> bool:
            if reconnect_id is not None:
                self._safe_source_remove(reconnect_id)
            if getattr(self, "_user_stopped", False):
                self._radio_reconnect_pending = False
                return False
            if (
                token != getattr(self, "_radio_reconnect_token", 0)
                or generation != getattr(self, "_playback_generation", 0)
                or not getattr(self, "_is_radio_stream", False)
            ):
                self._radio_reconnect_pending = False
                return False
            if getattr(self, "_desired_play_state", Gst.State.PLAYING) != (
                Gst.State.PLAYING
            ):
                self._radio_reconnect_pending = False
                return False
            current_uri = getattr(self, "_radio_stream_uri", None)
            if not current_uri:
                current_uri = source_uri or stream_uri
            if not current_uri:
                self._radio_reconnect_pending = False
                return False
            self._radio_reconnect_pending = False
            if not hasattr(self, "player") or not self.player:
                return False
            if not self._prepare_playback(current_uri, title, is_radio=True):
                logging.getLogger(__name__).warning(
                    "Radio reconnect failed to start for %s; retrying", current_uri
                )
                if hasattr(self, "set_status_message"):
                    GLib.idle_add(
                        self.set_status_message, "Reconnect failed - retrying..."
                    )
                if self._radio_reconnect_failures < RADIO_RECONNECT_MAX_FAILURES:
                    backoff = min(5000, 1500 * (2 ** min(4, self._radio_reconnect_failures - 1)))
                    self._register_timer(backoff, _do_atomic_reconnect)
                    return False
                else:
                    self._radio_reconnect_failures = 0
                    if hasattr(self, "set_status_message"):
                        GLib.idle_add(self.set_status_message, "Station unavailable")
                    self._set_playback_state(False)
                    self._invalidate_radio_reconnect()
                return False
            self._radio_reconnect_failures = 0
            if hasattr(self, "set_status_message"):
                GLib.idle_add(self.set_status_message, "Reconnecting to stream...")
            return False

        if failures >= 3 and source_uri and source_uri != stream_uri:

            def _on_resolved(resolved_uri) -> None:
                if (
                    token != getattr(self, "_radio_reconnect_token", 0)
                    or generation != getattr(self, "_playback_generation", 0)
                    or not getattr(self, "_is_radio_stream", False)
                    or getattr(self, "_user_stopped", False)
                    or getattr(self, "_desired_play_state", Gst.State.PLAYING)
                    != Gst.State.PLAYING
                ):
                    self._radio_reconnect_pending = False
                    return
                if resolved_uri and _validate_url(resolved_uri):
                    self._radio_stream_uri = resolved_uri
                    self._radio_reconnect_failures = 0
                    logging.getLogger(__name__).info(
                        "Radio reconnect: re-resolved source URL to a fresh endpoint"
                    )
                else:
                    logging.getLogger(__name__).warning(
                        "Radio reconnect: re-resolution failed, "
                        "falling back to current stream URI"
                    )
                _do_atomic_reconnect()

            self._resolve_radio_url_async(source_uri, _on_resolved)
            return
        backoff = min(5000, 1500 * (2 ** min(4, failures - 1)))
        reconnect_id = self._register_timer(backoff, _do_atomic_reconnect)
        return

    def _on_radio_tag(self, key, value):
        display = value
        if key == "StreamTitle" and " - " in value:
            display = value.split(" - ", 1)[1]
        elif key == "icy-name" and value:
            display = f"🎙 {value}"
        if hasattr(self, "player_tab") and hasattr(self.player_tab, "track_label"):
            GLib.idle_add(self.player_tab.track_label.set_text, display)
            GLib.idle_add(self.set_title, f"LinAmp - {display}")
        if hasattr(self, "header_nowplaying") and self.header_nowplaying:
            GLib.idle_add(self.header_nowplaying.set_text, display)
