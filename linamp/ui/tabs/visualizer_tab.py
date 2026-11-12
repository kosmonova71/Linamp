from __future__ import annotations

import logging
import threading
import time
from collections import deque
from pathlib import Path

import gi

gi.require_version("Gdk", "4.0")
gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gst, Gtk

from renderer.audio_reactivity import AudioReactivity

from ...core import *
from ...core import MILKDROP_AVAILABLE, NUMPY_AVAILABLE, MilkdropRenderer, np
from ...media import *
from ...models import *
from ...podcasts import *
from ...radio import *


class VisualizerTab(Gtk.Box):
    """GTK tab that hosts a MilkDrop visualizer driven by live GStreamer audio."""

    def __init__(self, player, available_presets=None):
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self.logger = logging.getLogger(__name__)
        self.player = player
        self.available_presets = available_presets or []
        self.visualizer = None
        self.visualizer_type = None
        self._rendering_enabled = False
        self.audio_probe_id = None
        self._probe_sink = None
        self._probe_pad = None
        self._probe_connecting = False
        self._probe_retry_id = None
        self._probe_retry_count = 0
        self._audio_sink_retry_id = None
        self._audio_sink_retry_count = 0
        self._map_handler_id = 0
        self._unmap_handler_id = 0
        self._map_render_timer_id = 0
        self._audio_lock = threading.Lock()
        self._latest_audio = None
        self._latest_audio_seq = 0
        self._last_consumed_audio_seq = 0
        self._audio_channels = 2
        self._audio_sample_rate = 44100.0
        self._audio_dt = 0.0
        self._last_audio_time = time.monotonic()
        # New audio reactivity analyzer (replaces legacy analysis)
        self._audio_analyzer = AudioReactivity(
            sample_rate=44100,
            fft_size=4096,
            spectrum_bins=SPECTRUM_BINS,
        )
        self._level_snapshot_buffer: deque = deque(maxlen=AUDIO_SNAPSHOT_BUFFER)
        # Rolling PCM buffer for FFT (4096 samples)
        self._rolling_pcm_target_size = 4096
        self._rolling_pcm_buffer = np.zeros(
            self._rolling_pcm_target_size, dtype=np.float32
        )
        self._rolling_pcm_size = 0
        self._audio_queue: list[tuple] = []
        self._audio_consume_timer_id = None
        self._render_timer_id = None
        self._render_tick_callback_id = None
        self._render_tick_widget = None
        self._last_vis_update = 0.0
        self._audio_active = False
        self._last_audio_activity = 0.0
        self._last_idle_render = 0.0
        self._deactivated = False
        self._visualizer_audio_diagnostic = ""
        self._probe_caps_watch_id: int | None = None
        self._audio_sink_watch_id: int | None = None
        self._audio_sink_watch_bin: object | None = None
        self._player_state_watch_id: int | None = None
        self.set_visible(True)
        self.set_hexpand(True)
        self.set_vexpand(True)
        self.set_halign(Gtk.Align.FILL)
        self.set_valign(Gtk.Align.FILL)
        self.add_css_class("visualizer-widget")
        self._map_handler_id = self.connect("map", self._on_map)
        self._unmap_handler_id = self.connect("unmap", self._on_unmap)
        self._create_visualizer()

    def _create_visualizer(self):
        """Create and append the MilkDrop renderer.
        Safe to call multiple times. Any previously created visualizer is
        removed first so the recovery path after an audio-sink failure can
        re-establish the renderer.
        This always succeeds when MILKDROP_AVAILABLE is True;
        WinampWindow.__init__() raises before this tab is constructed
        if the renderer is unavailable, so no fallback placeholder
        is needed here.
        """
        self._audio_analyzer.reset()
        with self._audio_lock:
            self._latest_audio = None
            self._latest_audio_seq = 0
            self._last_consumed_audio_seq = 0
            self._audio_channels = 2
            self._audio_sample_rate = 44100.0
            self._audio_dt = 0.0
            self._rolling_pcm_size = 0
        child = self.get_first_child()
        while child is not None:
            next_child = child.get_next_sibling()
            self.remove(child)
            child = next_child
        use_milkdrop = MILKDROP_AVAILABLE and MilkdropRenderer is not None
        if use_milkdrop and MilkdropRenderer is not None:
            try:
                auto_switch = False
                if hasattr(self.player, "settings"):
                    auto_switch = getattr(
                        self.player.settings, "milkdrop_auto_switch", False
                    )
                self.visualizer = MilkdropRenderer(auto_switch=auto_switch)
                self.visualizer.set_hexpand(True)
                self.visualizer.set_vexpand(True)
                self.visualizer.set_halign(Gtk.Align.FILL)
                self.visualizer.set_valign(Gtk.Align.FILL)
                self.visualizer.set_visible(True)
                self.visualizer.add_css_class("visualizer-glarea")
                self.append(self.visualizer)
                self.visualizer_type = "milkdrop"
                if hasattr(self.visualizer, "available_milk_files"):
                    self.visualizer.available_milk_files = sorted(
                        Path(p) for p in self.available_presets
                    )
                if hasattr(self.visualizer, "update_audio"):
                    self.update_audio = self.visualizer.update_audio
            except Exception:  # noqa: BLE001
                self.visualizer = None
                self.visualizer_type = None
        if hasattr(self, "set_audio_analysis_available"):
            self.set_audio_analysis_available(
                getattr(self.player, "_visualizer_audio_available", False)
            )

    def set_available_presets(self, presets):
        self.available_presets = list(presets)
        milk_files = sorted(Path(p) for p in self.available_presets)
        if self.visualizer is not None:
            self.visualizer.available_milk_files = milk_files
        if self.visualizer is not None and hasattr(
            self.visualizer, "set_available_presets"
        ):
            self.visualizer.set_available_presets(self.available_presets)
        # MilkDrop always presents a preset on startup; an unseeded renderer
        # would clear its feedback texture every frame (the main visual pass
        # early-returns when no preset shader is compiled) and the widget would
        # show a frozen/static screen.  Seed the first available preset whenever
        # the renderer has none active, regardless of ``auto_switch`` (which only
        # governs periodic switching, not whether an initial preset loads).
        # ``smooth_transition=False`` installs immediately so the first frame
        # resolves to real shader output; if the GLArea is not realized yet, the
        # staged preset compiles on the first render once the context is current
        # (load_milk_preset never blocks on GL when opengl_ready is False).
        if (
            self.visualizer is not None
            and getattr(self.visualizer, "available_milk_files", None)
            and not getattr(self.visualizer, "current_milk_file", None)
        ):
            if getattr(self.player.settings, "milkdrop_auto_switch", False):
                import random

                initial_preset = random.choice(self.visualizer.available_milk_files)
            else:
                initial_preset = self.visualizer.available_milk_files[0]
            self.visualizer.load_milk_preset(initial_preset, smooth_transition=False)
            if not self.visualizer.get_realized():

                def _on_realize_for_preset(area):
                    area.disconnect_by_func(_on_realize_for_preset)
                    if hasattr(area, "queue_render"):
                        area.queue_render()

                self.visualizer.connect("realize", _on_realize_for_preset)

    def set_audio_analysis_available(self, available: bool) -> None:
        """Report audio analysis availability and schedule recovery.
        When the controlled F32LE audio sink cannot be created, the
        visualizer retains its renderer widget but the audio analysis
        is inactive. A recovery timer attempts to re-establish the
        audio sink while keeping the visualizer in place.
        When audio becomes available again, the consume timer resumes.
        """
        if available:
            if getattr(self, "_audio_analysis_error_shown", False):
                self._audio_analysis_error_shown = False
                if self._audio_sink_retry_id is not None:
                    self.player._safe_source_remove(self._audio_sink_retry_id)
                    self._audio_sink_retry_id = None
            if getattr(self, "_rendering_enabled", False):
                self._start_audio_consume_timer()
            return
        if self.visualizer is None:
            return
        if not getattr(self, "_audio_analysis_error_shown", False):
            self._audio_analysis_error_shown = True
            self.logger.warning(
                "Visualizer audio analysis unavailable — controlled F32LE sink "
                "could not be created. Retaining renderer and attempting recovery."
            )
            GLib.idle_add(
                self.player.set_status_message,
                "Visualizer audio analysis unavailable — attempting recovery",
            )
            self._schedule_audio_sink_recovery()

    def _on_map(self, widget):
        """Enable rendering when the widget is mapped.
        Sets ``_rendering_enabled`` so the GTK frame-clock render callback
        keeps running, independently of audio availability.
        Audio consumption is also started when analysis is available.
        The one-shot 50 ms delayed-render timer produces the first visible
        frame as soon as the GLArea has a real size.
        """
        if self._map_render_timer_id:
            self.player._safe_source_remove(self._map_render_timer_id)
        self._rendering_enabled = True
        if getattr(self.player, "_visualizer_audio_available", False):
            self._start_audio_consume_timer()
        else:
            self._start_render_loop()
        self._map_render_timer_id = self.player._register_timer(
            50, self._delayed_render_trigger
        )

    def _on_unmap(self, widget):
        """Disable rendering when the widget is unmapped.
        Stops the render loop and audio-consume timers so we do not
        burn CPU / GPU on a hidden visualizer.  The timers are
        re-armed by ``_on_map`` when the tab becomes visible again.
        """
        self._rendering_enabled = False
        self._stop_audio_consume_timer()

    def _delayed_render_trigger(self):
        if self.visualizer is None:
            return False
        w = self.visualizer.get_width()
        h = self.visualizer.get_height()
        if w > 10 and h > 10:
            self.visualizer.queue_render()
        self._map_render_timer_id = 0
        return False

    def cleanup(self):
        """Stop all audio probing and rendering. Safe to call multiple times."""
        self.deactivate()
        if getattr(self, "_audio_sink_retry_id", None) is not None:
            self.player._safe_source_remove(self._audio_sink_retry_id)
            self._audio_sink_retry_id = None
            self._audio_sink_retry_count = 0
        if self._map_render_timer_id:
            self.player._safe_source_remove(self._map_render_timer_id)
            self._map_render_timer_id = 0
        if self._map_handler_id:
            try:
                self.disconnect(self._map_handler_id)
            except (TypeError, GLib.Error):
                pass
            self._map_handler_id = 0
        if self._unmap_handler_id:
            try:
                self.disconnect(self._unmap_handler_id)
            except (TypeError, GLib.Error):
                pass
            self._unmap_handler_id = 0

    def _get_audio_sink_pad(self):
        """Return (pad, sink) using the retained vis_caps source pad.
        Uses the capsfilter source pad captured at build time rather
        than rediscovering it by iterating the bin's pads. This avoids
        the fragility of pad-name matching and ensures the probe is
        always attached to the controlled F32LE source.
        """
        player_obj = getattr(self.player, "player", self.player)
        if player_obj is None or not hasattr(player_obj, "get_property"):
            return None, None
        try:
            audio_sink = player_obj.get_property("audio-sink")
        except Exception:  # noqa: BLE001
            return None, None
        if not audio_sink:
            return None, None
        capsfilter = getattr(self.player, "_vis_caps_filter", None)
        if capsfilter is not None and audio_sink is not None:
            src = capsfilter.get_static_pad("src")
            if src:
                return src, audio_sink
        return None, None

    def _is_audio_bypassed(self):
        """Check if the visualizer audio path is in bypass mode.
        Returns True when the controlled F32LE sink could not be built and
        playback fell back to a plain autoaudiosink — no probe can be
        attached in that case.
        """
        player_obj = getattr(self.player, "player", self.player)
        path = getattr(player_obj, "_visualizer_audio_path", None)
        if path is not None:
            return path == "bypass"
        return getattr(self.player, "_visualizer_audio_path", None) == "bypass"

    def _safe_source_remove(self, timer_id):
        """Delegate timer removal to the player (WinampWindow)."""
        return self.player._safe_source_remove(timer_id)

    def _reconnect_audio_probe(self):
        """Attach or re-attach the probe when the sink/pad changes."""
        if self._is_audio_bypassed():
            if self._probe_retry_id is not None:
                self._safe_source_remove(self._probe_retry_id)
                self._probe_retry_id = None
            return
        if self._probe_connecting:
            return
        sink_pad, audio_sink = self._get_audio_sink_pad()
        if sink_pad is None or audio_sink is None:
            if not getattr(self.player, "_visualizer_audio_available", False):
                return
            if self._probe_retry_id is None:
                self._probe_retry_count = 0
                self._probe_retry_id = self.player._register_timer(
                    250, self._retry_audio_probe
                )
            return
        need_reconnect = (
            self.audio_probe_id is None
            or self._probe_sink is not audio_sink
            or self._probe_pad is not sink_pad
        )
        if not need_reconnect:
            return
        self._probe_connecting = True
        try:
            if self.audio_probe_id is not None and self._probe_pad is not None:
                self._probe_pad.remove_probe(self.audio_probe_id)
            self.audio_probe_id = sink_pad.add_probe(
                Gst.PadProbeType.BUFFER,
                self._milkdrop_audio_probe_callback,
                None,
            )
            self._probe_sink = audio_sink
            self._probe_pad = sink_pad
            if self._probe_retry_id is not None:
                self.player._safe_source_remove(self._probe_retry_id)
                self._probe_retry_id = None
                self._probe_retry_count = 0
            self._audio_sample_rate = float(
                getattr(self.player, "_audio_sample_rate", 44100.0)
            )
            self._start_audio_consume_timer()
        finally:
            self._probe_connecting = False

    def _retry_audio_probe(self):
        """Retry attaching the audio probe with a bounded backoff.
        Bounded to 40 attempts (~10 s) so a persistent sink/pad failure
        cannot run the GTK main loop forever. After the limit, the retry
        transitions to a slow-polling cadence instead of stopping
        outright: if the GStreamer sink topology becomes available later,
        the visualizer can still recover without waiting for an unrelated
        event-driven reconnect attempt.
        """
        if self._is_audio_bypassed():
            if self._probe_retry_id is not None:
                self._safe_source_remove(self._probe_retry_id)
                self._probe_retry_id = None
            return False
        if self.audio_probe_id is not None:
            self._probe_retry_id = None
            self._probe_retry_count = 0
            return False
        self._probe_retry_count += 1
        if self._probe_retry_count >= 40:
            logging.getLogger(__name__).warning(
                "Audio probe retry limit reached after %d attempts; "
                "switching to slow-polling reconnect.",
                self._probe_retry_count,
            )
            self._probe_retry_count = 0
            self._probe_retry_id = self.player._register_timer(
                5000, self._retry_audio_probe
            )
            return False
        self._reconnect_audio_probe()
        if self.audio_probe_id is not None:
            self._probe_retry_id = None
            self._probe_retry_count = 0
            return False
        return True

    def _schedule_audio_sink_recovery(self):
        """Schedule a retry of the controlled F32LE audio sink creation."""
        if self._audio_sink_retry_id is not None:
            return
        self._audio_sink_retry_count = 0
        self._audio_sink_retry_id = self.player._register_timer(
            2000, self._retry_audio_sink_creation
        )

    def _retry_audio_sink_creation(self):
        """Retry creating the controlled F32LE audio sink."""
        self._audio_sink_retry_count += 1
        if self._audio_sink_retry_count > 25:
            self.logger.warning(
                "Visualizer audio sink recovery abandoned after %d attempts",
                self._audio_sink_retry_count,
            )
            GLib.idle_add(
                self.player.set_status_message,
                "Visualizer audio recovery failed — check audio system",
            )
            if self._audio_sink_retry_id is not None:
                self.player._safe_source_remove(self._audio_sink_retry_id)
                self._audio_sink_retry_id = None
            return False
        result = self.player._ensure_audio_sink()
        if result is True:
            if self._audio_sink_retry_id is not None:
                self.player._safe_source_remove(self._audio_sink_retry_id)
                self._audio_sink_retry_id = None
            if not getattr(self.player, "_visualizer_audio_available", False):
                return False
            self._reconnect_audio_probe()
            return False
        return True

    def _milkdrop_audio_probe_callback(self, pad, info, user_data):
        """GStreamer pad probe: capture buffer and schedule analysis on the main loop."""
        if not self._rendering_enabled:
            return Gst.PadProbeReturn.OK
        if not NUMPY_AVAILABLE or np is None:
            return Gst.PadProbeReturn.OK
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            return Gst.PadProbeReturn.OK
        try:
            audio, rate, channels = self._buffer_to_float32(pad, map_info)
            if audio is None or len(audio) == 0:
                return Gst.PadProbeReturn.OK
            audio_array = np.asarray(audio, dtype=np.float32)
            audio_array = np.nan_to_num(audio_array, nan=0.0, posinf=1.0, neginf=-1.0)
            audio_array = np.clip(audio_array, -1.0, 1.0)
            with self._audio_lock:
                self._latest_audio = audio_array
                self._audio_channels = channels
                self._audio_sample_rate = float(rate)
                self._latest_audio_seq = (self._latest_audio_seq + 1) & 0xFFFFFFFF
                self._audio_queue.append((audio_array, "F32LE", rate, channels))
                now = time.monotonic()
                self._audio_dt = now - self._last_audio_time
                self._last_audio_time = now
                self._audio_active = True
                self._last_audio_activity = now
            self._accumulate_rolling_pcm(audio_array, copy=False)
        finally:
            buffer.unmap(map_info)
        return Gst.PadProbeReturn.OK

    def _start_audio_consume_timer(self, *, start_render: bool = True):
        """Start periodic timers for audio consumption and rendering.
        Each timer is started independently so that a missing render
        callback can be restored even when the audio-consume timer is
        already armed (the render timer may have self-cancelled earlier
        because ``_rendering_enabled`` was still False during widget
        map / audio-sink setup).
        """
        if self._audio_consume_timer_id is None:
            self._audio_consume_timer_id = self.player._register_timer(
                33, self._consume_latest_audio
            )
        if start_render and not self._render_loop_is_active():
            self._start_render_loop()

    def _stop_audio_consume_timer(self):
        """Stop the periodic audio-consume timer."""
        if self._audio_consume_timer_id is not None:
            self.player._safe_source_remove(self._audio_consume_timer_id)
            self._audio_consume_timer_id = None
        self._stop_render_loop()

    def _consume_latest_audio(self):
        """Periodic timeout callback: process latest audio with AudioReactivity
        and push processed levels to the renderer. Returns True to keep running."""
        if not self._rendering_enabled or self.visualizer is None:
            self._audio_consume_timer_id = None
            return False
        with self._audio_lock:
            if self._audio_queue:
                chunks = list(self._audio_queue)
                self._audio_queue.clear()
                first = chunks[0]
                if isinstance(first, tuple) and len(first) >= 4:
                    rate = first[2]
                    channels = first[3]
                else:
                    rate = self._audio_sample_rate
                    channels = self._audio_channels or 1
                last = chunks[-1]
                audio = last[0] if isinstance(last, tuple) else last
                self._audio_active = True
                self._last_audio_activity = time.monotonic()
            else:
                seq = getattr(self, "_latest_audio_seq", 0)
                if seq == getattr(self, "_last_consumed_audio_seq", -1):
                    # No new audio since last consume — mark inactive so the
                    # adaptive cadence can drop the render rate.
                    self._audio_active = False
                    return True
                audio = getattr(self, "_latest_audio", None)
                if audio is None:
                    self._audio_active = False
                    return True
                chunks = [audio]
                rate = self._audio_sample_rate
                channels = self._audio_channels or 1
                self._last_consumed_audio_seq = seq
                self._audio_active = True
                self._last_audio_activity = time.monotonic()

        # Downmix to mono for spectrum analysis
        x = np.asarray(audio, dtype=np.float32)
        if x.size and channels > 1:
            frames = x.size // channels
            if frames >= 1:
                x = x[: frames * channels].reshape(frames, channels).mean(axis=1)

        # Get rolling PCM buffer for FFT (4096 samples matching AudioReactivity FFT size)
        pcm_for_fft = self._accumulate_rolling_pcm(x, copy=True)
        if pcm_for_fft.size < 2:
            return True

        # Compute magnitude spectrum with 4096-point FFT (matching AudioReactivity)
        fft_size = 4096
        if pcm_for_fft.size >= fft_size:
            windowed = pcm_for_fft[-fft_size:] * np.hanning(fft_size).astype(np.float32)
        else:
            windowed = np.zeros(fft_size, dtype=np.float32)
            windowed[-pcm_for_fft.size:] = pcm_for_fft * np.hanning(pcm_for_fft.size).astype(np.float32)
        spectrum = np.abs(np.fft.rfft(windowed)).astype(np.float32)

        # Process with AudioReactivity
        levels = self._audio_analyzer.process_frame(
            spectrum=spectrum,
            pcm=audio,
            sample_rate=int(rate),
        )

        # Push processed levels to renderer
        if self.visualizer is not None and self._rendering_enabled:
            self.visualizer.update_audio(levels)

        # Store for snapshot buffer
        self._level_snapshot_buffer.append(dict(levels))
        return True

    def _accumulate_rolling_pcm(self, chunk, *, copy: bool = True):
        """Shift old samples out and fill the rolling PCM window with new data.
        Returns a *copy* of the current window contents (length up to
        ``_rolling_pcm_target_size``).  When ``chunk`` is empty the
        previous window contents are returned without modification.

        ``copy=False`` skips the copy and is used by the pad-probe hot path,
        which only mutates the window and discards the result -- copying up
        to 16 KB per audio buffer was pure overhead there.
        """
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.size == 0:
            view = self._rolling_pcm_buffer[: self._rolling_pcm_size]
            return view.copy() if copy else view
        target = self._rolling_pcm_target_size
        buf = self._rolling_pcm_buffer
        n = chunk.size
        if n >= target:
            buf[:target] = chunk[-target:]
            self._rolling_pcm_size = target
            view = buf[:target]
            return view.copy() if copy else view
        if self._rolling_pcm_size + n > target:
            shift = self._rolling_pcm_size + n - target
            buf[:-shift] = buf[shift:]
            self._rolling_pcm_size -= shift
        start = self._rolling_pcm_size
        end = start + n
        buf[start:end] = chunk
        self._rolling_pcm_size = end
        view = buf[: self._rolling_pcm_size]
        return view.copy() if copy else view

    def _start_render_loop(self):
        """Drive redraws from GTK's frame clock when the widget supports it."""
        if self._render_loop_is_active():
            return
        visualizer = getattr(self, "visualizer", None)
        add_tick_callback = getattr(type(visualizer), "add_tick_callback", None)
        if visualizer is not None and callable(add_tick_callback):
            self._render_tick_widget = visualizer
            self._render_tick_callback_id = visualizer.add_tick_callback(
                self._on_render_tick
            )
            return
        self._render_timer_id = self.player._register_timer(16, self._render_loop)

    def _render_loop_is_active(self) -> bool:
        return (
            getattr(self, "_render_timer_id", None) is not None
            or getattr(self, "_render_tick_callback_id", None) is not None
        )

    def _stop_render_loop(self):
        tick_id = getattr(self, "_render_tick_callback_id", None)
        tick_widget = getattr(self, "_render_tick_widget", None)
        if tick_id is not None:
            if tick_widget is not None:
                tick_widget.remove_tick_callback(tick_id)
            self._render_tick_callback_id = None
            self._render_tick_widget = None
        render_timer_id = getattr(self, "_render_timer_id", None)
        if render_timer_id is not None:
            self._safe_source_remove(render_timer_id)
            self._render_timer_id = None

    def _on_render_tick(self, widget, frame_clock, user_data=None):
        """Request one GLArea frame on each GTK frame-clock tick.

        The tick callback is the *only* driver of the render loop.  It does
        not call ``queue_render`` directly — that would create a render
        storm (``on_render`` previously re-queued itself, so a single tick
        could cascade into dozens of renders before the next clock pulse).
        Instead it delegates to :meth:`_render_loop`, which decides whether
        a frame is warranted and requests exactly one ``queue_render``.

        Errors in the render path must never kill the tick callback: the
        frame-clock will keep calling us, and the renderer's own recovery
        machinery handles transient GL failures.  Any exception here is
        logged and the callback keeps running so the visualizer can come
        back once the underlying issue resolves.
        """
        try:
            keep_running = self._render_loop()
        except Exception:
            self.logger.exception("_render_loop raised; keeping tick alive")
            keep_running = True
        if not keep_running:
            self._render_tick_callback_id = None
            self._render_tick_widget = None
        return keep_running

    def _render_loop(self):
        """Render-loop driver (frame-clock tick callback).

        Returns ``True`` to keep the tick alive, ``False`` to stop.

        Gates:
        - ``_deactivated`` / ``_rendering_enabled``: widget lifecycle.
        - ``_is_widget_visible``: GTK visibility — never render off-screen.
        - ``_should_render_now``: adaptive cadence — drops to a low rate
          when paused, stopped, or when the audio source is silent so the
          visualizer does not burn CPU/GPU while no music is playing.
        """
        if self._deactivated:
            self._render_timer_id = None
            return False
        if not self._rendering_enabled or self.visualizer is None:
            self._render_timer_id = None
            return False
        if not self._is_widget_visible():
            return False
        if not self._should_render_now():
            return True
        try:
            w = self.visualizer.get_width()
            h = self.visualizer.get_height()
            if w > 0 and h > 0:
                self.visualizer.queue_render()
        except Exception:
            self.logger.exception("_render_loop failed, stopping timer")
            self._render_timer_id = None
            return False
        return True

    def _is_widget_visible(self) -> bool:
        """Return True when the visualizer is on a mapped, visible tab.

        The GTK ``map`` / ``unmap`` signals set ``_rendering_enabled`` and
        toggle the render loop, but the stack ``notify::visible-child``
        handler only flips *activity* (audio probe).  Between the moment
        the tab is switched away and the widget's ``unmap`` fires, the
        GLArea can still receive ``queue_render`` requests and burn cycles
        rendering into a hidden surface.  Checking the actual mapped
        visibility here closes that gap without waiting for ``unmap``.
        """
        visualizer = getattr(self, "visualizer", None)
        if visualizer is None:
            return False
        try:
            return bool(visualizer.get_mapped() and visualizer.get_visible())
        except Exception:  # noqa: BLE001
            return True

    def _should_render_now(self) -> bool:
        """Adaptive render cadence.

        Returns True when a full render is warranted this tick, False to
        skip and wait for the next frame-clock pulse.

        - Always render when audio is flowing (``_audio_active``).
        - When paused/stopped, drop to a 4 FPS idle cadence so the
          visualizer still shows its last frame (feedback decay, evolving
          textures) without pegging a CPU core.
        - When the audio source is silent for an extended period, drop to
          1 FPS so even a still visualizer costs almost nothing.
        """
        if getattr(self, "_audio_active", False):
            return True
        now = time.monotonic()
        last = getattr(self, "_last_audio_activity", 0.0)
        idle_for = now - last if last else 0.0
        if idle_for > 30.0:
            # Long silence — render at most once per second.
            if now - getattr(self, "_last_idle_render", 0.0) < 1.0:
                return False
            self._last_idle_render = now
            return True
        # Brief silence (paused between tracks, etc.) — 4 FPS.
        if now - getattr(self, "_last_idle_render", 0.0) < 0.25:
            return False
        self._last_idle_render = now
        return True

    def _attach_audio_probe_watch(self):
        """Attach the GStreamer bus/signal watches for audio sink recovery."""
        self._connect_audio_sink_watch()

    def _detach_audio_probe_watch(self):
        """Detach the audio-sink signal/bus watch."""
        if self._audio_sink_watch_id is not None:
            self._safe_source_remove(self._audio_sink_watch_id)
            self._audio_sink_watch_id = None
        self._audio_sink_watch_bin = None

    def _disconnect_probe_caps_watch(self):
        """Disconnect the probe caps watch."""
        if self._probe_caps_watch_id is not None:
            self._safe_source_remove(self._probe_caps_watch_id)
            self._probe_caps_watch_id = None

    def _connect_audio_sink_watch(self):
        """Connect to the audio sink's bus message watch.
        This is a placeholder — the existing ``_reconnect_audio_probe``
        flow handles pad attachment.  Subclasses or callers may override.
        """

    def activate(self):
        """Activate the visualizer: start render loop and probe watch."""
        self._deactivated = False
        self._rendering_enabled = True
        try:
            self._start_render_loop()
        except Exception:  # noqa: BLE001, S110
            pass
        try:
            self._start_audio_consume_timer(start_render=False)
        except Exception:  # noqa: BLE001, S110
            pass
        self._attach_audio_probe_watch()
        return True

    def deactivate(self):
        """Deactivate the visualizer: stop timers, detach probe, clear state."""
        self._deactivated = True
        self._rendering_enabled = False
        self._stop_render_loop()
        if self._audio_consume_timer_id is not None:
            self.player._safe_source_remove(self._audio_consume_timer_id)
            self._audio_consume_timer_id = None
        if self._probe_retry_id is not None:
            self.player._safe_source_remove(self._probe_retry_id)
            self._probe_retry_id = None
        if self._probe_pad is not None and self.audio_probe_id is not None:
            try:
                self._probe_pad.remove_probe(self.audio_probe_id)
            except Exception:  # noqa: BLE001, S110
                pass
        self.audio_probe_id = None
        self._probe_sink = None
        self._probe_pad = None
        self._probe_connecting = False
        with self._audio_lock:
            self._audio_queue.clear()
            self._rolling_pcm_size = 0
        self._disconnect_probe_caps_watch()
        self._detach_audio_probe_watch()

    def _buffer_to_float32(self, pad, map_info):
        """Convert a mapped GStreamer buffer to normalized float32 samples.
        Supported formats: F32LE, F64LE, S16LE, S24LE, S24_32LE, S32LE.
        Returns (audio, rate, channels) or (None, rate, channels) on failure.
        """
        rate = float(getattr(self, "_audio_sample_rate", 44100.0))
        channels = max(1, int(getattr(self, "_audio_channels", 2) or 2))
        fmt = None
        if pad is not None:
            caps = pad.get_current_caps()
            if caps is not None and caps.get_size() > 0:
                structure = caps.get_structure(0)
                if structure is not None:
                    fmt = structure.get_string("format")
                    ok, ch = structure.get_int("channels")
                    if ok and ch > 0:
                        channels = ch
                    ok, r = structure.get_int("rate")
                    if ok and r > 0:
                        rate = float(r)
        if fmt is None:
            fmt = "F32LE"
        buf = memoryview(map_info.data)
        try:
            if fmt == "F32LE":
                usable = (len(buf) // 4) * 4
                audio = np.frombuffer(buf[:usable], dtype=np.float32).copy()
            elif fmt == "F64LE":
                usable = (len(buf) // 8) * 8
                audio = (
                    np.frombuffer(buf[:usable], dtype=np.float64)
                    .astype(np.float32)
                    .copy()
                )
            elif fmt == "S16LE":
                usable = (len(buf) // 2) * 2
                audio = (
                    np.frombuffer(buf[:usable], dtype=np.int16).astype(np.float32)
                    / 32768.0
                ).copy()
            elif fmt == "S32LE":
                usable = (len(buf) // 4) * 4
                audio = (
                    np.frombuffer(buf[:usable], dtype=np.int32).astype(np.float32)
                    / 2147483648.0
                ).copy()
            elif fmt == "S24_32LE":
                usable = (len(buf) // 4) * 4
                ints = np.frombuffer(buf[:usable], dtype=np.int32)
                audio = ((ints >> 8).astype(np.float32) / 8388608.0).copy()
            elif fmt == "S24LE":
                sample_count = len(buf) // 3
                if sample_count == 0:
                    return None, rate, channels
                raw = np.frombuffer(buf[: sample_count * 3], dtype=np.uint8).reshape(
                    sample_count, 3
                )
                ints = (
                    raw[:, 0].astype(np.int32)
                    | (raw[:, 1].astype(np.int32) << 8)
                    | (raw[:, 2].astype(np.int32) << 16)
                )
                ints = np.where(ints & 0x800000, ints - 0x1000000, ints)
                audio = (ints.astype(np.float32) / 8388608.0).copy()
            else:
                return None, rate, channels
        except (ValueError, BufferError):
            return None, rate, channels
        if channels > 1:
            remainder = audio.size % channels
            if remainder:
                audio = audio[:-remainder]
        return audio, rate, channels
