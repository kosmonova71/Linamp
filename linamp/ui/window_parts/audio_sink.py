"""WinampWindow mixin: Equalizer setup and the audio sink lifecycle."""

from __future__ import annotations

import gi

from .base import WindowBase

gi.require_version("Gdk", "4.0")
gi.require_version("Gtk", "4.0")
gi.require_version("Gst", "1.0")
import logging
import math

from gi.repository import GLib, Gst


class AudioSinkMixin(WindowBase):
    """Equalizer setup and the audio sink lifecycle.

    The methods are split out of ``WinampWindow`` so the window class
    stays readable; they keep operating on the window instance and
    share its playback, playlist and settings state.
    """

    @staticmethod
    def _clamp_eq_band(value) -> float:
        """Clamp an equalizer band gain to the +/-12 dB range the UI exposes."""
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        if math.isnan(number):
            return 0.0
        return max(-12.0, min(12.0, number))

    def _setup_equalizer(self):
        previous = getattr(self, "equalizer", None)
        if previous is not None:
            try:
                self.player.set_property("audio-filter", None)
            except (TypeError, GLib.Error, AttributeError):
                pass
            try:
                previous.set_state(Gst.State.NULL)
            except (AttributeError, GLib.Error):
                pass
            self.equalizer = None
        self.equalizer = Gst.ElementFactory.make("equalizer-10bands", "equalizer")
        if not self.equalizer:
            return False
        self.player.set_property("audio-filter", self.equalizer)
        for i in range(10):
            self.equalizer.set_property(f"band{i}", 0.0)
        return True

    def _build_visualizer_audio_sink(self):
        """Build an audio-sink bin that delivers a known format to the visualizer.
        The visualizer's pad-probe reads buffer bytes as ``np.float32`` and runs
        an FFT to derive bass/mid/treb bands. That assumption only holds if the
        sink actually emits float32 stereo at a known sample rate. A bare
        ``autoaudiosink`` negotiates whatever the system sink wants (often S16LE
        integer PCM), so the float32 reinterpretation produces garbage and the
        visuals stop tracking the music.
        Force ``F32LE, 2ch, 44100 Hz`` so the probe's interpretation is correct.
        Returns ``None`` if any element is unavailable or linking fails.
        """
        convert = Gst.ElementFactory.make("audioconvert", "vis_convert")
        resample = Gst.ElementFactory.make("audioresample", "vis_resample")
        capsfilter = Gst.ElementFactory.make("capsfilter", "vis_caps")
        sink = Gst.ElementFactory.make("autoaudiosink", "vis_sink")
        if not (convert and resample and capsfilter and sink):
            missing = []
            if not convert:
                missing.append("audioconvert")
            if not resample:
                missing.append("audioresample")
            if not capsfilter:
                missing.append("capsfilter")
            if not sink:
                missing.append("autoaudiosink")
            logging.getLogger(__name__).error(
                "Visualizer audio sink: failed to create elements: %s",
                ", ".join(missing),
            )
            return None
        caps = Gst.Caps.from_string(
            "audio/x-raw, format=F32LE, rate=44100, channels=2, layout=interleaved"
        )
        capsfilter.set_property("caps", caps)
        audio_bin = Gst.Bin.new("vis_audio_sink")
        for element in (convert, resample, capsfilter, sink):
            audio_bin.add(element)
        try:
            link_results = (
                convert.link(resample),
                resample.link(capsfilter),
                capsfilter.link(sink),
            )
        except (RuntimeError, TypeError, GLib.Error):
            for element in (convert, resample, capsfilter, sink):
                try:
                    element.set_state(Gst.State.NULL)
                except (AttributeError, GLib.Error):
                    pass
                try:
                    audio_bin.remove(element)
                except (TypeError, GLib.Error):
                    pass
            try:
                audio_bin.set_state(Gst.State.NULL)
            except (AttributeError, GLib.Error):
                pass
            return None
        if any(not result for result in link_results):
            link_names = [
                "convert->resample",
                "resample->capsfilter",
                "capsfilter->sink",
            ]
            failed = [name for name, ok in zip(link_names, link_results) if not ok]
            logging.getLogger(__name__).error(
                "Visualizer audio sink: link failed for: %s",
                ", ".join(failed),
            )
            for element in (convert, resample, capsfilter, sink):
                try:
                    element.set_state(Gst.State.NULL)
                except (AttributeError, GLib.Error):
                    pass
                try:
                    audio_bin.remove(element)
                except (TypeError, GLib.Error):
                    pass
            try:
                audio_bin.set_state(Gst.State.NULL)
            except (AttributeError, GLib.Error):
                pass
            return None
        sink_pad = convert.get_static_pad("sink")
        if sink_pad is None:
            return None
        audio_bin.add_pad(Gst.GhostPad.new("sink", sink_pad))
        self._audio_sample_rate = 44100.0
        self._audio_channels = 2
        self._vis_caps_filter = capsfilter
        return audio_bin

    def _build_plain_audio_sink(self):
        """Build a plain autoaudiosink bin without format constraints.
        Used as a fallback when the controlled F32LE visualizer sink cannot
        be created or installed. Playback continues normally; only
        visualization is unavailable.
        """
        convert = Gst.ElementFactory.make("audioconvert", "plain_convert")
        resample = Gst.ElementFactory.make("audioresample", "plain_resample")
        sink = Gst.ElementFactory.make("autoaudiosink", "plain_sink")
        if not (convert and resample and sink):
            missing = []
            if not convert:
                missing.append("audioconvert")
            if not resample:
                missing.append("audioresample")
            if not sink:
                missing.append("autoaudiosink")
            logging.getLogger(__name__).error(
                "Plain audio sink: failed to create elements: %s",
                ", ".join(missing),
            )
            return None
        audio_bin = Gst.Bin.new("plain_audio_sink")
        for element in (convert, resample, sink):
            audio_bin.add(element)
        try:
            link_results = (
                convert.link(resample),
                resample.link(sink),
            )
        except (RuntimeError, TypeError, GLib.Error):
            for element in (convert, resample, sink):
                try:
                    element.set_state(Gst.State.NULL)
                except (AttributeError, GLib.Error):
                    pass
                try:
                    audio_bin.remove(element)
                except (TypeError, GLib.Error):
                    pass
            try:
                audio_bin.set_state(Gst.State.NULL)
            except (AttributeError, GLib.Error):
                pass
            return None
        if any(not result for result in link_results):
            failed = [
                name
                for name, ok in zip(
                    ["convert->resample", "resample->sink"], link_results
                )
                if not ok
            ]
            logging.getLogger(__name__).error(
                "Plain audio sink: link failed for: %s",
                ", ".join(failed),
            )
            for element in (convert, resample, sink):
                try:
                    element.set_state(Gst.State.NULL)
                except (AttributeError, GLib.Error):
                    pass
                try:
                    audio_bin.remove(element)
                except (TypeError, GLib.Error):
                    pass
            try:
                audio_bin.set_state(Gst.State.NULL)
            except (AttributeError, GLib.Error):
                pass
            return None
        sink_pad = convert.get_static_pad("sink")
        if sink_pad is None:
            return None
        audio_bin.add_pad(Gst.GhostPad.new("sink", sink_pad))
        return audio_bin

    def _release_audio_sink(self, old_sink=None):
        with self._sink_lock:
            return self._release_audio_sink_locked(old_sink)

    def _release_audio_sink_locked(self, old_sink=None):
        if old_sink is not None:
            try:
                old_sink.set_state(Gst.State.NULL)
            except (AttributeError, GLib.Error):
                pass
            return
        if getattr(self, "player", None) is not None:
            try:
                self.player.set_property("audio-sink", None)
                self.player.set_property("audio-filter", None)
            except (TypeError, GLib.Error):
                pass
        elem = getattr(self, "audio_sink", None)
        if elem is not None:
            try:
                elem.set_state(Gst.State.NULL)
            except (AttributeError, GLib.Error):
                pass
            self.audio_sink = None
        self._vis_caps_filter = None
        equalizer = getattr(self, "equalizer", None)
        if equalizer is not None:
            try:
                equalizer.set_state(Gst.State.NULL)
            except (AttributeError, GLib.Error):
                pass
            self.equalizer = None

    def _ensure_audio_sink(self) -> bool | None:
        with self._sink_lock:
            return self._ensure_audio_sink_locked()

    def _ensure_audio_sink_locked(self) -> bool | None:
        """Ensure an audio sink is installed for playback.
        First tries to build the controlled F32LE visualizer audio sink.
        If that fails (missing elements, link failure, etc.), falls back
        to a plain autoaudiosink so playback continues — visualization
        is simply unavailable.  Only the F32LE controlled sink triggers
        visualizer analysis; playback never depends on it.
        Returns True on success (playback always continues).
        Returns None if the player is not available.
        """
        if not getattr(self, "player", None):
            return None
        existing_sink = getattr(self, "audio_sink", None)
        if existing_sink is not None:
            try:
                if existing_sink.get_state(0)[1] != Gst.State.NULL:
                    try:
                        existing_sink.set_state(Gst.State.NULL)
                    except (AttributeError, GLib.Error):
                        pass
                    return True
            except (AttributeError, GLib.Error):
                pass
        audio_sink = self._build_visualizer_audio_sink()
        is_visualizer_sink = audio_sink is not None
        if audio_sink is not None:
            try:
                self.player.set_property("audio-sink", audio_sink)
            except (TypeError, GLib.Error):
                try:
                    audio_sink.set_state(Gst.State.NULL)
                except (AttributeError, GLib.Error):
                    pass
                logging.getLogger(__name__).error("Failed to set visualizer audio sink")
                audio_sink = None
                self._vis_caps_filter = None
                is_visualizer_sink = False
        if audio_sink is None:
            logging.getLogger(__name__).info(
                "Falling back to plain autoaudiosink — visualizer unavailable"
            )
            audio_sink = self._build_plain_audio_sink()
            if audio_sink is None:
                logging.getLogger(__name__).error("Could not create any audio sink")
                return False
            try:
                self.player.set_property("audio-sink", audio_sink)
            except (TypeError, GLib.Error):
                try:
                    audio_sink.set_state(Gst.State.NULL)
                except (AttributeError, GLib.Error):
                    pass
                logging.getLogger(__name__).error("Failed to set plain audio sink")
                return False
            self._vis_caps_filter = None
            is_visualizer_sink = False
        if existing_sink is not None:
            try:
                existing_sink.set_state(Gst.State.NULL)
            except (AttributeError, GLib.Error):
                pass
        self.audio_sink = audio_sink
        self._visualizer_audio_available = (
            is_visualizer_sink and self._vis_caps_filter is not None
        )
        self._visualizer_audio_path = "controlled" if is_visualizer_sink else "bypass"
        try:
            eq_success = self._setup_equalizer()
        except Exception:
            logging.getLogger(__name__).exception(
                "Unexpected error during equalizer setup — continuing without EQ"
            )
            eq_success = False
        self._equalizer_available = eq_success
        if eq_success and self.settings.equalizer_settings and self.equalizer is not None:
            for i, value in enumerate(self.settings.equalizer_settings[:10]):
                self.equalizer.set_property(f"band{i}", self._clamp_eq_band(value))
        if not eq_success:
            logging.getLogger(__name__).warning(
                "Equalizer unavailable — sink installed but EQ not active"
            )
        if (
            self._visualizer_audio_available
            and hasattr(self, "visualizer_tab")
            and self.visualizer_tab is not None
            and hasattr(self.visualizer_tab, "set_audio_analysis_available")
        ):
            self.visualizer_tab.set_audio_analysis_available(True)
        elif (
            not self._visualizer_audio_available
            and hasattr(self, "visualizer_tab")
            and self.visualizer_tab is not None
            and hasattr(self.visualizer_tab, "set_audio_analysis_available")
        ):
            self.visualizer_tab.set_audio_analysis_available(False)
        return True
