"""Audio reactivity policy.

This module owns the *analysis* half of audio reactivity: turning a raw
spectrum/PCM frame into the smoothed band levels, attack responses, and beat
signal that the renderer uploads as GPU uniforms.  It deliberately does not
know anything about the GPU, the enhancement layer, or the preset VM.

The prior implementation mixed FFT band energy, autogain, envelope following,
attack-smoothed bands, spectral-flux beat detection, and spectrum
down-sampling in one function.  Those steps are independent with distinct
tunable parameters, so they belong here, in separate, documented,
unit-testable code.  The output contract is a plain dict with the keys the
renderer consumes:

``bass`` / ``bass_att`` / ``mid`` / ``mid_att`` / ``treb`` / ``treb_att`` /
``beat`` / ``beat_flash`` / ``rms`` / ``peak`` / ``_spectrum`` / ``_timestamp``

Auto-gain, enhancement-layer responses, brightness, feedback, and bloom stay
in the renderer / enhancement layer.
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Any

import numpy as np

FFT_SIZE: int = 4096
LOG_FLOOR: float = 1e-10
LOG_RANGE: float = 8.0


def _to_float_array(x: Any) -> np.ndarray:
    """Convert input to float array."""
    if x is None:
        return np.array([], dtype=np.float32)
    arr = np.asarray(x)
    if arr.size == 0:
        return np.array([], dtype=np.float32)
    # Handle integer types (e.g., int16 PCM)
    if arr.dtype == np.int16:
        arr = arr.astype(np.float32) / 32768.0
        return arr
    arr = arr.astype(np.float32)
    return arr


def _safe_log10(x: float) -> float:
    if x <= 0.0:
        return math.log10(LOG_FLOOR)
    return math.log10(x)


def _safe_log10_array(x: np.ndarray) -> np.ndarray:
    """Vectorized safe log10."""
    x = np.maximum(x, LOG_FLOOR)
    return np.log10(x).astype(np.float32)


class AudioReactivity:
    """Stateful audio analysis producing renderer-consumable levels."""

    def __init__(
        self,
        sample_rate: int = 44100,
        fft_size: int = FFT_SIZE,
        band_edges: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
            (20.0, 250.0),
            (250.0, 2000.0),
            (2000.0, 16000.0),
        ),
        attack_times: tuple[float, float, float] = (0.05, 0.05, 0.05),
        release_times: tuple[float, float, float] = (0.3, 0.3, 0.3),
        attack_times_att: tuple[float, float, float] = (0.005, 0.005, 0.005),
        release_times_att: tuple[float, float, float] = (0.15, 0.15, 0.15),
        spectrum_bins: int = 128,
        log_spectrum: bool = True,
        beat_sensitivity: float = 1.3,
        beat_history: int = 43,
        beat_min_interval: float = 0.15,
        beat_decay: float = 0.82,
        bass_boost: float = 1.15,
        spectrum_smooth: float = 0.85,
        envelope_mode: str = "exponential",
        peak_hold: float = 0.0,
        peak_decay: float = 0.99,
        sub_harmonic_balance: float = 0.0,
        vocal_cut: float = 0.0,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.fft_size = int(fft_size)
        self.band_edges = band_edges
        self.attack_times = attack_times
        self.release_times = release_times
        self.attack_times_att = attack_times_att
        self.release_times_att = release_times_att
        self.spectrum_bins = int(spectrum_bins)
        self.log_spectrum = bool(log_spectrum)
        self.beat_sensitivity = float(beat_sensitivity)
        self.beat_history = int(max(1, beat_history))
        self.beat_min_interval = float(beat_min_interval)
        self.beat_decay = float(beat_decay)
        self.bass_boost = float(bass_boost)
        self.spectrum_smooth = float(max(0.0, min(1.0, spectrum_smooth)))
        self.peak_hold = float(peak_hold)
        self.envelope_mode = envelope_mode

        # Validate envelope mode
        if self.envelope_mode not in ["exponential", "linear", "logarithmic", "root", "db"]:
            self.envelope_mode = "exponential"

        # Envelopes for raw band energies
        self._bass_env = 0.0
        self._mid_env = 0.0
        self._treb_env = 0.0
        # Attack/release envelopes (att)
        self._bass_att_env = 0.0
        self._mid_att_env = 0.0
        self._treb_att_env = 0.0

        # Beat detection state
        self._flux_history: deque[float] = deque(maxlen=self.beat_history)
        self._last_beat_time: float = -1e18
        self._beat_flash = 0.0
        self._prev_spectrum_mag: np.ndarray | None = None

        # Timing
        self._last_time: float | None = None

        # Frequency caching
        self._cached_freqs: np.ndarray | None = None
        self._cached_freqs_sr: int | None = None
        self._cached_freqs_fft: int | None = None

        # Band masks cache
        self._band_masks: list[np.ndarray] | None = None
        self._band_masks_sr: int | None = None
        self._band_masks_fft: int | None = None
        self._band_masks_n: int | None = None

        self._last_flux: float = 0.0
        # Peak hold state
        self._peak_hold_bass = 0.0
        self._peak_hold_mid = 0.0
        self._peak_hold_treb = 0.0
        self._peak_hold_att_bass = 0.0
        self._peak_hold_att_mid = 0.0
        self._peak_hold_att_treb = 0.0
        # Spectrum downsampling state
        self._spectrum_peak_smooth = 1.0
        self._spectrum_prev_out = None

    def _get_band_masks(self, n: int) -> list[np.ndarray]:
        if (
            self._band_masks is not None
            and self._band_masks_sr == self.sample_rate
            and self._band_masks_fft == self.fft_size
            and self._band_masks_n == n
        ):
            return self._band_masks
        freqs = self._get_freqs(n)
        masks = []
        for low, high in self.band_edges:
            if freqs.size == 0:
                masks.append(np.array([], dtype=bool))
            else:
                masks.append((freqs >= low) & (freqs < high))
        self._band_masks = masks
        self._band_masks_sr = self.sample_rate
        self._band_masks_fft = self.fft_size
        self._band_masks_n = n
        return masks

    def _get_freqs(self, n: int) -> np.ndarray:
        if (
            self._cached_freqs is not None
            and self._cached_freqs_sr == self.sample_rate
            and self._cached_freqs.size == n
        ):
            return self._cached_freqs
        if n <= 1:
            freqs = np.zeros(max(1, n), dtype=np.float32)
        else:
            fft_len = max(self.fft_size, 2 * (n - 1))
            freqs = np.fft.rfftfreq(fft_len, 1.0 / self.sample_rate)
            if freqs.size != n:
                freqs = np.linspace(0.0, self.sample_rate / 2.0, n, dtype=np.float32)
        freqs = np.asarray(freqs, dtype=np.float32)
        self._cached_freqs = freqs
        self._cached_freqs_sr = self.sample_rate
        self._cached_freqs_fft = self.fft_size
        return freqs

    def _band_indices(self, freqs: np.ndarray, low: float, high: float) -> np.ndarray:
        if freqs.size == 0:
            return np.array([], dtype=int)
        mask = (freqs >= low) & (freqs < high)
        return np.where(mask)[0]

    def _compute_band_energies(self, spectrum: np.ndarray) -> tuple[float, float, float]:
        if spectrum is None or spectrum.size == 0:
            return 0.0, 0.0, 0.0
        n = spectrum.size
        masks = self._get_band_masks(n)
        bass_e = 0.0
        mid_e = 0.0
        treb_e = 0.0
        for i, mask in enumerate(masks):
            if mask.size == 0 or not np.any(mask):
                val = 0.0
            else:
                band_mag = spectrum[mask]
                band_pow = band_mag * band_mag
                mean_pow = float(np.mean(band_pow))
                val = math.log10(max(mean_pow, LOG_FLOOR)) / LOG_RANGE
                val = max(0.0, val)
            if i == 0:
                bass_e = val * self.bass_boost
            elif i == 1:
                mid_e = val
            elif i == 2:
                treb_e = val
        bass_e = min(bass_e, 1.5)
        mid_e = min(mid_e, 1.5)
        treb_e = min(treb_e, 1.5)
        bass_e = max(0.0, min(1.0, bass_e))
        mid_e = max(0.0, min(1.0, mid_e))
        treb_e = max(0.0, min(1.0, treb_e))
        return bass_e, mid_e, treb_e

    @staticmethod
    def _envelope(current: float, target: float, tc_attack: float, tc_release: float, dt: float, mode: str = "exponential") -> float:
        if dt <= 0.0 or dt > 0.25:
            dt = 1.0 / 60.0
        if mode == "exponential":
            if target > current:
                alpha = 1.0 - math.exp(-dt / max(tc_attack, 1e-6))
            else:
                alpha = 1.0 - math.exp(-dt / max(tc_release, 1e-6))
            return current + alpha * (target - current)
        elif mode == "linear":
            step = (target - current) * min(dt / max(tc_attack + tc_release, 1e-6), 1.0)
            if target > current:
                return min(target, current + step)
            else:
                return max(target, current - step)
        elif mode == "logarithmic":
            if target > current:
                alpha = 1.0 - math.exp(-dt / max(tc_attack, 1e-6))
                return current + alpha * (target - current) * (1.0 + abs(math.log10(max(target, 1e-10))))
            else:
                alpha = 1.0 - math.exp(-dt / max(tc_release, 1e-6))
                return current * alpha + target * (1.0 - alpha) * (1.0 + abs(math.log10(max(current, 1e-10))))
        elif mode == "root":
            power = 0.5
            if target > current:
                alpha = 1.0 - math.exp(-dt / max(tc_attack, 1e-6))
                diff = target - current
                return current + diff * (1.0 - alpha) * (diff ** power)
            else:
                alpha = 1.0 - math.exp(-dt / max(tc_release, 1e-6))
                diff = current - target
                return current * alpha + target * (1.0 - alpha) * (diff ** power)
        elif mode == "db":
            ref = 1e-5
            target_db = 20.0 * math.log10(max(target, 1e-10) / ref)
            current_db = 20.0 * math.log10(max(current, 1e-10) / ref)
            if target_db > current_db:
                alpha = 1.0 - math.exp(-dt / max(tc_attack, 1e-6))
                return current + (target - current) * (1.0 - alpha)
            else:
                alpha = 1.0 - math.exp(-dt / max(tc_release, 1e-6))
                return current * alpha + target * (1.0 - alpha)
        # Default to exponential
        if target > current:
            alpha = 1.0 - math.exp(-dt / max(tc_attack, 1e-6))
        else:
            alpha = 1.0 - math.exp(-dt / max(tc_release, 1e-6))
        return current + alpha * (target - current)
    def _apply_peak_hold(self, env: float, peak_hold_env: float, dt: float) -> float:
        """Apply peak hold - hold near-peak values for a configurable duration."""
        if self.peak_hold <= 0.0:
            return env

        # Decay the peak hold towards zero
        hold_decay = math.exp(-dt / max(self.peak_hold, 1e-6))
        peak_hold_env *= hold_decay

        # If the current env is within peak hold threshold, keep it
        if env > peak_hold_env * 0.95:  # close to peak
            # Don't decay below the held value too quickly
            return max(env, peak_hold_env)
        return env

    def _compute_beat(self, spectrum: np.ndarray, now: float) -> tuple[float, float]:
        if spectrum is None or spectrum.size == 0:
            self._beat_flash *= self.beat_decay
            return 0.0, max(0.0, self._beat_flash)
        if self._prev_spectrum_mag is None or self._prev_spectrum_mag.shape != spectrum.shape:
            self._prev_spectrum_mag = spectrum.copy()
            self._beat_flash *= self.beat_decay
            return 0.0, max(0.0, self._beat_flash)
        # Spectral flux on log-magnitude (much more level-invariant)
        curr = np.maximum(spectrum, LOG_FLOOR)
        prev = np.maximum(self._prev_spectrum_mag, LOG_FLOOR)
        log_diff = np.log10(curr) - np.log10(prev)
        flux = float(np.sum(log_diff, where=log_diff > 0.0))
        self._last_flux = flux

        self._prev_spectrum_mag = spectrum.copy()
        self._flux_history.append(flux)

        min_history = max(8, self.beat_history // 4)
        if len(self._flux_history) < min_history:
            self._beat_flash *= self.beat_decay
            return 0.0, max(0.0, self._beat_flash)

        flux_arr = np.fromiter(self._flux_history, dtype=np.float32)
        median = float(np.median(flux_arr))
        # MAD is more robust than pure median * sensitivity for outlier frames
        mad = float(np.median(np.abs(flux_arr - median))) + 1e-12
        threshold = median + self.beat_sensitivity * 1.4826 * mad  # ~1σ for normal

        beat = 1.0 if flux > threshold and (now - self._last_beat_time) > self.beat_min_interval else 0.0
        if beat > 0.0:
            self._last_beat_time = now
            self._beat_flash = 1.0
        else:
            self._beat_flash *= self.beat_decay

        return beat, max(0.0, self._beat_flash)

    def _downsample_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        if spectrum is None or spectrum.size == 0:
            return np.zeros(self.spectrum_bins, dtype=np.float32)
        n = spectrum.size
        freqs = self._get_freqs(n)
        if self.log_spectrum and freqs[-1] > freqs[0] and freqs[0] > 0 and n > 1:
            min_f = max(freqs[1] if freqs.size > 1 else freqs[0], 20.0)
            max_f = min(freqs[-1], 16000.0)
            if max_f <= min_f:
                out = np.zeros(self.spectrum_bins, dtype=np.float32)
                step = n / self.spectrum_bins
                for i in range(self.spectrum_bins):
                    s = int(i * step)
                    e = int((i + 1) * step)
                    if e <= s:
                        e = s + 1
                    e = min(e, n)
                    out[i] = float(np.mean(spectrum[s:e]))
            else:
                log_min = math.log10(min_f)
                log_max = math.log10(max_f)
                edges = np.logspace(log_min, log_max, self.spectrum_bins + 1)
                out = np.zeros(self.spectrum_bins, dtype=np.float32)
                # Use searchsorted for efficiency
                idxs = np.searchsorted(freqs, edges)
                for i in range(self.spectrum_bins):
                    s = int(idxs[i])
                    e = int(idxs[i + 1])
                    if e <= s:
                        # fallback to nearest
                        m_idx = min(s, n - 1)
                        out[i] = float(spectrum[m_idx]) if n > 0 else 0.0
                    else:
                        e = min(e, n)
                        out[i] = float(np.mean(spectrum[s:e]))
        else:
            out = np.zeros(self.spectrum_bins, dtype=np.float32)
            step = n / self.spectrum_bins
            for i in range(self.spectrum_bins):
                s = int(i * step)
                e = int((i + 1) * step)
                if e <= s:
                    e = s + 1
                e = min(e, n)
                out[i] = float(np.mean(spectrum[s:e]))
        m = float(np.max(out)) if out.size > 0 else 1.0
        peak_cur = m if m > 1e-12 else 1.0
        if peak_cur > self._spectrum_peak_smooth:
            self._spectrum_peak_smooth = 0.25 * self._spectrum_peak_smooth + 0.75 * peak_cur
        else:
            self._spectrum_peak_smooth = self.spectrum_smooth * self._spectrum_peak_smooth + (1.0 - self.spectrum_smooth) * peak_cur
        if self._spectrum_peak_smooth > 1e-12:
            out = out / self._spectrum_peak_smooth

        # Optional temporal smoothing of the spectrum itself (helps visual stability)
        if self._spectrum_prev_out is not None and self._spectrum_prev_out.shape == out.shape:
            alpha = 0.55  # higher = more responsive
            out = alpha * out + (1.0 - alpha) * self._spectrum_prev_out
        self._spectrum_prev_out = out.copy()

        out = np.clip(out, 0.0, 1.0)
        return out.astype(np.float32)

    def reset(self) -> None:
        self._bass_env = 0.0
        self._mid_env = 0.0
        self._treb_env = 0.0
        self._bass_att_env = 0.0
        self._mid_att_env = 0.0
        self._treb_att_env = 0.0
        self._flux_history.clear()
        self._last_beat_time = -1e18
        self._beat_flash = 0.0
        self._prev_spectrum_mag = None
        self._last_time = None
        self._spectrum_peak_smooth = 1.0
        self._spectrum_prev_out = None
        self._last_flux = 0.0

    @property
    def last_flux(self) -> float:
        return self._last_flux

    @property
    def bass_env(self) -> float:
        return self._bass_env

    @property
    def mid_env(self) -> float:
        return self._mid_env

    @property
    def treb_env(self) -> float:
        return self._treb_env

    @property
    def bass_att_env(self) -> float:
        return self._bass_att_env

    @property
    def mid_att_env(self) -> float:
        return self._mid_att_env

    @property
    def treb_att_env(self) -> float:
        return self._treb_att_env

    def process_frame(
        self,
        spectrum: Any = None,
        pcm: Any = None,
        sample_rate: int | None = None,
        timestamp: float | None = None,
    ) -> dict[str, Any]:
        """Process one frame and return audio levels dict."""
        now = time.monotonic() if timestamp is None else float(timestamp)
        if sample_rate is not None and self.sample_rate != int(sample_rate):
            self.sample_rate = int(sample_rate)
            # Invalidate frequency cache if sample rate changes
            self._cached_freqs = None
            self._cached_freqs_sr = None
            self._cached_freqs_fft = None
            self._band_masks = None
            self._band_masks_sr = None
            self._band_masks_fft = None
            self._band_masks_n = None

        if self._last_time is None:
            dt = 1.0 / 60.0
        else:
            dt = now - self._last_time
        if dt < 0.0:
            dt = 1.0 / 60.0
        if dt > 0.5:
            dt = 1.0 / 60.0
        self._last_time = now

        spec_arr = _to_float_array(spectrum)
        # If spectrum is magnitude-like, use as-is; if complex, take abs
        if spec_arr.size > 0 and np.iscomplexobj(spec_arr):
            spec_arr = np.abs(spec_arr).astype(np.float32)

        bass_e, mid_e, treb_e = self._compute_band_energies(spec_arr)
        # Smooth envelopes
        self._bass_env = self._envelope(self._bass_env, bass_e, self.attack_times[0], self.release_times[0], dt, self.envelope_mode)
        self._mid_env = self._envelope(self._mid_env, mid_e, self.attack_times[1], self.release_times[1], dt, self.envelope_mode)
        self._treb_env = self._envelope(self._treb_env, treb_e, self.attack_times[2], self.release_times[2], dt, self.envelope_mode)

        # Attack envelopes (separate fast followers)
        self._bass_att_env = self._envelope(self._bass_att_env, bass_e, self.attack_times_att[0], self.release_times_att[0], dt, self.envelope_mode)
        self._mid_att_env = self._envelope(self._mid_att_env, mid_e, self.attack_times_att[1], self.release_times_att[1], dt, self.envelope_mode)
        self._treb_att_env = self._envelope(self._treb_att_env, treb_e, self.attack_times_att[2], self.release_times_att[2], dt, self.envelope_mode)

        beat, beat_flash = self._compute_beat(spec_arr, now)

        # RMS / peak from PCM if provided
        rms = 0.0
        peak = 0.0
        pcm_arr = _to_float_array(pcm)
        if pcm_arr.size > 0:
            if np.iscomplexobj(pcm_arr):
                pcm_arr = np.real(pcm_arr)
            pcm_arr_float = pcm_arr.astype(np.float32)
            rms = float(np.sqrt(np.mean(pcm_arr_float ** 2)))
            peak = float(np.max(np.abs(pcm_arr_float)))
            if max(abs(rms), abs(peak)) > 1.5:
                rms = min(1.0, rms / 32768.0)
                peak = min(1.0, peak / 32768.0)

        spectrum_out = self._downsample_spectrum(spec_arr)

# Apply peak hold to main and attack envelopes\n        if self.peak_hold > 0.0:\n            self._bass_env = self._apply_peak_hold(self._bass_env, self._peak_hold_bass, dt)\n            self._mid_env = self._apply_peak_hold(self._mid_env, self._peak_hold_mid, dt)\n            self._treb_env = self._apply_peak_hold(self._treb_env, self._peak_hold_treb, dt)\n            self._bass_att_env = self._apply_peak_hold(self._bass_att_env, self._peak_hold_att_bass, dt)\n            self._mid_att_env = self._apply_peak_hold(self._mid_att_env, self._peak_hold_att_mid, dt)\n            self._treb_att_env = self._apply_peak_hold(self._treb_att_env, self._peak_hold_att_treb, dt)\n\n
        return {
            "bass": float(max(0.0, min(1.0, self._bass_env))),
            "bass_att": float(max(0.0, min(1.0, self._bass_att_env))),
            "mid": float(max(0.0, min(1.0, self._mid_env))),
            "mid_att": float(max(0.0, min(1.0, self._mid_att_env))),
            "treb": float(max(0.0, min(1.0, self._treb_env))),
            "treb_att": float(max(0.0, min(1.0, self._treb_att_env))),
            "beat": float(max(0.0, min(1.0, beat))),
            "beat_flash": float(max(0.0, min(1.0, beat_flash))),
            "rms": float(max(0.0, min(1.0, rms))),
            "peak": float(max(0.0, min(1.0, peak))),
            "_spectrum": spectrum_out,
            "_timestamp": float(now),
        }


# Thin functional convenience API
_default_analyzer: AudioReactivity | None = None


def get_default_analyzer() -> AudioReactivity:
    global _default_analyzer
    if _default_analyzer is None:
        _default_analyzer = AudioReactivity()
    return _default_analyzer


def process_frame(
    spectrum: Any = None,
    pcm: Any = None,
    sample_rate: int | None = None,
    timestamp: float | None = None,
    analyzer: AudioReactivity | None = None,
) -> dict[str, Any]:
    """Thin functional convenience API."""
    if analyzer is None:
        analyzer = get_default_analyzer()
    return analyzer.process_frame(
        spectrum=spectrum, pcm=pcm, sample_rate=sample_rate, timestamp=timestamp
    )
