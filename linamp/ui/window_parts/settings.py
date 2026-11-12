"""WinampWindow mixin: Settings persistence and settings/state synchronisation."""

from __future__ import annotations

import gi

from .base import WindowBase

gi.require_version("Gdk", "4.0")
gi.require_version("Gtk", "4.0")
gi.require_version("Gst", "1.0")
import json
from dataclasses import asdict

from gi.repository import Gst  # type: ignore[reportMissingModuleSource]

from ...core import (
    AUTO_SAVE_DELAY,
    SETTINGS_FILE,
    _atomic_write_json,
    _is_valid_number_in_range,
)
from ...models import (
    PlayerSettings,
)


class SettingsMixin(WindowBase):
    """Settings persistence and settings/state synchronisation.

    The methods are split out of ``WinampWindow`` so the window class
    stays readable; they keep operating on the window instance and
    share its playback, playlist and settings state.
    """

    def save_settings(self) -> None:
        data = asdict(self.settings)
        data["schema_version"] = getattr(self.settings, "schema_version", 1)
        _atomic_write_json(SETTINGS_FILE, data, mode=0o600)

    def load_settings(self):
        if not SETTINGS_FILE.exists():
            return
        try:
            with open(SETTINGS_FILE, encoding="utf-8") as f:
                content = f.read()
            if not content.strip():
                return
            settings_dict = json.loads(content)
            if not isinstance(settings_dict, dict):
                raise TypeError("Settings file must contain a JSON object")
            self._migrate_settings(settings_dict)
            self._sanitize_settings(settings_dict)
            self.settings = PlayerSettings.from_dict(settings_dict)
            self._apply_settings_to_state()
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            self.logger.warning("Invalid settings file: %s", exc)
            self.settings = PlayerSettings()

    def _migrate_settings(self, settings_dict: dict) -> None:
        """Migrate settings between schema versions."""
        version = settings_dict.get("schema_version", 0)
        if not isinstance(version, int):
            version = 0
        if version < 1:
            # Add new fields with defaults if missing
            settings_dict.setdefault("schema_version", 1)
        settings_dict["schema_version"] = getattr(
            PlayerSettings, "schema_version", 1
        ) if hasattr(PlayerSettings, "schema_version") else 1
        # For future migrations, add cases here

    def _sanitize_settings(self, settings_dict: dict) -> None:
        """Coerce/validate persisted settings in place, leaving only safe values.
        Only window_size, window_position and repeat_mode were validated before,
        so a malformed settings file could still feed invalid types into
        PlayerSettings. Every persisted field is now guarded so a corrupt file
        degrades to defaults instead of crashing startup.
        """
        bool_fields = (
            "auto_play_next",
            "shuffle_mode",
            "beat_aware_enabled",
            "auto_download_covers",
            "show_cover_art",
            "milkdrop_auto_switch",
        )
        default_true_fields = {
            "auto_play_next",
            "beat_aware_enabled",
            "show_cover_art",
            "milkdrop_auto_switch",
        }
        for field_name in bool_fields:
            if field_name in settings_dict and not isinstance(
                settings_dict[field_name], bool
            ):
                settings_dict[field_name] = field_name in default_true_fields
        if "repeat_mode" in settings_dict and settings_dict["repeat_mode"] not in (
            "none",
            "one",
            "all",
        ):
            settings_dict["repeat_mode"] = "none"
        if "window_size" in settings_dict:
            ws = settings_dict["window_size"]
            if (
                not isinstance(ws, (list, tuple))
                or len(ws) != 2
                or not all(isinstance(v, int) and v > 0 for v in ws)
            ):
                settings_dict["window_size"] = [600, 450]
            else:
                settings_dict["window_size"] = [
                    max(300, min(3840, ws[0])),
                    max(200, min(2160, ws[1])),
                ]
        if "window_position" in settings_dict:
            wp = settings_dict["window_position"]
            if (
                not isinstance(wp, (list, tuple))
                or len(wp) != 2
                or not all(isinstance(v, int) for v in wp)
            ):
                settings_dict["window_position"] = [100, 100]
            else:
                settings_dict["window_position"] = [
                    max(-3840, min(3840, wp[0])),
                    max(-2160, min(2160, wp[1])),
                ]
        if "volume" in settings_dict and not _is_valid_number_in_range(
            settings_dict["volume"], 0.0, 1.0
        ):
            settings_dict["volume"] = 1.0
        if "position" in settings_dict and not _is_valid_number_in_range(
            settings_dict["position"], 0.0
        ):
            settings_dict["position"] = 0.0
        if "beat_threshold" in settings_dict and not _is_valid_number_in_range(
            settings_dict["beat_threshold"], 0.0, 1.0
        ):
            settings_dict["beat_threshold"] = 0.1
        if "cover_art_size" in settings_dict and not _is_valid_number_in_range(
            settings_dict["cover_art_size"], 50.0, 800.0
        ):
            settings_dict["cover_art_size"] = 650
        if "cover_art_shape" in settings_dict and (
            not isinstance(settings_dict["cover_art_shape"], str)
            or settings_dict["cover_art_shape"] not in ("square", "circle", "rounded")
        ):
            settings_dict["cover_art_shape"] = "square"
        if "last_played_track" in settings_dict and not isinstance(
            settings_dict["last_played_track"], str
        ):
            settings_dict["last_played_track"] = ""
        if "last_played_position" in settings_dict and not _is_valid_number_in_range(
            settings_dict["last_played_position"], 0.0
        ):
            settings_dict["last_played_position"] = 0.0
        eq = settings_dict.get("equalizer_settings")
        if (
            not isinstance(eq, list)
            or len(eq) != 10
            or not all(_is_valid_number_in_range(v, -12.0, 12.0) for v in eq)
        ):
            settings_dict["equalizer_settings"] = [0.0] * 10

    def _update_settings_from_state(self):
        if not hasattr(self, "settings"):
            return
        self.settings.auto_play_next = self.auto_play_next
        self.settings.shuffle_mode = self.shuffle_mode
        self.settings.repeat_mode = self.repeat_mode
        self.settings.beat_aware_enabled = self.beat_aware_enabled
        self.settings.beat_threshold = self.beat_threshold
        self.settings.auto_download_covers = self.auto_download_covers
        self.settings.milkdrop_auto_switch = self.milkdrop_auto_switch
        if self.current_track >= 0 and self.current_track < len(self.playlist):
            current_item = self.playlist[self.current_track]
            self.settings.last_played_track = current_item.path
            if self.playing and hasattr(self, "player") and self.player:
                success, position = self.player.query_position(Gst.Format.TIME)
                if success:
                    self.settings.last_played_position = position / Gst.SECOND
                else:
                    self.settings.last_played_position = 0.0
        if hasattr(self, "player") and self.player:
            volume = self.player.get_property("volume")
            if volume is not None:
                self.settings.volume = volume
        if hasattr(self, "get_width") and hasattr(self, "get_height"):
            width = self.get_width()
            height = self.get_height()
            if width > 0 and height > 0:
                self.settings.window_size = (width, height)
            elif hasattr(self, "get_default_size"):
                size = self.get_default_size()
                if size[0] > 0 and size[1] > 0:
                    self.settings.window_size = (size[0], size[1])
        elif hasattr(self, "get_default_size"):
            size = self.get_default_size()
            if size[0] > 0 and size[1] > 0:
                self.settings.window_size = (size[0], size[1])
        if hasattr(self, "equalizer") and self.equalizer:
            self.settings.equalizer_settings = []
            for i in range(10):
                value = self.equalizer.get_property("band" + str(i))
                self.settings.equalizer_settings.append(value)

    def _apply_settings_to_state(self):
        if not hasattr(self, "settings"):
            return
        self.auto_play_next = self.settings.auto_play_next
        self.shuffle_mode = self.settings.shuffle_mode
        self.repeat_mode = self.settings.repeat_mode
        self.beat_aware_enabled = self.settings.beat_aware_enabled
        self.beat_threshold = self.settings.beat_threshold
        self.auto_download_covers = self.settings.auto_download_covers
        self.milkdrop_auto_switch = self.settings.milkdrop_auto_switch
        if hasattr(self, "player") and self.player:
            volume = max(0.0, min(1.0, self.settings.volume))
            self.player.set_property("volume", volume)
        if hasattr(self, "player_tab") and hasattr(self.player_tab, "volume_scale"):
            volume_percent = self.settings.volume * 100
            clamped_volume = max(0, min(100, volume_percent))
            self.player_tab.volume_scale.set_value(clamped_volume)
            if hasattr(self.player_tab, "volume_label"):
                self.player_tab.volume_label.set_label(f"{int(clamped_volume)}%")
        if hasattr(self, "equalizer_tab") and self.settings.equalizer_settings:
            for i, value in enumerate(self.settings.equalizer_settings[:10]):
                if i < len(self.equalizer_tab.band_scales):
                    scale, value_label = self.equalizer_tab.band_scales[i]
                    clamped_value = max(-12, min(12, value))
                    scale.set_value(clamped_value)
                    value_label.set_text(f"{clamped_value:+.1f} dB")
        if hasattr(self, "player_tab") and hasattr(self.player_tab, "cover_art_widget"):
            try:
                self.player_tab.cover_art_widget.set_shape(
                    getattr(self.settings, "cover_art_shape", "square")
                )
                if hasattr(self.player_tab.cover_art_widget, "set_size"):
                    self.player_tab.cover_art_widget.set_size(
                        int(getattr(self.settings, "cover_art_size", 150))
                    )
            except (AttributeError, TypeError):
                pass
        if hasattr(self, "milkdrop_auto_switch"):
            self.milkdrop_auto_switch = bool(
                getattr(self.settings, "milkdrop_auto_switch", True)
            )

    def _apply_ui_settings_delayed(self):
        if not hasattr(self, "settings"):
            return False
        if hasattr(self, "player_tab") and hasattr(self.player_tab, "volume_scale"):
            volume_percent = self.settings.volume * 100
            clamped_volume = max(0, min(100, volume_percent))
            self.player_tab.volume_scale.set_value(clamped_volume)
            if hasattr(self.player_tab, "volume_label"):
                self.player_tab.volume_label.set_label(f"{int(clamped_volume)}%")
        if hasattr(self, "equalizer_tab") and self.settings.equalizer_settings:
            for i, value in enumerate(self.settings.equalizer_settings[:10]):
                if i < len(self.equalizer_tab.band_scales):
                    scale, value_label = self.equalizer_tab.band_scales[i]
                    clamped_value = max(-12, min(12, value))
                    scale.set_value(clamped_value)
                    value_label.set_text(f"{clamped_value:+.1f} dB")
        return

    def _apply_equalizer_settings_delayed(self):
        if (
            hasattr(self, "equalizer")
            and self.equalizer
            and hasattr(self, "settings")
            and self.settings.equalizer_settings
        ):
            for i, value in enumerate(self.settings.equalizer_settings[:10]):
                self.equalizer.set_property("band" + str(i), value)
        return False

    def save_settings_on_track_change(self):
        self._update_settings_from_state()
        self.save_settings()

    def save_settings_on_stop(self):
        self._update_settings_from_state()
        self.save_settings()

    def periodic_auto_save(self):
        if self.playing:
            self.auto_save_settings()
        return True

    def auto_save_settings(self):
        if hasattr(self, "_auto_save_timer") and self._auto_save_timer:
            self._safe_source_remove(self._auto_save_timer)
        self._auto_save_timer = self._register_timer(
            AUTO_SAVE_DELAY, self._auto_save_callback
        )

    def _auto_save_callback(self):
        self._update_settings_from_state()
        self.save_settings()
        self._auto_save_timer = 0
        return False
