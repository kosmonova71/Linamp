from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gi

gi.require_version("Gdk", "4.0")
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk

from ...core import *
from ...media import *
from ...models import *
from ...podcasts import *
from ...radio import *


class EqualizerTab(Gtk.Box):
    GRID_COLUMNS = 4
    EQ_PRESETS_PATH = Path(__file__).parent / "configs" / "eq_presets.json"

    def __init__(self, player: Any) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self.player = player
        self.add_css_class("eq-tab")
        self.bands = 10
        self.band_scales = []
        self.eq_presets = self._load_eq_presets()
        self.eq_presets_map = {p["name"]: p["values"] for p in self.eq_presets}
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16)
        main_box.set_margin_top(16)
        main_box.set_margin_bottom(16)
        main_box.set_margin_start(16)
        main_box.set_margin_end(16)
        self.append(main_box)
        header_section = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        main_box.append(header_section)
        header_label = Gtk.Label(label="10-Band Equalizer")
        header_label.add_css_class("section-header")
        header_label.set_halign(Gtk.Align.CENTER)
        header_section.append(header_label)
        bands_container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        main_box.append(bands_container)
        freq_labels_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        freq_labels_row.set_halign(Gtk.Align.CENTER)
        bands_container.append(freq_labels_row)
        freqs = ["60", "170", "310", "600", "1k", "3k", "6k", "12k", "14k", "16k"]
        for freq in freqs:
            freq_label = Gtk.Label(label=freq)
            freq_label.add_css_class("eq-label")
            freq_label.set_size_request(35, -1)
            freq_label.set_halign(Gtk.Align.CENTER)
            freq_labels_row.append(freq_label)
        sliders_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        sliders_row.set_halign(Gtk.Align.CENTER)
        bands_container.append(sliders_row)
        for i in range(self.bands):
            band_container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
            band_container.add_css_class("eq-band")
            band_container.set_size_request(50, 180)
            value_label = Gtk.Label(label="0 dB")
            value_label.add_css_class("eq-value-label")
            value_label.set_size_request(40, -1)
            value_label.set_halign(Gtk.Align.CENTER)
            band_container.append(value_label)
            lower, upper = -12, 12
            value = max(lower, min(upper, 0))
            adjustment = Gtk.Adjustment(
                value=value,
                lower=lower,
                upper=upper,
                step_increment=0.1,
                page_increment=1,
                page_size=0,
            )
            scale = Gtk.Scale(
                orientation=Gtk.Orientation.VERTICAL, adjustment=adjustment
            )
            scale.set_inverted(True)
            scale.set_draw_value(False)
            scale.set_size_request(40, 140)
            scale.set_margin_top(4)
            scale.set_margin_bottom(4)
            scale.add_css_class("eq-scale")
            scale.connect("value-changed", self.on_band_changed, i, value_label)
            scale.add_mark(0, Gtk.PositionType.LEFT, "0")
            scale.add_mark(-12, Gtk.PositionType.LEFT, "-12")
            scale.add_mark(12, Gtk.PositionType.LEFT, "+12")
            band_container.append(scale)
            sliders_row.append(band_container)
            self.band_scales.append((scale, value_label))
        presets_section = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        main_box.append(presets_section)
        presets_header = Gtk.Label(label="Presets")
        presets_header.add_css_class("section-header")
        presets_header.set_halign(Gtk.Align.START)
        presets_section.append(presets_header)
        presets_grid = Gtk.Grid()
        presets_grid.set_column_spacing(8)
        presets_grid.set_row_spacing(8)
        presets_grid.set_halign(Gtk.Align.CENTER)
        presets_section.append(presets_grid)
        presets = []
        for idx, preset in enumerate(self.eq_presets):
            col = idx % self.GRID_COLUMNS
            row = idx // self.GRID_COLUMNS
            presets.append((preset["name"], col, row))
        for preset_name, col, row in presets:
            btn = self._create_preset_button(preset_name)
            btn.connect("clicked", self.on_preset_clicked, preset_name)
            presets_grid.attach(btn, col, row, 1, 1)
        reset_container = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        reset_container.set_halign(Gtk.Align.CENTER)
        reset_container.set_margin_top(12)
        presets_section.append(reset_container)
        reset_btn = self._create_preset_button("Reset to Flat")
        reset_btn.connect("clicked", self.on_reset_clicked)
        reset_container.append(reset_btn)
        self.load_saved_settings()

    def _load_eq_presets(self) -> list[dict[str, Any]]:
        defaults = [
            {"name": "Flat", "values": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
            {"name": "Pop", "values": [4, 3, 2, 1, 0, -1, -2, -2, -2, -2]},
            {"name": "Rock", "values": [6, 4, 2, 0, -2, -3, -3, -3, -3, -3]},
            {"name": "Jazz", "values": [3, 2, 1, 1, 0, -1, -2, -2, -2, -2]},
            {"name": "Classical", "values": [4, 3, 2, 1, 0, -1, -2, -3, -4, -5]},
            {"name": "Electronic", "values": [5, 4, 3, 1, 0, 1, 2, 3, 4, 5]},
            {"name": "Hip-Hop", "values": [5, 4, 2, 0, -1, 1, 3, 4, 5, 5]},
            {"name": "R&B", "values": [3, 6, 5, 2, -1, -1, 2, 3, 3, 4]},
            {"name": "Metal", "values": [7, 6, 5, 3, 1, -1, -2, -3, -3, -3]},
            {"name": "Dance", "values": [6, 5, 2, 0, 0, -2, -3, -3, 0, 0]},
            {"name": "Acoustic", "values": [-2, -1, 0, 2, 4, 4, 3, 2, 1, 0]},
            {"name": "Vocal", "values": [-1, 0, 2, 3, 4, 3, 2, 1, 0, -1]},
            {"name": "Bass Boost", "values": [8, 7, 6, 4, 2, 0, -1, -2, -3, -4]},
            {"name": "Treble Boost", "values": [-4, -3, -2, 0, 2, 4, 6, 7, 8, 8]},
            {"name": "Lounge", "values": [-2, -1, 1, 3, 4, 2, 0, -1, -1, -2]},
            {"name": "Live", "values": [-1, 0, 2, 3, 3, 3, 2, 1, 1, 1]},
        ]
        try:
            with open(self.EQ_PRESETS_PATH, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return defaults
        loaded = []
        seen = set()
        if not isinstance(data, list):
            return defaults
        for item in data:
            if not isinstance(item, dict):
                continue
            name = item.get("name", "")
            values = item.get("values", [])
            if not isinstance(name, str):
                continue
            if not isinstance(values, (list, tuple)) or len(values) != 10:
                continue
            if not name or name in seen:
                continue
            try:
                bands = [float(v) for v in values]
            except (TypeError, ValueError):
                continue
            if not all(_is_valid_number(v) for v in bands):
                continue
            seen.add(name)
            loaded.append({"name": name, "values": bands})
        return loaded if loaded else defaults

    def load_saved_settings(self) -> None:
        if hasattr(self.player, "settings") and self.player.settings.equalizer_settings:
            for i, value in enumerate(self.player.settings.equalizer_settings[:10]):
                if i < len(self.band_scales):
                    try:
                        numeric = float(value)
                    except (TypeError, ValueError):
                        numeric = 0.0
                    clamped_value = max(-12.0, min(12.0, numeric))
                    scale, value_label = self.band_scales[i]
                    scale.set_value(clamped_value)
                    value_label.set_text(f"{clamped_value:+.1f} dB")

    def _create_preset_button(self, text: str) -> Gtk.Button:
        button = Gtk.Button(label=text)
        button.add_css_class("eq-preset-btn")
        button.set_size_request(80, 28)
        return button

    def on_band_changed(
        self, scale: Gtk.Scale, band: int, value_label: Gtk.Label
    ) -> None:
        if hasattr(self.player, "equalizer") and self.player.equalizer:
            value = scale.get_value()
            clamped_value = max(-12, min(12, value))
            if clamped_value != value:
                scale.set_value(clamped_value)
                value = clamped_value
            self.player.equalizer.set_property(f"band{band}", value)
            value_label.set_text(f"{value:+.1f} dB")
            if not hasattr(self, "_eq_save_timer_id"):
                self._eq_save_timer_id = None
            elif self._eq_save_timer_id is not None:
                self.player._safe_source_remove(self._eq_save_timer_id)

            def _debounced_save() -> bool:
                self._eq_save_timer_id = None
                if hasattr(self.player, "auto_save_settings"):
                    self.player.auto_save_settings()
                return False

            self._eq_save_timer_id = self.player._register_timer(1000, _debounced_save)

    def on_reset_clicked(self, button: Gtk.Button) -> None:
        for scale, value_label in self.band_scales:
            scale.set_value(0)
        if hasattr(self.player, "auto_save_settings"):
            self.player.auto_save_settings()

    def on_preset_clicked(self, button: Gtk.Button, preset_name: str) -> None:
        if preset_name in self.eq_presets_map:
            values = self.eq_presets_map[preset_name]
            for i, (scale, value_label) in enumerate(self.band_scales):
                scale.set_value(values[i])
            if hasattr(self.player, "auto_save_settings"):
                self.player.auto_save_settings()
