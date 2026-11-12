from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .core import (
    DEFAULT_RADIO_STATIONS,
    PIL_AVAILABLE,
    RADIO_LOGO_CACHE_DIR,
    RADIO_STATIONS_FILE,
    REQUESTS_AVAILABLE,
    _atomic_write_json,
    _get_download_executor,
    _image_cache_path,
    _linamp_attr,
    fetch_image_to_cache,
)


def load_radio_stations() -> list[dict[str, str]]:
    defaults_by_name: dict[str, dict[str, str]] = {
        s.get("name", ""): s
        for s in _linamp_attr("DEFAULT_RADIO_STATIONS", DEFAULT_RADIO_STATIONS)
    }

    def _merge_defaults(stations: list) -> list[dict[str, str]]:
        """Fill missing logo/homepage fields from built-in defaults."""
        merged: list[dict[str, str]] = []
        for s in stations:
            if not isinstance(s, dict):
                continue
            item: dict[str, str] = dict(s)
            name = item.get("name", "")
            if not isinstance(name, str):
                name = ""
            item["name"] = name
            base = defaults_by_name.get(name, {})
            if not item.get("logo") and base.get("logo"):
                item["logo"] = base["logo"]
            if not item.get("homepage") and base.get("homepage"):
                item["homepage"] = base["homepage"]
            merged.append(item)
        return merged

    _stations_file = _linamp_attr("RADIO_STATIONS_FILE", RADIO_STATIONS_FILE)
    _defaults: list[dict[str, str]] = _linamp_attr(
        "DEFAULT_RADIO_STATIONS", DEFAULT_RADIO_STATIONS
    )
    if not _stations_file.exists():
        return _defaults.copy()
    try:
        with open(_stations_file, encoding="utf-8") as f:
            stations = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logging.getLogger(__name__).warning(
            "radio_stations.json is unreadable (%s); falling back to defaults",
            e,
        )
        try:
            _stations_file.replace(_stations_file.with_suffix(".json.corrupt"))
        except OSError:
            pass
        return _defaults.copy()
    if isinstance(stations, list):
        return _merge_defaults(stations) if stations else []
    logging.getLogger(__name__).warning(
        "radio_stations.json has an unexpected top-level type; ignoring it"
    )
    return []


def save_radio_stations(stations: list[dict[str, str]]) -> bool:
    _stations_file = _linamp_attr("RADIO_STATIONS_FILE", RADIO_STATIONS_FILE)
    return _atomic_write_json(_stations_file, stations)


def fetch_radio_logo(logo_url: str, timeout: float = 8.0) -> Path | None:
    _logo_cache = _linamp_attr("RADIO_LOGO_CACHE_DIR", RADIO_LOGO_CACHE_DIR)
    return fetch_image_to_cache(
        logo_url, _logo_cache, (128, 128), timeout, allow_local=True
    )


def _radio_logo_cache_path(logo_url: str) -> Path:
    _logo_cache = _linamp_attr("RADIO_LOGO_CACHE_DIR", RADIO_LOGO_CACHE_DIR)
    return _image_cache_path(logo_url, _logo_cache)


def preload_radio_logos(stations: list[dict]) -> None:
    """Best-effort background preload of station logos into the cache."""
    if not REQUESTS_AVAILABLE or not PIL_AVAILABLE:
        return

    def _worker() -> None:
        for s in stations:
            url = (s or {}).get("logo") or ""
            if url:
                try:
                    fetch_radio_logo(url)
                except Exception:
                    logging.getLogger(__name__).exception("Radio logo preload failed")

    executor = _get_download_executor()
    if executor is None:
        return
    try:
        executor.submit(_worker)
    except RuntimeError:
        logging.getLogger(__name__).debug(
            "Download executor unavailable; skipping preload"
        )


def submit_radio_logo(logo_url: str, executor: Any) -> None:
    """Submit a radio logo download to *executor*.
    Resilient to a dead/shutdown executor: returns ``None`` on failure
    instead of propagating RuntimeError or TypeError.
    """
    if not logo_url or not REQUESTS_AVAILABLE or not PIL_AVAILABLE:
        return
    try:
        executor.submit(fetch_radio_logo, logo_url)
    except (RuntimeError, TypeError, AttributeError):
        return


__all__ = [
    "_radio_logo_cache_path",
    "fetch_radio_logo",
    "load_radio_stations",
    "preload_radio_logos",
    "save_radio_stations",
    "submit_radio_logo",
]
