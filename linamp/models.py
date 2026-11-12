from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from typing import Any

from .core import (
    AUDIO_EXTENSIONS,
    DEFAULT_VOLUME,
    _is_safe_feed_id,
    _is_valid_number,
    _is_valid_number_in_range,
)


@dataclass
class RadioStation:
    name: str = ""
    genre: str = ""
    url: str = ""
    logo: str = ""
    homepage: str = ""

    def to_dict(self) -> dict[str, str]:
        d = {"name": self.name, "genre": self.genre, "url": self.url}
        if self.logo:
            d["logo"] = self.logo
        if self.homepage:
            d["homepage"] = self.homepage
        return d

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> RadioStation:
        return cls(
            name=data.get("name", ""),
            genre=data.get("genre", ""),
            url=data.get("url", ""),
            logo=data.get("logo", ""),
            homepage=data.get("homepage", ""),
        )


@dataclass
class PodcastEpisode:
    guid: str = ""
    title: str = ""
    description: str = ""
    pub_date: str = ""
    duration: int = 0
    audio_url: str = ""
    artwork_url: str = ""
    played: bool = False
    position: int = 0
    downloaded_path: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> PodcastEpisode:
        if not isinstance(data, dict):
            data = {}
        field_types = {f.name: f.type for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            if key not in field_types:
                continue
            ftype = field_types[key]
            if key in ("duration", "position"):
                if (
                    isinstance(value, str) and value.strip().lstrip("-").isdigit()
                ) or _is_valid_number(value):
                    kwargs[key] = max(0, int(value))
                else:
                    kwargs[key] = 0
            elif key == "played":
                if isinstance(value, bool):
                    kwargs[key] = value
                elif isinstance(value, str):
                    kwargs[key] = value.lower() in ("true", "1", "yes")
                else:
                    kwargs[key] = bool(value)
            elif key == "downloaded_path" or ftype == "str":
                kwargs[key] = str(value) if value is not None else ""
            else:
                kwargs[key] = value
        return cls(**kwargs)


@dataclass
class PodcastFeed:
    feed_id: str = ""
    title: str = ""
    author: str = ""
    description: str = ""
    artwork_url: str = ""
    homepage: str = ""
    feed_url: str = ""
    last_updated: str = ""
    episodes: list[PodcastEpisode] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["episodes"] = [ep.to_dict() for ep in self.episodes]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> PodcastFeed:
        if not isinstance(data, dict):
            data = {}
        episodes_data = data.get("episodes", [])
        if not isinstance(episodes_data, list):
            episodes_data = []
        valid = {f.name for f in fields(cls)}
        feed_data = {
            key: value
            for key, value in data.items()
            if key != "episodes" and key in valid
        }
        raw_feed_id = feed_data.get("feed_id")
        if not isinstance(raw_feed_id, str) or not _is_safe_feed_id(raw_feed_id):
            raw_feed_url = feed_data.get("feed_url")
            if not isinstance(raw_feed_url, str):
                raw_feed_url = ""
            feed_data["feed_id"] = cls.generate_feed_id(raw_feed_url)
        episodes = [
            PodcastEpisode.from_dict(ep) for ep in episodes_data if isinstance(ep, dict)
        ]
        return cls(episodes=episodes, **feed_data)

    @staticmethod
    def generate_feed_id(feed_url: str) -> str:
        return hashlib.md5(feed_url.encode("utf-8")).hexdigest()


@dataclass
class PlaylistItem:
    path: str
    title: str = ""
    duration: int = 0
    cover_art_path: str = ""
    id: str = ""
    artist: str = ""
    album: str = ""
    track_number: int = 0
    total_tracks: int = 0
    genre: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        self.path = os.path.abspath(os.path.expanduser(str(self.path)))
        if not isinstance(self.title, str):
            self.title = ""
        if not isinstance(self.id, str):
            self.id = ""
        if not isinstance(self.cover_art_path, str):
            self.cover_art_path = ""
        if not isinstance(self.artist, str):
            self.artist = ""
        if not isinstance(self.album, str):
            self.album = ""
        if not _is_valid_number_in_range(self.track_number, 0, 10000):
            self.track_number = 0
        else:
            self.track_number = int(self.track_number)
        if not _is_valid_number_in_range(self.total_tracks, 0, 10000):
            self.total_tracks = 0
        else:
            self.total_tracks = int(self.total_tracks)
        if not _is_valid_number_in_range(self.duration, 0, 86400):
            self.duration = 0
        else:
            self.duration = int(self.duration)
        if not self.title:
            self.title = os.path.splitext(os.path.basename(self.path))[0]
        else:
            _, ext = os.path.splitext(self.title)
            if ext.lower() in AUDIO_EXTENSIONS:
                self.title = self.title[: -len(ext)]
        if not self.id:
            self.id = hashlib.md5(f"{self.path}:{self.title}".encode()).hexdigest()[:32]
        if not self.genre:
            self.genre = ""
        if not self.description:
            self.description = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "title": self.title,
            "duration": self.duration,
            "cover_art_path": self.cover_art_path,
            "id": self.id,
            "artist": self.artist,
            "album": self.album,
            "track_number": self.track_number,
            "total_tracks": self.total_tracks,
            "genre": self.genre,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlaylistItem:
        if not isinstance(data, dict):
            raise TypeError("Input data must be a dictionary")
        item_data = data.copy()
        field_types = {f.name: f.type for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        for key, value in item_data.items():
            if key not in field_types:
                continue
            ftype = field_types[key]
            if key in ("duration", "track_number", "total_tracks"):
                if (
                    isinstance(value, str) and value.strip().lstrip("-").isdigit()
                ) or _is_valid_number(value):
                    kwargs[key] = max(0, int(value))
                else:
                    kwargs[key] = 0
            elif key == "played":
                if isinstance(value, bool):
                    kwargs[key] = value
                elif isinstance(value, str):
                    kwargs[key] = value.lower() in ("true", "1", "yes")
                else:
                    kwargs[key] = bool(value)
            elif key == "downloaded_path" or ftype == "str":
                kwargs[key] = str(value) if value is not None else ""
            else:
                kwargs[key] = value
        return cls(**kwargs)

    def exists(self) -> bool:
        return os.path.isfile(self.path)

    def get_display_name(self) -> str:
        return self.title if self.title else os.path.basename(self.path)


@dataclass
class PlayerSettings:
    auto_play_next: bool = True
    shuffle_mode: bool = False
    repeat_mode: str = "none"
    beat_aware_enabled: bool = True
    beat_threshold: float = 0.1
    auto_download_covers: bool = False
    volume: float = DEFAULT_VOLUME
    position: float = 0.0
    window_size: tuple[int, int] = (600, 450)
    window_position: tuple[int, int] = (100, 100)
    equalizer_settings: list[float] = field(default_factory=lambda: [0.0] * 10)
    last_played_track: str = ""
    last_played_position: float = 0.0
    show_cover_art: bool = True
    cover_art_size: int = 650
    cover_art_shape: str = "square"
    milkdrop_auto_switch: bool = True
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.equalizer_settings is None:
            self.equalizer_settings = [0.0] * 10
        self._validate_and_coerce()

    def _validate_and_coerce(self) -> None:
        """Validate and coerce field values to safe defaults.
        This makes the dataclass robust against malformed data from
        any caller of from_dict(), not just the normal loading path.
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
            value = getattr(self, field_name)
            if not isinstance(value, bool):
                setattr(self, field_name, field_name in default_true_fields)
        if self.repeat_mode not in ("none", "one", "all"):
            self.repeat_mode = "none"
        ws = self.window_size
        if (
            not isinstance(ws, (list, tuple))
            or len(ws) != 2
            or not all(isinstance(v, int) and v > 0 for v in ws)
        ):
            self.window_size = (600, 450)
        else:
            self.window_size = (
                max(300, min(3840, ws[0])),
                max(200, min(2160, ws[1])),
            )
        wp = self.window_position
        if (
            not isinstance(wp, (list, tuple))
            or len(wp) != 2
            or not all(isinstance(v, int) for v in wp)
        ):
            self.window_position = (100, 100)
        else:
            self.window_position = (
                max(-3840, min(3840, wp[0])),
                max(-2160, min(2160, wp[1])),
            )
        if not _is_valid_number_in_range(self.volume, 0.0, 1.0):
            self.volume = 1.0
        if not _is_valid_number_in_range(self.position, 0.0):
            self.position = 0.0
        if not _is_valid_number_in_range(self.beat_threshold, 0.0, 1.0):
            self.beat_threshold = 0.1
        if not _is_valid_number_in_range(self.cover_art_size, 50.0, 800.0):
            self.cover_art_size = 650
        else:
            self.cover_art_size = int(self.cover_art_size)
        if not isinstance(self.cover_art_shape, str) or self.cover_art_shape not in (
            "square",
            "circle",
            "rounded",
        ):
            self.cover_art_shape = "square"
        eq = self.equalizer_settings
        if not isinstance(eq, list) or len(eq) != 10:
            self.equalizer_settings = [0.0] * 10
        else:
            self.equalizer_settings = [
                max(-12.0, min(12.0, float(v) if _is_valid_number(v) else 0.0))
                for v in eq
            ]
        if not isinstance(self.last_played_track, str):
            self.last_played_track = ""
        if not _is_valid_number_in_range(self.last_played_position, 0.0):
            self.last_played_position = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "auto_play_next": self.auto_play_next,
            "shuffle_mode": self.shuffle_mode,
            "repeat_mode": self.repeat_mode,
            "beat_aware_enabled": self.beat_aware_enabled,
            "beat_threshold": self.beat_threshold,
            "auto_download_covers": self.auto_download_covers,
            "volume": self.volume,
            "position": self.position,
            "window_size": self.window_size,
            "window_position": self.window_position,
            "equalizer_settings": self.equalizer_settings,
            "last_played_track": self.last_played_track,
            "last_played_position": self.last_played_position,
            "show_cover_art": self.show_cover_art,
            "cover_art_size": self.cover_art_size,
            "cover_art_shape": self.cover_art_shape,
            "milkdrop_auto_switch": self.milkdrop_auto_switch,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlayerSettings:
        if not isinstance(data, dict):
            data = {}
        return cls(
            schema_version=data.get("schema_version", 1),
            auto_play_next=data.get("auto_play_next", True),
            shuffle_mode=data.get("shuffle_mode", False),
            repeat_mode=data.get("repeat_mode", "none"),
            beat_aware_enabled=data.get("beat_aware_enabled", True),
            beat_threshold=data.get("beat_threshold", 0.1),
            auto_download_covers=data.get("auto_download_covers", False),
            volume=data.get("volume", 1.0),
            position=data.get("position", 0.0),
            window_size=data.get("window_size", (600, 450)),
            window_position=data.get("window_position", (100, 100)),
            equalizer_settings=data.get("equalizer_settings", [0.0] * 10),
            last_played_track=data.get("last_played_track", ""),
            last_played_position=data.get("last_played_position", 0.0),
            show_cover_art=data.get("show_cover_art", True),
            cover_art_size=data.get("cover_art_size", 650),
            cover_art_shape=data.get("cover_art_shape", "square"),
            milkdrop_auto_switch=data.get("milkdrop_auto_switch", True),
        )


class PlaybackStatus(Enum):
    """Playback status used by the watchdog and transaction lifecycle.

    Members are compared by identity (``is``) in the playback code, so each
    member is a singleton as guaranteed by :class:`enum.Enum`.
    """

    UNKNOWN = "unknown"
    CONNECTING = "connecting"
    QUEUED = "queued"
    PLAYING = "playing"
    PAUSED = "paused"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class PlaybackTransaction:
    """A single playback/transaction attempt.

    Carries the generation, URI, requested GStreamer state, current status
    and whether the stream is a podcast episode. Used by the watchdog and
    transaction lifecycle to detect stale callbacks.
    """

    generation: int
    uri: str
    requested_state: object
    status: PlaybackStatus
    is_podcast: bool = False


class PlaybackController:
    """Controller for a playback transaction lifecycle.

    Holds a reference to the owning window and a logger. Tests may construct
    it via ``__new__`` and set attributes manually; normal construction runs
    ``__init__``.
    """

    window: Any
    logger: Any

    def __init__(self) -> None:
        self.window = None
        self.logger = None


def _format_playback_failure(record: dict) -> str:
    """Format a structured playback failure record into readable text.

    Emits one ``label: value`` line per known field so the failure record can
    be surfaced in the UI status message. Fields missing from ``record`` are
    reported as ``None`` so every expected label is always present.
    """

    fields = (
        ("uri", "URI:"),
        ("source", "source:"),
        ("error", "error:"),
        ("debug", "debug:"),
        ("error_domain", "error_domain:"),
        ("error_code", "error_code:"),
        ("playback_status", "playback_status:"),
        ("generation", "generation:"),
        ("playback_generation", "playback_generation:"),
        ("playbin_state", "playbin_state:"),
        ("media_kind", "media_kind:"),
        ("visualizer_audio_path", "visualizer_audio_path:"),
    )
    lines = [f"{label} {record.get(key)}" for key, label in fields]
    return "\n".join(lines)


__all__ = [
    "PlaybackController",
    "PlaybackStatus",
    "PlaybackTransaction",
    "PlayerSettings",
    "PlaylistItem",
    "PodcastEpisode",
    "PodcastFeed",
    "RadioStation",
    "_format_playback_failure",
]
