"""Window mixin modules.

The main window class is composed from focused mixins so each subsystem
(settings, playlist, podcasts, playback, radio, audio sink, display and
navigation) lives in its own module.
"""

from __future__ import annotations

from .audio_sink import AudioSinkMixin
from .display import DisplayMixin
from .navigation import NavigationMixin
from .playback import PlaybackMixin
from .playlist import PlaylistMixin
from .podcasts import PodcastsMixin
from .radio import RadioMixin
from .settings import SettingsMixin

__all__ = [
    "AudioSinkMixin",
    "DisplayMixin",
    "NavigationMixin",
    "PlaybackMixin",
    "PlaylistMixin",
    "PodcastsMixin",
    "RadioMixin",
    "SettingsMixin",
]
