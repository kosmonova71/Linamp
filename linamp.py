from __future__ import annotations

import hashlib
import ipaddress
import json
import locale
import logging
import os
import random
import re
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import quote, urlparse

import defusedxml.ElementTree as DefusedET
import gi
import mutagen
import numpy as np
import requests
from PIL import Image, ImageEnhance

gi.require_version("Gdk", "4.0")
gi.require_version("Gtk", "4.0")
gi.require_version("Gst", "1.0")
from gi.repository import Gdk, GdkPixbuf, Gio, GLib, Gst, Gtk, Pango
from mutagen import File
from mutagen.flac import FLAC
from mutagen.id3 import ID3
from mutagen.mp3 import MP3
from mutagen.mp4 import MP4

from metadata_fixer import MetadataFixer, _extract_id3_text
from milkdrop_renderer import MilkdropRenderer, RendererConfig, RendererSettings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Apply locale settings programmatically
try:
    locale.setlocale(locale.LC_ALL, "C")
except locale.Error:
    pass


def _get_default_screen_size() -> tuple[int, int] | None:
    """Return (width, height) of the default display's bounding box."""
    display = Gdk.Display.get_default()
    if display and hasattr(display, "get_geometry"):
        geometry = display.get_geometry()
        if geometry and geometry.width > 0 and geometry.height > 0:
            return geometry.width, geometry.height
    return None


COVER_ART_DEPENDENCIES = {}

COVER_ART_DEPENDENCIES["requests"] = {
    "available": True,
    "version": requests.__version__,
}

COVER_ART_DEPENDENCIES["pillow"] = {
    "available": True,
    "version": Image.__version__,
}

COVER_ART_DEPENDENCIES["mutagen"] = {
    "available": True,
    "version": getattr(
        mutagen, "__version__", getattr(mutagen, "version_string", "unknown")
    ),
}

COVER_ART_DEPENDENCIES["gtk"] = {
    "available": True,
    "version": getattr(Gtk, "get_major_version", lambda: 4)(),
}

COVER_ART_DEPENDENCIES["gdkpixbuf"] = {
    "available": True,
    "version": getattr(GdkPixbuf, "get_major_version", lambda: 0)(),
}

COVER_DOWNLOAD_AVAILABLE = True
COVER_ART_AVAILABLE = True

DEFAULT_VOLUME = 0.7
BEAT_DETECTION_INTERVAL = 50
BEAT_HISTORY_SIZE = 10
UPDATE_DISPLAY_INTERVAL = 100
AUTO_SAVE_DELAY = 2000
PERIODIC_SAVE_INTERVAL = 30000
AUDIO_EXTENSIONS = {
    ".mp3",
    ".mp4",
    ".flac",
    ".ogg",
    ".wav",
    ".m4a",
    ".wma",
    ".aac",
    ".opus",
}

MAX_FEED_CONTENT_SIZE = 5 * 1024 * 1024
MAX_IMAGE_CONTENT_SIZE = 10 * 1024 * 1024

RADIO_STATIONS_FILE = Path.home() / ".config" / "linamp" / "radio_stations.json"

DEFAULT_RADIO_STATIONS = [
    {
        "name": "AfterHours FM (AH.FM)",
        "genre": "Trance",
        "url": "http://198.244.201.21:8000/live",
        "logo": "https://ah.fm/wp-content/uploads/2023/03/transparent-white.png",
        "homepage": "https://ah.fm/",
    },
    {
        "name": "Party107",
        "genre": "Trance",
        "url": "https://party107.com/listen/party107-192.mp3",
        "logo": "https://www.party107.com/images/logo.jpg",
        "homepage": "https://www.party107.com/",
    },
    {
        "name": "Ganorium Radio",
        "genre": "Uplifting Trance",
        "url": "https://stream.ganorium.com",
        "logo": "https://ik.imagekit.io/8lskxw561nyg/ganoriumradio/wp-content/uploads/2024/06/ganorium-radio-2024-icon.png",
        "homepage": "https://www.ganorium.com/",
    },
    {
        "name": "SKellyWays Radio",
        "genre": "Trance",
        "url": "https://s31.ssl-stream.com:8200/radio.mp3",
        "logo": "https://skellywaysradio.com/SkellyWaysBadge.png",
        "homepage": "https://skellywaysradio.com/",
    },
    {
        "name": "SKellyWays Classic Trance",
        "genre": "Classic Trance",
        "url": "https://s31.ssl-stream.com:8210/radio.mp3",
        "logo": "https://skellywaysradio.com/SkellyWaysBadge.png",
        "homepage": "https://skellywaysradio.com/",
    },
    {
        "name": "SomaFM - The Trip",
        "genre": "Progressive Trance",
        "url": "https://ice5.somafm.com/thetrip-128-mp3",
        "logo": "https://somafm.com/logos/400/thetrip400.jpg",
        "homepage": "https://somafm.com/thetrip/",
    },
    {
        "name": "Intense Radio",
        "genre": "Dance / Trance",
        "url": "https://stream.intenseradio.net:8000/stream",
        "logo": "https://www.intenseradio.net/wp-content/uploads/2023/09/intense2017-logo3.png",
        "homepage": "https://www.intenseradio.net/",
    },
]


def load_radio_stations() -> list[dict[str, str]]:
    defaults_by_name = {s.get("name", ""): s for s in DEFAULT_RADIO_STATIONS}

    def _merge_defaults(stations: list) -> list[dict[str, str]]:
        """Fill missing logo/homepage fields from built-in defaults."""
        merged = []
        for s in stations:
            if not isinstance(s, dict):
                continue
            item = dict(s)
            base = defaults_by_name.get(item.get("name", ""), {})
            if not item.get("logo") and base.get("logo"):
                item["logo"] = base["logo"]
            if not item.get("homepage") and base.get("homepage"):
                item["homepage"] = base["homepage"]
            merged.append(item)
        return merged

    if not RADIO_STATIONS_FILE.exists():
        return DEFAULT_RADIO_STATIONS.copy()
    try:
        with open(RADIO_STATIONS_FILE, encoding="utf-8") as f:
            stations = json.load(f)
        if isinstance(stations, list) and stations:
            return _merge_defaults(stations)
    except (json.JSONDecodeError, OSError):
        pass
    return DEFAULT_RADIO_STATIONS.copy()


def save_radio_stations(stations: list[dict[str, str]]) -> bool:
    config_dir = RADIO_STATIONS_FILE.parent
    config_dir.mkdir(parents=True, exist_ok=True)
    try:
        with open(RADIO_STATIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(stations, f, indent=2, ensure_ascii=False)
        return True
    except OSError:
        return False


RADIO_LOGO_CACHE_DIR = Path.home() / ".cache" / "linamp" / "radio_logos"

# Podcast storage configuration
PODCASTS_CONFIG_DIR = Path.home() / ".config" / "linamp"
PODCASTS_FILE = PODCASTS_CONFIG_DIR / "podcasts.json"
PODCAST_EPISODES_DIR = PODCASTS_CONFIG_DIR / "podcast_episodes"
PODCAST_ART_CACHE_DIR = Path.home() / ".cache" / "linamp" / "podcast_art"
MAX_EPISODES_PER_FEED = 200


def _image_cache_path(url: str, cache_dir: Path) -> Path:
    key = hashlib.md5(url.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{key}.png"


def fetch_image_to_cache(
    url: str,
    cache_dir: Path,
    thumbnail_size: tuple[int, int] = (128, 128),
    timeout: float = 8.0,
) -> Path | None:
    """Download an image to the local cache and return its path."""
    if not url or not isinstance(url, str):
        return None
    if not url.startswith(("http://", "https://")):
        p = Path(url).expanduser()
        if p.is_file() and p.is_absolute():
            return p
        return None
    try:
        if not _validate_url(url):
            return None
    except (ValueError, ipaddress.AddressValueError):
        return None
    cache_path = _image_cache_path(url, cache_dir)
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 LinAmp/1.0"
            }
        )
        resp = session.get(url, timeout=timeout, verify=True, stream=True)
        if resp.status_code != 200:
            return None
        content = resp.raw.read(MAX_IMAGE_CONTENT_SIZE + 1)
        if len(content) > MAX_IMAGE_CONTENT_SIZE:
            return None
        img = Image.open(BytesIO(content))
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGBA")
        else:
            img = img.convert("RGB")
        img.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
        img.save(cache_path, format="PNG", optimize=True)
        return cache_path
    except (requests.RequestException, OSError, Image.UnidentifiedImageError) as e:
        logging.getLogger(__name__).debug(
            "Failed to fetch image from %s: %s", url, e
        )
        return None


def fetch_radio_logo(logo_url: str, timeout: float = 8.0) -> Path | None:
    return fetch_image_to_cache(logo_url, RADIO_LOGO_CACHE_DIR, (128, 128), timeout)


def _radio_logo_cache_path(logo_url: str) -> Path:
    return _image_cache_path(logo_url, RADIO_LOGO_CACHE_DIR)


def preload_radio_logos(stations: list[dict]) -> None:
    """Best-effort background preload of station logos into the cache."""

    def _worker() -> None:
        for s in stations:
            url = (s or {}).get("logo") or ""
            if url:
                try:
                    fetch_radio_logo(url)
                except Exception as e:
                    logging.getLogger(__name__).debug(
                        "Radio logo preload failed: %s", e
                    )

    threading.Thread(target=_worker, daemon=True, name="RadioLogoPreload").start()


# Podcast artwork caching
def fetch_podcast_artwork(artwork_url: str, timeout: float = 8.0) -> Path | None:
    return fetch_image_to_cache(artwork_url, PODCAST_ART_CACHE_DIR, (300, 300), timeout)


def _podcast_art_cache_path(artwork_url: str) -> Path:
    return _image_cache_path(artwork_url, PODCAST_ART_CACHE_DIR)


# RSS/Atom feed parsing
def parse_rss_feed(feed_url: str, content: str) -> PodcastFeed:
    """Parse RSS 2.0 feed content and return PodcastFeed."""
    if len(content) > MAX_FEED_CONTENT_SIZE:
        raise ValueError(f"Feed content from {feed_url} exceeds size limit")
    # Check if content is HTML instead of RSS/Atom XML
    if content.strip().lower().startswith(("<!doctype html", "<html")):
        # Try to extract RSS feed URL from HTML page
        import re

        rss_match = re.search(
            r'href=["\']([^"\']*\.xml|[^"\']*/feed|[^"\']*/rss[^"\']*)["\']',
            content,
            re.IGNORECASE,
        )
        if rss_match:
            suggested_url = rss_match.group(1)
            if suggested_url.startswith("/"):
                # Convert relative URL to absolute
                from urllib.parse import urljoin

                suggested_url = urljoin(feed_url, suggested_url)
            raise ValueError(
                f"The URL {feed_url} appears to be a webpage, not an RSS feed. "
                f"Try this RSS feed URL instead: {suggested_url}"
            )
        raise ValueError(
            f"The URL {feed_url} appears to be a webpage, not an RSS feed. "
            "Please provide the actual RSS feed URL (usually ends with /feed or .xml)"
        )

    try:
        root = DefusedET.fromstring(content)
    except ET.ParseError as e:
        # Try to recover by cleaning common XML issues
        logger = logging.getLogger(__name__)
        logger.warning("Failed to parse RSS feed from %s: %s", feed_url, e)

        # Try cleaning the content
        try:
            # Remove BOM if present
            content = content.removeprefix("\ufeff")
            import re

            content = re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", "", content)
            root = ET.fromstring(content)
        except ET.ParseError:
            logger.error("Failed to recover RSS feed parsing: %s", e)
            raise ValueError(f"Invalid RSS feed content from {feed_url}: {e}") from e

    # Handle both RSS and Atom namespaces
    if root.tag.endswith("rss"):
        channel = root.find("channel")
        if channel is None:
            raise ValueError("Invalid RSS feed: no channel element")

        title = _get_text(channel, "title") or ""
        author = (
            _get_text(channel, "itunes:author") or _get_text(channel, "author") or ""
        )
        description = _get_text(channel, "description") or ""
        artwork_url = (
            _get_text(channel, "itunes:image") or _get_text(channel, "image/url") or ""
        )
        homepage = _get_text(channel, "link") or ""

        episodes = []
        for item in channel.findall("item"):
            episode = _parse_rss_item(item)
            if episode:
                episodes.append(episode)

    elif root.tag.endswith("feed"):
        # Atom feed
        title = _get_text(root, "title") or ""
        author = _get_text(root, "author/name") or ""
        description = _get_text(root, "subtitle") or _get_text(root, "summary") or ""
        artwork_url = _get_text(root, "logo") or ""
        homepage = (
            _get_text(root, "link[@rel='alternate']") or _get_text(root, "link") or ""
        )

        episodes = []
        for entry in root.findall("entry"):
            episode = _parse_atom_entry(entry)
            if episode:
                episodes.append(episode)
    else:
        raise ValueError("Unknown feed format")

    feed_id = PodcastFeed.generate_feed_id(feed_url)
    last_updated = datetime.now(timezone.utc).isoformat()

    return PodcastFeed(
        feed_id=feed_id,
        title=title,
        author=author,
        description=description,
        artwork_url=artwork_url,
        homepage=homepage,
        feed_url=feed_url,
        last_updated=last_updated,
        episodes=episodes[:MAX_EPISODES_PER_FEED],
    )


def _get_text(element, tag: str) -> str:
    """Get text content from an element, handling namespaces."""
    # Try without namespace first
    child = element.find(tag)
    if child is not None and child.text:
        return child.text.strip()

    # Try with common namespaces
    namespaces = {
        "itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd",
        "atom": "http://www.w3.org/2005/Atom",
        "media": "http://search.yahoo.com/mrss/",
    }

    for ns in namespaces.values():
        child = element.find(f"{{{ns}}}{tag.split(':')[-1]}")
        if child is not None and child.text:
            return child.text.strip()

    return ""


def _parse_rss_item(item) -> PodcastEpisode | None:
    """Parse an RSS item into a PodcastEpisode."""
    guid = _get_text(item, "guid") or ""
    title = _get_text(item, "title") or ""
    description = (
        _get_text(item, "description") or _get_text(item, "itunes:summary") or ""
    )
    pub_date = _get_text(item, "pubDate") or ""

    # Parse duration
    duration_str = _get_text(item, "itunes:duration") or ""
    duration = _parse_duration(duration_str)

    # Get audio URL from enclosure or media:content
    audio_url = ""
    enclosure = item.find("enclosure")
    if enclosure is not None:
        audio_url = enclosure.get("url", "")
    if not audio_url:
        media_content = item.find("media:content")
        if media_content is not None:
            audio_url = media_content.get("url", "")

    # Get artwork from iTunes image or media:thumbnail
    artwork_url = _get_text(item, "itunes:image") or ""
    if not artwork_url:
        media_thumbnail = item.find("media:thumbnail")
        if media_thumbnail is not None:
            artwork_url = media_thumbnail.get("url", "")

    if not audio_url:
        return None

    return PodcastEpisode(
        guid=guid or audio_url,
        title=title,
        description=description,
        pub_date=pub_date,
        duration=duration,
        audio_url=audio_url,
        artwork_url=artwork_url,
    )


def _parse_atom_entry(entry) -> PodcastEpisode | None:
    """Parse an Atom entry into a PodcastEpisode."""
    guid = _get_text(entry, "id") or ""
    title = _get_text(entry, "title") or ""
    description = _get_text(entry, "summary") or _get_text(entry, "content") or ""
    pub_date = _get_text(entry, "published") or _get_text(entry, "updated") or ""

    # Get audio URL from enclosure or link
    audio_url = ""
    enclosure = entry.find("link[@rel='enclosure']")
    if enclosure is not None:
        audio_url = enclosure.get("href", "")
    if not audio_url:
        link = entry.find("link[@type='audio/mpeg']")
        if link is not None:
            audio_url = link.get("href", "")

    # Get duration from media:content
    duration = 0
    media_content = entry.find("media:content")
    if media_content is not None:
        duration_str = media_content.get("duration", "")
        duration = _parse_duration(duration_str)

    # Get artwork from media:thumbnail
    artwork_url = ""
    media_thumbnail = entry.find("media:thumbnail")
    if media_thumbnail is not None:
        artwork_url = media_thumbnail.get("url", "")

    if not audio_url:
        return None

    return PodcastEpisode(
        guid=guid or audio_url,
        title=title,
        description=description,
        pub_date=pub_date,
        duration=duration,
        audio_url=audio_url,
        artwork_url=artwork_url,
    )


def _parse_duration(duration_str) -> int:
    """Parse itunes:duration to seconds. Accepts seconds or H:MM:SS / M:SS."""
    if duration_str is None:
        return 0
    if isinstance(duration_str, (int, float)):
        return max(0, int(duration_str))
    text = str(duration_str).strip()
    if not text:
        return 0
    # Plain seconds: "3815"
    if text.isdigit():
        return int(text)
    # Timed: "1:03:35" or "3:35"
    parts = text.split(":")
    try:
        nums = [int(float(p)) for p in parts]
    except ValueError:
        return 0
    if len(nums) == 3:  # H:MM:SS
        return nums[0] * 3600 + nums[1] * 60 + nums[2]
    if len(nums) == 2:  # M:SS
        return nums[0] * 60 + nums[1]
    if len(nums) == 1:
        return nums[0]
    return 0


def fetch_feed_content(feed_url: str, timeout: float = 15.0) -> str:
    """Fetch feed content from URL using requests only."""
    if not _validate_url(feed_url):
        return ""
    try:
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 LinAmp/1.0"
            }
        )
        resp = session.get(feed_url, timeout=timeout, verify=True, stream=True)
        if resp.status_code == 200:
            chunks = []
            total = 0
            for chunk in resp.iter_content(chunk_size=8192):
                total += len(chunk)
                if total > MAX_FEED_CONTENT_SIZE:
                    return ""
                chunks.append(chunk)
            return b"".join(chunks).decode("utf-8", errors="replace")
    except requests.RequestException:
        return ""
    return ""


def sync_feed(feed_url: str, existing_feed: PodcastFeed | None = None) -> PodcastFeed:
    """Sync a podcast feed, merging with existing episodes to preserve playback state."""
    content = fetch_feed_content(feed_url)
    if not content:
        raise ValueError(f"Failed to fetch feed from {feed_url}")

    new_feed = parse_rss_feed(feed_url, content)

    # Merge with existing feed to preserve playback state
    if existing_feed and existing_feed.episodes:
        existing_episodes_by_guid = {ep.guid: ep for ep in existing_feed.episodes}

        for new_episode in new_feed.episodes:
            existing_episode = existing_episodes_by_guid.get(new_episode.guid)
            if existing_episode:
                # Preserve playback state
                new_episode.played = existing_episode.played
                new_episode.position = existing_episode.position
                new_episode.downloaded_path = existing_episode.downloaded_path

    return new_feed


# Podcast storage functions
def load_podcasts() -> list[PodcastFeed]:
    """Load all podcast feeds from storage."""
    if not PODCASTS_FILE.exists():
        return []

    try:
        with open(PODCASTS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [PodcastFeed.from_dict(feed_data) for feed_data in data]
    except (json.JSONDecodeError, OSError):
        pass

    return []


def save_podcasts(feeds: list[PodcastFeed]) -> bool:
    """Save all podcast feeds to storage."""
    PODCASTS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(PODCASTS_FILE, "w", encoding="utf-8") as f:
            json.dump(
                [feed.to_dict() for feed in feeds], f, indent=2, ensure_ascii=False
            )
        return True
    except OSError:
        return False


def load_feed_episodes(feed_id: str) -> list[PodcastEpisode]:
    """Load episodes for a specific feed from storage."""
    episodes_file = PODCAST_EPISODES_DIR / f"{feed_id}.json"
    if not episodes_file.exists():
        return []

    try:
        with open(episodes_file, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [PodcastEpisode.from_dict(ep_data) for ep_data in data]
    except (json.JSONDecodeError, OSError):
        pass

    return []


def save_feed_episodes(feed_id: str, episodes: list[PodcastEpisode]) -> bool:
    """Save episodes for a specific feed to storage."""
    PODCAST_EPISODES_DIR.mkdir(parents=True, exist_ok=True)
    episodes_file = PODCAST_EPISODES_DIR / f"{feed_id}.json"
    try:
        with open(episodes_file, "w", encoding="utf-8") as f:
            json.dump(
                [ep.to_dict() for ep in episodes], f, indent=2, ensure_ascii=False
            )
        return True
    except OSError:
        return False


def delete_feed_episodes(feed_id: str) -> bool:
    """Delete episodes file for a specific feed."""
    episodes_file = PODCAST_EPISODES_DIR / f"{feed_id}.json"
    try:
        if episodes_file.exists():
            episodes_file.unlink()
        return True
    except OSError:
        return False


def _create_icon_button(text: str, icon_name: str | None = None) -> Gtk.Button:
    button = Gtk.Button()
    if icon_name:
        content_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        icon = Gtk.Image.new_from_icon_name(icon_name)
        icon.set_pixel_size(14)
        content_box.append(icon)
        label = Gtk.Label(label=text)
        content_box.append(label)
        button.set_child(content_box)
    else:
        button.set_label(text)
    return button


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

    @staticmethod
    def validate_url(url: str) -> bool:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        hostname = parsed.hostname
        if not hostname:
            return False
        return hostname.lower() not in ("localhost", "127.0.0.1", "::1")


# Podcast data classes
@dataclass
class PodcastEpisode:
    guid: str = ""
    title: str = ""
    description: str = ""
    pub_date: str = ""
    duration: int = 0  # in seconds
    audio_url: str = ""
    artwork_url: str = ""
    played: bool = False
    position: int = 0  # playback position in seconds
    downloaded_path: str = ""  # local path if downloaded

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> PodcastEpisode:
        return cls(**data)


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
        episodes_data = data.pop("episodes", [])
        episodes = [PodcastEpisode.from_dict(ep) for ep in episodes_data]
        return cls(episodes=episodes, **data)

    @staticmethod
    def generate_feed_id(feed_url: str) -> str:
        return hashlib.md5(feed_url.encode("utf-8")).hexdigest()


COVER_ART_PATTERNS = [
    "cover.jpg",
    "cover.jpeg",
    "cover.png",
    "cover.webp",
    "cover.gif",
    "cover.bmp",
    "folder.jpg",
    "folder.jpeg",
    "folder.png",
    "folder.webp",
    "folder.gif",
    "folder.bmp",
    "album.jpg",
    "album.jpeg",
    "album.png",
    "album.webp",
    "album.gif",
    "album.bmp",
    "art.jpg",
    "art.jpeg",
    "art.png",
    "art.webp",
    "art.gif",
    "art.bmp",
    "front.jpg",
    "front.jpeg",
    "front.png",
    "front.webp",
    "front.gif",
    "front.bmp",
    ".cover.jpg",
    ".cover.jpeg",
    ".cover.png",
    ".cover.webp",
    ".cover.gif",
    ".cover.bmp",
]

SETTINGS_FILE = Path(__file__).parent / "linamp_settings.json"
COVER_CACHE_DIR = Path.home() / ".cache" / "linamp" / "covers"
LYRICS_CACHE_DIR = Path.home() / ".cache" / "linamp" / "lyrics"


@dataclass
class CoverArt:
    path: str = ""
    source: str = "embedded"
    width: int = 0
    height: int = 0
    format: str = "unknown"
    bytes_data: bytes | None = None

    @staticmethod
    def get_cache_key(audio_path: str) -> str:
        stat = os.stat(audio_path)
        file_data = f"{audio_path}:{stat.st_size}:{stat.st_mtime}"
        return hashlib.md5(file_data.encode()).hexdigest()[:16]

    @staticmethod
    def normalize_format(format_str: str) -> str:
        fmt = format_str.lower().strip(" .")
        if fmt in ("jpg", "jpeg"):
            return "jpg"
        if fmt in ("png", "gif", "webp", "bmp"):
            return fmt
        return "jpg"

    @classmethod
    def get_cache_path(cls, audio_path: str) -> Path | None:
        key = cls.get_cache_key(audio_path)
        cache_files = list(COVER_CACHE_DIR.glob(f"{key}.*"))
        if cache_files:
            return cache_files[0]
        return None

    def save_to_cache(self, audio_path: str) -> str | None:
        if not self.bytes_data:
            return None
        key = self.get_cache_key(audio_path)
        ext = self.normalize_format(self.format)
        cache_path = COVER_CACHE_DIR / f"{key}.{ext}"
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                f.write(self.bytes_data)
            return str(cache_path)
        except OSError:
            return None

    @classmethod
    def get_cache_path_for_audio(cls, audio_path: str) -> Path | None:
        return cls.get_cache_path(audio_path)

    @classmethod
    def get_cached_cover_path(cls, audio_path: str) -> Path | None:
        return cls.get_cache_path_for_audio(audio_path)

    @classmethod
    def load_from_cache(cls, audio_path: str) -> CoverArt | None:
        cache_path = cls.get_cache_path_for_audio(audio_path)
        if cache_path and cache_path.exists():
            with open(cache_path, "rb") as f:
                data = f.read()
            ext = cache_path.suffix.lower().lstrip(".")
            return cls(
                path=str(cache_path),
                source="cache",
                bytes_data=data,
                format=ext,
            )
        return None


def _tag_key_matches(key: str | tuple, prefix: str) -> bool:
    if isinstance(key, str):
        return key.startswith(prefix)
    if isinstance(key, tuple):
        return any(isinstance(item, str) and item.startswith(prefix) for item in key)
    return str(key).startswith(prefix)


def _extract_cover_art_from_file(file_path: str) -> CoverArt | None:
    audio = File(file_path)
    if audio is None:
        return None

    def _make_cover(data: bytes, fmt: str = "jpg") -> CoverArt:
        return CoverArt(
            path="",
            source="embedded",
            bytes_data=data,
            format=CoverArt.normalize_format(fmt),
        )

    def _try_prefix_tags() -> CoverArt | None:
        if not hasattr(audio, "tags") or not audio.tags:
            return None
        for tag_name in ["APIC:", "APIC", "cover", "Cover", "albumart", "coverart"]:
            if tag_name.endswith(":"):
                for key in audio.tags:
                    if _tag_key_matches(key, tag_name):
                        tag = audio.tags[key]
                        if hasattr(tag, "data"):
                            return _make_cover(tag.data, getattr(tag, "mime", "jpg"))
                        elif hasattr(tag, "value"):
                            return _make_cover(tag.value)
            else:
                if tag_name in audio.tags:
                    tag = audio.tags[tag_name]
                    if hasattr(tag, "data"):
                        return _make_cover(tag.data, getattr(tag, "mime", "jpg"))
                    elif hasattr(tag, "value"):
                        return _make_cover(tag.value)
        return None

    def _try_covr() -> CoverArt | None:
        if not (hasattr(audio, "tags") and audio.tags and "covr" in audio.tags):
            return None
        covr_data = audio.tags["covr"]
        if isinstance(covr_data, list) and covr_data:
            cover_image = covr_data[0]
            if hasattr(cover_image, "data"):
                return _make_cover(
                    cover_image.data, getattr(cover_image, "mime", "jpg")
                )
            elif isinstance(cover_image, bytes):
                return _make_cover(cover_image)
            return None

    def _try_pictures() -> CoverArt | None:
        if hasattr(audio, "pictures") and audio.pictures:
            picture = audio.pictures[0]
            if hasattr(picture, "data") and picture.data:
                return _make_cover(picture.data, getattr(picture, "mime", "jpg"))
        return None

    def _try_alternatives() -> CoverArt | None:
        methods = [
            ("artwork", "artwork"),
            ("cover", "cover"),
            ("coverart", "coverart"),
            ("albumart", "albumart"),
            ("covers", "covers"),
            ("pictures", "pictures"),
            ("covr", "covr"),
        ]
        for attr, _key in methods:
            try:
                val = getattr(audio, attr, None)
                if val:
                    return _make_cover(val)
            except (AttributeError, TypeError, ValueError) as e:
                logging.getLogger(__name__).debug("Cover extraction failed: %s", e)
        return None

    for func in [_try_prefix_tags, _try_covr, _try_pictures, _try_alternatives]:
        result = func()
        if result:
            return result
    return None


def _find_cover_in_folder(audio_path: str) -> CoverArt | None:
    audio_dir = os.path.dirname(os.path.abspath(audio_path))
    if not os.path.isdir(audio_dir):
        return None
    for pattern in COVER_ART_PATTERNS:
        cover_path = os.path.join(audio_dir, pattern)
        if os.path.isfile(cover_path):
            try:
                with open(cover_path, "rb") as f:
                    data = f.read()
                ext = os.path.splitext(pattern)[1].lstrip(".")
                return CoverArt(
                    path=cover_path, source="folder", bytes_data=data, format=ext
                )
            except OSError:
                pass
    return None


def _validate_file_path(file_path: str, allow_symlinks: bool = False) -> bool:
    if not file_path or not isinstance(file_path, str):
        return False
    if "\0" in file_path:
        return False
    if not allow_symlinks and os.path.islink(file_path):
        return False
    return os.path.isfile(file_path)


def _validate_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    hostname = parsed.hostname
    if not hostname:
        return False
    hostname_lower = hostname.lower()
    if hostname_lower in ("localhost", "127.0.0.1", "::1"):
        return False
    try:
        if hostname[0].isdigit() or ":" in hostname:
            addr = ipaddress.ip_address(hostname)
            return not (
                addr.is_private
                or addr.is_link_local
                or addr.is_loopback
                or addr.is_reserved
                or addr.is_multicast
            )
    except (ValueError, ipaddress.AddressValueError):
        pass
    return True


def extract_cover_art(
    audio_path: str, allow_download: bool = False
) -> tuple[CoverArt | None, str | None]:
    if not _validate_file_path(audio_path):
        return None, None
    cached = CoverArt.load_from_cache(audio_path)
    if cached:
        return cached, cached.path if cached.path else None
    cover = _extract_cover_art_from_file(audio_path)
    if cover:
        cache_path = cover.save_to_cache(audio_path)
        if cache_path:
            cover.path = cache_path
        return cover, cache_path
    cover = _find_cover_in_folder(audio_path)
    if cover:
        cache_path = cover.save_to_cache(audio_path)
        if cache_path:
            cover.path = cache_path
        return cover, cache_path
    return cover, None


class CoverArtDownloader:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        )
        self.timeout = (3, 5)

    def get_metadata(self, file_path: str) -> tuple[str, str]:
        if not _validate_file_path(file_path):
            return "", ""
        audio_path = Path(file_path)
        if audio_path.suffix.lower() == ".mp3":
            audio = MP3(file_path, ID3=ID3)
            artist = audio.tags.get("TPE1", [""])[0] if audio.tags else ""
            album = audio.tags.get("TALB", [""])[0] if audio.tags else ""
        elif audio_path.suffix.lower() == ".flac":
            audio = FLAC(file_path)
            artist = audio.get("artist", [""])[0]
            album = audio.get("album", [""])[0]
        elif audio_path.suffix.lower() in [".m4a", ".mp4"]:
            audio = MP4(file_path)
            artist = audio.get("\xa9ART", [""])[0] if audio.get("\xa9ART") else ""
            album = audio.get("\xa9alb", [""])[0] if audio.get("\xa9alb") else ""
        else:
            return "", ""
        return artist.strip(), album.strip()

    def search_cover(self, artist: str, album: str) -> str | None:
        if not artist or not album:
            return None
        query = f"{artist} {album} album cover"
        sources = [
            self._search_deezer,
            self._search_lastfm,
        ]
        for source_func in sources:
            url = source_func(query, artist, album)
            if url:
                return url
            time.sleep(0.5)
        return None

    def _search_deezer(self, query: str, artist: str, album: str) -> str | None:
        search_url = f"https://api.deezer.com/search/album?q={quote(query)}"
        if not _validate_url(search_url):
            return None
        response = self.session.get(search_url, timeout=self.timeout, verify=True)
        if response.status_code == 200:
            data = response.json()
            if data.get("data") and len(data["data"]) > 0:
                album_data = data["data"][0]
                cover_url = (
                    album_data.get("cover_xl")
                    or album_data.get("cover_big")
                    or album_data.get("cover_medium")
                )
                if cover_url and _validate_url(cover_url):
                    return cover_url

    def _search_google_images(self, query: str, artist: str, album: str) -> str | None:
        search_url = f"https://www.google.com/search?tbm=isch&q={quote(query)}"
        if not _validate_url(search_url):
            return None
        response = self.session.get(search_url, timeout=self.timeout, verify=True)
        if response.status_code == 200:
            pattern = r'\["(https://[^"]+?\.(?:jpg|jpeg|png|webp))",\d+,\d+\]'
            matches = re.findall(pattern, response.text)
            if matches:
                for url in matches[:3]:
                    if self._is_likely_album_cover(url) and _validate_url(url):
                        return url

    def _search_lastfm(self, query: str, artist: str, album: str) -> str | None:
        search_url = f"https://www.last.fm/music/{quote(artist)}/{quote(album)}"
        if not _validate_url(search_url):
            return None
        response = self.session.get(search_url, timeout=self.timeout, verify=True)
        if response.status_code == 200:
            pattern = (
                r'https://lastfm\.freetls\.fastly\.net/i/u/[^"]+\.(?:jpg|jpeg|png)'
            )
            matches = re.findall(pattern, response.text)
            if matches:
                valid_matches = [url for url in matches if _validate_url(url)]
                if valid_matches:
                    return max(valid_matches, key=len)

    def _is_likely_album_cover(self, url: str) -> bool:
        exclude_patterns = ["icon", "logo", "avatar", "thumb", "small", "tiny"]
        url_lower = url.lower()
        for pattern in exclude_patterns:
            if pattern in url_lower:
                return False
        return True

    def download_cover(self, url: str) -> bytes | None:
        if not _validate_url(url):
            return None
        response = self.session.get(url, timeout=self.timeout, verify=True, stream=True)
        if response.status_code != 200:
            return None
        content = response.raw.read(MAX_IMAGE_CONTENT_SIZE + 1)
        if len(content) > MAX_IMAGE_CONTENT_SIZE:
            return None
        img = Image.open(BytesIO(content))
        if img.mode in ("RGBA", "LA", "P"):
            background = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
            img = background
        img.thumbnail((500, 500), Image.Resampling.LANCZOS)
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.1)
        output = BytesIO()
        img.save(output, format="JPEG", quality=85, optimize=True)
        return output.getvalue()


_downloader = None


def get_cover_downloader() -> CoverArtDownloader | None:
    global _downloader
    if _downloader is None:
        _downloader = CoverArtDownloader()
    return _downloader


def download_cover_for_audio(audio_path: str) -> CoverArt | None:
    downloader = get_cover_downloader()
    artist, album = downloader.get_metadata(audio_path)
    if not artist or not album:
        return None
    cover_url = downloader.search_cover(artist, album)
    if not cover_url:
        return None
    cover_data = downloader.download_cover(cover_url)
    if not cover_data:
        return None
    cover = CoverArt(path="", source="downloaded", bytes_data=cover_data, format="jpg")
    cache_path = cover.save_to_cache(audio_path)
    if cache_path:
        cover.path = cache_path
    return cover


class LyricsManager:
    LRCLIB_BASE = "https://lrclib.net/api"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "LinAmp/1.0"})

    def get_lyrics(
        self,
        audio_path: str,
        artist: str = "",
        title: str = "",
        album: str = "",
        duration: int = 0,
    ) -> tuple[str, str, bool]:
        audio_path = os.path.abspath(audio_path)
        cache_key = self._cache_key(audio_path, artist, title)
        cached = self._cache_get(cache_key)
        if cached:
            return cached["text"], cached["source"], cached.get("synced", False)

        text, source, is_synced = self._fetch_lyrics(
            audio_path, artist, title, album, duration
        )
        if text:
            self._cache_set(
                cache_key, {"text": text, "source": source, "synced": is_synced}
            )
        return text, source, is_synced

    def _fetch_lyrics(
        self, audio_path: str, artist: str, title: str, album: str, duration: int
    ) -> tuple[str, str, bool]:
        lrc_path = audio_path.rsplit(".", 1)[0] + ".lrc"
        if os.path.isfile(lrc_path):
            text, is_synced = self._read_lrc(lrc_path)
            if text:
                return text, "lrc-file", is_synced

        text, is_synced = self._get_embedded_lyrics(audio_path)
        if text:
            return text, "embedded", is_synced

        if self.session and artist and title:
            try:
                text, is_synced = self._get_lrclib_lyrics(
                    artist, title, album, duration
                )
                if text:
                    return text, "lrclib", is_synced
            except Exception as exc:
                logging.getLogger(__name__).debug("LRCLIB lyrics fetch failed: %s", exc)

        return "", "none", False

    def _read_lrc(self, path: str) -> tuple[str, bool]:
        try:
            with open(path, encoding="utf-8") as f:
                content = f.read()
            is_synced = bool(re.search(r"\[\d{2}:\d{2}", content))
            return content, is_synced
        except OSError:
            return "", False

    def _get_embedded_lyrics(self, audio_path: str) -> tuple[str, bool]:
        try:
            audio = File(audio_path)
            if audio is None or not getattr(audio, "tags", None):
                return "", False

            lyrics_text = ""
            is_synced = False

            if isinstance(audio, MP3):
                for frame_id in ("USLT", "SYLT"):
                    frame = audio.tags.get(frame_id)
                    if frame:
                        lyrics_text = self._get_id3_text(frame)
                        is_synced = frame_id == "SYLT"
                        break
            elif isinstance(audio, FLAC):
                for key in ("lyrics", "LYRICS", "unsyncedlyrics", "UNSYNCEDLYRICS"):
                    values = audio.get(key)
                    if values:
                        lyrics_text = str(values[0]) if values else ""
                        is_synced = key.lower() == "lyrics"
                        break
                if not lyrics_text:
                    for key in ("lyrics", "LYRICS", "unsyncedlyrics", "UNSYNCEDLYRICS"):
                        for tag in audio.tags.values():
                            if (
                                hasattr(tag, "key")
                                and tag.key
                                and tag.key.lower() == key.lower()
                            ):
                                lyrics_text = (
                                    str(tag.value) if hasattr(tag, "value") else ""
                                )
                                is_synced = key.lower() == "lyrics"
                                break
                        if lyrics_text:
                            break
            elif isinstance(audio, MP4):
                for key in ("\xa9lyr",):
                    values = audio.get(key)
                    if values:
                        lyrics_text = str(values[0]) if values else ""
                        break

            return lyrics_text.strip(), is_synced
        except Exception as exc:
            logging.getLogger(__name__).debug("Embedded lyrics read failed: %s", exc)
            return "", False

    def _get_id3_text(self, frame) -> str:
        return _extract_id3_text(frame)

    def _get_lrclib_lyrics(
        self, artist: str, title: str, album: str, duration: int
    ) -> tuple[str, bool]:
        params = {
            "track_name": title,
            "artist_name": artist,
        }
        if album:
            params["album_name"] = album
        if duration and 1 <= duration <= 3600:
            params["duration"] = duration

        response = self.session.get(
            f"{self.LRCLIB_BASE}/get",
            params=params,
            timeout=15,
        )
        if response.status_code == 404:
            return "", False
        try:
            data = response.json()
        except (ValueError, requests.exceptions.RequestException) as exc:
            logging.getLogger(__name__).debug("LRCLIB lyrics fetch failed: %s", exc)
            return "", False

        synced = data.get("syncedLyrics") or ""
        plain = data.get("plainLyrics") or ""
        text = synced or plain
        is_synced = bool(synced)
        return text, is_synced

    def _cache_key(self, audio_path: str, artist: str, title: str) -> str:
        base = f"{audio_path}:{artist}:{title}"
        return hashlib.md5(base.encode()).hexdigest()[:16]

    def _cache_get(self, key: str) -> dict | None:
        path = LYRICS_CACHE_DIR / f"{key}.json"
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    def _cache_set(self, key: str, data: dict) -> None:
        try:
            LYRICS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path = LYRICS_CACHE_DIR / f"{key}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except OSError:
            pass


@dataclass
class PlaylistItem:
    path: str
    title: str = ""
    duration: int = 0
    cover_art_path: str = ""
    id: str = ""

    def __post_init__(self) -> None:
        self.path = os.path.abspath(os.path.expanduser(str(self.path)))
        if not self.title:
            self.title = os.path.splitext(os.path.basename(self.path))[0]
        else:
            _, ext = os.path.splitext(self.title)
            if ext.lower() in AUDIO_EXTENSIONS:
                self.title = self.title[: -len(ext)]
        if not self.id:
            self.id = hashlib.md5(f"{self.path}:{self.title}".encode()).hexdigest()[:32]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "title": self.title,
            "duration": self.duration,
            "cover_art_path": self.cover_art_path,
            "id": self.id,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlaylistItem:
        if not isinstance(data, dict):
            raise TypeError("Input data must be a dictionary")
        item_data = data.copy()
        if "filename" in item_data and "path" not in item_data:
            item_data["path"] = item_data.pop("filename")
        if "path" not in item_data:
            raise ValueError("Playlist item must contain a 'path' or 'filename' field")
        return cls(**item_data)

    def exists(self) -> bool:
        return os.path.isfile(self.path)

    def get_display_name(self) -> str:
        return self.title if self.title else os.path.basename(self.path)

    def has_cover_art(self) -> bool:
        return bool(self.cover_art_path and os.path.exists(self.cover_art_path))


class CoverArtWidget(Gtk.Box):
    def __init__(self, player: Any, size: int = 150) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self.player = player
        self.size = size
        self.current_cover_path = None
        self.loading = False
        self.executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="CoverArtWidget"
        )
        self.set_size_request(size, size)
        self.set_halign(Gtk.Align.CENTER)
        self.set_valign(Gtk.Align.CENTER)
        self.image = Gtk.Image()
        self.image.set_size_request(size, size)
        self.image.set_halign(Gtk.Align.CENTER)
        self.image.set_valign(Gtk.Align.CENTER)
        self.image.set_icon_size(Gtk.IconSize.LARGE)
        self.append(self.image)
        self.show_placeholder()

    def show_placeholder(self) -> None:
        self.image.set_from_icon_name("folder-music")
        self.image.set_pixel_size(self.size // 2)
        self.loading = False

    def _create_texture_from_bytes(self, bytes_data: bytes) -> Gdk.Texture | None:
        """Create a Gdk.Texture from raw image bytes, scaled to self.size."""
        try:
            loader = GdkPixbuf.PixbufLoader()
            loader.write(bytes_data)
            loader.close()
            pixbuf = loader.get_pixbuf()
            if not pixbuf:
                return None
            width = pixbuf.get_width()
            height = pixbuf.get_height()
            if width > height:
                scale = self.size / width
            else:
                scale = self.size / height
            new_width = int(width * scale)
            new_height = int(height * scale)
            scaled = pixbuf.scale_simple(
                new_width, new_height, GdkPixbuf.InterpType.BILINEAR
            )
            data = scaled.save_to_bufferv("png", [], [])
            if not data:
                return None
            return Gdk.Texture.new_from_bytes(GLib.Bytes.new(data))
        except (GLib.GError, OSError, TypeError):
            return None

    def load_cover(self, cover_art: CoverArt | None, audio_path: str = "") -> None:
        if not cover_art:
            self.show_placeholder()
            return
        if cover_art.path and os.path.exists(cover_art.path):
            try:
                texture = Gdk.Texture.new_from_filename(cover_art.path)
                self.image.set_from_paintable(texture)
                self.image.set_size_request(self.size, self.size)
                self.image.set_halign(Gtk.Align.FILL)
                self.image.set_valign(Gtk.Align.FILL)
                self.current_cover_path = cover_art.path
            except GLib.GError:
                if cover_art.bytes_data:
                    texture = self._create_texture_from_bytes(cover_art.bytes_data)
                    if texture:
                        self.image.set_from_paintable(texture)
                        self.current_cover_path = audio_path
                    else:
                        self.show_placeholder()
                else:
                    self.show_placeholder()
        elif cover_art.bytes_data:
            texture = self._create_texture_from_bytes(cover_art.bytes_data)
            if texture:
                self.image.set_from_paintable(texture)
                self.current_cover_path = audio_path
            else:
                self.show_placeholder()
        else:
            self.show_placeholder()

    def update_settings(self) -> None:
        settings = self.player.settings
        if hasattr(settings, "cover_art_shape"):
            self._apply_shape()

    def _apply_shape(self) -> None:
        settings = self.player.settings
        if not hasattr(settings, "cover_art_shape"):
            return
        shape = settings.cover_art_shape.lower()
        for css_class in ["cover-square", "cover-circle", "cover-rounded"]:
            self.remove_css_class(css_class)
        if shape == "circle":
            self.add_css_class("cover-circle")
        elif shape == "rounded":
            self.add_css_class("cover-rounded")
        elif shape == "square":
            self.add_css_class("cover-square")
        self.queue_draw()

    def set_size(self, size: int) -> None:
        if size < 50:
            size = 50
        elif size > 800:
            size = 800
        self.size = size
        self.set_size_request(size, size)
        self.image.set_size_request(size, size)

    def load_cover_from_path(self, cover_path: str | Path) -> None:
        if not cover_path or not os.path.exists(cover_path):
            self.show_placeholder()
            return
        texture = Gdk.Texture.new_from_filename(str(cover_path))
        self.image.set_from_paintable(texture)
        self.image.set_size_request(self.size, self.size)
        self.image.set_halign(Gtk.Align.FILL)
        self.image.set_valign(Gtk.Align.FILL)
        self.current_cover_path = str(cover_path)

    def clear(self) -> None:
        self.current_cover_path = None
        self.show_placeholder()

    def cleanup(self) -> None:
        if hasattr(self, "executor") and self.executor:
            if not getattr(self.executor, "_shutdown", False):
                self.executor.shutdown(wait=False, cancel_futures=True)
            self.executor = None


@dataclass
class PlayerSettings:
    auto_play_next: bool = True
    shuffle_mode: bool = False
    repeat_mode: str = "none"
    beat_aware_enabled: bool = True
    beat_threshold: float = 0.1
    auto_download_covers: bool = False
    volume: float = 1.0
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

    def __post_init__(self) -> None:
        if self.equalizer_settings is None:
            self.equalizer_settings = [0.0] * 10

    def to_dict(self) -> dict[str, Any]:
        return {
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
        return cls(
            auto_play_next=data.get("auto_play_next", True),
            shuffle_mode=data.get("shuffle_mode", False),
            repeat_mode=data.get("repeat_mode", "none"),
            beat_aware_enabled=data.get("beat_aware_enabled", True),
            beat_threshold=data.get("beat_threshold", 0.1),
            auto_download_covers=data.get("auto_download_covers", False),
            volume=data.get("volume", 1.0),
            position=data.get("position", 0.0),
            window_size=tuple(data.get("window_size", (400, 300))),
            window_position=tuple(data.get("window_position", (100, 100))),
            equalizer_settings=data.get("equalizer_settings", [0.0] * 10),
            last_played_track=data.get("last_played_track", ""),
            last_played_position=data.get("last_played_position", 0.0),
            show_cover_art=data.get("show_cover_art", True),
            cover_art_size=data.get("cover_art_size", 650),
            cover_art_shape=data.get("cover_art_shape", "square"),
            milkdrop_auto_switch=data.get("milkdrop_auto_switch", True),
        )


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
            {"name": "Metal", "values": [7, 6, 5, 3, 1, -1, -2, -3, -3, -3]},
            {"name": "Acoustic", "values": [-2, -1, 0, 2, 4, 4, 3, 2, 1, 0]},
            {"name": "Vocal", "values": [-1, 0, 2, 3, 4, 3, 2, 1, 0, -1]},
            {"name": "Bass Boost", "values": [8, 7, 6, 4, 2, 0, -1, -2, -3, -4]},
            {"name": "Treble Boost", "values": [-4, -3, -2, 0, 2, 4, 6, 7, 8, 8]},
        ]
        try:
            with open(self.EQ_PRESETS_PATH, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return defaults
        loaded = []
        seen = set()
        for item in data:
            name = item.get("name", "")
            values = item.get("values", [])
            if len(values) != 10 or not name or name in seen:
                continue
            seen.add(name)
            loaded.append({"name": name, "values": [float(v) for v in values]})
        return loaded if loaded else defaults

    def load_saved_settings(self) -> None:
        if hasattr(self.player, "settings") and self.player.settings.equalizer_settings:
            for i, value in enumerate(self.player.settings.equalizer_settings[:10]):
                if i < len(self.band_scales):
                    scale, value_label = self.band_scales[i]
                    clamped_value = max(-12, min(12, value))
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
        if hasattr(self.player, "equalizer"):
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
                GLib.source_remove(self._eq_save_timer_id)

            def _debounced_save() -> bool:
                self._eq_save_timer_id = None
                if hasattr(self.player, "auto_save_settings"):
                    self.player.auto_save_settings()
                return False

            self._eq_save_timer_id = GLib.timeout_add(1000, _debounced_save)

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


class LyricsTab(Gtk.Box):
    def __init__(self, player: Any) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self.player = player
        self.add_css_class("lyrics-tab")
        self._lyrics_manager = LyricsManager()
        self._executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="LyricsTab"
        )
        self._current_audio_path = ""
        self._current_source = "none"
        self._is_synced = False
        self._fetch_in_progress = False

        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        main_box.set_margin_top(8)
        main_box.set_margin_bottom(8)
        main_box.set_margin_start(8)
        main_box.set_margin_end(8)
        self.append(main_box)

        toolbar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        toolbar.set_halign(Gtk.Align.CENTER)
        main_box.append(toolbar)

        self.source_label = Gtk.Label(label="Source: none")
        self.source_label.add_css_class("lyrics-source-label")
        self.source_label.set_halign(Gtk.Align.START)
        self.source_label.set_hexpand(True)
        toolbar.append(self.source_label)

        self.synced_badge = Gtk.Label(label="")
        self.synced_badge.add_css_class("lyrics-synced-badge")
        self.synced_badge.set_visible(False)
        toolbar.append(self.synced_badge)

        fetch_btn = _create_icon_button("Fetch Online", "network-workgroup-symbolic")
        fetch_btn.add_css_class("lyrics-action-btn")
        fetch_btn.connect("clicked", self._on_fetch_online)
        toolbar.append(fetch_btn)

        reload_btn = _create_icon_button("Reload", "view-refresh-symbolic")
        reload_btn.add_css_class("lyrics-action-btn")
        reload_btn.connect("clicked", self._on_reload)
        toolbar.append(reload_btn)

        scrolled = Gtk.ScrolledWindow()
        scrolled.set_hexpand(True)
        scrolled.set_vexpand(True)
        scrolled.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        main_box.append(scrolled)

        self.text_view = Gtk.TextView()
        self.text_view.set_editable(False)
        self.text_view.set_cursor_visible(False)
        self.text_view.set_wrap_mode(Gtk.WrapMode.WORD)
        self.text_view.set_left_margin(12)
        self.text_view.set_right_margin(12)
        self.text_view.set_top_margin(12)
        self.text_view.set_bottom_margin(12)
        self.text_view.add_css_class("lyrics-text-view")
        scrolled.set_child(self.text_view)

        self.buffer = self.text_view.get_buffer()
        self._show_placeholder()

    def _show_placeholder(self) -> None:
        self.buffer.set_text("No lyrics loaded.\nPlay a track to load lyrics.")
        self.source_label.set_label("Source: none")
        self.synced_badge.set_visible(False)
        self._current_source = "none"
        self._is_synced = False

    def _on_fetch_online(self, button: Gtk.Button) -> None:
        if not self._current_audio_path:
            return
        item = self._get_current_item()
        if not item:
            return
        self._fetch_in_progress = True
        self.source_label.set_label("Source: fetching...")
        self.synced_badge.set_visible(False)

        def _task():
            artist = self._get_artist(item)
            title = self._clean_lyrics_title(
                item.title or os.path.splitext(os.path.basename(item.path))[0]
            )
            album = self._get_album(item)
            duration = self._get_duration(item)
            text, is_synced = self._lyrics_manager._get_lrclib_lyrics(
                artist, title, album, duration
            )
            source = "lrclib" if text else "none"
            cache_key = self._lyrics_manager._cache_key(item.path, artist, title)
            self._lyrics_manager._cache_set(
                cache_key,
                {"text": text, "source": source, "synced": is_synced},
            )
            return text, source, is_synced

        def _on_done(future):
            self._fetch_in_progress = False
            try:
                text, source, is_synced = future.result()
            except Exception as exc:
                logging.getLogger(__name__).debug("Lyrics fetch failed: %s", exc)
                text, source, is_synced = "", "none", False
            self._current_source = source
            self._is_synced = is_synced
            self._update_display(text, source, is_synced)

        future = self._executor.submit(_task)
        future.add_done_callback(lambda f: GLib.idle_add(_on_done, f))

    def _on_reload(self, button: Gtk.Button) -> None:
        item = self._get_current_item()
        if not item:
            return
        self.load_lyrics(item)

    @staticmethod
    def _clean_lyrics_title(title: str) -> str:
        if not title:
            return ""
        title = os.path.splitext(title)[0]
        title = re.sub(r"^(\d+)[\s\.\-_]+", "", title)
        parts = re.split(r"\s*-\s*", title)
        parts = [p for p in parts if not re.match(r"^\d+$", p.strip())]
        if len(parts) > 2:
            return parts[-1].strip()
        if len(parts) == 2:
            last = parts[-1].strip()
            if re.search(r"\b(Vol\.?|Volume|Pt\.?|Part|Compilation|Anthology|Collection|Series)\b", last, re.IGNORECASE):
                return parts[0].strip()
        return title.strip()

    def load_lyrics(self, item: Any) -> None:
        self._current_audio_path = item.path
        self.source_label.set_label("Source: loading...")
        self.synced_badge.set_visible(False)
        self.buffer.set_text("Loading lyrics...")

        def _task():
            artist = self._get_artist(item)
            title = self._clean_lyrics_title(
                item.title or os.path.splitext(os.path.basename(item.path))[0]
            )
            album = self._get_album(item)
            duration = self._get_duration(item)
            return self._lyrics_manager._fetch_lyrics(
                item.path, artist, title, album, duration
            )

        def _on_done(future):
            try:
                text, source, is_synced = future.result()
            except Exception as exc:
                logging.getLogger(__name__).debug("Lyrics fetch failed: %s", exc)
                text, source, is_synced = "", "none", False
            self._current_source = source
            self._is_synced = is_synced
            self._update_display(text, source, is_synced)

        future = self._executor.submit(_task)
        future.add_done_callback(lambda f: GLib.idle_add(_on_done, f))

    def _update_display(self, text: str, source: str, is_synced: bool) -> None:
        self.source_label.set_label(f"Source: {source}")
        if is_synced:
            self.synced_badge.set_label("Synced")
            self.synced_badge.set_visible(True)
        else:
            self.synced_badge.set_visible(False)
        if text:
            self.buffer.set_text(text)
        else:
            self.buffer.set_text("No lyrics available for this track.")
            self.source_label.set_label("Source: none")

    def _get_current_item(self) -> Any:
        if hasattr(
            self.player, "current_track"
        ) and 0 <= self.player.current_track < len(self.player.playlist):
            return self.player.playlist[self.player.current_track]
        return None

    def _get_artist(self, item: Any) -> str:
        try:
            audio = File(item.path)
            if audio is None:
                return ""
            if isinstance(audio, MP3):
                frame = audio.tags.get("TPE1") if audio.tags else None
                if frame:
                    text = getattr(frame, "text", None)
                    if text:
                        return str(text[0]).strip()
            elif isinstance(audio, FLAC):
                values = audio.get("artist")
                if values:
                    return str(values[0]).strip()
            elif isinstance(audio, MP4):
                values = audio.get("\xa9ART")
                if values:
                    return str(values[0]).strip()
        except Exception as exc:
            logging.getLogger(__name__).debug("Artist lookup failed: %s", exc)
        return ""

    def _get_album(self, item: Any) -> str:
        try:
            audio = File(item.path)
            if audio is None:
                return ""
            if isinstance(audio, MP3):
                frame = audio.tags.get("TALB") if audio.tags else None
                if frame:
                    text = getattr(frame, "text", None)
                    if text:
                        return str(text[0]).strip()
            elif isinstance(audio, FLAC):
                values = audio.get("album")
                if values:
                    return str(values[0]).strip()
            elif isinstance(audio, MP4):
                values = audio.get("\xa9alb")
                if values:
                    return str(values[0]).strip()
        except Exception as exc:
            logging.getLogger(__name__).debug("Album lookup failed: %s", exc)
        return ""

    def _get_duration(self, item: Any) -> int:
        duration = getattr(item, "duration", 0)
        if duration and duration > 0:
            return int(duration)
        try:
            audio = File(item.path)
            if (
                audio is not None
                and hasattr(audio, "info")
                and hasattr(audio.info, "length")
            ):
                return int(audio.info.length)
        except Exception as exc:
            logging.getLogger(__name__).debug("Duration lookup failed: %s", exc)
        return 0


class PlaylistTab(Gtk.Box):
    def __init__(self, player: Any) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self.player = player
        self.add_css_class("playlist-tab")
        self.playlist_store = Gtk.StringList()
        self.filtered_store = None
        self.filter_model = None
        self._search_debounce_id = None
        self._playlist_by_id = {}
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        main_box.set_margin_top(16)
        main_box.set_margin_bottom(16)
        main_box.set_margin_start(16)
        main_box.set_margin_end(16)
        self.append(main_box)
        search_section = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        main_box.append(search_section)
        search_header = Gtk.Label(label="Search Playlist")
        search_header.add_css_class("section-header")
        search_header.set_halign(Gtk.Align.START)
        search_section.append(search_header)
        search_container = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        search_section.append(search_container)
        self.search_entry = Gtk.Entry()
        self.search_entry.set_placeholder_text("Type to filter playlist...")
        self.search_entry.set_hexpand(True)
        self.search_entry.add_css_class("search-entry")
        self.search_entry.connect("changed", self.on_search_changed)
        search_container.append(self.search_entry)
        clear_search_btn = _create_icon_button("Clear", "edit-clear-symbolic")
        clear_search_btn.connect("clicked", self.on_clear_search)
        search_container.append(clear_search_btn)
        toolbar_section = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        main_box.append(toolbar_section)
        toolbar_header = Gtk.Label(label="Playlist Actions")
        toolbar_header.add_css_class("section-header")
        toolbar_header.set_halign(Gtk.Align.START)
        toolbar_section.append(toolbar_header)
        main_toolbar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        main_toolbar.set_homogeneous(True)
        toolbar_section.append(main_toolbar)
        add_btn = _create_icon_button("Add Files", "list-add-symbolic")
        add_btn.connect("clicked", self.on_add_files)
        main_toolbar.append(add_btn)
        add_folder_btn = _create_icon_button("Add Folder", "folder-open-symbolic")
        add_folder_btn.connect("clicked", self.on_add_folder)
        main_toolbar.append(add_folder_btn)
        remove_btn = _create_icon_button("Remove", "list-remove-symbolic")
        remove_btn.connect("clicked", self.on_remove)
        main_toolbar.append(remove_btn)
        clear_btn = _create_icon_button("Clear All", "edit-delete-symbolic")
        clear_btn.connect("clicked", self.on_clear)
        main_toolbar.append(clear_btn)
        shuffle_btn = _create_icon_button("Shuffle", "media-playlist-shuffle-symbolic")
        shuffle_btn.connect("clicked", self.on_shuffle)
        main_toolbar.append(shuffle_btn)
        secondary_toolbar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        secondary_toolbar.set_homogeneous(True)
        toolbar_section.append(secondary_toolbar)
        sort_dropdown = Gtk.DropDown.new_from_strings(
            ["Default Order", "By Title", "By Path", "By Duration"]
        )
        sort_dropdown.add_css_class("sort-dropdown")
        sort_dropdown.connect("notify::selected", self.on_sort_changed)
        secondary_toolbar.append(sort_dropdown)
        self.sort_dropdown = sort_dropdown
        import_btn = _create_icon_button("Import", "document-open-symbolic")
        import_btn.connect("clicked", self.on_import_m3u)
        secondary_toolbar.append(import_btn)
        export_btn = _create_icon_button("Export", "document-save-symbolic")
        export_btn.connect("clicked", self.on_export_m3u)
        secondary_toolbar.append(export_btn)
        remove_dups_btn = _create_icon_button("Remove Dups", "view-refresh-symbolic")
        remove_dups_btn.connect("clicked", self.on_remove_duplicates)
        secondary_toolbar.append(remove_dups_btn)
        fix_tags_btn = _create_icon_button(
            "Complete Tags", "document-edit-symbolic"
        )
        fix_tags_btn.set_tooltip_text(
            "Automatically complete missing ID3/metadata tags from filenames"
        )
        fix_tags_btn.connect("clicked", self.on_fix_tags)
        secondary_toolbar.append(fix_tags_btn)
        self.fix_tags_btn = fix_tags_btn
        stats_container = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        stats_container.set_halign(Gtk.Align.END)
        stats_container.set_margin_top(8)
        toolbar_section.append(stats_container)
        self.stats_label = Gtk.Label(label="0 tracks")
        self.stats_label.add_css_class("stats-label")
        stats_container.append(self.stats_label)
        playlist_container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        playlist_container.set_margin_top(12)
        main_box.append(playlist_container)
        playlist_header = Gtk.Label(label="Track List")
        playlist_header.add_css_class("section-header")
        playlist_header.set_halign(Gtk.Align.START)
        playlist_container.append(playlist_header)
        scrolled = Gtk.ScrolledWindow(
            hscrollbar_policy=Gtk.PolicyType.AUTOMATIC,
            vscrollbar_policy=Gtk.PolicyType.AUTOMATIC,
        )
        scrolled.add_css_class("playlist-scrolled")
        scrolled.set_hexpand(True)
        scrolled.set_vexpand(True)
        playlist_container.append(scrolled)
        self.sort_model = Gtk.SortListModel(model=self.playlist_store)
        self.filter_model = Gtk.FilterListModel(model=self.sort_model)
        self.selection_model = Gtk.SingleSelection(model=self.filter_model)
        self.column_view = Gtk.ColumnView(model=self.selection_model)
        self.column_view.add_css_class("playlist-view")
        self.column_view.set_hexpand(True)
        self.column_view.set_vexpand(True)
        self.column_view.set_size_request(-1, 180)
        factory = Gtk.SignalListItemFactory()
        factory.connect("setup", self._on_factory_setup)
        factory.connect("bind", self._on_factory_bind)
        column = Gtk.ColumnViewColumn(title="Tracks", factory=factory)
        self.column_view.append_column(column)
        scrolled.set_child(self.column_view)
        self.selection_model.connect("selection-changed", self.on_selection_changed)
        self.column_view.connect("activate", self.on_row_activated)

    def on_add_files(self, button):
        dialog = Gtk.FileDialog(title="Add Files")
        dialog.set_modal(True)
        audio_filter = Gtk.FileFilter()
        audio_filter.set_name("Audio files")
        audio_filter.add_mime_type("audio/*")
        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(audio_filter)
        dialog.set_filters(filters)
        dialog.open_multiple(self.player, None, self._on_files_selected)

    def _on_files_selected(self, dialog, result):
        files = dialog.open_multiple_finish(result)
        file_paths = [file.get_path() for file in files if file.get_path() is not None]
        if file_paths:
            self.player.add_to_playlist(file_paths)
        else:
            self.player.set_status_message("No files were selected.")

    def on_add_folder(self, button):
        dialog = Gtk.FileDialog(title="Add Music Folder")
        dialog.set_modal(True)
        dialog.select_folder(self.player, None, self._on_folder_selected)

    def _on_folder_selected(self, dialog, result):
        folder = dialog.select_folder_finish(result)
        if folder:
            folder_path = folder.get_path()
            if folder_path:
                self.player.add_folder_to_playlist(folder_path)
            else:
                self.player.set_status_message(
                    "Folder selection did not return a valid path."
                )

    def on_remove(self, button):
        position = self.selection_model.get_selected()
        if position != Gtk.INVALID_LIST_POSITION:
            item = self.selection_model.get_item(position)
            if item is not None:
                stored_string = item.get_string()
                if "|" in stored_string:
                    track_title, track_id = stored_string.split("|", 1)
                else:
                    track_title = stored_string
                    track_id = None
                    for playlist_item in self.player.playlist:
                        if (
                            playlist_item.title == track_title
                            or playlist_item.path.endswith(track_title)
                        ):
                            track_id = playlist_item.id
                            break
                    if track_id is None:
                        track_id = hashlib.md5(track_title.encode()).hexdigest()[:16]
                idx = self.player._get_playlist_index().get(track_id)
                if idx is not None:
                    self.player.playlist.pop(idx)
                self.player._invalidate_playlist_index()
                self.player._update_playlist_display()
                self.update_statistics()

    def on_clear(self, button):
        while self.playlist_store.get_n_items() > 0:
            self.playlist_store.remove(0)
        self.player.playlist.clear()
        self.player._invalidate_playlist_index()
        self.update_statistics()

    def _on_factory_setup(self, factory, list_item):
        label = Gtk.Label()
        label.set_halign(Gtk.Align.START)
        list_item.set_child(label)

    def _on_factory_bind(self, factory, list_item):
        label = list_item.get_child()
        item = list_item.get_item()
        if item is not None:
            text = item.get_string() if hasattr(item, "get_string") else str(item)
            if "|" in text:
                text = text.split("|", 1)[0]
            label.set_text(text)

    def on_selection_changed(self, selection, position, n_items):
        selected_pos = selection.get_selected()
        if selected_pos != Gtk.INVALID_LIST_POSITION:
            item = self.selection_model.get_item(selected_pos)
            if item is not None:
                stored_string = item.get_string()
                if "|" in stored_string:
                    track_title, track_id = stored_string.split("|", 1)
                else:
                    track_title = stored_string
                    track_id = None
                    for playlist_item in self.player.playlist:
                        if (
                            playlist_item.title == track_title
                            or playlist_item.path.endswith(track_title)
                        ):
                            track_id = playlist_item.id
                            break
                    if track_id is None:
                        track_id = hashlib.md5(track_title.encode()).hexdigest()[:16]
                idx = self.player._get_playlist_index().get(track_id)
                if idx is not None:
                    self.player.current_track = idx

    def on_search_changed(self, entry):
        search_text = (entry.get_text() or "").strip().lower()
        if self._search_debounce_id is not None:
            GLib.source_remove(self._search_debounce_id)
            self._search_debounce_id = None

        def apply_filter():
            if not search_text:
                self.filter_model.set_filter(None)
            else:
                filter_obj = Gtk.CustomFilter.new(
                    lambda item: self._filter_func(item, search_text)
                )
                self.filter_model.set_filter(filter_obj)
            self._search_debounce_id = None
            return False

        self._search_debounce_id = GLib.timeout_add(120, apply_filter)

    def _filter_func(self, item, search_text):
        if item is None:
            return False
        raw = item.get_string() if hasattr(item, "get_string") else str(item)
        title, track_id = (raw.split("|", 1) + [""])[:2] if "|" in raw else (raw, "")
        terms = [t for t in search_text.split() if t]
        if not terms:
            return True
        haystacks = [title.lower()]
        if track_id:
            pitem = self._playlist_by_id.get(track_id)
            if pitem is not None:
                haystacks.append((pitem.path or "").lower())
        for term in terms:
            if not any(term in h for h in haystacks):
                return False
        return True

    def on_clear_search(self, button):
        self.search_entry.set_text("")
        self.filter_model.set_filter(None)

    def on_shuffle(self, button):
        if self.player.playlist:
            random.shuffle(self.player.playlist)
            self.player._invalidate_playlist_index()
            self.player._update_playlist_display()
            self.player.save_playlist()

    def on_sort_changed(self, dropdown, pspec):
        selected = dropdown.get_selected()
        if selected == 1:
            self.player.playlist.sort(key=lambda x: x.title.lower())
        elif selected == 2:
            self.player.playlist.sort(key=lambda x: x.path.lower())
        elif selected == 3:
            self.player.playlist.sort(key=lambda x: x.duration)
        if selected != 0:
            self.player._invalidate_playlist_index()
            self.player._update_playlist_display()
            self.player.save_playlist()

    def update_statistics(self):
        total_tracks = len(self.player.playlist)
        by_id = {}
        total_duration = 0
        for item in self.player.playlist:
            if isinstance(item, PlaylistItem):
                if item.id:
                    by_id[item.id] = item
                dur = item.duration or 0
            elif isinstance(item, dict):
                tid = item.get("id")
                if tid:
                    by_id[tid] = PlaylistItem.from_dict(item)
                dur = item.get("duration", 0) or 0
            else:
                continue
            if dur > 0:
                total_duration += int(dur)
        self._playlist_by_id = by_id
        if total_duration > 0:
            hours = total_duration // 3600
            minutes = (total_duration % 3600) // 60
            if hours:
                duration_text = f"{hours}h {minutes}m"
            else:
                duration_text = f"{minutes}m"
            stats_text = f"{total_tracks} tracks • {duration_text}"
        else:
            stats_text = f"{total_tracks} tracks"
        self.stats_label.set_text(stats_text)

    def on_import_m3u(self, button):
        dialog = Gtk.FileDialog(title="Import M3U Playlist")
        dialog.set_modal(True)
        m3u_filter = Gtk.FileFilter()
        m3u_filter.set_name("M3U playlist files")
        m3u_filter.add_mime_type("audio/x-mpegurl")
        m3u_filter.add_pattern("*.m3u")
        m3u_filter.add_pattern("*.M3U")
        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(m3u_filter)
        dialog.set_filters(filters)
        dialog.open(self.player, None, self._on_m3u_import_selected)

    def _on_m3u_import_selected(self, dialog, result):
        file = dialog.open_finish(result)
        if file:
            filepath = file.get_path()
            if filepath:
                self._import_m3u_file(filepath)

    def _import_m3u_file(self, filepath):
        with open(filepath, encoding="utf-8") as f:
            lines = f.readlines()
        imported_files = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if not os.path.isabs(line):
                m3u_dir = os.path.dirname(filepath)
                line = os.path.join(m3u_dir, line)
            line = os.path.abspath(os.path.expanduser(line))
            if os.path.exists(line) and os.path.isfile(line):
                ext = os.path.splitext(line)[1].lower()
                if ext in AUDIO_EXTENSIONS:
                    imported_files.append(line)
        if imported_files:
            self.player.add_to_playlist(imported_files)

    def on_export_m3u(self, button):
        if not self.player.playlist:
            return
        dialog = Gtk.FileDialog(title="Export M3U Playlist")
        dialog.set_modal(True)
        m3u_filter = Gtk.FileFilter()
        m3u_filter.set_name("M3U playlist files")
        m3u_filter.add_mime_type("audio/x-mpegurl")
        m3u_filter.add_pattern("*.m3u")
        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(m3u_filter)
        dialog.set_filters(filters)
        dialog.set_initial_name("playlist.m3u")
        dialog.save(self.player, None, self._on_m3u_export_selected)

    def _on_m3u_export_selected(self, dialog, result):
        file = dialog.save_finish(result)
        if file:
            filepath = file.get_path()
            if filepath:
                if not filepath.lower().endswith(".m3u"):
                    filepath += ".m3u"
                self._export_m3u_file(filepath)

    def _export_m3u_file(self, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("#EXTM3U\n")
            for item in self.player.playlist:
                if hasattr(item, "path") and os.path.exists(item.path):
                    if (
                        hasattr(item, "duration")
                        and item.duration > 0
                        and hasattr(item, "title")
                    ):
                        f.write(f"#EXTINF:{item.duration},{item.title}\n")
                    f.write(f"{item.path}\n")

    def on_remove_duplicates(self, button):
        if not self.player.playlist:
            return
        seen_paths = set()
        duplicates_removed = 0
        unique_playlist = []
        for item in self.player.playlist:
            norm_path = os.path.realpath(item.path)
            if norm_path in seen_paths:
                duplicates_removed += 1
                continue
            seen_paths.add(norm_path)
            unique_playlist.append(item)
        if duplicates_removed > 0:
            self.player.playlist = unique_playlist
            self.player._invalidate_playlist_index()
            self.player._update_playlist_display()
            self.player.save_playlist()

    def on_row_activated(self, column_view, position):
        if hasattr(self, "filter_model") and self.filter_model.get_filter():
            item = self.selection_model.get_item(position)
            if item is not None:
                stored_string = item.get_string()
                if "|" in stored_string:
                    track_title, track_id = stored_string.split("|", 1)
                else:
                    track_title = stored_string
                    track_id = None
                    for playlist_item in self.player.playlist:
                        if (
                            playlist_item.title == track_title
                            or playlist_item.path.endswith(track_title)
                        ):
                            track_id = playlist_item.id
                            break
                    if track_id is None:
                        track_id = hashlib.md5(track_title.encode()).hexdigest()[:16]
                idx = self.player._get_playlist_index().get(track_id)
                if idx is not None:
                    self.player.play_track(idx)
        else:
            item = self.playlist_store.get_item(position)
            if item is not None:
                self.player.play_track(position)

    def on_fix_tags(self, button):
        if not hasattr(self.player, "playlist") or not self.player.playlist:
            self.player.set_status_message("No tracks in playlist to fix")
            return
        self._run_fix_tags()

    def _run_fix_tags(self):
        if not self.player.playlist:
            return
        file_paths = [
            item.path
            for item in self.player.playlist
            if hasattr(item, "path") and item.path and os.path.exists(item.path)
        ]
        if not file_paths:
            self.player.set_status_message("No valid files to fix")
            return
        self.player.set_status_message(
            f"Completing ID3 tags for {len(file_paths)} tracks..."
        )

        def fix_in_background():
            fixer = MetadataFixer()
            results = fixer.fix_files_threaded(file_paths)
            updated = sum(1 for r in results if r.updated_fields)
            failed = sum(1 for r in results if not r.success)
            skipped = len(results) - updated - failed
            GLib.idle_add(self._on_fix_tags_complete, updated, failed, skipped, results)

        thread = threading.Thread(target=fix_in_background, daemon=True)
        thread.start()

    def _on_fix_tags_complete(
        self, updated: int, failed: int, skipped: int, results: list
    ):
        if getattr(self, "_fix_tags_in_progress", False):
            return
        self._fix_tags_in_progress = True
        GLib.idle_add(
            self._on_fix_tags_complete_impl, updated, failed, skipped, results
        )

    def _on_fix_tags_complete_impl(
        self, updated: int, failed: int, skipped: int, results: list
    ):
        self._fix_tags_in_progress = False
        self.player.set_status_message(
            f"ID3 tags completed: {updated} updated, {failed} failed/skipped"
        )
        if hasattr(self.player, "_update_playlist_display"):
            self.player._update_playlist_display()

        current_path = None
        if self.player.current_track >= 0 and self.player.current_track < len(
            self.player.playlist
        ):
            current_item = self.player.playlist[self.player.current_track]
            if hasattr(current_item, "path"):
                current_path = current_item.path

        if current_path:
            for result in results:
                if (
                    result.success
                    and result.new_metadata
                    and result.file_path == current_path
                ):
                    new_title = result.new_metadata.get("title", "")
                    if new_title and hasattr(self.player, "player_tab"):
                        display = f"{new_title} ({Path(current_path).name})"
                        GLib.idle_add(
                            self.player.player_tab.track_label.set_text, display
                        )
                        GLib.idle_add(self.player.set_title, f"LinAmp - {new_title}")
                    break


class RadioTab(Gtk.Box):
    def __init__(self, player):
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self.player = player
        self.add_css_class("radio-tab")
        self.set_spacing(8)
        self.set_margin_top(8)
        self.set_margin_bottom(8)
        self.set_margin_start(8)
        self.set_margin_end(8)
        self._is_playing_radio = False
        self._radio_buffer_timer = 0
        self._stations = load_radio_stations()
        self._filtered_stations = []
        self._stream_tag_update_id = None
        # Warm logo cache in background so list rows paint quickly
        try:
            preload_radio_logos(self._stations)
        except Exception as e:
            logging.getLogger(__name__).debug("Radio logo preload failed: %s", e)

        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        header.set_halign(Gtk.Align.START)
        self.append(header)
        title_label = Gtk.Label(label="Internet Radio")
        title_label.add_css_class("section-header")
        title_label.set_halign(Gtk.Align.START)
        header.append(title_label)

        self.live_indicator = Gtk.Label(label="")
        self.live_indicator.add_css_class("live-indicator")
        self.live_indicator.set_halign(Gtk.Align.END)
        header.append(self.live_indicator)

        main_layout = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.append(main_layout)

        toolbar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        main_layout.append(toolbar)

        add_btn = _create_icon_button("Add Stream", "list-add-symbolic")
        add_btn.connect("clicked", self.on_add_stream)
        toolbar.append(add_btn)

        remove_btn = _create_icon_button("Remove", "list-remove-symbolic")
        remove_btn.connect("clicked", self.on_remove_station)
        toolbar.append(remove_btn)

        defaults_btn = _create_icon_button("Load Defaults", "document-open-symbolic")
        defaults_btn.connect("clicked", self.on_load_defaults)
        toolbar.append(defaults_btn)

        refresh_btn = _create_icon_button("Refresh", "view-refresh-symbolic")
        refresh_btn.connect("clicked", self.on_refresh)
        toolbar.append(refresh_btn)

        search_container = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        main_layout.append(search_container)

        self.search_entry = Gtk.Entry()
        self.search_entry.set_placeholder_text("Filter stations...")
        self.search_entry.set_hexpand(True)
        self.search_entry.add_css_class("search-entry")
        self.search_entry.connect("changed", self.on_search_changed)
        search_container.append(self.search_entry)

        self.genre_filter = Gtk.DropDown.new_from_strings(
            [
                "All Genres",
                "Electronic",
                "Jazz",
                "Classical",
                "Indie",
                "Alternative",
                "Rock",
                "Pop",
                "Lounge",
                "Mixed",
                "Varied",
            ]
        )
        self.genre_filter.add_css_class("sort-dropdown")
        self.genre_filter.set_selected(0)
        self.genre_filter.connect("notify::selected", self.on_genre_filter_changed)
        search_container.append(self.genre_filter)

        station_container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        station_container.set_margin_top(8)
        main_layout.append(station_container)

        station_header = Gtk.Label(label="Stations")
        station_header.add_css_class("section-header")
        station_header.set_halign(Gtk.Align.START)
        station_container.append(station_header)

        scrolled = Gtk.ScrolledWindow(
            hscrollbar_policy=Gtk.PolicyType.AUTOMATIC,
            vscrollbar_policy=Gtk.PolicyType.AUTOMATIC,
        )
        scrolled.add_css_class("radio-scrolled")
        scrolled.set_hexpand(True)
        scrolled.set_vexpand(True)
        station_container.append(scrolled)

        self.store = Gtk.StringList()
        self._populate_store()

        self.selection_model = Gtk.SingleSelection(model=self.store)
        self.column_view = Gtk.ColumnView(model=self.selection_model)
        self.column_view.add_css_class("radio-view")
        self.column_view.set_hexpand(True)
        self.column_view.set_vexpand(True)
        factory = Gtk.SignalListItemFactory()
        factory.connect("setup", self._on_station_factory_setup)
        factory.connect("bind", self._on_station_factory_bind)
        col = Gtk.ColumnViewColumn(title="Stations", factory=factory)
        self.column_view.append_column(col)
        self.column_view.connect("activate", self.on_station_activated)
        scrolled.set_child(self.column_view)
        self.selection_model.connect("selection-changed", self.on_selection_changed)

    def _populate_store(self):
        self.store.splice(0, self.store.get_n_items())
        self._filtered_stations = []
        filter_text = (
            self.search_entry.get_text().lower()
            if hasattr(self, "search_entry")
            else ""
        )
        genre_idx = (
            self.genre_filter.get_selected() if hasattr(self, "genre_filter") else 0
        )
        genre = (
            [
                "All Genres",
                "Electronic",
                "Jazz",
                "Classical",
                "Indie",
                "Alternative",
                "Rock",
                "Pop",
                "Lounge",
                "Mixed",
                "Varied",
            ][genre_idx]
            if genre_idx
            < len(
                [
                    "All Genres",
                    "Electronic",
                    "Jazz",
                    "Classical",
                    "Indie",
                    "Alternative",
                    "Rock",
                    "Pop",
                    "Lounge",
                    "Mixed",
                    "Varied",
                ]
            )
            else "All Genres"
        )
        for station in self._stations:
            name = station.get("name", "")
            station_genre = station.get("genre", "")
            url = station.get("url", "")
            if (
                filter_text
                and filter_text not in name.lower()
                and filter_text not in url.lower()
            ):
                continue
            if genre != "All Genres" and station_genre.lower() != genre.lower():
                continue
            display = f"{name}  ·  {station_genre}" if station_genre else name
            self.store.append(display)
            self._filtered_stations.append(station)

    def _on_station_factory_setup(self, factory, list_item):
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        row.set_margin_top(4)
        row.set_margin_bottom(4)
        row.set_margin_start(8)
        row.set_margin_end(8)

        logo = Gtk.Image()
        logo.set_size_request(40, 40)
        logo.set_pixel_size(40)
        logo.add_css_class("radio-station-logo")
        logo.set_from_icon_name("audio-x-generic")
        row.append(logo)

        text_col = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        text_col.set_hexpand(True)
        text_col.set_valign(Gtk.Align.CENTER)

        title_label = Gtk.Label(label="")
        title_label.set_hexpand(True)
        title_label.set_halign(Gtk.Align.START)
        title_label.set_ellipsize(Pango.EllipsizeMode.END)
        title_label.add_css_class("radio-station-name")
        text_col.append(title_label)

        genre_label = Gtk.Label(label="")
        genre_label.add_css_class("radio-station-genre")
        genre_label.set_halign(Gtk.Align.START)
        genre_label.set_ellipsize(Pango.EllipsizeMode.END)
        text_col.append(genre_label)

        row.append(text_col)

        play_btn = Gtk.Button()
        play_btn.add_css_class("flat")
        play_btn.set_tooltip_text("Play station")
        play_btn.set_valign(Gtk.Align.CENTER)
        play_img = Gtk.Image.new_from_icon_name("media-playback-start-symbolic")
        play_btn.set_child(play_img)
        play_btn.connect("clicked", self._on_station_play_clicked, list_item)
        row.append(play_btn)

        list_item.set_child(row)

    def _on_station_factory_bind(self, factory, list_item):
        item = list_item.get_item()
        if not item:
            return
        position = list_item.get_position()
        station_data = None
        if (
            position is not None
            and position >= 0
            and position < len(self._filtered_stations)
        ):
            station_data = self._filtered_stations[position]
        if not station_data:
            return

        row = list_item.get_child()
        if not row or not isinstance(row, Gtk.Box):
            return

        logo = row.get_first_child()
        text_col = logo.get_next_sibling() if logo else None
        title_label = text_col.get_first_child() if text_col else None
        genre_label = title_label.get_next_sibling() if title_label else None

        if title_label and isinstance(title_label, Gtk.Label):
            title_label.set_text(station_data.get("name", ""))
        if genre_label and isinstance(genre_label, Gtk.Label):
            genre_label.set_text(station_data.get("genre", ""))

        if logo and isinstance(logo, Gtk.Image):
            logo_url = station_data.get("logo", "")
            self._apply_station_logo(logo, logo_url)

    def _apply_station_logo(self, image_widget: Gtk.Image, logo_url: str) -> None:
        """Load a station logo into a Gtk.Image (cached; async-safe via idle)."""
        if not logo_url:
            image_widget.set_from_icon_name("audio-x-generic")
            return

        def _set_from_path(path_str: str) -> bool:
            try:
                texture = Gdk.Texture.new_from_filename(path_str)
                image_widget.set_from_paintable(texture)
                image_widget.set_size_request(40, 40)
            except Exception:
                image_widget.set_from_icon_name("audio-x-generic")
            return False

        # Local bundled / already-cached path
        local = Path(logo_url).expanduser()
        if local.is_file() and local.is_absolute():
            _set_from_path(str(local))
            return

        cache = _radio_logo_cache_path(logo_url)
        if cache.exists():
            _set_from_path(str(cache))
            return

        # Placeholder while downloading
        image_widget.set_from_icon_name("audio-x-generic")

        def _download():
            path = fetch_radio_logo(logo_url)
            if path and path.exists():
                GLib.idle_add(_set_from_path, str(path))

        threading.Thread(target=_download, daemon=True, name="RadioLogoFetch").start()

    def on_selection_changed(self, selection, position, n_items):
        pass  # Intentional: selection changes are handled by row activation

    def on_search_changed(self, entry):
        def apply_filter():
            self._populate_store()
            return False

        if self._search_debounce_id:
            GLib.source_remove(self._search_debounce_id)
        self._search_debounce_id = GLib.timeout_add(200, apply_filter)

    def on_genre_filter_changed(self, dropdown, pspec):
        self._populate_store()

    def on_station_activated(self, column_view, position):
        if position < 0 or position >= len(self._filtered_stations):
            return
        station = self._filtered_stations[position]
        self._play_station(station)

    def _on_station_play_clicked(self, button, list_item):
        position = list_item.get_position()
        if (
            position is not None
            and position >= 0
            and position < len(self._filtered_stations)
        ):
            self._play_station(self._filtered_stations[position])

    def _play_station(self, station):
        if not station or not station.get("url"):
            return
        url = station["url"]
        name = station.get("name", url)
        if not RadioStation.validate_url(url):
            self.player.set_status_message("Invalid stream URL")
            return
        if hasattr(self.player, "play_uri"):
            self.player.play_uri(url, name)
            self._is_playing_radio = True
            self._radio_start_time = time.time()
            self._update_live_indicator()
            GLib.idle_add(self._start_buffer_timer)
            # Show station logo on the player cover-art widget when available
            logo_url = station.get("logo") or ""
            if logo_url:

                def _show_logo():
                    path = fetch_radio_logo(logo_url)
                    cover = getattr(
                        getattr(self.player, "player_tab", None),
                        "cover_art_widget",
                        None,
                    )
                    if path and cover and hasattr(cover, "load_cover_from_path"):
                        GLib.idle_add(cover.load_cover_from_path, str(path))
                    return False

                threading.Thread(
                    target=_show_logo, daemon=True, name="RadioLogoPlay"
                ).start()

    def _start_buffer_timer(self):
        self._radio_buffer_timer = GLib.timeout_add(2000, self._check_buffer)
        return False

    def _check_buffer(self):
        if (
            self._is_playing_radio
            and hasattr(self.player, "playing")
            and self.player.playing
        ):
            self.live_indicator.set_text("")
            return True
        return False

    def _update_live_indicator(self):
        if not self._is_playing_radio:
            GLib.idle_add(self.live_indicator.set_text, "")
            return
        elapsed = (
            int(time.time() - self._radio_start_time)
            if hasattr(self, "_radio_start_time")
            else 0
        )
        mins = elapsed // 60
        secs = elapsed % 60
        GLib.idle_add(self.live_indicator.set_text, f"LIVE {mins}:{secs:02d}")
        self._live_timer_id = GLib.timeout_add(1000, self._update_live_indicator)

    def on_refresh(self, button):
        self._stations = load_radio_stations()
        self._populate_store()
        self.player.set_status_message("Radio stations refreshed")

    def on_load_defaults(self, button):
        self._stations = DEFAULT_RADIO_STATIONS.copy()
        save_radio_stations(self._stations)
        self._populate_store()
        self.player.set_status_message("Default stations loaded")

    def on_add_stream(self, button):
        dialog = Gtk.Dialog(title="Add Radio Stream", transient_for=self.get_root())
        dialog.set_modal(True)
        dialog.add_css_class("radio-dialog")

        content = dialog.get_content_area()
        content.set_spacing(8)
        content.set_margin_top(12)
        content.set_margin_bottom(12)
        content.set_margin_start(12)
        content.set_margin_end(12)

        name_entry = Gtk.Entry()
        name_entry.set_placeholder_text("Station name")
        content.append(name_entry)

        genre_entry = Gtk.Entry()
        genre_entry.set_placeholder_text("Genre (optional)")
        content.append(genre_entry)

        url_entry = Gtk.Entry()
        url_entry.set_placeholder_text("http:// or https:// stream URL")
        content.append(url_entry)

        dialog.add_button("Cancel", Gtk.ResponseType.CANCEL)
        dialog.add_button("Add", Gtk.ResponseType.OK)
        dialog.set_default_response(Gtk.ResponseType.OK)

        def on_response(dlg, response_id):
            if response_id == Gtk.ResponseType.OK:
                name = name_entry.get_text().strip()
                genre = genre_entry.get_text().strip()
                url = url_entry.get_text().strip()
                if not url or not RadioStation.validate_url(url):
                    self.player.set_status_message(
                        "Invalid URL — must start with http:// or https://"
                    )
                    dialog.destroy()
                    return
                if not name:
                    name = url
                self._stations.append({"name": name, "genre": genre, "url": url})
                save_radio_stations(self._stations)
                self._populate_store()
                self.player.set_status_message(f"Added station: {name}")
            dialog.destroy()

        dialog.connect("response", on_response)
        dialog.set_visible(True)

    def on_remove_station(self, button):
        selected = self.selection_model.get_selected()
        if selected is None:
            self.player.set_status_message("No station selected")
            return
        if selected < 0 or selected >= len(self._stations):
            return
        removed = self._stations.pop(selected)
        save_radio_stations(self._stations)
        self._populate_store()
        name = removed.get("name", "Unknown")
        self.player.set_status_message(f"Removed station: {name}")


class PodcastsTab(Gtk.Box):
    """GTK tab for podcast subscription management and episode playback."""

    def __init__(self, player):
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self.player = player

        # Load existing feeds
        self.feeds = load_podcasts()
        self.selected_feed = None
        self.episodes = []
        self._filtered_episodes = []
        self.sync_threads = {}

        # UI setup
        self._setup_ui()
        self._populate_feeds()

        # Add default NPR Up First feed if no feeds exist (deferred to avoid initialization issues)
        if not self.feeds:
            GLib.idle_add(self._add_default_feed)

    def _setup_ui(self):
        # Header
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

        # Main toolbar
        toolbar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        toolbar.set_margin_start(8)
        toolbar.set_margin_end(8)
        toolbar.set_margin_bottom(8)
        self.append(toolbar)

        add_btn = _create_icon_button("Add Feed", "list-add-symbolic")
        add_btn.connect("clicked", self.on_add_feed)
        toolbar.append(add_btn)

        remove_btn = _create_icon_button("Remove Feed", "list-remove-symbolic")
        remove_btn.connect("clicked", self.on_remove_feed)
        toolbar.append(remove_btn)

        sync_selected_btn = _create_icon_button(
            "Sync Selected", "view-refresh-symbolic"
        )
        sync_selected_btn.connect("clicked", self.on_sync_selected)
        toolbar.append(sync_selected_btn)

        sync_all_btn = _create_icon_button("Sync All", "emblem-synchronizing-symbolic")
        sync_all_btn.connect("clicked", self.on_sync_all)
        toolbar.append(sync_all_btn)

        # Split pane for feeds and episodes
        paned = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        paned.set_hexpand(True)
        paned.set_vexpand(True)
        paned.set_margin_start(8)
        paned.set_margin_end(8)
        paned.set_margin_bottom(8)
        self.append(paned)

        # Left pane - Feed list
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

        # Right pane - Episode list
        right_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        paned.set_end_child(right_box)

        episode_header = Gtk.Label(label="Episodes")
        episode_header.add_css_class("section-header")
        episode_header.set_halign(Gtk.Align.START)
        episode_header.set_margin_bottom(6)
        right_box.append(episode_header)

        # Episode filter
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

        # Episode actions toolbar
        episode_toolbar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        episode_toolbar.set_margin_top(6)
        right_box.append(episode_toolbar)

        mark_played_btn = _create_icon_button("Mark Played", "emblem-ok-symbolic")
        mark_played_btn.connect("clicked", self.on_mark_played)
        episode_toolbar.append(mark_played_btn)

        mark_unplayed_btn = _create_icon_button(
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
        # Parse the stored string format: "title|duration|played"
        parts = string_obj.get_string().split("|")
        if len(parts) >= 3:
            title, duration, played = parts[0], parts[1], parts[2] == "True"

            played_indicator = box.get_first_child()
            played_indicator.set_visible(played)

            label = played_indicator.get_next_sibling()
            label.set_text(title)

            duration_label = label.get_next_sibling()
            duration_label.set_text(duration)

    def _populate_feeds(self):
        self.feed_store.splice(0, self.feed_store.get_n_items())
        for feed in self.feeds:
            self.feed_store.append(feed.title)

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
        if selected is not None and selected >= 0 and selected < len(self.feeds):
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
        if (
            self.selected_feed
            and position >= 0
            and position < len(self._filtered_episodes)
        ):
            episode = self._filtered_episodes[position]
            if episode.audio_url and hasattr(self.player, "play_uri"):
                self.player.play_uri(episode.audio_url, episode.title)

                # Show episode artwork on player cover widget
                if episode.artwork_url:

                    def _show_artwork():
                        path = fetch_podcast_artwork(episode.artwork_url)
                        cover = getattr(
                            getattr(self.player, "player_tab", None),
                            "cover_art_widget",
                            None,
                        )
                        if path and cover and hasattr(cover, "load_cover_from_path"):
                            GLib.idle_add(cover.load_cover_from_path, str(path))
                        return False

                    threading.Thread(
                        target=_show_artwork, daemon=True, name="PodcastArtworkPlay"
                    ).start()

    def on_add_feed(self, button):
        dialog = Gtk.Dialog(title="Add Podcast Feed", transient_for=self.get_root())
        dialog.set_modal(True)
        dialog.add_css_class("radio-dialog")

        content = dialog.get_content_area()
        content.set_spacing(8)
        content.set_margin_top(12)
        content.set_margin_bottom(12)
        content.set_margin_start(12)
        content.set_margin_end(12)

        url_entry = Gtk.Entry()
        url_entry.set_placeholder_text("RSS or Atom feed URL")
        content.append(url_entry)

        title_entry = Gtk.Entry()
        title_entry.set_placeholder_text("Custom title (optional)")
        content.append(title_entry)

        dialog.add_button("Cancel", Gtk.ResponseType.CANCEL)
        dialog.add_button("Add & Sync", Gtk.ResponseType.OK)
        dialog.set_default_response(Gtk.ResponseType.OK)

        def on_response(dlg, response_id):
            if response_id == Gtk.ResponseType.OK:
                url = url_entry.get_text().strip()
                custom_title = title_entry.get_text().strip()

                if not url:
                    if hasattr(self.player, "status_label"):
                        self.player.set_status_message("URL is required")
                    dialog.destroy()
                    return

                # Add feed and sync
                self._add_and_sync_feed(url, custom_title)
            dialog.destroy()

        dialog.connect("response", on_response)
        dialog.set_visible(True)

    def _add_and_sync_feed(self, feed_url: str, custom_title: str = ""):
        # Don't start sync if window isn't fully initialized
        if not hasattr(self.player, "status_label"):
            return

        def _sync_worker():
            try:
                feed_id = PodcastFeed.generate_feed_id(feed_url)
                existing_feed = None
                for feed in self.feeds:
                    if feed.feed_id == feed_id:
                        existing_feed = feed
                        break

                new_feed = sync_feed(feed_url, existing_feed)

                if custom_title:
                    new_feed.title = custom_title

                # Update or add feed
                if existing_feed:
                    idx = self.feeds.index(existing_feed)
                    self.feeds[idx] = new_feed
                else:
                    self.feeds.append(new_feed)

                # Save feeds and episodes
                save_podcasts(self.feeds)
                save_feed_episodes(new_feed.feed_id, new_feed.episodes)

                # Update UI
                GLib.idle_add(self._populate_feeds)
                if hasattr(self.player, "status_label"):
                    GLib.idle_add(
                        self.player.set_status_message, f"Synced: {new_feed.title}"
                    )

            except (ValueError, URLError, OSError, ET.ParseError) as e:
                if hasattr(self.player, "status_label"):
                    GLib.idle_add(self.player.set_status_message, f"Sync failed: {e}")
            except Exception:
                # Catch any other exceptions (including requests.RequestException if available)
                if hasattr(self.player, "status_label"):
                    GLib.idle_add(
                        self.player.set_status_message, "Sync failed: unexpected error"
                    )

        thread = threading.Thread(target=_sync_worker, daemon=True, name="PodcastSync")
        thread.start()
        self.sync_threads[feed_url] = thread
        if hasattr(self.player, "status_label"):
            self.player.set_status_message("Syncing feed...")

    def on_remove_feed(self, button):
        selected = self.feed_selection.get_selected()
        if selected is None or selected < 0 or selected >= len(self.feeds):
            if hasattr(self.player, "status_label"):
                self.player.set_status_message("No feed selected")
            return

        feed = self.feeds[selected]

        # Remove from list
        self.feeds.pop(selected)
        save_podcasts(self.feeds)

        # Delete episode cache
        delete_feed_episodes(feed.feed_id)

        # Clear selection
        self.selected_feed = None
        self.episodes = []

        # Update UI
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
        def _sync_worker():
            try:
                existing_feed = feed
                new_feed = sync_feed(feed.feed_url, existing_feed)

                # Update feed in list
                idx = self.feeds.index(existing_feed)
                self.feeds[idx] = new_feed

                # Save feeds and episodes
                save_podcasts(self.feeds)
                save_feed_episodes(new_feed.feed_id, new_feed.episodes)

                # Update UI if this is the selected feed
                if (
                    self.selected_feed
                    and self.selected_feed.feed_id == new_feed.feed_id
                ):
                    self.episodes = new_feed.episodes
                    GLib.idle_add(self._populate_episodes)

                if hasattr(self.player, "status_label"):
                    GLib.idle_add(
                        self.player.set_status_message, f"Synced: {new_feed.title}"
                    )

            except (ValueError, URLError, OSError, ET.ParseError) as e:
                if hasattr(self.player, "status_label"):
                    GLib.idle_add(
                        self.player.set_status_message,
                        f"Sync failed for {feed.title}: {e}",
                    )
            except Exception:
                if hasattr(self.player, "status_label"):
                    GLib.idle_add(
                        self.player.set_status_message,
                        f"Sync failed for {feed.title}: unexpected error",
                    )

        thread = threading.Thread(target=_sync_worker, daemon=True, name="PodcastSync")
        thread.start()
        self.sync_threads[feed.feed_url] = thread

    def on_mark_played(self, button):
        selected = self.episode_selection.get_selected()
        if selected is None or selected < 0 or selected >= len(self.episodes):
            if hasattr(self.player, "status_label"):
                self.player.set_status_message("No episode selected")
            return

        episode = self.episodes[selected]
        episode.played = True
        episode.position = 0

        # Save episodes
        if self.selected_feed:
            save_feed_episodes(self.selected_feed.feed_id, self.episodes)

        self._populate_episodes()
        if hasattr(self.player, "status_label"):
            self.player.set_status_message(f"Marked as played: {episode.title}")

    def on_mark_unplayed(self, button):
        selected = self.episode_selection.get_selected()
        if selected is None or selected < 0 or selected >= len(self.episodes):
            if hasattr(self.player, "status_label"):
                self.player.set_status_message("No episode selected")
            return

        episode = self.episodes[selected]
        episode.played = False
        episode.position = 0

        # Save episodes
        if self.selected_feed:
            save_feed_episodes(self.selected_feed.feed_id, self.episodes)

        self._populate_episodes()
        if hasattr(self.player, "status_label"):
            self.player.set_status_message(f"Marked as unplayed: {episode.title}")

    def _add_default_feed(self):
        """Add NPR Up First as the default starter feed."""
        # Check if window is fully initialized before adding feed
        if not hasattr(self.player, "status_label"):
            return True  # Retry later
        npr_feed_url = "https://feeds.npr.org/510289/podcast.xml"
        self._add_and_sync_feed(npr_feed_url, "NPR Up First")
        return False  # Return False so GLib.idle_add doesn't call again


class EvoVisualizerTab(Gtk.Box):
    """GTK tab that hosts a Milkdrop-style visualizer driven by live GStreamer audio."""

    def __init__(self, player, available_presets=None):
        super().__init__(orientation=Gtk.Orientation.VERTICAL)

        self.player = player
        self.available_presets = available_presets or []

        # Visualizer state
        self.visualizer = None
        self.visualizer_type = None
        self._rendering_enabled = True

        # Audio probe state
        self.audio_probe_id = None
        self._probe_sink = None
        self._probe_pad = None
        self._probe_connecting = False
        self._probe_retry_id = None
        self._probe_retry_count = 0

        # Latest audio buffer (protected by lock)
        self._audio_lock = threading.Lock()
        self._latest_audio = None
        self._latest_audio_seq = 0
        self._last_consumed_audio_seq = 0
        self._audio_channels = 2
        self._audio_sample_rate = 44100.0
        self._audio_dt = 0.0
        self._last_audio_time = time.time()

        # Analysis state (beat / smoothing)
        self._beat_bass_avg = 0.0
        self._beat_bass_history = []
        self._last_beat_time = 0.0
        self._bass_smooth = 0.0
        self._mid_smooth = 0.0
        self._treb_smooth = 0.0
        self._beat_smooth = 0.0
        self._rms_smooth = 0.0
        self._peak_smooth = 0.0

        self._audio_consume_timer_id = None
        self._last_vis_update = 0.0

        # Widget setup
        self.set_visible(True)
        self.set_hexpand(True)
        self.set_vexpand(True)
        self.set_hexpand_set(True)
        self.set_vexpand_set(True)
        self.set_halign(Gtk.Align.FILL)
        self.set_valign(Gtk.Align.FILL)
        self.add_css_class("visualizer-widget")

        self.visualizer = MilkdropRenderer(
            auto_switch=self.player.settings.milkdrop_auto_switch
        )
        self.visualizer.set_hexpand(True)
        self.visualizer.set_vexpand(True)
        self.visualizer.set_hexpand_set(True)
        self.visualizer.set_vexpand_set(True)
        self.visualizer.set_halign(Gtk.Align.FILL)
        self.visualizer.set_valign(Gtk.Align.FILL)
        self.visualizer.set_visible(True)
        self.visualizer.add_css_class("visualizer-glarea")
        self.append(self.visualizer)
        self.visualizer_type = "evo"

        self.visualizer.available_milk_files = sorted(
            p for p in self.available_presets
        )

        self.connect("map", self._on_map)
        self.update_audio = self.visualizer.update_audio

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _on_map(self, widget):
        """Trigger a render once the widget is mapped and has a real size."""
        GLib.timeout_add(50, self._delayed_render_trigger)

    def _delayed_render_trigger(self):
        if self.visualizer is None:
            return False
        w = self.visualizer.get_allocated_width()
        h = self.visualizer.get_allocated_height()
        if w > 10 and h > 10:
            self.visualizer.queue_render()
        return False

    def cleanup(self):
        """Stop all audio probing and rendering. Safe to call multiple times."""
        self._rendering_enabled = False

        if self._audio_consume_timer_id is not None:
            GLib.source_remove(self._audio_consume_timer_id)
            self._audio_consume_timer_id = None

        if self._probe_retry_id is not None:
            GLib.source_remove(self._probe_retry_id)
            self._probe_retry_id = None

        if self.audio_probe_id is not None and self._probe_pad is not None:
            self._probe_pad.remove_probe(self.audio_probe_id)

        self.audio_probe_id = None
        self._probe_sink = None
        self._probe_pad = None
        self._probe_connecting = False
        self._probe_retry_count = 0

    # ------------------------------------------------------------------
    # Probe attachment
    # ------------------------------------------------------------------

    def _get_audio_sink_pad(self):
        """Return (pad, sink) preferring a post-conversion pad named 'vis_caps'."""
        player_obj = getattr(self.player, "player", self.player)
        if not hasattr(player_obj, "get_property"):
            return None, None

        audio_sink = player_obj.get_property("audio-sink")
        if not audio_sink:
            return None, None

        # Prefer the pad after format conversion (F32LE stereo)
        vis_caps = audio_sink.get_by_name("vis_caps")
        if vis_caps is not None:
            src = vis_caps.get_static_pad("src")
            if src:
                return src, audio_sink

        # Fallbacks
        sink_pad = audio_sink.get_static_pad("sink")
        if sink_pad:
            return sink_pad, audio_sink

        pads = audio_sink.get_pads()
        if pads:
            return pads[0], audio_sink

        return None, None

    def _reconnect_audio_probe(self):
        """Attach or re-attach the probe when the sink/pad changes."""
        if self._probe_connecting:
            return

        sink_pad, audio_sink = self._get_audio_sink_pad()
        if sink_pad is None or audio_sink is None:
            if self._probe_retry_id is None:
                self._probe_retry_count = 0
                self._probe_retry_id = GLib.timeout_add(250, self._retry_audio_probe)
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
            self._probe_retry_id = None
            self._probe_retry_count = 0

            self._audio_sample_rate = float(
                getattr(self.player, "_audio_sample_rate", 44100.0)
            )
            self._start_audio_consume_timer()
        finally:
            self._probe_connecting = False

    def _retry_audio_probe(self):
        """Bounded retry until the audio sink becomes available."""
        if self.audio_probe_id is not None:
            self._probe_retry_id = None
            return False

        self._probe_retry_count += 1
        if self._probe_retry_count > 40:
            self._probe_retry_id = None
            return False

        self._reconnect_audio_probe()
        if self.audio_probe_id is not None:
            self._probe_retry_id = None
            return False
        return True

    # ------------------------------------------------------------------
    # Probe callback → analysis → visualizer
    # ------------------------------------------------------------------

    def _milkdrop_audio_probe_callback(self, pad, info, user_data):
        """GStreamer pad probe: capture buffer and schedule analysis on the main loop."""
        if not self._rendering_enabled:
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
                self._latest_audio = audio_array.copy()
                self._audio_channels = channels
                self._audio_sample_rate = float(rate)
                self._latest_audio_seq = (self._latest_audio_seq + 1) & 0xFFFFFFFF

                now = time.time()
                self._audio_dt = now - self._last_audio_time
                self._last_audio_time = now
        finally:
            buffer.unmap(map_info)

        return Gst.PadProbeReturn.OK

    def _start_audio_consume_timer(self):
        """Start a periodic timer that consumes the latest audio buffer."""
        if self._audio_consume_timer_id is not None:
            return
        self._audio_consume_timer_id = GLib.timeout_add(33, self._consume_latest_audio)

    def _stop_audio_consume_timer(self):
        """Stop the periodic audio-consume timer."""
        if self._audio_consume_timer_id is not None:
            GLib.source_remove(self._audio_consume_timer_id)
            self._audio_consume_timer_id = None

    def _consume_latest_audio(self):
        """Periodic timeout callback: consume the latest audio buffer, compute
        levels, and push to the visualizer. Returns True to keep running."""
        if not self._rendering_enabled or self.visualizer is None:
            return True

        with self._audio_lock:
            seq = self._latest_audio_seq
            if seq == self._last_consumed_audio_seq:
                return True
            audio = self._latest_audio
            if audio is None:
                return True
            audio = audio.copy()
            rate = self._audio_sample_rate
            channels = self._audio_channels or 1
            dt = self._audio_dt
            self._last_consumed_audio_seq = seq

        levels = self._compute_audio_levels(audio, rate, channels, dt)
        self.visualizer.update_audio(levels)
        return True

    def _compute_audio_levels(self, audio, rate, channels, dt):
        """FFT-based band energy + beat detection + smoothing."""
        x = np.asarray(audio, dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        if x.size < 2:
            return self._empty_levels(dt)

        # Downmix to mono
        if channels > 1:
            frames = x.size // channels
            if frames < 1:
                return self._empty_levels(dt)
            xx = x[: frames * channels].reshape(frames, channels).mean(axis=1)
        else:
            xx = x

        nn = xx.size
        if nn < 2:
            return self._empty_levels(dt)

        # Windowed FFT
        fft_size = max(512, 1 << (nn - 1).bit_length())
        fft_size = min(fft_size, 16384)

        if nn >= fft_size:
            windowed = xx[-fft_size:] * np.hanning(fft_size).astype(np.float32)
        else:
            windowed = np.zeros(fft_size, dtype=np.float32)
            windowed[-nn:] = xx * np.hanning(nn).astype(np.float32)

        spec = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(fft_size, d=1.0 / rate)

        def band_energy(lo, hi):
            mask = (freqs >= lo) & (freqs < hi)
            if not np.any(mask):
                return 0.0
            return float(np.sqrt(np.mean(spec[mask] ** 2) + 1e-12))

        bass_e = band_energy(20.0, 250.0)
        mid_e = band_energy(250.0, 2000.0)
        treb_e = band_energy(2000.0, 8000.0)

        # Log compression + clamp
        bass = float(np.clip(np.log1p(bass_e * 10.0) * 2.5, 0.0, 2.0))
        mid = float(np.clip(np.log1p(mid_e * 10.0) * 2.5, 0.0, 2.0))
        treb = float(np.clip(np.log1p(treb_e * 10.0) * 2.5, 0.0, 2.0))

        rms = float(np.sqrt(np.mean(xx**2)))
        peak = float(np.max(np.abs(xx)))

        # Simple beat detector
        self._beat_bass_avg = self._beat_bass_avg * 0.9 + bass_e * 0.1
        self._beat_bass_history.append(bass_e)
        if len(self._beat_bass_history) > 50:
            self._beat_bass_history.pop(0)

        avg_bass = (
            float(np.mean(self._beat_bass_history)) if self._beat_bass_history else 0.0
        )
        beat = 0.0
        now = time.time()
        if bass > avg_bass * 1.35 and bass > 0.08 and now - self._last_beat_time > 0.18:
            beat = 1.0
            self._last_beat_time = now

        # Attack / decay smoothing
        attack, decay = 0.4, 0.08

        def smooth(prev, target, att=attack, dec=decay):
            if target > prev:
                return att * prev + (1.0 - att) * target
            return dec * prev + (1.0 - dec) * target

        self._bass_smooth = smooth(self._bass_smooth, bass)
        self._mid_smooth = smooth(self._mid_smooth, mid)
        self._treb_smooth = smooth(self._treb_smooth, treb)
        self._beat_smooth = smooth(self._beat_smooth, beat, att=0.5, dec=0.1)
        self._rms_smooth = smooth(self._rms_smooth, rms)
        self._peak_smooth = smooth(self._peak_smooth, peak)

        return {
            "bass": self._bass_smooth,
            "bass_att": self._bass_smooth,
            "mid": self._mid_smooth,
            "mid_att": self._mid_smooth,
            "treb": self._treb_smooth,
            "treb_att": self._treb_smooth,
            "beat": self._beat_smooth,
            "rms": self._rms_smooth,
            "peak": self._peak_smooth,
            "dt": dt,
        }

    @staticmethod
    def _empty_levels(dt):
        return {
            "bass": 0.0,
            "bass_att": 0.0,
            "mid": 0.0,
            "mid_att": 0.0,
            "treb": 0.0,
            "treb_att": 0.0,
            "beat": 0.0,
            "rms": 0.0,
            "peak": 0.0,
            "dt": dt,
        }

    # ------------------------------------------------------------------
    # Buffer → float32 conversion
    # ------------------------------------------------------------------

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
                audio = np.frombuffer(buf[:usable], dtype=np.float32)

            elif fmt == "F64LE":
                usable = (len(buf) // 8) * 8
                audio = np.frombuffer(buf[:usable], dtype=np.float64).astype(
                    np.float32, copy=False
                )

            elif fmt == "S16LE":
                usable = (len(buf) // 2) * 2
                audio = (
                    np.frombuffer(buf[:usable], dtype=np.int16).astype(
                        np.float32, copy=False
                    )
                    / 32768.0
                )

            elif fmt == "S32LE":
                usable = (len(buf) // 4) * 4
                audio = (
                    np.frombuffer(buf[:usable], dtype=np.int32).astype(
                        np.float32, copy=False
                    )
                    / 2147483648.0
                )

            elif fmt == "S24_32LE":
                usable = (len(buf) // 4) * 4
                ints = np.frombuffer(buf[:usable], dtype=np.int32)
                audio = (ints >> 8).astype(np.float32, copy=False) / 8388608.0

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
                # Sign-extend 24-bit
                ints = np.where(ints & 0x800000, ints - 0x1000000, ints)
                audio = ints.astype(np.float32, copy=False) / 8388608.0

            else:
                return None, rate, channels

        except (ValueError, BufferError):
            return None, rate, channels

        # Truncate to whole frames
        if channels > 1:
            remainder = audio.size % channels
            if remainder:
                audio = audio[:-remainder]

        return audio, rate, channels


class PlayerTab(Gtk.Box):
    def __init__(self, player):
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self.player = player
        self.render_lock = threading.RLock()
        self._frame_counter = 0
        self._last_beat_time = 0
        self._beat_times = []
        self.add_css_class("player-tab")
        self.set_spacing(8)
        self.set_margin_top(8)
        self.set_margin_bottom(8)
        self.set_margin_start(8)
        self.set_margin_end(8)
        top_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self.top_row = top_row
        self.append(top_row)
        self.cover_art_widget = CoverArtWidget(player, size=200)
        self.cover_art_widget.add_css_class("player-cover")
        top_row.append(self.cover_art_widget)
        if not player.settings.show_cover_art:
            self.cover_art_widget.set_visible(False)
        info_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        info_box.set_hexpand(True)
        top_row.append(info_box)
        self.track_label = Gtk.Label(label="No track playing")
        self.track_label.set_ellipsize(Pango.EllipsizeMode.END)
        self.track_label.add_css_class("track-label")
        self.track_label.set_halign(Gtk.Align.START)
        info_box.append(self.track_label)
        self.time_label = Gtk.Label(label="--:-- / --:--")
        self.time_label.add_css_class("time-label")
        self.time_label.set_halign(Gtk.Align.START)
        info_box.append(self.time_label)
        seek_adjustment = Gtk.Adjustment(
            value=0,
            lower=0,
            upper=100,
            step_increment=0.1,
            page_increment=1,
            page_size=0,
        )
        self.progress = Gtk.Scale(
            orientation=Gtk.Orientation.HORIZONTAL, adjustment=seek_adjustment
        )
        self.progress.set_hexpand(True)
        self.progress.set_size_request(-1, 20)
        self.progress.add_css_class("seek-bar")
        self.progress.set_draw_value(False)
        self.progress_controller = Gtk.GestureClick()
        self.progress_controller.connect("pressed", self.on_progress_pressed)
        self.progress.add_controller(self.progress_controller)
        info_box.append(self.progress)
        main_controls = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        main_controls.set_halign(Gtk.Align.CENTER)
        main_controls.set_homogeneous(True)
        self.append(main_controls)
        self.prev_btn = self._create_icon_button(
            "media-skip-backward-symbolic", "Previous"
        )
        self.play_btn = self._create_icon_button(
            "media-playback-start-symbolic", "Play"
        )
        self.pause_btn = self._create_icon_button(
            "media-playback-pause-symbolic", "Pause"
        )
        self.stop_btn = self._create_icon_button("media-playback-stop-symbolic", "Stop")
        self.next_btn = self._create_icon_button("media-skip-forward-symbolic", "Next")
        for btn in [
            self.prev_btn,
            self.play_btn,
            self.pause_btn,
            self.stop_btn,
            self.next_btn,
        ]:
            btn.add_css_class("control-button")
            btn.set_size_request(32, 32)
            main_controls.append(btn)
        self.play_btn.add_css_class("suggested-action")
        self.prev_btn.connect("clicked", self.player.on_prev)
        self.play_btn.connect("clicked", self.player.on_play)
        self.pause_btn.connect("clicked", self.player.on_pause)
        self.stop_btn.connect("clicked", self.player.on_stop)
        self.next_btn.connect("clicked", self.player.on_next)
        secondary_controls = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=2)
        secondary_controls.set_halign(Gtk.Align.CENTER)
        self.append(secondary_controls)
        self.repeat_btn = self._create_icon_button(
            "media-playlist-repeat-symbolic", "Repeat"
        )
        self.shuffle_btn = self._create_icon_button(
            "media-playlist-shuffle-symbolic", "Shuffle"
        )
        self.beat_btn = self._create_icon_button("process-working-symbolic", "Beat")
        self.download_btn = self._create_icon_button(
            "folder-download-symbolic", "Download"
        )
        self.autonext_btn = self._create_icon_button("go-next-symbolic", "Auto Next")
        self.cover_btn = self._create_icon_button(
            "image-x-generic-symbolic", "Cover"
        )
        for btn in [
            self.repeat_btn,
            self.shuffle_btn,
            self.beat_btn,
            self.download_btn,
            self.autonext_btn,
            self.cover_btn,
        ]:
            btn.add_css_class("icon-button")
            btn.set_size_request(24, 24)
            secondary_controls.append(btn)
        self.repeat_btn.connect("clicked", lambda b: self.player.toggle_repeat_mode())
        self.shuffle_btn.connect("clicked", lambda b: self.player.toggle_shuffle_mode())
        self.beat_btn.connect("clicked", lambda b: self.player.toggle_beat_aware())
        self.download_btn.connect(
            "clicked", lambda b: self.player.toggle_auto_download_covers()
        )
        self.autonext_btn.connect(
            "clicked", lambda b: self.player.toggle_auto_play_next()
        )
        self.cover_btn.connect(
            "clicked", lambda b: self.player.toggle_show_cover_art()
        )
        vol_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        vol_row.set_halign(Gtk.Align.CENTER)
        self.append(vol_row)
        self.volume_button = self._create_volume_button()
        self.volume_button.add_css_class("volume-button")
        self.volume_button.set_size_request(24, 24)
        vol_row.append(self.volume_button)
        lower, upper = 0, 100
        value = max(lower, min(upper, 70))
        vol_adjustment = Gtk.Adjustment(
            value=value,
            lower=lower,
            upper=upper,
            step_increment=1,
            page_increment=10,
            page_size=0,
        )
        self.volume_scale = Gtk.Scale(
            orientation=Gtk.Orientation.HORIZONTAL, adjustment=vol_adjustment
        )
        self.volume_scale.set_hexpand(True)
        self.volume_scale.set_draw_value(False)
        self.volume_scale.set_size_request(150, 20)
        self.volume_scale.connect("value-changed", self.player.on_volume_changed)
        vol_row.append(self.volume_scale)
        self.volume_label = Gtk.Label(label="70%")
        self.volume_label.add_css_class("volume-label")
        self.volume_label.set_size_request(35, -1)
        vol_row.append(self.volume_label)
        self.update_playback_state(False)

    def _create_volume_button(self):
        button = Gtk.Button()
        button.add_css_class("icon-button")
        button.set_tooltip_text("Mute/Unmute (M)")
        button.set_size_request(32, 28)
        button.connect("clicked", self.player.on_mute_toggle)
        self._update_volume_button_icon()
        return button

    def _update_volume_button_icon(self):
        if (
            not hasattr(self, "volume_button")
            or not hasattr(self, "player")
            or not self.player
        ):
            return
        player_obj = (
            self.player.player if hasattr(self.player, "player") else self.player
        )
        if not hasattr(player_obj, "get_property"):
            return
        current_volume = player_obj.get_property("volume")
        if current_volume == 0 or getattr(self.player, "_is_muted", False):
            icon_name = "audio-volume-muted-symbolic"
        elif current_volume < 0.3:
            icon_name = "audio-volume-low-symbolic"
        elif current_volume < 0.7:
            icon_name = "audio-volume-medium-symbolic"
        else:
            icon_name = "audio-volume-high-symbolic"
        icon = Gtk.Image.new_from_icon_name(icon_name)
        icon.set_pixel_size(16)
        self.volume_button.set_child(icon)

    def _create_icon_button(self, icon_name, tooltip_text):
        button = Gtk.Button()
        button.add_css_class("icon-button")
        button.set_tooltip_text(tooltip_text)
        button.set_size_request(36, 32)
        icon_image = Gtk.Image.new_from_icon_name(icon_name)
        icon_image.set_pixel_size(16)
        button.set_child(icon_image)
        return button

    def update_playback_state(self, is_playing):
        if is_playing:
            self.play_btn.set_visible(False)
            self.pause_btn.set_visible(True)
            self.add_css_class("playing")
        else:
            self.play_btn.set_visible(True)
            self.pause_btn.set_visible(False)
            self.remove_css_class("playing")

    def on_progress_pressed(self, gesture, n_press, x, y):
        width = self.progress.get_width()
        ratio = x / width if width > 0 else 0
        ratio = max(0.0, min(1.0, ratio))
        player_obj = self.player
        if hasattr(player_obj, "player"):
            player_obj = player_obj.player
        duration = player_obj.query_duration(Gst.Format.TIME)[1]
        if duration > 0:
            seek_time = int(ratio * duration)
            player_obj.seek(
                1.0,
                Gst.Format.TIME,
                Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT,
                Gst.SeekType.SET,
                seek_time,
                Gst.SeekType.NONE,
                0,
            )

    def update_track_info(self, title, artist="", album=""):
        if not title:
            title = "No track playing"
            artist = ""
        screen_size = _get_default_screen_size()
        if screen_size is not None:
            screen_width, screen_height = screen_size
            if hasattr(self, "get_surface") and self.get_surface():
                x, y = self.get_surface().get_position()
                max_x = screen_width - 450
                max_y = screen_height - 350
                if x > max_x or y > max_y or x < 0 or y < 0:
                    new_x = max(50, min(x, max_x))
                    new_y = max(50, min(y, max_y))
                    self.move(new_x, new_y)
        if artist and album:
            info_text = (
                f"<b>{GLib.markup_escape_text(title)}</b>\n"
                f"<small>{GLib.markup_escape_text(artist)} • {GLib.markup_escape_text(album)}</small>"
            )
        elif artist:
            info_text = (
                f"<b>{GLib.markup_escape_text(title)}</b>\n"
                f"<small>{GLib.markup_escape_text(artist)}</small>"
            )
        else:
            info_text = f"<b>{GLib.markup_escape_text(title)}</b>"
        self.track_label.set_markup(info_text)
        self.track_label.set_tooltip_text(
            f"{title}\n{artist}{' • ' + album if album else ''}"
        )
        window = self.get_root()
        if window:
            window.set_title(f"{title} - LinAmp")


class WinampWindow(Gtk.ApplicationWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)
        self.set_title("LinAmp")
        self.add_css_class("linamp-window")
        self.header_bar = Gtk.HeaderBar()
        self.header_bar.add_css_class("app-header")
        self.set_titlebar(self.header_bar)
        title_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        title_box.add_css_class("title-box")
        title_label = Gtk.Label(label="LinAmp")
        title_label.add_css_class("app-title")
        title_label.set_halign(Gtk.Align.START)
        self.header_nowplaying = Gtk.Label(label="Ready")
        self.header_nowplaying.add_css_class("app-subtitle")
        self.header_nowplaying.set_halign(Gtk.Align.START)
        self.header_nowplaying.set_ellipsize(Pango.EllipsizeMode.END)
        title_box.append(title_label)
        title_box.append(self.header_nowplaying)
        self.header_bar.set_title_widget(title_box)
        search_btn = Gtk.Button()
        search_btn.set_tooltip_text("Search playlist")
        search_btn.add_css_class("icon-button")
        search_btn.set_child(Gtk.Image.new_from_icon_name("system-search-symbolic"))
        search_btn.connect("clicked", lambda *_: self.focus_playlist_search())
        self.header_bar.pack_end(search_btn)
        self.set_resizable(True)
        self._active_timers = set()
        self.playlist = []
        self._playlist_index = {}
        self.current_track = -1
        self.shuffled_indices = []
        self.shuffle_position = 0
        self._is_muted = False
        self._previous_volume = 1.0
        self._playback_in_progress = False
        self._is_radio_stream = False
        self._radio_stream_uri = ""
        self._radio_stream_title = ""
        self._radio_start_time = 0
        self.beat_detection_timer = 0
        self._last_avg_level = 0.0
        self._last_beat_timestamp = 0.0
        self._bus_watch_id = None
        self._main_bus_signal_watch_attached = False
        self._internal_buses_signal_watches_attached = []
        self.is_compact_mode = False
        self.current_window_width = 650
        self.last_beat_time = 0
        self.beat_threshold = 0.1
        self.beat_interval_history = []
        self.settings = PlayerSettings()
        self.load_settings()
        self.auto_play_next = self.settings.auto_play_next
        if self.settings.window_size[0] > 0 and self.settings.window_size[1] > 0:
            self.set_default_size(*self.settings.window_size)
        self.shuffle_mode = self.settings.shuffle_mode
        self.repeat_mode = self.settings.repeat_mode
        self.beat_aware_enabled = self.settings.beat_aware_enabled
        self.beat_threshold = self.settings.beat_threshold
        self.auto_download_covers = self.settings.auto_download_covers
        self.milkdrop_auto_switch = self.settings.milkdrop_auto_switch
        self.connect("close-request", self.on_close_request)
        self.connect("destroy", self.on_window_destroy)
        self.connect("notify::default-width", self.on_window_size_changed)
        self.connect("notify::default-height", self.on_window_size_changed)
        self.connect("notify::width", self.on_window_size_changed)
        self.connect("notify::height", self.on_window_size_changed)

        def suppress_source_remove_warning(
            log_domain, log_level, message, user_data=None
        ):
            if "Source ID" in str(message) and "was not found" in str(message):
                return
            return

        GLib.log_set_handler(
            "GLib",
            GLib.LogLevelFlags.LEVEL_WARNING | GLib.LogLevelFlags.LEVEL_CRITICAL,
            suppress_source_remove_warning,
            None,
        )
        self.setup_player()
        self.apply_xmms_css()
        self.main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.main_box.add_css_class("main-container")
        self.main_box.set_hexpand(True)
        self.main_box.set_vexpand(True)
        self.main_box.set_halign(Gtk.Align.FILL)
        self.main_box.set_valign(Gtk.Align.FILL)
        self.main_box.set_hexpand_set(True)
        self.main_box.set_vexpand_set(True)
        self.set_child(self.main_box)
        self.notebook = Gtk.Notebook()
        self.notebook.add_css_class("main-notebook")
        self.notebook.set_hexpand(True)
        self.notebook.set_vexpand(True)
        self.notebook.set_halign(Gtk.Align.FILL)
        self.notebook.set_valign(Gtk.Align.FILL)
        self.notebook.set_hexpand_set(True)
        self.notebook.set_vexpand_set(True)
        self.notebook.set_scrollable(False)

        def on_critical_log(log_domain, log_level, message, user_data=None):
            if "CRITICAL" in str(log_level):
                pass

        GLib.log_set_handler(
            "Gtk", GLib.LogLevelFlags.LEVEL_CRITICAL, on_critical_log, None
        )
        self.main_box.append(self.notebook)
        self.player_tab = PlayerTab(self)
        self.equalizer_tab = EqualizerTab(self)
        self.playlist_tab = PlaylistTab(self)
        self.available_presets = []
        presets_dir = Path("./presets")
        _t0 = time.time()
        if presets_dir.exists():
            self.available_presets = [
                p for p in presets_dir.rglob("*.milk") if p.exists()
            ]
        _scan_dt = time.time() - _t0
        try:
            self.visualizer_tab = EvoVisualizerTab(self, self.available_presets)
        except Exception as e:
            self.logger.error("EvoVisualizerTab creation failed: %s", e)
            self.visualizer_tab = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
            self.visualizer_tab.add_css_class("visualizer-widget")
            error_label = Gtk.Label(label="Visualizer failed to load")
            error_label.add_css_class("placeholder-label")
            self.visualizer_tab.append(error_label)
        for tab in (self.player_tab, self.equalizer_tab, self.playlist_tab):
            tab.set_hexpand(True)
            tab.set_vexpand(True)
        self.notebook.append_page(
            self.player_tab,
            self._create_tab_widget("Player", "media-playback-start-symbolic"),
        )
        self.notebook.append_page(
            self.equalizer_tab,
            self._create_tab_widget("Equalizer", "audio-equalizer-symbolic"),
        )
        self.notebook.append_page(
            self.playlist_tab, self._create_tab_widget("Playlist", "view-list-symbolic")
        )
        self.notebook.append_page(
            self.visualizer_tab,
            self._create_tab_widget(
                "Visualizer", "preferences-desktop-wallpaper-symbolic"
            ),
        )
        self.notebook.set_current_page(0)
        self.notebook.set_hexpand(True)
        self.notebook.set_vexpand(True)
        self.notebook.set_halign(Gtk.Align.FILL)
        self.notebook.set_valign(Gtk.Align.FILL)
        self.notebook.set_hexpand_set(True)
        self.notebook.set_vexpand_set(True)
        self.radio_tab = RadioTab(self)
        self.radio_tab.set_hexpand(True)
        self.radio_tab.set_vexpand(True)
        self.radio_tab.set_halign(Gtk.Align.FILL)
        self.radio_tab.set_valign(Gtk.Align.FILL)
        radio_tab_widget = self._create_tab_widget("Radio", "radio-symbolic")
        self.notebook.append_page(self.radio_tab, radio_tab_widget)

        # Add Podcasts tab after Radio
        self.podcasts_tab = PodcastsTab(self)
        self.podcasts_tab.set_hexpand(True)
        self.podcasts_tab.set_vexpand(True)
        self.podcasts_tab.set_halign(Gtk.Align.FILL)
        self.podcasts_tab.set_valign(Gtk.Align.FILL)
        podcasts_tab_widget = self._create_tab_widget("Podcasts", "go-home-symbolic")
        self.notebook.append_page(self.podcasts_tab, podcasts_tab_widget)

        self.lyrics_tab = LyricsTab(self)
        self.lyrics_tab.set_hexpand(True)
        self.lyrics_tab.set_vexpand(True)
        self.lyrics_tab.set_halign(Gtk.Align.FILL)
        self.lyrics_tab.set_valign(Gtk.Align.FILL)
        lyrics_tab_widget = self._create_tab_widget("Lyrics", "audio-lyrics-symbolic")
        self.notebook.append_page(self.lyrics_tab, lyrics_tab_widget)

        def on_switch_page(notebook, page, page_num):
            if (
                page is self.player_tab
                and hasattr(self, "visualizer_tab")
                and hasattr(self.visualizer_tab, "cleanup")
            ):
                self.visualizer_tab.cleanup()
            if (
                page is self.visualizer_tab
                and hasattr(self, "visualizer_tab")
                and hasattr(self.visualizer_tab, "_rendering_enabled")
            ):
                self.visualizer_tab._rendering_enabled = True
                if hasattr(self.visualizer_tab, "_reconnect_audio_probe"):
                    self.visualizer_tab._reconnect_audio_probe()
            if (
                page is not self.radio_tab
                and hasattr(self, "radio_tab")
                and hasattr(self.radio_tab, "_is_playing_radio")
                and self.radio_tab._is_playing_radio
            ):
                self.radio_tab._is_playing_radio = False
                if (
                    hasattr(self.radio_tab, "_live_timer_id")
                    and self.radio_tab._live_timer_id
                ):
                    GLib.source_remove(self.radio_tab._live_timer_id)
                    self.radio_tab._live_timer_id = 0
                self.radio_tab.live_indicator.set_text("")
            if page is self.lyrics_tab and 0 <= self.current_track < len(self.playlist):
                self.lyrics_tab.load_lyrics(self.playlist[self.current_track])

        self.notebook.connect("switch-page", on_switch_page)
        GLib.idle_add(self.apply_responsive_layout)

        # --- Player controls ---
        self.player_tab.play_btn.connect("clicked", self.on_play)
        self.player_tab.pause_btn.connect("clicked", self.on_pause)
        self.player_tab.stop_btn.connect("clicked", self.on_stop)
        self.player_tab.prev_btn.connect("clicked", self.on_prev)
        self.player_tab.next_btn.connect("clicked", self.on_next)
        self.player_tab.volume_scale.connect("value-changed", self.on_volume_changed)
        self.player_tab.repeat_btn.connect(
            "clicked", lambda b: self.toggle_repeat_mode()
        )
        self.player_tab.shuffle_btn.connect(
            "clicked", lambda b: self.toggle_shuffle_mode()
        )
        self.player_tab.beat_btn.connect("clicked", lambda b: self.toggle_beat_aware())
        self.player_tab.download_btn.connect(
            "clicked", lambda b: self.toggle_auto_download_covers()
        )
        self.player_tab.autonext_btn.connect(
            "clicked", lambda b: self.toggle_auto_play_next()
        )

        # --- Status bar ---
        self.status_bar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        self.status_bar.set_size_request(-1, 28)
        self.status_bar.add_css_class("statusbar")
        self.status_label = Gtk.Label()
        self.status_label.set_halign(Gtk.Align.START)
        self.status_label.set_hexpand(True)
        self.status_label.set_margin_start(12)
        self.status_label.set_margin_end(12)
        self.status_bar.append(self.status_label)
        self.main_box.append(self.status_bar)

        # --- Final setup ---
        self.setup_drag_drop()
        timer1 = GLib.timeout_add(500, self._apply_equalizer_settings_delayed)

        self._active_timers.add(timer1)
        timer2 = GLib.timeout_add(1000, self._apply_ui_settings_delayed)
        self._active_timers.add(timer2)
        timer3 = GLib.timeout_add(UPDATE_DISPLAY_INTERVAL, self.update_display)
        self._active_timers.add(timer3)
        timer4 = GLib.timeout_add(PERIODIC_SAVE_INTERVAL, self.periodic_auto_save)
        self._active_timers.add(timer4)
        self.load_playlist()

    def on_window_size_changed(self, widget, pspec):
        width = widget.get_width()
        if width <= 0:
            width = self.current_window_width
        was_compact = self.is_compact_mode
        if width < 600:
            self.is_compact_mode = True
        else:
            self.is_compact_mode = False
        if was_compact != self.is_compact_mode or width != self.current_window_width:
            self.apply_responsive_layout()
        self.current_window_width = width

    def apply_responsive_layout(self):
        window_width = (
            self.current_window_width if hasattr(self, "current_window_width") else 600
        )
        if window_width <= 400:
            cover_size = 200
        elif window_width <= 500:
            cover_size = 100
        elif window_width <= 600:
            cover_size = 150
        else:
            cover_size = 250
        if self.is_compact_mode:
            if hasattr(self.player_tab, "volume_scale"):
                self.player_tab.volume_scale.set_size_request(120, -1)
            for btn in [
                self.player_tab.prev_btn,
                self.player_tab.play_btn,
                self.player_tab.pause_btn,
                self.player_tab.stop_btn,
                self.player_tab.next_btn,
            ]:
                btn.set_size_request(28, 24)
                btn.set_margin_start(2)
                btn.set_margin_end(2)
            if hasattr(self.playlist_tab, "search_entry"):
                self.playlist_tab.search_entry.set_margin_start(4)
                self.playlist_tab.search_entry.set_margin_end(4)
                self.playlist_tab.search_entry.set_size_request(-1, 32)
            if hasattr(self.player_tab, "cover_art_widget"):
                compact_size = max(60, cover_size // 2)
                self.player_tab.cover_art_widget.set_size(compact_size)
                cover_size = compact_size
            if hasattr(self.player_tab, "progress"):
                self.player_tab.progress.set_size_request(-1, 20)
        elif self.current_window_width < 800:
            if hasattr(self.player_tab, "volume_scale"):
                self.player_tab.volume_scale.set_size_request(160, -1)
            for btn in [
                self.player_tab.prev_btn,
                self.player_tab.play_btn,
                self.player_tab.pause_btn,
                self.player_tab.stop_btn,
                self.player_tab.next_btn,
            ]:
                btn.set_size_request(36, 30)
                btn.set_margin_start(4)
                btn.set_margin_end(4)
            if hasattr(self.playlist_tab, "search_entry"):
                self.playlist_tab.search_entry.set_margin_start(8)
                self.playlist_tab.search_entry.set_margin_end(8)
                self.playlist_tab.search_entry.set_size_request(-1, 36)
            if hasattr(self.player_tab, "cover_art_widget"):
                medium_size = max(100, cover_size - 30)
                self.player_tab.cover_art_widget.set_size(medium_size)
                cover_size = medium_size
            if hasattr(self.player_tab, "progress"):
                self.player_tab.progress.set_size_request(-1, 24)
        else:
            if hasattr(self.player_tab, "volume_scale"):
                self.player_tab.volume_scale.set_size_request(200, -1)
            for btn in [
                self.player_tab.prev_btn,
                self.player_tab.play_btn,
                self.player_tab.pause_btn,
                self.player_tab.stop_btn,
                self.player_tab.next_btn,
            ]:
                btn.set_size_request(44, 36)
                btn.set_margin_start(6)
                btn.set_margin_end(6)
            if hasattr(self.playlist_tab, "search_entry"):
                self.playlist_tab.search_entry.set_margin_start(16)
                self.playlist_tab.search_entry.set_margin_end(16)
                self.playlist_tab.search_entry.set_size_request(-1, 40)
            if self.current_window_width >= 1200:
                large_size = min(350, cover_size + 100)
            elif self.current_window_width >= 1000:
                large_size = min(300, cover_size + 50)
            else:
                large_size = cover_size
                if hasattr(self.player_tab, "cover_art_widget"):
                    cover_widget = self.player_tab.cover_art_widget
                    current_track = None
                    if (
                        hasattr(self, "current_track")
                        and 0 <= self.current_track < len(self.playlist)
                    ):
                        current_track = self.playlist[self.current_track]
                    if current_track is not None:
                        cached_cover_path = CoverArt.get_cached_cover_path(current_track.path)
                        if cached_cover_path and cached_cover_path.exists():
                            cover_widget.load_cover_from_path(cached_cover_path)
                    self.player_tab.cover_art_widget.set_size(large_size)
                    cover_size = large_size
        if hasattr(self.player_tab, "progress"):
            self.player_tab.progress.set_size_request(-1, 28)
        if self.is_compact_mode:
            self.main_box.add_css_class("compact-layout")
            if hasattr(self, "notebook"):
                self.notebook.add_css_class("compact-notebook")
        else:
            self.main_box.remove_css_class("compact-layout")
            if hasattr(self, "notebook"):
                self.notebook.remove_css_class("compact-notebook")

    def set_status_message(self, message):
        self.status_label.set_label(message)

    def _create_tab_widget(self, text, icon_name):
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        box.add_css_class("tab-widget")
        box.set_visible(True)
        icon = Gtk.Image.new_from_icon_name(icon_name)
        icon.set_pixel_size(16)
        icon.set_visible(True)
        box.append(icon)
        label = Gtk.Label(label=text)
        label.add_css_class("tab-label")
        box.append(label)
        return box

    def save_settings(self):
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(asdict(self.settings), f, indent=2)

    def load_settings(self):
        if SETTINGS_FILE.exists():
            with open(SETTINGS_FILE, encoding="utf-8") as f:
                content = f.read()
            if not content.strip():
                return
            settings_dict = json.loads(content)
            self.settings = PlayerSettings.from_dict(settings_dict)
            self._apply_settings_to_state()
            if (
                hasattr(self.settings, "last_played_track")
                and self.settings.last_played_track
            ):
                self._resume_last_played()

    def _invalidate_playlist_index(self) -> None:
        self._playlist_index = {}

    def _get_playlist_index(self) -> dict[str, int]:
        if (
            not hasattr(self, "_playlist_index")
            or not self._playlist_index
            or len(self._playlist_index) != len(self.playlist)
        ):
            self._playlist_index = {item.id: i for i, item in enumerate(self.playlist)}
        return self._playlist_index

    def _safe_source_remove(self, timer_id):
        if timer_id is None or timer_id == 0:
            return True
        if not hasattr(self, "_active_timers"):
            self._active_timers = set()
        if timer_id in self._active_timers:
            self._active_timers.discard(timer_id)
            try:
                GLib.source_remove(timer_id)
                return True
            except (TypeError, ValueError):
                return False
        else:
            return False

    def cleanup(self):
        if hasattr(self, "_cleanup_in_progress") and self._cleanup_in_progress:
            return
        self._cleanup_in_progress = True
        self._update_settings_from_state()
        self.save_settings()
        self.stop_beat_detection()
        if hasattr(self, "_active_timers"):
            for timer_id in list(self._active_timers):
                self._safe_source_remove(timer_id)
            self._active_timers.clear()
        if hasattr(self, "player") and self.player:
            bus = self.player.get_bus()
            if bus:
                if hasattr(self, "_bus_watch_id") and self._bus_watch_id is not None:
                    bus.disconnect(self._bus_watch_id)
                    self._bus_watch_id = None
                if self._main_bus_signal_watch_attached:
                    bus.remove_signal_watch()
                    self._main_bus_signal_watch_attached = False
                self.player.set_state(Gst.State.NULL)
        if hasattr(self, "equalizer") and self.equalizer:
            self.equalizer.set_state(Gst.State.NULL)
        if hasattr(self, "_internal_buses") and self._internal_buses:
            for internal_bus in self._internal_buses:
                if internal_bus in self._internal_buses_signal_watches_attached:
                    internal_bus.remove_signal_watch()
                    self._internal_buses_signal_watches_attached.remove(internal_bus)
            self._internal_buses = []
        if hasattr(self, "player") and self.player:
            self.player.set_state(Gst.State.NULL)
            self.player = None
        if hasattr(self, "equalizer") and self.equalizer:
            self.equalizer.set_state(Gst.State.NULL)
            self.equalizer = None
        if (
            hasattr(self, "cover_art_widget")
            and self.cover_art_widget
            and hasattr(self.cover_art_widget, "executor")
            and self.cover_art_widget.executor
        ):
            self.cover_art_widget.executor.shutdown(wait=False)
        self._cleanup_in_progress = False

    def on_window_destroy(self, *args):
        if hasattr(self, "full_cleanup") and not getattr(
            self, "_cleanup_in_progress", False
        ):
            self.full_cleanup()
        if self.get_application():
            self.get_application().quit()

    def resize_to_cover_art(self):
        if hasattr(self, "player_tab") and hasattr(self.player_tab, "cover_art_widget"):
            cover_widget = self.player_tab.cover_art_widget
            if cover_widget and hasattr(cover_widget, "size"):
                size = cover_widget.size
                padding = 280
                self.set_default_size(size + padding, size + padding)
                self.resize(size + padding, size + padding)

    def _resume_last_played(self):
        if not self.settings.last_played_track:
            return
        track_index = None
        for i, item in enumerate(self.playlist):
            if item.path == self.settings.last_played_track:
                track_index = i
                break
        if track_index is not None and self.play_track(track_index):
            if self.settings.last_played_position > 2.0:
                GLib.timeout_add(
                    1000, self._seek_to_position, self.settings.last_played_position
                )
        else:
            pass

    def _seek_to_position(self, position):
        if hasattr(self, "player") and self.player and self.playing:
            seek_pos = int(position * Gst.SECOND)
            self.player.seek_simple(Gst.Format.TIME, Gst.SeekFlags.FLUSH, seek_pos)
        return False

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
                if size.width > 0 and size.height > 0:
                    self.settings.window_size = (size.width, size.height)
        elif hasattr(self, "get_default_size"):
            size = self.get_default_size()
            if size.width > 0 and size.height > 0:
                self.settings.window_size = (size.width, size.height)
        if hasattr(self, "get_position"):
            self.settings.window_position = self.get_position()
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
        if hasattr(self, "move") and self.settings.window_position:
            x, y = self.settings.window_position
            screen_size = _get_default_screen_size()
            if screen_size is None:
                raise RuntimeError("Display geometry unavailable")
            screen_width, screen_height = screen_size
            max_x = screen_width - 450
            max_y = screen_height - 350
            x = max(50, min(x, max_x))
            y = max(50, min(y, max_y))
            self.move(x, y)
        if (
            hasattr(self, "equalizer")
            and self.equalizer
            and self.settings.equalizer_settings
        ):
            for i, value in enumerate(self.settings.equalizer_settings[:10]):
                self.equalizer.set_property("band" + str(i), value)
        if hasattr(self, "equalizer_tab") and self.settings.equalizer_settings:
            for i, value in enumerate(self.settings.equalizer_settings[:10]):
                if i < len(self.equalizer_tab.band_scales):
                    scale, value_label = self.equalizer_tab.band_scales[i]
                    clamped_value = max(-12, min(12, value))
                    scale.set_value(clamped_value)
                    value_label.set_text(f"{clamped_value:+.1f} dB")
        if (
            hasattr(self, "visualizer_tab")
            and hasattr(self.visualizer_tab, "visualizer")
            and self.visualizer_tab.visualizer is not None
        ):
            self.visualizer_tab.visualizer._settings = RendererSettings(
                renderer=RendererConfig(
                    preset_name=self.visualizer_tab.visualizer._settings.renderer.preset_name,
                ),
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
                    value_label.set_text(f"{value:+.1f} dB")
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

    def apply_xmms_css(self):
        css_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "theme", "gtk.css"
        )
        css_provider = Gtk.CssProvider()
        css_provider.load_from_path(css_file)
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(), css_provider, Gtk.STYLE_PROVIDER_PRIORITY_USER
        )

    def setup_player(self):
        self.player = Gst.ElementFactory.make("playbin", "player")
        if not self.player:
            GLib.idle_add(self.set_status_message, "GStreamer initialization failed")
            return
        self.playing = False
        self.equalizer = None
        self.audio_sink = None
        self.audio_convert = None
        self.audio_resample = None
        self._audio_sample_rate = 44100.0
        self._audio_channels = 2
        audio_sink = self._build_visualizer_audio_sink()
        if not audio_sink:
            audio_sink = Gst.ElementFactory.make("autoaudiosink", "audio_sink")
        if not audio_sink:
            audio_sink = Gst.ElementFactory.make("pulsesink", "audio_sink")
        if not audio_sink:
            audio_sink = Gst.ElementFactory.make("alsasink", "audio_sink")
        if audio_sink:
            self.player.set_property("audio-sink", audio_sink)
            self._setup_equalizer()
        self.player.set_property("volume", DEFAULT_VOLUME)
        self._song_finished_id = self.player.connect(
            "about-to-finish", self.on_song_finished
        )
        self._setup_player_bus()

    def _setup_equalizer(self):
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
        Falls back to ``None`` (caller falls back to a plain autoaudiosink) if any
        element is unavailable.
        """
        convert = Gst.ElementFactory.make("audioconvert", "vis_convert")
        resample = Gst.ElementFactory.make("audioresample", "vis_resample")
        capsfilter = Gst.ElementFactory.make("capsfilter", "vis_caps")
        sink = Gst.ElementFactory.make("autoaudiosink", "vis_sink")
        if not (convert and resample and capsfilter and sink):
            return None
        caps = Gst.Caps.from_string(
            "audio/x-raw, format=F32LE, rate=44100, channels=2, layout=interleaved"
        )
        capsfilter.set_property("caps", caps)
        audio_bin = Gst.Bin.new("vis_audio_sink")
        for element in (convert, resample, capsfilter, sink):
            audio_bin.add(element)
        if not (
            convert.link(resample)
            and resample.link(capsfilter)
            and capsfilter.link(sink)
        ):
            return None
        sink_pad = convert.get_static_pad("sink")
        audio_bin.add_pad(Gst.GhostPad.new("sink", sink_pad))
        self.audio_convert = convert
        self.audio_resample = resample
        self.audio_sink = audio_bin
        self._audio_sample_rate = 44100.0
        self._audio_channels = 2
        return audio_bin

    def _setup_player_bus(self):
        bus = self.player.get_bus()
        if bus:
            if self._main_bus_signal_watch_attached:
                bus.remove_signal_watch()
                self._main_bus_signal_watch_attached = False
            if hasattr(self, "_bus_watch_id") and self._bus_watch_id is not None:
                bus.disconnect(self._bus_watch_id)
                self._bus_watch_id = None
            bus.add_signal_watch()
            self._main_bus_signal_watch_attached = True
            self._bus_watch_id = bus.connect("message", self.on_bus_message)

    def cleanup_timers(self):
        if hasattr(self, "_active_timers"):
            for timer_id in list(self._active_timers):
                self._safe_source_remove(timer_id)
            self._active_timers.clear()
        timer_attributes = [
            "beat_detection_timer",
            "update_display_timer",
            "auto_save_timer",
        ]
        for timer_attr in timer_attributes:
            if hasattr(self, timer_attr):
                timer_id = getattr(self, timer_attr)
                if timer_id:
                    try:
                        GLib.source_remove(timer_id)
                    except GLib.Error:
                        pass
                    setattr(self, timer_attr, 0)

    def full_cleanup(self):
        if hasattr(self, "_cleanup_in_progress") and self._cleanup_in_progress:
            return
        self._cleanup_in_progress = True
        self.playing = False
        self.stop_beat_detection()
        if hasattr(self, "_active_timers"):
            for timer_id in list(self._active_timers):
                self._safe_source_remove(timer_id)
            self._active_timers.clear()
        if hasattr(self, "player") and self.player:
            self.player.set_state(Gst.State.NULL)
            self.player.get_state(Gst.CLOCK_TIME_NONE)
        if hasattr(self, "player") and self.player:
            self.cleanup_pipeline(self.player)
            self.player = None
        if hasattr(self, "equalizer") and self.equalizer:
            self.equalizer.set_state(Gst.State.NULL)
            self.equalizer = None
        if hasattr(self, "player_tab") and hasattr(self.player_tab, "cover_art_widget"):
            self.player_tab.cover_art_widget.cleanup()
        if hasattr(self, "_download_executor") and self._download_executor:
            try:
                self._download_executor.shutdown(wait=False, cancel_futures=True)
            except Exception as e:
                logging.getLogger(__name__).debug(
                    "Download executor shutdown failed: %s", e
                )
            self._download_executor = None
        self._cleanup_in_progress = False

    def on_close_request(self, window):
        self.full_cleanup()
        if self.get_application():
            self.get_application().quit()
        return True

    def setup_drag_drop(self):
        self.dnd = Gtk.DropTarget.new(Gio.File, Gdk.DragAction.COPY)
        self.dnd.connect("drop", self.on_file_dropped)
        self.add_controller(self.dnd)

    def on_file_dropped(self, drop, file, x, y):
        path = file.get_path()
        if path:
            self.add_to_playlist([path])
            if not self.playing and self.playlist:
                self.play_track(0)
        return True

    def _set_player_state_thread_safe(self, state):
        return self.player.set_state(state)

    def _set_player_property_thread_safe(self, property_name, value):
        self.player.set_property(property_name, value)
        return True

    def play_file(self, filepath, title=None):
        if not os.path.exists(filepath):
            return False
        if not hasattr(self, "player") or not self.player:
            return False
        if hasattr(self.player_tab, "track_label"):
            GLib.idle_add(
                self.player_tab.track_label.set_text,
                f"Loading: {os.path.basename(filepath)}...",
            )
        uri = Gst.filename_to_uri(os.path.abspath(filepath))
        self.player.set_state(Gst.State.NULL)
        self.player.set_property("uri", uri)
        state_change = self.player.set_state(Gst.State.PLAYING)
        if state_change == Gst.StateChangeReturn.FAILURE:
            GLib.idle_add(self.set_status_message, "Failed to start playback")
            return False
        GLib.timeout_add(100, self._verify_playback_state, filepath, title)
        return True

    def _verify_playback_state(self, stream_id, title):
        if not hasattr(self, "player") or not self.player:
            return False
        success, current_state, _pending_state = self.player.get_state(
            Gst.CLOCK_TIME_NONE
        )
        if not success:
            return False
        if current_state == Gst.State.PLAYING:
            self.playing = True
            is_radio = getattr(self, "_is_radio_stream", False)
            if is_radio and self._radio_stream_uri == stream_id:
                track_name = self._radio_stream_title or title or "Radio Stream"
                GLib.idle_add(self.player_tab.track_label.set_text, track_name)
                GLib.idle_add(self.set_title, f"LinAmp - {track_name}")
                GLib.idle_add(self.radio_tab._update_live_indicator)
            else:
                track_name = title or os.path.basename(stream_id)
                GLib.idle_add(self.player_tab.track_label.set_text, track_name)
                GLib.idle_add(self.set_title, f"LinAmp - {track_name}")
            if (
                hasattr(self, "radio_tab")
                and hasattr(self.radio_tab, "_is_playing_radio")
                and self.radio_tab._is_playing_radio
            ):
                if (
                    hasattr(self.radio_tab, "_radio_buffer_timer")
                    and self.radio_tab._radio_buffer_timer
                ):
                    GLib.source_remove(self.radio_tab._radio_buffer_timer)
                    self.radio_tab._radio_buffer_timer = 0
                self.radio_tab._is_playing_radio = False
                self.radio_tab.live_indicator.set_text("")
            if hasattr(self, "visualizer_tab") and self.visualizer_tab is not None:
                self.visualizer_tab.player = self.player
                if hasattr(self.visualizer_tab, "_rendering_enabled"):
                    self.visualizer_tab._rendering_enabled = True
                if hasattr(self.visualizer_tab, "_reconnect_audio_probe"):
                    self.visualizer_tab._reconnect_audio_probe()
            return False
        if current_state in [Gst.State.READY, Gst.State.PAUSED]:
            self.player.seek_simple(
                Gst.Format.TIME, Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT, 0
            )
            GLib.idle_add(self.player.set_state, Gst.State.PLAYING)
            self.playing = True
            if self.beat_aware_enabled:
                self.start_beat_detection()
        return False

    def play_uri(self, uri, title="Radio Stream"):
        if not uri or not RadioStation.validate_url(uri):
            GLib.idle_add(self.set_status_message, "Invalid stream URL")
            return False
        if not hasattr(self, "player") or not self.player:
            return False
        self._is_radio_stream = True
        self._radio_stream_uri = uri
        self._radio_stream_title = title
        self._radio_start_time = time.time()
        if hasattr(self.player_tab, "track_label"):
            GLib.idle_add(
                self.player_tab.track_label.set_text, f"Connecting: {title}..."
            )
        self.player.set_state(Gst.State.NULL)
        self.player.set_property("uri", uri)
        state_change = self.player.set_state(Gst.State.PLAYING)
        if state_change == Gst.StateChangeReturn.FAILURE:
            self._is_radio_stream = False
            GLib.idle_add(self.set_status_message, "Failed to connect to stream")
            return False
        GLib.timeout_add(100, self._verify_playback_state, uri, title)
        return True

    def on_pause(self, button):
        if self.playing:
            GLib.idle_add(self.player.set_state, Gst.State.PAUSED)
            self.playing = False
            self.stop_beat_detection()

    def on_play(self, button):
        if self.playing:
            self.on_pause(button)
        else:
            if self.current_track >= 0 and self.current_track < len(self.playlist):
                self.play_track(self.current_track)
            elif self.playlist:
                self.play_track(0)
            else:
                GLib.idle_add(
                    self.set_status_message, "No tracks in playlist - add files first"
                )

    def stop(self):
        self.on_stop(None)

    def on_stop(self, button):
        if hasattr(self, "_cleanup_in_progress") and self._cleanup_in_progress:
            return
        self._is_radio_stream = False
        if hasattr(self, "_radio_buffer_timer") and self._radio_buffer_timer:
            GLib.source_remove(self._radio_buffer_timer)
            self._radio_buffer_timer = 0
        if hasattr(self, "player") and self.player:
            try:
                if hasattr(self, "_song_finished_id") and self._song_finished_id:
                    self.player.disconnect(self._song_finished_id)
                    self._song_finished_id = None
            except TypeError:
                pass
            self.player.set_state(Gst.State.NULL)
            self.cleanup_pipeline(self.player)
        self.playing = False
        self.stop_beat_detection()
        if hasattr(self, "player_tab"):
            self.player_tab.time_label.set_text("0:00 / 0:00")
            self.player_tab.progress.set_value(0)
            self.player_tab.track_label.set_text("No track playing")
            if hasattr(self.player_tab, "cover_art_widget"):
                self.player_tab.cover_art_widget.clear()
        self.set_title("LinAmp - XMMS Style")
        self.save_settings_on_stop()

    def on_volume_changed(self, scale):
        volume = scale.get_value() / 100.0
        if hasattr(self, "player_tab") and hasattr(self.player_tab, "volume_label"):
            self.player_tab.volume_label.set_label(f"{int(scale.get_value())}%")
        self.player.set_property("volume", volume)
        if hasattr(self, "_is_muted") and self._is_muted and volume > 0:
            self._is_muted = False
            self._previous_volume = volume
        if hasattr(self, "player_tab") and hasattr(
            self.player_tab, "_update_volume_button_icon"
        ):
            self.player_tab._update_volume_button_icon()
        self.auto_save_settings()

    def on_mute_toggle(self, button):
        if not hasattr(self, "player") or not self.player:
            return
        current_volume = self.player.get_property("volume")
        if hasattr(self, "_is_muted") and self._is_muted:
            self.player.set_property("volume", self._previous_volume)
            self._is_muted = False
        elif current_volume > 0:
            self._previous_volume = current_volume
            self.player.set_property("volume", 0.0)
            self._is_muted = True
        else:
            self._previous_volume = 1.0
            self.player.set_property("volume", 1.0)
            self._is_muted = False
        if hasattr(self, "player_tab") and hasattr(
            self.player_tab, "_update_volume_button_icon"
        ):
            self.player_tab._update_volume_button_icon()
        actual_volume = self.player.get_property("volume")
        if hasattr(self, "player_tab") and hasattr(self.player_tab, "volume_scale"):
            self.player_tab.volume_scale.set_value(actual_volume * 100)

    def on_song_finished(self, element):
        GLib.idle_add(self._reset_pipeline_state)

    def cleanup_pipeline(self, player):
        if player is None:
            return
        bus = player.get_bus()
        if bus:
            if getattr(self, "_main_bus_signal_watch_attached", False):
                try:
                    bus.remove_signal_watch()
                except (TypeError, GLib.Error):
                    pass
                self._main_bus_signal_watch_attached = False
            if getattr(self, "_bus_watch_id", None) is not None:
                try:
                    bus.disconnect(self._bus_watch_id)
                except (TypeError, GLib.Error):
                    pass
                self._bus_watch_id = None
        if getattr(self, "_song_finished_id", None) is not None:
            try:
                player.disconnect(self._song_finished_id)
            except (TypeError, GLib.Error):
                pass
            self._song_finished_id = None
        if (
            getattr(self, "audio_probe_id", None) is not None
            and getattr(self, "_probe_pad", None) is not None
        ):
            try:
                self._probe_pad.remove_probe(self.audio_probe_id)
            except (TypeError, GLib.Error):
                pass
            self.audio_probe_id = None
            self._probe_pad = None
            self._probe_sink = None
        for attr in ("audio_convert", "audio_resample", "audio_sink"):
            elem = getattr(self, attr, None)
            if elem is not None:
                try:
                    elem.set_state(Gst.State.NULL)
                except (AttributeError, GLib.Error):
                    pass
                setattr(self, attr, None)
        if getattr(self, "equalizer", None) is not None:
            try:
                self.equalizer.set_state(Gst.State.NULL)
            except (AttributeError, GLib.Error):
                pass
            self.equalizer = None

    def _reset_pipeline_state(self):
        if self.player is None:
            self.playing = False
            return
        try:
            current_state = self.player.get_state(Gst.CLOCK_TIME_NONE)[1]
            if current_state == Gst.State.PLAYING:
                return  # Already playing a new track
            self.player.set_state(Gst.State.READY)
        except GLib.Error:
            pass
        self.playing = False
        if getattr(self, "_is_radio_stream", False):
            GLib.idle_add(self._radio_auto_reconnect)
            return
        if self.auto_play_next:
            next_index = self.get_next_track_index()
            if next_index is not None:
                self.play_next_track()
            else:
                if hasattr(self.player_tab, "cover_art_widget"):
                    self.player_tab.cover_art_widget.clear()

    def on_bus_message(self, bus, message):
        if message.type == Gst.MessageType.ERROR:
            err, _debug = message.parse_error()
            error_msg = str(err.message).lower()
            if "internal" in error_msg and "stream" in error_msg:
                current_uri = self.player.get_property("uri") if self.player else None
                GLib.idle_add(self.player.set_state, Gst.State.NULL)
                recovery_id = GLib.timeout_add(
                    100, lambda: GLib.idle_add(self._attempt_recovery, current_uri)
                )
                if hasattr(self, "_active_timers"):
                    self._active_timers.add(recovery_id)
                GLib.idle_add(
                    self.set_status_message, "Internal stream error - recovering..."
                )
                return
            if self.playlist and self.current_track < len(self.playlist) - 1:
                GLib.idle_add(self.play_track, self.current_track + 1)
            else:
                GLib.idle_add(self.set_status_message, f"Error: {err.message}")
            if getattr(self, "_is_radio_stream", False):
                GLib.idle_add(self._radio_auto_reconnect)
        elif message.type == Gst.MessageType.EOS:
            if getattr(self, "_is_radio_stream", False):
                GLib.idle_add(self._radio_auto_reconnect)
            else:
                GLib.idle_add(self._handle_eos)
        elif message.type == Gst.MessageType.TAG:
            if getattr(self, "_is_radio_stream", False):
                tag_list = message.parse_tag()
                if tag_list:
                    for key in ["title", "StreamTitle", "icy-name", "icy-url"]:
                        success, value = tag_list.get_string(key)
                        if success and value:
                            GLib.idle_add(self._on_radio_tag, key, value)
                            break
        return Gst.BusSyncReply.PASS

    def _handle_eos(self):
        if getattr(self, "_is_radio_stream", False):
            GLib.idle_add(self._radio_auto_reconnect)
            return
        ret = self.player.set_state(Gst.State.READY)
        if ret == Gst.StateChangeReturn.FAILURE:
            pass
        self.playing = False
        if self.auto_play_next:
            if self.repeat_mode == "one":
                self.play_track(self.current_track)
            elif self.repeat_mode == "all":
                self.play_next_track()
            else:
                next_index = self.get_next_track_index()
                if next_index is not None:
                    self.play_track(next_index)
                else:
                    self.on_stop(None)
        else:
            self.on_stop(None)

    def _on_state_changed(self, bus, message):
        if message.src == self.player:
            _old_state, new_state, _pending_state = message.parse_state_changed()
            if new_state == Gst.State.PLAYING:
                self.playing = True
            elif new_state in [Gst.State.PAUSED, Gst.State.NULL]:
                self.playing = False
        return Gst.BusSyncReply.PASS

    def _attempt_recovery(self, uri):
        if not uri:
            GLib.idle_add(
                self.set_status_message, "Recovery failed - trying next track"
            )
            self._advance_after_error()
            return False
        self.player.set_property("uri", uri)
        GLib.idle_add(self.player.set_state, Gst.State.PLAYING)
        check_id = GLib.timeout_add(500, self._check_recovery_success)
        if hasattr(self, "_active_timers"):
            self._active_timers.add(check_id)
        return False

    def _check_recovery_success(self):
        if not hasattr(self, "player") or not self.player:
            GLib.idle_add(
                self.set_status_message, "Recovery failed - trying next track"
            )
            self._advance_after_error()
            return False
        _old, new_state, _pending = self.player.get_state(Gst.CLOCK_TIME_NONE)
        if new_state != Gst.State.PLAYING:
            GLib.idle_add(
                self.set_status_message, "Recovery failed - trying next track"
            )
            self._advance_after_error()
            return False
        GLib.idle_add(self.set_status_message, "Recovery succeeded")
        return False

    def _advance_after_error(self):
        if self.playlist and self.current_track < len(self.playlist) - 1:
            GLib.idle_add(self.play_track, self.current_track + 1)
        elif self.playlist and self.repeat_mode == "all":
            GLib.idle_add(self.play_track, 0)
        else:
            GLib.idle_add(self.on_stop, None)

    def _radio_auto_reconnect(self):
        if not getattr(self, "_is_radio_stream", False):
            return False
        uri = getattr(self, "_radio_stream_uri", None)
        if not uri:
            return False
        current_uri = self.player.get_property("uri") if self.player else None
        if current_uri == uri:
            return False
        GLib.idle_add(self.set_status_message, "Reconnecting to stream...")
        GLib.timeout_add(
            2000,
            lambda: self.play_uri(
                uri, getattr(self, "_radio_stream_title", "Radio Stream")
            ),
        )
        return False

    def _on_radio_tag(self, key, value):
        if hasattr(self, "player_tab") and hasattr(self.player_tab, "track_label"):
            display = value
            if key == "StreamTitle" and " - " in value:
                display = value
            elif key == "icy-name" and value:
                display = f"🎙 {value}"
            GLib.idle_add(self.player_tab.track_label.set_text, display)
            GLib.idle_add(self.set_title, f"LinAmp - {display}")
        if hasattr(self, "header_nowplaying") and self.header_nowplaying:
            GLib.idle_add(self.header_nowplaying.set_text, value)
        return False

    def _check_visualizer_allocation(self, viz):
        if (
            hasattr(self.visualizer_tab, "visualizer")
            and self.visualizer_tab.visualizer is not None
        ):
            alloc_w = viz.get_allocated_width()
            alloc_h = viz.get_allocated_height()
            tab_alloc_w = self.visualizer_tab.get_allocated_width()
            tab_alloc_h = self.visualizer_tab.get_allocated_height()
            hexpand = viz.get_hexpand()
            vexpand = viz.get_vexpand()
            self.logger.debug(
                f"GLArea alloc (delayed)={alloc_w}x{alloc_h} expand=h={hexpand} v={vexpand}"
            )
            self.logger.debug(
                f"VisualizerTab alloc (delayed)={tab_alloc_w}x{tab_alloc_h}"
            )
        return False

    def update_display(self):
        is_radio = getattr(self, "_is_radio_stream", False)
        if hasattr(self, "player") and self.player and self.playing:
            if is_radio:
                elapsed = (
                    int(time.time() - self._radio_start_time)
                    if hasattr(self, "_radio_start_time")
                    else 0
                )
                mins = elapsed // 60
                secs = elapsed % 60
                live_str = f"LIVE {mins}:{secs:02d}"
                if hasattr(self, "player_tab") and self.player_tab:
                    if (
                        hasattr(self.player_tab, "time_label")
                        and self.player_tab.time_label
                    ):
                        self.player_tab.time_label.set_text(live_str)
                        self.player_tab.time_label.add_css_class("time-label")
                    if (
                        hasattr(self.player_tab, "progress")
                        and self.player_tab.progress
                    ):
                        self.player_tab.progress.set_value(0)
                if hasattr(self, "header_nowplaying") and self.header_nowplaying:
                    title = (
                        self._radio_stream_title
                        if hasattr(self, "_radio_stream_title")
                        else "Radio"
                    )
                    self.header_nowplaying.set_text(f"{title}  •  {live_str}")
            else:
                success, position = self.player.query_position(Gst.Format.TIME)
                if not success:
                    position = 0
                success, duration = self.player.query_duration(Gst.Format.TIME)
                if not success:
                    duration = 0
                position_sec = position // Gst.SECOND
                duration_sec = duration // Gst.SECOND
                pos_str = f"{position_sec // 60}:{position_sec % 60:02d}"
                dur_str = f"{duration_sec // 60}:{duration_sec % 60:02d}"
                if hasattr(self, "player_tab") and self.player_tab:
                    if (
                        hasattr(self.player_tab, "time_label")
                        and self.player_tab.time_label
                    ):
                        self.player_tab.time_label.set_text(f"{pos_str} / {dur_str}")
                        self.player_tab.time_label.add_css_class("time-label")
                    if (
                        hasattr(self.player_tab, "progress")
                        and self.player_tab.progress
                    ):
                        if duration_sec > 0 and position_sec >= 0:
                            fraction = position_sec / duration_sec
                            fraction = max(0.0, min(1.0, fraction))
                            current_value = self.player_tab.progress.get_value()
                            if abs(fraction * 100 - current_value) > 0.1:
                                self.player_tab.progress.set_value(fraction * 100)
                                if fraction > 0.75:
                                    self.player_tab.progress.add_css_class("near-end")
                                    self.player_tab.progress.remove_css_class("halfway")
                                elif fraction > 0.5:
                                    self.player_tab.progress.remove_css_class(
                                        "near-end"
                                    )
                                    self.player_tab.progress.add_css_class("halfway")
                                else:
                                    self.player_tab.progress.remove_css_class(
                                        "near-end"
                                    )
                                    self.player_tab.progress.remove_css_class("halfway")
                        else:
                            self.player_tab.progress.set_value(0)
                    if (
                        hasattr(self.player_tab, "track_label")
                        and self.player_tab.track_label
                    ):
                        self.player_tab.track_label.add_css_class("playing-indicator")
                if (
                    hasattr(self, "header_nowplaying")
                    and self.header_nowplaying
                    and self.playlist
                    and 0 <= self.current_track < len(self.playlist)
                ):
                    current = self.playlist[self.current_track]
                    title = getattr(current, "title", "") or "Playing"
                    self.header_nowplaying.set_text(
                        f"{title}  •  {pos_str} / {dur_str}"
                    )
        status_parts = []
        if self.repeat_mode != "none":
            status_parts.append(f"Repeat: {self.repeat_mode.upper()}")
        if self.shuffle_mode:
            status_parts.append("SHUFFLE")
        if not self.auto_play_next:
            status_parts.append("NO AUTO-NEXT")
        if self.beat_aware_enabled:
            status_parts.append("BEAT-AWARE")
        status_text = " | ".join(status_parts) if status_parts else "Ready"
        if self.playing:
            status_text = f"🎵 {status_text}"
            if hasattr(self, "status_bar"):
                self.status_bar.add_css_class("playing")
        else:
            if hasattr(self, "status_bar"):
                self.status_bar.remove_css_class("playing")
                if hasattr(self, "header_nowplaying") and self.header_nowplaying:
                    if self.playlist and 0 <= self.current_track < len(self.playlist):
                        current = self.playlist[self.current_track]
                        title = getattr(current, "title", "") or "Ready"
                        self.header_nowplaying.set_text(title)
                    else:
                        self.header_nowplaying.set_text("Ready")
        GLib.idle_add(self.set_status_message, status_text)
        return True

    def focus_playlist_search(self):
        if hasattr(self, "notebook"):
            self.notebook.set_current_page(2)
        if (
            hasattr(self, "playlist_tab")
            and hasattr(self.playlist_tab, "search_entry")
            and self.playlist_tab.search_entry
        ):
            self.playlist_tab.search_entry.grab_focus()

    def play_track(self, index):
        if 0 <= index < len(self.playlist):
            item = self.playlist[index]
            if self.repeat_mode == "one" and index == self.current_track:
                GLib.idle_add(self.player.set_state, Gst.State.NULL)
            if self.play_file(item.path, item.title):
                self.current_track = index
                self.save_settings_on_track_change()
                if self.shuffle_mode and index in self.shuffled_indices:
                    self.shuffle_position = self.shuffled_indices.index(index)
                if hasattr(self, "playlist_tab") and hasattr(
                    self.playlist_tab, "selection_model"
                ):
                    self.playlist_tab.selection_model.set_selected(index)
                self._load_cover_art_for_track(item)
                if hasattr(self, "lyrics_tab"):
                    self.lyrics_tab.load_lyrics(item)
                return True
        return False

    def _load_cover_art_for_track(self, item):
        if not hasattr(self, "player_tab") or not hasattr(
            self.player_tab, "cover_art_widget"
        ):
            return
        cover_widget = self.player_tab.cover_art_widget
        current_audio_path = cover_widget.current_cover_path
        if current_audio_path:
            if current_audio_path.startswith(str(COVER_CACHE_DIR)):
                try:
                    cache_filename = Path(current_audio_path).name
                    cache_key = cache_filename.split(".")[0]
                    item_cache_key = CoverArt.get_cache_key(item.path)
                    if cache_key == item_cache_key:
                        return
                except (OSError, ValueError):
                    pass
            elif current_audio_path == item.path:
                return
        allow_download = getattr(self, "auto_download_covers", False)
        cover, _cache_path = extract_cover_art(item.path, allow_download=False)
        if cover:
            cover_widget.load_cover(cover, item.path)
        else:
            if allow_download:
                if not hasattr(self, "_download_executor"):
                    self._download_executor = ThreadPoolExecutor(max_workers=2)

                def download_and_load():
                    cover = download_cover_for_audio(item.path)
                    if cover:
                        GLib.idle_add(cover_widget.load_cover, cover, item.path)

                self._download_executor.submit(download_and_load)
            else:
                cover_widget.load_cover(None, item.path)

    def on_prev(self, button):
        self.play_previous_track()

    def play_previous_track(self):
        if not self.playlist:
            return
        prev_index = self.get_previous_track_index()
        if prev_index is not None:
            self.play_track(prev_index)

    def get_previous_track_index(self):
        if not self.playlist:
            return None
        if self.shuffle_mode:
            return self.get_previous_shuffled_index()
        else:
            if self.current_track > 0:
                return self.current_track - 1
            elif self.repeat_mode == "all":
                return len(self.playlist) - 1
            else:
                return None

    def get_previous_shuffled_index(self):
        if not self.shuffled_indices:
            self.regenerate_shuffle_list()
        if self.shuffle_position > 0:
            self.shuffle_position -= 1
            return self.shuffled_indices[self.shuffle_position]
        elif self.repeat_mode == "all":
            self.shuffle_position = len(self.shuffled_indices) - 1
            return self.shuffled_indices[self.shuffle_position]
        else:
            return None

    def toggle_repeat_mode(self):
        modes = ["none", "one", "all"]
        current_index = modes.index(self.repeat_mode)
        self.repeat_mode = modes[(current_index + 1) % len(modes)]
        self.update_status_display()
        self.auto_save_settings()

    def toggle_shuffle_mode(self):
        self.shuffle_mode = not self.shuffle_mode
        if self.shuffle_mode:
            self.regenerate_shuffle_list()
            if self.current_track >= 0:
                try:
                    self.shuffle_position = self.shuffled_indices.index(
                        self.current_track
                    )
                except ValueError:
                    self.shuffle_position = -1
        self.update_status_display()
        self.auto_save_settings()

    def toggle_auto_play_next(self):
        self.auto_play_next = not self.auto_play_next
        self.update_status_display()
        self.auto_save_settings()

    def toggle_beat_aware(self):
        self.beat_aware_enabled = not self.beat_aware_enabled
        self.update_status_display()
        self.auto_save_settings()

    def toggle_auto_download_covers(self):
        self.auto_download_covers = not self.auto_download_covers
        status = "enabled" if self.auto_download_covers else "disabled"
        self.set_status_message(f"Auto download covers: {status}")
        self.update_status_display()
        self.auto_save_settings()
        if (
            self.auto_download_covers
            and self.current_track >= 0
            and self.current_track < len(self.playlist)
        ):
            current_item = self.playlist[self.current_track]
            self._load_cover_art_for_track(current_item)

    def toggle_show_cover_art(self):
        self.settings.show_cover_art = not self.settings.show_cover_art
        self.player_tab.cover_art_widget.set_visible(self.settings.show_cover_art)
        if self.settings.show_cover_art and 0 <= self.current_track < len(self.playlist):
                current_item = self.playlist[self.current_track]
                self._load_cover_art_for_track(current_item)
        status = "enabled" if self.settings.show_cover_art else "disabled"
        self.set_status_message(f"Cover art: {status}")
        self.update_status_display()
        self.auto_save_settings()

    def auto_save_settings(self):
        if hasattr(self, "_auto_save_timer") and self._auto_save_timer:
            self._safe_source_remove(self._auto_save_timer)
        self._auto_save_timer = GLib.timeout_add(
            AUTO_SAVE_DELAY, self._auto_save_callback
        )
        self._active_timers.add(self._auto_save_timer)

    def _auto_save_callback(self):
        self._update_settings_from_state()
        self.save_settings()
        if self.beat_aware_enabled:
            self.start_beat_detection()
        else:
            self.stop_beat_detection()
        self._auto_save_timer = 0
        return False

    def update_status_display(self):
        status_parts = []
        if self.repeat_mode != "none":
            status_parts.append(f"Repeat: {self.repeat_mode.upper()}")
        if self.shuffle_mode:
            status_parts.append("SHUFFLE")
        if not self.auto_play_next:
            status_parts.append("NO AUTO-NEXT")
        if self.beat_aware_enabled:
            status_parts.append("BEAT-AWARE")
        if self.auto_download_covers:
            status_parts.append("AUTO-DL")
        status_text = " | ".join(status_parts) if status_parts else "Ready"
        GLib.idle_add(self.set_status_message, status_text)
        self.update_button_states()

    def update_button_states(self):
        if not hasattr(self, "player_tab"):
            return
        if hasattr(self.player_tab, "repeat_btn"):
            if self.repeat_mode == "none":
                self.player_tab.repeat_btn.remove_css_class("active")
                self.player_tab.repeat_btn.set_tooltip_text("Repeat: Off")
            elif self.repeat_mode == "one":
                self.player_tab.repeat_btn.add_css_class("active")
                self.player_tab.repeat_btn.set_tooltip_text("Repeat: One Track")
            else:
                self.player_tab.repeat_btn.add_css_class("active")
                self.player_tab.repeat_btn.set_tooltip_text("Repeat: All")
        if hasattr(self.player_tab, "shuffle_btn"):
            if self.shuffle_mode:
                self.player_tab.shuffle_btn.add_css_class("active")
                self.player_tab.shuffle_btn.set_tooltip_text("Shuffle: On")
            else:
                self.player_tab.shuffle_btn.remove_css_class("active")
                self.player_tab.shuffle_btn.set_tooltip_text("Shuffle: Off")
        if hasattr(self.player_tab, "beat_btn"):
            if self.beat_aware_enabled:
                self.player_tab.beat_btn.add_css_class("active")
                self.player_tab.beat_btn.set_tooltip_text("Beat Detection: On")
            else:
                self.player_tab.beat_btn.remove_css_class("active")
                self.player_tab.beat_btn.set_tooltip_text("Beat Detection: Off")
        if hasattr(self.player_tab, "download_btn"):
            if self.auto_download_covers:
                self.player_tab.download_btn.add_css_class("active")
                self.player_tab.download_btn.set_tooltip_text(
                    "Auto Download Covers: On"
                )
            else:
                self.player_tab.download_btn.remove_css_class("active")
                self.player_tab.download_btn.set_tooltip_text(
                    "Auto Download Covers: Off"
                )
        if hasattr(self.player_tab, "autonext_btn"):
            if self.auto_play_next:
                self.player_tab.autonext_btn.add_css_class("active")
                self.player_tab.autonext_btn.set_tooltip_text("Auto Next: On")
            else:
                self.player_tab.autonext_btn.remove_css_class("active")
                self.player_tab.autonext_btn.set_tooltip_text("Auto Next: Off")
        if hasattr(self.player_tab, "cover_btn"):
            if self.settings.show_cover_art:
                self.player_tab.cover_btn.add_css_class("active")
                self.player_tab.cover_btn.set_tooltip_text("Cover Art: On")
            else:
                self.player_tab.cover_btn.remove_css_class("active")
                self.player_tab.cover_btn.set_tooltip_text("Cover Art: Off")

    def on_next(self, button):
        self.play_next_track()

    def play_next_track(self):
        if not self.playlist:
            return
        next_index = self.get_next_track_index()
        if next_index is not None:
            if hasattr(self, "playlist_tab") and hasattr(
                self.playlist_tab, "selection_model"
            ):
                self.playlist_tab.selection_model.set_selected(next_index)
            self.play_track(next_index)

    def get_next_track_index(self):
        if not self.playlist:
            return None
        if self.shuffle_mode:
            return self.get_next_shuffled_index()
        else:
            if self.current_track < len(self.playlist) - 1:
                return self.current_track + 1
            elif self.repeat_mode == "all":
                return 0
            else:
                return None

    def get_next_shuffled_index(self):
        if not self.shuffled_indices:
            self.regenerate_shuffle_list()
        if self.shuffle_position < len(self.shuffled_indices) - 1:
            self.shuffle_position += 1
            return self.shuffled_indices[self.shuffle_position]
        elif self.repeat_mode == "all":
            self.regenerate_shuffle_list()
            self.shuffle_position = 0
            return self.shuffled_indices[0]
        else:
            return None

    def regenerate_shuffle_list(self):
        self.original_indices = list(range(len(self.playlist)))
        self.shuffled_indices = self.original_indices.copy()
        random.shuffle(self.shuffled_indices)
        self.shuffle_position = -1

    def start_beat_detection(self):
        if not self.beat_aware_enabled or not self.playing:
            return
        self.beat_detection_timer = GLib.timeout_add(
            BEAT_DETECTION_INTERVAL, self.detect_beat
        )
        self._active_timers.add(self.beat_detection_timer)

    def stop_beat_detection(self):
        self._safe_source_remove(self.beat_detection_timer)
        self.beat_detection_timer = 0

    def detect_beat(self):
        if not self.beat_aware_enabled or not self.playing:
            return False
        try:
            current_time = time.time()
            if current_time - self.last_beat_time < 0.1:
                return False
            levels = []
            if hasattr(self, "audio_levels") and self.audio_levels:
                levels = self.audio_levels
            if len(levels) > 0:
                avg_level = sum(levels) / len(levels)
                peak_level = max(levels)
                if hasattr(self, "_last_avg_level"):
                    level_increase = avg_level - self._last_avg_level
                    if (
                        level_increase > 0.15 and peak_level > 0.7
                    ) or peak_level > 0.95:
                        self.last_beat_time = current_time
                        self._last_avg_level = avg_level
                        current_beat_time = current_time
                        if hasattr(self, "_last_beat_timestamp"):
                            beat_interval = (
                                current_beat_time - self._last_beat_timestamp
                            )
                            if 0.2 <= beat_interval <= 2.0:
                                self.beat_interval_history.append(beat_interval)
                                if len(self.beat_interval_history) > BEAT_HISTORY_SIZE:
                                    self.beat_interval_history.pop(0)
                                if len(self.beat_interval_history) >= 2:
                                    sorted_intervals = sorted(
                                        self.beat_interval_history
                                    )
                                    median_interval = sorted_intervals[
                                        len(sorted_intervals) // 2
                                    ]
                                    if median_interval > 0:
                                        bpm = 60.0 / median_interval
                                        if 60 <= bpm <= 200:
                                            self.on_beat_detected(bpm)
                                            return True
                        self._last_beat_timestamp = current_beat_time
                        return True
                self._last_avg_level = avg_level
            return False
        except (TypeError, ValueError, ZeroDivisionError):
            return False

    def on_beat_detected(self, bpm):
        pass  # Hook for beat-reactive UI effects; currently no-op

    def add_to_playlist(self, file_paths):
        added_items = []
        for path in file_paths:
            try:
                normalized_path = os.path.abspath(os.path.expanduser(str(path)))
                if "\0" in normalized_path:
                    continue
                real_path = os.path.realpath(normalized_path)
                if not os.path.exists(real_path):
                    continue
                if not os.path.isfile(real_path):
                    continue
                if real_path.startswith(("/proc/", "/sys/")):
                    continue
            except (OSError, ValueError):
                continue
            item = PlaylistItem(
                path=normalized_path, title=os.path.basename(normalized_path)
            )
            self.playlist.append(item)
            added_items.append(item)
        if hasattr(self, "playlist_tab") and added_items:
            for item in added_items:
                self.playlist_tab.playlist_store.append(f"{item.title}|{item.id}")
            self.playlist_tab.update_statistics()
            self._invalidate_playlist_index()
        self.save_playlist()

    def add_folder_to_playlist(self, folder_path):
        try:
            normalized_path = os.path.abspath(os.path.expanduser(str(folder_path)))
            if "\0" in normalized_path:
                return
            if not os.path.exists(normalized_path) or not os.path.isdir(
                normalized_path
            ):
                return
        except (OSError, ValueError):
            return
        GLib.idle_add(
            self.set_status_message,
            f"Scanning folder: {os.path.basename(normalized_path)}...",
        )

        def scan_folder():
            added_items = []
            errors = []
            try:
                for root, dirs, files in os.walk(normalized_path):
                    dirs[:] = [d for d in dirs if not d.startswith(".")]
                    for file in files:
                        try:
                            file_path = os.path.join(root, file)
                            file_ext = os.path.splitext(file)[1].lower()
                            if file_ext in AUDIO_EXTENSIONS:
                                rel_path = os.path.relpath(file_path, normalized_path)
                                title = rel_path if rel_path != file_path else file
                                item = PlaylistItem(path=file_path, title=title)
                                added_items.append(item)
                        except (OSError, ValueError) as e:
                            errors.append(f"{file}: {e}")
            except (OSError, PermissionError) as e:
                errors.append(f"Scan failed: {e}")
            GLib.idle_add(self._on_folder_scan_complete, added_items, errors)

        thread = threading.Thread(target=scan_folder, daemon=True)
        thread.start()

    def _on_folder_scan_complete(self, added_items, errors=None):
        if errors:
            for error in errors:
                pass
        if added_items:
            self.playlist.extend(added_items)
            if hasattr(self, "playlist_tab"):
                for item in added_items:
                    self.playlist_tab.playlist_store.append(f"{item.title}|{item.id}")
                self.playlist_tab.update_statistics()
                self._invalidate_playlist_index()
            self.save_playlist()
            GLib.idle_add(
                self.set_status_message, f"Added {len(added_items)} tracks to playlist"
            )
        else:
            GLib.idle_add(self.set_status_message, "No audio files found in folder")

    def save_playlist(self, filepath: str | None = None) -> bool:
        if not hasattr(self, "playlist") or not self.playlist:
            return False
        if not filepath:
            playlist_dir = os.path.expanduser("~/.config/linamp")
            filepath = os.path.join(playlist_dir, "playlist.json")
        else:
            filepath = os.path.abspath(os.path.expanduser(filepath))
            playlist_dir = os.path.dirname(filepath)
            os.makedirs(playlist_dir, exist_ok=True)
        playlist_data = []
        for item in self.playlist:
            if hasattr(item, "to_dict") and callable(item.to_dict):
                if hasattr(item, "path") and item.path and os.path.exists(item.path):
                    playlist_data.append(item.to_dict())
                else:
                    pass
        if not playlist_data:
            return False
        temp_file = None
        fd, temp_file = tempfile.mkstemp(
            prefix=".playlist_", suffix=".tmp", dir=playlist_dir, text=True
        )
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(playlist_data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        if os.path.exists(filepath):
            os.replace(temp_file, filepath)
        else:
            os.rename(temp_file, filepath)
        return True

    def _clear_playlist_store(self) -> bool:
        if not hasattr(self, "playlist_tab") or not hasattr(
            self.playlist_tab, "playlist_store"
        ):
            return False
        store = self.playlist_tab.playlist_store
        store.freeze_notify()
        if hasattr(store, "splice") and callable(store.splice):
            store.splice(0, store.get_n_items(), [])
        else:
            while store.get_n_items() > 0:
                store.remove(store.get_n_items() - 1)
        store.thaw_notify()
        return True

    def _update_playlist_display(self) -> bool:
        if not hasattr(self, "playlist_tab") or not hasattr(
            self.playlist_tab, "playlist_store"
        ):
            return False
        store = self.playlist_tab.playlist_store
        if not self.playlist:
            self._clear_playlist_store()
            return True
        store.freeze_notify()
        current_count = store.get_n_items()
        target_count = len(self.playlist)
        if current_count > target_count:
            if hasattr(store, "splice") and callable(store.splice):
                store.splice(target_count, current_count - target_count, [])
            else:
                while store.get_n_items() > target_count:
                    store.remove(store.get_n_items() - 1)
        display_names = []
        for i, item in enumerate(self.playlist):
            if isinstance(item, PlaylistItem):
                display_names.append(f"{item.get_display_name()}|{item.id}")
            elif isinstance(item, dict):
                title = item.get("title") or os.path.basename(item.get("path", ""))
                path = item.get("path", "")
                track_id = (
                    item.get("id")
                    or hashlib.md5(f"{path}:{title}".encode()).hexdigest()[:16]
                )
                display_names.append(f"{title}|{track_id}")
            else:
                s = str(item)
                track_id = hashlib.md5(s.encode()).hexdigest()[:16]
                display_names.append(f"{s}|{track_id}")
        if hasattr(store, "splice") and callable(store.splice):
            if current_count < target_count:
                store.splice(current_count, 0, display_names[current_count:])
            else:
                store.splice(
                    0, min(current_count, target_count), display_names[:target_count]
                )
        else:
            while store.get_n_items() > 0:
                store.remove(store.get_n_items() - 1)
            for name in display_names:
                store.append(name)
        store.thaw_notify()
        return True

    def load_playlist(self, filepath: str | None = None) -> bool:
        if not filepath:
            playlist_dir = os.path.expanduser("~/.config/linamp")
            filepath = os.path.join(playlist_dir, "playlist.json")
        else:
            filepath = os.path.abspath(os.path.expanduser(filepath))
        if not os.path.exists(filepath):
            return False
        if not os.access(filepath, os.R_OK):
            return False
        try:
            with open(filepath, encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    return False
            if not isinstance(data, list):
                return False
            if len(data) > 100000:
                return False
        except OSError:
            return False
        playlist = []
        invalid_items = 0
        for i, item_data in enumerate(data, 1):
            if not isinstance(item_data, dict):
                invalid_items += 1
                continue
            try:
                item = PlaylistItem.from_dict(item_data)
                if item.exists():
                    playlist.append(item)
            except ValueError:
                invalid_items += 1
        if invalid_items > 0:
            pass
        self.playlist = playlist
        self._invalidate_playlist_index()
        if hasattr(self, "playlist_tab") and self.playlist_tab:
            self._clear_playlist_store()
            self._update_playlist_display()
        return len(playlist) > 0

    def cleanup_invalid_tracks(self):
        if not self.playlist:
            return
        original_count = len(self.playlist)
        valid_tracks = []
        for item in self.playlist:
            if item.exists():
                valid_tracks.append(item)
            else:
                pass
        if len(valid_tracks) != original_count:
            self.playlist = valid_tracks
            self._invalidate_playlist_index()
            self._update_playlist_display()
            self.save_playlist()
            self.set_status_message(
                f"Removed {original_count - len(valid_tracks)} invalid tracks"
            )


class LinAmpApp(Gtk.Application):
    def __init__(self):
        super().__init__(application_id="org.example.linamp.xmms")
        self.win = None
        self.set_property("application-id", "org.example.linamp.xmms")
        Gtk.Settings.get_default().set_property(
            "gtk-application-prefer-dark-theme", True
        )

    def do_activate(self):
        if not self.win:
            self.win = WinampWindow(application=self, title="LinAmp")
            self.win.present()
            self.win.set_visible(True)
            self.win.grab_focus()
            GLib.timeout_add(500, self.apply_dark_theme_css)
        else:
            self.win.present()

    def apply_dark_theme_css(self):
        css_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "theme", "gtk.css"
        )
        css_provider = Gtk.CssProvider()
        css_provider.load_from_path(css_file)
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(), css_provider, Gtk.STYLE_PROVIDER_PRIORITY_USER
        )
        return False

    def do_shutdown(self):
        if hasattr(self, "win") and self.win is not None:
            self.win.full_cleanup()
        Gtk.Application.do_shutdown(self)

    def do_startup(self):
        Gtk.Application.do_startup(self)
        self.setup_menu()
        resize_action = Gio.SimpleAction.new("resize_to_cover_art", None)
        resize_action.connect("activate", self.on_resize_to_cover_art)
        self.add_action(resize_action)
        self.set_accels_for_action("app.resize_to_cover_art", ["<Control>r"])
        if len(sys.argv) > 1:
            if not self.win:
                self.win = WinampWindow(application=self, title="LinAmp")
            self.win.add_to_playlist(sys.argv[1:])
            if self.win.playlist:
                self.win.cleanup_invalid_tracks()
                if self.win.playlist:
                    self.win.play_track(0)

    def setup_menu(self):
        open_action = Gio.SimpleAction.new("open", None)
        open_action.connect("activate", self.on_open)
        open_folder_action = Gio.SimpleAction.new("open_folder", None)
        open_folder_action.connect("activate", self.on_open_folder)
        add_to_playlist_action = Gio.SimpleAction.new("add_to_playlist", None)
        add_to_playlist_action.connect("activate", self.on_add_to_playlist)
        add_folder_to_playlist_action = Gio.SimpleAction.new(
            "add_folder_to_playlist", None
        )
        add_folder_to_playlist_action.connect(
            "activate", self.on_add_folder_to_playlist
        )
        cleanup_action = Gio.SimpleAction.new("cleanup_invalid", None)
        cleanup_action.connect("activate", self.on_cleanup_invalid)
        resize_cover_action = Gio.SimpleAction.new("resize_to_cover_art", None)
        resize_cover_action.connect("activate", self.on_resize_to_cover_art)
        quit_action = Gio.SimpleAction.new("quit", None)
        quit_action.connect("activate", lambda *_: self.quit())
        self.add_action(open_action)
        self.add_action(open_folder_action)
        self.add_action(add_to_playlist_action)
        self.add_action(add_folder_to_playlist_action)
        self.add_action(cleanup_action)
        self.add_action(resize_cover_action)
        self.add_action(quit_action)
        menu = Gio.Menu()
        file_menu = Gio.Menu()
        file_menu.append("Open File", "app.open")
        file_menu.append("Open Folder", "app.open_folder")
        file_menu.append("Add to Playlist", "app.add_to_playlist")
        file_menu.append("Add Folder to Playlist", "app.add_folder_to_playlist")
        file_menu.append("Cleanup Invalid Tracks", "app.cleanup_invalid")
        file_menu.append("Resize Window to Cover Art", "app.resize_to_cover_art")
        file_menu.append("Quit", "app.quit")
        menu.append_submenu("File", file_menu)
        self.set_menubar(menu)

    def on_open(self, action, param):
        dialog = Gtk.FileChooserNative.new(
            title="Open Audio File", parent=self.win, action=Gtk.FileChooserAction.OPEN
        )
        audio_filter = Gtk.FileFilter()
        audio_filter.set_name("Audio files")
        audio_filter.add_mime_type("audio/*")
        dialog.add_filter(audio_filter)
        dialog.connect("response", self.on_file_chooser_response)
        dialog.set_visible(True)

    def on_add_to_playlist(self, action, param):
        dialog = Gtk.FileChooserNative.new(
            title="Add to Playlist",
            parent=self.win,
            action=Gtk.FileChooserAction.OPEN,
            select_multiple=True,
        )
        audio_filter = Gtk.FileFilter()
        audio_filter.set_name("Audio files")
        audio_filter.add_mime_type("audio/*")
        dialog.add_filter(audio_filter)
        dialog.connect("response", self.on_add_to_playlist_response)
        dialog.set_visible(True)

    def on_open_folder(self, action, param):
        dialog = Gtk.FileChooserNative.new(
            title="Open Music Folder",
            parent=self.win,
            action=Gtk.FileChooserAction.SELECT_FOLDER,
        )
        dialog.connect("response", self.on_folder_chooser_response)
        dialog.set_visible(True)

    def on_add_folder_to_playlist(self, action, param):
        dialog = Gtk.FileChooserNative.new(
            title="Add Music Folder to Playlist",
            parent=self.win,
            action=Gtk.FileChooserAction.SELECT_FOLDER,
        )
        dialog.connect("response", self.on_add_folder_to_playlist_response)
        dialog.set_visible(True)

    def on_file_chooser_response(self, dialog, response):
        if response == Gtk.ResponseType.OK:
            file = dialog.get_property("file")
            if file:
                self.win.playlist.clear()
                self.win._invalidate_playlist_index()
                if hasattr(self.win, "playlist_tab"):
                    store = self.win.playlist_tab.playlist_store
                    while store.get_n_items() > 0:
                        store.remove(0)
                self._handle_playlist_addition(file.get_path(), is_folder=False)
                self.win.play_track(0)
        dialog.destroy()

    def on_add_to_playlist_response(self, dialog, response):
        if response == Gtk.ResponseType.OK:
            files = dialog.get_files()
            if files:
                for file in files:
                    file_path = file.get_path()
                    if file_path and os.path.isfile(file_path):
                        self.win.add_to_playlist([file_path])
                self.win.play_track(0)
        dialog.destroy()

    def on_folder_chooser_response(self, dialog, response):
        if response == Gtk.ResponseType.OK:
            folder = dialog.get_file()
            if folder:
                self.win.playlist.clear()
                if hasattr(self.win, "playlist_tab"):
                    store = self.win.playlist_tab.playlist_store
                    while store.get_n_items() > 0:
                        store.remove(0)
        dialog.destroy()

    def on_add_folder_to_playlist_response(self, dialog, response):
        if response == Gtk.ResponseType.OK:
            folder = dialog.get_file()
            if folder:
                self._handle_playlist_addition(folder.get_path(), is_folder=True)
        dialog.destroy()

    def _handle_playlist_addition(self, file_path, is_folder=False):
        if not file_path or not hasattr(self, "win") or not self.win:
            return False
        was_empty = not self.win.playlist
        if is_folder:
            self.win.add_folder_to_playlist(file_path)
        else:
            self.win.add_to_playlist([file_path])
        if was_empty and self.win.playlist:
            self.win.play_track(0)
        return True

    def on_cleanup_invalid(self, action, param):
        if hasattr(self, "win") and self.win:
            self.win.cleanup_invalid_tracks()

    def on_resize_to_cover_art(self, action, param):
        if hasattr(self.win, "cover_art_widget") and self.win.cover_art_widget:
            self.win.resize_to_cover_art()


def main() -> None:
    app = LinAmpApp()
    app.run(sys.argv)


if __name__ == "__main__":
    sys.exit(main())
