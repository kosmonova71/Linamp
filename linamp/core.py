#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import ipaddress
import json
import locale
import logging
import math
import os
import re
import socket
import sys
import tempfile
import threading
import time
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

try:
    import defusedxml.ElementTree as DefusedET
except ImportError as e:
    raise RuntimeError(
        "defusedxml is required for XML feed parsing — "
        "install it with: pip install defusedxml"
    ) from e
try:
    import gi

    gi.require_version("Gdk", "4.0")
    gi.require_version("GdkPixbuf", "2.0")
    gi.require_version("Gtk", "4.0")
    gi.require_version("Gst", "1.0")
    from gi.repository import GdkPixbuf, Gst, Gtk
except ImportError as e:
    raise RuntimeError(
        "GTK4 or GStreamer not installed. Please install them to run this application."
    ) from e

Gst.init(None)
APP_ID = "org.linamp.LinAmp"
APP_NAME = "LinAmp"
CONFIG_DIR = Path.home() / ".config" / "linamp"
CACHE_DIR = Path.home() / ".cache" / "linamp"
PLAYLIST_FILE = CONFIG_DIR / "playlist.json"
SETTINGS_FILE = CONFIG_DIR / "settings.json"
_THEME_CANDIDATES = (
    Path(__file__).resolve().parent / "theme" / "gtk.css",
    Path(__file__).resolve().parent.parent / "theme" / "gtk.css",
)
THEME_FILE = next(
    (path for path in _THEME_CANDIDATES if path.is_file()), _THEME_CANDIDATES[0]
)
RADIO_LOGO_CACHE_DIR = CACHE_DIR / "radio_logos"
PODCAST_ART_CACHE_DIR = CACHE_DIR / "podcast_art"
COVER_CACHE_DIR = CACHE_DIR / "covers"
LYRICS_CACHE_DIR = CACHE_DIR / "lyrics"
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("linamp")
logger = logging.getLogger(__name__)
try:
    locale.setlocale(locale.LC_ALL, "C")
except locale.Error:
    pass


def _thread_excepthook(args):
    """Log unhandled exceptions from worker threads.

    Without this, daemon-thread crashes (e.g. the DNS-validation thread
    that was seen raising ``socket.gaierror`` before it was wrapped) are
    only printed to stderr by Python's default hook — invisible in a GTK
    GUI application.  Routing them through the application logger makes
    them visible in the normal log output and avoids silent loss.
    """
    thread_name = (
        getattr(args.thread, "name", None)
        or getattr(args.thread, "ident", "unknown")
    )
    logger.error(
        "Unhandled exception in thread %s: %s",
        thread_name,
        args.exc_value,
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
    )


threading.excepthook = _thread_excepthook
MilkdropRenderer: Any = None
RendererConfig: Any = None
RendererSettings: Any = None
MILKDROP_AVAILABLE: bool = False
try:
    from milkdrop_renderer import (
        MilkdropRenderer as _MilkdropRenderer,
    )
    from milkdrop_renderer import (
        RendererConfig as _RendererConfig,
    )
    from milkdrop_renderer import (
        RendererSettings as _RendererSettings,
    )

    MilkdropRenderer = _MilkdropRenderer
    RendererConfig = _RendererConfig
    RendererSettings = _RendererSettings
    MILKDROP_AVAILABLE = True
except ImportError as exc:
    _missing = getattr(exc, "name", None)
    if _missing is not None and _missing.split(".")[0] in {
        "numpy",
        "OpenGL",
        "PIL",
        "gi",
    }:
        logging.getLogger(__name__).warning(
            "Milkdrop optional dependency missing: %s", exc
        )
    else:
        logging.getLogger(__name__).exception(
            "milkdrop_renderer failed to import (%s) - this is a code bug, "
            "not a missing optional dependency",
            type(exc).__name__,
        )
np: Any = None
NUMPY_AVAILABLE: bool = False
try:
    import numpy as _np

    np = _np
    NUMPY_AVAILABLE = True
except ImportError:
    pass
MetadataFixer: Any = None
_extract_id3_text: Any = None
METADATA_FIXER_AVAILABLE: bool = False
try:
    from metadata_fixer import (
        MetadataFixer as _MetadataFixer,
    )
    from metadata_fixer import (
        _extract_id3_text as _extract_id3_text_func,
    )

    MetadataFixer = _MetadataFixer
    _extract_id3_text = _extract_id3_text_func
    METADATA_FIXER_AVAILABLE = True
except ImportError:
    pass
requests: Any = None
Image: Any = None
ImageEnhance: Any = None
REQUESTS_AVAILABLE: bool = False
PIL_AVAILABLE: bool = False
try:
    import requests as _requests
    from PIL import Image as _Image
    from PIL import ImageEnhance as _ImageEnhance

    requests = _requests
    Image = _Image
    ImageEnhance = _ImageEnhance
    REQUESTS_AVAILABLE = True
    PIL_AVAILABLE = True
except ImportError:
    pass
mutagen: Any = None
File: Any = None
MP3: Any = None
ID3: Any = None
FLAC: Any = None
MP4: Any = None
MutagenError: type[Exception] = Exception
try:
    import mutagen as _mutagen
    from mutagen import File as _File
    from mutagen.flac import FLAC as _FLAC
    from mutagen.id3 import ID3 as _ID3
    from mutagen.mp3 import MP3 as _MP3
    from mutagen.mp4 import MP4 as _MP4

    mutagen = _mutagen
    File = _File
    MP3 = _MP3
    ID3 = _ID3
    FLAC = _FLAC
    MP4 = _MP4
    MutagenError = _mutagen.MutagenError
except ImportError:
    pass
if not NUMPY_AVAILABLE and any(
    name in sys.argv[0].lower() for name in ["visualizer", "projectm"]
):
    sys.stderr.write(
        "NumPy is required for the visualizer but is not installed.\n"
        "Install it with: pip install numpy\n"
    )
    sys.exit(1)
COVER_ART_DEPENDENCIES = {}
COVER_ART_DEPENDENCIES["requests"] = {
    "available": REQUESTS_AVAILABLE,
    "version": getattr(requests, "__version__", None) if requests else None,
}
COVER_ART_DEPENDENCIES["pillow"] = {
    "available": PIL_AVAILABLE,
    "version": getattr(Image, "__version__", None) if PIL_AVAILABLE else None,
}
if mutagen is not None:
    COVER_ART_DEPENDENCIES["mutagen"] = {
        "available": True,
        "version": getattr(
            mutagen, "__version__", getattr(mutagen, "version_string", "unknown")
        ),
    }
else:
    COVER_ART_DEPENDENCIES["mutagen"] = {
        "available": False,
        "version": None,
        "error": "mutagen not installed",
    }
COVER_ART_DEPENDENCIES["gtk"] = {
    "available": True,
    "version": getattr(Gtk, "get_major_version", lambda: 4)(),
}
COVER_ART_DEPENDENCIES["gdkpixbuf"] = {
    "available": True,
    "version": getattr(GdkPixbuf, "get_major_version", lambda: 0)(),
}
COVER_DOWNLOAD_AVAILABLE = REQUESTS_AVAILABLE and PIL_AVAILABLE
COVER_ART_AVAILABLE = (
    COVER_DOWNLOAD_AVAILABLE
    and COVER_ART_DEPENDENCIES["gtk"]["available"]
    and COVER_ART_DEPENDENCIES["gdkpixbuf"]["available"]
)
_DOWNLOAD_EXECUTOR: ThreadPoolExecutor | None = None
_DOWNLOAD_EXECUTOR_LOCK = threading.Lock()
_DOWNLOAD_EXECUTOR_RETIRED = False


def _get_download_executor() -> ThreadPoolExecutor | None:
    global _DOWNLOAD_EXECUTOR
    if _DOWNLOAD_EXECUTOR_RETIRED:
        return None
    if _DOWNLOAD_EXECUTOR is None:
        with _DOWNLOAD_EXECUTOR_LOCK:
            if _DOWNLOAD_EXECUTOR is None:
                if _DOWNLOAD_EXECUTOR_RETIRED:
                    return None
                _DOWNLOAD_EXECUTOR = ThreadPoolExecutor(
                    max_workers=4, thread_name_prefix="Download"
                )
    return _DOWNLOAD_EXECUTOR


def _shutdown_download_executor() -> None:
    global _DOWNLOAD_EXECUTOR, _DOWNLOAD_EXECUTOR_RETIRED
    with _DOWNLOAD_EXECUTOR_LOCK:
        _DOWNLOAD_EXECUTOR_RETIRED = True
        if _DOWNLOAD_EXECUTOR is None:
            return
        executor = _DOWNLOAD_EXECUTOR
        _DOWNLOAD_EXECUTOR = None
        executor.shutdown(wait=False, cancel_futures=True)


DEFAULT_VOLUME = 0.7
BEAT_DETECTION_INTERVAL = 50
BEAT_BASS_EMA_DECAY = 0.9
BEAT_BASS_EMA_RISE = 0.1
BEAT_THRESHOLD_MULTIPLIER = 1.35
BEAT_MIN_BASS_LEVEL = 0.08
BEAT_MIN_INTERVAL_SECONDS = 0.18
LEVELS_ATTACK = 0.4
LEVELS_DECAY = 0.08
BEAT_SMOOTH_ATTACK = 0.5
BEAT_SMOOTH_DECAY = 0.1
ATT_ATTACK = 0.15
ATT_DECAY = 0.02
BEAT_FLUX_SENSITIVITY = 2.0
BEAT_FLUX_DECAY = 0.94
BEAT_MIN_FLUX = 0.001
BEAT_ENVELOPE_DECAY = 0.93
BEAT_SOFT_FACTOR = 0.4
AUTO_GAIN_TARGET = 0.75
AUTO_GAIN_ADAPT = 0.01
AUTO_GAIN_SENSITIVITY = 0.5
SPECTRUM_BINS = 64
WAVEFORM_BUFFER_SAMPLES = 576
AUDIO_SNAPSHOT_BUFFER = 16
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
MAX_IMAGE_PIXELS = 20_000_000
PLAYBACK_VERIFY_INTERVAL_MS = 50
PLAYBACK_VERIFY_MAX_ATTEMPTS = 200
PLAYBACK_RECOVERY_MAX_ATTEMPTS = 3
RADIO_RECONNECT_MAX_FAILURES = 10
MAX_REPORTED_SCAN_ERRORS = 20
_LOCAL_IMAGE_SUFFIXES = frozenset(
    {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff"}
)
RADIO_STATIONS_FILE = CONFIG_DIR / "radio_stations.json"
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
PODCASTS_CONFIG_DIR = CONFIG_DIR
PODCASTS_FILE = PODCASTS_CONFIG_DIR / "podcasts.json"
PODCAST_EPISODES_DIR = PODCASTS_CONFIG_DIR / "podcast_episodes"
PODCAST_STATE_JOURNAL = PODCASTS_CONFIG_DIR / "podcasts_state.journal"
MAX_EPISODES_PER_FEED = 200
_PODCAST_STATE_LOCK = threading.RLock()
_FED_ID_RE = re.compile(r"[A-Za-z0-9_\-]+")


def _is_safe_feed_id(feed_id: str) -> bool:
    """Validate that *feed_id* is safe to use as a filename component.
    Real identifiers are 32-character MD5 hex digests, but the check is
    intentionally permissive (alphanumeric, underscore, hyphen) to accept
    any identifier that cannot traverse paths or clobber neighbouring files.
    """
    return bool(feed_id and isinstance(feed_id, str) and _FED_ID_RE.fullmatch(feed_id))


_DeferredPersistEntry = namedtuple("_DeferredPersistEntry", ["callback", "snapshot"])


def _image_cache_path(url: str, cache_dir: Path) -> Path:
    key = hashlib.md5(url.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{key}.png"


CACHE_DIR_MAX_BYTES: dict[str, int] = {
    "radio_logos": 64 * 1024 * 1024,
    "podcast_art": 128 * 1024 * 1024,
    "covers": 256 * 1024 * 1024,
    "lyrics": 32 * 1024 * 1024,
}
_CACHE_PRUNE_INTERVAL = 300.0
_cache_last_prune: dict[str, float] = {}
_cache_prune_lock = threading.Lock()


def _unique_cache_temp_path(cache_path: Path) -> Path:
    """Return a per-writer temp path next to *cache_path*.
    A shared ``<key>.tmp`` name lets two cache workers (the download pool and
    the cover pool are both multi-threaded) truncate and interleave each
    other's write, after which the loser's ``os.replace`` publishes a corrupt
    file.  The temp name is also deliberately not ``<key>.tmp`` so that
    ``CoverArt.get_cache_path``/``prune_cache_dir`` never mistake an
    in-flight temp for a cache entry.
    """
    return cache_path.with_name(
        f".{cache_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )


def _write_file_atomic(target: Path, write_body) -> bool:
    """Write *target* via a unique temp file plus ``os.replace``.
    ``write_body`` receives the open temp file object.  Mirrors
    ``_atomic_write_json`` but for arbitrary producers (Pillow ``img.save``,
    raw bytes) that cannot be serialised to JSON.
    """
    tmp = _unique_cache_temp_path(target)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "wb") as f:
            write_body(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
        return True
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def prune_cache_dir(cache_dir: Path, max_bytes: int, *, force: bool = False) -> int:
    """Delete least-recently-modified files until *cache_dir* fits its budget.
    Returns the number of bytes reclaimed.  Pruning is best-effort: a cache
    that cannot be inspected or written is left alone rather than raising
    into the download path.  In-flight ``*.tmp`` files and symlinks are
    skipped: the former belong to a live writer, and the latter would charge
    the cache for their target's size.
    """
    if max_bytes <= 0:
        return 0
    if not cache_dir.exists():
        return 0
    now = time.monotonic()
    with _cache_prune_lock:
        last = _cache_last_prune.get(str(cache_dir), 0.0)
        if not force and (now - last) < _CACHE_PRUNE_INTERVAL:
            return 0
        _cache_last_prune[str(cache_dir)] = now
    entries: list[tuple[float, int, Path]] = []
    total = 0
    try:
        for path in cache_dir.iterdir():
            if path.suffix == ".tmp" or path.is_symlink() or not path.is_file():
                continue
            try:
                stat = path.lstat()
            except OSError:
                continue
            entries.append((stat.st_mtime, stat.st_size, path))
            total += stat.st_size
    except OSError:
        return 0
    if total <= max_bytes:
        return 0
    entries.sort(key=lambda item: item[0])
    reclaimed = 0
    for _mtime, size, path in entries:
        if total <= max_bytes:
            break
        try:
            path.unlink(missing_ok=True)
        except OSError:
            continue
        total -= size
        reclaimed += size
    return reclaimed


def _linamp_attr(name: str, fallback):
    """Resolve *name* from the live ``linamp`` package namespace.

    Lets tests monkeypatch private helpers (``_validate_url_dns``,
    ``_safe_http_get``, ``_read_capped_body``, etc.) by patching the
    ``linamp`` module object rather than having to know which internal
    module the local binding lives in.
    """
    mod = sys.modules.get("linamp")
    if mod is not None:
        value = getattr(mod, name, None)
        if value is not None:
            return value
    return fallback


def _maybe_prune_cache(cache_dir: Path) -> None:
    """Rate-limited prune for a known cache directory."""
    budget = CACHE_DIR_MAX_BYTES.get(cache_dir.name)
    if budget:
        prune_cache_dir(cache_dir, budget)


def prune_all_caches() -> None:
    """Bring every on-disk cache back within its byte budget.
    Called once at startup so a cache that outgrew its budget while the app
    was not running is trimmed even if nothing new is downloaded.
    """
    for directory in (
        RADIO_LOGO_CACHE_DIR,
        PODCAST_ART_CACHE_DIR,
        COVER_CACHE_DIR,
        LYRICS_CACHE_DIR,
    ):
        budget = CACHE_DIR_MAX_BYTES.get(directory.name)
        if budget:
            prune_cache_dir(directory, budget, force=True)


def fetch_image_to_cache(
    url: str,
    cache_dir: Path,
    thumbnail_size: tuple[int, int] = (128, 128),
    timeout: float = 8.0,
    allow_local: bool = False,
) -> Path | None:
    """Download an image to the local cache and return its path.
    *allow_local* permits a filesystem path instead of an http(s) URL.  It is
    only safe for values the user authored themselves (radio station logos);
    feed-supplied artwork must never be able to point the app at an arbitrary
    local file such as its own settings or keyring data.
    """
    if not url or not isinstance(url, str):
        return None
    if not url.startswith(("http://", "https://")):
        if not allow_local:
            return None
        p = Path(url).expanduser()
        try:
            real = p.resolve()
        except OSError:
            return None
        if not p.is_absolute() or not real.is_file():
            return None
        if real.suffix.lower() not in _LOCAL_IMAGE_SUFFIXES:
            return None
        for private in (
            PODCASTS_CONFIG_DIR,
            PODCAST_EPISODES_DIR,
            COVER_CACHE_DIR,
            LYRICS_CACHE_DIR,
            RADIO_LOGO_CACHE_DIR,
        ):
            try:
                if private == private.parent or real.is_relative_to(private):
                    return None
            except (OSError, ValueError):
                return None
        return real
    try:
        # Allow monkeypatching via the linamp module namespace
        _validate_dns = _linamp_attr("_validate_url_dns", _validate_url_dns)
        if not _validate_dns(url):
            return None
    except (ValueError, ipaddress.AddressValueError, OSError, socket.gaierror):
        return None
    cache_path = _image_cache_path(url, cache_dir)
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path
    if not REQUESTS_AVAILABLE or not PIL_AVAILABLE:
        return None
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        _http_get = _linamp_attr("_safe_http_get", _safe_http_get)
        resp = _http_get(url, timeout=timeout, stream=True)
        try:
            if resp is None or resp.status_code != 200:
                return None
            _read_body = _linamp_attr("_read_capped_body", _read_capped_body)
            content = _read_body(resp, MAX_IMAGE_CONTENT_SIZE)
            if content is None:
                return None
            img: Any = Image.open(BytesIO(content))
            if img.width * img.height > MAX_IMAGE_PIXELS:
                raise ValueError("image pixel count exceeds limit")
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGBA")
            else:
                img = img.convert("RGB")
            img.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
            _write_file_atomic(
                cache_path, lambda f: img.save(f, format="PNG", optimize=True)
            )
            _maybe_prune_cache(cache_dir)
            return cache_path
        finally:
            if resp is not None:
                resp.close()
    except (OSError, ValueError) as e:
        logging.getLogger(__name__).debug("Failed to fetch image from %s: %s", url, e)
        return None


def _validate_file_path(file_path: str, allow_symlinks: bool = True) -> bool:
    if not file_path or not isinstance(file_path, str):
        return False
    if "\0" in file_path:
        return False
    try:
        normalized_path = os.path.abspath(os.path.expanduser(file_path))
        real_path = os.path.realpath(normalized_path)
    except (OSError, ValueError):
        return False
    if real_path.startswith(("/proc/", "/sys/")):
        return False
    if not allow_symlinks and os.path.islink(normalized_path):
        return False
    return os.path.isfile(real_path)


def _is_valid_number(value: Any) -> bool:
    """Return True only for a real, finite numeric value (excludes bool, NaN, inf).
    ``math.isfinite`` raises ``OverflowError`` for ints too large to convert to
    float, and a JSON integer literal of arbitrary length decodes to a Python
    int, so a corrupt/hand-edited settings file could otherwise raise out of
    ``load_settings`` (which only catches ``ValueError``-family errors) and
    crash startup.  Treat any non-convertible magnitude as invalid.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, ValueError):
        return False


def _is_valid_number_in_range(
    value: Any, minimum: float | None = None, maximum: float | None = None
) -> bool:
    """Return True for a valid number within optional [minimum, maximum] bounds."""
    if not _is_valid_number(value):
        return False
    if minimum is not None and value < minimum:
        return False
    return not (maximum is not None and value > maximum)


def _normalize_hostname(hostname: str) -> str | None:
    """Canonicalize hostname text for validation.

    Raw URL hostnames may include a trailing dot (for example
    ``localhost.`` or ``127.0.0.1.``).  The trailing dot is semantically
    insignificant for host matching, and canonicalizing it prevents loopback
    and private addresses from being misclassified as public hostnames.
    """
    if not isinstance(hostname, str):
        return None
    normalized = hostname.strip().rstrip(".")
    if not normalized:
        return None
    return normalized.lower()


def _looks_like_ip_literal(hostname: str) -> bool:
    """True when *hostname* is shaped like an address rather than a name.
    Anything made only of dot-separated number groups is meant to be an IP, so
    an unparseable one ("999.999.999.999", "127.1") must never be waved through
    as an ordinary hostname -- that only stays safe while a later DNS lookup
    happens to reject it.
    """
    hostname = _normalize_hostname(hostname) or ""
    if ":" in hostname:
        return True
    return re.fullmatch(r"\d{1,3}(?:\.\d{1,3})+", hostname) is not None


def _parse_ip_literal(
    hostname: str,
) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    normalized = _normalize_hostname(hostname)
    if normalized is None or not _looks_like_ip_literal(normalized):
        return None
    try:
        return ipaddress.ip_address(normalized)
    except (ValueError, TypeError):
        return None


_SHARED_ADDRESS_SPACE = ipaddress.ip_network("100.64.0.0/10")


def _is_public_address(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True only for globally routable unicast addresses.
    The deny-list covers the well-known private / loopback / link-local /
    reserved / multicast buckets.  ``addr.is_global`` is an additional,
    defence-in-depth gate: Python's ``ipaddress`` flags are periodically updated
    to track new special-use registries, and any address that is *not* marked
    global -- even when it slips past the explicit deny-list -- is treated as
    private here.  This catches, for example, IPv6 unique-local addresses
    (``fc00::/7``) on Python versions whose ``is_private`` flag lags behind the
    current IANA registry.
    """
    if (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
        or not addr.is_global
    ):
        return False
    mapped = getattr(addr, "ipv4_mapped", None)
    if mapped is not None:
        addr = mapped
    return addr not in _SHARED_ADDRESS_SPACE


def _validate_url(url: str) -> bool:
    if not isinstance(url, str):
        return False
    try:
        parsed = urlparse(url)
    except (TypeError, ValueError):
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    hostname = _normalize_hostname(parsed.hostname or "")
    if not hostname:
        return False
    if hostname in ("localhost", "127.0.0.1", "::1"):
        return False
    addr = _parse_ip_literal(hostname)
    if addr is None:
        return not _looks_like_ip_literal(hostname)
    return _is_public_address(addr)


_DNS_VALIDATION_CACHE: dict[str, tuple[float, bool]] = {}
_DNS_VALIDATION_CACHE_LOCK = threading.Lock()
_DNS_VALIDATION_CACHE_MAX = 256
_DNS_VALIDATION_TTL = 60.0
_DNS_RESOLVE_TIMEOUT = 5.0


def _resolve_hostname(hostname: str, timeout: float) -> tuple[list | None, bool]:
    """Resolve *hostname*; return ``(getaddrinfo results, completed)``.
    ``completed`` is False when the lookup was abandoned at *timeout*.  The
    abandoned thread only holds a socket lookup and exits on its own once the
    system resolver gives up, so nothing is leaked -- but it must not publish
    a verdict afterwards, which is why the caller needs the flag.
    """
    box: list = []
    done = threading.Event()

    def _lookup():
        try:
            box.append(socket.getaddrinfo(hostname, None))
        except (OSError, socket.gaierror, TypeError, ValueError) as exc:
            logger.debug("DNS resolution for %s failed: %s", hostname, exc)
            box.append(None)
        except Exception:
            logger.debug(
                "Unexpected error during DNS resolution for %s",
                hostname,
                exc_info=True,
            )
            box.append(None)
        finally:
            done.set()

    worker = threading.Thread(
        target=_lookup, daemon=True, name=f"DnsValidate-{hostname[:32]}"
    )
    worker.start()
    if not done.wait(timeout):
        return None, False
    return (box[0] if box else None), True


def _validate_url_dns(url: str) -> bool:
    """Validate URL with DNS resolution to catch private-address hostnames.
    Resolves the hostname via DNS and rejects private/loopback/link-local
    addresses, protecting against DNS-rebinding SSRF attacks on
    user-controlled network inputs (podcast feeds, artwork URLs).
    Results are memoised per host for ``_DNS_VALIDATION_TTL`` seconds:
    redirect chains, cover-art searches and station lists otherwise re-resolve
    the same host many times, each of them synchronously on the caller's
    thread.  The memo deliberately expires -- see ``_DNS_VALIDATION_CACHE``.
    Note: this is DNS-only validation. A race between getaddrinfo() here
    and requests' own DNS lookup can still theoretically allow DNS rebinding.
    Connection-level address pinning would be required for stronger guarantees.
    """
    if not _validate_url(url):
        return False
    parsed = urlparse(url)
    hostname = _normalize_hostname(parsed.hostname or "")
    if not hostname:
        return False
    if _parse_ip_literal(hostname) is not None:
        return True
    cache_key = hostname
    now = time.monotonic()
    with _DNS_VALIDATION_CACHE_LOCK:
        cached = _DNS_VALIDATION_CACHE.get(cache_key)
        if cached is not None and now < cached[0]:
            return cached[1]
    # Allow monkeypatching
    import sys

    timeout = _DNS_RESOLVE_TIMEOUT
    for mod_name in ("linamp",):
        mod = sys.modules.get(mod_name)
        if mod and hasattr(mod, "_DNS_RESOLVE_TIMEOUT"):
            timeout = mod._DNS_RESOLVE_TIMEOUT
            break
    result = _validate_resolved_hostname(cache_key, timeout, _DNS_VALIDATION_CACHE)
    return result


def _validate_resolved_hostname(
    cache_key: str, timeout: float, cache: dict[str, tuple[float, bool]]
) -> bool:
    """Resolve *cache_key* and cache the SSRF verdict until the TTL expires."""
    addrinfo, completed = _resolve_hostname(cache_key, timeout)
    if not completed:
        logging.getLogger(__name__).warning(
            "DNS validation for %s did not complete within %.1fs; rejecting",
            cache_key,
            timeout,
        )
        return False
    if addrinfo is None:
        logger.debug(
            "DNS resolution for %s returned no results; rejecting",
            cache_key,
        )
        return False
    verdict = _dns_verdict(addrinfo)
    with _DNS_VALIDATION_CACHE_LOCK:
        if len(cache) >= _DNS_VALIDATION_CACHE_MAX:
            cache.clear()
        cache[cache_key] = (time.monotonic() + _DNS_VALIDATION_TTL, verdict)
    return verdict


def _dns_verdict(addrinfo) -> bool:
    """Return True only if every resolved address is a public one.
    Both verdicts are cached by the caller: re-resolving a host that already
    answered does not change the answer within the TTL, and a feed listing
    thousands of entries pointing at one hostname must not trigger thousands
    of blocking lookups.
    """
    public_found = False
    for _family, _stype, _proto, _canname, sockaddr in addrinfo:
        addr_str = sockaddr[0]
        try:
            addr = ipaddress.ip_address(addr_str)
        except ValueError:
            continue
        if not _is_public_address(addr):
            return False
        public_found = True
    return public_found


_SHARED_SESSION_THREAD = threading.local()
_SHARED_SESSION_LOCK = threading.Lock()
_MAX_REDIRECTS = 5
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 LinAmp/1.0"


def _get_shared_session() -> Any:
    """Return a lazily-initialized thread-local requests.Session.
    One session per thread avoids concurrent-mutation issues with
    requests.Session while still reusing connections within a thread
    (e.g. during radio-logo preloading or cover-art downloads).
    """
    session = getattr(_SHARED_SESSION_THREAD, "session", None)
    if session is None and REQUESTS_AVAILABLE:
        with _SHARED_SESSION_LOCK:
            session = getattr(_SHARED_SESSION_THREAD, "session", None)
            if session is None:
                session = requests.Session()
                session.headers.update({"User-Agent": USER_AGENT})
                _SHARED_SESSION_THREAD.session = session
    return session


_MAX_SEARCH_PAYLOAD = 2 * 1024 * 1024


def _read_capped_body(response: Any, limit: int) -> bytes | None:
    """Read at most *limit* decoded bytes from a streamed response.
    ``response.text`` / ``response.json()`` materialise whatever the server
    sends.  The request timeout bounds wall-clock time, not bytes, so a
    trickling response would be buffered without any limit.  Returns ``None``
    when the cap is exceeded or the body cannot be read.
    """
    if response is None:
        return None
    try:
        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_content(64 * 1024):
            if not chunk:
                continue
            total += len(chunk)
            if total > limit:
                return None
            chunks.append(chunk)
        return b"".join(chunks)
    except (OSError, ValueError, TypeError, AttributeError):
        return None
    except Exception as exc:
        logging.getLogger(__name__).debug(
            "Capped body read failed: %s", exc, exc_info=True
        )
        return None


def _safe_http_get(
    url: str,
    session: Any = None,
    timeout: float | tuple[int, int] = 8.0,
    stream: bool = False,
    method: str = "GET",
    **kwargs: Any,
) -> Any:
    """GET *url* following redirects while re-validating every hop.
    ``requests`` follows redirects by default, so a hostname that initially
    resolves to a public address could 302-redirect to a private/internal
    address.  This wrapper validates each redirect destination with
    :func:`_validate_url_dns`, preventing DNS-rebinding / SSRF bypass on
    user-controlled network inputs (podcast feeds, artwork, radio logos,
    cover downloads).
    """
    # Allow monkeypatching
    import sys

    requests_available = REQUESTS_AVAILABLE
    for mod_name in ("linamp",):
        mod = sys.modules.get(mod_name)
        if mod and hasattr(mod, "REQUESTS_AVAILABLE"):
            requests_available = mod.REQUESTS_AVAILABLE
            break
    if not requests_available:
        return None
    if session is None:
        # Allow monkeypatching via linamp module
        import sys

        _get_sess = _get_shared_session
        for mod_name in ("linamp",):
            mod = sys.modules.get(mod_name)
            if mod and hasattr(mod, "_get_shared_session"):
                fn = mod._get_shared_session
                if fn is not _get_shared_session:
                    _get_sess = fn
                    break
        sess = _get_sess()
    else:
        sess = session
    if sess is None:
        return None
    current_url = url
    for _ in range(_MAX_REDIRECTS + 1):
        _validate_dns = _linamp_attr("_validate_url_dns", _validate_url_dns)
        if not _validate_dns(current_url):
            logging.getLogger(__name__).warning(
                "SSRF block: refusing HTTP %s — URL did not pass DNS validation",
                current_url[:200],
            )
            return None
        try:
            resp = sess.request(
                method,
                current_url,
                timeout=timeout,
                verify=True,
                stream=stream,
                allow_redirects=False,
                **kwargs,
            )
        except requests.exceptions.RequestException:
            return None
        if resp.status_code in (301, 302, 303, 307, 308):
            location = resp.headers.get("Location")
            resp.close()
            if not location:
                return None
            current_url = urljoin(current_url, location)
            continue
        return resp
    return None


def _get_audio_duration(path: str) -> int:
    if not File:
        return 0
    try:
        audio = File(path)
        if audio is not None:
            length = audio.info.length if audio.info else 0
            return int(length)
    except (MutagenError, OSError):
        logging.getLogger(__name__).debug(
            "Could not extract duration for %s", path, exc_info=True
        )
    return 0


def _fsync_dir(path: Path) -> None:
    """Flush directory metadata to disk to durabilize file replacements."""
    try:
        dir_fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass


def _atomic_write_json(path: Path, data: Any, mode: int | None = None) -> bool:
    """Write JSON to *path* atomically via temp-file + fsync + rename.
    If *mode* is given, both the temp file and the final file are chmod'd
    to that permission mask (e.g. ``mode=0o600`` for sensitive data).
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=f".{path.name}_", suffix=".tmp", dir=path.parent, text=True
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            if mode is not None:
                os.chmod(tmp_path, mode)
            os.replace(tmp_path, path)
            if mode is not None:
                os.chmod(path, mode)
        except BaseException:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass
            raise
        _fsync_dir(path.parent)
        return True
    except (OSError, TypeError, ValueError):
        return False


__all__ = [
    "APP_ID",
    "APP_NAME",
    "ATT_ATTACK",
    "ATT_DECAY",
    "AUDIO_EXTENSIONS",
    "AUDIO_SNAPSHOT_BUFFER",
    "AUTO_GAIN_ADAPT",
    "AUTO_GAIN_SENSITIVITY",
    "AUTO_GAIN_TARGET",
    "AUTO_SAVE_DELAY",
    "BEAT_BASS_EMA_DECAY",
    "BEAT_BASS_EMA_RISE",
    "BEAT_DETECTION_INTERVAL",
    "BEAT_ENVELOPE_DECAY",
    "BEAT_FLUX_DECAY",
    "BEAT_FLUX_SENSITIVITY",
    "BEAT_MIN_BASS_LEVEL",
    "BEAT_MIN_FLUX",
    "BEAT_MIN_INTERVAL_SECONDS",
    "BEAT_SMOOTH_ATTACK",
    "BEAT_SMOOTH_DECAY",
    "BEAT_SOFT_FACTOR",
    "BEAT_THRESHOLD_MULTIPLIER",
    "CACHE_DIR",
    "CACHE_DIR_MAX_BYTES",
    "CONFIG_DIR",
    "COVER_ART_AVAILABLE",
    "COVER_ART_DEPENDENCIES",
    "COVER_CACHE_DIR",
    "COVER_DOWNLOAD_AVAILABLE",
    "DEFAULT_RADIO_STATIONS",
    "DEFAULT_VOLUME",
    "LEVELS_ATTACK",
    "LEVELS_DECAY",
    "LOGGER",
    "LYRICS_CACHE_DIR",
    "MAX_EPISODES_PER_FEED",
    "MAX_FEED_CONTENT_SIZE",
    "MAX_IMAGE_CONTENT_SIZE",
    "MAX_IMAGE_PIXELS",
    "MAX_REPORTED_SCAN_ERRORS",
    "PERIODIC_SAVE_INTERVAL",
    "PLAYBACK_RECOVERY_MAX_ATTEMPTS",
    "PLAYBACK_VERIFY_INTERVAL_MS",
    "PLAYBACK_VERIFY_MAX_ATTEMPTS",
    "PLAYLIST_FILE",
    "PODCASTS_CONFIG_DIR",
    "PODCASTS_FILE",
    "PODCAST_ART_CACHE_DIR",
    "PODCAST_EPISODES_DIR",
    "PODCAST_STATE_JOURNAL",
    "RADIO_LOGO_CACHE_DIR",
    "RADIO_RECONNECT_MAX_FAILURES",
    "RADIO_STATIONS_FILE",
    "SETTINGS_FILE",
    "SPECTRUM_BINS",
    "THEME_FILE",
    "UPDATE_DISPLAY_INTERVAL",
    "USER_AGENT",
    "WAVEFORM_BUFFER_SAMPLES",
    "_CACHE_PRUNE_INTERVAL",
    "_DNS_RESOLVE_TIMEOUT",
    "_DNS_VALIDATION_CACHE",
    "_DNS_VALIDATION_CACHE_LOCK",
    "_DNS_VALIDATION_CACHE_MAX",
    "_DNS_VALIDATION_TTL",
    "_DOWNLOAD_EXECUTOR_LOCK",
    "_DOWNLOAD_EXECUTOR_RETIRED",
    "_FED_ID_RE",
    "_LOCAL_IMAGE_SUFFIXES",
    "_MAX_REDIRECTS",
    "_MAX_SEARCH_PAYLOAD",
    "_PODCAST_STATE_LOCK",
    "_SHARED_ADDRESS_SPACE",
    "_SHARED_SESSION_LOCK",
    "_SHARED_SESSION_THREAD",
    "DefusedET",
    "MutagenError",
    "_DeferredPersistEntry",
    "_atomic_write_json",
    "_cache_prune_lock",
    "_dns_verdict",
    "_fsync_dir",
    "_get_audio_duration",
    "_get_download_executor",
    "_get_shared_session",
    "_image_cache_path",
    "_is_public_address",
    "_is_safe_feed_id",
    "_is_valid_number",
    "_is_valid_number_in_range",
    "_looks_like_ip_literal",
    "_maybe_prune_cache",
    "_parse_ip_literal",
    "_read_capped_body",
    "_resolve_hostname",
    "_safe_http_get",
    "_shutdown_download_executor",
    "_unique_cache_temp_path",
    "_validate_file_path",
    "_validate_resolved_hostname",
    "_validate_url",
    "_validate_url_dns",
    "_write_file_atomic",
    "fetch_image_to_cache",
    "logger",
    "prune_all_caches",
    "prune_cache_dir",
]
