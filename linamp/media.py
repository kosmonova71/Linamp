from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .core import (
    _LOCAL_IMAGE_SUFFIXES,
    _MAX_SEARCH_PAYLOAD,
    COVER_ART_DEPENDENCIES,
    COVER_CACHE_DIR,
    COVER_DOWNLOAD_AVAILABLE,
    FLAC,
    ID3,
    LYRICS_CACHE_DIR,
    MAX_IMAGE_CONTENT_SIZE,
    MP3,
    MP4,
    PIL_AVAILABLE,
    REQUESTS_AVAILABLE,
    File,
    Image,
    ImageEnhance,
    MutagenError,
    _atomic_write_json,
    _extract_id3_text,
    _get_shared_session,
    _linamp_attr,
    _maybe_prune_cache,
    _read_capped_body,
    _safe_http_get,
    _validate_file_path,
    _validate_url_dns,
    _write_file_atomic,
    mutagen,
    requests,
)

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


@dataclass
class CoverArt:
    path: str = ""
    source: str = "embedded"
    width: int = 0
    height: int = 0
    format: str = "unknown"
    bytes_data: bytes | None = None

    @staticmethod
    def get_cache_key(audio_path: str) -> str | None:
        try:
            stat = os.stat(audio_path)
        except OSError:
            return None
        file_data = f"{audio_path}:{stat.st_size}:{stat.st_mtime}"
        return hashlib.md5(file_data.encode()).hexdigest()[:16]

    @staticmethod
    def normalize_format(format_str: str) -> str:
        fmt = format_str.lower().strip(" .")
        if "/" in fmt:
            fmt = fmt.rsplit("/", 1)[-1]
        if fmt in ("jpg", "jpeg"):
            return "jpg"
        if fmt in ("png", "gif", "webp", "bmp"):
            return fmt
        return "jpg"

    @classmethod
    def get_cache_path(cls, audio_path: str) -> Path | None:
        key = cls.get_cache_key(audio_path)
        if key is None:
            return None
        _cache_dir = _linamp_attr("COVER_CACHE_DIR", COVER_CACHE_DIR)
        cache_files: list[Path] = [
            p
            for p in _cache_dir.glob(f"{key}.*")
            if p.suffix.lower() in _LOCAL_IMAGE_SUFFIXES
        ]
        if cache_files:
            return cache_files[0]
        return None

    def save_to_cache(self, audio_path: str) -> str | None:
        if not self.bytes_data:
            return None
        key = self.get_cache_key(audio_path)
        if key is None:
            return None
        ext = self.normalize_format(self.format)
        _cache_dir = _linamp_attr("COVER_CACHE_DIR", COVER_CACHE_DIR)
        cache_path = _cache_dir / f"{key}.{ext}"
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            _write_file_atomic(cache_path, lambda f: f.write(self.bytes_data))
            _maybe_prune_cache(_cache_dir)
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
            try:
                file_size = cache_path.stat().st_size
                if file_size > MAX_IMAGE_CONTENT_SIZE:
                    return None
                with open(cache_path, "rb") as f:
                    data = f.read()
                ext = cache_path.suffix.lower().lstrip(".")
                return cls(
                    path=str(cache_path),
                    source="cache",
                    bytes_data=data,
                    format=ext,
                )
            except OSError:
                return None
        return None


def _tag_key_matches(key: str | tuple, prefix: str) -> bool:
    if isinstance(key, str):
        return key.startswith(prefix)
    if isinstance(key, tuple):
        return any(isinstance(item, str) and item.startswith(prefix) for item in key)
    return str(key).startswith(prefix)


def _extract_cover_art_from_file(file_path: str) -> CoverArt | None:
    try:
        return _extract_cover_art_from_file_impl(file_path)
    except Exception as e:
        logging.getLogger(__name__).debug(
            "Cover extraction failed for %s: %s", file_path, e, exc_info=True
        )
        return None


def _extract_cover_art_from_file_impl(file_path: str) -> CoverArt | None:
    if not COVER_ART_DEPENDENCIES.get("mutagen", {}).get("available", False):
        return None
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
                file_size = os.path.getsize(cover_path)
                if file_size > MAX_IMAGE_CONTENT_SIZE:
                    continue
                with open(cover_path, "rb") as f:
                    data = f.read()
                ext = os.path.splitext(pattern)[1].lstrip(".")
                return CoverArt(
                    path=cover_path, source="folder", bytes_data=data, format=ext
                )
            except OSError:
                pass
    return None


def extract_cover_art(audio_path: str) -> tuple[CoverArt | None, str | None]:
    try:
        return _extract_cover_art_impl(audio_path)
    except Exception as e:
        logging.getLogger(__name__).debug(
            "Cover lookup failed for %s: %s", audio_path, e, exc_info=True
        )
        return None, None


_COVER_DISPLAY_MAX = 500


def _resize_cover_for_display(
    cover: CoverArt, max_size: int = _COVER_DISPLAY_MAX
) -> CoverArt:
    if PIL_AVAILABLE and cover.bytes_data:
        try:
            img = Image.open(BytesIO(cover.bytes_data))
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            buf = BytesIO()
            fmt = "PNG" if "A" in img.mode else "JPEG"
            img.save(buf, format=fmt)
            cover = CoverArt(
                path=cover.path,
                source=cover.source,
                width=img.width,
                height=img.height,
                format=cover.format,
                bytes_data=buf.getvalue(),
            )
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            logging.getLogger(__name__).debug(
                "Cover resize failed for %s: %s", cover.path, exc
            )
    return cover


def _extract_cover_art_impl(audio_path: str) -> tuple[CoverArt | None, str | None]:
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
        if not COVER_DOWNLOAD_AVAILABLE:
            raise ImportError("requests and PIL are required for cover downloading")
        self.timeout = (3, 5)

    def get_metadata(self, file_path: str) -> tuple[str, str]:
        if not _validate_file_path(file_path):
            return "", ""
        if not COVER_ART_DEPENDENCIES.get("mutagen", {}).get("available", False):
            return "", ""
        audio_path = Path(file_path)
        try:
            if audio_path.suffix.lower() == ".mp3":
                audio = MP3(file_path, ID3=ID3)
                artist = (audio.tags.get("TPE1") or [""])[0] if audio.tags else ""
                album = (audio.tags.get("TALB") or [""])[0] if audio.tags else ""
            elif audio_path.suffix.lower() == ".flac":
                audio = FLAC(file_path)
                artist = (audio.get("artist") or [""])[0]
                album = (audio.get("album") or [""])[0]
            elif audio_path.suffix.lower() in [".m4a", ".mp4"]:
                audio = MP4(file_path)
                artist = (audio.get("\xa9ART") or [""])[0]
                album = (audio.get("\xa9alb") or [""])[0]
            else:
                return "", ""
            return str(artist).strip(), str(album).strip()
        except (MutagenError, OSError, ValueError) as exc:
            logging.getLogger(__name__).debug(
                "Metadata extraction failed for %s: %s", file_path, exc, exc_info=True
            )
            return "", ""

    def search_cover(self, artist: str, album: str) -> str | None:
        if not artist or not album:
            return None
        query = f"{artist} {album} album cover"
        sources = [
            self._search_deezer,
            self._search_lastfm,
        ]
        for source_func in sources:
            try:
                url = source_func(query, artist, album)
            except (
                getattr(requests, "RequestException", OSError),
                OSError,
                ValueError,
                TypeError,
                AttributeError,
            ) as exc:
                logging.getLogger(__name__).debug(
                    "Cover search source %s failed: %s", source_func.__name__, exc
                )
                url = None
            if url:
                return url
            time.sleep(0.5)
        return None

    def _search_deezer(self, query: str, artist: str, album: str) -> str | None:
        search_url = f"https://api.deezer.com/search/album?q={quote(query)}"
        _validate_dns = _linamp_attr("_validate_url_dns", _validate_url_dns)
        if not _validate_dns(search_url):
            return None
        _http_get = _linamp_attr("_safe_http_get", _safe_http_get)
        _get_sess = _linamp_attr("_get_shared_session", _get_shared_session)
        try:
            response = _http_get(
                search_url,
                session=_get_sess(),
                timeout=self.timeout,
                stream=True,
            )
        except (
            getattr(requests, "RequestException", OSError),
            OSError,
            ValueError,
            TypeError,
            AttributeError,
        ) as exc:
            logging.getLogger(__name__).debug("Cover search HTTP failed: %s", exc)
            return None
        try:
            if response is not None and response.status_code == 200:
                _read_body = _linamp_attr("_read_capped_body", _read_capped_body)
                raw = _read_body(response, _MAX_SEARCH_PAYLOAD)
                if raw is None:
                    return None
                data = json.loads(raw)
                if not isinstance(data, dict):
                    return None
                results = data.get("data")
                if isinstance(results, list) and results:
                    album_data = results[0]
                    if not isinstance(album_data, dict):
                        return None
                    cover_url = (
                        album_data.get("cover_xl")
                        or album_data.get("cover_big")
                        or album_data.get("cover_medium")
                    )
                    if cover_url and _validate_dns(cover_url):
                        return str(cover_url) if cover_url else None
        except (ValueError, TypeError, AttributeError) as e:
            logging.getLogger(__name__).debug(
                "Deezer cover search returned an unexpected payload: %s", e
            )
        finally:
            if response is not None:
                response.close()
        return None

    def _search_lastfm(self, query: str, artist: str, album: str) -> str | None:
        search_url = f"https://www.last.fm/music/{quote(artist)}/{quote(album)}"
        _validate_dns = _linamp_attr("_validate_url_dns", _validate_url_dns)
        if not _validate_dns(search_url):
            return None
        _http_get = _linamp_attr("_safe_http_get", _safe_http_get)
        _get_sess = _linamp_attr("_get_shared_session", _get_shared_session)
        try:
            response = _http_get(
                search_url,
                session=_get_sess(),
                timeout=self.timeout,
                stream=True,
            )
        except (
            getattr(requests, "RequestException", OSError),
            OSError,
            ValueError,
            TypeError,
            AttributeError,
        ) as exc:
            logging.getLogger(__name__).debug("Cover search HTTP failed: %s", exc)
            return None
        try:
            if response is not None and response.status_code == 200:
                _read_body = _linamp_attr("_read_capped_body", _read_capped_body)
                raw = _read_body(response, _MAX_SEARCH_PAYLOAD)
                if raw is None:
                    return None
                pattern = (
                    r'https://lastfm\.freetls\.fastly\.net/i/u/[^"]+\.(?:jpg|jpeg|png)'
                )
                matches = re.findall(pattern, raw.decode("utf-8", errors="replace"))
                if matches:
                    valid_matches = [
                        url
                        for url in matches
                        if self._is_likely_album_cover(url) and _validate_dns(url)
                    ]
                    if valid_matches:
                        result = max(valid_matches, key=len)
                        return str(result) if result else None
        finally:
            if response is not None:
                response.close()
        return None

    def _is_likely_album_cover(self, url: str) -> bool:
        exclude_patterns = ["icon", "logo", "avatar", "thumb", "small", "tiny"]
        url_lower = url.lower()
        for pattern in exclude_patterns:
            if pattern in url_lower:
                return False
        return True

    def download_cover(self, url: str) -> bytes | None:
        _validate_dns = _linamp_attr("_validate_url_dns", _validate_url_dns)
        if not _validate_dns(url):
            return None
        _http_get = _linamp_attr("_safe_http_get", _safe_http_get)
        _get_sess = _linamp_attr("_get_shared_session", _get_shared_session)
        try:
            response = _http_get(
                url, session=_get_sess(), timeout=self.timeout, stream=True
            )
        except (
            getattr(requests, "RequestException", OSError),
            OSError,
            ValueError,
            TypeError,
            AttributeError,
        ) as exc:
            logging.getLogger(__name__).debug("Cover download HTTP failed: %s", exc)
            return None
        try:
            if response is None or response.status_code != 200:
                return None
            _read_body = _linamp_attr("_read_capped_body", _read_capped_body)
            content = _read_body(response, MAX_IMAGE_CONTENT_SIZE)
            if content is None:
                return None
            try:
                img: Any = Image.open(BytesIO(content))
                if img.mode in ("RGBA", "LA", "P"):
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    if img.mode == "P":
                        img = img.convert("RGBA")
                    background.paste(
                        img, mask=img.split()[-1] if img.mode == "RGBA" else None
                    )
                    img = background
                img.thumbnail((500, 500), Image.Resampling.LANCZOS)
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(1.1)
                output = BytesIO()
                img.save(output, format="JPEG", quality=85, optimize=True)
                return output.getvalue()
            except (
                getattr(requests, "RequestException", OSError),
                OSError,
                getattr(Image, "UnidentifiedImageError", OSError),
                getattr(Image, "DecompressionBombError", OSError),
            ) as e:
                logging.getLogger(__name__).debug(
                    "Cover image processing failed for %s: %s", url, e
                )
                return None
        finally:
            if response is not None:
                response.close()


_downloader = None


def get_cover_downloader() -> CoverArtDownloader | None:
    global _downloader
    if _downloader is None and COVER_DOWNLOAD_AVAILABLE:
        _downloader = CoverArtDownloader()
    return _downloader


def download_cover_for_audio(audio_path: str) -> CoverArt | None:
    downloader = get_cover_downloader()
    if not downloader:
        return None
    try:
        artist, album = downloader.get_metadata(audio_path)
    except (MutagenError, OSError, ValueError) as exc:
        logging.getLogger(__name__).debug(
            "Cover metadata lookup failed for %s: %s", audio_path, exc, exc_info=True
        )
        return None
    if not artist or not album:
        return None
    try:
        cover_url = downloader.search_cover(artist, album)
    except (
        getattr(requests, "RequestException", OSError),
        OSError,
        ValueError,
        TypeError,
        AttributeError,
    ) as exc:
        logging.getLogger(__name__).debug("Cover search failed: %s", exc)
        cover_url = None
    if not cover_url:
        return None
    try:
        cover_data = downloader.download_cover(cover_url)
    except (
        getattr(requests, "RequestException", OSError),
        OSError,
        ValueError,
        TypeError,
        AttributeError,
    ) as exc:
        logging.getLogger(__name__).debug("Cover download failed: %s", exc)
        cover_data = None
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
        pass

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
        if REQUESTS_AVAILABLE and artist and title:
            try:
                text, is_synced = self._get_lrclib_lyrics(
                    artist, title, album, duration
                )
                if text:
                    return text, "lrclib", is_synced
            except (requests.RequestException, OSError, ValueError) as exc:
                logging.getLogger(__name__).debug(
                    "LRCLIB lyrics fetch failed: %s", exc, exc_info=True
                )
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
        if mutagen is None:
            return "", False
        try:
            audio = File(audio_path)
            if audio is None or not getattr(audio, "tags", None):
                return "", False
            tags = audio.tags
            assert tags is not None
            lyrics_text = ""
            is_synced = False
            if isinstance(audio, MP3):
                for frame_id in ("USLT", "SYLT"):
                    frame = tags.get(frame_id)
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
                        for tag in tags.values():
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
        except (MutagenError, AttributeError, UnicodeDecodeError, ValueError) as exc:
            logging.getLogger(__name__).debug("Embedded lyrics read failed: %s", exc)
            return "", False

    def _get_id3_text(self, frame: Any) -> str:
        if _extract_id3_text is not None:
            result = _extract_id3_text(frame)
            return str(result) if result is not None else ""
        text = getattr(frame, "text", None)
        if text:
            return str(text[0]) if text else ""
        return str(frame)

    def _get_lrclib_lyrics(
        self, artist: str, title: str, album: str, duration: int
    ) -> tuple[str, bool]:
        if not REQUESTS_AVAILABLE:
            return "", False
        params: dict[str, str | int] = {
            "track_name": title,
            "artist_name": artist,
        }
        if album:
            params["album_name"] = album
        if duration and 1 <= duration <= 3600:
            params["duration"] = duration
        _http_get = _linamp_attr("_safe_http_get", _safe_http_get)
        _get_sess = _linamp_attr("_get_shared_session", _get_shared_session)
        response = _http_get(
            f"{self.LRCLIB_BASE}/get",
            session=_get_sess(),
            timeout=15,
            params=params,
            stream=True,
        )
        if response is None:
            return "", False
        try:
            if response.status_code == 404:
                return "", False
            _read_body = _linamp_attr("_read_capped_body", _read_capped_body)
            raw = _read_body(response, _MAX_SEARCH_PAYLOAD)
            if raw is None:
                logging.getLogger(__name__).debug(
                    "LRCLIB response exceeded the payload cap"
                )
                return "", False
            try:
                data: Any = json.loads(raw)
            except (ValueError, TypeError) as exc:
                logging.getLogger(__name__).debug("LRCLIB lyrics fetch failed: %s", exc)
                return "", False
            if not isinstance(data, dict):
                return "", False
            synced = data.get("syncedLyrics") or ""
            plain = data.get("plainLyrics") or ""
            text = synced or plain
            is_synced = bool(synced)
            return str(text) if text else "", is_synced
        finally:
            response.close()

    def _cache_key(self, audio_path: str, artist: str, title: str) -> str:
        base = f"{audio_path}:{artist}:{title}"
        return hashlib.md5(base.encode()).hexdigest()[:16]

    def _cache_set(self, key: str, data: dict) -> None:
        _cache_dir = _linamp_attr("LYRICS_CACHE_DIR", LYRICS_CACHE_DIR)
        try:
            _cache_dir.mkdir(parents=True, exist_ok=True)
            path = _cache_dir / f"{key}.json"
            _atomic_write_json(path, data)
            _maybe_prune_cache(_cache_dir)
        except OSError:
            pass


__all__ = [
    "COVER_ART_PATTERNS",
    "_COVER_DISPLAY_MAX",
    "CoverArt",
    "CoverArtDownloader",
    "LyricsManager",
    "_downloader",
    "_extract_cover_art_from_file",
    "_extract_cover_art_from_file_impl",
    "_extract_cover_art_impl",
    "_find_cover_in_folder",
    "_resize_cover_for_display",
    "_tag_key_matches",
    "download_cover_for_audio",
    "extract_cover_art",
    "get_cover_downloader",
]
