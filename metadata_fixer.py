#!/usr/bin/env python3
"""
Metadata Fixer Module

Automatically completes missing ID3/metadata tags by parsing filenames
and inferring album from parent folder structure.
"""

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    from mutagen import File
    from mutagen._util import MutagenError
    from mutagen.flac import FLAC
    from mutagen.id3 import ID3NoHeaderError, ID3UnsupportedVersionError
    from mutagen.mp3 import MP3, HeaderNotFoundError
    from mutagen.mp4 import MP4
    from mutagen.oggvorbis import OggVorbis

    MUTAGEN_AVAILABLE = True
    _MUTAGEN_EXCEPTIONS = (
        OSError,
        AttributeError,
        KeyError,
        ID3NoHeaderError,
        ID3UnsupportedVersionError,
        HeaderNotFoundError,
        MutagenError,
    )
except ImportError:
    MUTAGEN_AVAILABLE = False
    ID3NoHeaderError = Exception
    ID3UnsupportedVersionError = Exception
    HeaderNotFoundError = Exception
    MutagenError = Exception
    OggVorbis = None
    _MUTAGEN_EXCEPTIONS = (OSError, AttributeError, KeyError)
    logger.warning("mutagen library not available - limited functionality")


@dataclass
class FixResult:
    """Result of fixing metadata for a single file."""
    file_path: str
    success: bool
    message: str
    old_metadata: dict[str, str] | None = None
    new_metadata: dict[str, str] | None = None
    updated_fields: list[str] = field(default_factory=list)


def _extract_id3_text(frame: Any) -> str:
    text = getattr(frame, "text", None)
    if text:
        return str(text[0]) if text else ""
    return str(frame)


class MetadataFixer:
    """Automatically complete missing metadata tags from filenames and folder structure."""

    def __init__(self, **kwargs):
        self.metadata = kwargs.get("metadata", {})
        self.cleanup_options = kwargs.get("cleanup_options", {})
        if not MUTAGEN_AVAILABLE:
            logger.warning("mutagen not available - using basic filename parsing")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def fix_files_threaded(
        self, file_paths: list[str], max_workers: int = 2, **kwargs
    ) -> list[FixResult]:
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(self._fix_single_file, path): path
                for path in file_paths
            }
            for future in as_completed(future_to_path):
                try:
                    result = future.result()
                    results.append(result)
                except (OSError, RuntimeError, ValueError) as e:
                    path = future_to_path[future]
                    results.append(
                        FixResult(
                            file_path=path,
                            success=False,
                            message=f"Thread error: {e!s}",
                        )
                    )
        return results

    def _fix_single_file(self, file_path: str) -> FixResult:
        try:
            if not os.path.exists(file_path):
                return FixResult(
                    file_path=file_path,
                    success=False,
                    message="File not found",
                )

            old_metadata = self._read_tags(file_path)
            inferred = self._infer_tags(file_path, old_metadata)
            updated_fields = self._apply_missing_tags(file_path, old_metadata, inferred)
            new_metadata = self._read_tags(file_path) if updated_fields else old_metadata

            if updated_fields:
                return FixResult(
                    file_path=file_path,
                    success=True,
                    message=f"Updated {', '.join(updated_fields)}",
                    old_metadata=old_metadata,
                    new_metadata=new_metadata,
                    updated_fields=updated_fields,
                )
            return FixResult(
                file_path=file_path,
                success=True,
                message="All tags present, nothing to update",
                old_metadata=old_metadata,
                new_metadata=old_metadata,
                updated_fields=[],
            )

        except _MUTAGEN_EXCEPTIONS as e:
            return FixResult(
                file_path=file_path,
                success=False,
                message=f"Error: {e!s}",
            )

    def _infer_tags(self, file_path: str, existing: dict[str, str] | None) -> dict[str, str]:
        filename = Path(file_path).stem
        parent = Path(file_path).parent.name
        existing = existing or {}

        artist = existing.get("artist", "")
        title = existing.get("title", "")
        album = existing.get("album", "")
        tracknumber = existing.get("tracknumber", "")

        if not artist or not title:
            parsed = self._parse_filename(filename)
            if not artist:
                artist = parsed.get("artist", "")
            if not title:
                title = parsed.get("title", "")
            if not tracknumber:
                tracknumber = parsed.get("tracknumber", "")

        if not album:
            album = self._infer_album(parent, artist)

        tags: dict[str, str] = {}
        if artist:
            tags["artist"] = artist
        if title:
            tags["title"] = title
        if album:
            tags["album"] = album
        if tracknumber:
            tags["tracknumber"] = tracknumber
        return tags

    def _parse_filename(self, filename: str) -> dict[str, str]:
        result: dict[str, str] = {}

        patterns = [
            r"^(?P<track>\d+)\s*[-\.]\s*(?P<artist>.+?)\s*-\s*(?P<title>.+)$",
            r"^(?P<track>\d+)\s*[-\.]\s*(?P<title>.+)$",
            r"^(?P<artist>.+?)\s*-\s*(?P<title>.+)$",
            r"^(?P<artist>[^_]+)_(?P<title>.+)$",
        ]

        for pattern in patterns:
            match = re.match(pattern, filename)
            if match:
                gd = match.groupdict()
                if gd.get("track"):
                    result["tracknumber"] = gd["track"].zfill(2)
                if gd.get("artist"):
                    result["artist"] = gd["artist"].strip()
                if gd.get("title"):
                    result["title"] = gd["title"].strip()
                if result.get("artist") or result.get("title"):
                    break

        return result

    def _infer_album(self, parent_name: str, known_artist: str) -> str:
        if not parent_name or parent_name in (".", "/", ".."):
            return ""
        album = parent_name
        match = re.match(r"^(?P<artist>[^-]+?)\s*-\s*(?P<album>.+)$", parent_name)
        if match:
            gd = match.groupdict()
            if known_artist and gd.get("artist", "").strip().lower() == known_artist.lower():
                return gd.get("album", "").strip()
            if not known_artist:
                return album.strip()
        return album.strip()

    def _read_tags(self, file_path: str) -> dict[str, str]:
        if not MUTAGEN_AVAILABLE:
            return {}
        try:
            audio = File(file_path)
            if audio is None or not getattr(audio, "tags", None):
                return {}
            return self._extract_tag_dict(audio)
        except _MUTAGEN_EXCEPTIONS:
            return {}

    def _extract_tag_dict(self, audio: Any) -> dict[str, str]:
        tags: dict[str, str] = {}
        if isinstance(audio, MP3):
            id3 = audio.tags
            if id3:
                tags["artist"] = self._get_id3_text(id3, "TPE1")
                tags["title"] = self._get_id3_text(id3, "TIT2")
                tags["album"] = self._get_id3_text(id3, "TALB")
                tags["tracknumber"] = self._get_id3_text(id3, "TRCK")
        elif isinstance(audio, (FLAC, OggVorbis)):
            tags["artist"] = self._get_first(audio, "artist")
            tags["title"] = self._get_first(audio, "title")
            tags["album"] = self._get_first(audio, "album")
            tags["tracknumber"] = self._get_first(audio, "tracknumber")
        elif isinstance(audio, MP4):
            tags["artist"] = self._get_mp4_text(audio, "\xa9ART")
            tags["title"] = self._get_mp4_text(audio, "\xa9nam")
            tags["album"] = self._get_mp4_text(audio, "\xa9alb")
            tags["tracknumber"] = self._get_mp4_text(audio, "trkn")
        else:
            raw = getattr(audio, "tags", None)
            if raw:
                for key, value in raw.items():
                    key_lower = str(key).lower()
                    if key_lower in ("artist", "title", "album", "tracknumber"):
                        tags[key_lower] = self._coerce_value(value)
        return {k: v.strip() for k, v in tags.items() if v}

    def _get_id3_text(self, id3: Any, frame_id: str) -> str:
        frame = id3.get(frame_id)
        if frame is None:
            return ""
        return _extract_id3_text(frame)

    def _get_first(self, audio: Any, key: str) -> str:
        values = audio.get(key)
        if not values:
            return ""
        if isinstance(values, list):
            return str(values[0]) if values else ""
        return str(values)

    def _get_mp4_text(self, audio: MP4, key: str) -> str:
        values = audio.get(key)
        if not values:
            return ""
        if isinstance(values, list):
            return str(values[0]) if values else ""
        return str(values)

    def _coerce_value(self, value: Any) -> str:
        if hasattr(value, "text") and value.text:
            return str(value.text[0])
        if hasattr(value, "value"):
            return str(value.value)
        return str(value)

    def _apply_missing_tags(self, file_path: str, existing: dict[str, str], inferred: dict[str, str]) -> list[str]:
        if not MUTAGEN_AVAILABLE:
            return []
        updated_fields: list[str] = []
        try:
            audio = File(file_path)
            if audio is None:
                return []
            if not getattr(audio, "tags", None):
                try:
                    audio.add_tags()
                except Exception:
                    logger.debug("add_tags failed", exc_info=True)

            changes: dict[str, str] = {}
            for field in ("artist", "title", "album", "tracknumber"):
                new_value = inferred.get(field, "")
                old_value = existing.get(field, "")
                if new_value and new_value != old_value:
                    changes[field] = new_value

            if not changes:
                return []

            if isinstance(audio, MP3):
                self._set_id3_tags(audio.tags, changes)
            elif isinstance(audio, FLAC):
                self._set_flac_tags(audio, changes)
            elif isinstance(audio, OggVorbis):
                self._set_vorbis_tags(audio, changes)
            elif isinstance(audio, MP4):
                self._set_mp4_tags(audio, changes)
            else:
                for field, value in changes.items():
                    try:
                        audio.tags[field] = value
                    except (KeyError, TypeError):
                        pass

            audio.save()
            updated_fields = list(changes.keys())
        except _MUTAGEN_EXCEPTIONS as e:
            logger.debug("Failed to apply missing tags to %s: %s", file_path, e)
        return updated_fields

    def _set_id3_tags(self, id3: Any, changes: dict[str, str]) -> None:
        mapping = {
            "artist": ("TPE1",),
            "title": ("TIT2",),
            "album": ("TALB",),
            "tracknumber": ("TRCK",),
        }
        for tag_name, frame_ids in mapping.items():
            value = changes.get(tag_name)
            if not value:
                continue
            for frame_id in frame_ids:
                frame = id3.get(frame_id)
                if frame is None:
                    from mutagen.id3 import Frame

                    frame_cls = Frame.__dict__.get(frame_id) or getattr(
                        __import__("mutagen.id3", fromlist=[frame_id]), frame_id
                    )
                    frame = frame_cls(encoding=3, text=[value])
                    id3.add(frame)
                else:
                    frame.encoding = 3
                    frame.text = [value]
                break

    def _set_flac_tags(self, audio: FLAC, changes: dict[str, str]) -> None:
        for tag_name, value in changes.items():
            audio[tag_name] = [value]

    def _set_vorbis_tags(self, audio: OggVorbis, changes: dict[str, str]) -> None:
        for tag_name, value in changes.items():
            audio[tag_name] = [value]

    def _set_mp4_tags(self, audio: MP4, changes: dict[str, str]) -> None:
        mapping = {
            "artist": "\xa9ART",
            "title": "\xa9nam",
            "album": "\xa9alb",
            "tracknumber": "trkn",
        }
        for tag_name, key in mapping.items():
            value = changes.get(tag_name)
            if not value:
                continue
            if key == "trkn":
                try:
                    track_num = int(re.match(r"(\d+)", value).group(1))
                    audio[key] = [(track_num, 0)]
                except (ValueError, AttributeError):
                    audio[key] = [value]
            else:
                audio[key] = [value]

    def close(self) -> None:
        pass
