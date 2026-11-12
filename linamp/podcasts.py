from __future__ import annotations

import json
import logging
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse
from xml.etree import ElementTree as ET

from . import core
from .core import (
    _PODCAST_STATE_LOCK,
    MAX_EPISODES_PER_FEED,
    MAX_FEED_CONTENT_SIZE,
    REQUESTS_AVAILABLE,
    DefusedET,
    _atomic_write_json,
    _fsync_dir,
    _image_cache_path,
    _is_safe_feed_id,
    _linamp_attr,
    _safe_http_get,
    _validate_url_dns,
    fetch_image_to_cache,
)
from .models import PodcastEpisode, PodcastFeed


def _get_podcasts_config_dir():
    return _linamp_attr("PODCASTS_CONFIG_DIR", core.PODCASTS_CONFIG_DIR)


def _get_podcasts_file():
    return _linamp_attr("PODCASTS_FILE", core.PODCASTS_FILE)


def _get_podcast_episodes_dir():
    return _linamp_attr("PODCAST_EPISODES_DIR", core.PODCAST_EPISODES_DIR)


def _get_podcast_state_journal():
    return _linamp_attr("PODCAST_STATE_JOURNAL", core.PODCAST_STATE_JOURNAL)


def _get_podcast_art_cache_dir():
    return _linamp_attr("PODCAST_ART_CACHE_DIR", core.PODCAST_ART_CACHE_DIR)


def fetch_podcast_artwork(artwork_url: str, timeout: float = 8.0) -> Path | None:
    return fetch_image_to_cache(
        artwork_url, _get_podcast_art_cache_dir(), (300, 300), timeout
    )


def _podcast_art_cache_path(artwork_url: str) -> Path:
    return _image_cache_path(artwork_url, _get_podcast_art_cache_dir())


def parse_rss_feed(feed_url: str, content: str) -> PodcastFeed:
    """Parse RSS 2.0 feed content and return PodcastFeed."""
    if len(content) > MAX_FEED_CONTENT_SIZE:
        raise ValueError(f"Feed content from {feed_url} exceeds size limit")
    _stripped = content.lstrip("\ufeff").lstrip().lower()
    _stripped = re.sub(r"^(?:<!--.*?-->\s*)+", "", _stripped, flags=re.DOTALL)
    if _stripped.startswith(("<!doctype html", "<html", "<head", "<body")):
        suggested_url = ""
        for attr_pattern in (
            r'<link[^>]+rel=["\']alternate["\'][^>]+type=["\']application/(?:rss\+xml|atom\+xml)["\'][^>]+href=["\']([^"\']+)["\']',
            r'<link[^>]+type=["\']application/(?:rss\+xml|atom\+xml)["\'][^>]+rel=["\']alternate["\'][^>]+href=["\']([^"\']+)["\']',
            r'href=["\']([^"\']*(?:\.xml|/feed|/rss)[^"\']*)["\']',
        ):
            match = re.search(attr_pattern, content, re.IGNORECASE)
            if match:
                suggested_url = match.group(1)
                break
        if suggested_url:
            if suggested_url.startswith("/"):
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
        logger = logging.getLogger(__name__)
        logger.warning("Failed to parse RSS feed from %s: %s", feed_url, e)
        try:
            content = content.removeprefix("\ufeff")
            content = re.sub(
                r"[\uD800-\uDBFF\uDC00-\uDFFF\x00-\x08\x0B\x0C\x0E-\x1F\x7F]",
                "",
                content,
            )
            root = DefusedET.fromstring(content)
        except (ET.ParseError, re.error) as cleanup_e:
            logger.error("Failed to recover RSS feed parsing: %s", cleanup_e)
            raise ValueError(f"Invalid RSS feed content from {feed_url}: {e}") from e
    root_name = _local_xml_name(root.tag).lower()
    if root_name == "rss":
        channel = _find_element(root, "channel")
        if channel is None:
            raise ValueError("Invalid RSS feed: no channel element")
        title = _get_text(channel, "title") or ""
        author = (
            _get_text(channel, "itunes:author") or _get_text(channel, "author") or ""
        )
        description = _get_text(channel, "description") or ""
        artwork_url = (
            _get_attr(channel, "itunes:image", "href")
            or _get_text(channel, "itunes:image")
            or _get_text(channel, "image/url")
            or ""
        )
        homepage = _get_text(channel, "link") or ""
        episodes: list[PodcastEpisode] = []
        for item in _find_children(channel, "item"):
            if len(episodes) >= MAX_EPISODES_PER_FEED:
                break
            episode = _parse_rss_item(item, feed_url)
            if episode:
                episodes.append(episode)
    elif root_name == "feed":
        title = _get_text(root, "title") or ""
        author = _get_text(root, "author/name") or ""
        description = _get_text(root, "subtitle") or _get_text(root, "summary") or ""
        artwork_url = (
            _get_attr(root, "itunes:image", "href")
            or _get_text(root, "logo")
            or _get_text(root, "icon")
            or ""
        )
        homepage = _get_attr(root, "link[@rel='alternate']", "href")
        if not homepage:
            homepage = _get_attr(root, "link", "href")
        episodes = []
        for entry in _find_children(root, "entry"):
            if len(episodes) >= MAX_EPISODES_PER_FEED:
                break
            episode = _parse_atom_entry(entry, feed_url)
            if episode:
                episodes.append(episode)
    else:
        raise ValueError("Unknown feed format")
    feed_id = PodcastFeed.generate_feed_id(feed_url)
    last_updated = datetime.now(UTC).isoformat()
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


_PODCAST_XML_NAMESPACES = {
    "itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd",
    "atom": "http://www.w3.org/2005/Atom",
    "media": "http://search.yahoo.com/mrss/",
}


def _local_xml_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].rsplit(":", 1)[-1]


def _expanded_xml_tag(tag: str) -> str:
    if tag.startswith("{") or ":" not in tag:
        return tag
    prefix, local_name = tag.split(":", 1)
    namespace = _PODCAST_XML_NAMESPACES.get(prefix)
    if namespace is None:
        return tag
    return f"{{{namespace}}}{local_name}"


def _xml_element_matches(element: ET.Element, tag: str) -> bool:
    if element.tag == tag:
        return True
    expanded = _expanded_xml_tag(tag)
    if element.tag == expanded:
        return True
    if ":" not in tag and "{" not in tag:
        return _local_xml_name(element.tag) == tag
    return False


def _xml_predicate_matches(element: ET.Element, predicate: str) -> bool:
    match = re.fullmatch(r"@([\w:-]+)\s*=\s*(['\"])(.*?)\2", predicate.strip())
    if match is None:
        return False
    attribute, _quote, value = match.groups()
    return element.get(attribute) == value


def _find_element(element: ET.Element, path: str) -> ET.Element | None:
    candidates = [element]
    for part in (part.strip() for part in path.split("/") if part.strip()):
        if part == ".":
            continue
        if part.startswith(".//"):
            tag = part[3:]
            descendants = [
                descendant
                for candidate in candidates
                for descendant in candidate.iter()
                if descendant is not candidate and _xml_element_matches(descendant, tag)
            ]
            candidates = descendants
            continue
        tag = part
        predicate = ""
        if tag.startswith("[") and tag.endswith("]"):
            predicate = tag[1:-1]
        elif "[" in tag and tag.endswith("]"):
            tag, predicate = tag.split("[", 1)
            predicate = predicate[:-1]
        next_candidates = []
        for candidate in candidates:
            for child in candidate:
                if _xml_element_matches(child, tag) and (
                    not predicate or _xml_predicate_matches(child, predicate)
                ):
                    next_candidates.append(child)
        candidates = next_candidates
        if not candidates:
            return None
    return candidates[0]


def _find_children(element: ET.Element, tag: str) -> list[ET.Element]:
    return [child for child in element if _xml_element_matches(child, tag)]


def _get_text(element: ET.Element | None, tag: str) -> str:
    """Get text from a namespace-aware direct or compound element path."""
    if element is None:
        return ""
    child = _find_element(element, tag)
    if child is None or child.text is None:
        return ""
    return str(child.text).strip()


def _get_attr(element: ET.Element | None, tag: str, attribute: str) -> str:
    """Get an attribute from a namespace-aware direct or compound element path."""
    if element is None:
        return ""
    child = _find_element(element, tag)
    if child is None:
        return ""
    return str(child.get(attribute, "") or "").strip()


def _resolve_feed_url(feed_url: str, value: str) -> str:
    if not value:
        return ""
    if feed_url and urlparse(value).scheme not in ("http", "https"):
        return urljoin(feed_url, value)
    return value


def _parse_rss_item(item: ET.Element, feed_url: str = "") -> PodcastEpisode | None:
    """Parse an RSS item into a PodcastEpisode."""
    guid = _get_text(item, "guid") or ""
    title = _get_text(item, "title") or ""
    description = (
        _get_text(item, "description") or _get_text(item, "itunes:summary") or ""
    )
    pub_date = _get_text(item, "pubDate") or ""
    duration_str = _get_text(item, "itunes:duration") or ""
    duration = _parse_duration(duration_str)
    audio_url = ""
    enclosure = _find_element(item, "enclosure")
    if enclosure is not None:
        audio_url = enclosure.get("url", "")
    if not audio_url:
        media_content = _find_element(item, "media:content")
        if media_content is not None:
            audio_url = media_content.get("url", "")
    artwork_url = _get_attr(item, "itunes:image", "href") or _get_text(
        item, "itunes:image"
    )
    if not artwork_url:
        media_thumbnail = _find_element(item, "media:thumbnail")
        if media_thumbnail is not None:
            artwork_url = media_thumbnail.get("url", "")
    audio_url = _resolve_feed_url(feed_url, audio_url)
    artwork_url = _resolve_feed_url(feed_url, artwork_url)
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


_PODCAST_AUDIO_MIME_TYPES = {
    "audio/mpeg",
    "audio/mp4",
    "audio/x-m4a",
    "audio/ogg",
    "audio/vorbis",
    "audio/wav",
    "audio/x-wav",
    "audio/webm",
    "audio/aac",
    "audio/flac",
    "audio/x-flac",
    "audio/opus",
    "audio/3gpp",
    "audio/3gpp2",
}


def _looks_like_audio_link(link: ET.Element) -> bool:
    """Return True when *link* is plausibly an audio enclosure.

    Enclosures are identified by ``rel="enclosure"`` or by an audio MIME type
    (or a type that omits the MIME prefix, e.g. ``mp3``).  Many Atom feeds
    omit the type entirely, so a bare ``rel="enclosure"`` is also accepted.
    """
    rel = (link.get("rel") or "").strip().lower()
    if rel == "enclosure":
        return True
    type_attr = (link.get("type") or "").strip().lower()
    if not type_attr:
        return False
    if type_attr in _PODCAST_AUDIO_MIME_TYPES:
        return True
    if "/" not in type_attr:
        return type_attr in {
            "mp3",
            "mp4",
            "m4a",
            "ogg",
            "wav",
            "webm",
            "aac",
            "flac",
            "opus",
        }
    return False


def _parse_atom_entry(entry: ET.Element, feed_url: str = "") -> PodcastEpisode | None:
    """Parse an Atom entry into a PodcastEpisode."""
    guid = _get_text(entry, "id") or ""
    title = _get_text(entry, "title") or ""
    description = _get_text(entry, "summary") or _get_text(entry, "content") or ""
    pub_date = _get_text(entry, "published") or _get_text(entry, "updated") or ""
    audio_url = ""
    for link in _find_children(entry, "link"):
        if _looks_like_audio_link(link):
            audio_url = link.get("href", "")
            if audio_url:
                break
    duration = 0
    media_content = _find_element(entry, "media:content")
    if media_content is not None:
        duration_str = media_content.get("duration", "")
        duration = _parse_duration(duration_str)
    if not duration:
        duration = _parse_duration(_get_text(entry, "itunes:duration"))
    artwork_url = _get_attr(entry, "itunes:image", "href") or _get_text(
        entry, "itunes:image"
    )
    if not artwork_url:
        media_thumbnail = _find_element(entry, "media:thumbnail")
        if media_thumbnail is not None:
            artwork_url = media_thumbnail.get("url", "")
    audio_url = _resolve_feed_url(feed_url, audio_url)
    artwork_url = _resolve_feed_url(feed_url, artwork_url)
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
    """Parse itunes:duration to seconds. Accepts seconds or H:MM:SS / M:SS.
    Tolerant of non-canonical times (e.g. "1:90:30") by carrying overflow
    from minutes/seconds into the next larger unit instead of rejecting them.
    """
    if duration_str is None:
        return 0
    if isinstance(duration_str, (int, float)):
        return max(0, int(duration_str))
    text = str(duration_str).strip()
    if not text:
        return 0
    if text.isdigit():
        return int(text)
    parts = text.split(":")
    try:
        nums = [int(float(p)) for p in parts]
    except (ValueError, OverflowError, TypeError):
        return 0
    if len(nums) == 3:
        hours, minutes, seconds = nums
        total = hours * 3600 + minutes * 60 + seconds
        return max(0, total)
    if len(nums) == 2:
        minutes, seconds = nums
        total = minutes * 60 + seconds
        return max(0, total)
    if len(nums) == 1:
        return max(0, nums[0])
    return 0


def fetch_feed_content(feed_url: str, timeout: float = 15.0) -> str:
    """Fetch feed content from URL using requests only.

    Returns the decoded body on success and an empty string on any failure
    (DNS failure, HTTP error, size limit, network timeout).  The caller is
    responsible for distinguishing the failure modes; the empty return is
    intentional so callers can treat "no content" uniformly.
    """
    logger = logging.getLogger(__name__)
    if not _validate_url_dns(feed_url):
        logger.warning("Feed fetch blocked by DNS validation: %s", feed_url)
        return ""
    if not REQUESTS_AVAILABLE:
        logger.warning("Feed fetch unavailable: requests not installed")
        return ""
    try:
        resp = _safe_http_get(feed_url, timeout=timeout, stream=True)
        try:
            if resp is None:
                logger.warning("Feed fetch returned no response: %s", feed_url)
                return ""
            if resp.status_code != 200:
                logger.warning(
                    "Feed fetch HTTP %s for %s",
                    resp.status_code,
                    feed_url,
                )
                return ""
            content_type = (
                (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            )
            if content_type and content_type not in (
                "application/rss+xml",
                "application/atom+xml",
                "text/xml",
                "application/xml",
                "text/plain",
            ):
                logger.warning(
                    "Feed fetch from %s returned non-XML Content-Type %r",
                    feed_url,
                    content_type,
                )
            chunks = []
            total = 0
            for chunk in resp.iter_content(chunk_size=8192):
                total += len(chunk)
                if total > MAX_FEED_CONTENT_SIZE:
                    logger.warning("Feed fetch from %s exceeded size limit", feed_url)
                    return ""
                chunks.append(chunk)
            return b"".join(chunks).decode("utf-8", errors="replace")
        finally:
            if resp is not None:
                resp.close()
    except (OSError, ValueError) as exc:
        logger.warning("Feed fetch failed for %s: %s", feed_url, exc)
        return ""


def sync_feed(feed_url: str, existing_feed: PodcastFeed | None = None) -> PodcastFeed:
    """Sync a podcast feed, merging with existing episodes to preserve playback state."""
    content = fetch_feed_content(feed_url)
    if not content:
        raise ValueError(f"Failed to fetch feed from {feed_url}")
    new_feed = parse_rss_feed(feed_url, content)
    if existing_feed and existing_feed.episodes:
        existing_episodes_by_guid = {ep.guid: ep for ep in existing_feed.episodes}
        for new_episode in new_feed.episodes:
            existing_episode = existing_episodes_by_guid.get(new_episode.guid)
            if existing_episode:
                new_episode.played = existing_episode.played
                new_episode.position = existing_episode.position
                new_episode.downloaded_path = existing_episode.downloaded_path
    return new_feed


def load_podcasts() -> list[PodcastFeed]:
    """Load all podcast feeds from storage, recovering from torn commits."""
    if not _get_podcasts_file().exists():
        return []
    try:
        with open(_get_podcasts_file(), encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(data, list):
        return []
    feeds = []
    for feed_data in data:
        if not isinstance(feed_data, dict):
            continue
        try:
            feeds.append(PodcastFeed.from_dict(feed_data))
        except (TypeError, ValueError, AttributeError, OverflowError):
            continue
    recovery_attempted = False
    recovery_succeeded = False
    if _get_podcast_state_journal().exists():
        try:
            journal = json.loads(
                _get_podcast_state_journal().read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, OSError, ValueError):
            journal = {}
        if isinstance(journal, dict):
            j_feed_id = journal.get("feed_id", "")
            j_type = journal.get("type", "")
            if j_feed_id and _is_safe_feed_id(j_feed_id):
                recovery_attempted = True
                if j_type == "remove":
                    feeds = [f for f in feeds if getattr(f, "feed_id", "") != j_feed_id]
                    # Persist the index BEFORE touching the episode file: if
                    # the index write fails we keep the episode file and the
                    # journal so the next start retries the whole removal.
                    if not _save_podcasts_legacy(feeds):
                        recovery_succeeded = False
                    else:
                        try:
                            ep_file = _get_podcast_episodes_dir() / f"{j_feed_id}.json"
                            if ep_file.exists():
                                ep_file.unlink()
                        except OSError:
                            pass
                        recovery_succeeded = True
                else:
                    episodes = load_feed_episodes(j_feed_id)
                    attached = False
                    for feed in feeds:
                        if getattr(feed, "feed_id", "") == j_feed_id:
                            feed.episodes = list(episodes)
                            attached = True
                            break
                    if not attached and episodes:
                        # The feed was missing from the index (partial previous
                        # write).  Reconstruct it so the episodes are not
                        # silently lost; the reconstructed feed will lack
                        # metadata, which is logged below.
                        logging.getLogger(__name__).warning(
                            "Podcast recovery: feed %s missing from index; "
                            "reconstructing from authoritative episode file",
                            j_feed_id,
                        )
                        feeds.append(
                            PodcastFeed(feed_id=j_feed_id, episodes=list(episodes))
                        )
                    recovery_succeeded = _save_podcasts_legacy(feeds)
        if recovery_attempted and not recovery_succeeded:
            logging.getLogger(__name__).warning(
                "Podcast recovery journal retained: could not persist "
                "recovered state; will retry on next start"
            )
        elif _get_podcast_state_journal().exists():
            try:
                _get_podcast_state_journal().unlink()
            except OSError:
                pass
    return feeds


def save_podcasts(feeds: list[PodcastFeed]) -> bool:
    """Save all podcast feeds to storage."""
    try:
        return _atomic_write_json(_get_podcasts_file(), [feed.to_dict() for feed in feeds])
    except (OSError, TypeError, ValueError):
        return False


def load_feed_episodes(feed_id: str) -> list[PodcastEpisode]:
    """Load episodes for a specific feed from storage."""
    if not _is_safe_feed_id(feed_id):
        return []
    episodes_file = _get_podcast_episodes_dir() / f"{feed_id}.json"
    if not episodes_file.exists():
        return []
    try:
        with open(episodes_file, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return []
        episodes = []
        for ep_data in data:
            if not isinstance(ep_data, dict):
                continue
            try:
                episodes.append(PodcastEpisode.from_dict(ep_data))
            except (TypeError, ValueError, AttributeError, OverflowError):
                continue
        return episodes
    except (json.JSONDecodeError, OSError):
        pass
    return []


def save_feed_episodes(feed_id: str, episodes: list[PodcastEpisode]) -> bool:
    """Save episodes for a specific feed to storage."""
    if not _is_safe_feed_id(feed_id):
        return False
    return _atomic_write_json(
        _get_podcast_episodes_dir() / f"{feed_id}.json",
        [ep.to_dict() for ep in episodes],
    )


def delete_feed_episodes(feed_id: str) -> bool:
    """Delete episodes file for a specific feed."""
    if not _is_safe_feed_id(feed_id):
        return False
    episodes_file = _get_podcast_episodes_dir() / f"{feed_id}.json"
    try:
        if episodes_file.exists():
            episodes_file.unlink()
        return True
    except OSError:
        return False


def _save_podcasts_legacy(feeds: list[PodcastFeed]) -> bool:
    """Save feeds without journal (legacy path used by recovery)."""
    return save_podcasts(feeds)


def _save_feed_episodes_legacy(feed_id: str, episodes: list[PodcastEpisode]) -> bool:
    """Save episodes without journal (legacy path used by recovery tests)."""
    return save_feed_episodes(feed_id, episodes)


def _podcast_index_is_populated() -> bool:
    """Return True when ``podcasts.json`` on disk still holds subscriptions."""
    try:
        data = json.loads(_get_podcasts_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(data, list) and any(isinstance(f, dict) for f in data)


def _commit_journal(journal_path: Path, payload: dict | None = None) -> bool:
    """Atomically install the journal file with the given payload.
    Returns False (and leaves the journal absent) if the install fails —
    callers treating the missing journal as "no uncommitted work" will then
    skip recovery, which is safe because a failed journal install means the
    commit never started.
    """
    return _atomic_write_json(journal_path, payload or {})


def save_podcast_state(
    feeds: list[PodcastFeed],
    feed_id: str,
    episodes: list[PodcastEpisode],
    journal: Path | None = None,
) -> bool:
    """Unified transactional save for podcast state.

    Writes the journal, then the episode file, then the feed index, then
    retires the journal.  If the journal cannot be installed the commit is
    aborted (returns False) so callers never observe a partially-written state.

    The *journal* argument is accepted for API compatibility with callers that
    manage an external journal path, but the function always resolves the
    canonical journal path internally via ``_get_podcast_state_journal`` so
    that test overrides of ``linamp.PODCAST_STATE_JOURNAL`` are honoured.
    """
    if not _is_safe_feed_id(feed_id):
        return False
    _get_podcasts_config_dir().mkdir(parents=True, exist_ok=True)
    _get_podcast_episodes_dir().mkdir(parents=True, exist_ok=True)
    with _PODCAST_STATE_LOCK:
        return _save_podcast_state_locked(feeds, feed_id, episodes)


def _save_podcast_state_locked(
    feeds: list[PodcastFeed],
    feed_id: str,
    episodes: list[PodcastEpisode],
) -> bool:
    if not _linamp_attr("_commit_journal", _commit_journal)(
        _get_podcast_state_journal(), {"feed_id": feed_id}
    ):
        return False
    try:
        ep_data = [ep.to_dict() for ep in episodes]
        feed_data = [feed.to_dict() for feed in feeds]
    except (AttributeError, TypeError):
        return False
    if not feed_data and _podcast_index_is_populated():
        logging.getLogger(__name__).error(
            "Refusing to persist an empty podcast index for %s: on-disk index "
            "still holds subscriptions (suspected caller bug)",
            feed_id,
        )
        return False
    if not _atomic_write_json(_get_podcast_episodes_dir() / f"{feed_id}.json", ep_data):
        return False
    if not _atomic_write_json(_get_podcasts_file(), feed_data):
        return False
    try:
        _get_podcast_state_journal().unlink()
    except OSError:
        pass
    return True


def remove_podcast_feed(feed_id: str, sync_tokens: list | None = None) -> bool:
    """Transactionally remove a podcast feed.

    Installs a removal journal, rewrites the index without the feed, deletes
    the episode file, then retires the journal.  Recovery on the read path
    reconciles any torn state left by a crash during the commit.

    The *sync_tokens* argument is accepted for callers that track live sync
    state alongside the removal (e.g. the UI passes its token dict so the
    caller can invalidate in-flight workers); it is not consumed here because
    the journal + index rewrite is self-contained.
    """
    if not _is_safe_feed_id(feed_id):
        return False
    _get_podcasts_config_dir().mkdir(parents=True, exist_ok=True)
    with _PODCAST_STATE_LOCK:
        if not _linamp_attr("_commit_journal", _commit_journal)(
            _get_podcast_state_journal(),
            {"feed_id": feed_id, "type": "remove", "started": time.time()},
        ):
            return False
        try:
            with open(_get_podcasts_file(), encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            data = []
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            logging.getLogger(__name__).error(
                "Refusing to remove feed %s: podcast index unreadable", feed_id
            )
            return False
        if not isinstance(data, list):
            logging.getLogger(__name__).error(
                "Refusing to remove feed %s: podcast index is not a list", feed_id
            )
            return False
        remaining = [
            f for f in data if isinstance(f, dict) and f.get("feed_id", "") != feed_id
        ]
        if not _atomic_write_json(_get_podcasts_file(), remaining):
            return False
        try:
            episodes_file = _get_podcast_episodes_dir() / f"{feed_id}.json"
            if episodes_file.exists():
                episodes_file.unlink()
        except OSError:
            pass
        try:
            _get_podcast_state_journal().unlink()
        except OSError:
            pass
    return True


__all__ = [
    "_PODCAST_XML_NAMESPACES",
    "_commit_journal",
    "_expanded_xml_tag",
    "_find_children",
    "_find_element",
    "_fsync_dir",
    "_get_attr",
    "_get_text",
    "_local_xml_name",
    "_parse_atom_entry",
    "_parse_duration",
    "_parse_rss_item",
    "_podcast_art_cache_path",
    "_podcast_index_is_populated",
    "_resolve_feed_url",
    "_save_feed_episodes_legacy",
    "_save_podcast_state_locked",
    "_save_podcasts_legacy",
    "_xml_element_matches",
    "_xml_predicate_matches",
    "delete_feed_episodes",
    "fetch_feed_content",
    "fetch_podcast_artwork",
    "load_feed_episodes",
    "load_podcasts",
    "parse_rss_feed",
    "remove_podcast_feed",
    "save_feed_episodes",
    "save_podcast_state",
    "save_podcasts",
    "sync_feed",
]
