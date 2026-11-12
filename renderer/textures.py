from __future__ import annotations

import logging
import threading
from collections import OrderedDict, deque
from queue import Empty
from typing import Any, TypedDict

import numpy as np
from OpenGL import GL
from PIL import Image

from gl_state import (
    GLResourceOwner,
)

logger = logging.getLogger("milkdrop_renderer")


class DecodedTexture(TypedDict):
    path: str
    image: Image.Image | Any


# --- Helpers -----------------------------------------------------
def _scan_balanced(expr: str, start: int, direction: int) -> int:
    """Scan ``expr`` from ``start`` in ``direction`` (1=forward, -1=backward)
    tracking balanced parentheses/brackets/braces.

    Returns the index of the first character outside the balanced region,
    or -1 if the structure is unbalanced.
    """
    depth = 0
    i = start
    n = len(expr)
    forward_open = "([{"
    forward_close = ")]}"
    # When scanning backward the roles of opening and closing delimiters
    # are reversed: a ')' is the first delimiter we encounter (it acts as
    # an "opening" in the backward direction) and '(' is the matching
    # closer.
    if direction == 1:
        opening, closing = forward_open, forward_close
    else:
        opening, closing = forward_close, forward_open
    # The start position must sit on the expected opening delimiter for
    # the given direction; otherwise there is no balanced region to scan.
    if not (0 <= i < n) or expr[i] not in opening:
        return -1
    while 0 <= i < n:
        ch = expr[i]
        if ch in opening:
            depth += 1
        elif ch in closing:
            depth -= 1
            if depth == 0:
                # Balanced region closed; return the position of the
                # matching delimiter so the caller can skip past it.
                return i
            if depth < 0:
                # Unmatched closer encountered.
                return -1
        i += direction
    return -1


# --- Texture cache -----------------------------------------------
class TextureLRU(OrderedDict[str, int]):
    def __init__(self, maxsize: int = 512, owner: GLResourceOwner | None = None):
        self.maxsize = maxsize
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self._owner = owner
        self._refcount: dict[int, int] = {}
        self._id_to_keys: dict[int, set[str]] = {}
        super().__init__()

    def acquire(self, tex_id: int) -> None:
        if tex_id and tex_id > 0:
            self._refcount[tex_id] = self._refcount.get(tex_id, 0) + 1

    def _retire_texture_id(self, tex_id: int) -> None:
        """Queue *tex_id* for deferred deletion if no cache entry still owns it.

        A single texture ID may be shared by multiple cache entries (e.g.
        two sampler keys pointing at the same underlying texture).  Deleting
        an ID that is still mapped by another key would silently invalidate
        that entry's texture, producing black textures.  This guard ensures
        deletion is only queued when the ID is no longer referenced by *any*
        cache entry.
        """
        if not tex_id or tex_id <= 0 or self._owner is None:
            return
        # Use reverse map for O(1) ownership check
        if self._id_to_keys.get(tex_id):
            return
        # Refuse to delete while an external acquire() is still outstanding.
        if self._refcount.get(tex_id, 0) > 0:
            return
        try:
            self._owner.queue_delete("texture", int(tex_id))
        except (GL.GLError, TypeError, ValueError, AttributeError):
            logger.debug("Failed to queue texture deletion for %d", tex_id)

    def release(self, tex_id: int) -> None:
        if not tex_id or tex_id <= 0:
            return
        count = self._refcount.get(tex_id, 0)
        if count <= 1:
            self._refcount.pop(tex_id, None)
            self._retire_texture_id(tex_id)
        else:
            self._refcount[tex_id] = count - 1

    def __delitem__(self, key: str) -> None:
        """Delete *key* and retire its texture ID if no other key owns it."""
        if key not in self:
            raise KeyError(key)
        old = self[key]
        super().__delitem__(key)
        # Update reverse map
        if old and old > 0:
            keys_set = self._id_to_keys.get(old)
            if keys_set is not None:
                keys_set.discard(key)
                if not keys_set:
                    self._id_to_keys.pop(old, None)
            self._retire_texture_id(old)

    def __setitem__(self, key, value):
        # Track the previous value so it can be released on replacement.
        old_tex_id = super().get(key, 0)
        if key in self:
            self.move_to_end(key)
            # Update reverse map if value changes
            if old_tex_id and old_tex_id > 0 and old_tex_id != value:
                keys_set = self._id_to_keys.get(old_tex_id)
                if keys_set is not None:
                    keys_set.discard(key)
                    if not keys_set:
                        self._id_to_keys.pop(old_tex_id, None)
        super().__setitem__(key, value)
        # Update reverse map for new value
        if value and value > 0:
            if value not in self._id_to_keys:
                self._id_to_keys[value] = set()
            self._id_to_keys[value].add(key)
        # Release the previous texture for this key, but only if it is
        # different from the new value and no other cache entry still
        # references it (checked inside _retire_texture_id).
        if old_tex_id and old_tex_id > 0 and old_tex_id != value:
            self._retire_texture_id(old_tex_id)
        # Evict to stay within capacity, skipping referenced entries.
        while len(self) > self.maxsize:
            # Search for an unreferenced eviction candidate.  If every cached
            # texture is currently referenced, defer eviction and allow
            # temporary cache growth rather than looping forever.
            candidate_key = None
            for k, v in self.items():
                if not (v and v > 0 and self._refcount.get(v, 0) > 0):
                    candidate_key = k
                    break
            if candidate_key is None:
                logger.warning(
                    "TextureLRU at capacity (%d entries); all entries "
                    "referenced, deferring eviction",
                    len(self),
                )
                break
            del self[candidate_key]
            self.evictions += 1

    def get(self, key: str, default: Any = None, /) -> Any:
        if key in self:
            self.hits += 1
            self.move_to_end(key)
            return self[key]
        else:
            self.misses += 1
            return default

    def stats(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "size": len(self),
            "maxsize": self.maxsize,
        }

    def clear(self) -> None:
        """Clear the cache and retire all textures that are no longer referenced."""
        for key in list(self.keys()):
            old = super().__getitem__(key) if key in self else 0
            if old and old > 0:
                keys_set = self._id_to_keys.get(old)
                if keys_set is not None:
                    keys_set.discard(key)
                    if not keys_set:
                        self._id_to_keys.pop(old, None)
            self._retire_texture_id(old)
        super().clear()
        self._id_to_keys.clear()


# --- Asynchronous texture loading ------------------------------
class DecodeResultQueue(deque[DecodedTexture]):
    """Thread-safe handoff of decoded textures from worker to GL thread.

    ``deque`` gives the atomic ``append``/``popleft`` the producer and the
    single consumer need, while staying directly inspectable (``len``,
    ``result[0]``, truthiness) for the drain loop.  ``get``/``put`` are
    provided for ``queue.Queue``-style call sites and use a
    ``threading.Condition`` so the consumer waits efficiently instead of
    polling with ``time.sleep``.
    """

    def __init__(self) -> None:
        super().__init__()
        self._cond: threading.Condition = threading.Condition()

    def put(
        self,
        item: DecodedTexture,
        block: bool = True,
        timeout: float | None = None,
    ) -> None:
        """Append a decoded result, notifying any waiting consumer."""
        with self._cond:
            self.append(item)
            self._cond.notify()

    def get(self, timeout: float | None = None) -> DecodedTexture:
        """Pop the oldest result, waiting at most *timeout* seconds.

        Raises ``queue.Empty`` when nothing arrives in time.
        """
        with self._cond:
            if not len(self):
                self._cond.wait(timeout=timeout)
            if len(self) > 0:
                result = self.popleft()
                return result
            raise Empty

    def popleft_nowait(self) -> DecodedTexture | None:
        """Pop the oldest result, or return ``None`` when the queue is empty.

        The drain loop runs on every rendered frame and the queue is empty
        most of the time, so the empty case must not be signalled by an
        exception: raising ``IndexError`` from ``deque.popleft`` would fire
        on nearly every frame and interrupt debuggers configured to break
        on raised exceptions.
        """
        with self._cond:
            if not len(self):
                return None
            result = self.popleft()
            return result


class AsyncTextureLoadMixin:
    """Background decoding for GL texture uploads.

    Decoding is CPU-bound (Pillow) while ``glTexImage2D`` needs the GL
    context, so the two are split across threads: a worker decodes the file
    into a PIL image and hands it to the render thread through
    ``_texture_load_result_queue``; the render thread — the only thread
    allowed to touch the GL context — performs the upload.

    Without this split, a preset that references several multi-megabyte
    textures stalls the render loop for the whole decode of every one of
    them, which shows up as multi-second freezes when a preset is loaded.
    """

    _texture_load_thread: threading.Thread | None
    _texture_load_queue: deque[str]
    _texture_load_inflight: set[str]
    #: Paths whose worker has decoded an image and published it to the
    #: result queue, but which the render thread has not yet drained or
    #: uploaded.  Entries are removed when the result is drained so that a
    #: failed upload does not permanently block re-decoding.
    _texture_load_done: set[str]
    #: Paths that have been successfully uploaded to GL.  Used for dedup
    #: so that already-uploaded textures are not re-decoded on every frame.
    _texture_load_uploaded: set[str]
    _texture_load_failed: set[str]
    _texture_load_result_queue: DecodeResultQueue
    _texture_load_stop: threading.Event
    #: Monotonic counter incremented on every worker (re)start.  The worker
    #: captures the value at start and exits when it no longer matches,
    #: preventing overlapping workers after a stop+restart cycle when the
    #: previous worker has not yet observed the stop event.
    _texture_load_generation: int
    #: Paths whose GL upload is already queued or in flight.  Shared by every
    #: owner of this mixin so a texture is never queued twice.
    _pending_texture_uploads: dict[str, int]
    #: Pixel dimensions of every uploaded texture, keyed by texture path.
    loaded_texture_sizes: dict[str, tuple[int, int]]
    #: Texture cache mapping texture paths to GL texture IDs.
    #: Initialized in the owning class's ``__init__`` (e.g. ``MilkdropRenderer``).
    loaded_textures: Any
    #: GL resource owner responsible for deferred texture deletion.
    #: Initialized in the owning class's ``__init__`` (e.g. ``MilkdropRenderer``).
    _gl_resource_owner: Any

    def _init_async_texture_loader(self) -> None:
        """Create the loader state.  Call from the owner's ``__init__``."""
        self._texture_load_lock = threading.RLock()
        self._texture_load_cond = threading.Condition(self._texture_load_lock)
        self._texture_load_queue = deque()
        self._texture_load_pending: set[str] = set()
        self._texture_load_inflight = set()
        self._texture_load_done = set()
        self._texture_load_uploaded = set()
        self._texture_load_failed = set()
        self._texture_load_result_queue = DecodeResultQueue()
        self._texture_load_thread = None
        self._texture_load_stop = threading.Event()
        self._texture_load_generation = 0
        self._pending_texture_uploads = {}
        self.loaded_texture_sizes = {}

    def _start_texture_load_thread(self) -> None:
        """Start the decode worker if it is not already running."""
        with self._texture_load_lock:
            thread = self._texture_load_thread
            if thread is not None and thread.is_alive():
                return
            # Ensure any previous (now-dead) thread has been fully joined
            # before starting a new one.  Without this, a stale thread
            # reference that is_alive() == False could be replaced while the
            # OS has not yet cleaned it up, leading to overlapping workers.
            if thread is not None and not thread.is_alive():
                thread.join(timeout=2.0)
            self._texture_load_stop.clear()
            self._texture_load_generation += 1
            thread = threading.Thread(
                target=self._texture_load_worker,
                name="texture-load",
                daemon=True,
            )
            self._texture_load_thread = thread
        thread.start()

    def _stop_texture_load_thread(self) -> None:
        """Signal the decode worker to exit and wait briefly for it.

        Increments the generation counter so that if the worker does not
        observe the stop event within the join timeout, a subsequent
        ``_start_texture_load_thread`` call will not create an overlapping
        worker — the old worker checks the generation on its next loop
        iteration and exits.  The thread reference is cleared under the
        lock only after the join attempt, so ``_start`` cannot race ahead
        of a still-running worker.
        """
        with self._texture_load_lock:
            self._texture_load_generation += 1  # Invalidate current worker
            self._texture_load_stop.set()
            try:
                self._texture_load_cond.notify_all()
            except Exception:  # noqa: BLE001,S110
                pass
            thread = self._texture_load_thread
            self._texture_load_thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

    def _reset_async_texture_loader(self) -> None:
        """Reset loader state for GL context loss / full re-init."""
        self._stop_texture_load_thread()
        with self._texture_load_lock:
            self._texture_load_queue.clear()
            self._texture_load_pending.clear()
            self._texture_load_inflight.clear()
            self._texture_load_done.clear()
            self._texture_load_uploaded.clear()
            self._texture_load_failed.clear()
            # Drain any already-decoded images that are now invalid
            try:
                while True:
                    res = self._texture_load_result_queue.popleft_nowait()
                    if res is None:
                        break
            except Exception:  # noqa: BLE001,S110
                pass
            self._pending_texture_uploads = {}
            self.loaded_texture_sizes = {}
        # Restart fresh
        self._texture_load_stop.clear()
        with self._texture_load_lock:
            self._texture_load_generation += 1
            self._texture_load_thread = None

    def _queue_texture_load(self, path: str) -> bool:
        """Request an async decode of *path*.

        Returns True when the request was newly queued, False when the same
        path is already queued, being decoded, decoded but awaiting upload,
        already uploaded, or has permanently failed.
        """
        with self._texture_load_lock:
            if (
                path in self._texture_load_pending
                or path in self._texture_load_inflight
                or path in self._texture_load_done
                or path in self._texture_load_uploaded
            ):
                return False
            if path in self._texture_load_failed:
                return False
            self._texture_load_queue.append(path)
            self._texture_load_pending.add(path)
            try:
                self._texture_load_cond.notify()
            except Exception:  # noqa: BLE001,S110
                pass
        self._start_texture_load_thread()
        return True

    def _texture_load_worker(self) -> None:
        """Decode queued image files and publish them for the GL thread."""
        gen = self._texture_load_generation
        stop = self._texture_load_stop
        while not stop.is_set() and gen == self._texture_load_generation:
            path = None
            with self._texture_load_lock:
                if self._texture_load_queue:
                    path = self._texture_load_queue.popleft()
                    self._texture_load_pending.discard(path)
                if path is not None and path in self._texture_load_inflight:
                    # robustness guard
                    path = None
                if path is not None:
                    self._texture_load_inflight.add(path)
            if path is None:
                with self._texture_load_lock:
                    if stop.is_set() or gen != self._texture_load_generation:
                        break
                    self._texture_load_cond.wait(timeout=0.02)
                gen = self._texture_load_generation
                continue
            image = None
            try:
                with Image.open(path) as source:
                    image = source.convert("RGBA")
            except Exception as exc:  # noqa: BLE001
                logger.debug("Async texture decode failed for %s: %s", path, exc)
            with self._texture_load_lock:
                self._texture_load_inflight.discard(path)
            if image is not None:
                with self._texture_load_lock:
                    self._texture_load_done.add(path)
                self._texture_load_result_queue.put(
                    {"path": path, "image": image}
                )
            else:
                with self._texture_load_lock:
                    self._texture_load_failed.add(path)

    def _drain_texture_load_results(self) -> list[DecodedTexture]:
        """Pop every decoded image waiting for upload.  Render thread only."""
        results: list[DecodedTexture] = []
        while True:
            result = self._texture_load_result_queue.popleft_nowait()
            if result is None:
                break
            results.append(result)
        if results:
            with self._texture_load_lock:
                for r in results:
                    self._texture_load_done.discard(r["path"])
        return results

    def _mark_texture_uploaded(self, path: str) -> None:
        """Mark *path* as successfully uploaded to GL."""
        with self._texture_load_lock:
            self._texture_load_done.discard(path)
            self._texture_load_uploaded.add(path)

    def _clear_texture_failure(self, path: str) -> None:
        """Allow a previously-failed decode to be retried."""
        with self._texture_load_lock:
            self._texture_load_failed.discard(path)

    def _process_texture_load_queue(self) -> None:
        if not getattr(self, "opengl_ready", False):
            return
        results = self._drain_texture_load_results()
        if not results:
            return
        create_func = getattr(self, "_create_gl_texture_func", None)
        if create_func is None:
            create_func = getattr(self, "_create_gl_texture_default", None)
        if create_func is None:
            logger.debug("No _create_gl_texture_func; dropping decoded results")
            return
        for result in results:
            path = result["path"]
            image = result["image"]
            old_tex = self.loaded_textures.get(path)
            try:
                img_data = np.asarray(image)
                # Flip vertically for OpenGL (which uses bottom-left origin).
                # Use slicing instead of np.flipud to reduce function call overhead.
                img_data = img_data[::-1, :]
                tex_id = create_func(
                    img_data,
                    image.width,
                    image.height,
                    GL.GL_RGBA8,
                    GL.GL_RGBA,
                    texture_path=path,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Texture upload error for %s: %s", path, exc)
                continue
            if old_tex is not None and old_tex != int(tex_id):
                try:
                    self._gl_resource_owner.queue_delete("texture", old_tex)
                except Exception:  # noqa: BLE001,S110
                    pass
            self.loaded_textures[path] = int(tex_id)
            self.loaded_texture_sizes[path] = (image.width, image.height)
            self._mark_texture_uploaded(path)
