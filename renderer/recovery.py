from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import gi
from gi.repository import GLib  # type: ignore[reportMissingModuleSource]

from resources import (
    PresetLoadResult,
    RecoveryOperation,
    RecoveryState,
)

gi.require_version("Gdk", "4.0")
gi.require_version("Gtk", "4.0")
gi.require_version("Gst", "1.0")

if TYPE_CHECKING:
    from .core import MilkdropRenderer

logger = logging.getLogger("milkdrop_renderer")


class RecoveryManager:
    """Centralised owner of deferred-recovery state and retry scheduling.

    A single ``_current_operation`` describes what is *running*.  Two
    booleans (``pending_gl_reinit`` / ``pending_shader_recovery``) describe
    what is *waiting*.  Retry state (``retry_count``, ``timer_id``) is
    scoped to the current operation, so a retry is a continuation rather
    than a new request.

    GL reinitialization always preempts shader recovery.
    """

    _RETRY_DELAYS: tuple[int, ...] = (100, 250, 1000, 2500)
    MAX_GL_REINIT_ATTEMPTS: int = 4
    MAX_SHADER_RECOVERY_ATTEMPTS: int = 4

    def __init__(self, renderer: MilkdropRenderer) -> None:
        self._r = renderer
        self._state = RecoveryState.IDLE
        self._current_operation = RecoveryOperation.NONE
        self.pending_gl_reinit = False
        self.pending_shader_recovery = False
        self.retry_count = 0
        self.timer_id = 0

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                          #
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Reset all recovery state to IDLE."""
        self._cancel_timeout()
        self._state = RecoveryState.IDLE
        self._current_operation = RecoveryOperation.NONE
        self.pending_gl_reinit = False
        self.pending_shader_recovery = False
        self.retry_count = 0
        self.timer_id = 0

    def _cancel_timeout(self) -> None:
        if self.timer_id:
            GLib.source_remove(self.timer_id)
            self.timer_id = 0

    # ------------------------------------------------------------------ #
    #  Properties / queries                                               #
    # ------------------------------------------------------------------ #

    @property
    def state(self) -> RecoveryState:
        return self._state

    @property
    def in_progress(self) -> bool:
        """True while a recovery operation (including retries) is active."""
        return self._state in (
            RecoveryState.REINIT_RUNNING,
            RecoveryState.SHADER_RUNNING,
        )

    # ------------------------------------------------------------------ #
    #  Request entry-points (called from render pass)                     #
    # ------------------------------------------------------------------ #

    def request_gl_reinit(self) -> None:
        """High-priority: request deferred GL context reinitialization."""
        self.pending_gl_reinit = True
        self.schedule_next()

    def request_shader_recovery(self) -> None:
        """Lower-priority: request deferred shader recovery."""
        self.pending_shader_recovery = True
        self.schedule_next()

    # ------------------------------------------------------------------ #
    #  Dispatch                                                           #
    # ------------------------------------------------------------------ #

    def schedule_next(self) -> None:
        """Schedule the highest-priority pending recovery.

        GL reinitialization preempts shader recovery.  Only one operation
        runs at a time.  After a recovery finishes (success or exhaustion)
        this method is called again to drain any remaining pending work.
        """
        if self.in_progress:
            return

        if self._state == RecoveryState.FAILED:
            if not self.pending_gl_reinit and not self.pending_shader_recovery:
                return
            self._state = RecoveryState.IDLE

        if self.pending_gl_reinit:
            self.pending_gl_reinit = False
            self._cancel_timeout()
            self.retry_count = 0
            self._current_operation = RecoveryOperation.GL_REINIT
            self._state = RecoveryState.REINIT_RUNNING
            self.timer_id = GLib.timeout_add(0, self._run_gl_reinitialization)
        elif self.pending_shader_recovery:
            self.pending_shader_recovery = False
            self._cancel_timeout()
            self.retry_count = 0
            self._current_operation = RecoveryOperation.SHADER_RECOVERY
            self._state = RecoveryState.SHADER_RUNNING
            self.timer_id = GLib.timeout_add(0, self._run_shader_recovery)

    # ------------------------------------------------------------------ #
    #  Shared attempt completion                                          #
    # ------------------------------------------------------------------ #

    def _finish_attempt(
        self,
        success: bool,
        max_attempts: int,
        on_exhausted: Callable[[], None] | None = None,
    ) -> bool:
        """Common success / retry / exhaustion logic.

        Always returns ``False`` so the GLib source does not auto-repeat.
        """
        if success:
            self.retry_count = 0
            self._state = RecoveryState.IDLE
            self._current_operation = RecoveryOperation.NONE
            self.timer_id = 0
            self.schedule_next()
            return False

        self.retry_count += 1
        if self.retry_count >= max_attempts:
            logger.error(
                "%s exhausted after %d attempts",
                self._current_operation.name,
                self.retry_count,
            )
            self.retry_count = 0
            self._state = RecoveryState.FAILED
            self._current_operation = RecoveryOperation.NONE
            self.timer_id = 0
            if on_exhausted is not None:
                on_exhausted()
            self.schedule_next()
            return False

        delay_ms = self._RETRY_DELAYS[
            min(self.retry_count - 1, len(self._RETRY_DELAYS) - 1)
        ]
        logger.warning(
            "%s deferred, retrying in %d ms (attempt %d/%d)",
            self._current_operation.name,
            delay_ms,
            self.retry_count,
            max_attempts,
        )
        # Stay in the current RUNNING state — this is a continuation.
        self._cancel_timeout()
        callback = (
            self._run_gl_reinitialization
            if self._current_operation is RecoveryOperation.GL_REINIT
            else self._run_shader_recovery
        )
        self.timer_id = GLib.timeout_add(delay_ms, callback)
        return False

    # ------------------------------------------------------------------ #
    #  GL reinitialization                                                #
    # ------------------------------------------------------------------ #

    def _run_gl_reinitialization(self) -> bool:
        """Idle-loop callback for GL context reinitialization with bounded retry."""
        r = self._r
        logger.info(
            "Executing deferred GL reinitialization "
            "(attempt %d/%d)",
            self.retry_count + 1,
            self.MAX_GL_REINIT_ATTEMPTS,
        )
        success = False
        try:
            with r._current_gl_context():
                success = r._reinitialize_gl_resources()
        except Exception as e:  # noqa: BLE001
            logger.error("Deferred GL reinitialization failed: %s", e)

        def _on_exhausted() -> None:
            r.opengl_ready = False
            r._realize_failed = True

        return self._finish_attempt(
            success,
            self.MAX_GL_REINIT_ATTEMPTS,
            on_exhausted=_on_exhausted,
        )

    # ------------------------------------------------------------------ #
    #  Shader recovery                                                    #
    # ------------------------------------------------------------------ #

    def _run_shader_recovery(self) -> bool:
        """Idle-loop callback for shader recovery with bounded retry."""
        r = self._r
        logger.info(
            "Executing deferred shader recovery "
            "(attempt %d/%d)",
            self.retry_count + 1,
            self.MAX_SHADER_RECOVERY_ATTEMPTS,
        )
        success = False
        try:
            with r._current_gl_context():
                current_preset = getattr(r, "current_milk_file", None)
                if not current_preset or not Path(str(current_preset)).exists():
                    raise RuntimeError(
                        "Shader recovery has no valid current MilkDrop preset"
                    )

                result = r.load_milk_preset(
                    str(current_preset), smooth_transition=False
                )

                if result is PresetLoadResult.INSTALLED:
                    logger.debug(
                        "Shader recovery reload succeeded for %s", current_preset
                    )
                    success = True
                elif result is PresetLoadResult.PENDING:
                    # Cannot wait for async compilation during recovery.
                    raise RuntimeError(
                        f"Shader recovery pending async compilation for {current_preset}"
                    )
                else:
                    # FAILED or STALE
                    raise RuntimeError(
                        f"Shader recovery reload returned {result.name} for {current_preset}"
                    )
        except Exception as e:  # noqa: BLE001
            logger.error("Deferred shader recovery failed: %s", e)

        return self._finish_attempt(
            success,
            self.MAX_SHADER_RECOVERY_ATTEMPTS,
            # On exhaustion we deliberately keep the current preset.
        )
