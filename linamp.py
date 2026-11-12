#!/usr/bin/env python3
"""LinAmp - GTK4/GStreamer Winamp-style music player."""

from __future__ import annotations

from linamp.ui.window import LinAmpApp, main, window_source

__all__ = ["LinAmpApp", "main", "window_source"]


if __name__ == "__main__":
    raise SystemExit(main())
