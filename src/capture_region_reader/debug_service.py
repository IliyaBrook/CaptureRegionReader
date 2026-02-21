"""Debug service: per-subtitle artifact saving and pipeline logging.

When ``CRR_DEBUG=1`` is set, DebugService creates a session directory under
``.tests/debug/`` and records every stage of the OCR → TTS pipeline.

Each spoken subtitle gets its own numbered folder::

    .tests/debug/session_YYYYMMDD_HHMMSS/
        pipeline.log
        001/
            text.txt          # text sent to TTS
            raw.png           # original screenshot
            processed.png     # image after text-isolation (what OCR sees)
        002/ ...
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
from PIL import Image

_DEBUG_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", ".tests", "debug")


def is_debug_enabled() -> bool:
    return os.environ.get("CRR_DEBUG", "0") == "1"


class DebugService:
    def __init__(self) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_dir = os.path.join(_DEBUG_ROOT, f"session_{ts}")
        os.makedirs(self._session_dir, exist_ok=True)

        log_path = os.path.join(self._session_dir, "pipeline.log")
        self._log_file = open(log_path, "w", encoding="utf-8")  # noqa: SIM115
        self._subtitle_count = 0

        self._cached_raw: tuple[bytes, int, int] | None = None
        self._cached_processed: tuple[bytes, int, int] | None = None

        self.log("SESSION", f"started at {ts}")
        print(f"[DEBUG] Session dir: {self._session_dir}")

    # ------------------------------------------------------------------
    # Frame caching (Qt signal slots)
    # ------------------------------------------------------------------

    def cache_raw_frame(self, data: bytes, width: int, height: int) -> None:
        self._cached_raw = (data, width, height)

    def cache_processed_frame(self, data: bytes, width: int, height: int) -> None:
        self._cached_processed = (data, width, height)

    # ------------------------------------------------------------------
    # Pipeline logging
    # ------------------------------------------------------------------

    def log(self, tag: str, text: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self._log_file.write(f"{ts}  [{tag}]  {text}\n")
        self._log_file.flush()

    # ------------------------------------------------------------------
    # Subtitle artifact saving
    # ------------------------------------------------------------------

    def save_subtitle(self, spoken_text: str) -> None:
        self._subtitle_count += 1
        folder = os.path.join(self._session_dir, f"{self._subtitle_count:03d}")
        os.makedirs(folder, exist_ok=True)

        # Text
        with open(os.path.join(folder, "text.txt"), "w", encoding="utf-8") as f:
            f.write(spoken_text)

        # Raw screenshot
        if self._cached_raw:
            self._save_frame(self._cached_raw, os.path.join(folder, "raw.png"))

        # Processed screenshot
        if self._cached_processed:
            self._save_frame(self._cached_processed, os.path.join(folder, "processed.png"))

        self.log("SAVED", f"subtitle {self._subtitle_count:03d}")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        self.log("SESSION", "ended")
        self._log_file.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _save_frame(frame: tuple[bytes, int, int], path: str) -> None:
        data, width, height = frame
        arr = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
        Image.fromarray(arr).save(path)
