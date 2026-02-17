from __future__ import annotations

from difflib import SequenceMatcher


class TextDiffer:
    def __init__(self, similarity_threshold: float = 0.92):
        self._last_text: str = ""
        self._threshold = similarity_threshold

    def get_new_text(self, current_text: str) -> str | None:
        """Return text to speak, or None if no meaningful change detected."""
        if not current_text:
            return None

        if not self._last_text:
            self._last_text = current_text
            return current_text

        ratio = SequenceMatcher(None, self._last_text, current_text).ratio()

        if ratio >= self._threshold:
            return None

        # Check if text grew (scrolling/streaming subtitles)
        old_lines = self._last_text.strip().splitlines()
        new_lines = current_text.strip().splitlines()

        if len(new_lines) > len(old_lines):
            # Find where old text ends in new text
            overlap = 0
            for i, old_line in enumerate(old_lines):
                if i < len(new_lines):
                    line_ratio = SequenceMatcher(None, old_line.strip(), new_lines[i].strip()).ratio()
                    if line_ratio > 0.8:
                        overlap = i + 1
                    else:
                        break

            if overlap > 0:
                new_portion = "\n".join(new_lines[overlap:])
                self._last_text = current_text
                return new_portion.strip() if new_portion.strip() else None

        self._last_text = current_text
        return current_text

    def reset(self) -> None:
        self._last_text = ""
