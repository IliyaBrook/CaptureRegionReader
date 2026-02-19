from __future__ import annotations

import re
from difflib import SequenceMatcher


def _clean(text: str) -> str:
    """Normalize text for comparison: collapse whitespace, strip OCR artifacts."""
    # Remove OCR cursor/boundary artifacts: |, ], [, ), \
    text = re.sub(r"[\|\]\[\)\\\}]+", "", text)
    return " ".join(text.split())


def _extract_words(text: str) -> list[str]:
    """Extract alphabetic words (≥3 chars) for word-level comparison."""
    words = []
    for w in text.split():
        # Strip punctuation from edges
        w = re.sub(r"^[^\w]+|[^\w]+$", "", w)
        if len(w) >= 3 and sum(1 for c in w if c.isalpha()) / max(len(w), 1) >= 0.7:
            words.append(w.lower())
    return words


class TextDiffer:
    """Detects new/changed text to avoid repeating already-spoken content.

    The key challenge: OCR runs every ~500ms and may produce:
    1. Identical text (same subtitle still on screen) → must NOT re-speak
    2. Slightly different text (OCR jitter: extra space, punctuation change) → must NOT re-speak
    3. Genuinely new text (subtitle changed) → MUST speak
    4. Growing text (scrolling/streaming subtitles) → wait for stabilization, then speak

    Stabilization: when text is detected as actively growing (typewriter
    subtitles), we buffer it and wait for it to stop changing before
    emitting.  This prevents reading partial words like "МНОГОЧИСЛЕНН"
    followed by "ЫЕ ПРОТИВНИКИ".
    """

    # How many consecutive OCR cycles the text must remain stable
    # before we emit buffered growing text.  At 500ms interval this
    # means ~500ms of stability — enough to confirm text stopped typing.
    STABLE_CYCLES = 1

    def __init__(self, similarity_threshold: float = 0.85):
        self._last_spoken: str = ""       # last text that was actually emitted (spoken)
        self._threshold = similarity_threshold

        # Stabilization state for growing text
        self._pending_text: str | None = None   # buffered text waiting to stabilize
        self._stable_count: int = 0              # how many cycles it's been unchanged

    def _is_growth(self, old_text: str, new_text: str) -> bool:
        """Check if new_text is old_text with more content appended.

        Uses both cleaned-string containment and word-level analysis
        to handle OCR artifacts like |, ], cursor chars.
        """
        clean_old = _clean(old_text)
        clean_new = _clean(new_text)

        # Direct containment after removing OCR artifacts
        if clean_old and clean_old in clean_new and len(clean_new) > len(clean_old) + 2:
            return True

        # Word-level: old words are a subset of new words, and new has more
        words_old = _extract_words(old_text)
        words_new = _extract_words(new_text)
        if len(words_old) >= 1 and len(words_new) > len(words_old):
            # Check that most old words appear in new text
            set_old = set(words_old)
            set_new = set(words_new)
            if len(set_old) > 0:
                overlap = len(set_old & set_new)
                if overlap / len(set_old) >= 0.6:
                    return True

        return False

    def _is_similar(self, text_a: str, text_b: str) -> bool:
        """Check if two texts are similar enough to be the same subtitle."""
        clean_a = _clean(text_a)
        clean_b = _clean(text_b)

        if clean_a == clean_b:
            return True

        if SequenceMatcher(None, clean_a, clean_b).ratio() >= self._threshold:
            return True

        return False

    def get_new_text(self, current_text: str) -> str | None:
        """Return text to speak, or None if no meaningful change detected."""
        if not current_text:
            # No text on screen — if we had pending text, emit it now
            # (subtitle disappeared, so whatever we had is final)
            return self._flush_pending()

        if not self._last_spoken:
            # First text ever — emit immediately (no stabilization needed)
            self._last_spoken = current_text
            self._clear_pending()
            return current_text

        # --- If we have pending (buffered growing) text, compare against it ---
        if self._pending_text is not None:
            clean_new = _clean(current_text)
            clean_pending = _clean(self._pending_text)

            # Exact match (cleaned) → definitely stable
            if clean_new == clean_pending:
                self._stable_count += 1
                self._pending_text = current_text
                print(f"[TextDiffer] Stability: {self._stable_count}/{self.STABLE_CYCLES}")
                if self._stable_count >= self.STABLE_CYCLES:
                    return self._flush_pending()
                return None

            # IMPORTANT: check growth BEFORE similarity!
            # Growing text can have 0.87+ similarity to pending
            # but must NOT be treated as "stable".

            # Pending is a prefix of new → text is STILL growing
            if self._is_growth(self._pending_text, current_text):
                self._pending_text = current_text
                self._stable_count = 0  # reset — still growing
                print(f"[TextDiffer] Still growing: ...{repr(clean_new[-60:])}")
                return None

            # New is a subset of pending → partial OCR, keep waiting
            if clean_new in clean_pending:
                return None

            # Similar to pending (OCR jitter, same length) → count as stable
            if self._is_similar(current_text, self._pending_text):
                self._stable_count += 1
                self._pending_text = current_text
                print(f"[TextDiffer] Stability (jitter): {self._stable_count}/{self.STABLE_CYCLES}")
                if self._stable_count >= self.STABLE_CYCLES:
                    return self._flush_pending()
                return None

            # Text changed completely while we were waiting → new subtitle
            # Flush pending and handle the new text
            flushed = self._flush_pending()
            if flushed:
                # Return flushed text.  Current text will be compared
                # against updated _last_spoken on the NEXT OCR cycle.
                return flushed

        # --- Normal (non-pending) comparison against last spoken text ---
        clean_spoken = _clean(self._last_spoken)
        clean_new = _clean(current_text)

        # Exact match — definitely skip
        if clean_spoken == clean_new:
            return None

        # New is a subset of spoken → partial OCR of same subtitle → skip
        if clean_new in clean_spoken:
            return None

        # Detect growth: last_spoken content is within new text
        if self._is_growth(self._last_spoken, current_text):
            # Text grew — buffer it, don't emit yet
            self._pending_text = current_text
            self._stable_count = 0
            print(f"[TextDiffer] Growth detected, buffering: ...{repr(clean_new[-60:])}")
            return None

        # Check line-based growth (new lines appeared)
        spoken_lines = self._last_spoken.strip().splitlines()
        new_lines = current_text.strip().splitlines()
        if len(new_lines) > len(spoken_lines):
            overlap = 0
            for i, spoken_line in enumerate(spoken_lines):
                if i < len(new_lines):
                    line_ratio = SequenceMatcher(
                        None, _clean(spoken_line), _clean(new_lines[i])
                    ).ratio()
                    if line_ratio > 0.7:
                        overlap = i + 1
                    else:
                        break
            if overlap > 0:
                # New lines appeared — buffer for stabilization
                self._pending_text = current_text
                self._stable_count = 0
                print(f"[TextDiffer] New lines detected, buffering")
                return None

        ratio = SequenceMatcher(None, clean_spoken, clean_new).ratio()

        # High similarity = OCR jitter, not a real change
        if ratio >= self._threshold:
            return None

        # Text changed substantially — this is a completely new subtitle
        self._last_spoken = current_text
        self._clear_pending()
        return current_text

    def _flush_pending(self) -> str | None:
        """Emit the pending (buffered) text and clear the buffer."""
        if self._pending_text is None:
            return None

        pending = self._pending_text
        # _last_spoken is the text that was ALREADY spoken
        old_text = self._last_spoken
        clean_old = _clean(old_text)
        clean_pending = _clean(pending)

        # Update state: pending becomes the new "last spoken" text
        self._last_spoken = pending
        self._clear_pending()

        # Extract only the NEW portion (what wasn't already spoken)
        # Case: within-line growth — old is a prefix of pending (cleaned)
        if clean_old and clean_old in clean_pending and len(clean_pending) > len(clean_old) + 3:
            tail = clean_pending[len(clean_old):].strip()
            if tail:
                print(f"[TextDiffer] Emit stabilized tail: {repr(tail[:100])}")
                return tail

        # Case: new lines appeared — extract just the new lines
        old_lines = old_text.strip().splitlines()
        new_lines = pending.strip().splitlines()
        if len(new_lines) > len(old_lines):
            overlap = 0
            for i, old_line in enumerate(old_lines):
                if i < len(new_lines):
                    line_ratio = SequenceMatcher(
                        None, _clean(old_line), _clean(new_lines[i])
                    ).ratio()
                    if line_ratio > 0.7:
                        overlap = i + 1
                    else:
                        break
            if overlap > 0:
                new_portion = "\n".join(new_lines[overlap:])
                if new_portion.strip():
                    print(f"[TextDiffer] Emit stabilized new lines: {repr(new_portion.strip()[:100])}")
                    return new_portion.strip()

        # Fallback: word-level diff — find new words not in old
        words_old = _extract_words(old_text)
        words_pending = _extract_words(pending)
        if words_old and words_pending:
            # Find the first word in pending that wasn't in old
            set_old = set(words_old)
            new_words = [w for w in words_pending if w not in set_old]
            if new_words:
                # Find position of first new word in the clean pending text
                # to extract everything from that point
                first_new = new_words[0]
                idx = clean_pending.lower().find(first_new)
                if idx >= 0:
                    tail = clean_pending[idx:].strip()
                    if tail:
                        print(f"[TextDiffer] Emit stabilized (word-diff): {repr(tail[:100])}")
                        return tail

        # Last resort: emit full pending text
        print(f"[TextDiffer] Emit full stabilized text: {repr(clean_pending[:100])}")
        return pending

    def _clear_pending(self) -> None:
        """Clear the stabilization buffer."""
        self._pending_text = None
        self._stable_count = 0

    def reset(self) -> None:
        self._last_spoken = ""
        self._clear_pending()
