from __future__ import annotations

import re
import time
from collections import Counter


def _clean(text: str) -> str:
    """Normalize text for comparison: collapse whitespace, strip OCR artifacts."""
    # Remove OCR cursor/boundary artifacts: |, ], [, ), \
    text = re.sub(r"[\|\]\[\)\\\}]+", "", text)
    return " ".join(text.split())


def _extract_words(text: str) -> list[str]:
    """Extract alphabetic words (>=3 chars) for word-level comparison."""
    words = []
    for w in text.split():
        # Strip punctuation from edges
        w = re.sub(r"^[^\w]+|[^\w]+$", "", w)
        if len(w) >= 3 and sum(1 for c in w if c.isalpha()) / max(len(w), 1) >= 0.7:
            words.append(w.lower())
    return words


# --- Triple similarity metrics (ported from RST Logic.cs) ---

# Stop words for keyword similarity — common words that don't carry meaning
_STOP_WORDS_EN = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "can", "could", "must", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "out",
    "off", "over", "under", "again", "further", "then", "once", "and",
    "but", "or", "nor", "not", "no", "so", "if", "than", "too", "very",
    "just", "about", "up", "down", "here", "there", "when", "where",
    "how", "all", "both", "each", "few", "more", "most", "other", "some",
    "such", "only", "own", "same", "that", "this", "what", "which", "who",
    "it", "its", "he", "she", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "his", "our", "their", "i", "you",
})

_STOP_WORDS_RU = frozenset({
    "и", "в", "на", "не", "что", "он", "она", "оно", "они", "мы", "вы",
    "я", "ты", "с", "а", "но", "да", "по", "к", "за", "из", "от", "до",
    "у", "о", "как", "все", "это", "так", "его", "её", "их", "уже", "бы",
    "ли", "же", "то", "ещё", "или", "мне", "тебе", "ему", "ей", "нам",
    "вам", "им", "себе", "был", "была", "было", "были", "быть", "есть",
    "будет", "тот", "тут", "там", "вот", "при", "для", "про", "без",
})

_STOP_WORDS = _STOP_WORDS_EN | _STOP_WORDS_RU


def _dice_coefficient(s1: str, s2: str) -> float:
    """Dice coefficient using character bigrams.

    Creates 2-grams from both strings and computes:
    dice = 2 * |intersection| / (|set1| + |set2|)
    """
    s1 = s1.lower()
    s2 = s2.lower()

    if len(s1) < 2 or len(s2) < 2:
        # For very short strings: character overlap
        if not s1 or not s2:
            return 0.0
        common = sum(1 for c in s1 if c in s2)
        return common / max(len(s1), len(s2))

    bigrams1 = Counter(s1[i:i + 2] for i in range(len(s1) - 1))
    bigrams2 = Counter(s2[i:i + 2] for i in range(len(s2) - 1))

    intersection = sum((bigrams1 & bigrams2).values())
    total = sum(bigrams1.values()) + sum(bigrams2.values())

    if total == 0:
        return 0.0
    return (2.0 * intersection) / total


def _keyword_similarity(s1: str, s2: str) -> float:
    """Keyword similarity: Dice on meaningful words (stop words removed)."""
    splitter = re.compile(r"[ ,.\-!?;:\n\r\t]+")

    kw1 = {w.lower() for w in splitter.split(s1) if w and w.lower() not in _STOP_WORDS}
    kw2 = {w.lower() for w in splitter.split(s2) if w and w.lower() not in _STOP_WORDS}

    if not kw1 or not kw2:
        return _dice_coefficient(s1, s2)

    common = len(kw1 & kw2)
    return (2.0 * common) / (len(kw1) + len(kw2))


def _word_overlap_jaccard(s1: str, s2: str) -> float:
    """Word overlap using Jaccard index."""
    words1 = {w.lower() for w in s1.split() if w}
    words2 = {w.lower() for w in s2.split() if w}

    if not words1 or not words2:
        return 0.0

    common = len(words1 & words2)
    union = len(words1) + len(words2) - common

    if union == 0:
        return 0.0
    return common / union


def is_text_similar(s1: str, s2: str, threshold: float = 0.75) -> bool:
    """Check if two texts are similar using triple metric (RST-style).

    Computes three similarity metrics and uses the maximum:
    1. Dice coefficient (bigram-based)
    2. Keyword similarity (stop words removed)
    3. Word overlap (Jaccard index)

    Returns True if max(metrics) >= threshold.
    """
    if not s1 and not s2:
        return True
    if not s1 or not s2:
        return False
    if s1 == s2:
        return True

    # For very short strings (< 5 chars), use exact matching
    if len(s1) < 5 or len(s2) < 5:
        return s1.strip().lower() == s2.strip().lower()

    dice = _dice_coefficient(s1, s2)
    keyword = _keyword_similarity(s1, s2)
    jaccard = _word_overlap_jaccard(s1, s2)

    max_sim = max(dice, keyword, jaccard)
    return max_sim >= threshold


class TextDiffer:
    """Detects new/changed text to avoid repeating already-spoken content.

    The key challenge: OCR runs every ~500ms and may produce:
    1. Identical text (same subtitle still on screen) -> must NOT re-speak
    2. Slightly different text (OCR jitter: extra space, punctuation change) -> must NOT re-speak
    3. Genuinely new text (subtitle changed) -> MUST speak
    4. Growing text (scrolling/streaming subtitles) -> wait for stabilization, then speak

    Stabilization: when text is detected as actively growing (typewriter
    subtitles), we buffer it and wait for settle_time_ms to elapse since
    the last change before emitting.  This prevents reading partial words.
    """

    def __init__(self, similarity_threshold: float = 0.75, settle_time_ms: int = 300):
        self._last_spoken: str = ""       # last text that was actually emitted (spoken)
        self._threshold = similarity_threshold

        # Stabilization state for growing text (time-based)
        self._settle_time_ms = settle_time_ms
        self._pending_text: str | None = None     # buffered text waiting to stabilize
        self._last_change_time: float | None = None  # monotonic time of last text change

    def set_settle_time(self, ms: int) -> None:
        """Update settle time (called when user changes the setting)."""
        self._settle_time_ms = ms

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

    def _is_settled(self) -> bool:
        """Check if enough time has passed since last text change."""
        if self._last_change_time is None:
            return True
        elapsed_ms = (time.monotonic() - self._last_change_time) * 1000
        return elapsed_ms >= self._settle_time_ms

    def _mark_changed(self) -> None:
        """Record that text just changed (reset settle timer)."""
        self._last_change_time = time.monotonic()

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

            # Exact match (cleaned) → check settle time
            if clean_new == clean_pending:
                self._pending_text = current_text
                if self._is_settled():
                    print(f"[TextDiffer] Settled (time elapsed)")
                    return self._flush_pending()
                return None

            # IMPORTANT: check growth BEFORE similarity!
            # Growing text can have 0.87+ similarity to pending
            # but must NOT be treated as "stable".

            # Pending is a prefix of new → text is STILL growing
            if self._is_growth(self._pending_text, current_text):
                self._pending_text = current_text
                self._mark_changed()  # reset settle timer
                print(f"[TextDiffer] Still growing: ...{repr(clean_new[-60:])}")
                return None

            # New is a subset of pending → partial OCR, keep waiting
            if clean_new in clean_pending:
                return None

            # Similar to pending (OCR jitter, same length) → check settle
            if is_text_similar(current_text, self._pending_text, self._threshold):
                self._pending_text = current_text
                if self._is_settled():
                    print(f"[TextDiffer] Settled (jitter, time elapsed)")
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
            self._mark_changed()
            print(f"[TextDiffer] Growth detected, buffering: ...{repr(clean_new[-60:])}")
            return None

        # Check line-based growth (new lines appeared)
        spoken_lines = self._last_spoken.strip().splitlines()
        new_lines = current_text.strip().splitlines()
        if len(new_lines) > len(spoken_lines):
            overlap = 0
            for i, spoken_line in enumerate(spoken_lines):
                if i < len(new_lines):
                    clean_s = _clean(spoken_line)
                    clean_n = _clean(new_lines[i])
                    if is_text_similar(clean_s, clean_n, 0.7):
                        overlap = i + 1
                    else:
                        break
            if overlap > 0:
                # New lines appeared — buffer for stabilization
                self._pending_text = current_text
                self._mark_changed()
                print(f"[TextDiffer] New lines detected, buffering")
                return None

        # High similarity = OCR jitter, not a real change
        if is_text_similar(clean_spoken, clean_new, self._threshold):
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
                    if is_text_similar(_clean(old_line), _clean(new_lines[i]), 0.7):
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
        self._last_change_time = None

    def reset(self) -> None:
        self._last_spoken = ""
        self._clear_pending()
