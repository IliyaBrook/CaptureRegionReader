"""Text deduplication and change detection for real-time OCR output.

Detects new/changed subtitle text to avoid repeating already-spoken content.
Uses a triple similarity metric (Dice + Keyword + Jaccard) with an optional
Jaro distance check for short texts.

Similarity algorithm inspired by Translumo's TextResultCacheService which
uses an average of Jaro and Dice coefficients for deduplication.
"""

from __future__ import annotations

import re
import time
import unicodedata
from collections import Counter, deque


# ---------------------------------------------------------------------------
# Similarity metric constants
# ---------------------------------------------------------------------------

# Default similarity threshold for deduplication.
# Texts with max(metrics) >= this value are considered "the same".
DEFAULT_SIMILARITY_THRESHOLD = 0.85

# Threshold for very short strings (< 5 chars) -- exact match only.
SHORT_STRING_LENGTH = 5

# Growth detection: minimum overlap ratio for word-level subset check.
GROWTH_WORD_OVERLAP_RATIO = 0.6

# Minimum length difference to consider text as "grown" (not just OCR jitter).
GROWTH_MIN_CHAR_DIFF = 2

# Growing subtitles mode constants
DEFAULT_GROWTH_TIMEOUT_MS = 2000   # Initial timeout before declaring growth done
MIN_ADAPTIVE_TIMEOUT_MS = 500      # Floor for adaptive timeout (prevents near-zero)
TIMEOUT_MULTIPLIER = 3             # adaptive_timeout = avg_interval * multiplier
GROWTH_INTERVALS_MAXLEN = 5        # Sliding window size for interval averaging


def _clean(text: str) -> str:
    """Normalize text for comparison: collapse whitespace, strip OCR artifacts."""
    # Remove OCR cursor/boundary artifacts: |, ], [, ), \, }
    text = re.sub(r"[\|\]\[\)\\\}]+", "", text)
    return " ".join(text.split())


_RE_NON_ALNUM = re.compile(r"[^\w\s]", re.UNICODE)


def _normalize_for_compare(text: str) -> str:
    """Aggressively normalize text for deduplication comparison only.

    Strips ALL punctuation, lowercases, applies NFKC unicode normalization,
    and collapses whitespace. Never used for output — only for equality
    and similarity checks against previously spoken text.
    """
    text = unicodedata.normalize("NFKC", text)
    text = _RE_NON_ALNUM.sub("", text)
    text = text.lower()
    return " ".join(text.split())


def _extract_words(text: str) -> list[str]:
    """Extract alphabetic words (>=3 chars) for word-level comparison.

    Strips punctuation from edges and requires >=70% alpha characters
    to filter out garbage tokens.
    """
    words = []
    for w in text.split():
        w = re.sub(r"^[^\w]+|[^\w]+$", "", w)
        if len(w) >= 3 and sum(1 for c in w if c.isalpha()) / max(len(w), 1) >= 0.7:
            words.append(w.lower())
    return words


# ---------------------------------------------------------------------------
# Similarity metrics
# ---------------------------------------------------------------------------

# Stop words for keyword similarity -- common words that don't carry meaning.
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
    "\u0438", "\u0432", "\u043d\u0430", "\u043d\u0435",
    "\u0447\u0442\u043e", "\u043e\u043d", "\u043e\u043d\u0430",
    "\u043e\u043d\u043e", "\u043e\u043d\u0438", "\u043c\u044b",
    "\u0432\u044b", "\u044f", "\u0442\u044b", "\u0441", "\u0430",
    "\u043d\u043e", "\u0434\u0430", "\u043f\u043e", "\u043a",
    "\u0437\u0430", "\u0438\u0437", "\u043e\u0442", "\u0434\u043e",
    "\u0443", "\u043e", "\u043a\u0430\u043a", "\u0432\u0441\u0435",
    "\u044d\u0442\u043e", "\u0442\u0430\u043a", "\u0435\u0433\u043e",
    "\u0435\u0451", "\u0438\u0445", "\u0443\u0436\u0435",
    "\u0431\u044b", "\u043b\u0438", "\u0436\u0435", "\u0442\u043e",
    "\u0435\u0449\u0451", "\u0438\u043b\u0438",
    "\u043c\u043d\u0435", "\u0442\u0435\u0431\u0435",
    "\u0435\u043c\u0443", "\u0435\u0439", "\u043d\u0430\u043c",
    "\u0432\u0430\u043c", "\u0438\u043c",
    "\u0441\u0435\u0431\u0435", "\u0431\u044b\u043b",
    "\u0431\u044b\u043b\u0430", "\u0431\u044b\u043b\u043e",
    "\u0431\u044b\u043b\u0438", "\u0431\u044b\u0442\u044c",
    "\u0435\u0441\u0442\u044c", "\u0431\u0443\u0434\u0435\u0442",
    "\u0442\u043e\u0442", "\u0442\u0443\u0442", "\u0442\u0430\u043c",
    "\u0432\u043e\u0442", "\u043f\u0440\u0438",
    "\u0434\u043b\u044f", "\u043f\u0440\u043e",
    "\u0431\u0435\u0437",
})

_STOP_WORDS = _STOP_WORDS_EN | _STOP_WORDS_RU

# Compiled regex for splitting on punctuation and whitespace.
_KEYWORD_SPLITTER = re.compile(r"[ ,.\-!?;:\n\r\t]+")


def _dice_coefficient(s1: str, s2: str) -> float:
    """Dice coefficient using character bigrams.

    Creates 2-grams from both strings and computes:
    dice = 2 * |intersection| / (|set1| + |set2|)

    Uses Counter (multiset) for accurate bigram overlap counting.
    """
    s1 = s1.lower()
    s2 = s2.lower()

    if len(s1) < 2 or len(s2) < 2:
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


def _jaro_similarity(s1: str, s2: str) -> float:
    """Jaro distance algorithm (ported from Translumo StringExtensions).

    Returns similarity from 0.0 to 1.0. Effective for catching single-character
    transpositions and short-string comparisons where bigram Dice breaks down.

    Compared to Dice:
    - Better for short strings (3-10 chars)
    - More sensitive to character order
    - Translumo uses average of Jaro + Dice for cache deduplication
    """
    if not s1 or not s2:
        return 0.0

    s1_lower = s1.lower()
    s2_lower = s2.lower()

    if s1_lower == s2_lower:
        return 1.0

    len1 = len(s1_lower)
    len2 = len(s2_lower)

    # Maximum distance for matching characters
    match_distance = (min(len1, len2) // 2) + 1

    s1_matches = [False] * len1
    s2_matches = [False] * len2

    matches = 0
    transpositions = 0

    # Find matching characters
    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance, len2)
        for j in range(start, end):
            if s2_matches[j] or s1_lower[i] != s2_lower[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    # Count transpositions
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1_lower[i] != s2_lower[k]:
            transpositions += 1
        k += 1

    transpositions //= 2

    return (
        (matches / (3.0 * len1))
        + (matches / (3.0 * len2))
        + ((matches - transpositions) / (3.0 * matches))
    )


def _keyword_similarity(s1: str, s2: str) -> float:
    """Keyword similarity: Dice on meaningful words (stop words removed).

    Extracts content-bearing words from both strings and computes
    set-based Dice coefficient. Falls back to character-level Dice
    if no keywords remain after stop-word removal.
    """
    kw1 = {w.lower() for w in _KEYWORD_SPLITTER.split(s1) if w and w.lower() not in _STOP_WORDS}
    kw2 = {w.lower() for w in _KEYWORD_SPLITTER.split(s2) if w and w.lower() not in _STOP_WORDS}

    if not kw1 or not kw2:
        return _dice_coefficient(s1, s2)

    common = len(kw1 & kw2)
    return (2.0 * common) / (len(kw1) + len(kw2))


def _word_overlap_jaccard(s1: str, s2: str) -> float:
    """Word overlap using Jaccard index.

    Jaccard = |intersection| / |union|
    Useful for catching rearranged subtitles where word order changed.
    """
    words1 = {w.lower() for w in s1.split() if w}
    words2 = {w.lower() for w in s2.split() if w}

    if not words1 or not words2:
        return 0.0

    common = len(words1 & words2)
    union = len(words1) + len(words2) - common

    if union == 0:
        return 0.0
    return common / union


def is_text_similar(s1: str, s2: str, threshold: float = DEFAULT_SIMILARITY_THRESHOLD) -> bool:
    """Check if two texts are similar using combined metrics.

    Computes four similarity metrics and uses the maximum:
    1. Dice coefficient (bigram-based)
    2. Jaro distance (character-level, transposition-aware)
    3. Keyword similarity (stop words removed)
    4. Word overlap (Jaccard index)

    Additionally computes a Translumo-style average of Jaro + Dice
    for balanced scoring.

    Returns True if any metric >= threshold.
    """
    if not s1 and not s2:
        return True
    if not s1 or not s2:
        return False
    if s1 == s2:
        return True

    # For very short strings, use exact matching to avoid false positives.
    if len(s1) < SHORT_STRING_LENGTH or len(s2) < SHORT_STRING_LENGTH:
        return s1.strip().lower() == s2.strip().lower()

    dice = _dice_coefficient(s1, s2)
    jaro = _jaro_similarity(s1, s2)
    keyword = _keyword_similarity(s1, s2)
    jaccard = _word_overlap_jaccard(s1, s2)

    # Translumo-style combined score: average of Jaro and Dice.
    # This gives a balanced metric that's better than either alone.
    jaro_dice_avg = ((jaro if jaro > 0 else dice) + dice) / 2.0

    max_sim = max(dice, jaro, keyword, jaccard, jaro_dice_avg)
    return max_sim >= threshold


# ---------------------------------------------------------------------------
# Text differ (change detection + stabilization)
# ---------------------------------------------------------------------------

class TextDiffer:
    """Detects new/changed text to avoid repeating already-spoken content.

    The key challenge: OCR runs every ~500ms and may produce:
    1. Identical text (same subtitle still on screen) -> must NOT re-speak
    2. Slightly different text (OCR jitter: extra space, punctuation change) -> must NOT re-speak
    3. Genuinely new text (subtitle changed) -> MUST speak
    4. Growing text (scrolling/streaming subtitles) -> wait for stabilization, then speak

    Stabilization: when text is detected as actively growing (typewriter
    subtitles), we buffer it and wait for settle_time_ms to elapse since
    the last change before emitting. This prevents reading partial words.
    """

    def __init__(
        self,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ):
        # --- Shared state (both modes) ---
        self._last_spoken: str = ""
        self._spoken_history: deque[str] = deque(maxlen=5)
        self._threshold = similarity_threshold

        # Post-speech cooldown: suppress similar texts shortly after emission
        self._last_emit_time: float = 0.0
        self._cooldown_ms: int = 2000

        # --- Normal mode state (growing_subtitles=False) ---
        self._settle_time_ms: int = 0  # 0 = immediate flush
        self._pending_text: str | None = None
        self._last_change_time: float | None = None

        # --- Growing subtitles mode state ---
        self._growing_subtitles: bool = False
        self._buffer: list[str] = []       # Accumulated words
        self._cursor: int = 0              # How far we've spoken into _buffer
        self._growth_intervals: deque[float] = deque(maxlen=GROWTH_INTERVALS_MAXLEN)
        self._adaptive_timeout_ms: float = DEFAULT_GROWTH_TIMEOUT_MS
        self._last_update_time: float | None = None
        self._last_visible_text: str = ""
        self._empty_frames: int = 0        # Consecutive empty OCR frames (debounce)

    def set_growing_subtitles(self, enabled: bool) -> None:
        """Switch between growing subtitles mode and normal mode."""
        self._growing_subtitles = enabled
        self.reset()

    def _is_in_history(self, text: str) -> bool:
        """Check if normalized text matches any recently spoken text."""
        norm = _normalize_for_compare(text)
        if not norm:
            return False
        for hist_norm in self._spoken_history:
            if norm == hist_norm:
                return True
            if is_text_similar(norm, hist_norm, self._threshold):
                return True
        return False

    def _record_spoken(self, text: str) -> None:
        """Record text as spoken: update last_spoken, history, and cooldown."""
        self._last_spoken = text
        norm = _normalize_for_compare(text)
        if norm:
            self._spoken_history.append(norm)
        self._last_emit_time = time.monotonic()

    def _is_growth(self, old_text: str, new_text: str) -> bool:
        """Check if new_text is old_text with more content appended.

        Uses both cleaned-string containment and word-level analysis
        to handle OCR artifacts like |, ], cursor chars.
        """
        clean_old = _clean(old_text)
        clean_new = _clean(new_text)

        # Direct containment after removing OCR artifacts
        if (
            clean_old
            and clean_old in clean_new
            and len(clean_new) >= len(clean_old) + GROWTH_MIN_CHAR_DIFF
        ):
            return True

        # Word-level: old words are a subset of new words, and new has more
        words_old = _extract_words(old_text)
        words_new = _extract_words(new_text)
        if len(words_old) >= 1 and len(words_new) > len(words_old):
            set_old = set(words_old)
            set_new = set(words_new)
            if len(set_old) > 0:
                overlap = len(set_old & set_new)
                if overlap / len(set_old) >= GROWTH_WORD_OVERLAP_RATIO:
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
        if self._growing_subtitles:
            return self._get_new_text_growing(current_text)
        return self._get_new_text_normal(current_text)

    def _get_new_text_normal(self, current_text: str) -> str | None:
        """Normal mode: immediate emit, no buffering for growing text."""
        if not current_text:
            # No text on screen -- if we had pending text, emit it now
            # (subtitle disappeared, so whatever we had is final)
            return self._flush_pending()

        if not self._last_spoken:
            # First text ever -- emit immediately
            self._record_spoken(current_text)
            self._clear_pending()
            return current_text

        # --- If we have pending (buffered growing) text, compare against it ---
        if self._pending_text is not None:
            norm_new = _normalize_for_compare(current_text)
            norm_pending = _normalize_for_compare(self._pending_text)

            # Exact match (normalized) -- check settle time
            if norm_new == norm_pending:
                self._pending_text = current_text
                if self._is_settled():
                    return self._flush_pending()
                return None

            # Check growth BEFORE similarity -- growing text can have high
            # similarity to pending but must NOT be treated as "stable".
            if self._is_growth(self._pending_text, current_text):
                self._pending_text = current_text
                self._mark_changed()
                return None

            # New is a subset of pending -- partial OCR, keep waiting
            clean_new = _clean(current_text)
            clean_pending = _clean(self._pending_text)
            if clean_new in clean_pending:
                return None

            # Similar to pending (OCR jitter) -- check settle
            if is_text_similar(current_text, self._pending_text, self._threshold):
                self._pending_text = current_text
                if self._is_settled():
                    return self._flush_pending()
                return None

            # Text changed completely while we were waiting -- new subtitle.
            # Flush pending and handle the new text.
            flushed = self._flush_pending()
            if flushed:
                # Return flushed text. Current text will be compared
                # against updated _last_spoken on the NEXT OCR cycle.
                return flushed

        # --- Normal comparison against last spoken text ---
        norm_spoken = _normalize_for_compare(self._last_spoken)
        norm_new = _normalize_for_compare(current_text)

        # Stage 1: fast exact match on aggressively normalized text
        if norm_spoken == norm_new:
            return None

        # Check against full history (catches A->B->A cycling)
        if self._is_in_history(current_text):
            return None

        # New is a subset of spoken -- partial OCR of same subtitle
        clean_spoken = _clean(self._last_spoken)
        clean_new = _clean(current_text)
        if clean_new in clean_spoken:
            return None

        # Detect growth: last_spoken content is within new text
        if self._is_growth(self._last_spoken, current_text):
            self._pending_text = current_text
            self._mark_changed()
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
                self._pending_text = current_text
                self._mark_changed()
                return None

        # Stage 2: similarity check with cooldown-aware threshold
        effective_threshold = self._threshold
        elapsed_since_emit = (time.monotonic() - self._last_emit_time) * 1000
        if elapsed_since_emit < self._cooldown_ms:
            effective_threshold = max(0.65, self._threshold - 0.15)

        if is_text_similar(norm_spoken, norm_new, effective_threshold):
            return None

        # Text changed substantially -- completely new subtitle
        self._record_spoken(current_text)
        self._clear_pending()
        return current_text

    def _flush_pending(self) -> str | None:
        """Emit the pending (buffered) text and clear the buffer.

        Tries to extract only the NEW portion that wasn't already spoken:
        1. Within-line growth: old is a prefix of pending
        2. New lines appeared: extract just the new lines
        3. Word-level diff: find words not in old text
        4. Fallback: emit full pending text
        """
        if self._pending_text is None:
            return None

        pending = self._pending_text
        old_text = self._last_spoken
        clean_old = _clean(old_text)
        clean_pending = _clean(pending)

        # Update state: pending becomes the new "last spoken"
        self._record_spoken(pending)
        self._clear_pending()

        # Case 1: within-line growth -- old is a prefix of pending (cleaned)
        if (
            clean_old
            and clean_old in clean_pending
            and len(clean_pending) > len(clean_old) + 3
        ):
            tail = clean_pending[len(clean_old):].strip()
            if tail:
                return tail

        # Case 2: new lines appeared -- extract just the new lines
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
                    return new_portion.strip()

        # Case 3: word-level diff -- find new words not in old
        words_old = _extract_words(old_text)
        words_pending = _extract_words(pending)
        if words_old and words_pending:
            set_old = set(words_old)
            new_words = [w for w in words_pending if w not in set_old]
            if new_words:
                first_new = new_words[0]
                idx = clean_pending.lower().find(first_new)
                if idx >= 0:
                    tail = clean_pending[idx:].strip()
                    if tail:
                        return tail

        # Case 4: fallback -- emit full pending text
        return pending

    def _clear_pending(self) -> None:
        """Clear the stabilization buffer."""
        self._pending_text = None
        self._last_change_time = None

    def reset(self) -> None:
        """Reset all state (used when region changes or reading restarts)."""
        self._last_spoken = ""
        self._spoken_history.clear()
        self._last_emit_time = 0.0
        self._clear_pending()
        # Growing mode state
        self._buffer = []
        self._cursor = 0
        self._growth_intervals.clear()
        self._adaptive_timeout_ms = DEFAULT_GROWTH_TIMEOUT_MS
        self._last_update_time = None
        self._last_visible_text = ""
        self._empty_frames = 0

    # ------------------------------------------------------------------
    # Growing subtitles mode
    # ------------------------------------------------------------------

    def _get_new_text_growing(self, current_text: str) -> str | None:
        """Growing subtitles mode: track growth and speak incrementally."""
        now = time.monotonic()

        # --- Check timeout first ---
        if self._last_update_time is not None and self._buffer:
            elapsed_ms = (now - self._last_update_time) * 1000
            if elapsed_ms > self._adaptive_timeout_ms:
                result = self._flush_growing_buffer()
                if current_text:
                    self._start_new_growing_cycle(current_text)
                return result

        # --- No text on screen ---
        if not current_text:
            if self._buffer:
                # Debounce: OCR can flicker (miss a frame). Require 2 empty
                # frames in a row before flushing to avoid losing text.
                self._empty_frames += 1
                if self._empty_frames >= 2:
                    self._empty_frames = 0
                    return self._flush_growing_buffer()
                return None
            return None

        self._empty_frames = 0

        # --- First text (empty buffer, no last_visible) ---
        if not self._last_visible_text and not self._buffer:
            self._start_new_growing_cycle(current_text)
            return None

        # --- Classify change ---
        situation = self._classify_growing_change(current_text)

        if situation == "same":
            return None
        elif situation == "grew":
            self._handle_growth(current_text, now)
            return None
        elif situation == "shifted":
            return self._handle_shift(current_text, now)
        else:  # "replaced"
            return self._handle_replacement(current_text)

    def _classify_growing_change(self, new_text: str) -> str:
        """Classify how on-screen text changed: same/grew/shifted/replaced.

        IMPORTANT: Growth and shift detection run BEFORE similarity check.
        Growing text has high similarity to its prefix (shared bigrams, words),
        so checking similarity first would misclassify growth as "same" and
        cause words to be silently dropped.
        """
        old = self._last_visible_text
        norm_old = _normalize_for_compare(old)
        norm_new = _normalize_for_compare(new_text)

        # Exact normalized match — definitely same
        if norm_old == norm_new:
            return "same"

        # Situation A — growth: old content still present, new words appended.
        # Must be checked BEFORE similarity to avoid misclassifying growth
        # as "same" (growing text shares most bigrams with its prefix).
        if self._is_growth(old, new_text):
            return "grew"

        # Situation B — shift: tail of old visible as prefix of new.
        # Also checked before similarity for the same reason.
        # shift_start must be > 0: at least one word disappeared from the top.
        # shift_start == 0 means all extracted words still match — not a shift.
        old_words = _extract_words(old)
        new_words = _extract_words(new_text)
        if old_words and new_words:
            shift_start = self._find_shift_overlap(old_words, new_words)
            if shift_start is not None and shift_start > 0:
                return "shifted"

        # OCR jitter: text didn't grow or shift, but is very similar
        if is_text_similar(old, new_text, self._threshold):
            return "same"

        return "replaced"

    def _find_shift_overlap(
        self, old_words: list[str], new_words: list[str]
    ) -> int | None:
        """Find where old text's suffix matches new text's prefix.

        Returns the index in old_words where the overlap starts, or None.
        E.g. old=["the","quick","brown","fox"], new=["brown","fox","jumps"]
             returns 2 (old[2:] matches new[:2]).
        """
        for start in range(len(old_words)):
            suffix = old_words[start:]
            if len(suffix) > len(new_words):
                continue
            match = True
            for i, old_w in enumerate(suffix):
                if i >= len(new_words):
                    match = False
                    break
                if old_w != new_words[i]:
                    match = False
                    break
            if match and len(suffix) >= 1:
                return start
        return None

    def _handle_growth(self, new_text: str, now: float) -> None:
        """Situation A: text grew (words appended)."""
        if self._last_update_time is not None:
            interval_ms = (now - self._last_update_time) * 1000
            self._growth_intervals.append(interval_ms)
            avg = sum(self._growth_intervals) / len(self._growth_intervals)
            self._adaptive_timeout_ms = max(MIN_ADAPTIVE_TIMEOUT_MS, avg * TIMEOUT_MULTIPLIER)

        # Extract new words using string-level comparison (robust to OCR jitter).
        clean_old = _clean(self._last_visible_text)
        clean_new = _clean(new_text)

        appended: list[str] = []
        if clean_old and clean_old in clean_new:
            tail = clean_new[len(clean_old):].strip()
            if tail:
                appended = tail.split()
        else:
            # Fallback: word-count difference
            new_words = new_text.split()
            old_word_count = len(self._last_visible_text.split())
            appended = new_words[old_word_count:]

        if appended:
            self._buffer.extend(appended)

        self._last_update_time = now
        self._last_visible_text = new_text

    def _handle_shift(self, new_text: str, now: float) -> str | None:
        """Situation B: text shifted (beginning disappeared, tail still visible)."""
        old_words_raw = self._last_visible_text.split()
        new_words_raw = new_text.split()

        old_extracted = _extract_words(self._last_visible_text)
        new_extracted = _extract_words(new_text)
        shift_start = self._find_shift_overlap(old_extracted, new_extracted)

        if shift_start is None:
            return self._handle_replacement(new_text)

        # Find how many raw words disappeared from the top.
        # Map old_extracted[shift_start] back to raw word index.
        overlap_word = old_extracted[shift_start]
        raw_shift_idx = None
        for i, w in enumerate(old_words_raw):
            cleaned = re.sub(r"^[^\w]+|[^\w]+$", "", w).lower()
            if cleaned == overlap_word:
                raw_shift_idx = i
                break

        if raw_shift_idx is None:
            return self._handle_replacement(new_text)

        # Words that disappeared from old visible text
        disappeared_count = raw_shift_idx

        # Speak from cursor to cursor + disappeared_count
        speak_end = min(self._cursor + disappeared_count, len(self._buffer))
        to_speak = self._buffer[self._cursor:speak_end]
        self._cursor = speak_end

        # Append new words that appeared after the overlap region
        overlap_length = len(old_extracted) - shift_start
        if len(new_extracted) > overlap_length:
            # Find raw index of last overlap word in new_words
            last_overlap_word = old_extracted[-1] if old_extracted else None
            raw_overlap_end = None
            if last_overlap_word:
                for i in range(len(new_words_raw) - 1, -1, -1):
                    cleaned = re.sub(r"^[^\w]+|[^\w]+$", "", new_words_raw[i]).lower()
                    if cleaned == last_overlap_word:
                        raw_overlap_end = i + 1
                        break
            if raw_overlap_end is not None and raw_overlap_end < len(new_words_raw):
                self._buffer.extend(new_words_raw[raw_overlap_end:])

        # Update interval tracking
        if self._last_update_time is not None:
            interval_ms = (now - self._last_update_time) * 1000
            self._growth_intervals.append(interval_ms)
            avg = sum(self._growth_intervals) / len(self._growth_intervals)
            self._adaptive_timeout_ms = max(MIN_ADAPTIVE_TIMEOUT_MS, avg * TIMEOUT_MULTIPLIER)

        self._last_update_time = now
        self._last_visible_text = new_text

        if to_speak:
            result = " ".join(to_speak)
            self._record_spoken(result)
            return result
        return None

    def _handle_replacement(self, new_text: str) -> str | None:
        """Situation C: text completely different (old subtitle ended)."""
        result = self._flush_growing_buffer()
        self._start_new_growing_cycle(new_text)
        return result

    def _start_new_growing_cycle(self, text: str) -> None:
        """Initialize a new growing subtitles cycle.

        If the text overlaps with what was already spoken, advances
        the cursor past the spoken prefix to avoid repetition.
        """
        self._buffer = text.split()
        self._cursor = 0
        self._growth_intervals.clear()
        self._adaptive_timeout_ms = DEFAULT_GROWTH_TIMEOUT_MS
        self._last_update_time = time.monotonic()
        self._last_visible_text = text
        self._empty_frames = 0

        # Skip already-spoken prefix to avoid repetition.
        if self._last_spoken:
            clean_spoken = _clean(self._last_spoken)
            clean_text = _clean(text)
            if clean_spoken and clean_text:
                if clean_text == clean_spoken:
                    # Exact same text — everything already spoken
                    self._cursor = len(self._buffer)
                elif clean_text.startswith(clean_spoken):
                    # New text contains spoken text as prefix — skip it
                    spoken_word_count = len(self._last_spoken.split())
                    self._cursor = min(spoken_word_count, len(self._buffer))

    def _flush_growing_buffer(self) -> str | None:
        """Speak all unspoken words from the buffer, then reset."""
        to_speak = self._buffer[self._cursor:]
        self._buffer = []
        self._cursor = 0
        self._growth_intervals.clear()
        self._adaptive_timeout_ms = DEFAULT_GROWTH_TIMEOUT_MS
        self._last_update_time = None
        self._last_visible_text = ""
        self._empty_frames = 0

        if to_speak:
            result = " ".join(to_speak)
            self._record_spoken(result)
            return result
        return None
