"""OCR worker thread: captures screen region, runs OCR, emits recognized text.

Supports pluggable OCR engines (Tesseract, EasyOCR) and handles:
- Screen capture via mss
- Image preprocessing (text isolation for Tesseract, HDR enhancement for EasyOCR)
- OCR garbage filtering (removes non-text noise before emitting)
- Upscaling for small capture regions

The worker emits raw OCR text (after garbage filtering). Text deduplication
and change detection are handled by TextDiffer in the app layer.
"""

from __future__ import annotations

import logging
import os
import re
import time

import numpy as np
import pytesseract
from mss import mss
from PIL import Image
from PyQt6.QtCore import QThread, pyqtSignal

from capture_region_reader.text_isolator import isolate_text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cyrillic ↔ Latin look-alike tables (for bilingual OCR repair)
# ---------------------------------------------------------------------------

# Full transliteration table: Cyrillic look-alike → Latin equivalent.
# Used by _transliterate_garbled() to convert garbled words back to Latin.
# This is intentionally broad — includes all visually similar pairs.
_CYR_TO_LAT: dict[str, str] = {
    # Uppercase Cyrillic → Latin
    "\u0410": "A",  # А → A
    "\u0412": "B",  # В → B (visually B/V)
    "\u0421": "C",  # С → C
    "\u0415": "E",  # Е → E
    "\u041d": "H",  # Н → H
    "\u041a": "K",  # К → K
    "\u041c": "M",  # М → M
    "\u041e": "O",  # О → O
    "\u0420": "P",  # Р → P
    "\u0422": "T",  # Т → T
    "\u0425": "X",  # Х → X
    "\u0423": "Y",  # У → Y (visually similar)
    # Lowercase Cyrillic → Latin
    "\u0430": "a",  # а → a
    "\u0435": "e",  # е → e
    "\u043e": "o",  # о → o
    "\u0440": "p",  # р → p
    "\u0441": "c",  # с → c
    "\u0443": "y",  # у → y
    "\u0445": "x",  # х → x
    "\u043a": "k",  # к → k
    "\u043d": "n",  # н → n (visually H/n)
    "\u0432": "v",  # в → v
    "\u043c": "m",  # м → m
    "\u0442": "t",  # т → t
    "\u0438": "i",  # и → i (very common confusion)
    "\u0439": "i",  # й → i
}

# STRICT subset for garbled detection: only characters that are
# PIXEL-IDENTICAL between Cyrillic and Latin glyphs in most fonts.
# These are the chars where Tesseract genuinely can't tell the difference.
# Crucially excludes м/m, к/k, и/i, н/n, в/v, т/t — these look different
# in most fonts and are extremely common in normal Russian words.
_IDENTICAL_CYR_CHARS = frozenset({
    # Uppercase: А=A, В=B, С=C, Е=E, К=K, М=M, Н=H, О=O, Р=P, Т=T, Х=X
    "\u0410", "\u0421", "\u0415", "\u041e", "\u0420", "\u0425", "\u0422",
    # Lowercase: а=a, с=c, е=e, о=o, р=p, х=x, у=y
    "\u0430", "\u0441", "\u0435", "\u043e", "\u0440", "\u0445", "\u0443",
})

# Common digit ↔ letter confusions in garbled OCR output.
_DIGIT_TO_LETTER: dict[str, str] = {
    "0": "O", "1": "l", "3": "E", "5": "S", "6": "G", "8": "B",
}

# OCR noise symbols that shouldn't appear inside real words.
_NOISE_CHARS = frozenset("\\|/{}[]$&")


def _is_garbled_word(word: str) -> bool:
    """Detect if a word is likely a Latin word garbled by Cyrillic OCR.

    Uses multiple heuristics, scored as a weighted combination:
    - Contains OCR noise symbols (\\, |, $, etc.) inside the word
    - Mixes Cyrillic and Latin characters within the same word
    - Contains digits interleaved with Cyrillic letters
    - Word is made ENTIRELY of pixel-identical look-alike characters

    Returns True if the word is likely garbled English.
    """
    if len(word) < 2:
        return False

    # Quick check: OCR noise symbols inside the word → definitely garbled
    if any(c in _NOISE_CHARS for c in word):
        return True

    alpha_chars = [c for c in word if c.isalpha()]
    cyr_chars = [c for c in alpha_chars if "\u0400" <= c <= "\u052f"]
    lat_chars = [c for c in alpha_chars if ("A" <= c <= "Z") or ("a" <= c <= "z")]
    digits = [c for c in word if c.isdigit()]

    # Mixed scripts within one word → always garbled
    if cyr_chars and lat_chars:
        return True

    # Digits mixed with Cyrillic letters → garbled
    # Real Russian subtitle text never has digits directly adjacent to Cyrillic
    # within a single word. But Tesseract garbling often produces digit-letter
    # mixtures like "01зпеу" (from "Disney") where some Latin chars are read
    # as digits (D→0, i→1).
    # Require at least 2 Cyrillic chars to avoid false positives on isolated
    # cases like "3а" (rare but possible in informal text).
    if digits and len(cyr_chars) >= 2:
        return True

    # Digits interleaved with Latin letters (not at start/end) → garbled
    if digits and lat_chars:
        positions = [(i, "d" if c.isdigit() else "a") for i, c in enumerate(word)
                     if c.isdigit() or c.isalpha()]
        if len(positions) >= 3:
            # Check for digit-alpha-digit or alpha-digit-alpha patterns
            for k in range(1, len(positions) - 1):
                if positions[k][1] != positions[k-1][1] and positions[k][1] != positions[k+1][1]:
                    return True

    # If no Cyrillic at all, not garbled (already Latin)
    if not cyr_chars:
        return False

    # If word has < 2 alpha chars, not enough to judge
    if len(alpha_chars) < 2:
        return False

    # Two-tier check for all-look-alike words:
    # Tier 1 (strict): 100% pixel-identical chars → definitely garbled
    identical_count = sum(1 for c in cyr_chars if c in _IDENTICAL_CYR_CHARS)
    if identical_count == len(cyr_chars) and len(cyr_chars) >= 3:
        return True

    # Tier 2 (relaxed): >= 70% pixel-identical AND the rest are from the
    # broader look-alike table. Catches words like "РОВЕРТ" where В is
    # not pixel-identical but is still a common Tesseract substitution.
    # Only triggers for words with 4+ Cyrillic chars to avoid false positives.
    if len(cyr_chars) >= 4 and identical_count >= len(cyr_chars) * 0.7:
        broad_count = sum(1 for c in cyr_chars if c in _CYR_TO_LAT)
        if broad_count == len(cyr_chars):
            return True

    return False


def _transliterate_garbled(word: str) -> str:
    """Convert a garbled Cyrillic word back to Latin by replacing look-alikes.

    Also fixes common digit→letter confusions.
    """
    result = []
    for c in word:
        if c in _CYR_TO_LAT:
            result.append(_CYR_TO_LAT[c])
        elif c in _DIGIT_TO_LETTER:
            result.append(_DIGIT_TO_LETTER[c])
        else:
            result.append(c)
    return "".join(result)


# ---------------------------------------------------------------------------
# Common Russian short words — prepositions, conjunctions, particles.
# These are very short words that should NEVER be replaced by English.
# Used by _is_likely_garbled_russian() to protect real Russian words.
# ---------------------------------------------------------------------------
_COMMON_RUS_SHORT: frozenset[str] = frozenset({
    # 1-char prepositions / particles / conjunctions
    "а", "в", "и", "к", "о", "с", "у", "я",
    # 2-char
    "бы", "вы", "да", "до", "ее", "ей", "ем",
    "за", "из", "их", "ли", "мы", "на", "не", "ни", "но",
    "он", "от", "по", "та", "те", "то", "ту", "ты", "уж", "же",
    # 3-char common words
    "без", "вам", "вас", "вот", "все", "вся", "где", "два",
    "для", "его", "ему", "еще", "ещё", "ими", "как", "кто", "мне",
    "мой", "нам", "нас", "наш", "нет",
    "них", "она", "они", "оно", "оба", "при", "про", "раз",
    "сам", "так", "там", "тем", "тех", "тот", "три",
    "тут", "уже", "что", "чем", "чей", "это", "эти", "эта",
})

# ---------------------------------------------------------------------------
# Forward transliteration table: Cyrillic → Latin phonetic representation.
# Used to check if an eng-pass word is just Tesseract's phonetic reading of
# a real Russian word (e.g., "Сегодня" → "Segodnya"). This is the OPPOSITE
# direction from _CYR_TO_LAT (which maps look-alikes for garble detection).
# ---------------------------------------------------------------------------
_RUS_TO_PHONETIC: dict[str, str] = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e",
    "ё": "yo", "ж": "zh", "з": "z", "и": "i", "й": "y",
    "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p",
    "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "kh",
    "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch", "ъ": "", "ы": "y",
    "ь": "", "э": "e", "ю": "yu", "я": "ya",
    # Uppercase
    "А": "A", "Б": "B", "В": "V", "Г": "G", "Д": "D", "Е": "E",
    "Ё": "Yo", "Ж": "Zh", "З": "Z", "И": "I", "Й": "Y",
    "К": "K", "Л": "L", "М": "M", "Н": "N", "О": "O", "П": "P",
    "Р": "R", "С": "S", "Т": "T", "У": "U", "Ф": "F", "Х": "Kh",
    "Ц": "Ts", "Ч": "Ch", "Ш": "Sh", "Щ": "Shch", "Ъ": "", "Ы": "Y",
    "Ь": "", "Э": "E", "Ю": "Yu", "Я": "Ya",
}


def _phonetic_transliterate(rus_word: str) -> str:
    """Convert a Russian word to its phonetic Latin transliteration.

    Uses standard Russian-to-Latin transliteration rules. This is used to
    check if an eng-pass word is just Tesseract reading a Russian word
    phonetically (e.g., "Сегодня" -> "Segodnya").
    """
    result: list[str] = []
    for c in rus_word:
        if c in _RUS_TO_PHONETIC:
            result.append(_RUS_TO_PHONETIC[c])
        else:
            result.append(c)
    return "".join(result)


def _is_phonetic_match(rus_word: str, eng_word: str) -> bool:
    """Check if eng_word is a phonetic transliteration of rus_word.

    Returns True if the eng_word closely matches the phonetic transliteration
    of the Russian word (>= 60% character overlap). This indicates the eng-pass
    was reading a real Russian word, not a genuine English word.

    Examples:
        "Сегодня" -> "Segodnya" (phonetic: "Segodnya") -> overlap 100% -> True
        "будем"   -> "budem"    (phonetic: "budem")    -> overlap 100% -> True
        "гатемогКк" -> "framework" (phonetic: "gatemogKk") -> overlap ~20% -> False
    """
    phonetic = _phonetic_transliterate(rus_word).lower()
    eng_lower = eng_word.lower()

    if not phonetic or not eng_lower:
        return False

    # Use longest common subsequence ratio for flexible matching.
    # Tesseract's eng-pass doesn't always produce perfect transliterations
    # (may drop soft signs, merge vowels, etc.), so we use LCS instead
    # of exact char-by-char comparison.
    # For performance (O(n*m)), this is fine since words are short (< 30 chars).
    m, n = len(phonetic), len(eng_lower)
    # Quick length check: if lengths differ by more than 50%, not a match
    if max(m, n) > 2 * min(m, n):
        return False

    # LCS via two-row DP
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if phonetic[i - 1] == eng_lower[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)

    lcs_len = prev[n]
    # Overlap ratio against the longer string.
    # Threshold 0.7 is calibrated to:
    # - Accept imperfect transliterations ("budem"/"bydem"=0.80, "novyy"/"novyi"=0.80)
    # - Reject garbled words that coincidentally share letters
    #   ("lartor"/"laptop"=0.67, "gatemogkk"/"framework"=0.44)
    overlap = lcs_len / max(m, n)
    return overlap >= 0.7


def _is_clean_latin_word(word: str) -> bool:
    """Check if a word looks like a properly-recognized English word.

    A "clean" Latin word is one that Tesseract's eng-pass read correctly:
    - At least 2 characters long
    - At least 80% Latin alphabetic characters (allows apostrophes, hyphens)
    - No OCR noise symbols
    - No Cyrillic characters at all
    """
    if len(word) < 2:
        return False

    # No noise chars allowed
    if any(c in _NOISE_CHARS for c in word):
        return False

    alpha_count = 0
    lat_count = 0
    cyr_count = 0
    for c in word:
        if c.isalpha():
            alpha_count += 1
            if ("A" <= c <= "Z") or ("a" <= c <= "z"):
                lat_count += 1
            elif "\u0400" <= c <= "\u052f":
                cyr_count += 1

    # Must have no Cyrillic at all
    if cyr_count > 0:
        return False

    # Must have at least 2 Latin alpha chars
    if lat_count < 2:
        return False

    # At least 80% of all characters must be Latin alpha
    # (allows digits, apostrophes, hyphens in words like "don't", "e-mail")
    if lat_count / max(len(word), 1) < 0.8:
        return False

    return True


def _is_all_cyrillic(word: str) -> bool:
    """Check if all alphabetic characters in a word are Cyrillic."""
    alpha_chars = [c for c in word if c.isalpha()]
    if not alpha_chars:
        return False
    return all("\u0400" <= c <= "\u052f" for c in alpha_chars)


def _unique_russian_ratio(word: str) -> float:
    """Calculate the ratio of uniquely-Russian characters in a Cyrillic word.

    Characters in _CYR_TO_LAT are "look-alike" chars that Tesseract commonly
    substitutes when garbling Latin text into Cyrillic. Characters NOT in
    this table (б, г, д, ж, з, л, п, ф, ц, ч, ш, щ, etc.) are "uniquely
    Russian" — they look nothing like any Latin character.

    Real Russian words typically have 20-50% uniquely Russian characters.
    Garbled English words tend to have 0-22% because Tesseract's substitution
    table favors look-alike characters.

    A ratio >= 0.3 is strong evidence that the word is real Russian.
    """
    cyr_chars = [c for c in word if "\u0400" <= c <= "\u052f"]
    if not cyr_chars:
        return 0.0
    unique = sum(1 for c in cyr_chars if c not in _CYR_TO_LAT)
    return unique / len(cyr_chars)


def _is_likely_garbled_russian(rus_word: str, eng_word: str) -> bool:
    """Determine if a Cyrillic word is likely a garbled version of an English word.

    This is the key heuristic for the positional comparison approach.
    Given a word from the rus-pass and the corresponding word from the eng-pass,
    decides if the rus-pass word should be replaced by the eng-pass word.

    Returns True only when ALL of these conditions are met:
    1. The rus-word is all-Cyrillic (no mixed script)
    2. The rus-word is not a common short Russian word (prepositions, etc.)
    3. The rus-word has >= 4 alpha chars (short words have too many false positives)
    4. Lengths are similar (garbling preserves ~char count, ratio 0.6-1.5)
    5. The rus-word does NOT have a high ratio of uniquely-Russian characters
       (chars NOT in the look-alike table, like б, г, д, ж, ш, etc.)
    6. The eng-word is NOT a phonetic transliteration of the rus-word

    Checks (5) and (6) are redundant-by-design: either one can protect a real
    Russian word. Check (5) catches cases where the eng-pass produces a
    completely unrelated word (e.g., "Available" for "Доступны"). Check (6)
    catches cases where the eng-pass produces a phonetic transliteration
    (e.g., "Segodnya" for "Сегодня").
    """
    # The rus-word must be all Cyrillic (no Latin chars mixed in)
    if not _is_all_cyrillic(rus_word):
        return False

    # Protect common Russian short words — these are never garbled English
    if rus_word.lower() in _COMMON_RUS_SHORT:
        return False

    rus_alpha_len = sum(1 for c in rus_word if c.isalpha())
    eng_alpha_len = sum(1 for c in eng_word if c.isalpha())

    # Very short words (<=3 alpha chars) are too risky — too many real Russian
    # words are this short (мир, дом, час, раз, тест, etc.).
    if rus_alpha_len <= 3:
        return False

    # Length check: garbled substitution is roughly char-by-char, so lengths
    # should be similar. If lengths are very different, the eng-word is not
    # a plausible source for the garbled word.
    if eng_alpha_len == 0:
        return False

    length_ratio = rus_alpha_len / eng_alpha_len
    if not (0.6 <= length_ratio <= 1.5):
        return False

    # HIGH-CONFIDENCE PROTECTION: if the word has >= 30% uniquely Russian
    # characters (those NOT in the look-alike table), it is almost certainly
    # a real Russian word, not garbled English.
    # Real Russian: "Доступны" (0.38), "Сегодня" (0.43), "будем" (0.40)
    # Garbled English: "гатемогКк" (0.22), "лартор" (0.17)
    if _unique_russian_ratio(rus_word) >= 0.3:
        return False

    # PHONETIC CHECK: is the eng-word a phonetic transliteration of the
    # rus-word? If yes, the rus-word is real Russian and the eng-pass was
    # just reading it phonetically. Do NOT replace.
    #
    # This catches words with LOW unique ratio (e.g., "тест"=0.0, "мир"=0.0)
    # where the unique ratio check can't help, but the eng-pass still
    # produces a recognizable transliteration.
    #
    # Examples:
    #   "Сегодня" -> phonetic "Segodnya" matches eng "Segodnya" -> keep Russian
    #   "будем"   -> phonetic "budem" matches eng "budem" -> keep Russian
    if _is_phonetic_match(rus_word, eng_word):
        return False

    # REVERSE TRANSLITERATION CHECK (positive confirmation of garbling):
    # Convert Cyrillic look-alike chars back to Latin and compare with eng-word.
    # If there's significant overlap (>= 0.4 LCS ratio), the rus-word is
    # likely a garbled version of the eng-word because the look-alike chars
    # map back to matching Latin letters.
    #
    # This check is essential because without it, we'd default to "garbled"
    # for any word that passes the unique ratio and phonetic checks, which
    # would incorrectly replace real Russian words like "Привет" with
    # unrelated English words like "Hello".
    #
    # Examples:
    #   "лартор" -> translit "лaptop" vs eng "laptop" -> ratio 0.83 -> garbled
    #   "гатемогКк" -> translit "гatemoгKk" vs eng "framework" -> ratio 0.44 -> garbled
    #   "Привет" -> translit "Пpivet" vs eng "Hello" -> ratio 0.17 -> NOT garbled
    #   "Скачай" -> translit "Ckaчai" vs eng "Download" -> ratio 0.12 -> NOT garbled
    translit = _transliterate_garbled(rus_word).lower()
    eng_lower = eng_word.lower()
    if translit and eng_lower:
        m, n = len(translit), len(eng_lower)
        # LCS computation (same algorithm as _is_phonetic_match)
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if translit[i - 1] == eng_lower[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(prev[j], curr[j - 1])
            prev, curr = curr, [0] * (n + 1)
        lcs_len = prev[n]
        overlap = lcs_len / max(m, n)
        if overlap >= 0.4:
            return True

    # Default: if none of the positive garble indicators fired, keep the
    # Russian word. Being conservative here is correct — false negatives
    # (keeping a garbled word) are preferable to false positives (replacing
    # a real Russian word with unrelated English).
    return False


def _strip_word_punctuation(word: str) -> tuple[str, str, str]:
    """Strip leading and trailing punctuation from a word.

    Returns (leading_punct, core, trailing_punct).
    """
    leading = ""
    trailing = ""
    core = word
    while core and not core[0].isalnum():
        leading += core[0]
        core = core[1:]
    while core and not core[-1].isalnum():
        trailing = core[-1] + trailing
        core = core[:-1]
    return leading, core, trailing


def _merge_bilingual_ocr(primary: str, eng_text: str) -> str:
    """Merge bilingual (rus) OCR with English-only OCR using positional comparison.

    Two-phase strategy for each word position:

    Phase 1 (Quick detect): Use _is_garbled_word() to catch EASY cases — words
    with noise chars (\\, |, $), mixed Cyrillic+Latin scripts, or words made
    entirely of pixel-identical look-alike characters. These are unambiguously
    garbled and we replace them with the eng-pass word at the same position.

    Phase 2 (Positional comparison): For words that pass the quick check (i.e.,
    they look like normal Cyrillic words), compare with the eng-pass output
    at the same position. If the eng-pass produced a clean Latin word AND the
    rus-pass word shows signs of being garbled (similar length, not a common
    Russian word), prefer the eng-pass word.

    This handles the hard case where Tesseract garbles "framework" into
    "гатемогКк" — a word made entirely of normal Cyrillic characters that
    _is_garbled_word() cannot detect.
    """
    primary_lines = primary.split("\n")
    eng_lines = eng_text.split("\n")

    result_lines: list[str] = []

    for line_idx, p_line in enumerate(primary_lines):
        p_line = p_line.strip()
        if not p_line:
            continue

        # Get the corresponding eng-pass line. Lines may not align perfectly
        # (different line counts, different line splitting), but for subtitle
        # text there's usually a 1:1 correspondence.
        eng_line_words: list[str] = []
        if line_idx < len(eng_lines):
            eng_line_words = eng_lines[line_idx].strip().split()

        p_words = p_line.split()
        merged: list[str] = []

        for word_idx, pw in enumerate(p_words):
            leading, core, trailing = _strip_word_punctuation(pw)

            if not core:
                merged.append(pw)
                continue

            # --- Phase 1: Quick garble detection (easy cases) ---
            if _is_garbled_word(core):
                replacement = _find_positional_replacement(
                    core, word_idx, eng_line_words
                )
                if replacement:
                    # When replacing garbled words, strip noise chars from
                    # leading/trailing punctuation — they are OCR artifacts,
                    # not real punctuation. E.g., "Р\и$" → trailing "$"
                    # should not be kept when replacing with "Plus".
                    clean_leading = "".join(
                        c for c in leading if c not in _NOISE_CHARS
                    )
                    clean_trailing = "".join(
                        c for c in trailing if c not in _NOISE_CHARS
                    )
                    merged.append(clean_leading + replacement + clean_trailing)
                    logger.debug(
                        "Bilingual fix (garbled): '%s' -> '%s'", core, replacement
                    )
                else:
                    # _is_garbled_word fired but no good eng replacement found.
                    # Could be a false positive on a real Russian word.
                    # Keep original rather than producing garbage.
                    merged.append(pw)
                continue

            # --- Phase 2: Positional comparison (hard cases) ---
            # Only applies when we have a corresponding eng-pass word
            if word_idx < len(eng_line_words):
                eng_raw = eng_line_words[word_idx]
                _, eng_core, _ = _strip_word_punctuation(eng_raw)

                if eng_core and _is_clean_latin_word(eng_core):
                    if _is_likely_garbled_russian(core, eng_core):
                        merged.append(leading + eng_core + trailing)
                        logger.debug(
                            "Bilingual fix (positional): '%s' -> '%s'",
                            core, eng_core,
                        )
                        continue

            # No replacement needed — keep the original word
            merged.append(pw)

        result_lines.append(" ".join(merged))

    return "\n".join(result_lines)


def _find_positional_replacement(
    garbled_core: str,
    word_index: int,
    eng_line_words: list[str],
) -> str | None:
    """Find the best English replacement for a word detected as garbled.

    Called only when _is_garbled_word() has already confirmed the word is
    garbled (noise chars, mixed scripts, all-look-alike chars). This function
    finds the correct English word to substitute.

    Strategies in order:
    1. Position-based: same word index in eng line (if it's a clean Latin word)
    2. Transliteration match: convert garbled chars to Latin, search eng line
    3. Nearby search: find a clean Latin word nearby (±2 positions) with
       similar length to the garbled word
    """
    has_noise = any(c in _NOISE_CHARS for c in garbled_core)
    garbled_alpha_len = sum(1 for c in garbled_core if c.isalpha())

    # Strategy 1: Same position in eng line
    if word_index < len(eng_line_words):
        _, candidate, _ = _strip_word_punctuation(eng_line_words[word_index])
        if candidate and _is_clean_latin_word(candidate):
            # For noisy garbled words, be lenient on length (noise inflates length)
            if has_noise:
                if abs(len(candidate) - garbled_alpha_len) <= 4:
                    return candidate
            else:
                # Standard check: transliteration overlap or similar length
                if abs(len(candidate) - len(garbled_core)) <= 3:
                    return candidate

    # Strategy 2: Transliteration → find in eng line words
    clean_garbled = "".join(c for c in garbled_core if c not in _NOISE_CHARS)
    if clean_garbled:
        translit = _transliterate_garbled(clean_garbled).lower()
        for ew_raw in eng_line_words:
            _, ew, _ = _strip_word_punctuation(ew_raw)
            if ew and ew.lower() == translit:
                return ew

    # Strategy 3: Nearby search — find clean Latin word ±2 positions
    best: str | None = None
    best_score = 999
    search_lo = max(0, word_index - 2)
    search_hi = min(len(eng_line_words), word_index + 3)
    for k in range(search_lo, search_hi):
        _, candidate, _ = _strip_word_punctuation(eng_line_words[k])
        if not candidate or not _is_clean_latin_word(candidate):
            continue
        len_diff = abs(len(candidate) - garbled_alpha_len)
        pos_diff = abs(k - word_index)
        score = len_diff * 2 + pos_diff
        if score < best_score:
            best = candidate
            best_score = score

    return best


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum dimensions for Tesseract OCR accuracy.
# Small capture regions produce blurry characters after Otsu thresholding.
MIN_WIDTH = 600
MIN_HEIGHT = 100


# Debug: save captures to .tests/debug/ for inspection.
DEBUG_SAVE = os.environ.get("CRR_DEBUG", "0") == "1"
_DEBUG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", ".tests", "debug")


# ---------------------------------------------------------------------------
# OCR engine (Tesseract only)
# ---------------------------------------------------------------------------

_TESSERACT_CONFIG = "--psm 6 --oem 1"

# Languages that need a supplementary English pass
_NEEDS_ENG_PASS = {"rus", "eng+rus", "rus+eng"}


def _recognize(image: Image.Image, language: str) -> str:
    """Run Tesseract OCR on a PIL image.

    Uses PSM 6 (single uniform block of text) and OEM 1 (LSTM neural net).

    Dual-pass strategy for any language mode containing Russian:
    When lang includes "rus", Tesseract may garble English words embedded in
    Russian text (brand names, tech terms, etc.) into pseudo-Cyrillic garbage.
    We run a second pass with eng-only and merge at word level — garbled
    pseudo-Cyrillic words are replaced with their clean English counterparts.
    """
    primary = pytesseract.image_to_string(
        image, lang=language, config=_TESSERACT_CONFIG
    ).strip()

    # Dual-pass: run eng-only to fix garbled English words
    if language in _NEEDS_ENG_PASS and primary:
        eng_only = pytesseract.image_to_string(
            image, lang="eng", config=_TESSERACT_CONFIG
        ).strip()
        if eng_only:
            primary = _merge_bilingual_ocr(primary, eng_only)

    return primary


# ---------------------------------------------------------------------------
# Garbage filtering
# ---------------------------------------------------------------------------

# Minimum alphanumeric-to-total ratio for a line to be considered text.
_GARBAGE_ALNUM_RATIO = 0.4
# Minimum alphabetic characters per line.
_GARBAGE_MIN_ALPHA = 2
# Maximum ratio of mixed-script words (Cyrillic+Latin in same word).
# Raised from 0.3 to 0.7 because the dual-pass OCR merge can produce
# legitimate lines with both Cyrillic and Latin words side by side.
# This check now only catches lines where MOST words are garbled fragments.
_GARBAGE_MIXED_SCRIPT_RATIO = 0.7
# Maximum ratio of very short words (<=2 chars) in lines with >=3 words.
_GARBAGE_SHORT_WORD_RATIO = 0.6
# Minimum unique character ratio (catches "aaaaaaa" garbage).
_GARBAGE_MIN_UNIQUE_RATIO = 0.15
# Minimum characters to run unique ratio check.
_GARBAGE_MIN_CHARS_FOR_UNIQUE = 4
# Maximum total words for average-word-length check.
_GARBAGE_SHORT_RESULT_MAX_WORDS = 3
# Minimum average word length for short results.
_GARBAGE_SHORT_RESULT_MIN_AVG_LEN = 3


def _is_mixed_script_word(word: str) -> bool:
    """Check if a word mixes Cyrillic and Latin characters."""
    cyr = sum(1 for c in word if "\u0400" <= c <= "\u052f")
    lat = sum(1 for c in word if ("A" <= c <= "Z") or ("a" <= c <= "z"))
    return cyr > 0 and lat > 0 and (cyr + lat) >= 3


def filter_ocr_garbage(text: str) -> str:
    """Filter out OCR garbage lines.

    Applies multiple heuristics to detect and remove lines that are
    OCR noise rather than real text:
    1. Low alphanumeric ratio (mostly symbols/punctuation)
    2. Too few alphabetic characters
    3. Mixed Cyrillic/Latin within same word (always OCR error)
    4. Too many very short words (fragmented noise)
    5. Low character diversity (repeated characters)
    6. Final check: very short results with tiny average word length
    """
    lines = text.split("\n")
    good_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check 1: alphanumeric ratio
        alnum = sum(1 for c in line if c.isalnum())
        non_space = len(line.replace(" ", ""))
        if non_space == 0 or alnum / non_space < _GARBAGE_ALNUM_RATIO:
            continue

        # Check 2: minimum alphabetic characters
        alpha = sum(1 for c in line if c.isalpha())
        if alpha < _GARBAGE_MIN_ALPHA:
            continue

        # Check 3: mixed-script words
        words = line.split()
        if len(words) > 0:
            mixed_count = sum(1 for w in words if _is_mixed_script_word(w))
            if mixed_count / len(words) > _GARBAGE_MIXED_SCRIPT_RATIO:
                continue

        # Check 4: too many short words
        if len(words) >= 3:
            short_words = sum(1 for w in words if len(w) <= 2)
            if short_words / len(words) > _GARBAGE_SHORT_WORD_RATIO:
                continue

        # Check 5: character diversity
        alpha_chars = [c.lower() for c in line if c.isalpha()]
        if len(alpha_chars) >= _GARBAGE_MIN_CHARS_FOR_UNIQUE:
            unique_ratio = len(set(alpha_chars)) / len(alpha_chars)
            if unique_ratio < _GARBAGE_MIN_UNIQUE_RATIO:
                continue

        good_lines.append(line)

    result = "\n".join(good_lines)

    # Final check: very short results with tiny average word length
    # are likely fragmented noise.
    if result:
        all_words = result.split()
        if len(all_words) <= _GARBAGE_SHORT_RESULT_MAX_WORDS:
            avg_word_len = sum(len(w) for w in all_words) / len(all_words)
            if avg_word_len < _GARBAGE_SHORT_RESULT_MIN_AVG_LEN:
                return ""

    return result


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def _upscale(img: Image.Image) -> Image.Image:
    """Upscale image to minimum dimensions needed for Tesseract OCR accuracy.

    Small capture regions produce blurry characters. This ensures
    the image is at least MIN_WIDTH x MIN_HEIGHT, scaling by an integer
    factor between 2x and 4x.
    """
    w, h = img.size
    scale = 1
    if w < MIN_WIDTH:
        scale = max(scale, (MIN_WIDTH + w - 1) // w)
    if h < MIN_HEIGHT:
        scale = max(scale, (MIN_HEIGHT + h - 1) // h)
    scale = max(scale, 2)
    scale = min(scale, 4)
    if scale > 1:
        img = img.resize((w * scale, h * scale), Image.Resampling.LANCZOS)
    return img


# ---------------------------------------------------------------------------
# OCR Worker thread
# ---------------------------------------------------------------------------

class OcrWorker(QThread):
    """Background thread that captures screen region and runs OCR in a loop.

    Signals:
    - text_recognized(str): emitted when OCR produces non-garbage text
    - frame_captured(bytes, int, int): preview image (RGB bytes, width, height)
    - error_occurred(str): error message for UI display
    - engine_unavailable(str, str): emitted when requested engine can't be loaded
      (requested_engine_name, error_message). UI should revert to the current
      working engine and inform the user.
    """
    text_recognized = pyqtSignal(str)
    frame_captured = pyqtSignal(bytes, int, int)
    raw_frame_captured = pyqtSignal(bytes, int, int)  # original RGB (before isolation)
    error_occurred = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._region: tuple[int, int, int, int] | None = None
        self._language: str = "eng+rus"
        self._interval_ms: int = 500
        self._running: bool = False
        self._capture_count: int = 0

    def configure(
        self,
        region: tuple[int, int, int, int],
        language: str,
        interval_ms: int,
    ) -> None:
        """Configure capture region and OCR parameters.

        Called before start() or when settings change while running.
        """
        self._region = region
        self._language = language
        self._interval_ms = interval_ms
        self._capture_count = 0
        logger.info(
            "OCR configured: region=(%d,%d,%d,%d), lang=%s",
            *region, language,
        )

    def set_language(self, language: str) -> None:
        self._language = language

    def set_interval(self, interval_ms: int) -> None:
        self._interval_ms = interval_ms

    def run(self) -> None:
        """Main OCR loop: capture -> preprocess -> recognize -> emit."""
        self._running = True

        with mss() as sct:
            while self._running:
                if not self._region:
                    self.msleep(100)
                    continue

                left, top, width, height = self._region
                monitor = {
                    "left": left,
                    "top": top,
                    "width": width,
                    "height": height,
                }

                try:
                    self._process_frame(sct, monitor)
                except Exception as e:
                    logger.error("OCR frame error: %s", e, exc_info=True)
                    self.error_occurred.emit(str(e))

                self.msleep(self._interval_ms)

    def _process_frame(self, sct, monitor: dict) -> None:
        """Process a single capture frame through the OCR pipeline."""
        screenshot = sct.grab(monitor)
        img_array = np.array(screenshot, dtype=np.uint8)

        # BGRA -> RGB
        raw_rgb = img_array[:, :, :3][:, :, ::-1].copy()

        # Always emit the raw frame (used by eyedropper for color picking)
        raw_h, raw_w = raw_rgb.shape[:2]
        self.raw_frame_captured.emit(raw_rgb.tobytes(), raw_w, raw_h)

        # Text isolation: detect subtitle region, binarize, clean artifacts
        isolated = isolate_text(raw_rgb)

        if isolated is None:
            logger.warning(
                "isolate_text returned None for frame %d (input shape=%s)",
                self._capture_count, raw_rgb.shape,
            )
            # Debug: save the input that caused None
            if DEBUG_SAVE and self._capture_count < 20:
                os.makedirs(_DEBUG_DIR, exist_ok=True)
                Image.fromarray(raw_rgb).save(
                    os.path.join(_DEBUG_DIR, f"crr_none_input_{self._capture_count}.png")
                )
            p_h, p_w = raw_rgb.shape[:2]
            self.frame_captured.emit(raw_rgb.tobytes(), p_w, p_h)
            # Emit empty string so TextDiffer knows text disappeared
            self.text_recognized.emit("")
            self.msleep(self._interval_ms)
            return

        ocr_img = _upscale(Image.fromarray(isolated))

        # Send the processed image to preview
        preview_rgb = np.array(ocr_img)
        p_h, p_w = preview_rgb.shape[:2]
        self.frame_captured.emit(preview_rgb.tobytes(), p_w, p_h)

        # Debug saves
        if DEBUG_SAVE and self._capture_count < 20:
            os.makedirs(_DEBUG_DIR, exist_ok=True)
            Image.fromarray(raw_rgb).save(
                os.path.join(_DEBUG_DIR, f"crr_raw_{self._capture_count}.png")
            )
            ocr_img.save(
                os.path.join(_DEBUG_DIR, f"crr_isolated_{self._capture_count}.png")
            )
            logger.info(
                "Debug frame %d saved: raw=%s, isolated=%s",
                self._capture_count, raw_rgb.shape, ocr_img.size,
            )

        self._capture_count += 1

        # Run OCR
        t0 = time.monotonic()
        text = _recognize(ocr_img, self._language)
        ocr_ms = int((time.monotonic() - t0) * 1000)

        logger.debug(
            "OCR (%dms) raw: %s",
            ocr_ms, repr(text[:200] if text else ""),
        )

        # Filter garbage and emit
        if text:
            filtered = filter_ocr_garbage(text)
            if filtered != text:
                logger.debug("OCR after filter: %s", repr(filtered[:200] if filtered else "<empty>"))
            text = filtered

        # Always emit (even empty) so TextDiffer can track text disappearance.
        self.text_recognized.emit(text)

    def stop(self) -> None:
        """Stop the OCR loop and wait for thread to finish."""
        self._running = False
        self.wait(3000)
