"""Clean and filter OCR output for TTS consumption.

Two main operations:
1. clean_for_tts() -- normalize and clean OCR text so TTS reads naturally
2. filter_by_language() -- remove lines that don't match expected language

Patterns inspired by Translumo's TextValidityPredictor preprocessing:
- Fullwidth-to-halfwidth character normalization
- Unicode-aware alphabet validation
- Structured replacement maps organized by category
"""

from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------------
# Character replacement maps (organized by category)
# ---------------------------------------------------------------------------

# Fullwidth forms -> ASCII equivalents (from Translumo _replacers).
# These appear in CJK contexts or when OCR misreads standard characters.
_FULLWIDTH_REPLACEMENTS: dict[str, str] = {
    "\uff10": "0", "\uff11": "1", "\uff12": "2", "\uff13": "3", "\uff14": "4",
    "\uff15": "5", "\uff16": "6", "\uff17": "7", "\uff18": "8", "\uff19": "9",
    "\uff0c": ",", "\uff01": "!", "\uff1f": "?", "\uff1b": ";", "\uff1a": ":",
    "\uff08": "(", "\uff09": ")", "\uff3b": "[", "\uff3d": "]",
    "\uff0e": ". ",
}

# CJK-specific punctuation -> ASCII equivalents.
_CJK_PUNCTUATION: dict[str, str] = {
    "\u3001": ",",    # ideographic comma
    "\u3010": "[",    # left tortoise shell bracket
    "\u3011": "]",    # right tortoise shell bracket
    "\u2026": "...",  # horizontal ellipsis
    "\u2e3a": "-",    # two-em dash
    "\u266a": "",     # eighth note (music symbol)
    "\u301f": '"',    # low double prime quotation mark
    "\u301d": '"',    # reversed double prime quotation mark
    "\u30fb\u30fb\u30fb": "...",  # katakana middle dot x3
    "\u2025": "..",   # two dot leader
}

# Typographic punctuation -> simple ASCII for TTS.
# TTS engines read many of these literally (e.g. "em dash" in Russian).
_TYPOGRAPHIC_REPLACEMENTS: dict[str, str] = {
    "\u2014": ",",    # em dash -> pause
    "\u2013": ",",    # en dash -> pause
    "\u2015": ",",    # horizontal bar -> pause
    "\u00ab": "",     # left guillemet
    "\u00bb": "",     # right guillemet
    "\u201e": "",     # double low quotation
    "\u201c": "",     # left double quotation
    "\u201d": "",     # right double quotation
    "\u2018": "",     # left single quotation
    "\u2019": "'",    # right single quotation -> apostrophe
    "\u2026": ".",    # ellipsis -> period
    "\u2032": "'",    # prime -> apostrophe
    "`": "'",         # backtick -> apostrophe
}

# Symbols that TTS would read literally (e.g. "backslash", "dollar sign").
_SYMBOL_REPLACEMENTS: dict[str, str] = {
    "\\": " ",   "/": " ",   "|": " ",   "~": " ",   "_": " ",
    "$": "",     "#": "",    "@": "",    "^": "",    "*": "",
    "+": "",     "=": "",    "<": "",    ">": "",
    "{": "",     "}": "",    "[": "",    "]": "",
    "\u00a9": "",  # copyright
    "\u00ae": "",  # registered
    "\u2122": "",  # trademark
    "\u00b0": " gradusov ",  # degree sign (Russian TTS context)
    "\u2116": "nomer ",      # numero sign
    "\u00a7": "",  # section sign
    "\u00b6": "",  # pilcrow
    "\u2020": "",  # dagger
    "\u2021": "",  # double dagger
    "\u2022": "",  # bullet
    "\u00b7": "",  # middle dot
    "\u2039": "",  # single left guillemet
    "\u203a": "",  # single right guillemet
}

# Ampersand is context-dependent; default to Russian "i".
_CONTEXT_REPLACEMENTS: dict[str, str] = {
    "&": " i ",
}

# Build combined replacement map (order matters: fullwidth first).
_ALL_REPLACEMENTS: dict[str, str] = {}
_ALL_REPLACEMENTS.update(_FULLWIDTH_REPLACEMENTS)
_ALL_REPLACEMENTS.update(_CJK_PUNCTUATION)
_ALL_REPLACEMENTS.update(_TYPOGRAPHIC_REPLACEMENTS)
_ALL_REPLACEMENTS.update(_SYMBOL_REPLACEMENTS)
_ALL_REPLACEMENTS.update(_CONTEXT_REPLACEMENTS)

# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

# Standalone single non-word characters surrounded by whitespace.
# Exceptions: common Russian single-letter words.
_RE_STANDALONE_CHAR = re.compile(
    r"(?<!\S)[^yaioavksuYAIOAVKSU\u044f\u0438\u043e\u0430\u0432\u043a\u0441\u0443"
    r"\u042f\u0418\u041e\u0410\u0412\u041a\u0421\u0423\s](?!\S)"
)

# Trailing standalone character at end of string.
_RE_TRAILING_STANDALONE = re.compile(r"\s+[^\s]$")

# Two or more consecutive non-alphanumeric, non-whitespace, non-basic-punctuation chars.
# Typical OCR garbage like "!@#$" or "}{><".
_RE_SYMBOL_SEQUENCE = re.compile(r"[^\w\s,.!?:;'\"-]{2,}")

# Multiple whitespace characters.
_RE_MULTI_SPACE = re.compile(r"\s{2,}")

# Leading/trailing dot sequences (from Translumo RegexStorage).
_RE_LEADING_DOTS = re.compile(r"^\.{3,}")
_RE_TRAILING_DOTS = re.compile(r"\.{3,}$")


# ---------------------------------------------------------------------------
# TTS text cleaning
# ---------------------------------------------------------------------------

def _normalize_characters(text: str) -> str:
    """Apply all character replacements in a single pass.

    Uses a translation table for single-char replacements and
    str.replace for multi-char sequences.
    """
    # Multi-char replacements first (before breaking into single chars)
    for old, new in _CJK_PUNCTUATION.items():
        if len(old) > 1 and old in text:
            text = text.replace(old, new)

    # Single-char replacements
    for old, new in _ALL_REPLACEMENTS.items():
        if len(old) == 1:
            text = text.replace(old, new)

    return text


def _preprocess_line(line: str) -> str:
    """Preprocess a single text line (inspired by Translumo PreProcessText).

    Steps:
    1. Normalize fullwidth/typographic characters
    2. Strip leading/trailing dashes (OCR artifact at subtitle edges)
    3. Remove wrapping quotes (OCR sometimes adds them)
    4. Remove leading/trailing dot sequences
    """
    line = _normalize_characters(line)
    line = line.strip("- ")

    # Remove wrapping quotes: 'text' or "text" where first == last
    if len(line) > 1 and line[0] == line[-1] and line[0] in ("'", '"'):
        line = line[1:-1]

    # Strip leading/trailing ellipsis-like dots
    line = _RE_LEADING_DOTS.sub("", line)
    line = _RE_TRAILING_DOTS.sub("", line)

    return line.strip()


def clean_for_tts(text: str) -> str:
    """Clean OCR output so TTS reads naturally without spelling out symbols.

    Removes or replaces characters that edge-tts would read literally
    (e.g. "naklonnaya cherta", "myagkiy znak", "bolshe chem").
    """
    if not text:
        return ""

    # Preprocess each line individually (handles quotes, dots, dashes per-line)
    lines = text.split("\n")
    processed_lines = []
    for line in lines:
        processed = _preprocess_line(line)
        if processed:
            processed_lines.append(processed)

    text = " ".join(processed_lines)

    # Remove standalone single characters that are OCR noise
    text = _RE_STANDALONE_CHAR.sub(" ", text)
    text = _RE_TRAILING_STANDALONE.sub("", text)

    # Remove sequences of random non-alphanumeric characters
    text = _RE_SYMBOL_SEQUENCE.sub(" ", text)

    # Clean up remaining quotes
    text = text.replace('"', "")
    text = text.replace("'", "")

    # Remove hard sign -- often OCR noise in subtitle context
    text = text.replace("\u044a", "")

    # Normalize whitespace
    text = _RE_MULTI_SPACE.sub(" ", text)

    # Remove leading/trailing punctuation clutter
    text = text.strip(" ,;:-.")

    # Filter out lines with fewer than 2 alphabetic characters
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        alpha_count = sum(1 for c in line if c.isalpha())
        if alpha_count >= 2:
            cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # Final whitespace cleanup
    text = _RE_MULTI_SPACE.sub(" ", text).strip()

    return text


# ---------------------------------------------------------------------------
# Language filtering
# ---------------------------------------------------------------------------

# Cyrillic letters that look identical to Latin letters.
# When Tesseract misreads English text as Cyrillic, the result is
# "words" made entirely of these ambiguous characters.
# E=E, T=T, O=O, P=R, A=A, H=N, C=S, B=V, M=M, K=K, X=X
_AMBIGUOUS_CYRILLIC = frozenset(
    "\u0415\u0422\u041e\u0420\u0410\u041d\u0421\u0412\u041c\u041a\u0425"  # uppercase
    "\u0435\u0442\u043e\u0440\u0430\u0441\u043d\u0432\u043c\u043a\u0445"  # lowercase
)


def _is_cyrillic(c: str) -> bool:
    """Check if a character is in any Cyrillic Unicode block."""
    # Cyrillic block: U+0400..U+04FF
    # Cyrillic Supplement: U+0500..U+052F
    cp = ord(c)
    return 0x0400 <= cp <= 0x052F


def _is_latin(c: str) -> bool:
    """Check if a character is in the Basic Latin alphabet range."""
    return ("A" <= c <= "Z") or ("a" <= c <= "z")


def _is_fake_cyrillic_word(word: str) -> bool:
    """Check if a Cyrillic word is likely a misrecognized Latin word.

    When Tesseract reads English text like "ROBERT" with lang=rus or
    lang=eng+rus, it may produce "OREBRT" in Cyrillic -- every letter
    happens to be one that looks like a Latin letter.

    Returns True if the word is suspicious (>80% of Cyrillic letters
    are ambiguous look-alikes of Latin characters).
    """
    cyr_letters = [c for c in word if _is_cyrillic(c)]
    if len(cyr_letters) < 3:
        return False

    ambiguous = sum(1 for c in cyr_letters if c in _AMBIGUOUS_CYRILLIC)
    return ambiguous / len(cyr_letters) > 0.8


def _has_mixed_scripts(word: str) -> bool:
    """Check if a word mixes Cyrillic and Latin characters.

    Words with both scripts are almost always OCR errors --
    real bilingual text separates scripts at word boundaries.
    """
    cyr = sum(1 for c in word if _is_cyrillic(c))
    lat = sum(1 for c in word if _is_latin(c))
    total_alpha = cyr + lat
    # Only flag if both scripts are present and the word is long enough
    return cyr > 0 and lat > 0 and total_alpha >= 3


def filter_by_language(text: str, language: str) -> str:
    """Filter out lines that don't match the expected language.

    If language is "rus" -- remove lines with mostly Latin characters
    and lines that look like misrecognized English (fake Cyrillic).
    If language is "eng" -- remove lines with mostly Cyrillic characters.
    If language is "eng+rus" -- only remove lines with mixed-script words
    (which are always OCR errors).

    This helps filter OCR noise where random UI text or other language
    text gets picked up alongside subtitles.
    """
    if not text:
        return text

    lines = text.split("\n")
    filtered = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        cyrillic = sum(1 for c in stripped if _is_cyrillic(c))
        latin = sum(1 for c in stripped if _is_latin(c))
        total_alpha = cyrillic + latin

        if total_alpha == 0:
            continue

        cyr_ratio = cyrillic / total_alpha

        if language == "eng+rus":
            # In bilingual mode, only reject lines with mixed-script words.
            # These are always OCR errors (real text doesn't mix scripts
            # within a single word).
            words = stripped.split()
            if len(words) > 0:
                mixed_count = sum(1 for w in words if _has_mixed_scripts(w))
                mixed_ratio = mixed_count / len(words)
                if mixed_ratio > 0.3:
                    continue
            filtered.append(line)

        elif language == "rus":
            # Keep lines with >40% Cyrillic
            if cyr_ratio <= 0.4:
                continue

            # Check for fake Cyrillic words (Latin misrecognized as Cyrillic)
            words = stripped.split()
            real_words = 0
            fake_words = 0
            for word in words:
                word_cyr = sum(1 for c in word if _is_cyrillic(c))
                if word_cyr >= 3:
                    if _is_fake_cyrillic_word(word):
                        fake_words += 1
                    else:
                        real_words += 1

            total_checked = real_words + fake_words
            if total_checked > 0 and fake_words / total_checked >= 0.5:
                continue

            filtered.append(line)

        elif language == "eng":
            # Keep lines with less than 60% Cyrillic
            if cyr_ratio < 0.6:
                filtered.append(line)
        else:
            filtered.append(line)

    return "\n".join(filtered)
