from __future__ import annotations

import re


def clean_for_tts(text: str) -> str:
    """Clean OCR output so TTS reads naturally without spelling out symbols.

    Removes or replaces characters that edge-tts would read literally
    (e.g. "наклонная черта", "мягкий знак", "больше чем").
    """
    if not text:
        return ""

    # Replace common OCR artifacts and symbols that TTS reads literally
    replacements = {
        "—": ",",      # em dash → pause
        "–": ",",      # en dash → pause
        "―": ",",      # horizontal bar → pause
        "«": "",       # left guillemet
        "»": "",       # right guillemet
        "„": "",       # double low quotation
        """: "",       # left double quotation
        """: "",       # right double quotation
        "'": "",       # left single quotation
        "'": "",       # right single quotation
        "…": ".",      # ellipsis → period
        "\\": " ",     # backslash (OCR artifact)
        "/": " ",      # forward slash
        "|": " ",      # pipe
        "$": "",       # dollar sign (OCR artifact)
        "#": "",       # hash
        "@": "",       # at sign
        "&": " и ",    # ampersand → "и" for Russian context
        "~": " ",      # tilde
        "^": "",       # caret
        "*": "",       # asterisk
        "+": "",       # plus
        "=": "",       # equals
        "<": "",       # less than
        ">": "",       # greater than
        "{": "",       # braces
        "}": "",
        "[": "",       # brackets
        "]": "",
        "_": " ",      # underscore
        "©": "",       # copyright
        "®": "",       # registered
        "™": "",       # trademark
        "°": " градусов ",  # degree sign
        "№": "номер ",     # numero sign
        "§": "",       # section sign
        "¶": "",       # pilcrow
        "†": "",       # dagger
        "‡": "",       # double dagger
        "•": "",       # bullet
        "·": "",       # middle dot
        "‹": "",       # single guillemet
        "›": "",       # single guillemet
        "ъ": "",       # hard sign is often OCR noise in subtitles
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove standalone single characters that are OCR noise
    # e.g. "программу К шв 9" → "программу шв"
    # Single letter/digit surrounded by spaces (except common Russian words: я, и, о, а, в, к, с, у)
    text = re.sub(r"(?<!\S)[^яиоавксуЯИОАВКСУ\s](?!\S)", " ", text)
    # Remove trailing standalone characters
    text = re.sub(r"\s+[^\s]$", "", text)

    # Remove sequences of random characters typical of OCR errors
    # (2+ consecutive non-alphanumeric non-space chars)
    text = re.sub(r"[^\w\s,.!?:;'\"-]{2,}", " ", text)

    # Clean up quotes — keep simple ones
    text = text.replace('"', "")
    text = text.replace("'", "")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing punctuation clutter
    text = text.strip(" ,;:-.")

    # Remove lines that are too short (likely OCR noise)
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        # Keep lines with at least 2 alphabetic characters
        alpha_count = sum(1 for c in line if c.isalpha())
        if alpha_count >= 2:
            cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # Final whitespace cleanup
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Cyrillic letters that look identical to Latin letters.
# When Tesseract misreads English text as Cyrillic, the result is
# "words" made entirely of these ambiguous characters.
# Е=E, Т=T, О=O, Р=P, А=A, Н=H, С=C, В=B, М=M, К=K, Х=X
_AMBIGUOUS_CYRILLIC = set("ЕТОРАНСВМКХетораснвмкх")


def _is_fake_cyrillic_word(word: str) -> bool:
    """Check if a Cyrillic word is likely a misrecognized Latin word.

    When Tesseract reads English text like "ROBERT" with lang=rus or
    lang=eng+rus, it may produce "ОРЕБРТ" — pure Cyrillic, but every
    letter happens to be one that looks like a Latin letter.

    Returns True if the word is suspicious (all or almost all letters
    are ambiguous Cyrillic that look like Latin).
    """
    cyr_letters = [c for c in word if "\u0400" <= c <= "\u04FF"]
    if len(cyr_letters) < 3:
        return False

    ambiguous = sum(1 for c in cyr_letters if c in _AMBIGUOUS_CYRILLIC)
    # If >80% of Cyrillic letters are the ambiguous ones, suspicious
    return ambiguous / len(cyr_letters) > 0.8


def filter_by_language(text: str, language: str) -> str:
    """Filter out lines that don't match the expected language.

    If language is "rus" — remove lines with mostly Latin characters
    and lines that look like misrecognized English (fake Cyrillic).
    If language is "eng" — remove lines with mostly Cyrillic characters.
    If language is "eng+rus" — keep everything.

    This helps filter OCR noise where random UI text or other language
    text gets picked up alongside subtitles.
    """
    if language == "eng+rus" or not text:
        return text

    lines = text.split("\n")
    filtered = []

    for line in lines:
        if not line.strip():
            continue

        cyrillic = sum(1 for c in line if "\u0400" <= c <= "\u04FF")
        latin = sum(1 for c in line if ("A" <= c <= "Z") or ("a" <= c <= "z"))
        total_alpha = cyrillic + latin

        if total_alpha == 0:
            continue

        cyr_ratio = cyrillic / total_alpha

        if language == "rus":
            # Keep lines with >40% Cyrillic
            if cyr_ratio <= 0.4:
                continue

            # Even if the line looks "Cyrillic", check if words are
            # actually misrecognized Latin (fake Cyrillic)
            words = line.split()
            real_words = 0
            fake_words = 0
            for word in words:
                word_cyr = sum(1 for c in word if "\u0400" <= c <= "\u04FF")
                if word_cyr >= 3:
                    if _is_fake_cyrillic_word(word):
                        fake_words += 1
                    else:
                        real_words += 1

            # If half or more of substantial words are fake Cyrillic, skip
            total_checked = real_words + fake_words
            if total_checked > 0 and fake_words / total_checked >= 0.5:
                continue

            filtered.append(line)

        elif language == "eng":
            if cyr_ratio < 0.6:
                filtered.append(line)
        else:
            filtered.append(line)

    return "\n".join(filtered)
