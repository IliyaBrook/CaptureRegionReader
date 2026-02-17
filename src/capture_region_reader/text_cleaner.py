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

    # Remove standalone digits that look like OCR noise (single digit surrounded by spaces)
    # e.g. "программу К шв 9" → "программу К шв"
    text = re.sub(r"\s+\d\s+", " ", text)
    # Remove trailing standalone digits
    text = re.sub(r"\s+\d$", "", text)

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
