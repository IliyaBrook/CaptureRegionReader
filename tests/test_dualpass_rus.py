"""Test dual-pass OCR merge: positional comparison approach.

Tests the _merge_bilingual_ocr() function which merges output from two
Tesseract passes:
  - Primary pass (lang=rus): reads Russian correctly, garbles English words
  - English pass (lang=eng): reads English correctly, garbles Russian words

The merge uses positional word-by-word comparison to pick the best version
of each word. Key techniques:
  Phase 1: _is_garbled_word() detects easy cases (noise chars, mixed scripts,
           digit+Cyrillic combos, all-look-alike chars)
  Phase 2: Positional comparison for hard cases (all-Cyrillic garbled words)
           Uses: unique Russian char ratio, phonetic match, reverse translit

Run: uv run python tests/test_dualpass_rus.py
"""
from __future__ import annotations

import sys

from capture_region_reader.ocr_worker import (
    TesseractEngine,
    _is_clean_latin_word,
    _is_garbled_word,
    _is_likely_garbled_russian,
    _is_phonetic_match,
    _merge_bilingual_ocr,
    _phonetic_transliterate,
    _transliterate_garbled,
    _unique_russian_ratio,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

_pass_count = 0
_fail_count = 0


def check(name: str, got, expected, detail: str = "") -> None:
    """Assert a test condition and print pass/fail."""
    global _pass_count, _fail_count
    ok = got == expected
    status = "PASS" if ok else "FAIL"
    if ok:
        _pass_count += 1
    else:
        _fail_count += 1
    suffix = f" -- {detail}" if detail else ""
    print(f"  [{status}] {name}{suffix}")
    if not ok:
        print(f"         got:      {got!r}")
        print(f"         expected: {expected!r}")


# ---------------------------------------------------------------------------
# 1. _is_garbled_word() tests
# ---------------------------------------------------------------------------

def test_is_garbled_word() -> None:
    """Test detection of garbled words (Phase 1 quick checks)."""
    print("\n=== _is_garbled_word() ===")

    # Noise chars inside word -> garbled
    check("noise backslash", _is_garbled_word("Р\\и"), True, "backslash is noise")
    check("noise pipe", _is_garbled_word("те|ст"), True, "pipe is noise")
    check("noise dollar", _is_garbled_word("те$т"), True, "dollar is noise")
    check("noise brackets", _is_garbled_word("те{ст}"), True, "brackets are noise")

    # Mixed Cyrillic + Latin -> garbled
    check("mixed CyrLat", _is_garbled_word("тестtest"), True, "mixed scripts")
    check("mixed LatCyr", _is_garbled_word("testтест"), True, "mixed scripts")

    # Digits mixed with Cyrillic -> garbled
    check("digits+cyr short", _is_garbled_word("01зп"), True, "digits adjacent to Cyrillic")
    check("digits+cyr long", _is_garbled_word("01зпеу"), True, "digits adjacent to Cyrillic")
    check("digits+cyr trail", _is_garbled_word("тест01"), True, "digits adjacent to Cyrillic")

    # All pixel-identical look-alikes -> garbled
    check("all identical 3", _is_garbled_word("АСЕ"), True, "all pixel-identical chars")
    check("all identical 4", _is_garbled_word("АСЕР"), True, "all pixel-identical chars")

    # Real Russian words -> NOT garbled
    check("real: Привет", _is_garbled_word("Привет"), False, "real Russian word")
    check("real: мир", _is_garbled_word("мир"), False, "real Russian word")
    check("real: Сегодня", _is_garbled_word("Сегодня"), False, "real Russian word")
    check("real: будем", _is_garbled_word("будем"), False, "real Russian word")
    check("real: использовать", _is_garbled_word("использовать"), False, "real Russian word")
    check("real: каждую", _is_garbled_word("каждую"), False, "real Russian word")
    check("real: Доступны", _is_garbled_word("Доступны"), False, "real Russian word")

    # ALL-Cyrillic garbled words (the hard case) -> NOT detected by Phase 1
    # These are handled by Phase 2 (positional comparison) instead.
    check("hard: гатемогКк", _is_garbled_word("гатемогКк"), False,
          "all-Cyrillic garbled -- Phase 1 cannot detect, Phase 2 handles")
    check("hard: лартор", _is_garbled_word("лартор"), False,
          "all-Cyrillic garbled -- Phase 1 cannot detect, Phase 2 handles")

    # Short words -> NOT garbled (too short to judge)
    check("short: я", _is_garbled_word("я"), False, "too short")
    check("short: на", _is_garbled_word("на"), False, "too short (single char)")

    # Latin-only words -> NOT garbled (already Latin)
    check("latin: hello", _is_garbled_word("hello"), False, "already Latin")
    check("latin: framework", _is_garbled_word("framework"), False, "already Latin")


# ---------------------------------------------------------------------------
# 2. _is_clean_latin_word() tests
# ---------------------------------------------------------------------------

def test_is_clean_latin_word() -> None:
    """Test detection of clean Latin words from eng-pass."""
    print("\n=== _is_clean_latin_word() ===")

    # Valid clean Latin words
    check("clean: framework", _is_clean_latin_word("framework"), True)
    check("clean: laptop", _is_clean_latin_word("laptop"), True)
    check("clean: Disney", _is_clean_latin_word("Disney"), True)
    check("clean: Plus", _is_clean_latin_word("Plus"), True)
    check("clean: FRAMEWORK", _is_clean_latin_word("FRAMEWORK"), True)
    check("clean: don't", _is_clean_latin_word("don't"), True, "apostrophe allowed")

    # NOT clean Latin
    check("cyrillic: Привет", _is_clean_latin_word("Привет"), False, "Cyrillic chars")
    check("mixed: testТест", _is_clean_latin_word("testТест"), False, "mixed scripts")
    check("noise: te|st", _is_clean_latin_word("te|st"), False, "noise chars")
    check("short: a", _is_clean_latin_word("a"), False, "too short")
    check("short: I", _is_clean_latin_word("I"), False, "too short")
    check("digits: 123", _is_clean_latin_word("123"), False, "no Latin alpha")
    check("mostly digits: a1", _is_clean_latin_word("a1"), False, "< 80% Latin")


# ---------------------------------------------------------------------------
# 3. _is_phonetic_match() tests
# ---------------------------------------------------------------------------

def test_is_phonetic_match() -> None:
    """Test phonetic transliteration matching."""
    print("\n=== _is_phonetic_match() ===")

    # Real Russian -> eng transliteration: should match
    check("phonetic: Сегодня/Segodnya", _is_phonetic_match("Сегодня", "Segodnya"), True)
    check("phonetic: будем/budem", _is_phonetic_match("будем", "budem"), True)
    check("phonetic: использовать/ispolzovat",
          _is_phonetic_match("использовать", "ispolzovat"), True)
    check("phonetic: новый/novyy", _is_phonetic_match("новый", "novyy"), True)
    check("phonetic: разработки/razrabotki",
          _is_phonetic_match("разработки", "razrabotki"), True)
    check("phonetic: Привет/Privet", _is_phonetic_match("Привет", "Privet"), True)
    check("phonetic: для/dlya", _is_phonetic_match("для", "dlya"), True)

    # Garbled -> should NOT match
    check("no match: гатемогКк/framework",
          _is_phonetic_match("гатемогКк", "framework"), False)
    check("no match: лартор/laptop", _is_phonetic_match("лартор", "laptop"), False)
    check("no match: Доступны/Available",
          _is_phonetic_match("Доступны", "Available"), False)
    check("no match: каждую/every", _is_phonetic_match("каждую", "every"), False)

    # Imperfect transliterations (Tesseract variations) -- should still match
    check("imperfect: будем/bydem", _is_phonetic_match("будем", "bydem"), True,
          "slight vowel variation")
    check("imperfect: новый/novyi", _is_phonetic_match("новый", "novyi"), True,
          "yy->yi variation")


# ---------------------------------------------------------------------------
# 4. _is_likely_garbled_russian() tests
# ---------------------------------------------------------------------------

def test_is_likely_garbled_russian() -> None:
    """Test the key heuristic for positional comparison."""
    print("\n=== _is_likely_garbled_russian() ===")

    # Garbled English words -> should return True (replace them)
    check("garbled: гатемогКк/framework",
          _is_likely_garbled_russian("гатемогКк", "framework"), True,
          "all-Cyrillic garbled English")
    check("garbled: лартор/laptop",
          _is_likely_garbled_russian("лартор", "laptop"), True,
          "all-Cyrillic garbled English")
    check("garbled: ГАТЕМОГКК/FRAMEWORK",
          _is_likely_garbled_russian("ГАТЕМОГКК", "FRAMEWORK"), True,
          "uppercase garbled")

    # Real Russian words -> should return False (keep them)
    check("keep: Сегодня/Segodnya",
          _is_likely_garbled_russian("Сегодня", "Segodnya"), False,
          "phonetic match protects")
    check("keep: будем/budem",
          _is_likely_garbled_russian("будем", "budem"), False,
          "phonetic match protects")
    check("keep: использовать/ispolzovat",
          _is_likely_garbled_russian("использовать", "ispolzovat"), False,
          "phonetic match protects")
    check("keep: Доступны/Available",
          _is_likely_garbled_russian("Доступны", "Available"), False,
          "high unique Russian ratio protects")
    check("keep: каждую/every",
          _is_likely_garbled_russian("каждую", "every"), False,
          "high unique Russian ratio protects")
    check("keep: Привет/Hello",
          _is_likely_garbled_russian("Привет", "Hello"), False,
          "reverse translit does NOT match Hello")

    # Short words -> always False (too risky)
    check("short: мир/mir", _is_likely_garbled_russian("мир", "mir"), False)
    check("short: для/for", _is_likely_garbled_russian("для", "for"), False)
    check("short: это/this", _is_likely_garbled_russian("это", "this"), False)

    # Common short words -> always False
    check("common: на/on", _is_likely_garbled_russian("на", "on"), False)
    check("common: мы/my", _is_likely_garbled_russian("мы", "my"), False)

    # Russian loanwords -> should return False (keep them)
    check("loanword: перформанс/performance",
          _is_likely_garbled_russian("перформанс", "performance"), False,
          "phonetic match catches loanwords")

    # Very different lengths -> should return False
    check("len mismatch: тест/framework",
          _is_likely_garbled_russian("тест", "framework"), False,
          "length ratio out of range + too short")

    # Non-Cyrillic word -> should return False
    check("not cyrillic: test/test",
          _is_likely_garbled_russian("test", "test"), False,
          "not all-Cyrillic")


# ---------------------------------------------------------------------------
# 5. _unique_russian_ratio() tests
# ---------------------------------------------------------------------------

def test_unique_russian_ratio() -> None:
    """Test unique Russian character ratio computation."""
    print("\n=== _unique_russian_ratio() ===")

    # Real Russian words typically have >= 0.2 unique ratio
    check("Доступны >= 0.3", _unique_russian_ratio("Доступны") >= 0.3, True,
          f"ratio={_unique_russian_ratio('Доступны'):.2f}")
    check("Сегодня >= 0.3", _unique_russian_ratio("Сегодня") >= 0.3, True,
          f"ratio={_unique_russian_ratio('Сегодня'):.2f}")
    check("каждую >= 0.3", _unique_russian_ratio("каждую") >= 0.3, True,
          f"ratio={_unique_russian_ratio('каждую'):.2f}")

    # Garbled words typically have < 0.3 unique ratio
    check("гатемогКк < 0.3", _unique_russian_ratio("гатемогКк") < 0.3, True,
          f"ratio={_unique_russian_ratio('гатемогКк'):.2f}")
    check("лартор < 0.3", _unique_russian_ratio("лартор") < 0.3, True,
          f"ratio={_unique_russian_ratio('лартор'):.2f}")

    # Edge: pure look-alike words have 0.0 ratio
    check("тест = 0.0", _unique_russian_ratio("тест"), 0.0,
          "all chars in look-alike table")


# ---------------------------------------------------------------------------
# 6. _merge_bilingual_ocr() integration tests
# ---------------------------------------------------------------------------

def test_merge_bilingual_ocr() -> None:
    """Test the full merge function with realistic and edge-case inputs."""
    print("\n=== _merge_bilingual_ocr() integration tests ===")

    # --- Required test cases from specification ---

    # Test case 1: Single English word in Russian sentence
    result = _merge_bilingual_ocr(
        "Сегодня мы будем использовать новый гатемогКк для разработки",
        "Segodnya my budem ispolzovat novyy framework dlya razrabotki",
    )
    check("TC1: framework merged",
          result,
          "Сегодня мы будем использовать новый framework для разработки")

    # Test case 2: Brand names with noise chars
    result = _merge_bilingual_ocr(
        "Доступны на 01зпеу Р\\и$ каждую ночь",
        "Available on Disney Plus every night",
    )
    check("TC2: Disney Plus merged",
          result,
          "Доступны на Disney Plus каждую ночь")

    # Test case 3: All Russian, no English words at all
    result = _merge_bilingual_ocr(
        "Привет мир это тест",
        "Privet mir eto test",
    )
    check("TC3: all Russian unchanged",
          result,
          "Привет мир это тест")

    # --- Additional integration tests ---

    # Mixed: multiple garbled words
    result = _merge_bilingual_ocr(
        "Я люблю гатемогКк и лартор",
        "I love framework and laptop",
    )
    check("multi-garble: framework + laptop",
          "framework" in result and "laptop" in result, True,
          f"result: {result}")

    # Uppercase garbled
    result = _merge_bilingual_ocr(
        "Используйте ГАТЕМОГКК и ЛАРТОР",
        "Use FRAMEWORK and LAPTOP",
    )
    check("uppercase garbled",
          "FRAMEWORK" in result and "LAPTOP" in result, True,
          f"result: {result}")

    # Russian loanwords should be preserved
    result = _merge_bilingual_ocr(
        "Он использует перформанс для тестирования",
        "He uses performance for testing",
    )
    check("loanword preserved",
          "перформанс" in result, True,
          f"result: {result}")

    # Punctuation preserved on replacement
    result = _merge_bilingual_ocr(
        "Установи гатемогКк.",
        "Install framework.",
    )
    check("punctuation after garbled",
          result.endswith("framework."), True,
          f"result: {result}")

    # Empty inputs
    check("empty primary", _merge_bilingual_ocr("", "hello"), "")
    check("empty eng", _merge_bilingual_ocr("Привет мир", ""), "Привет мир")

    # All Russian kept when eng has unrelated translations
    result = _merge_bilingual_ocr(
        "Привет мир это тест",
        "Hello world this test",
    )
    check("Russian vs English translations",
          result,
          "Привет мир это тест",
          "Should NOT replace with unrelated English words")

    # Multi-line text
    result = _merge_bilingual_ocr(
        "Первая строка\nгатемогКк и лартор",
        "First line\nframework and laptop",
    )
    check("multi-line: has framework",
          "framework" in result and "laptop" in result, True,
          f"result: {result}")

    # Noise char cleanup on replacement
    result = _merge_bilingual_ocr(
        "Тест Р\\и$ конец",
        "Test Plus end",
    )
    check("noise cleanup: no trailing $",
          "Plus$" not in result and "Plus" in result, True,
          f"result: {result}")


# ---------------------------------------------------------------------------
# 7. Live Tesseract dual-pass test (requires Tesseract installed)
# ---------------------------------------------------------------------------

def test_live_tesseract() -> None:
    """Test with actual Tesseract OCR on rendered images.

    Requires: Tesseract, DejaVu font.
    Skipped if dependencies are not available.
    """
    print("\n=== Live Tesseract dual-pass test ===")
    try:
        import pytesseract
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("  [SKIP] Missing PIL or pytesseract")
        return

    try:
        pytesseract.get_tesseract_version()
    except Exception:
        print("  [SKIP] Tesseract not installed")
        return

    def make_img(text: str, font_size: int = 32) -> Image.Image:
        img = Image.new("RGB", (1400, 120), (0, 0, 0))
        d = ImageDraw.Draw(img)
        try:
            f = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size
            )
        except (OSError, IOError):
            f = ImageFont.load_default()
        d.text((20, 30), text, fill=(255, 255, 255), font=f)
        return img

    engine = TesseractEngine()

    # Test: pure Russian text should remain mostly Russian after dual-pass
    text = "Привет мир это просто русский текст"
    img = make_img(text)
    result = engine.recognize(img, "rus")
    print(f"\n  Input:  {text}")
    print(f"  Output: {result}")

    # Tesseract output varies by environment/font. Just verify:
    # 1. We got some output (not empty)
    # 2. Most words are still Cyrillic (not all replaced with English)
    check("live: non-empty output", len(result.strip()) > 0, True)

    words = result.split()
    if words:
        cyr_words = sum(
            1 for w in words
            if any("\u0400" <= c <= "\u052f" for c in w)
        )
        cyr_ratio = cyr_words / len(words)
        # Threshold is lenient (>=40%) because Tesseract quality varies
        # by font/rendering. The important thing is that the merge didn't
        # replace ALL Russian words with English garbage.
        check("live: has Cyrillic words (>=40%)", cyr_ratio >= 0.4, True,
              f"{cyr_words}/{len(words)} Cyrillic words, ratio={cyr_ratio:.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global _pass_count, _fail_count

    print("=" * 70)
    print("Dual-pass OCR merge tests (positional comparison approach)")
    print("=" * 70)

    test_is_garbled_word()
    test_is_clean_latin_word()
    test_is_phonetic_match()
    test_is_likely_garbled_russian()
    test_unique_russian_ratio()
    test_merge_bilingual_ocr()
    test_live_tesseract()

    print("\n" + "=" * 70)
    total = _pass_count + _fail_count
    print(f"Results: {_pass_count}/{total} passed, {_fail_count} failed")
    print("=" * 70)

    if _fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
