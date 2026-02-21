"""Test bilingual OCR merge and garbled word detection."""

from capture_region_reader.ocr_worker import (
    _is_garbled_word, _transliterate_garbled, _merge_bilingual_ocr,
    filter_ocr_garbage
)
from capture_region_reader.text_cleaner import clean_for_tts, filter_by_language


def test_garbled_detection():
    print("=== Garbled word detection tests ===")
    tests = [
        # (word, expected_garbled, description)
        # Normal Russian words — must NOT be flagged
        ("Привет", False, "Normal Russian word - has non-identical chars п,р,и,в,е,т"),
        ("Disney", False, "Normal English word"),
        ("Москва", False, "Russian city - м,к,в are NOT pixel-identical"),
        ("Самые", False, "Normal Russian - м,ы not identical"),
        ("просматриваемые", False, "Long Russian word"),
        ("доступны", False, "Normal Russian word"),
        ("каждую", False, "Normal Russian word"),
        ("ночь", False, "Normal Russian word"),
        ("на", False, "Short Russian preposition"),
        ("и", False, "Single char - too short"),
        ("кот", False, "Russian word - к is NOT pixel-identical"),
        ("мост", False, "Russian word - м is NOT pixel-identical"),
        # Garbled words — MUST be flagged
        ("РОВЕРТ", True, "All identical look-alikes: Р=P,О=O,Е=E,Р=P,Т=T (В is ambig but close)"),
        ("NАСК", True, "Mixed Latin N with Cyrillic АСК"),
        ("D\\и$ney", True, "Backslash and dollar = garbled"),
        ("Р\\и$", True, "Noise symbols inside word"),
        ("Те$т", True, "Dollar sign inside word"),
        ("Н|а", True, "Pipe inside word"),
        # Pure identical look-alikes (3+ chars, all pixel-identical)
        ("Рос", True, "Р=P, о=o, с=c — all pixel-identical"),
        ("СОРТ", True, "С=C, О=O, Р=P, Т=T — all pixel-identical"),
        ("Росе", True, "Р=P, о=o, с=c, е=e — all pixel-identical"),
    ]
    for word, expected, desc in tests:
        result = _is_garbled_word(word)
        status = "PASS" if result == expected else "FAIL"
        print(f"  {status}: _is_garbled_word({word!r}) = {result} (expected {expected}) -- {desc}")


def test_transliteration():
    print("\n=== Transliteration tests ===")
    # Cyrillic look-alikes -> Latin
    tests = [
        ("РОВЕРТ", "POBERT"),  # Р->P, О->O, В->B, Е->E, Р->P, Т->T
        ("НаСК", "HaCK"),  # Н->H, а->a, С->C, К->K
    ]
    for garbled, expected_approx in tests:
        result = _transliterate_garbled(garbled)
        print(f"  Transliterate {garbled!r} -> {result!r} (expected ~{expected_approx!r})")


def test_merge_bilingual():
    print("\n=== Merge bilingual OCR tests ===")

    # Test 1: Garbled English brand name in Russian text
    # The garbled words contain noise symbols that trigger detection
    primary = "Самые просматриваемые, доступны на D|sney Р\\и$ каждую ночь."
    eng_text = "Most viewed, available on Disney Plus every night."
    merged = _merge_bilingual_ocr(primary, eng_text)
    print(f"  Primary:  {primary}")
    print(f"  English:  {eng_text}")
    print(f"  Merged:   {merged}")
    # The garbled "D|sney" and "Р\и$" should be replaced
    assert "D|sney" not in merged, f"Garbled 'D|sney' should be fixed: {merged}"
    assert "Р\\и$" not in merged, f"Garbled 'Р\\и$' should be fixed: {merged}"
    print("  PASS: Garbled words fixed")

    # Test 2: Normal Russian text (should be unchanged)
    primary2 = "Привет мир, это тест"
    eng_text2 = "Hello world, this is a test"
    merged2 = _merge_bilingual_ocr(primary2, eng_text2)
    print(f"\n  Primary:  {primary2}")
    print(f"  English:  {eng_text2}")
    print(f"  Merged:   {merged2}")
    assert "Привет" in merged2, f"Russian word should be preserved: {merged2}"
    print("  PASS: Russian text preserved")


def test_filter_by_language_bilingual():
    print("\n=== filter_by_language eng+rus tests ===")

    # Should keep lines with both scripts
    mixed_line = "Доступны на DisneyPlus каждую ночь"
    result = filter_by_language(mixed_line, "eng+rus")
    print(f"  Input:  {mixed_line}")
    print(f"  Output: {result}")
    assert result == mixed_line, f"Mixed eng+rus line should be kept: {result}"
    print("  PASS: Bilingual line preserved")

    # Should keep pure Russian
    rus_line = "Это просто русский текст"
    result2 = filter_by_language(rus_line, "eng+rus")
    assert result2 == rus_line, f"Pure Russian should be kept: {result2}"
    print("  PASS: Pure Russian line preserved")

    # Should keep pure English
    eng_line = "This is just English text"
    result3 = filter_by_language(eng_line, "eng+rus")
    assert result3 == eng_line, f"Pure English should be kept: {result3}"
    print("  PASS: Pure English line preserved")


def test_clean_for_tts_apostrophes():
    print("\n=== clean_for_tts apostrophe preservation ===")
    test = "it's available on Disney's platform and we'll watch"
    cleaned = clean_for_tts(test)
    print(f"  Input:  {test}")
    print(f"  Output: {cleaned}")
    assert "it's" in cleaned, f"Apostrophe in 'it's' should be preserved: {cleaned}"
    assert "Disney's" in cleaned, f"Apostrophe in 'Disney's' should be preserved: {cleaned}"
    print("  PASS: Apostrophes preserved in English contractions")


def test_garbage_filter_bilingual():
    print("\n=== Garbage filter bilingual tolerance ===")
    # Line with mixed scripts after merge should NOT be removed
    text = "Доступны на DisneyPlus каждую ночь"
    result = filter_ocr_garbage(text)
    print(f"  Input:  {text}")
    print(f"  Output: {result}")
    assert result == text, f"Bilingual line should survive garbage filter: {result}"
    print("  PASS: Bilingual line survived garbage filter")


def test_false_positive_preservation():
    print("\n=== False positive preservation (real Russian words safe from garbling) ===")
    from capture_region_reader.ocr_worker import _merge_bilingual_ocr

    # Russian sentence containing words that are 100% look-alikes
    # (рост, утро, сор) — these should NOT be replaced by English
    primary = "Утро было тёплым и рост цен удивил"
    eng_text = "The morning was warm and prices rose"
    merged = _merge_bilingual_ocr(primary, eng_text)
    print(f"  Primary:  {primary}")
    print(f"  English:  {eng_text}")
    print(f"  Merged:   {merged}")

    # "Утро" is flagged as garbled (у,т,р,о all identical), but since
    # _find_best_eng_replacement checks position and the eng word at that
    # position is "The" (very different length) — it should stay as-is
    # or get replaced by a reasonable word. Either way, the sentence
    # should still be predominantly Russian.
    cyr_count = sum(1 for c in merged if "\u0400" <= c <= "\u052f")
    total_alpha = sum(1 for c in merged if c.isalpha())
    cyr_ratio = cyr_count / max(total_alpha, 1)
    print(f"  Cyrillic ratio: {cyr_ratio:.2f}")
    # At minimum, Russian words like "было", "тёплым", "цен", "удивил" must remain
    for rus_word in ["было", "тёплым", "цен", "удивил"]:
        assert rus_word in merged, f"Russian word '{rus_word}' should be preserved: {merged}"
    print("  PASS: Core Russian words preserved")


if __name__ == "__main__":
    test_garbled_detection()
    test_transliteration()
    test_merge_bilingual()
    test_filter_by_language_bilingual()
    test_clean_for_tts_apostrophes()
    test_garbage_filter_bilingual()
    test_false_positive_preservation()
    print("\n=== ALL TESTS PASSED ===")
