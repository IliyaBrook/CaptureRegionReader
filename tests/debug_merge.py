"""Debug the merge replacement for garbled words."""
from capture_region_reader.ocr_worker import (
    _is_garbled_word, _transliterate_garbled, _merge_bilingual_ocr,
    _is_clean_latin_word, _is_likely_garbled_russian,
    _find_positional_replacement,
)

# The core problem: "Р\и$" stripped to "Р\и" — needs to find "Plus"
test_cores = ["Р\\и", "Р\\и$"]
for core in test_cores:
    print(f"\nTesting core: {core!r}")
    print(f"  Is garbled: {_is_garbled_word(core)}")
    print(f"  Transliterated: {_transliterate_garbled(core)!r}")

eng_line_words = ["Most", "viewed,", "available", "on", "Disney", "Plus", "every", "night."]

# The garbled word "Р\и$" is at index 5 in primary
# English word "Plus" is at index 5 in eng line
print("\n--- Position-based replacement ---")
repl = _find_positional_replacement("Р\\и$", 5, eng_line_words)
print(f"Replacement at index 5: {repl!r}")

# Test the full merge
print("\n--- Full merge ---")
primary = "Самые просматриваемые, доступны на D|sney Р\\и$ каждую ночь."
eng_text = "Most viewed, available on Disney Plus every night."
merged = _merge_bilingual_ocr(primary, eng_text)
print(f"Primary:  {primary}")
print(f"English:  {eng_text}")
print(f"Merged:   {merged}")

# Test hard case: fully-Cyrillic garbled words
print("\n--- Hard case: fully-Cyrillic garbled ---")
primary2 = "Сегодня мы будем использовать новый гатемогКк для разработки"
eng_text2 = "Segodnya my budem ispolzovat novyy framework dlya razrabotki"
merged2 = _merge_bilingual_ocr(primary2, eng_text2)
print(f"Primary:  {primary2}")
print(f"English:  {eng_text2}")
print(f"Merged:   {merged2}")

# Check individual helpers
print("\n--- Helper checks ---")
print(f"Is 'framework' clean latin: {_is_clean_latin_word('framework')}")
print(f"Is 'гатемогКк' likely garbled vs 'framework': {_is_likely_garbled_russian('гатемогКк', 'framework')}")
print(f"Is 'Сегодня' likely garbled vs 'Segodnya': {_is_likely_garbled_russian('Сегодня', 'Segodnya')}")
print(f"Is 'будем' likely garbled vs 'budem': {_is_likely_garbled_russian('будем', 'budem')}")
