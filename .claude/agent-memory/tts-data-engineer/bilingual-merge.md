# Bilingual OCR Merge: Design Notes

## Problem
When Tesseract runs with `lang=rus`, it garbles ALL English words into fully-Cyrillic
nonsense (e.g., "framework" -> "гатемогКк", "laptop" -> "лартор"). The old approach
used `_is_garbled_word()` to detect these, but it fundamentally fails because garbled
words like "гатемогКк" are made entirely of normal Cyrillic characters.

## Solution: Two-Phase Positional Comparison

### Phase 1: Quick Garble Detection (_is_garbled_word)
Catches EASY cases that are unambiguously garbled:
- OCR noise chars (\, |, $, {, }, [, ]) inside the word
- Mixed Cyrillic+Latin scripts within one word
- Digits mixed with Cyrillic (>=2 Cyrillic chars) -- added Feb 2026
- All pixel-identical look-alike chars (e.g., СОРТ = COPT)

### Phase 2: Positional Comparison (_is_likely_garbled_russian)
For words that pass Phase 1 (look like normal Cyrillic), compares with eng-pass word
at the same position. Three layered checks (any one can protect a real Russian word):

1. **Unique Russian Ratio (>= 0.3)**: Characters NOT in `_CYR_TO_LAT` look-alike table
   (б, г, д, ж, з, л, п, ф, ц, ч, ш, щ, etc.). Real Russian words typically 20-50%;
   garbled words 0-22%. Threshold 0.3 separates cleanly.
   - Protects: Доступны(0.38), Сегодня(0.43), каждую(0.50)
   - Doesn't protect: Привет(0.17), тест(0.00), лартор(0.17)

2. **Phonetic Match (>= 0.7 LCS)**: Forward-transliterate Russian word using
   `_RUS_TO_PHONETIC` table and compare with eng-pass word via LCS ratio.
   When eng-pass reads real Russian, it produces a phonetic transliteration.
   - Protects: Сегодня/Segodnya(1.0), будем/budem(1.0), Привет/Privet(1.0)
   - Doesn't protect: гатемогКк/framework(0.44), лартор/laptop(0.67)
   - Threshold 0.7 calibrated to accept imperfect Tesseract variations (e.g., budem/bydem=0.80)

3. **Reverse Transliteration Match (>= 0.4 LCS)**: Convert Cyrillic look-alike chars
   BACK to Latin via `_CYR_TO_LAT` and compare with eng-word. If overlap is high,
   the word IS garbled. This is the POSITIVE confirmation step.
   - Confirms garble: лартор->"лaptop" vs "laptop"=0.83, гатемогКк->"гatemoгKk" vs "framework"=0.44
   - Rejects false positive: Привет->"Пpivet" vs "Hello"=0.17

### Key Insight: Conservative Default
If none of the positive garble indicators fire, the word is kept as Russian.
False negatives (keeping a garbled word) are preferable to false positives
(replacing a real Russian word with unrelated English).

## Performance Notes
- LCS computation is O(n*m) but words are short (<30 chars), so negligible
- `_RUS_TO_PHONETIC` and `_COMMON_RUS_SHORT` are module-level constants
- No dictionary lookups or external data files needed

## Known Limitations
- Words made entirely of pixel-identical look-alikes (рост, утро) are caught by
  Phase 1 `_is_garbled_word()` which has false positives. These words would need
  a dictionary or ML model to distinguish reliably.
- Very short garbled words (<=3 chars) are never replaced (too many false positive risks).
- Word alignment assumes 1:1 line correspondence between passes. Different line
  splitting can cause misalignment.

## Test Coverage
- `.tests/test_dualpass_rus.py`: 87 unit tests covering all helpers and integration
- `.tests/test_bilingual.py`: existing tests still pass (backward compatible)
- `.tests/debug_merge.py`: imports old function names, needs updating
