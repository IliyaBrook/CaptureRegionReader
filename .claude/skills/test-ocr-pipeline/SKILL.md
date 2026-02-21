---
name: test-ocr-pipeline
description: "Test the OCR pipeline (text isolation, OCR engines, text cleaning) using test assets from tests/. Use when debugging OCR accuracy, text isolation, or text cleaning issues."
disable-model-invocation: true
---

# Test OCR Pipeline

## Test Assets

Located in `tests/`:

| Asset | Purpose |
|-------|---------|
| `tests/test-box-search/1.png`, `2.png` | Box search mode testing (subtitle bars) |
| `tests/test-box-search/3.png`, `pay.png` | Small region box search |
| `tests/test_bilingual.py` | Bilingual OCR merging test |
| `tests/test_dualpass_rus.py` | Dual-pass Russian OCR test |
| `tests/test_isolator_vs_raw.py` | Compare isolated vs raw OCR |
| `tests/test_tesseract_bilingual.py` | Tesseract bilingual config test |
| `tests/debug_merge.py` | Debug text merging logic |

## Running Test Scripts

```bash
# Run individual test scripts
uv run python tests/test_bilingual.py
uv run python tests/test_dualpass_rus.py
uv run python tests/test_isolator_vs_raw.py
uv run python tests/test_tesseract_bilingual.py
uv run python tests/debug_merge.py
```

## Quick OCR Test (inline)

To quickly test OCR on a specific image:

```python
import cv2
from capture_region_reader.text_isolator import TextIsolator, IsolatorConfig
from capture_region_reader.ocr_worker import TesseractEngine

# Load test image
img = cv2.imread("tests/test-box-search/1.png")

# Test with text isolation
isolator = TextIsolator(IsolatorConfig())
processed = isolator.isolate(img)

# Run OCR
engine = TesseractEngine(lang="eng+rus")
text = engine.recognize(processed)
print(f"Detected: {text}")
```

## Testing Text Isolation Modes

### dark_box mode (default)
Detects dark subtitle bars with light text:
```python
config = IsolatorConfig(mode="dark_box")
isolator = TextIsolator(config)
result = isolator.isolate(img)
```

### box_search mode
User-defined color region detection:
```python
config = IsolatorConfig(mode="box_search", box_search_color=(R, G, B))
isolator = TextIsolator(config)
result = isolator.isolate(img)
```

## Testing Text Cleaning

```python
from capture_region_reader.text_cleaner import TextCleaner

cleaner = TextCleaner()
raw = "Пrивeт мир"  # Mixed Latin/Cyrillic
cleaned = cleaner.clean(raw, lang="rus")
print(f"Cleaned: {cleaned}")
```

## Key Things to Verify

1. **Text isolation**: Does `isolate()` produce a clean binary image with readable text?
2. **OCR accuracy**: Does the engine correctly read the isolated text?
3. **Fake Cyrillic**: Does TextCleaner catch Latin chars misrecognized as Cyrillic?
4. **Bilingual merging**: Does dual-pass OCR correctly merge eng+rus results?
5. **Deduplication**: Does TextDiffer correctly suppress repeated/similar text?
