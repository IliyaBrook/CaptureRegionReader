# Agent Memory - TTS Data Engineer

## Project Architecture
- **Threading**: 3 threads - Main (Qt UI), OCR (QThread), TTS (QThread)
- **Data flow**: RegionSelector -> OcrWorker -> TextDiffer -> TextCleaner -> TtsWorker
- **Key files**: `text_isolator.py`, `text_cleaner.py`, `text_differ.py`, `ocr_worker.py`, `app.py`
- **Build system**: `uv` (pyproject.toml), Python 3.10+ (pyenv 3.10.16 available)
- **pyenv note**: `.python-version` says 3.13 but only 3.10.16 installed; use `uv run` for commands

## Key Design Decisions (Feb 2026 Refactoring)

### text_isolator.py
- All magic numbers extracted into `IsolatorConfig` dataclass (48 fields)
- Config is optional param on `isolate_text()` -- default singleton for backward compat
- `_measure_contour_boxes()` extracted as reusable helper (was duplicated for bright/dark strategies)
- CLAHE enhancement only on cropped regions, never before `_find_char_boxes()` (breaks brightness thresholds)

### text_cleaner.py
- Character replacements organized into 5 named dicts: fullwidth, CJK, typographic, symbols, context
- Added `_preprocess_line()` (from Translumo pattern): strips dashes, unwraps quotes, removes dot sequences
- `filter_by_language()` for eng+rus mode now rejects mixed-script words (Cyr+Lat in same word)
- Uses `_is_cyrillic()` and `_is_latin()` helper functions with Unicode range checks

### text_differ.py
- Added `_jaro_similarity()` (ported from Translumo StringExtensions.GetJaroSimilarity)
- `is_text_similar()` now uses 5 metrics: Dice, Jaro, Keyword, Jaccard, Jaro+Dice average
- Removed all `print()` debug logging, module uses no stdout output
- Constants extracted: DEFAULT_SIMILARITY_THRESHOLD, GROWTH_WORD_OVERLAP_RATIO, etc.

### ocr_worker.py
- Removed redundant dedup logic (_texts_similar, _extract_real_words, _strip_ocr_artifacts, growth detection)
- All deduplication now in TextDiffer (via app.py signal handler)
- Worker always emits text (even empty "") so TextDiffer can track subtitle disappearance
- `_filter_ocr_garbage` renamed to `filter_ocr_garbage` (public API)
- `_process_frame()` extracted from run loop for readability
- Switched from `print()` to `logging` module

## Reference: Translumo Patterns
- See `/mnt/DiskE_Crucial/codding/Translumo/` for C# reference project
- Key files: `TextValidityPredictor.cs`, `TextResultCacheService.cs`, `StringExtensions.cs`
- Translumo uses ML model (ONNX) for text validity scoring -- not adopted (too complex, needs training data)
- Translumo uses Jaro+Dice average for cache dedup -- adopted in text_differ.py
- Translumo Tesseract preprocessing: simple grayscale->threshold(150)->morphOpen -- simpler than our pipeline
- Translumo fullwidth char normalization -- adopted in text_cleaner.py
