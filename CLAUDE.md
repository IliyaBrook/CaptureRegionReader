# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CaptureRegionReader — PyQt6 desktop app that captures a screen region, runs OCR on it in a loop, and speaks detected text aloud via neural TTS. Primary use case: reading game subtitles, video captions, or any on-screen text in real-time.

## Build & Run Commands

```bash
uv sync              # Install dependencies (or: make setup)
uv run capture-region-reader   # Run the app (or: make run)
uv run python -m capture_region_reader  # Dev mode (or: make dev)
make clean           # Remove __pycache__, .egg-info, build artifacts
```

**System dependencies** (must be installed separately):
```bash
sudo apt install tesseract-ocr tesseract-ocr-eng tesseract-ocr-rus ffmpeg libxcb-xinerama0 libxcb-cursor0
```

There is no automated test suite. The `.tests/` directory contains manual test assets only.

## Architecture

**Threading model** — three threads communicating via Qt signals:
- **Main thread**: Qt event loop, UI (MainWindow), region selection overlay (RegionSelector)
- **OCR thread** (QThread): continuous screenshot → text isolation → Tesseract OCR (OcrWorker)
- **TTS thread** (QThread): async queue-based edge-tts synthesis + ffplay playback (TtsWorker)

**Data pipeline**:
```
RegionSelector → OcrWorker (mss capture → TextIsolator → pytesseract)
    → TextDiffer (dedup) → TextCleaner (garbage removal) → TtsWorker (speech)
```

**app.py** is the orchestrator — creates all workers and the window, wires all Qt signals between them, manages settings load/save lifecycle. It is not a god-class; each component is self-contained.

## Key Implementation Details

- **Coordinate spaces**: RegionSelector converts Qt logical coordinates to physical X11 pixels (for mss). High-DPI device pixel ratio is applied in region_selector.py. This is a common source of bugs.
- **Text isolation** (text_isolator.py): OpenCV pipeline that detects text contours via Otsu thresholding, clusters characters into lines, scores candidate subtitle lines, and crops/binarizes the result. Handles both light-on-dark and dark-on-light text.
- **Triple deduplication** in TextDiffer: exact match, 85% similarity threshold, and growth detection (for scrolling/streaming subtitles).
- **Fake Cyrillic detection** in TextCleaner: catches Latin words misrecognized as Cyrillic by Tesseract.
- **TTS voices**: English uses `en-US-AndrewNeural`, Russian uses `ru-RU-DmitryNeural`. Language can be auto-detected or fixed.
- **Settings persistence**: AppSettings dataclass serialized via QSettings. Loaded in app.py at startup, saved on shutdown.
- **Global hotkeys**: pynput-based listener on a daemon thread (hotkey_manager.py). Default: Ctrl+Alt+R (start/stop), Ctrl+Alt+S (select region).

## Languages

The app supports English and Russian OCR/TTS. Language-specific logic is spread across text_cleaner.py (filtering), tts_worker.py (voice selection), and ocr_worker.py (Tesseract lang parameter).
