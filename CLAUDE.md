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

There is no automated test suite. The `tests/` directory contains manual test assets only.

## Architecture

**Threading model** — three threads communicating via Qt signals:
- **Main thread**: Qt event loop, UI (MainWindow), region selection overlay (RegionSelector)
- **OCR thread** (QThread): continuous screenshot → text isolation → OCR engine (OcrWorker)
- **TTS thread** (QThread): async queue-based TTS synthesis + playback (TtsWorker)

**Data pipeline**:
```
RegionSelector → OcrWorker (mss capture → subtitle 3-step pipeline → OCR engine)
    → TextDiffer (dedup) → TextCleaner (garbage removal) → TtsWorker (speech)
```

**app.py** is the orchestrator — creates all workers and the window, wires all Qt signals between them, manages settings load/save lifecycle. It is not a god-class; each component is self-contained.

## OCR Engine

Uses Tesseract OCR (PSM 6, OEM 1) with the 3-step text isolation pipeline as preprocessing. Supports `eng`, `rus`, and `eng+rus` language modes.

## TTS Engines

- **edge-tts** (default): cloud-based Microsoft Edge neural TTS, uses `en-US-AndrewNeural` / `ru-RU-DmitryNeural`.
- **Silero**: local neural TTS via torch.hub. Russian only (`xenia` voice at 48kHz); falls back to edge-tts for English text.

## Key Implementation Details

- **Coordinate spaces**: RegionSelector converts Qt logical coordinates to physical X11 pixels (for mss). High-DPI device pixel ratio is applied in region_selector.py. This is a common source of bugs.
- **Text isolation** — 3-step OpenCV pipeline (ported from square-cropper):
  1. **subtitle_detector.py**: character detection → block formation → boundary refinement → cropping. Handles both dark-background and light-background subtitles adaptively.
  2. **subtitle_binarizer.py**: Otsu thresholding → noise component removal → black-text-on-white output.
  3. **subtitle_cleaner.py**: connected component analysis → line grouping → artifact removal → final crop.
  - **text_isolator.py** is a thin wrapper: `isolate_text(rgb)` converts RGB→BGR, calls `detect_and_crop()`, converts result BGR→RGB.
  - All thresholds are relative to image dimensions (no hardcoded pixel sizes). Output is scaled to original width via INTER_LANCZOS4.
- **Two preview panels**: "Raw Capture" (original screenshot) and "Processed Preview" (what OCR engine sees after isolation). OcrWorker emits both `raw_frame_captured` and `frame_captured` signals.
- **Triple deduplication** in TextDiffer: exact match, 85% similarity threshold, and growth detection (for scrolling/streaming subtitles).
- **Fake Cyrillic detection** in TextCleaner: catches Latin words misrecognized as Cyrillic by Tesseract.
- **TTS voices** (edge-tts): English uses `en-US-AndrewNeural`, Russian uses `ru-RU-DmitryNeural`. Language can be auto-detected or fixed.
- **Settings persistence**: AppSettings dataclass serialized via QSettings. Loaded in app.py at startup, saved on shutdown.
- **Global hotkeys**: pynput-based listener on a daemon thread (hotkey_manager.py). Default: Ctrl+Alt+R (start/stop), Ctrl+Alt+S (select region).

## Languages

The app supports English and Russian OCR/TTS. Language-specific logic is spread across text_cleaner.py (filtering), tts_worker.py (voice selection), and ocr_worker.py (Tesseract lang parameter).
