---
name: run-and-test
description: "Launch the app for manual testing and verify it works after code changes. Use when the user asks to test, run, or check the app."
disable-model-invocation: true
---

# Run & Test CaptureRegionReader

## Quick Launch

```bash
# Standard run
uv run capture-region-reader

# Dev mode (direct module execution)
uv run python -m capture_region_reader

# Debug mode (verbose logging)
CRR_DEBUG=1 uv run capture-region-reader
```

## Pre-flight Checks

Before launching, verify system dependencies:

```bash
make check-deps
```

Required system packages:
- `tesseract-ocr`, `tesseract-ocr-eng`, `tesseract-ocr-rus`
- `ffmpeg` (provides `ffplay` for audio playback)
- `libxcb-xinerama0`, `libxcb-cursor0` (Qt/X11)

Install missing:
```bash
sudo apt install tesseract-ocr tesseract-ocr-eng tesseract-ocr-rus ffmpeg libxcb-xinerama0 libxcb-cursor0
```

## Common Issues

### App won't start
- **"Could not load the Qt platform plugin"**: Missing xcb libraries. Install `libxcb-xinerama0 libxcb-cursor0`.
- **"tesseract not found"**: Install `tesseract-ocr`.
- **Wayland issues**: Try `QT_QPA_PLATFORM=xcb uv run capture-region-reader`.

### OCR not working
- Check Tesseract lang packs: `tesseract --list-langs`
- Verify region is selected (Ctrl+Alt+S)
- Check text isolation mode matches your content

### TTS not working
- **edge-tts**: Requires internet connection.
- **XTTS**: First run downloads ~2GB model. Check disk space.
- **No audio**: Verify `ffplay` is installed (`which ffplay`).

## Testing Workflow

1. Run `make check-deps` to verify environment
2. Launch with `make dev` or `make debug` for verbose output
3. Select a screen region with Ctrl+Alt+S
4. Start capture with Ctrl+Alt+R
5. Verify OCR output appears in the UI
6. Verify TTS speaks detected text
7. Check both English and Russian text if applicable
