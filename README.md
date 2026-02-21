# CaptureRegionReader

A Linux desktop application that captures a screen region, performs real-time OCR using Tesseract, and reads the recognized text aloud using neural text-to-speech. Designed for reading subtitles in movies, YouTube videos, games, and any on-screen text.

## Features

- **Screen region capture** — select any area of your screen with a visual overlay
- **Real-time OCR** — continuous screenshot capture with Tesseract OCR (PSM 6, LSTM engine)
- **Text-to-speech** — two TTS engines:
  - **edge-tts** (default) — cloud-based Microsoft Neural voices (`en-US-AndrewNeural`, `ru-RU-DmitryNeural`)
  - **Silero** — local neural TTS via PyTorch (Russian only, falls back to edge-tts for English)
- **Text isolation** — 3-step OpenCV pipeline that detects subtitle text, binarizes it, and cleans artifacts for a high-quality OCR input
- **Smart deduplication** — triple-level similarity check (Dice + Keyword + Jaccard at 85% threshold) prevents re-reading the same text
- **Language support** — English, Russian, or auto-detect (eng+rus)
- **Global hotkeys** — configurable keyboard shortcuts via pynput
- **Dual preview** — shows both the raw capture and the processed image that Tesseract receives
- **Garbage filtering** — removes OCR artifacts, mixed-script noise, and fake Cyrillic

## Requirements

- **OS**: Linux with X11 (Wayland is not supported)
- **Python**: 3.12+
- **[uv](https://docs.astral.sh/uv/)** package manager

### System dependencies

```bash
sudo apt install tesseract-ocr tesseract-ocr-eng tesseract-ocr-rus ffmpeg libxcb-xinerama0 libxcb-cursor0
```

| Package | Purpose |
|---------|---------|
| `tesseract-ocr` | OCR engine |
| `tesseract-ocr-eng`, `tesseract-ocr-rus` | Language data for Tesseract |
| `ffmpeg` | Audio playback for TTS |
| `libxcb-xinerama0`, `libxcb-cursor0` | PyQt6 X11 support |

## Installation

```bash
git clone https://github.com/your-username/CaptureRegionReader.git
cd CaptureRegionReader
uv sync
```

## Usage

```bash
uv run capture-region-reader
```

1. Click **Select Screen Region** (or press `Ctrl+Alt+S`) to draw a rectangle around the subtitles
2. Click **Start Reading** (or press `Ctrl+Alt+R`) to begin OCR + TTS
3. The app continuously captures the selected region, extracts text, and reads new text aloud
4. Press the hotkey again or click **Stop Reading** to stop

### Makefile shortcuts

```bash
make setup       # Install dependencies + check system packages
make run         # Run the app
make dev         # Run in dev mode (python -m)
make debug       # Run with CRR_DEBUG=1 (saves debug frames)
make clean       # Remove build artifacts
make check-deps  # Verify tesseract and ffmpeg are installed
```

## Configuration

| Setting | Description | Default |
|---------|-------------|---------|
| Start/Stop hotkey | Toggle OCR reading on/off | `Ctrl+Alt+R` |
| Select Region hotkey | Open region selection overlay | `Ctrl+Alt+S` |
| Language | OCR language (English, Russian, Auto) | Auto (eng+rus) |
| TTS Engine | edge-tts (cloud) or Silero (local, Russian only) | edge-tts |
| Speed | TTS speech rate (50–350 wpm) | 150 |
| Volume | TTS volume (0–100%) | 100% |
| OCR interval | Time between captures (200–3000 ms) | 500 ms |
| Settle time | Delay before first capture after start (ms) | 300 ms |

Settings are saved automatically between sessions.

## How Text Isolation Works

Raw screenshots contain backgrounds, logos, and UI elements that confuse OCR. The text isolator preprocesses each frame through a 3-step pipeline:

1. **Detection** (`subtitle_detector.py`) — finds character-like contours, groups them into text blocks by proximity, refines block boundaries, and crops around the detected subtitle region
2. **Binarization** (`subtitle_binarizer.py`) — applies Otsu thresholding and removes noise components to produce clean black text on white background
3. **Cleaning** (`subtitle_cleaner.py`) — connected component analysis groups characters into lines, removes stray artifacts, and produces the final cropped image

If the pipeline returns no result, the raw screenshot is passed directly to Tesseract as a fallback.

## Project Structure

```
src/capture_region_reader/
├── app.py                # Application controller, signal wiring
├── main_window.py        # PyQt6 UI (settings, preview, hotkey recorders)
├── region_selector.py    # Fullscreen overlay for drawing capture region
├── ocr_worker.py         # QThread: capture → isolate → OCR → emit text
├── text_isolator.py      # Wrapper for the 3-step isolation pipeline
├── subtitle_detector.py  # Step 1: character detection, block grouping, cropping
├── subtitle_binarizer.py # Step 2: Otsu binarization, noise removal
├── subtitle_cleaner.py   # Step 3: line grouping, artifact removal, final crop
├── text_cleaner.py       # Post-OCR cleanup: garbage filter, fake Cyrillic
├── text_differ.py        # Detects new/changed text between OCR results
├── tts_worker.py         # QThread: TTS synthesis + playback (edge-tts / Silero)
├── hotkey_manager.py     # Global hotkeys via pynput
└── settings.py           # Dataclass + QSettings persistence
```

## License

MIT
