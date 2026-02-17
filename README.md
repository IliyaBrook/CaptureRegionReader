# CaptureRegionReader

A desktop application that captures a screen region, performs real-time OCR, and reads the recognized text aloud using text-to-speech. Designed for reading subtitles in movies, YouTube videos, games, and any on-screen text.

## Features

- **Screen region capture** — select any area of your screen with a visual overlay
- **Real-time OCR** — continuous screenshot capture with Tesseract OCR
- **Text-to-speech** — reads recognized text aloud using Microsoft Neural voices (edge-tts)
- **Text isolation** — OpenCV-based preprocessing that detects subtitle text, crops around it, filters background noise by brightness, and creates a clean image for OCR
- **Smart deduplication** — triple-level dedup (OCR, text differ, TTS queue) prevents re-reading the same text
- **Language support** — English, Russian, or auto-detect (eng+rus)
- **Global hotkeys** — configurable keyboard shortcuts for start/stop and region selection
- **Capture preview** — shows the processed image that Tesseract receives, useful for debugging
- **Garbage filtering** — removes OCR artifacts, mixed-script noise, and fake Cyrillic

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Tesseract OCR with language data:
  ```bash
  sudo apt install tesseract-ocr tesseract-ocr-eng tesseract-ocr-rus
  ```
- System dependencies for PyQt6:
  ```bash
  sudo apt install libxcb-xinerama0 libxcb-cursor0
  ```

## Installation

```bash
cd /path/to/CaptureRegionReader
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

## Configuration

| Setting | Description | Default |
|---------|-------------|---------|
| Start/Stop hotkey | Toggle OCR reading on/off | `Ctrl+Alt+R` |
| Select Region hotkey | Open region selection overlay | `Ctrl+Alt+S` |
| Language | OCR language (English, Russian, Auto) | Auto (eng+rus) |
| Speed | TTS speech rate (50–350 wpm) | 150 |
| Volume | TTS volume (0–100%) | 100% |
| OCR interval | Time between captures (200–3000 ms) | 500 ms |

Settings are saved automatically between sessions.

## How Text Isolation Works

Raw screenshots contain background images, logos, UI elements, and semi-transparent overlays that confuse OCR. The text isolator preprocesses each frame:

1. **Otsu threshold** — finds bright (or dark) regions as text candidates
2. **Brightness filtering** — measures each contour's pixel brightness, keeps only the dominant brightness group (e.g., white subtitles at ~240 brightness, discards grey background text at ~120)
3. **Line clustering** — groups character boxes into horizontal text lines by vertical center proximity
4. **Dense core extraction** — within each line, finds the longest run of tightly-spaced characters, removes distant outliers (logos, icons)
5. **Line scoring** — ranks lines by character count, width, height consistency, density, and vertical position (bottom = subtitle)
6. **Crop & threshold** — crops to the text bounding box, applies Otsu binarization for a clean black-on-white result

## Project Structure

```
src/capture_region_reader/
├── app.py              # Application controller, signal wiring
├── main_window.py      # PyQt6 UI (settings, preview, hotkey recorders)
├── region_selector.py  # Fullscreen overlay for drawing capture region
├── ocr_worker.py       # QThread: capture → isolate → OCR → emit text
├── text_isolator.py    # OpenCV text detection, brightness filtering, cropping
├── text_cleaner.py     # Post-OCR cleanup: garbage filter, fake Cyrillic, TTS prep
├── text_differ.py      # Detects new/changed text between consecutive OCR results
├── tts_worker.py       # QThread: edge-tts speech synthesis with queue management
├── hotkey_manager.py   # Global hotkeys via pynput (start/stop + select region)
└── settings.py         # Dataclass + QSettings persistence
```

## License

MIT
