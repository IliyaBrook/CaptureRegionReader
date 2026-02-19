from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
import time
import wave
from pathlib import Path
from queue import Empty, Queue
from typing import Protocol

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal


# edge-tts voice IDs
VOICE_EN = "en-US-AndrewNeural"
VOICE_RU = "ru-RU-DmitryNeural"

# Piper model directory (shared with ai-reader-assistant)
_PIPER_MODELS_DIR = Path("/mnt/DiskE_Crucial/codding/My_Projects/ai-reader-assistant/models/piper")
PIPER_VOICE_EN = "en_US-amy-medium"
PIPER_VOICE_RU = "ru_RU-irina-medium"


def detect_language(text: str) -> str:
    """Detect dominant language based on character analysis."""
    if not text:
        return "en"
    cyrillic = sum(1 for c in text if "\u0400" <= c <= "\u04FF")
    latin = sum(1 for c in text if "A" <= c <= "Z" or "a" <= c <= "z")
    total = cyrillic + latin
    if total == 0:
        return "en"
    return "ru" if cyrillic / total > 0.5 else "en"


# ── TTS Engine Protocol ────────────────────────────────────────────

class TtsEngine(Protocol):
    """Protocol for TTS backends."""

    def generate(self, text: str, lang: str, output_path: str) -> None:
        """Generate audio file from text.

        Args:
            text: Text to synthesize.
            lang: Language code ("en" or "ru").
            output_path: Path to write the audio file.
        """
        ...

    @property
    def audio_suffix(self) -> str:
        """File extension for generated audio ('.mp3' or '.wav')."""
        ...


# ── Edge-TTS Engine (cloud, requires internet) ────────────────────

class EdgeTtsEngine:
    """Microsoft Edge neural TTS via edge-tts library.

    High quality, requires internet connection.
    Generates MP3 files.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._rate: str = "+0%"
        self._volume: str = "+0%"

    @property
    def audio_suffix(self) -> str:
        return ".mp3"

    def set_rate(self, rate_pct: str) -> None:
        self._rate = rate_pct

    def set_volume(self, volume_pct: str) -> None:
        self._volume = volume_pct

    def generate(self, text: str, lang: str, output_path: str) -> None:
        import edge_tts

        if self._loop is None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

        voice = VOICE_RU if lang == "ru" else VOICE_EN
        communicate = edge_tts.Communicate(
            text, voice, rate=self._rate, volume=self._volume
        )
        self._loop.run_until_complete(communicate.save(output_path))

    def close(self) -> None:
        if self._loop is not None:
            self._loop.close()
            self._loop = None


# ── Piper TTS Engine (local ONNX, offline) ────────────────────────

class PiperTtsEngine:
    """Piper TTS — fast local neural TTS using ONNX runtime.

    Works offline, no internet required.
    Generates WAV files.
    Models: en_US-amy-medium, ru_RU-irina-medium.
    """

    def __init__(self) -> None:
        self._voice_en = None
        self._voice_ru = None
        self._loaded = False

    @property
    def audio_suffix(self) -> str:
        return ".wav"

    def _ensure_loaded(self, lang: str) -> None:
        """Lazy-load Piper voices on first use."""
        if lang == "ru" and self._voice_ru is not None:
            return
        if lang == "en" and self._voice_en is not None:
            return

        from piper import PiperVoice

        if lang == "ru" and self._voice_ru is None:
            model = _PIPER_MODELS_DIR / f"{PIPER_VOICE_RU}.onnx"
            config = _PIPER_MODELS_DIR / f"{PIPER_VOICE_RU}.onnx.json"
            if model.exists() and config.exists():
                self._voice_ru = PiperVoice.load(str(model), str(config))
                print(f"[Piper] Loaded Russian voice: {PIPER_VOICE_RU}")
            else:
                raise FileNotFoundError(
                    f"Piper Russian model not found at {model}"
                )

        if lang == "en" and self._voice_en is None:
            model = _PIPER_MODELS_DIR / f"{PIPER_VOICE_EN}.onnx"
            config = _PIPER_MODELS_DIR / f"{PIPER_VOICE_EN}.onnx.json"
            if model.exists() and config.exists():
                self._voice_en = PiperVoice.load(str(model), str(config))
                print(f"[Piper] Loaded English voice: {PIPER_VOICE_EN}")
            else:
                raise FileNotFoundError(
                    f"Piper English model not found at {model}"
                )

    def generate(self, text: str, lang: str, output_path: str) -> None:
        self._ensure_loaded(lang)

        voice = self._voice_ru if lang == "ru" else self._voice_en
        if voice is None:
            raise RuntimeError(f"Piper voice for '{lang}' not loaded")

        with wave.open(output_path, "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)

    def close(self) -> None:
        self._voice_en = None
        self._voice_ru = None


# ── Engine factory ─────────────────────────────────────────────────

def create_tts_engine(name: str) -> TtsEngine:
    """Create a TTS engine by name.

    Args:
        name: "edge-tts" or "piper".

    Returns:
        A TTS engine instance.
    """
    if name == "piper":
        try:
            import piper  # noqa: F401
            return PiperTtsEngine()
        except ImportError:
            print("[TTS] piper-tts not installed, falling back to edge-tts")
            return EdgeTtsEngine()
    return EdgeTtsEngine()


# ── TTS Worker Thread ──────────────────────────────────────────────

class TtsWorker(QThread):
    speech_started = pyqtSignal()
    speech_finished = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._queue: Queue[str | None] = Queue()
        self._rate_wpm: int = 150
        self._speech_rate: str = "+0%"  # edge-tts rate format
        self._volume: str = "+0%"
        self._volume_float: float = 1.0
        self._auto_language: bool = True
        self._fixed_language: str = "en"
        self._cancel_requested: bool = False
        self._current_process: subprocess.Popen | None = None
        self._engine: TtsEngine = EdgeTtsEngine()
        self._engine_name: str = "edge-tts"

    def set_engine(self, name: str) -> None:
        """Switch TTS engine at runtime.

        Called from main thread — the engine is recreated on the TTS thread
        via a flag, since Piper models must be loaded on the worker thread.
        """
        if name == self._engine_name:
            return
        self._engine_name = name
        # Close old engine
        if hasattr(self._engine, "close"):
            self._engine.close()
        self._engine = create_tts_engine(name)
        # Re-apply rate/volume to the new engine
        self._apply_rate_volume()
        print(f"[TTS] Switched to engine: {name}")

    def set_rate(self, rate: int) -> None:
        self._rate_wpm = rate
        # Convert wpm slider (50-350, default 150) to edge-tts percentage
        # 150 wpm = +0%, 50 wpm = -67%, 350 wpm = +133%
        pct = int((rate - 150) / 150 * 100)
        self._speech_rate = f"{pct:+d}%"
        self._apply_rate_volume()

    def set_volume(self, volume: float) -> None:
        self._volume_float = volume
        # Convert 0.0-1.0 to edge-tts format
        # 1.0 = +0%, 0.5 = -50%, 0.0 = -100%
        pct = int((volume - 1.0) * 100)
        self._volume = f"{pct:+d}%"
        self._apply_rate_volume()

    def _apply_rate_volume(self) -> None:
        """Forward rate/volume to the active engine (if it supports it)."""
        if isinstance(self._engine, EdgeTtsEngine):
            self._engine.set_rate(self._speech_rate)
            self._engine.set_volume(self._volume)

    def set_language(self, lang: str) -> None:
        if lang == "eng+rus":
            self._auto_language = True
        else:
            self._auto_language = False
            self._fixed_language = {"eng": "en", "rus": "ru"}.get(lang, "en")

    def speak(self, text: str) -> None:
        self._cancel_requested = False
        self._queue.put(text)

    def cancel(self) -> None:
        self._cancel_requested = True
        # Kill current playback if any
        if self._current_process and self._current_process.poll() is None:
            self._current_process.terminate()
        # Drain the queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Empty:
                break

    def run(self) -> None:
        while True:
            try:
                text = self._queue.get(timeout=0.5)
            except Empty:
                continue

            if text is None:
                break

            if self._cancel_requested:
                continue

            try:
                # Determine language
                if self._auto_language:
                    lang = detect_language(text)
                else:
                    lang = self._fixed_language

                self.speech_started.emit()
                print(f"[TTS/{self._engine_name}] Generating: {repr(text[:60])}")

                # Generate audio
                suffix = self._engine.audio_suffix
                tmp_file = tempfile.mktemp(suffix=suffix)
                try:
                    t0 = time.monotonic()
                    self._engine.generate(text, lang, tmp_file)
                    gen_ms = int((time.monotonic() - t0) * 1000)
                    print(f"[TTS/{self._engine_name}] Generated in {gen_ms}ms, playing...")

                    if self._cancel_requested:
                        continue

                    # Play audio with ffplay
                    self._current_process = subprocess.Popen(
                        [
                            "ffplay",
                            "-nodisp",
                            "-autoexit",
                            "-loglevel",
                            "error",
                            tmp_file,
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    self._current_process.wait()
                    self._current_process = None
                finally:
                    try:
                        os.unlink(tmp_file)
                    except OSError:
                        pass

                self.speech_finished.emit()
            except Exception as e:
                self.error_occurred.emit(str(e))

    def shutdown(self) -> None:
        self.cancel()
        self._queue.put(None)
        self.wait(5000)
        if hasattr(self._engine, "close"):
            self._engine.close()
