from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
import tempfile
import time
import wave
from queue import Empty, Queue
from typing import Protocol

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

logger = logging.getLogger(__name__)


# edge-tts voice IDs
VOICE_EN = "en-US-AndrewNeural"
VOICE_RU = "ru-RU-DmitryNeural"

# Silero TTS settings (same as ai-reader-assistant)
SILERO_MODEL_ID = "v4_ru"
SILERO_SPEAKER = "xenia"
SILERO_SAMPLE_RATE = 48000


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


# ── Silero TTS Engine (local, GPU-accelerated) ────────────────────

class SileroTtsEngine:
    """Silero TTS v4 — high-quality local neural TTS.

    Uses torch.hub to download models on first use.
    Russian: speaker 'xenia' at 48kHz — expressive, natural voice.
    English: falls back to edge-tts (Silero v4 is Russian-only).
    Works offline after first download.
    """

    def __init__(self) -> None:
        self._model = None
        self._device = None
        self._edge_fallback: EdgeTtsEngine | None = None

    @property
    def audio_suffix(self) -> str:
        return ".wav"

    def _ensure_loaded(self) -> None:
        """Lazy-load Silero model on first use."""
        if self._model is not None:
            return

        import torch

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Silero] Loading model {SILERO_MODEL_ID} on {self._device}...")

        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language="ru",
            speaker=SILERO_MODEL_ID,
        )
        model.to(self._device)
        self._model = model
        print(f"[Silero] Model loaded on {self._device}")

    def generate(self, text: str, lang: str, output_path: str) -> None:
        # Silero v4 only supports Russian; for English fall back to edge-tts
        if lang != "ru":
            if self._edge_fallback is None:
                self._edge_fallback = EdgeTtsEngine()
                print("[Silero] English text → falling back to Edge-TTS")
            # Edge-TTS generates MP3, but we need WAV suffix consistency
            # Just generate MP3 with edge-tts and let ffplay handle it
            mp3_path = output_path.replace(".wav", ".mp3")
            self._edge_fallback.generate(text, lang, mp3_path)
            # Rename so caller finds the file at output_path
            os.rename(mp3_path, output_path)
            return

        self._ensure_loaded()

        import torch
        import scipy.io.wavfile

        # Split into sentences for better intonation
        sentences = re.split(r"(?<=[.!?])\s+", text)
        audio_chunks = []

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            audio = self._model.apply_tts(
                text=sentence,
                speaker=SILERO_SPEAKER,
                sample_rate=SILERO_SAMPLE_RATE,
            )
            audio_np = audio.cpu().numpy() if torch.is_tensor(audio) else audio
            audio_chunks.append(audio_np)

        if audio_chunks:
            full_audio = np.concatenate(audio_chunks)
            scipy.io.wavfile.write(output_path, SILERO_SAMPLE_RATE, full_audio)
        else:
            # Empty audio — write a silent WAV
            with wave.open(output_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SILERO_SAMPLE_RATE)
                wf.writeframes(b"")

    def close(self) -> None:
        self._model = None
        self._device = None
        if self._edge_fallback is not None:
            self._edge_fallback.close()
            self._edge_fallback = None



# ── Engine factory ─────────────────────────────────────────────────

def create_tts_engine(name: str) -> TtsEngine:
    """Create a TTS engine by name.

    Args:
        name: "edge-tts" or "silero".

    Returns:
        A TTS engine instance.

    Raises:
        ImportError: if the requested engine's dependencies are missing.
        The caller is responsible for handling the error and informing the user.
    """
    if name == "silero":
        import torch  # noqa: F401
        return SileroTtsEngine()
    if name == "edge-tts":
        import edge_tts  # noqa: F401
        return EdgeTtsEngine()
    raise ValueError(f"Unknown TTS engine: {name}")


# ── TTS Worker Thread ──────────────────────────────────────────────

class TtsWorker(QThread):
    speech_started = pyqtSignal()
    speech_finished = pyqtSignal()
    error_occurred = pyqtSignal(str)
    engine_unavailable = pyqtSignal(str, str)  # (engine_name, error_message)

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
        self._engine: TtsEngine | None = None
        self._engine_name: str = ""  # empty so first set_engine() always triggers

    def set_engine(self, name: str) -> None:
        """Switch TTS engine at runtime.

        If the requested engine is not installed, emits engine_unavailable
        signal so the UI can revert the combo box and inform the user.
        Does NOT silently fall back to another engine.
        """
        if name == self._engine_name:
            return
        try:
            old_name = self._engine_name
            new_engine = create_tts_engine(name)
            # Close old engine only after new one is successfully created
            if self._engine is not None and hasattr(self._engine, "close"):
                self._engine.close()
            self._engine = new_engine
            self._engine_name = name
            # Re-apply rate/volume to the new engine
            self._apply_rate_volume()
            logger.info("TTS engine switched: %s -> %s", old_name, name)
        except (ImportError, ValueError) as e:
            logger.error("Failed to switch TTS to %s: %s", name, e)
            # Emit detailed error so the UI can show a dialog
            self.engine_unavailable.emit(name, str(e))
            # Do NOT silently fall back. If no engine loaded at all, load default.
            if self._engine is None:
                try:
                    self._engine = EdgeTtsEngine()
                    self._engine_name = "edge-tts"
                    logger.info("No TTS engine loaded, defaulting to edge-tts")
                except Exception:
                    logger.error("Could not load any TTS engine")

    def set_rate(self, rate: int) -> None:
        self._rate_wpm = rate
        pct = int((rate - 150) / 150 * 100)
        self._speech_rate = f"{pct:+d}%"
        self._apply_rate_volume()

    def set_volume(self, volume: float) -> None:
        self._volume_float = volume
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
        if self._engine is not None and hasattr(self._engine, "close"):
            self._engine.close()
