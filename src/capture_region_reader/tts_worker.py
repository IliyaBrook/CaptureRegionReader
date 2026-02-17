from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
from queue import Empty, Queue

from PyQt6.QtCore import QThread, pyqtSignal


# edge-tts voice IDs
VOICE_EN = "en-US-AndrewNeural"
VOICE_RU = "ru-RU-DmitryNeural"


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


class TtsWorker(QThread):
    speech_started = pyqtSignal()
    speech_finished = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._queue: Queue[str | None] = Queue()
        self._speech_rate: str = "+0%"  # edge-tts rate format
        self._volume: str = "+0%"
        self._auto_language: bool = True
        self._fixed_language: str = "en"
        self._cancel_requested: bool = False
        self._current_process: subprocess.Popen | None = None

    def set_rate(self, rate: int) -> None:
        # Convert wpm slider (50-350, default 150) to edge-tts percentage
        # 150 wpm = +0%, 50 wpm = -67%, 350 wpm = +133%
        pct = int((rate - 150) / 150 * 100)
        self._speech_rate = f"{pct:+d}%"

    def set_volume(self, volume: float) -> None:
        # Convert 0.0-1.0 to edge-tts format
        # 1.0 = +0%, 0.5 = -50%, 0.0 = -100%
        pct = int((volume - 1.0) * 100)
        self._volume = f"{pct:+d}%"

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
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

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
                # Determine voice
                if self._auto_language:
                    lang = detect_language(text)
                else:
                    lang = self._fixed_language

                voice = VOICE_RU if lang == "ru" else VOICE_EN

                self.speech_started.emit()

                # Generate audio with edge-tts
                tmp_file = tempfile.mktemp(suffix=".mp3")
                try:
                    loop.run_until_complete(
                        self._generate_audio(text, voice, tmp_file)
                    )

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

        loop.close()

    async def _generate_audio(self, text: str, voice: str, output_path: str) -> None:
        import edge_tts

        communicate = edge_tts.Communicate(
            text,
            voice,
            rate=self._speech_rate,
            volume=self._volume,
        )
        await communicate.save(output_path)

    def shutdown(self) -> None:
        self.cancel()
        self._queue.put(None)
        self.wait(5000)
