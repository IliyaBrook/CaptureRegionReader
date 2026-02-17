from __future__ import annotations

from queue import Empty, Queue

from PyQt6.QtCore import QThread, pyqtSignal


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
        self._speech_rate: int = 150
        self._volume: float = 1.0
        self._auto_language: bool = True
        self._fixed_language: str = "en"
        self._cancel_requested: bool = False

    def set_rate(self, rate: int) -> None:
        self._speech_rate = rate

    def set_volume(self, volume: float) -> None:
        self._volume = volume

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
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Empty:
                break

    def run(self) -> None:
        import pyttsx3

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
                engine = pyttsx3.init()
                engine.setProperty("rate", self._speech_rate)
                engine.setProperty("volume", self._volume)

                # Select voice based on language
                if self._auto_language:
                    target_lang = detect_language(text)
                else:
                    target_lang = self._fixed_language

                voices = engine.getProperty("voices")
                # espeak-ng voice IDs: "gmw/en-us", "zle/ru", etc.
                best_voice = None
                for voice in voices:
                    vid = voice.id.lower()
                    if target_lang == "ru" and "/ru" in vid:
                        best_voice = voice.id
                        break
                    elif target_lang == "en" and "/en-us" in vid:
                        best_voice = voice.id
                        break
                    elif target_lang == "en" and "/en" in vid and not best_voice:
                        best_voice = voice.id
                if best_voice:
                    engine.setProperty("voice", best_voice)

                self.speech_started.emit()
                engine.say(text)
                engine.runAndWait()
                engine.stop()
                self.speech_finished.emit()
            except Exception as e:
                self.error_occurred.emit(str(e))

    def shutdown(self) -> None:
        self._queue.put(None)
        self.wait(3000)
