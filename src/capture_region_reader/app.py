from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from capture_region_reader.hotkey_manager import HotkeyManager
from capture_region_reader.main_window import MainWindow
from capture_region_reader.ocr_worker import OcrWorker
from capture_region_reader.region_selector import RegionSelector
from capture_region_reader.settings import AppSettings
from capture_region_reader.text_differ import TextDiffer
from capture_region_reader.tts_worker import TtsWorker


class App:
    def __init__(self) -> None:
        self._qt_app = QApplication(sys.argv)
        self._qt_app.setApplicationName("CaptureRegionReader")
        self._qt_app.setOrganizationName("CaptureRegionReader")

        self._settings = AppSettings.load()

        # Workers
        self._ocr_worker = OcrWorker()
        self._tts_worker = TtsWorker()
        self._text_differ = TextDiffer()
        self._hotkey_manager = HotkeyManager()

        # UI
        self._window = MainWindow(self._settings)
        self._region_selector: RegionSelector | None = None

        self._is_reading = False

        self._connect_signals()
        self._apply_settings()

    def _connect_signals(self) -> None:
        w = self._window

        # Region selection
        w.select_region_clicked.connect(self._on_select_region)

        # Toggle reading
        w.toggle_reading.connect(self._on_toggle_reading)
        self._hotkey_manager.hotkey_triggered.connect(self._on_toggle_reading)

        # OCR results
        self._ocr_worker.text_recognized.connect(self._on_text_recognized)
        self._ocr_worker.error_occurred.connect(w.show_error)

        # TTS status
        self._tts_worker.speech_started.connect(
            lambda: w.show_status("Speaking...")
        )
        self._tts_worker.speech_finished.connect(
            lambda: w.show_status("Reading..." if self._is_reading else "Ready")
        )
        self._tts_worker.error_occurred.connect(w.show_error)

        # Settings changes
        w.language_changed.connect(self._on_language_changed)
        w.rate_changed.connect(self._tts_worker.set_rate)
        w.volume_changed.connect(self._tts_worker.set_volume)
        w.hotkey_changed.connect(self._hotkey_manager.set_hotkey)
        w.interval_changed.connect(self._ocr_worker.set_interval)

    def _apply_settings(self) -> None:
        s = self._settings
        self._tts_worker.set_rate(s.speech_rate)
        self._tts_worker.set_volume(s.volume)
        self._tts_worker.set_language(s.language)
        self._ocr_worker.set_language(s.language)
        self._ocr_worker.set_interval(s.ocr_interval_ms)
        self._hotkey_manager.set_hotkey(s.hotkey)

        # Start TTS thread (waits on queue)
        self._tts_worker.start()

    def _on_select_region(self) -> None:
        was_reading = self._is_reading
        if was_reading:
            self._stop_reading()

        self._region_selector = RegionSelector()
        self._region_selector.region_selected.connect(
            lambda l, t, w, h: self._on_region_selected(l, t, w, h, was_reading)
        )
        self._region_selector.showFullScreen()

    def _on_region_selected(
        self, left: int, top: int, width: int, height: int, restart: bool
    ) -> None:
        self._window.on_region_selected(left, top, width, height)
        self._text_differ.reset()
        if restart:
            self._start_reading()

    def _on_toggle_reading(self) -> None:
        if self._is_reading:
            self._stop_reading()
        else:
            self._start_reading()

    def _start_reading(self) -> None:
        if not self._settings.region:
            self._window.show_status("Select a region first!")
            return

        self._is_reading = True
        self._text_differ.reset()
        self._window.set_reading_state(True)
        self._window.show_status("Reading...")

        self._ocr_worker.configure(
            region=self._settings.region,
            language=self._settings.language,
            interval_ms=self._settings.ocr_interval_ms,
        )
        self._ocr_worker.start()

    def _stop_reading(self) -> None:
        self._is_reading = False
        self._ocr_worker.stop()
        self._tts_worker.cancel()
        self._window.set_reading_state(False)
        self._window.show_status("Stopped")

    def _on_text_recognized(self, text: str) -> None:
        self._window.update_text_display(text)

        new_text = self._text_differ.get_new_text(text)
        if new_text:
            self._tts_worker.speak(new_text)

    def _on_language_changed(self, lang: str) -> None:
        self._ocr_worker.set_language(lang)
        self._tts_worker.set_language(lang)

    def run(self) -> int:
        self._window.show()
        exit_code = self._qt_app.exec()

        # Cleanup
        self._ocr_worker.stop()
        self._tts_worker.shutdown()
        self._hotkey_manager.stop()
        self._settings.save()

        return exit_code


def main() -> None:
    app = App()
    sys.exit(app.run())
