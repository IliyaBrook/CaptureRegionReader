from __future__ import annotations

import sys

import numpy as np
from mss import mss
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from capture_region_reader.debug_service import DebugService, is_debug_enabled
from capture_region_reader.hotkey_manager import HotkeyManager
from capture_region_reader.main_window import MainWindow
from capture_region_reader.ocr_worker import OcrWorker
from capture_region_reader.region_selector import RegionOverlay, RegionSelector
from capture_region_reader.settings import AppSettings, CaptureZone
from capture_region_reader.text_cleaner import clean_for_tts, filter_by_language
from capture_region_reader.text_differ import TextDiffer
from capture_region_reader.tts_worker import TtsWorker


class App:
    def __init__(self) -> None:
        self._qt_app = QApplication(sys.argv)
        self._qt_app.setApplicationName("CaptureRegionReader")
        self._qt_app.setOrganizationName("CaptureRegionReader")

        self._settings = AppSettings.load()

        self._debug: DebugService | None = None
        if is_debug_enabled():
            self._debug = DebugService()

        # Workers
        self._ocr_worker = OcrWorker()
        self._tts_worker = TtsWorker()
        self._text_differ = TextDiffer()
        self._hotkey_manager = HotkeyManager()

        # UI
        self._window = MainWindow(self._settings)
        self._region_selectors: list[RegionSelector] = []
        self._region_overlay = RegionOverlay()

        self._is_reading = False
        self._show_region_on = False
        self._zone_select_pending: int | None = None  # zone index awaiting region selection

        self._connect_signals()
        self._apply_settings()

    def _connect_signals(self) -> None:
        w = self._window

        # Region selection
        w.select_region_clicked.connect(self._on_select_region)
        w.show_region_toggled.connect(self._on_show_region_toggled)

        # Toggle reading
        w.toggle_reading.connect(self._on_toggle_reading)
        self._hotkey_manager.hotkey_triggered.connect(self._on_toggle_reading)

        # Select region hotkey
        self._hotkey_manager.select_region_triggered.connect(self._on_select_region)

        # OCR results
        self._ocr_worker.text_recognized.connect(self._on_text_recognized)
        self._ocr_worker.frame_captured.connect(w.update_capture_preview)
        self._ocr_worker.raw_frame_captured.connect(w.update_raw_preview)
        self._ocr_worker.error_occurred.connect(w.show_error)

        # Debug: cache frames for per-subtitle artifact saving
        if self._debug:
            self._ocr_worker.raw_frame_captured.connect(self._debug.cache_raw_frame)
            self._ocr_worker.frame_captured.connect(self._debug.cache_processed_frame)

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
        w.select_region_hotkey_changed.connect(self._hotkey_manager.set_select_region_hotkey)
        w.interval_changed.connect(self._ocr_worker.set_interval)
        w.tts_engine_changed.connect(self._tts_worker.set_engine)
        w.growing_subtitles_changed.connect(self._text_differ.set_growing_subtitles)

        # Engine unavailable errors — show dialog and revert combo box
        self._tts_worker.engine_unavailable.connect(w.revert_tts_engine)

        # Zone management
        w.zone_added.connect(self._on_zone_added)
        w.zone_deleted.connect(self._on_zone_deleted)
        w.zone_activated.connect(self._on_zone_activated)
        w.zone_select_region.connect(self._on_zone_select_region)
        w.zone_hotkey_changed.connect(self._on_zone_hotkey_changed)
        w.zone_name_changed.connect(self._on_zone_name_changed)

        # Zone hotkey from HotkeyManager
        self._hotkey_manager.zone_hotkey_triggered.connect(self._on_zone_activated)

    def _apply_settings(self) -> None:
        s = self._settings
        self._tts_worker.set_engine(s.tts_engine)
        self._tts_worker.set_rate(s.speech_rate)
        self._tts_worker.set_volume(s.volume)
        self._tts_worker.set_language(s.language)
        self._ocr_worker.set_language(s.language)
        self._ocr_worker.set_interval(s.ocr_interval_ms)
        self._text_differ.set_growing_subtitles(s.growing_subtitles)
        self._hotkey_manager.set_hotkey(s.hotkey)
        self._hotkey_manager.set_select_region_hotkey(s.select_region_hotkey)

        # Start TTS thread (waits on queue)
        self._tts_worker.start()

        # Initialize zone hotkeys
        self._sync_zone_hotkeys()

        # Build zone rows in the UI
        self._window.rebuild_zones(s.zones, s.active_zone_index)

        # If a zone was active, show its name in the region label
        if (
            s.active_zone_index is not None
            and s.active_zone_index < len(s.zones)
            and s.region
        ):
            zone = s.zones[s.active_zone_index]
            self._window.update_region_display(s.region, zone.name)

        # Show preview for saved region on startup
        if s.region:
            self._grab_single_preview(*s.region)

    # ------------------------------------------------------------------
    # Region selection
    # ------------------------------------------------------------------

    def _on_select_region(self) -> None:
        was_reading = self._is_reading
        if was_reading:
            self._stop_reading()

        self._close_all_selectors()

        # Capture full virtual desktop BEFORE showing overlays (mss reads
        # the X11 framebuffer directly, so fullscreen games are captured).
        from PyQt6.QtGui import QImage, QPixmap

        full_pixmap: QPixmap | None = None
        desktop_left = 0
        desktop_top = 0
        try:
            with mss() as sct:
                desktop = sct.monitors[0]  # full virtual desktop
                desktop_left = desktop["left"]
                desktop_top = desktop["top"]
                screenshot = sct.grab(desktop)
                w, h = screenshot.size
                raw_rgb = screenshot.rgb
                qimage = QImage(
                    raw_rgb, w, h, w * 3, QImage.Format.Format_RGB888,
                )
                full_pixmap = QPixmap.fromImage(qimage)
        except Exception:
            pass

        # Create one selector window per screen (Spectacle approach).
        # Each gets a cropped portion of the desktop screenshot matching
        # its physical pixel area — no cross-screen coordinate issues.
        screens = self._qt_app.screens()
        for screen in screens:
            screen_bg: QPixmap | None = None
            if full_pixmap:
                geo = screen.geometry()
                dpr = screen.devicePixelRatio()
                # geo.x()/y() are PHYSICAL on X11; size is LOGICAL
                crop_x = int(geo.x() - desktop_left)
                crop_y = int(geo.y() - desktop_top)
                phys_w = int(geo.width() * dpr)
                phys_h = int(geo.height() * dpr)
                screen_bg = full_pixmap.copy(crop_x, crop_y, phys_w, phys_h)

            selector = RegionSelector(screen=screen, background=screen_bg)
            selector.region_selected.connect(
                lambda l, t, w, h, _r=was_reading: self._on_region_selected(
                    l, t, w, h, _r,
                )
            )
            selector.cancelled.connect(self._close_all_selectors)
            self._region_selectors.append(selector)

        for selector in self._region_selectors:
            selector.show()
            selector.raise_()

    def _close_all_selectors(self) -> None:
        """Close every per-screen selector window."""
        for s in self._region_selectors:
            s.close()
        self._region_selectors.clear()
        self._zone_select_pending = None

    def _on_region_selected(
        self, left: int, top: int, width: int, height: int, restart: bool
    ) -> None:
        # Save pending zone index BEFORE closing selectors (which resets it)
        pending_zone = self._zone_select_pending

        self._close_all_selectors()

        if pending_zone is not None:
            # Zone-specific region selection
            zone_idx = pending_zone

            if zone_idx < len(self._settings.zones):
                self._settings.zones[zone_idx].region = (left, top, width, height)

                # If this zone is currently active, also update the active region
                if self._settings.active_zone_index == zone_idx:
                    self._settings.region = (left, top, width, height)
                    self._window.update_region_display(
                        (left, top, width, height),
                        self._settings.zones[zone_idx].name,
                    )

                self._window.rebuild_zones(
                    self._settings.zones, self._settings.active_zone_index
                )
                self._window.show_status(
                    f'Region set for zone "{self._settings.zones[zone_idx].name}"'
                )
        else:
            # Manual selection — clear active zone
            self._settings.active_zone_index = None
            self._window.on_region_selected(left, top, width, height)
            self._window.rebuild_zones(self._settings.zones, None)

        self._text_differ.reset()
        self._grab_single_preview(left, top, width, height)

        # Update overlay position if it's currently visible
        if self._region_overlay.isVisible():
            self._region_overlay.set_region(left, top, width, height)
        if restart:
            self._start_reading()

    def _on_show_region_toggled(self, checked: bool) -> None:
        self._show_region_on = checked
        if checked and self._settings.region:
            self._region_overlay.set_region(*self._settings.region)
            self._region_overlay.show()
        else:
            self._region_overlay.hide()

    def _grab_single_preview(self, left: int, top: int, width: int, height: int) -> None:
        """Capture a single frame for the preview right after region selection.

        Shows the processed image so user can see what OCR will receive
        — useful for debugging text detection.
        """
        from capture_region_reader.text_isolator import isolate_text
        from capture_region_reader.ocr_worker import _upscale
        from PIL import Image

        try:
            with mss() as sct:
                monitor = {"left": left, "top": top, "width": width, "height": height}
                screenshot = sct.grab(monitor)
                img_array = np.array(screenshot, dtype=np.uint8)
                raw_rgb = img_array[:, :, :3][:, :, ::-1].copy()

                # Show raw frame
                raw_h, raw_w = raw_rgb.shape[:2]
                self._window.update_raw_preview(raw_rgb.tobytes(), raw_w, raw_h)

                # Text isolation → upscale
                isolated = isolate_text(raw_rgb)
                if isolated is not None:
                    preview_img = _upscale(Image.fromarray(isolated))
                else:
                    preview_img = Image.fromarray(raw_rgb)

                preview_rgb = np.array(preview_img)
                p_h, p_w = preview_rgb.shape[:2]
                self._window.update_capture_preview(preview_rgb.tobytes(), p_w, p_h)
        except Exception:
            pass  # non-critical, preview will update when reading starts

    # ------------------------------------------------------------------
    # Zone management
    # ------------------------------------------------------------------

    def _on_zone_added(self) -> None:
        new_zone = CaptureZone(name=f"Zone {len(self._settings.zones) + 1}")
        self._settings.zones.append(new_zone)
        self._window.rebuild_zones(
            self._settings.zones, self._settings.active_zone_index
        )
        self._sync_zone_hotkeys()

    def _on_zone_deleted(self, zone_index: int) -> None:
        zones = self._settings.zones
        if zone_index < 0 or zone_index >= len(zones):
            return

        # If deleting the active zone, fall back to manual mode
        if self._settings.active_zone_index == zone_index:
            self._settings.active_zone_index = None
            self._window.show_status("Active zone deleted, using last region")
        elif (
            self._settings.active_zone_index is not None
            and self._settings.active_zone_index > zone_index
        ):
            # Shift index down since a zone before it was removed
            self._settings.active_zone_index -= 1

        zones.pop(zone_index)
        self._window.rebuild_zones(zones, self._settings.active_zone_index)
        self._sync_zone_hotkeys()

    def _on_zone_activated(self, zone_index: int) -> None:
        zones = self._settings.zones
        if zone_index < 0 or zone_index >= len(zones):
            return
        zone = zones[zone_index]
        if zone.region is None:
            self._window.show_status(f'Zone "{zone.name}" has no region set')
            return

        # Cancel pending TTS from old zone
        self._tts_worker.cancel()
        self._text_differ.reset()

        # Update the active region
        self._settings.region = zone.region
        self._settings.active_zone_index = zone_index

        # Update UI
        self._window.update_region_display(zone.region, zone.name)
        self._window.rebuild_zones(zones, zone_index)
        self._window.show_status(f'Switched to zone "{zone.name}"')

        # Visual feedback: flash the overlay for ~1 second
        self._flash_region_overlay(zone.region)

        # If reading is active, hot-swap the region (no stop/restart needed)
        if self._is_reading:
            self._ocr_worker.configure(
                region=zone.region,
                language=self._settings.language,
                interval_ms=self._settings.ocr_interval_ms,
            )

        # Grab preview for the new region
        self._grab_single_preview(*zone.region)

    def _on_zone_select_region(self, zone_index: int) -> None:
        """Start region selection for a specific zone."""
        # Call _on_select_region first (it calls _close_all_selectors which
        # resets _zone_select_pending), then set the pending index.
        self._on_select_region()
        self._zone_select_pending = zone_index

    def _on_zone_hotkey_changed(self, zone_index: int, hotkey: str) -> None:
        if zone_index < len(self._settings.zones):
            self._settings.zones[zone_index].hotkey = hotkey
            self._sync_zone_hotkeys()

    def _on_zone_name_changed(self, zone_index: int, name: str) -> None:
        if zone_index < len(self._settings.zones):
            self._settings.zones[zone_index].name = name
            # If this zone is active, update the region label
            if self._settings.active_zone_index == zone_index and self._settings.region:
                self._window.update_region_display(self._settings.region, name)

    def _sync_zone_hotkeys(self) -> None:
        """Rebuild the HotkeyManager's zone hotkey map from settings."""
        zone_map = {
            i: z.hotkey
            for i, z in enumerate(self._settings.zones)
            if z.hotkey
        }
        self._hotkey_manager.set_zone_hotkeys(zone_map)

    def _flash_region_overlay(self, region: tuple[int, int, int, int]) -> None:
        """Flash the region overlay for ~1 second, then hide."""
        self._region_overlay.set_region(*region)
        self._region_overlay.show()
        QTimer.singleShot(1000, self._on_flash_timeout)

    def _on_flash_timeout(self) -> None:
        """Hide overlay after flash, unless Show Region is permanently toggled on."""
        if not self._show_region_on:
            self._region_overlay.hide()

    # ------------------------------------------------------------------
    # Reading control
    # ------------------------------------------------------------------

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
        d = self._debug

        # Apply language filter (removes lines with wrong language)
        filtered = filter_by_language(text, self._settings.language)

        if d and text:
            d.log("OCR RAW", repr(text))
            if filtered != text:
                d.log("LANG FILTER", repr(filtered))

        # Update display only when there's actual text (avoid flickering)
        if filtered:
            self._window.update_text_display(filtered)

        # In growing subtitles mode, feed raw text (after OCR garbage filter
        # but before lang filter) to the differ.  The lang filter can strip
        # partially-visible text to empty, which confuses the growing-mode
        # buffer by making it think the subtitle disappeared.
        # Lang filter is applied to the differ's output before TTS instead.
        growing = self._settings.growing_subtitles
        differ_input = text if growing else filtered

        new_text = self._text_differ.get_new_text(differ_input)
        if new_text:
            if d:
                d.log("DIFFER NEW", repr(new_text))

            # In growing mode, apply lang filter to the output
            if growing:
                new_text = filter_by_language(new_text, self._settings.language)

            # Clean text for natural TTS reading (remove symbols, OCR artifacts)
            cleaned = clean_for_tts(new_text)
            if cleaned:
                if d:
                    d.log("TTS SPEAK", repr(cleaned))
                    d.save_subtitle(cleaned)
                self._tts_worker.speak(cleaned)
            elif d:
                d.log("TTS SKIP", "clean_for_tts returned empty")
        elif d and (filtered if not growing else text):
            d.log("DIFFER SKIP", "duplicate/similar, not speaking")

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

        if self._debug:
            self._debug.shutdown()

        return exit_code


def main() -> None:
    app = App()
    sys.exit(app.run())
