from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QIcon, QImage, QKeySequence, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSlider,
    QStatusBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from pathlib import Path

from capture_region_reader.settings import AppSettings

# Icon path: assets/icon.png relative to project root
_ICON_PATH = Path(__file__).resolve().parent.parent.parent / "assets" / "icon.png"


class HotkeyRecorder(QLineEdit):
    """Line edit that captures key combinations when recording."""

    hotkey_recorded = pyqtSignal(str)

    def __init__(self, initial_text: str = "") -> None:
        super().__init__(initial_text)
        self.setReadOnly(True)
        self._recording = False

    def start_recording(self) -> None:
        self._recording = True
        self.setText("Press key combination...")
        self.setStyleSheet("background-color: #ffcccc; font-weight: bold;")
        self.grabKeyboard()

    def stop_recording(self) -> None:
        self._recording = False
        self.setStyleSheet("")
        self.releaseKeyboard()

    def keyPressEvent(self, event) -> None:
        if not self._recording:
            return super().keyPressEvent(event)

        key = event.key()

        # Ignore standalone modifier presses
        if key in (
            Qt.Key.Key_Control,
            Qt.Key.Key_Alt,
            Qt.Key.Key_Shift,
            Qt.Key.Key_Meta,
        ):
            return

        # Escape cancels recording
        if key == Qt.Key.Key_Escape:
            self.stop_recording()
            return

        modifiers = event.modifiers()
        parts: list[str] = []

        if modifiers & Qt.KeyboardModifier.ControlModifier:
            parts.append("Ctrl")
        if modifiers & Qt.KeyboardModifier.AltModifier:
            parts.append("Alt")
        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            parts.append("Shift")
        if modifiers & Qt.KeyboardModifier.MetaModifier:
            parts.append("Super")

        key_name = QKeySequence(key).toString()
        if key_name:
            parts.append(key_name)

        combo = "+".join(parts)
        self.setText(combo)
        self.stop_recording()
        self.hotkey_recorded.emit(combo)


class MainWindow(QMainWindow):
    select_region_clicked = pyqtSignal()
    toggle_reading = pyqtSignal()
    language_changed = pyqtSignal(str)
    rate_changed = pyqtSignal(int)
    volume_changed = pyqtSignal(float)
    hotkey_changed = pyqtSignal(str)
    select_region_hotkey_changed = pyqtSignal(str)
    interval_changed = pyqtSignal(int)
    ocr_engine_changed = pyqtSignal(str)
    settle_time_changed = pyqtSignal(int)

    def __init__(self, settings: AppSettings) -> None:
        super().__init__()
        self.settings = settings
        self._is_reading = False

        self.setWindowTitle("CaptureRegionReader")
        if _ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(_ICON_PATH)))
        self.setMinimumSize(500, 800)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(10)

        # --- Region section ---
        region_group = QGroupBox("Screen Region")
        region_layout = QVBoxLayout(region_group)

        self._btn_select = QPushButton("Select Screen Region")
        self._btn_select.setMinimumHeight(40)
        self._btn_select.setStyleSheet(
            "QPushButton { font-size: 14px; font-weight: bold; }"
        )
        self._btn_select.clicked.connect(self.select_region_clicked.emit)
        region_layout.addWidget(self._btn_select)

        self._lbl_region = QLabel("No region selected")
        self._lbl_region.setAlignment(Qt.AlignmentFlag.AlignCenter)
        region_layout.addWidget(self._lbl_region)

        if settings.region:
            l, t, w, h = settings.region
            self._lbl_region.setText(f"Region: ({l}, {t}) {w}x{h}")

        layout.addWidget(region_group)

        # --- Controls section ---
        controls_group = QGroupBox("Controls")
        controls_layout = QHBoxLayout(controls_group)

        self._btn_toggle = QPushButton("Start Reading")
        self._btn_toggle.setMinimumHeight(45)
        self._btn_toggle.setEnabled(settings.region is not None)
        self._btn_toggle.setStyleSheet(
            "QPushButton { font-size: 14px; font-weight: bold; }"
        )
        self._btn_toggle.clicked.connect(self._on_toggle_clicked)
        controls_layout.addWidget(self._btn_toggle)

        self._lbl_state = QLabel("Idle")
        self._lbl_state.setStyleSheet(
            "QLabel { font-size: 13px; padding: 8px; color: #888; }"
        )
        controls_layout.addWidget(self._lbl_state)

        layout.addWidget(controls_group)

        # --- Hotkey section ---
        hotkey_group = QGroupBox("Hotkeys")
        hotkey_layout = QVBoxLayout(hotkey_group)

        # Toggle Start/Stop hotkey
        toggle_row = QHBoxLayout()
        toggle_row.addWidget(QLabel("Start/Stop:"))
        self._hotkey_edit = HotkeyRecorder(settings.hotkey)
        self._hotkey_edit.hotkey_recorded.connect(self._on_hotkey_recorded)
        toggle_row.addWidget(self._hotkey_edit, stretch=1)
        self._btn_record = QPushButton("Record")
        self._btn_record.clicked.connect(self._hotkey_edit.start_recording)
        toggle_row.addWidget(self._btn_record)
        hotkey_layout.addLayout(toggle_row)

        # Select Region hotkey
        region_hotkey_row = QHBoxLayout()
        region_hotkey_row.addWidget(QLabel("Select Region:"))
        self._region_hotkey_edit = HotkeyRecorder(settings.select_region_hotkey)
        self._region_hotkey_edit.hotkey_recorded.connect(self._on_region_hotkey_recorded)
        region_hotkey_row.addWidget(self._region_hotkey_edit, stretch=1)
        self._btn_record_region = QPushButton("Record")
        self._btn_record_region.clicked.connect(self._region_hotkey_edit.start_recording)
        region_hotkey_row.addWidget(self._btn_record_region)
        hotkey_layout.addLayout(region_hotkey_row)

        layout.addWidget(hotkey_group)

        # --- Language & OCR mode section ---
        lang_group = QGroupBox("Language && OCR Mode")
        lang_layout = QVBoxLayout(lang_group)

        lang_row = QHBoxLayout()
        lang_row.addWidget(QLabel("Language:"))
        self._combo_lang = QComboBox()
        self._combo_lang.addItem("Auto (English + Russian)", "eng+rus")
        self._combo_lang.addItem("English", "eng")
        self._combo_lang.addItem("Russian", "rus")

        # Set current from settings
        for i in range(self._combo_lang.count()):
            if self._combo_lang.itemData(i) == settings.language:
                self._combo_lang.setCurrentIndex(i)
                break

        self._combo_lang.currentIndexChanged.connect(self._on_lang_changed)
        lang_row.addWidget(self._combo_lang, stretch=1)
        lang_layout.addLayout(lang_row)

        # OCR engine selector
        engine_row = QHBoxLayout()
        engine_row.addWidget(QLabel("OCR Engine:"))
        self._combo_engine = QComboBox()
        self._combo_engine.addItem("Tesseract", "tesseract")
        self._combo_engine.addItem("EasyOCR (GPU)", "easyocr")

        for i in range(self._combo_engine.count()):
            if self._combo_engine.itemData(i) == settings.ocr_engine:
                self._combo_engine.setCurrentIndex(i)
                break

        self._combo_engine.currentIndexChanged.connect(self._on_engine_changed)
        engine_row.addWidget(self._combo_engine, stretch=1)
        lang_layout.addLayout(engine_row)

        # Settle time slider
        settle_row = QHBoxLayout()
        settle_row.addWidget(QLabel("Settle time:"))
        self._slider_settle = QSlider(Qt.Orientation.Horizontal)
        self._slider_settle.setRange(0, 2000)
        self._slider_settle.setSingleStep(50)
        self._slider_settle.setValue(settings.settle_time_ms)
        self._slider_settle.valueChanged.connect(self._on_settle_changed)
        settle_row.addWidget(self._slider_settle, stretch=1)
        self._lbl_settle = QLabel(f"{settings.settle_time_ms} ms")
        self._lbl_settle.setMinimumWidth(60)
        settle_row.addWidget(self._lbl_settle)
        lang_layout.addLayout(settle_row)

        layout.addWidget(lang_group)

        # --- Speech settings ---
        speech_group = QGroupBox("Speech Settings")
        speech_layout = QVBoxLayout(speech_group)

        # Rate slider
        rate_row = QHBoxLayout()
        rate_row.addWidget(QLabel("Speed:"))
        self._slider_rate = QSlider(Qt.Orientation.Horizontal)
        self._slider_rate.setRange(50, 350)
        self._slider_rate.setValue(settings.speech_rate)
        self._slider_rate.valueChanged.connect(self._on_rate_changed)
        rate_row.addWidget(self._slider_rate, stretch=1)
        self._lbl_rate = QLabel(f"{settings.speech_rate} wpm")
        self._lbl_rate.setMinimumWidth(70)
        rate_row.addWidget(self._lbl_rate)
        speech_layout.addLayout(rate_row)

        # Volume slider
        vol_row = QHBoxLayout()
        vol_row.addWidget(QLabel("Volume:"))
        self._slider_volume = QSlider(Qt.Orientation.Horizontal)
        self._slider_volume.setRange(0, 100)
        self._slider_volume.setValue(int(settings.volume * 100))
        self._slider_volume.valueChanged.connect(self._on_volume_changed)
        vol_row.addWidget(self._slider_volume, stretch=1)
        self._lbl_volume = QLabel(f"{int(settings.volume * 100)}%")
        self._lbl_volume.setMinimumWidth(50)
        vol_row.addWidget(self._lbl_volume)
        speech_layout.addLayout(vol_row)

        # OCR interval
        interval_row = QHBoxLayout()
        interval_row.addWidget(QLabel("OCR interval:"))
        self._slider_interval = QSlider(Qt.Orientation.Horizontal)
        self._slider_interval.setRange(200, 3000)
        self._slider_interval.setSingleStep(100)
        self._slider_interval.setValue(settings.ocr_interval_ms)
        self._slider_interval.valueChanged.connect(self._on_interval_changed)
        interval_row.addWidget(self._slider_interval, stretch=1)
        self._lbl_interval = QLabel(f"{settings.ocr_interval_ms} ms")
        self._lbl_interval.setMinimumWidth(60)
        interval_row.addWidget(self._lbl_interval)
        speech_layout.addLayout(interval_row)

        layout.addWidget(speech_group)

        # --- Text display ---
        text_group = QGroupBox("Recognized Text")
        text_layout = QVBoxLayout(text_group)

        self._text_display = QTextEdit()
        self._text_display.setReadOnly(True)
        self._text_display.setMinimumHeight(120)
        self._text_display.setPlaceholderText("Recognized text will appear here...")
        text_layout.addWidget(self._text_display)

        layout.addWidget(text_group)

        # --- Capture preview ---
        preview_group = QGroupBox("Capture Preview")
        preview_layout = QVBoxLayout(preview_group)

        # Toolbar: Fit/1:1 toggle + size info
        preview_toolbar = QHBoxLayout()
        self._btn_preview_fit = QPushButton("Fit")
        self._btn_preview_fit.setCheckable(True)
        self._btn_preview_fit.setChecked(True)
        self._btn_preview_fit.setMaximumWidth(60)
        self._btn_preview_fit.clicked.connect(self._on_preview_fit_toggled)
        preview_toolbar.addWidget(self._btn_preview_fit)
        self._lbl_preview_info = QLabel("")
        self._lbl_preview_info.setStyleSheet("QLabel { color: #888; font-size: 11px; }")
        preview_toolbar.addWidget(self._lbl_preview_info)
        preview_toolbar.addStretch()
        preview_layout.addLayout(preview_toolbar)

        # Scrollable image area
        self._preview_scroll = QScrollArea()
        self._preview_scroll.setMinimumHeight(120)
        self._preview_scroll.setStyleSheet(
            "QScrollArea { background-color: #1a1a1a; border: 1px solid #333; }"
        )
        self._preview_scroll.setWidgetResizable(False)

        self._capture_preview = QLabel("Capture preview will appear here...")
        self._capture_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._capture_preview.setStyleSheet(
            "QLabel { background-color: #1a1a1a; color: #666; padding: 4px; }"
        )
        self._preview_scroll.setWidget(self._capture_preview)

        preview_layout.addWidget(self._preview_scroll)

        self._preview_fit_mode = True
        self._preview_original_pixmap: QPixmap | None = None

        layout.addWidget(preview_group)

        # --- Status bar ---
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready")

    # --- Slots ---

    def _on_toggle_clicked(self) -> None:
        self.toggle_reading.emit()

    def _on_hotkey_recorded(self, combo: str) -> None:
        self.settings.hotkey = combo
        self.hotkey_changed.emit(combo)

    def _on_region_hotkey_recorded(self, combo: str) -> None:
        self.settings.select_region_hotkey = combo
        self.select_region_hotkey_changed.emit(combo)

    def _on_lang_changed(self, index: int) -> None:
        lang = self._combo_lang.itemData(index)
        self.settings.language = lang
        self.language_changed.emit(lang)

    def _on_engine_changed(self, index: int) -> None:
        engine = self._combo_engine.itemData(index)
        self.settings.ocr_engine = engine
        self.ocr_engine_changed.emit(engine)

    def _on_settle_changed(self, value: int) -> None:
        self._lbl_settle.setText(f"{value} ms")
        self.settings.settle_time_ms = value
        self.settle_time_changed.emit(value)

    def _on_rate_changed(self, value: int) -> None:
        self._lbl_rate.setText(f"{value} wpm")
        self.settings.speech_rate = value
        self.rate_changed.emit(value)

    def _on_volume_changed(self, value: int) -> None:
        vol = value / 100.0
        self._lbl_volume.setText(f"{value}%")
        self.settings.volume = vol
        self.volume_changed.emit(vol)

    def _on_interval_changed(self, value: int) -> None:
        self._lbl_interval.setText(f"{value} ms")
        self.settings.ocr_interval_ms = value
        self.interval_changed.emit(value)

    # --- Public methods for app.py to call ---

    def on_region_selected(self, left: int, top: int, width: int, height: int) -> None:
        self.settings.region = (left, top, width, height)
        self._lbl_region.setText(f"Region: ({left}, {top}) {width}x{height}")
        self._btn_toggle.setEnabled(True)
        self._status_bar.showMessage(f"Region selected: ({left}, {top}) {width}x{height}")

    def set_reading_state(self, is_reading: bool) -> None:
        self._is_reading = is_reading
        if is_reading:
            self._btn_toggle.setText("Stop Reading")
            self._btn_toggle.setStyleSheet(
                "QPushButton { font-size: 14px; font-weight: bold; "
                "background-color: #cc3333; color: white; }"
            )
            self._lbl_state.setText("Reading...")
            self._lbl_state.setStyleSheet(
                "QLabel { font-size: 13px; padding: 8px; color: #33aa33; font-weight: bold; }"
            )
        else:
            self._btn_toggle.setText("Start Reading")
            self._btn_toggle.setStyleSheet(
                "QPushButton { font-size: 14px; font-weight: bold; }"
            )
            self._lbl_state.setText("Idle")
            self._lbl_state.setStyleSheet(
                "QLabel { font-size: 13px; padding: 8px; color: #888; }"
            )

    def update_text_display(self, text: str) -> None:
        self._text_display.setPlainText(text)

    def update_capture_preview(self, raw_bytes: bytes, width: int, height: int) -> None:
        """Update the capture preview with the latest screenshot frame."""
        qimg = QImage(raw_bytes, width, height, width * 3, QImage.Format.Format_RGB888)
        self._preview_original_pixmap = QPixmap.fromImage(qimg)
        self._lbl_preview_info.setText(f"{width}x{height} px")
        self._apply_preview_pixmap()

    def _apply_preview_pixmap(self) -> None:
        """Apply the stored pixmap in Fit or 1:1 mode."""
        if self._preview_original_pixmap is None:
            return
        if self._preview_fit_mode:
            # Scale to fit scroll area width
            scroll_w = self._preview_scroll.viewport().width() - 4
            if scroll_w > 0 and self._preview_original_pixmap.width() > scroll_w:
                scaled = self._preview_original_pixmap.scaledToWidth(
                    scroll_w, Qt.TransformationMode.SmoothTransformation
                )
            else:
                scaled = self._preview_original_pixmap
            self._capture_preview.setPixmap(scaled)
            self._capture_preview.resize(scaled.size())
        else:
            # 1:1 — full resolution, scrollable
            self._capture_preview.setPixmap(self._preview_original_pixmap)
            self._capture_preview.resize(self._preview_original_pixmap.size())

    def _on_preview_fit_toggled(self) -> None:
        self._preview_fit_mode = self._btn_preview_fit.isChecked()
        self._btn_preview_fit.setText("Fit" if self._preview_fit_mode else "1:1")
        self._apply_preview_pixmap()

    def show_status(self, message: str) -> None:
        self._status_bar.showMessage(message)

    def show_error(self, message: str) -> None:
        self._status_bar.showMessage(f"Error: {message}")
