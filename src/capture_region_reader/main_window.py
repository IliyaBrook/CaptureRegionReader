from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QImage, QKeySequence, QMouseEvent, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from pathlib import Path

from capture_region_reader.settings import AppSettings

# Icon path: assets/icon.png relative to project root
_ICON_PATH = Path(__file__).resolve().parent.parent.parent / "assets" / "icon.png"

# Shared style constants
_COMBO_STYLE = "QComboBox { min-height: 14px; font-size: 13px; }"
_FORM_LABEL_STYLE = "font-size: 13px;"
_SLIDER_LABEL_W = 80
_SLIDER_VALUE_W = 70


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
    tts_engine_changed = pyqtSignal(str)
    settle_time_changed = pyqtSignal(int)
    isolator_mode_changed = pyqtSignal(str)
    box_color_changed = pyqtSignal(object)       # tuple[int,int,int] | None
    box_color_tolerance_changed = pyqtSignal(int)

    def __init__(self, settings: AppSettings) -> None:
        super().__init__()
        self.settings = settings
        self._is_reading = False
        self._eyedropper_active = False
        self._preview_fit_mode = True
        self._preview_original_pixmap: QPixmap | None = None
        self._raw_pixmap: QPixmap | None = None  # original frame for eyedropper

        self.setWindowTitle("CaptureRegionReader")
        if _ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(_ICON_PATH)))
        self.setMinimumSize(520, 620)

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setSpacing(6)
        root_layout.setContentsMargins(6, 6, 6, 6)

        # === Top bar: Region + Controls (always visible, above tabs) ===
        self._build_top_bar(root_layout, settings)

        # === Tab widget ===
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        root_layout.addWidget(self._tabs, stretch=1)

        # Tab 1: Settings
        self._tabs.addTab(self._build_settings_tab(settings), "Settings")

        # Tab 2: Output (Recognized Text + Capture Preview)
        self._tabs.addTab(self._build_output_tab(settings), "Output")

        # --- Status bar ---
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready")

    # ------------------------------------------------------------------
    # Top bar (always visible above tabs)
    # ------------------------------------------------------------------

    def _build_top_bar(self, parent_layout: QVBoxLayout, settings: AppSettings) -> None:
        """Region selector + Start/Stop controls — always visible."""
        top = QWidget()
        top_layout = QVBoxLayout(top)
        top_layout.setSpacing(6)
        top_layout.setContentsMargins(0, 0, 0, 0)

        # Region row
        region_row = QHBoxLayout()
        region_row.setSpacing(8)

        self._btn_select = QPushButton("Select Region")
        self._btn_select.setMinimumHeight(36)
        self._btn_select.setStyleSheet(
            "QPushButton { font-size: 13px; font-weight: bold; }"
        )
        self._btn_select.clicked.connect(self.select_region_clicked.emit)
        region_row.addWidget(self._btn_select)

        self._lbl_region = QLabel("No region selected")
        self._lbl_region.setStyleSheet("QLabel { color: #888; font-size: 12px; }")
        if settings.region:
            l, t, w, h = settings.region
            self._lbl_region.setText(f"({l}, {t}) {w}\u00d7{h}")
        region_row.addWidget(self._lbl_region, stretch=1)

        self._btn_toggle = QPushButton("Start Reading")
        self._btn_toggle.setMinimumHeight(36)
        self._btn_toggle.setMinimumWidth(130)
        self._btn_toggle.setEnabled(settings.region is not None)
        self._btn_toggle.setStyleSheet(
            "QPushButton { font-size: 13px; font-weight: bold; }"
        )
        self._btn_toggle.clicked.connect(self._on_toggle_clicked)
        region_row.addWidget(self._btn_toggle)

        self._lbl_state = QLabel("Idle")
        self._lbl_state.setFixedWidth(80)
        self._lbl_state.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_state.setStyleSheet(
            "QLabel { font-size: 12px; color: #888; }"
        )
        region_row.addWidget(self._lbl_state)

        top_layout.addLayout(region_row)
        parent_layout.addWidget(top)

    # ------------------------------------------------------------------
    # Tab 1: Settings
    # ------------------------------------------------------------------

    def _build_settings_tab(self, settings: AppSettings) -> QWidget:
        """Build the Settings tab containing all configuration controls."""
        tab = QWidget()
        # Wrap in a scroll area so everything is accessible even at small sizes
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(8)
        layout.setContentsMargins(4, 4, 4, 4)

        # --- Hotkeys ---
        self._build_hotkeys_group(layout, settings)

        # --- Language / OCR / TTS ---
        self._build_engines_group(layout, settings)

        # --- Text Isolation ---
        self._build_isolator_group(layout, settings)

        # --- Sliders (speed, volume, intervals) ---
        self._build_sliders_group(layout, settings)

        layout.addStretch()
        scroll.setWidget(content)

        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.addWidget(scroll)
        return tab

    def _build_hotkeys_group(self, parent: QVBoxLayout, settings: AppSettings) -> None:
        group = QGroupBox("Hotkeys")
        vbox = QVBoxLayout(group)
        vbox.setSpacing(6)
        vbox.setContentsMargins(10, 14, 10, 10)

        # Toggle Start/Stop hotkey
        toggle_row = QHBoxLayout()
        toggle_row.addWidget(QLabel("Start/Stop:"))
        self._hotkey_edit = HotkeyRecorder(settings.hotkey)
        self._hotkey_edit.hotkey_recorded.connect(self._on_hotkey_recorded)
        toggle_row.addWidget(self._hotkey_edit, stretch=1)
        self._btn_record = QPushButton("Record")
        self._btn_record.clicked.connect(self._hotkey_edit.start_recording)
        toggle_row.addWidget(self._btn_record)
        vbox.addLayout(toggle_row)

        # Select Region hotkey
        region_hotkey_row = QHBoxLayout()
        region_hotkey_row.addWidget(QLabel("Select Region:"))
        self._region_hotkey_edit = HotkeyRecorder(settings.select_region_hotkey)
        self._region_hotkey_edit.hotkey_recorded.connect(self._on_region_hotkey_recorded)
        region_hotkey_row.addWidget(self._region_hotkey_edit, stretch=1)
        self._btn_record_region = QPushButton("Record")
        self._btn_record_region.clicked.connect(self._region_hotkey_edit.start_recording)
        region_hotkey_row.addWidget(self._btn_record_region)
        vbox.addLayout(region_hotkey_row)

        parent.addWidget(group)

    def _build_engines_group(self, parent: QVBoxLayout, settings: AppSettings) -> None:
        group = QGroupBox("Language && Engines")
        form = QFormLayout(group)
        form.setSpacing(8)
        form.setContentsMargins(10, 14, 10, 10)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form.setLabelAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        # Language
        self._combo_lang = QComboBox()
        self._combo_lang.setStyleSheet(_COMBO_STYLE)
        self._combo_lang.addItem("Auto (English + Russian)", "eng+rus")
        self._combo_lang.addItem("English", "eng")
        self._combo_lang.addItem("Russian", "rus")
        for i in range(self._combo_lang.count()):
            if self._combo_lang.itemData(i) == settings.language:
                self._combo_lang.setCurrentIndex(i)
                break
        self._combo_lang.currentIndexChanged.connect(self._on_lang_changed)
        lbl = QLabel("Language:")
        lbl.setStyleSheet(_FORM_LABEL_STYLE)
        form.addRow(lbl, self._combo_lang)

        # OCR engine
        self._combo_engine = QComboBox()
        self._combo_engine.setStyleSheet(_COMBO_STYLE)
        self._combo_engine.addItem("Tesseract", "tesseract")
        self._combo_engine.addItem("EasyOCR (GPU)", "easyocr")
        for i in range(self._combo_engine.count()):
            if self._combo_engine.itemData(i) == settings.ocr_engine:
                self._combo_engine.setCurrentIndex(i)
                break
        self._combo_engine.currentIndexChanged.connect(self._on_engine_changed)
        lbl = QLabel("OCR Engine:")
        lbl.setStyleSheet(_FORM_LABEL_STYLE)
        form.addRow(lbl, self._combo_engine)

        # TTS engine
        self._combo_tts = QComboBox()
        self._combo_tts.setStyleSheet(_COMBO_STYLE)
        self._combo_tts.addItem("XTTS-v2 (local, multilingual)", "xtts")
        self._combo_tts.addItem("Silero (local, Russian only)", "silero")
        self._combo_tts.addItem("Edge-TTS (cloud)", "edge-tts")
        for i in range(self._combo_tts.count()):
            if self._combo_tts.itemData(i) == settings.tts_engine:
                self._combo_tts.setCurrentIndex(i)
                break
        self._combo_tts.currentIndexChanged.connect(self._on_tts_engine_changed)
        lbl = QLabel("TTS Voice:")
        lbl.setStyleSheet(_FORM_LABEL_STYLE)
        form.addRow(lbl, self._combo_tts)

        parent.addWidget(group)

    def _build_isolator_group(self, parent: QVBoxLayout, settings: AppSettings) -> None:
        group = QGroupBox("Text Isolation")
        form = QFormLayout(group)
        form.setSpacing(8)
        form.setContentsMargins(10, 14, 10, 10)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form.setLabelAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        # Mode combo
        self._combo_isolator = QComboBox()
        self._combo_isolator.setStyleSheet(_COMBO_STYLE)
        self._combo_isolator.addItem("Default (auto detect)", "default")
        self._combo_isolator.addItem("Box Search (pick color)", "box_search")
        for i in range(self._combo_isolator.count()):
            if self._combo_isolator.itemData(i) == settings.isolator_mode:
                self._combo_isolator.setCurrentIndex(i)
                break
        self._combo_isolator.currentIndexChanged.connect(self._on_isolator_mode_changed)
        lbl = QLabel("Mode:")
        lbl.setStyleSheet(_FORM_LABEL_STYLE)
        form.addRow(lbl, self._combo_isolator)

        # Color picker row: swatch + pick button + eyedropper button
        color_row = QHBoxLayout()
        color_row.setSpacing(6)

        self._color_swatch = QLabel()
        self._color_swatch.setFixedSize(24, 24)
        self._color_swatch.setStyleSheet(
            "QLabel { border: 2px solid #555; border-radius: 3px; }"
        )
        self._update_color_swatch(settings.box_color)
        color_row.addWidget(self._color_swatch)

        self._btn_pick_color = QPushButton("Pick Color")
        self._btn_pick_color.setToolTip("Open a color dialog to choose the background color")
        self._btn_pick_color.clicked.connect(self._on_pick_color_dialog)
        color_row.addWidget(self._btn_pick_color)

        self._btn_eyedropper = QPushButton("Eyedropper")
        self._btn_eyedropper.setToolTip(
            "Click on the capture preview to sample a color"
        )
        self._btn_eyedropper.setCheckable(True)
        self._btn_eyedropper.clicked.connect(self._on_eyedropper_toggled)
        color_row.addWidget(self._btn_eyedropper)
        color_row.addStretch()

        color_widget = QWidget()
        color_widget.setLayout(color_row)
        _lbl_color = QLabel("Box Color:")
        _lbl_color.setStyleSheet(_FORM_LABEL_STYLE)
        form.addRow(_lbl_color, color_widget)

        # Tolerance slider
        tol_row = QHBoxLayout()
        tol_row.setSpacing(6)
        self._slider_tolerance = QSlider(Qt.Orientation.Horizontal)
        self._slider_tolerance.setRange(10, 150)
        self._slider_tolerance.setSingleStep(5)
        self._slider_tolerance.setValue(settings.box_color_tolerance)
        self._slider_tolerance.valueChanged.connect(self._on_tolerance_changed)
        tol_row.addWidget(self._slider_tolerance, stretch=1)
        self._lbl_tolerance = QLabel(f"{settings.box_color_tolerance}")
        self._lbl_tolerance.setFixedWidth(35)
        self._lbl_tolerance.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        tol_row.addWidget(self._lbl_tolerance)

        tol_widget = QWidget()
        tol_widget.setLayout(tol_row)
        _lbl_tol = QLabel("Tolerance:")
        _lbl_tol.setStyleSheet(_FORM_LABEL_STYLE)
        form.addRow(_lbl_tol, tol_widget)

        # Show/hide box-search controls
        self._box_search_widgets = [_lbl_color, color_widget, _lbl_tol, tol_widget]
        is_box_mode = settings.isolator_mode == "box_search"
        for w_ in self._box_search_widgets:
            w_.setVisible(is_box_mode)

        parent.addWidget(group)

    def _build_sliders_group(self, parent: QVBoxLayout, settings: AppSettings) -> None:
        group = QGroupBox("Timing && Audio")
        vbox = QVBoxLayout(group)
        vbox.setSpacing(6)
        vbox.setContentsMargins(10, 14, 10, 10)

        def _slider_row(
            label_text: str, slider: QSlider, value_label: QLabel,
        ) -> QHBoxLayout:
            row = QHBoxLayout()
            lbl = QLabel(label_text)
            lbl.setFixedWidth(_SLIDER_LABEL_W)
            row.addWidget(lbl)
            row.addWidget(slider, stretch=1)
            value_label.setFixedWidth(_SLIDER_VALUE_W)
            value_label.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            row.addWidget(value_label)
            return row

        # Settle time
        self._slider_settle = QSlider(Qt.Orientation.Horizontal)
        self._slider_settle.setRange(0, 2000)
        self._slider_settle.setSingleStep(50)
        self._slider_settle.setValue(settings.settle_time_ms)
        self._slider_settle.valueChanged.connect(self._on_settle_changed)
        self._lbl_settle = QLabel(f"{settings.settle_time_ms} ms")
        vbox.addLayout(_slider_row("Settle time:", self._slider_settle, self._lbl_settle))

        # Speech speed
        self._slider_rate = QSlider(Qt.Orientation.Horizontal)
        self._slider_rate.setRange(50, 350)
        self._slider_rate.setValue(settings.speech_rate)
        self._slider_rate.valueChanged.connect(self._on_rate_changed)
        self._lbl_rate = QLabel(f"{settings.speech_rate} wpm")
        vbox.addLayout(_slider_row("Speed:", self._slider_rate, self._lbl_rate))

        # Volume
        self._slider_volume = QSlider(Qt.Orientation.Horizontal)
        self._slider_volume.setRange(0, 100)
        self._slider_volume.setValue(int(settings.volume * 100))
        self._slider_volume.valueChanged.connect(self._on_volume_changed)
        self._lbl_volume = QLabel(f"{int(settings.volume * 100)}%")
        vbox.addLayout(_slider_row("Volume:", self._slider_volume, self._lbl_volume))

        # OCR interval
        self._slider_interval = QSlider(Qt.Orientation.Horizontal)
        self._slider_interval.setRange(200, 3000)
        self._slider_interval.setSingleStep(100)
        self._slider_interval.setValue(settings.ocr_interval_ms)
        self._slider_interval.valueChanged.connect(self._on_interval_changed)
        self._lbl_interval = QLabel(f"{settings.ocr_interval_ms} ms")
        vbox.addLayout(_slider_row("OCR interval:", self._slider_interval, self._lbl_interval))

        parent.addWidget(group)

    # ------------------------------------------------------------------
    # Tab 2: Output (Recognized Text + Capture Preview)
    # ------------------------------------------------------------------

    def _build_output_tab(self, settings: AppSettings) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(8)
        layout.setContentsMargins(4, 4, 4, 4)

        # --- Recognized Text ---
        text_group = QGroupBox("Recognized Text")
        text_layout = QVBoxLayout(text_group)
        text_layout.setContentsMargins(6, 14, 6, 6)

        self._text_display = QTextEdit()
        self._text_display.setReadOnly(True)
        self._text_display.setMinimumHeight(80)
        self._text_display.setPlaceholderText("Recognized text will appear here...")
        text_layout.addWidget(self._text_display)

        layout.addWidget(text_group)

        # --- Raw Capture (for eyedropper + understanding what is captured) ---
        raw_group = QGroupBox("Raw Capture (click to pick color)")
        raw_layout = QVBoxLayout(raw_group)
        raw_layout.setContentsMargins(6, 14, 6, 6)

        self._raw_preview_scroll = QScrollArea()
        self._raw_preview_scroll.setMinimumHeight(100)
        self._raw_preview_scroll.setStyleSheet(
            "QScrollArea { background-color: #1a1a1a; border: 1px solid #333; }"
        )
        self._raw_preview_scroll.setWidgetResizable(False)

        self._raw_preview = QLabel("Raw capture will appear here...")
        self._raw_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._raw_preview.setStyleSheet(
            "QLabel { background-color: #1a1a1a; color: #666; padding: 4px; }"
        )
        # Eyedropper picks color from the RAW preview (not binarized)
        self._raw_preview.mousePressEvent = self._preview_mouse_press
        self._raw_preview_scroll.setWidget(self._raw_preview)

        raw_layout.addWidget(self._raw_preview_scroll, stretch=1)
        layout.addWidget(raw_group, stretch=1)

        # --- Processed Preview (what OCR engine sees) ---
        preview_group = QGroupBox("Processed Preview (OCR input)")
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.setContentsMargins(6, 14, 6, 6)

        # Toolbar: Fit/1:1 toggle + size info
        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)
        self._btn_preview_fit = QPushButton("Fit")
        self._btn_preview_fit.setCheckable(True)
        self._btn_preview_fit.setChecked(True)
        self._btn_preview_fit.setMaximumWidth(50)
        self._btn_preview_fit.clicked.connect(self._on_preview_fit_toggled)
        toolbar.addWidget(self._btn_preview_fit)
        self._lbl_preview_info = QLabel("")
        self._lbl_preview_info.setStyleSheet("QLabel { color: #888; font-size: 11px; }")
        toolbar.addWidget(self._lbl_preview_info)
        toolbar.addStretch()
        preview_layout.addLayout(toolbar)

        self._preview_scroll = QScrollArea()
        self._preview_scroll.setMinimumHeight(100)
        self._preview_scroll.setStyleSheet(
            "QScrollArea { background-color: #1a1a1a; border: 1px solid #333; }"
        )
        self._preview_scroll.setWidgetResizable(False)

        self._capture_preview = QLabel("Processed preview will appear here...")
        self._capture_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._capture_preview.setStyleSheet(
            "QLabel { background-color: #1a1a1a; color: #666; padding: 4px; }"
        )
        self._preview_scroll.setWidget(self._capture_preview)

        preview_layout.addWidget(self._preview_scroll, stretch=1)
        layout.addWidget(preview_group, stretch=1)

        return tab

    # ==================================================================
    # Slots — Settings changes
    # ==================================================================

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

    def _on_tts_engine_changed(self, index: int) -> None:
        engine = self._combo_tts.itemData(index)
        self.settings.tts_engine = engine
        self.tts_engine_changed.emit(engine)

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

    # --- Isolator mode ---

    def _on_isolator_mode_changed(self, index: int) -> None:
        mode = self._combo_isolator.itemData(index)
        self.settings.isolator_mode = mode
        is_box = mode == "box_search"
        for w_ in self._box_search_widgets:
            w_.setVisible(is_box)
        self.isolator_mode_changed.emit(mode)

    def _update_color_swatch(self, color: tuple[int, int, int] | None) -> None:
        """Update the color swatch label to show the current box color."""
        if color is not None:
            r, g, b = color
            self._color_swatch.setStyleSheet(
                f"QLabel {{ background-color: rgb({r},{g},{b}); "
                f"border: 2px solid #555; border-radius: 3px; }}"
            )
            self._color_swatch.setToolTip(f"RGB({r}, {g}, {b})")
        else:
            self._color_swatch.setStyleSheet(
                "QLabel { border: 2px solid #555; border-radius: 3px; "
                "background-color: transparent; }"
            )
            self._color_swatch.setToolTip("No color selected")

    def _on_pick_color_dialog(self) -> None:
        """Open a QColorDialog to pick the box background color."""
        initial = QColor(0, 0, 0)
        if self.settings.box_color:
            initial = QColor(*self.settings.box_color)
        color = QColorDialog.getColor(initial, self, "Pick Subtitle Background Color")
        if color.isValid():
            rgb = (color.red(), color.green(), color.blue())
            self._set_box_color(rgb)

    def _on_eyedropper_toggled(self, checked: bool) -> None:
        """Toggle eyedropper mode: next click on raw preview picks a color."""
        self._eyedropper_active = checked
        if checked:
            self._btn_eyedropper.setStyleSheet(
                "QPushButton { background-color: #ffcc00; font-weight: bold; }"
            )
            self._raw_preview.setCursor(Qt.CursorShape.CrossCursor)
            self._status_bar.showMessage(
                "Eyedropper active \u2014 click on the Raw Capture to pick a color"
            )
            # Switch to Output tab so user can click on the raw preview
            self._tabs.setCurrentIndex(1)
        else:
            self._btn_eyedropper.setStyleSheet("")
            self._raw_preview.setCursor(Qt.CursorShape.ArrowCursor)
            self._status_bar.showMessage("Ready")

    def _preview_mouse_press(self, event: QMouseEvent) -> None:
        """Handle mouse clicks on the raw preview for eyedropper."""
        if not self._eyedropper_active:
            return
        if self._raw_pixmap is None:
            return

        # Map click position to the original (unscaled) raw pixmap coordinates
        click_x = event.position().x()
        click_y = event.position().y()

        displayed = self._raw_preview.pixmap()
        if displayed is None:
            return

        # Scale click to raw pixmap coordinates
        scale_x = self._raw_pixmap.width() / max(displayed.width(), 1)
        scale_y = self._raw_pixmap.height() / max(displayed.height(), 1)
        orig_x = int(click_x * scale_x)
        orig_y = int(click_y * scale_y)

        # Clamp to pixmap bounds
        orig_x = max(0, min(orig_x, self._raw_pixmap.width() - 1))
        orig_y = max(0, min(orig_y, self._raw_pixmap.height() - 1))

        # Get the pixel color from the RAW pixmap (not the binarized preview)
        img = self._raw_pixmap.toImage()
        pixel = img.pixelColor(orig_x, orig_y)
        rgb = (pixel.red(), pixel.green(), pixel.blue())

        self._set_box_color(rgb)

        # Deactivate eyedropper
        self._btn_eyedropper.setChecked(False)
        self._on_eyedropper_toggled(False)
        self._status_bar.showMessage(
            f"Color picked: RGB({rgb[0]}, {rgb[1]}, {rgb[2]})"
        )

    def _set_box_color(self, rgb: tuple[int, int, int]) -> None:
        """Set box search color and update UI + settings."""
        self.settings.box_color = rgb
        self._update_color_swatch(rgb)
        self.box_color_changed.emit(rgb)

    def _on_tolerance_changed(self, value: int) -> None:
        self._lbl_tolerance.setText(str(value))
        self.settings.box_color_tolerance = value
        self.box_color_tolerance_changed.emit(value)

    # ==================================================================
    # Public methods (called by app.py)
    # ==================================================================

    def on_region_selected(self, left: int, top: int, width: int, height: int) -> None:
        self.settings.region = (left, top, width, height)
        self._lbl_region.setText(f"({left}, {top}) {width}\u00d7{height}")
        self._btn_toggle.setEnabled(True)
        self._status_bar.showMessage(
            f"Region selected: ({left}, {top}) {width}\u00d7{height}"
        )

    def set_reading_state(self, is_reading: bool) -> None:
        self._is_reading = is_reading
        if is_reading:
            self._btn_toggle.setText("Stop Reading")
            self._btn_toggle.setStyleSheet(
                "QPushButton { font-size: 13px; font-weight: bold; "
                "background-color: #cc3333; color: white; }"
            )
            self._lbl_state.setText("Reading...")
            self._lbl_state.setStyleSheet(
                "QLabel { font-size: 12px; color: #33aa33; font-weight: bold; }"
            )
        else:
            self._btn_toggle.setText("Start Reading")
            self._btn_toggle.setStyleSheet(
                "QPushButton { font-size: 13px; font-weight: bold; }"
            )
            self._lbl_state.setText("Idle")
            self._lbl_state.setStyleSheet(
                "QLabel { font-size: 12px; color: #888; }"
            )

    def update_text_display(self, text: str) -> None:
        self._text_display.setPlainText(text)

    def update_raw_preview(self, raw_bytes: bytes, width: int, height: int) -> None:
        """Update the raw capture preview (original screenshot, for eyedropper)."""
        qimg = QImage(raw_bytes, width, height, width * 3, QImage.Format.Format_RGB888)
        self._raw_pixmap = QPixmap.fromImage(qimg)
        self._apply_raw_pixmap()

    def update_capture_preview(self, raw_bytes: bytes, width: int, height: int) -> None:
        """Update the processed capture preview (what OCR engine sees)."""
        qimg = QImage(raw_bytes, width, height, width * 3, QImage.Format.Format_RGB888)
        self._preview_original_pixmap = QPixmap.fromImage(qimg)
        self._lbl_preview_info.setText(f"{width}\u00d7{height} px")
        self._apply_preview_pixmap()

    def _apply_raw_pixmap(self) -> None:
        """Apply the raw pixmap (always fit to scroll width)."""
        if self._raw_pixmap is None:
            return
        scroll_w = self._raw_preview_scroll.viewport().width() - 4
        if scroll_w > 0 and self._raw_pixmap.width() > scroll_w:
            scaled = self._raw_pixmap.scaledToWidth(
                scroll_w, Qt.TransformationMode.SmoothTransformation
            )
        else:
            scaled = self._raw_pixmap
        self._raw_preview.setPixmap(scaled)
        self._raw_preview.resize(scaled.size())

    def _apply_preview_pixmap(self) -> None:
        """Apply the stored pixmap in Fit or 1:1 mode."""
        if self._preview_original_pixmap is None:
            return
        if self._preview_fit_mode:
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

    def revert_ocr_engine(self, failed_engine: str, error_msg: str) -> None:
        """Revert OCR engine combo box and show error dialog."""
        actual_engine = self.settings.ocr_engine
        if actual_engine == failed_engine:
            actual_engine = "tesseract"
            self.settings.ocr_engine = actual_engine

        self._combo_engine.blockSignals(True)
        for i in range(self._combo_engine.count()):
            if self._combo_engine.itemData(i) == actual_engine:
                self._combo_engine.setCurrentIndex(i)
                break
        self._combo_engine.blockSignals(False)

        engine_names = {"easyocr": "EasyOCR", "tesseract": "Tesseract"}
        display_name = engine_names.get(failed_engine, failed_engine)

        QMessageBox.warning(
            self,
            "OCR Engine Unavailable",
            f"Failed to load <b>{display_name}</b> engine.\n\n"
            f"Error: {error_msg}\n\n"
            f"Install the required packages or choose a different engine.\n\n"
            f"For EasyOCR:\n"
            f"  uv pip install easyocr torch",
        )

    def revert_tts_engine(self, failed_engine: str, error_msg: str) -> None:
        """Revert TTS engine combo box and show error dialog."""
        actual_engine = self.settings.tts_engine
        if actual_engine == failed_engine:
            actual_engine = "edge-tts"
            self.settings.tts_engine = actual_engine

        self._combo_tts.blockSignals(True)
        for i in range(self._combo_tts.count()):
            if self._combo_tts.itemData(i) == actual_engine:
                self._combo_tts.setCurrentIndex(i)
                break
        self._combo_tts.blockSignals(False)

        engine_names = {"silero": "Silero TTS", "edge-tts": "Edge-TTS", "xtts": "XTTS-v2"}
        display_name = engine_names.get(failed_engine, failed_engine)

        QMessageBox.warning(
            self,
            "TTS Engine Unavailable",
            f"Failed to load <b>{display_name}</b> engine.\n\n"
            f"Error: {error_msg}\n\n"
            f"Install the required packages or choose a different engine.\n\n"
            f"For Silero TTS (local, GPU):\n"
            f"  uv pip install torch\n\n"
            f"For Edge-TTS (cloud):\n"
            f"  uv pip install edge-tts",
        )
