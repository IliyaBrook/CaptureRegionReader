from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal, QPoint, QRect
from PyQt6.QtGui import QPainter, QColor, QPen, QGuiApplication
from PyQt6.QtWidgets import QWidget


class RegionSelector(QWidget):
    region_selected = pyqtSignal(int, int, int, int)  # left, top, width, height

    def __init__(self) -> None:
        super().__init__()
        self._start_pos: QPoint | None = None
        self._current_pos: QPoint | None = None

        # Compute virtual desktop geometry spanning all monitors
        screens = QGuiApplication.screens()
        if screens:
            combined = screens[0].geometry()
            for screen in screens[1:]:
                combined = combined.united(screen.geometry())
        else:
            combined = QRect(0, 0, 1920, 1080)

        self.setGeometry(combined)

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.X11BypassWindowManagerHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setMouseTracking(True)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Semi-transparent dark overlay
        overlay_color = QColor(0, 0, 0, 100)
        painter.fillRect(self.rect(), overlay_color)

        if self._start_pos and self._current_pos:
            sel_rect = QRect(
                self.mapFromGlobal(self._start_pos),
                self.mapFromGlobal(self._current_pos),
            ).normalized()

            # Clear the selected area
            painter.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_Clear
            )
            painter.fillRect(sel_rect, Qt.GlobalColor.transparent)

            # Draw green border
            painter.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_SourceOver
            )
            pen = QPen(QColor(0, 200, 0), 2)
            painter.setPen(pen)
            painter.drawRect(sel_rect)

            # Show size label
            w = sel_rect.width()
            h = sel_rect.height()
            label = f"{w} x {h}"
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(sel_rect.left() + 4, sel_rect.top() - 6, label)

        painter.end()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._start_pos = event.globalPosition().toPoint()
            self._current_pos = self._start_pos

    def mouseMoveEvent(self, event) -> None:
        if self._start_pos:
            self._current_pos = event.globalPosition().toPoint()
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._start_pos:
            end_pos = event.globalPosition().toPoint()

            # Qt globalPosition coordinates match mss coordinates on X11
            x1 = min(self._start_pos.x(), end_pos.x())
            y1 = min(self._start_pos.y(), end_pos.y())
            x2 = max(self._start_pos.x(), end_pos.x())
            y2 = max(self._start_pos.y(), end_pos.y())

            width = x2 - x1
            height = y2 - y1

            if width > 10 and height > 10:
                self.region_selected.emit(x1, y1, width, height)

            self.close()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.close()
