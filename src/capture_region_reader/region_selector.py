from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal, QPoint, QRect
from PyQt6.QtGui import QPainter, QColor, QPen, QGuiApplication
from PyQt6.QtWidgets import QWidget


def qt_point_to_physical(qt_x: int, qt_y: int) -> tuple[int, int]:
    """Convert a Qt global coordinate to physical X11/mss coordinate.

    Qt with QT_SCREEN_SCALE_FACTORS uses a mixed coordinate space on X11:
    - Screen origins (QScreen.geometry().x/y) are in PHYSICAL pixels
    - Screen sizes (QScreen.geometry().width/height) are in LOGICAL pixels
    - Mouse globalPosition() returns coords in this mixed space

    MSS expects pure physical X11 root window coordinates.

    To convert: find which screen the point is on, compute the offset
    within that screen in logical pixels, multiply by DPR, and add to
    the screen's physical origin.
    """
    screens = QGuiApplication.screens()
    for screen in screens:
        geo = screen.geometry()
        if geo.contains(qt_x, qt_y):
            dpr = screen.devicePixelRatio()
            # Offset within screen in logical pixels
            offset_x = qt_x - geo.x()
            offset_y = qt_y - geo.y()
            # Convert to physical: origin (already physical) + offset * DPR
            phys_x = int(geo.x() + offset_x * dpr)
            phys_y = int(geo.y() + offset_y * dpr)
            return phys_x, phys_y

    # Fallback: point not on any screen (e.g., in a gap between screens)
    # Try to find the nearest screen and use its DPR
    if screens:
        # Use first screen's DPR as fallback
        dpr = screens[0].devicePixelRatio()
        return int(qt_x * dpr), int(qt_y * dpr)
    return qt_x, qt_y


class RegionSelector(QWidget):
    # Signal emits PHYSICAL pixel coordinates for mss
    region_selected = pyqtSignal(int, int, int, int)  # left, top, width, height

    def __init__(self) -> None:
        super().__init__()
        self._start_pos: QPoint | None = None
        self._current_pos: QPoint | None = None

        # Compute virtual desktop geometry spanning all monitors
        # Qt reports this in its mixed coordinate space
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

            # Show size label (physical pixels for accuracy)
            p1 = qt_point_to_physical(
                min(self._start_pos.x(), self._current_pos.x()),
                min(self._start_pos.y(), self._current_pos.y()),
            )
            p2 = qt_point_to_physical(
                max(self._start_pos.x(), self._current_pos.x()),
                max(self._start_pos.y(), self._current_pos.y()),
            )
            pw = p2[0] - p1[0]
            ph = p2[1] - p1[1]
            label = f"{pw} x {ph} px"
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

            # Convert Qt mixed coordinates to physical X11 coordinates for mss
            phys_start = qt_point_to_physical(
                self._start_pos.x(), self._start_pos.y()
            )
            phys_end = qt_point_to_physical(end_pos.x(), end_pos.y())

            x1 = min(phys_start[0], phys_end[0])
            y1 = min(phys_start[1], phys_end[1])
            x2 = max(phys_start[0], phys_end[0])
            y2 = max(phys_start[1], phys_end[1])

            width = x2 - x1
            height = y2 - y1

            if width > 10 and height > 10:
                print(
                    f"[Region] Qt start=({self._start_pos.x()},{self._start_pos.y()}) "
                    f"end=({end_pos.x()},{end_pos.y()}) → "
                    f"Physical ({x1},{y1}) {width}x{height}"
                )
                self.region_selected.emit(x1, y1, width, height)

            self.close()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.close()
