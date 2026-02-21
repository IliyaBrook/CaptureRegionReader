from __future__ import annotations

from PyQt6.QtCore import Qt, QRectF, pyqtSignal, QPoint, QRect
from PyQt6.QtGui import QPainter, QPainterPath, QColor, QPen, QGuiApplication, QPixmap
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


def physical_to_qt_rect(
    phys_left: int, phys_top: int, phys_w: int, phys_h: int,
) -> tuple[int, int, int, int]:
    """Convert physical X11/mss coordinates back to Qt logical coordinates."""
    screens = QGuiApplication.screens()
    for screen in screens:
        geo = screen.geometry()
        dpr = screen.devicePixelRatio()
        # Screen physical extent (origin is already physical on X11)
        phys_right = geo.x() + geo.width() * dpr
        phys_bottom = geo.y() + geo.height() * dpr
        if geo.x() <= phys_left < phys_right and geo.y() <= phys_top < phys_bottom:
            qt_x = geo.x() + (phys_left - geo.x()) / dpr
            qt_y = geo.y() + (phys_top - geo.y()) / dpr
            qt_w = phys_w / dpr
            qt_h = phys_h / dpr
            return int(qt_x), int(qt_y), int(qt_w), int(qt_h)
    # Fallback: use first screen DPR
    if screens:
        dpr = screens[0].devicePixelRatio()
        return (
            int(phys_left / dpr), int(phys_top / dpr),
            int(phys_w / dpr), int(phys_h / dpr),
        )
    return phys_left, phys_top, phys_w, phys_h


class RegionOverlay(QWidget):
    """Persistent on-screen border showing the selected capture region."""

    def __init__(self) -> None:
        super().__init__()
        self._border = 2
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.X11BypassWindowManagerHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def set_region(self, left: int, top: int, width: int, height: int) -> None:
        """Position overlay from physical (mss) coordinates."""
        qt_x, qt_y, qt_w, qt_h = physical_to_qt_rect(left, top, width, height)
        b = self._border
        self.setGeometry(qt_x - b, qt_y - b, qt_w + 2 * b, qt_h + 2 * b)
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        b = self._border
        # Semi-transparent interior so the region is visible even over
        # fullscreen apps where the compositor can't show true transparency
        painter.fillRect(
            b, b, self.width() - 2 * b, self.height() - 2 * b,
            QColor(0, 200, 0, 18),
        )
        # Green border
        pen = QPen(QColor(0, 200, 0), b)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        half = b // 2
        painter.drawRect(half, half, self.width() - b, self.height() - b)
        painter.end()


class RegionSelector(QWidget):
    """Per-screen overlay for region selection.

    Following the Spectacle (KDE) approach: one window per QScreen,
    each with its own cropped desktop screenshot as background.
    This avoids coordinate-mapping issues on multi-monitor setups and
    prevents fullscreen apps from going black.
    """

    # Signal emits PHYSICAL pixel coordinates for mss
    region_selected = pyqtSignal(int, int, int, int)  # left, top, width, height
    cancelled = pyqtSignal()  # Escape or invalid click — close all sibling windows

    def __init__(self, screen: QScreen | None = None, background: QPixmap | None = None) -> None:  # noqa: F821
        super().__init__()
        self._start_pos: QPoint | None = None
        self._current_pos: QPoint | None = None
        self._background = background  # cropped screenshot for THIS screen

        if screen:
            self.setGeometry(screen.geometry())
        else:
            # Fallback: single screen
            screens = QGuiApplication.screens()
            if screens:
                self.setGeometry(screens[0].geometry())
            else:
                self.setGeometry(0, 0, 1920, 1080)

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

        # Draw frozen desktop screenshot as background so the user can
        # see screen content even if a fullscreen app goes black.
        if self._background and not self._background.isNull():
            painter.drawPixmap(self.rect(), self._background)

        overlay = QColor(0, 0, 0, 100)

        if self._start_pos and self._current_pos:
            sel_rect = QRect(
                self.mapFromGlobal(self._start_pos),
                self.mapFromGlobal(self._current_pos),
            ).normalized()

            # Dark overlay everywhere EXCEPT the selection rectangle.
            # QPainterPath with OddEven fill: outer rect + inner rect
            # → the intersection (selection) stays un-tinted.
            path = QPainterPath()
            path.addRect(QRectF(self.rect()))
            path.addRect(QRectF(sel_rect))
            painter.fillPath(path, overlay)

            # Green border
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
        else:
            # No selection yet — full dark overlay
            painter.fillRect(self.rect(), overlay)

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
                # Find which screen the start point is on for diagnostics
                for _s in QGuiApplication.screens():
                    _g = _s.geometry()
                    if _g.contains(self._start_pos.x(), self._start_pos.y()):
                        print(
                            f"[Region] Screen: {_s.name()} geo=({_g.x()},{_g.y()}) "
                            f"{_g.width()}x{_g.height()} DPR={_s.devicePixelRatio()}"
                        )
                        break
                print(
                    f"[Region] Qt start=({self._start_pos.x()},{self._start_pos.y()}) "
                    f"end=({end_pos.x()},{end_pos.y()}) → "
                    f"Physical ({x1},{y1}) {width}x{height}"
                )
                self.region_selected.emit(x1, y1, width, height)
            else:
                self.cancelled.emit()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.cancelled.emit()
