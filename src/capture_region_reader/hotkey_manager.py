from __future__ import annotations

from pynput import keyboard
from PyQt6.QtCore import QObject, pyqtSignal


def qt_hotkey_to_pynput(hotkey_str: str) -> str:
    """Convert Qt-style hotkey string to pynput format.

    Example: 'Ctrl+Alt+Shift+F1' -> '<ctrl>+<alt>+<shift>+<F1>'
    Example: 'Ctrl+R' -> '<ctrl>+r'
    """
    mapping = {
        "Ctrl": "<ctrl>",
        "Alt": "<alt>",
        "Shift": "<shift>",
        "Meta": "<cmd>",
        "Super": "<cmd>",
    }

    parts = [p.strip() for p in hotkey_str.split("+")]
    converted = []
    for part in parts:
        if part in mapping:
            converted.append(mapping[part])
        elif len(part) == 1:
            converted.append(part.lower())
        else:
            # Function keys and special keys
            converted.append(f"<{part.lower()}>")
    return "+".join(converted)


class HotkeyManager(QObject):
    hotkey_triggered = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self._listener: keyboard.GlobalHotKeys | None = None
        self._hotkey_str: str = ""

    def set_hotkey(self, hotkey_str: str) -> None:
        """Set hotkey from Qt-style string like 'Ctrl+Alt+R'."""
        self.stop()
        self._hotkey_str = hotkey_str
        self.start()

    def start(self) -> None:
        if not self._hotkey_str:
            return
        try:
            pynput_str = qt_hotkey_to_pynput(self._hotkey_str)
            self._listener = keyboard.GlobalHotKeys({
                pynput_str: self._on_hotkey,
            })
            self._listener.daemon = True
            self._listener.start()
        except Exception:
            pass

    def stop(self) -> None:
        if self._listener:
            self._listener.stop()
            self._listener = None

    def _on_hotkey(self) -> None:
        self.hotkey_triggered.emit()
