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
    select_region_triggered = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self._listener: keyboard.GlobalHotKeys | None = None
        self._toggle_hotkey: str = ""
        self._select_region_hotkey: str = ""

    def set_hotkey(self, hotkey_str: str) -> None:
        """Set the toggle start/stop hotkey from Qt-style string."""
        self._toggle_hotkey = hotkey_str
        self._restart_listener()

    def set_select_region_hotkey(self, hotkey_str: str) -> None:
        """Set the select region hotkey from Qt-style string."""
        self._select_region_hotkey = hotkey_str
        self._restart_listener()

    def _restart_listener(self) -> None:
        """Restart the global hotkey listener with all configured hotkeys."""
        self.stop()
        self.start()

    def start(self) -> None:
        hotkeys: dict[str, callable] = {}

        if self._toggle_hotkey:
            try:
                pynput_str = qt_hotkey_to_pynput(self._toggle_hotkey)
                hotkeys[pynput_str] = self._on_toggle
            except Exception:
                pass

        if self._select_region_hotkey:
            try:
                pynput_str = qt_hotkey_to_pynput(self._select_region_hotkey)
                hotkeys[pynput_str] = self._on_select_region
            except Exception:
                pass

        if not hotkeys:
            return

        try:
            self._listener = keyboard.GlobalHotKeys(hotkeys)
            self._listener.daemon = True
            self._listener.start()
        except Exception:
            pass

    def stop(self) -> None:
        if self._listener:
            self._listener.stop()
            self._listener = None

    def _on_toggle(self) -> None:
        self.hotkey_triggered.emit()

    def _on_select_region(self) -> None:
        self.select_region_triggered.emit()
