from __future__ import annotations

from dataclasses import dataclass
from PyQt6.QtCore import QSettings


@dataclass
class AppSettings:
    region: tuple[int, int, int, int] | None = None  # (left, top, width, height)
    language: str = "eng+rus"
    hotkey: str = "Ctrl+Alt+R"
    select_region_hotkey: str = "Ctrl+Alt+S"
    speech_rate: int = 150
    volume: float = 1.0
    ocr_interval_ms: int = 500

    def save(self) -> None:
        s = QSettings("CaptureRegionReader", "CaptureRegionReader")
        if self.region:
            s.setValue("region/left", self.region[0])
            s.setValue("region/top", self.region[1])
            s.setValue("region/width", self.region[2])
            s.setValue("region/height", self.region[3])
        else:
            s.remove("region")
        s.setValue("language", self.language)
        s.setValue("hotkey", self.hotkey)
        s.setValue("select_region_hotkey", self.select_region_hotkey)
        s.setValue("speech_rate", self.speech_rate)
        s.setValue("volume", self.volume)
        s.setValue("ocr_interval_ms", self.ocr_interval_ms)

    @classmethod
    def load(cls) -> AppSettings:
        s = QSettings("CaptureRegionReader", "CaptureRegionReader")
        region = None
        if s.contains("region/left"):
            region = (
                int(s.value("region/left", 0)),
                int(s.value("region/top", 0)),
                int(s.value("region/width", 100)),
                int(s.value("region/height", 100)),
            )
        return cls(
            region=region,
            language=str(s.value("language", "eng+rus")),
            hotkey=str(s.value("hotkey", "Ctrl+Alt+R")),
            select_region_hotkey=str(s.value("select_region_hotkey", "Ctrl+Alt+S")),
            speech_rate=int(s.value("speech_rate", 150)),
            volume=float(s.value("volume", 1.0)),
            ocr_interval_ms=int(s.value("ocr_interval_ms", 500)),
        )
