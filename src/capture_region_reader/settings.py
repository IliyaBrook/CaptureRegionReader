from __future__ import annotations

from dataclasses import dataclass, field
from PyQt6.QtCore import QSettings


@dataclass
class CaptureZone:
    name: str
    region: tuple[int, int, int, int] | None = None  # (left, top, width, height)
    hotkey: str = ""  # optional, Qt-style like "Ctrl+Alt+1"


@dataclass
class AppSettings:
    region: tuple[int, int, int, int] | None = None  # (left, top, width, height)
    language: str = "eng+rus"
    hotkey: str = "Ctrl+Alt+R"
    select_region_hotkey: str = "Ctrl+Alt+S"
    speech_rate: int = 150
    volume: float = 1.0
    ocr_interval_ms: int = 500
    tts_engine: str = "edge-tts"   # "silero" (local, RU) | "edge-tts" (cloud)
    growing_subtitles: bool = False  # Adaptive mode for word-by-word subtitles
    zones: list[CaptureZone] = field(default_factory=list)
    active_zone_index: int | None = None  # None = manual region, int = zone index

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
        s.setValue("tts_engine", self.tts_engine)
        s.setValue("growing_subtitles", self.growing_subtitles)

        # Save zones
        s.beginWriteArray("zones", len(self.zones))
        for i, zone in enumerate(self.zones):
            s.setArrayIndex(i)
            s.setValue("name", zone.name)
            s.setValue("hotkey", zone.hotkey)
            if zone.region:
                s.setValue("region_left", zone.region[0])
                s.setValue("region_top", zone.region[1])
                s.setValue("region_width", zone.region[2])
                s.setValue("region_height", zone.region[3])
            else:
                s.remove("region_left")
                s.remove("region_top")
                s.remove("region_width")
                s.remove("region_height")
        s.endArray()

        if self.active_zone_index is not None:
            s.setValue("active_zone_index", self.active_zone_index)
        else:
            s.remove("active_zone_index")

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

        # Load zones
        zones: list[CaptureZone] = []
        count = s.beginReadArray("zones")
        for i in range(count):
            s.setArrayIndex(i)
            name = str(s.value("name", f"Zone {i + 1}"))
            hotkey = str(s.value("hotkey", ""))
            zone_region = None
            if s.contains("region_left"):
                zone_region = (
                    int(s.value("region_left", 0)),
                    int(s.value("region_top", 0)),
                    int(s.value("region_width", 100)),
                    int(s.value("region_height", 100)),
                )
            zones.append(CaptureZone(name=name, region=zone_region, hotkey=hotkey))
        s.endArray()

        active_zone_index = None
        if s.contains("active_zone_index"):
            idx = int(s.value("active_zone_index", -1))
            if 0 <= idx < len(zones):
                active_zone_index = idx

        return cls(
            region=region,
            language=str(s.value("language", "eng+rus")),
            hotkey=str(s.value("hotkey", "Ctrl+Alt+R")),
            select_region_hotkey=str(s.value("select_region_hotkey", "Ctrl+Alt+S")),
            speech_rate=int(s.value("speech_rate", 150)),
            volume=float(s.value("volume", 1.0)),
            ocr_interval_ms=int(s.value("ocr_interval_ms", 500)),
            tts_engine=str(s.value("tts_engine", "edge-tts")),
            growing_subtitles=str(s.value("growing_subtitles", "false")).lower() in ("true", "1", "yes"),
            zones=zones,
            active_zone_index=active_zone_index,
        )
