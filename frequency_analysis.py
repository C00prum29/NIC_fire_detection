from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time

import cv2
import numpy as np
from scipy.signal import welch


@dataclass(frozen=True)
class FrequencyConfig:

    columns: int = 12                    # число ячеек по горизонтали
    rows: int = 8                        # число ячеек по вертикали
    history_seconds: float = 4.0         # длина сохраняемой истории, с
    minimum_seconds: float = 3.0         # минимум данных перед анализом, с
    analysis_every_n_frames: int = 8     # как часто пересчитывать спектр
    min_frequency_hz: float = 0.8        # нижняя частота эксперимента, Гц
    max_frequency_hz: float = 10.0       # верхняя частота эксперимента, Гц
    min_mean_brightness: float = 65.0    # средняя яркость ячейки, 0..255
    min_band_rms: float = 3.0            # RMS колебаний яркости в диапазоне
    min_band_fraction: float = 0.45      # доля спектральной энергии диапазона
    compensate_global_flicker: bool = True  # подавлять общие изменения света

    def __post_init__(self) -> None:
        if self.columns < 1 or self.rows < 1:
            raise ValueError("Grid dimensions must be positive")
        if not 0 < self.minimum_seconds <= self.history_seconds:
            raise ValueError("Invalid temporal window")
        if self.analysis_every_n_frames < 1:
            raise ValueError("analysis_every_n_frames must be positive")
        if not 0 < self.min_frequency_hz < self.max_frequency_hz:
            raise ValueError("Invalid frequency band")
        if not 0 <= self.min_mean_brightness <= 255:
            raise ValueError("min_mean_brightness must be 0..255")
        if self.min_band_rms < 0:
            raise ValueError("min_band_rms must be non-negative")
        if not 0 <= self.min_band_fraction <= 1:
            raise ValueError("min_band_fraction must be 0..1")


@dataclass
class FrequencyResult:

    gray: np.ndarray                     # чёрно-белый кадр, uint8
    annotated: np.ndarray                # GRAY -> BGR с рамками кандидатов
    mask: np.ndarray                     # бинарная маска ячеек с мерцанием
    ready: bool                          # достаточно ли данных для анализа
    candidate_count: int                 # число связных областей-кандидатов
    detected_tiles: int                  # число ячеек, прошедших критерии
    estimated_fps: float | None          # частота поступления кадров
    strongest_frequency_hz: float | None # частота наиболее сильного кандидата
    status: str                          # диагностический статус


class FrequencyAnalyzer:

    def __init__(self, config: FrequencyConfig | None = None) -> None:
        self.config = config or FrequencyConfig()
        self.reset()

    def reset(self) -> None:
        self._history: deque[tuple[float, np.ndarray, np.ndarray]] = deque()
        self._shape: tuple[int, int] | None = None
        self._frame_number = 0
        self._active_tiles = np.zeros(
            (self.config.rows, self.config.columns), dtype=bool
        )
        self._peak_hz = np.zeros(
            (self.config.rows, self.config.columns), dtype=np.float64
        )
        self._power_rms = np.zeros_like(self._peak_hz)
        self._estimated_fps: float | None = None
        self._status = "COLLECTING DATA"
        self._ready = False

    def _tile_means(self, gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape
        y_edges = np.linspace(0, h, self.config.rows + 1, dtype=int)
        x_edges = np.linspace(0, w, self.config.columns + 1, dtype=int)
        out = np.empty((self.config.rows, self.config.columns), dtype=np.float64)

        for row in range(self.config.rows):
            for col in range(self.config.columns):
                tile = gray[
                    y_edges[row]:y_edges[row + 1],
                    x_edges[col]:x_edges[col + 1],
                ]
                out[row, col] = float(tile.mean())
        return out

    def _analyze_history(self) -> None:
        self._ready = False
        self._active_tiles.fill(False)
        self._peak_hz.fill(0)
        self._power_rms.fill(0)

        if len(self._history) < 32:
            self._status = "COLLECTING DATA"
            return

        timestamps = np.fromiter(
            (item[0] for item in self._history), dtype=np.float64
        )
        if timestamps[-1] - timestamps[0] < self.config.minimum_seconds:
            self._status = "COLLECTING DATA"
            return

        intervals = np.diff(timestamps)
        dt = float(np.median(intervals))
        if dt <= 0 or not np.isfinite(dt):
            self._status = "INVALID TIMESTAMPS"
            return

        if np.any(intervals > 3 * dt):
            self._status = "DROPPED FRAMES / IRREGULAR TIME"
            return

        sample_count = max(2, int(round((timestamps[-1] - timestamps[0]) / dt)) + 1)
        uniform_times = np.linspace(timestamps[0], timestamps[-1], sample_count)
        self._estimated_fps = (sample_count - 1) / (
            uniform_times[-1] - uniform_times[0]
        )
        fs = self._estimated_fps
        high = min(self.config.max_frequency_hz, fs * 0.45)
        if high <= self.config.min_frequency_hz:
            self._status = "FPS TOO LOW FOR SELECTED FREQUENCIES"
            return

        raw = np.stack([item[1].ravel() for item in self._history], axis=0)
        corrected = np.stack([item[2].ravel() for item in self._history], axis=0)

        if np.allclose(
            np.diff(timestamps), np.diff(timestamps)[0], rtol=1e-3, atol=1e-6
        ) and len(timestamps) == sample_count:
            signal = corrected
        else:
            signal = np.stack(
                [np.interp(uniform_times, timestamps, corrected[:, j])
                 for j in range(corrected.shape[1])],
                axis=1,
            )

        nperseg = min(64, len(signal))
        frequencies, psd = welch(
            signal,
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=nperseg // 2,
            detrend="linear",
            scaling="density",
            axis=0,
        )

        frequency_mask = (
            (frequencies >= self.config.min_frequency_hz)
            & (frequencies <= high)
        )
        if not np.any(frequency_mask):
            self._status = "FREQUENCY RESOLUTION TOO LOW"
            return

        df = float(frequencies[1] - frequencies[0])
        band_power = psd[frequency_mask].sum(axis=0) * df
        total_power = psd[frequencies > 0].sum(axis=0) * df
        band_rms = np.sqrt(np.maximum(band_power, 0.0))
        band_fraction = band_power / (total_power + 1e-12)

        mean_brightness = raw[-min(8, len(raw)):].mean(axis=0)
        accepted = (
            (mean_brightness >= self.config.min_mean_brightness)
            & (band_rms >= self.config.min_band_rms)
            & (band_fraction >= self.config.min_band_fraction)
        )
        valid_freqs = frequencies[frequency_mask]
        peak_positions = np.argmax(psd[frequency_mask], axis=0)
        peak_freq = valid_freqs[peak_positions]

        self._active_tiles = accepted.reshape(self.config.rows, self.config.columns)
        self._peak_hz = peak_freq.reshape(self.config.rows, self.config.columns)
        self._power_rms = band_rms.reshape(self.config.rows, self.config.columns)
        self._ready = True
        self._status = "ANALYSIS READY"

    def process(self, frame: np.ndarray, timestamp: float | None = None) -> FrequencyResult:
        if frame.ndim == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif frame.ndim == 2:
            gray = frame
        else:
            raise ValueError("Expected BGR image HxWx3 or grayscale HxW")
        if gray.dtype != np.uint8:
            raise ValueError("Expected uint8 image")
        h, w = gray.shape
        if h < self.config.rows or w < self.config.columns:
            raise ValueError("Frame is smaller than the analysis grid")

        if self._shape is not None and self._shape != gray.shape:
            self.reset()               
        self._shape = gray.shape

        now = time.perf_counter() if timestamp is None else float(timestamp)
        if not np.isfinite(now):
            raise ValueError("timestamp must be finite")
        if self._history:
            dt = now - self._history[-1][0]
            if dt <= 0 or dt > max(0.5, self.config.history_seconds / 2):
                self.reset()           
                self._shape = gray.shape

        raw_tiles = self._tile_means(gray)
        if self.config.compensate_global_flicker:
            corrected = raw_tiles - float(np.median(raw_tiles))
        else:
            corrected = raw_tiles.copy()

        self._history.append((now, raw_tiles, corrected))
        while (
            len(self._history) > 1
            and now - self._history[0][0] > self.config.history_seconds
        ):
            self._history.popleft()
        self._frame_number += 1

        if not self._ready or self._frame_number % self.config.analysis_every_n_frames == 0:
            self._analyze_history()

        mask = np.zeros_like(gray, dtype=np.uint8)
        y_edges = np.linspace(0, h, self.config.rows + 1, dtype=int)
        x_edges = np.linspace(0, w, self.config.columns + 1, dtype=int)
        if self._ready:
            for row, col in np.argwhere(self._active_tiles):
                mask[
                    y_edges[row]:y_edges[row + 1],
                    x_edges[col]:x_edges[col + 1],
                ] = 255

        annotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            x, y, box_w, box_h = cv2.boundingRect(contour)
            cv2.rectangle(
                annotated, (x, y), (x + box_w, y + box_h), (0, 255, 255), 2
            )

        candidates = len(contours)
        active = int(np.count_nonzero(self._active_tiles)) if self._ready else 0
        peak = None
        if active:
            strongest = np.argmax(np.where(
                self._active_tiles, self._power_rms, -1.0
            ))
            peak = float(self._peak_hz.flat[strongest])

        cv2.rectangle(annotated, (0, 0), (w, 54), (0, 0, 0), -1)
        cv2.putText(
            annotated,
            f"Flicker candidates: {candidates}  |  NOT confirmed fire",
            (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1,
            cv2.LINE_AA,
        )
        diagnostics = self._status
        if peak is not None:
            diagnostics += f"  |  strongest: {peak:.2f} Hz"
        cv2.putText(
            annotated, diagnostics[:85], (8, 43),
            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA,
        )

        return FrequencyResult(
            gray=gray,
            annotated=annotated,
            mask=mask,
            ready=self._ready,
            candidate_count=candidates,
            detected_tiles=active,
            estimated_fps=self._estimated_fps,
            strongest_frequency_hz=peak,
            status=self._status,
        )
