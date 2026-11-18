"""Захват и воспроизведение видео с независимой цветовой и частотной детекцией.

Частотный детектор работает только на новых кадрах. Пауза не дублирует
измерения во временной истории; перемотка полностью сбрасывает историю.
"""

from pathlib import Path
import time

import cv2
import numpy as np

from color_pattern import find_color_candidates
from frequency_analysis import FrequencyAnalyzer
from settings import read_settings
from vis import save_results, show_and_get_key


EXIT_KEYS = (ord("q"), ord("Q"), 27)
SAVE_KEYS = (ord("s"), ord("S"))
PAUSE_KEYS = (32, ord("p"), ord("P"))


def process_camera(camera_index, output_dir):
    """Веб-камера: параллельные HSV/YCrCb и анализ мерцания в GRAY."""
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"Cannot open camera {camera_index}. Check camera index or permissions."
        )

    frequency_detector = FrequencyAnalyzer()
    last_time = None
    fps_smoothed = None
    print("CAMERA: S — save, Q/ESC — exit. Flicker != confirmed fire.")

    try:
        while True:
            ok, frame = capture.read()
            timestamp = time.perf_counter()  # сразу после получения кадра
            if not ok or frame is None:
                print("Camera frame could not be read; stopping.")
                break

            # Модули независимы: не подменяем маску цвета маской мерцания.
            annotated, mask, _ = find_color_candidates(frame, read_settings())
            frequency_result = frequency_detector.process(frame, timestamp=timestamp)

            if last_time is not None:
                interval = timestamp - last_time
                if interval > 0:
                    instantaneous_fps = 1.0 / interval
                    fps_smoothed = (
                        instantaneous_fps if fps_smoothed is None
                        else 0.85 * fps_smoothed + 0.15 * instantaneous_fps
                    )
            last_time = timestamp

            if fps_smoothed is not None:
                cv2.putText(
                    annotated, f"FPS: {fps_smoothed:.1f}", (8, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255),
                    2, cv2.LINE_AA,
                )

            key = show_and_get_key(annotated, mask, 1, frequency_result)
            if key in EXIT_KEYS:
                break
            if key in SAVE_KEYS:
                save_results(annotated, mask, frequency_result, output_dir)
    finally:
        capture.release()


def process_video(video_path, output_dir):
    """Видеофайл: перемотка, пауза, обработка без поддельных кадров."""
    video_path = Path(video_path).expanduser()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open video: {video_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(source_fps) or source_fps <= 0:
        print("Warning: unknown video FPS; using 25 FPS for timestamps/playback")
        source_fps = 25.0

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        capture.release()
        raise RuntimeError("Cannot determine video frame count for seeking")

    frequency_detector = FrequencyAnalyzer()
    timeline_window = "Video timeline"
    cv2.namedWindow(timeline_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(timeline_window, 700, 100)
    seek_state = {"target": None, "updating": False}

    def on_seek(position):
        # Изменение ползунка самой программой не считается перемоткой.
        if not seek_state["updating"]:
            seek_state["target"] = position

    cv2.createTrackbar("Frame", timeline_window, 0, total_frames - 1, on_seek)

    paused = False
    last_frame = None
    last_color_result = None
    last_settings = None
    last_frequency_result = None

    print("VIDEO: Space/P — pause, timeline — seek, S — save, Q/ESC — exit.")
    print("Flicker analysis needs several seconds of successive frames.")

    try:
        while True:
            seek_requested = seek_state["target"] is not None
            if seek_requested:
                target = seek_state["target"]
                seek_state["target"] = None
                frequency_detector.reset()  # нельзя смешивать старую и новую историю
                if not capture.set(cv2.CAP_PROP_POS_FRAMES, target):
                    print(f"Warning: seeking to frame {target} was not supported")
                    paused = True
                    seek_requested = False

            processing_ms = 0.0
            if not paused or seek_requested or last_frame is None:
                start_time = time.perf_counter()
                ok, frame = capture.read()
                if ok and frame is not None:
                    last_frame = frame
                    current_frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                    current_frame = max(0, min(current_frame, total_frames - 1))

                    # Видео с постоянным FPS: позиция на шкале видеозаписи,
                    # а не момент вызова функции / задержки OpenCV.
                    timestamp = current_frame / source_fps
                    last_frequency_result = frequency_detector.process(
                        frame, timestamp=timestamp
                    )
                    last_settings = read_settings()
                    last_color_result = find_color_candidates(frame, last_settings)

                    seek_state["updating"] = True
                    try:
                        cv2.setTrackbarPos("Frame", timeline_window, current_frame)
                    finally:
                        seek_state["updating"] = False

                    processing_ms = (time.perf_counter() - start_time) * 1000
                else:
                    if last_frame is None:
                        print("Cannot read video frame")
                        break
                    # При достижении конца файла показываем последний кадр.
                    paused = True

            if last_frame is None or last_color_result is None:
                break

            # На паузе можно менять пороги HSV/YCrCb — текущий кадр
            # пересчитывается, но в частотную историю НЕ добавляется.
            settings = read_settings()
            if settings != last_settings:
                last_settings = settings
                last_color_result = find_color_candidates(last_frame, settings)

            annotated, mask, _ = last_color_result
            if paused:
                delay = 30
            else:
                delay = max(1, round(1000 / source_fps - processing_ms))

            key = show_and_get_key(
                annotated, mask, delay, last_frequency_result
            )
            if key in EXIT_KEYS:
                break
            if key in PAUSE_KEYS:
                paused = not paused
            if key in SAVE_KEYS:
                save_results(annotated, mask, last_frequency_result, output_dir)

    finally:
        capture.release()
        cv2.destroyWindow(timeline_window)
