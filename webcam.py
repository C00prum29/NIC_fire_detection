import argparse #Убрать, разделить проект на 2 (Вебкамера и видео)
from pathlib import Path
import time

import cv2
import numpy as np

from settings import read_settings, setup_trackbars
from Color_pattern import find_color_candidates


RESULT_WINDOW = "Candidate regions (NOT confirmed fire)"
MASK_WINDOW = "Binary color mask"


def show_and_get_key(annotated, mask, delay):
    cv2.imshow(RESULT_WINDOW, annotated)
    cv2.imshow(MASK_WINDOW, mask)
    return cv2.waitKey(delay) & 0xFF


def process_camera(camera_index, output_dir):
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(
            f"Cannot open camera {camera_index}. Try --camera 1, "
            "or check camera permissions."
        )
    last_time = time.perf_counter()
    fps_smoothed = None
    print("CAMERA mode: change sliders, press S to save frame, Q or ESC to exit")
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                print("Camera frame could not be read; stopping.")
                break

            annotated, mask, _ = find_color_candidates(frame, read_settings())
            current_time = time.perf_counter()
            interval = current_time - last_time
            last_time = current_time
            if interval > 0:
                instantaneous_fps = 1.0 / interval
                fps_smoothed = (
                    instantaneous_fps if fps_smoothed is None
                    else 0.85 * fps_smoothed + 0.15 * instantaneous_fps
                )
            if fps_smoothed is not None:
                cv2.putText(
                    annotated,
                    f"FPS: {fps_smoothed:.1f}",
                    (8, 52),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            key = show_and_get_key(annotated, mask, 1)
            if key in (ord("q"), ord("Q"), 27):
                break
    finally:
        capture.release()

def process_video(video_path, output_dir):
    video_path = video_path.expanduser()

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS)

    if not np.isfinite(source_fps) or source_fps <= 0:
        source_fps = 25.0

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        capture.release()
        raise RuntimeError("Cannot determine video frame count")

# Создаём окно перемотки
    timeline_window = "Video timeline"

    cv2.namedWindow(timeline_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(timeline_window, 700, 100)

# Состояние перемотки
    seek_state = {
        "target": None,
        "updating": False
    }

# Вызывается при изменении ползунка
    def on_seek(position):
        if not seek_state["updating"]:
            seek_state["target"] = position

# Создаём ползунок
    cv2.createTrackbar(
        "Frame",
        timeline_window,
        0,
        total_frames - 1,
        on_seek
    )

    paused = False
    last_result = None
    
    print("VIDEO mode: press S to save, Q or ESC to exit")

    try:
        while True:
            seek_requested = seek_state["target"] is not None

            if seek_requested:
                target = seek_state["target"]
                seek_state["target"] = None

                capture.set(cv2.CAP_PROP_POS_FRAMES, target)

            if not paused or seek_requested or last_result is None:

                start_time = time.perf_counter()

                ret, frame = capture.read()

                if ret and frame is not None:

                    annotated, mask, _ = find_color_candidates(
                        frame, read_settings()
                    )

                    last_result = (annotated, mask)

                    current_frame = int(
                        capture.get(cv2.CAP_PROP_POS_FRAMES)
                    ) - 1

                    current_frame = max(
                        0, min(current_frame, total_frames - 1)
                    )

                    seek_state["updating"] = True

                    cv2.setTrackbarPos(
                        "Frame",
                        timeline_window,
                        current_frame
                    )

                    seek_state["updating"] = False

                    processing_ms = (
                        time.perf_counter() - start_time
                    ) * 1000

                else:
                    if last_result is None:
                        print("Cannot read video frame")
                        break

                    paused = True
                    processing_ms = 0

            else:
                processing_ms = 0

            annotated, mask = last_result

            if paused:
                delay = 30
            else:
                delay = max(
                    1,
                    round(1000 / source_fps - processing_ms)
                )

            key = show_and_get_key(annotated, mask, delay)

            if key in (ord("q"), ord("Q"), 27):
                break

            if key in (32, ord("p"), ord("P")):
                paused = not paused

    finally:
        cv2.destroyWindow(timeline_window)
        capture.release()


def main():
    parser = argparse.ArgumentParser(
        description="Part 1: detect COLOR candidates, not verified fire. No neural nets."
    )
    source = parser.add_mutually_exclusive_group()

    source.add_argument(
        "--camera",
        type=int,
        help="Webcam index"
    )

    source.add_argument(
        "--video",
        type=Path,
        help="Path to video file"
    )

    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent / "results",
        help="Directory for saved images and masks (default: results next to main.py)",
    )
    arguments = parser.parse_args()
    if (
        arguments.camera is None
        and arguments.video is None
    ):
        print("\nChoose video source:")
        print("1 - Webcam")
        print("2 - Video file")

        choice = input("Select mode (1/2): ").strip()

        if choice == "1":
            arguments.camera = 0

        elif choice == "2":
            path = input("Enter video path: ").strip()
            path = path.strip('"').strip("'")

            if not path:
                parser.error("Video path cannot be empty")

            arguments.video = Path(path)

        else:
            parser.error("Invalid mode")
    try:
        setup_trackbars()
        cv2.namedWindow(RESULT_WINDOW, cv2.WINDOW_NORMAL)
        cv2.namedWindow(MASK_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(RESULT_WINDOW, 960, 600)
        cv2.resizeWindow(MASK_WINDOW, 640, 400)
        if arguments.camera is not None:
            process_camera(arguments.camera, arguments.output_dir)

        elif arguments.video is not None:
            process_video(arguments.video, arguments.output_dir)

    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()