import argparse
from pathlib import Path

import cv2

from settings import setup_trackbars
from video_stream import process_camera, process_video
from vis import setup_result_windows


def main():
    parser = argparse.ArgumentParser(
        description="FireVision: color + grayscale flicker candidates, no neural nets."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, help="Webcam index (usually 0)")
    source.add_argument("--video", type=Path, help="Path to video file")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Directory for saved image/mask pairs",
    )
    arguments = parser.parse_args()

    if arguments.camera is None and arguments.video is None:
        print("\nChoose video source:")
        print("1 - Webcam")
        print("2 - Video file")
        choice = input("Select mode (1/2): ").strip()
        if choice == "1":
            arguments.camera = 0
        elif choice == "2":
            path = input("Enter video path: ").strip().strip('"').strip("'")
            if not path:
                parser.error("Video path cannot be empty")
            arguments.video = Path(path)
        else:
            parser.error("Invalid mode")

    try:
        setup_trackbars()
        setup_result_windows()
        if arguments.camera is not None:
            process_camera(arguments.camera, arguments.output_dir)
        else:
            process_video(arguments.video, arguments.output_dir)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
