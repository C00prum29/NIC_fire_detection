from datetime import datetime
from pathlib import Path

import cv2


RESULT_WINDOW = "Candidate regions (NOT confirmed fire)"
MASK_WINDOW = "Binary color mask"
FREQUENCY_WINDOW = "Frequency analysis (NOT confirmed fire)"
FREQUENCY_MASK_WINDOW = "Frequency mask"


def setup_result_windows():
    for title, width, height in (
        (RESULT_WINDOW, 960, 600),
        (MASK_WINDOW, 640, 400),
        (FREQUENCY_WINDOW, 960, 600),
        (FREQUENCY_MASK_WINDOW, 640, 400),
    ):
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, width, height)


def show_and_get_key(annotated, mask, delay, frequency_result=None):
    cv2.imshow(RESULT_WINDOW, annotated)
    cv2.imshow(MASK_WINDOW, mask)
    if frequency_result is not None:
        cv2.imshow(FREQUENCY_WINDOW, frequency_result.annotated)
        cv2.imshow(FREQUENCY_MASK_WINDOW, frequency_result.mask)
    return cv2.waitKey(max(1, int(delay))) & 0xFF


def save_results(annotated, mask, frequency_result, output_dir):
    folder = Path(output_dir).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    images = {
        "color_result": annotated,
        "color_mask": mask,
    }
    if frequency_result is not None:
        images["frequency_result"] = frequency_result.annotated
        images["frequency_mask"] = frequency_result.mask

    written = []
    for suffix, image in images.items():
        target = folder / f"{stamp}_{suffix}.png"
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError(f"Failed to encode: {target}")
        target.write_bytes(encoded.tobytes())
        written.append(target)

    print(f"Saved {len(written)} images to {folder}")
    return written
