import cv2
import numpy as np
from settings import read_settings

def find_color_candidates(frame, settings):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low_hsv = (0, settings["s_min"], settings["v_min"])
    high_hsv = (settings["h_max"], 255, 255)
    primary_mask = cv2.inRange(hsv, low_hsv, high_hsv)

    red_mask = cv2.inRange(
        hsv,
        (170, settings["s_min"], settings["v_min"]),
        (179, 255, 255),
    )
    mask = cv2.bitwise_or(primary_mask, red_mask)

    if settings["use_ycrcb"]:
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        ycrcb_mask = cv2.inRange(
            ycrcb,
            (0, settings["cr_min"], 0),
            (255, 255, settings["cb_max"]),
        )
        mask = cv2.bitwise_and(mask, ycrcb_mask)

    open_kernel = np.ones((3, 3), dtype=np.uint8)
    close_kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    annotated = frame.copy()
    accepted = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < settings["min_area"]:
            continue
        accepted += 1
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(
            annotated,
            f"Color candidate {accepted}",
            (x, max(22, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 31), (0, 0, 0), -1)
    cv2.putText(
        annotated,
        f"Color candidates: {accepted}  |  Not confirmed fire",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return annotated, mask, accepted