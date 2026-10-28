import cv2

def _on_trackbar(_value):
        """Trackbars are polled while the program is running."""

def setup_trackbars():
    cv2.namedWindow("Color settings", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Color settings", 460, 380)
    initial = (
        ("H max", 40, 70), #Снизить 70 до 60 для повышения точности и экспериментально откалибровать максимально допустимый цветовой тон
        ("S min", 85, 255), #Экспериментально откалибровать минимальную насыщенность
        ("V min", 140, 255), #Экспериментально откалибровать минимальную яркость
        ("Min area px", 200, 20000), #Экспериментально подобрать значения размера области (приоритет на маленькие)
        ("Cr min", 135, 255), #Экспериментально откалибровать минимальную красную составляющую
        ("Cb max", 170, 255), #Экспериментально откалибровать максимальную синюю составляющую
    )
    for name, value, maximum in initial:
        cv2.createTrackbar(name, "Color settings", value, maximum, _on_trackbar)

def read_settings():
    def val(name):
        return cv2.getTrackbarPos(name, "Color settings")

    return {
        "h_max": val("H max"),
        "s_min": val("S min"),
        "v_min": val("V min"),
        "min_area": val("Min area px"),
        "use_ycrcb": True,
        "cr_min": val("Cr min"),
        "cb_max": val("Cb max"),
    }