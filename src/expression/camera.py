"""Shared face detection, cropping and webcam handling."""

import cv2
import numpy as np
from PIL import Image

from .config import FACE_SIZE

_face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def face_crops(bgr: np.ndarray):
    """Return [(box, face)] where face is a FACE_SIZE x FACE_SIZE greyscale PIL image."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    out = []
    for (x, y, w, h) in _face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)):
        # Pad the box slightly: FER2013 crops include some forehead and chin.
        pad = int(0.1 * w)
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(bgr.shape[1], x + w + pad), min(bgr.shape[0], y + h + pad)
        # Reduce to FER2013's 48x48 greyscale so live crops look like the training data.
        face = Image.fromarray(gray[y0:y1, x0:x1]).resize((FACE_SIZE, FACE_SIZE), Image.BILINEAR)
        out.append(((x0, y0, x1, y1), face))
    return out


def webcam_frames(camera: int, max_failed_reads: int = 60):
    """Yield BGR frames from a camera, tolerating warm-up dropouts."""
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise SystemExit(
            f"Could not open camera {camera}. Check System Settings > Privacy & Security > Camera "
            "for your terminal/VS Code, or try --camera 1."
        )
    print(f"Camera {camera} opened. Press q in the video window to quit.")
    failed = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                failed += 1
                if failed >= max_failed_reads:
                    raise SystemExit(f"Camera {camera} stopped returning frames. Try --camera 1.")
                cv2.waitKey(50)
                continue
            failed = 0
            yield frame
    finally:
        cap.release()
        cv2.destroyAllWindows()
