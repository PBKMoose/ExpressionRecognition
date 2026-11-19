"""Shared face detection, cropping and webcam handling."""

import cv2
import numpy as np
from PIL import Image

from .config import FACE_SIZE

_face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def _overlap(a, b) -> float:
    """Intersection area divided by the smaller box's area (catches one box nested inside another)."""
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return ix * iy / min(a[2] * a[3], b[2] * b[3])


def detect_faces(gray: np.ndarray):
    """Detect faces, largest first, merging duplicate detections of the same face."""
    found = _face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60))
    kept = []
    for box in sorted((tuple(map(int, b)) for b in found), key=lambda b: b[2] * b[3], reverse=True):
        if all(_overlap(box, k) < 0.3 for k in kept):
            kept.append(box)
    return kept


def crop_face(gray: np.ndarray, box) -> tuple[tuple[int, int, int, int], Image.Image]:
    """Pad a (x, y, w, h) box and return ((x0, y0, x1, y1), 48x48 greyscale face)."""
    x, y, w, h = box
    # Pad the box slightly: FER2013 crops include some forehead and chin.
    pad = int(0.1 * w)
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(gray.shape[1], x + w + pad), min(gray.shape[0], y + h + pad)
    # Reduce to FER2013's 48x48 greyscale so live crops look like the training data.
    face = Image.fromarray(gray[y0:y1, x0:x1]).resize((FACE_SIZE, FACE_SIZE), Image.BILINEAR)
    return (x0, y0, x1, y1), face


def face_crops(bgr: np.ndarray):
    """Return [(box, face)] for each distinct face, largest first."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return [crop_face(gray, box) for box in detect_faces(gray)]


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
