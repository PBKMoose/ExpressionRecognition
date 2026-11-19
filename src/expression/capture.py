"""Record labelled frames of your own face for personal fine-tuning.

Usage: uv run python -m expression.capture

Keys (in the video window):
    h / s / n  start recording happy / sad / neutral (press again, or space, to stop)
    q          quit

Each start/stop is a "session". Record each expression in 3+ short sessions, varying head angle,
distance and lighting a little each time, and use the *natural, subtle* versions you want recognised.
Sessions are kept together when splitting train/validation, so validation is an honest test.
"""

import argparse
import time

import cv2

from .camera import face_crops, webcam_frames
from .config import CLASSES, PERSONAL_DATA_DIR

KEYS = {ord(c[0]): c for c in CLASSES}  # h -> happy, s -> sad, n -> neutral


def counts():
    return {c: len(list((PERSONAL_DATA_DIR / c).glob("*.png"))) for c in CLASSES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--every", type=int, default=3, help="Save every Nth frame (avoids near-duplicates)")
    args = ap.parse_args()

    for c in CLASSES:
        (PERSONAL_DATA_DIR / c).mkdir(parents=True, exist_ok=True)

    recording, session, frame_no = None, None, 0
    totals = counts()
    print("Press h / s / n to record happy / sad / neutral; space to stop; q to quit.")
    for frame in webcam_frames(args.camera):
        frame_no += 1
        faces = face_crops(frame)
        # Only save when exactly one face is visible, so the label is unambiguous.
        if recording and len(faces) == 1 and frame_no % args.every == 0:
            box, face = faces[0]
            face.save(PERSONAL_DATA_DIR / recording / f"{session}_{frame_no:06d}.png")
            totals[recording] += 1

        display = frame.copy()
        for (x0, y0, x1, y1), _ in faces:
            cv2.rectangle(display, (x0, y0), (x1, y1), (0, 0, 255) if recording else (0, 200, 0), 2)
        status = f"RECORDING {recording.upper()}" if recording else "idle"
        cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255) if recording else (0, 200, 0), 2)
        cv2.putText(display, "  ".join(f"{c}: {n}" for c, n in totals.items()), (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if len(faces) != 1:
            cv2.putText(display, "need exactly one face", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("Capture (h/s/n record, space stop, q quit)", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" ") or (key in KEYS and KEYS[key] == recording):
            recording = None
        elif key in KEYS:
            recording, session = KEYS[key], f"s{int(time.time())}"

    print("Saved frames per class:", counts())


if __name__ == "__main__":
    main()
