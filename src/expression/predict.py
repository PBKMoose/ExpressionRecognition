"""Inference: detect faces in an image (or webcam stream) and classify each expression.

Uses checkpoints/personal.pt if it exists (see expression.finetune), otherwise checkpoints/best.pt.

Usage:
    uv run python -m expression.predict path/to/photo.jpg
    uv run python -m expression.predict --webcam
    uv run python -m expression.predict --webcam --checkpoint checkpoints/best.pt
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from .camera import crop_face, detect_faces, face_crops, webcam_frames
from .config import BEST_CHECKPOINT, CLASSES, FACE_SIZE, PERSONAL_CHECKPOINT, get_device
from .data import eval_transform
from .model import load_trained


def default_checkpoint() -> Path:
    return PERSONAL_CHECKPOINT if PERSONAL_CHECKPOINT.exists() else BEST_CHECKPOINT


class ExpressionRecognizer:
    def __init__(self, checkpoint: Path | None = None, device=None):
        self.device = device or get_device()
        self.checkpoint = Path(checkpoint or default_checkpoint())
        self.model = load_trained(self.checkpoint, device=self.device)
        self.transform = eval_transform()

    @torch.no_grad()
    def classify_face(self, face: Image.Image) -> dict[str, float]:
        # Match the training data: FER2013 faces are 48x48 greyscale, upscaled by the transform.
        face = face.convert("L").resize((FACE_SIZE, FACE_SIZE), Image.BILINEAR)
        x = self.transform(face).unsqueeze(0).to(self.device)
        logits = self.model(x) + self.model(torch.flip(x, dims=[3]))
        probs = torch.softmax(logits / 2, dim=1)[0].cpu().tolist()
        return dict(zip(CLASSES, probs))

    def analyse(self, bgr: np.ndarray):
        """Return [(box, probs)] for every face found in a BGR image."""
        return [(box, self.classify_face(face)) for box, face in face_crops(bgr)]


COLOURS = {"happy": (60, 200, 60), "sad": (230, 140, 30), "neutral": (170, 170, 170)}  # BGR
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Bar panel layout (pixels).
BAR_W, BAR_H, BAR_GAP, NAME_W, PAD = 240, 30, 10, 105, 12


def put_text(img, text, org, scale, colour=(255, 255, 255), thickness=2):
    """White text with a dark outline so it stays readable on any background."""
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, colour, thickness, cv2.LINE_AA)


def draw(bgr, results):
    H, W = bgr.shape[:2]
    for (x0, y0, x1, y1), probs in results:
        label = max(probs, key=probs.get)
        colour = COLOURS.get(label, (60, 200, 60))

        # Thick face box, scaled with face size.
        thick = max(5, (x1 - x0) // 35)
        cv2.rectangle(bgr, (x0, y0), (x1, y1), colour, thick, cv2.LINE_AA)

        # Solid label tag above the box (inside it if there's no room above).
        tag = f"{label.upper()} {probs[label]:.0%}"
        (tw, th), _ = cv2.getTextSize(tag, FONT, 1.1, 3)
        ty = y0 - thick // 2 - th - 16 if y0 - th - 16 - thick > 0 else y0 + thick
        cv2.rectangle(bgr, (x0 - thick // 2, ty), (x0 + tw + 20, ty + th + 16), colour, -1)
        cv2.putText(bgr, tag, (x0 + 10, ty + th + 8), FONT, 1.1, (0, 0, 0), 3, cv2.LINE_AA)

        # Probability panel: a full-length track per class, filled to its probability.
        n = len(probs)
        pw = PAD * 2 + NAME_W + BAR_W
        ph = PAD * 2 + n * BAR_H + (n - 1) * BAR_GAP
        # Prefer below the box; otherwise beside it (right, then left), so it doesn't cover the face.
        px, py = min(max(0, x0), max(0, W - pw)), y1 + thick + 10
        if py + ph > H:
            py = min(max(0, y0), max(0, H - ph))
            if x1 + thick + 10 + pw <= W:
                px = x1 + thick + 10
            elif x0 - thick - 10 - pw >= 0:
                px = x0 - thick - 10 - pw
            else:
                py = max(0, H - ph)
        panel = bgr[py:py + ph, px:px + pw]
        panel[:] = (panel * 0.3).astype(panel.dtype)  # darken behind the bars

        for i, (name, p) in enumerate(probs.items()):
            c = COLOURS.get(name, (60, 200, 60))
            by = py + PAD + i * (BAR_H + BAR_GAP)
            bx = px + PAD + NAME_W
            put_text(bgr, name, (px + PAD, by + BAR_H - 8), 0.75, c if name != label else (255, 255, 255))
            cv2.rectangle(bgr, (bx, by), (bx + BAR_W, by + BAR_H), (70, 70, 70), -1)  # track
            cv2.rectangle(bgr, (bx, by), (bx + int(BAR_W * p), by + BAR_H), c, -1)  # fill
            if name == label:
                cv2.rectangle(bgr, (bx - 2, by - 2), (bx + BAR_W + 2, by + BAR_H + 2), (255, 255, 255), 2)
            put_text(bgr, f"{p:.0%}", (bx + BAR_W - 62, by + BAR_H - 8), 0.7)
    return bgr


def run_webcam(rec: ExpressionRecognizer, camera: int, hold_frames: int = 10):
    """Track the largest face, smoothing its box and probabilities so the overlay is steady."""
    box = probs = None
    missed = 0
    for frame in webcam_frames(camera):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray)
        if faces:
            missed = 0
            new_box, face = crop_face(gray, faces[0])  # largest face only
            new_probs = rec.classify_face(face)
            if box is None:
                box, probs = new_box, new_probs
            else:
                # Exponential moving averages: box glides instead of jittering, label doesn't flicker.
                box = tuple(int(0.6 * b + 0.4 * n) for b, n in zip(box, new_box))
                probs = {k: 0.7 * probs[k] + 0.3 * v for k, v in new_probs.items()}
        else:
            # Detector misses the odd frame; keep showing the last result briefly instead of blinking.
            missed += 1
            if missed > hold_frames:
                box = probs = None
        results = [(box, probs)] if box is not None else []
        cv2.imshow("Expression recognition (q to quit)", draw(frame, results))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", help="Path to an image")
    ap.add_argument("--webcam", action="store_true", help="Run live on the default camera (q to quit)")
    ap.add_argument("--camera", type=int, default=0, help="Camera index (try 1 if 0 is the wrong camera)")
    ap.add_argument("--checkpoint", type=Path, help="Model weights (default: personal.pt if present, else best.pt)")
    ap.add_argument("--whole-image", action="store_true", help="Skip face detection; classify the full image")
    args = ap.parse_args()

    rec = ExpressionRecognizer(args.checkpoint)
    print(f"Using model: {rec.checkpoint}")

    if args.webcam:
        run_webcam(rec, args.camera)
        return

    if not args.image:
        ap.error("provide an image path or --webcam")

    if args.whole_image:
        probs = rec.classify_face(Image.open(args.image))
        print({k: round(v, 3) for k, v in probs.items()})
        return

    bgr = cv2.imread(args.image)
    results = rec.analyse(bgr)
    if not results:
        print("No face detected; try --whole-image")
    for i, (box, probs) in enumerate(results):
        print(f"face {i} at {box}: {max(probs, key=probs.get)}  {({k: round(v, 3) for k, v in probs.items()})}")


if __name__ == "__main__":
    main()
