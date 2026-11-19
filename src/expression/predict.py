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

from .camera import face_crops, webcam_frames
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


COLOURS = {"happy": (0, 200, 0), "sad": (220, 120, 0), "neutral": (180, 180, 180)}  # BGR


def draw(bgr, results):
    for (x0, y0, x1, y1), probs in results:
        label = max(probs, key=probs.get)
        colour = COLOURS.get(label, (0, 200, 0))
        cv2.rectangle(bgr, (x0, y0), (x1, y1), colour, 2)
        cv2.putText(bgr, f"{label} {probs[label]:.0%}", (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
        # A bar per class so you can see how close the runner-up is.
        for i, (name, p) in enumerate(probs.items()):
            y = y1 + 20 + i * 22
            c = COLOURS.get(name, (0, 200, 0))
            cv2.rectangle(bgr, (x0, y - 12), (x0 + int(150 * p), y + 4), c, -1)
            cv2.putText(bgr, f"{name} {p:.0%}", (x0 + 155, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
    return bgr


def run_webcam(rec: ExpressionRecognizer, camera: int):
    smoothed = None
    for frame in webcam_frames(camera):
        results = rec.analyse(frame)
        if len(results) == 1:
            # Smooth over frames to stop the label flickering (single-face case).
            box, probs = results[0]
            smoothed = {k: 0.7 * smoothed.get(k, v) + 0.3 * v for k, v in probs.items()} if smoothed else probs
            results = [(box, smoothed)]
        else:
            smoothed = None
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
