"""Shared configuration for training and inference."""

from pathlib import Path

import torch

# Classes we recognise. Extend this list (and SOURCE_LABELS) to add more expressions.
CLASSES = ["happy", "sad", "neutral"]

# Source dataset: FER2013 with its official train / publicTest / privateTest splits.
DATASET_ID = "Aaryan333/fer2013_train_publicTest_privateTest"
# Map our class name -> label name used by the source dataset.
SOURCE_LABELS = {"happy": "Happy", "sad": "Sad", "neutral": "Neutral"}

IMAGE_SIZE = 160
BACKBONE = "convnext_tiny.fb_in22k"  # ImageNet-22k pretrained ConvNeXt

CHECKPOINT_DIR = Path("checkpoints")
BEST_CHECKPOINT = CHECKPOINT_DIR / "best.pt"


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# Personal adaptation: your own labelled webcam frames, and the model fine-tuned on them.
PERSONAL_DATA_DIR = Path("data/personal")
PERSONAL_CHECKPOINT = CHECKPOINT_DIR / "personal.pt"
FACE_SIZE = 48  # FER2013 resolution; webcam crops are reduced to this to match training data.
