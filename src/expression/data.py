"""Dataset loading and preprocessing."""

from collections import Counter

import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from .config import CLASSES, DATASET_ID, IMAGE_SIZE, SOURCE_LABELS

# ImageNet statistics, matching the pretrained backbone.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def train_transform() -> T.Compose:
    return T.Compose([
        T.Grayscale(num_output_channels=3),
        T.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.3, contrast=0.3),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
        T.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])


def eval_transform() -> T.Compose:
    return T.Compose([
        T.Grayscale(num_output_channels=3),
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


class ExpressionDataset(Dataset):
    """Wraps a Hugging Face split, keeping only our classes and remapping labels."""

    def __init__(self, hf_split, transform):
        names = hf_split.features["label"].names
        src_to_ours = {names.index(SOURCE_LABELS[c]): i for i, c in enumerate(CLASSES)}
        keep = [i for i, lbl in enumerate(hf_split["label"]) if lbl in src_to_ours]
        self.ds = hf_split.select(keep)
        self.labels = [src_to_ours[lbl] for lbl in self.ds["label"]]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx):
        img: Image.Image = self.ds[idx]["image"]
        return self.transform(img.convert("L")), self.labels[idx]

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights to counter class imbalance."""
        counts = Counter(self.labels)
        total = len(self.labels)
        return torch.tensor([total / (len(CLASSES) * counts[i]) for i in range(len(CLASSES))])


def load_splits():
    raw = load_dataset(DATASET_ID)
    return (
        ExpressionDataset(raw["train"], train_transform()),
        ExpressionDataset(raw["publicTest"], eval_transform()),
        ExpressionDataset(raw["privateTest"], eval_transform()),
    )
