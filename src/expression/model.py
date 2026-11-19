"""Model construction and checkpoint loading."""

import timm
import torch

from .config import BACKBONE, BEST_CHECKPOINT, CLASSES


def build_model(pretrained: bool = True) -> torch.nn.Module:
    return timm.create_model(BACKBONE, pretrained=pretrained, num_classes=len(CLASSES), drop_path_rate=0.1)


def load_trained(path=BEST_CHECKPOINT, device="cpu") -> torch.nn.Module:
    ckpt = torch.load(path, map_location=device)
    if ckpt["classes"] != CLASSES:
        raise ValueError(f"Checkpoint classes {ckpt['classes']} != config classes {CLASSES}")
    model = build_model(pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()
