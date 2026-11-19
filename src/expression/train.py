"""Fine-tune a pretrained backbone on FER2013 (happy / sad / neutral).

Usage: uv run python -m expression.train --epochs 15
"""

import argparse
import time

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from .config import BEST_CHECKPOINT, CHECKPOINT_DIR, CLASSES, get_device
from .data import load_splits
from .model import build_model


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        logits = model(x.to(device))
        # Test-time augmentation: average with horizontally flipped input.
        logits = logits + model(torch.flip(x, dims=[3]).to(device))
        preds += logits.argmax(1).cpu().tolist()
        targets += y.tolist()
    acc = sum(p == t for p, t in zip(preds, targets)) / len(targets)
    return acc, preds, targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = get_device()
    print(f"Device: {device}")
    train_ds, val_ds, test_ds = load_splits()
    print(f"Train {len(train_ds)} | Val {len(val_ds)} | Test {len(test_ds)}")

    loader_kw = dict(batch_size=args.batch_size, num_workers=args.workers, persistent_workers=args.workers > 0)
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kw)
    val_dl = DataLoader(val_ds, **loader_kw)
    test_dl = DataLoader(test_ds, **loader_kw)

    model = build_model().to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=train_ds.class_weights().to(device), label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps = args.epochs * len(train_dl)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=steps, pct_start=0.1)

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running, n = time.time(), 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += loss.item() * len(y)
            n += len(y)

        val_acc, _, _ = evaluate(model, val_dl, device)
        print(f"Epoch {epoch:2d}/{args.epochs} loss {running / n:.4f} val_acc {val_acc:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "classes": CLASSES, "val_acc": val_acc}, BEST_CHECKPOINT)
            print(f"  saved new best -> {BEST_CHECKPOINT}", flush=True)

    # Final evaluation of the best checkpoint on the held-out test split.
    model.load_state_dict(torch.load(BEST_CHECKPOINT, map_location=device)["model"])
    test_acc, preds, targets = evaluate(model, test_dl, device)
    print(f"\nBest val acc {best_acc:.4f} | Test acc {test_acc:.4f}\n")
    print(classification_report(targets, preds, target_names=CLASSES, digits=3))
    print("Confusion matrix (rows = true, cols = predicted):", CLASSES)
    print(confusion_matrix(targets, preds))


if __name__ == "__main__":
    main()
