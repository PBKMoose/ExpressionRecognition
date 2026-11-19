"""Adapt the FER2013 model to your own face using frames recorded with expression.capture.

Usage: uv run python -m expression.finetune --epochs 6

Starts from checkpoints/best.pt and writes checkpoints/personal.pt (which predict then uses by default).
Mixes in FER2013 images so the model keeps working on other faces, and reports accuracy on
held-out recording sessions before and after, so you can see whether it actually helped.
"""

import argparse
import random
from collections import defaultdict

import torch
from PIL import Image
from sklearn.metrics import confusion_matrix
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from .config import BEST_CHECKPOINT, CHECKPOINT_DIR, CLASSES, PERSONAL_CHECKPOINT, PERSONAL_DATA_DIR, get_device
from .data import eval_transform, load_splits, train_transform
from .model import load_trained
from .train import evaluate


class PersonalDataset(Dataset):
    def __init__(self, items, transform):
        self.items, self.transform = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        return self.transform(Image.open(path).convert("L")), label


def split_by_session(val_frac: float, seed: int = 0):
    """Hold out whole recording sessions per class, so validation frames aren't near-copies of training ones."""
    rng = random.Random(seed)
    train, val = [], []
    for label, cls in enumerate(CLASSES):
        sessions = defaultdict(list)
        for p in sorted((PERSONAL_DATA_DIR / cls).glob("*.png")):
            sessions[p.stem.split("_")[0]].append((p, label))
        if not sessions:
            raise SystemExit(f"No frames for '{cls}' in {PERSONAL_DATA_DIR}. Run: uv run python -m expression.capture")
        names = sorted(sessions)
        if len(names) >= 2:
            rng.shuffle(names)
            n_val = max(1, round(len(names) * val_frac))
            val += [i for s in names[:n_val] for i in sessions[s]]
            train += [i for s in names[n_val:] for i in sessions[s]]
        else:
            # Single session: fall back to holding out the last part of it in time.
            frames = sessions[names[0]]
            cut = int(len(frames) * (1 - val_frac))
            train += frames[:cut]
            val += frames[cut:]
            print(f"  note: only one session for '{cls}'; record more for a more trustworthy validation score")
    return train, val


def report(name, acc, preds, targets):
    print(f"{name}: acc {acc:.3f}")
    print("  confusion (rows = true, cols = predicted):", CLASSES)
    for row in confusion_matrix(targets, preds, labels=range(len(CLASSES))):
        print("   ", row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--repeat", type=int, default=3, help="Times to repeat your frames per epoch")
    ap.add_argument("--fer-ratio", type=float, default=1.0, help="FER2013 images per personal image (anti-forgetting)")
    args = ap.parse_args()

    device = get_device()
    train_items, val_items = split_by_session(args.val_frac)
    print(f"Personal frames: train {len(train_items)} | val {len(val_items)}")

    fer_train, fer_val, _ = load_splits()
    personal_train = ConcatDataset([PersonalDataset(train_items, train_transform())] * args.repeat)
    n_fer = min(len(fer_train), int(len(personal_train) * args.fer_ratio))
    fer_subset = Subset(fer_train, random.Random(1).sample(range(len(fer_train)), n_fer))
    train_dl = DataLoader(ConcatDataset([personal_train, fer_subset]), batch_size=args.batch_size,
                          shuffle=True, drop_last=True, num_workers=2)
    personal_val_dl = DataLoader(PersonalDataset(val_items, eval_transform()), batch_size=64)
    fer_val_dl = DataLoader(fer_val, batch_size=64, num_workers=2)

    model = load_trained(BEST_CHECKPOINT, device=device)
    before = evaluate(model, personal_val_dl, device)
    fer_before = evaluate(model, fer_val_dl, device)[0]
    report("BEFORE  your held-out sessions", *before)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    best_acc, best_state = before[0], None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
            n += len(y)
        acc = evaluate(model, personal_val_dl, device)[0]
        print(f"Epoch {epoch}/{args.epochs} loss {total / n:.4f} personal_val_acc {acc:.3f}", flush=True)
        if acc >= best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        print("\nFine-tuning did not beat the original model on your held-out sessions; personal.pt not written.")
        print("Try recording more (and more varied) sessions, then run again.")
        return

    model.load_state_dict(best_state)
    report("AFTER   your held-out sessions", *evaluate(model, personal_val_dl, device))
    fer_after = evaluate(model, fer_val_dl, device)[0]
    print(f"FER2013 val acc (other people's faces): {fer_before:.3f} -> {fer_after:.3f}")

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    torch.save({"model": best_state, "classes": CLASSES, "val_acc": best_acc}, PERSONAL_CHECKPOINT)
    print(f"\nSaved {PERSONAL_CHECKPOINT}; predict will now use it by default.")


if __name__ == "__main__":
    main()
