"""
Overfit a single batch to sanity-check the model and data pipeline.
Loss should reach ~0.85 (label smoothing floor) within 200 steps.

Usage:
  python scripts/overfit_batch.py --traversal rowwise
"""

import argparse
import contextlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from vision_lstm.vision_lstm2 import VisionLSTM2
from vision_lstm.vision_lstm_traversal import SequenceTraversal

TRAVERSAL_MAP = {
    "rowwise": SequenceTraversal.ROWWISE_FROM_TOP_LEFT,
    "zigzag":  SequenceTraversal.ZIGZAG_FROM_TOP_LEFT,
    "spiral":  SequenceTraversal.SPIRAL_OUTWARD,
}

MEAN = (0.4802, 0.4481, 0.3975)
STD  = (0.2770, 0.2691, 0.2821)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--traversal",  choices=list(TRAVERSAL_MAP), default="rowwise")
    p.add_argument("--data-dir",   default="data/tiny-imagenet-200")
    p.add_argument("--amp",        action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.amp and device.type == "cuda"
        else contextlib.nullcontext()
    )

    model = VisionLSTM2(
        dim=192,
        input_shape=(3, 64, 64),
        patch_size=8,
        depth=12,
        output_shape=(200,),
        traversal=TRAVERSAL_MAP[args.traversal],
        conv_kind="causal1d",
    ).to(device)

    train_tf = transforms.Compose([
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    train_ds = datasets.ImageFolder(str(Path(args.data_dir) / "train"), train_tf)
    loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)

    print("Overfitting single batch (32 samples) — loss should reach ~0.85 (label smoothing floor) within 200 steps.")
    for step in range(1, 201):
        optimizer.zero_grad()
        with amp_ctx:
            loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        if step % 20 == 0:
            acc = (model(x).argmax(1) == y).float().mean().item()
            print(f"  step {step:3d}  loss {loss.item():.4f}  acc {acc:.3f}")


if __name__ == "__main__":
    main()
