"""
Train VisionLSTM2 on Tiny ImageNet with a specified patch traversal strategy.

Usage:
  python train.py --traversal rowwise
  python train.py --traversal zigzag --epochs 100 --batch-size 256
  python train.py --traversal spiral  --epochs 100 --batch-size 256

Results saved to experiments/<traversal>_<timestamp>/
  config.json      — all hyperparameters
  train_log.csv    — per-epoch metrics
  model_best.pth   — checkpoint at best val top-1
  model_final.pth  — checkpoint after last epoch
  eval_results.json
  plots/curves.png
"""

import argparse
import csv
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from vision_lstm.vision_lstm2 import VisionLSTM2
from vision_lstm.vision_lstm_traversal import SequenceTraversal

TRAVERSAL_MAP = {
    "rowwise": SequenceTraversal.ROWWISE_FROM_TOP_LEFT,
    "zigzag":  SequenceTraversal.ZIGZAG_FROM_TOP_LEFT,
    "spiral":  SequenceTraversal.SPIRAL_CLOCKWISE,
}

# Tiny ImageNet channel statistics
MEAN = (0.4802, 0.4481, 0.3975)
STD  = (0.2770, 0.2691, 0.2821)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--traversal",      choices=list(TRAVERSAL_MAP), required=True)
    p.add_argument("--data-dir",       default="data/tiny-imagenet-200")
    p.add_argument("--exp-dir",        default="experiments")
    p.add_argument("--epochs",         type=int,   default=100)
    p.add_argument("--batch-size",     type=int,   default=256)
    p.add_argument("--lr",             type=float, default=2.5e-4)
    p.add_argument("--weight-decay",   type=float, default=0.05)
    p.add_argument("--warmup-epochs",  type=int,   default=5)
    p.add_argument("--drop-path",      type=float, default=0.0)
    p.add_argument("--workers",        type=int,   default=4)
    p.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--overfit-batch",   action="store_true",
                   help="Overfit a single batch to sanity-check the model.")
    p.add_argument("--find-batch-size", action="store_true",
                   help="Estimate the maximum safe batch size and exit.")
    return p.parse_args()


def setup_experiment(args) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(args.exp_dir) / f"{args.traversal}_{timestamp}"
    (exp_dir / "plots").mkdir(parents=True)
    return exp_dir


def get_dataloaders(args):
    train_tf = transforms.Compose([
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_ds = datasets.ImageFolder(os.path.join(args.data_dir, "train"), train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(args.data_dir, "val"),   val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )
    return train_loader, val_loader


def build_model(args) -> VisionLSTM2:
    return VisionLSTM2(
        dim=192,
        input_shape=(3, 64, 64),
        patch_size=8,
        depth=12,
        output_shape=(200,),
        traversal=TRAVERSAL_MAP[args.traversal],
        conv_kind="causal1d",
        drop_path_rate=args.drop_path,
    )


def build_optimizer_scheduler(args, model, steps_per_epoch):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    total_steps  = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, epoch, total_epochs):
    model.train()
    total_loss = correct = total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:3d}/{total_epochs} [train]", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        correct    += (logits.detach().argmax(1) == y).sum().item()
        total      += bs
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.3f}")

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, epoch, total_epochs):
    model.eval()
    total_loss = correct_top1 = correct_top5 = total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:3d}/{total_epochs} [val]  ", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * y.size(0)
        top5 = logits.topk(5, dim=1).indices
        correct_top1 += (top5[:, 0] == y).sum().item()
        correct_top5 += (top5 == y.unsqueeze(1)).any(1).sum().item()
        total += y.size(0)

    return total_loss / total, correct_top1 / total, correct_top5 / total


def save_plots(log: list, exp_dir: Path):
    epochs = [r["epoch"] for r in log]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, [r["train_loss"] for r in log], label="train")
    ax1.plot(epochs, [r["val_loss"]   for r in log], label="val")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Loss"); ax1.legend()

    ax2.plot(epochs, [r["train_acc"] for r in log], label="train")
    ax2.plot(epochs, [r["val_top1"]  for r in log], label="val top-1")
    ax2.plot(epochs, [r["val_top5"]  for r in log], label="val top-5")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.set_title("Accuracy"); ax2.legend()

    fig.suptitle(exp_dir.name)
    fig.tight_layout()
    fig.savefig(exp_dir / "plots" / "curves.png", dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    if args.find_batch_size:
        if not torch.cuda.is_available():
            print("--find-batch-size requires a CUDA device.")
            return
        model = build_model(args).to(device)
        criterion_tmp = nn.CrossEntropyLoss(label_smoothing=0.1)

        def measure_peak_mb(batch_size):
            model.zero_grad()
            torch.cuda.reset_peak_memory_stats(device)
            x = torch.randn(batch_size, 3, 64, 64, device=device)
            y = torch.zeros(batch_size, dtype=torch.long, device=device)
            criterion_tmp(model(x), y).backward()
            return torch.cuda.max_memory_allocated(device) / 1024 ** 2

        mem1 = measure_peak_mb(1)
        mem2 = measure_peak_mb(2)
        per_sample_mb = mem2 - mem1
        fixed_mb      = mem1 - per_sample_mb
        free_mb = torch.cuda.mem_get_info(device)[0] / 1024 ** 2
        max_batch = int((free_mb * 0.8 - fixed_mb) / per_sample_mb)
        max_batch = 2 ** int(math.log2(max_batch))
        base_lr = args.lr
        scaled_lr = base_lr * max_batch / args.batch_size
        print(f"Fixed overhead    : {fixed_mb:.0f} MB  (model + grads; excludes AdamW state ~47MB covered by 0.8 margin)")
        print(f"Per sample        : {per_sample_mb:.1f} MB")
        print(f"Free GPU memory   : {free_mb:.0f} MB")
        print(f"Suggested batch   : {max_batch}")
        print(f"Scaled LR         : {scaled_lr:.2e}  (linear scaling from {base_lr:.2e} at batch {args.batch_size})")
        return

    if args.overfit_batch:
        model = build_model(args).to(device)
        train_loader, _ = get_dataloaders(args)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        x, y = next(iter(train_loader))
        x, y = x[:32].to(device), y[:32].to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        print("Overfitting single batch (32 samples) — loss should reach ~0.85 (label smoothing floor) within 200 steps.")
        for step in range(1, 201):
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            if step % 20 == 0:
                acc = (model(x).argmax(1) == y).float().mean().item()
                print(f"  step {step:3d}  loss {loss.item():.4f}  acc {acc:.3f}")
        return

    exp_dir = setup_experiment(args)
    print(f"Experiment : {exp_dir}")

    with open(exp_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    train_loader, val_loader = get_dataloaders(args)
    model = build_model(args).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters : {n_params / 1e6:.2f}M")
    print(f"Traversal  : {args.traversal}  |  conv: causal1d  |  device: {device}")
    print(f"Epochs     : {args.epochs}  |  batch: {args.batch_size}  |  lr: {args.lr}")
    print()

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer, scheduler = build_optimizer_scheduler(args, model, len(train_loader))

    log = []
    best_top1 = 0.0
    csv_path = exp_dir / "train_log.csv"
    fields = ["epoch", "train_loss", "train_acc", "val_loss", "val_top1", "val_top5", "lr"]

    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch, args.epochs,
        )
        val_loss, val_top1, val_top5 = evaluate(
            model, val_loader, criterion, device, epoch, args.epochs,
        )

        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        is_best = val_top1 > best_top1

        row = dict(
            epoch=epoch,
            train_loss=round(train_loss, 4),
            train_acc=round(train_acc, 4),
            val_loss=round(val_loss, 4),
            val_top1=round(val_top1, 4),
            val_top5=round(val_top5, 4),
            lr=round(lr, 8),
        )
        log.append(row)

        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)

        print(
            f"[{epoch:3d}/{args.epochs}]  "
            f"loss {train_loss:.4f}/{val_loss:.4f}  "
            f"acc {train_acc:.3f}/{val_top1:.3f}  "
            f"top5 {val_top5:.3f}  "
            f"lr {lr:.2e}  "
            f"[{elapsed:.0f}s]"
            + ("  *best*" if is_best else "")
        )

        if is_best:
            best_top1 = val_top1
            torch.save(model.state_dict(), exp_dir / "model_best.pth")

    torch.save(model.state_dict(), exp_dir / "model_final.pth")
    save_plots(log, exp_dir)

    eval_results = {
        "best_val_top1": round(best_top1, 4),
        "best_val_top5": round(max(r["val_top5"] for r in log), 4),
        "final_val_top1": log[-1]["val_top1"],
        "final_val_top5": log[-1]["val_top5"],
    }
    with open(exp_dir / "eval_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)

    print(f"\nBest val top-1 : {best_top1:.4f}")
    print(f"Results saved  : {exp_dir}")


if __name__ == "__main__":
    main()
