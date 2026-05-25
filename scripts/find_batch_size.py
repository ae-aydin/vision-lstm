"""
Estimate the maximum safe batch size for training on the current GPU.

Usage:
  python scripts/find_batch_size.py
  python scripts/find_batch_size.py --traversal zigzag --lr 5e-4
"""

import argparse
import contextlib
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from vision_lstm.vision_lstm2 import VisionLSTM2
from vision_lstm.vision_lstm_traversal import SequenceTraversal

TRAVERSAL_MAP = {
    "rowwise": SequenceTraversal.ROWWISE_FROM_TOP_LEFT,
    "zigzag":  SequenceTraversal.ZIGZAG_FROM_TOP_LEFT,
    "spiral":  SequenceTraversal.SPIRAL_OUTWARD,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--traversal", choices=list(TRAVERSAL_MAP), default="rowwise")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Base LR at batch 1024 (paper's linear scaling formula).")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    if device.type != "cuda":
        print("CUDA device required.")
        return

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.amp
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
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    def measure_peak_mb(batch_size):
        model.zero_grad()
        torch.cuda.reset_peak_memory_stats(device)
        x = torch.randn(batch_size, 3, 64, 64, device=device)
        y = torch.zeros(batch_size, dtype=torch.long, device=device)
        with amp_ctx:
            loss = criterion(model(x), y)
        loss.backward()
        return torch.cuda.max_memory_allocated(device) / 1024 ** 2

    mem1 = measure_peak_mb(1)
    mem2 = measure_peak_mb(2)
    per_sample_mb = mem2 - mem1
    fixed_mb      = mem1 - per_sample_mb
    free_mb = torch.cuda.mem_get_info(device)[0] / 1024 ** 2
    max_batch = int((free_mb * 0.90 - fixed_mb) / per_sample_mb)
    max_batch = 2 ** int(math.log2(max_batch))
    scaled_lr = args.lr * max_batch / 1024
    amp_label = "bf16" if args.amp else "fp32"

    print(f"Fixed overhead    : {fixed_mb:.0f} MB  (model + grads; excludes AdamW state ~47MB covered by 0.90 margin; amp={amp_label})")
    print(f"Per sample        : {per_sample_mb:.1f} MB")
    print(f"Free GPU memory   : {free_mb:.0f} MB")
    print(f"Suggested batch   : {max_batch}")
    print(f"Scaled LR         : {scaled_lr:.2e}  (lr = {args.lr:.0e} × batch / 1024)")


if __name__ == "__main__":
    main()
