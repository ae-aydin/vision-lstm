"""
Correctness checks for patch traversal permutations and forward passes.
Run with: python tests/test_traversal.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from vision_lstm.vision_lstm2 import VisionLSTM2
from vision_lstm.vision_lstm_traversal import (
    SequenceTraversal,
    _get_zigzag_perm,
    _get_spiral_outward_perm,
    _get_hilbert_perm,
    _get_random_perm,
)


# ---------------------------------------------------------------------------
# 1. Permutation correctness
# ---------------------------------------------------------------------------

def check_perm(perm: torch.Tensor, H: int, W: int, name: str):
    S = H * W
    assert perm.shape == (S,), f"{name}: expected length {S}, got {perm.shape}"
    assert perm.min().item() == 0, f"{name}: min index is not 0"
    assert perm.max().item() == S - 1, f"{name}: max index is not {S - 1}"
    assert perm.unique().numel() == S, f"{name}: duplicate indices found"
    print(f"  {name}({H}x{W}): perm OK  (first={perm[0].item()}, last={perm[-1].item()})")


print("--- Permutation checks ---")
for H, W in [(8, 8), (4, 4), (7, 5)]:
    check_perm(_get_zigzag_perm(H, W),        H, W, "zigzag")
    check_perm(_get_spiral_outward_perm(H, W), H, W, "spiral")
    check_perm(_get_random_perm(H, W),         H, W, "random")

for H, W in [(8, 8), (4, 4)]:
    check_perm(_get_hilbert_perm(H, W), H, W, "hilbert")

assert _get_zigzag_perm(8, 8)[0].item() == 0, "zigzag should start at top-left (index 0)"
center_candidates = {27, 28, 35, 36}
assert _get_spiral_outward_perm(8, 8)[0].item() in center_candidates, (
    f"spiral should start near center, got {_get_spiral_outward_perm(8, 8)[0].item()}"
)
# hilbert on 8x8: first patch must be one of the four corners
corner_candidates = {0, 7, 56, 63}
assert _get_hilbert_perm(8, 8)[0].item() in corner_candidates, (
    f"hilbert should start at a corner, got {_get_hilbert_perm(8, 8)[0].item()}"
)
# random perm with same seed must be reproducible
assert (_get_random_perm(8, 8) == _get_random_perm(8, 8)).all(), "random perm not reproducible"
print("  zigzag starts at top-left: OK")
print("  spiral starts near center: OK")
print("  hilbert starts at a corner: OK")
print("  random perm is reproducible: OK")


# ---------------------------------------------------------------------------
# 2. Inverse permutation correctness (argsort must restore original order)
# ---------------------------------------------------------------------------

print("\n--- Inverse permutation checks ---")
for name, perm in [
    ("zigzag",  _get_zigzag_perm(8, 8)),
    ("spiral",  _get_spiral_outward_perm(8, 8)),
    ("hilbert", _get_hilbert_perm(8, 8)),
    ("random",  _get_random_perm(8, 8)),
]:
    inv = torch.argsort(perm)
    x = torch.arange(64)
    restored = x[perm][inv]
    assert (restored == x).all(), f"{name}: argsort does not restore original order"
    print(f"  {name}: inverse perm OK")


# ---------------------------------------------------------------------------
# 3. Forward pass — conv_kind="causal1d" (fair comparison config)
# ---------------------------------------------------------------------------

print("\n--- Forward pass (conv_kind='causal1d') ---")
x = torch.randn(2, 3, 64, 64)
cfg = dict(dim=192, input_shape=(3, 64, 64), patch_size=8, depth=12,
           output_shape=(200,), conv_kind="causal1d")

for name, traversal in [
    ("rowwise",  SequenceTraversal.ROWWISE_FROM_TOP_LEFT),
    ("zigzag",   SequenceTraversal.ZIGZAG_FROM_TOP_LEFT),
    ("spiral",   SequenceTraversal.SPIRAL_OUTWARD),
    ("hilbert",  SequenceTraversal.HILBERT),
    ("random",   SequenceTraversal.RANDOM_FIXED),
]:
    model = VisionLSTM2(traversal=traversal, **cfg)
    out = model(x)
    assert out.shape == (2, 200), f"{name}: expected (2, 200), got {tuple(out.shape)}"
    assert out.isfinite().all(), f"{name}: output contains NaN or Inf"
    print(f"  {name}: {tuple(out.shape)}  OK")

print("\nAll checks passed.")
