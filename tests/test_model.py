"""
Sanity checks for VisionLSTM2: output shapes, finite outputs, parameter count.
Run with: python tests/test_model.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from vision_lstm.vision_lstm2 import VisionLSTM2

print("--- Output shape checks ---")

x_224 = torch.randn(2, 3, 224, 224)

model = VisionLSTM2(
    dim=192, depth=12, patch_size=16, input_shape=(3, 224, 224), output_shape=(1000,)
).eval()
with torch.no_grad():
    out = model(x_224)
assert out.shape == (2, 1000), f"Expected (2, 1000), got {tuple(out.shape)}"
assert out.isfinite().all(), "Classifier output contains NaN or Inf"
print(f"  classifier 224x224 : {tuple(out.shape)}  OK")

model = VisionLSTM2(
    dim=192,
    depth=12,
    patch_size=16,
    input_shape=(3, 224, 224),
    output_shape=None,
    mode="features",
    pooling="to_image",
).eval()
with torch.no_grad():
    out = model(torch.randn(1, 3, 224, 224))
assert out.shape[0] == 1, f"Batch dim mismatch: {tuple(out.shape)}"
assert out.isfinite().all(), "Feature output contains NaN or Inf"
print(f"  feature mode 224x224: {tuple(out.shape)}  OK")

model = VisionLSTM2(
    dim=192, depth=2, patch_size=16, input_shape=(3, 384, 384), output_shape=(1000,)
).eval()
with torch.no_grad():
    out = model(torch.randn(1, 3, 384, 384))
assert out.shape == (1, 1000), f"Expected (1, 1000), got {tuple(out.shape)}"
assert out.isfinite().all(), "High-res output contains NaN or Inf"
print(f"  classifier 384x384 : {tuple(out.shape)}  OK")


print("\n--- Parameter count (experiment model: dim=192, depth=12, patch=8) ---")

model = VisionLSTM2(
    dim=192, depth=12, patch_size=8, input_shape=(3, 64, 64), output_shape=(200,)
)
n_params = sum(p.numel() for p in model.parameters())
assert 5_000_000 <= n_params <= 10_000_000, f"Unexpected param count: {n_params:,}"
print(f"  params: {n_params:,}  OK")

print("\nAll checks passed.")
