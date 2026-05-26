"""
Visualize patch traversal strategies on the actual 8x8 experiment grid.

Usage:
  uv run python scripts/visualize_traversals.py

PDFs saved to outputs/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

import torch
from vision_lstm.vision_lstm_traversal import (
    _get_zigzag_perm,
    _get_spiral_outward_perm,
    _get_hilbert_perm,
    _get_random_perm,
)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['STIX Two Text', 'STIXGeneral', 'Times New Roman', 'serif']

H, W = 8, 8  # actual experiment patch grid (64×64 image, patch_size=8)

OUT_DIR = Path(__file__).resolve().parents[1] / 'outputs'
OUT_DIR.mkdir(exist_ok=True)


def perm_to_points(perm: torch.Tensor) -> list[tuple[float, float]]:
    pts = []
    for idx in perm.tolist():
        r, c = idx // W, idx % W
        pts.append((c + 0.5, (H - 1 - r) + 0.5))
    return pts


def save_curve_pdf(pts, title, filename):
    fig, ax = plt.subplots(figsize=(5, 5))

    pts = np.array(pts)
    x, y = pts[:, 0], pts[:, 1]

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    norm = Normalize(0, len(pts) - 1)
    lc = LineCollection(list(segments), cmap='viridis', norm=norm)
    lc.set_array(np.arange(len(pts) - 1))
    lc.set_linewidth(3.0)
    lc.set_joinstyle('round')
    lc.set_capstyle('round')
    ax.add_collection(lc)

    ax.plot(x[0], y[0], 'o', color='green', markersize=13, zorder=5)
    ax.plot(x[-1], y[-1], 'X', color='red', markersize=13, zorder=5)

    ax.set_facecolor('#dcdcdc')
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_xticks(np.arange(W + 1))
    ax.set_yticks(np.arange(H + 1))
    ax.grid(True, color='white', linestyle='-', linewidth=2.0)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(title, fontsize=16, pad=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, format='pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f'Saved {filename}')


rowwise_perm  = torch.arange(H * W)
colwise_perm  = torch.tensor([r * W + c for c in range(W) for r in range(H)], dtype=torch.long)

curves = [
    (rowwise_perm,                    'Row-wise',           'traversal_rowwise.pdf'),
    (colwise_perm,                    'Column-wise',        'traversal_colwise.pdf'),
    (_get_zigzag_perm(H, W),          'Diagonal Zig-Zag',   'traversal_zigzag.pdf'),
    (_get_spiral_outward_perm(H, W),  'Spiral (Outward)',   'traversal_spiral.pdf'),
    (_get_hilbert_perm(H, W),         'Hilbert Curve',      'traversal_hilbert.pdf'),
    (_get_random_perm(H, W),          'Random (seed=42)',   'traversal_random.pdf'),  # type: ignore[call-arg]
]

for perm, title, filename in curves:
    save_curve_pdf(perm_to_points(perm), title, filename)

print(f'\nAll traversals saved to {OUT_DIR}')
