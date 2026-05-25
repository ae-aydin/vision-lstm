from enum import Enum

import torch


class SequenceTraversal(Enum):
    ROWWISE_FROM_TOP_LEFT = "rowwise_from_top_left"
    ROWWISE_FROM_BOT_RIGHT = "rowwise_from_bot_right"
    ZIGZAG_FROM_TOP_LEFT = "zigzag_from_top_left"
    ZIGZAG_FROM_BOT_RIGHT = "zigzag_from_bot_right"
    SPIRAL_OUTWARD = "spiral_outward"
    SPIRAL_INWARD = "spiral_inward"
    HILBERT = "hilbert"
    HILBERT_REVERSED = "hilbert_reversed"
    RANDOM_FIXED = "random_fixed"
    RANDOM_FIXED_REVERSED = "random_fixed_reversed"


def _get_zigzag_perm(H: int, W: int) -> torch.Tensor:
    """Diagonal anti-zigzag traversal from top-left to bottom-right corner."""
    indices = []
    for d in range(H + W - 1):
        if d % 2 == 0:
            r, c = min(d, H - 1), d - min(d, H - 1)
            while r >= 0 and c < W:
                indices.append(r * W + c)
                r -= 1
                c += 1
        else:
            c, r = min(d, W - 1), d - min(d, W - 1)
            while c >= 0 and r < H:
                indices.append(r * W + c)
                r += 1
                c -= 1
    return torch.tensor(indices, dtype=torch.long)


def _get_spiral_outward_perm(H: int, W: int) -> torch.Tensor:
    """Clockwise outward spiral starting from the center patch."""
    indices = []
    top, bottom, left, right = 0, H - 1, 0, W - 1
    while top <= bottom and left <= right:
        for c in range(left, right + 1):
            indices.append(top * W + c)
        top += 1
        for r in range(top, bottom + 1):
            indices.append(r * W + right)
        right -= 1
        if top <= bottom:
            for c in range(right, left - 1, -1):
                indices.append(bottom * W + c)
            bottom -= 1
        if left <= right:
            for r in range(bottom, top - 1, -1):
                indices.append(r * W + left)
            left += 1
    return torch.tensor(indices[::-1], dtype=torch.long)


def _xy2d(n: int, x: int, y: int) -> int:
    """Convert (x, y) grid coordinates to Hilbert curve index for an n×n grid."""
    d = 0
    s = n // 2
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        s //= 2
    return d


def _get_hilbert_perm(H: int, W: int) -> torch.Tensor:
    """Hilbert space-filling curve traversal. Requires square power-of-2 grid."""
    assert H == W and (H & (H - 1)) == 0, \
        f"Hilbert curve requires square power-of-2 grid, got {H}x{W}"
    hilbert_indices = [_xy2d(H, i % W, i // W) for i in range(H * W)]
    order = sorted(range(H * W), key=lambda i: hilbert_indices[i])
    return torch.tensor(order, dtype=torch.long)


def _get_random_perm(H: int, W: int, seed: int = 42) -> torch.Tensor:
    """Fixed random permutation — reproducible across runs via seed."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    return torch.randperm(H * W, generator=rng)


# Maps each "forward" traversal strategy to its (forward, reverse) direction pair.
# The reverse direction is the same traversal read backwards, matching the
# alternating-flip design used in the original row-wise Vision-LSTM.
_TRAVERSAL_PAIRS = {
    SequenceTraversal.ROWWISE_FROM_TOP_LEFT: (
        SequenceTraversal.ROWWISE_FROM_TOP_LEFT,
        SequenceTraversal.ROWWISE_FROM_BOT_RIGHT,
    ),
    SequenceTraversal.ZIGZAG_FROM_TOP_LEFT: (
        SequenceTraversal.ZIGZAG_FROM_TOP_LEFT,
        SequenceTraversal.ZIGZAG_FROM_BOT_RIGHT,
    ),
    SequenceTraversal.SPIRAL_OUTWARD: (
        SequenceTraversal.SPIRAL_OUTWARD,
        SequenceTraversal.SPIRAL_INWARD,
    ),
    SequenceTraversal.HILBERT: (
        SequenceTraversal.HILBERT,
        SequenceTraversal.HILBERT_REVERSED,
    ),
    SequenceTraversal.RANDOM_FIXED: (
        SequenceTraversal.RANDOM_FIXED,
        SequenceTraversal.RANDOM_FIXED_REVERSED,
    ),
}
