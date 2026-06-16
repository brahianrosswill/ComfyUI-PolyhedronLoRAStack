#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_v261_trim_amount.py — regression guard for the v261 Trim-strength feature.

The per-group Trim stepper ultimately feeds a `keep_fraction` into
`_trim_channel_indices(B_f, A_f, keep_fraction)`. This test pins down the two
properties the UI relies on:

  1. DETERMINISM      — same factors + fraction → identical kept indices.
  2. STRONGEST-K KEPT — a fraction keeps exactly round(rank * fraction) channels,
                        and they are the highest-magnitude ones (not random).

It also confirms the "keep all / nothing to trim" path (returns None), which is
what the *Auto* default and the Trim-off state rely on to stay bit-identical.

Self-contained and CPU-only: it AST-loads the SHIPPED function from
nodes/uls_stack_node.py (no full module import, no GPU). Run it from anywhere:

    python tests/test_v261_trim_amount.py
"""

import ast
import pathlib
import sys

# ── Load the shipped function in isolation (tests real code, not a copy) ──────
SRC = pathlib.Path(__file__).resolve().parents[1] / "nodes" / "uls_merge_math.py"
_tree = ast.parse(SRC.read_text(encoding="utf-8"))
_fn = next((n for n in _tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "_trim_channel_indices"), None)
if _fn is None:
    print("FAIL: _trim_channel_indices not found in nodes/uls_merge_math.py")
    sys.exit(1)

_ns = {}
exec(compile(ast.Module(body=[_fn], type_ignores=[]), str(SRC), "exec"), _ns)
_trim_channel_indices = _ns["_trim_channel_indices"]

try:
    import torch
except ImportError:
    print("SKIP: torch not installed — run this on the ComfyUI machine.")
    sys.exit(0)


def _make_factors(rank: int, channel_mags):
    """Build (B_f, A_f) so rank-channel r has contribution magnitude
    channel_mags[r]. B columns are unit-norm; A rows carry the magnitude.
      B_f: [out, rank]   A_f: [rank, in]
    """
    assert len(channel_mags) == rank
    B_f = torch.eye(rank)                      # [rank, rank] → ‖B[:, r]‖ = 1
    A_f = torch.zeros(rank, rank)
    for r, m in enumerate(channel_mags):
        A_f[r, r] = float(m)                   # ‖A[r, :]‖ = |m|
    return B_f, A_f


def main() -> int:
    fails = []

    # Magnitudes strictly increasing with index → strongest = highest indices.
    rank = 8
    mags = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    B_f, A_f = _make_factors(rank, mags)

    # 1) keep_fraction = 0.5 → keep_count = 4 → the four strongest = {4,5,6,7}.
    keep = _trim_channel_indices(B_f, A_f, 0.5)
    got = keep.tolist() if keep is not None else None
    if got != [4, 5, 6, 7]:
        fails.append(f"0.5 keep expected [4,5,6,7], got {got}")

    # 2) Determinism — same call twice, identical result.
    keep_again = _trim_channel_indices(B_f, A_f, 0.5)
    if keep is None or keep_again is None or not torch.equal(keep, keep_again):
        fails.append("0.5 keep not deterministic across calls")

    # 3) Stronger trim: 0.25 → keep_count = 2 → the two strongest = {6,7}.
    keep25 = _trim_channel_indices(B_f, A_f, 0.25)
    got25 = keep25.tolist() if keep25 is not None else None
    if got25 != [6, 7]:
        fails.append(f"0.25 keep expected [6,7], got {got25}")

    # 4) Count matches round(rank * fraction) for the clamp range stops.
    for frac, exp_count in [(0.9, 7), (0.8, 6), (0.7, 6), (0.6, 5), (0.5, 4)]:
        k = _trim_channel_indices(B_f, A_f, frac)
        n = (rank if k is None else k.numel())
        # round(8*0.7)=6, round(8*0.9)=7 (banker's rounding-safe: 7.2→7, 5.6→6…)
        if n != exp_count:
            fails.append(f"frac {frac}: expected keep_count {exp_count}, got {n}")

    # 5) keep_fraction = 1.0 → nothing to trim → None (Auto / keep-all path).
    if _trim_channel_indices(B_f, A_f, 1.0) is not None:
        fails.append("1.0 should return None (keep all)")

    # 6) rank <= 1 → None (degenerate, nothing to trim).
    B1, A1 = _make_factors(1, [3.0])
    if _trim_channel_indices(B1, A1, 0.5) is not None:
        fails.append("rank<=1 should return None")

    # 7) Kept channels really are the strongest, even with shuffled magnitudes.
    shuffled = [5.0, 1.0, 8.0, 2.0, 7.0, 3.0, 6.0, 4.0]   # strongest: idx 2,4,6 (8,7,6)
    Bs, As = _make_factors(8, shuffled)
    keep_top3 = _trim_channel_indices(Bs, As, 0.375)        # round(8*0.375)=3
    got3 = keep_top3.tolist() if keep_top3 is not None else None
    if got3 != [2, 4, 6]:
        fails.append(f"shuffled top-3 expected [2,4,6], got {got3}")

    if fails:
        print("FAIL  test_v261_trim_amount")
        for f in fails:
            print("  -", f)
        return 1
    print("PASS  test_v261_trim_amount  (determinism + strongest-K + keep-all)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
