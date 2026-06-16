# -*- coding: utf-8 -*-
"""
test_v264_merge_analyzer.py
═══════════════════════════
Guards the v264 Merge Analyzer's measurement math WITHOUT needing ComfyUI
(AST-loads the pure functions, like the other tests):

  [1] _true_resolved_delta (analyzer) == manual TIES (γ=sign(ΣΔ), disjoint mean)
      → the analyzer measures the SAME resolved delta the merge produces.
  [2] Energy invariant: the actual _resolve_sign_elect re-pack error matches the
      SVD energy the analyzer reports (1 - rel_err² ≈ energy@sum_rank), i.e. the
      randomized SVD is near-optimal and the analyzer's % is trustworthy.
  [3] Determinism of _resolve_sign_elect (same seed → identical).

The v264 change is purely additive (a new passive node); the merge path is
untouched, so test_v259/test_v261 remain the bit-identity guard. Run those too.
"""
import os
import ast
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STACK = os.path.join(HERE, "..", "nodes", "uls_stack_node.py")
MERGEMATH = os.path.join(HERE, "..", "nodes", "uls_merge_math.py")  # v348: _resolve_sign_elect moved here
ANALYZER = os.path.join(HERE, "..", "nodes", "uls_resolve_inspector.py")


def _load_fn(path, name, preamble="import torch\n"):
    src = open(path, "r", encoding="utf-8").read()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            ns = {}
            exec(preamble + ast.get_source_segment(src, node), ns)
            return ns[name]
    raise SystemExit(f"{name} not found in {path}")


def main():
    try:
        import torch
    except ImportError:
        print("SKIP test_v264 — torch not available"); return 0

    resolve = _load_fn(MERGEMATH, "_resolve_sign_elect")
    true_delta = _load_fn(ANALYZER, "_true_resolved_delta")

    torch.manual_seed(0)
    ok = True

    # ── [1] _true_resolved_delta == manual TIES ──
    out, inn, ranks = 96, 80, [8, 8, 8]
    base = torch.randn(out, max(ranks))
    bs, as_ = [], []
    for i, r in enumerate(ranks):
        sign = 1.0 if i % 2 == 0 else -1.0
        bs.append((base[:, :r] + 0.3 * torch.randn(out, r)).float() * sign)
        as_.append(torch.randn(r, inn).float())

    def manual_ties(bs, as_, out, inn):
        deltas = [bs[i] @ as_[i] for i in range(len(bs))]
        g = torch.sign(sum(deltas))
        num = torch.zeros(out, inn); den = torch.zeros(out, inn)
        for W in deltas:
            ag = (torch.sign(W) == g) & (g != 0)
            num += torch.where(ag, W, torch.zeros_like(W)); den += ag.float()
        return num / den.clamp(min=1.0)

    Wt = true_delta(bs, as_, out, inn, torch)
    Wm = manual_ties(bs, as_, out, inn)
    d1 = float((Wt - Wm).abs().max())
    p1 = d1 < 1e-5
    ok &= p1
    print(f"[1] true_resolved_delta == manual TIES : max|Δ|={d1:.2e}  {'PASS' if p1 else 'FAIL'}")

    # ── [2] Energie-Invariante: 1 - rel_err² ≈ Energie@sum_rank ──
    sum_rank = sum(ranks)
    sv = torch.linalg.svdvals(Wt.float())
    energy_at = float((sv[:sum_rank] ** 2).sum()) / float((sv ** 2).sum())
    res = resolve(bs, as_, out, inn, seed=0, device="cpu", use_fp16=False)
    Wa = res[0].reshape(out, sum_rank).float() @ res[1].reshape(sum_rank, inn).float()
    rel = float((Wt - Wa).norm() / Wt.norm())
    implied_energy = 1.0 - rel * rel
    d2 = abs(implied_energy - energy_at)
    p2 = d2 < 0.03   # randomisierte SVD ist nahezu optimal → enge Übereinstimmung
    ok &= p2
    print(f"[2] Energie-Invariante: Energie@Σrk={energy_at*100:.1f}%  vs  "
          f"1-relErr²={implied_energy*100:.1f}%  (Δ={d2*100:.1f}pp)  {'PASS' if p2 else 'FAIL'}")

    # ── [3] Determinismus ──
    r1 = resolve(bs, as_, out, inn, seed=123, device="cpu", use_fp16=False)
    r2 = resolve(bs, as_, out, inn, seed=123, device="cpu", use_fp16=False)
    p3 = torch.equal(r1[0], r2[0]) and torch.equal(r1[1], r2[1])
    ok &= p3
    print(f"[3] Determinismus (gleicher Seed → identisch): {'PASS' if p3 else 'FAIL'}")

    print("=" * 56)
    print("RESULT:", "ALL CHECKS PASS" if ok else "FAILURE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
