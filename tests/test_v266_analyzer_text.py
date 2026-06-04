# -*- coding: utf-8 -*-
"""
test_v266_analyzer_text.py
══════════════════════════
Guards the v266 Merge-Analyzer changes (both purely diagnostic / textual —
measurement math untouched, numbers identical to v264/v265):

  [1] Wiring (always runs, no ComfyUI/torch needed): the live console progress
      and the trim-aware report wording are present in the source, and the old
      misleading amplitude-scalar recommendation is gone. Specifically:
        • `_analyze_group(..., label="")` accepts the group label
        • a throttled `[PLS]   ANALYZE [...]` progress line with flush=True
        • the call site passes label=group
        • header says "largest-contribution layers" (audit A-7b)
        • per-group + footer trim-blind notes ("TRIMMED delta", "trim-blind")
        • the report no longer recommends an amplitude scalar
          (B-1 was closed by the render A/B: Trim strength was the culprit)
  [2] Numbers unchanged: AST-loads the shipped `_true_resolved_delta` and
      verifies it still equals manual TIES (same guard as test_v264 [1]) —
      proving the text-only edit did not drift into the math. SKIPs w/o torch.
"""
import os
import ast
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYZER = os.path.join(HERE, "..", "nodes", "uls_resolve_inspector.py")


def main():
    ok = True
    src = open(ANALYZER, "r", encoding="utf-8").read()

    checks = [
        ("label param on _analyze_group",
         'def _analyze_group(names, weights, trim_keep, max_layers, dev, torch, label="")' in src),
        ("throttled ANALYZE progress line",       '[PLS]   ANALYZE [{label}]' in src),
        ("progress line is flushed",              src.count("flush=True") >= 1),
        ("call site passes label=group",          "label=group)" in src),
        ("header: largest-contribution layers",   "largest-contribution layers" in src),
        ("per-group trim-blind note",             "all metrics measure the TRIMMED" in src),
        ("footer trim-blind note",                "the analysis is trim-blind" in src),
        ("amplitude-scalar recommendation gone",
         "The lever would be amplitude" not in src and "cheap lever" not in src),
    ]
    for label, cond in checks:
        ok &= bool(cond)
        print(f"[1] {label:<42} {'PASS' if cond else 'FAIL'}")

    # ── [2] math unchanged (needs torch) ──
    try:
        import torch
    except ImportError:
        print("[2] math-unchanged check: SKIP (torch not available)")
    else:
        tree = ast.parse(src)
        fn = next(n for n in tree.body
                  if isinstance(n, ast.FunctionDef) and n.name == "_true_resolved_delta")
        ns = {}
        exec("import torch\n" + ast.get_source_segment(src, fn), ns)
        true_delta = ns["_true_resolved_delta"]
        torch.manual_seed(0)
        out, inn = 64, 48
        bs = [torch.randn(out, 6), -torch.randn(out, 6)]
        as_ = [torch.randn(6, inn), torch.randn(6, inn)]
        deltas = [bs[i] @ as_[i] for i in range(2)]
        g = torch.sign(sum(deltas))
        num = torch.zeros(out, inn); den = torch.zeros(out, inn)
        for W in deltas:
            ag = (torch.sign(W) == g) & (g != 0)
            num += torch.where(ag, W, torch.zeros_like(W)); den += ag.float()
        d = float((true_delta(bs, as_, out, inn, torch) - num / den.clamp(min=1.0)).abs().max())
        p2 = d < 1e-6
        ok &= p2
        print(f"[2] _true_resolved_delta == manual TIES : max|Δ|={d:.2e}  {'PASS' if p2 else 'FAIL'}")

    print("=" * 56)
    print("RESULT:", "ALL CHECKS PASS" if ok else "FAILURE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
