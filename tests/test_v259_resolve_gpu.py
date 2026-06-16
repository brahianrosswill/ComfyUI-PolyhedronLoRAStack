#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_v259_resolve_gpu.py — regression guard for the v259 GPU RESOLVE path.

Run anywhere with torch installed:

    python tests/test_v259_resolve_gpu.py

What it proves
--------------
A) CPU bit-identity: the *shipped* v259 `_resolve_sign_elect` on the CPU-fp32
   path is BYTE-IDENTICAL to v258. A frozen copy of the v258 function is embedded
   below as the reference, so this test needs no v258 checkout.  -> guarantees the
   CPU fallback reproduces the exact old image.

B) Determinism: same seed -> identical result, run to run (CPU always; CUDA too
   when a GPU is present).

C) GPU<->CPU closeness (CUDA only): the GPU/fp16 path differs from CPU/fp32 by
   design; we check the *reconstructed* merged delta (B@A, invariant to SVD
   sign/rotation ambiguity) is very close — cosine > 0.999.

The other merge modes (SEQ / CONCAT / DARE+TRIM) are untouched by v259 (their
code path is not modified), so they remain bit-identical by construction and are
not re-tested here.

Exit code 0 = all checks pass, 1 = a mismatch was found.
"""
import os
import sys
import ast

import pytest

# v300 (audit B-3): skip cleanly on a torch-less machine instead of erroring
# the whole pytest collection (matches test_v272/test_v274 style).
torch = pytest.importorskip("torch")


# ── Load the SHIPPED v259 function straight from the source file ─────────────
def _load_shipped_resolve():
    here = os.path.dirname(os.path.abspath(__file__))
    backend = os.path.normpath(os.path.join(here, "..", "nodes", "uls_merge_math.py"))
    src = open(backend, encoding="utf-8").read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_resolve_sign_elect":
            ns = {}
            exec("import torch\n" + ast.get_source_segment(src, node), ns)
            return ns["_resolve_sign_elect"]
    raise SystemExit(f"_resolve_sign_elect not found in {backend}")


# ── Frozen v258 reference (CPU, fp32, two-pass, niter=4) ─────────────────────
def _resolve_sign_elect_v258(bs, as_, out_dim, in_dim_flat, seed=0):
    n = len(bs)
    ranks = [b.shape[1] for b in bs]
    sum_rank = int(sum(ranks))
    b_trail = list(bs[0].shape[2:])
    a_trail = list(as_[0].shape[1:])

    def _delta(i):
        B2 = bs[i].reshape(out_dim, ranks[i]).float()
        A2 = as_[i].reshape(ranks[i], in_dim_flat).float()
        return B2 @ A2

    S = torch.zeros(out_dim, in_dim_flat, dtype=torch.float32)
    for i in range(n):
        W = _delta(i)
        S += W
        del W
    gamma = torch.sign(S)
    del S

    num = torch.zeros(out_dim, in_dim_flat, dtype=torch.float32)
    den = torch.zeros(out_dim, in_dim_flat, dtype=torch.float32)
    for i in range(n):
        W = _delta(i)
        agree = (torch.sign(W) == gamma) & (gamma != 0)
        num += torch.where(agree, W, torch.zeros_like(W))
        den += agree.to(torch.float32)
        del W, agree
    W_merged = num / den.clamp(min=1.0)
    del num, den, gamma

    if (not torch.isfinite(W_merged).all()) or float(W_merged.abs().max()) == 0.0:
        return None

    min_dim = min(out_dim, in_dim_flat)
    r = max(1, min(sum_rank, min_dim))
    q = min(min_dim, r + 8)
    rng_state = torch.get_rng_state()
    try:
        torch.manual_seed(seed & 0x7FFFFFFF)
        U, Sv, V = torch.svd_lowrank(W_merged, q=q, niter=4)
    finally:
        torch.set_rng_state(rng_state)
    del W_merged

    U = U[:, :r].contiguous()
    Sv = Sv[:r]
    V = V[:, :r].contiguous()
    B_merged = (U * Sv.unsqueeze(0)).reshape([out_dim, r] + b_trail).contiguous()
    A_merged = V.transpose(0, 1).reshape([r] + a_trail).contiguous()
    return B_merged, A_merged


# ── Helpers ──────────────────────────────────────────────────────────────────
def _make_inputs(out_dim, in_dim, ranks, seed=0):
    g = torch.Generator().manual_seed(seed)
    bs = [torch.randn(out_dim, r, generator=g, dtype=torch.float32) for r in ranks]
    as_ = [torch.randn(r, in_dim, generator=g, dtype=torch.float32) for r in ranks]
    return bs, as_


def _recon(B, A):
    # linear-shaped factors: B [out, r], A [r, in] -> full delta [out, in]
    return B.reshape(B.shape[0], B.shape[1]) @ A.reshape(A.shape[0], A.shape[1])


CASES = [
    (320, 256, [8, 16, 8, 32]),
    (1280, 1024, [16, 16, 16, 16, 16]),
    (512, 512, [4, 64]),
    (2048, 1536, [32, 32, 64]),
]
SEED = 0x5A5A5A5A


def main():
    shipped = _load_shipped_resolve()
    failures = []
    has_cuda = torch.cuda.is_available()
    print(f"torch {torch.__version__} | CUDA available: {has_cuda}"
          + (f" ({torch.cuda.get_device_name(0)})" if has_cuda else ""))

    # ── A) CPU bit-identity vs v258 ───────────────────────────────────────────
    print("\n[A] CPU-fp32 v259 == v258  (must be BIT-IDENTICAL)")
    for i, (o, n, ranks) in enumerate(CASES):
        bs, as_ = _make_inputs(o, n, ranks, seed=1000 + i)
        Bo, Ao = _resolve_sign_elect_v258([b.clone() for b in bs], [a.clone() for a in as_], o, n, seed=SEED)
        Bn, An = shipped([b.clone() for b in bs], [a.clone() for a in as_], o, n, seed=SEED, device="cpu", use_fp16=False)
        eq = torch.equal(Bo, Bn) and torch.equal(Ao, An)
        dB = (Bo - Bn).abs().max().item()
        dA = (Ao - An).abs().max().item()
        print(f"    {o}x{n} ranks={ranks}: equal={eq} (max|dB|={dB:.2e} max|dA|={dA:.2e}) "
              f"{'PASS' if eq else 'FAIL'}")
        if not eq:
            failures.append(f"A:{o}x{n}")

    # ── B) determinism ────────────────────────────────────────────────────────
    print("\n[B] determinism (same seed -> identical)")
    bs, as_ = _make_inputs(1280, 1024, [16, 16, 16, 16], seed=7)
    Ba, Aa = shipped([b.clone() for b in bs], [a.clone() for a in as_], 1280, 1024, seed=99, device="cpu")
    Bb, Ab = shipped([b.clone() for b in bs], [a.clone() for a in as_], 1280, 1024, seed=99, device="cpu")
    eq_cpu = torch.equal(Ba, Bb) and torch.equal(Aa, Ab)
    print(f"    CPU: equal={eq_cpu} {'PASS' if eq_cpu else 'FAIL'}")
    if not eq_cpu:
        failures.append("B:cpu")
    if has_cuda:
        Bg1, Ag1 = shipped([b.clone() for b in bs], [a.clone() for a in as_], 1280, 1024, seed=99, device="cuda", use_fp16=True)
        Bg2, Ag2 = shipped([b.clone() for b in bs], [a.clone() for a in as_], 1280, 1024, seed=99, device="cuda", use_fp16=True)
        eq_gpu = torch.equal(Bg1, Bg2) and torch.equal(Ag1, Ag2)
        print(f"    GPU/fp16: equal={eq_gpu} {'PASS' if eq_gpu else 'FAIL'}")
        if not eq_gpu:
            failures.append("B:gpu")
    else:
        print("    GPU/fp16: SKIPPED (no CUDA)")

    # ── C) GPU<->CPU closeness ────────────────────────────────────────────────
    print("\n[C] GPU/fp16 vs CPU/fp32 closeness (reconstructed delta, cosine > 0.999)")
    if has_cuda:
        all_close = True
        for i, (o, n, ranks) in enumerate(CASES):
            bs, as_ = _make_inputs(o, n, ranks, seed=2000 + i)
            Bc, Ac = shipped([b.clone() for b in bs], [a.clone() for a in as_], o, n, seed=SEED, device="cpu", use_fp16=False)
            Bg, Ag = shipped([b.clone() for b in bs], [a.clone() for a in as_], o, n, seed=SEED, device="cuda", use_fp16=True)
            dc = _recon(Bc, Ac).flatten()
            dg = _recon(Bg.cpu(), Ag.cpu()).flatten()
            cos = torch.nn.functional.cosine_similarity(dc, dg, dim=0).item()
            rel = (dc - dg).norm().item() / (dc.norm().item() + 1e-12)
            ok = cos > 0.999
            all_close &= ok
            print(f"    {o}x{n}: cosine={cos:.6f} rel_frob_err={rel:.4f} {'PASS' if ok else 'FAIL'}")
            if not ok:
                failures.append(f"C:{o}x{n}")
        if not all_close:
            failures.append("C")
    else:
        print("    SKIPPED (no CUDA) — run on the ComfyUI machine to verify the GPU path")

    print("\n" + "=" * 60)
    if failures:
        print("RESULT: *** FAIL *** ->", ", ".join(failures))
        sys.exit(1)
    print("RESULT: ALL CHECKS PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
