#!/usr/bin/env python3
"""test_v654_loader_gray_mask -- the loader's grayscale-mask rule, DRIVEN.

What runs (no Pillow, no torch -- pure numpy by construction):
  * rgb_and_mask_from_still against analytic frames:
      - real alpha -> the LoadImage convention EXACTLY as before
        (opaque -> 0), luminance flag off  (regression);
      - opaque grayscale ramp -> the luminance IS the mask, byte-exact,
        white -> 1, flag on  (the Save mode-L round-trip);
      - opaque COLOR frame -> all-zero mask as before, flag off;
      - grayscale WITH partial alpha -> alpha WINS (convention ordering);
      - all-black opaque frame -> empty mask, flag off (no phantom mask).
  * statics: the loader's still decode funnels through the new split
    (Bug-B wiring -- the ceremony is declared, the new fingerprint is the
    changelog's business), the header doc names the rule.

MUTATIONS (wound injected into a COPY, catch proven): M1 the grayscale
equality check falls open (color frames become luminance masks), M2 the
alpha convention flips (mask = alpha instead of 1 - alpha), M3 the
alpha-wins ordering is lost (grayscale beats real alpha). Each must fail
its probe.
"""
import importlib.util
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
UTIL_PY = os.path.join(ROOT, "nodes", "ph_media_util.py")
LOADER_PY = os.path.join(ROOT, "nodes", "ph_media_loader.py")


def _fail(msg):
    print("[test_v654_loader_gray_mask] FAIL: " + msg)
    sys.exit(1)


def _need(cond, msg):
    if not cond:
        _fail(msg)


try:
    import numpy as np
except Exception:
    _fail("numpy missing")


def _import_from(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


U = _import_from(UTIL_PY, "ph_media_util_guard_v654")


def _frames():
    ramp = (np.arange(24, dtype=np.float32).reshape(4, 6) / 23.0)
    gray = np.stack([ramp, ramp, ramp, np.ones_like(ramp)], axis=-1)
    color = gray.copy()
    color[..., 0] = np.minimum(1.0, ramp + 0.25)      # r != g -> not grayscale
    alpha = gray.copy()
    alpha[..., 3] = 1.0 - ramp                        # real alpha information
    black = np.zeros((4, 6, 4), np.float32)
    black[..., 3] = 1.0
    return ramp, gray, color, alpha, black


def run_pure(fn, strict=True):
    ramp, gray, color, alpha, black = _frames()
    rgb, m, lum = fn(gray)
    gray_ok = lum is True and (m == ramp).all() and (rgb == gray[..., :3]).all()
    rgb, m, lum = fn(color)
    color_ok = lum is False and (m == 0.0).all()
    rgb, m, lum = fn(alpha)
    # NOTE: 1-(1-x) is not bit-exact in float32 -- the convention probe
    # compares against the same expression the code computes, exactly.
    alpha_ok = lum is False and (m == (1.0 - alpha[..., 3])).all() \
        and np.allclose(m, ramp, atol=1e-6)
    rgb, m, lum = fn(black)
    black_ok = lum is False and (m == 0.0).all()
    if not strict:
        return gray_ok and color_ok and alpha_ok and black_ok
    _need(gray_ok, "opaque grayscale must yield its luminance byte-exact, "
                   "white -> 1")
    _need(color_ok, "opaque color must keep its all-zero mask")
    _need(alpha_ok, "real alpha must keep the LoadImage convention exactly")
    _need(black_ok, "an all-black frame must stay an empty mask")
    return True


def run_static():
    src = open(LOADER_PY, encoding="utf-8").read()
    _need("rgb, mask, _ = rgb_and_mask_from_still(rgba)" in src,
          "the loader's still decode must funnel through "
          "rgb_and_mask_from_still (Bug-B wiring)")
    _need("rgb, mask = rgb_and_mask_from_rgba(rgba)" not in src,
          "the old direct split must be gone from the still decode")
    _need("carries its luminance as the" in src,
          "the header doc must name the grayscale rule")


def _wounded(src_path, old, new, name):
    s = open(src_path, encoding="utf-8").read()
    _need(s.count(old) == 1, "mutation target not unique: " + name)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(s.replace(old, new))
        return f.name


def run_mutations():
    m1 = _wounded(
        UTIL_PY,
        "        if (r == rgba_float[..., 1]).all() and "
        "(r == rgba_float[..., 2]).all():\n",
        "        if True:\n", "M1")
    m2 = _wounded(
        UTIL_PY,
        "    mask = 1.0 - rgba_float[..., 3]\n",
        "    mask = rgba_float[..., 3]\n", "M2")
    m3 = _wounded(
        UTIL_PY,
        "    if (a >= 1.0).all():\n",
        "    if True:\n", "M3")
    caught = 0
    for path, probe in ((m1, "equality"), (m2, "convention"), (m3, "ordering")):
        try:
            W = _import_from(path, "ph_media_util_wound_" + probe)
            ok = run_pure(W.rgb_and_mask_from_still, strict=False)
            if not ok:
                caught += 1
            else:
                _fail("mutation " + probe + " NOT caught")
        finally:
            os.remove(path)
    _need(caught == 3, "mutation coverage incomplete")


def main():
    run_pure(U.rgb_and_mask_from_still)
    run_static()
    run_mutations()
    print("[test_v654_loader_gray_mask] PASS: grayscale luminance mask "
          "byte-exact (white -> 1), alpha convention regression-exact and "
          "winning the ordering, color/black stay empty, loader wiring "
          "pinned, 3/3 mutations caught")


if __name__ == "__main__":
    main()
