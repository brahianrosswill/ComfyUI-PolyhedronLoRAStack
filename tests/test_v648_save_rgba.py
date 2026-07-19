#!/usr/bin/env python3
"""test_v648_save_rgba -- the v648 still-alpha decision + composition, DRIVEN.

DECLARED CHANGE under guard: a 4-channel IMAGE now carries its own alpha
into png/webp stills; a wired MASK keeps priority (1-mask LoadImage
convention, unchanged); jpg always flattens; plain RGB is byte-identical
to the old path. All four cases run through the REAL pure production
functions (still_uses_alpha, compose_still) against analytic truth, and
the node source is pinned to actually CALL them at both still sites.

MUTATIONS: M1 the 4-channel branch is dropped (decision ignores channels),
M2 mask priority is dropped (image alpha wins over a wired mask), M3 the
RGBA composition slices :3 and refills opaque. Each must be caught.
"""
import importlib.util
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
UTIL_PY = os.path.join(ROOT, "nodes", "ph_save_util.py")
SAVE_PY = os.path.join(ROOT, "nodes", "ph_save.py")


def _fail(msg):
    print("[test_v648_save_rgba] FAIL: " + msg)
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


U = _import_from(UTIL_PY, "ph_save_util_guard")


def _frames():
    rgb = np.random.RandomState(11).rand(6, 5, 3).astype(np.float32)
    a = np.linspace(0.0, 1.0, 30, dtype=np.float32).reshape(6, 5)
    rgba = np.concatenate([rgb, a[..., None]], axis=-1)
    mask = np.zeros((6, 5), np.float32)
    mask[:3] = 1.0  # top half transparent (MASK convention)
    return rgb, rgba, a, mask


def run_decision(U):
    _need(U.still_uses_alpha(False, 3, "png") is False, "3ch/no-mask must be RGB")
    _need(U.still_uses_alpha(False, 4, "png") is True, "4ch png must be RGBA")
    _need(U.still_uses_alpha(False, 4, "webp") is True, "4ch webp must be RGBA")
    _need(U.still_uses_alpha(False, 4, "jpg") is False, "jpg must flatten")
    _need(U.still_uses_alpha(True, 3, "png") is True, "wired mask stays RGBA")


def run_compose(U):
    rgb, rgba, a, mask = _frames()
    want_rgb = (rgb * 255.0 + 0.5).astype(np.uint8)

    arr, mode = U.compose_still(rgb, None, "png")
    _need(mode == "RGB" and (arr == want_rgb).all(),
          "plain RGB must stay byte-identical to the old path")

    arr, mode = U.compose_still(rgba, None, "png")
    _need(mode == "RGBA" and arr.shape[-1] == 4, "4ch png not RGBA")
    _need((arr[..., :3] == want_rgb).all(), "RGB part mutated in 4ch compose")
    _need((arr[..., 3] == (a * 255.0 + 0.5).astype(np.uint8)).all(),
          "alpha must be the image's OWN 4th channel")

    arr, mode = U.compose_still(rgba, mask, "png")
    want_alpha = ((1.0 - mask) * 255.0 + 0.5).astype(np.uint8)
    _need(mode == "RGBA" and (arr[..., 3] == want_alpha).all(),
          "wired mask must keep priority (alpha = 1-mask), image alpha ignored")

    arr, mode = U.compose_still(rgba, None, "jpg")
    _need(mode == "RGB" and arr.shape[-1] == 3, "jpg must drop the 4th channel")


def run_wiring():
    src = open(SAVE_PY, encoding="utf-8").read()
    _need(src.count("U.compose_still(") == 2,
          "both still sites must run through compose_still")
    _need("U.still_uses_alpha(mask is not None, img.shape[-1], fmt)" in src,
          "_save_image must take the v648 decision")


def _wounded(old, new, name):
    s = open(UTIL_PY, encoding="utf-8").read()
    _need(s.count(old) == 1, "mutation target not unique: " + name)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(s.replace(old, new))
        return f.name


def run_mutations():
    muts = [
        ("M1", "    return bool(has_mask) or int(channels) >= 4",
         "    return bool(has_mask)"),
        ("M2", "    if mask_float_or_none is not None:",
         "    if False:"),
        ("M3", "    return ((frame[..., :4] * 255.0 + 0.5).astype(np.uint8), \"RGBA\")",
         "    solid = np.concatenate([frame[..., :3],"
         " np.ones_like(frame[..., :1])], axis=-1)\n"
         "    return ((solid * 255.0 + 0.5).astype(np.uint8), \"RGBA\")"),
    ]
    for name, old, new in muts:
        path = _wounded(old, new, name)
        try:
            W = _import_from(path, "ph_save_util_wound_" + name)
            rgb, rgba, a, mask = _frames()
            if name == "M1":
                ok = W.still_uses_alpha(False, 4, "png") is True
            elif name == "M2":
                arr, _ = W.compose_still(rgba, mask, "png")
                ok = (arr[..., 3] == ((1.0 - mask) * 255.0 + 0.5).astype(np.uint8)).all()
            else:
                arr, _ = W.compose_still(rgba, None, "png")
                ok = (arr[..., 3] == (a * 255.0 + 0.5).astype(np.uint8)).all()
            if ok:
                _fail("mutation " + name + " NOT caught")
        finally:
            os.remove(path)


def main():
    run_decision(U)
    run_compose(U)
    run_wiring()
    run_mutations()
    print("[test_v648_save_rgba] PASS: decision matrix analytic, all four "
          "compose cases byte-exact (RGB unchanged, own alpha, mask "
          "priority, jpg flatten), both node sites wired, 3/3 mutations "
          "caught")


if __name__ == "__main__":
    main()
