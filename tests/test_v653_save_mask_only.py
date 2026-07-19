#!/usr/bin/env python3
"""test_v653_save_mask_only -- the mask-only Save lane, DRIVEN.

What runs (no ComfyUI -- by construction):
  * resolve_media_kind REGRESSION MATRIX: every pre-v653 combination keeps
    its exact verdict (video wins, image+audio -> video, multi-frame image
    -> video, image -> image, audio -> audio, forced kinds + their errors),
    mask NEVER outranks real media, and ONLY-mask -> "mask" instead of the
    old raise; truly nothing still raises.
  * the mask writer against a REAL tmpdir: PNG lands in grayscale mode L,
    pixel values match the analytic ramp byte-exactly, a 2D mask is
    accepted as one frame, a batch writes one file per entry, jpg works.
    (Needs PIL + numpy; PIL absent -> SKIP-AS-PASS for the writer half,
    the resolve matrix runs regardless.)

MUTATIONS (wound injected into a COPY, catch proven): M1 mask outranks
audio in resolve, M2 the writer saves RGB instead of grayscale L,
M3 the nothing-wired raise falls silent. Each must fail its probe.
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
    print("[test_v653_save_mask_only] FAIL: " + msg)
    sys.exit(1)


def _need(cond, msg):
    if not cond:
        _fail(msg)


def _import_from(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _import_save(path, name):
    """ph_save.py does `from . import ph_save_util` -- give it a synthetic
    parent package whose search path is the real nodes/ dir."""
    import types
    pkg = "phg_pkg_" + name
    parent = types.ModuleType(pkg)
    parent.__path__ = [os.path.join(ROOT, "nodes")]
    sys.modules[pkg] = parent
    try:
        spec = importlib.util.spec_from_file_location(pkg + "." + name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[pkg + "." + name] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        pass  # parent stays registered for the module's lifetime


U = _import_from(UTIL_PY, "ph_save_util_guard_v653")

try:
    import numpy as np
except Exception:
    _fail("numpy missing")


# --------------------------------------------------------------------------
# resolve_media_kind regression matrix
# --------------------------------------------------------------------------

def run_resolve(U, strict=True):
    R = U.resolve_media_kind
    ok = True
    # pre-v653 verdicts, byte-for-byte -- with and without a mask riding along
    for hm in (False, True):
        ok &= R("auto", True, True, True, 1, has_mask=hm) == "video"
        ok &= R("auto", True, False, True, 1, has_mask=hm) == "video"
        ok &= R("auto", True, False, False, 8, has_mask=hm) == "video"
        ok &= R("auto", True, False, False, 1, has_mask=hm) == "image"
        ok &= R("auto", False, False, True, 1, has_mask=hm) == "audio"
        ok &= R("image", True, False, False, 1, has_mask=hm) == "image"
        ok &= R("video", False, True, False, 1, has_mask=hm) == "video"
    # v653: ONLY a mask -> "mask"
    ok &= R("auto", False, False, False, 1, has_mask=True) == "mask"
    if not strict:
        # mutated build reporting: also demand the raise on truly nothing
        try:
            R("auto", False, False, False, 1, has_mask=False)
            return False
        except ValueError:
            pass
        return ok
    _need(ok, "resolve matrix drifted")
    for kind, args in (("auto", (False, False, False)),
                       ("image", (False, True, True)),
                       ("video", (False, False, True))):
        try:
            R(kind, *args, 1, has_mask=False)
            _fail("resolve must raise for kind=%s %r" % (kind, args))
        except ValueError:
            pass
    # default keeps every old call: has_mask omitted == False
    try:
        R("auto", False, False, False)
        _fail("nothing wired must still raise when has_mask is omitted")
    except ValueError:
        pass
    return True


# --------------------------------------------------------------------------
# the mask writer against a real tmpdir
# --------------------------------------------------------------------------

def run_writer(save_path, strict=True):
    try:
        from PIL import Image
    except Exception:
        if strict:
            print("[test_v653_save_mask_only] note: PIL absent -- writer half "
                  "skipped as pass; resolve matrix + mutations still ran.")
            return True
        return True  # cannot probe the wound without PIL; matrix carries M2? no
    S = _import_save(save_path, "ph_save_guard_v653")
    node = S.ULSSave() if hasattr(S, "ULSSave") else None
    if node is None:  # class name lookup, robust
        for n in dir(S):
            o = getattr(S, n)
            if isinstance(o, type) and hasattr(o, "_save_mask"):
                node = o()
                break
    _need(node is not None, "save node class with _save_mask not found")
    with tempfile.TemporaryDirectory() as d:
        os.environ["ULS_SAVE_HOME"] = d  # honored? _resolve_out may use folder_paths
        try:
            import types as _t
            # steer _resolve_out into the tmpdir regardless of environment
            S._resolve_out = lambda prefix, w, h, so: (d, "guardmask", 0, "guard", "output")
            ramp = (np.arange(12, dtype=np.float32).reshape(3, 4) / 11.0)
            batch = np.stack([ramp, 1.0 - ramp])
            out = node._save_mask(batch, "png", 100, "x", True, "disable", None, None)
            files = sorted(f for f in os.listdir(d) if f.endswith(".png"))
            n_ok = len(files) == 2
            with Image.open(os.path.join(d, files[0])) as im:
                mode_ok = im.mode == "L"
                px = np.asarray(im)
            want = (ramp.clip(0, 1) * 255.0).round().astype("uint8")
            px_ok = px.shape == (3, 4) and (px == want).all()
            two_d = node._save_mask(ramp, "jpg", 90, "x", True, "disable", None, None)
            jpgs = [f for f in os.listdir(d) if f.endswith(".jpg")]
            jpg_ok = len(jpgs) == 1 and isinstance(two_d, dict)
            if not strict:
                return n_ok and mode_ok and px_ok and jpg_ok
            _need(n_ok, "batch of 2 must write 2 files")
            _need(mode_ok, "mask still must be grayscale mode L")
            _need(px_ok, "mask pixels must match the analytic ramp byte-exactly")
            _need(jpg_ok, "a 2D mask must save as one jpg frame")
        finally:
            os.environ.pop("ULS_SAVE_HOME", None)
    return True


# --------------------------------------------------------------------------
# mutations: wound a COPY, prove the catch
# --------------------------------------------------------------------------

def _wounded(src_path, old, new, name):
    s = open(src_path, encoding="utf-8").read()
    _need(s.count(old) == 1, "mutation target not unique: " + name)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(s.replace(old, new))
        return f.name


def run_mutations():
    m1 = _wounded(
        UTIL_PY,
        "    if has_image:\n        return \"image\"\n    if has_audio:\n        return \"audio\"\n",
        "    if has_mask:\n        return \"mask\"\n"
        "    if has_image:\n        return \"image\"\n    if has_audio:\n        return \"audio\"\n",
        "M1")
    m3 = _wounded(
        UTIL_PY,
        "    if has_mask:\n        return \"mask\"\n",
        "    return \"mask\"\n", "M3")
    try:
        from PIL import Image  # noqa: F401
        m2 = _wounded(
            SAVE_PY,
            '            pil = Image.fromarray(arr, "L")',
            '            pil = Image.fromarray('
            'np.stack([arr, arr, arr], axis=-1), "RGB")', "M2")
    except Exception:
        m2 = None
    caught = 0
    need = 2 + (1 if m2 else 0)
    for path, probe in ((m1, "rank"), (m2, "mode"), (m3, "raise")):
        if path is None:
            continue
        try:
            if probe == "mode":
                W = _import_save(path, "ph_save_wound_mode")
                # numpy must be visible to the wounded copy's writer
                W.np = np
                ok = run_writer(path, strict=False)
            else:
                Wu = _import_from(path, "ph_save_util_wound_" + probe)
                ok = run_resolve(Wu, strict=False)
            if not ok:
                caught += 1
            else:
                _fail("mutation " + probe + " NOT caught")
        finally:
            os.remove(path)
    _need(caught == need, "mutation coverage incomplete (%d/%d)" % (caught, need))


def main():
    run_resolve(U)
    run_writer(SAVE_PY)
    run_mutations()
    print("[test_v653_save_mask_only] PASS: resolve matrix regression-exact "
          "(mask never outranks media, only-mask saves, nothing still "
          "raises), grayscale L writer byte-exact against the ramp, "
          "mutations caught")


if __name__ == "__main__":
    main()
