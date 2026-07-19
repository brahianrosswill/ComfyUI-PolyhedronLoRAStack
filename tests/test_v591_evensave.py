"""v591 guard: THE ENCODER GETS EVEN EDGES, OR IT NEVER OPENS.

Frank's 2026-07-14 run, 18 minutes and 36 seconds in, on a finished upscale:

    av.error.ExternalError: [Errno 542398533] Generic error in an external
    library: 'avcodec_open2(libx264)'
    ph_save.py:513 -> vid.save_to(path, codec=VideoCodec.H264)

The canvas was 1075x1075. That is 768 * final_upscale_by 1.40 = 1075.2, floored.
Odd. H.264 subsamples chroma 2x2 - an odd edge has no chroma sample to carry it
and libx264 refuses to OPEN, on frame zero, after the whole run is already
spent. Reproduced against libx264 in a sandbox before a line was written:

    1075x1075 -> ExternalError (identical errno)
    1074x1074 -> writes
    1074x1075 -> ExternalError

ph_save had no dimension check anywhere. Three write paths, none of them asked.

The law: before the backend fork - where the frames and the mask are still
together and no path can slip past - a chroma-subsampled target gets its
canvas cropped to even, LOUDLY, and the user's dial is left alone. The crop is
one pixel of 1075 (0.09%, invisible); a pad would be a duplicated row that
moves with the image (a seam, visible). Images, GIF/WebP and yuv444 are
untouched: they have no chroma plane to subsample.

Pinned as RETURN VALUES of the pure functions, not as the mechanism that
computes them (lesson 4): the arithmetic may be rewritten, 1075 must still
land on 1074.
"""
import ast
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SAVE = ROOT / "nodes" / "ph_save.py"


def _fail(msg):
    print(f"[test_v591_evensave] FAIL: {msg}")
    sys.exit(1)


def _pure(src, name):
    """Lift a module-level pure function out of the source and exec it alone -
    ph_save imports torch/av, which no guard is allowed to need."""
    m = re.search(r"^def " + name + r"\(.*?(?=\n(?:def |class )|\Z)",
                  src, re.S | re.M)
    if not m:
        _fail(f"{name}() is gone - the encoder floor has no pure form left")
    ns = {}
    exec(m.group(0), ns)  # noqa: S102 - our own source, measured not believed
    return ns[name]


def main():
    src = SAVE.read_text(encoding="utf-8")
    even = _pure(src, "_even_dims")
    needs = _pure(src, "_needs_even")

    # ---- 1: the crash case, and the shape of the law -------------------------
    matrix = [
        ((1075, 1075), (1074, 1074)),   # Frank's canvas. The whole reason.
        ((1074, 1074), (1074, 1074)),   # idempotent: an even canvas is untouched
        ((1074, 1075), (1074, 1074)),   # ONE odd edge is enough to break libx264
        ((1075, 1074), (1074, 1074)),   # ...either edge
        ((768, 768), (768, 768)),       # the common case pays nothing
        ((1920, 1080), (1920, 1080)),   # nor does 16:9
        ((1081, 1081), (1080, 1080)),
    ]
    for args, want in matrix:
        got = even(*args)
        if tuple(got) != want:
            _fail(f"_even_dims{args} = {tuple(got)}, expected {want} - "
                  f"an odd edge that survives here is a dead run at the encoder")
    # Never grows: a crop cannot invent pixels.
    for w, h in [(1075, 1075), (2, 3), (999, 1000)]:
        gw, gh = even(w, h)
        if gw > w or gh > h:
            _fail(f"_even_dims({w},{h}) GREW the canvas to ({gw},{gh}) - "
                  f"the fix is a crop, not a pad")

    # ---- 2: who has to pay it -----------------------------------------------
    for backend, fmt, want in [
        ("native", None, True),        # comfy_api writes H.264/yuv420p. Always.
        ("native", "yuv444p", True),   # ...regardless of what is passed alongside
        ("pillow", "yuv420p", False),  # GIF/WebP have no chroma plane
        ("pyav", "yuv420p", True),
        ("pyav", "yuv422p10le", True),
        ("pyav", "nv12", True),
        ("pyav", "yuv444p", False),    # a chroma sample per pixel: odd is fine
        ("pyav", "gbrp", False),
        ("pyav", "rgb24", False),
    ]:
        got = bool(needs(backend, fmt))
        if got is not want:
            _fail(f"_needs_even({backend!r}, {fmt!r}) = {got}, expected {want} - "
                  f"either a run dies at the encoder or a canvas is cropped "
                  f"for no reason")

    # ---- 3: the seam is BEFORE the fork -------------------------------------
    # A check that sits inside one writer leaves the other two open. This pins
    # that the crop happens in _save_video, ahead of every _write_* call.
    tree = ast.parse(src)
    fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_save_video":
            fn = node
            break
    if fn is None:
        _fail("_save_video is gone")
    first_check, first_write = None, None
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in ("_needs_even", "_even_dims"):
            if first_check is None or node.lineno < first_check:
                first_check = node.lineno
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and str(node.func.attr).startswith("_write_"):
            if first_write is None or node.lineno < first_write:
                first_write = node.lineno
    if first_check is None:
        _fail("_save_video never asks about even edges - the writers are back "
              "on their own, which is exactly how v590 shipped")
    if first_write is None:
        _fail("_save_video calls no writer any more")
    if first_check > first_write:
        _fail(f"the even-edge check (line {first_check}) runs AFTER the first "
              f"writer (line {first_write}) - a crop behind the encoder is a "
              f"crash with extra steps")

    # ---- 4: it is SAID ------------------------------------------------------
    if "odd edge" not in src or "not negotiable" not in src:
        _fail("the crop must SAY it happened and why - a silent one-pixel "
              "change to the user's canvas is exactly what v584 taught us "
              "never to ship")

    print("PASS: v591 -- even-edge floor pinned by return value (1075->1074), "
          "crop never pads, applied before the backend fork, only where chroma "
          "is subsampled, and spoken")


if __name__ == "__main__":
    main()
