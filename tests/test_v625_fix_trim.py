#!/usr/bin/env python3
"""
test_v625_fix_trim -- a fixed-length trim window keeps its EXACT frame count while it
slides, the listing carries the native fps the count is computed from, and the video
play badge reads amber.

v625 adds Frank's fixed trim: type a frame count (video, e.g. 121) or a length in
seconds (audio, e.g. 3) and the selection becomes a fixed window that slides along the
timeline as a whole -- handles, the green keep zone and the Start/End fields all MOVE it,
none of them can change its length. The frame count counts in NATIVE fps (the fps the
backend's _slice_frames cuts with; force_fps is only the output label), which the
media listing now provides (uls_routes: _media_vid_dims probes CAP_PROP_FPS in the same
header read).

Guards, all must hold, mutation-tested (inject the wound, prove the catch):

  STATIC -- _fixWindow exists; the fixed-window branches gate the video and audio drag
            paths (trimFixFrames / trimFixSec); the keep zone only drags while fixed;
            the listing probe reads CAP_PROP_FPS and ships an "fps" field; the video
            play badge (.ph-vid) is CTE amber (#ff8c00).

  DRIVEN -- run the REAL _fixWindow at Frank's own numbers (121 frames at a native
            ~33.333 fps clip of 5.43 s): slide it to the middle, clamp it at both ends,
            snap it from a crooked start -- the frame count is 121 EVERY time and the
            start lands on a frame boundary. Mutating the end-clamp away lets the window
            hang past the clip -- caught here.
"""
import os
import re
import sys
import json
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
JS = open(os.path.join(ROOT, "web/js/ph_media_loader.js"), encoding="utf-8").read()
# v362 (public build): the Media Loader routes live in their own module,
# nodes/ph_media_routes.py -- uls_routes.py stays the Stack's file. Same
# source text, different path; the checks below are unchanged.
PY = open(os.path.join(ROOT, "nodes/ph_media_routes.py"), encoding="utf-8").read()


def _fail(msg):
    print("[test_v625_fix_trim] FAIL: " + msg)
    sys.exit(1)


def _lift(sig):
    if sig not in JS:
        _fail("could not find `%s`" % sig)
    s = JS[JS.index(sig):]
    return s[:s.index("\n}") + 2]


# ---------------------------------------------------------------------------
# STATIC -- wiring, listing fps, amber badge.
# ---------------------------------------------------------------------------
for sym, why in [
    ("function _fixWindow(start, len, dur, snap)",
     "the pure fixed-window helper is gone -- nothing enforces the length"),
    ("trimFixFrames", "the video fix is no longer stored on ph_media_state"),
    ("trimFixSec", "the audio fix is no longer stored on audioSel"),
    ('if (this._vFixLen()) startDrag("w", e)',
     "the video keep zone no longer gates window-drag on the fix"),
    ('if (this._aFixLen()) startDrag("w", e)',
     "the audio keep zone no longer gates window-drag on the fix"),
]:
    if sym not in JS:
        _fail(why)

m = re.search(r"\.ph-media-tile \.ph-vid \{[^}]*\}", JS)
if not m or "#ff8c00" not in m.group(0):
    _fail("the video play badge (.ph-vid) is no longer CTE amber (#ff8c00)")

if "CAP_PROP_FPS" not in PY:
    _fail("ph_media_routes no longer probes CAP_PROP_FPS -- the listing cannot carry fps")
if '"fps": fps' not in PY:
    _fail("the media listing no longer ships the fps field -- the frontend cannot "
          "count frames in native fps")


# ---------------------------------------------------------------------------
# DRIVEN -- the REAL _fixWindow at Frank's numbers, then a MUTANT.
# ---------------------------------------------------------------------------
BODY = """
const FPS = 33.333333;              // Frank's clip: 1.2 s ~ 40 frames -> ~33.33 native fps
const DUR = 5.43;                   // seconds
const N = 121;                      // the fixed frame count
const LEN = N / FPS, SNAP = 1 / FPS;
const frames = (w) => Math.round((w[1] - w[0]) * FPS);
const onGrid = (w) => Math.abs(Math.round(w[0] / SNAP) - (w[0] / SNAP)) < 1e-6;

const out = {};
{   const w = _fixWindow(1.234567, LEN, DUR, SNAP);      // slide to a crooked middle
    out.midFrames = frames(w); out.midOnGrid = onGrid(w); out.midStart = w[0]; }
{   const w = _fixWindow(-3, LEN, DUR, SNAP);             // clamp at the head
    out.headFrames = frames(w); out.headStart = w[0]; }
{   const w = _fixWindow(99, LEN, DUR, SNAP);             // clamp at the tail
    out.tailFrames = frames(w); out.tailEnd = w[1]; out.dur = DUR; }
{   const w = _fixWindow(1.5, 3.0, 10.0, 0.1);            // audio: 3.0 s window, tenths
    out.audLen = Math.round((w[1] - w[0]) * 10) / 10; }
console.log(JSON.stringify(out));
"""


def run(swap_src, label):
    src = swap_src + "\n" + BODY
    with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False, encoding="utf-8") as fh:
        fh.write(src)
        path = fh.name
    try:
        res = subprocess.run(["node", path], capture_output=True, text=True, timeout=30)
    finally:
        os.unlink(path)
    if res.returncode != 0:
        _fail("%s harness did not run: %s" % (label, res.stderr.strip()[:400]))
    return json.loads(res.stdout.strip().splitlines()[-1])


FIX = _lift("function _fixWindow(start, len, dur, snap)")

# 1) REAL -- 121 frames everywhere, start on the frame grid, window inside the clip.
real = run(FIX, "real")
if real["midFrames"] != 121:
    _fail("sliding the fixed window changed the frame count (got %s, expected 121) -- "
          "'121 means 121' is broken" % real["midFrames"])
if not real["midOnGrid"]:
    _fail("the slid window start (%.6f s) is not on a frame boundary -- the backend "
          "slice becomes non-deterministic" % real["midStart"])
if real["headFrames"] != 121 or real["headStart"] != 0:
    _fail("head-clamp broke the window (start %s, frames %s; expected 0 / 121)"
          % (real["headStart"], real["headFrames"]))
if real["tailFrames"] != 121:
    _fail("tail-clamp changed the frame count (got %s, expected 121)" % real["tailFrames"])
if real["tailEnd"] > real["dur"] + 1e-6:
    _fail("tail-clamp let the window hang past the clip end (%.4f > %.4f)"
          % (real["tailEnd"], real["dur"]))
if real["audLen"] != 3.0:
    _fail("the audio fixed window is not 3.0 s (got %s)" % real["audLen"])

# 2) MUTANT -- remove the end clamps (both: pre- and post-snap -- the second would
#    heal the first); the tail window must now hang past the clip.
MUT = FIX.replace("dur - len", "dur")
if MUT == FIX:
    _fail("mutation target (end clamp) not found -- harness out of sync with _fixWindow")
mut = run(MUT, "mutant")
if mut["tailEnd"] <= mut["dur"] + 1e-6:
    _fail("MUTATION NOT CAUGHT: with the end clamp removed, the tail window still stayed "
          "inside the clip (end %.4f <= dur %.4f) -- the DRIVEN check does not prove the "
          "clamp is load-bearing" % (mut["tailEnd"], mut["dur"]))

print("[test_v625_fix_trim] PASS -- 121 frames stay 121 through slide/head-clamp/tail-clamp, "
      "start lands on the frame grid, the 3.0 s audio window holds; removing the end clamp "
      "hangs the window past the clip (mutation caught); fps in the listing + amber badge present")
sys.exit(0)
