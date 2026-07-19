#!/usr/bin/env python3
"""
test_v626_full_view -- the trim fields row can never be clipped away, the height floor
leaves room for the Selection panes, and a selected video plays ONCE then stops.

v626 fixes Frank's hidden-areas report: with the v625 Fix field the trim row
(Start . End . Fix . len) outgrew a narrow Selection column and the column's
overflow:hidden CLIPPED it -- fields only appeared when the node was dragged very wide.
And a short node clipped the trim/audio panes at the bottom, because _computeDomMin's
floor never counted them. Plus Frank's play-once rule: a fresh selection or reload runs
the clip exactly one time and holds; pressing play again loops -- and the natural END
must not persist as a user pause (or the next reload would start frozen).

Guards, all must hold, mutation-tested (inject the wound, prove the catch):

  STATIC -- .ph-trim-fields wraps (flex-wrap); the Selection preview video is created
            with loop=false, carries the once-ended loop-restore, and the pause-persist
            is gated on !v.ended.

  DRIVEN -- lift and RUN the REAL _computeDomMin against a mock node with a visible
            trim pane (64px) and caption (14px): the floor must include them. Mutating
            the pane sum away (paneH -> 0) drops the floor back to the clipping height
            -- caught here.
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


def _fail(msg):
    print("[test_v626_full_view] FAIL: " + msg)
    sys.exit(1)


# ---------------------------------------------------------------------------
# STATIC -- wrap CSS + play-once wiring.
# ---------------------------------------------------------------------------
m = re.search(r"\.ph-trim-fields \{[^}]*\}", JS)
if not m or "flex-wrap:wrap" not in m.group(0):
    _fail(".ph-trim-fields no longer wraps -- a narrow Selection column clips the "
          "Start/End/Fix/len row again (Frank's hidden-areas report)")

if "v.muted = true; v.loop = false; v.playsInline = true; v.controls = true;" not in JS:
    _fail("the Selection preview video no longer starts with loop=false -- it loops "
          "forever instead of playing once and stopping")
if 'v.addEventListener("ended", () => { v.loop = true; }, { once: true });' not in JS:
    _fail("the once-ended loop-restore is gone -- pressing play after the single pass "
          "no longer loops")
if 'v.addEventListener("pause", () => { if (v.isConnected && !v.ended) this.paused = true; });' not in JS:
    _fail("the pause-persist is no longer gated on !v.ended -- the natural end counts "
          "as a user pause and the next reload starts frozen instead of playing once")


# ---------------------------------------------------------------------------
# DRIVEN -- run the REAL _computeDomMin with a visible trim pane, then a MUTANT.
# ---------------------------------------------------------------------------
sig = "    _computeDomMin() {"
if sig not in JS:
    _fail("_computeDomMin is gone")
s = JS[JS.index(sig):]
BODY_SRC = s[:s.index("\n    }") + len("\n    }")]
FN = "const _computeDomMin = " + BODY_SRC.strip().replace("_computeDomMin() {", "function () {", 1)

HARNESS = """
%s
const mk = (h) => ({ getBoundingClientRect: () => ({ height: h }) });
const node = { barEl: mk(28), batchStatusEl: mk(0), videoTrimEl: mk(64),
               audioPaneEl: mk(0), previewCapEl: mk(14), _canvasScale: () => 1 };
const withPanes = _computeDomMin.call(node);
node.videoTrimEl = mk(0); node.previewCapEl = mk(0);
const withoutPanes = _computeDomMin.call(node);
console.log(JSON.stringify({ withPanes, withoutPanes }));
"""


def run(fn_src, label):
    src = HARNESS % fn_src
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


# 1) REAL -- a visible 64px trim pane + 14px caption must raise the floor by exactly 78.
real = run(FN, "real")
if real["withPanes"] - real["withoutPanes"] != 78:
    _fail("the floor does not count the visible Selection panes (with %s / without %s, "
          "expected +78) -- a short node clips the trim strip again"
          % (real["withPanes"], real["withoutPanes"]))
if real["withoutPanes"] < 150:
    _fail("the base floor collapsed (%s) -- bar/grid/foot are no longer counted"
          % real["withoutPanes"])

# 2) MUTANT -- zero the pane sum; the floor must stop counting the panes.
MUT = re.sub(r"const paneH = [^;]+;", "const paneH = 0;", FN, count=1)
if MUT == FN:
    _fail("mutation target (const paneH = ...) not found -- harness out of sync")
mut = run(MUT, "mutant")
if mut["withPanes"] != mut["withoutPanes"]:
    _fail("MUTATION NOT CAUGHT: with the pane sum zeroed the floor still differs "
          "(with %s / without %s) -- the DRIVEN check does not prove the pane term is "
          "load-bearing" % (mut["withPanes"], mut["withoutPanes"]))

print("[test_v626_full_view] PASS -- fields row wraps; floor counts visible panes "
      "(+78 for 64px trim + 14px caption) and zeroing the pane sum is caught; "
      "selection video plays once (loop=false, once-ended restore, ended != pause)")
sys.exit(0)
