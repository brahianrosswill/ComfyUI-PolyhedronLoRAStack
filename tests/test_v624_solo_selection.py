#!/usr/bin/env python3
"""
test_v624_solo_selection -- Solo-Selection remembers the node size PER MODE, the dims
badge lives top-right, and a removed node no longer leaks its document keydown handler.

v624 adds a Solo mode to the Media Loader: the tile grid (and its pager) hide, the
Selection fills the whole node like a plain Load node. Toggling records the size you are
LEAVING under that mode's slot and restores the size the target mode was last used at --
so tiles come back at their old height, and Solo comes back at its enlarged one.

Guards, all must hold, mutation-tested (inject the wound, prove the catch):

  STATIC -- the CSS hides grid+pager under .ph-solo and unclamps the preview; the dims
            badge rule anchors right (and the old top:26px tile override is gone); the
            markup carries the ph-solo-toggle; _destroy removes the document keydown
            handler (the leak found in the v624 review).

  DRIVEN -- lift and RUN the REAL _viewSwap through a full toggle sequence:
              tiles[400,500] -> Solo (first visit: no size to restore, tiles size recorded)
              resize in Solo, -> Tiles: restores [400,500], Solo size recorded
              -> Solo again: restores the enlarged Solo size.
            Mutating the restore lookup away (target -> null) leaves both restores empty
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
    print("[test_v624_solo_selection] FAIL: " + msg)
    sys.exit(1)


def _lift(sig):
    if sig not in JS:
        _fail("could not find `%s`" % sig)
    s = JS[JS.index(sig):]
    return s[:s.index("\n}") + 2]


# ---------------------------------------------------------------------------
# STATIC -- CSS, markup, badge, and the destroy leak fix.
# ---------------------------------------------------------------------------
if ".ph-media.ph-solo .ph-media-grid, .ph-media.ph-solo .ph-media-pager { display:none; }" not in JS:
    _fail("solo CSS no longer hides the tile grid + pager -- Solo mode shows both")
if ".ph-media.ph-solo .ph-media-preview { max-width:none; flex:1 1 auto; }" not in JS:
    _fail("solo CSS no longer unclamps the Selection column -- it stays a narrow side pane")
if "ph-solo-toggle" not in JS:
    _fail("the Solo toggle button is gone from the Selection header")

m = re.search(r"\.ph-media-tile \.ph-dim, \.ph-media-pop \.ph-dim \{[^}]*\}", JS)
if not m:
    _fail("the shared dims-badge rule is gone")
if "right:3px" not in m.group(0) or "left:3px" in m.group(0):
    _fail("the dims badge is no longer anchored top-RIGHT (found: %s)" % m.group(0)[:90])
if re.search(r"\.ph-media-tile \.ph-dim \{ top:26px", JS):
    _fail("the old top:26px tile override is back -- the badge would sit below the corner again")
mprev = re.search(r"\.ph-media-prev-media \.ph-dim-prev \{[^}]*\}", JS)
if not mprev or "right:5px" not in mprev.group(0):
    _fail("the Selection (preview) dims badge is no longer anchored top-right")

DESTROY = _lift("    _destroy() {")
if 'removeEventListener("keydown", this._selKeyHandler, true)' not in DESTROY:
    _fail("_destroy no longer removes the document keydown handler -- every deleted loader "
          "node leaks a live capture-phase handler plus its whole UI tree")


# ---------------------------------------------------------------------------
# DRIVEN -- run the REAL _viewSwap through the toggle sequence, then a MUTANT.
# ---------------------------------------------------------------------------
BODY = """
const out = {};
// tiles at [400,500] -> Solo. First visit: nothing to restore, tiles size recorded.
const r1 = _viewSwap(null, true, [400, 500]);
out.firstRestore = r1.size;                       // must be null
out.tilesRecorded = r1.view.tilesSize;            // must be [400,500]
// user enlarges the node in Solo to [800,900], then toggles back to Tiles.
const r2 = _viewSwap(r1.view, false, [800, 900]);
out.tilesRestore = r2.size;                       // must be [400,500]
out.soloRecorded = r2.view.soloSize;              // must be [800,900]
// back to Solo -- the enlarged size must come back.
const r3 = _viewSwap(r2.view, true, [400, 500]);
out.soloRestore = r3.size;                        // must be [800,900]
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


SWAP = _lift("function _viewSwap(view, goSolo, curSize)")

# 1) REAL -- full sequence.
real = run(SWAP, "real")
if real["firstRestore"] is not None:
    _fail("first switch into Solo restored a size (%s) although none was ever recorded"
          % real["firstRestore"])
if real["tilesRecorded"] != [400, 500]:
    _fail("switching into Solo did not record the tiles size (got %s, expected [400,500])"
          % real["tilesRecorded"])
if real["tilesRestore"] != [400, 500]:
    _fail("switching back to Tiles did not restore the remembered tiles size (got %s, "
          "expected [400,500]) -- Frank's 'springt auf die festgelegte Hoehe zurueck' is broken"
          % real["tilesRestore"])
if real["soloRecorded"] != [800, 900]:
    _fail("leaving Solo did not record the enlarged Solo size (got %s, expected [800,900])"
          % real["soloRecorded"])
if real["soloRestore"] != [800, 900]:
    _fail("re-entering Solo did not restore its remembered size (got %s, expected [800,900]) "
          "-- 'merkt sich den letzten Stand' is broken" % real["soloRestore"])

# 2) MUTANT -- kill the restore lookup; both restores must go null.
MUT = SWAP.replace("const t = v.solo ? v.soloSize : v.tilesSize;", "const t = null;")
if MUT == SWAP:
    _fail("mutation target 'const t = ...' not found -- harness out of sync with _viewSwap")
mut = run(MUT, "mutant")
if mut["tilesRestore"] == [400, 500]:
    _fail("MUTATION NOT CAUGHT: with the restore lookup removed, switching back to Tiles still "
          "restored [400,500] -- the DRIVEN check does not prove the lookup carries the memory")

print("[test_v624_solo_selection] PASS -- per-mode size memory holds through the full toggle "
      "sequence (tiles [400,500] <-> solo [800,900]); killing the restore lookup breaks it "
      "(mutation caught); badge top-right + destroy leak fix present")
sys.exit(0)
