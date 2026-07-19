"""
Guard v536 -- box-fit preview PLUS firm node bounds (rows can't telescope).

onResize derives BOTH floors from one computeSize() call: a width floor
(max(NODE_MIN_W, computeSize[0])) and a height floor (widgetStack + minimum
preview). Everything from v535 (box-fit hug + drag-scale, loop default-off, no
fullscreen) is kept. Structure locks only; negative locks keep defects out.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _fail(msg):
    print("FAIL: " + msg)
    sys.exit(1)


def main():
    src = open(os.path.join(ROOT, "web", "js", "ph_save.js"), encoding="utf-8").read()

    # v536 firm bounds
    if "widgetStack" not in src:
        _fail("ph_save.js: height floor (widgetStack) missing")
    if "Math.max(NODE_MIN_W, Math.ceil(cs[0]" not in src.replace("  ", " ").replace(" ", ""):
        # tolerant check: width floor uses NODE_MIN_W and computeSize width
        if "NODE_MIN_W" not in src or "cs[0]" not in src:
            _fail("ph_save.js: width floor (NODE_MIN_W vs computeSize width) missing")
    if "floorTotal" not in src:
        _fail("ph_save.js: height floor (floorTotal) missing")

    # box-fit machinery (v535) kept
    for tok in ("_phBudget", "_phRelayout", "object-fit", ".style.width =",
                ".style.height =", "_phMedia"):
        if tok not in src:
            _fail(f"ph_save.js: box-fit token '{tok}' missing")
    if "el.style.maxHeight" in src:
        _fail("ph_save.js: max-height natural-size cap is back (dead space)")

    # shared bar + loop toggle default-off, no fullscreen
    for tok in ('type = "range"', '" / "', "loopBtn", "_phLoop", "\\u27f3"):
        if tok not in src:
            _fail(f"ph_save.js: bar/loop token '{tok}' missing")
    if "el.loop = !!this._phLoop" not in src:
        _fail("ph_save.js: video loop not gated on the toggle")
    if "requestFullscreen" in src or "Fullscreen" in src:
        _fail("ph_save.js: fullscreen is back")

    # autoplay-once muted, banner
    for tok in ("el.muted = true", "el.autoplay = true", "onloadedmetadata",
                "onConfigure", "_phFlipTimer", "[PLS v536 DIAG] ph_save.js"):
        if tok not in src:
            _fail(f"ph_save.js: '{tok}' missing")

    # negative locks
    if "if (item.autoplay)" in src:
        _fail("ph_save.js: opt-in autoplay gate is back")
    if "setSize(this.computeSize())" in src.replace(" ", ""):
        _fail("ph_save.js: whole-computeSize width-shrink fit is back")
    if "firstElementChild" in src:
        _fail("ph_save.js: onResize targets the wrapper again")
    if "el.loop = N > 1" in src or "el.loop = true" in src:
        _fail("ph_save.js: unconditional loop is back")

    if "for e in ui" not in open(os.path.join(ROOT, "nodes", "ph_save.py"), encoding="utf-8").read():
        _fail("ph_save.py: still path no longer sends every saved still")

    print("PASS: v536 box-fit preview + firm node bounds (no row telescoping)")
    sys.exit(0)


if __name__ == "__main__":
    main()
