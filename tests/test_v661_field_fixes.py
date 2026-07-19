#!/usr/bin/env python3
"""
test_v661_field_fixes -- the three findings measured on Frank's screen on 2026-07-19,
each pinned so it cannot come back.

B-01  The batch PREVIEW judged the whole folder instead of the checked set. With 10
      uniform sprites checked in a folder of 13 (3 of them a different size), the
      Selection info box warned "13 frames - sizes differ / 10 differ - none(strict)
      will refuse" while the actual run went through fine (Save: 10/10 frames). The
      backend has given an explicit checked set precedence since v528
      (_load_image_batch / _proc_resolve_files); the preview now mirrors that, with
      skip / every-nth / cap slicing ON the checked set (select_slice parity).

B-03  A dropped/uploaded .m4a was renamed to .png, because _safe_media_name only
      whitelisted image + video extensions. Extension-driven kind detection then
      routed an AAC file into the image path: black preview, no audio pairing.
      Audio has been a browsable kind since v457, so its extensions belong in the
      whitelist. Second half of the same finding: when the thumb fallback failed
      too, the <img> stayed silently blank -- a black Selection with no hint that
      survived every folder change (navigation deliberately keeps the selection,
      v458). The preview now says so.

B-05  The batch panel showed no options at all in "Separate files" mode: the proc
      block, the Selection row and the live status line were all missing. Cause was
      structural -- .ph-rub.ph-mrow-proc sat INSIDE .ph-rub.ph-mrow-frames, so
      applyModeSwitch's display:none on the frames rubric took everything with it.
      The two rubrics are siblings again.

Guards, all must hold, mutation-tested (inject the wound, prove the catch):

  STATIC  -- _safe_media_name whitelists audio; the preview's second-stage onerror
             renders a message; the batch preview consults the checked set; the
             sliced-only helper exists.

  DRIVEN  -- parse the batch card's markup and prove BOTH rubrics are siblings under
             .ph-batch-card and that the Selection row + live line are NOT inside a
             rubric (the wound: re-nest proc inside frames -> caught).
          -- run the REAL _sliceFrames/_selectFrames pair in node: a checked set is
             sliced without the name filter, and skip/every-nth still bite.
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
    print("[test_v661_field_fixes] FAIL: " + msg)
    sys.exit(1)


# ---------------------------------------------------------------------------
# STATIC -- B-03 upload naming, B-03 preview message, B-01 wiring.
# ---------------------------------------------------------------------------
m = re.search(r"def _safe_media_name\(filename: str\) -> str:.*?\n    return .*?\n", PY, re.S)
if not m:
    _fail("_safe_media_name is gone -- the upload name defense cannot be checked")
if "_MEDIA_AUDIO_EXTS" not in m.group(0):
    _fail("_safe_media_name no longer whitelists audio extensions -- a dropped .m4a "
          "would be renamed to .png again (B-03)")

if "Cannot preview " not in JS:
    _fail("the Selection preview no longer reports an undecodable file -- a broken "
          "pick falls back to a silent black panel (B-03)")

for sym, why in [
    ("_sliceFrames(ordered, cfg, skip, cap)",
     "the slice-only stage is gone -- a checked set cannot be sliced without the filter"),
    ("checkedSet.length",
     "the batch preview no longer branches on the checked set (B-01)"),
]:
    if sym not in JS:
        _fail(why)


# ---------------------------------------------------------------------------
# DRIVEN 1 -- the batch card's structure (B-05).
# ---------------------------------------------------------------------------
def card_shape(src):
    """(frames_parent, proc_parent, selection_in_rubric, live_in_rubric, balance)."""
    i = src.find('<div class="ph-batch-card">')
    if i < 0:
        _fail("the batch card markup is gone")
    blk = src[i:src.index("`;", i)]
    balance = len(re.findall(r"<div\b", blk)) - len(re.findall(r"</div>", blk))
    depth, stack = 0, []
    frames_parent = proc_parent = None
    sel_in_rubric = live_in_rubric = None
    for m in re.finditer(r'<div\b[^>]*class="([^"]*)"[^>]*>|</div>', blk):
        if m.group(0).startswith("</"):
            depth -= 1
            if stack:
                stack.pop()
            continue
        cls = m.group(1)
        parent = stack[-1] if stack else "-"
        in_rubric = any("ph-rub" in x for x in stack)
        if "ph-mrow-frames" in cls:
            frames_parent = parent
        elif "ph-mrow-proc" in cls:
            proc_parent = parent
        elif "ph-sel-btns" in cls:
            sel_in_rubric = in_rubric
        elif "ph-batch-live" in cls:
            live_in_rubric = in_rubric
        depth += 1
        stack.append(cls.split()[0])
    return frames_parent, proc_parent, sel_in_rubric, live_in_rubric, balance


fp, pp, sel_r, live_r, bal = card_shape(JS)
if bal != 0:
    _fail("the batch card's <div> tags do not balance (%+d) -- the panel would leak "
          "rows out of the card" % bal)
if fp != "ph-batch-card":
    _fail("the Video-frames rubric is not a direct child of the batch card (parent: %s)" % fp)
if pp != "ph-batch-card":
    _fail("the Separate-files rubric is nested inside another block (parent: %s) -- "
          "hiding the frames rubric would hide it too (B-05)" % pp)
if sel_r:
    _fail("the Selection row sits inside a mode rubric -- it vanishes in one mode (B-05)")
if live_r:
    _fail("the live status line sits inside a mode rubric -- it vanishes in one mode (B-05)")

# MUTANT -- put proc back inside frames; the guard must catch it.
wound = JS.replace('            </div>\n            <div class="ph-rub ph-mrow-proc">',
                   '            <div class="ph-rub ph-mrow-proc">', 1)
if wound == JS:
    _fail("could not build the B-05 mutant -- the closing tag before the proc rubric moved")
_fp, _pp, _s, _l, _bal = card_shape(wound)
if _pp == "ph-batch-card" and _bal == 0:
    _fail("the B-05 guard does not catch a re-nested proc rubric")


# ---------------------------------------------------------------------------
# DRIVEN 2 -- the REAL slice/select pair (B-01).
# ---------------------------------------------------------------------------
def _lift_method(sig):
    if sig not in JS:
        _fail("could not find `%s`" % sig)
    s = JS[JS.index(sig):]
    return s[:s.index("\n    }") + 6]


SLICE = _lift_method("_sliceFrames(ordered, cfg, skip, cap)")
SELECT = _lift_method("_selectFrames(names, cfg, skip, cap, byName)")

BODY = """
const T = new (class {
  _filterNames(names, f) { return f === "*" ? names.slice() : names.filter((n) => n.includes(f)); }
  _orderNames(names) { return names.slice().sort(); }
  %s
  %s
})();

const folder = ["a1.png","a2.png","a3.png","b1.png","b2.png"];
const checked = ["a1.png","a3.png","b2.png"];
const cfg = { name_filter: "a", sort_mode: "name (natural)", every_nth: 1 };
const out = {};
// the checked set is sliced WITHOUT the name filter (backend parity: select_slice)
out.checked = T._sliceFrames(T._orderNames(folder.filter((n) => checked.includes(n))), cfg, 0, 0);
// no checked set -> the filter still rules
out.filtered = T._selectFrames(folder, cfg, 0, 0, {});
// every-nth still bites ON the checked set
out.everySecond = T._sliceFrames(T._orderNames(folder.filter((n) => checked.includes(n))),
                                 { ...cfg, every_nth: 2 }, 0, 0);
console.log(JSON.stringify(out));
""" % (SLICE, SELECT)


def run(src, label):
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


got = run(BODY, "real")
if got["checked"] != ["a1.png", "a3.png", "b2.png"]:
    _fail("a checked set is not carried through untouched by the name filter: %r"
          % (got["checked"],))
if got["filtered"] != ["a1.png", "a2.png", "a3.png"]:
    _fail("without a checked set the name filter no longer rules: %r" % (got["filtered"],))
if got["everySecond"] != ["a1.png", "b2.png"]:
    _fail("every-nth no longer slices the checked set: %r" % (got["everySecond"],))

# MUTANT -- run the checked set through the filter as well; the guard must catch it.
MUT = BODY.replace(
    'out.checked = T._sliceFrames(T._orderNames(folder.filter((n) => checked.includes(n))), cfg, 0, 0);',
    'out.checked = T._selectFrames(folder.filter((n) => checked.includes(n)), cfg, 0, 0, {});')
mut = run(MUT, "mutant")
if mut["checked"] == ["a1.png", "a3.png", "b2.png"]:
    _fail("the B-01 guard does not catch the filter being applied to a checked set")

print("[test_v661_field_fixes] OK -- B-01 preview parity, B-03 upload naming + preview "
      "message, B-05 rubric siblings (all mutation-tested)")
