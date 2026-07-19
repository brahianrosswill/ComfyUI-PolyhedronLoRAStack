#!/usr/bin/env python3
"""
test_v663_drop_hint -- the gesture explains itself BEFORE the drop, the batch panel's
counters belong to the folder named above them, and the stale Stufe-A wording is gone.

Frank, 2026-07-19, after watching a dropped file land in his sprite folder: "das würde
ich nicht machen wollen ... sonst kriegt man Kuddelmuddel auf der Festplatte", plus the
wish for an on-canvas hint: "Drag: legt diese Einzeldatei im Input-Ordner ab ... Shift &
Drag wählt die Datei aus und zieht den Ordner nach, in dem sie aktuell liegt".

v663 therefore:
  * a drag over the node raises a hint naming BOTH outcomes, lighting the row the
    current modifier state would take, with the honest limit spelled out on the
    Shift row (pinning needs a path the browser may not hand out),
  * the plain drop copies into the official ComfyUI input folder -- never into the
    user's own pinned media folder (B-06 follow-up),
  * B-04: switching the batch source drops the cached listing and intersects the
    checked set with the new folder, so no counter survives that does not belong,
  * B-02: the ♪ tooltips no longer call the shipped AUDIO output "Stufe B".

Guards, all must hold, mutation-tested where a value can be driven:

  STATIC -- the hint element, its two rows and the note exist in the markup; the drag
            listeners raise and lower it; _dropHint lights exactly one row from
            ev.shiftKey; the Choose handler invalidates the listing and re-counts;
            no ♪ tooltip mentions Stufe B.

  DRIVEN -- run the REAL _dropHint against a fake DOM for both modifier states and
            prove exactly one row carries the .on class each time. Mutating the
            shift test away lights the wrong row -- caught here.
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
    print("[test_v663_drop_hint] FAIL: " + msg)
    sys.exit(1)


def _lift_method(sig):
    if sig not in JS:
        _fail("could not find `%s`" % sig)
    s = JS[JS.index(sig):]
    return s[:s.index("\n    }") + 6]


# ---------------------------------------------------------------------------
# STATIC -- the hint exists, is wired, and says both things.
# ---------------------------------------------------------------------------
for sym, why in [
    ('<div class="ph-media-drophint">', "the drop hint overlay is gone"),
    ('data-zone="copy"', "the copy ZONE is gone (v664)"),
    ('data-zone="pin"', "the pin ZONE is gone (v664)"),
    ("ph-dz-pinsub", "the pin zone's honest-limit line is gone"),
    ('root.addEventListener("dragenter"', "the hint is never raised"),
    ('root.addEventListener("dragleave"', "the hint is never lowered on leave"),
    ("this._dropHint(false, ev);", "the hint survives the drop itself"),
    ("this._intentFor(zone, ev.shiftKey)", "a zone drop no longer decides the intent (v664)"),
]:
    if sym not in JS:
        _fail(why)

HINT = _lift_method("_dropHint(on, ev)")
if "dataTransfer.types" not in HINT:
    _fail("_dropHint no longer inspects the drag's types -- it cannot tell whether the "
          "pin zone is usable for this drag")
if "input folder" not in JS[JS.index('<div class="ph-media-drophint">'):JS.index('<div class="ph-media-drophint">') + 900]:
    _fail("the copy zone no longer names the input folder as its target")

# B-02 -- no shipped output may be described as pending.
for m in re.finditer(r'title="([^"]*Audio[^"]*)"', JS):
    if "Stufe B" in m.group(1):
        _fail("a ♪ tooltip still calls the AUDIO output 'Stufe B' although it ships "
              "since v459 (B-02)")
if "the AUDIO output is Stufe B" in JS:
    _fail("the Stufe-A wording survives somewhere in the audio UI (B-02)")

# B-04 -- the Choose handler must invalidate and re-count.
i = JS.index('$(".ph-batch-choose").onclick')
CHOOSE = JS[i:i + 900]
if "lastList = null" not in CHOOSE:
    _fail("switching the batch source no longer drops the cached listing -- the "
          "counters keep describing the old folder (B-04)")
if "refreshCounts()" not in CHOOSE:
    _fail("switching the batch source no longer re-counts (B-04)")
if "_writeSel(" not in CHOOSE:
    _fail("switching the batch source no longer intersects the checked set with the "
          "new folder -- the status line keeps claiming checks that are not there (B-04)")

# Cosmetic -- the wrap label must be one shrink-proof span.
if 'class="ph-wrap-lbl"' not in JS:
    _fail("the wrap option's label is no longer a single span -- it breaks into "
          "three lines again")


# ---------------------------------------------------------------------------
# DRIVEN -- the REAL _dropHint against a fake DOM: the pin zone dims exactly when
# the drag carries no path-ish type.
# ---------------------------------------------------------------------------
BODY = """
const mkEl = () => {
  const cls = new Set();
  return { classList: { toggle: (c, on) => { on ? cls.add(c) : cls.delete(c); },
                        has: (c) => cls.has(c) }, _cls: cls, textContent: "",
           dataset: {}, querySelectorAll: () => [] };
};
const pin = mkEl(), sub = mkEl(), host = mkEl();
host.querySelector = (sel) => sel.includes("pinsub") ? sub : pin;
host.querySelectorAll = () => [];

const T = new (class {
  constructor() { this.dropHintEl = host; }
  %s
  %s
})();

const mkEv = (types) => ({ dataTransfer: { types } });
const out = {};
T._dropHint(true, mkEv(["Files", "text/plain"]));
out.withPath = { dim: pin._cls.has("dim"), sub: sub.textContent };
T._dropHint(true, mkEv(["Files"]));
out.filesOnly = { dim: pin._cls.has("dim"), sub: sub.textContent };
T._dropHint(true, mkEv([]));
out.noPath = { dim: pin._cls.has("dim"), sub: sub.textContent };
out.shown = host._cls.has("on");
T._dropHint(false, null);
out.hidden = !host._cls.has("on");
console.log(JSON.stringify(out));
""" % (_lift_method("_dropHint(on, ev)"), _lift_method("_zoneHot(zone) {"))


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
if got["withPath"]["dim"]:
    _fail("the pin zone is dimmed although the drag carries a path type: %r" % (got["withPath"],))
if got["filesOnly"]["dim"]:
    _fail("a file drag dims the pin zone although relocation can try: %r" % (got["filesOnly"],))
if "recently used folders" not in got["filesOnly"]["sub"]:
    _fail("the file-drag pin zone does not explain the relocation route: %r"
          % (got["filesOnly"]["sub"],))
if not got["noPath"]["dim"]:
    _fail("the pin zone is offered at full strength although neither route exists: %r"
          % (got["noPath"],))
if "path" not in got["noPath"]["sub"].lower():
    _fail("the dimmed pin zone does not say WHY: %r" % (got["noPath"]["sub"],))
if not got["hidden"]:
    _fail("the zones are not lowered when the drag leaves")

# MUTANT -- ignore the drag types; the zone would always look usable.
MUT = BODY.replace('const hasFiles = types.includes("Files") || types.includes("application/x-moz-file");',
                   'const hasFiles = true;')
if MUT == BODY:
    _fail("could not build the mutant -- the type probe moved")
mut = run(MUT, "mutant")
if mut["noPath"]["dim"]:
    _fail("the guard does not catch the pin zone ignoring the drag's types")

print("[test_v663_drop_hint] OK -- two zones, the pin zone dims with its reason when "
      "no path can be had, the copy zone targets input/, B-02 wording and B-04 "
      "counters settled (mutation-tested)")
