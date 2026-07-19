#!/usr/bin/env python3
"""
test_v664_drop_zones -- the drop ZONE decides, not a modifier and not the drag source.

Frank, 2026-07-19: "Wenn wir statt einem Feld ZWEI Felder anzeigen, die bereits die
Auswahl SIND: Links neue Datei in Input // Rechts Datei als Auswahl mit Folder laden"
-- and: no permanent buttons, the choice appears only while something is being dragged
in.

v664 therefore raises two zones during a drag. Aiming at one IS the decision, so the
gesture needs no prior knowledge and nothing held down. Shift survives only as a silent
alias for the "load from where it is" zone, and a drop that misses both zones follows
that same rule -- one mental model, never two competing ones.

Guards, all must hold, mutation-tested:

  STATIC -- both zones carry a data-zone; every zone listens for its own dragenter /
            dragover / dragleave / drop; the zone's drop calls _intentFor with the
            zone; the root's drop passes an empty zone (the miss case); the zones are
            pointer-active while the overlay itself stays click-through; the overlay
            only exists inside the drag (no permanent widget was added).

  DRIVEN -- run the REAL _intentFor over every combination of zone and modifier: the
            zone always wins, and only a miss consults Shift. Mutating the zone test
            away lets the modifier override an explicit aim -- caught here.
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
    print("[test_v664_drop_zones] FAIL: " + msg)
    sys.exit(1)


def _lift_method(sig):
    if sig not in JS:
        _fail("could not find `%s`" % sig)
    s = JS[JS.index(sig):]
    return s[:s.index("\n    }") + 6]


# ---------------------------------------------------------------------------
# STATIC -- the zones, their listeners, the intent handover.
# ---------------------------------------------------------------------------
for sym, why in [
    ('data-zone="copy"', "the copy zone lost its zone marker"),
    ('data-zone="pin"', "the pin zone lost its zone marker"),
    ("const zone = z.dataset.zone", "the zone listeners no longer read which zone they are"),
    ('z.addEventListener("dragenter"', "a zone no longer lights on entry"),
    ('z.addEventListener("dragleave"', "a zone no longer un-lights on leave"),
    ('z.addEventListener("drop"', "a zone can no longer be dropped on"),
    ("this._onDrop(ev, this._intentFor(zone, ev.shiftKey))",
     "a zone drop no longer hands its own zone to the intent"),
    ('this._onDrop(ev, this._intentFor("", ev.shiftKey))',
     "a drop that misses the zones no longer falls back to the modifier rule"),
]:
    if sym not in JS:
        _fail(why)

# The overlay must stay click-through while the zones inside it take the pointer,
# otherwise the node cannot be used normally.
m = re.search(r"\.ph-media-drophint \{[^}]*\}", JS)
if not m or "pointer-events:none" not in m.group(0):
    _fail("the drop overlay is no longer click-through -- it would swallow clicks on "
          "the node")
m = re.search(r"\.ph-media-drophint \.ph-dz \{[^}]*\}", JS)
if not m or "pointer-events:auto" not in m.group(0):
    _fail("the zones do not take the pointer -- they could never be aimed at")
m = re.search(r"\.ph-media-drophint \{[^}]*\}", JS)
if "display:none" not in m.group(0):
    _fail("the zones are visible outside a drag -- Frank asked for no permanent "
          "additions to the node's face")

# Wrapping instead of a JS width threshold (a narrow node stacks the zones).
m = re.search(r"\.ph-media-drophint \.ph-dz \{[^}]*\}", JS)
if "min-width" not in m.group(0) or "flex:1 1" not in m.group(0):
    _fail("the zones no longer wrap on a narrow node")


# ---------------------------------------------------------------------------
# DRIVEN -- the REAL _intentFor across zone x modifier.
# ---------------------------------------------------------------------------
BODY = """
const T = new (class {
  %s
})();
const out = {
  copyZonePlain: T._intentFor("copy", false),
  copyZoneShift: T._intentFor("copy", true),
  pinZonePlain:  T._intentFor("pin", false),
  pinZoneShift:  T._intentFor("pin", true),
  missPlain:     T._intentFor("", false),
  missShift:     T._intentFor("", true),
};
console.log(JSON.stringify(out));
""" % _lift_method("_intentFor(zone, shift)")


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
if got["copyZonePlain"] != "copy" or got["copyZoneShift"] != "copy":
    _fail("the copy zone does not win over the modifier: %r" % (got,))
if got["pinZonePlain"] != "pin" or got["pinZoneShift"] != "pin":
    _fail("the pin zone does not win over the modifier: %r" % (got,))
if got["missPlain"] != "copy" or got["missShift"] != "pin":
    _fail("a drop that missed both zones no longer follows the modifier rule: %r" % (got,))

# MUTANT -- let the modifier speak even when a zone was aimed at.
MUT = BODY.replace('if (zone === "copy" || zone === "pin") return zone;', "")
if MUT == BODY:
    _fail("could not build the mutant -- the zone test moved")
mut = run(MUT, "mutant")
if mut["copyZoneShift"] == "copy":
    _fail("the guard does not catch the modifier overriding an explicit aim")

print("[test_v664_drop_zones] OK -- the aimed zone always wins, Shift only settles a "
      "miss, the zones exist only during a drag (mutation-tested)")
