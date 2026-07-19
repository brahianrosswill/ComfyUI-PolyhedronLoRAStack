#!/usr/bin/env python3
"""
test_v662_drop_intent -- a drop's OUTCOME is decided by the user, not by the source
application, and the node always says which of the two things it did.

Frank, 2026-07-19: "manchmal wird der ganze Ordner mit reingeladen, manchmal nur eine
Datei". Cause (B-06): _onDrop branched on what the drag source happened to put into
the DataTransfer -- a path TEXT pinned the origin folder (nothing copied), real FILES
were copied into the pinned folder. Both are useful; which one you got was luck.

v662 split the intent; v663 settled the mapping (Frank) and the copy target:
  copy intent -> ALWAYS copy into the official ComfyUI input folder. One outcome in
                 every environment, and never a write into the user's own folders.
  pin intent  -> leave the file where it is: PIN its folder and select it. Needs a
                 path (dropped path text, or file.path in the desktop client); where
                 none is available the drop says so and copies instead.
v664 moved the choice onto two drop ZONES (see test_v664_drop_zones); this guard
covers what each intent then DOES.
Either branch ends with a self-clearing note in the status line, including the count
for a multi-file drop (which copies every file but can only select one -- the third
face of the same confusion).

Guards, all must hold, mutation-tested (inject the wound, prove the catch):

  STATIC -- the shift lever exists and gates the pin branch; the path helper reads
            BOTH the text and file.path; every exit writes a note; the note clears
            itself by re-rendering the status line.

  DRIVEN -- run the REAL _dropPathOf against four DataTransfer shapes (path text,
            Windows file.path, a bare name with no path, junk text) and prove it
            only ever returns an absolute path. Mutating the absolute-path test away
            lets a bare file name through -- caught here.
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
    print("[test_v662_drop_intent] FAIL: " + msg)
    sys.exit(1)


def _lift_method(sig):
    if sig not in JS:
        _fail("could not find `%s`" % sig)
    s = JS[JS.index(sig):]
    return s[:s.index("\n    }") + 6]


# ---------------------------------------------------------------------------
# STATIC -- the intent lever, the path sources, the notes.
# ---------------------------------------------------------------------------
DROP = _lift_method("async _onDrop(ev, intent)")

if "intent === \"pin\"" not in DROP:
    _fail("the drop no longer takes its intent from the aimed zone (v664)")
if "if (wantPin && path)" not in DROP:
    _fail("the pin branch is no longer gated on the intent -- the source application "
          "decides the outcome again (B-06)")
if "_resolveInputPath()" not in DROP:
    _fail("a dropped copy no longer targets the ComfyUI input folder -- it would "
          "scatter files into whatever folder happens to be pinned")
if "encodeURIComponent(this.folder)" in DROP:
    _fail("the drop still uploads into the pinned folder (v663 target is input/)")
if "/uls/media/upload" not in DROP or "/uls/media/resolve" not in DROP:
    _fail("one of the two drop routes is gone")

HELPER = _lift_method("_dropPathOf(dt)")
if "text/plain" not in HELPER:
    _fail("_dropPathOf no longer reads the dropped path text")
if "f.path" not in HELPER:
    _fail("_dropPathOf no longer honours file.path -- the desktop client could pin "
          "the origin folder but would be forced to copy")

NOTE = _lift_method("_dropNote(text, color)")
if "_renderBatchStatus()" not in NOTE:
    _fail("the drop note never restores the status line -- it would sit there forever")
if DROP.count("this._dropNote(") < 4:
    _fail("not every drop outcome reports itself (found %d notes, expected the pin, "
          "the copy, the no-folder case and the failures)" % DROP.count("this._dropNote("))
if "input folder" not in DROP:
    _fail("the copy note no longer names the input folder as the target")

# The old unconditional 'files present -> copy' shortcut must be gone.
if "(!dt.files || !dt.files.length) &&" in DROP:
    _fail("the drop still refuses the pin branch whenever files are present (B-06)")


# ---------------------------------------------------------------------------
# DRIVEN -- the REAL _dropPathOf against four DataTransfer shapes.
# ---------------------------------------------------------------------------
BODY = """
const T = new (class {
  %s
})();

const mk = (text, files) => ({ getData: () => text, files: files || [] });
const out = {};
out.fromText  = T._dropPathOf(mk("C:\\\\GO_TRAINING\\\\FLUX_RAIN\\\\rain.png", []));
out.fromPath  = T._dropPathOf(mk("", [{ name: "clip.mp4", path: "D:\\\\media\\\\clip.mp4" }]));
out.bareName  = T._dropPathOf(mk("rain.png", [{ name: "rain.png", path: "rain.png" }]));
out.junk      = T._dropPathOf(mk("just some text", []));
out.posix     = T._dropPathOf(mk("/home/frank/media/a.wav", []));
console.log(JSON.stringify(out));
""" % (_lift_method("_dropPathOf(dt)") + "\n  " + _lift_method("_pathFromUri(c) {"))


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
if got["fromText"] != "C:\\GO_TRAINING\\FLUX_RAIN\\rain.png":
    _fail("a dropped path text is no longer resolved: %r" % (got["fromText"],))
if got["fromPath"] != "D:\\media\\clip.mp4":
    _fail("a desktop-client file.path is no longer used: %r" % (got["fromPath"],))
if got["posix"] != "/home/frank/media/a.wav":
    _fail("a POSIX path is no longer accepted: %r" % (got["posix"],))
if got["bareName"]:
    _fail("a bare file name is treated as a path -- resolve would be called with "
          "nonsense: %r" % (got["bareName"],))
if got["junk"]:
    _fail("arbitrary dropped text is treated as a path: %r" % (got["junk"],))

# MUTANT -- drop the absolute-path test on the file.path branch.
MUT = BODY.replace('return /^([A-Za-z]:[\\\\/]|\\/)/.test(c) ? c : "";', 'return c;')
if MUT == BODY:
    _fail("could not build the mutant -- the file.path guard moved")
mut = run(MUT, "mutant")
if not mut["bareName"]:
    _fail("the guard does not catch a bare file name being passed off as a path")

print("[test_v662_drop_intent] OK -- Shift decides copy vs pin, file.path is honoured, "
      "every outcome reports itself (mutation-tested)")
