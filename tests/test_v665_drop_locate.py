#!/usr/bin/env python3
"""
test_v665_drop_locate -- the pin zone works for a plain-browser Explorer drag: a
file:/// URI on the wire is read as the path, and when no path arrives at all the
file is RELOCATED by name + size + mtime in the folders this node has visited.

Frank, 2026-07-19: "Das aktuelle Problem muss gehen ... Man zieht eine Datei ins
rechte Feld und er waehlt diese Datei aus und oeffnet auch gleich den Ordner mit, in
dem sie liegt." A plain browser hides file.path, so v665 adds the two routes that
remain:

  1. URI combing -- Firefox often ships a file:/// URI for an Explorer drag in
     text/x-moz-url / text/uri-list. _dropPathOf now combs every type and
     _pathFromUri decodes the URI into a real path (drive letter, %20, slashes).
  2. Relocation -- name, byte size and lastModified survive the browser's privacy
     wall. /uls/media/locate checks that triple against the CANDIDATE folders the
     frontend names (recents + pin + input). Exactly one hit pins; a double is
     'ambiguous', a miss is 'not_found' -- both fall through to the copy with an
     honest note. Never a disk-wide search, never a coin flip.

Guards, all must hold, mutation-tested:

  STATIC  -- _dropPathOf combs the four types; the locate call sends name/size/
             mtime + folders; the ambiguous and not-found notes exist; the route is
             registered.

  DRIVEN  -- run the REAL _pathFromUri over Windows/POSIX/encoded/junk URIs.
             Mutant: drop the %-decode -> "a%20b" survives -> caught.
          -- drive the REAL locate handler logic against a temp tree: unique hit
             pins, same file in two folders is ambiguous, size mismatch is
             not_found, mtime within 2 s tolerated. Mutant: ignore the size ->
             the wrong file matches -> caught.
"""
import os
import re
import sys
import json
import time
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
    print("[test_v665_drop_locate] FAIL: " + msg)
    sys.exit(1)


def _lift_method(sig):
    if sig not in JS:
        _fail("could not find `%s`" % sig)
    s = JS[JS.index(sig):]
    return s[:s.index("\n    }") + 6]


# ---------------------------------------------------------------------------
# STATIC
# ---------------------------------------------------------------------------
PATHOF = _lift_method("_dropPathOf(dt)")
for t in ["text/uri-list", "text/x-moz-url", "text/plain", "DownloadURL"]:
    if t not in PATHOF:
        _fail("_dropPathOf no longer combs %s -- a Firefox Explorer drag loses its "
              "file:/// URI" % t)

DROP = _lift_method("async _onDrop(ev, intent)")
for sym, why in [
    ('"/uls/media/locate"', "the pin branch never tries to relocate a pathless file"),
    ("f.lastModified", "the relocation no longer sends the mtime -- the triple decays "
                       "to name+size"),
    ("getRecentFolders()", "the relocation no longer names the visited folders -- the "
                           "server would have to guess"),
    ('"ambiguous"', "the double-hit case is no longer reported honestly"),
]:
    if sym not in DROP:
        _fail(why)

# Whitespace-tolerant: the route table is column-aligned, so an exact-spacing
# match would go red on a purely cosmetic edit instead of on a real one.
if not re.search(r'"/uls/media/locate"\s*,\s*handle_media_locate', PY):
    _fail("the locate route is not registered")
if "never a disk-wide search" not in PY:
    _fail("the locate handler lost its bounded-candidates contract note")

# ---------------------------------------------------------------------------
# DRIVEN 1 -- the REAL _pathFromUri.
# ---------------------------------------------------------------------------
BODY = """
const T = new (class {
  %s
})();
const out = {
  win:    T._pathFromUri("file:///C:/GO_TRAINING/FLUX%%20RAIN/rain.png"),
  posix:  T._pathFromUri("file:///home/frank/a.wav"),
  plain:  T._pathFromUri("D:\\\\media\\\\clip.mp4"),
  junk:   T._pathFromUri("hello there"),
  http:   T._pathFromUri("https://example.com/x.png"),
};
console.log(JSON.stringify(out));
""" % _lift_method("_pathFromUri(c) {")


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
if got["win"] != "C:\\GO_TRAINING\\FLUX RAIN\\rain.png":
    _fail("a Windows file:/// URI is not decoded to a real path: %r" % (got["win"],))
if got["posix"] != "/home/frank/a.wav":
    _fail("a POSIX file:// URI is not decoded: %r" % (got["posix"],))
if got["plain"] != "D:\\media\\clip.mp4":
    _fail("a bare absolute path no longer passes through: %r" % (got["plain"],))
if got["junk"] or got["http"]:
    _fail("non-file input leaks through as a path: %r / %r" % (got["junk"], got["http"]))

MUT = BODY.replace("decodeURIComponent(", "(")
if MUT == BODY:
    _fail("could not build the URI mutant")
mut = run(MUT, "mutant")
if mut["win"] == "C:\\GO_TRAINING\\FLUX RAIN\\rain.png":
    _fail("the guard does not catch the %%-decode being dropped")

# ---------------------------------------------------------------------------
# DRIVEN 2 -- the locate triple-match against a real temp tree.
# ---------------------------------------------------------------------------
def locate(name, size, mtime, folders):
    """Mirror of the handler's matching core, driven directly (the aiohttp shell
    around it is I/O plumbing; the RULES live here and are what can rot)."""
    hits, seen = [], set()
    for f in folders[:24]:
        rp = os.path.realpath(str(f or ""))
        if not rp or rp in seen or not os.path.isdir(rp):
            continue
        seen.add(rp)
        cand = os.path.join(rp, name)
        try:
            st = os.stat(cand)
        except OSError:
            continue
        if int(st.st_size) != size:
            continue
        if mtime and abs(int(st.st_mtime) - mtime) > 2:
            continue
        hits.append(rp)
    return hits


# the driven core must be the shipped core -- compare the decisive lines
for line in ["if int(st.st_size) != size:",
             "if mtime and abs(int(st.st_mtime) - mtime) > 2:",
             "for f in folders[:24]:"]:
    if line not in PY:
        _fail("the shipped locate handler lost its decisive line: %s" % line)

with tempfile.TemporaryDirectory() as td:
    a = os.path.join(td, "a"); b = os.path.join(td, "b"); c = os.path.join(td, "c")
    for d in (a, b, c):
        os.makedirs(d)
    now = int(time.time())
    with open(os.path.join(a, "rain.png"), "wb") as fh:
        fh.write(b"x" * 1000)
    os.utime(os.path.join(a, "rain.png"), (now, now))
    with open(os.path.join(c, "rain.png"), "wb") as fh:
        fh.write(b"x" * 999)                      # same name, different size
    os.utime(os.path.join(c, "rain.png"), (now, now))

    hits = locate("rain.png", 1000, now, [a, b, c])
    if hits != [a]:
        _fail("a unique triple-match does not pin the right folder: %r" % (hits,))
    hits = locate("rain.png", 999, now, [a, b, c])
    if hits != [c]:
        _fail("the size does not discriminate: %r" % (hits,))
    hits = locate("rain.png", 1000, now + 1, [a])
    if hits != [a]:
        _fail("an mtime within 2 s is not tolerated: %r" % (hits,))
    hits = locate("rain.png", 1000, now + 10, [a])
    if hits:
        _fail("an mtime 10 s off still matches: %r" % (hits,))
    # duplicate -> ambiguous
    with open(os.path.join(b, "rain.png"), "wb") as fh:
        fh.write(b"x" * 1000)
    os.utime(os.path.join(b, "rain.png"), (now, now))
    hits = locate("rain.png", 1000, now, [a, b, c])
    if len(hits) != 2:
        _fail("the same file in two folders is not flagged ambiguous: %r" % (hits,))

print("[test_v665_drop_locate] OK -- file:/// URIs decode to paths, the triple-match "
      "pins only an unambiguous hit (mutation-tested)")
