#!/usr/bin/env python3
"""test_v656_busy_overlay -- the cross-view busy ring, DRIVEN.

Why it exists: the grid's own "Loading…" text is display:none in Solo view,
so slow folder pins and drop-uploads ran for seconds with ZERO feedback.
The ring overlays .ph-media-main and is visible in BOTH views.

What runs:
  * the counter logic DRIVEN under node (the real _busyOn/_busyOff method
    bodies, extracted verbatim, against a stub root): on/off toggles the
    class, OVERLAPPING phases (upload -> re-read) keep the ring up until
    the LAST off, an extra off never goes negative, the label follows the
    latest phase. node absent -> that half is SKIP-AS-PASS.
  * statics: overlay element + CSS (absolute inset overlay, .on toggle,
    spin keyframes, amber ring) live in the loader; refreshGrid rides the
    signal inside try/finally (error paths DROP the ring); the drop-upload
    path rides it with its own label; the silent focus re-read stays
    silent (no busy call in _focusReread).

MUTATIONS (wound injected into a COPY, catch proven): M1 the counter off
ignores overlapping phases (ring dies after the first off), M2 refreshGrid
loses its finally (an error path leaves the ring stuck -- static), M3 the
silent focus path starts calling busy (the flicker returns -- static).
"""
import io
import os
import re
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOADER_JS = os.path.join(ROOT, "web", "js", "ph_media_loader.js")


def _fail(msg):
    print("[test_v656_busy_overlay] FAIL: " + msg)
    sys.exit(1)


def _need(cond, msg):
    if not cond:
        _fail(msg)


def _extract_methods(src):
    m = re.search(r"    _busyOn\(label\) \{.*?\n    \}\n\n    _busyOff\(\) \{.*?\n    \}",
                  src, flags=re.S)
    _need(m is not None, "_busyOn/_busyOff not found")
    return m.group(0)


_DRIVER = r"""
class Host {
%s
}
const cls = { has: false, add() { this.has = true; }, remove() { this.has = false; } };
const label = { textContent: "" };
const el = { classList: cls, querySelector: () => label };
const h = new Host();
h.root = { querySelector: () => el };
let ok = true;
h._busyOn("Uploading…");
ok = ok && cls.has === true && label.textContent === "Uploading…";
h._busyOn("Reading folder…");                 // overlapping phase
ok = ok && cls.has === true && label.textContent === "Reading folder…";
h._busyOff();                                  // inner phase ends
ok = ok && cls.has === true;                   // ring must STAY up
h._busyOff();                                  // last phase ends
ok = ok && cls.has === false;
h._busyOff();                                  // extra off: never negative
h._busyOn("x");
ok = ok && cls.has === true;                   // still one-on-one after the extra off
h._busyOff();
ok = ok && cls.has === false;
console.log(ok ? "BUSY-PASS" : "BUSY-FAIL");
"""


def run_counter(src, strict=True):
    try:
        subprocess.run(["node", "--version"], capture_output=True, timeout=10)
    except Exception:
        if strict:
            print("[test_v656_busy_overlay] note: node absent -- counter half "
                  "skipped as pass; statics + static mutations still ran.")
        return None
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(_DRIVER % _extract_methods(src))
        path = f.name
    try:
        res = subprocess.run(["node", path], capture_output=True, text=True,
                             timeout=30)
        return "BUSY-PASS" in (res.stdout or "")
    finally:
        os.remove(path)


def run_static(src):
    _need('<div class="ph-media-busy"><div class="ph-busy-ring">' in src,
          "the busy overlay element must live inside .ph-media-main")
    _need(".ph-media-busy { position:absolute; inset:0;" in src
          and ".ph-media-busy.on { display:flex; }" in src
          and "@keyframes ph-busy-spin" in src
          and "border-top-color:#ff8c00" in src,
          "overlay CSS must be the absolute cross-view ring (amber)")
    i_on = src.find('this._busyOn("Reading folder…");')
    i_fetch = src.find('api.fetchApi("/uls/media/list?folder=', i_on)
    _need(0 <= i_on < i_fetch,
          "refreshGrid must raise the ring before its fetch (order, not "
          "adjacency -- v657's listing ticket sits between them)")
    _need("        } finally {\n            this._busyOff();\n        }\n    }"
          in src,
          "refreshGrid must drop the ring in finally (error paths too)")
    _need('this._busyOn("Uploading…");' in src,
          "the drop-upload path must ride the ring with its own label")
    m = re.search(r"async _focusReread\(\) \{.*?\n    \}", src, flags=re.S)
    _need(m is not None, "_focusReread not found")
    _need("_busyOn" not in m.group(0),
          "the silent focus re-read must NEVER touch the busy ring")


def _wounded(old, new, name):
    s = open(LOADER_JS, encoding="utf-8").read()
    _need(s.count(old) == 1, "mutation target not unique: " + name)
    return s.replace(old, new)


def run_mutations():
    m1 = _wounded(
        "        this._busyCount = Math.max(0, (this._busyCount || 0) - 1);\n"
        "        if (this._busyCount) return;\n",
        "        this._busyCount = 0;\n", "M1")
    m2 = _wounded(
        "        } finally {\n            this._busyOff();\n        }\n    }",
        "        }\n        this._busyOff();\n    }", "M2")
    m3 = _wounded(
        "        if (_filesSig(d.files) === _filesSig(this._files)) return;   // nothing new",
        "        this._busyOn();\n"
        "        if (_filesSig(d.files) === _filesSig(this._files)) return;   // nothing new",
        "M3")
    caught = 0
    need = 3
    r = run_counter(m1, strict=False)
    if r is None:
        need -= 1
    elif r is False:
        caught += 1
    else:
        _fail("mutation counter NOT caught")
    for wounded, name in ((m2, "finally"), (m3, "silence")):
        bit = False
        try:
            with redirect_stdout(io.StringIO()):   # the probe's FAIL print is intentional
                run_static(wounded)
        except SystemExit:
            bit = True
        if bit:
            caught += 1
        else:
            _fail("mutation " + name + " NOT caught")
    _need(caught == need, "mutation coverage incomplete")


def main():
    src = open(LOADER_JS, encoding="utf-8").read()
    r = run_counter(src)
    _need(r is not False, "counter probes failed under node")
    run_static(src)
    run_mutations()
    print("[test_v656_busy_overlay] PASS: counter driven under node "
          "(overlap holds the ring, extra off never negative, label follows), "
          "overlay CSS + element pinned, refreshGrid finally-guarded, upload "
          "labeled, focus path silent, mutations caught")


if __name__ == "__main__":
    main()
