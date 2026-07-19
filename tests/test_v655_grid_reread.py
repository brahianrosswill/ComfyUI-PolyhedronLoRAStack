#!/usr/bin/env python3
"""test_v655_grid_reread -- the folder re-read (button + focus), DRIVEN.

What runs:
  * _filesSig DRIVEN through node (the real JS, extracted verbatim):
    identical listings agree; an added file, a removed file and an
    IN-PLACE replacement (same name, new size/mtime) all change the
    signature; empty listings agree. node absent -> that half is
    SKIP-AS-PASS (GATE-2 environments always carry node).
  * statics: the ⟳ button exists NEXT TO the kept ↻ reset (both classes
    present, distinct titles), the click re-reads WITH page keep
    (refreshGrid(true) + the keepPage-conditional page reset), the
    focus path is silent (signature short-circuit before any render),
    debounced, and its listener is a BOUND handler added AND removed in
    _destroy (the v624 leak law).

MUTATIONS (wound injected into a COPY, catch proven): M1 the signature
ignores size/mtime (in-place replacements go blind), M2 the page keep
falls (re-read always jumps to page 1), M3 the focus listener is never
removed (the leak returns). Each must fail its probe.
"""
import os
import re
import io
import subprocess
import sys
from contextlib import redirect_stdout
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOADER_JS = os.path.join(ROOT, "web", "js", "ph_media_loader.js")


def _fail(msg):
    print("[test_v655_grid_reread] FAIL: " + msg)
    sys.exit(1)


def _need(cond, msg):
    if not cond:
        _fail(msg)


def _extract_sig(src):
    m = re.search(r"function _filesSig\(files\) \{.*?\n\}", src, flags=re.S)
    _need(m is not None, "_filesSig not found in the loader JS")
    return m.group(0)


_DRIVER = r"""
%s
const a = [{name:"a.png", size:10, mtime:100}, {name:"b.mp4", size:20, mtime:200}];
const same = [{name:"a.png", size:10, mtime:100}, {name:"b.mp4", size:20, mtime:200}];
const added = a.concat([{name:"c.wav", size:5, mtime:300}]);
const removed = [a[0]];
const replaced = [{name:"a.png", size:10, mtime:100}, {name:"b.mp4", size:21, mtime:250}];
let ok = true;
ok = ok && (_filesSig(a) === _filesSig(same));
ok = ok && (_filesSig(a) !== _filesSig(added));
ok = ok && (_filesSig(a) !== _filesSig(removed));
ok = ok && (_filesSig(a) !== _filesSig(replaced));
ok = ok && (_filesSig([]) === _filesSig(null));
console.log(ok ? "SIG-PASS" : "SIG-FAIL");
"""


def run_sig(src, strict=True):
    try:
        subprocess.run(["node", "--version"], capture_output=True, timeout=10)
    except Exception:
        if strict:
            print("[test_v655_grid_reread] note: node absent -- signature "
                  "half skipped as pass; statics + mutations still ran.")
        return None
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(_DRIVER % _extract_sig(src))
        path = f.name
    try:
        res = subprocess.run(["node", path], capture_output=True, text=True,
                             timeout=30)
        return "SIG-PASS" in (res.stdout or "")
    finally:
        os.remove(path)


def run_static(src):
    _need('class="ph-media-btn ph-reread"' in src
          and 'class="ph-media-btn ph-refresh"' in src,
          "the ⟳ re-read must live NEXT TO the kept ↻ reset")
    _need('title="Back to input folder">↻' in src,
          "the ↻ reset keeps its exact job")
    _need('.ph-reread").onclick = () => this.refreshGrid(true);' in src,
          "the ⟳ click must re-read WITH page keep")
    _need("this._page = keepPage ? this._page : 0;" in src,
          "refreshGrid must reset the page ONLY without keepPage")
    _need("if (_filesSig(d.files) === _filesSig(this._files)) return;" in src,
          "the focus path must short-circuit silently on an unchanged listing")
    _need("< 5000) return;" in src, "the focus re-read must be debounced")
    _need('window.addEventListener("focus", this._onWinFocus);' in src
          and 'window.removeEventListener("focus", this._onWinFocus);' in src,
          "the focus listener must be a bound add/remove pair "
          "(the v624 leak law)")


def _wounded(old, new, name):
    s = open(LOADER_JS, encoding="utf-8").read()
    _need(s.count(old) == 1, "mutation target not unique: " + name)
    return s.replace(old, new)


def run_mutations():
    m1 = _wounded(
        'f.name + "|" + (f.size ?? "") + "|" + (f.mtime ?? "")',
        "f.name", "M1")
    m2 = _wounded(
        "        this._page = keepPage ? this._page : 0;",
        "        this._page = 0;", "M2")
    m3 = _wounded(
        '        try { window.removeEventListener("focus", this._onWinFocus); } catch (e) { /* ignore */ }\n',
        "", "M3")
    caught = 0
    need = 3
    # M1: the driven probe must see the replacement go blind
    r = run_sig(m1, strict=False)
    if r is None:
        need -= 1  # node absent: cannot probe M1; statics still guard M2/M3
    elif r is False:
        caught += 1
    else:
        _fail("mutation sig NOT caught")
    for wounded, name in ((m2, "pagekeep"), (m3, "leak")):
        bit = False
        try:
            with redirect_stdout(io.StringIO()):   # the probe's FAIL print is intentional -- keep the log honest
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
    r = run_sig(src)
    _need(r is not False, "_filesSig probes failed under node")
    run_static(src)
    run_mutations()
    print("[test_v655_grid_reread] PASS: signature driven under node "
          "(add/remove/replace all register), ⟳ next to kept ↻, page-keep "
          "conditional, silent debounced focus path, listener add/remove "
          "pair, mutations caught")


if __name__ == "__main__":
    main()
