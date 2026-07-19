"""Guard v536 -- Bug-B evidence set complete; reverted v529 no-op absent."""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _fail(msg):
    print("FAIL: " + msg)
    sys.exit(1)


def main():
    if "[PLS v536 DIAG]" not in open(os.path.join(ROOT, "nodes", "ph_media_loader.py"), encoding="utf-8").read():
        _fail("backend arrival diagnostic missing")
    jsrc = open(os.path.join(ROOT, "web", "js", "ph_media_loader.js"), encoding="utf-8").read()
    if "[PLS v536 DIAG] ph_media_loader.js" not in jsrc:
        _fail("ph_media_loader.js banner missing")
    if "MediaLoader restore(): batch_config=" not in jsrc:
        _fail("restore() dump missing")
    if "[PLS v536 DIAG] ph_save.js" not in open(os.path.join(ROOT, "web", "js", "ph_save.js"), encoding="utf-8").read():
        _fail("ph_save.js banner missing")
    m = re.search(r"hideWidget\s*=\s*\(w\)\s*=>\s*\{.*?\n\s*\};", jsrc, re.S)
    if m and "serializeValue" in m.group(0):
        _fail("reverted v529 no-op is back")
    print("PASS: v536 diagnostic set complete, reverted no-op absent")
    sys.exit(0)


if __name__ == "__main__":
    main()
