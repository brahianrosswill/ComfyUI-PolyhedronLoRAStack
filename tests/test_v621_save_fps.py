#!/usr/bin/env python3
"""
test_v621_save_fps -- the Save node exposes a resolved `fps` output, appended.

v621 adds two things to ULSSave (nodes/ph_save.py), both additive:
  1. image_quality default 95 -> 100 (lossless webp / max jpeg by default).
  2. a second output `fps` (Slot 1, APPENDED after `path`) carrying the frame rate the node
     resolves via _fps_of(video, frame_rate): a wired VIDEO's own rate, else the frame_rate
     widget (>0), else 24. save() resolves it once and appends float(fps) to every return
     path's result tuple, leaving the ui payload untouched.

The append-only output law (HANDOVER 4) is the load-bearing invariant here: `path` MUST stay
Slot 0 and `fps` MUST be Slot 1, or every saved graph silently re-indexes its output links.

Two guards, both must hold, each mutation-tested (inject the wound, prove the catch):

  STATIC -- RETURN_TYPES == ("STRING","FLOAT"), RETURN_NAMES == ("path","fps") in THAT order
            (path Slot 0, fps Slot 1); image_quality default == 100; save() calls
            _fps_of(video, frame_rate) and appends it on the RIGHT of the result tuple
            (a prepend is rejected).

  DRIVEN -- _fps_of is lifted from source and executed: no wired video -> the frame_rate widget
            (50 -> 50.0; 0 -> 24.0 fallback; None -> 24.0), a wired video's own rate wins
            (-> 30.0). Mutating the fallback or the video-preference makes these checks fail.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = open(os.path.join(ROOT, "nodes", "ph_save.py"), encoding="utf-8").read()


def _fail(msg):
    print("[test_v621_save_fps] FAIL: " + msg)
    sys.exit(1)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _lift_pyfunc(src, name):
    """Lift a top-level def by name: the def line plus every indented/blank line that
    follows, stopping at the next column-0 line. No AST -- pure text, like the JS _lift."""
    out, inside = [], False
    for ln in src.splitlines(keepends=True):
        if not inside:
            if ln.startswith("def " + name + "("):
                inside = True
                out.append(ln)
            continue
        if ln.strip() == "" or ln[:1] in (" ", "\t"):
            out.append(ln)
        else:
            break
    return "".join(out)


def _tuple_after(src, key):
    """Return the list of string literals in the (...) assigned to `key` = (...)."""
    m = re.search(re.escape(key) + r"\s*=\s*\(([^)]*)\)", src)
    if not m:
        return None
    return re.findall(r'"([^"]*)"', m.group(1))


def _lift_method(src, name):
    """Lift a method `def name(` at ANY indentation, up to the next sibling def/class/decorator
    at the same-or-lower indent. Methods are indented, so _lift_pyfunc (column-0 only) misses
    them -- this is the class-body equivalent."""
    out, inside, indent = [], False, None
    for ln in src.splitlines(keepends=True):
        if not inside:
            m = re.match(r"^(\s*)def " + re.escape(name) + r"\(", ln)
            if m:
                inside, indent = True, len(m.group(1))
                out.append(ln)
            continue
        if ln.strip() == "":
            out.append(ln)
            continue
        cur = len(ln) - len(ln.lstrip())
        if cur <= indent and re.match(r"^\s*(def |class |@)", ln):
            break
        out.append(ln)
    return "".join(out)


def _save_body(src):
    """The body of def save(self, ...) (an indented method) up to the next sibling def."""
    return _lift_method(src, "save")


def _make_fps_of(fps_src, has_video):
    ns = {"_HAS_VIDEO_API": has_video}
    exec(fps_src, ns)  # noqa: S102 -- lifting our own source is the doctrine (run it, don't read it)
    return ns["_fps_of"]


class _Comp:
    frame_rate = 30.0


class _Vid:
    def get_components(self):
        return _Comp()


# ---------------------------------------------------------------------------
# STATIC checks -- each returns True on the healthy source, raises/False on a wound
# ---------------------------------------------------------------------------
def check_return_order(src):
    types = _tuple_after(src, "RETURN_TYPES")
    names = _tuple_after(src, "RETURN_NAMES")
    if types != ["STRING", "FLOAT"]:
        return False
    if names != ["path", "fps"]:
        return False
    # path Slot 0, fps Slot 1 -- append-only
    return names[0] == "path" and names[1] == "fps" and len(names) == 2


def check_default_quality(src):
    m = re.search(r'"image_quality":\s*\("INT",\s*\{\s*"default":\s*(\d+)', src)
    return bool(m) and int(m.group(1)) == 100


def check_append_not_prepend(src):
    body = _save_body(src)
    if "_fps_of(video, frame_rate)" not in body:
        return False
    if "+ (float(fps),)" not in body:      # fps appended on the RIGHT
        return False
    if "(float(fps),) +" in body:          # a prepend is the wound
        return False
    return True


def check_fps_of_driven(src):
    fps_src = _lift_pyfunc(src, "_fps_of")
    if "def _fps_of(" not in fps_src:
        return False
    f = _make_fps_of(fps_src, has_video=False)   # no video API -> widget path
    if not (f(None, 24.0) == 24.0 and f(None, 50.0) == 50.0):
        return False
    if not (f(None, 0.0) == 24.0 and f(None, None) == 24.0):   # fallback to 24
        return False
    g = _make_fps_of(fps_src, has_video=True)     # video API present
    if g(_Vid(), 24.0) != 30.0:                   # a wired video's own rate wins
        return False
    if g(None, 24.0) != 24.0:                      # no video -> widget
        return False
    return True


CHECKS = [
    ("return order (path Slot 0, fps Slot 1, append-only)", check_return_order),
    ("image_quality default == 100", check_default_quality),
    ("save() appends _fps_of on the right (no prepend)", check_append_not_prepend),
    ("_fps_of resolution (DRIVEN, executed)", check_fps_of_driven),
]

# ---------------------------------------------------------------------------
# 1) the real source must pass every check
# ---------------------------------------------------------------------------
for label, fn in CHECKS:
    try:
        ok = fn(SRC)
    except Exception as exc:  # a check that throws on healthy source is itself broken
        _fail("check '%s' raised on healthy source: %r" % (label, exc))
    if not ok:
        _fail("healthy source failed check: %s" % label)

# ---------------------------------------------------------------------------
# 2) mutation harness -- inject each wound, prove the matching check catches it.
#    A guard that never fails is indistinguishable from one that cannot.
# ---------------------------------------------------------------------------
MUTATIONS = [
    # (label, mutated_source, check_that_must_now_fail)
    ("M1 swap output order -> (fps, path)",
     SRC.replace('RETURN_TYPES = ("STRING", "FLOAT")', 'RETURN_TYPES = ("FLOAT", "STRING")')
        .replace('RETURN_NAMES = ("path", "fps")', 'RETURN_NAMES = ("fps", "path")'),
     check_return_order),
    ("M2 revert default to 95",
     SRC.replace('"default": 100, "min": 1, "max": 100', '"default": 95, "min": 1, "max": 100'),
     check_default_quality),
    ("M3 prepend fps instead of append",
     SRC.replace("res + (float(fps),)", "(float(fps),) + res"),
     check_append_not_prepend),
    ("M4 break _fps_of fallback (24 -> 0)",
     SRC.replace("return fr if fr > 0 else 24.0", "return fr if fr > 0 else 0.0"),
     check_fps_of_driven),
    ("M5 break _fps_of video preference (> -> <)",
     SRC.replace(
         "            fr = float(video.get_components().frame_rate)\n            if fr > 0:",
         "            fr = float(video.get_components().frame_rate)\n            if fr < 0:"),
     check_fps_of_driven),
]

for label, mutant, fn in MUTATIONS:
    if mutant == SRC:
        _fail("mutation '%s' did not change the source -- the harness is out of sync with the "
              "code (the target string was not found)" % label)
    try:
        still_ok = fn(mutant)
    except Exception:
        still_ok = False  # a wound that makes the check throw is a catch
    if still_ok:
        _fail("mutation NOT caught: %s (check '%s' still passed on the wounded source)"
              % (label, fn.__name__))

print("[test_v621_save_fps] PASS -- fps output append-only, default 100, _fps_of driven; "
      "5/5 mutations caught")
sys.exit(0)
