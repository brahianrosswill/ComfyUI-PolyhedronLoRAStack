"""Guard v551 -- Save: media_kind='auto' recognises a multi-frame IMAGE batch
as a clip.

BEHAVIOURAL: resolve_media_kind is imported and EXECUTED against the full
decision matrix (measure > believe), including the two contracts that keep
old behaviour reachable: the explicit media_kind='image' opt-out (a deliberate
still batch - the FBX sprite case) and the n_frames default of 1 (every
pre-v551 caller keeps its exact behaviour). Text pins hold the caller side:
ph_save.py must pass the real frame count. Script-style: exit 0 = pass.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fail(msg):
    print("[test_v551_autovideo] FAIL: " + msg)
    sys.exit(1)


def main():
    sys.path.insert(0, os.path.join(ROOT, "nodes"))
    try:
        import ph_save_util as U
    except Exception as exc:
        _fail(f"ph_save_util must stay import-light (stdlib only): {exc}")

    R = U.resolve_media_kind
    # (kind, has_image, has_video, has_audio, n_frames) -> expected
    matrix = [
        (("auto", True, False, False, 1), "image"),    # single still: SaveImage-like
        (("auto", True, False, False, 65), "video"),   # v551: many frames ARE a clip
        (("auto", True, False, False, 2), "video"),    # boundary: 2 frames = clip
        (("auto", True, False, True, 1), "video"),     # image + audio: muxed clip
        (("auto", False, True, False, 1), "video"),    # wired VIDEO wins
        (("auto", False, False, True, 1), "audio"),    # audio-only
        (("image", True, False, False, 65), "image"),  # explicit opt-out (FBX case)
        (("video", True, False, False, 1), "video"),   # explicit opt-in, single frame
    ]
    for args, want in matrix:
        got = R(*args)
        if got != want:
            _fail(f"resolve_media_kind{args} -> {got!r}, expected {want!r}")

    # Back-compat contract: a pre-v551 call (no n_frames) behaves like n=1.
    if R("auto", True, False, False) != "image":
        _fail("n_frames must default to 1 (pre-v551 callers keep their behaviour)")

    # Nothing wired must still raise.
    try:
        R("auto", False, False, False)
        _fail("auto with nothing wired must raise")
    except ValueError:
        pass

    # ---- caller side (text pins) ---------------------------------------------
    py = open(os.path.join(ROOT, "nodes", "ph_save.py"), encoding="utf-8").read()
    if "n_frames=(int(image.shape[0]) if image is not None else 1)" not in py:
        _fail("ph_save.py must pass the REAL frame count into resolve_media_kind")
    util = open(os.path.join(ROOT, "nodes", "ph_save_util.py"),
                encoding="utf-8").read()
    if "int(n_frames) > 1" not in util:
        _fail("the v551 rule (many frames ARE a clip) is gone")
    if not re.search(r"media_kind='image' always forces stills", util):
        _fail("the documented opt-out for deliberate still batches is gone")

    print("PASS: v551 auto-video -- decision matrix measured in-process, "
          "opt-out + back-compat contracts hold")
    sys.exit(0)


if __name__ == "__main__":
    main()
