"""Guard v666 -- Save documents the loader's `video_audio` output on the
`video` input; NO new pin, pin set and save() signature untouched.

Decision (v666, Frank): the loader's `video_audio` output is plain VIDEO
(frames + muxed companion track) and already plugs into the Save node's
existing `video` input. Guarded contract:
  1. the `video` tooltip names `video_audio` (wiring discoverable on the node);
  2. optional inputs stay exactly (image, video, audio, mask), in order;
  3. save() grew no `video_audio` argument.
Torch-free: source-text checks only.
"""
import ast
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _fail(msg):
    print("FAIL: " + msg)
    sys.exit(1)


def main():
    src = open(os.path.join(ROOT, "nodes", "ph_save.py"), encoding="utf-8").read()

    m = re.search(r'"optional"\s*:\s*\{(.*?)\n\s*\}\s*,\s*\n\s*"hidden"', src, re.S)
    if not m:
        _fail("optional input block not found in ph_save.py")
    opt = m.group(1)

    vm = re.search(r'"video"\s*:\s*\("VIDEO",\s*\{(.*?)\}\)', opt, re.S)
    if not vm:
        _fail("video input not found")
    tooltip = vm.group(1)
    if "video_audio" not in tooltip:
        _fail("video tooltip must name the video_audio output")
    if not re.search(r"companion\s+track", tooltip):
        _fail("tooltip should explain the companion track rides along")

    names = re.findall(r'"(image|video|audio|mask|video_audio)"\s*:\s*\(', opt)
    if names != ["image", "video", "audio", "mask"]:
        _fail("optional inputs must stay exactly (image, video, audio, mask); got %r" % (names,))

    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == "save":
            if "video_audio" in [a.arg for a in node.args.args]:
                _fail("save() must not grow a video_audio arg")
            break
    else:
        _fail("save() not found")

    print("PASS: v666 video_audio doc contract intact, pin set unchanged")
    sys.exit(0)


if __name__ == "__main__":
    main()
