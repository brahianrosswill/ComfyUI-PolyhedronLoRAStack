"""Guard v543 -- two rubrics named by PURPOSE; labels are functions, tooltips explain."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def _fail(m): print("FAIL: " + m); sys.exit(1)

def main():
    ml = open(os.path.join(ROOT, "web", "js", "ph_media_loader.js"), encoding="utf-8").read()
    # purpose vocabulary, everywhere (dialog + node status + button)
    for n in ("Video frames", "Separate files"):
        if ml.count(n) < 3: _fail(f"vocabulary {n!r} not used consistently (dialog/status/button)")
    for old in ("All at once", "One per run", "Batch ON (Frames)", "Batch ON (Processing)"):
        if old in ml: _fail(f"old mechanic vocabulary still present: {old!r}")
    # one explanatory line per mode -- the purpose split has to be said once
    if "ph-mode-info" not in ml or "frames of ONE video" not in ml:
        _fail("per-mode info line missing")
    # rubric cards carry the mode class (whole card hides, titled + colour coded)
    if 'ph-rub ph-mrow-frames' not in ml or 'ph-rub ph-mrow-proc' not in ml:
        _fail("rubric cards missing")
    if 'ph-rub-hd' not in ml: _fail("rubric titles missing")
    # labels are FUNCTIONS; explanations live in tooltips
    for lbl in (">Number<", ">A\\u2013Z<", ">Date modified<", ">Date created<",
                ">Stop<", ">Stretch<", ">Letterbox<", ">Crop center<"):
        if lbl not in ml: _fail(f"function label missing: {lbl}")
    for explain in ("Numbered order", "img2 before img10<", "Stop and tell me",
                    "Add bars (letterbox)", "Strict A"):
        if explain in ml: _fail(f"explanation still inside a label: {explain!r}")
    if 'title="Stacks by the numbers' not in ml: _fail("Number tooltip missing")
    if 'title="Refuses to run if the frames differ' not in ml: _fail("Stop tooltip missing")
    # wire values byte-identical (label != value)
    for v in ('"name (natural)"', '"name (literal)"', '"mtime (oldest first)"', '"created"',
              '"none (strict)"', '"resize to first"', '"pad to first"', '"center crop to first"'):
        if v not in ml: _fail(f"wire value {v} changed -- parity/backward compat broken")
    print("PASS: v543 rubrics -- purpose vocabulary, function labels, tooltips, wire values intact")
    sys.exit(0)

if __name__ == "__main__":
    main()
