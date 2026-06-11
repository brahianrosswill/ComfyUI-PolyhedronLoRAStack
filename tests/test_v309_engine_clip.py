"""
test_v309_engine_clip.py — source-wiring checks for the Engine wClip fix.

v302 brought the per-row CLIP strength UI + backend to both nodes, but the
two Engine serialization mappers dropped `wClip` (only the Stack mappers
carried it). v309 closes that gap. This suite verifies the wiring at the
source level — sandbox-safe, no torch/comfy required.

Run directly:  python3 tests/test_v309_engine_clip.py
"""

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

JS_PATH = os.path.join(ROOT, "web", "js", "uls_node.js")
PY_PATH = os.path.join(ROOT, "nodes", "uls_stack_node.py")

passed = 0
failed = 0


def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}")


def extract_block(src, anchor, span=2000):
    """Return `span` chars of source starting at `anchor` (or '' if absent)."""
    i = src.find(anchor)
    return src[i:i + span] if i >= 0 else ""


print("════════════════════════════════════════════════════════")
print(" test_v309_engine_clip — Engine wClip wiring")
print("════════════════════════════════════════════════════════")

js = open(JS_PATH, encoding="utf-8").read()
py = open(PY_PATH, encoding="utf-8").read()

# ── 1. Engine onSerialize mapper (o._engine) carries wClip ─────────────
print("\n[1] Engine workflow persistence (onSerialize → o._engine)")
blk = extract_block(js, "o._engine = JSON.stringify({", span=400)
check("o._engine mapper exists", bool(blk))
check("o._engine mapper includes wClip", "wClip: r.wClip" in blk)
check("o._engine mapper still includes weight", "weight: r.weight" in blk)

# ── 2. Engine _ulsSync mapper (engine_config) carries wClip ────────────
print("\n[2] Engine backend wire (_ulsSync → engine_config)")
# The engine_config sync block is the JSON.stringify that also sets
# dare_variant (snake_case) — unique to the Engine widget sync.
m = re.search(
    r"w\.value = JSON\.stringify\(\{.{0,600}?dare_variant", js, re.S)
blk = m.group(0) if m else ""
check("engine_config sync mapper exists", bool(blk))
check("engine_config mapper includes wClip", "wClip: r.wClip" in blk)
check("engine_config mapper still includes weight", "weight: r.weight" in blk)
check("engine_config mapper keeps mode field", '"SEQ"' in blk or "mode:" in blk)

# ── 3. Engine onConfigure restores arbitrary row fields (spread) ───────
print("\n[3] Engine restore path (onConfigure spread)")
check("onConfigure spreads saved row over newEngineRow",
      "...newEngineRow(), ...r" in js.replace(" ", "").replace("{...newEngineRow(),...r}", "...newEngineRow(), ...r")
      or re.search(r"\{\s*\.\.\.newEngineRow\(\),\s*\.\.\.r\s*\}", js) is not None)

# ── 4. Engine UI mechanic present (set/draw paths) ─────────────────────
print("\n[4] Engine UI mechanic (canvas, shared v302 pattern)")
# The Engine row cluster uses row.weight as fallback base — distinct from
# the Stack cluster which uses row.wLow.
check("Engine two-line cell draws blue c-line",
      re.search(r"row\.wClip\s*===?\s*\"number\".{0,200}row\.weight", js, re.S)
      is not None or "row.wClip !== row.weight" in js)
check("Engine decouple/recouple via popup (delete on equal weight)",
      re.search(r"if \(v === \(row\.weight \|\| 0\)\) delete row\.wClip; else row\.wClip = v",
                js) is not None)
check("Engine Shift-step writes wClip from weight base",
      re.search(r"\(typeof row\.wClip === \"number\"\) \? row\.wClip : \(row\.weight \|\| 0\)",
                js) is not None)

# ── 5. Stack mappers (v302) untouched ──────────────────────────────────
print("\n[5] Stack mappers (v302) still intact")
check("Stack o._uls mapper includes wClip",
      re.search(r"o\._uls = JSON\.stringify\(\{.{0,400}?wClip: r\.wClip", js, re.S)
      is not None)
check("Stack lora_config sync includes wClip",
      js.count("wClip: r.wClip") >= 4)  # 2 Stack (v302) + 2 Engine (v309)

# ── 6. Backend consumer side (existed since v302, must stay) ───────────
print("\n[6] Backend consumer (uls_stack_node.py)")
check("_row_clip_weight helper exists", "def _row_clip_weight(" in py)
check("helper falls back to model weight",
      re.search(r"def _row_clip_weight.*?return fallback", py, re.S) is not None)

# Engine apply: active_clip built via _row_clip_weight and passed onward.
eng = extract_block(py, "class ULSAccelerator", span=8000)
check("Engine apply builds active_clip via _row_clip_weight",
      "_row_clip_weight(row, w)" in eng)
check("Engine apply passes clip_weights to apply_lora_set",
      "clip_weights=active_clip" in eng)

# ── 7. Round-trip simulation of the two mappers ────────────────────────
print("\n[7] Mapper round-trip semantics (simulated)")
# Mirror the JS mapper semantics in Python: undefined (=absent) wClip must
# fall out of the JSON; present wClip must survive.
import json

def js_mapper(row):
    out = {"enabled": row.get("enabled", True),
           "name": row.get("name", "None"),
           "weight": row.get("weight", 1.0)}
    if "wClip" in row:          # JSON.stringify drops undefined
        out["wClip"] = row["wClip"]
    return out

plain = js_mapper({"enabled": True, "name": "a.safetensors", "weight": 0.5})
coupled = json.loads(json.dumps(plain))
check("row without wClip serializes without wClip key", "wClip" not in coupled)

dec = js_mapper({"enabled": True, "name": "a.safetensors",
                 "weight": 0.5, "wClip": 0.8})
dec_rt = json.loads(json.dumps(dec))
check("row with wClip survives round-trip", dec_rt.get("wClip") == 0.8)

# Backend fallback semantics on the round-tripped rows:
def row_clip_weight(row, fallback):
    if "wClip" in row:
        try:
            v = float(row["wClip"])
            if v == v:  # not NaN
                return v
        except (TypeError, ValueError):
            pass
    return fallback

check("backend: plain row → CLIP = model weight",
      row_clip_weight(coupled, 0.5) == 0.5)
check("backend: decoupled row → CLIP = wClip",
      row_clip_weight(dec_rt, 0.5) == 0.8)

# ── Result ─────────────────────────────────────────────────────────────
print("\n════════════════════════════════════════════════════════")
total = passed + failed
if failed == 0:
    print(f"RESULT: ALL CHECKS PASS ({passed}/{total})")
    sys.exit(0)
else:
    print(f"RESULT: {failed} FAILED ({passed}/{total} passed)")
    sys.exit(1)
