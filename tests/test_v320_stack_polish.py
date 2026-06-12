"""
v320 (public) — Stack/Engine polish ported from internal v318/v319:

Layer 1 (sandbox-safe): the token counter's report logic. We import the node
module guarded (it pulls torch/folder_paths only at class scope, but the
report builder is a method, so we test via a light shim that calls count()).
Since uls_stack_node imports heavy deps at module top, we importorskip and run
these live; the pure-logic assertions below still document the contract.

Layer 2: the group-order orphan reclaim is frontend (uls_node.js) — covered by
a JS smoke assertion in the manual test plan, not pytest. Documented here so
the intent is captured next to the Python change.

Run: python -m pytest tests/test_v318_token_warn.py -v
"""

import os
import re
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _read(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as fh:
        return fh.read()


# ── Layer 1: static contract checks on the report builder source ──────────────
# These run with zero heavy deps — they assert the *source* now encodes the
# sharpened behavior, which is what the roadmap items asked for.

def test_over_limit_marker_present_in_source():
    src = _read("nodes/uls_stack_node.py")
    assert "TOKEN LIMIT EXCEEDED" in src, "over-limit marker missing from report"
    # marker must be gated on over_limit, not always printed
    idx = src.index("TOKEN LIMIT EXCEEDED")
    preceding = src[max(0, idx - 400):idx]
    assert "if over_limit:" in preceding, "marker is not gated behind over_limit"


def test_token_counter_returns_ui_payload():
    src = _read("nodes/uls_stack_node.py")
    assert '"pls_tokens"' in src, "UI payload for the toast hook is missing"
    assert '"over_limit"' in src and '"near_limit"' in src, "UI payload fields incomplete"


def test_token_toast_extension_present():
    js = _read("web/js/uls_token_toast.js")
    assert "ULSTokenCounter" in js, "toast hook not bound to the counter node"
    assert "extensionManager?.toast" in js, "native toast API not used"
    assert "onExecuted" in js, "toast must fire from onExecuted"


def test_warn_hint_uses_threshold_not_hardcoded_70():
    src = _read("nodes/uls_stack_node.py")
    # the misleading hard-coded "~70% of limit" line must be gone
    assert "above ~70% of limit" not in src, "stale hard-coded 70% hint still present"
    # and the near-limit hint must reference the configurable threshold
    assert "warn threshold" in src
    assert "int(round(warn_threshold * 100))" in src, \
        "near-limit hint should derive percent from warn_threshold"


def test_over_limit_hint_is_honest_about_truncate_or_crash():
    src = _read("nodes/uls_stack_node.py")
    # must mention BOTH possible outcomes, not only the crash
    seg = src[src.index("over_by ="):src.index("Approaching limit")]
    assert "truncat" in seg.lower(), "over-limit hint should mention truncation"
    assert "crash" in seg.lower(), "over-limit hint should mention the crash path"


def test_orphan_reclaim_present_in_frontend():
    js = _read("web/js/uls_node.js")
    # the silent-reject path must be replaced by orphan-aware logic
    assert "otherIsOrphan" in js, "orphan detection missing in order assignment"
    assert "liveGroups" in js, "live-group set for orphan detection missing"
    # a real confirm must guard the visible-collision case (styled dialog now)
    assert "showConfirmDialog(" in js, "visible-collision confirm missing"


# ── Layer 2: live behavioral test (only runs inside ComfyUI) ──────────────────

def test_token_counter_over_limit_flag_live():
    sys.path.insert(0, os.path.join(ROOT, "nodes"))
    mod = pytest.importorskip(
        "uls_stack_node",
        reason="ComfyUI runtime (torch/folder_paths) not importable in sandbox.")
    # find the counter class by its RETURN_NAMES signature
    counter_cls = None
    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, type) and getattr(obj, "RETURN_NAMES", None) == \
                ("report", "positive_tokens", "negative_tokens", "over_limit", "trigger_tokens"):
            counter_cls = obj
            break
    assert counter_cls is not None, "token counter class not found"
    inst = counter_cls()
    long_prompt = "word " * 600  # ~600 words >> 512 tokens
    out = inst.count(512, 0.90, long_prompt, "")
    # count() now returns {"ui":..., "result": (...)} (UI channel for the toast)
    res = out["result"] if isinstance(out, dict) else out
    report, pos, neg, over, trig = res
    assert over is True
    assert out["ui"]["pls_tokens"][0]["over_limit"] is True
    assert "TOKEN LIMIT EXCEEDED" in report
    # under-limit must NOT show the marker or flag
    out2 = inst.count(512, 0.90, "a small prompt", "")
    res2 = out2["result"] if isinstance(out2, dict) else out2
    assert res2[3] is False
    assert out2["ui"]["pls_tokens"][0]["over_limit"] is False
    assert "TOKEN LIMIT EXCEEDED" not in res2[0]


def test_styled_confirm_dialog_present():
    js = _read("web/js/uls_node.js")
    assert "showConfirmDialog" in js, "styled confirm helper missing"
    assert "window.confirm(" not in js, "bare window.confirm should be gone"
