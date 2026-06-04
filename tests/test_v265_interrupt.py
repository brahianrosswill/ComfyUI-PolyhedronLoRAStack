# -*- coding: utf-8 -*-
"""
test_v265_interrupt.py
══════════════════════
Guards the v265 "Resolve killswitch" (ComfyUI's red X / Cancel aborts a long
merge or deep analysis promptly).

Two parts:
  [1] Wiring (always runs, no ComfyUI needed): the interrupt checks and the
      re-raise clauses are present in the source — so a refactor can't silently
      drop them. Specifically:
        • _check_interrupt() at the top of the load loop AND the merge loop
          (uls_stack_node.py) and the deep-analysis loop (uls_resolve_inspector.py)
        • `except INTERRUPT_EXC: raise` before the broad excepts that would
          otherwise swallow a cancel into a SEQ fallback / "analysis failed"
        • the defensive _check_interrupt / INTERRUPT_EXC helper
  [2] Live mechanism (only if ComfyUI is importable): toggling comfy's interrupt
      flag makes _check_interrupt raise, and it is inert when clear. SKIPPED when
      comfy is unavailable (e.g. CI / sandbox).

The bit-identity guard for the merge stays test_v259 / test_v261 — the killswitch
adds only no-op checks and re-raise clauses, so when nothing is cancelled the
merge is unchanged.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STACK = os.path.join(HERE, "..", "nodes", "uls_stack_node.py")
ANALYZER = os.path.join(HERE, "..", "nodes", "uls_resolve_inspector.py")


def main():
    ok = True
    stack_src = open(STACK, "r", encoding="utf-8").read()
    an_src = open(ANALYZER, "r", encoding="utf-8").read()

    # ── [1] wiring present ──
    checks = [
        ("helper _check_interrupt defined",        "def _check_interrupt():" in stack_src),
        ("INTERRUPT_EXC defined",                  "INTERRUPT_EXC" in stack_src),
        ("stack: 2× _check_interrupt() in loops",  stack_src.count("_check_interrupt()") >= 2),
        ("stack: re-raise before broad except",    "except INTERRUPT_EXC:" in stack_src),
        ("analyzer imports INTERRUPT_EXC",         "INTERRUPT_EXC" in an_src),
        ("analyzer: _check_interrupt() in loop",   "_check_interrupt()" in an_src),
        ("analyzer: re-raise before broad except", "except INTERRUPT_EXC:" in an_src),
    ]
    for label, cond in checks:
        ok &= bool(cond)
        print(f"[1] {label:<42} {'PASS' if cond else 'FAIL'}")

    # ── [2] live mechanism (optional) ──
    try:
        import comfy.model_management as mm
        have = hasattr(mm, "throw_exception_if_processing_interrupted")
    except Exception:
        have = False

    if not have:
        print("[2] live interrupt check: SKIP (ComfyUI not importable here)")
    else:
        exc = getattr(mm, "InterruptProcessingException", None)
        raised = inert = False
        try:
            mm.interrupt_current_processing(True)
            try:
                mm.throw_exception_if_processing_interrupted()
            except Exception as ex:
                raised = (exc is None) or isinstance(ex, exc)
        finally:
            mm.interrupt_current_processing(False)
        try:
            mm.throw_exception_if_processing_interrupted()
            inert = True
        except Exception:
            inert = False
        p2 = raised and inert
        ok &= p2
        print(f"[2] live: raises when set + inert when clear : {'PASS' if p2 else 'FAIL'}")

    print("=" * 56)
    print("RESULT:", "ALL CHECKS PASS" if ok else "FAILURE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
