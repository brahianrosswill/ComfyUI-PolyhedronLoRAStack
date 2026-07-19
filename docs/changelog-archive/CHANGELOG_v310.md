# v310 — Engine: "Weight / CLIP Strength" header (Stack v308 mirror)

## Feature

The Engine node now shows the same two-tone column header as the Stack:
amber **Weight** / blue **CLIP Strength** + drawn 🛈 info icon, with the
hover tooltip explaining the Shift interaction (Shift+Click to decouple,
Shift+◀▶ to step, re-enter the model weight to re-link).

Implementation is an exact mirror of the Stack v305/v308 composite:

- Same baseline as the Engine's mode label (`modeY + MODE_BTN_H + 14`) —
  one header line: mode label left, composite right.
- RIGHT-anchored at the node content edge (`W - PAD`, above the ✕ column),
  identical to the Stack v308 anchoring decision.
- All offsets via `measureText`, never eyeballed (v304/v307 lesson).
- Hover area = the full composite (text + icon); rect height 12 keeps it
  clear of row 0.
- Tooltip drawn LAST in `onDrawForeground` (Stack v304 pattern), anchored
  left of the composite; hover check runs FIRST in `onMouseMove`
  (mirrors the Stack handler order).

No serialization, backend, or row-layout changes — pure header drawing +
hover. Completes the v309 Engine CLIP mirror (mechanic v309, header v310).

## Tests

- New `tests/test_v310_engine_header.py`: source-wiring checks (composite
  present in Engine block, measureText-based, right-anchored, hover zone
  checked first, tooltip overlay present, Stack header untouched).
