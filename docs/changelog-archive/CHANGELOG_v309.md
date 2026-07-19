# v309 — Engine: per-row CLIP strength fully wired

## Fix

**Polyhedron LoRA Engine: decoupled CLIP strength now actually works.**

v302 introduced the optional per-row `wClip` field (Shift-click on the weight
value, Shift+arrows to step, two-line cell with blue `c X.XX`). The shared
canvas UI and the backend reader (`_row_clip_weight` → `clip_weights` in
`apply_lora_set`) were present in **both** the Stack and the Engine — but the
two Engine serialization mappers still emitted only `{enabled, name, weight}`:

- `onSerialize` → `o._engine` (workflow persistence): `wClip` did not survive
  save/reload.
- `_ulsSync` → `engine_config` widget: `wClip` never reached the backend, so
  a decoupled CLIP weight set in the Engine UI had **no effect** on the merge.

Both mappers now include `wClip: r.wClip` — the exact v302 Stack pattern.
`undefined` falls out of the JSON automatically, so rows without a decoupled
CLIP weight serialize byte-identically to v308. `onConfigure` needed no
change: its `{...newEngineRow(), ...r}` spread restores a persisted `wClip`
as-is. No backend change (the consumer side existed since v302).

## Invariants

- Rows without `wClip`: output byte-identical to v308 (CLIP = model weight).
- DARE/RESOLVE seeds: still derived from names + model weights only —
  unaffected by this change (frontend-only).
- Stack node: untouched.
- WAN bridge: untouched.

## Tests

- New `tests/test_v309_engine_clip.py` (script-style, pattern of
  `test_v302_clip_weight.py`): source-wiring checks that both Engine mappers
  carry `wClip`, that the Engine backend reads `_row_clip_weight` and passes
  `clip_weights`, and that the Stack mappers (v302) are still intact.
- `tests/test_v302_clip_weight.py`: mapper-count expectation updated 2 → 4
  (the v302 check encoded "Stack mappers only"; with the Engine mirrored,
  four mappers correctly carry `wClip`).
