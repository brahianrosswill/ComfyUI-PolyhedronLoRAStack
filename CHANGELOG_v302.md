# v302 — Per-row CLIP strength (decoupled from model weight)

Addresses the second external feedback item: CLIP (text-encoder) strength
could not be controlled independently from model strength.

## How it works
- New optional row field `wClip`. Absent → CLIP follows the model weight,
  exactly as before. Every pre-v302 workflow loads and behaves byte-identical.
- **UI (Stack + Engine):** Shift-click the weight value to type a separate
  CLIP strength; Shift-click the ◀ ▶ steppers to nudge it. When decoupled,
  the cell renders two lines (model weight + blue `c X.XX`). Entering the
  model weight again re-links the two. No layout changes when unused.
- **SEQ path:** `load_lora(model, clip, name, w_model, w_clip)` — native
  per-strength application.
- **Merged path (CONCAT/DARE/±TRIM/±RESOLVE):** text-encoder layers
  (`lora_te*`, `text_encoder.*`, `lora_prior_te*`) are pre-scaled with the
  CLIP weight inside the same one-pass merge; all other layers keep the model
  weight. Unrecognised TE conventions gracefully fall back to the model
  weight (= old behaviour, never corruption).
- **Determinism guarantee:** DARE/RESOLVE seeds remain derived from names +
  model weights only — existing masks and merge results stay bit-identical
  until a row is actually decoupled. WAN LoRAs carry no TE keys, so WAN
  workflows are unaffected by design.
- CLIP-only rows (model 0, CLIP ≠ 0) are now valid and survive filtering.

## Files touched
- `nodes/uls_stack_node.py` — helpers `_row_clip_weight` / `_is_te_base`,
  threading through `apply_lora_set` → `_apply_seq` / `_apply_concat_or_dare`
  (incl. all 10 SEQ fallbacks + OOM retry), Stack + Engine call sites.
- `web/js/uls_node.js` — Shift interactions + two-line cell (Stack & Engine),
  `wClip` in both serialization mappers (undefined is dropped by JSON →
  saved workflows only grow when the feature is used).
- `README.md` — "Per-row CLIP strength" section.
- `tests/test_v302_clip_weight.py` — 35 checks (pure helpers, wiring, filter).
- Version triple: 3.2.0 / banner v302 / uls_compat v302.
