# v313 — Public release: per-row CLIP strength, Nodes 2.0 compatibility, hardening

Consolidated release notes for the jump from the last published version
(v270) to v313. Internal development versions in between covered additional
experimental features that are not part of this release; everything
user-visible for the published node set is summarized here. Per-version
details for this release's features ship alongside (`CHANGELOG_v301.md` …
`CHANGELOG_v310.md`).

## New: Per-row CLIP strength (v302, v309, v310)

Each LoRA row can now carry a **CLIP (text-encoder) strength decoupled from
its model weight** — on both the **LoRA Stack** and the **LoRA Engine**:

- **Shift-click** the weight value to enter a separate CLIP strength; the
  cell switches to two lines (amber model weight, blue `c 0.40`).
- **Shift + ◀ ▶** steps the CLIP strength; plain interactions keep editing
  the model weight. Re-entering the model weight re-links the row.
- New `Weight / CLIP Strength ⓘ` column header with a hover explainer on
  both nodes.
- Works across all merge modes: in SEQ the value is passed straight to the
  loader; in CONCAT/DARE the text-encoder layers are scaled with the CLIP
  strength inside the same one-pass merge.
- **Backward compatible by construction:** rows without a decoupled value
  serialize and compute byte-identically to before. DARE/RESOLVE seeds stay
  derived from model weights only, so existing results don't shift. WAN
  LoRAs carry no text-encoder keys, so WAN workflows are unaffected either
  way.

## New: Nodes 2.0 ("Modern Node Design") compatibility layer (v301–v307)

The Stack/Engine canvas UI requires the classic LiteGraph renderer. The
compat layer now detects the Vue-based Nodes 2.0 renderer with
viewport-evidence (no false positives from offscreen nodes), shows a clear
in-node notice plus a one-time toast in that case, and self-heals the
notice the moment the canvas provably draws. Diagnostics:
`window.__POLYHEDRON_COMPAT__`.

## Fixed

- Long-standing header misalignment: the `Group` / `Trigger` column captions
  were offset a few pixels from their columns (v307).
- Notice widgets are double-excluded from serialization so they can never
  corrupt saved stack rows (v301).

## Housekeeping

- All remaining German comments/strings in the shipped node code translated
  to English (`requirements.txt`, route docstrings, canvas UI comments).
- README: step-by-step CLIP strength guide with screenshot; documentation
  links refreshed.

## Tests

The pack ships its sandbox-runnable test suite, including dedicated suites
for the CLIP strength wiring (`tests/test_v302_clip_weight.py`,
`tests/test_v309_engine_clip.py`, `tests/test_v310_engine_header.py`).
