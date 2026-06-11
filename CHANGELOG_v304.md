# v304 — CLIP UX polish (user feedback from live testing)

Three refinements to the v302 per-row CLIP strength, all UI-only:

- **CLIP popup is now CLIP-blue.** `showWeightInput` gained an optional
  `accent` parameter (default amber, unchanged for model weight); the
  "CLIP Weight" popups (Stack + Engine) pass the same blue (#6aa0d0) as the
  in-cell `c X.XX` line — title, border, value and background tint match.
- **Header reads "Weight/CLIP"** (two-tone: amber "Weight", blue "/CLIP")
  instead of "Weight". Stack only — the Engine has no column headers.
- **Hover tooltip on the header** explains the interaction:
  "Click: model weight. Shift+Click: set a per-LoRA CLIP strength
  (decoupled). Shift+◀ ▶ steps it. Enter the model weight again to
  re-link." Implemented exactly in the existing flatMode-pill tooltip
  pattern (hover zone + drawn-last overlay), anchored to the left of the
  header so it never leaves the node.

No backend changes; serialization untouched; behavior identical when the
feature is unused.

## Files touched
- `web/js/uls_node.js` (popup accent, header, hover zone, tooltip)
- Version triple: 3.4.0 / banner v304 / uls_compat v304
