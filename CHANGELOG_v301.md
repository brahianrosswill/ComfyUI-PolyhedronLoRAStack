# v301 — Nodes 2.0: visible in-node fallback notice

Addresses external user feedback: under ComfyUI's new Vue renderer ("Modern
Node Design" / Nodes 2.0) the hand-drawn canvas UI of the Stack and Engine
nodes does not render, leaving the user with a blank node and — until now —
only a transient toast explaining why.

## What changed
- `web/js/uls_compat.js`: when the compat probe detects a placed Polyhedron
  canvas node whose draw path never fired, it now injects a **display-only
  text widget into the affected node itself**: "UI needs LiteGraph renderer —
  disable 'Modern Node Design' (Nodes 2.0) in Settings. Your rows are safe."
  Standard widgets ARE rendered by the Vue renderer, so the guidance appears
  exactly where the user is looking. The widget is excluded from serialization
  (`serialize: false`, set on both the widget and its options) so it can never
  enter `widgets_values` and disturb the rows-JSON layout. The probe now
  flags every silent node individually instead of stopping at the first node
  that drew. The existing one-time toast remains.
- `README.md`: new **Compatibility** section documenting the renderer
  limitation, the workaround, and that data/backend are unaffected (with the
  rgthree precedent for context).
- Version triple: `pyproject.toml` 3.1.0 / banner v301 / `uls_compat.js` v301.

## What did NOT change
- No node logic, no backend files, no canvas UI code (`uls_node.js` untouched).
- Behavior under the classic LiteGraph renderer is identical: the probe only
  injects when the draw path never fired, and a renderer switch requires a
  frontend reload, after which a working canvas UI means no injection.

## Roadmap note
Separate per-row CLIP strength (second feedback item) is planned as its own
release (optional `wClip` row field + opt-in UI column, fully backward
compatible).
