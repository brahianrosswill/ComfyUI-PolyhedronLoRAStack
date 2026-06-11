# v303 — Renderer notice: self-healing (false-positive fix)

Live testing of v301/v302 surfaced a false positive: the in-node renderer
notice appeared on nodes whose canvas UI was rendering perfectly fine.

## Root cause
The compat probe judges "never drew" via `_ulsDrawFired` a fixed grace period
after node creation. LiteGraph culls offscreen nodes — `onDrawForeground`
never runs for a node outside the viewport — so on large workflows or slow
first draws a perfectly healthy node could be flagged, and once injected the
notice widget stuck around.

## Fix (two-sided)
- `web/js/uls_node.js`: both draw paths (Stack + Engine) now REMOVE the
  injected `polyhedron_renderer_notice` widget the moment the canvas path
  provably runs. This makes the mechanism false-positive-safe by
  construction: a wrongly injected notice heals itself on first real draw.
- `web/js/uls_compat.js`: grace period 3s → 8s to reduce spurious flashes on
  slow loads. (An offscreen-but-healthy node may still receive the notice —
  it now disappears the moment the node scrolls into view and draws.)

Under a genuine Nodes 2.0 / Vue renderer the draw path never runs, so the
notice stays — exactly as intended.

## Files touched
- `web/js/uls_node.js` (2 draw sites), `web/js/uls_compat.js` (grace + docs)
- Version triple: 3.3.0 / banner v303 / uls_compat v303
