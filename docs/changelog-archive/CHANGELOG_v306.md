# v306 — Renderer notice: viewport-evidence judgement (no more load toast)

The v303 self-healing fixed the sticky in-node widget, but the one-time toast
could still fire spuriously when the Stack/Engine nodes were simply outside
the viewport at load time (LiteGraph culls offscreen nodes, so their draw
path legitimately never runs within the grace period).

## New judgement logic (uls_compat.js)
After the grace period:
1. **Any Polyhedron canvas node drew** → canvas path is alive; silent
   siblings are just offscreen. No toast, no widget injection (the v303
   self-healing draw path still covers any theoretical stragglers).
2. **None drew, and at least one silent node is INSIDE the viewport** →
   that node would have been drawn by a live canvas; this is real evidence
   of a dead renderer (Nodes 2.0 / Vue). Toast + in-node notices, as before.
3. **None drew and all are offscreen** → no evidence either way; the probe
   re-arms and checks again (bounded at 12 re-checks ≈ 96 s, then gives up
   with a console.debug, never a user-facing warning).

The viewport test reads LiteGraph canvas internals (`ds.offset/scale`,
canvas element size) and **fails open**: if those internals are missing or
throw — plausible under a genuine Vue renderer — the node counts as visible,
so the real Nodes-2.0 case can never be swallowed by the new logic.

Diagnostics: `window.__POLYHEDRON_COMPAT__` gains `probeRearms` and
`lastJudgement` ("drew" | "toast" | "deferred" | "gave-up").

UI-only; `uls_node.js` untouched; triple 3.6.0 / v306 / v306.
