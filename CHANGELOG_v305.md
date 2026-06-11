# v305 — Header centering fix + CLIP info icon

Live-test feedback on v304:

- **Centering:** the "Weight/CLIP" header composite is now centered over the
  column using `measureText` (the v304 eyeballed right/left-aligned offsets
  sat visibly right of the cell center). Group/Trigger labels were never
  moved — their coordinates are untouched in both versions.
- **Info icon:** a small CLIP-blue 🛈 (stroked circle + "i") sits after
  "/CLIP". It is decorative by design — the whole header cell remains the
  hover area for the Shift-interaction tooltip; the icon just advertises
  that there is something to hover.
- Draw state (font, textAlign) is explicitly restored after the block, so
  the subsequent Group/Trigger labels render exactly as before.

UI-only, one block in `web/js/uls_node.js`; triple 3.5.0 / v305 / v305.
