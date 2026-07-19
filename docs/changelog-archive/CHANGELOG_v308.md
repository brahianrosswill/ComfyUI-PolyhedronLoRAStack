# v308 — Header reads "Weight / CLIP Strength"

- The weight-column header now reads **Weight / CLIP Strength** (amber /
  blue) followed by the small info icon.
- The composite is wider than the 72px weight cell, so it is RIGHT-anchored:
  the icon's right edge sits at the node content edge (above the ✕ column,
  which carries no header label) and the text extends left into the cell —
  maximum clearance from the "Group" label, all positions via measureText.
- The tooltip hover area now covers the full composite (text + icon), and
  the tooltip headline matches the new label.
- Fallback plan if a platform font renders wider than expected: stack the
  two parts on two lines — prepared conceptually, not needed yet.

UI-only, one block in `web/js/uls_node.js`; triple 3.8.0 / v308 / v308.
