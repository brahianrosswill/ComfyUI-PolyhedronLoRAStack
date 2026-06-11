# v307 — Header alignment (pre-existing bug) + smaller info icon

- **Root cause of the "shifted headers":** the column-header block used a
  stale `_GRP_W = 36` while the rows draw the group pill with `GRP_W = 50`.
  That pushed "Group" 7px and "Trigger" 14px right of their columns — a
  long-standing bug, unrelated to v304/v305, only noticed now that the
  headers are under scrutiny. Header constants now mirror the row layout
  constants exactly (documented as a MUST in a comment).
- **Weight/CLIP centering:** the text composite now centers over the column
  exactly like the other labels; the info icon hangs after it, deliberately
  outside the centering math (decorative — the whole header cell remains the
  tooltip hover area).
- **Info icon smaller:** radius 4.5 → 3.2, stroke 1 → 0.9, "i" 7px → 5.5px.

UI-only, one block in `web/js/uls_node.js`; triple 3.7.0 / v307 / v307.
