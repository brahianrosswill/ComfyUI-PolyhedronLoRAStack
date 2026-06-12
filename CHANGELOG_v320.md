# Changelog — v320 (3.20.0)

Stack/Engine quality-of-life improvements. No changes to merge math, the WAN
bridge, or the sampler hand-off; existing workflows load unchanged.

## Group stack-order: orphan-aware reassignment

Stack-order numbers (the gold order badges) are tracked per category. When a
category stopped being represented by any active LoRA row, its order entry used
to linger invisibly and still occupy that number, making it impossible to
assign the same number to a visible group (the assignment was silently
rejected with a brief red flash, and the phantom holder was unfindable).

Now:
- If the number's current holder is **not present among the live rows** (an
  orphan), its stale entry is reclaimed silently.
- If the holder is a **currently visible** group, a themed confirmation dialog
  appears ("Stack order N is already assigned to group X. Reassign N to Y?").
  Decline → the existing holder flashes so you can see who owns it.
- Order numbers are **not** pruned on every sync, so temporarily emptying a
  category (e.g. disabling all its LoRAs to re-add them) does not destroy a
  carefully chosen order number.

The confirmation uses an in-canvas dialog matching the node's dark styling
instead of the browser's default dialog.

## Token Counter: native over-limit toast + honest warnings

The ⬡ Polyhedron Token Counter now raises a native ComfyUI toast when a prompt
is over (or near) the model's token limit — visible regardless of where you are
working on the graph, not just in the node's text output:
- a sticky **error** toast on over-limit, and a brief **warn** toast on
  near-limit.

The report text is also clearer:
- The over-limit hint is honest about *both* possible outcomes — silent
  truncation of the prompt tail **or** kijai's negative-dimension crash — and
  states the overflow amount.
- The near-limit hint derives its percentage from the configurable
  `warn_threshold` instead of a hard-coded value.

Implementation: the counter returns a small UI payload alongside its outputs;
a new `web/js/uls_token_toast.js` hooks the node's execution and raises
`app.extensionManager.toast`. Auto-loaded via the existing web directory.
