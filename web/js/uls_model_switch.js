// Polyhedron Select Model Switch — UI label tweaks.
//
// The Python side declares six COMBO inputs `model_1` .. `model_6`. ComfyUI
// auto-renders each widget label as its Python identifier, which gives us
// long redundant labels ("model_1", "model_2", ...). We override just the
// displayed label to a single digit ("1", "2", ...) so the dropdown takes
// the full node width. Python names stay unchanged — only the display
// string is swapped on the widget object.
//
// Fallback: if this extension fails to load for any reason, widgets fall
// back to their default labels (model_1..model_6). No crash, no broken
// slots — the node still works identically.

import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "polyhedron.modelswitch.labels",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "ULSModelSwitch") return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = origOnNodeCreated?.apply(this, arguments);
            try {
                for (const w of this.widgets || []) {
                    const m = typeof w.name === "string" && w.name.match(/^model_(\d+)$/);
                    if (m) {
                        w.label = m[1];   // "model_1" -> "1"
                    }
                }
            } catch (e) {
                // Silent fallback: keep default labels.
                console.warn("[PLS] ModelSwitch label patch skipped:", e);
            }
            return r;
        };
    },
});
