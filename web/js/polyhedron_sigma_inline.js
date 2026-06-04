/**
 * Polyhedron Sigma — Inline Widget Outputs
 * Repositions INT/FLOAT output dots to sit next to their widget rows.
 */

import { app } from "../../scripts/app.js";

// Only the (deprecated) single-schedule node exposes passthrough INT/FLOAT
// outputs that benefit from inline dots. The Dual Sigma Curve outputs only
// sigmas_high / sigmas_low, so it has nothing to inline — its former
// passthrough mapping was dead (the slot names never matched) and is gone.
const SINGLE_INLINE = {
    "steps":     "steps",
    "sigma_max": "sigma_max",
    "sigma_min": "sigma_min",
    "rho":       "rho",
};

function patchNode(node, inlineMap) {
    if (node._sigma_patched) return;
    node._sigma_patched = true;

    const origGetConnectionPos = node.getConnectionPos.bind(node);

    node.getConnectionPos = function(isInput, slot, out) {
        if (!isInput) {
            const slotName = this.outputs?.[slot]?.name;
            const widgetName = inlineMap[slotName];
            if (widgetName) {
                const widget = this.widgets?.find(w => w.name === widgetName);
                if (widget) {
                    // last_y is set by LiteGraph after first draw — use it if available
                    const localY = (widget.last_y ?? null) !== null
                        ? widget.last_y + 10
                        : this.getWidgetLocalY(widgetName);

                    if (localY !== null) {
                        const x = this.pos[0] + this.size[0];
                        const y = this.pos[1] + localY;
                        if (out) { out[0] = x; out[1] = y; return out; }
                        return [x, y];
                    }
                }
            }
        }
        return origGetConnectionPos(isInput, slot, out);
    };

    // Helper: compute widget local Y from index (fallback when last_y not yet set)
    node.getWidgetLocalY = function(widgetName) {
        const idx = this.widgets?.findIndex(w => w.name === widgetName) ?? -1;
        if (idx === -1) return null;
        const inputCount = this.inputs?.length ?? 0;
        const TITLE  = LiteGraph.NODE_TITLE_HEIGHT ?? 30;
        const ISLOTH = LiteGraph.NODE_SLOT_HEIGHT  ?? 20;
        const WH     = LiteGraph.NODE_WIDGET_HEIGHT ?? 20;
        return TITLE + inputCount * ISLOTH + idx * (WH + 4) + WH * 0.5;
    };

    // Force redraw after a short delay so last_y gets populated
    setTimeout(() => {
        node.setDirtyCanvas?.(true, true);
    }, 100);
}

app.registerExtension({
    name: "Polyhedron.SigmaInlineOutputs",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        const inlineMap =
            nodeData.name === "ULSWanSigmaSchedule" ? SINGLE_INLINE :
            null;
        if (!inlineMap) return;

        const _onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            _onCreated?.apply(this, arguments);
            patchNode(this, inlineMap);
        };

        const _onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function(info) {
            _onConfigure?.apply(this, arguments);
            patchNode(this, inlineMap);
        };

        // Also patch drawNode to ensure position is correct after every draw
        const _onDrawFg = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function(ctx) {
            _onDrawFg?.apply(this, arguments);
            // Draw custom output dots next to widget rows
            if (!this.outputs) return;
            this.outputs.forEach((output, i) => {
                const widgetName = inlineMap[output.name];
                if (!widgetName) return;
                const widget = this.widgets?.find(w => w.name === widgetName);
                if (!widget?.last_y) return;
                const x = this.size[0] - 8;
                const y = widget.last_y + 10;
                // Draw dot
                ctx.beginPath();
                ctx.arc(x, y, 5, 0, Math.PI * 2);
                ctx.fillStyle = output.links?.length ? "#7fff7f" : "#aaaaaa";
                ctx.fill();
                ctx.strokeStyle = "#000";
                ctx.lineWidth = 1;
                ctx.stroke();
                // Label
                ctx.fillStyle = "#ccc";
                ctx.font = "11px Arial";
                ctx.textAlign = "right";
                ctx.fillText(output.name, x - 8, y + 4);
            });
        };
    },
});
