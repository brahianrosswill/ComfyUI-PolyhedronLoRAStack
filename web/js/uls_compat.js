/**
 * Polyhedron LoRA Stack — Compatibility & Diagnostics Layer
 * ═════════════════════════════════════════════════════════
 *
 * Purpose
 * -------
 * The Stack and Engine nodes render their entire UI by hand on the LiteGraph
 * canvas (onDrawForeground + onMouseDown/Move/Up). ComfyUI's frontend now ships
 * a SECOND renderer — the Vue-based "Nodes 2.0" path — which does NOT call
 * onDrawForeground for a node. Under that renderer the hand-built Polyhedron UI
 * would simply not appear, with no error: the user just sees an empty node and
 * has no idea why.
 *
 * This module is the early-warning system for exactly that situation. It is:
 *
 *   • Purely ADDITIVE — it registers its own extension namespace, reads state,
 *     adds an About-page badge / a setting / a one-time toast. It does NOT
 *     change how any node renders or behaves. It cannot break the plugin.
 *
 *   • RENDERER-AGNOSTIC by design. It does not sniff version strings or guess
 *     setting keys (both churn release-to-release). Instead it observes a fact
 *     that is true regardless of how ComfyUI names its modes: did our node's
 *     canvas draw path actually run? uls_node.js sets `node._ulsDrawFired`
 *     inside onDrawForeground; if that never fires for a placed Polyhedron
 *     canvas node, the active renderer isn't drawing our UI → warn, with
 *     actionable guidance.
 *
 * Why this has lasting value
 * --------------------------
 * The recurring pain in this project's history is "ComfyUI changed under us
 * and a node silently broke". This layer turns that silent failure mode into a
 * clear, actionable message — and centralises the capability checks in ONE
 * place, so future ComfyUI changes surface here instead of scattered across
 * the 3000-line main UI file. It is also the detection foundation any eventual
 * Vue/Nodes-2.0 migration will build on (you cannot migrate what you cannot
 * reliably detect).
 *
 * Inspect at runtime via the browser console:  window.__POLYHEDRON_COMPAT__
 */

import { app } from "../../scripts/app.js";

// Node class names whose UI is hand-drawn on the LiteGraph canvas.
// Backend-only nodes (Bridge, Sigma, FrameInflate, Token Counter, Inspector,
// Model Switch) render via standard widgets and are unaffected by the renderer
// change — they are deliberately NOT in this set.
const POLY_CANVAS_NODES = new Set(["UltimateLoraStack", "ULSAccelerator"]);

const PLUGIN_VERSION = "v270";
const GRACE_MS       = 3000;  // time to allow at least one draw before judging

// ── Best-effort environment snapshot (all reads guarded) ──────────────────
function readFrontendVersion() {
    try {
        if (typeof window !== "undefined" && window.__COMFYUI_FRONTEND_VERSION__) {
            return String(window.__COMFYUI_FRONTEND_VERSION__);
        }
    } catch (e) { /* ignore */ }
    return "unknown";
}

const COMPAT = {
    plugin:             PLUGIN_VERSION,
    frontendVersion:    readFrontendVersion(),
    hasExtensionManager: false,   // new (Vue-capable) frontend exposes this
    hasLiteGraph:        (typeof LiteGraph !== "undefined"),
    canvasDrawObserved:  false,   // set true once any Polyhedron canvas node draws
    rendererWarningShown: false,
    checkedAt:           null,
    note: "Inspect this object for a quick Polyhedron environment report.",
};
try { window.__POLYHEDRON_COMPAT__ = COMPAT; } catch (e) { /* ignore */ }

// ── One-time, dismissible renderer warning ────────────────────────────────
let _warned = false;

function suppressed() {
    try {
        return !!app.extensionManager?.setting?.get?.(
            "Polyhedron.compat.suppressRendererWarning");
    } catch (e) { return false; }
}

function notify(severity, summary, detail, life) {
    // Prefer the documented toast API; fall back gracefully to console only.
    try {
        const tm = app.extensionManager?.toast;
        if (tm?.add) { tm.add({ severity, summary, detail, life }); return; }
        if (tm?.addAlert) { tm.addAlert(detail); return; }
    } catch (e) { /* ignore */ }
}

function warnRendererOnce() {
    if (_warned) return;
    _warned = true;
    COMPAT.rendererWarningShown = true;
    const detail =
        "A Polyhedron Stack/Engine node is placed but its custom UI hasn't " +
        "rendered. Your ComfyUI is likely using the new Vue node renderer " +
        "(\"Nodes 2.0\"), which doesn't draw the hand-built canvas UI. To use " +
        "the full Polyhedron interface, switch to LiteGraph rendering in " +
        "Settings (disable Nodes 2.0 / Vue nodes). Backend nodes — Bridge, " +
        "Sigma, Token Counter, Inspector, Model Switch — work fine either way.";
    console.warn("[Polyhedron] ⚠ " + detail +
        "  (Suppress via Settings → Polyhedron, or set " +
        "window.__POLYHEDRON_COMPAT__ aside for debugging.)");
    notify("warn", "Polyhedron LoRA Stack — renderer notice", detail, 15000);
}

// ── Probe: after a node is created, give it a grace period to draw ─────────
function armProbe() {
    setTimeout(() => {
        try {
            if (suppressed()) return;

            const nodes = app.graph?._nodes || [];
            let present = false;
            let drew = false;
            for (const n of nodes) {
                const t = n?.comfyClass || n?.type;
                if (!POLY_CANVAS_NODES.has(t)) continue;
                present = true;
                if (n._ulsDrawFired) { drew = true; break; }
            }

            // If the user removed the node again, there's nothing to warn about.
            if (!present) return;

            COMPAT.canvasDrawObserved = drew;
            COMPAT.checkedAt = new Date().toISOString();

            if (drew) return;          // canvas path is alive → all good, silent
            warnRendererOnce();        // placed but never drew → renderer notice
        } catch (e) {
            // Diagnostics must never throw into ComfyUI.
            console.debug("[Polyhedron] compat probe skipped:", e);
        }
    }, GRACE_MS);
}

app.registerExtension({
    name: "Polyhedron.compat",

    // About-page badge (frontend ignores this field on older versions).
    aboutPageBadges: [
        {
            label: "Polyhedron LoRA Stack " + PLUGIN_VERSION,
            url: "https://civitai.red/user/Polyhedron_AI",
            icon: "pi pi-box",
        },
    ],

    // User-facing toggle (frontend ignores this field on older versions).
    settings: [
        {
            id: "Polyhedron.compat.suppressRendererWarning",
            name: "Polyhedron: suppress renderer-compatibility warning",
            type: "boolean",
            defaultValue: false,
            tooltip: "Hide the one-time notice shown when the active renderer " +
                     "(e.g. Vue / Nodes 2.0) doesn't draw the Polyhedron canvas UI.",
        },
    ],

    async setup() {
        COMPAT.hasExtensionManager = !!(app && app.extensionManager);
        COMPAT.checkedAt = new Date().toISOString();
        console.log(
            "[Polyhedron] compat layer ready — " +
            `plugin=${COMPAT.plugin}, frontend=${COMPAT.frontendVersion}, ` +
            `extensionManager=${COMPAT.hasExtensionManager}, ` +
            `litegraph=${COMPAT.hasLiteGraph}`
        );
    },

    // Documented lifecycle hook — fires for every node added to the graph.
    nodeCreated(node) {
        const t = node?.comfyClass || node?.type;
        if (!POLY_CANVAS_NODES.has(t)) return;
        armProbe();
    },
});
