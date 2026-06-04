"""
Polyhedron LoRA Stack
"""
import importlib.util

def _check_dep(module, pip_spec, optional=False):
    ok = importlib.util.find_spec(module) is not None
    if not ok:
        print(f"[PLS] {'○' if optional else '✗'} {pip_spec} [{'optional' if optional else 'MISSING'}]")
        if not optional: print(f"[PLS]   → pip install {pip_spec}")
    return ok

print("[PLS] Checking dependencies...")
_HAS_PIL      = _check_dep("PIL",      "Pillow>=9.0.0",    optional=True)
_HAS_REQUESTS = _check_dep("requests", "requests>=2.28.0", optional=True)
if not _HAS_PIL or not _HAS_REQUESTS:
    print("[PLS] ⚠ Missing packages → limited mode")
else:
    print("[PLS] ✓ All dependencies available")

# Node groups load INDEPENDENTLY (v254). A breaking ComfyUI/Core change in one
# group — e.g. comfy.lora moving, which only the Stack/Engine use — must not
# abort the whole pack. Each group is wrapped like the Bridge already was; on a
# failed import it logs a clear, actionable line and is simply skipped, so the
# remaining groups still register. When everything imports (the normal case)
# this is behaviour-identical to before: same classes, same display names.
_STACK_OK = _SWITCH_OK = _INFLATE_OK = _SIGMA_OK = _ANALYZER_OK = False
try:
    from .nodes.uls_stack_node import UltimateLoraStack, ULSAccelerator, ULSInspector, ULSTokenCounter
    _STACK_OK = True
except Exception as e:
    print(f"[PLS] ✗ Stack / Engine / Inspector / TokenCounter unavailable — import failed: {e!r}")
    print("[PLS]   (usually a changed ComfyUI Core API, e.g. comfy.lora). Other nodes still load.")
try:
    from .nodes.uls_resolve_inspector import ULSResolveInspector
    _ANALYZER_OK = True
except Exception as e:
    print(f"[PLS] ✗ Merge Analyzer unavailable — import failed: {e!r}")
try:
    from .nodes.uls_model_switch import ULSModelSwitch
    _SWITCH_OK = True
except Exception as e:
    print(f"[PLS] ✗ Model Switch unavailable — import failed: {e!r}")
try:
    from .nodes.wan_frame_inflate import ULSWanFrameInflate, ULSImagePickFrame
    _INFLATE_OK = True
except Exception as e:
    print(f"[PLS] ✗ Frame Inflate / Pick Frame unavailable — import failed: {e!r}")
try:
    from .nodes.wan_sigma_schedule import (ULSWanSigmaSchedule, ULSWanSplitNoiseSchedule,
                                            ULSUniversalSigmaCurve)
    _SIGMA_OK = True
except Exception as e:
    print(f"[PLS] ✗ Sigma Schedule nodes unavailable — import failed: {e!r}")

# Bridge — fragile, depends on ComfyUI/kijai internals. Already isolated.
_BRIDGE_OK = False
try:
    from .nodes.wan_model_bridge import ULSWanBridge, ULSWanBridgeReverse
    _BRIDGE_OK = True
except Exception as e:
    print(f"[PLS] ⚠ Bridge nodes failed to load: {e}")
    print("[PLS]   Bridge (MODEL ↔ WANVIDEOMODEL) will be unavailable this session.")


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

if _STACK_OK:
    NODE_CLASS_MAPPINGS.update({
        "UltimateLoraStack": UltimateLoraStack,
        "ULSAccelerator":    ULSAccelerator,
        "ULSInspector":      ULSInspector,
        "ULSTokenCounter":   ULSTokenCounter,
    })
    NODE_DISPLAY_NAME_MAPPINGS.update({
        "UltimateLoraStack": "⬡ Polyhedron LoRA Stack",
        "ULSAccelerator":    "⬡ Polyhedron LoRA Engine",
        "ULSInspector":      "⬡ Polyhedron LoRA Inspector",
        "ULSTokenCounter":   "⬡ Polyhedron Token Counter",
    })

if _SWITCH_OK:
    NODE_CLASS_MAPPINGS["ULSModelSwitch"] = ULSModelSwitch
    NODE_DISPLAY_NAME_MAPPINGS["ULSModelSwitch"] = "⬡ Polyhedron Select Model Switch"

if _ANALYZER_OK:
    NODE_CLASS_MAPPINGS["ULSResolveInspector"] = ULSResolveInspector
    NODE_DISPLAY_NAME_MAPPINGS["ULSResolveInspector"] = "⬡ Polyhedron Merge Analyzer"

if _INFLATE_OK:
    NODE_CLASS_MAPPINGS.update({
        "ULSWanFrameInflate": ULSWanFrameInflate,
        "ULSImagePickFrame":  ULSImagePickFrame,
    })
    NODE_DISPLAY_NAME_MAPPINGS.update({
        "ULSWanFrameInflate": "⬡ Polyhedron Wan Frame Inflate (T2I LoRA fix)",
        "ULSImagePickFrame":  "⬡ Polyhedron Pick Frame",
    })

if _SIGMA_OK:
    NODE_CLASS_MAPPINGS.update({
        "ULSWanSigmaSchedule":      ULSWanSigmaSchedule,
        "ULSWanSplitNoiseSchedule": ULSWanSplitNoiseSchedule,
        "ULSUniversalSigmaCurve":   ULSUniversalSigmaCurve,
    })
    NODE_DISPLAY_NAME_MAPPINGS.update({
        "ULSWanSigmaSchedule":      "⬡ Polyhedron Noise Schedule [deprecated]",
        "ULSWanSplitNoiseSchedule": "⬡ Polyhedron Dual Sigma Curve",
        "ULSUniversalSigmaCurve":   "⬡ Polyhedron Sigma Curve",
    })

if _BRIDGE_OK:
    NODE_CLASS_MAPPINGS["ULSWanBridge"]        = ULSWanBridge
    NODE_CLASS_MAPPINGS["ULSWanBridgeReverse"] = ULSWanBridgeReverse
    NODE_DISPLAY_NAME_MAPPINGS["ULSWanBridge"]        = "⬡ Polyhedron Wan Bridge (MODEL → WANVIDEOMODEL)"
    NODE_DISPLAY_NAME_MAPPINGS["ULSWanBridgeReverse"] = "⬡ Polyhedron Wan Bridge (WANVIDEOMODEL → MODEL)"

WEB_DIRECTORY = "./web/js"

try:
    from .nodes.uls_routes import register_routes
    register_routes()
except Exception as e:
    print(f"[PLS] ⚠ Routes not registered: {e}")

_node_count = len(NODE_CLASS_MAPPINGS)
_bridge_str = "✅" if _BRIDGE_OK else "⚠ unavailable"
print(f"""
⚡ ============================================================
   Polyhedron LoRA Stack  v268
   {_node_count} Nodes  |  Pillow: {'✅' if _HAS_PIL else '❌'}  |  requests: {'✅' if _HAS_REQUESTS else '❌'}  |  Bridge: {_bridge_str}
⚡ ============================================================
""")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
