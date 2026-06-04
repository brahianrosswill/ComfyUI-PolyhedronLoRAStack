"""
Polyhedron Wan Bridge — Backend
═══════════════════════════════════════
Two nodes:
  - ULSWanBridge        MODEL → WANVIDEOMODEL
  - ULSWanBridgeReverse WANVIDEOMODEL → MODEL

Both are the same comfy.model_patcher.ModelPatcher under the hood. ComfyUI
refuses connections between different type labels even when the Python class
is identical — these nodes re-label the patcher so classical and Wan-wrapper
nodes can be mixed in one workflow.

Core vs kijai WanModel:
  Core's comfy.ldm.wan.model.WanModel uses its own attention path
  (comfy.ldm.modules.attention.optimized_attention) controlled by ComfyUI's
  global settings (--use-sage-attention, xformers, sdpa fallback). Kijai's
  attention_mode attribute on the transformer has no effect on Core's forward
  pass — the Bridge therefore does not expose an attention_mode widget.
  Use ComfyUI's startup flags or the PatchSageAttentionKJ node instead.
"""

import os
import types
import torch as _torch
import comfy.model_patcher


# Verbose logging — off by default for a quiet console (v254). The per-call
# classify/diagnostic prints add up over a sampling run; enable them only when
# debugging by setting the env var PLS_BRIDGE_VERBOSE=1 (or true/yes/on).
# Critical warnings and errors do NOT go through this gate — they always print.
_VERBOSE = os.environ.get("PLS_BRIDGE_VERBOSE", "").strip().lower() in ("1", "true", "yes", "on")


def _vlog(msg: str) -> None:
    """Verbose-gated print. Honours _VERBOSE module flag."""
    if _VERBOSE:
        print(msg)


# ─── Monkey-Patch: kijai's load_weights ─────────────────────────────────────
#
# Kijai's nodes_sampler.WanVideoSampler.process() calls load_weights() from
# nodes_model_loading.py to push the state_dict onto the GPU. That function
# iterates over the transformer's parameters and looks each one up in the
# `sd` dict — crashing with KeyError if the dict is empty, or with
# QuantizedTensor.__new__() complaints if the dict contains ComfyUI Core's
# quantised tensors.
#
# When the bridge forwards a model that ComfyUI Core has ALREADY loaded onto
# the GPU (the normal case for "Load Diffusion Model"), we do not want kijai
# to re-load anything. So we monkey-patch load_weights to recognise a marker
# attribute on the transformer and return a no-op in that case. All other
# callers (kijai's own loader, etc.) are unaffected.
#
# We patch the function in kijai's module dict at import time, once.

_BRIDGE_SKIP_MARKER = "_uls_bridge_skip_kijai_load"

# Keyword arguments Core's WanModel.forward() actually accepts. Kijai's sampler
# passes a wider set (seq_len extras, kijai-only control tensors, …); we forward
# only these to Core and drop the rest. If Core ADDS a new *required* parameter,
# the forward call below raises TypeError — caught and reported with a pointer
# to the Bridge-Master update checklist (6.2) rather than a bare stack trace.
# Defined at module scope so it is not rebuilt on every sampling step.
_CORE_FORWARD_KEYS = frozenset({
    "x", "timestep", "context", "clip_fea",
    "seq_len", "time_dim_concat", "transformer_options",
})

# Three independent patch guards — each installs once per process regardless
# of the others. This avoids the bug where a combined guard prevented
# re-patching when one symbol wasn't available yet on the first call.
_PATCH_LOAD_WEIGHTS    = False
_PATCH_SET_LORA_PARAMS = False
_PATCH_REMOVE_LORA     = False

# v254 — diagnostic only. When kijai's WanVideoWrapper IS loaded but an expected
# function is absent, that almost always means the installed kijai version
# renamed or moved it. The installers used to return silently in that case,
# indistinguishable from "kijai not loaded yet" — so a kijai rename meant the
# bridge quietly never patched and the user saw no reason for any LoRA/weight
# misbehaviour. This emits ONE loud, actionable warning per missing symbol.
# It changes NO patch decision, touches NO forward/sampler/hand-off logic, and
# never raises.
_WARNED_MISSING = set()


def _warn_symbol_missing_once(symbol: str) -> None:
    try:
        if symbol in _WARNED_MISSING:
            return
        _WARNED_MISSING.add(symbol)
        print(
            f"[ULSWanBridge] ⚠ kijai's WanVideoWrapper is loaded but "
            f"'{symbol}' was not found — the installed kijai version may have "
            f"renamed or moved it. The bridge will NOT patch it; kijai may then "
            f"re-load weights / manage LoRAs itself on the WAN path, which can "
            f"cause double-loading or LoRA conflicts. If WAN LoRA behaviour "
            f"looks wrong or doubled, check for a kijai WanVideoWrapper update "
            f"against this bridge version."
        )
    except Exception:
        pass


class _NoOpTeaCache:
    """
    Dummy stand-in for kijai's TeaCache state object.
    Kijai's utils.offload_transformer() calls .clear_all() unconditionally
    after every sampling step when teacache_state is not None. Since we stub
    teacache_state to None, offload_transformer crashes with
    'NoneType has no attribute clear_all'. We use this sentinel instead of
    None so the check `if transformer.teacache_state is not None` still
    evaluates to False-ish … actually kijai checks `is not None`, so we need
    to let that branch run. Instead we provide a real object with no-op
    methods so the call succeeds harmlessly.
    NOTE: kijai also checks `if teacache_state is not None` before enabling
    tea-cache logic inside the forward pass. To keep tea-cache disabled we
    override __bool__ to return False.
    """
    def __bool__(self):      return False
    def clear_all(self):     pass
    def __repr__(self):      return "<ULS NoOpTeaCache>"


def _find_kijai_modules():
    """Return (mod_loading, sampler_mod, cl_mod) — any may be None."""
    import sys
    mod_loading = sampler_mod = cl_mod = None
    for name, m in list(sys.modules.items()):
        if "WanVideoWrapper" not in name:
            continue
        if name.endswith("nodes_model_loading"):
            mod_loading = m
        elif name.endswith("nodes_sampler"):
            sampler_mod = m
        elif name.endswith("custom_linear"):
            cl_mod = m
    return mod_loading, sampler_mod, cl_mod


def _install_load_weights_bypass():
    """Bypass kijai's load_weights(). Idempotent. Patches all WanVideoWrapper modules."""
    global _PATCH_LOAD_WEIGHTS
    if _PATCH_LOAD_WEIGHTS:
        return
    import sys
    mod_loading, _, _ = _find_kijai_modules()
    if mod_loading is None:
        return  # kijai not loaded yet — will retry on next bridge() call
    if not hasattr(mod_loading, "load_weights"):
        _warn_symbol_missing_once("load_weights")
        return
    if getattr(mod_loading.load_weights, "_uls_bridge_wrapper", False):
        _PATCH_LOAD_WEIGHTS = True
        return

    orig = mod_loading.load_weights

    def _wrapper(transformer, sd, *args, **kwargs):
        if getattr(transformer, _BRIDGE_SKIP_MARKER, False):
            _vlog("[ULSWanBridge]   bypassing kijai's load_weights() — "
                  "weights already loaded by ComfyUI Core.")
            return
        return orig(transformer, sd, *args, **kwargs)

    _wrapper._uls_bridge_wrapper = True

    patched_in = []
    for mod_name, mod in list(sys.modules.items()):
        if "WanVideoWrapper" not in mod_name:
            continue
        if hasattr(mod, "load_weights"):
            ref = getattr(mod, "load_weights")
            if not getattr(ref, "_uls_bridge_wrapper", False):
                mod.load_weights = _wrapper
                patched_in.append(mod_name.split(".")[-1])

    print(f"[ULSWanBridge]   installed load_weights bypass in: "
          f"{patched_in if patched_in else '(none found)'}")
    _PATCH_LOAD_WEIGHTS = True


def _install_set_lora_params_bypass():
    """Bypass kijai's set_lora_params(). Idempotent. Patches all WanVideoWrapper modules."""
    global _PATCH_SET_LORA_PARAMS
    if _PATCH_SET_LORA_PARAMS:
        return
    import sys
    _, _, cl_mod = _find_kijai_modules()
    if cl_mod is None:
        return  # kijai not loaded yet — will retry on next bridge() call
    if not hasattr(cl_mod, "set_lora_params"):
        _warn_symbol_missing_once("set_lora_params")
        return
    if getattr(cl_mod.set_lora_params, "_uls_bridge_wrapper", False):
        _PATCH_SET_LORA_PARAMS = True
        return

    orig = cl_mod.set_lora_params

    def _wrapper(transformer, *args, **kwargs):
        if getattr(transformer, _BRIDGE_SKIP_MARKER, False):
            _vlog("[ULSWanBridge]   bypassing kijai's set_lora_params() — "
                  "LoRAs managed by ComfyUI Core patcher.")
            return
        return orig(transformer, *args, **kwargs)

    _wrapper._uls_bridge_wrapper = True

    patched_in = []
    for mod_name, mod in list(sys.modules.items()):
        if "WanVideoWrapper" not in mod_name:
            continue
        if hasattr(mod, "set_lora_params"):
            ref = getattr(mod, "set_lora_params")
            if not getattr(ref, "_uls_bridge_wrapper", False):
                mod.set_lora_params = _wrapper
                patched_in.append(mod_name.split(".")[-1])

    print(f"[ULSWanBridge]   installed set_lora_params bypass in: "
          f"{patched_in if patched_in else '(none found)'}")
    _PATCH_SET_LORA_PARAMS = True


def _install_remove_lora_bypass():
    """Bypass kijai's remove_lora_from_module(). Idempotent.

    Wraps kijai's function across every WanVideoWrapper submodule that imported
    it (including utils.py which is called from offload_transformer). The
    wrapper short-circuits when the transformer carries our BRIDGE_SKIP_MARKER.

    Defence-in-depth: the marker is only set for bridges that route LoRAs
    through ComfyUI Core. Combined with our pipe["sd"] = None setting (which
    prevents kijai's _replace_linear() from creating CustomLinear layers in
    the first place), the bypass should never even need to fire for the
    classical Load-Diffusion-Model path. For the new kijai-wan path, no
    marker is set and kijai's original function runs unmodified.
    """
    global _PATCH_REMOVE_LORA
    if _PATCH_REMOVE_LORA:
        return
    import sys
    _, sampler_mod, cl_mod = _find_kijai_modules()
    if cl_mod is None:
        return
    if not hasattr(cl_mod, "remove_lora_from_module"):
        _warn_symbol_missing_once("remove_lora_from_module")
        return
    if getattr(cl_mod.remove_lora_from_module, "_uls_bridge_wrapper", False):
        _PATCH_REMOVE_LORA = True
        return

    orig = cl_mod.remove_lora_from_module

    def _wrapper(module, *args, **kwargs):
        # Fast path: transformer is ours — skip entirely.
        if getattr(module, _BRIDGE_SKIP_MARKER, False):
            _vlog("[ULSWanBridge]   bypassing kijai's remove_lora_from_module() — "
                  "no kijai LoRA layers present.")
            return
        return orig(module, *args, **kwargs)

    _wrapper._uls_bridge_wrapper = True

    # Patch every known reference to remove_lora_from_module across all
    # WanVideoWrapper submodules — including utils.py which imports it
    # independently and is called from offload_transformer().
    patched_in = []
    for mod_name, mod in list(sys.modules.items()):
        if "WanVideoWrapper" not in mod_name:
            continue
        if hasattr(mod, "remove_lora_from_module"):
            ref = getattr(mod, "remove_lora_from_module")
            if not getattr(ref, "_uls_bridge_wrapper", False):
                mod.remove_lora_from_module = _wrapper
                patched_in.append(mod_name.split(".")[-1])

    print(f"[ULSWanBridge]   installed remove_lora_from_module bypass in: "
          f"{patched_in if patched_in else '(none found)'}")
    _PATCH_REMOVE_LORA = True


def _install_kijai_bypasses():
    """Install all three bypasses. Called on every bridge() invocation (all are idempotent)."""
    _install_load_weights_bypass()
    _install_set_lora_params_bypass()
    _install_remove_lora_bypass()


# ─── Helpers ────────────────────────────────────────────────────────────────

def _classify_diffusion_model(model) -> tuple:
    """
    Inspect a ModelPatcher and return (kind, module_name, class_name).

    kind in {"kijai-wan", "core-wan", "other-wan", "non-wan", "unknown"}

    Kijai structure: patcher.model IS the WanVideoModel (no diffusion_model attr).
    Core structure:  patcher.model.diffusion_model IS the WanModel.
    """
    inner = getattr(model, "model", None)
    if inner is None:
        return ("unknown", "?", "?")

    dm = getattr(inner, "diffusion_model", None)
    # Kijai wraps WanVideoModel directly as .model without a .diffusion_model layer.
    # Fall back to inner itself so we classify kijai models correctly.
    target = dm if dm is not None else inner

    cls_name = type(target).__name__
    mod_name = type(target).__module__

    is_wan_like = ("Wan" in cls_name) or ("wan" in mod_name.lower())
    if not is_wan_like:
        return ("non-wan", mod_name, cls_name)

    if "wanvideo.modules" in mod_name:
        return ("kijai-wan", mod_name, cls_name)
    if "comfy.ldm.wan" in mod_name:
        return ("core-wan", mod_name, cls_name)
    return ("other-wan", mod_name, cls_name)


# ─── Forward Bridge: MODEL → WANVIDEOMODEL ─────────────────────────────────

class ULSWanBridge:
    """
    Polyhedron Wan Bridge — converts classical ComfyUI MODEL output (lila)
    into the WANVIDEOMODEL input (grün) expected by kijai's WanVideoSampler.

    Same universal pass-through philosophy as the rest of the Polyhedron pack:
    no model patching beyond what is required for the wrapper's bookkeeping
    (pipeline dict, transformer_options).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "ComfyUI MODEL — typically from UNETLoader or "
                               "LoraLoader. Must be a Wan model under the hood."
                }),
            },
        }

    RETURN_TYPES  = ("WANVIDEOMODEL",)
    RETURN_NAMES  = ("model",)
    FUNCTION      = "bridge"
    CATEGORY      = "Polyhedron/Bridge"
    OUTPUT_NODE   = False

    def bridge(self, model):
        if not isinstance(model, comfy.model_patcher.ModelPatcher):
            raise TypeError(
                "[ULSWanBridge] Expected a ComfyUI ModelPatcher, got "
                f"{type(model).__name__}. Connect the input to a real MODEL "
                "output (UNETLoader, LoraLoader, …)."
            )

        kind, mod_name, cls_name = _classify_diffusion_model(model)

        # Diagnostic: print the classification result so we can spot
        # mismatches between what kijai gives us and what we think we see.
        # Routed through _vlog so it can be silenced for production runs.
        if _VERBOSE:
            _inner_diag = getattr(model, "model", None)
            _dm_diag = getattr(_inner_diag, "diffusion_model", None) if _inner_diag is not None else None
            _vlog(
                f"[ULSWanBridge]   classify: kind={kind}, "
                f"inner={type(_inner_diag).__module__ if _inner_diag is not None else 'None'}."
                f"{type(_inner_diag).__name__ if _inner_diag is not None else 'None'}, "
                f"dm={type(_dm_diag).__module__ if _dm_diag is not None else 'None'}."
                f"{type(_dm_diag).__name__ if _dm_diag is not None else 'None'}"
            )

        # --- Information only, no gating ----------------------------------------
        if kind == "non-wan":
            print(
                f"[ULSWanBridge] ⚠ input does not look like a Wan model "
                f"({mod_name}.{cls_name}) — forwarding anyway."
            )
        elif kind == "core-wan":
            _vlog(
                f"[ULSWanBridge] core-ComfyUI Wan detected "
                f"({mod_name}.{cls_name}) — forwarding to kijai sampler."
            )
        elif kind == "kijai-wan":
            _vlog(
                f"[ULSWanBridge] kijai Wan detected "
                f"({mod_name}.{cls_name}) — simple type re-label."
            )

        # --- Subscript magic & pipeline dict ----------------------------------
        # Kijai's WanVideoModel implements __getitem__/__setitem__ that read
        # from self.pipeline. Kijai's sampler then does model["base_dtype"],
        # model["sd"], model["compile_args"], etc. ComfyUI's core WanModel is
        # a plain torch.nn.Module subclass and does NOT support subscription.
        #
        # We add the magic methods at runtime (bound on the instance), and
        # pre-populate the pipeline dict with the keys kijai's sampler reads.
        #
        # KIJAI-WAN: skip the whole block. Kijai's own WanVideoModel already
        # implements __getitem__ and the pipeline dict is already filled by
        # the loader. We have no business touching either.
        inner = getattr(model, "model", None)
        if inner is not None and kind != "kijai-wan":
            # 1) Ensure pipeline dict exists
            if not hasattr(inner, "pipeline") or not isinstance(
                    getattr(inner, "pipeline", None), dict):
                try:
                    inner.pipeline = {}
                    print("[ULSWanBridge]   created .pipeline dict on inner model")
                except Exception as e:
                    print(f"[ULSWanBridge] ⚠ could not set .pipeline: {e}")

            # 2) Inject __getitem__ / __setitem__ via the CLASS, since dunder
            #    methods are looked up on the type, not the instance. We patch
            #    the class once per process; subsequent passes are no-ops.
            cls = type(inner)
            if not hasattr(cls, "__getitem__") or getattr(
                    cls, "_uls_bridge_subscript_patched", False) is not True:
                def __ulsbridge_getitem__(self, key):
                    try:
                        return self.pipeline[key]
                    except (KeyError, AttributeError):
                        # Sampler-friendly defaults instead of KeyError storms.
                        # Returning None for unknown keys mirrors how missing
                        # optional config would behave in kijai's own loader.
                        return None
                def __ulsbridge_setitem__(self, key, value):
                    if not hasattr(self, "pipeline") or \
                            not isinstance(self.pipeline, dict):
                        self.pipeline = {}
                    self.pipeline[key] = value
                try:
                    cls.__getitem__ = __ulsbridge_getitem__
                    cls.__setitem__ = __ulsbridge_setitem__
                    cls._uls_bridge_subscript_patched = True
                    print(
                        f"[ULSWanBridge]   patched __getitem__/__setitem__ on "
                        f"{cls.__module__}.{cls.__name__}"
                    )
                except Exception as e:
                    print(
                        f"[ULSWanBridge] ⚠ could not patch subscript "
                        f"methods on {cls.__name__}: {e}"
                    )

            # 3) Pre-populate keys kijai's sampler reads from model[...]
            #    Defaults chosen to mirror an unquantised fp16/bf16 load.
            pipe = inner.pipeline
            # Detect base dtype from the first parameter we can find
            try:
                first_param = next(iter(inner.parameters()))
                detected_dtype = first_param.dtype
            except (StopIteration, AttributeError):
                detected_dtype = None
            pipe.setdefault("base_dtype", detected_dtype)
            pipe.setdefault("dtype", detected_dtype)
            pipe.setdefault("weight_dtype", detected_dtype)
            pipe.setdefault("manual_cast_dtype", None)
            # Quantisation flags — classical loader has none
            pipe.setdefault("quantization", "disabled")
            pipe.setdefault("gguf", False)
            pipe.setdefault("scaled_fp8", False)
            # Block-swap & compile — not used in classical workflow
            pipe.setdefault("block_swap_args", None)
            pipe.setdefault("compile_args", None)
            # Offload management — None means "let ComfyUI handle it"
            pipe.setdefault("manual_offloading", False)
            pipe.setdefault("auto_cpu_offload", False)
            # State-dict for LoRA-key matching / weight loading.
            #
            # We set sd=None to prevent kijai's _replace_linear() from running.
            # _replace_linear() would replace nn.Linear with CustomLinear on the
            # shared diffusion_model object — permanently, across all clones and
            # queue jobs. ComfyUI Core's DynamicVRAM then offloads weights to
            # Meta tensors which CustomLinear doesn't understand, causing a
            # "Cannot copy out of meta tensor" crash on the second queue job.
            #
            # sd=None means kijai skips _replace_linear → forward falls back to
            # Core's own Linear + LoRA-hook system. This is slower than
            # CustomLinear's fp8-optimised path, but stable across repeated runs.
            pipe.setdefault("sd", None)
            # Detected model type — best effort from in_channels heuristic
            if "model_type" not in pipe:
                try:
                    dm = inner.diffusion_model
                    in_ch = getattr(dm, "in_dim", None) or \
                            getattr(dm, "in_channels", None)
                    # i2v models have in_dim=36 (16 latent + 4 mask + 16 ref),
                    # t2v models have in_dim=16
                    if in_ch is not None and in_ch >= 32:
                        pipe["model_type"] = "i2v"
                    else:
                        pipe["model_type"] = "t2v"
                except Exception:
                    pipe["model_type"] = "t2v"
            print(
                f"[ULSWanBridge]   pipeline pre-populated: "
                f"base_dtype={pipe['base_dtype']}, "
                f"model_type={pipe['model_type']}, "
                f"sd={pipe['sd']!r}"
            )

        # --- attention_mode attribute on the diffusion model ------------------
        # Kijai's sampler reads dm.attention_mode. Core's forward ignores it —
        # Core uses comfy's own attention path (sdpa/xformers/sage via CLI).
        # We set a fixed default so kijai's sampler doesn't trip on a missing attr.
        if inner is not None:
            dm = getattr(inner, "diffusion_model", None)
            if dm is not None:
                try:
                    if not hasattr(dm, "attention_mode"):
                        dm.attention_mode = "sdpa"
                except Exception:
                    pass

        # --- Stub out kijai-only feature attributes on the transformer -------
        # Kijai's sampler probes for many optional sub-modules (audio, motion,
        # face control, dual-control, etc.) via simple `if transformer.X is
        # not None` checks. The Core-ComfyUI WanModel doesn't have these,
        # which causes AttributeError instead of returning None.
        #
        # Strategy: explicitly set the known names to None so the gating
        # checks pass cleanly. This is a list of every optional attribute
        # we've observed kijai accessing in nodes_sampler.py and similar.
        # If a future kijai update adds more, the next crash trace will
        # tell us the name and we add it here.
        #
        # KIJAI-WAN EXCEPTION: skip stubbing entirely — kijai's own loader
        # already set every attribute correctly with real values. Stubbing
        # would only fire on missing attrs (hasattr check), but the optional
        # attribute names overlap with attrs kijai uses for VRAM management
        # (main_device, offload_device, block_swap_args). Even though hasattr
        # protects us, skipping the whole pass is cleaner and faster.
        if inner is not None and kind != "kijai-wan":
            dm = getattr(inner, "diffusion_model", None)
            if dm is not None:
                kijai_optional_attrs = [
                    # Audio-driven generation
                    "audio_model",
                    "audio_proj",
                    "multitalk_audio_proj",
                    "multitalk_model_type",
                    # Motion / pose control
                    "motion_proj",
                    "ip_adapter",
                    # Face / portrait control
                    "fantasytalking_model",
                    "fantasyportrait_model",
                    "lynx_model",
                    # Reference & extra conditioning
                    "ref_conv",
                    "add_conv_in",
                    "add_proj",
                    # Dual / multi controller
                    "dual_controller",
                    # Cache mechanisms — teacache_state must NOT be None:
                    # kijai calls .clear_all() on it unconditionally in
                    # offload_transformer(). We use a no-op sentinel that
                    # evaluates to False so tea-cache stays disabled but
                    # the .clear_all() call doesn't crash.
                    # magcache / easycache are only checked with `is not None`
                    # before calling methods too — same treatment.
                    "magcache_state",
                    "easycache_state",
                    # UniAnimate
                    "unianimate_model",
                    # Misc feature flags / state
                    "lora_scheduling_enabled",
                    "block_swap_args",
                    "main_device",
                    "offload_device",
                ]
                # teacache_state gets a special no-op object, not None
                _TEACACHE_ATTRS = {"teacache_state", "magcache_state", "easycache_state"}
                stubbed = []
                for name in kijai_optional_attrs:
                    if not hasattr(dm, name):
                        try:
                            val = _NoOpTeaCache() if name in _TEACACHE_ATTRS else None
                            object.__setattr__(dm, name, val)
                            stubbed.append(name)
                        except Exception as e:
                            print(
                                f"[ULSWanBridge] ⚠ could not stub {name}: {e}"
                            )
                # Also ensure teacache_state is NoOpTeaCache even if it was
                # already set to None by a previous run
                for name in _TEACACHE_ATTRS:
                    if getattr(dm, name, None) is None:
                        try:
                            object.__setattr__(dm, name, _NoOpTeaCache())
                        except Exception:
                            pass
                if stubbed:
                    print(
                        f"[ULSWanBridge]   stubbed {len(stubbed)} optional "
                        f"kijai attributes as None: {stubbed}"
                    )

                # Plus a generic fallback __getattr__ at the CLASS level for
                # any optional attribute we forgot. nn.Module's __getattr__
                # raises AttributeError when nothing is found; we override
                # to return None for "kijai-style" attribute names. This
                # only fires for attributes not in __dict__, _parameters,
                # _buffers, or _modules — so it's safe for normal usage.
                dm_cls = type(dm)
                if not getattr(dm_cls, "_uls_bridge_getattr_patched", False):
                    orig_getattr = dm_cls.__getattr__
                    def __ulsbridge_getattr__(self, name):
                        try:
                            return orig_getattr(self, name)
                        except AttributeError:
                            # Only return None for clearly-optional names.
                            # Anything that looks like a private/dunder or a
                            # core PyTorch attribute should still raise.
                            if name.startswith("_") or name in (
                                    "training", "dump_patches", "call_super_init",
                                    "_parameters", "_buffers", "_modules",
                                    "_backward_hooks", "_forward_hooks",
                                    "_state_dict_hooks", "_load_state_dict_pre_hooks"):
                                raise
                            return None
                    try:
                        dm_cls.__getattr__ = __ulsbridge_getattr__
                        dm_cls._uls_bridge_getattr_patched = True
                        print(
                            f"[ULSWanBridge]   patched __getattr__ on "
                            f"{dm_cls.__module__}.{dm_cls.__name__} "
                            f"(unknown attributes → None)"
                        )
                    except Exception as e:
                        print(
                            f"[ULSWanBridge] ⚠ could not patch __getattr__: {e}"
                        )

        # --- transformer_options (kijai sampler also reads from here) --------
        # Kijai's sampler reads several keys from transformer_options via
        # subscript access (which raises KeyError on missing keys, unlike
        # .get()). We must provide defaults for all of them, otherwise it
        # crashes on the first missing one.
        #
        # KIJAI-WAN: kijai's own loader already populates these correctly with
        # real values (block_swap_args from BlockSwap node, etc.). Overwriting
        # with our defaults via setdefault would be harmless (setdefault skips
        # existing keys), but we still skip to keep the kijai-wan path clean.
        if kind != "kijai-wan":
            if "transformer_options" not in model.model_options:
                model.model_options["transformer_options"] = {}
            tops = model.model_options["transformer_options"]
            tops.setdefault("attention_mode", "sdpa")  # Core ignores this; fixed default for kijai sampler
            tops.setdefault("block_swap_args", None)
            # LoRA-related — we already merged LoRAs via classical patchers, so:
            tops.setdefault("merge_loras", True)
            tops.setdefault("lora_scheduling_enabled", False)
            tops.setdefault("low_mem_load", False)
            # IMPORTANT: do NOT set optimized_attention_override or attention_mode_override
            # to None here. Core's attention code in comfy/ldm/modules/attention.py calls
            # transformer_options["optimized_attention_override"](func, ...) directly
            # without a None-check — so None causes "NoneType is not callable".
            # Only set these if they have a real callable value; otherwise leave them absent.
            # UltraVico / Radial attention scalars — safe to set as None (checked with 'is not None' in kijai)
            tops.setdefault("ultravico_alpha", None)
            tops.setdefault("dense_attention_mode", None)
            tops.setdefault("dense_blocks", None)
            tops.setdefault("dense_vace_blocks", None)
            tops.setdefault("dense_timesteps", None)

        # --- Diagnostic dump --------------------------------------------------
        if inner is not None:
            dm = getattr(inner, "diffusion_model", None)
            if dm is not None:
                probes = [
                    ("pipeline",         "model attribute"),
                    ("blocks",           "diffusion_model attribute (BlockList)"),
                    ("patch_embedding",  "diffusion_model attribute"),
                    ("dim",              "diffusion_model attribute"),
                    ("num_heads",        "diffusion_model attribute"),
                    ("attention_mode",   "diffusion_model attribute"),
                ]
                missing = []
                present = []
                for name, where in probes:
                    target = inner if "model attribute" == where else dm
                    if hasattr(target, name):
                        present.append(name)
                    else:
                        missing.append(name)
                if missing:
                    print(
                        f"[ULSWanBridge]   diagnostic: STILL missing: {missing}"
                    )
                if present:
                    print(
                        f"[ULSWanBridge]   diagnostic: now present: {present}"
                    )

        # --- Install kijai load_weights bypass and mark transformer -----------
        # Kijai's sampler calls load_weights(transformer, sd, …) before the
        # sampling loop. The transformer it passes is patcher.model.diffusion_model.
        # We install a monkey-patch on kijai's load_weights that returns
        # early when it sees our marker, leaving the already-loaded weights
        # untouched. The patch is installed once per process; all subsequent
        # bridge calls are no-ops for the install step.
        # KIJAI-WAN: kijai's loader already finished load_weights() correctly
        # before the model ever reached the bridge. Setting the SKIP_MARKER on
        # a kijai model would prevent kijai's sampler from doing further weight
        # management it might still need (block-swap re-placement on each step).
        # So we install bypasses + marker only for non-kijai paths, and actively
        # REMOVE any stale marker that a previous run may have left on the model.
        if kind == "kijai-wan" and inner is not None:
            dm = getattr(inner, "diffusion_model", None) or inner
            if getattr(dm, _BRIDGE_SKIP_MARKER, False):
                try:
                    object.__setattr__(dm, _BRIDGE_SKIP_MARKER, False)
                    _vlog(
                        f"[ULSWanBridge]   removed stale {_BRIDGE_SKIP_MARKER} "
                        f"from kijai model — kijai's load_weights will run normally."
                    )
                except Exception as e:
                    print(f"[ULSWanBridge] ⚠ could not clear marker: {e}")

        if kind != "kijai-wan":
            try:
                _install_kijai_bypasses()
            except Exception as e:
                print(
                    f"[ULSWanBridge] ⚠ failed to install kijai bypasses: {e} "
                    f"— Make sure ComfyUI-WanVideoWrapper is installed."
                )

            if inner is not None:
                dm = getattr(inner, "diffusion_model", None)
                if dm is not None:
                    try:
                        object.__setattr__(dm, _BRIDGE_SKIP_MARKER, True)
                        print(
                            f"[ULSWanBridge]   marked transformer with "
                            f"{_BRIDGE_SKIP_MARKER}=True — kijai's load_weights "
                            f"will skip it."
                        )
                    except Exception as e:
                        print(
                            f"[ULSWanBridge] ⚠ could not set bypass marker: {e}"
                        )

        # --- Forward-Adapter: kijai call convention → Core call convention ----
        # Kijai's sampler does:  transformer(context=..., x=[z], t=timestep, ...)
        # Core WanModel.forward: (self, x, t, context, ...)
        #   - x must be a 5D Tensor [B,C,T,H,W], not a list and not 4D
        #   - keyword arg is 't', not 'timestep'
        #   - returns a Tensor, kijai expects ([tensor], None, None)
        #
        # IMPORTANT: We patch the INSTANCE via __call__ override, not the CLASS,
        # to avoid the recursive double-call seen when patching dm.__class__.forward
        # (the class patch applies to itself on the second invocation).
        #
        # KIJAI-WAN EXCEPTION: If the underlying diffusion_model is already a
        # kijai WanModel, kijai's sampler can call it directly — no adapter needed.
        # Installing the adapter on a kijai WanModel causes a meta-tensor crash
        # because kijai's forward() tries to move original_patch_embedding to
        # main_device (which is None in our context).
        #
        # FORCE-CLEANUP: If a previous run installed the adapter on this same dm
        # object (e.g. workflow was changed mid-session from core-wan to kijai-wan),
        # the adapter persists on the instance. Remove it so kijai's native
        # forward() is restored.
        if inner is not None and kind == "kijai-wan":
            dm = getattr(inner, "diffusion_model", None)
            if dm is None:
                dm = inner  # kijai's WanVideoModel-BaseModel wraps WanModel as dm,
                            # but some loaders put WanModel directly as inner
            if dm is not None and getattr(dm, "_uls_forward_adapted", False):
                try:
                    # Restore class-level forward by deleting instance-bound one
                    if "forward" in dm.__dict__:
                        del dm.__dict__["forward"]
                    object.__setattr__(dm, "_uls_forward_adapted", False)
                    _vlog(
                        f"[ULSWanBridge]   removed stale forward-adapter from "
                        f"{dm.__class__.__module__}.{dm.__class__.__name__} "
                        f"(kijai-wan: kijai's native forward restored)"
                    )
                except Exception as e:
                    print(f"[ULSWanBridge] ⚠ could not remove stale adapter: {e}")
            # Diagnostic: report what we see (verbose only)
            _vlog(
                f"[ULSWanBridge]   kijai-wan path: inner={type(inner).__module__}.{type(inner).__name__}, "
                f"dm={type(dm).__module__}.{type(dm).__name__ if dm is not None else 'None'}, "
                f"adapter_flag={getattr(dm, '_uls_forward_adapted', 'absent') if dm is not None else 'n/a'}"
            )

        if inner is not None and kind != "kijai-wan":
            dm = getattr(inner, "diffusion_model", None)
            if dm is not None and not getattr(dm, "_uls_forward_adapted", False):
                try:
                    _orig_cls_forward = dm.__class__.forward

                    def _uls_adapted_call(self_dm, *args, **kwargs):
                        # v229 Core-signature fix:
                        # Core WanModel.forward() changed parameter name from 't' → 'timestep'
                        # (around ComfyUI 0.19.x). Kijai's nodes_sampler.py line 1437 still
                        # sends 't': timestep in base_params. So we rename t → timestep
                        # (opposite direction to the pre-v164 historical rename).
                        if "t" in kwargs and "timestep" not in kwargs:
                            kwargs["timestep"] = kwargs.pop("t")
                        # Unpack x from list
                        if "x" in kwargs and isinstance(kwargs["x"], list):
                            kwargs["x"] = kwargs["x"][0]
                        # Add batch dim if x is 4D
                        if "x" in kwargs and isinstance(kwargs["x"], _torch.Tensor) \
                                and kwargs["x"].ndim == 4:
                            kwargs["x"] = kwargs["x"].unsqueeze(0)
                        # I2V: kijai sends y=[image_cond] separately (20ch mask+image).
                        # Core's patch_embedding expects x already concatenated to 36ch.
                        # Concatenate along channel dim: [B, 16+20, T, H, W] → 36ch.
                        if "y" in kwargs and kwargs["y"] is not None:
                            y = kwargs["y"]
                            if isinstance(y, list) and len(y) > 0 and y[0] is not None:
                                y_tensor = y[0]
                                if isinstance(y_tensor, _torch.Tensor):
                                    if y_tensor.ndim == 4:
                                        y_tensor = y_tensor.unsqueeze(0)
                                    x = kwargs.get("x")
                                    if x is not None and isinstance(x, _torch.Tensor):
                                        # Match spatial dims if needed
                                        if y_tensor.shape[2:] == x.shape[2:]:
                                            kwargs["x"] = _torch.cat([x, y_tensor], dim=1)
                                        else:
                                            print(f"[ULSWanBridge] ⚠ x/y shape mismatch: "
                                                  f"x={x.shape} y={y_tensor.shape} — skipping concat")
                        # Map clip_fea directly (already in kwargs from kijai)
                        # Strip kwargs Core doesn't accept (module-level whitelist).
                        filtered = {k: v for k, v in kwargs.items()
                                    if k in _CORE_FORWARD_KEYS}
                        # Core's attention code calls
                        # transformer_options["optimized_attention_override"](...)
                        # directly without a None-check, so a None/non-callable
                        # value there crashes Core. Remove such entries — but do
                        # it on a SHALLOW COPY so kijai's live transformer_options
                        # dict is never mutated across sampling steps.
                        tops = filtered.get("transformer_options")
                        if isinstance(tops, dict):
                            bad = [k for k in ("optimized_attention_override",
                                               "attention_mode_override")
                                   if k in tops and not callable(tops[k])]
                            if bad:
                                tops = dict(tops)            # copy-on-write
                                for k in bad:
                                    del tops[k]
                                filtered["transformer_options"] = tops
                        # Call the original class forward directly (bypasses our hook).
                        # A TypeError here is the canonical breakage mode: Core's
                        # WanModel.forward() signature changed and our kwargs no
                        # longer fit. Surface it with an actionable pointer instead
                        # of a bare trace, then re-raise (behaviour unchanged).
                        try:
                            result = _orig_cls_forward(self_dm, *args, **filtered)
                        except TypeError as _sig_err:
                            print(
                                "[ULSWanBridge] ✗ Core forward rejected the bridged "
                                f"kwargs {sorted(filtered.keys())}: {_sig_err}\n"
                                "[ULSWanBridge]   Core's WanModel.forward() signature "
                                "likely changed. Compare comfy/ldm/wan/model.py against "
                                "_CORE_FORWARD_KEYS and the t→timestep rename in this "
                                "adapter (see Bridge-Master 6.2)."
                            )
                            raise
                        # Core returns Tensor; kijai expects ([tensor_4d], None, None)
                        # where the inner tensor has shape [C, T, H, W] — without batch dim.
                        # Kijai later does `noise_pred_in.unsqueeze(0)` to add batch back
                        # before feeding to the scheduler. If we return 5D, kijai's
                        # unsqueeze(0) makes it 6D and downstream geometry quietly
                        # rots — no crash, but wrong/no progress.
                        if isinstance(result, tuple):
                            return result
                        # Strip batch dim if we added it
                        if isinstance(result, _torch.Tensor) and result.ndim == 5 \
                                and result.shape[0] == 1:
                            result = result.squeeze(0)
                        return ([result], None, None)

                    # Bind as instance method so it shadows the class method
                    dm.forward = types.MethodType(_uls_adapted_call, dm)
                    object.__setattr__(dm, "_uls_forward_adapted", True)
                    print(
                        f"[ULSWanBridge]   installed forward-adapter on "
                        f"{dm.__class__.__module__}.{dm.__class__.__name__} "
                        f"(t→timestep, x-list→5D-tensor, result→tuple)"
                    )
                except Exception as e:
                    print(f"[ULSWanBridge] ⚠ could not install forward-adapter: {e}")

        # --- Clone, so other paths using the same MODEL stay clean ------------
        bridged = model.clone()
        return (bridged,)


# ─── Reverse Bridge: WANVIDEOMODEL → MODEL ─────────────────────────────────

class ULSWanBridgeReverse:
    """
    Polyhedron Wan Bridge (Reverse) — WANVIDEOMODEL → MODEL.

    Loads via kijai's WanVideoModelLoader (full optimisations: fp8, Sage
    Attention, Block Swap), then re-labels the patcher as MODEL so the
    Polyhedron LoRA Stack and other classical tools can attach LoRAs.

    Key-remapping: kijai's WanVideoModel exposes state_dict keys without
    the 'diffusion_model.' prefix that Core's LoRA loader expects. This
    bridge installs a key-remap so Core's LoRA patches find their targets.

    After LoRA application the model goes back into kijai's WanVideoSampler
    with all kijai optimisations (Sage Attention, Block Swap) intact.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("WANVIDEOMODEL", {
                    "tooltip": "WANVIDEOMODEL — typically from WanVideoModelLoader. "
                               "All kijai optimisations (fp8, Sage Attention, "
                               "Block Swap) are preserved."
                }),
            },
        }

    RETURN_TYPES  = ("MODEL",)
    RETURN_NAMES  = ("model",)
    FUNCTION      = "bridge"
    CATEGORY      = "Polyhedron/Bridge"
    OUTPUT_NODE   = False

    def bridge(self, model):
        if not isinstance(model, comfy.model_patcher.ModelPatcher):
            raise TypeError(
                "[ULSWanBridgeReverse] Expected a ModelPatcher, got "
                f"{type(model).__name__}."
            )

        kind, mod_name, cls_name = _classify_diffusion_model(model)

        if kind not in ("kijai-wan", "core-wan", "other-wan"):
            print(
                f"[ULSWanBridgeReverse] ⚠ model is {kind} ({mod_name}.{cls_name}) "
                f"— not a Wan model. Passing through without patching."
            )
            return (model.clone(),)

        bridged = model.clone()

        # --- Key-remap so Core's LoRA loader finds kijai's model layers -------
        #
        # Core's LoraLoader matches LoRA keys against the ModelPatcher's
        # get_key_patches() which ultimately calls model.model.state_dict().
        #
        # Core's WAN21 wrapper:   state_dict() keys start with 'diffusion_model.'
        # Kijai's WanVideoModel:  state_dict() keys start directly with 'blocks.'
        #
        # LoRA files trained for Wan are saved in Core format, so their keys
        # have the 'diffusion_model.' prefix. Without remapping, Core's LoRA
        # loader finds 0 matching patches on a kijai model.
        #
        # Fix: install a model_key_remap on the cloned patcher that strips
        # the 'diffusion_model.' prefix when looking up keys in the inner model,
        # and adds it back when reporting keys to the outside world.

        if kind == "kijai-wan":
            self._install_key_remap(bridged)

        print(
            f"[ULSWanBridgeReverse] {kind}  ({mod_name}.{cls_name})  "
            f"→ MODEL (LoRA key-remap {'installed' if kind == 'kijai-wan' else 'not needed'})"
        )
        return (bridged,)

    def _install_key_remap(self, patcher):
        """
        Wrap the inner model's state_dict() so Core's LoRA loader sees
        'diffusion_model.X' keys instead of kijai's bare 'X' keys.

        Core's lora_loader matches LoRA file keys against the model's
        state_dict keys. WAN LoRA files use Core's naming convention with
        'diffusion_model.' prefix; kijai's WanVideoModel exposes bare keys.
        By prepending the prefix at state_dict() time, Core's LoraLoader
        finds its matches and Core's ModelPatcher.add_patches() applies them.
        """
        inner = getattr(patcher, "model", None)
        if inner is None:
            return

        dm = getattr(inner, "diffusion_model", None)
        # Kijai's structure: patcher.model IS the WanVideoModel (no diffusion_model attr)
        # or patcher.model.diffusion_model is the WanVideoModel
        # We need to find the actual nn.Module that has 'blocks'
        wan_module = dm if dm is not None else inner

        if getattr(wan_module, "_uls_reverse_bridge_remapped", False):
            return  # already done

        orig_state_dict = wan_module.__class__.state_dict

        def _remapped_state_dict(self_mod, *args, **kwargs):
            sd = orig_state_dict(self_mod, *args, **kwargs)
            # Add 'diffusion_model.' prefix to all keys
            return {"diffusion_model." + k: v for k, v in sd.items()}

        # Bind on the instance — not the class — so other kijai-loaded
        # models without the bridge are unaffected.
        wan_module.state_dict = types.MethodType(_remapped_state_dict, wan_module)
        object.__setattr__(wan_module, "_uls_reverse_bridge_remapped", True)

        print(
            "[ULSWanBridgeReverse]   installed key-remap: "
            "state_dict() keys now prefixed with 'diffusion_model.'"
        )


