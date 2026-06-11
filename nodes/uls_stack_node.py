"""
Polyhedron LoRA Stack — Backend (v267)
═══════════════════════════════════════
Group-aware LoRA application for ComfyUI.

Two nodes:
  - UltimateLoraStack ("⬡ Polyhedron LoRA Stack")
      Group-ordered application with per-group merge mode.
      Outputs: MODEL, CLIP, debug_info, uls_config_out, trigger_words
  - ULSAccelerator   ("⬡ Polyhedron LoRA Engine")
      Flat list, single global merge mode for engine LoRAs.
      Outputs: MODEL, CLIP, debug_info

Design notes:
  - Universal: no model-type assumptions, works for FLUX / WAN / SDXL / SD1.5.
  - One node = one path. Two stacks side-by-side for dual-noise (HIGH+LOW).
  - Merge modes per group:
      SEQ      sequential native LoraLoader (cached) — default, always works
      CONCAT   rank-concatenation, mathematically identical to SEQ but
               travels through a different float path (potentially slightly
               different float-rounding behaviour worth comparing empirically)
      DARE     CONCAT + Bernoulli mask. Two variants per group:
                 channel — drop entire rank-channels (LoRA-aware)
                 element — drop individual tensor elements (classic paper)
  - Cleanup switches (modifiers beside the four modes, CONCAT/DARE only):
      Trim     drop the weakest rank-channels per LoRA by magnitude before
               merge (deterministic; against the "many quiet LoRAs → interference"
               pile-up). Output stays low-rank.
      Resolve  TIES sign-election across conflicting LoRAs, then a truncated
               SVD re-pack to low-rank so it rejoins the same hand-off.
               Composes after Trim; the DARE mask is skipped under Resolve.
  - DARE variant is per-group (Stack) or global (Engine).
  - DARE seed is deterministic across processes (hashlib.sha1).
  - DARE density auto-scales with group size: 1.0 - 0.05·(n-1), floor 0.5.
  - Schema: rows use `weight` for strength. Legacy `wLow`/`wHigh` are read
    as fallback (auto-migration on read).
"""

import os
import re
import json
import math
import time
import hashlib
import threading

import folder_paths

# Native ComfyUI LoraLoader — has built-in caching via self.loaded_lora.
# Going through this loader is what makes SEQ as efficient as Power LoRA Loader.
try:
    from nodes import LoraLoader as _NativeLoraLoader
except ImportError as e:
    raise ImportError(
        "[PLS] Could not import ComfyUI's native LoraLoader from `nodes`. "
        "This usually means ComfyUI is not on the Python path or the install "
        "is broken. Ensure this addon lives under ComfyUI/custom_nodes/."
    ) from e

import comfy.lora

from collections import OrderedDict


# v265: optional interrupt hook — lets ComfyUI's red X (Cancel) abort a long
# merge promptly instead of only after it finishes. Resolved ONCE at import; if
# a future ComfyUI changes/removes the API, the check degrades to a no-op (no
# crash), consistent with the pack's isolated-failure design. It does NOT touch
# merge math: when not interrupted the call does nothing, so merges stay
# bit-identical. When the user cancels, comfy raises InterruptProcessingException
# which propagates up and ComfyUI handles it as a cancelled run.
class _NeverInterrupt(Exception):
    """Sentinel used as the 'interrupt' type when ComfyUI's API is absent, so
    `except INTERRUPT_EXC` clauses stay inert (this is never raised)."""
    pass

try:
    import comfy.model_management as _mm
    _throw_if_interrupted = _mm.throw_exception_if_processing_interrupted
    # The exception comfy raises on Cancel. Broad excepts in the merge/analysis
    # re-raise THIS so a cancel actually aborts instead of being swallowed into a
    # SEQ fallback or an "analysis failed" line.
    INTERRUPT_EXC = getattr(_mm, "InterruptProcessingException", _NeverInterrupt)

    def _check_interrupt():
        _throw_if_interrupted()
except Exception:
    INTERRUPT_EXC = _NeverInterrupt

    def _check_interrupt():
        pass


# Merge timing (v258): print a per-merge wall-time breakdown (load / trim /
# resolve / other) for CONCAT/DARE groups. ON by default — it only fires for an
# actual multi-LoRA merge (SEQ never reaches this path), and it answers "where
# do the seconds go" while calibrating Trim/Resolve. The clock reads never
# change a merge result; only the optional report is gated. Silence with
# PLS_TIMING=0 (or false/no/off).
_TIMING = os.environ.get("PLS_TIMING", "1").strip().lower() not in ("0", "false", "no", "off")


def _timing_bar(frac: float, width: int = 32) -> str:
    """Proportional ASCII bar for the merge-timing report (frac in 0..1)."""
    frac = 0.0 if frac < 0 else (1.0 if frac > 1 else frac)
    filled = int(round(frac * width))
    return "█" * filled + "░" * (width - filled)


# ─── CONCAT/DARE tensor-dict cache ─────────────────────────────────────────
# IS_CHANGED returns NaN (always re-execute), which is fine for SEQ because
# the native LoraLoader caches the file internally. CONCAT/DARE, however, read
# every safetensors file from disk via comfy.utils.load_torch_file on every
# queue run — expensive for the typical 5–8 LoRA DARE group. This bounded LRU
# caches the raw tensor dicts keyed by (path, mtime, size) so repeated runs
# with unchanged files skip the disk I/O entirely. Execution semantics are
# unchanged — the cache only avoids redundant file reads, it never affects the
# merge result. Bounded by BOTH an entry count AND a byte budget (v251): the
# count covers the realistic case, the byte budget guards the pathological one
# (e.g. many large LoRAs) so CPU-RAM stays actually capped, not just count-capped.
# Oldest entries are evicted first; at least one entry is always kept.
_TD_CACHE = OrderedDict()             # key -> (tensor_dict, nbytes)
_TD_CACHE_MAX = 32                    # hard cap on entries (primary, realistic case)
_TD_CACHE_MAX_BYTES = 4 * 1024 ** 3   # 4 GiB cap on cached CPU tensors (pathological guard)
_TD_CACHE_BYTES = 0                   # running total of cached bytes
# v254: guards the OrderedDict + running byte total so they stay consistent if
# ComfyUI ever executes graphs concurrently. Today's single execution worker is
# unaffected; the lock is uncontended and result-neutral. The slow disk load
# happens OUTSIDE the lock so loads never serialise behind one another.
_TD_CACHE_LOCK = threading.Lock()


def _td_nbytes(td) -> int:
    """Best-effort byte size of a tensor dict (sum of tensor storage sizes).
    Never raises — a value we can't measure simply counts as 0."""
    total = 0
    try:
        for v in td.values():
            try:
                total += v.numel() * v.element_size()
            except Exception:
                pass
    except Exception:
        pass
    return total


def _cached_load_torch_file(path: str):
    """Load a LoRA safetensors tensor dict with a small LRU cache keyed on
    (path, mtime, size). Falls back to a direct load on any stat/IO hiccup.
    Eviction is bounded by entry count and a byte budget (see note above).
    Cache mutations are guarded by _TD_CACHE_LOCK (v254); the disk load runs
    outside the lock so concurrent loads don't serialise behind it."""
    import comfy.utils
    global _TD_CACHE_BYTES
    try:
        st = os.stat(path)
        key = (path, int(st.st_mtime), int(st.st_size))
    except OSError:
        key = None

    # Fast path: serve from cache under the lock.
    if key is not None:
        with _TD_CACHE_LOCK:
            if key in _TD_CACHE:
                _TD_CACHE.move_to_end(key)
                return _TD_CACHE[key][0]

    # Slow path: load OUTSIDE the lock (disk I/O must not serialise behind it).
    td = comfy.utils.load_torch_file(path, safe_load=True)

    if key is not None and td:
        nb = _td_nbytes(td)
        with _TD_CACHE_LOCK:
            # Another thread may have inserted the same key while we loaded;
            # prefer the existing entry and drop our duplicate.
            if key in _TD_CACHE:
                _TD_CACHE.move_to_end(key)
                return _TD_CACHE[key][0]
            _TD_CACHE[key] = (td, nb)
            _TD_CACHE_BYTES += nb
            _TD_CACHE.move_to_end(key)
            # Evict oldest-first by entry count AND byte budget; always keep ≥1 entry
            # so a single oversized LoRA can still be served from cache.
            while len(_TD_CACHE) > _TD_CACHE_MAX or (
                    _TD_CACHE_BYTES > _TD_CACHE_MAX_BYTES and len(_TD_CACHE) > 1):
                _, (_ev_td, ev_nb) = _TD_CACHE.popitem(last=False)
                _TD_CACHE_BYTES -= ev_nb
    return td


# ─── Group Configuration ──────────────────────────────────────────────────
# Application order: broadest first, most specific last.
GROUP_ORDER = ["—", "acc", "style", "scene", "motion", "subject", "detail", "custom"]


# ─── Safetensors Metadata ──────────────────────────────────────────────────

def _read_meta(path: str) -> dict:
    """Read safetensors __metadata__ block. Returns {} on any error."""
    try:
        with open(path, "rb") as f:
            header_bytes = f.read(8)
            if len(header_bytes) < 8:
                return {}
            n = int.from_bytes(header_bytes, "little")
            if n <= 0 or n > 50 * 1024 * 1024:
                return {}
            raw = f.read(n)
            if len(raw) < n:
                return {}
            return json.loads(raw.decode("utf-8", errors="replace")).get("__metadata__", {})
    except Exception:
        return {}


def _flatten_tag_frequency(tw):
    """Normalize ss_tag_frequency into a comma-separated trigger string.

    Handles three real-world formats:
      1. dict of dicts: {"1_du8ne": {"du8ne": 12, "rare_tag": 1}, ...}
      2. flat dict:     {"trigger_a": 5, "trigger_b": 3, ...}
      3. JSON string:   '{"1_du8ne": {"du8ne": 12}}'  (kohya stores it as text)
      4. plain string:  "trigger_a, trigger_b"

    For nested dicts, the OUTER keys are concept-folder labels (e.g. "1_du8ne")
    — we discard those and use the INNER keys, which are the real tags learned
    during training. We pick the inner tag with the highest frequency per
    concept-folder, since that's the canonical trigger for that concept.
    """
    if not tw:
        return ""

    # Format 3: string that's actually JSON
    if isinstance(tw, str):
        s = tw.strip()
        if s.startswith("{"):
            try:
                tw = json.loads(s)
            except (ValueError, json.JSONDecodeError):
                return s   # not valid JSON — return as-is
        else:
            return s       # plain comma-separated string

    if not isinstance(tw, dict):
        return str(tw)

    # Detect nested format (dict of dicts)
    sample_val = next(iter(tw.values()), None)
    if isinstance(sample_val, dict):
        # Format 1: {"1_du8ne": {"du8ne": 12, ...}, ...}
        # Collect highest-frequency tag from each concept-folder.
        triggers = []
        for outer_key, inner in tw.items():
            if not isinstance(inner, dict) or not inner:
                continue
            # Sort inner tags by frequency, take the most frequent
            try:
                top_tag = max(inner.items(), key=lambda kv: kv[1])[0]
            except (TypeError, ValueError):
                top_tag = next(iter(inner.keys()))
            triggers.append(top_tag)
        return ", ".join(triggers[:20])

    # Format 2: flat dict {"trigger": frequency, ...}
    return ", ".join(list(tw.keys())[:20])


def _extract_lora_info(path: str) -> dict:
    meta = _read_meta(path)
    tw_raw = meta.get("ss_tag_frequency", meta.get("trigger_words", ""))
    tw = _flatten_tag_frequency(tw_raw)
    return {
        "trigger_words": tw,
        "base_model":    meta.get("ss_base_model_version",
                         meta.get("modelspec.architecture", "?")),
        "rank":          meta.get("ss_network_dim", "?"),
        "algo":          meta.get("ss_network_module", "lora").split(".")[-1],
        "description":   meta.get("modelspec.description", ""),
        "raw":           meta,
    }


def _path_within_loras(path: str) -> bool:
    """Defense-in-depth (v251): confirm a resolved path lives inside one of the
    configured 'loras' directories. On current ComfyUI, folder_paths.get_full_path
    already enforces containment, so this never fires in normal use — it only
    matters if an older/unpatched get_full_path is in play. Belt-and-suspenders,
    never the primary defense. Never raises; on any doubt it returns False."""
    try:
        rp = os.path.realpath(path)
        for d in folder_paths.get_folder_paths("loras"):
            base = os.path.realpath(d)
            if rp == base or rp.startswith(base + os.sep):
                return True
    except Exception:
        pass
    return False


def _find_preview(lora_name: str) -> dict:
    result = {}
    try:
        path = folder_paths.get_full_path("loras", lora_name)
        if not path or not _path_within_loras(path):
            return result
        base = os.path.splitext(path)[0]
        for ext in [".preview.png", ".preview.jpg", ".preview.jpeg",
                    ".jpg", ".jpeg", ".png"]:
            if os.path.isfile(base + ext):
                result["image"] = base + ext
                break
        for ext in [".preview.mp4", ".preview.gif", ".preview.webm",
                    ".mp4", ".gif", ".webm"]:
            if os.path.isfile(base + ext):
                result["video"] = base + ext
                break
    except Exception:
        pass
    return result


def _read_txt_trigger(lora_name: str) -> str:
    """Read trigger words from .txt file (read-only).
    Supports comma-separated and/or newline-separated values."""
    try:
        path = folder_paths.get_full_path("loras", lora_name)
        if not path or not _path_within_loras(path):
            return ""
        txt_path = os.path.splitext(path)[0] + ".txt"
        if os.path.isfile(txt_path):
            with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read().strip()
            parts = [p.strip() for p in re.split(r"[\n\r,]+", content) if p.strip()]
            return ", ".join(parts)
    except Exception:
        pass
    return ""


# ─── Helpers ───────────────────────────────────────────────────────────────

def _safe_weight(value, default: float = 1.0) -> float:
    try:
        w = float(value)
        if math.isnan(w) or math.isinf(w):
            return default
        return max(-10.0, min(10.0, w))
    except (TypeError, ValueError):
        return default


def _row_weight(row: dict, default: float = 1.0) -> float:
    """Read weight from a row dict, supporting both v097 (`weight`) and legacy
    (`wLow`/`wHigh`) schemas. Picks the first valid value found."""
    for key in ("weight", "wLow", "wHigh"):
        if key in row:
            v = _safe_weight(row.get(key), default=float("nan"))
            if not math.isnan(v):
                return v
    return default


def _row_clip_weight(row: dict, fallback: float) -> float:
    """v302: per-row CLIP strength. Reads optional `wClip`; anything missing or
    non-numeric falls back to the model weight — which makes every pre-v302
    workflow byte-identical (CLIP strength == model strength, as before)."""
    if "wClip" in row:
        v = _safe_weight(row.get("wClip"), default=float("nan"))
        if not math.isnan(v):
            return v
    return fallback


# v302: lora-key prefixes that target the text encoder (kohya `lora_te`,
# `lora_te1/2/3` for SDXL/SD3 duals, diffusers `text_encoder.`, cascade
# `lora_prior_te`). Used to pick the CLIP weight inside the merged build.
# A miss is graceful: an unrecognised TE layer is simply scaled with the
# model weight — i.e. exactly the pre-v302 behaviour, never corruption.
_TE_KEY_PREFIXES = ("lora_te", "text_encoder", "lora_prior_te")


def _is_te_base(base: str) -> bool:
    """True if a lora base key targets the text encoder (CLIP) rather than
    the diffusion model."""
    return str(base).startswith(_TE_KEY_PREFIXES)


def _short_name(lora_name: str, n: int = 38) -> str:
    """Filename without extension, truncated. Cross-platform safe."""
    return os.path.basename(lora_name).replace(".safetensors", "")[:n]


def _dare_density(n: int) -> float:
    """Auto-scale density with group size: 2 LoRAs → 0.95, 5 → 0.80, 10 → 0.55,
    floor at 0.5. Empirically reasonable; not a hard rule."""
    if n <= 1:
        return 1.0
    return max(0.5, 1.0 - 0.05 * (n - 1))


def _trim_keep_fraction(n: int) -> float:
    """Fraction of rank-channels to KEEP for the deterministic magnitude trim
    (the 'Trim' cleanup switch). Gentle, group-scaled — the more LoRAs are
    stacked concurrently, the more aggressively the weak tail is pruned to keep
    background-noise interference out of the merge:
        2 LoRAs → ~0.95, 10 → ~0.73, 15+ → floor 0.60.
    Empirically reasonable starting point; calibrate against real runs."""
    if n <= 1:
        return 1.0
    return max(0.60, min(0.95, 1.0 - 0.03 * (n - 1)))


def _trim_channel_indices(B_f, A_f, keep_fraction: float):
    """Deterministic counterpart to the DARE channel mask: instead of dropping
    *random* rank-channels, drop the *weakest* ones by contribution magnitude.

    The contribution of rank-channel r is the rank-1 outer product
    B[:, r] ⊗ A[r, :]; its Frobenius norm is ‖B[:, r]‖·‖A[r, :]‖. We keep the
    top `keep_fraction` channels by that norm and drop the rest. Operates purely
    in factor space — no full-delta reconstruction, so it is cheap and the
    output stays low-rank (same hand-off path as plain CONCAT/DARE).

    Returns a sorted LongTensor of channel indices to keep, or None when there
    is nothing to trim (keep all)."""
    import torch
    rank = A_f.shape[0]
    if rank <= 1:
        return None
    keep_count = max(1, int(round(rank * keep_fraction)))
    if keep_count >= rank:
        return None
    # B_f: [out, rank, ...] → channel dim is 1 ; A_f: [rank, in, ...] → dim 0.
    b = B_f.transpose(0, 1).reshape(rank, -1)   # [rank, out·…]
    a = A_f.reshape(rank, -1)                    # [rank, in·…]
    mag = b.norm(dim=1) * a.norm(dim=1)          # [rank]
    keep = torch.topk(mag, keep_count, largest=True).indices
    # Sort so concat order is stable and reproducible run-to-run.
    keep, _ = torch.sort(keep)
    return keep.to(torch.long)


def _dare_seed(names: list, weights: list) -> int:
    """Deterministic seed across processes — must NOT use Python's hash()
    because PYTHONHASHSEED is randomised per process by default."""
    h = hashlib.sha1()
    for name, w in zip(names, weights):
        h.update(name.encode("utf-8", errors="replace"))
        h.update(b"|")
        h.update(f"{round(float(w), 4):.4f}".encode("ascii"))
        h.update(b";")
    # 31-bit positive int, fits torch generator
    return int.from_bytes(h.digest()[:4], "big") & 0x7FFFFFFF


def _resolve_pick_device(min_free_gb: float = 3.0) -> str:
    """Pick the device for the RESOLVE math: CUDA when it is available AND has at
    least `min_free_gb` free, otherwise CPU. At merge time the diffusion model is
    not loaded yet, so the GPU is normally idle and CUDA is chosen; the choice is
    therefore stable run-to-run, which keeps a workflow reproducible."""
    import torch
    if not torch.cuda.is_available():
        return "cpu"
    try:
        free, _total = torch.cuda.mem_get_info()
        if free >= int(min_free_gb * (1024 ** 3)):
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _resolve_sign_elect(bs, as_, out_dim: int, in_dim_flat: int,
                        seed: int = 0, device: str = "cpu", use_fp16: bool = False):
    """RESOLVE (v256; GPU path added v259) - TIES sign-election + disjoint merge
    for ONE base layer.

    Each (B_f, A_f) pair is already weight/scale-folded (and, if Trim is on,
    already trimmed). Three steps:
      1. Elect a sign per weight element from the SUM of all per-LoRA deltas.
      2. Disjoint mean: average ONLY the LoRAs whose sign agrees with the elected
         one at that element; the sign-fighting contributions are dropped (this is
         what cuts the "many LoRAs cancel each other out" effect).
      3. Re-pack the resolved full delta as a LOW-RANK LoRA via a truncated
         randomized SVD (rank = sum of input ranks, i.e. CONCAT footprint), so the
         result rejoins the standard load_lora hand-off completely unchanged.

    v259 - device / dtype:
      * `device` selects where the heavy math runs ("cuda" or "cpu").
      * On CUDA the per-source delta matmuls (B @ A) run in fp16 (tensor cores)
        when `use_fp16`; the sign SUM, the disjoint MEAN and the SVD always run in
        fp32 for stability. The sign election only needs the sign, so the fp16
        matmul error is irrelevant there.
      * The resolved low-rank pair is moved back to CPU before return and every
        device temporary is freed, so VRAM stays flat across all layers.
      * CUDA OOM is intentionally NOT caught here - it propagates so the caller
        can retry the whole merge on CPU.

    v259 - single pass: each source delta is built ONCE and reused for both the
    sign election and the disjoint mean (v258 rebuilt it twice to cap memory). On
    the CPU-fp32 path this is bit-identical to v258 (same ops, same order, same
    seed), so CPU stays a byte-exact fallback for the GPU path. The SVD is seeded
    deterministically (RNG state saved/restored) so the merge is reproducible
    run-to-run on a given device, like the DARE path.

    Returns (B_merged, A_merged) on CPU, reshaped to the layer's factor shapes,
    None on full cancellation, or raises on a shape it cannot represent (caller
    falls back to SEQ) or on CUDA OOM (caller retries on CPU).
    """
    import torch

    tgt = torch.device(device)
    mm_dtype = torch.float16 if (use_fp16 and tgt.type == "cuda") else torch.float32

    n = len(bs)
    ranks = [b.shape[1] for b in bs]
    sum_rank = int(sum(ranks))
    b_trail = list(bs[0].shape[2:])     # up trailing dims (linear: []; conv: [1,1])
    a_trail = list(as_[0].shape[1:])    # down trailing dims ([in] or [in, kh, kw])

    # Build each source delta ONCE, on the target device, in the matmul dtype.
    deltas = []
    for i in range(n):
        B2 = bs[i].reshape(out_dim, ranks[i]).to(tgt, dtype=mm_dtype)
        A2 = as_[i].reshape(ranks[i], in_dim_flat).to(tgt, dtype=mm_dtype)
        deltas.append(B2 @ A2)          # [out, in_flat] in mm_dtype on tgt
        del B2, A2

    # Pass 1 - elected sign from the running sum (fp32 accumulation).
    S = torch.zeros(out_dim, in_dim_flat, dtype=torch.float32, device=tgt)
    for W in deltas:
        S += W.float()
    gamma = torch.sign(S)               # {-1, 0, +1}, fp32
    del S

    # Pass 2 - disjoint mean of the sign-agreeing contributions (fp32).
    num = torch.zeros(out_dim, in_dim_flat, dtype=torch.float32, device=tgt)
    den = torch.zeros(out_dim, in_dim_flat, dtype=torch.float32, device=tgt)
    for W in deltas:
        Wf = W.float()
        agree = (torch.sign(Wf) == gamma) & (gamma != 0)
        num += torch.where(agree, Wf, torch.zeros_like(Wf))
        den += agree.to(torch.float32)
        del Wf, agree
    deltas.clear()
    del gamma
    W_merged = num / den.clamp(min=1.0)
    del num, den

    if (not torch.isfinite(W_merged).all()) or float(W_merged.abs().max()) == 0.0:
        del W_merged
        return None                     # full cancellation -> no patch for this base

    # Re-pack low-rank: rank = sum of input ranks (CONCAT parity), capped by dims.
    min_dim = min(out_dim, in_dim_flat)
    r = max(1, min(sum_rank, min_dim))
    q = min(min_dim, r + 8)             # small oversample for SVD accuracy
    cpu_state = torch.get_rng_state()   # keep the merge reproducible
    # v267 (N-7): manual_seed below reseeds CPU *and all CUDA devices*. Save/
    # restore the CUDA RNG state whenever CUDA is initialised — also on the
    # CPU path of a CUDA machine (the OOM→CPU-retry case) — so the merge
    # never perturbs the global CUDA RNG. Output-identical: state save and
    # restore only, no tensor op changes (test_v259 [A] stays bit-identical).
    cuda_state = (torch.cuda.get_rng_state_all()
                  if (torch.cuda.is_available() and torch.cuda.is_initialized())
                  else None)
    try:
        torch.manual_seed(seed & 0x7FFFFFFF)   # seeds CPU + all CUDA devices
        U, Sv, V = torch.svd_lowrank(W_merged, q=q, niter=4)
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
    del W_merged

    U = U[:, :r].contiguous()
    Sv = Sv[:r]
    V = V[:, :r].contiguous()
    B_merged = (U * Sv.unsqueeze(0)).reshape([out_dim, r] + b_trail).contiguous()
    A_merged = V.transpose(0, 1).reshape([r] + a_trail).contiguous()

    # Back to CPU for the standard hand-off; free the device copies.
    B_cpu = B_merged.to("cpu", dtype=torch.float32).contiguous()
    A_cpu = A_merged.to("cpu", dtype=torch.float32).contiguous()
    del B_merged, A_merged, U, Sv, V
    return B_cpu, A_cpu


# ═══ Group Apply Modes ════════════════════════════════════════════════════
#
# A group with 2+ LoRAs can be applied in three different ways:
#
#   SEQ      — sequentially patch each LoRA via the cached native loader.
#              Always correct. Default.
#
#   CONCAT   — concatenate the lora_A / lora_B factors of all LoRAs in the
#              group along the rank dimension, then apply ONCE. Mathematically:
#                  [B1·w1; B2·w2] @ [A1; A2]ᵀ  =  w1·B1·A1 + w2·B2·A2
#              Same delta as SEQ in exact arithmetic. Float rounding paths
#              differ, so empirical bit-for-bit equivalence is NOT guaranteed.
#
#   DARE     — like CONCAT but with a Bernoulli mask on B before concatenation.
#              Surviving entries are rescaled by 1/density to preserve the
#              per-LoRA expectation. Two variants:
#                  channel — drop entire rank-channels
#                            (LoRA-aware: reduces channel-overlap between
#                             concurrently-active LoRAs in the same group)
#                  element — drop individual tensor elements
#                            (classic DARE paper behaviour)
#
# Naming conventions handled (model-agnostic):
#   - Kohya / SD / SDXL:        lora_up.weight   / lora_down.weight
#   - WAN / FLUX / HunyuanVideo: lora_B.weight   / lora_A.weight
# ════════════════════════════════════════════════════════════════════════════

_LORA_CONVENTIONS = [
    (".lora_up.weight",   ".lora_down.weight"),   # Kohya / SD / SDXL
    (".lora_B.weight",    ".lora_A.weight"),       # WAN / FLUX / HunyuanVideo
]


def _detect_convention(td: dict):
    """Return (up_suffix, down_suffix) of the dominant convention, or None
    if the tensor dict matches no known LoRA layout (e.g. LoHA / LoKr)."""
    best = None
    best_count = 0
    for conv in _LORA_CONVENTIONS:
        up_suffix = conv[0]
        cnt = sum(1 for k in td if k.endswith(up_suffix))
        if cnt > best_count:
            best_count = cnt
            best = conv
    return best


def _collect_factor_keys(td: dict, conv: tuple) -> list:
    """Triples (base, up_key, down_key, alpha_key_or_None) for this LoRA."""
    up_suffix, down_suffix = conv
    triples = []
    for k in td:
        if not k.endswith(up_suffix):
            continue
        base = k[: -len(up_suffix)]
        dk = base + down_suffix
        if dk not in td:
            continue
        ak_found = None
        for ak in [base + ".alpha", base + ".lora_alpha"]:
            if ak in td:
                ak_found = ak
                break
        triples.append((base, k, dk, ak_found))
    return triples


def _has_mid_tensor(td: dict) -> bool:
    """True if the LoRA carries a CP/Tucker 'mid' tensor (LoCon / conv LoRA),
    i.e. a '<base>.lora_mid.weight' key. Our CONCAT/DARE path reconstructs the
    layer delta as up @ down only; a present mid makes the true delta
    up · mid · down, which this path does NOT represent — and convention
    detection still matches on lora_up/lora_B, so without this guard the group
    would NOT fall back. A group containing any such LoRA is routed to SEQ,
    whose native loader applies the mid correctly. Detection is by key suffix
    only (no tensor is read), so it is cheap and never touches a merge result.
    (This is exactly the mid key ComfyUI's own load_lora maps, so SEQ handles
    it natively.)"""
    return any(k.endswith(".lora_mid.weight") for k in td)


def _get_clip_model(clip):
    """Return the inner CLIP model object across ComfyUI versions, or None."""
    if clip is None:
        return None
    m = getattr(clip, "cond_stage_model", None)
    if m is not None:
        return m
    patcher = getattr(clip, "patcher", None)
    if patcher is not None:
        return getattr(patcher, "model", None)
    return None


# ─── Apply: SEQ ────────────────────────────────────────────────────────────

def _apply_seq(loader, model, clip, names: list, weights: list,
               clip_weights: list = None) -> tuple:
    """Sequentially apply each LoRA via the cached native loader.
    v302: optional per-LoRA CLIP strength (None → CLIP follows model weight,
    the pre-v302 behaviour). Returns (model, clip, [error_strings])."""
    if clip_weights is None:
        clip_weights = list(weights)
    m, c = model, clip
    errors = []
    for name, w, wc in zip(names, weights, clip_weights):
        if (abs(w) < 1e-6 and abs(wc) < 1e-6) or not name or name == "None":
            continue
        path = folder_paths.get_full_path("loras", name)
        if not path:
            msg = f"⚠ LoRA not found: {name}"
            print(f"[PLS] {msg}")
            errors.append(msg)
            continue
        try:
            m, c = loader.load_lora(m, c, name, w, wc)
        except Exception as ex:
            short = _short_name(name)
            msg = f"✗ Skipped (incompatible): {short}"
            print(f"[PLS] {msg}: {ex}")
            errors.append(msg)
    return m, c, errors


# ─── Apply: CONCAT / DARE ──────────────────────────────────────────────────

def _apply_concat_or_dare(loader, model, clip, names: list, weights: list,
                          mode: str, dare_variant: str = "channel",
                          trim: bool = False, resolve: bool = False,
                          trim_amount: float = None,
                          force_resolve_device: str = None,
                          clip_weights: list = None) -> tuple:
    """
    Build a synthetic merged LoRA tensor dict by concatenating B/A factors
    along the rank dimension, then hand it to ComfyUI's standard load_lora()
    pipeline. CONCAT skips the mask; DARE applies a Bernoulli mask first.

    Cleanup switches (v250):
      trim    — drop the weakest rank-channels per LoRA by magnitude before
                concatenation (deterministic; reduces the background-noise
                pile-up of many concurrently-stacked LoRAs).
      resolve — TIES sign-election across conflicting LoRAs (v256). Elects the
                dominant sign per weight element and averages only the agreeing
                LoRAs, then re-packs the resolved delta as a low-rank LoRA via a
                truncated SVD so it uses the same hand-off. Composes after Trim;
                the DARE mask is not applied when resolve is on (resolve IS the
                merge). Falls back to SEQ on any layer it can't represent.

    Falls back to SEQ on any structural failure (LyCORIS / conv-mid /
    shape mismatch / empty merge).
    """
    import comfy.utils
    import torch

    # Merge-timing accumulators (v258). Reading the clock never changes the
    # merge result; only the optional report at the end is gated on _TIMING.
    _t_start = time.perf_counter()
    _t_load = _t_trim = _t_resolve = 0.0
    _resolve_layers = 0

    mode = mode.upper()
    dare_variant = (dare_variant or "channel").lower()
    if dare_variant not in ("channel", "element"):
        dare_variant = "channel"

    # resolve (TIES sign-election) is handled per base layer below, in place of
    # the plain rank-concat. See _resolve_sign_elect. Honest report after the loop.

    # --- Load all tensor dicts ---
    # v302: clip_weights rides along through the same validity filter so the
    # three lists stay index-aligned. None → CLIP follows the model weight.
    if clip_weights is None:
        clip_weights = list(weights)
    _t0 = time.perf_counter()
    raw, valid_names, valid_weights, valid_clip_weights = [], [], [], []
    for name, w, wc in zip(names, weights, clip_weights):
        _check_interrupt()                     # v265: red X (Cancel) aborts during loading
        if (abs(w) < 1e-6 and abs(wc) < 1e-6) or not name or name == "None":
            continue
        path = folder_paths.get_full_path("loras", name)
        if not path:
            print(f"[PLS] ⚠ Not found: {name}")
            continue
        try:
            td = _cached_load_torch_file(path)
            if td:
                raw.append(td)
                valid_names.append(name)
                valid_weights.append(float(w))
                valid_clip_weights.append(float(wc))
        except Exception as ex:
            print(f"[PLS] ✗ Load failed: {name}: {ex}")
    _t_load += time.perf_counter() - _t0

    if len(raw) < 2:
        # 0 or 1 valid → fall back to SEQ
        return _apply_seq(loader, model, clip, valid_names, valid_weights, valid_clip_weights)

    # --- Detect convention per LoRA ---
    convs = [_detect_convention(td) for td in raw]
    unrecognised = [valid_names[i] for i, c in enumerate(convs) if c is None]
    if unrecognised:
        print(f"[PLS] ⚠ {mode}: {len(unrecognised)} LoRA(s) use non-standard "
              f"format (LyCORIS/LoHA/LoKr?), falling back to SEQ:")
        for n in unrecognised:
            print(f"[PLS]      - {_short_name(n)}")
        return _apply_seq(loader, model, clip, valid_names, valid_weights, valid_clip_weights)

    # Use the first LoRA's convention as the output naming.
    out_up_suffix, out_down_suffix = convs[0]

    # All LoRAs in ONE merge group must share ONE naming convention. Mixing
    # kohya (.lora_up/.lora_down) with WAN/FLUX (.lora_B/.lora_A) here would
    # re-suffix bases collected under one convention with the OTHER's output
    # suffix → unmappable keys. The dangerous case is partial mapping: if some
    # keys still resolve, the non-matching LoRAs get silently dropped from the
    # merge while the report claims success. SEQ applies each LoRA under its
    # own convention, so it is the correct, safe path for a mixed group.
    if len({c for c in convs}) > 1:
        print(f"[PLS] ⚠ {mode}: group mixes LoRA naming conventions "
              f"(kohya vs WAN/FLUX) — falling back to SEQ so each LoRA is "
              f"applied correctly under its own convention.")
        return _apply_seq(loader, model, clip, valid_names, valid_weights, valid_clip_weights)

    # --- Conv/LoCon CP-decomposition guard (v253) ---
    # A LoRA carrying a 'mid' tensor has layer delta up · mid · down, but the
    # concat path reconstructs up @ down only. Concatenating up/down while
    # dropping mid would silently produce a wrong delta for those layers — and
    # convention detection still matches on lora_up/lora_B, so the group would
    # NOT otherwise fall back. Route the whole group to SEQ (native loader is
    # mid-aware). Result-neutral for linear LoRAs (no mid → guard never fires).
    mid_loras = [valid_names[i] for i, td in enumerate(raw) if _has_mid_tensor(td)]
    if mid_loras:
        print(f"[PLS] ⚠ {mode}: {len(mid_loras)} LoRA(s) carry a conv 'mid' "
              f"tensor (LoCon/CP) the concat path can't represent — falling "
              f"back to SEQ so each is applied correctly:")
        for n in mid_loras:
            print(f"[PLS]      - {_short_name(n)}")
        return _apply_seq(loader, model, clip, valid_names, valid_weights, valid_clip_weights)

    # --- Per-LoRA: enumerate (base, up_key, down_key, alpha_key) ---
    per_lora_keys = [_collect_factor_keys(td, conv) for td, conv in zip(raw, convs)]

    # --- Group keys by their base name across LoRAs ---
    base_to_sources = {}   # base_name → [(lora_idx, base, up_key, down_key, alpha_key), …]
    for li, triples in enumerate(per_lora_keys):
        for base, uk, dk, ak in triples:
            base_to_sources.setdefault(base, []).append((li, base, uk, dk, ak))

    if not base_to_sources:
        print(f"[PLS] ⚠ {mode}: no factor keys found, falling back to SEQ")
        return _apply_seq(loader, model, clip, valid_names, valid_weights, valid_clip_weights)

    # --- Build synthetic merged tensor dict ---
    # resolve takes over the merge (sign-election), so the DARE mask is NOT
    # applied when resolve is on — resolve IS the merge strategy.
    use_mask = (mode == "DARE") and (not resolve)
    n_active = len(valid_names)
    density  = _dare_density(n_active)
    # v261: per-group Trim strength. `trim_amount` is a kept-fraction (0.5–1.0).
    # When set it OVERRIDES the auto group-size formula; None keeps the v260 auto
    # behaviour (bit-identical). Clamped so a stray value can't gut the LoRA.
    if not trim:
        trim_keep = 1.0
    elif trim_amount is not None:
        trim_keep = float(max(0.5, min(1.0, trim_amount)))
    else:
        trim_keep = _trim_keep_fraction(n_active)
    resolve_seed = _dare_seed(valid_names, valid_weights) if resolve else 0
    # v259: pick the RESOLVE compute device once per merge so the path is
    # stable (stable path -> reproducible result). CPU fallback is rare + loud.
    _resolve_dev = "cpu"
    _resolve_fp16 = False
    if resolve:
        _resolve_dev = force_resolve_device or _resolve_pick_device()
        _resolve_fp16 = (_resolve_dev == "cuda")
        _dev_note = (" (fp16 matmuls, fp32 elect+SVD)" if _resolve_fp16
                     else (" (fp32, CPU fallback)" if force_resolve_device
                           else " (fp32)"))
        print(f"[PLS]   RESOLVE device: {_resolve_dev}{_dev_note}")
    rng = None
    if use_mask:
        seed = _dare_seed(valid_names, valid_weights)
        rng = torch.Generator(device="cpu").manual_seed(seed)
        print(f"[PLS]   DARE: variant={dare_variant}  density={density:.3f}  "
              f"n={n_active}  seed={seed}")
    if trim:
        print(f"[PLS]   TRIM: keep_fraction={trim_keep:.3f}  n={n_active}  "
              f"(dropping weakest rank-channels, deterministic)")

    merged_td = {}
    skipped_shape_mismatch = 0
    alpha_missing_count = 0
    trim_channels_kept = 0
    trim_channels_total = 0
    _n_bases = len(base_to_sources)   # v260: denominator for the live RESOLVE progress line

    for base, sources in base_to_sources.items():
        _check_interrupt()                     # v265: red X (Cancel) aborts during the merge
        bs, as_ = [], []
        out_dim = None
        in_dim_flat = None
        ref_dtype = None

        for (li, _, uk, dk, ak) in sources:
            td = raw[li]
            try:
                B = td[uk].cpu().contiguous()    # [out, rank, ...]
                A = td[dk].cpu().contiguous()    # [rank, in,  ...]
            except Exception:
                continue

            if ref_dtype is None:
                ref_dtype = B.dtype

            B_out_dim = B.shape[0]
            A_in_dim_flat = 1
            for s in A.shape[1:]:
                A_in_dim_flat *= s

            # Cross-LoRA shape check — out_dim and in_dim must match,
            # only the rank may differ (which is the whole point of concat).
            if out_dim is None:
                out_dim = B_out_dim
                in_dim_flat = A_in_dim_flat
            elif B_out_dim != out_dim or A_in_dim_flat != in_dim_flat:
                skipped_shape_mismatch += 1
                continue

            # alpha / rank scale
            rank = A.shape[0]
            alpha_val = None
            if ak is not None:
                try:
                    alpha_val = td[ak].item() if hasattr(td[ak], "item") else float(td[ak])
                except Exception:
                    pass
            if alpha_val is None:
                alpha_missing_count += 1
                scale = 1.0
            else:
                scale = float(alpha_val / rank) if rank > 0 else 1.0

            # v302: text-encoder layers are scaled with the per-LoRA CLIP
            # weight; everything else with the model weight. With wClip unset
            # both are equal → bit-identical to pre-v302. The DARE/RESOLVE
            # seed stays derived from the model weights only, so existing
            # WAN workflows keep their exact masks.
            w = valid_clip_weights[li] if _is_te_base(base) else valid_weights[li]

            # float32 for arithmetic, back to original dtype at end.
            B_f = B.float()
            A_f = A.float()

            # Fold (w * scale) into B once: delta = (w·scale·B) @ A
            B_f = B_f * (w * scale)

            # TRIM (v250): deterministically drop the weakest rank-channels.
            # Done in factor space, BEFORE the (random) DARE mask, so the two
            # switches compose cleanly: trim keeps the strongest channels,
            # DARE may then still thin the survivors. Output stays low-rank.
            trim_channels_total += rank
            if trim:
                _t1 = time.perf_counter()
                keep_idx = _trim_channel_indices(B_f, A_f, trim_keep)
                if keep_idx is not None and keep_idx.numel() < rank:
                    B_f = B_f.index_select(1, keep_idx).contiguous()
                    A_f = A_f.index_select(0, keep_idx).contiguous()
                    rank = A_f.shape[0]   # downstream DARE mask uses trimmed rank
                _t_trim += time.perf_counter() - _t1
            trim_channels_kept += rank

            # DARE mask
            if use_mask and 0.0 < density < 1.0:
                if dare_variant == "channel":
                    # mask shape [1, rank, 1, 1, ...] — drops entire rank channels
                    mask_1d = torch.bernoulli(torch.full((rank,), density), generator=rng)
                    mask_shape = [1, rank] + [1] * (B_f.dim() - 2)
                    mask = mask_1d.view(mask_shape)
                else:
                    # element-wise (classic DARE)
                    mask = torch.bernoulli(torch.full_like(B_f, density), generator=rng)
                B_f = (B_f * mask) / density

            bs.append(B_f.to(ref_dtype))
            as_.append(A_f.to(ref_dtype))

        if not bs:
            continue

        # Combine the per-LoRA factors into the stored low-rank pair.
        if resolve:
            # RESOLVE (v256): TIES sign-election + disjoint merge, re-packed
            # low-rank. v259: runs on _resolve_dev (cuda/fp16 when available).
            try:
                _t2 = time.perf_counter()
                res = _resolve_sign_elect(bs, as_, out_dim, in_dim_flat,
                                          seed=resolve_seed,
                                          device=_resolve_dev, use_fp16=_resolve_fp16)
                _dt = time.perf_counter() - _t2          # was inline; numerically identical
                _t_resolve += _dt
                _resolve_layers += 1
                # v260: throttled live progress so a long merge is visibly moving and a
                # CPU fallback is spotted immediately. Diagnostic ONLY - no tensor math is
                # touched, so the merge result stays bit-identical to v259. flush=True
                # forces the line out DURING the loop instead of buffering it to the end.
                if _resolve_layers == 1 or _resolve_layers % 25 == 0:
                    _vram = ""
                    if _resolve_dev == "cuda":
                        try:
                            _free, _ = torch.cuda.mem_get_info()
                            _vram = f"  free={_free / (1024 ** 3):.1f}G"
                        except Exception:
                            pass
                    print(f"[PLS]   RESOLVE {_resolve_layers}/{_n_bases}  "
                          f"{_resolve_dev}{'/fp16' if _resolve_fp16 else ''}  "
                          f"layer={_dt:.2f}s  cum={_t_resolve:.1f}s{_vram}", flush=True)
            except RuntimeError as ex:
                if _resolve_dev == "cuda" and "out of memory" in str(ex).lower():
                    print(f"[PLS] ⚠ RESOLVE: CUDA out of memory on layer "
                          f"'{base}' - retrying the WHOLE merge on CPU (slower; "
                          f"result is the CPU variant, NOT identical to GPU runs).")
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    return _apply_concat_or_dare(loader, model, clip, names, weights,
                                                 mode, dare_variant, trim, resolve,
                                                 trim_amount=trim_amount,
                                                 force_resolve_device="cpu",
                                                 clip_weights=clip_weights)
                print(f"[PLS] ⚠ RESOLVE: layer '{base}' could not be sign-elected "
                      f"({ex}) - falling back to SEQ for the whole group.")
                return _apply_seq(loader, model, clip, valid_names, valid_weights, valid_clip_weights)
            except Exception as ex:
                print(f"[PLS] ⚠ RESOLVE: layer '{base}' could not be sign-elected "
                      f"({ex}) - falling back to SEQ for the whole group.")
                return _apply_seq(loader, model, clip, valid_names, valid_weights, valid_clip_weights)
            if res is None:
                continue   # full cancellation at this layer → no patch
            B_concat, A_concat = res
        else:
            # Concatenate along the RANK dimension (CONCAT / DARE).
            try:
                B_concat = torch.cat(bs,  dim=1)
                A_concat = torch.cat(as_, dim=0)
            except Exception as ex:
                print(f"[PLS]   skip {base}: concat failed ({ex})")
                continue

        merged_td[base + out_up_suffix]   = B_concat
        merged_td[base + out_down_suffix] = A_concat
        # alpha = rank → scale = 1.0 (we already folded scale + weight into B).
        # Explicit float32 so add_patches treats this as a clean scalar.
        merged_td[base + ".alpha"]        = torch.tensor(float(B_concat.shape[1]),
                                                          dtype=torch.float32)

    if not merged_td:
        print(f"[PLS] ⚠ {mode}: merged dict empty, falling back to SEQ")
        return _apply_seq(loader, model, clip, valid_names, valid_weights, valid_clip_weights)

    if skipped_shape_mismatch:
        print(f"[PLS]   {mode}: skipped {skipped_shape_mismatch} shape-mismatched layer source(s)")
    if alpha_missing_count:
        print(f"[PLS]   {mode}: {alpha_missing_count} layer(s) had no alpha → assumed scale=1.0")
    if trim and trim_channels_total > 0:
        dropped = trim_channels_total - trim_channels_kept
        print(f"[PLS]   TRIM: kept {trim_channels_kept}/{trim_channels_total} "
              f"rank-channels (dropped {dropped} weakest across all sources)")
    if resolve:
        print(f"[PLS]   RESOLVE: sign-election + disjoint merge over {n_active} "
              f"LoRAs{' (after trim)' if trim else ''}; resolved delta re-packed "
              f"low-rank per layer")

    # --- Hand off to ComfyUI's standard LoRA pipeline ---
    try:
        model_keymap = comfy.lora.model_lora_keys_unet(model.model, {})

        clip_model = _get_clip_model(clip)
        clip_keymap = (comfy.lora.model_lora_keys_clip(clip_model, {})
                       if clip_model is not None else {})

        full_keymap = {**model_keymap, **clip_keymap}

        loaded = comfy.lora.load_lora(merged_td, full_keymap)
        if not loaded:
            print(f"[PLS] ⚠ {mode}: ComfyUI mapped 0 patches, falling back to SEQ")
            return _apply_seq(loader, model, clip, valid_names, valid_weights, valid_clip_weights)

        new_model = model.clone()
        new_model.add_patches(loaded, 1.0, 1.0)

        new_clip = clip
        if clip is not None and clip_keymap:
            clip_target_keys = set(clip_keymap.values())
            clip_loaded = {k: v for k, v in loaded.items() if k in clip_target_keys}
            if clip_loaded:
                new_clip = clip.clone()
                new_clip.add_patches(clip_loaded, 1.0, 1.0)

        shorts = [_short_name(n, 18) for n in valid_names]
        mode_tag = mode + (" +TRIM" if trim else "") + (" +RESOLVE" if resolve else "")
        print(f"[PLS] ✓ {mode_tag} merged {n_active} LoRAs [{', '.join(shorts)}]  "
              f"layers={len(merged_td)//3}  patches={len(loaded)}")
        if _TIMING:
            _t_total = time.perf_counter() - _t_start
            _t_other = _t_total - _t_load - _t_trim - _t_resolve
            if _t_other < 0:
                _t_other = 0.0
            _layers = len(merged_td) // 3
            _denom = _t_total if _t_total > 1e-9 else 1.0
            _rows = [
                ("load",    _t_load,    "safetensors read (cached on re-run)"),
                ("trim",    _t_trim,    "magnitude top-k" if trim else ""),
                ("resolve", _t_resolve, (f"TIES + SVD x{_resolve_layers} layers [{_resolve_dev}{'/fp16' if _resolve_fp16 else ''}]" if resolve else "")),
                ("other",   _t_other,   "concat + DARE mask + hand-off"),
            ]
            print(f"[PLS]   ⏱ merge timing  {mode_tag}  n={n_active}, {_layers} layers "
                  f"— CPU, once (before the sampler)")
            for _label, _sec, _note in _rows:
                _frac = _sec / _denom
                _suffix = f"   {_note}" if _note else ""
                print(f"[PLS]       {_label:<8}{_sec:7.2f}s  {_timing_bar(_frac)}  "
                      f"{_frac * 100:4.0f} %{_suffix}")
            print(f"[PLS]       {'─' * 52}")
            print(f"[PLS]       {'total':<8}{_t_total:7.2f}s")
        return new_model, new_clip, []

    except INTERRUPT_EXC:
        raise                          # v265: let a Cancel (red X) abort; don't swallow into SEQ
    except Exception as ex:
        print(f"[PLS] ✗ {mode} apply failed ({ex}), falling back to SEQ")
        import traceback; traceback.print_exc()
        return _apply_seq(loader, model, clip, valid_names, valid_weights, valid_clip_weights)


# ─── Unified Apply Helper ──────────────────────────────────────────────────

def apply_lora_set(loader, model, clip, names: list, weights: list,
                   mode: str = "SEQ", dare_variant: str = "channel",
                   trim: bool = False, resolve: bool = False,
                   trim_amount: float = None,
                   clip_weights: list = None) -> tuple:
    """
    THE unified apply helper. Used by both Stack (per group) and Engine.

    - mode: "SEQ" | "CONCAT" | "DARE"  (case-insensitive, unknown→SEQ)
    - dare_variant: "channel" | "element"  (only used when mode=DARE)
    - trim / resolve: cleanup switches (v250). Only meaningful for CONCAT/DARE —
      SEQ never sees them (LoRAs are never side-by-side under SEQ), which is
      exactly why the UI greys them out for SEQ.
    - Single LoRA always uses SEQ regardless of mode (mode would be a no-op).

    Returns (model, clip, [error_strings]).
    """
    if not names:
        return model, clip, []

    mode = (mode or "SEQ").upper()
    if mode not in ("SEQ", "CONCAT", "DARE"):
        mode = "SEQ"

    # v302: clip strengths ride along (None → CLIP follows model weight).
    if clip_weights is None:
        clip_weights = list(weights)

    # Filter out empties / zero weights up front. A row survives if EITHER
    # strength is non-zero (model 0 + clip 0.8 is a valid CLIP-only row).
    triples = [(n, w, wc) for n, w, wc in zip(names, weights, clip_weights)
               if n and n != "None"
               and (abs(float(w)) >= 1e-6 or abs(float(wc)) >= 1e-6)]
    if not triples:
        return model, clip, []

    f_names   = [t[0] for t in triples]
    f_weights = [t[1] for t in triples]
    f_clip    = [t[2] for t in triples]

    if len(f_names) == 1 or mode == "SEQ":
        return _apply_seq(loader, model, clip, f_names, f_weights, f_clip)

    return _apply_concat_or_dare(loader, model, clip, f_names, f_weights,
                                  mode=mode, dare_variant=dare_variant,
                                  trim=trim, resolve=resolve, trim_amount=trim_amount,
                                  clip_weights=f_clip)


# ─── Trigger Words ─────────────────────────────────────────────────────────

# ─── .uls-meta.json — canonical location + legacy migration ────────────────
#
# Companion metadata (user-curated trigger words, civitai ids, …) lives in a
# JSON file next to the .safetensors. Historically TWO different paths were
# written by different code paths, which silently desynced the data:
#
#   canonical : foo.uls-meta.json            (matches the .txt / .jpg companion
#                                              convention — splitext base)
#   legacy    : foo.safetensors.uls-meta.json (older Civitai-fetch builds)
#
# The frontend overlay's "Save triggers" wrote the canonical name, but the
# Stack backend only ever read the legacy name — so overlay-saved triggers
# were invisible at generation time. These helpers are the single source of
# truth: read tolerates both (canonical wins), write always emits canonical
# and folds-in + removes any legacy file so the two locations converge.

def _uls_meta_path_canonical(full_path: str) -> str:
    """foo.safetensors → foo.uls-meta.json (companion-file convention)."""
    return os.path.splitext(full_path)[0] + ".uls-meta.json"


def _uls_meta_path_legacy(full_path: str) -> str:
    """foo.safetensors → foo.safetensors.uls-meta.json (older builds)."""
    return full_path + ".uls-meta.json"


def _uls_meta_read(full_path: str) -> dict:
    """Read companion metadata, tolerating both historical locations.
    Canonical overrides legacy on key conflicts. Returns {} on any error."""
    data = {}
    if not full_path:
        return data
    # Legacy first, then canonical, so canonical values win on .update().
    for p in (_uls_meta_path_legacy(full_path), _uls_meta_path_canonical(full_path)):
        try:
            if os.path.isfile(p):
                with open(p, "r", encoding="utf-8", errors="replace") as f:
                    d = json.load(f)
                if isinstance(d, dict):
                    data.update(d)
        except Exception:
            pass
    return data


def _uls_meta_write(full_path: str, updates: dict) -> str:
    """Merge `updates` into the existing companion metadata and write the
    result to the canonical path. Any legacy file is folded in first, then
    removed, so the two historical locations converge to one. Returns the
    path written, or "" on failure."""
    if not full_path:
        return ""
    canonical = _uls_meta_path_canonical(full_path)
    legacy    = _uls_meta_path_legacy(full_path)
    merged = _uls_meta_read(full_path)   # existing canonical + legacy, merged
    merged.update(updates or {})
    try:
        with open(canonical, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[PLS] ⚠ Failed to write {os.path.basename(canonical)}: {e}")
        return ""
    # Retire the legacy file now that its content is safely in canonical.
    if legacy != canonical and os.path.isfile(legacy):
        try:
            os.remove(legacy)
        except Exception:
            pass
    return canonical


def _read_uls_meta_trigger(lora_name: str) -> str:
    """Read trigger words from .uls-meta.json next to the LoRA file.
    This is where Save-Triggers writes user-curated trigger words from the
    frontend overlay — highest priority because it's user intent."""
    try:
        path = folder_paths.get_full_path("loras", lora_name)
        if not path:
            return ""
        tw = _uls_meta_read(path).get("trigger_words", "")
        if isinstance(tw, str):
            return tw.strip()
    except Exception:
        pass
    return ""


def _get_trigger(lora_name: str) -> str:
    """All trigger words for a LoRA as comma-separated string.
    Priority: .uls-meta.json (user-curated) → .txt (read-only) →
              safetensors header → filename fallback."""
    # 1. User-curated trigger words from the frontend overlay
    tw = _read_uls_meta_trigger(lora_name)
    if tw:
        return tw
    # 2. Companion .txt file (often shipped with Civitai LoRAs)
    tw = _read_txt_trigger(lora_name)
    if tw:
        return tw
    # 3. Embedded safetensors metadata (ss_tag_frequency / trigger_words)
    path = folder_paths.get_full_path("loras", lora_name)
    if path:
        info = _extract_lora_info(path)
        tw = info.get("trigger_words", "")
        if tw:
            return tw
    # 4. Last resort — derive from filename
    base = os.path.basename(lora_name).replace(".safetensors", "")
    cleaned = re.sub(r'_(high|low|hd|ld)_noise$', '', base, flags=re.IGNORECASE)
    cleaned = re.sub(r'_(high|low)$',             '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'wan\d+[\._]\d+_?',         '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'polyhedron_?',             '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'v\d+$',                    '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'_+', '_', cleaned).strip('_')
    parts = [p for p in cleaned.split('_') if p]
    if parts:
        return parts[-1]
    return base.split('_')[0] or base


# ─── Group Sorting ─────────────────────────────────────────────────────────

def _sort_active_rows(rows: list, flat_mode: bool = False,
                      custom_order: dict = None):
    """Filter active rows, bucket by group, sort by order.

    flat_mode=True  → skip group bucketing entirely, return rows in list order
                      as a single virtual group "—". Useful for simple sequential
                      stacking without any group logic.
    custom_order    → dict mapping group name to int priority (lower = first).
                      Groups not in the dict fall back to GROUP_ORDER index.
                      Only used when flat_mode=False.

    Returns [(group, [row, ...], [weight, ...]), …]."""

    active = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if not row.get("enabled", True):
            continue
        name = row.get("name", "None")
        if not name or name == "None":
            continue
        active.append(row)

    if flat_mode:
        # Simple sequential: all rows in list order, one virtual group
        if not active:
            return []
        weights = [round(_row_weight(r, default=1.0), 4) for r in active]
        return [("—", active, weights)]

    # Group bucketing
    groups = {g: ([], []) for g in GROUP_ORDER}
    for row in active:
        group = str(row.get("group", "—"))
        if group not in groups:
            group = "custom"
        w = _row_weight(row, default=1.0)
        groups[group][0].append(row)
        groups[group][1].append(round(w, 4))

    # Determine sort key for each group
    def _sort_key(group):
        if custom_order and group in custom_order:
            try:
                return (0, int(custom_order[group]))
            except (ValueError, TypeError):
                pass
        # Fallback: standard GROUP_ORDER index
        try:
            return (1, GROUP_ORDER.index(group))
        except ValueError:
            return (1, 999)

    sorted_groups = sorted(
        [g for g in GROUP_ORDER if groups[g][0]],
        key=_sort_key
    )

    return [(g, groups[g][0], groups[g][1]) for g in sorted_groups]


# ═══ Stack Node ══════════════════════════════════════════════════════════

class UltimateLoraStack:
    """
    Polyhedron LoRA Stack — applies multiple LoRAs to MODEL (and optionally CLIP)
    in a deterministic group-based order. For dual-noise architectures (WAN 2.x):
    use TWO instances side-by-side, one per model.

    Universal: works for FLUX / WAN / SDXL / SD1.5 — no model-type assumptions.
    """

    def __init__(self):
        # Per-instance native loader holds the LoRA cache (self.loaded_lora).
        # Persisting it across executions avoids re-reading safetensors files.
        self._loader = _NativeLoraLoader()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "clip": ("CLIP",),
                # MUST live in 'optional' (NOT 'hidden') — ComfyUI silently
                # discards custom STRING widgets in 'hidden'. Frontend hides
                # this widget visually via _ulsHideConfigWidget.
                "uls_config": ("STRING", {
                    "default": '{"rows":[],"mult":1.0}',
                    "multiline": False,
                }),
            },
            "hidden": {
                "node_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES  = ("MODEL", "CLIP", "STRING", "STRING", "STRING")
    RETURN_NAMES  = ("MODEL", "CLIP", "debug_info", "uls_config_out", "trigger_words")
    FUNCTION      = "apply"
    CATEGORY      = "Polyhedron/Loaders"
    OUTPUT_NODE   = False

    def apply(self, model, clip=None, uls_config='{"rows":[],"mult":1.0}', node_id=None):
        if not uls_config or not uls_config.strip():
            uls_config = '{"rows":[],"mult":1.0}'
        try:
            cfg = json.loads(uls_config)
        except json.JSONDecodeError as e:
            print(f"[PLS] ⚠ uls_config JSON invalid: {e}")
            cfg = {"rows": [], "mult": 1.0}

        rows = cfg.get("rows", []) if isinstance(cfg.get("rows"), list) else []

        # Per-group apply modes from frontend: {"scene": "DARE", "detail": "CONCAT", ...}
        group_modes = cfg.get("group_modes", {}) if isinstance(cfg.get("group_modes"), dict) else {}

        # Per-group DARE variants (v098): {"detail": "channel", "scene": "element", ...}
        # Legacy global dare_variant key is used as fallback for old workflows.
        group_dare = cfg.get("group_dare", {}) if isinstance(cfg.get("group_dare"), dict) else {}
        legacy_dare_variant = str(cfg.get("dare_variant", "channel")).lower()
        if legacy_dare_variant not in ("channel", "element"):
            legacy_dare_variant = "channel"

        # Per-group cleanup switches (v250): {"subject": true, ...}. Absent → off,
        # so old workflows are bit-identical. Only act on CONCAT/DARE groups.
        group_trim    = cfg.get("group_trim", {})    if isinstance(cfg.get("group_trim"), dict)    else {}
        group_resolve = cfg.get("group_resolve", {}) if isinstance(cfg.get("group_resolve"), dict) else {}
        # v261: per-group Trim strength (kept-fraction). Absent group → Auto formula.
        group_trim_amount = cfg.get("group_trim_amount", {}) if isinstance(cfg.get("group_trim_amount"), dict) else {}

        # v105: flat_mode disables group sorting — rows applied in list order.
        flat_mode = bool(cfg.get("flatMode", False))

        # v105: custom group order — {"subject": 1, "detail": 2, "scene": 3, ...}
        custom_order = cfg.get("groupOrder", {}) if isinstance(cfg.get("groupOrder"), dict) else {}

        ordered      = _sort_active_rows(rows, flat_mode=flat_mode,
                                         custom_order=custom_order or None)
        total_active = sum(len(grp_rows) for _, grp_rows, _ in ordered)

        model_out = model
        clip_out  = clip
        all_errors = []

        sort_mode_label = "FLAT" if flat_mode else ("CUSTOM ORDER" if custom_order else "GROUP ORDER")
        lines = [
            "═══ Polyhedron LoRA Stack ═══",
            f"  CLIP        : {'connected' if clip is not None else 'not connected'}",
            f"  Rows in     : {len(rows)}  (active: {total_active})",
            f"  Sort mode   : {sort_mode_label}",
            f"  Groups      : {len(ordered)}",
            "───────────────────────────────",
        ]
        if not rows:
            lines.append("  ⚠ No rows received from frontend!")
            lines.append(f"  uls_config: {uls_config[:80]}")

        for group, grp_rows, grp_weights in ordered:
            n = len(grp_rows)
            grp_label = f"[{group}]" if group != "—" else "[—]"
            mode = (group_modes.get(group) or "SEQ").upper()
            if mode not in ("SEQ", "CONCAT", "DARE"):
                mode = "SEQ"

            # Per-group DARE variant, fallback to legacy global setting
            dare_variant = str(group_dare.get(group, legacy_dare_variant)).lower()
            if dare_variant not in ("channel", "element"):
                dare_variant = "channel"

            # Per-group cleanup switches (v250). Only meaningful for CONCAT/DARE.
            trim    = bool(group_trim.get(group, False))    and mode != "SEQ"
            resolve = bool(group_resolve.get(group, False)) and mode != "SEQ"
            # v261: optional Trim strength override (only a real number counts;
            # anything else → None → auto group-size formula).
            trim_amount = None
            if trim:
                _ta = group_trim_amount.get(group, None)
                if isinstance(_ta, (int, float)):
                    trim_amount = float(_ta)

            names = [r.get("name", "None") for r in grp_rows]
            # v302: per-row CLIP strength (defaults to the model weight)
            grp_clip = [round(_row_clip_weight(r, w), 4)
                        for r, w in zip(grp_rows, grp_weights)]

            if n == 1:
                short = _short_name(names[0])
                lines.append(f"  {grp_label} {short}  ×{grp_weights[0]}")
            else:
                dare_suffix = f" [{dare_variant[:4].upper()}]" if mode == "DARE" else ""
                clean_suffix = (" +TRIM" if trim else "") + (" +RESOLVE" if resolve else "")
                lines.append(f"  {grp_label} {mode}{dare_suffix}{clean_suffix} ({n} LoRAs):")

            model_out, clip_out, errs = apply_lora_set(
                self._loader, model_out, clip_out,
                names, grp_weights, mode=mode, dare_variant=dare_variant,
                trim=trim, resolve=resolve, trim_amount=trim_amount,
                clip_weights=grp_clip
            )
            all_errors.extend(errs)

            if n >= 2:
                err_set = set(errs)
                for row, w in zip(grp_rows, grp_weights):
                    short = _short_name(row.get("name", ""), 35)
                    if any(short in e for e in err_set):
                        lines.append(f"    ⚠ {short}  skipped")
                    else:
                        lines.append(f"    • {short}  ×{w}")
            elif errs:
                # n==1 with error
                for e in errs:
                    lines.append(f"    {e}")

        if all_errors:
            lines.append("───────────────────────────────")
            lines.append(f"  ⚠ {len(all_errors)} LoRA(s) skipped (incompatible model)")
        lines.append("───────────────────────────────")
        debug = "\n".join(lines)
        print(f"\n[PLS]\n{debug}\n")

        # Collect trigger words + build lora_info for Inspector
        triggers = []
        lora_info = []  # [{name, weight, group, trigger_words}, ...]
        for group, grp_rows, grp_weights in ordered:
            for row, w in zip(grp_rows, grp_weights):
                name = row.get("name", "")
                tw = _get_trigger(name)
                if tw and tw not in triggers:
                    triggers.append(tw)
                lora_info.append({
                    "name":          os.path.basename(name).replace(".safetensors", ""),
                    "weight":        w,
                    "group":         group,
                    "trigger_words": tw,
                })
        trigger_words = ", ".join(triggers)

        # Attach lora_info to uls_config_out so Inspector can read it
        try:
            cfg_out = json.loads(uls_config)
        except Exception:
            cfg_out = {}
        cfg_out["lora_info"] = lora_info
        uls_config_out = json.dumps(cfg_out)

        return (model_out, clip_out, debug, uls_config_out, trigger_words)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Force re-execution every time — model identity may change
        # without uls_config changing. Cached loader avoids real I/O cost.
        # **kwargs accepts any combination of args ComfyUI throws at us
        # (model, clip, uls_config, node_id) even when some are missing
        # during workflow validation.
        return float("nan")


# ═══ Engine Node ══════════════════════════════════════════════════════════

class ULSAccelerator:
    """
    Polyhedron LoRA Engine — applies engine-class LoRAs (Lightning, Turbo,
    LCM, FusionX, LightXT2V, CausVid, …) before the main creative stack.
    These modify HOW the model computes (inference trajectory) rather than
    WHAT it depicts.

    Flat list (no groups), single global merge mode.
    Same universal behaviour as the Stack — no model-type assumptions.

    Class name kept as `ULSAccelerator` for workflow back-compat.
    """

    def __init__(self):
        self._loader = _NativeLoraLoader()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "clip": ("CLIP",),
                "engine_config": ("STRING", {
                    "default": '{"rows":[],"mode":"SEQ"}',
                    "multiline": False,
                }),
            },
            "hidden": {
                "node_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES  = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES  = ("MODEL", "CLIP", "debug_info")
    FUNCTION      = "apply"
    CATEGORY      = "Polyhedron/Loaders"
    OUTPUT_NODE   = False

    def apply(self, model, clip=None, engine_config='{"rows":[],"mode":"SEQ"}', node_id=None):
        if not engine_config or not engine_config.strip():
            engine_config = '{"rows":[],"mode":"SEQ"}'
        try:
            cfg = json.loads(engine_config)
        except json.JSONDecodeError as e:
            print(f"[Engine] ⚠ engine_config JSON invalid: {e}")
            cfg = {"rows": [], "mode": "SEQ"}

        rows = cfg.get("rows", []) if isinstance(cfg.get("rows"), list) else []
        mode = (cfg.get("mode") or "SEQ").upper()
        if mode not in ("SEQ", "CONCAT", "DARE"):
            mode = "SEQ"

        dare_variant = str(cfg.get("dare_variant", "channel")).lower()
        if dare_variant not in ("channel", "element"):
            dare_variant = "channel"

        # Optional global cleanup switches (v250). No dedicated Engine UI yet;
        # default off. Only meaningful for CONCAT/DARE.
        trim    = bool(cfg.get("trim", False))    and mode != "SEQ"
        resolve = bool(cfg.get("resolve", False)) and mode != "SEQ"

        # Filter active rows. Engine uses `weight`; tolerate legacy too via _row_weight.
        active_names, active_weights, active_clip = [], [], []
        for row in rows:
            if not isinstance(row, dict):           continue
            if not row.get("enabled", True):        continue
            name = row.get("name", "None")
            if not name or name == "None":          continue
            w = _row_weight(row, default=1.0)
            active_names.append(name)
            active_weights.append(round(w, 4))
            active_clip.append(round(_row_clip_weight(row, w), 4))

        n = len(active_names)
        lines = [
            "═══ Polyhedron LoRA Engine ═══",
            f"  CLIP        : {'connected' if clip is not None else 'not connected'}",
            f"  Active      : {n} engine LoRA(s)",
            f"  Mode        : {mode}",
        ]
        if mode == "DARE":
            lines.append(f"  DARE variant: {dare_variant}")
        if trim or resolve:
            _cleanup = []
            if trim:    _cleanup.append("TRIM (magnitude)")
            if resolve: _cleanup.append("RESOLVE (TIES)")
            lines.append("  Cleanup     : " + " + ".join(_cleanup))
        lines.append("──────────────────────────────")

        if n == 0:
            lines.append("  (no engine LoRAs active — pass-through)")
            debug = "\n".join(lines)
            print(f"\n[Engine]\n{debug}\n")
            return (model, clip, debug)

        model_out, clip_out, errs = apply_lora_set(
            self._loader, model, clip,
            active_names, active_weights,
            mode=mode, dare_variant=dare_variant,
            trim=trim, resolve=resolve,
            clip_weights=active_clip,
        )

        err_set = set(errs)
        for name, w in zip(active_names, active_weights):
            short = _short_name(name)
            if any(short in e for e in err_set):
                lines.append(f"  ⚠ {short}  skipped")
            else:
                lines.append(f"  • {short}  ×{w}")

        if errs:
            lines.append("──────────────────────────────")
            lines.append(f"  ⚠ {len(errs)} LoRA(s) skipped (incompatible model)")
        lines.append("──────────────────────────────")
        debug = "\n".join(lines)
        print(f"\n[Engine]\n{debug}\n")
        return (model_out, clip_out, debug)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # See ULS Stack IS_CHANGED — same reasoning. **kwargs makes the
        # method robust against any arg combination ComfyUI may pass
        # (e.g. missing model during validation phase).
        return float("nan")


# ═══ Inspector Node ═══════════════════════════════════════════════════════════

class ULSInspector:
    """
    Polyhedron LoRA Inspector — passive consistency-check node.

    Reads active LoRAs + their trigger words from uls_config_out (Stack output),
    then checks whether each trigger word appears in the supplied prompt string.
    Outputs a formatted report as STRING.

    No model patching — purely informational.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "uls_config_out": ("STRING", {
                    "default": '{"rows":[]}',
                    "multiline": False,
                    "forceInput": True,
                }),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "forceInput": True,
                }),
            },
        }

    RETURN_TYPES  = ("STRING",)
    RETURN_NAMES  = ("inspector_report",)
    FUNCTION      = "inspect"
    CATEGORY      = "Polyhedron/Utils"
    OUTPUT_NODE   = False

    def inspect(self, uls_config_out: str, prompt: str) -> tuple:
        # Parse config
        try:
            cfg = json.loads(uls_config_out)
        except Exception:
            cfg = {}

        lora_info = cfg.get("lora_info", [])

        if not lora_info:
            report = "⬡ Polyhedron LoRA Inspector\n  (no lora_info — connect uls_config_out from Stack v119+)"
            return (report,)

        # Build prompt token map: word → explicit weight or "plain"
        # Matches (word:1.2) syntax and plain words
        prompt_weights = {}
        for m in re.finditer(r'\(([^():,]+?):([\d.+-]+)\)', prompt):
            word = m.group(1).strip().lower()
            try:
                prompt_weights[word] = float(m.group(2))
            except ValueError:
                prompt_weights[word] = None

        prompt_lower = prompt.lower()

        # Build report
        lines = [
            "═══ Polyhedron LoRA Inspector ═══",
            f"  LoRAs active : {len(lora_info)}",
            "─────────────────────────────────",
        ]

        col_name = 32
        col_lora = 8
        col_trig = 24

        header = f"  {'LoRA':<{col_name}} {'Weight':>{col_lora}}   {'Trigger':<{col_trig}}  {'In Prompt'}"
        lines.append(header)
        lines.append("  " + "─" * (col_name + col_lora + col_trig + 20))

        found_count = 0
        missing_triggers = []

        for entry in lora_info:
            name    = entry.get("name", "?")[:col_name]
            weight  = entry.get("weight", 0.0)
            tw_raw  = entry.get("trigger_words", "")

            if not tw_raw:
                # No trigger words known for this LoRA
                lora_col  = f"×{weight:.2f}"
                trig_col  = "(none)"
                match_col = "—"
                lines.append(f"  {name:<{col_name}} {lora_col:>{col_lora}}   {trig_col:<{col_trig}}  {match_col}")
                continue

            # Split trigger words and check each
            triggers = [t.strip() for t in tw_raw.split(",") if t.strip()]
            best_match = None
            best_weight_str = ""

            for t in triggers:
                t_lower = t.lower()
                # Word-boundary match — avoids false positives like
                # "ring" matching inside "stinging" or "string"
                pattern = r'\b' + re.escape(t_lower) + r'\b'
                if t_lower in prompt_weights:
                    w = prompt_weights[t_lower]
                    best_match = t
                    best_weight_str = f"({t}:{w})" if w is not None else f"({t}:?)"
                    break
                elif re.search(pattern, prompt_lower):
                    best_match = t
                    best_weight_str = f"{t}  plain"
                    break

            lora_col = f"×{weight:.2f}"
            trig_col = triggers[0][:col_trig]  # show primary trigger

            if best_match:
                found_count += 1
                match_col = f"✓  {best_weight_str}"
            else:
                missing_triggers.append(name)
                match_col = "✗  NOT IN PROMPT"

            lines.append(f"  {name:<{col_name}} {lora_col:>{col_lora}}   {trig_col:<{col_trig}}  {match_col}")

        lines.append("  " + "─" * (col_name + col_lora + col_trig + 20))
        missing = len(lora_info) - found_count
        no_trigger = sum(1 for e in lora_info if not e.get("trigger_words"))
        lines.append(f"  ✓ {found_count} matched   ✗ {missing - no_trigger} missing   — {no_trigger} no trigger defined")

        if missing_triggers:
            lines.append("  Missing:")
            for n in missing_triggers:
                lines.append(f"    • {n}")

        lines.append("─────────────────────────────────")
        report = "\n".join(lines)
        print(f"\n[PLS Inspector]\n{report}\n")
        return (report,)


# ─── Token Counter ─────────────────────────────────────────────────────────

# Optional: try to use the real UMT5-XXL tokenizer for exact counts. Falls
# back to a heuristic if transformers / tokenizer files are not available.
# We import lazily inside the count function so import-time stays cheap and
# the node also loads on systems without `transformers` installed.
_UMT5_TOKENIZER = None  # cached after first successful load
_UMT5_LOAD_ATTEMPTED = False


def _try_load_umt5_tokenizer():
    """Lazy-load UMT5-XXL tokenizer once. Returns tokenizer or None."""
    global _UMT5_TOKENIZER, _UMT5_LOAD_ATTEMPTED
    if _UMT5_LOAD_ATTEMPTED:
        return _UMT5_TOKENIZER
    _UMT5_LOAD_ATTEMPTED = True
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("[PLS Tokens] transformers not installed — using heuristic estimator")
        return None
    # v267 (audit A-4): genuinely offline-friendly two-stage attempt — first
    # the LOCAL cache only (no network round-trip, no online etag check when
    # cached), then the regular online fetch (which caches for next time).
    # Name matches WAN's config.
    for name in ("google/umt5-xxl", "google/umt5-base"):
        for _local_only in (True, False):
            try:
                _UMT5_TOKENIZER = AutoTokenizer.from_pretrained(
                    name, legacy=False, local_files_only=_local_only)
                _src = "local cache" if _local_only else "online (now cached)"
                print(f"[PLS Tokens] ✓ Loaded {name} tokenizer (exact counts, {_src})")
                return _UMT5_TOKENIZER
            except Exception:
                continue
    print("[PLS Tokens] UMT5 tokenizer not in local cache — using heuristic estimator")
    print("[PLS Tokens]   For exact counts: pip install transformers + first online run will cache it")
    return None


def _heuristic_token_count(text: str) -> int:
    """
    SentencePiece-style heuristic for UMT5-XXL token count.

    Calibrated against the empirical observation that the user's v163 crash
    prompt of 2712 characters / 348 words produced exactly 635 tokens (the
    crash log said 'negative dimension -123', i.e. 512 - actual = -123 →
    actual = 635). The calibration yields ~5% accuracy on prompts of this
    style; mileage on heavily non-English or symbol-dense prompts may vary.

    Two-term formula:
      base       = chars / 4.3   (UMT5-XXL English-mix average)
      digit_bonus = 1.5 per digit-mid-word (stor8m, ne8ttle, w3tcl0, ...)
                    — these trigger fragment splits in SentencePiece, each
                    costing roughly 2 extra tokens vs the surface form.

    For exact counts the node prefers the real UMT5-XXL tokenizer via
    `transformers`; this heuristic is the always-available fallback.
    """
    if not text:
        return 0
    # Base: UMT5-XXL averages ~4.3 chars per token for English-mix text
    # (validated against transformers UMT5-XXL output on the v163 prompt)
    base = len(text) / 4.3
    # Correction: digit-in-the-middle words like "stor8m" or "ne8ttle" get
    # aggressively fragmented by SentencePiece. Each adds ~1.5 extra tokens
    # over the chars/4.3 baseline.
    digit_words = re.findall(r"\b\w*\d\w+\b", text)
    digit_bonus = len(digit_words) * 1.5
    return max(1, round(base + digit_bonus))


def _count_tokens(text: str) -> tuple:
    """
    Returns (count, method) where method is "exact" or "heuristic".
    """
    if not text:
        return (0, "exact")
    tok = _try_load_umt5_tokenizer()
    if tok is not None:
        try:
            return (len(tok.encode(text, add_special_tokens=True)), "exact")
        except Exception as e:
            print(f"[PLS Tokens] tokenizer failed ({e}) — falling back to heuristic")
    return (_heuristic_token_count(text), "heuristic")


def _make_bar(used: int, total: int, width: int = 32) -> str:
    """ASCII progress bar."""
    if total <= 0:
        return "[" + "?" * width + "]"
    ratio = min(1.5, used / total)  # cap at 150% for visual
    filled = int(round(ratio * width))
    if filled <= width:
        bar = "█" * filled + "░" * (width - filled)
    else:
        overflow = filled - width
        bar = "█" * width + "▓" * min(overflow, 8) + "!" * max(0, overflow - 8)
    return f"[{bar}]"


class ULSTokenCounter:
    """
    Polyhedron Token Counter — diagnostic node for WAN prompt budgets.

    WAN 2.x models use the UMT5-XXL text encoder with a hard limit of
    512 tokens (max sequence length). Exceeding this in kijai's WanVideo-
    Wrapper produces a hard crash:
        RuntimeError: Trying to create tensor with negative dimension

    This node estimates the token count of positive and negative prompts
    and warns before they exceed the model limit. Uses the real UMT5-XXL
    tokenizer if `transformers` is installed and the model files are
    cached locally; otherwise falls back to a calibrated heuristic.

    No model patching — purely informational. Hook the same STRING that
    feeds your CLIP Text Encode (or WanVideoTextEncode) into this node
    to monitor your budget.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_limit": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 8192,
                    "step": 64,
                    "tooltip": "Max tokens supported by the target model's text "
                               "encoder. WAN 2.1 / WAN 2.2 = 512 (UMT5-XXL hard "
                               "limit). FLUX = 512 (T5-XXL). SDXL = 75 per CLIP "
                               "chunk. Leave at 512 for WAN.",
                }),
                "warn_threshold": ("FLOAT", {
                    "default": 0.90,
                    "min": 0.50,
                    "max": 1.00,
                    "step": 0.05,
                    "tooltip": "Fraction of the limit at which to warn. 0.90 "
                               "means warn at ≥460 of 512 tokens.",
                }),
            },
            "optional": {
                "positive_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "forceInput": True,
                    "tooltip": "Positive prompt to count.",
                }),
                "negative_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "forceInput": True,
                    "tooltip": "Negative prompt to count.",
                }),
                "trigger_words": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "forceInput": True,
                    "tooltip": "Optional: wire the Stack's trigger_words output here "
                               "to see how much of the budget the auto-collected "
                               "triggers alone consume. Diagnostic only — these are "
                               "normally already part of the positive prompt, so they "
                               "are NOT added to over_limit.",
                }),
            },
        }

    RETURN_TYPES  = ("STRING", "INT", "INT", "BOOLEAN", "INT")
    RETURN_NAMES  = ("report", "positive_tokens", "negative_tokens", "over_limit", "trigger_tokens")
    FUNCTION      = "count"
    CATEGORY      = "Polyhedron/Utils"
    OUTPUT_NODE   = False

    def count(self,
              model_limit: int,
              warn_threshold: float,
              positive_prompt: str = "",
              negative_prompt: str = "",
              trigger_words: str = "") -> tuple:

        pos_count, pos_method = _count_tokens(positive_prompt)
        neg_count, neg_method = _count_tokens(negative_prompt)
        # Trigger words are diagnostic only — see the report note and the
        # RETURN comment. Always defined so the trigger_tokens output is stable.
        trig_count = 0
        if trigger_words and trigger_words.strip():
            trig_count, _ = _count_tokens(trigger_words)

        # Use the same method label if both are exact, otherwise show mixed
        method = pos_method if pos_method == neg_method else "mixed"
        method_label = {
            "exact":     "UMT5-XXL tokenizer (exact)",
            "heuristic": "heuristic estimator (±5%)",
            "mixed":     "mixed (some exact, some heuristic)",
        }.get(method, method)

        pos_pct = (pos_count / model_limit * 100) if model_limit else 0.0
        neg_pct = (neg_count / model_limit * 100) if model_limit else 0.0
        warn_at = int(model_limit * warn_threshold)

        pos_status = self._status(pos_count, model_limit, warn_at)
        neg_status = self._status(neg_count, model_limit, warn_at)

        over_limit = (pos_count > model_limit) or (neg_count > model_limit)

        # Build report
        lines = [
            "═══ Polyhedron Token Counter ═══",
            f"  Limit       : {model_limit} tokens",
            f"  Warn at     : {warn_at} tokens ({int(warn_threshold * 100)}%)",
            f"  Method      : {method_label}",
            "─────────────────────────────────",
            f"  POSITIVE    : {pos_count:>4} / {model_limit}  ({pos_pct:5.1f}%)  {pos_status}",
            f"                {_make_bar(pos_count, model_limit)}",
        ]
        if pos_count > 0:
            words = len(positive_prompt.split())
            ratio = pos_count / max(1, words)
            lines.append(f"                {words} words → {ratio:.2f} tokens/word")
        lines.append("")
        lines.append(f"  NEGATIVE    : {neg_count:>4} / {model_limit}  ({neg_pct:5.1f}%)  {neg_status}")
        lines.append(f"                {_make_bar(neg_count, model_limit)}")
        if neg_count > 0:
            words = len(negative_prompt.split())
            ratio = neg_count / max(1, words)
            lines.append(f"                {words} words → {ratio:.2f} tokens/word")
        # Trigger words (optional 3rd input) — diagnostic only. These are
        # typically ALREADY part of the positive prompt (wired via JoinStrings),
        # so they are deliberately NOT folded into over_limit; showing them
        # separately answers "how much of my budget do the auto-collected
        # triggers eat?" (open point #2 from the roadmap).
        if trig_count > 0:
            trig_pct = (trig_count / model_limit * 100) if model_limit else 0.0
            lines.append("")
            lines.append(f"  TRIGGERS    : {trig_count:>4} / {model_limit}  ({trig_pct:5.1f}%)  (auto-collected)")
            lines.append(f"                {_make_bar(trig_count, model_limit)}")
            lines.append("                ℹ usually already inside POSITIVE — informational, not added to over-limit")
        lines.append("─────────────────────────────────")

        # Actionable hints
        hints = []
        if over_limit:
            hints.append("⚠ OVER LIMIT — kijai's WanVideoSampler will crash with")
            hints.append("    'RuntimeError: Trying to create tensor with negative dimension'")
            hints.append("  Options:")
            hints.append("    • Shorten prompt (drop redundant tags, remove (word:1.x) syntax")
            hints.append("      — WAN/UMT5 ignores ComfyUI weight syntax, it just eats tokens)")
            hints.append("    • Use kijai's WanVideoTextEncode node instead of CLIPTextEncode +")
            hints.append("      WanVideoTextEmbedBridge — it truncates silently at 512.")
        elif (pos_count >= warn_at) or (neg_count >= warn_at):
            hints.append("⚠ Approaching limit — quality degrades noticeably above ~70% of limit")
            hints.append("  (motion slows, 'grid' patterns appear in output — kijai issue #1781).")
        if method == "heuristic":
            hints.append("ℹ For exact counts: `pip install transformers` and ensure the UMT5-XXL")
            hints.append("  tokenizer is downloaded (first online use will cache it).")

        if hints:
            for h in hints:
                lines.append(f"  {h}")
            lines.append("─────────────────────────────────")

        report = "\n".join(lines)
        print(f"\n[PLS Tokens]\n{report}\n")

        return (report, pos_count, neg_count, over_limit, trig_count)

    @staticmethod
    def _status(count: int, limit: int, warn_at: int) -> str:
        if count == 0:
            return "(empty)"
        if count > limit:
            return f"✗ OVER by {count - limit}"
        if count >= warn_at:
            return "⚠ near limit"
        return "✓ ok"
