# Changelog — v361

## Same nodes, redrawn on the ComfyUI v3 schema (Nodes 2.0) + security hardening

This public release brings the LoRA-stack edition up to the v361 line. **No node was
removed, renamed, or changed in behaviour** — existing workflows load unchanged. The work
is about *how* the nodes are drawn and registered, plus one security fix and an internal
refactor. (The public edition has no 3D nodes; the intermediate internal versions
v321–v360 were 3D-cockpit work and are not part of this package.)

### V3 schema migration (drop-in, with legacy fallback)

Six nodes now register as ComfyUI **v3 `io.ComfyNode`** classes when the host exposes
`comfy_api.latest`, and fall back to the proven legacy class on older ComfyUI:

- ⬡ Polyhedron LoRA Inspector
- ⬡ Polyhedron Merge Analyzer
- ⬡ Polyhedron Pick Frame
- ⬡ Polyhedron Wan Frame Inflate (T2I LoRA fix)
- ⬡ Polyhedron Dual Sigma Curve
- ⬡ Polyhedron Sigma Curve

Each v3 node uses the **same `node_id`** as its legacy counterpart and exposes the full V1
interface, so it is a true drop-in: same inputs, same outputs, same category — workflows do
not need to be touched, and a host without `comfy_api` simply gets the legacy node. The
registration is centralised in `nodes/uls_v3_extension.py`; if that import fails for any
reason, every migrated node falls back to legacy so nothing disappears.

### Security — Civitai fetch hardening

The Civitai preview/trigger fetch now validates the safetensors `sshs_model_hash` as
**hex-only** (SHA256 or the shorter AutoV2 hash) before interpolating it into the request
URL, so a crafted header value can no longer manipulate the outgoing request.

### Internal refactor (no behaviour change)

The shared merge math (Trim channel selection, Resolve sign election, etc.) moved into a
dedicated `nodes/uls_merge_math.py`. The Stack, Engine, Merge Analyzer, Sigma and Bridge
node files carry the accumulated non-3D fixes since v320. Merge results are unchanged.

### Dependencies

Unchanged and minimal: the nodes need **no** hard third-party dependency (the runtime
Civitai fetch uses aiohttp, which ComfyUI ships). `Pillow` + `requests` remain an optional
extra, used only by the standalone CLI preview generator.

### Notes

- Documentation PDF is unchanged for this release and will be refreshed separately.
- Compatibility: ComfyUI with the v3 node API gets the v3 nodes; older ComfyUI keeps the
  legacy nodes automatically.
