# v270 — example workflows (2.70.0)

Packaging release, no code changes.

- New `example_workflows/` folder with four starter templates (JSON + JPG
  thumbnail each). They appear in ComfyUI's template browser
  (Workflow -> Browse Templates) once the pack is installed:
  - `polyhedron_lora_stack_ksampler_base` /
    `polyhedron_lora_stack_ksampler_lightning` — native KSampler (Advanced)
    path, without / with the Wan2.2-Lightning v1.1 8-step preset
    (4/4 split, CFG 1.0, euler/simple, shift 5).
  - `polyhedron_lora_stack_wanvideo_base` /
    `polyhedron_lora_stack_wanvideo_accelerator` — WanVideoWrapper (kijai)
    path, without / with the Lightning accelerator. The kijai #1827
    single-frame workaround is documented in-canvas; Frame Inflate ships
    bypassed for the GGUF default.
  - All four: grouped canvas, neutral prompts, fixed seeds; Stack and
    Engine rows ship as disabled examples with showcase group modes
    pre-set.
- README: new "Example workflows" section incl. external-pack requirements.
- Version strings bumped to v270 / 2.70.0 (`pyproject.toml`, `__init__.py`,
  `web/js/uls_compat.js`).

Merge math, persistence, bridge hand-off and all node behaviour are
identical to v269. No Python/JS logic touched.
