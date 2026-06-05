# v269 — wording release (2.69.0)

Terminology cleanup, no functional changes.

- Replaced the informal term "mush" with "multi-LoRA interference"
  (the term used in the TIES-Merging literature) across all public text:
  - `pyproject.toml` description (shown on the Comfy Registry and in ComfyUI-Manager)
  - `README.md` intro
  - Trim toggle hint in `web/js/uls_node.js`
  - source comments/docstrings in `nodes/uls_stack_node.py`
- User documentation updated to v125 (same wording change in §3.7;
  `docs/Polyhedron_LoRA_Stack_Documentation_v125.pdf` replaces v124).
- Version strings bumped to v269 / 2.69.0 (`pyproject.toml`, `__init__.py`,
  `web/js/uls_compat.js`).

No code paths were touched: merge math, persistence, bridge hand-off and all
node behaviour are identical to v268/v267. Outputs remain bit-identical.
