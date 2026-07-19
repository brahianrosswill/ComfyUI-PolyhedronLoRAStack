# CHANGELOG v268 — Public-Release-Packaging (GitHub + Comfy Registry)

**Datum:** 4. Juni 2026
**Basis:** v267 (zweifach vollauditiert, `AUDIT_v267_full.md`)
**Typ:** Reine Release-/Packaging-Version — **null funktionale Änderung.**
Alle Merge-, Sigma-, Routen- und UI-Logikdateien sind **byte-identisch** zu
`ComfyUI-PolyhedronLoRAStack_v267_clean.zip` (sha256-Abgleich, Tabelle unten).
Anlass: Erstveröffentlichung auf GitHub + Registrierung in der Comfy Registry,
damit „Install Missing Custom Nodes" im ComfyUI-Manager das Pack findet.

---

## Geänderte Dateien (4)

1. **`pyproject.toml`** — auf Registry-Spezifikation gebracht:
   - `name = "polyhedron-lora-stack"` *(vorher `ComfyUI-PolyhedronLoRAStack`)*.
     **⚠ Geflaggte Entscheidung:** Das ist die **unveränderliche Node-ID** der
     Registry (URL + `comfy node install polyhedron-lora-stack`). Die Spec rät
     ausdrücklich von „ComfyUI" im Namen ab. Der **GitHub-Repo-Name bleibt**
     `ComfyUI-PolyhedronLoRAStack` (wird beim Klonen der Ordnername unter
     `custom_nodes`).
   - `version = "2.68.0"`, `license = { file = "LICENSE" }` *(vorher Text-Form)*.
   - `[tool.comfy]`: `Icon` jetzt URL auf `assets/icon.png` *(vorher ungültiges
     Emoji „⬡" — Spec verlangt Bild-URL, quadratisch, ≤400px)*, neu `Banner`
     (21:9), `Tags`-Key entfernt *(nicht in der Spec)*.
   - Neu: `classifiers` (OS Independent), `Documentation`- und
     `Bug Tracker`-URLs.
2. **`README.md`** — vollständige öffentliche Neufassung (Englisch): alle
   13 Nodes, Merge-Modi, Cleanup-Schalter inkl. **Trim-Arbeitskorridor 70–80 %**
   und Max-50-%-Warnung (B-1-Ergebnis), Merge Analyzer, Installation
   (Manager/comfy-cli/git), Datei-Konventionen, Routen, Doku-Link,
   Civitai/Patreon. *(Alte Fassung dokumentierte nur 2 von 13 Nodes; sie bleibt
   im v267-Archiv erhalten.)*
3. **`__init__.py`** — Banner-String `v267` → `v268` (eine Zeile,
   Auslieferungs-Konvention: Banner = Pack-Version).
4. **`web/js/uls_compat.js`** — `PLUGIN_VERSION = "v268"` (eine Zeile, dito).

## Neue Dateien (8)

- **`LICENSE`** — MIT, © 2026 Polyhedron. **⚠ Geflaggte Entscheidung:** MIT war
  seit jeher in pyproject/README deklariert; mit der Veröffentlichung wird es
  wirksam (Nutzung/Änderung/Weiterverbreitung inkl. kommerziell erlaubt,
  Copyright-Hinweis muss bleiben). Die User-Doku-PDF bleibt davon getrennt
  „All rights reserved" (Hinweis im README-Lizenzabschnitt).
- **`.github/workflows/publish.yml`** — offizieller Registry-Auto-Publish
  (Comfy-Org/publish-node-action; triggert bei pyproject-Änderung auf `main`
  + manuell per `workflow_dispatch`; erwartet Repo-Secret
  `REGISTRY_ACCESS_TOKEN`).
- **`assets/icon.png`** (400×400) + **`assets/banner.png`** (1260×540 = 21:9) —
  Hexagon-Stack-Motiv, generiert; von `[tool.comfy]` referenziert.
- **`assets/screenshot_stack_panel.png` / `…_cleanup_switches.png` /
  `…_merge_analyzer.png`** — S1/S2/S4 aus dem Doku-v124-Satz, fürs README.
- **`docs/Polyhedron_LoRA_Stack_Documentation_v124.pdf`** — User-Doku (34 S.).
- **`CHANGELOG_v268.md`** — dieses Dokument.

## Byte-Identitäts-Beleg

sha256-Abgleich gegen `…_v267_clean.zip`: **23 von 27 Dateien byte-identisch**;
geändert ausschließlich die vier oben genannten (`pyproject.toml`, `README.md`,
`__init__.py`-Banner, `uls_compat.js`-Versionsstring). Alle `nodes/*.py`, alle
6 Tests, `uls_node.js`, `uls_styles.css`, `polyhedron_sigma_inline.js`,
`uls_model_switch.js`, `requirements.txt`, `COMPATIBILITY.md`, `install.py`,
`uls_preview_gen.py` und `CHANGELOG_v267.md` unverändert. `py_compile` +
`node --check` sauber; Wiring-Suiten `test_v266`/`test_v267` grün
(torch-Teile SKIP, Sandbox ohne torch). Gesamtbestand v268: 36 Dateien
(27 + 9 neue inkl. dieses Changelogs).

## Tabu-Pfade

kijai-Sampler-Loop und Bridge-Hand-off **nicht angefasst** (keine Code-Edits
in dieser Version).

## Offen / nächste Schritte

GitHub-Account angelegt (Username `PolyhedronAI` — alle sechs URLs in
`pyproject.toml`/`README.md` darauf gesetzt). Noch offen: Repo anlegen,
Inhalt hochladen, Registry-Publisher `polyhedron` (Fallback `polyhedron-ai`)
+ API-Key, Secret `REGISTRY_ACCESS_TOKEN` setzen, Workflow auslösen,
Manager-Sichtbarkeit verifizieren. Falls die Publisher-ID vergeben ist:
nur `PublisherId` in `pyproject.toml` anpassen (eine Zeile).
