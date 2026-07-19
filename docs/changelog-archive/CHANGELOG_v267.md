# CHANGELOG v267 — Härtungspass: Audit-Befunde A-1…A-8 + Neufunde N-1…N-9

**Datum:** 4. Juni 2026
**Basis:** v266
**Anlass:** Zweites unabhängiges Voll-Audit (frische Sandbox, eigene Sonden gegen
die echten Versandfunktionen) — Details in `AUDIT_v267_full.md`.
**Tabu respektiert:** kijai-Sampler-Loop und Bridge-Hand-off-Pfad unangetastet.
**Merge-Mathematik:** output-identisch — Bit-Identität in der Sandbox belegt
(CONCAT max|Δ|=3.81e-06 unverändert, `_dare_seed`=636800090 unverändert,
Analyzer e1=83.6 / amp=0.914 unverändert, test_v259 [A] grün).

---

## ⚠ Geflaggte Entscheidungen (bitte lesen — beide bewusst, beide reversibel)

### A-2: Civitai-Preview-Auswahl ist jetzt FILTER-STRIKT
**Vorher:** Die In-App-Route akzeptierte Bilder bis `nsfwLevel ≤ 4` (inkl.
*Mature*) und fiel, wenn **kein** Bild den Filter bestand, auf `images[0]`
zurück — also ausgerechnet auf ein Bild, das den Filter **nicht** bestanden
hatte (potenziell X-rated). Das widersprach der dokumentierten Zusage
„mit NSFW-Filter".
**Jetzt:** Neue pure Hilfsfunktion `_pick_preview_url(images, max_nsfw_level=2)`
— erstes Bild vom Typ `image` mit `nsfwLevel ≤ 2` (Civitai-Skala: 1=None,
2=Soft, 4=Mature, 8+=X; Level 3 existiert nicht). **Kein Fallback mehr.**
**Konsequenz:** Für LoRAs, deren Civitai-Galerie ausschließlich Mature/X-Bilder
enthält, wird **kein Preview mehr gespeichert** (vorher: das erstbeste Bild).
**Rückbau, falls unerwünscht:** `max_nsfw_level=4` am Aufrufer (eine Zahl) —
für Mature-Previews; der entfernte `images[0]`-Fallback sollte in keinem Fall
zurückkommen.

### A-6: Whitelist auf `POST /uls/groups`
Unbekannte Gruppennamen werden mit **400** abgelehnt
(`_VALID_GROUPS = {acc, style, scene, motion, subject, detail, custom}`).
Rein defensiv: Die UI sendet ohnehin nur diese Werte, und das Backend mappte
Unbekanntes beim Sortieren sowieso auf `custom` — jetzt erreichen
Fantasiestrings die JSON-Datei gar nicht erst. Legacy-Namen alter
`uls_groups.json`-Bestände sind unberührt (Migration läuft beim **Laden**).

---

## Der eigentliche Bugfix: N-2 (einziger echter Verhaltens-Bug des Audits)

**Symptom:** Gruppenwechsel über die Gruppen-Buttons im **Thumbnail-Overlay**
landete nicht sofort im versteckten `uls_config`-Widget. Wer die Gruppe dort
umstellte und **direkt danach queuete**, renderte mit der **alten** Gruppe —
der neue Wert kam erst bei der nächsten beliebigen anderen Interaktion an.
**Ursache:** Der Button-Handler rief `node._ulsSync()` nicht auf — die
DARE-Variant-Buttons im selben Overlay und der GRP-Pill-Zyklus taten es längst;
dies war der eine vergessene Pfad.
**Fix:** `node?._ulsSync?.();` direkt nach dem Setzen (Optional Chaining hält
den Engine-Pfad, der das Overlay mit `node=null` öffnet, als No-op).

## Weitere Fixes

| # | Befund | Fix |
|---|--------|-----|
| A-1 | `handle_civitai_fetch` las die safetensors-Header-Länge ohne Limit — korrupte/bösartige lokale Datei konnte eine Riesen-Allokation auslösen | 50-MB-Cap (identisch zu `_read_meta`) **plus** Kurz-Header-Guard; beides antwortet mit sauberem Fehler-JSON statt zu allozieren |
| A-3 | CLI-`download_file` (Preview-Generator) ohne Größenlimit | 32-MB-Cap wie die In-App-Route: `Content-Length`-Vorabprüfung **und** Mid-Stream-Abbruch (lügende/chunked Antworten), Teildatei wird per `unlink` entfernt |
| A-4 | Tokenizer-Load war nicht wirklich „offline-friendly" (transformers macht auch bei Cache-Treffer Online-Etag-Checks) | Zwei-Stufen-Versuch je Modellname: erst `local_files_only=True` (echter Offline-Pfad), dann regulär online (cached für das nächste Mal) |
| A-5 | Veralteter JSDoc „Resolve … (disabled until its backend lands)" | → „(live since v256)" |
| A-8 | Dual-Sigma-Handoff hatte 1 float32-ULP Restdifferenz (gemessen max 5.96e-08 über 256 Schedule-Kombos) | `sigmas_low_np[split_step] = handoff` nach dem Rescale — **exakte Bit-Gleichheit** (Sweep: max Diff 0.0, Monotonie erhalten); deckt auch den rescale-übersprungenen Zweig ab |
| N-1 | NSFW-Schwellen inkonsistent: Route `≤ 4` vs. CLI `> 3` (effektiv `≤ 2`, da Level 3 nicht existiert) | Vereinheitlicht auf `≤ 2` (Route via A-2-Helfer, CLI explizit `> 2` mit Erklärkommentar) |
| N-3 | README nannte die Trigger-Priorität verkehrt (`.txt` vor `.uls-meta.json`) — Code und Master sagen das Gegenteil | Beide README-Stellen korrigiert: `.uls-meta.json` (user-curated) → `.txt` → Header → Dateiname |
| N-5 | RFC-7233-Suffix-Ranges (`bytes=-N` = **letzte** N Bytes) wurden als `0..N` fehlinterpretiert | Eigener Suffix-Zweig; Browser-Player senden hier nie Suffix-Ranges, normale Wiedergabe unverändert (Live-Test: `bytes=-4` → korrekt letzte 4 Bytes + `Content-Range`) |
| N-7 | `_resolve_sign_elect` sicherte den CUDA-RNG-State nur bei `tgt==cuda`; `torch.manual_seed` reseedet aber **alle** Devices → der CPU-Retry-Pfad auf einer CUDA-Maschine (OOM→CPU) perturbierte den globalen CUDA-RNG | State wird gesichert/wiederhergestellt, sobald CUDA verfügbar **und initialisiert** ist (Init ist hier garantiert: `_resolve_pick_device` hat `mem_get_info` bereits aufgerufen). Reines State-Handling, output-identisch |
| N-9 | 4× veraltete `[ULS v052]`-Konsolen-Tags | → `[ULS]` |

## Dokumentiert, bewusst NICHT gefixt

- **N-4 (info):** `uls_styles.css` enthält tote Selektoren aus der DOM-Ära —
  die Canvas-UI nutzt die Klassen nicht. Aufräumkandidat für die ohnehin
  anstehende Nodes-2.0-/DOM-Overlay-Arbeit.
- **N-6 (info):** Die Voll-Video-Response (ohne `Range`-Header) lädt die ganze
  Datei in den RAM. Browser-Player nutzen praktisch immer Ranges; Streaming-
  Umbau lohnt erst, falls große Previews üblich werden.
- **N-8 (info):** Tiefen-Analyse vermisst nur lineare Layer — seit v266 als
  Code-Kommentar am `ndim`-Guard dokumentiert (WAN/FLUX: ohne Belang).

## Tests

- **Neu:** `tests/test_v267_hardening.py` — 27 Checks in drei Teilen:
  [1] Wiring jedes Fixes (inkl. „Gefahr wirklich entfernt": kein
  `images[0]`-Fallback, keine `[ULS v052]`-Tags),
  [2] `_pick_preview_url` funktional (AST-geladen, pur): All-NSFW → `None`,
  erstes SFW-Bild gewinnt, Video übersprungen, Schwellen-Override,
  [3] Sigma-Exaktheit: `HIGH[split] == LOW[split]` **bit-exakt** über alle
  Schedule-Kombos + Monotonie (ohne torch: SKIP).
- **Sandbox-Verifikation:** alle 6 Suiten grün (v259 CPU / v261 / v264 / v265
  live / v266 / v267); 12 unabhängige Baseline-Sonden nach den Patches
  **wertgleich** zur v265-Baseline; Live-Routen-Sonden für A-1/A-6/N-5 grün;
  `py_compile` + `node --check` sauber.

## Geänderte Dateien

`nodes/uls_routes.py` (A-1, A-2, A-6, N-5) · `nodes/uls_stack_node.py`
(A-4, N-7, Header) · `uls_preview_gen.py` (A-3, N-1) ·
`nodes/wan_sigma_schedule.py` (A-8) · `web/js/uls_node.js` (A-5, N-2, N-9) ·
`README.md` (N-3, 2 Stellen) · `tests/test_v267_hardening.py` (neu) ·
Versionsstrings (`__init__.py` Banner v267, `pyproject.toml` 2.67.0,
`web/js/uls_compat.js` v267).
