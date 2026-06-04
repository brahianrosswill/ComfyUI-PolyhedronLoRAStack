"""
Polyhedron LoRA Stack — install.py
══════════════
Wird von ComfyUI Manager automatisch beim Installieren der Custom Node
ausgeführt. Installiert alle Python-Abhängigkeiten.

Kann auch manuell aufgerufen werden:
  python install.py
"""

import sys
import subprocess
import importlib.util
from pathlib import Path

REQUIRED = [
    ("PIL",      "Pillow>=9.0.0",    "Preview-Generierung (Info-Cards)"),
    ("requests", "requests>=2.28.0", "Civitai-API + Preview-Download"),
]


def is_installed(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def install_package(pip_spec: str) -> bool:
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            pip_spec,
            "--quiet",
            "--no-warn-script-location",
        ])
        return True
    except subprocess.CalledProcessError:
        return False


def main():
    print("\n[PLS] Installiere Abhängigkeiten...")
    all_ok = True

    for module, pip_spec, description in REQUIRED:
        if is_installed(module):
            print(f"  ✓ {pip_spec.split('>=')[0]:<12} bereits installiert")
            continue

        print(f"  ⬇ {pip_spec:<24} ({description})")
        ok = install_package(pip_spec)
        if ok:
            print(f"  ✓ {pip_spec.split('>=')[0]} installiert")
        else:
            print(f"  ✗ Fehler bei {pip_spec} — bitte manuell installieren:")
            print(f"    pip install {pip_spec}")
            all_ok = False

    if all_ok:
        print("\n[PLS] ✅ Alle Abhängigkeiten bereit.\n")
    else:
        print("\n[PLS] ⚠ Einige Pakete fehlen — Node läuft im eingeschränkten Modus.\n")

    return all_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
