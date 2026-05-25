"""
main.py — Einstiegspunkt für den Batcap (PyQt6).
"""
import sys
import logging
import argparse
from pathlib import Path

# captioner/-Verzeichnis muss immer zuerst in sys.path stehen
_captioner_dir = str(Path(__file__).parent)
if _captioner_dir in sys.path:
    sys.path.remove(_captioner_dir)
sys.path.insert(0, _captioner_dir)

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt = "%H:%M:%S",
)


def main():
    parser = argparse.ArgumentParser(description="Batcap")
    parser.add_argument("--debug", action="store_true", help="Debug-Logging")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Bootstrap (ComfyUI + Modell-Manager) ─────────────────────────────
    from pipeline.bootstrap import setup
    try:
        setup()
    except RuntimeError as e:
        print(f"\n❌ Fehler beim Start:\n{e}\n")
        sys.exit(1)

    # ── PyQt6 App starten ────────────────────────────────────────────────
    from ui.app import run_app
    run_app()


if __name__ == "__main__":
    main()
