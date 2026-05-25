#!/usr/bin/env bash
# =============================================================================
# start.sh — Batcap starten
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMFY_PATH="/home/lexter/ComfyUI"

# Python-Binary aus dem ComfyUI-venv
if   [ -f "$COMFY_PATH/venv/bin/python"  ]; then PYTHON="$COMFY_PATH/venv/bin/python"
elif [ -f "$COMFY_PATH/.venv/bin/python" ]; then PYTHON="$COMFY_PATH/.venv/bin/python"
else PYTHON="python3"
     echo "⚠  Kein venv gefunden, nutze System-Python."
fi

echo "Python : $PYTHON"
echo "ComfyUI: $COMFY_PATH"
echo ""

# PyQt6 installieren falls noch nicht vorhanden
"$PYTHON" -c "import PyQt6" 2>/dev/null || {
    echo "📦  Installiere PyQt6 …"
    "$PYTHON" -m pip install PyQt6 --quiet
}

# Wechsle ins ComfyUI-Verzeichnis (wird von folder_paths.py erwartet)
cd "$COMFY_PATH"

exec "$PYTHON" "$SCRIPT_DIR/main.py" "$@"
