"""
File-Utilities — Bilder laden, .txt-Dateien schreiben, Skip-Logik.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


# ---------------------------------------------------------------------------
# Bilder laden
# ---------------------------------------------------------------------------

def load_image(path: str | Path) -> Image.Image:
    """Lädt ein Bild und gibt es als PIL Image zurück."""
    img = Image.open(path)
    # EXIF-Rotation korrigieren
    try:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    return img.copy()


def collect_images(source: str | Path) -> list[Path]:
    """
    Gibt eine sortierte Liste aller Bildpfade aus einer Datei oder einem Ordner zurück.
    """
    source = Path(source)
    if source.is_file():
        if source.suffix.lower() in SUPPORTED_EXTENSIONS:
            return [source]
        return []
    if source.is_dir():
        images = sorted(
            p for p in source.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        return images
    return []


# ---------------------------------------------------------------------------
# Output-Pfad bestimmen
# ---------------------------------------------------------------------------

def get_output_path(image_path: Path, output_dir: str | Path | None) -> Path:
    """
    Bestimmt den Ordner, in dem die .txt-Datei gespeichert wird.
    Wenn output_dir None oder leer ist, wird der Bild-Ordner selbst verwendet.
    """
    if output_dir:
        d = Path(output_dir)
        d.mkdir(parents=True, exist_ok=True)
        return d
    return image_path.parent


def get_txt_path(image_path: Path, output_dir: str | Path | None) -> Path:
    """Gibt den vollständigen .txt-Ausgabepfad für ein Bild zurück."""
    out_dir = get_output_path(image_path, output_dir)
    return out_dir / (image_path.stem + ".txt")


# ---------------------------------------------------------------------------
# Skip-Logik
# ---------------------------------------------------------------------------

def should_skip(image_path: Path, output_dir: str | Path | None, overwrite: bool) -> bool:
    """
    Gibt True zurück wenn das Bild übersprungen werden soll
    (bereits verarbeitet und overwrite=False).
    """
    if overwrite:
        return False
    txt_path = get_txt_path(image_path, output_dir)
    return txt_path.exists()


# ---------------------------------------------------------------------------
# Ergebnis speichern
# ---------------------------------------------------------------------------

def save_result(
    image_path: Path,
    tags:       str,
    caption:    str,
    output_dir: str | Path | None,
) -> Path:
    """
    Speichert Tags und Caption in einer .txt-Datei.
    Format: tags\n\ncaption
    """
    txt_path = get_txt_path(image_path, output_dir)
    content  = tags + "\n\n" + caption
    txt_path.write_text(content, encoding="utf-8")
    logger.info(f"Gespeichert: {txt_path}")
    return txt_path


# ---------------------------------------------------------------------------
# Ergebnis laden (für Galerie)
# ---------------------------------------------------------------------------

def load_result(image_path: Path, output_dir: str | Path | None) -> tuple[str, str] | None:
    """
    Liest eine bereits generierte .txt-Datei.
    Gibt (tags, caption) zurück, oder None wenn nicht vorhanden.
    """
    txt_path = get_txt_path(image_path, output_dir)
    if not txt_path.exists():
        return None
    content = txt_path.read_text(encoding="utf-8")
    parts   = content.split("\n\n", 1)
    tags    = parts[0].strip() if parts else ""
    caption = parts[1].strip() if len(parts) > 1 else ""
    return tags, caption
