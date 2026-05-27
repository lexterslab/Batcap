"""
pipeline/downloader.py — Lädt alle Tagger-Modelle nach models/taggers/.

Quellen (aus ComfyUI-RR-JointTagger/config.json):
  JTP PILOT v2  → RedRocket/JointTaggerProject  (HF)
  JTP-3 Hydra   → RedRocket/JTP-3               (HF)
  DINOv3        → lodestones/taggerine           (HF)
  WD EVA02      → SmilingWolf/wd-eva02-large-tagger-v3 (HF, via timm)

Aufruf: ensure_all_models()  — prüft und lädt fehlende Dateien.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing  import Callable

logger = logging.getLogger(__name__)

_APP_ROOT   = Path(__file__).parent.parent
_MODELS_DIR = _APP_ROOT / "models" / "taggers"
_TAGS_DIR   = _MODELS_DIR / "tags"
_HF_BASE    = "https://huggingface.co"
_HF_SPACES  = "https://huggingface.co/spaces"


# ─────────────────────────────────────────────────────────────────────────────
# Download-Manifest
# ─────────────────────────────────────────────────────────────────────────────

# Jeder Eintrag: (name, url, dest_path_relative_to_MODELS_DIR_or_TAGS_DIR, is_tags)
_MANIFEST: list[tuple[str, str, Path, bool]] = [

    # ── JTP PILOT v2 ─────────────────────────────────────────────────────────
    (
        "JTP PILOT v2 weights (~670 MB)",
        f"{_HF_BASE}/RedRocket/JointTaggerProject/resolve/main/JTP_PILOT2"
        "/JTP_PILOT2-e3-vit_so400m_patch14_siglip_384.safetensors",
        _MODELS_DIR / "JTP_PILOT2-e3-vit_so400m_patch14_siglip_384.safetensors",
        False,
    ),

    # ── JTP-3 Hydra ───────────────────────────────────────────────────────────
    (
        "JTP-3 Hydra weights (~956 MB)",
        f"{_HF_BASE}/RedRocket/JTP-3/resolve/main/models/jtp-3-hydra.safetensors",
        _MODELS_DIR / "jtp-3-hydra.safetensors",
        False,
    ),
    (
        "JTP-3 tag list",
        f"{_HF_BASE}/RedRocket/JTP-3/resolve/main/data/jtp-3-hydra-tags.csv",
        _TAGS_DIR / "jtp-3-hydra-tags.csv",
        True,
    ),
    (
        "JTP-3 validation list",
        f"{_HF_BASE}/RedRocket/JTP-3/resolve/main/data/jtp-3-hydra-val.csv",
        _TAGS_DIR / "jtp-3-hydra-val.csv",
        True,
    ),

    # ── DINOv3 / tagger-proto ─────────────────────────────────────────────────
    (
        "DINOv3 tagger-proto weights (~5.3 GB)",
        f"{_HF_BASE}/lodestones/taggerine/resolve/main/tagger_proto.safetensors",
        _MODELS_DIR / "tagger-proto.safetensors",
        False,
    ),
    (
        "DINOv3 vocab",
        f"{_HF_BASE}/lodestones/taggerine/resolve/main/tagger_vocab.json",
        _TAGS_DIR / "tagger_vocab.json",
        True,
    ),
    (
        "DINOv3 categorised vocab",
        f"{_HF_BASE}/lodestones/taggerine/resolve/main"
        "/tagger_vocab_with_categories_and_alias_updated.json",
        _TAGS_DIR / "tagger_vocab_with_categories_and_alias_updated.json",
        True,
    ),
    (
        "DINOv3 tag2implicit map",
        f"{_HF_SPACES}/silveroxides/lode-tagger-experiment/resolve/main"
        "/tag2implicit.json",
        _TAGS_DIR / "tag2implicit.json",
        True,
    ),

    # ── WD EVA02 ─────────────────────────────────────────────────────────────
    (
        "WD EVA02 weights (~1.26 GB)",
        f"{_HF_BASE}/SmilingWolf/wd-eva02-large-tagger-v3/resolve/main"
        "/model.safetensors",
        _MODELS_DIR / "wd-eva02-large-tagger-v3" / "model.safetensors",
        False,
    ),
    (
        "WD EVA02 tag list",
        f"{_HF_BASE}/SmilingWolf/wd-eva02-large-tagger-v3/resolve/main"
        "/selected_tags.csv",
        _MODELS_DIR / "wd-eva02-large-tagger-v3" / "selected_tags.csv",
        False,
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Download-Kern
# ─────────────────────────────────────────────────────────────────────────────

def _download(
    url:      str,
    dest:     Path,
    name:     str,
    progress: Callable[[str], None] | None = None,
) -> bool:
    """
    Lädt url → dest. Unterstützt Resume via Range-Header.
    Gibt True zurück wenn erfolgreich, False bei Fehler.
    """
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    headers = {}
    resume_pos = 0
    if tmp.exists():
        resume_pos = tmp.stat().st_size
        headers["Range"] = f"bytes={resume_pos}-"

    try:
        r = requests.get(url, headers=headers, stream=True, timeout=60)
        if r.status_code == 416:
            # Already fully downloaded to tmp
            shutil.move(str(tmp), str(dest))
            return True
        r.raise_for_status()

        total  = int(r.headers.get("Content-Length", 0)) + resume_pos
        mode   = "ab" if resume_pos else "wb"
        done   = resume_pos
        last_pct = -1

        with open(tmp, mode) as f:
            for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB
                if chunk:
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = int(done * 100 / total)
                        if pct != last_pct and pct % 5 == 0:
                            msg = f"  {name}: {pct}% ({done/1e6:.0f}/{total/1e6:.0f} MB)"
                            logger.info(msg)
                            if progress:
                                progress(msg)
                            last_pct = pct

        shutil.move(str(tmp), str(dest))
        logger.info("✓ %s → %s", name, dest.name)
        return True

    except Exception as e:
        logger.error("✗ %s: %s", name, e)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Öffentliche API
# ─────────────────────────────────────────────────────────────────────────────

def missing_files() -> list[tuple[str, str, Path, bool]]:
    """Gibt nur die Einträge zurück die noch nicht lokal vorhanden sind."""
    return [(n, u, d, t) for n, u, d, t in _MANIFEST if not d.exists()]


def ensure_all_models(
    progress: Callable[[str], None] | None = None,
    skip_keys: set[str] | None = None,
) -> dict[str, bool]:
    """
    Lädt alle fehlenden Modell-Dateien herunter.
    skip_keys: Menge von Modell-Schlüsselwörtern die übersprungen werden
               (z.B. {"wd_eva02"} wenn WD EVA02 nicht benötigt wird).
    Gibt dict {name: success} zurück.
    """
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    _TAGS_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, bool] = {}
    todo = missing_files()

    if not todo:
        logger.info("Alle Tagger-Modelle vorhanden ✓")
        return {}

    logger.info("%d Datei(en) fehlen — starte Download …", len(todo))

    for name, url, dest, _ in todo:
        # Skip-Filter
        if skip_keys:
            lower = name.lower()
            if any(k.lower() in lower for k in skip_keys):
                logger.info("Übersprungen (deaktiviert): %s", name)
                continue

        if progress:
            progress(f"Lade: {name}")
        results[name] = _download(url, dest, name, progress)

    ok  = sum(v for v in results.values())
    err = len(results) - ok
    logger.info("Downloads abgeschlossen: %d OK, %d Fehler", ok, err)
    return results


def download_summary() -> str:
    """Gibt einen lesbaren Status-String zurück."""
    missing = missing_files()
    if not missing:
        return "Alle Tagger-Modelle vorhanden."
    names = "\n  • ".join(n for n, *_ in missing)
    return f"Fehlende Dateien ({len(missing)}):\n  • {names}"
