"""
Phasenbasierte Batch-Verarbeitung für den Captioner.

Phase 1 — Tag it!:   Alle 3 Tagger laufen über alle Bilder.
                      Ergebnisse in .batcap.json neben dem Bild gespeichert.
Phase 2 — Cap it!:   Qwen lädt einmal, liest Tags aus .batcap.json,
                      generiert Captions für alle Bilder.
Tag & Cap it!:        Phase 1 direkt gefolgt von Phase 2.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib    import Path
from PIL        import Image

from cap_app.file_utils import load_image, collect_images, save_result

logger = logging.getLogger(__name__)

_STATE_SUFFIX = ".batcap.json"


# ── Zwischenspeicher pro Bild ─────────────────────────────────────────────────

def _state_path(image_path: Path) -> Path:
    return image_path.parent / (image_path.stem + _STATE_SUFFIX)

def _save_state(image_path: Path, data: dict) -> None:
    _state_path(image_path).write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

def _load_state(image_path: Path) -> dict:
    p = _state_path(image_path)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def save_edited_result(image_path: Path, tags: str, caption: str,
                       output_dir: str | None = None) -> None:
    """Speichert editierte Tags/Caption (Editor-Button) zurück."""
    data = _load_state(image_path)
    data["tags_clean"] = tags
    data["caption"]    = caption
    data["status"]     = "complete" if (tags and caption) else data.get("status", "tagged")
    _save_state(image_path, data)
    save_result(image_path, tags, caption, output_dir)

def load_existing(image_path: Path, output_dir: str | None = None) -> tuple[str, str]:
    """Lädt gespeicherte Tags+Caption für die Vorschau beim Bildklick."""
    data = _load_state(image_path)
    tags    = data.get("tags_clean", "")
    caption = data.get("caption", "")
    # Fallback: .txt lesen
    if not tags and not caption:
        txt = (Path(output_dir) if output_dir else image_path.parent) \
              / (image_path.stem + ".txt")
        if txt.exists():
            parts   = txt.read_text(encoding="utf-8").strip().split("\n\n", 1)
            tags    = parts[0]
            caption = parts[1] if len(parts) > 1 else ""
    return tags, caption


# ── Ergebnis-Objekt ───────────────────────────────────────────────────────────

@dataclass
class JobResult:
    image_path: Path
    tags:    str  = ""
    caption: str  = ""
    skipped: bool = False
    error:   str  = ""
    success: bool = False
    phase:   str  = ""   # "tag" | "cap"


# ── Stop-Zustand ──────────────────────────────────────────────────────────────

@dataclass
class BatchState:
    total:     int  = 0
    done:      int  = 0
    running:   bool = False
    stop_flag: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def request_stop(self):
        with self._lock: self.stop_flag = True
    def should_stop(self) -> bool:
        with self._lock: return self.stop_flag
    def reset(self, total: int):
        with self._lock:
            self.total = total; self.done = 0
            self.running = True; self.stop_flag = False

state = BatchState()


# ── Bilder sammeln ────────────────────────────────────────────────────────────

def _images(source: str | list[str]) -> list[Path]:
    if isinstance(source, list):
        return [Path(p) for p in source]
    return collect_images(source)


def _cleanup_taggers() -> None:
    """Entlädt alle Tagger nach einem Batch. Sicher wenn nichts geladen."""
    try:
        from pipeline.tagger import unload_all_taggers
        unload_all_taggers()
    except Exception as e:
        logger.warning("Tagger-Cleanup: %s", e)


def _cleanup_captioner() -> None:
    """Entlädt Qwen nach einem Batch. Sicher wenn nicht geladen."""
    try:
        from pipeline.captioner import unload_model
        unload_model()
    except Exception as e:
        logger.warning("Captioner-Cleanup: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Tagging
# ─────────────────────────────────────────────────────────────────────────────

def run_tag_batch(
    source:     str | list[str],
    output_dir: str | None = None,
    prompt_key: str        = "photorealism",
):
    """
    Lädt jeden Tagger EINMAL und lässt ihn über alle Bilder laufen.
    Speichert Tags+Merged in .batcap.json.
    """
    from pipeline.tagger  import run_all
    from pipeline.merger  import merge_all
    from pipeline.cleanup import clean_tags

    images = _images(source)
    if not images:
        return
    state.reset(len(images))
    logger.info("Tag-Phase: %d Bilder", len(images))

    for img_path in images:
        if state.should_stop():
            break
        r = JobResult(image_path=img_path, phase="tag")
        try:
            image   = load_image(img_path)
            results = run_all(image)
            merged  = merge_all(results)

            # Einzelne Tagger-Ergebnisse für .batcap.json auslesen
            j2t,  j2s  = results.get("jtp_pilot2", ("", {}))
            j3t,  j3s  = results.get("jtp3_hydra",  ("", {}))
            dt,   ds   = results.get("dinov3",       ("", {}))
            wdt,  wds  = results.get("wd_eva02",     ("", {}))
            clean  = clean_tags(merged)

            data = _load_state(img_path)
            data.update({
                "jtp2_tags":  j2t,  "jtp2_scores":  j2s,
                "jtp3_tags":  j3t,  "jtp3_scores":  j3s,
                "dino_tags":  dt,   "dino_scores":  ds,
                "wd_tags":    wdt,  "wd_scores":    wds,
                "tags_merged": merged, "tags_clean": clean,
                "status": "tagged",
            })
            _save_state(img_path, data)
            # .txt schreiben (Tags ohne Caption) — sichtbar beim Bildklick und für externe Tools
            save_result(img_path, clean, "", output_dir)

            r.tags = clean; r.success = True
        except Exception as e:
            logger.exception("Tag-Fehler %s: %s", img_path.name, e)
            r.error = str(e)
        state.done += 1
        yield r

    # Tagger nach Batch vollständig entladen
    _cleanup_taggers()
    state.running = False


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Captioning
# ─────────────────────────────────────────────────────────────────────────────

def run_cap_batch(
    source:     str | list[str],
    output_dir: str | None = None,
    prompt_key: str        = "photorealism",
):
    """
    Lädt Qwen EINMAL und captioniert alle Bilder.
    Tags werden aus .batcap.json gelesen.
    """
    from pipeline.captioner import generate_caption

    images = _images(source)
    if not images:
        return
    state.reset(len(images))
    logger.info("Cap-Phase: %d Bilder", len(images))

    for img_path in images:
        if state.should_stop():
            break
        r = JobResult(image_path=img_path, phase="cap")
        data = _load_state(img_path)
        tags = data.get("tags_clean", "")

        if not tags:
            logger.warning("Keine Tags für %s — übersprungen", img_path.name)
            r.skipped = True; r.success = True
            state.done += 1
            yield r
            continue

        try:
            image   = load_image(img_path)
            caption = generate_caption(image, tags, prompt_key)
            data["caption"] = caption
            data["status"]  = "complete"
            _save_state(img_path, data)
            save_result(img_path, tags, caption, output_dir)
            r.tags = tags; r.caption = caption; r.success = True
        except Exception as e:
            logger.exception("Cap-Fehler %s: %s", img_path.name, e)
            r.error = str(e)
        state.done += 1
        yield r

    # Qwen nach Batch vollständig entladen
    _cleanup_captioner()
    state.running = False


# ─────────────────────────────────────────────────────────────────────────────
# Tag & Cap kombiniert
# ─────────────────────────────────────────────────────────────────────────────

def run_batch(
    source:     str | list[str],
    output_dir: str | None = None,
    prompt_key: str        = "photorealism",
):
    """Erst alle Bilder taggen, dann alle captionieren."""
    yield from run_tag_batch(source, output_dir, prompt_key)
    if not state.should_stop():
        yield from run_cap_batch(source, output_dir, prompt_key)
