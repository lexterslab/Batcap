"""
pipeline/tagger_wd.py — WD EVA02-Large Tagger v3 (SmilingWolf)

Lädt via timm + HuggingFace Hub, kein ComfyUI-Custom-Node erforderlich.
Modell: SmilingWolf/wd-eva02-large-tagger-v3 (~1.26 GB F32)
Tag-Kategorien: 0 = general, 4 = character, 9 = rating
"""
from __future__ import annotations

import logging
from pathlib import Path
from PIL   import Image

logger = logging.getLogger(__name__)

_MODEL_ID  = "SmilingWolf/wd-eva02-large-tagger-v3"
_TAGS_FILE = "selected_tags.csv"

# Lokaler Cache-Pfad (optional — falls Weights manuell heruntergeladen)
_LOCAL_DIR = Path(__file__).parent.parent / "models" / "taggers" / "wd-eva02-large-tagger-v3"

_CAT_GENERAL   = 0
_CAT_CHARACTER = 4
_CAT_RATING    = 9

_model     = None
_transform = None
_tag_df    = None


# ── Laden ────────────────────────────────────────────────────────────────────

def _load() -> None:
    global _model, _transform, _tag_df
    if _model is not None:
        return

    import timm
    import pandas as pd
    from huggingface_hub import hf_hub_download
    import comfy.model_management

    logger.info("Lade WD EVA02-Large Tagger v3 …")

    local_sf = _LOCAL_DIR / "model.safetensors"
    if local_sf.exists():
        # Lokale Weights laden (ohne HF-Download)
        import safetensors.torch
        _model = timm.create_model("hf-hub:" + _MODEL_ID, pretrained=False)
        state  = safetensors.torch.load_file(str(local_sf))
        _model.load_state_dict(state)
        logger.info("WD EVA02: lokale Weights aus %s", local_sf)
    else:
        # Von HuggingFace Hub laden (wird gecacht)
        _model = timm.create_model("hf-hub:" + _MODEL_ID, pretrained=True)
        logger.info("WD EVA02: von HuggingFace Hub geladen")

    _model.eval()
    device  = comfy.model_management.get_torch_device()
    _model  = _model.to(device)

    # Transform aus Modell-Config ableiten
    data_cfg   = timm.data.resolve_model_data_config(_model)
    _transform = timm.data.create_transform(**data_cfg, is_training=False)

    # Tag-CSV
    local_csv = _LOCAL_DIR / _TAGS_FILE
    csv_path  = str(local_csv) if local_csv.exists() else \
                hf_hub_download(_MODEL_ID, _TAGS_FILE)
    _tag_df   = pd.read_csv(csv_path)

    logger.info("WD EVA02 bereit: %d Tags", len(_tag_df))


def unload() -> None:
    global _model, _transform, _tag_df
    if _model is not None:
        try:
            import torch
            _model.cpu()          # Gewichte auf CPU schieben
            del _model            # Referenz löschen → GC kann freigeben
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    _model = _transform = _tag_df = None
    logger.info("WD EVA02 entladen.")


# ── Inferenz ─────────────────────────────────────────────────────────────────

def run_wd_eva02(image: Image.Image, cfg: dict) -> tuple[str, dict]:
    """
    Klassifiziert ein Bild und gibt (tag_string, scores_dict) zurück.
    """
    _load()

    import torch
    import comfy.model_management

    threshold     = cfg.get("threshold", 0.35)
    threshold_chr = cfg.get("threshold_character", 0.75)
    topk          = cfg.get("topk", 60)
    mode          = cfg.get("mode", "topk")
    exclude       = {t.strip().lower()
                     for t in cfg.get("exclude_tags", "").split(",") if t.strip()}

    device = comfy.model_management.get_torch_device()
    tensor = _transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = _model(tensor).sigmoid().squeeze(0).float().cpu().numpy()

    df = _tag_df
    pairs: list[tuple[str, float]] = []

    for i, row in df.iterrows():
        cat   = int(row["category"])
        score = float(probs[i])
        name  = str(row["name"])

        if cat == _CAT_GENERAL   and score < threshold:     continue
        if cat == _CAT_CHARACTER and score < threshold_chr: continue
        if cat == _CAT_RATING:                              continue   # Ratings weglassen

        # Exclude-Filter (Glob-Patterns mit fnmatch)
        disp = name.replace("_", " ")
        if exclude:
            import fnmatch
            if any(fnmatch.fnmatch(disp.lower(), pat) for pat in exclude):
                continue

        pairs.append((disp, score))

    # Sortieren und ggf. auf topk begrenzen
    pairs.sort(key=lambda x: x[1], reverse=True)
    if mode == "topk" and topk > 0:
        pairs = pairs[:topk]

    scores  = {name: score for name, score in pairs}
    tag_str = ", ".join(name for name, _ in pairs)

    logger.debug("WD EVA02: %d Tags", len(pairs))
    return tag_str, scores
