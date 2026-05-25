"""
Tagger — führt alle drei Modelle auf einem PIL-Bild aus.
Verwendet bootstrap.get_model_path() statt ComfyExtensionConfig,
um den ComfyUI-Server-Import zu vermeiden.
"""
import logging
import json
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)

_SETTINGS: dict | None = None


def _s() -> dict:
    global _SETTINGS
    if _SETTINGS is None:
        p = Path(__file__).parent.parent / "config" / "settings.json"
        with open(p, encoding="utf-8") as f:
            _SETTINGS = json.load(f)
    return _SETTINGS


def _resolve_model(display_name: str) -> tuple[str, str]:
    """
    Gibt (model_name, tags_name) zurück — das sind die Config-Schlüssel
    die JtpModelManager/JtpTagManager intern erwarten.
    """
    from pipeline.bootstrap import get_model_path, get_tags_path
    model_key = get_model_path(display_name)
    tags_key  = get_tags_path(display_name)
    if model_key:
        logger.info(f"Model resolved: '{display_name}' → key='{model_key}'")
        return model_key, tags_key or model_key
    # Fallback: display_name direkt weitergeben
    logger.warning(
        f"'{display_name}' nicht in config.json — "
        f"übergebe direkt. Bitte settings.json prüfen."
    )
    return display_name, display_name


# ---------------------------------------------------------------------------
# JTP PILOT v2
# ---------------------------------------------------------------------------

def run_jtp_pilot2(image: Image.Image) -> tuple[str, dict]:
    from rr_joint_tagger.redrocket.classifier import JtpInference
    import comfy.model_management

    cfg        = _s()["models"]["jtp_pilot2"]
    device     = comfy.model_management.get_torch_device()
    model_name, tags_name = _resolve_model(cfg["model"])

    tags_str, scores = JtpInference.run_classifier(
        model_name         = model_name,
        tags_name          = tags_name,
        device             = device,
        image              = image.convert("RGBA"),
        steps              = cfg.get("steps", 255),
        threshold          = cfg["threshold"],
        exclude_tags       = cfg.get("exclude_tags", ""),
        replace_underscore = cfg.get("replace_underscore", False),
        trailing_comma     = cfg.get("trailing_comma", False),
        implications_mode  = cfg.get("implications_mode", "off"),
        exclude_categories = cfg.get("exclude_categories", ""),
        prefix             = cfg.get("prefix", ""),
        mode               = cfg.get("mode", "topk"),
        topk               = cfg.get("topk", 40),
        max_tags           = cfg.get("max_tags", 0),
    )
    logger.debug(f"JTP PILOT v2: {len(tags_str.split(','))} Tags")
    return tags_str, scores


# ---------------------------------------------------------------------------
# JTP-3 Hydra
# ---------------------------------------------------------------------------

def run_jtp3_hydra(image: Image.Image) -> tuple[str, dict]:
    from rr_joint_tagger.redrocket.classifier_v3 import JtpInferenceV3
    import comfy.model_management

    cfg    = _s()["models"]["jtp3_hydra"]
    device = comfy.model_management.get_torch_device()

    tags_str, scores, _cam = JtpInferenceV3.run_classifier(
        model_name         = cfg["model"],
        device             = device,
        image              = image.convert("RGBA"),
        threshold          = cfg["threshold"],
        cam_depth          = cfg.get("cam_depth", 1),
        seqlen             = cfg.get("seqlen", 1024),
        implications_mode  = cfg.get("implications_mode", "off"),
        exclude_tags       = cfg.get("exclude_tags", ""),
        exclude_categories = cfg.get("exclude_categories", ""),
        prefix             = cfg.get("prefix", ""),
        original_tags      = cfg.get("original_tags", False),
        seed               = 0,
        replace_underscore = cfg.get("replace_underscore", False),
        trailing_comma     = cfg.get("trailing_comma", False),
        cam_mode           = cfg.get("cam_mode", "none"),
        cam_tag            = cfg.get("cam_tag", ""),
        mode               = cfg.get("mode", "topk"),
        topk               = cfg.get("topk", 40),
        max_tags           = cfg.get("max_tags", 0),
    )
    logger.debug(f"JTP-3 Hydra: {len(tags_str.split(','))} Tags")
    return tags_str, scores


# ---------------------------------------------------------------------------
# DINOv3 / Taggerine
# ---------------------------------------------------------------------------

def run_dinov3(image: Image.Image) -> tuple[str, dict]:
    from rr_joint_tagger.redrocket.classifier_dino import DINOv3Inference
    import comfy.model_management

    cfg    = _s()["models"]["dinov3"]
    device = comfy.model_management.get_torch_device()

    tags_str, scores = DINOv3Inference.run_classifier(
        model_name         = cfg["model"],
        device             = device,
        image              = image.convert("RGB"),
        mode               = cfg.get("mode", "topk"),
        topk               = cfg.get("topk", 40),
        threshold          = cfg["threshold"],
        max_size           = cfg.get("max_size", 1024),
        implications_mode  = cfg.get("implications_mode", "off"),
        exclude_tags       = cfg.get("exclude_tags", ""),
        exclude_categories = cfg.get("exclude_categories", ""),
        max_tags           = cfg.get("max_tags", 0),
        trailing_comma     = cfg.get("trailing_comma", False),
        prefix             = cfg.get("prefix", ""),
        seed               = 0,
        category_config    = None,
        check_updates      = cfg.get("check_updates", False),
        use_aliases        = cfg.get("use_aliases", False),
    )
    logger.debug(f"DINOv3: {len(tags_str.split(','))} Tags")
    return tags_str, scores


# ---------------------------------------------------------------------------
# Alle drei zusammen
# ---------------------------------------------------------------------------

def run_all(image: Image.Image) -> tuple:
    # Qwen (aus vorherigem Bild) aus dem VRAM räumen bevor DINOv3 (~5,3 GB) lädt.
    # Qwen und DINOv3 passen nicht gleichzeitig in den VRAM (14 + 5 GB > 15,8 GB).
    # generate_caption() lädt Qwen beim nächsten Durchlauf automatisch neu.
    try:
        import comfy.model_management, torch
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass

    logger.info("Starte JTP PILOT v2 …")
    jtp2 = run_jtp_pilot2(image)

    logger.info("Starte JTP-3 Hydra …")
    jtp3 = run_jtp3_hydra(image)

    logger.info("Starte DINOv3 …")
    dino = run_dinov3(image)

    return jtp2, jtp3, dino
