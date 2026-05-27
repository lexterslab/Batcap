"""
Tagger — führt alle aktiven Modelle auf einem PIL-Bild aus.
Deaktivierte Tagger (enabled: false in settings.json) werden übersprungen.
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


def reload_settings() -> None:
    global _SETTINGS
    _SETTINGS = None


def _enabled(key: str) -> bool:
    return _s()["models"].get(key, {}).get("enabled", True)


def _resolve_model(display_name: str) -> tuple[str, str]:
    from pipeline.bootstrap import get_model_path, get_tags_path
    model_key = get_model_path(display_name)
    tags_key  = get_tags_path(display_name)
    if model_key:
        return model_key, tags_key or model_key
    logger.warning("'%s' nicht in config.json — übergebe direkt.", display_name)
    return display_name, display_name


# ── JTP PILOT v2 ─────────────────────────────────────────────────────────────

def run_jtp_pilot2(image: Image.Image) -> tuple[str, dict]:
    from rr_joint_tagger.redrocket.classifier import JtpInference
    import comfy.model_management

    cfg  = _s()["models"]["jtp_pilot2"]
    dev  = comfy.model_management.get_torch_device()
    mn, tn = _resolve_model(cfg["model"])

    tags_str, scores = JtpInference.run_classifier(
        model_name         = mn,
        tags_name          = tn,
        device             = dev,
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
    logger.debug("JTP PILOT v2: %d Tags", len(tags_str.split(",")))
    return tags_str, scores


# ── JTP-3 Hydra ──────────────────────────────────────────────────────────────

def run_jtp3_hydra(image: Image.Image) -> tuple[str, dict]:
    from rr_joint_tagger.redrocket.classifier_v3 import JtpInferenceV3
    import comfy.model_management

    cfg  = _s()["models"]["jtp3_hydra"]
    dev  = comfy.model_management.get_torch_device()

    tags_str, scores, _ = JtpInferenceV3.run_classifier(
        model_name         = cfg["model"],
        device             = dev,
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
    logger.debug("JTP-3 Hydra: %d Tags", len(tags_str.split(",")))
    return tags_str, scores


# ── DINOv3 ────────────────────────────────────────────────────────────────────

def run_dinov3(image: Image.Image) -> tuple[str, dict]:
    from rr_joint_tagger.redrocket.classifier_dino import DINOv3Inference
    import comfy.model_management

    cfg  = _s()["models"]["dinov3"]
    dev  = comfy.model_management.get_torch_device()

    tags_str, scores = DINOv3Inference.run_classifier(
        model_name         = cfg["model"],
        device             = dev,
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
    logger.debug("DINOv3: %d Tags", len(tags_str.split(",")))
    return tags_str, scores


# ── WD EVA02 ──────────────────────────────────────────────────────────────────

def run_wd_eva02(image: Image.Image) -> tuple[str, dict]:
    from pipeline.tagger_wd import run_wd_eva02 as _run
    return _run(image, _s()["models"]["wd_eva02"])


# ── Cleanup ───────────────────────────────────────────────────────────────────

def unload_all_taggers() -> None:
    """
    Entlädt alle Tagger-Modelle vollständig aus dem VRAM.
    Aufruf: nach Abschluss eines Tagging-Batches.
    """
    try:
        from pipeline.tagger_wd import unload as _unload_wd
        _unload_wd()                     # WD EVA02 — nicht durch ComfyUI verwaltet
    except Exception as e:
        logger.debug("WD EVA02 unload: %s", e)

    try:
        import comfy.model_management, torch
        comfy.model_management.unload_all_models()   # JTP v2/v3, DINOv3
        comfy.model_management.soft_empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        logger.info("Alle Tagger entladen.")
    except Exception as e:
        logger.warning("Tagger-Cleanup: %s", e)


# ── Alle aktiven Tagger zusammen ─────────────────────────────────────────────

def run_all(image: Image.Image) -> dict[str, tuple[str, dict]]:
    """
    Führt alle aktivierten Tagger aus.
    Gibt dict zurück: {"jtp_pilot2": (tags, scores), ...}
    Deaktivierte Tagger: ("", {})
    """
    # Qwen aus vorherigem Bild aus VRAM räumen
    try:
        import comfy.model_management, torch
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass

    empty = ("", {})
    results: dict[str, tuple[str, dict]] = {}

    if _enabled("jtp_pilot2"):
        logger.info("Starte JTP PILOT v2 …")
        results["jtp_pilot2"] = run_jtp_pilot2(image)
    else:
        results["jtp_pilot2"] = empty

    if _enabled("jtp3_hydra"):
        logger.info("Starte JTP-3 Hydra …")
        results["jtp3_hydra"] = run_jtp3_hydra(image)
    else:
        results["jtp3_hydra"] = empty

    if _enabled("dinov3"):
        logger.info("Starte DINOv3 …")
        results["dinov3"] = run_dinov3(image)
    else:
        results["dinov3"] = empty

    if _enabled("wd_eva02"):
        logger.info("Starte WD EVA02 …")
        results["wd_eva02"] = run_wd_eva02(image)
    else:
        results["wd_eva02"] = empty

    active = [k for k, (t, _) in results.items() if t]
    logger.info("Tagger fertig: %s", ", ".join(active) if active else "keiner aktiv")
    return results
