"""
Captioner — lädt Qwen 3.5 über die comfy-Bibliothek und generiert Captions.
Repliziert die TextGenerateQwen35SystemPrompt-Logik aus SamplingUtils von Silveroxides.
"""
from __future__ import annotations
import json
import logging
import math
import numpy as np
import torch
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)

_SETTINGS: dict | None  = None
_PROMPTS:  dict | None  = None
_clip_model              = None   # gecachtes Qwen-Modell
_clip_model_file: str | None = None  # welche Datei gerade geladen ist


# ---------------------------------------------------------------------------
# Settings & Prompts
# ---------------------------------------------------------------------------

def _s() -> dict:
    global _SETTINGS
    if _SETTINGS is None:
        p = Path(__file__).parent.parent / "config" / "settings.json"
        with open(p, encoding="utf-8") as f:
            _SETTINGS = json.load(f)
    return _SETTINGS


def _p() -> dict:
    global _PROMPTS
    if _PROMPTS is None:
        p = Path(__file__).parent.parent / "config" / "prompts.json"
        with open(p, encoding="utf-8") as f:
            _PROMPTS = json.load(f)
    return _PROMPTS


def reload_prompts() -> None:
    """Lädt die Prompts-Datei neu (z.B. nach Änderungen in der UI)."""
    global _PROMPTS
    _PROMPTS = None


def reload_settings() -> None:
    """Settings-Cache leeren — wird nach Modelländerung aufgerufen."""
    global _SETTINGS
    _SETTINGS = None


# ---------------------------------------------------------------------------
# Modell-Scanner
# ---------------------------------------------------------------------------

def list_available_models() -> list[str]:
    """
    Scannt alle text_encoders-Verzeichnisse von ComfyUI nach .safetensors-Dateien.
    Unterstützte Formate: fp16, bf16, fp8, mxfp8, int8 — ComfyUI erkennt
    das Quantisierungsformat automatisch aus dem safetensors-Header.
    Gibt relative Pfade zurück (so wie sie in settings.json stehen).
    """
    try:
        import folder_paths
        models: list[str] = []
        for base_str in folder_paths.get_folder_paths("text_encoders"):
            base = Path(base_str)
            if not base.exists():
                continue
            for f in sorted(base.rglob("*.safetensors")):
                rel = str(f.relative_to(base))
                if rel not in models:
                    models.append(rel)
        return models if models else ["(keine Modelle gefunden)"]
    except Exception as e:
        logger.warning(f"Modell-Scan fehlgeschlagen: {e}")
        return []


# ---------------------------------------------------------------------------
# Silveroxides-kompatibler Loader: Hilfsfunktionen
# ---------------------------------------------------------------------------

def _find_silveroxides_pkg(comfy_path: str) -> Path | None:
    """Findet den Ordner mit unified_ops.py (Silveroxides' Paket-Root)."""
    base = Path(comfy_path) / "custom_nodes"
    if not base.exists():
        return None
    for folder in sorted(base.iterdir()):
        if not folder.is_dir():
            continue
        if (folder / "unified_ops.py").exists():
            return folder
        # Eine Ebene tiefer (z.B. pkg/nodes/loader_nodes.py → pkg/unified_ops.py)
        for sub in folder.iterdir():
            if sub.is_dir() and (sub / "unified_ops.py").exists():
                return sub
    return None


def _try_get_quant_ops(comfy_path: str):
    """
    Versucht make_quant_ops aus Silveroxides' unified_ops zu importieren.
    Gibt die Funktion zurück oder None wenn nicht verfügbar.
    """
    try:
        pkg_root = _find_silveroxides_pkg(comfy_path)
        if pkg_root is None:
            logger.debug("unified_ops.py nicht gefunden.")
            return None

        import sys, importlib.util, types

        pkg_name = pkg_root.name.replace("-", "_")

        # Paket und alle Unterordner als Namespace registrieren
        if pkg_name not in sys.modules:
            mod = types.ModuleType(pkg_name)
            mod.__path__    = [str(pkg_root)]
            mod.__package__ = pkg_name
            sys.modules[pkg_name] = mod
            for sub in pkg_root.iterdir():
                if sub.is_dir():
                    sn = f"{pkg_name}.{sub.name}"
                    if sn not in sys.modules:
                        sm = types.ModuleType(sn)
                        sm.__path__    = [str(sub)]
                        sm.__package__ = sn
                        sys.modules[sn] = sm

        # unified_ops.py laden
        spec = importlib.util.spec_from_file_location(
            f"{pkg_name}.unified_ops",
            str(pkg_root / "unified_ops.py"),
        )
        umod = importlib.util.module_from_spec(spec)
        umod.__package__ = pkg_name
        sys.modules[f"{pkg_name}.unified_ops"] = umod
        spec.loader.exec_module(umod)

        fn = getattr(umod, "make_quant_ops", None)
        if fn:
            logger.info(f"make_quant_ops geladen aus: {pkg_root.name}")
        return fn

    except Exception as e:
        logger.debug(f"make_quant_ops nicht verfügbar: {e}")
        return None


def _detect_llama_quant(sd: dict) -> dict:
    """
    Erkennt Quantisierung in LLaMA/Qwen-State-Dicts (vereinfachte Version
    von Silveroxides' _detect_te_quantization).
    Gibt model_options-Ergänzungen zurück oder leeres Dict.
    """
    import torch

    has_comfy_quant = any(k.endswith(".comfy_quant") for k in sd)

    has_int8 = any(
        sd[k].dtype == torch.int8
        for k in sd if k.endswith(".weight")
    )

    try:
        _fp8 = (torch.float8_e4m3fn, torch.float8_e5m2)
        has_fp8 = any(sd[k].dtype in _fp8 for k in sd if k.endswith(".weight"))
    except AttributeError:
        has_fp8 = False

    if not (has_comfy_quant or has_int8 or has_fp8):
        return {}

    # Dtype aus Normalisierungs-Layern ableiten
    dtype = torch.float16
    for k, v in sd.items():
        if k.endswith(".weight") and "norm" in k.lower():
            dtype = v.dtype
            break

    return {
        "llama_quantization_metadata": {"mixed_ops": True},
        "quantization_metadata":       {"mixed_ops": True},
        "dtype_llama": dtype,
    }


def _configure_int8_backend(kernel_backend: str) -> None:
    """Konfiguriert comfy_kitchen Backend für INT8 (1:1 aus Silveroxides' Loader)."""
    try:
        import comfy_kitchen as ck
        if kernel_backend == "triton":
            ck.set_backend_priority(["triton", "cuda", "eager"])
        else:
            ck.set_backend_priority(["cuda", "triton", "eager"])
        logger.debug(f"comfy_kitchen Backend: {kernel_backend}")
    except (ImportError, Exception) as e:
        logger.debug(f"comfy_kitchen nicht verfügbar: {e}")


# ---------------------------------------------------------------------------
# Qwen-Modell laden (Singleton mit Reload-Logik)
# ---------------------------------------------------------------------------

def _get_clip():
    global _clip_model, _clip_model_file

    cfg        = _s()["models"]["qwen"]
    model_file = cfg["model_file"]

    if _clip_model is not None and _clip_model_file == model_file:
        return _clip_model

    if _clip_model is not None:
        logger.info(f"Modell wechselt: {_clip_model_file} → {model_file}")
        unload_model()

    import comfy.sd
    import comfy.model_management
    import comfy.utils
    import folder_paths

    quant_format   = cfg.get("quant_format", "auto")
    kernel_backend = cfg.get("kernel_backend", "pytorch")
    comfy_path     = _s()["comfyui_path"]

    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()

    logger.info(f"Lade Modell: {model_file}  [format={quant_format}, backend={kernel_backend}] …")

    model_path = folder_paths.get_full_path("text_encoders", model_file)
    if model_path is None:
        raise FileNotFoundError(
            f"Modell nicht gefunden: {model_file}\n"
            f"Erwartet in: {folder_paths.get_folder_paths('text_encoders')}"
        )

    # ── Optimierter Lader (nach Silveroxides' QuantizedCLIPLoader) ─────
    try:
        # 1. State-Dict laden (mit Metadaten)
        sd, metadata = comfy.utils.load_torch_file(
            model_path, safe_load=True, return_metadata=True
        )

        # 2. Quantisierungsinfo aus State-Dict erkennen
        quant_info = _detect_llama_quant(sd)
        if quant_info:
            logger.info(f"Quantisierung erkannt: {list(quant_info.keys())}")

        # 3. model_options aufbauen
        model_options: dict = {
            "initial_device": comfy.model_management.text_encoder_offload_device()
        }
        model_options.update(quant_info)

        # INT8-Backend konfigurieren
        if quant_format in ("int8", "int8_tensorwise") or (
            quant_format == "auto" and
            any(sd[k].dtype == __import__("torch").int8
                for k in sd if k.endswith(".weight"))
        ):
            _configure_int8_backend(kernel_backend)

        # make_quant_ops einhängen (Silveroxides' optimierte Kernel)
        make_quant_ops = _try_get_quant_ops(comfy_path)
        if make_quant_ops is not None:
            model_options["custom_operations"] = make_quant_ops(
                model_options.get("custom_operations")
            )
        elif quant_info:
            # Fallback: ComfyUI soll seine eigenen Quant-Ops verwenden
            logger.info("make_quant_ops nicht gefunden — nutze ComfyUI built-in Quant-Ops")

        # 4. Schnelles Laden: state_dicts + disable_dynamic=True
        _clip_model = comfy.sd.load_text_encoder_state_dicts(
            state_dicts         = [sd],
            clip_type           = comfy.sd.CLIPType.STABLE_DIFFUSION,
            model_options       = model_options,
            embedding_directory = folder_paths.get_folder_paths("embeddings"),
            disable_dynamic     = True,
        )
        logger.info("Modell geladen (optimierter Pfad).")

    except Exception as e:
        logger.warning(
            f"Optimierter Loader fehlgeschlagen ({e}), "
            f"nutze Standard-load_clip …"
        )
        _clip_model = comfy.sd.load_clip(
            ckpt_paths          = [model_path],
            embedding_directory = folder_paths.get_folder_paths("embeddings"),
            clip_type           = comfy.sd.CLIPType.STABLE_DIFFUSION,
        )
        logger.info("Modell geladen (Standard-Pfad).")

    _clip_model_file = model_file
    return _clip_model


def unload_model() -> None:
    """Gibt Qwen vollständig aus dem VRAM frei."""
    global _clip_model, _clip_model_file
    if _clip_model is not None:
        try:
            import comfy.model_management, torch
            comfy.model_management.unload_all_models()
            comfy.model_management.soft_empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception as e:
            logger.debug("Qwen-Unload: %s", e)
        _clip_model      = None
        _clip_model_file = None
        logger.info("Qwen aus VRAM entladen.")


# ---------------------------------------------------------------------------
# Chat-Template (1:1-Port von TextGenerateQwen35SystemPrompt._build_prompt)
# ---------------------------------------------------------------------------

def _build_prompt(
    prompt:         str,
    system_message: str,
    has_image:      bool,
    thinking:       bool = False,
) -> str:
    """
    Baut den vollständigen Qwen3.5-Chat-Template-String auf.
    Identisch mit TextGenerateQwen35SystemPrompt._build_prompt() aus SamplingUtils.
    """
    result = ""

    # Optionaler System-Block
    if system_message and system_message.strip():
        result = "<|im_start|>system\n" + system_message + "<|im_end|>\n"

    # User-Block — Bild-Token kommt vor dem Text (qwen35.py:761)
    result += "<|im_start|>user\n"
    if has_image:
        result += "<|vision_start|><|image_pad|><|vision_end|>"
    result += prompt + "<|im_end|>\n"

    # Assistant-Block
    result += "<|im_start|>assistant\n"

    # Thinking unterdrücken (qwen35.py:784-785)
    if not thinking:
        result += "<think>\n</think>\n"

    return result


# ---------------------------------------------------------------------------
# Bild für Qwen vorbereiten
# ---------------------------------------------------------------------------

def _prepare_image(image: Image.Image) -> torch.Tensor:
    """
    Skaliert das Bild auf max. ~1 MP und konvertiert zu ComfyUI-Tensor
    [1, H, W, C] float32 im Bereich [0, 1].
    """
    cfg        = _s().get("image_processing", {})
    max_pixels = cfg.get("max_pixels", 1048576)

    w, h = image.size
    total = w * h
    if total > max_pixels:
        scale = math.sqrt(max_pixels / total)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        # Auf 16er-Raster abrunden (wie AdjustedResolutionParameters)
        new_w = (new_w // 16) * 16
        new_h = (new_h // 16) * 16
        image = image.resize((new_w, new_h), Image.LANCZOS)
        logger.debug(f"Bild skaliert: {w}×{h} → {new_w}×{new_h}")

    arr = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)  # [1, H, W, C]


# ---------------------------------------------------------------------------
# Caption generieren
# ---------------------------------------------------------------------------

def generate_caption(
    image:        Image.Image,
    tags:         str,
    prompt_key:   str | None = None,
) -> str:
    """
    Generiert eine Caption für das Bild.

    Args:
        image:      PIL-Bild (beliebige Größe)
        tags:       Bereinigte Tag-String (mit Leerzeichen, kein Underscore)
        prompt_key: Key aus prompts.json ("photorealism" / "artwork").
                    Wenn None, wird der aktive Prompt aus der Config verwendet.

    Returns:
        Generierter Caption-Text.
    """
    prompts_cfg = _p()
    key         = prompt_key or prompts_cfg.get("active_prompt", "photorealism")
    prompt_data = prompts_cfg["prompts"].get(key, {})

    system_message  = prompt_data.get("system_message", "")
    tag_prefix      = prompts_cfg.get("tag_context_prefix", "")
    qwen_cfg        = _s()["models"]["qwen"]

    # User-Prompt = Tag-Kontext-Prefix + Tags
    user_prompt = tag_prefix + tags

    # Bild vorbereiten
    image_tensor = _prepare_image(image)

    # Chat-Template zusammenbauen
    formatted = _build_prompt(
        prompt         = user_prompt,
        system_message = system_message,
        has_image      = True,
        thinking       = qwen_cfg.get("thinking", False),
    )

    # Qwen noch nicht geladen? → Tagger-Modelle zuerst entladen.
    # (Tagger ~5–6 GB + Qwen ~14 GB > 15,8 GB VRAM)
    # Beim zweiten und weiteren Bildern im Batch bleibt Qwen auf der GPU —
    # kein unnötiger CPU↔GPU-Transfer zwischen Captions.
    import comfy.model_management
    import torch

    if _clip_model is None:
        # Erster Aufruf im Batch oder nach Modell-Wechsel:
        # WD EVA02 (nicht durch ComfyUI verwaltet) explizit entladen
        try:
            from pipeline.tagger_wd import unload as _unload_wd
            _unload_wd()
        except Exception:
            pass
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    clip = _get_clip()
    logger.info(f"Generiere Caption (Modus: {key}) …")

    # Bild wird NICHT an den Tokenizer übergeben:
    # - Vermeidet den fp8/float32-Bias-Fehler im visuellen Encoder
    # - Tags liefern bereits alle visuellen Informationen als Kontext
    # - Funktioniert mit allen Quantisierungsformaten (fp8, mxfp8, int8)
    tokens = clip.tokenize(
        formatted,
        image        = image_tensor,
        skip_template = True,
        min_length   = 1,
        thinking     = qwen_cfg.get("thinking", False),
    )

    # ── Geräte-Fix für Qwen3.5 SSM/Mamba-Parameter ───────────────────────
    # A_log und dt_bias sind SSM-Parameter die ComfyUIs async weight offloading
    # nicht verwaltet → bleiben auf CPU während der Rest auf CUDA läuft.
    # Lösung: forward pre-hooks registrieren die GENAU BEIM Forward-Call
    # die Parameter in-place auf das korrekte Device verschieben.
    import comfy.model_management
    import torch

    device   = comfy.model_management.get_torch_device()
    _SSM_ATTRS = ("A_log", "dt_bias", "A_log_frac", "dt_low", "dt_high",
                  "D", "A", "dt_proj_weight", "dt_proj_bias")

    def _ssm_pre_hook(module, inputs):
        """Verschiebt SSM-Parameter in-place auf das Device des ersten Inputs."""
        tgt = device
        for inp in (inputs or []):
            if isinstance(inp, (torch.Tensor,)) and inp.device.type != "cpu":
                tgt = inp.device
                break
        for attr in _SSM_ATTRS:
            p = getattr(module, attr, None)
            if isinstance(p, torch.Tensor) and p.device != tgt:
                # .data assignment: in-place, preserves Parameter status
                p.data = p.data.to(tgt, non_blocking=False)

    # Hooks auf ALLEN Modulen mit SSM-Attributen registrieren
    _hooks = []
    _hooked = set()
    try:
        for root in (clip.patcher.model, getattr(clip, "cond_stage_model", None)):
            if root is None:
                continue
            for mod in root.modules():
                if id(mod) in _hooked:
                    continue
                if any(isinstance(getattr(mod, a, None), torch.Tensor)
                       for a in _SSM_ATTRS):
                    _hooks.append(mod.register_forward_pre_hook(_ssm_pre_hook))
                    _hooked.add(id(mod))
        logger.info(f"SSM device-fix hooks: {len(_hooks)} Module gefunden")
    except Exception as e:
        logger.warning(f"SSM hook registration fehlgeschlagen: {e}")

    try:
        generated_ids = clip.generate(
            tokens,
            do_sample          = True,
            max_length         = qwen_cfg.get("max_length", 1024),
            temperature        = qwen_cfg.get("temperature", 0.7),
            top_k              = qwen_cfg.get("top_k", 64),
            top_p              = qwen_cfg.get("top_p", 0.95),
            min_p              = qwen_cfg.get("min_p", 0.05),
            repetition_penalty = qwen_cfg.get("repetition_penalty", 1.05),
            presence_penalty   = qwen_cfg.get("presence_penalty", 0.0),
            seed               = qwen_cfg.get("seed", 0),
        )
    finally:
        for h in _hooks:
            try: h.remove()
            except Exception: pass

    caption = clip.decode(generated_ids, skip_special_tokens=True)
    logger.info("Caption generiert.")
    return caption.strip()
