"""
Bootstrap — muss als allererstes aufgerufen werden.

Richtet sys.path ein, registriert das JointTagger-Paket korrekt
und liest die Modellkonfiguration direkt aus der config.json —
ohne ComfyUI-Server oder ComfyLogger zu laden.
"""
import sys
import json
import types
import importlib.util
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

CONFIG_DIR   = Path(__file__).parent.parent / "config"
_initialized = False
_model_map:  dict = {}   # "RedRocket PILOT v2" → {"model": "...", "tags": "..."}
_model_map_v3: dict = {}
_model_map_dino: dict = {}


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def get_settings() -> dict:
    with open(CONFIG_DIR / "settings.json", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Modell-Name → Datei-Mapping (direkt aus config.json, ohne Server-Import)
# ---------------------------------------------------------------------------

def get_model_path(display_name: str) -> str | None:
    """Gibt den Config-Schlüssel zurück, den JtpModelManager erwartet."""
    entry = _model_map.get(display_name)
    return entry.get("config_key") if entry else None


def get_tags_path(display_name: str) -> str | None:
    """Gibt den Config-Schlüssel zurück, den JtpTagManager erwartet."""
    entry = _model_map.get(display_name)
    return entry.get("config_key") if entry else None


def _load_jt_config(rr_path: Path) -> None:
    """
    Liest die config.json des JointTagger-Pakets direkt (kein Server-Import).
    Füllt _model_map, _model_map_v3, _model_map_dino.
    Speichert den äußeren CONFIG-SCHLÜSSEL (nicht entry["model"]),
    denn JtpModelManager erwartet diesen Schlüssel in run_classifier().
    """
    global _model_map, _model_map_v3, _model_map_dino

    candidates = [
        rr_path / "config.json",
        rr_path / "helpers" / "config.json",
        rr_path / "redrocket" / "config.json",
    ]
    cfg_file = next((p for p in candidates if p.exists()), None)

    if cfg_file is None:
        logger.warning("JointTagger config.json nicht gefunden — nutze settings.json direkt.")
        return

    try:
        with open(cfg_file, encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        logger.warning(f"Konnte config.json nicht lesen: {e}")
        return

    # ── v1/v2-Modelle ────────────────────────────────────────────────────
    for config_key, val in cfg.get("models", {}).items():
        if isinstance(val, dict) and "name" in val:
            display_name = val["name"]
            _model_map[display_name] = {
                "config_key": config_key,          # Das ist was JtpModelManager braucht
                "raw_model":  val.get("model"),    # Nur zu Diagnosezwecken
                "raw_tags":   val.get("tags"),
            }
            logger.info(
                f"  v1/v2-Modell: '{display_name}' → config_key='{config_key}' "
                f"(model-Typ: {type(val.get('model')).__name__})"
            )

    # ── v3-Modelle ────────────────────────────────────────────────────────
    for config_key, val in cfg.get("models_v3", {}).items():
        if isinstance(val, dict):
            _model_map_v3[config_key] = val
            logger.info(f"  v3-Modell: '{config_key}'")

    # ── DINO-Modelle ──────────────────────────────────────────────────────
    for config_key, val in cfg.get("models_dino", {}).items():
        if isinstance(val, dict):
            _model_map_dino[config_key] = val
            logger.info(f"  DINO-Modell: '{config_key}'")

    logger.info(
        f"config.json geladen: "
        f"{len(_model_map)} v1/v2-Modelle, "
        f"{len(_model_map_v3)} v3-Modelle, "
        f"{len(_model_map_dino)} DINO-Modelle"
    )


# ---------------------------------------------------------------------------
# Namespace-Paket-Registrierung (relative Imports in redrocket/* ermöglichen)
# ---------------------------------------------------------------------------

def _register_ns(path: Path, name: str) -> None:
    if name in sys.modules:
        return
    mod             = types.ModuleType(name)
    mod.__path__    = [str(path)]
    mod.__package__ = name
    init            = path / "__init__.py"
    if init.exists():
        mod.__file__ = str(init)
        mod.__spec__ = importlib.util.spec_from_file_location(
            name, str(init),
            submodule_search_locations=[str(path)]
        )
    sys.modules[name] = mod


def _register_package_tree(root: Path, root_name: str, depth: int = 3) -> None:
    _register_ns(root, root_name)
    if depth <= 0:
        return
    try:
        for item in root.iterdir():
            if item.is_dir() and not item.name.startswith("."):
                _register_package_tree(item, f"{root_name}.{item.name}", depth - 1)
    except PermissionError:
        pass


# ---------------------------------------------------------------------------
# Model-Manager initialisieren
# ---------------------------------------------------------------------------

def _init_managers(model_basepath: str, tags_basepath: str) -> None:
    try:
        from rr_joint_tagger.redrocket.model_manager import JtpModelManager
        from rr_joint_tagger.redrocket.tag_manager   import JtpTagManager
        JtpModelManager(model_basepath=model_basepath)
        JtpTagManager(tags_basepath=tags_basepath)
        logger.info("JTP PILOT v2 Manager initialisiert.")
    except Exception as e:
        logger.warning(f"JTP PILOT v2 Manager: {e}")

    try:
        from rr_joint_tagger.redrocket.model_v3_manager import JtpModelV3Manager
        from rr_joint_tagger.redrocket.tag_v3_manager   import JtpTagV3Manager
        from rr_joint_tagger.redrocket.image_v3_manager import JtpImageV3Manager
        JtpModelV3Manager(model_basepath=model_basepath)
        JtpTagV3Manager(tags_basepath=tags_basepath)
        JtpImageV3Manager()
        logger.info("JTP-3 Hydra Manager initialisiert.")
    except Exception as e:
        logger.warning(f"JTP-3 Hydra Manager: {e}")

    try:
        from rr_joint_tagger.redrocket.model_dino_manager import DINOv3ModelManager
        from rr_joint_tagger.redrocket.tag_dino_manager   import DINOv3TagManager
        from rr_joint_tagger.redrocket.image_dino_manager import DINOv3ImageManager
        DINOv3ModelManager(model_basepath=model_basepath)
        DINOv3TagManager(tags_basepath=tags_basepath)
        DINOv3ImageManager()
        logger.info("DINOv3 Manager initialisiert.")
    except Exception as e:
        logger.warning(f"DINOv3 Manager: {e}")


# ---------------------------------------------------------------------------
# Hauptfunktion
# ---------------------------------------------------------------------------

def _find_custom_node(custom_nodes_dir: Path, fragments: list) -> Path | None:
    if not custom_nodes_dir.exists():
        return None
    for folder in custom_nodes_dir.iterdir():
        if not folder.is_dir():
            continue
        for frag in fragments:
            if frag.lower() in folder.name.lower():
                return folder
    return None


def _install_server_stub() -> None:
    """
    Der redrocket-Code nutzt ComfyLogger, der ComfyUIs kompletten Webserver
    (server.py → app.frontend_management → aiohttp …) importieren würde.
    Wir injizieren ein Stub-'server'-Modul, dessen PromptServer alle
    Attribut-Zugriffe und Aufrufe still absorbiert.
    """
    if "server" in sys.modules:
        return

    class _Absorb:
        def __getattr__(self, _):    return _Absorb()
        def __call__(self, *a, **k): return _Absorb()
        def __iter__(self):          return iter(())
        def __bool__(self):          return False

    stub = types.ModuleType("server")
    stub.PromptServer = _Absorb()
    sys.modules["server"] = stub
    logger.info("server-Stub installiert (kein ComfyUI-Webserver-Import).")


def setup() -> dict:
    global _initialized
    if _initialized:
        return get_settings()

    settings   = get_settings()
    comfy_path = Path(settings["comfyui_path"])

    if not comfy_path.exists():
        raise RuntimeError(
            f"ComfyUI-Pfad nicht gefunden: {comfy_path}\n"
            "Bitte in config/settings.json unter 'comfyui_path' korrigieren."
        )

    # server-Stub installieren BEVOR irgendein redrocket-Code läuft
    _install_server_stub()

    # ComfyUI ans Ende von sys.path – captioner/ (von main.py gesetzt) bleibt vorne
    if str(comfy_path) not in sys.path:
        sys.path.append(str(comfy_path))

    # RR JointTagger finden und als Paket registrieren
    rr_path = _find_custom_node(
        comfy_path / "custom_nodes",
        ["ComfyUI-RR-JointTagger", "JointTagger", "rr-joint"]
    )
    if rr_path is None:
        raise RuntimeError(
            f"ComfyUI-RR-JointTagger nicht gefunden in: {comfy_path / 'custom_nodes'}"
        )
    logger.info(f"RR JointTagger gefunden: {rr_path}")

    # Als korrekte Pakethierarchie registrieren (ermöglicht relative Imports)
    _register_package_tree(rr_path, "rr_joint_tagger")

    # config.json direkt lesen — KEIN Server-Import, KEIN ComfyLogger
    _load_jt_config(rr_path)

    # folder_paths initialisieren
    import folder_paths

    model_basepath = str(Path(folder_paths.models_dir) / "RedRocket")
    tags_basepath  = str(Path(folder_paths.models_dir) / "RedRocket" / "tags")

    _init_managers(model_basepath, tags_basepath)

    # ComfyExtensionConfig-Cache vorladen (für Modellnamen-Auflösung)
    try:
        from rr_joint_tagger.helpers.config import ComfyExtensionConfig
        ext = ComfyExtensionConfig()
        ext.get(property="models")          # triggert das Laden der config.json
        logger.info("ComfyExtensionConfig vorgeladen.")
    except Exception as e:
        logger.warning(f"ComfyExtensionConfig vorladen fehlgeschlagen: {e}")

    _initialized = True
    logger.info("Bootstrap abgeschlossen.")
    return settings
