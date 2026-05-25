"""
ui/i18n.py — Internationalisierung für Batcap.
Sprachen: Deutsch (de) | English (en)
"""
from __future__ import annotations

_LANG: str = "de"   # aktive Sprache

_STRINGS: dict[str, dict[str, str]] = {
    # ── Fenster & Dialoge ────────────────────────────────────────────────────
    "win_title":            {"de": "Batcap",          "en": "Batcap"},
    "dlg_prompts":          {"de": "Prompts bearbeiten",       "en": "Edit Prompts"},
    "dlg_settings":         {"de": "Einstellungen",            "en": "Settings"},

    # ── Menü ─────────────────────────────────────────────────────────────────
    "menu_edit":            {"de": "&Bearbeiten",              "en": "&Edit"},
    "act_prompts":          {"de": "Prompts bearbeiten …",     "en": "Edit Prompts …"},
    "act_settings":         {"de": "Einstellungen …",          "en": "Settings …"},

    # ── Toolbar ───────────────────────────────────────────────────────────────
    "tb_open_folder":       {"de": "📁  Ordner öffnen",        "en": "📁  Open Folder"},
    "tb_add_files":         {"de": "🖼  Dateien hinzufügen",   "en": "🖼  Add Files"},
    "tb_clear":             {"de": "🗑  Liste leeren",          "en": "🗑  Clear List"},
    "tb_select_all":        {"de": "☑  Alle wählen",           "en": "☑  Select All"},
    "tb_select_none":       {"de": "☐  Auswahl aufheben",      "en": "☐  Deselect All"},
    "count_none":           {"de": "  0 Bilder",               "en": "  0 images"},
    "count_loaded":         {"de": "  {n} Bilder geladen",     "en": "  {n} images loaded"},
    "count_selected":       {"de": "  {n} Bilder · {s} ausgewählt",
                             "en": "  {n} images · {s} selected"},

    # ── Vorschau & Textfelder ─────────────────────────────────────────────────
    "preview_hint":         {"de": "← Bild in der Sidebar auswählen",
                             "en": "← Select an image in the sidebar"},
    "lbl_tags":             {"de": "Tags",                     "en": "Tags"},
    "lbl_caption":          {"de": "Caption",                  "en": "Caption"},
    "ph_tags":              {"de": "Tags erscheinen hier nach der Verarbeitung …",
                             "en": "Tags appear here after processing …"},
    "ph_caption":           {"de": "Caption erscheint hier nach der Verarbeitung …",
                             "en": "Caption appears here after processing …"},
    "btn_save":             {"de": "💾  Speichern",             "en": "💾  Save"},

    # ── Controls ──────────────────────────────────────────────────────────────
    "lbl_llm":              {"de": "LLM:",                     "en": "LLM:"},
    "llm_tooltip":          {"de": "Alle .safetensors aus ComfyUI/models/text_encoders/\n"
                                   "Unterstützte Formate: fp16, bf16, fp8, mxfp8, int8\n"
                                   "Leer = Modell aus settings.json",
                             "en": "All .safetensors from ComfyUI/models/text_encoders/\n"
                                   "Supported formats: fp16, bf16, fp8, mxfp8, int8\n"
                                   "Empty = model from settings.json"},
    "llm_refresh_tip":      {"de": "Modell-Liste neu scannen",  "en": "Rescan model list"},
    "grp_mode":             {"de": "Modus",                    "en": "Mode"},
    "lbl_output":           {"de": "Ausgabe-Ordner:",          "en": "Output folder:"},
    "ph_output":            {"de": "Leer = neben den Bildern speichern",
                             "en": "Empty = save next to images"},
    "btn_tag":              {"de": "🏷  Tag it!",               "en": "🏷  Tag it!"},
    "btn_cap":              {"de": "✍  Cap it!",               "en": "✍  Cap it!"},
    "btn_full":             {"de": "⚡  Tag && Cap it!",        "en": "⚡  Tag && Cap it!"},
    "btn_stop":             {"de": "⏹",                        "en": "⏹"},
    "tip_tag":              {"de": "Alle Bilder taggen (kein LLM)",
                             "en": "Tag all images (no LLM)"},
    "tip_cap":              {"de": "Captions für bereits getaggte Bilder",
                             "en": "Caption already-tagged images"},
    "tip_full":             {"de": "Taggen und Captionieren in einem Durchlauf",
                             "en": "Tag and caption in one pass"},

    # ── Status ────────────────────────────────────────────────────────────────
    "status_ready":         {"de": "Bereit.",                  "en": "Ready."},
    "status_loading":       {"de": "{n} Bilder geladen — Thumbnails werden generiert …",
                             "en": "{n} images loaded — generating thumbnails …"},
    "status_no_new":        {"de": "Keine neuen Bilder gefunden.",
                             "en": "No new images found."},
    "status_progress":      {"de": "⏳  {done}/{total} — {name}",
                             "en": "⏳  {done}/{total} — {name}"},
    "status_done":          {"de": "✅  Fertig — {total} verarbeitet, {skip} übersprungen, {err} Fehler",
                             "en": "✅  Done — {total} processed, {skip} skipped, {err} errors"},
    "status_stopped":       {"de": "⏹ Stopp angefordert …",   "en": "⏹ Stop requested …"},
    "status_saved":         {"de": "💾 Gespeichert: {name}",   "en": "💾 Saved: {name}"},
    "status_save_err":      {"de": "❌ Speichern fehlgeschlagen: {err}",
                             "en": "❌ Save failed: {err}"},
    "status_model":         {"de": "Modell gewechselt: {m} — wird beim nächsten Verarbeiten geladen",
                             "en": "Model changed: {m} — will be loaded on next run"},
    "status_prompts":       {"de": "Prompts gespeichert.",     "en": "Prompts saved."},
    "status_settings":      {"de": "Einstellungen gespeichert.", "en": "Settings saved."},

    # ── Dialoge ───────────────────────────────────────────────────────────────
    "warn_no_sel_title":    {"de": "Keine Auswahl",            "en": "No Selection"},
    "warn_no_sel_text":     {"de": "Bitte wähle mindestens ein Bild in der Sidebar aus.",
                             "en": "Please select at least one image in the sidebar."},

    # ── Prompt-Editor ─────────────────────────────────────────────────────────
    "lbl_prompt":           {"de": "Prompt:",                  "en": "Prompt:"},
    "lbl_tag_prefix":       {"de": "Tag-Kontext-Prefix (wird vor den Tags eingefügt):",
                             "en": "Tag context prefix (prepended to tags):"},
    "lbl_sysprompt":        {"de": "System-Prompt:",           "en": "System Prompt:"},

    # ── Einstellungen ─────────────────────────────────────────────────────────
    "lbl_comfy_path":       {"de": "ComfyUI-Pfad:",            "en": "ComfyUI path:"},
    "lbl_out_path":         {"de": "Standard-Ausgabe:",        "en": "Default output:"},
    "ph_out_path":          {"de": "leer = neben den Bildern", "en": "empty = next to images"},
    "lbl_temperature":      {"de": "Qwen Temperature:",        "en": "Qwen Temperature:"},
    "lbl_max_tokens":       {"de": "Qwen Max. Tokens:",        "en": "Qwen Max. Tokens:"},
    "lbl_quant":            {"de": "Quant-Format:",            "en": "Quant Format:"},
    "tip_quant":            {"de": "Nur ändern wenn auto nicht korrekt erkannt wird",
                             "en": "Only change if auto-detection is incorrect"},
    "lbl_backend":          {"de": "Kernel-Backend:",          "en": "Kernel Backend:"},
    "lbl_thr_jtp2":         {"de": "JTP PILOT v2 Threshold:",  "en": "JTP PILOT v2 Threshold:"},
    "lbl_thr_jtp3":         {"de": "JTP-3 Hydra Threshold:",   "en": "JTP-3 Hydra Threshold:"},
    "lbl_thr_dino":         {"de": "DINOv3 Threshold:",        "en": "DINOv3 Threshold:"},
    "hint_restart":         {"de": "Pfadänderungen werden erst nach App-Neustart wirksam.",
                             "en": "Path changes take effect after restarting the app."},

    # ── Datei-Dialog-Filter ───────────────────────────────────────────────────
    "fdlg_folder":          {"de": "Ordner auswählen",         "en": "Select Folder"},
    "fdlg_files":           {"de": "Bilder auswählen",         "en": "Select Images"},
    "fdlg_output":          {"de": "Ausgabe-Ordner wählen",    "en": "Select Output Folder"},
    "fdlg_filter":          {"de": "Bilder (*.jpg *.jpeg *.png *.webp *.bmp *.tiff *.tif);;Alle Dateien (*)",
                             "en": "Images (*.jpg *.jpeg *.png *.webp *.bmp *.tiff *.tif);;All Files (*)"},
}


def set_lang(lang: str) -> None:
    global _LANG
    _LANG = lang if lang in ("de", "en") else "de"


def get_lang() -> str:
    return _LANG


def tr(key: str, **kwargs) -> str:
    """Gibt den übersetzten String für den aktuellen Sprachkontext zurück."""
    entry = _STRINGS.get(key)
    if entry is None:
        return key   # Fallback: Key selbst zurückgeben
    s = entry.get(_LANG) or entry.get("de") or key
    return s.format(**kwargs) if kwargs else s
