"""
ui/app.py — PyQt6 Desktop-App mit Lightroom-ähnlichem Layout.

Wichtig: Das Stylesheet ist GEZIELT (nur objectName-Selektoren), damit es
nicht in native Dateidialoge "leakt" und deren Text unsichtbar macht.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from PyQt6.QtCore    import Qt, QSize, QThread, pyqtSignal
from PyQt6.QtGui     import QPixmap, QIcon, QColor, QImage, QAction
from PyQt6.QtWidgets import (
    QAbstractItemView, QApplication, QButtonGroup, QCheckBox, QComboBox,
    QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QProgressBar, QPushButton, QRadioButton,
    QSizePolicy, QSpinBox, QSplitter, QTextEdit, QVBoxLayout, QWidget,
)

from ui.i18n import tr, set_lang, get_lang

logger     = logging.getLogger(__name__)
CONFIG_DIR = Path(__file__).parent.parent / "config"


# ─────────────────────────────────────────────────────────────────────────────
# Style Sheet — NUR gezielte objectName-Selektoren (leakt nicht in Dialoge)
# ─────────────────────────────────────────────────────────────────────────────

STYLE = """
/* Toolbar — dunkel */
QToolBar#maintb {
    background: #111827; border: none; padding: 4px 8px; spacing: 2px;
}
QToolBar#maintb QToolButton {
    color: #f9fafb; background: transparent; border: none;
    padding: 6px 12px; border-radius: 5px; font-size: 13px;
}
QToolBar#maintb QToolButton:hover   { background: rgba(255,255,255,0.12); }
QToolBar#maintb QToolButton:pressed { background: rgba(255,255,255,0.22); }
QToolBar#maintb QLabel { color: #9ca3af; font-size: 12px; padding-left: 8px; }
QToolBar#maintb::separator { background: #374151; width: 1px; margin: 5px 6px; }

/* Sidebar — dunkel */
QListWidget#sidebar {
    background: #1f2937; border: none; color: #e5e7eb;
    font-size: 12px; outline: none;
}
QListWidget#sidebar::item {
    padding: 4px 6px; border-radius: 4px; margin: 1px 4px;
}
QListWidget#sidebar::item:selected         { background: #2563eb; color: white; }
QListWidget#sidebar::item:hover:!selected  { background: #374151; }

/* Aktionsbuttons */
QPushButton#process_btn {
    background: #2563eb; color: white; border: none;
    border-radius: 6px; font-weight: bold; font-size: 14px;
}
QPushButton#process_btn:hover    { background: #1d4ed8; }
QPushButton#process_btn:disabled { background: #93c5fd; }
QPushButton#stop_btn {
    background: #ef4444; color: white; border: none;
    border-radius: 6px; font-size: 13px;
}
QPushButton#stop_btn:disabled { background: #fca5a5; }
QPushButton#tag_btn {
    background: #059669; color: white; border: none;
    border-radius: 6px; font-weight: bold; font-size: 13px;
}
QPushButton#tag_btn:hover    { background: #047857; }
QPushButton#tag_btn:disabled { background: #6ee7b7; }
QPushButton#cap_btn {
    background: #7c3aed; color: white; border: none;
    border-radius: 6px; font-weight: bold; font-size: 13px;
}
QPushButton#cap_btn:hover    { background: #6d28d9; }
QPushButton#cap_btn:disabled { background: #c4b5fd; }
QPushButton#full_btn {
    background: #d97706; color: white; border: none;
    border-radius: 6px; font-weight: bold; font-size: 13px;
}
QPushButton#full_btn:hover    { background: #b45309; }
QPushButton#full_btn:disabled { background: #fcd34d; }
QPushButton#lang_btn {
    background: #374151; color: #e5e7eb; border: none;
    border-radius: 5px; font-size: 12px; font-weight: bold;
}
QPushButton#lang_btn:hover { background: #4b5563; }

QGroupBox#tagger_toggle_box {
    font-size: 11px; font-weight: bold;
    border: 1px solid #374151; border-radius: 6px;
    margin-top: 6px; padding: 6px 4px 4px 4px;
}
QGroupBox#tagger_toggle_box::title {
    subcontrol-origin: margin; left: 8px; padding: 0 3px;
    color: #9ca3af;
}
QLabel#tagger_lbl {
    font-size: 10px; color: #d1d5db;
}
QPushButton#tagger_toggle {
    border: none; border-radius: 5px;
    font-size: 11px; font-weight: bold; min-width: 46px;
}
QPushButton#tagger_toggle:checked {
    background: #22c55e; color: white;
}
QPushButton#tagger_toggle:!checked {
    background: #374151; color: #6b7280;
}
QPushButton#tagger_toggle:checked:hover { background: #16a34a; }
QPushButton#tagger_toggle:!checked:hover { background: #4b5563; }

/* Vorschau-Platzhalter */
QLabel#preview {
    color: #9ca3af; background: #f3f4f6; border-radius: 6px;
}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Settings / Prompts laden & speichern
# ─────────────────────────────────────────────────────────────────────────────

def load_settings() -> dict:
    with open(CONFIG_DIR / "settings.json", encoding="utf-8") as f:
        return json.load(f)

def save_settings(s: dict):
    with open(CONFIG_DIR / "settings.json", "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2, ensure_ascii=False)

def load_prompts() -> dict:
    with open(CONFIG_DIR / "prompts.json", encoding="utf-8") as f:
        return json.load(f)

def save_prompts(p: dict):
    with open(CONFIG_DIR / "prompts.json", "w", encoding="utf-8") as f:
        json.dump(p, f, indent=2, ensure_ascii=False)
    try:
        from pipeline.captioner import reload_prompts
        reload_prompts()
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Background Worker: Thumbnails
# ─────────────────────────────────────────────────────────────────────────────

class ThumbnailWorker(QThread):
    thumbnail_ready = pyqtSignal(int, QPixmap)

    def __init__(self, paths: list[str], start_idx: int):
        super().__init__()
        self.paths     = paths
        self.start_idx = start_idx

    def run(self):
        SIZE = 64
        for offset, path in enumerate(self.paths):
            if self.isInterruptionRequested():
                return
            try:
                img = QImage(path)
                if img.isNull():
                    continue
                pix = QPixmap.fromImage(img).scaled(
                    SIZE, SIZE,
                    Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                    Qt.TransformationMode.SmoothTransformation,
                )
                x = (pix.width()  - SIZE) // 2
                y = (pix.height() - SIZE) // 2
                self.thumbnail_ready.emit(self.start_idx + offset,
                                          pix.copy(x, y, SIZE, SIZE))
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Background Worker: Verarbeitung
# ─────────────────────────────────────────────────────────────────────────────

class ProcessingWorker(QThread):
    image_done       = pyqtSignal(int, str, str, str)   # idx, tags, caption, status
    progress_updated = pyqtSignal(int, int, str)        # done, total, name
    all_done         = pyqtSignal(int, int, int)        # total, skipped, errors

    def __init__(self, all_paths, indices, output_dir, prompt_key, mode="both"):
        super().__init__()
        self.all_paths  = all_paths
        self.indices    = indices
        self.output_dir = output_dir
        self.prompt_key = prompt_key
        self.mode       = mode  # "tag" | "cap" | "both"

    def stop(self):
        from cap_app.batch import state
        state.request_stop()

    def run(self):
        from cap_app.batch import run_tag_batch, run_cap_batch, run_batch as _full
        selected = [self.all_paths[i] for i in self.indices if i < len(self.all_paths)]
        errors = skipped = done = 0

        _fn = {"tag": run_tag_batch, "cap": run_cap_batch}.get(self.mode, _full)
        for result in _fn(
            source     = selected,
            output_dir = self.output_dir,
            prompt_key = self.prompt_key,
        ):
            done += 1
            try:
                gidx = self.all_paths.index(str(result.image_path))
            except ValueError:
                gidx = -1

            if result.skipped:
                status = "skipped"; skipped += 1
            elif result.success:
                status = "done"
            else:
                status = "error";   errors += 1

            if gidx >= 0:
                self.image_done.emit(gidx, result.tags, result.caption, status)
            self.progress_updated.emit(done, len(selected), result.image_path.name)

        self.all_done.emit(done, skipped, errors)


# ─────────────────────────────────────────────────────────────────────────────
# Dialog: Prompt-Editor
# ─────────────────────────────────────────────────────────────────────────────

class PromptEditorDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("dlg_prompts"))
        self.resize(640, 560)
        self._data = load_prompts()

        layout = QVBoxLayout(self)

        # Prompt-Auswahl
        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel(tr("lbl_prompt")))
        self._combo = QComboBox()
        for key, val in self._data["prompts"].items():
            self._combo.addItem(val.get("name", key), key)
        self._combo.currentIndexChanged.connect(self._on_switch)
        sel_row.addWidget(self._combo, 1)
        layout.addLayout(sel_row)

        # Tag-Kontext-Prefix
        layout.addWidget(QLabel(tr("lbl_tag_prefix")))
        self._prefix = QTextEdit()
        self._prefix.setMaximumHeight(110)
        self._prefix.setPlainText(self._data.get("tag_context_prefix", ""))
        layout.addWidget(self._prefix)

        # System-Prompt
        layout.addWidget(QLabel(tr("lbl_sysprompt")))
        self._sysmsg = QTextEdit()
        layout.addWidget(self._sysmsg, 1)

        # Buttons
        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save |
            QDialogButtonBox.StandardButton.Cancel
        )
        bb.accepted.connect(self._save)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)

        self._current_key = self._combo.currentData()
        self._load_current()

    def _load_current(self):
        key = self._combo.currentData()
        self._sysmsg.setPlainText(
            self._data["prompts"].get(key, {}).get("system_message", "")
        )

    def _on_switch(self):
        # Aktuellen Text zwischenspeichern bevor gewechselt wird
        self._data["prompts"][self._current_key]["system_message"] = \
            self._sysmsg.toPlainText()
        self._current_key = self._combo.currentData()
        self._load_current()

    def _save(self):
        self._data["prompts"][self._current_key]["system_message"] = \
            self._sysmsg.toPlainText()
        self._data["tag_context_prefix"] = self._prefix.toPlainText()
        save_prompts(self._data)
        self.accept()


# ─────────────────────────────────────────────────────────────────────────────
# Dialog: Einstellungen
# ─────────────────────────────────────────────────────────────────────────────

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("dlg_settings"))
        self.resize(520, 420)
        self._s = load_settings()

        layout = QVBoxLayout(self)
        form   = QFormLayout()

        # Pfade
        self._comfy = QLineEdit(self._s.get("comfyui_path", ""))
        form.addRow(tr("lbl_comfy_path"), self._comfy)

        self._out = QLineEdit(self._s.get("output", {}).get("path", ""))
        self._out.setPlaceholderText(tr("ph_out_path"))
        form.addRow(tr("lbl_out_path"), self._out)

        # Qwen
        q = self._s["models"]["qwen"]
        self._temp = QDoubleSpinBox()
        self._temp.setRange(0.0, 2.0); self._temp.setSingleStep(0.05)
        self._temp.setValue(q.get("temperature", 0.7))
        form.addRow(tr("lbl_temperature"), self._temp)

        self._maxlen = QSpinBox()
        self._maxlen.setRange(64, 8192); self._maxlen.setSingleStep(64)
        self._maxlen.setValue(q.get("max_length", 1024))
        form.addRow(tr("lbl_max_tokens"), self._maxlen)

        self._qformat = QComboBox()
        self._qformat.addItems([
            "auto", "int8", "int8_tensorwise",
            "float8_e4m3fn", "float8_e4m3fn_blockwise", "float8_e4m3fn_rowwise",
            "mxfp8", "hybrid_mxfp8", "nvfp4",
        ])
        idx = self._qformat.findText(q.get("quant_format", "auto"))
        self._qformat.setCurrentIndex(max(0, idx))
        self._qformat.setToolTip("auto = ComfyUI erkennt das Format selbst\n" + tr("tip_quant"))
        form.addRow(tr("lbl_quant"), self._qformat)

        self._kbackend = QComboBox()
        self._kbackend.addItems(["pytorch", "triton"])
        idx = self._kbackend.findText(q.get("kernel_backend", "pytorch"))
        self._kbackend.setCurrentIndex(max(0, idx))
        form.addRow(tr("lbl_backend"), self._kbackend)

        # Tagger-Schwellenwerte
        self._t1 = QDoubleSpinBox()
        self._t1.setRange(0.0, 1.0); self._t1.setSingleStep(0.05)
        self._t1.setValue(self._s["models"]["jtp_pilot2"].get("threshold", 0.25))
        form.addRow(tr("lbl_thr_jtp2"), self._t1)

        self._t2 = QDoubleSpinBox()
        self._t2.setRange(0.0, 1.0); self._t2.setSingleStep(0.05)
        self._t2.setValue(self._s["models"]["jtp3_hydra"].get("threshold", 0.35))
        form.addRow(tr("lbl_thr_jtp3"), self._t2)

        self._t3 = QDoubleSpinBox()
        self._t3.setRange(0.0, 1.0); self._t3.setSingleStep(0.05)
        self._t3.setValue(self._s["models"]["dinov3"].get("threshold", 0.85))
        form.addRow(tr("lbl_thr_dino"), self._t3)

        layout.addLayout(form)

        hint = QLabel(tr("hint_restart"))
        hint.setStyleSheet("color: #6b7280; font-size: 11px;")
        layout.addWidget(hint)

        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save |
            QDialogButtonBox.StandardButton.Cancel
        )
        bb.accepted.connect(self._save)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)

    def _save(self):
        self._s["comfyui_path"]                    = self._comfy.text().strip()
        self._s["output"]["path"]                  = self._out.text().strip()
        self._s["models"]["qwen"]["temperature"]   = self._temp.value()
        self._s["models"]["qwen"]["max_length"]    = self._maxlen.value()
        self._s["models"]["qwen"]["quant_format"]  = self._qformat.currentText()
        self._s["models"]["qwen"]["kernel_backend"]= self._kbackend.currentText()
        self._s["models"]["jtp_pilot2"]["threshold"] = self._t1.value()
        self._s["models"]["jtp3_hydra"]["threshold"] = self._t2.value()
        self._s["models"]["dinov3"]["threshold"]     = self._t3.value()
        save_settings(self._s)
        self.accept()


# ─────────────────────────────────────────────────────────────────────────────
# Haupt-Fenster
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.all_paths:     list[str]        = []
        self._thumb_worker: ThumbnailWorker  | None = None
        self._proc_worker:  ProcessingWorker | None = None
        self._build_ui()
        self.setStyleSheet(STYLE)

    # ── UI-Aufbau ────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.setWindowTitle(tr("win_title"))
        self.setMinimumSize(1050, 680)
        self.resize(1280, 800)

        self._build_menubar()
        self._build_toolbar()

        main_split = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(main_split)

        main_split.addWidget(self._build_sidebar())

        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(self._build_preview())
        right.addWidget(self._build_controls())
        right.setSizes([520, 230])
        main_split.addWidget(right)

        main_split.setSizes([260, 1020])
        main_split.setHandleWidth(4)

        self._build_statusbar()
        self._lang_btn.setText("🇩🇪 DE")   # Standardsprache

    def _build_menubar(self):
        mb = self.menuBar()
        # Menüleiste IM Fenster anzeigen (nicht in KDE/GNOME globaler Leiste)
        mb.setNativeMenuBar(False)
        self._edit_menu = mb.addMenu(tr("menu_edit"))
        edit_menu = self._edit_menu

        self._act_prompts = QAction(tr("act_prompts"), self)
        self._act_prompts.triggered.connect(self._open_prompts)
        edit_menu.addAction(self._act_prompts)

        self._act_settings = QAction(tr("act_settings"), self)
        self._act_settings.triggered.connect(self._open_settings)
        edit_menu.addAction(self._act_settings)

    def _build_toolbar(self):
        tb = self.addToolBar("Werkzeuge")
        tb.setObjectName("maintb")
        tb.setMovable(False)
        tb.setIconSize(QSize(18, 18))
        self._tb_actions = []

        def _act(label, slot):
            a = QAction(label, self)
            a.triggered.connect(slot)
            tb.addAction(a)
            self._tb_actions.append(a)
            return a

        _act(tr("tb_open_folder"), self.open_folder)
        _act(tr("tb_add_files"),   self.open_files)
        _act(tr("tb_clear"),       self.clear_list)
        tb.addSeparator()
        _act(tr("tb_select_all"),  self._select_all)
        _act(tr("tb_select_none"), self._select_none)

        self._count_label = QLabel(tr("count_none"))
        tb.addWidget(self._count_label)

        # Sprach-Toggle rechts
        spacer = QWidget(); spacer.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        tb.addWidget(spacer)
        self._lang_btn = QPushButton("DE | EN")
        self._lang_btn.setObjectName("lang_btn")
        self._lang_btn.setFixedWidth(72)
        self._lang_btn.setToolTip("Sprache wechseln / Switch language")
        self._lang_btn.clicked.connect(self._toggle_lang)
        tb.addWidget(self._lang_btn)

    def _select_all(self):
        if hasattr(self, "_list"):
            self._list.selectAll()

    def _select_none(self):
        if hasattr(self, "_list"):
            self._list.clearSelection()

    def _build_sidebar(self) -> QWidget:
        self._list = QListWidget()
        self._list.setObjectName("sidebar")
        self._list.setIconSize(QSize(64, 64))
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._list.setSpacing(2)
        self._list.setUniformItemSizes(True)
        self._list.itemSelectionChanged.connect(self._on_selection_changed)
        self._list.keyPressEvent = self._list_key_press
        return self._list

    def _build_preview(self) -> QWidget:
        w      = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(8, 8, 8, 4)

        self._preview = QLabel(tr("preview_hint"))
        self._preview.setObjectName("preview")
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self._preview, stretch=3)

        text_split = QSplitter(Qt.Orientation.Horizontal)

        tags_box = QGroupBox(tr("lbl_tags"))
        self._tags_box = tags_box
        tbl      = QVBoxLayout(tags_box)
        self._tags = QTextEdit()
        self._tags.setPlaceholderText(tr("ph_tags"))
        tbl.addWidget(self._tags)
        text_split.addWidget(tags_box)

        cap_box = QGroupBox(tr("lbl_caption"))
        self._cap_box = cap_box
        cbl     = QVBoxLayout(cap_box)
        self._cap = QTextEdit()
        self._cap.setPlaceholderText(tr("ph_caption"))
        cbl.addWidget(self._cap)
        text_split.addWidget(cap_box)

        save_row = QHBoxLayout()
        self._save_btn = QPushButton(tr("btn_save"))
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._save_current)
        save_row.addStretch()
        save_row.addWidget(self._save_btn)
        layout.addLayout(save_row)

        layout.addWidget(text_split, stretch=2)
        return w

    def _build_controls(self) -> QWidget:
        w      = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(8, 4, 8, 8)
        layout.setSpacing(6)

        # ── LLM-Modell-Auswahl ───────────────────────────────────────────
        llm_row = QHBoxLayout()
        self._llm_lbl = QLabel(tr("lbl_llm"))
        llm_row.addWidget(self._llm_lbl)
        self._llm_combo = QComboBox()
        self._llm_combo.setToolTip(
            "Alle .safetensors aus ComfyUI/models/text_encoders/\n"
            "Unterstützte Formate: fp16, bf16, fp8, mxfp8, int8\n"
            "(ComfyUI erkennt das Quantisierungsformat automatisch)"
        )
        self._llm_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        llm_row.addWidget(self._llm_combo, 1)
        llm_refresh = QPushButton("🔄")
        llm_refresh.setFixedWidth(36)
        llm_refresh.setToolTip(tr("llm_refresh_tip"))
        llm_refresh.clicked.connect(self._refresh_llm_models)
        llm_row.addWidget(llm_refresh)
        layout.addLayout(llm_row)

        # Scan nach dem Aufbau (verzögert, damit Bootstrap sicher fertig ist)
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(200, self._refresh_llm_models)
        self._llm_combo.currentTextChanged.connect(self._on_llm_changed)

        # ── Modus + Überschreiben ────────────────────────────────────────
        row1 = QHBoxLayout()
        mode_box    = QGroupBox(tr("grp_mode"))
        self._mode_box = mode_box
        mode_layout = QHBoxLayout(mode_box)
        self._radio_group   = QButtonGroup(self)
        self._prompt_radios = {}

        prompts = load_prompts()
        active  = prompts.get("active_prompt", "photorealism")
        for key, data in prompts["prompts"].items():
            rb = QRadioButton(data.get("name", key))
            self._radio_group.addButton(rb)
            mode_layout.addWidget(rb)
            self._prompt_radios[key] = rb
            if key == active:
                rb.setChecked(True)
        row1.addWidget(mode_box)

        row1.addStretch()
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        self._lbl_output = QLabel(tr("lbl_output"))
        row2.addWidget(self._lbl_output)
        self._out_edit = QLineEdit()
        self._out_edit.setPlaceholderText(tr("ph_output"))
        self._out_edit.setText(load_settings()["output"].get("path", ""))
        row2.addWidget(self._out_edit)
        browse = QPushButton("📁")
        browse.setFixedWidth(40)
        browse.clicked.connect(self.browse_output)
        row2.addWidget(browse)
        layout.addLayout(row2)

        # ── Tagger ON/OFF-Switches ───────────────────────────────────────────
        toggle_box = QGroupBox("Tagger")
        toggle_box.setObjectName("tagger_toggle_box")
        toggle_lay = QHBoxLayout(toggle_box)
        toggle_lay.setSpacing(8)
        self._tagger_toggles: dict[str, QPushButton] = {}
        _TAGGERS = [
            ("jtp_pilot2", "JTP v2"),
            ("jtp3_hydra",  "JTP-3"),
            ("dinov3",      "DINOv3"),
            ("wd_eva02",    "WD EVA02"),
        ]
        for key, label in _TAGGERS:
            col = QVBoxLayout()
            col.setSpacing(2)
            lbl = QLabel(label)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setObjectName("tagger_lbl")
            btn = QPushButton("ON")
            btn.setObjectName("tagger_toggle")
            btn.setCheckable(True)
            btn.setFixedHeight(26)
            # Initialer Zustand aus settings.json
            try:
                import json
                _cfg_path = Path(__file__).parent.parent / "config" / "settings.json"
                _cfg = json.loads(_cfg_path.read_text())
                _on  = _cfg["models"].get(key, {}).get("enabled", True)
            except Exception:
                _on = True
            btn.setChecked(_on)
            btn.setText("ON" if _on else "OFF")
            btn.clicked.connect(lambda checked, k=key, b=btn: self._on_tagger_toggle(k, b))
            self._tagger_toggles[key] = btn
            col.addWidget(lbl)
            col.addWidget(btn)
            toggle_lay.addLayout(col)
        layout.addWidget(toggle_box)

        row3 = QHBoxLayout()
        self._tag_btn = QPushButton(tr("btn_tag"))
        self._tag_btn.setObjectName("tag_btn")
        self._tag_btn.setMinimumHeight(40)
        self._tag_btn.setToolTip(tr("tip_tag"))
        self._tag_btn.clicked.connect(lambda: self.start_processing("tag"))

        self._cap_btn = QPushButton(tr("btn_cap"))
        self._cap_btn.setObjectName("cap_btn")
        self._cap_btn.setMinimumHeight(40)
        self._cap_btn.setToolTip(tr("tip_cap"))
        self._cap_btn.clicked.connect(lambda: self.start_processing("cap"))

        self._full_btn = QPushButton(tr("btn_full"))
        self._full_btn.setObjectName("full_btn")
        self._full_btn.setMinimumHeight(40)
        self._full_btn.setToolTip(tr("tip_full"))
        self._full_btn.clicked.connect(lambda: self.start_processing("both"))

        self._stop_btn = QPushButton("⏹")
        self._stop_btn.setObjectName("stop_btn")
        self._stop_btn.setMinimumHeight(40)
        self._stop_btn.setFixedWidth(50)
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self.stop_processing)

        row3.addWidget(self._tag_btn,  stretch=2)
        row3.addWidget(self._cap_btn,  stretch=2)
        row3.addWidget(self._full_btn, stretch=3)
        row3.addWidget(self._stop_btn, stretch=0)
        layout.addLayout(row3)
        return w

    def _build_statusbar(self):
        self._status = QLabel(tr("status_ready"))
        self.statusBar().addWidget(self._status, 1)
        self._progress = QProgressBar()
        self._progress.setFixedWidth(220)
        self._progress.hide()
        self.statusBar().addPermanentWidget(self._progress)

    # ── LLM-Modell ───────────────────────────────────────────────────────────

    def _refresh_llm_models(self):
        """Scannt text_encoders-Verzeichnis und füllt das Dropdown."""
        from pipeline.captioner import list_available_models
        models = list_available_models()

        current = load_settings()["models"]["qwen"].get("model_file", "")

        self._llm_combo.blockSignals(True)
        self._llm_combo.clear()
        self._llm_combo.addItems(models)

        # Aktuell konfiguriertes Modell vorauswählen
        idx = self._llm_combo.findText(current)
        if idx >= 0:
            self._llm_combo.setCurrentIndex(idx)
        elif models and models[0] != "(keine Modelle gefunden)":
            self._llm_combo.setCurrentIndex(0)

        self._llm_combo.blockSignals(False)
        self._status.setText(
            f"{'⚠ Kein Modell gefunden' if not models or models[0].startswith('(') else f'{len(models)} Modell(e) verfügbar'}"
        )

    def _on_llm_changed(self, model_file: str):
        """Speichert neues Modell und entlädt das aktuelle aus dem VRAM."""
        if not model_file or model_file.startswith("("):
            return
        s = load_settings()
        if s["models"]["qwen"].get("model_file") == model_file:
            return  # keine Änderung
        s["models"]["qwen"]["model_file"] = model_file
        save_settings(s)

        from pipeline.captioner import unload_model, reload_settings
        unload_model()
        reload_settings()
        self._status.setText(tr("status_model", m=model_file))

    # ── Menü-Aktionen ────────────────────────────────────────────────────────

    def _open_prompts(self):
        dlg = PromptEditorDialog(self)
        if dlg.exec():
            self._refresh_mode_radios()
            self._status.setText(tr("status_prompts"))

    def _open_settings(self):
        dlg = SettingsDialog(self)
        if dlg.exec():
            self._out_edit.setText(load_settings()["output"].get("path", ""))
            self._status.setText(tr("status_settings"))

    def _refresh_mode_radios(self):
        prompts = load_prompts()
        for key, rb in self._prompt_radios.items():
            if key in prompts["prompts"]:
                rb.setText(prompts["prompts"][key].get("name", key))

    # ── Datei-Operationen ────────────────────────────────────────────────────

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, tr("fdlg_folder"), str(Path.home()))
        if folder:
            self._load_from_sources([folder])

    def open_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, tr("fdlg_files"), str(Path.home()), tr("fdlg_filter"))
        if files:
            self._load_from_sources(files)

    def clear_list(self):
        self._list.clear()
        self.all_paths.clear()
        self._preview.clear()
        self._preview.setText(tr("preview_hint"))
        self._tags.clear()
        self._cap.clear()
        self._update_count()

    def browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, tr("fdlg_output"), str(Path.home()))
        if folder:
            self._out_edit.setText(folder)

    def _load_from_sources(self, sources: list[str]):
        from cap_app.file_utils import collect_images

        new_paths = []
        for src in sources:
            for p in collect_images(src):
                sp = str(p)
                if sp not in self.all_paths:
                    new_paths.append(sp)

        if not new_paths:
            self._status.setText(tr("status_no_new"))
            return

        start_idx = len(self.all_paths)
        self.all_paths.extend(new_paths)

        placeholder = self._placeholder_pixmap()
        for path in new_paths:
            item = QListWidgetItem(Path(path).name)
            item.setData(Qt.ItemDataRole.UserRole, path)
            item.setSizeHint(QSize(240, 74))
            item.setIcon(QIcon(placeholder))
            self._list.addItem(item)

        self._update_count()
        self._status.setText(tr("status_loading", n=len(new_paths)))

        if self._thumb_worker and self._thumb_worker.isRunning():
            self._thumb_worker.requestInterruption()
            self._thumb_worker.wait(500)
        self._thumb_worker = ThumbnailWorker(new_paths, start_idx)
        self._thumb_worker.thumbnail_ready.connect(self._on_thumb_ready)
        self._thumb_worker.finished.connect(
            lambda: self._status.setText(tr("status_ready"))
        )
        self._thumb_worker.start()

    # ── Thumbnails ───────────────────────────────────────────────────────────

    def _placeholder_pixmap(self) -> QPixmap:
        pix = QPixmap(64, 64)
        pix.fill(QColor("#374151"))
        return pix

    def _on_thumb_ready(self, idx: int, pixmap: QPixmap):
        item = self._list.item(idx)
        if item:
            item.setIcon(QIcon(pixmap))

    # ── Auswahl + Vorschau ───────────────────────────────────────────────────

    def _on_selection_changed(self):
        self._update_count()
        items = self._list.selectedItems()
        if not items:
            self._preview.clear()
            self._preview.setText(tr("preview_hint"))
            self._tags.clear()
            self._cap.clear()
            return
        self._show_preview(items[-1].data(Qt.ItemDataRole.UserRole))

    def _show_preview(self, path: str):
        pix = QPixmap(path)
        if not pix.isNull():
            self._preview.setPixmap(pix.scaled(
                max(self._preview.width(),  200),
                max(self._preview.height(), 200),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ))

        from cap_app.batch import load_existing
        tags, caption = load_existing(
            Path(path), self._out_edit.text().strip() or None
        )
        self._tags.setPlainText(tags)
        self._cap.setPlainText(caption)
        self._save_btn.setEnabled(bool(tags or caption))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        items = self._list.selectedItems() if hasattr(self, "_list") else []
        if items:
            pix = QPixmap(items[-1].data(Qt.ItemDataRole.UserRole))
            if not pix.isNull():
                self._preview.setPixmap(pix.scaled(
                    max(self._preview.width(),  200),
                    max(self._preview.height(), 200),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                ))

    # ── Verarbeitung ─────────────────────────────────────────────────────────

    def _active_prompt_key(self) -> str:
        for key, rb in self._prompt_radios.items():
            if rb.isChecked():
                return key
        return "photorealism"

    def start_processing(self, mode: str = "both"):
        selected = self._list.selectedItems()
        if not selected:
            QMessageBox.warning(self, tr("warn_no_sel_title"), tr("warn_no_sel_text"))
            return

        indices = []
        for item in selected:
            path = item.data(Qt.ItemDataRole.UserRole)
            try:
                indices.append(self.all_paths.index(path))
            except ValueError:
                pass

        self._tag_btn.setEnabled(False)
        self._cap_btn.setEnabled(False)
        self._full_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._progress.setMaximum(len(indices))
        self._progress.setValue(0)
        self._progress.show()

        self._proc_worker = ProcessingWorker(
            self.all_paths, indices,
            self._out_edit.text().strip() or None,
            self._active_prompt_key(),
            mode=mode,
        )
        self._proc_worker.image_done.connect(self._on_image_done)
        self._proc_worker.progress_updated.connect(self._on_progress)
        self._proc_worker.all_done.connect(self._on_all_done)
        self._proc_worker.start()

    def stop_processing(self):
        if self._proc_worker:
            self._proc_worker.stop()
        self._status.setText(tr("status_stopped"))

    def _on_image_done(self, idx: int, tags: str, caption: str, status: str):
        item = self._list.item(idx)
        if item:
            name   = Path(self.all_paths[idx]).name
            prefix = {"done": "✅ ", "error": "❌ ", "skipped": "⏭ "}.get(status, "")
            item.setText(prefix + name)
            bg = {
                "done":    QColor("#1a3a2a"),
                "error":   QColor("#3a1a1a"),
                "skipped": QColor("#2a2a2a"),
            }.get(status)
            if bg:
                item.setBackground(bg)

        cur = self._list.selectedItems()
        if cur and cur[-1].data(Qt.ItemDataRole.UserRole) == self.all_paths[idx]:
            self._tags.setPlainText(tags)
            self._cap.setPlainText(caption)
            self._save_btn.setEnabled(bool(tags or caption))

    def _on_progress(self, done: int, total: int, name: str):
        self._progress.setValue(done)
        self._status.setText(tr("status_progress", done=done, total=total, name=name))

    def _on_all_done(self, total: int, skipped: int, errors: int):
        self._tag_btn.setEnabled(True)
        self._cap_btn.setEnabled(True)
        self._full_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._progress.hide()
        self._status.setText(tr("status_done", total=total, skip=skipped, err=errors))

    # ── Editor ───────────────────────────────────────────────────────────────

    def _save_current(self):
        """Speichert editierte Tags/Caption für das aktuelle Bild."""
        from cap_app.batch import save_edited_result
        items = self._list.selectedItems()
        if not items:
            return
        path    = Path(items[-1].data(Qt.ItemDataRole.UserRole))
        tags    = self._tags.toPlainText().strip()
        caption = self._cap.toPlainText().strip()
        try:
            save_edited_result(path, tags, caption,
                               self._out_edit.text().strip() or None)
            self._status.setText(tr("status_saved", name=path.name))
        except Exception as e:
            self._status.setText(tr("status_save_err", err=e))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _toggle_lang(self):
        from ui.i18n import set_lang, get_lang
        set_lang("en" if get_lang() == "de" else "de")
        self.retranslate()

    def retranslate(self):
        """Aktualisiert alle UI-Texte nach Sprachwechsel."""
        from ui.i18n import get_lang
        lang = get_lang()
        self.setWindowTitle(tr("win_title"))
        if hasattr(self, "_lang_btn"):
            self._lang_btn.setText("🇩🇪 DE" if lang == "de" else "🇬🇧 EN")
        if hasattr(self, "_tb_actions"):
            for action, key in zip(self._tb_actions,
                                   ["tb_open_folder", "tb_add_files", "tb_clear",
                                    "tb_select_all", "tb_select_none"]):
                action.setText(tr(key))
        if hasattr(self, "_edit_menu"):    self._edit_menu.setTitle(tr("menu_edit"))
        if hasattr(self, "_act_prompts"):  self._act_prompts.setText(tr("act_prompts"))
        if hasattr(self, "_act_settings"): self._act_settings.setText(tr("act_settings"))
        if hasattr(self, "_preview"):
            if not self._preview.pixmap() or self._preview.pixmap().isNull():
                self._preview.setText(tr("preview_hint"))
        if hasattr(self, "_tags_box"):   self._tags_box.setTitle(tr("lbl_tags"))
        if hasattr(self, "_cap_box"):    self._cap_box.setTitle(tr("lbl_caption"))
        if hasattr(self, "_tags"):       self._tags.setPlaceholderText(tr("ph_tags"))
        if hasattr(self, "_cap"):        self._cap.setPlaceholderText(tr("ph_caption"))
        if hasattr(self, "_save_btn"):   self._save_btn.setText(tr("btn_save"))
        if hasattr(self, "_mode_box"):   self._mode_box.setTitle(tr("grp_mode"))
        if hasattr(self, "_lbl_output"): self._lbl_output.setText(tr("lbl_output"))
        if hasattr(self, "_out_edit"):   self._out_edit.setPlaceholderText(tr("ph_output"))
        if hasattr(self, "_llm_lbl"):    self._llm_lbl.setText(tr("lbl_llm"))
        if hasattr(self, "_llm_combo"):  self._llm_combo.setToolTip(tr("llm_tooltip"))
        if hasattr(self, "_tag_btn"):
            self._tag_btn.setText(tr("btn_tag")); self._tag_btn.setToolTip(tr("tip_tag"))
        if hasattr(self, "_cap_btn"):
            self._cap_btn.setText(tr("btn_cap")); self._cap_btn.setToolTip(tr("tip_cap"))
        if hasattr(self, "_full_btn"):
            self._full_btn.setText(tr("btn_full")); self._full_btn.setToolTip(tr("tip_full"))
        self._update_count()

    def _list_key_press(self, event):
        """Entfernt markierte Bilder mit Entf/Del aus der Liste."""
        from PyQt6.QtCore import Qt
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self._remove_selected()
        else:
            QListWidget.keyPressEvent(self._list, event)

    def _remove_selected(self):
        """Entfernt alle markierten Einträge aus Liste und all_paths."""
        rows = sorted(
            {self._list.row(item) for item in self._list.selectedItems()},
            reverse=True   # von unten nach oben löschen, damit Indizes stimmen
        )
        if not rows:
            return
        for row in rows:
            self._list.takeItem(row)
            if row < len(self.all_paths):
                del self.all_paths[row]
        # Vorschau leeren wenn kein Bild mehr ausgewählt ist
        if not self._list.selectedItems():
            self._preview.clear()
            self._preview.setText(tr("preview_hint"))
            self._tags.clear()
            self._cap.clear()
            self._save_btn.setEnabled(False)
        self._update_count()

    def _on_tagger_toggle(self, key: str, btn: "QPushButton") -> None:
        """Schreibt den enabled-Flag in settings.json und aktualisiert Button-Text."""
        import json
        checked = btn.isChecked()
        btn.setText("ON" if checked else "OFF")
        try:
            cfg_path = Path(__file__).parent.parent / "config" / "settings.json"
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            cfg["models"].setdefault(key, {})["enabled"] = checked
            cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False),
                                encoding="utf-8")
            # Settings-Cache in tagger.py leeren
            try:
                from pipeline.tagger import reload_settings
                reload_settings()
            except Exception:
                pass
        except Exception as e:
            self._status.setText(f"Toggle-Fehler: {e}")

    def _update_count(self):
        n = len(self.all_paths)
        s = len(self._list.selectedItems())
        if   n == 0: self._count_label.setText(tr("count_none"))
        elif s > 0:  self._count_label.setText(tr("count_selected", n=n, s=s))
        else:        self._count_label.setText(tr("count_loaded", n=n))


# ─────────────────────────────────────────────────────────────────────────────
# App-Einstiegspunkt
# ─────────────────────────────────────────────────────────────────────────────

def run_app():
    import sys
    app = QApplication(sys.argv)
    app.setApplicationName(tr("win_title"))
    app.setStyle("Fusion")          # einheitlicher, moderner Look
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
