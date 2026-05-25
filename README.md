# Batcap

A desktop application for batch-tagging and captioning images using three specialized taggers and a Qwen 3.5 vision-language model. Designed to run on top of an existing [ComfyUI](https://github.com/comfyanonymous/ComfyUI) installation — no standalone model management needed.

---

## Features

- **Three independent taggers** — JTP PILOT v2, JTP-3 Hydra, DINOv3 — results are merged and deduplicated
- **Qwen 3.5 captioner** — generates natural-language captions from tags and an image, loaded via ComfyUI's quantization pipeline
- **Phase-based workflow** — tag all images first, then caption all at once; LLM loads once per batch instead of per image
- **Intermediate state** — per-image `.batcap.json` files store tags between phases so you can review and edit before captioning
- **In-place editor** — tags and captions are editable directly in the UI; save overwrites both the `.batcap.json` and the `.txt` output
- **DE / EN interface** — language can be switched at runtime via the toolbar button
- **ComfyUI VRAM management** — taggers and LLM share a 16 GB GPU cleanly; each model is unloaded before the next loads

---

## Requirements

| Dependency | Notes |
|---|---|
| [ComfyUI](https://github.com/comfyanonymous/ComfyUI) | Must be installed; `start.sh` uses its Python venv |
| [ComfyUI-RR-JointTagger](https://github.com/RedRocket-AI/ComfyUI-RR-JointTagger) | Custom node — provides all three taggers |
| [ComfyUI-QuantOps](https://github.com/silveroxides/ComfyUI-QuantOps) | Custom node — provides quantized inference for fp8/mxfp8 LLMs |
| PyQt6 | Installed automatically by `start.sh` if missing |
| A Qwen 3.5 model file | `.safetensors` in `ComfyUI/models/text_encoders/` |

**GPU:** An NVIDIA GPU with at least 16 GB VRAM is recommended. The Qwen 3.5 9B model in mxfp8 uses ~14 GB; DINOv3 uses ~5 GB. The app unloads one before loading the other.

---

## Installation

```bash
# 1. Clone or copy the captioner folder into your ComfyUI installation
cp -r captioner/ /path/to/ComfyUI/captioner/

# 2. Edit config/settings.json and set your ComfyUI path
#    (the default assumes /home/<user>/ComfyUI)

# 3. Place your Qwen model in ComfyUI/models/text_encoders/
#    Example: Qwen3.5-9B-Uncensored-merged.safetensors

# 4. Launch
cd /path/to/ComfyUI/captioner
./start.sh
```

`start.sh` automatically detects the ComfyUI Python venv, installs PyQt6 if needed, and sets the working directory that ComfyUI's `folder_paths` module expects.

---

## Directory Structure

```
captioner/
├── start.sh                  # Launch script
├── main.py                   # Entry point
├── config/
│   ├── settings.json         # Model paths, thresholds, generation parameters
│   └── prompts.json          # System prompts and tag prefixes per mode
├── pipeline/
│   ├── bootstrap.py          # sys.path setup, JointTagger registration
│   ├── tagger.py             # Runs all three taggers, VRAM cycle
│   ├── captioner.py          # Loads Qwen via ComfyUI, generates captions
│   ├── merger.py             # Merges and deduplicates tag lists
│   └── cleanup.py            # Post-processes merged tags
├── cap_app/
│   ├── batch.py              # Phase-based batch runner, .batcap.json state
│   └── file_utils.py         # Image collection, .txt output
└── ui/
    ├── app.py                # PyQt6 main window
    └── i18n.py               # DE/EN string translations
```

---

## Usage

### Basic Workflow

1. **Load images** — drag a folder onto the sidebar, use *Open Folder*, or *Add Files*
2. **Select images** — click individual items or use *Select All*
3. **Tag it!** — runs all three taggers; tags are saved to `.batcap.json` next to each image
4. **Review** — click through images; tags appear in the Tags field. Edit if needed, then click *💾 Save*
5. **Cap it!** — loads Qwen once, reads tags from `.batcap.json`, generates and saves captions
6. **Tag & Cap it!** — does both phases in sequence unattended

Output is written as `<imagename>.txt` next to each image (or in the configured output folder).

### Output Format

Each `.txt` file contains tags on the first line and the caption after a blank line:

```
1girl, solo, blonde hair, blue eyes, smile, looking at viewer

A young woman with long blonde hair and striking blue eyes smiles warmly
at the camera. Her expression is open and inviting, with soft lighting
highlighting her features against a neutral background.
```

### Phase Separation

The two-phase design means Qwen never has to compete with the taggers for VRAM:

```
Tag it!      →  JTP2 → JTP3 → DINOv3 → .batcap.json (no LLM loaded)
Cap it!      →  Qwen loads once → all images → Qwen unloads
Tag & Cap!   →  both phases back to back
```

If you stop after tagging, your tag data is safe in the `.batcap.json` files. You can run *Cap it!* later, or on a different selection.

---

## Configuration

### `config/settings.json`

```jsonc
{
  "comfyui_path": "/home/user/ComfyUI",   // Path to your ComfyUI installation
  "output": {
    "path": ""                             // Output folder; empty = next to images
  },
  "models": {
    "jtp_pilot2": {
      "threshold": 0.25,                  // Minimum score to include a tag
      "topk": 60,                         // Maximum number of tags returned
      "implications_mode": "constrain",   // "constrain" | "inherit" | "off"
      "exclude_tags": "male*, *artwork*"  // Glob patterns for tags to suppress
    },
    "jtp3_hydra": {
      "threshold": 0.35,
      "topk": 70,
      "cam_depth": 1                      // Class Activation Map depth (0 = off)
    },
    "dinov3": {
      "threshold": 0.85,
      "topk": 70,
      "use_aliases": true                 // Map aliases to canonical tag names
    },
    "qwen": {
      "model_file": "Qwen/model.safetensors",  // Relative to text_encoders/
      "quant_format": "auto",             // "auto" | "fp8" | "mxfp8" | "int8"
      "max_length": 1024,                 // Maximum tokens to generate
      "temperature": 0.7,
      "top_k": 64,
      "top_p": 0.95,
      "repetition_penalty": 1.05,
      "thinking": false                   // Enable Qwen3 extended thinking mode
    }
  }
}
```

### `config/prompts.json`

Contains one or more named prompt modes. Each mode has a `system_message` (the instruction given to Qwen) and an optional `tag_context_prefix` (prepended to the tag list in the user message). The active mode is selected via the **Mode** radio buttons in the UI and can also be edited live under *Edit → Edit Prompts*.

---

## The `.batcap.json` Intermediate File

After tagging, each image gets a sidecar file:

```jsonc
{
  "jtp2_tags": "1girl, solo, smile, ...",
  "jtp2_scores": { "1girl": 0.98, "solo": 0.95, ... },
  "jtp3_tags": "1girl, solo, ...",
  "jtp3_scores": { ... },
  "dino_tags": "...",
  "dino_scores": { ... },
  "tags_merged": "1girl, solo, smile, blonde hair, ...",
  "tags_clean":  "1girl, solo, smile, blonde hair, ...",
  "caption": "A young woman ...",         // filled after Cap it!
  "status": "tagged"                      // "tagged" | "complete"
}
```

These files are never deleted automatically. They let you re-run captioning with different prompts without re-tagging, and serve as a record of what each tagger contributed.

---

## Keyboard Shortcuts

| Action | Shortcut |
|---|---|
| Select all images | Toolbar button |
| Deselect all | Toolbar button |
| Stop current batch | ⏹ button |

---

## Troubleshooting

**App does not start / `ModuleNotFoundError: comfy`**
The app must be run via `start.sh`, which sets up the working directory and Python path that ComfyUI requires. Do not run `python main.py` directly.

**Tags show up but disappear when switching images**
Make sure you are using the current version. Earlier versions wrote tags only to the `.txt` output file; the current version writes them to `.batcap.json` immediately and reads from there on image selection.

**CUDA out of memory during captioning**
Your GPU has less than ~15 GB free VRAM. Try reducing `max_length` in `settings.json` to `512`. The tagger phase should complete normally; only the Qwen forward pass is affected.

**`Cap it!` skips all images**
The images have not been tagged yet — no `.batcap.json` with `tags_clean` exists. Run *Tag it!* first.

**Qwen generates empty or very short captions**
Check that `thinking` is set to `false` in `settings.json`. With `thinking: true`, the model may spend its token budget on the reasoning trace and truncate the visible output if `max_length` is too low.

---

## License

This project is a thin application layer on top of ComfyUI and its custom nodes. Please refer to the licenses of [ComfyUI](https://github.com/comfyanonymous/ComfyUI), [ComfyUI-RR-JointTagger](https://github.com/RedRocket-AI/ComfyUI-RR-JointTagger), and [ComfyUI-QuantOps](https://github.com/silveroxides/ComfyUI-QuantOps) for the terms that apply to the underlying components.
