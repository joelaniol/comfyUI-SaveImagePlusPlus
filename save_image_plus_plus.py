"""
SaveImagePlusPlus – erweiterter Output-Node für ComfyUI
v2025-06-05  (Joel-Fix 5)

• PNG, JPEG, WEBP, TIFF speichern exakt dieselben Metadaten
  (prompt, workflow …) wie der Standard-Save-PNG-Node.
• Nutzt die von ComfyUI gelieferten Hidden-Inputs 'prompt' und
  'extra_pnginfo' – dadurch versionssicher.
• Neuer Optionsschalter **lossless**:
    – JPEG:  quality = 100, subsampling = 0 (4:4:4), optimize = True  
    – WEBP:  lossless = True  
    – TIFF:  compression = "none"  
    – PNG:   bleibt unverändert, da ohnehin verlustfrei
"""

import os
import re
import json
import struct
from datetime import datetime

import numpy as np
from PIL import Image, PngImagePlugin

# ─ PyTorch ist in ComfyUI fast immer vorhanden ──────────────────────────────
try:
    import torch
except ImportError:
    torch = None


# ─────────────────────────────────────────────────────────────────────────────
# 1) Platzhalter %date:…%  → aktuelles Datum/Zeit
# ─────────────────────────────────────────────────────────────────────────────
_DATE_TOKEN_RE = re.compile(r"%date:([^%]+)%")
_DATE_MAP      = {"yyyy": "%Y", "yy": "%y",
                  "MM": "%m",  "dd": "%d",
                  "HH": "%H",  "hh": "%H",
                  "mm": "%M",  "ss": "%S"}

def _expand_placeholders(text: str) -> str:
    def _repl(m):
        fmt = m.group(1)
        for k, v in _DATE_MAP.items():
            fmt = fmt.replace(k, v)
        return datetime.now().strftime(fmt)
    return _DATE_TOKEN_RE.sub(_repl, text)


# ─────────────────────────────────────────────────────────────────────────────
# 2) Bild-Helper  (Tensor / Array → CHW → PIL.Image)
# ─────────────────────────────────────────────────────────────────────────────
def _to_chw(t):
    is_torch = torch is not None and isinstance(t, torch.Tensor)
    if t.ndim == 4: t = t[0]                         # Batch entfernen
    if t.ndim == 2: t = (t[None] if is_torch else t[..., None])
    if t.ndim != 3: raise TypeError(f"Unerwartete Dimension: {t.shape}")
    c_first, c_last = t.shape[0] <= 4, t.shape[-1] <= 4
    chw = (t if c_first and not c_last else
           (t.permute(2, 0, 1) if is_torch else np.transpose(t, (2, 0, 1)))
           if c_last and not c_first else t)
    if   chw.shape[0] == 1:
        chw = chw.repeat(3, 1, 1) if is_torch else np.repeat(chw, 3, 0)
    elif chw.shape[0] not in (3, 4):
        g = chw[0:1]
        chw = g.repeat(3, 1, 1) if is_torch else np.repeat(g, 3, 0)
    return chw

def _tensor_to_pil(t):
    arr = _to_chw(t).detach().clamp(0, 1).mul(255).to(torch.uint8).cpu().numpy()
    return Image.fromarray(np.transpose(arr, (1, 2, 0)))

def _numpy_to_pil(a):
    arr = _to_chw(a)
    arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(np.transpose(arr, (1, 2, 0)))


# ─────────────────────────────────────────────────────────────────────────────
# 3) JPEG-Kommentar-Segment (Fallback < Pillow 9.4)
# ─────────────────────────────────────────────────────────────────────────────
def _make_jpeg_comment_segment(payload: bytes) -> bytes:
    return b"\xFF\xFE" + struct.pack(">H", 2 + len(payload)) + payload


# ─────────────────────────────────────────────────────────────────────────────
# 4) SaveImagePlusPlus-Node
# ─────────────────────────────────────────────────────────────────────────────
class SaveImagePlusPlus:
    CATEGORY, OUTPUT_NODE = "image/output", True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":  ("IMAGE",),
                "file_name": ("STRING", {
                    "default": "%date:yyyyMMdd%_%date:HHmmss%_standard_"
                }),
                "format": (["PNG", "JPEG", "WEBP", "TIFF"], {"default": "PNG"}),
                "dpi":    ("INT", {"default": 300, "min": 1, "max": 12000}),
                "quality":("INT", {"default": 100, "min": 1, "max": 100}),
                "lossless": ("BOOLEAN", {"default": False}),
                "metadata": ("BOOLEAN", {"default": False}),
                "include_workflow_meta": ("BOOLEAN", {"default": False}),
                "use_custom_folder": ("BOOLEAN", {"default": False}),
                "custom_folder": ("STRING", {"default": ""}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }

    RETURN_TYPES = ()
    FUNCTION     = "save_image"

    # -------------------------------------------------------------------------
    def save_image(
        self, image, file_name, format, dpi, quality, lossless,
        metadata, include_workflow_meta, use_custom_folder, custom_folder,
        prompt=None, extra_pnginfo=None,
    ):
        # 4.1 Zielordner
        import folder_paths
        base_dir = folder_paths.get_output_directory()
        if use_custom_folder and custom_folder.strip():
            safe = os.path.normpath(custom_folder.strip())
            if os.path.isabs(safe) or safe.startswith(".."):
                raise ValueError("Ungültiger Unterordner")
            base_dir = os.path.join(base_dir, safe)
        os.makedirs(base_dir, exist_ok=True)

        # 4.2 Dateiname
        name = _expand_placeholders(file_name.strip())
        fmt  = format.upper()
        if not name.lower().endswith("." + fmt.lower()):
            name += "." + fmt.lower()
        out_path = os.path.join(base_dir, name)
        if os.path.exists(out_path):
            stem, ext = os.path.splitext(name)
            i = 1
            while True:
                alt = os.path.join(base_dir, f"{stem}_{i}{ext}")
                if not os.path.exists(alt):
                    out_path = alt
                    break
                i += 1

        # 4.3 PIL-Bild erzeugen
        if torch is not None and isinstance(image, torch.Tensor):
            img = _tensor_to_pil(image)
        elif isinstance(image, np.ndarray):
            img = _numpy_to_pil(image)
        elif isinstance(image, Image.Image):
            img = image
        else:
            raise TypeError(f"Bildtyp nicht unterstützt: {type(image).__name__}")

        # 4.4 Basis-Save-Parameter
        kwargs = {"dpi": (dpi, dpi)}
        if fmt == "JPEG":
            if lossless:
                kwargs.update({"quality": 100, "subsampling": 0, "optimize": True})
            else:
                kwargs["quality"] = int(quality)
        elif fmt == "WEBP":
            if lossless:
                kwargs["lossless"] = True
            else:
                kwargs["quality"] = int(quality)
        elif fmt == "TIFF":
            kwargs["compression"] = "none" if lossless else "tiff_adobe_deflate"
        # PNG hat immer verlustfreie Kompression; keine weiteren Flags nötig.

        # 4.5 Metadaten sammeln
        meta = {}
        if include_workflow_meta:
            if isinstance(extra_pnginfo, dict):
                meta.update(extra_pnginfo)       # enthält prompt, workflow, …
            if prompt is not None and "prompt" not in meta:
                meta["prompt"] = prompt
        if metadata:
            meta["GeneratedWith"] = "ComfyUI SaveImagePlusPlus"

        # 4.6 PNG-Pfad
        if fmt == "PNG" and meta:
            info = PngImagePlugin.PngInfo()
            for k, v in meta.items():
                try:
                    info.add_text(str(k), v if isinstance(v, str) else json.dumps(v))
                except Exception:
                    pass
            kwargs["pnginfo"] = info

        # 4.7 JPEG / WEBP / TIFF
        elif fmt in ("JPEG", "WEBP", "TIFF") and meta:
            meta_json = json.dumps(meta, ensure_ascii=False)
            if fmt == "JPEG":
                # Pillow ≥ 9.4 unterstützt comment=
                try:
                    img.save(os.devnull, "JPEG", comment=b"")
                    kwargs["comment"] = meta_json
                except Exception:
                    exif = img.getexif()
                    exif[0x9286] = meta_json       # UserComment
                    kwargs["exif"] = exif
            else:                                  # WEBP / TIFF
                exif = img.getexif()
                exif[0x9286] = meta_json
                kwargs["exif"] = exif

        # 4.8 Speichern
        img.save(out_path, fmt, **kwargs)
        return ()


# ─ Registrierung für ComfyUI ────────────────────────────────────────────────
NODE_CLASS_MAPPINGS        = {"SaveImagePlusPlus": SaveImagePlusPlus}
NODE_DISPLAY_NAME_MAPPINGS = {"SaveImagePlusPlus": "Save Image Plus++"}
