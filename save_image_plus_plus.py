import os
import re
from datetime import datetime

import numpy as np
from PIL import Image, PngImagePlugin

# — PyTorch ist in ComfyUI fast immer vorhanden —
try:
    import torch
except ImportError:
    torch = None


# ------------------------------------------------------------
# 1) Platzhalter %date:…%  →  aktuelles Datum/Zeit einsetzen
# ------------------------------------------------------------
_DATE_TOKEN_RE = re.compile(r"%date:([^%]+)%")
_DATE_MAP = {"yyyy": "%Y", "yy": "%y",
             "MM": "%m",  "dd": "%d",
             "HH": "%H",  "hh": "%H",   # 24-h
             "mm": "%M",  "ss": "%S"}

def _expand_placeholders(text: str) -> str:
    def _repl(m):
        fmt = m.group(1)
        for k, v in _DATE_MAP.items():
            fmt = fmt.replace(k, v)
        return datetime.now().strftime(fmt)
    return _DATE_TOKEN_RE.sub(_repl, text)


# ------------------------------------------------------------
# 2) Kanal-Heuristik  →  Tensor/Array in Form [3, H, W] bringen
# ------------------------------------------------------------
def _to_chw(t):
    is_torch = torch is not None and isinstance(t, torch.Tensor)

    # -------- Batch-Dimension entfernen --------
    if t.ndim == 4:              # [B, …]  →  erstes Bild
        t = t[0]

    # -------- 2-D Graustufe unterstützen -------
    if t.ndim == 2:              # [H,W] → [1,H,W] oder [H,W,1]
        t = (t[None, ...] if is_torch else t[..., None])

    if t.ndim != 3:
        raise TypeError(f"Unerwartete Dimension: {tuple(t.shape)}")

    # -------- Erkennen, wo die Kanäle liegen ----
    c_first = t.shape[0] <= 4           # [C,H,W]
    c_last  = t.shape[-1] <= 4          # [H,W,C]

    if c_first and not c_last:          # Kanäle vorn
        chw = t
    elif c_last and not c_first:        # Kanäle hinten
        chw = t.permute(2, 0, 1) if is_torch else np.transpose(t, (2, 0, 1))
    else:                               # beides oder keins → nehmen vorn
        chw = t

    # -------- Kanalzahl normalisieren ----------
    if chw.shape[0] == 1:               # Graustufe → RGB
        chw = chw.repeat(3, 1, 1) if is_torch else np.repeat(chw, 3, axis=0)
    elif chw.shape[0] in (3, 4):        # RGB / RGBA ok
        pass
    else:                               # z. B. 1536 Kanäle → 1. Kanal als Graustufe
        g = chw[0:1]
        chw = g.repeat(3, 1, 1) if is_torch else np.repeat(g, 3, axis=0)

    return chw


# ------------------------------------------------------------
# 3) Hilfsfunktionen: nach PIL.Image konvertieren
# ------------------------------------------------------------
def _tensor_to_pil(t):
    chw = _to_chw(t)
    arr = (
        chw.detach().clamp(0, 1).mul(255).to(torch.uint8).cpu().numpy()
    )
    arr = np.transpose(arr, (1, 2, 0))        # [H,W,C]
    return Image.fromarray(arr)

def _numpy_to_pil(a):
    chw = _to_chw(a)
    arr = (np.clip(chw, 0, 1) * 255).astype(np.uint8)
    arr = np.transpose(arr, (1, 2, 0))
    return Image.fromarray(arr)


def _get_default_pnginfo():
    """Best effort: retrieve ComfyUI's default metadata."""
    import importlib
    candidates = [
        ("nodes.save_image", ("get_pnginfo", "get_png_info", "get_metadata")),
        ("nodes", ("get_pnginfo", "get_png_info", "get_metadata")),
        ("comfy.utils", ("get_pnginfo", "generate_pnginfo", "generate_workflow_metadata")),
    ]
    for mod_name, funcs in candidates:
        try:
            module = importlib.import_module(mod_name)
        except Exception:
            continue
        for fn in funcs:
            func = getattr(module, fn, None)
            if callable(func):
                try:
                    data = func()
                except Exception:
                    continue
                if isinstance(data, dict):
                    return data
    return {}


# ------------------------------------------------------------
# 4) Der eigentliche ComfyUI-Node
# ------------------------------------------------------------
class SaveImagePlusPlus:
    """
    Speichert ein Bild:
      • Dateiname mit %date:…%-Platzhaltern
      • Standard-Output-Ordner oder eigener Unterordner
      • PNG / JPEG / WEBP / TIFF, Qualität 100
      • DPI-Flag, optionale PNG-Metadaten
      • akzeptiert torch.Tensor, numpy.ndarray, PIL.Image
      • optional: Workflow-Metadaten im PNG speichern
    """
    CATEGORY    = "image/output"
    OUTPUT_NODE = True

    # -------- Eingänge / Widgets ------------------------------
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":             ("IMAGE",),
                "file_name":         ("STRING",
                                      {"default":
                                       "%date:yyyyMMdd%_%date:HHmmss%_standard_"}),
                "format":            (["PNG", "JPEG", "WEBP", "TIFF"],
                                      {"default": "PNG"}),
                "dpi":               ("INT",
                                      {"default": 300, "min": 1, "max": 12000}),
                "quality":           ("INT",
                                      {"default": 100, "min": 1, "max": 100}),
                "metadata":          ("BOOLEAN",
                                      {"default": False}),
                "include_workflow_meta": ("BOOLEAN",
                                          {"default": False}),
                "use_custom_folder": ("BOOLEAN",
                                      {"default": False}),
                "custom_folder":     ("STRING",
                                      {"default": ""}),
            }
        }

    RETURN_TYPES = ()          # kein sichtbarer Output
    FUNCTION     = "save_image"

    # -------- Hauptlogik --------------------------------------
    def save_image(self,
                   image,
                   file_name,
                   format,
                   dpi,
                   quality,
                   metadata,
                   include_workflow_meta,
                   use_custom_folder,
                   custom_folder):

        # 4.1  Zielordner bestimmen
        import folder_paths
        base_dir = folder_paths.get_output_directory()
        if use_custom_folder and custom_folder.strip():
            # Pfad bereinigen und Traversal verhindern
            sanitized = os.path.normpath(custom_folder.strip())
            if os.path.isabs(sanitized) or sanitized.startswith('..') or sanitized.startswith(f"..{os.sep}"):
                raise ValueError("Ungültiger Unterordner")
            base_dir = os.path.join(base_dir, sanitized)
        os.makedirs(base_dir, exist_ok=True)

        # 4.2  Dateinamen mit Platzhaltern
        name = _expand_placeholders(file_name.strip())
        fmt  = format.upper()
        ext  = "." + fmt.lower()
        if not name.lower().endswith(ext):
            name += ext

        out_path = os.path.join(base_dir, name)
        if os.path.exists(out_path):               # durchnummerieren
            stem, ext_only = os.path.splitext(name)
            i = 1
            while True:
                alt = os.path.join(base_dir, f"{stem}_{i}{ext_only}")
                if not os.path.exists(alt):
                    out_path = alt
                    break
                i += 1

        # 4.3  Eingabe → PIL.Image
        if torch is not None and isinstance(image, torch.Tensor):
            img = _tensor_to_pil(image)
        elif isinstance(image, np.ndarray):
            img = _numpy_to_pil(image)
        elif isinstance(image, Image.Image):
            img = image
        else:
            raise TypeError(f"Bildtyp nicht unterstützt: {type(image).__name__}")

        # 4.4  Save-Parameter
        kwargs = {"dpi": (dpi, dpi)}
        if fmt in ("JPEG", "WEBP"):
            kwargs["quality"] = int(quality)

        if fmt == "PNG" and (metadata or include_workflow_meta):
            info = PngImagePlugin.PngInfo()
            if include_workflow_meta:
                default_meta = _get_default_pnginfo()
                for k, v in default_meta.items():
                    try:
                        info.add_text(str(k), str(v))
                    except Exception:
                        pass
            if metadata:
                info.add_text("GeneratedWith", "ComfyUI SaveImagePlusPlus")
            kwargs["pnginfo"] = info

        # 4.5  Speichern
        img.save(out_path, fmt, **kwargs)
        return ()


# -------- Registrierung für ComfyUI --------------------------
NODE_CLASS_MAPPINGS        = {"SaveImagePlusPlus": SaveImagePlusPlus}
NODE_DISPLAY_NAME_MAPPINGS = {"SaveImagePlusPlus": "Save Image Plus++"}
