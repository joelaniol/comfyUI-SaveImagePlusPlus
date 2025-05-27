import os
import re
from datetime import datetime

import numpy as np
from PIL import Image, PngImagePlugin

# Torch ist in ComfyUI fast immer vorhanden;
# falls nicht, funktioniert die Node trotzdem weiter
try:
    import torch
except ImportError:
    torch = None

# ------------------------------------------------------------
#  Hilfsfunktion: Platzhalter %date:FORMAT% ersetzen
# ------------------------------------------------------------
_DATE_TOKEN_RE = re.compile(r"%date:([^%]+)%")
_DATE_MAP = {
    "yyyy": "%Y",
    "yy":   "%y",
    "MM":   "%m",
    "dd":   "%d",
    "HH":   "%H",  # 24-h
    "hh":   "%H",  # 12-h → hier ebenfalls 24-h
    "mm":   "%M",
    "ss":   "%S",
}
def _expand_placeholders(text: str) -> str:
    def _repl(match):
        fmt = match.group(1)
        for k, v in _DATE_MAP.items():
            fmt = fmt.replace(k, v)
        return datetime.now().strftime(fmt)
    return _DATE_TOKEN_RE.sub(_repl, text)


# ------------------------------------------------------------
#  Torch-Tensor → PIL.Image
# ------------------------------------------------------------
def _tensor_to_pil(t: "torch.Tensor") -> Image.Image:
    """
    Erwartet einen Tensor [C,H,W] oder [1,C,H,W] im Bereich 0…1.
    """
    if t.dim() == 4:             # Batch → erstes Bild
        t = t[0]
    if t.shape[0] == 1:          # Graustufen → 3 Kanäle
        t = t.repeat(3, 1, 1)
    arr = (
        t.detach()
         .clamp(0, 1)
         .mul(255)
         .byte()
         .cpu()
         .numpy()
    )                            # → [C,H,W]
    arr = np.transpose(arr, (1, 2, 0))  # → [H,W,C]
    return Image.fromarray(arr)


class SaveImagePlusPlus:
    """
    Speichert ein Bild mit frei definierbarem Dateinamen
    (Platzhalter %date:…%) wahlweise im Standard-Output-Ordner
    oder in einem benutzerdefinierten Unterordner.
    Unterstützt PNG/JPEG/WEBP/TIFF, DPI, Qualität & Metadaten.
    """
    CATEGORY    = "image/output"
    OUTPUT_NODE = True

    # --------------------------------------------------------
    # Eingänge / Widgets
    # --------------------------------------------------------
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
                                      {"default": 300,
                                       "min": 1,
                                       "max": 12000}),
                "quality":           ("INT",
                                      {"default": 100,
                                       "min": 1,
                                       "max": 100}),
                "metadata":          ("BOOLEAN",
                                      {"default": False}),
                "use_custom_folder": ("BOOLEAN",
                                      {"default": False}),
                "custom_folder":     ("STRING",
                                      {"default": ""}),
            }
        }

    RETURN_TYPES = ()          # kein sichtbarer Ausgang
    FUNCTION     = "save_image"

    # --------------------------------------------------------
    # Hauptfunktion
    # --------------------------------------------------------
    def save_image(self,
                   image,
                   file_name,
                   format,
                   dpi,
                   quality,
                   metadata,
                   use_custom_folder,
                   custom_folder):

        # 1) Zielordner ermitteln ---------------------------------
        import folder_paths          # nur vorhanden, wenn unter ComfyUI
        base_dir = folder_paths.get_output_directory()
        if use_custom_folder and custom_folder.strip():
            base_dir = os.path.join(base_dir, custom_folder.strip())
        os.makedirs(base_dir, exist_ok=True)

        # 2) Dateinamen zusammensetzen ----------------------------
        name = _expand_placeholders(file_name.strip())
        fmt  = format.upper()
        ext  = "." + fmt.lower()
        if not name.lower().endswith(ext):
            name += ext

        # 3) Kollisionen vermeiden (Durchnummerieren) -------------
        out_path = os.path.join(base_dir, name)
        if os.path.exists(out_path):
            stem, ext_only = os.path.splitext(name)
            counter = 1
            while True:
                alt_path = os.path.join(base_dir,
                                        f"{stem}_{counter}{ext_only}")
                if not os.path.exists(alt_path):
                    out_path = alt_path
                    break
                counter += 1

        # 4) Eingangs-Bild in PIL umwandeln -----------------------
        img = image
        if torch is not None and isinstance(img, torch.Tensor):
            img = _tensor_to_pil(img)
        elif isinstance(img, np.ndarray):
            arr = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            img = Image.fromarray(arr)
        elif not isinstance(img, Image.Image):
            raise TypeError("Unbekannter Bildtyp "
                            f"({type(img).__name__}); erwarte Tensor, "
                            "NumPy-Array oder PIL.Image")

        # 5) Save-Parameter zusammenstellen -----------------------
        kwargs = {"dpi": (dpi, dpi)}
        if fmt in ("JPEG", "WEBP"):
            kwargs["quality"] = int(quality)

        if metadata and fmt == "PNG":
            pnginfo = PngImagePlugin.PngInfo()
            pnginfo.add_text("GeneratedWith", "ComfyUI SaveImagePlusPlus")
            kwargs["pnginfo"] = pnginfo

        # 6) Speichern -------------------------------------------
        img.save(out_path, fmt, **kwargs)
        return ()                 # kein sichtbarer Output


# ------------------------------------------------------------
# ComfyUI-Registrierung
# ------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "SaveImagePlusPlus": SaveImagePlusPlus,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveImagePlusPlus": "Save Image Plus++",
}
