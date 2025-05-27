import os
import re
from datetime import datetime

import numpy as np
from PIL import Image, PngImagePlugin

# ------------------------------------------------------------
#  Torch nur importieren, wenn vorhanden
# ------------------------------------------------------------
try:
    import torch
except ImportError:
    torch = None


# ------------------------------------------------------------
#  %date:FORMAT%  →  Zeit/Datum einsetzen
# ------------------------------------------------------------
_DATE_TOKEN_RE = re.compile(r"%date:([^%]+)%")
_DATE_MAP = {
    "yyyy": "%Y", "yy": "%y",
    "MM":   "%m", "dd": "%d",
    "HH":   "%H", "hh": "%H",
    "mm":   "%M", "ss": "%S",
}
def _expand_placeholders(text: str) -> str:
    def _repl(m):
        fmt = m.group(1)
        for k, v in _DATE_MAP.items():
            fmt = fmt.replace(k, v)
        return datetime.now().strftime(fmt)
    return _DATE_TOKEN_RE.sub(_repl, text)


# ------------------------------------------------------------
#  Torch-Tensor -> PIL.Image  (robust gegen >3 Kanäle)
# ------------------------------------------------------------
def _tensor_to_pil(t: "torch.Tensor") -> Image.Image:
    """
    Akzeptiert 2-, 3- oder 4-D Tensoren im Bereich 0…1.
    • [C,H,W] oder [1,C,H,W]
    • C beliebig; bei C==1 → Graustufe → RGB
      bei C>4 → 1. Kanal als Graustufe → RGB
    """
    if t.dim() == 4:            # Batch → 1. Element
        t = t[0]

    if t.dim() == 2:            # [H,W] → [1,H,W]
        t = t.unsqueeze(0)
    if t.dim() != 3:
        raise TypeError(f"Unerwartete Tensor-Dimensionen: {t.shape}")

    # --- Kanäle normalisieren --------------------------------
    c, h, w = t.shape
    if c == 1:                  # Graustufe → drei Kanäle
        t = t.repeat(3, 1, 1)
    elif c in (3, 4):
        pass                    # RGB / RGBA ok
    else:
        # z. B. 1536 Kanäle → ersten Kanal nehmen → als Graustufe
        t = t[0].unsqueeze(0).repeat(3, 1, 1)

    # --- 0…1 → 0…255, nach NumPy & transponieren -------------
    arr = (
        t.detach()
         .clamp(0, 1)
         .mul(255)
         .byte()
         .cpu()
         .numpy()
    )                           # [C,H,W]
    arr = np.transpose(arr, (1, 2, 0))  # [H,W,C]
    return Image.fromarray(arr)


# ------------------------------------------------------------
#  NumPy-Array -> PIL.Image  (gleiches Prinzip)
# ------------------------------------------------------------
def _numpy_to_pil(a: np.ndarray) -> Image.Image:
    """
    Erwartet [H,W] oder [H,W,C] im Bereich 0…1.
    Bei C==1 → Graustufe zu RGB,
    bei C>4  → Kanal 0 als Graustufe zu RGB
    """
    if a.ndim == 2:             # Graustufe
        a = a[:, :, None]

    if a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    elif a.shape[2] in (3, 4):
        pass
    else:
        a = np.repeat(a[:, :, 0:1], 3, axis=2)

    arr = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(arr)


class SaveImagePlusPlus:
    """
    Speichert Bilder aus ComfyUI-Workflows.
    • Platzhalter %date:…% im Dateinamen
    • Standard- oder eigener Unterordner
    • PNG / JPEG / WEBP / TIFF
    • DPI, Qualität (Default 100), optionale Metadaten
    • Akzeptiert torch.Tensor, numpy.ndarray, PIL.Image
    """
    CATEGORY    = "image/output"
    OUTPUT_NODE = True

    # --------------------------------------------------------
    # Eingänge
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
                                      {"default": 300, "min": 1, "max": 12000}),
                "quality":           ("INT",
                                      {"default": 100, "min": 1, "max": 100}),
                "metadata":          ("BOOLEAN",
                                      {"default": False}),
                "use_custom_folder": ("BOOLEAN",
                                      {"default": False}),
                "custom_folder":     ("STRING",
                                      {"default": ""}),
            }
        }

    RETURN_TYPES = ()                 # kein sichtbarer Output
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

        # 1) Zielordner bestimmen ------------------------------
        import folder_paths
        base_dir = folder_paths.get_output_directory()
        if use_custom_folder and custom_folder.strip():
            base_dir = os.path.join(base_dir, custom_folder.strip())
        os.makedirs(base_dir, exist_ok=True)

        # 2) Dateinamen aufbereiten ----------------------------
        name = _expand_placeholders(file_name.strip())
        fmt  = format.upper()
        ext  = "." + fmt.lower()
        if not name.lower().endswith(ext):
            name += ext

        out_path = os.path.join(base_dir, name)
        if os.path.exists(out_path):               # durchnummerieren
            stem, ext_only = os.path.splitext(name)
            counter = 1
            while True:
                alt_path = os.path.join(base_dir,
                                        f"{stem}_{counter}{ext_only}")
                if not os.path.exists(alt_path):
                    out_path = alt_path
                    break
                counter += 1

        # 3) Eingabe -> PIL.Image ------------------------------
        if torch is not None and isinstance(image, torch.Tensor):
            img = _tensor_to_pil(image)
        elif isinstance(image, np.ndarray):
            img = _numpy_to_pil(image)
        elif isinstance(image, Image.Image):
            img = image
        else:
            raise TypeError("Bildtyp nicht unterstützt: "
                            f"{type(image).__name__}")

        # 4) Save-Parameter ------------------------------------
        kwargs = {"dpi": (dpi, dpi)}
        if fmt in ("JPEG", "WEBP"):
            kwargs["quality"] = int(quality)

        if metadata and fmt == "PNG":
            pnginfo = PngImagePlugin.PngInfo()
            pnginfo.add_text("GeneratedWith", "ComfyUI SaveImagePlusPlus")
            kwargs["pnginfo"] = pnginfo

        # 5) Speichern ----------------------------------------
        img.save(out_path, fmt, **kwargs)
        return ()


# ------------------------------------------------------------
#  Registrierung für ComfyUI
# ------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "SaveImagePlusPlus": SaveImagePlusPlus,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveImagePlusPlus": "Save Image Plus++",
}
