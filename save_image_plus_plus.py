import os
import numpy as np
from PIL import Image, PngImagePlugin


class SaveImagePlusPlus:
    """
    Einzelner Node zum Speichern von Bildern mit einstellbarem
    Dateiformat, DPI, JPEG-/WEBP-Qualität, Metadaten und Unterordner.
    """
    CATEGORY = "image/output"
    OUTPUT_NODE = True        # Markiert den Node als Ausgabe-Knoten

    # ------------------------------------------------------------
    # Eingänge / Widgets
    # ------------------------------------------------------------
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":     ("IMAGE",),
                "file_path": ("STRING",  {"default": "output/image.png"}),
                "format":    (["PNG", "JPEG", "WEBP", "TIFF"], {"default": "PNG"}),
                "dpi":       ("INT2",
                              {"default": [300, 300],
                               "min":     [1, 1],
                               "max":     [12000, 12000],
                               "step":    [1, 1]}),
                "quality":   ("INT",     {"default": 95, "min": 1, "max": 100}),
                "metadata":  ("BOOLEAN", {"default": False}),
                "subfolder": ("STRING",  {"default": ""})
            }
        }

    # ------------------------------------------------------------
    # *** Keine Ausgänge: leeres Tupel! ***
    # ------------------------------------------------------------
    RETURN_TYPES = ()
    FUNCTION = "save_image"

    # ------------------------------------------------------------
    # Hauptfunktion
    # ------------------------------------------------------------
    def save_image(self,
                   image,
                   file_path,
                   format,
                   dpi,
                   quality,
                   metadata,
                   subfolder):
        # 1) Ziel­verzeichnis + Unterordner anlegen ----------------
        base_dir, name = os.path.split(file_path)
        if subfolder:
            base_dir = os.path.join(base_dir, subfolder)
        os.makedirs(base_dir, exist_ok=True)

        # 2) Numpy-Array → PIL.Image -------------------------------
        img = image
        if isinstance(img, np.ndarray):
            arr = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            img = Image.fromarray(arr)

        # 3) Save-Parameter zusammenstellen ------------------------
        kwargs = {"dpi": tuple(dpi)}
        fmt = format.upper()
        if fmt in ("JPEG", "WEBP"):
            kwargs["quality"] = int(quality)

        # 4) optionale Metadaten (PNG tEXt-Chunk) ------------------
        if metadata and fmt == "PNG":
            pnginfo = PngImagePlugin.PngInfo()
            pnginfo.add_text("GeneratedWith", "ComfyUI SaveImagePlusPlus")
            kwargs["pnginfo"] = pnginfo

        # 5) Datei speichern --------------------------------------
        out_path = os.path.join(base_dir, name)
        img.save(out_path, fmt, **kwargs)

        # 6) Node hat keinen Ausgang ------------------------------
        return ()


# ------------------------------------------------------------
# ComfyUI-Registrierung
# ------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "SaveImagePlusPlus": SaveImagePlusPlus
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveImagePlusPlus": "Save Image Plus++"
}
