# comfyUI-SaveImagePlusPlus

`SaveImagePlusPlus` ist ein Plugin für ComfyUI, das es ermöglicht, Bilder mit erweiterten Optionen abzuspeichern. Der Node akzeptiert ein Bild und speichert es in einem wählbaren Format (PNG, JPEG, WEBP oder TIFF). Zusätzlich können DPI, Qualität und optionale Metadaten gesetzt werden. Optional kann über `use_custom_folder` und `custom_folder` ein eigener Unterordner im Ausgabeverzeichnis definiert werden.

## Funktionsweise

1. **Dateipfad & Unterordner**
   Der Dateiname kann `%date:…%`-Platzhalter enthalten. Über `use_custom_folder` und `custom_folder` lässt sich optional ein Unterordner anlegen, aus dem der finale Pfad zur Ausgabedatei gebildet wird.
2. **Bildverarbeitung**  
   Wird ein Numpy-Array übergeben, wird es zu einem `PIL.Image` konvertiert, um es in gängige Bildformate speichern zu können.
3. **Speicheroptionen**  
   - `dpi`: Auflösung, z. B. `[300, 300]`.
   - `quality`: JPEG/WEBP-Qualität (1–100).
   - `metadata`: Bei PNGs kann ein einfacher `tEXt`-Chunk mit dem Hinweis "GeneratedWith: ComfyUI SaveImagePlusPlus" gespeichert werden.
4. **Speichern**  
   Das Bild wird mit den gewählten Optionen abgespeichert und erzeugt keinen weiteren Node-Ausgang.

## Registrierung in ComfyUI

Der Node wird über folgende Mappings verfügbar gemacht:

```python
NODE_CLASS_MAPPINGS = {
    "SaveImagePlusPlus": SaveImagePlusPlus
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveImagePlusPlus": "Save Image Plus++"
}
```

Damit taucht er in ComfyUI unter "image/output" auf.

---

Dieses Plugin richtet sich an Nutzer, die Bilder aus ihren ComfyUI-Workflows präzise und mit Metadaten versehen speichern möchten.

## Installation

Kopiere die Python-Datei in dein ComfyUI-Installationsverzeichnis unter `custom_nodes/` und starte ComfyUI neu. Der Node erscheint anschließend im Node-Browser.


