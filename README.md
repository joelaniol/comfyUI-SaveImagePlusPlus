# comfyUI-SaveImagePlusPlus

`SaveImagePlusPlus` ist ein Plugin für ComfyUI, das es ermöglicht, Bilder mit erweiterten Optionen abzuspeichern. Der Node akzeptiert ein Bild und speichert es in einem wählbaren Format (PNG, JPEG, WEBP oder TIFF). Zusätzlich können DPI, Qualität und optionale Metadaten gesetzt werden. Über den Parameter `subfolder` lassen sich Unterordner im Ausgabeverzeichnis anlegen. Der angegebene Pfad wird dabei mit `os.path.normpath` bereinigt; absolute Pfade oder Angaben, die mit `..` beginnen, werden verworfen.

## Funktionsweise

1. **Dateipfad & Unterordner**  
   Der Node erstellt bei Bedarf den angegebenen Unterordner im Zielverzeichnis und baut daraus den finalen Pfad zur Ausgabedatei.
2. **Bildverarbeitung**  
   Wird ein Numpy-Array übergeben, wird es zu einem `PIL.Image` konvertiert, um es in gängige Bildformate speichern zu können.
3. **Speicheroptionen**  
   - `dpi`: Auflösung, z. B. `[300, 300]`.
   - `quality`: JPEG/WEBP-Qualität (1–100).
   - `metadata`: Bei PNGs kann ein einfacher `tEXt`-Chunk mit dem Hinweis "GeneratedWith: ComfyUI SaveImagePlusPlus" gespeichert werden.
   - `include_workflow_meta`: Fügt die von ComfyUI erzeugten Workflow-Metadaten mit in den PNG-`tEXt`-Chunk ein. Nicht-String-Werte werden dabei als JSON gespeichert.
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


