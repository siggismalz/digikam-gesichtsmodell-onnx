# Gesichter erkennen



```sh
python -m pip install -r requirements.txt
python testen.py "Pfad/zum/Bild.jpg"
```

Du kannst auch einen ganzen Ordner angeben. Oder lege JPEG-/PNG-Bilder in `testbilder` und starte einfach `python testen.py`.

Das Skript gibt zu jedem Bild die erkannten Namen aus. Unsichere Gesichter erscheinen als `Unbekannt`. Deine Bilder werden nicht verändert oder hochgeladen. digiKam brauchst du dafür nicht.
