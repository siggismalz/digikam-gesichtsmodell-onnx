# Gesichter erkennen

Mein mit digiKam angelerntes Gesichtsmodell als ONNX-Datei, mit einem kleinen Python-Skript zum Ausprobieren. Es erkennt die Personen, für die es angelernt wurde, und läuft lokal.

Du brauchst Python 3.12. Lade das Repository herunter, entpacke es und öffne ein Terminal im Ordner. Dann:

```sh
python -m pip install -r requirements.txt
python testen.py "Pfad/zum/Bild.jpg"
```

Du kannst auch einen ganzen Ordner angeben. Oder lege JPEG-/PNG-Bilder in `testbilder` und starte einfach `python testen.py`.

Das Skript gibt zu jedem Bild die erkannten Namen aus. Unsichere Gesichter erscheinen als `Unbekannt`. Deine Bilder werden nicht verändert oder hochgeladen. digiKam brauchst du dafür nicht.

Falls dein Rechner den Befehl `python` nicht kennt, verwende `python3`.

Die Modelle liegen in `modelle`. SFace wurde mit meinem Personenklassifikator zusammengefasst; YuNet übernimmt die Gesichtssuche. Die Basismodelle stammen aus [OpenCV Zoo](https://github.com/opencv/opencv_zoo): SFace von Yaoyao Zhong, ONNX-Konvertierung von Chengrui Wang (Apache 2.0), und YuNet von Shiqi Yu (MIT). Die Lizenztexte liegen in `LIZENZEN`.
