# Persönliches Gesichtsmodell als ONNX

Ein lokal trainiertes Personenmodell und ein eigenständiges deutsches Testskript. Zum Ausführen werden weder digiKam noch eine Datenbank, ein GPU-Zugang oder ein KI-Konto benötigt. Internet wird nur zum Herunterladen des Repositorys und der Python-Pakete gebraucht.

Das mitgelieferte Modell erkennt die Personen aus seinem Trainingsbestand. Es lernt durch das Testskript keine neuen Personen. Unsichere Treffer erscheinen als `Unbekannt`; die Abweisung unbekannter Personen ist noch nicht unabhängig kalibriert.

## Installation

Voraussetzung: **Python 3.12, 64 Bit**. Das private Repository kann von Personen mit freigeschaltetem GitHub-Zugriff geklont oder als ZIP heruntergeladen werden. Bei einem ZIP-Download zuerst vollständig entpacken.

```shell
git clone https://github.com/siggismalz/digikam-gesichtsmodell-onnx.git
cd digikam-gesichtsmodell-onnx
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe testen.py --selbsttest
```

Linux und macOS:

```shell
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python testen.py --selbsttest
```

Auf Systemen, auf denen Python als `python3` installiert ist, für das Anlegen der Umgebung `python3 -m venv .venv` verwenden. Die obigen Befehle benötigen keine Aktivierung der Umgebung.

## Eigene Bilder testen

Ein Bild oder einen Ordner übergeben; Pfade mit Leerzeichen in Anführungszeichen setzen:

```powershell
.\.venv\Scripts\python.exe testen.py "C:\Fotos\bild.jpg"
.\.venv\Scripts\python.exe testen.py "C:\Fotos"
```

```shell
./.venv/bin/python testen.py "/home/name/Fotos/bild.jpg"
./.venv/bin/python testen.py "/home/name/Fotos"
```

Alternativ eigene JPEG-/PNG-Dateien in `testbilder` legen und `testen.py` ohne Bildpfad starten. Die Prüfung eines Ordners betrifft dessen direkte Bilddateien. Ausrichtung aus EXIF-Metadaten wird berücksichtigt.

Die Ausgabe nennt pro Bild die erkannten Personen oder `Unbekannt`. Fotos werden lokal gelesen und unverändert gelassen. Persönliche Testfotos und lokale digiKam-Einstellungen sind nicht Teil des Repositorys.

## Mitgelieferte Dateien

- `testen.py`: Gesichtssuche, Ausrichtung und ONNX-Ausführung.
- `modelle/gesichtsmodell.onnx`: SFace einschließlich trainierter SVM, Referenzmerkmale, Personennamen und Zurückweisung unsicherer Treffer.
- `modelle/face_detection_yunet_2023mar.onnx`: Gesichtssuche für ganze Fotos.
- `modelle/pruefsummen.json`: SHA-256-Prüfsummen beider Modelldateien.
- `requirements.txt`: festgelegte Paketversionen.
- `LIZENZEN` und `MODELLHINWEISE.txt`: Herkunft und Lizenztexte der Basismodelle.

Die ONNX-Dateien liegen vollständig im Repository. Git LFS und ein separater Modelldownload sind nicht erforderlich. Das Personenmodell enthält persönliche Namen und Gesichtsreferenzen; deshalb ist dieses Repository privat.

## Direkt mit ONNX Runtime laden

```python
import onnxruntime as ort

modell = ort.InferenceSession(
    "modelle/gesichtsmodell.onnx",
    providers=["CPUExecutionProvider"],
)
namen = modell.run(["name"], {"gesicht": bildtensor})[0]
```

`bildtensor` muss ein bereits ausgerichtetes RGB-Gesicht enthalten: `float32`, Form `[1, 3, 112, 112]`, Pixelwerte 0 bis 255. Für ganze Fotos übernimmt `testen.py` die nötige Suche und Ausrichtung.

Der Selbsttest prüft Modelldateien und technische Ausführung. Er ist keine Messung der Erkennungsgenauigkeit. Falsche ursprüngliche Personenbestätigungen können sich im Modell widerspiegeln.

## Neue Trainingsstände

Der Export aus digiKam bleibt im lokalen Projekt des Besitzers. Zum Bereitstellen eines neuen Stands müssen die beiden ONNX-Dateien und ihre Prüfsummen gemeinsam aktualisiert werden. Das Testskript selbst benötigt keinen Zugriff auf die Trainingsdatenbank.

## Lizenzen und Quellen

SFace basiert auf dem Modell aus [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface), dort unter Apache 2.0. YuNet stammt ebenfalls aus [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) und steht dort unter MIT. Die jeweiligen Lizenztexte sind unverändert beigefügt. Einzelheiten und Änderungen am Personenmodell stehen in `MODELLHINWEISE.txt`.
