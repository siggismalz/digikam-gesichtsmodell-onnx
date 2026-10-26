"""Erkennt die angelernten Personen auf einem Bild oder in einem Ordner."""
from pathlib import Path
import sys

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

PROJEKTORDNER = Path(__file__).resolve().parent
MODELLORDNER = PROJEKTORDNER / "modelle"



def gesicht_ausrichten(farbbild, gesicht):
    """Richtet fünf Gesichtslandmarken mit einer Ähnlichkeitstransformation aus."""
    quellpunkte = np.asarray(gesicht[4:14], dtype=np.float32).reshape(5, 2)
    zielpunkte = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                          [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
    quellmitte = quellpunkte.mean(axis=0)
    zielmitte = np.array([56.0262, 71.9008], dtype=np.float32)
    quellabstand, zielabstand = quellpunkte - quellmitte, zielpunkte - zielmitte
    # Float32-Produkte mit Float64-Summen entsprechen der OpenCV-Vorverarbeitung.
    kreuzprodukte = zielabstand[:, :, None] * quellabstand[:, None, :]
    kovarianz = kreuzprodukte.sum(axis=0, dtype=np.float64) / 5
    singulaerwerte, links, rechts = cv2.SVDecomp(kovarianz)
    vorzeichen = np.array([1.0, -1.0 if np.linalg.det(kovarianz) < 0 else 1.0])
    drehung = links @ np.diag(vorzeichen) @ rechts
    streuung = np.square(quellabstand).sum(dtype=np.float64) / 5
    if streuung < 1e-12:
        raise ValueError("Die Gesichtslandmarken erlauben keine Ausrichtung.")
    faktor = float((singulaerwerte.ravel() * vorzeichen).sum() / streuung)
    verschiebung = zielmitte - faktor * (drehung @ quellmitte)
    transformation = np.column_stack((faktor * drehung, verschiebung))
    return cv2.warpAffine(farbbild, transformation, (112, 112), flags=cv2.INTER_LINEAR)


def gesichter_suchen(gesichtssuche, farbbild, schwelle):
    gesichtssuche.setInputSize((farbbild.shape[1], farbbild.shape[0]))
    gesichtssuche.setScoreThreshold(schwelle)
    _, gesichter = gesichtssuche.detect(farbbild)
    return [] if gesichter is None else gesichter


def namen_erkennen(bildpfad, sitzung, gesichtssuche):
    """Sucht Gesichter im Foto und führt die gespeicherte ONNX-Personenerkennung aus."""
    with Image.open(bildpfad) as bild:
        farbbild = cv2.cvtColor(np.asarray(ImageOps.exif_transpose(bild).convert("RGB")), cv2.COLOR_RGB2BGR)
    hoehe, breite = farbbild.shape[:2]
    faktor = min(1.0, 1600 / max(hoehe, breite))
    verkleinert = cv2.resize(farbbild, (max(1, round(breite * faktor)), max(1, round(hoehe * faktor))))
    namen = []
    for gesicht in gesichter_suchen(gesichtssuche, verkleinert, 0.8):
        linkswert, obenwert, gesichtsbreite, gesichtshoehe = gesicht[:4] / faktor
        links, oben = max(0, int(linkswert)), max(0, int(obenwert))
        rechts, unten = min(breite, int(np.ceil(linkswert + gesichtsbreite))), min(hoehe, int(np.ceil(obenwert + gesichtshoehe)))
        if rechts <= links or unten <= oben:
            continue
        ausschnitt = cv2.copyMakeBorder(farbbild[oben:unten, links:rechts], 60, 60, 60, 60,
                                       cv2.BORDER_CONSTANT, value=(0, 0, 0))
        kandidaten = gesichter_suchen(gesichtssuche, ausschnitt, 0.3)
        if len(kandidaten) == 0:
            namen.append("Unbekannt")
            continue
        mitte = np.array([ausschnitt.shape[1] / 2, ausschnitt.shape[0] / 2])
        kandidat = min(kandidaten, key=lambda treffer: float(np.linalg.norm(treffer[:2] + treffer[2:4] / 2 - mitte)))
        ausgerichtet = gesicht_ausrichten(ausschnitt, kandidat)
        eingabe = cv2.dnn.blobFromImage(ausgerichtet, swapRB=True)
        name = sitzung.run(["name"], {"gesicht": eingabe})[0][0]
        namen.append(str(name))
    return namen


if __name__ == "__main__":
    pfad = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJEKTORDNER / "testbilder"
    bilder = sorted(bild for bild in pfad.iterdir() if bild.suffix.lower() in (".jpg", ".jpeg", ".png")) if pfad.is_dir() else [pfad]
    modell = ort.InferenceSession(str(MODELLORDNER / "gesichtsmodell.onnx"), providers=["CPUExecutionProvider"])
    suche = cv2.FaceDetectorYN.create(str(MODELLORDNER / "face_detection_yunet_2023mar.onnx"), "", (320, 320), 0.8, 0.3, 5000)
    if not bilder:
        print("Lege Bilder in testbilder ab oder starte: python testen.py Pfad/zum/Bild.jpg")
    for bild in bilder:
        namen = namen_erkennen(bild, modell, suche)
        print(f"{bild.name}: {', '.join(namen) or 'Kein Gesicht erkannt'}")
