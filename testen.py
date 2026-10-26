"""Erkennt die angelernten Personen mit den mitgelieferten ONNX-Modellen."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

PROJEKTORDNER = Path(__file__).resolve().parent
MODELLORDNER = PROJEKTORDNER / "modelle"



def sitzung_laden(modell):
    """Verwendet die normale CPU-Ausführung von ONNX Runtime."""
    optionen = ort.SessionOptions()
    optionen.intra_op_num_threads = 2
    optionen.inter_op_num_threads = 1
    return ort.InferenceSession(modell, sess_options=optionen, providers=["CPUExecutionProvider"])


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


def modell_laden():
    """Prüft und lädt die mitgelieferten Modelle ohne digiKam oder Kontozugang."""
    dateien = ("gesichtsmodell.onnx", "face_detection_yunet_2023mar.onnx")
    pruefsummen = json.loads((MODELLORDNER / "pruefsummen.json").read_text(encoding="utf-8"))
    if set(pruefsummen) != set(dateien):
        raise ValueError("Die Liste der Modelldateien ist unvollständig.")
    for dateiname in dateien:
        datei = MODELLORDNER / dateiname
        if not datei.is_file():
            raise FileNotFoundError(f"Die Modelldatei fehlt: {datei}")
        if hashlib.sha256(datei.read_bytes()).hexdigest() != pruefsummen[dateiname]:
            raise ValueError(f"Die Modelldatei ist beschädigt oder unvollständig: {dateiname}")
    sitzung = sitzung_laden(str(MODELLORDNER / dateien[0]))
    suche = cv2.FaceDetectorYN.create(str(MODELLORDNER / dateien[1]), "", (320, 320), 0.8, 0.3, 5000)
    return sitzung, suche


def selbsttest(sitzung, suche):
    """Führt beide Netze aus, ohne persönliche Testfotos zu benötigen."""
    if [(eingabe.name, eingabe.shape, eingabe.type) for eingabe in sitzung.get_inputs()] != [
            ("gesicht", [1, 3, 112, 112], "tensor(float)")]:
        raise ValueError("Das Personenmodell hat ein unerwartetes Eingabeformat.")
    merkmale, namen = sitzung.run(["gesichtsmerkmale", "name"],
                                  {"gesicht": np.zeros((1, 3, 112, 112), dtype=np.float32)})
    if merkmale.shape != (1, 128) or not np.isfinite(merkmale).all() or len(namen) != 1:
        raise ValueError("Das Personenmodell liefert ungültige Ergebnisse.")
    if len(gesichter_suchen(suche, np.zeros((320, 320, 3), dtype=np.uint8), 0.8)) != 0:
        raise ValueError("Die Gesichtssuche erkennt im leeren Prüfbild unerwartet ein Gesicht.")
    personen = json.loads(sitzung.get_modelmeta().custom_metadata_map["namen"])
    print(f"Selbsttest bestanden: beide Modelle geladen und ausgeführt, {len(personen)} angelernte Personen.")


def hauptprogramm():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser._positionals.title = "Argumente"
    parser._optionals.title = "Optionen"
    parser.add_argument("-h", "--hilfe", action="help", help="Diese Hilfe anzeigen.")
    parser.add_argument("bildpfad", nargs="?", type=Path, default=PROJEKTORDNER / "testbilder",
                        help="Ein Bild oder ein Ordner mit eigenen Bildern.")
    parser.add_argument("--selbsttest", action="store_true", help="Installation und Modelle ohne Fotos prüfen.")
    argumente = parser.parse_args()
    sitzung, suche = modell_laden()
    if argumente.selbsttest:
        selbsttest(sitzung, suche)
        return 0
    if argumente.bildpfad.is_file():
        bilder = [argumente.bildpfad]
    elif argumente.bildpfad.is_dir():
        bilder = sorted(bild for bild in argumente.bildpfad.iterdir()
                        if bild.is_file() and bild.suffix.lower() in (".jpg", ".jpeg", ".png"))
    else:
        raise FileNotFoundError(f"Der Bildpfad wurde nicht gefunden: {argumente.bildpfad}")
    if not bilder:
        raise FileNotFoundError("Lege eigene JPEG-/PNG-Bilder in testbilder ab oder gib einen Bildpfad an.")
    fehlerzahl = 0
    for bild in bilder:
        try:
            namen = namen_erkennen(bild, sitzung, suche)
            print(f"{bild.name}: {', '.join(namen) or 'Kein Gesicht erkannt'}", flush=True)
        except Exception as fehler:
            print(f"{bild.name}: Verarbeitung fehlgeschlagen: {fehler}", flush=True)
            fehlerzahl += 1
    return 1 if fehlerzahl else 0


if __name__ == "__main__":
    try:
        raise SystemExit(hauptprogramm())
    except Exception as fehler:
        raise SystemExit(f"Test fehlgeschlagen: {fehler}")
