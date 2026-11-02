import json
import os
from functools import lru_cache

import cv2
import gdown
import numpy as np
from PIL import Image, ImageOps

IMG_SIZE = 224

MODEL_ID = "1GCbmEalyX3W9HhUHwzeBQhiZCQ1eaFmp"
META_ID = "1nqJ2qERadMD32uK6Zi43ewpyFDvdldzV"

MODEL_DIR = os.path.join("/tmp", "forgery_models")
MODEL_PATH = os.path.join(MODEL_DIR, "ff_xception_final.keras")
META_PATH = os.path.join(MODEL_DIR, "ff_meta.json")

# Face detection settings: strict first, then looser. Each is tried at several rotations.
DETECT_SETTINGS = [
    dict(scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)),
    dict(scaleFactor=1.05, minNeighbors=3, minSize=(40, 40)),
]
ANGLES = [0, -15, 15, -30, 30, -45, 45]


def download_file(file_id, output_path):
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return

    print("Downloading:", os.path.basename(output_path))

    result = gdown.download(
        url="https://drive.google.com/uc?id=" + file_id,
        output=output_path,
        quiet=False,
    )

    if result is None or not os.path.exists(output_path):
        raise RuntimeError("Google Drive download failed. Check file permissions.")


@lru_cache(maxsize=1)
def _load():
    from tensorflow import keras

    download_file(MODEL_ID, MODEL_PATH)
    download_file(META_ID, META_PATH)

    model = keras.models.load_model(MODEL_PATH)

    with open(META_PATH, "r") as f:
        metadata = json.load(f)

    threshold = float(metadata["threshold"])

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    return model, threshold, cascade


def _rotate(rgb, angle):
    """Rotate the image by angle degrees, enlarging the canvas so nothing is cut off."""
    if angle == 0:
        return rgb
    h, w = rgb.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += new_w / 2 - w / 2
    m[1, 2] += new_h / 2 - h / 2
    return cv2.warpAffine(rgb, m, (new_w, new_h))


def _find_face(rgb, cascade):
    """Try strict then loose settings, at several rotations.
    Returns (image_used, (x, y, w, h)) or (None, None)."""
    for settings in DETECT_SETTINGS:
        for angle in ANGLES:
            rotated = _rotate(rgb, angle)
            gray = cv2.cvtColor(rotated, cv2.COLOR_RGB2GRAY)
            faces = cascade.detectMultiScale(gray, **settings)
            if len(faces) > 0:
                return rotated, max(faces, key=lambda f: f[2] * f[3])
    return None, None


def predict_face(path):
    """Returns (label, score). score: high = DEEPFAKE, 0.5 = the trained cutoff.
    Returns (None, None) if no face is found."""
    model, threshold, cascade = _load()

    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    rgb = np.array(img)

    rgb, box = _find_face(rgb, cascade)
    if box is None:
        return None, None

    x, y, w, h = box
    margin = int(0.2 * max(w, h))
    x1 = max(x - margin, 0)
    y1 = max(y - margin, 0)
    x2 = min(x + w + margin, rgb.shape[1])
    y2 = min(y + h + margin, rgb.shape[0])

    crop = cv2.resize(rgb[y1:y2, x1:x2], (IMG_SIZE, IMG_SIZE))

    prediction = model.predict(
        np.expand_dims(crop.astype("float32") / 255.0, axis=0),
        verbose=0,
    )[0][0]
    p = float(prediction)

    # Rescale so 0.5 equals the trained cutoff (threshold) and the score fits 0-1
    if p < threshold:
        score = 0.5 * p / threshold
    else:
        score = 0.5 + 0.5 * (p - threshold) / (1 - threshold)

    label = "DEEPFAKE" if score >= 0.5 else "REAL"
    return label, score
