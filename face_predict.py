import json
import os
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_SIZE = 224
_candidates = [os.path.join(HERE, "models"), HERE]
MODEL_DIR = next((p for p in _candidates
                  if os.path.exists(os.path.join(p, "ff_xception_final.keras"))), _candidates[0])


@lru_cache(maxsize=1)
def _load():
    from tensorflow import keras
    model = keras.models.load_model(os.path.join(MODEL_DIR, "ff_xception_final.keras"))
    with open(os.path.join(MODEL_DIR, "ff_meta.json")) as f:
        threshold = json.load(f)["threshold"]
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return model, threshold, cascade


def predict_face(path):
    """Returns (label, score). score: high = DEEPFAKE, 0.5 = the trained cutoff.
    Returns (None, None) if no face is found."""
    model, t, cascade = _load()
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    rgb = np.array(img)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) == 0:
        return None, None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    m = int(0.2 * max(w, h))
    x1, y1 = max(x - m, 0), max(y - m, 0)
    x2, y2 = min(x + w + m, rgb.shape[1]), min(y + h + m, rgb.shape[0])
    crop = cv2.resize(rgb[y1:y2, x1:x2], (IMG_SIZE, IMG_SIZE))
    p = float(model.predict(np.expand_dims(crop.astype("float32") / 255.0, 0), verbose=0)[0][0])
    # Rescale so 0.5 equals the trained cutoff (threshold) and the score fits the 0-1 range
    score = 0.5 * p / t if p < t else 0.5 + 0.5 * (p - t) / (1 - t)
    return ("DEEPFAKE" if score >= 0.5 else "REAL"), score
