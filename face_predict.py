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

# ---------------------------------------------------------------- face detection settings
# One strict setting only. The old loose setting (minNeighbors=3) caused false positives
# on objects like fruit.
DETECT_SETTINGS = [
    dict(scaleFactor=1.1, minNeighbors=6, minSize=(60, 60)),
]
ANGLES = [0, -15, 15, -30, 30]

# A face must be at least this fraction of the image's shorter side.
MIN_FACE_FRACTION = 0.08

# Require two eyes inside the face box. Set to False if real faces start getting missed.
REQUIRE_EYES = True

# Set to True to print what the detector found (shows up in the terminal / logs).
DEBUG = False


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

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )

    return model, threshold, face_cascade, eye_cascade


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


def _looks_like_face(gray, box, eye_cascade):
    """Extra checks to reject false positives (fruit, round objects, textures)."""
    x, y, w, h = box
    img_h, img_w = gray.shape[:2]

    # 1) Face must be a reasonable size compared to the image.
    if min(w, h) < MIN_FACE_FRACTION * min(img_h, img_w):
        return False

    if not REQUIRE_EYES:
        return True

    # 2) Two eyes must be found in the upper part of the face box.
    roi = gray[y:y + int(0.6 * h), x:x + w]
    min_eye = max(w // 12, 10)
    eyes = eye_cascade.detectMultiScale(
        roi, scaleFactor=1.1, minNeighbors=5, minSize=(min_eye, min_eye)
    )
    if len(eyes) < 2:
        return False

    # The two eyes must be clearly apart horizontally (not one eye detected twice).
    centers = sorted(ex + ew / 2 for ex, ey, ew, eh in eyes)
    return (centers[-1] - centers[0]) >= 0.25 * w


def _find_face(rgb, face_cascade, eye_cascade):
    """Returns (image_used, (x, y, w, h)) or (None, None) when no real face is found."""
    for settings in DETECT_SETTINGS:
        for angle in ANGLES:
            rotated = _rotate(rgb, angle)
            gray = cv2.cvtColor(rotated, cv2.COLOR_RGB2GRAY)
            faces = cface = face_cascade.detectMultiScale(gray, **settings)

            if DEBUG:
                print("candidates:", settings, "angle", angle, "->", list(cface))

            # Try the biggest candidates first.
            for f in sorted(faces, key=lambda f: f[2] * f[3], reverse=True):
                if _looks_like_face(gray, f, eye_cascade):
                    if DEBUG:
                        print("FACE ACCEPTED:", f, "angle", angle)
                    return rotated, f
    return None, None


def predict_face(path):
    """Returns (label, score). score: high = DEEPFAKE, 0.5 = the trained cutoff.
    Returns (None, None) if no face is found."""
    model, threshold, face_cascade, eye_cascade = _load()

    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    rgb = np.array(img)

    rgb, box = _find_face(rgb, face_cascade, eye_cascade)
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
