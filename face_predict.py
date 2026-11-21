import json
import os
import urllib.request
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

# YuNet face detector (small ONNX file, downloaded once)
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
YUNET_PATH = os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx")

# A detection must have at least this confidence to count as a face.
# Raise (e.g. 0.95) if non-faces still get through; lower (e.g. 0.85) if real faces are missed.
FACE_CONFIDENCE = 0.90

# Face must be at least this fraction of the image's shorter side.
MIN_FACE_FRACTION = 0.05

# Also try slightly tilted versions of the image (helps with tilted selfies).
ANGLES = [0, -30, 30]

# Longest image side used for detection/cropping (keeps it fast).
MAX_SIDE = 1280

# Prints detections in the Streamlit logs. Set to False once everything works.
DEBUG = True


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


def download_yunet():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if os.path.exists(YUNET_PATH) and os.path.getsize(YUNET_PATH) > 0:
        return
    print("Downloading: YuNet face detector")
    urllib.request.urlretrieve(YUNET_URL, YUNET_PATH)


@lru_cache(maxsize=1)
def _load():
    from tensorflow import keras

    download_file(MODEL_ID, MODEL_PATH)
    download_file(META_ID, META_PATH)
    download_yunet()

    model = keras.models.load_model(MODEL_PATH)

    with open(META_PATH, "r") as f:
        metadata = json.load(f)

    threshold = float(metadata["threshold"])

    detector = cv2.FaceDetectorYN.create(
        YUNET_PATH, "", (320, 320),
        score_threshold=FACE_CONFIDENCE,
        nms_threshold=0.3,
        top_k=50,
    )

    return model, threshold, detector


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


def _find_face(rgb, detector):
    """Returns (image_used, (x, y, w, h)) or (None, None) when no confident face is found."""
    for angle in ANGLES:
        rotated = _rotate(rgb, angle)
        h, w = rotated.shape[:2]
        bgr = cv2.cvtColor(rotated, cv2.COLOR_RGB2BGR)

        detector.setInputSize((w, h))
        _, faces = detector.detect(bgr)

        if DEBUG:
            print("angle", angle, "->", None if faces is None else faces[:, [0, 1, 2, 3, -1]].round(2).tolist())

        if faces is None:
            continue

        # columns: x, y, w, h, 10 landmark values, score
        good = [f for f in faces
                if f[-1] >= FACE_CONFIDENCE
                and min(f[2], f[3]) >= MIN_FACE_FRACTION * min(h, w)]
        if good:
            best = max(good, key=lambda f: f[2] * f[3])
            x, y, bw, bh = [int(v) for v in best[:4]]
            x, y = max(x, 0), max(y, 0)
            if DEBUG:
                print("FACE ACCEPTED:", (x, y, bw, bh), "score", float(best[-1]), "angle", angle)
            return rotated, (x, y, bw, bh)

    return None, None


def predict_face(path):
    """Returns (label, score). score: high = DEEPFAKE, 0.5 = the trained cutoff.
    Returns (None, None) if no face is found."""
    model, threshold, detector = _load()

    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    rgb = np.array(img)

    rgb, box = _find_face(rgb, detector)
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
