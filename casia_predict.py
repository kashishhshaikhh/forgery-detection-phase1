import io
import os

import numpy as np
import tensorflow as tf
from PIL import Image, ImageChops, ImageEnhance

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "casia_model.keras"
)

ELA_QUALITY = 90      # must match training
INPUT_SIZE = (128, 128)
THRESHOLD = 0.5       # score >= THRESHOLD -> TAMPERED
DEBUG = True          # set False to silence the terminal prints

model = tf.keras.models.load_model(MODEL_PATH)


def create_ela_image(image_path, quality=ELA_QUALITY):
    """Same ELA as training, but the re-saved JPEG is kept in memory
    (no shared temp file, so two uploads can't overwrite each other)."""
    original = Image.open(image_path).convert("RGB")

    buffer = io.BytesIO()
    original.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer).convert("RGB")

    ela = ImageChops.difference(original, compressed)

    extrema = ela.getextrema()
    max_difference = max(value[1] for value in extrema)
    if max_difference == 0:
        max_difference = 1

    scale = 255.0 / max_difference
    return ImageEnhance.Brightness(ela).enhance(scale)


def predict_casia(image_path):
    ela_image = create_ela_image(image_path).resize(INPUT_SIZE)

    image_array = np.array(ela_image).astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    score = float(model.predict(image_array, verbose=0)[0][0])
    result = "TAMPERED" if score >= THRESHOLD else "REAL"

    if DEBUG:
        print(
            f"[CASIA] file={os.path.basename(image_path)} "
            f"shape={image_array.shape} min={image_array.min():.3f} "
            f"max={image_array.max():.3f} mean={image_array.mean():.3f} "
            f"score={score:.6f} -> {result}"
        )

    return result, score
