import os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageChops, ImageEnhance

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "casia_model.keras"
)

model = tf.keras.models.load_model(MODEL_PATH)


def create_ela_image(image_path, quality=90):

    original = Image.open(image_path).convert("RGB")

    temp_path = os.path.join(
        os.path.dirname(__file__),
        "temp_ela.jpg"
    )

    original.save(
        temp_path,
        "JPEG",
        quality=quality
    )

    compressed = Image.open(temp_path).convert("RGB")

    ela = ImageChops.difference(
        original,
        compressed
    )

    extrema = ela.getextrema()

    max_difference = max(
        value[1] for value in extrema
    )

    if max_difference == 0:
        max_difference = 1

    scale = 255.0 / max_difference

    ela = ImageEnhance.Brightness(
        ela
    ).enhance(scale)

    return ela


def predict_casia(image_path):

    ela_image = create_ela_image(image_path)

    ela_image = ela_image.resize(
        (128, 128)
    )

    image_array = np.array(
        ela_image
    ).astype("float32") / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    score = float(
        model.predict(
            image_array,
            verbose=0
        )[0][0]
    )

    # CASIA mapping:
    # high score -> TAMPERED
    # low score  -> REAL
    if score >= 0.5:
        result = "TAMPERED"
    else:
        result = "REAL"

    return result, score