import os

import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "cifake_model.keras"
)

model = tf.keras.models.load_model(MODEL_PATH)


def predict_cifake(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((32, 32))

    # No division by 255 here: the model has its own Rescaling layer.
    image_array = np.array(image).astype("float32")
    image_array = np.expand_dims(image_array, axis=0)

    score = float(model.predict(image_array, verbose=0)[0][0])

    # Class mapping verified on 500 REAL + 500 FAKE CIFAKE test images:
    #   score near 1 = AI-GENERATED
    #   score near 0 = REAL
    if score >= 0.5:
        result = "AI-GENERATED"
    else:
        result = "REAL"

    return result, score
