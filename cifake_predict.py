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

    # IMPORTANT:
    # The CIFAKE model already contains its own Rescaling layer.
    # Therefore, do NOT divide the image by 255 here.
    image_array = np.array(image).astype("float32")
    image_array = np.expand_dims(image_array, axis=0)

    score = float(
        model.predict(image_array, verbose=0)[0][0]
    )

    # CIFAKE model mapping used in our trained model:
    # high score -> REAL
    # low score  -> AI-GENERATED
    if score >= 0.5:
        result = "REAL"
    else:
        result = "AI-GENERATED"

    return result, score