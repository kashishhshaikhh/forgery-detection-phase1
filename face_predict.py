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

def download_file(file_id, output_path):
os.makedirs(MODEL_DIR, exist_ok=True)

```
if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
    return

print(f"Downloading {os.path.basename(output_path)}...")

result = gdown.download(
    url=f"https://drive.google.com/uc?id={file_id}",
    output=output_path,
    quiet=False
)

if result is None or not os.path.exists(output_path):
    raise RuntimeError(
        f"Could not download {os.path.basename(output_path)}. "
        "Check Google Drive sharing permissions."
    )
```

@lru_cache(maxsize=1)
def _load():
from tensorflow import keras

```
download_file(MODEL_ID, MODEL_PATH)
download_file(META_ID, META_PATH)

model = keras.models.load_model(MODEL_PATH)

with open(META_PATH, "r") as f:
    threshold = json.load(f)["threshold"]

cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

return model, threshold, cascade
```

def predict_face(path):

```
model, threshold, cascade = _load()

img = ImageOps.exif_transpose(
    Image.open(path)
).convert("RGB")

rgb = np.array(img)

gray = cv2.cvtColor(
    rgb,
    cv2.COLOR_RGB2GRAY
)

faces = cascade.detectMultiScale(
    gray,
    1.1,
    5,
    minSize=(60, 60)
)

if len(faces) == 0:
    return None, None

x, y, w, h = max(
    faces,
    key=lambda f: f[2] * f[3]
)

margin = int(0.2 * max(w, h))

x1 = max(x - margin, 0)
y1 = max(y - margin, 0)

x2 = min(x + w + margin, rgb.shape[1])
y2 = min(y + h + margin, rgb.shape[0])

crop = cv2.resize(
    rgb[y1:y2, x1:x2],
    (IMG_SIZE, IMG_SIZE)
)

prediction = model.predict(
    np.expand_dims(
        crop.astype("float32") / 255.0,
        axis=0
    ),
    verbose=0
)[0][0]

p = float(prediction)

score = (
    0.5 * p / threshold
    if p < threshold
    else 0.5 + 0.5 * (p - threshold) / (1 - threshold)
)

label = "DEEPFAKE" if score >= 0.5 else "REAL"

return label, score
```

