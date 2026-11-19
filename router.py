"""Content router: decides which detectors should see an image.

A pretrained MobileNetV2 (ImageNet) looks at WHAT the image shows, not at faces.
  - Confident object / animal / scene  -> "OBJECT"  (CASIA + CIFAKE only)
  - Anything else, or low confidence   -> "UNSURE"  (all three detectors)

"UNSURE" is the safe default: when the router is not sure, no detector is skipped.
"""
from functools import lru_cache

import numpy as np
from PIL import Image, ImageOps

# Router must be at least this confident before it is allowed to skip a detector.
CONF = 0.50

# ImageNet labels that often fire on portraits (clothing, accessories, people).
# If one of these is the top guess, the image is treated as UNSURE (-> all detectors).
PERSON_LIKE = {
    "groom", "scuba_diver", "ballplayer", "jersey", "suit", "sweatshirt", "cardigan",
    "academic_gown", "cloak", "trench_coat", "lab_coat", "military_uniform", "bow_tie",
    "Windsor_tie", "sunglasses", "sunglass", "wig", "lipstick", "face_powder", "mask",
    "ski_mask", "bonnet", "cowboy_hat", "sombrero", "stole", "poncho", "miniskirt",
    "gown", "kimono", "abaya", "hoopskirt", "maillot", "bikini", "swimming_trunks",
    "brassiere", "diaper", "pajama", "fur_coat", "velvet", "bulletproof_vest",
    "neck_brace", "hair_slide", "hair_spray", "shower_cap", "crash_helmet",
    "football_helmet", "oxygen_mask", "gasmask", "mortarboard", "cuirass",
}


@lru_cache(maxsize=1)
def _load():
    from tensorflow.keras.applications import MobileNetV2

    return MobileNetV2(weights="imagenet")


def route_image(path):
    """Returns (route, top_label, confidence). route is "OBJECT" or "UNSURE"."""
    from tensorflow.keras.applications.mobilenet_v2 import decode_predictions, preprocess_input

    model = _load()
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB").resize((224, 224))
    x = preprocess_input(np.expand_dims(np.array(img, dtype="float32"), axis=0))
    preds = model.predict(x, verbose=0)
    _, label, prob = decode_predictions(preds, top=1)[0][0]
    prob = float(prob)

    if prob >= CONF and label not in PERSON_LIKE:
        return "OBJECT", label.replace("_", " "), prob
    return "UNSURE", label.replace("_", " "), prob
