import time
from functools import lru_cache
from . import config
from .common import Measurement, existing_file


@lru_cache(maxsize=1)
def load_model():
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    from tensorflow.keras.applications import MobileNetV2

    return MobileNetV2(weights="imagenet", include_top=True, alpha=1.0, input_shape=(224, 224, 3))


def classify(path, top_k=5):
    existing_file(path)
    if not 1 <= top_k <= 1000:
        raise ValueError("top_k must be between 1 and 1000.")
    with Measurement() as measurement:
        import numpy as np
        from PIL import Image, ImageOps
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

        start = time.perf_counter()
        model = load_model()
        load_seconds = time.perf_counter() - start
        with Image.open(path) as source:
            rgb = ImageOps.exif_transpose(source).convert("RGB").resize((224, 224))
            pixels = np.asarray(rgb, dtype=np.float32)[None, ...]
        start = time.perf_counter()
        probabilities = model(preprocess_input(pixels), training=False).numpy()
        inference_seconds = time.perf_counter() - start
        predictions = [
            {"class_id": cls, "label": label, "score": round(float(score), 6)}
            for cls, label, score in decode_predictions(probabilities, top=top_k)[0]
        ]
    return {
        "task": "image_classification",
        "model": "MobileNetV2 ImageNet alpha=1.0",
        "framework": "TensorFlow/Keras",
        "input_file": str(path),
        "predictions": predictions,
        "load_seconds": round(load_seconds, 4),
        "inference_seconds": round(inference_seconds, 4),
        **measurement.result(),
    }
