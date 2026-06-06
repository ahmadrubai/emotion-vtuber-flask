import os
import threading
import traceback

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

from config import (
    MODEL_INPUT_SIZE,
    MODEL_PREPROCESSING_MODE,
    MODEL_USE_TM_PREPROCESSING,
)

from src.utils import load_class_names


class EmotionModel:
    def __init__(self, model_path=None):

        if model_path:
            self.model_path = str(model_path)
        else:
            self.model_path = "models/emotion_mobilenetv2.keras"

        self.class_names = load_class_names()

        self.model = None
        self.is_loaded = False

        self.lock = threading.RLock()

        self.load_model()

    def load_model(self):
        with self.lock:

            try:
                print("\n" + "=" * 60)
                print("LOADING MODEL")
                print("=" * 60)

                print("TensorFlow Version :", tf.__version__)
                print("Model Path         :", self.model_path)
                print("File Exists        :", os.path.exists(self.model_path))

                if not os.path.exists(self.model_path):
                    raise FileNotFoundError(
                        f"Model tidak ditemukan: {self.model_path}"
                    )

                self.class_names = load_class_names()

                self.model = tf.keras.models.load_model(
                    self.model_path,
                    safe_mode=False
                )

                self.is_loaded = True

                print("\n[SUCCESS] Model berhasil dimuat")
                print("Input Shape :", self.model.input_shape)
                print("Output Shape:", self.model.output_shape)
                print("Classes     :", self.class_names)

                print("=" * 60)

            except Exception:

                self.model = None
                self.is_loaded = False

                print("\n")
                print("=" * 60)
                print("[ERROR] GAGAL LOAD MODEL")
                print("=" * 60)

                traceback.print_exc()

                print("=" * 60)
                print("\n")

    def reload(self, model_path=None):

        if model_path is not None:
            self.model_path = str(model_path)

        self.load_model()

        if not self.is_loaded:
            raise RuntimeError("Model gagal diload.")

    def get_output_count(self):

        if self.model is None:
            return 0

        output_shape = self.model.output_shape

        if isinstance(output_shape, list):
            output_shape = output_shape[0]

        return int(output_shape[-1])

    def predict(self, image_bgr):

        with self.lock:

            if not self.is_loaded:
                return "neutral", 0.0, {}

            try:

                image_rgb = image_bgr[:, :, ::-1]

                if MODEL_USE_TM_PREPROCESSING:

                    image = Image.fromarray(
                        image_rgb.astype(np.uint8),
                        mode="RGB"
                    )

                    image = ImageOps.fit(
                        image,
                        (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
                        Image.Resampling.LANCZOS
                    )

                    image_array = np.asarray(image).astype(np.float32)

                    image_array = (image_array / 127.5) - 1

                else:

                    image_array = tf.image.resize(
                        image_rgb,
                        (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
                    ).numpy()

                    image_array = image_array.astype(np.float32)

                    if MODEL_PREPROCESSING_MODE == "rescale_0_1":
                        image_array /= 255.0

                batch = np.expand_dims(
                    image_array,
                    axis=0
                )

                predictions = self.model.predict(
                    batch,
                    verbose=0
                )[0]

                print("Predictions:", predictions)

                class_index = int(np.argmax(predictions))
                confidence = float(predictions[class_index])

                if class_index >= len(self.class_names):
                    return "unknown", confidence, {}

                emotion = self.class_names[class_index]

                probabilities = {
                    self.class_names[i]: float(predictions[i])
                    for i in range(
                        min(
                            len(predictions),
                            len(self.class_names)
                        )
                    )
                }

                return (
                    emotion,
                    confidence,
                    probabilities
                )

            except Exception:

                print("\n[PREDICT ERROR]")
                traceback.print_exc()

                return "neutral", 0.0, {}