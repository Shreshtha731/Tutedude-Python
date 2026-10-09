from collections import deque
import json
import os
from config import (
    ACTIONS,
    ACTIONS_PATH,
    FEATURES_PER_FRAME,
    MODEL_PATH,
    PREDICTION_CONFIDENCE_THRESHOLD,
    SEQUENCE_LENGTH,
)
import numpy as np
import tensorflow as tf


class SignRecognizer:

  def __init__(self):
    # Load Action Classes
    if os.path.exists(ACTIONS_PATH):
      with open(ACTIONS_PATH, "r") as f:
        self.actions = np.array(json.load(f))
    else:
      self.actions = np.array(ACTIONS)

    # Load Deep Model[cite: 16]
    if os.path.exists(MODEL_PATH):
      self.model = tf.keras.models.load_model(MODEL_PATH)
      print(
          f"[+] Loaded model from {MODEL_PATH} with {len(self.actions)} classes."
      )
    else:
      self.model = None
      print(f"[!] Warning: {MODEL_PATH} not found. Run train_model.py first.")

    # 30-Frame Rolling Sequence Window[cite: 1, 14, 16]
    self.sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)

  def predict(self, keypoints_126d):
    """Buffers incoming 126-D landmark vectors and executes inference when the buffer is full[cite: 1, 14].

    Returns: (predicted_class_name or None, confidence_float)
    """
    self.sequence_buffer.append(keypoints_126d)

    # Require full 30-frame sequence before evaluating[cite: 1, 14]
    if len(self.sequence_buffer) < SEQUENCE_LENGTH or self.model is None:
      return None, 0.0

    # Tensor shape: (1, 30, 126)[cite: 14, 16, 18]
    input_tensor = np.expand_dims(np.array(self.sequence_buffer), axis=0)

    predictions = self.model.predict(input_tensor, verbose=0)[0]
    best_idx = int(np.argmax(predictions))
    confidence = float(predictions[best_idx])

    if confidence >= PREDICTION_CONFIDENCE_THRESHOLD:  #[cite: 14]
      return self.actions[best_idx], confidence

    return None, confidence