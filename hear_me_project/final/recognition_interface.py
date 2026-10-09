"""
Recognition engine interface.

Whoever owns the model/recognition engine should hand you an object with a
`.predict(frame) -> (label: str, confidence: float)` method. Wrap it to match
`RecognitionEngine` below (or just make sure it has a `predict` method with
this signature) and pass it into main.py / dashboard.py.

This file also ships a MockRecognitionEngine so the UI/output engine can be
built, tested, and demoed standalone before the real model is wired in.
"""
import random
import time
from abc import ABC, abstractmethod


class RecognitionEngine(ABC):
    """Contract that any gesture recognition backend must implement."""

    @abstractmethod
    def predict(self, frame):
        """
        Args:
            frame: a single BGR frame (numpy array) from the camera.
        Returns:
            (label: str, confidence: float in [0, 1])
        """
        raise NotImplementedError


class MockRecognitionEngine(RecognitionEngine):
    """
    Fake engine that cycles through a vocabulary with jittery confidence,
    purely for developing/testing the UI without a trained model attached.
    """

    VOCAB = ["hello", "thank_you", "yes", "no", "please", "sorry",
             "help", "space", "delete", "clear", "speak", "idle"]

    def __init__(self, hold_seconds=1.2):
        self._current = "idle"
        self._last_switch = time.time()
        self._hold_seconds = hold_seconds

    def predict(self, frame):
        now = time.time()
        if now - self._last_switch > self._hold_seconds:
            self._current = random.choice(self.VOCAB)
            self._last_switch = now
            self._hold_seconds = random.uniform(0.8, 2.0)

        if self._current == "idle":
            conf = random.uniform(0.1, 0.4)
        else:
            conf = random.uniform(0.6, 0.98)
        return self._current, conf
