"""
Gesture stabilization, sign buffering, and sentence construction.

Raw per-frame predictions from a recognition engine are noisy: a single
frame's label can flicker due to motion blur, transition poses between
signs, or borderline confidence. This module smooths that stream into
stable, "confirmed" words and assembles them into a sentence.

Algorithm (temporal majority vote with cooldown):
  1. Every frame's (label, confidence) is pushed into a fixed-size buffer.
  2. Frames below MIN_CONFIDENCE are treated as idle noise, not votes.
  3. Once the buffer is full, if one label holds >= CONFIRM_THRESHOLD of
     the buffer AND it differs from the last confirmed sign, it "confirms".
  4. After confirming, a cooldown suppresses re-confirmation for a few
     frames so a held pose doesn't spam the same word repeatedly.
"""
from collections import deque, Counter
import config as cfg


class SignStabilizer:
    def __init__(self,
                 buffer_size=cfg.BUFFER_SIZE,
                 confirm_threshold=cfg.CONFIRM_THRESHOLD,
                 min_confidence=cfg.MIN_CONFIDENCE,
                 cooldown_frames=cfg.COOLDOWN_FRAMES,
                 idle_label=cfg.IDLE_LABEL):
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.confirm_threshold = confirm_threshold
        self.min_confidence = min_confidence
        self.cooldown_frames = cooldown_frames
        self.idle_label = idle_label

        self._cooldown_left = 0
        self.last_confirmed = None
        self.last_confidence = 0.0

    def push(self, label, confidence):
        """
        Feed one frame's raw prediction in. Returns a confirmed label (str)
        if this frame caused a NEW sign to be confirmed, else None.
        """
        self.last_confidence = confidence

        effective_label = label if confidence >= self.min_confidence else self.idle_label
        self.buffer.append(effective_label)

        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            return None

        if len(self.buffer) < self.buffer_size:
            return None

        counts = Counter(self.buffer)
        top_label, top_count = counts.most_common(1)[0]
        agreement = top_count / len(self.buffer)

        if (top_label != self.idle_label
                and agreement >= self.confirm_threshold
                and top_label != self.last_confirmed):
            self.last_confirmed = top_label
            self._cooldown_left = self.cooldown_frames
            self.buffer.clear()
            return top_label

        return None

    def current_vote_strength(self):
        """0-1 measure of how 'settled' the buffer currently is, for UI feedback."""
        if not self.buffer:
            return 0.0
        counts = Counter(self.buffer)
        _, top_count = counts.most_common(1)[0]
        return top_count / len(self.buffer)


class SentenceBuilder:
    """Turns a stream of confirmed sign labels into a readable sentence."""

    def __init__(self, space_gesture=cfg.SPACE_GESTURE,
                 delete_gesture=cfg.DELETE_GESTURE,
                 clear_gesture=cfg.CLEAR_GESTURE,
                 speak_gesture=cfg.SPEAK_GESTURE):
        self.words = []
        self.space_gesture = space_gesture
        self.delete_gesture = delete_gesture
        self.clear_gesture = clear_gesture
        self.speak_gesture = speak_gesture
        self.speak_requested = False

    def add(self, label):
        """Feed one confirmed sign label; handles control gestures + vocabulary words."""
        if label == self.clear_gesture:
            self.words = []
        elif label == self.delete_gesture:
            if self.words:
                self.words.pop()
        elif label == self.space_gesture:
            pass  # reserved for future punctuation logic; words already render space-separated
        elif label == self.speak_gesture:
            self.speak_requested = True
        else:
            self.words.append(label.replace("_", " "))

    def text(self):
        return " ".join(self.words)

    def consume_speak_request(self):
        """Returns True once and resets, so TTS fires exactly once per gesture."""
        if self.speak_requested:
            self.speak_requested = False
            return True
        return False

    def is_empty(self):
        return len(self.words) == 0
