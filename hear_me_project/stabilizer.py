from collections import Counter, deque
from config import COOLDOWN_FRAMES, CONSENSUS_THRESHOLD, STABILIZER_BUFFER_SIZE


class PredictionStabilizer:

  def __init__(
      self,
      buffer_size=STABILIZER_BUFFER_SIZE,
      consensus_threshold=CONSENSUS_THRESHOLD,
      cooldown=COOLDOWN_FRAMES,
  ):
    self.buffer_size = buffer_size  # 15 frames[cite: 1]
    self.consensus_threshold = (
        consensus_threshold  # 75% majority agreement[cite: 1]
    )
    self.cooldown_period = cooldown  # 20 frames cooldown[cite: 1]

    self.prediction_buffer = deque(maxlen=self.buffer_size)
    self.cooldown_counter = 0
    self.last_confirmed_gesture = None
    self.sentence_history = []

  def update(self, predicted_gesture, confidence):
    """Feeds current frame prediction and returns (confirmed_token_to_speak or None)."""
    # Track cooldown
    if self.cooldown_counter > 0:
      self.cooldown_counter -= 1

    self.prediction_buffer.append(predicted_gesture)

    # Need full buffer for consensus check
    if len(self.prediction_buffer) < self.buffer_size:
      return None

    counts = Counter(self.prediction_buffer)
    most_common_gesture, frequency = counts.most_common(1)[0]
    agreement_ratio = frequency / self.buffer_size

    # Gate: Must meet 75% consensus, cooldown must have expired, and must not repeat current word immediately[cite: 1]
    if (
        agreement_ratio >= self.consensus_threshold
        and self.cooldown_counter == 0
        and most_common_gesture != self.last_confirmed_gesture
    ):

      self.last_confirmed_gesture = most_common_gesture
      self.cooldown_counter = self.cooldown_period
      self.sentence_history.append(most_common_gesture)

      # Keep last 6 words in displayed sentence
      if len(self.sentence_history) > 6:
        self.sentence_history.pop(0)

      return most_common_gesture

    return None

  def get_sentence(self):
    return (
        " ".join(self.sentence_history)
        if self.sentence_history
        else "Awaiting gestures..."
    )

  def clear_sentence(self):
    self.sentence_history.clear()
    self.last_confirmed_gesture = None
    self.prediction_buffer.clear()
    self.cooldown_counter = 0   