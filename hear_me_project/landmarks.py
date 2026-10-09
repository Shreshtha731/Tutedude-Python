import cv2
import mediapipe as mp
import numpy as np
from config import FEATURES_PER_FRAME


class LandmarkExtractor:

  def __init__(self):
    self.mp_hands = mp.solutions.hands
    self.hands = self.mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,  #[cite: 23]
        min_detection_confidence=0.5,  #[cite: 23]
        min_tracking_confidence=0.5,  #[cite: 23]
    )

  def extract_keypoints(self, frame):
    """Processes an RGB frame and returns:

    - keypoints: np.ndarray of shape (126,)[cite: 14, 16]
    - results: MediaPipe multi_hand_landmarks object for HUD rendering [cite:
    23]
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  #[cite: 1, 23]
    results = self.hands.process(frame_rgb)  #[cite: 23]

    # Return exactly 126 zeros when hands are not detected[cite: 14, 23]
    if not results.multi_hand_landmarks:
      return np.zeros(FEATURES_PER_FRAME, dtype=np.float32), results

    keypoints = []
    # Process up to 2 detected hands[cite: 23]
    for hand_landmarks in results.multi_hand_landmarks[:2]:
      pts = np.array(
          [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
          dtype=np.float32,
      )  #[cite: 7, 23]

      # 1. Wrist-origin translation centering (P_i - P_0)[cite: 1, 16, 23]
      wrist = pts[0]  #[cite: 23]
      centered = pts - wrist  #[cite: 23]

      # 2. Maximum Euclidean span scale normalization[cite: 1, 16, 23]
      max_norm = np.max(np.linalg.norm(centered, axis=1))  #[cite: 23]
      normalized = (
          centered / max_norm if max_norm > 1e-6 else centered
      )  #[cite: 23]
      keypoints.extend(normalized.flatten())  #[cite: 23]

    # Zero-pad if only one hand is visible to enforce strict 126-D vector[cite: 14, 16, 23]
    if len(keypoints) < FEATURES_PER_FRAME:
      keypoints.extend([0.0] * (FEATURES_PER_FRAME - len(keypoints)))  #[cite: 23]

    return np.array(keypoints[:FEATURES_PER_FRAME], dtype=np.float32), results  #[cite: 23]

  def close(self):
    self.hands.close()