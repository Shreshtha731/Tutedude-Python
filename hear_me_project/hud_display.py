import cv2
import mediapipe as mp


class HUDDisplay:

  def __init__(self):
    self.mp_drawing = mp.solutions.drawing_utils
    self.mp_hands = mp.solutions.hands

    # Visual styling
    self.landmark_style = self.mp_drawing.DrawingSpec(
        color=(0, 255, 0), thickness=2, circle_radius=3
    )
    self.connection_style = self.mp_drawing.DrawingSpec(
        color=(0, 215, 255), thickness=2
    )

  def draw_landmarks(self, frame, results):
    """Overlays dual-hand joints and bones on the live video frame[cite: 17, 18]."""
    if results and results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        self.mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.landmark_style,
            self.connection_style,
        )

  def draw_hud(self, frame, gesture, confidence, sentence, fps):
    """Renders top telemetry bar, confidence gauge, and bottom subtitle banner[cite: 15, 17]."""
    h, w, _ = frame.shape

    # 1. Top Status Banner[cite: 17]
    cv2.rectangle(frame, (0, 0), (w, 48), (20, 20, 20), -1)
    status_text = f"FPS: {fps:.1f} | STATUS: ACTIVE"
    cv2.putText(
        frame,
        status_text,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 150),
        2,
    )

    # 2. Real-Time Detection Card (Top Right)
    display_gesture = gesture.upper().replace("_", " ") if gesture else "..."
    cv2.putText(
        frame,
        f"SIGN: {display_gesture}",
        (w - 280, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    # 3. Dynamic Confidence Meter[cite: 15, 17]
    bar_width = int(confidence * 160)
    bar_color = (0, 255, 0) if confidence > 0.80 else (0, 165, 255)
    cv2.rectangle(frame, (w - 280, 38), (w - 120, 44), (70, 70, 70), -1)
    cv2.rectangle(
        frame, (w - 280, 38), (w - 280 + bar_width, 44), bar_color, -1
    )

    # 4. Bottom Subtitle Ribbon[cite: 15, 17]
    cv2.rectangle(frame, (0, h - 60), (w, h), (15, 15, 15), -1)
    cv2.putText(
        frame,
        "SUBTITLES:",
        (15, h - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 215, 255),
        2,
    )
    cv2.putText(
        frame,
        sentence.upper(),
        (130, h - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    return frame