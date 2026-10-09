import time
from config import CAMERA_INDEX, CAPTURE_HEIGHT, CAPTURE_WIDTH
import cv2
from hud_display import HUDDisplay
from landmarks import LandmarkExtractor
from recognition_interface import SignRecognizer
from stabilizer import PredictionStabilizer
from tts_module import TextToSpeechEngine


def main():
  # 1. Initialize Modular Subsystems[cite: 1, 16]
  extractor = LandmarkExtractor()
  recognizer = SignRecognizer()
  stabilizer = PredictionStabilizer()
  hud = HUDDisplay()
  tts = TextToSpeechEngine(target_lang="en")

  # 2. Hardware Video Capture (Forced V4L2 & MJPG for WSL2 / Linux)[cite: 1]
  cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
  cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

  if not cap.isOpened():
    print(
        f"[!] Error: Could not initialize camera at index {CAMERA_INDEX} via"
        " V4L2."
    )
    return

  prev_time = time.time()
  current_gesture = ""
  current_confidence = 0.0

  print("\n" + "=" * 65)
  print("   THE HEAR ME PROJECT: MULTILINGUAL TRANSLATION SYSTEM")
  print("   - Hold gesture steady in webcam view for recognition")
  print("   - Press 't' to toggle audio translation (English <-> Hindi)")
  print("   - Press 'c' to clear subtitle sentence history")
  print("   - Press 'q' to terminate application")
  print("=" * 65 + "\n")

  try:
    while cap.isOpened():
      ret, frame = cap.read()
      if not ret:
        print("[!] Frame read failed. Check camera connection.")
        break

      # Mirror frame horizontally for natural user interaction
      frame = cv2.flip(frame, 1)

      # Step 1: Invariant Keypoint Extraction (Dual Hands -> 126-D)[cite: 1, 14, 16]
      keypoints, results = extractor.extract_keypoints(frame)

      # Step 2: Temporal Bi-LSTM Sequence Inference[cite: 1, 14, 16]
      pred_gesture, conf = recognizer.predict(keypoints)
      if pred_gesture is not None:
        current_gesture = pred_gesture
        current_confidence = conf
      else:
        # Smoothly decay confidence bar when below threshold
        current_confidence = (
            conf if conf > 0.0 else max(0.0, current_confidence * 0.9)
        )

      # Step 3: Majority Vote Consensus Stabilization[cite: 1, 17]
      confirmed_token = stabilizer.update(current_gesture, current_confidence)
      if confirmed_token:
        print(
            f"[!] Confirmed Gesture: {confirmed_token.upper()} | Audio Lang:"
            f" {tts.target_lang.upper()}"
        )
        tts.speak(
            confirmed_token
        )  # Non-blocking audio dispatch to daemon thread[cite: 1, 17]

      # Step 4: Frame Rate Telemetry Calculation
      current_time = time.time()
      time_diff = current_time - prev_time
      fps = 1.0 / time_diff if time_diff > 0 else 30.0
      prev_time = current_time

      # Step 5: Visual HUD & Skeletal Connection Rendering[cite: 15, 17]
      hud.draw_landmarks(frame, results)
      frame = hud.draw_hud(
          frame,
          current_gesture,
          current_confidence,
          stabilizer.get_sentence(),
          fps,
      )

      # Step 6: Render Translation Language Badge on Screen
      lang_label = f"AUDIO: {tts.target_lang.upper()}"
      cv2.rectangle(frame, (150, 8), (260, 36), (40, 40, 40), -1)
      cv2.putText(
          frame,
          lang_label,
          (158, 28),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.55,
          (0, 215, 255),
          2,
      )

      cv2.imshow("The Hear Me Project - Live Stream", frame)

      # Interactive Keystroke Controls
      key = cv2.waitKey(1) & 0xFF
      if key == ord("q"):
        break
      elif key == ord("c"):
        stabilizer.clear_sentence()
        print("[*] Subtitle history cleared.")
      elif key == ord("t"):
        # Toggle translation target between English and Hindi
        new_lang = "hi" if tts.target_lang == "en" else "en"
        tts.set_language(new_lang)

  finally:
    # Graceful shutdown of hardware, threads, and UI resources
    cap.release()
    cv2.destroyAllWindows()
    extractor.close()
    tts.stop()
    print("[*] Translation engine shut down cleanly.")


if __name__ == "__main__":
  main()