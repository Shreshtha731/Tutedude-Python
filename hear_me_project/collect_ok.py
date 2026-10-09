import os
import time
from config import (
    CAMERA_INDEX,
    CAPTURE_HEIGHT,
    CAPTURE_WIDTH,
    DATA_PATH,
    SEQUENCE_LENGTH,
)
import cv2
from landmarks import LandmarkExtractor
import numpy as np

TARGET_ACTION = "ok"
NUM_SEQUENCES = 12


def main():
  extractor = LandmarkExtractor()

  cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
  cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

  if not cap.isOpened():
    print(f"[!] Error: Could not open camera {CAMERA_INDEX}")
    return

  action_dir = os.path.join(DATA_PATH, TARGET_ACTION)
  for seq in range(NUM_SEQUENCES):
    os.makedirs(os.path.join(action_dir, str(seq)), exist_ok=True)

  print(f"\n[*] Recording ONLY '{TARGET_ACTION.upper()}' (12 takes)...")
  print("    Tip: Form the 'O' with your thumb and index finger facing camera.")
  time.sleep(1.0)

  for seq in range(NUM_SEQUENCES):
    # 3-second positioning countdown
    for countdown in range(3, 0, -1):
      ret, frame = cap.read()
      if not ret:
        break
      frame = cv2.flip(frame, 1)

      cv2.putText(
          frame,
          f"PREPARE: {TARGET_ACTION.upper()}",
          (30, 50),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.9,
          (0, 215, 255),
          2,
      )
      cv2.putText(
          frame,
          f"Take {seq + 1}/{NUM_SEQUENCES} in {countdown}s...",
          (30, 95),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.7,
          (255, 255, 255),
          2,
      )
      cv2.imshow("Record OK Sign", frame)
      if cv2.waitKey(1000) & 0xFF == ord("q"):
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()
        return

    # Record 30 frames
    for frame_idx in range(SEQUENCE_LENGTH):
      ret, frame = cap.read()
      if not ret:
        break
      frame = cv2.flip(frame, 1)

      keypoints, _ = extractor.extract_keypoints(frame)
      npy_path = os.path.join(action_dir, str(seq), f"{frame_idx}.npy")
      np.save(npy_path, keypoints)

      cv2.putText(
          frame,
          f"RECORDING: {TARGET_ACTION.upper()} ({frame_idx + 1}/{SEQUENCE_LENGTH})",
          (30, 50),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.85,
          (0, 0, 255),
          2,
      )
      progress_w = int(
          ((frame_idx + 1) / SEQUENCE_LENGTH) * (CAPTURE_WIDTH - 60)
      )
      cv2.rectangle(frame, (30, 70), (30 + progress_w, 85), (0, 0, 255), -1)

      cv2.imshow("Record OK Sign", frame)
      if cv2.waitKey(20) & 0xFF == ord("q"):
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()
        return

  cap.release()
  cv2.destroyAllWindows()
  extractor.close()
  print(f"\n[✓] Recording finished for '{TARGET_ACTION}'.")


if __name__ == "__main__":
  main()