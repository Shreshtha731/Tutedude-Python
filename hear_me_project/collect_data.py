import os
import time
from config import (
    ACTIONS,
    CAMERA_INDEX,
    CAPTURE_HEIGHT,
    CAPTURE_WIDTH,
    DATA_PATH,
    SEQUENCE_LENGTH,
)
import cv2
from landmarks import LandmarkExtractor
import numpy as np

# Training Samples per Gesture Class
NUM_SEQUENCES_PER_ACTION = 12
FRAMES_PER_SEQUENCE = SEQUENCE_LENGTH  # 30 frames per gesture take[cite: 1, 14]


def main():
  extractor = LandmarkExtractor()

  # Force Linux V4L2 Backend and MJPG Compression for WSL2[cite: 1]
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

  # Create directories for every class and take
  for action in ACTIONS:
    for seq in range(NUM_SEQUENCES_PER_ACTION):
      os.makedirs(os.path.join(DATA_PATH, action, str(seq)), exist_ok=True)

  print("\n" + "=" * 60)
  print("   THE HEAR ME PROJECT: RAPID DATA COLLECTION")
  print(f"   Actions ({len(ACTIONS)}): {ACTIONS}")
  print(f"   Takes per action: {NUM_SEQUENCES_PER_ACTION}")
  print("   Press 'q' at any time to exit.")
  print("=" * 60)

  time.sleep(1.0)

  for action in ACTIONS:
    for seq in range(NUM_SEQUENCES_PER_ACTION):

      # Stage 1: Countdown Pause (Position Hands)
      for countdown in range(3, 0, -1):
        ret, frame = cap.read()
        if not ret:
          print("[!] Failed to read camera frame.")
          break

        frame = cv2.flip(frame, 1)

        cv2.putText(
            frame,
            f"READY: {action.upper()}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 215, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Take {seq + 1}/{NUM_SEQUENCES_PER_ACTION} starting in"
            f" {countdown}s...",
            (30, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Data Collector", frame)
        if cv2.waitKey(1000) & 0xFF == ord("q"):
          cap.release()
          cv2.destroyAllWindows()
          extractor.close()
          return

      # Stage 2: Capture Exactly 30 Normalized Feature Frames[cite: 1, 14]
      for frame_idx in range(FRAMES_PER_SEQUENCE):
        ret, frame = cap.read()
        if not ret:
          break

        frame = cv2.flip(frame, 1)

        # Extract centered and scaled 126-D keypoint array[cite: 1, 14, 16]
        keypoints, results = extractor.extract_keypoints(frame)

        # Save coordinate vector
        npy_path = os.path.join(
            DATA_PATH, action, str(seq), f"{frame_idx}.npy"
        )
        np.save(npy_path, keypoints)

        # UI Progress Bar
        cv2.putText(
            frame,
            f"RECORDING: {action.upper()} ({frame_idx + 1}/{FRAMES_PER_SEQUENCE})",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 0, 255),
            2,
        )
        progress_width = int(
            ((frame_idx + 1) / FRAMES_PER_SEQUENCE) * (CAPTURE_WIDTH - 60)
        )
        cv2.rectangle(
            frame, (30, 70), (30 + progress_width, 85), (0, 0, 255), -1
        )

        cv2.imshow("Data Collector", frame)
        if cv2.waitKey(20) & 0xFF == ord("q"):
          cap.release()
          cv2.destroyAllWindows()
          extractor.close()
          return

  cap.release()
  cv2.destroyAllWindows()
  extractor.close()
  print("\n[✓] Data collection complete! Run `python train_model.py` next.")


if __name__ == "__main__":
  main()