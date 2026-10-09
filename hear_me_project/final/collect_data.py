import os
import sys
import cv2
import numpy as np
from landmarks import (
    draw_styled_landmarks,
    extract_keypoints,
    mediapipe_detection,
    mp_holistic,
)

DATA_PATH = os.path.join("MP_Data")
ACTIONS = np.array(["hello", "thanks", "yes", "no", "i_love_you"])
NO_SEQUENCES = 30
SEQUENCE_LENGTH = 30

# Create dataset folder structure
for action in ACTIONS:
    for sequence in range(NO_SEQUENCES):
        os.makedirs(
            os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True
        )

# Initialize camera with verified WSL V4L2/MJPG parameters
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

with mp_holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
) as holistic:
    for action in ACTIONS:
        for sequence in range(NO_SEQUENCES):
            for frame_num in range(SEQUENCE_LENGTH):
                ret, frame = cap.read()
                if not ret:
                    print(
                        f"Error reading frame at action: {action}, seq: {sequence}, frame: {frame_num}"
                    )
                    continue

                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)

                # Prompt display
                if frame_num == 0:
                    cv2.putText(
                        image,
                        "STARTING COLLECTION",
                        (120, 200),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        4,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        image,
                        f"Action: {action} | Video #{sequence}",
                        (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow("Data Collection", image)
                    # 1.5-second prep pause before each 30-frame recording starts
                    if cv2.waitKey(1500) & 0xFF == ord("q"):
                        cap.release()
                        cv2.destroyAllWindows()
                        sys.exit(0)
                else:
                    cv2.putText(
                        image,
                        f"Action: {action} | Video #{sequence}",
                        (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow("Data Collection", image)

                # Export extracted keypoints (126 features)
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(
                    DATA_PATH, action, str(sequence), f"{frame_num}.npy"
                )
                np.save(npy_path, keypoints)

                if cv2.waitKey(10) & 0xFF == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    sys.exit(0)

cap.release()
cv2.destroyAllWindows()