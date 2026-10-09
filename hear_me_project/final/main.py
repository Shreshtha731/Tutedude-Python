from collections import deque
import queue
import threading
import cv2
from landmarks import (
    draw_styled_landmarks,
    extract_keypoints,
    mediapipe_detection,
    mp_holistic,
)
import numpy as np
import pyttsx3
from tensorflow.keras.models import load_model

# 1. Load trained Bi-LSTM model and action labels
model = load_model("action_model.h5")
ACTIONS = np.load("actions.npy")

# 2. Setup Non-Blocking TTS Worker
speech_queue = queue.Queue()


def tts_worker():
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    while True:
        text = speech_queue.get()
        if text is None:
            break
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")
        speech_queue.task_done()


threading.Thread(target=tts_worker, daemon=True).start()

# 3. Initialize Camera with WSL V4L2 & MJPG Configuration
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# Inference and Display State Variables
sequence = deque(maxlen=30)
sentence = []
last_action = ""
threshold = 0.80

with mp_holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Unable to fetch frame from camera.")
            break

        # Process frame and extract 126 keypoints
        image, results = mediapipe_detection(frame, holistic)
        draw_styled_landmarks(image, results)
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)

        # Predict only when the 30-frame temporal window is filled
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            predicted_idx = np.argmax(res)
            confidence = res[predicted_idx]

            if confidence > threshold:
                current_action = ACTIONS[predicted_idx]

                # Debounce: only trigger TTS and append if sign has changed
                if current_action != last_action:
                    last_action = current_action
                    sentence.append(current_action)
                    speech_queue.put(current_action)

                    # Keep only the last 5 recognized signs on screen
                    if len(sentence) > 5:
                        sentence = sentence[-5:]

        # Subtitle Top Bar
        cv2.rectangle(image, (0, 0), (640, 45), (245, 117, 16), -1)
        subtitle_text = " ".join(sentence)
        cv2.putText(
            image,
            f"Translation: {subtitle_text}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Audio-Visual Sign Translator", image)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

# Cleanup
cap.release()
cv2.destroyAllWindows()
speech_queue.put(None)