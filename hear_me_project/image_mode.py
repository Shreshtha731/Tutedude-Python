import cv2
import numpy as np
import pyttsx3
from landmarks import LandmarkExtractor

# Single-frame heuristic classifier for static poses
def classify_static_pose(keypoints):
    """
    Evaluates geometric properties from 63-element keypoints (21 landmarks x 3).
    Returns gesture label and confidence.
    """
    if np.all(keypoints == 0):
        return "No Hand Detected", 0.0

    # Reshape back to 21 x 3
    pts = keypoints.reshape((21, 3))
    
    # Thumb tip (4), Index tip (8), Middle tip (12), Ring tip (16), Pinky tip (20)
    # Wrist is at (0, 0, 0)
    index_open = pts[8, 1] < pts[6, 1]
    middle_open = pts[12, 1] < pts[10, 1]
    ring_open = pts[16, 1] < pts[14, 1]
    pinky_open = pts[20, 1] < pts[18, 1]
    
    if index_open and middle_open and ring_open and pinky_open:
        return "HELLO / OPEN PALM", 0.95
    elif index_open and not middle_open and not ring_open and not pinky_open:
        return "POINT / ONE", 0.92
    elif not index_open and not middle_open and not ring_open and not pinky_open:
        return "FIST / YES", 0.90
    else:
        return "CUSTOM GESTURE", 0.85

def process_image(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Could not load image from {image_path}")
        return

    extractor = LandmarkExtractor()
    results = extractor.process_frame(frame)
    annotated = extractor.draw_landmarks(frame, results)
    keypoints = extractor.extract_normalized_keypoints(results)

    label, confidence = classify_static_pose(keypoints)
    print(f"[PREDICTION] Detected: {label} (Confidence: {confidence*100:.1f}%)")

    # Vocalize result
    engine = pyttsx3.init()
    engine.say(label)
    engine.runAndWait()

    # Display result
    cv2.putText(annotated, f"{label} ({confidence*100:.0f}%)", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Static Snapshot Result", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Test on a snapshot or image file
    process_image("sample_sign.jpg")