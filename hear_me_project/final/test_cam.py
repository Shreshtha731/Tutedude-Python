import cv2

for idx in [0, 2]:  # Testing RGB stream nodes
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)

    if cap.isOpened():
        # Set MJPEG compression (Critical for WSL)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        ret, frame = cap.read()
        if ret and frame is not None:
            print(
                f"[SUCCESS] Camera works on index {idx} with MJPG! Resolution: {frame.shape[1]}x{frame.shape[0]}"
            )
            cap.release()
            break
        else:
            print(f"[FAILED] Index {idx} timed out on MJPG read.")
        cap.release()
    else:
        print(f"[FAILED] Index {idx} could not be opened.")